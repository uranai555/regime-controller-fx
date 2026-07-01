# Regime Controller FX

FX/CB回収EA向け **市場レジーム判定 → 戦略許可/禁止** の上位フィルター。

## コンセプト

単なる「勝つEA」ではなく、**相場環境・ブローカー環境・ポジション状態を見て、戦略を動かす/止める上位制御モデル**。

```
市場データ → 特徴量化 → [HMM分類 + ブローカー品質] → レジーム判定 → 戦略ON/OFF → リスク調整
```

**コア思想**: レジーム判定は直接売買シグナルに使わない。あくまで戦略許可/禁止のフィルター。

## フェーズ計画

| Phase | 内容 | 状態 |
|-------|------|------|
| **Phase 1** | ルールベース cb_run_score + persistence filter | ✅ 完了 |
| **Phase 2** | HMM (GaussianHMM) によるレジーム分類 | ✅ 完了 |
| **Phase 3** | ブローカー別 Execution Quality Model | ✅ 完了 |

## アーキテクチャ (Phase 1 + 2 + 3)

```
bars / ticks / positions / execution_stats / existing_layer3_regime / broker_id
  ↓
FeatureCollector
  ↓
RegimeFeatures
  ↓
┌─────────────────┐  ┌──────────────────────┐
│ HMMRegimeClassifier │  │ BrokerQualityModel  │  ← Phase 2/3
│ (GaussianHMM)   │  │ (EWMA profile)       │
└───────┬─────────┘  └──────────┬───────────┘
        ↓                       ↓
    hmm_regime_state      broker_quality_score
        ↓                       ↓
SubScoreCalculator (9サブスコア)
  ↓
cb_run_score (重み付き平均)
  ↓
raw_mode 判定 (cb_run_score + hard block 条件)
  ↓
PersistenceFilter (危険即時・安全3バー確認)
  ↓
RegimeDecision
```

### 出力モード

| モード | cb_run_score | 動作 |
|--------|-------------|------|
| NORMAL | >= 80 | 通常稼働 (risk_multiplier=1.0) |
| CAUTION | 60-80 | 小ロット稼働 (risk_multiplier=0.5) |
| NO_NEW_ENTRY | 40-60 | 新規停止・既存管理のみ |
| REDUCE_ONLY | 20-40 | 決済優先・新規禁止 |
| FORCE_EXIT | < 20 | 緊急撤退候補 |

### 9つのサブスコア

| スコア | 重み | 判定内容 |
|--------|------|----------|
| volatility_score | 0.15 | ATR百分位で「低すぎず高すぎないボラ」 |
| trend_safety_score | 0.10 | 既存Layer3レジームからトレンド強度 |
| spread_score | 0.18 | 現在スプレッド ÷ 平均スプレッド |
| event_safety_score | 0.13 | 指標イベントまでの残り時間 |
| inventory_score | 0.13 | ポジション偏り・含み損速度・証拠金 |
| execution_score | 0.08 | 約定品質（欠損時はデフォルト80） |
| cb_efficiency_score | 0.08 | CB収益 vs 実質コスト |
| **hmm_regime_score** | 0.08 | HMM隠れ状態からの安全度 (Phase 2) |
| **broker_quality_score** | 0.07 | ブローカー別約定品質 (Phase 3) |

### hard block 条件（スコアより優先）

- `spread_ratio > 3.0` → **REDUCE_ONLY**
- `spread_ratio > 2.0` → **NO_NEW_ENTRY**
- `margin_level < 300` → **FORCE_EXIT**
- `floating_pnl_velocity < -30` → **REDUCE_ONLY**
- 指標15分前 → **NO_NEW_ENTRY**

## Phase 2: HMM レジーム分類

GaussianHMM により価格時系列から3つの隠れ状態を推定:

| 隠れ状態 | 意味 | hmm_regime_score |
|----------|------|------------------|
| LOW_VOL | 低ボラ・レンジ | 100 |
| TRENDING | 方向性あり | 55 |
| HIGH_VOL | 高ボラ・クラッシュ | 15 |

**特徴量**: log_returns, realized_vol (5-bar rolling), range_ratio

```python
from regime import RegimeController

rc = RegimeController()

# HMM を学習（200本以上の bars が必要）
rc.fit_hmm(historical_bars)

# 以降の evaluate() で HMM 推定が自動適用される
decision = rc.evaluate(symbol="XAUUSD", bars=recent_bars, ...)
print(decision.features.get("hmm_regime_state"))  # "LOW_VOL" / "TRENDING" / "HIGH_VOL"
```

**フォールバック**: `hmmlearn` 未インストール時は Phase 1 のルールベースのみで動作。

## Phase 3: ブローカー別 Execution Quality Model

ブローカーごとの約定品質を EWMA で学習し、`broker_quality_score` を算出:

| メトリクス | 説明 |
|-----------|------|
| slippage_mean / p95 | 平均・P95スリッページ |
| reject_rate | 注文リジェクト率 |
| fill_rate | フィル率 |
| requote_rate | リクオート率 |
| latency_mean / p95 | 約定レイテンシ |
| spread_markup | ブローカー固有スプレッド上乗せ |
| hourly_quality | 時間帯別品質ファクター |

```python
from regime import RegimeController, ExecutionEvent

rc = RegimeController()

# 約定イベントごとにプロファイル更新
rc.update_broker_profile(ExecutionEvent(
    broker_id="BrokerA",
    slippage=0.3,
    latency_ms=120,
    rejected=False,
    filled=True,
    hour=14,
))

# evaluate 時に broker_id を指定
decision = rc.evaluate(symbol="XAUUSD", bars=bars, broker_id="BrokerA", ...)
print(decision.features.get("broker_quality_score"))  # 0-100
```

## クイックスタート

```python
from regime import RegimeController

rc = RegimeController()

# dictベースでデータを渡す
decision = rc.evaluate(
    symbol="XAUUSD",
    bars=[{"open": 100, "close": 101, "high": 102, "low": 99, "spread_avg": 0.3}],
    positions=[{"direction": "BUY", "volume": 1.0, "is_open": True, "unrealized_pnl_pips": 2.0}],
    execution_stats={"slippage_avg": 0.1, "order_reject_rate": 0.01},
    existing_layer3_regime="RANGING",
    current_time_ms=1624277100000,
    broker_id="BrokerA",  # Phase 3
)

print(decision.mode)             # NORMAL / CAUTION / NO_NEW_ENTRY / ...
print(decision.cb_run_score)     # 0.0-100.0
print(decision.allow_new_entry)  # bool
print(decision.risk_multiplier)  # 0.0-1.0
print(decision.reason_codes)     # ["VOLATILITY_SCORE_DEFAULTED", ...]
```

## 設定

```python
from regime import RegimeController, RegimeControllerConfig, HMMConfig, BrokerModelConfig

config = RegimeControllerConfig(
    confirm_bars=3,
    score_weights={
        "volatility_score": 0.15,
        "trend_safety_score": 0.10,
        "spread_score": 0.18,
        "event_safety_score": 0.13,
        "inventory_score": 0.13,
        "execution_score": 0.08,
        "cb_efficiency_score": 0.08,
        "hmm_regime_score": 0.08,
        "broker_quality_score": 0.07,
    },
    hmm_config=HMMConfig(
        n_states=3,
        feature_window=60,
        min_bars_for_fit=200,
    ),
    broker_model_config=BrokerModelConfig(
        ewma_alpha=0.05,
        min_samples=20,
    ),
)
rc = RegimeController(config=config)
```

## Vaultログ

`RegimeVaultLogger` で JSONL 出力:

```python
from regime.vault_logger import RegimeVaultLogger
logger = RegimeVaultLogger(vault_root="/path/to/vault")
logger.write(decision)
```

出力例: `vault/regime/regime_decisions_20260621.jsonl`

## テスト

```bash
python3 -m pytest tests/ -v
```

現在 **98 tests, all passing**.

## 依存関係

- Python 3.10+
- コア: 標準ライブラリのみ（zero external dependencies）
- Phase 2 HMM: `pip install regime-controller-fx[hmm]` → numpy + hmmlearn (optional)

## 関連リソース

- [QuantStart: HMM risk filter](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
- [hmmlearn](https://hmmlearn.readthedocs.io/)
- [statsmodels MarkovRegression](https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html)
- [HMM + RL allocation paper](https://arxiv.org/abs/2605.27848)