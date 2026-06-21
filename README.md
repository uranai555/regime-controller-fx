# Regime Controller FX

FX/CB回収EA向け **市場レジーム判定 → 戦略許可/禁止** の上位フィルター。

## コンセプト

単なる「勝つEA」ではなく、**相場環境・ブローカー環境・ポジション状態を見て、戦略を動かす/止める上位制御モデル**。

```
市場データ → 特徴量化 → レジーム判定 → 戦略ON/OFF → リスク調整
```

**コア思想**: レジーム判定は直接売買シグナルに使わない。あくまで戦略許可/禁止のフィルター。

## フェーズ計画

| Phase | 内容 | 状態 |
|-------|------|------|
| **Phase 1** | ルールベース cb_run_score + persistence filter | ✅ 完了 |
| Phase 2 | HMM (GaussianHMM) によるレジーム分類 | 📅 未着手 |
| Phase 3 | ブローカー別 Execution Quality Model | 📅 未着手 |

## Phase 1 アーキテクチャ

```
bars / ticks / positions / execution_stats / existing_layer3_regime
  ↓
FeatureCollector
  ↓
RegimeFeatures
  ↓
SubScoreCalculator (6サブスコア)
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

### 6つのサブスコア

| スコア | 重み | 判定内容 |
|--------|------|----------|
| volatility_score | 0.20 | ATR百分位で「低すぎず高すぎないボラ」 |
| trend_safety_score | 0.20 | 既存Layer3レジームからトレンド強度 |
| spread_score | 0.20 | 現在スプレッド ÷ 平均スプレッド |
| event_safety_score | 0.15 | 指標イベントまでの残り時間 |
| inventory_score | 0.15 | ポジション偏り・含み損速度・証拠金 |
| execution_score | 0.10 | 約定品質（欠損時はデフォルト80） |

### hard block 条件（スコアより優先）

- `spread_ratio > 3.0` → **REDUCE_ONLY**
- `spread_ratio > 2.0` → **NO_NEW_ENTRY**
- `margin_level < 300` → **FORCE_EXIT**
- `floating_pnl_velocity < -30` → **REDUCE_ONLY**
- 指標15分前 → **NO_NEW_ENTRY**

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
)

print(decision.mode)             # NORMAL / CAUTION / NO_NEW_ENTRY / ...
print(decision.cb_run_score)     # 0.0-100.0
print(decision.allow_new_entry)  # bool
print(decision.risk_multiplier)  # 0.0-1.0
print(decision.reason_codes)     # ["VOLATILITY_SCORE_DEFAULTED", ...]
```

## 設定

```python
from regime import RegimeController, RegimeControllerConfig
from regime.sub_scores import ScoreConfig

config = RegimeControllerConfig(
    confirm_bars=3,
    score_weights={
        "volatility_score": 0.20,
        "trend_safety_score": 0.20,
        "spread_score": 0.20,
        "event_safety_score": 0.15,
        "inventory_score": 0.15,
        "execution_score": 0.10,
    },
    normal_score=80.0,
    caution_score=60.0,
    no_new_entry_score=40.0,
    reduce_only_score=20.0,
    no_new_entry_spread_ratio=2.0,
    reduce_only_spread_ratio=3.0,
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

現在 **26 tests, all passing**.

## 依存関係

- Python 3.10+
- 標準ライブラリのみ（zero external dependencies）

## 関連リソース

- [QuantStart: HMM risk filter](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
- [hmmlearn](https://hmmlearn.readthedocs.io/)
- [statsmodels MarkovRegression](https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html)
- [HMM + RL allocation paper](https://arxiv.org/abs/2605.27848)