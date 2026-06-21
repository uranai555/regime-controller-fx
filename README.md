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
| **Phase 2** | GaussianHMM によるレジーム分類 | ✅ 完了 |
| **Phase 3** | ブローカー別 Execution Quality Model | ✅ 完了 |

## Phase 1 アーキテクチャ

```
bars / ticks / positions / execution_stats / existing_layer3_regime
  ↓
FeatureCollector
  ↓
RegimeFeatures
  ↓
SubScoreCalculator (7サブスコア)
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

### 7つのサブスコア

| スコア | 重み | 判定内容 |
|--------|------|----------|
| volatility_score | 0.18 | ATR百分位で「低すぎず高すぎないボラ」 |
| trend_safety_score | 0.18 | 既存Layer3レジームからトレンド強度 |
| spread_score | 0.18 | 現在スプレッド ÷ 平均スプレッド |
| event_safety_score | 0.13 | 指標イベントまでの残り時間 |
| inventory_score | 0.13 | ポジション偏り・含み損速度・証拠金 |
| execution_score | 0.10 | 約定品質（欠損時はデフォルト80） |
| cb_efficiency_score | 0.10 | CB収益 ÷ コスト |

### hard block 条件（スコアより優先）

- `spread_ratio > 3.0` → **REDUCE_ONLY**
- `spread_ratio > 2.0` → **NO_NEW_ENTRY**
- `margin_level < 300` → **FORCE_EXIT**
- `floating_pnl_velocity < -30` → **REDUCE_ONLY**
- 指標15分前 → **NO_NEW_ENTRY**

## Phase 2: HMM レジーム分類

### 概要

自前実装の GaussianHMM (EM + Viterbi) でマーケットの隠れ状態を推定する。
外部依存ゼロ（標準ライブラリのみ）。

### 入力特徴量

| 特徴量 | 計算元 | 次元 |
|--------|--------|------|
| `returns_series` | bars の close → log returns | T |
| `volatility_series` | 5バー窓の rolling std(returns) | T-4 |
| `spread_series` | bars の spread_avg | T |

HMM への入力は T×3 の行列（returns, volatility, spread）。

### 学習単位

**symbol 単位**。通貨ペアごとに独立した HMM を保持する。

```python
hmm = HMMModel(config=HMMConfig(n_states=3, n_features=3, min_obs=30))
hmm.fit(observations)  # T×3 行列
hmm.save("models/XAUUSD_hmm.json")
```

### 出力モード

HMM の状態はボラティリティで自動ソートされ、RegimeMode にマッピングされる:

| HMM State | ボラ特性 | デフォルト RegimeMode |
|-----------|----------|----------------------|
| 0 | 低ボラ | NORMAL |
| 1 | 中ボラ | CAUTION |
| 2 | 高ボラ | NO_NEW_ENTRY |

`predict_proba()` で各状態の確率も取得可能:

```python
state = hmm.predict(obs)     # 最新ステップの状態 (int)
proba = hmm.predict_proba(obs)  # 各状態の確率 [p0, p1, p2]
mode = hmm.predict_mode(obs)    # RegimeMode 文字列
```

### フォールバック条件

以下の場合は **Phase 1 のルールベース判定がそのまま使われる**:

1. `HMMModel` が未学習 (`is_fitted == False`)
2. `HMMModel` が `RegimeController` に渡されていない (`hmm_model=None`)
3. 入力時系列の長さが `min_obs` 未満
4. EM が収束しなかった場合
5. `raw_mode` が hard block (FORCE_EXIT / REDUCE_ONLY) の場合 → HMM は上書きしない

### パイプライン統合

```
FeatureCollector → RegimeFeatures
  ↓
SubScoreCalculator → cb_run_score → raw_mode (Phase 1)
  ↓
_apply_hmm_override → raw_mode (Phase 2: 学習済みHMMがあれば上書き)
  ↓
PersistenceFilter → confirmed_mode
  ↓
RegimeDecision
```

## Phase 3: Execution Quality Model

### 概要

ブローカー別に slippage / reject / fill の履歴を蓄積し、
`broker_execution_score` (0-100) を算出する。

### 入力

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `broker_id` | str | ブローカー識別子（例: "XM", "Titan"） |
| `account_id` | str | 口座番号 |
| `server_name` | str | MT4/MT5 サーバー名 |

約定記録は `ExecutionQualityModel.record()` で蓄積:

```python
eq_model = ExecutionQualityModel()
eq_model.record(broker_id="XM", slippage=0.3, rejected=False, filled=True)
```

### スコアリング

| 指標 | 良好 | 警告 | 危険 | ペナルティ |
|------|------|------|------|-----------|
| slippage_avg | < 0.2 | 0.2-0.5 | 0.5-1.0 | -10/-20/-40 |
| reject_rate | < 1% | 1-3% | 3-5% | -10/-20/-35 |
| fill_rate | > 98% | 95-98% | < 95% | 0/-10/-25 |

### フォールバック条件

- `broker_id` が未指定 → Phase 1 の `execution_score` をそのまま使用
- 履歴が `min_records`（デフォルト10）未満 → デフォルト 80 点を返す
- `ExecutionQualityModel` が `RegimeController` に渡されていない → Phase 1 のまま

### パイプライン統合

```
ExecutionQualityModel.enrich_features()
  → features.broker_execution_score に書き込み
  ↓
SubScoreCalculator → sub_scores["execution_score"]
  ↓
broker_execution_score があれば execution_score を上書き
  ↓
cb_run_score 計算に反映
```

## クイックスタート

### Phase 1 のみ（従来互換）

```python
from regime import RegimeController

rc = RegimeController()
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

### Phase 2 + Phase 3 統合

```python
from regime import (
    RegimeController, HMMModel, HMMConfig,
    ExecutionQualityModel, ExecutionQualityConfig,
)

# Phase 2: HMM モデル（事前学習済み or ロード）
hmm = HMMModel(config=HMMConfig(n_states=3, n_features=3))
# hmm.fit(training_data)  # T×3 行列で学習
# hmm = HMMModel.load("models/XAUUSD_hmm.json")

# Phase 3: Execution Quality Model
eq = ExecutionQualityModel()
eq.record(broker_id="XM", slippage=0.2, rejected=False, filled=True)
# ... 約定ごとに record() を呼ぶ

rc = RegimeController(
    hmm_model=hmm,
    execution_quality_model=eq,
)

decision = rc.evaluate(
    symbol="XAUUSD",
    bars=[...],
    broker_id="XM",
    account_id="12345",
    server_name="XMTrading-MT5",
    current_time_ms=1624277100000,
)
```

## 設定

```python
from regime import RegimeController, RegimeControllerConfig
from regime.sub_scores import ScoreConfig

config = RegimeControllerConfig(
    confirm_bars=3,
    score_weights={
        "volatility_score": 0.18,
        "trend_safety_score": 0.18,
        "spread_score": 0.18,
        "event_safety_score": 0.13,
        "inventory_score": 0.13,
        "execution_score": 0.10,
        "cb_efficiency_score": 0.10,
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

## 依存関係

- Python 3.10+
- 標準ライブラリのみ（zero external dependencies）
- GaussianHMM は自前実装（`regime/hmm_model.py`）

## 公開 API

| クラス | モジュール | Phase |
|--------|-----------|-------|
| `RegimeController` | `regime.regime_controller` | 1/2/3 |
| `RegimeControllerConfig` | `regime.regime_controller` | 1 |
| `RegimeFeatures` | `regime.types` | 1/2/3 |
| `RegimeDecision` | `regime.types` | 1 |
| `RegimeMode` | `regime.types` | 1 |
| `HMMModel` | `regime.hmm_model` | 2 |
| `HMMConfig` | `regime.hmm_model` | 2 |
| `GaussianHMM` | `regime.hmm_model` | 2 |
| `ExecutionQualityModel` | `regime.execution_quality_model` | 3 |
| `ExecutionQualityConfig` | `regime.execution_quality_model` | 3 |
| `ExecutionRecord` | `regime.execution_quality_model` | 3 |
| `PersistenceFilter` | `regime.persistence_filter` | 1 |
| `ScoreConfig` | `regime.sub_scores` | 1 |

## 関連リソース

- [QuantStart: HMM risk filter](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
- [hmmlearn](https://hmmlearn.readthedocs.io/)
- [statsmodels MarkovRegression](https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html)
- [HMM + RL allocation paper](https://arxiv.org/abs/2605.27848)
