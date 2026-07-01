# Phase 2 & Phase 3 実装計画

## 1. 現状の関連ファイルと責務

### regime/ 配下のモジュール一覧

| ファイル | 責務 | Phase 2/3 への影響 |
|----------|------|-------------------|
| `types.py` | `RegimeMode`, `RegimeFeatures`, `RegimeDecision` データ型定義 | HMM出力の型追加、ブローカー品質メトリクスの型追加が必要 |
| `feature_collector.py` | bars/ticks/positions/execution_stats → `RegimeFeatures` 変換 | HMM用の特徴量ベクトル抽出追加、ブローカー別統計の収集追加 |
| `sub_scores.py` | 7つのサブスコア計算 + `aggregate()` で cb_run_score | HMM出力をサブスコアとして統合、execution_score をブローカーモデルで置換 |
| `regime_controller.py` | パイプラインオーケストレーター | HMM分類器を統合ポイントに追加、ブローカー品質モデルの注入 |
| `persistence_filter.py` | 非対称フィルター（危険即時・安全確認） | 変更不要（Phase 2/3 の出力もこのフィルタを通る） |
| `vault_logger.py` | JSONL 永続化 | HMM regime / broker_quality をログに追加 |
| `__init__.py` | パッケージ公開API | 新クラスのexport追加 |

### tests/ 配下

| ファイル | 内容 |
|----------|------|
| `test_regime.py` | 43テスト (全pass) — SubScore, PersistenceFilter, Controller, FeatureCollector |

---

## 2. Phase 2: HMM (GaussianHMM) によるレジーム分類

### 2.1 設計方針

**目的**: ルールベース判定に加え、価格・ボラティリティの時系列パターンから隠れ状態（レジーム）を推定し、`trend_safety_score` の精度を向上させる。

**依存関係**: `numpy` + `hmmlearn` を optional dependency として追加。HMM未インストール時はPhase 1のルールベースにフォールバック。

**HMM の入力特徴量**:
- `log_returns` (対数リターン)
- `realized_volatility` (ローリング実現ボラ)
- `range_ratio` (high-low / close)
- `volume_change` (出来高変化率、利用可能な場合)

**HMM の出力**: 隠れ状態ラベル → セマンティックマッピング:
- State 0: LOW_VOL (低ボラ・レンジ) → 安全
- State 1: TRENDING (方向性あり) → 中程度リスク
- State 2: HIGH_VOL (高ボラ・クラッシュ) → 危険

### 2.2 実装タスク

| # | タスク | 新規/既存 | 対象ファイル |
|---|--------|-----------|-------------|
| 2.1 | `regime/hmm_regime.py` 新規作成 — GaussianHMM wrapper | 新規 | `regime/hmm_regime.py` |
| 2.2 | HMM 特徴量抽出メソッド追加 | 既存修正 | `regime/feature_collector.py` |
| 2.3 | `RegimeFeatures` に HMM 出力フィールド追加 | 既存修正 | `regime/types.py` |
| 2.4 | `hmm_regime_score` サブスコア追加 | 既存修正 | `regime/sub_scores.py` |
| 2.5 | RegimeController に HMM 統合 | 既存修正 | `regime/regime_controller.py` |
| 2.6 | HMM config dataclass 追加 | 既存修正 | `regime/regime_controller.py` |
| 2.7 | VaultLogger に HMM regime 記録 | 既存修正 | `regime/vault_logger.py` |
| 2.8 | `__init__.py` に export 追加 | 既存修正 | `regime/__init__.py` |
| 2.9 | テスト追加: HMM 利用可能時 | 新規 | `tests/test_hmm_regime.py` |
| 2.10 | テスト追加: HMM フォールバック（hmmlearn 未インストール時） | 新規 | `tests/test_hmm_regime.py` |
| 2.11 | `setup.py` に extras_require 追加 | 既存修正 | `setup.py` |

### 2.3 `regime/hmm_regime.py` 設計

```python
class HMMRegimeClassifier:
    """GaussianHMM による市場レジーム分類器。

    - fit(bars): 過去データから HMM パラメータを推定
    - predict(bars): 現在のレジーム状態を推定
    - フォールバック: hmmlearn 未インストール → None 返却
    """

    def __init__(self, n_states=3, feature_window=60, ...):
        ...

    def fit(self, bars: List[Dict]) -> "HMMRegimeClassifier":
        """学習（オンライン用 or 事前学習済みパラメータロード）"""
        ...

    def predict(self, bars: List[Dict]) -> Optional[HMMRegimeState]:
        """現在バーのレジーム推定"""
        ...

    def save_params(self, path: str) -> None:
        """学習済みパラメータの永続化"""
        ...

    def load_params(self, path: str) -> None:
        """パラメータ復元"""
        ...
```

### 2.4 Score 統合方式

`hmm_regime_score` を新しいサブスコアとして追加:
- HMM State = LOW_VOL → score 100
- HMM State = TRENDING → score 55
- HMM State = HIGH_VOL → score 15
- HMM 利用不可 → `trend_safety_score` のフォールバック値を使用

重みの再配分（Phase 2 適用時）:
```
volatility_score:    0.15  (was 0.18)
trend_safety_score:  0.10  (was 0.18, HMM が一部置換)
hmm_regime_score:    0.15  (新規)
spread_score:        0.18  (unchanged)
event_safety_score:  0.13  (unchanged)
inventory_score:     0.13  (unchanged)
execution_score:     0.08  (was 0.10, Phase 3 で再配分)
cb_efficiency_score: 0.08  (was 0.10)
```

---

## 3. Phase 3: ブローカー別 Execution Quality Model

### 3.1 設計方針

**目的**: ブローカーごとの約定品質特性（スリッページ傾向、リジェクト率、フィルレート、時間帯依存性）を学習し、`execution_score` の精度を大幅に向上。

**コア思想**: ブローカーAでは安全でもブローカーBでは危険、というケースを検出する。

### 3.2 ブローカー品質メトリクス

| メトリクス | 説明 | 計算方法 |
|-----------|------|----------|
| `slippage_mean` | 平均スリッページ (pips) | EWMA |
| `slippage_p95` | 95th パーセンタイルスリッページ | ローリング |
| `reject_rate` | 注文リジェクト率 | count/total |
| `fill_rate` | フィル率 | filled/submitted |
| `requote_rate` | リクオート率 | requotes/total |
| `latency_mean_ms` | 平均約定レイテンシ | EWMA |
| `latency_p95_ms` | P95 レイテンシ | ローリング |
| `time_of_day_factor` | 時間帯別品質変動 | hourly bucket |
| `spread_markup` | ブローカー固有スプレッド上乗せ | broker_spread - ecn_ref |

### 3.3 実装タスク

| # | タスク | 新規/既存 | 対象ファイル |
|---|--------|-----------|-------------|
| 3.1 | `regime/broker_model.py` 新規作成 — BrokerQualityModel | 新規 | `regime/broker_model.py` |
| 3.2 | `BrokerProfile` config dataclass | 新規 | `regime/broker_model.py` |
| 3.3 | `RegimeFeatures` にブローカー品質フィールド追加 | 既存修正 | `regime/types.py` |
| 3.4 | FeatureCollector にブローカー統計収集追加 | 既存修正 | `regime/feature_collector.py` |
| 3.5 | `execution_score` をブローカーモデル対応に拡張 | 既存修正 | `regime/sub_scores.py` |
| 3.6 | RegimeController にブローカーモデル注入 | 既存修正 | `regime/regime_controller.py` |
| 3.7 | ブローカープロファイル永続化 (JSON) | 新規 | `regime/broker_model.py` |
| 3.8 | VaultLogger にブローカー品質記録 | 既存修正 | `regime/vault_logger.py` |
| 3.9 | `__init__.py` に export 追加 | 既存修正 | `regime/__init__.py` |
| 3.10 | テスト追加: BrokerQualityModel 単体 | 新規 | `tests/test_broker_model.py` |
| 3.11 | テスト追加: ブローカーモデル統合 | 新規 | `tests/test_broker_model.py` |
| 3.12 | `setup.py` 更新（numpy optional） | 既存修正 | `setup.py` |

### 3.4 `regime/broker_model.py` 設計

```python
@dataclass
class BrokerProfile:
    """ブローカー固有の約定品質プロファイル。"""
    broker_id: str
    slippage_mean: float = 0.0
    slippage_p95: float = 0.0
    reject_rate: float = 0.0
    fill_rate: float = 1.0
    requote_rate: float = 0.0
    latency_mean_ms: float = 0.0
    latency_p95_ms: float = 0.0
    spread_markup: float = 0.0
    hourly_quality: Dict[int, float] = field(default_factory=dict)  # hour → quality_factor
    sample_count: int = 0


class BrokerQualityModel:
    """ブローカー別の約定品質を学習・スコア化するモデル。

    - update(execution_event): 新しい約定イベントでプロファイルを更新
    - score(broker_id, hour): 現在の品質スコア (0-100) を返す
    - save/load: プロファイルの永続化
    """
    ...
```

### 3.5 Score 統合方式

`execution_score` の計算を拡張:
1. ブローカーモデルが利用可能 → `BrokerQualityModel.score()` を使用
2. 利用不可 → Phase 1 のルールベース (slippage_avg, reject_rate, fill_rate) にフォールバック

ブローカーモデル有効時の hard block 追加:
- `broker_quality_score < 20` かつ `latency_p95 > 2000ms` → NO_NEW_ENTRY
- `reject_rate > 0.15` → REDUCE_ONLY

---

## 4. 設定・永続化

### 新規設定項目

```python
@dataclass
class HMMConfig:
    enabled: bool = True
    n_states: int = 3
    feature_window: int = 60
    min_bars_for_fit: int = 200
    refit_interval: int = 500
    params_path: Optional[str] = None  # 事前学習パラメータのパス

@dataclass
class BrokerModelConfig:
    enabled: bool = True
    ewma_alpha: float = 0.05
    profile_path: Optional[str] = None
    hard_block_quality_threshold: float = 20.0
    hard_block_latency_ms: float = 2000.0
    hard_block_reject_rate: float = 0.15
```

### 永続化

| データ | 形式 | 保存先 |
|--------|------|--------|
| HMM パラメータ | JSON (means, covars, transmat) | `vault/regime/hmm_params_{symbol}.json` |
| ブローカープロファイル | JSON | `vault/regime/broker_profiles.json` |
| 判定ログ | JSONL (拡張) | `vault/regime/regime_decisions_{date}.jsonl` |

---

## 5. 実装順序（推奨）

```
Phase 2.1-2.3  → types + feature_collector + hmm_regime.py (コア)
Phase 2.4-2.6  → sub_scores + controller 統合
Phase 2.7-2.11 → logger + tests + setup.py

Phase 3.1-3.3  → broker_model.py + types
Phase 3.4-3.6  → feature_collector + sub_scores + controller 統合
Phase 3.7-3.12 → 永続化 + tests + setup.py
```

---

## 6. 既存テストへの影響

- Phase 1 のテスト (43件) は全て backward compatible に維持
- HMM/ブローカーモデルは optional — 無効時は Phase 1 と同じ挙動
- 新規テスト: ~25件追加見込み
