"""regime-controller-fx — 市場レジーム判定 → 戦略許可/禁止の上位フィルター

FX/CB回収EA向け。
Phase 1: ルールベースの cb_run_score + persistence filter
Phase 2: GaussianHMM による隠れ状態推定（学習済みモデルがなければ Phase 1 にフォールバック）
Phase 3: ブローカー別 Execution Quality Model
"""

__version__ = "0.2.0"

from regime.regime_controller import RegimeController, RegimeControllerConfig
from regime.sub_scores import ScoreConfig
from regime.types import RegimeDecision, RegimeFeatures, RegimeMode
from regime.persistence_filter import PersistenceFilter, PersistenceFilterState
from regime.hmm_model import HMMModel, HMMConfig, GaussianHMM
from regime.execution_quality_model import (
    ExecutionQualityModel,
    ExecutionQualityConfig,
    ExecutionRecord,
)

__all__ = [
    "RegimeController",
    "RegimeControllerConfig",
    "ScoreConfig",
    "RegimeDecision",
    "RegimeFeatures",
    "RegimeMode",
    "PersistenceFilter",
    "PersistenceFilterState",
    "HMMModel",
    "HMMConfig",
    "GaussianHMM",
    "ExecutionQualityModel",
    "ExecutionQualityConfig",
    "ExecutionRecord",
]
