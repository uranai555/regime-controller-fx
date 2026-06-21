"""regime-controller-fx — 市場レジーム判定 → 戦略許可/禁止の上位フィルター

FX/CB回収EA向け。
Phase 1: ルールベースの cb_run_score + persistence filter
Phase 2: HMM (GaussianHMM) によるレジーム分類
Phase 3: ブローカー別 Execution Quality Model
"""

__version__ = "0.2.0"

from regime.regime_controller import RegimeController, RegimeControllerConfig
from regime.sub_scores import ScoreConfig
from regime.types import RegimeDecision, RegimeFeatures, RegimeMode
from regime.persistence_filter import PersistenceFilter, PersistenceFilterState
from regime.hmm_regime import (
    HMMRegimeClassifier,
    HMMConfig,
    HMMRegimeState,
    HMMResult,
)
from regime.broker_model import (
    BrokerQualityModel,
    BrokerModelConfig,
    BrokerProfile,
    ExecutionEvent,
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
    # Phase 2
    "HMMRegimeClassifier",
    "HMMConfig",
    "HMMRegimeState",
    "HMMResult",
    # Phase 3
    "BrokerQualityModel",
    "BrokerModelConfig",
    "BrokerProfile",
    "ExecutionEvent",
]
