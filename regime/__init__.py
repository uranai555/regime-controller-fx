"""regime-controller-fx — 市場レジーム判定 → 戦略許可/禁止の上位フィルター

FX/CB回収EA向け。
Phase 1: ルールベースの cb_run_score + persistence filter。
"""

__version__ = "0.1.0"

from regime.regime_controller import RegimeController, RegimeControllerConfig
from regime.sub_scores import ScoreConfig
from regime.types import RegimeDecision, RegimeFeatures, RegimeMode
from regime.persistence_filter import PersistenceFilter, PersistenceFilterState

__all__ = [
    "RegimeController",
    "RegimeControllerConfig",
    "ScoreConfig",
    "RegimeDecision",
    "RegimeFeatures",
    "RegimeMode",
    "PersistenceFilter",
    "PersistenceFilterState",
]
