"""types.py — RegimeController のデータ型定義

Phase 1 ではルールベースの cb_run_score + persistence filter で
レジーム判定を行う。このモジュールには全てのデータクラスと enum を
集約し、他モジュール間の依存を types.py だけに閉じる。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RegimeMode(str, Enum):
    """出力モード（危険度昇順）。

    危険度順（PersistenceFilter で使う）:
      NORMAL < CAUTION < NO_NEW_ENTRY < REDUCE_ONLY < FORCE_EXIT
    """
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    NO_NEW_ENTRY = "NO_NEW_ENTRY"
    REDUCE_ONLY = "REDUCE_ONLY"
    FORCE_EXIT = "FORCE_EXIT"

    @property
    def risk_level(self) -> int:
        return _RISK_ORDER[self]


_RISK_ORDER: Dict[RegimeMode, int] = {
    RegimeMode.NORMAL: 0,
    RegimeMode.CAUTION: 1,
    RegimeMode.NO_NEW_ENTRY: 2,
    RegimeMode.REDUCE_ONLY: 3,
    RegimeMode.FORCE_EXIT: 4,
}


@dataclass
class RegimeFeatures:
    """RegimeController への入力特徴量。

    全てのフィールドは Optional。欠損したものは missing_fields に
    理由が入る。sub_scores.py 側で安全寄りのデフォルト値にフォールバックする。
    """

    symbol: str
    timestamp: str  # ISO8601

    # ── Price / Volatility ──
    atr: Optional[float] = None
    atr_pct: Optional[float] = None
    realized_volatility: Optional[float] = None
    range_ratio: Optional[float] = None
    trend_strength: Optional[float] = None
    adx: Optional[float] = None
    ma_slope: Optional[float] = None
    consecutive_directional_bars: Optional[int] = None

    # ── Spread / Cost ──
    spread: Optional[float] = None       # current spread (pips)
    spread_avg: Optional[float] = None    # trailing mean
    spread_ratio: Optional[float] = None  # spread / spread_avg

    # ── Time / Event ──
    hour: Optional[int] = None
    weekday: Optional[int] = None
    minutes_to_high_impact_event: Optional[int] = None

    # ── Position / Inventory ──
    floating_pnl: Optional[float] = None
    floating_pnl_velocity: Optional[float] = None
    inventory_imbalance: Optional[float] = None
    margin_level: Optional[float] = None
    open_position_count: Optional[int] = None
    long_lots: Optional[float] = None
    short_lots: Optional[float] = None
    net_lots: Optional[float] = None

    # ── Execution Quality ──
    slippage_avg: Optional[float] = None
    order_reject_rate: Optional[float] = None
    fill_rate: Optional[float] = None

    # ── CB / Cost Efficiency ──
    rebate_per_lot: Optional[float] = None
    cost_per_lot: Optional[float] = None
    cb_edge_per_lot: Optional[float] = None

    # ── Existing v3 Layer 3 Output ──
    existing_layer3_regime: Optional[str] = None

    # ── 欠損追跡 ──
    missing_fields: List[str] = field(default_factory=list)


@dataclass
class RegimeDecision:
    """RegimeController の出力。

    このオブジェクトは BUY/SELL の方向を持たない。
    あくまで「戦略を許可/禁止する」ための司令として使う。
    """

    mode: RegimeMode            # persistence filter 通過後の確定モード
    raw_mode: RegimeMode        # persistence filter 適用前の生モード

    allow_new_entry: bool
    allow_add_position: bool
    reduce_only: bool
    force_exit: bool

    risk_multiplier: float      # 0.0 ~ 1.0 — ロット倍率

    cb_run_score: float         # 0.0 ~ 100.0
    sub_scores: Dict[str, float]
    reason_codes: List[str]

    features: Dict[str, Any]    # ログ用にフラット化した特徴量
    missing_fields: List[str]

    timestamp: str = ""
    symbol: str = ""
