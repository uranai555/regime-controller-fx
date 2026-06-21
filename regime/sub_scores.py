"""sub_scores.py — RegimeFeatures から 6つのサブスコア + cb_run_score を計算する

各サブスコアは 0〜100。100 = 良好（正常稼働可能）、0 = 危険（停止）。

Phase 1 ではルールベース。全てのパラメータは config で上書き可能。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from regime.types import RegimeFeatures


# ── デフォルト閾値 ──

@dataclass
class ScoreConfig:
    """サブスコア計算に使う閾値一覧。config.yaml から注入される。"""
    # volatility_score
    atr_pct_very_low: float = 0.0001
    atr_pct_normal_low: float = 0.0003
    atr_pct_normal_high: float = 0.002
    atr_pct_high: float = 0.003
    atr_pct_extreme: float = 0.005
    volatility_default: int = 70

    # trend_safety_score
    trend_safety_default: int = 100

    # spread_score
    spread_ratio_widening: float = 1.5
    spread_ratio_wide: float = 2.0
    spread_ratio_danger: float = 3.0
    spread_default: int = 70

    # event_safety_score
    event_default: int = 100
    event_danger_minutes: int = 15
    event_warning_minutes: int = 60

    # inventory_score
    imbalance_warning: float = 0.4
    imbalance_danger: float = 0.6
    imbalance_severe: float = 0.8
    pnl_velocity_warning: float = -5.0
    pnl_velocity_danger: float = -15.0
    margin_warning: float = 500.0
    margin_danger: float = 300.0
    position_usage_warning: float = 0.6
    position_usage_danger: float = 0.8
    inventory_default: int = 100

    # execution_score
    execution_default: int = 80
    reject_rate_warning: float = 0.02
    reject_rate_danger: float = 0.05
    slippage_warning: float = 0.5
    slippage_danger: float = 1.0
    fill_rate_warning: float = 0.95

    # cb_efficiency_score
    cb_edge_warning: float = 0.0
    cb_edge_positive: float = 0.3
    cb_edge_good: float = 0.5
    cb_default: int = 70


# ── Reason Code 定数 ──

# Volatility
RC_VOLATILITY_DEFAULTED = "VOLATILITY_SCORE_DEFAULTED"
RC_VOL_TOO_LOW = "VOLATILITY_TOO_LOW"
RC_VOL_HIGH = "VOLATILITY_HIGH"
RC_VOL_EXTREME = "VOLATILITY_EXTREME"

# Trend
RC_TREND_DEFAULTED = "TREND_SCORE_DEFAULTED"
RC_LAYER3_VOLATILE = "LAYER3_VOLATILE"
RC_LAYER3_TRENDING = "LAYER3_TRENDING"

# Spread
RC_SPREAD_DEFAULTED = "SPREAD_SCORE_DEFAULTED"
RC_SPREAD_WIDENING = "SPREAD_WIDENING"
RC_SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
RC_SPREAD_DANGER = "SPREAD_DANGER"

# Event
RC_EVENT_DEFAULTED = "EVENT_SCORE_DEFAULTED"
RC_EVENT_NEAR = "HIGH_IMPACT_EVENT_NEAR"
RC_EVENT_SOON = "HIGH_IMPACT_EVENT_SOON"

# Inventory
RC_INVENTORY_DEFAULTED = "INVENTORY_SCORE_DEFAULTED"
RC_INVENTORY_IMBALANCED = "INVENTORY_IMBALANCED"
RC_INVENTORY_SEVERE = "INVENTORY_SEVERELY_IMBALANCED"
RC_PNL_DETERIORATING = "FLOATING_PNL_DETERIORATING"
RC_MARGIN_LOW = "MARGIN_LEVEL_LOW"
RC_MARGIN_DANGER = "MARGIN_LEVEL_CRITICAL"

# Execution
RC_EXECUTION_DEFAULTED = "EXECUTION_SCORE_DEFAULTED"
RC_REJECT_HIGH = "ORDER_REJECT_RATE_HIGH"
RC_SLIPPAGE_HIGH = "SLIPPAGE_HIGH"
RC_FILL_LOW = "FILL_RATE_LOW"

# CB
RC_CB_DEFAULTED = "CB_SCORE_DEFAULTED"
RC_CB_NEGATIVE = "CB_EDGE_NEGATIVE"


class SubScoreCalculator:
    """RegimeFeatures からサブスコアと reason codes を計算する。

    全ての計算は 0〜100 に正規化される。
    """

    def __init__(self, config: ScoreConfig = None):  # type: ignore[assignment]
        self.cfg = config or ScoreConfig()

    def calculate(self, features: RegimeFeatures) -> Tuple[Dict[str, float], List[str]]:
        """全てのサブスコアを計算し、(sub_scores, reason_codes) を返す。"""
        sub_scores: Dict[str, float] = {}
        reason_codes: List[str] = []

        sub_scores["volatility_score"], codes = self._volatility_score(features)
        reason_codes.extend(codes)

        sub_scores["trend_safety_score"], codes = self._trend_safety_score(features)
        reason_codes.extend(codes)

        sub_scores["spread_score"], codes = self._spread_score(features)
        reason_codes.extend(codes)

        sub_scores["event_safety_score"], codes = self._event_safety_score(features)
        reason_codes.extend(codes)

        sub_scores["inventory_score"], codes = self._inventory_score(features)
        reason_codes.extend(codes)

        sub_scores["execution_score"], codes = self._execution_score(features)
        reason_codes.extend(codes)

        sub_scores["cb_efficiency_score"], codes = self._cb_efficiency_score(features)
        reason_codes.extend(codes)

        return sub_scores, reason_codes

    def aggregate(
        self,
        sub_scores: Dict[str, float],
        weights: Dict[str, float] = None,  # type: ignore[assignment]
    ) -> float:
        """重み付き平均で cb_run_score を計算する。"""
        if weights is None:
            weights = {
                "volatility_score": 0.18,
                "trend_safety_score": 0.18,
                "spread_score": 0.18,
                "event_safety_score": 0.13,
                "inventory_score": 0.13,
                "execution_score": 0.10,
                "cb_efficiency_score": 0.10,
            }

        score = 0.0
        total_weight = 0.0
        for key, w in weights.items():
            if key in sub_scores:
                score += sub_scores[key] * w
                total_weight += w

        if total_weight > 0:
            return round(score / total_weight, 2)
        return 0.0

    # ── Sub-score calculators ──

    def _volatility_score(
        self, features: RegimeFeatures
    ) -> Tuple[float, List[str]]:
        """CB回転に適した「低すぎず高すぎないボラ」を評価。"""
        codes: List[str] = []

        if features.atr_pct is None:
            codes.append(RC_VOLATILITY_DEFAULTED)
            return self.cfg.volatility_default, codes

        p = features.atr_pct

        if p < self.cfg.atr_pct_very_low:
            codes.append(RC_VOL_TOO_LOW)
            return 60.0, codes
        if p < self.cfg.atr_pct_normal_low:
            # very_low〜normal_low: 低ボラだが極端ではない
            return 80.0, codes
        if p <= self.cfg.atr_pct_normal_high:
            return 100.0, codes  # 理想
        if p <= self.cfg.atr_pct_high:
            return 75.0, codes
        if self.cfg.atr_pct_high < p <= self.cfg.atr_pct_extreme:
            codes.append(RC_VOL_HIGH)
            return 40.0, codes

        codes.append(RC_VOL_EXTREME)
        return 10.0, codes

    def _trend_safety_score(
        self, features: RegimeFeatures
    ) -> Tuple[float, List[str]]:
        """強トレンドはCB回転に危険→ lower score。"""
        codes: List[str] = []

        # 既存 Layer 3 レジームを主要指標として使う
        regime = features.existing_layer3_regime
        if regime is None:
            codes.append(RC_TREND_DEFAULTED)
            return self.cfg.trend_safety_default, codes

        if regime == "VOLATILE":
            codes.append(RC_LAYER3_VOLATILE)
            return 30.0, codes
        if regime in ("TRENDING_UP", "TRENDING_DOWN"):
            codes.append(RC_LAYER3_TRENDING)
            return 50.0, codes
        if regime == "RANGING":
            return 100.0, codes

        # 未知のレジーム文字列
        return self.cfg.trend_safety_default, codes

    def _spread_score(
        self, features: RegimeFeatures
    ) -> Tuple[float, List[str]]:
        """スプレッド悪化を評価。"""
        codes: List[str] = []

        if features.spread_ratio is None:
            codes.append(RC_SPREAD_DEFAULTED)
            return self.cfg.spread_default, codes

        r = features.spread_ratio

        if r <= 1.2:
            return 100.0, codes
        if r <= self.cfg.spread_ratio_widening:
            return 80.0, codes
        if r <= self.cfg.spread_ratio_wide:
            codes.append(RC_SPREAD_WIDENING)
            return 50.0, codes

        codes.append(RC_SPREAD_TOO_WIDE)
        if r <= self.cfg.spread_ratio_danger:
            return 20.0, codes

        codes.append(RC_SPREAD_DANGER)
        return 0.0, codes

    def _event_safety_score(
        self, features: RegimeFeatures
    ) -> Tuple[float, List[str]]:
        """イベントリスクを評価。

        Phase 1 では minutes_to_high_impact_event が None なら 100 にフォールバック。
        """
        codes: List[str] = []

        minutes = features.minutes_to_high_impact_event
        if minutes is None:
            codes.append(RC_EVENT_DEFAULTED)
            return self.cfg.event_default, codes

        if 0 <= minutes <= self.cfg.event_danger_minutes:
            codes.append(RC_EVENT_NEAR)
            return 0.0, codes
        if minutes <= self.cfg.event_warning_minutes:
            codes.append(RC_EVENT_SOON)
            return 40.0, codes

        return 100.0, codes

    def _inventory_score(
        self, features: RegimeFeatures
    ) -> Tuple[float, List[str]]:
        """ポジション偏り・含み損速度・証拠金リスクを評価。"""
        codes: List[str] = []
        score = 100.0

        # ポジションなし → 満点
        if features.open_position_count is None or features.open_position_count == 0:
            return score, codes

        # インベントリインバランス
        imb = features.inventory_imbalance
        if imb is not None:
            if imb > self.cfg.imbalance_severe:
                codes.append(RC_INVENTORY_SEVERE)
                score -= 50
            elif imb > self.cfg.imbalance_danger:
                codes.append(RC_INVENTORY_IMBALANCED)
                score -= 30
            elif imb > self.cfg.imbalance_warning:
                score -= 10

        # 含み損速度
        vel = features.floating_pnl_velocity
        if vel is not None and vel < 0:
            if vel < self.cfg.pnl_velocity_danger:
                codes.append(RC_PNL_DETERIORATING)
                score -= 40
            elif vel < self.cfg.pnl_velocity_warning:
                codes.append(RC_PNL_DETERIORATING)
                score -= 20

        # 証拠金
        margin = features.margin_level
        if margin is not None:
            if margin < self.cfg.margin_danger:
                codes.append(RC_MARGIN_DANGER)
                score -= 60
            elif margin < self.cfg.margin_warning:
                codes.append(RC_MARGIN_LOW)
                score -= 30

        # ポジション使用率
        if features.open_position_count is not None and features.open_position_count > 0:
            if features.open_position_count >= 8:
                score -= 20
            elif features.open_position_count >= 5:
                score -= 10

        return max(0.0, min(100.0, score)), codes

    def _execution_score(
        self, features: RegimeFeatures
    ) -> Tuple[float, List[str]]:
        """約定品質の悪化を評価。データがない場合はデフォルト80。"""
        codes: List[str] = []

        if "execution_stats_not_provided" in features.missing_fields:
            codes.append(RC_EXECUTION_DEFAULTED)
            return self.cfg.execution_default, codes

        score = 100.0

        reject_rate = features.order_reject_rate
        if reject_rate is not None:
            if reject_rate > self.cfg.reject_rate_danger:
                codes.append(RC_REJECT_HIGH)
                score -= 40
            elif reject_rate > self.cfg.reject_rate_warning:
                codes.append(RC_REJECT_HIGH)
                score -= 20

        slippage = features.slippage_avg
        if slippage is not None:
            if slippage > self.cfg.slippage_danger:
                codes.append(RC_SLIPPAGE_HIGH)
                score -= 35
            elif slippage > self.cfg.slippage_warning:
                codes.append(RC_SLIPPAGE_HIGH)
                score -= 15

        fill_rate = features.fill_rate
        if fill_rate is not None:
            if fill_rate < self.cfg.fill_rate_warning:
                codes.append(RC_FILL_LOW)
                score -= 20

        return max(0.0, min(100.0, score)), codes

    def _cb_efficiency_score(
        self, features: RegimeFeatures
    ) -> Tuple[float, List[str]]:
        """CB収益が実質コストに対して残るかを評価。"""
        codes: List[str] = []

        if features.cb_edge_per_lot is None:
            codes.append(RC_CB_DEFAULTED)
            return self.cfg.cb_default, codes

        edge = features.cb_edge_per_lot
        if edge < self.cfg.cb_edge_warning:
            codes.append(RC_CB_NEGATIVE)
            return 0.0, codes
        if edge < self.cfg.cb_edge_positive:
            return 30.0, codes
        if edge < self.cfg.cb_edge_good:
            return 55.0, codes

        return 100.0, codes
