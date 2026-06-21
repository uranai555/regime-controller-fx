"""regime_controller.py — RegimeController オーケストレーター

FeatureCollector → SubScoreCalculator → raw_mode → PersistenceFilter
のパイプラインを統括する。出力は BUY/SELL ではなく戦略許可/禁止。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from regime.feature_collector import FeatureCollector
from regime.persistence_filter import PersistenceFilter
from regime.sub_scores import ScoreConfig, SubScoreCalculator
from regime.types import RegimeDecision, RegimeFeatures, RegimeMode

from regime.sub_scores import (
    RC_SPREAD_TOO_WIDE,
    RC_SPREAD_DANGER,
    RC_SPREAD_WIDENING,
    RC_EVENT_NEAR,
    RC_MARGIN_DANGER,
    RC_MARGIN_LOW,
    RC_PNL_DETERIORATING,
    RC_INVENTORY_SEVERE,
    RC_VOL_EXTREME,
)


@dataclass
class RegimeControllerConfig:
    """RegimeController 全体の設定。"""

    enabled: bool = True
    confirm_bars: int = 3

    # 重み
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        "volatility_score": 0.20,
        "trend_safety_score": 0.20,
        "spread_score": 0.20,
        "event_safety_score": 0.15,
        "inventory_score": 0.15,
        "execution_score": 0.10,
    })

    # モード閾値
    normal_score: float = 80.0
    caution_score: float = 60.0
    no_new_entry_score: float = 40.0
    reduce_only_score: float = 20.0

    # spread hard block
    no_new_entry_spread_ratio: float = 2.0
    reduce_only_spread_ratio: float = 3.0

    # リスク倍率
    risk_multiplier_map: Dict[str, float] = field(default_factory=lambda: {
        RegimeMode.NORMAL.value: 1.0,
        RegimeMode.CAUTION.value: 0.5,
        RegimeMode.NO_NEW_ENTRY.value: 0.0,
        RegimeMode.REDUCE_ONLY.value: 0.0,
        RegimeMode.FORCE_EXIT.value: 0.0,
    })


class RegimeController:
    """レジーム判定 → 戦略許可/禁止の上位フィルター。

    Phase 1 ではルールベース（HMMなし / 各バー評価）。
    """

    def __init__(
        self,
        config: Optional[RegimeControllerConfig] = None,
        score_config: Optional[ScoreConfig] = None,
    ):
        self.config = config or RegimeControllerConfig()
        self.feature_collector = FeatureCollector()
        self.scorer = SubScoreCalculator(score_config or ScoreConfig())
        self.persistence_filter = PersistenceFilter(
            confirm_bars=self.config.confirm_bars
        )

    def evaluate(
        self,
        symbol: str = "",
        bars: Optional[List[Dict[str, Any]]] = None,
        ticks: Optional[List[Dict[str, Any]]] = None,
        positions: Optional[List[Dict[str, Any]]] = None,
        execution_stats: Optional[Dict[str, Any]] = None,
        existing_layer3_regime: Optional[str] = None,
        account_state: Optional[Dict[str, Any]] = None,
        cb_config: Optional[Dict[str, Any]] = None,
        current_time_ms: Optional[int] = None,
    ) -> RegimeDecision:
        """FeatureCollector → SubScoreCalculator → raw_mode → PersistenceFilter を実行する。

        config.enabled=False の場合は常に NORMAL を返す（旧挙動との互換性）。
        """
        if not self.config.enabled:
            return self._disabled_decision(symbol, current_time_ms)

        # 1. 特徴量収集
        features = self.feature_collector.collect(
            symbol=symbol,
            bars=bars,
            ticks=ticks,
            positions=positions,
            execution_stats=execution_stats,
            existing_layer3_regime=existing_layer3_regime,
            account_state=account_state,
            cb_config=cb_config,
            current_time_ms=current_time_ms,
        )

        # 2. サブスコア計算
        sub_scores, reason_codes = self.scorer.calculate(features)
        cb_run_score = self.scorer.aggregate(sub_scores, self.config.score_weights)

        # 3. raw_mode 判定
        raw_mode = self._decide_raw_mode(features, sub_scores, cb_run_score, reason_codes)

        # 4. Persistence filter
        confirmed_mode = self.persistence_filter.update(raw_mode)

        # 5. RegimeDecision に変換
        return self._build_decision(
            mode=confirmed_mode,
            raw_mode=raw_mode,
            cb_run_score=cb_run_score,
            sub_scores=sub_scores,
            reason_codes=reason_codes,
            features=features,
        )

    def _decide_raw_mode(
        self,
        features: RegimeFeatures,
        sub_scores: Dict[str, float],
        cb_run_score: float,
        reason_codes: List[str],
    ) -> RegimeMode:
        """cb_run_score と hard block 条件から raw_mode を決める。

        hard block はスコアより優先される。
        """

        # ── hard block: margin ──
        if features.margin_level is not None and features.margin_level < 300:
            return RegimeMode.FORCE_EXIT

        # ── hard block: spread ──
        if features.spread_ratio is not None:
            if features.spread_ratio > self.config.reduce_only_spread_ratio:
                if RC_SPREAD_DANGER not in reason_codes:
                    reason_codes.append(RC_SPREAD_DANGER)
                return RegimeMode.REDUCE_ONLY
            if features.spread_ratio > self.config.no_new_entry_spread_ratio:
                if RC_SPREAD_TOO_WIDE not in reason_codes:
                    reason_codes.append(RC_SPREAD_TOO_WIDE)
                return RegimeMode.NO_NEW_ENTRY

        # ── hard block: event ──
        if features.minutes_to_high_impact_event is not None:
            if 0 <= features.minutes_to_high_impact_event <= 15:
                if RC_EVENT_NEAR not in reason_codes:
                    reason_codes.append(RC_EVENT_NEAR)
                return RegimeMode.NO_NEW_ENTRY

        # ── hard block: floating PnL velocity ──
        if features.floating_pnl_velocity is not None:
            if features.floating_pnl_velocity < -30:
                if RC_PNL_DETERIORATING not in reason_codes:
                    reason_codes.append(RC_PNL_DETERIORATING)
                return RegimeMode.REDUCE_ONLY

        # ── score-based ──
        if cb_run_score >= self.config.normal_score:
            return RegimeMode.NORMAL
        if cb_run_score >= self.config.caution_score:
            return RegimeMode.CAUTION
        if cb_run_score >= self.config.no_new_entry_score:
            return RegimeMode.NO_NEW_ENTRY
        if cb_run_score >= self.config.reduce_only_score:
            return RegimeMode.REDUCE_ONLY

        return RegimeMode.FORCE_EXIT

    def _build_decision(
        self,
        mode: RegimeMode,
        raw_mode: RegimeMode,
        cb_run_score: float,
        sub_scores: Dict[str, float],
        reason_codes: List[str],
        features: RegimeFeatures,
    ) -> RegimeDecision:
        """RegimeMode + スコアから RegimeDecision を組み立てる。"""
        multiplier = self.config.risk_multiplier_map.get(mode.value, 0.0)

        return RegimeDecision(
            mode=mode,
            raw_mode=raw_mode,
            allow_new_entry=mode in (RegimeMode.NORMAL, RegimeMode.CAUTION),
            allow_add_position=mode in (RegimeMode.NORMAL, RegimeMode.CAUTION),
            reduce_only=mode in (RegimeMode.REDUCE_ONLY, RegimeMode.FORCE_EXIT),
            force_exit=mode == RegimeMode.FORCE_EXIT,
            risk_multiplier=multiplier,
            cb_run_score=cb_run_score,
            sub_scores=sub_scores,
            reason_codes=reason_codes,
            features=self._features_to_dict(features),
            missing_fields=features.missing_fields.copy(),
            timestamp=features.timestamp,
            symbol=features.symbol,
        )

    def reset(self) -> None:
        """内部状態をリセット（新規バックテスト開始時など）。"""
        self.persistence_filter.reset(RegimeMode.NORMAL)

    def _disabled_decision(
        self, symbol: str, current_time_ms: Optional[int]
    ) -> RegimeDecision:
        """regime_controller.enabled=False の場合の恒常 NORMAL 判定。"""
        return RegimeDecision(
            mode=RegimeMode.NORMAL,
            raw_mode=RegimeMode.NORMAL,
            allow_new_entry=True,
            allow_add_position=True,
            reduce_only=False,
            force_exit=False,
            risk_multiplier=1.0,
            cb_run_score=100.0,
            sub_scores={},
            reason_codes=["REGIME_CONTROLLER_DISABLED"],
            features={},
            missing_fields=[],
            timestamp="",
            symbol=symbol,
        )

    @staticmethod
    def _features_to_dict(features: RegimeFeatures) -> Dict[str, Any]:
        """RegimeFeatures を flat dict に変換する（ログ用）。"""
        d = {}
        for field_name in [
            "atr", "atr_pct", "realized_volatility", "range_ratio",
            "trend_strength", "adx", "ma_slope", "consecutive_directional_bars",
            "spread", "spread_avg", "spread_ratio",
            "hour", "weekday", "minutes_to_high_impact_event",
            "floating_pnl", "floating_pnl_velocity", "inventory_imbalance",
            "margin_level", "open_position_count", "long_lots", "short_lots", "net_lots",
            "slippage_avg", "order_reject_rate", "fill_rate",
            "rebate_per_lot", "cost_per_lot", "cb_edge_per_lot",
            "existing_layer3_regime",
        ]:
            val = getattr(features, field_name, None)
            if val is not None:
                # enum や float は素直に突っ込める
                d[field_name] = val
        return d
