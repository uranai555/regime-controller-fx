"""feature_collector.py — 外部データから RegimeFeatures を構築する

スタンドアロン版。v3 の BarSnapshot / TickSnapshot / Position への
依存はなく、dict ベースの入力を受け付ける。

Phase 1 では欠損を許容。データがない特徴量は None のまま残し、
missing_fields に記録する。sub_scores.py 側で安全寄りのデフォルト値に
フォールバックする。
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from regime.types import RegimeFeatures


class FeatureCollector:
    """bars / ticks / positions / execution_stats から RegimeFeatures に変換する。

    Args:
        atr_period: ATR計算のバー数
        vol_period: 実現ボラ計算のバー数
        spread_period: 平均スプレッド計算のバー数
    """

    def __init__(
        self,
        atr_period: int = 14,
        vol_period: int = 20,
        spread_period: int = 20,
    ):
        self.atr_period = atr_period
        self.vol_period = vol_period
        self.spread_period = spread_period
        self._prev_floating_pnl: Optional[float] = None

    def collect(
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
        minutes_to_high_impact_event: Optional[int] = None,
    ) -> RegimeFeatures:
        """dict ベースの入力から RegimeFeatures を組み立てる。

        bars は以下のキーを持つ dict のリスト:
          open, high, low, close, spread_avg (必須)
          bid_open, bid_high, bid_low, bid_close (推奨)
          ask_open, ask_high, ask_low, ask_close (推奨)

        positions は以下のキーを持つ dict のリスト:
          direction ("BUY"/"SELL"), volume (float),
          is_open (bool), unrealized_pnl_pips (float, optional)
        """
        missing: List[str] = []
        features = RegimeFeatures(
            symbol=symbol,
            timestamp=self._now_iso(current_time_ms),
        )
        bars = bars or []
        positions = positions or []

        # ── Price / Volatility ──
        self._collect_price_vol(bars, features, missing)

        # ── Spread ──
        self._collect_spread(bars, features, missing)

        # ── Time / Event ──
        self._collect_time(features, missing, current_time_ms, minutes_to_high_impact_event)

        # ── Position / Inventory ──
        self._collect_inventory(positions, features, missing)

        # ── Account State (margin) ──
        self._collect_account(account_state, features, missing)

        # ── Execution Quality ──
        self._collect_execution(execution_stats, features, missing)

        # ── CB / Cost Efficiency ──
        self._collect_cb(cb_config, features, missing)

        # ── Layer 3 ──
        features.existing_layer3_regime = existing_layer3_regime

        features.missing_fields = missing
        return features

    # ── Internal collectors ──

    def _collect_price_vol(
        self,
        bars: List[Dict[str, Any]],
        features: RegimeFeatures,
        missing: List[str],
    ) -> None:
        if len(bars) < 2:
            missing.append("price_vol_insufficient_bars")
            return

        # ATR
        if len(bars) >= self.atr_period + 1:
            true_ranges = []
            for i in range(len(bars) - self.atr_period, len(bars)):
                prev_close = bars[i - 1].get("close", 0.0) if i > 0 else bars[i].get("close", 0.0)
                hi = bars[i].get("high", 0.0)
                lo = bars[i].get("low", 0.0)
                tr = max(
                    hi - lo,
                    abs(hi - prev_close),
                    abs(lo - prev_close),
                )
                true_ranges.append(tr)
            if true_ranges:
                features.atr = sum(true_ranges) / len(true_ranges)
                close = bars[-1].get("close", 1.0)
                features.atr_pct = features.atr / close if close > 0 else 0.0
        else:
            missing.append("atr_insufficient_bars")

        # Realized Volatility
        if len(bars) >= self.vol_period + 1:
            returns = []
            for i in range(len(bars) - self.vol_period, len(bars)):
                prev_close = bars[i - 1].get("close", 1.0)
                curr_close = bars[i].get("close", 1.0)
                if prev_close > 0:
                    ret = math.log(curr_close / prev_close)
                    returns.append(ret)
            if returns:
                mean_ret = sum(returns) / len(returns)
                variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
                features.realized_volatility = math.sqrt(variance)
        else:
            missing.append("realized_vol_insufficient_bars")

        # MA slope
        if len(bars) >= 5:
            closes = [b.get("close", 0.0) for b in bars[-10:]]
            if len(closes) >= 10:
                short_ma = sum(closes[-5:]) / 5
                long_ma = sum(closes) / 10
                features.ma_slope = (short_ma - long_ma) / long_ma if long_ma > 0 else 0.0
                features.trend_strength = abs(features.ma_slope)

    def _collect_spread(
        self,
        bars: List[Dict[str, Any]],
        features: RegimeFeatures,
        missing: List[str],
    ) -> None:
        if not bars:
            missing.append("spread_no_bars")
            return

        features.spread = bars[-1].get("spread_avg")

        recent_spreads = [
            b.get("spread_avg", 0.0) for b in bars[-self.spread_period:]
            if b.get("spread_avg") is not None
        ]
        if recent_spreads:
            features.spread_avg = sum(recent_spreads) / len(recent_spreads)
            if features.spread_avg and features.spread is not None and features.spread_avg > 0:
                features.spread_ratio = features.spread / features.spread_avg
        else:
            missing.append("spread_avg_not_calculable")

    def _collect_time(
        self,
        features: RegimeFeatures,
        missing: List[str],
        current_time_ms: Optional[int],
        minutes_to_high_impact_event: Optional[int] = None,
    ) -> None:
        if current_time_ms is not None:
            try:
                dt = datetime.fromtimestamp(
                    current_time_ms / 1000.0, tz=timezone.utc
                )
                features.hour = dt.hour
                features.weekday = dt.weekday()
            except (ValueError, OSError, OverflowError):
                missing.append("timestamp_parse_failed")
        else:
            missing.append("timestamp_not_provided")

        if minutes_to_high_impact_event is not None:
            features.minutes_to_high_impact_event = minutes_to_high_impact_event
        else:
            missing.append("event_calendar_not_connected")

    def _collect_inventory(
        self,
        positions: List[Dict[str, Any]],
        features: RegimeFeatures,
        missing: List[str],
    ) -> None:
        if not positions:
            features.open_position_count = 0
            features.long_lots = 0.0
            features.short_lots = 0.0
            features.net_lots = 0.0
            features.inventory_imbalance = 0.0
            features.floating_pnl = 0.0
            features.floating_pnl_velocity = 0.0
            return

        open_positions = [p for p in positions if p.get("is_open", True)]

        features.open_position_count = len(open_positions)

        volume_defaulted = any("volume" not in p for p in open_positions)
        if volume_defaulted:
            missing.append("position_volume_defaulted_to_1.0")
            logger.warning("Some positions missing 'volume' key — defaulting to 1.0 lot")

        long_lots = sum(p.get("volume", 1.0) for p in open_positions if p.get("direction") == "BUY")
        short_lots = sum(p.get("volume", 1.0) for p in open_positions if p.get("direction") == "SELL")
        features.long_lots = long_lots
        features.short_lots = short_lots
        features.net_lots = long_lots - short_lots

        total_lots = long_lots + short_lots
        if total_lots > 0:
            features.inventory_imbalance = abs(long_lots - short_lots) / total_lots
        else:
            features.inventory_imbalance = 0.0

        # floating PnL from positions
        total_pnl = sum(p.get("unrealized_pnl_pips", 0.0) for p in open_positions)
        features.floating_pnl = total_pnl

        if self._prev_floating_pnl is not None:
            features.floating_pnl_velocity = total_pnl - self._prev_floating_pnl
        else:
            features.floating_pnl_velocity = 0.0
        self._prev_floating_pnl = total_pnl

    def _collect_account(
        self,
        account_state: Optional[Dict[str, Any]],
        features: RegimeFeatures,
        missing: List[str],
    ) -> None:
        if account_state is None:
            missing.append("account_state_not_provided")
            return

        features.margin_level = account_state.get("margin_level")
        if features.margin_level is None:
            missing.append("margin_level_not_available")

    def _collect_execution(
        self,
        execution_stats: Optional[Dict[str, Any]],
        features: RegimeFeatures,
        missing: List[str],
    ) -> None:
        if execution_stats is None:
            missing.append("execution_stats_not_provided")
            return

        features.slippage_avg = execution_stats.get("slippage_avg")
        features.order_reject_rate = execution_stats.get("order_reject_rate")
        features.fill_rate = execution_stats.get("fill_rate")

        if features.slippage_avg is None:
            missing.append("slippage_not_available")
        if features.order_reject_rate is None:
            missing.append("order_reject_rate_not_available")

    def _collect_cb(
        self,
        cb_config: Optional[Dict[str, Any]],
        features: RegimeFeatures,
        missing: List[str],
    ) -> None:
        if cb_config is None:
            missing.append("cb_config_not_provided")
            return

        features.rebate_per_lot = cb_config.get("rebate_per_lot")
        features.cost_per_lot = cb_config.get("cost_per_lot")

        if features.rebate_per_lot is not None and features.cost_per_lot is not None:
            features.cb_edge_per_lot = features.rebate_per_lot - features.cost_per_lot
        else:
            if features.rebate_per_lot is None:
                missing.append("rebate_per_lot_missing")
            if features.cost_per_lot is None:
                missing.append("cost_per_lot_missing")

    @staticmethod
    def _now_iso(current_time_ms: Optional[int]) -> str:
        if current_time_ms is not None:
            try:
                dt = datetime.fromtimestamp(current_time_ms / 1000.0, tz=timezone.utc)
                return dt.isoformat()
            except (ValueError, OSError, OverflowError):
                pass
        return datetime.now(timezone.utc).isoformat()