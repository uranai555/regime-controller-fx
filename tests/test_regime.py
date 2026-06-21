import math
import unittest
from datetime import datetime, timezone

from regime.types import RegimeMode, RegimeFeatures, RegimeDecision
from regime.sub_scores import SubScoreCalculator, ScoreConfig, RC_SPREAD_WIDENING, RC_SPREAD_TOO_WIDE, RC_EXECUTION_DEFAULTED, RC_VOL_HIGH
from regime.persistence_filter import PersistenceFilter
from regime.regime_controller import RegimeController, RegimeControllerConfig
from regime.feature_collector import FeatureCollector


class TestRegimeModeRiskLevel(unittest.TestCase):
    def test_risk_order(self):
        self.assertLess(RegimeMode.NORMAL.risk_level, RegimeMode.CAUTION.risk_level)
        self.assertLess(RegimeMode.CAUTION.risk_level, RegimeMode.NO_NEW_ENTRY.risk_level)
        self.assertLess(RegimeMode.NO_NEW_ENTRY.risk_level, RegimeMode.REDUCE_ONLY.risk_level)
        self.assertLess(RegimeMode.REDUCE_ONLY.risk_level, RegimeMode.FORCE_EXIT.risk_level)


class TestSubScoreCalculator(unittest.TestCase):
    def setUp(self):
        self.scorer = SubScoreCalculator()

    def _features(self, **overrides) -> RegimeFeatures:
        f = RegimeFeatures(symbol="TEST", timestamp="2026-06-21T12:00:00")
        for k, v in overrides.items():
            setattr(f, k, v)
        return f

    def test_volatility_score_defaulted(self):
        f = self._features()
        score, codes = self.scorer._volatility_score(f)
        self.assertEqual(score, 70)
        self.assertIn("VOLATILITY_SCORE_DEFAULTED", codes)

    def test_volatility_score_ideal(self):
        f = self._features(atr_pct=0.001)
        score, codes = self.scorer._volatility_score(f)
        self.assertEqual(score, 100)
        self.assertEqual(codes, [])

    def test_volatility_score_high(self):
        f = self._features(atr_pct=0.004)
        score, codes = self.scorer._volatility_score(f)
        self.assertEqual(score, 40)
        self.assertIn(RC_VOL_HIGH, codes)

    def test_spread_score_defaulted(self):
        f = self._features()
        score, codes = self.scorer._spread_score(f)
        self.assertEqual(score, 70)
        self.assertIn("SPREAD_SCORE_DEFAULTED", codes)

    def test_spread_score_good(self):
        f = self._features(spread_ratio=1.0)
        score, codes = self.scorer._spread_score(f)
        self.assertEqual(score, 100)
        self.assertEqual(codes, [])

    def test_spread_score_widening(self):
        f = self._features(spread_ratio=1.8)
        score, codes = self.scorer._spread_score(f)
        self.assertEqual(score, 50)
        self.assertIn(RC_SPREAD_WIDENING, codes)

    def test_spread_score_too_wide(self):
        f = self._features(spread_ratio=2.5)
        score, codes = self.scorer._spread_score(f)
        self.assertEqual(score, 20)
        self.assertIn(RC_SPREAD_TOO_WIDE, codes)

    def test_event_safety_defaulted(self):
        f = self._features()
        score, codes = self.scorer._event_safety_score(f)
        self.assertEqual(score, 100)

    def test_execution_defaulted(self):
        f = self._features(missing_fields=["execution_stats_not_provided"])
        score, codes = self.scorer._execution_score(f)
        self.assertEqual(score, 80)
        self.assertIn(RC_EXECUTION_DEFAULTED, codes)

    def test_trend_safety_via_layer3(self):
        f = self._features(existing_layer3_regime="VOLATILE")
        score, codes = self.scorer._trend_safety_score(f)
        self.assertEqual(score, 30)
        self.assertIn("LAYER3_VOLATILE", codes)

    def test_inventory_imbalance(self):
        f = self._features(open_position_count=5, inventory_imbalance=0.7)
        score, codes = self.scorer._inventory_score(f)
        self.assertLess(score, 100)
        self.assertIn("INVENTORY_IMBALANCED", codes)

    def test_inventory_severe(self):
        f = self._features(open_position_count=5, inventory_imbalance=0.9)
        score, codes = self.scorer._inventory_score(f)
        self.assertLess(score, 60)
        self.assertIn("INVENTORY_SEVERELY_IMBALANCED", codes)


class TestPersistenceFilter(unittest.TestCase):
    def test_danger_immediate(self):
        pf = PersistenceFilter(confirm_bars=3)
        pf.update(RegimeMode.NORMAL)
        result = pf.update(RegimeMode.FORCE_EXIT)
        self.assertEqual(result, RegimeMode.FORCE_EXIT)

    def test_safe_needs_three_bars(self):
        pf = PersistenceFilter(confirm_bars=3)
        pf.update(RegimeMode.FORCE_EXIT)
        # 1回目
        r1 = pf.update(RegimeMode.NORMAL)
        self.assertEqual(r1, RegimeMode.FORCE_EXIT)  # まだ戻らない
        # 2回目
        r2 = pf.update(RegimeMode.NORMAL)
        self.assertEqual(r2, RegimeMode.FORCE_EXIT)
        # 3回目
        r3 = pf.update(RegimeMode.NORMAL)
        self.assertEqual(r3, RegimeMode.NORMAL)  # やっと戻る

    def test_same_mode_resets_pending(self):
        pf = PersistenceFilter(confirm_bars=3)
        pf.update(RegimeMode.NORMAL)
        # 1回安全方向試す
        pf.update(RegimeMode.NORMAL)  # 同レベル → pending キャンセル
        # pending_count が 0 になっているはず
        self.assertEqual(pf._state.pending_count, 0)

    def test_reset(self):
        pf = PersistenceFilter(confirm_bars=3)
        pf.update(RegimeMode.FORCE_EXIT)
        pf.reset(RegimeMode.NORMAL)
        self.assertEqual(pf.current_mode, RegimeMode.NORMAL)


class TestRegimeController(unittest.TestCase):
    def setUp(self):
        self.rc = RegimeController()

    def test_disabled_returns_normal(self):
        rc_disabled = RegimeController(config=RegimeControllerConfig(enabled=False))
        decision = rc_disabled.evaluate()
        self.assertEqual(decision.mode, RegimeMode.NORMAL)
        self.assertEqual(decision.risk_multiplier, 1.0)
        self.assertIn("REGIME_CONTROLLER_DISABLED", decision.reason_codes)

    def test_empty_data_returns_normal(self):
        decision = self.rc.evaluate(symbol="TEST", current_time_ms=1624277100000)
        self.assertEqual(decision.mode, RegimeMode.NORMAL)
        self.assertTrue(decision.allow_new_entry)

    def test_spread_wide_blocks_entry(self):
        # 20本のバーで平均0.5、最新が3.0 → spread_ratio≈6.0 → REDUCE_ONLY
        bars = [{"open": 100, "high": 101, "low": 99, "close": 100, "spread_avg": 0.5} for _ in range(19)]
        bars.append({"open": 100, "high": 101, "low": 99, "close": 100, "spread_avg": 3.0})
        decision = self.rc.evaluate(
            symbol="TEST",
            bars=bars,
            current_time_ms=1624277100000,
        )
        self.assertFalse(decision.allow_new_entry)
        self.assertIn(decision.mode, (RegimeMode.NO_NEW_ENTRY, RegimeMode.REDUCE_ONLY))

    def test_good_spread_normal(self):
        decision = self.rc.evaluate(
            symbol="TEST",
            bars=[
                {"open": 100, "high": 101, "low": 99, "close": 100, "spread_avg": 0.5},
                {"open": 100, "high": 101, "low": 99, "close": 100, "spread_avg": 0.5},
            ],
            current_time_ms=1624277100000,
        )
        self.assertEqual(decision.mode, RegimeMode.NORMAL)
        self.assertTrue(decision.allow_new_entry)


class TestFeatureCollector(unittest.TestCase):
    def setUp(self):
        self.collector = FeatureCollector()

    def test_empty_bars(self):
        f = self.collector.collect(symbol="TEST", current_time_ms=1624277100000)
        self.assertIn("price_vol_insufficient_bars", f.missing_fields)
        self.assertIn("spread_no_bars", f.missing_fields)
        self.assertIsNone(f.atr)

    def test_simple_bars(self):
        bars = []
        base = 100.0
        for i in range(21):
            bars.append({"open": base, "high": base + 0.5, "low": base - 0.5, "close": base + 0.1, "spread_avg": 0.2})
            base += 0.1
        f = self.collector.collect(symbol="TEST", bars=bars, current_time_ms=1624277100000)
        self.assertIsNotNone(f.atr)
        self.assertIsNotNone(f.realized_volatility)
        self.assertIsNotNone(f.spread_ratio)
        self.assertAlmostEqual(f.spread_ratio, 1.0, places=1)

    def test_positions(self):
        positions = [
            {"direction": "BUY", "volume": 1.0, "is_open": True, "unrealized_pnl_pips": 5.0},
            {"direction": "BUY", "volume": 1.0, "is_open": True, "unrealized_pnl_pips": -2.0},
            {"direction": "SELL", "volume": 1.0, "is_open": True, "unrealized_pnl_pips": 3.0},
        ]
        f = self.collector.collect(symbol="TEST", positions=positions, current_time_ms=1624277100000)
        self.assertEqual(f.open_position_count, 3)
        self.assertEqual(f.long_lots, 2.0)
        self.assertEqual(f.short_lots, 1.0)
        self.assertEqual(f.net_lots, 1.0)
        self.assertAlmostEqual(f.inventory_imbalance, 1.0 / 3.0)
        self.assertEqual(f.floating_pnl, 6.0)

    def test_execution_stats(self):
        stats = {"slippage_avg": 0.3, "order_reject_rate": 0.01}
        f = self.collector.collect(symbol="TEST", execution_stats=stats, current_time_ms=1624277100000)
        self.assertEqual(f.slippage_avg, 0.3)
        self.assertEqual(f.order_reject_rate, 0.01)
        self.assertNotIn("execution_stats_not_provided", f.missing_fields)

    def test_cb_config(self):
        cb = {"rebate_per_lot": 10.0, "cost_per_lot": 4.0}
        f = self.collector.collect(symbol="TEST", cb_config=cb, current_time_ms=1624277100000)
        self.assertEqual(f.rebate_per_lot, 10.0)
        self.assertEqual(f.cost_per_lot, 4.0)
        self.assertEqual(f.cb_edge_per_lot, 6.0)


if __name__ == "__main__":
    unittest.main()