import math
import unittest
from datetime import datetime, timezone
from pathlib import Path

from regime.types import RegimeMode, RegimeFeatures, RegimeDecision
from regime.sub_scores import SubScoreCalculator, ScoreConfig, RC_SPREAD_WIDENING, RC_SPREAD_TOO_WIDE, RC_EXECUTION_DEFAULTED, RC_VOL_HIGH
from regime.persistence_filter import PersistenceFilter
from regime.regime_controller import RegimeController, RegimeControllerConfig
from regime.feature_collector import FeatureCollector
from regime.hmm_model import HMMModel, HMMConfig, GaussianHMM
from regime.execution_quality_model import ExecutionQualityModel, ExecutionQualityConfig


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


class TestVolatilityScoreGapFix(unittest.TestCase):
    """Bug fix: atr_pct between very_low and normal_low was falling through to EXTREME."""

    def setUp(self):
        self.scorer = SubScoreCalculator()

    def _features(self, **overrides) -> RegimeFeatures:
        f = RegimeFeatures(symbol="TEST", timestamp="2026-06-21T12:00:00")
        for k, v in overrides.items():
            setattr(f, k, v)
        return f

    def test_atr_pct_between_very_low_and_normal_low(self):
        f = self._features(atr_pct=0.0002)
        score, codes = self.scorer._volatility_score(f)
        self.assertEqual(score, 80.0)
        self.assertEqual(codes, [])

    def test_atr_pct_at_very_low_boundary(self):
        f = self._features(atr_pct=0.0001)
        score, codes = self.scorer._volatility_score(f)
        self.assertEqual(score, 80.0)
        self.assertEqual(codes, [])

    def test_atr_pct_just_below_very_low(self):
        f = self._features(atr_pct=0.00005)
        score, codes = self.scorer._volatility_score(f)
        self.assertEqual(score, 60.0)
        self.assertIn("VOLATILITY_TOO_LOW", codes)


class TestCbEfficiencyScoreInAggregation(unittest.TestCase):
    """Bug fix: cb_efficiency_score was calculated but not used in cb_run_score."""

    def test_cb_efficiency_included_in_default_weights(self):
        scorer = SubScoreCalculator()
        sub_scores = {
            "volatility_score": 100.0,
            "trend_safety_score": 100.0,
            "spread_score": 100.0,
            "event_safety_score": 100.0,
            "inventory_score": 100.0,
            "execution_score": 100.0,
            "cb_efficiency_score": 0.0,
        }
        agg = scorer.aggregate(sub_scores)
        self.assertLess(agg, 100.0)

    def test_cb_efficiency_weight_in_controller_config(self):
        config = RegimeControllerConfig()
        self.assertIn("cb_efficiency_score", config.score_weights)


class TestCollectTimeFix(unittest.TestCase):
    """Bug fix: _collect_time None handling and event pass-through."""

    def test_none_timestamp_gives_missing_field(self):
        collector = FeatureCollector()
        f = collector.collect(symbol="TEST", current_time_ms=None)
        self.assertIn("timestamp_not_provided", f.missing_fields)
        self.assertIsNone(f.hour)
        self.assertIsNone(f.weekday)

    def test_valid_timestamp_sets_hour_weekday(self):
        collector = FeatureCollector()
        f = collector.collect(symbol="TEST", current_time_ms=1624277100000)
        self.assertIsNotNone(f.hour)
        self.assertIsNotNone(f.weekday)
        self.assertNotIn("timestamp_not_provided", f.missing_fields)

    def test_minutes_to_event_passthrough(self):
        collector = FeatureCollector()
        f = collector.collect(
            symbol="TEST",
            current_time_ms=1624277100000,
            minutes_to_high_impact_event=10,
        )
        self.assertEqual(f.minutes_to_high_impact_event, 10)
        self.assertNotIn("event_calendar_not_connected", f.missing_fields)

    def test_no_event_marks_missing(self):
        collector = FeatureCollector()
        f = collector.collect(symbol="TEST", current_time_ms=1624277100000)
        self.assertIn("event_calendar_not_connected", f.missing_fields)
        self.assertIsNone(f.minutes_to_high_impact_event)


class TestEventHardBlock(unittest.TestCase):
    """Integration: minutes_to_high_impact_event triggers NO_NEW_ENTRY via hard block."""

    def test_event_near_blocks_entry(self):
        rc = RegimeController()
        decision = rc.evaluate(
            symbol="TEST",
            current_time_ms=1624277100000,
            minutes_to_high_impact_event=5,
        )
        self.assertFalse(decision.allow_new_entry)
        self.assertIn(decision.mode, (RegimeMode.NO_NEW_ENTRY, RegimeMode.REDUCE_ONLY, RegimeMode.FORCE_EXIT))


class TestAccountStateCollector(unittest.TestCase):
    """Bug fix: margin_level had no collection path via account_state."""

    def test_margin_level_from_account_state(self):
        collector = FeatureCollector()
        f = collector.collect(
            symbol="TEST",
            account_state={"margin_level": 250.0},
            current_time_ms=1624277100000,
        )
        self.assertEqual(f.margin_level, 250.0)

    def test_no_account_state_marks_missing(self):
        collector = FeatureCollector()
        f = collector.collect(symbol="TEST", current_time_ms=1624277100000)
        self.assertIn("account_state_not_provided", f.missing_fields)

    def test_margin_level_force_exit_integration(self):
        rc = RegimeController()
        decision = rc.evaluate(
            symbol="TEST",
            account_state={"margin_level": 150.0},
            current_time_ms=1624277100000,
        )
        self.assertEqual(decision.mode, RegimeMode.FORCE_EXIT)
        self.assertTrue(decision.force_exit)


class TestPositionVolumeDefaultWarning(unittest.TestCase):
    """Bug fix: missing volume key silently defaulted to 1.0."""

    def test_missing_volume_tracked_in_missing_fields(self):
        collector = FeatureCollector()
        positions = [
            {"direction": "BUY", "is_open": True, "unrealized_pnl_pips": 0.0},
        ]
        f = collector.collect(symbol="TEST", positions=positions, current_time_ms=1624277100000)
        self.assertIn("position_volume_defaulted_to_1.0", f.missing_fields)
        self.assertEqual(f.long_lots, 1.0)

    def test_with_volume_no_warning(self):
        collector = FeatureCollector()
        positions = [
            {"direction": "BUY", "volume": 0.5, "is_open": True, "unrealized_pnl_pips": 0.0},
        ]
        f = collector.collect(symbol="TEST", positions=positions, current_time_ms=1624277100000)
        self.assertNotIn("position_volume_defaulted_to_1.0", f.missing_fields)
        self.assertEqual(f.long_lots, 0.5)


class TestDebugSummary(unittest.TestCase):
    """RegimeDecision.debug_summary() provides human-readable output."""

    def test_debug_summary_format(self):
        rc = RegimeController()
        decision = rc.evaluate(
            symbol="XAUUSD",
            current_time_ms=1624277100000,
        )
        summary = decision.debug_summary()
        self.assertIn("XAUUSD", summary)
        self.assertIn("mode=", summary)
        self.assertIn("score=", summary)
        self.assertIn("entry=", summary)


class TestFeaturesToDictIntrospection(unittest.TestCase):
    """_features_to_dict uses dataclass fields instead of hardcoded list."""

    def test_all_non_none_fields_included(self):
        rc = RegimeController()
        bars = []
        base = 100.0
        for i in range(21):
            bars.append({
                "open": base, "high": base + 0.5, "low": base - 0.5,
                "close": base + 0.1, "spread_avg": 0.2,
            })
            base += 0.1
        decision = rc.evaluate(
            symbol="TEST",
            bars=bars,
            current_time_ms=1624277100000,
        )
        self.assertIn("atr", decision.features)
        self.assertIn("spread_ratio", decision.features)
        self.assertNotIn("symbol", decision.features)
        self.assertNotIn("timestamp", decision.features)
        self.assertNotIn("missing_fields", decision.features)


# ═══════════════════════════════════════════════════════════════════
# Phase 2: HMM Model Tests
# ═══════════════════════════════════════════════════════════════════


class TestGaussianHMM(unittest.TestCase):
    """GaussianHMM の EM + Viterbi が動作すること。"""

    def _make_synthetic_data(self, n_samples: int = 100) -> list:
        """3状態の合成データを生成する。"""
        import random
        rng = random.Random(42)
        data = []
        for i in range(n_samples):
            if i < n_samples // 3:
                # state 0: low vol
                data.append([rng.gauss(0.0, 0.01), rng.gauss(0.005, 0.002), rng.gauss(0.3, 0.05)])
            elif i < 2 * n_samples // 3:
                # state 1: mid vol
                data.append([rng.gauss(0.0, 0.03), rng.gauss(0.02, 0.005), rng.gauss(0.5, 0.1)])
            else:
                # state 2: high vol
                data.append([rng.gauss(0.0, 0.08), rng.gauss(0.06, 0.01), rng.gauss(1.0, 0.2)])
        return data

    def test_fit_converges(self):
        data = self._make_synthetic_data(120)
        cfg = HMMConfig(n_states=3, n_features=3, max_iter=50, min_obs=30)
        hmm = GaussianHMM(cfg)
        converged = hmm.fit(data)
        self.assertTrue(hmm.is_fitted)

    def test_fit_too_few_samples(self):
        cfg = HMMConfig(n_states=3, n_features=3, min_obs=30)
        hmm = GaussianHMM(cfg)
        result = hmm.fit([[0.0, 0.0, 0.0]] * 10)
        self.assertFalse(result)
        self.assertFalse(hmm.is_fitted)

    def test_predict_returns_states(self):
        data = self._make_synthetic_data(120)
        cfg = HMMConfig(n_states=3, n_features=3, max_iter=50, min_obs=30)
        hmm = GaussianHMM(cfg)
        hmm.fit(data)
        states = hmm.predict(data)
        self.assertEqual(len(states), len(data))
        self.assertTrue(all(0 <= s < 3 for s in states))

    def test_predict_proba_sums_to_one(self):
        data = self._make_synthetic_data(120)
        cfg = HMMConfig(n_states=3, n_features=3, max_iter=50, min_obs=30)
        hmm = GaussianHMM(cfg)
        hmm.fit(data)
        proba = hmm.predict_proba(data)
        self.assertEqual(len(proba), len(data))
        for row in proba:
            self.assertAlmostEqual(sum(row), 1.0, places=5)

    def test_predict_raises_when_not_fitted(self):
        cfg = HMMConfig(n_states=3, n_features=3)
        hmm = GaussianHMM(cfg)
        with self.assertRaises(RuntimeError):
            hmm.predict([[0.0, 0.0, 0.0]])

    def test_serialize_deserialize(self):
        data = self._make_synthetic_data(120)
        cfg = HMMConfig(n_states=3, n_features=3, max_iter=50, min_obs=30)
        hmm = GaussianHMM(cfg)
        hmm.fit(data)
        d = hmm.to_dict()
        hmm2 = GaussianHMM.from_dict(d)
        self.assertTrue(hmm2.is_fitted)
        states1 = hmm.predict(data)
        states2 = hmm2.predict(data)
        self.assertEqual(states1, states2)


class TestHMMModel(unittest.TestCase):
    """HMMModel のラッパー機能テスト。"""

    def _make_synthetic_data(self, n_samples: int = 100) -> list:
        import random
        rng = random.Random(42)
        data = []
        for i in range(n_samples):
            if i < n_samples // 3:
                data.append([rng.gauss(0.0, 0.01), rng.gauss(0.005, 0.002), rng.gauss(0.3, 0.05)])
            elif i < 2 * n_samples // 3:
                data.append([rng.gauss(0.0, 0.03), rng.gauss(0.02, 0.005), rng.gauss(0.5, 0.1)])
            else:
                data.append([rng.gauss(0.0, 0.08), rng.gauss(0.06, 0.01), rng.gauss(1.0, 0.2)])
        return data

    def test_unfitted_returns_none(self):
        model = HMMModel()
        self.assertIsNone(model.predict([[0.0, 0.0, 0.0]]))
        self.assertIsNone(model.predict_proba([[0.0, 0.0, 0.0]]))
        self.assertIsNone(model.predict_mode([[0.0, 0.0, 0.0]]))

    def test_fitted_returns_state(self):
        data = self._make_synthetic_data(120)
        model = HMMModel(config=HMMConfig(n_states=3, n_features=3, max_iter=50, min_obs=30))
        model.fit(data)
        state = model.predict(data)
        self.assertIsNotNone(state)
        self.assertIn(state, [0, 1, 2])

    def test_predict_mode_returns_regime_string(self):
        data = self._make_synthetic_data(120)
        model = HMMModel(config=HMMConfig(n_states=3, n_features=3, max_iter=50, min_obs=30))
        model.fit(data)
        mode = model.predict_mode(data)
        self.assertIn(mode, ["NORMAL", "CAUTION", "NO_NEW_ENTRY"])

    def test_save_and_load(self):
        import tempfile
        import os
        data = self._make_synthetic_data(120)
        model = HMMModel(config=HMMConfig(n_states=3, n_features=3, max_iter=50, min_obs=30))
        model.fit(data)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_hmm.json")
            model.save(path)
            loaded = HMMModel.load(path)
            self.assertIsNotNone(loaded)
            self.assertTrue(loaded.is_fitted)
            self.assertEqual(model.predict(data), loaded.predict(data))

    def test_load_nonexistent_returns_none(self):
        result = HMMModel.load("/nonexistent/path/model.json")
        self.assertIsNone(result)


class TestHMMFallbackInController(unittest.TestCase):
    """Phase 2: HMM が未学習の場合、Phase 1 にフォールバックすること。"""

    def test_no_hmm_fallback(self):
        rc = RegimeController()
        decision = rc.evaluate(symbol="TEST", current_time_ms=1624277100000)
        self.assertEqual(decision.mode, RegimeMode.NORMAL)
        self.assertNotIn("HMM_OVERRIDE_NORMAL", decision.reason_codes)

    def test_unfitted_hmm_fallback(self):
        hmm = HMMModel()
        rc = RegimeController(hmm_model=hmm)
        decision = rc.evaluate(symbol="TEST", current_time_ms=1624277100000)
        self.assertEqual(decision.mode, RegimeMode.NORMAL)
        self.assertNotIn("HMM_OVERRIDE_NORMAL", decision.reason_codes)

    def test_hmm_does_not_override_hard_block(self):
        """hard block (FORCE_EXIT) は HMM が上書きしない。"""
        import random
        rng = random.Random(42)
        data = []
        for i in range(120):
            data.append([rng.gauss(0.0, 0.01), rng.gauss(0.005, 0.002), rng.gauss(0.3, 0.05)])
        hmm = HMMModel(config=HMMConfig(n_states=3, n_features=3, max_iter=50, min_obs=30))
        hmm.fit(data)

        rc = RegimeController(hmm_model=hmm)
        decision = rc.evaluate(
            symbol="TEST",
            account_state={"margin_level": 150.0},
            current_time_ms=1624277100000,
        )
        self.assertEqual(decision.mode, RegimeMode.FORCE_EXIT)


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Execution Quality Model Tests
# ═══════════════════════════════════════════════════════════════════


class TestExecutionQualityModel(unittest.TestCase):
    """ExecutionQualityModel の単体テスト。"""

    def test_no_history_returns_default(self):
        eq = ExecutionQualityModel()
        score, details = eq.score("UNKNOWN_BROKER")
        self.assertEqual(score, 80.0)
        self.assertEqual(details, {})

    def test_none_broker_returns_default(self):
        eq = ExecutionQualityModel()
        score, details = eq.score(None)
        self.assertEqual(score, 80.0)

    def test_insufficient_records_returns_default(self):
        eq = ExecutionQualityModel(config=ExecutionQualityConfig(min_records=10))
        for _ in range(5):
            eq.record(broker_id="XM", slippage=0.1)
        score, details = eq.score("XM")
        self.assertEqual(score, 80.0)

    def test_good_broker_scores_high(self):
        eq = ExecutionQualityModel(config=ExecutionQualityConfig(min_records=5))
        for _ in range(10):
            eq.record(broker_id="XM", slippage=0.1, rejected=False, filled=True)
        score, details = eq.score("XM")
        self.assertGreaterEqual(score, 90.0)
        self.assertIn("slippage_avg", details)

    def test_bad_broker_scores_low(self):
        eq = ExecutionQualityModel(config=ExecutionQualityConfig(min_records=5))
        for _ in range(10):
            eq.record(broker_id="BAD", slippage=1.5, rejected=True, filled=False)
        score, details = eq.score("BAD")
        self.assertLessEqual(score, 30.0)

    def test_multiple_brokers_independent(self):
        eq = ExecutionQualityModel(config=ExecutionQualityConfig(min_records=5))
        for _ in range(10):
            eq.record(broker_id="GOOD", slippage=0.1, rejected=False, filled=True)
            eq.record(broker_id="BAD", slippage=1.5, rejected=True, filled=False)
        score_good, _ = eq.score("GOOD")
        score_bad, _ = eq.score("BAD")
        self.assertGreater(score_good, score_bad)

    def test_history_count(self):
        eq = ExecutionQualityModel()
        for _ in range(5):
            eq.record(broker_id="XM", slippage=0.1)
        self.assertEqual(eq.get_history_count("XM"), 5)
        self.assertEqual(eq.get_history_count("UNKNOWN"), 0)

    def test_list_brokers(self):
        eq = ExecutionQualityModel()
        eq.record(broker_id="XM", slippage=0.1)
        eq.record(broker_id="Titan", slippage=0.2)
        brokers = eq.list_brokers()
        self.assertIn("XM", brokers)
        self.assertIn("Titan", brokers)

    def test_clear(self):
        eq = ExecutionQualityModel()
        eq.record(broker_id="XM", slippage=0.1)
        eq.record(broker_id="Titan", slippage=0.2)
        eq.clear("XM")
        self.assertEqual(eq.get_history_count("XM"), 0)
        self.assertEqual(eq.get_history_count("Titan"), 1)
        eq.clear()
        self.assertEqual(len(eq.list_brokers()), 0)

    def test_max_history_cap(self):
        eq = ExecutionQualityModel(config=ExecutionQualityConfig(max_history=10))
        for i in range(20):
            eq.record(broker_id="XM", slippage=0.1 * i)
        self.assertEqual(eq.get_history_count("XM"), 10)


class TestExecutionQualityInController(unittest.TestCase):
    """Phase 3: ExecutionQualityModel が RegimeController に統合されること。"""

    def test_broker_score_overrides_execution_score(self):
        eq = ExecutionQualityModel(config=ExecutionQualityConfig(min_records=5))
        for _ in range(10):
            eq.record(broker_id="BAD", slippage=1.5, rejected=True, filled=False)
        rc = RegimeController(execution_quality_model=eq)
        decision = rc.evaluate(
            symbol="TEST",
            broker_id="BAD",
            current_time_ms=1624277100000,
        )
        self.assertIn("EXECUTION_SCORE_FROM_BROKER_MODEL", decision.reason_codes)

    def test_no_broker_id_no_override(self):
        eq = ExecutionQualityModel(config=ExecutionQualityConfig(min_records=5))
        for _ in range(10):
            eq.record(broker_id="XM", slippage=0.1)
        rc = RegimeController(execution_quality_model=eq)
        decision = rc.evaluate(
            symbol="TEST",
            current_time_ms=1624277100000,
        )
        self.assertNotIn("EXECUTION_SCORE_FROM_BROKER_MODEL", decision.reason_codes)

    def test_no_eq_model_no_override(self):
        rc = RegimeController()
        decision = rc.evaluate(
            symbol="TEST",
            broker_id="XM",
            current_time_ms=1624277100000,
        )
        self.assertNotIn("EXECUTION_SCORE_FROM_BROKER_MODEL", decision.reason_codes)


# ═══════════════════════════════════════════════════════════════════
# Phase 2/3: Feature Collector Extensions
# ═══════════════════════════════════════════════════════════════════


class TestFeatureCollectorPhase2(unittest.TestCase):
    """Phase 2: 時系列特徴量の収集。"""

    def test_time_series_from_bars(self):
        collector = FeatureCollector()
        bars = []
        base = 100.0
        for i in range(30):
            bars.append({
                "open": base, "high": base + 0.5, "low": base - 0.5,
                "close": base + 0.1 * (i % 3 - 1), "spread_avg": 0.2 + 0.01 * i,
            })
            base += 0.1
        f = collector.collect(symbol="TEST", bars=bars, current_time_ms=1624277100000)
        self.assertIsNotNone(f.returns_series)
        self.assertEqual(len(f.returns_series), 29)
        self.assertIsNotNone(f.volatility_series)
        self.assertGreater(len(f.volatility_series), 0)
        self.assertIsNotNone(f.spread_series)
        self.assertEqual(len(f.spread_series), 30)

    def test_insufficient_bars_no_time_series(self):
        collector = FeatureCollector()
        f = collector.collect(symbol="TEST", bars=[], current_time_ms=1624277100000)
        self.assertIsNone(f.returns_series)
        self.assertIn("time_series_insufficient_bars", f.missing_fields)


class TestFeatureCollectorPhase3(unittest.TestCase):
    """Phase 3: ブローカー/口座識別子の収集。"""

    def test_broker_info_collected(self):
        collector = FeatureCollector()
        f = collector.collect(
            symbol="TEST",
            broker_id="XM",
            account_id="12345",
            server_name="XMTrading-MT5",
            current_time_ms=1624277100000,
        )
        self.assertEqual(f.broker_id, "XM")
        self.assertEqual(f.account_id, "12345")
        self.assertEqual(f.server_name, "XMTrading-MT5")
        self.assertNotIn("broker_id_not_provided", f.missing_fields)

    def test_no_broker_info_marks_missing(self):
        collector = FeatureCollector()
        f = collector.collect(symbol="TEST", current_time_ms=1624277100000)
        self.assertIsNone(f.broker_id)
        self.assertIn("broker_id_not_provided", f.missing_fields)


# ═══════════════════════════════════════════════════════════════════
# Phase 2/3: RegimeFeatures extensions
# ═══════════════════════════════════════════════════════════════════


class TestRegimeFeaturesPhase2Fields(unittest.TestCase):
    """Phase 2/3 の新しいフィールドが RegimeFeatures に存在すること。"""

    def test_phase2_fields_exist(self):
        f = RegimeFeatures(symbol="TEST", timestamp="2026-06-21T12:00:00")
        self.assertIsNone(f.returns_series)
        self.assertIsNone(f.volatility_series)
        self.assertIsNone(f.spread_series)
        self.assertIsNone(f.volume_series)
        self.assertIsNone(f.hmm_state)
        self.assertIsNone(f.hmm_state_proba)

    def test_phase3_fields_exist(self):
        f = RegimeFeatures(symbol="TEST", timestamp="2026-06-21T12:00:00")
        self.assertIsNone(f.broker_id)
        self.assertIsNone(f.account_id)
        self.assertIsNone(f.server_name)
        self.assertIsNone(f.broker_slippage_avg)
        self.assertIsNone(f.broker_reject_rate)
        self.assertIsNone(f.broker_fill_rate)
        self.assertIsNone(f.broker_execution_score)


# ═══════════════════════════════════════════════════════════════════
# Public API import test
# ═══════════════════════════════════════════════════════════════════


class TestPublicAPIImports(unittest.TestCase):
    """__init__.py から新しいクラスがインポート可能なこと。"""

    def test_phase2_imports(self):
        from regime import HMMModel, HMMConfig, GaussianHMM
        self.assertIsNotNone(HMMModel)
        self.assertIsNotNone(HMMConfig)
        self.assertIsNotNone(GaussianHMM)

    def test_phase3_imports(self):
        from regime import ExecutionQualityModel, ExecutionQualityConfig, ExecutionRecord
        self.assertIsNotNone(ExecutionQualityModel)
        self.assertIsNotNone(ExecutionQualityConfig)
        self.assertIsNotNone(ExecutionRecord)


# ═══════════════════════════════════════════════════════════════════
# Improvements: auto-sort, save/load, train_hmm, features_to_dict
# ═══════════════════════════════════════════════════════════════════


class TestHMMAutoSortStates(unittest.TestCase):
    """fit() 後に状態がボラティリティ順にソートされること。"""

    def test_states_sorted_by_volatility(self):
        import random
        rng = random.Random(42)
        data = []
        for i in range(120):
            if i < 40:
                data.append([rng.gauss(0.0, 0.01), rng.gauss(0.005, 0.002), rng.gauss(0.3, 0.05)])
            elif i < 80:
                data.append([rng.gauss(0.0, 0.03), rng.gauss(0.02, 0.005), rng.gauss(0.5, 0.1)])
            else:
                data.append([rng.gauss(0.0, 0.08), rng.gauss(0.06, 0.01), rng.gauss(1.0, 0.2)])

        model = HMMModel(config=HMMConfig(n_states=3, n_features=3, max_iter=50, min_obs=30))
        model.fit(data)
        # After sort: state 0 should have lowest total variance
        variances = model._hmm.variances
        total_var = [sum(v) for v in variances]
        self.assertLessEqual(total_var[0], total_var[1])
        self.assertLessEqual(total_var[1], total_var[2])


class TestExecutionQualitySaveLoad(unittest.TestCase):
    """ExecutionQualityModel の save/load 永続化。"""

    def test_save_and_load(self):
        import tempfile
        import os
        eq = ExecutionQualityModel(config=ExecutionQualityConfig(min_records=5))
        for i in range(10):
            eq.record(broker_id="XM", slippage=0.1 * i, rejected=(i % 5 == 0), filled=True, timestamp_ms=i * 1000)
        eq.record(broker_id="Titan", slippage=0.3, rejected=False, filled=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "eq_history.json")
            eq.save(path)

            loaded = ExecutionQualityModel.load(path, config=ExecutionQualityConfig(min_records=5))
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.get_history_count("XM"), 10)
            self.assertEqual(loaded.get_history_count("Titan"), 1)

            score_orig, _ = eq.score("XM")
            score_loaded, _ = loaded.score("XM")
            self.assertAlmostEqual(score_orig, score_loaded, places=2)

    def test_load_nonexistent_returns_none(self):
        result = ExecutionQualityModel.load("/nonexistent/path.json")
        self.assertIsNone(result)


class TestTrainHMM(unittest.TestCase):
    """RegimeController.train_hmm() ヘルパー。"""

    def _make_bars(self, n: int = 100) -> list:
        import random
        rng = random.Random(42)
        bars = []
        price = 100.0
        for i in range(n):
            change = rng.gauss(0, 0.5)
            close = price + change
            bars.append({
                "open": price,
                "high": max(price, close) + abs(rng.gauss(0, 0.2)),
                "low": min(price, close) - abs(rng.gauss(0, 0.2)),
                "close": close,
                "spread_avg": 0.2 + rng.random() * 0.3,
            })
            price = close
        return bars

    def test_train_creates_fitted_model(self):
        rc = RegimeController()
        self.assertIsNone(rc.hmm_model)
        bars = self._make_bars(100)
        result = rc.train_hmm(bars)
        self.assertIsNotNone(rc.hmm_model)
        self.assertTrue(rc.hmm_model.is_fitted)

    def test_train_insufficient_bars(self):
        rc = RegimeController()
        result = rc.train_hmm([{"close": 100, "spread_avg": 0.2}])
        self.assertFalse(result)

    def test_trained_hmm_used_in_evaluate(self):
        rc = RegimeController()
        bars = self._make_bars(100)
        rc.train_hmm(bars, config=HMMConfig(n_states=3, n_features=3, min_obs=10))

        decision = rc.evaluate(
            symbol="TEST",
            bars=bars[-30:],
            current_time_ms=1624277100000,
        )
        hmm_codes = [c for c in decision.reason_codes if c.startswith("HMM_")]
        self.assertTrue(len(hmm_codes) > 0)


class TestFeaturesToDictSeries(unittest.TestCase):
    """_features_to_dict が時系列をサマリ化すること。"""

    def test_series_summarized(self):
        f = RegimeFeatures(symbol="TEST", timestamp="2026-06-21T12:00:00")
        f.returns_series = [0.01, -0.02, 0.03, -0.01, 0.005]
        f.spread_series = [0.3, 0.4, 0.5]

        d = RegimeController._features_to_dict(f)
        self.assertIn("returns_series", d)
        self.assertIsInstance(d["returns_series"], dict)
        self.assertEqual(d["returns_series"]["len"], 5)
        self.assertIn("mean", d["returns_series"])
        self.assertIn("std", d["returns_series"])
        self.assertIn("last", d["returns_series"])

    def test_scalar_fields_unchanged(self):
        f = RegimeFeatures(symbol="TEST", timestamp="2026-06-21T12:00:00")
        f.atr = 1.5
        f.broker_id = "XM"
        f.hmm_state = 1
        f.hmm_state_proba = [0.2, 0.5, 0.3]

        d = RegimeController._features_to_dict(f)
        self.assertEqual(d["atr"], 1.5)
        self.assertEqual(d["broker_id"], "XM")
        self.assertEqual(d["hmm_state"], 1)
        self.assertEqual(d["hmm_state_proba"], [0.2, 0.5, 0.3])


class TestVaultLoggerPhase23(unittest.TestCase):
    """VaultLogger が Phase 2/3 フィールドを含む decision を書けること。"""

    def test_write_with_phase23_features(self):
        import tempfile
        import os
        from regime.vault_logger import RegimeVaultLogger

        decision = RegimeDecision(
            mode=RegimeMode.CAUTION,
            raw_mode=RegimeMode.NORMAL,
            allow_new_entry=True,
            allow_add_position=True,
            reduce_only=False,
            force_exit=False,
            risk_multiplier=0.5,
            cb_run_score=72.0,
            sub_scores={"execution_score": 85.0},
            reason_codes=["HMM_OVERRIDE_CAUTION", "EXECUTION_SCORE_FROM_BROKER_MODEL"],
            features={
                "hmm_state": 1,
                "hmm_state_proba": [0.2, 0.5, 0.3],
                "broker_id": "XM",
                "broker_execution_score": 85.0,
                "returns_series": {"len": 29, "mean": 0.001, "std": 0.02, "last": -0.003},
            },
            missing_fields=[],
            timestamp="2026-06-21T12:00:00",
            symbol="XAUUSD",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RegimeVaultLogger(vault_root=tmpdir)
            result = logger.write(decision)
            self.assertTrue(result)

            import json
            files = list((Path(tmpdir) / "regime").glob("*.jsonl"))
            self.assertEqual(len(files), 1)
            with files[0].open() as fh:
                record = json.loads(fh.readline())
            self.assertEqual(record["mode"], "CAUTION")
            self.assertIn("HMM_OVERRIDE_CAUTION", record["reason_codes"])
            self.assertEqual(record["features"]["hmm_state"], 1)
            self.assertEqual(record["features"]["broker_id"], "XM")


if __name__ == "__main__":
    unittest.main()