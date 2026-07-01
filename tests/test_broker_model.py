"""test_broker_model.py — Phase 3: ブローカー別 Execution Quality Model のテスト"""

import json
import tempfile
import unittest

from regime.broker_model import (
    BrokerModelConfig,
    BrokerProfile,
    BrokerQualityModel,
    ExecutionEvent,
    RC_BROKER_INSUFFICIENT_DATA,
    RC_BROKER_LATENCY_HIGH,
    RC_BROKER_QUALITY_LOW,
    RC_BROKER_REJECT_HIGH,
    RC_BROKER_SLIPPAGE_HIGH,
    RC_BROKER_HOUR_DEGRADED,
)
from regime.sub_scores import SubScoreCalculator, RC_BROKER_DEFAULTED
from regime.types import RegimeFeatures
from regime.regime_controller import RegimeController, RegimeControllerConfig


class TestBrokerProfile(unittest.TestCase):
    """BrokerProfile のシリアライズ/デシリアライズ。"""

    def test_to_dict(self):
        profile = BrokerProfile(
            broker_id="BrokerA",
            slippage_mean=0.3,
            reject_rate=0.02,
            sample_count=50,
        )
        d = profile.to_dict()
        self.assertEqual(d["broker_id"], "BrokerA")
        self.assertEqual(d["slippage_mean"], 0.3)
        self.assertEqual(d["reject_rate"], 0.02)
        self.assertEqual(d["sample_count"], 50)

    def test_from_dict(self):
        data = {
            "broker_id": "BrokerB",
            "slippage_mean": 0.5,
            "reject_rate": 0.03,
            "fill_rate": 0.97,
            "latency_mean_ms": 150.0,
            "hourly_quality": {"14": 0.8, "3": 0.4},
            "sample_count": 100,
        }
        profile = BrokerProfile.from_dict(data)
        self.assertEqual(profile.broker_id, "BrokerB")
        self.assertEqual(profile.slippage_mean, 0.5)
        self.assertEqual(profile.hourly_quality[14], 0.8)
        self.assertEqual(profile.hourly_quality[3], 0.4)

    def test_roundtrip(self):
        profile = BrokerProfile(
            broker_id="X",
            slippage_mean=1.0,
            latency_p95_ms=500.0,
            hourly_quality={10: 0.9, 22: 0.3},
            sample_count=200,
        )
        restored = BrokerProfile.from_dict(profile.to_dict())
        self.assertEqual(restored.broker_id, "X")
        self.assertEqual(restored.slippage_mean, 1.0)
        self.assertEqual(restored.latency_p95_ms, 500.0)
        self.assertEqual(restored.hourly_quality, {10: 0.9, 22: 0.3})


class TestBrokerQualityModelUpdate(unittest.TestCase):
    """BrokerQualityModel の update テスト。"""

    def setUp(self):
        self.model = BrokerQualityModel()

    def test_first_update_creates_profile(self):
        event = ExecutionEvent(broker_id="A", slippage=0.3, latency_ms=100)
        self.model.update(event)
        profile = self.model.get_profile("A")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.sample_count, 1)

    def test_multiple_updates_ewma(self):
        for _ in range(60):
            self.model.update(ExecutionEvent(
                broker_id="A", slippage=0.5, latency_ms=200, filled=True
            ))
        profile = self.model.get_profile("A")
        self.assertEqual(profile.sample_count, 60)
        # EWMA should converge toward 0.5 (alpha=0.05, 60 samples)
        self.assertAlmostEqual(profile.slippage_mean, 0.5, delta=0.15)

    def test_reject_rate_tracking(self):
        # 20 normal fills, then 10 rejects
        for _ in range(20):
            self.model.update(ExecutionEvent(broker_id="B", rejected=False))
        for _ in range(10):
            self.model.update(ExecutionEvent(broker_id="B", rejected=True))
        profile = self.model.get_profile("B")
        # reject_rate should be elevated (EWMA, not exact ratio)
        self.assertGreater(profile.reject_rate, 0.0)

    def test_hourly_quality(self):
        # Good events at hour 10
        for _ in range(20):
            self.model.update(ExecutionEvent(
                broker_id="C", slippage=0.1, latency_ms=50, hour=10
            ))
        # Bad events at hour 3
        for _ in range(20):
            self.model.update(ExecutionEvent(
                broker_id="C", slippage=2.0, latency_ms=2000, rejected=True, hour=3
            ))
        profile = self.model.get_profile("C")
        self.assertGreater(profile.hourly_quality.get(10, 0), profile.hourly_quality.get(3, 1))

    def test_disabled_model_no_update(self):
        config = BrokerModelConfig(enabled=False)
        model = BrokerQualityModel(config)
        model.update(ExecutionEvent(broker_id="X", slippage=1.0))
        self.assertIsNone(model.get_profile("X"))


class TestBrokerQualityModelScore(unittest.TestCase):
    """BrokerQualityModel の score テスト。"""

    def setUp(self):
        self.model = BrokerQualityModel(BrokerModelConfig(min_samples=5))

    def _fill_profile(self, broker_id: str, n: int = 30, **kwargs):
        defaults = {"slippage": 0.2, "latency_ms": 100, "filled": True}
        defaults.update(kwargs)
        for _ in range(n):
            self.model.update(ExecutionEvent(broker_id=broker_id, **defaults))

    def test_insufficient_data_default(self):
        self.model.update(ExecutionEvent(broker_id="A"))  # only 1 sample
        score, reasons = self.model.score("A")
        self.assertEqual(score, 80.0)
        self.assertIn(RC_BROKER_INSUFFICIENT_DATA, reasons)

    def test_unknown_broker_default(self):
        score, reasons = self.model.score("UNKNOWN")
        self.assertEqual(score, 80.0)
        self.assertIn(RC_BROKER_INSUFFICIENT_DATA, reasons)

    def test_good_broker_high_score(self):
        self._fill_profile("Good", slippage=0.1, latency_ms=50)
        score, reasons = self.model.score("Good")
        self.assertGreater(score, 80)

    def test_bad_slippage_penalized(self):
        self._fill_profile("BadSlip", slippage=1.5, latency_ms=50)
        score, reasons = self.model.score("BadSlip")
        self.assertLess(score, 70)
        self.assertIn(RC_BROKER_SLIPPAGE_HIGH, reasons)

    def test_high_latency_penalized(self):
        self._fill_profile("Slow", slippage=0.1, latency_ms=3000)
        score, reasons = self.model.score("Slow")
        self.assertLess(score, 80)
        self.assertIn(RC_BROKER_LATENCY_HIGH, reasons)

    def test_high_reject_rate_penalized(self):
        # Fill with all rejects
        for _ in range(30):
            self.model.update(ExecutionEvent(broker_id="Rejecter", rejected=True))
        score, reasons = self.model.score("Rejecter")
        self.assertLessEqual(score, 60)
        self.assertIn(RC_BROKER_REJECT_HIGH, reasons)

    def test_hourly_degradation(self):
        # Set up bad hour 3
        for _ in range(30):
            self.model.update(ExecutionEvent(
                broker_id="HourBad", slippage=2.0, latency_ms=2000, rejected=True, hour=3
            ))
        score_bad_hour, reasons = self.model.score("HourBad", hour=3)
        self.assertIn(RC_BROKER_HOUR_DEGRADED, reasons)


class TestBrokerProfilePersistence(unittest.TestCase):
    """プロファイルの永続化テスト。"""

    def test_save_and_load(self):
        model = BrokerQualityModel()
        for _ in range(10):
            model.update(ExecutionEvent(broker_id="A", slippage=0.3, latency_ms=100))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        self.assertTrue(model.save_profiles(path))

        # Load into new model
        new_model = BrokerQualityModel()
        self.assertTrue(new_model.load_profiles(path))
        profile = new_model.get_profile("A")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.sample_count, 10)

    def test_save_no_path(self):
        model = BrokerQualityModel(BrokerModelConfig(profile_path=None))
        self.assertFalse(model.save_profiles())

    def test_load_nonexistent(self):
        model = BrokerQualityModel()
        self.assertFalse(model.load_profiles("/tmp/nonexistent_broker.json"))


class TestBrokerSubScore(unittest.TestCase):
    """Phase 3: broker_quality_score のサブスコア計算テスト。"""

    def setUp(self):
        self.scorer = SubScoreCalculator()

    def _features(self, **overrides) -> RegimeFeatures:
        f = RegimeFeatures(symbol="TEST", timestamp="2026-06-21T12:00:00")
        for k, v in overrides.items():
            setattr(f, k, v)
        return f

    def test_broker_score_defaulted_when_none(self):
        f = self._features()
        score, codes = self.scorer._broker_quality_score(f)
        self.assertEqual(score, 80)
        self.assertIn(RC_BROKER_DEFAULTED, codes)

    def test_broker_score_passthrough(self):
        f = self._features(broker_quality_score=45.0)
        score, codes = self.scorer._broker_quality_score(f)
        self.assertEqual(score, 45.0)
        self.assertEqual(codes, [])

    def test_broker_score_high(self):
        f = self._features(broker_quality_score=95.0)
        score, codes = self.scorer._broker_quality_score(f)
        self.assertEqual(score, 95.0)


class TestBrokerIntegration(unittest.TestCase):
    """Phase 3: ブローカーモデルと RegimeController の統合テスト。"""

    def test_controller_has_broker_model(self):
        rc = RegimeController()
        self.assertIsNotNone(rc.broker_model)
        self.assertIsInstance(rc.broker_model, BrokerQualityModel)

    def test_controller_config_includes_broker(self):
        config = RegimeControllerConfig()
        self.assertIsNotNone(config.broker_model_config)
        self.assertIn("broker_quality_score", config.score_weights)

    def test_broker_score_in_decision(self):
        rc = RegimeController()
        decision = rc.evaluate(symbol="TEST", current_time_ms=1624277100000)
        self.assertIn("broker_quality_score", decision.sub_scores)

    def test_broker_id_passed_to_evaluate(self):
        rc = RegimeController()
        # Populate broker data
        for _ in range(30):
            rc.update_broker_profile(ExecutionEvent(
                broker_id="TestBroker", slippage=0.2, latency_ms=80
            ))
        decision = rc.evaluate(
            symbol="TEST",
            broker_id="TestBroker",
            current_time_ms=1624277100000,
        )
        self.assertIn("broker_quality_score", decision.sub_scores)
        # Broker score should not be defaulted
        self.assertIn("broker_id", decision.features)

    def test_update_broker_profile(self):
        rc = RegimeController()
        rc.update_broker_profile(ExecutionEvent(broker_id="X", slippage=0.5))
        profile = rc.broker_model.get_profile("X")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.sample_count, 1)


class TestBrokerEWMA(unittest.TestCase):
    """EWMA 計算の単体テスト。"""

    def test_ewma_converges(self):
        result = BrokerQualityModel._ewma(0.0, 1.0, 0.1)
        self.assertAlmostEqual(result, 0.1)

    def test_ewma_steady_state(self):
        val = 0.0
        for _ in range(100):
            val = BrokerQualityModel._ewma(val, 1.0, 0.05)
        self.assertAlmostEqual(val, 1.0, delta=0.01)

    def test_percentile(self):
        values = list(range(100))
        p95 = BrokerQualityModel._percentile(values, 0.95)
        self.assertEqual(p95, 95)

    def test_percentile_empty(self):
        self.assertEqual(BrokerQualityModel._percentile([], 0.95), 0.0)


if __name__ == "__main__":
    unittest.main()
