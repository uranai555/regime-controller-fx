"""test_hmm_regime.py — Phase 2: HMM レジーム分類のテスト

hmmlearn/numpy がインストールされていない環境でも安全にテストできる。
"""

import math
import tempfile
import unittest
from unittest.mock import patch

from regime.hmm_regime import (
    HMMConfig,
    HMMRegimeClassifier,
    HMMRegimeState,
    HMMResult,
    _HMM_AVAILABLE,
)
from regime.sub_scores import SubScoreCalculator, ScoreConfig, RC_HMM_DEFAULTED, RC_HMM_HIGH_VOL, RC_HMM_TRENDING
from regime.types import RegimeFeatures
from regime.regime_controller import RegimeController, RegimeControllerConfig


def _generate_bars(n: int, base_price: float = 100.0, volatility: float = 0.01):
    """テスト用のバーデータを生成する。

    3つのレジーム（低ボラ、トレンド、高ボラ）を混合して
    HMM が分離可能なデータを作る。
    """
    import random
    random.seed(42)
    bars = []
    price = base_price
    for i in range(n):
        # レジーム切替: 低ボラ → トレンド → 高ボラ のサイクル
        phase = (i * 3) // n
        if phase == 0:
            vol = volatility * 0.3   # 低ボラ
            drift = 0.0
        elif phase == 1:
            vol = volatility * 0.8   # トレンド
            drift = 0.002
        else:
            vol = volatility * 2.5   # 高ボラ
            drift = -0.001

        change = random.gauss(drift, vol)
        price *= (1 + change)
        high = price * (1 + abs(random.gauss(0, vol * 0.7)))
        low = price * (1 - abs(random.gauss(0, vol * 0.7)))
        bars.append({
            "open": price * (1 + random.gauss(0, 0.001)),
            "high": high,
            "low": low,
            "close": price,
            "spread_avg": 0.3,
        })
    return bars


class TestHMMRegimeClassifierAvailability(unittest.TestCase):
    """HMM の利用可能性テスト。"""

    def test_available_reflects_import(self):
        classifier = HMMRegimeClassifier()
        self.assertEqual(classifier.available, _HMM_AVAILABLE)

    def test_disabled_config(self):
        config = HMMConfig(enabled=False)
        classifier = HMMRegimeClassifier(config)
        self.assertFalse(classifier.available)

    def test_predict_returns_none_when_not_fitted(self):
        classifier = HMMRegimeClassifier()
        bars = _generate_bars(100)
        result = classifier.predict(bars)
        self.assertIsNone(result)

    def test_predict_returns_none_when_unavailable(self):
        config = HMMConfig(enabled=False)
        classifier = HMMRegimeClassifier(config)
        bars = _generate_bars(100)
        result = classifier.predict(bars)
        self.assertIsNone(result)


class TestHMMRegimeClassifierConfig(unittest.TestCase):
    """HMMConfig のデフォルト値テスト。"""

    def test_default_config(self):
        config = HMMConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.n_states, 3)
        self.assertEqual(config.feature_window, 60)
        self.assertEqual(config.min_bars_for_fit, 200)
        self.assertEqual(config.refit_interval, 500)

    def test_custom_config(self):
        config = HMMConfig(n_states=4, feature_window=30, min_bars_for_fit=100)
        self.assertEqual(config.n_states, 4)
        self.assertEqual(config.feature_window, 30)
        self.assertEqual(config.min_bars_for_fit, 100)


class TestHMMRegimeState(unittest.TestCase):
    """HMMRegimeState enum テスト。"""

    def test_values(self):
        self.assertEqual(HMMRegimeState.LOW_VOL.value, "LOW_VOL")
        self.assertEqual(HMMRegimeState.TRENDING.value, "TRENDING")
        self.assertEqual(HMMRegimeState.HIGH_VOL.value, "HIGH_VOL")
        self.assertEqual(HMMRegimeState.UNKNOWN.value, "UNKNOWN")


@unittest.skipUnless(_HMM_AVAILABLE, "hmmlearn/numpy not installed")
class TestHMMRegimeClassifierFit(unittest.TestCase):
    """HMM の fit/predict テスト（hmmlearn 利用可能時のみ）。"""

    def test_fit_insufficient_bars(self):
        classifier = HMMRegimeClassifier()
        bars = _generate_bars(50)  # min_bars_for_fit=200 より少ない
        result = classifier.fit(bars)
        self.assertFalse(result)
        self.assertFalse(classifier.is_fitted)

    def test_fit_sufficient_bars(self):
        classifier = HMMRegimeClassifier()
        bars = _generate_bars(250)
        result = classifier.fit(bars)
        self.assertTrue(result)
        self.assertTrue(classifier.is_fitted)

    def test_predict_after_fit(self):
        classifier = HMMRegimeClassifier()
        bars = _generate_bars(250)
        classifier.fit(bars)
        result = classifier.predict(bars[-80:])
        self.assertIsNotNone(result)
        self.assertIsInstance(result, HMMResult)
        self.assertIn(result.state, list(HMMRegimeState))
        self.assertIn(result.state_index, [0, 1, 2])
        self.assertGreater(len(result.probabilities), 0)

    def test_predict_insufficient_window(self):
        config = HMMConfig(feature_window=60)
        classifier = HMMRegimeClassifier(config)
        bars = _generate_bars(250)
        classifier.fit(bars)
        # 10本だけ渡す（feature_window=60 未満）
        result = classifier.predict(bars[-10:])
        self.assertIsNone(result)

    def test_needs_refit(self):
        config = HMMConfig(refit_interval=5)
        classifier = HMMRegimeClassifier(config)
        bars = _generate_bars(250)
        classifier.fit(bars)
        self.assertFalse(classifier.needs_refit())
        # predict を 5 回呼ぶ
        for _ in range(5):
            classifier.predict(bars[-80:])
        self.assertTrue(classifier.needs_refit())

    def test_state_mapping_assigns_labels(self):
        classifier = HMMRegimeClassifier()
        bars = _generate_bars(250)
        classifier.fit(bars)
        # 全ての状態がマッピングされている
        self.assertEqual(len(classifier._state_mapping), 3)
        mapped_states = set(classifier._state_mapping.values())
        self.assertIn(HMMRegimeState.LOW_VOL, mapped_states)
        self.assertIn(HMMRegimeState.TRENDING, mapped_states)
        self.assertIn(HMMRegimeState.HIGH_VOL, mapped_states)


@unittest.skipUnless(_HMM_AVAILABLE, "hmmlearn/numpy not installed")
class TestHMMParamsPersistence(unittest.TestCase):
    """HMM パラメータ永続化テスト。"""

    def test_save_and_load(self):
        classifier = HMMRegimeClassifier()
        bars = _generate_bars(250)
        classifier.fit(bars)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        # Save
        self.assertTrue(classifier.save_params(path))

        # Load into new classifier
        new_classifier = HMMRegimeClassifier()
        self.assertFalse(new_classifier.is_fitted)
        self.assertTrue(new_classifier.load_params(path))
        self.assertTrue(new_classifier.is_fitted)

        # Predict should work
        result = new_classifier.predict(bars[-80:])
        self.assertIsNotNone(result)

    def test_save_not_fitted(self):
        classifier = HMMRegimeClassifier()
        self.assertFalse(classifier.save_params("/tmp/notfitted.json"))

    def test_load_nonexistent_file(self):
        classifier = HMMRegimeClassifier()
        self.assertFalse(classifier.load_params("/tmp/nonexistent_hmm_params.json"))


class TestHMMSubScore(unittest.TestCase):
    """Phase 2: hmm_regime_score の計算テスト。"""

    def setUp(self):
        self.scorer = SubScoreCalculator()

    def _features(self, **overrides) -> RegimeFeatures:
        f = RegimeFeatures(symbol="TEST", timestamp="2026-06-21T12:00:00")
        for k, v in overrides.items():
            setattr(f, k, v)
        return f

    def test_hmm_score_defaulted_when_none(self):
        f = self._features()
        score, codes = self.scorer._hmm_regime_score(f)
        self.assertEqual(score, 70)
        self.assertIn(RC_HMM_DEFAULTED, codes)

    def test_hmm_score_low_vol(self):
        f = self._features(hmm_regime_state="LOW_VOL")
        score, codes = self.scorer._hmm_regime_score(f)
        self.assertEqual(score, 100.0)
        self.assertEqual(codes, [])

    def test_hmm_score_trending(self):
        f = self._features(hmm_regime_state="TRENDING")
        score, codes = self.scorer._hmm_regime_score(f)
        self.assertEqual(score, 55.0)
        self.assertIn(RC_HMM_TRENDING, codes)

    def test_hmm_score_high_vol(self):
        f = self._features(hmm_regime_state="HIGH_VOL")
        score, codes = self.scorer._hmm_regime_score(f)
        self.assertEqual(score, 15.0)
        self.assertIn(RC_HMM_HIGH_VOL, codes)

    def test_hmm_score_unknown(self):
        f = self._features(hmm_regime_state="UNKNOWN")
        score, codes = self.scorer._hmm_regime_score(f)
        self.assertEqual(score, 70)
        self.assertIn(RC_HMM_DEFAULTED, codes)


class TestHMMIntegration(unittest.TestCase):
    """Phase 2: HMM と RegimeController の統合テスト。"""

    def test_controller_has_hmm_classifier(self):
        rc = RegimeController()
        self.assertIsNotNone(rc.hmm_classifier)
        self.assertIsInstance(rc.hmm_classifier, HMMRegimeClassifier)

    def test_controller_config_includes_hmm(self):
        config = RegimeControllerConfig()
        self.assertIsNotNone(config.hmm_regime_config)
        self.assertIn("hmm_regime_score", config.score_weights)

    def test_hmm_score_in_decision_sub_scores(self):
        rc = RegimeController()
        decision = rc.evaluate(symbol="TEST", current_time_ms=1624277100000)
        self.assertIn("hmm_regime_score", decision.sub_scores)

    @unittest.skipUnless(_HMM_AVAILABLE, "hmmlearn/numpy not installed")
    def test_fit_hmm_and_evaluate(self):
        rc = RegimeController()
        bars = _generate_bars(250)
        # Fit HMM
        success = rc.fit_hmm(bars)
        self.assertTrue(success)
        # Evaluate with enough bars for prediction
        decision = rc.evaluate(
            symbol="TEST",
            bars=bars[-80:],
            current_time_ms=1624277100000,
        )
        # HMM should have contributed
        self.assertIn("hmm_regime_score", decision.sub_scores)
        # If HMM was applied, hmm_regime_state should be in features
        if rc.hmm_classifier.is_fitted:
            self.assertIn("hmm_regime_state", decision.features)


if __name__ == "__main__":
    unittest.main()
