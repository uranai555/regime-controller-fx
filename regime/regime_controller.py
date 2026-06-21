"""regime_controller.py — RegimeController オーケストレーター

Phase 1: FeatureCollector → SubScoreCalculator → raw_mode → PersistenceFilter
Phase 2: HMMModel による隠れ状態推定 + HMMRegimeClassifier でサブスコア化
Phase 3: ExecutionQualityModel + BrokerQualityModel によるブローカー別品質評価

出力は BUY/SELL ではなく戦略許可/禁止。
"""

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from regime.feature_collector import FeatureCollector
from regime.persistence_filter import PersistenceFilter
from regime.sub_scores import ScoreConfig, SubScoreCalculator
from regime.types import RegimeDecision, RegimeFeatures, RegimeMode
from regime.hmm_model import HMMConfig, HMMModel
from regime.execution_quality_model import ExecutionQualityConfig, ExecutionQualityModel
from regime.hmm_regime import HMMRegimeClassifier, HMMConfig as HMMRegimeConfig, HMMResult
from regime.broker_model import BrokerQualityModel, BrokerModelConfig, ExecutionEvent

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

    # 重み (Phase 2/3 対応: HMM + broker_quality 追加)
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        "volatility_score": 0.15,
        "trend_safety_score": 0.10,
        "spread_score": 0.18,
        "event_safety_score": 0.13,
        "inventory_score": 0.13,
        "execution_score": 0.08,
        "cb_efficiency_score": 0.08,
        "hmm_regime_score": 0.08,
        "broker_quality_score": 0.07,
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

    # Phase 2: HMMRegimeClassifier config
    hmm_regime_config: HMMRegimeConfig = field(default_factory=HMMRegimeConfig)

    # Phase 3: BrokerQualityModel config
    broker_model_config: BrokerModelConfig = field(default_factory=BrokerModelConfig)


class RegimeController:
    """レジーム判定 → 戦略許可/禁止の上位フィルター。

    Phase 1: ルールベース cb_run_score + PersistenceFilter
    Phase 2: HMMModel が学習済みなら raw_mode を上書き +
             HMMRegimeClassifier でサブスコア化
    Phase 3: ExecutionQualityModel でブローカー品質を execution_score に反映 +
             BrokerQualityModel で broker_quality_score サブスコア
    """

    def __init__(
        self,
        config: Optional[RegimeControllerConfig] = None,
        score_config: Optional[ScoreConfig] = None,
        hmm_model: Optional[HMMModel] = None,
        execution_quality_model: Optional[ExecutionQualityModel] = None,
    ):
        self.config = config or RegimeControllerConfig()
        self.feature_collector = FeatureCollector()
        self.scorer = SubScoreCalculator(score_config or ScoreConfig())
        self.persistence_filter = PersistenceFilter(
            confirm_bars=self.config.confirm_bars
        )
        # Phase 2: PR#2 HMMModel (injection-based)
        self.hmm_model = hmm_model
        # Phase 2: HMMRegimeClassifier (config-based)
        self.hmm_classifier = HMMRegimeClassifier(self.config.hmm_regime_config)
        # Phase 3: PR#2 ExecutionQualityModel (injection-based)
        self.execution_quality_model = execution_quality_model
        # Phase 3: BrokerQualityModel (config-based)
        self.broker_model = BrokerQualityModel(self.config.broker_model_config)

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
        minutes_to_high_impact_event: Optional[int] = None,
        broker_id: Optional[str] = None,
        account_id: Optional[str] = None,
        server_name: Optional[str] = None,
    ) -> RegimeDecision:
        """FeatureCollector → [HMM + BrokerModel] → SubScoreCalculator → raw_mode → PersistenceFilter

        config.enabled=False の場合は常に NORMAL を返す（旧挙動との互換性）。
        """
        if not self.config.enabled:
            logger.debug("RegimeController disabled — returning NORMAL")
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
            minutes_to_high_impact_event=minutes_to_high_impact_event,
            broker_id=broker_id,
            account_id=account_id,
            server_name=server_name,
        )

        # 1b. Phase 3 (PR#2): ExecutionQualityModel でブローカー品質を特徴量に反映
        if self.execution_quality_model is not None and broker_id is not None:
            self.execution_quality_model.enrich_features(broker_id, features)

        # 2. Phase 2: HMMRegimeClassifier でサブスコア用 features をセット
        self._apply_hmm_regime(features, bars)

        # 3. Phase 3: BrokerQualityModel でサブスコア用 features をセット
        self._apply_broker_model(features, broker_id)

        # 4. サブスコア計算
        sub_scores, reason_codes = self.scorer.calculate(features)

        # 4b. Phase 3 (PR#2): broker_execution_score で execution_score を上書き
        if features.broker_execution_score is not None:
            sub_scores["execution_score"] = features.broker_execution_score
            reason_codes.append("EXECUTION_SCORE_FROM_BROKER_MODEL")

        cb_run_score = self.scorer.aggregate(sub_scores, self.config.score_weights)
        logger.debug(
            "[%s] sub_scores=%s cb_run_score=%.2f missing=%s",
            symbol, sub_scores, cb_run_score, features.missing_fields,
        )

        # 5. raw_mode 判定
        raw_mode = self._decide_raw_mode(features, sub_scores, cb_run_score, reason_codes)

        # 5b. Phase 2 (PR#2): HMMModel が学習済みなら raw_mode を上書き
        raw_mode = self._apply_hmm_override(features, raw_mode, reason_codes)

        # 6. Persistence filter
        confirmed_mode = self.persistence_filter.update(raw_mode)
        if confirmed_mode != raw_mode:
            logger.info(
                "[%s] PersistenceFilter: raw=%s → confirmed=%s",
                symbol, raw_mode.value, confirmed_mode.value,
            )

        # 7. RegimeDecision に変換
        return self._build_decision(
            mode=confirmed_mode,
            raw_mode=raw_mode,
            cb_run_score=cb_run_score,
            sub_scores=sub_scores,
            reason_codes=reason_codes,
            features=features,
        )

    def fit_hmm(self, bars: List[Dict[str, Any]]) -> bool:
        """HMMRegimeClassifier を学習する。"""
        return self.hmm_classifier.fit(bars)

    def update_broker_profile(self, event: ExecutionEvent) -> None:
        """ブローカープロファイルを更新する。"""
        self.broker_model.update(event)

    # ── Phase 2/3 integration ──

    def _apply_hmm_regime(
        self,
        features: RegimeFeatures,
        bars: Optional[List[Dict[str, Any]]],
    ) -> None:
        """HMMRegimeClassifier の分類結果を features にセットする。"""
        if not self.hmm_classifier.available or bars is None:
            return

        result = self.hmm_classifier.predict(bars)
        if result is not None:
            features.hmm_regime_state = result.state.value
            features.hmm_regime_probabilities = result.probabilities
            features.hmm_log_likelihood = result.log_likelihood

    def _apply_broker_model(
        self,
        features: RegimeFeatures,
        broker_id: Optional[str],
    ) -> None:
        """BrokerQualityModel のスコアを features にセットする。"""
        if not self.broker_model.enabled or broker_id is None:
            return

        features.broker_id = broker_id
        score, _reasons = self.broker_model.score(broker_id, hour=features.hour)
        features.broker_quality_score = score

        profile = self.broker_model.get_profile(broker_id)
        if profile is not None:
            features.broker_slippage_mean = profile.slippage_mean
            features.broker_slippage_p95 = profile.slippage_p95
            features.broker_latency_p95_ms = profile.latency_p95_ms
            features.broker_requote_rate = profile.requote_rate
            features.broker_spread_markup = profile.spread_markup
            if features.hour is not None and features.hour in profile.hourly_quality:
                features.broker_hourly_quality_factor = profile.hourly_quality[features.hour]

    def _apply_hmm_override(
        self,
        features: RegimeFeatures,
        raw_mode: RegimeMode,
        reason_codes: List[str],
    ) -> RegimeMode:
        """Phase 2 (PR#2): HMMModel が学習済みなら raw_mode を上書きする。

        hard block (FORCE_EXIT, REDUCE_ONLY by margin/spread) は上書きしない。
        HMM が未学習 or 入力不足なら raw_mode をそのまま返す（Phase 1 フォールバック）。
        """
        if self.hmm_model is None or not self.hmm_model.is_fitted:
            return raw_mode

        if raw_mode in (RegimeMode.FORCE_EXIT, RegimeMode.REDUCE_ONLY):
            return raw_mode

        obs = self._build_hmm_observations(features)
        if obs is None:
            reason_codes.append("HMM_INPUT_INSUFFICIENT")
            return raw_mode

        hmm_mode_str = self.hmm_model.predict_mode(obs)
        if hmm_mode_str is None:
            return raw_mode

        try:
            hmm_mode = RegimeMode(hmm_mode_str)
        except ValueError:
            logger.warning("HMM returned unknown mode: %s", hmm_mode_str)
            return raw_mode

        state = self.hmm_model.predict(obs)
        proba = self.hmm_model.predict_proba(obs)
        features.hmm_state = state
        features.hmm_state_proba = proba

        reason_codes.append(f"HMM_OVERRIDE_{hmm_mode.value}")
        logger.info("HMM override: %s → %s", raw_mode.value, hmm_mode.value)
        return hmm_mode

    @staticmethod
    def _build_hmm_observations(features: RegimeFeatures) -> Optional[List[List[float]]]:
        """RegimeFeatures の時系列から HMM 入力行列を構築する。"""
        ret = features.returns_series
        vol = features.volatility_series
        spr = features.spread_series

        if ret is None or vol is None or spr is None:
            return None

        n = min(len(ret), len(vol), len(spr))
        if n < 5:
            return None

        ret = ret[-n:]
        vol = vol[-n:]
        spr = spr[-n:]

        return [[ret[i], vol[i], spr[i]] for i in range(n)]

    # ── raw_mode decision ──

    def _decide_raw_mode(
        self,
        features: RegimeFeatures,
        sub_scores: Dict[str, float],
        cb_run_score: float,
        reason_codes: List[str],
    ) -> RegimeMode:
        """cb_run_score + hard block 条件から raw_mode を決定する。"""

        # ── Hard blocks (スコアに関わらず強制) ──
        if features.margin_level is not None:
            if features.margin_level < 300:
                reason_codes.append(RC_MARGIN_DANGER)
                return RegimeMode.FORCE_EXIT

        if features.floating_pnl_velocity is not None:
            if features.floating_pnl_velocity < -30:
                reason_codes.append(RC_PNL_DETERIORATING)
                return RegimeMode.REDUCE_ONLY

        if features.spread_ratio is not None:
            if features.spread_ratio > self.config.reduce_only_spread_ratio:
                reason_codes.append(RC_SPREAD_DANGER)
                return RegimeMode.REDUCE_ONLY
            if features.spread_ratio > self.config.no_new_entry_spread_ratio:
                reason_codes.append(RC_SPREAD_TOO_WIDE)
                return RegimeMode.NO_NEW_ENTRY

        if features.minutes_to_high_impact_event is not None:
            if features.minutes_to_high_impact_event <= 15:
                reason_codes.append(RC_EVENT_NEAR)
                return RegimeMode.NO_NEW_ENTRY

        # ── Score-based thresholds ──
        if cb_run_score >= self.config.normal_score:
            return RegimeMode.NORMAL
        if cb_run_score >= self.config.caution_score:
            return RegimeMode.CAUTION
        if cb_run_score >= self.config.no_new_entry_score:
            return RegimeMode.NO_NEW_ENTRY
        if cb_run_score >= self.config.reduce_only_score:
            return RegimeMode.REDUCE_ONLY
        return RegimeMode.FORCE_EXIT

    # ── decision builder ──

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
        """RegimeFeatures を flat dict に変換する。ログ用。"""
        _SKIP = {"symbol", "timestamp", "missing_fields"}
        d = {}
        for f in dataclasses.fields(features):
            if f.name in _SKIP:
                continue
            val = features.__dict__.get(f.name)
            if val is not None:
                d[f.name] = val
        return d
