"""regime_controller.py — RegimeController オーケストレーター

Phase 1: FeatureCollector → SubScoreCalculator → raw_mode → PersistenceFilter
Phase 2: HMMModel による状態推定をパイプラインに差し込み
Phase 3: ExecutionQualityModel によるブローカー別品質評価

出力は BUY/SELL ではなく戦略許可/禁止。
"""

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from regime.execution_quality_model import ExecutionQualityConfig, ExecutionQualityModel
from regime.feature_collector import FeatureCollector
from regime.hmm_model import HMMConfig, HMMModel
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
        "volatility_score": 0.18,
        "trend_safety_score": 0.18,
        "spread_score": 0.18,
        "event_safety_score": 0.13,
        "inventory_score": 0.13,
        "execution_score": 0.10,
        "cb_efficiency_score": 0.10,
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

    Phase 1: ルールベース cb_run_score + PersistenceFilter
    Phase 2: HMMModel が学習済みなら raw_mode を上書き
    Phase 3: ExecutionQualityModel でブローカー品質を execution_score に反映
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
        self.hmm_model = hmm_model
        self.execution_quality_model = execution_quality_model

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
        """FeatureCollector → SubScoreCalculator → raw_mode → PersistenceFilter を実行する。

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

        # 1b. Phase 3: ブローカー別品質を特徴量に反映
        if self.execution_quality_model is not None and broker_id is not None:
            self.execution_quality_model.enrich_features(broker_id, features)

        # 2. サブスコア計算
        sub_scores, reason_codes = self.scorer.calculate(features)

        # 2b. Phase 3: broker_execution_score で execution_score を上書き
        if features.broker_execution_score is not None:
            sub_scores["execution_score"] = features.broker_execution_score
            reason_codes.append("EXECUTION_SCORE_FROM_BROKER_MODEL")

        cb_run_score = self.scorer.aggregate(sub_scores, self.config.score_weights)
        logger.debug(
            "[%s] sub_scores=%s cb_run_score=%.2f missing=%s",
            symbol, sub_scores, cb_run_score, features.missing_fields,
        )

        # 3. raw_mode 判定
        raw_mode = self._decide_raw_mode(features, sub_scores, cb_run_score, reason_codes)

        # 3b. Phase 2: HMM が学習済みなら raw_mode を上書き
        raw_mode = self._apply_hmm_override(features, raw_mode, reason_codes)

        # 4. Persistence filter
        confirmed_mode = self.persistence_filter.update(raw_mode)
        if confirmed_mode != raw_mode:
            logger.info(
                "[%s] PersistenceFilter: raw=%s → confirmed=%s",
                symbol, raw_mode.value, confirmed_mode.value,
            )

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

    def _apply_hmm_override(
        self,
        features: RegimeFeatures,
        raw_mode: RegimeMode,
        reason_codes: List[str],
    ) -> RegimeMode:
        """Phase 2: HMM が学習済みなら raw_mode を上書きする。

        hard block (FORCE_EXIT, REDUCE_ONLY by margin/spread) は上書きしない。
        HMM が未学習 or 入力不足なら raw_mode をそのまま返す（Phase 1 フォールバック）。
        """
        if self.hmm_model is None or not self.hmm_model.is_fitted:
            return raw_mode

        # hard block で決まった raw_mode は上書きしない
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

        # HMM state/proba を features に書き込む
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

        # align lengths (shortest wins)
        n = min(len(ret), len(vol), len(spr))
        if n < 5:
            return None

        # take tail
        ret = ret[-n:]
        vol = vol[-n:]
        spr = spr[-n:]

        return [[ret[i], vol[i], spr[i]] for i in range(n)]

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

    def train_hmm(
        self,
        bars: List[Dict[str, Any]],
        config: Optional[HMMConfig] = None,
    ) -> bool:
        """bars から HMM を学習し、self.hmm_model に設定する。

        FeatureCollector._collect_time_series と同じロジックで
        bars → returns/volatility/spread の T×3 行列を構築し、HMM を fit する。

        Args:
            bars: 学習用ヒストリカル bars (dict リスト)
            config: HMMConfig (省略時はデフォルト)

        Returns:
            True if converged
        """
        import math as _math

        if len(bars) < 2:
            logger.warning("train_hmm: insufficient bars (%d)", len(bars))
            return False

        # returns
        returns: List[float] = []
        for i in range(1, len(bars)):
            prev_c = bars[i - 1].get("close", 1.0)
            curr_c = bars[i].get("close", 1.0)
            if prev_c > 0:
                returns.append(_math.log(curr_c / prev_c))

        # rolling volatility (5-bar)
        vol_series: List[float] = []
        if len(returns) >= 5:
            for i in range(4, len(returns)):
                window = returns[i - 4 : i + 1]
                mean_r = sum(window) / len(window)
                var = sum((r - mean_r) ** 2 for r in window) / len(window)
                vol_series.append(_math.sqrt(var))

        # spread
        spread_vals = [
            b.get("spread_avg", 0.0) for b in bars
            if b.get("spread_avg") is not None
        ]

        # align
        n = min(len(returns), len(vol_series), len(spread_vals))
        if n < 5:
            logger.warning("train_hmm: aligned series too short (%d)", n)
            return False

        obs = [
            [returns[-n + i], vol_series[-n + i], spread_vals[-n + i]]
            for i in range(n)
        ]

        cfg = config or HMMConfig()
        self.hmm_model = HMMModel(config=cfg)
        result = self.hmm_model.fit(obs)
        logger.info("train_hmm: fitted=%s, n_obs=%d", self.hmm_model.is_fitted, n)
        return result

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
        """RegimeFeatures を flat dict に変換する（ログ用）。

        時系列リスト (returns_series 等) はログ肥大化を防ぐため
        長さとサマリ統計 (mean, std, last) に圧縮する。
        """
        _SKIP = {"symbol", "timestamp", "missing_fields"}
        _SERIES_FIELDS = {
            "returns_series", "volatility_series",
            "spread_series", "volume_series",
        }
        d: Dict[str, Any] = {}
        for f in dataclasses.fields(features):
            if f.name in _SKIP:
                continue
            val = features.__dict__.get(f.name)
            if val is None:
                continue
            if f.name in _SERIES_FIELDS and isinstance(val, list) and len(val) > 0:
                import math as _math
                n = len(val)
                mean = sum(val) / n
                var = sum((v - mean) ** 2 for v in val) / n
                d[f.name] = {
                    "len": n,
                    "mean": round(mean, 6),
                    "std": round(_math.sqrt(var), 6),
                    "last": round(val[-1], 6),
                }
            else:
                d[f.name] = val
        return d
