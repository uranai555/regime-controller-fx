"""hmm_regime.py — GaussianHMM によるレジーム分類

Phase 2: 価格・ボラティリティの時系列パターンから隠れ状態を推定し、
trend_safety_score の精度向上に使う。

hmmlearn / numpy が未インストールの場合は graceful degrade:
- HMMRegimeClassifier.available = False
- predict() → None
- RegimeController は Phase 1 のルールベースにフォールバック
"""

import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from hmmlearn.hmm import GaussianHMM

    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False


class HMMRegimeState(str, Enum):
    """HMM が推定する隠れレジーム状態。"""
    LOW_VOL = "LOW_VOL"        # 低ボラ・レンジ相場 → 安全
    TRENDING = "TRENDING"      # 方向性あり → 中程度リスク
    HIGH_VOL = "HIGH_VOL"      # 高ボラ・クラッシュ → 危険
    UNKNOWN = "UNKNOWN"        # 推定不可


@dataclass
class HMMConfig:
    """HMM レジーム分類の設定。"""
    enabled: bool = True
    n_states: int = 3
    feature_window: int = 60       # predict に必要な最小バー数
    min_bars_for_fit: int = 200    # fit に必要な最小バー数
    refit_interval: int = 500      # 再学習までのバー間隔
    covariance_type: str = "diag"  # "diag" が小データセットに安定
    n_iter: int = 100
    random_state: int = 42
    params_path: Optional[str] = None


@dataclass
class HMMResult:
    """HMM 推定結果。"""
    state: HMMRegimeState
    state_index: int               # 0, 1, 2
    probabilities: Dict[str, float]  # state_name → probability
    log_likelihood: float = 0.0
    bars_used: int = 0


class HMMRegimeClassifier:
    """GaussianHMM による市場レジーム分類器。

    特徴量:
      - log_returns: 対数リターン
      - realized_vol: ローリング実現ボラティリティ
      - range_ratio: (high - low) / close

    Usage:
        classifier = HMMRegimeClassifier()
        if classifier.available:
            classifier.fit(bars)
            result = classifier.predict(bars)
    """

    def __init__(self, config: Optional[HMMConfig] = None):
        self.config = config or HMMConfig()
        self._model: Any = None
        self._is_fitted: bool = False
        self._bars_since_fit: int = 0
        self._state_mapping: Dict[int, HMMRegimeState] = {}

    @property
    def available(self) -> bool:
        """hmmlearn/numpy が利用可能かつ config.enabled であるか。"""
        return _HMM_AVAILABLE and self.config.enabled

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, bars: List[Dict[str, Any]]) -> bool:
        """過去データから HMM パラメータを推定する。

        Args:
            bars: OHLC バーの dict リスト (open, high, low, close 必須)

        Returns:
            True if fit succeeded, False otherwise
        """
        if not self.available:
            logger.debug("HMM not available — skipping fit")
            return False

        if len(bars) < self.config.min_bars_for_fit:
            logger.warning(
                "Insufficient bars for HMM fit: %d < %d",
                len(bars), self.config.min_bars_for_fit,
            )
            return False

        features = self._extract_features(bars)
        if features is None or len(features) < self.config.min_bars_for_fit - 1:
            logger.warning("Feature extraction failed or insufficient data")
            return False

        try:
            model = GaussianHMM(
                n_components=self.config.n_states,
                covariance_type=self.config.covariance_type,
                n_iter=self.config.n_iter,
                random_state=self.config.random_state,
            )
            model.fit(features)
            self._model = model
            self._is_fitted = True
            self._bars_since_fit = 0

            # 状態マッピング: ボラティリティ列の平均値でソート
            self._assign_state_mapping(features)

            logger.info(
                "HMM fit complete: %d bars, %d states, score=%.2f",
                len(features), self.config.n_states, model.score(features),
            )
            return True

        except Exception as e:
            logger.error("HMM fit failed: %s", e)
            self._is_fitted = False
            return False

    def predict(self, bars: List[Dict[str, Any]]) -> Optional[HMMResult]:
        """現在のバーデータからレジーム状態を推定する。

        Args:
            bars: 直近のバー (最低 feature_window 本)

        Returns:
            HMMResult or None if prediction unavailable
        """
        if not self.available or not self._is_fitted:
            return None

        if len(bars) < self.config.feature_window:
            return None

        features = self._extract_features(bars[-self.config.feature_window:])
        if features is None or len(features) == 0:
            return None

        try:
            # 最新バーの状態を推定
            state_sequence = self._model.predict(features)
            current_state_idx = int(state_sequence[-1])

            # 状態確率を取得
            posteriors = self._model.predict_proba(features)
            current_probs = posteriors[-1]

            # シーケンス全体の対数尤度
            total_ll = float(self._model.score(features))

            regime_state = self._state_mapping.get(
                current_state_idx, HMMRegimeState.UNKNOWN
            )

            probabilities = {}
            for idx, state in self._state_mapping.items():
                if idx < len(current_probs):
                    probabilities[state.value] = float(current_probs[idx])

            self._bars_since_fit += 1

            return HMMResult(
                state=regime_state,
                state_index=current_state_idx,
                probabilities=probabilities,
                log_likelihood=total_ll,
                bars_used=len(features),
            )

        except Exception as e:
            logger.error("HMM predict failed: %s", e)
            return None

    def needs_refit(self) -> bool:
        """再学習が必要かどうか。"""
        if not self._is_fitted:
            return True
        return self._bars_since_fit >= self.config.refit_interval

    def save_params(self, path: str) -> bool:
        """学習済みパラメータを JSON で永続化する。"""
        if not self._is_fitted or self._model is None:
            return False

        try:
            params = {
                "n_states": self.config.n_states,
                "covariance_type": self.config.covariance_type,
                "means": self._model.means_.tolist(),
                "covars": self._model.covars_.tolist(),
                "transmat": self._model.transmat_.tolist(),
                "startprob": self._model.startprob_.tolist(),
                "state_mapping": {
                    str(k): v.value for k, v in self._state_mapping.items()
                },
            }

            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as f:
                json.dump(params, f, indent=2)

            logger.info("HMM params saved to %s", path)
            return True

        except Exception as e:
            logger.error("HMM save_params failed: %s", e)
            return False

    def load_params(self, path: str) -> bool:
        """永続化されたパラメータを復元する。"""
        if not _HMM_AVAILABLE:
            return False

        try:
            p = Path(path)
            if not p.exists():
                logger.warning("HMM params file not found: %s", path)
                return False

            with p.open("r", encoding="utf-8") as f:
                params = json.load(f)

            cov_type = params["covariance_type"]
            n_components = params["n_states"]

            model = GaussianHMM(
                n_components=n_components,
                covariance_type=cov_type,
            )
            model.means_ = np.array(params["means"])
            model.transmat_ = np.array(params["transmat"])
            model.startprob_ = np.array(params["startprob"])

            # covars_ の shape は covariance_type に依存
            covars = np.array(params["covars"])
            if cov_type == "diag":
                # diag: (n_components, n_features)
                if covars.ndim == 3:
                    # full 形式で保存された場合は対角要素を抽出
                    covars = np.array([
                        np.diag(covars[i]) for i in range(n_components)
                    ])
            model.covars_ = covars

            self._model = model
            self._is_fitted = True
            self._bars_since_fit = 0

            # 状態マッピング復元
            self._state_mapping = {
                int(k): HMMRegimeState(v)
                for k, v in params["state_mapping"].items()
            }

            logger.info("HMM params loaded from %s", path)
            return True

        except Exception as e:
            logger.error("HMM load_params failed: %s", e)
            return False

    # ── Internal ──

    def _extract_features(
        self, bars: List[Dict[str, Any]]
    ) -> Optional[Any]:
        """バーデータから HMM 入力特徴量を抽出する。

        特徴量:
          [0] log_return
          [1] realized_vol (5-bar rolling std of returns)
          [2] range_ratio ((high - low) / close)

        Returns:
            numpy array of shape (n_bars - 1, 3) or None
        """
        if not _HMM_AVAILABLE:
            return None

        if len(bars) < 2:
            return None

        log_returns = []
        range_ratios = []

        for i in range(1, len(bars)):
            prev_close = bars[i - 1].get("close", 0.0)
            curr_close = bars[i].get("close", 0.0)
            high = bars[i].get("high", curr_close)
            low = bars[i].get("low", curr_close)

            if prev_close > 0 and curr_close > 0:
                lr = math.log(curr_close / prev_close)
            else:
                lr = 0.0

            rr = (high - low) / curr_close if curr_close > 0 else 0.0

            log_returns.append(lr)
            range_ratios.append(rr)

        # rolling volatility (5-bar window)
        vol_window = 5
        realized_vols = []
        for i in range(len(log_returns)):
            if i < vol_window - 1:
                # 不足分はその時点までの std
                window = log_returns[:i + 1]
            else:
                window = log_returns[i - vol_window + 1:i + 1]

            if len(window) > 1:
                mean_r = sum(window) / len(window)
                var = sum((r - mean_r) ** 2 for r in window) / len(window)
                realized_vols.append(math.sqrt(var))
            else:
                realized_vols.append(0.0)

        features = np.column_stack([
            np.array(log_returns),
            np.array(realized_vols),
            np.array(range_ratios),
        ])

        return features

    def _assign_state_mapping(self, features: Any) -> None:
        """HMM の内部状態を意味のあるラベルにマッピングする。

        ボラティリティ（特徴量[1]）の平均値でソート:
          lowest vol → LOW_VOL
          middle vol → TRENDING
          highest vol → HIGH_VOL
        """
        if self._model is None:
            return

        # 各状態の平均ボラティリティ（特徴量 index 1）
        vol_means = []
        for state_idx in range(self.config.n_states):
            vol_means.append((state_idx, self._model.means_[state_idx][1]))

        # ボラティリティ昇順でソート
        vol_means.sort(key=lambda x: x[1])

        labels = [HMMRegimeState.LOW_VOL, HMMRegimeState.TRENDING, HMMRegimeState.HIGH_VOL]
        self._state_mapping = {}
        for rank, (state_idx, _) in enumerate(vol_means):
            if rank < len(labels):
                self._state_mapping[state_idx] = labels[rank]
            else:
                self._state_mapping[state_idx] = HMMRegimeState.UNKNOWN
