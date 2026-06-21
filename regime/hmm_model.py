"""hmm_model.py — Phase 2: GaussianHMM によるレジーム状態推定

自前実装の GaussianHMM (EM + Viterbi)。外部依存ゼロ。
学習済みモデルがない場合は Phase 1 のルールベースにフォールバックする。

学習単位: symbol 単位（通貨ペアごとに独立した HMM を持つ）
入力特徴量: returns_series, volatility_series, spread_series
出力: hmm_state (int), hmm_state_proba (List[float])

フォールバック条件:
  - モデルが未学習 (is_fitted == False)
  - 入力系列の長さが min_obs 未満
  - EM が収束しなかった場合
"""

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HMMConfig:
    """HMM のハイパーパラメータ。"""
    n_states: int = 3
    n_features: int = 3
    max_iter: int = 100
    tol: float = 1e-4
    min_obs: int = 30
    random_seed: int = 42


class GaussianHMM:
    """自前実装の GaussianHMM (対角共分散)。

    状態数 n_states, 特徴量数 n_features の多変量ガウス分布を仮定。
    EM (Baum-Welch) で学習し、Viterbi で最尤状態列を返す。
    """

    def __init__(self, config: HMMConfig):
        self.cfg = config
        self._fitted = False

        n = config.n_states
        d = config.n_features

        # transition matrix (row-stochastic)
        self.trans: List[List[float]] = []
        # initial state distribution
        self.pi: List[float] = []
        # emission means [state][feature]
        self.means: List[List[float]] = []
        # emission variances [state][feature] (diagonal)
        self.variances: List[List[float]] = []

        self._init_params()

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _init_params(self) -> None:
        rng = random.Random(self.cfg.random_seed)
        n = self.cfg.n_states
        d = self.cfg.n_features

        self.pi = [1.0 / n] * n
        self.trans = [[1.0 / n] * n for _ in range(n)]
        self.means = [[rng.gauss(0, 0.01) for _ in range(d)] for _ in range(n)]
        self.variances = [[1.0] * d for _ in range(n)]

    def fit(self, observations: List[List[float]]) -> bool:
        """EM (Baum-Welch) でパラメータを推定する。

        Args:
            observations: T×D の特徴量行列 (list of lists)

        Returns:
            True if converged, False otherwise
        """
        T = len(observations)
        if T < self.cfg.min_obs:
            logger.warning("HMM fit: insufficient observations (%d < %d)", T, self.cfg.min_obs)
            return False

        n = self.cfg.n_states
        d = self.cfg.n_features

        self._init_params()

        prev_log_lik = -math.inf

        for iteration in range(self.cfg.max_iter):
            # E-step
            alpha, scale = self._forward(observations)
            beta = self._backward(observations, scale)
            gamma, xi = self._compute_gamma_xi(observations, alpha, beta, scale)

            log_lik = -sum(math.log(max(s, 1e-300)) for s in scale)

            # check convergence
            if abs(log_lik - prev_log_lik) < self.cfg.tol:
                logger.info("HMM converged at iteration %d (log_lik=%.4f)", iteration, log_lik)
                self._fitted = True
                return True
            prev_log_lik = log_lik

            # M-step
            # update pi
            gamma_sum_0 = sum(gamma[0])
            if gamma_sum_0 > 0:
                self.pi = [gamma[0][j] / gamma_sum_0 for j in range(n)]

            for j in range(n):
                # update means
                gamma_j_total = sum(gamma[t][j] for t in range(T))
                if gamma_j_total < 1e-300:
                    continue

                for f in range(d):
                    self.means[j][f] = sum(
                        gamma[t][j] * observations[t][f] for t in range(T)
                    ) / gamma_j_total

                # update variances (diagonal)
                for f in range(d):
                    var = sum(
                        gamma[t][j] * (observations[t][f] - self.means[j][f]) ** 2
                        for t in range(T)
                    ) / gamma_j_total
                    self.variances[j][f] = max(var, 1e-6)

                # update transition matrix
                xi_j_total = sum(xi[t][j][k] for t in range(T - 1) for k in range(n))
                if xi_j_total > 0:
                    for k in range(n):
                        self.trans[j][k] = sum(
                            xi[t][j][k] for t in range(T - 1)
                        ) / xi_j_total

        logger.warning("HMM did not converge after %d iterations", self.cfg.max_iter)
        self._fitted = True
        return False

    def predict(self, observations: List[List[float]]) -> List[int]:
        """Viterbi アルゴリズムで最尤状態列を返す。"""
        if not self._fitted:
            raise RuntimeError("HMM is not fitted. Call fit() first.")

        T = len(observations)
        n = self.cfg.n_states

        # log domain viterbi
        log_delta: List[List[float]] = []
        psi: List[List[int]] = []

        # init
        first_row: List[float] = []
        for j in range(n):
            lp = _safe_log(self.pi[j]) + self._log_emission(observations[0], j)
            first_row.append(lp)
        log_delta.append(first_row)
        psi.append([0] * n)

        for t in range(1, T):
            row: List[float] = []
            psi_row: List[int] = []
            for j in range(n):
                best_val = -math.inf
                best_i = 0
                for i in range(n):
                    val = log_delta[t - 1][i] + _safe_log(self.trans[i][j])
                    if val > best_val:
                        best_val = val
                        best_i = i
                row.append(best_val + self._log_emission(observations[t], j))
                psi_row.append(best_i)
            log_delta.append(row)
            psi.append(psi_row)

        # backtrack
        states: List[int] = [0] * T
        states[T - 1] = _argmax(log_delta[T - 1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1][states[t + 1]]

        return states

    def predict_proba(self, observations: List[List[float]]) -> List[List[float]]:
        """各タイムステップの状態確率 (gamma) を返す。"""
        if not self._fitted:
            raise RuntimeError("HMM is not fitted. Call fit() first.")

        alpha, scale = self._forward(observations)
        beta = self._backward(observations, scale)

        T = len(observations)
        n = self.cfg.n_states
        proba: List[List[float]] = []

        for t in range(T):
            row: List[float] = []
            total = sum(alpha[t][j] * beta[t][j] for j in range(n))
            if total < 1e-300:
                row = [1.0 / n] * n
            else:
                row = [alpha[t][j] * beta[t][j] / total for j in range(n)]
            proba.append(row)

        return proba

    # ── internal ──

    def _forward(
        self, obs: List[List[float]]
    ) -> Tuple[List[List[float]], List[float]]:
        T = len(obs)
        n = self.cfg.n_states
        alpha: List[List[float]] = []
        scale: List[float] = []

        # t=0
        a0 = [self.pi[j] * self._emission(obs[0], j) for j in range(n)]
        s0 = sum(a0) or 1e-300
        alpha.append([v / s0 for v in a0])
        scale.append(s0)

        for t in range(1, T):
            at: List[float] = []
            for j in range(n):
                s = sum(alpha[t - 1][i] * self.trans[i][j] for i in range(n))
                at.append(s * self._emission(obs[t], j))
            st = sum(at) or 1e-300
            alpha.append([v / st for v in at])
            scale.append(st)

        return alpha, scale

    def _backward(
        self, obs: List[List[float]], scale: List[float]
    ) -> List[List[float]]:
        T = len(obs)
        n = self.cfg.n_states
        beta: List[List[float]] = [[0.0] * n for _ in range(T)]
        beta[T - 1] = [1.0] * n

        for t in range(T - 2, -1, -1):
            for i in range(n):
                beta[t][i] = sum(
                    self.trans[i][j] * self._emission(obs[t + 1], j) * beta[t + 1][j]
                    for j in range(n)
                )
            st = scale[t + 1] if scale[t + 1] > 0 else 1e-300
            beta[t] = [b / st for b in beta[t]]

        return beta

    def _compute_gamma_xi(
        self,
        obs: List[List[float]],
        alpha: List[List[float]],
        beta: List[List[float]],
        scale: List[float],
    ) -> Tuple[List[List[float]], List[List[List[float]]]]:
        T = len(obs)
        n = self.cfg.n_states

        gamma: List[List[float]] = []
        for t in range(T):
            total = sum(alpha[t][j] * beta[t][j] for j in range(n))
            if total < 1e-300:
                gamma.append([1.0 / n] * n)
            else:
                gamma.append([alpha[t][j] * beta[t][j] / total for j in range(n)])

        xi: List[List[List[float]]] = []
        for t in range(T - 1):
            xi_t: List[List[float]] = []
            denom = sum(
                alpha[t][i] * self.trans[i][j] * self._emission(obs[t + 1], j) * beta[t + 1][j]
                for i in range(n) for j in range(n)
            )
            if denom < 1e-300:
                xi_t = [[1.0 / (n * n)] * n for _ in range(n)]
            else:
                for i in range(n):
                    row: List[float] = []
                    for j in range(n):
                        val = (
                            alpha[t][i] * self.trans[i][j]
                            * self._emission(obs[t + 1], j) * beta[t + 1][j]
                        ) / denom
                        row.append(val)
                    xi_t.append(row)
            xi.append(xi_t)

        return gamma, xi

    def _emission(self, obs_t: List[float], state: int) -> float:
        """多変量ガウス (対角共分散) の密度。"""
        d = self.cfg.n_features
        p = 1.0
        for f in range(d):
            mu = self.means[state][f]
            var = self.variances[state][f]
            diff = obs_t[f] - mu
            p *= (1.0 / math.sqrt(2 * math.pi * var)) * math.exp(-0.5 * diff * diff / var)
        return max(p, 1e-300)

    def _log_emission(self, obs_t: List[float], state: int) -> float:
        d = self.cfg.n_features
        log_p = 0.0
        for f in range(d):
            mu = self.means[state][f]
            var = self.variances[state][f]
            diff = obs_t[f] - mu
            log_p += -0.5 * math.log(2 * math.pi * var) - 0.5 * diff * diff / var
        return log_p

    def to_dict(self) -> Dict:
        return {
            "n_states": self.cfg.n_states,
            "n_features": self.cfg.n_features,
            "pi": self.pi,
            "trans": self.trans,
            "means": self.means,
            "variances": self.variances,
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GaussianHMM":
        cfg = HMMConfig(
            n_states=data["n_states"],
            n_features=data["n_features"],
        )
        hmm = cls(cfg)
        hmm.pi = data["pi"]
        hmm.trans = data["trans"]
        hmm.means = data["means"]
        hmm.variances = data["variances"]
        hmm._fitted = data.get("fitted", True)
        return hmm


def _safe_log(x: float) -> float:
    return math.log(max(x, 1e-300))


def _argmax(lst: List[float]) -> int:
    best_i = 0
    best_v = lst[0]
    for i in range(1, len(lst)):
        if lst[i] > best_v:
            best_v = lst[i]
            best_i = i
    return best_i


# ── State-to-RegimeMode mapping ──

# HMM states are ordered by volatility after fitting.
# State 0 = low vol → NORMAL, State 1 = mid vol → CAUTION, State 2 = high vol → regime-dependent
_DEFAULT_STATE_MODE_MAP: Dict[int, str] = {
    0: "NORMAL",
    1: "CAUTION",
    2: "NO_NEW_ENTRY",
}


class HMMModel:
    """Phase 2 のレジーム状態推定モデル。

    fit() でパラメータを学習し、predict() / predict_proba() で状態推定する。
    学習済みモデルがない場合は None を返し、呼び出し側が Phase 1 にフォールバックする。
    """

    def __init__(
        self,
        config: Optional[HMMConfig] = None,
        state_mode_map: Optional[Dict[int, str]] = None,
    ):
        self.cfg = config or HMMConfig()
        self._hmm = GaussianHMM(self.cfg)
        self.state_mode_map = state_mode_map or dict(_DEFAULT_STATE_MODE_MAP)

    @property
    def is_fitted(self) -> bool:
        return self._hmm.is_fitted

    def fit(self, observations: List[List[float]]) -> bool:
        """観測系列で HMM を学習する。

        学習後に状態をボラティリティ順に自動ソートする。
        state 0 = 最低ボラ (NORMAL), state N-1 = 最高ボラ (危険)

        Args:
            observations: T×D の特徴量行列。
                D=3 の場合: [returns, volatility, spread_ratio]

        Returns:
            True if converged
        """
        result = self._hmm.fit(observations)
        if self._hmm.is_fitted:
            self._sort_states_by_volatility()
        return result

    def predict(self, observations: List[List[float]]) -> Optional[int]:
        """最新タイムステップの状態を返す。未学習なら None。"""
        if not self._hmm.is_fitted:
            return None
        if len(observations) < 1:
            return None
        states = self._hmm.predict(observations)
        return states[-1]

    def predict_proba(self, observations: List[List[float]]) -> Optional[List[float]]:
        """最新タイムステップの状態確率を返す。未学習なら None。"""
        if not self._hmm.is_fitted:
            return None
        if len(observations) < 1:
            return None
        proba = self._hmm.predict_proba(observations)
        return proba[-1]

    def predict_mode(self, observations: List[List[float]]) -> Optional[str]:
        """HMM 状態を RegimeMode 文字列にマッピングして返す。"""
        state = self.predict(observations)
        if state is None:
            return None
        return self.state_mode_map.get(state)

    def _sort_states_by_volatility(self) -> None:
        """学習後に状態を volatility (分散の大きさ) でソートする。

        state 0 = 最低ボラ (NORMAL), state N-1 = 最高ボラ (危険)
        """
        if not self._hmm.is_fitted:
            return

        n = self.cfg.n_states
        # volatility index = sum of variances per state
        vol_index = [sum(self._hmm.variances[j]) for j in range(n)]
        order = sorted(range(n), key=lambda j: vol_index[j])

        # re-order all parameters
        self._hmm.pi = [self._hmm.pi[order[j]] for j in range(n)]
        self._hmm.means = [self._hmm.means[order[j]] for j in range(n)]
        self._hmm.variances = [self._hmm.variances[order[j]] for j in range(n)]
        new_trans = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                new_trans[i][j] = self._hmm.trans[order[i]][order[j]]
        self._hmm.trans = new_trans

    def save(self, path: str) -> None:
        """モデルを JSON ファイルに保存する。"""
        data = self._hmm.to_dict()
        data["state_mode_map"] = {str(k): v for k, v in self.state_mode_map.items()}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2))
        logger.info("HMM model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> Optional["HMMModel"]:
        """JSON ファイルからモデルをロードする。ファイルがなければ None。"""
        p = Path(path)
        if not p.exists():
            logger.info("HMM model file not found: %s", path)
            return None
        try:
            data = json.loads(p.read_text())
            model = cls(
                config=HMMConfig(
                    n_states=data["n_states"],
                    n_features=data["n_features"],
                ),
            )
            model._hmm = GaussianHMM.from_dict(data)
            if "state_mode_map" in data:
                model.state_mode_map = {int(k): v for k, v in data["state_mode_map"].items()}
            return model
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to load HMM model from %s: %s", path, e)
            return None
