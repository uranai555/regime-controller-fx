from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from statistics import median
from typing import Sequence

from .event_match import EventMatch


@dataclass(frozen=True)
class PairStability:
    leader: str
    follower: str
    matched_events: int
    windows: int
    positive_median_window_share: float
    median_window_lag_ms: float
    lag_median_bootstrap_lo_ms: float | None
    lag_median_bootstrap_hi_ms: float | None
    hourly: dict[int, dict[str, float | int]]

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(values: Sequence[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        raise ValueError("values required")
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def bootstrap_median_ci(values: Sequence[float], *, confidence: float = 0.95, resamples: int = 1000, seed: int = 7) -> tuple[float, float] | None:
    if not values:
        return None
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0,1)")
    if resamples <= 0:
        raise ValueError("resamples must be > 0")
    rng = random.Random(seed)
    n = len(values)
    medians = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        medians.append(median(sample))
    alpha = (1.0 - confidence) / 2.0
    return _percentile(medians, alpha), _percentile(medians, 1.0 - alpha)


def rolling_window_medians(matches: Sequence[EventMatch], *, window_ms: int = 30 * 60 * 1000, min_events: int = 5) -> list[float]:
    if window_ms <= 0 or min_events <= 0:
        raise ValueError("window_ms/min_events must be positive")
    if not matches:
        return []
    ordered = sorted(matches, key=lambda m: m.leader.t_ms)
    origin = ordered[0].leader.t_ms
    buckets: dict[int, list[int]] = {}
    for m in ordered:
        idx = (m.leader.t_ms - origin) // window_ms
        buckets.setdefault(idx, []).append(m.lag_ms)
    return [median(lags) for _, lags in sorted(buckets.items()) if len(lags) >= min_events]


def hourly_lag_stats(matches: Sequence[EventMatch]) -> dict[int, dict[str, float | int]]:
    buckets: dict[int, list[int]] = {}
    for m in matches:
        local_s = m.leader.t_local_s
        if local_s is None:
            continue
        hour = (local_s // 3600) % 24
        buckets.setdefault(hour, []).append(m.lag_ms)
    return {
        hour: {
            "events": len(lags),
            "median_lag_ms": float(median(lags)),
            "positive_lag_share": sum(v > 0 for v in lags) / len(lags),
        }
        for hour, lags in sorted(buckets.items())
    }


def summarize_pair_stability(
    leader: str,
    follower: str,
    matches: Sequence[EventMatch],
    *,
    window_ms: int = 30 * 60 * 1000,
    min_events_per_window: int = 5,
    bootstrap_resamples: int = 1000,
) -> PairStability:
    windows = rolling_window_medians(matches, window_ms=window_ms, min_events=min_events_per_window)
    lags = [m.lag_ms for m in matches]
    ci = bootstrap_median_ci(lags, resamples=bootstrap_resamples) if lags else None
    return PairStability(
        leader=leader,
        follower=follower,
        matched_events=len(matches),
        windows=len(windows),
        positive_median_window_share=(sum(v > 0 for v in windows) / len(windows)) if windows else 0.0,
        median_window_lag_ms=float(median(windows)) if windows else 0.0,
        lag_median_bootstrap_lo_ms=ci[0] if ci else None,
        lag_median_bootstrap_hi_ms=ci[1] if ci else None,
        hourly=hourly_lag_stats(matches),
    )
