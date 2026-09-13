from __future__ import annotations

import random
from bisect import bisect_left
from dataclasses import asdict, dataclass
from statistics import median
from typing import Sequence

from .event_match import EventMatch, PriceEvent, match_events


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
    lag_consistent_window_share: float = 1.0

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


def bootstrap_median_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 7,
) -> tuple[float, float] | None:
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


def rolling_window_medians(
    matches: Sequence[EventMatch],
    *,
    window_ms: int = 30 * 60 * 1000,
    min_events: int = 5,
) -> list[float]:
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
    return [
        median(lags)
        for _, lags in sorted(buckets.items())
        if len(lags) >= min_events
    ]


def _window_local_matches(
    leader_events: Sequence[PriceEvent],
    follower_events: Sequence[PriceEvent],
    *,
    window_ms: int,
    min_events: int,
    min_lag_ms: int = -250,
    max_lag_ms: int = 1000,
) -> tuple[list[float | None], list[EventMatch]]:
    """Estimate lag independently in every eligible leader-time window.

    Windows with enough leader events are counted even if no local lag can be
    recovered. This prevents a global lag mode from deleting off-mode periods
    before the stability calculation sees them.
    """
    if not leader_events:
        return [], []
    ordered_leaders = sorted(leader_events, key=lambda e: e.t_ms)
    ordered_followers = sorted(follower_events, key=lambda e: e.t_ms)
    follower_times = [e.t_ms for e in ordered_followers]
    origin = ordered_leaders[0].t_ms
    last_idx = (ordered_leaders[-1].t_ms - origin) // window_ms
    medians: list[float | None] = []
    all_matches: list[EventMatch] = []

    left = 0
    for idx in range(last_idx + 1):
        start = origin + idx * window_ms
        end = start + window_ms
        while left < len(ordered_leaders) and ordered_leaders[left].t_ms < start:
            left += 1
        right = left
        while right < len(ordered_leaders) and ordered_leaders[right].t_ms < end:
            right += 1
        local_leaders = ordered_leaders[left:right]
        if len(local_leaders) < min_events:
            continue

        flo = bisect_left(follower_times, start + min_lag_ms)
        fhi = bisect_left(follower_times, end + max_lag_ms + 1)
        local_followers = ordered_followers[flo:fhi]
        local_matches = match_events(
            local_leaders,
            local_followers,
            min_lag_ms=min_lag_ms,
            max_lag_ms=max_lag_ms,
        )
        medians.append(
            float(median(m.lag_ms for m in local_matches)) if local_matches else None
        )
        all_matches.extend(local_matches)
    return medians, all_matches


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
    leader_events: Sequence[PriceEvent] | None = None,
    follower_events: Sequence[PriceEvent] | None = None,
    lag_consistency_tolerance_ms: float = 20.0,
) -> PairStability:
    if window_ms <= 0 or min_events_per_window <= 0:
        raise ValueError("window_ms/min_events_per_window must be positive")
    if lag_consistency_tolerance_ms < 0:
        raise ValueError("lag_consistency_tolerance_ms must be >= 0")

    if leader_events is not None and follower_events is not None:
        window_values, stability_matches = _window_local_matches(
            leader_events,
            follower_events,
            window_ms=window_ms,
            min_events=min_events_per_window,
        )
        window_count = len(window_values)
        valid_windows = [v for v in window_values if v is not None]
        lags = [m.lag_ms for m in stability_matches]
        hourly_source = stability_matches
    else:
        valid_windows = rolling_window_medians(
            matches, window_ms=window_ms, min_events=min_events_per_window
        )
        window_values = list(valid_windows)
        window_count = len(valid_windows)
        lags = [m.lag_ms for m in matches]
        hourly_source = matches

    ci = bootstrap_median_ci(lags, resamples=bootstrap_resamples) if lags else None
    center = float(median(valid_windows)) if valid_windows else 0.0
    positive_share = (
        sum(v is not None and v > 0 for v in window_values) / window_count
        if window_count
        else 0.0
    )
    consistent_share = (
        sum(
            v is not None and abs(v - center) <= lag_consistency_tolerance_ms
            for v in window_values
        )
        / window_count
        if window_count
        else 0.0
    )

    return PairStability(
        leader=leader,
        follower=follower,
        matched_events=len(lags),
        windows=window_count,
        positive_median_window_share=positive_share,
        median_window_lag_ms=center,
        lag_median_bootstrap_lo_ms=ci[0] if ci else None,
        lag_median_bootstrap_hi_ms=ci[1] if ci else None,
        hourly=hourly_lag_stats(hourly_source),
        lag_consistent_window_share=consistent_share,
    )
