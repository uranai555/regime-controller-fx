from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from statistics import median
from typing import Sequence

from .schema import NormalizedTick


@dataclass(frozen=True)
class PriceEvent:
    source_id: str
    symbol: str
    t_ms: int
    direction: int
    delta_mid: float
    bid: float
    ask: float
    mid: float
    threshold: float
    point: float
    t_local_s: int | None = None


@dataclass(frozen=True)
class EventMatch:
    leader: PriceEvent
    follower: PriceEvent
    lag_ms: int


@dataclass(frozen=True)
class LagEstimate:
    lag_ms: int
    objective: float
    signed_score: float
    coverage: float
    ambiguous: bool


def detect_price_events(
    ticks: Sequence[NormalizedTick],
    *,
    min_move_points: float = 1.0,
    spread_fraction: float = 0.5,
    spread_window: int = 51,
) -> list[PriceEvent]:
    if len(ticks) < 2:
        return []
    spreads: list[float] = []
    events: list[PriceEvent] = []
    prev = ticks[0]
    for tick in ticks[1:]:
        if tick.symbol != prev.symbol or tick.source_id != prev.source_id:
            raise ValueError("detect_price_events expects a single source/symbol stream")
        spreads.append(tick.spread)
        if len(spreads) > spread_window:
            spreads.pop(0)
        rolling_spread = median(spreads)
        threshold = max(min_move_points * tick.point, spread_fraction * rolling_spread)
        delta = tick.mid - prev.mid
        if abs(delta) >= threshold:
            events.append(
                PriceEvent(
                    source_id=tick.source_id,
                    symbol=tick.symbol,
                    t_ms=tick.t_host_ms,
                    direction=1 if delta > 0 else -1,
                    delta_mid=delta,
                    bid=tick.bid,
                    ask=tick.ask,
                    mid=tick.mid,
                    threshold=threshold,
                    point=tick.point,
                    t_local_s=tick.t_local_s,
                )
            )
        prev = tick
    return events


def _sample_indices(start: int, stop: int, limit: int) -> list[int]:
    """Return deterministic, evenly spaced indices from [start, stop)."""
    count = max(0, stop - start)
    if count <= limit:
        return list(range(start, stop))
    if limit <= 1:
        return [start + count // 2]
    step = (count - 1) / (limit - 1)
    return sorted({start + int(round(i * step)) for i in range(limit)})


def _cluster_lag_bins(
    candidates_by_lag: dict[int, list[tuple[int, int, float, int]]],
    *,
    max_span_ms: int,
) -> list[list[int]]:
    """Cluster lag bins around the strongest local peaks.

    A support-seeded cluster prevents a sparse boundary bin from splitting a
    dense adjacent jitter peak (for example support at 20ms once, 40ms nine
    times, and 41ms ten times). Bins exactly one full cluster-width away are
    treated as competing hypotheses rather than silently chain-merged.
    """
    if max_span_ms < 0:
        raise ValueError("max_span_ms must be >= 0")
    if not candidates_by_lag:
        return []

    def support(lag: int) -> tuple[float, float, int, int]:
        rows = candidates_by_lag[lag]
        same = sum(weight for _, _, weight, agreement in rows if agreement > 0)
        total = sum(weight for _, _, weight, _ in rows)
        return same, total, -abs(lag), -lag

    remaining = set(candidates_by_lag)
    clusters: list[list[int]] = []
    while remaining:
        seed = max(remaining, key=support)
        if max_span_ms == 0:
            cluster = [seed]
        else:
            cluster = sorted(
                lag for lag in remaining if abs(lag - seed) < max_span_ms
            )
            if seed not in cluster:
                cluster.append(seed)
                cluster.sort()
        clusters.append(cluster)
        remaining.difference_update(cluster)
    return sorted(clusters, key=lambda xs: min(xs))


def _interval_assignment(
    leader_events: Sequence[PriceEvent],
    follower_events: Sequence[PriceEvent],
    *,
    target_lag_ms: int,
    tolerance_ms: int,
    same_direction_only: bool,
) -> list[tuple[PriceEvent, PriceEvent, int, float, int]]:
    """Maximum-cardinality ordered assignment for a lag hypothesis.

    Each leader defines an interval centered on ``leader.t_ms + target_lag_ms``.
    Matching the earliest still-available follower inside each ordered interval is
    maximum-cardinality for this interval/point graph. For final event matches we
    solve positive and negative directions separately; for estimator scoring we
    keep opposite-direction matches so they remain negative evidence instead of
    being cherry-picked away.
    """
    if tolerance_ms < 0:
        raise ValueError("tolerance_ms must be >= 0")

    def assign_one(
        leaders: Sequence[PriceEvent], followers: Sequence[PriceEvent]
    ) -> list[tuple[PriceEvent, PriceEvent, int, float, int]]:
        out: list[tuple[PriceEvent, PriceEvent, int, float, int]] = []
        j = 0
        for leader in leaders:
            lower = leader.t_ms + target_lag_ms - tolerance_ms
            upper = leader.t_ms + target_lag_ms + tolerance_ms
            while j < len(followers) and followers[j].t_ms < lower:
                j += 1
            if j >= len(followers):
                break
            follower = followers[j]
            if follower.t_ms > upper:
                continue
            if follower.symbol != leader.symbol:
                continue
            lag = follower.t_ms - leader.t_ms
            weight = min(abs(leader.delta_mid), abs(follower.delta_mid))
            if weight <= 0:
                j += 1
                continue
            agreement = 1 if leader.direction == follower.direction else -1
            out.append((leader, follower, lag, weight, agreement))
            j += 1
        return out

    if not same_direction_only:
        return assign_one(leader_events, follower_events)

    selected: list[tuple[PriceEvent, PriceEvent, int, float, int]] = []
    for direction in (-1, 1):
        leaders = [e for e in leader_events if e.direction == direction]
        followers = [e for e in follower_events if e.direction == direction]
        selected.extend(assign_one(leaders, followers))
    selected.sort(key=lambda row: (row[0].t_ms, row[1].t_ms))
    return selected


def estimate_event_lag(
    leader_events: Sequence[PriceEvent],
    follower_events: Sequence[PriceEvent],
    *,
    min_lag_ms: int = -250,
    max_lag_ms: int = 1000,
    ambiguity_ratio: float = 0.98,
    lag_cluster_width_ms: int = 20,
    max_estimation_events: int = 5000,
    max_candidates_per_event: int = 256,
) -> LagEstimate | None:
    """Estimate signed lag with bounded discovery and one-to-one scoring.

    Discovery is deliberately bounded for dense captures: at most
    ``max_estimation_events`` leader events and ``max_candidates_per_event``
    follower candidates per sampled leader are inspected to discover lag peaks.
    Each candidate peak is then scored on the full event sequences with an
    ordered maximum-cardinality one-to-one assignment. Nearby jitter bins are
    compared as one support-seeded peak, while genuinely separated near-ties are
    marked ambiguous and fail closed downstream.
    """
    if not leader_events or not follower_events:
        return None
    if min_lag_ms > max_lag_ms:
        raise ValueError("min_lag_ms must be <= max_lag_ms")
    if not (0.0 < ambiguity_ratio <= 1.0):
        raise ValueError("ambiguity_ratio must be in (0, 1]")
    if lag_cluster_width_ms < 0:
        raise ValueError("lag_cluster_width_ms must be >= 0")
    if max_estimation_events <= 0 or max_candidates_per_event <= 0:
        raise ValueError("estimation bounds must be positive")

    follower_times = [event.t_ms for event in follower_events]
    candidates_by_lag: dict[int, list[tuple[int, int, float, int]]] = {}
    leader_indices = _sample_indices(0, len(leader_events), max_estimation_events)

    for leader_idx in leader_indices:
        leader = leader_events[leader_idx]
        lo = bisect_left(follower_times, leader.t_ms + min_lag_ms)
        hi = bisect_left(follower_times, leader.t_ms + max_lag_ms + 1)
        for follower_idx in _sample_indices(lo, hi, max_candidates_per_event):
            follower = follower_events[follower_idx]
            if follower.symbol != leader.symbol:
                continue
            lag = follower.t_ms - leader.t_ms
            weight = min(abs(leader.delta_mid), abs(follower.delta_mid))
            if weight <= 0:
                continue
            agreement = 1 if leader.direction == follower.direction else -1
            candidates_by_lag.setdefault(lag, []).append(
                (leader_idx, follower_idx, weight, agreement)
            )

    if not candidates_by_lag:
        return None

    clusters = _cluster_lag_bins(
        candidates_by_lag, max_span_ms=lag_cluster_width_ms
    )
    denominator = max(1, min(len(leader_events), len(follower_events)))
    score_tolerance_ms = min(5, max(1, lag_cluster_width_ms // 4)) if lag_cluster_width_ms else 0
    scored: list[tuple[float, float, float, int]] = []

    def lag_support(lag: int) -> tuple[float, float, int, int]:
        rows = candidates_by_lag[lag]
        same = sum(weight for _, _, weight, agreement in rows if agreement > 0)
        total = sum(weight for _, _, weight, _ in rows)
        return same, total, -abs(lag), -lag

    for cluster in clusters:
        # Score only a handful of strong hypotheses per local peak. Include the
        # cluster median to avoid depending entirely on a single exact-ms bin.
        hypotheses = sorted(cluster, key=lag_support, reverse=True)[:4]
        cluster_median = int(round(median(cluster)))
        if cluster_median not in hypotheses:
            hypotheses.append(cluster_median)

        best_cluster: tuple[float, float, float, int] | None = None
        for hypothesis in hypotheses:
            selected = _interval_assignment(
                leader_events,
                follower_events,
                target_lag_ms=hypothesis,
                tolerance_ms=score_tolerance_ms,
                same_direction_only=False,
            )
            if not selected:
                continue
            total_weight = sum(row[3] for row in selected)
            if total_weight <= 0:
                continue
            signed_score = sum(row[3] * row[4] for row in selected) / total_weight
            coverage = len(selected) / denominator
            objective = signed_score * coverage
            representative_lag = int(round(median(row[2] for row in selected)))
            row = (objective, coverage, signed_score, representative_lag)
            if best_cluster is None or row > best_cluster:
                best_cluster = row
        if best_cluster is not None:
            scored.append(best_cluster)

    if not scored:
        return None
    scored.sort(
        key=lambda row: (row[0], row[1], row[2], -abs(row[3]), -row[3]),
        reverse=True,
    )
    best = scored[0]
    if best[0] <= 0:
        return None

    ambiguous = False
    if len(scored) > 1 and scored[1][0] >= best[0] * ambiguity_ratio:
        ambiguous = True

    return LagEstimate(
        lag_ms=best[3],
        objective=best[0],
        signed_score=best[2],
        coverage=best[1],
        ambiguous=ambiguous,
    )


def match_events(
    leader_events: Sequence[PriceEvent],
    follower_events: Sequence[PriceEvent],
    *,
    min_lag_ms: int = -250,
    max_lag_ms: int = 1000,
    target_lag_ms: int | None = None,
    lag_tolerance_ms: int = 20,
) -> list[EventMatch]:
    """Maximum-cardinality same-direction matches around a signed lag."""
    if lag_tolerance_ms < 0:
        raise ValueError("lag_tolerance_ms must be >= 0")
    if not leader_events or not follower_events:
        return []

    if target_lag_ms is None:
        estimate = estimate_event_lag(
            leader_events,
            follower_events,
            min_lag_ms=min_lag_ms,
            max_lag_ms=max_lag_ms,
            lag_cluster_width_ms=lag_tolerance_ms,
        )
        if estimate is None or estimate.ambiguous:
            return []
        target_lag_ms = estimate.lag_ms

    if target_lag_ms < min_lag_ms or target_lag_ms > max_lag_ms:
        return []

    selected = _interval_assignment(
        leader_events,
        follower_events,
        target_lag_ms=target_lag_ms,
        tolerance_ms=lag_tolerance_ms,
        same_direction_only=True,
    )
    return [
        EventMatch(leader=row[0], follower=row[1], lag_ms=row[2])
        for row in selected
        if min_lag_ms <= row[2] <= max_lag_ms
    ]
