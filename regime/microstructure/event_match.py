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


def detect_price_events(ticks: Sequence[NormalizedTick], *, min_move_points: float = 1.0, spread_fraction: float = 0.5, spread_window: int = 51) -> list[PriceEvent]:
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
            events.append(PriceEvent(
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
            ))
        prev = tick
    return events


def _cluster_lag_bins(lags: Sequence[int], *, max_span_ms: int) -> list[list[int]]:
    """Group neighboring lag bins into local peaks without chain-merging far lags."""
    if max_span_ms < 0:
        raise ValueError("max_span_ms must be >= 0")
    clusters: list[list[int]] = []
    current: list[int] = []
    for lag in sorted(lags):
        if not current or lag - current[0] <= max_span_ms:
            current.append(lag)
        else:
            clusters.append(current)
            current = [lag]
    if current:
        clusters.append(current)
    return clusters


def estimate_event_lag(
    leader_events: Sequence[PriceEvent],
    follower_events: Sequence[PriceEvent],
    *,
    min_lag_ms: int = -250,
    max_lag_ms: int = 1000,
    ambiguity_ratio: float = 0.98,
    lag_cluster_width_ms: int = 20,
) -> LagEstimate | None:
    """Estimate a signed event lag before final one-to-one matching.

    Candidate integer lags are first grouped into local lag clusters so ordinary
    millisecond jitter (for example 40/41ms) is treated as one peak rather than
    two competing explanations. Each cluster is then scored using a *one-to-one*
    leader/follower assignment. This prevents a same-ms burst from inflating
    coverage through a Cartesian product of event pairs.

    Coverage is therefore the fraction of uniquely matched events, while the
    signed score rewards same-direction moves and penalizes opposite-direction
    moves. Only genuinely separated lag clusters participate in the ambiguity
    check. Ambiguous estimates fail closed in ``match_events``.
    """
    if not leader_events or not follower_events:
        return None
    if min_lag_ms > max_lag_ms:
        raise ValueError("min_lag_ms must be <= max_lag_ms")
    if not (0.0 < ambiguity_ratio <= 1.0):
        raise ValueError("ambiguity_ratio must be in (0, 1]")
    if lag_cluster_width_ms < 0:
        raise ValueError("lag_cluster_width_ms must be >= 0")

    follower_times = [event.t_ms for event in follower_events]
    # lag -> (leader index, follower index, weight, direction agreement)
    candidates_by_lag: dict[int, list[tuple[int, int, float, int]]] = {}

    for leader_idx, leader in enumerate(leader_events):
        lo = bisect_left(follower_times, leader.t_ms + min_lag_ms)
        hi = bisect_left(follower_times, leader.t_ms + max_lag_ms + 1)
        for follower_idx in range(lo, hi):
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

    denominator = max(1, min(len(leader_events), len(follower_events)))
    scored: list[tuple[float, float, float, int]] = []

    for cluster in _cluster_lag_bins(
        list(candidates_by_lag), max_span_ms=lag_cluster_width_ms
    ):
        cluster_center = median(cluster)
        cluster_pairs: list[tuple[int, int, int, float, int]] = []
        for lag in cluster:
            for leader_idx, follower_idx, weight, agreement in candidates_by_lag[lag]:
                cluster_pairs.append(
                    (lag, leader_idx, follower_idx, weight, agreement)
                )

        # Neutral one-to-one assignment: closest to the cluster center first,
        # then larger moves. Direction is deliberately not used in assignment.
        cluster_pairs.sort(
            key=lambda row: (
                abs(row[0] - cluster_center),
                -row[3],
                row[1],
                row[2],
            )
        )
        used_leaders: set[int] = set()
        used_followers: set[int] = set()
        selected: list[tuple[int, int, int, float, int]] = []
        for pair in cluster_pairs:
            _, leader_idx, follower_idx, _, _ = pair
            if leader_idx in used_leaders or follower_idx in used_followers:
                continue
            used_leaders.add(leader_idx)
            used_followers.add(follower_idx)
            selected.append(pair)

        if not selected:
            continue
        total_weight = sum(row[3] for row in selected)
        if total_weight <= 0:
            continue
        signed_score = sum(row[3] * row[4] for row in selected) / total_weight
        coverage = len(selected) / denominator
        objective = signed_score * coverage
        representative_lag = int(round(median(row[0] for row in selected)))
        scored.append((objective, coverage, signed_score, representative_lag))

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
    if len(scored) > 1:
        second = scored[1]
        if second[0] >= best[0] * ambiguity_ratio:
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
    """One-to-one same-direction matches anchored to a global signed lag.

    A global lag is estimated first rather than choosing the event nearest 0ms.
    This prevents high-density periodic streams from flipping the inferred leader.
    Adjacent lag jitter is clustered, while separated nearly-tied lag hypotheses
    fail closed and return no matches.
    """
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

    follower_times = [event.t_ms for event in follower_events]
    used: set[int] = set()
    matches: list[EventMatch] = []

    for leader in leader_events:
        target_time = leader.t_ms + target_lag_ms
        lo = bisect_left(follower_times, target_time - lag_tolerance_ms)
        hi = bisect_left(follower_times, target_time + lag_tolerance_ms + 1)
        best_idx = None
        best_error = None
        for idx in range(lo, hi):
            if idx in used:
                continue
            follower = follower_events[idx]
            if follower.symbol != leader.symbol or follower.direction != leader.direction:
                continue
            lag = follower.t_ms - leader.t_ms
            if lag < min_lag_ms or lag > max_lag_ms:
                continue
            error = abs(lag - target_lag_ms)
            if best_error is None or error < best_error:
                best_idx = idx
                best_error = error
        if best_idx is not None:
            used.add(best_idx)
            follower = follower_events[best_idx]
            matches.append(EventMatch(
                leader=leader,
                follower=follower,
                lag_ms=follower.t_ms - leader.t_ms,
            ))

    return matches
