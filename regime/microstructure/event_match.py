from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
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


def estimate_event_lag(
    leader_events: Sequence[PriceEvent],
    follower_events: Sequence[PriceEvent],
    *,
    min_lag_ms: int = -250,
    max_lag_ms: int = 1000,
    ambiguity_ratio: float = 0.98,
) -> LagEstimate | None:
    """Estimate a signed event lag before one-to-one matching.

    The previous matcher minimized ``abs(lag)`` directly. In dense or periodic
    event streams that can alias a genuinely delayed event to the *next* leader
    event (for example true A->C=95ms appearing as C->A=+5ms when events arrive
    every 100ms).

    Here we first score every observed integer lag by signed move agreement.
    Same-direction move amplitudes add weight and opposite-direction moves
    subtract weight. Coverage is included in the objective so a lag supported by
    the whole event sequence beats a coincidental local alignment. The final
    one-to-one matcher is then anchored around this signed lag.

    When two separated lags are essentially tied, the estimate is marked
    ambiguous and callers should fail closed rather than manufacture a lead/lag.
    """
    if not leader_events or not follower_events:
        return None
    if min_lag_ms > max_lag_ms:
        raise ValueError("min_lag_ms must be <= max_lag_ms")
    if not (0.0 < ambiguity_ratio <= 1.0):
        raise ValueError("ambiguity_ratio must be in (0, 1]")

    follower_times = [event.t_ms for event in follower_events]
    signed_weight: dict[int, float] = defaultdict(float)
    total_weight: dict[int, float] = defaultdict(float)
    pair_count: dict[int, int] = defaultdict(int)

    for leader in leader_events:
        lo = bisect_left(follower_times, leader.t_ms + min_lag_ms)
        hi = bisect_left(follower_times, leader.t_ms + max_lag_ms + 1)
        for follower in follower_events[lo:hi]:
            if follower.symbol != leader.symbol:
                continue
            lag = follower.t_ms - leader.t_ms
            weight = min(abs(leader.delta_mid), abs(follower.delta_mid))
            if weight <= 0:
                continue
            signed_weight[lag] += weight if leader.direction == follower.direction else -weight
            total_weight[lag] += weight
            pair_count[lag] += 1

    denominator = max(1, min(len(leader_events), len(follower_events)))
    scored: list[tuple[float, float, float, int]] = []
    for lag, total in total_weight.items():
        if total <= 0:
            continue
        signed_score = signed_weight[lag] / total
        coverage = min(1.0, pair_count[lag] / denominator)
        objective = signed_score * coverage
        scored.append((objective, coverage, signed_score, lag))

    if not scored:
        return None

    scored.sort(key=lambda row: (row[0], row[1], row[2], -abs(row[3]), -row[3]), reverse=True)
    best = scored[0]
    if best[0] <= 0:
        return None

    ambiguous = False
    if len(scored) > 1:
        second = scored[1]
        if second[0] >= best[0] * ambiguity_ratio and second[3] != best[3]:
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
    Ambiguous estimates fail closed and return no matches.
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
