from __future__ import annotations

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


def match_events(leader_events: Sequence[PriceEvent], follower_events: Sequence[PriceEvent], *, min_lag_ms: int = -250, max_lag_ms: int = 1000) -> list[EventMatch]:
    """Greedy same-direction event match with one-time follower consumption."""
    matches: list[EventMatch] = []
    used: set[int] = set()
    start = 0
    for leader in leader_events:
        while start < len(follower_events) and follower_events[start].t_ms < leader.t_ms + min_lag_ms:
            start += 1
        best_idx = None
        best_abs = None
        for idx in range(start, len(follower_events)):
            f = follower_events[idx]
            lag = f.t_ms - leader.t_ms
            if lag > max_lag_ms:
                break
            if idx in used or f.symbol != leader.symbol or f.direction != leader.direction:
                continue
            a = abs(lag)
            if best_abs is None or a < best_abs:
                best_idx, best_abs = idx, a
        if best_idx is not None:
            used.add(best_idx)
            f = follower_events[best_idx]
            matches.append(EventMatch(leader=leader, follower=f, lag_ms=f.t_ms - leader.t_ms))
    return matches
