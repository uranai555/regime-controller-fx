from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Mapping, Sequence

from .event_match import EventMatch, PriceEvent, detect_price_events, match_events
from .schema import NormalizedTick


def _pct(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


@dataclass(frozen=True)
class PairLagStats:
    leader: str
    follower: str
    leader_events: int
    matched_events: int
    match_rate: float
    median_lag_ms: float
    p75_lag_ms: float
    p90_lag_ms: float
    p95_lag_ms: float
    positive_lag_share: float


def summarize_matches(leader: str, follower: str, leader_event_count: int, matches: Sequence[EventMatch]) -> PairLagStats:
    lags = [m.lag_ms for m in matches]
    return PairLagStats(
        leader=leader,
        follower=follower,
        leader_events=leader_event_count,
        matched_events=len(matches),
        match_rate=(len(matches) / leader_event_count) if leader_event_count else 0.0,
        median_lag_ms=median(lags) if lags else 0.0,
        p75_lag_ms=_pct(lags, 0.75),
        p90_lag_ms=_pct(lags, 0.90),
        p95_lag_ms=_pct(lags, 0.95),
        positive_lag_share=(sum(1 for x in lags if x > 0) / len(lags)) if lags else 0.0,
    )


def pairwise_lag_matrix(streams: Mapping[str, Sequence[NormalizedTick]], **event_kwargs) -> list[PairLagStats]:
    events = {sid: detect_price_events(ticks, **event_kwargs) for sid, ticks in streams.items()}
    out: list[PairLagStats] = []
    for leader, lev in events.items():
        for follower, fev in events.items():
            if leader == follower:
                continue
            matches = match_events(lev, fev)
            out.append(summarize_matches(leader, follower, len(lev), matches))
    return out


def tick_at_or_before(ticks: Sequence[NormalizedTick], t_ms: int) -> NormalizedTick | None:
    lo, hi = 0, len(ticks) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if ticks[mid].t_host_ms <= t_ms:
            best = ticks[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def tick_strictly_before(ticks: Sequence[NormalizedTick], t_ms: int) -> NormalizedTick | None:
    """Return last quote with timestamp < t_ms, never an ambiguous same-ms quote."""
    lo, hi = 0, len(ticks) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if ticks[mid].t_host_ms < t_ms:
            best = ticks[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def consensus_mid_at(streams: Mapping[str, Sequence[NormalizedTick]], t_ms: int) -> float | None:
    mids = []
    for ticks in streams.values():
        tick = tick_at_or_before(ticks, t_ms)
        if tick is not None:
            mids.append(tick.mid)
    return median(mids) if mids else None


@dataclass(frozen=True)
class PassiveMarkout:
    source_id: str
    event_t_ms: int
    direction: int
    horizon_ms: int
    gross_markout: float
    gross_markout_points: float


def passive_markout(event: PriceEvent, source_ticks: Sequence[NormalizedTick], consensus_streams: Mapping[str, Sequence[NormalizedTick]], *, horizon_ms: int) -> PassiveMarkout | None:
    """Markout from the quote that actually generated ``event``.

    Do not look the source quote up again by millisecond timestamp: multiple quotes
    may share a GetTickCount millisecond, which would introduce look-ahead.
    """
    future_mid = consensus_mid_at(consensus_streams, event.t_ms + horizon_ms)
    if future_mid is None:
        return None
    if event.direction > 0:
        gross = future_mid - event.ask
    else:
        gross = event.bid - future_mid
    return PassiveMarkout(
        source_id=event.source_id,
        event_t_ms=event.t_ms,
        direction=event.direction,
        horizon_ms=horizon_ms,
        gross_markout=gross,
        gross_markout_points=gross / event.point,
    )


def passive_stale_markout(leader_event: PriceEvent, target_source_id: str, target_ticks: Sequence[NormalizedTick], consensus_streams: Mapping[str, Sequence[NormalizedTick]], *, horizon_ms: int) -> PassiveMarkout | None:
    """Theoretical markout available on a follower's stale quote at leader-event time."""
    # Cross-terminal ordering inside the same GetTickCount millisecond is unknown.
    # Use the last strictly earlier follower quote to avoid look-ahead.
    target_tick = tick_strictly_before(target_ticks, leader_event.t_ms)
    future_mid = consensus_mid_at(consensus_streams, leader_event.t_ms + horizon_ms)
    if target_tick is None or future_mid is None:
        return None
    if leader_event.direction > 0:
        gross = future_mid - target_tick.ask
    else:
        gross = target_tick.bid - future_mid
    return PassiveMarkout(
        source_id=target_source_id,
        event_t_ms=leader_event.t_ms,
        direction=leader_event.direction,
        horizon_ms=horizon_ms,
        gross_markout=gross,
        gross_markout_points=gross / target_tick.point,
    )
