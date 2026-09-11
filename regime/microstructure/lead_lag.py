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


def consensus_mid_at(
    streams: Mapping[str, Sequence[NormalizedTick]],
    t_ms: int,
    *,
    max_age_ms: int = 250,
    min_sources: int = 2,
) -> float | None:
    """Fresh multi-source consensus at or before ``t_ms``.

    A quote older than ``max_age_ms`` is excluded so an ended/stalled stream is
    never reused as a future-horizon price.
    """
    if max_age_ms < 0 or min_sources <= 0:
        raise ValueError("max_age_ms must be >=0 and min_sources must be >0")
    mids = []
    for ticks in streams.values():
        tick = tick_at_or_before(ticks, t_ms)
        if tick is not None and 0 <= t_ms - tick.t_host_ms <= max_age_ms:
            mids.append(tick.mid)
    return median(mids) if len(mids) >= min_sources else None


@dataclass(frozen=True)
class PassiveMarkout:
    source_id: str
    event_t_ms: int
    direction: int
    horizon_ms: int
    gross_markout: float
    gross_markout_points: float
    entry_spread_points: float = 0.0


def passive_markout(event: PriceEvent, source_ticks: Sequence[NormalizedTick], consensus_streams: Mapping[str, Sequence[NormalizedTick]], *, horizon_ms: int, consensus_max_age_ms: int = 250, consensus_min_sources: int = 2) -> PassiveMarkout | None:
    future_mid = consensus_mid_at(
        consensus_streams,
        event.t_ms + horizon_ms,
        max_age_ms=consensus_max_age_ms,
        min_sources=consensus_min_sources,
    )
    if future_mid is None:
        return None
    gross = future_mid - event.ask if event.direction > 0 else event.bid - future_mid
    return PassiveMarkout(
        source_id=event.source_id,
        event_t_ms=event.t_ms,
        direction=event.direction,
        horizon_ms=horizon_ms,
        gross_markout=gross,
        gross_markout_points=gross / event.point,
        entry_spread_points=(event.ask - event.bid) / event.point,
    )


def passive_stale_markout(leader_event: PriceEvent, target_source_id: str, target_ticks: Sequence[NormalizedTick], consensus_streams: Mapping[str, Sequence[NormalizedTick]], *, horizon_ms: int, consensus_max_age_ms: int = 250, consensus_min_sources: int = 2, max_entry_quote_age_ms: int = 1000) -> PassiveMarkout | None:
    """Theoretical markout available on a follower's still-stale quote at leader-event time."""
    target_tick = tick_strictly_before(target_ticks, leader_event.t_ms)
    if target_tick is None or leader_event.t_ms - target_tick.t_host_ms > max_entry_quote_age_ms:
        return None
    future_mid = consensus_mid_at(
        consensus_streams,
        leader_event.t_ms + horizon_ms,
        max_age_ms=consensus_max_age_ms,
        min_sources=consensus_min_sources,
    )
    if future_mid is None:
        return None
    gross = future_mid - target_tick.ask if leader_event.direction > 0 else target_tick.bid - future_mid
    return PassiveMarkout(
        source_id=target_source_id,
        event_t_ms=leader_event.t_ms,
        direction=leader_event.direction,
        horizon_ms=horizon_ms,
        gross_markout=gross,
        gross_markout_points=gross / target_tick.point,
        entry_spread_points=target_tick.spread_points,
    )
