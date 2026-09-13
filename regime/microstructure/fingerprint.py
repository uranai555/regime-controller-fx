from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Sequence

from .lead_lag import PairLagStats, PassiveMarkout, _pct
from .schema import NormalizedTick


@dataclass(frozen=True)
class BrokerFingerprint:
    broker_id: str
    symbol: str
    ticks: int
    median_intertick_ms: float
    p95_intertick_ms: float
    median_spread_points: float
    leader_share: float
    median_response_lag_ms: float
    p95_response_lag_ms: float
    positive_markout_rate_100ms: float | None
    theoretical_ev_100ms_points: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def build_fingerprint(broker_id: str, ticks: Sequence[NormalizedTick], pair_stats: Sequence[PairLagStats], markouts_100ms: Sequence[PassiveMarkout] = ()) -> BrokerFingerprint:
    if not ticks:
        raise ValueError("ticks required")
    intervals = [b.t_host_ms - a.t_host_ms for a, b in zip(ticks, ticks[1:]) if b.t_host_ms >= a.t_host_ms]
    spreads = [t.spread_points for t in ticks]
    outgoing = [s for s in pair_stats if s.leader == broker_id]
    incoming = [s for s in pair_stats if s.follower == broker_id and s.matched_events]
    positive_outgoing = [s for s in outgoing if s.median_lag_ms > 0]
    positive_incoming = [s for s in incoming if s.median_lag_ms > 0]
    leader_share = (len(positive_outgoing) / len(outgoing)) if outgoing else 0.0

    # Directed pair statistics are signed. A negative incoming lag means this
    # broker actually led the nominal leader in that reverse row, so it must not
    # be interpreted as this broker's response latency.
    response_lags = [s.median_lag_ms for s in positive_incoming]
    response_p95s = [s.p95_lag_ms for s in positive_incoming]
    marks = [m.gross_markout_points for m in markouts_100ms]
    return BrokerFingerprint(
        broker_id=broker_id,
        symbol=ticks[0].symbol,
        ticks=len(ticks),
        median_intertick_ms=median(intervals) if intervals else 0.0,
        p95_intertick_ms=_pct(intervals, 0.95) if intervals else 0.0,
        median_spread_points=median(spreads),
        leader_share=leader_share,
        median_response_lag_ms=median(response_lags) if response_lags else 0.0,
        p95_response_lag_ms=median(response_p95s) if response_p95s else 0.0,
        positive_markout_rate_100ms=(sum(1 for x in marks if x > 0) / len(marks)) if marks else None,
        theoretical_ev_100ms_points=(sum(marks) / len(marks)) if marks else None,
    )
