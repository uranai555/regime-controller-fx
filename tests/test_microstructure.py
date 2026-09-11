import pytest

from regime.microstructure.clock import TickCountExtender, UINT32_MOD
from regime.microstructure.event_match import detect_price_events
from regime.microstructure.ingest import TickLogError, encode_header, encode_record, loads_ticks
from regime.microstructure.lead_lag import pairwise_lag_matrix, passive_markout
from regime.microstructure.schema import NormalizedTick, RawTick


def nt(source, seq, t, mid, spread=0.2, point=0.1):
    return NormalizedTick(source, "XAUUSD", seq, t, None, mid-spread/2, mid+spread/2, mid, spread, spread/point, point)


def test_tickcount_wrap_and_backward_rejection():
    e = TickCountExtender()
    assert e.extend(UINT32_MOD - 3) == UINT32_MOD - 3
    assert e.extend(5) == UINT32_MOD + 5
    e2 = TickCountExtender()
    e2.extend(1000)
    with pytest.raises(ValueError):
        e2.extend(900)


def test_binary_roundtrip_and_truncated_fail_closed():
    raw = RawTick(1, 123, 999, 10, 11, 100.0, 100.2, None, 7, 0.1, 1, 0)
    data = encode_header() + encode_record(raw)
    out = loads_ticks(data, source_id="A", symbol="XAUUSD")
    assert len(out) == 1 and out[0].spread_points == pytest.approx(2.0)
    with pytest.raises(TickLogError):
        loads_ticks(data[:-1], source_id="A", symbol="XAUUSD")


def synthetic(delay_ms, source):
    mids = [100, 100, 101, 101, 102, 102, 101, 101, 103, 103, 102]
    return [nt(source, i, i*100 + delay_ms, mid, spread=0.2, point=0.1) for i, mid in enumerate(mids)]


def test_known_lag_recovery():
    a = synthetic(0, "A")
    b = synthetic(40, "B")
    c = synthetic(95, "C")
    stats = pairwise_lag_matrix({"A": a, "B": b, "C": c}, min_move_points=5, spread_fraction=0.0)
    ab = next(s for s in stats if s.leader == "A" and s.follower == "B")
    ac = next(s for s in stats if s.leader == "A" and s.follower == "C")
    assert ab.median_lag_ms == pytest.approx(40)
    assert ac.median_lag_ms == pytest.approx(95)
    assert ab.positive_lag_share == 1.0


def test_spread_aware_markout_can_reject_statistical_edge():
    b_ticks = [nt("B", 0, 0, 100.0, spread=0.2, point=0.1), nt("B", 1, 200, 100.05, spread=0.2, point=0.1)]
    events = detect_price_events([nt("B",0,0,100.0,0.2,0.1), nt("B",1,10,100.2,0.2,0.1)], min_move_points=1, spread_fraction=0.0)
    e = events[0]
    consensus = {"A": [nt("A",0,0,100.0,0.2,0.1), nt("A",1,100,100.05,0.2,0.1)], "B": b_ticks}
    m = passive_markout(e, b_ticks, consensus, horizon_ms=100)
    assert m is not None
    assert m.gross_markout_points < 0


def test_event_quote_used_for_markout_when_same_ms_has_later_quote():
    from regime.microstructure.event_match import PriceEvent
    event = PriceEvent("B", "XAUUSD", 100, 1, 0.2, 99.9, 100.1, 100.0, 0.1, 0.1)
    source = [
        nt("B", 0, 99, 100.0, 0.2, 0.1),
        NormalizedTick("B", "XAUUSD", 1, 100, None, 99.7, 99.9, 99.8, 0.2, 2.0, 0.1),
    ]
    consensus = {"A": [nt("A", 0, 200, 100.15, 0.2, 0.1)]}
    m = passive_markout(event, source, consensus, horizon_ms=100)
    assert m is not None
    assert m.gross_markout == pytest.approx(0.05)


def test_stale_markout_uses_strictly_prior_follower_quote():
    from regime.microstructure.event_match import PriceEvent
    from regime.microstructure.lead_lag import passive_stale_markout
    leader_event = PriceEvent("A", "XAUUSD", 100, 1, 0.2, 100.0, 100.2, 100.1, 0.1, 0.1)
    follower = [
        nt("B", 0, 99, 100.0, 0.2, 0.1),
        nt("B", 1, 100, 100.5, 0.2, 0.1),
    ]
    consensus = {"A": [nt("A", 0, 200, 100.4, 0.2, 0.1)]}
    m = passive_stale_markout(leader_event, "B", follower, consensus, horizon_ms=100)
    assert m is not None
    assert m.gross_markout == pytest.approx(0.3)
