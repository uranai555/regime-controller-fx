import pytest

from regime.microstructure.clock import TickCountExtender, UINT32_MOD
from regime.microstructure.event_match import (
    PriceEvent,
    detect_price_events,
    estimate_event_lag,
    match_events,
)
from regime.microstructure.ingest import TickLogError, encode_header, encode_record, loads_ticks
from regime.microstructure.lead_lag import pairwise_lag_matrix, passive_markout
from regime.microstructure.schema import NormalizedTick, RawTick


def nt(source, seq, t, mid, spread=0.2, point=0.1):
    return NormalizedTick(source, "XAUUSD", seq, t, None, mid-spread/2, mid+spread/2, mid, spread, spread/point, point)


def pe(source, t, direction, delta=1.0):
    mid = 100.0
    return PriceEvent(
        source, "XAUUSD", t, direction, direction * delta,
        mid - 0.1, mid + 0.1, mid, 0.1, 0.1,
    )


def test_tickcount_wrap_and_backward_rejection():
    e = TickCountExtender()
    assert e.extend(UINT32_MOD - 3) == UINT32_MOD - 3
    assert e.extend(5) == UINT32_MOD + 5
    e2 = TickCountExtender(); e2.extend(1000)
    with pytest.raises(ValueError): e2.extend(900)


def test_binary_roundtrip_and_truncated_fail_closed():
    raw = RawTick(0, 123, 999, 10, 11, 100.0, 100.2, None, 7, 0.1, 1, 0)
    data = encode_header() + encode_record(raw)
    out = loads_ticks(data, source_id="A", symbol="XAUUSD")
    assert len(out) == 1 and out[0].spread_points == pytest.approx(2.0)
    with pytest.raises(TickLogError): loads_ticks(data[:-1], source_id="A", symbol="XAUUSD")


def synthetic(delay_ms, source):
    mids = [100,100,101,101,102,102,101,101,103,103,102]
    return [nt(source, i, i*100+delay_ms, mid) for i, mid in enumerate(mids)]


def test_known_lag_recovery():
    stats = pairwise_lag_matrix({"A": synthetic(0,"A"), "B": synthetic(40,"B"), "C": synthetic(95,"C")}, min_move_points=5, spread_fraction=0.0)
    assert next(s for s in stats if s.leader=="A" and s.follower=="B").median_lag_ms == pytest.approx(40)
    assert next(s for s in stats if s.leader=="A" and s.follower=="C").median_lag_ms == pytest.approx(95)


def test_dense_periodic_events_do_not_flip_true_leader():
    stats = pairwise_lag_matrix({"A": synthetic(0,"A"), "C": synthetic(95,"C")}, min_move_points=5, spread_fraction=0.0)
    a_to_c = next(s for s in stats if s.leader == "A" and s.follower == "C")
    c_to_a = next(s for s in stats if s.leader == "C" and s.follower == "A")
    assert a_to_c.median_lag_ms == pytest.approx(95)
    assert c_to_a.median_lag_ms == pytest.approx(-95)
    assert c_to_a.positive_lag_share == 0.0


def test_neighboring_jitter_bins_form_one_lag_peak():
    leaders = [pe("A", i * 100, 1 if i % 2 == 0 else -1) for i in range(20)]
    followers = [
        pe("B", i * 100 + (40 if i % 2 == 0 else 41), 1 if i % 2 == 0 else -1)
        for i in range(20)
    ]
    estimate = estimate_event_lag(leaders, followers, min_lag_ms=0, max_lag_ms=100)
    assert estimate is not None
    assert not estimate.ambiguous
    assert 40 <= estimate.lag_ms <= 41
    assert estimate.coverage == pytest.approx(1.0)
    matches = match_events(leaders, followers, min_lag_ms=0, max_lag_ms=100)
    assert len(matches) == 20
    avg_lag = sum(m.lag_ms for m in matches) / len(matches)
    assert 39 <= avg_lag <= 43


def test_same_ms_burst_does_not_cartesian_inflate_coverage():
    leaders = [pe("A", i * 1000, 1) for i in range(90)]
    followers = [pe("B", i * 1000 + 200, -1) for i in range(90)]
    leaders.extend(pe("A", 100000, 1) for _ in range(10))
    followers.extend(pe("B", 100040, 1) for _ in range(10))

    estimate = estimate_event_lag(leaders, followers, min_lag_ms=0, max_lag_ms=100)
    assert estimate is not None
    assert estimate.lag_ms == pytest.approx(40)
    # 10 unique events participate; the 10x10 Cartesian burst must not become 100% coverage.
    assert estimate.coverage == pytest.approx(0.10)
    assert estimate.objective == pytest.approx(0.10)


def test_spread_aware_markout_can_reject_statistical_edge():
    b = [nt("B",0,0,100.0), nt("B",1,200,100.05)]
    e = detect_price_events([nt("B",0,0,100.0), nt("B",1,10,100.2)], min_move_points=1, spread_fraction=0.0)[0]
    consensus={"A":[nt("A",0,0,100.0),nt("A",1,100,100.05)],"B":b}
    m=passive_markout(e,b,consensus,horizon_ms=100)
    assert m is not None and m.gross_markout_points < 0


def test_event_quote_used_when_same_ms_has_later_quote():
    event=PriceEvent("B","XAUUSD",100,1,.2,99.9,100.1,100.0,.1,.1)
    source=[nt("B",0,99,100.0), NormalizedTick("B","XAUUSD",1,100,None,99.7,99.9,99.8,.2,2.0,.1)]
    m=passive_markout(event,source,{"A":[nt("A",0,200,100.15)]},horizon_ms=100,consensus_min_sources=1)
    assert m is not None and m.gross_markout == pytest.approx(.05)


def test_stale_markout_uses_strictly_prior_follower_quote():
    from regime.microstructure.lead_lag import passive_stale_markout
    leader=PriceEvent("A","XAUUSD",100,1,.2,100.0,100.2,100.1,.1,.1)
    follower=[nt("B",0,99,100.0),nt("B",1,100,100.5)]
    m=passive_stale_markout(leader,"B",follower,{"A":[nt("A",0,200,100.4)]},horizon_ms=100,consensus_min_sources=1)
    assert m is not None and m.gross_markout == pytest.approx(.3)


def test_consensus_rejects_stale_future_quotes():
    from regime.microstructure.lead_lag import consensus_mid_at
    streams={"A":[nt("A",0,900,100.0)],"B":[nt("B",0,920,100.1)]}
    assert consensus_mid_at(streams,1200,max_age_ms=250,min_sources=2) is None


def test_consensus_requires_minimum_fresh_sources():
    from regime.microstructure.lead_lag import consensus_mid_at
    streams={"A":[nt("A",0,1100,100.0)],"B":[nt("B",0,800,100.1)]}
    assert consensus_mid_at(streams,1200,max_age_ms=250,min_sources=2) is None
    assert consensus_mid_at(streams,1200,max_age_ms=250,min_sources=1) == pytest.approx(100.0)


def test_stale_entry_quote_age_is_bounded():
    from regime.microstructure.lead_lag import passive_stale_markout
    leader=PriceEvent("A","XAUUSD",2000,1,.2,100,100.2,100.1,.1,.1)
    follower=[nt("B",0,500,99.9)]
    consensus={"A":[nt("A",0,2100,100.5)],"B":[nt("B",1,2100,100.5)]}
    assert passive_stale_markout(leader,"B",follower,consensus,horizon_ms=100,max_entry_quote_age_ms=1000) is None
