from pathlib import Path

import regime.microstructure.pipeline as pipeline
from regime.microstructure.event_match import EventMatch, PriceEvent
from regime.microstructure.lead_lag import PassiveMarkout
from regime.microstructure.pipeline import analyze_streams
from regime.microstructure.schema import NormalizedTick


def nt(source, seq, t, mid, spread=0.2, point=0.1):
    return NormalizedTick(source, "XAUUSD", seq, t, None, mid-spread/2, mid+spread/2, mid, spread, spread/point, point)


def synthetic(delay_ms, source):
    mids = [100, 100, 101, 101, 102, 102, 101, 101, 103, 103, 102]
    return [nt(source, i, i*100 + delay_ms, mid) for i, mid in enumerate(mids)]


def test_pipeline_writes_required_outputs(tmp_path: Path):
    stats, fps, verdict = analyze_streams(
        {"A": synthetic(0, "A"), "B": synthetic(40, "B"), "C": synthetic(95, "C")},
        output_dir=tmp_path,
        min_move_points=5,
        spread_fraction=0.0,
    )
    assert (tmp_path / "broker_lag_matrix.csv").exists()
    assert (tmp_path / "broker_fingerprint.json").exists()
    assert (tmp_path / "microstructure_report.md").exists()
    assert (tmp_path / "cost_stress.json").exists()
    assert (tmp_path / "lag_stability.json").exists()
    ab = next(s for s in stats if s.leader == "A" and s.follower == "B")
    assert ab.median_lag_ms == 40
    assert verdict == "DATA_INSUFFICIENT"


def _pe(source: str, t_ms: int) -> PriceEvent:
    return PriceEvent(source, "XAUUSD", t_ms, 1, 1.0, 99.9, 100.1, 100.0, 0.1, 0.1)


def test_pipeline_markouts_are_not_conditioned_on_follower_match(monkeypatch, tmp_path: Path):
    a1, a2 = _pe("A", 100), _pe("A", 300)
    b1 = _pe("B", 140)
    event_map = {"A": [a1, a2], "B": [b1]}

    monkeypatch.setattr(pipeline, "align_stream_wrap_epochs", lambda streams: streams)
    monkeypatch.setattr(
        pipeline,
        "detect_price_events",
        lambda ticks, **kwargs: event_map[ticks[0].source_id],
    )

    def fake_match(leaders, followers):
        if leaders and followers and leaders[0].source_id == "A" and followers[0].source_id == "B":
            return [EventMatch(a1, b1, 40)]
        if leaders and followers and leaders[0].source_id == "B" and followers[0].source_id == "A":
            return [EventMatch(b1, a1, -40)]
        return []

    monkeypatch.setattr(pipeline, "match_events", fake_match)
    evaluated = []

    def fake_markout(leader_event, target_source_id, *args, **kwargs):
        evaluated.append((leader_event.t_ms, target_source_id))
        return PassiveMarkout(target_source_id, leader_event.t_ms, 1, 100, 0.1, 1.0, 2.0)

    monkeypatch.setattr(pipeline, "passive_stale_markout", fake_markout)
    streams = {
        "A": [nt("A", 0, 0, 100), nt("A", 1, 1000, 100)],
        "B": [nt("B", 0, 0, 100), nt("B", 1, 1000, 100)],
    }
    analyze_streams(streams, output_dir=tmp_path, stability_min_events=1)

    # Both ex-ante A opportunities must be evaluated even though only a1 later
    # has a same-direction B match. Conditioning on the match would bias EV.
    assert evaluated == [(100, "B"), (300, "B")]
