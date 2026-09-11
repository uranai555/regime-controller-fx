from pathlib import Path

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
