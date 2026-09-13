from pathlib import Path

import pytest

from regime.microstructure.fingerprint import BrokerFingerprint, build_fingerprint
from regime.microstructure.friction import StressResult
from regime.microstructure.lead_lag import PairLagStats
from regime.microstructure.report import choose_verdict, write_markdown_report
from regime.microstructure.schema import NormalizedTick
from regime.microstructure.stability import PairStability


def fp(name, ticks=100000, leader=.7, ev=1.0):
    return BrokerFingerprint(name, "XAUUSD", ticks, 10, 100, 2, leader, 30, 100, .6, ev)


def nt(source, seq, t, mid=100.0, spread=0.2, point=0.1):
    return NormalizedTick(source, "XAUUSD", seq, t, None, mid-spread/2, mid+spread/2, mid, spread, spread/point, point)


def pls(leader, follower, median_lag, p95=None):
    p95 = median_lag if p95 is None else p95
    return PairLagStats(leader, follower, 10, 10, 1.0, median_lag, median_lag, median_lag, p95, 1.0 if median_lag > 0 else 0.0)


def test_verdict_never_profitable(tmp_path: Path):
    fps = [fp("A"), fp("B", leader=.2, ev=.1)]
    verdict = choose_verdict(fps)
    assert verdict == "OBSERVATIONAL_EDGE_ONLY"
    out = tmp_path / "r.md"
    write_markdown_report(out, fps, [])
    assert "PROFITABLE" not in out.read_text()


def test_insufficient_data_gate():
    assert choose_verdict([fp("A", ticks=10), fp("B", ticks=10)]) == "DATA_INSUFFICIENT"


def test_stress_and_stability_can_graduate_candidate():
    fps = [fp("A", leader=.7), fp("B", leader=.2, ev=.1)]
    positive = StressResult("slippage_1.0x_spread", 100, .4, .3, .6, -1, 2)
    stress = {"A": [], "B": [positive]}
    stable = PairStability("A", "B", 100, 3, .67, 40, 35, 45, {})
    assert choose_verdict(fps, stress=stress, stability=[stable]) == "CANDIDATE_FOR_EXECUTION_PROBE"


def test_stress_without_stable_windows_remains_observational():
    fps = [fp("A"), fp("B", leader=.2, ev=.1)]
    positive = StressResult("slippage_1.0x_spread", 100, .4, .3, .6, -1, 2)
    stress = {"B": [positive]}
    unstable = PairStability("A", "B", 100, 3, .34, -5, -20, 10, {})
    assert choose_verdict(fps, stress=stress, stability=[unstable]) == "OBSERVATIONAL_EDGE_ONLY"


def test_fingerprint_response_lag_ignores_negative_reverse_rows():
    ticks = [nt("A", 0, 0), nt("A", 1, 100)]
    stats = [
        pls("B", "A", 40, 50),
        pls("C", "A", -95, -90),
        pls("A", "B", -40, -30),
        pls("A", "C", 95, 100),
    ]
    result = build_fingerprint("A", ticks, stats)
    assert result.median_response_lag_ms == pytest.approx(40)
    assert result.p95_response_lag_ms == pytest.approx(50)
    assert result.leader_share == pytest.approx(0.5)
