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


def stable_pair(leader="A", follower="B", share=.67, median_lag=40, lo=35, hi=45):
    return PairStability(leader, follower, 100, 3, share, median_lag, lo, hi, {})


def positive_stress(n=100, lo=.1):
    return StressResult("slippage_1.0x_spread", n, .4, .3, .6, -1, 2, lo)


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
    stress = {("A", "B"): [positive_stress()]}
    assert choose_verdict(fps, stress=stress, stability=[stable_pair()]) == "CANDIDATE_FOR_EXECUTION_PROBE"


def test_stress_without_stable_windows_remains_observational():
    fps = [fp("A"), fp("B", leader=.2, ev=.1)]
    stress = {("A", "B"): [positive_stress()]}
    unstable = stable_pair(share=.34, median_lag=-5, lo=-20, hi=10)
    assert choose_verdict(fps, stress=stress, stability=[unstable]) == "OBSERVATIONAL_EDGE_ONLY"


def test_profitable_different_pair_cannot_graduate_stable_pair():
    fps = [fp("A"), fp("B", leader=.2, ev=.1), fp("C")]
    stress = {("C", "B"): [positive_stress()]}
    assert choose_verdict(fps, stress=stress, stability=[stable_pair("A", "B")]) == "OBSERVATIONAL_EDGE_ONLY"


def test_bootstrap_lower_bound_must_confirm_positive_lag():
    fps = [fp("A"), fp("B", leader=.2, ev=.1)]
    stress = {("A", "B"): [positive_stress()]}
    inconclusive = stable_pair(lo=-2, hi=45)
    assert choose_verdict(fps, stress=stress, stability=[inconclusive]) == "OBSERVATIONAL_EDGE_ONLY"


def test_stress_requires_mean_confidence_lower_bound_above_zero():
    fps = [fp("A"), fp("B", leader=.2, ev=.1)]
    stress = {("A", "B"): [positive_stress(lo=-0.01)]}
    assert choose_verdict(fps, stress=stress, stability=[stable_pair()]) == "OBSERVATIONAL_EDGE_ONLY"


def test_stress_requires_minimum_observation_count():
    fps = [fp("A"), fp("B", leader=.2, ev=.1)]
    stress = {("A", "B"): [positive_stress(n=1, lo=.3)]}
    assert choose_verdict(fps, stress=stress, stability=[stable_pair()]) == "OBSERVATIONAL_EDGE_ONLY"


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
