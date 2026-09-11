from pathlib import Path

from regime.microstructure.fingerprint import BrokerFingerprint
from regime.microstructure.friction import StressResult
from regime.microstructure.report import choose_verdict, write_markdown_report
from regime.microstructure.stability import PairStability


def fp(name, ticks=100000, leader=.7, ev=1.0):
    return BrokerFingerprint(name, "XAUUSD", ticks, 10, 100, 2, leader, 30, 100, .6, ev)


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
