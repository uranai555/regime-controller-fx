from pathlib import Path

from regime.microstructure.fingerprint import BrokerFingerprint
from regime.microstructure.friction import StressResult
from regime.microstructure.report import choose_verdict, write_markdown_report


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


def test_stress_can_graduate_candidate_when_leader_and_net_edge_exist():
    fps = [fp("A", leader=.7), fp("B", leader=.2, ev=.1)]
    positive = StressResult("slippage_1.0x_spread", 100, .4, .3, .6, -1, 2)
    stress = {"A": [positive], "B": [positive]}
    assert choose_verdict(fps, stress=stress) == "CANDIDATE_FOR_EXECUTION_PROBE"
