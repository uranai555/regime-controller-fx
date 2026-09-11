from pathlib import Path

from regime.microstructure.fingerprint import BrokerFingerprint
from regime.microstructure.report import choose_verdict, write_markdown_report


def fp(name, ticks=100000, leader=.7, ev=1.0):
    return BrokerFingerprint(name, "XAUUSD", ticks, 10, 100, 2, leader, 30, 100, .6, ev)


def test_verdict_is_execution_probe_not_profit_claim(tmp_path: Path):
    fps = [fp("A"), fp("B", leader=.2, ev=.1)]
    verdict = choose_verdict(fps)
    assert verdict == "CANDIDATE_FOR_EXECUTION_PROBE"
    out = tmp_path / "r.md"
    write_markdown_report(out, fps, [])
    assert "PROFITABLE" not in out.read_text()


def test_insufficient_data_gate():
    assert choose_verdict([fp("A", ticks=10), fp("B", ticks=10)]) == "DATA_INSUFFICIENT"
