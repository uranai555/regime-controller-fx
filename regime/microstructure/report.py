from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .fingerprint import BrokerFingerprint
from .lead_lag import PairLagStats

VERDICTS = {"NO_EDGE", "OBSERVATIONAL_EDGE_ONLY", "CANDIDATE_FOR_EXECUTION_PROBE", "DATA_INSUFFICIENT"}


def write_lag_matrix(path: str | Path, stats: Sequence[PairLagStats]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(stats[0]).keys()) if stats else [
        "leader", "follower", "leader_events", "matched_events", "match_rate",
        "median_lag_ms", "p75_lag_ms", "p90_lag_ms", "p95_lag_ms", "positive_lag_share",
    ]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in stats:
            w.writerow(asdict(row))


def write_fingerprints(path: str | Path, fps: Sequence[BrokerFingerprint]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump([fp.to_dict() for fp in fps], f, indent=2, ensure_ascii=False)


def choose_verdict(fps: Sequence[BrokerFingerprint], *, min_ticks: int = 100_000) -> str:
    if len(fps) < 2 or any(fp.ticks < min_ticks for fp in fps):
        return "DATA_INSUFFICIENT"
    evs = [fp.theoretical_ev_100ms_points for fp in fps if fp.theoretical_ev_100ms_points is not None]
    if not evs or max(evs) <= 0:
        return "NO_EDGE"
    leaders = [fp for fp in fps if fp.leader_share >= 0.6]
    if leaders:
        return "CANDIDATE_FOR_EXECUTION_PROBE"
    return "OBSERVATIONAL_EDGE_ONLY"


def write_markdown_report(path: str | Path, fps: Sequence[BrokerFingerprint], stats: Sequence[PairLagStats], *, verdict: str | None = None) -> str:
    verdict = verdict or choose_verdict(fps)
    if verdict not in VERDICTS:
        raise ValueError(f"invalid verdict: {verdict}")
    lines = [
        "# Broker Microstructure Report", "", f"**Verdict:** `{verdict}`", "",
        "> Phase 4A is passive observation only. Theoretical markout is not realized trade P&L.", "",
        "## Broker fingerprints", "",
        "| Broker | Ticks | Median intertick ms | P95 intertick ms | Median spread pts | Leader share | EV@100ms pts |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fp in fps:
        ev = "n/a" if fp.theoretical_ev_100ms_points is None else f"{fp.theoretical_ev_100ms_points:.3f}"
        lines.append(f"| {fp.broker_id} | {fp.ticks} | {fp.median_intertick_ms:.1f} | {fp.p95_intertick_ms:.1f} | {fp.median_spread_points:.2f} | {fp.leader_share:.1%} | {ev} |")
    lines += ["", "## Pairwise lead/lag", "", "| Leader | Follower | Events | Match rate | Median lag ms | P95 lag ms | Positive lag share |", "|---|---|---:|---:|---:|---:|---:|"]
    for s in stats:
        lines.append(f"| {s.leader} | {s.follower} | {s.leader_events} | {s.match_rate:.1%} | {s.median_lag_ms:.1f} | {s.p95_lag_ms:.1f} | {s.positive_lag_share:.1%} |")
    lines += ["", "## Gate", "", "Do not move to live/min-lot probing until data, lead/lag stability and friction-stressed economic gates are satisfied.", ""]
    text = "\n".join(lines)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return verdict
