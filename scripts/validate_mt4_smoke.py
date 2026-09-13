#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import median

# Support both `python -m scripts.validate_mt4_smoke` and direct execution from
# a clean checkout without requiring an editable package install first.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from regime.microstructure.clock import align_stream_wrap_epochs
from regime.microstructure.ingest import RECORD_SIZE, load_ticks
from regime.microstructure.pipeline import analyze_streams

HEADER_SIZE = 8
REQUIRED_OUTPUTS = (
    "broker_lag_matrix.csv",
    "broker_fingerprint.json",
    "lag_stability.json",
    "cost_stress.json",
    "microstructure_report.md",
)


def _source_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be ALIAS=/path/to/log.bin")
    alias, raw_path = value.split("=", 1)
    if not alias or not raw_path:
        raise argparse.ArgumentTypeError("source must be ALIAS=/path/to/log.bin")
    return alias, Path(raw_path)


def _capture_duration_ms(ticks) -> int:
    return ticks[-1].t_host_ms - ticks[0].t_host_ms


def main() -> int:
    p = argparse.ArgumentParser(description="Validate 1-2 minute real MT4 Phase 4A smoke captures")
    p.add_argument("--source", action="append", type=_source_arg, required=True, help="ALIAS=/path/to/log.bin (repeat >=2)")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--output-dir", default="output/microstructure-smoke")
    p.add_argument("--min-duration-s", type=float, default=30.0)
    p.add_argument("--min-overlap-share", type=float, default=0.95)
    args = p.parse_args()

    if len(args.source) < 2:
        p.error("at least two --source arguments are required")
    if not (0.0 < args.min_overlap_share <= 1.0):
        p.error("--min-overlap-share must be in (0, 1]")

    failures: list[str] = []
    streams = {}
    seen_aliases: set[str] = set()
    seen_files: dict[Path, str] = {}

    for alias, path in args.source:
        if alias in seen_aliases:
            failures.append(f"duplicate source alias: {alias}")
            continue
        seen_aliases.add(alias)

        if not path.exists():
            failures.append(f"{alias}: file not found: {path}")
            continue
        resolved = path.resolve()
        if resolved in seen_files:
            failures.append(
                f"{alias}: duplicate capture file also used by {seen_files[resolved]}: {resolved}"
            )
            continue
        seen_files[resolved] = alias

        size = resolved.stat().st_size
        if size < HEADER_SIZE + RECORD_SIZE:
            failures.append(f"{alias}: file too small ({size} bytes)")
            continue
        if (size - HEADER_SIZE) % RECORD_SIZE != 0:
            failures.append(f"{alias}: binary size violates 8+N*{RECORD_SIZE} contract ({size} bytes)")
            continue
        try:
            ticks = load_ticks(resolved, source_id=alias, symbol=args.symbol)
        except Exception as exc:
            failures.append(f"{alias}: parser rejected file: {exc}")
            continue
        if not ticks:
            failures.append(f"{alias}: zero ticks")
            continue
        duration_ms = _capture_duration_ms(ticks)
        if duration_ms < args.min_duration_s * 1000:
            failures.append(f"{alias}: capture too short ({duration_ms/1000:.1f}s < {args.min_duration_s:.1f}s)")
        med_spread = median(t.spread_points for t in ticks)
        if med_spread <= 0:
            failures.append(f"{alias}: non-positive median spread ({med_spread})")
        streams[alias] = ticks
        print(
            f"{alias}: ticks={len(ticks)} duration_s={duration_ms/1000:.1f} "
            f"median_spread_pts={med_spread:.3f} file={resolved}"
        )

    if len(streams) < 2 and not failures:
        failures.append("fewer than two distinct valid capture streams")

    if len(streams) >= 2:
        try:
            aligned = align_stream_wrap_epochs(streams)
            starts = [ticks[0].t_host_ms for ticks in aligned.values()]
            ends = [ticks[-1].t_host_ms for ticks in aligned.values()]
            durations = [end - start for start, end in zip(starts, ends)]
            overlap_ms = max(0, min(ends) - max(starts))
            shorter_ms = min(durations)
            overlap_share = overlap_ms / shorter_ms if shorter_ms > 0 else 0.0
            print(f"overlap_ms={overlap_ms} overlap_share={overlap_share:.3%}")
            if overlap_share < args.min_overlap_share:
                failures.append(
                    f"capture overlap {overlap_share:.1%} < required {args.min_overlap_share:.1%}"
                )
        except Exception as exc:
            failures.append(f"cross-stream clock alignment failed: {exc}")

    if failures:
        print("\nSMOKE_BLOCKED")
        for item in failures:
            print(f"- {item}")
        return 2

    out = Path(args.output_dir)
    try:
        stats, fps, verdict = analyze_streams(streams, output_dir=out)
    except Exception as exc:
        print("\nSMOKE_BLOCKED")
        print(f"- end-to-end analysis failed: {exc}")
        return 2

    missing = [name for name in REQUIRED_OUTPUTS if not (out / name).exists()]
    if missing:
        print("\nSMOKE_BLOCKED")
        print(f"- missing outputs: {', '.join(missing)}")
        return 2

    print(f"pairs={len(stats)} brokers={len(fps)} verdict={verdict}")
    print(f"outputs={out.resolve()}")
    print("SMOKE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
