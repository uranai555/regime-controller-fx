#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from regime.microstructure.ingest import load_ticks
from regime.microstructure.pipeline import analyze_streams


def _source_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be ALIAS=/path/to/log.bin")
    alias, path = value.split("=", 1)
    if not alias or not path:
        raise argparse.ArgumentTypeError("source must be ALIAS=/path/to/log.bin")
    return alias, Path(path)


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze passive MT4 broker lead/lag logs")
    p.add_argument("--source", action="append", type=_source_arg, required=True, help="ALIAS=/path/to/log.bin (repeat >=2)")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--output-dir", default="output/microstructure")
    p.add_argument("--min-move-points", type=float, default=1.0)
    p.add_argument("--spread-fraction", type=float, default=0.5)
    p.add_argument("--commission-points", type=float, default=0.0)
    p.add_argument("--cashback-points", type=float, default=0.0)
    p.add_argument("--stability-window-min", type=float, default=30.0)
    p.add_argument("--stability-min-events", type=int, default=5)
    p.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = p.parse_args()
    if len(args.source) < 2:
        p.error("at least two --source arguments are required")
    streams = {alias: load_ticks(path, source_id=alias, symbol=args.symbol) for alias, path in args.source}
    stats, fps, verdict = analyze_streams(
        streams,
        output_dir=args.output_dir,
        min_move_points=args.min_move_points,
        spread_fraction=args.spread_fraction,
        commission_points=args.commission_points,
        cashback_points=args.cashback_points,
        stability_window_ms=int(args.stability_window_min * 60_000),
        stability_min_events=args.stability_min_events,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    print(f"brokers={len(fps)} pairs={len(stats)} verdict={verdict}")
    print(Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
