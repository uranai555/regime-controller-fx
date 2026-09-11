#!/usr/bin/env python3
from __future__ import annotations

import argparse
from statistics import median

from regime.microstructure.ingest import load_ticks


def main() -> int:
    p = argparse.ArgumentParser(description="Validate and summarize a Phase 4A MT4 binary tick log")
    p.add_argument("path")
    p.add_argument("--source-id", required=True)
    p.add_argument("--symbol", default="XAUUSD")
    args = p.parse_args()
    ticks = load_ticks(args.path, source_id=args.source_id, symbol=args.symbol)
    if not ticks:
        print("ticks=0")
        return 2
    intervals = [b.t_host_ms - a.t_host_ms for a, b in zip(ticks, ticks[1:])]
    print(f"ticks={len(ticks)} symbol={args.symbol} source={args.source_id}")
    print(f"t_first_ms={ticks[0].t_host_ms} t_last_ms={ticks[-1].t_host_ms}")
    print(f"median_spread_points={median(t.spread_points for t in ticks):.3f}")
    if intervals:
        print(f"median_intertick_ms={median(intervals):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
