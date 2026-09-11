from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .clock import align_stream_wrap_epochs
from .event_match import detect_price_events, match_events
from .fingerprint import BrokerFingerprint, build_fingerprint
from .friction import default_scenarios, stress_markouts
from .lead_lag import PairLagStats, passive_stale_markout, summarize_matches
from .report import write_cost_stress, write_fingerprints, write_lag_matrix, write_markdown_report, write_stability
from .schema import NormalizedTick
from .stability import PairStability, summarize_pair_stability


def analyze_streams(
    streams: Mapping[str, Sequence[NormalizedTick]],
    *,
    output_dir: str | Path,
    min_move_points: float = 1.0,
    spread_fraction: float = 0.5,
    markout_horizon_ms: int = 100,
    commission_points: float = 0.0,
    cashback_points: float = 0.0,
    stability_window_ms: int = 30 * 60 * 1000,
    stability_min_events: int = 5,
    bootstrap_resamples: int = 1000,
) -> tuple[list[PairLagStats], list[BrokerFingerprint], str]:
    if len(streams) < 2:
        raise ValueError("at least two broker streams are required")
    streams = align_stream_wrap_epochs(streams)
    symbols = {ticks[0].symbol for ticks in streams.values() if ticks}
    if len(symbols) != 1 or any(not ticks for ticks in streams.values()):
        raise ValueError("all streams must be non-empty and share one canonical symbol")
    events = {
        sid: detect_price_events(ticks, min_move_points=min_move_points, spread_fraction=spread_fraction)
        for sid, ticks in streams.items()
    }
    stats: list[PairLagStats] = []
    stability: list[PairStability] = []
    markouts_by_target = {sid: [] for sid in streams}
    for leader, leader_events in events.items():
        for follower, follower_events in events.items():
            if leader == follower:
                continue
            matches = match_events(leader_events, follower_events)
            stats.append(summarize_matches(leader, follower, len(leader_events), matches))
            stability.append(summarize_pair_stability(
                leader,
                follower,
                matches,
                window_ms=stability_window_ms,
                min_events_per_window=stability_min_events,
                bootstrap_resamples=bootstrap_resamples,
            ))
            for match in matches:
                if match.lag_ms <= 0:
                    continue
                m = passive_stale_markout(match.leader, follower, streams[follower], streams, horizon_ms=markout_horizon_ms)
                if m is not None:
                    markouts_by_target[follower].append(m)
    fps = [build_fingerprint(sid, ticks, stats, markouts_by_target[sid]) for sid, ticks in streams.items()]
    scenarios = default_scenarios(commission_points=commission_points, cashback_points=cashback_points)
    stress = {sid: stress_markouts(markouts, scenarios) for sid, markouts in markouts_by_target.items()}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_lag_matrix(out / "broker_lag_matrix.csv", stats)
    write_fingerprints(out / "broker_fingerprint.json", fps)
    write_stability(out / "lag_stability.json", stability)
    write_cost_stress(out / "cost_stress.json", stress)
    verdict = write_markdown_report(out / "microstructure_report.md", fps, stats, stress=stress, stability=stability)
    return stats, fps, verdict
