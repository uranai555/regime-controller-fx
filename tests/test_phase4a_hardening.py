import sys

import pytest

from regime.microstructure.event_match import PriceEvent, estimate_event_lag
from regime.microstructure.fingerprint import BrokerFingerprint
from regime.microstructure.friction import StressResult
from regime.microstructure.ingest import encode_header, encode_record
from regime.microstructure.lead_lag import passive_markout
from regime.microstructure.report import choose_verdict
from regime.microstructure.schema import NormalizedTick, RawTick
from regime.microstructure.stability import PairStability, summarize_pair_stability
from scripts.validate_mt4_smoke import main as smoke_main


def pe(source, t, direction, delta=1.0):
    return PriceEvent(
        source, "XAUUSD", t, direction, direction * delta,
        99.9, 100.1, 100.0, 0.1, 0.1,
    )


def nt(source, seq, t, mid=100.0):
    return NormalizedTick(
        source, "XAUUSD", seq, t, None,
        mid - 0.1, mid + 0.1, mid, 0.2, 2.0, 0.1,
    )


def test_sparse_boundary_bin_does_not_split_dense_jitter_peak():
    leaders = [pe("A", i * 100, 1 if i % 2 == 0 else -1) for i in range(20)]
    delays = [20] + [40] * 9 + [41] * 10
    followers = [
        pe("B", i * 100 + delays[i], 1 if i % 2 == 0 else -1)
        for i in range(20)
    ]
    estimate = estimate_event_lag(leaders, followers, min_lag_ms=0, max_lag_ms=100)
    assert estimate is not None
    assert not estimate.ambiguous
    assert 39 <= estimate.lag_ms <= 42
    assert estimate.coverage >= 0.9


def test_close_events_preserve_exact_max_cardinality_alignment():
    directions = [1, -1, 1]
    leaders = [pe("A", t, d) for t, d in zip((0, 10, 20), directions)]
    followers = [pe("B", t, d) for t, d in zip((0, 10, 20), directions)]
    estimate = estimate_event_lag(leaders, followers, min_lag_ms=0, max_lag_ms=20)
    assert estimate is not None
    assert not estimate.ambiguous
    assert estimate.lag_ms == 0
    assert estimate.coverage == pytest.approx(1.0)
    assert estimate.signed_score == pytest.approx(1.0)


def test_exact_horizon_final_tick_does_not_prove_future_coverage():
    event = pe("B", 100, 1)
    consensus = {"A": [nt("A", 0, 199, 100.2), nt("A", 1, 200, 100.4)]}
    assert passive_markout(
        event, [], consensus, horizon_ms=100, consensus_min_sources=1
    ) is None


def test_window_local_stability_detects_lag_regime_shift():
    leaders = [pe("A", i * 1000, 1) for i in range(10)]
    followers = [
        pe("B", i * 1000 + (40 if i < 5 else 100), 1)
        for i in range(10)
    ]
    result = summarize_pair_stability(
        "A",
        "B",
        [],
        window_ms=5000,
        min_events_per_window=3,
        bootstrap_resamples=100,
        leader_events=leaders,
        follower_events=followers,
        lag_consistency_tolerance_ms=20,
    )
    assert result.windows == 2
    assert result.positive_median_window_share == pytest.approx(1.0)
    assert result.lag_consistent_window_share < 0.8


def test_probe_verdict_requires_lag_consistent_windows():
    fps = [
        BrokerFingerprint("A", "XAUUSD", 100000, 10, 100, 2, .7, 30, 100, .6, 1.0),
        BrokerFingerprint("B", "XAUUSD", 100000, 10, 100, 2, .2, 30, 100, .6, 1.0),
    ]
    positive = StressResult("slippage_1.0x_spread", 100, .4, .3, .6, -1, 2)
    shifted = PairStability(
        "A", "B", 100, 4, 1.0, 70, 35, 105, {},
        lag_consistent_window_share=0.5,
    )
    assert choose_verdict(
        fps,
        stress={("A", "B"): [positive]},
        stability=[shifted],
    ) == "OBSERVATIONAL_EDGE_ONLY"


def test_smoke_validator_rejects_duplicate_capture_file(tmp_path, monkeypatch, capsys):
    path = tmp_path / "capture.bin"
    rows = [
        RawTick(0, 100, 1, 1, 1, 100.0, 100.2, None, 1, 0.1, 1, 0),
        RawTick(1, 31100, 2, 32, 32, 100.1, 100.3, None, 1, 0.1, 1, 0),
    ]
    path.write_bytes(encode_header() + b"".join(encode_record(r) for r in rows))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_mt4_smoke",
            "--source", f"A={path}",
            "--source", f"B={path}",
            "--min-duration-s", "0",
        ],
    )
    assert smoke_main() == 2
    assert "duplicate capture file" in capsys.readouterr().out
