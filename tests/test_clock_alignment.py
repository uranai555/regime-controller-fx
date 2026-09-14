import pytest

from regime.microstructure.clock import UINT32_MOD, align_stream_wrap_epochs
from regime.microstructure.lead_lag import pairwise_lag_matrix
from regime.microstructure.schema import NormalizedTick


def anchored(source, seq, t_host, raw, local_s, mid=100.1):
    return NormalizedTick(source, "XAUUSD", seq, t_host, None, mid-.1, mid+.1, mid, .2, 2, .1, local_s, raw)


def test_align_logs_across_uint32_wrap():
    a_raw = UINT32_MOD - 100
    b_raw = 900
    a = [anchored("A", 0, a_raw, a_raw, 1_000_000)]
    b = [anchored("B", 0, b_raw, b_raw, 1_000_001)]
    aligned = align_stream_wrap_epochs({"A": a, "B": b})
    assert aligned["B"][0].t_host_ms - aligned["A"][0].t_host_ms == 1000


def test_pairwise_lag_matrix_aligns_independently_loaded_wrap_epochs():
    # A starts just before the uint32 rollover; B starts just after it. Their
    # file-local t_host_ms values differ by ~49.7 days until the public pairwise
    # helper performs the same cross-file alignment as the full pipeline.
    a = [
        anchored("A", 0, UINT32_MOD - 100, UINT32_MOD - 100, 1_000_000, 100.0),
        anchored("A", 1, UINT32_MOD - 40, UINT32_MOD - 40, 1_000_000, 101.0),
    ]
    b = [
        anchored("B", 0, 0, 0, 1_000_001, 100.0),
        anchored("B", 1, 40, 40, 1_000_001, 101.0),
    ]
    stats = pairwise_lag_matrix({"A": a, "B": b}, min_move_points=5, spread_fraction=0.0)
    ab = next(s for s in stats if s.leader == "A" and s.follower == "B")
    assert ab.median_lag_ms == pytest.approx(80)
    assert ab.median_lag_ms < 1000
