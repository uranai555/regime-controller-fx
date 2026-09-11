from regime.microstructure.clock import UINT32_MOD, align_stream_wrap_epochs
from regime.microstructure.schema import NormalizedTick


def anchored(source, seq, t_host, raw, local_s):
    return NormalizedTick(source, "XAUUSD", seq, t_host, None, 100, 100.2, 100.1, .2, 2, .1, local_s, raw)


def test_align_logs_across_uint32_wrap():
    a_raw = UINT32_MOD - 100
    b_raw = 900
    a = [anchored("A", 0, a_raw, a_raw, 1_000_000)]
    b = [anchored("B", 0, b_raw, b_raw, 1_000_001)]
    aligned = align_stream_wrap_epochs({"A": a, "B": b})
    assert aligned["B"][0].t_host_ms - aligned["A"][0].t_host_ms == 1000
