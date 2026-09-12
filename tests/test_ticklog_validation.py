import math

import pytest

from regime.microstructure.ingest import TickLogError, encode_header, encode_record, loads_ticks
from regime.microstructure.schema import RawTick


def raw(seq, *, bid=100.0, ask=100.2, point=0.1, last=100.1):
    return RawTick(
        sequence=seq,
        host_tick_ms_raw=1000 + seq,
        program_us=10_000 + seq,
        server_time_sec=1_700_000_000,
        local_time_sec=1_700_000_000,
        bid=bid,
        ask=ask,
        last=last,
        volume=1,
        point=point,
        digits=1,
        flags=0,
    )


def test_sequence_must_start_at_zero_and_be_contiguous():
    good = encode_header() + encode_record(raw(0)) + encode_record(raw(1))
    assert len(loads_ticks(good, source_id="A", symbol="XAUUSD")) == 2

    gap = encode_header() + encode_record(raw(0)) + encode_record(raw(2))
    with pytest.raises(TickLogError, match="non-contiguous sequence"):
        loads_ticks(gap, source_id="A", symbol="XAUUSD")

    nonzero_start = encode_header() + encode_record(raw(1))
    with pytest.raises(TickLogError, match="non-contiguous sequence"):
        loads_ticks(nonzero_start, source_id="A", symbol="XAUUSD")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bid": math.nan},
        {"ask": math.inf},
        {"point": math.nan},
        {"point": math.inf},
        {"last": math.nan},
    ],
)
def test_nonfinite_market_values_fail_closed(kwargs):
    data = encode_header() + encode_record(raw(0, **kwargs))
    with pytest.raises(TickLogError):
        loads_ticks(data, source_id="A", symbol="XAUUSD")
