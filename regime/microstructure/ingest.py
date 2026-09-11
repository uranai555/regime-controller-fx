from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import BinaryIO, Iterable

from .clock import TickCountExtender
from .schema import NormalizedTick, RawTick

MAGIC = b"RFX4"
VERSION = 1
_RECORD = struct.Struct("<Q I Q q q d d d Q d i I")
_HEADER = struct.Struct("<4s H H")
RECORD_SIZE = _RECORD.size


class TickLogError(ValueError):
    pass


def encode_header() -> bytes:
    return _HEADER.pack(MAGIC, VERSION, RECORD_SIZE)


def encode_record(raw: RawTick) -> bytes:
    return _RECORD.pack(
        raw.sequence,
        raw.host_tick_ms_raw,
        raw.program_us,
        -1 if raw.server_time_sec is None else raw.server_time_sec,
        -1 if raw.local_time_sec is None else raw.local_time_sec,
        raw.bid,
        raw.ask,
        0.0 if raw.last is None else raw.last,
        0 if raw.volume is None else raw.volume,
        raw.point,
        raw.digits,
        raw.flags,
    )


def _decode_record(data: bytes) -> RawTick:
    vals = _RECORD.unpack(data)
    return RawTick(
        sequence=vals[0],
        host_tick_ms_raw=vals[1],
        program_us=vals[2],
        server_time_sec=None if vals[3] < 0 else vals[3],
        local_time_sec=None if vals[4] < 0 else vals[4],
        bid=vals[5],
        ask=vals[6],
        last=None if vals[7] == 0 else vals[7],
        volume=None if vals[8] == 0 else vals[8],
        point=vals[9],
        digits=vals[10],
        flags=vals[11],
    )


def iter_raw_ticks(stream: BinaryIO) -> Iterable[RawTick]:
    header = stream.read(_HEADER.size)
    if len(header) != _HEADER.size:
        raise TickLogError("missing/truncated header")
    magic, version, record_size = _HEADER.unpack(header)
    if magic != MAGIC:
        raise TickLogError(f"bad magic: {magic!r}")
    if version != VERSION:
        raise TickLogError(f"unsupported version: {version}")
    if record_size != RECORD_SIZE:
        raise TickLogError(f"record size mismatch: file={record_size}, expected={RECORD_SIZE}")

    last_seq: int | None = None
    while True:
        chunk = stream.read(RECORD_SIZE)
        if not chunk:
            return
        if len(chunk) != RECORD_SIZE:
            raise TickLogError("truncated final record")
        raw = _decode_record(chunk)
        if last_seq is not None and raw.sequence <= last_seq:
            raise TickLogError(f"non-increasing sequence: {last_seq} -> {raw.sequence}")
        last_seq = raw.sequence
        yield raw


def load_ticks(path: str | Path, *, source_id: str, symbol: str) -> list[NormalizedTick]:
    extender = TickCountExtender()
    out: list[NormalizedTick] = []
    with Path(path).open("rb") as f:
        for raw in iter_raw_ticks(f):
            try:
                t_host_ms = extender.extend(raw.host_tick_ms_raw)
                out.append(NormalizedTick.from_raw(raw, source_id=source_id, symbol=symbol, t_host_ms=t_host_ms))
            except ValueError as exc:
                raise TickLogError(str(exc)) from exc
    return out


def loads_ticks(data: bytes, *, source_id: str, symbol: str) -> list[NormalizedTick]:
    extender = TickCountExtender()
    out: list[NormalizedTick] = []
    for raw in iter_raw_ticks(io.BytesIO(data)):
        try:
            t_host_ms = extender.extend(raw.host_tick_ms_raw)
            out.append(NormalizedTick.from_raw(raw, source_id=source_id, symbol=symbol, t_host_ms=t_host_ms))
        except ValueError as exc:
            raise TickLogError(str(exc)) from exc
    return out
