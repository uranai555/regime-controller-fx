from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawTick:
    sequence: int
    host_tick_ms_raw: int
    program_us: int
    server_time_sec: int | None
    local_time_sec: int | None
    bid: float
    ask: float
    last: float | None
    volume: int | None
    point: float
    digits: int
    flags: int = 0


@dataclass(frozen=True)
class NormalizedTick:
    source_id: str
    symbol: str
    seq: int
    t_host_ms: int
    t_server_s: int | None
    bid: float
    ask: float
    mid: float
    spread: float
    spread_points: float
    point: float
    t_local_s: int | None = None
    t_host_raw_ms: int | None = None

    @classmethod
    def from_raw(cls, raw: RawTick, *, source_id: str, symbol: str, t_host_ms: int) -> "NormalizedTick":
        if raw.point <= 0:
            raise ValueError("point must be > 0")
        if not (raw.bid > 0 and raw.ask > 0 and raw.ask >= raw.bid):
            raise ValueError("invalid bid/ask")
        spread = raw.ask - raw.bid
        return cls(
            source_id=source_id,
            symbol=symbol,
            seq=raw.sequence,
            t_host_ms=t_host_ms,
            t_server_s=raw.server_time_sec,
            bid=raw.bid,
            ask=raw.ask,
            mid=(raw.bid + raw.ask) / 2.0,
            spread=spread,
            spread_points=spread / raw.point,
            point=raw.point,
            t_local_s=raw.local_time_sec,
            t_host_raw_ms=raw.host_tick_ms_raw,
        )
