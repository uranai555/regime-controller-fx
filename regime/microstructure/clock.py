from __future__ import annotations

from dataclasses import dataclass

UINT32_MOD = 2**32
UINT32_HALF = 2**31


@dataclass
class TickCountExtender:
    """Extend MQL4 GetTickCount() uint32 values to monotonic uint64 milliseconds."""

    last_raw: int | None = None
    wraps: int = 0

    def extend(self, raw: int) -> int:
        if not 0 <= raw < UINT32_MOD:
            raise ValueError("GetTickCount raw value must fit uint32")
        if self.last_raw is not None and raw < self.last_raw:
            if self.last_raw >= UINT32_HALF and raw < UINT32_HALF:
                self.wraps += 1
            else:
                raise ValueError(f"non-wrap backward clock jump: {self.last_raw} -> {raw}")
        self.last_raw = raw
        return raw + self.wraps * UINT32_MOD


def align_stream_wrap_epochs(streams):
    """Align independently loaded GetTickCount streams using TimeLocal anchors.

    Each file extends wraps relative to its own first record. If files start on
    opposite sides of the 49.7-day uint32 rollover, their local wrap epochs differ.
    The recorded wall-clock second identifies the relative integer multiple of
    2**32 ms. The earliest wall-clock stream is used as the arbitrary epoch zero.
    """
    from dataclasses import replace

    nonempty = {sid: list(ticks) for sid, ticks in streams.items() if ticks}
    if len(nonempty) != len(streams):
        raise ValueError("all streams must be non-empty")
    anchors = []
    for sid, ticks in nonempty.items():
        first = ticks[0]
        if first.t_local_s is None or first.t_host_raw_ms is None:
            continue
        anchors.append((first.t_local_s, sid, first))
    if not anchors:
        return nonempty
    if len(anchors) != len(nonempty):
        raise ValueError("cannot mix anchored and unanchored streams for epoch alignment")

    _, ref_sid, ref = min(anchors)
    ref_phase = ref.t_local_s * 1000 - ref.t_host_raw_ms
    out = {}
    for sid, ticks in nonempty.items():
        first = ticks[0]
        phase = first.t_local_s * 1000 - first.t_host_raw_ms
        relative_wraps = round((phase - ref_phase) / UINT32_MOD)
        offset = relative_wraps * UINT32_MOD
        out[sid] = [replace(t, t_host_ms=t.t_host_ms + offset) for t in ticks]
    return out
