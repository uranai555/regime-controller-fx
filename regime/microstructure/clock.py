from __future__ import annotations

from dataclasses import dataclass

UINT32_MOD = 2**32
UINT32_HALF = 2**31


@dataclass
class TickCountExtender:
    """Extend MQL4 GetTickCount() uint32 values to monotonic uint64 milliseconds.

    A wrap is accepted only when the previous value was in the upper half of the
    uint32 range and the new value is in the lower half. Smaller backward moves
    are treated as corruption/reordering and rejected.
    """

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
