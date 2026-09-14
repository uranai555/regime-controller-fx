from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .schema import NormalizedTick


@dataclass(frozen=True)
class HYResult:
    best_lag_ms: int
    best_contrast: float
    zero_lag_contrast: float
    observations_x: int
    observations_y: int

    @property
    def improvement_over_zero(self) -> float:
        base = abs(self.zero_lag_contrast)
        if base == 0:
            return float("inf") if self.best_contrast != 0 else 0.0
        return (self.best_contrast - self.zero_lag_contrast) / base


def _interval_returns(ticks: Sequence[NormalizedTick]) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    for a, b in zip(ticks, ticks[1:]):
        if b.t_host_ms <= a.t_host_ms:
            continue
        r = b.mid - a.mid
        if r != 0.0:
            out.append((a.t_host_ms, b.t_host_ms, r))
    return out


def hy_contrast(x_ticks: Sequence[NormalizedTick], y_ticks: Sequence[NormalizedTick], *, lag_ms: int = 0) -> float:
    """Hayashi–Yoshida-style asynchronous cross-covariance contrast.

    Positive ``lag_ms`` means X is hypothesized to lead Y by that many ms; Y's
    observation intervals are shifted earlier by ``lag_ms`` before overlap is
    evaluated. No regular-grid interpolation is used.

    Because the observations are interval-valued, the estimated optimum is only
    meaningful up to the native inter-tick resolution; use this as a validator of
    the event-matching estimate, not as the execution clock itself.
    """
    x = _interval_returns(x_ticks)
    y = _interval_returns(y_ticks)
    if not x or not y:
        return 0.0

    i = j = 0
    cov = 0.0
    while i < len(x) and j < len(y):
        xs, xe, xr = x[i]
        ys, ye, yr = y[j]
        ys -= lag_ms
        ye -= lag_ms
        if max(xs, ys) < min(xe, ye):
            cov += xr * yr
        if xe < ye:
            i += 1
        elif ye < xe:
            j += 1
        else:
            i += 1
            j += 1
    return cov


def estimate_hy_lead_lag(
    x_ticks: Sequence[NormalizedTick],
    y_ticks: Sequence[NormalizedTick],
    *,
    lag_grid_ms: Iterable[int] | None = None,
) -> HYResult:
    """Estimate X→Y lead/lag by maximizing asynchronous HY-style contrast."""
    grid = list(lag_grid_ms if lag_grid_ms is not None else range(-1000, 1001, 5))
    if not grid:
        raise ValueError("lag grid must not be empty")
    scored = [(lag, hy_contrast(x_ticks, y_ticks, lag_ms=lag)) for lag in grid]
    best_lag, best = max(scored, key=lambda item: (item[1], -abs(item[0]), -item[0]))
    zero = hy_contrast(x_ticks, y_ticks, lag_ms=0)
    return HYResult(
        best_lag_ms=best_lag,
        best_contrast=best,
        zero_lag_contrast=zero,
        observations_x=len(_interval_returns(x_ticks)),
        observations_y=len(_interval_returns(y_ticks)),
    )
