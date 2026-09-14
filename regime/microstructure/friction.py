from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from statistics import mean, median, stdev
from typing import Iterable, Sequence

from .lead_lag import PassiveMarkout


@dataclass(frozen=True)
class CostScenario:
    """Transaction-cost stress scenario expressed in source-broker points.

    ``slippage_spread_multiple`` applies a conservative slippage charge equal to
    a multiple of the observed entry spread. Entry spread has already affected
    the gross markout through bid/ask execution; this term is an *additional*
    adverse-fill stress, not a second spread charge.
    """

    name: str
    commission_points: float = 0.0
    slippage_spread_multiple: float = 0.0
    cashback_points: float = 0.0


@dataclass(frozen=True)
class StressResult:
    scenario: str
    observations: int
    mean_net_points: float
    median_net_points: float
    positive_rate: float
    p05_net_points: float
    p95_net_points: float
    mean_net_ci95_lo_points: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _pct(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _mean_ci95_lower(values: Sequence[float]) -> float | None:
    """Approximate two-sided 95% lower confidence bound for the mean.

    This is intentionally paired with a minimum-observation gate downstream;
    for n < 2 there is no usable dispersion estimate and the bound is None.
    """
    if len(values) < 2:
        return None
    mu = mean(values)
    se = stdev(values) / sqrt(len(values))
    return mu - 1.96 * se


def net_points(markout: PassiveMarkout, scenario: CostScenario) -> float:
    if scenario.commission_points < 0 or scenario.slippage_spread_multiple < 0:
        raise ValueError("commission/slippage stress must be non-negative")
    return (
        markout.gross_markout_points
        - scenario.commission_points
        - scenario.slippage_spread_multiple * markout.entry_spread_points
        + scenario.cashback_points
    )


def stress_markouts(markouts: Sequence[PassiveMarkout], scenarios: Iterable[CostScenario]) -> list[StressResult]:
    results: list[StressResult] = []
    for scenario in scenarios:
        values = [net_points(m, scenario) for m in markouts]
        if not values:
            results.append(StressResult(scenario.name, 0, 0.0, 0.0, 0.0, 0.0, 0.0, None))
            continue
        results.append(
            StressResult(
                scenario=scenario.name,
                observations=len(values),
                mean_net_points=mean(values),
                median_net_points=median(values),
                positive_rate=sum(v > 0 for v in values) / len(values),
                p05_net_points=_pct(values, 0.05),
                p95_net_points=_pct(values, 0.95),
                mean_net_ci95_lo_points=_mean_ci95_lower(values),
            )
        )
    return results


def default_scenarios(*, commission_points: float = 0.0, cashback_points: float = 0.0) -> list[CostScenario]:
    return [
        CostScenario("no_extra_slippage", commission_points, 0.0, cashback_points),
        CostScenario("slippage_0.5x_spread", commission_points, 0.5, cashback_points),
        CostScenario("slippage_1.0x_spread", commission_points, 1.0, cashback_points),
        CostScenario("slippage_2.0x_spread", commission_points, 2.0, cashback_points),
    ]
