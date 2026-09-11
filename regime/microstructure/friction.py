from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, median
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
            results.append(StressResult(scenario.name, 0, 0.0, 0.0, 0.0, 0.0, 0.0))
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
