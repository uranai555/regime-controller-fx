import pytest

from regime.microstructure.friction import CostScenario, default_scenarios, net_points, stress_markouts
from regime.microstructure.lead_lag import PassiveMarkout


def m(gross, spread=2.0):
    return PassiveMarkout("B", 100, 1, 100, gross, gross, spread)


def test_cost_stress_accounts_for_commission_slippage_and_cashback():
    scenario = CostScenario("s", commission_points=0.3, slippage_spread_multiple=0.5, cashback_points=0.4)
    assert net_points(m(2.0, 2.0), scenario) == pytest.approx(1.1)


def test_default_stress_matrix_gets_monotonically_harder():
    results = stress_markouts([m(3.0), m(1.0), m(-1.0)], default_scenarios(commission_points=.2, cashback_points=.1))
    means = [r.mean_net_points for r in results]
    assert means == sorted(means, reverse=True)
    assert results[-1].positive_rate <= results[0].positive_rate


def test_negative_stress_rejected():
    with pytest.raises(ValueError):
        net_points(m(1.0), CostScenario("bad", commission_points=-1))
