import random

from regime.microstructure.hy import estimate_hy_lead_lag
from regime.microstructure.schema import NormalizedTick


def nt(source, seq, t, mid):
    spread = 0.02
    point = 0.01
    return NormalizedTick(source, "XAUUSD", seq, t, None, mid-spread/2, mid+spread/2, mid, spread, 2.0, point)


def series(source, delay, mids):
    times = []
    t = 0
    for i in range(len(mids)):
        t += 7 + (i % 5) * 3
        times.append(t + delay)
    return [nt(source, i, t, m) for i, (t, m) in enumerate(zip(times, mids))]


def test_hy_validator_recovers_known_delay_within_tick_resolution():
    rng = random.Random(7)
    price = 100.0
    mids = [price]
    for _ in range(300):
        price += rng.choice([-0.03, -0.02, 0.02, 0.03])
        mids.append(round(price, 5))
    a = series("A", 0, mids)
    b = series("B", 40, mids)
    result = estimate_hy_lead_lag(a, b, lag_grid_ms=range(-100, 101))
    assert abs(result.best_lag_ms - 40) <= 10
    assert result.best_contrast > result.zero_lag_contrast
