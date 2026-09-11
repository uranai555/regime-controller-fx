from regime.microstructure.event_match import EventMatch, PriceEvent
from regime.microstructure.stability import bootstrap_median_ci, summarize_pair_stability


def ev(source, t, local_s=None):
    return PriceEvent(source, "XAUUSD", t, 1, .1, 100, 100.2, 100.1, .1, .1, local_s)


def match(i, lag=40, local_s=None):
    a = ev("A", i * 1000, local_s)
    b = ev("B", i * 1000 + lag, None)
    return EventMatch(a, b, lag)


def test_bootstrap_ci_deterministic_and_contains_median():
    ci1 = bootstrap_median_ci([35, 40, 40, 42, 45], resamples=200, seed=7)
    ci2 = bootstrap_median_ci([35, 40, 40, 42, 45], resamples=200, seed=7)
    assert ci1 == ci2
    assert ci1[0] <= 40 <= ci1[1]


def test_window_stability_and_hourly_stats():
    matches = [match(i, 40, local_s=14*3600 + i) for i in range(12)]
    s = summarize_pair_stability("A", "B", matches, window_ms=5000, min_events_per_window=3, bootstrap_resamples=100)
    assert s.windows >= 2
    assert s.positive_median_window_share == 1.0
    assert 14 in s.hourly
    assert s.hourly[14]["median_lag_ms"] == 40.0
