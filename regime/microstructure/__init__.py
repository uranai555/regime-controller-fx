"""Passive broker microstructure research primitives (Phase 4A)."""

from .clock import TickCountExtender
from .event_match import EventMatch, PriceEvent, detect_price_events, match_events
from .fingerprint import BrokerFingerprint, build_fingerprint
from .friction import CostScenario, StressResult, default_scenarios, net_points, stress_markouts
from .ingest import TickLogError, encode_header, encode_record, load_ticks, loads_ticks
from .hy import HYResult, estimate_hy_lead_lag, hy_contrast
from .lead_lag import PairLagStats, PassiveMarkout, pairwise_lag_matrix, passive_markout, passive_stale_markout
from .schema import NormalizedTick, RawTick

__all__ = [
    "TickCountExtender", "RawTick", "NormalizedTick", "TickLogError",
    "encode_header", "encode_record", "load_ticks", "loads_ticks",
    "PriceEvent", "EventMatch", "detect_price_events", "match_events",
    "PairLagStats", "PassiveMarkout", "pairwise_lag_matrix", "passive_markout", "passive_stale_markout",
    "BrokerFingerprint", "build_fingerprint", "CostScenario", "StressResult", "default_scenarios", "net_points", "stress_markouts", "HYResult", "estimate_hy_lead_lag", "hy_contrast",
]
