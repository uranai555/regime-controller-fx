"""broker_model.py — ブローカー別 Execution Quality Model

Phase 3: ブローカーごとの約定品質特性を学習し、execution_score の精度を向上。
ブローカーAでは安全でもブローカーBでは危険、というケースを検出する。

numpy は optional — 未インストール時は純粋な Python で EWMA 計算。
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BrokerModelConfig:
    """ブローカー品質モデルの設定。"""
    enabled: bool = True
    ewma_alpha: float = 0.05          # EWMA の減衰率
    min_samples: int = 20             # スコア算出に必要な最小サンプル数
    profile_path: Optional[str] = None  # プロファイル永続化先
    # hard block 閾値
    hard_block_quality_threshold: float = 20.0
    hard_block_latency_ms: float = 2000.0
    hard_block_reject_rate: float = 0.15


@dataclass
class BrokerProfile:
    """ブローカー固有の約定品質プロファイル。

    EWMA (Exponentially Weighted Moving Average) で更新。
    """
    broker_id: str
    slippage_mean: float = 0.0
    slippage_p95: float = 0.0
    reject_rate: float = 0.0
    fill_rate: float = 1.0
    requote_rate: float = 0.0
    latency_mean_ms: float = 0.0
    latency_p95_ms: float = 0.0
    spread_markup: float = 0.0
    # 時間帯別品質 (hour 0-23 → quality_factor 0.0-1.0)
    hourly_quality: Dict[int, float] = field(default_factory=dict)
    sample_count: int = 0
    # 内部: P95 計算用のローリングバッファ
    _slippage_buffer: List[float] = field(default_factory=list)
    _latency_buffer: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """シリアライズ用 dict 変換。"""
        return {
            "broker_id": self.broker_id,
            "slippage_mean": self.slippage_mean,
            "slippage_p95": self.slippage_p95,
            "reject_rate": self.reject_rate,
            "fill_rate": self.fill_rate,
            "requote_rate": self.requote_rate,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "spread_markup": self.spread_markup,
            "hourly_quality": {str(k): v for k, v in self.hourly_quality.items()},
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrokerProfile":
        """dict からプロファイルを復元。"""
        hourly = {int(k): v for k, v in data.get("hourly_quality", {}).items()}
        return cls(
            broker_id=data["broker_id"],
            slippage_mean=data.get("slippage_mean", 0.0),
            slippage_p95=data.get("slippage_p95", 0.0),
            reject_rate=data.get("reject_rate", 0.0),
            fill_rate=data.get("fill_rate", 1.0),
            requote_rate=data.get("requote_rate", 0.0),
            latency_mean_ms=data.get("latency_mean_ms", 0.0),
            latency_p95_ms=data.get("latency_p95_ms", 0.0),
            spread_markup=data.get("spread_markup", 0.0),
            hourly_quality=hourly,
            sample_count=data.get("sample_count", 0),
        )


@dataclass
class ExecutionEvent:
    """1回の約定イベント（プロファイル更新用入力）。"""
    broker_id: str
    slippage: float = 0.0       # pips
    latency_ms: float = 0.0     # ミリ秒
    rejected: bool = False
    filled: bool = True
    requoted: bool = False
    spread_at_execution: float = 0.0
    ecn_reference_spread: float = 0.0
    hour: Optional[int] = None  # 約定時の UTC hour


# ── Reason Code 定数 ──

RC_BROKER_QUALITY_LOW = "BROKER_QUALITY_LOW"
RC_BROKER_LATENCY_HIGH = "BROKER_LATENCY_HIGH"
RC_BROKER_REJECT_HIGH = "BROKER_REJECT_RATE_HIGH"
RC_BROKER_SLIPPAGE_HIGH = "BROKER_SLIPPAGE_HIGH"
RC_BROKER_HOUR_DEGRADED = "BROKER_HOUR_QUALITY_DEGRADED"
RC_BROKER_INSUFFICIENT_DATA = "BROKER_INSUFFICIENT_DATA"


_P95_BUFFER_SIZE = 100


class BrokerQualityModel:
    """ブローカー別の約定品質を学習・スコア化するモデル。

    Usage:
        model = BrokerQualityModel()
        model.update(ExecutionEvent(broker_id="BrokerA", slippage=0.3, ...))
        score, reasons = model.score("BrokerA", hour=14)
    """

    def __init__(self, config: Optional[BrokerModelConfig] = None):
        self.config = config or BrokerModelConfig()
        self._profiles: Dict[str, BrokerProfile] = {}

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def get_profile(self, broker_id: str) -> Optional[BrokerProfile]:
        """ブローカーのプロファイルを取得。"""
        return self._profiles.get(broker_id)

    def update(self, event: ExecutionEvent) -> None:
        """約定イベントでブローカープロファイルを更新する。

        EWMA (alpha) で滑らかに追従。P95 はローリングバッファで近似。
        """
        if not self.config.enabled:
            return

        profile = self._profiles.get(event.broker_id)
        if profile is None:
            profile = BrokerProfile(broker_id=event.broker_id)
            self._profiles[event.broker_id] = profile

        alpha = self.config.ewma_alpha
        profile.sample_count += 1

        # Slippage
        profile.slippage_mean = self._ewma(
            profile.slippage_mean, event.slippage, alpha
        )
        profile._slippage_buffer.append(event.slippage)
        if len(profile._slippage_buffer) > _P95_BUFFER_SIZE:
            profile._slippage_buffer = profile._slippage_buffer[-_P95_BUFFER_SIZE:]
        profile.slippage_p95 = self._percentile(profile._slippage_buffer, 0.95)

        # Latency
        profile.latency_mean_ms = self._ewma(
            profile.latency_mean_ms, event.latency_ms, alpha
        )
        profile._latency_buffer.append(event.latency_ms)
        if len(profile._latency_buffer) > _P95_BUFFER_SIZE:
            profile._latency_buffer = profile._latency_buffer[-_P95_BUFFER_SIZE:]
        profile.latency_p95_ms = self._percentile(profile._latency_buffer, 0.95)

        # Reject rate
        reject_val = 1.0 if event.rejected else 0.0
        profile.reject_rate = self._ewma(profile.reject_rate, reject_val, alpha)

        # Fill rate
        fill_val = 1.0 if event.filled else 0.0
        profile.fill_rate = self._ewma(profile.fill_rate, fill_val, alpha)

        # Requote rate
        requote_val = 1.0 if event.requoted else 0.0
        profile.requote_rate = self._ewma(profile.requote_rate, requote_val, alpha)

        # Spread markup
        if event.ecn_reference_spread > 0:
            markup = event.spread_at_execution - event.ecn_reference_spread
            profile.spread_markup = self._ewma(profile.spread_markup, markup, alpha)

        # Hourly quality
        if event.hour is not None:
            # 時間帯別の品質ファクター (0=bad, 1=good)
            event_quality = self._event_to_quality(event)
            prev = profile.hourly_quality.get(event.hour, 1.0)
            profile.hourly_quality[event.hour] = self._ewma(prev, event_quality, alpha)

    def score(
        self,
        broker_id: str,
        hour: Optional[int] = None,
    ) -> Tuple[float, List[str]]:
        """ブローカーの品質スコアを計算する。

        Returns:
            (score: 0-100, reason_codes: List[str])
        """
        reasons: List[str] = []

        profile = self._profiles.get(broker_id)
        if profile is None or profile.sample_count < self.config.min_samples:
            reasons.append(RC_BROKER_INSUFFICIENT_DATA)
            return 80.0, reasons  # デフォルト

        score = 100.0

        # Slippage penalty
        if profile.slippage_mean > 1.0:
            reasons.append(RC_BROKER_SLIPPAGE_HIGH)
            score -= 35
        elif profile.slippage_mean > 0.5:
            reasons.append(RC_BROKER_SLIPPAGE_HIGH)
            score -= 15

        # Reject rate penalty
        if profile.reject_rate > self.config.hard_block_reject_rate:
            reasons.append(RC_BROKER_REJECT_HIGH)
            score -= 40
        elif profile.reject_rate > 0.05:
            reasons.append(RC_BROKER_REJECT_HIGH)
            score -= 20

        # Latency penalty
        if profile.latency_p95_ms > self.config.hard_block_latency_ms:
            reasons.append(RC_BROKER_LATENCY_HIGH)
            score -= 30
        elif profile.latency_p95_ms > 500:
            reasons.append(RC_BROKER_LATENCY_HIGH)
            score -= 10

        # Requote penalty
        if profile.requote_rate > 0.05:
            score -= 15
        elif profile.requote_rate > 0.02:
            score -= 5

        # Spread markup penalty
        if profile.spread_markup > 0.5:
            score -= 10
        elif profile.spread_markup > 0.2:
            score -= 5

        # Hourly degradation
        if hour is not None and hour in profile.hourly_quality:
            hourly_factor = profile.hourly_quality[hour]
            if hourly_factor < 0.5:
                reasons.append(RC_BROKER_HOUR_DEGRADED)
                score -= 15
            elif hourly_factor < 0.7:
                reasons.append(RC_BROKER_HOUR_DEGRADED)
                score -= 5

        if score < self.config.hard_block_quality_threshold:
            reasons.append(RC_BROKER_QUALITY_LOW)

        return max(0.0, min(100.0, score)), reasons

    def save_profiles(self, path: Optional[str] = None) -> bool:
        """全ブローカープロファイルを JSON で永続化。"""
        save_path = path or self.config.profile_path
        if not save_path:
            return False

        try:
            data = {
                bid: profile.to_dict()
                for bid, profile in self._profiles.items()
            }
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info("Broker profiles saved: %d brokers → %s", len(data), save_path)
            return True

        except (OSError, IOError) as e:
            logger.warning("Broker profile save failed: %s", e)
            return False

    def load_profiles(self, path: Optional[str] = None) -> bool:
        """永続化された全プロファイルを復元。"""
        load_path = path or self.config.profile_path
        if not load_path:
            return False

        try:
            p = Path(load_path)
            if not p.exists():
                logger.warning("Broker profile file not found: %s", load_path)
                return False

            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)

            self._profiles = {
                bid: BrokerProfile.from_dict(profile_data)
                for bid, profile_data in data.items()
            }

            logger.info("Broker profiles loaded: %d brokers from %s", len(self._profiles), load_path)
            return True

        except (OSError, IOError, json.JSONDecodeError) as e:
            logger.warning("Broker profile load failed: %s", e)
            return False

    # ── Internal ──

    @staticmethod
    def _ewma(prev: float, new: float, alpha: float) -> float:
        """指数加重移動平均。"""
        return alpha * new + (1 - alpha) * prev

    @staticmethod
    def _percentile(values: List[float], pct: float) -> float:
        """ソート不要の簡易パーセンタイル計算。"""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * pct)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    @staticmethod
    def _event_to_quality(event: ExecutionEvent) -> float:
        """約定イベントを 0-1 の品質ファクターに変換。"""
        quality = 1.0

        # スリッページが大きいほど品質低下
        if event.slippage > 1.0:
            quality -= 0.4
        elif event.slippage > 0.5:
            quality -= 0.2

        # リジェクト
        if event.rejected:
            quality -= 0.5

        # レイテンシ
        if event.latency_ms > 1000:
            quality -= 0.3
        elif event.latency_ms > 500:
            quality -= 0.1

        return max(0.0, min(1.0, quality))
