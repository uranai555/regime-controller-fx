"""execution_quality_model.py — Phase 3: ブローカー別 Execution Quality Model

broker_id ごとに slippage / reject / fill の履歴を蓄積し、
broker_execution_score (0-100) を算出する。

スコアは RegimeFeatures.broker_execution_score に書き込まれ、
RegimeController が既存の execution_score をブローカー品質で
上書き or ブレンドする際に使われる。

フォールバック:
  - broker_id が未指定 → Phase 1 の execution_score をそのまま使用
  - 履歴が min_records 未満 → デフォルト 80 点を返す
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRecord:
    """1回の約定記録。"""
    slippage: float = 0.0
    rejected: bool = False
    filled: bool = True
    timestamp_ms: int = 0


@dataclass
class ExecutionQualityConfig:
    """Execution Quality Model の閾値。"""
    min_records: int = 10
    default_score: float = 80.0

    # slippage thresholds (pips)
    slippage_good: float = 0.2
    slippage_warning: float = 0.5
    slippage_danger: float = 1.0

    # reject rate thresholds
    reject_rate_good: float = 0.01
    reject_rate_warning: float = 0.03
    reject_rate_danger: float = 0.05

    # fill rate thresholds
    fill_rate_good: float = 0.98
    fill_rate_warning: float = 0.95

    # history window
    max_history: int = 500


class ExecutionQualityModel:
    """ブローカー別の約定品質を追跡・評価するモデル。

    record() で履歴を蓄積し、score() でブローカー別スコアを返す。
    """

    def __init__(self, config: Optional[ExecutionQualityConfig] = None):
        self.cfg = config or ExecutionQualityConfig()
        self._history: Dict[str, List[ExecutionRecord]] = defaultdict(list)

    def record(
        self,
        broker_id: str,
        slippage: float = 0.0,
        rejected: bool = False,
        filled: bool = True,
        timestamp_ms: int = 0,
    ) -> None:
        """約定記録を追加する。"""
        rec = ExecutionRecord(
            slippage=slippage,
            rejected=rejected,
            filled=filled,
            timestamp_ms=timestamp_ms,
        )
        history = self._history[broker_id]
        history.append(rec)
        if len(history) > self.cfg.max_history:
            self._history[broker_id] = history[-self.cfg.max_history :]

    def score(self, broker_id: Optional[str]) -> Tuple[float, Dict[str, float]]:
        """ブローカーの execution_score を返す。

        Returns:
            (score, details) where details = {slippage_avg, reject_rate, fill_rate}
        """
        if broker_id is None or broker_id not in self._history:
            return self.cfg.default_score, {}

        history = self._history[broker_id]
        if len(history) < self.cfg.min_records:
            return self.cfg.default_score, {}

        # aggregate
        slippage_avg = sum(r.slippage for r in history) / len(history)
        reject_rate = sum(1 for r in history if r.rejected) / len(history)
        fill_rate = sum(1 for r in history if r.filled) / len(history)

        details = {
            "slippage_avg": round(slippage_avg, 4),
            "reject_rate": round(reject_rate, 4),
            "fill_rate": round(fill_rate, 4),
        }

        # scoring
        score = 100.0

        # slippage penalty
        if slippage_avg > self.cfg.slippage_danger:
            score -= 40
        elif slippage_avg > self.cfg.slippage_warning:
            score -= 20
        elif slippage_avg > self.cfg.slippage_good:
            score -= 10

        # reject rate penalty
        if reject_rate > self.cfg.reject_rate_danger:
            score -= 35
        elif reject_rate > self.cfg.reject_rate_warning:
            score -= 20
        elif reject_rate > self.cfg.reject_rate_good:
            score -= 10

        # fill rate penalty
        if fill_rate < self.cfg.fill_rate_warning:
            score -= 25
        elif fill_rate < self.cfg.fill_rate_good:
            score -= 10

        return max(0.0, min(100.0, score)), details

    def enrich_features(
        self,
        broker_id: Optional[str],
        features: "RegimeFeatures",  # noqa: F821 — forward ref
    ) -> None:
        """RegimeFeatures にブローカー別の実績を書き込む。"""
        broker_score, details = self.score(broker_id)
        features.broker_execution_score = broker_score
        if details:
            features.broker_slippage_avg = details.get("slippage_avg")
            features.broker_reject_rate = details.get("reject_rate")
            features.broker_fill_rate = details.get("fill_rate")

    def get_history_count(self, broker_id: str) -> int:
        return len(self._history.get(broker_id, []))

    def list_brokers(self) -> List[str]:
        return list(self._history.keys())

    def clear(self, broker_id: Optional[str] = None) -> None:
        """履歴をクリアする。broker_id=None で全クリア。"""
        if broker_id is None:
            self._history.clear()
        elif broker_id in self._history:
            del self._history[broker_id]

    def save(self, path: str) -> None:
        """ブローカー別履歴を JSON ファイルに保存する。"""
        data: Dict[str, List[Dict]] = {}
        for broker_id, records in self._history.items():
            data[broker_id] = [
                {
                    "slippage": r.slippage,
                    "rejected": r.rejected,
                    "filled": r.filled,
                    "timestamp_ms": r.timestamp_ms,
                }
                for r in records
            ]
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2))
        logger.info("ExecutionQualityModel saved to %s (%d brokers)", path, len(data))

    @classmethod
    def load(cls, path: str, config: Optional["ExecutionQualityConfig"] = None) -> Optional["ExecutionQualityModel"]:
        """JSON ファイルからモデルをロードする。ファイルがなければ None。"""
        p = Path(path)
        if not p.exists():
            logger.info("ExecutionQualityModel file not found: %s", path)
            return None
        try:
            data = json.loads(p.read_text())
            model = cls(config=config)
            for broker_id, records in data.items():
                for rec in records:
                    model.record(
                        broker_id=broker_id,
                        slippage=rec.get("slippage", 0.0),
                        rejected=rec.get("rejected", False),
                        filled=rec.get("filled", True),
                        timestamp_ms=rec.get("timestamp_ms", 0),
                    )
            return model
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to load ExecutionQualityModel from %s: %s", path, e)
            return None
