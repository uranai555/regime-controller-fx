"""persistence_filter.py — レジーム判定の flickering を防ぐフィルター

設計方針:
- より危険なモードへの遷移: 即時（1バーで反映）
- より安全なモードへの遷移: confirm_bars 回連続で同じ判定が出たら反映
- 同じモードが続く限りカウンターは進む
- raw_mode が変わったらカウンターリセット
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

from regime.types import RegimeMode, _RISK_ORDER


@dataclass
class PersistenceFilterState:
    """PersistenceFilter の内部状態（シリアライズ可能）。"""
    current_mode: RegimeMode = RegimeMode.NORMAL
    pending_mode: Optional[RegimeMode] = None
    pending_count: int = 0


class PersistenceFilter:
    """レジームモードの flickering を防ぐ非対称フィルター。

    Args:
        confirm_bars: 安全方向への遷移を確定させるのに必要な連続確認数
    """

    def __init__(self, confirm_bars: int = 3):
        if confirm_bars < 1:
            raise ValueError(f"confirm_bars must be >= 1, got {confirm_bars}")
        self.confirm_bars = confirm_bars
        self._state = PersistenceFilterState()

    @property
    def current_mode(self) -> RegimeMode:
        return self._state.current_mode

    def reset(self, mode: RegimeMode = RegimeMode.NORMAL) -> None:
        """状態をリセットする（新規バックテスト開始時など）。"""
        self._state = PersistenceFilterState(current_mode=mode)

    def update(self, raw_mode: RegimeMode) -> RegimeMode:
        """新しい raw_mode を受け取り、確定モードを返す。

        Returns:
            RegimeMode — フィルター適用後の確定モード
        """
        current_level = _RISK_ORDER[self._state.current_mode]
        raw_level = _RISK_ORDER[raw_mode]

        if raw_level > current_level:
            # ── 危険方向: 即時反映 ──
            logger.info(
                "Immediate escalation: %s → %s",
                self._state.current_mode.value, raw_mode.value,
            )
            self._state.current_mode = raw_mode
            self._state.pending_mode = None
            self._state.pending_count = 0
            return self._state.current_mode

        if raw_level == current_level:
            # ── 同レベル: ペンディングキャンセル（既に同じ状態にいる） ──
            self._state.pending_mode = None
            self._state.pending_count = 0
            return self._state.current_mode

        # ── 安全方向: confirm_bars 回連続で確認 ──
        if self._state.pending_mode == raw_mode:
            self._state.pending_count += 1
        else:
            self._state.pending_mode = raw_mode
            self._state.pending_count = 1

        if self._state.pending_count >= self.confirm_bars:
            logger.info(
                "Safe transition confirmed (%d bars): %s → %s",
                self.confirm_bars,
                self._state.current_mode.value,
                raw_mode.value,
            )
            self._state.current_mode = raw_mode
            self._state.pending_mode = None
            self._state.pending_count = 0

        return self._state.current_mode

    def state_report(self) -> PersistenceFilterState:
        """現在の内部状態を取得（ログ・デバッグ用）。"""
        return PersistenceFilterState(
            current_mode=self._state.current_mode,
            pending_mode=self._state.pending_mode,
            pending_count=self._state.pending_count,
        )