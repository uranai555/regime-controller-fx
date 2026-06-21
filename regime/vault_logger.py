"""vault_logger.py — RegimeDecision を Vault に JSONL 保存する

Phase 1 では vault/regime/ 以下に datetime 別の JSONL ファイルを出力する。
CSV 出力は後続 Phase で追加可能。
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from regime.types import RegimeDecision


class RegimeVaultLogger:
    """RegimeDecision を JSONL で Vault に保存する。

    Args:
        vault_root: Vault のルートディレクトリ
        enabled: ログ出力の ON/OFF
    """

    def __init__(
        self,
        vault_root: str = "",
        enabled: bool = True,
    ):
        self.vault_root = vault_root
        self.enabled = enabled
        self._file: Optional[Path] = None

    def _ensure_file(self) -> Path:
        today = datetime.now(timezone.utc).strftime("%Y%m%d")

        if self.vault_root:
            base = Path(self.vault_root) / "regime"
        else:
            # 環境変数 OBSIDIAN_VAULT またはデフォルトパス
            base = Path(os.environ.get("OBSIDIAN_VAULT", str(Path.home() / ".hermes" / "vault"))) / "regime"

        base.mkdir(parents=True, exist_ok=True)
        path = base / f"regime_decisions_{today}.jsonl"

        if self._file != path:
            self._file = path  # 日付が変わったらファイルを切り替え

        return path

    def write(self, decision: RegimeDecision) -> bool:
        """1行の JSONL を書き込む。"""
        if not self.enabled:
            return False

        try:
            path = self._ensure_file()

            record = {
                "timestamp": decision.timestamp or "",
                "symbol": decision.symbol or "",
                "mode": decision.mode.value,
                "raw_mode": decision.raw_mode.value,
                "cb_run_score": decision.cb_run_score,
                "sub_scores": decision.sub_scores,
                "reason_codes": decision.reason_codes,
                "allow_new_entry": decision.allow_new_entry,
                "allow_add_position": decision.allow_add_position,
                "reduce_only": decision.reduce_only,
                "force_exit": decision.force_exit,
                "risk_multiplier": decision.risk_multiplier,
                "features": decision.features,
                "missing_fields": decision.missing_fields,
            }

            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

            return True
        except (OSError, IOError) as e:
            logger.warning("Vault write failed: %s", e)
            return False

    def flush(self) -> None:
        """明示的なフラッシュ（必要に応じて）。"""
        pass  # 都度 open/write/close しているので不要
