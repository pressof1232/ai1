"""Local persistent state tracking (last file, last hash, cooldowns)."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.loader import AppConfig

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS local_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class LocalState:
    """Lightweight SQLite-backed key-value store for pipeline state."""

    def __init__(self, cfg: AppConfig) -> None:
        db_path = Path(cfg.paths.state_db)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(db_path), check_same_thread=False)
        self._con.execute(_DDL)
        self._con.commit()

    # ------------------------------------------------------------------
    # Generic key/value helpers
    # ------------------------------------------------------------------

    def _get(self, key: str) -> Optional[str]:
        row = self._con.execute(
            "SELECT value FROM local_state WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def _set(self, key: str, value: str) -> None:
        self._con.execute(
            "INSERT INTO local_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._con.commit()

    # ------------------------------------------------------------------
    # Domain accessors
    # ------------------------------------------------------------------

    @property
    def last_processed_file(self) -> Optional[str]:
        return self._get("last_processed_file")

    @last_processed_file.setter
    def last_processed_file(self, value: str) -> None:
        self._set("last_processed_file", value)

    @property
    def last_frame_hash(self) -> Optional[str]:
        return self._get("last_frame_hash")

    @last_frame_hash.setter
    def last_frame_hash(self, value: str) -> None:
        self._set("last_frame_hash", value)

    @property
    def last_subtitle(self) -> Optional[str]:
        return self._get("last_subtitle")

    @last_subtitle.setter
    def last_subtitle(self, value: str) -> None:
        self._set("last_subtitle", value)

    @property
    def last_accepted_timestamp(self) -> Optional[datetime]:
        raw = self._get("last_accepted_timestamp")
        if raw is None:
            return None
        return datetime.fromisoformat(raw)

    @last_accepted_timestamp.setter
    def last_accepted_timestamp(self, value: datetime) -> None:
        self._set("last_accepted_timestamp", value.isoformat())

    @property
    def last_subtitle_timestamp(self) -> Optional[datetime]:
        raw = self._get("last_subtitle_timestamp")
        if raw is None:
            return None
        return datetime.fromisoformat(raw)

    @last_subtitle_timestamp.setter
    def last_subtitle_timestamp(self, value: datetime) -> None:
        self._set("last_subtitle_timestamp", value.isoformat())

    def close(self) -> None:
        self._con.close()
