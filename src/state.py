"""Persistent state tracking.

Uses SQLite (via the stdlib ``sqlite3`` module) to store the last processed
frame information across restarts.  This allows the app to avoid duplicate
processing even if it is restarted between frames.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from src.logging_setup import get_logger

log: logging.Logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class StateStore:
    """Lightweight key-value persistent store backed by SQLite.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def last_file(self) -> Optional[str]:
        """Path of the last successfully processed file."""
        return self._get("last_file")

    @last_file.setter
    def last_file(self, value: str) -> None:
        self._set("last_file", value)

    @property
    def last_subtitle(self) -> Optional[str]:
        """Last subtitle text sent to the text model."""
        return self._get("last_subtitle")

    @last_subtitle.setter
    def last_subtitle(self, value: Optional[str]) -> None:
        self._set("last_subtitle", value or "")

    @property
    def last_scene(self) -> Optional[str]:
        """Last scene description sent to the text model."""
        return self._get("last_scene")

    @last_scene.setter
    def last_scene(self, value: str) -> None:
        self._set("last_scene", value)

    @property
    def last_assistant_response(self) -> Optional[str]:
        """Last internal assistant response from the text model."""
        return self._get("last_assistant_response")

    @last_assistant_response.setter
    def last_assistant_response(self, value: str) -> None:
        self._set("last_assistant_response", value)

    @property
    def last_scene_hash(self) -> Optional[str]:
        """Perceptual hash string of the last processed frame."""
        return self._get("last_scene_hash")

    @last_scene_hash.setter
    def last_scene_hash(self, value: str) -> None:
        self._set("last_scene_hash", value)

    def update_from_vision(
        self,
        file_path: str,
        subtitles: Optional[str],
        scene: str,
        assistant_response: str,
        scene_hash: str,
    ) -> None:
        """Atomically update all state fields after a successful pipeline run.

        Args:
            file_path: Path of the processed screenshot file.
            subtitles: Subtitle text or None.
            scene: Scene description.
            assistant_response: Text model response text.
            scene_hash: Perceptual hash string.
        """
        with self._connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                [
                    ("last_file", file_path),
                    ("last_subtitle", subtitles or ""),
                    ("last_scene", scene),
                    ("last_assistant_response", assistant_response),
                    ("last_scene_hash", scene_hash),
                ],
            )
        log.debug("State updated: file=%s", Path(file_path).name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.executescript(_SCHEMA)
        log.debug("State database initialised: %s", self._db_path)

    def _get(self, key: str) -> Optional[str]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT value FROM state WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def _set(self, key: str, value: Optional[str]) -> None:
        with self._connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                (key, value or ""),
            )

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._db_path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
