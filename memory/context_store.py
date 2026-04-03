"""
Scene memory / context store — SQLite-backed persistent memory layer.

This module is the heart of the memory-first architecture.
It accepts VisualAnalysis events and:
  - stores recent frame analyses
  - maintains a rolling scene state
  - tracks subtitle history (deduplicated)
  - stores compact scene-change summaries
  - prunes old entries automatically
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config.loader import AppConfig
from schemas.vision_schema import SceneState, VisualAnalysis

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS frames (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT    NOT NULL,
    data      TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS subtitles (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT    NOT NULL,
    text      TEXT    NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS scene_state (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    scene_summary  TEXT NOT NULL DEFAULT '',
    important      TEXT NOT NULL DEFAULT '[]',
    subtitle_sum   TEXT NOT NULL DEFAULT '',
    last_updated   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scene_history (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    summary TEXT NOT NULL
);
"""


class ContextStore:
    """SQLite-backed storage for frames, subtitles, scene state, and history."""

    def __init__(self, cfg: AppConfig) -> None:
        db_path = Path(cfg.paths.scene_memory_db)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(db_path), check_same_thread=False)
        self._con.executescript(_DDL)
        self._con.commit()
        self._max_frames = cfg.scene_memory.max_frames
        self._max_subtitles = cfg.scene_memory.max_subtitle_lines
        self._max_summaries = cfg.scene_memory.max_scene_summaries

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------

    def add_frame(self, analysis: VisualAnalysis) -> None:
        self._con.execute(
            "INSERT INTO frames (ts, data) VALUES (?, ?)",
            (analysis.timestamp.isoformat(), analysis.model_dump_json()),
        )
        self._prune_frames()
        self._con.commit()

    def get_recent_frames(self, limit: Optional[int] = None) -> List[VisualAnalysis]:
        n = limit or self._max_frames
        rows = self._con.execute(
            "SELECT data FROM frames ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        results = []
        for (data,) in reversed(rows):
            try:
                results.append(VisualAnalysis.model_validate_json(data))
            except Exception as exc:
                logger.warning("scene_memory.frame_parse_error: %s", exc)
        return results

    def _prune_frames(self) -> None:
        self._con.execute(
            "DELETE FROM frames WHERE id NOT IN "
            "(SELECT id FROM frames ORDER BY id DESC LIMIT ?)",
            (self._max_frames,),
        )

    # ------------------------------------------------------------------
    # Subtitles
    # ------------------------------------------------------------------

    def add_subtitle(self, text: str) -> bool:
        """Add a unique subtitle line. Returns True if it was new."""
        try:
            self._con.execute(
                "INSERT INTO subtitles (ts, text) VALUES (?, ?)",
                (datetime.utcnow().isoformat(), text.strip()),
            )
            self._prune_subtitles()
            self._con.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # duplicate

    def get_recent_subtitles(self, limit: Optional[int] = None) -> List[str]:
        n = limit or self._max_subtitles
        rows = self._con.execute(
            "SELECT text FROM subtitles ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [r[0] for r in reversed(rows)]

    def _prune_subtitles(self) -> None:
        self._con.execute(
            "DELETE FROM subtitles WHERE id NOT IN "
            "(SELECT id FROM subtitles ORDER BY id DESC LIMIT ?)",
            (self._max_subtitles,),
        )

    # ------------------------------------------------------------------
    # Rolling scene state
    # ------------------------------------------------------------------

    def update_scene_state(self, state: SceneState) -> None:
        self._con.execute(
            "INSERT INTO scene_state (id, scene_summary, important, subtitle_sum, last_updated) "
            "VALUES (1, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "  scene_summary = excluded.scene_summary, "
            "  important = excluded.important, "
            "  subtitle_sum = excluded.subtitle_sum, "
            "  last_updated = excluded.last_updated",
            (
                state.scene_summary,
                json.dumps(state.important_entities, ensure_ascii=False),
                state.subtitle_summary,
                state.last_updated.isoformat(),
            ),
        )
        self._con.commit()

    def get_scene_state(self) -> Optional[SceneState]:
        row = self._con.execute(
            "SELECT scene_summary, important, subtitle_sum, last_updated FROM scene_state WHERE id = 1"
        ).fetchone()
        if row is None:
            return None
        summary, important_json, subtitle_sum, updated = row
        return SceneState(
            scene_summary=summary,
            important_entities=json.loads(important_json),
            subtitle_summary=subtitle_sum,
            last_updated=datetime.fromisoformat(updated),
        )

    # ------------------------------------------------------------------
    # Scene history
    # ------------------------------------------------------------------

    def add_scene_history_entry(self, summary: str) -> None:
        self._con.execute(
            "INSERT INTO scene_history (ts, summary) VALUES (?, ?)",
            (datetime.utcnow().isoformat(), summary),
        )
        self._prune_history()
        self._con.commit()

    def get_scene_history(self, limit: Optional[int] = None) -> List[str]:
        n = limit or self._max_summaries
        rows = self._con.execute(
            "SELECT summary FROM scene_history ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [r[0] for r in reversed(rows)]

    def _prune_history(self) -> None:
        self._con.execute(
            "DELETE FROM scene_history WHERE id NOT IN "
            "(SELECT id FROM scene_history ORDER BY id DESC LIMIT ?)",
            (self._max_summaries,),
        )

    def close(self) -> None:
        self._con.close()
