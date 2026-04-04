"""
Scene memory / context store — SQLite-backed persistent memory layer.

This module is the heart of the memory-first architecture.
It accepts VisualAnalysis events and:
  - stores recent frame analyses
  - maintains a rolling scene state
  - tracks subtitle history (deduplicated)
  - stores compact scene-change summaries
  - prunes old entries automatically

Memory is isolated by (series_name, episode_label) — the episode scope.
session_id is stored in every row as metadata/audit information but is NOT
used as a retrieval key.  Resuming the same episode in a new session
therefore reuses the same accumulated memory.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config.loader import AppConfig
from schemas.vision_schema import ClusterContext, SceneState, VisualAnalysis

logger = logging.getLogger(__name__)

# DDL for fresh installations.
# Primary isolation key: (series_name, episode_label).
# session_id is stored as metadata but is not part of any UNIQUE constraint.
_DDL_FRESH = """
CREATE TABLE IF NOT EXISTS frames (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    data          TEXT    NOT NULL,
    series_name   TEXT    NOT NULL DEFAULT 'default',
    episode_label TEXT    NOT NULL DEFAULT 'default',
    session_id    TEXT    NOT NULL DEFAULT 'default'
);
CREATE TABLE IF NOT EXISTS subtitles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    text          TEXT    NOT NULL,
    series_name   TEXT    NOT NULL DEFAULT 'default',
    episode_label TEXT    NOT NULL DEFAULT 'default',
    session_id    TEXT    NOT NULL DEFAULT 'default',
    UNIQUE(text, series_name, episode_label)
);
CREATE TABLE IF NOT EXISTS scene_state (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    series_name   TEXT    NOT NULL DEFAULT 'default',
    episode_label TEXT    NOT NULL DEFAULT 'default',
    session_id    TEXT    NOT NULL DEFAULT 'default',
    scene_summary TEXT    NOT NULL DEFAULT '',
    important     TEXT    NOT NULL DEFAULT '[]',
    subtitle_sum  TEXT    NOT NULL DEFAULT '',
    last_updated  TEXT    NOT NULL,
    UNIQUE(series_name, episode_label)
);
CREATE TABLE IF NOT EXISTS scene_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    summary       TEXT    NOT NULL,
    series_name   TEXT    NOT NULL DEFAULT 'default',
    episode_label TEXT    NOT NULL DEFAULT 'default',
    session_id    TEXT    NOT NULL DEFAULT 'default'
);
"""


def _migrate_schema(con: sqlite3.Connection) -> None:
    """
    Additive schema migration for pre-existing databases.

    Pass 1 — add missing columns (series_name / episode_label / session_id).
    Pass 2 — fix UNIQUE constraints that incorrectly included session_id:
              subtitles and scene_state are recreated if their stored SQL
              shows session_id inside a UNIQUE index.
    """

    def _column_exists(table: str, col: str) -> bool:
        rows = con.execute(f"PRAGMA table_info({table})").fetchall()
        return any(r[1] == col for r in rows)

    def _table_exists(table: str) -> bool:
        row = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        return row is not None

    def _table_sql(table: str) -> str:
        row = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        return row[0] if row else ""

    cluster_cols = [
        ("series_name", "TEXT NOT NULL DEFAULT 'default'"),
        ("episode_label", "TEXT NOT NULL DEFAULT 'default'"),
        ("session_id", "TEXT NOT NULL DEFAULT 'default'"),
    ]

    # --- Pass 1: add missing columns ---

    # frames / scene_history — simple additive column additions
    for table in ("frames", "scene_history"):
        if _table_exists(table):
            for col, col_def in cluster_cols:
                if not _column_exists(table, col):
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")
                    logger.info("schema_migration: added %s.%s", table, col)

    # subtitles — needs column additions if missing, handled via recreate below
    if _table_exists("subtitles") and not _column_exists("subtitles", "series_name"):
        con.executescript("""
            ALTER TABLE subtitles RENAME TO subtitles_old;
            CREATE TABLE subtitles (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            TEXT    NOT NULL,
                text          TEXT    NOT NULL,
                series_name   TEXT    NOT NULL DEFAULT 'default',
                episode_label TEXT    NOT NULL DEFAULT 'default',
                session_id    TEXT    NOT NULL DEFAULT 'default',
                UNIQUE(text, series_name, episode_label)
            );
            INSERT OR IGNORE INTO subtitles (id, ts, text)
                SELECT id, ts, text FROM subtitles_old;
            DROP TABLE subtitles_old;
        """)
        logger.info("schema_migration: added cluster columns to subtitles")

    # scene_state — pre-cluster had id=1 singleton; recreate with episode-scoped UNIQUE
    if _table_exists("scene_state") and not _column_exists("scene_state", "series_name"):
        con.executescript("""
            ALTER TABLE scene_state RENAME TO scene_state_old;
            CREATE TABLE scene_state (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                series_name   TEXT    NOT NULL DEFAULT 'default',
                episode_label TEXT    NOT NULL DEFAULT 'default',
                session_id    TEXT    NOT NULL DEFAULT 'default',
                scene_summary TEXT    NOT NULL DEFAULT '',
                important     TEXT    NOT NULL DEFAULT '[]',
                subtitle_sum  TEXT    NOT NULL DEFAULT '',
                last_updated  TEXT    NOT NULL,
                UNIQUE(series_name, episode_label)
            );
            INSERT OR IGNORE INTO scene_state
                (scene_summary, important, subtitle_sum, last_updated)
                SELECT scene_summary, important, subtitle_sum, last_updated
                FROM scene_state_old WHERE id = 1;
            DROP TABLE scene_state_old;
        """)
        logger.info("schema_migration: added cluster columns to scene_state")

    # --- Pass 2: fix session_id being part of UNIQUE constraints ---
    # This corrects databases created by the previous version of this code
    # where session_id was incorrectly included in the UNIQUE key.

    # subtitles: drop session_id from UNIQUE(text, series_name, episode_label, session_id)
    if _table_exists("subtitles"):
        sql = _table_sql("subtitles").lower()
        if "unique(text, series_name, episode_label, session_id)" in sql:
            con.executescript("""
                ALTER TABLE subtitles RENAME TO subtitles_old;
                CREATE TABLE subtitles (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts            TEXT    NOT NULL,
                    text          TEXT    NOT NULL,
                    series_name   TEXT    NOT NULL DEFAULT 'default',
                    episode_label TEXT    NOT NULL DEFAULT 'default',
                    session_id    TEXT    NOT NULL DEFAULT 'default',
                    UNIQUE(text, series_name, episode_label)
                );
                INSERT OR IGNORE INTO subtitles (id, ts, text, series_name, episode_label, session_id)
                    SELECT id, ts, text, series_name, episode_label, session_id
                    FROM subtitles_old;
                DROP TABLE subtitles_old;
            """)
            logger.info("schema_migration: removed session_id from subtitles UNIQUE constraint")

    # scene_state: drop session_id from UNIQUE(series_name, episode_label, session_id)
    if _table_exists("scene_state"):
        sql = _table_sql("scene_state").lower()
        if "unique(series_name, episode_label, session_id)" in sql:
            con.executescript("""
                ALTER TABLE scene_state RENAME TO scene_state_old;
                CREATE TABLE scene_state (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    series_name   TEXT    NOT NULL DEFAULT 'default',
                    episode_label TEXT    NOT NULL DEFAULT 'default',
                    session_id    TEXT    NOT NULL DEFAULT 'default',
                    scene_summary TEXT    NOT NULL DEFAULT '',
                    important     TEXT    NOT NULL DEFAULT '[]',
                    subtitle_sum  TEXT    NOT NULL DEFAULT '',
                    last_updated  TEXT    NOT NULL,
                    UNIQUE(series_name, episode_label)
                );
                INSERT OR IGNORE INTO scene_state
                    (series_name, episode_label, session_id,
                     scene_summary, important, subtitle_sum, last_updated)
                    SELECT series_name, episode_label, session_id,
                           scene_summary, important, subtitle_sum, last_updated
                    FROM scene_state_old;
                DROP TABLE scene_state_old;
            """)
            logger.info("schema_migration: removed session_id from scene_state UNIQUE constraint")

    con.commit()


class ContextStore:
    """SQLite-backed storage for frames, subtitles, scene state, and history.

    Memory is isolated by (series_name, episode_label).  session_id is
    written on every insert as metadata but is never used as a filter in
    SELECT / DELETE queries.  This means that resuming an episode in a new
    viewing session transparently reuses all previously accumulated memory.
    """

    def __init__(self, cfg: AppConfig) -> None:
        db_path = Path(cfg.paths.scene_memory_db)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(db_path), check_same_thread=False)
        # Fresh install: create tables with correct episode-scoped constraints.
        self._con.executescript(_DDL_FRESH)
        # Existing install: migrate old tables to episode-scoped schema.
        _migrate_schema(self._con)
        self._con.commit()
        self._max_frames = cfg.scene_memory.max_frames
        self._max_subtitles = cfg.scene_memory.max_subtitle_lines
        self._max_summaries = cfg.scene_memory.max_scene_summaries

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------

    def add_frame(self, analysis: VisualAnalysis, cluster: ClusterContext) -> None:
        self._con.execute(
            "INSERT INTO frames (ts, data, series_name, episode_label, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                analysis.timestamp.isoformat(),
                analysis.model_dump_json(),
                cluster.series_name,
                cluster.episode_label,
                cluster.session_id,
            ),
        )
        self._prune_frames(cluster)
        self._con.commit()

    def get_recent_frames(
        self, cluster: ClusterContext, limit: Optional[int] = None
    ) -> List[VisualAnalysis]:
        n = limit or self._max_frames
        rows = self._con.execute(
            "SELECT data FROM frames "
            "WHERE series_name = ? AND episode_label = ? "
            "ORDER BY id DESC LIMIT ?",
            (cluster.series_name, cluster.episode_label, n),
        ).fetchall()
        results = []
        for (data,) in reversed(rows):
            try:
                results.append(VisualAnalysis.model_validate_json(data))
            except Exception as exc:
                logger.warning("scene_memory.frame_parse_error: %s", exc)
        return results

    def _prune_frames(self, cluster: ClusterContext) -> None:
        ek = (cluster.series_name, cluster.episode_label)
        self._con.execute(
            "DELETE FROM frames "
            "WHERE series_name = ? AND episode_label = ? "
            "AND id NOT IN ("
            "  SELECT id FROM frames "
            "  WHERE series_name = ? AND episode_label = ? "
            "  ORDER BY id DESC LIMIT ?"
            ")",
            (*ek, *ek, self._max_frames),
        )

    # ------------------------------------------------------------------
    # Subtitles
    # ------------------------------------------------------------------

    def add_subtitle(self, text: str, cluster: ClusterContext) -> bool:
        """Add a unique subtitle line for the episode. Returns True if it was new."""
        try:
            self._con.execute(
                "INSERT INTO subtitles (ts, text, series_name, episode_label, session_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.utcnow().isoformat(),
                    text.strip(),
                    cluster.series_name,
                    cluster.episode_label,
                    cluster.session_id,
                ),
            )
            self._prune_subtitles(cluster)
            self._con.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # duplicate within this episode

    def get_recent_subtitles(
        self, cluster: ClusterContext, limit: Optional[int] = None
    ) -> List[str]:
        n = limit or self._max_subtitles
        rows = self._con.execute(
            "SELECT text FROM subtitles "
            "WHERE series_name = ? AND episode_label = ? "
            "ORDER BY id DESC LIMIT ?",
            (cluster.series_name, cluster.episode_label, n),
        ).fetchall()
        return [r[0] for r in reversed(rows)]

    def _prune_subtitles(self, cluster: ClusterContext) -> None:
        ek = (cluster.series_name, cluster.episode_label)
        self._con.execute(
            "DELETE FROM subtitles "
            "WHERE series_name = ? AND episode_label = ? "
            "AND id NOT IN ("
            "  SELECT id FROM subtitles "
            "  WHERE series_name = ? AND episode_label = ? "
            "  ORDER BY id DESC LIMIT ?"
            ")",
            (*ek, *ek, self._max_subtitles),
        )

    # ------------------------------------------------------------------
    # Rolling scene state
    # ------------------------------------------------------------------

    def update_scene_state(self, state: SceneState, cluster: ClusterContext) -> None:
        self._con.execute(
            "INSERT INTO scene_state "
            "  (series_name, episode_label, session_id, "
            "   scene_summary, important, subtitle_sum, last_updated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(series_name, episode_label) DO UPDATE SET "
            "  session_id    = excluded.session_id, "
            "  scene_summary = excluded.scene_summary, "
            "  important     = excluded.important, "
            "  subtitle_sum  = excluded.subtitle_sum, "
            "  last_updated  = excluded.last_updated",
            (
                cluster.series_name,
                cluster.episode_label,
                cluster.session_id,
                state.scene_summary,
                json.dumps(state.important_entities, ensure_ascii=False),
                state.subtitle_summary,
                state.last_updated.isoformat(),
            ),
        )
        self._con.commit()

    def get_scene_state(self, cluster: ClusterContext) -> Optional[SceneState]:
        row = self._con.execute(
            "SELECT scene_summary, important, subtitle_sum, last_updated "
            "FROM scene_state "
            "WHERE series_name = ? AND episode_label = ?",
            (cluster.series_name, cluster.episode_label),
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

    def add_scene_history_entry(self, summary: str, cluster: ClusterContext) -> None:
        self._con.execute(
            "INSERT INTO scene_history (ts, summary, series_name, episode_label, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.utcnow().isoformat(),
                summary,
                cluster.series_name,
                cluster.episode_label,
                cluster.session_id,
            ),
        )
        self._prune_history(cluster)
        self._con.commit()

    def get_scene_history(
        self, cluster: ClusterContext, limit: Optional[int] = None
    ) -> List[str]:
        n = limit or self._max_summaries
        rows = self._con.execute(
            "SELECT summary FROM scene_history "
            "WHERE series_name = ? AND episode_label = ? "
            "ORDER BY id DESC LIMIT ?",
            (cluster.series_name, cluster.episode_label, n),
        ).fetchall()
        return [r[0] for r in reversed(rows)]

    def _prune_history(self, cluster: ClusterContext) -> None:
        ek = (cluster.series_name, cluster.episode_label)
        self._con.execute(
            "DELETE FROM scene_history "
            "WHERE series_name = ? AND episode_label = ? "
            "AND id NOT IN ("
            "  SELECT id FROM scene_history "
            "  WHERE series_name = ? AND episode_label = ? "
            "  ORDER BY id DESC LIMIT ?"
            ")",
            (*ek, *ek, self._max_summaries),
        )

    def close(self) -> None:
        self._con.close()

