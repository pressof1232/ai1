"""
Episode-aware filesystem storage.

Creates and manages a physical folder structure rooted at
``anime_storage.root_folder``:

    <root>/
      <series_name>/
        <episode_label>/
          screenshots/   ← accepted screenshot files
          metadata/      ← future: per-episode metadata blobs
          frames.jsonl   ← one JSON record per accepted frame

Folder identity is based exclusively on (series_name, episode_label).
Resuming the same episode in a new session reuses the same folder;
session_id is stored inside each record as metadata only.

Usage::

    store = EpisodeStore(cfg)
    store.export_frame(analysis, cluster, source_path)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional

from config.loader import AppConfig
from schemas.vision_schema import ClusterContext, VisualAnalysis

logger = logging.getLogger(__name__)

# Characters that are illegal in Windows/Linux folder names.
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_name(name: str) -> str:
    """Replace filesystem-unsafe characters with underscores."""
    return _UNSAFE_CHARS.sub("_", name).strip(". ") or "default"


class EpisodeStore:
    """Manages the physical episode folder structure and per-frame exports.

    If ``anime_storage.enabled`` is False or ``root_folder`` is empty,
    all methods are no-ops so the pipeline is never affected.
    """

    def __init__(self, cfg: AppConfig) -> None:
        storage_cfg = cfg.anime_storage
        self._enabled = storage_cfg.enabled and bool(storage_cfg.root_folder)
        self._root: Optional[Path] = (
            Path(os.path.expandvars(storage_cfg.root_folder))
            if self._enabled
            else None
        )
        if self._enabled:
            logger.info("episode_store.enabled: root=%s", self._root)
        else:
            logger.debug("episode_store.disabled")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def episode_dir(self, cluster: ClusterContext) -> Optional[Path]:
        """Return the episode directory path (does NOT create it)."""
        if not self._enabled or self._root is None:
            return None
        return (
            self._root
            / _safe_name(cluster.series_name)
            / _safe_name(cluster.episode_label)
        )

    def ensure_episode_dirs(self, cluster: ClusterContext) -> Optional[Path]:
        """Create episode subdirectories and return the episode root path."""
        ep_dir = self.episode_dir(cluster)
        if ep_dir is None:
            return None
        (ep_dir / "screenshots").mkdir(parents=True, exist_ok=True)
        (ep_dir / "metadata").mkdir(parents=True, exist_ok=True)
        return ep_dir

    def export_frame(
        self,
        analysis: VisualAnalysis,
        cluster: ClusterContext,
        source_path: Path,
    ) -> None:
        """Copy the screenshot and append a JSONL record for this frame.

        Safe to call at all times — exceptions are caught and logged so the
        main pipeline is never interrupted.
        """
        if not self._enabled:
            return
        try:
            self._export(analysis, cluster, source_path)
        except Exception as exc:
            logger.warning(
                "episode_store.export_failed: file=%s error=%s",
                source_path.name,
                exc,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _export(
        self,
        analysis: VisualAnalysis,
        cluster: ClusterContext,
        source_path: Path,
    ) -> None:
        ep_dir = self.ensure_episode_dirs(cluster)
        if ep_dir is None:
            return

        # Copy screenshot (skip if source no longer exists, e.g. test mode)
        screenshots_dir = ep_dir / "screenshots"
        if source_path.exists():
            dest = screenshots_dir / source_path.name
            if not dest.exists():
                shutil.copy2(source_path, dest)
                logger.debug(
                    "episode_store.screenshot_copied: %s → %s", source_path.name, dest
                )

        # Build the frame record
        record = {
            "source_file": analysis.source_file or source_path.name,
            "timestamp": analysis.timestamp.isoformat(),
            "series_name": cluster.series_name,
            "episode_label": cluster.episode_label,
            "session_id": cluster.session_id,
            "subtitles": analysis.subtitles,
            "scene": analysis.scene,
            "important": analysis.important,
            "uncertainty": analysis.uncertainty,
            "image_quality_note": analysis.image_quality_note,
            "frame_hash": analysis.frame_hash,
        }

        # Append to frames.jsonl
        frames_file = ep_dir / "frames.jsonl"
        with frames_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            "episode_store.frame_exported: series=%s episode=%s file=%s",
            cluster.series_name,
            cluster.episode_label,
            source_path.name,
        )
