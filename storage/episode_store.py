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
from typing import Dict, Optional, Set

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
            # os.path.expandvars is intentional — same convention used by
            # AppConfig.ensure_dirs() for all other configured paths.
            Path(os.path.expandvars(storage_cfg.root_folder))
            if self._enabled
            else None
        )
        # Per-episode cache of known frame hashes: episode_dir → set of hashes.
        # Populated lazily on first access from the existing frames.jsonl file,
        # then kept in sync as new records are written.
        self._known_hashes: Dict[Path, Set[str]] = {}
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

    def _load_known_hashes(self, ep_dir: Path) -> Set[str]:
        """Return the set of frame_hash values already in frames.jsonl.

        Results are cached in ``self._known_hashes`` so the file is read at
        most once per episode per process lifetime.
        """
        if ep_dir in self._known_hashes:
            return self._known_hashes[ep_dir]
        hashes: Set[str] = set()
        frames_file = ep_dir / "frames.jsonl"
        if frames_file.exists():
            with frames_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        h = rec.get("frame_hash")
                        if h:
                            hashes.add(h)
                    except json.JSONDecodeError:
                        pass  # skip malformed lines
        self._known_hashes[ep_dir] = hashes
        return hashes

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
                # Skip if a file with the same name already exists — duplicate
                # screenshots from resumed sessions should not overwrite the
                # original copy (filenames are typically timestamp-based and
                # unique per capture, so collisions only happen on exact re-runs).
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

        # Deduplicate by frame_hash before writing.
        # frame_hash may be None for frames where hashing failed; those are
        # always written (None cannot be used as a stable dedup key).
        frames_file = ep_dir / "frames.jsonl"
        frame_hash = analysis.frame_hash
        known = self._load_known_hashes(ep_dir)
        if frame_hash and frame_hash in known:
            logger.debug(
                "episode_store.frame_hash_duplicate_skipped: hash=%s file=%s",
                frame_hash,
                source_path.name,
            )
            return

        # Append to frames.jsonl
        with frames_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Update the in-memory cache so repeated calls in the same session
        # also benefit from deduplication without re-reading the file.
        if frame_hash:
            known.add(frame_hash)

        logger.info(
            "episode_store.frame_exported: series=%s episode=%s file=%s",
            cluster.series_name,
            cluster.episode_label,
            source_path.name,
        )
