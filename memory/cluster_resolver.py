"""
Cluster resolver — determines the active ClusterContext from config and optional path hints.

Usage:
    cluster = resolve_startup_cluster(cfg)            # at startup
    cluster = resolve_cluster_for_path(cfg, path, base_cluster)  # per screenshot
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.loader import AppConfig
from schemas.vision_schema import ClusterContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Episode pattern matchers (ordered from most specific to least)
# ---------------------------------------------------------------------------

_RE_SXEX = re.compile(r"[Ss](\d{1,2})[Ee](\d{1,3})")          # S01E03
_RE_EPISODE_WORD = re.compile(r"episode[-_\s]?(\d{1,3})", re.IGNORECASE)  # episode-03
_RE_EP_SHORT = re.compile(r"\bep[-_]?(\d{1,3})\b", re.IGNORECASE)         # ep03


def parse_cluster_from_path(path: Path) -> Optional[ClusterContext]:
    """
    Lightweight heuristic parser: infer series/episode from a file path.

    Returns a ClusterContext if an episode label can be detected, else None.
    series_name is inferred from a parent directory name when available.
    session_id is always left as "default" here; startup resolution sets it.
    """
    # Build a single string from all path components for pattern matching
    parts = path.parts
    combined = "/".join(str(p) for p in parts)

    episode_label: Optional[str] = None

    # Try S01E03 style first
    m = _RE_SXEX.search(combined)
    if m:
        season = m.group(1).zfill(2)
        episode = m.group(2).zfill(2)
        episode_label = f"S{season}E{episode}"
    else:
        # Try "episode-N" style
        m = _RE_EPISODE_WORD.search(combined)
        if m:
            episode_label = f"EP{m.group(1).zfill(2)}"
        else:
            # Try "ep03" style
            m = _RE_EP_SHORT.search(combined)
            if m:
                episode_label = f"EP{m.group(1).zfill(2)}"

    if episode_label is None:
        return None  # could not determine episode — do not guess

    # Try to infer series name from a parent directory that is not the episode folder
    series_name = "default"
    episode_folder_re = re.compile(
        r"^([Ss]\d+[Ee]\d+|[Ee][Pp]\d+|[Ee]pisode[-_\s]?\d+)$", re.IGNORECASE
    )
    for part in reversed(parts[:-1]):  # walk from filename upward, skip filename
        if not part or part in (".", "..") or part.startswith("."):
            continue
        if episode_folder_re.match(part):
            continue  # this part IS the episode folder label
        # Accept it as series name if it looks like a real name (not a drive or root)
        if len(part) >= 2 and not re.match(r"^[A-Za-z]:\\?$", part) and part != "/":
            series_name = part
            break

    logger.debug(
        "cluster_resolver.path_parsed: episode=%s series=%s path=%s",
        episode_label,
        series_name,
        path,
    )
    return ClusterContext(
        series_name=series_name,
        episode_label=episode_label,
        session_id="default",  # session is set by startup resolver
    )


def resolve_startup_cluster(cfg: AppConfig) -> ClusterContext:
    """
    Resolve the active ClusterContext at startup.

    Reads series_name and episode_label from config.session.
    Generates or uses the configured session_id.
    """
    session_cfg = cfg.session

    series_name = session_cfg.series_name or "default"
    episode_label = session_cfg.episode_label or "default"
    session_id = session_cfg.session_id or ""

    if not session_id:
        if session_cfg.auto_generate_session_id:
            session_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
            logger.debug("cluster_resolver.session_id_generated: %s", session_id)
        else:
            session_id = "default"

    cluster = ClusterContext(
        series_name=series_name,
        episode_label=episode_label,
        session_id=session_id,
    )
    logger.info("cluster_resolver.active_cluster: %s", cluster)
    return cluster


def resolve_cluster_for_path(
    cfg: AppConfig,
    path: Path,
    base_cluster: ClusterContext,
) -> ClusterContext:
    """
    Resolve the cluster to use when storing a specific screenshot.

    If parse_from_path is enabled in config, attempts to infer series/episode
    from the file path and merges the result with base_cluster (keeping
    base_cluster's session_id in all cases).

    Falls back to base_cluster if parsing fails or is disabled.
    """
    if not cfg.session.parse_from_path:
        return base_cluster

    parsed = parse_cluster_from_path(path)
    if parsed is None:
        return base_cluster

    # Merge: parsed values override base only when non-default
    series_name = (
        parsed.series_name
        if parsed.series_name != "default"
        else base_cluster.series_name
    )
    episode_label = (
        parsed.episode_label
        if parsed.episode_label != "default"
        else base_cluster.episode_label
    )

    if series_name == base_cluster.series_name and episode_label == base_cluster.episode_label:
        return base_cluster  # nothing new — reuse the base object

    merged = ClusterContext(
        series_name=series_name,
        episode_label=episode_label,
        session_id=base_cluster.session_id,
    )
    logger.debug("cluster_resolver.path_cluster: %s → %s", path.name, merged)
    return merged
