"""
Scene memory — accepts VisualAnalysis events and updates the context store.

This is the memory-first layer. The assistant is NEVER called here.
Only when a user query arrives does the ContextRetriever assemble a bundle.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from config.loader import AppConfig
from memory.context_store import ContextStore
from schemas.vision_schema import SceneState, VisualAnalysis

logger = logging.getLogger(__name__)


class SceneMemory:
    """Accepts analysis events and keeps the context store up to date."""

    def __init__(self, cfg: AppConfig, store: ContextStore) -> None:
        self._store = store
        self._cfg = cfg

    def ingest(self, analysis: VisualAnalysis) -> None:
        """
        Ingest a new VisualAnalysis into persistent memory.

        Steps
        -----
        1. Store the frame.
        2. Update subtitle history (deduplicated).
        3. Update the rolling scene state.
        4. Optionally add a scene-change summary entry.
        """
        logger.info("scene_memory.update: file=%s", analysis.source_file or "?")

        # 1. Store frame
        self._store.add_frame(analysis)

        # 2. Subtitle
        if analysis.subtitles:
            new_subtitle = self._store.add_subtitle(analysis.subtitles)
            if new_subtitle:
                logger.debug("scene_memory.new_subtitle: %r", analysis.subtitles[:80])
            else:
                logger.debug("scene_memory.subtitle_already_known: %r", analysis.subtitles[:60])

        # 3. Update rolling scene state
        current_state = self._store.get_scene_state() or SceneState()
        new_state = self._build_updated_state(current_state, analysis)
        self._store.update_scene_state(new_state)

        # 4. Scene history: add a compact entry when scene description changes meaningfully
        if self._is_significant_change(current_state, analysis):
            entry = self._make_history_entry(analysis)
            self._store.add_scene_history_entry(entry)
            logger.debug("scene_memory.history_entry_added")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_updated_state(
        self, current: SceneState, analysis: VisualAnalysis
    ) -> SceneState:
        """Merge new analysis into the rolling state."""
        # Scene summary: prefer latest non-empty
        scene_summary = analysis.scene or current.scene_summary

        # Important entities: merge with existing, deduplicated
        existing = set(current.important_entities)
        for item in analysis.important:
            existing.add(item)
        # Keep only the most recent N entities to avoid unbounded growth
        entities = list(existing)[-20:]

        # Subtitle summary: use last few unique subtitle lines
        recent_subs = self._store.get_recent_subtitles(limit=5)
        subtitle_summary = " | ".join(recent_subs) if recent_subs else ""

        return SceneState(
            scene_summary=scene_summary,
            important_entities=entities,
            subtitle_summary=subtitle_summary,
            last_updated=datetime.utcnow(),
        )

    def _is_significant_change(
        self, current: SceneState, analysis: VisualAnalysis
    ) -> bool:
        """Heuristic: is the new scene description meaningfully different?"""
        if not current.scene_summary:
            return True
        # Simple length-difference heuristic; could use fuzzy matching later
        old_words = set(current.scene_summary.lower().split())
        new_words = set(analysis.scene.lower().split())
        if not old_words:
            return True
        overlap = len(old_words & new_words) / max(len(old_words), len(new_words))
        return overlap < 0.5  # less than 50 % word overlap → significant change

    def _make_history_entry(self, analysis: VisualAnalysis) -> str:
        ts = analysis.timestamp.strftime("%H:%M:%S")
        parts = [f"[{ts}]", analysis.scene[:120]]
        if analysis.subtitles:
            parts.append(f"«{analysis.subtitles[:60]}»")
        return " ".join(parts)
