"""Context retriever — assembles a ContextBundle from stored memory for assistant queries."""

from __future__ import annotations

import logging
from datetime import datetime

from config.loader import AppConfig
from memory.context_store import ContextStore
from schemas.vision_schema import ContextBundle, SceneState

logger = logging.getLogger(__name__)


class ContextRetriever:
    """Reads from ContextStore and assembles a ContextBundle on demand."""

    def __init__(self, cfg: AppConfig, store: ContextStore) -> None:
        self._store = store
        self._recent_frames_limit = min(5, cfg.scene_memory.max_frames)
        self._subtitle_limit = 10

    def retrieve(self) -> ContextBundle:
        """
        Build a ContextBundle from current stored memory.
        Called only when the user asks a question.
        """
        logger.info("context_retriever.retrieve: assembling context bundle")

        scene_state = self._store.get_scene_state() or SceneState()
        recent_frames = self._store.get_recent_frames(limit=self._recent_frames_limit)
        recent_subtitles = self._store.get_recent_subtitles(limit=self._subtitle_limit)
        history = self._store.get_scene_history(limit=5)
        history_summary = "\n".join(history) if history else None

        bundle = ContextBundle(
            rolling_state=scene_state,
            recent_frames=recent_frames,
            recent_subtitles=recent_subtitles,
            scene_history_summary=history_summary,
            query_timestamp=datetime.utcnow(),
        )

        logger.info(
            "context_retriever.bundle_ready: frames=%d subtitles=%d history_entries=%d",
            len(bundle.recent_frames),
            len(bundle.recent_subtitles),
            len(history),
        )
        return bundle
