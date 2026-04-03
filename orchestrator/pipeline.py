"""
Main orchestration pipeline.

Wires all stages together:
  watcher → preprocessor → deduplicator → vision → scene memory
  (user query) → context retriever → assistant sink
"""

from __future__ import annotations

import logging
from pathlib import Path

from assistant.anythingllm_sink import AnythingLLMSink
from assistant.base_sink import AssistantSink, HandoffUnavailableError
from assistant.ollama_text_sink import OllamaTextSink
from config.loader import AppConfig
from dedup.deduplicator import Deduplicator
from memory.context_retriever import ContextRetriever
from memory.context_store import ContextStore
from memory.scene_memory import SceneMemory
from preprocessing.image_processor import ImageProcessor
from state.local_state import LocalState
from vision.ollama_vision_client import OllamaVisionClient
from watcher.screenshot_watcher import ScreenshotWatcher

logger = logging.getLogger(__name__)


class Pipeline:
    """Full VLC → memory pipeline with user query support."""

    def __init__(self, cfg: AppConfig) -> None:
        cfg.ensure_dirs()
        self._cfg = cfg

        self._state = LocalState(cfg)
        self._store = ContextStore(cfg)
        self._processor = ImageProcessor(cfg)
        self._dedup = Deduplicator(cfg, self._state)
        self._vision = OllamaVisionClient(cfg)
        self._scene_memory = SceneMemory(cfg, self._store)
        self._retriever = ContextRetriever(cfg, self._store)

        self._preferred_sink: AssistantSink = AnythingLLMSink(cfg)
        self._fallback_sink: AssistantSink = OllamaTextSink(cfg)

    # ------------------------------------------------------------------
    # Watcher entry point
    # ------------------------------------------------------------------

    async def run_forever(self) -> None:
        watcher = ScreenshotWatcher(self._cfg, self._on_new_screenshot)
        await watcher.run_forever()

    # ------------------------------------------------------------------
    # Screenshot processing (called by watcher for each stable file)
    # ------------------------------------------------------------------

    async def _on_new_screenshot(self, path: Path) -> None:
        """Full pipeline for one screenshot. Never crashes the watcher."""
        try:
            await self._process_screenshot(path)
        except Exception as exc:
            logger.error(
                "pipeline.error: %s — %s", path.name, exc, exc_info=True
            )

    async def _process_screenshot(self, path: Path) -> None:
        # 1. Preprocessing
        try:
            full_img, _subtitle_crop = self._processor.process(path)
        except Exception as exc:
            logger.error("pipeline.preprocess_failed: %s — %s", path.name, exc)
            return

        # 2. Deduplication
        is_dup, frame_hash = self._dedup.is_duplicate_frame(full_img)
        if is_dup:
            logger.info("pipeline.duplicate_skipped: %s", path.name)
            return

        # 3. Vision analysis
        try:
            analysis = await self._vision.analyze(
                full_img,
                source_file=path.name,
                frame_hash=frame_hash,
            )
        except Exception as exc:
            logger.error("pipeline.vision_failed: %s — %s", path.name, exc)
            return

        # 4. Subtitle dedup check (after we have the analysis)
        if self._dedup.is_duplicate_subtitle(analysis.subtitles):
            logger.info(
                "pipeline.subtitle_deduplicated: %s", path.name
            )
            # Still store the frame but clear the duplicate subtitle
            analysis.subtitles = None

        # 5. Accept frame (update state timestamps/hashes)
        self._dedup.accept(frame_hash, analysis.subtitles)
        self._state.last_processed_file = path.name

        # 6. Update scene memory (SILENT — no assistant called here)
        self._scene_memory.ingest(analysis)
        logger.info(
            "pipeline.scene_memory_updated: file=%s subtitles=%r",
            path.name,
            analysis.subtitles,
        )

    # ------------------------------------------------------------------
    # User query entry point
    # ------------------------------------------------------------------

    async def answer_user_question(self, question: str) -> str:
        """
        Called when the user asks something.
        Retrieves stored context and routes to the appropriate assistant sink.
        """
        logger.info("pipeline.user_query: %r", question[:80])

        context = self._retriever.retrieve()

        if context.is_empty():
            logger.warning("pipeline.no_context_yet")
            return (
                "Пока нет достаточного контекста. "
                "Подожди немного — система накапливает данные из скриншотов."
            )

        # Try preferred sink (AnythingLLM), fall back to Ollama text
        sink = await self._select_sink()
        try:
            answer = await sink.query(question, context)
            return answer
        except HandoffUnavailableError as exc:
            logger.warning("pipeline.sink_unavailable: %s — trying fallback", exc)
            if sink is not self._fallback_sink:
                try:
                    return await self._fallback_sink.query(question, context)
                except HandoffUnavailableError as exc2:
                    logger.error("pipeline.fallback_also_failed: %s", exc2)
            return "Ассистент временно недоступен. Попробуй позже."

    async def _select_sink(self) -> AssistantSink:
        sink_mode = self._cfg.assistant.sink
        if sink_mode == "anythingllm":
            return self._preferred_sink
        if sink_mode == "ollama":
            return self._fallback_sink
        # "auto": try preferred first
        if await self._preferred_sink.is_available():
            logger.info("pipeline.sink_selected: anythingllm")
            return self._preferred_sink
        logger.info("pipeline.sink_selected: ollama_fallback")
        return self._fallback_sink
