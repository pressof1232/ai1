"""Orchestration pipeline.

The :class:`Pipeline` class ties all modules together:

1. Receives a path from the watcher queue.
2. Waits for file stability.
3. Pre-processes the image.
4. Checks for duplicate frames / subtitles.
5. Calls the vision model.
6. Calls the text model.
7. Persists state.

All errors are caught and logged so the watcher continues running.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from src.config import AppConfig
from src.deduplication import FrameDeduplicator
from src.logging_setup import get_logger
from src.preprocessor import preprocess_screenshot
from src.schemas import SceneUpdate
from src.stabilizer import wait_until_stable
from src.state import StateStore
from src.text_client import TextClient
from src.vision_client import VisionClient

log: logging.Logger = get_logger(__name__)


class Pipeline:
    """Main processing pipeline for a single screenshot.

    Args:
        config: Application configuration.
        vision: Initialised vision model client.
        text: Initialised text model client.
        state: Persistent state store.
        dedup: Frame deduplicator instance.
    """

    def __init__(
        self,
        config: AppConfig,
        vision: VisionClient,
        text: TextClient,
        state: StateStore,
        dedup: FrameDeduplicator,
    ) -> None:
        self._config = config
        self._vision = vision
        self._text = text
        self._state = state
        self._dedup = dedup
        self._processed_paths: set[str] = set()

    async def process(self, path: Path) -> None:
        """Run the full pipeline for *path*.

        Errors are caught and logged; the method never raises so the watcher
        loop can continue processing subsequent files.

        Args:
            path: Path to the new screenshot file.
        """
        path_str = str(path.resolve())

        # Prevent double-processing within a session.
        if path_str in self._processed_paths:
            log.debug("Already processed in this session: %s", path.name)
            return
        if self._state.last_file == path_str:
            log.debug("Already processed in previous session: %s", path.name)
            return

        try:
            await self._run(path, path_str)
        except Exception as exc:
            log.error("Error occurred processing %s: %s", path.name, exc, exc_info=True)

    # ------------------------------------------------------------------
    # Private pipeline stages
    # ------------------------------------------------------------------

    async def _run(self, path: Path, path_str: str) -> None:
        """Internal pipeline execution (may raise)."""
        cfg = self._config

        # --- Stage 1: Wait for file to stabilise ---
        log.info("File stable check: %s", path.name)
        stable = await wait_until_stable(
            path,
            poll_interval=cfg.stabilization_poll_interval,
            stable_count=cfg.stabilization_stable_count,
            timeout=cfg.stabilization_timeout,
        )
        if not stable:
            log.warning("File did not stabilise, skipping: %s", path.name)
            return
        log.info("File stable: %s", path.name)

        # --- Stage 2: Preprocessing ---
        log.info("Preprocessing started: %s", path.name)
        frame = preprocess_screenshot(path, cfg.preprocessing)
        log.info("Preprocessing completed: %s", path.name)

        # --- Stage 3: Deduplication ---
        if self._dedup.is_duplicate_frame(frame.full_frame_bytes):
            log.info("Duplicate frame skipped: %s", path.name)
            return

        # --- Stage 4: Vision model ---
        vision_result = await self._vision.analyse(
            frame.full_frame_bytes,
            frame.subtitle_crop_bytes,
        )

        # --- Stage 5: Subtitle dedup ---
        if self._dedup.is_duplicate_subtitle(vision_result.subtitles):
            log.info("Duplicate subtitles skipped: %s", path.name)
            return

        # --- Stage 6: Text model ---
        scene_update = SceneUpdate(
            subtitles=vision_result.subtitles,
            scene=vision_result.scene,
            important=vision_result.important,
            uncertainty=vision_result.uncertainty,
        )
        assistant_response = await self._text.react_to_scene(scene_update)

        # --- Stage 7: State update ---
        self._dedup.mark_processed(frame.full_frame_bytes, vision_result.subtitles)
        self._processed_paths.add(path_str)
        self._state.update_from_vision(
            file_path=path_str,
            subtitles=vision_result.subtitles,
            scene=vision_result.scene,
            assistant_response=assistant_response.raw_text,
            scene_hash=str(self._dedup._last_hash),
        )
        log.info("State updated for: %s", path.name)

        # Log assistant response internally (not shown to user by default).
        log.debug("Assistant internal response: %s", assistant_response.raw_text[:200])


async def run_watcher_loop(
    config: AppConfig,
    queue: asyncio.Queue[Path],
    pipeline: Pipeline,
) -> None:
    """Main async loop that drains the watcher queue.

    Args:
        config: Application configuration (unused here but kept for future use).
        queue: Queue populated by :class:`~src.watcher.FolderWatcher`.
        pipeline: Configured :class:`Pipeline` instance.
    """
    log.info("Watcher loop running — waiting for screenshots…")
    while True:
        path = await queue.get()
        log.info("Processing: %s", path.name)
        await pipeline.process(path)
        queue.task_done()
