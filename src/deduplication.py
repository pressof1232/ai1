"""Frame deduplication.

Prevents sending nearly-identical frames or repeated subtitles to the
vision model, reducing unnecessary API calls and model load.

Uses perceptual hashing (pHash) via *imagehash* to compare frames.
"""
from __future__ import annotations

import io
import logging
import time
from collections import deque
from typing import Optional

import imagehash
from PIL import Image

from src.config import DeduplicationConfig
from src.logging_setup import get_logger

log: logging.Logger = get_logger(__name__)


class FrameDeduplicator:
    """Stateful deduplicator for screenshot frames.

    Args:
        config: Deduplication configuration.
    """

    def __init__(self, config: DeduplicationConfig) -> None:
        self._config = config
        self._last_hash: Optional[imagehash.ImageHash] = None
        self._last_processed_at: float = 0.0
        self._subtitle_history: deque[str] = deque(
            maxlen=config.subtitle_history_size
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_duplicate_frame(self, image_bytes: bytes) -> bool:
        """Return ``True`` if *image_bytes* is too similar to the last frame.

        Also enforces the cooldown interval.

        Args:
            image_bytes: JPEG bytes of the full (possibly resized) frame.

        Returns:
            ``True`` when the frame should be skipped.
        """
        now = time.monotonic()
        elapsed = now - self._last_processed_at

        if elapsed < self._config.cooldown_seconds:
            log.debug(
                "Cooldown active (%.1f s / %.1f s)",
                elapsed,
                self._config.cooldown_seconds,
            )
            return True

        current_hash = self._compute_hash(image_bytes)
        if self._last_hash is not None:
            distance = self._last_hash - current_hash
            if distance <= self._config.hash_distance_threshold:
                log.info(
                    "Duplicate frame skipped (hash distance=%d, threshold=%d)",
                    distance,
                    self._config.hash_distance_threshold,
                )
                return True

        return False

    def is_duplicate_subtitle(self, subtitle: Optional[str]) -> bool:
        """Return ``True`` if *subtitle* was already seen recently.

        Args:
            subtitle: The subtitle string extracted by the vision model, or
                      ``None`` if no subtitles are present.

        Returns:
            ``True`` when the subtitle should be considered a repeat.
        """
        if not subtitle:
            return False
        if subtitle in self._subtitle_history:
            log.info("Duplicate subtitle skipped: %r", subtitle[:60])
            return True
        return False

    def mark_processed(self, image_bytes: bytes, subtitle: Optional[str]) -> None:
        """Record that a frame was successfully processed.

        Updates internal state so subsequent duplicates are detected.

        Args:
            image_bytes: JPEG bytes of the processed frame.
            subtitle: Subtitle text (may be ``None``).
        """
        self._last_hash = self._compute_hash(image_bytes)
        self._last_processed_at = time.monotonic()
        if subtitle:
            self._subtitle_history.append(subtitle)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hash(image_bytes: bytes) -> imagehash.ImageHash:
        """Compute a perceptual hash from JPEG bytes."""
        img = Image.open(io.BytesIO(image_bytes))
        return imagehash.phash(img)
