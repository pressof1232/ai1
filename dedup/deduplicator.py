"""Perceptual-hash based frame deduplication."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import imagehash
from PIL import Image

from config.loader import AppConfig
from state.local_state import LocalState

logger = logging.getLogger(__name__)


class Deduplicator:
    """Decides whether a frame should be skipped as a duplicate."""

    def __init__(self, cfg: AppConfig, state: LocalState) -> None:
        self._threshold = cfg.dedup.similarity_threshold
        self._hash_size = cfg.dedup.hash_size
        self._sub_cooldown = timedelta(seconds=cfg.dedup.subtitle_cooldown_seconds)
        self._frame_cooldown = timedelta(seconds=cfg.dedup.frame_cooldown_seconds)
        self._state = state

    def compute_hash(self, img: Image.Image) -> str:
        """Return the perceptual hash string for an image."""
        return str(imagehash.phash(img, hash_size=self._hash_size))

    def is_duplicate_frame(self, img: Image.Image) -> tuple[bool, str]:
        """
        Check if this frame is too similar to the previous accepted frame.

        Returns
        -------
        (is_dup, hash_str)
        """
        current_hash = self.compute_hash(img)
        last_hash = self._state.last_frame_hash

        # Frame cooldown check
        last_ts = self._state.last_accepted_timestamp
        if last_ts is not None:
            elapsed = datetime.utcnow() - last_ts
            if elapsed < self._frame_cooldown:
                logger.debug(
                    "dedup.frame_cooldown: %.1fs since last accepted frame",
                    elapsed.total_seconds(),
                )
                return True, current_hash

        if last_hash is None:
            return False, current_hash

        try:
            distance = imagehash.hex_to_hash(current_hash) - imagehash.hex_to_hash(last_hash)
        except Exception:
            return False, current_hash

        if distance <= self._threshold:
            logger.debug(
                "dedup.similar_frame skipped (hamming=%d, threshold=%d)",
                distance,
                self._threshold,
            )
            return True, current_hash

        return False, current_hash

    def is_duplicate_subtitle(self, subtitle: Optional[str]) -> bool:
        """Check if this subtitle text was seen recently (within cooldown)."""
        if not subtitle:
            return False
        last_sub = self._state.last_subtitle
        if last_sub != subtitle:
            return False
        last_ts = self._state.last_subtitle_timestamp
        if last_ts is None:
            return False
        elapsed = datetime.utcnow() - last_ts
        if elapsed < self._sub_cooldown:
            logger.debug("dedup.subtitle_cooldown: same subtitle within %.1fs", elapsed.total_seconds())
            return True
        return False

    def accept(self, frame_hash: str, subtitle: Optional[str]) -> None:
        """Record that a frame was accepted for processing."""
        now = datetime.utcnow()
        self._state.last_frame_hash = frame_hash
        self._state.last_accepted_timestamp = now
        if subtitle:
            self._state.last_subtitle = subtitle
            self._state.last_subtitle_timestamp = now
