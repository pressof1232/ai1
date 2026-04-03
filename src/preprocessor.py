"""Image preprocessing.

Loads a screenshot, optionally resizes the full frame, and optionally
crops the subtitle region.  Returns ready-to-encode image bytes for each
variant that the caller wants to send to the vision model.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from src.config import PreprocessingConfig
from src.logging_setup import get_logger

log: logging.Logger = get_logger(__name__)


@dataclass
class PreprocessedFrame:
    """Container for one or two preprocessed image variants.

    Attributes:
        full_frame_bytes: JPEG bytes of the (possibly resized) full frame.
        subtitle_crop_bytes: JPEG bytes of the subtitle region crop, or
                             ``None`` if subtitle cropping is disabled.
        original_size: ``(width, height)`` of the original image.
    """

    full_frame_bytes: bytes
    subtitle_crop_bytes: Optional[bytes]
    original_size: tuple[int, int]


def preprocess_screenshot(
    path: Path,
    config: PreprocessingConfig,
) -> PreprocessedFrame:
    """Load and preprocess a screenshot.

    Args:
        path: Path to the source image file.
        config: Preprocessing configuration.

    Returns:
        A :class:`PreprocessedFrame` with encoded variants.

    Raises:
        OSError: If the image cannot be opened.
        ValueError: If the image is corrupted or unreadable.
    """
    log.debug("Preprocessing: %s", path.name)

    try:
        img: Image.Image = Image.open(path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot open image '{path}': {exc}") from exc

    original_size = img.size  # (width, height)
    log.debug("Original size: %dx%d", original_size[0], original_size[1])

    # --- Full frame ---
    full = img.copy()
    if config.resize_full_frame:
        full = _resize_keep_aspect(full, config.full_frame_width)
        log.debug("Resized to: %dx%d", full.width, full.height)

    full_bytes = _to_jpeg_bytes(full)

    # --- Subtitle crop ---
    subtitle_bytes: Optional[bytes] = None
    if config.crop_subtitle_region:
        subtitle_bytes = _crop_subtitle(img, config.subtitle_region_height_fraction)

    log.debug("Preprocessing complete: full=%d B, subtitle=%s B",
              len(full_bytes),
              len(subtitle_bytes) if subtitle_bytes else "N/A")

    return PreprocessedFrame(
        full_frame_bytes=full_bytes,
        subtitle_crop_bytes=subtitle_bytes,
        original_size=original_size,
    )


def _resize_keep_aspect(img: Image.Image, target_width: int) -> Image.Image:
    """Resize *img* so its width equals *target_width*, preserving aspect ratio."""
    if img.width <= target_width:
        return img
    ratio = target_width / img.width
    new_height = max(1, int(img.height * ratio))
    return img.resize((target_width, new_height), Image.LANCZOS)


def _crop_subtitle(img: Image.Image, height_fraction: float) -> bytes:
    """Crop the bottom *height_fraction* of the image (subtitle area).

    Args:
        img: Source PIL image.
        height_fraction: Fraction of total height to crop from the bottom.

    Returns:
        JPEG bytes of the cropped region.
    """
    w, h = img.size
    crop_top = int(h * (1.0 - height_fraction))
    cropped = img.crop((0, crop_top, w, h))
    return _to_jpeg_bytes(cropped)


def _to_jpeg_bytes(img: Image.Image, quality: int = 85) -> bytes:
    """Encode a PIL image as JPEG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
