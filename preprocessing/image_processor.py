"""Image preprocessing: resize and subtitle crop."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from config.loader import AppConfig

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Resize and optionally crop images for the vision pipeline."""

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._max_size: Tuple[int, int] = (
            cfg.preprocessing.max_width,
            cfg.preprocessing.max_height,
        )
        self._subtitle_ratio = cfg.preprocessing.subtitle_crop_top_ratio
        self._save_temp = cfg.preprocessing.save_temp_images
        self._temp_dir = Path(cfg.paths.temp_folder)

    def process(self, image_path: Path) -> Tuple[Image.Image, Optional[Image.Image]]:
        """
        Load, resize, and optionally crop the subtitle region.

        Returns
        -------
        (full_frame, subtitle_crop)
            full_frame  — resized full image
            subtitle_crop — cropped bottom region (None if ratio is 0 or 1)
        """
        logger.debug("preprocessing.started: %s", image_path.name)
        img = Image.open(image_path).convert("RGB")

        # Resize preserving aspect ratio
        img.thumbnail(self._max_size, Image.LANCZOS)

        subtitle_crop: Optional[Image.Image] = None
        ratio = self._subtitle_ratio
        if 0.0 < ratio < 1.0:
            w, h = img.size
            top = int(h * ratio)
            subtitle_crop = img.crop((0, top, w, h))

        if self._save_temp:
            self._save_to_temp(image_path.stem, img, subtitle_crop)

        logger.debug("preprocessing.completed: %s", image_path.name)
        return img, subtitle_crop

    def _save_to_temp(
        self,
        stem: str,
        full: Image.Image,
        crop: Optional[Image.Image],
    ) -> None:
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        full_path = self._temp_dir / f"{stem}_full.png"
        full.save(full_path)
        if crop is not None:
            crop_path = self._temp_dir / f"{stem}_subtitle.png"
            crop.save(crop_path)
        logger.debug("preprocessing.temp_saved: %s", stem)
