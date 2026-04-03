"""Manual test mode.

Allows feeding a local image file directly through the pipeline without
starting the folder watcher.  Useful for prompt tuning and validation.

Usage::

    python main.py --test path/to/screenshot.png
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from src.config import AppConfig
from src.deduplication import FrameDeduplicator
from src.logging_setup import get_logger
from src.preprocessor import preprocess_screenshot
from src.prompts import build_text_user_message
from src.schemas import SceneUpdate
from src.vision_client import VisionClient
from src.text_client import TextClient

log: logging.Logger = get_logger(__name__)


async def run_test(config: AppConfig, image_path: Path) -> None:
    """Run the vision→text pipeline on a single image and print results.

    This is intended for interactive debugging.  All intermediate results are
    printed to stdout regardless of the ``debug`` setting.

    Args:
        config: Application configuration.
        image_path: Path to the image file to process.
    """
    if not image_path.exists():
        print(f"[ERROR] File not found: {image_path}")
        return

    print(f"\n{'='*60}")
    print(f"  Manual test mode — {image_path.name}")
    print(f"{'='*60}\n")

    # Preprocessing
    print("[1/4] Preprocessing image…")
    frame = preprocess_screenshot(image_path, config.preprocessing)
    print(
        f"      Original: {frame.original_size[0]}x{frame.original_size[1]} px | "
        f"Full frame: {len(frame.full_frame_bytes):,} B | "
        f"Subtitle crop: {len(frame.subtitle_crop_bytes):,} B"
        if frame.subtitle_crop_bytes
        else f"      Original: {frame.original_size[0]}x{frame.original_size[1]} px | "
             f"Full frame: {len(frame.full_frame_bytes):,} B | Subtitle crop: disabled"
    )

    # Vision model
    print(f"\n[2/4] Calling vision model ({config.vision_model})…")
    vision_client = VisionClient(config)
    try:
        vision_result = await vision_client.analyse(
            frame.full_frame_bytes,
            frame.subtitle_crop_bytes,
        )
    finally:
        await vision_client.aclose()

    print("\n  --- Vision result (structured) ---")
    print(json.dumps(vision_result.model_dump(), ensure_ascii=False, indent=2))

    # Scene update
    scene_update = SceneUpdate(
        subtitles=vision_result.subtitles,
        scene=vision_result.scene,
        important=vision_result.important,
        uncertainty=vision_result.uncertainty,
    )

    # Text model
    print(f"\n[3/4] Calling text model ({config.text_model})…")
    user_msg = build_text_user_message(
        subtitles=scene_update.subtitles,
        scene=scene_update.scene,
        important=scene_update.important,
        uncertainty=scene_update.uncertainty,
    )
    print("\n  --- Text model user message ---")
    print(user_msg)

    text_client = TextClient(config)
    try:
        assistant_response = await text_client.react_to_scene(scene_update)
    finally:
        await text_client.aclose()

    print("\n  --- Assistant internal response ---")
    print(assistant_response.raw_text)
    print(f"\n{'='*60}")
    print("  Test complete.")
    print(f"{'='*60}\n")
