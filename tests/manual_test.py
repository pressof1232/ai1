"""
Manual test mode — feed a single image through the full pipeline and print debug output.

Usage:
    python main.py --test path/to/screenshot.png
    python main.py --test path/to/screenshot.png --query "что происходит?"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config.loader import AppConfig
from dedup.deduplicator import Deduplicator
from memory.cluster_resolver import resolve_cluster_for_path, resolve_startup_cluster
from memory.context_retriever import ContextRetriever
from memory.context_store import ContextStore
from memory.scene_memory import SceneMemory
from orchestrator.pipeline import Pipeline
from preprocessing.image_processor import ImageProcessor
from state.local_state import LocalState
from vision.ollama_vision_client import OllamaVisionClient

logger = logging.getLogger(__name__)


def _hr(title: str = "") -> None:
    width = 60
    if title:
        print(f"\n{'─' * 4} {title} {'─' * max(0, width - len(title) - 6)}")
    else:
        print("─" * width)


async def run_manual_test(cfg: AppConfig, image_path: Path) -> None:
    """
    Run the full pipeline on one image and print structured debug output.
    Does NOT require the watcher — processes the image directly.
    """
    cfg.ensure_dirs()

    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        return

    # Resolve active cluster (startup cluster + optional path refinement)
    base_cluster = resolve_startup_cluster(cfg)
    cluster = resolve_cluster_for_path(cfg, image_path, base_cluster)

    print(f"\n{'═' * 60}")
    print(f"  MANUAL TEST MODE")
    print(f"  Image: {image_path}")
    print(f"  Active cluster: {cluster}")
    print(f"{'═' * 60}")

    state = LocalState(cfg)
    store = ContextStore(cfg)
    processor = ImageProcessor(cfg)
    dedup = Deduplicator(cfg, state)
    vision = OllamaVisionClient(cfg)
    scene_memory = SceneMemory(cfg, store)
    retriever = ContextRetriever(cfg, store)

    # 1. Preprocessing
    _hr("1. PREPROCESSING")
    try:
        full_img, subtitle_crop = processor.process(image_path)
        print(f"Full image size:     {full_img.size}")
        print(f"Subtitle crop:       {subtitle_crop.size if subtitle_crop else 'N/A'}")
    except Exception as exc:
        print(f"[ERROR] Preprocessing failed: {exc}")
        return

    # 2. Deduplication check
    _hr("2. DEDUPLICATION")
    is_dup, frame_hash = dedup.is_duplicate_frame(full_img)
    print(f"Frame hash:          {frame_hash}")
    print(f"Is duplicate:        {is_dup}")
    if is_dup:
        print("[INFO] Frame would be skipped in watcher mode.")
        print("       Continuing in test mode anyway...\n")

    # 3. Vision analysis
    _hr("3. VISION ANALYSIS  (calling Ollama...)")
    try:
        analysis = await vision.analyze(
            full_img,
            source_file=image_path.name,
            frame_hash=frame_hash,
        )
        print(f"Subtitles:           {analysis.subtitles!r}")
        print(f"Scene:               {analysis.scene}")
        print(f"Important:           {analysis.important}")
        print(f"People estimate:     {analysis.visible_people_estimate}")
        print(f"Uncertainty:         {analysis.uncertainty}")
        print(f"Image quality note:  {analysis.image_quality_note}")
        print(f"Timestamp:           {analysis.timestamp.isoformat()}")
    except Exception as exc:
        print(f"[ERROR] Vision analysis failed: {exc}")
        print("        Is Ollama running? Is gemma3:4b pulled?")
        return

    # 4. Scene memory update
    _hr("4. SCENE MEMORY UPDATE")
    sub_dup = dedup.is_duplicate_subtitle(analysis.subtitles)
    if sub_dup:
        print(f"[INFO] Subtitle duplicate — clearing: {analysis.subtitles!r}")
        analysis.subtitles = None
    dedup.accept(frame_hash, analysis.subtitles)
    scene_memory.ingest(analysis, cluster)
    print(f"Scene memory updated successfully (cluster: {cluster}).")

    # 5. Context retrieval
    _hr("5. CONTEXT RETRIEVAL")
    bundle = retriever.retrieve(cluster)
    print(f"Recent frames:       {len(bundle.recent_frames)}")
    print(f"Recent subtitles:    {bundle.recent_subtitles}")
    print(f"Scene summary:       {bundle.rolling_state.scene_summary[:120]}")
    print(f"Important entities:  {bundle.rolling_state.important_entities}")

    # 6. Full JSON dump
    _hr("6. FULL ANALYSIS JSON")
    print(json.dumps(analysis.model_dump(mode="json"), ensure_ascii=False, indent=2))

    _hr("TEST COMPLETE")
    print()

    # 7. Optional interactive query
    print("You can now run:  python main.py --query \"что происходит?\"")
    print("to test assistant query from stored memory.\n")
