"""Entry point for the anime-watcher automation service.

Usage:
    # Normal watcher mode (requires config.yaml):
    python main.py

    # Normal watcher mode with explicit config path:
    python main.py --config /path/to/config.yaml

    # Manual test mode (no watcher, runs one image through the pipeline):
    python main.py --test path/to/screenshot.png
    python main.py --test path/to/screenshot.png --config /path/to/config.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

from src.config import AppConfig, load_config
from src.deduplication import FrameDeduplicator
from src.logging_setup import get_logger, setup_logging
from src.pipeline import Pipeline, run_watcher_loop
from src.state import StateStore
from src.text_client import TextClient
from src.vision_client import VisionClient
from src.watcher import FolderWatcher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local anime watch-together automation using VLC and Ollama."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--test",
        metavar="IMAGE",
        help="Run manual test mode: feed IMAGE through the pipeline and exit.",
    )
    return parser.parse_args()


async def _run_watcher(config: AppConfig) -> None:
    """Start the folder watcher and the async processing loop."""
    log = get_logger(__name__)

    queue: asyncio.Queue[Path] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    vision = VisionClient(config)
    text = TextClient(config)
    state = StateStore(config.state_db_path)
    dedup = FrameDeduplicator(config.deduplication)
    pipeline = Pipeline(config, vision, text, state, dedup)

    watcher = FolderWatcher(config.screenshot_folder, queue, loop)
    watcher.start()

    stop_event = asyncio.Event()

    def _shutdown(*_: object) -> None:
        log.info("Shutdown signal received.")
        stop_event.set()

    # Register graceful shutdown on SIGINT / SIGTERM.
    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGINT, _shutdown)
        loop.add_signal_handler(signal.SIGTERM, _shutdown)
    else:
        # On Windows signal handling in asyncio is limited; rely on KeyboardInterrupt.
        pass

    watcher_task = asyncio.create_task(run_watcher_loop(config, queue, pipeline))

    try:
        if sys.platform == "win32":
            # Poll for KeyboardInterrupt on Windows.
            while not stop_event.is_set():
                await asyncio.sleep(0.5)
        else:
            await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("Keyboard interrupt received.")
    finally:
        watcher.stop()
        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass
        await vision.aclose()
        await text.aclose()
        log.info("Anime watcher stopped.")


def main() -> None:
    """CLI entry point."""
    args = _parse_args()

    # Load config first so we know the log level.
    try:
        config: AppConfig = load_config(args.config)
    except FileNotFoundError as exc:
        # Logging not yet configured; print directly.
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] Failed to load configuration: {exc}", file=sys.stderr)
        sys.exit(1)

    setup_logging(level=config.log_level)
    log = get_logger(__name__)
    log.info("App startup")
    log.info("Config loaded: %s", args.config)

    if args.test:
        # Manual test mode.
        from src.test_mode import run_test
        image_path = Path(args.test)
        asyncio.run(run_test(config, image_path))
        return

    # Normal watcher mode.
    log.info("Starting watcher mode…")
    try:
        asyncio.run(_run_watcher(config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
