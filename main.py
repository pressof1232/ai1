"""
VLC Anime Assistant — entry point.

Usage:
    python main.py                          # start the watcher (daemon mode)
    python main.py --test path/to/image     # manual test mode: run pipeline on one image
    python main.py --query "что происходит?"  # query assistant using stored memory

Cluster overrides (optional — override config.session values):
    python main.py --series "Blood-C" --episode "S01E01" --query "что происходит?"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLC Anime Assistant — local automation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml next to main.py)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--test",
        metavar="IMAGE_PATH",
        help="Manual test: run the full pipeline on a single image and print debug output",
    )
    group.add_argument(
        "--query",
        metavar="QUESTION",
        help="Query the assistant using stored scene memory",
    )
    # Optional cluster overrides
    parser.add_argument(
        "--series",
        metavar="SERIES",
        default=None,
        help="Override series name for the active cluster (overrides config.session.series_name)",
    )
    parser.add_argument(
        "--episode",
        metavar="EPISODE",
        default=None,
        help="Override episode label for the active cluster (overrides config.session.episode_label)",
    )
    parser.add_argument(
        "--session",
        metavar="SESSION_ID",
        default=None,
        help="Override session ID for the active cluster (overrides config.session.session_id)",
    )
    return parser.parse_args()


def _apply_cluster_overrides(cfg, args) -> None:
    """Apply --series / --episode / --session CLI flags into the config session block."""
    if args.series:
        cfg.session.series_name = args.series
    if args.episode:
        cfg.session.episode_label = args.episode
    if args.session:
        cfg.session.session_id = args.session
        cfg.session.auto_generate_session_id = False


async def _run_watcher(config_path: Path, args) -> None:
    from config.loader import load_config
    from logging_setup.logger import setup_logging
    from orchestrator.pipeline import Pipeline

    cfg = load_config(config_path)
    _apply_cluster_overrides(cfg, args)
    setup_logging(cfg)
    pipeline = Pipeline(cfg)
    await pipeline.run_forever()


async def _run_test(config_path: Path, image_path: str, args) -> None:
    from config.loader import load_config
    from logging_setup.logger import setup_logging
    from tests.manual_test import run_manual_test

    cfg = load_config(config_path)
    _apply_cluster_overrides(cfg, args)
    setup_logging(cfg)
    await run_manual_test(cfg, Path(image_path))


async def _run_query(config_path: Path, question: str, args) -> None:
    from config.loader import load_config
    from logging_setup.logger import setup_logging
    from orchestrator.pipeline import Pipeline

    cfg = load_config(config_path)
    _apply_cluster_overrides(cfg, args)
    setup_logging(cfg)
    pipeline = Pipeline(cfg)
    print(f"[Active cluster]: {pipeline._cluster}")
    answer = await pipeline.answer_user_question(question)
    print(f"\n[Assistant]: {answer}\n")


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)

    if args.test:
        asyncio.run(_run_test(config_path, args.test, args))
    elif args.query:
        asyncio.run(_run_query(config_path, args.query, args))
    else:
        asyncio.run(_run_watcher(config_path, args))


if __name__ == "__main__":
    main()
