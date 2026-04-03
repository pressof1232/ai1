"""Structured logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from config.loader import AppConfig


def setup_logging(cfg: AppConfig) -> None:
    """Configure root logger from AppConfig."""
    level = getattr(logging, cfg.logging.level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]

    if cfg.logging.log_to_file:
        logs_dir = Path(cfg.paths.logs_folder)
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / cfg.logging.log_filename
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        handlers.append(file_handler)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )
    logging.getLogger("watchdog").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
