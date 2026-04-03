"""Logging setup.

Configures the root logger with a consistent format suitable for both
terminal and file output.  Call :func:`setup_logging` once at startup.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional


_LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """Configure root logger.

    Args:
        level: Python logging level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to a log file; if given, logs are written
                  there in addition to stdout.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        handlers.append(file_handler)

    logging.basicConfig(
        level=numeric_level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )

    # Silence overly noisy third-party loggers at INFO level.
    for noisy in ("httpx", "httpcore", "PIL", "watchdog"):
        logging.getLogger(noisy).setLevel(
            logging.DEBUG if level.upper() == "DEBUG" else logging.WARNING
        )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)
