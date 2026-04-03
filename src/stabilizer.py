"""File stabilization.

Waits until a newly created file has been fully written by checking that
its size stops changing for a configurable number of consecutive polls.
This prevents reading truncated or partially written screenshot files.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from src.logging_setup import get_logger

log: logging.Logger = get_logger(__name__)


async def wait_until_stable(
    path: Path,
    poll_interval: float = 0.3,
    stable_count: int = 3,
    timeout: float = 10.0,
) -> bool:
    """Wait until *path* is fully written (stable size).

    Polls the file size every *poll_interval* seconds.  Considers the file
    stable when its size does not change for *stable_count* consecutive polls.

    Args:
        path: Path to the file to monitor.
        poll_interval: Seconds between size checks.
        stable_count: Number of consecutive equal-size polls required.
        timeout: Maximum seconds to wait before giving up.

    Returns:
        ``True`` when the file is stable, ``False`` if it timed out or the
        file disappeared.
    """
    elapsed = 0.0
    consecutive = 0
    last_size = -1

    while elapsed < timeout:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            log.warning("File disappeared while waiting for stability: %s", path.name)
            return False

        if size == last_size:
            consecutive += 1
            if consecutive >= stable_count:
                log.debug(
                    "File stable after %.1f s: %s (%d bytes)",
                    elapsed,
                    path.name,
                    size,
                )
                return True
        else:
            consecutive = 0
            last_size = size

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    log.warning("Stability timeout reached for: %s", path.name)
    return False
