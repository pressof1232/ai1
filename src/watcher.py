"""File watcher.

Uses *watchdog* to monitor a folder for new image files.  When a new file
is detected it is handed off to a caller-supplied async callback via
``asyncio.Queue``.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional

from watchdog.events import FileCreatedEvent, FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from src.logging_setup import get_logger

log: logging.Logger = get_logger(__name__)

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class _ImageCreatedHandler(FileSystemEventHandler):
    """Watchdog event handler that enqueues newly created image files."""

    def __init__(self, queue: asyncio.Queue[Path], loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self._queue = queue
        self._loop = loop

    def on_created(self, event: FileSystemEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        path = Path(str(event.src_path))
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            return
        log.info("File detected: %s", path.name)
        self._loop.call_soon_threadsafe(self._queue.put_nowait, path)


class FolderWatcher:
    """Watches a folder for new screenshots and feeds them into a queue.

    Args:
        folder: Directory to watch.
        queue: Asyncio queue where detected :class:`pathlib.Path` objects
               are placed.
        loop: The running event loop (needed for thread-safe queue access).
    """

    def __init__(
        self,
        folder: Path,
        queue: asyncio.Queue[Path],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._folder = folder
        self._queue = queue
        self._loop = loop
        self._observer: Optional[Observer] = None

    def start(self) -> None:
        """Start the background watchdog observer thread."""
        if not self._folder.exists():
            raise FileNotFoundError(
                f"Screenshot folder does not exist: {self._folder}"
            )
        handler = _ImageCreatedHandler(self._queue, self._loop)
        self._observer = Observer()
        self._observer.schedule(handler, str(self._folder), recursive=False)
        self._observer.start()
        log.info("Watcher started on: %s", self._folder)

    def stop(self) -> None:
        """Stop the watchdog observer thread gracefully."""
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join(timeout=5)
            log.info("Watcher stopped.")
