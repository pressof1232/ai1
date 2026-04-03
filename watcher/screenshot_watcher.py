"""Screenshot watcher — monitors a folder for new VLC screenshots."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Callable, Coroutine, Any, Set

from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
from watchdog.observers import Observer

from config.loader import AppConfig

logger = logging.getLogger(__name__)

# Callback type: async function that receives a stable Path
FrameCallback = Callable[[Path], Coroutine[Any, Any, None]]


class _ScreenshotEventHandler(FileSystemEventHandler):
    """Watchdog handler that queues new image files for processing."""

    def __init__(
        self,
        extensions: Set[str],
        queue: asyncio.Queue,  # type: ignore[type-arg]
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__()
        self._exts = extensions
        self._queue = queue
        self._loop = loop
        self._seen: Set[str] = set()

    def _enqueue(self, path: Path) -> None:
        if path.suffix.lstrip(".").lower() not in self._exts:
            return
        if path.name.startswith(".") or path.name.startswith("~"):
            return  # temp/hidden files
        key = str(path)
        if key in self._seen:
            return
        self._seen.add(key)
        logger.info("watcher.file_detected: %s", path.name)
        asyncio.run_coroutine_threadsafe(self._queue.put(path), self._loop)

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        if not event.is_directory:
            self._enqueue(Path(str(event.src_path)))

    def on_modified(self, event: FileModifiedEvent) -> None:  # type: ignore[override]
        if not event.is_directory:
            self._enqueue(Path(str(event.src_path)))


async def _wait_for_stable(
    path: Path,
    poll_interval: float,
    checks: int,
) -> bool:
    """
    Wait until the file size is stable across `checks` consecutive polls.
    Returns False if the file disappears or is unreadable.
    """
    last_size = -1
    stable_count = 0

    while stable_count < checks:
        await asyncio.sleep(poll_interval)
        if not path.exists():
            return False
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size == last_size and size > 0:
            stable_count += 1
        else:
            stable_count = 0
        last_size = size

    logger.info("watcher.file_stable: %s (%d bytes)", path.name, last_size)
    return True


class ScreenshotWatcher:
    """Watches a folder and calls `callback` for each stable screenshot."""

    def __init__(self, cfg: AppConfig, callback: FrameCallback) -> None:
        self._folder = Path(cfg.paths.screenshots_folder)
        self._extensions: Set[str] = {
            ext.lower().lstrip(".") for ext in cfg.watcher.supported_extensions
        }
        self._poll_interval = cfg.watcher.poll_interval_seconds
        self._stability_checks = cfg.watcher.stability_checks
        self._callback = callback

    async def run_forever(self) -> None:
        self._folder.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Path] = asyncio.Queue()

        handler = _ScreenshotEventHandler(self._extensions, queue, loop)
        observer = Observer()
        observer.schedule(handler, str(self._folder), recursive=False)
        observer.start()
        logger.info("watcher.started: watching %s", self._folder)

        try:
            while True:
                path = await queue.get()
                asyncio.create_task(self._process(path))
        except asyncio.CancelledError:
            pass
        finally:
            observer.stop()
            observer.join()
            logger.info("watcher.stopped")

    async def _process(self, path: Path) -> None:
        stable = await _wait_for_stable(
            path, self._poll_interval, self._stability_checks
        )
        if not stable:
            logger.warning("watcher.file_disappeared_or_unstable: %s", path.name)
            return
        try:
            await self._callback(path)
        except Exception as exc:
            logger.error("watcher.callback_error: %s — %s", path.name, exc, exc_info=True)
