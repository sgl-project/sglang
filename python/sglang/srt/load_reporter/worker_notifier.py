"""Coalesce multi-tokenizer refresh events into IPC requests."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Final, Optional

from sglang.srt.managers.io_struct import (
    LoadReporterRefreshIpcReq,
    LoadReporterRefreshReason,
)

logger = logging.getLogger(__name__)

DEFAULT_COALESCE_WINDOW_MS: Final[int] = 50

_REASON_PRIORITY: Final[Dict[LoadReporterRefreshReason, int]] = {
    LoadReporterRefreshReason.DISPATCH: 1,
    LoadReporterRefreshReason.COMPLETION: 2,
    LoadReporterRefreshReason.ABORT: 3,
}


class LoadReporterRefreshNotifier:
    """Single-task coalescer for refresh events."""

    def __init__(self, worker_id: str, send: Callable[[Any], None]) -> None:
        """Initialize per-HTTP-worker refresh coalescer."""
        self._worker_id = worker_id
        self._send = send
        self._coalesce_window_ms: int = DEFAULT_COALESCE_WINDOW_MS
        self._active: bool = False
        self._accumulated_count: int = 0
        self._accumulated_reason: Optional[LoadReporterRefreshReason] = None
        self._event: asyncio.Event = asyncio.Event()
        self._task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        """Start the background coalescer task."""
        self._active = True
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def close(self) -> None:
        """Stop and await the background task."""
        self._active = False
        if self._task is None or self._task.done():
            return
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass

    def notify(self, reason: LoadReporterRefreshReason, event_count: int = 1) -> None:
        """Accumulate a refresh event and wake the background task."""
        self._accumulated_count += event_count
        if self._accumulated_reason is None or (
            _REASON_PRIORITY[reason] > _REASON_PRIORITY[self._accumulated_reason]
        ):
            self._accumulated_reason = reason
        self._event.set()

    async def _run(self) -> None:
        """Coalescer loop: wait for event, sleep one window, send once."""
        try:
            while True:
                await self._event.wait()
                await asyncio.sleep(self._coalesce_window_ms / 1000.0)
                count = self._accumulated_count
                reason = self._accumulated_reason
                self._accumulated_count = 0
                self._accumulated_reason = None
                self._event.clear()
                if self._active and reason is not None and count > 0:
                    self._send(
                        LoadReporterRefreshIpcReq(
                            worker_id=self._worker_id,
                            reason=reason,
                            event_count=count,
                        )
                    )
        except asyncio.CancelledError:
            raise
