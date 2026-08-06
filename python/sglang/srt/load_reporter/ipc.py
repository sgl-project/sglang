"""Refresh notifier for the multi-tokenizer load reporter.

This module provides a single collaborator:

* ``LoadReporterRefreshNotifier`` -- a single-background-task coalescer that
  emits at most ONE ``LoadReporterRefreshIpcReq`` per broadcast window.
  Deterministic priority: ABORT > COMPLETION > DISPATCH.  Event counts are
  summed across all ``notify()`` calls within one window.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Final, Optional

from sglang.srt.managers.io_struct import (
    LoadReporterRefreshIpcReq,
    LoadReporterRefreshReason,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_COALESCE_WINDOW_MS: Final[int] = 50

# ---------------------------------------------------------------------------
# Reason priority for coalescing
# ---------------------------------------------------------------------------

# Higher integer = higher priority (ABORT wins over everything).
_REASON_PRIORITY: Final[Dict[LoadReporterRefreshReason, int]] = {
    LoadReporterRefreshReason.DISPATCH: 1,
    LoadReporterRefreshReason.COMPLETION: 2,
    LoadReporterRefreshReason.ABORT: 3,
}


# ---------------------------------------------------------------------------
# LoadReporterRefreshNotifier
# ---------------------------------------------------------------------------


class LoadReporterRefreshNotifier:
    """Single-background-task coalescer for load-reporter refresh events.

    At most ONE ``LoadReporterRefreshIpcReq`` is sent per broadcast window.
    The background task waits for an ``asyncio.Event``, sleeps for the
    configured window, then atomically swaps out the accumulated state and
    calls ``send`` exactly once.

    Coalescing semantics:
    * ``event_count`` values are **summed** across all ``notify()`` calls in
      the window.
    * ``reason`` is the **maximum-priority** value seen (ABORT > COMPLETION >
      DISPATCH).

    The completion/abort/dispatch hooks MUST NOT send socket messages directly;
    they only call ``notify()``.  Only ``_run()`` calls ``send``.
    """

    def __init__(self, worker_id: str, send: Callable[[Any], None]) -> None:
        """Initialize one per-HTTP-worker refresh coalescer.

        Args:
            worker_id: Stable diagnostic identifier for the HTTP worker.
            send: Synchronous IPC dispatch callback.

        Returns:
            None.
        """
        self._worker_id = worker_id
        self._send = send
        # Coalesce-window duration in milliseconds.
        self._coalesce_window_ms: int = DEFAULT_COALESCE_WINDOW_MS
        # Whether the notifier is currently active.
        self._active: bool = False
        # Accumulated state for the current window (None = no notification pending).
        self._accumulated_count: int = 0
        self._accumulated_reason: Optional[LoadReporterRefreshReason] = None
        # Event set by notify(); cleared atomically at the start of each send.
        self._event: asyncio.Event = asyncio.Event()
        # Single background task; set by start().
        self._task: Optional[asyncio.Task[None]] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the single background coalescer task.

        Returns:
            None.
        """
        self._active = True
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def close(self) -> None:
        """Wake the background task and await its completion.

        Returns:
            None.
        """
        self._active = False
        if self._task is None or self._task.done():
            return
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass

    # ------------------------------------------------------------------
    # Event hooks (synchronous — safe to call from any coroutine)
    # ------------------------------------------------------------------

    def notify(self, reason: LoadReporterRefreshReason, event_count: int = 1) -> None:
        """Accumulate a refresh event and wake the background task.

        Args:
            reason: Highest-priority event type observed by this call.
            event_count: Number of events represented by the call.

        Returns:
            None.

        Counts are summed; reason is updated to the maximum priority value.
        """
        self._accumulated_count += event_count
        if self._accumulated_reason is None or (
            _REASON_PRIORITY[reason] > _REASON_PRIORITY[self._accumulated_reason]
        ):
            self._accumulated_reason = reason
        self._event.set()

    # ------------------------------------------------------------------
    # Background task
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Coalescer loop: wait for event, sleep one window, send once."""
        try:
            while True:
                await self._event.wait()
                # Sleep the coalesce window to gather more events.
                await asyncio.sleep(self._coalesce_window_ms / 1000.0)
                # Atomically swap out accumulated state.
                count = self._accumulated_count
                reason = self._accumulated_reason
                self._accumulated_count = 0
                self._accumulated_reason = None
                self._event.clear()
                # Only send if still active and there is something to send.
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
