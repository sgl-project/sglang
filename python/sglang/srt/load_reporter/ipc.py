"""Control proxy and refresh notifier for the multi-tokenizer load reporter.

This module provides two collaborators used by the worker-side load reporter:

* ``LoadReporterControlProxy`` -- correlates async ``start_reporting`` calls to
  their IPC responses via a ``request_id``-keyed Future dict.  Every non-OK
  ``LoadReporterIpcCode`` is converted to a stable typed exception before
  propagating to the caller.  A configurable timeout cleans up the pending
  Future on expiry.  Cancellation always propagates and never leaks.

* ``LoadReporterRefreshNotifier`` -- a single-background-task coalescer that
  emits at most ONE ``LoadReporterRefreshIpcReq`` per broadcast window.
  Deterministic priority: ABORT > COMPLETION > DISPATCH.  Event counts are
  summed across all ``notify()`` calls within one window.  ``handle_state``
  activates/deactivates the notifier; an ``active=False`` broadcast discards
  any accumulated state so nothing is sent for that window.

Three stable facade exceptions are defined here so callers have a single import
point that does not depend on gRPC or transport internals:

* ``LoadReporterUnavailableError``           (maps to HTTP 503)
* ``LoadReporterDependencyUnavailableError`` (maps to HTTP 501)
* ``LoadReporterInternalError``              (maps to HTTP 500)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Callable, Dict, Final, Optional

from sglang.srt.load_reporter.registration import (
    RuntimeClosingError,
    StartReportingRequest,
    StartReportingResponse,
    WorkerIdentityConflict,
)
from sglang.srt.managers.io_struct import (
    LoadReporterIpcCode,
    LoadReporterRefreshIpcReq,
    LoadReporterRefreshReason,
    LoadReporterStartIpcReqInput,
    LoadReporterStartIpcReqOutput,
    LoadReporterStateBroadcastReq,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

CONTROL_TIMEOUT_SECONDS: Final[float] = 3.0
DEFAULT_COALESCE_WINDOW_MS: Final[int] = 50

# ---------------------------------------------------------------------------
# Stable facade exceptions
# ---------------------------------------------------------------------------


class LoadReporterUnavailableError(Exception):
    """The load reporter owner did not respond in time (maps to HTTP 503)."""


class LoadReporterDependencyUnavailableError(Exception):
    """A downstream dependency of the load reporter is unavailable (HTTP 501)."""


class LoadReporterInternalError(Exception):
    """The load reporter encountered an unexpected internal error (HTTP 500)."""


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
# IPC-code → exception conversion
# ---------------------------------------------------------------------------


def _ipc_code_to_exception(
    code: LoadReporterIpcCode, message: Optional[str]
) -> Exception:
    """Convert a non-OK IPC code to the appropriate typed exception.

    CONFLICT is mapped to ``WorkerIdentityConflict`` so the existing HTTP 409
    arm in ``registration.py`` fires without modification. A full
    ``MonitorKey`` is unavailable at the proxy boundary, so the owner-provided
    message is retained and ``key`` remains ``None``.
    """
    detail = message or "load reporter error"
    if code is LoadReporterIpcCode.CONFLICT:
        return WorkerIdentityConflict(message=detail)
    if code is LoadReporterIpcCode.CLOSING:
        return RuntimeClosingError(detail)
    if code is LoadReporterIpcCode.UNAVAILABLE:
        return LoadReporterUnavailableError(detail)
    if code is LoadReporterIpcCode.DEPENDENCY_UNAVAILABLE:
        return LoadReporterDependencyUnavailableError(detail)
    if code is LoadReporterIpcCode.INTERNAL:
        return LoadReporterInternalError(detail)
    # Unknown future codes: treat as internal rather than silently swallowing.
    return LoadReporterInternalError(f"unhandled IPC code {code!r}: {detail}")


# ---------------------------------------------------------------------------
# LoadReporterControlProxy
# ---------------------------------------------------------------------------


class LoadReporterControlProxy:
    """Correlates async start_reporting calls to IPC responses by request_id.

    Each call to ``start_reporting`` allocates a UUID ``request_id``, stores
    an ``asyncio.Future`` in ``_pending``, and sends the
    ``LoadReporterStartIpcReqInput`` via the injected ``send`` callable.  When
    the corresponding ``LoadReporterStartIpcReqOutput`` arrives (via
    ``handle_response``), the future is resolved.  A ``timeout_seconds``-
    bounded ``wait_for`` with a shielded future ensures the pending dict is
    always cleaned up — on timeout, on cancellation, and on success.

    Cancellation contract: ``asyncio.CancelledError`` is NEVER caught; it
    propagates out of ``start_reporting`` after the ``finally`` block cleans up
    the future.
    """

    def __init__(
        self,
        send: Callable[[Any], None],
        *,
        timeout_seconds: float = CONTROL_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the HTTP-worker control facade.

        Args:
            send: Synchronous IPC dispatch callback.
            timeout_seconds: Maximum time to wait for the router owner.

        Returns:
            None.
        """
        self._send = send
        self._timeout_seconds = timeout_seconds
        self._pending: Dict[str, asyncio.Future[LoadReporterStartIpcReqOutput]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_reporting(
        self, payload: StartReportingRequest, worker_addr: str
    ) -> StartReportingResponse:
        """Send a start-reporting IPC request and await its response.

        Args:
            payload: Validated Router target and lease settings.
            worker_addr: Canonical identity of the reporting worker.

        Returns:
            The owner-accepted lease and renewal timing.

        Raises:
            LoadReporterUnavailableError: on timeout.
            WorkerIdentityConflict: on CONFLICT code.
            RuntimeClosingError: on CLOSING code.
            LoadReporterDependencyUnavailableError: on DEPENDENCY_UNAVAILABLE.
            LoadReporterInternalError: on INTERNAL or unknown code.
            asyncio.CancelledError: if the caller cancels the task.
        """
        request_id = uuid.uuid4().hex
        request = LoadReporterStartIpcReqInput(
            request_id=request_id,
            router_host=str(payload.ip),
            router_port=payload.port,
            report_interval_ms=payload.report_interval_ms,
            lease_ttl_ms=payload.lease_ttl_ms,
            worker_addr=worker_addr,
        )
        future: asyncio.Future[LoadReporterStartIpcReqOutput] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[request_id] = future
        self._send(request)
        try:
            response = await asyncio.wait_for(
                asyncio.shield(future), self._timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            raise LoadReporterUnavailableError("load reporter owner timed out") from exc
        finally:
            self._pending.pop(request_id, None)
            if not future.done():
                future.cancel()

        if response.code is LoadReporterIpcCode.OK:
            return StartReportingResponse(
                status=response.status or "reporting",
                lease_ttl_ms=response.lease_ttl_ms or 0,
                renew_after_ms=response.renew_after_ms or 0,
            )
        raise _ipc_code_to_exception(response.code, response.message)

    def handle_response(self, response: LoadReporterStartIpcReqOutput) -> None:
        """Resolve the pending future for the given response.request_id.

        If the request_id is not found (stale or spurious response), a warning
        is logged and no other pending future is disturbed.
        """
        future = self._pending.get(response.request_id)
        if future is None:
            logger.warning(
                "load reporter: received response for unknown request_id %r",
                response.request_id,
            )
            return
        if not future.done():
            future.set_result(response)

    @property
    def pending_count(self) -> int:
        """Number of requests currently awaiting a response."""
        return len(self._pending)

    async def close(self) -> None:
        """Cancel and remove all pending futures."""
        for future in list(self._pending.values()):
            if not future.done():
                future.cancel()
        self._pending.clear()


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
    * ``handle_state(active=False)`` clears accumulated state before the window
      fires, suppressing the message for that window.

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
        # Coalesce-window duration in milliseconds; updated by handle_state.
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
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def close(self) -> None:
        """Wake the background task and await its completion.

        Returns:
            None.
        """
        if self._task is None or self._task.done():
            return
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass

    # ------------------------------------------------------------------
    # State / event hooks (synchronous — safe to call from any coroutine)
    # ------------------------------------------------------------------

    def handle_state(self, state: LoadReporterStateBroadcastReq) -> None:
        """React to a broadcaster state update.

        Args:
            state: Router-owned active state and coalescing window.

        Returns:
            None.

        On ``active=True``: update the window and enable sending.
        On ``active=False``: disable sending and discard any accumulated state
        so no message is sent for the current window.
        """
        self._coalesce_window_ms = state.coalesce_window_ms
        self._active = state.active
        if not state.active:
            # Discard accumulated state — nothing should be sent for this window.
            self._accumulated_count = 0
            self._accumulated_reason = None
            self._event.clear()

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
