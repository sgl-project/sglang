"""Load reporter runtime: one shared fire loop broadcasting snapshot pulls."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from sglang.srt.load_reporter.config import (
    SHUTDOWN_TIMEOUT_SECONDS,
    SNAPSHOT_PULL_TIMEOUT_SECONDS,
    LoadReporterConfig,
    WorkerMetadata,
)
from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
from sglang.srt.load_reporter.report_builder import ReportBuilder, SequenceAllocator
from sglang.srt.load_reporter.snapshot_validation import (
    RankSetMismatchError,
    validate_full_snapshot,
)

logger = logging.getLogger(__name__)


def _validate_timing(
    report_interval_ms: Optional[int] = None,
    lease_ttl_ms: Optional[int] = None,
) -> None:
    """Reject non-positive session timing before mutating runtime state."""
    if report_interval_ms is not None and report_interval_ms <= 0:
        raise ValueError("report_interval_ms must be greater than zero")
    if lease_ttl_ms is not None and lease_ttl_ms <= 0:
        raise ValueError("lease_ttl_ms must be greater than zero")


class _RouterSession:
    """One inbound Router stream: passive bookkeeping plus a response queue."""

    def __init__(
        self,
        router_id: str,
        report_interval_ms: int,
        lease_ttl_ms: int,
        on_close: Callable[[str, _RouterSession], None],
        state_changed: Callable[[], None],
    ) -> None:
        _validate_timing(report_interval_ms, lease_ttl_ms)
        now = time.monotonic()
        self._router_id = router_id
        self._report_interval_ms = report_interval_ms
        self._lease_ttl_ms = lease_ttl_ms
        # Due immediately: the next fire delivers the initial report.
        self._next_report_deadline = now
        self._lease_expires_at = now + lease_ttl_ms / 1000.0
        self._on_close = on_close
        self._state_changed = state_changed

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._done: asyncio.Event = asyncio.Event()

    @property
    def queue(self) -> asyncio.Queue:
        """Response queue; contains LoadReport or None (session ended)."""
        return self._queue

    @property
    def report_interval_ms(self) -> int:
        """Current report cadence for this session."""
        return self._report_interval_ms

    def refresh_lease(self) -> None:
        """Reset lease timer to now + current lease_ttl_ms."""
        self._lease_expires_at = time.monotonic() + self._lease_ttl_ms / 1000.0

    def update_config(
        self,
        report_interval_ms: Optional[int] = None,
        lease_ttl_ms: Optional[int] = None,
    ) -> None:
        """Atomically update timing and re-anchor schedules from now."""
        _validate_timing(report_interval_ms, lease_ttl_ms)
        now = time.monotonic()

        if report_interval_ms is not None:
            self._report_interval_ms = report_interval_ms
            self._next_report_deadline = now + report_interval_ms / 1000.0
        if lease_ttl_ms is not None:
            self._lease_ttl_ms = lease_ttl_ms
        self._lease_expires_at = now + self._lease_ttl_ms / 1000.0

        if report_interval_ms is not None or lease_ttl_ms is not None:
            self._state_changed()

    def stop(self) -> None:
        """Idempotent stop: close the queue and deregister the session."""
        if self._done.is_set():
            return
        self._done.set()
        self._enqueue(None)  # sentinel: write loop exits
        self._state_changed()
        try:
            self._on_close(self._router_id, self)
        except Exception:
            logger.exception(
                "on_close callback failed for router_id=%s", self._router_id
            )

    def _enqueue(self, item: Any) -> None:
        """Latest-wins enqueue: drop old item if queue is full."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            pass  # consumed between check and put — skip

    def _advance_deadline(self, now: float) -> None:
        """Move the deadline forward one interval; re-anchor when behind."""
        interval_sec = self._report_interval_ms / 1000.0
        self._next_report_deadline += interval_sec
        if self._next_report_deadline <= now:
            self._next_report_deadline = now + interval_sec


class LoadReporterRuntime:
    """Own reporter sessions and one shared pull timer (the fire loop)."""

    def __init__(
        self,
        snapshot_source: Any,
        server_args: Any,
    ) -> None:
        """Assemble reporter collaborators around one snapshot source."""
        self._closing = False
        self._close_task: Optional[asyncio.Task[None]] = None
        self._config = LoadReporterConfig.from_server_args(server_args)
        self._worker_metadata = WorkerMetadata.from_server_args(server_args)
        self._snapshot_source = snapshot_source

        self._builder = ReportBuilder(
            str(uuid.uuid4()),
            self._config.snapshot_stale_after_ms,
            SequenceAllocator(),
        )
        self._sessions: Dict[str, _RouterSession] = {}
        self._state_changed: asyncio.Event = asyncio.Event()
        self._fire_task: asyncio.Task = asyncio.create_task(
            self._fire_loop(), name="load-reporter-fire-loop"
        )

    def register_session(
        self,
        router_id: str,
        report_interval_ms: int,
        lease_ttl_ms: int,
    ) -> Tuple[pb.RegisterResponse, _RouterSession]:
        """Register or replace a Router session and return its acknowledgement."""
        if self._closing:
            raise RuntimeError("load reporter is shutting down")
        if not router_id or not router_id.strip():
            raise ValueError("router_id must be non-empty")
        _validate_timing(report_interval_ms, lease_ttl_ms)

        # Replace any existing session for this router_id.
        old = self._sessions.pop(router_id, None)
        if old is not None:
            old.stop()

        session = _RouterSession(
            router_id=router_id,
            report_interval_ms=report_interval_ms,
            lease_ttl_ms=lease_ttl_ms,
            on_close=self._on_session_closed,
            state_changed=self._state_changed.set,
        )
        self._sessions[router_id] = session
        self._state_changed.set()

        renew_after_ms = max(1, lease_ttl_ms // 3)
        ack = pb.RegisterResponse(
            lease_ttl_ms=lease_ttl_ms,
            renew_after_ms=renew_after_ms,
        )
        return ack, session

    def _on_session_closed(self, router_id: str, session: _RouterSession) -> None:
        """Remove a session only when it still owns its Router ID."""
        if self._sessions.get(router_id) is session:
            del self._sessions[router_id]

    def update_expected_dp_ranks(self, expected_dp_ranks: Iterable[int]) -> bool:
        """Update a rank-aware snapshot source after elastic scaling."""
        update = getattr(self._snapshot_source, "update_expected_dp_ranks", None)
        return bool(update is not None and update(expected_dp_ranks))

    async def _pull_report(self, timeout_seconds: float) -> pb.LoadReport:
        """Pull once (retrying one rank-set change), build one report."""
        report_time_unix_ms = time.time_ns() // 1_000_000
        # One fire budget shared by the initial attempt and its one retry.
        deadline = time.monotonic() + timeout_seconds
        try:
            for attempt in (1, 2):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                try:
                    loads = await asyncio.wait_for(
                        self._snapshot_source.get_loads(), timeout=remaining
                    )
                    report_time_unix_ms = time.time_ns() // 1_000_000
                    ranks = validate_full_snapshot(
                        loads,
                        expected_dp_ranks=self._snapshot_source.expected_dp_ranks(),
                        fallback_time_unix_ms=report_time_unix_ms,
                    )
                    break
                except RankSetMismatchError as exc:
                    if attempt == 2:
                        raise
                    logger.warning(
                        "DP rank set changed during pull; retrying once: %s", exc
                    )
            return self._builder.build(
                ranks,
                self._worker_metadata,
                report_time_unix_ms=report_time_unix_ms,
            )
        except asyncio.TimeoutError:
            error: BaseException | str = "load snapshot pull timed out"
        except Exception as exc:
            logger.warning("Load reporter snapshot pull failed: %s", exc)
            error = exc

        return self._builder.build_unreachable(
            self._worker_metadata,
            report_time_unix_ms=time.time_ns() // 1_000_000,
            error=error,
        )

    async def _fire_loop(self) -> None:
        """The reporter's single timer: pull once per fire, broadcast to due sessions."""
        try:
            while not self._closing:
                self._state_changed.clear()
                now = time.monotonic()
                for session in list(self._sessions.values()):
                    if now >= session._lease_expires_at:
                        session.stop()
                if not self._sessions:
                    await self._state_changed.wait()
                    continue

                next_fire = min(
                    min(s._next_report_deadline, s._lease_expires_at)
                    for s in self._sessions.values()
                )
                sleep_sec = max(0.0, next_fire - time.monotonic())
                try:
                    await asyncio.wait_for(
                        self._state_changed.wait(), timeout=sleep_sec
                    )
                except asyncio.TimeoutError:
                    pass
                if self._state_changed.is_set():
                    continue  # membership or timing changed; recompute

                now = time.monotonic()
                due = [
                    s
                    for s in self._sessions.values()
                    if s._next_report_deadline <= now and not s._done.is_set()
                ]
                if not due:
                    continue  # fired for lease expiry; the loop reaps them

                # Pull with the source-level budget; leases only gate delivery.
                report = await self._pull_report(SNAPSHOT_PULL_TIMEOUT_SECONDS)
                if self._closing:
                    break

                # Broadcast to every session due at completion time (covers sessions registered during the pull).
                now = time.monotonic()
                for session in list(self._sessions.values()):
                    if (
                        session._done.is_set()
                        or now >= session._lease_expires_at
                        or session._next_report_deadline > now
                    ):
                        continue
                    session._enqueue(report)
                    session._advance_deadline(now)
        except asyncio.CancelledError:
            raise

    async def close(self) -> None:
        """Bounded, idempotent shutdown."""
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(
                self._close_impl(), name="load-reporter-runtime-close"
            )
        await asyncio.shield(self._close_task)

    async def _close_impl(self) -> None:
        """Stop every session, then cancel the fire task directly."""
        self._closing = True
        self._state_changed.set()
        for session in list(self._sessions.values()):
            session.stop()
        task = self._fire_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, SHUTDOWN_TIMEOUT_SECONDS)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        self._sessions.clear()
