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
    SNAPSHOT_STALE_AFTER_MS,
    WorkerMetadata,
    validate_session_timing,
)
from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
from sglang.srt.load_reporter.report_builder import ReportBuilder
from sglang.srt.load_reporter.snapshot_source import LoadSnapshotSource
from sglang.srt.load_reporter.snapshot_validation import (
    RankSetMismatchError,
    validate_full_snapshot,
)

logger = logging.getLogger(__name__)


class _RouterSession:
    """One inbound Router stream: passive lease state plus a response queue."""

    def __init__(
        self,
        router_id: str,
        report_interval_ms: int,
        lease_ttl_ms: int,
        on_close: Callable[[str, _RouterSession], None],
        state_changed: Callable[[], None],
    ) -> None:
        now = time.monotonic()
        self._router_id = router_id
        self._report_interval_ms = report_interval_ms
        self._lease_ttl_ms = lease_ttl_ms
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

    def _update_timing(
        self,
        report_interval_ms: Optional[int] = None,
        lease_ttl_ms: Optional[int] = None,
        *,
        now: float,
    ) -> None:
        """Apply validated timing values; Runtime owns schedule membership."""
        if report_interval_ms is not None:
            self._report_interval_ms = report_interval_ms
        if lease_ttl_ms is not None:
            self._lease_ttl_ms = lease_ttl_ms
        self._lease_expires_at = now + self._lease_ttl_ms / 1000.0

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


class _PeriodSchedule:
    """One shared periodic deadline for all sessions using an interval."""

    def __init__(self, interval_ms: int, schedule_epoch: float, now: float) -> None:
        self._interval_seconds = interval_ms / 1000.0
        self._schedule_epoch = schedule_epoch
        self.next_deadline = self._next_aligned_deadline(now)
        self.sessions: set[_RouterSession] = set()

    def _next_aligned_deadline(self, now: float) -> float:
        """Return the first reporter-epoch boundary strictly after ``now``."""
        elapsed_seconds = max(0.0, now - self._schedule_epoch)
        elapsed_periods = int(elapsed_seconds / self._interval_seconds)
        deadline = self._schedule_epoch + (elapsed_periods + 1) * self._interval_seconds
        if deadline <= now:
            deadline += self._interval_seconds
        return deadline

    def advance(self, now: float) -> None:
        """Advance one tick, skipping missed ticks without changing phase."""
        self.next_deadline = self._next_aligned_deadline(now)


class LoadReporterRuntime:
    """Own Router sessions, shared period schedules, and one pull timer."""

    def __init__(
        self,
        snapshot_source: LoadSnapshotSource,
        server_args: Any,
    ) -> None:
        """Assemble reporter collaborators around one snapshot source."""
        self._closing = False
        self._close_task: Optional[asyncio.Task[None]] = None
        self._worker_metadata = WorkerMetadata.from_server_args(server_args)
        self._snapshot_source = snapshot_source
        self._schedule_epoch = time.monotonic()

        self._builder = ReportBuilder(
            str(uuid.uuid4()),
            SNAPSHOT_STALE_AFTER_MS,
        )
        self._sessions: Dict[str, _RouterSession] = {}
        self._period_schedules: Dict[int, _PeriodSchedule] = {}
        self._initial_due_sessions: set[_RouterSession] = set()
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
        validate_session_timing(report_interval_ms, lease_ttl_ms)

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
        self._add_to_period_schedule(session, now=time.monotonic())
        # Initial delivery is independent of the periodic schedule. Joining an
        # existing period must not reset that period's shared deadline.
        self._initial_due_sessions.add(session)
        self._state_changed.set()

        renew_after_ms = max(1, lease_ttl_ms // 3)
        ack = pb.RegisterResponse(
            lease_ttl_ms=lease_ttl_ms,
            renew_after_ms=renew_after_ms,
        )
        return ack, session

    def _on_session_closed(self, router_id: str, session: _RouterSession) -> None:
        """Remove session-owned state without disturbing a replacement."""
        self._initial_due_sessions.discard(session)
        self._remove_from_period_schedule(session)
        if self._sessions.get(router_id) is session:
            del self._sessions[router_id]

    def _add_to_period_schedule(
        self, session: _RouterSession, *, now: float
    ) -> _PeriodSchedule:
        """Join the interval bucket, creating its aligned timer when needed."""
        interval_ms = session.report_interval_ms
        schedule = self._period_schedules.get(interval_ms)
        if schedule is None:
            schedule = _PeriodSchedule(interval_ms, self._schedule_epoch, now)
            self._period_schedules[interval_ms] = schedule
        schedule.sessions.add(session)
        return schedule

    def _remove_from_period_schedule(self, session: _RouterSession) -> None:
        """Leave the current interval bucket and delete it when empty."""
        interval_ms = session.report_interval_ms
        schedule = self._period_schedules.get(interval_ms)
        if schedule is None:
            return
        schedule.sessions.discard(session)
        if not schedule.sessions:
            del self._period_schedules[interval_ms]

    def update_session_config(
        self,
        session: _RouterSession,
        report_interval_ms: Optional[int] = None,
        lease_ttl_ms: Optional[int] = None,
    ) -> bool:
        """Apply timing atomically and migrate interval-bucket membership."""
        validate_session_timing(report_interval_ms, lease_ttl_ms)
        if self._sessions.get(session._router_id) is not session:
            return False
        if report_interval_ms is None and lease_ttl_ms is None:
            return False

        now = time.monotonic()
        interval_changed = (
            report_interval_ms is not None
            and report_interval_ms != session.report_interval_ms
        )
        if interval_changed:
            self._remove_from_period_schedule(session)
        session._update_timing(
            report_interval_ms=report_interval_ms,
            lease_ttl_ms=lease_ttl_ms,
            now=now,
        )
        if interval_changed:
            self._add_to_period_schedule(session, now=now)
        self._state_changed.set()
        return True

    def update_expected_dp_ranks(self, expected_dp_ranks: Iterable[int]) -> bool:
        """Update a rank-aware snapshot source after elastic scaling."""
        update = getattr(self._snapshot_source, "update_expected_dp_ranks", None)
        return bool(update is not None and update(expected_dp_ranks))

    def _collect_due(
        self, now: float
    ) -> Tuple[list[_PeriodSchedule], set[_RouterSession]]:
        """Return current due periods and the union of their sessions."""
        schedules = [
            schedule
            for schedule in self._period_schedules.values()
            if schedule.next_deadline <= now
        ]
        sessions = set(self._initial_due_sessions)
        for schedule in schedules:
            sessions.update(schedule.sessions)
        return schedules, sessions

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
        """Pull once for all due periods, then broadcast to their sessions."""
        while not self._closing:
            self._state_changed.clear()
            now = time.monotonic()
            for session in list(self._sessions.values()):
                if now >= session._lease_expires_at:
                    session.stop()
            if not self._sessions:
                await self._state_changed.wait()
                continue

            if not self._initial_due_sessions:
                next_fire = min(
                    min(
                        schedule.next_deadline
                        for schedule in self._period_schedules.values()
                    ),
                    min(s._lease_expires_at for s in self._sessions.values()),
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
            _, due_sessions = self._collect_due(now)
            due_sessions = {
                session
                for session in due_sessions
                if not session._done.is_set() and now < session._lease_expires_at
            }
            if not due_sessions:
                continue  # fired for lease expiry; the loop reaps them

            # Pull with the source-level budget; leases only gate delivery.
            report = await self._pull_report(SNAPSHOT_PULL_TIMEOUT_SECONDS)
            if self._closing:
                break

            # Recompute after the await so registration or interval changes
            # during a pull use the current period membership.
            now = time.monotonic()
            due_schedules, due_sessions = self._collect_due(now)

            delivered = set()
            for session in due_sessions:
                if session._done.is_set() or now >= session._lease_expires_at:
                    continue
                session._enqueue(report)
                delivered.add(session)
            self._initial_due_sessions.difference_update(delivered)
            for schedule in due_schedules:
                schedule.advance(now)

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
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, SHUTDOWN_TIMEOUT_SECONDS)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        self._sessions.clear()
        self._period_schedules.clear()
        self._initial_due_sessions.clear()
