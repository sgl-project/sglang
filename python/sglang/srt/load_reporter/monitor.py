"""Per-target gRPC monitors and the map that owns them.

``MonitorManager`` owns the ``MonitorKey -> MonitorTask`` map, performs strict
identity-safe lease upserts, and tracks the cached minimum report interval.
``MonitorTask`` is the self-contained data-plane state machine: exactly one
``grpc.aio`` h2c client stream per Router target, a fixed-rate report deadline,
bounded-jitter reconnect, and lease/stop-driven teardown. Each task belongs to
one manager-assigned generation so a terminating task cannot delete a newer
registration for the same target.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import logging
import random
import time
from typing import Awaitable, Callable, Dict, Optional

import grpc

from sglang.srt.load_reporter.config import (
    GRPC_CLOSE_TIMEOUT_SECONDS,
    GRPC_CONNECT_TIMEOUT_SECONDS,
    RECONNECT_INITIAL_SECONDS,
    RECONNECT_MAX_SECONDS,
)
from sglang.srt.load_reporter.proto import load_monitor_pb2_grpc as pb_grpc
from sglang.srt.load_reporter.registration import (
    MonitorKey,
    MonitorRegistration,
    StartReportingRequest,
    StartReportingResponse,
    WorkerIdentityConflict,
)
from sglang.srt.load_reporter.report_builder import ReportBuilder, WorkerIdentity
from sglang.srt.load_reporter.store import LatestSnapshotStore

logger = logging.getLogger(__name__)


# gRPC status codes worth retrying with backoff on the same registration.
_RETRYABLE = {
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
}
# Permanent-for-this-registration codes: stop reconnecting and wait until the
# Router renews the lease (a new revision) before trying again.
_WAIT_FOR_RENEWAL = {
    grpc.StatusCode.INVALID_ARGUMENT,
    grpc.StatusCode.UNAUTHENTICATED,
    grpc.StatusCode.PERMISSION_DENIED,
    grpc.StatusCode.UNIMPLEMENTED,
}


class _StatusAction(enum.Enum):
    """Exhaustive lifecycle action for a terminal gRPC status."""

    RETRY = enum.auto()
    WAIT_FOR_RENEWAL = enum.auto()
    TERMINATE = enum.auto()


def _classify_status(code: grpc.StatusCode) -> _StatusAction:
    """Classify known transient/renewable statuses, failing closed otherwise."""
    if code in _RETRYABLE:
        return _StatusAction.RETRY
    if code in _WAIT_FOR_RENEWAL:
        return _StatusAction.WAIT_FOR_RENEWAL
    return _StatusAction.TERMINATE


def _next_backoff(current_seconds: float, random_value: float) -> tuple[float, float]:
    """Return ``(sleep_seconds, next_base)`` with +-20% jitter, capped."""
    bounded = min(current_seconds, RECONNECT_MAX_SECONDS)
    jittered_seconds = bounded * (0.8 + 0.4 * random_value)
    return jittered_seconds, min(bounded * 2, RECONNECT_MAX_SECONDS)


class _StopRequested(Exception):
    """Internal signal: stop()/lease-expiry preempted an in-flight await."""


class _RetryConnection(Exception):
    """Internal signal: an explicitly transient gRPC status may reconnect."""

    def __init__(self, code: grpc.StatusCode) -> None:
        """Initialize a retry signal for one transient gRPC status."""
        super().__init__(str(code))
        self.code = code


class _WaitForRenewal(Exception):
    """Internal signal: permanent-for-this-registration gRPC status."""

    def __init__(self, code: grpc.StatusCode, rejected_revision: int) -> None:
        """Initialize a renewal wait for one rejected registration revision."""
        super().__init__(str(code))
        self.code = code
        self.rejected_revision = rejected_revision


class _TerminateMonitor(Exception):
    """Internal signal: a non-recoverable gRPC status must fail closed."""

    def __init__(self, code: grpc.StatusCode) -> None:
        """Initialize a terminal signal for one non-recoverable gRPC status."""
        super().__init__(str(code))
        self.code = code


def _status_signal(code: grpc.StatusCode, rejected_revision: int) -> Exception:
    """Build the lifecycle control signal for one terminal gRPC status."""
    action = _classify_status(code)
    if action is _StatusAction.RETRY:
        return _RetryConnection(code)
    if action is _StatusAction.WAIT_FOR_RENEWAL:
        return _WaitForRenewal(code, rejected_revision)
    return _TerminateMonitor(code)


class MonitorTask:
    """One Router target: one channel, one client stream, one fixed-rate loop."""

    def __init__(
        self,
        registration: MonitorRegistration,
        store: LatestSnapshotStore,
        builder: ReportBuilder,
        on_stopped: Callable[[MonitorKey, int], Awaitable[None]],
        *,
        generation: int,
        monotonic: Callable[[], float] = time.monotonic,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        """Initialize one generation-owned Router stream state machine.

        Args:
            registration: Initial target, identity, interval, and lease state.
            store: Latest validated snapshots used to build reports.
            builder: Pure snapshot-to-protobuf report builder.
            on_stopped: Callback that removes this generation from its manager.
            generation: Manager-assigned ownership generation.
            monotonic: Injectable monotonic clock for deadlines.
            random_value: Injectable random value for reconnect jitter.

        Returns:
            None.
        """
        self._registration = registration
        self._store = store
        self._builder = builder
        self._on_stopped = on_stopped
        self._generation = generation
        self._monotonic = monotonic
        self._random_value = random_value

        self._accepting_updates = True
        self._stop_event = asyncio.Event()
        self._updated_event = asyncio.Event()
        self._io_deadline_updated_event = asyncio.Event()
        self._channel: Optional[grpc.aio.Channel] = None
        self._call = None
        self._connected_this_epoch = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def registration(self) -> MonitorRegistration:
        """Return the monitor's current immutable registration."""
        return self._registration

    @property
    def generation(self) -> int:
        """Return the manager-assigned ownership generation."""
        return self._generation

    @property
    def accepting_updates(self) -> bool:
        """Return whether this generation can accept a lease renewal."""
        return self._accepting_updates

    def try_update_registration(self, registration: MonitorRegistration) -> bool:
        """Apply a live in-generation update, or reject terminal ownership."""
        if (
            not self._accepting_updates
            or self._stop_event.is_set()
            or self._lease_expired()
        ):
            self._accepting_updates = False
            return False
        self._registration = registration
        self._updated_event.set()
        self._io_deadline_updated_event.set()
        return True

    async def stop(self) -> None:
        """Request teardown; returns once ``run()`` has converged.

        Never sends a final report. The waiting is done by the manager, which
        awaits the backing task after calling this.
        """
        self._accepting_updates = False
        self._stop_event.set()
        self._updated_event.set()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Own this target's lifecycle until stopped; never leak an exception."""
        backoff = RECONNECT_INITIAL_SECONDS
        try:
            while not self._stop_event.is_set():
                if self._lease_expired():
                    break

                self._connected_this_epoch = False
                try:
                    await self._run_connection_epoch()
                except _StopRequested:
                    break
                except _WaitForRenewal as exc:
                    # Close the dead stream before the (possibly long) wait; the
                    # trailing finally's _close_epoch() is then a no-op.
                    await self._close_epoch()
                    logger.warning(
                        "Load monitor %s got %s; waiting for lease renewal",
                        self._registration.key.authority,
                        exc.code,
                    )
                    self._store.record_error(f"router rejected report: {exc.code}")
                    if not await self._wait_for_renewal(exc.rejected_revision):
                        break
                    backoff = RECONNECT_INITIAL_SECONDS
                    continue
                except _TerminateMonitor as exc:
                    logger.warning(
                        "Load monitor %s got terminal status %s; stopping",
                        self._registration.key.authority,
                        exc.code,
                    )
                    self._store.record_error(
                        f"router terminated report stream: {exc.code}"
                    )
                    break
                except _RetryConnection as exc:
                    logger.debug(
                        "Load monitor %s got retryable status %s",
                        self._registration.key.authority,
                        exc.code,
                    )
                except Exception as exc:
                    # Non-status connect or transport failures remain retryable.
                    logger.debug(
                        "Load monitor %s stream ended: %s",
                        self._registration.key.authority,
                        exc,
                    )
                finally:
                    await self._close_epoch()

                if self._stop_event.is_set() or self._lease_expired():
                    break

                if self._connected_this_epoch:
                    # A healthy epoch resets the reconnect sequence.
                    backoff = RECONNECT_INITIAL_SECONDS
                sleep_seconds, backoff = _next_backoff(backoff, self._random_value())
                if not await self._sleep_preemptible(sleep_seconds):
                    break
        finally:
            self._accepting_updates = False
            await self._close_epoch()
            with contextlib.suppress(Exception):
                await self._on_stopped(
                    self._registration.key,
                    self._generation,
                )

    # ------------------------------------------------------------------
    # One connection attempt
    # ------------------------------------------------------------------

    async def _run_connection_epoch(self) -> None:
        """Connect one stream, send immediately, and serve until preempted.

        Returns:
            None.

        Raises:
            _StopRequested: If stop or lease expiry preempts I/O.
            _RetryConnection: If a transient gRPC status ends the stream.
            _WaitForRenewal: If a new registration revision is required.
            _TerminateMonitor: If a non-recoverable status ends the monitor.
        """
        epoch_revision = self._registration.revision
        try:
            channel = grpc.aio.insecure_channel(self._registration.key.authority)
            self._channel = channel
            await self._await_preemptible(
                channel.channel_ready(),
                operation_timeout=GRPC_CONNECT_TIMEOUT_SECONDS,
            )
            self._call = pb_grpc.LoadMonitorServiceStub(channel).Report()
            self._connected_this_epoch = True

            # Send once immediately on (re)connect, then hold the fixed rate.
            await self._write_current(self._call)
            interval = self._registration.report_interval_ms / 1000.0
            next_report_at = self._monotonic() + interval

            await self._serve_stream(
                self._call,
                next_report_at,
                epoch_revision,
            )
        except grpc.aio.AioRpcError as exc:
            raise _status_signal(exc.code(), epoch_revision) from exc

    async def _serve_stream(
        self, call, next_report_at: float, epoch_revision: int
    ) -> None:
        """Drive one open stream: fixed-rate writes until preempted or done."""
        completion = asyncio.ensure_future(call.code())
        try:
            while True:
                now = self._monotonic()
                lease_remaining = self._registration.lease_expires_at - now
                report_remaining = next_report_at - now
                timeout = max(0.0, min(lease_remaining, report_remaining))

                await self._wait_any(completion, timeout)

                if self._stop_event.is_set() or self._lease_expired():
                    raise _StopRequested()
                if completion.done():
                    # Stream ended: surface its terminal status for classification.
                    raise _status_signal(await call.code(), epoch_revision)
                if self._updated_event.is_set():
                    self._updated_event.clear()
                    # Re-anchor the deadline off the registration update time.
                    interval = self._registration.report_interval_ms / 1000.0
                    next_report_at = self._registration.updated_at + interval
                    continue
                if self._monotonic() >= next_report_at:
                    await self._write_current(call)
                    # Skip any missed periods rather than bursting to catch up.
                    interval = self._registration.report_interval_ms / 1000.0
                    next_report_at = self._monotonic() + interval
        finally:
            completion.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await completion

    async def _write_current(self, call) -> None:
        """Build and write the latest report to an open stream.

        Args:
            call: Active gRPC client-streaming call.

        Returns:
            None.
        """
        report = self._builder.build(
            self._store.view(),
            self._registration.worker_identity,
            report_time_unix_ms=time.time_ns() // 1_000_000,
        )
        await self._await_preemptible(call.write(report))

    # ------------------------------------------------------------------
    # Waiting / preemption helpers
    # ------------------------------------------------------------------

    async def _wait_any(self, completion: asyncio.Future, timeout: float) -> None:
        """Wake on stop, registration update, stream completion, or timeout."""
        waiters = {
            asyncio.ensure_future(self._stop_event.wait()),
            asyncio.ensure_future(self._updated_event.wait()),
            completion,
        }
        try:
            await asyncio.wait(
                waiters,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for waiter in waiters:
                if waiter is not completion and not waiter.done():
                    waiter.cancel()
                    with contextlib.suppress(Exception, asyncio.CancelledError):
                        await waiter

    async def _await_preemptible(
        self, awaitable: Awaitable, *, operation_timeout: Optional[float] = None
    ):
        """Await ``awaitable`` but abort if stop/lease-expiry fires first."""
        operation_deadline = (
            None
            if operation_timeout is None
            else self._monotonic() + max(0.0, operation_timeout)
        )
        op = asyncio.ensure_future(_as_coro(awaitable))
        try:
            while True:
                if op.done():
                    return op.result()
                if self._stop_event.is_set():
                    raise _StopRequested()

                registration = self._registration
                observed_revision = registration.revision
                now = self._monotonic()
                lease_remaining = registration.lease_expires_at - now
                operation_remaining = (
                    None if operation_deadline is None else operation_deadline - now
                )
                if lease_remaining <= 0:
                    raise _StopRequested()
                if operation_remaining is not None and operation_remaining <= 0:
                    raise asyncio.TimeoutError()

                # Keep deadline wakeups separate from the connection loop's
                # report-schedule event so neither consumer loses the other's
                # signal. The revision predicate closes the check/clear race.
                self._io_deadline_updated_event.clear()
                if self._registration.revision != observed_revision:
                    continue

                stop = asyncio.ensure_future(self._stop_event.wait())
                deadline_updated = asyncio.ensure_future(
                    self._io_deadline_updated_event.wait()
                )
                timeout = lease_remaining
                if operation_remaining is not None:
                    timeout = min(timeout, operation_remaining)
                try:
                    done, _ = await asyncio.wait(
                        {op, stop, deadline_updated},
                        timeout=max(0.0, timeout),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if op in done:
                        return op.result()
                    if stop in done:
                        raise _StopRequested()
                    # An update or elapsed deadline is classified at the top of
                    # the loop against the latest registration and fixed
                    # operation deadline.
                finally:
                    for waiter in (stop, deadline_updated):
                        if not waiter.done():
                            waiter.cancel()
                            with contextlib.suppress(Exception, asyncio.CancelledError):
                                await waiter
        finally:
            if not op.done():
                op.cancel()
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await op

    async def _sleep_preemptible(self, seconds: float) -> bool:
        """Sleep unless stop/update fires. Returns False if we should exit."""
        stop = asyncio.ensure_future(self._stop_event.wait())
        updated = asyncio.ensure_future(self._updated_event.wait())
        lease_remaining = self._registration.lease_expires_at - self._monotonic()
        try:
            await asyncio.wait(
                {stop, updated},
                timeout=max(0.0, min(seconds, lease_remaining)),
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for fut in (stop, updated):
                if not fut.done():
                    fut.cancel()
                    with contextlib.suppress(Exception, asyncio.CancelledError):
                        await fut
        return not self._stop_event.is_set() and not self._lease_expired()

    async def _wait_for_renewal(self, rejected_revision: int) -> bool:
        """Wait for a higher revision, stop, or the current lease deadline."""
        while not self._stop_event.is_set():
            registration = self._registration
            if registration.revision > rejected_revision:
                return True

            lease_remaining = registration.lease_expires_at - self._monotonic()
            if lease_remaining <= 0:
                return False

            self._updated_event.clear()
            if self._registration.revision > rejected_revision:
                continue

            stop = asyncio.ensure_future(self._stop_event.wait())
            updated = asyncio.ensure_future(self._updated_event.wait())
            try:
                await asyncio.wait(
                    {stop, updated},
                    timeout=lease_remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for fut in (stop, updated):
                    if not fut.done():
                        fut.cancel()
                        with contextlib.suppress(Exception, asyncio.CancelledError):
                            await fut
            if self._stop_event.is_set():
                return False
        return False

    # ------------------------------------------------------------------
    # State / cleanup
    # ------------------------------------------------------------------

    def _lease_expired(self) -> bool:
        """Return whether the current registration lease has expired."""
        return self._monotonic() >= self._registration.lease_expires_at

    async def _close_epoch(self, *, cancel_call: bool = True) -> None:
        """Detach and close epoch resources without allowing cleanup to hang."""
        call, channel = self._call, self._channel
        self._call = None
        self._channel = None

        if call is not None:
            if cancel_call:
                with contextlib.suppress(Exception):
                    call.cancel()
            else:
                try:
                    await asyncio.wait_for(
                        _as_coro(call.done_writing()),
                        timeout=GRPC_CLOSE_TIMEOUT_SECONDS,
                    )
                except asyncio.CancelledError:
                    with contextlib.suppress(Exception):
                        call.cancel()
                    raise
                except Exception:
                    with contextlib.suppress(Exception):
                        call.cancel()

        if channel is not None:
            try:
                await asyncio.wait_for(
                    _as_coro(channel.close()),
                    timeout=GRPC_CLOSE_TIMEOUT_SECONDS,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                pass


async def _as_coro(awaitable: Awaitable):
    """Wrap any awaitable so it can be scheduled with ``ensure_future``."""
    return await awaitable


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


MonitorStoppedCallback = Callable[[MonitorKey, int], Awaitable[None]]
MonitorTaskFactory = Callable[
    [MonitorRegistration, int, MonitorStoppedCallback], MonitorTask
]


class _MonitorEntry:
    __slots__ = ("generation", "monitor", "task")

    def __init__(
        self,
        generation: int,
        monitor: MonitorTask,
        task: asyncio.Task[None],
    ) -> None:
        """Initialize one generation, monitor, and backing-task tuple.

        Args:
            generation: Ownership generation for stale-callback protection.
            monitor: Monitor state machine owned by the entry.
            task: Async task executing the monitor.

        Returns:
            None.
        """
        self.generation = generation
        self.monitor = monitor
        self.task = task


class MonitorManager:
    """Owns the live ``MonitorKey -> MonitorTask`` map and the min interval."""

    def __init__(
        self,
        factory: MonitorTaskFactory,
        schedule_changed: Callable[[], None],
        worker_metadata,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the monitor ownership map.

        Args:
            factory: Callback that constructs one generation-owned monitor.
            schedule_changed: Callback invoked when active intervals change.
            worker_metadata: Stable worker fields copied into registrations.
            monotonic: Injectable monotonic clock for lease decisions.

        Returns:
            None.
        """
        self._factory = factory
        self._schedule_changed = schedule_changed
        self._worker_metadata = worker_metadata
        self._monotonic = monotonic

        self._lock = asyncio.Lock()
        self._entries: Dict[MonitorKey, _MonitorEntry] = {}
        self._next_generation = 0
        self._min_report_interval_ms: Optional[int] = None

    @property
    def min_report_interval_ms(self) -> Optional[int]:
        """Return the minimum interval across live monitors, if any."""
        return self._min_report_interval_ms

    @property
    def monitor_count(self) -> int:
        """Number of live monitor entries currently in the map."""
        return len(self._entries)

    async def upsert(
        self, value: StartReportingRequest, worker_addr: str
    ) -> StartReportingResponse:
        """Create or renew one identity-safe Router monitor.

        Args:
            value: Validated target, interval, and lease request.
            worker_addr: Canonical identity of the reporting worker.

        Returns:
            Accepted lease and recommended renewal timing.

        Raises:
            WorkerIdentityConflict: If another live worker owns the target.
        """
        key = MonitorKey.from_request(value)
        async with self._lock:
            now = self._monotonic()
            current = self._entries.get(key)
            can_renew = current is not None and self._entry_can_renew(current, now)
            if can_renew and (
                current.monitor.registration.worker_identity.worker_addr != worker_addr
            ):
                raise WorkerIdentityConflict(key)

            if can_renew:
                registration = self._next_registration(
                    current,
                    value,
                    worker_addr,
                    now,
                )
                if not current.monitor.try_update_registration(registration):
                    registration = self._next_registration(
                        None,
                        value,
                        worker_addr,
                        now,
                    )
                    self._start_generation_locked(registration)
            else:
                registration = self._next_registration(
                    None,
                    value,
                    worker_addr,
                    now,
                )
                self._start_generation_locked(registration)
            self._recompute_min_interval_locked()

        self._schedule_changed()
        return StartReportingResponse(
            status="reporting",
            lease_ttl_ms=value.lease_ttl_ms,
            renew_after_ms=max(1, value.lease_ttl_ms // 3),
        )

    @staticmethod
    def _entry_can_renew(current: _MonitorEntry, now: float) -> bool:
        """Return whether an entry can accept an in-generation renewal."""
        return (
            not current.task.done()
            and current.monitor.accepting_updates
            and current.monitor.registration.lease_expires_at > now
        )

    def _start_generation_locked(self, registration: MonitorRegistration) -> None:
        """Create and publish a new monitor generation while holding the lock."""
        self._next_generation += 1
        generation = self._next_generation
        monitor = self._factory(
            registration,
            generation,
            self._remove_if_generation,
        )
        task = asyncio.create_task(
            monitor.run(),
            name=(f"load-monitor-{registration.key.authority}-g{generation}"),
        )
        self._entries[registration.key] = _MonitorEntry(
            generation,
            monitor,
            task,
        )

    def _next_registration(
        self,
        current: Optional[_MonitorEntry],
        value: StartReportingRequest,
        worker_addr: str,
        now: float,
    ) -> MonitorRegistration:
        """Build the next immutable registration revision.

        Args:
            current: Existing renewable entry, or ``None`` for a new generation.
            value: Validated registration request.
            worker_addr: Canonical worker identity.
            now: Monotonic registration time.

        Returns:
            The next registration revision.
        """
        identity = WorkerIdentity(
            worker_addr=worker_addr,
            worker_type=self._worker_metadata.worker_type,
            model=self._worker_metadata.model,
            zone=self._worker_metadata.zone,
        )
        revision = 1 if current is None else current.monitor.registration.revision + 1
        return MonitorRegistration(
            key=MonitorKey.from_request(value),
            worker_identity=identity,
            report_interval_ms=value.report_interval_ms,
            lease_expires_at=now + value.lease_ttl_ms / 1000.0,
            updated_at=now,
            revision=revision,
        )

    def _recompute_min_interval_locked(self) -> None:
        """Recompute the cached minimum interval while holding the map lock."""
        if not self._entries:
            self._min_report_interval_ms = None
            return
        self._min_report_interval_ms = min(
            entry.monitor.registration.report_interval_ms
            for entry in self._entries.values()
        )

    async def _remove_if_generation(self, key: MonitorKey, generation: int) -> None:
        """Remove the key only while the stopped generation still owns it."""
        async with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.generation == generation:
                del self._entries[key]
                self._recompute_min_interval_locked()
                changed = True
            else:
                changed = False
        if changed:
            self._schedule_changed()

    async def close(self) -> None:
        """Stop every monitor and await convergence."""
        async with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
            self._min_report_interval_ms = None
        for entry in entries:
            await entry.monitor.stop()
        if entries:
            await asyncio.gather(
                *(entry.task for entry in entries), return_exceptions=True
            )
            self._schedule_changed()

    async def cancel_remaining(self) -> None:
        """Hard-cancel any monitor tasks that failed to converge in time."""
        async with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
            self._min_report_interval_ms = None
        for entry in entries:
            entry.task.cancel()
        if entries:
            await asyncio.gather(
                *(entry.task for entry in entries), return_exceptions=True
            )
            self._schedule_changed()
