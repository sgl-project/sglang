"""Load reporter runtime."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from sglang.srt.load_reporter.config import (
    INITIAL_SAMPLE_TIMEOUT_SECONDS,
    SHUTDOWN_TIMEOUT_SECONDS,
    LoadReporterConfig,
    WorkerMetadata,
)
from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
from sglang.srt.load_reporter.report_builder import ReportBuilder, SequenceAllocator
from sglang.srt.load_reporter.sampler import LoadSampler
from sglang.srt.load_reporter.store import LatestSnapshotStore

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
    """One inbound Router stream with a latest-wins response queue."""

    def __init__(
        self,
        router_id: str,
        report_interval_ms: int,
        lease_ttl_ms: int,
        store: LatestSnapshotStore,
        builder: ReportBuilder,
        identity: WorkerMetadata,
        on_close: Callable[[str, _RouterSession], None],
        on_schedule_changed: Callable[[], None],
        sample_baseline: int,
        sample_generation: Callable[[], int],
        sample_event: Callable[[], asyncio.Event],
    ) -> None:
        _validate_timing(report_interval_ms, lease_ttl_ms)
        now = time.monotonic()
        self._router_id = router_id
        self._report_interval_ms = report_interval_ms
        self._lease_ttl_ms = lease_ttl_ms
        self._next_report_deadline = now + report_interval_ms / 1000.0
        self._lease_expires_at = now + lease_ttl_ms / 1000.0
        self._store = store
        self._builder = builder
        self._identity = identity
        self._on_close = on_close
        self._on_schedule_changed = on_schedule_changed
        # Require a sample completed after this session registered.
        self._sample_baseline = sample_baseline
        self._sample_generation = sample_generation
        self._sample_event = sample_event

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._done: asyncio.Event = asyncio.Event()
        self._config_changed: asyncio.Event = asyncio.Event()
        self._task: asyncio.Task = asyncio.create_task(
            self._run(), name=f"lr-session-{router_id}"
        )

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
            self._config_changed.set()
        if report_interval_ms is not None:
            self._on_schedule_changed()

    def stop(self) -> None:
        """Idempotent stop: signal the report loop to exit."""
        self._done.set()
        self._config_changed.set()

    def cancel(self) -> None:
        """Hard-cancel the report task without triggering on_close."""
        self._on_close = lambda _rid, _session: None  # defuse callback before cancel
        self._task.cancel()

    async def wait_stopped(self) -> None:
        """Await the report loop task (used during shutdown)."""
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

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

    def _build_report(self) -> pb.LoadReport:
        """Build a report from the current snapshot store view."""
        view = self._store.view()
        return self._builder.build(
            view,
            self._identity,
            report_time_unix_ms=int(time.time() * 1000),
        )

    def _fresh_sample_ready(self) -> bool:
        """True once a sampling attempt has completed after registration."""
        return self._sample_generation() > self._sample_baseline

    async def _wait_for_initial_sample(self) -> bool:
        """Wait for a post-registration sampling attempt, bounded by timeout/lease."""
        deadline = time.monotonic() + INITIAL_SAMPLE_TIMEOUT_SECONDS
        while True:
            if self._done.is_set():
                return False

            now = time.monotonic()
            if now >= self._lease_expires_at:
                logger.info("Lease expired for router_id=%s", self._router_id)
                return False
            if self._fresh_sample_ready() or now >= deadline:
                return True

            self._config_changed.clear()
            if self._done.is_set() or self._fresh_sample_ready():
                continue

            wait_timeout = max(
                0.0, min(deadline, self._lease_expires_at) - time.monotonic()
            )
            sample_wait = asyncio.create_task(self._sample_event().wait())
            config_wait = asyncio.create_task(self._config_changed.wait())
            try:
                await asyncio.wait(
                    (sample_wait, config_wait),
                    timeout=wait_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                sample_wait.cancel()
                config_wait.cancel()
                await asyncio.gather(sample_wait, config_wait, return_exceptions=True)

    async def _run(self) -> None:
        """Background report loop: sampled first report, then periodic."""
        try:
            if not await self._wait_for_initial_sample():
                return
            self._enqueue(self._build_report())
            self._next_report_deadline = (
                time.monotonic() + self._report_interval_ms / 1000.0
            )

            while not self._done.is_set():
                self._config_changed.clear()
                now = time.monotonic()
                sleep_sec = max(
                    0.0,
                    min(self._next_report_deadline, self._lease_expires_at) - now,
                )

                try:
                    await asyncio.wait_for(
                        self._config_changed.wait(), timeout=sleep_sec
                    )
                except asyncio.TimeoutError:
                    pass

                if self._done.is_set():
                    break
                if self._config_changed.is_set():
                    continue

                now = time.monotonic()
                if now >= self._lease_expires_at:
                    logger.info("Lease expired for router_id=%s", self._router_id)
                    break

                if now < self._next_report_deadline:
                    continue

                self._enqueue(self._build_report())
                interval_sec = self._report_interval_ms / 1000.0
                self._next_report_deadline += interval_sec
                if self._next_report_deadline <= now:
                    self._next_report_deadline = now + interval_sec
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception(
                "Session report loop error for router_id=%s", self._router_id
            )
        finally:
            self._enqueue(None)  # sentinel: write loop exits
            self._done.set()
            try:
                self._on_close(self._router_id, self)
            except Exception:
                logger.exception(
                    "on_close callback failed for router_id=%s", self._router_id
                )


class LoadReporterRuntime:
    """Own reporter sessions and snapshot sampling."""

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

        self._store = LatestSnapshotStore()
        # Sessions wait for a sample completed after registration.
        self._sample_generation = 0
        self._sample_completed = asyncio.Event()
        self._builder = ReportBuilder(
            str(uuid.uuid4()),
            self._config.snapshot_stale_after_ms,
            SequenceAllocator(),
        )
        self._sessions: Dict[str, _RouterSession] = {}
        self._sampler = LoadSampler(
            snapshot_source,
            self._store,
            interval_provider=self._min_report_interval_ms,
            on_sample_completed=self._on_sample_completed,
        )

    def _on_sample_completed(self) -> None:
        """Advance the sampling generation and wake waiters once per sample."""
        self._sample_generation += 1
        completed, self._sample_completed = self._sample_completed, asyncio.Event()
        completed.set()

    def _current_sample_generation(self) -> int:
        """Return the latest completed sampling generation."""
        return self._sample_generation

    def _sample_event(self) -> asyncio.Event:
        """Return the event that fires when the next sampling attempt completes."""
        return self._sample_completed

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

        # Capture the baseline before activation to avoid reusing stale samples.
        sample_baseline = self._sample_generation

        session = _RouterSession(
            router_id=router_id,
            report_interval_ms=report_interval_ms,
            lease_ttl_ms=lease_ttl_ms,
            store=self._store,
            builder=self._builder,
            identity=self._worker_metadata,
            on_close=self._on_session_closed,
            on_schedule_changed=self._on_schedule_changed,
            sample_baseline=sample_baseline,
            sample_generation=self._current_sample_generation,
            sample_event=self._sample_event,
        )
        self._sessions[router_id] = session
        self._on_schedule_changed()

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
            self._on_schedule_changed()

    def _min_report_interval_ms(self) -> Optional[int]:
        """Minimum interval across active sessions; None when no sessions."""
        if not self._sessions:
            return None
        return min(s.report_interval_ms for s in self._sessions.values())

    def _on_schedule_changed(self) -> None:
        """Activate or deactivate the sampler based on session count."""
        active = bool(self._sessions)
        if active:
            self._sampler.activate()
            self._sampler.notify_schedule_changed()
        else:
            self._sampler.deactivate()

    def notify_refresh(self) -> None:
        """Synchronous, non-throwing refresh signal."""
        try:
            if not self._closing:
                self._sampler.notify_refresh()
        except Exception:
            logger.exception("Load reporter notify_refresh failed")

    def notify_source_changed(self) -> None:
        """Signal that the snapshot source may have new data."""
        self.notify_refresh()

    def update_expected_dp_ranks(self, expected_dp_ranks: Iterable[int]) -> bool:
        """Update a rank-aware snapshot source after elastic scaling."""
        update = getattr(self._snapshot_source, "update_expected_dp_ranks", None)
        if update is None or not update(expected_dp_ranks):
            return False
        self.notify_source_changed()
        return True

    async def close(self) -> None:
        """Bounded, idempotent shutdown."""
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(
                self._close_impl(), name="load-reporter-runtime-close"
            )
        await asyncio.shield(self._close_task)

    async def _close_impl(self) -> None:
        """Run one shared close attempt to completion for every caller."""
        sessions = list(self._sessions.values())
        for session in sessions:
            session.stop()

        force_close = False

        try:
            await asyncio.wait_for(
                asyncio.gather(
                    self._sampler.close(),
                    *(session.wait_stopped() for session in sessions),
                ),
                SHUTDOWN_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Load reporter shutdown exceeded %.1fs; cancelling remaining tasks",
                SHUTDOWN_TIMEOUT_SECONDS,
            )
            force_close = True
        except asyncio.CancelledError:
            logger.warning(
                "Load reporter graceful shutdown was cancelled; "
                "cancelling remaining tasks"
            )
            force_close = True
        except Exception:
            logger.exception("Load reporter shutdown failed")
            force_close = True

        if force_close:
            self._sampler.cancel()
            for session in sessions:
                session.cancel()
            await asyncio.gather(
                self._sampler.wait_stopped(),
                *(session._task for session in sessions),
                return_exceptions=True,
            )

        self._sessions.clear()
