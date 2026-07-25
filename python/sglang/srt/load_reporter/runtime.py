"""Top-level assembly of the embedded load reporter.

``LoadReporterRuntime`` is the composition root: it constructs the store,
builder, sampler, and monitor manager, wires them into one asyncio event loop,
and exposes the seams the HTTP layer uses -- ``start_reporting`` (control
plane), ``notify_refresh`` / ``notify_request_finished`` /
``notify_source_changed`` (data-plane refresh), and ``close``
(bounded shutdown). Nothing here computes load metrics; it only orders the
collaborators.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Callable, Iterable, Optional

from sglang.srt.load_reporter.config import (
    SHUTDOWN_TIMEOUT_SECONDS,
    LoadReporterConfig,
    WorkerMetadata,
)
from sglang.srt.load_reporter.monitor import MonitorManager, MonitorTask
from sglang.srt.load_reporter.registration import (
    RuntimeClosingError,
    StartReportingRequest,
    StartReportingResponse,
)
from sglang.srt.load_reporter.report_builder import ReportBuilder, SequenceAllocator
from sglang.srt.load_reporter.sampler import LoadSampler
from sglang.srt.load_reporter.store import LatestSnapshotStore

logger = logging.getLogger(__name__)


class LoadReporterRuntime:
    """Owns the reporter collaborators for a single-tokenizer HTTP process."""

    def __init__(
        self,
        snapshot_source: Any,
        server_args: Any,
        *,
        active_changed: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Assemble reporter collaborators around one snapshot source.

        Args:
            snapshot_source: Adapter providing load snapshots and expected ranks.
            server_args: SGLang server configuration.
            active_changed: Optional callback for zero-to-one monitor transitions.
        """
        self._closing = False
        self._config = LoadReporterConfig.from_server_args(server_args)
        self._worker_metadata = WorkerMetadata.from_server_args(server_args)
        self._active_changed: Callable[[bool], None] = (
            active_changed if active_changed is not None else lambda _active: None
        )
        self._last_active = False
        self._snapshot_source = snapshot_source

        self._store = LatestSnapshotStore()
        self._builder = ReportBuilder(
            str(uuid.uuid4()),
            self._config.snapshot_stale_after_ms,
            SequenceAllocator(),
        )
        self._sampler = LoadSampler(
            snapshot_source,
            self._store,
            interval_provider=lambda: self._manager.min_report_interval_ms,
        )
        self._manager = MonitorManager(
            factory=self._new_monitor,
            schedule_changed=self._on_schedule_changed,
            worker_metadata=self._worker_metadata,
        )

    # ------------------------------------------------------------------
    # Collaborator wiring
    # ------------------------------------------------------------------

    def _on_schedule_changed(self) -> None:
        """Synchronize sampler activation with the live monitor schedule.

        Returns:
            None.
        """
        active = self._manager.monitor_count > 0
        if active and not self._last_active:
            self._sampler.activate()
        elif active:
            self._sampler.notify_schedule_changed()
        elif self._last_active:
            self._sampler.deactivate()

        if active != self._last_active:
            self._last_active = active
            self._active_changed(active)

    def _new_monitor(self, registration, generation, on_stopped) -> MonitorTask:
        """Construct one generation-owned monitor task.

        Args:
            registration: Immutable target registration.
            generation: Manager-assigned ownership generation.
            on_stopped: Callback invoked when the monitor exits.

        Returns:
            A configured MonitorTask.
        """
        return MonitorTask(
            registration,
            self._store,
            self._builder,
            on_stopped,
            generation=generation,
        )

    # ------------------------------------------------------------------
    # Control plane / request-end seams
    # ------------------------------------------------------------------

    async def start_reporting(
        self, payload: StartReportingRequest, worker_addr: str
    ) -> StartReportingResponse:
        """Register or renew one Router target and activate sampling.

        Args:
            payload: Validated reporting interval, lease, and Router target.
            worker_addr: Canonical address identifying this worker.

        Returns:
            The accepted lease and renewal timing.

        Raises:
            RuntimeClosingError: If shutdown has already started.
            WorkerIdentityConflict: If another worker owns the live target.
        """
        if self._closing:
            raise RuntimeClosingError("load reporter is shutting down")
        return await self._manager.upsert(payload, worker_addr)

    def notify_refresh(self) -> None:
        """Synchronous, non-throwing refresh signal."""
        try:
            if not self._closing:
                self._sampler.notify_refresh()
        except Exception:
            logger.exception("Load reporter notify_refresh failed")

    def notify_request_finished(self) -> None:
        """Synchronous, non-throwing request-end refresh signal."""
        try:
            self.notify_refresh()
        except Exception:
            logger.exception("Load reporter request-finished notification failed")

    def notify_source_changed(self) -> None:
        """Signal that the snapshot source may have new data."""
        self.notify_refresh()

    def update_expected_dp_ranks(self, expected_dp_ranks: Iterable[int]) -> bool:
        """Update a rank-aware snapshot source after elastic scaling.

        Args:
            expected_dp_ranks: DP ranks expected in the next aggregate snapshot.

        Returns:
            ``True`` when the source accepted a changed rank set; otherwise
            ``False`` for unchanged or non-rank-aware sources.
        """
        update = getattr(self._snapshot_source, "update_expected_dp_ranks", None)
        if update is None or not update(expected_dp_ranks):
            return False
        self.notify_source_changed()
        return True

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Bounded, idempotent shutdown. Never constructs a final report."""
        if self._closing:
            return
        self._closing = True

        async def close_in_order() -> None:
            """Stop sampling before closing all monitor streams."""
            await self._sampler.close()
            await self._manager.close()

        try:
            await asyncio.wait_for(close_in_order(), SHUTDOWN_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning(
                "Load reporter shutdown exceeded %.1fs; cancelling remaining tasks",
                SHUTDOWN_TIMEOUT_SECONDS,
            )
            await self._sampler.close()
            await self._manager.cancel_remaining()
        except Exception:
            logger.exception("Load reporter shutdown failed")
