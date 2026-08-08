"""Single-flight load sampler with coalesced refreshes."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Collection, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class LoadSnapshotSource(Protocol):
    """Protocol for a load-snapshot data source."""

    async def get_loads(self) -> list:
        """Return the source's latest scheduler load snapshots."""
        raise NotImplementedError

    def expected_dp_ranks(self) -> frozenset:
        """Return the authoritative DP ranks required for a full snapshot."""
        raise NotImplementedError


class ManagerLoadSnapshotSource:
    """Adapt a manager to the load-snapshot protocol."""

    def __init__(
        self,
        manager: Any,
        expected_dp_ranks: Collection[int],
        *,
        snapshot_reader: Optional[Any] = None,
    ) -> None:
        """Wrap a manager with an authoritative rank fallback."""
        self._manager = manager
        self._snapshot_reader = snapshot_reader
        self._expected: frozenset[int] = frozenset(expected_dp_ranks)

    async def get_loads(self) -> list:
        """Fetch core load snapshots from the wrapped manager."""
        if self._snapshot_reader is not None:
            return self._snapshot_reader.read_all()
        return await self._manager.get_loads(include=["core"])

    def expected_dp_ranks(self) -> frozenset[int]:
        """Return the manager's current authoritative DP rank set."""
        worker_count = getattr(self._manager, "elastic_worker_count", None)
        if (
            isinstance(worker_count, int)
            and not isinstance(worker_count, bool)
            and worker_count > 0
        ):
            return frozenset(range(worker_count))
        return self._expected


class RouterLoadSnapshotSource:
    """Adapt a load-snapshot reader to the load-snapshot protocol."""

    def __init__(self, reader: Any, expected_dp_ranks: Collection[int]) -> None:
        """Wrap a shared-memory reader with an authoritative rank set."""
        self._reader = reader
        self._expected: frozenset[int] = frozenset(expected_dp_ranks)

    async def get_loads(self) -> list:
        """Read all load snapshots currently published in shared memory."""
        return self._reader.read_all()

    def expected_dp_ranks(self) -> frozenset[int]:
        """Return the current authoritative DP rank set."""
        return self._expected

    def update_expected_dp_ranks(self, ranks: Collection[int]) -> bool:
        """Update the authoritative rank set.  Returns True if it changed."""
        updated = frozenset(ranks)
        if updated == self._expected:
            return False
        self._expected = updated
        return True


class LoadSampler:
    """Background sampler that coalesces refresh requests."""

    def __init__(
        self,
        snapshot_source: Any,
        store: Any,
        interval_provider: Callable[[], Optional[int]],
        on_sample_completed: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initialize the coalescing sampler."""
        self._snapshot_source = snapshot_source
        self._store = store
        self._interval_provider = interval_provider
        self._on_sample_completed = on_sample_completed

        self._wake: asyncio.Event = asyncio.Event()
        self._active: bool = False
        self._closing: bool = False
        self._task: Optional[asyncio.Task[None]] = None

    def activate(self) -> None:
        """Activate the sampler and start its task if needed."""
        if self._closing:
            return
        self._active = True
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="load-reporter-sampler")
        self._wake.set()

    def deactivate(self) -> None:
        """Deactivate sampling while leaving the background task reusable."""
        if self._closing:
            return
        self._active = False
        self._wake.set()

    def notify_refresh(self) -> None:
        """Request a fresh sample."""
        if self._active and not self._closing:
            self._wake.set()

    def notify_schedule_changed(self) -> None:
        """Wake the loop to recompute its timer interval."""
        if self._active and not self._closing:
            self._wake.set()

    async def close(self) -> None:
        """Shut down the background task gracefully."""
        self._active = False
        self._closing = True
        self._wake.set()
        task = self._task
        if task is None:
            return

        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Load reporter sampler task raised: %s", exc)
        finally:
            if task.done() and self._task is task:
                self._task = None

    def cancel(self) -> None:
        """Request immediate cancellation of the sampler task."""
        self._active = False
        self._closing = True
        self._wake.set()
        if self._task is not None:
            self._task.cancel()

    async def wait_stopped(self) -> None:
        """Join a cancelled sampler task without propagating its result."""
        task = self._task
        if task is None:
            return

        try:
            result = (await asyncio.gather(task, return_exceptions=True))[0]
            if isinstance(result, Exception):
                logger.warning("Load reporter sampler task raised: %s", result)
        finally:
            if task.done() and self._task is task:
                self._task = None

    async def _refresh_once(self) -> None:
        """Execute one full sample cycle and write the result into the store."""
        try:
            loads = await self._snapshot_source.get_loads()
            completed_unix_ms = time.time_ns() // 1_000_000
            completed_monotonic = time.monotonic()
            self._store.apply_full_snapshot(
                loads,
                expected_dp_ranks=self._snapshot_source.expected_dp_ranks(),
                collected_at_unix_ms=completed_unix_ms,
                collected_at_monotonic=completed_monotonic,
            )
        except Exception as exc:
            self._store.record_error(exc)
            logger.warning("Load reporter sampling failed: %s", exc)
        finally:
            if self._on_sample_completed is not None:
                try:
                    self._on_sample_completed()
                except Exception:
                    logger.exception("Load reporter sample callback failed")

    async def _run(self) -> None:
        """Run the single-flight refresh loop."""
        while not self._closing:
            if not self._active:
                # Clear a stale deactivation wake before waiting. Re-check the
                # state to avoid losing an activation racing with ``clear()``.
                self._wake.clear()
                if not self._active and not self._closing:
                    await self._wake.wait()
                continue

            # ---- wait for a trigger or timer ----
            interval_ms = self._interval_provider()
            if interval_ms is not None and interval_ms > 0:
                interval_sec: Optional[float] = interval_ms / 1000.0
            else:
                interval_sec = None  # wait indefinitely on wake only

            if not self._wake.is_set():
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=interval_sec)
                    # wake fired (not a timeout)
                except asyncio.TimeoutError:
                    # Timer expired — proceed to refresh
                    pass

            if self._closing:
                break
            if not self._active:
                continue

            # ---- single refresh (coalescing loop) ----
            # Clear BEFORE the refresh so notifications during it re-set.
            self._wake.clear()
            await self._refresh_once()

            if self._closing:
                break
            if not self._active:
                continue

            # If a notification arrived during the refresh the event will
            # be set again.  Drain it with exactly one follow-up refresh.
            if self._wake.is_set():
                self._wake.clear()
                await self._refresh_once()
