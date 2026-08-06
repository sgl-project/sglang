"""Single-flight async load sampler for the embedded load reporter.

Owns the background task that calls snapshot_source.get_loads() and forwards
results into LatestSnapshotStore. Only one in-flight get_loads call is active
at a time.

Coalescing: idle + trigger → start refresh; inflight + trigger → set pending;
refresh done + pending → one more refresh; refresh done, no pending → idle.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Collection, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source protocol and adapters
# ---------------------------------------------------------------------------


@runtime_checkable
class LoadSnapshotSource(Protocol):
    """Minimal protocol for a load-snapshot data source.

    ``LoadSampler`` depends only on this protocol, not on any concrete
    manager type. Two adapters are provided: one that wraps a live manager
    (single-owner path) and one that wraps the router's shared-memory reader
    (multi-tokenizer path).
    """

    async def get_loads(self) -> list:
        """Return the source's latest scheduler load snapshots."""
        raise NotImplementedError

    def expected_dp_ranks(self) -> frozenset:
        """Return the authoritative DP ranks required for a full snapshot."""
        raise NotImplementedError


class ManagerLoadSnapshotSource:
    """Adapts a manager to the ``LoadSnapshotSource`` protocol.

    The expected DP rank set follows ``manager.elastic_worker_count`` when the
    manager exposes it, and otherwise falls back to the construction-time set.
    Used for both TokenizerManager and GrpcRequestManager.
    """

    def __init__(
        self,
        manager: Any,
        expected_dp_ranks: Collection[int],
        *,
        snapshot_reader: Optional[Any] = None,
    ) -> None:
        """Wrap a manager with an authoritative rank fallback.

        Args:
            manager: Manager exposing get_loads(include=["core"]).
            expected_dp_ranks: DP ranks required for a full snapshot.
            snapshot_reader: Optional synchronous reader used when the reporter
                runs on an event loop different from the manager's loop.
        """
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
    """Adapts a load-snapshot reader to the ``LoadSnapshotSource`` protocol.

    The authoritative DP rank set is maintained separately from the reader
    so the router can update it (e.g. after a scale event) without
    replacing the reader object.
    """

    def __init__(self, reader: Any, expected_dp_ranks: Collection[int]) -> None:
        """Wrap a shared-memory reader with an authoritative rank set.

        Args:
            reader: Reader exposing a synchronous read_all method.
            expected_dp_ranks: DP ranks required for a full router snapshot.
        """
        self._reader = reader
        self._expected: frozenset[int] = frozenset(expected_dp_ranks)

    async def get_loads(self) -> list:
        """Read all load snapshots currently published in shared memory."""
        # read_all() is a fast synchronous SHM read; safe to call on the event loop.
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


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class LoadSampler:
    """Background single-flight sampler.

    Parameters
    ----------
    snapshot_source:
        Object satisfying the ``LoadSnapshotSource`` protocol: exposes
        ``async get_loads() -> list[LoadSnapshot]`` and
        ``expected_dp_ranks() -> frozenset[int]``.
    store:
        Object with synchronous ``.apply_full_snapshot(...)`` and
        ``.record_error(exc)`` methods (``LatestSnapshotStore``).
    interval_provider:
        Synchronous callable returning the current minimum report interval
        in milliseconds across all active monitors, or ``None`` when no
        monitor is active.
    """

    def __init__(
        self,
        snapshot_source: Any,
        store: Any,
        interval_provider: Callable[[], Optional[int]],
        on_sample_completed: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initialize the coalescing sampler.

        Args:
            snapshot_source: Source implementing the LoadSnapshotSource protocol.
            store: Destination receiving validated full snapshots and errors.
            interval_provider: Callback returning the active sampling interval.
            on_sample_completed: Optional callback after each sampling attempt.
        """
        self._snapshot_source = snapshot_source
        self._store = store
        self._interval_provider = interval_provider
        self._on_sample_completed = on_sample_completed

        self._wake: asyncio.Event = asyncio.Event()
        self._active: bool = False
        self._closing: bool = False
        self._task: Optional[asyncio.Task[None]] = None

    # ------------------------------------------------------------------
    # Public synchronous API (must not raise)
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Activate the sampler and start the background task if needed.

        Idempotent once active.  No-op if ``close()`` has already been
        called.
        """
        if self._closing:
            return
        self._active = True
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="load-reporter-sampler")
        self._wake.set()

    def deactivate(self) -> None:
        """Deactivate sampling while keeping the background task reusable.

        The current in-flight sample, if any, is allowed to finish. Subsequent
        timer and request notifications remain dormant until ``activate()`` is
        called again.

        Returns:
            None.
        """
        if self._closing:
            return
        self._active = False
        self._wake.set()

    def notify_refresh(self) -> None:
        """Signal that a fresh sample is desired (e.g. request-end hook).

        No-op before activation or after close.  Never raises.
        """
        if self._active and not self._closing:
            self._wake.set()

    def notify_schedule_changed(self) -> None:
        """Signal that the timer interval may have changed.

        Wakes the loop so it recomputes its next deadline from
        ``interval_provider()``.  Never raises.
        """
        if self._active and not self._closing:
            self._wake.set()

    # ------------------------------------------------------------------
    # Async lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Shut down the background task gracefully.

        Sets the closing flag, wakes the loop, and awaits the task.
        Idempotent: safe to call more than once. The sampler task is shielded
        so cancellation of a bounded graceful close does not corrupt its
        state; the owner can then cancel and join it explicitly.
        """
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        """Background loop — exactly one task ever calls ``_refresh_once``.

        State machine:
        1. On activation the wake event is already set; fall straight into
           the first refresh.
        2. Before each refresh, clear the wake event so any notification
           that arrives *during* the refresh will re-set it and cause
           exactly one follow-up refresh (coalescing).
        3. After the refresh, check whether the event was re-set.
           - If yes: do one more refresh (the coalesced follow-up), then
             go idle.
           - If no: wait for either a wake signal or the periodic timer.
        4. Timer fires -> refresh, then schedule the next deadline from
           *now* (missed deadlines are not caught up).
        5. Loop exits when ``_closing`` is set.
        """
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
