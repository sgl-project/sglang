"""Deadline-bounded execution of HiCache storage-backend calls.

The HiCache controller talks to the L3 backend from single worker threads, and a
backend call that never returns wedges that thread: the peer ranks block in the
MIN collectives behind it, every later prefetch or backup queues behind it, and a
``reset()`` can no longer join the thread. The guard runs each backend call on a
per-kind worker and waits with a deadline. A call that misses it is published as
the controller's ordinary miss; the worker that still runs it is orphaned and a
fresh worker serves the next call. The host slots an orphaned call may still
read or write stay quarantined in their pool until the call returns, so a late
completion can never land in memory that was handed to another operation.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import threading
import time
from enum import Enum
from queue import Queue
from typing import Any, Callable, Optional, Sequence, TypeVar

import msgspec
import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Granularity at which a waiting controller thread re-checks the stop event.
_STOP_POLL_INTERVAL_S = 0.5
# How long a stop request (reset / detach) lets the call already in flight
# finish before orphaning it; below the controller's 10 s thread join so the
# thread stays joinable, long enough that a detach rarely closes a backend
# under a live call. Calls issued after the stop are not dispatched at all.
_STOP_GRACE_S = 5.0
# Minimum spacing between WARNING lines per call kind.
_WARN_INTERVAL_S = 5.0


class StorageCallKind(Enum):
    # Existence queries issued by the prefetch worker.
    QUERY = "query"
    # Existence probes issued by the scheduler thread (the PD-decode hit probe,
    # PP0's ticket submission), which may overlap a QUERY; same deadline key.
    PROBE = "probe"
    GET = "get"
    SET = "set"


_EXTRA_CONFIG_KEYS = {
    StorageCallKind.QUERY: "storage_query_timeout_s",
    StorageCallKind.GET: "storage_get_timeout_s",
    StorageCallKind.SET: "storage_set_timeout_s",
}


class StorageCallTimeouts(msgspec.Struct, frozen=True):
    """Deadline in seconds for one backend call of each kind; None disables it.

    Each call covers one STORAGE_BATCH_SIZE-page batch. The defaults sit one to
    two orders of magnitude above the healthy tails measured on remote backends
    (ms-scale queries, up to about a second per batch) while keeping the wait of
    anything queued behind a hung call short.
    """

    query_s: Optional[float] = 5.0
    get_s: Optional[float] = 20.0
    set_s: Optional[float] = 60.0
    # Orphaned (timed-out, still running) calls tolerated per kind before new
    # calls of that kind are refused as misses. Arbitrary. It also bounds the
    # host memory a hanging backend can pin: max_orphans batches of
    # STORAGE_BATCH_SIZE pages per kind and rank.
    max_orphans: int = 4

    @classmethod
    def disabled(cls) -> StorageCallTimeouts:
        return cls(query_s=None, get_s=None, set_s=None)

    @classmethod
    def pop_from_extra_config(
        cls, extra_config: Optional[dict[str, Any]]
    ) -> StorageCallTimeouts:
        """Take the ``storage_*_timeout_s`` keys (0 or null disables) out of
        ``--hicache-storage-backend-extra-config`` so backends never see them."""
        if not extra_config:
            return cls()
        values: dict[str, Optional[float]] = {}
        for kind, key in _EXTRA_CONFIG_KEYS.items():
            if key not in extra_config:
                continue
            raw = extra_config.pop(key)
            if raw is not None and (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(raw)
                or raw < 0
            ):
                raise ValueError(
                    f"{key} must be a finite number of seconds >= 0, got {raw!r}"
                )
            values[f"{kind.value}_s"] = None if not raw else float(raw)
        return cls(**values)

    def for_kind(self, kind: StorageCallKind) -> Optional[float]:
        if kind in (StorageCallKind.QUERY, StorageCallKind.PROBE):
            return self.query_s
        if kind is StorageCallKind.GET:
            return self.get_s
        return self.set_s


class HostSpan(msgspec.Struct, frozen=True):
    """Host slots a backend call reads or writes; ``pool`` owns their quarantine."""

    pool: Any
    indices: torch.Tensor


class StorageCallGuardStats(msgspec.Struct, frozen=True):
    """Per-kind counts: timeouts and errors since the previous snapshot,
    orphaned calls still running now."""

    timeouts: dict[str, int]
    errors: dict[str, int]
    orphaned: dict[str, int]


class _GuardTimeout(Exception):
    """The guard's own deadline expired (never a backend-raised TimeoutError)."""


class _Worker:
    def __init__(self, kind: StorageCallKind):
        self.inbox: Queue = Queue()
        self.orphaned = False
        self.thread = threading.Thread(
            target=self._loop, name=f"hicache-storage-{kind.value}", daemon=True
        )
        self.thread.start()

    def _loop(self) -> None:
        while True:
            job = self.inbox.get()
            if job is None:
                return
            fn, future = job
            try:
                result = fn()
            except BaseException as error:  # noqa: BLE001 - delivered to the caller
                future.set_exception(error)
            else:
                future.set_result(result)
            if self.orphaned:
                return


class _Orphan:
    def __init__(
        self,
        kind: StorageCallKind,
        spans: Sequence[HostSpan],
        op: str,
        started_at: float,
        thread_ident: Optional[int],
    ):
        self.kind = kind
        self.spans = spans
        self.op = op
        self.started_at = started_at
        self.thread_ident = thread_ident


class _Lane:
    def __init__(self):
        self.worker: Optional[_Worker] = None
        # True while a call is dispatched and not yet returned or orphaned.
        self.busy = False
        self.next_orphan_id = 0
        self.orphans: dict[int, _Orphan] = {}
        self.refusing = False
        self.timeouts = 0
        self.errors = 0
        self.last_warn_at = -_WARN_INTERVAL_S
        self.suppressed_warnings = 0


class StorageCallGuard:
    """Runs backend calls under per-kind deadlines; see the module docstring.

    Each kind has exactly one calling thread (the controller's query, transfer
    and backup workers for QUERY, GET and SET; the scheduler thread for PROBE),
    so calls of one kind run strictly one at a time unless the one in flight
    was orphaned. A second caller on a kind is a programming error and raises.
    """

    def __init__(
        self,
        timeouts: StorageCallTimeouts,
        *,
        label: str = "",
        stop_event: Optional[threading.Event] = None,
    ):
        self.timeouts = timeouts
        self._label = label
        self._stop_event = stop_event
        self._lock = threading.Lock()
        self._lanes = {kind: _Lane() for kind in StorageCallKind}

    def run(
        self,
        kind: StorageCallKind,
        fn: Callable[[], T],
        *,
        miss: Callable[[], T],
        op: str,
        num_keys: int,
        spans: Optional[Callable[[], Sequence[HostSpan]]] = None,
        lease: Optional[Callable[[], Any]] = None,
    ) -> T:
        """Call ``fn`` with the deadline of ``kind``.

        On timeout or exception the result is ``miss()``, the controller's own
        shape for "nothing came back"; ``spans()`` names the host slots a timed
        out call keeps touching so they can be quarantined until it returns.
        ``lease()`` is a context manager the worker holds around the call so an
        orphan keeps pinning the host layout its indices were resolved against.
        """
        timeout = self.timeouts.for_kind(kind)
        if timeout is None:
            return fn()
        if self._stop_event is not None and self._stop_event.is_set():
            # Shutting down: the result is discarded anyway, and dispatching
            # would hand every queued call another grace period.
            return miss()
        if lease is not None:
            unleased_fn = fn

            def fn():
                with lease():
                    return unleased_fn()

        lane = self._lanes[kind]
        started_at = time.monotonic()
        with self._lock:
            if lane.busy:
                raise RuntimeError(
                    f"HiCache storage {kind.value} calls must come from one thread; "
                    f"{op} was issued while another call is in flight"
                )
            pending_orphans = len(lane.orphans)
            refused = (
                lane.worker is None and pending_orphans >= self.timeouts.max_orphans
            )
            if refused:
                lane.errors += 1
                first_refusal = not lane.refusing
                lane.refusing = True
            else:
                if lane.worker is None:
                    lane.worker = _Worker(kind)
                worker = lane.worker
                future: concurrent.futures.Future = concurrent.futures.Future()
                lane.busy = True
                worker.inbox.put((fn, future))
        if refused:
            if first_refusal:
                logger.error(
                    "%sHiCache storage %s calls are refused until one of the %d "
                    "hung %s calls returns: the storage backend is unhealthy and "
                    "every %s call is now a miss",
                    self._log_prefix(),
                    kind.value,
                    pending_orphans,
                    kind.value,
                    kind.value,
                )
            return miss()
        try:
            result = self._wait(future, timeout)
        except _GuardTimeout:
            self._orphan(kind, lane, worker, future, spans, op, started_at)
            self._warn(
                lane,
                "HiCache storage %s call %s did not return within %.1fs (%d keys); "
                "published as a miss, the call keeps running on an orphaned "
                "worker (%d orphaned %s calls pending)",
                kind.value,
                op,
                time.monotonic() - started_at,
                num_keys,
                len(lane.orphans),
                kind.value,
            )
            return miss()
        except Exception as error:  # noqa: BLE001 - every backend failure is a miss
            with self._lock:
                lane.busy = False
                lane.errors += 1
            self._warn(
                lane,
                "HiCache storage %s call %s failed (%d keys): %r; published as a miss",
                kind.value,
                op,
                num_keys,
                error,
                exc_info=True,
            )
            return miss()
        with self._lock:
            lane.busy = False
        return result

    def _wait(self, future: concurrent.futures.Future, timeout: float):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # A call that completed between the last poll and this check is
                # still orphaned and reported as a miss (its pins are released
                # by the done callback at once): a late result is never trusted.
                raise _GuardTimeout()
            try:
                return future.result(timeout=min(remaining, _STOP_POLL_INTERVAL_S))
            except concurrent.futures.TimeoutError:
                if future.done():
                    # The backend raised a TimeoutError of its own: a failure,
                    # not our deadline (the two are the same class on 3.11+).
                    return future.result()
                if self._stop_event is not None and self._stop_event.is_set():
                    deadline = min(deadline, time.monotonic() + _STOP_GRACE_S)

    def _orphan(
        self,
        kind: StorageCallKind,
        lane: _Lane,
        worker: _Worker,
        future: concurrent.futures.Future,
        spans: Optional[Callable[[], Sequence[HostSpan]]],
        op: str,
        started_at: float,
    ) -> None:
        # Retire the worker and free the lane before pinning: a failure while
        # pinning must not leave the lane busy, which would turn every later
        # call of this kind into the one-caller error.
        orphan = _Orphan(kind, (), op, started_at, worker.thread.ident)
        with self._lock:
            lane.busy = False
            lane.timeouts += 1
            worker.orphaned = True
            lane.worker = None
            # Wakes the worker if the call already finished and it is idle.
            worker.inbox.put(None)
            orphan_id = lane.next_orphan_id
            lane.next_orphan_id += 1
            lane.orphans[orphan_id] = orphan
        held: list[HostSpan] = []
        pool_name = None
        try:
            for span in spans() if spans is not None else ():
                pool_name = type(span.pool).__name__
                span.pool.quarantine(span.indices)
                held.append(span)
        except Exception:  # noqa: BLE001 - an unprotected orphan is loud, not fatal
            logger.exception(
                "%sHiCache storage %s call %s timed out but its host span could "
                "not be quarantined (%s, %d spans pinned): the orphaned call may "
                "still write slots that are handed out again",
                self._log_prefix(),
                kind.value,
                op,
                pool_name,
                len(held),
            )
        with self._lock:
            orphan.spans = tuple(held)
        future.add_done_callback(
            lambda done: self._orphan_returned(lane, orphan_id, done)
        )

    def _orphan_returned(
        self, lane: _Lane, orphan_id: int, future: concurrent.futures.Future
    ) -> None:
        with self._lock:
            orphan = lane.orphans[orphan_id]
            released: list[HostSpan] = []
            try:
                for span in orphan.spans:
                    span.pool.release_quarantine(span.indices)
                    released.append(span)
            finally:
                if len(released) == len(orphan.spans):
                    del lane.orphans[orphan_id]
                else:
                    # A failed release keeps the orphan, and its remaining
                    # spans pinned, visible instead of silently dropping them.
                    orphan.spans = tuple(
                        span for span in orphan.spans if span not in released
                    )
            recovered = lane.refusing and len(lane.orphans) < self.timeouts.max_orphans
            if recovered:
                lane.refusing = False
        error = future.exception()
        logger.warning(
            "%sHiCache storage %s call %s orphaned by its deadline returned after "
            "%.1fs (%s); its host span is released%s",
            self._log_prefix(),
            orphan.kind.value,
            orphan.op,
            time.monotonic() - orphan.started_at,
            "ok" if error is None else repr(error),
            f", {orphan.kind.value} calls are served again" if recovered else "",
        )

    def _log_prefix(self) -> str:
        return f"[{self._label}] " if self._label else ""

    def _warn(self, lane: _Lane, message: str, *args, exc_info: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
            if now - lane.last_warn_at < _WARN_INTERVAL_S:
                lane.suppressed_warnings += 1
                return
            suppressed = lane.suppressed_warnings
            lane.suppressed_warnings = 0
            lane.last_warn_at = now
        if suppressed:
            message += " (%d similar warnings suppressed)"
            args = (*args, suppressed)
        logger.warning(self._log_prefix() + message, *args, exc_info=exc_info)

    def orphan_thread_idents(self) -> set[int]:
        """Threads still running an orphaned call."""
        with self._lock:
            return {
                orphan.thread_ident
                for lane in self._lanes.values()
                for orphan in lane.orphans.values()
                if orphan.thread_ident is not None
            }

    def take_stats(self) -> StorageCallGuardStats:
        """Snapshot the counters; timeouts and errors reset to zero."""
        with self._lock:
            stats = StorageCallGuardStats(
                timeouts={k.value: lane.timeouts for k, lane in self._lanes.items()},
                errors={k.value: lane.errors for k, lane in self._lanes.items()},
                orphaned={
                    k.value: len(lane.orphans) for k, lane in self._lanes.items()
                },
            )
            for lane in self._lanes.values():
                lane.timeouts = 0
                lane.errors = 0
            return stats

    def shutdown(self) -> None:
        """Retire the idle workers; orphans keep running until their call returns."""
        with self._lock:
            for lane in self._lanes.values():
                if lane.worker is not None:
                    lane.worker.inbox.put(None)
                    lane.worker = None
