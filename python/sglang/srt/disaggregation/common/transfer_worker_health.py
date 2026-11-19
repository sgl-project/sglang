"""Liveness of a prefill KV manager's transfer worker threads.

Each transfer worker drains one shard queue, so a worker that exits or hangs
strands every later request hashed onto that queue while the server keeps
answering health checks. This tracker records both states so the scheduler
can turn them into a failed health check.
"""

import concurrent.futures
import logging
import threading
import time
from typing import Callable, Dict, Optional

import torch
from prometheus_client import Counter

logger = logging.getLogger(__name__)

TRANSFER_WORKER_DEATHS = Counter(
    "sglang:pd_transfer_worker_deaths_total",
    "Number of PD prefill KV transfer worker threads that exited on an error.",
)
TRANSFER_WORKER_REQUEST_FAILURES = Counter(
    "sglang:pd_transfer_worker_request_failures_total",
    "Number of requests a PD prefill KV transfer worker failed on an "
    "unexpected error while it kept running.",
)


_ACCELERATOR_ERROR = getattr(torch, "AcceleratorError", ())


def is_worker_fatal_error(exc: BaseException) -> bool:
    """Whether an error breaks the worker itself rather than one request.

    Only two kinds qualify; every other error is scoped to the chunk that
    raised it and fails only that request.
    - A CUDA (or HIP) error: it is sticky, so every later event wait on this
      device raises too and the worker could not send anything again.
    - A thread pool that is broken or shut down: every later send is submitted
      to the same pool and would fail the same way.
    """
    if isinstance(exc, _ACCELERATOR_ERROR):
        return True
    if isinstance(exc, concurrent.futures.BrokenExecutor):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc)
        return (
            "CUDA error" in msg
            or "HIP error" in msg
            or "cannot schedule new futures" in msg
        )
    return False


class TransferWorkerHealth:
    """Thread-safe record of which transfer workers died or are stuck on one chunk.

    A death is permanent: the thread is gone and its queue is never drained
    again. A stall clears here once the worker finishes the chunk, but the
    server stays unhealthy once one was reported: a chunk stuck past the limit
    has outlived the waiting timeout of every request behind it.
    """

    def __init__(
        self,
        stall_timeout_s: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._lock = threading.Lock()
        self._clock = clock
        self.stall_timeout_s = stall_timeout_s
        self._dead: Dict[int, str] = {}
        self._busy_since: Dict[int, float] = {}

    def mark_busy(self, worker_index: int) -> None:
        with self._lock:
            self._busy_since[worker_index] = self._clock()

    def mark_idle(self, worker_index: int) -> None:
        with self._lock:
            self._busy_since.pop(worker_index, None)

    def mark_dead(self, worker_index: int, reason: str) -> None:
        with self._lock:
            if worker_index in self._dead:
                return
            self._dead[worker_index] = reason
            self._busy_since.pop(worker_index, None)
        TRANSFER_WORKER_DEATHS.inc()

    def is_dead(self, worker_index: int) -> bool:
        with self._lock:
            return worker_index in self._dead

    def record_request_failure(self) -> None:
        TRANSFER_WORKER_REQUEST_FAILURES.inc()

    def unhealthy_reason(self) -> Optional[str]:
        """Why the workers cannot serve every shard, or None when they can."""
        with self._lock:
            if self._dead:
                worker_index, reason = min(self._dead.items())
                return (
                    f"KV transfer worker {worker_index} is dead "
                    f"({len(self._dead)} dead): {reason}"
                )
            if self.stall_timeout_s > 0 and self._busy_since:
                worker_index, since = min(self._busy_since.items(), key=lambda x: x[1])
                stalled_s = self._clock() - since
                if stalled_s > self.stall_timeout_s:
                    return (
                        f"KV transfer worker {worker_index} has been on one chunk "
                        f"for {stalled_s:.0f}s (limit {self.stall_timeout_s:.0f}s)"
                    )
        return None
