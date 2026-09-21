"""Controller-owned completion and admission records for asynchronous L2 I/O.

No CUDA Event is impersonated by a Future. A failed-but-drained task is terminal,
but never successful; an uncertain task retains its resource owners.
"""

import time
from collections import defaultdict
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum

from sglang.srt.kv_compression.types import BufferDrainError


def should_skip_full_load(count, threshold, *, has_aux=False):
    """HiCache policy; auxiliary transfers retain their ordinary exception."""
    return count < max(1, threshold) and not has_aux


def exceeds_load_quota(count, protected_delta, quota):
    return quota is not None and count + protected_delta > quota


def record_load_back_metrics(collector, tokens_by_pool, num_bytes, gpu_seconds=None):
    """Record completed I/O once, separately from request prefix adoption."""
    if collector is None:
        return
    for pool, count in (tokens_by_pool or {}).items():
        if count > 0:
            collector.increment_load_back_num_tokens(num_tokens=count, pool=pool)
    if num_bytes > 0:
        collector.increment_load_back_num_bytes(num_bytes)
    if gpu_seconds is not None:
        collector.observe_load_back_duration(gpu_seconds)


@dataclass(frozen=True)
class RestoreTransferResult:
    pages: int
    actual_bytes: int
    logical_bytes: int
    queue_seconds: float
    execution_seconds: float
    gpu_seconds: float | None = None


class TransferState(Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


@dataclass
class TransferCompletion:
    start_event: object = None
    finish_event: object = None
    timing_enabled: bool = False
    future: object = None
    started: float = field(default_factory=time.perf_counter)

    def query(self):
        if self.future is not None:
            return self.future.done()
        return self.finish_event.query()

    @property
    def state(self):
        if not self.query():
            return TransferState.PENDING
        try:
            error = self.future.exception() if self.future is not None else None
        except CancelledError as exc:
            error = exc
        if isinstance(error, BufferDrainError):
            return TransferState.UNCERTAIN
        return TransferState.FAILED if error is not None else TransferState.SUCCESS

    def result(self):
        if not self.query():
            raise RuntimeError("L2 result requested before completion")
        return self.wait()

    def wait(self):
        """Explicit blocking drain, for write-back/teardown only, never polling."""
        if self.future is not None:
            return self.future.result()
        self.finish_event.synchronize()
        return None

    @property
    def actual_bytes(self):
        if self.state is TransferState.SUCCESS and self.future is not None:
            value = self.future.result()
            if isinstance(value, RestoreTransferResult):
                return value.actual_bytes
            return value if isinstance(value, int) else None
        return None


@dataclass
class RestoreTicket:
    anchor: object
    host_indices: object
    device_indices: object
    device_lock: object
    host_lock: object
    leases: list
    completion: TransferCompletion
    req: object
    page_refs: tuple
    ready: bool = False
    consumed: bool = False
    abandoned: bool = False


class AsyncL2State:
    """Execution/admission state owned by CacheController, not a cache adapter."""

    def __init__(self, runtime, pool, provider):
        self.runtime, self.pool, self.provider = runtime, pool, provider
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hicache-l2")
        self.restore_ticket = None
        self.backup_pins = {}
        self.backup_completions = {}
        self.backup_dependencies = {}
        self.quarantined = []
        self.stats = defaultdict(
            float,
            dict.fromkeys(
                [
                    "backed_up_pages",
                    "restored_pages",
                    "published_lz4_pages",
                    "published_raw_pages",
                    "verified_backup_pages",
                    "restored_lz4_pages",
                    "verified_restored_pages",
                    "backup_failures",
                    "restore_failures",
                    "cancelled_restores",
                    "restore_admission_retries",
                    "backup_duplicate_skips",
                    "admission_skips",
                    "backup_submitted",
                    "backup_completed",
                    "backup_parent_missing",
                    "backup_window_pages",
                    "backup_queue_seconds",
                    "backup_wait_seconds",
                    "backup_d2h_seconds",
                    "backup_d2h_gpu_ms",
                    "backup_publish_seconds",
                    "backup_verify_seconds",
                    "backup_admission_seconds",
                    "d2h_bytes",
                    "h2d_bytes",
                    "restore_seconds",
                    "completed_restore_pages",
                    "completed_restore_logical_bytes",
                    "restore_queue_seconds",
                    "restore_execution_seconds",
                ],
                0,
            ),
        )
        self.last_metrics = 0.0
        self.closed = False
        self.io_uncertain = False

    def submit_backup(self, handles, refs, indices, ready, node_ids=()):
        if self.quarantined:
            raise BufferDrainError("L2 is quarantined")
        handle_list, index_list = handles.tolist(), indices.cpu().tolist()
        enqueued = time.perf_counter()
        dependencies = [
            self.backup_dependencies[n]
            for n in node_ids
            if n in self.backup_dependencies
        ]

        def write():
            self.stats["backup_queue_seconds"] += time.perf_counter() - enqueued
            if self.io_uncertain:
                raise BufferDrainError("Earlier L2 I/O did not drain")
            try:
                for dependency in dependencies:
                    dependency.future.result()  # Same FIFO worker: parent precedes child.
                return self.pool.io._write(handle_list, refs, index_list, ready)
            except BufferDrainError as exc:
                self.io_uncertain = True
                # P/D can share this executor: isolate it too, not just L2.
                quarantine = getattr(self.runtime, "quarantine", None)
                if quarantine is not None:
                    quarantine(exc, (handles, indices, refs))
                raise

        return TransferCompletion(future=self.worker.submit(write))

    def idle(self):
        return (
            not (
                self.restore_ticket
                or self.backup_pins
                or self.quarantined
                or self.io_uncertain
            )
            and self.runtime.idle()
            and not getattr(self.pool, "has_readers", lambda: False)()
        )

    def close(self):
        if not self.closed:
            self.worker.shutdown(wait=True)
            self.runtime.close()
            self.closed = True
