"""Materialize shared-memory request data outside the scheduler thread."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from math import prod
from threading import Lock
from typing import Callable, Optional

import msgspec

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    BaseBatchReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.mm_utils import ShmPointerMMData
from sglang.srt.managers.schedule_batch import (
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.utils.nvtx_utils import operations_nvtx_range

# Received values (SHM-backed or not) queued for in-order publication.
_MAX_ITEMS = 64
# Values one poll receives under an unbounded scheduler poll; half the queue
# leaves room for local aborts and entries still pending from the last poll.
_RECV_BATCH = _MAX_ITEMS // 2
# Threads materializing SHM segments.
_WORKERS = 1


class AsyncShmEntry(msgspec.Struct, frozen=True):
    """Store one FIFO entry and its materialization future."""

    value: object
    size: int
    future: Optional[Future]


class AsyncShmStats(msgspec.Struct, frozen=True):
    """Report async SHM queue and resource usage."""

    submitted: int
    fallbacks: int
    active_items: int
    active_bytes: int
    max_active_items: int
    max_active_bytes: int
    pending_entries: int


def _validate_positive(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class AsyncShmMaterializer:
    """Materialize SHM entries asynchronously and publish them in FIFO order."""

    def __init__(
        self,
        *,
        max_recv_per_poll: int,
        discard: Callable[[object], None],
        max_items: int = _MAX_ITEMS,
        workers: int = _WORKERS,
    ) -> None:
        # The request receiver stops each poll at this many values.
        self.max_recv_per_poll = _validate_positive(
            "max_recv_per_poll", max_recv_per_poll
        )
        self.max_items = _validate_positive("max_items", max_items)
        self.max_bytes = (
            _validate_positive(
                "SGLANG_UOCR_ASYNC_SHM_MAX_MB",
                envs.SGLANG_UOCR_ASYNC_SHM_MAX_MB.get(),
            )
            * 1024
            * 1024
        )
        if self.max_recv_per_poll > self.max_items:
            raise ValueError("max_recv_per_poll must not exceed max_items")
        workers = _validate_positive("workers", workers)
        self.discard = discard
        self.entries = deque()
        self.active_items = 0
        self.active_bytes = 0
        self.max_active_items = 0
        self.max_active_bytes = 0
        self.submitted = 0
        self.fallbacks = 0
        self.receive_reservation = 0
        self.closing = False
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="uocr-shm-materialize",
        )

    def reserve_receive(self, *, extra_items: int = 0) -> bool:
        """Reserve bounded queue capacity before receiving requests."""
        if extra_items < 0:
            raise ValueError("extra_items must be nonnegative")
        with self.lock:
            if self.closing or self.receive_reservation:
                return False
            reservation = self.max_recv_per_poll + extra_items
            if len(self.entries) + reservation > self.max_items:
                return False
            self.receive_reservation = reservation
            return True

    def receive_reserved(self) -> bool:
        """Return whether a receive reservation is active."""
        with self.lock:
            return self.receive_reservation > 0

    def has_pending(self) -> bool:
        """Return whether received values are still queued for publication."""
        with self.lock:
            return bool(self.entries)

    def wait_for_progress(self) -> None:
        """Briefly yield the scheduler while the FIFO head materializes."""
        with self.lock:
            future = self.entries[0].future if self.entries else None
        if future is not None:
            wait((future,), timeout=0.001)

    def cancel_receive(self) -> None:
        """Cancel the active receive reservation."""
        with self.lock:
            self.receive_reservation = 0

    def submit(
        self,
        values,
        *,
        finalize: Callable[[list], None],
        has_shm: Callable[[list], bool],
    ) -> None:
        """Submit received values while preserving top-level FIFO order."""
        values = tuple(values)
        with self.lock:
            if self.closing:
                self.receive_reservation = 0
                raise RuntimeError("async SHM materializer is closed")
            if not self.receive_reservation:
                raise RuntimeError(
                    "async SHM materializer requires a receive reservation"
                )
            if len(values) > self.receive_reservation:
                self.receive_reservation = 0
                raise RuntimeError(
                    "async SHM materializer receive reservation exceeded"
                )
            if len(self.entries) + len(values) > self.max_items:
                self.receive_reservation = 0
                raise RuntimeError("async SHM materializer item reservation exceeded")
            self.receive_reservation = 0
            for value in values:
                if not has_shm([value]):
                    self.entries.append(AsyncShmEntry(value, 0, None))
                    continue
                size = estimate_shm_bytes(value)
                if (
                    self.active_items >= self.max_items
                    or self.active_bytes + size > self.max_bytes
                ):
                    with operations_nvtx_range("uocr.async_shm.fallback"):
                        finalize([value])
                    self.fallbacks += 1
                    self.entries.append(AsyncShmEntry(value, 0, None))
                    continue
                with operations_nvtx_range("uocr.async_shm.submit"):
                    future = self.executor.submit(self._run, finalize, value)
                self.active_items += 1
                self.active_bytes += size
                self.submitted += 1
                self.max_active_items = max(self.max_active_items, self.active_items)
                self.max_active_bytes = max(self.max_active_bytes, self.active_bytes)
                self.entries.append(AsyncShmEntry(value, size, future))

    @staticmethod
    def _run(finalize: Callable[[list], None], value) -> None:
        with operations_nvtx_range("uocr.async_shm.worker"):
            finalize([value])

    def pop_ready(self) -> list:
        """Pop the contiguous completed prefix from the FIFO queue."""
        ready = []
        with operations_nvtx_range("uocr.async_shm.ready"):
            with self.lock:
                while self.entries:
                    entry = self.entries[0]
                    if entry.future is not None and not entry.future.done():
                        break
                    self.entries.popleft()
                    if entry.future is not None:
                        self.active_items -= 1
                        self.active_bytes -= entry.size
                        try:
                            entry.future.result()
                        except BaseException:
                            self.discard(entry.value)
                            raise
                    ready.append(entry.value)
        return ready

    def stats(self) -> AsyncShmStats:
        """Return a snapshot of queue and resource counters."""
        with self.lock:
            return AsyncShmStats(
                submitted=self.submitted,
                fallbacks=self.fallbacks,
                active_items=self.active_items,
                active_bytes=self.active_bytes,
                max_active_items=self.max_active_items,
                max_active_bytes=self.max_active_bytes,
                pending_entries=len(self.entries),
            )

    def shutdown(self) -> None:
        """Stop submissions and release all queued SHM entries."""
        with self.lock:
            if self.closing:
                return
            self.closing = True
            self.receive_reservation = 0
            entries = tuple(self.entries)
            for entry in entries:
                if entry.future is not None:
                    entry.future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        with self.lock:
            self.entries.clear()
            for entry in entries:
                if entry.future is None:
                    continue
                if entry.future.cancelled():
                    self.discard(entry.value)
                    continue
                try:
                    entry.future.result()
                except BaseException:
                    self.discard(entry.value)
            self.active_items = 0
            self.active_bytes = 0


def create_async_shm_materializer(
    *,
    max_recv_per_poll: int,
    tp_size: int,
    pp_size: int,
    dp_size: int,
    attn_cp_size: int,
    nnodes: int,
    mm_feature_transport: Optional[str],
    scripted_runtime: bool,
    rust_server: bool,
    is_unlimited_ocr: bool,
) -> Optional[AsyncShmMaterializer]:
    """Create the materializer only for supported bounded configurations.

    A negative ``max_recv_per_poll`` (unbounded scheduler polls) receives
    ``_RECV_BATCH`` values per poll instead; other models keep their polls.
    """
    if not envs.SGLANG_ENABLE_UOCR_ASYNC_SHM_MATERIALIZATION.get():
        return None
    recv_batch = _RECV_BATCH if max_recv_per_poll < 0 else max_recv_per_poll
    if (
        not is_unlimited_ocr
        or recv_batch == 0
        or recv_batch > _MAX_ITEMS
        or tp_size != 1
        or pp_size != 1
        or dp_size != 1
        or attn_cp_size != 1
        or nnodes != 1
        or mm_feature_transport != "cpu"
        or scripted_runtime
        or rust_server
    ):
        return None
    from sglang.srt.managers.mm_utils import discard_shm_features

    return AsyncShmMaterializer(
        max_recv_per_poll=recv_batch,
        discard=discard_shm_features,
    )


def _pointer_bytes(candidate, pointers: dict[str, int]) -> None:
    values = candidate if isinstance(candidate, (list, tuple)) else (candidate,)
    for value in values:
        if isinstance(value, ShmPointerMMData):
            pointers[value.shm_name] = prod(value.shape) * value.dtype.itemsize


def estimate_shm_bytes(value) -> int:
    """Estimate unique logical SHM bytes owned by one top-level value."""
    pointers: dict[str, int] = {}
    requests = value.batch if isinstance(value, BaseBatchReq) else (value,)
    for request in requests:
        if not isinstance(
            request, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
        ):
            continue
        if not isinstance(
            request.mm_inputs, (MultimodalProcessorOutput, MultimodalInputs)
        ):
            continue
        for item in request.mm_inputs.mm_items:
            _pointer_bytes(item.feature, pointers)
            _pointer_bytes(item.precomputed_embeddings, pointers)
            if item.model_specific_data is not None:
                for candidate in item.model_specific_data.values():
                    _pointer_bytes(candidate, pointers)
    return sum(pointers.values())
