# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Process-wide CUDA graph memory pool shared across the prefill and
decode graph backends. The two phases never replay concurrently, so
sharing one pool reserves only the larger phase's capture footprint.
Serial capture passes also share one stream to reuse allocator scratch.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Iterator, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_resources, get_stream
from sglang.srt.utils import is_cuda
from sglang.srt.utils.cuda_vmm_utils import BumpArenaStub

logger = logging.getLogger(__name__)

_MIB = 1 << 20
_CAPTURE_STREAM_NAME = "cuda_graph_capture"


@dataclass(eq=False)
class GraphPoolBorrowState:
    """Mutable borrowing state and free-run cache for one runtime."""

    active_user: Optional[str] = None
    stub: Optional[BumpArenaStub] = None
    mem_pool: Optional[torch.cuda.MemPool] = None
    stream: Optional[torch.cuda.Stream] = None
    disabled_reason: Optional[str] = None
    static_runs: Optional[list[tuple[int, int]]] = None
    check_pending: bool = False
    extents_total: int = 0
    largest_logged_borrow: int = 0
    snapshot_free_runs: Any = field(
        default_factory=lambda: cache(find_free_graph_pool_runs), init=False, repr=False
    )


def _get_graph_pool_borrow_state() -> GraphPoolBorrowState:
    resources = get_resources()
    if resources.graph_pool_borrow is None:
        resources.graph_pool_borrow = GraphPoolBorrowState()
    return resources.graph_pool_borrow


def _log_graph_pool_borrow_capacity(runs: list[tuple[int, int]]) -> None:
    total_bytes = sum(nbytes for _, nbytes in runs)
    largest_bytes = max((nbytes for _, nbytes in runs), default=0)
    logger.info(
        "Graph pool borrow capacity: %.1f MiB available, largest contiguous "
        "region %.1f MiB, %d regions",
        total_bytes / _MIB,
        largest_bytes / _MIB,
        len(runs),
    )


def disable_graph_pool_borrow(reason: str) -> None:
    """Disable borrowing when graph storage is managed outside the shared pool."""
    _get_graph_pool_borrow_state().disabled_reason = reason
    _teardown_borrow_pool()
    logger.info("Graph pool borrow disabled: %s", reason)


def set_graph_pool_borrow_runs(runs: list[tuple[int, int]]) -> None:
    """Use fixed graph-storage extents instead of snapshots of the shared pool.

    This supports graph storage whose addresses are managed externally but
    remain stable for the process lifetime. Registering an empty list disables
    borrowing.
    """
    state = _get_graph_pool_borrow_state()
    static_runs = sorted(runs, key=lambda run: run[1], reverse=True)[
        : BumpArenaStub.MAX_EXTENTS
    ]
    _teardown_borrow_pool()
    state.static_runs = static_runs
    _log_graph_pool_borrow_capacity(state.static_runs)


def get_global_graph_memory_pool() -> Optional[Any]:
    return get_resources().graph_memory_pool


def set_global_graph_memory_pool(val: Any) -> None:
    get_resources().graph_memory_pool = val


def get_or_create_global_graph_memory_pool(device_module: Any) -> Any:
    """Return the shared graph memory pool, creating it on first use so
    later backends reuse the same handle."""
    resources = get_resources()
    if resources.graph_memory_pool is None:
        resources.graph_memory_pool = device_module.graph_pool_handle()
    return resources.graph_memory_pool


def get_or_create_global_graph_capture_stream() -> Any:
    """Return the shared graph capture stream, creating it on first use so every
    capture pass reserves the pool's scratch once instead of per stream.

    CUDA only — the NPU / XPU / CPU graph runners keep their own streams.
    """
    return get_stream(_CAPTURE_STREAM_NAME)


class GraphPoolPrecarve:
    """Pre-carve the memory pool to reduce fragmentation."""

    def __init__(self) -> None:
        self.nbytes = 0
        self.minted = False

    @contextmanager
    def measure(self) -> Iterator[None]:
        """Wrap one eager warmup. the last one before ``mint`` sets the size."""
        if self.minted or not envs.SGLANG_ENABLE_GRAPH_POOL_PRECARVE.get():
            yield
            return
        torch.cuda.synchronize()
        # Shrink the cache first so the warmup's reserved growth is its own
        # footprint. Reserved (not allocated) is the stat to use: allocated
        # peak is the live-byte sum and undershoots by exactly the packing
        # holes the carved span has to absorb.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_stats()["reserved_bytes.all.current"]
        yield
        torch.cuda.synchronize()
        self.nbytes = torch.cuda.memory_stats()["reserved_bytes.all.peak"] - base

    def mint(self) -> None:
        """Pre-allocate the space"""
        if self.minted:
            return
        self.minted = True
        if self.nbytes <= 0:
            return
        span = torch.empty(self.nbytes, dtype=torch.uint8, device="cuda")
        del span
        logger.info("Graph pool pre-carved: %.2f GB", self.nbytes / 2**30)


def graph_pool_borrow_enabled() -> bool:
    state = _get_graph_pool_borrow_state()
    if (
        state.disabled_reason is not None
        or not envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.get()
        or not is_cuda()
    ):
        return False
    if state.static_runs is not None:
        return len(state.static_runs) > 0
    return get_global_graph_memory_pool() is not None


@contextmanager
def graph_pool_user_scope(user: str) -> Iterator[None]:
    state = _get_graph_pool_borrow_state()
    # Graph replay silently overwrites aliases of its allocator-free blocks.
    if state.active_user is not None:
        raise RuntimeError(
            f"graph pool already has live user {state.active_user!r}; "
            f"cannot use it for {user!r}"
        )
    state.active_user = user
    try:
        yield
    finally:
        state.active_user = None


@contextmanager
def graph_pool_replay_scope() -> Iterator[None]:
    if not graph_pool_borrow_enabled():
        yield
        return
    with graph_pool_user_scope("CUDA graph"):
        if _get_graph_pool_borrow_state().check_pending:
            _raise_on_live_borrows("graph replay")
        yield


@contextmanager
def graph_pool_capture_scope() -> Iterator[None]:
    if not graph_pool_borrow_enabled():
        yield
        return
    with graph_pool_user_scope("CUDA graph"):
        # Capture re-carves the pool's free space: the borrow extents go stale,
        # so retire the borrow pool before capturing.
        _teardown_borrow_pool()
        yield


def find_free_graph_pool_runs(pool_id: Any) -> list[tuple[int, int]]:
    """Return contiguous inactive runs across all pool segments, largest first;
    live and pending-free blocks break runs, so a run never overlaps live data.
    """
    runs: list[tuple[int, int]] = []
    for segment in torch.cuda.memory_snapshot(pool_id, include_traces=False):
        run_address = 0
        run_bytes = 0
        for block in segment["blocks"]:
            if block["state"] == "inactive":
                if run_bytes == 0:
                    run_address = block["address"]
                run_bytes += block["size"]
                continue
            if run_bytes:
                runs.append((run_address, run_bytes))
            run_bytes = 0
        if run_bytes:
            runs.append((run_address, run_bytes))
    runs.sort(key=lambda run: run[1], reverse=True)
    return runs


def graph_pool_borrow_largest_run() -> int:
    """Largest contiguous free extent a single borrow can occupy, in bytes."""
    if not graph_pool_borrow_enabled():
        return 0
    state = _get_graph_pool_borrow_state()
    if state.static_runs is not None:
        return state.static_runs[0][1]
    runs = state.snapshot_free_runs(get_global_graph_memory_pool())
    return runs[0][1] if runs else 0


def _raise_on_live_borrows(event: str) -> None:
    """Reject borrows that graph replay or pool teardown would overwrite."""
    state = _get_graph_pool_borrow_state()
    state.check_pending = False
    if state.mem_pool is None:
        return
    live = sum(
        block["size"]
        for segment in torch.cuda.memory_snapshot(
            state.mem_pool.id, include_traces=False
        )
        for block in segment["blocks"]
        if block["state"] == "active_allocated"
    )
    if live:
        raise RuntimeError(
            f"Graph-pool borrow leak at {event}: {live} bytes still referenced"
        )


def _teardown_borrow_pool() -> None:
    """Retire the persistent borrow pool after draining deferred frees."""
    state = _get_graph_pool_borrow_state()
    state.snapshot_free_runs.cache_clear()
    if state.mem_pool is None:
        return
    if state.check_pending:
        _raise_on_live_borrows("borrow pool teardown")
    # Borrowed blocks that saw cross-stream use can remain in event limbo.
    # Synchronize, then drive allocator event processing before dropping the pool.
    torch.cuda.synchronize()
    torch.empty(1, device="cuda")
    state.mem_pool = None
    state.stream = None


_PRECARVE_MIN_RUN_BYTES = 64 << 20
# Left uncarved per run so 2 MiB small-pool segments keep a home.
_PRECARVE_SMALL_RESERVE_BYTES = 32 << 20
# The caching allocator rounds large segment requests up to 2 MiB.
_PRECARVE_GRANULARITY = 2 << 20


def graph_pool_borrow_can_fit(nbytes: int) -> bool:
    """Whether one free run fits the payload plus allocator padding and reserve."""
    if nbytes <= 0:
        return False
    required = (
        nbytes + _PRECARVE_GRANULARITY - 1
    ) // _PRECARVE_GRANULARITY * _PRECARVE_GRANULARITY + _PRECARVE_SMALL_RESERVE_BYTES
    return graph_pool_borrow_largest_run() >= required


def _precarve_run_segments(runs: list[tuple[int, int]]) -> None:
    """Seed coalescible segments on the stream that will allocate borrows."""
    for _, run_bytes in runs:
        seed = (
            (run_bytes - _PRECARVE_SMALL_RESERVE_BYTES) // _PRECARVE_GRANULARITY
        ) * _PRECARVE_GRANULARITY
        if seed >= _PRECARVE_MIN_RUN_BYTES:
            torch.empty(seed, dtype=torch.uint8, device="cuda")


@contextmanager
def borrow_graph_pool(user: str) -> Iterator[None]:
    """Route this thread's torch allocations onto the graph pool's free runs.

    All borrows must use the stream that first creates the borrow pool, so
    the caching allocator can reuse its pre-carved segments.
    Tensors allocated inside must be released before the next graph replay,
    which rewrites their bytes; the next replay (or pool teardown) raises if
    any are still referenced. An allocation no run can hold raises the
    allocator's normal OOM. This is a no-op while borrowing is disabled.
    """
    state = _get_graph_pool_borrow_state()
    if not graph_pool_borrow_enabled():
        yield
        return
    with graph_pool_user_scope(user):
        stream = torch.cuda.current_stream()
        if state.mem_pool is not None:
            if stream != state.stream:
                raise RuntimeError(
                    "Graph-pool borrow must use the stream that created the borrow pool: "
                    f"expected {state.stream}, got {stream}"
                )
            # Return completed cross-stream frees to the cache. The allocator
            # processes their events on a later allocation.
            torch.empty(1, device="cuda")
            if state.stub.freed_bytes:
                # The bump arena cannot reuse segments returned by empty_cache().
                _teardown_borrow_pool()
        if state.mem_pool is None:
            if state.stub is None:
                state.stub = BumpArenaStub()
                # Runs are sorted largest first, so first fit would carve every
                # small allocation out of the run a probability matrix needs.
                state.stub.set_best_fit(True)
            if state.static_runs is not None:
                runs = state.static_runs
            else:
                runs = state.snapshot_free_runs(get_global_graph_memory_pool())[
                    : BumpArenaStub.MAX_EXTENTS
                ]
            state.stub.set_extents(runs)
            # Keep one caching layer across borrows so normal block reuse and
            # stream-ordered deferred frees remain allocator-managed. Capture
            # retires it because capture changes the underlying free extents.
            state.mem_pool = torch.cuda.MemPool(state.stub.allocator)
            state.stream = stream
            with torch.cuda.use_mem_pool(state.mem_pool):
                _precarve_run_segments(runs)
            # Only growth beyond the precarve is worth another log line.
            state.largest_logged_borrow = state.stub.cursor_bytes
            state.extents_total = sum(run_bytes for _, run_bytes in runs)
            _log_graph_pool_borrow_capacity(runs)
            logger.info(
                "Graph pool borrow pre-carved: %.1f MiB",
                state.stub.cursor_bytes / _MIB,
            )
        with torch.cuda.use_mem_pool(state.mem_pool):
            yield
        state.check_pending = True
        consumed = state.stub.cursor_bytes
        if consumed > state.largest_logged_borrow:
            logger.info(
                "Graph pool borrow high-water mark: %.1f MiB used of %.1f MiB "
                "available",
                consumed / _MIB,
                state.extents_total / _MIB,
            )
            state.largest_logged_borrow = consumed
