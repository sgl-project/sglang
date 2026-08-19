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
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch

from sglang.srt.cuda_vmm_utils import BumpArenaStub
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_resources
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)
_active_graph_pool_user: Optional[str] = None
_borrow_stub: Optional[BumpArenaStub] = None
_borrow_mem_pool: Optional[torch.cuda.MemPool] = None
_borrow_disabled_reason: Optional[str] = None
_borrow_static_runs: Optional[list[tuple[int, int]]] = None
_borrow_extents_total = 0
_largest_logged_graph_pool_borrow = 0


def disable_graph_pool_borrow(reason: str) -> None:
    """Disable borrowing when graph storage is managed outside the shared pool."""
    global _borrow_disabled_reason
    _borrow_disabled_reason = reason
    logger.info("Graph pool borrow disabled: %s", reason)


def set_graph_pool_borrow_runs(runs: list[tuple[int, int]]) -> None:
    """Use fixed graph-storage extents instead of snapshots of the shared pool.

    This supports graph storage whose addresses are managed externally but
    remain stable for the process lifetime. Registering an empty list disables
    borrowing.
    """
    global _borrow_static_runs
    _borrow_static_runs = sorted(runs, key=lambda run: run[1], reverse=True)[
        : BumpArenaStub.MAX_EXTENTS
    ]
    logger.info(
        "Graph pool borrow runs pinned: runs=%d free=%d",
        len(_borrow_static_runs),
        sum(nbytes for _, nbytes in _borrow_static_runs),
    )


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


def graph_pool_borrow_enabled() -> bool:
    if (
        _borrow_disabled_reason is not None
        or not envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.get()
        or not is_cuda()
    ):
        return False
    if _borrow_static_runs is not None:
        return len(_borrow_static_runs) > 0
    return get_global_graph_memory_pool() is not None


@contextmanager
def graph_pool_user_scope(user: str) -> Iterator[None]:
    global _active_graph_pool_user
    # Graph replay silently overwrites aliases of its allocator-free blocks.
    if _active_graph_pool_user is not None:
        raise RuntimeError(
            f"graph pool already has live user {_active_graph_pool_user!r}; "
            f"cannot use it for {user!r}"
        )
    _active_graph_pool_user = user
    try:
        yield
    finally:
        _active_graph_pool_user = None


@contextmanager
def graph_pool_replay_scope() -> Iterator[None]:
    if not graph_pool_borrow_enabled():
        yield
        return
    with graph_pool_user_scope("CUDA graph"):
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
    """Return contiguous inactive runs inside wholly-free segments, largest first."""
    runs: list[tuple[int, int]] = []
    for segment in torch.cuda.memory_snapshot(pool_id, include_traces=False):
        if segment["allocated_size"] != 0:
            continue
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


def _teardown_borrow_pool() -> None:
    """Retire the persistent borrow pool after draining deferred frees."""
    global _borrow_mem_pool
    if _borrow_mem_pool is None:
        return
    # Borrowed blocks that saw cross-stream use can remain in event limbo.
    # Synchronize, then drive allocator event processing before dropping the pool.
    torch.cuda.synchronize()
    torch.empty(1, device="cuda")
    _borrow_mem_pool = None


@contextmanager
def borrow_graph_pool(user: str) -> Iterator[None]:
    """Route this thread's torch allocations onto the graph pool's free runs.

    Tensors allocated inside must not survive the current step: the next graph
    replay may overwrite their bytes. An allocation no run can hold raises the
    allocator's normal OOM. This is a no-op while borrowing is disabled.
    """
    global _borrow_stub, _borrow_mem_pool, _borrow_extents_total
    global _largest_logged_graph_pool_borrow
    if not graph_pool_borrow_enabled():
        yield
        return
    with graph_pool_user_scope(user):
        if _borrow_mem_pool is not None:
            # Return completed cross-stream frees to the cache. The allocator
            # processes their events on a later allocation.
            torch.empty(1, device="cuda")
            if (
                _borrow_stub.freed_bytes
                or _borrow_stub.cursor_bytes > _borrow_extents_total // 2
            ):
                # Rebuild if empty_cache released the arena's segments or if
                # unresolved deferred frees consumed half of the extents.
                _teardown_borrow_pool()
        if _borrow_mem_pool is None:
            if _borrow_stub is None:
                _borrow_stub = BumpArenaStub()
            if _borrow_static_runs is not None:
                runs = _borrow_static_runs
            else:
                runs = find_free_graph_pool_runs(get_global_graph_memory_pool())[
                    : BumpArenaStub.MAX_EXTENTS
                ]
            _borrow_stub.set_extents(runs)
            # Keep one caching layer across borrows so normal block reuse and
            # stream-ordered deferred frees remain allocator-managed. Capture
            # retires it because capture changes the underlying free extents.
            _borrow_mem_pool = torch.cuda.MemPool(_borrow_stub.allocator)
            _borrow_extents_total = sum(run_bytes for _, run_bytes in runs)
            logger.info(
                "Graph pool borrow extents: runs=%d free=%d",
                len(runs),
                _borrow_extents_total,
            )
        with torch.cuda.use_mem_pool(_borrow_mem_pool):
            yield
        consumed = _borrow_stub.cursor_bytes
        if consumed > _largest_logged_graph_pool_borrow:
            logger.info("Graph pool borrow: consumed=%d", consumed)
            _largest_logged_graph_pool_borrow = consumed
