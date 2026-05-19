# Copyright 2023-2024 SGLang Team
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
"""CPU-only scheduler for the Rust (msgpack IPC) execution path.

In this architecture:
  - GPU workers run as separate subprocesses (one per TP/PP rank), each
    wrapping a TpModelWorker inside a TpWorkerServer ZMQ loop.
  - A single CpuScheduler process (one per PP stage) drives all TP workers
    via TpWorkerClientGroup over ZMQ PAIR sockets.

The CpuScheduler never touches a GPU directly:
  - No CUDA streams (schedule_stream / forward_stream removed).
  - req_to_token_pool lives on CPU and tracks prefix-hit slot indices for
    the radix tree.  New-token slot indices are allocated by the GPU worker
    (PagedTokenToKVPoolAllocator with Triton kernels) and sent back to the
    CPU after each batch via GenerationBatchResult.deferred_alloc.
  - CpuPageTracker replaces PagedTokenToKVPoolAllocator on the CPU side:
    it tracks only the free-page count (for scheduling budget) and
    accumulates freed slot indices to forward to the GPU worker.
"""

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.managers.scheduler import Scheduler, dispatch_event_loop
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.server_args import PortArgs, ServerArgs, set_global_server_args_for_scheduler

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal KV cache placeholder — the real GPU KV data lives in the worker
# ---------------------------------------------------------------------------

class _CpuKvCacheProxy:
    """Placeholder KVCache used by CpuPageTracker.

    The CPU scheduler only needs the allocator for bookkeeping (tracking
    which slot indices are free/allocated).  The actual KV tensors live on
    the GPU worker.  This proxy satisfies the KVCache interface without
    allocating any GPU memory.
    """

    def __init__(self, size: int, page_size: int = 1) -> None:
        self.size = size
        self.device = "cpu"
        self.page_size = page_size
        self.mem_usage = 0.0

    def maybe_get_custom_mem_pool(self):
        return None


# ---------------------------------------------------------------------------
# CPU-side page-count tracker — defers slot computation to GPU worker
# ---------------------------------------------------------------------------

class CpuPageTracker:
    """Minimal KV page allocator that defers slot computation to the GPU worker.

    The CPU scheduler uses this to:
      1. Track the free page count (approximation, synced from GPU results)
         for scheduling budget decisions.
      2. Accumulate freed slot indices (from request completion / eviction)
         to forward to the GPU worker via ModelWorkerBatch.indices_to_free.

    All actual Triton-kernel slot index computation (alloc_extend_kernel /
    alloc_decode_kernel) happens in TpWorkerServer._allocate_kv_deferred()
    on the GPU worker process.
    """

    # Signal checked by alloc_for_extend / alloc_for_decode in common.py
    deferred_to_gpu: bool = True
    is_not_in_free_group: bool = True

    def __init__(self, total_pages: int, page_size: int) -> None:
        self.total_pages = total_pages
        self.page_size = page_size
        # Start with all pages free except the reserved page 0
        self._free_page_count: int = total_pages
        self._pending_free: List[torch.Tensor] = []
        self._kvcache = _CpuKvCacheProxy(
            size=total_pages * page_size, page_size=page_size
        )
        self.device = "cpu"

    # ------------------------------------------------------------------
    # Interface required by Scheduler / PrefillAdder / evict_from_tree_cache
    # ------------------------------------------------------------------

    def available_size(self) -> int:
        return self._free_page_count * self.page_size

    def get_kvcache(self) -> _CpuKvCacheProxy:
        return self._kvcache

    def free(self, indices: torch.Tensor) -> None:
        """Accumulate freed slot indices to forward to the GPU worker."""
        if indices.numel() == 0:
            return
        cpu_indices = indices.cpu()
        self._pending_free.append(cpu_indices)
        # Approximate count update (page-aligned unique pages)
        freed_pages = int(torch.unique(cpu_indices // self.page_size).numel())
        self._free_page_count += freed_pages

    def drain_pending_free(self) -> Optional[torch.Tensor]:
        """Return and clear all accumulated free indices (sent with next batch)."""
        if not self._pending_free:
            return None
        # Common case: a single freed-indices tensor — skip torch.cat.
        if len(self._pending_free) == 1:
            result = self._pending_free[0]
            self._pending_free.clear()
            return result
        result = torch.cat(self._pending_free)
        self._pending_free.clear()
        return result

    def update_free_count(self, free_pages: int) -> None:
        """Sync free page count from GPU worker's authoritative report."""
        self._free_page_count = free_pages

    # ------------------------------------------------------------------
    # Stubs: alloc_* are no-ops; real allocation happens on the GPU worker
    # ------------------------------------------------------------------

    def alloc(self, *args, **kwargs):
        return None

    def alloc_extend(self, *args, **kwargs):
        return None

    def alloc_decode(self, *args, **kwargs):
        return None

    def backup_state(self):
        return None

    def restore_state(self, state):
        pass

    def free_group_begin(self):
        pass

    def free_group_end(self):
        pass

    def clear(self):
        self._free_page_count = self.total_pages - 1
        self._pending_free.clear()

    def merge_and_sort_free(self):
        pass


# ---------------------------------------------------------------------------
# CPU-only Scheduler
# ---------------------------------------------------------------------------

class CpuScheduler(Scheduler):
    """Scheduler that runs entirely on CPU, delegating GPU work to TpWorkerServer.

    Key differences from the base Scheduler:

    1. init_model_worker  — forces self.device = "cpu" after the base class
       connects to the GPU workers, so all ScheduleBatch tensors are on CPU.

    2. _acquire_memory_pools — creates:
         - CPU-resident ReqToTokenPool  (for req slot assignment and radix tree)
         - CpuPageTracker              (count-only, defers slot computation to GPU)

    3. init_overlap  — replaces CUDA stream setup with no-ops, forces
       enable_overlap = False (overlap scheduling requires CUDA streams).

    4. run_event_loop — calls dispatch_event_loop directly, without wrapping
       the loop in a CUDA StreamContext.

    5. update_cache_from_scheduler — applies GPU-returned slot indices to the
       CPU req_to_token_pool so the radix tree stays accurate.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        dp_rank: Optional[int],
    ):
        set_global_server_args_for_scheduler(server_args)

        super().__init__(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            attn_cp_rank,
            moe_dp_rank,
            dp_rank,
        )

    # ------------------------------------------------------------------
    # Override: force CPU device after worker connection
    # ------------------------------------------------------------------

    def init_model_worker(self) -> None:
        super().init_model_worker()
        # Force CPU: all ScheduleBatch tensors are created on self.device.
        # The GPU worker receives CPU tensors and moves them to GPU in
        # TpWorkerServer._prepare_batch().
        self.device = "cpu"
        self.forward_stream = None

    # ------------------------------------------------------------------
    # Override: CPU req pool + GPU-deferred KV allocator
    # ------------------------------------------------------------------

    def _acquire_memory_pools(
        self,
    ) -> Tuple["ReqToTokenPool", "CpuPageTracker"]:
        """Build CPU-side pools using sizes from the GPU worker handshake.

        ReqToTokenPool lives on CPU and is used by the radix tree to store
        prefix-hit slot indices.  CpuPageTracker is a lightweight page-count
        tracker; actual KV slot allocation is deferred to the GPU worker.
        """
        worker_info = self.tp_worker.get_worker_info()
        req_pool_size: int = worker_info[9]
        req_pool_max_ctx: int = worker_info[10]
        kv_pool_size: int = worker_info[11]
        page_size: int = self.server_args.page_size

        req_to_token_pool = ReqToTokenPool(
            size=req_pool_size,
            max_context_len=req_pool_max_ctx,
            device="cpu",
            enable_memory_saver=False,
        )

        num_pages = kv_pool_size // page_size
        page_tracker = CpuPageTracker(total_pages=num_pages, page_size=page_size)

        logger.info(
            "CpuScheduler: created CPU req pool (size=%d, max_ctx=%d) "
            "and CpuPageTracker (pages=%d, page_size=%d) — "
            "KV slot allocation deferred to GPU worker",
            req_pool_size,
            req_pool_max_ctx,
            num_pages,
            page_size,
        )
        return req_to_token_pool, page_tracker

    # ------------------------------------------------------------------
    # Override: no-op overlap init (CUDA streams not available on CPU)
    # ------------------------------------------------------------------

    def init_overlap(self) -> None:
        self.device_module = type(
            "_CpuDeviceModule", (), {"Stream": lambda **_: None, "Event": lambda: None}
        )()

        self.enable_overlap = False
        self.enable_overlap_mlx = False

        self.forward_stream_ctx = nullcontext()
        self.copy_stream = None
        self.copy_stream_ctx = nullcontext()

        self.future_map = None

    # ------------------------------------------------------------------
    # Override: no schedule_stream wrapping needed
    # ------------------------------------------------------------------

    def run_event_loop(self) -> None:
        dispatch_event_loop(self)

    # ------------------------------------------------------------------
    # Override: sync GPU-allocated slot indices back into CPU req_to_token
    # ------------------------------------------------------------------

    def update_cache_from_scheduler(
        self,
        schedule_batch: "ScheduleBatch",
        batch_result: "GenerationBatchResult",
    ) -> None:
        """Apply GPU-returned KV slot assignments to the CPU req_to_token_pool.

        After TpWorkerServer._allocate_kv_deferred() runs Triton kernels on
        the GPU, it returns the newly allocated slot indices in
        batch_result.deferred_alloc.  We write these into the CPU-side
        req_to_token_pool so the radix tree can use them for future prefix
        matching, and update CpuPageTracker's free-page budget.
        """
        if batch_result is None:
            return
        da = getattr(batch_result, "deferred_alloc", None)
        if da is None:
            return

        pool = self.req_to_token_pool
        req_pool_indices = da["req_pool_indices"]  # int64 CPU tensor [bs]
        out_cache_loc = da["out_cache_loc"]        # int32 CPU tensor [tokens]

        if da["mode"] == "extend":
            prefix_lens = da["prefix_lens"]    # int64 CPU tensor [bs]
            extend_lens = da["extend_lens"]    # int64 CPU tensor [bs]
            bs = req_pool_indices.numel()
            # Single-request fast path — skip the index construction overhead.
            if bs == 1:
                req_idx = int(req_pool_indices[0])
                pre = int(prefix_lens[0])
                ext = int(extend_lens[0])
                pool.req_to_token[req_idx, pre : pre + ext] = (
                    out_cache_loc[:ext].to(torch.int32)
                )
            else:
                # Vectorized scatter: build (row, col) index pairs once, then
                # do a single fancy-indexed write. Avoids the O(bs) Python loop
                # and O(bs) .item() calls in the original implementation.
                extend_lens_long = extend_lens.long()
                # For each token in out_cache_loc, its (req_pool_row, col) is:
                #   row = repeat_interleave(req_pool_indices, extend_lens)
                #   col = (prefix_lens repeated) + (per-request 0..ext-1 offset)
                row_idx = torch.repeat_interleave(req_pool_indices.long(), extend_lens_long)
                # Per-token offset within its request's extend slice.
                # cumulative_extend[i] is the start of req i's slice in out_cache_loc.
                cumulative_extend = torch.cumsum(extend_lens_long, dim=0)
                # Compute per-token starts via repeat_interleave of the slice offsets.
                slice_starts = torch.cat(
                    [torch.zeros(1, dtype=torch.long), cumulative_extend[:-1]]
                )
                # token_idx_within_slice[k] = k - slice_starts of its request
                total_tokens = int(cumulative_extend[-1])
                token_indices = torch.arange(total_tokens, dtype=torch.long)
                slice_start_per_token = torch.repeat_interleave(slice_starts, extend_lens_long)
                token_offset = token_indices - slice_start_per_token
                col_idx = (
                    torch.repeat_interleave(prefix_lens.long(), extend_lens_long)
                    + token_offset
                )
                pool.req_to_token[row_idx, col_idx] = out_cache_loc.to(torch.int32)

        elif da["mode"] == "decode":
            # Single indexed write instead of a per-request Python loop:
            # for each i, set pool.req_to_token[req_pool_indices[i], seq_lens_minus1[i]] = out_cache_loc[i]
            seq_lens_minus1 = da["seq_lens_minus1"]  # int64 CPU tensor [bs]
            pool.req_to_token[
                req_pool_indices.long(),
                seq_lens_minus1.long(),
            ] = out_cache_loc.to(torch.int32)

        # Sync free page count from the GPU worker's authoritative allocator
        free_pages = da.get("free_pages_remaining")
        if free_pages is not None:
            self.token_to_kv_pool_allocator.update_free_count(free_pages)
