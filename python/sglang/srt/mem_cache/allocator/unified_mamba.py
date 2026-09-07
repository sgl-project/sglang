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
"""Unified-memory composite for hybrid Mamba models: the full-attention and
mamba-state end pools of one `UnifiedKVPool`."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence

import torch
from torch.profiler import record_function

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.unified_sub_pool import (
    MultiEndedAllocator,
    _chain_byte_accounting_violations,
    _end_pair_chain,
)
from sglang.srt.mem_cache.unified_memory_pool import UnifiedKVPool
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


class UnifiedMambaTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Composite allocator for the MHA (full-attn) + Mamba hybrid pair.

    The token-slot surface is the full-attn side; the mamba sub-pool's per-request
    `alloc(1)` is driven separately by `UnifiedHybridReqToTokenPool`. The two
    sub-allocators own independent virtual-id spaces.
    """

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        kvcache,  # HybridLinearKVPool
        device: str,
        page_size: int = 1,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
        lazy_compaction: bool = False,
    ):
        full_max = unified_buffer.max_slots("full")
        dcp_size = get_parallel().attn_dcp_size
        super().__init__(
            size=(full_max - 1) * dcp_size,
            page_size=page_size * dcp_size,
            dtype=unified_buffer.spec("full").get_dtype(),
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.unified_buffer = unified_buffer
        self._kvcache = kvcache
        # Widened under DCP, matching the full sub-allocator; see its __init__.
        self.page_size = page_size * dcp_size
        self.lazy_compaction = lazy_compaction

        # Only FULL shards under DCP; the mamba state is replicated on every rank
        # and stays page_size=1, orthogonal to the full side's per-token paging.
        self.full_attn_allocator = MultiEndedAllocator(
            kvcache=kvcache.full_kv_pool,
            unified_buffer=unified_buffer,
            sub_pool_name="full",
            device=device,
            is_id_owner=True,
            page_size=page_size,
            shards_under_dcp=True,
            need_sort=need_sort,
            forward_stream=forward_stream,
            lazy_compaction=lazy_compaction,
        )
        self.mamba_allocator = MultiEndedAllocator(
            kvcache=kvcache.mamba_pool,
            unified_buffer=unified_buffer,
            sub_pool_name="mamba",
            device=device,
            is_id_owner=True,
            page_size=1,  # Mamba state stays slot-granular (1-per-req)
            need_sort=need_sort,
            forward_stream=forward_stream,
            lazy_compaction=lazy_compaction,
        )
        self.full_attn_allocator.bind_peer(self.mamba_allocator)
        self.mamba_allocator.bind_peer(self.full_attn_allocator)

        # `init_unified_mamba_pools` later wraps `self.mamba_allocator` in a
        # `UnifiedMambaSlotAllocator` owning the v2p translate; the KV pools get no
        # allocator (write locations resolve in the attention metadata).

        self.free_group = None
        self.free_page_reps_group: Optional[List[torch.Tensor]] = None
        # Base init left these None; we use watermark math, not free-lists.
        self.free_pages = torch.empty(0, dtype=torch.int64, device=device)
        self.release_pages = torch.empty(0, dtype=torch.int64, device=device)

        logger.info(
            "[unified-memory-pool] UnifiedMambaTokenToKVPoolAllocator ready: "
            "full max_slots=%d (min_slot_index=%d, page_size=%d, "
            "num_pages=%d), mamba max_slots=%d (min_slot_index=%d), "
            "full_available=%d, mamba_available=%d",
            self.full_attn_allocator.max_slots,
            self.full_attn_allocator.min_slot_index,
            self.full_attn_allocator.page_size,
            self.full_attn_allocator.num_pages,
            self.mamba_allocator.max_slots,
            self.mamba_allocator.min_slot_index,
            self.full_attn_allocator.available_size(),
            self.mamba_allocator.available_size(),
        )

    # -- size: dynamic --
    @property
    def size(self) -> int:
        # TOKENS. MUST use the SAME available view as `available_size()`, so the
        # available term cancels out of the leak invariant.
        return (
            self.full_attn_allocator.schedulable_available_size()
            + self.full_attn_allocator.allocated_count()
        )

    @size.setter
    def size(self, value) -> None:
        pass  # base init writes here; computed dynamically

    # -- token-slot surface: MHA side --

    # Realizable-with-compaction view, so the retract gate / evict / schedule_policy
    # do not over-retract while the mamba peer holds drainable holes.
    def available_size(self) -> int:
        return self.full_attn_allocator.schedulable_available_size()

    def full_available_size(self) -> int:
        return self.full_attn_allocator.schedulable_available_size()

    def mamba_slot_full_token_cost(self) -> int:
        """Full-token-equivalents of shared-gap bytes ONE mamba state consumes; the
        prefill planner reserves this so admission stays inside the JOINT budget,
        rounded UP. The `dcp_size` factor is there because that budget is in widened
        tokens, one of which is `entry_bytes / dcp_size` local bytes.
        """
        return -(
            -self.mamba_allocator.entry_bytes_per_page
            * get_parallel().attn_dcp_size
            // self.full_attn_allocator.entry_bytes
        )

    @property
    def size_full(self) -> int:
        # Widened like `size`: a logical token capacity, not a row count.
        return (self.full_attn_allocator.max_slots - 1) * get_parallel().attn_dcp_size

    @property
    def draft_virtual_id_space(self) -> int:
        return self.size_full

    @property
    def size_mamba(self) -> int:
        return self.mamba_allocator.max_slots - 1

    def debug_print(self) -> str:
        return (
            f"#full-available={self.full_attn_allocator.available_size()}, "
            f"#mamba-available={self.mamba_allocator.available_size()}"
        )

    def get_kvcache(self):
        return self._kvcache

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        with record_function("UnifiedMambaAlloc.alloc"):
            return self.full_attn_allocator.alloc(need_size)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Paged extend. Mamba state is per-request (doesn't advance per-token),
        so forward only to the full sub-allocator."""
        with record_function("UnifiedMambaAlloc.alloc_extend"):
            return self.full_attn_allocator.alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
                num_new_pages=num_new_pages,
            )

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Paged decode. Mamba side stays untouched per-decode."""
        with record_function("UnifiedMambaAlloc.alloc_decode"):
            return self.full_attn_allocator.alloc_decode(
                seq_lens, seq_lens_cpu, last_loc
            )

    def translate_kv_loc(
        self,
        loc: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full-pool virtual TOKEN ids -> physical TOKEN ids; `-1` passes through as
        `-1` (padding downstream). ``out=`` supports cuda-graph buffer stability."""
        result = self.full_attn_allocator.translate_kv_loc(loc, out=out)
        return result

    @property
    def kernel_page_multiplier(self) -> int:
        return self.full_attn_allocator.kernel_page_multiplier

    @property
    def full_v2p_page_table(self) -> torch.Tensor:
        """Page-level virtual->physical table of the full sub-pool. Kernels that
        build the MLA block table straight from req_to_token gather through this,
        then scale by `kernel_page_multiplier` to reach the per-page block."""
        return self.full_attn_allocator.virtual_to_physical

    @property
    def full_p2v_page_table(self) -> torch.Tensor:
        """Page-level physical->virtual table of the full sub-pool."""
        return self.full_attn_allocator.physical_to_virtual

    def translate_kv_loc_for_kernel(
        self,
        loc: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full-pool virtual TOKEN ids -> kernel-facing ids."""
        return self.full_attn_allocator.translate_kv_loc_for_kernel(loc, out=out)

    def translate_write_loc_for_kernel(
        self,
        loc: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Widened virtual WRITE loc -> DENSE id; see the sub-allocator's copy."""
        return self.full_attn_allocator.translate_write_loc_for_kernel(loc, out=out)

    def translate_kv_indices_for_transfer(
        self, kv_indices: torch.Tensor
    ) -> torch.Tensor:
        """Virtual TOKEN ids -> PHYSICAL token ids for the PD transfer engine.
        PHYSICAL, not kernel-facing: the transfer registers page ENVELOPES (see
        `UnifiedMLATokenToKVPool.get_contiguous_buf_infos`)."""
        # Defensive: `_validate_unified_memory_dcp` rejects this pairing at
        # argument validation, so reaching it means a config path got past that.
        assert get_parallel().attn_dcp_size == 1, (
            "PD-disaggregation transfer with the unified memory pool does not "
            "support decode context parallelism: the transfer ships whole page "
            "envelopes, which hold only this rank's shard of each widened page."
        )
        return self.full_attn_allocator.translate_kv_loc(kv_indices.to(torch.int64))

    def set_disagg_move_gate(self, gate: Callable[[], bool]) -> None:
        """Install the PD-disaggregation move gate on both sub-allocators."""
        assert self.lazy_compaction, (
            "PD disaggregation with the unified memory pool requires lazy "
            "compaction (eager free-path compaction moves pages under "
            "in-flight transfers)."
        )
        self.full_attn_allocator.disagg_move_gate = gate
        self.mamba_allocator.disagg_move_gate = gate

    def is_slot_allocated(self, slot: int) -> bool:
        return self.full_attn_allocator.is_slot_allocated(slot)

    def allocator_state_str(self) -> str:
        return self.full_attn_allocator.allocator_state_str()

    def free(self, free_index: torch.Tensor) -> None:
        with record_function("UnifiedMambaAlloc.free"):
            if free_index is None or free_index.numel() == 0:
                return
            if self.free_group is not None:
                self.free_group.append(self._copy_for_free_group(free_index))
                return
            self.full_attn_allocator.free(free_index)
            self.full_attn_allocator.clear_inverse_history()
            self.mamba_allocator.clear_inverse_history()

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int) -> None:
        """Fixed-shape counterpart of `free()`; see `MultiEndedAllocator._page_reps`.
        The mamba sub-pool is slot-granular and untouched by a token free."""
        if free_index is None or free_index.numel() == 0:
            return
        if self.page_size == 1:
            self.free(free_index)
            return
        reps = self.full_attn_allocator._page_reps(
            free_index.detach().to(torch.int64), start_pos
        )
        if self.free_page_reps_group is None:
            self._release_page_reps((reps,))
        else:
            self.free_page_reps_group.append(reps)

    def _release_page_reps(self, pieces: Sequence[torch.Tensor]) -> None:
        reps = pieces[0] if len(pieces) == 1 else torch.cat(tuple(pieces))
        self.full_attn_allocator.free(reps, _pages=reps // self.page_size)
        self.full_attn_allocator.clear_inverse_history()
        self.mamba_allocator.clear_inverse_history()

    def verify_byte_accounting(self) -> List[str]:
        return _chain_byte_accounting_violations(
            _end_pair_chain(self.mamba_allocator, self.full_attn_allocator)
        )

    def free_group_begin(self) -> None:
        super().free_group_begin()
        self.free_page_reps_group = []

    def free_group_end(self) -> None:
        pending, self.free_page_reps_group = self.free_page_reps_group, None
        super().free_group_end()
        if pending:
            self._release_page_reps(pending)

    def clear(self) -> None:
        self.full_attn_allocator.clear()
        self.mamba_allocator.clear()
        self.free_group = None
        self.free_page_reps_group = None

    # -- Lazy compaction hooks --

    def set_latest_forward_done_event(self, event: Optional[torch.cuda.Event]) -> None:
        """Forward the per-batch `forward_done` event to BOTH sub-allocators."""
        with record_function("UnifiedMambaAlloc.set_latest_forward_done_event"):
            self.full_attn_allocator.set_latest_forward_done_event(event)
            self.mamba_allocator.set_latest_forward_done_event(event)

    def set_inflight_forward(
        self,
        forward_done: torch.cuda.Event,
        out_cache_loc_virtual: Optional[torch.Tensor],
    ) -> None:
        """Hand the forward's metadata to BOTH sub-pools; the mamba state is written
        by mamba kernels, not `set_kv_buffer`, so its write-set is `None`."""
        with record_function("UnifiedMambaAlloc.set_inflight_forward"):
            self.full_attn_allocator.set_inflight_forward(
                forward_done, out_cache_loc_virtual
            )
            self.mamba_allocator.set_inflight_forward(forward_done, None)

    def flush_opportunistic(self) -> int:
        """Non-urgent flush of BOTH sub-allocators; sync-free."""
        with record_function("UnifiedMambaAlloc.flush_opportunistic"):
            fa = self.full_attn_allocator
            ma = self.mamba_allocator
            if (
                fa._free_phys_pages.numel() == 0
                and not fa._pending_reuse
                and ma._free_phys_pages.numel() == 0
                and not ma._pending_reuse
            ):
                return 0
            return fa.flush_opportunistic() + ma.flush_opportunistic()
