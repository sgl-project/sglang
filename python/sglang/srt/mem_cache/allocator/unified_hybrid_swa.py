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
"""Unified-memory composites for hybrid SWA models: the full-attention and SWA
sub-pools of one `UnifiedKVPool`, and the tri-pool variant that adds mamba state."""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
from torch.profiler import record_function

from sglang.srt.mem_cache.allocator.hybrid import BaseHybridSWAKVAllocator
from sglang.srt.mem_cache.allocator.pairing import VirtualIdPairing
from sglang.srt.mem_cache.allocator.unified_chain import UnifiedChain
from sglang.srt.mem_cache.allocator.unified_side import (
    VirtualFullKVPoolSide,
    VirtualSWAKVPoolSide,
)
from sglang.srt.mem_cache.allocator.unified_sub_pool import (
    FloatMultiEndedKVPool,
    MultiEndedKVPool,
    _end_pair_chain,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_memory_pool import UnifiedKVPool
from sglang.srt.utils.common import get_num_new_pages

logger = logging.getLogger(__name__)


class UnifiedHybridSWAKVAllocator(BaseHybridSWAKVAllocator):
    """Composite allocator for the hybrid SWA pair (full + swa MHA sub-pools).

    The two sides are MultiEndedKVPool sub-pools sharing one virtual id
    space; the swa side's v2p table is the pairing, so there is no full -> swa
    mapping tensor. One alloc(N) binds N pages on BOTH sides under the same
    virtual id, so `available_size()` (joint bytes, in TOKENS) is the only safe
    alloc pre-check.
    """

    @property
    def size(self) -> int:
        return min(self._size_full, self._size_swa)

    # Static caps, not `max_slots - 1` (~= full_max + swa_max, which over-promises).
    @property
    def size_full(self) -> int:
        return self._size_full

    @property
    def size_swa(self) -> int:
        return self._size_swa

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        kvcache,  # UnifiedSWAKVPool
        device: str,
        full_max_total_num_tokens: int,
        swa_max_total_num_tokens: int,
        page_size: int = 1,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
        lazy_compaction: bool = False,
    ):
        # Set _size_full / _size_swa BEFORE base init (read during it). STATIC
        # partition caps -- the slot-conservation value the leak invariant expects.
        self._size_full = full_max_total_num_tokens
        self._size_swa = swa_max_total_num_tokens
        self._full_max_total_num_tokens = full_max_total_num_tokens
        self._swa_max_total_num_tokens = swa_max_total_num_tokens
        self.page_size = page_size
        self.dtype = unified_buffer.mha_spec("full").store_dtype
        self.device = device
        self.need_sort = need_sort
        self.unified_buffer = unified_buffer
        self._kvcache = kvcache
        self.lazy_compaction = lazy_compaction

        full_pool = MultiEndedKVPool(
            kvcache=kvcache.full_kv_pool,
            unified_buffer=unified_buffer,
            sub_pool_name="full",
            device=device,
            is_id_owner=True,
            page_size=page_size,
            need_sort=need_sort,
            forward_stream=forward_stream,
            lazy_compaction=lazy_compaction,
        )
        swa_pool = self._build_swa_pool(
            kvcache=kvcache.swa_kv_pool,
            unified_buffer=unified_buffer,
            device=device,
            page_size=page_size,
            need_sort=need_sort,
            forward_stream=forward_stream,
            lazy_compaction=lazy_compaction,
            # swa binds the virtual pages full mints, so it must address
            # full's whole id space.
            virtual_num_pages=full_pool.num_virtual_ids,
        )
        self.sides = {
            ComponentType.SWA: VirtualSWAKVPoolSide(
                swa_pool, conserve_cap=swa_max_total_num_tokens
            ),
            ComponentType.FULL: VirtualFullKVPoolSide(
                full_pool, conserve_cap=full_max_total_num_tokens
            ),
        }
        self.pairing = VirtualIdPairing(swa_pool)
        self.chain = self._build_chain(full_pool, swa_pool)

        # The full/SWA KV pools need no allocator wiring (write locations resolved
        # in attention metadata); the composite keeps allocators for read-path translates.
        kvcache.attach_allocators(
            full_allocator=self.full.pool,
            swa_allocator=self.swa.pool,
        )

        logger.info(
            "[unified-memory-pool] UnifiedHybridSWAKVAllocator ready: "
            "full max_slots=%d (min_slot_index=%d, entry_bytes=%d), "
            "swa max_slots=%d (min_slot_index=%d, entry_bytes=%d), "
            "static caps full=%d swa=%d, joint available=%d",
            self.full.pool.max_slots,
            self.full.pool.min_slot_index,
            self.full.pool.entry_bytes,
            self.swa.pool.max_slots,
            self.swa.pool.min_slot_index,
            self.swa.pool.entry_bytes,
            self._full_max_total_num_tokens,
            self._swa_max_total_num_tokens,
            self.available_size(),
        )

    # -- construction hooks (the tri-pool subclass overrides both) --

    def _build_swa_pool(self, **kwargs) -> MultiEndedKVPool:
        """The swa sub-allocator: an END pool in the 2-pool pair."""
        return MultiEndedKVPool(
            sub_pool_name="swa",
            is_id_owner=False,  # non-owner; consumes virtuals minted by full
            **kwargs,
        )

    def _build_chain(
        self, full_pool: MultiEndedKVPool, swa_pool: MultiEndedKVPool
    ) -> UnifiedChain:
        """Two END pools facing each other across the shared gap."""
        return UnifiedChain(
            _end_pair_chain(full_pool, swa_pool),
            token_members=(full_pool, swa_pool),
            lazy_compaction=self.lazy_compaction,
        )

    # -- capacity reporting (three-way split) --

    def available_size(self) -> int:
        """Tokens available for `alloc(N)` / `alloc_extend(N)`: the chain's joint
        byte budget, since one alloc binds a page on both sides."""
        return self.chain.joint_available_tokens()

    @property
    def draft_virtual_id_space(self) -> int:
        return self.full.pool.max_slots - 1

    def debug_print(self) -> str:
        return (
            f"#full-available={self.full.pool.available_size()}, "
            f"#swa-available={self.swa.pool.available_size()}, "
            f"#joint-available={self.available_size()}"
        )

    def get_kvcache(self):
        return self._kvcache

    def translate_kv_loc(
        self,
        loc: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full-layer read path: virtual TOKEN ids -> full-physical TOKEN ids.
        ``out=`` writes in place, for cuda-graph buffer stability."""
        result = self.full.pool.translate_kv_loc(loc, out=out)
        return result

    def translate_loc_from_full_to_swa(
        self,
        kv_indices: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """SWA-layer read path: virtual TOKEN ids -> swa kernel-facing ids."""
        return self.swa.pool.translate_kv_loc_for_kernel(kv_indices, out=out)

    @property
    def kernel_page_multiplier(self) -> int:
        return self.full.pool.kernel_page_multiplier

    @property
    def full_v2p_page_table(self) -> torch.Tensor:
        """Page-level virtual->physical table of the full sub-pool."""
        return self.full.pool.virtual_to_physical

    @property
    def full_p2v_page_table(self) -> torch.Tensor:
        """Page-level physical->virtual table of the full sub-pool."""
        return self.full.pool.physical_to_virtual

    def translate_kv_loc_for_kernel(
        self,
        loc: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full-pool virtual TOKEN ids -> kernel-facing ids."""
        return self.full.pool.translate_kv_loc_for_kernel(loc, out=out)

    def translate_write_loc_for_kernel(
        self,
        loc: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Widened virtual WRITE loc -> kernel-facing id. DCP is rejected for this
        composite at argument validation, so it coincides with the read translate."""
        return self.full.pool.translate_write_loc_for_kernel(loc, out=out)

    @property
    def swa_kernel_page_multiplier(self) -> int:
        return self.swa.pool.kernel_page_multiplier

    @property
    def swa_v2p_page_table(self) -> torch.Tensor:
        """Page-level virtual->physical table of the SWA sub-pool."""
        return self.swa.pool.virtual_to_physical

    # -- alloc --

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        with record_function("UnifiedSWAAlloc.alloc"):
            # Joint pre-check. Both sides are mutual peers (each side's compaction
            # opens gap for the other), so flush BOTH on shortfall.
            if need_size > self.available_size():
                if not self.chain.relieve(need_size):
                    return None
            # Snapshot the virtual PAGES full will consume, to bind them on swa too.
            num_pages = need_size // self.page_size
            fa = self.full.pool
            new_virtual_pages = fa.free_virtual_ids[:num_pages].clone()

            v_tokens = fa.alloc(need_size)
            # Post-pre-check failure can only be internal-state inconsistency.
            assert v_tokens is not None, (
                "UnifiedSWA.alloc: full.alloc returned None after joint "
                "pre-check passed — internal-state inconsistency"
            )
            self.swa.pool.alloc_with_virtual(new_virtual_pages)
            return v_tokens

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ) -> Optional[torch.Tensor]:
        """Paged extend; returns virtual TOKEN ids. The same virtual page maps to
        full- and swa-physical, so swa binds exactly what the full kernel consumed."""
        with record_function("UnifiedSWAAlloc.alloc_extend"):
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
            need_tokens = num_new_pages * self.page_size
            if need_tokens > self.available_size():
                if not self.chain.relieve(need_tokens):
                    return None

            # Snapshot the virtual PAGES the kernel will consume; clone so swa keeps
            # its view after the slice is consumed.
            fa = self.full.pool
            new_virtual_pages = fa.free_virtual_ids[:num_new_pages].clone()

            out_indices = fa.alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
                num_new_pages=num_new_pages,
            )
            assert out_indices is not None, (
                "UnifiedSWA.alloc_extend: full.alloc_extend returned None "
                "after joint pre-check passed — internal-state inconsistency"
            )
            self.swa.pool.alloc_with_virtual(new_virtual_pages)
            return out_indices  # virtual TOKEN ids

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Paged decode: one new token per request, consuming a page only when the
        decode wraps."""
        with record_function("UnifiedSWAAlloc.alloc_decode"):
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
            )
            need_tokens = num_new_pages * self.page_size
            if need_tokens > self.available_size():
                if not self.chain.relieve(need_tokens):
                    return None

            fa = self.full.pool
            new_virtual_pages = fa.free_virtual_ids[:num_new_pages].clone()

            out_indices = fa.alloc_decode(seq_lens, seq_lens_cpu, last_loc)
            assert out_indices is not None, (
                "UnifiedSWA.alloc_decode: full.alloc_decode returned None "
                "after joint pre-check passed — internal-state inconsistency"
            )

            if new_virtual_pages.numel() > 0:
                self.swa.pool.alloc_with_virtual(new_virtual_pages)

            return out_indices  # virtual TOKEN ids

    def is_slot_allocated(self, slot: int) -> bool:
        """Token-slot surface = the full side (which owns the virtual ids)."""
        return self.full.pool.is_slot_allocated(slot)

    def allocator_state_str(self) -> str:
        return self.full.pool.allocator_state_str()

    def verify_byte_accounting(self) -> List[str]:
        return self.chain.verify_byte_accounting()

    def clear(self) -> None:
        self.full.pool.clear()
        self.swa.pool.clear()

    # -- Lazy compaction hooks --


class UnifiedMambaHybridSWAKVAllocator(UnifiedHybridSWAKVAllocator):
    """Tri-pool composite for models with full KV + SWA KV + mamba/conv state
    (both `mambaish_config` and `is_hybrid_swa`).

    Chain (low byte -> high byte):

        [ mamba/conv (grow-up END) | swa (FLOAT middle) | full (grow-down END) ]

    The ends never relocate, so they take the per-request state pool and the
    unbounded per-step grower; SWA is window-capped with the cheapest slots to
    move, and its out-of-window tombstones become float holes recycled in place.
    Per-request state is served through `mamba_allocator`, wrapped by
    `UnifiedMambaSlotAllocator`.
    """

    def __init__(
        self,
        *,
        unified_buffer: UnifiedKVPool,
        kvcache,  # UnifiedSWAKVPool
        mamba_kvcache,  # UnifiedMambaPool (req_to_token_pool.mamba_pool)
        device: str,
        full_max_total_num_tokens: int,
        swa_max_total_num_tokens: int,
        page_size: int = 1,
        need_sort: bool = False,
        forward_stream: Optional[torch.cuda.Stream] = None,
        lazy_compaction: bool = False,
    ):
        # Read by `_build_chain`, which the base init calls once the pair exists.
        self._mamba_kvcache = mamba_kvcache
        self._forward_stream = forward_stream
        super().__init__(
            unified_buffer=unified_buffer,
            kvcache=kvcache,
            device=device,
            full_max_total_num_tokens=full_max_total_num_tokens,
            swa_max_total_num_tokens=swa_max_total_num_tokens,
            page_size=page_size,
            need_sort=need_sort,
            forward_stream=forward_stream,
            lazy_compaction=lazy_compaction,
        )
        logger.info(
            "[unified-memory-pool] UnifiedMambaHybridSWAKVAllocator ready: "
            "chain=[mamba(up) | swa(float) | full(down)], "
            "mamba max_slots=%d (entry_bytes=%d), joint available=%d",
            self.mamba_allocator.max_slots,
            self.mamba_allocator.entry_bytes,
            self.available_size(),
        )

    # -- construction hooks --

    def _build_swa_pool(self, **kwargs) -> MultiEndedKVPool:
        # The swa side is the FLOAT middle: it never runs the lazy event pipeline
        # regardless of the composite's flag (frees mark holes, allocs reuse them).
        kwargs["lazy_compaction"] = False
        return FloatMultiEndedKVPool(
            sub_pool_name="swa",
            is_id_owner=False,  # non-owner; consumes virtuals minted by full
            **kwargs,
        )

    def _build_chain(
        self, full_pool: MultiEndedKVPool, swa_pool: MultiEndedKVPool
    ) -> UnifiedChain:
        """mamba (grow-up END) | swa (FLOAT) | full (grow-down END)."""
        # Per-request state END pool (page_size=1: state is per request,
        # orthogonal to KV paging).
        self.mamba_allocator = MultiEndedKVPool(
            kvcache=self._mamba_kvcache,
            unified_buffer=self.unified_buffer,
            sub_pool_name="mamba",
            device=self.device,
            is_id_owner=True,
            page_size=1,
            need_sort=self.need_sort,
            forward_stream=self._forward_stream,
            lazy_compaction=self.lazy_compaction,
        )
        return UnifiedChain(
            (self.mamba_allocator, swa_pool, full_pool),
            token_members=(full_pool, swa_pool),
            state_member=self.mamba_allocator,
            lazy_compaction=self.lazy_compaction,
        )

    # -- capacity --

    def debug_print(self) -> str:
        sa = self.swa.pool
        return (
            super().debug_print()
            + f", #mamba-available={self.mamba_allocator.available_size()}"
            + f", swa-float span=[{sa.low_wm_page},{sa.high_wm_page}) "
            + f"holes={sa._hole_pages()}"
        )

    # -- lifecycle fanout (adds the mamba end) --

    def clear(self) -> None:
        super().clear()
        self.mamba_allocator.clear()

    def evict_to_free_tokens(self, tree_cache, num_tokens: int) -> None:
        """Joint-aware eviction: one tri-lifetime node frees bytes on several sides
        at once, so re-check the JOINT gate instead of the per-side shortfall."""
        from sglang.srt.mem_cache.common import evict_from_tree_cache

        # Arbitrary retry bound; a round that frees nothing ends the loop anyway.
        for _ in range(4):
            before = self.available_size()
            if before >= num_tokens:
                return
            evict_from_tree_cache(tree_cache, num_tokens)
            if self.available_size() <= before:
                return  # no progress
