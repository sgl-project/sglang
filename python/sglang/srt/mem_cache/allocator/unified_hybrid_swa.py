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
from sglang.srt.mem_cache.allocator.unified_side import (
    VirtualFullKVPoolSide,
    VirtualSWAKVPoolSide,
)
from sglang.srt.mem_cache.allocator.unified_sub_pool import (
    FloatMultiEndedKVAllocator,
    MultiEndedKVAllocator,
    _chain_byte_accounting_violations,
    _end_pair_chain,
    _float_open_short_side,
    _relieve_for_alloc,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_memory_pool import UnifiedKVPool
from sglang.srt.utils.common import get_num_new_pages

logger = logging.getLogger(__name__)


class UnifiedHybridSWAKVAllocator(BaseHybridSWAKVAllocator):
    """Composite allocator for the hybrid SWA pair (full + swa MHA sub-pools).

    The two sides are MultiEndedKVAllocator sub-pools sharing one virtual id
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

        full_pool = MultiEndedKVAllocator(
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
        self._wire_peers()

        # Epoch-keyed memo for the joint capacity view (any chain member's
        # mutation invalidates -- see `MultiEndedKVAllocator._chain_capacity_epoch`).
        self._joint_avail_memo_epoch: Optional[int] = None
        self._joint_avail_memo_tokens: int = 0

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

    def _build_swa_pool(self, **kwargs) -> MultiEndedKVAllocator:
        """The swa sub-allocator: an END pool in the 2-pool pair."""
        return MultiEndedKVAllocator(
            sub_pool_name="swa",
            is_id_owner=False,  # non-owner; consumes virtuals minted by full
            **kwargs,
        )

    def _wire_peers(self) -> None:
        self.full.pool.bind_peer(self.swa.pool)
        self.swa.pool.bind_peer(self.full.pool)

    # -- capacity reporting (three-way split) --

    def available_size(self) -> int:
        """Tokens available for `alloc(N)` / `alloc_extend(N)` (TOKENS)."""
        epoch = self.full.pool._chain_capacity_epoch()
        if self._joint_avail_memo_epoch != epoch:
            self._joint_avail_memo_tokens = self._compute_available_size()
            self._joint_avail_memo_epoch = epoch
        return self._joint_avail_memo_tokens

    def _compute_available_size(self) -> int:
        """Joint byte budget in TOKENS: each composite alloc(1) consumes one
        full-side AND one swa-side page under the same virtual id."""
        fa, sa = self.full.pool, self.swa.pool
        e_f = fa.entry_bytes_per_page
        e_s = sa.entry_bytes_per_page
        # Direction-agnostic shared gap: the free byte band between the two pools.
        if fa.grow_direction == "up":
            gap_bytes = max(0, sa._byte_low_frontier() - fa._byte_high_frontier())
        else:
            gap_bytes = max(0, fa._byte_low_frontier() - sa._byte_high_frontier())
        R_f = fa.num_pages - fa.min_page_index - fa._allocated_pages()
        R_s = sa.num_pages - sa.min_page_index - sa._allocated_pages()

        if not self.lazy_compaction:
            pages_by_bytes = gap_bytes // (e_f + e_s)
            return min(pages_by_bytes, R_f, R_s) * self.page_size

        H_f = len(fa._free_phys_pages)
        H_s = len(sa._free_phys_pages)

        K1 = min(H_f, H_s)  # Phase 1: both drain

        # Phase 2: fewer-holes side extends; more-holes side keeps draining.
        if H_f <= H_s:
            e_phase2 = e_f
            K_phase2_max = H_s
        else:
            e_phase2 = e_s
            K_phase2_max = H_f
        K2_room = K_phase2_max - K1
        K2 = min(K2_room, gap_bytes // e_phase2) if e_phase2 > 0 else K2_room
        gap_bytes -= K2 * e_phase2

        K3 = gap_bytes // (e_f + e_s)  # Phase 3: both extend

        K_total = K1 + K2 + K3
        K_total = min(K_total, H_f + R_f, H_s + R_s)  # index-space caps
        return K_total * self.page_size

    def _flush_targets(self):
        """Flush ALL members, including ones that are not short themselves: a
        one-sided hole is unusable, and compacting it yields SHARED gap."""
        return (self.full.pool, self.swa.pool)

    def _ask_float_for_room(self, need_tokens: int) -> None:
        """No float in a two-END chain -- nothing can slide."""
        return None

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
                if not _relieve_for_alloc(self, need_size):
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
                if not _relieve_for_alloc(self, need_tokens):
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
                if not _relieve_for_alloc(self, need_tokens):
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
        return (
            _chain_byte_accounting_violations(
                _end_pair_chain(self.full.pool, self.swa.pool)
            )
            + self._joint_capacity_memo_violations()
        )

    def _joint_capacity_memo_violations(self) -> List[str]:
        """Idle-time twin of `MultiEndedKVAllocator._capacity_memo_violations`
        for the composite joint view. Empty == healthy."""
        if self._joint_avail_memo_epoch != self.full.pool._chain_capacity_epoch():
            return []
        actual = self._compute_available_size()
        if self._joint_avail_memo_tokens == actual:
            return []
        return [
            f"[joint] stale available_size memo: "
            f"cached={self._joint_avail_memo_tokens}, actual={actual}"
        ]

    def clear(self) -> None:
        self.full.pool.clear()
        self.swa.pool.clear()

    # -- Lazy compaction hooks --

    def set_latest_forward_done_event(self, event: Optional[torch.cuda.Event]) -> None:
        """Forward the per-batch `forward_done` event to BOTH sub-allocators."""
        with record_function("UnifiedSWAAlloc.set_latest_forward_done_event"):
            self.full.pool.set_latest_forward_done_event(event)
            self.swa.pool.set_latest_forward_done_event(event)

    def set_inflight_forward(
        self,
        forward_done: torch.cuda.Event,
        out_cache_loc_virtual: Optional[torch.Tensor],
    ) -> None:
        """Hand the forward's metadata to BOTH sub-pools; each materializes its own
        write-set via its OWN v2p, and the forward writes both sides per token."""
        with record_function("UnifiedSWAAlloc.set_inflight_forward"):
            self.full.pool.set_inflight_forward(forward_done, out_cache_loc_virtual)
            self.swa.pool.set_inflight_forward(forward_done, out_cache_loc_virtual)

    def flush_opportunistic(self) -> int:
        """Non-urgent flush of BOTH sub-allocators; sync-free."""
        with record_function("UnifiedSWAAlloc.flush_opportunistic"):
            fa = self.full.pool
            sa = self.swa.pool
            if (
                fa._free_phys_pages.numel() == 0
                and not fa._pending_reuse
                and sa._free_phys_pages.numel() == 0
                and not sa._pending_reuse
            ):
                return 0
            return fa.flush_opportunistic() + sa.flush_opportunistic()


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
        # Per-request state END pool (grow-up; page_size=1 -- state is
        # per-request, orthogonal to KV paging).
        self.mamba_allocator = MultiEndedKVAllocator(
            kvcache=mamba_kvcache,
            unified_buffer=unified_buffer,
            sub_pool_name="mamba",
            device=device,
            is_id_owner=True,
            page_size=1,
            need_sort=need_sort,
            forward_stream=forward_stream,
            lazy_compaction=lazy_compaction,
        )
        # Chain wiring: mamba <-> swa(float) <-> full.
        self.mamba_allocator.bind_high_peer(self.swa.pool)
        self.swa.pool.bind_low_peer(self.mamba_allocator)
        self.swa.pool.bind_high_peer(self.full.pool)
        self.full.pool.bind_low_peer(self.swa.pool)

        logger.info(
            "[unified-memory-pool] UnifiedMambaHybridSWAKVAllocator ready: "
            "chain=[mamba(up) | swa(float) | full(down)], "
            "mamba max_slots=%d (entry_bytes=%d), joint available=%d",
            self.mamba_allocator.max_slots,
            self.mamba_allocator.entry_bytes,
            self.available_size(),
        )

    # -- construction hooks --

    def _build_swa_pool(self, **kwargs) -> MultiEndedKVAllocator:
        # The swa side is the FLOAT middle: it never runs the lazy event pipeline
        # regardless of the composite's flag (frees mark holes, allocs reuse them).
        kwargs["lazy_compaction"] = False
        return FloatMultiEndedKVAllocator(
            sub_pool_name="swa",
            is_id_owner=False,  # non-owner; consumes virtuals minted by full
            **kwargs,
        )

    def _wire_peers(self) -> None:
        # Chain wired in __init__ once the mamba end exists.
        return

    # -- capacity --

    def _compute_available_size(self) -> int:
        """Joint TOKENS for `alloc(N)`: N costs N full pages AND N swa pages, drawn
        from DIFFERENT bands -- full extends only into the high band, the float into
        either side but only ONE per batch alloc. Feasibility is monotone in N, so
        binary search; the order matches the alloc path (full takes the high band).
        """
        fa, sa = self.full.pool, self.swa.pool
        e_f, e_s = fa.entry_bytes_per_page, sa.entry_bytes_per_page
        # full is grow-down: its chain gap IS the high band.
        b_high = fa._current_gap_bytes()
        if sa._is_frontier_transparent():
            b_low = 0
        else:
            b_low = max(
                0,
                sa._byte_low_frontier() - sa._chain_high_frontier_below_bytes(),
            )
        h_f = len(fa._free_phys_pages) if fa.lazy_compaction else 0
        h_s = sa._hole_pages()
        r_f = fa.num_pages - fa.min_page_index - fa._allocated_pages()
        r_s = sa.num_pages - sa.min_page_index - sa._allocated_pages()

        def feasible(n: int) -> bool:
            if n > h_f + r_f or n > h_s + r_s:
                return False
            ext_f = max(0, n - h_f)
            if ext_f * e_f > b_high:
                return False
            ext_s = max(0, n - h_s)
            # On the float's page grid, never in raw bytes: a byte budget
            # credits a page `take_physical_pages` cannot yield.
            full_low_after = fa._byte_low_frontier() - ext_f * e_f
            if sa._is_frontier_transparent():
                room = sa.pages_in_band(
                    low_byte=sa._chain_high_frontier_below_bytes(),
                    high_byte=full_low_after,
                )
                return ext_s <= room
            p_low = sa.pages_in_band(
                low_byte=sa._chain_high_frontier_below_bytes(),
                high_byte=sa._byte_low_frontier(),
            )
            p_high = sa.pages_in_band(
                low_byte=sa._byte_high_frontier(),
                high_byte=full_low_after,
            )
            return ext_s <= max(p_low, p_high)

        lo_n, hi_n = 0, min(h_f + r_f, h_s + r_s)
        while lo_n < hi_n:
            mid = (lo_n + hi_n + 1) // 2
            if feasible(mid):
                lo_n = mid
            else:
                hi_n = mid - 1
        return lo_n * self.page_size

    def _flush_targets(self):
        """All three members, float FIRST: its zero-copy boundary absorption must
        land before the deficit math prices a relocation it already covered."""
        return (
            self.swa.pool,
            self.full.pool,
            self.mamba_allocator,
        )

    def _alloc_demand(self, need_tokens: int):
        """Demand VECTOR for one composite allocation, in PAGES per band. A token
        never draws a state slot, so mamba is an explicit 0, not an omission."""
        need_n = -(-need_tokens // self.page_size)
        return {
            self.full.pool: need_n,
            self.swa.pool: need_n,
            self.mamba_allocator: 0,
        }

    def _ask_float_for_room(self, need_tokens: int) -> None:
        """Composite shortfall: hand the demand vector to the shared policy;
        the float is whichever demanded band floats."""
        demand = self._alloc_demand(need_tokens)
        flt = None
        for b in demand:
            if isinstance(b, FloatMultiEndedKVAllocator):
                flt = b
        _float_open_short_side(flt, demand)

    def mamba_slot_full_token_cost(self) -> int:
        """Full-token-equivalents one mamba/conv slot removes from the shared buffer:
        a tri-pool token costs e_f + e_s bytes, and the quotient is rounded UP."""
        e_tok = self.full.pool.entry_bytes + self.swa.pool.entry_bytes
        return -(-self.mamba_allocator.entry_bytes_per_page // e_tok)

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

    def set_latest_forward_done_event(self, event: Optional[torch.cuda.Event]) -> None:
        super().set_latest_forward_done_event(event)
        self.mamba_allocator.set_latest_forward_done_event(event)

    def set_inflight_forward(
        self,
        forward_done: torch.cuda.Event,
        out_cache_loc_virtual: Optional[torch.Tensor],
    ) -> None:
        # The mamba state is written by the conv kernels, not through
        # `out_cache_loc`, so its in-flight write-set is None.
        super().set_inflight_forward(forward_done, out_cache_loc_virtual)
        self.mamba_allocator.set_inflight_forward(forward_done, None)

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

    def verify_byte_accounting(self) -> List[str]:
        return (
            _chain_byte_accounting_violations(
                [
                    self.mamba_allocator,
                    self.swa.pool,
                    self.full.pool,
                ]
            )
            + self._joint_capacity_memo_violations()
        )

    def flush_opportunistic(self) -> int:
        """Per-step reclaim across the whole chain. The float participates for its
        deferred boundary absorption, which is where its single D2H is paid."""
        fa, ma = self.full.pool, self.mamba_allocator
        sa = self.swa.pool
        if (
            fa._free_phys_pages.numel() == 0
            and not fa._pending_reuse
            and ma._free_phys_pages.numel() == 0
            and not ma._pending_reuse
            and sa._free_phys_pages.numel() == 0
        ):
            return 0
        return (
            fa.flush_opportunistic()
            + ma.flush_opportunistic()
            + sa.flush_opportunistic()
        )
