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
from typing import List, Optional, Sequence

import torch
from torch.profiler import record_function

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.unified_sub_pool import (
    FloatMultiEndedAllocator,
    MultiEndedAllocator,
    _chain_byte_accounting_violations,
    _end_pair_chain,
    _float_open_short_side,
    _relieve_for_alloc,
)
from sglang.srt.mem_cache.unified_memory_pool import UnifiedKVPool
from sglang.srt.utils.common import get_num_new_pages

logger = logging.getLogger(__name__)


class UnifiedSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    """Composite allocator for the hybrid SWA pair (full + swa MHA sub-pools).

    Inherits from `SWATokenToKVPoolAllocator` only for the isinstance contract;
    we call grand-parent `BaseTokenToKVPoolAllocator.__init__` directly to skip
    the parent's static-partition sub-pool allocation (which unified-memory-pool
    replaces).

    Capacity views:
    - `available_size()`: joint byte-budget, the only safe `alloc(N)` pre-check
      (N slots cost N*(entry_full + entry_swa) shared-gap bytes).
    - `_conserve_*`: slot-conservation, for the LEAK invariant only.
    - `schedulable_*`: byte-coordinated, realizable-with-compaction.
    - `full_available_size()` / `swa_available_size()`: per-side scheduler view
      = min(conserve, schedulable).
    """

    # Parent's `size` property has no setter but base init does `self.size = size`;
    # override with a no-op setter. Reading returns `min(_size_full, _size_swa)`.
    @property
    def size(self) -> int:
        return min(self._size_full, self._size_swa)

    @size.setter
    def size(self, value) -> None:
        pass

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
        # partition caps — the slot-conservation value the leak invariant expects.
        self._size_full = full_max_total_num_tokens
        self._size_swa = swa_max_total_num_tokens
        self._full_max_total_num_tokens = full_max_total_num_tokens
        self._swa_max_total_num_tokens = swa_max_total_num_tokens
        self.page_size = page_size

        # Skip SWATokenToKVPoolAllocator.__init__; call grand-parent base init
        # directly (its `self.size = size` is absorbed by our no-op setter).
        BaseTokenToKVPoolAllocator.__init__(
            self,
            size=full_max_total_num_tokens,
            page_size=page_size,
            dtype=unified_buffer.mha_spec("full").store_dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.unified_buffer = unified_buffer
        self._kvcache = kvcache
        self.lazy_compaction = lazy_compaction

        self.full_attn_allocator = MultiEndedAllocator(
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
        self.swa_attn_allocator = self._build_swa_attn_allocator(
            kvcache=kvcache.swa_kv_pool,
            unified_buffer=unified_buffer,
            device=device,
            page_size=page_size,
            need_sort=need_sort,
            forward_stream=forward_stream,
            lazy_compaction=lazy_compaction,
            # swa binds the virtual pages full mints, so it must address
            # full's whole id space.
            virtual_num_pages=self.full_attn_allocator.num_virtual_ids,
        )
        self._wire_peers()

        # Epoch-keyed memo for the joint capacity view (any chain member's
        # mutation invalidates -- see `MultiEndedAllocator._chain_capacity_epoch`).
        self._joint_avail_memo_epoch: Optional[int] = None
        self._joint_avail_memo_tokens: int = 0

        # The full/SWA KV pools need no allocator wiring (write locations resolved
        # in attention metadata); the composite keeps allocators for read-path translates.
        kvcache.attach_allocators(
            full_allocator=self.full_attn_allocator,
            swa_allocator=self.swa_attn_allocator,
        )

        self.free_group = None
        self.free_page_reps_group: Optional[List[torch.Tensor]] = None
        self.full_free_group: List[torch.Tensor] = []
        # Empty (not None) for the leak checker.
        self.free_pages = torch.empty(0, dtype=torch.int64, device=device)
        self.release_pages = torch.empty(0, dtype=torch.int64, device=device)

        logger.info(
            "[unified-memory-pool] UnifiedSWATokenToKVPoolAllocator ready: "
            "full max_slots=%d (min_slot_index=%d, entry_bytes=%d), "
            "swa max_slots=%d (min_slot_index=%d, entry_bytes=%d), "
            "static caps full=%d swa=%d, joint available=%d",
            self.full_attn_allocator.max_slots,
            self.full_attn_allocator.min_slot_index,
            self.full_attn_allocator.entry_bytes,
            self.swa_attn_allocator.max_slots,
            self.swa_attn_allocator.min_slot_index,
            self.swa_attn_allocator.entry_bytes,
            self._full_max_total_num_tokens,
            self._swa_max_total_num_tokens,
            self.available_size(),
        )

    # -- construction hooks (the tri-pool subclass overrides both) --

    def _build_swa_attn_allocator(self, **kwargs) -> MultiEndedAllocator:
        """The swa sub-allocator: an END pool here (2-pool pair); the tri-pool
        subclass overrides to build the swa FLOAT middle instead."""
        return MultiEndedAllocator(
            sub_pool_name="swa",
            is_id_owner=False,  # non-owner; consumes virtuals minted by full
            **kwargs,
        )

    def _wire_peers(self) -> None:
        """2-pool end-pair wiring; the tri-pool subclass wires the full chain
        (mamba end <-> swa float <-> full end) after its mamba end exists."""
        self.full_attn_allocator.bind_peer(self.swa_attn_allocator)
        self.swa_attn_allocator.bind_peer(self.full_attn_allocator)

    # -- capacity reporting (three-way split) --

    def available_size(self) -> int:
        """Tokens available for `alloc(N)` / `alloc_extend(N)` (TOKENS).

        Memoized on the chain capacity epoch (the compute walks every chain
        frontier; see `_compute_available_size`, which the tri-pool subclass
        overrides with its three-band variant).
        """
        epoch = self.full_attn_allocator._chain_capacity_epoch()
        if self._joint_avail_memo_epoch != epoch:
            self._joint_avail_memo_tokens = self._compute_available_size()
            self._joint_avail_memo_epoch = epoch
        return self._joint_avail_memo_tokens

    def _compute_available_size(self) -> int:
        """Joint byte-budget: each composite alloc(1) consumes one full-side AND one
        swa-side page (same virtual id). The 3-phase lazy formula consumes both
        sides' holes maximally before extending toward the gap (H_f/H_s = holes,
        e_f/e_s = bytes/page, R_f/R_s = extension room, G = byte gap):
            Phase 1 (both drain, free):     K1 = min(H_f, H_s)
            Phase 2 (fewer-holes side extends): K2 limited by remaining holes & G
            Phase 3 (both extend):          K3 = G // (e_f + e_s)
        Total capped by index-space rooms (H_f + R_f, H_s + R_s). ps==1 collapses
        to slot math. Eager has no holes → original joint formula.
        """
        fa, sa = self.full_attn_allocator, self.swa_attn_allocator
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

    # Slot-conservation views — the ONLY views the leak invariant should see
    # (returning the byte-coordinated value would flag spurious leaks).
    # `allocated_count()` is in TOKENS (the unit the leak check expects).
    def _conserve_full_available_size(self) -> int:
        return (
            self._full_max_total_num_tokens - self.full_attn_allocator.allocated_count()
        )

    def _conserve_swa_available_size(self) -> int:
        return (
            self._swa_max_total_num_tokens - self.swa_attn_allocator.allocated_count()
        )

    # PHYSICAL per-side views read by scheduling / eviction consumers. The
    # `min(...)` is sound under dynamic borrowing: the static-conserve cap bounds
    # the lending side, the byte-coordinated `schedulable_*` bounds the side that
    # has grown into the shared gap; whichever is tighter wins.
    def full_available_size(self) -> int:
        return min(
            self._conserve_full_available_size(),
            self.schedulable_full_available_size(),
        )

    def swa_available_size(self) -> int:
        return min(
            self._conserve_swa_available_size(),
            self.schedulable_swa_available_size(),
        )

    # Slot-conservation views for the LEAK INVARIANT only, which pairs the static
    # per-layer total with (static cap - live). Schedulers keep the `min(...)`
    # views above: under the floating boundary the byte term dips below the
    # conserve cap, so bytes lent to a peer sub-pool would read as a leak.
    def conserve_full_available_size(self) -> int:
        return self._conserve_full_available_size()

    def conserve_swa_available_size(self) -> int:
        return self._conserve_swa_available_size()

    # Byte-coordinated, realizable-with-compaction views (peer drainable holes
    # credited — see `MultiEndedAllocator.schedulable_available_size`).
    def schedulable_full_available_size(self) -> int:
        return self.full_attn_allocator.schedulable_available_size()

    def schedulable_swa_available_size(self) -> int:
        return self.swa_attn_allocator.schedulable_available_size()

    def _flush_targets(self):
        """A coupled alloc consumes a page on EVERY member under one virtual
        id, so a hole on ONE side is unusable once the gap is dry — there is
        nothing on the other side to pair it with. Each member's compaction
        converts such dead one-sided holes into SHARED gap, which serves the
        joint gate: flush ALL members, including ones that are themselves
        short.
        """
        return (self.full_attn_allocator, self.swa_attn_allocator)

    def _ask_float_for_room(self, need_tokens: int) -> None:
        """No float in a two-END chain -- nothing can slide."""
        return None

    # `size_full` / `size_swa` are inherited; they read `_size_full`/`_size_swa`
    # (set to the static caps). We do NOT report `max_slots - 1`: under unified
    # memory pool that ~= full_max + swa_max and would over-promise.

    @property
    def draft_virtual_id_space(self) -> int:
        return self.full_attn_allocator.max_slots - 1

    def debug_print(self) -> str:
        return (
            f"#full-available={self.full_attn_allocator.available_size()}, "
            f"#swa-available={self.swa_attn_allocator.available_size()}, "
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
        Delegates to the full-side sub-allocator. Supports ``out=`` for cuda-graph.
        """
        result = self.full_attn_allocator.translate_kv_loc(loc, out=out)
        return result

    def translate_loc_from_full_to_swa(
        self,
        kv_indices: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """SWA-layer read path: virtual TOKEN ids -> swa kernel-facing ids."""
        return self.swa_attn_allocator.translate_kv_loc_for_kernel(kv_indices, out=out)

    @property
    def kernel_page_multiplier(self) -> int:
        return self.full_attn_allocator.kernel_page_multiplier

    @property
    def full_v2p_page_table(self) -> torch.Tensor:
        """Page-level virtual->physical table of the full sub-pool."""
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
        """Widened virtual WRITE loc -> kernel-facing id; see the sub-allocator's
        copy. DCP is rejected for this composite at argument validation, so this
        is the dcp_size == 1 identity with the read translate."""
        return self.full_attn_allocator.translate_write_loc_for_kernel(loc, out=out)

    @property
    def swa_kernel_page_multiplier(self) -> int:
        return self.swa_attn_allocator.kernel_page_multiplier

    @property
    def swa_v2p_page_table(self) -> torch.Tensor:
        """Page-level virtual->physical table of the SWA sub-pool."""
        return self.swa_attn_allocator.virtual_to_physical

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
            fa = self.full_attn_allocator
            new_virtual_pages = fa.free_virtual_ids[:num_pages].clone()

            v_tokens = fa.alloc(need_size)
            # Post-pre-check failure can only be internal-state inconsistency.
            assert v_tokens is not None, (
                "UnifiedSWA.alloc: full.alloc returned None after joint "
                "pre-check passed — internal-state inconsistency"
            )
            self.swa_attn_allocator.alloc_with_virtual(new_virtual_pages)
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
        """Paged extend. Runs the kernel ONCE in virtual space, then binds the
        consumed virtual PAGES on the swa side via `alloc_with_virtual`. Returns
        virtual TOKEN ids respecting the tail-page-reuse contract and the
        cross-sub-pool identity (same virtual page maps to full- and swa-physical).
        """
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
            fa = self.full_attn_allocator
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
            self.swa_attn_allocator.alloc_with_virtual(new_virtual_pages)
            return out_indices  # virtual TOKEN ids

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Paged decode. One new token per request (a page is consumed iff the
        decode wraps). Same one-kernel-in-virtual-space discipline as ``alloc_extend``.
        """
        with record_function("UnifiedSWAAlloc.alloc_decode"):
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
            )
            need_tokens = num_new_pages * self.page_size
            if need_tokens > self.available_size():
                if not _relieve_for_alloc(self, need_tokens):
                    return None

            fa = self.full_attn_allocator
            new_virtual_pages = fa.free_virtual_ids[:num_new_pages].clone()

            out_indices = fa.alloc_decode(seq_lens, seq_lens_cpu, last_loc)
            assert out_indices is not None, (
                "UnifiedSWA.alloc_decode: full.alloc_decode returned None "
                "after joint pre-check passed — internal-state inconsistency"
            )

            if new_virtual_pages.numel() > 0:
                self.swa_attn_allocator.alloc_with_virtual(new_virtual_pages)

            return out_indices  # virtual TOKEN ids

    def is_slot_allocated(self, slot: int) -> bool:
        """Token-slot surface = the full side (which owns the virtual ids)."""
        return self.full_attn_allocator.is_slot_allocated(slot)

    def allocator_state_str(self) -> str:
        return self.full_attn_allocator.allocator_state_str()

    # -- free --

    def free(self, free_index: torch.Tensor) -> None:
        with record_function("UnifiedSWAAlloc.free"):
            if free_index is None or free_index.numel() == 0:
                return
            if self.free_group is not None:
                self.free_group.append(self._copy_for_free_group(free_index))
                return
            # Free both peers; the per-sub-pool v2p IS the mapping, so order isn't
            # load-bearing. Filter the swa side to skip already-tombstoned virtuals
            # (`swa.v2p_page == -1` from an earlier `free_swa`); the full side needs
            # no filter (it's the lifecycle owner, so every value is still bound).
            v = free_index.detach().to(torch.int64)
            v_pages = v // self.page_size
            swa_v2p_pages = self.swa_attn_allocator.virtual_to_physical[v_pages]
            # `> 0` strict: -1 = tombstoned, 0 = padding-sink page; both skipped.
            live_token_mask = swa_v2p_pages > 0
            live_tokens = v[live_token_mask]
            if live_tokens.numel() > 0:
                self.swa_attn_allocator.free(live_tokens)
            self.full_attn_allocator.free(v)
            self.full_attn_allocator.clear_inverse_history()
            self.swa_attn_allocator.clear_inverse_history()

    def free_swa(
        self, free_index: torch.Tensor, *, start_pos: Optional[int] = None
    ) -> None:
        """SWA tombstone path: release swa-physical, leave virtual id and
        full-physical live. Called by the per-step window ratchet and by radix
        SWA eviction when a node ages past the sliding-window horizon.
        `swa.v2p_page[v_page] = -1` IS the tombstone.

        ``start_pos`` is the `free_segment` contract: when the caller frees a
        CONTIGUOUS ascending range whose first token sits at prefix position
        `start_pos` (the window ratchet does — host-int, page-aligned bounds),
        page representatives come from stride arithmetic and the swa side is
        freed with caller-supplied page ids — no `torch.unique`, keeping the
        per-decode-step free host-sync-free. Without it (radix eviction hands
        arbitrary node values) the swa side falls back to its own dedup.
        """
        if free_index is None or free_index.numel() == 0:
            return
        v = free_index.detach().to(torch.int64)
        ps = self.page_size
        if start_pos is not None and ps > 1:
            reps = self.swa_attn_allocator._page_reps(v, start_pos)
            # Keep only pages still bound on swa (freeing a tombstoned one
            # would corrupt the hole list). `> 0` strict: -1 = tombstoned,
            # page 0 = padding sink (never freeable).
            rep_pages = reps // ps
            swa_v2p_pages = self.swa_attn_allocator.virtual_to_physical[rep_pages]
            live_reps = reps[swa_v2p_pages > 0]
            if live_reps.numel() == 0:
                return
            self.swa_attn_allocator.free(live_reps, _pages=live_reps // ps)
            self.swa_attn_allocator.clear_inverse_history()
            return
        v_pages = v // ps
        # `> 0` strict: -1 = tombstoned, page 0 = padding sink (never freeable).
        swa_v2p_pages = self.swa_attn_allocator.virtual_to_physical[v_pages]
        live = v[swa_v2p_pages > 0]
        if live.numel() == 0:
            return
        if ps == 1:
            # token == page and the live filter just deduped against the v2p
            # table, so these ARE unique page ids -- same skip as `_free_lazy`.
            self.swa_attn_allocator.free(live, _pages=live)
        else:
            self.swa_attn_allocator.free(live)
        self.swa_attn_allocator.clear_inverse_history()

    def free_full(self, free_index: torch.Tensor) -> None:
        """Release the full-physical page and the virtual id, leaving the swa
        side alone -- the caller already tombstoned it (`swa.v2p_page == -1`)."""
        if free_index is None or free_index.numel() == 0:
            return
        if self.free_group is not None:
            self.full_free_group.append(self._copy_for_free_group(free_index))
            return
        self.full_attn_allocator.free(free_index.detach().to(torch.int64))
        self.full_attn_allocator.clear_inverse_history()

    def free_full_segment(self, free_index: torch.Tensor, *, start_pos: int) -> None:
        if free_index is None or free_index.numel() == 0:
            return
        if self.page_size == 1:
            # token == page: free_full already frees by exact ids, no dedup.
            self.free_full(free_index)
            return
        # The swa v2p is the mapping, so a tombstoned swa page drops out of the
        # two-sided segment path by itself; full-only is the same call.
        self.free_segment(free_index, start_pos=start_pos)

    def set_full_to_swa_mapping(
        self, full_indices: torch.Tensor, swa_indices: torch.Tensor
    ) -> None:
        """No-op stub for HiCache load-back compatibility. In shared mode there is
        no mapping tensor (the swa v2p IS the mapping); HiCache for shared SWA is
        out of scope.
        """
        return

    def clear_full_to_swa_mapping(self, full_indices: torch.Tensor) -> None:
        # Paired with set_full_to_swa_mapping: shared mode has no mapping tensor.
        return

    # -- free-group --

    # Not the SWA parent's hooks: those open the parent's paged full allocator
    # as a free group, and this composite's sub-pools defer on their own.
    def free_group_begin(self) -> None:
        BaseTokenToKVPoolAllocator.free_group_begin(self)
        self.free_page_reps_group = []
        self.full_free_group = []

    def free_group_end(self) -> None:
        pending, self.free_page_reps_group = self.free_page_reps_group, None
        full_free_group, self.full_free_group = self.full_free_group, []
        BaseTokenToKVPoolAllocator.free_group_end(self)
        if full_free_group:
            self.full_attn_allocator.free(torch.cat(full_free_group))
            self.full_attn_allocator.clear_inverse_history()
        if pending:
            self._release_page_reps(pending)

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int) -> None:
        """Fixed-shape counterpart of `free()`; see
        `MultiEndedAllocator._page_reps`. Both sides share one
        derivation -- neither repeats the position-less dedup.
        """
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
        v_pages = reps // self.page_size
        # Same tombstone filter as `free`, but at PAGE granularity (page_size
        # times smaller): `> 0` strict -- -1 = tombstoned, 0 = padding sink.
        swa_v2p_pages = self.swa_attn_allocator.virtual_to_physical[v_pages]
        live_pages = v_pages[swa_v2p_pages > 0]
        if live_pages.numel() > 0:
            self.swa_attn_allocator.free(live_pages * self.page_size, _pages=live_pages)
        self.full_attn_allocator.free(reps, _pages=v_pages)
        self.full_attn_allocator.clear_inverse_history()
        self.swa_attn_allocator.clear_inverse_history()

    def verify_byte_accounting(self) -> List[str]:
        return (
            _chain_byte_accounting_violations(
                _end_pair_chain(self.full_attn_allocator, self.swa_attn_allocator)
            )
            + self._joint_capacity_memo_violations()
        )

    def _joint_capacity_memo_violations(self) -> List[str]:
        """Idle-time twin of `MultiEndedAllocator._capacity_memo_violations`
        for the composite joint view. Empty == healthy."""
        if (
            self._joint_avail_memo_epoch
            != self.full_attn_allocator._chain_capacity_epoch()
        ):
            return []
        actual = self._compute_available_size()
        if self._joint_avail_memo_tokens == actual:
            return []
        return [
            f"[joint] stale available_size memo: "
            f"cached={self._joint_avail_memo_tokens}, actual={actual}"
        ]

    def clear(self) -> None:
        self.full_attn_allocator.clear()
        self.swa_attn_allocator.clear()
        self.free_group = None
        self.free_page_reps_group = None
        self.full_free_group = []

    # -- Lazy compaction hooks --

    def set_latest_forward_done_event(self, event: Optional[torch.cuda.Event]) -> None:
        """Forward the per-batch `forward_done` event to BOTH sub-allocators."""
        with record_function("UnifiedSWAAlloc.set_latest_forward_done_event"):
            self.full_attn_allocator.set_latest_forward_done_event(event)
            self.swa_attn_allocator.set_latest_forward_done_event(event)

    def set_inflight_forward(
        self,
        forward_done: torch.cuda.Event,
        out_cache_loc_virtual: Optional[torch.Tensor],
    ) -> None:
        """Hand the forward's metadata to BOTH sub-pools. Each materializes its
        write-set via its OWN v2p; the forward writes both sides per new token,
        so both get a non-empty in-flight tensor.
        """
        with record_function("UnifiedSWAAlloc.set_inflight_forward"):
            self.full_attn_allocator.set_inflight_forward(
                forward_done, out_cache_loc_virtual
            )
            self.swa_attn_allocator.set_inflight_forward(
                forward_done, out_cache_loc_virtual
            )

    def flush_opportunistic(self) -> int:
        """Non-urgent flush of BOTH sub-allocators; sync-free. Composite empty-set
        fast-path skips both calls when neither side has work.
        """
        with record_function("UnifiedSWAAlloc.flush_opportunistic"):
            fa = self.full_attn_allocator
            sa = self.swa_attn_allocator
            if (
                fa._free_phys_pages.numel() == 0
                and not fa._pending_reuse
                and sa._free_phys_pages.numel() == 0
                and not sa._pending_reuse
            ):
                return 0
            return fa.flush_opportunistic() + sa.flush_opportunistic()


class UnifiedMambaSWATokenToKVPoolAllocator(UnifiedSWATokenToKVPoolAllocator):
    """Tri-pool composite for models with full KV + SWA KV + mamba/conv state
    (Inkling-class: both `mambaish_config` and `is_hybrid_swa`).

    Chain (low byte -> high byte):

        [ mamba/conv (grow-up END) | swa (FLOAT middle) | full (grow-down END) ]

    Placement rationale: end pools never relocate — the request-granular,
    fat-slot state pool and the unbounded per-step grower (full) take the
    ends; SWA is window-capped (steady-state span ~= sum(min(seq, window)))
    with the cheapest slots to move, so it floats. Out-of-window `free_swa`
    tombstones become the float's interior HOLES, recycled in place by the
    next per-step allocs — steady-state SWA churn costs zero copies.

    Token surface: inherited from the SWA composite (full = id-owner of the
    per-token virtual ids; swa binds the same ids via `alloc_with_virtual`,
    now on a `FloatMultiEndedAllocator`). Per-request state surface: the
    `mamba_allocator` end MEA, wrapped by `UnifiedMambaSlotAllocator` exactly
    like the 2-pool mamba composite.
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
        self.mamba_allocator = MultiEndedAllocator(
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
        self.mamba_allocator.bind_high_peer(self.swa_attn_allocator)
        self.swa_attn_allocator.bind_low_peer(self.mamba_allocator)
        self.swa_attn_allocator.bind_high_peer(self.full_attn_allocator)
        self.full_attn_allocator.bind_low_peer(self.swa_attn_allocator)

        # None, not empty: the checker's mamba census mixes physical free-lists
        # with tree-held VIRTUAL ids, meaningless here. `free_pages is None` is
        # its documented skip contract.
        self.free_pages = None
        self.release_pages = None

        logger.info(
            "[unified-memory-pool] UnifiedMambaSWATokenToKVPoolAllocator ready: "
            "chain=[mamba(up) | swa(float) | full(down)], "
            "mamba max_slots=%d (entry_bytes=%d), joint available=%d",
            self.mamba_allocator.max_slots,
            self.mamba_allocator.entry_bytes,
            self.available_size(),
        )

    # -- construction hooks --

    def _build_swa_attn_allocator(self, **kwargs) -> MultiEndedAllocator:
        # The swa side is the FLOAT middle. Holes-first: the float never runs
        # the lazy event pipeline regardless of the composite's flag (frees
        # mark holes; allocs recycle them in place).
        kwargs["lazy_compaction"] = False
        return FloatMultiEndedAllocator(
            sub_pool_name="swa",
            is_id_owner=False,  # non-owner; consumes virtuals minted by full
            **kwargs,
        )

    def _wire_peers(self) -> None:
        # Chain wired in __init__ once the mamba end exists.
        return

    # -- capacity --

    def _compute_available_size(self) -> int:
        """Joint TOKENS for `alloc(N)`: N costs N full pages AND N swa pages.

        (Memoized by the inherited `available_size` wrapper — the chain epoch
        covers the mamba end via the frontier walks below.)

        The two sides draw on DIFFERENT free bands: full extends only downward
        into the HIGH band (between the float's high frontier — or the mamba
        end's when the float is empty/transparent — and full's low frontier);
        the swa float extends either side but a single batch alloc extends ONE
        side. Monotone feasibility predicate, solved by binary search:

            ext_f = max(0, N - H_f)      must fit:  ext_f*e_f <= B_high
            ext_s = max(0, N - H_s)      must fit:  ext_s*e_s <= max(B_low,
                                                     B_high - ext_f*e_f)
            N <= H_f + R_f,  N <= H_s + R_s          (index-space caps)

        where H_* are drainable holes (full: lazy only; swa: always — holes
        are the float's design), B_low is the band between the mamba end and
        the float's low frontier (0 when the float is transparent — the whole
        region is already in B_high), and R_* are index rooms. Order matches
        the alloc path: full takes from B_high first, then the float extends.
        """
        fa, sa = self.full_attn_allocator, self.swa_attn_allocator
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
        """All three members, same reasoning as the 2-pool pair with one
        addition each way: the FLOAT's `_flush` is zero-copy boundary
        absorption, and running it before `_ask_float_for_room` keeps the
        deficit math from pricing a span that still claims absorbed holes
        (which would buy a relocation the free shrink already covered); the
        MAMBA end's compaction feeds the low band, which the float's own
        extension for the same tokens can draw on.
        """
        return (
            self.swa_attn_allocator,
            self.full_attn_allocator,
            self.mamba_allocator,
        )

    def _alloc_demand(self, need_tokens: int):
        """Demand VECTOR for one composite allocation, in pages per band --
        zero for bands the operation does not touch. A composite token
        (prefill extend and decode alike) needs a full page AND a swa page;
        it never draws a state slot — those are per-REQUEST allocations that
        run the band-level ladder with their own {mamba: k} vector, so mamba
        is an explicit 0 here, not an omission. A future 3-pool composite
        (e.g. C128 | swa-float | C4) overrides just this vector and inherits
        the whole relocation policy.
        """
        need_n = -(-need_tokens // self.page_size)
        return {
            self.full_attn_allocator: need_n,
            self.swa_attn_allocator: need_n,
            self.mamba_allocator: 0,
        }

    def _ask_float_for_room(self, need_tokens: int) -> None:
        """Composite shortfall: hand the demand vector to the shared policy;
        the float is whichever demanded band floats."""
        demand = self._alloc_demand(need_tokens)
        flt = None
        for b in demand:
            if isinstance(b, FloatMultiEndedAllocator):
                flt = b
        _float_open_short_side(flt, demand)

    def mamba_slot_full_token_cost(self) -> int:
        """Full-token-equivalents one mamba/conv slot removes from the shared
        buffer. A tri-pool token costs e_f + e_s bytes, so:
        ceil(mamba_entry_bytes / (e_f + e_s)). Conservative (rounded up)."""
        e_tok = (
            self.full_attn_allocator.entry_bytes + self.swa_attn_allocator.entry_bytes
        )
        return -(-self.mamba_allocator.entry_bytes_per_page // e_tok)

    def debug_print(self) -> str:
        sa = self.swa_attn_allocator
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
        # full + swa are written per new token via set_kv_buffer; the mamba
        # state is written by the conv kernels, not out_cache_loc -- pass None
        # (the 2-pool mamba composite's convention).
        super().set_inflight_forward(forward_done, out_cache_loc_virtual)
        self.mamba_allocator.set_inflight_forward(forward_done, None)

    def evict_to_free_tokens(self, tree_cache, num_tokens: int) -> None:
        """Joint-aware eviction: evicting one tri-lifetime tree node frees
        bytes on several sides at once, and the default single pass's per-side
        shortfall math can leave the JOINT gate short. Bounded re-check loop:
        evict until the joint availability covers the ask or a pass stops
        making progress (then the capacity gate reports the shortfall)."""
        from sglang.srt.mem_cache.common import evict_from_tree_cache

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
                    self.swa_attn_allocator,
                    self.full_attn_allocator,
                ]
            )
            + self._joint_capacity_memo_violations()
        )

    def flush_opportunistic(self) -> int:
        """Per-step reclaim across the whole chain. The float participates:
        its holes are not flushable BACKLOG (never moved here), but its
        deferred boundary absorption is exactly the work this quiescent point
        exists for -- and it is where the float's single D2H is paid."""
        fa, ma = self.full_attn_allocator, self.mamba_allocator
        sa = self.swa_attn_allocator
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
