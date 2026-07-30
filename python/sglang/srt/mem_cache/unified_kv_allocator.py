"""
Allocator for the unified HP + int2 mixed KV pool.

Three tiers in one flat slot-id namespace::

    quant     : [0, num_quant_pages * N_Q)
    HP-prefix : [HP_OFFSET, HP_OFFSET + num_hp_prefix_slots)        (paged, shared)
    HP-recent : [HP_OFFSET + num_hp_prefix_slots, ...)              (per-req slab)
    HP_OFFSET = num_quant_pages * N_Q

Quant + HP-prefix are pooled and contribute to ``allocator.size``. HP-recent
slabs are statically reserved per req-slot (released by
``UnifiedInt2HPKVPool.release_req_slab``) and never enter the radix cache.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


class UnifiedInt2HPKVAllocator(BaseTokenToKVPoolAllocator):
    """Paged quant + HP-prefix free-lists, plus per-req HP-recent indexer."""

    def __init__(
        self,
        num_quant_pages: int,
        quant_tokens_per_page: int,
        hp_prefix_tokens: int,
        hp_recent_tokens: int,
        hp_recent_ring_size: int,
        max_req_slots: int,
        num_hp_prefix_slots: int,
        dtype,
        hp_dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
        scheduler_size: int | None = None,
    ):
        self.num_quant_pages = int(num_quant_pages)
        self.N_Q = int(quant_tokens_per_page)
        self.max_req_slots = int(max_req_slots)
        self.hp_prefix_tokens = int(hp_prefix_tokens)
        self.hp_recent_tokens = int(hp_recent_tokens)
        self.hp_recent_ring_size = int(hp_recent_ring_size)
        self.num_hp_prefix_slots = int(num_hp_prefix_slots)
        if self.N_Q <= 0:
            raise ValueError(f"quant_tokens_per_page must be positive, got {self.N_Q}")
        if self.num_hp_prefix_slots % self.N_Q != 0:
            raise ValueError(
                f"num_hp_prefix_slots ({self.num_hp_prefix_slots}) must be "
                f"a multiple of N_Q ({self.N_Q})"
            )

        self._hp_offset = self.num_quant_pages * self.N_Q
        # HP-prefix is paged with ``N_Q`` slots per page; this many pages
        # are pooled in ``hp_prefix_free_pages``.
        self.num_hp_prefix_pages = self.num_hp_prefix_slots // self.N_Q
        # Slot ids for HP-recent start past the shared HP-prefix region.
        self._hp_recent_offset = self._hp_offset + self.num_hp_prefix_slots

        # Scheduler-facing capacity = quant pool + HP-prefix pool, in slot
        # units. Page 0 of the quant arena is reserved for padded writes.
        if scheduler_size is None:
            scheduler_size = (
                self.num_quant_pages - 1
            ) * self.N_Q + self.num_hp_prefix_slots
        super().__init__(
            int(scheduler_size), self.N_Q, dtype, device, kvcache, need_sort
        )
        self.hp_dtype = hp_dtype
        self.clear()

    # -- Configuration accessors -------------------------------------------

    @property
    def hp_global_offset(self) -> int:
        return self._hp_offset

    # Compatibility alias for callers that referenced ``num_pages`` on the
    # legacy shared-arena allocator. Prefer ``num_quant_pages``.
    @property
    def num_pages(self) -> int:
        return self.num_quant_pages

    # -- Clear / state -----------------------------------------------------

    def clear(self):
        # Quant: reserve page 0 for padded/dummy writes.
        self.free_pages = torch.arange(
            1, self.num_quant_pages, dtype=torch.int64, device=self.device
        )
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)
        # HP-prefix: shared paged free list, all pages free initially.
        self.hp_prefix_free_pages = torch.arange(
            0, self.num_hp_prefix_pages, dtype=torch.int64, device=self.device
        )
        self.hp_prefix_release_pages = torch.empty(
            (0,), dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []

    # -- Availability ------------------------------------------------------

    def _free_page_count(self) -> int:
        # ``.numel()`` is host-known tensor metadata; no D2H sync.
        return self.free_pages.numel() + self.release_pages.numel()

    def _hp_prefix_free_slots(self) -> int:
        return (
            self.hp_prefix_free_pages.numel() + self.hp_prefix_release_pages.numel()
        ) * self.N_Q

    def available_size(self):
        """Pooled-slot capacity (quant + shared HP-prefix), in slot units."""
        return self._free_page_count() * self.N_Q + self._hp_prefix_free_slots()

    def debug_print(self) -> str:
        return (
            f"#quant-free-pages: {self.free_pages.numel()}, "
            f"#quant-release-pages: {self.release_pages.numel()}, "
            f"#hp-prefix-free-pages: {self.hp_prefix_free_pages.numel()}, "
            f"#hp-prefix-release-pages: {self.hp_prefix_release_pages.numel()}, "
            f"size: {self.size}, N_Q={self.N_Q}, "
            f"hp_offset={self._hp_offset}, "
            f"hp_recent_offset={self._hp_recent_offset}, "
            f"recent_ring={self.hp_recent_ring_size}, "
            f"max_req_slots={self.max_req_slots}"
        )

    def backup_state(self):
        return (
            self.free_pages.clone(),
            self.release_pages.clone(),
            self.hp_prefix_free_pages.clone(),
            self.hp_prefix_release_pages.clone(),
        )

    def restore_state(self, state):
        (
            self.free_pages,
            self.release_pages,
            self.hp_prefix_free_pages,
            self.hp_prefix_release_pages,
        ) = state

    # -- Default alloc path (extend / generic uses quant tier) -------------

    def alloc(self, need_size: int):
        return self.alloc_quant(need_size)

    # -- Per-req recent-slab base ----------------------------------------

    def _recent_slab_base_slots(self, req_pool_indices: torch.Tensor) -> torch.Tensor:
        """Global HP slot id of recent_slab[0] for each given req_pool_idx."""
        return (
            req_pool_indices.to(self.device).to(torch.int64) * self.hp_recent_ring_size
            + self._hp_recent_offset
        )

    # -- HP-prefix allocation (shared, paged) ----------------------------

    def alloc_hp_prefix(
        self,
        req_pool_indices: torch.Tensor,
        per_req_counts: Sequence[int],
    ) -> torch.Tensor:
        # Pooled allocation; req_pool_indices kept for API symmetry with
        # alloc_hp_recent (only used to drive iteration order). Total must be
        # N_Q-aligned so the result lands on whole HP-prefix pages.
        per_req_counts = list(per_req_counts)
        total = int(sum(per_req_counts))
        if total <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        for c in per_req_counts:
            if c < 0 or c > self.hp_prefix_tokens:
                raise ValueError(
                    f"alloc_hp_prefix per-req count {c} out of range "
                    f"[0, {self.hp_prefix_tokens}]"
                )
        if total % self.N_Q != 0:
            raise ValueError(
                f"alloc_hp_prefix total ({total}) must be a multiple of "
                f"N_Q={self.N_Q}; round per-req counts up at the caller."
            )

        num_pages = total // self.N_Q
        if self.need_sort and num_pages > self.hp_prefix_free_pages.numel():
            self.merge_and_sort_free()
        if num_pages > self.hp_prefix_free_pages.numel():
            raise RuntimeError(
                "Mixed KV: HP-prefix pool exhausted; consider raising "
                "SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS or evicting cache. "
                f"{self.debug_print()}"
            )

        pages = self.hp_prefix_free_pages[:num_pages]
        self.hp_prefix_free_pages = self.hp_prefix_free_pages[num_pages:]
        # Each HP-prefix page contributes N_Q consecutive slots.
        # Layout: page p → local indices [p * N_Q, p * N_Q + N_Q).
        slots_local = (
            pages[:, None] * self.N_Q
            + torch.arange(self.N_Q, dtype=torch.int64, device=self.device)
        ).reshape(-1)
        # Translate to global HP slot ids.
        return (slots_local + self._hp_offset).contiguous()

    def alloc_hp_recent(
        self,
        req_pool_indices: torch.Tensor,
        per_req_counts: Sequence[int],
    ) -> torch.Tensor:
        # Per-req ring cursor on the pool's ``_next_slab_offset``. Decode hot
        # path (uniform count=1) takes the vectorised sync-free branch.
        if isinstance(per_req_counts, torch.Tensor):
            counts_list = per_req_counts.tolist()
        else:
            counts_list = list(per_req_counts)
        bs = len(counts_list)
        total = int(sum(counts_list))
        if total <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        for c in counts_list:
            if c < 0 or c > self.hp_recent_ring_size:
                raise ValueError(
                    f"alloc_hp_recent per-req count {c} out of range "
                    f"[0, {self.hp_recent_ring_size}]"
                )

        cursor_buf = self._kvcache._next_slab_offset
        ring = self.hp_recent_ring_size

        idx_dev = req_pool_indices.to(self.device).to(torch.int64)
        bases = idx_dev * ring + self._hp_recent_offset  # [bs]
        old_cursors = cursor_buf[idx_dev].to(torch.int64)  # [bs]

        # Decode hot path: every request asks for one HP-recent slot.
        if total == bs and all(c == 1 for c in counts_list):
            slots = bases + old_cursors
            new_cursors = ((old_cursors + 1) % ring).to(torch.int32)
            cursor_buf[idx_dev] = new_cursors
            return slots.contiguous()

        # General path (extend, ragged counts).
        bases_h = bases.tolist()
        cursors_h = old_cursors.tolist()
        chunks = []
        new_cursors_list = []
        for base, count, cur in zip(bases_h, counts_list, cursors_h):
            if count == 0:
                new_cursors_list.append(cur)
                continue
            offsets = [(cur + j) % ring for j in range(count)]
            new_cursors_list.append((cur + count) % ring)
            offsets_t = torch.tensor(offsets, dtype=torch.int64, device=self.device)
            chunks.append(int(base) + offsets_t)

        # Single host→device write to update cursors.
        new_cursor_t = torch.tensor(
            new_cursors_list, dtype=torch.int32, device=self.device
        )
        cursor_buf[idx_dev] = new_cursor_t

        if not chunks:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        return torch.cat(chunks).contiguous()

    # -- Quant allocation --------------------------------------------------

    def alloc_quant(self, need_size: int):
        # Allocates whole quant pages; need_size must be N_Q-aligned.
        if need_size <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        if need_size % self.N_Q != 0:
            raise ValueError(
                f"alloc_quant requires a multiple of N_Q={self.N_Q}, got {need_size}"
            )
        num_pages = need_size // self.N_Q
        if self.need_sort and num_pages > self.free_pages.numel():
            self.merge_and_sort_free()
        if num_pages > self.free_pages.numel():
            return None

        pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
        slots = (
            pages[:, None] * self.N_Q
            + torch.arange(self.N_Q, dtype=torch.int64, device=self.device)
        ).reshape(-1)
        return slots.contiguous()

    # -- Free path (sync-free, GPU-resident) -------------------------------

    def free(self, free_index: torch.Tensor):
        # Quant -> free quant page pool; HP-prefix -> free HP-prefix page
        # pool; HP-recent -> no-op (reset by release_req_slab). Masked-select
        # branches host-sync, matching the surrounding flush path.
        if free_index.numel() == 0:
            return
        if not self.is_not_in_free_group:
            self.free_group.append(free_index)
            return

        idx = free_index.to(torch.int64)
        # Quant slot ids: aggregate to whole quant pages.
        is_quant = idx < self._hp_offset
        quant_index = idx[is_quant]
        if quant_index.numel() > 0:
            quant_pages = torch.unique((quant_index // self.N_Q).to(torch.int64))
            if self.need_sort:
                self.release_pages = torch.cat([quant_pages, self.release_pages])
            else:
                self.free_pages = torch.cat([quant_pages, self.free_pages])

        # HP-prefix slot ids: aggregate to whole HP-prefix pages.
        if self.num_hp_prefix_slots > 0:
            is_hp_prefix = (idx >= self._hp_offset) & (idx < self._hp_recent_offset)
            hp_prefix_index = idx[is_hp_prefix]
            if hp_prefix_index.numel() > 0:
                local = hp_prefix_index - self._hp_offset
                hp_pages = torch.unique((local // self.N_Q).to(torch.int64))
                if self.need_sort:
                    self.hp_prefix_release_pages = torch.cat(
                        [hp_pages, self.hp_prefix_release_pages]
                    )
                else:
                    self.hp_prefix_free_pages = torch.cat(
                        [hp_pages, self.hp_prefix_free_pages]
                    )

        # HP-recent slot ids: no-op (per-req slab).

    def merge_and_sort_free(self):
        if self.release_pages.numel() > 0:
            self.free_pages = torch.cat([self.free_pages, self.release_pages])
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )
        if self.hp_prefix_release_pages.numel() > 0:
            self.hp_prefix_free_pages = torch.cat(
                [self.hp_prefix_free_pages, self.hp_prefix_release_pages]
            )
            self.hp_prefix_free_pages, _ = torch.sort(self.hp_prefix_free_pages)
            self.hp_prefix_release_pages = torch.empty(
                (0,), dtype=self.hp_prefix_release_pages.dtype, device=self.device
            )

    # -- Passthroughs ------------------------------------------------------

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)
