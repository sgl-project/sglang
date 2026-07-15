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

"""Allocator for logical-page KV sharding (see mem_cache/page_interleave.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class PageInterleavePoolAllocator(PagedTokenToKVPoolAllocator):
    """Paged allocator handing out ATOMIC, shard-aligned groups of ``shard_size``
    physical pages (one per shard-group rank), guaranteeing every rank allocates
    the same number of physical pages for every sequence.

    The inherited ``page_size`` attribute is the ALLOCATION GRANULE
    (``shard_size * physical_page_size``) — consumed by the alloc rounding.
    Kernels, pool tensors, attention page tables, RDMA descriptors, and the
    radix-tree match quantum all keep ``physical_page_size``.

    Because the tree quantum (``physical_page_size``) is finer than the
    allocation granule, frees arrive at physical-page granularity and may
    cover only part of a group. ``free()`` therefore tracks per-group
    liveness: freed physical pages are marked dead, and a group re-enters
    the free list only when its last live page dies. Dead pages inside a
    partially-live group are *stranded* — neither allocatable nor evictable —
    until the group drains (``DESIGN_kv_shard_subgranule_reuse.md`` §4.2).

    Every liveness transition is a pure function of the mirrored alloc/free
    stream, so ``_live_pages`` is byte-identical across shard-group ranks by
    construction, like the free list it guards (SPMD, no consensus protocol).
    """

    def __init__(
        self,
        size: int,
        physical_page_size: int,
        shard_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        # size              — PHYSICAL token slots of one rank's pool (1x HBM)
        # size * shard_size — logical slots managed (index-space widening only)
        # One allocator page = one logical page = shard_size physical pages,
        # one per rank; the allocator therefore has size/physical_page_size
        # pages — exactly today's per-rank physical page count.
        assert shard_size > 1, "PageInterleavePoolAllocator requires shard_size > 1"
        # Set before super().__init__: the base constructor calls clear(),
        # which sizes the liveness table from these.
        self.physical_page_size = physical_page_size
        self.shard_size = shard_size
        super().__init__(
            size * shard_size,
            page_size=physical_page_size * shard_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )

    def clear(self):
        super().clear()
        # Live physical pages per logical group (index = group id; group 0 is
        # the reserved padded page, never allocated). Groups on the free list
        # are kept at shard_size — "ready for their next allocation" — so no
        # alloc path needs a hook: free() decrements and resets on
        # reclamation; adoption and tail-padding marking subtract dead pages
        # explicitly.
        self._live_pages = torch.full(
            (self.num_pages + 1,),
            self.shard_size,
            dtype=torch.int64,
            device=self.device,
        )

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if not self.is_not_in_free_group:
            self.free_group.append(free_index)
            return

        # The tree quantum is the physical page, so every free covers whole
        # physical pages (any partial in-page coverage means the rest of that
        # page belongs to the same free — stock whole-page semantics, one
        # level down). Mark them dead; reclaim groups that drained to zero.
        phys_pages = torch.unique(free_index.long() // self.physical_page_size)
        groups, dead_counts = torch.unique(
            phys_pages // self.shard_size, return_counts=True
        )
        if self.debug_mode:
            assert torch.all(self._live_pages[groups] >= dead_counts), (
                f"double free: group live counts {self._live_pages[groups]} "
                f"< freed page counts {dead_counts}"
            )
        self._live_pages[groups] -= dead_counts
        reclaimed = groups[self._live_pages[groups] == 0]
        if reclaimed.numel() > 0:
            self._live_pages[reclaimed] = self.shard_size
            if self.need_sort:
                self.release_pages = torch.cat((reclaimed, self.release_pages))
            else:
                self.free_pages = torch.cat((reclaimed, self.free_pages))

        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int = None,
    ):
        out_indices = super().alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages,
        )
        if out_indices is None:
            return None
        self._mark_tail_padding_dead(
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens_cpu=seq_lens_cpu,
            out_indices=out_indices,
        )
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        # Sharding is prefill-disaggregation-only; a decode allocation would
        # have to resurrect tail-padding pages already marked dead by
        # _mark_tail_padding_dead, silently corrupting the liveness table.
        raise NotImplementedError(
            "PageInterleavePoolAllocator does not support decode allocation "
            "(logical-page KV sharding runs on prefill nodes only)"
        )

    def adopt_partial_prefix_groups(
        self, prefix_lens_cpu: torch.Tensor, last_loc: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Fresh-group adoption: for each request whose prefix ends mid-group,
        pop one free group and return a synthetic ``last_loc`` placing the
        first new token at the position-congruent offset of that group.

        The popped group's head pages ``[0, prefix_len % granule)`` are
        position-covered by the (shared, read-only) boundary group of the
        prefix, so they are dead at birth; stock ``alloc_extend`` then
        continues from the synthetic ``last_loc`` and its page arithmetic
        works out unchanged (``DESIGN_kv_shard_subgranule_reuse.md`` §4.3).

        Requests with group-aligned prefixes keep their ``last_loc`` entries —
        the path is bit-identical to no-adoption. Returns None when the free
        list cannot supply the adopted groups (caller must have evicted).
        """
        gs, ps = self.page_size, self.physical_page_size
        offsets = prefix_lens_cpu % gs
        need = offsets > 0
        n_adopt = int(need.sum())
        if n_adopt == 0:
            return last_loc
        if self.debug_mode:
            # ps-aligned by the radix tree's match quantum.
            assert torch.all(offsets % ps == 0)

        if self.need_sort and n_adopt > len(self.free_pages):
            self.merge_and_sort_free()
        if n_adopt > len(self.free_pages):
            return None
        adopted_groups = self.free_pages[:n_adopt]
        self.free_pages = self.free_pages[n_adopt:]

        adopted_offsets = offsets[need].to(self.device)
        self._live_pages[adopted_groups] = self.shard_size - adopted_offsets // ps

        new_last_loc = last_loc.clone()
        new_last_loc[need.to(self.device)] = adopted_groups * gs + adopted_offsets - 1
        return new_last_loc

    def _mark_tail_padding_dead(
        self,
        prefix_lens_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        out_indices: torch.Tensor,
    ):
        """Mark each request's tail-group padding pages dead at birth.

        Whole-group allocation consumes those slots, but they are never
        entered into ``req_to_token``, so no tree free will ever cover them;
        without this the tail group could never drain to zero and would leak.
        Mid-request chunk boundaries are granule-aligned (scheduler flooring),
        so this fires at most once per request, on its final chunk.
        """
        gs, ps = self.page_size, self.physical_page_size
        tail_lens = seq_lens_cpu % gs
        extend_lens = seq_lens_cpu - prefix_lens_cpu
        mask = (tail_lens > 0) & (extend_lens > 0)
        if not bool(mask.any()):
            return
        dead_tails = ((gs - tail_lens[mask]) // ps).to(self.device)
        last_slot_idx = (torch.cumsum(extend_lens, 0) - 1)[mask].to(self.device)
        tail_groups = out_indices[last_slot_idx] // gs
        self._live_pages[tail_groups] -= dead_tails

    def dead_size(self) -> int:
        """Logical token slots dead inside partially-live groups: freed by the
        tree (or dead at birth) but not yet reclaimable because their group
        still holds live pages ("stranded" in the design docs). Free-list
        groups count zero (their live count is reset to shard_size on
        reclamation). Completes the accounting identity with the allocator's
        available_size and the tree's evictable_size/protected_size."""
        live = int(self._live_pages[1:].sum().item())
        return (self.num_pages * self.shard_size - live) * self.physical_page_size

    def backup_state(self):
        return (*super().backup_state(), self._live_pages.clone())

    def restore_state(self, state):
        free_pages, release_pages, live_pages = state
        super().restore_state((free_pages, release_pages))
        self._live_pages = live_pages

    def debug_check_equal_allocation(self):
        # Groups leave the free list whole, so consumed capacity is a whole
        # number of groups at all times (stranded dead pages live inside
        # consumed groups); per-rank physical usage is identical on every
        # rank without any cross-rank consensus.
        used_slots = self.size - self.available_size()
        assert used_slots % self.page_size == 0, (
            f"partial logical-page group leaked: used_slots={used_slots} is not "
            f"a multiple of the group span {self.page_size}"
        )


def page_interleave_shard_size(allocator: BaseTokenToKVPoolAllocator) -> int:
    """Shard-group size of a widened allocator, or 1 for stock allocators.

    The scheduler-side seam predicate: the sites that must treat the allocator
    page as the working quantum (alloc-path page size, budget widening,
    invariant slack, adoption) branch on this, exactly parallel to their
    existing DCP conditions.
    """
    if isinstance(allocator, PageInterleavePoolAllocator):
        return allocator.shard_size
    return 1
