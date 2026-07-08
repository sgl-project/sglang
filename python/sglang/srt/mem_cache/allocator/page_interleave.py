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

from typing import TYPE_CHECKING

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
    (``shard_size * physical_page_size``) — consumed by the alloc rounding and
    the radix-tree match quantum. Kernels, pool tensors, attention page tables,
    and RDMA descriptors all keep ``physical_page_size``. Mechanically this is
    the landed DCP configuration (widened index space over stock 1x pools,
    ``model_runner_kv_cache_mixin.py``) restated as an explicit class at page
    granularity instead of token granularity.
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
        super().__init__(
            size * shard_size,
            page_size=physical_page_size * shard_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )
        self.physical_page_size = physical_page_size
        self.shard_size = shard_size

    def debug_check_equal_allocation(self):
        # Consumed capacity is a whole number of atomic groups at all times
        # (groups alloc and free atomically), so per-rank physical usage is
        # identical on every rank without any cross-rank consensus.
        used_slots = self.size - self.available_size()
        assert used_slots % self.page_size == 0, (
            f"partial logical-page group leaked: used_slots={used_slots} is not "
            f"a multiple of the group span {self.page_size}"
        )


def page_interleave_shard_size(allocator: BaseTokenToKVPoolAllocator) -> int:
    """Shard-group size of a widened allocator, or 1 for stock allocators.

    The scheduler-side seam predicate: the sites that must treat the allocator
    page as the working quantum (tree page size, alloc-path page size, budget
    widening, invariant slack) branch on this, exactly parallel to their
    existing DCP conditions.
    """
    if isinstance(allocator, PageInterleavePoolAllocator):
        return allocator.shard_size
    return 1
