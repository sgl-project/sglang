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

"""Allocator for logical-page KV sharding (see mem_cache/page_interleave.py).

Rotated owner-classed page allocation (DESIGN_kv_shard_classed_page_alloc.md):
physical pages are handed out one at a time from N mirrored per-rank class
free lists. Logical page ``l = loc // ps`` is owned by rank ``l % N`` and is
that rank's local physical page ``l // N``, so class ``r``'s free list IS
rank ``r``'s free pages. Ownership is derived from ``loc`` everywhere (write
filter, gather plan, scratch translation, P/D send filter), which makes any
class choice correct; balance is pure policy:

- ``class(position-page P of a chain) = (b + P) % N`` where ``b`` is the
  chain's rotation base — a new chain draws ``b`` from the least-full class,
  an extension continues from the phase recorded on ``req.last_node``
  (``TreeNode.rotation_base``). Within one cached prefix the owners are
  exactly cyclic, so per-rank page counts differ by <= 1 and the prefix
  gather stays a regular padded allgather.

Every list, cursor decision, and pop is a pure function of the mirrored
alloc/free stream, so the state is byte-identical across shard-group ranks
by construction (SPMD, no consensus protocol). A freed page is immediately
reusable — the whole-group design's per-turn granule stranding (adoption,
liveness counting, dead-at-birth padding) is gone.

``available_size`` reports the MIN-CLASS capacity floor
(``N * min_r free_pages(r) * ps``): in-flight P/D transfers lock tree nodes,
so an aggregate gate could admit a request whose tight class has nothing
evictable — fail-loud where min-class admission defers. Rotation plus
least-full seeding keeps the classes near-balanced, so the floor tracks the
aggregate within the bounded skew (design §4.1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class PageInterleavePoolAllocator(PagedTokenToKVPoolAllocator):
    """Classed paged allocator over the shard-widened logical index space.

    The inherited ``page_size`` is the PHYSICAL page — the working quantum of
    every seam that reads it (radix-tree match, chunk flooring, admission
    reserve arithmetic, free alignment, wire sampling). The widening is pure
    index space: ``size`` logical slots = ``shard_size`` x one rank's
    physical slots; per-rank HBM is unchanged.
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
        assert shard_size > 1, "PageInterleavePoolAllocator requires shard_size > 1"
        # Set before super().__init__: the base constructor calls clear(),
        # which builds the class lists from these.
        self.physical_page_size = physical_page_size
        self.shard_size = shard_size
        super().__init__(
            size * shard_size,
            page_size=physical_page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )

    def clear(self):
        # Logical pages l ∈ [0, shard_size) are reserved — physical page 0 of
        # every rank (loc ∈ [0, shard_size * ps) is the padded/trash range).
        # Class r's free list holds the allocatable logical pages with
        # l % shard_size == r, ascending — exactly rank r's free physical
        # pages 1..pages_per_rank at local page l // shard_size.
        pages_per_rank = self.num_pages // self.shard_size
        self.class_free_pages: List[torch.Tensor] = [
            torch.arange(1, pages_per_rank + 1, dtype=torch.int64, device=self.device)
            * self.shard_size
            + r
            for r in range(self.shard_size)
        ]
        self.class_release_pages: List[torch.Tensor] = [
            torch.empty((0,), dtype=torch.int64, device=self.device)
            for _ in range(self.shard_size)
        ]
        self.is_not_in_free_group = True
        self.free_group = []
        # Neutralize the base single-list attributes: every consumer of this
        # allocator must go through the classed API (or fail loud on None),
        # never a stale flat free list.
        self.free_pages = None
        self.release_pages = None

    # ---- classed accounting ---------------------------------------------------

    def class_free_page_counts(self) -> List[int]:
        """Free pages per class (free + release) — the per-class watermarks
        exported at the scheduler invariant-checker seam."""
        return [
            len(free) + len(release)
            for free, release in zip(self.class_free_pages, self.class_release_pages)
        ]

    def available_size(self) -> int:
        # Min-class capacity floor: a K-page request needs up to
        # ceil(K / N) pages of EACH class (cyclic draws), so admission must
        # gate on the tightest class, not the aggregate — the aggregate can
        # be large while one class is fully protected by locked chains.
        return self.shard_size * self.page_size * min(self.class_free_page_counts())

    def aggregate_free_size(self) -> int:
        """Total free logical slots across all classes — the accounting
        identity's `available` term (the invariant checker); NOT an admission
        gate (see available_size)."""
        return self.page_size * sum(self.class_free_page_counts())

    def least_full_class(self) -> int:
        """Rotation base for a new chain: the class with the most free pages,
        ties broken by the lowest class id. A pure function of the mirrored
        class fills (SPMD-safe). Least-full seeding is required, not
        cosmetic: oblivious round-robin drifts unboundedly under adversarial
        traffic, least-full self-corrects the per-chain <= 1-page remainders
        to multinomial noise (design §4.1)."""
        counts = self.class_free_page_counts()
        return max(range(self.shard_size), key=lambda r: (counts[r], -r))

    def merge_and_sort_free(self):
        for r in range(self.shard_size):
            if len(self.class_release_pages[r]) > 0:
                merged = torch.cat(
                    (self.class_free_pages[r], self.class_release_pages[r])
                )
                self.class_free_pages[r], _ = torch.sort(merged)
                self.class_release_pages[r] = torch.empty(
                    (0,), dtype=torch.int64, device=self.device
                )

    # ---- alloc / free -----------------------------------------------------------

    def alloc(self, need_size: int):
        raise NotImplementedError(
            "PageInterleavePoolAllocator allocates through alloc_extend only "
            "(class draws follow the chain's rotation; a bare alloc has no "
            "rotation base)"
        )

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int = None,
        rotation_base: Optional[int] = None,
    ):
        """Pop one page per new position-page, class ``(b + P) % N``.

        Sync-free: per-class pop counts are closed-form from the host
        lengths and the host rotation base (the reason the base is host
        metadata on radix nodes rather than derived from device locs).
        Returns None when some needed class cannot supply its pages — the
        caller's admission gate (min-class available_size + the N*ps
        reserve) makes that unreachable outside true OOM.
        """
        ps, shard_size = self.page_size, self.shard_size
        assert len(prefix_lens_cpu) == 1, (
            "logical-page KV sharding v1 allocates one request per extend "
            f"batch, got {len(prefix_lens_cpu)} (the rotation plan and the "
            "assembly scratch are sized for a single request)"
        )
        assert rotation_base is not None, (
            "sharded alloc_extend needs the chain's rotation base "
            "(alloc_paged_token_slots_extend derives it from req.last_node)"
        )
        prefix_len = int(prefix_lens_cpu[0])
        seq_len = int(seq_lens_cpu[0])
        assert prefix_len % ps == 0, (
            f"sharded extends start page-aligned (the radix-tree match "
            f"quantum and the chunk flooring guarantee it), got "
            f"prefix_len={prefix_len}, page_size={ps}"
        )
        assert extend_num_tokens == seq_len - prefix_len
        new_pages = -(-seq_len // ps) - prefix_len // ps
        assert new_pages > 0
        start_class = (rotation_base + prefix_len // ps) % shard_size

        if self.debug_mode and prefix_len > 0:
            # Owner congruence of the prefix end: the host-tracked phase must
            # agree with the loc-derived owner (one D2H sync, debug only). A
            # mismatch means a stale rotation base — the pool's plan would
            # translate into the wrong rank's scratch block.
            actual = int(last_loc[0].item()) // ps % shard_size
            expected = (start_class - 1) % shard_size
            assert actual == expected, (
                f"rotation base out of sync with the prefix locs: owner of "
                f"the last prefix page is {actual}, host phase says {expected}"
            )

        counts = [
            new_pages // shard_size
            + (1 if (c - start_class) % shard_size < new_pages % shard_size else 0)
            for c in range(shard_size)
        ]
        if self.need_sort and any(
            counts[c] > len(self.class_free_pages[c]) for c in range(shard_size)
        ):
            self.merge_and_sort_free()
        if any(counts[c] > len(self.class_free_pages[c]) for c in range(shard_size)):
            return None

        # Interleave the class pops into one position-ordered page vector:
        # position-page j draws class (start_class + j) % N, so class c fills
        # plan slots (c - start_class) % N, +N, +2N, ...
        pages = torch.empty((new_pages,), dtype=torch.int64, device=self.device)
        for c in range(shard_size):
            if counts[c] == 0:
                continue
            pages[
                torch.arange(
                    (c - start_class) % shard_size,
                    new_pages,
                    shard_size,
                    device=self.device,
                )
            ] = self.class_free_pages[c][: counts[c]]
            self.class_free_pages[c] = self.class_free_pages[c][counts[c] :]

        offsets = torch.arange(extend_num_tokens, dtype=torch.int64, device=self.device)
        out_indices = pages[offsets // ps] * ps + offsets % ps

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        raise NotImplementedError(
            "PageInterleavePoolAllocator does not support decode allocation "
            "(logical-page KV sharding runs on prefill nodes only)"
        )

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if not self.is_not_in_free_group:
            self.free_group.append(free_index)
            return

        # The tree quantum is the physical page, so every free covers whole
        # pages (any partial in-page coverage means the rest of that page
        # belongs to the same free). Split by owner class; a freed page is
        # immediately reusable — nothing strands.
        pages = torch.unique(free_index.long() // self.page_size)
        owners = pages % self.shard_size
        for r in range(self.shard_size):
            freed = pages[owners == r]
            if freed.numel() == 0:
                continue
            if self.need_sort:
                self.class_release_pages[r] = torch.cat(
                    (freed, self.class_release_pages[r])
                )
            else:
                self.class_free_pages[r] = torch.cat((freed, self.class_free_pages[r]))

        if self.debug_mode:
            self.debug_check_classes()

    # ---- state / debug ----------------------------------------------------------

    def backup_state(self):
        return (list(self.class_free_pages), list(self.class_release_pages))

    def restore_state(self, state):
        class_free_pages, class_release_pages = state
        self.class_free_pages = list(class_free_pages)
        self.class_release_pages = list(class_release_pages)

    def resize(self, config) -> None:
        raise NotImplementedError(
            "post-capture KV resizing is not supported under logical-page "
            "KV sharding"
        )

    def debug_all_free_pages(self) -> torch.Tensor:
        """All free logical page ids across classes (free + release) — for
        the scheduler invariant checker's use-after-free / double-free sweep
        (page unit = the physical page = self.page_size)."""
        return torch.cat(self.class_free_pages + self.class_release_pages)

    def debug_check_classes(self):
        for r in range(self.shard_size):
            merged = torch.cat((self.class_free_pages[r], self.class_release_pages[r]))
            assert bool(
                (merged % self.shard_size == r).all()
            ), f"page of another owner leaked into class {r}"
            assert len(torch.unique(merged)) == len(
                merged
            ), f"double free: duplicate pages in class {r}"


def page_interleave_shard_size(allocator: BaseTokenToKVPoolAllocator) -> int:
    """Shard-group size of a widened allocator, or 1 for stock allocators.

    The scheduler-side seam predicate: the sites that must treat the
    index space as shard-widened (capacity reporting, admission reserve,
    rotation-base plumbing, invariant slack) branch on this, exactly
    parallel to their existing DCP conditions.
    """
    if isinstance(allocator, PageInterleavePoolAllocator):
        return allocator.shard_size
    return 1
