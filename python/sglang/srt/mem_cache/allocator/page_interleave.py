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

Rotated owner-classed page allocation:
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
reusable: nothing strands a page until its whole logical group is free.

``available_size`` reports the MIN-CLASS capacity floor
(``N * min_r free_pages(r) * ps``): in-flight P/D transfers lock tree nodes,
so an aggregate gate could admit a request whose tight class has nothing
evictable — fail-loud where min-class admission defers. Rotation plus
least-full seeding keeps the classes near-balanced, so the floor tracks the
aggregate within the bounded skew.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
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
        shard_spec=None,
    ):
        assert shard_size > 1, "PageInterleavePoolAllocator requires shard_size > 1"
        # Set before super().__init__: the base constructor calls clear(),
        # which builds the class lists from these.
        self.physical_page_size = physical_page_size
        self.shard_size = shard_size
        # PageShardSpec (None only in unit tests): carries the assembly
        # scratch capacities the PrefillAdder gates batch admission on.
        self.shard_spec = shard_spec
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
        # need_sort only: freed pages per class, staged unsorted until an
        # alloc runs short (merge_and_sort_free), with host-side counts.
        self.class_staged_pages: List[List[torch.Tensor]] = [
            [] for _ in range(self.shard_size)
        ]
        self.class_num_staged_pages: List[int] = [0] * self.shard_size
        # None: free right away. A list: hold frees until free_group_end().
        self.free_group = None
        # Grouped free_segment() page ids, released at free_group_end().
        self.free_page_ids_group = []
        # Neutralize the base single-list attributes: every consumer of this
        # allocator must go through the classed API (or fail loud on None),
        # never a stale flat free list.
        self.free_pages = None
        self.release_pages = None
        self.staged_pages = None
        self.num_staged_pages = 0
        # Reusable pinned host buffer for the alloc_extend plan; the event
        # guards reuse until the previous H2D copy has completed.
        self._plan_host: Optional[torch.Tensor] = None
        self._plan_copied = None

    # ---- classed accounting ---------------------------------------------------

    def class_free_page_counts(self) -> List[int]:
        """Free pages per class (free + staged) — the per-class watermarks
        exported at the scheduler invariant-checker seam."""
        return [
            len(free) + staged
            for free, staged in zip(self.class_free_pages, self.class_num_staged_pages)
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
        class fills (SPMD-safe). Least-full seeding self-corrects the
        per-chain <= 1-page remainders instead of letting them drift."""
        counts = self.class_free_page_counts()
        return max(range(self.shard_size), key=lambda r: (counts[r], -r))

    def merge_and_sort_free(self):
        for r in range(self.shard_size):
            if self.class_staged_pages[r]:
                merged = torch.cat(
                    (self.class_free_pages[r], *self.class_staged_pages[r])
                )
                self.class_free_pages[r], _ = torch.sort(merged)
                self.class_staged_pages[r] = []
                self.class_num_staged_pages[r] = 0

    # ---- alloc / free -----------------------------------------------------------

    def alloc(self, need_size: int):
        raise NotImplementedError(
            "PageInterleavePoolAllocator allocates through alloc_extend only "
            "(class draws follow the chain's rotation; a bare alloc has no "
            "rotation base)"
        )

    @staticmethod
    def _class_counts(start_class: int, new_pages: int, shard_size: int) -> List[int]:
        """Pages drawn from each class by a cyclic run of ``new_pages`` pages
        starting at ``start_class`` — closed-form, host-only."""
        return [
            new_pages // shard_size
            + (1 if (c - start_class) % shard_size < new_pages % shard_size else 0)
            for c in range(shard_size)
        ]

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int = None,
        rotation_bases: Optional[List[Optional[int]]] = None,
    ):
        """Pop one page per new position-page, class ``(b_i + P) % N`` per
        request.

        ``rotation_bases`` carries one entry per request: the chain's
        rotation base, or None for a new chain. None entries are resolved IN
        PLACE from the least-full class at that request's turn — a draw sees
        the pops of earlier requests in the same batch, keeping the balance
        policy exact at bs > 1 — and the caller reads the resolved values
        back to stamp the requests for the radix insert.

        Sync-free: per-class pop counts are closed-form from the host
        lengths and the host rotation bases (the reason the base is host
        metadata on radix nodes rather than derived from device locs).
        Returns None when some needed class cannot supply its pages — the
        caller's admission gate (min-class available_size + the N*ps
        per-request reserve) makes that unreachable outside true OOM.
        """
        ps, shard_size = self.page_size, self.shard_size
        # Host lengths as Python ints: per-element tensor indexing costs a
        # dispatch per read, several reads per request.
        prefix_list = _host_int_list(prefix_lens_cpu)
        seq_list = _host_int_list(seq_lens_cpu)
        bs = len(prefix_list)
        assert rotation_bases is not None and len(rotation_bases) == bs, (
            "sharded alloc_extend needs one rotation base slot per request "
            "(alloc_paged_token_slots_extend derives them from req.last_node)"
        )

        # Pass 1 (host, mirrored): resolve draw-at-turn bases and check the
        # per-class supply against simulated fills, so the batch either
        # commits whole or defers whole.
        sim_counts = self.class_free_page_counts()
        start_classes: List[int] = []
        new_pages_list: List[int] = []
        for i in range(bs):
            prefix_len = prefix_list[i]
            seq_len = seq_list[i]
            assert prefix_len % ps == 0, (
                f"sharded extends start page-aligned (the radix-tree match "
                f"quantum and the chunk flooring guarantee it), got "
                f"prefix_len={prefix_len}, page_size={ps}"
            )
            new_pages = -(-seq_len // ps) - prefix_len // ps
            assert new_pages > 0
            if rotation_bases[i] is None:
                # New chain: least-full over the simulated fills (mirrored).
                rotation_bases[i] = max(
                    range(shard_size), key=lambda r: (sim_counts[r], -r)
                )
            start_class = (rotation_bases[i] + prefix_len // ps) % shard_size
            start_classes.append(start_class)
            new_pages_list.append(new_pages)
            for c, need in enumerate(
                self._class_counts(start_class, new_pages, shard_size)
            ):
                sim_counts[c] -= need
            if self.debug_mode and prefix_len > 0:
                # Owner congruence of the prefix end: the host-tracked phase
                # must agree with the loc-derived owner (one D2H sync, debug
                # only). A mismatch means a stale rotation base — the pool's
                # plan would translate into the wrong rank's scratch block.
                actual = int(last_loc[i].item()) // ps % shard_size
                expected = (start_class - 1) % shard_size
                assert actual == expected, (
                    f"rotation base out of sync with the prefix locs (req "
                    f"{i}): owner of the last prefix page is {actual}, host "
                    f"phase says {expected}"
                )
        assert extend_num_tokens == sum(seq_list) - sum(prefix_list)
        if min(sim_counts) < 0:
            return None
        # Under need_sort the pops slice class_free_pages only; merge the
        # staged pages in whenever any class's free list alone is shorter
        # than its total need (= total count - simulated remainder).
        needs = [
            total - remaining
            for total, remaining in zip(self.class_free_page_counts(), sim_counts)
        ]
        if self.need_sort and any(
            needs[c] > len(self.class_free_pages[c]) for c in range(shard_size)
        ):
            self.merge_and_sort_free()

        # Pass 2: commit the pops with a batch-size-independent number of
        # device ops. Each class pops its whole-batch need once, and a host
        # plan maps every output row to a (popped page, in-page offset) pair.
        # Request i's position-page j draws class c = (start_i + j) % N and
        # takes that class's (j // N)-th pop for the request, after the pops
        # of earlier requests in the batch.
        per_req_counts = [
            self._class_counts(start_classes[i], new_pages_list[i], shard_size)
            for i in range(bs)
        ]
        totals = [sum(cnt[c] for cnt in per_req_counts) for c in range(shard_size)]
        class_base = np.zeros(shard_size, dtype=np.int64)
        np.cumsum(totals[:-1], out=class_base[1:])
        popped = []
        for c in range(shard_size):
            if totals[c]:
                popped.append(self.class_free_pages[c][: totals[c]])
                self.class_free_pages[c] = self.class_free_pages[c][totals[c] :]

        plan = self._plan_buffer(extend_num_tokens)
        slot, offset = plan.numpy()
        consumed = np.zeros(shard_size, dtype=np.int64)
        row = 0
        for i in range(bs):
            extend_len = seq_list[i] - prefix_list[i]
            j = np.arange(new_pages_list[i])
            cls = (start_classes[i] + j) % shard_size
            page_slot = class_base[cls] + consumed[cls] + j // shard_size
            consumed += per_req_counts[i]
            t = np.arange(extend_len)
            slot[row : row + extend_len] = page_slot[t // ps]
            offset[row : row + extend_len] = t % ps
            row += extend_len

        if plan.is_pinned():
            # A pinned source keeps the copy ordered on the current stream
            # without a host sync; the event guards the buffer's reuse.
            plan = plan.to(self.device, non_blocking=True)
            self._plan_copied.record()
        else:
            plan = plan.to(self.device)
        pool = torch.cat(popped) if len(popped) > 1 else popped[0]
        out_indices = pool[plan[0]] * ps + plan[1]

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)
        return out_indices

    def _plan_buffer(self, num_tokens: int) -> torch.Tensor:
        """A contiguous (2, num_tokens) int64 host buffer for the plan.

        On CUDA it views a reused pinned buffer instead of pinning a fresh copy
        per call. The buffer is reused only once its previous copy completed;
        otherwise this call takes a new one rather than wait on a stream that
        may be queued behind a forward.
        """
        if torch.device(self.device).type != "cuda":
            return torch.empty((2, num_tokens), dtype=torch.int64)
        host = self._plan_host
        if (
            host is None
            or host.numel() < 2 * num_tokens
            or not self._plan_copied.query()
        ):
            capacity = max(2 * num_tokens, 0 if host is None else host.numel())
            host = torch.empty((capacity,), dtype=torch.int64, pin_memory=True)
            self._plan_host = host
            self._plan_copied = torch.cuda.Event()
        return host[: 2 * num_tokens].view(2, num_tokens)

    def alloc_cyclic(self, start_class: int, need_size: int) -> Optional[torch.Tensor]:
        """One chain's whole pages at a fixed rotation: owners cycle from
        ``start_class`` (taken mod N). HiCache copies keep their pages' owners."""
        lens = torch.tensor([0, need_size], dtype=torch.int64)
        return self.alloc_extend(
            prefix_lens=lens[:1],
            prefix_lens_cpu=lens[:1],
            seq_lens=lens[1:],
            seq_lens_cpu=lens[1:],
            last_loc=lens[:1],
            extend_num_tokens=need_size,
            rotation_bases=[start_class],
        )

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

        if self.free_group is not None:
            # Match the base allocator's ownership contract: callers may
            # overwrite req_to_token views before free_group_end consumes the
            # deferred indices.
            self.free_group.append(self._copy_for_free_group(free_index))
            return

        # The tree quantum is the physical page, so every free covers whole
        # pages (any partial in-page coverage means the rest of that page
        # belongs to the same free).
        self._free_pages(torch.unique(free_index.long() // self.page_size))

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
        """Fixed-shape free of ``kv_row[start_pos : start_pos + n]``.

        A page's tokens sit consecutively in the row and the start is
        page-aligned, so ``free_index[::ps]`` hits each page once: no
        ``torch.unique``. Contract (base): a page is freed by only one call
        per group.
        """
        if free_index.numel() == 0:
            return
        ps = self.page_size
        assert start_pos % ps == 0, f"segment start {start_pos} is not page-aligned"
        # Integer division copies, so a group holds no req_to_token view.
        page_ids = free_index[::ps].long() // ps
        if self.debug_mode:
            assert torch.equal(
                torch.sort(page_ids.cpu())[0],
                torch.unique(free_index.cpu().long() // ps),
            )
        if self.free_group is not None:
            self.free_page_ids_group.append(page_ids)
            return
        self._free_pages(page_ids)

    def free_group_end(self):
        pending, self.free_group = self.free_group, None
        page_ids, self.free_page_ids_group = self.free_page_ids_group, []
        if pending:
            merged = torch.cat(pending) if len(pending) > 1 else pending[0]
            page_ids.append(torch.unique(merged.long() // self.page_size))
        if page_ids:
            # Like the paged base: free() ids are deduped, segment ids are
            # stride samples, and the segment contract frees a page once per
            # group, so the union is distinct. One release covers the group.
            self._free_pages(torch.cat(page_ids) if len(page_ids) > 1 else page_ids[0])

    def _free_pages(self, pages: torch.Tensor) -> None:
        """Return distinct logical pages to their owner classes.

        A freed page is immediately reusable — nothing strands.
        """
        owners = pages % self.shard_size
        # One host sync for all classes (boolean-mask selection syncs once per
        # class). The stable sort keeps each class in input order, as the
        # masks did.
        counts = (
            torch.zeros(self.shard_size, dtype=torch.int64, device=pages.device)
            .scatter_add_(0, owners, torch.ones_like(owners))
            .tolist()
        )
        grouped = torch.split(pages[torch.argsort(owners, stable=True)], counts)
        for r, freed in enumerate(grouped):
            if counts[r] == 0:
                continue
            if self.need_sort:
                # Stage the split view; merge_and_sort_free concatenates.
                self.class_staged_pages[r].append(freed)
                self.class_num_staged_pages[r] += counts[r]
            else:
                self.class_free_pages[r] = torch.cat((freed, self.class_free_pages[r]))

        if self.debug_mode:
            self.debug_check_classes()

    def _release_page_ids(self, *page_ids: torch.Tensor):
        raise NotImplementedError(
            "PageInterleavePoolAllocator keeps free pages in per-owner class "
            "lists; a caller reaching the stock free_pages list would leak them"
        )

    def _debug_check_no_duplicate_pages(self):
        # The base sweep concatenates the neutralized flat lists; the classed
        # census is the equivalent check here.
        self.debug_check_classes()

    # ---- state / debug ----------------------------------------------------------

    def resize(self, config) -> None:
        raise NotImplementedError(
            "post-capture KV resizing is not supported under logical-page KV sharding"
        )

    def get_all_free_pages(self) -> torch.Tensor:
        """All free logical page ids across classes (free + staged) — for
        the scheduler invariant checker's use-after-free / double-free sweep
        (page unit = the physical page = self.page_size). The paged base
        reads the flat lists this allocator neutralizes, so the classed
        census is the override."""
        staged = [
            pages for class_pages in self.class_staged_pages for pages in class_pages
        ]
        return torch.cat(self.class_free_pages + staged)

    def debug_check_classes(self):
        for r in range(self.shard_size):
            merged = torch.cat((self.class_free_pages[r], *self.class_staged_pages[r]))
            assert bool((merged % self.shard_size == r).all()), (
                f"page of another owner leaked into class {r}"
            )
            assert len(torch.unique(merged)) == len(merged), (
                f"double free: duplicate pages in class {r}"
            )


def _host_int_list(values) -> List[int]:
    return values.tolist() if isinstance(values, torch.Tensor) else list(values)


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
