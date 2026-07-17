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
"""Unit tests for logical-page KV cache sharding (CPU only).

Pins the pure arithmetic the whole design hangs on:

1. The placement bijection ``loc = Q*(N*ps) + r*ps + o`` — owner / local-row
   round-trip, disjoint equal partition across ranks.
2. ``PageInterleavePoolAllocator`` — index-space widening over an un-widened
   physical page count; per-group liveness (sub-granule frees strand pages
   until their group drains), tail-padding marking, and fresh-group adoption
   (``DESIGN_kv_shard_subgranule_reuse.md``).
3. ``translate_loc_to_scratch`` — the per-batch group->position lookup mapping
   any consumer index vector onto the owner-major ``[prefix | chunk | trash]``
   scratch, checked against a brute-force reference.
4. ``begin_shard_extend`` plan capture (group positions, send rows) with the
   gather stubbed out, following the SimpleNamespace binding pattern of
   ``test_dsa_layer_shard_utils.py``.
5. The P/D ownership send filter ``filter_kv_indices_for_shard_rank``.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sglang.srt.disaggregation.utils import filter_kv_indices_for_shard_rank
from sglang.srt.mem_cache.allocator.page_interleave import (
    PageInterleavePoolAllocator,
    page_interleave_shard_size,
)
from sglang.srt.mem_cache.allocator.paged import alloc_extend_naive
from sglang.srt.mem_cache.page_interleave import (
    PageInterleavePlacement,
    PageShardSpec,
)
from sglang.srt.mem_cache.page_interleave_pool import PageInterleaveKVPoolMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

N = 4  # shard size
PS = 16  # physical page size
GS = N * PS  # logical page (granule)


def _make_spec(shard_rank=0, max_prefix_groups=64, chunk_groups=8):
    return PageShardSpec(
        shard_rank=shard_rank,
        shard_size=N,
        page_size=PS,
        max_prefix_tokens=max_prefix_groups * GS,
        chunk_tokens=chunk_groups * GS,
    )


class _NaiveExtendKernelShim:
    """Stands in for the triton alloc_extend_kernel on CPU: same launch
    convention (grid ``__getitem__`` then call), same index arithmetic via
    ``alloc_extend_naive`` — so the tests drive the REAL
    ``PageInterleavePoolAllocator.alloc_extend`` (free-list accounting and
    the tail-padding-marking hook included, not a test re-implementation)."""

    def __getitem__(self, grid):
        def launch(
            prefix_lens, seq_lens, last_loc, free_pages, out_indices, bs, page_size
        ):
            alloc_extend_naive(
                prefix_lens,
                seq_lens,
                last_loc,
                free_pages,
                out_indices,
                page_size,
                device="cpu",
            )

        return launch


def _alloc_extend_cpu(alloc, prefix_len, seq_len, last_loc):
    """Drive the real alloc_extend on CPU with the triton kernel shimmed."""
    import sglang.srt.mem_cache.allocator.paged as paged_mod

    with mock.patch.object(paged_mod, "alloc_extend_kernel", _NaiveExtendKernelShim()):
        out = alloc.alloc_extend(
            prefix_lens=torch.tensor([prefix_len], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([prefix_len], dtype=torch.int64),
            seq_lens=torch.tensor([seq_len], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
            last_loc=torch.tensor([last_loc], dtype=torch.int64),
            extend_num_tokens=seq_len - prefix_len,
        )
    assert out is not None
    return out


class TestPlacement(CustomTestCase):
    def test_owner_local_round_trip(self):
        pl = PageInterleavePlacement(_make_spec())
        loc = torch.arange(0, 37 * GS + 5)
        owner = pl.owner_of(loc)
        local = pl.local_index(loc)
        # Reconstruct loc from (group, owner, in-page offset): the bijection.
        group = loc // GS
        self.assertTrue(torch.equal(group * GS + owner * PS + loc % PS, loc))
        # Local rows are group-major: [Q*ps, (Q+1)*ps) — identical on every
        # rank (symmetric allocation); owner only selects WHICH rank stores.
        self.assertTrue(torch.equal(local, group * PS + loc % PS))

    def test_filter_local_partitions_disjoint_and_equal(self):
        pl = PageInterleavePlacement(_make_spec())
        loc = torch.arange(0, 10 * GS)
        parts = [pl.filter_local(loc, r) for r in range(N)]
        self.assertEqual(sum(p.numel() for p in parts), loc.numel())
        # Equal shares of whole groups.
        self.assertEqual(len({p.numel() for p in parts}), 1)
        # Every rank's local rows for a full range are the same integers
        # (each rank stores its own stripe at the SAME rows).
        for p in parts[1:]:
            self.assertTrue(torch.equal(p, parts[0]))

    def test_owned_tokens_form_page_runs(self):
        pl = PageInterleavePlacement(_make_spec(shard_rank=2))
        loc = torch.arange(0, 3 * GS)
        mask = pl.local_mask(loc, 2)
        # Owner-2 tokens are exactly [2*ps, 3*ps) of every group.
        expect = (loc % GS >= 2 * PS) & (loc % GS < 3 * PS)
        self.assertTrue(torch.equal(mask, expect))


class TestAllocator(CustomTestCase):
    def _alloc(self):
        return PageInterleavePoolAllocator(
            size=32 * PS,  # physical token slots of one rank
            physical_page_size=PS,
            shard_size=N,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )

    def test_index_space_widened_pages_not(self):
        alloc = self._alloc()
        self.assertEqual(alloc.size, 32 * PS * N)  # logical slots
        self.assertEqual(alloc.page_size, GS)  # allocation granule
        # One allocator page per PHYSICAL page of one rank's pool.
        self.assertEqual(alloc.num_pages, 32)
        self.assertEqual(page_interleave_shard_size(alloc), N)

    def test_atomic_group_alloc_free(self):
        alloc = self._alloc()
        total = alloc.available_size()
        out = alloc.alloc(3 * GS)
        self.assertEqual(out.numel(), 3 * GS)
        # Page 0 (group 0) is the reserved padded page — never handed out.
        self.assertGreaterEqual(int(out.min()), GS)
        self.assertEqual(alloc.available_size(), total - 3 * GS)
        alloc.debug_check_equal_allocation()
        # Freeing every slot of a group releases the whole group.
        alloc.free(out)
        self.assertEqual(alloc.available_size(), total)
        self.assertEqual(alloc.dead_size(), 0)
        alloc.debug_check_equal_allocation()

    def test_partial_free_strands_until_group_drains(self):
        """§4.2 of DESIGN_kv_shard_subgranule_reuse.md: with the tree quantum
        at the physical page, a node split inside one group lets eviction free
        a sub-group range. The freed pages must NOT release the group (its
        other pages are live elsewhere in the tree); they strand until the
        group's last live page dies."""
        alloc = self._alloc()
        total = alloc.available_size()
        out = alloc.alloc(GS)
        # Free pages 1..3 (a partial in-page coverage of page 1 included:
        # any touched physical page dies whole — its remaining slots belong
        # to the same tree free, one level down).
        alloc.free(out[PS : PS + 3])
        alloc.free(out[2 * PS :])
        self.assertEqual(alloc.available_size(), total - GS)  # nothing reclaimed
        self.assertEqual(alloc.dead_size(), 3 * PS)
        alloc.debug_check_equal_allocation()
        # The last live page drains the group back to the free list.
        alloc.free(out[:PS])
        self.assertEqual(alloc.available_size(), total)
        self.assertEqual(alloc.dead_size(), 0)

    def test_tail_padding_dead_at_alloc_extend(self):
        """Whole-group allocation consumes padding slots past the last token,
        but they never enter req_to_token, so no tree free ever covers them.
        alloc_extend must mark them dead at birth or the tail group can never
        drain and leaks permanently."""
        alloc = self._alloc()
        total = alloc.available_size()
        out = _alloc_extend_cpu(alloc, prefix_len=0, seq_len=3 * PS, last_loc=-1)
        self.assertEqual(alloc.available_size(), total - GS)
        self.assertEqual(alloc.dead_size(), PS)  # the padding page
        # Freeing only the written token slots reclaims the whole group.
        alloc.free(out)
        self.assertEqual(alloc.available_size(), total)
        self.assertEqual(alloc.dead_size(), 0)

    def test_adoption_worked_example(self):
        """Appendix A of DESIGN_kv_shard_subgranule_reuse.md, scaled to
        shard 4 x ps 16: a prefix hit ending mid-group extends into a fresh
        group at the position-congruent offset (I3), stock page counting
        stays exact, and everything drains at eviction."""
        alloc = self._alloc()
        total = alloc.available_size()

        # Turn 1: request A = 7 pages (112 tokens), no prefix. 2 groups;
        # the tail group's padding page is dead at birth.
        a_out = _alloc_extend_cpu(alloc, prefix_len=0, seq_len=112, last_loc=-1)
        self.assertEqual(alloc.available_size(), total - 2 * GS)
        self.assertEqual(alloc.dead_size(), PS)
        # I3 for a group-aligned start: loc congruent to position mod GS.
        self.assertTrue(torch.equal(a_out % GS, torch.arange(112) % GS))

        # Turn 2: request B matches all 112 (ps-aligned, mid-group: 112 % 64
        # = 48) and extends to 176. Adoption pops a fresh group, dead head
        # 48/16 = 3 pages, and hands back the synthetic continuation point.
        last_loc = alloc.adopt_partial_prefix_groups(
            torch.tensor([112]), torch.tensor([int(a_out[111])])
        )
        q_f = int(last_loc[0]) // GS
        self.assertEqual(int(last_loc[0]) % GS, 47)
        self.assertEqual(int(alloc._live_pages[q_f]), 1)

        b_out = _alloc_extend_cpu(
            alloc, prefix_len=112, seq_len=176, last_loc=int(last_loc[0])
        )
        # Position-slot congruence for every new token (positions 112..175).
        self.assertTrue(
            torch.equal(b_out % GS, torch.arange(112, 176, dtype=torch.int64) % GS)
        )
        # The first new token continues the adopted group at offset 48.
        self.assertEqual(int(b_out[0]), q_f * GS + 48)
        # Free-list consumption is exact: 1 adopted + 1 counted group.
        self.assertEqual(alloc.available_size(), total - 4 * GS)
        alloc.debug_check_equal_allocation()

        # Eviction drains everything: adopted dead head + both tail paddings
        # unstrand as their groups' live pages are freed.
        alloc.free(b_out)
        alloc.free(a_out)
        self.assertEqual(alloc.available_size(), total)
        self.assertEqual(alloc.dead_size(), 0)

    def test_adoption_noop_for_aligned_prefix(self):
        alloc = self._alloc()
        n_free = len(alloc.free_pages)
        last_loc = torch.tensor([2 * GS - 1])
        out = alloc.adopt_partial_prefix_groups(torch.tensor([GS]), last_loc)
        self.assertIs(out, last_loc)  # bit-identical path, no group popped
        self.assertEqual(len(alloc.free_pages), n_free)

    def test_adoption_oom_returns_none(self):
        alloc = self._alloc()
        alloc.alloc(alloc.num_pages * GS)  # exhaust the free list
        out = alloc.adopt_partial_prefix_groups(
            torch.tensor([PS]), torch.tensor([PS - 1])
        )
        self.assertIsNone(out)

    def test_alloc_decode_rejected(self):
        """Decode allocation would resurrect tail-padding pages already marked
        dead, silently corrupting the liveness table — it must fail loud."""
        alloc = self._alloc()
        with self.assertRaises(NotImplementedError):
            alloc.alloc_decode(
                torch.tensor([GS + 1]), torch.tensor([GS + 1]), torch.tensor([GS - 1])
            )

    def test_backup_restore_roundtrip_liveness(self):
        """restore_state must bring the liveness table back with the free
        list, or a rolled-back partial free would leave phantom dead pages."""
        alloc = self._alloc()
        out = alloc.alloc(GS)
        state = alloc.backup_state()
        alloc.free(out[PS:])
        self.assertEqual(alloc.dead_size(), 3 * PS)
        alloc.restore_state(state)
        self.assertEqual(alloc.dead_size(), 0)
        # The restored group is fully live again: freeing all of it reclaims.
        total_before = alloc.available_size()
        alloc.free(out)
        self.assertEqual(alloc.available_size(), total_before + GS)


class TestEvictUntilAllocatable(CustomTestCase):
    """The evict-then-allocate contract under sub-granule stranding: one
    evict() sized in tokens can reclaim less allocatable memory than it freed
    (freed pages strand in partially-live groups), so the alloc path iterates.
    Guards the two termination conditions of _evict_until_allocatable."""

    def _allocator_with_dead_pages(self):
        alloc = PageInterleavePoolAllocator(
            size=4 * PS,  # 4 groups
            physical_page_size=PS,
            shard_size=N,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        outs = [alloc.alloc(GS) for _ in range(4)]
        # Strand every group: free all but the first page.
        for out in outs:
            alloc.free(out[PS:])
        assert alloc.available_size() == 0
        return alloc, outs

    def _tree_stub(self, alloc, frees):
        from sglang.srt.mem_cache.base_prefix_cache import EvictResult

        stub = SimpleNamespace(calls=0)

        def evict(params):
            stub.calls += 1
            if not frees:
                return EvictResult(num_tokens_evicted=0)
            head = frees.pop(0)
            alloc.free(head)
            return EvictResult(num_tokens_evicted=head.numel())

        stub.evict = evict
        return stub

    def test_iterates_until_whole_groups_free(self):
        from sglang.srt.mem_cache.common import _evict_until_allocatable

        alloc, outs = self._allocator_with_dead_pages()
        # Each eviction round drains one group's last live page: reaching
        # 2 whole free groups takes 2 rounds beyond the caller's first evict.
        frees = [out[:PS] for out in outs]
        tree = self._tree_stub(alloc, frees)
        _evict_until_allocatable(tree, alloc, 2 * GS)
        self.assertGreaterEqual(alloc.available_size(), 2 * GS)
        self.assertEqual(tree.calls, 2)

    def test_terminates_when_tree_dry(self):
        from sglang.srt.mem_cache.common import _evict_until_allocatable

        alloc, _ = self._allocator_with_dead_pages()
        tree = self._tree_stub(alloc, [])  # nothing evictable
        _evict_until_allocatable(tree, alloc, GS)
        self.assertEqual(alloc.available_size(), 0)  # need unmet, but no hang
        self.assertEqual(tree.calls, 1)


def _make_translation_stub(spec, prefix_groups, chunk_groups, num_physical_pages=512):
    """A SimpleNamespace carrying exactly the state translate_loc_to_scratch
    reads, with the plan installed the way begin_shard_extend would."""
    stub = SimpleNamespace()
    stub.shard_spec = spec
    stub._chunk_base = spec.max_prefix_tokens
    stub._trash_base = spec.max_prefix_tokens + spec.chunk_tokens
    stub._group_pos = torch.full((num_physical_pages + 2,), -1, dtype=torch.int32)
    n_prefix = len(prefix_groups)
    stub._group_pos[torch.tensor(prefix_groups, dtype=torch.int64)] = torch.arange(
        n_prefix, dtype=torch.int32
    )
    stub._group_pos[torch.tensor(chunk_groups, dtype=torch.int64)] = torch.arange(
        n_prefix, n_prefix + len(chunk_groups), dtype=torch.int32
    )
    stub._n_prefix_groups = n_prefix
    return stub


def _reference_scratch_row(spec, prefix_groups, chunk_groups, loc):
    """Brute-force reference: owner-major prefix, sequence-order chunk."""
    n_prefix = len(prefix_groups)
    block = n_prefix * spec.page_size
    q = loc // spec.logical_page_size
    in_group = loc % spec.logical_page_size
    if q in prefix_groups:
        j = prefix_groups.index(q)
        owner = in_group // spec.page_size
        return owner * block + j * spec.page_size + loc % spec.page_size
    if q in chunk_groups:
        j = chunk_groups.index(q)
        return spec.max_prefix_tokens + j * spec.logical_page_size + in_group
    return spec.max_prefix_tokens + spec.chunk_tokens + loc % spec.page_size


class TestScratchTranslation(CustomTestCase):
    def test_translation_matches_reference(self):
        spec = _make_spec()
        # Deliberately fragmented, unordered allocator groups.
        prefix_groups = [7, 3, 12, 40, 41]
        chunk_groups = [9, 25]
        stub = _make_translation_stub(spec, prefix_groups, chunk_groups)

        all_groups = prefix_groups + chunk_groups
        locs = []
        for q in all_groups:
            locs.extend(range(q * GS, (q + 1) * GS))
        locs.extend([0, 5, GS - 1])  # group 0 = reserved/padded -> trash
        locs.extend(range(30 * GS, 30 * GS + PS))  # unknown group -> trash
        loc_t = torch.tensor(locs, dtype=torch.int64)

        got = PageInterleaveKVPoolMixin.translate_loc_to_scratch(stub, loc_t)
        expect = torch.tensor(
            [
                _reference_scratch_row(spec, prefix_groups, chunk_groups, l)
                for l in locs
            ],
            dtype=torch.int64,
        )
        self.assertTrue(torch.equal(got, expect))

    def test_translation_is_injective_over_the_plan(self):
        spec = _make_spec()
        prefix_groups = [2, 30, 5]
        chunk_groups = [11]
        stub = _make_translation_stub(spec, prefix_groups, chunk_groups)
        locs = []
        for q in prefix_groups + chunk_groups:
            locs.extend(range(q * GS, (q + 1) * GS))
        rows = PageInterleaveKVPoolMixin.translate_loc_to_scratch(
            stub, torch.tensor(locs, dtype=torch.int64)
        )
        self.assertEqual(len(torch.unique(rows)), len(locs))
        # Prefix rows stay inside the prefix region, chunk rows inside chunk.
        n_prefix_tokens = len(prefix_groups) * GS
        self.assertTrue(bool((rows[:n_prefix_tokens] < spec.max_prefix_tokens).all()))
        self.assertTrue(
            bool(
                (rows[n_prefix_tokens:] >= spec.max_prefix_tokens).all()
                and (rows[n_prefix_tokens:] < stub._trash_base).all()
            )
        )

    def test_int32_page_table_input(self):
        spec = _make_spec()
        stub = _make_translation_stub(spec, [4], [8])
        table = torch.tensor([4 * GS, 4 * GS + PS, 8 * GS, 0], dtype=torch.int32)
        rows = PageInterleaveKVPoolMixin.translate_loc_to_scratch(stub, table)
        self.assertEqual(rows.dtype, torch.int64)
        # Page-aligned inputs land on page-aligned scratch rows (the FA3
        # stride-divide contract).
        self.assertTrue(bool((rows[:3] % PS == 0).all()))
        self.assertEqual(int(rows[3]), stub._trash_base)


class TestBeginShardExtendPlan(CustomTestCase):
    def _run_begin(self, prefix_len, seq_len, groups_row, row_override=None):
        spec = _make_spec()
        stub = SimpleNamespace()
        stub.shard_spec = spec
        stub.device = "cpu"
        stub.start_layer = 0
        stub._chunk_base = spec.max_prefix_tokens
        stub._trash_base = spec.max_prefix_tokens + spec.chunk_tokens
        stub._group_pos = torch.full((512 + 2,), -1, dtype=torch.int32)
        stub._epoch = 0
        stub._write_plan_key = stub._write_plan = None
        stub.prefetched = []
        stub._prefetch_layer = lambda layer_id: stub.prefetched.append(layer_id)

        if row_override is not None:
            row = row_override
        else:
            # req_to_token row: token loc = groups_row[pos // GS] * GS + pos % GS
            row = torch.empty(seq_len, dtype=torch.int32)
            for j, q in enumerate(groups_row):
                start = j * GS
                n = min(GS, seq_len - start)
                if n <= 0:
                    break
                row[start : start + n] = torch.arange(
                    q * GS, q * GS + n, dtype=torch.int32
                )
        req_to_token = row.unsqueeze(0)

        PageInterleaveKVPoolMixin.begin_shard_extend(
            stub, req_to_token, torch.tensor([0]), [prefix_len], [seq_len]
        )
        return stub

    def test_plan_with_prefix(self):
        # 3 prefix groups + 2 chunk groups (last one partial).
        groups_row = [10, 4, 22, 7, 31]
        stub = self._run_begin(3 * GS, 4 * GS + PS + 3, groups_row)
        self.assertEqual(stub._n_prefix_groups, 3)
        self.assertTrue(stub._prefix_active)
        self.assertTrue(stub._shard_extend_active)
        self.assertEqual(stub._epoch, 1)
        self.assertEqual(stub.prefetched, [0])  # first layer kicked
        # group -> plan position
        for j, q in enumerate([10, 4, 22]):
            self.assertEqual(int(stub._group_pos[q]), j)
        for j, q in enumerate([7, 31]):
            self.assertEqual(int(stub._group_pos[q]), 3 + j)
        # send rows: every rank contributes rows [Q*ps, (Q+1)*ps) per prefix
        # group, in plan order.
        expect = torch.cat([torch.arange(q * PS, (q + 1) * PS) for q in [10, 4, 22]])
        self.assertTrue(torch.equal(stub._send_rows, expect))

    def test_plan_without_prefix(self):
        stub = self._run_begin(0, GS + 5, [3, 9])
        self.assertEqual(stub._n_prefix_groups, 0)
        self.assertFalse(stub._prefix_active)
        self.assertTrue(stub._shard_extend_active)
        self.assertEqual(stub.prefetched, [])  # nothing to gather
        self.assertIsNone(stub._send_rows)
        self.assertEqual(int(stub._group_pos[3]), 0)
        self.assertEqual(int(stub._group_pos[9]), 1)

    def test_unaligned_prefix_rejected(self):
        # The tree quantum is the PHYSICAL page: a prefix that is not a
        # ps-multiple can never come out of match_prefix.
        with self.assertRaises(AssertionError):
            self._run_begin(GS + 3, 2 * GS, [3, 9])

    def _adopted_row(self, boundary_group, adopted_group, tail_groups, seq_len):
        """A req_to_token row after fresh-group adoption at prefix 112 =
        GS + 3*PS: positions 0..63 in group g0, 64..111 at offsets 0..47 of
        the boundary group, 112..127 at offsets 48..63 of the ADOPTED group
        (position-slot congruence), then aligned groups from 128."""
        g0 = 5
        row = torch.empty(seq_len, dtype=torch.int32)
        row[0:GS] = torch.arange(g0 * GS, (g0 + 1) * GS, dtype=torch.int32)
        row[GS : GS + 3 * PS] = torch.arange(
            boundary_group * GS, boundary_group * GS + 3 * PS, dtype=torch.int32
        )
        n = min(seq_len, 2 * GS) - (GS + 3 * PS)
        row[GS + 3 * PS : GS + 3 * PS + n] = torch.arange(
            adopted_group * GS + 3 * PS,
            adopted_group * GS + 3 * PS + n,
            dtype=torch.int32,
        )
        pos = 2 * GS
        for q in tail_groups:
            k = min(GS, seq_len - pos)
            if k <= 0:
                break
            row[pos : pos + k] = torch.arange(q * GS, q * GS + k, dtype=torch.int32)
            pos += k
        return row

    def test_plan_with_adopted_prefix(self):
        """§4.4 of DESIGN_kv_shard_subgranule_reuse.md: the chunk of a
        mid-group prefix hit needs the two-part extraction. The naive
        stride-gs sample from prefix_len sits at offset (prefix % gs) inside
        each group, so a trailing group holding fewer tokens than that offset
        has no sample point and would silently translate to the trash page."""
        prefix_len = GS + 3 * PS  # 112: boundary group shared mid-group
        # Trailing group g2 holds only PS tokens (16 < 48) — the naive
        # stride sample from 112 (112, 176, ...) misses it entirely.
        seq_len = 2 * GS + PS
        row = self._adopted_row(
            boundary_group=9, adopted_group=7, tail_groups=[2], seq_len=seq_len
        )
        stub = self._run_begin(prefix_len, seq_len, None, row_override=row)
        # Prefix: stride-gs sampling ceils — the partial boundary group ships.
        self.assertEqual(stub._n_prefix_groups, 2)
        self.assertEqual(int(stub._group_pos[5]), 0)
        self.assertEqual(int(stub._group_pos[9]), 1)
        # Chunk: adopted group + the trailing group the naive stride misses.
        self.assertEqual(int(stub._group_pos[7]), 2)
        self.assertEqual(int(stub._group_pos[2]), 3)
        # Send rows cover exactly the prefix groups (boundary group whole —
        # its dead tail pages ship but are never referenced).
        expect = torch.cat([torch.arange(q * PS, (q + 1) * PS) for q in [5, 9]])
        self.assertTrue(torch.equal(stub._send_rows, expect))

    def test_plan_prefix_with_interior_adoption_seam(self):
        """Regression: a cached prefix CONTAINING an adoption seam (turn 3
        reusing turn 2's cache across its adoption boundary, or chunk 2 of a
        request that adopted at admission) has one gs-window of positions
        spanning TWO groups. Stride-gs prefix sampling missed the adopted
        group entirely, so its positions translated to the trash page and
        attention silently read garbage KV."""
        # Granule-aligned prefix 192 with a seam at position 112: positions
        # 112..127 live in adopted group 7 at offsets 48..63 (I3).
        seq_len = 3 * GS + PS  # one chunk group after the prefix
        row = self._adopted_row(
            boundary_group=9, adopted_group=7, tail_groups=[2, 11], seq_len=seq_len
        )
        stub = self._run_begin(3 * GS, seq_len, None, row_override=row)
        # A 3-granule prefix spans FOUR groups (the seam adds one), in order.
        self.assertEqual(stub._n_prefix_groups, 4)
        for j, q in enumerate([5, 9, 7, 2]):
            self.assertEqual(int(stub._group_pos[q]), j)
        self.assertEqual(int(stub._group_pos[11]), 4)  # the chunk group
        # The gather ships the adopted group's stripe like any other.
        expect = torch.cat([torch.arange(q * PS, (q + 1) * PS) for q in [5, 9, 7, 2]])
        self.assertTrue(torch.equal(stub._send_rows, expect))
        # Seam positions land in the prefix scratch region, never the trash.
        seam_rows = PageInterleaveKVPoolMixin.translate_loc_to_scratch(
            stub, row[112:128].long()
        )
        self.assertTrue(bool((seam_rows < stub.shard_spec.max_prefix_tokens).all()))

    def test_plan_chunk_within_adopted_group(self):
        """Two-part extraction, degenerate case: the whole chunk fits inside
        the adopted group (no aligned trailing part)."""
        prefix_len = GS + 3 * PS
        seq_len = prefix_len + PS  # chunk = one page inside the adopted group
        row = self._adopted_row(
            boundary_group=9, adopted_group=7, tail_groups=[], seq_len=seq_len
        )
        stub = self._run_begin(prefix_len, seq_len, None, row_override=row)
        self.assertEqual(stub._n_prefix_groups, 2)
        self.assertEqual(int(stub._group_pos[7]), 2)
        # No spurious extra chunk group.
        self.assertEqual(int((stub._group_pos >= 0).sum()), 3)


class TestWritePlan(CustomTestCase):
    def test_owner_filter_cached_per_loc_tensor(self):
        spec = _make_spec(shard_rank=2)
        stub = SimpleNamespace()
        stub.placement = PageInterleavePlacement(spec)
        stub.shard_rank = 2
        stub._epoch = 1
        stub._write_plan_key = stub._write_plan = None

        loc = torch.arange(5 * GS, 7 * GS)  # two whole groups
        owned_idx, local_rows = PageInterleaveKVPoolMixin._get_write_plan(stub, loc)
        self.assertEqual(owned_idx.numel(), 2 * PS)
        # Owned rows are ps-contiguous runs at [Q*ps, (Q+1)*ps).
        self.assertTrue(
            torch.equal(
                local_rows,
                torch.cat([torch.arange(5 * PS, 6 * PS), torch.arange(6 * PS, 7 * PS)]),
            )
        )
        # Same tensor + same epoch -> cached (identity).
        again = PageInterleaveKVPoolMixin._get_write_plan(stub, loc)
        self.assertIs(again[0], owned_idx)
        # Epoch bump invalidates.
        stub._epoch = 2
        fresh = PageInterleaveKVPoolMixin._get_write_plan(stub, loc)
        self.assertIsNot(fresh[0], owned_idx)

    def test_partial_tail_group_may_own_nothing(self):
        spec = _make_spec(shard_rank=3)
        stub = SimpleNamespace()
        stub.placement = PageInterleavePlacement(spec)
        stub.shard_rank = 3
        stub._epoch = 1
        stub._write_plan_key = stub._write_plan = None
        # 10 tokens: all inside owner-0's page of the group.
        loc = torch.arange(8 * GS, 8 * GS + 10)
        owned_idx, local_rows = PageInterleaveKVPoolMixin._get_write_plan(stub, loc)
        self.assertEqual(owned_idx.numel(), 0)
        self.assertEqual(local_rows.numel(), 0)


class TestShardSendFilter(CustomTestCase):
    def test_partition_and_positional_pairing(self):
        mgr = SimpleNamespace(kv_shard_rank=0, kv_shard_size=N)
        # 9 physical pages, group ids as values (page p of group j -> Q_j).
        group_ids = np.array([5, 5, 5, 5, 9, 9, 9, 9, 2], dtype=np.int32)
        sl = slice(0, 9)
        got = {}
        for r in range(N):
            mgr.kv_shard_rank = r
            vals, pos = filter_kv_indices_for_shard_rank(
                mgr, group_ids, sl, page_offset=0
            )
            got[r] = (vals, pos)
            np.testing.assert_array_equal(pos % N, r)
            np.testing.assert_array_equal(vals, group_ids[pos])
        # Disjoint cover of all positions.
        all_pos = np.sort(np.concatenate([got[r][1] for r in range(N)]))
        np.testing.assert_array_equal(all_pos, np.arange(9))

    def test_page_offset_shifts_ownership(self):
        mgr = SimpleNamespace(kv_shard_rank=1, kv_shard_size=N)
        group_ids = np.arange(100, 108, dtype=np.int32)
        # This chunk starts at absolute page 6 (decode cached 6 pages).
        vals, pos = filter_kv_indices_for_shard_rank(
            mgr, group_ids, slice(0, 8), page_offset=6
        )
        np.testing.assert_array_equal((pos + 6) % N, 1)
        np.testing.assert_array_equal(vals, group_ids[pos])

    def test_mid_request_chunk(self):
        mgr = SimpleNamespace(kv_shard_rank=2, kv_shard_size=N)
        group_ids = np.arange(50, 62, dtype=np.int32)
        vals, pos = filter_kv_indices_for_shard_rank(
            mgr, group_ids, slice(20, 32), page_offset=0
        )
        # Positions are canonical (chunk-global), values from this chunk.
        np.testing.assert_array_equal(pos % N, 2)
        self.assertTrue(((pos >= 20) & (pos < 32)).all())
        np.testing.assert_array_equal(vals, group_ids[pos - 20])


if __name__ == "__main__":
    unittest.main()
