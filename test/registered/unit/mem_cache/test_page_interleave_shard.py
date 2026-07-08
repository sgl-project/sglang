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
   physical page count; atomic whole-group frees.
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

import numpy as np
import torch

from sglang.srt.disaggregation.utils import filter_kv_indices_for_shard_rank
from sglang.srt.mem_cache.allocator.page_interleave import (
    PageInterleavePoolAllocator,
    page_interleave_shard_size,
)
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
        # Freeing ANY index of a group releases the whole group.
        alloc.free(out[::GS].clone())
        self.assertEqual(alloc.available_size(), total)
        alloc.debug_check_equal_allocation()


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
    def _run_begin(self, prefix_len, seq_len, groups_row):
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

        # req_to_token row: token loc = groups_row[pos // GS] * GS + pos % GS
        row = torch.empty(seq_len, dtype=torch.int32)
        for j, q in enumerate(groups_row):
            start = j * GS
            n = min(GS, seq_len - start)
            if n <= 0:
                break
            row[start : start + n] = torch.arange(q * GS, q * GS + n, dtype=torch.int32)
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
        with self.assertRaises(AssertionError):
            self._run_begin(GS + PS, 2 * GS, [3, 9])


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
