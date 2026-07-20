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

Pins the pure arithmetic the design hangs on
(DESIGN_kv_shard_classed_page_alloc.md — rotated owner-classed allocation):

1. The placement bijection ``loc = Q*(N*ps) + r*ps + o`` — owner / local-row
   round-trip, disjoint equal partition across ranks.
2. ``PageInterleavePoolAllocator`` — N mirrored class free lists, rotated
   class draws (owners exactly cyclic along a chain), least-full root
   seeding, min-class admission accounting, zero stranding (a freed page is
   immediately reusable).
3. The host rotation base on radix ``TreeNode`` — stamped at insert, copied
   on split, read through ``last_node``.
4. ``translate_loc_to_scratch`` — the per-batch page->position lookup mapping
   any consumer index vector onto the owner-major ``[prefix | chunk | trash]``
   scratch, checked against a brute-force reference.
5. ``begin_shard_extend`` plan capture (page positions, padded send rows,
   owner-congruence guard) with the gather stubbed out, following the
   SimpleNamespace binding pattern of ``test_dsa_layer_shard_utils.py``.
6. The value-derived P/D ownership send filter
   ``filter_kv_indices_for_shard_rank``.
"""

import unittest
from array import array
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation.utils import filter_kv_indices_for_shard_rank
from sglang.srt.mem_cache.allocator.page_interleave import (
    PageInterleavePoolAllocator,
    page_interleave_shard_size,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.page_interleave import (
    PageInterleavePlacement,
    PageShardSpec,
)
from sglang.srt.mem_cache.page_interleave_pool import PageInterleaveKVPoolMixin
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

N = 4  # shard size
PS = 16  # physical page size
GS = N * PS  # full-group span (N physical pages)


def _make_spec(shard_rank=0, max_prefix_groups=64, chunk_pages=32):
    return PageShardSpec(
        shard_rank=shard_rank,
        shard_size=N,
        page_size=PS,
        max_prefix_tokens=max_prefix_groups * GS,
        chunk_tokens=chunk_pages * PS,
    )


def _make_allocator(pages_per_rank=32, need_sort=False):
    return PageInterleavePoolAllocator(
        size=pages_per_rank * PS,  # physical token slots of one rank
        physical_page_size=PS,
        shard_size=N,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )


def _alloc_extend(alloc, prefix_len, seq_len, rotation_base, last_loc=-1):
    return alloc.alloc_extend(
        prefix_lens=torch.tensor([prefix_len], dtype=torch.int64),
        prefix_lens_cpu=torch.tensor([prefix_len], dtype=torch.int64),
        seq_lens=torch.tensor([seq_len], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
        last_loc=torch.tensor([last_loc], dtype=torch.int64),
        extend_num_tokens=seq_len - prefix_len,
        rotation_base=rotation_base,
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


class TestClassedAllocator(CustomTestCase):
    def test_index_space_widened_classes_mirror_ranks(self):
        alloc = _make_allocator(pages_per_rank=32)
        self.assertEqual(alloc.size, 32 * PS * N)  # logical slots
        self.assertEqual(alloc.page_size, PS)  # the PHYSICAL page quantum
        self.assertEqual(page_interleave_shard_size(alloc), N)
        # Class r holds exactly rank r's allocatable pages: l % N == r,
        # local pages 1..32 (page 0 reserved on every rank).
        self.assertEqual(alloc.class_free_page_counts(), [32] * N)
        for r in range(N):
            pages = alloc.class_free_pages[r]
            self.assertTrue(bool((pages % N == r).all()))
            self.assertTrue(torch.equal(pages // N, torch.arange(1, 33)))

    def test_rotation_worked_example_zero_stranding(self):
        """Appendix A of DESIGN_kv_shard_classed_page_alloc.md, scaled to
        ps=16: turn 1 allocates cyclic owners from the root base; the
        turn-boundary free returns its page whole (immediately reusable —
        the whole-group design stranded one granule here); turn 2 continues
        the rotation and reuses the freed page first."""
        alloc = _make_allocator()
        total = alloc.available_size()

        # Turn 1: 122 tokens = 8 position-pages, root base 0.
        base = alloc.least_full_class()
        self.assertEqual(base, 0)  # all classes equal -> lowest id
        out = _alloc_extend(alloc, 0, 122, base)
        pages = out[::PS] // PS
        # Owners exactly cyclic from the base; in-page offsets positional.
        self.assertTrue(torch.equal(pages % N, torch.arange(8) % N))
        self.assertTrue(torch.equal(out % PS, torch.arange(122) % PS))

        # Boundary: cache 112 (7 pages), free the sub-ps tail's page whole.
        alloc.free(out[112:122])
        # Page 7's owner is (0 + 7) % 4 = 3: back on class 3, reusable now.
        self.assertEqual(alloc.class_free_page_counts(), [30, 30, 30, 31])

        # Turn 2: prefix 112, extend to 244 (9 new pages P7..P15).
        out2 = _alloc_extend(alloc, 112, 244, base, last_loc=int(out[111]))
        pages2 = out2[::PS] // PS
        self.assertTrue(torch.equal(pages2 % N, (7 + torch.arange(9)) % N))
        # The freed page is the class-3 head: reused before any fresh page.
        self.assertEqual(int(pages2[0]), int(pages[7]))
        # Nothing stranded: freeing the chain restores full capacity.
        alloc.free(out[:112])
        alloc.free(out2)
        self.assertEqual(alloc.available_size(), total)
        self.assertEqual(alloc.class_free_page_counts(), [32] * N)

    def test_min_class_admission_accounting(self):
        """available_size is the MIN-CLASS floor: draining one class must
        zero the admission budget even while the aggregate stays large —
        an aggregate gate would over-admit into the alloc path's fail-loud
        RuntimeError when the tight class is protected (design §4)."""
        alloc = _make_allocator(pages_per_rank=4)
        outs = [_alloc_extend(alloc, 0, PS, rotation_base=3) for _ in range(4)]
        self.assertEqual(alloc.class_free_page_counts(), [4, 4, 4, 0])
        self.assertEqual(alloc.available_size(), 0)
        self.assertEqual(alloc.aggregate_free_size(), 12 * PS)
        # A draw needing the empty class defers (None), never raises.
        self.assertIsNone(_alloc_extend(alloc, 0, N * PS, rotation_base=0))
        # A free of one class-3 page lifts the floor by one page per class.
        alloc.free(outs[0])
        self.assertEqual(alloc.available_size(), N * PS)

    def test_least_full_root_seeding(self):
        """Roots draw from the class with the most free pages (ties: lowest
        id). Uniform 1-page roots therefore spread with skew <= 1 — the
        oblivious policies (fixed class, round-robin from a fixed origin)
        the design rejects concentrate on one class and drift."""
        alloc = _make_allocator(pages_per_rank=32)
        for i in range(2 * N + 1):
            base = alloc.least_full_class()
            _alloc_extend(alloc, 0, PS, rotation_base=base)
            counts = alloc.class_free_page_counts()
            self.assertLessEqual(max(counts) - min(counts), 1, counts)
        # 9 single-page roots at N=4: classes filled 3,2,2,2.
        self.assertEqual(alloc.class_free_page_counts(), [29, 30, 30, 30])

    def test_chain_rotation_run_property(self):
        """Within one chain (root + arbitrary ps-aligned extensions) the
        owners are exactly cyclic, so per-rank owned page counts differ by
        <= 1 — the padded-allgather block contract ceil(K/N). Guards the
        class-interleave scatter in alloc_extend."""
        for shard_size in (2, 4, 8):
            alloc = PageInterleavePoolAllocator(
                size=256 * PS,
                physical_page_size=PS,
                shard_size=shard_size,
                dtype=torch.bfloat16,
                device="cpu",
                kvcache=None,
                need_sort=False,
            )
            lens = [3 * PS, 5 * PS, PS, 7 * PS]  # chunked extensions
            base = alloc.least_full_class()
            chain = []
            prefix = 0
            for ext in lens:
                out = _alloc_extend(alloc, prefix, prefix + ext, base)
                chain.append(out)
                prefix += ext
            locs = torch.cat(chain)
            pages = locs[::PS] // PS
            owners = pages % shard_size
            expect = torch.arange(pages.numel()) % shard_size
            self.assertTrue(torch.equal(owners, (int(owners[0]) + expect) % shard_size))
            per_rank = torch.bincount(owners, minlength=shard_size)
            self.assertLessEqual(int(per_rank.max() - per_rank.min()), 1)

    def test_free_splits_by_owner_class(self):
        alloc = _make_allocator()
        out = _alloc_extend(alloc, 0, 6 * PS, rotation_base=1)
        before = alloc.class_free_page_counts()
        # Free pages 2 and 3 of the chain (owners 3 and 0) in one call, via
        # the free-group batching path the scheduler uses.
        alloc.free_group_begin()
        alloc.free(out[2 * PS : 3 * PS])
        alloc.free(out[3 * PS : 4 * PS])
        alloc.free_group_end()
        after = alloc.class_free_page_counts()
        deltas = [a - b for a, b in zip(after, before)]
        self.assertEqual(deltas, [1, 0, 0, 1])  # classes (1+2)%4=3 and (1+3)%4=0

    def test_backup_restore_roundtrip(self):
        alloc = _make_allocator()
        out = _alloc_extend(alloc, 0, 3 * PS, rotation_base=2)
        state = alloc.backup_state()
        counts = alloc.class_free_page_counts()
        alloc.free(out)
        self.assertNotEqual(alloc.class_free_page_counts(), counts)
        alloc.restore_state(state)
        self.assertEqual(alloc.class_free_page_counts(), counts)

    def test_need_sort_merges_per_class(self):
        alloc = _make_allocator(pages_per_rank=4, need_sort=True)
        out = _alloc_extend(alloc, 0, 4 * N * PS, rotation_base=0)  # everything
        self.assertEqual(alloc.available_size(), 0)
        alloc.free(out)  # lands in the per-class release lists
        self.assertEqual(alloc.available_size(), 4 * N * PS)
        # A fresh draw forces the per-class merge+sort and succeeds.
        out2 = _alloc_extend(alloc, 0, N * PS, rotation_base=0)
        self.assertIsNotNone(out2)
        pages = out2[::PS] // PS
        self.assertTrue(torch.equal(pages % N, torch.arange(N) % N))

    def test_unsupported_paths_fail_loud(self):
        alloc = _make_allocator()
        with self.assertRaises(NotImplementedError):
            alloc.alloc(GS)
        with self.assertRaises(NotImplementedError):
            alloc.alloc_decode(
                torch.tensor([PS + 1]), torch.tensor([PS + 1]), torch.tensor([PS - 1])
            )
        with self.assertRaises(AssertionError):
            # bs > 1 is out of v1 scope and must fail loud (R9).
            alloc.alloc_extend(
                prefix_lens=torch.tensor([0, 0]),
                prefix_lens_cpu=torch.tensor([0, 0]),
                seq_lens=torch.tensor([PS, PS]),
                seq_lens_cpu=torch.tensor([PS, PS]),
                last_loc=torch.tensor([-1, -1]),
                extend_num_tokens=2 * PS,
                rotation_base=0,
            )


class TestEvictUntilAllocatable(CustomTestCase):
    """The evict-then-allocate contract under min-class accounting: one
    evict() sized in tokens can raise the tight class by less than the
    tokens it freed (evicted pages spread across classes), so the alloc
    path iterates. Guards the two termination conditions of
    _evict_until_allocatable."""

    def _allocator_with_tight_class(self):
        alloc = _make_allocator(pages_per_rank=4)
        # Four 1-page chains, all in class 3: the tight class.
        outs = [_alloc_extend(alloc, 0, PS, rotation_base=3) for _ in range(4)]
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

    def test_iterates_until_min_class_covers(self):
        from sglang.srt.mem_cache.common import _evict_until_allocatable

        alloc, outs = self._allocator_with_tight_class()
        # Each round frees ONE class-3 page (a whole 1-page chain): reaching
        # a min-class floor of 2 pages takes 2 rounds.
        tree = self._tree_stub(alloc, list(outs))
        _evict_until_allocatable(tree, alloc, 2 * N * PS)
        self.assertGreaterEqual(alloc.available_size(), 2 * N * PS)
        self.assertEqual(tree.calls, 2)

    def test_terminates_when_tree_dry(self):
        from sglang.srt.mem_cache.common import _evict_until_allocatable

        alloc, _ = self._allocator_with_tight_class()
        tree = self._tree_stub(alloc, [])  # nothing evictable
        _evict_until_allocatable(tree, alloc, PS)
        self.assertEqual(alloc.available_size(), 0)  # need unmet, but no hang
        self.assertEqual(tree.calls, 1)


class TestRadixRotationBase(CustomTestCase):
    """The host rotation base on TreeNode: the one new piece of metadata.
    node.value is a device tensor, so the base must survive inserts and
    splits purely host-side or the alloc path gains a D2H sync."""

    def _tree(self):
        return RadixCache.create_simulated(page_size=4)

    def test_insert_stamps_split_copies(self):
        tree = self._tree()
        key = RadixKey(array("q", range(12)))
        tree.insert(InsertParams(key=key, value=torch.arange(12), rotation_base=2))
        # A shorter lookup splits the node at the match boundary: BOTH halves
        # keep the chain's base (position-page P keeps owner (b+P)%N on both
        # sides of any split).
        probe = RadixKey(array("q", list(range(8)) + [99, 98, 97, 96]))
        res = tree.match_prefix(MatchPrefixParams(key=probe))
        node = res.last_device_node
        self.assertEqual(node.rotation_base, 2)
        (child,) = node.children.values()
        self.assertEqual(child.rotation_base, 2)

    def test_new_chain_gets_its_own_base(self):
        tree = self._tree()
        tree.insert(
            InsertParams(
                key=RadixKey(array("q", range(8))),
                value=torch.arange(8),
                rotation_base=1,
            )
        )
        tree.insert(
            InsertParams(
                key=RadixKey(array("q", range(100, 108))),
                value=torch.arange(8),
                rotation_base=3,
            )
        )
        r1 = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", range(8)))))
        r2 = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", range(100, 108))))
        )
        self.assertEqual(r1.last_device_node.rotation_base, 1)
        self.assertEqual(r2.last_device_node.rotation_base, 3)

    def test_extension_tail_node_stamped_from_request(self):
        tree = self._tree()
        tree.insert(
            InsertParams(
                key=RadixKey(array("q", range(8))),
                value=torch.arange(8),
                rotation_base=1,
            )
        )
        # A longer insert of the same chain dedups the prefix and stamps the
        # tail node with the (same, chain-constant) base.
        tree.insert(
            InsertParams(
                key=RadixKey(array("q", range(16))),
                value=torch.arange(16),
                rotation_base=1,
            )
        )
        res = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", range(16)))))
        self.assertEqual(res.last_device_node.rotation_base, 1)

    def test_unsharded_inserts_keep_none(self):
        tree = self._tree()
        tree.insert(InsertParams(key=RadixKey(array("q", range(8)))))
        res = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", range(8)))))
        self.assertIsNone(res.last_device_node.rotation_base)


class _GraftReq:
    """Minimal Req stand-in for cache_unfinished/finished_req."""

    def __init__(self, fill_ids, req_pool_idx=0):
        self.fill_ids = list(fill_ids)
        self.origin_input_ids = array("q", fill_ids)
        self.output_ids = array("q", [])
        self.req_pool_idx = req_pool_idx
        self.extra_key = None
        self.cache_protected_len = 0
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.last_node = None
        self.priority = 0
        self.kv_rotation_base = None

    def get_fill_ids(self):
        return array("q", self.fill_ids)


class TestRotationGraftDecline(CustomTestCase):
    """Bug regression (adversarial-review finding, 2026-07-20): the overlap
    disagg-prefill loop plans batch t+1 before batch t's radix insert lands,
    so two requests sharing a prefix can allocate under different rotation
    bases; the second one's insert then dedups the shared prefix and used to
    GRAFT its differently-rotated tail under the first chain. The grafted
    path's page owners are not one cyclic run, so a later reader either
    crashes on a negative allgather pad or silently reads the wrong rank's
    scratch rows. Inserts must refuse the graft instead."""

    PS = 4  # tree quantum for these tests

    def _tree_with_pools(self):
        from unittest.mock import MagicMock

        allocator = MagicMock()
        allocator.device = torch.device("cpu")
        req_to_token = torch.zeros(4, 64, dtype=torch.int64)
        pool = MagicMock()
        pool.req_to_token = req_to_token
        pool.write = lambda idx, values: req_to_token.__setitem__(idx, values)
        tree = RadixCache.create_simulated(
            mock_allocator=allocator, page_size=self.PS
        )
        tree.req_to_token_pool = pool
        return tree, allocator, req_to_token

    def _seed_chain(self, tree, tokens, base):
        tree.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=torch.arange(1000, 1000 + len(tokens)),
                rotation_base=base,
            )
        )

    def test_foreign_base_tail_declined(self):
        tree = RadixCache.create_simulated(page_size=self.PS)
        self._seed_chain(tree, list(range(12)), base=1)
        # Same 8-token prefix, different suffix, allocated under base 3.
        key = RadixKey(array("q", list(range(8)) + [90, 91, 92, 93]))
        res = tree.insert(
            InsertParams(key=key, value=torch.arange(12), rotation_base=3)
        )
        self.assertTrue(res.rotation_tail_declined)
        self.assertEqual(res.prefix_len, 8)
        # The suffix is NOT cached: a full-key match stops at the seam.
        m = tree.match_prefix(MatchPrefixParams(key=key))
        self.assertEqual(len(m.device_indices), 8)

    def test_same_base_tail_attaches(self):
        tree = RadixCache.create_simulated(page_size=self.PS)
        self._seed_chain(tree, list(range(12)), base=1)
        key = RadixKey(array("q", list(range(8)) + [90, 91, 92, 93]))
        res = tree.insert(
            InsertParams(key=key, value=torch.arange(12), rotation_base=1)
        )
        self.assertFalse(res.rotation_tail_declined)
        m = tree.match_prefix(MatchPrefixParams(key=key))
        self.assertEqual(len(m.device_indices), 12)

    def test_cache_unfinished_decline_keeps_request_on_own_pages(self):
        tree, allocator, req_to_token = self._tree_with_pools()
        self._seed_chain(tree, list(range(8)), base=1)
        req = _GraftReq(list(range(8)) + [90, 91, 92, 93])
        req.kv_rotation_base = 3
        own_locs = torch.arange(500, 512, dtype=torch.int64)
        req_to_token[0, :12] = own_locs
        tree.cache_unfinished_req(req)
        # No dedup free, no rebind: the request keeps its own locs whole.
        allocator.free.assert_not_called()
        self.assertTrue(torch.equal(req.prefix_indices, own_locs))
        self.assertEqual(req.cache_protected_len, 0)
        self.assertTrue(torch.equal(req_to_token[0, :12], own_locs))

    def test_cache_finished_decline_frees_duplicates_and_suffix(self):
        tree, allocator, req_to_token = self._tree_with_pools()
        self._seed_chain(tree, list(range(8)), base=1)
        req = _GraftReq(list(range(8)) + [90, 91, 92, 93])
        req.kv_rotation_base = 3
        own_locs = torch.arange(500, 512, dtype=torch.int64)
        req_to_token[0, :12] = own_locs
        tree.cache_finished_req(req, kv_len_to_handle=12)
        freed = torch.cat(
            [torch.as_tensor(c.args[0]) for c in allocator.free.call_args_list]
        )
        # Everything past the protected prefix is released: the duplicates of
        # the matched region AND the declined tail (nothing leaks, nothing is
        # grafted).
        self.assertEqual(set(freed.tolist()), set(own_locs.tolist()))
        m = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", req.fill_ids)))
        )
        self.assertEqual(len(m.device_indices), 8)


def _chain_pages(base, n_pages, local_start=5):
    """Logical page ids of one chain: page P has owner (base + P) % N and an
    arbitrary (here: increasing) local page on its owner."""
    counter = {r: local_start for r in range(N)}
    pages = []
    for p in range(n_pages):
        r = (base + p) % N
        pages.append(counter[r] * N + r)
        counter[r] += 1
    return pages


def _chain_row(pages, seq_len):
    row = torch.empty(seq_len, dtype=torch.int32)
    for i in range(seq_len):
        row[i] = pages[i // PS] * PS + i % PS
    return row


def _make_pool_stub(spec, shard_rank=0, debug=True, table_pages=4096):
    """A SimpleNamespace carrying exactly the state begin_shard_extend /
    translate_loc_to_scratch read."""
    stub = SimpleNamespace()
    stub.shard_spec = spec
    stub.shard_rank = shard_rank
    stub.device = "cpu"
    stub.start_layer = 0
    stub._chunk_base = spec.max_prefix_tokens
    stub._trash_base = spec.max_prefix_tokens + spec.chunk_tokens
    stub._page_pos = torch.full((table_pages,), -1, dtype=torch.int32)
    stub._epoch = 0
    stub._write_plan_key = stub._write_plan = None
    stub._debug_plan_checks = debug
    stub.prefetched = []
    stub._prefetch_layer = lambda layer_id: stub.prefetched.append(layer_id)
    return stub


def _run_begin(stub, prefix_len, seq_len, row):
    PageInterleaveKVPoolMixin.begin_shard_extend(
        stub, row.unsqueeze(0), torch.tensor([0]), [prefix_len], [seq_len]
    )
    return stub


class TestBeginShardExtendPlan(CustomTestCase):
    def test_plan_with_rotated_prefix(self):
        """7 prefix pages of a base-2 chain + 9 chunk pages (last partial):
        plan positions in position order, send rows owner-filtered in
        position order and padded to ceil(7/4)=2 pages."""
        pages = _chain_pages(base=2, n_pages=16)
        prefix_len, seq_len = 7 * PS, 16 * PS - 5
        row = _chain_row(pages, seq_len)
        for rank in range(N):
            stub = _run_begin(
                _make_pool_stub(_make_spec(), rank), prefix_len, seq_len, row
            )
            self.assertEqual(stub._n_prefix_pages, 7)
            self.assertEqual(stub._block_pages, 2)
            self.assertTrue(stub._prefix_active)
            self.assertTrue(stub._shard_extend_active)
            self.assertEqual(stub._epoch, 1)
            self.assertEqual(stub.prefetched, [0])  # first layer kicked
            for k, page in enumerate(pages[:7]):
                self.assertEqual(int(stub._page_pos[page]), k)
            for k, page in enumerate(pages[7:]):
                self.assertEqual(int(stub._page_pos[page]), 7 + k)
            own = [p for p in pages[:7] if p % N == rank]
            expect = torch.cat(
                [torch.arange((p // N) * PS, (p // N + 1) * PS) for p in own]
            )
            if len(own) < 2:  # padded with the trash page (local page 0)
                expect = torch.cat([expect, torch.arange(PS)])
            self.assertTrue(torch.equal(stub._send_rows, expect))

    def test_plan_without_prefix(self):
        pages = _chain_pages(base=0, n_pages=2)
        stub = _run_begin(
            _make_pool_stub(_make_spec()), 0, PS + 5, _chain_row(pages, PS + 5)
        )
        self.assertEqual(stub._n_prefix_pages, 0)
        self.assertFalse(stub._prefix_active)
        self.assertTrue(stub._shard_extend_active)
        self.assertEqual(stub.prefetched, [])  # nothing to gather
        self.assertIsNone(stub._send_rows)
        self.assertEqual(int(stub._page_pos[pages[0]]), 0)
        self.assertEqual(int(stub._page_pos[pages[1]]), 1)

    def test_unaligned_prefix_rejected(self):
        # The tree quantum is the PHYSICAL page: a prefix that is not a
        # ps-multiple can never come out of match_prefix.
        pages = _chain_pages(base=0, n_pages=4)
        with self.assertRaises(AssertionError):
            _run_begin(
                _make_pool_stub(_make_spec()),
                PS + 3,
                4 * PS,
                _chain_row(pages, 4 * PS),
            )

    def test_multi_request_batch_rejected(self):
        pages = _chain_pages(base=0, n_pages=2)
        row = _chain_row(pages, 2 * PS)
        stub = _make_pool_stub(_make_spec())
        with self.assertRaises(AssertionError):
            PageInterleaveKVPoolMixin.begin_shard_extend(
                stub,
                torch.stack([row, row]),
                torch.tensor([0, 1]),
                [0, 0],
                [PS, PS],
            )

    def test_owner_congruence_guard(self):
        """A rotation-phase bug that breaks prefix-owner cyclicity makes the
        arithmetic within-owner index (k // N) disagree with the send
        layout; translation would silently address the next rank's block.
        The debug guard must catch it at plan time."""
        pages = _chain_pages(base=1, n_pages=8)
        pages[2], pages[5] = pages[5], pages[2]  # same multiset, not cyclic
        with self.assertRaises(AssertionError) as ctx:
            _run_begin(
                _make_pool_stub(_make_spec()),
                6 * PS,
                8 * PS,
                _chain_row(pages, 8 * PS),
            )
        self.assertIn("cyclic", str(ctx.exception))


class TestScratchTranslation(CustomTestCase):
    def _plan(self, base=2, n_prefix=7, n_chunk=9, rank=1):
        pages = _chain_pages(base=base, n_pages=n_prefix + n_chunk)
        seq_len = (n_prefix + n_chunk) * PS
        stub = _run_begin(
            _make_pool_stub(_make_spec(), rank),
            n_prefix * PS,
            seq_len,
            _chain_row(pages, seq_len),
        )
        return stub, pages[:n_prefix], pages[n_prefix:]

    def _reference_row(self, stub, prefix_pages, chunk_pages, loc):
        """Brute-force reference: owner-major prefix, sequence-order chunk."""
        spec = stub.shard_spec
        page, off = loc // PS, loc % PS
        if page in prefix_pages:
            k = prefix_pages.index(page)
            block = stub._block_pages * PS
            return (page % N) * block + (k // N) * PS + off
        if page in chunk_pages:
            k = chunk_pages.index(page)
            return spec.max_prefix_tokens + k * PS + off
        return stub._trash_base + off

    def test_translation_matches_reference(self):
        stub, prefix_pages, chunk_pages = self._plan()
        locs = (
            [p * PS + o for p in prefix_pages + chunk_pages for o in (0, 3, PS - 1)]
            + list(range(0, N))  # reserved pages -> trash
            + [3000, 3001]  # off-plan -> trash
        )
        got = PageInterleaveKVPoolMixin.translate_loc_to_scratch(
            stub, torch.tensor(locs, dtype=torch.int64)
        )
        expect = torch.tensor(
            [self._reference_row(stub, prefix_pages, chunk_pages, l) for l in locs],
            dtype=torch.int64,
        )
        self.assertTrue(torch.equal(got, expect))

    def test_translation_is_injective_over_the_plan(self):
        stub, prefix_pages, chunk_pages = self._plan(base=3, n_prefix=5, n_chunk=4)
        locs = [p * PS + o for p in prefix_pages + chunk_pages for o in range(PS)]
        rows = PageInterleaveKVPoolMixin.translate_loc_to_scratch(
            stub, torch.tensor(locs, dtype=torch.int64)
        )
        self.assertEqual(len(torch.unique(rows)), len(locs))
        # Prefix rows stay inside the (padded) gather span, chunk rows inside
        # the chunk region.
        n_prefix_tokens = len(prefix_pages) * PS
        self.assertTrue(
            bool((rows[:n_prefix_tokens] < N * stub._block_pages * PS).all())
        )
        self.assertTrue(
            bool(
                (rows[n_prefix_tokens:] >= stub.shard_spec.max_prefix_tokens).all()
                and (rows[n_prefix_tokens:] < stub._trash_base).all()
            )
        )

    def test_int32_page_table_input(self):
        stub, prefix_pages, chunk_pages = self._plan(base=0, n_prefix=4, n_chunk=1)
        table = torch.tensor(
            [prefix_pages[0] * PS, prefix_pages[1] * PS, chunk_pages[0] * PS, 0],
            dtype=torch.int32,
        )
        rows = PageInterleaveKVPoolMixin.translate_loc_to_scratch(stub, table)
        self.assertEqual(rows.dtype, torch.int64)
        # Page-aligned inputs land on page-aligned scratch rows (the FA3
        # stride-divide contract).
        self.assertTrue(bool((rows[:3] % PS == 0).all()))
        self.assertEqual(int(rows[3]), stub._trash_base)


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

    def test_partial_tail_page_may_own_nothing(self):
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
    """Value-derived P/D ownership partition: owner = logical page % N, wire
    value = logical page // N — independent of positions, so it stays
    correct under any placement policy (rotation, future re-seeds)."""

    def test_partition_and_positional_pairing(self):
        mgr = SimpleNamespace(kv_shard_rank=0, kv_shard_size=N)
        # A base-2 rotated chain: owners are NOT position-congruent.
        logical_pages = np.array(_chain_pages(base=2, n_pages=9), dtype=np.int64)
        sl = slice(0, 9)
        got = {}
        for r in range(N):
            mgr.kv_shard_rank = r
            wire, pos = filter_kv_indices_for_shard_rank(mgr, logical_pages, sl)
            got[r] = (wire, pos)
            # Ownership by VALUE (l % N == r), wire id = owner's local page.
            np.testing.assert_array_equal(logical_pages[pos] % N, r)
            np.testing.assert_array_equal(wire, logical_pages[pos] // N)
        # Disjoint cover of all positions.
        all_pos = np.sort(np.concatenate([got[r][1] for r in range(N)]))
        np.testing.assert_array_equal(all_pos, np.arange(9))

    def test_wire_value_matches_unrotated_scheme(self):
        """The wire byte contract: l // N == loc // (N * ps) — the same
        owner-local page id the position-congruent scheme sent, so decode
        needs no change."""
        mgr = SimpleNamespace(kv_shard_rank=1, kv_shard_size=N)
        logical_pages = np.array(_chain_pages(base=0, n_pages=8), dtype=np.int64)
        locs = logical_pages * PS  # page-head loc of each entry
        wire, pos = filter_kv_indices_for_shard_rank(mgr, logical_pages, slice(0, 8))
        np.testing.assert_array_equal(wire, locs[pos] // (N * PS))

    def test_mid_request_chunk_positions_canonical(self):
        mgr = SimpleNamespace(kv_shard_rank=2, kv_shard_size=N)
        logical_pages = np.array(
            _chain_pages(base=1, n_pages=32)[20:32], dtype=np.int64
        )
        wire, pos = filter_kv_indices_for_shard_rank(mgr, logical_pages, slice(20, 32))
        # Positions are canonical (chunk-global), values from this chunk.
        self.assertTrue(((pos >= 20) & (pos < 32)).all())
        np.testing.assert_array_equal(logical_pages[pos - 20] % N, 2)
        np.testing.assert_array_equal(wire, logical_pages[pos - 20] // N)

    def test_rank_owning_nothing_in_chunk(self):
        """A short chunk can leave a rank with zero pages — the case whose
        skipped NIXL notif hung the receiver (the prep-commit fix sends the
        empty chunk notif standalone). The filter itself must return empty
        arrays, not error."""
        mgr = SimpleNamespace(kv_shard_rank=3, kv_shard_size=N)
        pages = np.array(_chain_pages(base=0, n_pages=2), dtype=np.int64)  # owners 0,1
        wire, pos = filter_kv_indices_for_shard_rank(mgr, pages, slice(0, 2))
        self.assertEqual(len(wire), 0)
        self.assertEqual(len(pos), 0)


if __name__ == "__main__":
    unittest.main()
