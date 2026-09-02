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

Pins the pure arithmetic that rotated owner-classed allocation hangs on:

1. The placement bijection ``loc = Q*(N*ps) + r*ps + o`` — owner / local-row
   round-trip, disjoint equal partition across ranks.
2. ``PageInterleavePoolAllocator`` — N mirrored class free lists, rotated
   class draws (owners exactly cyclic along a chain), least-full root
   seeding, min-class admission accounting, zero stranding (a freed page is
   immediately reusable).
3. The host rotation base on radix ``TreeNode`` — stamped at insert, copied
   on split, read through ``last_node``.
"""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator.page_interleave import (
    PageInterleavePoolAllocator,
    page_interleave_shard_size,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.page_interleave import (
    PageInterleavePlacement,
    PageShardSpec,
)
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


def _alloc_extend_batch(alloc, prefix_lens, seq_lens, rotation_bases, last_locs=None):
    """Drive alloc_extend for a batch; rotation_bases is resolved in place."""
    if last_locs is None:
        last_locs = [-1] * len(prefix_lens)
    return alloc.alloc_extend(
        prefix_lens=torch.tensor(prefix_lens, dtype=torch.int64),
        prefix_lens_cpu=torch.tensor(prefix_lens, dtype=torch.int64),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
        last_loc=torch.tensor(last_locs, dtype=torch.int64),
        extend_num_tokens=sum(s - p for p, s in zip(prefix_lens, seq_lens)),
        rotation_bases=rotation_bases,
    )


def _alloc_extend(alloc, prefix_len, seq_len, rotation_base, last_loc=-1):
    return _alloc_extend_batch(
        alloc, [prefix_len], [seq_len], [rotation_base], [last_loc]
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
        """Two-turn worked example at ps=16: turn 1 allocates cyclic owners
        from the root base; the turn-boundary free returns its page whole and
        immediately reusable; turn 2 continues the rotation and reuses the
        freed page before any fresh one."""
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
        RuntimeError when the tight class is protected."""
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
        id). Uniform 1-page roots therefore spread with skew <= 1."""
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

    def test_grouped_free_owns_indices_before_caller_mutation(self):
        """Deferred frees must snapshot req_to_token views: the scheduler may
        overwrite the backing row before free_group_end consumes them."""
        alloc = _make_allocator()
        out = _alloc_extend(alloc, 0, 2 * PS, rotation_base=2)
        first_page = out[:PS]
        owner = int(first_page[0] // PS % N)
        before = alloc.class_free_page_counts()

        alloc.free_group_begin()
        alloc.free(first_page)
        first_page.zero_()
        alloc.free_group_end()

        after = alloc.class_free_page_counts()
        self.assertEqual(after[owner], before[owner] + 1)
        self.assertEqual(
            [after[r] - before[r] for r in range(N)],
            [1 if r == owner else 0 for r in range(N)],
        )

    def test_free_segment_returns_pages_to_their_classes(self):
        # The radix cache frees through free_segment/free_segments. The paged
        # base routes those to the stock free_pages list, which this allocator
        # never reads, so the override must land them in the class lists.
        alloc = _make_allocator()
        total = alloc.available_size()
        out = _alloc_extend(alloc, 0, 3 * PS, rotation_base=2)
        self.assertLess(alloc.available_size(), total)
        alloc.free_segment(out, start_pos=0)
        self.assertEqual(alloc.available_size(), total)
        self.assertEqual(alloc.class_free_page_counts(), [32] * N)

    def test_free_segments_splits_at_a_page_boundary(self):
        alloc = _make_allocator()
        total = alloc.available_size()
        out = _alloc_extend(alloc, 0, 4 * PS, rotation_base=0)
        # Two disjoint ascending segments of one request's kv row.
        alloc.free_segments([(out[: 2 * PS], 0), (out[2 * PS :], 2 * PS)])
        self.assertEqual(alloc.available_size(), total)
        self.assertEqual(alloc.class_free_page_counts(), [32] * N)

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

    def test_batch_alloc_per_request_rotation(self):
        """bs > 1: each request draws its own cyclic run; out_cache_loc is
        the batch-order concatenation (write_cache_indices' contract), and a
        None base is resolved from the least-full class AT THAT REQUEST'S
        TURN — the draw must see earlier requests' pops in the same batch,
        or uniform short batches would all pile onto one class."""
        alloc = _make_allocator()
        # req0: extension of a base-1 chain with a 2-page prefix;
        # req1 and req2: new chains (drawn in place).
        bases = [1, None, None]
        out = _alloc_extend_batch(
            alloc,
            prefix_lens=[2 * PS, 0, 0],
            seq_lens=[5 * PS, 3 * PS, PS],
            rotation_bases=bases,
            # req0's last prefix page must carry owner (1 + 1) % 4 = 2.
            last_locs=[(5 * N + 2) * PS + PS - 1, -1, -1],
        )
        self.assertEqual(out.numel(), 3 * PS + 3 * PS + PS)
        # Batch-order concatenation, per-request cyclic owners.
        pages = out[::PS] // PS
        self.assertTrue(
            torch.equal(pages[:3] % N, (1 + 2 + torch.arange(3)) % N)  # req0
        )
        b1, b2 = bases[1], bases[2]
        self.assertIsNotNone(b1)
        self.assertIsNotNone(b2)
        self.assertTrue(torch.equal(pages[3:6] % N, (b1 + torch.arange(3)) % N))
        self.assertEqual(int(pages[6]) % N, b2)
        # req1's draw saw req0's pops (classes 3,0,1 used once each -> class
        # 2 is fullest... all equal except used {3,0,1} -> least-full = 2);
        # req2's draw saw req1's pops on top.
        self.assertEqual(b1, 2)
        self.assertEqual(b2, 1)  # after req1 used {2,3,0}: class 1 fullest
        # The whole batch is one no-duplicate allocation.
        self.assertEqual(len(torch.unique(out)), out.numel())

    def test_batch_alloc_defers_whole_when_a_class_is_short(self):
        """A batch either commits whole or returns None (mirrored decision):
        partial commits would desync the free lists from the retry."""
        alloc = _make_allocator(pages_per_rank=2)
        counts_before = alloc.class_free_page_counts()
        out = _alloc_extend_batch(
            alloc,
            prefix_lens=[0, 0],
            seq_lens=[4 * PS, 5 * PS],  # 9 pages: class need exceeds 2 somewhere
            rotation_bases=[0, 0],
        )
        self.assertIsNone(out)
        self.assertEqual(alloc.class_free_page_counts(), counts_before)


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
        self.kv = SimpleNamespace(req_pool_idx=req_pool_idx, cache_protected_len=0)
        self.extra_key = None
        self.cache_salt = None
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.last_node = None
        self.priority = 0
        self.kv_rotation_base = None

    def get_fill_ids(self):
        return array("q", self.fill_ids)


class TestRotationGraftDecline(CustomTestCase):
    """The overlap disagg-prefill loop plans batch t+1 before batch t's radix
    insert lands, so two requests sharing a prefix can allocate under
    different rotation bases. Grafting the second one's tail under the first
    chain leaves the cached path's page owners not one cyclic run, so a later
    reader either crashes on a negative allgather pad or silently reads the
    wrong rank's scratch rows. Inserts must refuse the graft."""

    PS = 4  # tree quantum for these tests

    def _tree_with_pools(self):
        from unittest.mock import MagicMock

        allocator = MagicMock()
        allocator.device = torch.device("cpu")
        req_to_token = torch.zeros(4, 64, dtype=torch.int64)
        pool = MagicMock()
        pool.req_to_token = req_to_token
        pool.write = lambda idx, values: req_to_token.__setitem__(idx, values)
        tree = RadixCache.create_simulated(mock_allocator=allocator, page_size=self.PS)
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

    def test_empty_page_aligned_key_insert(self):
        """A finished request with fewer cached tokens than one tree page
        inserts an EMPTY page-aligned key; the empty-key early return must
        match insert()'s 3-tuple unpack."""
        tree = RadixCache.create_simulated(page_size=self.PS)
        res = tree.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2])),
                value=torch.arange(2),
                rotation_base=1,
            )
        )
        self.assertEqual(res.prefix_len, 0)
        self.assertFalse(res.rotation_tail_declined)

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
        self.assertEqual(req.kv.cache_protected_len, 0)
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
            [
                torch.as_tensor(seg)
                for call in allocator.free_segments.call_args_list
                for seg, _start_pos in call.args[0]
            ]
        )
        # Everything past the protected prefix is released: the duplicates of
        # the matched region AND the declined tail (nothing leaks, nothing is
        # grafted).
        self.assertEqual(set(freed.tolist()), set(own_locs.tolist()))
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", req.fill_ids))))
        self.assertEqual(len(m.device_indices), 8)


if __name__ == "__main__":
    unittest.main()
