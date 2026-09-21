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
"""Tests for logical-page KV cache sharding.

Two sections. The first (CPU only, what the CPU CI job runs) pins the pure
arithmetic that rotated owner-classed allocation hangs on:

1. The placement bijection ``loc = Q*(N*ps) + r*ps + o`` — owner / local-row
   round-trip, disjoint equal partition across ranks.
2. ``PageInterleavePoolAllocator`` — N mirrored class free lists, rotated
   class draws (owners exactly cyclic along a chain), least-full root
   seeding, min-class admission accounting, zero stranding (a freed page is
   immediately reusable).
3. The host rotation base on ``UnifiedTreeNode`` — stamped at insert, copied
   on split, read through ``last_node``, and the pre-flight that declines an
   insert whose pages carry a different base than the chain it would join.
4. ``translate_loc_to_scratch`` — the per-batch page->scratch-page lookup mapping
   any consumer index vector onto the owner-major ``[prefix | chunk | trash]``
   scratch, checked against a brute-force reference.
5. ``begin_shard_extend`` plan capture (page positions, padded send rows,
   owner-congruence guard) with the gather stubbed out, following the
   SimpleNamespace binding pattern of ``test_dsa_layer_shard_utils.py``.

The second section (``TestPageInterleaveGatherMultiGpu``, at the bottom) drives
real pools over a real 2-rank process group. It is the only check that the plan
the CPU stub validates actually addresses the bytes NCCL delivers, so it is
skipped rather than dropped when fewer than 2 CUDA devices are visible — which
is every run of the CPU suite this file is registered to.
"""

import os
import unittest
import unittest.mock
from array import array
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.mem_cache.allocator.page_interleave import (
    PageInterleavePoolAllocator,
    page_interleave_shard_size,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictResult,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import _evict_until_allocatable
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.page_interleave import (
    PageInterleavePlacement,
    PageShardSpec,
    get_kv_shard_group,
)
from sglang.srt.mem_cache.page_interleave_pool import (
    PageInterleaveKVPoolMixin,
    PageInterleaveMHATokenToKVPool,
    PageInterleaveMLATokenToKVPool,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_parallel, publish
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import ceil_div
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

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
        alloc, outs = self._allocator_with_tight_class()
        # Each round frees ONE class-3 page (a whole 1-page chain): reaching
        # a min-class floor of 2 pages takes 2 rounds.
        tree = self._tree_stub(alloc, list(outs))
        _evict_until_allocatable(tree, alloc, 2 * N * PS)
        self.assertGreaterEqual(alloc.available_size(), 2 * N * PS)
        self.assertEqual(tree.calls, 2)

    def test_terminates_when_tree_dry(self):
        alloc, _ = self._allocator_with_tight_class()
        tree = self._tree_stub(alloc, [])  # nothing evictable
        _evict_until_allocatable(tree, alloc, PS)
        self.assertEqual(alloc.available_size(), 0)  # need unmet, but no hang
        self.assertEqual(tree.calls, 1)


def _unified_tree(page_size=4, pool_size=256, disable=False):
    """A CPU-only UnifiedRadixCache with only the Full component."""
    dtype = torch.float16
    kv_pool = MHATokenToKVPool(
        size=pool_size,
        page_size=page_size,
        dtype=dtype,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = PagedTokenToKVPoolAllocator(
        size=pool_size,
        page_size=page_size,
        dtype=dtype,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    req_pool = ReqToTokenPool(
        size=8,
        max_context_len=128,
        device="cpu",
        enable_memory_saver=False,
    )
    return UnifiedRadixCache(
        CacheInitParams(
            disable=disable,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            eviction_policy="lru",
            tree_components=(ComponentType.FULL,),
        )
    )


def _insert(tree, tokens, rotation_base=None, value=None):
    if value is None:
        value = tree.token_to_kv_pool_allocator.alloc(len(tokens))
    return tree.insert(
        InsertParams(
            key=RadixKey(array("q", tokens)),
            value=value.to(dtype=torch.int64),
            rotation_base=rotation_base,
        )
    )


def _node(tree, node_id):
    return tree.tree_core.node_by_id(node_id)


def _match_len(tree, tokens):
    res = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
    return len(res.device_indices)


class TestUnifiedRotationBase(CustomTestCase):
    """The host rotation base on UnifiedTreeNode: the one new piece of
    metadata. The Full component's value is a device tensor, so the base must
    survive inserts and splits purely host-side or the alloc path gains a D2H
    sync."""

    def test_insert_stamps_split_copies(self):
        tree = _unified_tree()
        _insert(tree, list(range(12)), rotation_base=2)
        # A shorter lookup splits the node at the match boundary: BOTH halves
        # keep the chain's base (position-page P keeps owner (b+P)%N on both
        # sides of any split).
        probe = list(range(8)) + [99, 98, 97, 96]
        _insert(tree, probe, rotation_base=2)
        res = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", probe))))
        tail = _node(tree, res.last_device_node)
        self.assertEqual(tail.rotation_base, 2)
        parent = tail.parent
        self.assertEqual(parent.rotation_base, 2)
        for child in parent.children.values():
            self.assertEqual(child.rotation_base, 2)

    def test_new_chain_gets_its_own_base(self):
        tree = _unified_tree()
        _insert(tree, list(range(8)), rotation_base=1)
        _insert(tree, list(range(100, 108)), rotation_base=3)
        r1 = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", range(8)))))
        r2 = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", range(100, 108))))
        )
        self.assertEqual(_node(tree, r1.last_device_node).rotation_base, 1)
        self.assertEqual(_node(tree, r2.last_device_node).rotation_base, 3)

    def test_extension_tail_node_stamped_from_request(self):
        tree = _unified_tree()
        _insert(tree, list(range(8)), rotation_base=1)
        # A longer insert of the same chain dedups the prefix and stamps the
        # tail node with the (same, chain-constant) base.
        _insert(tree, list(range(16)), rotation_base=1)
        res = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", range(16)))))
        self.assertEqual(_node(tree, res.last_device_node).rotation_base, 1)

    def test_unsharded_inserts_keep_none(self):
        tree = _unified_tree()
        _insert(tree, list(range(8)))
        res = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", range(8)))))
        self.assertIsNone(_node(tree, res.last_device_node).rotation_base)

    def test_rotation_base_of_reads_through_the_cache_boundary(self):
        """The alloc path holds a NodeId, not a node: the base must be
        readable through the tree-cache API (BasePrefixCache.rotation_base_of
        defaults to None, so an unsharded cache sends the alloc path to the
        request's recorded base)."""
        tree = _unified_tree()
        _insert(tree, list(range(8)), rotation_base=2)
        res = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", range(8)))))
        self.assertEqual(tree.rotation_base_of(res.last_device_node), 2)
        self.assertIsNone(tree.rotation_base_of(None))
        self.assertIsNone(tree.rotation_base_of(tree.tree_core.root_node_handle()))


class TestShardedCoreGate(CustomTestCase):
    """A tree core that does not model rotation_base would never decline a
    cross-base graft. Pairing one with a sharded allocator must fail at
    construction, not produce wrong-owner gathers at serve time."""

    def test_python_core_supports_rotation_base(self):
        tree = _unified_tree()
        self.assertTrue(tree.tree_core.supports_rotation_base)

    def test_sharded_allocator_rejects_a_core_without_rotation_base(self):
        tree = _unified_tree()
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=tree.req_to_token_pool,
            token_to_kv_pool_allocator=_make_allocator(pages_per_rank=8),
            page_size=PS,
            eviction_policy="lru",
            tree_components=(ComponentType.FULL,),
        )
        with unittest.mock.patch.object(
            UnifiedTreeCore, "supports_rotation_base", False
        ):
            with self.assertRaisesRegex(ValueError, "rotation bases"):
                UnifiedRadixCache(params)

    def test_unsharded_allocator_accepts_any_core(self):
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=ReqToTokenPool(
                size=8, max_context_len=128, device="cpu", enable_memory_saver=False
            ),
            token_to_kv_pool_allocator=_unified_tree().token_to_kv_pool_allocator,
            page_size=4,
            eviction_policy="lru",
            tree_components=(ComponentType.FULL,),
        )
        with unittest.mock.patch.object(
            UnifiedTreeCore, "supports_rotation_base", False
        ):
            UnifiedRadixCache(params)  # no raise: sharding is off


class _GraftReq:
    """Minimal Req stand-in for cache_unfinished/finished_req."""

    def __init__(self, fill_ids, req_pool_idx=0):
        self.fill_ids = list(fill_ids)
        self.origin_input_ids = array("q", fill_ids)
        self.output_ids = array("q", [])
        self.kv = SimpleNamespace(
            req_pool_idx=req_pool_idx,
            cache_protected_len=0,
            swa_evicted_seqlen=0,
        )
        self.extra_key = None
        self.cache_salt = None
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.last_node = None
        self.priority = 0
        self.kv_rotation_base = None
        self.lock_receipt = DecLockRefParams()
        self.swa_prefix_lock_released = False
        self.finished_reason = None
        self.session = None
        self.session_id = None

    def get_fill_ids(self):
        return array("q", self.fill_ids)


class TestRotationGraftDecline(CustomTestCase):
    """The overlap disagg-prefill loop plans batch t+1 before batch t's radix
    insert lands, so two requests sharing a prefix can allocate under
    different rotation bases. Grafting the second one's tail under the first
    chain leaves the cached path's page owners not one cyclic run, so a later
    reader either crashes on a negative allgather pad or silently reads the
    wrong rank's scratch rows. Inserts must refuse the graft.

    Unlike the flat radix tree, the unified insert also transfers page
    ownership inside the MATCHED region (an evicted node restored from the
    request's fresh pages, a component re-pointing a matched node's Full value
    at them). The decline therefore runs before the walk and refuses the whole
    insert, which additionally keeps the walk's duplicate frees from running
    under a request that is about to keep its own pages.
    """

    PS = 4  # tree quantum for these tests

    def _tree_with_spy(self):
        """Record every KV row release. The unified tree frees through two
        seams: the caller's free_kv_row -> free_segments, and the insert
        walk's FreeDeviceKV -> free_segment."""
        tree = _unified_tree(page_size=self.PS)
        allocator = tree.token_to_kv_pool_allocator
        freed = []
        real_free_segments = allocator.free_segments
        real_free_segment = allocator.free_segment

        def spy_segments(segments):
            freed.extend(torch.as_tensor(seg).clone() for seg, _start in segments)
            return real_free_segments(segments)

        def spy_segment(free_index, *, start_pos):
            freed.append(torch.as_tensor(free_index).clone())
            return real_free_segment(free_index, start_pos=start_pos)

        allocator.free_segments = spy_segments
        allocator.free_segment = spy_segment
        return tree, freed

    def _seed_chain(self, tree, tokens, base):
        _insert(tree, tokens, rotation_base=base)

    def _own_row(self, tree, req, n_tokens):
        """Give the request its own allocated KV row and return the locs."""
        locs = tree.token_to_kv_pool_allocator.alloc(n_tokens).to(dtype=torch.int64)
        tree.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :n_tokens] = locs
        return locs

    def test_foreign_base_tail_declined(self):
        tree = _unified_tree(page_size=self.PS)
        self._seed_chain(tree, list(range(12)), base=1)
        # Same 8-token prefix, different suffix, allocated under base 3.
        key = list(range(8)) + [90, 91, 92, 93]
        res = _insert(tree, key, rotation_base=3)
        self.assertTrue(res.rotation_tail_declined)
        self.assertEqual(res.prefix_len, 8)
        # The suffix is NOT cached: a full-key match stops at the seam.
        self.assertEqual(_match_len(tree, key), 8)

    def test_empty_page_aligned_key_insert(self):
        """A finished request with fewer cached tokens than one tree page
        inserts an EMPTY page-aligned key; the empty-key early return must
        precede the rotation pre-flight."""
        tree = _unified_tree(page_size=self.PS)
        res = _insert(tree, [1, 2], rotation_base=1)
        self.assertEqual(res.prefix_len, 0)
        self.assertFalse(res.rotation_tail_declined)

    def test_same_base_tail_attaches(self):
        tree = _unified_tree(page_size=self.PS)
        self._seed_chain(tree, list(range(12)), base=1)
        key = list(range(8)) + [90, 91, 92, 93]
        res = _insert(tree, key, rotation_base=1)
        self.assertFalse(res.rotation_tail_declined)
        self.assertEqual(_match_len(tree, key), 12)

    def test_no_matched_chain_never_declines(self):
        """A request that matches nothing starts its own chain: the guard
        only fires against an EXISTING chain's base."""
        tree = _unified_tree(page_size=self.PS)
        self._seed_chain(tree, list(range(12)), base=1)
        res = _insert(tree, list(range(50, 62)), rotation_base=3)
        self.assertFalse(res.rotation_tail_declined)
        self.assertEqual(_match_len(tree, list(range(50, 62))), 12)

    def test_unsharded_insert_onto_a_based_chain_never_declines(self):
        """rotation_base=None means sharding is off for this insert: the
        pre-flight must not fire, or every unsharded path would stop caching."""
        tree = _unified_tree(page_size=self.PS)
        self._seed_chain(tree, list(range(8)), base=1)
        key = list(range(8)) + [90, 91, 92, 93]
        res = _insert(tree, key, rotation_base=None)
        self.assertFalse(res.rotation_tail_declined)
        self.assertEqual(_match_len(tree, key), 12)

    def test_full_match_under_a_foreign_base_declines(self):
        """No tail to graft, but the unified insert would still hand the
        request's pages to the matched chain (unevict-on-insert, component
        Full re-point). The pre-flight declines that too."""
        tree = _unified_tree(page_size=self.PS)
        self._seed_chain(tree, list(range(12)), base=1)
        res = _insert(tree, list(range(12)), rotation_base=3)
        self.assertTrue(res.rotation_tail_declined)

    def test_cache_unfinished_decline_keeps_request_on_own_pages(self):
        tree, freed = self._tree_with_spy()
        self._seed_chain(tree, list(range(8)), base=1)
        req = _GraftReq(list(range(8)) + [90, 91, 92, 93])
        req.kv_rotation_base = 3
        own_locs = self._own_row(tree, req, 12)
        tree.cache_unfinished_req(req)
        # No dedup free, no rebind: the request keeps its own locs whole.
        self.assertEqual([t.tolist() for t in freed], [])
        self.assertTrue(torch.equal(req.prefix_indices, own_locs))
        self.assertEqual(req.kv.cache_protected_len, 0)
        self.assertTrue(
            torch.equal(tree.req_to_token_pool.req_to_token[0, :12], own_locs)
        )

    def test_cache_finished_decline_frees_duplicates_and_suffix(self):
        tree, freed = self._tree_with_spy()
        self._seed_chain(tree, list(range(8)), base=1)
        req = _GraftReq(list(range(8)) + [90, 91, 92, 93])
        req.kv_rotation_base = 3
        own_locs = self._own_row(tree, req, 12)
        tree.cache_finished_req(req, owned_kv_len=12)
        released = torch.cat(freed)
        # Everything past the protected prefix is released: the duplicates of
        # the matched region AND the declined tail (nothing leaks, nothing is
        # grafted).
        self.assertEqual(set(released.tolist()), set(own_locs.tolist()))
        self.assertEqual(_match_len(tree, req.fill_ids), 8)

    def test_cache_finished_same_base_keeps_the_tail_cached(self):
        """Control for the decline test: with an agreeing base the tail is
        grafted and only the matched duplicates are freed."""
        tree, freed = self._tree_with_spy()
        self._seed_chain(tree, list(range(8)), base=1)
        req = _GraftReq(list(range(8)) + [90, 91, 92, 93])
        req.kv_rotation_base = 1
        own_locs = self._own_row(tree, req, 12)
        tree.cache_finished_req(req, owned_kv_len=12)
        self.assertEqual(_match_len(tree, req.fill_ids), 12)
        released = torch.cat(freed) if freed else torch.empty(0, dtype=torch.int64)
        # Only the 8 duplicate rows go back; the tail stays live in the tree.
        self.assertEqual(set(released.tolist()), set(own_locs[:8].tolist()))


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
    stub._page_pos = torch.full(
        (table_pages,), stub._trash_base // PS, dtype=torch.int32
    )
    stub._local_page_stride = table_pages
    stub._epoch = 0
    stub._write_plan_key = stub._write_plan = None
    stub._translate_cache = {}
    stub._debug_plan_checks = debug
    stub.translate_loc_to_scratch = lambda loc: (
        PageInterleaveKVPoolMixin.translate_loc_to_scratch(stub, loc)
    )
    stub.prefetched = []
    stub._prefetch_layer = lambda layer_id: stub.prefetched.append(layer_id)
    return stub


def _run_begin(stub, prefix_lens, seq_lens, rows):
    width = max(r.numel() for r in rows)
    padded = [
        torch.cat([r, torch.zeros(width - r.numel(), dtype=torch.int32)]) for r in rows
    ]
    PageInterleaveKVPoolMixin.begin_shard_extend(
        stub,
        torch.stack(padded),
        torch.arange(len(rows)),
        prefix_lens,
        seq_lens,
    )
    return stub


def _reference_prefix_slots(per_request_prefix_pages):
    """Brute-force reference of the owner-major slot assignment: the batch's
    unique prefix pages sorted by (owner, local page), rank r's pages
    contiguous at r * block; block = sum of per-request ceil(K_i / N)."""
    block = sum(ceil_div(len(pages), N) for pages in per_request_prefix_pages)
    uniq = sorted({p for pages in per_request_prefix_pages for p in pages})
    slots = {}
    counts = {r: 0 for r in range(N)}
    for page in sorted(uniq, key=lambda p: (p % N, p // N)):
        owner = page % N
        slots[page] = owner * block + counts[owner]
        counts[owner] += 1
    return slots, block


class TestBeginShardExtendPlan(CustomTestCase):
    def test_plan_with_rotated_prefix(self):
        """7 prefix pages of a base-2 chain + 9 chunk pages (last partial):
        owner-major slots, send rows owner-filtered in the same order and
        padded to the block bound ceil(7/4) = 2 pages."""
        pages = _chain_pages(base=2, n_pages=16)
        prefix_len, seq_len = 7 * PS, 16 * PS - 5
        row = _chain_row(pages, seq_len)
        slots, block = _reference_prefix_slots([pages[:7]])
        for rank in range(N):
            stub = _run_begin(
                _make_pool_stub(_make_spec(), rank), [prefix_len], [seq_len], [row]
            )
            self.assertEqual(stub._block_pages, block)
            self.assertTrue(stub._shard_extend_active)
            self.assertEqual(stub._epoch, 1)
            self.assertEqual(stub.prefetched, [0])  # first layer kicked
            for page, slot in slots.items():
                self.assertEqual(int(stub._page_pos[page]), slot)
            for j, page in enumerate(pages[7:]):
                self.assertEqual(int(stub._page_pos[page]), stub._chunk_base // PS + j)
            own = sorted((p for p in pages[:7] if p % N == rank), key=lambda p: p // N)
            expect = torch.cat(
                [torch.arange((p // N) * PS, (p // N + 1) * PS) for p in own]
            )
            if len(own) < block:  # padded with the trash page (local page 0)
                expect = torch.cat([expect, torch.arange((block - len(own)) * PS)])
            self.assertTrue(torch.equal(stub._send_rows, expect))

    def test_multi_request_plan_shared_prefix_dedup(self):
        """bs > 1: request 0 and request 1 share a 3-page cached prefix
        (request 1 extends it by 2 pages); request 2 is an unrelated base-2
        chain. Shared pages must gather into ONE slot (no duplicate plan
        entries), the block is the per-request ceil sum, and every request's
        locs translate through the same table."""
        chain_a = _chain_pages(base=0, n_pages=5)
        chain_c = _chain_pages(base=2, n_pages=4, local_start=20)
        # rows: request 0 = A[:3] prefix + 1 chunk page; request 1 = A[:5]
        # prefix + 2 chunk pages; request 2 = C[:2] prefix + 2 chunk pages.
        chunk0 = _chain_pages(base=3, n_pages=1, local_start=40)
        chunk1 = _chain_pages(base=1, n_pages=2, local_start=50)
        chunk2 = _chain_pages(base=0, n_pages=2, local_start=60)
        rows = [
            _chain_row(chain_a[:3] + chunk0, 4 * PS),
            _chain_row(chain_a[:5] + chunk1, 7 * PS),
            _chain_row(chain_c[:2] + chunk2, 4 * PS - 3),
        ]
        stub = _run_begin(
            _make_pool_stub(_make_spec()),
            [3 * PS, 5 * PS, 2 * PS],
            [4 * PS, 7 * PS, 4 * PS - 3],
            rows,
        )
        slots, block = _reference_prefix_slots([chain_a[:3], chain_a[:5], chain_c[:2]])
        self.assertEqual(block, 1 + 2 + 1)
        self.assertEqual(stub._block_pages, block)
        for page, slot in slots.items():
            self.assertEqual(int(stub._page_pos[page]), slot)
        # Chunk slots are absolute scratch pages in batch order.
        for j, page in enumerate(chunk0 + chunk1 + chunk2):
            self.assertEqual(int(stub._page_pos[page]), stub._chunk_base // PS + j)
        # Shared pages: both requests' locs hit the SAME scratch rows.
        shared_loc_r0 = rows[0][:PS].long()
        shared_loc_r1 = rows[1][:PS].long()
        t0 = PageInterleaveKVPoolMixin.translate_loc_to_scratch(stub, shared_loc_r0)
        t1 = PageInterleaveKVPoolMixin.translate_loc_to_scratch(stub, shared_loc_r1)
        self.assertTrue(torch.equal(t0, t1))
        # Per-rank send lists fit the block and pad with the trash page.
        all_prefix = sorted(set(chain_a[:5] + chain_c[:2]))
        for rank in range(N):
            stub_r = _run_begin(
                _make_pool_stub(_make_spec(), rank),
                [3 * PS, 5 * PS, 2 * PS],
                [4 * PS, 7 * PS, 4 * PS - 3],
                rows,
            )
            own = sorted((p for p in all_prefix if p % N == rank), key=lambda p: p // N)
            self.assertLessEqual(len(own), block)
            self.assertEqual(stub_r._send_rows.numel(), block * PS)
            expect_head = torch.cat(
                [torch.arange((p // N) * PS, (p // N + 1) * PS) for p in own]
            )
            self.assertTrue(
                torch.equal(stub_r._send_rows[: len(own) * PS], expect_head)
            )

    def test_send_order_follows_local_page_not_position(self):
        """A freed-and-reused page can give a chain a LOWER local page id at
        a later position. Slot assignment and send packing must both order
        by local page id (they only need to agree — a mismatch reads the
        wrong rank rows)."""
        # Owner-0 pages appear at positions 0 and 4 with locals 9 then 3.
        pages = [9 * N + 0, 5 * N + 1, 5 * N + 2, 5 * N + 3, 3 * N + 0]
        row = _chain_row(pages, 5 * PS)
        stub = _run_begin(
            _make_pool_stub(_make_spec(), shard_rank=0),
            [5 * PS],
            [5 * PS + PS],
            [torch.cat([row, _chain_row([7 * N + 1], PS)])],
        )
        slots, block = _reference_prefix_slots([pages])
        self.assertEqual(block, 2)
        # local 3 gets owner-0's first slot although it sits at position 4.
        self.assertEqual(int(stub._page_pos[3 * N + 0]), 0)
        self.assertEqual(int(stub._page_pos[9 * N + 0]), 1)
        expect = torch.cat(
            [torch.arange(3 * PS, 4 * PS), torch.arange(9 * PS, 10 * PS)]
        )
        self.assertTrue(torch.equal(stub._send_rows, expect))

    def test_plan_without_prefix(self):
        pages = _chain_pages(base=0, n_pages=2)
        stub = _run_begin(
            _make_pool_stub(_make_spec()), [0], [PS + 5], [_chain_row(pages, PS + 5)]
        )
        self.assertEqual(stub._block_pages, 0)
        self.assertTrue(stub._shard_extend_active)
        self.assertEqual(stub.prefetched, [])  # nothing to gather
        self.assertIsNone(stub._send_rows)
        self.assertEqual(int(stub._page_pos[pages[0]]), stub._chunk_base // PS)
        self.assertEqual(int(stub._page_pos[pages[1]]), stub._chunk_base // PS + 1)

    def test_unaligned_prefix_rejected(self):
        # The tree quantum is the PHYSICAL page: a prefix that is not a
        # ps-multiple can never come out of match_prefix.
        pages = _chain_pages(base=0, n_pages=4)
        with self.assertRaises(AssertionError):
            _run_begin(
                _make_pool_stub(_make_spec()),
                [PS + 3],
                [4 * PS],
                [_chain_row(pages, 4 * PS)],
            )

    def test_owner_congruence_guard(self):
        """A rotation-base bug that breaks a request's prefix-owner
        cyclicity invalidates the sync-free block bound (a rank can own more
        than ceil(K/N) pages); the debug guard must catch it at plan time."""
        pages = _chain_pages(base=1, n_pages=8)
        pages[2], pages[5] = pages[5], pages[2]  # same multiset, not cyclic
        with self.assertRaises(AssertionError) as ctx:
            _run_begin(
                _make_pool_stub(_make_spec()),
                [6 * PS],
                [8 * PS],
                [_chain_row(pages, 8 * PS)],
            )
        self.assertIn("cyclic", str(ctx.exception))


class TestScratchTranslation(CustomTestCase):
    def _plan(self, base=2, n_prefix=7, n_chunk=9, rank=1):
        pages = _chain_pages(base=base, n_pages=n_prefix + n_chunk)
        seq_len = (n_prefix + n_chunk) * PS
        stub = _run_begin(
            _make_pool_stub(_make_spec(), rank),
            [n_prefix * PS],
            [seq_len],
            [_chain_row(pages, seq_len)],
        )
        return stub, pages[:n_prefix], pages[n_prefix:]

    def _reference_row(self, stub, prefix_pages, chunk_pages, loc):
        """Brute-force reference: owner-major (owner, local-page)-sorted
        prefix slots, sequence-order chunk."""
        spec = stub.shard_spec
        page, off = loc // PS, loc % PS
        if page in prefix_pages:
            slots, _ = _reference_prefix_slots([prefix_pages])
            return slots[page] * PS + off
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

    def test_translation_cache_cleared_with_new_plan(self):
        pages = _chain_pages(base=0, n_pages=2)
        stub = _make_pool_stub(_make_spec())

        # The first batch treats page 0 as part of the current chunk.
        _run_begin(stub, [0], [PS], [_chain_row(pages[:1], PS)])
        loc = _chain_row(pages[:1], PS).long()
        first = PageInterleaveKVPoolMixin._translate_loc_cached(stub, loc)
        again = PageInterleaveKVPoolMixin._translate_loc_cached(stub, loc)
        self.assertIs(again, first)

        # The next batch reuses the same loc tensor after page 0 becomes a
        # cached prefix. Installing the new plan must discard the old mapping.
        _run_begin(stub, [PS], [2 * PS], [_chain_row(pages, 2 * PS)])
        fresh = PageInterleaveKVPoolMixin._translate_loc_cached(stub, loc)
        self.assertIsNot(fresh, first)
        self.assertFalse(torch.equal(fresh, first))


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


# =============================================================================
# Multi-GPU: the real NCCL layer-ahead gather (2 GPUs).
#
# Everything above is pure arithmetic on a CPU stub. This section drives real
# pools over a real process group, which is the only check that the plan the
# stub validates actually addresses the bytes the collective delivers:
#
# 1. MLA pool sharded across the attention-TP group: replicated writes are
#    owner-filtered into disjoint pool stripes; a later batch's chunked-prefix
#    read (get_mla_kv_buffer) assembles the full prefix from all ranks via the
#    layer-ahead NCCL allgather and must return the canonical bytes.
# 2. MHA pool sharded across the attention-CP group: the post-allgather full
#    chunk is staged into the scratch chunk region and owner-persisted; a later
#    batch reads prefix+chunk through the translated page table (the scratch),
#    and the assembled rows must match the canonical bytes.
#
# Skipped unless 2 CUDA devices are visible, so it is inert on the CPU runner
# this file is registered to. Run it explicitly with:
#   CUDA_VISIBLE_DEVICES=0,1 python3 test/registered/unit/mem_cache/\
#       test_page_interleave_shard.py TestPageInterleaveGatherMultiGpu
# =============================================================================

_GATHER_WORLD = 2
_GATHER_LAYER_NUM = 4
_GATHER_PAGE_SIZE = 16
_GATHER_GRANULE = _GATHER_WORLD * _GATHER_PAGE_SIZE
_GATHER_SIZE = _GATHER_PAGE_SIZE * 64  # physical token slots per rank
_GATHER_KV_LORA_RANK = 128
_GATHER_QK_ROPE = 32
_GATHER_HEAD_NUM = 2
_GATHER_HEAD_DIM = 32
_GATHER_DTYPE = torch.bfloat16


def _mla_value(loc, dim):
    """Deterministic canonical latent value for logical slot ``loc``."""
    loc = loc.to(torch.float32)
    return (loc.unsqueeze(-1) + torch.arange(dim, device=loc.device) * 0.001).to(
        _GATHER_DTYPE
    )


def _dist_init(rank, world, port, attn_cp_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    torch.cuda.set_device(rank)

    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    # Publish the widths the groups below are about to be built at. The derived
    # quotients (attn_tp_size, attn_dcp_size, ...) are projected from these
    # leaves at publish; initialize_model_parallel no longer supplies them, and
    # MLATokenToKVPool.set_mla_kv_buffer reads attn_dcp_size on the write path.
    publish(
        ServerArgs(model_path="dummy", tp_size=world, attn_cp_size=attn_cp_size),
        role="scheduler",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world,
        attention_context_model_parallel_size=attn_cp_size,
    )


def _gather_make_spec(shard_rank, max_prefix_groups=16, chunk_groups=4):
    return PageShardSpec(
        shard_rank=shard_rank,
        shard_size=_GATHER_WORLD,
        page_size=_GATHER_PAGE_SIZE,
        max_prefix_tokens=max_prefix_groups * _GATHER_GRANULE,
        chunk_tokens=chunk_groups * _GATHER_GRANULE,
    )


def _fake_req_to_token(groups, seq_len, device):
    """req_to_token row where sequence group j is allocator group groups[j]."""
    row = torch.zeros(
        (1, len(groups) * _GATHER_GRANULE), dtype=torch.int32, device=device
    )
    for j, q in enumerate(groups):
        row[0, j * _GATHER_GRANULE : (j + 1) * _GATHER_GRANULE] = torch.arange(
            q * _GATHER_GRANULE,
            (q + 1) * _GATHER_GRANULE,
            dtype=torch.int32,
            device=device,
        )
    return row[:, :seq_len] if seq_len < row.shape[1] else row


def _check(rank, name, got, expect, atol=0.0):
    ok = torch.allclose(got.float(), expect.float(), atol=atol, rtol=0)
    max_err = (got.float() - expect.float()).abs().max().item()
    print(f"[rank {rank}] {name}: max_err={max_err:.6f} {'OK' if ok else 'FAIL'}")
    assert ok, f"[rank {rank}] {name} mismatch (max_err={max_err})"


def _run_mla(rank, world, port):
    _dist_init(rank, world, port, attn_cp_size=1)

    group = get_parallel().attn_tp_group
    assert group.world_size == world
    # Topology-first shard-group selection: no CP here, so MLA falls back to
    # the attn-TP axis, while GQA has no replicated axis (world_size 1).
    assert get_kv_shard_group(use_mla_backend=True) is group
    assert get_kv_shard_group(use_mla_backend=False).world_size == 1
    spec = _gather_make_spec(shard_rank=group.rank_in_group)

    pool = PageInterleaveMLATokenToKVPool(
        _GATHER_SIZE,
        page_size=_GATHER_PAGE_SIZE,
        dtype=_GATHER_DTYPE,
        kv_lora_rank=_GATHER_KV_LORA_RANK,
        qk_rope_head_dim=_GATHER_QK_ROPE,
        layer_num=_GATHER_LAYER_NUM,
        device=f"cuda:{rank}",
        enable_memory_saver=False,
        start_layer=0,
        end_layer=_GATHER_LAYER_NUM - 1,
        shard_spec=spec,
        shard_group=group,
    )
    device = pool.kv_buffer[0].device

    # ---- chunk 1: replicated write, owner-filtered persist -----------------
    # "Allocator" hands out fragmented groups (identical on every rank).
    chunk1_groups = [5, 2, 9]
    chunk1_locs = _fake_req_to_token(chunk1_groups, 3 * _GATHER_GRANULE, device)[
        0
    ].long()
    for layer_id in range(_GATHER_LAYER_NUM):
        layer = SimpleNamespace(layer_id=layer_id)
        vals = _mla_value(
            chunk1_locs + layer_id * 1000, _GATHER_KV_LORA_RANK + _GATHER_QK_ROPE
        )
        pool.set_mla_kv_buffer(
            layer,
            chunk1_locs,
            vals[:, :_GATHER_KV_LORA_RANK].unsqueeze(1),
            vals[:, _GATHER_KV_LORA_RANK:].unsqueeze(1),
        )
    torch.cuda.synchronize()
    torch.distributed.barrier()

    # Pool holds only the owned stripe: group Q sits at local rows [Q*ps,(Q+1)*ps)
    # on every rank, holding that rank's page of the group.
    for q in chunk1_groups:
        local_rows = torch.arange(
            q * _GATHER_PAGE_SIZE, (q + 1) * _GATHER_PAGE_SIZE, device=device
        )
        owned_locs = (
            q * _GATHER_GRANULE
            + group.rank_in_group * _GATHER_PAGE_SIZE
            + torch.arange(_GATHER_PAGE_SIZE, device=device)
        )
        got = pool.kv_buffer[0][local_rows, 0, :].view(_GATHER_DTYPE)
        expect = _mla_value(owned_locs, _GATHER_KV_LORA_RANK + _GATHER_QK_ROPE)
        _check(rank, f"mla owned stripe g{q}", got, expect)

    # ---- chunk 2: prefix gather + staged chunk, both read styles -----------
    seq_groups = chunk1_groups + [12]  # one new chunk group
    prefix_len = 3 * _GATHER_GRANULE
    seq_len = prefix_len + _GATHER_GRANULE
    req_to_token = _fake_req_to_token(seq_groups, seq_len, device)
    chunk2_locs = req_to_token[0, prefix_len:seq_len].long()
    pool.begin_shard_extend(req_to_token, torch.tensor([0]), [prefix_len], [seq_len])

    for layer_id in range(_GATHER_LAYER_NUM):
        layer = SimpleNamespace(layer_id=layer_id)
        # Write the current chunk (stages it into the slot + persists the
        # owned stripe), like the extend forward does before attention.
        chunk_vals = _mla_value(
            chunk2_locs + layer_id * 1000, _GATHER_KV_LORA_RANK + _GATHER_QK_ROPE
        )
        pool.set_mla_kv_buffer(
            layer,
            chunk2_locs,
            chunk_vals[:, :_GATHER_KV_LORA_RANK].unsqueeze(1),
            chunk_vals[:, _GATHER_KV_LORA_RANK:].unsqueeze(1),
        )
        # Chunked-prefix MHA style: fetch an arbitrary sub-range of the
        # prefix through get_mla_kv_buffer.
        sub = chunk1_locs[_GATHER_PAGE_SIZE // 2 : prefix_len - 3]
        k_nope, k_rope = pool.get_mla_kv_buffer(layer, sub, _GATHER_DTYPE)
        expect = _mla_value(
            sub + layer_id * 1000, _GATHER_KV_LORA_RANK + _GATHER_QK_ROPE
        )
        _check(
            rank,
            f"mla prefix read l{layer_id}",
            k_nope[:, 0, :],
            expect[:, :_GATHER_KV_LORA_RANK],
        )
        _check(
            rank,
            f"mla prefix rope l{layer_id}",
            k_rope[:, 0, :],
            expect[:, _GATHER_KV_LORA_RANK:],
        )
        # Absorbed-MLA style (what MLA-under-CP uses): read [prefix | chunk]
        # from get_key_buffer through the translated page table.
        all_locs = req_to_token[0, :seq_len].long()
        rows = pool.translate_loc_to_scratch(all_locs)
        kv_scratch = pool.get_key_buffer(layer_id)
        _check(
            rank,
            f"mla absorbed read l{layer_id}",
            kv_scratch[rows, 0, :],
            _mla_value(
                all_locs + layer_id * 1000, _GATHER_KV_LORA_RANK + _GATHER_QK_ROPE
            ),
        )

    torch.distributed.barrier()
    if rank == 0:
        print("PASS: MLA page-interleave shard (attn-TP axis)")


def _run_mha(rank, world, port):
    _dist_init(rank, world, port, attn_cp_size=world)

    group = get_parallel().attn_cp_group
    assert group.world_size == world
    # Topology-first shard-group selection: with an active CP group, both
    # GQA and MLA shard across CP (CP replicates KV for every attention
    # type; the TP axis is only the no-CP MLA fallback).
    assert get_kv_shard_group(use_mla_backend=False) is group
    assert get_kv_shard_group(use_mla_backend=True) is group
    spec = _gather_make_spec(shard_rank=group.rank_in_group)

    pool = PageInterleaveMHATokenToKVPool(
        _GATHER_SIZE,
        page_size=_GATHER_PAGE_SIZE,
        dtype=_GATHER_DTYPE,
        head_num=_GATHER_HEAD_NUM,
        head_dim=_GATHER_HEAD_DIM,
        layer_num=_GATHER_LAYER_NUM,
        device=f"cuda:{rank}",
        enable_memory_saver=False,
        start_layer=0,
        end_layer=_GATHER_LAYER_NUM - 1,
        enable_alt_stream=False,
        shard_spec=spec,
        shard_group=group,
    )
    device = pool.k_buffer[0].device

    def kv_value(locs, layer_id, is_v):
        base = locs.to(torch.float32) + layer_id * 1000 + (500000 if is_v else 0)
        return (
            base.view(-1, 1, 1)
            + torch.arange(_GATHER_HEAD_NUM, device=device).view(1, -1, 1) * 0.01
            + torch.arange(_GATHER_HEAD_DIM, device=device).view(1, 1, -1) * 0.0001
        ).to(_GATHER_DTYPE)

    # ---- chunk 1 (prefix-less batch): stage + owner-persist ----------------
    chunk1_groups = [7, 3]
    chunk1_locs = _fake_req_to_token(chunk1_groups, 2 * _GATHER_GRANULE, device)[
        0
    ].long()
    req_to_token = _fake_req_to_token(chunk1_groups, 2 * _GATHER_GRANULE, device)
    pool.begin_shard_extend(req_to_token, torch.tensor([0]), [0], [2 * _GATHER_GRANULE])
    for layer_id in range(_GATHER_LAYER_NUM):
        layer = SimpleNamespace(layer_id=layer_id)
        pool.set_kv_buffer(
            layer,
            chunk1_locs,
            kv_value(chunk1_locs, layer_id, False),
            kv_value(chunk1_locs, layer_id, True),
        )
        # The current chunk must be readable through the scratch right away.
        k_scratch = pool.get_key_buffer(layer_id)
        rows = pool.translate_loc_to_scratch(chunk1_locs)
        _check(
            rank,
            f"mha chunk stage l{layer_id}",
            k_scratch[rows],
            kv_value(chunk1_locs, layer_id, False),
        )
    torch.cuda.synchronize()
    torch.distributed.barrier()

    # ---- chunk 2: prefix gathered from peers via translated page table -----
    seq_groups = chunk1_groups + [11]
    prefix_len = 2 * _GATHER_GRANULE
    seq_len = prefix_len + _GATHER_GRANULE
    req_to_token = _fake_req_to_token(seq_groups, seq_len, device)
    chunk2_locs = req_to_token[0, prefix_len:seq_len].long()
    pool.begin_shard_extend(req_to_token, torch.tensor([0]), [prefix_len], [seq_len])

    for layer_id in range(_GATHER_LAYER_NUM):
        layer = SimpleNamespace(layer_id=layer_id)
        pool.set_kv_buffer(
            layer,
            chunk2_locs,
            kv_value(chunk2_locs, layer_id, False),
            kv_value(chunk2_locs, layer_id, True),
        )
        all_locs = req_to_token[0, :seq_len].long()
        rows = pool.translate_loc_to_scratch(all_locs)
        k_scratch = pool.get_key_buffer(layer_id)
        v_scratch = pool.get_value_buffer(layer_id)
        _check(
            rank,
            f"mha seq read k l{layer_id}",
            k_scratch[rows],
            kv_value(all_locs, layer_id, False),
        )
        _check(
            rank,
            f"mha seq read v l{layer_id}",
            v_scratch[rows],
            kv_value(all_locs, layer_id, True),
        )

    torch.distributed.barrier()
    if rank == 0:
        print("PASS: MHA page-interleave shard (attn-CP axis)")


@unittest.skipIf(
    torch.cuda.device_count() < 2, "page-interleave gather needs 2 CUDA devices"
)
class TestPageInterleaveGatherMultiGpu(CustomTestCase):
    """Real pools, real NCCL, 2 ranks — one mp.spawn per phase.

    Separate spawns (and separate ports) because each phase builds its own
    process group with a different attention-CP width.
    """

    def test_mla_shard_over_attention_tp(self):
        mp.spawn(_run_mla, args=(_GATHER_WORLD, 29811), nprocs=_GATHER_WORLD, join=True)

    def test_mha_shard_over_attention_cp(self):
        mp.spawn(_run_mha, args=(_GATHER_WORLD, 29812), nprocs=_GATHER_WORLD, join=True)


if __name__ == "__main__":
    unittest.main()
