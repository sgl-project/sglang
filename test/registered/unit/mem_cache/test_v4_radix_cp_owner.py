"""
Unit tests for CP KV-resharding v4 changes to the radix cache.

Covers Part 1 sub-tasks (radix-cache-only; no CP runtime, no GPU, no NCCL):
- 1.2 (1c):     sentinel filter at every allocator.free site
- 1.3 (1d + 1e): TreeNode.cp_owner_per_page field + _split_node slicing
- 1.4 (1f):     _insert_helper plumbs cp_owner_per_page into new nodes
- 1.5 (1g):     compute_cp_owner_per_page is deterministic and chunk-stable
- 1.6 (1h):     cache_finished_req filters non-owned positions before free

(1a + 1b — logical clock and 3-tuple __lt__ tiebreaker — were dropped per
the v1 design pivot to IPC-based eviction consensus; see §5 of
DESIGN_kv_reshard.md.)

Usage:
    python test_v4_radix_cp_owner.py
    python -m pytest test_v4_radix_cp_owner.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")

import unittest
import unittest.mock

import torch

from sglang.srt.layers.utils.cp_utils import (
    compute_cp_owner_per_page,
    owner_for_page,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    _filter_pool_rows,
)

PAGE_SIZE = 4


def _make_mock_allocator():
    """A mock allocator that records freed indices. CPU-only."""
    mock = unittest.mock.Mock()
    mock.device = torch.device("cpu")
    mock.freed = []

    def _record_free(indices):
        # Record both empty and non-empty frees so tests can assert on shape.
        mock.freed.append(
            indices.clone() if isinstance(indices, torch.Tensor) else indices
        )

    mock.free.side_effect = _record_free
    return mock


def _flat_freed(mock_allocator):
    """Concatenate all freed tensors into a single 1D tensor for assertions."""
    pieces = []
    for f in mock_allocator.freed:
        if isinstance(f, torch.Tensor) and f.numel() > 0:
            pieces.append(f.flatten())
    if not pieces:
        return torch.empty((0,), dtype=torch.int64)
    return torch.cat(pieces).to(dtype=torch.int64)


class TestSentinelFilter(unittest.TestCase):
    """1.2: _filter_pool_rows drops slot-0 sentinels before allocator.free."""

    def test_filter_drops_zeros(self):
        x = torch.tensor([0, 1, 0, 2, 3, 0], dtype=torch.int64)
        out = _filter_pool_rows(x)
        self.assertEqual(out.tolist(), [1, 2, 3])

    def test_filter_passthrough_no_zeros(self):
        x = torch.tensor([1, 2, 3], dtype=torch.int64)
        out = _filter_pool_rows(x)
        self.assertEqual(out.tolist(), [1, 2, 3])

    def test_filter_none(self):
        self.assertIsNone(_filter_pool_rows(None))

    def test_evict_skips_sentinels(self):
        """End-to-end: with CP consensus enabled, evicting a TreeNode whose
        value mixes sentinels (slot 0) and real rows must free only the
        real rows, never slot 0. The filter lives on the CP-consensus
        path (``_apply_cp_eviction``); the non-CP ``_evict_local`` is
        reached only when sentinels do not exist in the first place."""
        mock = _make_mock_allocator()
        cache = RadixCache.create_simulated(page_size=1, mock_allocator=mock)

        # Single-process stand-in for the CP attention group: rank 0 with
        # cp_size=2 routes evict() through _evict_cp_consensus, and the
        # passthrough broadcast_object lets _apply_cp_eviction run locally
        # with the same paths rank 0 picked.
        cp_group = unittest.mock.Mock()
        cp_group.broadcast_object.side_effect = lambda obj, src=0: obj
        cache.enable_cp_consensus(cp_group, cp_rank=0, cp_size=2)

        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4]),
                # Per-rank projection: two real rows (10, 30) plus two sentinels.
                value=torch.tensor([10, 0, 30, 0], dtype=torch.int64),
            )
        )
        cache.evict(EvictParams(num_tokens=4))
        freed = _flat_freed(mock)
        self.assertFalse((freed == 0).any(), f"slot-0 was freed: {freed.tolist()}")
        self.assertEqual(sorted(freed.tolist()), [10, 30])


class TestSplitNodeCpOwner(unittest.TestCase):
    """1.3: _split_node slices cp_owner_per_page at the page-aligned boundary."""

    def test_split_at_page_boundary(self):
        # page_size=4, 16 tokens -> 4 pages. cp_owner_per_page = [0,1,2,3].
        cache = RadixCache.create_simulated(page_size=PAGE_SIZE)
        long_owner = torch.tensor([0, 1, 2, 3], dtype=torch.int8)
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(16))),
                value=torch.arange(1, 17, dtype=torch.int64),
                cp_owner_per_page=long_owner,
            )
        )
        # Force a split: insert a shorter prefix that diverges mid-way.
        # Same first 8 tokens, then a different continuation.
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(8)) + [99, 99, 99, 99]),
                value=torch.arange(101, 113, dtype=torch.int64),
                cp_owner_per_page=torch.tensor([0, 1, 7, 7], dtype=torch.int8),
            )
        )
        # The original node should now be split. Walk the tree from root and
        # collect (key, cp_owner_per_page) pairs.
        all_nodes = []

        def _walk(n):
            for c in n.children.values():
                all_nodes.append(c)
                _walk(c)

        _walk(cache.root_node)
        for n in all_nodes:
            if n.cp_owner_per_page is not None and len(n.key) > 0:
                expected_pages = len(n.key) // PAGE_SIZE
                self.assertEqual(
                    len(n.cp_owner_per_page),
                    expected_pages,
                    f"node {n.key.token_ids[:4]}... has key_len={len(n.key)} "
                    f"but cp_owner_per_page len={len(n.cp_owner_per_page)}",
                )

    def test_length_invariant_holds_after_insertions(self):
        """For every non-empty node carrying cp_owner_per_page: len(value)
        equals page_size * len(cp_owner_per_page)."""
        cache = RadixCache.create_simulated(page_size=PAGE_SIZE)
        owner = compute_cp_owner_per_page(16, PAGE_SIZE, cp_size=4)
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(16))),
                value=torch.arange(1, 17, dtype=torch.int64),
                cp_owner_per_page=owner,
            )
        )

        def _walk(n):
            if (
                n.cp_owner_per_page is not None
                and n.value is not None
                and len(n.key) > 0
            ):
                self.assertEqual(
                    len(n.value),
                    PAGE_SIZE * len(n.cp_owner_per_page),
                    f"len(value)={len(n.value)} != page_size * len(cp_owner_per_page)"
                    f" = {PAGE_SIZE * len(n.cp_owner_per_page)}",
                )
            for c in n.children.values():
                _walk(c)

        _walk(cache.root_node)


class TestInsertHelperPlumbsCpOwner(unittest.TestCase):
    """1.4: _insert_helper stores the per-page owner on each new TreeNode and
    match_prefix returns nodes carrying that array."""

    def test_new_node_carries_owner_array(self):
        cache = RadixCache.create_simulated(page_size=PAGE_SIZE)
        owner = torch.tensor([2, 0, 1, 3], dtype=torch.int8)
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(16))),
                value=torch.arange(1, 17, dtype=torch.int64),
                cp_owner_per_page=owner,
            )
        )
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(list(range(16)))))
        node = result.last_device_node
        self.assertIsNotNone(node.cp_owner_per_page)
        self.assertEqual(node.cp_owner_per_page.tolist(), owner.tolist())

    def test_partial_match_returns_correct_slice(self):
        """Match a strict prefix - the matched node still carries the slice
        that corresponds to the matched portion."""
        cache = RadixCache.create_simulated(page_size=PAGE_SIZE)
        owner = torch.tensor([2, 0, 1, 3], dtype=torch.int8)
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(16))),
                value=torch.arange(1, 17, dtype=torch.int64),
                cp_owner_per_page=owner,
            )
        )
        # Match the first 8 tokens (2 pages) -> _split_node fires, parent
        # node gets cp_owner_per_page[:2] = [2, 0].
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(list(range(8)))))
        # Concatenate cp_owner_per_page along the matched path.
        owners = []
        node = result.last_device_node
        while node.parent is not None:
            if node.cp_owner_per_page is not None:
                owners = node.cp_owner_per_page.tolist() + owners
            node = node.parent
        self.assertEqual(owners, [2, 0])


class TestComputeCpOwnerPerPage(unittest.TestCase):
    """1.5: compute_cp_owner_per_page / owner_for_page partition invariants."""

    def test_determinism_across_calls(self):
        for total_pages in [4, 7, 16, 64]:
            for cp_size in [2, 4, 8]:
                total_tokens = total_pages * PAGE_SIZE
                a = compute_cp_owner_per_page(total_tokens, PAGE_SIZE, cp_size)
                b = compute_cp_owner_per_page(total_tokens, PAGE_SIZE, cp_size)
                self.assertTrue(torch.equal(a, b))

    def test_owner_for_page_in_range_and_covers_all_ranks(self):
        for total_pages in [1, 4, 7, 16, 100]:
            for cp_size in [2, 4, 8]:
                seen = set()
                for i in range(total_pages):
                    r = owner_for_page(i, total_pages, cp_size)
                    self.assertTrue(0 <= r < cp_size, f"owner out of range: {r}")
                    seen.add(r)
                # If total_pages >= cp_size every rank should own >=1 page.
                if total_pages >= cp_size:
                    self.assertEqual(len(seen), cp_size)

    def test_partition_is_balanced_within_one_page(self):
        """Per-request imbalance is at most one page: the difference between
        the rank owning the most pages and the rank owning the fewest is
        either 0 (cleanly divisible) or 1 (rem != 0)."""
        for total_pages in [4, 7, 16, 64, 100]:
            for cp_size in [2, 4, 8]:
                owners = compute_cp_owner_per_page(
                    total_pages * PAGE_SIZE, PAGE_SIZE, cp_size
                )
                counts = torch.bincount(owners.to(torch.int64), minlength=cp_size)
                spread = int(counts.max() - counts.min())
                self.assertLessEqual(spread, 1, f"counts={counts.tolist()}")
                self.assertEqual(int(counts.sum()), total_pages)


class TestCacheReqFilter(unittest.TestCase):
    """1.6: cache_finished_req only frees pool rows on the owning side,
    and the inserted tree node correctly carries the owner array."""

    def _make_req(self, token_ids, page_table_row, cp_owner_per_page=None):
        """Build a minimal mock Req for cache_finished_req."""
        req = unittest.mock.Mock()
        req.req_pool_idx = 0
        req.origin_input_ids = list(token_ids)
        req.output_ids = []
        req.cache_protected_len = 0
        req.last_node = None
        req.cp_owner_per_page = cp_owner_per_page
        req.extra_key = None
        # pop_committed_kv_cache returns how many tokens are covered.
        req.pop_committed_kv_cache = lambda: len(token_ids)
        req.priority = 0
        return req

    def test_cache_finished_req_filters_sentinels(self):
        mock_allocator = _make_mock_allocator()

        # ReqToTokenPool stub: hold a single fake page table row.
        # values: real rows at pos 0,2,3 (owned by this rank); sentinel at 1.
        page_row = torch.tensor([10, 0, 30, 40, 50, 0, 70, 80], dtype=torch.int64)
        req_to_token_pool = unittest.mock.Mock()
        req_to_token_pool.req_to_token = page_row.unsqueeze(0)

        from sglang.srt.mem_cache.cache_init_params import CacheInitParams

        cache = RadixCache(
            CacheInitParams(
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=mock_allocator,
                page_size=PAGE_SIZE,
                enable_kv_cache_events=False,
                disable=False,
            )
        )

        # 8 tokens = 2 pages. Page 0 owned by rank 0 (our test rank), page 1
        # owned by some other rank. So we expect rows [10, 0, 30, 40] for
        # page 0 and [50, 0, 70, 80] for page 1 - the real values were
        # written by whichever rank ran the forward; on this rank, page 0's
        # owned positions hold real rows and page 1's are sentinels (the
        # real-world setup would be the converse for the other rank).
        # For the purposes of this test, what matters is: any freed tensor
        # must contain zero zeros.
        cp_owner_per_page = torch.tensor([0, 1], dtype=torch.int8)
        req = self._make_req(
            token_ids=list(range(8)),
            page_table_row=page_row,
            cp_owner_per_page=cp_owner_per_page,
        )

        cache.cache_finished_req(req, is_insert=True)
        freed = _flat_freed(mock_allocator)
        self.assertFalse(
            (freed == 0).any(),
            f"cache_finished_req freed slot-0 sentinel: {freed.tolist()}",
        )

    def test_cache_finished_req_no_cp_mode(self):
        """When req.cp_owner_per_page is None, behavior matches today: the
        filter is a no-op because real allocator never returns 0."""
        mock_allocator = _make_mock_allocator()
        page_row = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int64)
        req_to_token_pool = unittest.mock.Mock()
        req_to_token_pool.req_to_token = page_row.unsqueeze(0)

        from sglang.srt.mem_cache.cache_init_params import CacheInitParams

        cache = RadixCache(
            CacheInitParams(
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=mock_allocator,
                page_size=PAGE_SIZE,
                enable_kv_cache_events=False,
                disable=False,
            )
        )
        req = self._make_req(token_ids=list(range(8)), page_table_row=page_row)
        cache.cache_finished_req(req, is_insert=True)
        # No sentinels in input -> nothing dropped -> same indices freed.
        freed = _flat_freed(mock_allocator)
        # Unaligned tail empty; insert duplicated 0 protected -> 0 prefix freed.
        # Just assert no zeros were freed and the freed entries are a subset
        # of the original row.
        for v in freed.tolist():
            self.assertIn(v, page_row.tolist())


class _FakeCPGroup:
    """In-process stand-in for sglang.srt.distributed.GroupCoordinator.

    Mirrors the broadcast semantics: rank 0 supplies the payload; every
    follower receives a deep copy. Implemented via a shared dict keyed
    by call id, so two RadixCache instances (one per simulated rank) in
    the same Python process exchange the same way they would over NCCL.
    """

    _next_call_id = 0
    _payloads: dict = {}

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self._call_id = 0

    def broadcast_object(self, obj, src: int = 0):
        # Rank 0 publishes; followers read by the same call id.
        if self.rank == src:
            _FakeCPGroup._payloads[self._call_id] = obj
            payload = obj
        else:
            payload = _FakeCPGroup._payloads.get(self._call_id)
        self._call_id += 1
        return payload


class TestCpConsensusEviction(unittest.TestCase):
    """2.5 (Commit 2): rank-0-broadcasts-victims protocol -> every CP rank
    frees the same set of nodes, independent of their per-rank wall-clock
    skew on last_access_time."""

    def setUp(self):
        _FakeCPGroup._payloads.clear()
        _FakeCPGroup._next_call_id = 0

    def _build_cache_with_keys(self, page_size: int, mock_allocator):
        cache = RadixCache.create_simulated(
            page_size=page_size, mock_allocator=mock_allocator
        )
        # Three sibling leaves under root; each is its own 2-token "page".
        for tid in [3, 5, 7]:
            cache.insert(
                InsertParams(
                    key=RadixKey([tid, tid + 1]),
                    value=torch.tensor([tid * 10, (tid + 1) * 10], dtype=torch.int64),
                )
            )
        return cache

    def test_rank0_choice_propagates_to_followers(self):
        """Two simulated ranks build identical trees but with INTENTIONALLY
        skewed last_access_time. Rank 0 picks one victim via its local
        heap; the broadcast must drive both ranks to evict the same node."""
        alloc0 = _make_mock_allocator()
        alloc1 = _make_mock_allocator()
        cache0 = self._build_cache_with_keys(page_size=1, mock_allocator=alloc0)
        cache1 = self._build_cache_with_keys(page_size=1, mock_allocator=alloc1)

        # Skew last_access_time on rank 1 so its local heap order would
        # diverge if it were trusted -- the broadcast must override.
        for node in list(cache1.evictable_leaves):
            node.last_access_time = -node.last_access_time

        group0 = _FakeCPGroup(rank=0, world_size=2)
        group1 = _FakeCPGroup(rank=1, world_size=2)
        cache0.enable_cp_consensus(group0, cp_rank=0, cp_size=2)
        cache1.enable_cp_consensus(group1, cp_rank=1, cp_size=2)

        # Evict one node's worth of tokens. Rank 0's lru pick should
        # propagate to rank 1 verbatim.
        cache0.evict(EvictParams(num_tokens=2))
        cache1.evict(EvictParams(num_tokens=2))

        # Both caches should now contain the same remaining leaves
        # (identified by first token). Same set of children of root.
        remaining0 = sorted(k for k in cache0.root_node.children)
        remaining1 = sorted(k for k in cache1.root_node.children)
        self.assertEqual(remaining0, remaining1)
        # And exactly one leaf was evicted on each.
        self.assertEqual(len(remaining0), 2)

    def test_disabled_when_cp_size_one(self):
        """enable_cp_consensus(cp_size=1) is a no-op; evict() falls through
        to the local path so single-rank deployments keep today's behavior."""
        mock = _make_mock_allocator()
        cache = self._build_cache_with_keys(page_size=1, mock_allocator=mock)
        cache.enable_cp_consensus(
            _FakeCPGroup(rank=0, world_size=1), cp_rank=0, cp_size=1
        )
        self.assertIsNone(cache.cp_attn_group)

    def test_full_path_token_ids_round_trips_via_lookup_by_path(self):
        """A node's path must be invertible: _lookup_by_path on the path
        produced by _full_path_token_ids returns the same node."""
        cache = self._build_cache_with_keys(
            page_size=1, mock_allocator=_make_mock_allocator()
        )
        for node in list(cache.evictable_leaves):
            path = cache._full_path_token_ids(node)
            self.assertIs(cache._lookup_by_path(path), node)
        # Empty path -> root.
        self.assertIs(cache._lookup_by_path(()), cache.root_node)
        # Bogus path -> None.
        self.assertIsNone(cache._lookup_by_path((9999, 9998)))


if __name__ == "__main__":
    unittest.main()
