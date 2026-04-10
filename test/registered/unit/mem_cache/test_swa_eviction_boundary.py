"""Unit tests for SWA eviction boundary fixes.

Bug: when page_size > sliding_window_size, _evict_swa could advance the
eviction frontier to exactly page_floor(seq_len), making all tokens being
inserted into the radix tree fully evicted (case 3). _insert_helper had no
handling for this, creating an incorrect non-tombstone node that caused
inflated swa_evictable_size_, negative usage, and potential double-free.

Two-sided fix:
1. _evict_swa subtracts extra page_size (preventive).
2. _insert_helper early-returns on case 3 (defensive).

Tests use real tree/allocator/pool with mock Req/ScheduleBatch wrappers.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _swa_alloc(allocator, need_size):
    """Allocate from SWA allocator for any page_size.

    SWATokenToKVPoolAllocator.alloc() asserts page_size == 1. For page_size > 1,
    allocate from the underlying paged allocators directly and set up the mapping.
    """
    if allocator.page_size == 1:
        return allocator.alloc(need_size)

    if need_size > allocator.full_attn_allocator.available_size():
        return None
    if need_size > allocator.swa_attn_allocator.available_size():
        return None

    full_indices = allocator.full_attn_allocator.alloc(need_size)
    swa_indices = allocator.swa_attn_allocator.alloc(need_size)
    assert full_indices is not None and swa_indices is not None
    allocator.full_to_swa_index_mapping[full_indices] = swa_indices
    return full_indices


def _build_swa_tree(page_size, sliding_window_size, kv_size=1024, kv_size_swa=512):
    head_num, head_dim, num_layers, global_interval = 8, 128, 24, 4
    dtype = torch.bfloat16
    device = get_device()
    full_ids = list(range(0, num_layers, global_interval))
    swa_ids = [i for i in range(num_layers) if i not in set(full_ids)]

    pool = ReqToTokenPool(
        size=8, max_context_len=2048, device=device, enable_memory_saver=False
    )
    kv_pool = SWAKVPool(
        size=kv_size,
        size_swa=kv_size_swa,
        page_size=page_size,
        dtype=dtype,
        head_num=head_num,
        head_dim=head_dim,
        swa_attention_layer_ids=swa_ids,
        full_attention_layer_ids=full_ids,
        enable_kvcache_transpose=False,
        device=device,
    )
    allocator = SWATokenToKVPoolAllocator(
        size=kv_size,
        size_swa=kv_size_swa,
        page_size=page_size,
        dtype=dtype,
        device=device,
        kvcache=kv_pool,
        need_sort=False,
    )
    tree = SWARadixCache(
        params=CacheInitParams(
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            disable=False,
            is_eagle=False,
            sliding_window_size=sliding_window_size,
        ),
    )
    return tree, allocator, pool


def _make_req(req_pool_idx, token_ids, cache_protected_len, tree):
    """Mock Req with fields needed by _evict_swa and cache_finished_req."""
    req = SimpleNamespace(
        req_pool_idx=req_pool_idx,
        origin_input_ids=token_ids,
        output_ids=[],
        cache_protected_len=cache_protected_len,
        swa_evicted_seqlen=0,
        extra_key=None,
        last_node=tree.root_node,
        swa_uuid_for_lock=None,
        prefix_indices=torch.tensor([], dtype=torch.int64, device=tree.device),
        _kv_committed_len=len(token_ids),
    )
    req.pop_committed_kv_cache = lambda: req._kv_committed_len
    return req


def _make_batch(tree, allocator, pool):
    """Mock ScheduleBatch with fields needed by _evict_swa."""
    return SimpleNamespace(
        tree_cache=tree,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
    )


def _page_ceil(n, page_size):
    return (n + page_size - 1) // page_size * page_size


def _page_floor(n, page_size):
    return n // page_size * page_size


def _evict_and_insert(tree, allocator, pool, seq_len, page_size, req_pool_idx=0):
    """Full flow: alloc -> _evict_swa -> cache_finished_req. Returns req."""
    alloc_size = _page_ceil(seq_len, page_size)
    kv_indices = _swa_alloc(allocator, alloc_size)
    assert kv_indices is not None, f"Alloc failed: need {alloc_size}"
    pool.write((req_pool_idx, slice(0, alloc_size)), kv_indices)

    req = _make_req(req_pool_idx, list(range(seq_len)), 0, tree)
    batch = _make_batch(tree, allocator, pool)

    ScheduleBatch._evict_swa(batch, req, seq_len - 1)
    tree.cache_finished_req(req, is_insert=True)
    return req


# ---------------------------------------------------------------------------
# Group 1: Eviction formula verification
# ---------------------------------------------------------------------------


class TestEvictionFormula(unittest.TestCase):
    """Verify _evict_swa never produces swa_evicted_seqlen >= insert_length."""

    def test_page_gt_window_sweep(self):
        """Sweep page_size > window combinations. The -page_size fix must prevent
        the eviction frontier from reaching page_floor(seq_len)."""
        for page_size in [4, 8, 16, 32, 64, 128, 256]:
            for window in [1, 2, 4, 8]:
                if page_size <= window:
                    continue
                tree, allocator, pool = _build_swa_tree(
                    page_size=page_size,
                    sliding_window_size=window,
                    kv_size=max(4096, page_size * 20),
                    kv_size_swa=max(2048, page_size * 10),
                )
                for seq_len in range(page_size + 1, page_size * 5):
                    alloc_size = _page_ceil(seq_len, page_size)
                    kv = _swa_alloc(allocator, alloc_size)
                    if kv is None:
                        break
                    pool.write((0, slice(0, alloc_size)), kv)

                    req = _make_req(0, list(range(seq_len)), 0, tree)
                    batch = _make_batch(tree, allocator, pool)
                    ScheduleBatch._evict_swa(batch, req, seq_len - 1)

                    insert_len = _page_floor(seq_len, page_size)
                    self.assertLess(
                        req.swa_evicted_seqlen,
                        insert_len,
                        f"Over-eviction: page={page_size}, win={window}, "
                        f"seq={seq_len}, evicted={req.swa_evicted_seqlen}",
                    )

                    # Clean up for next iteration
                    allocator.free(kv)

    def test_page_leq_window(self):
        """page_size <= window: eviction always stays well below insert boundary.
        The -page_size fix should cause no regression."""
        page_size = 4
        window = 8
        tree, allocator, pool = _build_swa_tree(
            page_size=page_size, sliding_window_size=window
        )

        for seq_len in [13, 17, 25, 33]:
            req = _evict_and_insert(tree, allocator, pool, seq_len, page_size)
            insert_len = _page_floor(seq_len, page_size)
            self.assertLess(req.swa_evicted_seqlen, insert_len)
            tree.sanity_check()

    def test_page_size_1(self):
        """page_size=1: no floor alignment. The -1 from page_size just means
        one less token evicted, equivalent to old decode convention."""
        page_size = 1
        window = 4
        tree, allocator, pool = _build_swa_tree(
            page_size=page_size, sliding_window_size=window
        )

        for seq_len in range(window + 2, 30):
            req = _evict_and_insert(tree, allocator, pool, seq_len, page_size)
            # With page_size=1, insert_length == seq_len, so evicted < seq_len always
            self.assertLess(req.swa_evicted_seqlen, seq_len)
            tree.sanity_check()

    def test_eviction_noop_short_sequence(self):
        """When seq_len is small, pre_len - window - page_size < 0 and
        eviction should not advance (no-op)."""
        page_size = 8
        window = 4
        tree, allocator, pool = _build_swa_tree(
            page_size=page_size, sliding_window_size=window
        )

        # seq_len = 10: pre_len=9, 9-4-8 = -3 -> evicted stays at 0
        seq_len = page_size + window - 2
        alloc_size = _page_ceil(seq_len, page_size)
        kv = _swa_alloc(allocator, alloc_size)
        pool.write((0, slice(0, alloc_size)), kv)

        req = _make_req(0, list(range(seq_len)), 0, tree)
        batch = _make_batch(tree, allocator, pool)
        ScheduleBatch._evict_swa(batch, req, seq_len - 1)

        self.assertEqual(req.swa_evicted_seqlen, 0)


# ---------------------------------------------------------------------------
# Group 2: _insert_helper case 1 / 2 / 3
# ---------------------------------------------------------------------------


class TestInsertCases(unittest.TestCase):
    """Verify _insert_helper handles all three swa_evicted_seqlen positions."""

    def test_case2_partial_tombstone(self):
        """Case 2: total_prefix_length < swa_evicted < total_length.

        Normal eviction with page_size > window: some new tokens are tombstone,
        the rest are non-tombstone. Verify swa_evictable accounting.
        """
        page_size = 8
        window = 2
        tree, allocator, pool = _build_swa_tree(
            page_size=page_size, sliding_window_size=window
        )

        # seq_len=25: insert_length=24, new_evicted should be 8 (1 page)
        seq_len = page_size * 3 + 1
        swa_evictable_before = tree.swa_evictable_size_

        req = _evict_and_insert(tree, allocator, pool, seq_len, page_size)

        insert_len = _page_floor(seq_len, page_size)
        self.assertGreater(req.swa_evicted_seqlen, 0, "Should have some eviction")
        self.assertLess(req.swa_evicted_seqlen, insert_len, "Should be partial")

        # Non-tombstone portion should be added to swa_evictable
        non_tombstone = insert_len - req.swa_evicted_seqlen
        self.assertEqual(tree.swa_evictable_size_, swa_evictable_before + non_tombstone)
        # Tombstone portion should be added to full_evictable but not swa_evictable
        self.assertGreater(tree.full_evictable_size_, 0)
        tree.sanity_check()

    def test_case1_evicted_within_matched(self):
        """Case 1: swa_evicted_seqlen <= total_prefix_length.

        Insert first request to populate tree, then a second request with
        overlapping prefix. Use _evict_swa with a small pre_len so eviction
        stays within the matched region. New tokens should all be non-tombstone.
        """
        page_size = 8
        window = 2
        tree, allocator, pool = _build_swa_tree(
            page_size=page_size, sliding_window_size=window
        )

        # First request: insert 16 tokens (2 pages) into tree
        first_len = page_size * 2
        _evict_and_insert(tree, allocator, pool, first_len, page_size, req_pool_idx=0)
        tree.sanity_check()

        # Second request: 24 tokens, first 16 overlap with tree
        second_len = page_size * 3
        alloc_size = _page_ceil(second_len, page_size)
        kv2 = _swa_alloc(allocator, alloc_size)
        pool.write((1, slice(0, alloc_size)), kv2)

        req2 = _make_req(1, list(range(second_len)), 0, tree)
        batch = _make_batch(tree, allocator, pool)

        # Evict with pre_len that keeps eviction within the first 16 (matched) tokens
        # pre_len=15: 15 - 2 - 8 = 5, floor to 8 -> 0. Eviction stays at 0.
        ScheduleBatch._evict_swa(batch, req2, first_len - 1)
        self.assertLessEqual(
            req2.swa_evicted_seqlen,
            first_len,
            "Precondition: eviction within matched region",
        )

        swa_evictable_before = tree.swa_evictable_size_
        tree.cache_finished_req(req2, is_insert=True)

        # New tokens [16, 24) should all be non-tombstone
        new_tokens = _page_floor(second_len, page_size) - first_len
        self.assertEqual(tree.swa_evictable_size_, swa_evictable_before + new_tokens)
        tree.sanity_check()

    def test_case3_defensive_early_return(self):
        """Case 3: swa_evicted_seqlen == total_length (defensive guard).

        Simulate the OLD formula (without -page_size) to produce a boundary
        value, then feed it through real cache_finished_req. Verify no
        non-tombstone node is created.
        """
        page_size = 8
        window = 2
        tree, allocator, pool = _build_swa_tree(
            page_size=page_size, sliding_window_size=window
        )

        # seq_len=11: old formula -> evicted=8 == insert_length=8
        seq_len = page_size + window + 1
        alloc_size = _page_ceil(seq_len, page_size)
        kv = _swa_alloc(allocator, alloc_size)
        pool.write((0, slice(0, alloc_size)), kv)

        # Compute using OLD formula (without -page_size)
        pre_len = seq_len - 1
        old_evicted = max(0, _page_floor(pre_len - window, page_size))
        insert_len = _page_floor(seq_len, page_size)
        self.assertEqual(
            old_evicted, insert_len, "Precondition: old formula hits boundary"
        )

        # Manually free SWA as _evict_swa would
        free_slots = pool.req_to_token[0, :old_evicted]
        allocator.free_swa(free_slots)

        req = _make_req(0, list(range(seq_len)), 0, tree)
        req.swa_evicted_seqlen = old_evicted

        swa_evictable_before = tree.swa_evictable_size_
        tree.cache_finished_req(req, is_insert=True)

        # Core assertion: no non-tombstone node created
        self.assertEqual(tree.swa_evictable_size_, swa_evictable_before)


# ---------------------------------------------------------------------------
# Group 3: Integration / multi-step
# ---------------------------------------------------------------------------


class TestIntegration(unittest.TestCase):
    """Multi-step and cross-scenario integration tests."""

    def test_multiple_decodes(self):
        """Simulate multiple decode turns with page_size > window.
        Each turn: alloc -> evict -> insert. Verify no over-eviction and
        tree stays consistent throughout."""
        page_size = 8
        window = 2
        tree, allocator, pool = _build_swa_tree(
            page_size=page_size, sliding_window_size=window
        )

        for turn in range(4):
            seq_len = page_size * (turn + 2) + 1
            req_pool_idx = turn % pool.size
            req = _evict_and_insert(
                tree, allocator, pool, seq_len, page_size, req_pool_idx
            )

            insert_len = _page_floor(seq_len, page_size)
            self.assertLess(
                req.swa_evicted_seqlen,
                insert_len,
                f"Over-eviction at turn {turn}",
            )
            tree.sanity_check()

    def test_page_size_1_full_flow(self):
        """End-to-end with page_size=1. Verify the fix is a no-op in practice
        (eviction changes by at most 1 token)."""
        page_size = 1
        window = 4
        tree, allocator, pool = _build_swa_tree(
            page_size=page_size, sliding_window_size=window
        )

        for seq_len in [10, 20, 30]:
            req = _evict_and_insert(tree, allocator, pool, seq_len, page_size)
            # With page_size=1, all tokens are inserted (no page truncation)
            expected_evicted = max(0, seq_len - 1 - window - page_size)
            self.assertEqual(req.swa_evicted_seqlen, expected_evicted)
            tree.sanity_check()


if __name__ == "__main__":
    unittest.main()
