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

register_cuda_ci(est_time=12, stage="stage-b", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

# ---------------------------------------------------------------------------
# Infrastructure helpers (shared setup, not logic)
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
        swa_prefix_lock_released=False,
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSWAEvictionBoundary(unittest.TestCase):

    # -- Eviction formula: page_size > window --

    def test_formula_page_gt_window_sweep(self):
        """Sweep page_size > window combinations. The -page_size fix must
        prevent eviction from reaching page_floor(seq_len)."""
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
                    alloc_size = (seq_len + page_size - 1) // page_size * page_size
                    kv = _swa_alloc(allocator, alloc_size)
                    if kv is None:
                        break
                    pool.write((0, slice(0, alloc_size)), kv)

                    req = _make_req(0, list(range(seq_len)), 0, tree)
                    batch = _make_batch(tree, allocator, pool)
                    ScheduleBatch._evict_swa(batch, req, seq_len - 1)

                    insert_len = seq_len // page_size * page_size
                    self.assertLess(
                        req.swa_evicted_seqlen,
                        insert_len,
                        f"page={page_size}, win={window}, seq={seq_len}",
                    )
                    allocator.free(kv)

    # -- Eviction formula: page_size <= window --

    def test_formula_page_leq_window(self):
        """page_size <= window: -page_size fix causes no regression."""
        page_size, window = 4, 8
        tree, allocator, pool = _build_swa_tree(
            page_size=page_size, sliding_window_size=window
        )

        for seq_len in [13, 17, 25, 33]:
            alloc_size = (seq_len + page_size - 1) // page_size * page_size
            kv = _swa_alloc(allocator, alloc_size)
            pool.write((0, slice(0, alloc_size)), kv)

            req = _make_req(0, list(range(seq_len)), 0, tree)
            batch = _make_batch(tree, allocator, pool)
            ScheduleBatch._evict_swa(batch, req, seq_len - 1)

            insert_len = seq_len // page_size * page_size
            self.assertLess(req.swa_evicted_seqlen, insert_len)

            tree.cache_finished_req(req, is_insert=True)
            tree.sanity_check()

    # -- Eviction formula: page_size == 1 --

    def test_formula_page_size_1(self):
        """page_size=1: no floor alignment, -1 just means one less token evicted."""
        page_size, window = 1, 4
        tree, allocator, pool = _build_swa_tree(
            page_size=page_size, sliding_window_size=window
        )

        for seq_len in range(window + 2, 30):
            kv = _swa_alloc(allocator, seq_len)
            pool.write((0, slice(0, seq_len)), kv)

            req = _make_req(0, list(range(seq_len)), 0, tree)
            batch = _make_batch(tree, allocator, pool)
            ScheduleBatch._evict_swa(batch, req, seq_len - 1)

            self.assertLess(req.swa_evicted_seqlen, seq_len)
            self.assertEqual(req.swa_evicted_seqlen, max(0, seq_len - 1 - window - 1))

            tree.cache_finished_req(req, is_insert=True)
            tree.sanity_check()

    # -- Eviction formula: no-op when seq too short --

    def test_formula_noop_short_sequence(self):
        """pre_len - window - page_size < 0: eviction stays at 0."""
        page_size, window = 8, 4
        tree, allocator, pool = _build_swa_tree(
            page_size=page_size, sliding_window_size=window
        )

        seq_len = page_size + window - 2  # = 10, formula gives 10-1-4-8 = -3
        alloc_size = (seq_len + page_size - 1) // page_size * page_size
        kv = _swa_alloc(allocator, alloc_size)
        pool.write((0, slice(0, alloc_size)), kv)

        req = _make_req(0, list(range(seq_len)), 0, tree)
        batch = _make_batch(tree, allocator, pool)
        ScheduleBatch._evict_swa(batch, req, seq_len - 1)

        self.assertEqual(req.swa_evicted_seqlen, 0)

    # -- Insert case 1: swa_evicted <= total_prefix_length --

    def test_insert_case1_evicted_within_matched(self):
        """Eviction within matched region. New tokens all non-tombstone."""
        page_size, window = 8, 2
        tree, allocator, pool = _build_swa_tree(
            page_size=page_size, sliding_window_size=window
        )

        # First request: populate tree with 16 tokens (2 pages)
        first_len = page_size * 2
        kv1 = _swa_alloc(allocator, first_len)
        pool.write((0, slice(0, first_len)), kv1)
        req1 = _make_req(0, list(range(first_len)), 0, tree)
        tree.cache_finished_req(req1, is_insert=True)
        tree.sanity_check()

        # Second request: 24 tokens, first 16 overlap with tree
        second_len = page_size * 3
        kv2 = _swa_alloc(allocator, second_len)
        pool.write((1, slice(0, second_len)), kv2)

        req2 = _make_req(1, list(range(second_len)), 0, tree)
        batch = _make_batch(tree, allocator, pool)

        # pre_len=15: 15-2-8=5, floor to 8 -> 0. Eviction stays within matched.
        ScheduleBatch._evict_swa(batch, req2, first_len - 1)
        self.assertLessEqual(req2.swa_evicted_seqlen, first_len)

        swa_evictable_before = tree.swa_evictable_size_
        tree.cache_finished_req(req2, is_insert=True)

        # New tokens [16, 24) should all be non-tombstone
        new_tokens = second_len // page_size * page_size - first_len
        self.assertEqual(tree.swa_evictable_size_, swa_evictable_before + new_tokens)
        tree.sanity_check()

    # -- Insert case 2: total_prefix_length < swa_evicted < total_length --

    def test_insert_case2_partial_tombstone(self):
        """Partial eviction: some tombstone, some non-tombstone."""
        page_size, window = 8, 2
        tree, allocator, pool = _build_swa_tree(
            page_size=page_size, sliding_window_size=window
        )

        # seq_len=25: insert_length=24, evicted should be 8 (1 page)
        seq_len = page_size * 3 + 1
        alloc_size = (seq_len + page_size - 1) // page_size * page_size
        kv = _swa_alloc(allocator, alloc_size)
        pool.write((0, slice(0, alloc_size)), kv)

        req = _make_req(0, list(range(seq_len)), 0, tree)
        batch = _make_batch(tree, allocator, pool)
        swa_evictable_before = tree.swa_evictable_size_

        ScheduleBatch._evict_swa(batch, req, seq_len - 1)
        insert_len = seq_len // page_size * page_size
        self.assertGreater(req.swa_evicted_seqlen, 0, "Should have some eviction")
        self.assertLess(req.swa_evicted_seqlen, insert_len, "Should be partial")

        tree.cache_finished_req(req, is_insert=True)

        non_tombstone = insert_len - req.swa_evicted_seqlen
        self.assertEqual(tree.swa_evictable_size_, swa_evictable_before + non_tombstone)
        self.assertGreater(tree.full_evictable_size_, 0)
        tree.sanity_check()

    # -- Insert case 3: swa_evicted == total_length (defensive) --

    def test_insert_case3_defensive_early_return(self):
        """Simulate OLD formula to trigger case 3. Defensive early return
        must prevent non-tombstone node creation."""
        page_size, window = 8, 2
        tree, allocator, pool = _build_swa_tree(
            page_size=page_size, sliding_window_size=window
        )

        # seq_len=11: OLD formula -> evicted=8 == insert_length=8
        seq_len = page_size + window + 1
        alloc_size = (seq_len + page_size - 1) // page_size * page_size
        kv = _swa_alloc(allocator, alloc_size)
        pool.write((0, slice(0, alloc_size)), kv)

        # OLD formula (without -page_size)
        pre_len = seq_len - 1
        old_evicted = max(0, (pre_len - window) // page_size * page_size)
        insert_len = seq_len // page_size * page_size
        self.assertEqual(
            old_evicted, insert_len, "Precondition: old formula hits boundary"
        )

        # Manually free SWA as _evict_swa would
        allocator.free_swa(pool.req_to_token[0, :old_evicted])

        req = _make_req(0, list(range(seq_len)), 0, tree)
        req.swa_evicted_seqlen = old_evicted
        swa_evictable_before = tree.swa_evictable_size_

        tree.cache_finished_req(req, is_insert=True)

        self.assertEqual(tree.swa_evictable_size_, swa_evictable_before)

    # -- Integration: multiple decode turns --

    def test_multiple_decodes(self):
        """Multiple decode turns with page_size > window. No over-eviction,
        tree stays consistent throughout."""
        page_size, window = 8, 2
        tree, allocator, pool = _build_swa_tree(
            page_size=page_size, sliding_window_size=window
        )

        for turn in range(4):
            seq_len = page_size * (turn + 2) + 1
            idx = turn % pool.size
            alloc_size = (seq_len + page_size - 1) // page_size * page_size
            kv = _swa_alloc(allocator, alloc_size)
            assert kv is not None, f"Alloc failed at turn {turn}"
            pool.write((idx, slice(0, alloc_size)), kv)

            req = _make_req(idx, list(range(seq_len)), 0, tree)
            batch = _make_batch(tree, allocator, pool)
            ScheduleBatch._evict_swa(batch, req, seq_len - 1)

            insert_len = seq_len // page_size * page_size
            self.assertLess(req.swa_evicted_seqlen, insert_len, f"turn {turn}")

            tree.cache_finished_req(req, is_insert=True)
            tree.sanity_check()

    # -- Integration: page_size=1 full flow --

    def test_page_size_1_full_flow(self):
        """End-to-end with page_size=1. Fix is near no-op."""
        page_size, window = 1, 4
        tree, allocator, pool = _build_swa_tree(
            page_size=page_size, sliding_window_size=window
        )

        for seq_len in [10, 20, 30]:
            kv = _swa_alloc(allocator, seq_len)
            pool.write((0, slice(0, seq_len)), kv)

            req = _make_req(0, list(range(seq_len)), 0, tree)
            batch = _make_batch(tree, allocator, pool)
            ScheduleBatch._evict_swa(batch, req, seq_len - 1)

            self.assertEqual(req.swa_evicted_seqlen, max(0, seq_len - 1 - window - 1))

            tree.cache_finished_req(req, is_insert=True)
            tree.sanity_check()


if __name__ == "__main__":
    unittest.main()
