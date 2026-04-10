"""Unit tests for SWA eviction boundary fixes.

Tests the fix for a bug where _evict_swa could over-evict when
page_size > sliding_window_size, causing _insert_helper to encounter
a fully-evicted key (case 3) and create an incorrect non-tombstone node.

Two-sided fix:
1. _evict_swa subtracts extra page_size to prevent reaching the boundary.
2. _insert_helper handles case 3 with early return (defensive).

Tests use real tree/allocator/pool objects and call real _evict_swa +
cache_finished_req code paths through thin mock wrappers for Req and
ScheduleBatch.
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


def _swa_alloc(allocator, need_size):
    """Allocate from SWA allocator for any page_size.

    SWATokenToKVPoolAllocator.alloc() asserts page_size == 1. For page_size > 1,
    allocate from the underlying paged allocators directly and set up the mapping.
    """
    if allocator.page_size == 1:
        return allocator.alloc(need_size)

    full_indices = allocator.full_attn_allocator.alloc(need_size)
    swa_indices = allocator.swa_attn_allocator.alloc(need_size)
    assert full_indices is not None and swa_indices is not None
    allocator.full_to_swa_index_mapping[full_indices] = swa_indices
    return full_indices


def _build_swa_tree(
    page_size,
    sliding_window_size,
    kv_size=1024,
    kv_size_swa=512,
    max_context_len=2048,
):
    head_num = 8
    head_dim = 128
    num_layers = 24
    global_interval = 4
    dtype = torch.bfloat16
    device = get_device()
    full_attention_layer_ids = list(range(0, num_layers, global_interval))
    full_set = set(full_attention_layer_ids)
    swa_attention_layer_ids = [i for i in range(num_layers) if i not in full_set]

    req_to_token_pool = ReqToTokenPool(
        size=8,
        max_context_len=max_context_len,
        device=device,
        enable_memory_saver=False,
    )
    kv_pool = SWAKVPool(
        size=kv_size,
        size_swa=kv_size_swa,
        page_size=page_size,
        dtype=dtype,
        head_num=head_num,
        head_dim=head_dim,
        swa_attention_layer_ids=swa_attention_layer_ids,
        full_attention_layer_ids=full_attention_layer_ids,
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
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            disable=False,
            is_eagle=False,
            sliding_window_size=sliding_window_size,
        ),
    )
    return tree, allocator, req_to_token_pool


def _make_mock_req(
    req_pool_idx, origin_input_ids, output_ids, cache_protected_len, tree
):
    """Create a mock Req with the fields needed by _evict_swa and cache_finished_req."""
    req = SimpleNamespace(
        req_pool_idx=req_pool_idx,
        origin_input_ids=origin_input_ids,
        output_ids=output_ids,
        cache_protected_len=cache_protected_len,
        swa_evicted_seqlen=0,
        extra_key=None,
        last_node=tree.root_node,
        swa_uuid_for_lock=None,
        prefix_indices=torch.tensor([], dtype=torch.int64, device=tree.device),
        _kv_committed_len=len(origin_input_ids) + len(output_ids),
    )
    req.pop_committed_kv_cache = lambda: req._kv_committed_len
    return req


def _make_mock_batch(tree, allocator, req_to_token_pool):
    """Create a mock ScheduleBatch with the fields needed by _evict_swa."""
    return SimpleNamespace(
        tree_cache=tree,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
    )


class TestSWAEvictionBoundary(unittest.TestCase):
    """End-to-end tests for the SWA eviction boundary fix.

    Uses real tree/allocator/pool objects with mock Req/ScheduleBatch wrappers
    to call the actual _evict_swa and cache_finished_req code paths.
    """

    def test_evict_then_insert_large_page(self):
        """Full flow: allocate -> _evict_swa -> cache_finished_req with page_size > window.

        With page_size=8 and window=2, simulate a decode at seq_len=11 where
        the eviction formula computes the boundary. Verify the tree and allocator
        accounting are correct after the full flow.
        """
        page_size = 8
        window = 2
        tree, allocator, req_to_token_pool = _build_swa_tree(
            page_size=page_size,
            sliding_window_size=window,
            kv_size=1024,
            kv_size_swa=512,
        )

        # Simulate a request that has been decoded to seq_len tokens
        seq_len = page_size + window + 1  # = 11
        token_ids = list(range(seq_len))
        req_pool_idx = 0

        # Allocate KV slots and write to req_to_token_pool (real allocation)
        alloc_len = seq_len
        # Round up to page boundary for allocation
        alloc_pages = (alloc_len + page_size - 1) // page_size * page_size
        kv_indices = _swa_alloc(allocator, alloc_pages)
        req_to_token_pool.write((req_pool_idx, slice(0, alloc_pages)), kv_indices)

        # Create mock req and batch
        req = _make_mock_req(
            req_pool_idx=req_pool_idx,
            origin_input_ids=token_ids,
            output_ids=[],
            cache_protected_len=0,
            tree=tree,
        )
        batch = _make_mock_batch(tree, allocator, req_to_token_pool)

        # Record state before eviction
        swa_evictable_before = tree.swa_evictable_size_

        # Call real _evict_swa (unbound method on mock batch)
        pre_len = seq_len - 1  # decode convention
        ScheduleBatch._evict_swa(batch, req, pre_len)

        # Verify: with the fix (-page_size), eviction stays below insert boundary
        insert_length = (seq_len // page_size) * page_size  # = 8
        self.assertLess(
            req.swa_evicted_seqlen,
            insert_length,
            f"Over-eviction: swa_evicted={req.swa_evicted_seqlen} "
            f">= insert_length={insert_length}",
        )

        # Call real cache_finished_req
        tree.cache_finished_req(req, is_insert=True)

        # Verify tree accounting: non-tombstone tokens should be evictable
        # insert_length=8 tokens inserted, swa_evicted of them are tombstone
        non_tombstone = insert_length - req.swa_evicted_seqlen
        self.assertEqual(
            tree.swa_evictable_size_,
            swa_evictable_before + non_tombstone,
        )
        tree.sanity_check()

    def test_evict_then_insert_multiple_decodes(self):
        """Simulate multiple decode steps with page_size > window.

        After enough decodes, swa_evicted_seqlen advances. Verify each
        cache_finished_req produces correct accounting.
        """
        page_size = 8
        window = 2
        tree, allocator, req_to_token_pool = _build_swa_tree(
            page_size=page_size,
            sliding_window_size=window,
            kv_size=1024,
            kv_size_swa=512,
        )

        # Simulate extending from seq_len=8 to seq_len=40 (multi-turn decoding)
        for turn in range(4):
            start_len = page_size + turn * page_size
            end_len = start_len + page_size
            token_ids = list(range(end_len))
            req_pool_idx = turn % req_to_token_pool.size

            alloc_pages = (end_len + page_size - 1) // page_size * page_size
            kv_indices = _swa_alloc(allocator, alloc_pages)
            assert kv_indices is not None, f"Allocation failed at turn {turn}"
            req_to_token_pool.write((req_pool_idx, slice(0, alloc_pages)), kv_indices)

            req = _make_mock_req(
                req_pool_idx=req_pool_idx,
                origin_input_ids=token_ids,
                output_ids=[],
                cache_protected_len=0,
                tree=tree,
            )
            batch = _make_mock_batch(tree, allocator, req_to_token_pool)

            # Evict at end of sequence
            pre_len = end_len - 1
            ScheduleBatch._evict_swa(batch, req, pre_len)

            insert_length = (end_len // page_size) * page_size
            self.assertLess(
                req.swa_evicted_seqlen,
                insert_length,
                f"Over-eviction at turn {turn}: swa_evicted={req.swa_evicted_seqlen} "
                f">= insert_length={insert_length}",
            )

            tree.cache_finished_req(req, is_insert=True)
            tree.sanity_check()

    def test_no_overeviction_sweep(self):
        """Sweep page_size and window combinations, verify _evict_swa never
        produces swa_evicted_seqlen >= insert_length."""
        for page_size in [4, 8, 16, 32]:
            for window in [1, 2, 4]:
                if page_size <= window:
                    continue
                tree, allocator, req_to_token_pool = _build_swa_tree(
                    page_size=page_size,
                    sliding_window_size=window,
                    kv_size=4096,
                    kv_size_swa=2048,
                )
                for seq_len in range(page_size + 1, page_size * 5):
                    alloc_pages = (seq_len + page_size - 1) // page_size * page_size
                    kv_indices = _swa_alloc(allocator, alloc_pages)
                    if kv_indices is None:
                        break
                    req_to_token_pool.write((0, slice(0, alloc_pages)), kv_indices)

                    req = _make_mock_req(
                        req_pool_idx=0,
                        origin_input_ids=list(range(seq_len)),
                        output_ids=[],
                        cache_protected_len=0,
                        tree=tree,
                    )
                    batch = _make_mock_batch(tree, allocator, req_to_token_pool)
                    ScheduleBatch._evict_swa(batch, req, seq_len - 1)

                    insert_length = (seq_len // page_size) * page_size
                    self.assertLess(
                        req.swa_evicted_seqlen,
                        insert_length,
                        f"Over-eviction: page_size={page_size}, window={window}, "
                        f"seq_len={seq_len}",
                    )

                    # Free back for next iteration
                    allocator.free_swa(kv_indices[req.swa_evicted_seqlen :])
                    allocator.full_attn_allocator.free(kv_indices)

    def test_defensive_insert_case3(self):
        """Directly test _insert_helper case 3 (swa_evicted == total_length).

        Even if _evict_swa is fixed, the defensive early return in _insert_helper
        must handle this case correctly. Simulates what would happen with the OLD
        formula by manually computing the boundary value.
        """
        page_size = 8
        window = 2
        tree, allocator, req_to_token_pool = _build_swa_tree(
            page_size=page_size,
            sliding_window_size=window,
            kv_size=1024,
            kv_size_swa=512,
        )

        seq_len = page_size + window + 1  # = 11
        token_ids = list(range(seq_len))
        alloc_pages = (seq_len + page_size - 1) // page_size * page_size
        kv_indices = _swa_alloc(allocator, alloc_pages)
        req_to_token_pool.write((0, slice(0, alloc_pages)), kv_indices)

        # Compute swa_evicted using OLD formula (without -page_size)
        pre_len = seq_len - 1
        old_evicted = max(0, pre_len - window)
        if page_size > 1:
            old_evicted = (old_evicted // page_size) * page_size
        insert_length = (seq_len // page_size) * page_size
        self.assertEqual(
            old_evicted, insert_length, "Precondition: old formula hits boundary"
        )

        # Manually free SWA tokens as _evict_swa would
        free_slots = req_to_token_pool.req_to_token[0, :old_evicted]
        allocator.free_swa(free_slots)

        # Create req with the old (buggy) swa_evicted_seqlen
        req = _make_mock_req(
            req_pool_idx=0,
            origin_input_ids=token_ids,
            output_ids=[],
            cache_protected_len=0,
            tree=tree,
        )
        req.swa_evicted_seqlen = old_evicted

        swa_evictable_before = tree.swa_evictable_size_

        # Call real cache_finished_req -- should hit case 3 early return
        tree.cache_finished_req(req, is_insert=True)

        # No non-tombstone node should be created (the core assertion for this bug)
        self.assertEqual(tree.swa_evictable_size_, swa_evictable_before)

    def test_case1_evicted_within_matched(self):
        """Case 1: swa_evicted_seqlen <= total_prefix_length.

        Insert a first request to populate tree nodes, then insert a second
        request with overlapping prefix. The second request's swa_evicted
        falls within the matched (already in tree) portion, so all new
        tokens are non-tombstone.
        """
        page_size = 8
        window = 2
        tree, allocator, req_to_token_pool = _build_swa_tree(
            page_size=page_size,
            sliding_window_size=window,
            kv_size=1024,
            kv_size_swa=512,
        )

        # First request: insert 16 tokens into tree (2 pages)
        first_len = page_size * 2
        first_ids = list(range(first_len))
        kv1 = _swa_alloc(allocator, first_len)
        req_to_token_pool.write((0, slice(0, first_len)), kv1)
        req1 = _make_mock_req(
            req_pool_idx=0,
            origin_input_ids=first_ids,
            output_ids=[],
            cache_protected_len=0,
            tree=tree,
        )
        tree.cache_finished_req(req1, is_insert=True)
        tree.sanity_check()

        # Second request: overlapping prefix (16 tokens) + 8 new tokens = 24
        second_len = page_size * 3
        second_ids = list(range(second_len))
        kv2 = _swa_alloc(allocator, second_len)
        req_to_token_pool.write((1, slice(0, second_len)), kv2)

        req2 = _make_mock_req(
            req_pool_idx=1,
            origin_input_ids=second_ids,
            output_ids=[],
            cache_protected_len=0,
            tree=tree,
        )
        batch = _make_mock_batch(tree, allocator, req_to_token_pool)

        # Evict only first page of req2's SWA (swa_evicted=8 < total_prefix_length=16)
        # Use a pre_len that produces small eviction
        req2.swa_evicted_seqlen = page_size  # manually set to 8
        swa_evictable_before = tree.swa_evictable_size_

        tree.cache_finished_req(req2, is_insert=True)

        # New tokens [16, 24) should all be non-tombstone (case 1: evicted <= matched)
        new_tokens = second_len - first_len  # 8
        self.assertEqual(
            tree.swa_evictable_size_,
            swa_evictable_before + new_tokens,
        )
        tree.sanity_check()

    def test_page_size_leq_window_no_regression(self):
        """When page_size <= sliding_window_size, the -page_size fix should not
        cause any regression. The eviction formula still works correctly and
        cache_finished_req produces correct accounting.
        """
        page_size = 4
        window = 8  # page_size < window
        tree, allocator, req_to_token_pool = _build_swa_tree(
            page_size=page_size,
            sliding_window_size=window,
            kv_size=1024,
            kv_size_swa=512,
        )

        for seq_len in [13, 17, 25, 33]:
            alloc_pages = (seq_len + page_size - 1) // page_size * page_size
            kv_indices = _swa_alloc(allocator, alloc_pages)
            if kv_indices is None:
                break
            req_pool_idx = 0
            req_to_token_pool.write((req_pool_idx, slice(0, alloc_pages)), kv_indices)

            req = _make_mock_req(
                req_pool_idx=req_pool_idx,
                origin_input_ids=list(range(seq_len)),
                output_ids=[],
                cache_protected_len=0,
                tree=tree,
            )
            batch = _make_mock_batch(tree, allocator, req_to_token_pool)

            pre_len = seq_len - 1
            ScheduleBatch._evict_swa(batch, req, pre_len)

            insert_length = (seq_len // page_size) * page_size
            # With page_size <= window, eviction should always stay well below
            self.assertLess(req.swa_evicted_seqlen, insert_length)

            tree.cache_finished_req(req, is_insert=True)
            tree.sanity_check()


if __name__ == "__main__":
    unittest.main()
