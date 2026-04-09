"""Unit tests for SWA eviction boundary fixes.

Tests the fix for a bug where _evict_swa could over-evict when
page_size > sliding_window_size, causing _insert_helper to encounter
a fully-evicted key (case 3) and create an incorrect non-tombstone node.

Two-sided fix:
1. _evict_swa subtracts extra page_size to prevent reaching the boundary.
2. _insert_helper handles case 3 with early return (defensive).
"""

import unittest

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


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
    return tree, allocator


class TestSWAEvictionBoundary(unittest.TestCase):
    """Test the _insert_helper boundary case where swa_evicted_seqlen == total_length."""

    def test_insert_all_evicted_boundary(self):
        """When swa_evicted_seqlen == total_prefix_length + len(key), the insert
        should free the value and return without creating a non-tombstone node.

        Before the fix, this case fell through to _add_new_node(swa_tombstone=False),
        inflating swa_evictable_size_ and causing a potential double-free.
        """
        page_size = 4
        window = 2
        tree, allocator = _build_swa_tree(
            page_size=page_size,
            sliding_window_size=window,
            kv_size=128,
            kv_size_swa=64,
        )

        full_before = allocator.full_available_size()
        swa_before = allocator.swa_available_size()
        swa_evictable_before = tree.swa_evictable_size_

        # Insert 8 tokens (2 pages) with swa_evicted_seqlen == 8 (all evicted)
        key_len = page_size * 2
        token_ids = list(range(key_len))
        kv_indices = allocator.alloc(key_len)

        result = tree.insert(
            InsertParams(
                key=RadixKey(token_ids),
                value=kv_indices,
                swa_evicted_seqlen=key_len,  # all tokens evicted
            )
        )

        # prefix_len should be 0 (nothing was inserted as non-tombstone)
        self.assertEqual(result.prefix_len, 0)

        # swa_evictable_size_ must not increase (no non-tombstone node created)
        self.assertEqual(tree.swa_evictable_size_, swa_evictable_before)

        # full pool: value was freed back, so full_available should recover
        self.assertEqual(allocator.full_available_size(), full_before)

    def test_insert_partial_evicted_still_works(self):
        """Regression: case 2 (partial tombstone) must still work correctly.

        swa_evicted_seqlen falls in the middle of the key -> split into
        tombstone + non-tombstone nodes.
        """
        page_size = 4
        window = 2
        tree, allocator = _build_swa_tree(
            page_size=page_size,
            sliding_window_size=window,
            kv_size=128,
            kv_size_swa=64,
        )

        swa_evictable_before = tree.swa_evictable_size_

        # Insert 8 tokens with swa_evicted_seqlen == 4 (first page evicted)
        key_len = page_size * 2
        token_ids = list(range(key_len))
        kv_indices = allocator.alloc(key_len)
        evicted = page_size  # first 4 tokens evicted

        result = tree.insert(
            InsertParams(
                key=RadixKey(token_ids),
                value=kv_indices,
                swa_evicted_seqlen=evicted,
            )
        )

        # Non-tombstone tail (4 tokens) should be evictable
        self.assertEqual(
            tree.swa_evictable_size_, swa_evictable_before + (key_len - evicted)
        )

        # Sanity check
        tree.sanity_check()

    def test_insert_no_eviction(self):
        """Regression: case 1 (swa_evicted <= total_prefix_length) works normally."""
        page_size = 4
        window = 2
        tree, allocator = _build_swa_tree(
            page_size=page_size,
            sliding_window_size=window,
            kv_size=128,
            kv_size_swa=64,
        )

        swa_evictable_before = tree.swa_evictable_size_

        key_len = page_size * 2
        token_ids = list(range(key_len))
        kv_indices = allocator.alloc(key_len)

        result = tree.insert(
            InsertParams(
                key=RadixKey(token_ids),
                value=kv_indices,
                swa_evicted_seqlen=0,  # nothing evicted
            )
        )

        # All 8 tokens should be non-tombstone evictable
        self.assertEqual(tree.swa_evictable_size_, swa_evictable_before + key_len)

        tree.sanity_check()

    def test_eviction_formula_large_page_size(self):
        """Verify the eviction formula does not over-evict when page_size > window.

        With page_size=8 and window=2, the old formula (pre_len - window) could
        floor-align to the insert boundary. The fix (- page_size) prevents this.
        """
        page_size = 8
        window = 2

        # Simulate the eviction formula for various seq_lens
        for seq_len in range(page_size + 1, page_size * 10):
            pre_len = seq_len - 1  # decode: pre_len = seq_len - 1
            insert_length = (seq_len // page_size) * page_size

            # New formula with the fix
            raw_evicted = pre_len - window - page_size
            evicted = max(0, (raw_evicted // page_size) * page_size)

            # The eviction frontier must never reach the insert boundary
            self.assertLess(
                evicted,
                insert_length,
                f"Over-eviction at seq_len={seq_len}: "
                f"evicted={evicted} >= insert_length={insert_length}",
            )

    def test_eviction_formula_degenerates_for_page_size_1(self):
        """When page_size=1, the extra -page_size just subtracts 1, which is
        equivalent to the old decode behavior (pre_len = seq_len - 1)."""
        page_size = 1
        window = 4

        for seq_len in range(window + 2, 100):
            pre_len = seq_len - 1
            old_evicted = max(0, pre_len - window)
            new_evicted = max(0, pre_len - window - page_size)

            # New formula evicts at most 1 less token
            self.assertGreaterEqual(old_evicted, new_evicted)
            self.assertLessEqual(old_evicted - new_evicted, 1)


if __name__ == "__main__":
    unittest.main()
