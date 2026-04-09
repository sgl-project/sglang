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


def _compute_swa_evicted(pre_len, sliding_window_size, page_size, use_fix):
    """Reproduce the _evict_swa formula (old or new)."""
    if use_fix:
        raw = pre_len - sliding_window_size - page_size
    else:
        raw = pre_len - sliding_window_size

    evicted = max(0, raw)
    if page_size > 1:
        evicted = (evicted // page_size) * page_size
    return evicted


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

    def test_old_formula_hits_boundary(self):
        """Demonstrate that the OLD eviction formula (without -page_size) produces
        swa_evicted_seqlen == insert_length when page_size > sliding_window_size.

        Example: page_size=8, window=2, seq_len=11
          insert_length = floor(11/8)*8 = 8
          old_evicted = floor((10-2)/8)*8 = floor(8/8)*8 = 8  == insert_length!
        """
        page_size = 8
        window = 2

        # Find a seq_len where the old formula hits the boundary
        found_boundary = False
        for seq_len in range(page_size + 1, page_size * 10):
            pre_len = seq_len - 1
            insert_length = (seq_len // page_size) * page_size
            if insert_length == 0:
                continue
            old_evicted = _compute_swa_evicted(
                pre_len, window, page_size, use_fix=False
            )
            if old_evicted == insert_length:
                found_boundary = True
                # The new formula must NOT hit the boundary
                new_evicted = _compute_swa_evicted(
                    pre_len, window, page_size, use_fix=True
                )
                self.assertLess(
                    new_evicted,
                    insert_length,
                    f"New formula still hits boundary at seq_len={seq_len}",
                )
                break

        self.assertTrue(
            found_boundary,
            "Expected to find a seq_len where old formula hits boundary",
        )

    def test_new_formula_never_hits_boundary(self):
        """The new eviction formula (with -page_size) must never produce
        swa_evicted_seqlen >= insert_length, for any page_size > window."""
        for page_size in [4, 8, 16, 32, 64, 128, 256]:
            for window in [1, 2, 4, 8]:
                if page_size <= window:
                    continue
                for seq_len in range(page_size + 1, page_size * 20):
                    pre_len = seq_len - 1
                    insert_length = (seq_len // page_size) * page_size
                    if insert_length == 0:
                        continue
                    evicted = _compute_swa_evicted(
                        pre_len, window, page_size, use_fix=True
                    )
                    self.assertLess(
                        evicted,
                        insert_length,
                        f"Over-eviction: page_size={page_size}, window={window}, "
                        f"seq_len={seq_len}, evicted={evicted}, "
                        f"insert_length={insert_length}",
                    )

    def test_insert_case3_with_paged_alloc(self):
        """End-to-end: use page_size > window, compute swa_evicted with OLD formula
        to trigger case 3, then verify _insert_helper handles it correctly.

        This tests the defensive fix in _insert_helper with real paged allocation.
        """
        page_size = 8
        window = 2
        tree, allocator = _build_swa_tree(
            page_size=page_size,
            sliding_window_size=window,
            kv_size=1024,
            kv_size_swa=512,
        )

        # Pick seq_len where old formula hits boundary
        # seq_len=11: insert_length=8, old_evicted=8
        seq_len = page_size + window + 1  # = 11
        pre_len = seq_len - 1
        insert_length = (seq_len // page_size) * page_size
        old_evicted = _compute_swa_evicted(pre_len, window, page_size, use_fix=False)
        self.assertEqual(
            old_evicted, insert_length, "Precondition: old formula hits boundary"
        )

        # Allocate page-aligned tokens and insert with the buggy swa_evicted_seqlen
        token_ids = list(range(insert_length))
        kv_indices = _swa_alloc(allocator, insert_length)
        swa_evictable_before = tree.swa_evictable_size_
        full_available_before = allocator.full_available_size()

        result = tree.insert(
            InsertParams(
                key=RadixKey(token_ids),
                value=kv_indices,
                swa_evicted_seqlen=old_evicted,  # == insert_length, case 3
            )
        )

        # With the fix: early return, no non-tombstone node created
        self.assertEqual(result.prefix_len, 0)
        self.assertEqual(tree.swa_evictable_size_, swa_evictable_before)
        # full pool value was freed back
        self.assertEqual(allocator.full_available_size(), full_available_before)

    def test_insert_case2_with_paged_alloc(self):
        """Regression: case 2 (partial tombstone) with page_size > 1 still works.

        Use the NEW formula which produces swa_evicted < insert_length.
        """
        page_size = 8
        window = 2
        tree, allocator = _build_swa_tree(
            page_size=page_size,
            sliding_window_size=window,
            kv_size=1024,
            kv_size_swa=512,
        )

        # seq_len=17: insert_length=16, new_evicted=floor((16-2-8)/8)*8=0
        # seq_len=25: insert_length=24, new_evicted=floor((24-2-8)/8)*8=8
        seq_len = page_size * 3 + 1  # = 25
        pre_len = seq_len - 1
        insert_length = (seq_len // page_size) * page_size  # = 24
        new_evicted = _compute_swa_evicted(pre_len, window, page_size, use_fix=True)
        self.assertGreater(new_evicted, 0, "Precondition: some eviction occurs")
        self.assertLess(new_evicted, insert_length, "Precondition: partial eviction")

        token_ids = list(range(insert_length))
        kv_indices = _swa_alloc(allocator, insert_length)
        swa_evictable_before = tree.swa_evictable_size_

        tree.insert(
            InsertParams(
                key=RadixKey(token_ids),
                value=kv_indices,
                swa_evicted_seqlen=new_evicted,
            )
        )

        # Non-tombstone tail should be evictable
        non_tombstone_len = insert_length - new_evicted
        self.assertEqual(
            tree.swa_evictable_size_, swa_evictable_before + non_tombstone_len
        )
        tree.sanity_check()

    def test_insert_case1_no_eviction(self):
        """Regression: case 1 (no eviction) with page_size > 1 works normally."""
        page_size = 8
        window = 2
        tree, allocator = _build_swa_tree(
            page_size=page_size,
            sliding_window_size=window,
            kv_size=1024,
            kv_size_swa=512,
        )

        insert_length = page_size * 2  # 16 tokens
        token_ids = list(range(insert_length))
        kv_indices = _swa_alloc(allocator, insert_length)
        swa_evictable_before = tree.swa_evictable_size_

        tree.insert(
            InsertParams(
                key=RadixKey(token_ids),
                value=kv_indices,
                swa_evicted_seqlen=0,
            )
        )

        self.assertEqual(tree.swa_evictable_size_, swa_evictable_before + insert_length)
        tree.sanity_check()

    def test_formula_degenerates_for_page_size_1(self):
        """When page_size=1, the extra -page_size subtracts 1 -- at most 1 less
        token evicted compared to the old formula."""
        page_size = 1
        window = 4

        for seq_len in range(window + 2, 100):
            pre_len = seq_len - 1
            old_evicted = _compute_swa_evicted(
                pre_len, window, page_size, use_fix=False
            )
            new_evicted = _compute_swa_evicted(pre_len, window, page_size, use_fix=True)

            self.assertGreaterEqual(old_evicted, new_evicted)
            self.assertLessEqual(old_evicted - new_evicted, 1)


if __name__ == "__main__":
    unittest.main()
