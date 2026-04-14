import unittest

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=7, suite="stage-b-test-1-gpu-small")


class TestSLRUAccuracy(unittest.TestCase):

    def setUp(self):
        """Setup minimal memory pools for testing"""
        torch.set_default_device(None)
        device = "cpu"
        dtype = torch.float16

        # Create smaller KV cache to ensure evictions occur
        self.kv_cache = MHATokenToKVPool(
            size=8,  # Very small size to trigger evictions quickly
            page_size=1,
            dtype=dtype,
            head_num=8,
            head_dim=64,
            layer_num=1,
            device=device,
            enable_memory_saver=False,
        )

        # Create token-to-KV pool allocator
        self.token_to_kv_pool = TokenToKVPoolAllocator(
            size=8, dtype=dtype, device=device, kvcache=self.kv_cache, need_sort=False
        )

        # Create req-to-token pool
        self.req_to_token_pool = ReqToTokenPool(
            size=8, max_context_len=1024, device=device, enable_memory_saver=False
        )

        # Create a cache with the memory pools
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool,
            page_size=1,
            eviction_policy="slru",
            enable_kv_cache_events=False,
        )

        self.cache = RadixCache(params)

    def test_eviction_mechanism(self):
        """Test that SLRU eviction mechanism works correctly"""

        # Insert one key-value three times (high frequency access)
        frequent_key = RadixKey(
            token_ids=[1, 2], extra_key=None
        )  # High hit rate, should be retained
        frequent_val = torch.tensor([10, 20], dtype=torch.int64)

        # Insert the frequent key multiple times to increase its hit count
        for _ in range(3):
            self.cache.insert(InsertParams(key=frequent_key, value=frequent_val))

        # Insert first low-frequency key-value pair that should be evicted
        first_low_freq_key = RadixKey(
            token_ids=[5, 6], extra_key=None
        )  # Low hit rate, should be evicted
        first_low_freq_val = torch.tensor([50, 60], dtype=torch.int64)

        self.cache.insert(
            InsertParams(key=first_low_freq_key, value=first_low_freq_val)
        )

        # Insert other key-values once each (low frequency access) - fill up the cache
        other_keys = []
        for i in range(4):  # Reduce the number to fit in our smaller cache
            key = RadixKey(
                token_ids=[i + 10], extra_key=None
            )  # Unique keys for low-frequency items
            val = torch.tensor([i + 100], dtype=torch.int64)
            self.cache.insert(InsertParams(key=key, value=val))
            other_keys.append(key)

        # Now insert more items to trigger evictions
        for i in range(6, 10):  # Add more items to definitely exceed capacity
            key = RadixKey(
                token_ids=[i * 2], extra_key=None
            )  # Different pattern to avoid conflicts
            val = torch.tensor([i * 200], dtype=torch.int64)
            self.cache.insert(InsertParams(key=key, value=val))

        # Now trigger eviction explicitly to make space
        evict_result = self.cache.evict(
            EvictParams(num_tokens=4)
        )  # Try to evict 4 tokens worth of space

        # Check if the frequently accessed key-value is still present
        # The frequent key should have higher hit count and remain in cache due to SLRU policy
        frequent_match_result = self.cache.match_prefix(
            MatchPrefixParams(key=frequent_key)
        )

        # Check if the first low-frequency key-value has been evicted
        # The first low-freq key should have lower hit count and be evicted due to SLRU policy
        first_low_freq_match_result = self.cache.match_prefix(
            MatchPrefixParams(key=first_low_freq_key)
        )

        # Verify the frequent key is still present in cache after evictions
        self.assertIsNotNone(
            frequent_match_result,
            "Frequently accessed key should still be in cache after evictions",
        )

        # Check if the tensor is empty, which indicates the key was not found (evicted)
        is_frequent_key_present = frequent_match_result.device_indices.numel() > 0
        self.assertTrue(
            is_frequent_key_present,
            "Frequently accessed key should still be in cache after evictions",
        )

        # Verify the first low-frequency key has been evicted
        # The device_indices tensor should be empty when the key is not found
        is_first_low_freq_key_present = (
            first_low_freq_match_result.device_indices.numel() > 0
        )
        self.assertFalse(
            is_first_low_freq_key_present,
            "First inserted low-frequency key should be evicted after evictions",
        )


if __name__ == "__main__":
    unittest.main()
