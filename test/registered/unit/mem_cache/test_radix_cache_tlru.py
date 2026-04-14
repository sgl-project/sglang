import time
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

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")


def _make_cache(size, eviction_policy="tlru", xi=None, q_hat=200):
    device = "cpu"
    dtype = torch.float16
    kv_cache = MHATokenToKVPool(
        size=size,
        page_size=1,
        dtype=dtype,
        head_num=8,
        head_dim=64,
        layer_num=1,
        device=device,
        enable_memory_saver=False,
    )
    token_to_kv_pool = TokenToKVPoolAllocator(
        size=size, dtype=dtype, device=device, kvcache=kv_cache, need_sort=False
    )
    req_to_token_pool = ReqToTokenPool(
        size=8, max_context_len=1024, device=device, enable_memory_saver=False
    )
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool,
        page_size=1,
        eviction_policy=eviction_policy,
        tlru_xi_tokens=xi,
        tlru_qhat_tokens=q_hat,
        enable_kv_cache_events=False,
    )
    return RadixCache(params)


class TestTLRUCumulativeTokens(unittest.TestCase):
    """Test that cumulative_tokens is tracked correctly on tree nodes."""

    def test_root_cumulative_tokens(self):
        cache = _make_cache(size=20, xi=500, q_hat=200)
        self.assertEqual(cache.root_node.cumulative_tokens, 0)

    def test_single_insert_cumulative_tokens(self):
        cache = _make_cache(size=20, xi=500, q_hat=200)
        key = RadixKey(token_ids=[1, 2, 3, 4, 5], extra_key=None)
        val = torch.arange(5, dtype=torch.int64)
        cache.insert(InsertParams(key=key, value=val))

        # The inserted node should have cumulative_tokens = 5
        child_key = key.token_ids[0]
        node = cache.root_node.children[child_key]
        self.assertEqual(node.cumulative_tokens, 5)

    def test_shared_prefix_cumulative_tokens(self):
        cache = _make_cache(size=20, xi=500, q_hat=200)

        # Insert [1,2,3,4,5]
        key1 = RadixKey(token_ids=[1, 2, 3, 4, 5], extra_key=None)
        val1 = torch.arange(5, dtype=torch.int64)
        cache.insert(InsertParams(key=key1, value=val1))

        # Insert [1,2,3,6,7] — shares prefix [1,2,3] with key1
        key2 = RadixKey(token_ids=[1, 2, 3, 6, 7], extra_key=None)
        val2 = torch.arange(5, dtype=torch.int64) + 10
        cache.insert(InsertParams(key=key2, value=val2))

        # After split, the shared prefix node [1,2,3] should have cumulative_tokens=3
        prefix_node = cache.root_node.children[1]
        self.assertEqual(len(prefix_node.key), 3)
        self.assertEqual(prefix_node.cumulative_tokens, 3)

        # Both children should have cumulative_tokens=5
        for child in prefix_node.children.values():
            self.assertEqual(child.cumulative_tokens, 5)


class TestTLRUEviction(unittest.TestCase):
    """Test the T-LRU eviction policy behavior."""

    def test_tel_safe_evicted_first(self):
        """Long conversation tail (TEL-safe) should be evicted before short conversation (protected)."""
        # Cache size = 10 tokens
        # xi = 5, q_hat = 2
        # For a conversation of length L:
        #   B = max(0, L + 2 - 5) = max(0, L - 3)
        #   TEL-safe if node_start >= B
        cache = _make_cache(size=10, xi=5, q_hat=2)

        # Insert a long conversation: [1,2,3,4,5,6] (L=6, B=max(0,6+2-5)=3)
        # Tokens at positions [0,1,2] are protected, [3,4,5] are TEL-safe
        long_key = RadixKey(token_ids=[1, 2, 3, 4, 5, 6], extra_key=None)
        long_val = torch.arange(6, dtype=torch.int64)
        cache.insert(InsertParams(key=long_key, value=long_val))

        # Small delay to ensure different access times
        time.sleep(0.01)

        # Insert a short conversation: [10,11] (L=2, B=max(0,2+2-5)=0)
        # B=0, so all blocks are TEL-safe too, but let's use a case where short is protected
        # Actually with L=2, q_hat=2, xi=5: B=max(0,2+2-5)=0, so all TEL-safe
        # Let's use xi=3, q_hat=2 instead for clearer distinction
        # Or better: use a short conversation that is protected
        # For L=2, xi=5, q_hat=2: B=0 → all TEL-safe (node_start=0 >= 0)
        # For L=6, xi=5, q_hat=2: B=3 → tokens [3,5] are TEL-safe
        # Both are TEL-safe at the node level since entire node is one leaf

        # Let's create a scenario where one conv is clearly TEL-safe and another is protected
        # Use xi=3, q_hat=1
        cache2 = _make_cache(size=10, xi=3, q_hat=1)

        # Conv A: [1,2,3,4,5] (L=5, B=max(0,5+1-3)=3, node_start=0 < 3 → protected)
        conv_a = RadixKey(token_ids=[1, 2, 3, 4, 5], extra_key=None)
        val_a = torch.arange(5, dtype=torch.int64)
        cache2.insert(InsertParams(key=conv_a, value=val_a))
        time.sleep(0.01)

        # Conv B: [10,11] (L=2, B=max(0,2+1-3)=0, node_start=0 >= 0 → TEL-safe)
        conv_b = RadixKey(token_ids=[10, 11], extra_key=None)
        val_b = torch.arange(2, dtype=torch.int64) + 100
        cache2.insert(InsertParams(key=conv_b, value=val_b))
        time.sleep(0.01)

        # Conv C: [20,21,22] (L=3, B=max(0,3+1-3)=1, node_start=0 < 1 → protected)
        conv_c = RadixKey(token_ids=[20, 21, 22], extra_key=None)
        val_c = torch.arange(3, dtype=torch.int64) + 200
        cache2.insert(InsertParams(key=conv_c, value=val_c))

        # Total cached: 5+2+3=10 (full). Evict 2 tokens.
        # Conv B (TEL-safe, node_start=0 >= B=0) should be evicted first.
        cache2.evict(EvictParams(num_tokens=2))

        # Conv B should be evicted (TEL-safe)
        match_b = cache2.match_prefix(MatchPrefixParams(key=conv_b))
        self.assertEqual(
            match_b.device_indices.numel(), 0, "TEL-safe conv B should be evicted"
        )

        # Conv A should still be present (protected)
        match_a = cache2.match_prefix(MatchPrefixParams(key=conv_a))
        self.assertTrue(
            match_a.device_indices.numel() > 0, "Protected conv A should be retained"
        )

        # Conv C should still be present (protected)
        match_c = cache2.match_prefix(MatchPrefixParams(key=conv_c))
        self.assertTrue(
            match_c.device_indices.numel() > 0, "Protected conv C should be retained"
        )

    def test_lru_fallback_when_no_tel_safe(self):
        """When all conversations are protected (no TEL-safe), fall back to standard LRU."""
        # xi=1, q_hat=1 → B = max(0, L+1-1) = L
        # node_start = L - len(node) = 0 for single-node leaves
        # 0 >= L is only true when L=0, so effectively all non-empty nodes are protected.
        # Fallback to LRU: oldest access time evicted first.
        cache = _make_cache(size=6, xi=1, q_hat=1)

        # Conv A: oldest
        conv_a = RadixKey(token_ids=[1, 2], extra_key=None)
        val_a = torch.arange(2, dtype=torch.int64)
        cache.insert(InsertParams(key=conv_a, value=val_a))
        time.sleep(0.01)

        # Conv B: newer
        conv_b = RadixKey(token_ids=[10, 11], extra_key=None)
        val_b = torch.arange(2, dtype=torch.int64) + 100
        cache.insert(InsertParams(key=conv_b, value=val_b))
        time.sleep(0.01)

        # Conv C: newest
        conv_c = RadixKey(token_ids=[20, 21], extra_key=None)
        val_c = torch.arange(2, dtype=torch.int64) + 200
        cache.insert(InsertParams(key=conv_c, value=val_c))

        # Evict 2 tokens — should evict Conv A (oldest LRU)
        cache.evict(EvictParams(num_tokens=2))

        match_a = cache.match_prefix(MatchPrefixParams(key=conv_a))
        self.assertEqual(
            match_a.device_indices.numel(),
            0,
            "Oldest conv A should be evicted via LRU fallback",
        )

        match_b = cache.match_prefix(MatchPrefixParams(key=conv_b))
        self.assertTrue(match_b.device_indices.numel() > 0, "Conv B should be retained")

        match_c = cache.match_prefix(MatchPrefixParams(key=conv_c))
        self.assertTrue(match_c.device_indices.numel() > 0, "Conv C should be retained")

    def test_short_conversation_all_tel_safe(self):
        """When L + q_hat <= xi, B=0 and the entire conversation is TEL-safe."""
        # xi=10, q_hat=2 → for L=3: B=max(0,3+2-10)=0, all TEL-safe
        # for L=7: B=max(0,7+2-10)=0, still all TEL-safe (just barely)
        # for L=9: B=max(0,9+2-10)=1, tokens [0] protected, rest TEL-safe
        cache = _make_cache(size=12, xi=10, q_hat=2)

        # Conv A: L=3, B=0, all TEL-safe
        conv_a = RadixKey(token_ids=[1, 2, 3], extra_key=None)
        val_a = torch.arange(3, dtype=torch.int64)
        cache.insert(InsertParams(key=conv_a, value=val_a))
        time.sleep(0.01)

        # Conv B: L=9, B=1, node_start=0 < 1 → protected (single node covers [0,8])
        conv_b = RadixKey(
            token_ids=[10, 11, 12, 13, 14, 15, 16, 17, 18], extra_key=None
        )
        val_b = torch.arange(9, dtype=torch.int64) + 100
        cache.insert(InsertParams(key=conv_b, value=val_b))

        # Total: 3+9=12 (full). Evict 3 tokens.
        # Conv A is TEL-safe (B=0), Conv B is protected (B=1, node_start=0 < 1).
        # Conv A should be evicted first.
        cache.evict(EvictParams(num_tokens=3))

        match_a = cache.match_prefix(MatchPrefixParams(key=conv_a))
        self.assertEqual(
            match_a.device_indices.numel(), 0, "Short TEL-safe conv A should be evicted"
        )

        match_b = cache.match_prefix(MatchPrefixParams(key=conv_b))
        self.assertTrue(
            match_b.device_indices.numel() > 0, "Protected conv B should be retained"
        )

    def test_xi_and_qhat_affect_behavior(self):
        """Different xi/q_hat values should change which nodes are TEL-safe."""
        # With xi=100, q_hat=1: B = max(0, L+1-100) = 0 for L<99
        # → everything is TEL-safe for short conversations, falls back to LRU within TEL-safe
        cache_loose = _make_cache(size=6, xi=100, q_hat=1)
        conv_a = RadixKey(token_ids=[1, 2], extra_key=None)
        conv_b = RadixKey(token_ids=[10, 11], extra_key=None)
        cache_loose.insert(
            InsertParams(key=conv_a, value=torch.arange(2, dtype=torch.int64))
        )
        time.sleep(0.01)
        cache_loose.insert(
            InsertParams(key=conv_b, value=torch.arange(2, dtype=torch.int64) + 100)
        )
        time.sleep(0.01)
        cache_loose.insert(
            InsertParams(
                key=RadixKey(token_ids=[20, 21], extra_key=None),
                value=torch.arange(2, dtype=torch.int64) + 200,
            )
        )

        # Both are TEL-safe, so LRU within TEL-safe: oldest (conv_a) evicted first
        cache_loose.evict(EvictParams(num_tokens=2))
        match_a = cache_loose.match_prefix(MatchPrefixParams(key=conv_a))
        self.assertEqual(
            match_a.device_indices.numel(),
            0,
            "With large xi, oldest TEL-safe should be evicted first",
        )


class TestTLRUWithSplitNodes(unittest.TestCase):
    """Test T-LRU with shared prefixes that cause node splits."""

    def test_split_preserves_cumulative_tokens(self):
        """After a split, cumulative_tokens should be correct for both nodes."""
        cache = _make_cache(size=20, xi=500, q_hat=200)

        # Insert [1,2,3,4,5]
        key1 = RadixKey(token_ids=[1, 2, 3, 4, 5], extra_key=None)
        val1 = torch.arange(5, dtype=torch.int64)
        cache.insert(InsertParams(key=key1, value=val1))

        # Insert [1,2,3,6,7] → splits at position 3
        key2 = RadixKey(token_ids=[1, 2, 3, 6, 7], extra_key=None)
        val2 = torch.arange(5, dtype=torch.int64) + 10
        cache.insert(InsertParams(key=key2, value=val2))

        # Walk the tree and verify
        prefix_node = cache.root_node.children[1]
        self.assertEqual(prefix_node.cumulative_tokens, 3)
        self.assertEqual(len(prefix_node.key), 3)

        for child in prefix_node.children.values():
            self.assertEqual(child.cumulative_tokens, 5)
            self.assertEqual(len(child.key), 2)


if __name__ == "__main__":
    unittest.main()
