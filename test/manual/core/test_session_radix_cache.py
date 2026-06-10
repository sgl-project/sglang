"""Manual test for the session radix cache (--enable-session-radix-cache).
Run directly: python test/manual/core/test_session_radix_cache.py
"""

import unittest
from array import array
from types import SimpleNamespace

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


class TestSessionRadixCache(unittest.TestCase):
    def setUp(self):
        dtype = torch.float16
        kv = MHATokenToKVPool(
            size=64, page_size=1, dtype=dtype, head_num=2, head_dim=8,
            layer_num=1, device="cpu", enable_memory_saver=False,
        )
        allocator = TokenToKVPoolAllocator(
            size=64, dtype=dtype, device="cpu", kvcache=kv, need_sort=False
        )
        req_to_token_pool = ReqToTokenPool(
            size=8, max_context_len=1024, device="cpu", enable_memory_saver=False
        )
        self.cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                eviction_policy="lru",
                enable_kv_cache_events=False,
            )
        )

    def _insert(self, toks):
        idx = self.cache.token_to_kv_pool_allocator.alloc(len(toks))
        self.cache.insert(
            InsertParams(key=RadixKey(array("q", toks)), value=idx.to(torch.int64))
        )

    def _tag(self, toks, sid):
        self.cache._tag_session_leaf(
            SimpleNamespace(session_id=sid), RadixKey(array("q", toks))
        )

    def _cached(self, toks):
        return int(
            self.cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", toks)))
            ).device_indices.numel()
        )

    def _leaf(self, toks):
        return self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", toks)))
        ).last_device_node

    def test_shared_prefix_frees_only_unique_tail(self):
        # A/B share prefix [1,2]; close(A) frees only A's tail, B + shared stay.
        self._insert([1, 2, 3, 4])
        self._tag([1, 2, 3, 4], "A")
        self._insert([1, 2, 5, 6])
        self._tag([1, 2, 5, 6], "B")
        self.assertGreater(self.cache.release_session("A"), 0)
        self.assertEqual(self._cached([1, 2, 3, 4]), 2)  # only shared [1,2] left
        self.assertEqual(self._cached([1, 2, 5, 6]), 4)  # B intact

    def test_same_leaf_freed_only_on_last_holder(self):
        # Identical content -> one leaf held by {A,B}; freed only on last close.
        self._insert([1, 2, 3, 4])
        self._tag([1, 2, 3, 4], "A")
        self._insert([1, 2, 3, 4])
        self._tag([1, 2, 3, 4], "B")
        self.assertEqual(self._leaf([1, 2, 3, 4]).session_ids, {"A", "B"})
        self.assertEqual(self.cache.release_session("A"), 0)  # B still holds
        self.assertEqual(self._cached([1, 2, 3, 4]), 4)
        self.assertEqual(self.cache.release_session("B"), 1)  # last holder frees
        self.assertEqual(self._cached([1, 2, 3, 4]), 0)

    def test_tag_is_lru_neutral_not_pinned(self):
        # The tag must add no lock/pin: a tagged, never-closed node is evictable.
        self._insert([1, 2, 3, 4])
        self._tag([1, 2, 3, 4], "S")
        leaf = self._leaf([1, 2, 3, 4])
        self.assertEqual(leaf.lock_ref, 0)
        self.assertEqual(self.cache.protected_size(), 0)
        self.assertIn(leaf, self.cache.evictable_leaves)
        self.cache.evict(EvictParams(num_tokens=4))  # LRU reclaims it while open
        self.assertEqual(self._cached([1, 2, 3, 4]), 0)
        self.assertEqual(self.cache.release_session("S"), 0)  # late close is a no-op


if __name__ == "__main__":
    unittest.main()
