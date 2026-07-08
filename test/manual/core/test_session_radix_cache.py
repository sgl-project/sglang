"""Manual tests for the session radix cache (--enable-session-radix-cache).
Run: PYTHONPATH=python python test/manual/core/test_session_radix_cache.py
"""

import unittest
from array import array
from types import SimpleNamespace

import torch

# Import before sglang: its triton stub breaks torch's lazy _inductor import
# on hosts without real triton.
import torch._inductor.runtime.triton_heuristics  # noqa: F401

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


def make_cache(enable: bool = True) -> RadixCache:
    dtype = torch.float16
    kv = MHATokenToKVPool(
        size=128,
        page_size=1,
        dtype=dtype,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=128, dtype=dtype, device="cpu", kvcache=kv, need_sort=False
    )
    req_to_token_pool = ReqToTokenPool(
        size=8, max_context_len=1024, device="cpu", enable_memory_saver=False
    )
    return RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            eviction_policy="lru",
            enable_kv_cache_events=False,
            enable_session_radix_cache=enable,
        )
    )


class SessionCacheTestBase(unittest.TestCase):
    enable = True

    def setUp(self):
        self.cache = make_cache(self.enable)

    def insert(self, toks):
        idx = self.cache.token_to_kv_pool_allocator.alloc(len(toks))
        result = self.cache.insert(
            InsertParams(key=RadixKey(array("q", toks)), value=idx.to(torch.int64))
        )
        return result.last_device_node

    def match_node(self, toks):
        return self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", toks)))
        ).last_device_node

    def matched_len(self, toks):
        return len(
            self.cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", toks)))
            ).device_indices
        )

    def register(self, toks, sid):
        req = SimpleNamespace(
            session_id=sid,
            last_node=self.match_node(toks),
            origin_input_ids=list(toks),
            output_ids=[],
            kv_committed_len=len(toks),
            extra_key=None,
        )
        self.cache.register_session_ref(req)

    def assert_conservation(self):
        self.assertEqual(
            self.cache.unused_evictable_size_ + self.cache.referenced_evictable_size_,
            self.cache.evictable_size(),
        )


class TestTierAccounting(SessionCacheTestBase):
    def test_insert_lands_in_unused_tier(self):
        self.insert([1, 2, 3, 4])
        self.assertEqual(self.cache.unused_evictable_size_, 4)
        self.assertEqual(self.cache.referenced_evictable_size_, 0)
        self.assert_conservation()

    def test_register_moves_path_to_referenced_tier(self):
        self.insert([1, 2, 3, 4])
        self.insert([7, 8])
        self.register([1, 2, 3, 4], "s1")
        self.assertEqual(self.cache.referenced_evictable_size_, 4)
        self.assertEqual(self.cache.unused_evictable_size_, 2)
        self.assert_conservation()

    def test_lock_ref_excludes_node_from_tier_sizes(self):
        node = self.insert([1, 2, 3, 4])
        self.register([1, 2, 3, 4], "s1")
        self.cache.inc_lock_ref(node)
        self.assertEqual(self.cache.referenced_evictable_size_, 0)
        self.assert_conservation()
        self.cache.dec_lock_ref(node)
        self.assertEqual(self.cache.referenced_evictable_size_, 4)
        self.assert_conservation()

    def test_split_propagates_ref_and_session_index(self):
        self.insert([1, 2, 3, 4])
        self.register([1, 2, 3, 4], "s1")
        self.insert([1, 2, 9])
        prefix_node = self.match_node([1, 2])
        self.assertEqual(prefix_node.session_ref, 1)
        self.assertIn("s1", prefix_node.tracked_session_ids)
        self.assertIn(prefix_node, self.cache.session_id_to_ref_nodes["s1"])
        self.assertEqual(self.cache.referenced_evictable_size_, 4)
        self.assert_conservation()


class TestRegisterRelease(SessionCacheTestBase):
    def test_register_marks_whole_path(self):
        self.insert([1, 2])
        self.insert([1, 2, 3, 4])
        self.register([1, 2, 3, 4], "s1")
        self.assertEqual(self.match_node([1, 2]).session_ref, 1)
        self.assertEqual(self.match_node([1, 2, 3, 4]).session_ref, 1)

    def test_shared_prefix_counts_each_session(self):
        self.insert([1, 2, 3, 4])
        self.register([1, 2, 3, 4], "s1")
        self.register([1, 2, 3, 4], "s2")
        self.assertEqual(self.match_node([1, 2, 3, 4]).session_ref, 2)
        self.cache.release_radix_session("s1")
        self.assertEqual(self.match_node([1, 2, 3, 4]).session_ref, 1)
        self.assertEqual(self.cache.referenced_evictable_size_, 4)

    def test_release_only_dereferences(self):
        self.insert([1, 2, 3, 4])
        self.register([1, 2, 3, 4], "s1")
        released = self.cache.release_radix_session("s1")
        self.assertEqual(released, 1)
        self.assertEqual(self.matched_len([1, 2, 3, 4]), 4)
        self.assertEqual(self.match_node([1, 2, 3, 4]).session_ref, 0)
        self.assertEqual(self.cache.referenced_evictable_size_, 0)
        self.assertEqual(self.cache.unused_evictable_size_, 4)
        self.assert_conservation()

    def test_release_unknown_session_is_noop(self):
        self.assertEqual(self.cache.release_radix_session("nope"), 0)

    def test_tombstone_blocks_late_register(self):
        self.insert([1, 2, 3, 4])
        self.cache.release_radix_session("s1")
        self.register([1, 2, 3, 4], "s1")
        self.assertEqual(self.match_node([1, 2, 3, 4]).session_ref, 0)

    def test_open_clears_tombstone(self):
        self.insert([1, 2, 3, 4])
        self.cache.release_radix_session("s1")
        self.cache.open_radix_session("s1")
        self.register([1, 2, 3, 4], "s1")
        self.assertEqual(self.match_node([1, 2, 3, 4]).session_ref, 1)

    def test_register_via_path_fallback(self):
        self.insert([1, 2])
        self.insert([1, 2, 3, 4])
        req = SimpleNamespace(
            session_id="s1",
            last_node=None,
            origin_input_ids=array("q", [1, 2, 3, 4]),
            output_ids=array("q", []),
            kv_committed_len=4,
            extra_key=None,
        )
        self.cache.register_session_ref(req)
        self.assertEqual(self.match_node([1, 2]).session_ref, 1)
        self.assertEqual(self.match_node([1, 2, 3, 4]).session_ref, 1)
        self.assertEqual(self.cache.referenced_evictable_size_, 4)


class TestTieredEviction(SessionCacheTestBase):
    def test_unused_evicted_before_referenced(self):
        self.insert([1, 2, 3, 4])
        self.insert([7, 8, 9])
        self.register([1, 2, 3, 4], "s1")
        self.cache.evict(EvictParams(num_tokens=3))
        self.assertEqual(self.matched_len([7, 8, 9]), 0)
        self.assertEqual(self.matched_len([1, 2, 3, 4]), 4)

    def test_referenced_evicted_as_fallback_by_default(self):
        self.insert([1, 2, 3, 4])
        self.register([1, 2, 3, 4], "s1")
        result = self.cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(result.num_tokens_evicted, 4)
        self.assertEqual(self.matched_len([1, 2, 3, 4]), 0)

    def test_release_then_evict_frees_kv(self):
        self.insert([1, 2, 3, 4])
        self.register([1, 2, 3, 4], "s1")
        self.cache.release_radix_session("s1")
        result = self.cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(result.num_tokens_evicted, 4)
        self.assertEqual(self.cache.unused_evictable_size_, 0)
        self.assert_conservation()


class TestDisabledFlag(SessionCacheTestBase):
    enable = False

    def test_register_and_tiers_are_noop(self):
        self.insert([1, 2, 3, 4])
        self.register([1, 2, 3, 4], "s1")
        self.assertEqual(self.match_node([1, 2, 3, 4]).session_ref, 0)
        self.assertIsNone(self.match_node([1, 2, 3, 4]).tracked_session_ids)
        self.assertEqual(self.cache.unused_evictable_size_, 0)
        self.assertEqual(self.cache.referenced_evictable_size_, 0)

    def test_plain_eviction_still_works(self):
        self.insert([1, 2, 3, 4])
        result = self.cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(result.num_tokens_evicted, 4)


if __name__ == "__main__":
    unittest.main()
