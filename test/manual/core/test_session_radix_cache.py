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

    def register(self, toks, sid, generation=None, last_node=True):
        req = SimpleNamespace(
            session_id=sid,
            session_generation=generation,
            last_node=self.match_node(toks) if last_node else None,
            origin_input_ids=array("q", toks),
            output_ids=array("q", []),
            kv_committed_len=len(toks),
            extra_key=None,
        )
        self.cache.register_session_ref(req)

    def leaves(self, sid):
        return self.cache._session_leaves.get(sid, set())

    def sids(self, node):
        # session_ids is attached dynamically to frontier leaves only.
        return getattr(node, "session_ids", None)


class TestRegisterRelease(SessionCacheTestBase):
    def test_register_tags_leaf_only(self):
        self.insert([1, 2, 3, 4])
        self.register([1, 2, 3, 4], "s1")
        leaf = self.match_node([1, 2, 3, 4])
        # whole path counted, but session_ids lives only on the frontier leaf.
        self.assertEqual(self.match_node([1, 2]).session_ref, 1)
        self.assertEqual(leaf.session_ref, 1)
        self.assertEqual(self.sids(leaf), {"s1"})
        self.assertIsNone(self.sids(self.match_node([1, 2])))
        self.assertEqual(self.leaves("s1"), {leaf})

    def test_multi_turn_advancing_frontier_is_exact(self):
        self.insert([1, 2])
        self.register([1, 2], "s1")
        self.insert([1, 2, 3, 4])
        self.register([1, 2, 3, 4], "s1")
        # Shared prefix counted exactly once; frontier moved to the deeper leaf.
        self.assertEqual(self.match_node([1, 2]).session_ref, 1)
        self.assertEqual(self.match_node([1, 2, 3, 4]).session_ref, 1)
        self.assertIsNone(self.sids(self.match_node([1, 2])))
        self.assertEqual(self.leaves("s1"), {self.match_node([1, 2, 3, 4])})

    def test_shared_leaf_counts_each_session(self):
        self.insert([1, 2, 3, 4])
        self.register([1, 2, 3, 4], "s1")
        self.register([1, 2, 3, 4], "s2")
        leaf = self.match_node([1, 2, 3, 4])
        self.assertEqual(leaf.session_ref, 2)
        self.assertEqual(self.sids(leaf), {"s1", "s2"})
        self.cache.release_radix_session("s1")
        self.assertEqual(leaf.session_ref, 1)
        self.assertEqual(self.sids(leaf), {"s2"})

    def test_release_only_dereferences(self):
        self.insert([1, 2, 3, 4])
        self.register([1, 2, 3, 4], "s1")
        released = self.cache.release_radix_session("s1")
        self.assertEqual(released, 0)  # freed = nodes actually evicted; lazy => 0
        self.assertEqual(self.matched_len([1, 2, 3, 4]), 4)
        self.assertEqual(self.match_node([1, 2, 3, 4]).session_ref, 0)
        self.assertIsNone(self.sids(self.match_node([1, 2, 3, 4])))
        self.assertEqual(self.leaves("s1"), set())

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
        self.register([1, 2, 3, 4], "s1", generation=None)
        self.assertEqual(self.match_node([1, 2, 3, 4]).session_ref, 1)

    def test_register_via_match_fallback(self):
        self.insert([1, 2, 3, 4])
        self.register([1, 2, 3, 4], "s1", last_node=False)
        self.assertEqual(self.match_node([1, 2, 3, 4]).session_ref, 1)
        self.assertEqual(self.sids(self.match_node([1, 2, 3, 4])), {"s1"})

    def test_register_root_node_is_noop(self):
        # No matching KV -> last_node resolves to root_node; must not be tagged.
        self.register([9], "s1")
        self.assertIsNone(self.sids(self.cache.root_node))
        self.assertEqual(self.leaves("s1"), set())


class TestSplit(SessionCacheTestBase):
    def test_split_copies_ref_not_session_ids(self):
        self.insert([1, 2, 3, 4])
        self.register([1, 2, 3, 4], "s1")
        self.insert([1, 2, 9])  # splits the shared node at [1, 2]
        prefix = self.match_node([1, 2])
        self.assertEqual(prefix.session_ref, 1)
        self.assertIsNone(self.sids(prefix))  # holder set NOT copied onto prefix
        self.assertEqual(self.sids(self.match_node([1, 2, 3, 4])), {"s1"})


class TestSessionGeneration(SessionCacheTestBase):
    def test_reopen_rejects_stale_generation(self):
        # ABA: after close+reopen of the same id, a stale in-flight request from
        # the first incarnation must not attach to the reopened session.
        gen1 = self.cache.open_radix_session("s1")
        self.insert([1, 2, 3, 4])
        self.cache.release_radix_session("s1")
        gen2 = self.cache.open_radix_session("s1")
        self.assertNotEqual(gen1, gen2)
        self.register([1, 2, 3, 4], "s1", generation=gen1)
        self.assertEqual(self.match_node([1, 2, 3, 4]).session_ref, 0)

    def test_reopen_accepts_current_generation(self):
        self.cache.open_radix_session("s1")
        self.cache.release_radix_session("s1")
        gen2 = self.cache.open_radix_session("s1")
        self.insert([1, 2, 3, 4])
        self.register([1, 2, 3, 4], "s1", generation=gen2)
        self.assertEqual(self.match_node([1, 2, 3, 4]).session_ref, 1)


class TestTieredEviction(SessionCacheTestBase):
    def test_unused_evicted_before_referenced(self):
        self.insert([1, 2, 3, 4])
        self.insert([7, 8, 9])
        self.register([1, 2, 3, 4], "s1")
        self.cache.evict(EvictParams(num_tokens=3))
        self.assertEqual(self.matched_len([7, 8, 9]), 0)
        self.assertEqual(self.matched_len([1, 2, 3, 4]), 4)

    def test_referenced_evicted_as_fallback(self):
        self.insert([1, 2, 3, 4])
        self.register([1, 2, 3, 4], "s1")
        result = self.cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(result.num_tokens_evicted, 4)
        self.assertEqual(self.matched_len([1, 2, 3, 4]), 0)

    def test_evicting_frontier_recedes_keeps_prefix_referenced(self):
        # s1 frontier [1,2,3,4]; a locked sibling [1,2,5,6] keeps parent [1,2]
        # alive. Evicting the frontier must recede s1 to [1,2] and leave its
        # session_ref intact (the open session still reuses the prefix).
        self.insert([1, 2, 3, 4])
        self.insert([1, 2, 5, 6])
        self.register([1, 2, 3, 4], "s1")
        parent = self.match_node([1, 2])
        self.assertEqual(parent.session_ref, 1)
        self.cache.inc_lock_ref(self.match_node([1, 2, 5, 6]))
        self.cache.evict(EvictParams(num_tokens=2))
        self.assertEqual(self.matched_len([1, 2, 3, 4]), 2)  # frontier gone
        self.assertEqual(parent.session_ref, 1)  # NOT decremented
        self.assertEqual(self.sids(parent), {"s1"})  # frontier receded here
        self.assertEqual(self.leaves("s1"), {parent})

    def test_release_then_evict_frees_kv(self):
        self.insert([1, 2, 3, 4])
        self.register([1, 2, 3, 4], "s1")
        self.cache.release_radix_session("s1")
        result = self.cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(result.num_tokens_evicted, 4)
        self.assertEqual(self.matched_len([1, 2, 3, 4]), 0)


class TestTierSizeAccounting(SessionCacheTestBase):
    def _sum_ok(self):
        self.assertEqual(
            self.cache.unused_evictable_size_ + self.cache.referenced_evictable_size_,
            self.cache.evictable_size(),
        )

    def test_tier_sizes_track_evictable_size(self):
        self.insert([1, 2, 3, 4])
        self.insert([7, 8, 9])
        self.assertEqual(self.cache.unused_evictable_size_, 7)
        self._sum_ok()
        self.register([1, 2, 3, 4], "s1")
        self.assertEqual(self.cache.referenced_evictable_size_, 4)
        self.assertEqual(self.cache.unused_evictable_size_, 3)
        self._sum_ok()
        node = self.match_node([1, 2, 3, 4])
        self.cache.inc_lock_ref(node)
        self._sum_ok()  # locking moves the leaf out of evictable
        self.cache.dec_lock_ref(node)
        self._sum_ok()
        self.cache.evict(EvictParams(num_tokens=3))  # unused evicted first
        self._sum_ok()
        self.cache.release_radix_session("s1")
        self.assertEqual(self.cache.referenced_evictable_size_, 0)
        self._sum_ok()


class TestDisabledFlag(SessionCacheTestBase):
    enable = False

    def test_internal_hooks_harmless_when_disabled(self):
        # register/open are caller-gated; only the unconditional cache hooks
        # (_session_on_split on insert-split, _discard_session_leaf on evict) run
        # when disabled -- they must not accrue any session state.
        self.insert([1, 2, 3, 4])
        self.insert([1, 2, 5, 6])  # splits [1, 2] -> exercises _session_on_split
        self.assertEqual(self.match_node([1, 2]).session_ref, 0)
        self.assertIsNone(self.sids(self.match_node([1, 2, 3, 4])))
        self.cache.evict(EvictParams(num_tokens=2))  # exercises _discard_session_leaf

    def test_plain_eviction_still_works(self):
        self.insert([1, 2, 3, 4])
        result = self.cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(result.num_tokens_evicted, 4)


if __name__ == "__main__":
    unittest.main()
