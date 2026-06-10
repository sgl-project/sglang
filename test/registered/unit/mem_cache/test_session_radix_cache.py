"""Unit tests for the session radix cache (``--enable-session-radix-cache``):
the ``SessionRadixCacheMixin`` tag / release path on a real ``RadixCache``.

Covers the two cases that motivated the per-node holder set (PR #27058):
  * shared PREFIX, distinct content -> close frees only the unique tail
  * byte-identical content -> SAME leaf, held by both sessions -> the node is
    freed only when its LAST holder closes (order-independent), never on the
    first close and never leaked.
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
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestSessionRadixCache(unittest.TestCase):
    def setUp(self):
        torch.set_default_device(None)
        dtype = torch.float16
        kv = MHATokenToKVPool(
            size=64,
            page_size=1,
            dtype=dtype,
            head_num=2,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
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

    # ---- helpers ----------------------------------------------------------
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

    # ---- tests ------------------------------------------------------------
    def test_non_session_req_not_tagged(self):
        """A request without a session_id leaves the leaf untagged."""
        self._insert([1, 2, 3, 4])
        self._tag([1, 2, 3, 4], None)
        self.assertIsNone(self._leaf([1, 2, 3, 4]).session_ids)
        self.assertEqual(self.cache.release_session("nope"), 0)

    def test_retag_same_session_is_idempotent(self):
        """The same session re-tagging its leaf (turn over turn) does not grow
        the holder set."""
        self._insert([1, 2, 3, 4])
        self._tag([1, 2, 3, 4], "A")
        self._tag([1, 2, 3, 4], "A")
        self.assertEqual(self._leaf([1, 2, 3, 4]).session_ids, {"A"})

    def test_shared_prefix_frees_only_unique_tail(self):
        """Sessions A/B share prefix [1,2] but diverge; close(A) frees only A's
        unique tail, B and the shared prefix stay resident."""
        self._insert([1, 2, 3, 4])
        self._tag([1, 2, 3, 4], "A")
        self._insert([1, 2, 5, 6])  # splits at [1,2]
        self._tag([1, 2, 5, 6], "B")

        self.assertEqual(self._cached([1, 2, 3, 4]), 4)
        self.assertEqual(self._cached([1, 2, 5, 6]), 4)

        freed = self.cache.release_session("A")
        self.assertGreater(freed, 0)
        self.assertEqual(self._cached([1, 2, 3, 4]), 2)  # only shared [1,2] left
        self.assertEqual(self._cached([1, 2, 5, 6]), 4)  # B intact
        self.assertEqual(self._cached([1, 2]), 2)  # shared prefix intact

    def test_same_leaf_freed_only_on_last_holder_A_then_B(self):
        """Byte-identical content -> one leaf held by {A,B}. close(A) keeps it
        (B still holds); close(B) frees it."""
        self._insert([1, 2, 3, 4])
        self._tag([1, 2, 3, 4], "A")
        self._insert([1, 2, 3, 4])  # identical -> same leaf
        self._tag([1, 2, 3, 4], "B")
        self.assertEqual(self._leaf([1, 2, 3, 4]).session_ids, {"A", "B"})

        self.assertEqual(self.cache.release_session("A"), 0)  # B still holds
        self.assertEqual(self._cached([1, 2, 3, 4]), 4)
        self.assertEqual(self._leaf([1, 2, 3, 4]).session_ids, {"B"})

        self.assertEqual(self.cache.release_session("B"), 1)  # last holder
        self.assertEqual(self._cached([1, 2, 3, 4]), 0)

    def test_tagged_node_is_lru_neutral_not_pinned(self):
        """The session tag must add NO lock/pin: a tagged-but-never-closed node
        stays evictable (lock_ref 0, counted as evictable not protected) and is
        reclaimed by ordinary LRU under pressure -- the core invariant of the
        feature (a session under pressure is just evicted and re-prefilled)."""
        self._insert([1, 2, 3, 4])
        self._tag([1, 2, 3, 4], "S")  # tagged, never closed
        leaf = self._leaf([1, 2, 3, 4])

        self.assertEqual(leaf.session_ids, {"S"})
        self.assertEqual(leaf.lock_ref, 0)  # tag added no lock
        self.assertIn(leaf, self.cache.evictable_leaves)
        self.assertEqual(self.cache.protected_size(), 0)  # tag pinned nothing
        self.assertEqual(self.cache.evictable_size(), 4)  # all evictable

        # LRU reclaims the session KV while the session is still open.
        self.cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(self._cached([1, 2, 3, 4]), 0)
        # A late close after LRU eviction is a graceful no-op (nothing leaked).
        self.assertEqual(self.cache.release_session("S"), 0)

    def test_same_leaf_freed_only_on_last_holder_B_then_A(self):
        """Same as above but closing in the other order -- result is identical
        (order-independent), unlike the last-writer-wins scalar tag."""
        self._insert([1, 2, 3, 4])
        self._tag([1, 2, 3, 4], "A")
        self._insert([1, 2, 3, 4])
        self._tag([1, 2, 3, 4], "B")

        self.assertEqual(self.cache.release_session("B"), 0)  # A still holds
        self.assertEqual(self._cached([1, 2, 3, 4]), 4)
        self.assertEqual(self.cache.release_session("A"), 1)  # last holder
        self.assertEqual(self._cached([1, 2, 3, 4]), 0)


if __name__ == "__main__":
    unittest.main()
