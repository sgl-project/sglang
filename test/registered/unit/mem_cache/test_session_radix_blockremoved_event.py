"""Unit tests for KV cache event emission on session release — no server.

Regression: ``RadixCache.release_radix_session`` freed KV blocks and deleted
radix leaves but did not emit ``BlockRemoved`` events, unlike the regular
``evict`` path. KV-aware routers consuming the event queue were therefore left
with stale entries for blocks a session release had already freed.

These tests construct a real ``RadixCache`` (CPU memory pools) with
``enable_kv_cache_events=True`` and ``enable_session_radix_cache=True``,
insert/tag prefixes, then assert that releasing a session emits one
``BlockRemoved`` per freed node and nothing for prefixes still held by another
session.
"""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.kv_events import BlockRemoved
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _make_cache(enable_events: bool) -> RadixCache:
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
    return RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            eviction_policy="lru",
            enable_kv_cache_events=enable_events,
            enable_session_radix_cache=True,
        )
    )


class TestSessionRadixBlockRemovedEvent(CustomTestCase):
    def setUp(self):
        self.cache = _make_cache(enable_events=True)

    # -- helpers (mirror test/manual/core/test_session_radix_cache.py) --

    def _insert(self, toks):
        idx = self.cache.token_to_kv_pool_allocator.alloc(len(toks))
        self.cache.insert(
            InsertParams(key=RadixKey(array("q", toks)), value=idx.to(torch.int64))
        )

    def _tag(self, toks, sid):
        leaf = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", toks)))
        ).last_device_node
        self.cache._tag_session_leaf(
            SimpleNamespace(session_id=sid),
            RadixKey(array("q", toks)),
            node=leaf,
        )

    def _drain(self):
        return self.cache.take_events()

    def _block_removed(self, events):
        return [e for e in events if isinstance(e, BlockRemoved)]

    # -- tests --

    def test_release_emits_blockremoved_for_freed_node(self):
        self._insert([1, 2, 3, 4])
        self._tag([1, 2, 3, 4], "A")
        self._drain()  # drop BlockStored / AllBlocksCleared from setup

        freed = self.cache.release_radix_session("A")
        self.assertGreater(freed, 0, "session release should free at least one node")

        removed = self._block_removed(self._drain())
        self.assertEqual(
            len(removed),
            freed,
            f"expected {freed} BlockRemoved events (one per freed node), got {len(removed)}",
        )
        # One event per radix node; 4 tokens with page_size=1 -> 4 block hashes.
        self.assertEqual(sum(len(e.block_hashes) for e in removed), 4)

    def test_shared_prefix_emits_only_for_unique_tail(self):
        # A and B share prefix [1,2]; releasing A must free only A's tail [3,4]
        # and emit a BlockRemoved for it, leaving the shared [1,2] (held by B) intact.
        self._insert([1, 2, 3, 4])
        self._tag([1, 2, 3, 4], "A")
        self._insert([1, 2, 5, 6])
        self._tag([1, 2, 5, 6], "B")
        self._drain()

        freed = self.cache.release_radix_session("A")
        self.assertEqual(freed, 1, "only A's unique tail node should be freed")

        removed = self._block_removed(self._drain())
        self.assertEqual(len(removed), 1)
        # Tail [3,4] -> 2 block hashes (page_size=1).
        self.assertEqual(len(removed[0].block_hashes), 2)

        # Shared prefix is still cached for B.
        self.assertEqual(
            int(
                self.cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", [1, 2, 5, 6])))
                ).device_indices.numel()
            ),
            4,
        )

    def test_release_with_last_holder_frees_shared_leaf(self):
        # Same content held by two sessions -> freed only on the last close.
        self._insert([1, 2, 3, 4])
        self._tag([1, 2, 3, 4], "A")
        self._insert([1, 2, 3, 4])
        self._tag([1, 2, 3, 4], "B")
        self._drain()

        # First close: leaf still held by B -> nothing freed, no BlockRemoved.
        self.assertEqual(self.cache.release_radix_session("A"), 0)
        self.assertEqual(len(self._block_removed(self._drain())), 0)

        # Last close: leaf freed -> exactly one BlockRemoved.
        self.assertEqual(self.cache.release_radix_session("B"), 1)
        removed = self._block_removed(self._drain())
        self.assertEqual(len(removed), 1)
        self.assertEqual(len(removed[0].block_hashes), 4)

    def test_events_disabled_release_is_noop_for_queue(self):
        cache = _make_cache(enable_events=False)
        idx = cache.token_to_kv_pool_allocator.alloc(4)
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3, 4])), value=idx.to(torch.int64)
            )
        )
        leaf = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4])))
        ).last_device_node
        cache._tag_session_leaf(
            SimpleNamespace(session_id="A"),
            RadixKey(array("q", [1, 2, 3, 4])),
            node=leaf,
        )
        # No crash, no events queued when events are disabled.
        self.assertGreater(cache.release_radix_session("A"), 0)
        self.assertEqual(cache.take_events(), [])


if __name__ == "__main__":
    unittest.main()
