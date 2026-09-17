"""Tests for the cache-salt TTL: expiry bookkeeping and the tree-side teardown."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from array import array

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.cache_salt_ttl import (
    CacheSaltTtlMode,
    CacheSaltTtlPolicy,
    CacheSaltTtlReaper,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.test_utils import CustomTestCase


def make_policy(**overrides) -> CacheSaltTtlPolicy:
    kwargs = dict(
        default_ttl_s=10.0,
        max_ttl_s=20.0,
        mode=CacheSaltTtlMode.LAST_USE,
        sweep_interval_s=1.0,
        max_tracked_salts=16,
    )
    kwargs.update(overrides)
    return CacheSaltTtlPolicy(**kwargs)


class TestCacheSaltTtlReaper(CustomTestCase):
    def test_last_use_extends_the_deadline_but_first_use_does_not(self):
        sliding = CacheSaltTtlReaper(make_policy(mode=CacheSaltTtlMode.LAST_USE))
        absolute = CacheSaltTtlReaper(make_policy(mode=CacheSaltTtlMode.FIRST_USE))
        for reaper in (sliding, absolute):
            reaper.observe("s", now=0.0)
            reaper.observe("s", now=8.0)

        self.assertEqual(sliding.sweep(now=11.0), [])
        self.assertEqual(sliding.sweep(now=19.0), ["s"])
        self.assertEqual(absolute.sweep(now=9.0), [])
        self.assertEqual(absolute.sweep(now=11.0), ["s"])

    def test_a_client_ttl_is_clamped_to_the_server_maximum(self):
        policy = make_policy()
        self.assertEqual(policy.resolve_ttl(None), 10.0)
        self.assertEqual(policy.resolve_ttl(3.0), 3.0)
        self.assertEqual(policy.resolve_ttl(1e6), 20.0)
        with self.assertRaises(ValueError):
            policy.resolve_ttl(-1.0)

    def test_a_shorter_client_ttl_tightens_an_already_armed_salt(self):
        reaper = CacheSaltTtlReaper(make_policy(mode=CacheSaltTtlMode.FIRST_USE))
        reaper.observe("s", now=0.0)
        reaper.observe("s", now=0.0, requested_ttl_s=2.0)
        self.assertEqual(reaper.sweep(now=3.0), ["s"])

    def test_overflow_expires_the_salts_closest_to_their_deadline(self):
        reaper = CacheSaltTtlReaper(make_policy(max_tracked_salts=2))
        for i in range(5):
            reaper.observe(f"s{i}", now=float(i))
        # Nothing is due yet, so only the overflow is reported -- oldest first.
        self.assertEqual(reaper.sweep(now=0.5), ["s0", "s1", "s2"])
        self.assertEqual(len(reaper), 2)

    def test_an_expired_salt_rearms_a_fresh_epoch(self):
        reaper = CacheSaltTtlReaper(make_policy(mode=CacheSaltTtlMode.FIRST_USE))
        reaper.observe("s", now=0.0)
        self.assertEqual(reaper.sweep(now=11.0), ["s"])
        reaper.observe("s", now=12.0)
        self.assertEqual(reaper.sweep(now=20.0), [])
        self.assertEqual(reaper.sweep(now=23.0), ["s"])


def make_cache() -> UnifiedRadixCache:
    dtype = torch.float16
    kv_pool = MHATokenToKVPool(
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
        size=64,
        dtype=dtype,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    req_pool = ReqToTokenPool(
        size=8,
        max_context_len=128,
        device="cpu",
        enable_memory_saver=False,
    )
    return UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            eviction_policy="lru",
            tree_components=(ComponentType.FULL,),
        )
    )


def insert(cache, token_ids, cache_salt=None):
    indices = cache.token_to_kv_pool_allocator.alloc(len(token_ids))
    return cache.insert(
        InsertParams(
            key=RadixKey(array("q", token_ids), cache_salt=cache_salt),
            value=indices.to(torch.int64),
        )
    ).last_device_node


def match_len(cache, token_ids, cache_salt=None) -> int:
    return len(
        cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", token_ids), cache_salt=cache_salt)
            )
        ).device_indices
    )


class TestCacheSaltExpiry(CustomTestCase):
    def test_expiry_frees_only_the_named_salt(self):
        cache = make_cache()
        insert(cache, [1, 2, 3], cache_salt="salt-a")
        insert(cache, [1, 2, 3], cache_salt="salt-b")
        insert(cache, [1, 2, 3])
        available_before = cache.token_to_kv_pool_allocator.available_size()

        self.assertEqual(cache.expire_cache_salts(["salt-a"]), 0)

        self.assertEqual(match_len(cache, [1, 2, 3], "salt-a"), 0)
        self.assertEqual(match_len(cache, [1, 2, 3], "salt-b"), 3)
        self.assertEqual(match_len(cache, [1, 2, 3]), 3)
        self.assertEqual(
            cache.token_to_kv_pool_allocator.available_size(), available_before + 3
        )
        self.assertFalse(cache.has_pending_cache_salt_expiry())
        cache.tree_core.sanity_check([], [])

    def test_a_locked_node_defers_until_the_lock_drops(self):
        cache = make_cache()
        insert(cache, [1, 2], cache_salt="salt-a")
        insert(cache, [1, 2, 7], cache_salt="salt-a")
        held = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2]), cache_salt="salt-a"))
        ).last_device_node
        lock = cache.inc_lock_ref(held)

        self.assertEqual(cache.expire_cache_salts(["salt-a"]), 1)
        # The unlocked branch is gone; the locked prefix survives.
        self.assertTrue(cache.has_pending_cache_salt_expiry())
        self.assertEqual(match_len(cache, [1, 2], "salt-a"), 2)
        self.assertEqual(match_len(cache, [1, 2, 7], "salt-a"), 2)
        cache.tree_core.sanity_check([], [])

        cache.dec_lock_ref(held, lock.to_dec_params())
        cache.drain_expiring_cache_salts()

        self.assertFalse(cache.has_pending_cache_salt_expiry())
        self.assertEqual(match_len(cache, [1, 2], "salt-a"), 0)
        cache.tree_core.sanity_check([], [])

    def test_a_new_epoch_cancels_a_deferred_expiry(self):
        cache = make_cache()
        insert(cache, [1, 2], cache_salt="salt-a")
        held = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2]), cache_salt="salt-a"))
        ).last_device_node
        cache.inc_lock_ref(held)

        cache.expire_cache_salts(["salt-a"])
        self.assertTrue(cache.has_pending_cache_salt_expiry())

        cache.cancel_pending_cache_salt_expiry("salt-a")

        self.assertFalse(cache.has_pending_cache_salt_expiry())
        cache.drain_expiring_cache_salts()
        self.assertEqual(match_len(cache, [1, 2], "salt-a"), 2)

    def test_reset_clears_pending_expiry(self):
        cache = make_cache()
        insert(cache, [1, 2], cache_salt="salt-a")
        held = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2]), cache_salt="salt-a"))
        ).last_device_node
        cache.inc_lock_ref(held)
        cache.expire_cache_salts(["salt-a"])
        self.assertTrue(cache.has_pending_cache_salt_expiry())

        cache.reset()

        self.assertFalse(cache.has_pending_cache_salt_expiry())


if __name__ == "__main__":
    unittest.main()
