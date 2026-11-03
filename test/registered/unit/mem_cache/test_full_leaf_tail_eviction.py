from array import array
from unittest import TestCase
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture, match_prefix

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestFullLeafTailEviction(TestCase):
    def make_cache(self, chunk=8, *, session_cache=False):
        with (
            envs.SGLANG_PYTHON_RADIX_EVICT_TAIL_CHUNK_SIZE.override(chunk),
            patch(
                "sglang.test.kits.unified_radix_cache.fixtures.get_device",
                return_value="cpu",
            ),
        ):
            return build_fixture(
                CacheConfig(page_size=4, kv_size=64, num_layers=1),
                tree_core_backend="python",
                enable_session_radix_cache=session_cache,
            )[:2]

    def insert(self, cache, allocator, tokens):
        indices = allocator.alloc(len(tokens))
        self.assertIsNotNone(indices)
        cache.insert(InsertParams(key=RadixKey(array("q", tokens)), value=indices))
        return indices

    def assert_prefix(self, cache, tokens, expected):
        match = match_prefix(cache, tokens)
        self.assertEqual(match.device_prefix_len, len(expected))
        actual = cache.tree_core.collect_full_device_indices(
            match.last_device_node, cache.root_node_handle()
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_allocation_shortfall_preserves_prefix_and_returns_slots_once(self):
        cache, allocator = self.make_cache()
        tokens = list(range(32))
        indices = self.insert(cache, allocator, tokens)
        free_before = allocator.available_size()
        result = cache.evict_for_alloc(EvictParams(num_tokens=5))
        self.assertEqual(result.num_tokens_evicted, 8)
        self.assertEqual(allocator.available_size(), free_before + 8)
        self.assert_prefix(cache, tokens, indices[:24])
        cache.sanity_check()
        result = cache.evict(EvictParams(num_tokens=24))
        self.assertEqual(result.num_tokens_evicted, 24)
        self.assertEqual(allocator.available_size(), free_before + 32)
        reused = allocator.alloc(allocator.available_size())
        self.assertEqual(reused.unique().numel(), reused.numel())

    def test_locked_prefix_survives_and_short_suffix_is_evicted_whole(self):
        cache, allocator = self.make_cache()
        tokens = list(range(32))
        indices = self.insert(cache, allocator, tokens)
        prefix = match_prefix(cache, tokens[:12]).last_device_node
        lock = cache.inc_lock_ref(prefix).to_dec_params()
        try:
            self.assertEqual(
                cache.evict(EvictParams(num_tokens=8)).num_tokens_evicted, 8
            )
            self.assertEqual(
                cache.evict(EvictParams(num_tokens=8)).num_tokens_evicted, 12
            )
            self.assert_prefix(cache, tokens, indices[:12])
            self.assertEqual(
                cache.evict(EvictParams(num_tokens=1)).num_tokens_evicted, 0
            )
            cache.sanity_check()
        finally:
            cache.dec_lock_ref(prefix, lock)

    def test_default_and_session_cache_keep_whole_leaf_eviction(self):
        for chunk, sessions in ((0, False), (8, True)):
            with self.subTest(chunk=chunk, sessions=sessions):
                cache, allocator = self.make_cache(chunk, session_cache=sessions)
                self.insert(cache, allocator, list(range(32)))
                self.assertEqual(
                    cache.evict(EvictParams(num_tokens=5)).num_tokens_evicted, 32
                )

    def test_invalid_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "page multiple"):
            self.make_cache(6)
