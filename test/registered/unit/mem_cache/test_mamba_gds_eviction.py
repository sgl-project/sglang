"""Unit tests for --radix-eviction-policy gds on the unified radix cache."""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=13, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture, match_prefix
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestMambaGDSEviction(UnifiedRadixCacheFixtures, CustomTestCase):
    cfg = CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        num_layers=2,
        full_attention_layer_ids=(0,),
        mamba_cache_size=32,
        kv_size=4096,
        max_num_reqs=32,
        max_context_len=4096,
        eviction_policy="gds",
    )
    tree_core_backend = "python"

    def setUp(self):
        self.cache, self.allocator, self.rtp = build_fixture(
            self.cfg, tree_core_backend=self.tree_core_backend
        )

    def _insert_state(self, tokens):
        """Cache a state at the end of ``tokens`` and return its node."""
        return self._insert(
            self.cache, self.allocator, self.rtp, tokens
        ).last_device_node

    def _resumes_at_end(self, tokens):
        return match_prefix(self.cache, tokens).device_prefix_len == len(tokens)

    def _evict_states(self, count):
        self.cache.evict(EvictParams(mamba_num=count))

    def test_cheap_restore_goes_before_older_expensive_state(self):
        """A shallow state goes before a colder state that saves more prefill."""
        deep_and_oldest, shallow = list(range(1000, 1400)), list(range(100))
        self._insert_state(deep_and_oldest)
        self._insert_state(shallow)
        self._insert_state(list(range(200)))

        self._evict_states(1)

        self.assertTrue(self._resumes_at_end(deep_and_oldest))
        self.assertFalse(self._resumes_at_end(shallow))

    def test_state_under_a_shallower_state_is_cheap_to_rebuild(self):
        """Eviction counts only the gap to the nearest shallower state, not the
        state's whole depth, so a deep state with a close parent goes first."""
        parent, child = list(range(200)), list(range(300))
        lone = list(range(1000, 1150))
        self._insert_state(parent)
        self._insert_state(child)
        self._insert_state(lone)

        self._evict_states(1)

        self.assertFalse(self._resumes_at_end(child))
        self.assertTrue(self._resumes_at_end(parent))
        self.assertTrue(self._resumes_at_end(lone))

    def test_evicting_an_ancestor_raises_a_descendant_gain_within_a_walk(self):
        """With its parent state gone, a state rebuilds from token 0, so it must
        outrank a lone state that a stale gain would have put above it."""
        parent, child = list(range(100)), list(range(200))
        lone = list(range(1000, 1150))
        self._insert_state(parent)
        self._insert_state(child)
        self._insert_state(lone)

        self._evict_states(2)

        self.assertFalse(self._resumes_at_end(parent))
        self.assertFalse(self._resumes_at_end(lone))
        self.assertTrue(self._resumes_at_end(child))

    def test_watermark_ages_an_unused_state(self):
        """A state nobody resumes ages out once the watermark passes its gain,
        even against a newer state with a smaller gain."""
        old, first_victim = list(range(1000, 1150)), list(range(2000, 2100))
        newer = list(range(3000, 3100))
        self._insert_state(old)
        self._insert_state(first_victim)
        self._evict_states(1)
        self.assertFalse(self._resumes_at_end(first_victim))

        self._insert_state(newer)
        self._evict_states(1)

        self.assertFalse(self._resumes_at_end(old))
        self.assertTrue(self._resumes_at_end(newer))

    def test_resumed_or_released_state_restarts_at_the_watermark(self):
        """Resuming a state, or its last request releasing it, restamps it at the
        watermark, so it outlives an unused state that is costlier to rebuild."""
        used, unused = list(range(1000, 1120)), list(range(2000, 2150))
        for how in ("match", "release"):
            with self.subTest(how=how):
                self.setUp()
                self._insert_state(unused)
                node = self._insert_state(used)
                self._insert_state(list(range(3000, 3100)))
                self._evict_states(1)
                if how == "match":
                    match_prefix(self.cache, used)
                else:
                    lock = self.cache.inc_lock_ref(node)
                    self.cache.dec_lock_ref(node, lock.to_dec_params())

                self._evict_states(1)

                self.assertTrue(self._resumes_at_end(used))
                self.assertFalse(self._resumes_at_end(unused))

    def test_state_cached_above_a_deeper_state_lowers_its_gain(self):
        """Caching a state again above a deeper one makes the deeper one cheap to
        rebuild, so it must go before a lone state its old gain would outrank."""
        parent, child = list(range(100)), list(range(200))
        lone = list(range(1000, 1150))
        self._insert_state(parent)
        self._insert_state(child)
        self._insert_state(list(range(2000, 2120)))
        self._insert_state(lone)
        self._evict_states(2)
        self._insert_state(parent)

        self._evict_states(1)

        self.assertFalse(self._resumes_at_end(child))
        self.assertTrue(self._resumes_at_end(parent))
        self.assertTrue(self._resumes_at_end(lone))

    def test_locked_state_is_not_evicted_even_when_cheapest(self):
        cheapest, expensive = list(range(100)), list(range(1000, 1400))
        node = self._insert_state(cheapest)
        self._insert_state(expensive)
        lock = self.cache.inc_lock_ref(node)

        self._evict_states(1)

        self.assertTrue(self._resumes_at_end(cheapest))
        self.assertFalse(self._resumes_at_end(expensive))
        self.cache.dec_lock_ref(node, lock.to_dec_params())


class TestMambaGDSEvictionRustCore(TestMambaGDSEviction):
    """The default core on Linux; a platform without it falls back to Python."""

    tree_core_backend = "rust"


if __name__ == "__main__":
    unittest.main()
