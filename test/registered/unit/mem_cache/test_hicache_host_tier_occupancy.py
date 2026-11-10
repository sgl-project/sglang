"""The host-tier occupancy gauges on both tree cores: host copies of
device-resident KV count as the device mirror, and the prefetch room is the
free host pool plus exactly what host eviction can reclaim."""

import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InitLoadBackParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.hicache_write_back_fixtures import (
    PAGE,
    insert_seq,
    match_seq,
    write_back_cache,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HOST_TOKENS = 8 * PAGE


class _HostTierOccupancy:
    tree_core_backend: str

    def _evicted_chain(self):
        """A host-only parent with a host-only child, both written back."""
        cache, allocator, host_pool = write_back_cache(
            self.tree_core_backend, device_tokens=16 * PAGE, host_tokens=HOST_TOKENS
        )
        parent_seq = [1000 + t for t in range(2 * PAGE)]
        child_seq = parent_seq + [2000 + t for t in range(PAGE)]
        insert_seq(cache, allocator, parent_seq)
        insert_seq(cache, allocator, child_seq)
        parent = match_seq(cache, parent_seq).last_device_node
        child = match_seq(cache, child_seq).last_device_node
        cache.evict(EvictParams(num_tokens=len(child_seq)))
        self.assertTrue(cache.tree_core.is_full_device_evicted(parent))
        return cache, host_pool, parent_seq, child_seq, parent, child

    def _assert_room_is_reclaimable(self, cache, host_pool):
        """The room gauge is exactly what host eviction can hand a prefetch."""
        _, _, room = cache.host_tier_occupancy(max_age_s=0)
        free = host_pool.available_size()
        if room > free:
            cache.evict_host(room - free)
        self.assertEqual(host_pool.available_size(), room)

    def test_room_counts_host_only_subtrees_host_eviction_can_take(self):
        cache, host_pool, _, child_seq, _, child = self._evicted_chain()
        self.assertEqual(cache.host_tier_occupancy(max_age_s=0), (0, 0, HOST_TOKENS))

        # A pinned child keeps itself and, until it goes, its parent.
        child_lock = cache.inc_host_lock_ref(child).to_dec_params()
        self.assertEqual(
            cache.host_tier_occupancy(max_age_s=0),
            (0, 0, HOST_TOKENS - len(child_seq)),
        )
        self._assert_room_is_reclaimable(cache, host_pool)
        cache.dec_host_lock_ref(child, child_lock)
        cache.sanity_check()

    def test_unpinned_room_is_reclaimable(self):
        cache, host_pool, *_ = self._evicted_chain()
        self._assert_room_is_reclaimable(cache, host_pool)
        cache.sanity_check()

    def test_loaded_back_prefix_is_mirror_and_write_back_duplicate_room(self):
        cache, host_pool, parent_seq, _, parent, _ = self._evicted_chain()
        cache.load_back_threshold = 0  # the prefix is shorter than the default
        match = match_seq(cache, parent_seq)
        host_lock = cache.inc_host_lock_ref(parent).to_dec_params()
        req = SimpleNamespace(
            rid="r",
            last_node=match.last_device_node,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
            kv=SimpleNamespace(mamba_pool_idx=None),
        )
        loaded, _ = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=parent, host_hit_length=len(parent_seq), req=req
            )
        )
        cache.dec_host_lock_ref(parent, host_lock)
        # Issue the load and drain its ack, as the scheduler does before a forward.
        cache.ready_to_load_host_cache()
        cache.loading_check()
        self.assertEqual(loaded, len(parent_seq))
        self.assertFalse(cache.tree_core.is_full_device_evicted(parent))

        # Write-back reclaims the loaded prefix's settled host copy first.
        self.assertEqual(
            cache.host_tier_occupancy(max_age_s=0),
            (len(parent_seq), 0, HOST_TOKENS),
        )
        with envs.SGLANG_HICACHE_SKIP_HOST_DUPLICATE_RECLAIM.override(True):
            self.assertEqual(
                cache.host_tier_occupancy(max_age_s=0),
                (len(parent_seq), 0, HOST_TOKENS - len(parent_seq)),
            )
            self._assert_room_is_reclaimable(cache, host_pool)
        cache.sanity_check()


class TestHostTierOccupancyRust(_HostTierOccupancy, CustomTestCase):
    tree_core_backend = "rust"


class TestHostTierOccupancyPython(_HostTierOccupancy, CustomTestCase):
    tree_core_backend = "python"


if __name__ == "__main__":
    unittest.main()
