"""Real radix-tree/allocator regression; only D2H transport is simulated."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

from test_unified_radix_cache_unittest import CacheConfig, build_fixture

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType


class Completion:
    def __init__(self):
        self.done = False

    def query(self):
        return self.done

    def synchronize(self):
        self.done = True


from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMambaWriteThroughPressure(unittest.TestCase):
    def make_cache(self, cached=30, active=2):
        cfg = CacheConfig(
            components=(ComponentType.FULL, ComponentType.MAMBA),
            mamba_cache_size=32,
            kv_size=1024,
        )
        with patch("test_unified_radix_cache_unittest.get_device", return_value="cpu"):
            cache, allocator, pool = build_fixture(cfg)
        pool.mamba_allocator.alloc(active)
        nodes = []
        kv_indices = allocator.alloc(cached) if cached else None
        for i in range(cached):
            result = cache.insert(
                InsertParams(
                    key=RadixKey(array("q", range(i + 1))),
                    value=kv_indices[: i + 1],
                    mamba_value=pool.mamba_allocator.alloc(1),
                    chunked=True,
                )
            )
            nodes.append(result.last_device_node)
        return cache, pool, nodes

    def submit_backup(self, cache, nodes):
        cc = SimpleNamespace(write_policy="write_through", ack_write_queue=[])
        cache.cache_controller = cc
        cache.tree_core.enable_hicache = True

        def transfer(node_id, device_value, comp_xfers, sidecars):
            for transfers in comp_xfers.values():
                for xfer in transfers:
                    xfer.host_indices = xfer.device_indices.clone()
            cc.ack_write_queue.append(
                SimpleNamespace(
                    node_ids=[node_id],
                    finish_event=Completion(),
                    num_tokens_by_pool={"mamba": 1},
                )
            )
            return device_value.clone()

        with patch.object(cache, "_execute_kv_backup", side_effect=transfer):
            action = cache.tree_core._build_backup_kv_action(
                cache.tree_core.node_by_id(nodes[-1])
            )
            cache._execute_and_commit_kv_backup(action)
        return cc

    def test_backup_locked_pool_recovers_without_draining_whole_chain(self):
        cache, pool, nodes = self.make_cache()
        self.assertEqual(cache.mamba_evictable_size(), 30)
        self.assertEqual(pool.mamba_allocator.available_size(), 0)
        cc = self.submit_backup(cache, nodes)
        self.assertEqual(cache.mamba_evictable_size(), 0)
        component = cache.tree_core.components_by_type[ComponentType.MAMBA]
        slot = component._alloc_mamba_slot()
        self.assertIsNotNone(slot)
        self.assertEqual(len(cc.ack_write_queue), 29)
        self.assertEqual(len(cache.ongoing_write_through), 29)

    def test_multi_slot_shortfall_waits_for_enough_backups_only(self):
        cache, pool, nodes = self.make_cache()
        cc = self.submit_backup(cache, nodes)
        result = cache.evict_for_alloc(EvictParams(mamba_num=3))
        self.assertIsNotNone(pool.mamba_allocator.alloc(3))
        self.assertEqual(result.mamba_num_evicted, 3)
        self.assertEqual(len(cc.ack_write_queue), 27)

    def test_partial_free_capacity_preserves_original_allocation_target(self):
        cache, pool, nodes = self.make_cache(cached=29)
        cc = self.submit_backup(cache, nodes)
        self.assertEqual(pool.mamba_allocator.available_size(), 1)
        cache.evict_for_alloc(EvictParams(mamba_num=2))
        self.assertIsNotNone(pool.mamba_allocator.alloc(3))
        self.assertEqual(len(cc.ack_write_queue), 27)

    def test_fifo_prefix_is_completed_before_mamba_backup(self):
        cache, pool, nodes = self.make_cache()
        cc = self.submit_backup(cache, nodes)
        earlier = Completion()
        cc.ack_write_queue.insert(
            0,
            SimpleNamespace(
                node_ids=[], finish_event=earlier, num_tokens_by_pool={"kv": 1}
            ),
        )
        component = cache.tree_core.components_by_type[ComponentType.MAMBA]
        self.assertIsNotNone(component._alloc_mamba_slot())
        self.assertTrue(earlier.done)
        self.assertEqual(len(cc.ack_write_queue), 29)

    def test_no_mamba_backup_does_not_wait_for_unrelated_io(self):
        cache, pool, nodes = self.make_cache(cached=0, active=32)
        unrelated = Completion()
        cache.cache_controller = SimpleNamespace(
            write_policy="write_through",
            ack_write_queue=[SimpleNamespace(node_ids=[], finish_event=unrelated)],
        )
        component = cache.tree_core.components_by_type[ComponentType.MAMBA]
        self.assertIsNone(component._alloc_mamba_slot())
        self.assertFalse(unrelated.done)

    def test_request_owned_pool_returns_none_without_hanging(self):
        cache, pool, nodes = self.make_cache(cached=0, active=32)
        cache.cache_controller = SimpleNamespace(
            write_policy="write_through", ack_write_queue=[]
        )
        component = cache.tree_core.components_by_type[ComponentType.MAMBA]
        self.assertIsNone(component._alloc_mamba_slot())

    def test_evictable_state_does_not_wait_for_other_backups(self):
        cache, pool, nodes = self.make_cache()
        cc = self.submit_backup(cache, nodes[:1])
        event = cc.ack_write_queue[0].finish_event
        component = cache.tree_core.components_by_type[ComponentType.MAMBA]
        self.assertIsNotNone(component._alloc_mamba_slot())
        self.assertFalse(event.done)
        self.assertEqual(len(cc.ack_write_queue), 1)

    def test_backup_completion_does_not_release_request_lock(self):
        cache, pool, nodes = self.make_cache()
        receipts = [cache.inc_lock_ref(n).to_dec_params() for n in nodes]
        cc = self.submit_backup(cache, nodes)
        component = cache.tree_core.components_by_type[ComponentType.MAMBA]
        self.assertIsNone(component._alloc_mamba_slot())
        self.assertEqual(pool.mamba_allocator.available_size(), 0)
        for node, receipt in zip(nodes, receipts):
            cache.dec_lock_ref(node, receipt)
        self.assertEqual(cache.mamba_evictable_size(), 30)


if __name__ == "__main__":
    unittest.main()
