import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.evict_policy import QoSAwareStrategy
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestQoSHiCacheAdmission(unittest.TestCase):
    def _cache(self, *, enabled=True):
        cache = object.__new__(HiRadixCache)
        cache.cache_controller = MagicMock()
        cache.cache_controller.write_policy = "write_through_selective"
        cache.write_through_threshold = 2
        cache.enable_qos_aware_prefix_cache = enabled
        cache.eviction_strategy = QoSAwareStrategy()
        cache.schedule_low_priority_values_first = False
        cache.qos_hicache_recompute_time_per_token = 0.1
        cache.qos_hicache_transfer_time_per_token = 0.05
        cache.qos_hicache_cost_ewma_alpha = 0.2
        cache.ongoing_load_back_stats = {}
        cache.page_size = 1
        return cache

    def _node(self, *, priority=1):
        node = TreeNode(priority=priority)
        node.key = RadixKey([1, 2, 3, 4])
        node.value = torch.tensor([1, 2, 3, 4])
        return node

    def test_frequency_threshold_is_first_filter(self):
        cache = self._cache()
        cache.write_backup = MagicMock(return_value=4)
        node = self._node()

        cache._inc_hit_count(node)
        cache.write_backup.assert_not_called()

        cache._inc_hit_count(node)
        cache.write_backup.assert_called_once_with(
            node, selective_admission=True
        )

    def test_disabled_gate_preserves_original_selective_write(self):
        cache = self._cache(enabled=False)
        cache.write_backup = MagicMock(return_value=4)
        node = self._node()

        cache._inc_hit_count(node)
        cache._inc_hit_count(node)

        cache.write_backup.assert_called_once_with(
            node, selective_admission=False
        )

    def test_rejects_zero_benefit_even_when_host_has_space(self):
        cache = self._cache()
        cache.qos_hicache_transfer_time_per_token = 0.2
        cache.cache_controller.mem_pool_host.available_size.return_value = 4
        cache.evict_host = MagicMock(return_value=0)
        node = self._node()
        node.hit_count = 2

        self.assertFalse(cache._prepare_selective_host_space(node))
        cache.evict_host.assert_not_called()

    def test_admits_when_host_has_free_space(self):
        cache = self._cache()
        cache.cache_controller.mem_pool_host.available_size.return_value = 4
        cache.evict_host = MagicMock(return_value=0)
        node = self._node()
        node.hit_count = 2

        self.assertTrue(cache._prepare_selective_host_space(node))
        cache.evict_host.assert_not_called()

    def test_replaces_only_nodes_colder_than_candidate(self):
        cache = self._cache()
        cache.cache_controller.mem_pool_host.available_size.return_value = 0
        cache.evict_host = MagicMock(return_value=4)
        node = self._node(priority=3)
        node.hit_count = 2
        candidate_priority = cache._get_host_admission_priority(node)

        self.assertTrue(cache._prepare_selective_host_space(node))
        cache.evict_host.assert_called_once()
        call = cache.evict_host.call_args
        self.assertEqual(call.args[0], 4)
        self.assertEqual(call.kwargs["max_priority"], candidate_priority)
        self.assertEqual(
            call.kwargs["priority_fn"], cache._get_host_admission_priority
        )

    def test_rejects_when_colder_nodes_cannot_free_enough_space(self):
        cache = self._cache()
        cache.cache_controller.mem_pool_host.available_size.return_value = 0
        cache.evict_host = MagicMock(return_value=2)
        node = self._node()
        node.hit_count = 2

        self.assertFalse(cache._prepare_selective_host_space(node))
        cache.evict_host.assert_called_once()

    def test_host_value_uses_logical_matches_and_net_benefit(self):
        cache = self._cache()
        node = self._node(priority=3)
        node.hit_count = 2
        node.host_match_count = 3
        node.host_load_count = 1
        node.host_transfer_time_per_token = 0.02

        value, _ = cache._get_host_admission_priority(node)

        self.assertAlmostEqual(value, 5 * (0.1 - 0.02) * 3)

    def test_host_value_is_zero_when_transfer_is_slower(self):
        cache = self._cache()
        node = self._node(priority=3)
        node.hit_count = 10
        node.host_load_count = 1
        node.host_transfer_time_per_token = 0.2

        value, _ = cache._get_host_admission_priority(node)

        self.assertEqual(value, 0.0)

    def test_load_back_is_skipped_when_recompute_is_faster(self):
        cache = self._cache()
        node = self._node()
        node.host_value = torch.tensor([1, 2, 3, 4])
        node.host_load_count = 1
        node.host_transfer_time_per_token = 0.2

        self.assertFalse(cache._should_load_back([node]))

    def test_cold_load_back_uses_global_cost_estimate(self):
        cache = self._cache()
        node = self._node()
        node.host_value = torch.tensor([1, 2, 3, 4])

        self.assertTrue(cache._should_load_back([node]))

    def test_completed_load_updates_node_and_global_cost(self):
        cache = self._cache()
        node = self._node()

        cache._record_host_load([node], num_tokens=4, duration=0.4)

        self.assertEqual(node.host_load_count, 1)
        self.assertAlmostEqual(node.host_transfer_time_per_token, 0.1)
        self.assertAlmostEqual(cache.qos_hicache_transfer_time_per_token, 0.06)

    def test_host_logical_match_counts_without_load_back(self):
        cache = self._cache()
        cache.root_node = TreeNode()
        node = self._node()
        node.value = None
        node.host_value = torch.tensor([1, 2, 3, 4])
        node.parent = cache.root_node
        key = RadixKey([1, 2, 3, 4])
        cache.root_node.children[key.child_key(cache.page_size)] = node

        cache._match_prefix_helper(
            cache.root_node, key, update_cache_stats=True
        )
        cache._match_prefix_helper(
            cache.root_node, key, update_cache_stats=False
        )

        self.assertEqual(node.host_match_count, 1)
        self.assertEqual(node.host_load_count, 0)

    def test_rejected_candidate_does_not_start_host_write(self):
        cache = self._cache()
        cache.root_node = TreeNode()
        node = self._node()
        node.parent = cache.root_node
        cache._prepare_selective_host_space = MagicMock(return_value=False)

        self.assertEqual(
            cache.write_backup(node, selective_admission=True), 0
        )
        cache.cache_controller.write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
