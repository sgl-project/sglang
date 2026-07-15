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

    def test_admits_when_host_has_free_space(self):
        cache = self._cache()
        cache.cache_controller.mem_pool_host.available_size.return_value = 4
        cache.evict_host = MagicMock(return_value=0)

        self.assertTrue(cache._prepare_selective_host_space(self._node()))
        cache.evict_host.assert_not_called()

    def test_replaces_only_nodes_colder_than_candidate(self):
        cache = self._cache()
        cache.cache_controller.mem_pool_host.available_size.return_value = 0
        cache.evict_host = MagicMock(return_value=4)
        node = self._node(priority=3)
        candidate_priority = (0.75, 123.0)
        cache.eviction_strategy.get_priority = MagicMock(
            return_value=candidate_priority
        )

        self.assertTrue(cache._prepare_selective_host_space(node))
        cache.evict_host.assert_called_once_with(
            4, max_priority=candidate_priority
        )

    def test_rejects_when_colder_nodes_cannot_free_enough_space(self):
        cache = self._cache()
        cache.cache_controller.mem_pool_host.available_size.return_value = 0
        cache.evict_host = MagicMock(return_value=2)

        self.assertFalse(cache._prepare_selective_host_space(self._node()))

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
