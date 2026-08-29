"""Regression tests for scheduler-driven HiCache storage prefetch."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSchedulerHiCachePrefetch(CustomTestCase):
    @patch("sglang.srt.managers.scheduler.get_memory")
    def test_buffer_mode_prefetch_extends_device_prefix(self, get_memory):
        """A remote suffix must extend the matching device-only prefix."""
        get_memory.return_value = SimpleNamespace(
            hicache_host_memory_mode="buffer_only"
        )

        root_node = object()
        device_node = object()
        tree_cache = MagicMock()
        tree_cache.hicache_storage_pass_prefix_keys = True
        tree_cache.is_root.side_effect = lambda node: node is root_node
        tree_cache.is_backuped.return_value = False
        tree_cache.get_last_hash_value.side_effect = (
            lambda node: "device-hash" if node is device_node else "root-hash"
        )
        tree_cache.get_prefix_hash_values.side_effect = (
            lambda node: ["device-prefix-key"] if node is device_node else []
        )

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hicache_storage = True
        scheduler.tree_cache = tree_cache

        req = SimpleNamespace(
            rid="request-id",
            last_host_node=root_node,
            last_node=device_node,
            prefix_indices=[0, 1],
            host_hit_length=0,
            full_untruncated_fill_ids=[10, 11, 12, 13, 14, 15],
            extra_key=None,
            cache_salt=None,
            init_next_round_input=MagicMock(),
            _compute_max_prefix_len=MagicMock(return_value=5),
        )

        scheduler._prefetch_kvcache(req)

        tree_cache.prefetch_from_storage.assert_called_once()
        args = tree_cache.prefetch_from_storage.call_args.args
        kwargs = tree_cache.prefetch_from_storage.call_args.kwargs
        self.assertIs(args[1], device_node)
        self.assertEqual(args[2], [12, 13, 14])
        self.assertEqual(args[3], "device-hash")
        self.assertEqual(args[4], ["device-prefix-key"])
        self.assertEqual(kwargs["matched_prefix_tokens"], [10, 11])


if __name__ == "__main__":
    unittest.main()
