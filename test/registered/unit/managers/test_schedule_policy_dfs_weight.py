"""Unit tests for DFS-weight schedule-policy delegation."""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestSchedulePolicyDfsWeight(CustomTestCase):
    def test_orders_requests_by_subtree_weight(self):
        class Node:
            def __init__(self):
                self.children = {}

        root = Node()
        branch_a = Node()
        branch_b = Node()
        leaf_a1 = Node()
        leaf_a2 = Node()
        root.children = {"a": branch_a, "b": branch_b}
        branch_a.children = {"a1": leaf_a1, "a2": leaf_a2}

        class TreeCache:
            dfs_weight_order = BasePrefixCache.dfs_weight_order

            def __init__(self):
                self.root_node = root

            @staticmethod
            def resolve_node_handle(node):
                return node

        waiting_queue = [
            SimpleNamespace(last_node=branch_b, name="b"),
            SimpleNamespace(last_node=leaf_a2, name="a2"),
            SimpleNamespace(last_node=leaf_a1, name="a1-first"),
            SimpleNamespace(last_node=leaf_a1, name="a1-second"),
            SimpleNamespace(last_node=branch_a, name="a-parent"),
        ]

        SchedulePolicy._sort_by_dfs_weight(waiting_queue, TreeCache())

        self.assertEqual(
            [req.name for req in waiting_queue],
            ["a1-first", "a1-second", "a2", "a-parent", "b"],
        )


if __name__ == "__main__":
    unittest.main()
