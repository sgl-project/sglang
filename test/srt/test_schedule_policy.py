import unittest

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    SchedulePolicy,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.test_utils import CustomTestCase


class TestSchedulePolicy(CustomTestCase):

    def setUp(self):
        self.tree_cache = RadixCache(None, None, False)

    def test_init_with_cache_aware_policy(self):
        policy = SchedulePolicy(
            policy="lpm", tree_cache=self.tree_cache, enable_hierarchical_cache=True
        )
        self.assertEqual(policy.policy, CacheAwarePolicy.LPM)

    def test_init_with_cache_agnostic_policy(self):
        policy = SchedulePolicy(
            policy="fcfs", tree_cache=self.tree_cache, enable_hierarchical_cache=True
        )
        self.assertEqual(policy.policy, CacheAgnosticPolicy.FCFS)

    def test_init_with_unknown_policy(self):
        with self.assertRaises(ValueError):
            SchedulePolicy(
                policy="invalid",
                tree_cache=self.tree_cache,
                enable_hierarchical_cache=True,
            )

    def test_init_with_disabled_cache(self):
        disabled_tree_cache = RadixCache(None, None, disable=True, page_size=1)
        policy = SchedulePolicy(
            policy="lpm", tree_cache=disabled_tree_cache, enable_hierarchical_cache=True
        )
        self.assertEqual(policy.policy, CacheAgnosticPolicy.FCFS)

    def test_calc_priority_fcfs(self):
        tree_cache = RadixCache(None, None, False)
        waiting_queue = [
            Req(1, "a b", [1, 2], SamplingParams()),
            Req(3, "a b c", [1, 2, 3], SamplingParams()),
            Req(2, "a", [1], SamplingParams()),
        ]

        policy = SchedulePolicy(
            policy="fcfs", tree_cache=tree_cache, enable_hierarchical_cache=True
        )
        policy.calc_priority(waiting_queue)
        # Check if FCFS keeps the original order
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 3)
        self.assertEqual(waiting_queue[2].rid, 2)


if __name__ == "__main__":
    unittest.main()
