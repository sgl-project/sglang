import unittest

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    SchedulePolicy,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.test_utils import CustomTestCase


class TestSchedulePolicy(CustomTestCase):

    def setUp(self):
        self.tree_cache = RadixCache.create_simulated()

    def test_init_with_cache_aware_policy(self):
        policy = SchedulePolicy(
            policy="lpm",
            tree_cache=self.tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        self.assertEqual(policy.policy, CacheAwarePolicy.LPM)

    def test_init_with_cache_agnostic_policy(self):
        policy = SchedulePolicy(
            policy="fcfs",
            tree_cache=self.tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        self.assertEqual(policy.policy, CacheAgnosticPolicy.FCFS)

    def test_init_with_unknown_policy(self):
        with self.assertRaises(ValueError):
            SchedulePolicy(
                policy="invalid",
                tree_cache=self.tree_cache,
                enable_hierarchical_cache=True,
                enable_priority_scheduling=False,
                schedule_low_priority_values_first=False,
            )

    def test_init_with_disabled_cache(self):
        tree_cache = RadixCache.create_simulated(disable=True)
        policy = SchedulePolicy(
            policy="lpm",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        self.assertEqual(policy.policy, CacheAgnosticPolicy.FCFS)

    def test_calc_priority_fcfs(self):
        tree_cache = RadixCache.create_simulated()
        waiting_queue = [
            Req(1, "a b", [1, 2], SamplingParams()),
            Req(3, "a b c", [1, 2, 3], SamplingParams()),
            Req(2, "a", [1], SamplingParams()),
        ]

        policy = SchedulePolicy(
            policy="fcfs",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue)
        # Check if FCFS keeps the original order
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 3)
        self.assertEqual(waiting_queue[2].rid, 2)

    def test_calc_priority_priority_enabled_fcfs_scheduling(self):
        tree_cache = RadixCache.create_simulated()
        r1 = Req(1, "a b", [1, 2], SamplingParams())
        r2 = Req(3, "a b c", [1, 2, 3], SamplingParams())
        r3 = Req(2, "a", [1], SamplingParams())
        r1.priority, r1.time_stats.wait_queue_entry_time = 1, 1
        r2.priority, r2.time_stats.wait_queue_entry_time = 0, 1
        r3.priority, r3.time_stats.wait_queue_entry_time = 0, 0

        waiting_queue = [r1, r2, r3]

        policy = SchedulePolicy(
            policy="fcfs",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue)

        # Check if priority enabled fcfs ordering is applied.
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 2)
        self.assertEqual(waiting_queue[2].rid, 3)

    def test_calc_priority_priority_enabled_fcfs_scheduling_with_low_priority_values_first(
        self,
    ):
        tree_cache = RadixCache.create_simulated()
        r1 = Req(1, "a b", [1, 2], SamplingParams())
        r2 = Req(3, "a b c", [1, 2, 3], SamplingParams())
        r3 = Req(2, "a", [1], SamplingParams())
        r1.priority, r1.time_stats.wait_queue_entry_time = -1, 1
        r2.priority, r2.time_stats.wait_queue_entry_time = 0, 1
        r3.priority, r3.time_stats.wait_queue_entry_time = 0, 0

        waiting_queue = [r1, r2, r3]

        policy = SchedulePolicy(
            policy="fcfs",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=True,
        )
        policy.calc_priority(waiting_queue)
        # Check if priority enabled fcfs ordering is applied.
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 2)
        self.assertEqual(waiting_queue[2].rid, 3)

    def test_calc_priority_longest_output_first_scheduling(self):
        tree_cache = RadixCache.create_simulated()

        waiting_queue = [
            Req(1, "a b", [1, 2], SamplingParams(max_new_tokens=1000)),
            Req(3, "a b c", [1, 2, 3], SamplingParams(max_new_tokens=10)),
            Req(2, "a", [1], SamplingParams(max_new_tokens=100)),
        ]

        policy = SchedulePolicy(
            policy="lof",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue)
        # Check if priority enabled fcfs ordering is applied.
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 2)
        self.assertEqual(waiting_queue[2].rid, 3)

    def test_calc_priority_priority_enabled_longest_output_first_scheduling(self):
        tree_cache = RadixCache.create_simulated()

        waiting_queue = [
            Req(1, "a b", [1, 2], SamplingParams(max_new_tokens=1), priority=1),
            Req(3, "a b c", [1, 2, 3], SamplingParams(max_new_tokens=10), priority=0),
            Req(2, "a", [1], SamplingParams(max_new_tokens=100), priority=0),
        ]

        policy = SchedulePolicy(
            policy="lof",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue)
        # Check if priority enabled fcfs ordering is applied.
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 2)
        self.assertEqual(waiting_queue[2].rid, 3)

    def test_calc_priority_priority_enabled_longest_output_first_scheduling_with_low_priority_values_first(
        self,
    ):
        tree_cache = RadixCache.create_simulated()

        waiting_queue = [
            Req(1, "a b", [1, 2], SamplingParams(max_new_tokens=1), priority=0),
            Req(3, "a b c", [1, 2, 3], SamplingParams(max_new_tokens=10), priority=1),
            Req(2, "a", [1], SamplingParams(max_new_tokens=100), priority=1),
        ]

        policy = SchedulePolicy(
            policy="lof",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=True,
        )
        policy.calc_priority(waiting_queue)
        # Check if priority enabled fcfs ordering is applied.
        self.assertEqual(waiting_queue[0].rid, 1)
        self.assertEqual(waiting_queue[1].rid, 2)
        self.assertEqual(waiting_queue[2].rid, 3)


if __name__ == "__main__":
    unittest.main()
