import unittest
from array import array
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    SchedulePolicy,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.test_utils import CustomTestCase


def _make_req(rid, origin_input_text, origin_input_ids, sampling_params=None, **kwargs):
    if sampling_params is None:
        sampling_params = SamplingParams()
    return Req(
        rid,
        origin_input_text,
        array("q", origin_input_ids),
        sampling_params,
        **kwargs,
    )


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
            _make_req(1, "a b", [1, 2]),
            _make_req(3, "a b c", [1, 2, 3]),
            _make_req(2, "a", [1]),
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

    def test_calc_priority_fcfs_cache_agnostic_in_batch_prefix_caching(self):
        tree_cache = RadixCache.create_simulated()
        shared_prefix = list(range(64))
        waiting_queue = [
            _make_req(1, "shared seed", shared_prefix + [1]),
            _make_req(2, "shared duplicate", shared_prefix + [2]),
            _make_req(3, "unrelated", [1000, 1001]),
            _make_req(4, "shared duplicate 2", shared_prefix + [3]),
        ]

        policy = SchedulePolicy(
            policy="fcfs",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        with patch(
            "sglang.srt.managers.schedule_policy."
            "ENABLE_CACHE_AGNOSTIC_IN_BATCH_PREFIX_CACHING",
            True,
        ):
            policy.calc_priority(waiting_queue)

        self.assertEqual([req.rid for req in waiting_queue], [1, 3, 2, 4])
        self.assertEqual(
            [req.defer_for_in_batch_prefix_cache for req in waiting_queue],
            [False, False, True, True],
        )

    def test_calc_priority_priority_enabled_fcfs_scheduling(self):
        tree_cache = RadixCache.create_simulated()
        r1 = _make_req(1, "a b", [1, 2])
        r2 = _make_req(3, "a b c", [1, 2, 3])
        r3 = _make_req(2, "a", [1])
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
        r1 = _make_req(1, "a b", [1, 2])
        r2 = _make_req(3, "a b c", [1, 2, 3])
        r3 = _make_req(2, "a", [1])
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
            _make_req(1, "a b", [1, 2], SamplingParams(max_new_tokens=1000)),
            _make_req(3, "a b c", [1, 2, 3], SamplingParams(max_new_tokens=10)),
            _make_req(2, "a", [1], SamplingParams(max_new_tokens=100)),
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
            _make_req(1, "a b", [1, 2], SamplingParams(max_new_tokens=1), priority=1),
            _make_req(
                3, "a b c", [1, 2, 3], SamplingParams(max_new_tokens=10), priority=0
            ),
            _make_req(2, "a", [1], SamplingParams(max_new_tokens=100), priority=0),
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
            _make_req(1, "a b", [1, 2], SamplingParams(max_new_tokens=1), priority=0),
            _make_req(
                3, "a b c", [1, 2, 3], SamplingParams(max_new_tokens=10), priority=1
            ),
            _make_req(2, "a", [1], SamplingParams(max_new_tokens=100), priority=1),
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

    def test_calc_priority_routing_key_scheduling(self):
        """Test routing-key policy: prioritize by routing key frequency in running batch."""
        tree_cache = RadixCache.create_simulated()

        running_reqs = [
            _make_req("r1", "a", [1], routing_key="key_a"),
            _make_req("r2", "b", [2], routing_key="key_a"),
            _make_req("r3", "c", [3], routing_key="key_b"),
        ]
        running_batch = ScheduleBatch(reqs=running_reqs)

        waiting_queue = [
            _make_req("w1", "d", [4], routing_key="key_b"),
            _make_req("w2", "e", [5], routing_key="key_a"),
            _make_req("w3", "f", [6], routing_key="key_c"),
        ]

        policy = SchedulePolicy(
            policy="routing-key",
            tree_cache=tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue, running_batch)

        self.assertEqual(waiting_queue[0].rid, "w2")
        self.assertEqual(waiting_queue[1].rid, "w1")
        self.assertEqual(waiting_queue[2].rid, "w3")

    def test_calc_priority_routing_key_tie_break_by_lexicographic_order(self):
        """Test routing-key policy: tie-break by lexicographic order."""
        tree_cache = RadixCache.create_simulated()

        running_reqs = [
            _make_req("r1", "a", [1], routing_key="key_b"),
            _make_req("r2", "b", [2], routing_key="key_a"),
        ]
        running_batch = ScheduleBatch(reqs=running_reqs)

        waiting_queue = [
            _make_req("w1", "d", [4], routing_key="key_b"),
            _make_req("w2", "e", [5], routing_key="key_a"),
        ]

        policy = SchedulePolicy(
            policy="routing-key",
            tree_cache=tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue, running_batch)

        self.assertEqual(waiting_queue[0].rid, "w2")
        self.assertEqual(waiting_queue[1].rid, "w1")

    def test_calc_priority_routing_key_no_match_deprioritized(self):
        """Test routing-key policy: requests without matching routing keys are deprioritized."""
        tree_cache = RadixCache.create_simulated()

        running_reqs = [
            _make_req("r1", "a", [1], routing_key="key_a"),
            _make_req("r2", "b", [2], routing_key="key_b"),
            _make_req("r3", "c", [3], routing_key="key_c"),
        ]
        running_batch = ScheduleBatch(reqs=running_reqs)

        waiting_queue = [
            _make_req("w1", "d", [4], routing_key="key_d"),
            _make_req("w2", "e", [5], routing_key="key_e"),
            _make_req("w3", "f", [6], routing_key="key_c"),
        ]

        policy = SchedulePolicy(
            policy="routing-key",
            tree_cache=tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue, running_batch)

        self.assertEqual(waiting_queue[0].rid, "w3")
        self.assertEqual(waiting_queue[1].rid, "w1")
        self.assertEqual(waiting_queue[2].rid, "w2")

    def test_calc_priority_routing_key_empty_running_batch(self):
        """Test routing-key policy: empty running batch keeps original order."""
        tree_cache = RadixCache.create_simulated()

        running_batch = ScheduleBatch(reqs=[])

        waiting_queue = [
            _make_req("w1", "d", [4], routing_key="key_a"),
            _make_req("w2", "e", [5], routing_key="key_b"),
            _make_req("w3", "f", [6], routing_key="key_c"),
        ]

        policy = SchedulePolicy(
            policy="routing-key",
            tree_cache=tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        policy.calc_priority(waiting_queue, running_batch)

        self.assertEqual(waiting_queue[0].rid, "w1")
        self.assertEqual(waiting_queue[1].rid, "w2")
        self.assertEqual(waiting_queue[2].rid, "w3")


if __name__ == "__main__":
    unittest.main()
