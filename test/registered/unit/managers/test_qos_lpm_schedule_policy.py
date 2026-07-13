import unittest
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import CacheAwarePolicy, SchedulePolicy
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestQoSLPMSchedulePolicy(unittest.TestCase):
    def _policy(self, tree_cache, low_priority_values_first=False):
        return SchedulePolicy(
            policy="qos-lpm",
            tree_cache=tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=low_priority_values_first,
        )

    def _req(self, rid, token_ids, priority, wait_time):
        req = Req(rid, "", token_ids, SamplingParams(), priority=priority)
        req.output_ids = []
        req.time_stats.wait_queue_entry_time = wait_time
        return req

    def test_init_with_qos_lpm_policy(self):
        policy = self._policy(RadixCache.create_simulated())
        self.assertEqual(policy.policy, CacheAwarePolicy.QOS_LPM)

    def test_higher_priority_scheduled_first(self):
        tree_cache = RadixCache.create_simulated()
        tree_cache.insert(
            InsertParams(key=RadixKey([1, 2, 3]), value=torch.tensor([1, 2, 3]))
        )
        waiting_queue = [
            self._req("low-long-prefix", [1, 2, 3], 1, 1.0),
            self._req("high-no-prefix", [9, 9, 9], 5, 2.0),
        ]

        self._policy(tree_cache).calc_priority(waiting_queue)

        self.assertEqual(
            [r.rid for r in waiting_queue],
            ["high-no-prefix", "low-long-prefix"],
        )

    def test_lower_priority_value_scheduled_first_when_configured(self):
        tree_cache = RadixCache.create_simulated()
        waiting_queue = [
            self._req("value-five", [5], 5, 1.0),
            self._req("value-one", [1], 1, 2.0),
        ]

        self._policy(tree_cache, low_priority_values_first=True).calc_priority(
            waiting_queue
        )

        self.assertEqual([r.rid for r in waiting_queue], ["value-one", "value-five"])

    def test_repeated_schedule_probes_do_not_inflate_cache_hotness(self):
        tree_cache = RadixCache.create_simulated(eviction_policy="qos-aware")
        key = RadixKey([1, 2, 3])
        tree_cache.insert(InsertParams(key=key, value=torch.tensor([1, 2, 3])))
        node = tree_cache.match_prefix(
            MatchPrefixParams(key=key, update_cache_stats=False)
        ).last_device_node
        initial_hit_count = node.hit_count
        initial_access_time = node.last_access_time
        waiting_queue = [self._req("waiting", [1, 2, 3], 1, 1.0)]

        self._policy(tree_cache).calc_priority(waiting_queue)
        self._policy(tree_cache).calc_priority(waiting_queue)

        self.assertEqual(node.hit_count, initial_hit_count)
        self.assertEqual(node.last_access_time, initial_access_time)

    def test_hbm_hit_scores_higher_than_same_length_dram_hit(self):
        hbm = self._req("hbm", [1, 2, 3, 4], 1, 99.0)
        dram = self._req("dram", [5, 6, 7, 8], 1, 99.0)
        hbm.prefix_indices = torch.tensor([1, 2, 3, 4])
        dram.prefix_indices = torch.tensor([], dtype=torch.int64)
        dram.host_hit_length = 4
        waiting_queue = [dram, hbm]

        with patch(
            "sglang.srt.managers.schedule_policy.time.monotonic", return_value=100.0
        ):
            SchedulePolicy._sort_by_qos_longest_prefix(
                waiting_queue,
                set(),
                False,
                shared_weight=1.0,
                delay_weight=0.0,
                priority_weight=0.0,
                dram_discount=0.5,
                delay_reference=10.0,
            )

        self.assertEqual([r.rid for r in waiting_queue], ["hbm", "dram"])

    def test_mixed_hit_uses_discounted_dram_length(self):
        mixed = self._req("mixed", [1, 2, 3, 4], 1, 99.0)
        device_only = self._req("device-only", [5, 6, 7, 8], 1, 99.0)
        mixed.prefix_indices = torch.tensor([1, 2])
        mixed.host_hit_length = 2
        device_only.prefix_indices = torch.tensor([5, 6])
        waiting_queue = [device_only, mixed]

        with patch(
            "sglang.srt.managers.schedule_policy.time.monotonic", return_value=100.0
        ):
            SchedulePolicy._sort_by_qos_longest_prefix(
                waiting_queue,
                set(),
                False,
                shared_weight=1.0,
                delay_weight=0.0,
                priority_weight=0.0,
                dram_discount=0.5,
                delay_reference=10.0,
            )

        self.assertEqual([r.rid for r in waiting_queue], ["mixed", "device-only"])

    def test_waiting_delay_can_prevent_low_qos_starvation(self):
        high_qos = self._req("high-qos", [1], 3, 99.0)
        old_low_qos = self._req("old-low-qos", [2], 1, 0.0)
        waiting_queue = [high_qos, old_low_qos]

        with patch(
            "sglang.srt.managers.schedule_policy.time.monotonic", return_value=100.0
        ):
            SchedulePolicy._sort_by_qos_longest_prefix(
                waiting_queue,
                set(),
                False,
                shared_weight=0.0,
                delay_weight=1.0,
                priority_weight=1.0,
                dram_discount=0.5,
                delay_reference=10.0,
            )

        self.assertEqual(
            [r.rid for r in waiting_queue], ["old-low-qos", "high-qos"]
        )

    def test_invalid_location_score_config_is_rejected(self):
        with self.assertRaises(ValueError):
            SchedulePolicy(
                policy="qos-lpm",
                tree_cache=RadixCache.create_simulated(),
                enable_hierarchical_cache=False,
                enable_priority_scheduling=False,
                schedule_low_priority_values_first=False,
                qos_lpm_dram_discount=1.5,
            )

    def test_same_priority_longer_prefix_scheduled_first(self):
        tree_cache = RadixCache.create_simulated()
        tree_cache.insert(
            InsertParams(key=RadixKey([1, 2, 3]), value=torch.tensor([1, 2, 3]))
        )
        waiting_queue = [
            self._req("short", [1, 2, 9], 2, 1.0),
            self._req("long", [1, 2, 3], 2, 1.0),
        ]

        self._policy(tree_cache).calc_priority(waiting_queue)

        self.assertEqual([r.rid for r in waiting_queue], ["long", "short"])

    def test_same_priority_and_prefix_uses_arrival_order(self):
        tree_cache = RadixCache.create_simulated()
        tree_cache.insert(
            InsertParams(key=RadixKey([1, 2]), value=torch.tensor([1, 2]))
        )
        waiting_queue = [
            self._req("later", [1, 2], 2, 2.0),
            self._req("earlier", [1, 2], 2, 1.0),
        ]

        self._policy(tree_cache).calc_priority(waiting_queue)

        self.assertEqual([r.rid for r in waiting_queue], ["earlier", "later"])


if __name__ == "__main__":
    unittest.main()
