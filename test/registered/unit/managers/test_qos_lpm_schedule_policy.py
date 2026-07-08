import unittest

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import CacheAwarePolicy, SchedulePolicy
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestQoSLPMSchedulePolicy(unittest.TestCase):
    def _policy(self, tree_cache):
        return SchedulePolicy(
            policy="qos-lpm",
            tree_cache=tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
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

    def test_same_priority_longer_prefix_scheduled_first(self):
        tree_cache = RadixCache.create_simulated()
        tree_cache.insert(
            InsertParams(key=RadixKey([1, 2, 3]), value=torch.tensor([1, 2, 3]))
        )
        waiting_queue = [
            self._req("short", [1, 2], 2, 1.0),
            self._req("long", [1, 2, 3], 2, 2.0),
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
