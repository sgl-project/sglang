"""LPM can price outstanding host loads separately from recomputation."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def request(name, *, cached, uncached=1, host=0, loaded=0):
    return SimpleNamespace(
        rid=name,
        origin_input_ids=range(cached + uncached),
        num_matched_prefix_tokens=cached,
        host_hit_length=host,
        host_loaded_length=loaded,
    )


class TestHiCacheLPMCost(CustomTestCase):
    def policy(self, weight, *, role="prefill", hicache=True):
        with (
            patch.object(
                envs.SGLANG_HICACHE_LPM_COMPUTE_WEIGHT, "get", return_value=weight
            ),
            patch(
                "sglang.srt.managers.schedule_policy.get_disagg",
                return_value=SimpleNamespace(disaggregation_mode=role),
            ),
        ):
            return SchedulePolicy(
                "lpm", RadixCache.create_simulated(), hicache, False, False
            )

    def test_enabled_through_calc_priority(self):
        policy = self.policy(64)
        cold = request("cold", cached=100, uncached=10)
        host = request("host", cached=200, uncached=1, host=150)
        device = request("device", cached=10, uncached=1)
        with patch.object(policy, "_compute_prefix_matches", return_value=set()):
            result = policy.calc_priority([cold, host, device])
        self.assertEqual([req.rid for req in result], ["device", "host", "cold"])

    def test_zero_preserves_longest_prefix_order(self):
        policy = self.policy(0)
        queue = [request("short", cached=10), request("long", cached=100, host=100)]
        with patch.object(policy, "_compute_prefix_matches", return_value=set()):
            result = policy.calc_priority(queue)
        self.assertEqual([req.rid for req in result], ["long", "short"])

    def test_completed_host_load_is_credited(self):
        queue = [
            request("pending", cached=100, host=100),
            request("ready", cached=100, host=100, loaded=100),
        ]
        SchedulePolicy._sort_by_longest_prefix(queue, set(), 64)
        self.assertEqual(queue[0].rid, "ready")

    def test_duplicate_prefix_deprioritization_wins(self):
        queue = [
            request("duplicate", cached=100),
            request("cold", cached=0, uncached=100),
        ]
        SchedulePolicy._sort_by_longest_prefix(queue, {"duplicate"}, 64)
        self.assertEqual(queue[0].rid, "cold")

    def test_equal_cost_keeps_arrival_order(self):
        queue = [request("first", cached=1), request("second", cached=100)]
        SchedulePolicy._sort_by_longest_prefix(queue, set(), 64)
        self.assertEqual([req.rid for req in queue], ["first", "second"])

    def test_negative_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.policy(-1)

    def test_only_pd_prefill_with_hicache_uses_weight(self):
        for role, hicache in [("decode", True), ("null", True), ("prefill", False)]:
            with self.subTest(role=role, hicache=hicache):
                self.assertEqual(
                    self.policy(64, role=role, hicache=hicache).lpm_compute_weight, 0
                )


if __name__ == "__main__":
    unittest.main()
