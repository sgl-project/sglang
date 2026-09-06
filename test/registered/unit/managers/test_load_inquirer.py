import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_components.load_inquirer import (
    SchedulerLoadInquirer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSchedulePolicyWaitingQueueMatching(unittest.TestCase):
    def make_policy(self, policy, supports_fast_match_prefix):
        schedule_policy = object.__new__(SchedulePolicy)
        schedule_policy.policy = policy
        schedule_policy.tree_cache = SimpleNamespace(
            supports_fast_match_prefix=lambda: supports_fast_match_prefix
        )
        return schedule_policy

    def test_cache_agnostic_policy_requires_fast_matching(self):
        policy = self.make_policy(CacheAgnosticPolicy.FCFS, False)
        self.assertFalse(policy.waiting_queue_prefix_matched([]))

        policy.tree_cache = SimpleNamespace(supports_fast_match_prefix=lambda: True)
        self.assertTrue(policy.waiting_queue_prefix_matched([]))

    def test_lpm_queue_limit_respects_fast_matching_capability(self):
        policy = self.make_policy(CacheAwarePolicy.LPM, False)
        self.assertTrue(policy.waiting_queue_prefix_matched([None] * 128))
        self.assertFalse(policy.waiting_queue_prefix_matched([None] * 129))

        policy.tree_cache = SimpleNamespace(supports_fast_match_prefix=lambda: True)
        self.assertTrue(policy.waiting_queue_prefix_matched([None] * 129))


class TestSchedulerLoadInquirer(unittest.TestCase):
    def make_inquirer(self, waiting_queue_prefix_matched):
        waiting_req = SimpleNamespace(seqlen=100, num_matched_prefix_tokens=20)
        chunked_req = SimpleNamespace(seqlen=50, prefix_indices=range(10))
        return SimpleNamespace(
            disaggregation_mode=DisaggregationMode.NULL,
            get_waiting_queue=lambda: [waiting_req],
            waiting_queue_prefix_matched=lambda: waiting_queue_prefix_matched,
            get_chunked_req=lambda: chunked_req,
            get_recent_cache_hit_rate=lambda: 0.75,
        )

    def test_waiting_tokens_are_estimated_when_prefix_matching_is_skipped(self):
        inquirer = self.make_inquirer(waiting_queue_prefix_matched=False)

        self.assertEqual(
            SchedulerLoadInquirer.get_num_waiting_uncached_tokens(inquirer),
            65,
        )

    def test_waiting_tokens_use_exact_match_when_prefix_matching_is_done(self):
        inquirer = self.make_inquirer(waiting_queue_prefix_matched=True)

        self.assertEqual(
            SchedulerLoadInquirer.get_num_waiting_uncached_tokens(inquirer),
            120,
        )


if __name__ == "__main__":
    unittest.main()
