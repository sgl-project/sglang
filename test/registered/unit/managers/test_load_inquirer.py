import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.schedule_policy import (
    CacheAwarePolicy,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_components.load_inquirer import (
    SchedulerLoadInquirer,
)
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.disagg_owner_shells import prefill_owner_shell
from sglang.test.scheduler_request_lists import make_request_registry
from sglang.test.test_utils import enter_scope, published_topology

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestSchedulePolicyWaitingQueueMatching(unittest.TestCase):
    def make_policy(self, policy, supports_fast_match_prefix):
        schedule_policy = object.__new__(SchedulePolicy)
        schedule_policy.policy = policy
        schedule_policy.tree_cache = SimpleNamespace(
            supports_fast_match_prefix=lambda: supports_fast_match_prefix
        )
        return schedule_policy

    def test_lpm_queue_limit_respects_fast_matching_capability(self):
        policy = self.make_policy(CacheAwarePolicy.LPM, False)
        self.assertTrue(policy.waiting_queue_prefix_matched([None] * 128))
        self.assertFalse(policy.waiting_queue_prefix_matched([None] * 129))

        policy.tree_cache = SimpleNamespace(supports_fast_match_prefix=lambda: True)
        self.assertTrue(policy.waiting_queue_prefix_matched([None] * 129))


class TestSchedulerLoadInquirer(unittest.TestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def make_inquirer(self, waiting_queue_prefix_matched, mode_owners=None):
        waiting_req = SimpleNamespace(seqlen=100, num_matched_prefix_tokens=20)
        chunked_req = SimpleNamespace(seqlen=50, prefix_len=10)
        core_lists = CoreLists()
        core_lists.waiting_queue = [waiting_req]
        core_lists.chunked_req = chunked_req
        return SimpleNamespace(
            request_registry=make_request_registry(
                core_lists=core_lists, mode_owners=mode_owners or ModeOwners()
            ),
            policy=SimpleNamespace(
                waiting_queue_prefix_matched=lambda reqs: waiting_queue_prefix_matched
            ),
            metrics_sink=SimpleNamespace(recent_cache_hit_rate=0.75),
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

    def test_pd_prefill_bootstrap_requests_are_estimated(self):
        owner = prefill_owner_shell()
        owner.bootstrap_queue.queue = [SimpleNamespace(seqlen=40)]
        for prefix_matched, expected in ((False, 65 + 10), (True, 120 + 10)):
            with self.subTest(prefix_matched=prefix_matched):
                inquirer = self.make_inquirer(
                    waiting_queue_prefix_matched=prefix_matched,
                    mode_owners=ModeOwners(disagg_prefill=owner),
                )
                self.assertEqual(
                    SchedulerLoadInquirer.get_num_waiting_uncached_tokens(inquirer),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
