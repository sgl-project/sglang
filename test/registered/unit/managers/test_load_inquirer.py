import unittest
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
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.runtime_context import get_disagg
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

    def make_inquirer(
        self,
        waiting_queue_prefix_matched=True,
        mode_owners=None,
        storage_span=None,
        waiting=None,
        chunked_req=SimpleNamespace(seqlen=50, prefix_len=10),
        cache_hit_rate=0.75,
    ):
        if waiting is None:
            waiting = [
                SimpleNamespace(
                    seqlen=100,
                    num_matched_prefix_tokens=20,
                    cache_request_handle=CacheRequestHandle("waiting", 0),
                )
            ]
        core_lists = CoreLists()
        core_lists.waiting_queue = waiting
        core_lists.chunked_req = chunked_req
        return SimpleNamespace(
            request_registry=make_request_registry(
                core_lists=core_lists, mode_owners=mode_owners or ModeOwners()
            ),
            policy=SimpleNamespace(
                waiting_queue_prefix_matched=lambda reqs: waiting_queue_prefix_matched
            ),
            metrics_sink=SimpleNamespace(recent_cache_hit_rate=cache_hit_rate),
            # The storage-tier span the tree accepted for a waiting request.
            kv=SimpleNamespace(
                tree_cache=SimpleNamespace(
                    waiting_storage_span=Mock(return_value=storage_span)
                )
            ),
        )

    @staticmethod
    def uncached(inquirer) -> int:
        return SchedulerLoadInquirer.get_num_waiting_uncached_tokens(inquirer)

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

    def test_storage_exclusion_is_clipped_to_uncached_prompt(self):
        for span, expected in (
            (None, 120),
            ((0, 10), 120),
            ((0, 40), 100),
            ((20, 80), 60),
            ((60, 120), 80),
            ((20, 120), 40),
            ((20, 20), 120),
            ((100, 120), 120),
        ):
            with self.subTest(span=span):
                inquirer = self.make_inquirer(storage_span=span)
                self.assertEqual(self.uncached(inquirer), expected)
                inquirer.kv.tree_cache.waiting_storage_span.assert_called_once_with(
                    CacheRequestHandle("waiting", 0)
                )

    def test_storage_exclusion_requires_per_request_matching(self):
        for prefix_matched, expected in ((True, 60), (False, 65)):
            with self.subTest(prefix_matched=prefix_matched):
                inquirer = self.make_inquirer(
                    waiting_queue_prefix_matched=prefix_matched,
                    storage_span=(20, 80),
                )
                self.assertEqual(self.uncached(inquirer), expected)
                if not prefix_matched:
                    inquirer.kv.tree_cache.waiting_storage_span.assert_not_called()

    def test_storage_exclusion_does_not_change_decode(self):
        enter_scope(self, get_disagg().override(disaggregation_mode="decode"))
        for prefix_matched in (False, True):
            with self.subTest(prefix_matched=prefix_matched):
                inquirer = self.make_inquirer(
                    waiting_queue_prefix_matched=prefix_matched,
                    storage_span=(20, 100),
                )
                self.assertEqual(self.uncached(inquirer), 0)
                inquirer.kv.tree_cache.waiting_storage_span.assert_not_called()

    def test_storage_exclusion_preserves_the_historical_hit_rate_estimate(self):
        for span in (None, (20, 50), (20, 80), (80, 100), (0, 120)):
            with self.subTest(span=span):
                inquirer = self.make_inquirer(
                    waiting_queue_prefix_matched=False,
                    storage_span=span,
                )
                self.assertEqual(self.uncached(inquirer), 65)
                inquirer.kv.tree_cache.waiting_storage_span.assert_not_called()

    def test_historical_estimate_already_includes_storage_hits(self):
        requests = [
            SimpleNamespace(seqlen=100, cache_request_handle=CacheRequestHandle(rid, 0))
            for rid in ("storage_hit", "cold")
        ]
        inquirer = self.make_inquirer(
            waiting_queue_prefix_matched=False,
            waiting=requests,
            chunked_req=None,
            cache_hit_rate=0.4,
        )
        inquirer.kv.tree_cache.waiting_storage_span.side_effect = lambda handle: (
            (0, 80) if handle.rid == "storage_hit" else None
        )

        # One 80-token storage hit and one cold prompt need 20 + 100 tokens.
        self.assertEqual(self.uncached(inquirer), 120)
        inquirer.kv.tree_cache.waiting_storage_span.assert_not_called()


if __name__ == "__main__":
    unittest.main()
