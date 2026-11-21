"""Live schedule_policy updates through /set_internal_state."""

import unittest

from sglang.srt import runtime_context as rc
from sglang.srt.managers.io_struct import SetInternalStateReq
from sglang.srt.managers.schedule_policy import CacheAgnosticPolicy, CacheAwarePolicy
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.internal_state import (
    SchedulerInternalStateController,
)
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLiveSchedulePolicy(CustomTestCase):
    def setUp(self):
        rc.reset_context()
        rc.publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(rc.reset_context)

    def _scheduler(self, *, cache_disabled=False, priority_scheduling=False):
        s = Scheduler.__new__(Scheduler)
        s.schedule_policy = "fcfs"
        s.tree_cache = RadixCache.create_simulated()
        s.tree_cache.disable = cache_disabled
        s.enable_hierarchical_cache = False
        s.enable_priority_scheduling = priority_scheduling
        s.schedule_low_priority_values_first = False
        s.dllm_config = None
        s.max_running_requests = 8
        s.waiting_queue = IndexedReqList(ListKind.WAITING)
        s.policy = s.create_schedule_policy(s.schedule_policy)
        self.controller = SchedulerInternalStateController(
            scheduler=s, record_step_time=False
        )
        return s

    def _set(self, **server_args) -> bool:
        req = SetInternalStateReq(server_args=server_args)
        return self.controller.set_internal_state(req).updated

    def _readback(self):
        return rc.get_context().resolved_server_args_dict()["schedule_policy"]

    def test_switch_rebuilds_the_policy_and_reports_the_value(self):
        s = self._scheduler()
        s.waiting_queue.set_on_leave(s.policy.retire_waiting_members)
        self.assertTrue(self._set(schedule_policy="lpm"))
        self.assertEqual(s.policy.policy, CacheAwarePolicy.LPM)
        self.assertEqual(s.schedule_policy, "lpm")
        self.assertEqual(rc.get_schedule().schedule_policy, "lpm")
        self.assertEqual(self._readback(), "lpm")
        # The old policy's departure hook must not outlive it.
        self.assertIsNone(s.waiting_queue._on_leave)

    def test_cache_aware_policy_without_prefix_cache_runs_as_fcfs(self):
        s = self._scheduler(cache_disabled=True)
        self.assertTrue(self._set(schedule_policy="lpm"))
        self.assertEqual(s.policy.policy, CacheAgnosticPolicy.FCFS)
        self.assertEqual(self._readback(), "lpm")

    def test_same_value_keeps_the_policy_object(self):
        s = self._scheduler()
        before = s.policy
        self.assertTrue(self._set(schedule_policy="fcfs"))
        self.assertIs(s.policy, before)

    def _assert_rejected(self, s, **server_args):
        before = s.policy
        self.assertFalse(self._set(**server_args))
        self.assertIs(s.policy, before)
        self.assertEqual(s.schedule_policy, "fcfs")
        self.assertEqual(self._readback(), "fcfs")

    def test_rejects_values_that_startup_rejects(self):
        s = self._scheduler()
        # "priority" is a CLI choice but not a SchedulePolicy policy.
        for value in ("not-a-policy", "priority", 1, None):
            with self.subTest(value=value):
                self._assert_rejected(s, schedule_policy=value)

    def test_rejects_shortest_prefill_first_without_prefix_cache(self):
        s = self._scheduler(cache_disabled=True)
        self._assert_rejected(s, schedule_policy="shortest-prefill-first")

    def test_rejects_non_fcfs_lof_with_priority_scheduling(self):
        s = self._scheduler(priority_scheduling=True)
        self._assert_rejected(s, schedule_policy="lpm")
        self.assertTrue(self._set(schedule_policy="lof"))
        self.assertEqual(s.policy.policy, CacheAgnosticPolicy.LOF)

    def test_rejects_policy_that_conflicts_with_other_startup_args(self):
        s = self._scheduler()
        rc.get_context().override("test", shortest_prefill_aging_factor=0.5)
        self._assert_rejected(s, schedule_policy="lpm")
        self.assertTrue(self._set(schedule_policy="shortest-prefill-first"))

    def test_rejected_key_in_the_same_request_leaves_the_policy_unchanged(self):
        s = self._scheduler()
        self._assert_rejected(s, schedule_policy="lpm", not_a_live_knob=1)

    def test_string_for_a_numeric_knob_is_rejected(self):
        s = self._scheduler()
        s.spec_algorithm = SpeculativeAlgorithm.NONE
        # The accept-threshold validator returns early without DFlash, so a
        # string must be stopped before it, not left for it to catch.
        for key in ("pp_max_micro_batch_size", "speculative_accept_threshold_single"):
            with self.subTest(key=key):
                self._assert_rejected(s, **{key: "0.5"})
        self.assertEqual(rc.get_context().overrides_log(), [])


if __name__ == "__main__":
    unittest.main()
