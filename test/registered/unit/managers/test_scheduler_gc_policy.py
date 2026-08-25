"""Unit tests for SchedulerGcPolicy (SGLANG_OPT_SCHEDULER_GC_COLLECT_INTERVAL_S)."""

import unittest
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.gc_policy import SchedulerGcPolicy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GC_POLICY_MOD = "sglang.srt.managers.scheduler_components.gc_policy"


class TestSchedulerGcPolicy(CustomTestCase):
    def test_enabled_by_default(self):
        policy = SchedulerGcPolicy()
        self.assertTrue(policy.enabled)

    def test_disabled_at_zero(self):
        with envs.SGLANG_OPT_SCHEDULER_GC_COLLECT_INTERVAL_S.override(0.0):
            policy = SchedulerGcPolicy()
        self.assertFalse(policy.enabled)
        with patch(f"{GC_POLICY_MOD}.gc") as mock_gc:
            policy.maybe_run()
        mock_gc.collect.assert_not_called()
        mock_gc.disable.assert_not_called()

    def test_activation_and_interval(self):
        with envs.SGLANG_OPT_SCHEDULER_GC_COLLECT_INTERVAL_S.override(30.0):
            policy = SchedulerGcPolicy()
        self.assertTrue(policy.enabled)

        clock = [1000.0]
        with (
            patch(f"{GC_POLICY_MOD}.gc") as mock_gc,
            patch(f"{GC_POLICY_MOD}.time") as mock_time,
        ):
            mock_time.monotonic.side_effect = lambda: clock[0]
            mock_time.perf_counter.side_effect = lambda: clock[0]
            mock_gc.collect.return_value = 0
            mock_gc.get_freeze_count.return_value = 123

            policy.maybe_run()
            self.assertEqual(mock_gc.collect.call_count, 1)
            mock_gc.freeze.assert_called_once()
            mock_gc.disable.assert_called_once()

            clock[0] += 29.0
            policy.maybe_run()
            self.assertEqual(mock_gc.collect.call_count, 1)

            clock[0] += 2.0
            policy.maybe_run()
            self.assertEqual(mock_gc.collect.call_count, 2)
            mock_gc.freeze.assert_called_once()

            clock[0] += 30.5
            policy.maybe_run()
            self.assertEqual(mock_gc.collect.call_count, 3)


if __name__ == "__main__":
    unittest.main()
