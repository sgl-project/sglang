import unittest
from unittest.mock import MagicMock

from sglang.srt.observability.req_time_stats import (
    RequestStage,
    SchedulerReqTimeStats,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-b-test-cpu")


class TestSchedulerReqTimeStats(unittest.TestCase):
    def test_prefill_queue_breakdown_from_first_attempt(self):
        collector = MagicMock()
        stats = SchedulerReqTimeStats()
        stats.set_metrics_collector(collector)

        stats.set_wait_queue_entry_time(ts=10.0)
        stats.set_first_prefill_attempt_time(ts=13.0)
        stats.set_forward_entry_time(ts=21.0)

        self.assertEqual(stats.get_queueing_time(), 11.0)
        self.assertEqual(stats.get_idle_in_queue_time(), 3.0)
        self.assertEqual(stats.get_prefill_budget_wait_time(), 8.0)

        collector.observe_queue_time.assert_called_once_with(11.0)
        collector.observe_per_stage_req_latency.assert_any_call(
            RequestStage.PREFILL_QUEUE_IDLE.stage_name, 3.0
        )
        collector.observe_per_stage_req_latency.assert_any_call(
            RequestStage.PREFILL_BUDGET_WAIT.stage_name, 8.0
        )

        meta = stats.convert_to_output_meta_info()
        self.assertEqual(meta["queue_time"], 11.0)
        self.assertEqual(meta["idle_in_queue_time"], 3.0)
        self.assertEqual(meta["prefill_budget_wait_time"], 8.0)

        duration = stats.convert_to_duration()
        self.assertIn("idle_in_queue_duration=3000.00ms", duration)
        self.assertIn("prefill_budget_wait_duration=8000.00ms", duration)

    def test_first_prefill_attempt_is_set_once_and_serialized(self):
        stats = SchedulerReqTimeStats()

        stats.set_first_prefill_attempt_time(ts=13.0)
        stats.set_first_prefill_attempt_time(ts=14.0)

        self.assertEqual(stats.first_prefill_attempt_time, 13.0)
        self.assertEqual(stats.__getstate__()["first_prefill_attempt_time"], 13.0)

    def test_retry_reset_clears_first_attempt(self):
        stats = SchedulerReqTimeStats()

        stats.set_wait_queue_entry_time(ts=10.0)
        stats.set_first_prefill_attempt_time(ts=13.0)
        stats.reset_prefill_retry_time()

        self.assertEqual(stats.wait_queue_entry_time, 0.0)
        self.assertEqual(stats.first_prefill_attempt_time, 0.0)


if __name__ == "__main__":
    unittest.main()
