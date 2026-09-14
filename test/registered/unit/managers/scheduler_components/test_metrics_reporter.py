import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
    _avg_queue_wait_time,
)
from sglang.srt.observability.metrics_collector import QueueCount, SchedulerStats

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_req(wait_queue_entry_time: float):
    return SimpleNamespace(
        time_stats=SimpleNamespace(wait_queue_entry_time=wait_queue_entry_time)
    )


class TestAvgQueueWaitTime(CustomTestCase):
    """Guards issue #6357: avg_request_queue_latency must reflect the live
    wait-so-far of currently-queued requests, not stay unset/zero."""

    def test_empty_queue_is_zero(self):
        self.assertEqual(_avg_queue_wait_time([], now=100.0), 0.0)

    def test_averages_wait_so_far_per_request(self):
        reqs = [_fake_req(90.0), _fake_req(95.0), _fake_req(70.0)]
        # Waits at now=100: 10, 5, 30 -> mean 15.
        self.assertAlmostEqual(_avg_queue_wait_time(reqs, now=100.0), 15.0)

    def test_single_request_just_enqueued(self):
        reqs = [_fake_req(100.0)]
        self.assertAlmostEqual(_avg_queue_wait_time(reqs, now=100.0), 0.0)


class TestMaybeLogIdleMetrics(CustomTestCase):
    """Guards issue #6357: the gauge must decay to 0 once the queue drains,
    instead of freezing at its last published value."""

    def _make_reporter(self) -> SchedulerMetricsReporter:
        reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
        reporter.current_scheduler_metrics_enabled = True

        reporter.stats = SchedulerStats()
        reporter.stats.avg_request_queue_latency = 15.0
        # One previously-running request; the drained batch below has none,
        # so `gauge_stale` is True and the idle-publish path is not skipped.
        reporter.stats.num_running_reqs = QueueCount(total=1)

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.waiting_queue = []
        scheduler.enable_priority_scheduling = False
        scheduler.grammar_manager = []
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.pool_stats_observer.streaming_session_count.return_value = 0
        scheduler.pool_stats_observer.session_held_tokens.return_value = 0
        reporter.scheduler = scheduler

        reporter.metrics_collector = MagicMock()
        return reporter

    def test_idle_publish_zeroes_avg_queue_latency_after_drain(self):
        reporter = self._make_reporter()

        reporter._maybe_log_idle_metrics()

        self.assertEqual(reporter.stats.avg_request_queue_latency, 0.0)
        reporter.metrics_collector.log_stats.assert_called_once_with(reporter.stats)


if __name__ == "__main__":
    unittest.main()
