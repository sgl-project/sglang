"""Pure-CPU unit tests for sglang:avg_request_queue_latency Prometheus metric.

Verifies:
1. SchedulerStats defines avg_request_queue_latency defaulting to 0.0.
2. SchedulerMetricsCollector registers and logs sglang:avg_request_queue_latency.
3. SchedulerMetricsReporter._calc_avg_request_queue_latency computes the mean
   waiting queue latency without allocating intermediate lists (O(1) memory).
4. Edge cases: empty queue, zero entry_times, clock jitter/future timestamps,
   and mixed valid/invalid requests.
"""

import time
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

from sglang.srt.runtime_context import get_context
from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollector,
    SchedulerStats,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)


class _RecordingMetric:
    """Mock metric to capture label calls and gauge sets."""

    def __init__(self, *args, name=None, labelnames=(), **kwargs):
        self.name = name if name is not None else args[0]
        self.labelnames = tuple(labelnames)
        self.sets = []

    def labels(self, *values, **labels):
        return self

    def set(self, value):
        self.sets.append(value)


class _RecordingSchedulerMetricsCollector(SchedulerMetricsCollector):
    _gauge_cls = _RecordingMetric


class TestSchedulerStatsAvgQueueLatency(unittest.TestCase):
    def test_default_value(self):
        stats = SchedulerStats()
        self.assertEqual(stats.avg_request_queue_latency, 0.0)

    def test_custom_assignment(self):
        stats = SchedulerStats()
        stats.avg_request_queue_latency = 2.45
        self.assertEqual(stats.avg_request_queue_latency, 2.45)


class TestMetricsCollectorAvgQueueLatency(unittest.TestCase):
    def test_gauge_registered_and_logged(self):
        labels = {
            "model_name": "test_model",
            "engine_type": "llm",
            "tp_rank": 0,
            "pp_rank": 0,
            "moe_ep_rank": 0,
        }
        with get_context().override_server_args():
            collector = _RecordingSchedulerMetricsCollector(
                labels=labels,
                server_args=None,
            )
            self.assertTrue(hasattr(collector, "avg_request_queue_latency"))
            self.assertEqual(
                collector.avg_request_queue_latency.name,
                "sglang:avg_request_queue_latency",
            )

            stats = SchedulerStats()
            stats.avg_request_queue_latency = 1.85
            collector.log_stats(stats)

            self.assertIn(1.85, collector.avg_request_queue_latency.sets)


class TestCalcAvgRequestQueueLatency(unittest.TestCase):
    def _create_reporter(self, waiting_queue):
        scheduler = SimpleNamespace(waiting_queue=waiting_queue)
        reporter = object.__new__(SchedulerMetricsReporter)
        reporter.scheduler = scheduler
        return reporter

    def test_empty_waiting_queue(self):
        reporter = self._create_reporter([])
        self.assertEqual(reporter._calc_avg_request_queue_latency(), 0.0)

    def test_none_waiting_queue(self):
        reporter = self._create_reporter(None)
        self.assertEqual(reporter._calc_avg_request_queue_latency(), 0.0)

    def test_requests_with_zero_or_uninitialized_entry_time(self):
        req1 = SimpleNamespace(time_stats=SimpleNamespace(wait_queue_entry_time=0.0))
        req2 = SimpleNamespace(time_stats=None)
        req3 = SimpleNamespace()
        reporter = self._create_reporter([req1, req2, req3])
        self.assertEqual(reporter._calc_avg_request_queue_latency(), 0.0)

    def test_requests_with_valid_latencies(self):
        now = time.perf_counter()
        req1 = SimpleNamespace(time_stats=SimpleNamespace(wait_queue_entry_time=now - 1.0))
        req2 = SimpleNamespace(time_stats=SimpleNamespace(wait_queue_entry_time=now - 2.0))
        req3 = SimpleNamespace(time_stats=SimpleNamespace(wait_queue_entry_time=now - 3.0))

        reporter = self._create_reporter([req1, req2, req3])
        latency = reporter._calc_avg_request_queue_latency()
        # Average of 1.0, 2.0, 3.0 is 2.0 seconds
        self.assertAlmostEqual(latency, 2.0, delta=0.05)

    def test_clock_jitter_future_entry_time_clamped(self):
        now = time.perf_counter()
        # Entry time in slight future due to clock jitter
        req = SimpleNamespace(time_stats=SimpleNamespace(wait_queue_entry_time=now + 0.1))
        reporter = self._create_reporter([req])
        latency = reporter._calc_avg_request_queue_latency()
        self.assertEqual(latency, 0.0)

    def test_mixed_valid_and_invalid_requests(self):
        now = time.perf_counter()
        req_valid1 = SimpleNamespace(time_stats=SimpleNamespace(wait_queue_entry_time=now - 1.0))
        req_valid2 = SimpleNamespace(time_stats=SimpleNamespace(wait_queue_entry_time=now - 3.0))
        req_invalid = SimpleNamespace(time_stats=SimpleNamespace(wait_queue_entry_time=0.0))

        reporter = self._create_reporter([req_valid1, req_invalid, req_valid2])
        latency = reporter._calc_avg_request_queue_latency()
        # Average of 1.0 and 3.0 is 2.0 seconds
        self.assertAlmostEqual(latency, 2.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
