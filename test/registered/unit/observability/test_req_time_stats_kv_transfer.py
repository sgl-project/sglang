"""Unit tests for KV transfer metric computation in SchedulerReqTimeStats.

Verifies that compute_and_observe_kv_transfer_metrics uses the actual KV
transfer window (prefill_transfer_queue_entry_time → prefill_kv_transfer_finish_time)
rather than the full request lifetime (…→ completion_time), which inflates
the reported latency and deflates the reported speed.
"""

import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

from sglang.srt.disaggregation.base.conn import KVTransferMetric  # noqa: E402
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats  # noqa: E402

# 16 MiB of transfer data
_TOTAL_BYTES = 16 * 1024 * 1024


def _make_stats(
    queue_entry: float,
    transfer_finish: float,
    completion: float,
    enable_metrics: bool = False,
) -> SchedulerReqTimeStats:
    """Create a SchedulerReqTimeStats with the three key timestamps pre-set."""
    stats = SchedulerReqTimeStats()
    stats.prefill_transfer_queue_entry_time = queue_entry
    stats.prefill_kv_transfer_finish_time = transfer_finish
    stats.completion_time = completion
    stats.enable_metrics = enable_metrics
    return stats


def _make_metric(latency_s=None, total_bytes=_TOTAL_BYTES) -> KVTransferMetric:
    """Build a KVTransferMetric for tests. latency_s=None forces timestamp path."""
    m = KVTransferMetric()
    m.transfer_latency_s = latency_s
    m.transfer_total_bytes = total_bytes
    return m


class TestKVTransferLatencyWindow(CustomTestCase):
    def _call(self, stats: SchedulerReqTimeStats, metric=None) -> dict:
        if metric is None:
            metric = _make_metric()
        return stats.compute_and_observe_kv_transfer_metrics(transfer_metric=metric)

    def test_latency_uses_transfer_finish_not_completion(self):
        # Transfer takes 0.1 s; decode adds another 5 s → completion is 5.1 s after queue entry.
        # The metric must reflect only the 0.1 s transfer window.
        stats = _make_stats(queue_entry=1.0, transfer_finish=1.1, completion=6.1)
        result = self._call(stats)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["latency_ms"], 100.0, places=3)

    def test_latency_not_inflated_by_decode_duration(self):
        # Before the fix, latency_ms would be (completion - queue_entry)*1000 = 5100 ms.
        stats = _make_stats(queue_entry=1.0, transfer_finish=1.1, completion=6.1)
        result = self._call(stats)
        self.assertLess(result["latency_ms"], 200.0)  # must not include decode time

    def test_speed_computed_from_correct_latency(self):
        # 16 MiB transferred in 0.016 s → speed = (16/1024) / 0.016 = 0.977 GB/s
        stats = _make_stats(queue_entry=1.0, transfer_finish=1.016, completion=6.016)
        result = self._call(stats)
        expected_speed = (16.0 / 1024) / 0.016
        self.assertAlmostEqual(result["speed_gb_s"], expected_speed, places=4)

    def test_returns_none_when_transfer_finish_not_set(self):
        # If prefill_kv_transfer_finish_time was never recorded the result should
        # be None (guard condition uses transfer_finish, not completion).
        stats = _make_stats(queue_entry=1.0, transfer_finish=0.0, completion=6.0)
        result = self._call(stats)
        self.assertIsNone(result)

    def test_returns_none_when_queue_entry_not_set(self):
        stats = _make_stats(queue_entry=0.0, transfer_finish=1.1, completion=6.0)
        result = self._call(stats)
        self.assertIsNone(result)

    def test_result_populated_even_when_completion_time_is_zero(self):
        # After the fix, completion_time is irrelevant to the KV transfer metric.
        # The metric must be computable even if completion hasn't been set yet.
        stats = _make_stats(queue_entry=1.0, transfer_finish=1.2, completion=0.0)
        result = self._call(stats)
        self.assertIn("latency_ms", result)
        self.assertAlmostEqual(result["latency_ms"], 200.0, places=3)

    def test_transfer_latency_s_takes_priority_over_timestamps(self):
        # When KVTransferMetric supplies transfer_latency_s directly, timestamps are ignored.
        stats = _make_stats(queue_entry=1.0, transfer_finish=1.1, completion=6.1)
        metric = _make_metric(latency_s=0.5)  # 500 ms, not the 100 ms from timestamps
        result = self._call(stats, metric=metric)
        self.assertAlmostEqual(result["latency_ms"], 500.0, places=3)

    def test_metrics_observer_called_when_enabled(self):
        stats = _make_stats(
            queue_entry=1.0, transfer_finish=1.5, completion=6.0, enable_metrics=True
        )
        mock_collector = MagicMock()
        stats.metrics_collector = mock_collector
        self._call(stats)
        mock_collector.observe_kv_transfer_metrics.assert_called_once()
        call_kwargs = mock_collector.observe_kv_transfer_metrics.call_args.kwargs
        self.assertAlmostEqual(call_kwargs["latency_ms"], 500.0, places=3)

    def test_metrics_observer_not_called_when_disabled(self):
        stats = _make_stats(
            queue_entry=1.0, transfer_finish=1.5, completion=6.0, enable_metrics=False
        )
        mock_collector = MagicMock()
        stats.metrics_collector = mock_collector
        self._call(stats)
        mock_collector.observe_kv_transfer_metrics.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
