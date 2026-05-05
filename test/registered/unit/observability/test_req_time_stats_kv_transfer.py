"""Unit tests for disaggregated KV transfer metric computation."""

import unittest
from unittest.mock import MagicMock

from sglang.srt.disaggregation.base.conn import KVTransferMetric
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


_TRANSFER_BYTES = 16 * 1024 * 1024


def _make_stats(
    queue_entry: float = 1.0,
    transfer_finish: float = 1.1,
    completion: float = 6.1,
    enable_metrics: bool = False,
) -> SchedulerReqTimeStats:
    stats = SchedulerReqTimeStats()
    stats.prefill_transfer_queue_entry_time = queue_entry
    stats.prefill_kv_transfer_finish_time = transfer_finish
    stats.completion_time = completion
    stats.enable_metrics = enable_metrics
    return stats


class TestKVTransferMetrics(CustomTestCase):
    def test_backend_latency_is_used_instead_of_completion_time(self):
        stats = _make_stats(queue_entry=1.0, transfer_finish=1.1, completion=6.1)
        result = stats.compute_and_observe_kv_transfer_metrics(
            KVTransferMetric(
                transfer_latency_s=0.25,
                transfer_total_bytes=_TRANSFER_BYTES,
            )
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["latency_ms"], 250.0, places=3)

    def test_timestamp_fallback_uses_transfer_finish_not_completion(self):
        stats = _make_stats(queue_entry=1.0, transfer_finish=1.1, completion=6.1)
        result = stats.compute_and_observe_kv_transfer_metrics(
            KVTransferMetric(transfer_total_bytes=_TRANSFER_BYTES)
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["latency_ms"], 100.0, places=3)
        self.assertLess(result["latency_ms"], 200.0)

    def test_timestamp_fallback_does_not_require_completion_time(self):
        stats = _make_stats(queue_entry=1.0, transfer_finish=1.2, completion=0.0)
        result = stats.compute_and_observe_kv_transfer_metrics(
            KVTransferMetric(transfer_total_bytes=_TRANSFER_BYTES)
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["latency_ms"], 200.0, places=3)

    def test_missing_transfer_finish_returns_none_for_transfer_metrics(self):
        stats = _make_stats(queue_entry=1.0, transfer_finish=0.0, completion=6.0)
        result = stats.compute_and_observe_kv_transfer_metrics(
            KVTransferMetric(transfer_total_bytes=_TRANSFER_BYTES)
        )

        self.assertIsNone(result)

    def test_missing_transfer_bytes_do_not_crash_request_path(self):
        stats = _make_stats()
        result = stats.compute_and_observe_kv_transfer_metrics(KVTransferMetric())

        self.assertIsNone(result)

    def test_zero_transfer_bytes_are_reported_without_crashing(self):
        stats = _make_stats()
        result = stats.compute_and_observe_kv_transfer_metrics(
            KVTransferMetric(transfer_latency_s=0.1, transfer_total_bytes=0)
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["total_mb"], 0.0)
        self.assertEqual(result["speed_gb_s"], 0.0)

    def test_speed_uses_selected_latency(self):
        stats = _make_stats(queue_entry=1.0, transfer_finish=1.016, completion=6.016)
        result = stats.compute_and_observe_kv_transfer_metrics(
            KVTransferMetric(transfer_total_bytes=_TRANSFER_BYTES)
        )

        expected_speed = (16.0 / 1024) / 0.016
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["speed_gb_s"], expected_speed, places=4)

    def test_metrics_observer_called_when_enabled(self):
        stats = _make_stats(enable_metrics=True)
        mock_collector = MagicMock()
        stats.metrics_collector = mock_collector

        stats.compute_and_observe_kv_transfer_metrics(
            KVTransferMetric(
                transfer_latency_s=0.5, transfer_total_bytes=_TRANSFER_BYTES
            )
        )

        mock_collector.observe_kv_transfer_metrics.assert_called_once()
        call_kwargs = mock_collector.observe_kv_transfer_metrics.call_args.kwargs
        self.assertAlmostEqual(call_kwargs["latency_ms"], 500.0, places=3)

    def test_metrics_observer_not_called_when_disabled(self):
        stats = _make_stats(enable_metrics=False)
        mock_collector = MagicMock()
        stats.metrics_collector = mock_collector

        stats.compute_and_observe_kv_transfer_metrics(
            KVTransferMetric(
                transfer_latency_s=0.5, transfer_total_bytes=_TRANSFER_BYTES
            )
        )

        mock_collector.observe_kv_transfer_metrics.assert_not_called()

    def test_kv_transfer_metrics_are_exported_in_meta_info(self):
        stats = _make_stats(queue_entry=1.0, transfer_finish=1.2, completion=6.0)
        result = stats.compute_and_observe_kv_transfer_metrics(
            KVTransferMetric(transfer_total_bytes=_TRANSFER_BYTES)
        )

        meta_info = stats.convert_to_output_meta_info()

        self.assertIsNotNone(result)
        self.assertIn("kv_transfer", meta_info)
        self.assertEqual(meta_info["kv_transfer"], result)
        self.assertAlmostEqual(
            meta_info["kv_transfer"]["latency_ms"], 200.0, places=3
        )

    def test_kv_transfer_meta_info_is_omitted_when_metrics_missing(self):
        stats = _make_stats(queue_entry=1.0, transfer_finish=0.0, completion=6.0)
        result = stats.compute_and_observe_kv_transfer_metrics(
            KVTransferMetric(transfer_total_bytes=_TRANSFER_BYTES)
        )

        meta_info = stats.convert_to_output_meta_info()

        self.assertIsNone(result)
        self.assertNotIn("kv_transfer", meta_info)


if __name__ == "__main__":
    unittest.main(verbosity=2)
