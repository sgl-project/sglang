"""Unit tests for the prefetch stage latency metric (per_stage_req_latency_seconds).

Guards the wiring that records the HiCache L3 -> L2 prefetch end-to-end latency as
a ``stage="prefetch"`` label of ``sglang:per_stage_req_latency_seconds``. The
prefetch start is recorded only when ``prefetch_from_storage`` actually issues an
operation (i.e. ``req_id`` lands in ``ongoing_prefetch``); the finish is observed
exactly once when ``check_prefetch_progress`` reports completion. ``prefetch_start_time``
is a tri-state field (0 = never issued / already observed, > 0 = issued, pending).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.test.test_utils import CustomTestCase


class TestPrefetchStageLatency(CustomTestCase):
    def _make_stats(self):
        from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats

        stats = SchedulerReqTimeStats()
        collector = MagicMock()
        stats.set_metrics_collector(collector)
        self.assertTrue(stats.enable_metrics)
        return stats, collector

    def test_observe_when_prefetch_issued(self):
        """Prefetch issued then completed: observe exactly once with stage='prefetch'
        and snapshot the latency into prefetch_duration for the log line."""
        stats, collector = self._make_stats()
        stats.set_prefetch_start_time()
        stats.observe_prefetch_stage_finish()

        collector.observe_per_stage_req_latency.assert_called_once()
        stage_name, latency = collector.observe_per_stage_req_latency.call_args[0]
        self.assertEqual(stage_name, "prefetch")
        self.assertGreaterEqual(latency, 0.0)
        self.assertGreater(stats.prefetch_duration, 0.0)

    def test_idempotent_on_repeated_finish(self):
        """Repeated finish calls observe only once (idempotent guard)."""
        stats, collector = self._make_stats()
        stats.set_prefetch_start_time()
        stats.observe_prefetch_stage_finish()
        stats.observe_prefetch_stage_finish()
        stats.observe_prefetch_stage_finish()

        self.assertEqual(collector.observe_per_stage_req_latency.call_count, 1)

    def test_no_observe_when_prefetch_never_issued(self):
        """Never-issued prefetch (early exit) must not observe."""
        stats, collector = self._make_stats()
        # prefetch_start_time stays 0.0 (set_prefetch_start_time never called)
        stats.observe_prefetch_stage_finish()

        collector.observe_per_stage_req_latency.assert_not_called()

    def test_reset_clears_stale_start(self):
        """reset_prefetch_start_time prevents a stale observe after re-queue."""
        stats, collector = self._make_stats()
        stats.set_prefetch_start_time()
        stats.reset_prefetch_start_time()
        stats.observe_prefetch_stage_finish()

        collector.observe_per_stage_req_latency.assert_not_called()

    def test_rerecord_after_reset(self):
        """After reset, a new start can be recorded and observed (re-queue flow)."""
        stats, collector = self._make_stats()
        stats.set_prefetch_start_time()
        stats.reset_prefetch_start_time()
        stats.set_prefetch_start_time()
        stats.observe_prefetch_stage_finish()

        collector.observe_per_stage_req_latency.assert_called_once()

    def test_reset_prefill_retry_time_clears_start(self):
        """reset_prefill_retry_time clears prefetch_start_time (disagg retry path)."""
        stats, collector = self._make_stats()
        stats.set_prefetch_start_time()
        stats.reset_prefill_retry_time()

        self.assertEqual(stats.prefetch_start_time, 0.0)
        stats.observe_prefetch_stage_finish()
        collector.observe_per_stage_req_latency.assert_not_called()

    def test_convert_to_duration_includes_prefetch(self):
        """convert_to_duration surfaces the snapshotted prefetch latency.

        Guards the wiring that propagates the once-captured prefetch latency
        into the time-stats log line (start is cleared after observe, so the
        snapshot field is the only source).
        """
        stats, collector = self._make_stats()
        # Fixed timestamps so the duration is a recognizable 1000.00ms.
        stats.set_prefetch_start_time(ts=1.0)
        stats.observe_prefetch_stage_finish(ts=2.0)

        line = stats.convert_to_duration()
        self.assertIn("prefetch_duration=1000.00ms", line)

    def test_convert_to_duration_omits_prefetch_when_not_issued(self):
        """No prefetch issued: the log line must not carry a prefetch field.

        Negative-branch contract -- keeps the log line clean for deployments
        without HiCache instead of emitting a noisy prefetch_duration=0.00ms.
        """
        stats, collector = self._make_stats()

        line = stats.convert_to_duration()
        self.assertNotIn("prefetch_duration", line)


if __name__ == "__main__":
    unittest.main()
