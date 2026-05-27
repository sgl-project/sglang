"""Manual unit test for _maybe_log_idle_metrics idle-flush behavior.

Verifies that the active→idle transition flush bypasses the 30 s idle
throttle on the first idle iteration, but subsequent idle iterations
still throttle as before. Pure mock-based (no GPU, no scheduler loop).

Usage:
    python3 -m unittest test/manual/test_metrics_reporter_idle_flush.py
    python3 test/manual/test_metrics_reporter_idle_flush.py
"""

import time
import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)


def _make_reporter():
    """Stub with just enough state for _maybe_log_idle_metrics to run."""
    r = MagicMock(spec=SchedulerMetricsReporter)
    r._idle_flush_pending = False
    r.current_scheduler_metrics_enabled = True
    r.metrics_collector = MagicMock()
    r.metrics_collector.last_log_time = time.perf_counter()
    r.scheduler = MagicMock()
    r.scheduler.disaggregation_mode = None  # neither PREFILL nor DECODE branch
    r.scheduler.enable_priority_scheduling = False
    r.scheduler.running_batch = MagicMock(reqs=[])
    r.scheduler.waiting_queue = []
    r.scheduler.grammar_manager = []
    pool_stats = MagicMock()
    pool_stats.update_scheduler_stats = MagicMock()
    r.scheduler.pool_stats_observer = MagicMock()
    r.scheduler.pool_stats_observer.get_pool_stats.return_value = pool_stats
    r.scheduler.pool_stats_observer.streaming_session_count.return_value = 0
    r.scheduler.pool_stats_observer.session_held_tokens.return_value = 0
    r.stats = MagicMock()
    return r


class TestIdleMetricFlush(unittest.TestCase):
    def test_throttle_blocks_when_no_flush_pending(self):
        r = _make_reporter()
        r._idle_flush_pending = False
        SchedulerMetricsReporter._maybe_log_idle_metrics(r)
        r.metrics_collector.log_stats.assert_not_called()

    def test_pending_flag_bypasses_throttle(self):
        r = _make_reporter()
        r._idle_flush_pending = True
        SchedulerMetricsReporter._maybe_log_idle_metrics(r)
        r.metrics_collector.log_stats.assert_called_once()
        self.assertFalse(
            r._idle_flush_pending,
            "flag should be cleared after the transition flush",
        )

    def test_throttle_expires_after_30s(self):
        r = _make_reporter()
        r._idle_flush_pending = False
        r.metrics_collector.last_log_time = time.perf_counter() - 31  # 31 s ago
        SchedulerMetricsReporter._maybe_log_idle_metrics(r)
        r.metrics_collector.log_stats.assert_called_once()

    def test_disabled_metrics_short_circuits(self):
        r = _make_reporter()
        r._idle_flush_pending = True
        r.current_scheduler_metrics_enabled = False
        SchedulerMetricsReporter._maybe_log_idle_metrics(r)
        r.metrics_collector.log_stats.assert_not_called()

    def test_second_idle_iter_after_flush_throttled(self):
        r = _make_reporter()
        r._idle_flush_pending = True
        SchedulerMetricsReporter._maybe_log_idle_metrics(r)  # first flush
        r.metrics_collector.log_stats.reset_mock()
        SchedulerMetricsReporter._maybe_log_idle_metrics(r)  # immediately again
        r.metrics_collector.log_stats.assert_not_called()


if __name__ == "__main__":
    unittest.main()
