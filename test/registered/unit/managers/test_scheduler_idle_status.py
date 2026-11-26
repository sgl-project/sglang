import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.utils.scheduler_status_logger import SchedulerStatusLogger

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSchedulerIdleStatus(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
        self.reporter.scheduler = SimpleNamespace(
            enable_fpm=False,
            running_batch=SimpleNamespace(reqs=[]),
            waiting_queue=[],
        )
        self.reporter.current_scheduler_metrics_enabled = True
        self.reporter._reset_device_timer_window = Mock()
        self.reporter.stats = SimpleNamespace(num_running_reqs=SimpleNamespace(total=0))
        self.reporter.metrics_collector = Mock(last_log_time=100.0)
        with patch(
            "sglang.srt.utils.scheduler_status_logger.create_log_targets",
            return_value=[],
        ):
            logger = SchedulerStatusLogger(targets=[], dump_interval=5.0)
        self.reporter.scheduler_status_logger = logger
        self.enterContext(
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter."
                "ENABLE_METRICS_DEVICE_TIMER",
                False,
            )
        )
        self.enterContext(patch("time.perf_counter", return_value=100.0))
        self.clock = self.enterContext(patch("time.time", return_value=100.0))
        self.emit = self.enterContext(
            patch("sglang.srt.utils.scheduler_status_logger.log_json")
        )

    def test_idle_refreshes_status_using_the_existing_interval(self):
        reporter = self.reporter
        busy = SimpleNamespace(reqs=[SimpleNamespace(rid="finished")])
        reporter.scheduler_status_logger.maybe_dump(busy, [])
        self.assertEqual(self.emit.call_args.args[2]["running_rids"], ["finished"])

        self.clock.return_value = 101.0
        reporter._maybe_log_idle_metrics()
        self.assertEqual(self.emit.call_count, 1)

        self.clock.return_value = 105.0
        reporter._maybe_log_idle_metrics()
        self.assertEqual(self.emit.call_count, 2)
        self.assertEqual(
            self.emit.call_args.args[2],
            {"rank": 0, "running_rids": [], "queued_rids": []},
        )

        self.clock.return_value = 106.0
        reporter._maybe_log_idle_metrics()
        self.assertEqual(self.emit.call_count, 2)
        self.clock.return_value = 110.0
        reporter._maybe_log_idle_metrics()
        self.assertEqual(self.emit.call_count, 3)
        reporter.metrics_collector.log_stats.assert_not_called()

    def test_zero_interval_emits_once_per_idle_stretch(self):
        reporter = self.reporter
        logger = reporter.scheduler_status_logger
        logger.dump_interval = 0
        reporter._maybe_log_idle_metrics()
        reporter._maybe_log_idle_metrics()
        self.assertEqual(self.emit.call_count, 1)
        self.emit.reset_mock()
        busy = SimpleNamespace(reqs=[SimpleNamespace(rid="running")])

        logger.maybe_dump(busy, [])
        logger.maybe_dump(busy, [])
        self.assertEqual(self.emit.call_count, 2)
        for _ in range(3):
            reporter._maybe_log_idle_metrics()
        self.assertEqual(self.emit.call_count, 3)
        self.assertEqual(self.emit.call_args.args[2]["running_rids"], [])

        logger.maybe_dump(busy, [])
        reporter._maybe_log_idle_metrics()
        self.assertEqual(self.emit.call_count, 5)

        logger.maybe_dump(
            reporter.scheduler.running_batch, [SimpleNamespace(rid="queued")]
        )
        self.assertEqual(self.emit.call_args.args[2]["queued_rids"], ["queued"])
        reporter._maybe_log_idle_metrics()
        self.assertEqual(self.emit.call_count, 7)

    def test_disabled_metrics_or_logger_does_not_dump(self):
        self.reporter.current_scheduler_metrics_enabled = False
        self.reporter._maybe_log_idle_metrics()
        self.emit.assert_not_called()

        self.reporter.current_scheduler_metrics_enabled = True
        self.reporter.scheduler_status_logger = None
        self.reporter._maybe_log_idle_metrics()
        self.emit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
