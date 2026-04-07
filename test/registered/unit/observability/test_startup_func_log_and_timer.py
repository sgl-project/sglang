"""Unit tests for startup_func_log_and_timer.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import unittest
from unittest.mock import MagicMock, patch

import sglang.srt.observability.startup_func_log_and_timer as mod
from sglang.srt.observability.startup_func_log_and_timer import (
    enable_startup_timer,
    get_max_duration,
    reset_startup_timers,
    set_startup_metric,
    startup_timer,
    time_startup_latency,
)


class TestStartupFuncLogAndTimer(unittest.TestCase):
    def setUp(self):
        self.orig_enable = mod.enable_startup_metrics
        self.orig_gauge = mod.STARTUP_LATENCY_SECONDS
        mod._max_durations.clear()

    def tearDown(self):
        mod.enable_startup_metrics = self.orig_enable
        mod.STARTUP_LATENCY_SECONDS = self.orig_gauge
        mod._max_durations.clear()

    @patch("prometheus_client.Gauge")
    def test_enable_startup_timer(self, MockGauge):
        enable_startup_timer()
        self.assertTrue(mod.enable_startup_metrics)
        self.assertIs(mod.STARTUP_LATENCY_SECONDS, MockGauge.return_value)
        MockGauge.assert_called_once()

    def test_reset_and_get_max_duration(self):
        mod._max_durations["ctx"] = 5.0
        self.assertAlmostEqual(get_max_duration("ctx"), 5.0)
        self.assertIsNone(get_max_duration("nonexistent"))
        reset_startup_timers()
        self.assertIsNone(get_max_duration("ctx"))

    def test_set_startup_metric_disabled(self):
        """When metrics disabled, returns early without tracking max."""
        mod.enable_startup_metrics = False
        set_startup_metric("ctx", 1.0)
        self.assertIsNone(get_max_duration("ctx"))

    def test_set_startup_metric_enabled(self):
        """Tracks max and updates gauge when enabled."""
        mock_gauge = MagicMock()
        mod.enable_startup_metrics = True
        mod.STARTUP_LATENCY_SECONDS = mock_gauge

        set_startup_metric("ctx", 1.0)
        self.assertAlmostEqual(get_max_duration("ctx"), 1.0)
        mock_gauge.labels.assert_called_with(context="ctx")

        # Lower value → not updated
        mock_gauge.reset_mock()
        set_startup_metric("ctx", 0.5)
        self.assertAlmostEqual(get_max_duration("ctx"), 1.0)
        mock_gauge.labels().set.assert_not_called()

    def test_set_startup_metric_no_log(self):
        mod.enable_startup_metrics = False
        with patch.object(mod.logger, "info") as mock_log:
            set_startup_metric("ctx", 1.0, should_log=False)
            mock_log.assert_not_called()

    def test_startup_timer_basic(self):
        with startup_timer("block"):
            pass
        self.assertGreaterEqual(get_max_duration("block"), 0.0)

    def test_startup_timer_with_gauge(self):
        """Gauge updated when metrics enabled and log_only=False."""
        mock_gauge = MagicMock()
        mod.enable_startup_metrics = True
        mod.STARTUP_LATENCY_SECONDS = mock_gauge

        with startup_timer("block"):
            pass
        mock_gauge.labels.assert_called_with(context="block")
        mock_gauge.labels().set.assert_called_once()

    def test_startup_timer_log_only(self):
        """log_only=True skips gauge but still tracks max."""
        mock_gauge = MagicMock()
        mod.enable_startup_metrics = True
        mod.STARTUP_LATENCY_SECONDS = mock_gauge

        with startup_timer("block", log_only=True):
            pass
        mock_gauge.labels.assert_not_called()
        self.assertIsNotNone(get_max_duration("block"))

    def test_decorator_direct(self):
        """Direct decorator @time_startup_latency preserves return value."""

        @time_startup_latency
        def add(a, b):
            return a + b

        self.assertEqual(add(2, 3), 5)
        self.assertIsNotNone(get_max_duration("add"))

    def test_decorator_factory_with_gauge(self):
        """Factory decorator with custom name, gauge updated."""
        mock_gauge = MagicMock()
        mod.enable_startup_metrics = True
        mod.STARTUP_LATENCY_SECONDS = mock_gauge

        @time_startup_latency(name="custom_op")
        def add(a, b):
            return a + b

        self.assertEqual(add(2, 3), 5)
        mock_gauge.labels.assert_called_with(context="custom_op")

    def test_decorator_log_only(self):
        """log_only=True skips gauge but still tracks max."""
        mock_gauge = MagicMock()
        mod.enable_startup_metrics = True
        mod.STARTUP_LATENCY_SECONDS = mock_gauge

        @time_startup_latency(log_only=True)
        def add(a, b):
            return a + b

        self.assertEqual(add(2, 3), 5)
        mock_gauge.labels.assert_not_called()
        self.assertIsNotNone(get_max_duration("add"))


if __name__ == "__main__":
    unittest.main()
