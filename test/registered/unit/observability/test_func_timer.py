"""Unit tests for func_timer.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

import asyncio
import unittest
from unittest.mock import MagicMock, patch

import sglang.srt.observability.func_timer as func_timer
from sglang.srt.observability.func_timer import enable_func_timer, time_func_latency


class TestFuncTimer(unittest.TestCase):
    def setUp(self):
        self.orig_enable = func_timer.enable_metrics
        self.orig_latency = func_timer.FUNC_LATENCY

    def tearDown(self):
        func_timer.enable_metrics = self.orig_enable
        func_timer.FUNC_LATENCY = self.orig_latency

    @patch("prometheus_client.Histogram")
    def test_enable_func_timer(self, MockHistogram):
        """Sets enable_metrics and creates FUNC_LATENCY histogram."""
        enable_func_timer()
        self.assertTrue(func_timer.enable_metrics)
        self.assertIs(func_timer.FUNC_LATENCY, MockHistogram.return_value)
        MockHistogram.assert_called_once()

    def test_sync_disabled(self):
        """Sync function passes through when metrics disabled."""
        func_timer.enable_metrics = False

        @time_func_latency
        def add(a, b):
            return a + b

        self.assertEqual(add(2, 3), 5)

    def test_sync_enabled(self):
        """Sync function timed with custom name when metrics enabled."""
        mock_histogram = MagicMock()
        func_timer.enable_metrics = True
        func_timer.FUNC_LATENCY = mock_histogram

        @time_func_latency(name="custom_op")
        def add(a, b):
            return a + b

        self.assertEqual(add(2, 3), 5)
        mock_histogram.labels.assert_called_with(name="custom_op")
        mock_histogram.labels().observe.assert_called_once()

    def test_async_disabled(self):
        """Async function passes through when metrics disabled."""
        func_timer.enable_metrics = False

        @time_func_latency
        async def add(a, b):
            return a + b

        self.assertEqual(asyncio.run(add(2, 3)), 5)

    def test_async_enabled(self):
        """Async function timed with default name when metrics enabled."""
        mock_histogram = MagicMock()
        func_timer.enable_metrics = True
        func_timer.FUNC_LATENCY = mock_histogram

        @time_func_latency
        async def add(a, b):
            return a + b

        self.assertEqual(asyncio.run(add(2, 3)), 5)
        mock_histogram.labels.assert_called_with(name="add")
        mock_histogram.labels().observe.assert_called_once()


if __name__ == "__main__":
    unittest.main()
