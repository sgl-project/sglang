"""Unit tests for func_timer.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-c-test-cpu")

import asyncio
import time
import unittest
from unittest.mock import MagicMock, patch

import sglang.srt.observability.func_timer as func_timer
from sglang.srt.observability.func_timer import enable_func_timer, time_func_latency


class TestEnableFuncTimer(unittest.TestCase):
    """Tests for enable_func_timer helper."""

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


class TestTimeFuncLatencyDecoratorDisabled(unittest.TestCase):
    """Tests for time_func_latency when metrics are DISABLED (default)."""

    def test_sync_passthrough_when_metrics_disabled(self):
        """Sync function should execute normally when metrics are disabled."""

        @time_func_latency
        def add(a, b):
            return a + b

        self.assertEqual(add(1, 2), 3)

    def test_sync_preserves_function_name(self):
        """Decorator should preserve the original function's __name__."""

        @time_func_latency
        def my_function():
            pass

        self.assertEqual(my_function.__name__, "my_function")

    def test_async_passthrough_when_metrics_disabled(self):
        """Async function should execute normally when metrics are disabled."""

        @time_func_latency
        async def async_add(a, b):
            return a + b

        result = asyncio.run(async_add(3, 4))
        self.assertEqual(result, 7)

    def test_async_preserves_function_name(self):
        """Decorator should preserve the async function's __name__."""

        @time_func_latency
        async def my_async_func():
            pass

        self.assertEqual(my_async_func.__name__, "my_async_func")

    def test_decorator_with_custom_name(self):
        """Decorator should accept an optional name parameter."""

        @time_func_latency(name="custom_op")
        def operation():
            return 42

        self.assertEqual(operation(), 42)
        self.assertEqual(operation.__name__, "operation")

    def test_sync_exception_propagates(self):
        """Exceptions from decorated sync functions should propagate."""

        @time_func_latency
        def fail():
            raise ValueError("test error")

        with self.assertRaises(ValueError) as ctx:
            fail()
        self.assertEqual(str(ctx.exception), "test error")

    def test_async_exception_propagates(self):
        """Exceptions from decorated async functions should propagate."""

        @time_func_latency
        async def async_fail():
            raise RuntimeError("async error")

        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(async_fail())
        self.assertEqual(str(ctx.exception), "async error")

    def test_kwargs_forwarding(self):
        """Decorator should forward keyword arguments correctly."""

        @time_func_latency
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        self.assertEqual(greet("World", greeting="Hi"), "Hi, World!")


class TestTimeFuncLatencyDecoratorEnabled(unittest.TestCase):
    """Tests for time_func_latency when metrics are ENABLED."""

    def setUp(self):
        """Set up mock Histogram for testing metric observation."""
        self.mock_histogram = MagicMock()
        self.mock_labels = MagicMock()
        self.mock_histogram.labels.return_value = self.mock_labels

        patcher_enable = patch(
            "sglang.srt.observability.func_timer.enable_metrics", True
        )
        patcher_latency = patch(
            "sglang.srt.observability.func_timer.FUNC_LATENCY",
            self.mock_histogram,
        )
        patcher_enable.start()
        patcher_latency.start()
        self.addCleanup(patcher_enable.stop)
        self.addCleanup(patcher_latency.stop)

    def test_sync_records_latency(self):
        """Sync function should record latency when metrics are enabled."""

        @time_func_latency
        def slow_op():
            time.sleep(0.05)
            return "done"

        result = slow_op()
        self.assertEqual(result, "done")
        self.mock_histogram.labels.assert_called_with(name="slow_op")
        self.mock_labels.observe.assert_called_once()
        observed_latency = self.mock_labels.observe.call_args[0][0]
        self.assertGreaterEqual(observed_latency, 0.04)  # at least 40ms

    def test_async_records_latency(self):
        """Async function should record latency when metrics are enabled."""

        @time_func_latency
        async def async_slow():
            await asyncio.sleep(0.05)
            return "async_done"

        result = asyncio.run(async_slow())
        self.assertEqual(result, "async_done")
        self.mock_histogram.labels.assert_called_with(name="async_slow")
        self.mock_labels.observe.assert_called_once()
        observed_latency = self.mock_labels.observe.call_args[0][0]
        self.assertGreaterEqual(observed_latency, 0.04)

    def test_custom_name_used_in_metric(self):
        """Custom name should be used as the Histogram label."""

        @time_func_latency(name="my_custom_metric")
        def operation():
            return 1

        operation()
        self.mock_histogram.labels.assert_called_with(name="my_custom_metric")

    def test_sync_exception_still_records_latency(self):
        """Latency should be recorded even when the function raises."""

        @time_func_latency
        def fails():
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            fails()

        # Metric should still have been observed due to try/finally
        self.mock_labels.observe.assert_called_once()

    def test_async_exception_still_records_latency(self):
        """Async latency should be recorded even when the function raises."""

        @time_func_latency
        async def async_fails():
            raise RuntimeError("async boom")

        with self.assertRaises(RuntimeError):
            asyncio.run(async_fails())

        self.mock_labels.observe.assert_called_once()


if __name__ == "__main__":
    unittest.main()
