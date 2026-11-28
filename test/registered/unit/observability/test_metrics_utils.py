import asyncio
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.observability import cpu_monitor, func_timer
from sglang.srt.observability.utils import (
    generate_buckets,
    two_sides_exponential_buckets,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="extra-test-cpu-intel")


class TestMetricsUtils(unittest.TestCase):
    """Test cases for metrics utility functions."""

    def test_two_sides_exponential_buckets_specific_values(self):
        """Test specific values for two_sides_exponential_buckets."""
        buckets = two_sides_exponential_buckets(middle=100.0, base=2.0, count=4)
        expected_values = [96.0, 98.0, 100.0, 102.0, 104.0]
        self.assertEqual(buckets, expected_values)

    def test_two_sides_exponential_buckets_negative_values(self):
        """Test two_sides_exponential_buckets with values that could go negative."""
        buckets = two_sides_exponential_buckets(middle=5.0, base=3.0, count=4)

        # Should not contain negative values (max(0, middle - distance))
        for bucket in buckets:
            self.assertGreaterEqual(bucket, 0.0)

        # Should contain the middle value
        self.assertIn(5.0, buckets)

    def test_generate_buckets_default(self):
        """Test generate_buckets with default rule."""
        default_buckets = [1.0, 5.0, 10.0, 50.0, 100.0]

        # Test with "default" rule
        result = generate_buckets(["default"], default_buckets)
        self.assertEqual(result, default_buckets)

        # Test with None (should default to "default")
        result = generate_buckets(None, default_buckets)
        self.assertEqual(result, default_buckets)

    def test_generate_buckets_preserves_order_and_type(self):
        """Test that generate_buckets preserves order and returns floats."""
        default_buckets = [1, 5, 10, 50, 100]  # integers

        # Test default rule
        result = generate_buckets(["default"], default_buckets)
        self.assertEqual(result, default_buckets)
        self.assertIsInstance(result, list)

        # Test custom rule with proper float conversion
        result = generate_buckets(
            ["custom", "100", "50", "10", "5", "1"], default_buckets
        )
        expected = [1.0, 5.0, 10.0, 50.0, 100.0]
        self.assertEqual(result, expected)

        # All values should be floats
        for value in result:
            self.assertIsInstance(value, float)


class TestFuncTimerAndCpuMonitor(unittest.TestCase):
    """Test cases for func_timer and cpu_monitor observability utilities."""

    def test_reused_time_func_latency_decorator_preserves_distinct_names(self):
        """Reusing a time_func_latency() decorator instance should not overwrite subsequent function labels."""
        decorator = func_timer.time_func_latency()

        @decorator
        def first_fn():
            return "first"

        @decorator
        def second_fn():
            return "second"

        @decorator
        async def third_fn():
            return "third"

        mock_histogram = MagicMock()
        with patch.object(func_timer, "enable_metrics", True), patch.object(
            func_timer, "FUNC_LATENCY", mock_histogram
        ):
            self.assertEqual(first_fn(), "first")
            self.assertEqual(second_fn(), "second")
            self.assertEqual(asyncio.run(third_fn()), "third")

        observed_names = [
            call.kwargs["name"] for call in mock_histogram.labels.call_args_list
        ]
        self.assertEqual(observed_names, ["first_fn", "second_fn", "third_fn"])

    def test_enable_func_timer_is_idempotent(self):
        """Calling enable_func_timer() multiple times in the same process should not duplicate registration."""
        func_timer.enable_func_timer()
        first_hist = func_timer.FUNC_LATENCY
        self.assertIsNotNone(first_hist)
        func_timer.enable_func_timer()
        self.assertIs(func_timer.FUNC_LATENCY, first_hist)

    def test_start_cpu_monitor_thread_is_idempotent(self):
        """Calling start_cpu_monitor_thread() multiple times in the same process should reuse the Counter."""
        with patch("threading.Thread") as mock_thread_cls:
            cpu_monitor.start_cpu_monitor_thread("scheduler", interval=60.0)
            first_counter = cpu_monitor._CPU_SECONDS_TOTAL
            self.assertIsNotNone(first_counter)
            cpu_monitor.start_cpu_monitor_thread("detokenizer", interval=60.0)
            self.assertIs(cpu_monitor._CPU_SECONDS_TOTAL, first_counter)
            self.assertEqual(mock_thread_cls.call_count, 2)
            self.assertEqual(mock_thread_cls.return_value.start.call_count, 2)


if __name__ == "__main__":
    unittest.main()
