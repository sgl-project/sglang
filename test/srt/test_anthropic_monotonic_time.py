"""
Unit tests to verify that the Anthropic API entrypoint uses monotonic_time()
(time.perf_counter) instead of time.time() for received_time, preventing
negative TTFT and e2e latency in Prometheus metrics.

Related issue: https://github.com/sgl-project/sglang/issues/22249
"""

import inspect
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.srt.observability.req_time_stats import monotonic_time


class TestMonotonicTimeAlias(unittest.TestCase):
    """Verify that monotonic_time is correctly aliased to time.perf_counter."""

    def test_monotonic_time_is_perf_counter(self):
        """monotonic_time must be time.perf_counter, not time.time."""
        self.assertIs(
            monotonic_time,
            time.perf_counter,
            "monotonic_time should be an alias for time.perf_counter",
        )

    def test_monotonic_time_returns_small_positive_value(self):
        """monotonic_time() should return a value much smaller than epoch time."""
        value = monotonic_time()
        # time.perf_counter() is typically in the range of 1e3-1e6 seconds
        # time.time() is epoch-based, around 1.7e9
        # If we ever accidentally switch back to time.time(), this will catch it.
        self.assertGreater(value, 0, "monotonic_time must return a positive value")
        self.assertLess(
            value,
            1e9,
            "monotonic_time should not return epoch-based time (would be ~1.7e9); "
            "it looks like time.time() is being used instead of time.perf_counter()",
        )

    def test_monotonic_time_is_monotonically_increasing(self):
        """Successive calls to monotonic_time() must be non-decreasing."""
        t1 = monotonic_time()
        t2 = monotonic_time()
        t3 = monotonic_time()
        self.assertLessEqual(t1, t2)
        self.assertLessEqual(t2, t3)


class TestAnthropicServingImports(unittest.TestCase):
    """Verify the Anthropic serving module imports and uses the right clock."""

    def test_no_time_time_usage_in_serving(self):
        """The Anthropic serving module must not use time.time() for timing."""
        from sglang.srt.entrypoints.anthropic import serving

        source = inspect.getsource(serving)
        # Check that time.time() is NOT used anywhere in the module
        self.assertNotIn(
            "time.time()",
            source,
            "Anthropic serving module should not use time.time(); "
            "use monotonic_time() instead to avoid clock mismatch with metrics",
        )

    def test_monotonic_time_imported_in_serving(self):
        """The Anthropic serving module must import monotonic_time."""
        from sglang.srt.entrypoints.anthropic import serving

        self.assertTrue(
            hasattr(serving, "monotonic_time"),
            "Anthropic serving module must import monotonic_time",
        )
        self.assertIs(
            serving.monotonic_time,
            time.perf_counter,
            "serving.monotonic_time must be time.perf_counter",
        )

    def test_no_received_time_perf_in_serving(self):
        """After the fix, received_time_perf should no longer be set separately.

        Since received_time itself is now monotonic (perf_counter based),
        there's no need for a separate received_time_perf field.
        """
        from sglang.srt.entrypoints.anthropic import serving

        source = inspect.getsource(serving)
        self.assertNotIn(
            "received_time_perf",
            source,
            "received_time_perf is redundant now that received_time uses monotonic_time(); "
            "it should be removed to avoid confusion",
        )


class TestClockConsistency(unittest.TestCase):
    """Verify that both OpenAI and Anthropic entrypoints use the same clock."""

    def test_openai_and_anthropic_use_same_clock(self):
        """Both entrypoints must import and use the same monotonic_time function."""
        from sglang.srt.entrypoints.anthropic import serving as anthropic_serving
        from sglang.srt.entrypoints.openai import serving_base as openai_serving

        self.assertIs(
            anthropic_serving.monotonic_time,
            openai_serving.monotonic_time,
            "Anthropic and OpenAI entrypoints must use the same monotonic_time function",
        )

    def test_latency_computation_yields_positive_values(self):
        """Simulate a latency computation to ensure it produces positive values.

        This replicates the pattern used in metrics collection:
            created_time = monotonic_time()  # at request arrival
            ... processing ...
            finished_time = monotonic_time()  # at completion
            latency = finished_time - created_time  # must be positive
        """
        created_time = monotonic_time()
        # Simulate some minimal processing
        _sum = sum(range(1000))
        finished_time = monotonic_time()

        latency = finished_time - created_time
        self.assertGreaterEqual(
            latency,
            0,
            "Latency computed from monotonic_time deltas must be non-negative",
        )
        # Should be small (< 1 second for trivial work)
        self.assertLess(
            latency,
            1.0,
            "Trivial work latency should be under 1 second",
        )

    def test_cross_clock_mismatch_produces_negative(self):
        """Demonstrate the bug: time.time() - time.perf_counter() is wildly wrong.

        This test documents WHY the fix is necessary: mixing wall-clock
        (time.time ~1.7e9) with monotonic (time.perf_counter ~1e5) produces
        huge negative deltas.
        """
        wall_clock = time.time()
        monotonic = time.perf_counter()

        # This simulates the old bug: received_time was wall clock,
        # but finished_time was perf_counter
        bad_latency = monotonic - wall_clock

        # The "latency" would be approximately -(1.7e9 - perf_counter) which is hugely negative
        self.assertLess(
            bad_latency,
            -1e8,
            "Mixing time.time() and time.perf_counter() produces large negative values; "
            "this is the exact bug that was fixed",
        )


if __name__ == "__main__":
    unittest.main()
