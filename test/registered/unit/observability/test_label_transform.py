"""Unit tests for label_transform — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-c-test-cpu")

import unittest

from sglang.srt.observability.label_transform import (
    UNKNOWN_PRIORITY_VALUE,
    transform_priority,
)


class TestTransformPriority(unittest.TestCase):
    """Test cases for the transform_priority function.

    The function maps optional integer priorities into bounded string labels
    to prevent Prometheus label cardinality explosion.
    """

    # ── None handling ──

    def test_none_returns_unknown(self):
        """None priority should return UNKNOWN."""
        self.assertEqual(transform_priority(None), UNKNOWN_PRIORITY_VALUE)

    # ── Below-range values ──

    def test_negative_one_returns_low(self):
        self.assertEqual(transform_priority(-1), "LOW")

    def test_large_negative_returns_low(self):
        self.assertEqual(transform_priority(-1000), "LOW")

    # ── In-range values ──

    def test_zero_returns_str(self):
        """Minimum in-range value (0) should be returned as string."""
        self.assertEqual(transform_priority(0), "0")

    def test_mid_range_value(self):
        self.assertEqual(transform_priority(15), "15")

    def test_max_in_range_value(self):
        """The largest in-range value is 30 (PRIORITY_MAX - 1)."""
        self.assertEqual(transform_priority(30), "30")

    # ── Above-range values ──

    def test_priority_max_returns_high(self):
        """PRIORITY_MAX (31) should return HIGH."""
        self.assertEqual(transform_priority(31), "HIGH")

    def test_large_positive_returns_high(self):
        self.assertEqual(transform_priority(9999), "HIGH")

    # ── Boundary sweep ──

    def test_boundary_sweep(self):
        """Sweep all boundary values to confirm correct transitions."""
        expected = {
            -2: "LOW",
            -1: "LOW",
            0: "0",
            1: "1",
            30: "30",
            31: "HIGH",
            32: "HIGH",
        }
        for val, exp in expected.items():
            with self.subTest(priority=val):
                self.assertEqual(transform_priority(val), exp)


if __name__ == "__main__":
    unittest.main()
