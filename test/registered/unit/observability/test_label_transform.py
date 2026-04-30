"""Unit tests for label_transform — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")

import unittest

from sglang.srt.observability.label_transform import (
    UNKNOWN_PRIORITY_VALUE,
    transform_priority,
)


class TestTransformPriority(unittest.TestCase):
    """Test cases for transform_priority."""

    def test_none_returns_unknown(self):
        """None priority returns UNKNOWN."""
        self.assertEqual(transform_priority(None), UNKNOWN_PRIORITY_VALUE)

    def test_negative_returns_low(self):
        """Priority below minimum returns LOW."""
        self.assertEqual(transform_priority(-1), "LOW")

    def test_above_max_returns_high(self):
        """Priority at or above max returns HIGH."""
        self.assertEqual(transform_priority(31), "HIGH")
        self.assertEqual(transform_priority(100), "HIGH")

    def test_in_range_returns_string(self):
        """Priority in valid range [0, 31) returns its string representation."""
        self.assertEqual(transform_priority(0), "0")
        self.assertEqual(transform_priority(15), "15")
        self.assertEqual(transform_priority(30), "30")


if __name__ == "__main__":
    unittest.main()
