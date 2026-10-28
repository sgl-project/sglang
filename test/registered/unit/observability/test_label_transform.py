"""Unit tests for label_transform — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=4, suite="stage-b-test-cpu-intel")

import unittest

from sglang.srt.observability.label_transform import (
    UNKNOWN_PRIORITY_VALUE,
    transform_priority,
)


class TestTransformPriority(unittest.TestCase):
    """Test cases for transform_priority."""

    def test_label_vocabulary(self):
        """The metric label vocabulary is bounded: None, out-of-range, and
        in-range priorities map to their pinned labels."""
        self.assertEqual(transform_priority(None), UNKNOWN_PRIORITY_VALUE)
        self.assertEqual(transform_priority(-1), "LOW")
        self.assertEqual(transform_priority(31), "HIGH")
        self.assertEqual(transform_priority(15), "15")


if __name__ == "__main__":
    unittest.main()
