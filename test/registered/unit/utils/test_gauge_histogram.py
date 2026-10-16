from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=4, suite="stage-b-test-cpu-intel")

import unittest

from sglang.srt.utils.gauge_histogram import BucketLabels


class TestBucketLabels(unittest.TestCase):
    """Test BucketLabels with hardcoded expected values."""

    def test_labels_basic(self):
        buckets = BucketLabels([10, 30, 60])
        self.assertEqual(
            list(buckets),
            [("0", "10"), ("10", "30"), ("30", "60"), ("60", "+Inf")],
        )

    def test_labels_single_bound(self):
        buckets = BucketLabels([100])
        self.assertEqual(list(buckets), [("0", "100"), ("100", "+Inf")])

    def test_labels_many_bounds(self):
        buckets = BucketLabels([1, 2, 5, 10])
        self.assertEqual(
            list(buckets),
            [("0", "1"), ("1", "2"), ("2", "5"), ("5", "10"), ("10", "+Inf")],
        )

    def test_len(self):
        buckets = BucketLabels([10, 30, 60])
        self.assertEqual(len(buckets), 4)


if __name__ == "__main__":
    unittest.main()
