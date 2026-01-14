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


class TestBucketLabelsCounts(unittest.TestCase):
    """Test BucketLabels.compute_bucket_counts with hardcoded expected values."""

    def test_empty_observations(self):
        buckets = BucketLabels([10, 30, 60])
        self.assertEqual(buckets.compute_bucket_counts([]), [0, 0, 0, 0])

    def test_single_value_first_bucket(self):
        # bounds: [10, 30, 60] -> buckets: (0,10], (10,30], (30,60], (60,+Inf]
        buckets = BucketLabels([10, 30, 60])
        self.assertEqual(buckets.compute_bucket_counts([5]), [1, 0, 0, 0])

    def test_single_value_last_bucket(self):
        buckets = BucketLabels([10, 30, 60])
        self.assertEqual(buckets.compute_bucket_counts([100]), [0, 0, 0, 1])

    def test_exact_boundary_values(self):
        # Values at exact boundaries: 10 -> (0,10], 30 -> (10,30], 60 -> (30,60]
        buckets = BucketLabels([10, 30, 60])
        self.assertEqual(buckets.compute_bucket_counts([10, 30, 60]), [1, 1, 1, 0])

    def test_just_above_boundary(self):
        # 11 -> (10,30], 31 -> (30,60], 61 -> (60,+Inf]
        buckets = BucketLabels([10, 30, 60])
        self.assertEqual(buckets.compute_bucket_counts([11, 31, 61]), [0, 1, 1, 1])

    def test_multiple_values_same_bucket(self):
        buckets = BucketLabels([10, 30, 60])
        self.assertEqual(buckets.compute_bucket_counts([1, 2, 3, 4, 5]), [5, 0, 0, 0])

    def test_all_overflow(self):
        buckets = BucketLabels([10, 30, 60])
        self.assertEqual(buckets.compute_bucket_counts([100, 200, 300]), [0, 0, 0, 3])

    def test_distribution(self):
        # 5 (<=10), 10 (<=10), 15 (<=30), 40 (<=60), 100 (+Inf)
        buckets = BucketLabels([10, 30, 60])
        self.assertEqual(
            buckets.compute_bucket_counts([5, 10, 15, 40, 100]), [2, 1, 1, 1]
        )

    def test_float_values(self):
        # 9.9 -> (0,10], 10.1 -> (10,30], 30.5 -> (30,60]
        buckets = BucketLabels([10, 30, 60])
        self.assertEqual(buckets.compute_bucket_counts([9.9, 10.1, 30.5]), [1, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
