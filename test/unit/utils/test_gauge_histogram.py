import unittest

from sglang.srt.utils.gauge_histogram import compute_bucket_counts, compute_bucket_labels


class TestComputeBucketLabels(unittest.TestCase):
    """Test compute_bucket_labels with hardcoded expected values."""

    def test_basic(self):
        labels = compute_bucket_labels([10, 30, 60])
        self.assertEqual(
            labels,
            [("0", "10"), ("10", "30"), ("30", "60"), ("60", "+Inf")],
        )

    def test_single_bound(self):
        labels = compute_bucket_labels([100])
        self.assertEqual(labels, [("0", "100"), ("100", "+Inf")])

    def test_many_bounds(self):
        labels = compute_bucket_labels([1, 2, 5, 10])
        self.assertEqual(
            labels,
            [("0", "1"), ("1", "2"), ("2", "5"), ("5", "10"), ("10", "+Inf")],
        )


class TestComputeBucketCounts(unittest.TestCase):
    """Test compute_bucket_counts with hardcoded expected values."""

    def test_empty_observations(self):
        counts = compute_bucket_counts([10, 30, 60], [])
        self.assertEqual(counts, [0, 0, 0, 0])

    def test_single_value_first_bucket(self):
        # bounds: [10, 30, 60] -> buckets: (0,10], (10,30], (30,60], (60,+Inf]
        counts = compute_bucket_counts([10, 30, 60], [5])
        self.assertEqual(counts, [1, 0, 0, 0])

    def test_single_value_last_bucket(self):
        counts = compute_bucket_counts([10, 30, 60], [100])
        self.assertEqual(counts, [0, 0, 0, 1])

    def test_exact_boundary_values(self):
        # Values at exact boundaries: 10 -> (0,10], 30 -> (10,30], 60 -> (30,60]
        counts = compute_bucket_counts([10, 30, 60], [10, 30, 60])
        self.assertEqual(counts, [1, 1, 1, 0])

    def test_just_above_boundary(self):
        # 11 -> (10,30], 31 -> (30,60], 61 -> (60,+Inf]
        counts = compute_bucket_counts([10, 30, 60], [11, 31, 61])
        self.assertEqual(counts, [0, 1, 1, 1])

    def test_multiple_values_same_bucket(self):
        counts = compute_bucket_counts([10, 30, 60], [1, 2, 3, 4, 5])
        self.assertEqual(counts, [5, 0, 0, 0])

    def test_all_overflow(self):
        counts = compute_bucket_counts([10, 30, 60], [100, 200, 300])
        self.assertEqual(counts, [0, 0, 0, 3])

    def test_distribution(self):
        # 5 (<=10), 10 (<=10), 15 (<=30), 40 (<=60), 100 (+Inf)
        counts = compute_bucket_counts([10, 30, 60], [5, 10, 15, 40, 100])
        self.assertEqual(counts, [2, 1, 1, 1])

    def test_float_values(self):
        # 9.9 -> (0,10], 10.1 -> (10,30], 30.5 -> (30,60]
        counts = compute_bucket_counts([10, 30, 60], [9.9, 10.1, 30.5])
        self.assertEqual(counts, [1, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
