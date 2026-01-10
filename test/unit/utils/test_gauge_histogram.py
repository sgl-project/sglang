import unittest

from sglang.srt.utils.gauge_histogram import GaugeHistogram


class TestGaugeHistogramComputeBucketCounts(unittest.TestCase):
    """Test GaugeHistogram._compute_bucket_counts with hardcoded expected values."""

    def _make_histogram(self, name_suffix, bucket_bounds):
        """Create a real GaugeHistogram for testing."""
        return GaugeHistogram(
            name=f"test_gauge_{name_suffix}",
            documentation="Test gauge histogram",
            labelnames=[],
            bucket_bounds=bucket_bounds,
        )

    def test_empty_observations(self):
        gh = self._make_histogram("empty", [10, 30, 60])
        self.assertEqual(gh._compute_bucket_counts([]), [0, 0, 0, 0])

    def test_single_value_first_bucket(self):
        # bounds: [10, 30, 60] -> buckets: (0,10], (10,30], (30,60], (60,+Inf]
        gh = self._make_histogram("first", [10, 30, 60])
        self.assertEqual(gh._compute_bucket_counts([5]), [1, 0, 0, 0])

    def test_single_value_last_bucket(self):
        gh = self._make_histogram("last", [10, 30, 60])
        self.assertEqual(gh._compute_bucket_counts([100]), [0, 0, 0, 1])

    def test_exact_boundary_values(self):
        # Values at exact boundaries should go to the bucket where value <= upper
        gh = self._make_histogram("boundary", [10, 30, 60])
        # 10 -> (0,10], 30 -> (10,30], 60 -> (30,60]
        self.assertEqual(gh._compute_bucket_counts([10, 30, 60]), [1, 1, 1, 0])

    def test_just_above_boundary(self):
        gh = self._make_histogram("above", [10, 30, 60])
        # 11 -> (10,30], 31 -> (30,60], 61 -> (60,+Inf]
        self.assertEqual(gh._compute_bucket_counts([11, 31, 61]), [0, 1, 1, 1])

    def test_multiple_values_same_bucket(self):
        gh = self._make_histogram("same", [10, 30, 60])
        self.assertEqual(gh._compute_bucket_counts([1, 2, 3, 4, 5]), [5, 0, 0, 0])

    def test_routing_key_scenario(self):
        # Simulate routing key counts: [1, 5, 15, 250]
        # bounds: [1, 2, 5, 10, 20, 50, 100, 200]
        # 1 -> (0,1], 5 -> (2,5], 15 -> (10,20], 250 -> (200,+Inf]
        gh = self._make_histogram("routing", [1, 2, 5, 10, 20, 50, 100, 200])
        self.assertEqual(
            gh._compute_bucket_counts([1, 5, 15, 250]),
            [1, 0, 1, 0, 1, 0, 0, 0, 1],
        )

    def test_inflight_age_scenario(self):
        # Simulate request ages in seconds: [0, 45, 100, 250, 500, 700]
        # bounds: [30, 60, 180, 300, 600, 1200]
        # 0 -> (0,30], 45 -> (30,60], 100 -> (60,180], 250 -> (180,300], 500 -> (300,600], 700 -> (600,1200]
        gh = self._make_histogram("inflight", [30, 60, 180, 300, 600, 1200])
        self.assertEqual(
            gh._compute_bucket_counts([0, 45, 100, 250, 500, 700]),
            [1, 1, 1, 1, 1, 1, 0],
        )

    def test_all_overflow(self):
        gh = self._make_histogram("overflow", [10, 30, 60])
        self.assertEqual(gh._compute_bucket_counts([100, 200, 300]), [0, 0, 0, 3])

    def test_float_values(self):
        gh = self._make_histogram("float", [10, 30, 60])
        # 9.9 -> (0,10], 10.1 -> (10,30], 30.5 -> (30,60]
        self.assertEqual(gh._compute_bucket_counts([9.9, 10.1, 30.5]), [1, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
