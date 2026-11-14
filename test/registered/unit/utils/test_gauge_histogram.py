from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=4, suite="stage-b-test-cpu-intel")

import unittest
from unittest.mock import patch

from sglang.srt.utils.gauge_histogram import BucketLabels, GaugeHistogram


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

    def test_labels_empty_bounds(self):
        buckets = BucketLabels([])
        self.assertEqual(list(buckets), [("0", "+Inf")])
        self.assertEqual(len(buckets), 1)
        self.assertEqual(buckets.compute_bucket_counts([1, 50, 999]), [3])

    def test_labels_many_bounds(self):
        buckets = BucketLabels([1, 2, 5, 10])
        self.assertEqual(
            list(buckets),
            [("0", "1"), ("1", "2"), ("2", "5"), ("5", "10"), ("10", "+Inf")],
        )

    def test_len(self):
        buckets = BucketLabels([10, 30, 60])
        self.assertEqual(len(buckets), 4)

    def test_upper_bounds_copied_and_validated(self):
        bounds = [10, 30, 60]
        buckets = BucketLabels(bounds)
        bounds.append(100)
        bounds[0] = 999
        self.assertEqual(
            buckets.compute_bucket_counts([5, 10, 25, 60, 61]), [2, 1, 1, 1]
        )

        with self.assertRaises(ValueError):
            BucketLabels([0, 10])
        with self.assertRaises(ValueError):
            BucketLabels([-5, 10])
        with self.assertRaises(ValueError):
            BucketLabels([10, 10, 20])
        with self.assertRaises(ValueError):
            BucketLabels([20, 10])


class TestGaugeHistogram(unittest.TestCase):
    """Test GaugeHistogram child handle caching and bucket count validation."""

    def test_handle_caching_and_observations(self):
        hist = GaugeHistogram(
            name="test_sglang_gauge_histogram_cached",
            documentation="Test histogram",
            labelnames=["model_name"],
            bucket_bounds=[1, 5, 10],
        )

        with patch.object(
            hist._gauge, "labels", wraps=hist._gauge.labels
        ) as mock_labels:
            hist.set_by_current_observations({"model_name": "m1"}, [1, 2, 5, 8, 20])
            self.assertEqual(mock_labels.call_count, 4)

            # Second update for the same label set reuses cached child Gauge handles
            hist.set_by_current_observations({"model_name": "m1"}, [3])
            self.assertEqual(mock_labels.call_count, 4)

            # New label set registers its 4 bucket handles once
            hist.set_by_current_observations({"model_name": "m2"}, [10, 11])
            self.assertEqual(mock_labels.call_count, 8)

        samples = {
            (s.labels["model_name"], s.labels["gt"], s.labels["le"]): s.value
            for metric in hist._gauge.collect()
            for s in metric.samples
        }
        self.assertEqual(samples[("m1", "0", "1")], 0.0)
        self.assertEqual(samples[("m1", "1", "5")], 1.0)
        self.assertEqual(samples[("m1", "5", "10")], 0.0)
        self.assertEqual(samples[("m1", "10", "+Inf")], 0.0)
        self.assertEqual(samples[("m2", "5", "10")], 1.0)
        self.assertEqual(samples[("m2", "10", "+Inf")], 1.0)

    def test_set_raw_validates_length(self):
        hist = GaugeHistogram(
            name="test_sglang_gauge_histogram_len_check",
            documentation="Test histogram length check",
            labelnames=["model_name"],
            bucket_bounds=[1, 5],
        )
        with self.assertRaises(ValueError):
            hist.set_raw({"model_name": "m1"}, [1, 2])
        with self.assertRaises(ValueError):
            hist.set_raw({"model_name": "m1"}, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
