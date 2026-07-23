import unittest
from unittest.mock import call, patch

from sglang.srt.utils.gauge_histogram import BucketLabels, GaugeHistogram
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-c-test-cpu")



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


class TestGaugeHistogram(CustomTestCase):
    @patch("prometheus_client.Gauge")
    def test_set_raw_writes_each_bucket_with_range_labels(self, mock_gauge):
        metric = mock_gauge.return_value
        histogram = GaugeHistogram(
            name="queued_requests",
            documentation="Queued requests by age",
            labelnames=["model"],
            bucket_bounds=[10, 30],
        )

        histogram.set_raw({"model": "test-model"}, [2, 3, 5])

        mock_gauge.assert_called_once_with(
            name="queued_requests",
            documentation="Queued requests by age",
            labelnames=["model", "gt", "le"],
            multiprocess_mode="mostrecent",
        )
        self.assertEqual(
            metric.labels.call_args_list,
            [
                call(model="test-model", gt="0", le="10"),
                call(model="test-model", gt="10", le="30"),
                call(model="test-model", gt="30", le="+Inf"),
            ],
        )
        self.assertEqual(
            metric.labels.return_value.set.call_args_list,
            [call(2), call(3), call(5)],
        )

    @patch("prometheus_client.Gauge")
    def test_set_by_current_observations_writes_non_cumulative_counts(
        self, mock_gauge
    ):
        metric = mock_gauge.return_value
        histogram = GaugeHistogram(
            name="queued_requests",
            documentation="Queued requests by age",
            labelnames=[],
            bucket_bounds=[10, 30],
        )

        histogram.set_by_current_observations({}, [5, 10, 11, 30, 31])

        self.assertEqual(
            metric.labels.return_value.set.call_args_list,
            [call(2), call(2), call(1)],
        )

    @patch("prometheus_client.Gauge")
    def test_set_raw_rejects_incomplete_bucket_values(self, mock_gauge):
        histogram = GaugeHistogram(
            name="queued_requests",
            documentation="Queued requests by age",
            labelnames=["model"],
            bucket_bounds=[10, 30],
        )

        with self.assertRaisesRegex(
            ValueError, "expected 3, got 2"
        ):
            histogram.set_raw({"model": "test-model"}, [2, 3])

        mock_gauge.return_value.labels.assert_not_called()




if __name__ == "__main__":
    unittest.main()
