"""CPU-only unit tests for priority-label bucketing.

The corresponding GPU smoke test that drives a real server is in
test/registered/observability/test_priority_metrics.py. This file isolates the
pure-logic surface (fold helper, QueueCount.from_reqs, PrefillStats.from_adder
plumbing, ServerArgs validation, and the in-process fold inside
TokenizerManager.collect_metrics) so it can run on the CPU CI lane.
"""

import unittest
from unittest.mock import Mock

from sglang.srt.observability.metrics_collector import (
    QueueCount,
    fold_priority_label,
)
from sglang.srt.observability.scheduler_metrics_mixin import PrefillStats
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")


class TestQueueCount(CustomTestCase):
    """Unit tests for QueueCount and the fold helper."""

    def test_queue_count_from_reqs(self):
        """QueueCount correctly counts per-priority breakdown."""
        reqs = [
            Mock(priority=1),
            Mock(priority=1),
            Mock(priority=5),
            Mock(priority=5),
            Mock(priority=10),
        ]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=True)
        self.assertEqual(qc.total, 5)
        self.assertEqual(qc.by_priority, {1: 2, 5: 2, 10: 1})

    def test_queue_count_from_reqs_disabled(self):
        """Priority scheduling disabled → no breakdown."""
        reqs = [Mock(priority=1), Mock(priority=5)]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=False)
        self.assertEqual(qc.total, 2)
        self.assertIsNone(qc.by_priority)

    def test_queue_count_empty(self):
        """Empty request list."""
        qc = QueueCount.from_reqs([], enable_priority_scheduling=True)
        self.assertEqual(qc.total, 0)
        self.assertEqual(qc.by_priority, {})

    def test_fold_priority_label_identity(self):
        """K=1 (default) is the identity — preserves existing label values exactly."""
        for p in [-7, -1, 0, 1, 5, 17, 999]:
            self.assertEqual(fold_priority_label(p, 1), p)
        self.assertIsNone(fold_priority_label(None, 1))
        self.assertIsNone(fold_priority_label(None, 10))

    def test_fold_priority_label_buckets(self):
        """K>1 folds priorities into the lower bound of [bucket, bucket+K)."""
        # Non-negative
        self.assertEqual(fold_priority_label(0, 10), 0)
        self.assertEqual(fold_priority_label(5, 10), 0)
        self.assertEqual(fold_priority_label(9, 10), 0)
        self.assertEqual(fold_priority_label(10, 10), 10)
        self.assertEqual(fold_priority_label(19, 10), 10)
        self.assertEqual(fold_priority_label(999, 10), 990)
        # Negative — Python floor division gives the mathematically lower bucket
        self.assertEqual(fold_priority_label(-1, 10), -10)
        self.assertEqual(fold_priority_label(-10, 10), -10)
        self.assertEqual(fold_priority_label(-11, 10), -20)

    def test_queue_count_with_bucketing(self):
        """Bucketed counting collapses adjacent priorities into shared keys."""
        reqs = [
            Mock(priority=0),
            Mock(priority=3),
            Mock(priority=9),
            Mock(priority=10),
            Mock(priority=15),
            Mock(priority=29),
        ]
        qc = QueueCount.from_reqs(
            reqs,
            enable_priority_scheduling=True,
            priority_label_bucket_size=10,
        )
        self.assertEqual(qc.total, 6)
        # 0,3,9 → bucket 0; 10,15 → bucket 10; 29 → bucket 20
        self.assertEqual(qc.by_priority, {0: 3, 10: 2, 20: 1})

    def test_queue_count_bucketing_disabled_when_priority_disabled(self):
        """Bucket size has no effect when priority scheduling is disabled."""
        reqs = [Mock(priority=0), Mock(priority=99)]
        qc = QueueCount.from_reqs(
            reqs,
            enable_priority_scheduling=False,
            priority_label_bucket_size=10,
        )
        self.assertEqual(qc.total, 2)
        self.assertIsNone(qc.by_priority)


class TestPriorityLabelBucketSizeValidation(CustomTestCase):
    """ServerArgs validation for --priority-label-bucket-size."""

    def _make(self, bucket_size: int) -> ServerArgs:
        return ServerArgs(model_path="dummy", priority_label_bucket_size=bucket_size)

    def test_default_accepted(self):
        """K=1 (default) passes validation."""
        self._make(1).check_server_args()

    def test_large_bucket_accepted(self):
        """Large K passes validation."""
        self._make(1000).check_server_args()

    def test_zero_rejected(self):
        """K=0 fails validation (would crash on floor-division)."""
        with self.assertRaises(AssertionError):
            self._make(0).check_server_args()

    def test_negative_rejected(self):
        """Negative K fails validation."""
        with self.assertRaises(AssertionError):
            self._make(-5).check_server_args()


class TestPrefillStatsBucketing(CustomTestCase):
    """PrefillStats.from_adder threads priority_label_bucket_size through."""

    def test_from_adder_folds_priorities(self):
        """priority_label_bucket_size kwarg reaches QueueCount in num_running_reqs."""
        adder = Mock(
            log_input_tokens=10,
            log_hit_tokens=5,
            new_token_ratio=1.0,
            can_run_list=[Mock(), Mock()],
        )
        running_reqs = [
            Mock(priority=3),
            Mock(priority=7),
            Mock(priority=12),
            Mock(priority=27),
        ]
        stats = PrefillStats.from_adder(
            adder,
            running_reqs,
            enable_priority_scheduling=True,
            priority_label_bucket_size=10,
        )
        # 3,7 → bucket 0; 12 → bucket 10; 27 → bucket 20
        self.assertEqual(stats.num_running_reqs.total, 4)
        self.assertEqual(stats.num_running_reqs.by_priority, {0: 2, 10: 1, 20: 1})

    def test_from_adder_default_is_identity(self):
        """Without bucket_size, PrefillStats keeps raw priority keys (back-compat)."""
        adder = Mock(
            log_input_tokens=0,
            log_hit_tokens=0,
            new_token_ratio=0.0,
            can_run_list=[],
        )
        stats = PrefillStats.from_adder(
            adder,
            [Mock(priority=3), Mock(priority=7)],
            enable_priority_scheduling=True,
        )
        self.assertEqual(stats.num_running_reqs.by_priority, {3: 1, 7: 1})


class TestTokenizerManagerCollectMetricsFold(CustomTestCase):
    """The fold is applied in TokenizerManager.collect_metrics before the
    priority is stringified into a Prometheus label.

    Tests the method as an unbound function with a mock self — no model load,
    no server, no GPU. Catches regressions where someone drops the
    fold_priority_label(...) wrap at the histogram label site.
    """

    def _invoke_collect_metrics(
        self,
        priority,
        bucket_size,
        finished=True,
        ttft_observed=False,
    ):
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        mock_self = Mock()
        mock_self.enable_priority_scheduling = True
        mock_self.priority_label_bucket_size = bucket_size
        mock_self.disaggregation_mode = DisaggregationMode.NULL
        mock_self.metrics_collector.labels = {"engine_type": "test"}
        mock_self._request_has_grammar.return_value = False

        recv_obj = Mock(
            completion_tokens=[5],
            cached_tokens=[0],
            cached_tokens_details=None,
            prompt_tokens=[10],
        )
        state = Mock(
            ttft_observed=ttft_observed,
            last_completion_tokens=0,
            finished=finished,
        )
        state.obj.priority = priority
        state.obj.custom_labels = None

        TokenizerManager.collect_metrics(mock_self, state, recv_obj, 0)
        return mock_self.metrics_collector

    def test_priority_folded_for_finished_request(self):
        """observe_one_finished_request gets the folded priority label."""
        mc = self._invoke_collect_metrics(priority=17, bucket_size=10)
        labels = mc.observe_one_finished_request.call_args.args[0]
        self.assertEqual(labels["priority"], "10")

    def test_priority_folded_for_ttft(self):
        """observe_time_to_first_token gets the folded priority label."""
        mc = self._invoke_collect_metrics(priority=23, bucket_size=10)
        labels = mc.observe_time_to_first_token.call_args.args[0]
        self.assertEqual(labels["priority"], "20")

    def test_priority_unchanged_at_default_bucket_size(self):
        """K=1 (default) preserves the raw priority — back-compat."""
        mc = self._invoke_collect_metrics(priority=17, bucket_size=1)
        labels = mc.observe_one_finished_request.call_args.args[0]
        self.assertEqual(labels["priority"], "17")

    def test_priority_none_skips_label(self):
        """priority=None must NOT produce a `priority` label (avoid 'None' string)."""
        mc = self._invoke_collect_metrics(priority=None, bucket_size=10)
        labels = mc.observe_one_finished_request.call_args.args[0]
        self.assertNotIn("priority", labels)


if __name__ == "__main__":
    unittest.main()
