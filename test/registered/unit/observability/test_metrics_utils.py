import unittest

from sglang.srt.observability.utils import (
    generate_buckets,
    two_sides_exponential_buckets,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="stage-b-test-cpu-intel")


class TestMetricsUtils(unittest.TestCase):
    """Test cases for metrics utility functions."""

    def test_two_sides_exponential_buckets_specific_values(self):
        """Test specific values for two_sides_exponential_buckets."""
        buckets = two_sides_exponential_buckets(middle=100.0, base=2.0, count=4)
        expected_values = [96.0, 98.0, 100.0, 102.0, 104.0]
        self.assertEqual(buckets, expected_values)

    def test_two_sides_exponential_buckets_negative_values(self):
        """Test two_sides_exponential_buckets with values that could go negative."""
        buckets = two_sides_exponential_buckets(middle=5.0, base=3.0, count=4)

        # Should not contain negative values (max(0, middle - distance))
        for bucket in buckets:
            self.assertGreaterEqual(bucket, 0.0)

        # Should contain the middle value
        self.assertIn(5.0, buckets)

    def test_generate_buckets_default(self):
        """Test generate_buckets with default rule."""
        default_buckets = [1.0, 5.0, 10.0, 50.0, 100.0]

        # Test with "default" rule
        result = generate_buckets(["default"], default_buckets)
        self.assertEqual(result, default_buckets)

        # Test with None (should default to "default")
        result = generate_buckets(None, default_buckets)
        self.assertEqual(result, default_buckets)

    def test_generate_buckets_custom(self):
        """Test generate_buckets with custom rule."""
        default_buckets = [1.0, 5.0, 10.0]

        # Test with "custom" rule
        result = generate_buckets(
            ["custom", "1.5", "3.2", "7.8", "15.6"], default_buckets
        )
        expected = [1.5, 3.2, 7.8, 15.6]
        self.assertEqual(result, expected)

    def test_generate_buckets_custom_with_integers(self):
        """Test generate_buckets with custom rule using integer strings."""
        default_buckets = [1.0, 5.0, 10.0]

        # Test with integer strings
        result = generate_buckets(["custom", "1", "5", "10", "50"], default_buckets)
        expected = [1.0, 5.0, 10.0, 50.0]
        self.assertEqual(result, expected)

    def test_generate_buckets_preserves_order_and_type(self):
        """Test that generate_buckets preserves order and returns floats."""
        default_buckets = [1, 5, 10, 50, 100]  # integers

        # Test default rule
        result = generate_buckets(["default"], default_buckets)
        self.assertEqual(result, default_buckets)
        self.assertIsInstance(result, list)

        # Test custom rule with proper float conversion
        result = generate_buckets(
            ["custom", "100", "50", "10", "5", "1"], default_buckets
        )
        expected = [1.0, 5.0, 10.0, 50.0, 100.0]
        self.assertEqual(result, expected)

        # All values should be floats
        for value in result:
            self.assertIsInstance(value, float)


class TestSchedulerMetricsLabels(unittest.TestCase):
    def test_attention_dp_populates_dp_rank_label(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from sglang.srt.observability.metrics_collector import (
            SchedulerMetricsCollector,
        )

        with (
            patch(
                "sglang.srt.observability.metrics_collector.get_parallel",
                return_value=SimpleNamespace(
                    tp_rank=0,
                    pp_rank=0,
                    moe_ep_rank=0,
                    dp_rank=None,
                    attn_tp_rank=0,
                    attn_cp_rank=0,
                    attn_dp_size=4,
                    attn_dp_rank=2,
                ),
            ),
            patch(
                "sglang.srt.observability.metrics_collector.get_serving",
                return_value=SimpleNamespace(
                    served_model_name="model", enable_streaming_session=False
                ),
            ),
            patch(
                "sglang.srt.observability.metrics_collector.get_disagg",
                return_value=SimpleNamespace(
                    enable_pd_role_switch=False, disaggregation_mode=None
                ),
            ),
            patch(
                "sglang.srt.observability.metrics_collector.get_observability",
                return_value=SimpleNamespace(
                    enable_metrics=True,
                    enable_metrics_for_all_schedulers=False,
                    kv_events_config=None,
                    extra_metric_labels=None,
                ),
            ),
            patch(
                "sglang.srt.observability.metrics_collector.resolve_collector_class",
                return_value=lambda labels, **kwargs: SimpleNamespace(labels=labels),
            ),
        ):
            ctx = SchedulerMetricsCollector.init_new(
                server_args=None,
                enable_priority_scheduling=False,
                enable_lora=False,
                enable_hierarchical_cache=False,
            )
            self.assertEqual(ctx.collector.labels["dp_rank"], 2)


if __name__ == "__main__":
    unittest.main()
