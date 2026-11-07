"""sglang:prefill_state_miss_tokens_total follows the prefill stats it is given."""

import unittest
from functools import partial
from unittest.mock import patch

import prometheus_client

from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

METRIC = "sglang:prefill_state_miss_tokens_total"


class TestPrefillStateMissMetric(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args()
        self.server_args = override.install()
        self.addCleanup(override.restore)

    def test_state_miss_tokens_increment_the_counter(self):
        registry = prometheus_client.CollectorRegistry()
        labels = {"model_name": "test", "moe_ep_rank": 0}
        sample_labels = {key: str(value) for key, value in labels.items()}
        with patch.multiple(
            prometheus_client,
            **{
                kind: partial(getattr(prometheus_client, kind), registry=registry)
                for kind in ("Counter", "Gauge", "Histogram", "Summary")
            },
        ):
            collector = SchedulerMetricsCollector(
                labels=labels, server_args=self.server_args
            )
        self.assertEqual(registry.get_sample_value(METRIC, sample_labels), 0.0)

        collector.increment_effective_prefill_tokens(
            input_tokens=300,
            device_hit_tokens=64,
            host_hit_tokens=0,
            storage_hit_tokens=0,
            state_miss_tokens=192,
        )
        collector.increment_effective_prefill_tokens(
            input_tokens=10,
            device_hit_tokens=0,
            host_hit_tokens=0,
            storage_hit_tokens=0,
        )
        self.assertEqual(registry.get_sample_value(METRIC, sample_labels), 192.0)


if __name__ == "__main__":
    unittest.main()
