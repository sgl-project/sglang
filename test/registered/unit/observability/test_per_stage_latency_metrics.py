"""Per-stage request latency reuses one histogram child per stage."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest import mock

import prometheus_client

from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _collector(registry):
    with (
        mock.patch("sglang.srt.observability.metrics_collector.GaugeHistogram"),
        mock.patch(
            "sglang.srt.observability.metrics_collector.get_schedule",
            return_value=SimpleNamespace(
                prefill_delayer_max_delay_passes=10,
                prefill_delayer_forward_passes_buckets=None,
                prefill_delayer_wait_seconds_buckets=None,
            ),
        ),
        mock.patch(
            "prometheus_client.Gauge",
            partial(prometheus_client.Gauge, registry=registry),
        ),
        mock.patch(
            "prometheus_client.Counter",
            partial(prometheus_client.Counter, registry=registry),
        ),
        mock.patch(
            "prometheus_client.Histogram",
            partial(prometheus_client.Histogram, registry=registry),
        ),
    ):
        return SchedulerMetricsCollector(
            labels={"model_name": "test", "moe_ep_rank": 1}, server_args=None
        )


class TestPerStageLatencyMetrics(CustomTestCase):
    def test_each_stage_observes_under_its_own_labels(self):
        registry = prometheus_client.CollectorRegistry()
        collector = _collector(registry)
        histogram = collector.per_stage_req_latency_seconds

        with mock.patch.object(histogram, "labels", wraps=histogram.labels) as labels:
            for latency in (0.002, 0.004):
                collector.observe_per_stage_req_latency("prefill_waiting", latency)
            collector.observe_per_stage_req_latency("decode_forward", 0.008)

        self.assertEqual(labels.call_count, 2)

        def sample(stage, suffix):
            return registry.get_sample_value(
                f"sglang:per_stage_req_latency_seconds_{suffix}",
                {"model_name": "test", "moe_ep_rank": "1", "stage": stage},
            )

        self.assertEqual(sample("prefill_waiting", "count"), 2)
        self.assertAlmostEqual(sample("prefill_waiting", "sum"), 0.006)
        self.assertEqual(sample("decode_forward", "count"), 1)


if __name__ == "__main__":
    unittest.main()
