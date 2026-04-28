import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollector,
    SchedulerStats,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class _FakeMetric:
    def __init__(self, *args, **kwargs):
        self.name = kwargs.get("name")
        self.value = None
        self.observations = []
        self.increments = []

    def labels(self, **kwargs):
        return self

    def set(self, value):
        self.value = value

    def observe(self, value):
        self.observations.append(value)

    def inc(self, value=1):
        self.increments.append(value)


def _scheduler_labels():
    return {
        "model_name": "test",
        "engine_type": "unified",
        "tp_rank": 0,
        "pp_rank": 0,
        "moe_ep_rank": 0,
    }


def _server_args():
    return SimpleNamespace(
        prefill_delayer_max_delay_passes=8,
        prefill_delayer_forward_passes_buckets=None,
        prefill_delayer_wait_seconds_buckets=None,
    )


def _make_collector():
    return SchedulerMetricsCollector(
        labels=_scheduler_labels(),
        server_args=_server_args(),
    )


@patch("prometheus_client.Summary", _FakeMetric)
@patch("prometheus_client.Histogram", _FakeMetric)
@patch("prometheus_client.Counter", _FakeMetric)
@patch("prometheus_client.Gauge", _FakeMetric)
class TestAdaptiveSpecMetricsCollector(CustomTestCase):
    def test_log_stats_emits_adaptive_spec_gauges(self):
        collector = _make_collector()
        stats = SchedulerStats(
            adaptive_spec_enabled=1,
            adaptive_spec_current_steps=3,
            adaptive_spec_previous_steps=1,
            adaptive_spec_num_tier_switches=2,
            adaptive_spec_ema_accept_length=2.25,
            adaptive_spec_last_batch_accept_length=2.5,
        )

        collector.log_stats(stats)

        self.assertEqual(collector.adaptive_spec_enabled.value, 1)
        self.assertEqual(collector.adaptive_spec_current_steps.value, 3)
        self.assertEqual(collector.adaptive_spec_previous_steps.value, 1)
        self.assertEqual(collector.adaptive_spec_num_tier_switches.value, 2)
        self.assertEqual(collector.adaptive_spec_ema_accept_length.value, 2.25)
        self.assertEqual(
            collector.adaptive_spec_last_batch_accept_length.value,
            2.5,
        )

    def test_log_stats_marks_adaptive_spec_disabled_by_default(self):
        collector = _make_collector()

        collector.log_stats(SchedulerStats())

        self.assertEqual(collector.adaptive_spec_enabled.value, 0)
        self.assertEqual(collector.adaptive_spec_current_steps.value, 0)


if __name__ == "__main__":
    unittest.main()
