from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest
from unittest.mock import patch

from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollector,
    bucket_decode_reqs,
    bucket_prefill_tokens,
)


class _FakeMetricChild:
    def __init__(self, parent, labels):
        self._parent = parent
        self._labels = labels

    def observe(self, value):
        self._parent.observations.append((self._labels, value))

    def set(self, value):
        self._parent.sets.append((self._labels, value))

    def inc(self, value=1):
        self._parent.incs.append((self._labels, value))


class _FakeMetric:
    def __init__(self, name, documentation="", labelnames=None, **kwargs):
        self.name = name
        self.documentation = documentation
        self.labelnames = list(labelnames or [])
        self.kwargs = kwargs
        self.observations = []
        self.sets = []
        self.incs = []

    def labels(self, **labels):
        return _FakeMetricChild(self, labels)


class _FakeSchedulerMetricsCollector(SchedulerMetricsCollector):
    _counter_cls = _FakeMetric
    _gauge_cls = _FakeMetric
    _histogram_cls = _FakeMetric
    _summary_cls = _FakeMetric


class _FakeGaugeHistogram:
    def __init__(self, name, documentation="", labelnames=None, **kwargs):
        self.name = name
        self.documentation = documentation
        self.labelnames = list(labelnames or [])
        self.kwargs = kwargs
        self.observations = []

    def set_by_current_observations(self, labels, observations):
        self.observations.append((labels, observations))


def _make_collector():
    with patch(
        "sglang.srt.observability.metrics_collector.GaugeHistogram",
        _FakeGaugeHistogram,
    ):
        return _FakeSchedulerMetricsCollector(
            labels={
                "model_name": "test-model",
                "engine_type": "server",
                "tp_rank": 0,
                "pp_rank": 0,
                "moe_ep_rank": 0,
            },
            server_args=types.SimpleNamespace(
                prefill_delayer_max_delay_passes=200,
                prefill_delayer_forward_passes_buckets=None,
                prefill_delayer_wait_seconds_buckets=None,
            ),
        )


class TestSchedulerMetricsCollector(unittest.TestCase):
    def test_prefill_token_buckets(self):
        cases = [
            (-1, "0"),
            (0, "0"),
            (1, "1_1k"),
            (1023, "1_1k"),
            (1024, "1k_4k"),
            (4095, "1k_4k"),
            (4096, "4k_16k"),
            (16383, "4k_16k"),
            (16384, "16k_64k"),
            (65535, "16k_64k"),
            (65536, "64k_plus"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(bucket_prefill_tokens(value), expected)

    def test_decode_request_buckets(self):
        cases = [
            (-1, "0"),
            (0, "0"),
            (1, "1"),
            (2, "2_4"),
            (4, "2_4"),
            (5, "5_8"),
            (8, "5_8"),
            (9, "9_16"),
            (16, "9_16"),
            (17, "17_32"),
            (32, "17_32"),
            (33, "33_plus"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(bucket_decode_reqs(value), expected)

    def test_prefill_only_does_not_observe_decode_step_latency(self):
        collector = _make_collector()

        collector.observe_forward_pass_interference(
            duration_seconds=0.123,
            phase="prefill_only",
            prefill_tokens=2048,
            decode_reqs=0,
        )

        self.assertEqual(len(collector.forward_pass_duration_seconds.observations), 1)
        labels, value = collector.forward_pass_duration_seconds.observations[0]
        self.assertEqual(value, 0.123)
        self.assertEqual(labels["phase"], "prefill_only")
        self.assertEqual(labels["prefill_tokens_bucket"], "1k_4k")
        self.assertEqual(labels["decode_reqs_bucket"], "0")
        self.assertEqual(len(collector.decode_step_latency_seconds.observations), 0)

    def test_mixed_observes_decode_step_with_prefill_bucket(self):
        collector = _make_collector()

        collector.observe_forward_pass_interference(
            duration_seconds=0.250,
            phase="mixed",
            prefill_tokens=20000,
            decode_reqs=6,
        )

        self.assertEqual(len(collector.forward_pass_duration_seconds.observations), 1)
        self.assertEqual(len(collector.decode_step_latency_seconds.observations), 1)
        labels, value = collector.decode_step_latency_seconds.observations[0]
        self.assertEqual(value, 0.250)
        self.assertEqual(labels["phase"], "mixed")
        self.assertEqual(labels["co_scheduled_prefill_bucket"], "16k_64k")
        self.assertEqual(labels["decode_reqs_bucket"], "5_8")

    def test_decode_only_observes_zero_prefill_bucket(self):
        collector = _make_collector()

        collector.observe_forward_pass_interference(
            duration_seconds=0.010,
            phase="decode_only",
            prefill_tokens=0,
            decode_reqs=1,
        )

        labels, value = collector.decode_step_latency_seconds.observations[0]
        self.assertEqual(value, 0.010)
        self.assertEqual(labels["phase"], "decode_only")
        self.assertEqual(labels["co_scheduled_prefill_bucket"], "0")
        self.assertEqual(labels["decode_reqs_bucket"], "1")


if __name__ == "__main__":
    unittest.main()
