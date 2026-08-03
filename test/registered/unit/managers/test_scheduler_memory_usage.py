import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.scheduler_components.memory_usage import build_memory_usage
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _RecordingMetric:
    instances = {}

    def __init__(self, name, documentation, labelnames=(), **kwargs):
        self.name = name
        self.labelnames = tuple(labelnames)
        self.calls = []
        self.instances[name] = self

    def labels(self, **labels):
        return _BoundRecordingMetric(self, labels)


class _BoundRecordingMetric:
    def __init__(self, parent, labels):
        self.parent = parent
        self.labels = labels

    def set(self, value):
        self.parent.calls.append((value, self.labels))


class _RecordingSchedulerMetricsCollector(SchedulerMetricsCollector):
    _counter_cls = _RecordingMetric
    _gauge_cls = _RecordingMetric
    _histogram_cls = _RecordingMetric
    _summary_cls = _RecordingMetric


class TestSchedulerMemoryUsage(CustomTestCase):
    def test_reports_swa_capacity_and_graph_breakdown(self):
        memory_usage = build_memory_usage(
            weight_gb=15.284,
            kv_cache_gb=136.225,
            startup_available_gb=44.514,
            token_capacity=991_936,
            token_capacity_swa=9_919_360,
            target_graph_memory_usage={"prefill": 0.126, "decode": 0.314},
            draft_graph_memory_usage={
                "draft_decode": 0.205,
                "draft_extend": 0.104,
            },
        )

        self.assertEqual(memory_usage["weight"], 15.28)
        self.assertEqual(memory_usage["kvcache"], 136.22)
        self.assertEqual(memory_usage["startup_available"], 44.51)
        self.assertEqual(memory_usage["token_capacity"], 991_936)
        self.assertEqual(memory_usage["token_capacity_swa"], 9_919_360)
        self.assertEqual(
            memory_usage["graph"],
            {
                "prefill": 0.13,
                "decode": 0.31,
                "target_verify": 0.0,
                "draft_prefill": 0.0,
                "draft_decode": 0.2,
                "draft_extend": 0.1,
            },
        )

    def test_non_swa_model_and_additional_phases(self):
        memory_usage = build_memory_usage(
            weight_gb=1,
            kv_cache_gb=2,
            startup_available_gb=0.484,
            token_capacity=3,
            token_capacity_swa=None,
            target_graph_memory_usage={"target_verify": 0.25},
            draft_graph_memory_usage={
                "target_verify": 0.5,
                "adaptive_draft": 0.75,
            },
        )

        self.assertIsNone(memory_usage["token_capacity_swa"])
        self.assertEqual(memory_usage["startup_available"], 0.48)
        self.assertEqual(memory_usage["graph"]["target_verify"], 0.75)
        self.assertEqual(memory_usage["graph"]["adaptive_draft"], 0.75)

    def test_emit_constants_publishes_memory_metrics(self):
        _RecordingMetric.instances.clear()
        server_args = SimpleNamespace(
            prefill_delayer_max_delay_passes=100,
            prefill_delayer_forward_passes_buckets=None,
            prefill_delayer_wait_seconds_buckets=None,
        )
        labels = {
            "model_name": "test-model",
            "engine_type": "unified",
            "tp_rank": 0,
            "pp_rank": 0,
            "moe_ep_rank": 0,
        }
        with patch(
            "sglang.srt.observability.metrics_collector.GaugeHistogram",
            _RecordingMetric,
        ):
            collector = _RecordingSchedulerMetricsCollector(
                labels=labels,
                server_args=server_args,
            )

        collector.emit_constants(
            max_total_num_tokens=991_936,
            max_total_num_tokens_swa=9_919_360,
            weight_memory_usage_gb=15.284,
            kv_cache_memory_usage_gb=136.225,
            graph_memory_usage_gb={"prefill": 0.126, "decode": 0.314},
            max_running_requests_under_SLO=None,
            page_size=64,
            num_pages=15_499,
            context_len=131_072,
            startup_available_gpu_memory_gb=2.5,
        )

        metrics = _RecordingMetric.instances
        self.assertEqual(
            metrics["sglang:max_total_num_tokens_swa"].calls,
            [(9_919_360, labels)],
        )
        self.assertEqual(
            metrics["sglang:weight_memory_usage_gb"].calls,
            [(15.284, labels)],
        )
        self.assertEqual(
            metrics["sglang:kv_cache_memory_usage_gb"].calls,
            [(136.225, labels)],
        )
        self.assertEqual(
            metrics["sglang:graph_memory_usage_gb"].calls,
            [
                (0.126, {**labels, "phase": "prefill"}),
                (0.314, {**labels, "phase": "decode"}),
            ],
        )


if __name__ == "__main__":
    unittest.main()
