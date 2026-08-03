import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.srt.observability.startup_time import (
    build_engine_startup_time,
    build_scheduler_startup_time,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _RecordingMetric:
    instances = {}

    def __init__(self, name, documentation, labelnames=(), **kwargs):
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


class _RecordingTokenizerMetricsCollector(TokenizerMetricsCollector):
    _counter_cls = _RecordingMetric
    _gauge_cls = _RecordingMetric
    _histogram_cls = _RecordingMetric


class TestStartupTime(CustomTestCase):
    def test_tokenizer_manager_stores_and_emits_startup_time(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.enable_metrics = True
        manager.metrics_collector = Mock()
        startup_time = {"load_weight": 1.0, "e2e": 2.0}

        manager.set_startup_time(startup_time)

        self.assertIs(manager.startup_time, startup_time)
        manager.metrics_collector.emit_startup_time.assert_called_once_with(
            startup_time
        )

    def test_builds_scheduler_breakdown(self):
        startup_time = build_scheduler_startup_time(
            target_load_weight=10.0,
            draft_load_weight=2.0,
            kv_cache_allocation=3.0,
            target_cuda_graph={"prefill": 4.0, "decode": 5.0},
            draft_cuda_graph={"target_verify": 6.0, "draft_decode": 7.0},
        )

        self.assertEqual(startup_time["load_weight"], 12.0)
        self.assertEqual(startup_time["kv_cache_allocation"], 3.0)
        self.assertEqual(startup_time["cuda_graph"]["prefill"], 4.0)
        self.assertEqual(startup_time["cuda_graph"]["decode"], 5.0)
        self.assertEqual(startup_time["cuda_graph"]["target_verify"], 6.0)
        self.assertEqual(startup_time["cuda_graph"]["draft_decode"], 7.0)

    def test_engine_uses_slowest_scheduler_rank(self):
        startup_time = build_engine_startup_time(
            [
                {
                    "load_weight": 10.0,
                    "kv_cache_allocation": 4.0,
                    "cuda_graph": {"prefill": 2.0, "decode": 5.0},
                },
                {
                    "load_weight": 11.0,
                    "kv_cache_allocation": 3.0,
                    "cuda_graph": {"prefill": 3.0, "decode": 4.0},
                },
            ],
            e2e=20.0,
        )

        self.assertEqual(startup_time["load_weight"], 11.0)
        self.assertEqual(startup_time["kv_cache_allocation"], 4.0)
        self.assertEqual(startup_time["cuda_graph"]["prefill"], 3.0)
        self.assertEqual(startup_time["cuda_graph"]["decode"], 5.0)
        self.assertEqual(startup_time["e2e"], 20.0)

    def test_emits_prometheus_breakdown(self):
        _RecordingMetric.instances.clear()
        labels = {"model_name": "test-model", "engine_type": "unified"}
        collector = _RecordingTokenizerMetricsCollector(
            server_args=SimpleNamespace(
                prompt_tokens_buckets=None,
                generation_tokens_buckets=None,
            ),
            labels=labels,
        )

        collector.emit_startup_time(
            {
                "load_weight": 10.0,
                "kv_cache_allocation": 3.0,
                "cuda_graph": {"prefill": 4.0, "decode": 5.0},
                "e2e": 20.0,
            }
        )

        metrics = _RecordingMetric.instances
        self.assertEqual(
            metrics["sglang:startup_time_seconds"].calls,
            [
                (10.0, {**labels, "phase": "load_weight"}),
                (3.0, {**labels, "phase": "kv_cache_allocation"}),
                (20.0, {**labels, "phase": "e2e"}),
            ],
        )
        self.assertEqual(
            metrics["sglang:startup_cuda_graph_time_seconds"].calls,
            [
                (4.0, {**labels, "phase": "prefill"}),
                (5.0, {**labels, "phase": "decode"}),
            ],
        )


if __name__ == "__main__":
    unittest.main()
