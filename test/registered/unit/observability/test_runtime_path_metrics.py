import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)

from sglang.srt.observability.metrics_collector import (
    RadixCacheMetricsCollector,
    SchedulerMetricsCollector,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _metric_line(metrics: str, name: str, *required_labels: str) -> str:
    return next(
        line
        for line in metrics.splitlines()
        if line.startswith(name + "{")
        and all(label in line for label in required_labels)
    )


class TestRuntimePathMetrics(CustomTestCase):
    def test_scheduler_prefill_grammar_and_hicache_metrics(self) -> None:
        registry = CollectorRegistry()
        counter = partial(Counter, registry=registry)
        gauge = partial(Gauge, registry=registry)
        histogram = partial(Histogram, registry=registry)
        summary = partial(Summary, registry=registry)

        with (
            patch.object(SchedulerMetricsCollector, "_counter_cls", counter),
            patch.object(SchedulerMetricsCollector, "_gauge_cls", gauge),
            patch.object(SchedulerMetricsCollector, "_histogram_cls", histogram),
            patch.object(SchedulerMetricsCollector, "_summary_cls", summary),
            patch.object(RadixCacheMetricsCollector, "_counter_cls", counter),
            patch.object(RadixCacheMetricsCollector, "_gauge_cls", gauge),
            patch.object(RadixCacheMetricsCollector, "_histogram_cls", histogram),
        ):
            radix = RadixCacheMetricsCollector(
                labels={"cache_type": "test", "rank": "0"}
            )
            radix.observe_hicache_scheduler_phase(
                "write_completion_sync",
                duration_seconds=0.125,
                calls=4,
                max_seconds=0.05,
            )
            radix.set_hicache_pending_operations("write_acks", 3)
            radix.observe_hicache_backup(512, 0.02)

            scheduler = SchedulerMetricsCollector(
                labels={"model_name": "test", "dp_rank": "0", "moe_ep_rank": 0},
                server_args=SimpleNamespace(
                    enable_metrics=True,
                    prefill_delayer_forward_passes_buckets=None,
                    prefill_delayer_max_delay_passes=30,
                    prefill_delayer_wait_seconds_buckets=None,
                ),
            )
            scheduler.add_scheduler_phase(
                "schedule_plan",
                duration_seconds=0.25,
                calls=8,
                max_seconds=0.075,
            )
            scheduler.set_runtime_gc_frozen()
            scheduler.observe_prefill_execution(
                outcome="cuda_graph",
                scheduled_tokens=129,
                executed_tokens=256,
                requests=7,
                bucket_tokens=256,
            )

        metrics = generate_latest(registry).decode()
        expected = {
            (
                "sglang:hicache_scheduler_phase_seconds_total",
                ('cache_type="test"', 'phase="write_completion_sync"', 'rank="0"'),
            ): "0.125",
            (
                "sglang:hicache_scheduler_phase_calls_total",
                ('cache_type="test"', 'phase="write_completion_sync"', 'rank="0"'),
            ): "4.0",
            (
                "sglang:hicache_scheduler_phase_max_seconds",
                ('cache_type="test"', 'phase="write_completion_sync"', 'rank="0"'),
            ): "0.05",
            (
                "sglang:hicache_pending_operations",
                ('cache_type="test"', 'kind="write_acks"', 'rank="0"'),
            ): "3.0",
            (
                "sglang:scheduler_phase_seconds_total",
                ('dp_rank="0"', 'model_name="test"', 'phase="schedule_plan"'),
            ): "0.25",
            (
                "sglang:scheduler_phase_calls_total",
                ('dp_rank="0"', 'model_name="test"', 'phase="schedule_plan"'),
            ): "8.0",
            (
                "sglang:scheduler_phase_max_seconds",
                ('dp_rank="0"', 'model_name="test"', 'phase="schedule_plan"'),
            ): "0.075",
            (
                "sglang:runtime_gc_frozen",
                ('dp_rank="0"', 'model_name="test"'),
            ): "1.0",
            (
                "sglang:prefill_graph_admissions_total",
                ('bucket="256"', 'outcome="cuda_graph"'),
            ): "1.0",
            (
                "sglang:prefill_graph_shapes_total",
                ('bucket="256"', 'request_slots="other"'),
            ): "1.0",
            (
                "sglang:prefill_execution_tokens_total",
                ('bucket="256"', 'kind="scheduled"', 'path="cuda_graph"'),
            ): "129.0",
            (
                "sglang:prefill_execution_tokens_total",
                ('bucket="256"', 'kind="executed"', 'path="cuda_graph"'),
            ): "256.0",
        }
        for (name, labels), value in expected.items():
            with self.subTest(metric=name, labels=labels):
                self.assertEqual(
                    _metric_line(metrics, name, *labels).split()[-1],
                    value,
                )

        self.assertIn(
            'sglang:hicache_backup_tokens_total{cache_type="test",rank="0"} 512.0',
            metrics,
        )


if __name__ == "__main__":
    unittest.main()
