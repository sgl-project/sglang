"""Prometheus metrics for HiSparse device and host pool occupancy."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest import mock

import prometheus_client

from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    PoolStats,
    SchedulerPoolStatsObserver,
)
from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollector,
    SchedulerStats,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _collector(registry, *, enable_hisparse=True):
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
            labels={"model_name": "test", "moe_ep_rank": 1},
            server_args=SimpleNamespace(enable_hisparse=enable_hisparse),
        )


def _hisparse_pool_stats(*, device_used, device_total, host_used, host_total):
    return PoolStats(
        full_num_used=0,
        full_token_usage=0.0,
        full_available_size=0,
        full_evictable_size=0,
        full_capacity=0,
        is_hisparse=True,
        hisparse_device_tokens=device_used,
        hisparse_device_capacity=device_total,
        hisparse_device_token_usage=(
            device_used / device_total if device_total else 0.0
        ),
        hisparse_host_tokens=host_used,
        hisparse_host_capacity=host_total,
        hisparse_host_token_usage=host_used / host_total if host_total else 0.0,
    )


class TestHiSparsePoolMetrics(CustomTestCase):
    def test_observer_propagates_coordinator_capacities(self):
        token_stats = SimpleNamespace(
            device_tokens=25,
            device_capacity=100,
            device_token_usage=0.25,
            host_tokens=60,
            host_capacity=300,
            host_token_usage=0.2,
        )
        observer = SchedulerPoolStatsObserver(
            tree_cache=None,
            storage=None,
            token_to_kv_pool_allocator=None,
            session_controller=None,
            hisparse_coordinator=SimpleNamespace(get_token_stats=lambda: token_stats),
            enable_hisparse=True,
        )
        pool_stats = observer._get_hisparse_token_info(
            PoolStats(
                full_num_used=0,
                full_token_usage=0.0,
                full_available_size=0,
                full_evictable_size=0,
                full_capacity=0,
            )
        )

        self.assertEqual(pool_stats.hisparse_device_tokens, 25)
        self.assertEqual(pool_stats.hisparse_device_capacity, 100)
        self.assertEqual(pool_stats.hisparse_device_token_usage, 0.25)
        self.assertEqual(pool_stats.hisparse_host_tokens, 60)
        self.assertEqual(pool_stats.hisparse_host_capacity, 300)
        self.assertEqual(pool_stats.hisparse_host_token_usage, 0.2)

    def test_idle_pools_report_zero_usage(self):
        stats = SchedulerStats()
        _hisparse_pool_stats(
            device_used=0, device_total=480_000, host_used=0, host_total=1_440_000
        ).update_scheduler_stats(stats)

        self.assertEqual(stats.hisparse_device_used_tokens, 0)
        self.assertEqual(stats.hisparse_device_total_tokens, 480_000)
        self.assertEqual(stats.hisparse_device_token_usage, 0.0)
        self.assertEqual(stats.hisparse_host_used_tokens, 0)
        self.assertEqual(stats.hisparse_host_total_tokens, 1_440_000)
        self.assertEqual(stats.hisparse_host_token_usage, 0.0)

    def test_collector_exports_device_and_host_metrics(self):
        stats = SchedulerStats()
        _hisparse_pool_stats(
            device_used=25, device_total=100, host_used=60, host_total=300
        ).update_scheduler_stats(stats)

        registry = prometheus_client.CollectorRegistry()
        _collector(registry).log_stats(stats)
        samples = {
            sample.name: sample.value
            for family in registry.collect()
            for sample in family.samples
            if sample.name.startswith("sglang:hisparse_")
        }

        self.assertEqual(samples["sglang:hisparse_device_used_tokens"], 25)
        self.assertEqual(samples["sglang:hisparse_device_total_tokens"], 100)
        self.assertEqual(samples["sglang:hisparse_device_token_usage"], 0.25)
        self.assertEqual(samples["sglang:hisparse_host_used_tokens"], 60)
        self.assertEqual(samples["sglang:hisparse_host_total_tokens"], 300)
        self.assertEqual(samples["sglang:hisparse_host_token_usage"], 0.2)

    def test_collector_omits_metrics_when_hisparse_is_disabled(self):
        registry = prometheus_client.CollectorRegistry()
        _collector(registry, enable_hisparse=False).log_stats(SchedulerStats())

        self.assertFalse(
            any(
                sample.name.startswith("sglang:hisparse_")
                for family in registry.collect()
                for sample in family.samples
            )
        )


if __name__ == "__main__":
    unittest.main()
