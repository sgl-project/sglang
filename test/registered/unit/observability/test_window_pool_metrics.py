"""Every sliding-window KV pool reports its usage under its own window name,
unified pools with several windows and static pools with one alike."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest import mock

import prometheus_client

from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.kv.storage import SubPoolUsage
from sglang.srt.mem_cache.static_kv_storage import StaticKVStorage, StaticPoolShape
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollector,
    SchedulerStats,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Three window slots of the prefix tree and the sub-pool behind each, the
# tree's first window first.
_SLOTS = {
    ComponentType.SWA: "window_128",
    "second window slot": "window_64",
    "third window slot": "window_32",
}


def _usage(name, metrics, capacity, available):
    return SubPoolUsage(
        name=name, metrics=metrics, label=name, capacity=capacity, available=available
    )


def _three_window_observer():
    # The storage lists sub-pools owner first, not in the tree's window order.
    storage = SimpleNamespace(
        usage=lambda: (
            _usage("full", "full", 1000, 600),
            _usage("window_32", "window", 100, 40),
            _usage("window_64", "window", 200, 120),
            _usage("window_128", "window", 400, 100),
        ),
        tree_sub_pool=lambda slot: _SLOTS[slot],
    )
    evictable = {"window_128": 100, "window_64": 30, "window_32": 10}
    tree_cache = SimpleNamespace(
        window_slots=lambda: tuple(_SLOTS),
        window_evictable_size=lambda slot: evictable[_SLOTS[slot]],
        full_evictable_size=lambda: 150,
    )
    return SchedulerPoolStatsObserver(
        tree_cache=tree_cache,
        storage=storage,
        token_to_kv_pool_allocator=SimpleNamespace(),
        session_controller=None,
        hisparse_coordinator=None,
        enable_hisparse=False,
    )


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


class TestWindowPoolMetrics(CustomTestCase):
    def test_three_windows_report_three_label_values(self):
        stats = _three_window_observer().get_pool_stats()
        # The tree's first window comes first, whatever the storage order.
        self.assertEqual(
            [w.name for w in stats.windows], ["window_128", "window_64", "window_32"]
        )
        by_name = {w.name: w for w in stats.windows}
        self.assertEqual(by_name["window_128"].num_used, 400 - (100 + 100))
        self.assertEqual(by_name["window_64"].num_used, 200 - (120 + 30))
        self.assertEqual(by_name["window_32"].num_used, 100 - (40 + 10))
        self.assertEqual(by_name["window_64"].token_usage, 50 / 200)
        # The bottleneck runs over every window.
        self.assertEqual(stats.get_max_pool_usage(), 0.5)
        self.assertEqual(stats.get_kv_token_stats(), (250, 0.5))

        scheduler_stats = SchedulerStats()
        stats.update_scheduler_stats(scheduler_stats)
        self.assertEqual(
            scheduler_stats.window_used_tokens,
            {"window_128": 200, "window_64": 50, "window_32": 50},
        )
        registry = prometheus_client.CollectorRegistry()
        _collector(registry).log_stats(scheduler_stats)
        samples = {
            (sample.name, sample.labels.get("window")): sample.value
            for family in registry.collect()
            for sample in family.samples
            if sample.name.startswith("sglang:window_")
        }
        for name, window in by_name.items():
            self.assertEqual(
                samples[("sglang:window_token_usage", name)], window.token_usage
            )
            self.assertEqual(
                samples[("sglang:window_available_tokens", name)],
                window.available_size,
            )
            self.assertEqual(
                samples[("sglang:window_evictable_tokens", name)],
                window.evictable_size,
            )
            self.assertEqual(
                samples[("sglang:window_used_tokens", name)], window.num_used
            )
        self.assertEqual(len(samples), 4 * 3)
        self.assertNotIn(
            "sglang:swa_token_usage",
            {family.name for family in registry.collect()},
        )

    def test_logs_name_each_window(self):
        parts = _three_window_observer().get_pool_stats().get_decode_usage_msg_parts()
        for name in ("window_128", "window_64", "window_32"):
            self.assertIn(f"#{name} token: ", " ".join(parts))
            self.assertIn(f"{name} token usage: ", " ".join(parts))

    def test_window_capacity_constant_per_window(self):
        registry = prometheus_client.CollectorRegistry()
        _collector(registry).emit_constants(
            max_total_num_tokens=1000,
            max_total_num_tokens_window={"window_128": 400, "window_64": 400},
            weight_memory_usage_gb=0.0,
            kv_cache_memory_usage_gb=0.0,
            graph_memory_usage_gb={},
            max_running_requests_under_SLO=None,
            page_size=1,
            num_pages=1000,
            context_len=128,
            startup_available_gpu_memory_gb=0.0,
        )
        samples = {
            sample.labels.get("window"): sample.value
            for family in registry.collect()
            for sample in family.samples
            if sample.name == "sglang:max_total_num_tokens_window"
        }
        self.assertEqual(samples, {"window_128": 400, "window_64": 400})

    def test_static_single_window_reports_its_length_name(self):
        allocator = SimpleNamespace(
            swa_capacity_and_available=lambda full_capacity, swa_capacity: (
                (full_capacity, 700),
                (swa_capacity, 90),
            )
        )
        storage = StaticKVStorage(
            None,
            allocator,
            shape=StaticPoolShape(
                has_window=True,
                has_state=False,
                token_capacity=1000,
                full_tokens_per_layer=1000,
                swa_tokens_per_layer=200,
                window=4096,
            ),
        )
        (full, window) = storage.usage()
        self.assertEqual((full.metrics, full.label), ("full", "full"))
        self.assertEqual(
            (window.metrics, window.label, window.capacity, window.available),
            ("window", "window_4096", 200, 90),
        )


if __name__ == "__main__":
    unittest.main()
