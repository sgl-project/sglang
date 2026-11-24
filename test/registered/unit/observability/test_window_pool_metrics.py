"""Every sliding-window KV pool reports its usage under its own window name,
unified pools with several windows and static pools with one alike. The
target's own window also keeps the single-window `swa` names."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest import mock

import prometheus_client

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.memory_usage import build_memory_usage
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.kv.storage import SubPoolUsage
from sglang.srt.mem_cache.static_kv_storage import StaticKVStorage, StaticPoolShape
from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollector,
    SchedulerStats,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The three window sub-pools the prefix tree keeps, in its window order,
# each with the history it retains.
_WINDOWS = (("window_128", 128), ("window_64", 64), ("window_32", 32))


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
    )
    evictable = {"window_128": 100, "window_64": 30, "window_32": 10}
    tree_cache = SimpleNamespace(
        window_sub_pools=lambda: _WINDOWS,
        window_evictable_size=lambda name: evictable[name],
        full_evictable_size=lambda: 150,
    )
    return SchedulerPoolStatsObserver(
        tree_cache=tree_cache,
        storage=storage,
        # The target's window is the longest.
        token_to_kv_pool_allocator=SimpleNamespace(
            target_window_pool=lambda: "window_128"
        ),
        session_controller=None,
        hisparse_coordinator=None,
        enable_hisparse=False,
    )


def _collector(registry, labels=None):
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
            labels=labels or {"model_name": "test", "moe_ep_rank": 1},
            server_args=None,
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
        # The single-window names report the target's own window.
        target = by_name["window_128"]
        self.assertEqual(
            _legacy_samples(registry),
            {
                "sglang:swa_token_usage": target.token_usage,
                "sglang:swa_available_tokens": target.available_size,
                "sglang:swa_evictable_tokens": target.evictable_size,
                "sglang:swa_used_tokens": target.num_used,
                "sglang:prealloc_swa_token_usage": 0.0,
            },
        )

    def test_logs_name_each_window(self):
        parts = _three_window_observer().get_pool_stats().get_decode_usage_msg_parts()
        for name in ("window_128", "window_64", "window_32"):
            self.assertIn(f"#{name} token: ", " ".join(parts))
            self.assertIn(f"{name} token usage: ", " ".join(parts))

    def test_a_custom_window_label_does_not_stop_startup(self):
        """BUG REGRESSION. A custom metric label named ``window`` must not break
        startup: window series carry the window's name, others the custom value."""
        registry = prometheus_client.CollectorRegistry()
        _collector(
            registry, labels={"model_name": "test", "moe_ep_rank": 1, "window": "x"}
        ).emit_constants(
            max_total_num_tokens=1000,
            max_total_num_tokens_window={"window_128": 400, "window_64": 400},
            max_total_num_tokens_swa=400,
            weight_memory_usage_gb=0.0,
            kv_cache_memory_usage_gb=0.0,
            graph_memory_usage_gb={},
            max_running_requests_under_SLO=None,
            page_size=1,
            num_pages=1000,
            context_len=128,
            startup_available_gpu_memory_gb=0.0,
        )
        windows = {
            sample.labels["window"]: sample.value
            for family in registry.collect()
            for sample in family.samples
            if sample.name == "sglang:max_total_num_tokens_window"
        }
        self.assertEqual(windows, {"window_128": 400, "window_64": 400})
        custom = {
            sample.labels.get("window")
            for family in registry.collect()
            for sample in family.samples
            if sample.name == "sglang:max_total_num_tokens"
        }
        self.assertEqual(custom, {"x"})

    def test_a_draft_only_window_reports_its_capacity(self):
        """BUG REGRESSION. A fused draft's own window must report its capacity
        when the target has no window layers."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.swa_tokens_per_layer = None
        storage = SimpleNamespace(
            usage=lambda: (
                _usage("full", "full", 1000, 600),
                _usage("mamba", "mamba", 8, 8),
                _usage("window_64", "window", 4096, 4096),
            )
        )
        with mock.patch.object(Scheduler, "_observed_storage", return_value=storage):
            self.assertEqual(scheduler.window_token_capacities(), {"window_64": 4096})

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

    def test_static_window_keeps_the_swa_names(self):
        """A static SWA pool keeps reporting its usage gauges, capacity gauge and
        `token_capacity_swa` under the single-window names."""
        allocator = SimpleNamespace(
            swa_capacity_and_available=lambda full_capacity, swa_capacity: (
                (full_capacity, 700),
                (swa_capacity, 90),
            ),
            target_window_pool=lambda: None,
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
        observer = SchedulerPoolStatsObserver(
            tree_cache=SimpleNamespace(
                window_sub_pools=lambda: (("swa", 4096),),
                window_evictable_size=lambda name: {"swa": 30}[name],
                full_evictable_size=lambda: 100,
            ),
            storage=storage,
            token_to_kv_pool_allocator=allocator,
            session_controller=None,
            hisparse_coordinator=None,
            enable_hisparse=False,
        )
        scheduler_stats = SchedulerStats()
        observer.get_pool_stats().update_scheduler_stats(scheduler_stats)
        registry = prometheus_client.CollectorRegistry()
        collector = _collector(registry)
        collector.log_stats(scheduler_stats)
        collector.emit_constants(
            max_total_num_tokens=1000,
            max_total_num_tokens_window={"window_4096": 200},
            max_total_num_tokens_swa=200,
            weight_memory_usage_gb=0.0,
            kv_cache_memory_usage_gb=0.0,
            graph_memory_usage_gb={},
            max_running_requests_under_SLO=None,
            page_size=1,
            num_pages=1000,
            context_len=128,
            startup_available_gpu_memory_gb=0.0,
        )
        # As a static SWA pool reported them: used = capacity - (available +
        # evictable) over the window pool's capacity.
        self.assertEqual(
            _legacy_samples(registry),
            {
                "sglang:swa_token_usage": (200 - (90 + 30)) / 200,
                "sglang:swa_available_tokens": 90,
                "sglang:swa_evictable_tokens": 30,
                "sglang:swa_used_tokens": 200 - (90 + 30),
                "sglang:prealloc_swa_token_usage": 0.0,
                "sglang:max_total_num_tokens_swa": 200,
            },
        )
        memory_usage = build_memory_usage(
            weight_gb=0.0,
            kv_cache_gb=0.0,
            startup_available_gb=0.0,
            token_capacity=1000,
            token_capacity_swa=200,
            token_capacity_window={"window_4096": 200},
            target_graph_memory_usage=None,
            draft_graph_memory_usage=None,
        )
        self.assertEqual(memory_usage["token_capacity_swa"], 200)
        self.assertEqual(memory_usage["token_capacity_window"], {"window_4096": 200})

    def test_no_window_keeps_the_swa_names_empty(self):
        registry = prometheus_client.CollectorRegistry()
        _collector(registry).log_stats(SchedulerStats())
        self.assertEqual(
            set(_legacy_samples(registry).values()), {0.0}, "as without windows"
        )
        memory_usage = build_memory_usage(
            weight_gb=0.0,
            kv_cache_gb=0.0,
            startup_available_gb=0.0,
            token_capacity=1000,
            token_capacity_swa=None,
            token_capacity_window={},
            target_graph_memory_usage=None,
            draft_graph_memory_usage=None,
        )
        self.assertIsNone(memory_usage["token_capacity_swa"])


def _legacy_samples(registry):
    """The single-window `swa` series in ``registry``, by name."""
    return {
        sample.name: sample.value
        for family in registry.collect()
        for sample in family.samples
        if "swa" in sample.name and "window" not in sample.labels
    }


if __name__ == "__main__":
    unittest.main()
