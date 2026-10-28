"""Host pool accounting and SWA prefix observability."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import prometheus_client as prom
import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability.metrics_collector import (
    RadixCacheMetricsCollector,
    SchedulerMetricsCollector,
    SchedulerStats,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs


class TestHostPoolMetrics(unittest.TestCase):
    def test_report_all_pools_and_preserve_legacy(self):
        full = SimpleNamespace(logical_size=100, available_size=lambda: 40)
        swa = SimpleNamespace(logical_size=24, available_size=lambda: 4)
        tree = SimpleNamespace(
            full_kv_pool_host=full,
            host_pool_group=SimpleNamespace(
                entries=[
                    SimpleNamespace(name=SimpleNamespace(value=name), host_pool=pool)
                    for name, pool in (("kv", full), ("swa", swa))
                ]
            ),
        )
        reporter = SimpleNamespace(
            scheduler=SimpleNamespace(enable_hierarchical_cache=True, tree_cache=tree),
            stats=SchedulerStats(),
        )
        SchedulerMetricsReporter._log_hicache_stats(reporter)
        self.assertEqual(
            reporter.stats.hicache_host_pool_tokens, {"kv": (60, 100), "swa": (20, 24)}
        )
        self.assertEqual(reporter.stats.hicache_host_used_tokens, 60)
        self.assertEqual(reporter.stats.hicache_host_total_tokens, 100)
        tree.host_pool_group = None
        SchedulerMetricsReporter._log_hicache_stats(reporter)
        self.assertEqual(reporter.stats.hicache_host_pool_tokens, {"kv": (60, 100)})

    def test_host_eviction_counts_host_frees_not_device_tracker(self):
        full, swa = ComponentType.FULL, ComponentType.SWA
        result = SimpleNamespace(
            tracker={full: 8, swa: 100},
            device_frees={swa: [torch.arange(96)]},
            host_frees={full: [torch.arange(8)], swa: [torch.arange(4)]},
        )
        cache = SimpleNamespace(
            host_memory_mode="cache",
            tree_core=SimpleNamespace(drive_host_eviction=Mock(return_value=result)),
            metrics_collector=Mock(),
        )
        # Production freeing consumes the lists; collect sizes before draining.
        cache._free_values = lambda device, host: (device.clear(), host.clear())
        self.assertEqual(UnifiedRadixCache.evict_host(cache, 4, swa), 100)
        calls = cache.metrics_collector.increment_host_evicted_tokens.call_args_list
        self.assertEqual(
            [c.kwargs for c in calls],
            [
                dict(num_tokens=8, pool="kv", trigger="swa"),
                dict(num_tokens=4, pool="swa", trigger="swa"),
            ],
        )
        cache.host_memory_mode = "buffer_only"
        cache.metrics_collector.reset_mock()
        self.assertEqual(UnifiedRadixCache.evict_host(cache, 4, swa), 0)
        cache.metrics_collector.increment_host_evicted_tokens.assert_not_called()

    def test_first_prefill_counts_only_swa_gap(self):
        for actual_prefix, expected in ((0, 30), (60, 30), (80, 10), (95, 0)):
            with self.subTest(actual_prefix=actual_prefix):
                req = SimpleNamespace(
                    retracted_stain=False,
                    _cache_breakdown_computed=False,
                    full_kv_hit_length=90,
                    prefix_swa_truncated_tokens=30,
                )
                collector = Mock()
                Req.record_prefix_swa_metrics(req, actual_prefix, collector)
                collector.increment_prefix_swa_truncated_tokens.assert_called_once_with(
                    expected
                )
                req._cache_breakdown_computed = True
                Req.record_prefix_swa_metrics(req, 0, collector)
                collector.increment_prefix_swa_truncated_tokens.assert_called_once()
                req._cache_breakdown_computed = False
                req.retracted_stain = True
                Req.record_prefix_swa_metrics(req, 0, collector)
                collector.increment_prefix_swa_truncated_tokens.assert_called_once()

    def test_prometheus_exposition_and_legacy_labels(self):
        registry = prom.CollectorRegistry()

        class SchedulerCollector(SchedulerMetricsCollector):
            _counter_cls = staticmethod(partial(prom.Counter, registry=registry))
            _gauge_cls = staticmethod(partial(prom.Gauge, registry=registry))
            _histogram_cls = staticmethod(partial(prom.Histogram, registry=registry))
            _summary_cls = staticmethod(partial(prom.Summary, registry=registry))

        class CacheCollector(RadixCacheMetricsCollector):
            _counter_cls = staticmethod(partial(prom.Counter, registry=registry))
            _histogram_cls = staticmethod(partial(prom.Histogram, registry=registry))

        labels = {"model_name": "test", "tp_rank": "0", "moe_ep_rank": "0"}
        args = ServerArgs(model_path="dummy")
        with get_context().override_server_args():
            collector = SchedulerCollector(labels, args, enable_hierarchical_cache=True)
            collector.log_stats(
                SchedulerStats(
                    hicache_host_used_tokens=60,
                    hicache_host_total_tokens=100,
                    hicache_host_pool_tokens={"kv": (60, 100), "swa": (20, 24)},
                )
            )
            cache_collector = CacheCollector(labels)
            cache_collector.increment_host_evicted_tokens(7, "swa", "kv")
            cache_collector.increment_prefix_swa_truncated_tokens(12)
        self.assertEqual(
            registry.get_sample_value("sglang:hicache_host_used_tokens", labels), 60
        )
        self.assertEqual(
            registry.get_sample_value(
                "sglang:hicache_host_pool_used_tokens", {**labels, "pool": "swa"}
            ),
            20,
        )
        self.assertEqual(
            registry.get_sample_value(
                "sglang:hicache_host_pool_total_tokens", {**labels, "pool": "swa"}
            ),
            24,
        )
        self.assertEqual(
            registry.get_sample_value(
                "sglang:hicache_host_evicted_tokens_total",
                {**labels, "pool": "swa", "trigger": "kv"},
            ),
            7,
        )
        self.assertEqual(
            registry.get_sample_value(
                "sglang:prefix_swa_truncated_tokens_total", labels
            ),
            12,
        )
        self.assertIn(
            b'sglang:hicache_host_pool_used_tokens{model_name="test",moe_ep_rank="0",pool="swa",tp_rank="0"} 20.0',
            prom.generate_latest(registry),
        )


if __name__ == "__main__":
    unittest.main()
