"""Unified memory's observer numbers: each sub-pool's live, hole and pending
pages, the free gaps between sub-pools, compactions split by urgency, and
the plans that served takes, exported as Prometheus series."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest import mock

import prometheus_client
import torch

from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.kv.alloc.stats import PLAN_OUTCOMES
from sglang.srt.mem_cache.kv.storage import SubPoolUsage
from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollector,
    SchedulerStats,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    build_tri_pool,
    chain_snapshot,
    setup_allocator_context,
)

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


def _samples(registry, prefix="sglang:unified_memory_"):
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in registry.collect()
        for sample in family.samples
        if sample.name.startswith(prefix) and not sample.name.endswith("_created")
    }


class TestUnifiedMemoryStats(CustomTestCase):
    def setUp(self):
        setup_allocator_context()
        self.bundle, self.allocator = build_tri_pool()
        self.full = self.allocator.full_attn_allocator

    def _take(self, n):
        return [self.allocator.alloc(1) for _ in range(n)]

    def test_layout_partitions_each_span_and_the_free_bytes(self):
        """A lazy end's span is its live pages plus its holes; the gaps hold
        every byte no sub-pool spans."""
        taken = self._take(8)
        self.allocator.free(torch.cat([taken[1], taken[3], taken[5]]))
        stats = self.allocator.stats()
        full = stats.bands[self.full.sub_pool_name]
        self.assertEqual((full.live_pages, full.hole_pages), (5, 3))

        snapshot = chain_snapshot(self.allocator)
        for band in snapshot.bands:
            layout = stats.bands[band.name]
            if band.lazy and band.grow != "float":
                self.assertEqual(
                    layout.live_pages + layout.hole_pages + layout.pending_pages,
                    band.span,
                )
        opaque = [b for b in snapshot.bands if not b.transparent]
        spanned = sum(b.high_byte - b.low_byte for b in opaque)
        self.assertEqual(
            stats.gap_bytes,
            opaque[-1].high_byte - opaque[0].low_byte - spanned,
        )
        self.assertLessEqual(stats.largest_gap_bytes, stats.gap_bytes)

    def test_compactions_are_counted_by_urgency(self):
        """An opportunistic pass and an urgent pass land in separate counts,
        each with the pages it moved."""
        taken = self._take(8)
        self.allocator.free(torch.cat([taken[1], taken[3]]))
        moved = self.full.flush_opportunistic()
        self.assertGreater(moved, 0)
        counters = self.allocator.stats().bands[self.full.sub_pool_name].counters
        self.assertEqual(counters.opportunistic_compactions, 1)
        self.assertEqual(counters.opportunistic_compaction_moves, moved)
        self.assertEqual(counters.urgent_compactions, 0)

        taken = self._take(4)
        self.allocator.free(torch.cat([taken[0], taken[2]]))
        moved = self.full.flush_for_allocation()
        self.assertGreater(moved, 0)
        counters = self.allocator.stats().bands[self.full.sub_pool_name].counters
        self.assertEqual(counters.urgent_compactions, 1)
        self.assertEqual(counters.urgent_compaction_moves, moved)
        self.assertEqual(counters.opportunistic_compactions, 1)

    def test_storage_reports_bands_by_published_label(self):
        storage = self.bundle.storage
        labels = {u.label for u in storage.usage()}
        self.assertEqual(set(storage.memory_stats().bands), labels)

    def test_snapshot_counters_do_not_follow_later_work(self):
        """What the observer read stays what it was when read."""
        before = self.allocator.stats()
        self._take(2)
        self.assertEqual(before.counters.takes, 0)
        self.assertEqual(before.counters.plans, {})


class TestUnifiedMemoryMetrics(CustomTestCase):
    def setUp(self):
        setup_allocator_context()
        self.bundle, self.allocator = build_tri_pool()

    def _observer(self, storage):
        return SchedulerPoolStatsObserver(
            tree_cache=SimpleNamespace(
                supports_mamba=lambda: False,
                window_sub_pools=lambda: (),
                evictable_size=lambda: 0,
                full_evictable_size=lambda: 0,
            ),
            storage=storage,
            token_to_kv_pool_allocator=SimpleNamespace(
                target_window_pool=lambda: None,
                state=SimpleNamespace(ckpt_pool=None),
            ),
            session_controller=None,
            hisparse_coordinator=None,
            enable_hisparse=False,
        )

    def _log(self, collector, storage):
        stats = SchedulerStats()
        self._observer(storage).get_pool_stats().update_scheduler_stats(stats)
        collector.log_stats(stats)

    def test_counters_take_the_increase_since_the_last_sample(self):
        """The allocator keeps totals; a sample with nothing new adds nothing."""
        storage = self.bundle.storage
        registry = prometheus_client.CollectorRegistry()
        collector = _collector(registry)
        for _ in range(3):
            self.allocator.alloc(1)
        self._log(collector, storage)
        self._log(collector, storage)
        takes = (
            "sglang:unified_memory_takes_total",
            (("model_name", "test"), ("moe_ep_rank", "1")),
        )
        self.assertEqual(_samples(registry)[takes], 3)
        self.allocator.alloc(1)
        self._log(collector, storage)
        samples = _samples(registry)
        self.assertEqual(samples[takes], 4)
        live = (
            "sglang:unified_memory_live_pages",
            (("model_name", "test"), ("moe_ep_rank", "1"), ("sub_pool", "full")),
        )
        self.assertEqual(samples[live], 4)

    def test_every_series_exists_before_its_first_event(self):
        """A rate over a counter needs its zero: no compaction, refusal or
        relayout has happened yet, and their series already read 0."""
        registry = prometheus_client.CollectorRegistry()
        self._log(_collector(registry), self.bundle.storage)
        base = (("model_name", "test"), ("moe_ep_rank", "1"))
        samples = _samples(registry)
        for step in PLAN_OUTCOMES:
            key = ("sglang:unified_memory_plans_total", (*base, ("step", step)))
            self.assertEqual(samples[key], 0)
        for mode in ("urgent", "opportunistic"):
            labels = tuple(sorted((*base, ("mode", mode), ("sub_pool", "full"))))
            key = ("sglang:unified_memory_compactions_total", labels)
            self.assertEqual(samples[key], 0)

    def test_static_pools_publish_no_unified_series(self):
        storage = SimpleNamespace(
            usage=lambda: (
                SubPoolUsage(
                    name="full", metrics="full", label="full", capacity=10, available=10
                ),
            ),
            memory_stats=lambda: None,
        )
        registry = prometheus_client.CollectorRegistry()
        self._log(_collector(registry), storage)
        self.assertEqual(_samples(registry), {})


if __name__ == "__main__":
    unittest.main()
