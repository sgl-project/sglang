"""prealloc_{full,swa}_token_usage: KV held by PD decode requests whose
transfer is pending, on the capacity basis of full/swa_token_usage."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest import mock

import prometheus_client

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollector,
    SchedulerStats,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 64


def _transfer_req(fill_len, *, swa_tail_len=None, host_staged=False):
    # What _pre_alloc leaves on the request: the whole fill allocated, and for
    # a tail-preallocating SWA pool the head [0, fill - tail) already evicted.
    kv = ReqKvInfo(kv_allocated_len=fill_len, kv_committed_len=fill_len)
    if swa_tail_len is not None:
        kv.set_evicted_seqlen(ComponentType.SWA, fill_len - swa_tail_len)
    return SimpleNamespace(req=SimpleNamespace(kv=kv), host_staged=host_staged)


def _queue(reqs):
    queue = object.__new__(DecodePreallocQueue)
    queue.transfer_queue = SimpleNamespace(queue=reqs)
    queue.token_to_kv_pool_allocator = SimpleNamespace(page_size=PAGE)
    return queue


class TestPreAllocatedTokenCounts(CustomTestCase):
    def test_counts_page_charged_kv_of_transfer_queue_reqs(self):
        queue = _queue(
            [
                _transfer_req(100),  # 2 pages
                _transfer_req(PAGE * 3, swa_tail_len=PAGE),  # 3 full, 1 swa page
                _transfer_req(5000, host_staged=True),  # host-staged: no device KV
            ]
        )
        self.assertEqual(
            queue.pre_allocated_token_counts(), (5 * PAGE, 2 * PAGE + PAGE)
        )

    def test_empty_transfer_queue_holds_nothing(self):
        self.assertEqual(_queue([]).pre_allocated_token_counts(), (0, 0))


def _reporter(*, counts, pool_stats):
    reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
    reporter.stats = SchedulerStats()
    reporter.scheduler = SimpleNamespace(
        disagg_decode_prealloc_queue=SimpleNamespace(
            queue=[], pre_allocated_token_counts=lambda: counts
        ),
        disagg_decode_transfer_queue=SimpleNamespace(
            queue=[], num_pending_deferred_releases=lambda: 0
        ),
        pool_stats_observer=SimpleNamespace(get_pool_stats=lambda: pool_stats),
    )
    return reporter


def _swa_pool_stats(full_capacity, swa_capacity):
    return PoolStats(
        full_num_used=0,
        full_token_usage=0.0,
        full_available_size=0,
        full_evictable_size=0,
        is_hybrid_swa=True,
        full_capacity=full_capacity,
        swa_capacity=swa_capacity,
    )


class TestDecodeQueueStats(CustomTestCase):
    def _update(self, reporter):
        with mock.patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.get_disagg",
            return_value=SimpleNamespace(
                disaggregation_decode_host_receive_threshold=1
            ),
        ):
            reporter._update_decode_queue_stats(False)

    def test_hybrid_swa_ratios_use_each_pool_capacity(self):
        reporter = _reporter(counts=(250, 40), pool_stats=_swa_pool_stats(1000, 200))
        self._update(reporter)
        self.assertEqual(reporter.stats.prealloc_full_token_usage, 0.25)
        self.assertEqual(reporter.stats.prealloc_swa_token_usage, 0.2)

    def test_plain_pool_reports_no_swa_share(self):
        pool_stats = PoolStats(
            full_num_used=300,
            full_token_usage=0.3,
            full_available_size=500,
            full_evictable_size=200,
            full_capacity=1000,
        )
        reporter = _reporter(counts=(100, 100), pool_stats=pool_stats)
        self._update(reporter)
        self.assertEqual(reporter.stats.prealloc_full_token_usage, 0.1)
        self.assertEqual(reporter.stats.prealloc_swa_token_usage, 0.0)

    def test_zero_capacity_reports_zero_like_token_usage(self):
        reporter = _reporter(counts=(0, 0), pool_stats=_swa_pool_stats(0, 0))
        self._update(reporter)
        self.assertEqual(reporter.stats.prealloc_full_token_usage, 0.0)
        self.assertEqual(reporter.stats.prealloc_swa_token_usage, 0.0)


class TestCollectorExport(CustomTestCase):
    def test_collector_exports_both_gauges(self):
        registry = prometheus_client.CollectorRegistry()
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
            collector = SchedulerMetricsCollector(
                labels={"model_name": "test", "moe_ep_rank": 1}, server_args=None
            )
        stats = SchedulerStats(
            prealloc_full_token_usage=0.25, prealloc_swa_token_usage=0.125
        )
        collector.log_stats(stats)
        samples = {
            sample.name: sample.value
            for family in registry.collect()
            for sample in family.samples
        }
        self.assertEqual(samples["sglang:prealloc_full_token_usage"], 0.25)
        self.assertEqual(samples["sglang:prealloc_swa_token_usage"], 0.125)


if __name__ == "__main__":
    unittest.main()
