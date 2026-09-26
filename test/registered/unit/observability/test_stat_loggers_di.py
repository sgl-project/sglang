"""Pure-CPU unit tests for ``ServerArgs.stat_loggers`` DI plumbing.

These tests cover the small, in-process pieces of the ``stat_loggers``
dependency injection feature:

* ``resolve_collector_class()`` returns the registered subclass when a role
  is present in ``stat_loggers`` and falls back to the default otherwise.
* Without any subclass override, collectors instantiate the real
  prometheus_client classes.

The full Engine-level integration test (which boots ``sgl.Engine`` and
verifies that emissions land on a FakeRayMetric-style recording double in
the scheduler subprocess) lives in
``test/registered/observability/test_metrics.py`` alongside the other
GPU-backed metrics tests.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import prometheus_client

from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_EXPERT_DISPATCH,
    STAT_LOGGER_ROLE_RADIX_CACHE,
    STAT_LOGGER_ROLE_SCHEDULER,
    STAT_LOGGER_ROLE_STORAGE,
    STAT_LOGGER_ROLE_TOKENIZER,
    RadixCacheMetricsCollector,
    SchedulerMetricsCollector,
    SchedulerStats,
    StorageMetricsCollector,
    TokenizerMetricsCollector,
    radix_cache_metric_labels,
    resolve_collector_class,
)
from sglang.srt.runtime_context import get_context, reset_context
from sglang.test.test_utils import CustomTestCase


class _BoundRecordingMetric:
    def __init__(self, metric, labels):
        self.metric = metric
        self.labels = labels

    def inc(self, value=1):
        self.metric.increments.append((self.labels, value))

    def observe(self, value):
        self.metric.observations.append((self.labels, value))

    def set(self, value):
        self.metric.sets.append((self.labels, value))


class _RecordingMetric:
    """Small prometheus_client-compatible metric that preserves labels."""

    def __init__(self, *args, name=None, labelnames=(), **kwargs):
        self.name = name if name is not None else args[0]
        self.labelnames = tuple(labelnames)
        self.increments = []
        self.observations = []
        self.sets = []

    def labels(self, *values, **labels):
        if values:
            labels = dict(zip(self.labelnames, values, strict=True))
        return _BoundRecordingMetric(self, labels)


class _RecordingTokenizerMetricsCollector(TokenizerMetricsCollector):
    _counter_cls = _RecordingMetric
    _gauge_cls = _RecordingMetric
    _histogram_cls = _RecordingMetric


class _RecordingStorageMetricsCollector(StorageMetricsCollector):
    _counter_cls = _RecordingMetric
    _histogram_cls = _RecordingMetric


class TestResolveCollectorClass(unittest.TestCase):
    """The role table is read from the published `observability` bag."""

    def _resolve(self, role, default_cls, **fields):
        if not fields:
            return resolve_collector_class(role, default_cls)
        with get_context().override_server_args(**fields):
            return resolve_collector_class(role, default_cls)

    def test_returns_default_when_nothing_is_published(self):
        reset_context()
        self.assertIs(
            resolve_collector_class("scheduler", SchedulerMetricsCollector),
            SchedulerMetricsCollector,
        )

    def test_returns_default_when_stat_loggers_none(self):
        cls = self._resolve("scheduler", SchedulerMetricsCollector, stat_loggers=None)
        self.assertIs(cls, SchedulerMetricsCollector)

    def test_returns_default_when_stat_loggers_empty(self):
        cls = self._resolve("scheduler", SchedulerMetricsCollector, stat_loggers={})
        self.assertIs(cls, SchedulerMetricsCollector)

    def test_returns_default_when_role_missing(self):
        class MyTokenizer(TokenizerMetricsCollector):
            pass

        cls = self._resolve(
            "scheduler",
            SchedulerMetricsCollector,
            stat_loggers={"tokenizer": MyTokenizer},
        )
        self.assertIs(cls, SchedulerMetricsCollector)

    def test_returns_subclass_when_role_registered(self):
        class MyScheduler(SchedulerMetricsCollector):
            pass

        cls = self._resolve(
            "scheduler",
            SchedulerMetricsCollector,
            stat_loggers={"scheduler": MyScheduler},
        )
        self.assertIs(cls, MyScheduler)

    def test_role_constants_match_collector_keys(self):
        """The exported role constants must be the exact strings the
        instantiation sites use to look up subclasses."""
        self.assertEqual(STAT_LOGGER_ROLE_SCHEDULER, "scheduler")
        self.assertEqual(STAT_LOGGER_ROLE_TOKENIZER, "tokenizer")
        self.assertEqual(STAT_LOGGER_ROLE_STORAGE, "storage")
        self.assertEqual(STAT_LOGGER_ROLE_RADIX_CACHE, "radix_cache")
        self.assertEqual(STAT_LOGGER_ROLE_EXPERT_DISPATCH, "expert_dispatch")


class TestDefaultBackend(unittest.TestCase):
    """Without any subclass override, collectors instantiate the real
    prometheus_client classes; the existing behavior is unchanged."""

    def test_default_path_uses_prometheus_client(self):
        labels = {"cache_type": "test_default"}
        collector = RadixCacheMetricsCollector(labels=labels)
        self.assertIsInstance(collector.eviction_num_tokens, prometheus_client.Counter)
        self.assertIsInstance(
            collector.eviction_duration_seconds, prometheus_client.Histogram
        )


class TestRadixCacheMetricLabels(unittest.TestCase):
    """Radix-cache series must stay distinct per scheduler rank: an unlabeled
    family is summed across local ranks by the multiprocess registry, which
    reported TP x the logical token count in production. The rank keys follow
    the storage collector's DP-aware convention so L2 and L3 series line up."""

    def test_labels_follow_the_storage_collector_rank_keys(self):
        parallel = SimpleNamespace(tp_rank=3, pp_rank=1, attn_tp_rank=1, attn_dp_rank=2)
        self.assertEqual(
            radix_cache_metric_labels("UnifiedRadixCache", parallel, True),
            {
                "cache_type": "UnifiedRadixCache",
                "tp_rank": 1,
                "pp_rank": 1,
                "dp_rank": 2,
            },
        )
        self.assertEqual(
            radix_cache_metric_labels("RadixCache", parallel, False),
            {"cache_type": "RadixCache", "tp_rank": 3, "pp_rank": 1, "dp_rank": 0},
        )


class TestHiCacheMetrics(unittest.TestCase):
    def test_cached_tokens_uses_literal_storage_source(self):
        labels = {"model_name": "test"}
        with get_context().override_server_args(
            prompt_tokens_buckets=None, generation_tokens_buckets=None
        ):
            collector = _RecordingTokenizerMetricsCollector(labels=labels)

        collector.observe_one_finished_request(
            labels=labels,
            prompt_tokens=20,
            generation_tokens=2,
            cached_tokens=12,
            e2e_latency=0.1,
            has_grammar=False,
            cached_tokens_details={
                "device": 3,
                "host": 4,
                "storage": 5,
                "storage_backend": "BackendShim",
            },
        )

        by_source = {
            metric_labels["cache_source"]: value
            for metric_labels, value in collector.cached_tokens_total.increments
        }
        self.assertEqual(by_source, {"device": 3, "host": 4, "storage": 5})

    def test_storage_prefetch_lifecycle_metrics(self):
        labels = {"model_name": "test"}
        collector = _RecordingStorageMetricsCollector(labels=labels)

        collector.log_storage_prefetch_hit_tokens(21)
        collector.log_storage_prefetch_unfulfilled_tokens(4, "storage_transfer")
        collector.log_storage_prefetch_deferred_tokens(7, "device_capacity")

        self.assertEqual(
            collector.storage_prefetch_hit_tokens_total.increments, [(labels, 21)]
        )
        self.assertEqual(
            collector.storage_prefetch_unfulfilled_tokens_total.increments,
            [({**labels, "reason": "storage_transfer"}, 4)],
        )
        self.assertEqual(
            collector.storage_prefetch_deferred_tokens_total.increments,
            [({**labels, "reason": "device_capacity"}, 7)],
        )


class TestDeferredKVReleaseMetrics(CustomTestCase):
    def setUp(self):
        self.labels = {
            "model_name": "test",
            "engine_type": "decode",
            "tp_rank": 0,
            "pp_rank": 0,
            "moe_ep_rank": 0,
        }
        self.collector = SchedulerMetricsCollector.__new__(SchedulerMetricsCollector)
        self.collector.labels = self.labels
        self.collector.decode_deferred_kv_release_seconds = _RecordingMetric(
            name="sglang:decode_deferred_kv_release_seconds",
            labelnames=self.labels.keys(),
        )
        self.collector.decode_deferred_kv_release_total = _RecordingMetric(
            name="sglang:decode_deferred_kv_release_total",
            labelnames=list(self.labels.keys()) + ["outcome"],
        )
        self.collector.num_decode_deferred_kv_release_reqs = _RecordingMetric(
            name="sglang:num_decode_deferred_kv_release_reqs",
            labelnames=self.labels.keys(),
        )

    def test_records_duration_and_bounded_outcome(self):
        self.collector.observe_decode_deferred_kv_release(
            duration_seconds=0.25,
            outcome="drained",
        )

        self.assertEqual(
            self.collector.decode_deferred_kv_release_seconds.observations,
            [(self.labels, 0.25)],
        )
        self.assertEqual(
            self.collector.decode_deferred_kv_release_total.increments,
            [({**self.labels, "outcome": "drained"}, 1)],
        )
        with self.assertRaisesRegex(ValueError, "Invalid deferred KV release outcome"):
            self.collector.observe_decode_deferred_kv_release(
                duration_seconds=0.5,
                outcome="room-123",
            )

    def test_logs_current_deferred_release_count(self):
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = []
        queue._deferred_releases = [object(), object(), object()]
        reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
        reporter.scheduler = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.DECODE,
            disagg_decode_transfer_queue=queue,
            disagg_decode_prealloc_queue=SimpleNamespace(queue=[]),
            running_batch=SimpleNamespace(reqs=[]),
            waiting_queue=[],
            grammar_manager=[],
            enable_priority_scheduling=False,
            pool_stats_observer=SimpleNamespace(
                get_pool_stats=lambda: SimpleNamespace(
                    update_scheduler_stats=lambda stats: None,
                ),
                streaming_session_count=lambda: 0,
                session_held_tokens=lambda: 0,
            ),
        )
        reporter.stats = SchedulerStats()
        reporter.current_scheduler_metrics_enabled = True
        collector = MagicMock()
        collector.last_log_time = 0
        collector.num_decode_deferred_kv_release_reqs = (
            self.collector.num_decode_deferred_kv_release_reqs
        )
        collector.labels = self.labels
        collector.log_stats.side_effect = lambda stats: (
            SchedulerMetricsCollector.log_stats(collector, stats)
        )
        collector._log_gauge.side_effect = lambda gauge, data: (
            SchedulerMetricsCollector._log_gauge(collector, gauge, data)
        )
        reporter.metrics_collector = collector

        with (
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.ENABLE_METRICS_DEVICE_TIMER",
                False,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_decode_host_receive_threshold=0
                ),
            ),
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.time.perf_counter",
                side_effect=[31.0, 31.0, 62.0, 62.0],
            ),
        ):
            reporter._maybe_log_idle_metrics()
            self.assertEqual(reporter.stats.num_decode_deferred_kv_release_reqs, 3)
            queue._deferred_releases.clear()
            reporter._maybe_log_idle_metrics()
            self.assertEqual(reporter.stats.num_decode_deferred_kv_release_reqs, 0)

        self.assertEqual(
            self.collector.num_decode_deferred_kv_release_reqs.sets,
            [(self.labels, 3), (self.labels, 0)],
        )


if __name__ == "__main__":
    unittest.main()
