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

import prometheus_client

from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_EXPERT_DISPATCH,
    STAT_LOGGER_ROLE_RADIX_CACHE,
    STAT_LOGGER_ROLE_SCHEDULER,
    STAT_LOGGER_ROLE_STORAGE,
    STAT_LOGGER_ROLE_TOKENIZER,
    RadixCacheMetricsCollector,
    SchedulerMetricsCollector,
    StorageMetricsCollector,
    TokenizerMetricsCollector,
    radix_cache_metric_labels,
    resolve_collector_class,
)
from sglang.srt.runtime_context import get_context, reset_context


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


class TestRequestTimePerOutputToken(unittest.TestCase):
    """Request-level TPOT is one observation per finished request (request-
    weighted, unlike the token-weighted ITL histogram). It is only observed
    when the caller could measure a decode period, and it carries the same
    is_streaming split as the e2e latency histogram."""

    def _collector(self, labels):
        with get_context().override_server_args(
            prompt_tokens_buckets=None, generation_tokens_buckets=None
        ):
            return _RecordingTokenizerMetricsCollector(labels=labels)

    def _finish(self, collector, labels, **kwargs):
        collector.observe_one_finished_request(
            labels=labels,
            prompt_tokens=20,
            generation_tokens=101,
            cached_tokens=0,
            e2e_latency=2.5,
            has_grammar=False,
            **kwargs,
        )

    def test_observed_with_streaming_label(self):
        labels = {"model_name": "test"}
        collector = self._collector(labels)

        self._finish(collector, labels, is_streaming=True, time_per_output_token=0.02)
        self._finish(collector, labels, is_streaming=False, time_per_output_token=0.08)

        self.assertEqual(
            collector.histogram_request_time_per_output_token.observations,
            [
                ({**labels, "is_streaming": "true"}, 0.02),
                ({**labels, "is_streaming": "false"}, 0.08),
            ],
        )

    def test_not_observed_when_undefined(self):
        labels = {"model_name": "test"}
        collector = self._collector(labels)

        self._finish(collector, labels, is_streaming=True, time_per_output_token=None)
        self._finish(collector, labels, is_streaming=True)

        self.assertEqual(
            collector.histogram_request_time_per_output_token.observations, []
        )
        # The rest of the finished-request bookkeeping is unaffected.
        self.assertEqual(len(collector.histogram_e2e_request_latency.observations), 2)


if __name__ == "__main__":
    unittest.main()
