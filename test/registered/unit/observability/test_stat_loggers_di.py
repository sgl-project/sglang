"""Pure-CPU unit tests for ``ServerArgs.stat_loggers`` DI plumbing.

These tests cover the small, in-process pieces of the ``stat_loggers``
dependency injection feature:

* The four DI hook class attributes (``_counter_cls``/``_gauge_cls``/
  ``_histogram_cls``/``_summary_cls``) default to ``None`` on every
  collector, so the existing prometheus_client backend is used unchanged.
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

import prometheus_client

from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_EXPERT_DISPATCH,
    STAT_LOGGER_ROLE_RADIX_CACHE,
    STAT_LOGGER_ROLE_SCHEDULER,
    STAT_LOGGER_ROLE_STORAGE,
    STAT_LOGGER_ROLE_TOKENIZER,
    ExpertDispatchCollector,
    RadixCacheMetricsCollector,
    SchedulerMetricsCollector,
    StorageMetricsCollector,
    TokenizerMetricsCollector,
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


class TestCollectorClassAttrs(unittest.TestCase):
    """All five collectors expose four DI hook class attrs, all defaulting to
    None so the existing prometheus_client backend is used unchanged."""

    def test_scheduler_collector_attrs_default_none(self):
        self.assertIsNone(SchedulerMetricsCollector._counter_cls)
        self.assertIsNone(SchedulerMetricsCollector._gauge_cls)
        self.assertIsNone(SchedulerMetricsCollector._histogram_cls)
        self.assertIsNone(SchedulerMetricsCollector._summary_cls)

    def test_tokenizer_collector_attrs_default_none(self):
        self.assertIsNone(TokenizerMetricsCollector._counter_cls)
        self.assertIsNone(TokenizerMetricsCollector._histogram_cls)

    def test_storage_collector_attrs_default_none(self):
        self.assertIsNone(StorageMetricsCollector._counter_cls)
        self.assertIsNone(StorageMetricsCollector._histogram_cls)

    def test_expert_dispatch_collector_attrs_default_none(self):
        self.assertIsNone(ExpertDispatchCollector._histogram_cls)

    def test_radix_cache_collector_attrs_default_none(self):
        self.assertIsNone(RadixCacheMetricsCollector._counter_cls)
        self.assertIsNone(RadixCacheMetricsCollector._histogram_cls)


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

        self.assertEqual(
            collector.storage_prefetch_hit_tokens_total.increments, [(labels, 21)]
        )
        self.assertEqual(
            collector.storage_prefetch_unfulfilled_tokens_total.increments,
            [({**labels, "reason": "storage_transfer"}, 4)],
        )


if __name__ == "__main__":
    unittest.main()
