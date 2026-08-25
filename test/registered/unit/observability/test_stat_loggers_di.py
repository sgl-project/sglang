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

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from unittest import mock

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
    StorageMetrics,
    StorageMetricsCollector,
    TokenizerMetricsCollector,
    resolve_collector_class,
)


class _StubArgs:
    """Minimal ServerArgs stand-in.

    Avoids triggering the heavy real ServerArgs import chain for unit-level
    ``resolve_collector_class`` cases.
    """

    def __init__(self, stat_loggers=None):
        self.stat_loggers = stat_loggers


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
    def test_returns_default_when_server_args_none(self):
        cls = resolve_collector_class(None, "scheduler", SchedulerMetricsCollector)
        self.assertIs(cls, SchedulerMetricsCollector)

    def test_returns_default_when_stat_loggers_none(self):
        cls = resolve_collector_class(
            _StubArgs(stat_loggers=None), "scheduler", SchedulerMetricsCollector
        )
        self.assertIs(cls, SchedulerMetricsCollector)

    def test_returns_default_when_stat_loggers_empty(self):
        cls = resolve_collector_class(
            _StubArgs(stat_loggers={}), "scheduler", SchedulerMetricsCollector
        )
        self.assertIs(cls, SchedulerMetricsCollector)

    def test_returns_default_when_role_missing(self):
        class MyTokenizer(TokenizerMetricsCollector):
            pass

        cls = resolve_collector_class(
            _StubArgs(stat_loggers={"tokenizer": MyTokenizer}),
            "scheduler",
            SchedulerMetricsCollector,
        )
        self.assertIs(cls, SchedulerMetricsCollector)

    def test_returns_subclass_when_role_registered(self):
        class MyScheduler(SchedulerMetricsCollector):
            pass

        cls = resolve_collector_class(
            _StubArgs(stat_loggers={"scheduler": MyScheduler}),
            "scheduler",
            SchedulerMetricsCollector,
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



class _FakeMetricChild:
    def __init__(self, metric, label_values):
        self.metric = metric
        self.label_values = label_values

    def inc(self, value=1):
        self.metric.values[self.label_values] = (
            self.metric.values.get(self.label_values, 0) + value
        )

    def observe(self, value):
        self.inc(value)

    def set(self, value):
        self.metric.values[self.label_values] = value


class _FakeMetric:
    instances = []

    def __init__(self, name, documentation, labelnames=(), **kwargs):
        self.name = name
        self.documentation = documentation
        self.labelnames = tuple(labelnames)
        self.values = {}
        self.__class__.instances.append(self)

    def labels(self, **labels):
        return _FakeMetricChild(self, tuple(sorted(labels.items())))


class TestStoragePrefetchOutcomeMetrics(unittest.TestCase):
    def setUp(self):
        _FakeMetric.instances = []
        self.patches = [
            mock.patch.object(StorageMetricsCollector, "_counter_cls", _FakeMetric),
            mock.patch.object(StorageMetricsCollector, "_gauge_cls", _FakeMetric),
            mock.patch.object(StorageMetricsCollector, "_histogram_cls", _FakeMetric),
        ]
        for patcher in self.patches:`n            patcher.start()`n            self.addCleanup(patcher.stop)
        self.collector = StorageMetricsCollector(labels={"model": "test"})
        self.counter = next(
            metric
            for metric in _FakeMetric.instances
            if metric.name == "sglang:hicache_prefetch_outcomes_total"
        )

    def _value(self, outcome):
        return self.counter.values.get((("model", "test"), ("outcome", outcome)), 0)

    def test_exports_deltas_without_duplicate_flushes_or_counter_regression(self):
        first = {"attempts": 2, "issued": 1, "revoked_full_miss": 1}
        self.collector.log_storage_metrics(StorageMetrics(prefetch_stats=first))
        self.collector.log_storage_metrics(StorageMetrics(prefetch_stats=first))
        self.assertEqual(self._value("attempts"), 2)
        self.assertEqual(self._value("issued"), 1)
        self.assertEqual(self._value("revoked_full_miss"), 1)

        self.collector.log_storage_metrics(
            StorageMetrics(
                prefetch_stats={
                    "attempts": 3,
                    "issued": 2,
                    "declined_rate_limited": 1,
                    "revoked_full_miss": 1,
                }
            )
        )
        self.assertEqual(self._value("attempts"), 3)
        self.assertEqual(self._value("issued"), 2)
        self.assertEqual(self._value("declined_rate_limited"), 1)

        self.collector.log_storage_metrics(
            StorageMetrics(prefetch_stats={"attempts": 1, "issued": 1})
        )
        self.assertEqual(self._value("attempts"), 4)
        self.assertEqual(self._value("issued"), 3)
if __name__ == "__main__":
    unittest.main()
