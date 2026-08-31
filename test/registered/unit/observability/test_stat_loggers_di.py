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

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

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


if __name__ == "__main__":
    unittest.main()
