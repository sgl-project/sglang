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


class _StubArgs:
    """Minimal ServerArgs stand-in.

    Avoids triggering the heavy real ServerArgs import chain for unit-level
    ``resolve_collector_class`` cases.
    """

    def __init__(self, stat_loggers=None):
        self.stat_loggers = stat_loggers


class _TokenizerArgs(_StubArgs):
    prompt_tokens_buckets = None
    generation_tokens_buckets = None


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


class _RecordingGauge:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.calls = []
        self._labels = {}
        type(self).instances.append(self)

    def labels(self, **kwargs):
        self._labels = kwargs
        return self

    def set(self, value):
        self.calls.append(("set", self._labels, value))

    def inc(self, amount=1):
        self.calls.append(("inc", self._labels, amount))


class _RecordingCounter(_RecordingGauge):
    pass


class _RecordingHistogram(_RecordingGauge):
    def observe(self, value):
        self.calls.append(("observe", self._labels, value))


class _RecordingSummary(_RecordingGauge):
    def observe(self, value):
        self.calls.append(("observe", self._labels, value))


class TestDefaultBackend(unittest.TestCase):
    """Collector backend behavior and component cache metric emissions."""

    def setUp(self):
        _RecordingGauge.instances = []
        _RecordingCounter.instances = []
        _RecordingHistogram.instances = []
        _RecordingSummary.instances = []

    def test_default_path_uses_prometheus_client(self):
        labels = {"cache_type": "test_default"}
        collector = RadixCacheMetricsCollector(labels=labels)
        self.assertIsInstance(collector.eviction_num_tokens, prometheus_client.Counter)
        self.assertIsInstance(
            collector.eviction_duration_seconds, prometheus_client.Histogram
        )

    def test_tokenizer_cached_tokens_by_component_metric_reports_sources(self):
        class RaySwapTokenizer(TokenizerMetricsCollector):
            _counter_cls = _RecordingCounter
            _histogram_cls = _RecordingHistogram

        collector = RaySwapTokenizer(server_args=_TokenizerArgs(), labels={})
        collector.observe_one_finished_request(
            labels={},
            prompt_tokens=10,
            generation_tokens=1,
            cached_tokens=10,
            e2e_latency=0.1,
            has_grammar=False,
            cached_tokens_details={
                "device": 6,
                "host": 0,
                "storage": 4,
                "storage_backend": "MooncakeStore",
            },
            cached_tokens_by_component={
                "full": {"device": 6, "storage": 4},
                "swa": {"device": 3, "host": 2},
            },
        )

        component_counter = next(
            c
            for c in _RecordingCounter.instances
            if c.kwargs["name"] == "sglang:cached_tokens_by_component_total"
        )
        self.assertEqual(
            component_counter.calls,
            [
                (
                    "inc",
                    {"component": "full", "cache_source": "device"},
                    6,
                ),
                (
                    "inc",
                    {"component": "full", "cache_source": "storage_MooncakeStore"},
                    4,
                ),
                ("inc", {"component": "swa", "cache_source": "device"}, 3),
                ("inc", {"component": "swa", "cache_source": "host"}, 2),
            ],
        )
        cached_tokens_counter = next(
            c
            for c in _RecordingCounter.instances
            if c.kwargs["name"] == "sglang:cached_tokens_total"
        )
        full_component_calls = [
            (method, {"cache_source": labels["cache_source"]}, value)
            for method, labels, value in component_counter.calls
            if labels["component"] == "full"
        ]
        self.assertEqual(full_component_calls, cached_tokens_counter.calls)

    def test_tokenizer_cached_tokens_by_component_metric_skips_decode_engine(self):
        class RaySwapTokenizer(TokenizerMetricsCollector):
            _counter_cls = _RecordingCounter
            _histogram_cls = _RecordingHistogram

        collector = RaySwapTokenizer(
            server_args=_TokenizerArgs(), labels={"engine_type": ""}
        )
        collector.observe_one_finished_request(
            labels={"engine_type": "decode"},
            prompt_tokens=10,
            generation_tokens=1,
            cached_tokens=10,
            e2e_latency=0.1,
            has_grammar=False,
            cached_tokens_details={"storage_backend": "MooncakeStore"},
            cached_tokens_by_component={
                "full": {"device": 6, "storage": 4},
            },
        )

        component_counter = next(
            c
            for c in _RecordingCounter.instances
            if c.kwargs["name"] == "sglang:cached_tokens_by_component_total"
        )
        self.assertEqual(component_counter.calls, [])


if __name__ == "__main__":
    unittest.main()
