"""Unit tests for class-level DI on the five *MetricsCollector classes via
ServerArgs.stat_loggers — no server, no model loading."""

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
from sglang.test.observability.fake_ray import (
    clear_fake_ray_modules,
    load_ray_wrappers_with_fake_ray,
)


class _StubArgs:
    """Minimal ServerArgs stand-in. Avoids triggering heavy ServerArgs import chain."""

    def __init__(self, stat_loggers=None):
        self.stat_loggers = stat_loggers


# ── _gauge_cls / _counter_cls / _histogram_cls / _summary_cls override surface ──


class TestCollectorClassAttrs(unittest.TestCase):
    """All five collectors expose four DI hook class attrs, all defaulting to None
    so the existing prometheus_client backend is used unchanged."""

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


# ── resolve_collector_class helper ──


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
        # Different role registered. Default still wins for "scheduler".
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


# ── DI swap behavior — actually instantiate with a custom backend ──


class _RecordingGauge:
    """Test double that mirrors prometheus_client.Gauge constructor signature.
    Records every instantiation so the test can assert the override took effect."""

    instances = []

    def __init__(self, *args, **kwargs):
        type(self).instances.append((args, kwargs))

    def labels(self, **kwargs):
        return self

    def set(self, value):
        pass

    def inc(self, amount=1):
        pass


class _RecordingCounter(_RecordingGauge):
    pass


class _RecordingHistogram(_RecordingGauge):
    def observe(self, value):
        pass


class _RecordingSummary(_RecordingGauge):
    def observe(self, value):
        pass


class TestDISwap(unittest.TestCase):
    """Subclasses that set the DI hooks at class level cause the collector to
    instantiate the test doubles instead of prometheus_client classes."""

    def setUp(self):
        _RecordingGauge.instances = []
        _RecordingCounter.instances = []
        _RecordingHistogram.instances = []
        _RecordingSummary.instances = []

    def test_radix_cache_di_swap(self):
        """Smallest collector (4 metrics, Counter + Histogram) — verifies the
        DI shim flows through both class types."""

        class RaySwapRadixCache(RadixCacheMetricsCollector):
            _counter_cls = _RecordingCounter
            _histogram_cls = _RecordingHistogram

        labels = {"cache_type": "test"}
        RaySwapRadixCache(labels=labels)

        # 4 instruments total in RadixCacheMetricsCollector:
        # eviction_duration_seconds (H), eviction_num_tokens (C),
        # load_back_duration_seconds (H), load_back_num_tokens (C).
        self.assertEqual(len(_RecordingCounter.instances), 2)
        self.assertEqual(len(_RecordingHistogram.instances), 2)

    def test_expert_dispatch_di_swap(self):
        """Smallest collector (1 Histogram metric)."""

        class RaySwapExpert(ExpertDispatchCollector):
            _histogram_cls = _RecordingHistogram

        RaySwapExpert(ep_size=4)
        self.assertEqual(len(_RecordingHistogram.instances), 1)

    def test_default_path_uses_prometheus_client(self):
        """Without any subclass override, the collector instantiates the real
        prometheus_client classes — the existing behavior is unchanged."""
        labels = {"cache_type": "test_default"}
        collector = RadixCacheMetricsCollector(labels=labels)
        # The instruments must be real prometheus_client objects, not test doubles.
        self.assertIsInstance(collector.eviction_num_tokens, prometheus_client.Counter)
        self.assertIsInstance(
            collector.eviction_duration_seconds, prometheus_client.Histogram
        )


# ── End-to-end DI emission flow with the Ray-backed wrappers ──
#
# The TestDISwap cases above prove the collector swap took effect by counting
# instantiations. These cases go one layer deeper: they construct a Ray-backed
# collector subclass through ``resolve_collector_class``, drive an actual
# ``inc``/``set``/``observe`` call, and verify the value and tags landed on
# the underlying (fake) Ray metric instance. This exercises the full chain:
#
#     stat_loggers → resolve_collector_class → Ray-backed collector
#     → RayCounterWrapper / RayGaugeWrapper / RayHistogramWrapper
#     → FakeRayMetric.calls
#
# The fakes used here are the same ones the wrapper unit tests use, kept in
# ``sglang.test.observability.fake_ray`` so both suites stay in sync.


class TestDIEmissionFlow(unittest.TestCase):
    """DI swap propagates all the way down to metric emission."""

    REPLICA_ID = "rep-di-emit"

    def setUp(self) -> None:
        self.rw = load_ray_wrappers_with_fake_ray(replica_id=self.REPLICA_ID)

    def tearDown(self) -> None:
        clear_fake_ray_modules()

    def test_radix_cache_counter_emit_reaches_fake_ray_metric(self):
        """Counter path: stat_loggers selects the Ray-backed RadixCache
        collector; an ``inc`` call lands on the FakeRayMetric with the
        ReplicaId tag injected by the wrapper."""

        cls = resolve_collector_class(
            _StubArgs(
                stat_loggers={
                    STAT_LOGGER_ROLE_RADIX_CACHE: self.rw.RayRadixCacheMetricsCollector
                }
            ),
            STAT_LOGGER_ROLE_RADIX_CACHE,
            RadixCacheMetricsCollector,
        )
        self.assertIs(cls, self.rw.RayRadixCacheMetricsCollector)

        labels = {"cache_type": "test_emit"}
        collector = cls(labels=labels)
        collector.eviction_num_tokens.labels(**labels).inc(42)

        op, value, tags = collector.eviction_num_tokens.metric.calls[-1]
        self.assertEqual(op, "inc")
        self.assertEqual(value, 42)
        self.assertEqual(tags["cache_type"], "test_emit")
        self.assertEqual(tags["ReplicaId"], self.REPLICA_ID)

    def test_radix_cache_histogram_emit_reaches_fake_ray_metric(self):
        """Histogram path: same flow as above but with ``observe``."""

        cls = resolve_collector_class(
            _StubArgs(
                stat_loggers={
                    STAT_LOGGER_ROLE_RADIX_CACHE: self.rw.RayRadixCacheMetricsCollector
                }
            ),
            STAT_LOGGER_ROLE_RADIX_CACHE,
            RadixCacheMetricsCollector,
        )
        labels = {"cache_type": "test_observe"}
        collector = cls(labels=labels)
        collector.eviction_duration_seconds.labels(**labels).observe(0.25)

        op, value, tags = collector.eviction_duration_seconds.metric.calls[-1]
        self.assertEqual(op, "observe")
        self.assertEqual(value, 0.25)
        self.assertEqual(tags["cache_type"], "test_observe")
        self.assertEqual(tags["ReplicaId"], self.REPLICA_ID)

    def test_expert_dispatch_emit_reaches_fake_ray_metric(self):
        """Smallest collector (a single Histogram) — proves the DI path works
        when only ``_histogram_cls`` is overridden, with no other class hooks."""

        cls = resolve_collector_class(
            _StubArgs(
                stat_loggers={
                    STAT_LOGGER_ROLE_EXPERT_DISPATCH: self.rw.RayExpertDispatchCollector
                }
            ),
            STAT_LOGGER_ROLE_EXPERT_DISPATCH,
            ExpertDispatchCollector,
        )
        collector = cls(ep_size=4)
        collector.eplb_gpu_physical_count.labels(layer="layer0").observe(2)

        op, value, tags = collector.eplb_gpu_physical_count.metric.calls[-1]
        self.assertEqual(op, "observe")
        self.assertEqual(value, 2)
        self.assertEqual(tags["layer"], "layer0")
        self.assertEqual(tags["ReplicaId"], self.REPLICA_ID)


if __name__ == "__main__":
    unittest.main()
