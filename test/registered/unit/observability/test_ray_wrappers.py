"""Unit tests for :mod:`sglang.srt.observability.ray_wrappers`.

The wrapper module is designed to import cleanly even without Ray installed; we
inject a fake ``ray``/``ray.util.metrics``/``ray.serve`` triple into
``sys.modules`` before importing the module so the tests run on the CPU CI
runners that don't ship Ray. The fakes are shared with the DI integration
tests in ``test_stat_loggers_di.py``.
"""

from __future__ import annotations

import sys
import unittest
from functools import partial
from types import SimpleNamespace

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.observability.fake_ray import (
    clear_fake_ray_modules,
    load_ray_wrappers_with_fake_ray,
    load_ray_wrappers_without_ray,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRayWrapperBase(unittest.TestCase):
    def setUp(self) -> None:
        self.rw = load_ray_wrappers_with_fake_ray(replica_id="rep-001")

    def tearDown(self) -> None:
        clear_fake_ray_modules()


class TestNameSanitization(TestRayWrapperBase):
    def test_replaces_colons_with_underscores(self):
        sanitized = self.rw.RayPrometheusMetric._get_sanitized_opentelemetry_name(
            "sglang:num_running_reqs"
        )
        self.assertEqual(sanitized, "sglang_num_running_reqs")

    def test_replaces_all_punctuation(self):
        sanitized = self.rw.RayPrometheusMetric._get_sanitized_opentelemetry_name(
            "sglang:foo.bar-baz/qux"
        )
        self.assertEqual(sanitized, "sglang_foo_bar_baz_qux")

    def test_keeps_already_valid_names(self):
        sanitized = self.rw.RayPrometheusMetric._get_sanitized_opentelemetry_name(
            "sglang_foo_bar"
        )
        self.assertEqual(sanitized, "sglang_foo_bar")


class TestReplicaIdInjection(TestRayWrapperBase):
    def test_tag_keys_include_replica_id(self):
        counter = self.rw.RayCounterWrapper(
            "sglang:requests", "doc", labelnames=["model_name"]
        )
        self.assertEqual(counter.metric._tag_keys, ("model_name", "ReplicaId"))

    def test_emit_uses_replica_id_tag(self):
        counter = self.rw.RayCounterWrapper(
            "sglang:requests", "doc", labelnames=["model_name"]
        )
        counter.labels(model_name="m1").inc(1)
        op, value, tags = counter.metric.calls[-1]
        self.assertEqual(op, "inc")
        self.assertEqual(value, 1)
        self.assertEqual(tags, {"model_name": "m1", "ReplicaId": "rep-001"})


class TestCounterWrapper(TestRayWrapperBase):
    def test_inc_forwards_value_and_tags(self):
        counter = self.rw.RayCounterWrapper("sglang:requests", "doc", labelnames=["m"])
        counter.labels(m="x").inc(5)
        self.assertEqual(
            counter.metric.calls[-1],
            ("inc", 5, {"m": "x", "ReplicaId": "rep-001"}),
        )

    def test_inc_zero_is_noop(self):
        counter = self.rw.RayCounterWrapper("sglang:requests", "doc", labelnames=["m"])
        counter.labels(m="x").inc(0)
        # No call recorded — inc(0) should short-circuit.
        self.assertEqual(counter.metric.calls, [])


class TestGaugeWrapper(TestRayWrapperBase):
    def test_set_forwards_value_and_tags(self):
        gauge = self.rw.RayGaugeWrapper("sglang:running", "doc", labelnames=["m"])
        gauge.labels(m="x").set(12)
        self.assertEqual(
            gauge.metric.calls[-1],
            ("set", 12, {"m": "x", "ReplicaId": "rep-001"}),
        )

    def test_set_to_current_time_uses_set(self):
        gauge = self.rw.RayGaugeWrapper("sglang:start_time", "doc")
        gauge.set_to_current_time()
        op, value, _ = gauge.metric.calls[-1]
        self.assertEqual(op, "set")
        self.assertIsInstance(value, float)
        self.assertGreater(value, 0)

    def test_accepts_multiprocess_mode_for_api_parity(self):
        # multiprocess_mode is irrelevant under Ray; the wrapper must still
        # accept the kwarg so existing call sites in metrics_collector.py work.
        gauge = self.rw.RayGaugeWrapper(
            "sglang:running", "doc", labelnames=["m"], multiprocess_mode="livesum"
        )
        gauge.labels(m="x").set(7)
        self.assertEqual(gauge.metric.calls[-1][0], "set")


class TestHistogramWrapper(TestRayWrapperBase):
    def test_observe_forwards_value_and_tags(self):
        hist = self.rw.RayHistogramWrapper(
            "sglang:ttft_seconds", "doc", labelnames=["m"], buckets=[0.1, 1.0]
        )
        hist.labels(m="x").observe(0.3)
        self.assertEqual(
            hist.metric.calls[-1],
            ("observe", 0.3, {"m": "x", "ReplicaId": "rep-001"}),
        )

    def test_buckets_translate_to_boundaries(self):
        hist = self.rw.RayHistogramWrapper(
            "sglang:ttft_seconds", "doc", buckets=[0.1, 0.5, 1.0, 2.0]
        )
        self.assertEqual(hist.metric.boundaries, [0.1, 0.5, 1.0, 2.0])

    def test_no_buckets_defaults_to_empty_list(self):
        hist = self.rw.RayHistogramWrapper("sglang:ttft_seconds", "doc")
        self.assertEqual(hist.metric.boundaries, [])

    def test_non_positive_boundaries_dropped(self):
        # Ray.util.metrics rejects boundaries <= 0; sglang's queue_time and a
        # few other histograms include 0.0 as their lowest bucket. The wrapper
        # silently filters non-positive entries so engine startup never breaks
        # when the Ray backend is in use.
        hist = self.rw.RayHistogramWrapper(
            "sglang:queue_time_seconds", "doc", buckets=[0.0, 0.001, 1.0]
        )
        self.assertEqual(hist.metric.boundaries, [0.001, 1.0])

    def test_negative_boundaries_dropped(self):
        hist = self.rw.RayHistogramWrapper(
            "sglang:demo", "doc", buckets=[-1.0, 0.0, 0.5]
        )
        self.assertEqual(hist.metric.boundaries, [0.5])


class TestSummaryWrapperFallback(TestRayWrapperBase):
    def test_observe_uses_default_boundaries(self):
        summary = self.rw.RaySummaryWrapper("sglang:request_latency", "doc")
        self.assertEqual(
            summary.metric.boundaries,
            self.rw.RaySummaryWrapper.DEFAULT_BOUNDARIES,
        )

    def test_observe_forwards_value_and_tags(self):
        summary = self.rw.RaySummaryWrapper(
            "sglang:request_latency", "doc", labelnames=["m"]
        )
        summary.labels(m="x").observe(0.42)
        self.assertEqual(
            summary.metric.calls[-1],
            ("observe", 0.42, {"m": "x", "ReplicaId": "rep-001"}),
        )


class TestLabelsCopyAndGuard(TestRayWrapperBase):
    def test_labels_returns_copy_not_self(self):
        counter = self.rw.RayCounterWrapper("sglang:requests", "doc", labelnames=["m"])
        labeled = counter.labels(m="x")
        self.assertIsNot(labeled, counter)

    def test_original_remains_unlabeled_after_labels_call(self):
        counter = self.rw.RayCounterWrapper("sglang:requests", "doc", labelnames=["m"])
        counter.labels(m="x")
        self.assertFalse(counter._is_labeled)

    def test_double_labels_raises(self):
        counter = self.rw.RayCounterWrapper("sglang:requests", "doc", labelnames=["m"])
        labeled = counter.labels(m="x")
        with self.assertRaises(ValueError):
            labeled.labels(m="y")

    def test_concurrent_labels_have_isolated_tags(self):
        counter = self.rw.RayCounterWrapper("sglang:requests", "doc", labelnames=["m"])
        a = counter.labels(m="alpha")
        b = counter.labels(m="beta")
        self.assertEqual(a._tags["m"], "alpha")
        self.assertEqual(b._tags["m"], "beta")

    def test_positional_label_args_supported(self):
        counter = self.rw.RayCounterWrapper(
            "sglang:requests", "doc", labelnames=["model", "engine"]
        )
        counter.labels("m1", "e1").inc(1)
        _, _, tags = counter.metric.calls[-1]
        self.assertEqual(tags["model"], "m1")
        self.assertEqual(tags["engine"], "e1")

    def test_wrong_positional_arity_raises(self):
        counter = self.rw.RayCounterWrapper(
            "sglang:requests", "doc", labelnames=["model", "engine"]
        )
        with self.assertRaises(ValueError):
            counter.labels("only_one_arg")


class TestCollectorSubclassWiring(TestRayWrapperBase):
    """Each Ray collector subclass must override only the ``_xxx_cls`` attrs the
    underlying collector actually uses."""

    def test_scheduler_overrides_all_four(self):
        cls = self.rw.RaySchedulerMetricsCollector
        self.assertIs(cls._counter_cls, self.rw.RayCounterWrapper)
        self.assertIs(cls._gauge_cls, self.rw.RayGaugeWrapper)
        self.assertIs(cls._histogram_cls, self.rw.RayHistogramWrapper)
        self.assertIs(cls._summary_cls, self.rw.RaySummaryWrapper)

    def test_tokenizer_overrides_counter_gauge_histogram(self):
        cls = self.rw.RayTokenizerMetricsCollector
        self.assertIs(cls._counter_cls, self.rw.RayCounterWrapper)
        self.assertIs(cls._gauge_cls, self.rw.RayGaugeWrapper)
        self.assertIs(cls._histogram_cls, self.rw.RayHistogramWrapper)
        self.assertIsNone(cls._summary_cls)

    def test_storage_overrides_counter_histogram_only(self):
        cls = self.rw.RayStorageMetricsCollector
        self.assertIs(cls._counter_cls, self.rw.RayCounterWrapper)
        self.assertIs(cls._histogram_cls, self.rw.RayHistogramWrapper)
        self.assertIsNone(cls._gauge_cls)
        self.assertIsNone(cls._summary_cls)

    def test_radix_overrides_counter_histogram_only(self):
        cls = self.rw.RayRadixCacheMetricsCollector
        self.assertIs(cls._counter_cls, self.rw.RayCounterWrapper)
        self.assertIs(cls._histogram_cls, self.rw.RayHistogramWrapper)
        self.assertIsNone(cls._gauge_cls)
        self.assertIsNone(cls._summary_cls)

    def test_expert_dispatch_overrides_histogram_only(self):
        cls = self.rw.RayExpertDispatchCollector
        self.assertIs(cls._histogram_cls, self.rw.RayHistogramWrapper)
        self.assertIsNone(cls._counter_cls)
        self.assertIsNone(cls._gauge_cls)
        self.assertIsNone(cls._summary_cls)


class TestRayMissingImportError(unittest.TestCase):
    """Importing the module must succeed even without Ray; instantiating a
    wrapper without Ray must raise ImportError with a clear message."""

    def setUp(self) -> None:
        self.rw = load_ray_wrappers_without_ray()

    def tearDown(self) -> None:
        clear_fake_ray_modules()

    def test_module_imports_without_ray(self):
        # If we got here without exception, the import succeeded.
        self.assertTrue(hasattr(self.rw, "RayCounterWrapper"))

    def test_instantiating_wrapper_without_ray_raises(self):
        with self.assertRaises(ImportError) as ctx:
            self.rw.RayCounterWrapper("sglang:foo", "doc")
        self.assertIn("Ray", str(ctx.exception))

    def test_get_replica_id_returns_none_without_ray(self):
        self.assertIsNone(self.rw._get_replica_id())


class TestAsciiDocumentation(TestRayWrapperBase):
    """Ray's metric backend rejects non-ASCII, so a wrapper whose constructor
    skips ``_get_ascii_documentation`` would only crash at deploy time."""

    def test_all_wrappers_fold_non_ascii_description(self):
        for cls in (
            self.rw.RayCounterWrapper,
            self.rw.RayGaugeWrapper,
            self.rw.RayHistogramWrapper,
            self.rw.RaySummaryWrapper,
        ):
            with self.subTest(wrapper=cls.__name__):
                metric = cls("sglang:x", documentation="load — seconds").metric
                self.assertEqual(metric.description, "load - seconds")


class _TokenizerCollectorCase(CustomTestCase):
    _BUCKETS = [0.05, 0.1, 0.5, 1.0]
    _LABELS = {"model_name": "m"}

    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            prompt_tokens_buckets=None,
            generation_tokens_buckets=None,
        )
        self.server_args = override.install()
        self.addCleanup(override.restore)

    def _build_collector(self, *, force_fallback: bool):
        # A private registry per collector avoids duplicate ``sglang:`` names.
        registry = CollectorRegistry()

        class _Collector(TokenizerMetricsCollector):
            _counter_cls = partial(Counter, registry=registry)
            _gauge_cls = partial(Gauge, registry=registry)
            _histogram_cls = partial(Histogram, registry=registry)

        collector = _Collector(
            server_args=self.server_args,
            labels=self._LABELS,
            bucket_time_to_first_token=[0.1, 1.0],
            bucket_inter_token_latency=self._BUCKETS,
            bucket_e2e_request_latency=[0.1, 1.0],
        )
        if not force_fallback:
            # _histogram_cls=None routes observe to the default-backend path.
            collector._histogram_cls = None
        return collector


class TestInterTokenLatencyEquivalence(_TokenizerCollectorCase):
    """``observe_inter_token_latency`` writes histogram internals directly for
    the default backend but replays ``observe()`` for an injected one; both must
    record identical sums and bucket counts, or ITL diverges between the default
    and Ray backends."""

    def test_fast_and_fallback_agree(self):
        fast = self._build_collector(force_fallback=False)
        fallback = self._build_collector(force_fallback=True)

        for collector in (fast, fallback):
            collector.observe_inter_token_latency(self._LABELS, 0.24, 4)
            collector.observe_inter_token_latency(self._LABELS, 6.0, 3)  # +Inf bucket
            collector.observe_inter_token_latency(self._LABELS, 0.3, 2)

        fast_h = fast.histogram_inter_token_latency.labels(**self._LABELS)
        fb_h = fallback.histogram_inter_token_latency.labels(**self._LABELS)
        self.assertEqual(
            [b.get() for b in fast_h._buckets],
            [b.get() for b in fb_h._buckets],
        )
        self.assertAlmostEqual(fast_h._sum.get(), fb_h._sum.get())


class TestTokenizerLatencyAccounting(_TokenizerCollectorCase):
    """``TokenizerManager.collect_metrics`` against the default-backend histogram
    path, where a negative token weight lands in the buckets unchecked."""

    def _run(self, role, completion_tokens, *, finish_type=None):
        collector = self._build_collector(force_fallback=False)
        manager = SimpleNamespace(
            metrics_collector=collector,
            disaggregation_mode=DisaggregationMode(role),
            enable_priority_scheduling=False,
            _request_has_grammar=lambda obj: False,
        )
        state = SimpleNamespace(
            obj=SimpleNamespace(stream=False),
            ttft_observed=False,
            last_completion_tokens=1,
            finished=False,
            time_stats=SimpleNamespace(
                get_first_token_latency=lambda: 0.2,
                get_interval=lambda: 0.2,
                get_e2e_latency=lambda: 1.0,
                set_last_time=lambda: None,
            ),
        )
        for j, count in enumerate(completion_tokens):
            last = j == len(completion_tokens) - 1
            state.finished = last and finish_type is not None
            recv = SimpleNamespace(
                finished_reasons=[{"type": finish_type} if state.finished else None],
                prompt_tokens=[100],
                cached_tokens=[0],
            )
            if count is not None:
                recv.completion_tokens = [count]
            TokenizerManager.collect_metrics(manager, state, recv, 0)
        return collector

    def _itl_samples(self, collector):
        his = collector.histogram_inter_token_latency.labels(**self._LABELS)
        return [b.get() for b in his._buckets] + [his._sum.get()]

    def _ttft_count(self, collector):
        his = collector.histogram_time_to_first_token.labels(
            **self._LABELS, is_streaming="false"
        )
        return sum(b.get() for b in his._buckets)

    def test_ttft_skips_only_aborted_requests(self):
        """Aborts must not add a TTFT sample, but embedding and zero-output
        requests still have a first output and must."""
        cases = [
            ("abort_null", "null", [0], "abort", 0),
            ("abort_decode", "decode", [0], "abort", 0),
            ("embedding", "null", [None], "stop", 1),
            ("zero_output", "null", [0], "length", 1),
        ]
        for name, role, counts, finish_type, expected in cases:
            with self.subTest(name):
                collector = self._run(role, counts, finish_type=finish_type)
                self.assertEqual(self._ttft_count(collector), expected)

    def test_prefill_never_observes_decode_intervals(self):
        """PD prefill's first report (0 tokens) is below the initial baseline of 1;
        the negative delta used to land in the buckets, breaking
        histogram_quantile and rate()."""
        collector = self._run("prefill", [0, 1, 8, 0], finish_type="stop")
        self.assertEqual(self._itl_samples(collector), [0.0] * 6)

    def test_token_count_decrease_resets_baseline(self):
        """A decreasing count must restart the baseline, not subtract tokens."""
        collector = self._run("decode", [8, 0, 0, 2])
        samples = self._itl_samples(collector)
        self.assertTrue(all(v >= 0 for v in samples), samples)
        self.assertEqual(sum(samples[:-1]), 2)
        self.assertAlmostEqual(samples[-1], 0.2)


if __name__ == "__main__":
    sys.exit(unittest.main())
