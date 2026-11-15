import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from prometheus_client import CollectorRegistry

from sglang.srt.observability.metrics_collector import (
    StorageMetricsCollector,
    _StatLoggerDIMixin,
    format_metric_name,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPrometheusMetricStyle(unittest.TestCase):
    def test_format_metric_name_direct(self):
        # Colon style preserves sglang: prefix
        self.assertEqual(
            format_metric_name("sglang:num_running_reqs", style="colon"),
            "sglang:num_running_reqs",
        )
        # Underscore style converts sglang: prefix to sglang_
        self.assertEqual(
            format_metric_name("sglang:num_running_reqs", style="underscore"),
            "sglang_num_running_reqs",
        )
        # Non-sglang names are preserved
        self.assertEqual(
            format_metric_name("other:metric_name", style="underscore"),
            "other:metric_name",
        )

    def test_format_metric_name_env_default(self):
        with patch.dict(os.environ, {"SGLANG_PROMETHEUS_METRIC_STYLE": "colon"}):
            self.assertEqual(
                format_metric_name("sglang:queue_time_seconds"),
                "sglang:queue_time_seconds",
            )

        with patch.dict(os.environ, {"SGLANG_PROMETHEUS_METRIC_STYLE": "underscore"}):
            self.assertEqual(
                format_metric_name("sglang:queue_time_seconds"),
                "sglang_queue_time_seconds",
            )

    def test_resolve_metric_classes_colon(self):
        mixin = _StatLoggerDIMixin()
        server_args = SimpleNamespace(prometheus_metric_style="colon")
        Counter, Gauge, Histogram, Summary = mixin._resolve_metric_classes(server_args)

        registry = CollectorRegistry()
        c = Counter("sglang:test_counter", "Test counter", registry=registry)
        g = Gauge(
            name="sglang:test_gauge", documentation="Test gauge", registry=registry
        )

        self.assertEqual(c._name, "sglang:test_counter")
        self.assertEqual(g._name, "sglang:test_gauge")

    def test_resolve_metric_classes_underscore(self):
        mixin = _StatLoggerDIMixin()
        server_args = SimpleNamespace(prometheus_metric_style="underscore")
        Counter, Gauge, Histogram, Summary = mixin._resolve_metric_classes(server_args)

        registry = CollectorRegistry()
        c = Counter("sglang:test_counter", "Test counter", registry=registry)
        g = Gauge(
            name="sglang:test_gauge", documentation="Test gauge", registry=registry
        )

        self.assertEqual(c._name, "sglang_test_counter")
        self.assertEqual(g._name, "sglang_test_gauge")

    def test_gauge_histogram_metric_style(self):
        from prometheus_client import Gauge

        for style, expected in (
            ("colon", "sglang:test"),
            ("underscore", "sglang_test"),
        ):
            with self.subTest(style=style):
                registry = CollectorRegistry()
                *_, factory = _StatLoggerDIMixin()._resolve_metric_classes(
                    SimpleNamespace(prometheus_metric_style=style),
                    include_gauge_histogram=True,
                )
                with patch(
                    "prometheus_client.Gauge",
                    side_effect=lambda **kwargs: Gauge(registry=registry, **kwargs),
                ):
                    metric = factory("sglang:test", "Test", [], [1, 2])
                metric.set_by_current_observations({}, [1, 2, 3])
                self.assertEqual(metric._gauge._name, expected)
                self.assertEqual(
                    [sample.value for sample in next(registry.collect()).samples],
                    [1, 1, 1],
                )

    def test_storage_collector_metric_style(self):
        # Test with colon style
        server_args_colon = SimpleNamespace(prometheus_metric_style="colon")
        collector_colon = StorageMetricsCollector(
            labels={}, server_args=server_args_colon
        )
        self.assertTrue(
            collector_colon.prefetched_tokens_total._name.startswith("sglang:")
        )

        # Test with underscore style
        server_args_underscore = SimpleNamespace(prometheus_metric_style="underscore")
        collector_underscore = StorageMetricsCollector(
            labels={}, server_args=server_args_underscore
        )
        self.assertTrue(
            collector_underscore.prefetched_tokens_total._name.startswith("sglang_")
        )

    def test_cli_and_environment_select_one_metric_style(self):
        from prometheus_client import Counter, Gauge

        from sglang.srt.environ import envs
        from sglang.srt.observability.http_request_metrics import _HttpRequestMetrics
        from sglang.srt.runtime_context import get_context
        from sglang.srt.server_args import prepare_server_args

        cases = (
            ("colon", None, "colon"),
            ("underscore", None, "underscore"),
            ("colon", "underscore", "underscore"),
            ("underscore", "colon", "colon"),
        )
        for env_style, cli_style, expected in cases:
            with self.subTest(env=env_style, cli=cli_style), patch.dict(os.environ):
                with envs.SGLANG_PROMETHEUS_METRIC_STYLE.override(env_style):
                    argv = ["--model-path", "dummy"]
                    if cli_style is not None:
                        argv.extend(["--prometheus-metric-style", cli_style])
                    server_args = prepare_server_args(argv)
                    with get_context().override_server_args(
                        prometheus_metric_style=server_args.prometheus_metric_style
                    ):
                        prefix = "sglang_" if expected == "underscore" else "sglang:"
                        counter, *_ = _StatLoggerDIMixin()._resolve_metric_classes(
                            server_args
                        )
                        metric = counter(
                            "sglang:test", "Test", registry=CollectorRegistry()
                        )
                        self.assertEqual(metric._name, prefix + "test")
                        registry = CollectorRegistry()
                        with (
                            patch(
                                "prometheus_client.Counter",
                                side_effect=lambda **kwargs: Counter(
                                    registry=registry, **kwargs
                                ),
                            ),
                            patch(
                                "prometheus_client.Gauge",
                                side_effect=lambda **kwargs: Gauge(
                                    registry=registry, **kwargs
                                ),
                            ),
                        ):
                            http_metrics = _HttpRequestMetrics()
                        self.assertEqual(
                            http_metrics.requests_active._name,
                            prefix + "http_requests_active",
                        )
                        self.assertEqual(
                            format_metric_name("sglang:cpu_percent"),
                            prefix + "cpu_percent",
                        )
                        self.assertEqual(server_args.prometheus_metric_style, expected)

    def test_collectors_follow_published_metric_style(self):
        from sglang.srt.runtime_context import get_context

        with get_context().override_server_args(
            prometheus_metric_style="colon"
        ) as args:
            get_context().override(
                "test metric style", prometheus_metric_style="underscore"
            )
            counter, *_ = _StatLoggerDIMixin()._resolve_metric_classes(args)
            metric = counter("sglang:test", "Test", registry=CollectorRegistry())
            self.assertEqual(metric._name, "sglang_test")
            self.assertEqual(format_metric_name("sglang:test"), "sglang_test")
            self.assertEqual(
                format_metric_name("sglang:test", style="colon"), "sglang:test"
            )

    def test_http_request_metrics_style(self):
        from sglang.srt.observability.http_request_metrics import _HttpRequestMetrics

        with patch.dict(os.environ, {"SGLANG_PROMETHEUS_METRIC_STYLE": "colon"}):
            metrics_colon = _HttpRequestMetrics()
            self.assertEqual(
                metrics_colon.requests_active._name, "sglang:http_requests_active"
            )

        with patch.dict(os.environ, {"SGLANG_PROMETHEUS_METRIC_STYLE": "underscore"}):
            metrics_under = _HttpRequestMetrics()
            self.assertEqual(
                metrics_under.requests_active._name, "sglang_http_requests_active"
            )


if __name__ == "__main__":
    unittest.main()
