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
        g = Gauge(name="sglang:test_gauge", documentation="Test gauge", registry=registry)

        self.assertEqual(c._name, "sglang:test_counter")
        self.assertEqual(g._name, "sglang:test_gauge")

    def test_resolve_metric_classes_underscore(self):
        mixin = _StatLoggerDIMixin()
        server_args = SimpleNamespace(prometheus_metric_style="underscore")
        Counter, Gauge, Histogram, Summary = mixin._resolve_metric_classes(server_args)

        registry = CollectorRegistry()
        c = Counter("sglang:test_counter", "Test counter", registry=registry)
        g = Gauge(name="sglang:test_gauge", documentation="Test gauge", registry=registry)

        self.assertEqual(c._name, "sglang_test_counter")
        self.assertEqual(g._name, "sglang_test_gauge")

    def test_storage_collector_metric_style(self):
        # Test with colon style
        server_args_colon = SimpleNamespace(prometheus_metric_style="colon")
        collector_colon = StorageMetricsCollector(labels={}, server_args=server_args_colon)
        self.assertTrue(collector_colon.prefetched_tokens_total._name.startswith("sglang:"))

        # Test with underscore style
        server_args_underscore = SimpleNamespace(prometheus_metric_style="underscore")
        collector_underscore = StorageMetricsCollector(labels={}, server_args=server_args_underscore)
        self.assertTrue(collector_underscore.prefetched_tokens_total._name.startswith("sglang_"))

    def test_http_request_metrics_style(self):
        from sglang.srt.observability.http_request_metrics import _HttpRequestMetrics

        with patch.dict(os.environ, {"SGLANG_PROMETHEUS_METRIC_STYLE": "colon"}):
            metrics_colon = _HttpRequestMetrics()
            self.assertEqual(metrics_colon.requests_active._name, "sglang:http_requests_active")

        with patch.dict(os.environ, {"SGLANG_PROMETHEUS_METRIC_STYLE": "underscore"}):
            metrics_under = _HttpRequestMetrics()
            self.assertEqual(metrics_under.requests_active._name, "sglang_http_requests_active")


if __name__ == "__main__":
    unittest.main()
