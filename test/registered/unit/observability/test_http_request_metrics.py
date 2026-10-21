from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import unittest
from unittest import mock

from fastapi import FastAPI, Response
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY, make_asgi_app

import sglang.srt.entrypoints.http_server as http_server
from sglang.srt.observability import http_request_metrics
from sglang.test.test_utils import CustomTestCase

UNKNOWN_PATH = "/__no_such_route__"


class TestHttpMetricsSkipProbes(CustomTestCase):
    def test_probes_and_scrapes_are_not_counted(self):
        app = FastAPI()

        @app.get("/ready")
        async def ready():
            return Response(status_code=503)  # a pod that is still loading

        @app.get("/health")
        async def health():
            return Response(status_code=200)

        @app.post("/v1/chat/completions")
        async def chat():
            return {"ok": True}

        app.mount("/metrics", make_asgi_app())
        http_request_metrics.add_http_request_metrics_middleware(app)

        # A mock keeps this test off the process-wide metrics, which
        # TestHttpRequestMetrics expects to be uninitialized.
        metrics = mock.MagicMock()
        with mock.patch.object(http_request_metrics, "_http_request_metrics", metrics):
            client = TestClient(app)
            self.assertEqual(client.get("/ready").status_code, 503)
            client.get("/health")
            client.get("/metrics")
            client.post("/v1/chat/completions")

        for series in ("requests_total", "requests_active", "responses_total"):
            endpoints = {
                call.kwargs["endpoint"]
                for call in getattr(metrics, series).labels.call_args_list
            }
            self.assertEqual(endpoints, {"/v1/chat/completions"}, series)

    def test_excluded_paths_are_served_routes(self):
        # /metrics is mounted at server startup, not on import.
        served = {getattr(route, "path", None) for route in http_server.app.routes}
        self.assertLessEqual(
            http_request_metrics.HTTP_METRICS_EXCLUDED_PATHS - {"/metrics"}, served
        )


class TestHttpRequestMetrics(CustomTestCase):
    def test_module_level_app_records_requests(self):
        # Tokenizer workers re-import http_server and serve its module-level app,
        # so the middleware must be on it, while the metrics must wait for
        # init_http_request_metrics().
        self.assertIsNone(http_request_metrics._http_request_metrics)

        http_request_metrics.init_http_request_metrics()
        TestClient(http_server.app).get(UNKNOWN_PATH)

        self.assertEqual(
            1.0,
            REGISTRY.get_sample_value(
                "sglang:http_requests_total",
                {"endpoint": UNKNOWN_PATH, "method": "GET"},
            ),
        )


if __name__ == "__main__":
    unittest.main()
