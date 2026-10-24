from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import unittest

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import sglang.srt.entrypoints.http_server as http_server
from sglang.srt.observability import http_request_metrics
from sglang.test.test_utils import CustomTestCase

UNKNOWN_PATH = "/__no_such_route__"


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
