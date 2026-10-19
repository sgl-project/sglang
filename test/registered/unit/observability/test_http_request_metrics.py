from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

import subprocess
import sys
import unittest

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import sglang.srt.entrypoints.http_server as http_server
from sglang.srt.observability import http_request_metrics
from sglang.test.test_utils import CustomTestCase

UNKNOWN_PATH = "/__no_such_route__"


class TestHttpRequestMetrics(CustomTestCase):
    def test_track_middleware_registered_at_import(self):
        # Tokenizer workers re-import http_server and serve its module-level app.
        dispatch_names = [
            m.kwargs["dispatch"].__name__
            for m in http_server.app.user_middleware
            if "dispatch" in m.kwargs
        ]
        self.assertIn("track_http_status_code", dispatch_names)

    def test_import_does_not_load_prometheus_client(self):
        # PROMETHEUS_MULTIPROC_DIR is set after http_server is imported.
        code = (
            "import sys, sglang.srt.entrypoints.http_server; "
            "print('prometheus_client' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual("False", result.stdout.strip(), result.stderr[-2000:])

    def test_records_requests(self):
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
