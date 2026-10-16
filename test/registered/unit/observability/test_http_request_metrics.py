"""Regression tests for the HTTP request metrics middleware.

With ``--tokenizer-worker-num > 1`` the app is served by worker processes that
uvicorn spawns and that re-import ``http_server``. The track-response
middleware therefore has to be part of the module-level app: registering it
later from the parent process leaves every worker without it, and
``sglang:http_requests_total`` and its siblings disappear from ``/metrics``.

Creating the metrics is deliberately a separate step
(``init_http_request_metrics``): prometheus_client fixes its value backend when
it is first imported, while SGLang only sets ``PROMETHEUS_MULTIPROC_DIR`` after
this module has been imported.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

import subprocess
import sys

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import sglang.srt.entrypoints.http_server as http_server
from sglang.srt.utils import common
from sglang.test.test_utils import CustomTestCase

UNKNOWN_PATH = "/__no_such_route__"
WORKER_ID = "7"


def _dispatch_names(app):
    """Names of the dispatch callables of the app's HTTP middleware entries."""
    names = []
    for middleware in app.user_middleware:
        dispatch = middleware.kwargs.get("dispatch")
        if dispatch is not None:
            names.append(dispatch.__name__)
    return names


class TestHttpRequestMetrics(CustomTestCase):
    def test_track_middleware_registered_at_import(self):
        # It has to be on the module-level app, because that is the app the
        # tokenizer worker processes re-import and serve.
        self.assertIn("track_http_status_code", _dispatch_names(http_server.app))

    def test_import_does_not_load_prometheus_client(self):
        # prometheus_client picks its value backend when it is first imported:
        # with PROMETHEUS_MULTIPROC_DIR unset it keeps values in memory, where
        # the MultiProcessCollector behind /metrics never sees them. SGLang sets
        # that variable only in _set_envs_and_config, which runs after this
        # module has been imported -- so importing it must not import
        # prometheus_client.
        code = (
            "import sys, sglang.srt.entrypoints.http_server; "
            "print('prometheus_client' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual("False", result.stdout.strip(), result.stderr[-2000:])

    def test_records_requests_and_labels_the_worker(self):
        common.init_http_request_metrics(worker_id=WORKER_ID)
        TestClient(http_server.app).get(UNKNOWN_PATH)

        self.assertEqual(
            1.0,
            REGISTRY.get_sample_value(
                "sglang:http_requests_total",
                {"endpoint": UNKNOWN_PATH, "method": "GET"},
            ),
        )
        # Present (net zero after the request) proves the per-worker gauge was
        # incremented and decremented under this worker id.
        self.assertIsNotNone(
            REGISTRY.get_sample_value(
                "sglang:http_worker_active_requests", {"worker_id": WORKER_ID}
            )
        )
