import ast
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from prometheus_client import Counter, values

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SOURCE = (
    Path(__file__).parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "entrypoints"
    / "grpc_server.py"
)


def _load_add_metrics_routes():
    tree = ast.parse(_SOURCE.read_text())
    route = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_add_metrics_routes"
    )
    namespace = {"web": web, "logger": logging.getLogger(__name__)}
    exec(
        compile(ast.Module(body=[route], type_ignores=[]), _SOURCE, "exec"),
        namespace,
    )
    return namespace["_add_metrics_routes"]


class TestGrpcMetrics(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_prometheus_metric_names(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": directory}),
        ):
            previous_value_class = values.ValueClass
            values.ValueClass = values.MultiProcessValue(lambda: os.getpid())
            try:
                Counter("sglang:test_requests", "Test requests", registry=None).inc(2)
            finally:
                values.ValueClass = previous_value_class

            app = web.Application()
            _load_add_metrics_routes()(app)
            async with TestClient(TestServer(app)) as client:
                response = await client.get("/metrics")
                body = await response.text()

            self.assertEqual(response.status, 200)
            self.assertTrue(response.headers["Content-Type"].startswith("text/plain;"))
            self.assertIn("sglang:test_requests_total 2.0", body)
            self.assertNotIn("sglang_test", body)


if __name__ == "__main__":
    unittest.main()
