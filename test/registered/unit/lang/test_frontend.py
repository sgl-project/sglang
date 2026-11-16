"""Regression tests for the sglang.lang frontend, run against stub backends."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import sglang as sgl
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _SglangServerHandler(BaseHTTPRequestHandler):
    """Serves the routes an SGLang server has; every other route is a 404."""

    def _reply(self, status, payload=None):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload or {}).encode())

    def do_GET(self):  # noqa: N802
        if self.path == "/get_model_info":
            self._reply(200, {"model_path": "stub-model"})
        else:
            self._reply(404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else None
        self.server.posts.append((self.path, body))
        if self.path == "/generate":
            self._reply(200, {"text": "", "meta_info": {}})
        else:
            self._reply(404)

    def log_message(self, fmt, *args):
        pass


class _StubServerTestCase(CustomTestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _SglangServerHandler)
        self.server.posts = []
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"


class TestRuntimeEndpointJoin(_StubServerTestCase):
    def test_concate_and_append_join_appends_the_fork_texts(self):
        """join(mode="concate_and_append") must work against an SGLang server, which
        has no route for merging the forks' KV caches."""

        @sgl.function
        def program(s):
            s += "A"
            forks = s.fork(2)
            forks += lambda i: f"[{i}]"
            forks.join(mode="concate_and_append")

        state = program.run(backend=RuntimeEndpoint(self.base_url))

        self.assertIsNone(state.error())
        self.assertEqual(state.text(), "A[0][1]")


if __name__ == "__main__":
    unittest.main()
