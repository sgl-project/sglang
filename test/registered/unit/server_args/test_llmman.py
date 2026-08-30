"""The `llmman serve` client: the daemon protocol behind `oci://` model paths.

Exercised against a real HTTP server on a loopback port rather than mocks, so
the NDJSON streaming contract is genuinely tested.
"""

import http.server
import json
import socketserver
import threading
import unittest

from sglang.srt.utils import llmman
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _ndjson(*objs):
    return "".join(json.dumps(o) + "\n" for o in objs)


class _FakeDaemon:
    """A minimal stand-in for `llmman serve`, on a real loopback port."""

    def __init__(self):
        self.version = {"version": "0.1.0", "pid": 1}
        self.pull_body = _ndjson({"status": "success"})
        self.pull_status = 200
        self.last_request = None
        daemon = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body, ctype):
                raw = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                self._send(200, json.dumps(daemon.version), "application/json")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                daemon.last_request = json.loads(self.rfile.read(length))
                self._send(daemon.pull_status, daemon.pull_body, "application/x-ndjson")

        self._server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()


class TestLlmmanDaemon(CustomTestCase):
    def setUp(self):
        self.daemon = _FakeDaemon()

    def tearDown(self):
        self.daemon.close()

    def test_accepts_a_llmman_daemon(self):
        llmman.check_daemon(self.daemon.url)

    def test_rejects_a_non_llmman_server(self):
        self.daemon.version = {"hello": "world"}
        with self.assertRaisesRegex(RuntimeError, "not an llmman daemon"):
            llmman.check_daemon(self.daemon.url)

    def test_reports_nothing_listening_actionably(self):
        with self.assertRaisesRegex(RuntimeError, "llmman serve"):
            llmman.check_daemon("http://127.0.0.1:1")

    def test_pull_succeeds_and_forwards_progress(self):
        self.daemon.pull_body = _ndjson(
            {"status": "pulling manifest"},
            {"status": "pulling blobs", "completed": 50, "total": 100},
            {"status": "success"},
        )
        seen = []
        llmman.pull(self.daemon.url, "ghcr.io/org/model:tag", lambda *a: seen.append(a))

        self.assertEqual(self.daemon.last_request, {"model": "ghcr.io/org/model:tag"})
        self.assertEqual(seen, [("pulling manifest", 0, 0), ("pulling blobs", 50, 100)])

    def test_reports_an_in_band_error_at_http_200(self):
        # The daemon streams errors in-band, so a 200 does not mean success.
        self.daemon.pull_body = _ndjson(
            {"status": "pulling"}, {"error": "unauthorized"}
        )
        with self.assertRaisesRegex(RuntimeError, "unauthorized"):
            llmman.pull(self.daemon.url, "ref")

    def test_rejects_a_stream_that_ends_without_success(self):
        self.daemon.pull_body = _ndjson({"status": "pulling blobs"})
        with self.assertRaisesRegex(RuntimeError, "without reporting success"):
            llmman.pull(self.daemon.url, "ref")

    def test_reports_a_non_ok_status(self):
        self.daemon.pull_status = 400
        self.daemon.pull_body = '{"error":"bad request"}'
        with self.assertRaises(RuntimeError):
            llmman.pull(self.daemon.url, "ref")

    def test_tolerates_a_non_json_diagnostic_line(self):
        self.daemon.pull_body = "not json\n" + _ndjson({"status": "success"})
        llmman.pull(self.daemon.url, "ref")


if __name__ == "__main__":
    unittest.main()
