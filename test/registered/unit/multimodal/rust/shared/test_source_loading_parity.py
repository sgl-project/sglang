"""Model-independent parity tests for Rust multimodal source loading."""

import base64
import http.server
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from sglang.srt.utils.common import get_image_bytes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _utils import load_core  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CORE = load_core()
FETCH = CORE and CORE.common.fetch_bytes


@unittest.skipUnless(FETCH, "sglang-mm fetch binding not built")
class TestRustMediaSourceLoading(CustomTestCase):
    DATA = b"native-mm-source"

    def test_inline_sources(self):
        encoded = base64.b64encode(self.DATA).decode()
        for source in (encoded, f"data:application/octet-stream;base64,{encoded}"):
            with self.subTest(source=source[:8]):
                self.assertEqual(bytes(FETCH(source)), get_image_bytes(source))

    def test_file_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(self.DATA)
            self.assertEqual(bytes(FETCH(str(path))), get_image_bytes(str(path)))
            self.assertEqual(bytes(FETCH(path.as_uri())), self.DATA)

    def test_http_source(self):
        data = self.DATA

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/image"
            self.assertEqual(bytes(FETCH(url)), get_image_bytes(url))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_invalid_sources_fail(self):
        for source in ("not base64!", "/definitely/missing/image.png"):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    FETCH(source)


if __name__ == "__main__":
    unittest.main()
