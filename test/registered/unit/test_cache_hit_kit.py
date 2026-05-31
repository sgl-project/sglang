import asyncio
import json
import os
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.cache_hit_kit import (
    async_request_openai_chat_completions,
    gen_payload_openai,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _ChatCompletionHandler(BaseHTTPRequestHandler):
    auth_headers = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        self.auth_headers.append(self.headers.get("Authorization"))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        chunks = [
            {"choices": [{"delta": {"content": "ok"}}], "usage": None},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
        ]
        for chunk in chunks:
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, fmt, *args):
        return


class TestOpenAIChatCompletionHeaders(CustomTestCase):
    def test_openai_api_key_is_sent_for_openai_benchmark_requests(self):
        _ChatCompletionHandler.auth_headers = []
        server = HTTPServer(("127.0.0.1", _free_port()), _ChatCompletionHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
        payload = gen_payload_openai(
            [{"role": "user", "content": "hello"}],
            output_len=1,
            model="test-model",
        )

        try:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "unit-test-token"}):
                output = asyncio.run(
                    async_request_openai_chat_completions(payload, url)
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertTrue(output.success)
        self.assertEqual(
            _ChatCompletionHandler.auth_headers, ["Bearer unit-test-token"]
        )


if __name__ == "__main__":
    unittest.main()
