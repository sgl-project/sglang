"""Unit tests for the sglang-embedding benchmark request function."""

import argparse
import asyncio
import base64
import json
import struct
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import sglang.benchmark.serving as bench_serving
from sglang.benchmark.serving import RequestFuncInput, async_request_openai_embeddings
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# get_request_headers() reads the module-level `args` that the CLI entrypoint
# sets; provide the same global for direct request-function calls.
bench_serving.args = argparse.Namespace(header=None)

EMBEDDING_DIM = 8
FLOAT_EMBEDDING = [0.125 * i for i in range(EMBEDDING_DIM)]
BASE64_EMBEDDING = base64.b64encode(
    struct.pack(f"<{EMBEDDING_DIM}f", *FLOAT_EMBEDDING)
).decode("ascii")


def _embedding_response(embedding):
    return json.dumps(
        {
            "object": "list",
            "data": [{"object": "embedding", "embedding": embedding, "index": 0}],
            "model": "test-model",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
    ).encode()


class _EmbeddingServer:
    """Serves canned /v1/embeddings responses and records request payloads."""

    def __init__(self, response_body: bytes, status: int = 200):
        self.requests = []
        self.response_body = response_body
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.requests.append(json.loads(self.rfile.read(length)))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(outer.response_body)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1/embeddings"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def shutdown(self):
        self.server.shutdown()


def _run_request(api_url: str, extra_request_body=None):
    request_func_input = RequestFuncInput(
        prompt="hello world",
        api_url=api_url,
        prompt_len=3,
        output_len=0,
        model="test-model",
        lora_name="",
        image_data=None,
        extra_request_body=extra_request_body or {},
    )
    return asyncio.run(async_request_openai_embeddings(request_func_input))


class TestEmbeddingRequestPayload(CustomTestCase):
    def test_float_response_records_payload_bytes(self):
        body = _embedding_response(FLOAT_EMBEDDING)
        server = _EmbeddingServer(body)
        try:
            output = _run_request(server.url)
        finally:
            server.shutdown()

        self.assertTrue(output.success)
        self.assertGreater(output.latency, 0)
        self.assertEqual(output.response_bytes, len(body))
        self.assertEqual(output.output_len, 0)

    def test_base64_response_records_payload_bytes(self):
        body = _embedding_response(BASE64_EMBEDDING)
        server = _EmbeddingServer(body)
        try:
            output = _run_request(
                server.url, extra_request_body={"encoding_format": "base64"}
            )
        finally:
            server.shutdown()

        self.assertTrue(output.success)
        self.assertEqual(output.response_bytes, len(body))
        # base64 of an 8-dim fp32 embedding is far smaller than its
        # JSON float-list form.
        self.assertLess(len(body), len(_embedding_response(FLOAT_EMBEDDING)))

    def test_extra_request_body_passes_encoding_format(self):
        server = _EmbeddingServer(_embedding_response(BASE64_EMBEDDING))
        try:
            _run_request(server.url, extra_request_body={"encoding_format": "base64"})
            payload = server.requests[0]
        finally:
            server.shutdown()

        self.assertEqual(payload["encoding_format"], "base64")
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["input"], "hello world")

    def test_invalid_json_body_fails_request(self):
        server = _EmbeddingServer(b"not json")
        try:
            output = _run_request(server.url)
        finally:
            server.shutdown()

        self.assertFalse(output.success)
        self.assertIn("JSONDecodeError", output.error)

    def test_non_200_fails_request(self):
        server = _EmbeddingServer(b'{"error": "boom"}', status=500)
        try:
            output = _run_request(server.url)
        finally:
            server.shutdown()

        self.assertFalse(output.success)
        self.assertEqual(output.response_bytes, 0)


if __name__ == "__main__":
    unittest.main()
