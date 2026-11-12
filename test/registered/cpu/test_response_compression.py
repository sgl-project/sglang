import asyncio
import unittest

import zstandard

from sglang.srt.entrypoints.http_response_compression import (
    _COMPRESS_MIN_BYTES,
    ResponseCompressionMiddleware,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")

LARGE = b'{"text":"' + b"hello world " * _COMPRESS_MIN_BYTES + b'"}'
SMALL = b'{"text":"hello world"}'


def _start(headers=()):
    return {
        "type": "http.response.start",
        "status": 200,
        "headers": [(b"content-type", b"application/json"), *headers],
    }


def _body(body, more_body=False):
    return {"type": "http.response.body", "body": body, "more_body": more_body}


def _drive(accept_encoding, messages):
    """Run the middleware over an app that sends `messages`; returns what reached the client."""
    headers = [] if accept_encoding is None else [(b"accept-encoding", accept_encoding)]
    scope = {"type": "http", "headers": headers}
    sent = []

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    async def app(inner_scope, inner_receive, inner_send):
        for message in messages:
            await inner_send(message)

    asyncio.run(ResponseCompressionMiddleware(app)(scope, receive, send))
    return sent


def _header(message, name):
    values = [v for k, v in message["headers"] if k == name]
    return values[0] if values else None


class TestResponseCompressionMiddleware(CustomTestCase):
    def test_compresses_zstd_body(self):
        sent = _drive(
            b"gzip, zstd",
            [_start([(b"content-length", str(len(LARGE)).encode())]), _body(LARGE)],
        )
        self.assertEqual(len(sent), 2)
        start, body = sent
        self.assertEqual(_header(start, b"content-encoding"), b"zstd")
        self.assertEqual(
            _header(start, b"content-length"), str(len(body["body"])).encode()
        )
        self.assertIn(b"Accept-Encoding", _header(start, b"vary"))
        self.assertEqual(_header(start, b"content-type"), b"application/json")
        self.assertLess(len(body["body"]), len(LARGE))
        self.assertEqual(
            zstandard.ZstdDecompressor().decompress(
                body["body"], max_output_size=len(LARGE)
            ),
            LARGE,
        )

    def test_passthrough_unless_zstd_accepted(self):
        for accept, compressed in (
            (None, False),
            (b"gzip, br", False),
            (b"zstd;q=0", False),
            (b"zstd;q=0.5", True),
        ):
            with self.subTest(accept=accept):
                start, _ = _drive(accept, [_start(), _body(LARGE)])
                self.assertEqual(
                    _header(start, b"content-encoding"), b"zstd" if compressed else None
                )

    def test_passthrough_for_small_streamed_or_encoded_body(self):
        for case, messages in (
            ("small", [_start(), _body(SMALL)]),
            ("streamed", [_start(), _body(LARGE, more_body=True), _body(LARGE)]),
            ("encoded", [_start([(b"content-encoding", b"gzip")]), _body(LARGE)]),
        ):
            with self.subTest(case=case):
                self.assertEqual(_drive(b"zstd", messages), messages)


if __name__ == "__main__":
    unittest.main()
