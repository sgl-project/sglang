import asyncio
import unittest

import zstandard

from sglang.srt.entrypoints.http_response_compression import (
    _COMPRESS_MIN_BYTES,
    ResponseCompressionMiddleware,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="extra-test-cpu-intel")

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
        for accept in (b"gzip, zstd", b"zstd;q=0.5"):
            with self.subTest(accept=accept):
                sent = _drive(
                    accept,
                    [
                        _start([(b"content-length", str(len(LARGE)).encode())]),
                        _body(LARGE),
                    ],
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

    def test_passes_other_responses_through_unchanged(self):
        for case, accept, messages in (
            ("no accept-encoding", None, [_start(), _body(LARGE)]),
            ("other encodings", b"gzip, br", [_start(), _body(LARGE)]),
            ("zstd refused", b"zstd;q=0", [_start(), _body(LARGE)]),
            ("small", b"zstd", [_start(), _body(SMALL)]),
            (
                "streamed",
                b"zstd",
                [_start(), _body(LARGE, more_body=True), _body(LARGE)],
            ),
            (
                "encoded",
                b"zstd",
                [_start([(b"content-encoding", b"gzip")]), _body(LARGE)],
            ),
            (
                "encoded, capitalized",
                b"zstd",
                [_start([(b"Content-Encoding", b"gzip")]), _body(LARGE)],
            ),
        ):
            with self.subTest(case=case):
                self.assertEqual(_drive(accept, messages), messages)


if __name__ == "__main__":
    unittest.main()
