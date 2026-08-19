import asyncio
import unittest

import zstandard

from sglang.srt.entrypoints.http_request_decompression import (
    RequestDecompressionMiddleware,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-b-test-cpu")

PAYLOAD = b'{"text":"hello world","n":7}'
COMPRESSED = zstandard.ZstdCompressor().compress(PAYLOAD)


def _drive(scope, body_chunks):
    """Drive the middleware once. Returns (seen, sent): `seen` is what the inner
    app received ({scope, body}) or None if the app was never called; `sent` is
    the list of ASGI messages the middleware emitted directly."""
    seen = {}
    sent = []
    chunks = list(body_chunks)

    async def receive():
        if chunks:
            chunk, more = chunks.pop(0)
            return {"type": "http.request", "body": chunk, "more_body": more}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    async def app(inner_scope, inner_receive, inner_send):
        body = b""
        more = True
        while more:
            message = await inner_receive()
            if message["type"] != "http.request":
                break
            body += message.get("body", b"")
            more = message.get("more_body", False)
        seen["scope"] = inner_scope
        seen["body"] = body

    asyncio.run(RequestDecompressionMiddleware(app)(scope, receive, send))
    return (seen or None), sent


class TestRequestDecompressionMiddleware(unittest.TestCase):
    def test_passthrough_when_header_absent(self):
        scope = {"type": "http", "headers": [(b"content-length", b"4")]}
        seen, sent = _drive(scope, [(b"abcd", False)])
        self.assertEqual(seen["body"], b"abcd")
        self.assertEqual(seen["scope"]["headers"], [(b"content-length", b"4")])
        self.assertEqual(sent, [])

    def test_decompresses_zstd_body(self):
        scope = {
            "type": "http",
            "headers": [
                (b"x-body-compressed", b"zstd"),
                (b"content-length", str(len(COMPRESSED)).encode()),
            ],
        }
        seen, sent = _drive(scope, [(COMPRESSED, False)])
        self.assertEqual(seen["body"], PAYLOAD)
        self.assertEqual(sent, [])

    def test_strips_header_and_fixes_content_length(self):
        scope = {
            "type": "http",
            "headers": [
                (b"x-body-compressed", b"zstd"),
                (b"content-length", str(len(COMPRESSED)).encode()),
                (b"content-type", b"application/json"),
            ],
        }
        seen, _ = _drive(scope, [(COMPRESSED, False)])
        self.assertEqual(
            seen["scope"]["headers"],
            [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(PAYLOAD)).encode()),
            ],
        )

    def test_unsupported_method_returns_400(self):
        scope = {"type": "http", "headers": [(b"x-body-compressed", b"gzip")]}
        seen, sent = _drive(scope, [(b"abcd", False)])
        self.assertIsNone(seen)
        self.assertEqual(sent[0]["type"], "http.response.start")
        self.assertEqual(sent[0]["status"], 400)

    def test_chunked_body_reassembled(self):
        half = len(COMPRESSED) // 2
        scope = {"type": "http", "headers": [(b"x-body-compressed", b"zstd")]}
        seen, _ = _drive(scope, [(COMPRESSED[:half], True), (COMPRESSED[half:], False)])
        self.assertEqual(seen["body"], PAYLOAD)

    def test_bad_body_returns_400(self):
        scope = {"type": "http", "headers": [(b"x-body-compressed", b"zstd")]}
        seen, sent = _drive(scope, [(b"not-zstd-data", False)])
        self.assertIsNone(seen)
        self.assertEqual(sent[0]["type"], "http.response.start")
        self.assertEqual(sent[0]["status"], 400)

    def test_non_http_scope_passthrough(self):
        scope = {"type": "lifespan", "headers": []}
        seen, sent = _drive(scope, [])
        self.assertEqual(seen["scope"]["type"], "lifespan")
        self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main()
