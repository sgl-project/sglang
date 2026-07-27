"""Pure-ASGI middleware that decompresses compressed request bodies.

Gated on `SGLANG_ENABLE_REQUEST_DECOMPRESSION` and request header
`x-body-compressed`, whose value names the method. For example, a caller that
compressed the body with zstd sets the `x-body-compressed: zstd` header.
"""

import asyncio
import io
import logging

import zstandard
from fastapi.responses import Response
from starlette.datastructures import Headers

logger = logging.getLogger(__name__)


def _zstd_decompress(raw: bytes) -> bytes:
    return zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)).read()


_DECOMPRESSORS = {"zstd": _zstd_decompress}


def _rewrite_headers(headers, new_len):
    """Update headers to reflect body status after decompression."""
    out = [
        (k, v)
        for (k, v) in headers
        if k not in (b"content-length", b"x-body-compressed")
    ]
    out.append((b"content-length", str(new_len).encode()))
    return out


class RequestDecompressionMiddleware:
    """Decompress request body per request header `x-body-compressed`."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # No-op passthrough for any request without the compression header.
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        method = Headers(scope=scope).get("x-body-compressed")
        if method is None:
            return await self.app(scope, receive, send)

        # Fail loud on an unsupported compression method.
        decompress = _DECOMPRESSORS.get(method)
        if decompress is None:
            return await Response(
                f"unsupported x-body-compressed {method!r}; "
                f"supported: {sorted(_DECOMPRESSORS)}",
                status_code=400,
            )(scope, receive, send)

        # Collect request body.
        body = b""
        more_body = True
        while more_body:
            message = await receive()
            # Incomplete body (e.g. client disconnect); hand off to later stages.
            if message["type"] != "http.request":
                return await self.app(scope, receive, send)
            body += message.get("body", b"")
            more_body = message.get("more_body", False)

        # Decompress off the event loop by releasing the GIL around the C decompress.
        try:
            loop = asyncio.get_running_loop()
            body = await loop.run_in_executor(None, decompress, body)
        except Exception as e:
            logger.warning("request body decompress failed: %s", e)
            return await Response("decompress failed", status_code=400)(
                scope, receive, send
            )

        # Update the headers after decompression
        scope = dict(scope)
        scope["headers"] = _rewrite_headers(scope["headers"], len(body))

        # Fake receiver to let later stages see the decompressed body.
        body_sent = False

        async def wrapped_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, wrapped_receive, send)
