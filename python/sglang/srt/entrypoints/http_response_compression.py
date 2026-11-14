"""Pure-ASGI middleware that zstd-compresses whole response bodies.

Gated on `SGLANG_ENABLE_RESPONSE_COMPRESSION` and a request `Accept-Encoding`
that lists zstd.
"""

import asyncio

import zstandard
from starlette.datastructures import Headers, MutableHeaders

# Smaller bodies gain too little to pay for the trip to the executor.
_COMPRESS_MIN_BYTES = 64 * 1024


def _accepts_zstd(scope) -> bool:
    for coding in Headers(scope=scope).get("accept-encoding", "").split(","):
        name, _, params = coding.partition(";")
        if name.strip().lower() != "zstd":
            continue
        weight = params.strip().lower().removeprefix("q=")
        try:
            return not weight or float(weight) > 0
        except ValueError:
            return False
    return False


def _zstd_compress(body: bytes) -> bytes:
    return zstandard.ZstdCompressor().compress(body)


class ResponseCompressionMiddleware:
    """Compress response body per request header `Accept-Encoding`."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not _accepts_zstd(scope):
            return await self.app(scope, receive, send)

        # Hold the start until the first body: only a whole body is compressed.
        initial_message = None
        started = False

        async def wrapped_send(message):
            nonlocal initial_message, started
            if started:
                return await send(message)
            if message["type"] == "http.response.start":
                initial_message = message
                return
            started = True
            body = message.get("body", b"")
            headers = MutableHeaders(raw=initial_message["headers"])
            if (
                message["type"] != "http.response.body"
                or message.get("more_body", False)
                or len(body) < _COMPRESS_MIN_BYTES
                or "content-encoding" in headers
            ):
                await send(initial_message)
                return await send(message)

            # Compress off the event loop by releasing the GIL around the C compress.
            body = await asyncio.get_running_loop().run_in_executor(
                None, _zstd_compress, body
            )
            headers["content-encoding"] = "zstd"
            headers["content-length"] = str(len(body))
            headers.add_vary_header("Accept-Encoding")
            await send(initial_message)
            await send({**message, "body": body})

        await self.app(scope, receive, wrapped_send)
