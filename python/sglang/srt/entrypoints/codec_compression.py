"""
Negotiated transport compression for Codec binary streaming responses.

Compression is opt-in (PKCE-style): clients advertise supported encodings
via the standard ``Accept-Encoding`` header, servers respond with whatever
overlap they choose, and the response is tagged with ``Content-Encoding``.
Clients that don't want compression simply omit the header and receive
identity-encoded frames as before.

Layered *outside* the Codec frame format — frames themselves are unchanged.
The compression covers the entire HTTP response stream so a single
compression context spans many frames (much better ratio than per-frame
compression for small frames).

Supported encodings, in server preference order:

  1. ``zstd``     — Zstandard. Best ratio + speed. Requires the optional
                    ``zstandard`` package; gracefully skipped if absent.
                    Browser support: Chrome 123+, Firefox 126+ (transparent
                    decoding via fetch / Content-Encoding).
  2. ``gzip``     — Universal fallback. Pure stdlib, always available.
                    Supported in 100% of browsers and Node 18+ via fetch.
  3. ``identity`` — No compression. Always available.
"""

from __future__ import annotations

import zlib
from typing import AsyncIterable, Optional

from fastapi.responses import StreamingResponse

try:
    import zstandard as zstd

    _ZSTD_AVAILABLE = True
except ImportError:
    _ZSTD_AVAILABLE = False


def _parse_accept_encoding(header: str) -> list[str]:
    """Return the encodings the client lists, ignoring q-values."""
    if not header:
        return []
    parts = []
    for part in header.split(","):
        name = part.strip().split(";", 1)[0].strip().lower()
        if name:
            parts.append(name)
    return parts


def negotiate_encoding(accept_encoding: str) -> Optional[str]:
    """Pick the best encoding both sides can speak.

    Returns ``"zstd"``, ``"gzip"``, or ``None`` (identity).
    """
    encs = _parse_accept_encoding(accept_encoding)
    if not encs:
        return None
    has_wildcard = "*" in encs

    if _ZSTD_AVAILABLE and ("zstd" in encs or has_wildcard):
        return "zstd"
    if "gzip" in encs or has_wildcard:
        return "gzip"
    return None


async def _compress_zstd(stream: AsyncIterable[bytes]) -> AsyncIterable[bytes]:
    cctx = zstd.ZstdCompressor(level=3)
    chunker = cctx.chunker(chunk_size=16384)
    async for chunk in stream:
        for out in chunker.compress(chunk):
            yield out
    for out in chunker.finish():
        yield out


async def _compress_gzip(stream: AsyncIterable[bytes]) -> AsyncIterable[bytes]:
    compressor = zlib.compressobj(level=6, wbits=31)  # 31 = gzip wrapper
    async for chunk in stream:
        out = compressor.compress(chunk)
        if out:
            yield out
    final = compressor.flush(zlib.Z_FINISH)
    if final:
        yield final


def wrap_streaming_response(
    accept_encoding: str,
    body_stream: AsyncIterable[bytes],
    *,
    media_type: str,
    background=None,
    extra_headers: Optional[dict] = None,
) -> StreamingResponse:
    """Build a StreamingResponse with compression chosen from Accept-Encoding.

    The Codec frame format is unchanged — compression is purely transport.
    Clients that don't include zstd/gzip in Accept-Encoding receive an
    uncompressed (identity-encoded) stream, which is the previous behavior.
    """
    encoding = negotiate_encoding(accept_encoding)
    headers: dict = {"Vary": "Accept-Encoding"}
    if extra_headers:
        headers.update(extra_headers)

    if encoding == "zstd":
        body = _compress_zstd(body_stream)
        headers["Content-Encoding"] = "zstd"
    elif encoding == "gzip":
        body = _compress_gzip(body_stream)
        headers["Content-Encoding"] = "gzip"
    else:
        body = body_stream

    return StreamingResponse(
        body,
        media_type=media_type,
        headers=headers,
        background=background,
    )
