# SPDX-License-Identifier: Apache-2.0
"""Lossless PNG encoding that deflates horizontal strips in parallel.

Each strip is filtered (PNG filter 2, Up), deflated as raw blocks ending in a
full flush, and written as its own IDAT chunk. The concatenated chunks form one
zlib stream whose Adler-32 is combined from the per-strip checksums, so no
step touches the whole image on one thread.
"""

import os
import struct
import zlib
from concurrent.futures import Executor, ThreadPoolExecutor

import numpy as np

# Measured at level 1 on 2048x2048 RGB: Pillow's single-threaded encoder is ~18x
# slower than this path; below 4 MiB it costs only tens of ms.
PARALLEL_PNG_MIN_BYTES = 4 << 20
# Arbitrary; keeps each strip large enough to amortize the thread hand-off.
_MIN_STRIP_BYTES = 1 << 20
_MAX_THREADS = 32

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_FILTER_UP = 2
# Channels -> PNG color type (grayscale, RGB, RGBA).
_COLOR_TYPE = {1: 0, 3: 2, 4: 6}
_ADLER_BASE = 65521

_executor: ThreadPoolExecutor | None = None


def _thread_count() -> int:
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    return max(1, min(_MAX_THREADS, cpus))


def png_encoder_executor() -> Executor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=_thread_count(), thread_name_prefix="png-encode"
        )
    return _executor


def parallel_png_strip_count(frame: np.ndarray) -> int:
    return max(
        1, min(_thread_count(), frame.nbytes // _MIN_STRIP_BYTES, frame.shape[0])
    )


def encode_png_parallel(
    frame: np.ndarray, *, compress_level: int, num_strips: int, executor: Executor
) -> bytes:
    """Encode an 8-bit HxW, HxWx1, HxWx3 or HxWx4 array as a PNG."""
    if frame.dtype != np.uint8:
        raise ValueError(f"PNG encoder expects uint8 pixels, got {frame.dtype}")
    channels = 1 if frame.ndim == 2 else frame.shape[-1]
    if frame.ndim not in (2, 3) or channels not in _COLOR_TYPE:
        raise ValueError(f"unsupported PNG frame shape {frame.shape}")
    height, width = frame.shape[:2]
    rows = np.ascontiguousarray(frame).reshape(height, width * channels)
    bounds = np.linspace(0, height, min(num_strips, height) + 1).astype(int)
    strips = list(zip(bounds[:-1].tolist(), bounds[1:].tolist()))
    last = len(strips) - 1

    futures = [
        executor.submit(
            _encode_strip,
            rows,
            start=start,
            stop=stop,
            compress_level=compress_level,
            is_first=index == 0,
            is_last=index == last,
        )
        for index, (start, stop) in enumerate(strips)
    ]
    encoded = [future.result() for future in futures]

    adler = 1
    for _, _, strip_adler, strip_len in encoded:
        adler = _adler32_combine(adler1=adler, adler2=strip_adler, len2=strip_len)
    ihdr = _ihdr(width=width, height=height, channels=channels)
    chunks = [_PNG_SIGNATURE, _chunk(tag=b"IHDR", data=ihdr)]
    for index, (payload, payload_crc, _, _) in enumerate(encoded):
        if index == last:
            trailer = struct.pack(">I", adler)
            payload += trailer
            payload_crc = zlib.crc32(trailer, payload_crc)
        chunks.append(_chunk_with_crc(tag=b"IDAT", data=payload, crc=payload_crc))
    chunks.append(_chunk(tag=b"IEND", data=b""))
    return b"".join(chunks)


# Returns (IDAT payload, its chunk CRC, Adler-32 and length of the filtered rows).
def _encode_strip(
    rows: np.ndarray,
    *,
    start: int,
    stop: int,
    compress_level: int,
    is_first: bool,
    is_last: bool,
) -> tuple[bytes, int, int, int]:
    filtered = np.empty((stop - start, rows.shape[1] + 1), dtype=np.uint8)
    filtered[:, 0] = _PNG_FILTER_UP
    previous = rows[start - 1] if start > 0 else 0
    np.subtract(rows[start], previous, out=filtered[0, 1:], casting="unsafe")
    np.subtract(rows[start + 1 : stop], rows[start : stop - 1], out=filtered[1:, 1:])

    compressor = zlib.compressobj(compress_level, zlib.DEFLATED, -zlib.MAX_WBITS)
    payload = compressor.compress(filtered) + compressor.flush(
        zlib.Z_FINISH if is_last else zlib.Z_FULL_FLUSH
    )
    if is_first:
        payload = _zlib_header(compress_level) + payload
    return (
        payload,
        zlib.crc32(payload, zlib.crc32(b"IDAT")),
        zlib.adler32(filtered),
        filtered.nbytes,
    )


def _zlib_header(compress_level: int) -> bytes:
    # RFC 1950: 32K-window deflate. FLEVEL (0-3) is informational: fastest, fast,
    # default, best; FCHECK makes the 16-bit header divisible by 31.
    if compress_level < 2:
        flevel = 0
    elif compress_level < 6:
        flevel = 1
    elif compress_level == 6:
        flevel = 2
    else:
        flevel = 3
    cmf = 0x78
    flg = flevel << 6
    flg |= 31 - ((cmf << 8 | flg) % 31)
    return bytes((cmf, flg))


def _adler32_combine(*, adler1: int, adler2: int, len2: int) -> int:
    # Port of zlib's adler32_combine, which Python's zlib module does not expose.
    rem = len2 % _ADLER_BASE
    sum1 = adler1 & 0xFFFF
    sum2 = (rem * sum1) % _ADLER_BASE
    sum1 = (sum1 + (adler2 & 0xFFFF) + _ADLER_BASE - 1) % _ADLER_BASE
    sum2 = (sum2 + (adler1 >> 16) + (adler2 >> 16) + _ADLER_BASE - rem) % _ADLER_BASE
    return sum2 << 16 | sum1


def _ihdr(*, width: int, height: int, channels: int) -> bytes:
    # 8-bit depth, deflate, adaptive filtering method 0, no interlace.
    return struct.pack(">IIBBBBB", width, height, 8, _COLOR_TYPE[channels], 0, 0, 0)


def _chunk(*, tag: bytes, data: bytes) -> bytes:
    return _chunk_with_crc(tag=tag, data=data, crc=zlib.crc32(data, zlib.crc32(tag)))


def _chunk_with_crc(*, tag: bytes, data: bytes, crc: int) -> bytes:
    return (
        struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc & 0xFFFFFFFF)
    )
