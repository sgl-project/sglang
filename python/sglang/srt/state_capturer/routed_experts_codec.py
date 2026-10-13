"""The wire format of returned routed experts (``meta_info["routed_experts"]``).

Base64 of a 16-byte header -- magic ``SGRE``, version, numpy dtype string, num_tokens,
num_layers, topk -- then one zstd frame of the rows laid out ``[num_layers,
num_tokens, topk]``. Rows come from the capturer sorted and in its narrow host dtype;
replay needs only each row's expert set, and sorted, layer-major rows compress best.
"""

import struct
from typing import List, Optional

import numpy as np
import pybase64
import zstandard

_LEVEL = 1
# Multi-threaded zstd only pays off once the frame spans several jobs.
_THREADED_MIN_BYTES = 4 << 20
_THREADS = 4

_MAGIC = b"SGRE"
_VERSION = 1
_HEADER = struct.Struct("<4sB3sIHH")
# The capturer's host dtypes (see routed_experts._host_dtype), the only ones on the wire.
ROUTED_EXPERTS_DTYPES = (np.dtype("|u1"), np.dtype("<i2"), np.dtype("<i4"))
_DTYPE_STRS = frozenset(dtype.str for dtype in ROUTED_EXPERTS_DTYPES)

_ROW_DTYPES = {
    1: np.dtype(np.uint8),
    2: np.dtype(np.uint16),
    4: np.dtype(np.uint32),
    8: np.dtype(np.uint64),
}
_TRANSPOSE_BLOCK = 1024


class RoutedExpertsEncoder:
    """Encodes per-request rows, reusing its zstd compressors; built once per process
    that serializes responses."""

    def __init__(self):
        self._compressor = zstandard.ZstdCompressor(level=_LEVEL)
        self._threaded_compressor = zstandard.ZstdCompressor(
            level=_LEVEL, threads=_THREADS
        )

    def encode(self, rows: np.ndarray) -> str:
        """Encode one request's ``[num_tokens, num_layers, topk]`` rows."""
        if rows.dtype.str not in _DTYPE_STRS:
            raise ValueError(f"routed experts dtype {rows.dtype} is not a host dtype")
        layer_major = _swap_leading_axes(np.ascontiguousarray(rows))
        compressor = (
            self._threaded_compressor
            if layer_major.nbytes >= _THREADED_MIN_BYTES
            else self._compressor
        )
        header = _HEADER.pack(
            _MAGIC, _VERSION, rows.dtype.str.encode("ascii"), *rows.shape
        )
        return pybase64.b64encode_as_string(header + compressor.compress(layer_major))

    def encode_per_request(
        self, rows_list: Optional[List[Optional[np.ndarray]]]
    ) -> Optional[List[Optional[str]]]:
        if rows_list is None:
            return None
        return [self.encode(rows) if rows is not None else None for rows in rows_list]


def decode_routed_experts(payload: str) -> np.ndarray:
    """Writable ``[num_tokens, num_layers, topk]`` rows in the captured dtype; any
    malformed payload, including the pre-v1 base64 int32 format, raises ``ValueError``."""
    data = pybase64.b64decode_as_bytearray(payload, validate=True)
    if len(data) < _HEADER.size:
        raise ValueError(f"routed experts payload of {len(data)} bytes has no header")
    magic, version, dtype_str, num_tokens, num_layers, topk = _HEADER.unpack_from(data)
    if magic != _MAGIC or version != _VERSION:
        raise ValueError(
            f"not a v{_VERSION} routed experts payload (magic {magic!r}, version "
            f"{version}); is the server's sglang older than this client?"
        )
    if dtype_str.decode("ascii", "replace") not in _DTYPE_STRS:
        raise ValueError(f"routed experts payload has unsupported dtype {dtype_str!r}")
    dtype = np.dtype(dtype_str.decode("ascii"))
    nbytes = num_tokens * num_layers * topk * dtype.itemsize
    try:
        raw = zstandard.ZstdDecompressor().decompress(
            memoryview(data)[_HEADER.size :], max_output_size=nbytes
        )
    except zstandard.ZstdError as exc:
        raise ValueError(f"corrupt routed experts zstd frame: {exc}") from exc
    if len(raw) != nbytes:
        raise ValueError(
            f"routed experts frame holds {len(raw)} bytes, header promises {nbytes}"
        )
    layer_major = np.frombuffer(raw, dtype=dtype).reshape(num_layers, num_tokens, topk)
    return _swap_leading_axes(layer_major)


def _swap_leading_axes(arr: np.ndarray) -> np.ndarray:
    """C-contiguous ``[a, b, topk]`` -> new contiguous ``[b, a, topk]``.

    Single-threaded, as the detokenizer must not spin a thread pool: each ``topk``
    row moves as one element, a block of the longer axis at a time so both sides
    of the copy stay in cache; several times faster than a plain strided copy.
    """
    assert arr.flags.c_contiguous, "rows must be C-contiguous to move as elements"
    a, b, topk = arr.shape
    if arr.size == 0:
        return np.empty((b, a, topk), dtype=arr.dtype)
    row_bytes = topk * arr.itemsize
    rows = arr.view(_ROW_DTYPES.get(row_bytes, np.dtype(f"V{row_bytes}"))).reshape(a, b)
    out = np.empty((b, a), dtype=rows.dtype)
    if a >= b:
        for start in range(0, a, _TRANSPOSE_BLOCK):
            out[:, start : start + _TRANSPOSE_BLOCK] = rows[
                start : start + _TRANSPOSE_BLOCK
            ].T
    else:
        for start in range(0, b, _TRANSPOSE_BLOCK):
            out[start : start + _TRANSPOSE_BLOCK] = rows[
                :, start : start + _TRANSPOSE_BLOCK
            ].T
    return out.view(arr.dtype).reshape(b, a, topk)
