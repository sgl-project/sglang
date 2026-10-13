"""Wire encodings of returned routed experts, chosen by ``--routed-experts-encoding``.

Rows arrive sorted and in the capturer's narrow host dtype. ``base64_int32`` (the
default, unlabelled) is base64 of them widened to int32 ``[num_tokens, num_layers,
topk]``. ``zstd`` (labelled in ``meta_info["routed_experts_encoding"]``) is base64 of
a ``<BIHH`` header (dtype itemsize, num_tokens, num_layers, topk) and one zstd frame
of the rows as captured, laid out ``[num_layers, num_tokens, topk]``.
"""

import struct
from typing import List, Optional

import numpy as np
import pybase64
import zstandard

BASE64_INT32 = "base64_int32"
ZSTD = "zstd"

_ZSTD_LEVEL = 1
# Multi-threaded zstd only pays off once the frame spans several jobs.
_ZSTD_THREADED_MIN_BYTES = 4 << 20
_ZSTD_THREADS = 4

_ZSTD_HEADER = struct.Struct("<BIHH")
_ROW_DTYPES = {
    1: np.dtype(np.uint8),
    2: np.dtype(np.uint16),
    4: np.dtype(np.uint32),
    8: np.dtype(np.uint64),
}
_TRANSPOSE_BLOCK = 1024
_ITEMSIZE_TO_DTYPE = {
    1: np.dtype(np.uint8),
    2: np.dtype(np.uint16),
    4: np.dtype(np.int32),
}


def _swap_leading_axes(arr: np.ndarray) -> np.ndarray:
    """C-contiguous ``[a, b, topk]`` -> contiguous ``[b, a, topk]``.

    Single-threaded, as the detokenizer must not spin a thread pool: each ``topk``
    row moves as one element, a block of the longer axis at a time so both sides
    of the copy stay in cache; several times faster than a plain strided copy.
    """
    a, b, topk = arr.shape
    if arr.size == 0:
        return arr.reshape(b, a, topk)
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


def encode_routed_experts(arr: np.ndarray, encoding: str) -> str:
    """Encode one request's ``[num_tokens, num_layers, topk]`` rows."""
    if encoding == BASE64_INT32:
        arr = arr.astype(np.int32, copy=False)
        return pybase64.b64encode(arr.tobytes()).decode("utf-8")
    if encoding != ZSTD:
        raise ValueError(f"unknown routed experts encoding {encoding!r}")
    # captured ids are non-negative, so int16 rows read the same as uint16
    rows = np.ascontiguousarray(arr.view(np.uint16) if arr.dtype == np.int16 else arr)
    layer_major = _swap_leading_axes(rows)
    threads = _ZSTD_THREADS if layer_major.nbytes >= _ZSTD_THREADED_MIN_BYTES else 0
    frame = zstandard.ZstdCompressor(level=_ZSTD_LEVEL, threads=threads).compress(
        layer_major.data
    )
    header = _ZSTD_HEADER.pack(rows.itemsize, *arr.shape)
    return pybase64.b64encode(header + frame).decode("utf-8")


def encode_routed_experts_per_request(
    data_list: Optional[List[Optional[np.ndarray]]], encoding: str
) -> Optional[List[Optional[str]]]:
    if data_list is None:
        return None
    return [
        encode_routed_experts(item, encoding) if item is not None else None
        for item in data_list
    ]


def decode_routed_experts(
    payload: str,
    encoding: Optional[str],
    num_tokens: Optional[int] = None,
    num_layers: Optional[int] = None,
) -> np.ndarray:
    """Inverse of ``encode_routed_experts``: ``[num_tokens, num_layers, topk]``,
    int32 for ``base64_int32`` and the transmitted narrow dtype for ``zstd``.

    ``base64_int32`` carries no shape: without ``num_tokens`` / ``num_layers`` the
    flat buffer is returned.
    """
    data = pybase64.b64decode(payload.encode("ascii"))
    if encoding in (None, BASE64_INT32):
        flat = np.frombuffer(data, dtype=np.int32)
        if num_tokens is None or num_layers is None:
            return flat
        if num_tokens == 0:
            return flat.reshape(0, num_layers, 0)
        return flat.reshape(num_tokens, num_layers, -1)
    if encoding != ZSTD:
        raise ValueError(f"unknown routed experts encoding {encoding!r}")
    itemsize, tokens, layers, topk = _ZSTD_HEADER.unpack_from(data)
    raw = (
        zstandard.ZstdDecompressor()
        .decompressobj()
        .decompress(data[_ZSTD_HEADER.size :])
    )
    layer_major = np.frombuffer(raw, dtype=_ITEMSIZE_TO_DTYPE[itemsize]).reshape(
        layers, tokens, topk
    )
    return _swap_leading_axes(layer_major)
