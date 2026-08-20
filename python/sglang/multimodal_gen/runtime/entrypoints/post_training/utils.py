"""Tensor serialization for post-training / rollout HTTP responses."""

from __future__ import annotations

from typing import Any

import msgspec
import numpy as np
import torch
from safetensors.torch import load, save

_SPLICE_MIN_BYTES = 1 << 20


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    return save({"t": t.detach().contiguous().cpu()})


def bytes_to_tensor(b: bytes) -> torch.Tensor:
    return load(b)["t"]


def _maybe_serialize(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return {
            "__tensor__": True,
            "data": tensor_to_bytes(obj),
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    if isinstance(obj, np.ndarray):
        return _maybe_serialize(torch.from_numpy(obj))
    if isinstance(obj, dict):
        return {k: _maybe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_maybe_serialize(v) for v in obj]
    return obj


def _container_header(tag_fix: int, tag16: bytes, tag32: bytes, n: int) -> bytes:
    if n <= 15:
        return bytes([tag_fix | n])
    if n <= 0xFFFF:
        return tag16 + n.to_bytes(2, "big")
    return tag32 + n.to_bytes(4, "big")


def msgpack_encode_spliced(obj: Any, threshold: int = _SPLICE_MIN_BYTES) -> list[bytes]:
    """Encode to msgpack as a list of buffers, splicing large ``bytes`` values.

    Container and bin32 headers are hand-written so a large ``bytes`` payload
    lands in the output by reference, never copied through the encoder; the
    concatenated parts are byte-identical to ``msgspec.msgpack.encode(obj)``.
    """
    parts: list[bytes] = []
    small = bytearray()

    def _emit(value: Any) -> None:
        if isinstance(value, bytes) and len(value) >= threshold:
            small.extend(b"\xc6" + len(value).to_bytes(4, "big"))
            parts.append(bytes(small))
            small.clear()
            parts.append(value)
        elif isinstance(value, dict):
            small.extend(_container_header(0x80, b"\xde", b"\xdf", len(value)))
            for key, item in value.items():
                small.extend(msgspec.msgpack.encode(key))
                _emit(item)
        elif isinstance(value, (list, tuple)):
            small.extend(_container_header(0x90, b"\xdc", b"\xdd", len(value)))
            for item in value:
                _emit(item)
        else:
            small.extend(msgspec.msgpack.encode(value))

    _emit(obj)
    if small:
        parts.append(bytes(small))
    return parts


def _maybe_deserialize(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get("__tensor__"):
            return bytes_to_tensor(obj["data"])
        return {k: _maybe_deserialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_maybe_deserialize(v) for v in obj]
    return obj


def _quantize_video_uint8(video: torch.Tensor) -> torch.Tensor:
    """Map the decoded [0,1] float video to 0..255 uint8; consumers divide by 255."""
    out = video.float()
    if float(out.max()) <= 1.0 + 1e-3:
        out = out * 255.0
    return out.clamp(0, 255).to(torch.uint8)
