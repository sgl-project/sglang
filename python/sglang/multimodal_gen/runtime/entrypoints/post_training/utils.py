"""Tensor serialization for post-training / rollout HTTP responses."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from safetensors.torch import load, save


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


def _maybe_deserialize(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get("__tensor__"):
            return bytes_to_tensor(obj["data"])
        return {k: _maybe_deserialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_maybe_deserialize(v) for v in obj]
    return obj
