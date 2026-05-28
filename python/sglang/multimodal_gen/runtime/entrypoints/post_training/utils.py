"""Tensor serialization for post-training / rollout HTTP responses."""

from __future__ import annotations

import base64
from typing import Any

import torch
from safetensors.torch import load, save


def tensor_to_base64(t: torch.Tensor) -> str:
    t = t.detach().contiguous().cpu()
    raw = save({"t": t})
    return base64.b64encode(raw).decode("ascii")


def base64_to_tensor(s: str) -> torch.Tensor:
    raw = base64.b64decode(s)
    return load(raw)["t"]


def _maybe_serialize(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return {
            "__tensor__": True,
            "data": tensor_to_base64(obj),
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    if isinstance(obj, dict):
        return {k: _maybe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_maybe_serialize(v) for v in obj]
    return obj


def _maybe_deserialize(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get("__tensor__"):
            return base64_to_tensor(obj["data"])
        return {k: _maybe_deserialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_maybe_deserialize(v) for v in obj]
    return obj
