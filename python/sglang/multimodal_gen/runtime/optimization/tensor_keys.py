# SPDX-License-Identifier: Apache-2.0
"""Stable tensor-aware cache keys and short shape summaries for autotune caches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def value_key(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_key(value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(value_key(item) for item in value))
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (str(key), value_key(item))
                for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
            ),
        )
    return ("object", type(value).__module__, type(value).__qualname__)


def tensor_key(tensor: torch.Tensor) -> tuple:
    device_index = tensor.device.index
    capability = None
    if tensor.device.type == "cuda":
        device_index = (
            torch.cuda.current_device() if device_index is None else device_index
        )
        capability = torch.cuda.get_device_capability(device_index)
    return (
        "tensor",
        tensor.device.type,
        device_index,
        capability,
        str(tensor.dtype),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.requires_grad,
    )


def value_summary(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"{tuple(value.shape)}:{value.dtype}:{value.device.type}"
    if isinstance(value, (list, tuple)):
        tensor_items = [value_summary(item) for item in value]
        tensor_items = [item for item in tensor_items if item]
        if tensor_items:
            return f"{type(value).__name__}[{', '.join(tensor_items[:4])}]"
    if isinstance(value, Mapping):
        tensor_items = [
            f"{key}:{value_summary(item)}"
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        ]
        tensor_items = [item for item in tensor_items if not item.endswith(":")]
        if tensor_items:
            return f"dict[{', '.join(tensor_items[:4])}]"
    return ""
