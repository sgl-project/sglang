"""Lazy wrapper for the optional SM70 E5M2 cache writer."""

from __future__ import annotations

from pathlib import Path

import torch
from sglang.kernels.sm70_paths import sm70_prebuilt

_OP = None
_CHECKED = False
_SCALE_TENSORS = {}


def _get_op():
    global _OP, _CHECKED
    if _CHECKED:
        return _OP
    _CHECKED = True
    namespace = getattr(torch.ops, "sglang_sm70_turbomind", None)
    if namespace is None or not hasattr(namespace, "fp8_e5m2_cache_write"):
        library = Path(__file__).with_name("_sm70_turbomind_v100.so")
        if not library.is_file():
            library = sm70_prebuilt("_sm70_turbomind_v100.so")
        if not library.is_file():
            return None
        try:
            torch.ops.load_library(str(library))
        except (OSError, RuntimeError):
            return None
        namespace = torch.ops.sglang_sm70_turbomind
    _OP = getattr(namespace, "fp8_e5m2_cache_write", None)
    return _OP


def _as_scale_tensor(scale, device):
    if scale is None:
        return None
    if isinstance(scale, torch.Tensor):
        return scale
    value = float(scale)
    if value == 1.0:
        return None
    key = (device.index, value)
    tensor = _SCALE_TENSORS.get(key)
    if tensor is None:
        tensor = torch.tensor(value, dtype=torch.float32, device=device)
        _SCALE_TENSORS[key] = tensor
    return tensor


def write_fp8_e5m2_cache_sm70(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    locations: torch.Tensor,
    k_scale=None,
    v_scale=None,
) -> bool:
    op = _get_op()
    if op is None:
        return False
    if (
        key.device.type != "cuda"
        or torch.cuda.get_device_capability(key.device) != (7, 0)
        or key.dtype != torch.float16
        or value.dtype != torch.float16
        or key_cache.dtype != torch.uint8
        or value_cache.dtype != torch.uint8
        or key_cache.ndim != 3
        or value_cache.ndim != 3
        or locations.dtype != torch.int64
    ):
        return False
    for scale in (k_scale, v_scale):
        if isinstance(scale, torch.Tensor) and (
            scale.device != key.device
            or scale.dtype != torch.float32
            or scale.numel() != 1
        ):
            return False
    op(
        key,
        value,
        key_cache,
        value_cache,
        locations,
        _as_scale_tensor(k_scale, key.device),
        _as_scale_tensor(v_scale, key.device),
    )
    return True
