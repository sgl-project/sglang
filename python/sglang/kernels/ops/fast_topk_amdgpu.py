# SPDX-License-Identifier: Apache-2.0
"""DeepSelect-compatible TopK over the AMDGPU radix topk (flash-float-jit-kernels).

YuE2's sampler needs ``topk(scores, k, sorted=True, indices_type=int64,
return_value=True)`` with ``k <= 100`` and shapes ``[B, V_pad]``. The AMDGPU
kernel supports k in {512, 1024, 2048}, so calls are padded to the next bucket
and sliced afterwards.

Key mapping: the kernel bins ``int(x)`` into 256 buckets, so scores are shifted
by ``hi - 255`` and clamped to [0, 255] (no rescaling). That keeps the
fractional part of every logit, and the kernel's 3 residual rounds resolve the
ordering inside the boundary bucket at 32-bit precision. ``-inf`` mask entries
land in bucket 0 and are never selected unless fewer than ``k`` tokens are
allowed. Values are gathered from the raw scores, so softmax is exact.

Every buffer is preallocated and keyed by shape; the call is HIP-graph
capturable (the kernel build must happen before capture, which
``is_deepselect_supported`` guarantees).
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch

_K_BUCKETS = (512, 1024, 2048)
_BUFFERS: dict[tuple, dict[str, torch.Tensor]] = {}


def _kernel():
    root = os.environ.get(
        "FLASH_FLOAT_JIT_ROOT", "/home/yiakwy/workspace/Github/flash-float-jit-kernels"
    )
    if root not in sys.path:
        sys.path.insert(0, root)
    from jit_kernel import topk_indexer  # noqa: PLC0415

    topk_indexer._jit_fast_topk_v3_module()  # build/load now, never in capture
    return topk_indexer.fast_topk_v3


def is_deepselect_supported(device: Optional[int] = None) -> bool:
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    try:
        arch = torch.cuda.get_device_properties(device or 0).gcnArchName.split(":")[0]
        if not arch.startswith("gfx"):
            return False
        _kernel()
        return True
    except Exception as exc:  # pragma: no cover - host dependent
        import logging

        logging.getLogger(__name__).warning("amdgpu topk unavailable: %s", exc)
        return False


def _k_pad(k: int) -> int:
    for bucket in _K_BUCKETS:
        if k <= bucket:
            return bucket
    raise RuntimeError(f"topk {k} exceeds the AMDGPU kernel's 2048 limit")


def _slots(rows: int, width: int, k_pad: int, device: torch.device) -> dict[str, torch.Tensor]:
    key = (rows, width, k_pad, device.index, device.type)
    slots = _BUFFERS.get(key)
    if slots is None:
        i64, f32 = torch.int64, torch.float32
        slots = dict(
            out32=torch.full((rows, k_pad), -1, dtype=torch.int32, device=device),
            idx64=torch.empty((rows, k_pad), dtype=i64, device=device),
            vals=torch.empty((rows, k_pad), dtype=f32, device=device),
            vals_sorted=torch.empty((rows, k_pad), dtype=f32, device=device),
            perm=torch.empty((rows, k_pad), dtype=i64, device=device),
            idx_sorted=torch.empty((rows, k_pad), dtype=i64, device=device),
            norm=torch.empty((rows, width), dtype=f32, device=device),
            hi=torch.empty((rows, 1), dtype=f32, device=device),
            hi_idx=torch.empty((rows, 1), dtype=i64, device=device),
            lengths=torch.full((rows,), width, dtype=torch.int32, device=device),
        )
        _BUFFERS[key] = slots
    return slots


def topk(
    input: torch.Tensor,
    topk: int,
    sorted: bool = False,
    begin=None,
    end=None,
    indices_type: torch.dtype = torch.int32,
    sorted_index: bool = False,
    hint=None,
    output_idx=None,
    output_idx_offset=None,
    idx_oob_fill_value: int = 2147483647,
    value_oob_fill_value: float = float("-inf"),
    return_value: bool = True,
    abort_when_nan_found: bool = True,
):
    if begin is not None or end is not None or hint is not None:
        raise ValueError("begin/end/hint are not supported by the AMDGPU backend")
    if input.dtype is not torch.float32 or input.ndim != 2 or input.stride(1) != 1:
        raise RuntimeError("input must be a [B, L] float32 tensor with stride(1) == 1")
    if sorted and sorted_index:
        raise RuntimeError("sorted and sorted_index cannot both be True")

    rows, width = input.shape
    k = int(topk)
    k_pad = _k_pad(k)
    b = _slots(rows, width, k_pad, input.device)

    # Shift-only normalization: hi -> 255, fractional parts preserved, -inf -> 0.
    torch.max(input, dim=-1, keepdim=True, out=(b["hi"], b["hi_idx"]))
    torch.sub(input, b["hi"], out=b["norm"])
    b["norm"].add_(255.0).clamp_(0.0, 255.0)

    _kernel()(b["norm"], b["lengths"], k_pad, out=b["out32"])
    b["idx64"].copy_(b["out32"])
    torch.gather(input, 1, b["idx64"], out=b["vals"])

    indices = b["idx64"]
    values = b["vals"]
    if sorted and return_value:
        torch.sort(b["vals"], dim=-1, descending=True, out=(b["vals_sorted"], b["perm"]))
        torch.gather(b["idx64"], 1, b["perm"], out=b["idx_sorted"])
        values, indices = b["vals_sorted"], b["idx_sorted"]

    values = values[:, :k] if return_value else None
    indices = indices[:, :k]
    if indices_type is torch.int32:
        indices = indices.to(torch.int32)
    if output_idx is not None:
        output_idx.copy_(indices)
        indices = output_idx
    return (values, indices) if return_value else (None, indices)
