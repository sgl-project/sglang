# SPDX-License-Identifier: Apache-2.0
"""Ascend fusion boundaries for the MiniMax H3 transformer."""

from functools import lru_cache

import torch
from torch import nn

import sglang.multimodal_gen.envs as envs


class MiniMaxH3RMSNorm(nn.RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.device.type == "npu"
            and x.dtype in (torch.float16, torch.bfloat16)
            and self.weight is not None
            and self.weight.device == x.device
            and self.weight.dtype == x.dtype
            and x.numel() > 0
            and hasattr(torch.ops.npu, "npu_rms_norm")
        ):
            eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
            return torch.ops.npu.npu_rms_norm(x, self.weight, eps)[0]
        return super().forward(x)


@lru_cache(maxsize=1)
def _triton_ops():
    try:
        import triton

        if "ascend" not in triton.backends.backends:
            return None
        from sglang.kernels.ops.diffusion.norm import indexed_modulation_npu

        return indexed_modulation_npu
    except ImportError:
        return None


def _rows_supported(x: torch.Tensor) -> bool:
    return (
        envs.SGLANG_DIFFUSION_MINIMAX_H3_NPU_TRITON
        and x.device.type == "npu"
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.ndim == 2
        and 0 < x.shape[1] <= 8192
        and x.shape[0] > 0
        and x.is_contiguous()
    )


def _table_supported(table: torch.Tensor, x: torch.Tensor) -> bool:
    return (
        table.device == x.device
        and table.dtype == x.dtype
        and table.ndim == 2
        and table.shape[0] > 0
        and table.shape[1] == x.shape[1]
        and table.stride(1) == 1
    )


def _indices_supported(indices: torch.Tensor, x: torch.Tensor) -> bool:
    return (
        indices.device == x.device
        and indices.dtype in (torch.int32, torch.int64)
        and indices.ndim == 1
        and indices.numel() == x.shape[0]
    )


def rmsnorm_modulate(
    x: torch.Tensor,
    norm: nn.RMSNorm,
    shift: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
    *,
    quantize: bool = False,
    keep_float: bool = True,
):
    if (
        not _rows_supported(x)
        or norm.weight is None
        or norm.weight.device != x.device
        or norm.weight.dtype != x.dtype
        or norm.weight.shape != (x.shape[1],)
        or not norm.weight.is_contiguous()
    ):
        return None
    if (shift is None) != (scale is None) or (shift is None) != (indices is None):
        return None
    if shift is not None and (
        scale is None
        or indices is None
        or not _table_supported(shift, x)
        or not _table_supported(scale, x)
        or shift.shape != scale.shape
        or not _indices_supported(indices, x)
    ):
        return None
    ops = _triton_ops()
    if ops is None:
        return None
    eps = norm.eps if norm.eps is not None else torch.finfo(x.dtype).eps
    # Wide rows exhaust UB when modulation and INT8 rounding are fused.
    split_quant = quantize and x.shape[1] > 4096
    normalized, data, scales = ops.rmsnorm_modulate(
        x,
        norm.weight,
        shift,
        scale,
        indices,
        eps,
        quantize and not split_quant,
        keep_float or split_quant,
    )
    if split_quant:
        data, scales = torch.ops.npu.npu_dynamic_quant(normalized, dst_type=torch.int8)
    return normalized if keep_float else None, data, scales


def indexed_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
    *,
    allow_inplace: bool,
):
    if (
        not _rows_supported(x)
        or other.shape != x.shape
        or other.device != x.device
        or other.dtype != x.dtype
        or not other.is_contiguous()
        or not _table_supported(gate, x)
        or not _indices_supported(indices, x)
    ):
        return None
    ops = _triton_ops()
    if ops is None:
        return None
    return ops.indexed_gate(x, gate, other, indices, allow_inplace)
