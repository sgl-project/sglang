from __future__ import annotations

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)


def pack_mega_query(q: torch.Tensor, padded_heads: int) -> torch.Tensor:
    tokens, heads, _ = q.shape
    out = q.new_zeros(tokens, 32, padded_heads, 16)
    out[:, :, :heads] = q.view(tokens, heads, 32, 16).transpose(1, 2)
    return out.view(tokens, padded_heads, 512)


def permute_mega_wq_b(linear: torch.nn.Module, num_heads: int) -> bool:
    method = linear.quant_method
    backend = getattr(method, "mxfp8_dense_backend", None)
    if (
        num_heads not in (64, 128)
        or not getattr(method, "use_mxfp8", False)
        or getattr(method, "use_marlin", False)
        or backend is None
        or not (backend.is_flashinfer_cutlass() or backend.is_flashinfer_cutedsl())
    ):
        return False
    weight, scale = linear.weight, linear.weight_scale_inv
    rows, cols = weight.shape
    if (
        rows != num_heads * 512
        or weight.dtype != torch.float8_e4m3fn
        or scale.dtype != torch.uint8
        or scale.shape != (rows, cols // 32)
    ):
        return False
    for tensor in (weight, scale):
        tensor.data.copy_(
            tensor.data.view(num_heads, 32, 16, -1).transpose(0, 1).reshape(rows, -1)
        )
    method._process_mxfp8_linear_weight_scale(linear)
    return True


def prepare_mega_wo_a(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups, rank, hidden = weight.shape
    permuted = (
        weight.view(groups, rank, 8, 16, 32).transpose(2, 3).reshape(weight.shape)
    )
    return sglang_per_token_group_quant_fp8(
        permuted,
        group_size=32,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        masked_m=torch.full((groups,), rank, dtype=torch.int32, device=weight.device),
    )
