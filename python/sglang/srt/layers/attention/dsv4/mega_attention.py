from __future__ import annotations

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)

HEAD_DIM = 512


def interleave_query_rows(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    return x.view(num_heads, 32, 16, -1).transpose(0, 1).reshape(x.shape)


def pack_query(q: torch.Tensor, padded_heads: int) -> torch.Tensor:
    tokens, heads, _ = q.shape
    out = q.new_zeros(tokens, 32, padded_heads, 16)
    out[:, :, :heads] = q.view(tokens, heads, 32, 16).transpose(1, 2)
    return out.view(tokens, padded_heads, HEAD_DIM)


def interleave_output_columns(w: torch.Tensor) -> torch.Tensor:
    return w.view(*w.shape[:-1], 8, 16, 32).transpose(-3, -2).reshape(w.shape)


def quantize_wo_a(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups, rank, _ = weight.shape
    return sglang_per_token_group_quant_fp8(
        weight.contiguous(),
        group_size=32,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        masked_m=torch.full((groups,), rank, dtype=torch.int32, device=weight.device),
    )
