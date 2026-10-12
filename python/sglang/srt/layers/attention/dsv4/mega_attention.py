"""Load-time WO-A preparation and projection of fused attention's MXFP8 output."""

from __future__ import annotations

import torch

from sglang.kernels.ops.attention.dsv4.mega_layout import permute_mega_wo_a
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)


def prepare_mega_wo_a(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep the checkpoint parameter unchanged. Each reload rebuilds this derived
    # pair, and the ordinary/draft path can still consume checkpoint-order WO-A.
    permuted = permute_mega_wo_a(weight)
    return sglang_per_token_group_quant_fp8(
        permuted,
        group_size=32,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )


def project_mega_attention(
    data: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    import deep_gemm

    tokens, groups, rank = data.shape[0], weight.shape[0], weight.shape[1]
    output = torch.empty(
        (tokens, groups, rank), dtype=torch.bfloat16, device=data.device
    )
    if tokens:
        # The fused kernel already applied inverse RoPE and produced packed,
        # TMA-aligned per-32 scales. No cast/quantization/repacking belongs here.
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (data[:, :groups], scale[:, :groups]),
            (weight, weight_scale),
            output,
            recipe=(1, 1, 32),
        )
    return output
