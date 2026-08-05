"""Fused Add + RMSNorm + FP8 per-token quantization JIT kernel.

Replaces two separate kernels:
  1. FusedAddRMSNorm(input, residual, weight)  — residual += input, normed = norm(residual)
  2. per_token_quant_fp8(normed, fp8_out, scale) — per-token FP8 quantization

with a single kernel that keeps normed values in registers, avoiding
an intermediate BF16 write+read of hidden_dim per token.

Outputs:
  - residual: updated in-place
  - output_bf16: normed BF16 (for non-FP8 paths, e.g. router/gate)
  - output_fp8: normed FP8 (for FP8 GEMM input)
  - output_scales: [M, 1] per-token scale
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "fused_add_rmsnorm_per_token_quant",
        *args,
        cuda_files=["elementwise/fused_add_rmsnorm_per_token_quant.cuh"],
        cuda_wrappers=[
            (
                "fused_add_rmsnorm_per_token_quant",
                f"fused_add_rmsnorm_per_token_quant<{args}>",
            ),
            (
                "fused_rmsnorm_per_token_quant",
                f"fused_rmsnorm_per_token_quant<{args}>",
            ),
        ],
    )


def fused_add_rmsnorm_per_token_quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused residual-add + RMSNorm + per-token FP8 quantization.

    Args:
        input: [M, D] bf16 — hidden states
        residual: [M, D] bf16 — residual (updated in-place)
        weight: [D] bf16 — RMSNorm weight
        eps: RMSNorm epsilon

    Returns:
        output_bf16: [M, D] bf16 — normed output
        output_fp8: [M, D] fp8_e4m3 — quantized normed output
        output_scales: [M, 1] float32 — per-token scales
    """
    assert input.is_cuda and input.dtype == torch.bfloat16
    assert residual.shape == input.shape and residual.dtype == torch.bfloat16
    assert weight.ndim == 1 and weight.shape[0] == input.shape[1]
    assert input.shape[1] % 8 == 0, "hidden_dim must be divisible by 8"

    m, d = input.shape
    output_bf16 = torch.empty_like(input)
    output_fp8 = torch.empty((m, d), dtype=torch.float8_e4m3fn, device=input.device)
    output_scales = torch.empty((m, 1), dtype=torch.float32, device=input.device)

    if m > 0:
        module = _jit_module(input.dtype)
        module.fused_add_rmsnorm_per_token_quant(
            input.contiguous(),
            residual.contiguous(),
            weight.contiguous(),
            output_bf16,
            output_fp8,
            output_scales,
            float(eps),
        )

    return output_bf16, output_fp8, output_scales


def fused_rmsnorm_per_token_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused RMSNorm + per-token FP8 quantization (no residual add).

    Args:
        input: [M, D] bf16
        weight: [D] bf16 — RMSNorm weight
        eps: RMSNorm epsilon

    Returns:
        output_bf16: [M, D] bf16
        output_fp8: [M, D] fp8_e4m3
        output_scales: [M, 1] float32
    """
    assert input.is_cuda and input.dtype == torch.bfloat16
    assert weight.ndim == 1 and weight.shape[0] == input.shape[1]
    assert input.shape[1] % 8 == 0

    m, d = input.shape
    output_bf16 = torch.empty_like(input)
    output_fp8 = torch.empty((m, d), dtype=torch.float8_e4m3fn, device=input.device)
    output_scales = torch.empty((m, 1), dtype=torch.float32, device=input.device)

    if m > 0:
        module = _jit_module(input.dtype)
        module.fused_rmsnorm_per_token_quant(
            input.contiguous(),
            weight.contiguous(),
            output_bf16,
            output_fp8,
            output_scales,
            float(eps),
        )

    return output_bf16, output_fp8, output_scales
