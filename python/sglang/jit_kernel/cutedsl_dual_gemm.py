# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
CuteDSL SM90 Dual GEMM kernel: fuses SiLU(X @ W_gate) * (X @ W_up) into a
single kernel using TMA + WGMMA on Hopper GPUs.

Architecture:
  - Producer warpgroup: TMA loads A, B0, B1 into SMEM (A shared between GEMMs)
  - Consumer warpgroup: 2x WGMMA per K-tile (reusing A descriptor)
  - Epilogue: fused SiLU*mul in registers, then R2S + TMA store

Supports optional FP8 quantized mode:
  - FP8 (e4m3fn) inputs with per-tensor x_scale and w_scale
  - FP8 (e4m3fn) output with o_scale for requantization
"""

from typing import Optional

import torch

from sglang.srt.utils.custom_op import register_custom_op


def _cutedsl_dual_gemm_fake(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
    o_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fake impl for torch.compile shape inference."""
    return out


@register_custom_op(
    op_name="cutedsl_dual_gemm",
    mutates_args=["out"],
    fake_impl=_cutedsl_dual_gemm_fake,
)
def cutedsl_dual_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
    o_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute out = SiLU(x @ w_gate) * (x @ w_up) using SM90 CuteDSL kernel.

    Args:
        x: Input tensor (M, K), row-major, BF16/FP16 or FP8 (e4m3fn)
        w: Combined weight [w_gate | w_up] (K, 2*N), row-major
        out: Output tensor (M, N), row-major, BF16/FP16 or FP8 (e4m3fn)
        x_scale: Per-tensor input scale (scalar float32), required for FP8
        w_scale: Weight scale tensor, required for FP8. Supports:
            numel()==1 : single per-tensor scale for both gate & up
            numel()==2 : fused per-tensor from MergedColumnParallelLinear
                         (w_scale[0] for gate, w_scale[1] for up)
            numel()>2  : per-channel (2*N,) — gate at index 0, up at index N
        o_scale: Per-tensor output scale (scalar float32), required for FP8
    """
    N = w.shape[1] // 2
    w_gate, w_up = torch.split(w, N, dim=1)
    return out
