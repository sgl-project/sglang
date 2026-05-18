"""
Pure-PyTorch reference implementation of SiluAndMulClampKernel::run.

Matches the CUDA kernel in:
  python/sglang/jit_kernel/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh

Algorithm per token row:
  1. Split input (M, 2*H) into gate (M, H) and up (M, H).
  2. Clamp in BF16 precision  (critical: matches DeepGEMM reference):
       gate = min(gate,  limit)          -- upper-clamp only
       up   = clamp(up, -limit, limit)   -- both sides
  3. SiLU(gate) * up  in fp32, cast back to the input dtype.

Reference:
  https://github.com/deepseek-ai/DeepGEMM/blob/7f2a703ed51ac1f7af07f5e1453b2d3267d37d50/
  deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L984-L997
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_and_mul_clamp_torch(
    input: torch.Tensor,       # (M, 2*H)  bf16 or fp16
    output: torch.Tensor,      # (M, H)    same dtype, pre-allocated
    swiglu_limit: float,
) -> None:
    """
    In-place fused SiLU-and-Mul with optional BF16 swiglu clamping.
    Writes result into *output*; returns nothing.

    Args:
        input:        (M, 2*H) tensor — gate and up concatenated on last dim.
        output:       (M, H)   tensor — pre-allocated output buffer.
        swiglu_limit: Clamping bound (positive scalar). Applied in BF16,
                      matching the CUDA __hmin2/__hmax2 semantics.
    """
    M, D = input.shape
    assert D % 2 == 0, "input last dim must be even (gate || up)"
    H = D // 2
    assert output.shape == (M, H), f"output must be ({M}, {H}), got {output.shape}"
    assert input.dtype in (torch.bfloat16, torch.float16), \
        "only bf16/fp16 supported (matches CUDA static_assert sizeof(DType)==2)"

    # ------------------------------------------------------------------
    # Step 1: Split into gate and up halves
    # Matches: gate_vec.load(input, bid*2+0); up_vec.load(input, bid*2+1)
    # in the CTA-tiled kernel where each CTA handles one token row.
    # ------------------------------------------------------------------
    gate = input[:, :H]   # (M, H)
    up   = input[:, H:]   # (M, H)

    # ------------------------------------------------------------------
    # Step 2: Clamp in BF16
    # CUDA (kApplySwigluLimit=true):
    #   gate = __hmin2(gate, {limit, limit})      → gate ≤ limit
    #   up   = __hmax2(up,  {-limit, -limit})     → up  ≥ -limit
    #   up   = __hmin2(up,  { limit,  limit})     → up  ≤  limit
    #
    # The comment in the CUDA source stresses that clamping MUST happen
    # in bf16 (not fp32) to match the DeepGEMM reference behaviour, so we
    # cast to bf16 before clamping even when the input is fp16.
    # ------------------------------------------------------------------
    gate_bf16 = gate.to(torch.bfloat16)
    up_bf16   = up.to(torch.bfloat16)

    gate_clamped = gate_bf16.clamp(max= swiglu_limit)                   # upper only
    up_clamped   = up_bf16.clamp(min=-swiglu_limit, max=swiglu_limit)   # both sides

    # ------------------------------------------------------------------
    # Step 3: SiLU(gate) * up  in fp32
    # CUDA silu_and_mul (kPrecise=true):
    #   silu0 = g0 / (1 + exp(-g0))
    #   val0  = silu0 * u0
    # ------------------------------------------------------------------
    gate_fp32 = gate_clamped.float()
    up_fp32   = up_clamped.float()

    silu_gate = gate_fp32 * torch.sigmoid(gate_fp32)   # equivalent to F.silu
    result    = silu_gate * up_fp32                    # (M, H)  fp32

    # ------------------------------------------------------------------
    # Step 4: Cast back to input dtype and write to output
    # Matches: tile.store(params.output, out, bid)
    # ------------------------------------------------------------------
    output.copy_(result.to(input.dtype))
