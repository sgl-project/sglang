"""ROCm SwiGLU with clamped branches in one launch, optionally quantized onto the fp8 grid."""

from __future__ import annotations

from typing import Union

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
    Fp8GridActivation,
    Mxfp8Activation,
    fp8_grid_quant,
    fp8_grid_round,
)
from sglang.srt.environ import envs


@triton.jit
def _silu_and_mul_clamp_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    inter_size,
    stride_xm,
    stride_om,
    stride_sm,
    limit,
    BLOCK_I: tl.constexpr,
    FP8_GRID: tl.constexpr,
    EMIT_FP8: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = offs < inter_size
    g = tl.load(x_ptr + pid_m * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + pid_m * stride_xm + inter_size + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    g = tl.minimum(g, limit)
    u = tl.minimum(tl.maximum(u, -limit), limit)
    y = g / (1.0 + tl.exp(-g)) * u
    if EMIT_FP8:
        # per-32 fp8 codes plus ue8m0 exponent: the native MXFP8 route's operand
        y = y.to(tl.bfloat16).to(tl.float32)
        q8, e8 = fp8_grid_quant(tl.reshape(y, (BLOCK_I // 32, 32)))
        tl.store(
            out_ptr + pid_m * stride_om + offs, tl.reshape(q8, (BLOCK_I,)), mask=mask
        )
        goffs = pid_i * (BLOCK_I // 32) + tl.arange(0, BLOCK_I // 32)
        tl.store(
            scale_ptr + pid_m * stride_sm + goffs,
            e8.to(tl.uint8),
            mask=goffs < inter_size // 32,
        )
    else:
        if FP8_GRID:
            # output dtype first, then the fp8 grid (the unfused order); padding never shares a group
            y = y.to(out_ptr.dtype.element_ty).to(tl.float32)
            y = tl.reshape(
                fp8_grid_round(tl.reshape(y, (BLOCK_I // 32, 32))), (BLOCK_I,)
            )
        tl.store(
            out_ptr + pid_m * stride_om + offs,
            y.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def silu_and_mul_clamp_fp8_grid_supported(
    intermediate: int, multi_block: bool = False
) -> bool:
    """The fp8-grid epilogue needs whole 32-wide groups inside one program; 1024-wide programs
    keep that for any multiple of 32 (multi_block)."""
    return intermediate % 32 == 0 and (multi_block or intermediate <= 1024)


def silu_and_mul_clamp_triton(
    gate_up: torch.Tensor,
    swiglu_limit: float,
    fp8_grid: bool = False,
    emit_fp8: bool = False,
) -> Union[torch.Tensor, Fp8GridActivation, Mxfp8Activation]:
    """gate_up [M, 2 * inter_size] -> [M, inter_size] = silu(min(g, lim)) * clamp(u, -lim, lim), as
    Fp8GridActivation with fp8_grid or Mxfp8Activation with emit_fp8."""
    assert gate_up.dim() == 2 and gate_up.shape[1] % 2 == 0, gate_up.shape
    assert gate_up.stride(1) == 1, "the kernel reads the up half at gate + inter_size"
    M, N = gate_up.shape
    inter_size = N // 2
    if (
        emit_fp8
        and envs.SGLANG_HIP_SILU_MXFP8_FLYDSL.get()
        and M >= envs.SGLANG_HIP_SILU_MXFP8_FLYDSL_MIN_M.get()
        and gate_up.dtype == torch.bfloat16
        and gate_up.stride(0) % 2 == 0
    ):
        from sglang.kernels.ops.activation.silu_and_mul_clamp_mxfp8_flydsl import (
            silu_and_mul_clamp_mxfp8_flydsl,
            silu_and_mul_clamp_mxfp8_supported,
        )

        if silu_and_mul_clamp_mxfp8_supported(inter_size):
            return silu_and_mul_clamp_mxfp8_flydsl(gate_up, swiglu_limit)
    fp8_grid = fp8_grid or emit_fp8
    if fp8_grid:
        assert silu_and_mul_clamp_fp8_grid_supported(inter_size, multi_block=True), (
            f"inter_size {inter_size}: the fp8-grid epilogue needs a multiple of 32"
        )
    out = torch.empty(
        (M, inter_size),
        dtype=torch.float8_e4m3fn if emit_fp8 else gate_up.dtype,
        device=gate_up.device,
    )
    if emit_fp8:
        scale = torch.empty(
            (M, inter_size // 32), dtype=torch.uint8, device=gate_up.device
        )
        activation = Mxfp8Activation(out, scale)
    else:
        scale = None
        activation = Fp8GridActivation(out) if fp8_grid else out
    if M == 0:
        return activation
    # 256-wide one-warp programs: wider ones mask most of the last block (2304 = 2.25 x 1024) and
    # the [G, 32] reshape of the fp8 epilogue gets slower with every extra warp
    block_i = min(256, triton.next_power_of_2(inter_size))
    _silu_and_mul_clamp_kernel[(M, triton.cdiv(inter_size, block_i))](
        gate_up,
        out,
        scale if scale is not None else out,
        inter_size,
        gate_up.stride(0),
        out.stride(0),
        scale.stride(0) if scale is not None else 0,
        float(swiglu_limit),
        BLOCK_I=block_i,
        FP8_GRID=fp8_grid,
        EMIT_FP8=emit_fp8,
        num_warps=1,
    )
    return activation
