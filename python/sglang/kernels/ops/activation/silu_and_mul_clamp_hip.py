"""ROCm SwiGLU with clamped branches in one launch, optionally quantized onto the fp8 grid."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
    Fp8GridActivation,
    Mxfp8Activation,
    fp8_grid_quant,
    fp8_grid_round,
)


@triton.jit
def _silu_and_mul_clamp_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    M,
    inter_size,
    stride_xm,
    stride_om,
    stride_sm,
    limit,
    eps,
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
        q8, e8 = fp8_grid_quant(tl.reshape(y, (BLOCK_I // 32, 32)), eps)
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
                fp8_grid_round(tl.reshape(y, (BLOCK_I // 32, 32)), eps), (BLOCK_I,)
            )
        tl.store(
            out_ptr + pid_m * stride_om + offs,
            y.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def silu_and_mul_clamp_fp8_grid_supported(intermediate: int) -> bool:
    """The fp8-grid epilogue needs whole 32-wide groups inside one program."""
    return intermediate % 32 == 0 and intermediate <= 1024


def silu_and_mul_clamp_triton(
    gate_up: torch.Tensor,
    swiglu_limit: float,
    fp8_grid: bool = False,
    eps: float = 1e-10,
    emit_fp8: bool = False,
):
    """gate_up [M, 2 * inter_size] -> [M, inter_size] = silu(min(g, lim)) * clamp(u, -lim, lim), as
    ``Fp8GridActivation`` with ``fp8_grid`` or ``Mxfp8Activation`` with ``emit_fp8``."""
    assert gate_up.dim() == 2 and gate_up.shape[1] % 2 == 0, gate_up.shape
    M, N = gate_up.shape
    inter_size = N // 2
    fp8_grid = fp8_grid or emit_fp8
    if fp8_grid:
        assert silu_and_mul_clamp_fp8_grid_supported(inter_size), (
            f"inter_size {inter_size}: the fp8-grid epilogue needs a multiple of 32 up to 1024"
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
        result = Mxfp8Activation(out, scale)
    else:
        scale = None
        result = Fp8GridActivation(out) if fp8_grid else out
    if M == 0:
        return result
    block_i = 1024 if inter_size >= 1024 else triton.next_power_of_2(inter_size)
    _silu_and_mul_clamp_kernel[(M, triton.cdiv(inter_size, block_i))](
        gate_up,
        out,
        scale if scale is not None else out,
        M,
        inter_size,
        gate_up.stride(0),
        out.stride(0),
        scale.stride(0) if scale is not None else 0,
        float(swiglu_limit),
        float(eps),
        BLOCK_I=block_i,
        FP8_GRID=fp8_grid,
        EMIT_FP8=emit_fp8,
        num_warps=4,
    )
    return result
