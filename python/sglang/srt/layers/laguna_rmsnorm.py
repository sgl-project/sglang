"""Fused RMSNorm for Laguna's BF16 residual emulation on SM70.

Laguna was trained with a BF16 residual stream.  V100 cannot execute the
model's GEMMs in BF16, so the model keeps projection outputs in FP16 while
transporting the residual in FP32 and explicitly rounding each residual update
to BF16.  Expressing that sequence with eager PyTorch launches many tiny
kernels per norm, which dominates batch-one decode latency.  This module keeps
the same rounding points in one Triton kernel.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _round_fp32_to_bf16_rne(value):
    """Round FP32 to BF16 and return the rounded value represented as FP32."""
    bits = value.to(tl.uint32, bitcast=True)
    # IEEE round-to-nearest-even before clearing the low 16 mantissa bits.
    rounded_bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def _laguna_silu_and_mul_sm70_kernel(
    output_ptr,
    output_scales_ptr,
    gate_up_ptr,
    n_cols: tl.constexpr,
    gate_up_scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply Laguna's BF16 SwiGLU semantics while transporting values in FP16."""
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    input_offsets = row * 2 * n_cols + offsets
    output_offsets = row * n_cols + offsets

    # W13 is evaluated from an input divided by gate_up_scale so its FP16
    # output cannot overflow. Restore the mathematical value in FP32, then
    # reproduce the BF16 rounding points used by the reference model.
    gate = tl.load(gate_up_ptr + input_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    up = tl.load(gate_up_ptr + input_offsets + n_cols, mask=mask, other=0.0).to(
        tl.float32
    )
    gate = _round_fp32_to_bf16_rne(gate * gate_up_scale)
    up = _round_fp32_to_bf16_rne(up * gate_up_scale)
    activated_gate = _round_fp32_to_bf16_rne(gate * tl.sigmoid(gate))
    output = _round_fp32_to_bf16_rne(activated_gate * up)
    # Keep headroom for the W2 GEMM and use a power-of-two scale so the
    # transport division itself introduces no extra rounding.
    max_abs = tl.max(tl.abs(output), axis=0)
    output_scale = tl.maximum(max_abs / 32752.0, 1.0)
    output_scale = tl.exp2(tl.ceil(tl.log2(output_scale)))
    tl.store(output_scales_ptr + row, output_scale)
    output /= output_scale
    tl.store(output_ptr + output_offsets, output, mask=mask)


@triton.jit
def _laguna_scale_output_sm70_kernel(
    output_ptr,
    row_scales_ptr,
    n_cols: tl.constexpr,
    residual_scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_offsets = row * n_cols + offsets
    value = tl.load(output_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    row_scale = tl.load(row_scales_ptr + row).to(tl.float32)
    value *= row_scale / residual_scale
    tl.store(output_ptr + row_offsets, value, mask=mask)


@triton.jit
def _laguna_rmsnorm_kernel(
    output_ptr,
    residual_output_ptr,
    x_ptr,
    residual_ptr,
    post_residual_ptr,
    weight_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    residual_scale: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_POST_RESIDUAL: tl.constexpr,
    CAST_X_BEFORE_OUT_MUL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_offsets = row * n_cols + offsets

    value = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(
            residual_ptr + row_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        value = value * residual_scale + residual
        if HAS_POST_RESIDUAL:
            value += tl.load(
                post_residual_ptr + row_offsets, mask=mask, other=0.0
            ).to(tl.float32)

        value = _round_fp32_to_bf16_rne(value)
        tl.store(residual_output_ptr + row_offsets, value, mask=mask)

    variance = tl.sum(value * value, axis=0) / n_cols
    normalized = value * tl.rsqrt(variance + eps)
    if CAST_X_BEFORE_OUT_MUL:
        normalized = _round_fp32_to_bf16_rne(normalized)

    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = _round_fp32_to_bf16_rne(weight)
    output = _round_fp32_to_bf16_rne(normalized * weight)
    tl.store(output_ptr + row_offsets, output, mask=mask)


def laguna_rmsnorm_sm70(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual_scale: float,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
    cast_x_before_out_mul: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Apply Laguna RMSNorm with its BF16 rounding points in one kernel."""
    if x.ndim != 2:
        raise ValueError(f"Laguna SM70 RMSNorm expects a 2D input, got {x.shape}.")
    if not x.is_contiguous():
        x = x.contiguous()
    if residual is not None and not residual.is_contiguous():
        residual = residual.contiguous()
    if post_residual_addition is not None and not post_residual_addition.is_contiguous():
        post_residual_addition = post_residual_addition.contiguous()

    n_rows, n_cols = x.shape
    if weight.ndim != 1 or weight.shape[0] != n_cols:
        raise ValueError(
            f"Laguna SM70 RMSNorm weight must have shape [{n_cols}], "
            f"got {tuple(weight.shape)}."
        )
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 65536:
        raise ValueError(
            f"Laguna SM70 RMSNorm only supports hidden sizes <= 65536, got {n_cols}."
        )

    output = torch.empty_like(x)
    has_residual = residual is not None
    # Laguna was trained with a BF16 residual stream.  Keep the SM70 transport
    # buffer in FP32 so values outside FP16's finite range survive the explicit
    # BF16 rounding performed by the kernel.  Using empty_like(residual) here
    # silently kept the stream in FP16 after the first layer and could turn
    # verifier features into inf/NaN before DFlash captured them.
    residual_output = (
        torch.empty_like(residual, dtype=torch.float32) if has_residual else output
    )
    residual_ptr = residual if has_residual else x
    post_residual_ptr = (
        post_residual_addition if post_residual_addition is not None else x
    )

    num_warps = max(min(triton.next_power_of_2(triton.cdiv(n_cols, 256)), 16), 4)
    _laguna_rmsnorm_kernel[(n_rows,)](
        output,
        residual_output,
        x,
        residual_ptr,
        post_residual_ptr,
        weight,
        n_cols=n_cols,
        eps=eps,
        residual_scale=residual_scale,
        HAS_RESIDUAL=has_residual,
        HAS_POST_RESIDUAL=post_residual_addition is not None,
        CAST_X_BEFORE_OUT_MUL=cast_x_before_out_mul,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    if has_residual:
        return output, residual_output
    return output


def laguna_silu_and_mul_sm70(
    gate_up: torch.Tensor,
    gate_up_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run overflow-safe BF16-equivalent SwiGLU on an SM70 FP16 tensor.

    ``gate_up`` must come from projecting an input divided by
    ``gate_up_scale``. Each output row is divided by the smallest power-of-two
    scale that keeps it in FP16 range. The returned row scales must be restored
    after W2, together with the residual-stream scale.
    """
    if gate_up.ndim < 2 or gate_up.shape[-1] % 2 != 0:
        raise ValueError(
            "Laguna SM70 SwiGLU expects [..., 2 * intermediate_size], "
            f"got {tuple(gate_up.shape)}."
        )
    if gate_up.dtype != torch.float16 or not gate_up.is_cuda:
        raise ValueError(
            "Laguna SM70 SwiGLU requires a CUDA FP16 input, "
            f"got device={gate_up.device}, dtype={gate_up.dtype}."
        )
    if gate_up_scale <= 0:
        raise ValueError(
            "Laguna SM70 SwiGLU gate_up_scale must be positive, "
            f"got {gate_up_scale}."
        )
    if not gate_up.is_contiguous():
        gate_up = gate_up.contiguous()

    n_cols = gate_up.shape[-1] // 2
    n_rows = gate_up.numel() // (2 * n_cols)
    output = torch.empty(
        (*gate_up.shape[:-1], n_cols),
        dtype=gate_up.dtype,
        device=gate_up.device,
    )
    output_scales = torch.empty(
        (n_rows,),
        dtype=torch.float32,
        device=gate_up.device,
    )
    block_size = triton.next_power_of_2(n_cols)
    num_warps = max(min(triton.next_power_of_2(triton.cdiv(n_cols, 256)), 16), 4)
    _laguna_silu_and_mul_sm70_kernel[(n_rows,)](
        output,
        output_scales,
        gate_up,
        n_cols=n_cols,
        gate_up_scale=gate_up_scale,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output, output_scales


def laguna_scale_output_sm70(
    output: torch.Tensor,
    row_scales: torch.Tensor,
    residual_scale: float,
) -> torch.Tensor:
    """Restore per-row SwiGLU scales while converting a W2 result to x/scale."""
    if output.ndim < 2 or output.dtype != torch.float16 or not output.is_cuda:
        raise ValueError(
            "Laguna SM70 W2 scaling requires a CUDA FP16 tensor with ndim >= 2, "
            f"got shape={tuple(output.shape)}, device={output.device}, "
            f"dtype={output.dtype}."
        )
    if residual_scale <= 0:
        raise ValueError(
            f"Laguna SM70 residual_scale must be positive, got {residual_scale}."
        )
    if not output.is_contiguous():
        raise ValueError("Laguna SM70 W2 scaling requires contiguous output.")

    n_cols = output.shape[-1]
    n_rows = output.numel() // n_cols
    if row_scales.numel() != n_rows:
        raise ValueError(
            "Laguna SM70 W2 row-scale count mismatch: "
            f"rows={n_rows}, scales={row_scales.numel()}."
        )
    block_size = triton.next_power_of_2(n_cols)
    num_warps = max(min(triton.next_power_of_2(triton.cdiv(n_cols, 256)), 16), 4)
    _laguna_scale_output_sm70_kernel[(n_rows,)](
        output,
        row_scales,
        n_cols=n_cols,
        residual_scale=residual_scale,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output
