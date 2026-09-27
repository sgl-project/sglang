# SPDX-License-Identifier: Apache-2.0
"""RMSNorm fused with the fp8 e4m3 fake-quant of the gfx950 bf16-dequant route: the row is
normalized in fp32, rounded to bf16 like the unfused norm, then put on the fp8 grid with the
per-32 ue8m0 rule of ``fake_quant_fp8_activation``; the un-quantized bf16 norm output is written
too when a second consumer needs it. One program per row with a reduction tree fixed by the row
width, so a row's result is batch-invariant and repeatable; only the fp32 sum-of-squares order
differs from the aiter / torch norm.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

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
def _rmsnorm_fake_quant_fp8_kernel(
    x_ptr,
    w_ptr,
    res_ptr,
    out_fq_ptr,
    out_norm_ptr,
    out_scale_ptr,
    K,
    stride_xm,
    stride_rm,
    stride_fm,
    stride_nm,
    stride_sm,
    eps,
    quant_eps,
    HAS_RESIDUAL: tl.constexpr,
    WRITE_NORM: tl.constexpr,
    EMIT_FP8: tl.constexpr,
    CHUNK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    """One program per row: (residual add,) RMSNorm in fp32, bf16 rounding, then
    the per-32 ue8m0 fp8 e4m3 quantize-dequantize of ``fake_quant_fp8_activation``."""
    row = tl.program_id(0).to(tl.int64)
    base = tl.arange(0, CHUNK)
    x_row = x_ptr + row * stride_xm
    res_row = res_ptr + row * stride_rm

    # Pass 1: sum of squares of the (residual-added) row in fp32.
    acc = tl.zeros((CHUNK,), dtype=tl.float32)
    for c in tl.static_range(NUM_CHUNKS):
        offs = c * CHUNK + base
        mask = offs < K
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            r = tl.load(res_row + offs, mask=mask, other=0.0).to(tl.float32)
            x = x + r
        acc += x * x
    var = tl.sum(acc, axis=0) / K
    rstd = 1.0 / tl.sqrt_rn(var + eps)

    # pass 2 recomputes the fp32 sum: its bf16 rounding is the new residual, the norm reads the fp32 value
    for c in tl.static_range(NUM_CHUNKS):
        offs = c * CHUNK + base
        mask = offs < K
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            r = tl.load(res_row + offs, mask=mask, other=0.0).to(tl.float32)
            x = x + r
            tl.store(res_row + offs, x.to(res_ptr.dtype.element_ty), mask=mask)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # round to bf16 first so the fake-quant sees the values the unfused route would have loaded
        y = (x * rstd * w).to(out_norm_ptr.dtype.element_ty)
        if WRITE_NORM:
            tl.store(out_norm_ptr + row * stride_nm + offs, y, mask=mask)
        yg = tl.reshape(y.to(tl.float32), (CHUNK // 32, 32))
        if EMIT_FP8:
            # fp8 codes plus ue8m0 exponent: the native MXFP8 operand, dequantizing exactly to the fake-quant below
            q8, e8 = fp8_grid_quant(yg, quant_eps)
            tl.store(
                out_fq_ptr + row * stride_fm + offs, tl.reshape(q8, (CHUNK,)), mask=mask
            )
            goffs = c * (CHUNK // 32) + tl.arange(0, CHUNK // 32)
            tl.store(
                out_scale_ptr + row * stride_sm + goffs,
                e8.to(tl.uint8),
                mask=goffs < K // 32,
            )
        else:
            q = fp8_grid_round(yg, quant_eps)
            tl.store(
                out_fq_ptr + row * stride_fm + offs,
                tl.reshape(q, (CHUNK,)).to(out_fq_ptr.dtype.element_ty),
                mask=mask,
            )


def _row_major_2d(x: torch.Tensor) -> torch.Tensor:
    """A row-major view of ``x`` (rows may be strided, e.g. a column slice of a
    wider projection output); copies only when the last dim is not unit-stride."""
    return x if x.stride(1) == 1 else x.contiguous()


def rmsnorm_fake_quant_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: Optional[torch.Tensor] = None,
    return_norm: bool = True,
    quant_eps: float = 1e-10,
    emit_fp8: bool = False,
) -> Tuple[Union[Fp8GridActivation, Mxfp8Activation], Optional[torch.Tensor]]:
    """``fake_quant_fp8_activation(RMSNorm(x))`` in one launch.

    ``x`` is ``[M, K]`` bf16/fp16 with ``K % 32 == 0``; ``weight`` is ``[K]``. With ``residual`` the
    sum ``x + residual`` is written back into ``residual`` and normalized (the ``fused_add_rmsnorm``
    contract). Returns ``(fake_quant, norm)``: the fp8-grid activation as ``Fp8GridActivation``, or
    as ``Mxfp8Activation`` (fp8 codes + ue8m0 bytes) with ``emit_fp8``, and the bf16 norm output
    (``None`` unless ``return_norm``).
    """
    assert x.dim() == 2 and x.shape[-1] % 32 == 0, x.shape
    assert weight.dim() == 1 and weight.shape[0] == x.shape[-1], weight.shape
    assert weight.dtype == x.dtype, (weight.dtype, x.dtype)
    x = _row_major_2d(x)
    weight = weight.contiguous()
    M, K = x.shape
    out_fq = torch.empty(
        (M, K), dtype=torch.float8_e4m3fn if emit_fp8 else x.dtype, device=x.device
    )
    out_scale = (
        torch.empty((M, K // 32), dtype=torch.uint8, device=x.device)
        if emit_fp8
        else None
    )
    out_norm = (
        torch.empty((M, K), dtype=x.dtype, device=x.device) if return_norm else None
    )
    if residual is not None:
        assert residual.shape == x.shape and residual.dtype == x.dtype
        assert residual.stride(1) == 1, "residual is updated in place"

    def wrap():
        if emit_fp8:
            return Mxfp8Activation(out_fq, out_scale)
        return Fp8GridActivation(out_fq)

    if M == 0:
        return wrap(), out_norm
    # the chunk (and so the reduction tree) depends on K alone, so a row's result is the same at every batch size
    CHUNK = min(max(32, triton.next_power_of_2(K)), 2048)
    _rmsnorm_fake_quant_fp8_kernel[(M,)](
        x,
        weight,
        residual if residual is not None else x,
        out_fq,
        out_norm if out_norm is not None else x,
        out_scale if out_scale is not None else out_fq,
        K,
        x.stride(0),
        residual.stride(0) if residual is not None else 0,
        out_fq.stride(0),
        out_norm.stride(0) if out_norm is not None else 0,
        out_scale.stride(0) if out_scale is not None else 0,
        eps,
        quant_eps,
        HAS_RESIDUAL=residual is not None,
        WRITE_NORM=out_norm is not None,
        EMIT_FP8=emit_fp8,
        CHUNK=CHUNK,
        NUM_CHUNKS=triton.cdiv(K, CHUNK),
        num_warps=4,
    )
    return wrap(), out_norm
