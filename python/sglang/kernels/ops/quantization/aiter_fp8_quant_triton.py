# SPDX-License-Identifier: Apache-2.0
"""Triton epilogues that quantize to FP8 like aiter's HIP quant kernels on gfx95, so a
fused producer hands the GEMM the bytes and scales the separate quant kernel writes:
per token as ``per_token_quant_hip`` for the ptpc GEMM, per 1 x 128 as
``get_hip_quant(QuantType.per_1x128)`` for the block-scale GEMMs."""

from typing import Optional

import torch
import triton
import triton.language as tl

_FP8_MAX = 448.0


@triton.jit
def _rcp(x):
    # aiter's per-token kernel multiplies by the approximate v_rcp_f32, not 1 / scale.
    return tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1",
        constraints="=v,v",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def fp8_per_token_quant(x, AXIS: tl.constexpr, FP8_MAX: tl.constexpr = _FP8_MAX):
    """Quantize fp32 ``x`` (the values the separate quant kernel reads, padded with
    zeros) along its token axis ``AXIS``; returns fp8 values and per-token scales."""
    scale = tl.max(tl.abs(x), axis=AXIS) * (1.0 / FP8_MAX)
    q = tl.clamp(x * tl.expand_dims(_rcp(scale), AXIS), -FP8_MAX, FP8_MAX)
    return q.to(tl.float8e4nv), scale


@triton.jit
def fp8_group_quant(x, GROUP: tl.constexpr, FP8_MAX: tl.constexpr = _FP8_MAX):
    """Quantize fp32 ``x`` (``[n]`` or ``[rows, n]``) per ``GROUP`` consecutive values of
    its last axis; returns fp8 values shaped like ``x`` and ``[..., n / GROUP]`` scales."""
    if len(x.shape) == 1:
        groups = tl.reshape(x, [x.shape[0] // GROUP, GROUP])
    else:
        groups = tl.reshape(x, [x.shape[0], x.shape[1] // GROUP, GROUP])
    AXIS: tl.constexpr = len(x.shape)
    # The group kernel starts its amax at 1e-10 and divides exactly (v_div_fixup).
    scale = tl.maximum(tl.max(tl.abs(groups), axis=AXIS), 1e-10) * (1.0 / FP8_MAX)
    inv = tl.math.div_rn(tl.full(scale.shape, 1.0, tl.float32), scale)
    q = tl.clamp(groups * tl.expand_dims(inv, AXIS), -FP8_MAX, FP8_MAX)
    return tl.reshape(q.to(tl.float8e4nv), x.shape), scale


@triton.jit
def swiglu_oai_fp8_rows(
    x_ptr,
    q_ptr,
    scale_ptr,
    rows,
    n,
    stride_row,
    pid,
    ALPHA: tl.constexpr,
    LIMIT: tl.constexpr,
    GROUP: tl.constexpr,
    TRANSPOSE_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Rows [pid * BLOCK_M, (pid + 1) * BLOCK_M) of swiglu_oai_fp8."""
    r = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    c = tl.arange(0, BLOCK_N)
    mask = (r < rows)[:, None] & (c < n)[None, :]
    base = x_ptr + r[:, None].to(tl.int64) * stride_row + c[None, :]
    gate = tl.load(base, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(base + n, mask=mask, other=0.0).to(tl.float32)
    # the op order Inductor emits for swiglu_no_interleaved_with_alpha_and_limit
    gate = tl.minimum(gate, LIMIT, propagate_nan=tl.PropagateNan.ALL)
    up = tl.maximum(up, -LIMIT, propagate_nan=tl.PropagateNan.ALL)
    up = tl.minimum(up, LIMIT, propagate_nan=tl.PropagateNan.ALL)
    act = gate * tl.sigmoid(gate * ALPHA) * (up + 1.0)
    # the separate quant reads the bf16 activation; padding stays 0
    act = tl.where(mask, act.to(tl.bfloat16).to(tl.float32), 0.0)
    if GROUP == 0:
        q, scale = fp8_per_token_quant(act, 1)
        tl.store(scale_ptr + r, scale, mask=r < rows)
    else:
        q, scale = fp8_group_quant(act, GROUP)
        g = tl.arange(0, BLOCK_N // GROUP)[None, :]
        if TRANSPOSE_SCALE:
            s_offs = g * rows + r[:, None]
        else:
            s_offs = r[:, None] * (n // GROUP) + g
        tl.store(scale_ptr + s_offs, scale, mask=(r < rows)[:, None] & (g < n // GROUP))
    tl.store(q_ptr + r[:, None].to(tl.int64) * n + c[None, :], q, mask=mask)


@triton.jit
def _swiglu_oai_fp8_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    rows,
    n,
    stride_row,
    ALPHA: tl.constexpr,
    LIMIT: tl.constexpr,
    GROUP: tl.constexpr,
    TRANSPOSE_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    swiglu_oai_fp8_rows(
        x_ptr,
        q_ptr,
        scale_ptr,
        rows,
        n,
        stride_row,
        tl.program_id(0),
        ALPHA,
        LIMIT,
        GROUP,
        TRANSPOSE_SCALE,
        BLOCK_M,
        BLOCK_N,
    )


def swiglu_oai_fp8_outputs(
    hidden: torch.Tensor,
    group_size: Optional[int] = None,
    transpose_scale: bool = False,
):
    """(q, scale, block_m, block_n) of swiglu_oai_fp8: its outputs as allocated and the
    rows / columns per program, which also size the program grid."""
    assert hidden.dim() == 2 and hidden.dtype == torch.bfloat16
    assert hidden.stride(1) == 1 and hidden.shape[1] % 2 == 0
    rows, n = hidden.shape[0], hidden.shape[1] // 2
    q = torch.empty(rows, n, dtype=torch.float8_e4m3fn, device=hidden.device)
    group = group_size or 0
    if group:
        assert n % group == 0
        groups = n // group
        scale = torch.empty(
            (groups, rows) if transpose_scale else (rows, groups),
            dtype=torch.float32,
            device=hidden.device,
        )
    else:
        scale = torch.empty(rows, 1, dtype=torch.float32, device=hidden.device)
    block_n = triton.next_power_of_2(n)
    # Arbitrary: about 2k elements per four-wave program.
    block_m = max(1, min(8, 2048 // block_n))
    return q, scale, block_m, block_n


def swiglu_oai_fp8(
    hidden: torch.Tensor,
    alpha: float,
    limit: float,
    group_size: Optional[int] = None,
    transpose_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``hidden [rows, 2n]`` bf16 (gate | up) -> quantized SwiGLU-OAI, equal to quantizing
    ``swiglu_no_interleaved_with_alpha_and_limit(hidden)`` with aiter: per token
    (``group_size`` None, scale ``[rows, 1]``) or per ``1 x group_size`` (scale
    ``[rows, n / group_size]``, column-major when ``transpose_scale``)."""
    q, scale, block_m, block_n = swiglu_oai_fp8_outputs(
        hidden, group_size, transpose_scale
    )
    rows, n = q.shape
    group = group_size or 0
    if rows > 0:
        _swiglu_oai_fp8_kernel[(triton.cdiv(rows, block_m),)](
            hidden,
            q,
            scale,
            rows,
            n,
            hidden.stride(0),
            ALPHA=alpha,
            LIMIT=limit,
            GROUP=group,
            TRANSPOSE_SCALE=transpose_scale,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
        )
    return q, scale.t() if group and transpose_scale else scale
