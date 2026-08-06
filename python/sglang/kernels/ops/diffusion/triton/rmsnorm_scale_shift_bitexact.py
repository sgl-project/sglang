# SPDX-License-Identifier: Apache-2.0
"""Bit-exact fused RMSNorm + adaLN scale/shift (optionally with a preceding
residual-gate add) for bf16 activations.

Replaces the eager ERNIE-Image adaLN chain

    ``norm(x) * (1 + scale) + shift``                    (4 kernels)
    ``res = residual + gate * update`` before the norm   (+1 kernel)

with one Triton kernel per site while reproducing the eager chain's rounding
*bit for bit* (``torch.equal``), so it can be wired unconditionally (no
quality gate), unlike a plain fp32 single-pass fusion which perturbs the
50-step denoising trajectory (T8: PSNR 18.83 dB at quality=high).

Numerics contract (each step matches the eager kernel boundary):

- ``RMSNorm.forward_cuda`` dispatches to ``sgl_kernel.rmsnorm`` ->
  flashinfer's CuTe-DSL ``RMSNormKernel``.  For contiguous bf16 rows with
  ``H == 64 * threads_per_row`` (threads_per_row 32 for H<=3072 else 64,
  cluster_n == 1) that kernel computes, per row:

  * thread ``tx`` owns columns ``{8*tpr*b + 8*tx + v : b, v in [0,8)}``;
    its fragment is ordered ``v`` fastest, then ``b``;
  * ``x_sq = x * x`` in fp32 (each square rounded separately, no FMA), then
    an *ordered* sequential fadd chain over the 64 fragment values
    (MLIR ``vector.reduction`` without reassoc);
  * warp reduction via ``shfl.bfly`` with offsets 1,2,4,8,16 == an
    adjacent-pairs fold tree; two warp sums are then added (tpr == 64);
  * ``rstd = rsqrt.approx.f32(sum_sq / H + eps)`` (``cute.math.rsqrt``
    with fastmath);
  * ``y = (bf16)(float(x) * rstd * (w + 0.0))`` -- one final rounding.

- The aten modulate chain rounds to bf16 after every op (fp32 opmath):
  ``round(1 + scale)``, ``round(y * that)``, ``round(prod + shift)``.

- The residual variant reproduces the eager pair ``round(gate * update)``,
  ``round(residual + that)`` (identical to ``residual_gate_add_cuda``) and
  feeds the rounded result into the same faithful norm.

The per-fragment chain is expressed as 64 ordered adds over (TPR,)-wide
strided loads, the fold trees with ``tl.reshape``/``tl.split`` (order-exact,
single adds), bf16 boundaries with a bitcast round-to-nearest-even helper,
and the square with an opaque ``mul.rn.f32`` so the compiler cannot contract
it into an FMA.  Verified ``torch.equal`` against the live eager chain on
(1,4216,4096)/(1,4096,4096)/(2,1140,4096)/(1,128,2048) bf16; callers should
still verify once at runtime and fall back if the platform's rmsnorm dispatch
ever changes (see ``ernie_image.py``).
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _round_bf16_to_fp32(value):
    # RNE round of an fp32 value to bf16 precision, staying in fp32 registers
    # (also blocks any fmul+fadd contraction across the boundary).
    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def _mul_rn_f32(x, y):
    # opaque mul.rn.f32: keeps the square a separately rounded fp32 op and
    # blocks contraction with the following add.
    return tl.inline_asm_elementwise(
        asm="mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rsqrt_approx_f32(x):
    return tl.inline_asm_elementwise(
        asm="rsqrt.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fold_adjacent(p, rows: tl.constexpr, width: tl.constexpr):
    # (rows, 2*width) -> (rows, width): add adjacent pairs (even + odd), the
    # tree a shfl butterfly with the smallest offset first produces.
    a, b = tl.split(tl.reshape(p, (rows, width, 2)))
    return a + b


@triton.jit
def _rmsnorm_scale_shift_kernel(
    out_ptr,
    res_out_ptr,
    x_ptr,  # norm input (no gate) / update (with gate)
    residual_ptr,
    gate_ptr,
    weight_ptr,
    scale_ptr,
    shift_ptr,
    seq_len,
    eps,
    D: tl.constexpr,
    TPR: tl.constexpr,
    WPR: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch = row // seq_len
    row_base = row * D
    vec_base = batch * D

    # ----- pass 1: sum of squares in the exact CuTe reduction order -----
    # "thread" tx of the replicated kernel owns columns 8*TPR*b + 8*tx + v;
    # its fragment is iterated v fastest, then b, as one ordered fadd chain
    # (each square separately rounded, no FMA).
    tx = tl.arange(0, TPR) * 8
    acc = tl.zeros((TPR,), dtype=tl.float32)
    for b in tl.static_range(8):
        for v in tl.static_range(8):
            col = tx + (b * 8 * TPR + v)
            if HAS_GATE:
                rj = tl.load(residual_ptr + row_base + col).to(tl.float32)
                uj = tl.load(x_ptr + row_base + col).to(tl.float32)
                gj = tl.load(gate_ptr + vec_base + col).to(tl.float32)
                # eager pair: bf16 round after gate*update and after the add
                xj = _round_bf16_to_fp32(rj + _round_bf16_to_fp32(gj * uj))
            else:
                xj = tl.load(x_ptr + row_base + col).to(tl.float32)
            acc = acc + _mul_rn_f32(xj, xj)

    # warp butterfly (offsets 1,2,4,8,16) == adjacent-pairs fold tree,
    # then the WPR warp sums are combined the same way.
    p = tl.reshape(acc, (WPR, 32))
    p = _fold_adjacent(p, WPR, 16)
    p = _fold_adjacent(p, WPR, 8)
    p = _fold_adjacent(p, WPR, 4)
    p = _fold_adjacent(p, WPR, 2)
    p = _fold_adjacent(p, WPR, 1)
    s = tl.reshape(p, (1, WPR))
    if WPR == 2:
        s = _fold_adjacent(s, 1, 1)
    rcp = tl.sum(_rsqrt_approx_f32(s / D + eps))  # single element, exact

    # ----- pass 2: normalize + modulate, contiguous chunks -----
    for i in tl.static_range(D // 1024):
        cols = i * 1024 + tl.arange(0, 1024)
        if HAS_GATE:
            r = tl.load(residual_ptr + row_base + cols).to(tl.float32)
            u = tl.load(x_ptr + row_base + cols).to(tl.float32)
            g = tl.load(gate_ptr + vec_base + cols).to(tl.float32)
            xin = _round_bf16_to_fp32(r + _round_bf16_to_fp32(g * u))
            tl.store(res_out_ptr + row_base + cols, xin)
        else:
            xin = tl.load(x_ptr + row_base + cols).to(tl.float32)
        w = tl.load(weight_ptr + cols).to(tl.float32)
        sc = tl.load(scale_ptr + vec_base + cols).to(tl.float32)
        sh = tl.load(shift_ptr + vec_base + cols).to(tl.float32)
        y = _round_bf16_to_fp32(xin * rcp * w)  # (bf16)(x * rstd * w)
        one_plus = _round_bf16_to_fp32(1.0 + sc)
        prod = _round_bf16_to_fp32(y * one_plus)
        tl.store(out_ptr + row_base + cols, prod + sh)  # store rounds to bf16


def _threads_per_row(hidden: int) -> int | None:
    # mirror of flashinfer RMSNormKernel._compute_threads_per_row for the
    # regime this kernel replicates (one 8-wide vector per (thread, block))
    tpr = 32 if hidden <= 3072 else 64 if hidden <= 6144 else None
    if tpr is None or hidden != 64 * tpr:
        return None
    return tpr


def _is_row_broadcast(t: torch.Tensor, x: torch.Tensor) -> bool:
    return (
        t.dtype is torch.bfloat16
        and t.is_cuda
        and t.device == x.device
        and t.shape == (x.shape[0], 1, x.shape[-1])
        and t.is_contiguous()
    )


def can_use_fused_rmsnorm_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> bool:
    return (
        x.dtype is torch.bfloat16
        and x.is_cuda
        and x.dim() == 3
        and x.is_contiguous()
        and _threads_per_row(x.shape[-1]) is not None
        and weight.dtype is torch.bfloat16
        and weight.is_cuda
        and weight.device == x.device
        and weight.shape == (x.shape[-1],)
        and weight.is_contiguous()
        and _is_row_broadcast(scale, x)
        and _is_row_broadcast(shift, x)
    )


def can_use_fused_scale_residual_rmsnorm_scale_shift(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> bool:
    return (
        can_use_fused_rmsnorm_scale_shift(residual, weight, scale, shift)
        and update.dtype is torch.bfloat16
        and update.is_cuda
        and update.device == residual.device
        and update.shape == residual.shape
        and update.is_contiguous()
        and _is_row_broadcast(gate, residual)
    )


def _fake_norm_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x)


@register_custom_op(
    op_name="triton_fused_rmsnorm_scale_shift_bitexact",
    mutates_args=[],
    fake_impl=_fake_norm_scale_shift,
)
def fused_rmsnorm_scale_shift_bitexact(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """``norm(x) * (1 + scale) + shift``, bit-exact vs the eager chain."""
    batch, seq_len, hidden = x.shape
    tpr = _threads_per_row(hidden)
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _rmsnorm_scale_shift_kernel[(batch * seq_len,)](
            out,
            out,
            x,
            x,
            x,
            weight,
            scale,
            shift,
            seq_len,
            eps,
            D=hidden,
            TPR=tpr,
            WPR=tpr // 32,
            HAS_GATE=False,
            # num_warps must match the replicated kernel's warps-per-row:
            # larger blocks trigger pathological Triton layout conversions
            # in the fold stage (measured 25us -> 500us at num_warps=4).
            num_warps=max(tpr // 32, 1),
        )
    return out


def _fake_scale_residual_norm_scale_shift(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(residual), torch.empty_like(residual)


@register_custom_op(
    op_name="triton_fused_scale_residual_rmsnorm_scale_shift_bitexact",
    mutates_args=[],
    fake_impl=_fake_scale_residual_norm_scale_shift,
)
def fused_scale_residual_rmsnorm_scale_shift_bitexact(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``res = residual + gate * update; norm(res) * (1 + scale) + shift``.

    Returns ``(modulated, res)``, both bit-exact vs the eager pair + norm
    chain (and therefore vs the ``residual_gate_add_cuda`` fast path).
    """
    batch, seq_len, hidden = residual.shape
    tpr = _threads_per_row(hidden)
    out = torch.empty_like(residual)
    res_out = torch.empty_like(residual)
    with torch.cuda.device(residual.device):
        _rmsnorm_scale_shift_kernel[(batch * seq_len,)](
            out,
            res_out,
            update,
            residual,
            gate,
            weight,
            scale,
            shift,
            seq_len,
            eps,
            D=hidden,
            TPR=tpr,
            WPR=tpr // 32,
            HAS_GATE=True,
            num_warps=max(tpr // 32, 1),
        )
    return out, res_out
