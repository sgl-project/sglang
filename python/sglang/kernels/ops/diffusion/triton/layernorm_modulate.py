# SPDX-License-Identifier: Apache-2.0
"""Fused LayerNorm + adaLN modulate Triton kernels for bf16 activations.

Two fusions, each replacing an eager multi-kernel chain with a single
launch while reproducing the eager results bit for bit (``torch.equal``),
so callers need no quality gate:

- ``fused_layernorm_modulate``: ``LN(x) * (1 + scale) + shift``
- ``fused_qk_head_layernorm``: per-head ``LN(q)`` / ``LN(k)``

Numerics contract (replicates torch 2.11's
``at::native::vectorized_layer_norm_kernel<c10::BFloat16, float, false>``,
which ``nn.LayerNorm`` dispatches to for bf16 rows with ``N % 4 == 0`` and
16-byte-aligned buffers; SASS-level derivation in PR #34008):

- 128 aten threads per row; thread ``t`` serially Welford-pushes the
  4-element vectors ``t, t+128, ... < N/4`` (for ``N % 512 != 0`` the
  trailing threads stop one iteration early and enter the fold with a
  smaller count), each scalar as
  ``mean' = fma(delta, rcp(count+1), mean)``,
  ``m2' = fma(delta, val - mean', m2)``, with ``rcp`` being nvcc's
  guarded-reciprocal fast path (``_rcp4``).
- Lane states fold via shfl.down offsets 16, 8, 4, 2, 1 through
  ``cuWelfordCombine`` (argument order: self = lower lane, other = upper;
  non-positive counts fold to zeros); the 4 warp results then combine
  pairwise (0,2), (1,3), (0,1).
- ``rstd = rsqrtf(div.rn.f32(m2, N) + eps)``: correctly-rounded fp32
  division, then ``MUFU.RSQ`` with a 2^24/2^12 rescale for subnormals.
- The normalized output is ``cvt.rn.bf16(rstd * (x - mean))`` (mul then
  add, no fma); the modulate chain then rounds to bf16 after every op:
  ``round(1 + scale)``, ``round(y * that)``, ``round(prod + shift)``.

For the qk kernel (per-head rows, ``dim_head % 4 == 0``, ``dim_head <=
128``) only the first ``dim_head / 4`` of the 128 aten threads carry data;
the remaining lanes and warps enter the fold with ``count == 0`` and are
reproduced faithfully.

Bit-exactness holds only for the dispatch above, so callers must verify
``torch.equal`` against the live eager chain once at runtime and fall back
on mismatch (see ``glm_image.py``).
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.ops.diffusion.triton.numerics import (
    cuda_rsqrtf,
    div_rn_f32,
    round_bf16_to_fp32,
)
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _rcp4(x):
    # nvcc's reciprocal fast path (always taken for our integer counts).
    return tl.inline_asm_elementwise(
        asm="""{
        .reg .f32 r0, e, e2;
        rcp.approx.f32 r0, $1;
        fma.rn.f32 e, $1, r0, 0fBF800000;
        sub.ftz.f32 e2, 0f80000000, e;
        fma.rn.f32 $0, r0, e2, r0;
        }""",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _welford_push(val, mean, m2, cnt, valid, MASKED: tl.constexpr):
    # ``valid`` masks lanes whose aten thread does not execute this
    # iteration (their state must stay untouched).
    delta = val - mean
    new_cnt = cnt + 1.0
    recip = _rcp4(new_cnt)
    new_mean = tl.fma(delta, recip, mean)
    t = val - new_mean
    new_m2 = tl.fma(delta, t, m2)
    if MASKED:
        new_mean = tl.where(valid, new_mean, mean)
        new_m2 = tl.where(valid, new_m2, m2)
        new_cnt = tl.where(valid, new_cnt, cnt)
    return new_mean, new_m2, new_cnt


@triton.jit
def _welford_combine(mean_b, m2_b, cnt_b, mean_a, m2_a, cnt_a):
    # b = self / lower lane, a = other / upper lane; the op order matters.
    count = cnt_a + cnt_b
    pos = count > 0.0
    coef = _rcp4(tl.where(pos, count, 1.0))
    delta = mean_b - mean_a
    n_b = coef * cnt_b
    d2 = delta * delta
    n_a = cnt_a * coef
    s = m2_a + m2_b
    t1 = n_b * mean_b
    mean = tl.fma(mean_a, n_a, t1)
    t2 = cnt_a * d2
    m2 = tl.fma(n_b, t2, s)
    mean = tl.where(pos, mean, 0.0)
    m2 = tl.where(pos, m2, 0.0)
    return mean, m2, count


@triton.jit
def _split_halves(x, rows: tl.constexpr, half: tl.constexpr):
    # (rows, 2*half) -> two (rows, half) tensors pairing lane i with i+half,
    # the tree a shfl.down fold with the largest offset first produces.
    x = tl.reshape(x, (rows, 2, half), can_reorder=False)
    x = tl.permute(x, (0, 2, 1))
    return tl.split(x)


@triton.jit
def _fold_halves(mean, m2, cnt, rows: tl.constexpr, half: tl.constexpr):
    mb, ma = _split_halves(mean, rows, half)
    sb, sa = _split_halves(m2, rows, half)
    cb, ca = _split_halves(cnt, rows, half)
    return _welford_combine(mb, sb, cb, ma, sa, ca)


@triton.jit
def _fold_tree_32(mean, m2, cnt, rows: tl.constexpr):
    # shfl.down offsets 16, 8, 4, 2, 1 over the 32 lane states -> (rows, 1).
    mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 16)
    mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 8)
    mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 4)
    mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 2)
    mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 1)
    return mean, m2, cnt


@triton.jit
def _push_vec4(
    x4,
    mean,
    m2,
    cnt,
    valid,
    rows: tl.constexpr,
    lanes: tl.constexpr,
    MASKED: tl.constexpr,
):
    # Push the 4 elements of one aligned_vector<bf16, 4> in exact serial
    # order.  x4 is (rows, lanes, 4) fp32.
    g = tl.reshape(x4, (rows, lanes, 2, 2), can_reorder=False)
    p02, p13 = tl.split(g)  # elements (0, 2) / (1, 3)
    e0, e2 = tl.split(tl.reshape(p02, (rows, lanes, 1, 2), can_reorder=False))
    e1, e3 = tl.split(tl.reshape(p13, (rows, lanes, 1, 2), can_reorder=False))
    e0 = tl.reshape(e0, (rows, lanes), can_reorder=False)
    e1 = tl.reshape(e1, (rows, lanes), can_reorder=False)
    e2 = tl.reshape(e2, (rows, lanes), can_reorder=False)
    e3 = tl.reshape(e3, (rows, lanes), can_reorder=False)
    mean, m2, cnt = _welford_push(e0, mean, m2, cnt, valid, MASKED)
    mean, m2, cnt = _welford_push(e1, mean, m2, cnt, valid, MASKED)
    mean, m2, cnt = _welford_push(e2, mean, m2, cnt, valid, MASKED)
    mean, m2, cnt = _welford_push(e3, mean, m2, cnt, valid, MASKED)
    return mean, m2, cnt


@triton.jit
def _layernorm_modulate_kernel(
    y_ptr,
    x_ptr,
    scale_ptr,
    shift_ptr,
    seq_len,
    n_rows,
    scale_row_stride,
    eps,
    D: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row_offs = pid * ROWS + tl.arange(0, ROWS)
    row_mask = row_offs < n_rows
    row_base = row_offs * D

    lanes = tl.arange(0, 128)
    mean = tl.zeros((ROWS, 128), dtype=tl.float32)
    m2 = tl.zeros((ROWS, 128), dtype=tl.float32)
    cnt = tl.zeros((ROWS, 128), dtype=tl.float32)

    # pass 1: per-"thread" serial Welford in aten's exact element order.
    # Out-of-range rows compute garbage that is never stored.
    for i in tl.static_range((D + 511) // 512):
        cols = i * 512 + lanes[:, None] * 4 + tl.arange(0, 4)[None, :]
        if (i + 1) * 512 <= D:
            x4 = tl.load(
                x_ptr + row_base[:, None, None] + cols[None, :, :],
                mask=row_mask[:, None, None],
                other=0.0,
            ).to(tl.float32)
            mean, m2, cnt = _push_vec4(
                x4, mean, m2, cnt, row_mask, ROWS, 128, MASKED=False
            )
        else:
            # partial tail chunk (D % 512 != 0): aten threads whose vector
            # index i*128+t reaches N/4 skip the iteration, leaving their
            # Welford state untouched.
            vec_valid = (i * 128 + lanes < D // 4)[None, :]
            x4 = tl.load(
                x_ptr + row_base[:, None, None] + cols[None, :, :],
                mask=row_mask[:, None, None] & vec_valid[:, :, None],
                other=0.0,
            ).to(tl.float32)
            mean, m2, cnt = _push_vec4(
                x4, mean, m2, cnt, vec_valid, ROWS, 128, MASKED=True
            )

    # warp fold trees, then the (0,2)/(1,3)/(0,1) inter-warp combines.
    mean = tl.reshape(mean, (ROWS * 4, 32), can_reorder=False)
    m2 = tl.reshape(m2, (ROWS * 4, 32), can_reorder=False)
    cnt = tl.reshape(cnt, (ROWS * 4, 32), can_reorder=False)
    mean, m2, cnt = _fold_tree_32(mean, m2, cnt, ROWS * 4)
    mean = tl.reshape(mean, (ROWS, 4), can_reorder=False)
    m2 = tl.reshape(m2, (ROWS, 4), can_reorder=False)
    cnt = tl.reshape(cnt, (ROWS, 4), can_reorder=False)
    mean, m2, cnt = _fold_halves(mean, m2, cnt, ROWS, 2)
    mean, m2, cnt = _fold_halves(mean, m2, cnt, ROWS, 1)

    denom = tl.zeros((ROWS, 1), dtype=tl.float32) + D
    rstd = cuda_rsqrtf(div_rn_f32(m2, denom) + eps)  # (ROWS, 1)

    batch = row_offs // seq_len

    # pass 2: normalize + modulate, in aten's rounding order.
    for i in tl.static_range((D + 511) // 512):
        cols = i * 512 + tl.arange(0, 512)
        mask = row_mask[:, None]
        if (i + 1) * 512 > D:
            mask = mask & (cols < D)[None, :]
        x = tl.load(
            x_ptr + row_base[:, None] + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        y = round_bf16_to_fp32(rstd * (x - mean))
        sc = tl.load(
            scale_ptr + batch[:, None] * scale_row_stride + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        sh = tl.load(
            shift_ptr + batch[:, None] * scale_row_stride + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        one_plus = round_bf16_to_fp32(1.0 + sc)
        y = round_bf16_to_fp32(y * one_plus) + sh
        tl.store(y_ptr + row_base[:, None] + cols[None, :], y, mask=mask)


@triton.jit
def _qk_ln_head_one(
    dst,
    src,
    pid,
    n_rows,
    eps,
    D: tl.constexpr,
    D_POW2: tl.constexpr,
    ROWS: tl.constexpr,
):
    row_offs = pid * ROWS + tl.arange(0, ROWS)
    row_mask = row_offs < n_rows
    row_base = row_offs * D

    lanes = tl.arange(0, 32)
    lane_valid = (lanes < D // 4)[None, :]
    mean = tl.zeros((ROWS, 32), dtype=tl.float32)
    m2 = tl.zeros((ROWS, 32), dtype=tl.float32)
    cnt = tl.zeros((ROWS, 32), dtype=tl.float32)

    cols = lanes[:, None] * 4 + tl.arange(0, 4)[None, :]
    x4 = tl.load(
        src + row_base[:, None, None] + cols[None, :, :],
        mask=row_mask[:, None, None] & lane_valid[:, :, None],
        other=0.0,
    ).to(tl.float32)
    mean, m2, cnt = _push_vec4(x4, mean, m2, cnt, lane_valid, ROWS, 32, MASKED=True)

    mean, m2, cnt = _fold_tree_32(mean, m2, cnt, ROWS)
    # inter-warp combines with the all-zero warps 1..3 of the aten block:
    # (0,2) with zero, then (0,1) where warp 1 combined two zero warps.
    zero = tl.zeros((ROWS, 1), dtype=tl.float32)
    mean, m2, cnt = _welford_combine(mean, m2, cnt, zero, zero, zero)
    mean, m2, cnt = _welford_combine(mean, m2, cnt, zero, zero, zero)

    denom = tl.zeros((ROWS, 1), dtype=tl.float32) + D
    rstd = cuda_rsqrtf(div_rn_f32(m2, denom) + eps)

    cols2 = tl.arange(0, D_POW2)
    out_mask = row_mask[:, None] & (cols2 < D)[None, :]
    x = tl.load(src + row_base[:, None] + cols2[None, :], mask=out_mask, other=0.0).to(
        tl.float32
    )
    y = rstd * (x - mean)
    tl.store(dst + row_base[:, None] + cols2[None, :], y, mask=out_mask)


@triton.jit
def _qk_ln_head_kernel(
    q_out_ptr,
    k_out_ptr,
    q_ptr,
    k_ptr,
    n_rows,
    eps,
    D: tl.constexpr,
    D_POW2: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    if tl.program_id(1) == 0:
        _qk_ln_head_one(q_out_ptr, q_ptr, pid, n_rows, eps, D, D_POW2, ROWS)
    else:
        _qk_ln_head_one(k_out_ptr, k_ptr, pid, n_rows, eps, D, D_POW2, ROWS)


def is_plain_layer_norm(norm: torch.nn.Module, hidden: int) -> bool:
    """True for a bare ``nn.LayerNorm((hidden,))`` without affine params."""
    return (
        type(norm) is torch.nn.LayerNorm
        and norm.weight is None
        and norm.bias is None
        and tuple(norm.normalized_shape) == (hidden,)
    )


def _is_bf16_cuda(t: torch.Tensor) -> bool:
    return t.is_cuda and t.dtype is torch.bfloat16


def _mod_row_stride(t: torch.Tensor, batch: int, hidden: int) -> int | None:
    # (batch, hidden) modulation rows, possibly strided views of a chunked
    # adaLN projection; the last dim must be packed.
    if t.dim() != 2 or t.shape != (batch, hidden) or t.stride(1) != 1:
        return None
    return t.stride(0) if batch > 1 else hidden


def can_use_fused_layernorm_modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> bool:
    if not (
        _is_bf16_cuda(x)
        and x.dim() == 3
        and x.numel() > 0
        and x.is_contiguous()
        and x.shape[-1] % 4 == 0
        and x.shape[-1] <= 8192
        and _is_bf16_cuda(scale)
        and _is_bf16_cuda(shift)
        and scale.device == x.device
        and shift.device == x.device
    ):
        return False
    batch, _, hidden = x.shape
    q = _mod_row_stride(scale, batch, hidden)
    v = _mod_row_stride(shift, batch, hidden)
    return q is not None and v is not None and q == v


def _fake_ln_modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
) -> torch.Tensor:
    return torch.empty_like(x)


def fused_layernorm_modulate_raw(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
) -> torch.Tensor:
    """``LN(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)``, bit-exact
    vs the eager aten chain (LayerNorm without affine).

    Direct-call variant without the ``torch.ops`` dispatch (which costs tens
    of microseconds per call); use it on CPU-launch-bound eager hot paths
    (e.g. Sana), and the registered custom op under ``torch.compile``.
    """
    batch, seq_len, hidden = x.shape
    n_rows = batch * seq_len
    rows = 2
    out = torch.empty_like(x)
    stride = _mod_row_stride(scale, batch, hidden)
    with torch.cuda.device(x.device):
        _layernorm_modulate_kernel[(triton.cdiv(n_rows, rows),)](
            out,
            x,
            scale,
            shift,
            seq_len,
            n_rows,
            stride,
            eps,
            D=hidden,
            ROWS=rows,
            # H200-tuned: 38.5us at (1, 4096, 4096) vs the 121.8us eager
            # chain, 14.3us at Sana's (2, 1024, 2240) vs 43.1us.  ROWS=1 +
            # 4 warps triggers pathological Triton layout conversions in
            # the fold stage (47-58us).
            num_warps=4 if hidden >= 2048 else 2,
        )
    return out


fused_layernorm_modulate = register_custom_op(
    fused_layernorm_modulate_raw,
    op_name="triton_fused_layernorm_modulate",
    mutates_args=[],
    fake_impl=_fake_ln_modulate,
)


def can_use_fused_qk_head_layernorm(q: torch.Tensor, k: torch.Tensor) -> bool:
    head_dim = q.shape[-1] if q.dim() == 4 else 0
    return (
        _is_bf16_cuda(q)
        and _is_bf16_cuda(k)
        and q.device == k.device
        and q.dim() == 4
        and q.shape == k.shape
        and head_dim % 4 == 0
        and 0 < head_dim <= 128
        and q.numel() > 0
        and q.is_contiguous()
        and k.is_contiguous()
    )


def _fake_qk_ln(
    q: torch.Tensor, k: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k)


@register_custom_op(
    op_name="triton_fused_qk_head_layernorm",
    mutates_args=[],
    fake_impl=_fake_qk_ln,
)
def fused_qk_head_layernorm(
    q: torch.Tensor, k: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head ``nn.LayerNorm(dim_head)`` (no affine) over q and k in one
    launch, bit-exact vs the eager aten kernel."""
    head_dim = q.shape[-1]
    n_rows = q.numel() // head_dim
    rows = 64
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    with torch.cuda.device(q.device):
        _qk_ln_head_kernel[(triton.cdiv(n_rows, rows), 2)](
            q_out,
            k_out,
            q,
            k,
            n_rows,
            eps,
            D=head_dim,
            D_POW2=triton.next_power_of_2(head_dim),
            ROWS=rows,
            # H200-tuned: 62us at (1, 4360, 32, 128) vs the 301us of the two
            # aten launches (one 128-thread block per head_dim-element row).
            num_warps=2,
        )
    return q_out, k_out
