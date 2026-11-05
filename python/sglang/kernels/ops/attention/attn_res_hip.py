"""Triton attention-residual aggregation for Kimi-K3 on ROCm.

The HIP counterpart of attn_res.py: same aggregation point (score the bank rows
against the current prefix, softmax, weighted sum, output RMSNorm), one launch,
but built for a GPU with no TMA and no tcgen05. See _agg_kernel for why the
shape differs so much from the SM100 kernel's.
"""

from __future__ import annotations

from functools import cache
from typing import Optional

import torch
import triton
import triton.language as tl

# _agg_kernel keeps a next_pow2(nvb) x next_pow2(H) fp32 tile in registers, and
# that is the whole basis of its speed. Past this budget it spills and loses to
# the 2-kernel Triton pipeline it replaces: measured at T=4 on MI355X,
# next_pow2(nvb)=8 is 2.0x faster, 16 is 1.2x, 32 is 0.2x. K3 sits right at the
# limit with H=7168 (one masked tile of 8192) and nvb <= 8.
MAX_REGISTER_TILE: int = 8 * 8192

_AGG_NUM_WARPS = 4


@cache
def supports_attn_res_hip(hidden_size: int, nvb: int) -> bool:
    """Whether this shape fits the register budget. Callers must additionally
    check that they are on ROCm; this is only the shape constraint."""
    return _tile_size(hidden_size, nvb) <= MAX_REGISTER_TILE


def _tile_size(hidden_size: int, nvb: int) -> int:
    return triton.next_power_of_2(max(nvb, 1)) * triton.next_power_of_2(hidden_size)


@triton.jit
def _agg_kernel(
    prefix_ptr,  # [T, H]
    addend_ptr,  # [T, H]; the pending residual, or prefix_ptr when not HAS_ADD
    prefix_out_ptr,  # [T, H]; materialized prefix, written when HAS_ADD
    bank_ptr,  # [T, NB, H]
    cw_ptr,  # [H] fp32; score_norm weight * score_proj weight
    ow_ptr,  # [H]; out RMSNorm weight, unread when not APPLY_OUT_NORM
    out_ptr,  # [T, H]
    out8_ptr,  # [T, H] E4M3 copy of out (unit scale, saturating), written when HAS_OUT8
    score_eps,
    out_eps,
    stride_pm,
    stride_am,
    stride_om,
    stride_bm,
    stride_bb,
    stride_o,
    stride_o8,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NVB: tl.constexpr,
    R_PAD: tl.constexpr,
    HAS_ADD: tl.constexpr,
    WRITE_BANK: tl.constexpr,
    APPLY_OUT_NORM: tl.constexpr,
    HAS_OUT8: tl.constexpr,
):
    """One CTA per token: score the NVB+1 rows, softmax, mix, apply the output
    RMSNorm, all in one launch.

    T is the decode batch size, so this runs a handful of CTAs on a 256-CU GPU
    and what binds is the load latency and bandwidth of a *single* CU. That is
    what dictates the shape, and why it is not a port of the SM100 kernel. That
    one's online softmax reads each row once but chains one block-wide reduction
    per row; here each link in that chain costs a full HBM round-trip —
    measured 1.1us/row, dead linear in NVB, because there is no TMA pipeline to
    hide it behind. Scoring the rows as one [R_PAD, BLOCK_H] tile instead puts
    every row's load in flight at once, and keeping that tile in registers lets
    the mix reuse it rather than re-reading the bank, which at one active CU is
    the difference between ~9us and ~12.5us at NVB=8.

    Holding the tile is why NVB is a constexpr and why MAX_REGISTER_TILE caps
    the shape: the register budget is what this trades away.

    The prefix row streams through anyway, so the pending residual add and the
    bank snapshot ride along for free — no other program reads bank row NVB, so
    those stores need no synchronization.

    Taking the global max before exponentiating makes this bit-comparable to
    the 2-kernel pipeline rather than to the SM100 kernel.
    """
    t = tl.program_id(0).to(tl.int64)  # t * stride_bm overflows int32
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    # The prefix row is score row NVB, and the only row that needs writing back.
    row = tl.load(prefix_ptr + t * stride_pm + offs, mask=mask, other=0.0)
    if HAS_ADD:
        # Round to the storage dtype before scoring: downstream readers see
        # these bits, so the score has to as well.
        row = (
            row.to(tl.float32)
            + tl.load(addend_ptr + t * stride_am + offs, mask=mask, other=0.0).to(
                tl.float32
            )
        ).to(prefix_out_ptr.dtype.element_ty)
        tl.store(prefix_out_ptr + t * stride_om + offs, row, mask=mask)
    if WRITE_BANK:
        tl.store(bank_ptr + t * stride_bm + NVB * stride_bb + offs, row, mask=mask)
    pv = row.to(tl.float32)

    cw = tl.load(cw_ptr + offs, mask=mask, other=0.0)
    p_score = tl.sum(pv * cw) / tl.sqrt(tl.sum(pv * pv) / H + score_eps)

    # The whole bank, in registers: every row's load is in flight at once, and
    # the mix below reuses it instead of going back to HBM.
    offs_r = tl.arange(0, R_PAD)
    mask_r = offs_r < NVB
    tile = tl.load(
        bank_ptr + t * stride_bm + offs_r[:, None] * stride_bb + offs[None, :],
        mask=mask_r[:, None] & mask[None, :],
        other=0.0,
    ).to(tl.float32)
    b_score = tl.sum(tile * cw[None, :], axis=1) / tl.sqrt(
        tl.sum(tile * tile, axis=1) / H + score_eps
    )

    m = tl.maximum(tl.max(tl.where(mask_r, b_score, -float("inf"))), p_score)
    b_w = tl.where(mask_r, tl.exp(b_score - m), 0.0)
    p_w = tl.exp(p_score - m)
    inv = 1.0 / (tl.sum(b_w) + p_w)

    acc = pv * (p_w * inv) + tl.sum(b_w[:, None] * inv * tile, axis=0)

    if APPLY_OUT_NORM:
        scale = 1.0 / tl.sqrt(tl.sum(acc * acc) / H + out_eps)
        ow = tl.load(ow_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc = acc * scale * ow
    ob = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + t * stride_o + offs, ob, mask=mask)
    if HAS_OUT8:
        # Same value the next static-FP8 GEMM would quantize: the rounded output,
        # saturated to the E4M3 range (unit activation scale).
        q = tl.minimum(tl.maximum(ob.to(tl.float32), -448.0), 448.0)
        tl.store(
            out8_ptr + t * stride_o8 + offs, q.to(out8_ptr.dtype.element_ty), mask=mask
        )


# Token count from which attn_res_hip switches to _agg_stream_kernel. Below it
# the launch is a few CTAs and _agg_kernel's all-rows-in-flight shape wins on
# latency (T=64, NVB=8: 15 vs 16.5us); from ~T=128 the two tie and by T=256 the
# stream kernel is ahead at every NVB (NVB=8: 19.5 vs 22.5us, T=512: 18.5 vs
# 32us, T=8192: 284 vs 411us). 256 keeps decode batches on the old kernel.
# Measured on MI355X (H=7168).
_STREAM_MIN_T = 256
_STREAM_NUM_WARPS = 4


@triton.jit
def _agg_stream_kernel(
    prefix_ptr,
    addend_ptr,
    prefix_out_ptr,
    bank_ptr,
    cw_ptr,
    ow_ptr,
    out_ptr,
    out8_ptr,
    score_eps,
    out_eps,
    stride_pm,
    stride_am,
    stride_om,
    stride_bm,
    stride_bb,
    stride_o,
    stride_o8,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NVB: tl.constexpr,
    HAS_ADD: tl.constexpr,
    WRITE_BANK: tl.constexpr,
    APPLY_OUT_NORM: tl.constexpr,
    HAS_OUT8: tl.constexpr,
):
    """Bandwidth-bound prefill variant of _agg_kernel: rows stream through an
    online softmax so the bank is read once with a small register tile (the
    full [next_pow2(NVB), 8192] tile costs occupancy at large T). Outputs match
    _agg_kernel up to rare 1-ulp bf16 flips from the running-max rescale."""
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    row = tl.load(prefix_ptr + t * stride_pm + offs, mask=mask, other=0.0)
    if HAS_ADD:
        row = (
            row.to(tl.float32)
            + tl.load(addend_ptr + t * stride_am + offs, mask=mask, other=0.0).to(
                tl.float32
            )
        ).to(prefix_out_ptr.dtype.element_ty)
        # Streaming stores (.cs): nothing in this launch reads these back.
        tl.store(
            prefix_out_ptr + t * stride_om + offs, row, mask=mask, cache_modifier=".cs"
        )
    if WRITE_BANK:
        tl.store(
            bank_ptr + t * stride_bm + NVB * stride_bb + offs,
            row,
            mask=mask,
            cache_modifier=".cs",
        )
    pv = row.to(tl.float32)

    cw = tl.load(cw_ptr + offs, mask=mask, other=0.0)
    # the prefix row seeds the running max, mix and denominator
    m = tl.sum(pv * cw) / tl.sqrt(tl.sum(pv * pv) / H + score_eps)
    acc = pv
    den = 1.0
    brow = bank_ptr + t * stride_bm
    for r in tl.range(NVB):
        x = tl.load(brow + r * stride_bb + offs, mask=mask, other=0.0).to(tl.float32)
        s = tl.sum(x * cw) / tl.sqrt(tl.sum(x * x) / H + score_eps)
        m_new = tl.maximum(m, s)
        alpha = tl.exp(m - m_new)
        e = tl.exp(s - m_new)
        acc = acc * alpha + e * x
        den = den * alpha + e
        m = m_new
    acc = acc * (1.0 / den)

    if APPLY_OUT_NORM:
        scale = 1.0 / tl.sqrt(tl.sum(acc * acc) / H + out_eps)
        ow = tl.load(ow_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc = acc * scale * ow
    ob = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + t * stride_o + offs, ob, mask=mask, cache_modifier=".cs")
    if HAS_OUT8:
        q = tl.minimum(tl.maximum(ob.to(tl.float32), -448.0), 448.0)
        tl.store(
            out8_ptr + t * stride_o8 + offs,
            q.to(out8_ptr.dtype.element_ty),
            mask=mask,
            cache_modifier=".cs",
        )


def attn_res_hip(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: Optional[torch.Tensor],
    out: torch.Tensor,
    nvb: int,
    score_eps: float,
    out_eps: float,
    *,
    addend: Optional[torch.Tensor] = None,
    prefix_out: Optional[torch.Tensor] = None,
    write_prefix: bool = False,
    out_fp8: Optional[torch.Tensor] = None,
) -> None:
    """Single-kernel attention-residual aggregation for ROCm.

    Restrictions: nvb >= 1, and the shape must pass supports_attn_res_hip().

    Parameters
    ----------
    prefix_sum : [T, H] bf16 — the running prefix, or its first term if addend
                 is given
    bank       : [T, NB, H] bf16 (rows 0..nvb-1 are aggregated)
    cw         : [H] fp32 — precomputed score_norm weight * proj weight
    ow         : [H] output RMSNorm weight, or None to return the pre-norm
                 softmax mixture (the aggregate-stream value)
    out        : [T, H] bf16 output buffer
    nvb        : number of valid bank rows (>= 1)
    score_eps, out_eps : RMSNorm epsilons; unlike the SM100 kernel these need
                 not be equal
    addend     : fold a pending residual add in, so the aggregated prefix is
                 prefix_sum + addend; requires prefix_out
    prefix_out : [T, H] bf16 buffer receiving that materialized prefix
    out_fp8    : optional [T, H] float8_e4m3fn buffer receiving the saturating
                 E4M3 cast of out (unit scale), for a static-FP8 consumer GEMM
    write_prefix : also snapshot the prefix row into bank[:, nvb, :] (bit-exact
                 copy, fused into the score pass which already has the row in
                 registers); requires NB > nvb
    """
    T, H = prefix_sum.shape
    assert nvb >= 1, "nvb == 0 has nothing to aggregate; the caller must handle it"
    assert supports_attn_res_hip(H, nvb), (
        f"attn_res_hip: register tile {_tile_size(H, nvb)} exceeds "
        f"{MAX_REGISTER_TILE} (H={H}, nvb={nvb})"
    )
    has_add = addend is not None
    assert not has_add or prefix_out is not None, "addend requires prefix_out"

    # Triton needs a real pointer for every argument; the flags decide whether
    # these are ever dereferenced.
    addend_arg = addend if has_add else prefix_sum
    prefix_out_arg = prefix_out if has_add else prefix_sum
    ow_arg = ow if ow is not None else cw
    out8_arg = out_fp8 if out_fp8 is not None else out

    if T >= _STREAM_MIN_T:
        _agg_stream_kernel[(T,)](
            prefix_sum,
            addend_arg,
            prefix_out_arg,
            bank,
            cw,
            ow_arg,
            out,
            out8_arg,
            score_eps,
            out_eps,
            prefix_sum.stride(0),
            addend_arg.stride(0),
            prefix_out_arg.stride(0),
            bank.stride(0),
            bank.stride(1),
            out.stride(0),
            out8_arg.stride(0),
            H=H,
            BLOCK_H=triton.next_power_of_2(H),
            NVB=nvb,
            HAS_ADD=has_add,
            WRITE_BANK=write_prefix,
            APPLY_OUT_NORM=ow is not None,
            HAS_OUT8=out_fp8 is not None,
            num_warps=_STREAM_NUM_WARPS,
        )
        return

    _agg_kernel[(T,)](
        prefix_sum,
        addend_arg,
        prefix_out_arg,
        bank,
        cw,
        ow_arg,
        out,
        out8_arg,
        score_eps,
        out_eps,
        prefix_sum.stride(0),
        addend_arg.stride(0),
        prefix_out_arg.stride(0),
        bank.stride(0),
        bank.stride(1),
        out.stride(0),
        out8_arg.stride(0),
        H=H,
        BLOCK_H=triton.next_power_of_2(H),
        NVB=nvb,
        R_PAD=triton.next_power_of_2(nvb),
        HAS_ADD=has_add,
        WRITE_BANK=write_prefix,
        APPLY_OUT_NORM=ow is not None,
        HAS_OUT8=out_fp8 is not None,
        num_warps=_AGG_NUM_WARPS,
    )
