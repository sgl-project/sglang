"""The DeepSeek-V4 MoE router at decode row counts on ROCm: a split-K bf16 GEMV
(:func:`rocm_router_gemv_split_k`, fp32 partials) and a sqrtsoftplus top-k gate
(:func:`rocm_router_gate`) that sums the partials in a fixed order and reproduces aiter's
``topk_gating_kernel_opt`` bit for bit, ties included. Serves ``M <= ROCM_ROUTER_MAX_TOKENS``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_gfx95_supported, is_hip

# doubles as the GEMV's compile-time row tile, so every M <= 16 runs the same tile (batch invariance)
ROCM_ROUTER_MAX_TOKENS = 16

_BLOCK_M = 16
_BLOCK_N = 16
_BLOCK_K = 512
_MAX_SPLIT_K = 32
# aiter's wave64 kernel holds EPT = 384 / 64 = 6 slots per lane; the sorting network below is the N == 6 one
_WARP = 64
_GATE_NUM_EXPERTS = _WARP * 6
_MAX_TOPK = 16

_FLT_MAX = tl.constexpr(3.4028234663852886e38)


def rocm_gemv_split_k_max_tokens(*, n: int, k: int, weight_dtype: torch.dtype) -> int:
    """Rows up to which :func:`rocm_router_gemv_split_k` serves an ``[M, k] @ [n, k].T`` bf16
    GEMV (one 16-wide N tile per 512 of K), -1 when the device or the shape rules it out."""
    if not (is_hip() and is_gfx95_supported()):
        return -1
    if weight_dtype != torch.bfloat16:
        return -1
    if n <= 0 or n % _BLOCK_N != 0:
        return -1
    if k % _BLOCK_K != 0 or not 0 < k // _BLOCK_K <= _MAX_SPLIT_K:
        return -1
    return ROCM_ROUTER_MAX_TOKENS


def rocm_router_max_tokens(
    *,
    num_experts: int,
    hidden_size: int,
    topk: int,
    weight_dtype: torch.dtype,
) -> int:
    """Rows up to which :func:`rocm_router_gemv_split_k` + :func:`rocm_router_gate`
    serve the router, -1 when the device or the shape rules them out."""
    if num_experts != _GATE_NUM_EXPERTS or not 0 < topk <= _MAX_TOPK:
        return -1
    return rocm_gemv_split_k_max_tokens(
        n=num_experts, k=hidden_size, weight_dtype=weight_dtype
    )


@triton.jit
def _router_gemv_split_k_kernel(
    x_ptr,
    w_ptr,
    part_ptr,
    M,
    stride_xm,
    stride_wn,
    stride_ps,
    stride_pm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = offs_m < M
    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    w = tl.load(w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None])
    acc = tl.dot(x, w)
    tl.store(
        part_ptr + pid_k * stride_ps + offs_m[:, None] * stride_pm + offs_n[None, :],
        acc,
        mask=row_mask[:, None],
    )


def rocm_router_gemv_split_k(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``x[M, K] @ w[N, K].T`` as fp32 split-K partials ``[K // 512, M, N]``.

    Sum the partials over dim 0 in order (:func:`rocm_router_reduce_partials`,
    or the fused gate) to get the logits.  ``M <= ROCM_ROUTER_MAX_TOKENS``.
    """
    M, K = x.shape
    N, K_w = w.shape
    assert K == K_w and K % _BLOCK_K == 0 and N % _BLOCK_N == 0
    assert 0 < M <= ROCM_ROUTER_MAX_TOKENS
    assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
    assert x.stride(1) == 1 and w.stride(1) == 1
    split_k = K // _BLOCK_K
    partials = torch.empty((split_k, M, N), dtype=torch.float32, device=x.device)
    _router_gemv_split_k_kernel[(N // _BLOCK_N, split_k)](
        x,
        w,
        partials,
        M,
        x.stride(0),
        w.stride(0),
        partials.stride(0),
        partials.stride(1),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=4,
    )
    return partials


@triton.jit
def _reduce_partials_kernel(
    part_ptr,
    out_ptr,
    M,
    stride_ps,
    stride_pm,
    stride_om,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs // N
    n = offs % N
    mask = m < M
    acc = tl.load(part_ptr + m * stride_pm + n, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLIT_K):
        acc += tl.load(
            part_ptr + s * stride_ps + m * stride_pm + n, mask=mask, other=0.0
        )
    tl.store(out_ptr + m * stride_om + n, acc, mask=mask)


def rocm_router_reduce_partials(partials: torch.Tensor, out: torch.Tensor) -> None:
    """``out[M, N] = partials[0] + partials[1] + ...`` in that order, in fp32:
    the same sum the fused gate computes."""
    split_k, M, N = partials.shape
    assert out.shape == (M, N) and out.dtype == torch.float32 and out.stride(1) == 1
    block = 1024
    _reduce_partials_kernel[(triton.cdiv(M * N, block),)](
        partials,
        out,
        M,
        partials.stride(0),
        partials.stride(1),
        out.stride(0),
        N=N,
        SPLIT_K=split_k,
        BLOCK=block,
        num_warps=4,
    )


@triton.jit
def _score(x):
    # aiter compute_score<SCORE_SQRTSOFTPLUS>: +inf clamps to FLT_MAX, NaN falls through, sqrt correctly rounded
    x = tl.where(x > _FLT_MAX, _FLT_MAX, x)
    t = tl.exp2(x * 1.4426950408889634)
    sp = tl.where(x > 20.0, x, tl.log2(1.0 + t) * 0.6931471805599453)
    return tl.sqrt_rn(sp)


@triton.jit
def _cas(vi, oi, ii, vj, oj, ij):
    # aiter _CAS_DESC: swap when vals[i] < vals[j]; equal values stay put.
    c = vi < vj
    return (
        tl.where(c, vj, vi),
        tl.where(c, oj, oi),
        tl.where(c, ij, ii),
        tl.where(c, vi, vj),
        tl.where(c, oi, oj),
        tl.where(c, ii, ij),
    )


@triton.jit
def _slot(cursor, a0, a1, a2, a3, a4, a5, other):
    r = tl.where(cursor == 0, a0, other)
    r = tl.where(cursor == 1, a1, r)
    r = tl.where(cursor == 2, a2, r)
    r = tl.where(cursor == 3, a3, r)
    r = tl.where(cursor == 4, a4, r)
    r = tl.where(cursor == 5, a5, r)
    return r


@triton.jit
def _load_logits(
    logits_ptr, part_ptr, row, e, stride_lm, stride_ps, stride_pm, SPLIT_K: tl.constexpr
):
    if SPLIT_K == 0:
        return tl.load(logits_ptr + row * stride_lm + e).to(tl.float32)
    else:
        acc = tl.load(part_ptr + row * stride_pm + e)
        for s in tl.static_range(1, SPLIT_K):
            acc += tl.load(part_ptr + s * stride_ps + row * stride_pm + e)
        return acc


@triton.jit
def _router_gate_kernel(
    logits_ptr,  # [M, 384] read when SPLIT_K == 0, written when WRITE_LOGITS
    part_ptr,  # [SPLIT_K, M, 384] fp32 (SPLIT_K > 0)
    bias_ptr,  # [384] fp32 or bf16 (HAS_BIAS)
    weights_ptr,  # [M, TOPK] fp32
    ids_ptr,  # [M, TOPK] int32
    stride_lm,
    stride_ps,
    stride_pm,
    stride_om,
    routed_scaling_factor,
    SPLIT_K: tl.constexpr,
    WRITE_LOGITS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RENORM: tl.constexpr,
    TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, 64)
    # Slot i of a lane holds expert lane + 64 * i, as in aiter's register kernel.
    i0 = lane
    i1 = lane + 64
    i2 = lane + 128
    i3 = lane + 192
    i4 = lane + 256
    i5 = lane + 320
    x0 = _load_logits(
        logits_ptr, part_ptr, row, i0, stride_lm, stride_ps, stride_pm, SPLIT_K
    )
    x1 = _load_logits(
        logits_ptr, part_ptr, row, i1, stride_lm, stride_ps, stride_pm, SPLIT_K
    )
    x2 = _load_logits(
        logits_ptr, part_ptr, row, i2, stride_lm, stride_ps, stride_pm, SPLIT_K
    )
    x3 = _load_logits(
        logits_ptr, part_ptr, row, i3, stride_lm, stride_ps, stride_pm, SPLIT_K
    )
    x4 = _load_logits(
        logits_ptr, part_ptr, row, i4, stride_lm, stride_ps, stride_pm, SPLIT_K
    )
    x5 = _load_logits(
        logits_ptr, part_ptr, row, i5, stride_lm, stride_ps, stride_pm, SPLIT_K
    )
    if WRITE_LOGITS:
        tl.store(logits_ptr + row * stride_lm + i0, x0)
        tl.store(logits_ptr + row * stride_lm + i1, x1)
        tl.store(logits_ptr + row * stride_lm + i2, x2)
        tl.store(logits_ptr + row * stride_lm + i3, x3)
        tl.store(logits_ptr + row * stride_lm + i4, x4)
        tl.store(logits_ptr + row * stride_lm + i5, x5)
    s0 = _score(x0)
    s1 = _score(x1)
    s2 = _score(x2)
    s3 = _score(x3)
    s4 = _score(x4)
    s5 = _score(x5)
    # as in aiter, a NaN score weighs 0 and its choice value is -inf
    o0 = tl.where(s0 != s0, 0.0, s0)
    o1 = tl.where(s1 != s1, 0.0, s1)
    o2 = tl.where(s2 != s2, 0.0, s2)
    o3 = tl.where(s3 != s3, 0.0, s3)
    o4 = tl.where(s4 != s4, 0.0, s4)
    o5 = tl.where(s5 != s5, 0.0, s5)
    if HAS_BIAS:
        v0 = s0 + tl.load(bias_ptr + i0).to(tl.float32)
        v1 = s1 + tl.load(bias_ptr + i1).to(tl.float32)
        v2 = s2 + tl.load(bias_ptr + i2).to(tl.float32)
        v3 = s3 + tl.load(bias_ptr + i3).to(tl.float32)
        v4 = s4 + tl.load(bias_ptr + i4).to(tl.float32)
        v5 = s5 + tl.load(bias_ptr + i5).to(tl.float32)
    else:
        v0 = s0
        v1 = s1
        v2 = s2
        v3 = s3
        v4 = s4
        v5 = s5
    ninf = float("-inf")
    v0 = tl.where(v0 != v0, ninf, v0)
    v1 = tl.where(v1 != v1, ninf, v1)
    v2 = tl.where(v2 != v2, ninf, v2)
    v3 = tl.where(v3 != v3, ninf, v3)
    v4 = tl.where(v4 != v4, ninf, v4)
    v5 = tl.where(v5 != v5, ninf, v5)
    # aiter sort_network_desc<6>: the 12-comparator network, same order.
    v0, o0, i0, v1, o1, i1 = _cas(v0, o0, i0, v1, o1, i1)
    v2, o2, i2, v3, o3, i3 = _cas(v2, o2, i2, v3, o3, i3)
    v4, o4, i4, v5, o5, i5 = _cas(v4, o4, i4, v5, o5, i5)
    v0, o0, i0, v2, o2, i2 = _cas(v0, o0, i0, v2, o2, i2)
    v1, o1, i1, v4, o4, i4 = _cas(v1, o1, i1, v4, o4, i4)
    v3, o3, i3, v5, o5, i5 = _cas(v3, o3, i3, v5, o5, i5)
    v0, o0, i0, v1, o1, i1 = _cas(v0, o0, i0, v1, o1, i1)
    v2, o2, i2, v3, o3, i3 = _cas(v2, o2, i2, v3, o3, i3)
    v4, o4, i4, v5, o5, i5 = _cas(v4, o4, i4, v5, o5, i5)
    v1, o1, i1, v2, o2, i2 = _cas(v1, o1, i1, v2, o2, i2)
    v3, o3, i3, v4, o4, i4 = _cas(v3, o3, i3, v4, o4, i4)
    v2, o2, i2, v3, o3, i3 = _cas(v2, o2, i2, v3, o3, i3)
    # k-way merge: every lane offers its head, the max wins, ties go to the lowest lane (aiter's ballot + ctz)
    cursor = tl.zeros([64], dtype=tl.int32)
    sel_val = tl.zeros([64], dtype=tl.float32)
    sel_idx = tl.zeros([64], dtype=tl.int32)
    total = 0.0
    for k in tl.static_range(TOPK):
        my_val = _slot(cursor, v0, v1, v2, v3, v4, v5, ninf)
        my_idx = _slot(cursor, i0, i1, i2, i3, i4, i5, 0)
        max_val = tl.max(my_val, axis=0)
        win_lane = tl.min(tl.where(my_val == max_val, lane, 64), axis=0)
        win_lane = tl.where(win_lane == 64, 0, win_lane)
        win_idx = tl.sum(tl.where(lane == win_lane, my_idx, 0), axis=0)
        i_won = (cursor < 6) & (my_idx == win_idx)
        my_orig = tl.where(i_won, _slot(cursor, o0, o1, o2, o3, o4, o5, 0.0), 0.0)
        cursor = cursor + i_won.to(tl.int32)
        src_lane = win_idx & 63
        weight_bits = tl.sum(
            tl.where(lane == src_lane, my_orig.to(tl.int32, bitcast=True), 0), axis=0
        )
        weight = weight_bits.to(tl.float32, bitcast=True)
        sel_val = tl.where(lane == k, weight, sel_val)
        sel_idx = tl.where(lane == k, win_idx, sel_idx)
        total = total + weight
    if RENORM:
        scale = routed_scaling_factor / tl.maximum(total, 1e-20)
    else:
        scale = routed_scaling_factor * 1.0
    out_mask = lane < TOPK
    tl.store(weights_ptr + row * stride_om + lane, sel_val * scale, mask=out_mask)
    tl.store(ids_ptr + row * stride_om + lane, sel_idx, mask=out_mask)


def rocm_router_gate(
    gating_output: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    topk: int,
    renormalize: bool,
    routed_scaling_factor: Optional[float],
    *,
    partials: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """aiter ``topk_gating(..., score_func="sqrtsoftplus")`` for 384 experts.

    Without ``partials``, ``gating_output`` ([M, 384] fp32 or bf16) is read; with ``partials``
    their fixed-order fp32 sum is the gating input and is written into ``gating_output`` (fp32
    [M, 384]). Returns fp32 weights and int32 ids, both [M, topk], identical to aiter's.
    """
    M, num_experts = gating_output.shape
    assert num_experts == _GATE_NUM_EXPERTS and 0 < topk <= _MAX_TOPK
    assert gating_output.stride(1) == 1
    if partials is not None:
        assert (
            partials.shape[1:] == (M, num_experts) and partials.dtype == torch.float32
        )
        assert gating_output.dtype == torch.float32
        split_k = partials.shape[0]
        stride_ps, stride_pm = partials.stride(0), partials.stride(1)
    else:
        partials = gating_output
        split_k = 0
        stride_ps = stride_pm = 0
    if correction_bias is not None:
        assert (
            correction_bias.shape == (num_experts,) and correction_bias.is_contiguous()
        )
    weights = torch.empty((M, topk), dtype=torch.float32, device=gating_output.device)
    ids = torch.empty((M, topk), dtype=torch.int32, device=gating_output.device)
    if M == 0:
        return weights, ids
    _router_gate_kernel[(M,)](
        gating_output,
        partials,
        correction_bias if correction_bias is not None else gating_output,
        weights,
        ids,
        gating_output.stride(0),
        stride_ps,
        stride_pm,
        topk,
        float(1.0 if routed_scaling_factor is None else routed_scaling_factor),
        SPLIT_K=split_k,
        WRITE_LOGITS=split_k > 0,
        HAS_BIAS=correction_bias is not None,
        RENORM=bool(renormalize),
        TOPK=topk,
        num_warps=1,
    )
    return weights, ids
