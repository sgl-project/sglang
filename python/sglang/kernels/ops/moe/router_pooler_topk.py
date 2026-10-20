"""Muse Spark MoE routing front: FP32 router + BF16 pooler projection with a shared top-8 / RMSNorm epilogue.

For M <= 64 rows of hidden ``x [M, 8192]`` (BF16): router logits ``fp32(x) @ W_router [8192, 256]`` (IEEE FP32
split-K), sigmoid scores, bias-corrected top-8 selection with normalized weights, and the pooled expert input
``rmsnorm(x @ W_pool.T)`` (``W_pool [4096, 8192]`` BF16, the projection rounded to BF16 before the norm).

* ``router_pooler_topk``: three ordered launches, router partials, pooler partials (both split-K) and one
  epilogue kernel whose programs either route a row or normalize a row. Above 16 rows the pooler output comes
  completed from hipBLASLt (or the tiled small-M GEMM when its copy is given) and only the epilogue is shared.
* ``joint_router_pooler_topk``: one split-K GEMM over a packed ``[pooler; router_hi; router_lo]`` BF16 weight
  (the FP32 router as high + low BF16 parts) and the same epilogue; 8 / 16 / 24-row verify batches.

Router ids follow the unfused routing sequence; the pooler and router projections differ from it only in
FP32 summation order (and, for the joint form, the two-part BF16 router weights).
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.bf16_splitk_gemm import bf16_splitk_gemm_partials
from sglang.kernels.ops.gemm.bf16_tiled_small_m_gemm import (
    tiled_gemm,
    tiled_gemm_applies,
)
from sglang.kernels.ops.gemm.router_gemm_fp32 import _router_partial_kernel


@triton.jit
def _reduce_route(
    P,
    BIAS,
    SCORES,
    IDS,
    M: tl.constexpr,
    N: tl.constexpr,
    SK: tl.constexpr,
    MULT: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    TREE: tl.constexpr,
):
    m = tl.program_id(0)
    n = tl.arange(0, N)
    if TREE:
        s = tl.arange(0, SK)
        p = tl.load(P + s[:, None] * M * N + m * N + n[None, :])
        logits = tl.sum(p, axis=0)
    else:
        logits = tl.zeros((N,), tl.float32)
        for s in tl.static_range(SK):
            logits += tl.load(P + s * M * N + m * N + n)
    scores = tl.sigmoid(logits * MULT)
    biased = scores + tl.load(BIAS + n)
    ks = tl.arange(0, 8)
    selected_ids = tl.full((8,), 0, tl.int32)
    selected_scores = tl.full((8,), 0, tl.float32)
    denom = tl.full((), 0, tl.float32)
    for k in tl.static_range(8):
        idx = tl.argmax(biased, axis=0)
        is_selected = n == idx
        score = tl.sum(tl.where(is_selected, scores, 0), axis=0)
        denom += score
        biased = tl.where(is_selected, -float("inf"), biased)
        selected_ids = tl.where(ks == k, idx, selected_ids)
        selected_scores = tl.where(ks == k, score, selected_scores)
    inv = SCALE / (denom + EPS)
    tl.store(SCORES + m * 8 + ks, selected_scores * inv)
    tl.store(IDS + m * 8 + ks, selected_ids)


@triton.jit
def _route_preproject_epilogue(
    RP,
    DP,
    BIAS,
    NORM,
    SCORES,
    IDS,
    Y,
    M: tl.constexpr,
    DN: tl.constexpr,
    DS: tl.constexpr,
    RS: tl.constexpr,
    MULT: tl.constexpr,
    SCALE: tl.constexpr,
    REPS: tl.constexpr,
    NORM_EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < M:
        _reduce_route(RP, BIAS, SCORES, IDS, M, 256, RS, MULT, SCALE, 1e-15, False)
    else:
        row = (pid - M) // REPS
        replica = (pid - M) % REPS
        n = tl.arange(0, DN)
        s = tl.arange(0, DS)
        part = tl.load(DP + s[:, None] * M * DN + row * DN + n[None, :])
        value = tl.sum(part, axis=0).to(tl.bfloat16).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(value * value, axis=0) / DN + NORM_EPS)
        weight = tl.load(NORM + n).to(tl.float32)
        result = value * rstd * weight
        tl.store(Y + row * DN + n, result, n // (DN // REPS) == replica)


@triton.jit
def _joint_epilogue(
    P,
    BIAS,
    NORM,
    SCORES,
    IDS,
    Y,
    M: tl.constexpr,
    SK: tl.constexpr,
    MULT: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    s = tl.arange(0, SK)
    if pid < M:
        n = tl.arange(0, 256)
        high = tl.load(P + s[:, None] * M * 4608 + pid * 4608 + 4096 + n[None, :])
        low = tl.load(P + s[:, None] * M * 4608 + pid * 4608 + 4352 + n[None, :])
        logits = tl.sum(high, axis=0) + tl.sum(low, axis=0)
        scores = tl.sigmoid(logits * MULT)
        biased = scores + tl.load(BIAS + n)
        slots = tl.arange(0, 8)
        ids = tl.full((8,), 0, tl.int32)
        selected = tl.full((8,), 0, tl.float32)
        denominator = tl.full((), 0, tl.float32)
        for j in tl.static_range(8):
            idx = tl.argmax(biased, axis=0)
            score = tl.sum(tl.where(n == idx, scores, 0), axis=0)
            denominator += score
            biased = tl.where(n == idx, -float("inf"), biased)
            ids = tl.where(slots == j, idx, ids)
            selected = tl.where(slots == j, score, selected)
        tl.store(IDS + pid * 8 + slots, ids)
        tl.store(SCORES + pid * 8 + slots, selected * (SCALE / (denominator + 1e-15)))
    else:
        row = pid - M
        dn = tl.arange(0, 4096)
        partial = tl.load(P + s[:, None] * M * 4608 + row * 4608 + dn[None, :])
        value = tl.sum(partial, axis=0).to(tl.bfloat16).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(value * value, axis=0) / 4096 + EPS)
        result = value * rstd * tl.load(NORM + dn).to(tl.float32)
        tl.store(Y + row * 4096 + dn, result)


def router_pooler_topk(
    x,
    router_weight,
    pooler_weight,
    bias,
    norm_weight,
    multiplier,
    scale,
    eps,
    pooler_tiled=None,
):
    """Return (top-8 scores FP32 [M, 8], expert ids int32 [M, 8], normalized expert input BF16 [M, 4096])."""
    m, k = x.shape
    if m > 16:
        return _router_topk_completed_pooler(
            x,
            router_weight,
            pooler_weight,
            bias,
            norm_weight,
            multiplier,
            scale,
            eps,
            pooler_tiled,
        )
    n = pooler_weight.shape[0]
    assert 0 < m <= 16 and k == 8192 and n == 4096
    rp = torch.empty((64, m, 256), device=x.device, dtype=torch.float32)
    dp = torch.empty((8, m, n), device=x.device, dtype=torch.float32)
    _router_partial_kernel[(2, 64, 1)](
        x,
        router_weight,
        rp,
        m,
        256,
        k,
        x.stride(0),
        router_weight.stride(0),
        rp.stride(0),
        rp.stride(1),
        BLOCK_M=16,
        BLOCK_N=128,
        BLOCK_K=64,
        SPLIT_K=64,
        num_warps=4,
        num_stages=2,
    )
    bf16_splitk_gemm_partials(x, pooler_weight, dp, block_n=32, block_k=128, split_k=8)
    scores = torch.empty((m, 8), device=x.device, dtype=torch.float32)
    ids = torch.empty((m, 8), device=x.device, dtype=torch.int32)
    expert_input = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    _route_preproject_epilogue[(2 * m,)](
        rp,
        dp,
        bias,
        norm_weight,
        scores,
        ids,
        expert_input,
        m,
        n,
        8,
        64,
        multiplier,
        scale,
        1,
        eps,
        num_warps=4,
    )
    return scores, ids, expert_input


def _router_topk_completed_pooler(
    x,
    router_weight,
    pooler_weight,
    bias,
    norm_weight,
    multiplier,
    scale,
    eps,
    pooler_tiled,
):
    """Share the norm/routing epilogue with a completed BF16 pooler output (tiled GEMM or hipBLASLt)."""
    m, k = x.shape
    n = pooler_weight.shape[0]
    assert 0 < m <= 64 and k == 8192 and n == 4096
    rp = torch.empty((64, m, 256), device=x.device, dtype=torch.float32)
    _router_partial_kernel[(2, 64, triton.cdiv(m, 16))](
        x,
        router_weight,
        rp,
        m,
        256,
        k,
        x.stride(0),
        router_weight.stride(0),
        rp.stride(0),
        rp.stride(1),
        BLOCK_M=16,
        BLOCK_N=128,
        BLOCK_K=64,
        SPLIT_K=64,
        num_warps=4,
        num_stages=2,
    )
    dp = (
        tiled_gemm(x, pooler_tiled, n, k)
        if tiled_gemm_applies(x, pooler_tiled)
        else torch.nn.functional.linear(x, pooler_weight)
    )
    scores = torch.empty((m, 8), device=x.device, dtype=torch.float32)
    ids = torch.empty((m, 8), device=x.device, dtype=torch.int32)
    expert_input = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    _route_preproject_epilogue[(2 * m,)](
        rp,
        dp,
        bias,
        norm_weight,
        scores,
        ids,
        expert_input,
        m,
        n,
        1,
        64,
        multiplier,
        scale,
        1,
        eps,
        num_warps=4,
    )
    return scores, ids, expert_input


def pack_joint_router_pooler_weight(router, pooler):
    """``[pooler; router.T high BF16; router.T low BF16]`` -> [4608, 8192] BF16."""
    assert router.shape == (8192, 256) and router.dtype == torch.float32
    assert pooler.shape == (4096, 8192) and pooler.dtype == torch.bfloat16
    high = router.T.to(torch.bfloat16)
    low = (router.T - high.float()).to(torch.bfloat16)
    return torch.cat([pooler, high, low], dim=0).contiguous()


def joint_router_pooler_topk(x, packed, bias, norm, multiplier, scale, eps):
    """Same outputs as ``router_pooler_topk`` from the packed joint weight, for 8 / 16 / 24 rows."""
    m, k = x.shape
    assert m in (8, 16, 24) and k == 8192 and packed.shape == (4608, 8192)
    if m == 24:
        from aiter.ops.triton.gemm_a16w16 import gemm_a16w16

        partial = gemm_a16w16(x, packed, dtype=torch.float32, skip_reduce=True)
        sk = partial.shape[0] if partial.ndim == 3 else 1
    else:
        sk = 8
        partial = torch.empty((sk, m, 4608), device=x.device, dtype=torch.float32)
        bf16_splitk_gemm_partials(
            x, packed, partial, block_n=64, block_k=128, split_k=sk
        )
    assert partial.dtype == torch.float32 and sk & (sk - 1) == 0
    scores = torch.empty((m, 8), device=x.device, dtype=torch.float32)
    ids = torch.empty((m, 8), device=x.device, dtype=torch.int32)
    expert_input = torch.empty((m, 4096), device=x.device, dtype=torch.bfloat16)
    _joint_epilogue[(2 * m,)](
        partial,
        bias,
        norm,
        scores,
        ids,
        expert_input,
        m,
        sk,
        multiplier,
        scale,
        eps,
        num_warps=4,
    )
    return scores, ids, expert_input
