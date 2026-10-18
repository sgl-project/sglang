"""Muse BF16 pre-expert projection and IEEE FP32 router, with shared epilogue.

Three ordered GPU launches. Router IDs/weights preserve the Day0 sequence;
the BF16 projection uses a different FP32 Split-K partition. No input
quantization is added. Model dispatch supplies contiguous Muse tensors only.
"""

import torch
import triton as tr
import triton.language as tl

from sglang.srt.rocm_muse_spark.muse_router_gemv import _router_partial_kernel


@tr.jit
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


@tr.jit
def _dense_partial(
    X,
    W,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
    CACHE: tl.constexpr,
):
    pn, pm, pk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    k = pk * (K // SK) + tl.arange(0, BK)
    ap = X + m[:, None] * K + k[None, :]
    bp = W + n[None, :] * K + k[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(K // SK // BK):
        a = tl.load(ap, m[:, None] < M, other=0)
        b = tl.load(bp, cache_modifier=CACHE)
        acc = tl.dot(a, b, acc)
        ap += BK
        bp += BK
    tl.store(P + pk * M * N + m[:, None] * N + n[None, :], acc, m[:, None] < M)


@tr.jit
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


def router_preproject(
    x, router_weight, dense_weight, bias, norm_weight, multiplier, scale, eps
):
    m, k = x.shape
    if m > 16:
        return router_preproject_native_dense(
            x, router_weight, dense_weight, bias, norm_weight, multiplier, scale, eps
        )
    n = dense_weight.shape[0]
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
    _dense_partial[(n // 32, 1, 8)](
        x,
        dense_weight,
        dp,
        m,
        n,
        k,
        16,
        32,
        128,
        8,
        ".cg",
        num_warps=4,
        num_stages=2,
        matrix_instr_nonkdim=16,
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
        8,
        64,
        multiplier,
        scale,
        1,
        eps,
        num_warps=4,
    )
    return scores, ids, expert_input


def router_preproject_native_dense(
    x, router_weight, dense_weight, bias, norm_weight, multiplier, scale, eps
):
    """Share the norm/routing epilogue with hipBLASLt's completed BF16 output."""
    m, k = x.shape
    n = dense_weight.shape[0]
    assert 0 < m <= 64 and k == 8192 and n == 4096
    rp = torch.empty((64, m, 256), device=x.device, dtype=torch.float32)
    _router_partial_kernel[(2, 64, tr.cdiv(m, 16))](
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
    dp = torch.nn.functional.linear(x, dense_weight)
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
