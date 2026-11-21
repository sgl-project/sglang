"""Muse Spark MoE routing front: FP32 router + BF16 pooler projection with a shared top-8 / RMSNorm epilogue.

For M <= 64 rows of hidden ``x [M, 8192]`` (BF16): router logits ``fp32(x) @ W_router [8192, 256]`` (IEEE FP32
split-K), sigmoid scores, bias-corrected top-8 selection with normalized weights, and the pooled expert input
``rmsnorm(x @ W_pool.T)`` (``W_pool [4096, 8192]`` BF16, the projection rounded to BF16 before the norm).

* ``router_pooler_topk``: three ordered launches, router partials, pooler partials (both split-K) and one
  epilogue kernel whose programs either route a row or normalize a row. Above 16 rows the pooler output comes
  completed from hipBLASLt (or the tiled small-M GEMM when its copy is given) and only the epilogue is shared.
* ``joint_router_pooler_topk``: one split-K GEMM over a packed ``[pooler; router_hi; router_lo]`` BF16 weight
  (the FP32 router as high + low BF16 parts) and the same epilogue; 8 / 16 / 24-row verify batches.
* ``joint_router_pooler_topk_fp8``: the joint form with the pooler tiles reading an FP8 e4m3 copy of the pooler
  weight (``quantize_joint_pooler``) and the per-row FP8 activation the boundary kernel produces; the router tiles
  keep the BF16 activation and the high / low router rows, so the routing is unchanged and the pooler streams
  half the bytes.

Router ids follow the unfused routing sequence; the pooler and router projections differ from it only in
FP32 summation order (and, for the joint form, the two-part BF16 router weights).
"""

import functools

import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

from sglang.kernels.arch import current_arch
from sglang.kernels.ops.gemm.bf16_splitk_gemm import bf16_splitk_gemm_partials
from sglang.kernels.ops.gemm.bf16_tiled_small_m_gemm import (
    _bm,
    flydsl_gemm,
    flydsl_gemm_applies,
    rowmajor_gemm_applies,
    rowmajor_gemm_partials,
    rowmajor_gemm_partials_split,
    tiled_gemm,
    tiled_gemm_applies,
)
from sglang.kernels.ops.gemm.fp8w_small_m_gemm import (
    quantize_weight as quantize_fp8_weight,
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
    Q,
    QS,
    M: tl.constexpr,
    DN: tl.constexpr,
    DS: tl.constexpr,
    RS: tl.constexpr,
    MULT: tl.constexpr,
    SCALE: tl.constexpr,
    REPS: tl.constexpr,
    NORM_EPS: tl.constexpr,
    QUANT: tl.constexpr,
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
        if QUANT:
            # The compact A4 experts' per-row MXFP4 input, from the BF16 row as stored.
            q, qs = _mxfp4_quant_op(
                result.to(tl.bfloat16).to(tl.float32)[None, :], DN, 1, 32
            )
            tl.store(
                Q + row * (DN // 2) + tl.arange(0, DN // 2), tl.reshape(q, (DN // 2,))
            )
            tl.store(
                QS + row * (DN // 32) + tl.arange(0, DN // 32),
                tl.reshape(qs, (DN // 32,)),
            )


@triton.jit
def _joint_epilogue(
    P,
    BIAS,
    NORM,
    SCORES,
    IDS,
    Y,
    Q,
    QS,
    XS,
    WS,
    M: tl.constexpr,
    SK: tl.constexpr,
    MULT: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    QUANT: tl.constexpr,
    FP8: tl.constexpr,
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
        value = tl.sum(partial, axis=0)
        if FP8:
            # FP8 pooler tiles: the row's activation scale and the per-output-channel weight scale.
            value = value * tl.load(XS + row) * tl.load(WS + dn)
        value = value.to(tl.bfloat16).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(value * value, axis=0) / 4096 + EPS)
        result = value * rstd * tl.load(NORM + dn).to(tl.float32)
        tl.store(Y + row * 4096 + dn, result)
        if QUANT:
            # The compact A4 experts' per-row MXFP4 input, from the BF16 row as stored.
            q, qs = _mxfp4_quant_op(
                result.to(tl.bfloat16).to(tl.float32)[None, :], 4096, 1, 32
            )
            tl.store(Q + row * 2048 + tl.arange(0, 2048), tl.reshape(q, (2048,)))
            tl.store(QS + row * 128 + tl.arange(0, 128), tl.reshape(qs, (128,)))


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
    pooler_rowmajor=False,
    quant_out=None,
):
    """Return (top-8 scores FP32 [M, 8], expert ids int32 [M, 8], normalized expert input BF16 [M, 4096]).

    ``quant_out`` = (uint8 [M, 2048], uint8 [M, 128]) also receives the expert input's per-row MXFP4 data and
    E8M0 scales (the compact A4 experts' input)."""
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
            pooler_rowmajor,
            quant_out,
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
        *(quant_out or (expert_input, expert_input)),
        m,
        n,
        8,
        64,
        multiplier,
        scale,
        1,
        eps,
        quant_out is not None,
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
    pooler_rowmajor,
    quant_out,
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
    if pooler_rowmajor and rowmajor_gemm_applies(x, pooler_weight):
        # the epilogue sums the split-K partials itself (no separate sum kernel)
        dp = rowmajor_gemm_partials(x, pooler_weight)
    elif tiled_gemm_applies(x, pooler_tiled):
        dp = tiled_gemm(x, pooler_tiled, n, k)
    else:
        dp = torch.nn.functional.linear(x, pooler_weight)
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
        *(quant_out or (expert_input, expert_input)),
        m,
        n,
        dp.shape[0] if dp.dim() == 3 else 1,
        64,
        multiplier,
        scale,
        1,
        eps,
        quant_out is not None,
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


def joint_router_pooler_applies(x, packed) -> bool:
    """Rows the joint path serves: 8 / 16 / 24, and the widths with a split-K row-major launch (FP32 partials keep
    the router's high + low halves exact)."""
    m = x.shape[0]
    return m in (8, 16, 24) or (
        rowmajor_gemm_applies(x, packed)
        and rowmajor_gemm_partials_split(m, *packed.shape) > 1
    )


def joint_router_pooler_topk(
    x, packed, bias, norm, multiplier, scale, eps, quant_out=None
):
    """Same outputs as ``router_pooler_topk`` from the packed joint weight (``joint_router_pooler_applies`` rows).
    ``quant_out`` as in ``router_pooler_topk``."""
    m, k = x.shape
    assert k == 8192 and packed.shape == (4608, 8192)
    if flydsl_gemm_applies(x, packed, torch.float32):
        partial = flydsl_gemm(x, packed)[None]
        sk = 1
    elif m not in (8, 16, 24):
        partial = rowmajor_gemm_partials(x, packed)
        sk = partial.shape[0]
    elif m == 24:
        from aiter.ops.triton.gemm_a16w16 import gemm_a16w16

        partial = gemm_a16w16(x, packed, dtype=torch.float32, skip_reduce=True)
        sk = partial.shape[0] if partial.ndim == 3 else 1
    else:
        sk = 8
        partial = torch.empty((sk, m, 4608), device=x.device, dtype=torch.float32)
        bf16_splitk_gemm_partials(
            x, packed, partial, block_n=64, block_k=128, split_k=sk
        )
    return _joint_finish(partial, sk, m, bias, norm, multiplier, scale, eps, quant_out)


# Warps per epilogue program; one program sums a 4096-wide row over SK partials (64 KB at SK 4).
JOINT_EPILOGUE_WARPS = 4


def _joint_finish(
    partial, sk, m, bias, norm, multiplier, scale, eps, quant_out, fp8_scales=None
):
    """The shared epilogue over the joint partials ``[SK, M, 4608]``; ``fp8_scales`` = (row activation scales,
    pooler channel scales) when the pooler tiles came from the FP8 GEMM."""
    assert partial.dtype == torch.float32 and sk & (sk - 1) == 0
    scores = torch.empty((m, 8), device=partial.device, dtype=torch.float32)
    ids = torch.empty((m, 8), device=partial.device, dtype=torch.int32)
    expert_input = torch.empty((m, 4096), device=partial.device, dtype=torch.bfloat16)
    _joint_epilogue[(2 * m,)](
        partial,
        bias,
        norm,
        scores,
        ids,
        expert_input,
        *(quant_out or (expert_input, expert_input)),
        *(fp8_scales or (expert_input, expert_input)),
        m,
        sk,
        multiplier,
        scale,
        eps,
        quant_out is not None,
        fp8_scales is not None,
        num_warps=JOINT_EPILOGUE_WARPS,
    )
    return scores, ids, expert_input


# FP8 joint form: one launch whose first 4096 / BN programs per K split run the pooler tile from the FP8
# activation and weight and whose remaining programs run the router rows from BF16, all writing the same
# ``[SK, M, 4608]`` partials. Launch rules from a kernelforge campaign on MI355X, accepted on every M in 17..512
# with cold (HBM-resident) weights; see joint_router_pooler_topk_fp8.
# BM -> (TILE_M, BN, BK, SK, warps, stages, BN_R, BK_R, waves_per_eu or 0, R_MS).
# M <= 128: few wide programs with deep pipelines (1 WG/CU, ~125-150 KB LDS) beat many small ones: BN128 cuts the
# A-tile re-staging (A L2 traffic ~ 4096/BN * M * K), and the BF16 router gets a narrower tile (BN_R 32, 16 programs
# per K slice) so router programs stream half the pooler's weight bytes and never sit on the critical path.
# 192 programs at SK4. Measured (GEMM + SK sum): M24 18.0 -> 14.7 us, M40 21.7 -> 15.7, M128 31.0 -> 21.7.
# M > 128: one 256-row M tile per weight tile (the 128-row tiles re-streamed every weight byte per M tile from HBM),
# with the router's 256 rows split R_MS=2 ways so its bf16-A-heavy programs are no longer than the pooler's.
# Measured (GEMM + SK sum): M168 36.1 -> 25.9 us, M224 40.4 -> 28.2, M320 49.3 -> 42.3, M512 71.3 -> 52.5.
FP8_JOINT = {
    32: (32, 128, 256, 4, 8, 3, 32, 128, 0, 1),
    64: (64, 128, 256, 4, 8, 3, 32, 128, 0, 1),
    128: (128, 128, 128, 4, 8, 3, 32, 128, 0, 1),
    256: (256, 128, 128, 4, 8, 3, 64, 128, 0, 2),
    512: (256, 128, 128, 4, 8, 3, 64, 128, 0, 2),
}
FP8_JOINT_MIN_M = 17
FP8_JOINT_MAX_M = 512


def _xcd_map(m: int) -> int:
    # Rules over M from an interleaved GEMM+SK-sum sweep of every multiple of 8 in 24..512 (+17, 23, 33, 41, 97, 217),
    # 8 cold weight sets, deltas vs canonical: map 1 -0.5..-0.9 us at M <= 40; map 2 -0.1..-0.6 us at 41..96 and
    # -0.2..-3.2 us above 216. In between (97..216) both maps slow the following SK sum (+0.2..+2.6 us), so canonical.
    if m <= 40:
        return 1
    if m <= 96 or m > 216:
        return 2
    return 0


@triton.jit
def _joint_fp8_partial(
    XQ,
    X,
    WQ,
    WR,
    P,
    M,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
    POOLER_TILES: tl.constexpr,
    BN_R: tl.constexpr,
    BK_R: tl.constexpr,
    R_MS: tl.constexpr,
    XCD_MAP: tl.constexpr = 0,
):
    N: tl.constexpr = 4608
    K: tl.constexpr = 8192
    KT_PER: tl.constexpr = K // BK // SK
    KT_R: tl.constexpr = K // BK_R // SK
    pid0, pid1, pm = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # XCD-aware program-ID remap. The hardware deals workgroups round-robin over the 8 XCDs by linear ID
    # hw = pid0 + G0*pid1 (+ G0*SK*pid2; G0*SK % 8 == 0 keeps hw % 8 the XCD and each pm slab a separate bijection).
    # Canonically XCD x gets every pk for pn = x mod 8, so every XCD streams the whole activation into its own L2.
    #   XCD_MAP 1: XCD x owns one K slice (pk = x // 2): activation L2 traffic / 4. Best at small M.
    #   XCD_MAP 2: each XCD owns 2 K slices, each pk is spread over 4 XCDs, and every XCD keeps the canonical
    #              16 pooler + 8 router mix with pooler dispatched first. Halves the activation traffic without the
    #              SK-sum slowdown map 1 causes at larger M.
    # Every map is a bijection onto (pn, pk), so results are bitwise identical to the canonical order.
    G0: tl.constexpr = POOLER_TILES + R_MS * (512 // BN_R)
    if XCD_MAP == 1 and (G0 * SK) % 8 == 0:
        PER: tl.constexpr = G0 * SK // 8
        hw = pid0 + G0 * pid1
        lin = (hw % 8) * PER + hw // 8
        pn = lin % G0
        pk = lin // G0
    elif XCD_MAP == 2 and SK == 4 and G0 % 4 == 0:
        hw = pid0 + G0 * pid1
        xcd = hw % 8
        j = hw // 8
        pk = 2 * (xcd % 2) + j % 2
        pn = 4 * (j // 2) + xcd // 2
    else:
        pn = pid0
        pk = pid1
    k0 = pk * (K // SK)
    if pn < POOLER_TILES:
        # FP8 pooler tile: (BM, BN) over this program's K slice.
        m = pm * BM + tl.arange(0, BM)
        n = pn * BN + tl.arange(0, BN)
        kk = k0 + tl.arange(0, BK)
        xq_ptr = XQ + m[:, None] * K + kk[None, :]
        wq_ptr = WQ + n[None, :].to(tl.int64) * K + kk[:, None]
        acc = tl.zeros((BM, BN), tl.float32)
        for _ in range(KT_PER):
            aq = tl.load(xq_ptr, mask=m[:, None] < M, other=0.0)
            bq = tl.load(wq_ptr, cache_modifier=".cg")
            acc = tl.dot(aq, bq, acc)
            xq_ptr += BK
            wq_ptr += BK
        tl.store(P + pk * M * N + m[:, None] * N + n[None, :], acc, mask=m[:, None] < M)
    else:
        # BF16 router tile: its own (BN_R, BK_R) and an R_MS-way row split of the M tile, so router programs (whose
        # bf16 A slice dominates their bytes) stay no longer than the pooler programs.
        RT: tl.constexpr = 512 // BN_R
        BM_R: tl.constexpr = BM // R_MS
        r = pn - POOLER_TILES
        nt = r % RT
        ms = r // RT
        mr = pm * BM + ms * BM_R + tl.arange(0, BM_R)
        nr = nt * BN_R + tl.arange(0, BN_R)
        kr = k0 + tl.arange(0, BK_R)
        x_ptr = X + mr[:, None] * K + kr[None, :]
        wr_ptr = WR + nr[None, :].to(tl.int64) * K + kr[:, None]
        accr = tl.zeros((BM_R, BN_R), tl.float32)
        for _ in range(KT_R):
            ax = tl.load(x_ptr, mask=mr[:, None] < M, other=0.0)
            bx = tl.load(wr_ptr, cache_modifier=".cg")
            accr = tl.dot(ax, bx, accr)
            x_ptr += BK_R
            wr_ptr += BK_R
        tl.store(
            P + pk * M * N + mr[:, None] * N + (nr + 4096)[None, :],
            accr,
            mask=mr[:, None] < M,
        )


def quantize_joint_pooler(packed):
    """(FP8 e4m3 pooler [4096, 8192], FP32 per-output-channel scales [4096]) of the packed weight's pooler prefix."""
    assert packed.shape == (4608, 8192) and packed.dtype == torch.bfloat16
    quantized = quantize_fp8_weight(packed[:4096].contiguous())
    assert quantized is not None
    return quantized


@functools.lru_cache(maxsize=1)
def _fp8_joint_runs_here() -> bool:
    """The FP8 joint kernel targets gfx950: its launch table needs up to ~150 KB of LDS per workgroup (gfx950 has
    160 KB per CU) and it uses OCP FP8 e4m3. Elsewhere callers fall back to the BF16 joint / shared forms."""
    hip = current_arch().hip
    return hip is not None and hip.gcn_arch == "gfx950"


def joint_router_pooler_fp8_applies(xq, packed, pooler_fp8) -> bool:
    """Rows the FP8 joint form serves: the per-row FP8 activation ``xq`` of up to ``FP8_JOINT_MAX_M`` rows."""
    return (
        pooler_fp8 is not None
        and _fp8_joint_runs_here()
        and xq.dim() == 2
        and xq.is_contiguous()
        and xq.dtype == torch.float8_e4m3fn
        and xq.shape[1] == 8192
        and FP8_JOINT_MIN_M <= xq.shape[0] <= FP8_JOINT_MAX_M
        and packed.shape == (4608, 8192)
    )


def joint_router_pooler_topk_fp8(
    xq, xs, x, packed, pooler_fp8, bias, norm, multiplier, scale, eps, quant_out=None
):
    """``joint_router_pooler_topk`` with the pooler projection from the per-row FP8 activation ``xq`` / scales
    ``xs`` and the FP8 pooler copy ``pooler_fp8`` (``quantize_joint_pooler``); the router reads the BF16 ``x``."""
    m, k = x.shape
    assert k == 8192 and packed.shape == (4608, 8192) and xq.shape == x.shape
    wq, ws = pooler_fp8
    tile_m, bn, bk, sk, warps, stages, bn_r, bk_r, wpe, r_ms = FP8_JOINT[_bm(m)]
    partial = torch.empty((sk, m, 4608), device=x.device, dtype=torch.float32)
    extra = {"waves_per_eu": wpe} if wpe else {}
    _joint_fp8_partial[(4096 // bn + r_ms * (512 // bn_r), sk, triton.cdiv(m, tile_m))](
        xq,
        x,
        wq,
        packed[4096:],
        partial,
        m,
        tile_m,
        bn,
        bk,
        sk,
        4096 // bn,
        bn_r,
        bk_r,
        r_ms,
        _xcd_map(m),
        num_warps=warps,
        num_stages=stages,
        matrix_instr_nonkdim=16,
        **extra,
    )
    return _joint_finish(
        partial, sk, m, bias, norm, multiplier, scale, eps, quant_out, (xs, ws)
    )
