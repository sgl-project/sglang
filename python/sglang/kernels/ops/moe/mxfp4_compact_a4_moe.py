"""Compact A4W4 MoE experts for small target-verify batches (MXFP4 weights and activations, gfx950).

Muse Spark EP8 / EP4 layout: 32 or 64 local experts, top-8 routing, hidden 4096, intermediate 4096, Quark MXFP4
checkpoint weights (per-32 E8M0 scales) used as stored. For M (a multiple of 8, up to MAX_ROWS) rows:

1. ``_prepare_a4``: per-expert route counts -> 16-row block map, and per-row MXFP4 quantization of the input;
2. ``_scatter_a4``: per local expert, rank its routes in token order into its blocks;
3. ``_expert_a4`` (ACT): gate/up ``dot_scaled`` GEMMs, SiLU-and-mul rounded to BF16, MXFP4 re-quantization;
4. ``_expert_a4``: down GEMM into unweighted per-(token, slot) BF16 rows.

Weights are read either as stored in the checkpoint or in aiter's FlyDSL a4w4 preshuffle layout (``shuffled``), so a
rank short on memory keeps one copy for both paths.

``compact_a4_experts`` returns those per-slot rows ``[M * 8, 4096]``; the caller's finalize applies the
per-expert norm and the route-weighted sum. Quark MXFP4 declares dynamic MXFP4 activations, so A4 is the
checkpoint numerics.
"""

import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

MAX_ROWS = 512
BLOCK_M = 16  # BM 32 / 64 give wrong rows and are no faster


@triton.jit
def _prepare_a4(
    X,
    Q,
    QS,
    EXPERT,
    NEXT,
    NB: tl.constexpr,
    NBB: tl.constexpr,
):
    # Program 0: clear the block->expert map and the block allocator for _scatter_a4.
    # Programs 1..4M quantize activation rows.
    pid = tl.program_id(0)
    if pid == 0:
        b = tl.arange(0, NBB)
        tl.store(EXPERT + b, -1, b < NB)
        tl.store(NEXT, 0)
    else:
        row = (pid - 1) // 4
        tile = (pid - 1) % 4
        col = tile * 1024 + tl.arange(0, 1024)
        x = tl.load(X + row * 4096 + col).to(tl.float32)
        q, qscale = _mxfp4_quant_op(x[None, :], 1024, 1, 32)
        packed_col = tile * 512 + tl.arange(0, 512)
        scale_col = tile * 32 + tl.arange(0, 32)
        tl.store(Q + row * 2048 + packed_col, tl.reshape(q, (512,)))
        tl.store(QS + row * 128 + scale_col, tl.reshape(qscale, (32,)))


@triton.jit
def _scatter_a4(
    IDS,
    SORTED,
    EXPERT,
    NEXT,
    M: tl.constexpr,
    START: tl.constexpr,
    S: tl.constexpr,
    PER_E: tl.constexpr,
    BM: tl.constexpr,
):
    # One program per local expert: count its routes, take its blocks from the shared allocator (block order
    # varies run to run; every block is computed and stored independently, so outputs do not), map them to the
    # expert and rank its routes in token order into them (sentinel M*8).
    e = tl.program_id(0)
    s = tl.arange(0, S)
    ids = tl.load(IDS + s, s < M * 8, other=-1)
    hit = (ids == START + e) & (s < M * 8)
    count = tl.sum(hit.to(tl.int32), axis=0)
    blocks = tl.cdiv(count, BM)
    if blocks > 0:
        first = tl.atomic_add(NEXT, blocks)
        eb = tl.arange(0, PER_E)
        tl.store(EXPERT + first + eb, e, eb < blocks)
        rank = tl.cumsum(hit.to(tl.int32), axis=0) - 1
        base = first * BM
        c = tl.arange(0, PER_E * BM)
        tl.store(SORTED + base + c, M * 8, c < blocks * BM)
        tl.debug_barrier()
        tl.store(SORTED + base + rank, s, hit)


@triton.jit
def _load_w(W, n0, k0, BN: tl.constexpr, KB: tl.constexpr, SHUF: tl.constexpr):
    """[KB bytes, BN rows] tile of one expert's [N, 2048] packed E2M1 rows from row n0, byte k0. Preshuffled layout:
    [N/16, 2048/16, 16 rows, 16 bytes] (aiter shuffle_weight (16, 16) and shuffle_weight_a16w4 (16) agree on it for
    K 2048), loaded as whole contiguous 16-row groups and permuted in registers."""
    if SHUF:
        g = tl.arange(0, BN // 16)
        kc = tl.arange(0, KB // 16)
        r = tl.arange(0, 16)
        b = tl.arange(0, 16)
        offset = (
            ((n0 // 16 + g) * 32768)[:, None, None, None]
            + ((k0 // 16 + kc) * 256)[None, :, None, None]
            + (r * 16)[None, None, :, None]
            + b[None, None, None, :]
        )
        x = tl.permute(tl.load(W + offset), (1, 3, 0, 2))
        return tl.reshape(x, (KB, BN))
    n = n0 + tl.arange(0, BN)
    k = k0 + tl.arange(0, KB)
    return tl.load(W + n[None, :] * 2048 + k[:, None])


@triton.jit
def _s_offset(n, c, SHUF: tl.constexpr):
    """[N, scale col] offsets in one expert's [N, 128] E8M0 scales. Preshuffled (aiter shuffle_scale, plain and a16w4
    alike): [N/32, 128/8, 4 (c % 4), 16 (n % 16), 2 ((c % 8) // 4), 2 ((n % 32) // 16)]."""
    if SHUF:
        n2, c2 = n[:, None], c[None, :]
        return (
            ((((n2 // 32) * 16 + c2 // 8) * 4 + c2 % 4) * 16 + n2 % 16) * 4
            + ((c2 % 8) // 4) * 2
            + (n2 % 32) // 16
        )
    return n[:, None] * 128 + c[None, :]


@triton.jit
def _expert_a4(
    A,
    AS,
    W,
    WS,
    SORTED,
    EXPERT,
    OUT,
    OS,
    ROWS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ACT: tl.constexpr,
    BM: tl.constexpr,
    SHUF: tl.constexpr,
):
    expert_block, n_block = tl.program_id(1), tl.program_id(0)
    expert = tl.load(EXPERT + expert_block)
    if expert >= 0:
        slot = tl.load(SORTED + expert_block * BM + tl.arange(0, BM))
        valid = slot < ROWS
        src = slot // 8 if ACT else slot
        n = n_block * BN + tl.arange(0, BN)
        kp = tl.arange(0, BK // 2)
        ks = tl.arange(0, BK // 32)
        wn: tl.constexpr = 8192 if ACT else 4096
        acc = tl.full((BM, BN), 0, tl.float32)
        if ACT:
            up = tl.full((BM, BN), 0, tl.float32)
        for tile in range(4096 // BK):
            a = tl.load(
                A + src[:, None] * 2048 + tile * (BK // 2) + kp[None, :],
                valid[:, None],
                other=0,
            )
            asc = tl.load(
                AS + src[:, None] * 128 + tile * (BK // 32) + ks[None, :],
                valid[:, None],
                other=127,
            )
            w = _load_w(
                W + expert * (wn * 2048),
                n_block * BN,
                tile * (BK // 2),
                BN,
                BK // 2,
                SHUF,
            )
            ws = tl.load(
                WS + expert * (wn * 128) + _s_offset(n, tile * (BK // 32) + ks, SHUF)
            )
            acc = tl.dot_scaled(a, asc, "e2m1", w, ws, "e2m1", acc)
            if ACT:
                wu = _load_w(
                    W + expert * (wn * 2048),
                    n_block * BN + 4096,
                    tile * (BK // 2),
                    BN,
                    BK // 2,
                    SHUF,
                )
                wsu = tl.load(
                    WS
                    + expert * (wn * 128)
                    + _s_offset(n + 4096, tile * (BK // 32) + ks, SHUF)
                )
                up = tl.dot_scaled(a, asc, "e2m1", wu, wsu, "e2m1", up)
        if ACT:
            activated = (acc * tl.sigmoid(acc) * up).to(tl.bfloat16).to(tl.float32)
            q, s = _mxfp4_quant_op(activated, BN, BM, 32)
            pn = n_block * (BN // 2) + tl.arange(0, BN // 2)
            sn = n_block * (BN // 32) + tl.arange(0, BN // 32)
            tl.store(OUT + slot[:, None] * 2048 + pn[None, :], q, valid[:, None])
            tl.store(OS + slot[:, None] * 128 + sn[None, :], s, valid[:, None])
        else:
            # Preserve the BF16 per-slot boundary before the Muse finalizer.
            tl.store(OUT + slot[:, None] * 4096 + n[None, :], acc, valid[:, None])


def compact_a4_applies(m: int, h: int, w13, w2, s13, s2) -> bool:
    e = w13.shape[0]
    return (
        0 < m <= MAX_ROWS
        and m % 8 == 0
        and h == 4096
        and e in (32, 64)
        and w13.shape == (e, 8192, 2048)
        and w2.shape == (e, 4096, 2048)
        and s13.shape == (e, 8192, 128)
        and s2.shape == (e, 4096, 128)
        and all(
            t.dtype == torch.uint8 and t.is_contiguous() for t in (w13, w2, s13, s2)
        )
    )


def compact_a4_experts(x, ids, w13, w2, s13, s2, *, local_start, shuffled=False):
    """Per-(token, slot) expert outputs BF16 [M * 8, 4096] for the 32 or 64 local experts starting at ``local_start``.

    Weights and scales are uint8 [E, N, K] views, as stored or (``shuffled``) in the a4w4 preshuffle layout."""
    m, h = x.shape
    if not (
        compact_a4_applies(m, h, w13, w2, s13, s2)
        and x.dtype == torch.bfloat16
        and ids.shape == (m, 8)
    ):
        raise ValueError(
            f"A4 requires M a multiple of 8 up to {MAX_ROWS}, local32 / local64 MXFP4 expert weights"
        )
    bm, e_local = BLOCK_M, w13.shape[0]
    rows, cap = m * 8, m * 8 + e_local * (bm - 1)
    nb = triton.cdiv(cap, bm)
    sid = torch.empty(cap, device=x.device, dtype=torch.int32)
    eid = torch.empty(nb, device=x.device, dtype=torch.int32)
    nxt = torch.empty(1, device=x.device, dtype=torch.int32)
    q = torch.empty((m, 2048), device=x.device, dtype=torch.uint8)
    qs = torch.empty((m, 128), device=x.device, dtype=torch.uint8)
    inter = torch.empty((rows, 2048), device=x.device, dtype=torch.uint8)
    inter_scale = torch.empty((rows, 128), device=x.device, dtype=torch.uint8)
    y = torch.empty((rows, h), device=x.device, dtype=torch.bfloat16)
    s_pow2, per_e = (
        triton.next_power_of_2(rows),
        triton.next_power_of_2(triton.cdiv(rows, bm)),
    )
    _prepare_a4[(1 + m * 4,)](
        x, q, qs, eid, nxt, nb, triton.next_power_of_2(nb), num_warps=4
    )
    _scatter_a4[(e_local,)](
        ids, sid, eid, nxt, m, local_start, s_pow2, per_e, bm, num_warps=4
    )
    grid = (4096 // 64, nb)
    options = dict(num_warps=4, num_stages=2, matrix_instr_nonkdim=16)
    _expert_a4[grid](
        q,
        qs,
        w13,
        s13,
        sid,
        eid,
        inter,
        inter_scale,
        rows,
        64,
        512,
        True,
        bm,
        shuffled,
        **options,
    )
    _expert_a4[grid](
        inter,
        inter_scale,
        w2,
        s2,
        sid,
        eid,
        y,
        y,
        rows,
        64,
        512,
        False,
        bm,
        shuffled,
        **options,
    )
    return y
