# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Fused Triton sparse-MLA prefill for DeepSeek Sparse Attention (DSA).

Relationship to ``dsa/triton_sparse_mla.py``
--------------------------------------------
That module implements the same core form — one Triton program per query token,
online softmax held in registers, no split/merge — and reaches it for the same
reason: after TP the attention tile is tiny, so a small per-token program beats
a wide cooperative block. It is reachable only on gfx950, only for an FP8 KV
cache, and only at one pinned shape (16 heads, ``d_v`` 512, tail 64, topk 2048),
so on NVIDIA the DSA prefill never takes it.

This module is the CUDA-side sibling: bf16 (the FP8 path quantises ``P`` before
the ``PV`` product; this one does not), no shape pin, a concatenated ``q``
matching the FlashMLA entry signature, and two exact fast paths the gfx950 path
does not have. On the base path the two are within ~10% of each other on real
captured indices; the separation comes from ``union`` (see below).

Against the kernels NVIDIA DSA prefill actually dispatches to, it removes two
structural costs:

* ``flash_mla_sparse_fwd`` requires ``num_heads % 64`` (Hopper) or ``% 128``
  (Blackwell). After TP the model has far fewer heads (8 at TP8), so the head
  dim is zero-padded and 7/8 (resp. 15/16) of the tensor-core work is wasted.
* The split-decode form writes ``O(T * splits)`` partials to HBM and merges
  them in a second kernel.

This kernel runs **one program per query token** with a ``BLOCK_H=16`` head tile
(pad-into-tile, not pad-into-grid), keeps the online softmax entirely in
registers, and reuses ``V`` as ``K[:, :d_v]`` (MLA latent) so the value rows are
never gathered twice. No global partials, no merge pass.

The ``BLOCK_H`` floor of 16 is deliberate: dropping it to the exact head count
at H=8 produces bitwise-identical output but runs ~7% slower, because the
narrower MMA does not pay for the register pressure it saves.

Interface matches ``sgl_kernel.flash_mla.flash_mla_sparse_fwd``::

    q       [T, H, 576] bf16    (absorbed MLA: 512 nope + 64 rope)
    kv      [S, 576]    bf16    (V is kv[:, :512])
    indices [T, topk]   int32   (-1 or >= S marks an invalid slot)
    out     [T, H, 512] bf16

A row of ``indices`` must not name the same KV position twice. Top-k selection
cannot, so this holds for every DSA caller; it is stated because ``union``
gathers the distinct union of G rows and weights each position once, whereas
the base path would weight a repeat twice.

Optional exact fast paths (opt-in, default off; both are algebraically
equivalent to the base path, not approximations):

``dense_prefix``
    Tokens whose top-k selects the whole causal prefix (``t + 1 <= topk``) are
    dense causal attention by definition; run FA-style tiles with zero gather
    behind an exact count/min/max guard that re-verifies the selection set.
``union``
    ``G`` adjacent query tokens share one gathered union index set; a per-row
    ownership bitmask restores the exact per-token softmax. This is where the
    win lives: on real GLM-5.1 captures the union of 4 neighbouring tokens'
    selections is only ~1.03x the size of one token's, because the indexer's
    scores move slowly in ``t``, so one gather serves four tokens. Uniformly
    random indices do not have that structure and understate the path badly —
    benchmark it on captured indices, not synthetic ones.

All tunables are explicit arguments — the module reads no environment
variables. If a tuned tile exceeds the device shared-memory budget (e.g. a
large head count on SM120's 100 KB), the launcher steps down through smaller
tiles instead of raising ``OutOfResources``.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Swept on the hardware named; anything else gets _UNTUNED_DEFAULT and a warning.
_PINNED = {
    (9, 0): (64, 8, 2),  # SM90, swept at T=8192
    (12, 0): (64, 4, 2),  # SM120, swept at T=8192
    (12, 1): (64, 4, 2),
}
_UNTUNED_DEFAULT = (64, 8, 3)


@triton.jit
def _nsa_prefill_kernel(
    q_ptr,
    kv_ptr,
    idx_ptr,
    len_ptr,
    o_ptr,
    scale_ptr,  # FP8 only: [s_qk = qs*ks, s_k = ks] fp32; unused when not FP8
    sm_scale,
    topk,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8: tl.constexpr,
    MATH_BF16: tl.constexpr,  # with FP8: cast tiles to bf16 after load (halve L1/L2
    # gather bytes, keep the SM90-fast bf16 mma path)
    IDX64: tl.constexpr,  # int64 row addressing only when the KV pool can overflow
    # int32*D_QK (rows > ~3.7M). int32 fast path: SASS drops
    # 36x IMAD.WIDE -> IMAD in the hot gather loop
    # (cuda-agent step-001, -97us z=8.11 @ T=8192).
):
    t = tl.program_id(0)
    D_TAIL: tl.constexpr = D_QK - D_V

    if FP8:
        # inputs pre-scaled by 448/amax in the wrapper; undo inside the math:
        # qk_real = qk_fp8 * qs*ks/448^2 ; pv_real = pv_fp8 * ks/448^2 (P carries x448)
        qk_scale = sm_scale * tl.load(scale_ptr) / (448.0 * 448.0)
        out_scale = tl.load(scale_ptr + 1) / (448.0 * 448.0)
    else:
        qk_scale = sm_scale
        out_scale = 1.0

    h = tl.arange(0, BLOCK_H)
    hmask = h < H
    dv = tl.arange(0, D_V)
    dt = tl.arange(0, D_TAIL)

    qb = q_ptr + t * H * D_QK
    q_main = tl.load(
        qb + h[:, None] * D_QK + dv[None, :], mask=hmask[:, None], other=0.0
    )
    q_tail = tl.load(
        qb + h[:, None] * D_QK + (D_V + dt)[None, :], mask=hmask[:, None], other=0.0
    )
    if FP8 and MATH_BF16:
        q_main = q_main.to(tl.bfloat16)
        q_tail = q_tail.to(tl.bfloat16)

    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D_V], tl.float32)

    n = tl.arange(0, BLOCK_N)
    k_len = tl.load(len_ptr + t)
    for k0 in tl.range(0, k_len, BLOCK_N):
        idx = tl.load(idx_ptr + t * topk + k0 + n, mask=(k0 + n) < k_len, other=-1)
        valid = idx >= 0
        if IDX64:
            row = tl.where(valid, idx, 0).to(tl.int64)
        else:
            row = tl.where(valid, idx, 0)
        kb = kv_ptr + row[:, None] * D_QK
        kv_main = tl.load(kb + dv[None, :], mask=valid[:, None], other=0.0)
        kv_tail = tl.load(kb + (D_V + dt)[None, :], mask=valid[:, None], other=0.0)
        if FP8 and MATH_BF16:
            kv_main = kv_main.to(tl.bfloat16)
            kv_tail = kv_tail.to(tl.bfloat16)

        qk = tl.dot(q_main, tl.trans(kv_main))
        qk = tl.dot(q_tail, tl.trans(kv_tail), qk) * qk_scale
        qk = tl.where(valid[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if FP8:
            p_q = (p * 448.0).to(kv_main.dtype)
        else:
            p_q = p.to(kv_main.dtype)
        acc = acc * alpha[:, None] + tl.dot(p_q, kv_main)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc * (out_scale / l_safe[:, None])
    tl.store(
        o_ptr + t * H * D_V + h[:, None] * D_V + dv[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=hmask[:, None],
    )


# ---------------------------------------------------------------------------
# Dense-prefix fast path (opt-in: GLM_NSA_DENSE_PREFIX=1). DSA semantics: token
# t with t+1 <= topk selects its ENTIRE prefix -> exact dense causal attention,
# FA2-tiled (M rows 100% real, zero gather). Guarded by an exact set check
# (count+min+max pin the selected set to base+{0..t} under unique indices) and
# rebased for pool-row offsets. Evidence: docs/dense-prefix-report.md (1.84x on
# the prefix region; +12%/+5% total at 4k/8k; whole-request 1.84x for T<=2048).
# ---------------------------------------------------------------------------


@triton.jit
def _dense_prefix_kernel(
    q_ptr,
    kv_ptr,
    o_ptr,
    sm_scale,
    P,
    H: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    D_TAIL: tl.constexpr = D_QK - D_V

    m0 = pid_m * BLOCK_M
    m = tl.arange(0, BLOCK_M)
    rows = m0 + m
    rmask = rows < P
    dv = tl.arange(0, D_V)
    dt = tl.arange(0, D_TAIL)

    qb = q_ptr + (rows * H + h).to(tl.int64)[:, None] * D_QK
    q_main = tl.load(qb + dv[None, :], mask=rmask[:, None], other=0.0)
    q_tail = tl.load(qb + (D_V + dt)[None, :], mask=rmask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D_V], tl.float32)
    n = tl.arange(0, BLOCK_N)

    # phase-1 may only cover WHOLE blocks strictly below m0; the remainder
    # [safe_lo, hi) goes through the causal-masked phase-2 (any BLOCK_M/BLOCK_N).
    safe_lo = (m0 // BLOCK_N) * BLOCK_N
    for k0 in tl.range(0, safe_lo, BLOCK_N):  # fully-unmasked blocks
        kb = kv_ptr + (k0 + n).to(tl.int64)[:, None] * D_QK
        kv_main = tl.load(kb + dv[None, :])
        kv_tail = tl.load(kb + (D_V + dt)[None, :])
        qk = tl.dot(q_main, tl.trans(kv_main))
        qk = tl.dot(q_tail, tl.trans(kv_tail), qk) * sm_scale
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv_main.dtype), kv_main)
        m_i = m_new

    hi = tl.minimum(m0 + BLOCK_M, P)
    for k0 in tl.range(safe_lo, hi, BLOCK_N):  # causal boundary blocks
        cmask = (k0 + n) < P
        kb = kv_ptr + tl.where(cmask, k0 + n, 0).to(tl.int64)[:, None] * D_QK
        kv_main = tl.load(kb + dv[None, :], mask=cmask[:, None], other=0.0)
        kv_tail = tl.load(kb + (D_V + dt)[None, :], mask=cmask[:, None], other=0.0)
        qk = tl.dot(q_main, tl.trans(kv_main))
        qk = tl.dot(q_tail, tl.trans(kv_tail), qk) * sm_scale
        causal = (k0 + n)[None, :] <= rows[:, None]
        qk = tl.where(causal & cmask[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv_main.dtype), kv_main)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    ob = o_ptr + (rows * H + h).to(tl.int64)[:, None] * D_V
    tl.store(ob + dv[None, :], acc.to(o_ptr.dtype.element_ty), mask=rmask[:, None])


def _dense_prefix_path(q, kv, indices, sm_scale, d_v, out, dense_config=None):
    """Handle rows [0, P) densely when they provably selected their full prefix.
    Returns P (rows handled; 0 = not applicable). Exact-set gate: with unique
    indices, count==t+1 & min==base & max==base+t pin the set to base+{0..t}."""
    T, h, d_qk = q.shape
    topk = indices.shape[-1]
    P = min(topk, T)
    if P < 128:
        return 0
    pre = indices[:P]
    valid = pre >= 0
    counts = valid.sum(dim=-1, dtype=torch.int32)
    want = torch.arange(P, dtype=torch.int32, device=indices.device)
    vmax = pre.amax(dim=-1)
    vmin = torch.where(valid, pre, pre.new_full((), 2**31 - 1)).amin(dim=-1)
    base = vmin[0]
    ok = (counts == want + 1).all() & (vmin == base).all() & (vmax == base + want).all()
    if not bool(ok):
        return 0
    b = int(base)
    if b + P > kv.shape[0]:
        return 0
    if dense_config is not None:
        bm, bn, warps, stages = dense_config
    else:
        _cap = torch.cuda.get_device_capability()
        # SM90: 228KB smem; SM120: 99KB -> smaller tiles (validated on-box 2026-07-22)
        bm, bn, warps, stages = (32, 64, 4, 2) if _cap[0] == 9 else (16, 32, 8, 2)
    _dense_prefix_kernel[(triton.cdiv(P, bm), h)](
        q[:P],
        kv[b:],
        out[:P],
        sm_scale,
        P,
        H=h,
        D_QK=d_qk,
        D_V=d_v,
        BLOCK_M=bm,
        BLOCK_N=bn,
        num_warps=warps,
        num_stages=stages,
    )
    return P


# ---------------------------------------------------------------------------
# v3 union tiling (opt-in: GLM_NSA_UNION=2|4). G adjacent tokens share one
# deduplicated KV gather; M = G*H rows all-real. Exact math (per-token -inf
# masking); evidence + gates in docs/decision-ledger.md v3 entries.
# ---------------------------------------------------------------------------

# Scratch for the union path, keyed by (group size, KV-span bucket, device) —
# deliberately NOT by the group count, which changes with every batch shape and
# would let the cache grow without bound over a server's lifetime. Buffers grow
# in place when a larger batch arrives, so the entry count stays bounded by the
# span buckets a model can reach.
_UNION_WS = {}
_UNION_SPAN_BUDGET = 512 << 20  # bytes for the [NG, span] int32 mark map


@triton.jit
def _union_mark_kernel(
    idx_ptr, map_ptr, K, S, base, epoch, G: tl.constexpr, BLOCK: tl.constexpr
):
    # Bytemap lanes are G-wide (not fixed 4) and carry an epoch value instead of
    # a sticky 1: stale bytes from earlier layers never match the current epoch,
    # so the compact pass needs no self-zeroing store (halves its traffic).
    pid = tl.program_id(0)
    g = pid // G
    tok = pid % G
    b = idx_ptr + pid.to(tl.int64) * K
    n = tl.arange(0, BLOCK)
    ep8 = tl.full([BLOCK], 0, tl.int8) + epoch
    for k0 in tl.range(0, K, BLOCK):
        v = tl.load(b + k0 + n, mask=(k0 + n) < K, other=-1)
        v = v - base
        valid = (v >= 0) & (v < S)
        addr = (g.to(tl.int64) * S + tl.where(valid, v, 0).to(tl.int64)) * G + tok
        tl.store(map_ptr + addr, ep8, mask=valid)


@triton.jit
def _union_compact_kernel(
    map_ptr,
    uidx_ptr,
    ubits_ptr,
    ulen_ptr,
    S,
    U_CAP,
    epoch,
    BLOCK: tl.constexpr,
    LANES: tl.constexpr,
    STAGES: tl.constexpr,
):
    g = tl.program_id(0).to(tl.int64)
    n = tl.arange(0, BLOCK)
    cursor = tl.zeros([], tl.int32)
    for s0 in tl.range(0, S, BLOCK, num_stages=STAGES):
        inb = (s0 + n) < S
        w = tl.load(map_ptr + g * S + s0 + n, mask=inb, other=0).to(tl.int32)
        bits = ((w & 255) == epoch).to(tl.int32)
        bits |= (((w >> 8) & 255) == epoch).to(tl.int32) * 2
        if LANES == 4:
            bits |= (((w >> 16) & 255) == epoch).to(tl.int32) * 4
            bits |= (((w >> 24) & 255) == epoch).to(tl.int32) * 8
        present = (bits != 0) & inb
        wpos = cursor + tl.cumsum(present.to(tl.int32), axis=0) - present.to(tl.int32)
        tl.store(uidx_ptr + g * U_CAP + wpos, (s0 + n).to(tl.int32), mask=present)
        tl.store(ubits_ptr + g * U_CAP + wpos, bits, mask=present)
        cursor += tl.sum(present.to(tl.int32))
    tl.store(ulen_ptr + g, cursor)


@triton.jit
def _nsa_prefill_union_kernel(
    q_ptr,
    kv_ptr,
    uidx_ptr,
    ubits_ptr,
    ulen_ptr,
    o_ptr,
    sm_scale,
    U_CAP,
    base,
    H: tl.constexpr,
    G: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    g = tl.program_id(0)
    D_TAIL: tl.constexpr = D_QK - D_V
    GH: tl.constexpr = G * H

    r = tl.arange(0, GH)
    tok_of_row = r // H
    dv = tl.arange(0, D_V)
    dt = tl.arange(0, D_TAIL)

    qb = q_ptr + g.to(tl.int64) * GH * D_QK
    q_main = tl.load(qb + r[:, None] * D_QK + dv[None, :])
    q_tail = tl.load(qb + r[:, None] * D_QK + (D_V + dt)[None, :])

    m_i = tl.full([GH], -float("inf"), tl.float32)
    l_i = tl.zeros([GH], tl.float32)
    acc = tl.zeros([GH, D_V], tl.float32)

    n = tl.arange(0, BLOCK_N)
    u_len = tl.load(ulen_ptr + g)
    ub = g.to(tl.int64) * U_CAP
    for k0 in tl.range(0, u_len, BLOCK_N):
        inb = (k0 + n) < u_len
        uidx = tl.load(uidx_ptr + ub + k0 + n, mask=inb, other=-1)
        bits = tl.load(ubits_ptr + ub + k0 + n, mask=inb, other=0)
        valid = uidx >= 0
        row = (tl.where(valid, uidx, 0) + base).to(tl.int64)
        kb = kv_ptr + row * D_QK
        kv_main = tl.load(kb[:, None] + dv[None, :], mask=valid[:, None], other=0.0)
        kv_tail = tl.load(
            kb[:, None] + (D_V + dt)[None, :], mask=valid[:, None], other=0.0
        )

        qk = tl.dot(q_main, tl.trans(kv_main))
        qk = tl.dot(q_tail, tl.trans(kv_tail), qk) * sm_scale
        sel = ((bits[None, :] >> tok_of_row[:, None]) & 1) != 0
        qk = tl.where(sel & valid[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv_main.dtype), kv_main)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    tl.store(
        o_ptr + g.to(tl.int64) * GH * D_V + r[:, None] * D_V + dv[None, :],
        acc.to(o_ptr.dtype.element_ty),
    )


def _union_path(q, kv, indices, sm_scale, d_v, out, G, union_config=None):
    """Returns True if handled. Budget-gated; tail rows (T % G) fall back."""
    T, h, d_qk = q.shape
    K = indices.shape[-1]
    while G > 1 and G * h > 32:  # keep the M tile within smem (configs tuned to GH<=32)
        G //= 2
    if G < 2:
        return False
    T_main = (T // G) * G
    if T_main == 0:
        return False
    # single fused reduction + ONE host sync (amax is -1-safe; amin masks -1 to INT_MAX)
    vmin_t = torch.where(
        indices >= 0,
        indices,
        torch.tensor(2**31 - 1, dtype=indices.dtype, device=indices.device),
    ).amin()
    vmax_t = indices.amax()
    vmin, vmax = torch.stack([vmin_t, vmax_t]).tolist()
    if vmax < 0:
        return False
    span = vmax - vmin + 1
    NG = T_main // G
    if NG * span * G > _UNION_SPAN_BUDGET:
        return False

    span_alloc = ((span + 4095) // 4096) * 4096
    key = (G, span_alloc, G * K, q.device)
    bufs = _UNION_WS.get(key)
    if bufs is None or bufs[0].shape[0] < NG:
        # Grow to this batch's group count; the old buffers (if any) are dropped
        # and their epoch counter restarts, which the zero-init below re-bases.
        bufs = (
            torch.zeros(NG, span_alloc * G, dtype=torch.int8, device=q.device),
            torch.empty(NG, G * K, dtype=torch.int32, device=q.device),
            torch.empty(NG, G * K, dtype=torch.int32, device=q.device),
            torch.empty(NG, dtype=torch.int32, device=q.device),
            [0],
        )
        _UNION_WS[key] = bufs
    ws_all, uidx, ubits, ulen, ep_box = bufs
    epoch = ep_box[0] + 1
    if epoch > 127:  # int8 epoch wrap: one cheap memset per 127 reuses
        # Zero every row, not just this batch's: a later, larger batch would
        # otherwise read marks left by an earlier cycle as belonging to it.
        ws_all.zero_()
        epoch = 1
    ep_box[0] = epoch
    ws, uidx, ubits, ulen = ws_all[:NG], uidx[:NG], ubits[:NG], ulen[:NG]
    U_CAP = G * K
    idx_main = indices[:T_main].contiguous()
    _union_mark_kernel[(NG * G,)](
        idx_main, ws, K, span_alloc, vmin, epoch, G=G, BLOCK=1024, num_warps=4
    )
    wsw = ws.view(torch.int16) if G == 2 else ws.view(torch.int32)
    _union_compact_kernel[(NG,)](
        wsw,
        uidx,
        ubits,
        ulen,
        span_alloc,
        U_CAP,
        epoch,
        BLOCK=1024,
        LANES=G,
        STAGES=4,
        num_warps=4,
    )
    if union_config is not None:
        bn, warps, stages = union_config
    elif torch.cuda.get_device_capability(q.device)[0] >= 12:
        # SM120 on-box sweeps: G=2 winner (64,4,3) 5.21 vs 5.50; G=4 winner (32,4,2)
        # 3.605 ms on real indices (BN=64 OORs >=115KB with the GH=32 Q tile; BN=32
        # restores the fit and the M=32 x N=32 tile beats every neighbor by >=12%).
        bn, warps, stages = (64, 4, 3) if G == 2 else (32, 4, 2)
    else:
        bn, warps, stages = (64, 4, 2) if G == 4 else (64, 8, 2)
    # The union Q tile is H*G rows, so its shared-memory footprint grows with
    # the head count: 16 heads at G=2 already exceeds SM120's 100 KB with the
    # tuned tile. Step down as the per-token launcher does rather than failing
    # the request.
    for bn_try, ns_try in _smem_fallbacks(bn, stages):
        try:
            _nsa_prefill_union_kernel[(NG,)](
                q[:T_main],
                kv,
                uidx,
                ubits,
                ulen,
                out[:T_main],
                sm_scale,
                U_CAP,
                vmin,
                H=h,
                G=G,
                D_QK=d_qk,
                D_V=d_v,
                BLOCK_N=bn_try,
                num_warps=warps,
                num_stages=ns_try,
            )
            break
        except triton.runtime.errors.OutOfResources:
            continue
    else:
        return False  # no tile fits; caller falls through to the per-token path
    if T_main < T:  # tail rows through the per-token kernel
        sparse_mla_prefill(
            q[T_main:],
            kv,
            indices[T_main:],
            sm_scale,
            d_v,
            out=out[T_main:],
            union=False,
        )
    return True


def _topk_length(indices, topk):
    valid = indices >= 0
    any_valid = valid.any(dim=-1)
    last = topk - torch.flip(valid, [-1]).int().argmax(dim=-1)
    return torch.where(any_valid, last, torch.zeros_like(last)).to(torch.int32)


_UNTUNED_ARCH_WARNED = set()


def _config(device):
    """Per-arch tuned (BLOCK_N, num_warps, num_stages); see _PINNED."""
    cap = torch.cuda.get_device_capability(device)
    if cap in _PINNED:
        return _PINNED[cap]
    if cap not in _UNTUNED_ARCH_WARNED:
        _UNTUNED_ARCH_WARNED.add(cap)
        # The kernel is correct on any SM90+ device, but the tile was only swept
        # on the architectures in _PINNED. Say so rather than quietly running a
        # config nobody has measured — add an entry there once swept.
        logger.warning(
            "triton_sparse_mla: no tuned tile for sm_%d%d; falling back to %s. "
            "Sweep and add it to _PINNED for best throughput.",
            cap[0],
            cap[1],
            _UNTUNED_DEFAULT,
        )
    return _UNTUNED_DEFAULT


def _smem_fallbacks(bn, stages):
    """Ordered (BLOCK_N, num_stages) candidates: the tuned config first, then
    progressively smaller smem footprints. Lets one pinned config serve head
    counts / devices whose smem budget the tuned tile would exceed."""
    seen, out = set(), []
    for cand in (
        (bn, stages),
        (bn, 2),
        (bn // 2, stages),
        (bn // 2, 2),
        (bn // 4, 2),
        (16, 2),
    ):
        b, ns = max(16, cand[0]), max(1, cand[1])
        if (b, ns) not in seen:
            seen.add((b, ns))
            out.append((b, ns))
    return out


def sparse_mla_prefill(
    q,
    kv,
    indices,
    sm_scale,
    d_v=512,
    *,
    topk_length=None,
    out=None,
    union=0,
    dense=False,
    config=None,
    union_config=None,
    dense_config=None,
    int64_indexing=None,
):
    """Fused sparse-MLA prefill. Returns ``out`` ``[T, H, d_v]`` bf16.

    Args:
        q: ``[T, H, d_qk]`` bf16 query (absorbed MLA; ``d_qk = d_v + rope``).
        kv: ``[S, d_qk]`` or ``[S, 1, d_qk]`` bf16 latent cache; ``V`` is
            ``kv[:, :d_v]`` (no separate value gather).
        indices: ``[T, topk]`` or ``[T, 1, topk]`` int32 selected slots; ``-1``
            or ``>= S`` marks an invalid slot and is skipped.
        sm_scale: softmax scale.
        d_v: value head dim (512 for DSA).
        topk_length: optional ``[T]`` int32 per-row valid count. Computed from
            ``indices`` when omitted; pass it to skip that reduction.
        out: optional preallocated ``[T, H, d_v]`` bf16 output.
        union: 0 (off), 2 or 4 — share one gathered union index set across ``G``
            adjacent query tokens. Exact, not an approximation: an ownership
            bitmask restores each token's own softmax support.
        dense: enable the dense-prefix identity fast path for tokens whose top-k
            covers the whole causal prefix. Guarded by an exact set check; falls
            back to the sparse path if the guard fails.
        config / union_config / dense_config: optional tile overrides
            ``(BLOCK_N, num_warps, num_stages)`` (dense takes ``(BM, BN, warps,
            stages)``). Defaults are the per-arch tuned entries in ``_PINNED``.
    """
    if kv.dim() == 3:  # [S, 1, D] -> [S, D]
        assert kv.shape[1] == 1
        kv = kv.squeeze(1)
    if indices.dim() == 3:  # [T, 1, K] -> [T, K]
        assert indices.shape[1] == 1
        indices = indices.squeeze(1)
    T, h, d_qk = q.shape
    topk = indices.shape[-1]
    q, kv, indices = q.contiguous(), kv.contiguous(), indices.contiguous()
    if out is None:
        out = torch.empty(T, h, d_v, dtype=torch.bfloat16, device=q.device)

    if dense:
        P = _dense_prefix_path(q, kv, indices, sm_scale, d_v, out, dense_config)
        if P >= T:
            return out
        if P > 0:  # remainder continues through the normal (or union) path below
            sparse_mla_prefill(
                q[P:],
                kv,
                indices[P:],
                sm_scale,
                d_v,
                out=out[P:],
                union=union,
                dense=False,
            )
            return out

    if union in (2, 4) and _union_path(
        q, kv, indices, sm_scale, d_v, out, union, union_config
    ):
        return out

    if topk_length is None:
        topk_length = _topk_length(indices, topk)

    q_in, kv_in = q, kv
    scales = torch.ones(2, dtype=torch.float32, device=q.device)

    bn, warps, stages = config or _config(q.device)
    # int32 gather addressing unless the pool could overflow int32 element offsets
    # (row*d_qk + d_qk-1 must fit in int32) — production pools can exceed this.
    # The threshold is ~3.7M rows at d_qk=576, which no test can allocate, so the
    # mode is overridable to keep the int64 path reachable from a test.
    if int64_indexing is None:
        idx64 = kv.shape[0] > (2**31 - 1 - (d_qk - 1)) // d_qk
    else:
        idx64 = bool(int64_indexing)
    block_h = max(16, triton.next_power_of_2(h))
    for bn_try, ns_try in _smem_fallbacks(bn, stages):
        try:
            _nsa_prefill_kernel[(T,)](
                q_in,
                kv_in,
                indices,
                topk_length,
                out,
                scales,
                sm_scale,
                topk,
                H=h,
                BLOCK_H=block_h,
                D_QK=d_qk,
                D_V=d_v,
                BLOCK_N=bn_try,
                num_warps=warps,
                num_stages=ns_try,
                FP8=False,
                MATH_BF16=False,
                IDX64=idx64,
            )
            return out
        except triton.runtime.errors.OutOfResources:
            # A larger head count (BLOCK_H) or a smaller smem budget can push the
            # pinned tile over the device limit (e.g. h=32 on SM120's 100 KB).
            # Step down the K tile / pipeline depth instead of failing the request.
            continue
    raise triton.runtime.errors.OutOfResources(
        0, 0, "shared memory: no fallback config fits this device/shape"
    )
