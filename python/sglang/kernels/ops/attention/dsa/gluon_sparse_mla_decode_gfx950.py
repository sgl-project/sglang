"""Gluon split-K sparse MLA decode for gfx950 (MI355X), fp8 KV cache.

Replaces the Triton split-K producer + reduce of ``triton_sparse_mla`` for the
DSA decode / target-verify shapes of GLM-5-class models: 8 or 16 local heads,
kv_lora_rank 512 + rope 64 (576-byte fp8 rows), top-k 2048.

  q_all:   [M, H, 576] fp8 e4m3 (q_nope | q_rope, one buffer)
  kv:      [num_slots, 1, 576] fp8 e4m3
  indices: [M, 2048] int32 KV slots, -1 = empty
  output:  [1, M, H, 512] bf16

Every M takes one producer launch writing head-major partials
[M, H, splits, 512] and one reduce launch. Most M use the generic producer;
the rows in ``_TUNED_ROUTES`` are the verify shapes (M = 6 * batch) with a
producer specialised for that exact shape.
"""

import functools
import math

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna4
from triton.experimental.gluon.language.amd.cdna4 import async_copy

try:
    from triton.experimental.gluon.language import barrier as _cta_barrier
except ImportError:  # Triton 3.6 names it thread_barrier.
    from triton.experimental.gluon.language import thread_barrier as _cta_barrier

from sglang.srt.utils import get_device_core_count


@gluon.jit
def _gen_offsets(
    irow,
    kt,
    NT: gl.constexpr,
    BK: gl.constexpr,
    KV_DIM: gl.constexpr,
    topk: gl.constexpr,
    blk: gl.constexpr,
    blt: gl.constexpr,
    ms: gl.constexpr,
):
    ktc = gl.minimum(kt, NT - 1)
    base = irow + ktc * BK
    nK = gl.arange(0, BK, layout=gl.SliceLayout(1, blk))
    nT = gl.arange(0, BK, layout=gl.SliceLayout(1, blt))
    nS = gl.arange(0, BK, layout=gl.SliceLayout(0, ms))
    rk = gl.load(base + nK)
    rt = gl.load(base + nT)
    rs = gl.load(base + nS)
    o_k = gl.maximum(rk, 0).to(gl.uint32) * KV_DIM
    o_t = gl.maximum(rt, 0).to(gl.uint32) * KV_DIM
    v = (rs >= 0) & (kt < NT)
    return (o_k, o_t, v)


@gluon.jit
def _gen_load(
    kv_ptr,
    o_k,
    o_t,
    D_V: gl.constexpr,
    D_TAIL: gl.constexpr,
    blk: gl.constexpr,
    blt: gl.constexpr,
):
    dk = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk))
    dtt = gl.arange(0, D_TAIL, layout=gl.SliceLayout(0, blt))
    kv = gl.load(kv_ptr + o_k[:, None] + dk[None, :], cache_modifier=".cg")
    kt_ = gl.load(kv_ptr + o_t[:, None] + (D_V + dtt)[None, :], cache_modifier=".cg")
    return (kv, kt_)


@gluon.jit
def _gen_step(
    q,
    qt,
    skv,
    skt,
    vv,
    m_i,
    l_i,
    acc,
    qk_scale,
    lg_fp8,
    BH: gl.constexpr,
    BK: gl.constexpr,
    mt: gl.constexpr,
    ms: gl.constexpr,
    mp: gl.constexpr,
    bt: gl.constexpr,
    ap: gl.constexpr,
    bp: gl.constexpr,
):
    NEG: gl.constexpr = -3.4028234663852886e38
    sc = gl.zeros([BH, BK], gl.float32, layout=mt)
    sc = cdna4.mfma(q, skv.permute((1, 0)).load(bt), sc)
    sc = cdna4.mfma(qt, skt.permute((1, 0)).load(bt), sc)
    sc = gl.convert_layout(sc, ms)
    sc = gl.where(vv[None, :], sc * qk_scale, NEG)
    m_new = gl.maximum(m_i, gl.max(sc, axis=1))
    p = gl.exp2(sc - (m_new - lg_fp8)[:, None])
    pd = gl.convert_layout(p.to(gl.float8e4nv), ap)
    alpha = gl.exp2(m_i - m_new)
    l_i = l_i * alpha + gl.sum(p, axis=1)
    alpha_p = gl.convert_layout(alpha, gl.SliceLayout(1, mp))
    acc = cdna4.mfma(pd, skv.load(bp), acc * alpha_p[:, None])
    return (m_new, l_i, acc)


# Generic producer: one CTA per (row, split) walks tps BK-key tiles; partials are
# head-major [M, H, ks, D_V] (bf16) plus lse [M, H, ks] (log2 domain).
@gluon.jit(do_not_specialize=["ks", "tps"])
def _sparse_mla_split_k_gen_kernel(
    q_ptr,
    qr_ptr,
    kv_ptr,
    idx_ptr,
    lse_ptr,
    acc_ptr,
    qk_scale,
    lg_fp8,
    ks,
    tps,
    topk: gl.constexpr,
    H: gl.constexpr,
    KV_DIM: gl.constexpr,
    D_V: gl.constexpr,
    D_TAIL: gl.constexpr,
    BH: gl.constexpr,
    BK: gl.constexpr,
    NT: gl.constexpr,
    STRIDE_QN_T: gl.constexpr,
    STRIDE_QN_H: gl.constexpr,
    STRIDE_QR_T: gl.constexpr,
    STRIDE_QR_H: gl.constexpr,
    PST_CM: gl.constexpr,
):
    t = gl.program_id(0)
    pk = gl.program_id(2)
    mt: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [1, 4])
    ms: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [4, 1])
    mp: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [1, 4])
    at: gl.constexpr = gl.DotOperandLayout(0, mt, 8)
    bt: gl.constexpr = gl.DotOperandLayout(1, mt, 8)
    ap: gl.constexpr = gl.DotOperandLayout(0, mp, 8)
    bp: gl.constexpr = gl.DotOperandLayout(1, mp, 8)
    blk: gl.constexpr = gl.BlockedLayout([1, 16], [2, 32], [4, 1], [1, 0])
    blt: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [1, 0])
    shk: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 16, [1, 0])
    sht: gl.constexpr = gl.SwizzledSharedLayout(8, 4, 4, [1, 0])
    NEG: gl.constexpr = -3.4028234663852886e38
    irow = idx_ptr + t * topk
    kt0 = pk * tps
    (o_k0, o_t0, v0) = _gen_offsets(irow, kt0, NT, BK, KV_DIM, topk, blk, blt, ms)
    (kvA, ktA) = _gen_load(kv_ptr, o_k0, o_t0, D_V, D_TAIL, blk, blt)
    vA = v0
    dk = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk))
    hr = gl.minimum(gl.arange(0, BH, layout=gl.SliceLayout(1, at)), H - 1)
    dtq = gl.arange(0, D_TAIL, layout=gl.SliceLayout(0, at))
    qt = gl.load(qr_ptr + t * STRIDE_QR_T + hr[:, None] * STRIDE_QR_H + dtq[None, :])
    hb = gl.minimum(gl.arange(0, BH, layout=gl.SliceLayout(1, blk)), H - 1)
    qb = gl.load(q_ptr + t * STRIDE_QN_T + hb[:, None] * STRIDE_QN_H + dk[None, :])
    q = gl.convert_layout(qb, at)
    skv = gl.allocate_shared_memory(kvA.dtype, [BK, D_V], shk)
    skt = gl.allocate_shared_memory(kvA.dtype, [BK, D_TAIL], sht)
    m_i = gl.full([BH], -1e30, gl.float32, layout=gl.SliceLayout(1, ms))
    l_i = gl.zeros([BH], gl.float32, layout=gl.SliceLayout(1, ms))
    acc = gl.zeros([BH, D_V], gl.float32, layout=mp)
    for i in range(0, tps - 1):
        vv = vA
        skv.store(kvA)
        skt.store(ktA)
        (o_kn, o_tn, vA) = _gen_offsets(
            irow, kt0 + i + 1, NT, BK, KV_DIM, topk, blk, blt, ms
        )
        (kvA, ktA) = _gen_load(kv_ptr, o_kn, o_tn, D_V, D_TAIL, blk, blt)
        (m_i, l_i, acc) = _gen_step(
            q,
            qt,
            skv,
            skt,
            vv,
            m_i,
            l_i,
            acc,
            qk_scale,
            lg_fp8,
            BH,
            BK,
            mt,
            ms,
            mp,
            bt,
            ap,
            bp,
        )
    skv.store(kvA)
    skt.store(ktA)
    (m_i, l_i, acc) = _gen_step(
        q,
        qt,
        skv,
        skt,
        vA,
        m_i,
        l_i,
        acc,
        qk_scale,
        lg_fp8,
        BH,
        BK,
        mt,
        ms,
        mp,
        bt,
        ap,
        bp,
    )
    den = gl.maximum(l_i, 1e-30)
    hd = l_i > 0.0
    inv = gl.convert_layout(gl.where(hd, 1.0 / den, 0.0), gl.SliceLayout(1, mp))
    acc = acc * inv[:, None]
    lse = gl.where(hd, gl.log2(l_i) - lg_fp8 + m_i, NEG)
    ho = gl.arange(0, BH, layout=gl.SliceLayout(1, ms))
    lse_off = (t * (H * ks) + ho * ks + pk).to(gl.int32)
    cdna4.buffer_store(lse, lse_ptr, lse_off, mask=ho < H)
    bso: gl.constexpr = gl.BlockedLayout([1, 8], [2, 32], [4, 1], [1, 0])
    accb = gl.convert_layout(acc.to(gl.bfloat16), bso)
    hs = gl.arange(0, BH, layout=gl.SliceLayout(1, bso))
    ds = gl.arange(0, D_V, layout=gl.SliceLayout(0, bso))
    so = (t * (H * ks * D_V) + hs[:, None] * (ks * D_V) + pk * D_V + ds[None, :]).to(
        gl.int32
    )
    cdna4.buffer_store(accb, acc_ptr, so, mask=(hs < H)[:, None], cache=PST_CM)


# Merges the ks splits of the generic producer; SP is the padded split extent.
@gluon.jit(do_not_specialize=["ks"])
def _sparse_mla_reduce_gen_kernel(
    lse_ptr,
    acc_ptr,
    out_ptr,
    ks,
    H: gl.constexpr,
    D_V: gl.constexpr,
    SP: gl.constexpr,
    TK: gl.constexpr,
    VEC: gl.constexpr,
    NW: gl.constexpr,
):
    DC: gl.constexpr = VEC * (64 // TK) * NW
    t = gl.program_id(0)
    h = gl.program_id(1)
    dc = gl.program_id(2)
    la: gl.constexpr = gl.BlockedLayout(
        [SP // TK, VEC], [TK, 64 // TK], [1, NW], [1, 0]
    )
    base = t * (H * ks) + h * ks
    kr = gl.arange(0, SP, layout=gl.SliceLayout(1, la))
    dd = dc * DC + gl.arange(0, DC, layout=gl.SliceLayout(0, la))
    a_off = (base * D_V + kr[:, None] * D_V + dd[None, :]).to(gl.int32)
    l_off = (base + kr).to(gl.int32)
    km = kr < ks
    a_p = cdna4.buffer_load(acc_ptr, a_off, mask=km[:, None], other=0.0, cache=".cg")
    lse_r = cdna4.buffer_load(lse_ptr, l_off, mask=km, other=-3.4e38)
    lse_max = gl.max(lse_r, axis=0)
    w_r = gl.exp2(lse_r - lse_max)
    num = gl.sum(a_p.to(gl.float32) * w_r[:, None], axis=0)
    den = gl.sum(w_r, axis=0)
    o = num * (1.0 / gl.maximum(den, 1e-30))
    o_off = (t * (H * D_V) + h * D_V + dd).to(gl.int32)
    cdna4.buffer_store(o.to(gl.bfloat16), out_ptr, o_off)


# Exact-shape producer for M in {6, 12, 18}: one 2*BK key tile per CTA,
# QK as a scaled fp8 MFMA, async (LDS-DMA) KV gather; DUP=1 serves 8 heads by
# duplicating them into the 16-row MFMA tile.
@gluon.jit
def _sparse_mla_split_k_ov_kernel(
    q_ptr,
    qr_ptr,
    kv_ptr,
    idx_ptr,
    lse_ptr,
    acc_ptr,
    qk_scale,
    lg_fp8,
    topk: gl.constexpr,
    H: gl.constexpr,
    KV_DIM: gl.constexpr,
    D_V: gl.constexpr,
    D_TAIL: gl.constexpr,
    BH: gl.constexpr,
    BK: gl.constexpr,
    KV_SPLITS: gl.constexpr,
    STRIDE_QN_T: gl.constexpr,
    STRIDE_QN_H: gl.constexpr,
    STRIDE_QR_T: gl.constexpr,
    STRIDE_QR_H: gl.constexpr,
    XCD_REMAP: gl.constexpr,
    XCD_N: gl.constexpr,
    XCD_PER: gl.constexpr,
    EPI: gl.constexpr,
    DUP: gl.constexpr,
    OVX: gl.constexpr = 0,
    OVK: gl.constexpr = 0,
    PRO: gl.constexpr = 0,
    PST_CM: gl.constexpr = "",
):
    if XCD_REMAP:
        _lin = gl.program_id(0)
        _j = _lin % XCD_N * XCD_PER + _lin // XCD_N
        t = _j // KV_SPLITS
        pk = _j % KV_SPLITS
    else:
        t = gl.program_id(0)
        pk = gl.program_id(2)
    BK2: gl.constexpr = 2 * BK
    if DUP:
        DC: gl.constexpr = D_V // 4
    else:
        DC: gl.constexpr = D_V // 2
    DH: gl.constexpr = D_V // 2
    mq: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 128], True, [1, 4])
    mt: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [1, 4])
    ms: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [4, 1])
    aq: gl.constexpr = gl.DotOperandLayout(0, mq, 16)
    bq: gl.constexpr = gl.DotOperandLayout(1, mq, 16)
    at: gl.constexpr = gl.DotOperandLayout(0, mt, 8)
    bt: gl.constexpr = gl.DotOperandLayout(1, mt, 8)
    blk: gl.constexpr = gl.BlockedLayout([1, 16], [2, 32], [4, 1], [1, 0])
    shk: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 16, [1, 0])
    if EPI == 1:
        mp: gl.constexpr = gl.amd.AMDMFMALayout(
            4, [16, 16, 32], True, [1, 4], tiles_per_warp=[1, DC // 64]
        )
        if DC == 256:
            bst: gl.constexpr = gl.DistributedLinearLayout(
                [[0, 1], [0, 2], [0, 4], [0, 32]],
                [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8], [0, 16]],
                [[0, 64], [0, 128]],
                [],
                [BH, DC],
            )
        else:
            bst: gl.constexpr = gl.DistributedLinearLayout(
                [[0, 1], [0, 2], [0, 4]],
                [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8], [0, 16]],
                [[0, 32], [0, 64]],
                [],
                [BH, DC],
            )
    else:
        mp: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [1, 4])
        bst: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    ap: gl.constexpr = gl.DotOperandLayout(0, mp, 8)
    bp: gl.constexpr = gl.DotOperandLayout(1, mp, 8)
    KH: gl.constexpr = BK2 // 2
    l3: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 0, 1], [0, 0, 2], [0, 0, 4]],
        [[0, 1, 0], [0, 2, 0], [0, 4, 0], [1, 0, 0], [0, 0, 8], [0, 0, 16]],
        [[0, 0, 0], [0, 0, 0]],
        [],
        [2, 8, 32],
    )
    lz: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 0, 1], [0, 0, 2], [0, 0, 4], [1, 0, 0]],
        [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 0, 0], [0, 0, 8], [0, 0, 16]],
        [[0, 0, 0], [0, 0, 0]],
        [],
        [2, 8, 32],
    )
    l2: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4]],
        [[1, 0], [2, 0], [4, 0], [0, 32], [0, 8], [0, 16]],
        [[0, 0], [0, 0]],
        [],
        [8, 64],
    )
    l3f: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 32]],
        [[0, 1, 0], [0, 2, 0], [0, 4, 0], [1, 0, 0], [0, 0, 8], [0, 0, 16]],
        [[0, 0, 0], [0, 0, 0]],
        [],
        [2, 8, 64],
    )
    OV8: gl.constexpr = (
        OVX & 4 != 0 and DUP != 0 and (H == 8) and (BK2 == 128 or BK2 == 64)
    )
    irow = idx_ptr + t * topk + pk * BK2
    dk = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk))
    bla: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [1, 0])
    BQ: gl.constexpr = BK // 2
    nKq = gl.arange(0, BQ, layout=gl.SliceLayout(1, blk))
    s_kq = ()
    for qi in gl.static_range(4):
        s_kq = s_kq + (gl.load(irow + qi * BQ + nKq),)
    nTa = gl.arange(0, BK2, layout=gl.SliceLayout(1, bla))
    s_ta = gl.load(irow + nTa)
    svv = gl.allocate_shared_memory(
        gl.float32, [BK2], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    _vi = gl.load(
        irow + gl.arange(0, BK2, layout=gl.BlockedLayout([1], [64], [4], [0]))
    )
    _vb = gl.where(_vi >= 0, 0.0, float("-inf"))
    svv.store(_vb)
    if DUP:
        hr = gl.arange(0, BH, layout=gl.SliceLayout(1, at)) % H
        hb = gl.arange(0, BH, layout=gl.SliceLayout(1, blk)) % H
    else:
        hr = gl.minimum(gl.arange(0, BH, layout=gl.SliceLayout(1, at)), H - 1)
        hb = gl.minimum(gl.arange(0, BH, layout=gl.SliceLayout(1, blk)), H - 1)
    dtq = gl.arange(0, D_TAIL, layout=gl.SliceLayout(0, at))
    qt = gl.load(qr_ptr + t * STRIDE_QR_T + hr[:, None] * STRIDE_QR_H + dtq[None, :])
    qb = gl.load(q_ptr + t * STRIDE_QN_T + hb[:, None] * STRIDE_QN_H + dk[None, :])
    shta: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 4, [1, 0])
    idl: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    skv = gl.allocate_shared_memory(gl.float8e4nv, [BK2, D_V], shk)
    skt = gl.allocate_shared_memory(gl.float8e4nv, [BK2, D_TAIL], shta)
    skv_w = skv._reinterpret(gl.float8e4nv, [BK2, D_V], idl)
    skt_w = skt._reinterpret(gl.float8e4nv, [BK2, D_TAIL], idl)
    kvr = ()
    for qi in gl.static_range(4, 4):
        o_q = gl.maximum(s_kq[qi], 0).to(gl.uint32) * KV_DIM
        kvr = kvr + (
            gl.load(kv_ptr + o_q[:, None] + dk[None, :], cache_modifier=".cg"),
        )
    if PRO & 2:
        dta = gl.arange(0, D_TAIL, layout=gl.SliceLayout(0, bla))
        o_ta = gl.maximum(s_ta, 0).to(gl.uint32) * KV_DIM
        offt = o_ta[:, None] + (D_V + (dta[None, :] ^ (nTa % 4 * 16)[:, None])).to(
            gl.uint32
        )
        offt = gl.max_contiguous(gl.multiple_of(offt, [1, 16]), [1, 16])
        async_copy.global_load_to_shared(skt_w, kv_ptr + offt, cache_modifier=".cg")
    swq = (dk[None, :] ^ (nKq % 16 * 16)[:, None]).to(gl.uint32)
    for qi in gl.static_range(4):
        o_q = gl.maximum(s_kq[qi], 0).to(gl.uint32) * KV_DIM
        offk = gl.max_contiguous(gl.multiple_of(o_q[:, None] + swq, [1, 16]), [1, 16])
        async_copy.global_load_to_shared(
            skv_w.slice(qi * BQ, BQ, 0), kv_ptr + offk, cache_modifier=".cg"
        )
    if PRO & 2 == 0:
        dta = gl.arange(0, D_TAIL, layout=gl.SliceLayout(0, bla))
        o_ta = gl.maximum(s_ta, 0).to(gl.uint32) * KV_DIM
        offt = o_ta[:, None] + (D_V + (dta[None, :] ^ (nTa % 4 * 16)[:, None])).to(
            gl.uint32
        )
        offt = gl.max_contiguous(gl.multiple_of(offt, [1, 16]), [1, 16])
        async_copy.global_load_to_shared(skt_w, kv_ptr + offt, cache_modifier=".cg")
    async_copy.commit_group()
    for qi in gl.static_range(4, 4):
        skv.slice(qi * BQ, BQ, 0).store(kvr[qi - 4])
    q = gl.convert_layout(qb, aq)
    async_copy.wait_group(0)
    if OV8:
        kb2 = svv._reinterpret(
            gl.float32, [2, KH], gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        ).load(gl.SliceLayout(1, l3))
    else:
        kbv = svv.load(gl.SliceLayout(0, ms))
    sq = gl.zeros([BH, BK2], gl.float32, layout=mq)
    sq = cdna4.mfma_scaled(
        q, None, "e4m3", skv.permute((1, 0)).load(bq), None, "e4m3", sq
    )
    sc = gl.convert_layout(sq, mt)
    sc = cdna4.mfma(qt, skt.permute((1, 0)).load(bt), sc)
    NEG: gl.constexpr = -3.4028234663852886e38
    if OV8:
        ss = gl.allocate_shared_memory(
            gl.float32, [BH, BK2], gl.SwizzledSharedLayout(4, 1, 8, [1, 0])
        )
        ss.store(sc)
        s3 = gl.convert_layout(
            gl.permute(gl.reshape(ss.slice(0, H, 0).load(l2), [H, 2, KH]), (1, 0, 2)),
            l3,
        )
        v0l = skv.slice(0, DC, 1).load(bp)
        v0h = skv.slice(DH, DC, 1).load(bp)
        v1l = skv.slice(DC, DC, 1).load(bp)
        v1h = skv.slice(DH + DC, DC, 1).load(bp)
        s3 = gl.fma(s3, qk_scale, gl.expand_dims(kb2, 1))
        m8 = gl.maximum(gl.max(gl.max(s3, axis=2), axis=0), -1e30)
        p3 = gl.exp2(s3 - gl.expand_dims(gl.expand_dims(m8 - lg_fp8, 0), 2))
        l8 = gl.sum(gl.sum(p3, axis=2), axis=0)
        den8 = gl.maximum(l8, 1e-30)
        hd8 = l8 > 0.0
        lg8 = gl.inline_asm_elementwise(
            "v_log_f32 $0, $1", "=v,v", [l8], dtype=gl.float32, is_pure=True, pack=1
        )
        lse8 = gl.where(hd8, lg8 - lg_fp8 + m8, NEG)
        ho8 = gl.arange(0, H, layout=gl.SliceLayout(0, gl.SliceLayout(2, l3)))
        cdna4.buffer_store(
            lse8,
            lse_ptr,
            (t * (H * KV_SPLITS) + ho8 * KV_SPLITS + pk).to(gl.int32),
            cache=PST_CM,
        )
        if OVK & 2:
            rinv8 = gl.inline_asm_elementwise(
                "v_rcp_f32 $0, $1",
                "=v,v",
                [den8],
                dtype=gl.float32,
                is_pure=True,
                pack=1,
            )
        else:
            rinv8 = 1.0 / den8
        inv2 = gl.broadcast(
            gl.expand_dims(gl.where(hd8, rinv8, 0.0), 0),
            gl.zeros([2, 8], gl.float32, layout=gl.SliceLayout(2, l3)),
        )[0]
        inv = gl.convert_layout(gl.reshape(inv2, [BH]), gl.SliceLayout(1, mp))
        pf3 = p3.to(gl.float8e4nv)
        pz = gl.convert_layout(pf3, lz)
        pr = gl.convert_layout(
            gl.reshape(gl.permute(pz, (1, 0, 2)), [H, BK2]), gl.SliceLayout(0, l3f)
        )
        pd = gl.convert_layout(
            gl.reshape(
                gl.broadcast(
                    gl.expand_dims(pr, 0),
                    gl.zeros([2, H, BK2], gl.float8e4nv, layout=l3f),
                )[0],
                [BH, BK2],
            ),
            ap,
        )
        (pz0, pz1) = gl.split(gl.permute(pz, (1, 2, 0)))
        pz0 = gl.convert_layout(pz0, gl.SliceLayout(0, l3))
        pz1 = gl.convert_layout(pz1, gl.SliceLayout(0, l3))
    else:
        sc = gl.convert_layout(sc, ms)
        sc = gl.fma(sc, qk_scale, kbv[None, :])
        m_i = gl.maximum(gl.max(sc, axis=1), -1e30)
        p = gl.exp2(sc - (m_i - lg_fp8)[:, None])
        l_i = gl.sum(p, axis=1)
        pd = gl.convert_layout(p.to(gl.float8e4nv), ap)
        den = gl.maximum(l_i, 1e-30)
        hd = l_i > 0.0
        lg2 = gl.inline_asm_elementwise(
            "v_log_f32 $0, $1", "=v,v", [l_i], dtype=gl.float32, is_pure=True, pack=1
        )
        lse = gl.where(hd, lg2 - lg_fp8 + m_i, NEG)
        ho = gl.arange(0, BH, layout=gl.SliceLayout(1, ms))
        lse_off = (t * (H * KV_SPLITS) + ho * KV_SPLITS + pk).to(gl.int32)
        cdna4.buffer_store(lse, lse_ptr, lse_off, mask=ho < H, cache=PST_CM)
        rinv = gl.inline_asm_elementwise(
            "v_rcp_f32 $0, $1", "=v,v", [den], dtype=gl.float32, is_pure=True, pack=1
        )
        inv = gl.convert_layout(gl.where(hd, rinv, 0.0), gl.SliceLayout(1, mp))
    # The epilogue must not start before every wave has finished the softmax stage:
    # without this barrier the partials race (run to run) on some Triton builds and configs.
    _cta_barrier()
    hs = gl.arange(0, BH, layout=gl.SliceLayout(1, bst))
    ds = gl.arange(0, DC, layout=gl.SliceLayout(0, bst))
    if OV8 and OVX & 8:
        so = (
            t * (H * KV_SPLITS * D_V)
            + (hs % H)[:, None] * (KV_SPLITS * D_V)
            + pk * D_V
            + (hs // H)[:, None] * DH
            + ds[None, :]
        ).to(gl.int32)
        lo = (gl.arange(0, BH, layout=gl.SliceLayout(1, mp)) < H)[:, None]
        z = gl.zeros([BH, DC], gl.float32, layout=mp)
        a0l = cdna4.mfma(pd, v0l, z)
        a0h = cdna4.mfma(pd, v0h, z)
        a1l = cdna4.mfma(pd, v1l, z)
        a1h = cdna4.mfma(pd, v1h, z)
        a0 = gl.where(lo, a0l, a0h)
        if OVX & 32 and EPI == 0:
            sel: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
                [[DC, 8]], [BH, DC], [1, 0]
            )
            se0 = gl.allocate_shared_memory(gl.bfloat16, [BH, DC], sel)
            se1 = gl.allocate_shared_memory(gl.bfloat16, [BH, DC], sel)
            se0.store((a0 * inv[:, None]).to(gl.bfloat16))
            a1 = gl.where(lo, a1l, a1h)
            se1.store((a1 * inv[:, None]).to(gl.bfloat16))
            cdna4.buffer_store(se0.load(bst), acc_ptr, so, cache=PST_CM)
            cdna4.buffer_store(se1.load(bst), acc_ptr, so + DC, cache=PST_CM)
        else:
            ab = gl.convert_layout((a0 * inv[:, None]).to(gl.bfloat16), bst)
            cdna4.buffer_store(ab, acc_ptr, so, cache=PST_CM)
            a1 = gl.where(lo, a1l, a1h)
            ab = gl.convert_layout((a1 * inv[:, None]).to(gl.bfloat16), bst)
            cdna4.buffer_store(ab, acc_ptr, so + DC, cache=PST_CM)
    else:
        so = (
            t * (H * KV_SPLITS * D_V)
            + hs[:, None] * (KV_SPLITS * D_V)
            + pk * D_V
            + ds[None, :]
        ).to(gl.int32)
        smask = (hs < H)[:, None]
        a_prev = gl.zeros([BH, DC], gl.float32, layout=mp)
        a_prev = cdna4.mfma(pd, skv.slice(0, DC, 1).load(bp), a_prev)
        for c in gl.static_range(1, 2):
            a_cur = gl.zeros([BH, DC], gl.float32, layout=mp)
            a_cur = cdna4.mfma(pd, skv.slice(c * DC, DC, 1).load(bp), a_cur)
            ab = gl.convert_layout((a_prev * inv[:, None]).to(gl.bfloat16), bst)
            cdna4.buffer_store(ab, acc_ptr, so + (c - 1) * DC, mask=smask, cache=PST_CM)
            a_prev = a_cur
        ab = gl.convert_layout((a_prev * inv[:, None]).to(gl.bfloat16), bst)
        cdna4.buffer_store(ab, acc_ptr, so + 1 * DC, mask=smask, cache=PST_CM)


# Exact-shape producer for M = 30 (8 heads): one 2*BK key tile per CTA.
@gluon.jit
def _sparse_mla_split_k_sb3_kernel(
    q_ptr,
    qr_ptr,
    kv_ptr,
    idx_ptr,
    lse_ptr,
    acc_ptr,
    qk_scale,
    lg_fp8,
    topk: gl.constexpr,
    H: gl.constexpr,
    KV_DIM: gl.constexpr,
    D_V: gl.constexpr,
    D_TAIL: gl.constexpr,
    BH: gl.constexpr,
    BK: gl.constexpr,
    KV_SPLITS: gl.constexpr,
    STRIDE_QN_T: gl.constexpr,
    STRIDE_QN_H: gl.constexpr,
    STRIDE_QR_T: gl.constexpr,
    STRIDE_QR_H: gl.constexpr,
    XCD_REMAP: gl.constexpr,
    XCD_N: gl.constexpr,
    XCD_PER: gl.constexpr,
):
    _lin = gl.program_id(0)
    _j = _lin % XCD_N * XCD_PER + _lin // XCD_N
    t = _j // KV_SPLITS
    pk = _j % KV_SPLITS
    BK2: gl.constexpr = 2 * BK
    mq: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 128], True, [1, 4])
    mt: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [1, 4])
    mp: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [1, 4])
    aq: gl.constexpr = gl.DotOperandLayout(0, mq, 16)
    bq: gl.constexpr = gl.DotOperandLayout(1, mq, 16)
    at: gl.constexpr = gl.DotOperandLayout(0, mt, 8)
    bt: gl.constexpr = gl.DotOperandLayout(1, mt, 8)
    ap: gl.constexpr = gl.DotOperandLayout(0, mp, 8)
    bp: gl.constexpr = gl.DotOperandLayout(1, mp, 8)
    blk: gl.constexpr = gl.BlockedLayout([1, 16], [2, 32], [4, 1], [1, 0])
    shk: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 16, [1, 0])
    irow = idx_ptr + t * topk + pk * BK2
    dk = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk))
    bla: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [1, 0])
    BQ: gl.constexpr = BK // 2
    nKq = gl.arange(0, BQ, layout=gl.SliceLayout(1, blk))
    s_kq = ()
    for qi in gl.static_range(4):
        s_kq = s_kq + (gl.load(irow + qi * BQ + nKq),)
    nTj = gl.arange(0, BK2, layout=gl.SliceLayout(1, bla))
    s_tj = gl.load(irow + nTj)
    mt0: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [1, 4])
    nSt = gl.arange(0, BK2, layout=gl.SliceLayout(0, mt0))
    vvt = gl.load(irow + nSt) >= 0
    hr = gl.minimum(gl.arange(0, BH, layout=gl.SliceLayout(1, at)), H - 1)
    dtq = gl.arange(0, D_TAIL, layout=gl.SliceLayout(0, at))
    qt = gl.load(qr_ptr + t * STRIDE_QR_T + hr[:, None] * STRIDE_QR_H + dtq[None, :])
    hb = gl.minimum(gl.arange(0, BH, layout=gl.SliceLayout(1, blk)), H - 1)
    qb = gl.load(q_ptr + t * STRIDE_QN_T + hb[:, None] * STRIDE_QN_H + dk[None, :])
    skv = gl.allocate_shared_memory(gl.float8e4nv, [BK2, D_V], shk)
    shta: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 4, [1, 0])
    idl: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    skt = gl.allocate_shared_memory(gl.float8e4nv, [BK2, D_TAIL], shta)
    skv_w = skv._reinterpret(gl.float8e4nv, [BK2, D_V], idl)
    skt_w = skt._reinterpret(gl.float8e4nv, [BK2, D_TAIL], idl)
    dta = gl.arange(0, D_TAIL, layout=gl.SliceLayout(0, bla))
    kvr = ()
    for qi in gl.static_range(1, 4):
        o_q = gl.maximum(s_kq[qi], 0).to(gl.uint32) * KV_DIM
        kvr = kvr + (
            gl.load(kv_ptr + o_q[:, None] + dk[None, :], cache_modifier=".cg"),
        )
    swq = (dk[None, :] ^ (nKq % 16 * 16)[:, None]).to(gl.uint32)
    for qi in gl.static_range(1):
        o_q = gl.maximum(s_kq[qi], 0).to(gl.uint32) * KV_DIM
        offk = gl.max_contiguous(gl.multiple_of(o_q[:, None] + swq, [1, 16]), [1, 16])
        async_copy.global_load_to_shared(
            skv_w.slice(qi * BQ, BQ, 0), kv_ptr + offk, cache_modifier=".cg"
        )
    o_tj = gl.maximum(s_tj, 0).to(gl.uint32) * KV_DIM
    offt = o_tj[:, None] + (D_V + (dta[None, :] ^ (nTj % 4 * 16)[:, None])).to(
        gl.uint32
    )
    offt = gl.max_contiguous(gl.multiple_of(offt, [1, 16]), [1, 16])
    async_copy.global_load_to_shared(skt_w, kv_ptr + offt, cache_modifier=".cg")
    async_copy.commit_group()
    for qi in gl.static_range(1, 4):
        skv.slice(qi * BQ, BQ, 0).store(kvr[qi - 1])
    q = gl.convert_layout(qb, aq)
    async_copy.wait_group(0)
    sq = gl.zeros([BH, BK2], gl.float32, layout=mq)
    sq = cdna4.mfma_scaled(
        q, None, "e4m3", skv.permute((1, 0)).load(bq), None, "e4m3", sq
    )
    sc = gl.convert_layout(sq, mt)
    sc = cdna4.mfma(qt, skt.permute((1, 0)).load(bt), sc)
    NEG: gl.constexpr = -3.4028234663852886e38
    sc = gl.where(vvt[None, :], sc * qk_scale, NEG)
    KH: gl.constexpr = BK2 // 2
    l3: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 32]],
        [[0, 1, 0], [0, 2, 0], [0, 4, 0], [1, 0, 0], [0, 0, 8], [0, 0, 16]],
        [[0, 0, 0], [0, 0, 0]],
        [],
        [2, 8, 64],
    )
    lz: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 32], [1, 0, 0]],
        [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 0, 0], [0, 0, 8], [0, 0, 16]],
        [[0, 0, 0], [0, 0, 0]],
        [],
        [2, 8, 64],
    )
    l2: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [0, 32]],
        [[1, 0], [2, 0], [4, 0], [0, 64], [0, 8], [0, 16]],
        [[0, 0], [0, 0]],
        [],
        [8, 128],
    )
    l3f: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 32], [0, 0, 64]],
        [[0, 1, 0], [0, 2, 0], [0, 4, 0], [1, 0, 0], [0, 0, 8], [0, 0, 16]],
        [[0, 0, 0], [0, 0, 0]],
        [],
        [2, 8, 128],
    )
    ssd: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 8, [1, 0])
    ss = gl.allocate_shared_memory(gl.float32, [BH, BK2], ssd)
    ss.store(sc)
    s3 = gl.convert_layout(
        gl.permute(gl.reshape(ss.slice(0, H, 0).load(l2), [H, 2, KH]), (1, 0, 2)), l3
    )
    vb = skv.load(bp)
    m8 = gl.maximum(gl.max(gl.max(s3, axis=2), axis=0), -1e30)
    p3 = gl.exp2(s3 - gl.expand_dims(gl.expand_dims(m8 - lg_fp8, 0), 2))
    l8 = gl.sum(gl.sum(p3, axis=2), axis=0)
    den8 = gl.maximum(l8, 1e-30)
    hd8 = l8 > 0.0
    lg8 = gl.log2(l8)
    lse8 = gl.where(hd8, lg8 - lg_fp8 + m8, NEG)
    ho8 = gl.arange(0, H, layout=gl.SliceLayout(0, gl.SliceLayout(2, l3)))
    cdna4.buffer_store(
        lse8,
        lse_ptr,
        (t * (H * KV_SPLITS) + ho8 * KV_SPLITS + pk).to(gl.int32),
        cache=".wt",
    )
    rinv8 = 1.0 / den8
    inv8h = gl.where(hd8, rinv8, 0.0)
    pf3 = p3.to(gl.float8e4nv)
    pz = gl.convert_layout(pf3, lz)
    pr = gl.convert_layout(
        gl.reshape(gl.permute(pz, (1, 0, 2)), [H, BK2]), gl.SliceLayout(0, l3f)
    )
    pd = gl.convert_layout(
        gl.reshape(
            gl.broadcast(
                gl.expand_dims(pr, 0), gl.zeros([2, H, BK2], gl.float8e4nv, layout=l3f)
            )[0],
            [BH, BK2],
        ),
        ap,
    )
    acc = gl.zeros([BH, D_V], gl.float32, layout=mp)
    acc = cdna4.mfma(pd, vb, acc)
    DH: gl.constexpr = D_V // 2
    a4 = gl.permute(gl.reshape(acc, [2, H, 2, DH]), (0, 1, 3, 2))
    (lo, hi) = gl.split(a4)
    l3s: gl.constexpr = lo.type.layout
    r3 = gl.arange(0, 2, layout=gl.SliceLayout(1, gl.SliceLayout(2, l3s)))
    sel = gl.where((r3 == 0)[:, None, None], lo, hi)
    invh = gl.convert_layout(inv8h, gl.SliceLayout(0, gl.SliceLayout(2, l3s)))
    sel = (sel * invh[None, :, None]).to(gl.bfloat16)
    bs3: gl.constexpr = gl.BlockedLayout([1, 1, 8], [1, 2, 32], [1, 4, 1], [2, 1, 0])
    selb = gl.convert_layout(sel, bs3)
    r3s = gl.arange(0, 2, layout=gl.SliceLayout(1, gl.SliceLayout(2, bs3)))
    hs3 = gl.arange(0, H, layout=gl.SliceLayout(0, gl.SliceLayout(2, bs3)))
    cs3 = gl.arange(0, DH, layout=gl.SliceLayout(0, gl.SliceLayout(1, bs3)))
    so3 = (
        t * (H * KV_SPLITS * D_V)
        + pk * D_V
        + r3s[:, None, None] * DH
        + hs3[None, :, None] * (KV_SPLITS * D_V)
        + cs3[None, None, :]
    ).to(gl.int32)
    cdna4.buffer_store(selb, acc_ptr, so3, cache=".wt")


@gluon.jit
def _mt_step(
    q,
    qt,
    skv,
    skt,
    vv,
    m_i,
    l_i,
    acc,
    qk_scale,
    lg_fp8,
    BH: gl.constexpr,
    BK: gl.constexpr,
    mt: gl.constexpr,
    ms: gl.constexpr,
    mp: gl.constexpr,
    bt: gl.constexpr,
    ap: gl.constexpr,
    bp: gl.constexpr,
):
    NEG: gl.constexpr = -3.4028234663852886e38
    sc = gl.zeros([BH, BK], gl.float32, layout=mt)
    sc = cdna4.mfma(q, skv.permute((1, 0)).load(bt), sc)
    sc = cdna4.mfma(qt, skt.permute((1, 0)).load(bt), sc)
    sc = gl.convert_layout(sc, ms)
    sc = gl.where(vv[None, :], sc * qk_scale, NEG)
    m_new = gl.maximum(m_i, gl.max(sc, axis=1))
    p = gl.exp2(sc - (m_new - lg_fp8)[:, None])
    pd = gl.convert_layout(p.to(gl.float8e4nv), ap)
    alpha = gl.exp2(m_i - m_new)
    l_i = l_i * alpha + gl.sum(p, axis=1)
    alpha_p = gl.convert_layout(alpha, gl.SliceLayout(1, mp))
    acc = cdna4.mfma(pd, skv.load(bp), acc * alpha_p[:, None])
    return (m_new, l_i, acc)


@gluon.jit
def _mt_offsets(
    irow,
    kt,
    NT: gl.constexpr,
    BK: gl.constexpr,
    BQ: gl.constexpr,
    KV_DIM: gl.constexpr,
    RAGGED: gl.constexpr,
    blk: gl.constexpr,
    btl: gl.constexpr,
    ms: gl.constexpr,
):
    if RAGGED:
        ktc = gl.minimum(kt, NT - 1)
    else:
        ktc = kt
    base = irow + ktc * BK
    nKq = gl.arange(0, BQ, layout=gl.SliceLayout(1, blk))
    nT = gl.arange(0, BK, layout=gl.SliceLayout(1, btl))
    nS = gl.arange(0, BK, layout=gl.SliceLayout(0, ms))
    oq = ()
    for qi in gl.static_range(4):
        oq = oq + (gl.maximum(gl.load(base + qi * BQ + nKq), 0).to(gl.uint32) * KV_DIM,)
    o_t = gl.maximum(gl.load(base + nT), 0).to(gl.uint32) * KV_DIM
    v = gl.load(base + nS) >= 0
    if RAGGED:
        v = v & (kt < NT)
    return (oq, o_t, v)


@gluon.jit
def _mt_dma(
    kv_ptr,
    oq,
    dst_w,
    ADQ: gl.constexpr,
    BQ: gl.constexpr,
    D_V: gl.constexpr,
    blk: gl.constexpr,
):
    dk = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk))
    nKq = gl.arange(0, BQ, layout=gl.SliceLayout(1, blk))
    swq = (dk[None, :] ^ (nKq % 16 * 16)[:, None]).to(gl.uint32)
    for qi in gl.static_range(ADQ):
        offk = gl.max_contiguous(
            gl.multiple_of(oq[qi][:, None] + swq, [1, 16]), [1, 16]
        )
        async_copy.global_load_to_shared(
            dst_w.slice(qi * BQ, BQ, 0), kv_ptr + offk, cache_modifier=".cg"
        )


@gluon.jit
def _mt_load(
    kv_ptr,
    oq,
    o_t,
    dst_w,
    dst_tw,
    ADQ: gl.constexpr,
    BQ: gl.constexpr,
    D_V: gl.constexpr,
    D_TAIL: gl.constexpr,
    TDM: gl.constexpr,
    DMF: gl.constexpr,
    blk: gl.constexpr,
    btl: gl.constexpr,
):
    dk = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk))
    dtt = gl.arange(0, D_TAIL, layout=gl.SliceLayout(0, btl))
    if DMF:
        _mt_dma(kv_ptr, oq, dst_w, ADQ, BQ, D_V, blk)
    kvr = ()
    for qi in gl.static_range(ADQ, 4):
        kvr = kvr + (
            gl.load(kv_ptr + oq[qi][:, None] + dk[None, :], cache_modifier=".cg"),
        )
    if TDM:
        kt_ = o_t
    else:
        kt_ = gl.load(
            kv_ptr + o_t[:, None] + (D_V + dtt)[None, :], cache_modifier=".cg"
        )
    if not DMF:
        _mt_dma(kv_ptr, oq, dst_w, ADQ, BQ, D_V, blk)
    if TDM:
        nT = gl.arange(0, o_t.shape[0], layout=gl.SliceLayout(1, btl))
        offt = o_t[:, None] + (D_V + (dtt[None, :] ^ (nT % 4 * 16)[:, None])).to(
            gl.uint32
        )
        offt = gl.max_contiguous(gl.multiple_of(offt, [1, 16]), [1, 16])
        async_copy.global_load_to_shared(dst_tw, kv_ptr + offt, cache_modifier=".cg")
    async_copy.commit_group()
    return (kvr, kt_)


@gluon.jit
def _mt_store(
    skb, skt, kvr, kt_, ADQ: gl.constexpr, BQ: gl.constexpr, TDM: gl.constexpr
):
    for qi in gl.static_range(ADQ, 4):
        skb.slice(qi * BQ, BQ, 0).store(kvr[qi - ADQ])
    if not TDM:
        skt.store(kt_)


# Exact-shape producer for 16 heads, M in {36, 42}: TPS tiles per split,
# double-buffered hybrid DMA + register KV gather.
@gluon.jit
def _sparse_mla_split_k_mt_kernel(
    q_ptr,
    qr_ptr,
    kv_ptr,
    idx_ptr,
    lse_ptr,
    acc_ptr,
    qk_scale,
    lg_fp8,
    topk: gl.constexpr,
    H: gl.constexpr,
    KV_DIM: gl.constexpr,
    D_V: gl.constexpr,
    D_TAIL: gl.constexpr,
    BH: gl.constexpr,
    BK: gl.constexpr,
    KV_SPLITS: gl.constexpr,
    TPS: gl.constexpr,
    NT: gl.constexpr,
    RAGGED: gl.constexpr,
    STRIDE_QN_T: gl.constexpr,
    STRIDE_QN_H: gl.constexpr,
    STRIDE_QR_T: gl.constexpr,
    STRIDE_QR_H: gl.constexpr,
    XCD_REMAP: gl.constexpr,
    XCD_N: gl.constexpr,
    XCD_PER: gl.constexpr,
    DYN: gl.constexpr = 0,
    ADQ: gl.constexpr = 2,
    TDM: gl.constexpr = 0,
    DMF: gl.constexpr = 0,
):
    if XCD_REMAP:
        _lin = gl.program_id(0)
        _j = _lin % XCD_N * XCD_PER + _lin // XCD_N
        t = _j // KV_SPLITS
        pk = _j % KV_SPLITS
    else:
        t = gl.program_id(0)
        pk = gl.program_id(2)
    mt: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [1, 4])
    ms: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [4, 1])
    mp: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [1, 4])
    at: gl.constexpr = gl.DotOperandLayout(0, mt, 8)
    bt: gl.constexpr = gl.DotOperandLayout(1, mt, 8)
    ap: gl.constexpr = gl.DotOperandLayout(0, mp, 8)
    bp: gl.constexpr = gl.DotOperandLayout(1, mp, 8)
    blk: gl.constexpr = gl.BlockedLayout([1, 16], [2, 32], [4, 1], [1, 0])
    blt: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    sht: gl.constexpr = gl.SwizzledSharedLayout(8, 4, 4, [1, 0])
    irow = idx_ptr + t * topk
    kt0 = pk * TPS
    NEG: gl.constexpr = -3.4028234663852886e38
    BQ: gl.constexpr = BK // 4
    shk3: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 16, [1, 0])
    idl3: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    skv2 = gl.allocate_shared_memory(gl.float8e4nv, [2, BK, D_V], shk3)
    skv2_w = skv2._reinterpret(gl.float8e4nv, [2, BK, D_V], idl3)
    if TDM:
        btl: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [1, 0])
        shta: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 4, [1, 0])
        skt2 = gl.allocate_shared_memory(gl.float8e4nv, [2, BK, D_TAIL], shta)
        skt2_w = skt2._reinterpret(gl.float8e4nv, [2, BK, D_TAIL], idl3)
        skt = skt2
    else:
        btl: gl.constexpr = blt
        skt = gl.allocate_shared_memory(gl.float8e4nv, [BK, D_TAIL], sht)
        skt2 = skt
        skt2_w = skv2_w
    (oq, o_tA, vA) = _mt_offsets(irow, kt0, NT, BK, BQ, KV_DIM, RAGGED, blk, btl, ms)
    (kvr, ktA) = _mt_load(
        kv_ptr,
        oq,
        o_tA,
        skv2_w.index(0),
        skt2_w.index(0),
        ADQ,
        BQ,
        D_V,
        D_TAIL,
        TDM,
        DMF,
        blk,
        btl,
    )
    dk = gl.arange(0, D_V, layout=gl.SliceLayout(0, blk))
    hr = gl.minimum(gl.arange(0, BH, layout=gl.SliceLayout(1, at)), H - 1)
    dtq = gl.arange(0, D_TAIL, layout=gl.SliceLayout(0, at))
    qt = gl.load(qr_ptr + t * STRIDE_QR_T + hr[:, None] * STRIDE_QR_H + dtq[None, :])
    hb = gl.minimum(gl.arange(0, BH, layout=gl.SliceLayout(1, blk)), H - 1)
    qb = gl.load(q_ptr + t * STRIDE_QN_T + hb[:, None] * STRIDE_QN_H + dk[None, :])
    q = gl.convert_layout(qb, at)
    m_i = gl.full([BH], -1e30, gl.float32, layout=gl.SliceLayout(1, ms))
    l_i = gl.zeros([BH], gl.float32, layout=gl.SliceLayout(1, ms))
    acc = gl.zeros([BH, D_V], gl.float32, layout=mp)
    if DYN:
        for i in range(0, TPS - 1):
            b = i % 2
            skb = skv2.index(b)
            sktb = skt
            vv = vA
            _mt_store(skb, sktb, kvr, ktA, ADQ, BQ, TDM)
            (oq, o_tA, vA) = _mt_offsets(
                irow, kt0 + i + 1, NT, BK, BQ, KV_DIM, RAGGED, blk, btl, ms
            )
            (kvr, ktA) = _mt_load(
                kv_ptr,
                oq,
                o_tA,
                skv2_w.index(1 - b),
                skt2_w.index(1 - b),
                ADQ,
                BQ,
                D_V,
                D_TAIL,
                TDM,
                DMF,
                blk,
                btl,
            )
            async_copy.wait_group(1)
            (m_i, l_i, acc) = _mt_step(
                q,
                qt,
                skb,
                sktb,
                vv,
                m_i,
                l_i,
                acc,
                qk_scale,
                lg_fp8,
                BH,
                BK,
                mt,
                ms,
                mp,
                bt,
                ap,
                bp,
            )
        skb = skv2.index((TPS - 1) % 2)
        sktb = skt
        _mt_store(skb, sktb, kvr, ktA, ADQ, BQ, TDM)
        async_copy.wait_group(0)
        (m_i, l_i, acc) = _mt_step(
            q,
            qt,
            skb,
            sktb,
            vA,
            m_i,
            l_i,
            acc,
            qk_scale,
            lg_fp8,
            BH,
            BK,
            mt,
            ms,
            mp,
            bt,
            ap,
            bp,
        )
    else:
        for i in gl.static_range(TPS):
            skb = skv2.index(i % 2)
            sktb = skt2.index(i % 2)
            vv = vA
            _mt_store(skb, sktb, kvr, ktA, ADQ, BQ, TDM)
            if i + 1 == TPS:
                async_copy.wait_group(0)
            if i + 1 < TPS:
                (oq, o_tA, vA) = _mt_offsets(
                    irow, kt0 + i + 1, NT, BK, BQ, KV_DIM, RAGGED, blk, btl, ms
                )
                (kvr, ktA) = _mt_load(
                    kv_ptr,
                    oq,
                    o_tA,
                    skv2_w.index((i + 1) % 2),
                    skt2_w.index((i + 1) % 2),
                    ADQ,
                    BQ,
                    D_V,
                    D_TAIL,
                    TDM,
                    DMF,
                    blk,
                    btl,
                )
                async_copy.wait_group(1)
            sc = gl.zeros([BH, BK], gl.float32, layout=mt)
            sc = cdna4.mfma(q, skb.permute((1, 0)).load(bt), sc)
            sc = cdna4.mfma(qt, sktb.permute((1, 0)).load(bt), sc)
            sc = gl.convert_layout(sc, ms)
            sc = gl.where(vv[None, :], sc * qk_scale, NEG)
            m_new = gl.maximum(m_i, gl.max(sc, axis=1))
            p = gl.exp2(sc - (m_new - lg_fp8)[:, None])
            pd = gl.convert_layout(p.to(gl.float8e4nv), ap)
            if i == 0:
                l_i = gl.sum(p, axis=1)
                acc = cdna4.mfma(pd, skb.load(bp), acc)
            else:
                alpha = gl.exp2(m_i - m_new)
                l_i = l_i * alpha + gl.sum(p, axis=1)
                alpha_p = gl.convert_layout(alpha, gl.SliceLayout(1, mp))
                acc = cdna4.mfma(pd, skb.load(bp), acc * alpha_p[:, None])
            m_i = m_new
    den = gl.maximum(l_i, 1e-30)
    hd = l_i > 0.0
    inv = gl.convert_layout(gl.where(hd, 1.0 / den, 0.0), gl.SliceLayout(1, mp))
    acc = acc * inv[:, None]
    lse = gl.where(hd, gl.log2(l_i) - lg_fp8 + m_i, NEG)
    ho = gl.arange(0, BH, layout=gl.SliceLayout(1, ms))
    lse_off = (t * (H * KV_SPLITS) + ho * KV_SPLITS + pk).to(gl.int32)
    cdna4.buffer_store(lse, lse_ptr, lse_off, mask=ho < H, cache=".wt")
    bso: gl.constexpr = gl.BlockedLayout([1, 8], [2, 32], [4, 1], [1, 0])
    accb = gl.convert_layout(acc.to(gl.bfloat16), bso)
    hs = gl.arange(0, BH, layout=gl.SliceLayout(1, bso))
    ds = gl.arange(0, D_V, layout=gl.SliceLayout(0, bso))
    so = (
        t * (H * KV_SPLITS * D_V)
        + hs[:, None] * (KV_SPLITS * D_V)
        + pk * D_V
        + ds[None, :]
    ).to(gl.int32)
    cdna4.buffer_store(accb, acc_ptr, so, mask=(hs < H)[:, None], cache=".wt")


# Reduce for the exact-shape producers and the small-M generic rows.
@gluon.jit
def _sparse_mla_reduce_tuned_kernel(
    lse_ptr,
    acc_ptr,
    out_ptr,
    H: gl.constexpr,
    D_V: gl.constexpr,
    SP: gl.constexpr,
    TK: gl.constexpr,
    VEC: gl.constexpr,
    NW: gl.constexpr,
    XCD_REMAP: gl.constexpr,
    XCD_N: gl.constexpr,
    XCD_PER: gl.constexpr,
    KS: gl.constexpr,
):
    DC: gl.constexpr = VEC * (64 // TK) * NW
    D_CHUNKS: gl.constexpr = D_V // DC
    if XCD_REMAP:
        _lin = gl.program_id(0)
        _j = _lin % XCD_N * XCD_PER + _lin // XCD_N
        t = _j // (H * D_CHUNKS)
        _r = _j % (H * D_CHUNKS)
        h = _r // D_CHUNKS
        dc = _r % D_CHUNKS
    else:
        t = gl.program_id(0)
        h = gl.program_id(1)
        dc = gl.program_id(2)
    la: gl.constexpr = gl.BlockedLayout(
        [SP // TK, VEC], [TK, 64 // TK], [1, NW], [1, 0]
    )
    base = t * (H * KS) + h * KS
    kr = gl.arange(0, SP, layout=gl.SliceLayout(1, la))
    dd = dc * DC + gl.arange(0, DC, layout=gl.SliceLayout(0, la))
    a_off = (base * D_V + kr[:, None] * D_V + dd[None, :]).to(gl.int32)
    l_off = (base + kr).to(gl.int32)
    if KS == SP:
        a_p = cdna4.buffer_load(acc_ptr, a_off, cache=".cg")
        lse_r = cdna4.buffer_load(lse_ptr, l_off)
    else:
        km = kr < KS
        a_p = cdna4.buffer_load(
            acc_ptr, a_off, mask=km[:, None], other=0.0, cache=".cg"
        )
        lse_r = cdna4.buffer_load(lse_ptr, l_off, mask=km, other=-3.4e38)
    lse_max = gl.max(lse_r, axis=0)
    w_r = gl.exp2(lse_r - lse_max)
    num = gl.sum(a_p.to(gl.float32) * w_r[:, None], axis=0)
    den = gl.sum(w_r, axis=0)
    o = num * (1.0 / gl.maximum(den, 1e-30))
    o_off = (t * (H * D_V) + h * D_V + dd).to(gl.int32)
    cdna4.buffer_store(o.to(gl.bfloat16), out_ptr, o_off, cache=".wt")


_D_V = 512
_D_TAIL = 64
_KV_DIM = _D_V + _D_TAIL
_TOPK = 2048
_BLOCK_H = 16
_LOG2E = 1.4426950408889634
_LOG2_FP8_MAX = math.log2(448.0)
_NUM_XCD = 8
# Largest M served per local head count; above it the Triton path is faster.
_MAX_M = {8: 128, 16: 240}

# Exact-shape producers, measured on MI355X; re-tune when a kernel changes.
# (H, M) -> (producer, constexprs, (TK, VEC) of the reduce)
_OV_A = dict(BK=32, KV_SPLITS=32, EPI=1, OVK=7, OVX=15, PST_CM=".wt")
_TUNED_ROUTES = {
    (8, 6): ("ov", _OV_A, (4, 2)),
    (8, 12): ("ov", _OV_A, (4, 2)),
    (8, 18): (
        "ov",
        dict(BK=32, KV_SPLITS=32, EPI=0, OVK=5, OVX=111, PRO=2, PST_CM=".wt"),
        (4, 4),
    ),
    (8, 30): ("sb3", dict(BK=64, KV_SPLITS=16), (4, 4)),
    (16, 6): ("ov", _OV_A, (4, 2)),
    (16, 36): (
        "mt",
        dict(BK=64, KV_SPLITS=8, TPS=4, ADQ=4, DMF=1, DYN=0, TDM=1),
        (1, 2),
    ),
    (16, 42): (
        "mt",
        dict(BK=64, KV_SPLITS=8, TPS=4, ADQ=3, DMF=0, DYN=1, TDM=0),
        (1, 4),
    ),
}

# Generic producer, 16 heads: (largest M, kv splits), measured on MI355X.
_H16_SPLITS = (
    (8, 32),
    (29, 16),
    (30, 8),
    (32, 16),
    (72, 8),
    (120, 4),
    (144, 8),
    (192, 4),
    (240, 2),
)


@functools.lru_cache(maxsize=8)
def _num_cus(device_index: int) -> int:
    return get_device_core_count(device_index)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def gluon_sparse_mla_decode_supported(
    q_all: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor
) -> bool:
    if q_all.dim() != 3:
        return False
    m, h, d = q_all.shape
    return (
        h in _MAX_M
        and 1 <= m <= _MAX_M[h]
        and d == _KV_DIM
        and q_all.dtype == torch.float8_e4m3fn
        and q_all.stride() == (h * _KV_DIM, _KV_DIM, 1)
        and kv.dtype == torch.float8_e4m3fn
        and kv.shape[-1] == _KV_DIM
        and kv.stride(0) == _KV_DIM
        # Row offsets are formed in uint32.
        and kv.shape[0] * _KV_DIM < (1 << 32)
        and indices.dtype == torch.int32
        and tuple(indices.shape) == (m, _TOPK)
    )


def gluon_sparse_mla_decode(
    q_all: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Sparse MLA over ``indices``; returns [1, M, H, 512] bf16."""
    m, h, _ = q_all.shape
    q_nope = q_all[:, :, :_D_V]
    q_rope = q_all[:, :, _D_V:]
    indices = indices.contiguous()
    qk_scale = float(sm_scale) * _LOG2E
    out = torch.empty(m, h, _D_V, dtype=torch.bfloat16, device=q_all.device)
    common = dict(
        topk=_TOPK,
        H=h,
        KV_DIM=_KV_DIM,
        D_V=_D_V,
        D_TAIL=_D_TAIL,
        BH=_BLOCK_H,
        STRIDE_QN_T=q_all.stride(0),
        STRIDE_QN_H=q_all.stride(1),
        STRIDE_QR_T=q_all.stride(0),
        STRIDE_QR_H=q_all.stride(1),
    )
    route = _TUNED_ROUTES.get((h, m))
    if route is None:
        _generic_decode(q_nope, q_rope, kv, indices, out, qk_scale, common)
    else:
        _tuned_decode(q_nope, q_rope, kv, indices, out, qk_scale, common, *route)
    return out.unsqueeze(0)


def _generic_decode(q_nope, q_rope, kv, indices, out, qk_scale, common):
    m, h, _ = out.shape
    if h == 8:
        block_k = 32 if m <= 3 or m == 60 else 64
        num_tiles = _TOPK // block_k
        # Largest power-of-two split count keeping the grid within ~3 waves.
        max_ctas = 3 * _num_cus(out.device.index or 0)
        splits = 2
        while splits * 2 <= num_tiles and m * splits * 2 <= max_ctas:
            splits *= 2
    else:
        block_k = 64
        num_tiles = _TOPK // block_k
        splits = next(s for m_hi, s in _H16_SPLITS if m <= m_hi)
    lse, acc = _partials(m, h, splits, out.device)
    _sparse_mla_split_k_gen_kernel[(m, 1, splits)](
        q_nope,
        q_rope,
        kv,
        indices,
        lse,
        acc,
        qk_scale,
        _LOG2_FP8_MAX,
        splits,
        num_tiles // splits,
        **common,
        BK=block_k,
        NT=num_tiles,
        PST_CM=".cs" if m == 48 else "",
        num_warps=4,
    )
    if h == 8 and m <= 48:
        tk = 8 if m <= 5 else (4 if m <= 24 else 1)
        _reduce_tuned(lse, acc, out, splits, tk, vec=4, xcd_n=1, xcd=False)
    else:
        _sparse_mla_reduce_gen_kernel[(m, h, _D_V // 128)](
            lse,
            acc,
            out,
            splits,
            H=h,
            D_V=_D_V,
            SP=num_tiles,
            TK=4,
            VEC=8,
            NW=1,
            num_warps=1,
        )


def _tuned_decode(
    q_nope, q_rope, kv, indices, out, qk_scale, common, producer, row, reduce_cfg
):
    m, h, _ = out.shape
    splits = row["KV_SPLITS"]
    tk, vec = reduce_cfg
    d_chunks = _D_V // (vec * (64 // tk))
    # Spread one row's CTAs over the XCDs (producer and reduce flip together).
    xcd = m % _NUM_XCD != 0 and (m * splits) % _NUM_XCD == 0
    xcd = xcd and (m * h * d_chunks) % _NUM_XCD == 0
    grid = (m * splits,) if xcd else (m, 1, splits)
    xcd_args = dict(
        XCD_REMAP=xcd, XCD_N=_NUM_XCD, XCD_PER=m * splits // _NUM_XCD if xcd else 0
    )
    lse, acc = _partials(m, h, splits, out.device)
    args = (q_nope, q_rope, kv, indices, lse, acc, qk_scale, _LOG2_FP8_MAX)
    if producer == "ov":
        _sparse_mla_split_k_ov_kernel[grid](
            *args, **common, **xcd_args, **row, DUP=int(2 * h == _BLOCK_H), num_warps=4
        )
    elif producer == "sb3":
        _sparse_mla_split_k_sb3_kernel[grid](
            *args, **common, **xcd_args, **row, num_warps=4
        )
    else:
        num_tiles = _TOPK // row["BK"]
        _sparse_mla_split_k_mt_kernel[grid](
            *args,
            **common,
            **xcd_args,
            **row,
            NT=num_tiles,
            RAGGED=splits * row["TPS"] > num_tiles,
            num_warps=4,
        )
    _reduce_tuned(lse, acc, out, splits, tk, vec, xcd_n=_NUM_XCD, xcd=xcd)


def _reduce_tuned(lse, acc, out, splits, tk, vec, xcd_n, xcd):
    m, h, _ = out.shape
    d_chunks = _D_V // (vec * (64 // tk))
    n = m * h * d_chunks
    _sparse_mla_reduce_tuned_kernel[(n,) if xcd else (m, h, d_chunks)](
        lse,
        acc,
        out,
        H=h,
        D_V=_D_V,
        SP=_next_pow2(splits),
        TK=tk,
        VEC=vec,
        NW=1,
        XCD_REMAP=xcd,
        XCD_N=xcd_n,
        XCD_PER=n // xcd_n if xcd else 0,
        KS=splits,
        num_warps=1,
    )


def _partials(m, h, splits, device):
    lse = torch.empty(m, h, splits, dtype=torch.float32, device=device)
    acc = torch.empty(m, h, splits, _D_V, dtype=torch.bfloat16, device=device)
    return lse, acc
