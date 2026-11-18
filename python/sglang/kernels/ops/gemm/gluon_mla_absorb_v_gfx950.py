"""Gluon absorb_v BMM for gfx950 MLA decode: ``attn @ w_vc`` with fp8 activation quant.

Replaces aiter ``batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant``
in ``rocm_absorb_v_bmm`` at decode sizes (K = kv_lora_rank 512, N = v_head_dim 256,
8 or 16 heads): per-128-group fp8 quant of the bf16 activations, four 16x16x128
fp8 MFMAs per tile, per-group scales applied in the epilogue. Same math as the
aiter kernel; the speedup is in the launch shape and the epilogue.

  x:   [M, H, 512] bf16
  w:   [H, 256, 512] fp8 e4m3 (per-tensor scale)
  out: [M, H, 256] bf16, written in place
"""

import contextlib
import threading

import torch
import triton
from triton import knobs
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna4
from triton.language.extra.hip.libdevice import fast_dividef


@gluon.constexpr_function
def _log2(n):
    return int(n).bit_length() - 1


@gluon.jit
def _pack_u64(lo, hi):
    return hi.to(gl.uint32, bitcast=True).to(gl.uint64) << 32 | lo.to(
        gl.uint32, bitcast=True
    ).to(gl.uint64)


@gluon.jit
def _keep_lhs(a, b):
    return a


@gluon.jit(do_not_specialize=["M"])
def _mla_absorb_v_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    M,
    NUM_BATCH: gl.constexpr,
    N: gl.constexpr,
    DTYPE_MAX: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    NW: gl.constexpr,
    WN: gl.constexpr,
    SINGLE_M: gl.constexpr,
    EVEN_M: gl.constexpr,
    PRELOAD_DW: gl.constexpr,
    CS_AB: gl.constexpr,
    CS_AM: gl.constexpr,
    CS_BB: gl.constexpr,
    CS_BN: gl.constexpr,
    CS_CB: gl.constexpr,
    CS_CM: gl.constexpr,
    MC: gl.constexpr = 0,
    STC: gl.constexpr = 0,
    BSE: gl.constexpr = 0,
    TAIL: gl.constexpr = 0,
    BCM: gl.constexpr = 0,
    EPI: gl.constexpr = 0,
    SLB: gl.constexpr = 0,
    QL: gl.constexpr = 0,
    HZ: gl.constexpr = 0,
    HB: gl.constexpr = 0,
):
    if MC > 0:
        Mr = MC
    else:
        Mr = M
    BK: gl.constexpr = 128
    SLBX: gl.constexpr = SLB == 3 or SLB == 4
    NG: gl.constexpr = 4
    LA3: gl.constexpr = (
        gl.BlockedLayout([1, 1, 32], [4, 4, 4], [NW, 1, 1], [2, 1, 0])
        if QL >= 1
        else gl.BlockedLayout([1, 1, 16], [2, 4, 8], [NW, 1, 1], [2, 1, 0])
    )
    KV: gl.constexpr = 32 if QL >= 1 else 16
    MMA: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 128], True, [NW // WN, WN])
    DA: gl.constexpr = gl.DotOperandLayout(0, MMA, 16)
    DB: gl.constexpr = gl.DotOperandLayout(1, MMA, 16)
    QMP: gl.constexpr = 8 if BM >= 16 else BM // 2
    SH_Q: gl.constexpr = gl.SwizzledSharedLayout(16, 2, QMP, [1, 0])
    SH_S: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    gid = gl.program_id(0)
    if HB > 0:
        batch_id = gid & NUM_BATCH - 1
        pid = gid >> _log2(NUM_BATCH)
    else:
        batch_id = gid % NUM_BATCH
        pid = gid // NUM_BATCH
    num_pid_n: gl.constexpr = N // BN
    if SINGLE_M:
        pid_m = 0
        pid_n = pid
    elif HB > 0:
        pid_m = pid >> _log2(num_pid_n)
        pid_n = pid & num_pid_n - 1
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    row0 = pid_m * BM
    if BSE == 1 and SLB == 0:
        b_scale = gl.load(b_scale_ptr)
    if SLB == 3:
        bs64 = gl.inline_asm_elementwise(
            "s_load_dwordx2 $0, $1, 0x0",
            "=s,s",
            [b_scale_ptr.to(gl.int64, bitcast=True)],
            dtype=gl.int64,
            is_pure=False,
            pack=1,
        )
    if HB & 1 != 0:
        a_base = a_ptr
        a_uo = batch_id * CS_AB + row0 * CS_AM
    else:
        a_base = a_ptr + batch_id * CS_AB + row0 * CS_AM
        a_uo = 0
    ra = gl.arange(0, BM, layout=gl.SliceLayout(1, gl.SliceLayout(2, LA3)))
    ga = gl.arange(0, NG, layout=gl.SliceLayout(0, gl.SliceLayout(2, LA3)))
    ka = gl.arange(0, BK, layout=gl.SliceLayout(0, gl.SliceLayout(1, LA3)))
    ra3 = gl.expand_dims(gl.expand_dims(ra, 1), 2)
    ga3 = gl.expand_dims(gl.expand_dims(ga, 0), 2)
    ka3 = gl.expand_dims(gl.expand_dims(ka, 0), 1)
    if QL == 2:
        kmem = ka3 % 16 + ka3 // 16 % 2 * 64 + ka3 // 32 * 16
        offa = ra3 * CS_AM + a_uo + ga3 * BK + kmem
        offa = gl.max_contiguous(gl.multiple_of(offa, [1, 1, 16]), [1, 1, 16])
    else:
        offa = ra3 * CS_AM + a_uo + ga3 * BK + ka3
    if EVEN_M:
        aw = cdna4.buffer_load(a_base, offa)
    else:
        am = row0 + ra3 < Mr
        aw = cdna4.buffer_load(a_base, offa, mask=am)
    if HB & 2 != 0:
        b_base = b_ptr
        b_uo = batch_id * CS_BB + pid_n * (BN * CS_BN)
    else:
        b_base = b_ptr + batch_id * CS_BB + pid_n * (BN * CS_BN)
        b_uo = 0
    kb = gl.arange(0, BK, layout=gl.SliceLayout(1, DB))
    nb = gl.arange(0, BN, layout=gl.SliceLayout(0, DB))
    offb = gl.expand_dims(kb, 1) + gl.expand_dims(nb * CS_BN + b_uo, 0)
    BCH: gl.constexpr = (
        ".cg" if BCM == 0 else "" if BCM == 1 else ".ca" if BCM == 2 else ".cv"
    )
    b0 = cdna4.buffer_load(b_base, offb, cache=BCH)
    b1 = cdna4.buffer_load(b_base, offb + BK, cache=BCH)
    b2 = cdna4.buffer_load(b_base, offb + 2 * BK, cache=BCH)
    b3 = cdna4.buffer_load(b_base, offb + 3 * BK, cache=BCH)
    if BSE == 0 and SLB == 0:
        b_scale = gl.load(b_scale_ptr)
    if TAIL > 0:
        r_epi = gl.arange(0, BM, layout=gl.SliceLayout(1, MMA))
        n_epi = gl.arange(0, BN, layout=gl.SliceLayout(0, MMA))
        if HB & 4 != 0:
            c_uo = batch_id * CS_CB + row0 * CS_CM + pid_n * BN
        else:
            c_uo = 0
        offE = gl.expand_dims(r_epi * CS_CM + c_uo, 1) + gl.expand_dims(n_epi, 0)
        if not EVEN_M:
            offE = gl.where(gl.expand_dims(row0 + r_epi < Mr, 1), offE, 1073741824)
        offE = gl.inline_asm_elementwise(
            "", "=v,0", [offE], dtype=gl.int32, is_pure=TAIL == 1, pack=1
        )
        offE = gl.max_contiguous(gl.multiple_of(offE, [1, 4]), [1, 4])
    (a_lo, a_hi) = gl.split(gl.reshape(aw, [BM, NG, BK // 2, 2]))
    f_lo = a_lo.to(gl.float32)
    f_hi = a_hi.to(gl.float32)
    a4 = gl.reshape(gl.abs(aw.to(gl.float32)), [BM, NG, BK // KV, KV])
    pmx = gl.max(a4, axis=3)
    if QL >= 1:
        pfull = gl.inline_asm_elementwise(
            "s_nop 1\n\tv_max_u32_dpp $0, $1, $1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\ts_nop 1\n\tv_max_u32_dpp $0, $0, $0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n\tv_max_f32_e32 $0, 0x2edbe6ff, $0",
            "=&v,v",
            [pmx],
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        p5 = gl.permute(gl.reshape(pmx, [2, BM // 2, NG, BK // 16]), (1, 2, 3, 0))
        (pa, pb) = gl.split(p5)
        (ra_, rb_) = gl.inline_asm_elementwise(
            "s_nop 1\n\tv_max_u32_dpp $0, $2, $2 row_half_mirror row_mask:0xf bank_mask:0xf\n\tv_max_u32_dpp $1, $3, $3 row_half_mirror row_mask:0xf bank_mask:0xf\n\ts_nop 0\n\tv_max_u32_dpp $0, $0, $0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\tv_max_u32_dpp $1, $1, $1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\ts_nop 0\n\tv_max_u32_dpp $0, $0, $0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n\tv_max_u32_dpp $1, $1, $1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n\tv_max_f32_e32 $0, 0x2edbe6ff, $0\n\tv_max_f32_e32 $1, 0x2edbe6ff, $1",
            "=&v,=&v,v,v",
            [pa, pb],
            dtype=(gl.float32, gl.float32),
            is_pure=True,
            pack=1,
        )
        pfull = gl.reshape(
            gl.permute(gl.join(ra_, rb_), (3, 0, 1, 2)), [BM, NG, BK // KV]
        )
    amx = gl.convert_layout(gl.reduce(pfull, 2, _keep_lhs), gl.SliceLayout(2, LA3))
    sm = amx * (1.0 / DTYPE_MAX)
    rcp = fast_dividef(1.0, sm)
    u_lo = f_lo.to(gl.uint32, bitcast=True).to(gl.uint64)
    u_hi = f_hi.to(gl.uint32, bitcast=True).to(gl.uint64)
    pr = u_hi << 32 | u_lo
    ru = (
        gl.expand_dims(
            gl.convert_layout(
                rcp, gl.SliceLayout(2, f_lo.type.layout), assert_trivial=True
            ),
            2,
        )
        .to(gl.uint32, bitcast=True)
        .to(gl.uint64)
    )
    rr = ru << 32 | ru
    pm = gl.inline_asm_elementwise(
        "v_pk_mul_f32 $0, $1, $2",
        "=v,v,v",
        [pr, rr],
        dtype=gl.uint64,
        is_pure=True,
        pack=1,
    )
    m_lo = (pm & 4294967295).to(gl.uint32).to(gl.float32, bitcast=True)
    m_hi = (pm >> 32).to(gl.uint32).to(gl.float32, bitcast=True)
    q3 = gl.convert_layout(gl.reshape(gl.join(m_lo, m_hi), [BM, NG, BK]), LA3).to(
        gl.float8e4nv
    )
    if QL == 2:
        q5 = gl.permute(gl.reshape(q3, [BM, NG, 4, 2, 16]), (1, 0, 3, 2, 4))
        sq = gl.allocate_shared_memory(
            gl.float8e4nv, [NG * BM, BK], SH_Q, gl.reshape(q5, [NG * BM, BK])
        )
    else:
        sq = gl.allocate_shared_memory(
            gl.float8e4nv,
            [NG * BM, BK],
            SH_Q,
            gl.reshape(gl.permute(q3, (1, 0, 2)), [NG * BM, BK]),
        )
    ss = gl.allocate_shared_memory(
        gl.float32, [NG * BM], SH_S, gl.reshape(gl.permute(sm, (1, 0)), [NG * BM])
    )
    qa0 = sq.slice(0 * BM, BM).load(DA)
    qa1 = sq.slice(1 * BM, BM).load(DA)
    qa2 = sq.slice(2 * BM, BM).load(DA)
    qa3 = sq.slice(3 * BM, BM).load(DA)
    SL: gl.constexpr = gl.SliceLayout(1, MMA)
    s0 = ss.slice(0 * BM, BM).load(SL)
    s1 = ss.slice(1 * BM, BM).load(SL)
    s2 = ss.slice(2 * BM, BM).load(SL)
    s3 = ss.slice(3 * BM, BM).load(SL)
    zero = gl.zeros([BM, BN], gl.float32, MMA)
    d0 = cdna4.mfma_scaled(qa0, None, "e4m3", b0, None, "e4m3", zero)
    d1 = cdna4.mfma_scaled(qa1, None, "e4m3", b1, None, "e4m3", zero)
    d2 = cdna4.mfma_scaled(qa2, None, "e4m3", b2, None, "e4m3", zero)
    d3 = cdna4.mfma_scaled(qa3, None, "e4m3", b3, None, "e4m3", zero)
    (e0l, e0h) = gl.split(gl.reshape(d0, [BM, BN // 2, 2]))
    (e1l, e1h) = gl.split(gl.reshape(d1, [BM, BN // 2, 2]))
    (e2l, e2h) = gl.split(gl.reshape(d2, [BM, BN // 2, 2]))
    (e3l, e3h) = gl.split(gl.reshape(d3, [BM, BN // 2, 2]))
    LE: gl.constexpr = e0l.type.layout
    SE: gl.constexpr = gl.SliceLayout(1, LE)
    p0 = _pack_u64(e0l, e0h)
    p1 = _pack_u64(e1l, e1h)
    p2 = _pack_u64(e2l, e2h)
    p3 = _pack_u64(e3l, e3h)
    t0 = gl.expand_dims(gl.convert_layout(s0, SE, assert_trivial=True), 1)
    t1 = gl.expand_dims(gl.convert_layout(s1, SE, assert_trivial=True), 1)
    t2 = gl.expand_dims(gl.convert_layout(s2, SE, assert_trivial=True), 1)
    t3 = gl.expand_dims(gl.convert_layout(s3, SE, assert_trivial=True), 1)
    HZS: gl.constexpr = (
        "s_nop 7\n\ts_nop 3\n\t"
        if HZ == 1 or HZ == 12
        else "s_nop 7\n\t"
        if HZ == 8
        else "s_nop 3\n\t"
        if HZ == 4
        else ""
    )
    if EPI >= 1:
        s01 = _pack_u64(t0, t1)
        s23 = _pack_u64(t2, t3)
        HZP: gl.constexpr = "s_nop 7\n\t" if HZ == 1 else ""
        pp = gl.inline_asm_elementwise(
            HZP
            + "v_pk_mul_f32 $0, $1, $4 op_sel_hi:[1,0]\n\tv_pk_fma_f32 $0, $2, $4, $0 op_sel:[0,1,0] op_sel_hi:[1,1,1]\n\tv_pk_fma_f32 $0, $3, $5, $0 op_sel_hi:[1,0,1]",
            "=&v,v,v,v,v,v",
            [p0, p1, p2, s01, s23],
            dtype=gl.uint64,
            is_pure=True,
            pack=1,
        )
        if not SLBX:
            bsc = gl.zeros([BM, BN // 2], gl.uint64, LE) | b_scale.to(
                gl.uint32, bitcast=True
            ).to(gl.uint64)
        if SLBX and EPI >= 2:
            pe = gl.inline_asm_elementwise(
                HZS
                + "v_pk_fma_f32 $0, $2, $6, $4 op_sel:[0,1,0] op_sel_hi:[1,1,1]\n\tv_pk_fma_f32 $1, $3, $7, $5 op_sel:[0,1,0] op_sel_hi:[1,1,1]\n\ts_waitcnt lgkmcnt(0)\n\tv_pk_mul_f32 $0, $0, $8 op_sel_hi:[1,0]\n\tv_pk_mul_f32 $1, $1, $9 op_sel_hi:[1,0]",
                "=&v,=&v,v,v,v,v,v,v,s,s",
                [p3, pp, s23, bs64],
                dtype=gl.uint64,
                is_pure=True,
                pack=2,
            )
        else:
            pe = gl.inline_asm_elementwise(
                HZS
                + "v_pk_fma_f32 $0, $2, $6, $4 op_sel:[0,1,0] op_sel_hi:[1,1,1]\n\tv_pk_fma_f32 $1, $3, $7, $5 op_sel:[0,1,0] op_sel_hi:[1,1,1]\n\tv_pk_mul_f32 $0, $0, $8 op_sel_hi:[1,0]\n\tv_pk_mul_f32 $1, $1, $9 op_sel_hi:[1,0]",
                "=&v,=&v,v,v,v,v,v,v,v,v",
                [p3, pp, s23, bsc],
                dtype=gl.uint64,
                is_pure=True,
                pack=2,
            )
    elif HZ >= 32:
        HZM: gl.constexpr = (
            "s_nop 0\n\t"
            if HZ == 33
            else "s_nop 1\n\t"
            if HZ == 34
            else "s_nop 2\n\t"
            if HZ == 35
            else "s_nop 3\n\t"
            if HZ == 36
            else "s_nop 4\n\t"
            if HZ == 37
            else "s_nop 5\n\t"
            if HZ == 38
            else "s_nop 6\n\t"
            if HZ == 39
            else "s_nop 7\n\t"
            if HZ == 40
            else ""
        )
        bsz = gl.zeros([BM, BN // 2], gl.uint64, LE) | b_scale.to(
            gl.uint32, bitcast=True
        ).to(gl.uint64)
        pe = gl.inline_asm_elementwise(
            "v_pk_mul_f32 $0, $2, $10 op_sel_hi:[1,0]\n\tv_pk_mul_f32 $1, $3, $11 op_sel_hi:[1,0]\n\tv_pk_fma_f32 $0, $4, $10, $0 op_sel:[0,1,0] op_sel_hi:[1,1,1]\n\tv_pk_fma_f32 $1, $5, $11, $1 op_sel:[0,1,0] op_sel_hi:[1,1,1]\n\tv_pk_fma_f32 $0, $6, $12, $0 op_sel_hi:[1,0,1]\n\tv_pk_fma_f32 $1, $7, $13, $1 op_sel_hi:[1,0,1]\n\t"
            + HZM
            + "v_pk_fma_f32 $0, $8, $12, $0 op_sel:[0,1,0] op_sel_hi:[1,1,1]\n\tv_pk_fma_f32 $1, $9, $13, $1 op_sel:[0,1,0] op_sel_hi:[1,1,1]\n\tv_pk_mul_f32 $0, $0, $14 op_sel_hi:[1,0]\n\tv_pk_mul_f32 $1, $1, $15 op_sel_hi:[1,0]",
            "=&v,=&v,v,v,v,v,v,v,v,v,v,v,v,v,v,v",
            [p0, p1, p2, p3, _pack_u64(t0, t1), _pack_u64(t2, t3), bsz],
            dtype=gl.uint64,
            is_pure=True,
            pack=2,
        )
    else:
        pe = gl.inline_asm_elementwise(
            HZS
            + "v_pk_mul_f32 $0, $1, $5 op_sel_hi:[1,0]\n\tv_pk_fma_f32 $0, $2, $5, $0 op_sel:[0,1,0] op_sel_hi:[1,1,1]\n\tv_pk_fma_f32 $0, $3, $6, $0 op_sel_hi:[1,0,1]\n\tv_pk_fma_f32 $0, $4, $6, $0 op_sel:[0,1,0] op_sel_hi:[1,1,1]\n\tv_pk_mul_f32 $0, $0, $7 op_sel_hi:[1,0]",
            "=&v,v,v,v,v,v,v,v",
            [
                p0,
                p1,
                p2,
                p3,
                _pack_u64(t0, t1),
                _pack_u64(t2, t3),
                gl.zeros([BM, BN // 2], gl.uint64, LE)
                | b_scale.to(gl.uint32, bitcast=True).to(gl.uint64),
            ],
            dtype=gl.uint64,
            is_pure=True,
            pack=1,
        )
    c_lo = (pe & 4294967295).to(gl.uint32).to(gl.float32, bitcast=True)
    c_hi = (pe >> 32).to(gl.uint32).to(gl.float32, bitcast=True)
    c = gl.convert_layout(
        gl.reshape(gl.join(c_lo, c_hi), [BM, BN]), MMA, assert_trivial=True
    ).to(gl.bfloat16)
    if HB & 4 != 0 and TAIL > 0:
        c_base = c_ptr
    else:
        c_base = c_ptr + batch_id * CS_CB + row0 * CS_CM + pid_n * BN
    if TAIL > 0:
        cdna4.buffer_store(c, c_base, offE, cache=".wt" if STC == 1 else None)
    else:
        rc = gl.arange(0, BM, layout=gl.SliceLayout(1, MMA))
        nc = gl.arange(0, BN, layout=gl.SliceLayout(0, MMA))
        c_base = c_ptr + batch_id * CS_CB + row0 * CS_CM + pid_n * BN
        offc = gl.expand_dims(rc * CS_CM, 1) + gl.expand_dims(nc, 0)
        if EVEN_M:
            cmask = None
        else:
            cmask = gl.expand_dims(row0 + rc < Mr, 1)
        if STC == 1:
            cdna4.buffer_store(c, c_base, offc, mask=cmask, cache=".wt")
        else:
            cdna4.buffer_store(c, c_base, offc, mask=cmask)


_K = 512
_N = 256
_BLOCK_N = 32
_NUM_WARPS = 2
_WARPS_N = 2

# Exact-M launch rows, measured on MI355X; re-tune when the kernel changes.
# STC: write-through C stores. PRELOAD: kernarg dwords preloaded into SGPRs.
# BSE/BCM/HB/TAIL/SLB/QL/EPI/HZ: scale-load order, B cache modifier, uniform-base
# bitmask, epilogue addressing, scale load, A layout, packed epilogue, s_nop pad
# before the epilogue asm (each row's HZ gives its asm >= 12 wait states).
_ROWS = {
    6: dict(BM=8, STC=1, MC=6, EPI=2, HZ=4),
    12: dict(BM=8, STC=1, PRELOAD=8, TAIL=2, BCM=1, EPI=2, SLB=3, QL=2, HZ=4, HB=3),
    18: dict(
        BM=8,
        STC=1,
        PRELOAD=4,
        MC=18,
        BSE=1,
        TAIL=2,
        BCM=1,
        EPI=2,
        SLB=3,
        QL=2,
        HZ=4,
        HB=7,
    ),
    24: dict(BM=8, STC=1, PRELOAD=4, TAIL=2, BCM=1, EPI=2, SLB=3, QL=2, HZ=4),
    30: dict(BM=8, PRELOAD=4, MC=30, TAIL=2, BCM=1, EPI=2, SLB=3, QL=2, HZ=4, WPE=0),
    36: dict(BM=16, BSE=1, BCM=1, HZ=37),
    42: dict(BM=16, BSE=1, BCM=1, HZ=37),
}
_SMALL_M_ROW = dict(BM=8, STC=1)
_ROW = dict(BM=16)

_KNOB_LOCK = threading.Lock()


@contextlib.contextmanager
def _kernarg_preload(dwords: int):
    # Read by the compiler only when this launch compiles the variant. The knob is
    # process-global and exists only in the ROCm Triton build; upstream Triton has
    # no per-kernel equivalent, so elsewhere the compiler default applies.
    if not hasattr(knobs.amd, "kernarg_preload_max_dwords"):
        yield
        return
    with _KNOB_LOCK:
        saved = knobs.amd.kernarg_preload_max_dwords
        knobs.amd.kernarg_preload_max_dwords = dwords
        try:
            yield
        finally:
            knobs.amd.kernarg_preload_max_dwords = saved


def _served(h: int, m: int) -> bool:
    # 16 heads at 72 < M <= 96 and above 120: the aiter kernel was as fast.
    if h == 8:
        return 1 <= m <= 128
    return h == 16 and (1 <= m <= 72 or 97 <= m <= 120)


def gluon_mla_absorb_v_supported(
    x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor, out: torch.Tensor
) -> bool:
    if x.dim() != 3:
        return False
    m, h, k = x.shape
    return (
        _served(h, m)
        and k == _K
        and x.dtype == torch.bfloat16
        and x.stride(2) == 1
        and w.dtype == torch.float8_e4m3fn
        and tuple(w.shape) == (h, _N, _K)
        and w.is_contiguous()
        and isinstance(w_scale, torch.Tensor)
        and w_scale.numel() == 1
        and w_scale.dtype == torch.float32
        and out.dtype == torch.bfloat16
        and tuple(out.shape) == (m, h, _N)
        and out.stride(2) == 1
        # x and out are addressed with 32-bit buffer offsets.
        and x.stride(0) * m + x.stride(1) * h < (1 << 30)
        and out.stride(0) * m + out.stride(1) * h < (1 << 30)
    )


def gluon_mla_absorb_v(
    x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, h, _ = x.shape
    row = _ROWS.get(m, _SMALL_M_ROW if m < 16 else _ROW)
    bm = row["BM"]
    preload = row.get("PRELOAD", 2)
    grid = (h * triton.cdiv(m, bm) * (_N // _BLOCK_N),)
    with _kernarg_preload(preload):
        _mla_absorb_v_kernel[grid](
            x,
            w,
            out,
            w_scale,
            m,
            NUM_BATCH=h,
            N=_N,
            DTYPE_MAX=448.0,
            BM=bm,
            BN=_BLOCK_N,
            NW=_NUM_WARPS,
            WN=_WARPS_N,
            SINGLE_M=m <= bm,
            EVEN_M=m % bm == 0,
            PRELOAD_DW=preload,
            CS_AB=x.stride(1),
            CS_AM=x.stride(0),
            CS_BB=w.stride(0),
            CS_BN=w.stride(1),
            CS_CB=out.stride(1),
            CS_CM=out.stride(0),
            MC=row.get("MC", 0),
            STC=row.get("STC", 0),
            BSE=row.get("BSE", 0),
            TAIL=row.get("TAIL", 0),
            BCM=row.get("BCM", 0),
            EPI=row.get("EPI", 0),
            SLB=row.get("SLB", 0),
            QL=row.get("QL", 0),
            # LLVM does not pad XDL-write -> VALU-read for inline-asm readers; gfx950
            # needs 12 wait states and the default epilogue reads the last MFMA after 4.
            HZ=row.get("HZ", 8),
            HB=row.get("HB", 0),
            num_warps=_NUM_WARPS,
            waves_per_eu=row.get("WPE", 1),
        )
    return out
