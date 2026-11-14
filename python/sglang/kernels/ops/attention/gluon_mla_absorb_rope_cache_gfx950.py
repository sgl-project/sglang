"""Gluon MLA decode prologue for gfx950: q absorb + RoPE + fp8 KV-cache write.

One launch replaces aiter ``fused_fp8_bmm_rope_cat_and_cache_mla`` on the DSA
decode / target-verify path (fp8 ``w_kc``, kv_lora_rank 512, qk_nope 192,
rope 64, 8 or 16 local heads). The 1-D grid is split by program id: the first
M programs rope q_pe into ``q_cat[:, :, 512:]`` and write one KV-cache row each
(GPT-J / interleaved rotation); the rest run the absorb GEMM
``q_nope @ w_kc`` with per-128-group fp8 activation quant and store fp8 into
``q_cat[:, :, :512]``.
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

# amax over 16 lanes with DPP, clamped at 1e-10 (0x2edbe6ff).
_DPP1X16 = gl.constexpr(
    "s_nop 1\n"
    "v_max_f32_dpp $0, $1, $1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n"
    "s_nop 1\n"
    "v_max_f32_dpp $0, $0, $0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n"
    "s_nop 1\n"
    "v_max_f32_dpp $0, $0, $0 row_half_mirror row_mask:0xf bank_mask:0xf\n"
    "s_nop 1\n"
    "v_max_f32_dpp $0, $0, $0 row_ror:8 row_mask:0xf bank_mask:0xf\n"
    "v_max_f32_e32 $0, 0x2edbe6ff, $0"
)


@gluon.constexpr_function
def _log2(n):
    return int(n).bit_length() - 1


@gluon.constexpr_function
def _a1_spt(BM):
    return [1, 4] if BM == 8 else [1, 8]


@gluon.constexpr_function
def _a1_tpw(BM):
    return [4, 16] if BM == 8 else [8, 8]


@gluon.constexpr_function
def _a1_layout(BM, NW, AL):
    if AL == 2:
        return gl.BlockedLayout([1, 4], [4, 16], [NW, 1], [1, 0])
    return gl.BlockedLayout(_a1_spt(BM), _a1_tpw(BM), [NW, 1], [1, 0])


@gluon.constexpr_function
def _lin(n, nw):
    """Blocked layout tiling a 1-D [n] tile over nw*64 lanes, contiguous per lane."""
    ept = n // (64 * nw)
    if ept < 1:
        ept = 1
    return gl.BlockedLayout([ept], [64], [nw], [0])


@gluon.jit
def _group_amax1x16(x):
    return gl.inline_asm_elementwise(
        _DPP1X16, "=&v,v", [x], dtype=gl.float32, is_pure=True, pack=1
    )


@gluon.jit
def _rope_kv_segment(
    row,
    q_pe_ptr,
    k_nope_ptr,
    k_pe_ptr,
    pos_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    kv_ptr,
    slot_ptr,
    k_scale_ptr,
    q_pe_sb,
    q_pe_sh,
    k_nope_sb,
    k_pe_sb,
    cos_sb,
    cos_sd,
    q_out_sb,
    q_out_sh,
    kv_sb,
    QH: gl.constexpr,
    D_NOPE: gl.constexpr,
    D_PE: gl.constexpr,
    HAVE_K_SCALE: gl.constexpr,
    NW: gl.constexpr,
    KVC: gl.constexpr,
    RKV: gl.constexpr = 0,
):
    """One query row: all QH heads of q-rope, plus this row's KV-cache line.

    The GPT-J / interleaved rotation `rot[2j] = -x[2j+1]; rot[2j+1] = x[2j]` (the stock
    `_get_gptj_rotated_x_1D`, where->reshape(32,2)->flip(1)->reshape) is formed by a SECOND
    load of the same already-resident 128 B pe line at the partner index `d ^ 1`, so the
    body needs no cross-lane flip and stays layout-free for every M and NW.
    """
    LQ: gl.constexpr = _lin(QH * D_PE, NW)
    LN: gl.constexpr = _lin(D_NOPE, NW)
    LP: gl.constexpr = _lin(D_PE, NW)
    i = gl.arange(0, QH * D_PE, layout=LQ).to(gl.int64)
    h = i // D_PE
    d = i % D_PE
    dx = d + 1 - 2 * (d % 2)
    ev = d % 2 == 0
    if RKV >= 1:
        dn = gl.arange(0, D_NOPE, layout=LN).to(gl.int64)
        dp = gl.arange(0, D_PE, layout=LP).to(gl.int64)
        dpx = dp + 1 - 2 * (dp % 2)
        evp = dp % 2 == 0
        pos = gl.load(pos_ptr + row)
        slot = gl.load(slot_ptr + row).to(gl.int64)
        qb = q_pe_ptr + row * q_pe_sb + h * q_pe_sh
        x = gl.load(qb + d)
        xr = gl.load(qb + dx)
        kn = gl.load(k_nope_ptr + row * k_nope_sb + dn)
        kb = k_pe_ptr + row * k_pe_sb
        y = gl.load(kb + dp)
        yr = gl.load(kb + dpx)
        if HAVE_K_SCALE:
            ks = gl.load(k_scale_ptr)
        else:
            ks = 1.0
        cq = gl.load(cos_ptr + pos * cos_sb + d // 2 * cos_sd)
        sq = gl.load(sin_ptr + pos * cos_sb + d // 2 * cos_sd)
        ck = gl.load(cos_ptr + pos * cos_sb + dp // 2 * cos_sd)
        sk = gl.load(sin_ptr + pos * cos_sb + dp // 2 * cos_sd)
        rcp = (1.0 / ks).to(gl.float32)
        qpe = x * cq + gl.where(ev, -xr, xr) * sq
        gl.store(
            q_out_ptr + row * q_out_sb + h * q_out_sh + (d + D_NOPE),
            qpe.to(gl.float8e4nv),
            cache_modifier=KVC,
        )
        kpe = y * ck + gl.where(evp, -yr, yr) * sk
        okv = slot >= 0
        kvo = kv_ptr + gl.maximum(slot, 0) * kv_sb
        gl.store(
            kvo + dn,
            (kn.to(gl.float32) * rcp).to(gl.float8e4nv),
            mask=(dn >= 0) & okv,
            cache_modifier=KVC,
        )
        gl.store(
            kvo + (dp + D_NOPE),
            (kpe.to(gl.float32) * rcp).to(gl.float8e4nv),
            mask=(dp >= 0) & okv,
            cache_modifier=KVC,
        )
    else:
        pos = gl.load(pos_ptr + row)
        cq = gl.load(cos_ptr + pos * cos_sb + d // 2 * cos_sd)
        sq = gl.load(sin_ptr + pos * cos_sb + d // 2 * cos_sd)
        qb = q_pe_ptr + row * q_pe_sb + h * q_pe_sh
        x = gl.load(qb + d)
        xr = gl.load(qb + dx)
        qpe = x * cq + gl.where(ev, -xr, xr) * sq
        gl.store(
            q_out_ptr + row * q_out_sb + h * q_out_sh + (d + D_NOPE),
            qpe.to(gl.float8e4nv),
            cache_modifier=KVC,
        )
        slot = gl.load(slot_ptr + row).to(gl.int64)
        if slot >= 0:
            if HAVE_K_SCALE:
                ks = gl.load(k_scale_ptr)
            else:
                ks = 1.0
            rcp = (1.0 / ks).to(gl.float32)
            dn = gl.arange(0, D_NOPE, layout=LN).to(gl.int64)
            dp = gl.arange(0, D_PE, layout=LP).to(gl.int64)
            dpx = dp + 1 - 2 * (dp % 2)
            evp = dp % 2 == 0
            ck = gl.load(cos_ptr + pos * cos_sb + dp // 2 * cos_sd)
            sk = gl.load(sin_ptr + pos * cos_sb + dp // 2 * cos_sd)
            kn = gl.load(k_nope_ptr + row * k_nope_sb + dn)
            kb = k_pe_ptr + row * k_pe_sb
            y = gl.load(kb + dp)
            yr = gl.load(kb + dpx)
            kpe = y * ck + gl.where(evp, -yr, yr) * sk
            kvo = kv_ptr + slot * kv_sb
            gl.store(
                kvo + dn,
                (kn.to(gl.float32) * rcp).to(gl.float8e4nv),
                cache_modifier=KVC,
            )
            gl.store(
                kvo + (dp + D_NOPE),
                (kpe.to(gl.float32) * rcp).to(gl.float8e4nv),
                cache_modifier=KVC,
            )


@gluon.jit(do_not_specialize=["M", "N_GEMM"])
def _mla_absorb_rope_cache_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    M,
    q_pe_ptr,
    k_nope_ptr,
    k_pe_ptr,
    pos_ptr,
    cos_ptr,
    sin_ptr,
    kv_ptr,
    slot_ptr,
    k_scale_ptr,
    q_pe_sb,
    q_pe_sh,
    k_nope_sb,
    k_pe_sb,
    cos_sb,
    cos_sd,
    kv_sb,
    CS_CB: gl.constexpr,
    D_PE: gl.constexpr,
    HAVE_K_SCALE: gl.constexpr,
    KVC: gl.constexpr,
    N_GEMM,
    NUM_BATCH: gl.constexpr,
    N: gl.constexpr,
    DTYPE_MAX: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    NW: gl.constexpr,
    SINGLE_M: gl.constexpr,
    EVEN_M: gl.constexpr,
    CS_AB: gl.constexpr,
    CS_AM: gl.constexpr,
    CS_BB: gl.constexpr,
    CS_BN: gl.constexpr,
    CS_CM: gl.constexpr,
    MC: gl.constexpr = 0,
    RKV: gl.constexpr = 0,
    TAIL: gl.constexpr = 0,
    AL: gl.constexpr = 0,
    HB: gl.constexpr = 0,
):
    if MC > 0:
        Mr = MC
    else:
        Mr = M
    BK: gl.constexpr = 128
    TK: gl.constexpr = 64
    MMA0: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 128], True, [1, NW])
    MMA1: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [1, NW])
    DA0: gl.constexpr = gl.DotOperandLayout(0, MMA0, 16)
    DB0: gl.constexpr = gl.DotOperandLayout(1, MMA0, 16)
    DA1: gl.constexpr = gl.DotOperandLayout(0, MMA1, 8)
    DB1: gl.constexpr = gl.DotOperandLayout(1, MMA1, 8)
    LA0: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [NW, 1], [1, 0])
    LA1: gl.constexpr = _a1_layout(BM, NW, AL)
    SH_A0: gl.constexpr = gl.SwizzledSharedLayout(16, 2, 8, [1, 0])
    SH_A1: gl.constexpr = gl.SwizzledSharedLayout(8, 4, 4, [1, 0])
    SH_S: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    gid = gl.program_id(0)
    is_rope = gid < M
    rope_pid = gid
    gemm_pid = gid - M
    if is_rope:
        _rope_kv_segment(
            rope_pid,
            q_pe_ptr,
            k_nope_ptr,
            k_pe_ptr,
            pos_ptr,
            cos_ptr,
            sin_ptr,
            c_ptr,
            kv_ptr,
            slot_ptr,
            k_scale_ptr,
            q_pe_sb,
            q_pe_sh,
            k_nope_sb,
            k_pe_sb,
            cos_sb,
            cos_sd,
            CS_CM,
            CS_CB,
            kv_sb,
            NUM_BATCH,
            N,
            D_PE,
            HAVE_K_SCALE,
            NW,
            KVC,
            RKV,
        )
    else:
        gid = gemm_pid
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
        if HB & 1 != 0:
            a_base = a_ptr
            a_uo = batch_id * CS_AB + row0 * CS_AM
        else:
            a_base = a_ptr + batch_id * CS_AB + row0 * CS_AM
            a_uo = 0
        if AL == 5:
            LAC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [2, 2], [1, 0])
            rac = gl.arange(0, BM, layout=gl.SliceLayout(1, LAC))
            kac = gl.arange(0, 256, layout=gl.SliceLayout(0, LAC))
            offac = gl.expand_dims(rac * CS_AM + a_uo, 1) + gl.expand_dims(kac, 0)
            mac = gl.expand_dims(kac < BK + TK, 0)
            if not EVEN_M:
                mac = mac & gl.expand_dims(row0 + rac < Mr, 1)
            ac = cdna4.buffer_load(a_base, offac, mask=mac)
        ra0 = gl.arange(0, BM, layout=gl.SliceLayout(1, LA0))
        ka0 = gl.arange(0, BK, layout=gl.SliceLayout(0, LA0))
        offa0 = gl.expand_dims(ra0 * CS_AM + a_uo, 1) + gl.expand_dims(ka0, 0)
        ra1 = gl.arange(0, BM, layout=gl.SliceLayout(1, LA1))
        ka1 = gl.arange(0, TK, layout=gl.SliceLayout(0, LA1))
        offa1 = gl.expand_dims(ra1 * CS_AM + a_uo, 1) + gl.expand_dims(ka1 + BK, 0)
        if AL == 5:
            pass
        elif EVEN_M:
            a0 = cdna4.buffer_load(a_base, offa0)
        else:
            m0k = gl.expand_dims(row0 + ra0 < Mr, 1)
            a0 = cdna4.buffer_load(a_base, offa0, mask=m0k)
        if HB & 2 != 0:
            b_base = b_ptr
            b_uo = batch_id * CS_BB + pid_n * (BN * CS_BN)
        else:
            b_base = b_ptr + batch_id * CS_BB + pid_n * (BN * CS_BN)
            b_uo = 0
        kb0 = gl.arange(0, BK, layout=gl.SliceLayout(1, DB0))
        nb0 = gl.arange(0, BN, layout=gl.SliceLayout(0, DB0))
        offb0 = gl.expand_dims(kb0, 1) + gl.expand_dims(nb0 * CS_BN + b_uo, 0)
        b0 = cdna4.buffer_load(b_base, offb0, cache=".cg")
        if AL == 5:
            pass
        elif EVEN_M:
            a1 = cdna4.buffer_load(a_base, offa1)
        else:
            m1k = gl.expand_dims(row0 + ra1 < Mr, 1)
            a1 = cdna4.buffer_load(a_base, offa1, mask=m1k)
        kb1 = gl.arange(0, TK, layout=gl.SliceLayout(1, DB1))
        nb1 = gl.arange(0, BN, layout=gl.SliceLayout(0, DB1))
        offb1 = gl.expand_dims(kb1 + BK, 1) + gl.expand_dims(nb1 * CS_BN + b_uo, 0)
        b1 = cdna4.buffer_load(b_base, offb1, cache=".cg")
        b_scale = gl.load(b_scale_ptr)
        if TAIL > 0:
            r_epi = gl.arange(0, BM, layout=gl.SliceLayout(1, MMA1))
            n_epi = gl.arange(0, BN, layout=gl.SliceLayout(0, MMA1))
            offE = gl.expand_dims(r_epi * CS_CM, 1) + gl.expand_dims(n_epi, 0)
            if not EVEN_M:
                maskE = gl.expand_dims(row0 + r_epi < Mr, 1) & (
                    gl.expand_dims(n_epi, 0) >= 0
                )
            else:
                maskE = None
            offE = gl.inline_asm_elementwise(
                "", "=v,0", [offE], dtype=gl.int32, is_pure=TAIL == 1, pack=1
            )
            offE = gl.max_contiguous(gl.multiple_of(offE, [1, 4]), [1, 4])
        if AL == 5:
            L4: gl.constexpr = gl.BlockedLayout(
                [1, 1, 1, 8], [4, 1, 16, 1], [2, 2, 1, 1], [3, 2, 1, 0]
            )
            acf = gl.convert_layout(
                gl.reshape(ac.to(gl.float32), [BM, 2, 16, 8]), L4, assert_trivial=True
            )
            pc = gl.max(gl.abs(acf), axis=3)
            mc = _group_amax1x16(pc)
            gc = mc * (1.0 / DTYPE_MAX)
            rc = fast_dividef(1.0, gc)
            qc = gl.convert_layout(
                gl.reshape((acf * gl.expand_dims(rc, 3)).to(gl.float8e4nv), [BM, 256]),
                LAC,
                assert_trivial=True,
            )
            ssc = gl.allocate_shared_memory(
                gl.float32,
                [32 * BM],
                SH_S,
                gl.reshape(gl.permute(gc, [1, 2, 0]), [32 * BM]),
            )
            sqc = gl.allocate_shared_memory(gl.float8e4nv, [BM, 256], SH_A0, qc)
            qa0 = sqc.slice(0, BK, dim=1).load(DA0)
            qa1 = sqc.slice(BK, TK, dim=1).load(DA1)
            sc0 = ssc.slice(0, BM).load(gl.SliceLayout(1, MMA0))
            sc1 = ssc.slice(16 * BM, BM).load(gl.SliceLayout(1, MMA1))
        else:
            a0f = a0.to(gl.float32)
            a1f = a1.to(gl.float32)
            m0 = gl.maximum(gl.max(gl.abs(a0f), axis=1), 1e-10)
            m1 = gl.maximum(gl.max(gl.abs(a1f), axis=1), 1e-10)
            s0 = m0 * (1.0 / DTYPE_MAX)
            s1 = m1 * (1.0 / DTYPE_MAX)
            r0 = fast_dividef(1.0, s0)
            r1 = fast_dividef(1.0, s1)
            q0 = (a0f * gl.expand_dims(r0, 1)).to(gl.float8e4nv)
            q1 = (a1f * gl.expand_dims(r1, 1)).to(gl.float8e4nv)
            sq0 = gl.allocate_shared_memory(gl.float8e4nv, [BM, BK], SH_A0, q0)
            sq1 = gl.allocate_shared_memory(gl.float8e4nv, [BM, TK], SH_A1, q1)
            ss0 = gl.allocate_shared_memory(gl.float32, [BM], SH_S, s0)
            ss1 = gl.allocate_shared_memory(gl.float32, [BM], SH_S, s1)
            qa0 = sq0.load(DA0)
            qa1 = sq1.load(DA1)
            sc0 = ss0.load(gl.SliceLayout(1, MMA0))
            sc1 = ss1.load(gl.SliceLayout(1, MMA1))
        zero0 = gl.zeros([BM, BN], gl.float32, MMA0)
        d0 = cdna4.mfma_scaled(qa0, None, "e4m3", b0, None, "e4m3", zero0)
        acc = gl.zeros([BM, BN], gl.float32, MMA0)
        acc = d0 * gl.expand_dims(sc0, 1) + acc
        acc = gl.convert_layout(acc, MMA1, assert_trivial=True)
        zero1 = gl.zeros([BM, BN], gl.float32, MMA1)
        d1 = cdna4.mfma(qa1, b1, zero1)
        acc = acc + d1 * gl.expand_dims(sc1, 1)
        acc = acc * b_scale
        c = acc.to(gl.bfloat16).to(gl.float8e4nv)
        if TAIL > 0:
            c_baseE = c_ptr + batch_id * CS_CB + row0 * CS_CM + pid_n * BN
            cdna4.buffer_store(c, c_baseE, offE, mask=maskE, cache=".wt")
        else:
            rc = gl.arange(0, BM, layout=gl.SliceLayout(1, MMA1))
            nc = gl.arange(0, BN, layout=gl.SliceLayout(0, MMA1))
            c_base = c_ptr + batch_id * CS_CB + row0 * CS_CM + pid_n * BN
            offc = gl.expand_dims(rc * CS_CM, 1) + gl.expand_dims(nc, 0)
            if EVEN_M:
                cmask = None
            else:
                cmask = gl.expand_dims(row0 + rc < Mr, 1)
            cdna4.buffer_store(c, c_base, offc, mask=cmask, cache=".wt")


_D_NOPE = 192
_D_PE = 64
_KV_LORA = 512
_BLOCK_N = 64
_NUM_WARPS = 4
# Kernarg dwords preloaded into SGPRs; measured on MI355X.
_PRELOAD_DWORDS = 2
# Largest M served per local head count.
_MAX_M = {8: 128, 16: 384}

# Exact-M launch rows, measured on MI355X; re-tune when the kernel changes.
# BM: GEMM rows per CTA. AL/HB/TAIL: A-operand layout, uniform-base address
# bitmask and epilogue address form. RKV: branchless rope/KV body.
# KVC: cache modifier of the q_cat / KV stores.
_ROWS = {
    6: dict(BM=16, RKV=1, TAIL=2),
    12: dict(BM=8, RKV=1, TAIL=1, AL=5, HB=3),
    18: dict(BM=8, RKV=1, TAIL=1, AL=5, HB=1),
    24: dict(BM=8, RKV=1, TAIL=1, AL=5, HB=1),
    30: dict(BM=16, TAIL=1, AL=2, KVC=".cs"),
    36: dict(BM=16, TAIL=1, AL=2),
    42: dict(BM=16, TAIL=1, AL=2, KVC=".cs"),
}
_GENERIC_ROW = dict(BM=16)


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


def gluon_mla_absorb_rope_cache_supported(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    w_kc: torch.Tensor,
    w_scale: torch.Tensor,
    cos_cache: torch.Tensor,
    k_scale: torch.Tensor | None,
) -> bool:
    if q_nope.dim() != 3:
        return False
    m, h, k = q_nope.shape
    return (
        h in _MAX_M
        and 1 <= m <= _MAX_M[h]
        and k == _D_NOPE
        and q_nope.dtype == torch.bfloat16
        and q_nope.stride(2) == 1
        and w_kc.dtype == torch.float8_e4m3fn
        and tuple(w_kc.shape) == (h, _KV_LORA, _D_NOPE)
        and w_kc.is_contiguous()
        and isinstance(w_scale, torch.Tensor)
        and w_scale.numel() == 1
        and w_scale.dtype == torch.float32
        and tuple(q_pe.shape) == (m, h, _D_PE)
        and q_pe.stride(2) == 1
        and tuple(k_nope.shape) == (m, 1, _KV_LORA)
        and k_nope.stride(2) == 1
        and tuple(k_pe.shape) == (m, 1, _D_PE)
        and k_pe.stride(2) == 1
        and kv_cache.dtype == torch.float8_e4m3fn
        and kv_cache.shape[1:] == (1, _KV_LORA + _D_PE)
        and kv_cache.stride(2) == 1
        and cos_cache.shape[-1] == _D_PE // 2
        and (k_scale is None or k_scale.numel() == 1)
        # q_nope and q_cat are addressed with 32-bit buffer offsets.
        and q_nope.stride(0) * m + q_nope.stride(1) * h < (1 << 30)
        and m * h * (_KV_LORA + _D_PE) < (1 << 30)
    )


def gluon_mla_absorb_rope_cache(
    q_nope: torch.Tensor,
    w_kc: torch.Tensor,
    w_scale: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    k_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Returns q_cat [M, H, 576] fp8 and writes kv_cache[slot_mapping]."""
    m, h, _ = q_nope.shape
    row = _ROWS.get(m, _GENERIC_ROW)
    bm = row["BM"]
    q_cat = torch.empty(
        (m, h, _KV_LORA + _D_PE), dtype=kv_cache.dtype, device=q_nope.device
    )
    n_gemm = h * triton.cdiv(m, bm) * (_KV_LORA // _BLOCK_N)
    with _kernarg_preload(_PRELOAD_DWORDS):
        _mla_absorb_rope_cache_kernel[(m + n_gemm,)](
            q_nope,
            w_kc,
            q_cat,
            w_scale,
            m,
            q_pe,
            k_nope,
            k_pe,
            positions,
            cos_cache,
            sin_cache,
            kv_cache,
            slot_mapping,
            k_scale,
            q_pe.stride(0),
            q_pe.stride(1),
            k_nope.stride(0),
            k_pe.stride(0),
            cos_cache.stride(0),
            cos_cache.stride(-1),
            kv_cache.stride(0),
            N_GEMM=n_gemm,
            CS_CB=q_cat.stride(1),
            D_PE=_D_PE,
            HAVE_K_SCALE=k_scale is not None,
            KVC=row.get("KVC", ""),
            NUM_BATCH=h,
            N=_KV_LORA,
            DTYPE_MAX=448.0,
            BM=bm,
            BN=_BLOCK_N,
            NW=_NUM_WARPS,
            SINGLE_M=m <= bm,
            EVEN_M=m % bm == 0,
            CS_AB=q_nope.stride(1),
            CS_AM=q_nope.stride(0),
            CS_BB=w_kc.stride(0),
            CS_BN=w_kc.stride(1),
            CS_CM=q_cat.stride(0),
            MC=m if m in _ROWS else 0,
            RKV=row.get("RKV", 0),
            TAIL=row.get("TAIL", 0),
            AL=row.get("AL", 0),
            HB=row.get("HB", 0),
            num_warps=_NUM_WARPS,
            waves_per_eu=0,
        )
    return q_cat
