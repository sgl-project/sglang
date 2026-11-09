# SPDX-License-Identifier: Apache-2.0
"""PTX / nvvm helpers for the Rubin build of the UMMA ReplaySSM kernel
(``gdn_replay_mtp_umma_rubin.py``): ``_umma_helpers`` plus a non-``.aligned``
named-barrier sync, L2 bulk prefetches, predicated global loads, a predicated
mbarrier arrive, the packed fp16 x fp32 FMA (FHFMA2, sm_107a) and the persistent
kernel's work-claim counter.

The common part, as in ``_umma_helpers`` (PTX / nvvm helpers for the UMMA
(tcgen05 / TMEM) ReplaySSM kernels):

Thin ``@dsl_user_op`` wrappers the kernels need and the CuTe DSL does not expose
directly: TMEM allocation and 32x32b loads / stores, kind::f16 ``tcgen05.mma``
with SMEM / TMEM operands and their descriptors, TMA tile and 1-D bulk copies,
predicated vector loads and stores, packed fp16 / bf16 conversions (including
``cvt.rs`` stochastic rounding), Philox4x32 and the LCG dither.
"""

from __future__ import annotations

import cutlass
from cutlass import Boolean, Float32, Int32, Int64, Uint32, Uint64, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op

# nvvm binding drift in cutlass-dsl 4.6+: Tcgen05GroupKind -> CTAGroupKind,
# tcgen05_commit_arrive -> tcgen05_commit (same members / signature).
_CTA1 = (getattr(nvvm, "Tcgen05GroupKind", None) or nvvm.CTAGroupKind).CTA_1
_tcgen05_commit = getattr(nvvm, "tcgen05_commit_arrive", None) or nvvm.tcgen05_commit
_SHAPE_32X32B = nvvm.Tcgen05LdStShape.SHAPE_32X32B


def _tmem_ptr(addr, *, loc=None, ip=None):
    ptr_ty = llvm.PointerType.get(cute.AddressSpace.tmem.value)
    return llvm.inttoptr(ptr_ty, Int32(addr).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)


# ----------------------------------------------------------------------------- TMEM
@dsl_user_op
def tmem_alloc(taddr: cute.Pointer, ncols: int, *, loc=None, ip=None) -> None:
    nvvm.tcgen05_alloc(
        taddr.to_llvm_ptr(loc=loc, ip=ip),
        Uint32(ncols).ir_value(loc=loc, ip=ip),
        group=_CTA1,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tmem_dealloc(tmem_addr: Int32, ncols: int, *, loc=None, ip=None) -> None:
    nvvm.tcgen05_dealloc(
        _tmem_ptr(tmem_addr, loc=loc, ip=ip),
        Int32(ncols).ir_value(loc=loc, ip=ip),
        group=_CTA1,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tmem_ld(row, col, num: int, *, loc=None, ip=None):
    """tcgen05.ld.32x32b.x{num}: ``num`` consecutive fp32 columns of this warp's 32 lanes."""
    ptr = _tmem_ptr((Int32(row) << Int32(16)) | Int32(col), loc=loc, ip=ip)
    if num == 1:
        reg = nvvm.tcgen05_ld(Int32.mlir_type, _SHAPE_32X32B, ptr, loc=loc, ip=ip)
        return [Float32(llvm.bitcast(Float32.mlir_type, reg, loc=loc, ip=ip))]
    vi = ir.VectorType.get([num], Int32.mlir_type, loc=loc)
    vf = ir.VectorType.get([num], Float32.mlir_type, loc=loc)
    regs = nvvm.tcgen05_ld(vi, _SHAPE_32X32B, ptr, loc=loc, ip=ip)
    return cute.TensorSSA(llvm.bitcast(vf, regs, loc=loc, ip=ip), (num,), Float32)


@dsl_user_op
def tmem_st_u32(row, col, words, *, loc=None, ip=None) -> None:
    """tcgen05.st.32x32b.x{len(words)} of raw 32-bit words."""
    ptr = _tmem_ptr((Int32(row) << Int32(16)) | Int32(col), loc=loc, ip=ip)
    n = len(words)
    if n == 1:
        nvvm.tcgen05_st(
            _SHAPE_32X32B,
            ptr,
            Uint32(words[0]).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        return
    vi = ir.VectorType.get([n], Int32.mlir_type, loc=loc)
    vec = vector.from_elements(
        vi, [Uint32(w).ir_value(loc=loc, ip=ip) for w in words], loc=loc, ip=ip
    )
    nvvm.tcgen05_st(_SHAPE_32X32B, ptr, vec, loc=loc, ip=ip)


@dsl_user_op
def tmem_wait_ld(*, loc=None, ip=None):
    nvvm.tcgen05_wait(nvvm.Tcgen05WaitKind.LOAD, loc=loc, ip=ip)


@dsl_user_op
def tmem_wait_st(*, loc=None, ip=None):
    nvvm.tcgen05_wait(nvvm.Tcgen05WaitKind.STORE, loc=loc, ip=ip)


@dsl_user_op
def fence_after_sync(*, loc=None, ip=None):
    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def fence_before_sync(*, loc=None, ip=None):
    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC, loc=loc, ip=ip)


# ----------------------------------------------------------------------------- named barriers
@dsl_user_op
def bar_sync_ua(barrier_id: int, number_of_threads: int, *, loc=None, ip=None) -> None:
    """Named-barrier sync without ``.aligned`` (``barrier.cta.sync``), for a barrier whose
    participants reach it at different instructions (another sync site, or arrives elsewhere).
    ``cute.arch.barrier`` emits ``bar.sync`` = ``barrier.cta.sync.aligned``, which promises that
    every participating thread executes that same instruction; compute-sanitizer synccheck
    reports a two-site aligned barrier as "Divergent thread(s) in block" (and the error action
    ends the launch). Same SASS opcode (BAR.SYNC); the cubin lists the site as non-aligned.
    The arrive side is ``cute.arch.barrier_arrive(..., aligned=False)`` (``barrier.cta.arrive``)."""
    nvvm.barrier_cta_sync(
        Int32(barrier_id).ir_value(loc=loc, ip=ip),
        thread_count=Int32(number_of_threads).ir_value(loc=loc, ip=ip),
        aligned=False,
        loc=loc,
        ip=ip,
    )


# ----------------------------------------------------------------------------- UMMA
def f16_idesc(m: int, n: int, *, a_mn_major: bool = False, b_mn_major: bool = False):
    """kind::f16 instruction descriptor: fp16 A/B, fp32 D, K-major unless told otherwise."""
    return Uint32(
        (1 << 4)
        | (int(a_mn_major) << 15)
        | (int(b_mn_major) << 16)
        | ((n >> 3) << 17)
        | ((m >> 4) << 24)
    )


def sdesc_sw128(lbo: int):
    """SMEM descriptor template, 128B swizzle, K-major, SBO = 8 rows x 128 B (OR in addr >> 4)."""
    return Uint64(((lbo >> 4) << 16) | ((1024 >> 4) << 32) | (1 << 46) | (2 << 61))


def sdesc_sw32(base16):
    """SMEM descriptor for [N, 16] fp16 K-major tiles, 32B swizzle (SBO = 8 rows x 32 B).

    ``base16`` is the runtime smem address >> 4.  Mode 6 (bit 63 set) is added
    around the runtime operand: the DSL constant-folds pure-constant shifts and
    the folded value overflows a signed int64.
    """
    half_mode = Uint64(3 << 61)
    lo = Uint64((1 << 16) | (16 << 32) | (1 << 46))
    return ((base16 + half_mode) + half_mode) | lo


@dsl_user_op
def umma_ss(d_tmem, a_desc, b_desc, idesc, accumulate, *, loc=None, ip=None) -> None:
    """tcgen05.mma.cta_group::1.kind::f16, A and B from SMEM descriptors."""
    with cute.arch.elect_one():
        nvvm.tcgen05_mma(
            nvvm.Tcgen05MMAKind.F16,
            _CTA1,
            _tmem_ptr(d_tmem, loc=loc, ip=ip),
            Uint64(a_desc).ir_value(loc=loc, ip=ip),
            Uint64(b_desc).ir_value(loc=loc, ip=ip),
            Int32(idesc).ir_value(loc=loc, ip=ip),
            Boolean(accumulate).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def umma_ts(d_tmem, a_tmem, b_desc, idesc, accumulate, *, loc=None, ip=None) -> None:
    """tcgen05.mma.cta_group::1.kind::f16, A from TMEM, B from an SMEM descriptor."""
    with cute.arch.elect_one():
        nvvm.tcgen05_mma(
            nvvm.Tcgen05MMAKind.F16,
            _CTA1,
            _tmem_ptr(d_tmem, loc=loc, ip=ip),
            _tmem_ptr(a_tmem, loc=loc, ip=ip),
            Uint64(b_desc).ir_value(loc=loc, ip=ip),
            Int32(idesc).ir_value(loc=loc, ip=ip),
            Boolean(accumulate).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def umma_commit(mbar, *, loc=None, ip=None):
    with cute.arch.elect_one():
        _tcgen05_commit(mbar.to_llvm_ptr(loc=loc, ip=ip), group=_CTA1, loc=loc, ip=ip)


# ----------------------------------------------------------------------------- copies
def simple_tma_copy(atom, src, dst, mbar=None):
    """TMA tile copy of one (grouped) box; call WITHOUT elect_one."""
    if isinstance(atom.op, cpasync.CopyBulkTensorTileG2SOp):
        gmem, smem = src, dst
    else:
        smem, gmem = src, dst
    s_part, g_part = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(smem, 0),
        cute.group_modes(gmem, 0),
    )
    if isinstance(atom.op, cpasync.CopyBulkTensorTileG2SOp):
        cute.copy(atom, g_part, s_part, tma_bar_ptr=mbar)
    else:
        cute.copy(atom, s_part, g_part)


# ----------------------------------------------------------------------------- misc PTX
@dsl_user_op
def exit_if_negative(x: Int32, *, loc=None, ip=None) -> None:
    """CTA-uniform early exit without an MLIR control-flow edge (padded rows)."""
    llvm.inline_asm(
        None,
        [Int32(x).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.lt.s32 p, $0, 0;\n\t@p exit;\n\t}",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ldg_v4(addr: Int64, *, loc=None, ip=None):
    """ld.global.v4.b32 (16 B, aligned) -> four u32 words."""
    st = llvm.StructType.get_literal([T.i32()] * 4)
    r = llvm.inline_asm(
        st,
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def stg_u16(addr: Int64, w: Uint32, *, loc=None, ip=None) -> None:
    """st.global.u16 of the low half of ``w``."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tst.global.u16 [$0], h;\n\t}",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_bf16_16816(a0, a1, a2, a3, b0, b1, c, *, loc=None, ip=None):
    """mma.sync m16n8k16 row.col, f32 += bf16 x bf16.  a*: 4 u32, b*: 2 u32, c: 4 f32."""
    st = llvm.StructType.get_literal([T.f32()] * 4)
    ops = [Uint32(x).ir_value(loc=loc, ip=ip) for x in (a0, a1, a2, a3, b0, b1)]
    ops += [Float32(x).ir_value(loc=loc, ip=ip) for x in c]
    r = llvm.inline_asm(
        st,
        ops,
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def mma_f16_16816(a0, a1, a2, a3, b0, b1, c, *, loc=None, ip=None):
    """mma.sync m16n8k16 row.col, f32 += f16 x f16 (the fp16 k-ring Gram).  a*: 4 u32, b*: 2 u32,
    c: 4 f32; same fragment layout as ``mma_bf16_16816``."""
    st = llvm.StructType.get_literal([T.f32()] * 4)
    ops = [Uint32(x).ir_value(loc=loc, ip=ip) for x in (a0, a1, a2, a3, b0, b1)]
    ops += [Float32(x).ir_value(loc=loc, ip=ip) for x in c]
    r = llvm.inline_asm(
        st,
        ops,
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def pack_f16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """cvt.rn.f16x2.f32: ``lo`` in bits 0..15, ``hi`` in bits 16..31."""
    r = llvm.inline_asm(
        T.i32(),
        [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
        "cvt.rn.f16x2.f32 $0, $1, $2;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


@dsl_user_op
def pack_f16x2_rs(
    lo: Float32, hi: Float32, rbits: Uint32, *, loc=None, ip=None
) -> Uint32:
    """cvt.rs.f16x2.f32 (stochastic rounding, sm_100a / sm_103a): ``lo`` in bits 0..15
    rounded with ``rbits[12:0]``, ``hi`` in bits 16..31 with ``rbits[28:16]`` -- the same
    operand mapping as ``conversion::cvt_rs_f16x2_f32`` (include/flashinfer/mamba)."""
    r = llvm.inline_asm(
        T.i32(),
        [
            Float32(hi).ir_value(loc=loc, ip=ip),
            Float32(lo).ir_value(loc=loc, ip=ip),
            Uint32(rbits).ir_value(loc=loc, ip=ip),
        ],
        "cvt.rs.f16x2.f32 $0, $1, $2, $3;",
        "=r,f,f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


_PHILOX_ROUND_A = 0xD2511F53
_PHILOX_ROUND_B = 0xCD9E8D57
_PHILOX_KEY_A = 0x9E3779B9
_PHILOX_KEY_B = 0xBB67AE85
_PHILOX_MAX_UNROLL = 6


@cute.jit
def philox4x32_streams(
    seed_lo: Uint32,
    seed_hi: Uint32,
    off_lo0: Uint32,
    off_hi: Uint32,
    n_streams: cutlass.Constexpr,
    n_rounds: cutlass.Constexpr,
):
    """Philox4x32 for the ``n_streams`` (2 or 4) counters (off_lo0 + 8 s, off_hi, 0, 0), key
    (seed_lo, seed_hi): ``[s][i]`` uint32 draws. Each stream is bit-identical to
    ``conversion::philox_randint4x`` (include/flashinfer/mamba) / Triton's ``tl.randint4x``
    with an int64 offset; the caller guarantees off_lo0 + 8 (n_streams - 1) does not carry
    into off_hi.

    Up to ``_PHILOX_MAX_UNROLL`` rounds are unrolled, with rounds 1-2 specialised (exact):
    c2 = c3 = 0 and c1 = off_hi is shared by every stream, so round 1 needs one multiply per
    stream and leaves c0 = off_hi ^ k0, c1 = 0 shared, and round 2's c0 product is one
    multiply per call (2 of the 4 per-stream multiplies of rounds 1-2 disappear). More rounds
    run as a plain loop from round 1: unrolled -- or with the peeled rounds repeated at every
    call site -- they push the kernel past the instruction cache (B200, T = 8, B = 512, 100%
    fold, 10 rounds: unrolled 3x the looped cost, peeled +20%)."""
    if cutlass.const_expr(n_rounds > _PHILOX_MAX_UNROLL):
        return _philox_loop(seed_lo, seed_hi, off_lo0, off_hi, n_streams, n_rounds)
    m32 = Uint64(0xFFFFFFFF)
    k0, k1 = Uint32(seed_lo), Uint32(seed_hi)
    # round 1 (per stream: c0 * A only) and the shared half of round 2
    x0 = Uint32(off_hi) ^ k0  # c0 after round 1, every stream
    k0b, k1b = k0 + Uint32(_PHILOX_KEY_A), k1 + Uint32(_PHILOX_KEY_B)
    pa2 = Uint64(x0) * Uint64(_PHILOX_ROUND_A)
    y2 = Uint32(pa2 >> Uint64(32)) ^ k1b  # c2 after round 2 = y2 ^ c3_r1
    z2 = Uint32(pa2 & m32)  # c3 after round 2, every stream
    st = []
    for s_ in cutlass.range_constexpr(n_streams):
        pa = Uint64(Uint32(off_lo0) + Uint32(8 * s_)) * Uint64(_PHILOX_ROUND_A)
        c2 = Uint32(pa >> Uint64(32)) ^ k1  # round 1: c1 = 0, c3 = lo(pa)
        pb = Uint64(c2) * Uint64(_PHILOX_ROUND_B)  # round 2, per stream
        st.append(
            [
                Uint32(pb >> Uint64(32)) ^ k0b,
                Uint32(pb & m32),
                y2 ^ Uint32(pa & m32),
                z2,
            ]
        )
    k0 = k0b + Uint32(_PHILOX_KEY_A)
    k1 = k1b + Uint32(_PHILOX_KEY_B)
    rest = n_rounds - 2
    unroll = rest
    if cutlass.const_expr(n_streams == 4):
        a0, b0, d0, e0 = st[0]
        a1, b1, d1, e1 = st[1]
        a2, b2, d2, e2 = st[2]
        a3, b3, d3, e3 = st[3]
        for _ in cutlass.range(rest, unroll=unroll):
            a0, b0, d0, e0 = _philox_round(a0, b0, d0, e0, k0, k1)
            a1, b1, d1, e1 = _philox_round(a1, b1, d1, e1, k0, k1)
            a2, b2, d2, e2 = _philox_round(a2, b2, d2, e2, k0, k1)
            a3, b3, d3, e3 = _philox_round(a3, b3, d3, e3, k0, k1)
            k0 = k0 + Uint32(_PHILOX_KEY_A)
            k1 = k1 + Uint32(_PHILOX_KEY_B)
        out = [[a0, b0, d0, e0], [a1, b1, d1, e1], [a2, b2, d2, e2], [a3, b3, d3, e3]]
    else:
        a0, b0, d0, e0 = st[0]
        a1, b1, d1, e1 = st[1]
        for _ in cutlass.range(rest, unroll=unroll):
            a0, b0, d0, e0 = _philox_round(a0, b0, d0, e0, k0, k1)
            a1, b1, d1, e1 = _philox_round(a1, b1, d1, e1, k0, k1)
            k0 = k0 + Uint32(_PHILOX_KEY_A)
            k1 = k1 + Uint32(_PHILOX_KEY_B)
        out = [[a0, b0, d0, e0], [a1, b1, d1, e1]]
    return out


@cute.jit
def _philox_loop(
    seed_lo: Uint32,
    seed_hi: Uint32,
    off_lo0: Uint32,
    off_hi: Uint32,
    n_streams: cutlass.Constexpr,
    n_rounds: cutlass.Constexpr,
):
    """Plain Philox4x32 rounds as a runtime loop (see ``philox4x32_streams``)."""
    k0, k1 = Uint32(seed_lo), Uint32(seed_hi)
    z = Uint32(0)
    a0, b0, d0, e0 = Uint32(off_lo0), Uint32(off_hi), z, z
    a1, b1, d1, e1 = Uint32(off_lo0) + Uint32(8), Uint32(off_hi), z, z
    if cutlass.const_expr(n_streams == 4):
        a2, b2, d2, e2 = Uint32(off_lo0) + Uint32(16), Uint32(off_hi), z, z
        a3, b3, d3, e3 = Uint32(off_lo0) + Uint32(24), Uint32(off_hi), z, z
        for _ in cutlass.range(n_rounds, unroll=1):
            a0, b0, d0, e0 = _philox_round(a0, b0, d0, e0, k0, k1)
            a1, b1, d1, e1 = _philox_round(a1, b1, d1, e1, k0, k1)
            a2, b2, d2, e2 = _philox_round(a2, b2, d2, e2, k0, k1)
            a3, b3, d3, e3 = _philox_round(a3, b3, d3, e3, k0, k1)
            k0 = k0 + Uint32(_PHILOX_KEY_A)
            k1 = k1 + Uint32(_PHILOX_KEY_B)
        out = [[a0, b0, d0, e0], [a1, b1, d1, e1], [a2, b2, d2, e2], [a3, b3, d3, e3]]
    else:
        for _ in cutlass.range(n_rounds, unroll=1):
            a0, b0, d0, e0 = _philox_round(a0, b0, d0, e0, k0, k1)
            a1, b1, d1, e1 = _philox_round(a1, b1, d1, e1, k0, k1)
            k0 = k0 + Uint32(_PHILOX_KEY_A)
            k1 = k1 + Uint32(_PHILOX_KEY_B)
        out = [[a0, b0, d0, e0], [a1, b1, d1, e1]]
    return out


def _philox_round(c0, c1, c2, c3, k0, k1):
    pb = Uint64(c2) * Uint64(_PHILOX_ROUND_B)
    pa = Uint64(c0) * Uint64(_PHILOX_ROUND_A)
    return (
        Uint32(pb >> Uint64(32)) ^ c1 ^ k0,
        Uint32(pb & Uint64(0xFFFFFFFF)),
        Uint32(pa >> Uint64(32)) ^ c3 ^ k1,
        Uint32(pa & Uint64(0xFFFFFFFF)),
    )


# ---- LCG stochastic-rounding dither (the cheap alternative to Philox) ----
LCG_A, LCG_C = 1664525, 1013904223  # full-period LCG mod 2^32 (Numerical Recipes)
LCG_A4 = LCG_A**4 % 2**32  # 4 steps at once: x_{n+4} = A4 x_n + C4
LCG_C4 = LCG_C * (LCG_A**3 + LCG_A**2 + LCG_A + 1) % 2**32
_GOLDEN, _MIX1, _MIX2 = 0x9E3779B9, 0x85EBCA6B, 0xC2B2AE35


def fmix32(x):
    """murmur3 32-bit finalizer (bijective avalanche)."""
    x = Uint32(x)
    x = x ^ (x >> Uint32(16))
    x = x * Uint32(_MIX1)
    x = x ^ (x >> Uint32(13))
    x = x * Uint32(_MIX2)
    return x ^ (x >> Uint32(16))


@dsl_user_op
def clock32(*, loc=None, ip=None) -> Uint32:
    """%clock (this SM's cycle counter): the unseeded LCG mode's entropy."""
    r = llvm.inline_asm(
        T.i32(),
        [],
        "mov.u32 $0, %clock;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


@dsl_user_op
def pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """cvt.rn.bf16x2.f32: ``lo`` in bits 0..15, ``hi`` in bits 16..31."""
    r = llvm.inline_asm(
        T.i32(),
        [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


@dsl_user_op
def u16_as_f32(w: Uint32, is_bf16: bool, *, loc=None, ip=None) -> Float32:
    """Low 16 bits of ``w`` (bf16 or fp16 payload) widened to fp32."""
    if is_bf16:
        asm = "shl.b32 $0, $1, 16;"
    else:
        asm = "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tcvt.f32.f16 $0, h;\n\t}"
    r = llvm.inline_asm(
        T.f32(),
        [Uint32(w).ir_value(loc=loc, ip=ip)],
        asm,
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def f32_to_u16(x: Float32, is_bf16: bool, *, loc=None, ip=None) -> Uint32:
    """Round ``x`` to bf16 / fp16 (RN), payload in the low 16 bits."""
    cvt = "cvt.rn.bf16.f32" if is_bf16 else "cvt.rn.f16.f32"
    r = llvm.inline_asm(
        T.i32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 h;\n\t" + cvt + " h, $1;\n\tcvt.u32.u16 $0, h;\n\t}",
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


@dsl_user_op
def lds_v4(addr: Int32, *, loc=None, ip=None):
    """ld.shared.v4.b32 -> four u32 words."""
    st = llvm.StructType.get_literal([T.i32()] * 4)
    r = llvm.inline_asm(
        st,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def sts_v4(addr: Int32, w0, w1, w2, w3, *, loc=None, ip=None) -> None:
    """st.shared.v4.b32."""
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)]
        + [Uint32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.shared.v4.b32 [$0], {$1,$2,$3,$4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts_u16(addr: Int32, w: Uint32, *, loc=None, ip=None) -> None:
    """st.shared.u16 of the low half of ``w``."""
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Uint32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tst.shared.u16 [$0], h;\n\t}",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def f16x2_lo_f32(w: Uint32, *, loc=None, ip=None) -> Float32:
    r = llvm.inline_asm(
        T.f32(),
        [Uint32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, $1;\n\tcvt.f32.f16 $0, l;\n\t}",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def f16x2_hi_f32(w: Uint32, *, loc=None, ip=None) -> Float32:
    r = llvm.inline_asm(
        T.f32(),
        [Uint32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, $1;\n\tcvt.f32.f16 $0, h;\n\t}",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def bulk_g2s(
    dst_smem: Int32,
    src_gmem: Int64,
    nbytes: Int32,
    mbar_smem: Int32,
    *,
    loc=None,
    ip=None,
):
    """cp.async.bulk global -> shared (1-D), completion counted on an mbarrier (complete_tx).
    ``nbytes`` a multiple of 16; addresses 16-B aligned."""
    llvm.inline_asm(
        T.i32(),
        [
            Int32(dst_smem).ir_value(loc=loc, ip=ip),
            Int64(src_gmem).ir_value(loc=loc, ip=ip),
            Int32(nbytes).ir_value(loc=loc, ip=ip),
            Int32(mbar_smem).ir_value(loc=loc, ip=ip),
        ],
        "{\n\tcp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [$1], [$2], $3, [$4];\n\tmov.u32 $0, 0;\n\t}\n",
        "=r,r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bulk_prefetch_l2(
    src_gmem: Int64, nbytes: int, evict_last: bool = True, *, loc=None, ip=None
) -> None:
    """cp.async.bulk.prefetch.L2 (1-D): pull ``nbytes`` (a multiple of 16, 16-B aligned
    address) of global memory into L2, by default at evict_last priority. Fire and
    forget: no completion mechanism, no destination registers."""
    if evict_last:
        asm = (
            "{\n\t.reg .b64 pol;\n\t"
            "createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n\t"
            "cp.async.bulk.prefetch.L2.global.L2::cache_hint"
            f" [$0], {int(nbytes)}, pol;\n\t}}"
        )
    else:
        asm = f"cp.async.bulk.prefetch.L2.global [$0], {int(nbytes)};"
    llvm.inline_asm(
        None,
        [Int64(src_gmem).ir_value(loc=loc, ip=ip)],
        asm,
        "l",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def lds_u16(addr: Int32, *, loc=None, ip=None) -> Uint32:
    """ld.shared.u16 zero-extended into a 32-bit register (raw 16-bit payload).  No
    conversion in the asm block: a cvt there would make every load wait for its data
    before the next one issues."""
    r = llvm.inline_asm(
        T.i32(),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.u16 $0, [$1];",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


def make_fake_tensor_lead_dyn(dtype, shape, divisibility=1, inner_shape=None):
    """Fake CuTe tensor whose inner dims have STATIC strides and whose leading stride is
    dynamic (paged / block-strided pools), so in-kernel addressing folds the inner offsets
    into immediates. The inner strides are those of a dense ``inner_shape`` (default
    ``shape[1:]``); a larger ``inner_shape`` describes a slice view of a wider tensor (e.g.
    some heads of every token)."""
    inner = 1
    strides = []
    for d in reversed(shape[1:] if inner_shape is None else inner_shape):
        strides.append(inner)
        inner *= d
    strides = [cute.sym_int64(divisibility=divisibility)] + list(reversed(strides))
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=tuple(strides),
        assumed_align=max(divisibility * dtype.width // 8, 1),
    )


@dsl_user_op
def bf16x2_lo_f32(w: Uint32, *, loc=None, ip=None) -> Float32:
    r = llvm.inline_asm(
        T.f32(),
        [Uint32(w).ir_value(loc=loc, ip=ip)],
        "shl.b32 $0, $1, 16;",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def bf16x2_hi_f32(w: Uint32, *, loc=None, ip=None) -> Float32:
    r = llvm.inline_asm(
        T.f32(),
        [Uint32(w).ir_value(loc=loc, ip=ip)],
        "and.b32 $0, $1, 0xffff0000;",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def stg_v4(addr: Int64, w0, w1, w2, w3, *, loc=None, ip=None) -> None:
    """st.global.v4.b32 (16 B, aligned)."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Uint32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.b32 [$0], {$1,$2,$3,$4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ldg_v4_if(pred: Int32, addr: Int64, *, loc=None, ip=None):
    """Predicated ld.global.v4.b32: threads with pred == 0 skip the load (values undefined)."""
    st = llvm.StructType.get_literal([T.i32()] * 4)
    r = llvm.inline_asm(
        st,
        [Int32(pred).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $4, 0;\n\t"
        "@p ld.global.v4.b32 {$0,$1,$2,$3}, [$5];\n\t}",
        "=r,=r,=r,=r,r,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def ldg_v4_if_keep(pred: Int32, addr: Int64, w0, w1, w2, w3, *, loc=None, ip=None):
    """Predicated ld.global.v4.b32 into the four given registers: threads with pred == 0 keep
    their values (tied operands), so one register set can hold either of two loads by role."""
    st = llvm.StructType.get_literal([T.i32()] * 4)
    r = llvm.inline_asm(
        st,
        [Int32(pred).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Uint32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $4, 0;\n\t"
        "@p ld.global.v4.b32 {$0,$1,$2,$3}, [$5];\n\t}",
        "=r,=r,=r,=r,r,l,0,1,2,3",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


def sdesc_mn_sw128(lbo: int, sbo: int):
    """SMEM descriptor template, 128B swizzle, MN-major operand (OR in addr >> 4).  For an
    operand stored row-major [K rows][MN] in 128 B swizzled rows (8-row atoms): SBO = byte
    stride between 8-row K atoms; LBO = stride between 64-element MN atoms (unused for N=64)."""
    return Uint64(((lbo >> 4) << 16) | ((sbo >> 4) << 32) | (1 << 46) | (2 << 61))


@dsl_user_op
def ldg_u16_if(pred: Int32, addr: Int64, *, loc=None, ip=None) -> Uint32:
    """Predicated ld.global.u16 (zero-extended): threads with pred == 0 skip the load (value 0)."""
    r = llvm.inline_asm(
        T.i32(),
        [Int32(pred).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\t.reg .b16 h;\n\tmov.b16 h, 0;\n\tsetp.ne.s32 p, $1, 0;\n\t"
        "@p ld.global.u16 h, [$2];\n\tcvt.u32.u16 $0, h;\n\t}",
        "=r,r,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


@dsl_user_op
def ldg_f32_if(pred: Int32, addr: Int64, *, loc=None, ip=None) -> Float32:
    """Predicated ld.global.f32: threads with pred == 0 skip the load (value 0.0)."""
    r = llvm.inline_asm(
        T.f32(),
        [Int32(pred).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tmov.f32 $0, 0f00000000;\n\tsetp.ne.s32 p, $1, 0;\n\t"
        "@p ld.global.f32 $0, [$2];\n\t}",
        "=f,r,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def mbar_arrive_if(
    pred: Int32, mbar_smem: Int32, fence: bool = False, *, loc=None, ip=None
) -> None:
    """mbarrier.arrive on ``mbar_smem`` where ``pred`` != 0; with ``fence``, a
    fence.proxy.async.shared::cta (same predicate) ahead of it."""
    f = "@p fence.proxy.async.shared::cta;\n\t" if fence else ""
    llvm.inline_asm(
        None,
        [
            Int32(pred).ir_value(loc=loc, ip=ip),
            Int32(mbar_smem).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\t.reg .b64 st;\n\tsetp.ne.s32 p, $0, 0;\n\t"
        + f
        + "@p mbarrier.arrive.shared::cta.b64 st, [$1];\n\t}",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ffma_f16x2_f32(
    w: Uint32, b: Float32, c0: Float32, c1: Float32, *, loc=None, ip=None
):
    """(f32(lo(w)) * b + c0, f32(hi(w)) * b + c1) for an fp16 pair ``w``: one packed
    ``fma.rn.f32x2.f16x2.f32x2.f32x2`` (SASS FHFMA2 on sm_107a; not available on sm_100a /
    sm_103a). The fp16 -> fp32 widening is exact and each lane is one round-to-nearest fp32
    FMA, so it is bit-identical to cvt.f32.f16 + fma.rn.f32 per value."""
    vt16 = ir.VectorType.get([2], cutlass.Float16.mlir_type, loc=loc)
    vt32 = ir.VectorType.get([2], Float32.mlir_type, loc=loc)
    va = llvm.bitcast(vt16, Uint32(w).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    bb = Float32(b).ir_value(loc=loc, ip=ip)
    vb = vector.from_elements(vt32, [bb, bb], loc=loc, ip=ip)
    vc = vector.from_elements(
        vt32,
        [Float32(c0).ir_value(loc=loc, ip=ip), Float32(c1).ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )
    r = nvvm.fma_packed_f32x2_f16x2_f32x2_f32x2(
        vt32, va, vb, vc, rnd=nvvm.FPRoundingMode.RN, loc=loc, ip=ip
    )
    return (
        Float32(
            vector.extract(r, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
        ),
        Float32(
            vector.extract(r, dynamic_position=[], static_position=[1], loc=loc, ip=ip)
        ),
    )


# ----------------------------------------------------------------------------- persistent kernel
@dsl_user_op
def bulk_prefetch_l2_n(
    src_gmem: Int64, nbytes: Int32, pred: Int32, *, loc=None, ip=None
) -> None:
    """cp.async.bulk.prefetch.L2 (1-D, evict_last) of a run-time ``nbytes`` (a multiple of 16,
    16-B aligned address) where ``pred`` != 0."""
    llvm.inline_asm(
        None,
        [
            Int64(src_gmem).ir_value(loc=loc, ip=ip),
            Int32(nbytes).ir_value(loc=loc, ip=ip),
            Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\t.reg .b64 pol;\n\tsetp.ne.s32 p, $2, 0;\n\t"
        "createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n\t"
        "@p cp.async.bulk.prefetch.L2.global.L2::cache_hint [$0], $1, pol;\n\t}",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tid_x_volatile(*, loc=None, ip=None) -> Int32:
    """%tid.x through a side-effecting asm: a loop body's thread-derived addresses are then
    recomputed in the body instead of hoisted out of the loop (and spilled) by the compilers."""
    r = llvm.inline_asm(
        T.i32(),
        [],
        "mov.u32 $0, %tid.x;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Int32(r)


@dsl_user_op
def ldg_v4_if_z(pred: Int32, addr: Int64, *, loc=None, ip=None):
    """ldg_v4_if with zeros where pred == 0. In a loop the predicated-off destination registers
    of ldg_v4_if would otherwise carry the previous iteration's values (ptxas keeps them live
    across the back edge, and spills them)."""
    st = llvm.StructType.get_literal([T.i32()] * 4)
    r = llvm.inline_asm(
        st,
        [Int32(pred).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tmov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\tmov.b32 $2, 0;\n\t"
        "mov.b32 $3, 0;\n\tsetp.ne.s32 p, $4, 0;\n\t"
        "@p ld.global.v4.b32 {$0,$1,$2,$3}, [$5];\n\t}",
        "=r,=r,=r,=r,r,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def claim_next(ctr: Int64, *, loc=None, ip=None) -> Int32:
    """Warp-wide: lane 0 does atom.global.add.u32 [ctr], 1 (relaxed, gpu scope); every lane gets
    the old value (shfl from lane 0). Call converged."""
    r = llvm.inline_asm(
        T.i32(),
        [Int64(ctr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\t.reg .u32 l, v;\n\tmov.u32 v, 0;\n\tmov.u32 l, %laneid;\n\t"
        "setp.eq.u32 p, l, 0;\n\t@p atom.global.add.u32 v, [$1], 1;\n\t"
        "shfl.sync.idx.b32 $0, v, 0, 31, -1;\n\t}",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Int32(r)


@dsl_user_op
def claim_done(ctr: Int64, n_cta: Int32, *, loc=None, ip=None) -> None:
    """Warp-wide, once per CTA after its last claim: lane 0 increments the done count [ctr + 4]
    (acq_rel, gpu scope); the CTA that brings it to n_cta zeroes both words (self-resetting
    counter: every claim of the launch happened before, the next launch sees zeros)."""
    llvm.inline_asm(
        None,
        [Int64(ctr).ir_value(loc=loc, ip=ip), Int32(n_cta).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p, q;\n\t.reg .u32 l, o, z, m;\n\tmov.u32 z, 0;\n\tmov.u32 o, 0;\n\t"
        "mov.u32 l, %laneid;\n\tsetp.eq.u32 p, l, 0;\n\tsub.u32 m, $1, 1;\n\t"
        "@p atom.acq_rel.gpu.global.add.u32 o, [$0+4], 1;\n\t"
        "setp.eq.and.u32 q, o, m, p;\n\t@q st.relaxed.gpu.global.v2.u32 [$0], {z, z};\n\t}",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


# --------------------------------------------------------------------------- fused verify conv
# (SGLang, SGLANG_ENABLE_GDN_UCACHE_FUSED_CONV: the flush kernel's conv prologue)


@dsl_user_op
def opaque_i64(x: Int64, *, loc=None, ip=None) -> Int64:
    """``x`` through a side-effecting move: addresses derived from it are computed after every
    side-effecting op before it (a barrier, a griddepcontrol.wait), so the pure-asm loads that use
    them cannot be scheduled above those."""
    r = llvm.inline_asm(
        T.i64(),
        [Int64(x).ir_value(loc=loc, ip=ip)],
        "mov.b64 $0, $1;",
        "=l,l",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Int64(r)


@dsl_user_op
def ldg_s32_if(pred: Int32, addr: Int64, *, loc=None, ip=None) -> Int32:
    """Predicated ld.global.s32: threads with pred == 0 skip the load (value 0)."""
    r = llvm.inline_asm(
        T.i32(),
        [Int32(pred).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tmov.b32 $0, 0;\n\tsetp.ne.s32 p, $1, 0;\n\t"
        "@p ld.global.s32 $0, [$2];\n\t}",
        "=r,r,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Int32(r)


@dsl_user_op
def stg_v4_if(pred: Int32, addr: Int64, w0, w1, w2, w3, *, loc=None, ip=None) -> None:
    """Predicated st.global.v4.b32 (16 B, aligned)."""
    llvm.inline_asm(
        None,
        [Int32(pred).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Uint32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $0, 0;\n\t"
        "@p st.global.v4.b32 [$1], {$2,$3,$4,$5};\n\t}",
        "r,l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def prmt(a: Uint32, b: Uint32, sel: int, *, loc=None, ip=None) -> Uint32:
    """prmt.b32 with a constant selector: 0x5410 = (a.lo, b.lo), 0x5432 = (a.hi, b.lo),
    0x7610 = (a.lo, b.hi), 0x7632 = (a.hi, b.hi) as (bits 0..15, bits 16..31)."""
    r = llvm.inline_asm(
        T.i32(),
        [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
        f"prmt.b32 $0, $1, $2, {int(sel)};",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


@dsl_user_op
def conv4_silu_bf16x2(c0, c1, c2, x, w0, w1, w2, w3, *, loc=None, ip=None) -> Uint32:
    """Two channels (bf16x2 lanes) of one width-4 causal-conv output with SiLU, as SGLang's Triton
    verify conv computes it: bf16 products (mul.rn.bf16x2), an fp32 sum in tap order from +0
    (oldest state column first, the token last), SiLU as acc / (1 + ex2.approx(-acc * log2 e))
    with div.full, one round to bf16 (the op sequence of gdn_prefill_conv_strip.py)."""
    r = llvm.inline_asm(
        T.i32(),
        [Uint32(v).ir_value(loc=loc, ip=ip) for v in (c0, c1, c2, x, w0, w1, w2, w3)],
        "{\n\t.reg .b32 p0, p1, p2, p3, t, l, h, e;\n\t"
        "mul.rn.bf16x2 p0, $1, $5;\n\tmul.rn.bf16x2 p1, $2, $6;\n\t"
        "mul.rn.bf16x2 p2, $3, $7;\n\tmul.rn.bf16x2 p3, $4, $8;\n\t"
        "mov.b32 l, 0f00000000;\n\tmov.b32 h, 0f00000000;\n\t"
        "shl.b32 t, p0, 16;\n\tadd.rn.f32 l, l, t;\n\tand.b32 t, p0, 0xffff0000;\n\tadd.rn.f32 h, h, t;\n\t"
        "shl.b32 t, p1, 16;\n\tadd.rn.f32 l, l, t;\n\tand.b32 t, p1, 0xffff0000;\n\tadd.rn.f32 h, h, t;\n\t"
        "shl.b32 t, p2, 16;\n\tadd.rn.f32 l, l, t;\n\tand.b32 t, p2, 0xffff0000;\n\tadd.rn.f32 h, h, t;\n\t"
        "shl.b32 t, p3, 16;\n\tadd.rn.f32 l, l, t;\n\tand.b32 t, p3, 0xffff0000;\n\tadd.rn.f32 h, h, t;\n\t"
        "neg.f32 t, l;\n\tmul.f32 t, t, 0f3FB8AA3B;\n\tex2.approx.f32 e, t;\n\t"
        "add.f32 e, e, 0f3F800000;\n\tdiv.full.f32 l, l, e;\n\t"
        "neg.f32 t, h;\n\tmul.f32 t, t, 0f3FB8AA3B;\n\tex2.approx.f32 e, t;\n\t"
        "add.f32 e, e, 0f3F800000;\n\tdiv.full.f32 h, h, e;\n\t"
        "cvt.rn.bf16x2.f32 $0, h, l;\n\t}",
        "=r,r,r,r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


@dsl_user_op
def addr_after(addr: Int64, deps, *, loc=None, ip=None) -> Int64:
    """``addr`` returned through an asm that also reads ``deps`` (u32 values): a store to it
    issues only once every load those values came from has returned."""
    r = llvm.inline_asm(
        T.i64(),
        [Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Uint32(d).ir_value(loc=loc, ip=ip) for d in deps],
        "mov.b64 $0, $1;",
        "=l,l" + ",r" * len(deps),
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Int64(r)


# --------------------------------------------------------------------------- fused gated norm
# (SGLang, SGLANG_ENABLE_GDN_UCACHE_FUSED_NORM: the flush epilogue's RMSNorm, in the op order of the
# Triton _layer_norm_fwd_1pass_kernel's 16-lane x 8-element row reduction)


@dsl_user_op
def norm_group_sumsq(x: Float32, *, loc=None, ip=None) -> Float32:
    """Sum of x^2 over the 8 lanes 8i .. 8i + 7 (one Triton thread's 8 row elements), in that
    thread's order: (x0, x1) .. (x6, x7) as f32x2 pairs, pairs added (0+2, 4+6), then the halves."""
    r = llvm.inline_asm(
        T.f32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .f32 s, o;\n\tmul.rn.f32 s, $1, $1;\n\t"
        "shfl.sync.bfly.b32 o, s, 2, 31, -1;\n\tadd.rn.f32 s, s, o;\n\t"
        "shfl.sync.bfly.b32 o, s, 4, 31, -1;\n\tadd.rn.f32 s, s, o;\n\t"
        "shfl.sync.bfly.b32 o, s, 1, 31, -1;\n\tadd.rn.f32 $0, s, o;\n\t}",
        "=f,f",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def norm_rstd16(p, eps: Float32, *, loc=None, ip=None) -> Float32:
    """rstd of one 128-wide row from its 16 thread partials ``p`` (thread i = elements 8i ..): the
    butterfly at lane offsets 8, 4, 2, 1, then rsqrt(sum * (1 / 128) + eps) as one fma."""
    r = llvm.inline_asm(
        T.f32(),
        [Float32(v).ir_value(loc=loc, ip=ip) for v in p] + [Float32(eps).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .f32 a<8>, b<4>, c<2>, d, n, rc;\n\t"
        + "".join(f"add.rn.f32 a{i}, ${1 + i}, ${1 + i + 8};\n\t" for i in range(8))
        + "".join(f"add.rn.f32 b{i}, a{i}, a{i + 4};\n\t" for i in range(4))
        + "add.rn.f32 c0, b0, b2;\n\tadd.rn.f32 c1, b1, b3;\n\tadd.rn.f32 d, c0, c1;\n\t"
        "mov.f32 n, 0f43000000;\n\tmov.f32 rc, 0f3F800000;\n\tdiv.approx.f32 rc, rc, n;\n\t"
        "fma.rn.f32 d, rc, d, $17;\n\trsqrt.approx.ftz.f32 $0, d;\n\t}",
        "=f" + ",f" * 17,
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def norm_gate_out(x: Float32, rstd: Float32, w: Float32, z: Float32, *, loc=None, ip=None) -> Uint32:
    """bf16 bits of (x * rstd) * w * sigmoid(z), sigmoid(z) = 1 / (1 + ex2.approx(z * -log2 e))
    with div.full, as the Triton gated norm (norm before gate) computes it."""
    r = llvm.inline_asm(
        T.i32(),
        [Float32(v).ir_value(loc=loc, ip=ip) for v in (x, rstd, w, z)],
        "{\n\t.reg .f32 xh, t, e, sg, y;\n\t.reg .b16 h;\n\t"
        "mul.rn.f32 xh, $2, $1;\n\t"
        "mul.rn.f32 t, $4, 0fBFB8AA3B;\n\tex2.approx.f32 e, t;\n\tadd.rn.f32 e, e, 0f3F800000;\n\t"
        "mov.f32 sg, 0f3F800000;\n\tdiv.full.f32 sg, sg, e;\n\t"
        "mul.rn.f32 y, xh, $3;\n\tmul.rn.f32 y, y, sg;\n\t"
        "cvt.rn.bf16.f32 h, y;\n\tcvt.u32.u16 $0, h;\n\t}",
        "=r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)
