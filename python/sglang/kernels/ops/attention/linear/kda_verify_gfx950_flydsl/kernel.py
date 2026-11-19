# SPDX-License-Identifier: Apache-2.0
"""FlyDSL gfx950 KDA target verify with the ReplaySSM ring: causal conv1d (width 4) +
safe-gate delta rule over T draft tokens, K = V = 128, one KDA head per (q, k, v) head.

One CTA of 256 threads per (request, v head, BVC-wide V tile). Threads 0..127 convolve
the q channels and compute the gate, threads 128..255 convolve the k channels and the
tile's v channels and compute beta; the results go through LDS. The recurrence keeps a
128 x BVC fp32 state tile in registers (a bf16 state pool is widened on load; verify
never writes the state): 8 adjacent lanes split K (lane ks owns float4
groups ks, ks + 8, ...) and reduce a column's dot products with DPP, each thread holding
BVC / 32 columns, 32 apart.

The ring values (raw k / v, gate, beta) and the intermediate conv window match
fused_kda_conv_gating_verify_kernel bit for bit: the conv products are rounded before
the adds, and exp / divide follow the Triton gfx950 lowering. The output differs from it
in reduction order and in the l2 norm, which uses v_rsq_f32.
"""

import functools
import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

_KD = 128
_THREADS = 256
_KSPLIT = 8
_KG = _KD // (4 * _KSPLIT)
_NCG = _THREADS // _KSPLIT
_LOG2E = struct.unpack("f", struct.pack("I", 0x3FB8AA3B))[0]
# Token t's q dot products must finish before token t + _LAG starts (see the recurrence).
_LAG = {32: 0, 64: 2}


def _rsrc(addr):
    return bo.create_buffer_resource_from_addr(addr)


def _load(addr, off, dtype, vec=1):
    return bo.buffer_load(_rsrc(addr), off, vec_width=vec, dtype=dtype)


def _load_uniform(addr, i):
    return fx.Int32(
        bo.buffer_load(_rsrc(addr), i, vec_width=1, dtype=T.i32, is_scalar=True)
    )


def _uniform(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v)))


def _dpp(v, ctrl):
    x = fx.Float32(v).bitcast(fx.Int32)
    return fx.Int32(update_dpp_i32(x, x, ctrl, 0xF, 0xF, False)).bitcast(fx.Float32)


def _sum8(v):
    """Sum over aligned groups of 8 lanes: quad xor 1, quad xor 2, row_half_mirror."""
    v = v + _dpp(v, 0xB1)
    v = v + _dpp(v, 0x4E)
    return v + _dpp(v, 0x141)


def _readlane(v, lane):
    x = fx.Float32(v).bitcast(fx.Int32)
    return fx.Int32(rocdl.readlane(T.i32, x, fx.Int32(lane))).bitcast(fx.Float32)


def _wave_sum(v):
    v = _sum8(v)
    v = v + _dpp(v, 0x140)
    return _readlane(v, 0) + _readlane(v, 16) + _readlane(v, 32) + _readlane(v, 48)


def _after(x, deps):
    """x through an opaque move that also reads deps, so its users wait for them."""
    args = [fx.Float32(x).bitcast(fx.Int32).ir_value()]
    args += [fx.Float32(d).bitcast(fx.Int32).ir_value() for d in deps]
    y = llvm.inline_asm(
        T.i32, args, "v_mov_b32 $0, $1", "=v" + ",v" * len(args), has_side_effects=False
    )
    return fx.Int32(y).bitcast(fx.Float32)


def _exp2(x):
    return fx.Float32(rocdl.exp2(T.f32, fx.Float32(x).ir_value()))


def _rsq(x):
    return fx.Float32(rocdl.rsq(T.f32, fx.Float32(x).ir_value()))


def _tl_exp(x):
    """tl.exp on gfx950: v_exp_f32(fma(x, log2e, off)) * 2^-off, off = 64 below 2^-126."""
    small = (x * fx.Float32(_LOG2E)) < fx.Float32(-126.0)
    off = small.select(fx.Float32(64.0), fx.Float32(0.0))
    r = _exp2(fmath.fma(x, fx.Float32(_LOG2E), off))
    return r * small.select(fx.Float32(2.0**-64), fx.Float32(1.0))


def _conv_silu(xs, w):
    """Width-4 conv tap sum (every product rounded) then SiLU, rounded to bf16."""
    acc = fx.Float32(0.0) + fx.Float32(xs[0]) * w[0]
    for i in range(1, 4):
        acc = acc + fx.Float32(xs[i]) * w[i]
    return (acc / (fx.Float32(1.0) + _tl_exp(-acc))).to(fx.BFloat16)


@functools.cache
def create_kda_verify_kernel(
    T_: int,
    block_v: int,
    num_q_heads: int,
    num_v_heads: int,
    lower_bound: float,
    scale: float,
    has_iw: bool,
    state_bf16: bool = False,
):
    TT, BVC, H, HV = T_, block_v, num_q_heads, num_v_heads
    G = HV // H
    VPT = BVC // _NCG
    LAG = _LAG[BVC]
    STATE_SLOT = HV * _KD * _KD
    STATE_BYTES = 2 if state_bf16 else 4

    @fx.struct
    class SharedStorage:
        qk: fx.Array[fx.Float32, 2 * TT * _KD, 16]
        dec: fx.Array[fx.Float32, TT * _KD, 16]
        v: fx.Array[fx.Float32, TT * BVC, 16]
        beta: fx.Array[fx.Float32, 16, 16]
        ss: fx.Array[fx.Float32, 4 * TT, 16]

    @flyc.kernel(
        name=f"kda_verify_t{TT}_v{BVC}_h{H}_hv{HV}_iw{int(has_iw)}"
        + ("_sbf16" if state_bf16 else ""),
        known_block_size=[_THREADS, 1, 1],
    )
    def kernel(
        x: Int64,
        w: Int64,
        cs: Int64,
        cs_idx: Int64,
        iw: Int64,
        iw_idx: Int64,
        a: Int64,
        b: Int64,
        A_log: Int64,
        dt_bias: Int64,
        state: Int64,
        h0_idx: Int64,
        o: Int64,
        rawv: Int64,
        rawk: Int64,
        ring_g: Int64,
        ring_beta: Int64,
        ring_idx: Int64,
        x_stride: Int32,
        cs_line: Int32,
        cs_tok: Int32,
        iw_line: Int32,
        iw_step: Int32,
        iw_win: Int32,
        a_stride: Int32,
        b_stride: Int32,
        rawv_slot: Int32,
        rawk_slot: Int32,
        g_slot: Int32,
        beta_slot: Int32,
        L: Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        vt = fx.Int32(fx.block_idx.x)
        by = fx.Int32(fx.block_idx.y)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_qk, s_dec, s_v = lds.qk.ptr, lds.dec.ptr, lds.v.ptr
        s_beta, s_ss = lds.beta.ptr, lds.ss.ptr

        n = by // HV
        hv = by % HV
        h = hv // G
        v0 = vt * BVC
        bos = n * TT
        cs_slot = _load_uniform(cs_idx, n)
        h0 = _load_uniform(h0_idx, n)
        ring = _load_uniform(ring_idx, n)
        live = cs_slot >= 0
        ring_ok = live & (ring >= 0)
        qk_owner = (vt == 0) & (hv % G == 0)
        ring_c = fx.Int64(fx.max(ring, 0))
        cg = tid // _KSPLIT
        ks = tid % _KSPLIT

        # Prep loads, all unconditional at clamped addresses. The state tile goes last:
        # vmcnt retires in order, so the prep would otherwise wait for it.
        ch = tid % _KD
        is_k = _uniform(tid) >= _KD
        qk_ch = is_k.select(fx.Int32(H * _KD), fx.Int32(0)) + h * _KD + ch
        cs_base = cs + fx.Int64(fx.max(cs_slot, 0)) * fx.Int64(cs_line) * 2
        xw = [fx.BFloat16(_load(cs_base, qk_ch + j * cs_tok, T.bf16)) for j in range(3)]
        xw += [
            fx.BFloat16(_load(x, (bos + t) * x_stride + qk_ch, T.bf16))
            for t in range(TT)
        ]
        wq = fx.Vector(_load(w, qk_ch * 4, T.f32, 4))
        has_v = is_k & (ch < BVC)
        v_ch = 2 * H * _KD + hv * _KD + v0 + fx.min(ch, BVC - 1)
        vw = [fx.BFloat16(_load(cs_base, v_ch + j * cs_tok, T.bf16)) for j in range(3)]
        vw += [
            fx.BFloat16(_load(x, (bos + t) * x_stride + v_ch, T.bf16))
            for t in range(TT)
        ]
        wv = fx.Vector(_load(w, v_ch * 4, T.f32, 4))
        dtb = fx.Float32(_load(dt_bias, hv * _KD + ch, T.f32))
        a_log = fx.Float32(_load(A_log, hv, T.f32))
        av = [
            fx.Float32(
                fx.BFloat16(_load(a, (bos + t) * a_stride + hv * _KD + ch, T.bf16))
            )
            for t in range(TT)
        ]
        b_row = bos + fx.min(ch, TT - 1)
        bb = fx.Float32(fx.BFloat16(_load(b, b_row * b_stride + hv, T.bf16)))
        st_base = state + fx.Int64(fx.max(h0, 0)) * fx.Int64(STATE_SLOT * STATE_BYTES)
        state_t = T.bf16 if state_bf16 else T.f32
        hs = []
        for j in range_constexpr(VPT):
            row = (hv * _KD + v0 + cg + _NCG * j) * _KD
            hs.append(
                [
                    fx.Vector(_load(st_base, row + (i * _KSPLIT + ks) * 4, state_t, 4))
                    for i in range(_KG)
                ]
            )

        # conv + silu, rounded to bf16 as the unfused path stores it
        qk = []
        wql = [wq[i] for i in range(4)]
        qk_row = is_k.select(fx.Int32(TT * _KD), fx.Int32(0))
        rawk_base = rawk + ring_c * fx.Int64(rawk_slot) * 2
        for t in range_constexpr(TT):
            r = _conv_silu(xw[t : t + 4], wql)
            qk.append(fx.Float32(r))
            fx.ptr_store(fx.Float32(r), s_qk + (qk_row + t * _KD + ch))
            if is_k & qk_owner & ring_ok & (fx.Int32(t) < L):
                bo.buffer_store(r, _rsrc(rawk_base), (h * L + t) * _KD + ch)
        wvl = [wv[i] for i in range(4)]
        rawv_base = rawv + ring_c * fx.Int64(rawv_slot) * 2
        if has_v:
            for t in range_constexpr(TT):
                r = _conv_silu(vw[t : t + 4], wvl)
                fx.ptr_store(fx.Float32(r), s_v + (t * BVC + ch))
                if ring_ok & (fx.Int32(t) < L):
                    bo.buffer_store(r, _rsrc(rawv_base), (hv * L + t) * _KD + v0 + ch)
        g_base = ring_g + ring_c * fx.Int64(g_slot) * 4
        if _uniform(tid) < _KD:
            ea = _tl_exp(a_log)
            for t in range_constexpr(TT):
                y = ea * (av[t] + dtb)
                g = fx.Float32(lower_bound) * (
                    fx.Float32(1.0) / (fx.Float32(1.0) + _tl_exp(-y))
                )
                fx.ptr_store(_tl_exp(g), s_dec + (t * _KD + ch))
                if (vt == 0) & ring_ok & (fx.Int32(t) < L):
                    bo.buffer_store(g, _rsrc(g_base), (hv * L + t) * _KD + ch)
        beta_base = ring_beta + ring_c * fx.Int64(beta_slot) * 4
        if is_k & (ch < TT):
            beta = fx.Float32(1.0) / (fx.Float32(1.0) + _tl_exp(-bb))
            fx.ptr_store(beta, s_beta + ch)
            if (vt == 0) & ring_ok & (ch < L):
                bo.buffer_store(beta, _rsrc(beta_base), hv * L + ch)
        if const_expr(has_iw):
            iw_slot = _load_uniform(iw_idx, n)
            iw_base = iw + fx.Int64(fx.max(iw_slot, 0)) * fx.Int64(iw_line) * 2
            iw_ok = live & (iw_slot >= 0)
            if iw_ok & qk_owner:
                for t in range_constexpr(TT):
                    for j in range_constexpr(3):
                        at = t * iw_step + qk_ch + j * iw_win
                        bo.buffer_store(xw[t + 1 + j], _rsrc(iw_base), at)
            if iw_ok & has_v:
                for t in range_constexpr(TT):
                    for j in range_constexpr(3):
                        at = t * iw_step + v_ch + j * iw_win
                        bo.buffer_store(vw[t + 1 + j], _rsrc(iw_base), at)
        # l2 norm sums of squares: waves 0-1 hold the q rows, waves 2-3 the k rows. The
        # norms are applied inside the recurrence instead of rewriting q / k.
        ss = [_wave_sum(qk[t] * qk[t]) for t in range(TT)]
        if tid % 64 == 0:
            for t in range_constexpr(TT):
                fx.ptr_store(ss[t], s_ss + ((tid // 64) * TT + t))
        gpu.barrier()

        alive = h0 >= 0
        hsf = [
            [
                alive.select(fx.Float32(hs[j][i // 4][i % 4]), fx.Float32(0.0))
                for i in range(4 * _KG)
            ]
            for j in range(VPT)
        ]
        v4 = fx.Vector.make_type(4, fx.Float32)
        beta_t = [fx.ptr_load(s_beta + t) for t in range(TT)]
        out = [[None] * TT for _ in range(VPT)]
        for t in range_constexpr(TT):
            rk = _rsq(
                fx.ptr_load(s_ss + (2 * TT + t))
                + fx.ptr_load(s_ss + (3 * TT + t))
                + fx.Float32(1e-6)
            )
            rq = _rsq(
                fx.ptr_load(s_ss + t) + fx.ptr_load(s_ss + (TT + t)) + fx.Float32(1e-6)
            ) * fx.Float32(scale)
            d, k, q = [], [], []
            for i in range_constexpr(_KG):
                off = t * _KD + (i * _KSPLIT + ks) * 4
                fd = fx.Vector(fx.ptr_load(s_dec + off, result_type=v4))
                fk = fx.Vector(fx.ptr_load(s_qk + (TT * _KD + off), result_type=v4))
                fq = fx.Vector(fx.ptr_load(s_qk + off, result_type=v4))
                d += [fd[e] for e in range(4)]
                k += [fk[e] for e in range(4)]
                q += [fq[e] for e in range(4)]
            if const_expr(LAG and t >= LAG):
                # Without this the scheduler defers every token's q dot products to the
                # end and keeps their operands live: 340 VGPRs and AGPR copies at BVC 64.
                d[0] = _after(d[0], [out[j][t - LAG] for j in range(VPT)])
            pk = []
            for j in range_constexpr(VPT):
                acc = [fx.Float32(0.0)] * 4
                for i in range_constexpr(4 * _KG):
                    hsf[j][i] = hsf[j][i] * d[i]
                    acc[i % 4] = fmath.fma(hsf[j][i], k[i], acc[i % 4])
                pk.append((acc[0] + acc[2]) + (acc[1] + acc[3]))
            pk = [_sum8(p) for p in pk]
            for j in range_constexpr(VPT):
                # h += (k * rk) * ((v - rk * (h . k)) * beta)
                vv = fx.ptr_load(s_v + (t * BVC + cg + _NCG * j))
                u = fmath.fma(-pk[j], rk, vv) * beta_t[t] * rk
                acc = [fx.Float32(0.0)] * 4
                for i in range_constexpr(4 * _KG):
                    hsf[j][i] = fmath.fma(k[i], u, hsf[j][i])
                    acc[i % 4] = fmath.fma(hsf[j][i], q[i], acc[i % 4])
                out[j][t] = (acc[0] + acc[2]) + (acc[1] + acc[3])
            for j in range_constexpr(VPT):
                out[j][t] = _sum8(out[j][t]) * rq
        # every lane of a column group holds the sums; lane ks stores token ks
        for j in range_constexpr(VPT):
            val = out[j][0]
            for t in range_constexpr(1, TT):
                val = (ks == t).select(out[j][t], val)
            if live & (ks < TT):
                at = ((bos + ks) * HV + hv) * _KD + v0 + cg + _NCG * j
                bo.buffer_store(val.to(fx.BFloat16), _rsrc(o), at)

    @flyc.jit
    def launch(
        x: Int64,
        w: Int64,
        cs: Int64,
        cs_idx: Int64,
        iw: Int64,
        iw_idx: Int64,
        a: Int64,
        b: Int64,
        A_log: Int64,
        dt_bias: Int64,
        state: Int64,
        h0_idx: Int64,
        o: Int64,
        rawv: Int64,
        rawk: Int64,
        ring_g: Int64,
        ring_beta: Int64,
        ring_idx: Int64,
        x_stride: Int32,
        cs_line: Int32,
        cs_tok: Int32,
        iw_line: Int32,
        iw_step: Int32,
        iw_win: Int32,
        a_stride: Int32,
        b_stride: Int32,
        rawv_slot: Int32,
        rawk_slot: Int32,
        g_slot: Int32,
        beta_slot: Int32,
        L: Int32,
        blocks: Int32,
        stream: fx.Stream,
    ):
        kernel(
            x,
            w,
            cs,
            cs_idx,
            iw,
            iw_idx,
            a,
            b,
            A_log,
            dt_bias,
            state,
            h0_idx,
            o,
            rawv,
            rawk,
            ring_g,
            ring_beta,
            ring_idx,
            x_stride,
            cs_line,
            cs_tok,
            iw_line,
            iw_step,
            iw_win,
            a_stride,
            b_stride,
            rawv_slot,
            rawk_slot,
            g_slot,
            beta_slot,
            L,
        ).launch(grid=(_KD // BVC, blocks, 1), block=(_THREADS, 1, 1), stream=stream)

    launch.compile_hints = {
        "llvm_options": {
            "amdgpu-expert-scheduling-mode": True,
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch
