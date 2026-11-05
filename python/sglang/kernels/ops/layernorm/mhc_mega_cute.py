"""CuTe DSL fused mHC seam for DSV4.1 (hidden 5120, hc 4) on SM100-family GPUs; bitwise equal to deep_gemm.mega_mhc.

Same contract as mhc_mega.mhc_mega_boundary in its shifted form with an MXFP8 output: the post (new residual), the
next hyper-connection's mix stats + Sinkhorn, and its pre-collapse + RMSNorm (bf16 rows and MXFP8 with UE8M0 scales,
DeepGEMM-packed or in FlashInfer's 128x4 layout, as the scale tensor's rank selects).

One persistent CTA per SM walks (128-row block, K split) tasks: 16 splits of 320 hidden columns per block, the split
structure DeepGEMM uses with 64-row blocks. Per split: TMA residual / x -> post (new residual, and
X1 = sum_o pre_o * R_o) -> TF32 UMMA of the new residual against fn (M=128, N=24, the same 8-column chunk order) ->
split partials to global memory -> per-token Sinkhorn mix once a block's 16 splits have landed. Every reduction replays
DeepGEMM's order (split partials, lane butterflies, sequential split sums), and an UMMA's per-row result does not
depend on M or on the A operand's source, so all outputs match bit for bit.
What differs from DeepGEMM is where X1 lives: DeepGEMM stores it to y_bf16 and re-reads it after the split barrier;
here each CTA keeps its 128 x 320 X1 slice in TMEM and normalizes it in place once the block's 16 per-split sums of
squares are published, staging the outputs through shared memory into whole-row-segment stores.
"""

import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Boolean, Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import GridConstant
from cutlass.cute.experimental import iket
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda.tensor_map import (
    TensorMap,
    TensorMapSwizzle,
    create_tensor_map_tiled,
)

ptx = cute.arch.inline_ptx


def _pure(res_type, asm: str, cons: str, args):
    return llvm.inline_asm(
        res_type, args, asm, cons, has_side_effects=False, is_align_stack=False
    )


@dsl_user_op
def smem_u32(p, *, loc=None, ip=None) -> Int32:
    return Int32(p.toint(loc=loc, ip=ip))


@dsl_user_op
def mbar_init(addr: Int32, count, *, loc=None, ip=None):
    ptx(
        "mbarrier.init.shared::cta.b64 [{$r0}], {$r1};",
        read_only_args=[addr, Int32(count)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fence_mbar_init(*, loc=None, ip=None):
    ptx("fence.mbarrier_init.release.cluster;", loc=loc, ip=ip)


@dsl_user_op
def mbar_arrive(addr: Int32, *, loc=None, ip=None):
    ptx(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [{$r0}];",
        read_only_args=[addr],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_arrive_expect_tx(addr: Int32, nbytes, *, loc=None, ip=None):
    ptx(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [{$r0}], {$r1};",
        read_only_args=[addr, Int32(nbytes)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_wait(addr: Int32, phase, *, loc=None, ip=None):
    """Spin until the barrier's current phase (parity `phase`) completes."""
    ptx(
        "{\n\t.reg .pred p;\n\tLAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [{$r0}], {$r1};\n\t"
        "@!p bra LAB_WAIT;\n\t}",
        read_only_args=[addr, Int32(phase)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bar_sync(bar_id, nthreads, *, loc=None, ip=None):
    ptx(
        "bar.sync {$r0}, {$r1};",
        read_only_args=[Int32(bar_id), Int32(nthreads)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def syncwarp(*, loc=None, ip=None):
    ptx("bar.warp.sync -1;", loc=loc, ip=ip)


@dsl_user_op
def fence_proxy_async_smem(*, loc=None, ip=None):
    ptx("fence.proxy.async.shared::cta;", loc=loc, ip=ip)


@dsl_user_op
@dsl_user_op
def tma_load_2d(desc: Int64, smem: Int32, mbar: Int32, c0, c1, *, loc=None, ip=None):
    ptx(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[{$r0}], [{$r1}, {{$r3}, {$r4}}], [{$r2}];",
        read_only_args=[smem, desc, mbar, Int32(c0), Int32(c1)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_load_3d(
    desc: Int64, smem: Int32, mbar: Int32, c0, c1, c2, *, loc=None, ip=None
):
    ptx(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[{$r0}], [{$r1}, {{$r3}, {$r4}, {$r5}}], [{$r2}];",
        read_only_args=[smem, desc, mbar, Int32(c0), Int32(c1), Int32(c2)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_store_3d(desc: Int64, smem: Int32, c0, c1, c2, *, loc=None, ip=None):
    ptx(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [{$r0}, {{$r2}, {$r3}, {$r4}}], [{$r1}];",
        read_only_args=[desc, smem, Int32(c0), Int32(c1), Int32(c2)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bulk_commit(*, loc=None, ip=None):
    ptx("cp.async.bulk.commit_group;", loc=loc, ip=ip)


@dsl_user_op
def bulk_wait_read0(*, loc=None, ip=None):
    ptx("cp.async.bulk.wait_group.read 0;", loc=loc, ip=ip)


@dsl_user_op
def bulk_wait0(*, loc=None, ip=None):
    ptx("cp.async.bulk.wait_group 0;", loc=loc, ip=ip)


@dsl_user_op
def ld_acquire_u64(addr: Int64, *, loc=None, ip=None) -> Int64:
    return ptx(
        "ld.acquire.gpu.global.b64 {$w0}, [{$r0}];",
        write_only_types=[Int64],
        read_only_args=[addr],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
@dsl_user_op
def red_release_add_u64(addr: Int64, val, *, loc=None, ip=None):
    ptx(
        "red.release.gpu.global.add.u64 [{$r0}], {$r1};",
        read_only_args=[addr, Int64(val)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def nanosleep(ns, *, loc=None, ip=None):
    ptx("nanosleep.u32 {$r0};", read_only_args=[Int32(ns)], loc=loc, ip=ip)


@dsl_user_op
def tmem_alloc(dst_smem: Int32, ncols, *, loc=None, ip=None):
    ptx(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [{$r0}], {$r1};",
        read_only_args=[dst_smem, Int32(ncols)],
        loc=loc,
        ip=ip,
    )
    ptx("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;", loc=loc, ip=ip)


@dsl_user_op
def tmem_dealloc(taddr: Int32, ncols, *, loc=None, ip=None):
    ptx(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 {$r0}, {$r1};",
        read_only_args=[taddr, Int32(ncols)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tc_fence_before(*, loc=None, ip=None):
    prims.tcgen05_fence("before_thread_sync", loc=loc, ip=ip)


@dsl_user_op
def tc_fence_after(*, loc=None, ip=None):
    prims.tcgen05_fence("after_thread_sync", loc=loc, ip=ip)


@dsl_user_op
def tc_wait_st(*, loc=None, ip=None):
    prims.tcgen05_wait("store", loc=loc, ip=ip)


@dsl_user_op
def tc_wait_ld(*, loc=None, ip=None):
    prims.tcgen05_wait("load", loc=loc, ip=ip)


def tptr(taddr, dtype=Float32):
    return prims.make_tmem_ptr(Int32(taddr), dtype)


@dsl_user_op
def tc_st(shape: str, taddr: Int32, vals, dtype=Float32, *, loc=None, ip=None):
    v = cute.make_rmem_tensor(len(vals), dtype)
    for i, x in enumerate(vals):
        v[i] = dtype(x)
    prims.tcgen05_st(shape, tptr(taddr, dtype), v.load(), loc=loc, ip=ip)


@dsl_user_op
def tc_ld(shape: str, taddr: Int32, num: int, dtype=Float32, *, loc=None, ip=None):
    return prims.tcgen05_ld(shape, tptr(taddr, dtype), num=num, loc=loc, ip=ip)


@dsl_user_op
@dsl_user_op
def mma_tf32_ts(
    d_taddr: Int32,
    a_taddr: Int32,
    bdesc: Int64,
    idesc: Int32,
    accumulate,
    *,
    loc=None,
    ip=None,
):
    prims.tcgen05_mma(
        "tf32",
        "cta_1",
        tptr(d_taddr),
        tptr(a_taddr),
        Int64(bdesc),
        Int32(idesc),
        Boolean(accumulate),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_commit(mbar: Int32, *, loc=None, ip=None):
    ptx(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [{$r0}];",
        read_only_args=[mbar],
        loc=loc,
        ip=ip,
    )


def idesc_tf32(M: int, N: int) -> int:
    """kind::tf32 instruction descriptor: F32 accum, TF32 A/B, both K-major."""
    return (1 << 4) | (2 << 7) | (2 << 10) | ((N >> 3) << 17) | ((M >> 4) << 24)


@dsl_user_op
def smem_desc(
    addr: Int32, lbo: int, sbo: int, swizzle: int, *, loc=None, ip=None
) -> Int64:
    """tcgen05 smem descriptor: start>>4 [0,14), LBO>>4 [16,30), SBO>>4 [32,46), version 1 [46], layout [61,64)."""
    hi = (sbo >> 4) | (1 << 14) | (swizzle << 29)
    lo = Int64((addr >> 4) & 0x3FFF) | Int64((lbo >> 4) << 16)
    return lo | (Int64(hi) << 32)


@dsl_user_op
def bf16x2_pack(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """__float22bfloat162_rn({lo, hi})."""
    return Uint32(
        _pure(
            T.i32(),
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            [
                Float32(hi).ir_value(loc=loc, ip=ip),
                Float32(lo).ir_value(loc=loc, ip=ip),
            ],
        )
    )


@dsl_user_op
def bf16_lo(u: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _pure(
            T.f32(), "shl.b32 $0, $1, 16;", "=f,r", [Uint32(u).ir_value(loc=loc, ip=ip)]
        )
    )


@dsl_user_op
def bf16_hi(u: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _pure(
            T.f32(),
            "and.b32 $0, $1, 0xffff0000;",
            "=f,r",
            [Uint32(u).ir_value(loc=loc, ip=ip)],
        )
    )


@dsl_user_op
def fma_rn(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _pure(
            T.f32(),
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c)],
        )
    )


@dsl_user_op
def mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _pure(
            T.f32(),
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b)],
        )
    )


@dsl_user_op
def add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _pure(
            T.f32(),
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b)],
        )
    )


def fma2(a, b, c):
    """__ffma2_rn on (x, y) tuples."""
    return cute.arch.fma_packed_f32x2(a, b, c, rnd="rn")


def mul2(a, b):
    return cute.arch.mul_packed_f32x2(a, b, rnd="rn")


@dsl_user_op
def rsqrt_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _pure(
            T.f32(),
            "rsqrt.approx.f32 $0, $1;",
            "=f,f",
            [Float32(x).ir_value(loc=loc, ip=ip)],
        )
    )


@dsl_user_op
def ex2_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _pure(
            T.f32(),
            "ex2.approx.f32 $0, $1;",
            "=f,f",
            [Float32(x).ir_value(loc=loc, ip=ip)],
        )
    )


@dsl_user_op
def rcp_approx_ftz(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _pure(
            T.f32(),
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            [Float32(x).ir_value(loc=loc, ip=ip)],
        )
    )


@dsl_user_op
def bf16x2_absmax(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """__hmax2(__habs2(a), __habs2(b))."""
    return Uint32(
        _pure(
            T.i32(),
            "{\n\t.reg .b32 x, y;\n\tabs.bf16x2 x, $1;\n\tabs.bf16x2 y, $2;\n\tmax.bf16x2 $0, x, y;\n\t}",
            "=r,r,r",
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
        )
    )


@dsl_user_op
def bf16x2_max(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _pure(
            T.i32(),
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
        )
    )


@dsl_user_op
def bf16x2_mul(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _pure(
            T.i32(),
            "mul.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
        )
    )


@dsl_user_op
def e4m3x2_from_bf16x2(u: Uint32, *, loc=None, ip=None) -> Uint32:
    """__nv_cvt_float2_to_fp8x2(__bfloat1622float2(u), satfinite, e4m3) in the low 16 bits."""
    return Uint32(
        _pure(
            T.i32(),
            "{\n\t.reg .b32 lo, hi;\n\t.reg .b16 r;\n\tshl.b32 lo, $1, 16;\n\t"
            "and.b32 hi, $1, 0xffff0000;\n\tcvt.rn.satfinite.e4m3x2.f32 r, hi, lo;\n\tcvt.u32.u16 $0, r;\n\t}",
            "=r,r",
            [Uint32(u).ir_value(loc=loc, ip=ip)],
        )
    )


def shfl_xor_f32(v: Float32, m: int) -> Float32:
    return cute.arch.shuffle_sync_bfly(Float32(v), offset=m)


@dsl_user_op
def lds128(addr: Int32, *, loc=None, ip=None):
    return ptx(
        "ld.shared.v4.u32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Uint32] * 4,
        read_only_args=[addr],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts128(addr: Int32, v0, v1, v2, v3, *, loc=None, ip=None):
    ptx(
        "st.shared.v4.u32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[addr, Uint32(v0), Uint32(v1), Uint32(v2), Uint32(v3)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ldg128_nc(addr: Int64, *, loc=None, ip=None):
    return ptx(
        "ld.global.nc.v4.u32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Uint32] * 4,
        read_only_args=[addr],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg128(addr: Int64, v0, v1, v2, v3, *, loc=None, ip=None):
    ptx(
        "st.global.v4.u32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[addr, Uint32(v0), Uint32(v1), Uint32(v2), Uint32(v3)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg16(addr: Int64, v, *, loc=None, ip=None):
    ptx(
        "{\n\t.reg .b16 t;\n\tcvt.u16.u32 t, {$r1};\n\tst.global.u16 [{$r0}], t;\n\t}",
        read_only_args=[addr, Uint32(v)],
        loc=loc,
        ip=ip,
    )


H = 5120
NR = 4
NOUT = 24
BM = 128
BK = 64
NSPLIT = 16
KPS = H // BK // NSPLIT  # 5 K blocks per split
SPLIT_COLS = KPS * BK  # 320

S_IO, S_FN, S_A, S_ACC, S_CO = 2, 2, 4, 2, 2
RES_STAGE = NR * BM * BK * 2  # 65536, route-major [4][128][64] SW128
X_STAGE = BM * BK * 2  # 16384
FN_STAGE = (
    NR * NOUT * 128
)  # 12288: one 32-column atom, route-major [4][24][32 fp32] SW128
CO_ROW = 24  # floats per row: pre4 | post4 | comb16
CO_STAGE = BM * CO_ROW * 4
OFF_RES = 0
OFF_X = OFF_RES + S_IO * RES_STAGE
OFF_FN = OFF_X + S_IO * X_STAGE
OFF_CO = OFF_FN + S_FN * FN_STAGE
OFF_ST = OFF_CO + S_CO * CO_STAGE  # hcn[2][128], x1s[2][128]
OFF_W = OFF_ST + 4 * BM * 4  # this task's RMSNorm weight slice (bf16 x 320)
NS_Y_ROW = 80  # staging row pitch (64 B of bf16 + 16 B pad keeps 16 B row stores conflict-free)
NS_F_ROW = 48  # 32 B of fp8 + 16 B pad
NS_WARP = 32 * (NS_Y_ROW + NS_F_ROW)
OFF_NS = (
    OFF_W + SPLIT_COLS * 2
)  # per-norm-warp output staging: [32 rows][80 B] bf16 | [32 rows][48 B] fp8
OFF_BAR = OFF_NS + 4 * NS_WARP
B_IO_FULL = 0
B_IO_EMPTY = B_IO_FULL + S_IO
B_ST_READY = B_IO_EMPTY + S_IO
B_FN_FULL = B_ST_READY + S_IO
B_FN_EMPTY = B_FN_FULL + S_FN
B_A_FULL = B_FN_EMPTY + S_FN
B_A_EMPTY = B_A_FULL + S_A
B_ACC_FULL = B_A_EMPTY + S_A
B_ACC_EMPTY = B_ACC_FULL + S_ACC
B_CO_FULL = B_ACC_EMPTY + S_ACC
B_CO_EMPTY = B_CO_FULL + S_CO
B_STATS_FULL = B_CO_EMPTY + S_CO
B_STATS_EMPTY = B_STATS_FULL + 1
B_X1_FULL = B_STATS_EMPTY + 1
B_X1_FREE = B_X1_FULL + 2
B_TMEM = B_X1_FREE + 2
SMEM_BYTES = OFF_BAR + (B_TMEM + 2) * 8

TM_A = 0  # S_A stages x 32 columns (4 routes x 8 hidden), lane = row
TM_ACC = S_A * 32  # S_ACC x 24
TM_X1 = 192  # 2 buffers x 160 columns: X1 bf16 pairs, column = kb * 32 + pair
X1_COLS = KPS * 32

NUM_THREADS = 768
CNT_STRIDE = 16  # u64 slots between counters (128 B lines)
# setmaxnreg budget per warpgroup (sum <= 480 = 6 WGs x 80): producers, post x2, drain, mix, norm
REG = [56, 112, 48, 48, 104]
# In-kernel event tracing ranges (run under run-iket); compiled out otherwise.
IKET_ON = os.environ.get("SGLANG_DSV41_MHC_CUTE_IKET", "0") == "1"


def _setreg(n):
    # Kernel launches at 80 registers per thread (768 threads); grow or shrink each warpgroup from there.
    if n > 80:
        cute.arch.setmaxregister_increase(n)
    else:
        cute.arch.setmaxregister_decrease(n)


def _bar(base, i):
    return base + OFF_BAR + i * 8


def _rs(name):
    return iket.range_start(name) if IKET_ON else None


def _re(tok):
    if IKET_ON:
        iket.range_end(tok)


@cute.jit
def load_coeff(sbase, lane, n_tok, pre_base, post_base, comb_base, task_c, slot):
    """Storer warp: one task's pre | post | comb rows into coefficient slot `slot % S_CO`."""
    mbar_wait(_bar(sbase, B_CO_EMPTY + slot % S_CO), ((slot // S_CO) & 1) ^ 1)
    mi = (task_c // NSPLIT) * BM
    for rr in cutlass.range_constexpr(BM // 32):
        r = lane + rr * 32
        tok = mi + r
        ob = sbase + OFF_CO + (slot % S_CO) * CO_STAGE + r * CO_ROW * 4
        if tok < n_tok:
            v = ldg128_nc(pre_base + Int64(tok) * 16)
            sts128(ob, v[0], v[1], v[2], v[3])
            v = ldg128_nc(post_base + Int64(tok) * 16)
            sts128(ob + 16, v[0], v[1], v[2], v[3])
            for j in cutlass.range_constexpr(4):
                v = ldg128_nc(comb_base + Int64(tok) * 64 + j * 16)
                sts128(ob + 32 + j * 16, v[0], v[1], v[2], v[3])
        else:
            for j in cutlass.range_constexpr(CO_ROW // 4):
                sts128(ob + j * 16, 0, 0, 0, 0)
    mbar_arrive(_bar(sbase, B_CO_FULL + slot % S_CO))


def deepgemm_uses_16_splits(rows: int, num_sms: int) -> bool:
    """Whether deep_gemm.mega_mhc splits K 16 ways at this row count (get_num_splits in its
    csrc/apis/mega_mhc.hpp: 64-row blocks, 64-column K blocks). This kernel always uses 16
    splits, so it matches DeepGEMM bit for bit only where DeepGEMM does too."""
    num_k_blocks = H // BK
    max_splits = min(64, max(NSPLIT, num_sms // -(-rows // 64)))
    return -(-num_k_blocks // -(-num_k_blocks // max_splits)) == NSPLIT


class MhcCute:
    def __init__(self, num_sms: int, swizzled_sf: bool):
        self.G = num_sms
        # Scales in FlashInfer's 128x4 layout; sf_stride_tok is then the stride between 128-row tiles.
        self.swizzled_sf = swizzled_sf

    @cute.jit
    def __call__(
        self,
        m_res: cute.Tensor,
        m_x: cute.Tensor,
        m_fn: cute.Tensor,
        m_newres: cute.Tensor,
        m_pre: cute.Tensor,
        m_post: cute.Tensor,
        m_comb: cute.Tensor,
        m_scale: cute.Tensor,
        m_base: cute.Tensor,
        m_weight: cute.Tensor,
        m_npre: cute.Tensor,
        m_npost: cute.Tensor,
        m_ncomb: cute.Tensor,
        m_y: cute.Tensor,
        m_yfp8: cute.Tensor,
        m_sf: cute.Tensor,
        m_wsg: cute.Tensor,
        m_wsh: cute.Tensor,
        m_wsx: cute.Tensor,
        m_cnt: cute.Tensor,
        n_tok: Int32,
        sf_stride_tok: Int32,
        sf_stride_word: Int32,
        hc_norm_eps: Float32,
        hc_pre_eps: Float32,
        hc_post_scale: Float32,
        sk_eps: Float32,
        sk_iters: Int32,
        rms_eps: Float32,
        rms_scale: Float32,
        stream: cuda.CUstream,
    ):
        d_res = create_tensor_map_tiled(
            m_res.iterator.toint(),
            cutlass.BFloat16,
            [H, n_tok, NR],
            [NR * H * 2 // 16, H * 2 // 16],
            [BK, BM, NR],
            swizzle=TensorMapSwizzle.s128b,
        )
        d_new = create_tensor_map_tiled(
            m_newres.iterator.toint(),
            cutlass.BFloat16,
            [H, n_tok, NR],
            [NR * H * 2 // 16, H * 2 // 16],
            [BK, BM, NR],
            swizzle=TensorMapSwizzle.s128b,
        )
        d_x = create_tensor_map_tiled(
            m_x.iterator.toint(),
            cutlass.BFloat16,
            [H, n_tok],
            [H * 2 // 16],
            [BK, BM],
            swizzle=TensorMapSwizzle.s128b,
        )
        # TF32 tensor map: TMA rounds fn to TF32 on load, as DeepGEMM's descriptor does.
        d_fn = create_tensor_map_tiled(
            m_fn.iterator.toint(),
            cutlass.TFloat32,
            [H, NOUT, NR],
            [NR * H * 4 // 16, H * 4 // 16],
            [32, NOUT, NR],
            swizzle=TensorMapSwizzle.s128b,
        )
        self.kernel(
            d_res,
            d_new,
            d_x,
            d_fn,
            m_pre,
            m_post,
            m_comb,
            m_scale,
            m_base,
            m_weight,
            m_npre,
            m_npost,
            m_ncomb,
            m_y,
            m_yfp8,
            m_sf,
            m_wsg,
            m_wsh,
            m_wsx,
            m_cnt,
            n_tok,
            sf_stride_tok,
            sf_stride_word,
            hc_norm_eps,
            hc_pre_eps,
            hc_post_scale,
            sk_eps,
            sk_iters,
            rms_eps,
            rms_scale,
        ).launch(
            grid=[self.G, 1, 1],
            block=[NUM_THREADS, 1, 1],
            smem=SMEM_BYTES,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        d_res: GridConstant[TensorMap],
        d_new: GridConstant[TensorMap],
        d_x: GridConstant[TensorMap],
        d_fn: GridConstant[TensorMap],
        m_pre: cute.Tensor,
        m_post: cute.Tensor,
        m_comb: cute.Tensor,
        m_scale: cute.Tensor,
        m_base: cute.Tensor,
        m_weight: cute.Tensor,
        m_npre: cute.Tensor,
        m_npost: cute.Tensor,
        m_ncomb: cute.Tensor,
        m_y: cute.Tensor,
        m_yfp8: cute.Tensor,
        m_sf: cute.Tensor,
        m_wsg: cute.Tensor,
        m_wsh: cute.Tensor,
        m_wsx: cute.Tensor,
        m_cnt: cute.Tensor,
        n_tok: Int32,
        sf_stride_tok: Int32,
        sf_stride_word: Int32,
        hc_norm_eps: Float32,
        hc_pre_eps: Float32,
        hc_post_scale: Float32,
        sk_eps: Float32,
        sk_iters: Int32,
        rms_eps: Float32,
        rms_scale: Float32,
    ):
        G = self.G
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        smem_ptr = cute.arch.get_dyn_smem(Uint8, alignment=1024)
        sbase = smem_u32(smem_ptr)
        s32 = cute.make_tensor(
            cute.recast_ptr(smem_ptr, dtype=Uint32), cute.make_layout(SMEM_BYTES // 4)
        )
        sf32 = cute.make_tensor(
            cute.recast_ptr(smem_ptr, dtype=Float32), cute.make_layout(SMEM_BYTES // 4)
        )
        num_m_blocks = (n_tok + BM - 1) // BM
        num_tasks = num_m_blocks * NSPLIT
        tag = Int64(0)  # counters arrive zeroed per call
        cnt_base = m_cnt.iterator.toint()

        if warp == 0:
            if lane == 0:
                for i in cutlass.range_constexpr(S_IO):
                    mbar_init(_bar(sbase, B_IO_FULL + i), 1)
                    mbar_init(_bar(sbase, B_IO_EMPTY + i), 1)
                    mbar_init(_bar(sbase, B_ST_READY + i), 8)
                for i in cutlass.range_constexpr(S_FN):
                    mbar_init(_bar(sbase, B_FN_FULL + i), 1)
                    mbar_init(_bar(sbase, B_FN_EMPTY + i), 1)
                for i in cutlass.range_constexpr(S_A):
                    mbar_init(_bar(sbase, B_A_FULL + i), 128)
                    mbar_init(_bar(sbase, B_A_EMPTY + i), 1)
                for i in cutlass.range_constexpr(S_ACC):
                    mbar_init(_bar(sbase, B_ACC_FULL + i), 1)
                    mbar_init(_bar(sbase, B_ACC_EMPTY + i), 4)
                for i in cutlass.range_constexpr(S_CO):
                    mbar_init(_bar(sbase, B_CO_FULL + i), 32)
                    mbar_init(_bar(sbase, B_CO_EMPTY + i), 8)
                mbar_init(_bar(sbase, B_STATS_FULL), 8)
                mbar_init(_bar(sbase, B_STATS_EMPTY), 4)
                for i in cutlass.range_constexpr(2):
                    mbar_init(_bar(sbase, B_X1_FULL + i), 8)
                    mbar_init(_bar(sbase, B_X1_FREE + i), 4)
                fence_mbar_init()
        if warp == 3:
            tmem_alloc(_bar(sbase, B_TMEM), 512)
        cute.arch.sync_threads()
        tbase = Int32(s32[(OFF_BAR + B_TMEM * 8) // 4])

        if warp == 0:
            # ---- TMA: residual (4 routes) + x per K block
            _setreg(REG[0])
            if lane == 0:
                it = Int32(0)
                for task in cutlass.range(bid, num_tasks, G, unroll=1):
                    m_idx = (task // NSPLIT) * BM
                    k0 = (task % NSPLIT) * SPLIT_COLS
                    for kb in cutlass.range(KPS, unroll=1):
                        st = it % S_IO
                        tk = _rs("t_wait_empty")
                        mbar_wait(_bar(sbase, B_IO_EMPTY + st), ((it // S_IO) & 1) ^ 1)
                        _re(tk)
                        fb = _bar(sbase, B_IO_FULL + st)
                        mbar_arrive_expect_tx(fb, RES_STAGE + X_STAGE)
                        tma_load_3d(
                            d_res.get_ptr().toint(),
                            sbase + OFF_RES + st * RES_STAGE,
                            fb,
                            k0 + kb * BK,
                            m_idx,
                            0,
                        )
                        tma_load_2d(
                            d_x.get_ptr().toint(),
                            sbase + OFF_X + st * X_STAGE,
                            fb,
                            k0 + kb * BK,
                            m_idx,
                        )
                        it += 1
        elif warp == 1:
            # ---- TMA: fn, one 32-column atom (4 UMMA chunks) per stage
            _setreg(REG[0])
            if lane == 0:
                it = Int32(0)
                for task in cutlass.range(bid, num_tasks, G, unroll=1):
                    k0 = (task % NSPLIT) * SPLIT_COLS
                    for kh in cutlass.range(2 * KPS, unroll=1):
                        st = it % S_FN
                        mbar_wait(_bar(sbase, B_FN_EMPTY + st), ((it // S_FN) & 1) ^ 1)
                        fb = _bar(sbase, B_FN_FULL + st)
                        mbar_arrive_expect_tx(fb, FN_STAGE)
                        tma_load_3d(
                            d_fn.get_ptr().toint(),
                            sbase + OFF_FN + st * FN_STAGE,
                            fb,
                            k0 + kh * 32,
                            0,
                            0,
                        )
                        it += 1
        elif warp == 2:
            # ---- MMA: D[128 rows, 24] += A(TMEM, 8 hidden of one route) x fn^T in DeepGEMM's chunk order
            _setreg(REG[0])
            if lane == 0:
                idesc = Int32(idesc_tf32(BM, NOUT))
                fit = Int32(0)
                ait = Int32(0)
                ti = Int32(0)
                for task in cutlass.range(bid, num_tasks, G, unroll=1):
                    acs = ti % S_ACC
                    mbar_wait(_bar(sbase, B_ACC_EMPTY + acs), ((ti // S_ACC) & 1) ^ 1)
                    tc_fence_after()
                    d_addr = tbase + TM_ACC + acs * NOUT
                    for kh in cutlass.range(2 * KPS, unroll=1):
                        fs = fit % S_FN
                        mbar_wait(_bar(sbase, B_FN_FULL + fs), (fit // S_FN) & 1)
                        tc_fence_after()
                        for a in cutlass.range_constexpr(4):
                            ast = ait % S_A
                            tk = _rs("m_wait_afull")
                            mbar_wait(_bar(sbase, B_A_FULL + ast), (ait // S_A) & 1)
                            _re(tk)
                            tc_fence_after()
                            for o in cutlass.range_constexpr(NR):
                                bdesc = smem_desc(
                                    sbase
                                    + OFF_FN
                                    + fs * FN_STAGE
                                    + o * (NOUT * 128)
                                    + a * 32,
                                    16,
                                    1024,
                                    2,
                                )
                                accum = (kh > 0) | (a > 0) | (o > 0)
                                mma_tf32_ts(
                                    d_addr,
                                    tbase + TM_A + ast * 32 + o * 8,
                                    bdesc,
                                    idesc,
                                    accum,
                                )
                            mma_commit(_bar(sbase, B_A_EMPTY + ast))
                            ait += 1
                        mma_commit(_bar(sbase, B_FN_EMPTY + fs))
                        fit += 1
                    mma_commit(_bar(sbase, B_ACC_FULL + acs))
                    ti += 1
        elif warp == 3:
            # ---- coefficients (ld.global -> smem ring, one task ahead) and new-residual TMA stores
            _setreg(REG[0])
            pre_base = m_pre.iterator.toint()
            post_base = m_post.iterator.toint()
            comb_base = m_comb.iterator.toint()

            if bid < num_tasks:
                load_coeff(
                    sbase, lane, n_tok, pre_base, post_base, comb_base, bid, Int32(0)
                )
            ti = Int32(0)
            it = Int32(0)
            for task in cutlass.range(bid, num_tasks, G, unroll=1):
                if task + G < num_tasks:
                    load_coeff(
                        sbase,
                        lane,
                        n_tok,
                        pre_base,
                        post_base,
                        comb_base,
                        task + G,
                        ti + 1,
                    )
                m_idx = (task // NSPLIT) * BM
                for kb in cutlass.range(KPS, unroll=1):
                    st = it % S_IO
                    tk = _rs("s_wait_ready")
                    mbar_wait(_bar(sbase, B_ST_READY + st), (it // S_IO) & 1)
                    _re(tk)
                    if lane == 0:
                        k0 = (task % NSPLIT) * SPLIT_COLS
                        tma_store_3d(
                            d_new.get_ptr().toint(),
                            sbase + OFF_RES + st * RES_STAGE,
                            k0 + kb * BK,
                            m_idx,
                            0,
                        )
                        bulk_commit()
                        bulk_wait_read0()
                        mbar_arrive(_bar(sbase, B_IO_EMPTY + st))
                    syncwarp()
                    it += 1
                ti += 1
            if lane == 0:
                bulk_wait0()
        elif warp < 12:
            # ---- Post: WG w owns the UMMA atoms of parity w; one row per thread, 4 column pairs per atom
            _setreg(REG[1])
            w = (warp - 4) // 4
            row = (warp % 4) * 32 + lane
            ti = Int32(0)
            it = Int32(0)
            ait = w
            for task in cutlass.range(bid, num_tasks, G, unroll=1):
                cs = ti % S_CO
                tk = _rs("p_wait_coeff")
                mbar_wait(_bar(sbase, B_CO_FULL + cs), (ti // S_CO) & 1)
                _re(tk)
                cob = (OFF_CO + cs * CO_STAGE) // 4 + row * CO_ROW
                pre = [sf32[cob + j] for j in range(NR)]
                pst = [sf32[cob + 4 + j] for j in range(NR)]
                cmb = [sf32[cob + 8 + j] for j in range(NR * NR)]
                syncwarp()
                if lane == 0:
                    mbar_arrive(_bar(sbase, B_CO_EMPTY + cs))
                xb = ti % 2
                tk = _rs("p_wait_x1free")
                mbar_wait(_bar(sbase, B_X1_FREE + xb), ((ti // 2) & 1) ^ 1)
                _re(tk)
                tc_fence_after()
                # per column-pair class c: hc-norm and X1 sums of squares (x, y halves)
                h0x, h0y, h1x, h1y, h2x, h2y, h3x, h3y = [Float32(0.0)] * 8
                q0x, q0y, q1x, q1y, q2x, q2y, q3x, q3y = [Float32(0.0)] * 8
                for kb in cutlass.range(KPS, unroll=1):
                    st = it % S_IO
                    tk = _rs("p_wait_io")
                    mbar_wait(_bar(sbase, B_IO_FULL + st), (it // S_IO) & 1)
                    _re(tk)
                    tk2 = _rs("p_kb")
                    rb = (OFF_RES + st * RES_STAGE) // 4
                    xbs = (OFF_X + st * X_STAGE) // 4
                    for i in cutlass.range_constexpr(4):
                        a = w + 2 * i
                        off = row * 32 + (
                            (a ^ (row & 7)) << 2
                        )  # 16-byte chunk a of the row, SW128
                        xv = list(lds128(sbase + (xbs + off) * 4))
                        rv = [
                            list(lds128(sbase + (rb + o * (RES_STAGE // 16) + off) * 4))
                            for o in range(NR)
                        ]
                        ast = ait % S_A
                        tk3 = _rs("p_wait_aempty")
                        mbar_wait(_bar(sbase, B_A_EMPTY + ast), ((ait // S_A) & 1) ^ 1)
                        _re(tk3)
                        tc_fence_after()
                        x1 = [(Float32(0.0), Float32(0.0))] * 4
                        hn = [(h0x, h0y), (h1x, h1y), (h2x, h2y), (h3x, h3y)]
                        for o in cutlass.range_constexpr(NR):
                            avals = []
                            bw = []
                            for q in cutlass.range_constexpr(4):
                                pv = mul2(
                                    (bf16_lo(xv[q]), bf16_hi(xv[q])), (pst[o], pst[o])
                                )
                                for ii in cutlass.range_constexpr(NR):
                                    cf = cmb[ii * NR + o]
                                    pv = fma2(
                                        (bf16_lo(rv[ii][q]), bf16_hi(rv[ii][q])),
                                        (cf, cf),
                                        pv,
                                    )
                                b = bf16x2_pack(pv[0], pv[1])
                                f = (bf16_lo(b), bf16_hi(b))
                                bw.append(b)
                                hn[q] = fma2(f, f, hn[q])
                                x1[q] = fma2(f, (pre[o], pre[o]), x1[q])
                                avals += [f[0], f[1]]
                            sts128(
                                sbase + (rb + o * (RES_STAGE // 16) + off) * 4,
                                bw[0],
                                bw[1],
                                bw[2],
                                bw[3],
                            )
                            tc_st("32x32b", tbase + TM_A + ast * 32 + o * 8, avals)
                        (h0x, h0y), (h1x, h1y), (h2x, h2y), (h3x, h3y) = hn
                        xw = [bf16x2_pack(x1[q][0], x1[q][1]) for q in range(4)]
                        qs = [(q0x, q0y), (q1x, q1y), (q2x, q2y), (q3x, q3y)]
                        for q in cutlass.range_constexpr(4):
                            fq = (bf16_lo(xw[q]), bf16_hi(xw[q]))
                            qs[q] = fma2(fq, fq, qs[q])
                        (q0x, q0y), (q1x, q1y), (q2x, q2y), (q3x, q3y) = qs
                        tc_st(
                            "32x32b",
                            tbase + TM_X1 + xb * X1_COLS + kb * 32 + 4 * a,
                            xw,
                            Uint32,
                        )
                        tc_wait_st()
                        tc_fence_before()
                        mbar_arrive(_bar(sbase, B_A_FULL + ast))
                        ait += 2
                    fence_proxy_async_smem()
                    syncwarp()
                    if lane == 0:
                        mbar_arrive(_bar(sbase, B_ST_READY + st))
                    _re(tk2)
                    it += 1
                tc_fence_before()
                syncwarp()
                if lane == 0:
                    mbar_arrive(_bar(sbase, B_X1_FULL + xb))
                # DeepGEMM: lanes c=0..3 of a row sum (x + y), then butterfly xor 2, xor 1
                hr = ((h0x + h0y) + (h2x + h2y)) + ((h1x + h1y) + (h3x + h3y))
                xr = ((q0x + q0y) + (q2x + q2y)) + ((q1x + q1y) + (q3x + q3y))
                tk = _rs("p_wait_stats")
                mbar_wait(_bar(sbase, B_STATS_EMPTY), (ti & 1) ^ 1)
                _re(tk)
                stb = OFF_ST // 4
                sf32[stb + w * BM + row] = hr
                sf32[stb + 2 * BM + w * BM + row] = xr
                syncwarp()
                if lane == 0:
                    mbar_arrive(_bar(sbase, B_STATS_FULL))
                ti += 1
        elif warp < 16:
            # ---- Drain: split statistics and accumulators -> global workspace, then publish
            _setreg(REG[2])
            row = (warp % 4) * 32 + lane
            ti = Int32(0)
            for task in cutlass.range(bid, num_tasks, G, unroll=1):
                mblk = task // NSPLIT
                cnt = cnt_base + Int64(mblk) * (CNT_STRIDE * 8)
                tk = _rs("d_wait_stats")
                mbar_wait(_bar(sbase, B_STATS_FULL), ti & 1)
                _re(tk)
                stb = OFF_ST // 4
                m_wsh[task * BM + row] = (Float32(0.0) + sf32[stb + row]) + sf32[
                    stb + BM + row
                ]
                m_wsx[task * BM + row] = (
                    Float32(0.0) + sf32[stb + 2 * BM + row]
                ) + sf32[stb + 3 * BM + row]
                syncwarp()
                if lane == 0:
                    mbar_arrive(_bar(sbase, B_STATS_EMPTY))
                bar_sync(1, 128)
                if warp == 12:
                    if lane == 0:
                        red_release_add_u64(cnt, 1)
                acs = ti % S_ACC
                tk = _rs("d_wait_acc")
                mbar_wait(_bar(sbase, B_ACC_FULL + acs), (ti // S_ACC) & 1)
                _re(tk)
                tc_fence_after()
                va = tc_ld("32x32b", tbase + TM_ACC + acs * NOUT, 16)
                vb = tc_ld("32x32b", tbase + TM_ACC + acs * NOUT + 16, 8)
                tc_wait_ld()
                valid = mblk * BM + row < n_tok
                gb = (task * BM + row) * NOUT
                for j in cutlass.range_constexpr(NOUT):
                    v = va[j] if j < 16 else vb[j - 16]
                    m_wsg[gb + j] = v if valid else Float32(0.0)
                tc_fence_before()
                syncwarp()
                if lane == 0:
                    mbar_arrive(_bar(sbase, B_ACC_EMPTY + acs))
                bar_sync(1, 128)
                if warp == 12:
                    if lane == 0:
                        red_release_add_u64(cnt + 64, 1)
                ti += 1
        elif warp < 20:
            # ---- Mix: one warp per token (DeepGEMM run_mix_task, shifted)
            _setreg(REG[3])
            wi = warp - 16
            for tok in cutlass.range(bid * 4 + wi, n_tok, G * 4, unroll=1):
                mblk = tok // BM
                r = tok % BM
                cnt = cnt_base + Int64(mblk) * (CNT_STRIDE * 8) + 64
                tk = _rs("x_spin")
                while ld_acquire_u64(cnt) != tag + NSPLIT:
                    nanosleep(64)
                _re(tk)
                tk2 = _rs("x_math")
                hsum = Float32(0.0)
                if lane < NSPLIT:
                    hsum = Float32(0.0) + m_wsh[(mblk * NSPLIT + lane) * BM + r]
                for off in cutlass.range_constexpr(5):
                    hsum = hsum + shfl_xor_f32(hsum, 16 >> off)
                safe = lane if lane < NOUT else NOUT - 1
                hc = Float32(0.0)
                for s in cutlass.range_constexpr(NSPLIT):
                    hc = hc + m_wsg[((mblk * NSPLIT + s) * BM + r) * NOUT + safe]
                if lane >= NOUT:
                    hc = Float32(0.0)
                rms = rsqrt_approx(fma_rn(hsum, Float32(1.0 / (NR * H)), hc_norm_eps))
                is_pre = lane < NR
                is_pp = lane < 2 * NR
                is_comb = (lane >= 2 * NR) & (lane < NOUT)
                mix = Float32(0.0)
                if lane < NOUT:
                    sidx = 0 if is_pre else (1 if is_pp else 2)
                    affine = fma_rn(mul_rn(hc, rms), m_scale[sidx], m_base[lane])
                    if is_pp:
                        sig = rcp_approx_ftz(
                            add_rn(
                                Float32(1.0),
                                ex2_approx(
                                    mul_rn(affine, Float32(-1.4426950216293334961))
                                ),
                            )
                        )
                        mix = (
                            add_rn(sig, hc_pre_eps)
                            if is_pre
                            else mul_rn(sig, hc_post_scale)
                        )
                    else:
                        mix = affine
                v = mix if is_comb else Float32(0.0)
                rmax = cute.arch.fmax(v, shfl_xor_f32(v, 2))
                rmax = cute.arch.fmax(rmax, shfl_xor_f32(rmax, 1))
                v = ex2_approx(mul_rn(add_rn(v, -rmax), Float32(1.4426950216293334961)))
                rsum = v + shfl_xor_f32(v, 2)
                rsum = rsum + shfl_xor_f32(rsum, 1)
                v = fma_rn(v, rcp_approx_ftz(rsum), sk_eps)
                csum = v + shfl_xor_f32(v, NR)
                csum = csum + shfl_xor_f32(csum, 6 * NR)
                v = mul_rn(v, rcp_approx_ftz(add_rn(csum, sk_eps)))
                for _ in cutlass.range(sk_iters - 1, unroll=1):
                    rsum = v + shfl_xor_f32(v, 2)
                    rsum = rsum + shfl_xor_f32(rsum, 1)
                    v = mul_rn(v, rcp_approx_ftz(add_rn(rsum, sk_eps)))
                    csum = v + shfl_xor_f32(v, NR)
                    csum = csum + shfl_xor_f32(csum, 6 * NR)
                    v = mul_rn(v, rcp_approx_ftz(add_rn(csum, sk_eps)))
                if is_pre:
                    m_npre[tok * NR + lane] = mix
                elif is_pp:
                    m_npost[tok * NR + lane - NR] = mix
                elif is_comb:
                    m_ncomb[tok * NR * NR + lane - 2 * NR] = v
                _re(tk2)
        else:
            # ---- Norm: this CTA's TMEM X1 slice -> RMSNorm -> bf16 + MXFP8 (UE8M0 per 32 columns)
            _setreg(REG[4])
            w_base = m_weight.iterator.toint()
            y_base = m_y.iterator.toint()
            f8_base = m_yfp8.iterator.toint()
            sf_base = m_sf.iterator.toint()
            row = (warp % 4) * 32 + lane
            ti = Int32(0)
            for task in cutlass.range(bid, num_tasks, G, unroll=1):
                mblk = task // NSPLIT
                split = task % NSPLIT
                xb = ti % 2
                tk = _rs("n_wait_x1")
                mbar_wait(_bar(sbase, B_X1_FULL + xb), (ti // 2) & 1)
                _re(tk)
                tc_fence_after()
                cnt = cnt_base + Int64(mblk) * (CNT_STRIDE * 8)
                tk = _rs("n_spin")
                while ld_acquire_u64(cnt) != tag + NSPLIT:
                    nanosleep(32)
                _re(tk)
                tk2 = _rs("n_task")
                # this split's weight slice -> smem (rows broadcast from it)
                bar_sync(2, 128)
                if row < SPLIT_COLS * 2 // 16:
                    v = ldg128_nc(w_base + Int64(split * SPLIT_COLS * 2 + row * 16))
                    sts128(sbase + OFF_W + row * 16, v[0], v[1], v[2], v[3])
                bar_sync(2, 128)
                tok = mblk * BM + row
                tok0 = mblk * BM + (warp % 4) * 32
                ns = sbase + OFF_NS + (warp % 4) * NS_WARP
                tk3 = _rs("n_rms")
                acc = Float32(0.0)
                for s in cutlass.range_constexpr(NSPLIT):
                    acc = acc + m_wsx[(mblk * NSPLIT + s) * BM + row]
                rms = mul_rn(
                    rsqrt_approx(fma_rn(acc, Float32(1.0 / H), rms_eps)), rms_scale
                )
                _re(tk3)
                for kb in cutlass.range(KPS, unroll=1):
                    col0 = split * SPLIT_COLS + kb * BK
                    tk4 = _rs("n_tmem")
                    xw = tc_ld(
                        "32x32b", tbase + TM_X1 + xb * X1_COLS + kb * 32, 32, Uint32
                    )
                    tc_wait_ld()
                    _re(tk4)
                    tk5 = _rs("n_group")
                    # both 32-column groups of the K block computed together (independent chains)
                    ys = []
                    for p4 in cutlass.range_constexpr(8):
                        wt4 = lds128(sbase + OFF_W + kb * BK * 2 + p4 * 16)
                        for j in cutlass.range_constexpr(4):
                            p = 4 * p4 + j
                            t1 = mul2((bf16_lo(xw[p]), bf16_hi(xw[p])), (rms, rms))
                            t2 = mul2(t1, (bf16_lo(wt4[j]), bf16_hi(wt4[j])))
                            ys.append(bf16x2_pack(t2[0], t2[1]))
                    sfes = []
                    f8 = []
                    for g in cutlass.range_constexpr(2):
                        yg = ys[16 * g : 16 * g + 16]
                        # DeepGEMM get_packed_bf16_amax: |.|-max tree over the group's bf16 pairs, then max of halves
                        t = [bf16x2_absmax(yg[p], yg[p + 8]) for p in range(8)]
                        t = [bf16x2_max(t[p], t[p + 4]) for p in range(4)]
                        t = [bf16x2_max(t[p], t[p + 2]) for p in range(2)]
                        am = bf16x2_max(t[0], t[1])
                        lo = am & 0xFFFF
                        hi = am >> 16
                        amax = lo if bf16_lo(lo) >= bf16_lo(hi) else hi
                        rexp = (amax + 0x7F - 0x60) >> 7
                        sfe = (rexp if rexp > 113 else Uint32(113)) - 8
                        sinv = ((254 - sfe) << 7) & 0xFFFF
                        sinv2 = sinv | (sinv << 16)
                        sfes.append(sfe)
                        for p in cutlass.range_constexpr(8):
                            lo8 = e4m3x2_from_bf16x2(bf16x2_mul(yg[2 * p], sinv2))
                            hi8 = e4m3x2_from_bf16x2(bf16x2_mul(yg[2 * p + 1], sinv2))
                            f8.append(lo8 | (hi8 << 16))
                    if tok < n_tok:
                        # two UE8M0 bytes of one SF word (groups 2k, 2k+1 of a 128-column word)
                        cgs = col0
                        if cutlass.const_expr(self.swizzled_sf):
                            tok_word = (
                                Int64(tok >> 7) * sf_stride_tok
                                + (tok & 31) * 4
                                + ((tok >> 5) & 3)
                            )
                        else:
                            tok_word = Int64(tok) * sf_stride_tok
                        stg16(
                            sf_base
                            + (tok_word + (cgs // 128) * sf_stride_word) * 4
                            + (cgs // 32) % 4,
                            sfes[0] | (sfes[1] << 8),
                        )
                    for g in cutlass.range_constexpr(2):
                        cg = col0 + 32 * g
                        # stage this warp's 32 rows, then store whole 64 B / 32 B row segments
                        for p4 in cutlass.range_constexpr(4):
                            sts128(
                                ns + lane * NS_Y_ROW + p4 * 16,
                                ys[16 * g + 4 * p4],
                                ys[16 * g + 4 * p4 + 1],
                                ys[16 * g + 4 * p4 + 2],
                                ys[16 * g + 4 * p4 + 3],
                            )
                        for p4 in cutlass.range_constexpr(2):
                            sts128(
                                ns + 32 * NS_Y_ROW + lane * NS_F_ROW + p4 * 16,
                                f8[8 * g + 4 * p4],
                                f8[8 * g + 4 * p4 + 1],
                                f8[8 * g + 4 * p4 + 2],
                                f8[8 * g + 4 * p4 + 3],
                            )
                        syncwarp()
                        for k in cutlass.range_constexpr(4):
                            rr = 8 * k + lane // 4
                            v = lds128(ns + rr * NS_Y_ROW + (lane % 4) * 16)
                            if tok0 + rr < n_tok:
                                stg128(
                                    y_base
                                    + (Int64(tok0 + rr) * H + cg) * 2
                                    + (lane % 4) * 16,
                                    v[0],
                                    v[1],
                                    v[2],
                                    v[3],
                                )
                        for k in cutlass.range_constexpr(2):
                            rr = 16 * k + lane // 2
                            v = lds128(
                                ns + 32 * NS_Y_ROW + rr * NS_F_ROW + (lane % 2) * 16
                            )
                            if tok0 + rr < n_tok:
                                stg128(
                                    f8_base
                                    + Int64(tok0 + rr) * H
                                    + cg
                                    + (lane % 2) * 16,
                                    v[0],
                                    v[1],
                                    v[2],
                                    v[3],
                                )
                        syncwarp()
                    _re(tk5)
                tc_fence_before()
                syncwarp()
                if lane == 0:
                    mbar_arrive(_bar(sbase, B_X1_FREE + xb))
                _re(tk2)
                ti += 1

        cute.arch.sync_threads()
        if warp == 3:
            tmem_dealloc(tbase, 512)


_KERNELS = {}


def _fake(dtype):
    return cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
    )


def _compiled(num_sms, swizzled_sf):
    """Compile once per (SM count, scale layout); TVM FFI takes torch tensors and the current torch stream directly."""
    key = (num_sms, swizzled_sf)
    if key not in _KERNELS:
        dts = [
            cutlass.BFloat16,
            cutlass.BFloat16,
            Float32,
            cutlass.BFloat16,
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Int32,
            Float32,
            Float32,
            Float32,
            Int32,
            Int32,
            Uint8,
            Float32,
            Float32,
            Float32,
            Int64,
        ]
        args = [_fake(d) for d in dts] + [
            Int32(0),
            Int32(0),
            Int32(0),
            Float32(0),
            Float32(0),
            Float32(0),
            Float32(0),
            Int32(0),
            Float32(0),
            Float32(0),
        ]
        _KERNELS[key] = cute.compile(
            MhcCute(num_sms, swizzled_sf),
            *args,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return _KERNELS[key]


_SMS = {}


def num_sms(device: torch.device) -> int:
    if device not in _SMS:
        _SMS[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SMS[device]


def mhc_mega_boundary_cute(
    x,
    residual,
    pre,
    post,
    comb,
    fn,
    scale,
    base,
    weight,
    rms_eps,
    hc_eps,
    norm_eps,
    sinkhorn_iters,
    y_fp8,
    y_gemm_sf,
    moe_input=None,
):
    """Drop-in for mhc_mega_boundary (shifted, MXFP8 output required). y_gemm_sf is DeepGEMM-packed
    ([rows, H / 128] int32, TMA-aligned column-major) or FlashInfer's 128x4 ([ceil(rows / 128), H / 128, 32, 4])."""
    assert moe_input is None, "the Mega MoE input seam keeps DeepGEMM"
    rows = x.shape[0]
    dev = x.device
    sms = num_sms(dev)
    assert deepgemm_uses_16_splits(rows, sms), (rows, sms)
    swizzled_sf = y_gemm_sf.dim() == 4
    if swizzled_sf:
        assert (
            y_gemm_sf.shape == (-(-rows // 128), H // 128, 32, 4)
            and y_gemm_sf.is_contiguous()
        ), y_gemm_sf.shape
    tasks = (rows + BM - 1) // BM * NSPLIT
    updated = torch.empty_like(residual)
    normalized = torch.empty_like(x)
    npre = torch.empty((rows, NR), device=dev, dtype=torch.float32)
    npost = torch.empty_like(npre)
    ncomb = torch.empty_like(comb)
    ws = torch.empty(tasks * BM * (NOUT + 2), device=dev, dtype=torch.float32)
    # per m-block arrival counters (norm at +0 B, mix at +64 B of a 128 B line)
    cnt = torch.zeros(tasks // NSPLIT * CNT_STRIDE, device=dev, dtype=torch.int64)
    sf = y_gemm_sf
    sf_bytes = torch.empty(0, dtype=torch.uint8, device=dev).set_(
        sf.untyped_storage(),
        sf.storage_offset() * 4,
        (sf.untyped_storage().nbytes() - sf.storage_offset() * 4,),
        (1,),
    )
    _compiled(sms, swizzled_sf)(
        residual.view(-1),
        x.view(-1),
        fn.view(-1),
        updated.view(-1),
        pre.contiguous().view(-1),
        post.contiguous().view(-1),
        comb.contiguous().view(-1),
        scale,
        base,
        weight.view(torch.int32),
        npre.view(-1),
        npost.view(-1),
        ncomb.view(-1),
        normalized.view(torch.int32).view(-1),
        y_fp8.view(torch.int32).view(-1),
        sf_bytes,
        ws[: tasks * BM * NOUT],
        ws[tasks * BM * NOUT : tasks * BM * (NOUT + 1)],
        ws[tasks * BM * (NOUT + 1) :],
        cnt,
        rows,
        sf.stride(0),
        sf.stride(1),
        rms_eps,
        hc_eps,
        2.0,
        hc_eps,
        sinkhorn_iters,
        norm_eps,
        1.0,
    )
    return updated, normalized, (npre, npost, ncomb)
