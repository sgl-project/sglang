# Vendored from the NVIDIA KDA_prefill package (benchmark/ Blackwell path)
# for the Kimi-K3 chunked prefill forward. Local deltas: fla.* imports
# re-pointed to sglang's vendored fla subset, flat sibling imports made
# package-relative, RCP_LN2 inlined. INTERNAL COLLABORATION ONLY.
# ruff: noqa  -- vendored kernel library, minimal local deltas
"""
Persistent Fused K1+K2+K3 Kernel for KDA.

Fuses gate activation + cumsum + scaling (K1), intra sub-chunk Aqk/Akk (K2),
and inter sub-chunk solve + merged inverse (K3) into a single persistent kernel.

Grid: (NUM_SMS, 1, 1) — 148 persistent blocks, each loops over work units
  Total work units = (NT/4) * H * B, distributed round-robin across SMs
  Block i processes work units i, i+NUM_SMS, i+2*NUM_SMS, ...
Block: 1024 threads (32 warps), warp-specialized with setmaxnreg (all groups 4-aligned):
  Warps 0-15:  TMA+K1 fused (8×2, vec2, prefetch pipeline) – 4 WGs, 56 regs
  Warps 16-27: K2 MMA compute (10 active + 2 idle for WG alignment) – 3 WGs, 72 regs
  Warps 28-31: Store/Inversion warps – 1 WG, 24 regs

Pipeline (single for_generate, warp groups separated by if-blocks):
  per work unit:
    Warps 0-15:  prefetch chunk 0→stage 0 (warp 0), then loop:
                    TMA next chunk (warp 0), wait cur chunk, K1 compute, arrive(k1_done)
    Warps 16-27: wait(k1_done)+wait(store_done), MMA, arrive(mma_done+stage_reuse)
    Warps 28-31: wait(mma_done), store sAqk/sAkk→GMEM, arrive(store_done)
  All warp-group invariants are computed inside each group's if-block (not hoisted)
  to eliminate cross-group register pressure — same budget as the _all version.
  Mbarrier phases self-reset after 4 iterations (2 stages × 2 phases).

Mbarriers:
  tma_mbars[2]:          count=1, warp 0 lane 0 → K1+MMA wait for TMA data
  stage_reuse_mbars[2]:  count=384, MMA(12 warps) → warp 0 waits before TMA reuse
  k1_done_mbars[2]:      count=512, K1(16 warps) → MMA waits for g_cumsum ready
  mma_done_mbars[2]:     count=384, MMA(12 warps) → Store waits for sAqk/sAkk ready
  store_done_mbars[2]:   count=128, Store(4 warps) → MMA waits for sAqk/sAkk stage free

SMEM: ~215KB (q+k+g × [64,128] bf16 × 2 stages + g_cumsum [64,136] fp32 × 2 stages
      + sPartialLast [8,132] fp32 + sAqk [16,168,2] bf16
      + sAkk [64,72,2] fp32 block-transposed upper-tri layout)

Inputs:
  g       [B,T,H,K]   bf16  raw gate
  k       [B,T,H,K]   bf16
  q       [B,T,H,K]   bf16
  A_log   [H]          fp32  per-head log decay
  beta    [B,T,H]      bf16  used for Akk unit lower triangular
  scale                fp32  1/sqrt(K)

Outputs (g_cumsum stays in SMEM, not written to GMEM):
  k_scaled   [B,T,H,K]   bf16
  q_scaled   [B,T,H,K]   bf16
  kg         [B,T,H,K]   bf16
  gk_last_exp[B,NT,H,K]  fp32
  A_qk       [B,T,H,BT]  bf16  full merged (diagonal + off-diagonal)
  A_kk       [B,T,H,BT]  fp32  block-transposed upper-tri (input to akk_inv)
"""

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
from cutlass import for_generate, yield_out
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

BT = 64
BC = 16
K_DIM = 128
K_PAD = 8
K_STRIDE = K_DIM + K_PAD  # 136, padded row stride to avoid bank conflicts
CHUNKS_PER_BLOCK = 4
NUM_SMS = 148  # Persistent kernel: one resident block per SM

NUM_K1_TMA_WARPS = 16  # Warps 0-15:  K1 compute (4 warpgroups, 8×2) -- TMA offloaded
NUM_MMA_WARPS = (
    11  # Warps 16-26: MMA (10 active + 1 TMA producer, dropped idle warp 27)
)
NUM_MMA_ACTIVE = 10  # mma_warp 0..9: actual MMA work
TMA_WARP_ID = NUM_K1_TMA_WARPS + NUM_MMA_ACTIVE  # warp 26 = dedicated TMA producer
NUM_STORE_WARPS = 4  # Warps 28-31: Store/Inversion (1 warpgroup)
NUM_WARPS = NUM_K1_TMA_WARPS + NUM_MMA_WARPS + NUM_STORE_WARPS  # 32
THREADS = NUM_WARPS * 32  # 1024

NUM_SUB_CHUNKS = BT // BC  # 4
NUM_TILES = NUM_SUB_CHUNKS * (NUM_SUB_CHUNKS + 1) // 2  # 10 lower-tri tiles
MMA_K_TILE = 16
NUM_MMA_K_TILES = K_DIM // MMA_K_TILE  # 8  (bf16 m16n8k16)
AQK_TILE_PAD = 8
AQK_TILE_STRIDE = (
    BT + AQK_TILE_PAD
)  # 72 — sAqk now 64x72 row-major (same shape as sAkk)

AKK_PAD = 8
AKK_STRIDE = BT + AKK_PAD  # 72

# For fused akk_inv: sAkk_pkd viewed as fp32 packed bf16x2, stride = BT/2 + small pad
AKK_PKD_PAD = 4
AKK_PKD_STRIDE = BT // 2 + AKK_PKD_PAD  # 36 fp32 units = 72 bf16 elements per row
AKK_TEMP_PAD = 8
AKK_TEMP_COLS = 16 + AKK_TEMP_PAD  # 24
AKK_TEMP_BUFS = 2

K1_ROW_GROUPS = 8
K1_COL_GROUPS = 2
ROWS_PER_K1_WARP = BT // K1_ROW_GROUPS  # 8
K1_COLS_PER_WARP = K_DIM // K1_COL_GROUPS  # 64
ROWS_PER_STORE_WARP = BT // NUM_STORE_WARPS  # 16

VEC = K1_COLS_PER_WARP // 32  # 2
K_VEC = K_DIM // VEC  # 64
NUM_STAGES = 2
PARTIAL_COLS = K_DIM + 4  # 132
PARTIAL_COLS_PER_WARP = K_DIM // NUM_K1_TMA_WARPS  # 8

_TILE_IQ = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
_TILE_IK = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
RCP_LN2 = LOG2E


@dsl_user_op
def k1_internal_barrier(*, loc=None, ip=None):
    """Named barrier for K1+TMA warps (0-15, 512 threads). barrier_id=2."""
    llvm.inline_asm(
        T.i32(),
        [],
        "membar.cta; bar.sync 2, 512; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def pack_bf16x2_f32(hi_f32, lo_f32, *, loc=None, ip=None):
    """Pack two fp32 values into a bf16x2 u32 register.
    Returns u32 with bf16(hi) in bits [31:16] and bf16(lo) in bits [15:0].
    PTX: cvt.rn.bf16x2.f32 d, a, b  -> d[31:16]=bf16(a), d[15:0]=bf16(b)
    """
    result = llvm.inline_asm(
        T.i32(),
        [hi_f32.ir_value(loc=loc, ip=ip), lo_f32.ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


@dsl_user_op
def mma_bf16_m16n8k16(
    a0,
    a1,
    a2,
    a3,  # 4 u32 (bf16x2 packed) A operands
    b0,
    b1,  # 2 u32 (bf16x2 packed) B operands
    c0,
    c1,
    c2,
    c3,  # 4 fp32 accumulators
    *,
    loc=None,
    ip=None,
):
    """bf16 MMA with fp32 accumulator, shape m16n8k16.
    D_fp32 = A_bf16 * B_bf16 + C_fp32
    """
    # a/b already i32 (from pack_bf16x2_f32) -> no bitcast needed
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [
            a0.ir_value(loc=loc, ip=ip),
            a1.ir_value(loc=loc, ip=ip),
            a2.ir_value(loc=loc, ip=ip),
            a3.ir_value(loc=loc, ip=ip),
            b0.ir_value(loc=loc, ip=ip),
            b1.ir_value(loc=loc, ip=ip),
            c0.ir_value(loc=loc, ip=ip),
            c1.ir_value(loc=loc, ip=ip),
            c2.ir_value(loc=loc, ip=ip),
            c3.ir_value(loc=loc, ip=ip),
        ],
        """{
            mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
                {$0, $1, $2, $3},
                {$4, $5, $6, $7},
                {$8, $9},
                {$10, $11, $12, $13};
        }""",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    d0 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip))
    d1 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip))
    d2 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip))
    d3 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip))
    return d0, d1, d2, d3


SHFL_W8_CLAMP = 0x1800


@dsl_user_op
def fast_rcp(x, *, loc=None, ip=None):
    """Hardware fast reciprocal: rcp.approx.ftz.f32 (~2 cycles vs ~20 for div)."""
    result = llvm.inline_asm(
        T.f32(),
        [x.ir_value(loc=loc, ip=ip)],
        "rcp.approx.ftz.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(result)


# ============================================================
# Akk inverse helpers (fused from Akk_inverse_lower_triangle_bf16.py)
# ============================================================
@dsl_user_op
def store_internal_barrier(*, loc=None, ip=None):
    """Named barrier for Store warps (4 warps, 128 threads). barrier_id=3."""
    llvm.inline_asm(
        T.i32(),
        [],
        "membar.cta; bar.sync 3, 128; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ak_pack_bf16x2(lo_f32, hi_f32, *, loc=None, ip=None):
    """cvt.rn.bf16x2.f32 -- pack two fp32 into bf16x2 (as fp32 bitcast view)."""
    result = llvm.inline_asm(
        T.i32(),
        [lo_f32.ir_value(loc=loc, ip=ip), hi_f32.ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(llvm.bitcast(T.f32(), result, loc=loc, ip=ip))


@dsl_user_op
def _ak_movmatrix_trans(src, *, loc=None, ip=None):
    src_b = llvm.bitcast(T.i32(), src.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.i32(),
        [src_b],
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(llvm.bitcast(T.f32(), result, loc=loc, ip=ip))


@dsl_user_op
def _ak_mask_packed_ltri(packed, row, pair, *, loc=None, ip=None):
    """Mask packed bf16x2 to lower-tri: zero elements where row < col."""
    p = llvm.bitcast(T.i32(), packed.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.i32(),
        [p, row.ir_value(loc=loc, ip=ip), pair.ir_value(loc=loc, ip=ip)],
        """{
            .reg .b32 %c0, %c1, %mlo, %mhi, %mask;
            .reg .pred %p0, %p1;
            shl.b32 %c0, $3, 1;
            add.u32 %c1, %c0, 1;
            setp.ge.s32 %p0, $2, %c0;
            setp.ge.s32 %p1, $2, %c1;
            selp.b32 %mlo, 0xFFFF, 0, %p0;
            selp.b32 %mhi, 0xFFFF0000, 0, %p1;
            or.b32 %mask, %mlo, %mhi;
            and.b32 $0, $1, %mask;
        }""",
        "=r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(llvm.bitcast(T.f32(), result, loc=loc, ip=ip))


@dsl_user_op
def _ak_unpack_bf16x2_lo(packed, *, loc=None, ip=None):
    p = llvm.bitcast(T.i32(), packed.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.i32(),
        [p],
        "shl.b32 $0, $1, 16;",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(llvm.bitcast(T.f32(), result, loc=loc, ip=ip))


@dsl_user_op
def _ak_unpack_bf16x2_hi(packed, *, loc=None, ip=None):
    p = llvm.bitcast(T.i32(), packed.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.i32(),
        [p],
        "and.b32 $0, $1, 0xFFFF0000;",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(llvm.bitcast(T.f32(), result, loc=loc, ip=ip))


@dsl_user_op
def _ak_mma(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """BF16 MMA m16n8k16, args are Float32 (bitcast-viewed as i32 for bf16x2)."""
    a0b = llvm.bitcast(T.i32(), a0.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a1b = llvm.bitcast(T.i32(), a1.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a2b = llvm.bitcast(T.i32(), a2.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a3b = llvm.bitcast(T.i32(), a3.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b0b = llvm.bitcast(T.i32(), b0.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b1b = llvm.bitcast(T.i32(), b1.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [
            a0b,
            a1b,
            a2b,
            a3b,
            b0b,
            b1b,
            c0.ir_value(loc=loc, ip=ip),
            c1.ir_value(loc=loc, ip=ip),
            c2.ir_value(loc=loc, ip=ip),
            c3.ir_value(loc=loc, ip=ip),
        ],
        """{
            mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
                {$0, $1, $2, $3},
                {$4, $5, $6, $7},
                {$8, $9},
                {$10, $11, $12, $13};
        }""",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    d0 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip))
    d1 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip))
    d2 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip))
    d3 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip))
    return d0, d1, d2, d3


@dsl_user_op
def _ak_invert_diag_neumann(sAkk, block_idx, lane_id, *, loc=None, ip=None):
    """Invert 16x16 diag block via Neumann series (I+L)^-1 = (I-L)(I+L^2)(I+L^4)(I+L^8).
    Reads/writes sAkk as fp32 packed bf16x2 (stride 36). Each block handled by 1 warp.
    """
    r_off = block_idx * 16
    c_off = block_idx * 8  # 16 cols = 8 pairs
    gid = lane_id // 4
    tid = lane_id % 4

    packed0 = cutlass.Float32(sAkk[r_off + gid, c_off + tid])
    packed1 = cutlass.Float32(sAkk[r_off + gid + 8, c_off + tid])
    packed2 = cutlass.Float32(sAkk[r_off + gid, c_off + 4 + tid])
    packed3 = cutlass.Float32(sAkk[r_off + gid + 8, c_off + 4 + tid])

    A_f0 = _ak_unpack_bf16x2_lo(packed0)
    A_f1 = _ak_unpack_bf16x2_hi(packed0)
    A_f2 = _ak_unpack_bf16x2_lo(packed1)
    A_f3 = _ak_unpack_bf16x2_hi(packed1)
    A_f4 = _ak_unpack_bf16x2_lo(packed2)
    A_f5 = _ak_unpack_bf16x2_hi(packed2)
    A_f6 = _ak_unpack_bf16x2_lo(packed3)
    A_f7 = _ak_unpack_bf16x2_hi(packed3)

    _one = cutlass.Float32(1.0)
    _zero = cutlass.Float32(0.0)
    I_f0 = _one * cutlass.Float32(gid == 2 * tid) + _zero * cutlass.Float32(
        gid != 2 * tid
    )
    I_f1 = _one * cutlass.Float32(gid == 2 * tid + 1) + _zero * cutlass.Float32(
        gid != 2 * tid + 1
    )
    I_f2 = _one * cutlass.Float32(gid + 8 == 2 * tid) + _zero * cutlass.Float32(
        gid + 8 != 2 * tid
    )
    I_f3 = _one * cutlass.Float32(gid + 8 == 2 * tid + 1) + _zero * cutlass.Float32(
        gid + 8 != 2 * tid + 1
    )
    I_f4 = _one * cutlass.Float32(gid == 8 + 2 * tid) + _zero * cutlass.Float32(
        gid != 8 + 2 * tid
    )
    I_f5 = _one * cutlass.Float32(gid == 8 + 2 * tid + 1) + _zero * cutlass.Float32(
        gid != 8 + 2 * tid + 1
    )
    I_f6 = _one * cutlass.Float32(gid + 8 == 8 + 2 * tid) + _zero * cutlass.Float32(
        gid + 8 != 8 + 2 * tid
    )
    I_f7 = _one * cutlass.Float32(gid + 8 == 8 + 2 * tid + 1) + _zero * cutlass.Float32(
        gid + 8 != 8 + 2 * tid + 1
    )

    L_f0 = A_f0 - I_f0
    L_f1 = A_f1 - I_f1
    L_f2 = A_f2 - I_f2
    L_f3 = A_f3 - I_f3
    L_f4 = A_f4 - I_f4
    L_f5 = A_f5 - I_f5
    L_f6 = A_f6 - I_f6
    L_f7 = A_f7 - I_f7
    INV_f0 = I_f0 - L_f0
    INV_f1 = I_f1 - L_f1
    INV_f2 = I_f2 - L_f2
    INV_f3 = I_f3 - L_f3
    INV_f4 = I_f4 - L_f4
    INV_f5 = I_f5 - L_f5
    INV_f6 = I_f6 - L_f6
    INV_f7 = I_f7 - L_f7

    _zf = cutlass.Float32(0.0)
    L_a0 = _ak_pack_bf16x2(L_f0, L_f1)
    L_a1 = _ak_pack_bf16x2(L_f2, L_f3)
    L_a2 = _ak_pack_bf16x2(L_f4, L_f5)
    L_a3 = _ak_pack_bf16x2(L_f6, L_f7)
    INV_a0 = _ak_pack_bf16x2(INV_f0, INV_f1)
    INV_a1 = _ak_pack_bf16x2(INV_f2, INV_f3)
    INV_a2 = _ak_pack_bf16x2(INV_f4, INV_f5)
    INV_a3 = _ak_pack_bf16x2(INV_f6, INV_f7)

    # Iter 1: L^2, INV += INV*L^2
    L_b0 = _ak_movmatrix_trans(L_a0)
    L_b1 = _ak_movmatrix_trans(L_a1)
    L_b2 = _ak_movmatrix_trans(L_a2)
    L_b3 = _ak_movmatrix_trans(L_a3)
    Lp_c0, Lp_c1, Lp_c2, Lp_c3 = _ak_mma(
        L_a0, L_a1, L_a2, L_a3, L_b0, L_b1, _zf, _zf, _zf, _zf
    )
    Lp_c4, Lp_c5, Lp_c6, Lp_c7 = _ak_mma(
        L_a0, L_a1, L_a2, L_a3, L_b2, L_b3, _zf, _zf, _zf, _zf
    )
    Lp_a0 = _ak_pack_bf16x2(Lp_c0, Lp_c1)
    Lp_a1 = _ak_pack_bf16x2(Lp_c2, Lp_c3)
    Lp_a2 = _ak_pack_bf16x2(Lp_c4, Lp_c5)
    Lp_a3 = _ak_pack_bf16x2(Lp_c6, Lp_c7)
    Lp_b0 = _ak_movmatrix_trans(Lp_a0)
    Lp_b1 = _ak_movmatrix_trans(Lp_a1)
    Lp_b2 = _ak_movmatrix_trans(Lp_a2)
    Lp_b3 = _ak_movmatrix_trans(Lp_a3)
    mm_c0, mm_c1, mm_c2, mm_c3 = _ak_mma(
        INV_a0, INV_a1, INV_a2, INV_a3, Lp_b0, Lp_b1, _zf, _zf, _zf, _zf
    )
    mm_c4, mm_c5, mm_c6, mm_c7 = _ak_mma(
        INV_a0, INV_a1, INV_a2, INV_a3, Lp_b2, Lp_b3, _zf, _zf, _zf, _zf
    )
    INV_f0 = INV_f0 + mm_c0
    INV_f1 = INV_f1 + mm_c1
    INV_f2 = INV_f2 + mm_c2
    INV_f3 = INV_f3 + mm_c3
    INV_f4 = INV_f4 + mm_c4
    INV_f5 = INV_f5 + mm_c5
    INV_f6 = INV_f6 + mm_c6
    INV_f7 = INV_f7 + mm_c7
    INV_a0 = _ak_pack_bf16x2(INV_f0, INV_f1)
    INV_a1 = _ak_pack_bf16x2(INV_f2, INV_f3)
    INV_a2 = _ak_pack_bf16x2(INV_f4, INV_f5)
    INV_a3 = _ak_pack_bf16x2(INV_f6, INV_f7)

    # Iter 2: L^4, INV += INV*L^4
    L4_c0, L4_c1, L4_c2, L4_c3 = _ak_mma(
        Lp_a0, Lp_a1, Lp_a2, Lp_a3, Lp_b0, Lp_b1, _zf, _zf, _zf, _zf
    )
    L4_c4, L4_c5, L4_c6, L4_c7 = _ak_mma(
        Lp_a0, Lp_a1, Lp_a2, Lp_a3, Lp_b2, Lp_b3, _zf, _zf, _zf, _zf
    )
    L4_a0 = _ak_pack_bf16x2(L4_c0, L4_c1)
    L4_a1 = _ak_pack_bf16x2(L4_c2, L4_c3)
    L4_a2 = _ak_pack_bf16x2(L4_c4, L4_c5)
    L4_a3 = _ak_pack_bf16x2(L4_c6, L4_c7)
    L4_b0 = _ak_movmatrix_trans(L4_a0)
    L4_b1 = _ak_movmatrix_trans(L4_a1)
    L4_b2 = _ak_movmatrix_trans(L4_a2)
    L4_b3 = _ak_movmatrix_trans(L4_a3)
    mm_c0, mm_c1, mm_c2, mm_c3 = _ak_mma(
        INV_a0, INV_a1, INV_a2, INV_a3, L4_b0, L4_b1, _zf, _zf, _zf, _zf
    )
    mm_c4, mm_c5, mm_c6, mm_c7 = _ak_mma(
        INV_a0, INV_a1, INV_a2, INV_a3, L4_b2, L4_b3, _zf, _zf, _zf, _zf
    )
    INV_f0 = INV_f0 + mm_c0
    INV_f1 = INV_f1 + mm_c1
    INV_f2 = INV_f2 + mm_c2
    INV_f3 = INV_f3 + mm_c3
    INV_f4 = INV_f4 + mm_c4
    INV_f5 = INV_f5 + mm_c5
    INV_f6 = INV_f6 + mm_c6
    INV_f7 = INV_f7 + mm_c7
    INV_a0 = _ak_pack_bf16x2(INV_f0, INV_f1)
    INV_a1 = _ak_pack_bf16x2(INV_f2, INV_f3)
    INV_a2 = _ak_pack_bf16x2(INV_f4, INV_f5)
    INV_a3 = _ak_pack_bf16x2(INV_f6, INV_f7)

    # Iter 3: L^8, INV += INV*L^8
    L8_c0, L8_c1, L8_c2, L8_c3 = _ak_mma(
        L4_a0, L4_a1, L4_a2, L4_a3, L4_b0, L4_b1, _zf, _zf, _zf, _zf
    )
    L8_c4, L8_c5, L8_c6, L8_c7 = _ak_mma(
        L4_a0, L4_a1, L4_a2, L4_a3, L4_b2, L4_b3, _zf, _zf, _zf, _zf
    )
    L8_a0 = _ak_pack_bf16x2(L8_c0, L8_c1)
    L8_a1 = _ak_pack_bf16x2(L8_c2, L8_c3)
    L8_a2 = _ak_pack_bf16x2(L8_c4, L8_c5)
    L8_a3 = _ak_pack_bf16x2(L8_c6, L8_c7)
    L8_b0 = _ak_movmatrix_trans(L8_a0)
    L8_b1 = _ak_movmatrix_trans(L8_a1)
    L8_b2 = _ak_movmatrix_trans(L8_a2)
    L8_b3 = _ak_movmatrix_trans(L8_a3)
    mm_c0, mm_c1, mm_c2, mm_c3 = _ak_mma(
        INV_a0, INV_a1, INV_a2, INV_a3, L8_b0, L8_b1, _zf, _zf, _zf, _zf
    )
    mm_c4, mm_c5, mm_c6, mm_c7 = _ak_mma(
        INV_a0, INV_a1, INV_a2, INV_a3, L8_b2, L8_b3, _zf, _zf, _zf, _zf
    )
    INV_f0 = INV_f0 + mm_c0
    INV_f1 = INV_f1 + mm_c1
    INV_f2 = INV_f2 + mm_c2
    INV_f3 = INV_f3 + mm_c3
    INV_f4 = INV_f4 + mm_c4
    INV_f5 = INV_f5 + mm_c5
    INV_f6 = INV_f6 + mm_c6
    INV_f7 = INV_f7 + mm_c7

    sAkk[r_off + gid, c_off + tid] = _ak_pack_bf16x2(INV_f0, INV_f1)
    sAkk[r_off + gid + 8, c_off + tid] = _ak_pack_bf16x2(INV_f2, INV_f3)
    sAkk[r_off + gid, c_off + 4 + tid] = _ak_pack_bf16x2(INV_f4, INV_f5)
    sAkk[r_off + gid + 8, c_off + 4 + tid] = _ak_pack_bf16x2(INV_f6, INV_f7)


@dsl_user_op
def _ak_invert_diag_neumann_inreg(
    sAkk,
    block_idx,
    lane_id,
    raw_f0,
    raw_f1,
    raw_f2,
    raw_f3,
    raw_f4,
    raw_f5,
    raw_f6,
    raw_f7,
    *,
    loc=None,
    ip=None,
):
    """In-register variant: invert 16x16 diag block from K2 acc fragment values
    (no SMEM read). Applies I+L mask, runs Neumann iterations, writes INV to sAkk.

    raw_f0..raw_f7 are K2 fp32 acc*beta values in C-fragment layout for m16n8k16:
      raw_f0,raw_f1 = (gid,   2*tid),   (gid,   2*tid+1)
      raw_f2,raw_f3 = (gid+8, 2*tid),   (gid+8, 2*tid+1)
      raw_f4,raw_f5 = (gid,   2*tid+8), (gid,   2*tid+9)
      raw_f6,raw_f7 = (gid+8, 2*tid+8), (gid+8, 2*tid+9)
    """
    r_off = block_idx * 16
    c_off = block_idx * 8
    gid = lane_id // 4
    tid = lane_id % 4

    _one = cutlass.Float32(1.0)
    _zero = cutlass.Float32(0.0)

    # Identity pattern (matches the SMEM-read variant's I_f computation)
    I_f0 = _one * cutlass.Float32(gid == 2 * tid) + _zero * cutlass.Float32(
        gid != 2 * tid
    )
    I_f1 = _one * cutlass.Float32(gid == 2 * tid + 1) + _zero * cutlass.Float32(
        gid != 2 * tid + 1
    )
    I_f2 = _one * cutlass.Float32(gid + 8 == 2 * tid) + _zero * cutlass.Float32(
        gid + 8 != 2 * tid
    )  # always 0
    I_f3 = _one * cutlass.Float32(gid + 8 == 2 * tid + 1) + _zero * cutlass.Float32(
        gid + 8 != 2 * tid + 1
    )  # always 0
    I_f4 = _one * cutlass.Float32(gid == 8 + 2 * tid) + _zero * cutlass.Float32(
        gid != 8 + 2 * tid
    )  # always 0
    I_f5 = _one * cutlass.Float32(gid == 8 + 2 * tid + 1) + _zero * cutlass.Float32(
        gid != 8 + 2 * tid + 1
    )  # always 0
    I_f6 = _one * cutlass.Float32(gid + 8 == 8 + 2 * tid) + _zero * cutlass.Float32(
        gid + 8 != 8 + 2 * tid
    )
    I_f7 = _one * cutlass.Float32(gid + 8 == 8 + 2 * tid + 1) + _zero * cutlass.Float32(
        gid + 8 != 8 + 2 * tid + 1
    )

    # Apply I+L mask to raw K2 values:
    #   diag (row==col): replace with 1
    #   strict upper (row<col within block): replace with 0
    #   strict lower (row>col within block): keep raw value (= L value)
    m_gt_a = cutlass.Float32(gid > 2 * tid)  # for cols 2*tid (with row=gid)
    m_gt_b = cutlass.Float32(gid > 2 * tid + 1)  # for cols 2*tid+1
    # (gid, 2*tid) and (gid, 2*tid+1): mask conditional
    A_f0 = m_gt_a * raw_f0 + I_f0
    A_f1 = m_gt_b * raw_f1 + I_f1
    # (gid+8, 2*tid) and (gid+8, 2*tid+1): always strict lower (gid+8 > 2*tid+1 always)
    A_f2 = raw_f2
    A_f3 = raw_f3
    # (gid, 2*tid+8) and (gid, 2*tid+9): always strict upper (gid <= 7 < 8 <= 2*tid+8)
    A_f4 = _zero
    A_f5 = _zero
    # (gid+8, 2*tid+8): row-col offset = gid+8 - (2*tid+8) = gid - 2*tid -> same as A_f0 mask
    # (gid+8, 2*tid+9): similar -> same as A_f1 mask
    A_f6 = m_gt_a * raw_f6 + I_f6
    A_f7 = m_gt_b * raw_f7 + I_f7

    # L = A - I, INV = I - L (first two Neumann terms)
    L_f0 = A_f0 - I_f0
    L_f1 = A_f1 - I_f1
    L_f2 = A_f2 - I_f2
    L_f3 = A_f3 - I_f3
    L_f4 = A_f4 - I_f4
    L_f5 = A_f5 - I_f5
    L_f6 = A_f6 - I_f6
    L_f7 = A_f7 - I_f7
    INV_f0 = I_f0 - L_f0
    INV_f1 = I_f1 - L_f1
    INV_f2 = I_f2 - L_f2
    INV_f3 = I_f3 - L_f3
    INV_f4 = I_f4 - L_f4
    INV_f5 = I_f5 - L_f5
    INV_f6 = I_f6 - L_f6
    INV_f7 = I_f7 - L_f7

    _zf = cutlass.Float32(0.0)
    L_a0 = _ak_pack_bf16x2(L_f0, L_f1)
    L_a1 = _ak_pack_bf16x2(L_f2, L_f3)
    L_a2 = _ak_pack_bf16x2(L_f4, L_f5)
    L_a3 = _ak_pack_bf16x2(L_f6, L_f7)
    INV_a0 = _ak_pack_bf16x2(INV_f0, INV_f1)
    INV_a1 = _ak_pack_bf16x2(INV_f2, INV_f3)
    INV_a2 = _ak_pack_bf16x2(INV_f4, INV_f5)
    INV_a3 = _ak_pack_bf16x2(INV_f6, INV_f7)

    # Iter 1: L^2, INV += INV*L^2
    L_b0 = _ak_movmatrix_trans(L_a0)
    L_b1 = _ak_movmatrix_trans(L_a1)
    L_b2 = _ak_movmatrix_trans(L_a2)
    L_b3 = _ak_movmatrix_trans(L_a3)
    Lp_c0, Lp_c1, Lp_c2, Lp_c3 = _ak_mma(
        L_a0, L_a1, L_a2, L_a3, L_b0, L_b1, _zf, _zf, _zf, _zf
    )
    Lp_c4, Lp_c5, Lp_c6, Lp_c7 = _ak_mma(
        L_a0, L_a1, L_a2, L_a3, L_b2, L_b3, _zf, _zf, _zf, _zf
    )
    Lp_a0 = _ak_pack_bf16x2(Lp_c0, Lp_c1)
    Lp_a1 = _ak_pack_bf16x2(Lp_c2, Lp_c3)
    Lp_a2 = _ak_pack_bf16x2(Lp_c4, Lp_c5)
    Lp_a3 = _ak_pack_bf16x2(Lp_c6, Lp_c7)
    Lp_b0 = _ak_movmatrix_trans(Lp_a0)
    Lp_b1 = _ak_movmatrix_trans(Lp_a1)
    Lp_b2 = _ak_movmatrix_trans(Lp_a2)
    Lp_b3 = _ak_movmatrix_trans(Lp_a3)
    mm_c0, mm_c1, mm_c2, mm_c3 = _ak_mma(
        INV_a0, INV_a1, INV_a2, INV_a3, Lp_b0, Lp_b1, _zf, _zf, _zf, _zf
    )
    mm_c4, mm_c5, mm_c6, mm_c7 = _ak_mma(
        INV_a0, INV_a1, INV_a2, INV_a3, Lp_b2, Lp_b3, _zf, _zf, _zf, _zf
    )
    INV_f0 = INV_f0 + mm_c0
    INV_f1 = INV_f1 + mm_c1
    INV_f2 = INV_f2 + mm_c2
    INV_f3 = INV_f3 + mm_c3
    INV_f4 = INV_f4 + mm_c4
    INV_f5 = INV_f5 + mm_c5
    INV_f6 = INV_f6 + mm_c6
    INV_f7 = INV_f7 + mm_c7
    INV_a0 = _ak_pack_bf16x2(INV_f0, INV_f1)
    INV_a1 = _ak_pack_bf16x2(INV_f2, INV_f3)
    INV_a2 = _ak_pack_bf16x2(INV_f4, INV_f5)
    INV_a3 = _ak_pack_bf16x2(INV_f6, INV_f7)

    # Iter 2: L^4, INV += INV*L^4
    L4_c0, L4_c1, L4_c2, L4_c3 = _ak_mma(
        Lp_a0, Lp_a1, Lp_a2, Lp_a3, Lp_b0, Lp_b1, _zf, _zf, _zf, _zf
    )
    L4_c4, L4_c5, L4_c6, L4_c7 = _ak_mma(
        Lp_a0, Lp_a1, Lp_a2, Lp_a3, Lp_b2, Lp_b3, _zf, _zf, _zf, _zf
    )
    L4_a0 = _ak_pack_bf16x2(L4_c0, L4_c1)
    L4_a1 = _ak_pack_bf16x2(L4_c2, L4_c3)
    L4_a2 = _ak_pack_bf16x2(L4_c4, L4_c5)
    L4_a3 = _ak_pack_bf16x2(L4_c6, L4_c7)
    L4_b0 = _ak_movmatrix_trans(L4_a0)
    L4_b1 = _ak_movmatrix_trans(L4_a1)
    L4_b2 = _ak_movmatrix_trans(L4_a2)
    L4_b3 = _ak_movmatrix_trans(L4_a3)
    mm_c0, mm_c1, mm_c2, mm_c3 = _ak_mma(
        INV_a0, INV_a1, INV_a2, INV_a3, L4_b0, L4_b1, _zf, _zf, _zf, _zf
    )
    mm_c4, mm_c5, mm_c6, mm_c7 = _ak_mma(
        INV_a0, INV_a1, INV_a2, INV_a3, L4_b2, L4_b3, _zf, _zf, _zf, _zf
    )
    INV_f0 = INV_f0 + mm_c0
    INV_f1 = INV_f1 + mm_c1
    INV_f2 = INV_f2 + mm_c2
    INV_f3 = INV_f3 + mm_c3
    INV_f4 = INV_f4 + mm_c4
    INV_f5 = INV_f5 + mm_c5
    INV_f6 = INV_f6 + mm_c6
    INV_f7 = INV_f7 + mm_c7
    INV_a0 = _ak_pack_bf16x2(INV_f0, INV_f1)
    INV_a1 = _ak_pack_bf16x2(INV_f2, INV_f3)
    INV_a2 = _ak_pack_bf16x2(INV_f4, INV_f5)
    INV_a3 = _ak_pack_bf16x2(INV_f6, INV_f7)

    # Iter 3: L^8, INV += INV*L^8
    L8_c0, L8_c1, L8_c2, L8_c3 = _ak_mma(
        L4_a0, L4_a1, L4_a2, L4_a3, L4_b0, L4_b1, _zf, _zf, _zf, _zf
    )
    L8_c4, L8_c5, L8_c6, L8_c7 = _ak_mma(
        L4_a0, L4_a1, L4_a2, L4_a3, L4_b2, L4_b3, _zf, _zf, _zf, _zf
    )
    L8_a0 = _ak_pack_bf16x2(L8_c0, L8_c1)
    L8_a1 = _ak_pack_bf16x2(L8_c2, L8_c3)
    L8_a2 = _ak_pack_bf16x2(L8_c4, L8_c5)
    L8_a3 = _ak_pack_bf16x2(L8_c6, L8_c7)
    L8_b0 = _ak_movmatrix_trans(L8_a0)
    L8_b1 = _ak_movmatrix_trans(L8_a1)
    L8_b2 = _ak_movmatrix_trans(L8_a2)
    L8_b3 = _ak_movmatrix_trans(L8_a3)
    mm_c0, mm_c1, mm_c2, mm_c3 = _ak_mma(
        INV_a0, INV_a1, INV_a2, INV_a3, L8_b0, L8_b1, _zf, _zf, _zf, _zf
    )
    mm_c4, mm_c5, mm_c6, mm_c7 = _ak_mma(
        INV_a0, INV_a1, INV_a2, INV_a3, L8_b2, L8_b3, _zf, _zf, _zf, _zf
    )
    INV_f0 = INV_f0 + mm_c0
    INV_f1 = INV_f1 + mm_c1
    INV_f2 = INV_f2 + mm_c2
    INV_f3 = INV_f3 + mm_c3
    INV_f4 = INV_f4 + mm_c4
    INV_f5 = INV_f5 + mm_c5
    INV_f6 = INV_f6 + mm_c6
    INV_f7 = INV_f7 + mm_c7

    sAkk[r_off + gid, c_off + tid] = _ak_pack_bf16x2(INV_f0, INV_f1)
    sAkk[r_off + gid + 8, c_off + tid] = _ak_pack_bf16x2(INV_f2, INV_f3)
    sAkk[r_off + gid, c_off + 4 + tid] = _ak_pack_bf16x2(INV_f4, INV_f5)
    sAkk[r_off + gid + 8, c_off + 4 + tid] = _ak_pack_bf16x2(INV_f6, INV_f7)


@dsl_user_op
def _ak_matmul_AB(sAkk, br_A, bc_A, br_B, bc_B, lane_id, *, loc=None, ip=None):
    gid = lane_id // 4
    tid = lane_id % 4
    _zf = cutlass.Float32(0.0)
    rA = br_A * 16
    cA = bc_A * 8
    rB = br_B * 16
    cB = bc_B * 8
    a0 = cutlass.Float32(sAkk[rA + gid, cA + tid])
    a1 = cutlass.Float32(sAkk[rA + gid + 8, cA + tid])
    a2 = cutlass.Float32(sAkk[rA + gid, cA + 4 + tid])
    a3 = cutlass.Float32(sAkk[rA + gid + 8, cA + 4 + tid])
    bA0 = cutlass.Float32(sAkk[rB + gid, cB + tid])
    bA1 = cutlass.Float32(sAkk[rB + gid + 8, cB + tid])
    bA2 = cutlass.Float32(sAkk[rB + gid, cB + 4 + tid])
    bA3 = cutlass.Float32(sAkk[rB + gid + 8, cB + 4 + tid])
    b0 = _ak_movmatrix_trans(bA0)
    b1 = _ak_movmatrix_trans(bA1)
    b2 = _ak_movmatrix_trans(bA2)
    b3 = _ak_movmatrix_trans(bA3)
    cn0_0, cn0_1, cn0_2, cn0_3 = _ak_mma(a0, a1, a2, a3, b0, b1, _zf, _zf, _zf, _zf)
    cn1_0, cn1_1, cn1_2, cn1_3 = _ak_mma(a0, a1, a2, a3, b2, b3, _zf, _zf, _zf, _zf)
    return cn0_0, cn0_1, cn0_2, cn0_3, cn1_0, cn1_1, cn1_2, cn1_3


@dsl_user_op
def _ak_chain_mma_B(sAkk, br_B, bc_B, a0, a1, a2, a3, lane_id, *, loc=None, ip=None):
    gid = lane_id // 4
    tid = lane_id % 4
    _zf = cutlass.Float32(0.0)
    rB = br_B * 16
    cB = bc_B * 8
    bA0 = cutlass.Float32(sAkk[rB + gid, cB + tid])
    bA1 = cutlass.Float32(sAkk[rB + gid + 8, cB + tid])
    bA2 = cutlass.Float32(sAkk[rB + gid, cB + 4 + tid])
    bA3 = cutlass.Float32(sAkk[rB + gid + 8, cB + 4 + tid])
    b0 = _ak_movmatrix_trans(bA0)
    b1 = _ak_movmatrix_trans(bA1)
    b2 = _ak_movmatrix_trans(bA2)
    b3 = _ak_movmatrix_trans(bA3)
    cn0_0, cn0_1, cn0_2, cn0_3 = _ak_mma(a0, a1, a2, a3, b0, b1, _zf, _zf, _zf, _zf)
    cn1_0, cn1_1, cn1_2, cn1_3 = _ak_mma(a0, a1, a2, a3, b2, b3, _zf, _zf, _zf, _zf)
    return cn0_0, cn0_1, cn0_2, cn0_3, cn1_0, cn1_1, cn1_2, cn1_3


@dsl_user_op
def _ak_chain_mma_A(sAkk, br_A, bc_A, b0, b1, b2, b3, lane_id, *, loc=None, ip=None):
    gid = lane_id // 4
    tid = lane_id % 4
    _zf = cutlass.Float32(0.0)
    rA = br_A * 16
    cA = bc_A * 8
    a0 = cutlass.Float32(sAkk[rA + gid, cA + tid])
    a1 = cutlass.Float32(sAkk[rA + gid + 8, cA + tid])
    a2 = cutlass.Float32(sAkk[rA + gid, cA + 4 + tid])
    a3 = cutlass.Float32(sAkk[rA + gid + 8, cA + 4 + tid])
    cn0_0, cn0_1, cn0_2, cn0_3 = _ak_mma(a0, a1, a2, a3, b0, b1, _zf, _zf, _zf, _zf)
    cn1_0, cn1_1, cn1_2, cn1_3 = _ak_mma(a0, a1, a2, a3, b2, b3, _zf, _zf, _zf, _zf)
    return cn0_0, cn0_1, cn0_2, cn0_3, cn1_0, cn1_1, cn1_2, cn1_3


@dsl_user_op
def _ak_store_neg_C(
    sAkk, br, bc, c0, c1, c2, c3, c4, c5, c6, c7, lane_id, *, loc=None, ip=None
):
    gid = lane_id // 4
    tid = lane_id % 4
    r = br * 16
    c = bc * 8
    sAkk[r + gid, c + tid] = _ak_pack_bf16x2(-c0, -c1)
    sAkk[r + gid + 8, c + tid] = _ak_pack_bf16x2(-c2, -c3)
    sAkk[r + gid, c + 4 + tid] = _ak_pack_bf16x2(-c4, -c5)
    sAkk[r + gid + 8, c + 4 + tid] = _ak_pack_bf16x2(-c6, -c7)


@dsl_user_op
def _ak_pack_C_to_A(c0, c1, c2, c3, c4, c5, c6, c7, *, loc=None, ip=None):
    a0 = _ak_pack_bf16x2(c0, c1)
    a1 = _ak_pack_bf16x2(c2, c3)
    a2 = _ak_pack_bf16x2(c4, c5)
    a3 = _ak_pack_bf16x2(c6, c7)
    return a0, a1, a2, a3


@dsl_user_op
def _ak_pack_C_to_B(c0, c1, c2, c3, c4, c5, c6, c7, *, loc=None, ip=None):
    a0 = _ak_pack_bf16x2(c0, c1)
    a1 = _ak_pack_bf16x2(c2, c3)
    a2 = _ak_pack_bf16x2(c4, c5)
    a3 = _ak_pack_bf16x2(c6, c7)
    b0 = _ak_movmatrix_trans(a0)
    b1 = _ak_movmatrix_trans(a1)
    b2 = _ak_movmatrix_trans(a2)
    b3 = _ak_movmatrix_trans(a3)
    return b0, b1, b2, b3


@dsl_user_op
def _ak_store_C_temp(
    sT, buf, c0, c1, c2, c3, c4, c5, c6, c7, lane_id, *, loc=None, ip=None
):
    gid = lane_id // 4
    tid = lane_id % 4
    sT[gid, 2 * tid, buf] = c0
    sT[gid, 2 * tid + 1, buf] = c1
    sT[gid + 8, 2 * tid, buf] = c2
    sT[gid + 8, 2 * tid + 1, buf] = c3
    sT[gid, 8 + 2 * tid, buf] = c4
    sT[gid, 8 + 2 * tid + 1, buf] = c5
    sT[gid + 8, 8 + 2 * tid, buf] = c6
    sT[gid + 8, 8 + 2 * tid + 1, buf] = c7


@dsl_user_op
def _ak_load_C_temp(sT, buf, lane_id, *, loc=None, ip=None):
    gid = lane_id // 4
    tid = lane_id % 4
    c0 = cutlass.Float32(sT[gid, 2 * tid, buf])
    c1 = cutlass.Float32(sT[gid, 2 * tid + 1, buf])
    c2 = cutlass.Float32(sT[gid + 8, 2 * tid, buf])
    c3 = cutlass.Float32(sT[gid + 8, 2 * tid + 1, buf])
    c4 = cutlass.Float32(sT[gid, 8 + 2 * tid, buf])
    c5 = cutlass.Float32(sT[gid, 8 + 2 * tid + 1, buf])
    c6 = cutlass.Float32(sT[gid + 8, 8 + 2 * tid, buf])
    c7 = cutlass.Float32(sT[gid + 8, 8 + 2 * tid + 1, buf])
    return c0, c1, c2, c3, c4, c5, c6, c7


@dsl_user_op
def opaque_zero_from_work_id(*, loc=None, ip=None):
    """
    Return 0 via opaque side-effectful asm (no inputs needed).

    Because has_side_effects=True, MLIR LICM treats this as having memory
    effects and will NOT hoist it outside the for_generate loop.  Any value
    computed from the result (_oz) therefore appears loop-variant to LICM,
    preventing get_slice() and scalar layout-invariant computations from being
    hoisted to the kernel prologue.  This keeps prologue register pressure < 64
    and eliminates the 440-byte stack frame that caused ~300-cycle L2 LDL
    penalties per iteration.
    """
    result = llvm.inline_asm(
        T.i32(),
        [],
        "mov.b32 $0, 0;",  # output = 0 (opaque to compiler constant-folding)
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result)


@cute.kernel
def fused_kernel123(
    tma_atom_Q: cute.CopyAtom,
    tma_tensor_Q: cute.Tensor,
    tma_atom_K: cute.CopyAtom,
    tma_tensor_K: cute.Tensor,
    tma_atom_G: cute.CopyAtom,
    tma_tensor_G: cute.Tensor,
    mA_log: cute.Tensor,
    mBeta: cute.Tensor,
    scale: cutlass.Float32,
    mKscaled: cute.Tensor,
    mKg: cute.Tensor,
    mQscaled: cute.Tensor,
    mGkLast: cute.Tensor,
    mAqk: cute.Tensor,  # 4D (B, T, H, BT) — used by per-tile pure path
    mAkk: cute.Tensor,  # 4D (B, T, H, BT) — used by per-tile pure path
    mAqk_v2: cute.Tensor,  # 5D (B, T, H, BT/2, 2) — used by vec autovec non-pure path
    mAkk_v2: cute.Tensor,  # 5D (B, T, H, BT/2, 2) — used by vec autovec non-pure path
    tiled_copy_qk_k1,
    tiled_mma_k2,
    tiled_copy_mma_A,
    tiled_copy_mma_B,
    tiled_copy_Gcum_norm,
    tiled_copy_Gcum_gate,
    qk_smem_layout,
    g_smem_layout,
    g_cumsum_layout,
    num_chunks: int,
    num_heads: int,
    batch_size: int,
    mCuSeqlens: cute.Tensor,
    mChunkIndices: cute.Tensor,
    IS_VARLEN: cutlass.Constexpr[int],
    mDtBias: cute.Tensor,
    lower_bound: cutlass.Float32,
    HAS_BIAS: cutlass.Constexpr[int],
    USE_SAFE_GATE: cutlass.Constexpr[int],
    VARLEN_PURE: cutlass.Constexpr[int] = 0,
):
    block_id, _, _ = cute.arch.block_idx()
    tidx = cute.arch.thread_idx()[0]
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_id = tidx % 32

    total_cgs_per_head = cutlass.Int32(0)
    cgs_per_head = cutlass.Int32(0)
    total_cgs = cutlass.Int32(0)
    if IS_VARLEN:
        total_cgs_per_head = (num_chunks + CHUNKS_PER_BLOCK - 1) // CHUNKS_PER_BLOCK
        total_cgs = total_cgs_per_head * num_heads
    else:
        cgs_per_head = num_chunks // CHUNKS_PER_BLOCK
        total_cgs = cgs_per_head * num_heads * batch_size

    # =====================================================================
    # SMEM allocation
    # =====================================================================
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(
        cutlass.BFloat16, qk_smem_layout.outer, 128, swizzle=qk_smem_layout.inner
    )
    sK = smem.allocate_tensor(
        cutlass.BFloat16, qk_smem_layout.outer, 128, swizzle=qk_smem_layout.inner
    )
    sG = smem.allocate_tensor(cutlass.BFloat16, g_smem_layout, 128)
    sGcum = smem.allocate_tensor(cutlass.Float32, g_cumsum_layout, 128)
    partial_last_layout = cute.make_layout(
        (K1_ROW_GROUPS, PARTIAL_COLS), stride=(PARTIAL_COLS, 1)
    )
    sPartialLast = smem.allocate_tensor(cutlass.Float32, partial_last_layout, 128)

    # sAqk: 64x72 row-major (BT rows, BT+pad cols), same shape as sAkk.
    # Each sub-tile (i_q, i_k) sits at SMEM rows [i_q*BC, (i_q+1)*BC) cols
    # [i_k*BC, (i_k+1)*BC) — directly mirrors the 64x64 attention matrix.
    aqk_tile_layout = cute.make_layout(
        (BT, AQK_TILE_STRIDE, NUM_STAGES),
        stride=(AQK_TILE_STRIDE, 1, BT * AQK_TILE_STRIDE),
    )
    sAqk = smem.allocate_tensor(cutlass.BFloat16, aqk_tile_layout, 128)

    akk_tile_layout = cute.make_layout(
        (BT, AKK_STRIDE, NUM_STAGES), stride=(AKK_STRIDE, 1, BT * AKK_STRIDE)
    )
    sAkk = smem.allocate_tensor(cutlass.BFloat16, akk_tile_layout, 128)
    # sAkk_pkd / sTemp removed: akk_inv runs as a separate kernel call (chained back-to-back).

    # sBeta: chunk's 64 beta values staged in SMEM. K1 reads from here in inner loop
    # to reduce register pressure / avoid repeated gmem broadcast.
    beta_smem_layout = cute.make_layout((BT, NUM_STAGES), stride=(1, BT))
    sBeta = smem.allocate_tensor(cutlass.BFloat16, beta_smem_layout, 128)

    # =====================================================================
    # Mbarrier allocation & init
    # =====================================================================
    tma_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    stage_reuse_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    k1_done_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    mma_done_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)
    store_done_mbars = smem.allocate_array(cutlass.Int64, NUM_STAGES)

    bytes_per_stage = BT * K_DIM * 2 * 3

    if tidx == 0:
        for s in range(NUM_STAGES):
            cute.arch.mbarrier_init(tma_mbars + s, 1)
            # TMA warp is waiter (not arriver) on stage_reuse; and it skips mma_done arrive
            cute.arch.mbarrier_init(stage_reuse_mbars + s, (NUM_MMA_WARPS - 1) * 32)
            cute.arch.mbarrier_init(k1_done_mbars + s, NUM_K1_TMA_WARPS * 32)
            cute.arch.mbarrier_init(mma_done_mbars + s, (NUM_MMA_WARPS - 1) * 32)
            cute.arch.mbarrier_init(store_done_mbars + s, NUM_STORE_WARPS * 32)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    # =====================================================================
    # SMEM init: zero out sAqk and sAkk valid 64x64 region (cols 64..71 are
    # padding for SMEM bank-conflict avoidance, never read or written by MMA
    # or store warps). Required for downstream row-major store optimizations
    # — positions outside MMA-written sub-tiles stay at 0.
    #
    # Cooperative pattern (32 warps × 32 lanes = 1024 threads):
    #   - Each warp owns 2 contiguous rows (warp_id*2, warp_id*2+1)
    #   - Each lane owns 2 contiguous bf16 cols (lane*2, lane*2+1)
    #   - Per lane: 2 stages × 2 rows × 2 buffers × 2 cols = 16 bf16 stores
    #   - Adjacent (lane*2, lane*2+1) bf16 pairs are 4-byte aligned →
    #     ptxas should fuse into STS.32 (8 wide stores per lane).
    # =====================================================================
    _warp_id_in_cta = tidx >> 5  # tidx // 32, range 0..31
    _lane_id_warp = tidx & 31  # tidx % 32, range 0..31
    _row_base = _warp_id_in_cta * 2  # this warp owns rows [_row_base, _row_base+1]
    _col_lo = _lane_id_warp * 2  # this lane owns cols [_col_lo, _col_lo+1]
    _col_hi = _col_lo + 1
    for _s in cutlass.range_constexpr(NUM_STAGES):
        for _ri in cutlass.range_constexpr(2):
            _row = _row_base + _ri
            sAqk[_row, _col_lo, _s] = cutlass.BFloat16(0.0)
            sAqk[_row, _col_hi, _s] = cutlass.BFloat16(0.0)
            sAkk[_row, _col_lo, _s] = cutlass.BFloat16(0.0)
            sAkk[_row, _col_hi, _s] = cutlass.BFloat16(0.0)
    cute.arch.barrier()

    # =====================================================================
    # Pre-arrive (MMA warps only)
    # stage_reuse_mbars: warp 0 waits before MMA arrives → pre-arrive all 12 MMA warps
    # store_done_mbars:  MMA waits before Store arrives → pre-arrive first 4 MMA warps
    # =====================================================================
    if (
        warp_idx >= NUM_K1_TMA_WARPS
        and warp_idx < NUM_K1_TMA_WARPS + NUM_MMA_WARPS
        and warp_idx != TMA_WARP_ID
    ):
        mma_warp_tmp = warp_idx - NUM_K1_TMA_WARPS
        for s in range(NUM_STAGES):
            cute.arch.mbarrier_arrive(stage_reuse_mbars + s)
            if mma_warp_tmp < NUM_STORE_WARPS:
                cute.arch.mbarrier_arrive(store_done_mbars + s)

    # =================================================================
    # Persistent outer loop. Single for_generate at top level (required).
    # Opaque asm barrier on work_id prevents MLIR LICM from hoisting
    # get_slice() and scalar layout invariants to the kernel prologue,
    # keeping register pressure < 64 and eliminating prologue spill.
    # =================================================================
    for work_id in for_generate(block_id, total_cgs, NUM_SMS):
        i_cg = cutlass.Int32(0)
        i_h = cutlass.Int32(0)
        i_b = cutlass.Int32(0)
        chunk_base = cutlass.Int32(0)
        if IS_VARLEN:
            i_cg = work_id % total_cgs_per_head
            i_h = work_id // total_cgs_per_head
            i_b = cutlass.Int32(0)
            chunk_base = i_cg * CHUNKS_PER_BLOCK
        else:
            i_cg = work_id % cgs_per_head
            i_h = (work_id // cgs_per_head) % num_heads
            i_b = work_id // (cgs_per_head * num_heads)
            chunk_base = i_cg * CHUNKS_PER_BLOCK

        # Anti-LICM barrier: _oz is always 0 but appears to depend on work_id.
        # Because this asm has side_effects=True, it stays inside the loop.
        # Any value computed from _oz/_lane/_warp is also loop-variant from
        # LICM's perspective → get_slice() and scalar invariants stay in-loop.
        _oz = opaque_zero_from_work_id()
        _lane = lane_id + _oz
        _warp = warp_idx + _oz

        # =============================================================
        # Warps 0-15: Fused TMA + K1
        # =============================================================
        if warp_idx < NUM_K1_TMA_WARPS:
            # Warp-layout invariants (scope-local → no cross-group register spill)
            k1_warp = _warp
            warp_row_group = k1_warp % K1_ROW_GROUPS
            warp_col_group = k1_warp // K1_ROW_GROUPS
            k1_row_start = warp_row_group * ROWS_PER_K1_WARP
            col_base = warp_col_group * K1_COLS_PER_WARP + _lane * VEC
            col_vec_idx = warp_col_group * (K1_COLS_PER_WARP // VEC) + _lane
            cumsum_scale = cutlass.Float32(RCP_LN2)
            thr_copy_k1 = tiled_copy_qk_k1.get_slice(_lane)

            rAcc = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)
            rPrefix = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)
            rGkLast = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)
            rKsOut = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.BFloat16)
            rQsOut = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.BFloat16)
            rKgOut = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.BFloat16)
            rGkOut = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)

            # exp_A depends on i_h (changes per work unit)
            exp_A = cute.exp(mA_log[i_h], fastmath=True)

            # Load dt_bias per (head, col) — broadcast across all rows
            rBias = cute.make_rmem_tensor(cute.make_layout((VEC,)), cutlass.Float32)
            if HAS_BIAS:
                for vi in cutlass.range_constexpr(VEC):
                    rBias[vi] = mDtBias[i_h, col_base + vi]
            else:
                for vi in cutlass.range_constexpr(VEC):
                    rBias[vi] = cutlass.Float32(0.0)

            # 3D TMA head slices (fixed for this work unit's head)
            gQ_head = tma_tensor_Q[(None, None, i_h)]
            gK_head = tma_tensor_K[(None, None, i_h)]
            gG_head = tma_tensor_G[(None, None, i_h)]

            for chunk_iter in cutlass.range_constexpr(CHUNKS_PER_BLOCK):
                cur_stage = chunk_iter % NUM_STAGES
                cur_phase = chunk_iter // NUM_STAGES % 2
                chunk_idx = chunk_base + chunk_iter
                chunk_start = cutlass.Int32(0)
                ci_eos = cutlass.Int32(0)
                if IS_VARLEN:
                    if chunk_idx < num_chunks:
                        _sid = cutlass.Int32(mChunkIndices[chunk_idx, 0])
                        chunk_start = (
                            cutlass.Int32(mCuSeqlens[_sid])
                            + cutlass.Int32(mChunkIndices[chunk_idx, 1]) * BT
                        )
                        ci_eos = cutlass.Int32(mCuSeqlens[_sid + 1])
                else:
                    chunk_start = chunk_idx * BT

                cute.arch.mbarrier_wait(tma_mbars + cur_stage, cur_phase)

                csG = sG[(None, None, cur_stage)]
                csGcum = sGcum[(None, None, cur_stage)]
                csQ = sQ[(None, None, cur_stage)]
                csK = sK[(None, None, cur_stage)]
                csBeta = sBeta[(None, cur_stage)]

                # Stage chunk's 64 beta values to SMEM (warp 0 of K1 group, 32 threads × 2).
                # Synced via the existing k1_internal_barrier() before Pass 2b reads it.
                if k1_warp == 0:
                    for _bi in cutlass.range_constexpr(BT // 32):
                        _idx = _bi * 32 + _lane
                        csBeta[_idx] = mBeta[i_b, chunk_start + _idx, i_h]

                rGact = cute.make_rmem_tensor(
                    cute.make_layout((ROWS_PER_K1_WARP, VEC)), cutlass.Float32
                )
                for vi in cutlass.range_constexpr(VEC):
                    rAcc[vi] = cutlass.Float32(0.0)

                for ri in cutlass.range_constexpr(ROWS_PER_K1_WARP):
                    row = k1_row_start + ri
                    for vi in cutlass.range_constexpr(VEC):
                        c = col_base + vi
                        g_val = csG[row, c].to(cutlass.Float32)
                        if HAS_BIAS:
                            g_val = g_val + rBias[vi]
                        g_activated = cutlass.Float32(0.0)
                        if USE_SAFE_GATE:
                            sigmoid_g = fast_rcp(
                                cutlass.Float32(1.0)
                                + cute.exp2(-exp_A * g_val * LOG2E, fastmath=True)
                            )
                            g_activated = lower_bound * sigmoid_g
                        else:
                            softplus_g = (
                                cute.log2(
                                    cutlass.Float32(1.0)
                                    + cute.exp2(g_val * LOG2E, fastmath=True),
                                    fastmath=True,
                                )
                                * LN2
                            )
                            g_activated = -exp_A * softplus_g
                        # Varlen: zero gate for out-of-bounds rows so cumsum
                        # stays flat beyond the last valid position.
                        # VARLEN_PURE=1 elides this at compile time — caller
                        # guarantees all seq lengths are multiples of BT so no
                        # chunk has OOB rows.
                        if IS_VARLEN and not VARLEN_PURE:
                            if chunk_start + row >= ci_eos:
                                g_activated = cutlass.Float32(0.0)
                        rGact[ri, vi] = g_activated
                        rAcc[vi] = rAcc[vi] + g_activated

                for vi in cutlass.range_constexpr(VEC):
                    sPartialLast[warp_row_group, col_base + vi] = rAcc[vi]

                k1_internal_barrier()

                prefix_col_start = k1_warp * PARTIAL_COLS_PER_WARP
                row_in_prefix = lane_id % K1_ROW_GROUPS
                col_in_group = lane_id // K1_ROW_GROUPS

                for j in cutlass.range_constexpr(PARTIAL_COLS_PER_WARP // 4):
                    col = prefix_col_start + j * 4 + col_in_group
                    val = cutlass.Float32(sPartialLast[row_in_prefix, col])
                    tmp = cute.arch.shuffle_sync_up(
                        val, 1, mask=-1, mask_and_clamp=SHFL_W8_CLAMP
                    )
                    if row_in_prefix >= 1:
                        val = val + tmp
                    tmp = cute.arch.shuffle_sync_up(
                        val, 2, mask=-1, mask_and_clamp=SHFL_W8_CLAMP
                    )
                    if row_in_prefix >= 2:
                        val = val + tmp
                    tmp = cute.arch.shuffle_sync_up(
                        val, 4, mask=-1, mask_and_clamp=SHFL_W8_CLAMP
                    )
                    if row_in_prefix >= 4:
                        val = val + tmp
                    sPartialLast[row_in_prefix, col] = val

                k1_internal_barrier()

                for vi in cutlass.range_constexpr(VEC):
                    rGkLast[vi] = sPartialLast[K1_ROW_GROUPS - 1, col_base + vi]

                for vi in cutlass.range_constexpr(VEC):
                    rPrefix[vi] = cutlass.Float32(0.0)
                if warp_row_group > 0:
                    for vi in cutlass.range_constexpr(VEC):
                        rPrefix[vi] = sPartialLast[warp_row_group - 1, col_base + vi]

                # ---- Pass 2a: ONLY cumsum + write csGcum (critical path, minimal work) ----
                for vi in cutlass.range_constexpr(VEC):
                    rAcc[vi] = rPrefix[vi]

                for ri in cutlass.range_constexpr(ROWS_PER_K1_WARP):
                    row = k1_row_start + ri
                    for vi in cutlass.range_constexpr(VEC):
                        rAcc[vi] = rAcc[vi] + rGact[ri, vi]
                        csGcum[row, col_base + vi] = rAcc[vi] * cumsum_scale

                # Signal MMA early: csGcum is ready
                cute.arch.mbarrier_arrive(k1_done_mbars + cur_stage)

                # ---- Pass 2b: recompute + write GMEM (overlaps with MMA, off critical path) ----
                for vi in cutlass.range_constexpr(VEC):
                    rAcc[vi] = rPrefix[vi]

                for ri in cutlass.range_constexpr(ROWS_PER_K1_WARP):
                    row = k1_row_start + ri
                    t = chunk_start + row

                    sK_tile = cute.local_tile(
                        csK, tiler=(1, K1_COLS_PER_WARP), coord=(row, warp_col_group)
                    )
                    tCsK = thr_copy_k1.partition_S(sK_tile)
                    tCrK = cute.make_fragment_like(tCsK)
                    cute.copy(tiled_copy_qk_k1, tCsK, thr_copy_k1.retile(tCrK))

                    sQ_tile = cute.local_tile(
                        csQ, tiler=(1, K1_COLS_PER_WARP), coord=(row, warp_col_group)
                    )
                    tCsQ = thr_copy_k1.partition_S(sQ_tile)
                    tCrQ = cute.make_fragment_like(tCsQ)
                    cute.copy(tiled_copy_qk_k1, tCsQ, thr_copy_k1.retile(tCrQ))

                    # Read beta from SMEM (staged once per chunk by warp 0).
                    beta_val = cutlass.Float32(csBeta[row])

                    for vi in cutlass.range_constexpr(VEC):
                        rAcc[vi] = rAcc[vi] + rGact[ri, vi]
                        cs = rAcc[vi] * cumsum_scale

                        k_val = tCrK[vi].to(cutlass.Float32)
                        q_val = tCrQ[vi].to(cutlass.Float32)

                        exp2_cs = cute.exp2(cs, fastmath=True)
                        gk_last_cs = rGkLast[vi] * cumsum_scale
                        exp2_kg = cute.exp2(gk_last_cs - cs, fastmath=True)

                        rKsOut[vi] = (k_val * exp2_cs).to(cutlass.BFloat16)
                        rQsOut[vi] = (q_val * exp2_cs * scale).to(cutlass.BFloat16)
                        rKgOut[vi] = (k_val * exp2_kg).to(cutlass.BFloat16)

                    if IS_VARLEN and not VARLEN_PURE:
                        if t < ci_eos:
                            cute.autovec_copy(
                                rKsOut, mKscaled[i_b, t, i_h, col_vec_idx, None]
                            )
                            cute.autovec_copy(
                                rQsOut, mQscaled[i_b, t, i_h, col_vec_idx, None]
                            )
                            cute.autovec_copy(
                                rKgOut, mKg[i_b, t, i_h, col_vec_idx, None]
                            )
                    else:
                        cute.autovec_copy(
                            rKsOut, mKscaled[i_b, t, i_h, col_vec_idx, None]
                        )
                        cute.autovec_copy(
                            rQsOut, mQscaled[i_b, t, i_h, col_vec_idx, None]
                        )
                        cute.autovec_copy(rKgOut, mKg[i_b, t, i_h, col_vec_idx, None])

                if warp_row_group == 0:
                    for vi in cutlass.range_constexpr(VEC):
                        rGkOut[vi] = cute.exp2(
                            rGkLast[vi] * cumsum_scale, fastmath=True
                        )
                    if IS_VARLEN:
                        if ci_eos > cutlass.Int32(0):
                            cute.autovec_copy(
                                rGkOut, mGkLast[i_b, chunk_idx, i_h, col_vec_idx, None]
                            )
                    else:
                        cute.autovec_copy(
                            rGkOut, mGkLast[i_b, chunk_idx, i_h, col_vec_idx, None]
                        )

        # =============================================================
        # Warp 26 (TMA_WARP_ID): dedicated TMA producer.
        # Waits stage_reuse (gated by MMA arrives), issues TMA for Q/K/G,
        # signals tma_mbar. Decouples MMA -> TMA dependency from K1 compute.
        # =============================================================
        if warp_idx == TMA_WARP_ID:
            gQ_head = tma_tensor_Q[(None, None, i_h)]
            gK_head = tma_tensor_K[(None, None, i_h)]
            gG_head = tma_tensor_G[(None, None, i_h)]

            # Prefetch chunk 0 -> stage 0 (stage_reuse[0] pre-arrived)
            pf_cs = cutlass.Int32(0)
            if IS_VARLEN:
                pf_seq_id_0 = cutlass.Int32(mChunkIndices[chunk_base, 0])
                pf_local_0 = cutlass.Int32(mChunkIndices[chunk_base, 1])
                pf_bos_0 = cutlass.Int32(mCuSeqlens[pf_seq_id_0])
                pf_cs = pf_bos_0 + pf_local_0 * BT
            else:
                pf_cs = i_b * num_chunks * BT + chunk_base * BT
            cute.arch.mbarrier_wait(stage_reuse_mbars, 0)
            if lane_id == 0:
                cute.arch.mbarrier_expect_tx(tma_mbars, bytes_per_stage)
            sQ_pf = sQ[(None, None, 0)]
            gQ_pf = cute.local_tile(
                cute.domain_offset((pf_cs, 0), gQ_head), (BT, K_DIM), (0, 0)
            )
            ts_pf, tg_pf = cpasync.tma_partition(
                tma_atom_Q,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ_pf, 0, 2),
                cute.group_modes(gQ_pf, 0, 2),
            )
            cute.copy(tma_atom_Q, tg_pf, ts_pf, tma_bar_ptr=tma_mbars)
            sK_pf = sK[(None, None, 0)]
            gK_pf = cute.local_tile(
                cute.domain_offset((pf_cs, 0), gK_head), (BT, K_DIM), (0, 0)
            )
            ts_pf, tg_pf = cpasync.tma_partition(
                tma_atom_K,
                0,
                cute.make_layout(1),
                cute.group_modes(sK_pf, 0, 2),
                cute.group_modes(gK_pf, 0, 2),
            )
            cute.copy(tma_atom_K, tg_pf, ts_pf, tma_bar_ptr=tma_mbars)
            sG_pf = sG[(None, None, 0)]
            gG_pf = cute.local_tile(
                cute.domain_offset((pf_cs, 0), gG_head), (BT, K_DIM), (0, 0)
            )
            ts_pf, tg_pf = cpasync.tma_partition(
                tma_atom_G,
                0,
                cute.make_layout(1),
                cute.group_modes(sG_pf, 0, 2),
                cute.group_modes(gG_pf, 0, 2),
            )
            cute.copy(tma_atom_G, tg_pf, ts_pf, tma_bar_ptr=tma_mbars)
            if lane_id == 0:
                cute.arch.mbarrier_arrive(tma_mbars)

            # Issue TMAs for chunks 1..CHUNKS_PER_BLOCK-1
            for next_i in cutlass.range_constexpr(1, CHUNKS_PER_BLOCK):
                next_stage = next_i % NUM_STAGES
                next_phase = next_i // NUM_STAGES % 2
                next_cs = cutlass.Int32(0)
                if IS_VARLEN:
                    next_chunk_idx = chunk_base + next_i
                    if next_chunk_idx < num_chunks:
                        _nsid = cutlass.Int32(mChunkIndices[next_chunk_idx, 0])
                        next_cs = (
                            cutlass.Int32(mCuSeqlens[_nsid])
                            + cutlass.Int32(mChunkIndices[next_chunk_idx, 1]) * BT
                        )
                else:
                    next_cs = i_b * num_chunks * BT + (chunk_base + next_i) * BT
                cute.arch.mbarrier_wait(stage_reuse_mbars + next_stage, next_phase)
                if lane_id == 0:
                    cute.arch.mbarrier_expect_tx(
                        tma_mbars + next_stage, bytes_per_stage
                    )
                sQ_ns = sQ[(None, None, next_stage)]
                gQ_ns = cute.local_tile(
                    cute.domain_offset((next_cs, 0), gQ_head), (BT, K_DIM), (0, 0)
                )
                ts_ns, tg_ns = cpasync.tma_partition(
                    tma_atom_Q,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sQ_ns, 0, 2),
                    cute.group_modes(gQ_ns, 0, 2),
                )
                cute.copy(tma_atom_Q, tg_ns, ts_ns, tma_bar_ptr=tma_mbars + next_stage)
                sK_ns = sK[(None, None, next_stage)]
                gK_ns = cute.local_tile(
                    cute.domain_offset((next_cs, 0), gK_head), (BT, K_DIM), (0, 0)
                )
                ts_ns, tg_ns = cpasync.tma_partition(
                    tma_atom_K,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK_ns, 0, 2),
                    cute.group_modes(gK_ns, 0, 2),
                )
                cute.copy(tma_atom_K, tg_ns, ts_ns, tma_bar_ptr=tma_mbars + next_stage)
                sG_ns = sG[(None, None, next_stage)]
                gG_ns = cute.local_tile(
                    cute.domain_offset((next_cs, 0), gG_head), (BT, K_DIM), (0, 0)
                )
                ts_ns, tg_ns = cpasync.tma_partition(
                    tma_atom_G,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sG_ns, 0, 2),
                    cute.group_modes(gG_ns, 0, 2),
                )
                cute.copy(tma_atom_G, tg_ns, ts_ns, tma_bar_ptr=tma_mbars + next_stage)
                if lane_id == 0:
                    cute.arch.mbarrier_arrive(tma_mbars + next_stage)

        # =============================================================
        # Warps 16-27 (excluding TMA_WARP_ID=26): K2 MMA Compute
        # =============================================================
        if (
            warp_idx >= NUM_K1_TMA_WARPS
            and warp_idx < NUM_K1_TMA_WARPS + NUM_MMA_WARPS
            and warp_idx != TMA_WARP_ID
        ):
            # Warp-layout invariants (scope-local → no cross-group register spill)
            _tid_in_group = _lane % 4
            _group_id = _lane // 4
            mma_warp = _warp - NUM_K1_TMA_WARPS
            my_i_q = cutlass.Int32(0)
            my_i_k = cutlass.Int32(0)
            if mma_warp < 1:
                my_i_q = cutlass.Int32(0)
                my_i_k = mma_warp
            elif mma_warp < 3:
                my_i_q = cutlass.Int32(1)
                my_i_k = mma_warp - 1
            elif mma_warp < 6:
                my_i_q = cutlass.Int32(2)
                my_i_k = mma_warp - 3
            elif mma_warp < NUM_MMA_ACTIVE:
                my_i_q = cutlass.Int32(3)
                my_i_k = mma_warp - 6
            q_row_base = my_i_q * BC
            k_row_base = my_i_k * BC
            akk_row_base = k_row_base
            akk_col_base = q_row_base
            norm_row = q_row_base
            if my_i_q == my_i_k:
                norm_row = q_row_base + cutlass.Int32(BC // 2)
            row0 = _group_id
            row1 = _group_id + 8
            col0 = _tid_in_group * 2
            col1 = _tid_in_group * 2 + 1
            col2 = 8 + _tid_in_group * 2
            col3 = 8 + _tid_in_group * 2 + 1
            thr_mma = tiled_mma_k2.get_slice(_lane)
            thr_copy_A = tiled_copy_mma_A.get_slice(_lane)
            thr_copy_B = tiled_copy_mma_B.get_slice(_lane)
            thr_copy_Gn = tiled_copy_Gcum_norm.get_slice(_tid_in_group)
            thr_copy_Ggate = tiled_copy_Gcum_gate.get_slice(_lane)

            for chunk_iter in cutlass.range_constexpr(CHUNKS_PER_BLOCK):
                s = chunk_iter % NUM_STAGES
                phase = chunk_iter // NUM_STAGES % 2
                chunk_idx = chunk_base + chunk_iter
                chunk_start = cutlass.Int32(0)
                if IS_VARLEN:
                    if chunk_idx < num_chunks:
                        _sid = cutlass.Int32(mChunkIndices[chunk_idx, 0])
                        chunk_start = (
                            cutlass.Int32(mCuSeqlens[_sid])
                            + cutlass.Int32(mChunkIndices[chunk_idx, 1]) * BT
                        )
                else:
                    chunk_start = chunk_idx * BT

                cute.arch.mbarrier_wait(k1_done_mbars + s, phase)
                cute.arch.mbarrier_wait(store_done_mbars + s, phase)

                if mma_warp < NUM_MMA_ACTIVE:
                    csQ = sQ[(None, None, s)]
                    csK = sK[(None, None, s)]
                    csGcum = sGcum[(None, None, s)]
                    csAqk = sAqk[(None, None, s)]
                    csAkk = sAkk[(None, None, s)]

                    _z = cutlass.Float32(0.0)

                    beta_row0 = mBeta[i_b, chunk_start + q_row_base + row0, i_h].to(
                        cutlass.Float32
                    )
                    beta_row1 = mBeta[i_b, chunk_start + q_row_base + row1, i_h].to(
                        cutlass.Float32
                    )

                    acc_aqk_n0_0, acc_aqk_n0_1, acc_aqk_n0_2, acc_aqk_n0_3 = (
                        _z,
                        _z,
                        _z,
                        _z,
                    )
                    acc_aqk_n1_0, acc_aqk_n1_1, acc_aqk_n1_2, acc_aqk_n1_3 = (
                        _z,
                        _z,
                        _z,
                        _z,
                    )
                    acc_akk_n0_0, acc_akk_n0_1, acc_akk_n0_2, acc_akk_n0_3 = (
                        _z,
                        _z,
                        _z,
                        _z,
                    )
                    acc_akk_n1_0, acc_akk_n1_1, acc_akk_n1_2, acc_akk_n1_3 = (
                        _z,
                        _z,
                        _z,
                        _z,
                    )

                    # bf16 m16n8k16 MMA: each k_block covers k=16.
                    for k_block in cutlass.range_constexpr(NUM_MMA_K_TILES):
                        # ---- Load Q/Kq bf16 fragments (16x16, 8 bf16/thread) ----
                        sQ_tile = cute.local_tile(
                            csQ, tiler=(16, 16), coord=(my_i_q, k_block)
                        )
                        tCrQ = tiled_mma_k2.make_fragment_A(
                            thr_mma.partition_A(sQ_tile)
                        )
                        cute.copy(
                            tiled_copy_mma_A,
                            thr_copy_A.partition_S(sQ_tile),
                            thr_copy_A.retile(tCrQ),
                        )

                        sKq_tile = cute.local_tile(
                            csK, tiler=(16, 16), coord=(my_i_q, k_block)
                        )
                        tCrKq = tiled_mma_k2.make_fragment_A(
                            thr_mma.partition_A(sKq_tile)
                        )
                        cute.copy(
                            tiled_copy_mma_A,
                            thr_copy_A.partition_S(sKq_tile),
                            thr_copy_A.retile(tCrKq),
                        )

                        # ---- Issue K n0/n1 LDSMs early for better ILP ----
                        sK_tile_n0 = cute.local_tile(
                            csK, tiler=(8, 16), coord=(my_i_k * 2, k_block)
                        )
                        tCrK_n0 = tiled_mma_k2.make_fragment_B(
                            thr_mma.partition_B(sK_tile_n0)
                        )
                        cute.copy(
                            tiled_copy_mma_B,
                            thr_copy_B.partition_S(sK_tile_n0),
                            thr_copy_B.retile(tCrK_n0),
                        )

                        sK_tile_n1 = cute.local_tile(
                            csK, tiler=(8, 16), coord=(my_i_k * 2 + 1, k_block)
                        )
                        tCrK_n1 = tiled_mma_k2.make_fragment_B(
                            thr_mma.partition_B(sK_tile_n1)
                        )
                        cute.copy(
                            tiled_copy_mma_B,
                            thr_copy_B.partition_S(sK_tile_n1),
                            thr_copy_B.retile(tCrK_n1),
                        )

                        # ---- Gate norm (2x k=8 covers k=16) ----
                        sGn_a = cute.local_tile(
                            csGcum, tiler=(1, 8), coord=(norm_row, k_block * 2)
                        )
                        tCsGn_a = thr_copy_Gn.partition_S(sGn_a)
                        tCrGn_a = cute.make_fragment_like(tCsGn_a, cutlass.Float32)
                        cute.copy(
                            tiled_copy_Gcum_norm, tCsGn_a, thr_copy_Gn.retile(tCrGn_a)
                        )
                        gn_a0 = tCrGn_a[0]
                        gn_a1 = tCrGn_a[1]

                        sGn_b = cute.local_tile(
                            csGcum, tiler=(1, 8), coord=(norm_row, k_block * 2 + 1)
                        )
                        tCsGn_b = thr_copy_Gn.partition_S(sGn_b)
                        tCrGn_b = cute.make_fragment_like(tCsGn_b, cutlass.Float32)
                        cute.copy(
                            tiled_copy_Gcum_norm, tCsGn_b, thr_copy_Gn.retile(tCrGn_b)
                        )
                        gn_b0 = tCrGn_b[0]
                        gn_b1 = tCrGn_b[1]

                        # ---- Gate Q (2x (16,8) partition_C covers m=16,k=16) ----
                        sGq_a = cute.local_tile(
                            csGcum, tiler=(16, 8), coord=(my_i_q, k_block * 2)
                        )
                        tCrGq_a = tiled_mma_k2.make_fragment_C(
                            thr_mma.partition_C(sGq_a)
                        )
                        cute.copy(
                            tiled_copy_Gcum_gate,
                            thr_copy_Ggate.partition_S(sGq_a),
                            thr_copy_Ggate.retile(tCrGq_a),
                        )

                        sGq_b = cute.local_tile(
                            csGcum, tiler=(16, 8), coord=(my_i_q, k_block * 2 + 1)
                        )
                        tCrGq_b = tiled_mma_k2.make_fragment_C(
                            thr_mma.partition_C(sGq_b)
                        )
                        cute.copy(
                            tiled_copy_Gcum_gate,
                            thr_copy_Ggate.partition_S(sGq_b),
                            thr_copy_Ggate.retile(tCrGq_b),
                        )

                        # 8 Q gate values per thread (matching A bf16 m16n8k16 layout):
                        # first half k=0..7 (tCrGq_a): a0=(r0,c0) a1=(r0,c0+1) a2=(r0+8,c0) a3=(r0+8,c0+1)
                        # second half k=8..15 (tCrGq_b): a4=(r0,c0+8) a5=(r0,c0+9) a6=(r0+8,c0+8) a7=(r0+8,c0+9)
                        gate_q_0 = cute.exp2(tCrGq_a[0] - gn_a0, fastmath=True)
                        gate_q_1 = cute.exp2(tCrGq_a[1] - gn_a1, fastmath=True)
                        gate_q_2 = cute.exp2(tCrGq_a[2] - gn_a0, fastmath=True)
                        gate_q_3 = cute.exp2(tCrGq_a[3] - gn_a1, fastmath=True)
                        gate_q_4 = cute.exp2(tCrGq_b[0] - gn_b0, fastmath=True)
                        gate_q_5 = cute.exp2(tCrGq_b[1] - gn_b1, fastmath=True)
                        gate_q_6 = cute.exp2(tCrGq_b[2] - gn_b0, fastmath=True)
                        gate_q_7 = cute.exp2(tCrGq_b[3] - gn_b1, fastmath=True)

                        # qa fp32 = Q*gate (8 per thread). tCrQ indexing assumption:
                        # [0..3] first k-chunk (k=0..7), [4..7] second k-chunk (k=8..15).
                        qa0 = tCrQ[0].to(cutlass.Float32) * gate_q_0
                        qa1 = tCrQ[1].to(cutlass.Float32) * gate_q_1
                        qa2 = tCrQ[2].to(cutlass.Float32) * gate_q_2
                        qa3 = tCrQ[3].to(cutlass.Float32) * gate_q_3
                        qa4 = tCrQ[4].to(cutlass.Float32) * gate_q_4
                        qa5 = tCrQ[5].to(cutlass.Float32) * gate_q_5
                        qa6 = tCrQ[6].to(cutlass.Float32) * gate_q_6
                        qa7 = tCrQ[7].to(cutlass.Float32) * gate_q_7

                        # Pack fp32 qa -> 4 u32 bf16x2 for MMA A (k-adjacent pairs per u32)
                        qa_u32_0 = pack_bf16x2_f32(qa1, qa0)  # reg0 = [qa0 lo | qa1 hi]
                        qa_u32_1 = pack_bf16x2_f32(qa3, qa2)
                        qa_u32_2 = pack_bf16x2_f32(qa5, qa4)
                        qa_u32_3 = pack_bf16x2_f32(qa7, qa6)

                        ka0 = tCrKq[0].to(cutlass.Float32) * gate_q_0
                        ka1 = tCrKq[1].to(cutlass.Float32) * gate_q_1
                        ka2 = tCrKq[2].to(cutlass.Float32) * gate_q_2
                        ka3 = tCrKq[3].to(cutlass.Float32) * gate_q_3
                        ka4 = tCrKq[4].to(cutlass.Float32) * gate_q_4
                        ka5 = tCrKq[5].to(cutlass.Float32) * gate_q_5
                        ka6 = tCrKq[6].to(cutlass.Float32) * gate_q_6
                        ka7 = tCrKq[7].to(cutlass.Float32) * gate_q_7

                        ka_u32_0 = pack_bf16x2_f32(ka1, ka0)
                        ka_u32_1 = pack_bf16x2_f32(ka3, ka2)
                        ka_u32_2 = pack_bf16x2_f32(ka5, ka4)
                        ka_u32_3 = pack_bf16x2_f32(ka7, ka6)

                        # ---- Gate K (2x (16,8) partition_C covers m=16,k=16) ----
                        sGk_a = cute.local_tile(
                            csGcum, tiler=(16, 8), coord=(my_i_k, k_block * 2)
                        )
                        tCrGk_a = tiled_mma_k2.make_fragment_C(
                            thr_mma.partition_C(sGk_a)
                        )
                        cute.copy(
                            tiled_copy_Gcum_gate,
                            thr_copy_Ggate.partition_S(sGk_a),
                            thr_copy_Ggate.retile(tCrGk_a),
                        )

                        sGk_b = cute.local_tile(
                            csGcum, tiler=(16, 8), coord=(my_i_k, k_block * 2 + 1)
                        )
                        tCrGk_b = tiled_mma_k2.make_fragment_C(
                            thr_mma.partition_C(sGk_b)
                        )
                        cute.copy(
                            tiled_copy_Gcum_gate,
                            thr_copy_Ggate.partition_S(sGk_b),
                            thr_copy_Ggate.retile(tCrGk_b),
                        )

                        # n0 uses rows 0..7 of (16,*) tile (tCrGk_*[0,1])
                        # n1 uses rows 8..15 of (16,*) tile (tCrGk_*[2,3])
                        gk_n0_0 = cute.exp2(gn_a0 - tCrGk_a[0], fastmath=True)
                        gk_n0_1 = cute.exp2(gn_a1 - tCrGk_a[1], fastmath=True)
                        gk_n0_2 = cute.exp2(gn_b0 - tCrGk_b[0], fastmath=True)
                        gk_n0_3 = cute.exp2(gn_b1 - tCrGk_b[1], fastmath=True)

                        gk_n1_0 = cute.exp2(gn_a0 - tCrGk_a[2], fastmath=True)
                        gk_n1_1 = cute.exp2(gn_a1 - tCrGk_a[3], fastmath=True)
                        gk_n1_2 = cute.exp2(gn_b0 - tCrGk_b[2], fastmath=True)
                        gk_n1_3 = cute.exp2(gn_b1 - tCrGk_b[3], fastmath=True)

                        # tCrK_n0 bf16 fragment: 4 elems/thread at (n_row, c0), (n_row, c0+1),
                        # (n_row, c0+8), (n_row, c0+9) — k-adjacent pairs
                        k_n0_b0 = tCrK_n0[0].to(cutlass.Float32) * gk_n0_0
                        k_n0_b1 = tCrK_n0[1].to(cutlass.Float32) * gk_n0_1
                        k_n0_b2 = tCrK_n0[2].to(cutlass.Float32) * gk_n0_2
                        k_n0_b3 = tCrK_n0[3].to(cutlass.Float32) * gk_n0_3

                        k_n1_b0 = tCrK_n1[0].to(cutlass.Float32) * gk_n1_0
                        k_n1_b1 = tCrK_n1[1].to(cutlass.Float32) * gk_n1_1
                        k_n1_b2 = tCrK_n1[2].to(cutlass.Float32) * gk_n1_2
                        k_n1_b3 = tCrK_n1[3].to(cutlass.Float32) * gk_n1_3

                        # Pack fp32 k -> 2 u32 bf16x2 for MMA B
                        k_n0_u32_0 = pack_bf16x2_f32(k_n0_b1, k_n0_b0)
                        k_n0_u32_1 = pack_bf16x2_f32(k_n0_b3, k_n0_b2)
                        k_n1_u32_0 = pack_bf16x2_f32(k_n1_b1, k_n1_b0)
                        k_n1_u32_1 = pack_bf16x2_f32(k_n1_b3, k_n1_b2)

                        # ---- 4 bf16 MMA calls ----
                        acc_aqk_n0_0, acc_aqk_n0_1, acc_aqk_n0_2, acc_aqk_n0_3 = (
                            mma_bf16_m16n8k16(
                                qa_u32_0,
                                qa_u32_1,
                                qa_u32_2,
                                qa_u32_3,
                                k_n0_u32_0,
                                k_n0_u32_1,
                                acc_aqk_n0_0,
                                acc_aqk_n0_1,
                                acc_aqk_n0_2,
                                acc_aqk_n0_3,
                            )
                        )
                        acc_aqk_n1_0, acc_aqk_n1_1, acc_aqk_n1_2, acc_aqk_n1_3 = (
                            mma_bf16_m16n8k16(
                                qa_u32_0,
                                qa_u32_1,
                                qa_u32_2,
                                qa_u32_3,
                                k_n1_u32_0,
                                k_n1_u32_1,
                                acc_aqk_n1_0,
                                acc_aqk_n1_1,
                                acc_aqk_n1_2,
                                acc_aqk_n1_3,
                            )
                        )
                        acc_akk_n0_0, acc_akk_n0_1, acc_akk_n0_2, acc_akk_n0_3 = (
                            mma_bf16_m16n8k16(
                                ka_u32_0,
                                ka_u32_1,
                                ka_u32_2,
                                ka_u32_3,
                                k_n0_u32_0,
                                k_n0_u32_1,
                                acc_akk_n0_0,
                                acc_akk_n0_1,
                                acc_akk_n0_2,
                                acc_akk_n0_3,
                            )
                        )
                        acc_akk_n1_0, acc_akk_n1_1, acc_akk_n1_2, acc_akk_n1_3 = (
                            mma_bf16_m16n8k16(
                                ka_u32_0,
                                ka_u32_1,
                                ka_u32_2,
                                ka_u32_3,
                                k_n1_u32_0,
                                k_n1_u32_1,
                                acc_akk_n1_0,
                                acc_akk_n1_1,
                                acc_akk_n1_2,
                                acc_akk_n1_3,
                            )
                        )

                    # sQ/sK/sG reads done, signal TMA before SMEM writes
                    cute.arch.mbarrier_arrive(stage_reuse_mbars + s)

                    # Dual-path MMA write (constexpr-gated):
                    # - non-pure: apply causal + diag=1 inline so SMEM matches
                    #   final GMEM layout (pairs with row-major vec autovec
                    #   store warp that does no causal).
                    # - pure: write all 16x16 unconditionally (pairs with
                    #   per-tile store warp that applies causal + diag in
                    #   store; this is the original baseline behavior — no
                    #   extra MMA-write cost).
                    _z16 = cutlass.BFloat16(0.0)
                    _one16 = cutlass.BFloat16(1.0)
                    if IS_VARLEN and not VARLEN_PURE:
                        if my_i_q == my_i_k:
                            # Diagonal sub-tile: causal-mask sAqk and write
                            # diag=1 / strict-lower=MMA*beta / strict-upper=0
                            # to sAkk so SMEM is in final form.
                            _v_q00 = (acc_aqk_n0_0 * scale).to(cutlass.BFloat16)
                            if row0 < col0:
                                _v_q00 = _z16
                            csAqk[q_row_base + row0, k_row_base + col0] = _v_q00
                            _v_q01 = (acc_aqk_n0_1 * scale).to(cutlass.BFloat16)
                            if row0 < col1:
                                _v_q01 = _z16
                            csAqk[q_row_base + row0, k_row_base + col1] = _v_q01
                            _v_q02 = (acc_aqk_n0_2 * scale).to(cutlass.BFloat16)
                            if row1 < col0:
                                _v_q02 = _z16
                            csAqk[q_row_base + row1, k_row_base + col0] = _v_q02
                            _v_q03 = (acc_aqk_n0_3 * scale).to(cutlass.BFloat16)
                            if row1 < col1:
                                _v_q03 = _z16
                            csAqk[q_row_base + row1, k_row_base + col1] = _v_q03
                            _v_q04 = (acc_aqk_n1_0 * scale).to(cutlass.BFloat16)
                            if row0 < col2:
                                _v_q04 = _z16
                            csAqk[q_row_base + row0, k_row_base + col2] = _v_q04
                            _v_q05 = (acc_aqk_n1_1 * scale).to(cutlass.BFloat16)
                            if row0 < col3:
                                _v_q05 = _z16
                            csAqk[q_row_base + row0, k_row_base + col3] = _v_q05
                            _v_q06 = (acc_aqk_n1_2 * scale).to(cutlass.BFloat16)
                            if row1 < col2:
                                _v_q06 = _z16
                            csAqk[q_row_base + row1, k_row_base + col2] = _v_q06
                            _v_q07 = (acc_aqk_n1_3 * scale).to(cutlass.BFloat16)
                            if row1 < col3:
                                _v_q07 = _z16
                            csAqk[q_row_base + row1, k_row_base + col3] = _v_q07

                            _v_k00 = (acc_akk_n0_0 * beta_row0).to(cutlass.BFloat16)
                            if row0 == col0:
                                _v_k00 = _one16
                            if row0 < col0:
                                _v_k00 = _z16
                            csAkk[akk_row_base + row0, akk_col_base + col0] = _v_k00
                            _v_k01 = (acc_akk_n0_1 * beta_row0).to(cutlass.BFloat16)
                            if row0 == col1:
                                _v_k01 = _one16
                            if row0 < col1:
                                _v_k01 = _z16
                            csAkk[akk_row_base + row0, akk_col_base + col1] = _v_k01
                            _v_k02 = (acc_akk_n0_2 * beta_row1).to(cutlass.BFloat16)
                            if row1 == col0:
                                _v_k02 = _one16
                            if row1 < col0:
                                _v_k02 = _z16
                            csAkk[akk_row_base + row1, akk_col_base + col0] = _v_k02
                            _v_k03 = (acc_akk_n0_3 * beta_row1).to(cutlass.BFloat16)
                            if row1 == col1:
                                _v_k03 = _one16
                            if row1 < col1:
                                _v_k03 = _z16
                            csAkk[akk_row_base + row1, akk_col_base + col1] = _v_k03
                            _v_k04 = (acc_akk_n1_0 * beta_row0).to(cutlass.BFloat16)
                            if row0 == col2:
                                _v_k04 = _one16
                            if row0 < col2:
                                _v_k04 = _z16
                            csAkk[akk_row_base + row0, akk_col_base + col2] = _v_k04
                            _v_k05 = (acc_akk_n1_1 * beta_row0).to(cutlass.BFloat16)
                            if row0 == col3:
                                _v_k05 = _one16
                            if row0 < col3:
                                _v_k05 = _z16
                            csAkk[akk_row_base + row0, akk_col_base + col3] = _v_k05
                            _v_k06 = (acc_akk_n1_2 * beta_row1).to(cutlass.BFloat16)
                            if row1 == col2:
                                _v_k06 = _one16
                            if row1 < col2:
                                _v_k06 = _z16
                            csAkk[akk_row_base + row1, akk_col_base + col2] = _v_k06
                            _v_k07 = (acc_akk_n1_3 * beta_row1).to(cutlass.BFloat16)
                            if row1 == col3:
                                _v_k07 = _one16
                            if row1 < col3:
                                _v_k07 = _z16
                            csAkk[akk_row_base + row1, akk_col_base + col3] = _v_k07
                        else:
                            # Non-diag (i_q > i_k): write all 16x16 unchanged.
                            csAqk[q_row_base + row0, k_row_base + col0] = (
                                acc_aqk_n0_0 * scale
                            ).to(cutlass.BFloat16)
                            csAqk[q_row_base + row0, k_row_base + col1] = (
                                acc_aqk_n0_1 * scale
                            ).to(cutlass.BFloat16)
                            csAqk[q_row_base + row1, k_row_base + col0] = (
                                acc_aqk_n0_2 * scale
                            ).to(cutlass.BFloat16)
                            csAqk[q_row_base + row1, k_row_base + col1] = (
                                acc_aqk_n0_3 * scale
                            ).to(cutlass.BFloat16)
                            csAqk[q_row_base + row0, k_row_base + col2] = (
                                acc_aqk_n1_0 * scale
                            ).to(cutlass.BFloat16)
                            csAqk[q_row_base + row0, k_row_base + col3] = (
                                acc_aqk_n1_1 * scale
                            ).to(cutlass.BFloat16)
                            csAqk[q_row_base + row1, k_row_base + col2] = (
                                acc_aqk_n1_2 * scale
                            ).to(cutlass.BFloat16)
                            csAqk[q_row_base + row1, k_row_base + col3] = (
                                acc_aqk_n1_3 * scale
                            ).to(cutlass.BFloat16)
                            csAkk[akk_row_base + row0, akk_col_base + col0] = (
                                acc_akk_n0_0 * beta_row0
                            ).to(cutlass.BFloat16)
                            csAkk[akk_row_base + row0, akk_col_base + col1] = (
                                acc_akk_n0_1 * beta_row0
                            ).to(cutlass.BFloat16)
                            csAkk[akk_row_base + row1, akk_col_base + col0] = (
                                acc_akk_n0_2 * beta_row1
                            ).to(cutlass.BFloat16)
                            csAkk[akk_row_base + row1, akk_col_base + col1] = (
                                acc_akk_n0_3 * beta_row1
                            ).to(cutlass.BFloat16)
                            csAkk[akk_row_base + row0, akk_col_base + col2] = (
                                acc_akk_n1_0 * beta_row0
                            ).to(cutlass.BFloat16)
                            csAkk[akk_row_base + row0, akk_col_base + col3] = (
                                acc_akk_n1_1 * beta_row0
                            ).to(cutlass.BFloat16)
                            csAkk[akk_row_base + row1, akk_col_base + col2] = (
                                acc_akk_n1_2 * beta_row1
                            ).to(cutlass.BFloat16)
                            csAkk[akk_row_base + row1, akk_col_base + col3] = (
                                acc_akk_n1_3 * beta_row1
                            ).to(cutlass.BFloat16)
                    else:
                        # PURE: write all 16x16 unconditionally — store warp
                        # applies causal+diag in its per-tile loop. This is
                        # the original baseline behavior (no extra MMA cost).
                        csAqk[q_row_base + row0, k_row_base + col0] = (
                            acc_aqk_n0_0 * scale
                        ).to(cutlass.BFloat16)
                        csAqk[q_row_base + row0, k_row_base + col1] = (
                            acc_aqk_n0_1 * scale
                        ).to(cutlass.BFloat16)
                        csAqk[q_row_base + row1, k_row_base + col0] = (
                            acc_aqk_n0_2 * scale
                        ).to(cutlass.BFloat16)
                        csAqk[q_row_base + row1, k_row_base + col1] = (
                            acc_aqk_n0_3 * scale
                        ).to(cutlass.BFloat16)
                        csAqk[q_row_base + row0, k_row_base + col2] = (
                            acc_aqk_n1_0 * scale
                        ).to(cutlass.BFloat16)
                        csAqk[q_row_base + row0, k_row_base + col3] = (
                            acc_aqk_n1_1 * scale
                        ).to(cutlass.BFloat16)
                        csAqk[q_row_base + row1, k_row_base + col2] = (
                            acc_aqk_n1_2 * scale
                        ).to(cutlass.BFloat16)
                        csAqk[q_row_base + row1, k_row_base + col3] = (
                            acc_aqk_n1_3 * scale
                        ).to(cutlass.BFloat16)
                        csAkk[akk_row_base + row0, akk_col_base + col0] = (
                            acc_akk_n0_0 * beta_row0
                        ).to(cutlass.BFloat16)
                        csAkk[akk_row_base + row0, akk_col_base + col1] = (
                            acc_akk_n0_1 * beta_row0
                        ).to(cutlass.BFloat16)
                        csAkk[akk_row_base + row1, akk_col_base + col0] = (
                            acc_akk_n0_2 * beta_row1
                        ).to(cutlass.BFloat16)
                        csAkk[akk_row_base + row1, akk_col_base + col1] = (
                            acc_akk_n0_3 * beta_row1
                        ).to(cutlass.BFloat16)
                        csAkk[akk_row_base + row0, akk_col_base + col2] = (
                            acc_akk_n1_0 * beta_row0
                        ).to(cutlass.BFloat16)
                        csAkk[akk_row_base + row0, akk_col_base + col3] = (
                            acc_akk_n1_1 * beta_row0
                        ).to(cutlass.BFloat16)
                        csAkk[akk_row_base + row1, akk_col_base + col2] = (
                            acc_akk_n1_2 * beta_row1
                        ).to(cutlass.BFloat16)
                        csAkk[akk_row_base + row1, akk_col_base + col3] = (
                            acc_akk_n1_3 * beta_row1
                        ).to(cutlass.BFloat16)
                else:
                    cute.arch.mbarrier_arrive(stage_reuse_mbars + s)

                cute.arch.mbarrier_arrive(mma_done_mbars + s)

        # =============================================================
        # Warps 28-31: Store/Inversion warps
        # =============================================================
        if warp_idx >= NUM_K1_TMA_WARPS + NUM_MMA_WARPS:
            store_warp = warp_idx - (NUM_K1_TMA_WARPS + NUM_MMA_WARPS)
            for chunk_iter in cutlass.range_constexpr(CHUNKS_PER_BLOCK):
                s = chunk_iter % NUM_STAGES
                phase = chunk_iter // NUM_STAGES % 2
                chunk_idx = chunk_base + chunk_iter
                chunk_start = cutlass.Int32(0)
                st_eos = cutlass.Int32(0)
                if IS_VARLEN:
                    if chunk_idx < num_chunks:
                        _sid = cutlass.Int32(mChunkIndices[chunk_idx, 0])
                        chunk_start = (
                            cutlass.Int32(mCuSeqlens[_sid])
                            + cutlass.Int32(mChunkIndices[chunk_idx, 1]) * BT
                        )
                        st_eos = cutlass.Int32(mCuSeqlens[_sid + 1])
                else:
                    chunk_start = chunk_idx * BT

                cute.arch.mbarrier_wait(mma_done_mbars + s, phase)

                csAqk = sAqk[(None, None, s)]
                csAkk = sAkk[(None, None, s)]

                # Dual-path row-major store. SMEM is in final GMEM layout
                # already (causal mask + diag=1 applied at MMA write; upper-tri
                # SMEM is zero from CTA-startup init). Both paths use vec2
                # autovec_copy → STG.E.32 (4-byte coalesced).
                #
                # mAqk_v2 / mAkk_v2 shape (B, T, H, BT/2, 2): last dim is vec2.
                #
                # NON-PURE: full 64-col row-major + row mask (chunk may
                #   overflow seq end). 32 lanes/warp × 1 vec2 = full row.
                # PURE: reduced cols. Warp s writes (s+1)*16 cols/row (no row
                #   mask, all rows in seq). Saves ~37% GMEM bandwidth vs full.
                col_lo = lane_id * 2
                col_hi = col_lo + 1
                col_vec_idx = lane_id
                rAqkOut = cute.make_rmem_tensor(
                    cute.make_layout((VEC,)), cutlass.BFloat16
                )
                rAkkOut = cute.make_rmem_tensor(
                    cute.make_layout((VEC,)), cutlass.BFloat16
                )

                if IS_VARLEN and not VARLEN_PURE:
                    # NON-PURE: row-major full-row vec autovec + row mask. SMEM
                    # upper-tri is zero (MMA-masked at write), so writing all 64
                    # cols is correct. STG.E.32 (32 lanes × 4 bytes per row).
                    row_base_warp = store_warp * (BT // NUM_STORE_WARPS)
                    for ri in cutlass.range_constexpr(BT // NUM_STORE_WARPS):
                        local_row = row_base_warp + ri
                        abs_row = chunk_start + local_row
                        rAqkOut[0] = csAqk[local_row, col_lo]
                        rAqkOut[1] = csAqk[local_row, col_hi]
                        rAkkOut[0] = csAkk[local_row, col_lo]
                        rAkkOut[1] = csAkk[local_row, col_hi]
                        if abs_row < st_eos:
                            cute.autovec_copy(
                                rAqkOut, mAqk_v2[i_b, abs_row, i_h, col_vec_idx, None]
                            )
                            cute.autovec_copy(
                                rAkkOut, mAkk_v2[i_b, abs_row, i_h, col_vec_idx, None]
                            )
                else:
                    # PURE: per-tile loop over 10 lower-tri sub-tiles, reading
                    # row-major sAqk/sAkk. MMA writes 16x16 unconditional so we
                    # must apply causal+diag mask in store. Each warp handles
                    # BC/4=4 rows per sub-tile, lanes 0..15 write 1 bf16 each.
                    # Original baseline behavior — no MMA-write overhead.
                    for tile_idx in cutlass.range_constexpr(NUM_TILES):
                        i_q = _TILE_IQ[tile_idx]
                        i_k = _TILE_IK[tile_idx]
                        is_diag = _TILE_IQ[tile_idx] == _TILE_IK[tile_idx]
                        gmem_aqk_row_base = chunk_start + i_q * BC
                        gmem_aqk_col_base = i_k * BC
                        gmem_akk_row_base = chunk_start + i_k * BC
                        gmem_akk_col_base = i_q * BC
                        smem_aqk_row_base = i_q * BC
                        smem_aqk_col_base = i_k * BC
                        smem_akk_row_base = i_k * BC
                        smem_akk_col_base = i_q * BC
                        for ri in cutlass.range_constexpr(BC // NUM_STORE_WARPS):
                            local_row = store_warp * (BC // NUM_STORE_WARPS) + ri
                            if lane_id < BC:
                                local_col = lane_id
                                aqk_val = csAqk[
                                    smem_aqk_row_base + local_row,
                                    smem_aqk_col_base + local_col,
                                ]
                                akk_val = csAkk[
                                    smem_akk_row_base + local_row,
                                    smem_akk_col_base + local_col,
                                ]
                                if is_diag and local_row < local_col:
                                    aqk_val = cutlass.BFloat16(0.0)
                                if is_diag and local_row < local_col:
                                    akk_val = cutlass.BFloat16(0.0)
                                if is_diag and local_row == local_col:
                                    akk_val = cutlass.BFloat16(1.0)
                                mAqk[
                                    i_b,
                                    gmem_aqk_row_base + local_row,
                                    i_h,
                                    gmem_aqk_col_base + local_col,
                                ] = aqk_val
                                mAkk[
                                    i_b,
                                    gmem_akk_row_base + local_row,
                                    i_h,
                                    gmem_akk_col_base + local_col,
                                ] = akk_val

                cute.arch.mbarrier_arrive(store_done_mbars + s)

        yield_out()


# =========================================================================
# Host function
# =========================================================================
def make_host_function(
    B,
    NT,
    H,
    is_varlen=False,
    T_padded=None,
    has_bias=False,
    use_safe_gate=False,
    varlen_pure=False,
):
    """
    `varlen_pure=True` asserts that all seq lengths in the batch are multiples
    of BT (= 64). Under that assumption every chunk has 64 valid rows so the
    four mask sites (K1 row mask, K1 store mask, MMA accumulator zero-fill,
    Store row mask) are guaranteed to never fire and are dead-code eliminated
    at compile time. Caller's data layout is unchanged — this is a hint only.
    """
    _B, _NT, _H = B, NT, H
    _IS_VARLEN = 1 if is_varlen else 0
    _HAS_BIAS = 1 if has_bias else 0
    _USE_SAFE_GATE = 1 if use_safe_gate else 0
    _VARLEN_PURE = 1 if (is_varlen and varlen_pure) else 0
    if is_varlen:
        assert _B == 1, "Varlen requires B=1"
        assert T_padded is not None, "T_padded required for varlen"
        _T = T_padded
    else:
        _T = _NT * BT

    if is_varlen:
        _total_cgs_val = ((_NT + CHUNKS_PER_BLOCK - 1) // CHUNKS_PER_BLOCK) * _H
    else:
        _total_cgs_val = (_NT // CHUNKS_PER_BLOCK) * _H * _B

    # 3D TMA view: (B*T, K_DIM, H) — domain_offset handles non-aligned addressing
    _T_total = _B * _T
    s_row = _H * K_DIM
    s_col = 1
    s_h = K_DIM

    @cute.jit
    def host_fn(
        mQ,
        mK,
        mG,
        mA_log,
        mBeta,
        scale,
        mKscaled,
        mKg,
        mQscaled,
        mGkLast,
        mAqk,
        mAkk,
        mCuSeqlens,
        mChunkIndices,
        mDtBias,
        lower_bound_val,
        stream: cuda_drv.CUstream,
    ):
        # 3D TMA view: (B*T, K_DIM, H). domain_offset in kernel shifts to
        # arbitrary chunk_start — no BT alignment required for varlen.
        view_layout_3d = cute.make_layout(
            (_T_total, K_DIM, _H),
            stride=(s_row, s_col, s_h),
        )
        mQ_view = cute.make_tensor(mQ.iterator, view_layout_3d)
        mK_view = cute.make_tensor(mK.iterator, view_layout_3d)
        mG_view = cute.make_tensor(mG.iterator, view_layout_3d)

        smem_atom_qk = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        qk_smem_2d = cute.tile_to_shape(smem_atom_qk, (BT, K_DIM), order=(0, 1))
        qk_smem_3d = cute.tile_to_shape(
            smem_atom_qk, (BT, K_DIM, NUM_STAGES), order=(0, 1, 2)
        )

        g_smem_2d = cute.make_layout((BT, K_DIM), stride=(K_DIM, 1))
        g_smem_3d = cute.make_layout(
            (BT, K_DIM, NUM_STAGES), stride=(K_DIM, 1, BT * K_DIM)
        )

        tma_op = cpasync.CopyBulkTensorTileG2SOp(cpasync.CtaGroup.ONE)
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            tma_op,
            mQ_view,
            qk_smem_2d,
            cute.product_each(qk_smem_2d.shape),
            num_multicast=1,
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            tma_op,
            mK_view,
            qk_smem_2d,
            cute.product_each(qk_smem_2d.shape),
            num_multicast=1,
        )
        tma_atom_G, tma_tensor_G = cpasync.make_tiled_tma_atom(
            tma_op,
            mG_view,
            g_smem_2d,
            cute.product_each(g_smem_2d.shape),
            num_multicast=1,
        )

        g_cumsum_layout = cute.make_layout(
            (BT, K_DIM, NUM_STAGES), stride=(K_STRIDE, 1, BT * K_STRIDE)
        )

        copy_atom_qk_k1 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=32
        )
        tiled_copy_qk_k1 = cute.make_tiled_copy_tv(
            copy_atom_qk_k1,
            thr_layout=cute.make_layout((1, 32)),
            val_layout=cute.make_layout((1, 2)),
        )

        out_v2_layout = cute.make_layout(
            (_B, _T, _H, K_VEC, VEC),
            stride=(_T * _H * K_DIM, _H * K_DIM, K_DIM, VEC, 1),
        )
        mKscaled_v2 = cute.make_tensor(mKscaled.iterator, out_v2_layout)
        mQscaled_v2 = cute.make_tensor(mQscaled.iterator, out_v2_layout)
        mKg_v2 = cute.make_tensor(mKg.iterator, out_v2_layout)

        # vec2 views for mAqk / mAkk (BT dimension instead of K_DIM).
        # Shape: (B, T, H, BT/2, 2). Each VEC=2 slot = 2 contiguous bf16 = 1
        # fp32-aligned 4-byte unit. Used by store warp's autovec_copy →
        # STG.E.32 with 32 lanes coalesced to 1 cache line per row.
        BT_VEC = BT // VEC  # 32
        akk_v2_layout = cute.make_layout(
            (_B, _T, _H, BT_VEC, VEC),
            stride=(_T * _H * BT, _H * BT, BT, VEC, 1),
        )
        mAqk_v2 = cute.make_tensor(mAqk.iterator, akk_v2_layout)
        mAkk_v2 = cute.make_tensor(mAkk.iterator, akk_v2_layout)

        gklast_v2_layout = cute.make_layout(
            (_B, _NT, _H, K_VEC, VEC),
            stride=(_NT * _H * K_DIM, _H * K_DIM, K_DIM, VEC, 1),
        )
        mGkLast_v2 = cute.make_tensor(mGkLast.iterator, gklast_v2_layout)

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (16, 8, 16)
        )
        tiled_mma_k2 = cute.make_tiled_mma(
            mma_op, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 8, 16)
        )

        tiled_copy_mma_A = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cutlass.BFloat16
            ),
            tiled_mma_k2,
        )
        tiled_copy_mma_B = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 2), cutlass.BFloat16
            ),
            tiled_mma_k2,
        )

        copy_atom_Gcum = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=64
        )
        tiled_copy_Gcum_norm = cute.make_tiled_copy_tv(
            copy_atom_Gcum,
            thr_layout=cute.make_layout((1, 4)),
            val_layout=cute.make_layout((1, 2)),
        )
        tiled_copy_Gcum_gate = cute.make_tiled_copy_C(copy_atom_Gcum, tiled_mma_k2)

        smem_size = (
            BT * K_DIM * 2 * 2 * NUM_STAGES
            + BT * K_DIM * 2 * NUM_STAGES
            + BT * K_STRIDE * 4 * NUM_STAGES
            + K1_ROW_GROUPS * PARTIAL_COLS * 4
            + BT * AQK_TILE_STRIDE * 2 * NUM_STAGES  # sAqk bf16 (64x72 row-major)
            + BT * AKK_STRIDE * 2 * NUM_STAGES  # sAkk bf16
            + 512
        )

        _grid_x = min(NUM_SMS, _total_cgs_val)

        fused_kernel123(
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_G,
            tma_tensor_G,
            mA_log,
            mBeta,
            scale,
            mKscaled_v2,
            mKg_v2,
            mQscaled_v2,
            mGkLast_v2,
            mAqk,
            mAkk,
            mAqk_v2,
            mAkk_v2,
            tiled_copy_qk_k1,
            tiled_mma_k2,
            tiled_copy_mma_A,
            tiled_copy_mma_B,
            tiled_copy_Gcum_norm,
            tiled_copy_Gcum_gate,
            qk_smem_3d,
            g_smem_3d,
            g_cumsum_layout,
            _NT,
            _H,
            _B,
            mCuSeqlens,
            mChunkIndices,
            _IS_VARLEN,
            mDtBias,
            lower_bound_val,
            _HAS_BIAS,
            _USE_SAFE_GATE,
            _VARLEN_PURE,
        ).launch(
            grid=(_grid_x, 1, 1),
            block=(THREADS, 1, 1),
            smem=smem_size,
            stream=stream,
        )

    return host_fn
