# Vendored from the NVIDIA KDA_prefill package (benchmark/ Blackwell path)
# for the Kimi-K3 chunked prefill forward. Local deltas: fla.* imports
# re-pointed to sglang's vendored fla subset, flat sibling imports made
# package-relative, RCP_LN2 inlined.
# ruff: noqa  -- vendored kernel library, minimal local deltas
"""
Akk 64×64 Lower Triangular Block Inversion — Full BF16.

BF16 input, BF16 SMEM (packed bf16x2), BF16 MMA m16n8k16 (FP32 accum), BF16 output.
Numerical precision aligned with Flash Attention / KDA kernels.

  (I+L)⁻¹ = (I-L)(I+L²)(I+L⁴)(I+L⁸)   [L strictly lower triangular, L¹⁶=0]

Architecture:
  - 4 warps (128 threads) per CTA, each CTA processes one 64×64 Akk matrix
  - sAkk [64,36] fp32 — each FP32 slot holds packed bf16x2 (stride 36)
    Total SMEM: 64*36*4 = 9216 bytes (vs 64*72*4 = 18432 in FP32 version)
  - sTemp [16, 24, 2] fp32: inter-warp FP32 accumulator communication
  - cp.async: BF16 global → packed bf16x2 in FP32 SMEM (raw bytes align)
  - BF16 MMA m16n8k16 with FP32 accumulator for ALL matmuls
  - movmatrix.sync.aligned.m8n8.trans.b16 for A→B layout conversion
  - C→A chain: cvt.rn.bf16x2.f32 to pack FP32 accum → bf16x2 A-operand

Block layout in sAkk (4×4 sub-blocks of 16×16, packed bf16x2):
  Upper tri = INPUT, Diagonal = in-place inversion, Lower tri = OUTPUT

Stages:
  0. cp.async load bf16 64×64 → sAkk (packed bf16x2)
  1. Invert 4 diagonal blocks via Neumann series (all 4 warps, BF16 MMA)
  2. Warps 0-2: Ai10, Ai21, Ai32 via chain MMA (C→A pack)
  3. Warps 0+2 → Ai20, warps 1+3 → Ai31 (parallel pairs, sTemp)
  4. Warps 0+1+2 → Ai30 (sTemp aggregation)
  5. All warps: store bf16x2 SMEM → bf16 global

Inputs:  A_in  [B, T, H, BT] bf16
Outputs: A_out [B, T, H, BT] bf16
"""

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op

BS = 64
SB = 16
THREADS = 128
TEMP_PAD = 8
TEMP_COLS = SB + TEMP_PAD  # 24
NUM_TEMPS = 2
AKK_PAD = 4
AKK_STRIDE = BS // 2 + AKK_PAD  # 36 (in FP32 units = 72 bf16 elements)


@dsl_user_op
def mma_bf16_m16n8k16(
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    c0,
    c1,
    c2,
    c3,
    *,
    loc=None,
    ip=None,
):
    """mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    A: 4×i32 (bf16x2 pairs), B: 2×i32, C/D: 4×f32 (FP32 accum)."""
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


# Warp-level hardware A->B transpose; works on any 16-bit format including BF16.
@dsl_user_op
def _movmatrix_trans(src, *, loc=None, ip=None):
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
def _pack_bf16x2(lo_f32, hi_f32, *, loc=None, ip=None):
    """Pack two FP32 values into one i32 holding two BF16 values.
    cvt.rn.bf16x2.f32 converts and packs in one instruction."""
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
def _mask_packed_ltri(packed, row, pair, *, loc=None, ip=None):
    """Apply lower-triangular mask to packed bf16x2.
    Zero out elements where row < col. col0=2*pair, col1=2*pair+1."""
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
def _unpack_bf16x2_lo(packed, *, loc=None, ip=None):
    """Unpack lower BF16 from packed i32 to FP32.
    BF16 is upper 16 bits of FP32, so shift left by 16."""
    p = llvm.bitcast(T.i32(), packed.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.i32(),
        [p],
        """{
            shl.b32 $0, $1, 16;
        }""",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(llvm.bitcast(T.f32(), result, loc=loc, ip=ip))


@dsl_user_op
def _unpack_bf16x2_hi(packed, *, loc=None, ip=None):
    """Unpack upper BF16 from packed i32 to FP32.
    Upper 16 bits are already in the right position for FP32."""
    p = llvm.bitcast(T.i32(), packed.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.i32(),
        [p],
        """{
            and.b32 $0, $1, 0xFFFF0000;
        }""",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(llvm.bitcast(T.f32(), result, loc=loc, ip=ip))


# Neumann diagonal 16×16 inversion using BF16 MMA m16n8k16 + FP32 accum
#
# (I+L)⁻¹ = (I-L)(I+L²)(I+L⁴)(I+L⁸)
#
# All intermediate values kept in FP32. Pack to bf16x2 only at MMA boundaries.
# sAkk is packed bf16x2: sAkk[row, pair] = {bf16[row, 2*pair], bf16[row, 2*pair+1]}
@dsl_user_op
def _invert_diag_neumann(sAkk: cute.Tensor, block_idx, lane_id, *, loc=None, ip=None):
    r_off = block_idx * 16
    c_off = block_idx * 8  # packed bf16x2: 16 cols → 8 pairs
    gid = lane_id // 4
    tid = lane_id % 4

    # --- Load A from packed bf16x2 SMEM → unpack to 8 FP32 values ---
    packed0 = cutlass.Float32(sAkk[r_off + gid, c_off + tid])
    packed1 = cutlass.Float32(sAkk[r_off + gid + 8, c_off + tid])
    packed2 = cutlass.Float32(sAkk[r_off + gid, c_off + 4 + tid])
    packed3 = cutlass.Float32(sAkk[r_off + gid + 8, c_off + 4 + tid])

    A_f0 = _unpack_bf16x2_lo(packed0)  # A[gid, 2*tid]
    A_f1 = _unpack_bf16x2_hi(packed0)  # A[gid, 2*tid+1]
    A_f2 = _unpack_bf16x2_lo(packed1)  # A[gid+8, 2*tid]
    A_f3 = _unpack_bf16x2_hi(packed1)  # A[gid+8, 2*tid+1]
    A_f4 = _unpack_bf16x2_lo(packed2)  # A[gid, 8+2*tid]
    A_f5 = _unpack_bf16x2_hi(packed2)  # A[gid, 8+2*tid+1]
    A_f6 = _unpack_bf16x2_lo(packed3)  # A[gid+8, 8+2*tid]
    A_f7 = _unpack_bf16x2_hi(packed3)  # A[gid+8, 8+2*tid+1]

    # Build identity (FP32)
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

    # L = A - I (FP32)
    L_f0 = A_f0 - I_f0
    L_f1 = A_f1 - I_f1
    L_f2 = A_f2 - I_f2
    L_f3 = A_f3 - I_f3
    L_f4 = A_f4 - I_f4
    L_f5 = A_f5 - I_f5
    L_f6 = A_f6 - I_f6
    L_f7 = A_f7 - I_f7

    # INV = I - L = 2I - A (FP32)
    INV_f0 = I_f0 - L_f0
    INV_f1 = I_f1 - L_f1
    INV_f2 = I_f2 - L_f2
    INV_f3 = I_f3 - L_f3
    INV_f4 = I_f4 - L_f4
    INV_f5 = I_f5 - L_f5
    INV_f6 = I_f6 - L_f6
    INV_f7 = I_f7 - L_f7

    _zf = cutlass.Float32(0.0)

    # Pack L and INV → bf16x2 for MMA
    L_a0 = _pack_bf16x2(L_f0, L_f1)
    L_a1 = _pack_bf16x2(L_f2, L_f3)
    L_a2 = _pack_bf16x2(L_f4, L_f5)
    L_a3 = _pack_bf16x2(L_f6, L_f7)

    INV_a0 = _pack_bf16x2(INV_f0, INV_f1)
    INV_a1 = _pack_bf16x2(INV_f2, INV_f3)
    INV_a2 = _pack_bf16x2(INV_f4, INV_f5)
    INV_a3 = _pack_bf16x2(INV_f6, INV_f7)

    # === Iteration 1: L² = L × L, then INV = INV + INV × L² ===
    L_b0 = _movmatrix_trans(L_a0)
    L_b1 = _movmatrix_trans(L_a1)
    L_b2 = _movmatrix_trans(L_a2)
    L_b3 = _movmatrix_trans(L_a3)

    Lp_c0, Lp_c1, Lp_c2, Lp_c3 = mma_bf16_m16n8k16(
        L_a0, L_a1, L_a2, L_a3, L_b0, L_b1, _zf, _zf, _zf, _zf
    )
    Lp_c4, Lp_c5, Lp_c6, Lp_c7 = mma_bf16_m16n8k16(
        L_a0, L_a1, L_a2, L_a3, L_b2, L_b3, _zf, _zf, _zf, _zf
    )

    Lp_a0 = _pack_bf16x2(Lp_c0, Lp_c1)
    Lp_a1 = _pack_bf16x2(Lp_c2, Lp_c3)
    Lp_a2 = _pack_bf16x2(Lp_c4, Lp_c5)
    Lp_a3 = _pack_bf16x2(Lp_c6, Lp_c7)

    Lp_b0 = _movmatrix_trans(Lp_a0)
    Lp_b1 = _movmatrix_trans(Lp_a1)
    Lp_b2 = _movmatrix_trans(Lp_a2)
    Lp_b3 = _movmatrix_trans(Lp_a3)

    # mm = INV × L²  (FP32 accum output)
    mm_c0, mm_c1, mm_c2, mm_c3 = mma_bf16_m16n8k16(
        INV_a0, INV_a1, INV_a2, INV_a3, Lp_b0, Lp_b1, _zf, _zf, _zf, _zf
    )
    mm_c4, mm_c5, mm_c6, mm_c7 = mma_bf16_m16n8k16(
        INV_a0, INV_a1, INV_a2, INV_a3, Lp_b2, Lp_b3, _zf, _zf, _zf, _zf
    )

    # INV += mm (FP32)
    INV_f0 = INV_f0 + mm_c0
    INV_f1 = INV_f1 + mm_c1
    INV_f2 = INV_f2 + mm_c2
    INV_f3 = INV_f3 + mm_c3
    INV_f4 = INV_f4 + mm_c4
    INV_f5 = INV_f5 + mm_c5
    INV_f6 = INV_f6 + mm_c6
    INV_f7 = INV_f7 + mm_c7

    INV_a0 = _pack_bf16x2(INV_f0, INV_f1)
    INV_a1 = _pack_bf16x2(INV_f2, INV_f3)
    INV_a2 = _pack_bf16x2(INV_f4, INV_f5)
    INV_a3 = _pack_bf16x2(INV_f6, INV_f7)

    # === Iteration 2: L⁴ = L² × L², then INV = INV + INV × L⁴ ===
    L4_c0, L4_c1, L4_c2, L4_c3 = mma_bf16_m16n8k16(
        Lp_a0, Lp_a1, Lp_a2, Lp_a3, Lp_b0, Lp_b1, _zf, _zf, _zf, _zf
    )
    L4_c4, L4_c5, L4_c6, L4_c7 = mma_bf16_m16n8k16(
        Lp_a0, Lp_a1, Lp_a2, Lp_a3, Lp_b2, Lp_b3, _zf, _zf, _zf, _zf
    )

    L4_a0 = _pack_bf16x2(L4_c0, L4_c1)
    L4_a1 = _pack_bf16x2(L4_c2, L4_c3)
    L4_a2 = _pack_bf16x2(L4_c4, L4_c5)
    L4_a3 = _pack_bf16x2(L4_c6, L4_c7)

    L4_b0 = _movmatrix_trans(L4_a0)
    L4_b1 = _movmatrix_trans(L4_a1)
    L4_b2 = _movmatrix_trans(L4_a2)
    L4_b3 = _movmatrix_trans(L4_a3)

    mm_c0, mm_c1, mm_c2, mm_c3 = mma_bf16_m16n8k16(
        INV_a0, INV_a1, INV_a2, INV_a3, L4_b0, L4_b1, _zf, _zf, _zf, _zf
    )
    mm_c4, mm_c5, mm_c6, mm_c7 = mma_bf16_m16n8k16(
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

    INV_a0 = _pack_bf16x2(INV_f0, INV_f1)
    INV_a1 = _pack_bf16x2(INV_f2, INV_f3)
    INV_a2 = _pack_bf16x2(INV_f4, INV_f5)
    INV_a3 = _pack_bf16x2(INV_f6, INV_f7)

    # === Iteration 3: L⁸ = L⁴ × L⁴, then INV = INV + INV × L⁸ ===
    L8_c0, L8_c1, L8_c2, L8_c3 = mma_bf16_m16n8k16(
        L4_a0, L4_a1, L4_a2, L4_a3, L4_b0, L4_b1, _zf, _zf, _zf, _zf
    )
    L8_c4, L8_c5, L8_c6, L8_c7 = mma_bf16_m16n8k16(
        L4_a0, L4_a1, L4_a2, L4_a3, L4_b2, L4_b3, _zf, _zf, _zf, _zf
    )

    L8_a0 = _pack_bf16x2(L8_c0, L8_c1)
    L8_a1 = _pack_bf16x2(L8_c2, L8_c3)
    L8_a2 = _pack_bf16x2(L8_c4, L8_c5)
    L8_a3 = _pack_bf16x2(L8_c6, L8_c7)

    L8_b0 = _movmatrix_trans(L8_a0)
    L8_b1 = _movmatrix_trans(L8_a1)
    L8_b2 = _movmatrix_trans(L8_a2)
    L8_b3 = _movmatrix_trans(L8_a3)

    mm_c0, mm_c1, mm_c2, mm_c3 = mma_bf16_m16n8k16(
        INV_a0, INV_a1, INV_a2, INV_a3, L8_b0, L8_b1, _zf, _zf, _zf, _zf
    )
    mm_c4, mm_c5, mm_c6, mm_c7 = mma_bf16_m16n8k16(
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

    # --- Store INV to packed bf16x2 SMEM ---
    sAkk[r_off + gid, c_off + tid] = _pack_bf16x2(INV_f0, INV_f1)
    sAkk[r_off + gid + 8, c_off + tid] = _pack_bf16x2(INV_f2, INV_f3)
    sAkk[r_off + gid, c_off + 4 + tid] = _pack_bf16x2(INV_f4, INV_f5)
    sAkk[r_off + gid + 8, c_off + 4 + tid] = _pack_bf16x2(INV_f6, INV_f7)


# 16×16 matmul: load A & B from packed bf16x2 sAkk, BF16 MMA, return FP32 C
@dsl_user_op
def _matmul_AB(
    sAkk: cute.Tensor, br_A, bc_A, br_B, bc_B, lane_id, *, loc=None, ip=None
):
    gid = lane_id // 4
    tid = lane_id % 4
    _zf = cutlass.Float32(0.0)
    rA = br_A * 16
    cA = bc_A * 8  # packed: 16 cols → 8 pairs
    rB = br_B * 16
    cB = bc_B * 8

    # Load A-operand (packed bf16x2, direct from SMEM)
    a0 = cutlass.Float32(sAkk[rA + gid, cA + tid])
    a1 = cutlass.Float32(sAkk[rA + gid + 8, cA + tid])
    a2 = cutlass.Float32(sAkk[rA + gid, cA + 4 + tid])
    a3 = cutlass.Float32(sAkk[rA + gid + 8, cA + 4 + tid])

    # Load B-operand: load in A-layout then movmatrix → B-layout
    bA0 = cutlass.Float32(sAkk[rB + gid, cB + tid])
    bA1 = cutlass.Float32(sAkk[rB + gid + 8, cB + tid])
    bA2 = cutlass.Float32(sAkk[rB + gid, cB + 4 + tid])
    bA3 = cutlass.Float32(sAkk[rB + gid + 8, cB + 4 + tid])
    b0 = _movmatrix_trans(bA0)
    b1 = _movmatrix_trans(bA1)
    b2 = _movmatrix_trans(bA2)
    b3 = _movmatrix_trans(bA3)

    # 16×16 = 2 × m16n8k16
    cn0_0, cn0_1, cn0_2, cn0_3 = mma_bf16_m16n8k16(
        a0, a1, a2, a3, b0, b1, _zf, _zf, _zf, _zf
    )
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_bf16_m16n8k16(
        a0, a1, a2, a3, b2, b3, _zf, _zf, _zf, _zf
    )

    # Return 8 FP32 C-registers (C-layout of m16n8k16 FP32 accum)
    return cn0_0, cn0_1, cn0_2, cn0_3, cn1_0, cn1_1, cn1_2, cn1_3


# Chain MMA: pre-loaded A (from C→A pack), load B from sAkk  (Stage 2)
# A-operand already packed as bf16x2 from previous C result.
@dsl_user_op
def _chain_mma_B(
    sAkk: cute.Tensor, br_B, bc_B, a0, a1, a2, a3, lane_id, *, loc=None, ip=None
):
    gid = lane_id // 4
    tid = lane_id % 4
    _zf = cutlass.Float32(0.0)
    rB = br_B * 16
    cB = bc_B * 8

    bA0 = cutlass.Float32(sAkk[rB + gid, cB + tid])
    bA1 = cutlass.Float32(sAkk[rB + gid + 8, cB + tid])
    bA2 = cutlass.Float32(sAkk[rB + gid, cB + 4 + tid])
    bA3 = cutlass.Float32(sAkk[rB + gid + 8, cB + 4 + tid])
    b0 = _movmatrix_trans(bA0)
    b1 = _movmatrix_trans(bA1)
    b2 = _movmatrix_trans(bA2)
    b3 = _movmatrix_trans(bA3)

    cn0_0, cn0_1, cn0_2, cn0_3 = mma_bf16_m16n8k16(
        a0, a1, a2, a3, b0, b1, _zf, _zf, _zf, _zf
    )
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_bf16_m16n8k16(
        a0, a1, a2, a3, b2, b3, _zf, _zf, _zf, _zf
    )

    return cn0_0, cn0_1, cn0_2, cn0_3, cn1_0, cn1_1, cn1_2, cn1_3


# Chain MMA (stages 3-4): load A from sAkk, B pre-loaded. B must already be in
# B-layout, i.e. packed bf16x2 after movmatrix -- not the raw FP32 shuffle result.
@dsl_user_op
def _chain_mma_A(
    sAkk: cute.Tensor, br_A, bc_A, b0, b1, b2, b3, lane_id, *, loc=None, ip=None
):
    gid = lane_id // 4
    tid = lane_id % 4
    _zf = cutlass.Float32(0.0)
    rA = br_A * 16
    cA = bc_A * 8

    a0 = cutlass.Float32(sAkk[rA + gid, cA + tid])
    a1 = cutlass.Float32(sAkk[rA + gid + 8, cA + tid])
    a2 = cutlass.Float32(sAkk[rA + gid, cA + 4 + tid])
    a3 = cutlass.Float32(sAkk[rA + gid + 8, cA + 4 + tid])

    cn0_0, cn0_1, cn0_2, cn0_3 = mma_bf16_m16n8k16(
        a0, a1, a2, a3, b0, b1, _zf, _zf, _zf, _zf
    )
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_bf16_m16n8k16(
        a0, a1, a2, a3, b2, b3, _zf, _zf, _zf, _zf
    )

    return cn0_0, cn0_1, cn0_2, cn0_3, cn1_0, cn1_1, cn1_2, cn1_3


# Store negated C result (16×16, FP32 accum) to packed bf16x2 sAkk
# C-layout for m16n8k16 FP32 accum:
#   cn0_0 = C[gid, 0..7 left half col0], cn0_1 = C[gid+8, left half col0]
#   cn0_2 = C[gid, left half col1], cn0_3 = C[gid+8, left half col1]
#   cn1_* = right half
# Packing: negate FP32 then pack pairs → bf16x2
@dsl_user_op
def _store_neg_C(
    sAkk: cute.Tensor,
    br,
    bc,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    lane_id,
    *,
    loc=None,
    ip=None,
):
    gid = lane_id // 4
    tid = lane_id % 4
    r = br * 16
    c = bc * 8  # packed

    # Pack negated FP32 pairs → bf16x2, then store
    sAkk[r + gid, c + tid] = _pack_bf16x2(-c0, -c1)
    sAkk[r + gid + 8, c + tid] = _pack_bf16x2(-c2, -c3)
    sAkk[r + gid, c + 4 + tid] = _pack_bf16x2(-c4, -c5)
    sAkk[r + gid + 8, c + 4 + tid] = _pack_bf16x2(-c6, -c7)


# Pack FP32 C-accum → bf16x2 A-operand for C→A chain
# C-layout (FP32 accum, m16n8k16): 8 floats → 4 bf16x2 A-regs
#   c0,c1 → a0 (rows gid/gid+8, left-half k0..7)
#   c2,c3 → a1
#   c4,c5 → a2 (right-half k8..15)
#   c6,c7 → a3
@dsl_user_op
def _pack_C_to_A(c0, c1, c2, c3, c4, c5, c6, c7, *, loc=None, ip=None):
    a0 = _pack_bf16x2(c0, c1)
    a1 = _pack_bf16x2(c2, c3)
    a2 = _pack_bf16x2(c4, c5)
    a3 = _pack_bf16x2(c6, c7)
    return a0, a1, a2, a3


# Convert FP32 C-accum → bf16x2 B-operand via pack + movmatrix
@dsl_user_op
def _pack_C_to_B(c0, c1, c2, c3, c4, c5, c6, c7, *, loc=None, ip=None):
    a0 = _pack_bf16x2(c0, c1)
    a1 = _pack_bf16x2(c2, c3)
    a2 = _pack_bf16x2(c4, c5)
    a3 = _pack_bf16x2(c6, c7)
    b0 = _movmatrix_trans(a0)
    b1 = _movmatrix_trans(a1)
    b2 = _movmatrix_trans(a2)
    b3 = _movmatrix_trans(a3)
    return b0, b1, b2, b3


# sTemp helpers (FP32, non-swizzled, for inter-warp accumulator exchange)
@dsl_user_op
def _store_C_temp(
    sT: cute.Tensor,
    buf,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    lane_id,
    *,
    loc=None,
    ip=None,
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
def _load_C_temp(sT: cute.Tensor, buf, lane_id, *, loc=None, ip=None):
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


@cute.kernel
def akk_inv_kernel(
    g2s_copy: cute.TiledCopy,
    gA_tensor: cute.Tensor,
    mOut: cute.Tensor,
    mBeta: cute.Tensor,
    akk_smem_layout: cute.Layout,
    temp_layout: cute.Layout,
    NT: int,
    H: int,
    mCuSeqlens: cute.Tensor,
    mChunkIndices: cute.Tensor,
    IS_VARLEN: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_id = tidx % 32
    h_idx, nt_idx, b_idx = cute.arch.block_idx()

    # ===== SMEM allocation =====
    smem = cutlass.utils.SmemAllocator()
    sAkk = smem.allocate_tensor(cutlass.Float32, akk_smem_layout, 128)
    sTemp = smem.allocate_tensor(cutlass.Float32, temp_layout, 128)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout(BS, stride=1), 128)

    # ===== Stage 0: cp.async load bf16 global → packed bf16x2 sAkk =====
    ld_bnt = nt_idx
    if IS_VARLEN:
        ld_seq_id = cutlass.Int32(mChunkIndices[nt_idx, 0])
        ld_local = cutlass.Int32(mChunkIndices[nt_idx, 1])
        ld_bos = cutlass.Int32(mCuSeqlens[ld_seq_id])
        ld_bnt = ld_bos + ld_local * BS
    gA_batch = gA_tensor[(None, None, h_idx, ld_bnt, b_idx)]

    thr_g2s = g2s_copy.get_slice(tidx)
    thr_gSrc = thr_g2s.partition_S(gA_batch)
    thr_sDst = thr_g2s.partition_D(sAkk)
    cute.copy(g2s_copy, thr_gSrc, thr_sDst)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()

    # Zero out-of-bounds rows for varlen partial chunks.
    if IS_VARLEN:
        _z_seq = cutlass.Int32(mChunkIndices[nt_idx, 0])
        _z_local = cutlass.Int32(mChunkIndices[nt_idx, 1])
        _z_bos = cutlass.Int32(mCuSeqlens[_z_seq])
        _z_eos = cutlass.Int32(mCuSeqlens[_z_seq + 1])
        _z_cs = _z_bos + _z_local * BS
        _z_vr = _z_eos - _z_cs
        _zr_start = warp_idx * SB
        for ri in cutlass.range_constexpr(SB):
            row = _zr_start + ri
            if row >= _z_vr:
                # Zero packed bf16x2 slots (each slot = 2 bf16 elements)
                c0 = lane_id
                if c0 < AKK_STRIDE:
                    sAkk[row, c0] = cutlass.Float32(0.0)
        cute.arch.barrier()

    # ===== Stage 0b: Load 64 per-token betas into sBeta (fp32) =====
    _b_chunk_start = nt_idx * BS
    _b_eos = cutlass.Int32(_b_chunk_start + BS)
    if IS_VARLEN:
        _b_seq_id = cutlass.Int32(mChunkIndices[nt_idx, 0])
        _b_local = cutlass.Int32(mChunkIndices[nt_idx, 1])
        _b_bos = cutlass.Int32(mCuSeqlens[_b_seq_id])
        _b_eos = cutlass.Int32(mCuSeqlens[_b_seq_id + 1])
        _b_chunk_start = _b_bos + _b_local * BS

    if warp_idx == 0:
        _bcol_lo = lane_id * 2
        _bcol_hi = _bcol_lo + 1
        if _bcol_lo < BS:
            _bt_lo = _b_chunk_start + _bcol_lo
            _bt_hi = _b_chunk_start + _bcol_hi
            if IS_VARLEN:
                if _bt_lo < _b_eos:
                    sBeta[_bcol_lo] = cutlass.Float32(mBeta[b_idx, _bt_lo, h_idx])
                else:
                    sBeta[_bcol_lo] = cutlass.Float32(0.0)
                if _bt_hi < _b_eos:
                    sBeta[_bcol_hi] = cutlass.Float32(mBeta[b_idx, _bt_hi, h_idx])
                else:
                    sBeta[_bcol_hi] = cutlass.Float32(0.0)
            else:
                sBeta[_bcol_lo] = cutlass.Float32(mBeta[b_idx, _bt_lo, h_idx])
                sBeta[_bcol_hi] = cutlass.Float32(mBeta[b_idx, _bt_hi, h_idx])
    cute.arch.barrier()

    # ===== Stage 1: Diagonal block inversion via Neumann series (BF16 MMA) =====
    _invert_diag_neumann(sAkk, warp_idx, lane_id)

    cute.arch.barrier()

    # ===== Stage 2: First batch — Ai10, Ai21, Ai32 =====
    # C→A chain: pack FP32 C-accum → bf16x2 A-operand via _pack_C_to_A
    if warp_idx == 0:
        t0, t1, t2, t3, t4, t5, t6, t7 = _matmul_AB(sAkk, 1, 1, 0, 1, lane_id)
        a0, a1, a2, a3 = _pack_C_to_A(t0, t1, t2, t3, t4, t5, t6, t7)
        r0, r1, r2, r3, r4, r5, r6, r7 = _chain_mma_B(
            sAkk, 0, 0, a0, a1, a2, a3, lane_id
        )
        _store_neg_C(sAkk, 1, 0, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)

    if warp_idx == 1:
        t0, t1, t2, t3, t4, t5, t6, t7 = _matmul_AB(sAkk, 2, 2, 1, 2, lane_id)
        a0, a1, a2, a3 = _pack_C_to_A(t0, t1, t2, t3, t4, t5, t6, t7)
        r0, r1, r2, r3, r4, r5, r6, r7 = _chain_mma_B(
            sAkk, 1, 1, a0, a1, a2, a3, lane_id
        )
        _store_neg_C(sAkk, 2, 1, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)

    if warp_idx == 2:
        t0, t1, t2, t3, t4, t5, t6, t7 = _matmul_AB(sAkk, 3, 3, 2, 3, lane_id)
        a0, a1, a2, a3 = _pack_C_to_A(t0, t1, t2, t3, t4, t5, t6, t7)
        r0, r1, r2, r3, r4, r5, r6, r7 = _chain_mma_B(
            sAkk, 2, 2, a0, a1, a2, a3, lane_id
        )
        _store_neg_C(sAkk, 3, 2, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)

    cute.arch.barrier()

    # ===== Stage 3: Second batch — Ai20, Ai31 (warp pairs via sTemp) =====
    _z = cutlass.Float32(0.0)
    t0 = _z
    t1 = _z
    t2 = _z
    t3 = _z
    t4 = _z
    t5 = _z
    t6 = _z
    t7 = _z

    # --- Ai20 = -Ai22 @ (Akk20 @ Ai00 + Akk21 @ Ai10) ---
    if warp_idx == 0:
        t0, t1, t2, t3, t4, t5, t6, t7 = _matmul_AB(sAkk, 0, 2, 0, 0, lane_id)

    if warp_idx == 2:
        s0, s1, s2, s3, s4, s5, s6, s7 = _matmul_AB(sAkk, 1, 2, 1, 0, lane_id)
        _store_C_temp(sTemp, 0, s0, s1, s2, s3, s4, s5, s6, s7, lane_id)

    # --- Ai31 = -Ai33 @ (Akk31 @ Ai11 + Akk32 @ Ai21) ---
    if warp_idx == 1:
        t0, t1, t2, t3, t4, t5, t6, t7 = _matmul_AB(sAkk, 1, 3, 1, 1, lane_id)

    if warp_idx == 3:
        s0, s1, s2, s3, s4, s5, s6, s7 = _matmul_AB(sAkk, 2, 3, 2, 1, lane_id)
        _store_C_temp(sTemp, 1, s0, s1, s2, s3, s4, s5, s6, s7, lane_id)

    cute.arch.barrier()

    # Warp 0: accumulate T1+T2, pack→B, multiply by Ai22
    if warp_idx == 0:
        e0, e1, e2, e3, e4, e5, e6, e7 = _load_C_temp(sTemp, 0, lane_id)
        t0 = t0 + e0
        t1 = t1 + e1
        t2 = t2 + e2
        t3 = t3 + e3
        t4 = t4 + e4
        t5 = t5 + e5
        t6 = t6 + e6
        t7 = t7 + e7
        b0, b1, b2, b3 = _pack_C_to_B(t0, t1, t2, t3, t4, t5, t6, t7)
        r0, r1, r2, r3, r4, r5, r6, r7 = _chain_mma_A(
            sAkk, 2, 2, b0, b1, b2, b3, lane_id
        )
        _store_neg_C(sAkk, 2, 0, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)

    # Warp 1: accumulate T1'+T2', pack→B, multiply by Ai33
    if warp_idx == 1:
        e0, e1, e2, e3, e4, e5, e6, e7 = _load_C_temp(sTemp, 1, lane_id)
        t0 = t0 + e0
        t1 = t1 + e1
        t2 = t2 + e2
        t3 = t3 + e3
        t4 = t4 + e4
        t5 = t5 + e5
        t6 = t6 + e6
        t7 = t7 + e7
        b0, b1, b2, b3 = _pack_C_to_B(t0, t1, t2, t3, t4, t5, t6, t7)
        r0, r1, r2, r3, r4, r5, r6, r7 = _chain_mma_A(
            sAkk, 3, 3, b0, b1, b2, b3, lane_id
        )
        _store_neg_C(sAkk, 3, 1, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)

    cute.arch.barrier()

    # ===== Stage 4: Third batch — Ai30 =====
    t0 = _z
    t1 = _z
    t2 = _z
    t3 = _z
    t4 = _z
    t5 = _z
    t6 = _z
    t7 = _z

    if warp_idx == 0:
        t0, t1, t2, t3, t4, t5, t6, t7 = _matmul_AB(sAkk, 0, 3, 0, 0, lane_id)

    if warp_idx == 1:
        s0, s1, s2, s3, s4, s5, s6, s7 = _matmul_AB(sAkk, 1, 3, 1, 0, lane_id)
        _store_C_temp(sTemp, 0, s0, s1, s2, s3, s4, s5, s6, s7, lane_id)

    if warp_idx == 2:
        s0, s1, s2, s3, s4, s5, s6, s7 = _matmul_AB(sAkk, 2, 3, 2, 0, lane_id)
        _store_C_temp(sTemp, 1, s0, s1, s2, s3, s4, s5, s6, s7, lane_id)

    cute.arch.barrier()

    # Warp 0: accumulate all three, pack→B, multiply by Ai33
    if warp_idx == 0:
        e0, e1, e2, e3, e4, e5, e6, e7 = _load_C_temp(sTemp, 0, lane_id)
        t0 = t0 + e0
        t1 = t1 + e1
        t2 = t2 + e2
        t3 = t3 + e3
        t4 = t4 + e4
        t5 = t5 + e5
        t6 = t6 + e6
        t7 = t7 + e7
        e0, e1, e2, e3, e4, e5, e6, e7 = _load_C_temp(sTemp, 1, lane_id)
        t0 = t0 + e0
        t1 = t1 + e1
        t2 = t2 + e2
        t3 = t3 + e3
        t4 = t4 + e4
        t5 = t5 + e5
        t6 = t6 + e6
        t7 = t7 + e7
        b0, b1, b2, b3 = _pack_C_to_B(t0, t1, t2, t3, t4, t5, t6, t7)
        r0, r1, r2, r3, r4, r5, r6, r7 = _chain_mma_A(
            sAkk, 3, 3, b0, b1, b2, b3, lane_id
        )
        _store_neg_C(sAkk, 3, 0, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)

    cute.arch.barrier()

    # ===== Stage 5: Store packed bf16x2 sAkk → global (b32 with direct bit-mask) =====
    vl_chunk_start = nt_idx * BS
    vl_eos = cutlass.Int32(vl_chunk_start + BS)
    if IS_VARLEN:
        vl_seq_id = cutlass.Int32(mChunkIndices[nt_idx, 0])
        vl_local = cutlass.Int32(mChunkIndices[nt_idx, 1])
        vl_bos = cutlass.Int32(mCuSeqlens[vl_seq_id])
        vl_eos = cutlass.Int32(mCuSeqlens[vl_seq_id + 1])
        vl_chunk_start = vl_bos + vl_local * BS

    row_start = warp_idx * SB

    # PDL: hint downstream K4 to pre-launch so its setup work overlaps with our
    # gmem writes below. K4 has griddepcontrol_wait after its setup but before
    # reading any gmem (TMA loads / gk_last_exp), so ordering is correct.
    cute.arch.griddepcontrol_launch_dependents()

    for ri in cutlass.range_constexpr(SB):
        row = row_start + ri
        pair = lane_id
        if pair < BS // 2:
            packed = cutlass.Float32(sAkk[row, pair])
            masked = _mask_packed_ltri(packed, cutlass.Int32(row), cutlass.Int32(pair))

            _beta_lo = cutlass.Float32(sBeta[2 * pair])
            _beta_hi = cutlass.Float32(sBeta[2 * pair + 1])
            _lo_f32 = _unpack_bf16x2_lo(masked) * _beta_lo
            _hi_f32 = _unpack_bf16x2_hi(masked) * _beta_hi
            final = _pack_bf16x2(_lo_f32, _hi_f32)

            t_row = vl_chunk_start + row
            if IS_VARLEN:
                if t_row < vl_eos:
                    mOut[b_idx, t_row, h_idx, pair] = final
            else:
                mOut[b_idx, t_row, h_idx, pair] = final


@cute.jit
def akk_inv_host(
    A_in: cute.Tensor,
    A_out: cute.Tensor,
    Beta_in: cute.Tensor,
    B: cutlass.Constexpr[int],
    NT: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    mCuSeqlens: cute.Tensor,
    mChunkIndices: cute.Tensor,
    IS_VARLEN: cutlass.Constexpr[int],
    T_VAL: cutlass.Constexpr[int],
    stream: cuda_drv.CUstream,
):
    # BF16 input: view as FP32 (packed bf16x2). Each FP32 element = 2 BF16.
    # Original BF16 shape: [B, T, H, BS] with stride per-element in BF16.
    # As FP32 packed bf16x2: shape [BS, BS//2, H, dim3, B]
    # The row stride in BF16 units is H*BS. In FP32 (bf16x2) units = H*BS/2.
    # The col stride in BF16 is 1. In bf16x2 pairs: pair stride = 1 (adjacent pairs in memory).
    _dim3_size = IS_VARLEN * T_VAL + (1 - IS_VARLEN) * (T_VAL // BS)
    _dim3_stride_bf16 = IS_VARLEN * (H * BS) + (1 - IS_VARLEN) * (BS * H * BS)
    # All strides in FP32 units (each FP32 = 2 BF16):
    _row_stride = H * BS // 2  # row stride (BF16 row stride = H*BS, /2 for bf16x2)
    _col_stride = 1  # adjacent bf16x2 pairs
    _h_stride = BS // 2  # head stride (BF16 = BS, /2 for bf16x2)
    _dim3_stride = _dim3_stride_bf16 // 2
    _batch_stride = T_VAL * H * BS // 2

    view_layout = cute.make_layout(
        (BS, BS // 2, H, _dim3_size, B),
        stride=(_row_stride, _col_stride, _h_stride, _dim3_stride, _batch_stride),
    )
    gA_view = cute.make_tensor(A_in.iterator, view_layout)

    # sAkk: packed bf16x2, logical shape (64, 32) with stride 36 for bank-conflict-free access
    akk_smem_2d = cute.make_layout((BS, BS // 2), stride=(AKK_STRIDE, 1))

    # cp.async G→S copy: 128-bit (4×fp32 = 8 bf16) vectorised
    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    # 128 threads x 4 FP32 (128b) each = 512 elements per iteration, so the
    # 64x32 source takes 4 iterations. The copy maps the source shape
    # (BS, BS//2) = (64, 32) onto the SMEM shape (BS, AKK_STRIDE) = (64, 36):
    # only the (64, 32) data region is written, padding cols 32-35 stay zero.
    g2s_copy = cute.make_tiled_copy_tv(
        copy_atom,
        thr_layout=cute.make_layout((16, 8), stride=(8, 1)),
        val_layout=cute.make_layout((1, 4)),
    )

    # sTemp layout (non-swizzled)
    temp_layout = cute.make_layout(
        (SB, TEMP_COLS, NUM_TEMPS), stride=(TEMP_COLS, 1, SB * TEMP_COLS)
    )

    smem_bytes = BS * AKK_STRIDE * 4 + SB * TEMP_COLS * NUM_TEMPS * 4 + BS * 4 + 256

    out_layout = cute.make_layout(
        (B, T_VAL, H, BS // 2), stride=(T_VAL * H * BS // 2, H * BS // 2, BS // 2, 1)
    )
    out_view = cute.make_tensor(A_out.iterator, out_layout)

    akk_inv_kernel(
        g2s_copy,
        gA_view,
        out_view,
        Beta_in,
        akk_smem_2d,
        temp_layout,
        NT,
        H,
        mCuSeqlens,
        mChunkIndices,
        IS_VARLEN,
    ).launch(
        grid=(H, NT, B),
        block=(THREADS, 1, 1),
        smem=smem_bytes,
        stream=stream,
    )
