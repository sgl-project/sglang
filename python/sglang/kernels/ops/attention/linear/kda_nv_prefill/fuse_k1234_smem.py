# Vendored from the NVIDIA KDA_prefill package (benchmark/ Blackwell path)
# for the Kimi-K3 chunked prefill forward. Local deltas: fla.* imports
# re-pointed to sglang's vendored fla subset, flat sibling imports made
# package-relative, RCP_LN2 inlined. INTERNAL COLLABORATION ONLY.
# ruff: noqa  -- vendored kernel library, minimal local deltas
"""Fused K1+K2+K3+K4 single kernel for KDA forward pass.

Architecture:
  640 threads (20 warps = 5 warpgroups), grid = (B*H, 1, 1) for eqlen.
  Per-chunk two-phase execution:
    Phase A (K123): K1/K2/K3 warps — gate activation, intra-attention, inversion
    Phase B (K4):   WG0 + WG1      — 6 MMAs, readout, state update

  K123 writes intermediates to Zone B SMEM + sGkLast. K4 reads from SMEM.
  Only O and S (final outputs) touch GMEM. SU and gk_last are SMEM-only.
  SMEM aliased between phases via flat pool (~225KB, within B200's 228KB).

Warp assignment (5 warpgroups, 128 threads each):
  WG0 (W0-3):   K4 MMA(W0) + TMA(W2) + idle(W1,W3) — idle during K123
  WG1 (W4-7):   state readout+decay (K123) AND W/NV/O readout (K4)
  WG2 (W8-11):  K1 — TMA+gate/cumsum/KG/KS/QS + K3 inversion+store (merged)
  WG3 (W12-15): K2 — intra-attention MMA (first 4 warps)
  WG4 (W16-19): K2 — intra-attention MMA (second 4 warps)

  Per-warpgroup register reallocation via setmaxnreg (CUDA 13.1 cu13 libs):
    WG0: dec(56)   donates 5120 regs (MMA+TMA+idle)
    WG1: inc(104)  claims 1024    (state readout+decay + K4 readout)
    WG2: default 96                (K1+K3 stays at base)
    WG3+4 (K2): inc(112) claims 4096 (heavy intra-attention, 8 warps total)
    Donate 5120 == Claim 5120 (exact balance — INC requires matching DEC).

  WG1 sync (mbarriers, asymmetric): st_ready_mbar(WG1→MMA: sST ready),
            state_decayed_mbar(WG1→MMA: TMEM ready), mma6_done_mbar(MMA→WG1),
            gk_last_ready_mbar(K1→WG1: gkLast ready)
"""

import glob
import os
import subprocess
import sys

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute import KeepCUBIN, KeepPTX
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05 import Field, OperandMajorMode, OperandSource
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

# =============================================================================
# Constants
# =============================================================================
mma_dtype = cutlass.BFloat16
acc_dtype = cutlass.Float32
out_dtype = cutlass.BFloat16

BT = 64  # chunk_size
BC = 16  # sub-chunk size
K_DIM = 128  # head dim
K_PAD = 8
K_STRIDE = K_DIM + K_PAD  # 136

# K4 tile dimensions
M4 = 64  # chunk_size
N4 = 128  # head dim (K or V)
K4_K = 64  # inner dim for MMA1/2/5
K4_K3 = 128  # inner dim for MMA3/4 (state)
M6 = 128  # MMA6 M
N6 = 128  # MMA6 N
K6 = 64  # MMA6 K

# Debug flags (set to 0 for normal operation)
# DEBUG_K4_LEVEL: 0=skip K4, 1=TMA only, 2=TMA+MMA, 3=full K4
DEBUG_K4_LEVEL = 3
# When True, skip MMA5 (AQC@NV) to isolate: O = OI = QS @ S only.
# If O becomes correct, bug is in sAQC. If still wrong, bug is in sQS.
DEBUG_SKIP_MMA5 = False

# Thread/warp counts
THREADS = 640
WARP_SIZE = 32
WG_SIZE = 128
NUM_WARPS = THREADS // WARP_SIZE  # 28

# K123 warp assignment
NUM_K1_WARPS = 4  # Warps 8-11 (WG2)
NUM_MMA_WARPS = 8  # Warps 12-19
NUM_MMA_ACTIVE = 8
NUM_STORE_WARPS = 4  # Warps 20-23
K1_FIRST_WARP = 8
K2_FIRST_WARP = 12
K3_FIRST_WARP = 20

# K4 warp assignment
K4_MMA_WARP = 0
K4_TMA_WARP = 2
K4_READOUT_WG = 1  # warpgroup index (warps 4-7)

# K123 sub-parameters
K1_ROW_GROUPS = 4
K1_COL_GROUPS = 1
ROWS_PER_K1_WARP = BT // K1_ROW_GROUPS  # 16
K1_COLS_PER_WARP = K_DIM // K1_COL_GROUPS  # 128
ROWS_PER_STORE_WARP = BT // NUM_STORE_WARPS  # 16
VEC = K1_COLS_PER_WARP // 32  # 4
K_VEC = K_DIM // VEC  # 32


NUM_SUB_CHUNKS = BT // BC  # 4
NUM_TILES = NUM_SUB_CHUNKS * (NUM_SUB_CHUNKS + 1) // 2  # 10
MMA_K_TILE = 8
NUM_MMA_K_TILES = K_DIM // MMA_K_TILE  # 16
AQK_TILE_COLS = NUM_TILES * BC  # 160
AQK_TILE_PAD = 8
AQK_TILE_STRIDE = AQK_TILE_COLS + AQK_TILE_PAD  # 168
AKK_PAD = 8
AKK_STRIDE = BT + AKK_PAD  # 72

TEMP_PAD = 8
TEMP_COLS = BC + TEMP_PAD  # 24
NUM_TEMPS = 2

_TILE_IQ = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
_TILE_IK = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
RCP_LN2 = LOG2E
SHFL_W4_CLAMP = 0x1C00

# SMEM layout byte offsets (informational — allocator handles actual placement)
# Persistent region (0-112KB): sAB(8) + sAQC(8) + sKG(16) + sQS(16) + sKS(16) + sST(32) + sV_ext(16)
#   sQS/sKS: time-shared with K123 TMA Q/K, overwritten by K4-format after K2
#   sST: persistent — readout WG writes during K123
# Zone A (112-192KB): sW(16) + sNV(16) + sO(16) + scratch(31KB)
#   K123 scratch aliases: sG(16) + sGcum(34) + sAqk(5) + sAkk(18) + sTemp(3)
# Total peak: ~192KB


# =============================================================================
# dsl_user_op helpers (from fuse_kernel123_no_persistent.py)
# =============================================================================


@dsl_user_op
def k1_internal_barrier(*, loc=None, ip=None):
    """Named barrier for K1 warps (0-3, 128 threads). barrier_id=2."""
    llvm.inline_asm(
        T.i32(),
        [],
        "membar.cta; bar.sync 2, 128; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_tf32_m16n8k8(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """TF32 MMA: D = A * B + C, shape m16n8k8"""
    a0_bits = llvm.bitcast(T.i32(), a0.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a1_bits = llvm.bitcast(T.i32(), a1.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a2_bits = llvm.bitcast(T.i32(), a2.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a3_bits = llvm.bitcast(T.i32(), a3.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b0_bits = llvm.bitcast(T.i32(), b0.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b1_bits = llvm.bitcast(T.i32(), b1.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [
            a0_bits,
            a1_bits,
            a2_bits,
            a3_bits,
            b0_bits,
            b1_bits,
            c0.ir_value(loc=loc, ip=ip),
            c1.ir_value(loc=loc, ip=ip),
            c2.ir_value(loc=loc, ip=ip),
            c3.ir_value(loc=loc, ip=ip),
        ],
        """{
            mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
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
def read_clock(*, loc=None, ip=None):
    """Read globaltimer (ns). Returns i64 as two i32 words packed into fp32 pair."""
    result = llvm.inline_asm(
        T.i64(),
        [],
        "mov.u64 $0, %globaltimer;",
        "=l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return result


@dsl_user_op
def fast_rcp(x, *, loc=None, ip=None):
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


@dsl_user_op
def inv_internal_barrier(*, loc=None, ip=None):
    llvm.inline_asm(
        T.i32(),
        [],
        "bar.sync 3, 128; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _invert_diag(sAkk: cute.Tensor, block_rc, lane_id, *, loc=None, ip=None):
    my_row = lane_id % 16
    halfwarp_base = (lane_id // 16) * 16
    r_off = block_rc * 16
    c_off = block_rc * 16
    rInv = cute.make_rmem_tensor(cute.make_layout((16,), stride=(1,)), cutlass.Float32)
    rInv[0] = cutlass.Float32(1.0)
    for x in range(1, 16):
        rInv[x] = cutlass.Float32(0.0)
    for d in range(1, 16):
        col_d = my_row - d
        valid = cutlass.Float32(col_d >= 0)
        a_val = cutlass.Float32(sAkk[r_off + my_row, c_off + col_d]) * valid
        acc = cutlass.Float32(0.0)
        for j in range(1, d):
            a_re = cutlass.Float32(sAkk[r_off + my_row, c_off + my_row - (d - j)])
            inv_shfl = cute.arch.shuffle_sync(rInv[j], halfwarp_base + my_row - d + j)
            acc = acc + a_re * inv_shfl
        rInv[d] = (-a_val - acc) * valid
    rInv[0] = cutlass.Float32(1.0)
    sAkk[r_off + my_row, c_off + my_row] = rInv[0]
    for d in range(1, 16):
        sAkk[r_off + my_row, c_off + (my_row + 16 - d) % 16] = rInv[
            d
        ] * cutlass.Float32(my_row >= d)


@dsl_user_op
def _matmul_AB(
    sAkk: cute.Tensor, br_A, bc_A, br_B, bc_B, lane_id, *, loc=None, ip=None
):
    gid = lane_id // 4
    tid = lane_id % 4
    _z = cutlass.Float32(0.0)
    rA = br_A * 16
    cA = bc_A * 16
    rB = br_B * 16
    cB = bc_B * 16
    a0 = cutlass.Float32(sAkk[rA + gid, cA + 2 * tid])
    a1 = cutlass.Float32(sAkk[rA + gid + 8, cA + 2 * tid])
    a2 = cutlass.Float32(sAkk[rA + gid, cA + 2 * tid + 1])
    a3 = cutlass.Float32(sAkk[rA + gid + 8, cA + 2 * tid + 1])
    b0n0 = cutlass.Float32(sAkk[rB + 2 * tid, cB + gid])
    b1n0 = cutlass.Float32(sAkk[rB + 2 * tid + 1, cB + gid])
    b0n1 = cutlass.Float32(sAkk[rB + 2 * tid, cB + 8 + gid])
    b1n1 = cutlass.Float32(sAkk[rB + 2 * tid + 1, cB + 8 + gid])
    cn0_0, cn0_1, cn0_2, cn0_3 = mma_tf32_m16n8k8(
        a0, a1, a2, a3, b0n0, b1n0, _z, _z, _z, _z
    )
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_tf32_m16n8k8(
        a0, a1, a2, a3, b0n1, b1n1, _z, _z, _z, _z
    )
    a0 = cutlass.Float32(sAkk[rA + gid, cA + 8 + 2 * tid])
    a1 = cutlass.Float32(sAkk[rA + gid + 8, cA + 8 + 2 * tid])
    a2 = cutlass.Float32(sAkk[rA + gid, cA + 8 + 2 * tid + 1])
    a3 = cutlass.Float32(sAkk[rA + gid + 8, cA + 8 + 2 * tid + 1])
    b0n0 = cutlass.Float32(sAkk[rB + 8 + 2 * tid, cB + gid])
    b1n0 = cutlass.Float32(sAkk[rB + 8 + 2 * tid + 1, cB + gid])
    b0n1 = cutlass.Float32(sAkk[rB + 8 + 2 * tid, cB + 8 + gid])
    b1n1 = cutlass.Float32(sAkk[rB + 8 + 2 * tid + 1, cB + 8 + gid])
    cn0_0, cn0_1, cn0_2, cn0_3 = mma_tf32_m16n8k8(
        a0, a1, a2, a3, b0n0, b1n0, cn0_0, cn0_1, cn0_2, cn0_3
    )
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_tf32_m16n8k8(
        a0, a1, a2, a3, b0n1, b1n1, cn1_0, cn1_1, cn1_2, cn1_3
    )
    return cn0_0, cn0_1, cn0_2, cn0_3, cn1_0, cn1_1, cn1_2, cn1_3


@dsl_user_op
def _chain_mma_B(
    sAkk: cute.Tensor,
    br_B,
    bc_B,
    a0k0,
    a1k0,
    a2k0,
    a3k0,
    a0k1,
    a1k1,
    a2k1,
    a3k1,
    lane_id,
    *,
    loc=None,
    ip=None,
):
    gid = lane_id // 4
    tid = lane_id % 4
    _z = cutlass.Float32(0.0)
    rB = br_B * 16
    cB = bc_B * 16
    b0n0 = cutlass.Float32(sAkk[rB + 2 * tid, cB + gid])
    b1n0 = cutlass.Float32(sAkk[rB + 2 * tid + 1, cB + gid])
    b0n1 = cutlass.Float32(sAkk[rB + 2 * tid, cB + 8 + gid])
    b1n1 = cutlass.Float32(sAkk[rB + 2 * tid + 1, cB + 8 + gid])
    cn0_0, cn0_1, cn0_2, cn0_3 = mma_tf32_m16n8k8(
        a0k0, a1k0, a2k0, a3k0, b0n0, b1n0, _z, _z, _z, _z
    )
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_tf32_m16n8k8(
        a0k0, a1k0, a2k0, a3k0, b0n1, b1n1, _z, _z, _z, _z
    )
    b0n0 = cutlass.Float32(sAkk[rB + 8 + 2 * tid, cB + gid])
    b1n0 = cutlass.Float32(sAkk[rB + 8 + 2 * tid + 1, cB + gid])
    b0n1 = cutlass.Float32(sAkk[rB + 8 + 2 * tid, cB + 8 + gid])
    b1n1 = cutlass.Float32(sAkk[rB + 8 + 2 * tid + 1, cB + 8 + gid])
    cn0_0, cn0_1, cn0_2, cn0_3 = mma_tf32_m16n8k8(
        a0k1, a1k1, a2k1, a3k1, b0n0, b1n0, cn0_0, cn0_1, cn0_2, cn0_3
    )
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_tf32_m16n8k8(
        a0k1, a1k1, a2k1, a3k1, b0n1, b1n1, cn1_0, cn1_1, cn1_2, cn1_3
    )
    return cn0_0, cn0_1, cn0_2, cn0_3, cn1_0, cn1_1, cn1_2, cn1_3


@dsl_user_op
def _chain_mma_A(
    sAkk: cute.Tensor,
    br_A,
    bc_A,
    b0_k0n0,
    b1_k0n0,
    b0_k0n1,
    b1_k0n1,
    b0_k1n0,
    b1_k1n0,
    b0_k1n1,
    b1_k1n1,
    lane_id,
    *,
    loc=None,
    ip=None,
):
    gid = lane_id // 4
    tid = lane_id % 4
    _z = cutlass.Float32(0.0)
    rA = br_A * 16
    cA = bc_A * 16
    a0 = cutlass.Float32(sAkk[rA + gid, cA + 2 * tid])
    a1 = cutlass.Float32(sAkk[rA + gid + 8, cA + 2 * tid])
    a2 = cutlass.Float32(sAkk[rA + gid, cA + 2 * tid + 1])
    a3 = cutlass.Float32(sAkk[rA + gid + 8, cA + 2 * tid + 1])
    cn0_0, cn0_1, cn0_2, cn0_3 = mma_tf32_m16n8k8(
        a0, a1, a2, a3, b0_k0n0, b1_k0n0, _z, _z, _z, _z
    )
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_tf32_m16n8k8(
        a0, a1, a2, a3, b0_k0n1, b1_k0n1, _z, _z, _z, _z
    )
    a0 = cutlass.Float32(sAkk[rA + gid, cA + 8 + 2 * tid])
    a1 = cutlass.Float32(sAkk[rA + gid + 8, cA + 8 + 2 * tid])
    a2 = cutlass.Float32(sAkk[rA + gid, cA + 8 + 2 * tid + 1])
    a3 = cutlass.Float32(sAkk[rA + gid + 8, cA + 8 + 2 * tid + 1])
    cn0_0, cn0_1, cn0_2, cn0_3 = mma_tf32_m16n8k8(
        a0, a1, a2, a3, b0_k1n0, b1_k1n0, cn0_0, cn0_1, cn0_2, cn0_3
    )
    cn1_0, cn1_1, cn1_2, cn1_3 = mma_tf32_m16n8k8(
        a0, a1, a2, a3, b0_k1n1, b1_k1n1, cn1_0, cn1_1, cn1_2, cn1_3
    )
    return cn0_0, cn0_1, cn0_2, cn0_3, cn1_0, cn1_1, cn1_2, cn1_3


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
    c = bc * 16
    sAkk[r + gid, c + 2 * tid] = -c0
    sAkk[r + gid, c + 2 * tid + 1] = -c1
    sAkk[r + gid + 8, c + 2 * tid] = -c2
    sAkk[r + gid + 8, c + 2 * tid + 1] = -c3
    sAkk[r + gid, c + 8 + 2 * tid] = -c4
    sAkk[r + gid, c + 8 + 2 * tid + 1] = -c5
    sAkk[r + gid + 8, c + 8 + 2 * tid] = -c6
    sAkk[r + gid + 8, c + 8 + 2 * tid + 1] = -c7


@dsl_user_op
def _shuffle_C_to_B(c0, c1, c2, c3, c4, c5, c6, c7, lane_id, *, loc=None, ip=None):
    gid = lane_id // 4
    tid = lane_id % 4
    src_a = 8 * tid + gid // 2
    src_b = src_a + 4
    f_odd = cutlass.Float32(gid % 2)
    f_even = cutlass.Float32(1) - f_odd
    c0_a = cute.arch.shuffle_sync(c0, src_a)
    c1_a = cute.arch.shuffle_sync(c1, src_a)
    c2_a = cute.arch.shuffle_sync(c2, src_a)
    c3_a = cute.arch.shuffle_sync(c3, src_a)
    c4_a = cute.arch.shuffle_sync(c4, src_a)
    c5_a = cute.arch.shuffle_sync(c5, src_a)
    c6_a = cute.arch.shuffle_sync(c6, src_a)
    c7_a = cute.arch.shuffle_sync(c7, src_a)
    c0_b = cute.arch.shuffle_sync(c0, src_b)
    c1_b = cute.arch.shuffle_sync(c1, src_b)
    c2_b = cute.arch.shuffle_sync(c2, src_b)
    c3_b = cute.arch.shuffle_sync(c3, src_b)
    c4_b = cute.arch.shuffle_sync(c4, src_b)
    c5_b = cute.arch.shuffle_sync(c5, src_b)
    c6_b = cute.arch.shuffle_sync(c6, src_b)
    c7_b = cute.arch.shuffle_sync(c7, src_b)
    return (
        c0_a * f_even + c1_a * f_odd,
        c0_b * f_even + c1_b * f_odd,
        c2_a * f_even + c3_a * f_odd,
        c2_b * f_even + c3_b * f_odd,
        c4_a * f_even + c5_a * f_odd,
        c4_b * f_even + c5_b * f_odd,
        c6_a * f_even + c7_a * f_odd,
        c6_b * f_even + c7_b * f_odd,
    )


@dsl_user_op
def tma_store_fence(*, loc=None, ip=None):
    """Fence: wait for all outstanding TMA S2G stores to finish reading SMEM.
    Must be called after TMA store and before SMEM is reused (e.g., by K123 phase)."""
    llvm.inline_asm(
        T.i32(),
        [],
        "cp.async.bulk.commit_group; cp.async.bulk.wait_group.read 0; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def threadfence_gl(*, loc=None, ip=None):
    """Global memory fence — flush L1 writes to L2 for TMA visibility."""
    llvm.inline_asm(
        T.i32(),
        [],
        "membar.gl; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _store_C_temp(
    sT: cute.Tensor, buf, c0, c1, c2, c3, c4, c5, c6, c7, lane_id, *, loc=None, ip=None
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
    return (
        cutlass.Float32(sT[gid, 2 * tid, buf]),
        cutlass.Float32(sT[gid, 2 * tid + 1, buf]),
        cutlass.Float32(sT[gid + 8, 2 * tid, buf]),
        cutlass.Float32(sT[gid + 8, 2 * tid + 1, buf]),
        cutlass.Float32(sT[gid, 8 + 2 * tid, buf]),
        cutlass.Float32(sT[gid, 8 + 2 * tid + 1, buf]),
        cutlass.Float32(sT[gid + 8, 8 + 2 * tid, buf]),
        cutlass.Float32(sT[gid + 8, 8 + 2 * tid + 1, buf]),
    )


# =============================================================================
# Fused K1234 Kernel
# =============================================================================


@cute.kernel
def fused_k1234_kernel(
    # --- K4 MMA descriptors ---
    tiled_mma_kmn: cute.TiledMma,
    tiled_mma_mn_mn: cute.TiledMma,
    # --- K4 TMA load atoms (V only; S goes TMEM→SMEM via readout WG) ---
    tma_atom_v: cute.CopyAtom,
    mV_nkl: cute.Tensor,
    v_sl: cute.ComposedLayout,
    s_sl: cute.ComposedLayout,  # sST SMEM layout (filled from TMEM, not TMA)
    # --- K4 Zone B swizzle layouts (for SMEM allocation + write views) ---
    ab_sl: cute.ComposedLayout,
    ks_sl: cute.ComposedLayout,
    qs_sl: cute.ComposedLayout,
    aqc_sl: cute.ComposedLayout,
    kg_sl: cute.ComposedLayout,
    # --- K4 TMA store ---
    tma_atom_o_st: cute.CopyAtom,
    mOo: cute.Tensor,
    store_sl: cute.ComposedLayout,
    # --- K4 readout/reinterpret layouts ---
    readout_k_sl: cute.ComposedLayout,
    nv_b_sl: cute.ComposedLayout,
    nv_a_sl: cute.ComposedLayout,
    # --- K123 TMA load atoms + tensors ---
    tma_atom_q_k123: cute.CopyAtom,
    tma_tensor_q_k123: cute.Tensor,
    tma_atom_k_k123: cute.CopyAtom,
    tma_tensor_k_k123: cute.Tensor,
    tma_atom_g_k123: cute.CopyAtom,
    tma_tensor_g_k123: cute.Tensor,
    # --- K123 SMEM layouts (passed from host) ---
    qk_smem_layout: cute.ComposedLayout,
    g_smem_layout,
    g_cumsum_layout,
    # --- K123 tiled copies ---
    tiled_copy_qk_k1,
    tiled_mma_k2,
    tiled_copy_mma_A,
    tiled_copy_mma_B,
    tiled_copy_Gcum_norm,
    tiled_copy_Gcum_gate,
    # --- K123 GMEM tensors (K123 outputs now go to Zone B SMEM, not GMEM) ---
    mA_log: cute.Tensor,
    mBeta: cute.Tensor,
    scale: cutlass.Float32,
    # --- K4 GMEM (state in TMEM; GMEM only for init/final store) ---
    mS_fp32: cute.Tensor,  # [V, K, B*H] fp32 — initial state + final state output
    # --- dt_bias / safe_gate ---
    mDtBias: cute.Tensor,
    lower_bound: cutlass.Float32,
    HAS_BIAS: cutlass.Constexpr[int],
    USE_SAFE_GATE: cutlass.Constexpr[int],
    # --- Clock profiling (optional) ---
    mClocks: cute.Tensor,  # [16, B*H] i64 — per-role clock profiling
    PROFILE_CLOCKS: cutlass.Constexpr[int],
    # --- Dimensions (dynamic — one compilation serves all shapes) ---
    num_chunks: cutlass.Int32,
    num_heads: cutlass.Int32,
    batch_size: cutlass.Int32,
):
    # === Thread indices ===
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    warpgroup_idx = cute.arch.make_warp_uniform(tidx // WG_SIZE)
    warpgroup_tidx = tidx % WG_SIZE
    lane_id = tidx % WARP_SIZE

    # ========================================================================
    # Per-warpgroup register reallocation (Blackwell SM100, CUDA 13.1).
    #
    # CRITICAL: setmaxnreg.inc is effectively BLOCKING on SM100. If CTAPOOL
    # doesn't have the full requested regs, INC spins forever (hang at
    # torch.cuda.synchronize). Total INC claims MUST be <= total DEC donations.
    #
    # Requirements:
    #   (1) nvidia-cutlass-dsl-libs-cu13==4.4.2 installed (start_env.sh)
    #   (2) CUDA_HOME = CUDA 13.1 toolkit (so 13.1 ptxas is used)
    #   (3) DEC target >= 40 (else ptxas rejects with C7507)
    #   (4) sum(DEC donations) >= sum(INC claims)
    #
    # Budget (20 warps, 640 threads, compiler default 96/thread):
    #   WG0 (W0-3,   128t): dec(56)  donates (96-56)*128 = 5120 regs
    #   WG1 (W4-7,   128t): inc(104) claims  (104-96)*128 = 1024 regs
    #   WG2 (W8-11,  128t): default 96, no call
    #   WG3+4 (W12-19, 256t): inc(112) claims (112-96)*256 = 4096 regs
    # Donate 5120 == Claim 5120 (exact balance).
    # ========================================================================
    if warpgroup_idx == 0:
        cute.arch.setmaxregister_decrease(56)
    elif warpgroup_idx == 1:
        cute.arch.setmaxregister_increase(104)
    elif warpgroup_idx >= 3:
        cute.arch.setmaxregister_increase(112)

    bid = cute.arch.block_idx()[0]
    i_b = bid // num_heads
    i_h = bid % num_heads

    # === K4 thr slices (created once, used every K4 phase) ===
    thr_kmn = tiled_mma_kmn.get_slice(0)
    thr_mn = tiled_mma_mn_mn.get_slice(0)
    dice = (None, None, None)

    # ========================================================================
    # SMEM ALLOCATION — Persistent + Aliased Layout (~224KB)
    #   Persistent (0-112KB): K4 operands + sST + sV_ext
    #     sQS/sKS: K4-only, NOT aliased by K123 (separate sQ_k123/sK_k123 in Zone A)
    #     sST: persistent — K1 warps write decayed state bf16 during K123.
    #   Zone A (112-224KB): K123 scratch / K4 readout (aliased)
    #     During K123: sQ, sK, sG, sGcum, sAqk, sAkk, sTemp (~109KB)
    #     During K4: sW, sNV, sO readout buffers (48KB)
    #   sGkLast (512B): separate allocation, persists from K1 to K4
    # State S[V,K] fp32 lives in TMEM offset 256-383 (persistent across chunks).
    # ========================================================================
    smem = cutlass.utils.SmemAllocator()
    AL = 128

    # --- Persistent region (written by K123, read by K4) ---
    sAB = smem.allocate_tensor(mma_dtype, ab_sl.outer, AL, ab_sl.inner)  # 8KB  A_kk_inv
    sAQC = smem.allocate_tensor(mma_dtype, aqc_sl.outer, AL, aqc_sl.inner)  # 8KB  A_qk
    sKG = smem.allocate_tensor(mma_dtype, kg_sl.outer, AL, kg_sl.inner)  # 16KB kg
    sQS = smem.allocate_tensor(
        mma_dtype, qs_sl.outer, AL, qs_sl.inner
    )  # 16KB q_scaled (K4 only)
    sKS = smem.allocate_tensor(
        mma_dtype, ks_sl.outer, AL, ks_sl.inner
    )  # 16KB k_scaled (K4 only)
    sST = smem.allocate_tensor(
        mma_dtype, s_sl.outer, AL, s_sl.inner
    )  # 32KB state bf16 (persistent)
    sV_ext = smem.allocate_tensor(
        mma_dtype, v_sl.outer, AL, v_sl.inner
    )  # 16KB V (non-aliased)
    # Persistent total: 112KB

    # --- Zone A: K4 readout (aliases K123 scratch) ---
    sW = smem.allocate_tensor(
        mma_dtype, readout_k_sl.outer, AL, readout_k_sl.inner
    )  # 16KB readout
    sNV = smem.allocate_tensor(
        mma_dtype, readout_k_sl.outer, AL, readout_k_sl.inner
    )  # 16KB readout
    sO = smem.allocate_tensor(
        mma_dtype, readout_k_sl.outer, AL, readout_k_sl.inner
    )  # 16KB readout
    # K4 readout: 48KB. K123 scratch needs ~111KB, extra ~64KB allocated below.
    _smem_extra = smem.allocate_array(cutlass.Float32, 16384)  # ~64KB for K123 overflow

    # --- K123 alias: sQ_k123/sK_k123 in Zone A (separate from sQS/sKS) ---
    # sQ_k123 aliases sW, sK_k123 aliases sNV during K123.
    # TMA loads Q/K here in K_SW128 swizzle layout. K1 writes KS/QS to persistent sQS/sKS.
    _za = sW.iterator  # Zone A base (bf16 pointer)
    sQ_k123 = cute.make_tensor(
        cute.recast_ptr(_za, qk_smem_layout.inner, cutlass.BFloat16),
        qk_smem_layout.outer,
    )
    sK_k123 = cute.make_tensor(
        cute.recast_ptr(_za + 8192, qk_smem_layout.inner, cutlass.BFloat16),
        qk_smem_layout.outer,
    )

    # --- K123 alias: scratch tensors (overlap Zone A, after sQ/sK) ---
    # sG_k123 [64,128] bf16 at +32KB (+16384 bf16), aliases sO during K4
    sG_k123 = cute.make_tensor(
        cute.recast_ptr(_za + 16384, dtype=cutlass.BFloat16), g_smem_layout
    )
    # sGcum [64,136] fp32 at +48KB (+24576 bf16), 34816B
    sGcum = cute.make_tensor(
        cute.recast_ptr(_za + 24576, dtype=cutlass.Float32), g_cumsum_layout
    )
    # sPartialLast removed — 1-col-per-thread K1 doesn't need cross-warp shuffle
    # sAqk_k123 [16,168] bf16 at +85KB (+43040 bf16), 5376B
    aqk_tile_layout = cute.make_layout(
        (BC, AQK_TILE_STRIDE), stride=(AQK_TILE_STRIDE, 1)
    )
    sAqk_k123 = cute.make_tensor(
        cute.recast_ptr(_za + 43040, dtype=cutlass.BFloat16), aqk_tile_layout
    )
    # sAkk_k123 [64,72] fp32 at +91KB (+45728 bf16), 18432B
    akk_tile_layout = cute.make_layout((BT, AKK_STRIDE), stride=(AKK_STRIDE, 1))
    sAkk_k123 = cute.make_tensor(
        cute.recast_ptr(_za + 45728, dtype=cutlass.Float32), akk_tile_layout
    )
    # sTemp [16,24,2] fp32 at +110KB (+54944 bf16), 3072B
    temp_layout = cute.make_layout(
        (BC, TEMP_COLS, NUM_TEMPS), stride=(TEMP_COLS, 1, BC * TEMP_COLS)
    )
    sTemp = cute.make_tensor(
        cute.recast_ptr(_za + 54944, dtype=cutlass.Float32), temp_layout
    )

    # --- K4 reinterpret views ---
    sNV_b = cute.make_tensor(
        cute.recast_ptr(sNV.iterator, nv_b_sl.inner, mma_dtype), nv_b_sl.outer
    )
    sNV_a = cute.make_tensor(
        cute.recast_ptr(sNV.iterator, nv_a_sl.inner, mma_dtype), nv_a_sl.outer
    )
    sO_st = cute.make_tensor(
        cute.recast_ptr(sO.iterator, store_sl.inner, out_dtype), store_sl.outer
    )

    # sSU_fp32 removed — SU stays in TMEM/RMEM, no SMEM copy needed

    # gk_last: 128 fp32 per (b,h) chunk — persists from K1 to K4 within same chunk
    _sGkLast_buf = smem.allocate_array(cutlass.Float32, K_DIM)  # 128 fp32 = 512 bytes
    sGkLast = cute.make_tensor(_sGkLast_buf, cute.make_layout((K_DIM,), stride=(1,)))

    # beta: 64 fp32 per chunk — loaded once in K1, read by K2/K3/K4
    _sBeta_buf = smem.allocate_array(cutlass.Float32, BT)  # 64 fp32 = 256 bytes
    sBeta = cute.make_tensor(_sBeta_buf, cute.make_layout((BT,), stride=(1,)))

    # ========================================================================
    # TMEM ALLOCATION (K4, warp 0 only)
    # ========================================================================
    tmem_smem = smem.allocate_array(cutlass.Int32, 1)
    if warp_idx == 0:
        cute.arch.alloc_tmem(512, tmem_smem)

    # ========================================================================
    # K123 MBARRIERS (single-stage, no double-buffering)
    # ========================================================================
    k123_tma_mbar = smem.allocate_array(cutlass.Int64, 1)
    k123_k1_done_mbar = smem.allocate_array(cutlass.Int64, 1)
    k123_mma_done_mbar = smem.allocate_array(cutlass.Int64, 1)
    # Asymmetric mbarriers for WG1 async (arrive = non-blocking signal, wait = blocking)
    mma6_done_mbar = smem.allocate_array(cutlass.Int64, 1)  # MMA→WG1: MMA6 complete
    st_ready_mbar = smem.allocate_array(cutlass.Int64, 1)  # WG1→MMA: sST bf16 ready
    gk_last_ready_mbar = smem.allocate_array(cutlass.Int64, 1)  # K1→WG1: gkLast ready
    state_decayed_mbar = smem.allocate_array(
        cutlass.Int64, 1
    )  # WG1→MMA: decayed state ready
    final_state_done_mbar = smem.allocate_array(
        cutlass.Int64, 1
    )  # WG1→MMA: final state stored

    # K4 named barriers (symmetric — both sides arrive and wait)
    sW_ready_nbar = pipeline.NamedBarrier(4, WG_SIZE + WARP_SIZE)
    sNV_ready_nbar = pipeline.NamedBarrier(5, WG_SIZE + WARP_SIZE)
    store_nbar = pipeline.NamedBarrier(6, WG_SIZE + 2 * WARP_SIZE)
    phase_nbar = pipeline.NamedBarrier(7, THREADS)  # all threads (WG1 now participates)

    elect_one = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
    wg_coop = pipeline.CooperativeGroup(pipeline.Agent.Thread, WG_SIZE)

    # K4 TMA pipelines
    def _make_tma_pipe(byte_count):
        ptr = smem.allocate_array(cutlass.Int64, 2)
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=ptr,
            num_stages=1,
            producer_group=elect_one,
            consumer_group=elect_one,
            tx_count=byte_count,
            defer_sync=True,
        ).make_participants()

    # Only V still uses TMA pipeline (KS/QS now in persistent SMEM)
    v_bytes = cute.size_in_bytes(mma_dtype, v_sl)
    v_prod, v_cons = _make_tma_pipe(v_bytes)

    # K4 UMMA pipelines (MMA → readout)
    def _make_umma_pipe():
        ptr = smem.allocate_array(cutlass.Int64, 2)
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=ptr,
            num_stages=1,
            producer_group=elect_one,
            consumer_group=wg_coop,
            defer_sync=True,
        ).make_participants()

    w_prod, w_cons = _make_umma_pipe()
    nv_prod, nv_cons = _make_umma_pipe()
    o_prod, o_cons = _make_umma_pipe()
    # kv_acc_pipeline removed: barrier() between K123→K4 ensures state readout/decay done

    # ========================================================================
    # INIT
    # ========================================================================
    k123_tma_bytes = BT * K_DIM * 2 * 3  # q+k+g bf16

    if tidx == 0:
        cute.arch.mbarrier_init(k123_tma_mbar, 1)
        cute.arch.mbarrier_init(k123_k1_done_mbar, NUM_K1_WARPS * WARP_SIZE)
        cute.arch.mbarrier_init(k123_mma_done_mbar, NUM_MMA_WARPS * WARP_SIZE)
        cute.arch.mbarrier_init(mma6_done_mbar, WARP_SIZE)  # MMA warp (W0, 32t) arrives
        cute.arch.mbarrier_init(st_ready_mbar, WG_SIZE)  # WG1 (128t) arrives
        cute.arch.mbarrier_init(
            gk_last_ready_mbar, NUM_K1_WARPS * WARP_SIZE
        )  # K1 (128t) arrives
        cute.arch.mbarrier_init(state_decayed_mbar, WG_SIZE)  # WG1 (128t) arrives
        cute.arch.mbarrier_init(final_state_done_mbar, WG_SIZE)  # WG1 (128t) arrives
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    # ========================================================================
    # TMEM tensors (K4, created once)
    # ========================================================================
    tmem_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Int32, 16, tmem_smem)

    tCtW_shape = tiled_mma_kmn.partition_shape_C((M4, N4))
    tCtW_fake = tiled_mma_kmn.make_fragment_C(tCtW_shape)
    tCtW = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 0, dtype=acc_dtype), tCtW_fake.layout
    )

    tCtNV_shape = tiled_mma_kmn.partition_shape_C((M4, N4))
    tCtNV_fake = tiled_mma_kmn.make_fragment_C(tCtNV_shape)
    tCtNV = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 128, dtype=acc_dtype), tCtNV_fake.layout
    )
    tCtO = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 384, dtype=acc_dtype), tCtNV_fake.layout
    )

    tCtS_shape = tiled_mma_mn_mn.partition_shape_C((M6, N6))
    tCtS_fake = tiled_mma_mn_mn.make_fragment_C(tCtS_shape)
    tCtS = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 128, dtype=acc_dtype), tCtS_fake.layout
    )

    # State [V=128, K=128] fp32 at TMEM offset 256 — persistent across chunks (GDN pattern)
    # Same MN-MN layout as SU (tCtS) so corresponding elements align for state update
    tCtState = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 256, dtype=acc_dtype), tCtS_fake.layout
    )

    # ========================================================================
    # WARP-SPECIALIZED INDEPENDENT LOOPS (GDN pattern)
    # Each role: per-role setup -> own chunk loop -> cleanup
    # (setmaxnreg deferred to Phase 2)
    # ========================================================================

    # ==============================================================
    # WG1 (warps 4-7): state readout + decay (K123) + W/NV/O readout (K4)
    # Both tasks in one WG — they never overlap (K123 vs K4 phases).
    # ==============================================================
    if warpgroup_idx == K4_READOUT_WG:

        # --- W/NV/O readout setup (M=64: Ld16x256bOp) ---
        tCtW_mn = transform_partitioned_tensor_layout(tCtW)
        tCtNV_mn = transform_partitioned_tensor_layout(tCtNV)
        tCtO_mn = transform_partitioned_tensor_layout(tCtO)

        atom_t2r = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition(1)), acc_dtype
        )
        tiled_t2r = tcgen05.make_tmem_copy(atom_t2r, tCtW[(None, None), 0, 0])
        thr_t2r = tiled_t2r.get_slice(warpgroup_tidx)

        tTR_W = thr_t2r.partition_S(tCtW_mn)
        tTR_NV = thr_t2r.partition_S(tCtNV_mn)
        tTR_O = thr_t2r.partition_S(tCtO_mn)

        atom_r2s_k = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.ROW_MAJOR, mma_dtype, acc_dtype, tiled_t2r
        )
        tiled_r2s_k = cute.make_tiled_copy_D(atom_r2s_k, tiled_t2r)
        thr_r2s_k = tiled_r2s_k.get_slice(warpgroup_tidx)
        tCsW = thr_r2s_k.partition_D(transform_partitioned_tensor_layout(sW))
        tCsO = thr_r2s_k.partition_D(transform_partitioned_tensor_layout(sO))
        tCsNV = thr_r2s_k.partition_D(transform_partitioned_tensor_layout(sNV))

        cId = cute.make_identity_tensor((M4, N4))
        tTR_cId = thr_t2r.partition_D(cId)

        # --- State TMEM setup (M=128, GDN pattern) ---
        cId_128 = cute.make_identity_tensor((M6, N6))

        # --- State TMEM copy atoms (M=128, GDN pattern) ---
        tCtState_mn = transform_partitioned_tensor_layout(tCtState)

        # TMEM->RMEM (Ld32x32bOp for M=128)
        atom_state_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), acc_dtype
        )
        tiled_state_t2r = tcgen05.make_tmem_copy(
            atom_state_t2r, tCtState[(None, None), 0, 0]
        )
        thr_state_t2r = tiled_state_t2r.get_slice(warpgroup_tidx)
        tTR_tCtState = thr_state_t2r.partition_S(tCtState_mn)
        tTR_tCcState = thr_state_t2r.partition_D(cId_128)
        tRrState = cute.make_rmem_tensor_like(tTR_tCcState, acc_dtype)  # fp32 RMEM

        # RMEM->TMEM (St32x32bOp for M=128, state write-back)
        atom_state_r2t = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(32)), acc_dtype
        )
        tiled_state_r2t = tcgen05.make_tmem_copy(
            atom_state_r2t, tCtState[(None, None), 0, 0]
        )
        thr_state_r2t = tiled_state_r2t.get_slice(warpgroup_tidx)
        tRT_tCtState = thr_state_r2t.partition_D(tCtState_mn)

        # RMEM bf16 -> swizzled sST SMEM (CopyUniversalOp, NO domain transpose)
        atom_state_r2s = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mma_dtype, num_bits_per_copy=16
        )
        tiled_state_r2s = cute.make_tiled_copy_D(atom_state_r2s, tiled_state_t2r)
        thr_state_r2s = tiled_state_r2s.get_slice(warpgroup_tidx)
        sST_mn_view = transform_partitioned_tensor_layout(sST)
        tCsState_inp = thr_state_r2s.partition_D(sST_mn_view)
        tRrState_bf16 = cute.make_rmem_tensor_like(tTR_tCcState, mma_dtype)  # bf16 RMEM
        tCrState_bf16 = tiled_state_r2s.retile(tRrState_bf16)

        # GMEM fp32 -> RMEM fp32 (for initial/final state, runs once)
        atom_state_g2r = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), acc_dtype, num_bits_per_copy=32
        )
        tiled_state_g2r = cute.make_tiled_copy_S(atom_state_g2r, tiled_state_r2t)
        thr_state_g2r = tiled_state_g2r.get_slice(warpgroup_tidx)

        # RMEM fp32 -> GMEM fp32 (for final state store, runs once)
        atom_state_r2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), acc_dtype, num_bits_per_copy=32
        )
        tiled_state_r2g = cute.make_tiled_copy_D(atom_state_r2g, tiled_state_t2r)
        thr_state_r2g = tiled_state_r2g.get_slice(warpgroup_tidx)

        # --- Initial state load: GMEM fp32 -> RMEM -> state TMEM ---
        gS_init = mS_fp32[(None, None, bid)]  # [V=128, K=128] fp32
        tGR_tCgState_in = thr_state_g2r.partition_S(gS_init)
        tGR_tCrState_in = thr_state_g2r.retile(tRrState)
        # 1. GMEM fp32 -> RMEM fp32 (coalesced CopyUniversalOp)
        cute.copy(tiled_state_g2r, tGR_tCgState_in, tGR_tCrState_in)
        # 2. RMEM fp32 -> state TMEM (St32x32bOp per sub-tile)
        num_state_subs = tRrState.shape[2]
        for sub in cutlass.range(num_state_subs):
            cute.copy(
                tiled_state_r2t, tRrState[None, 0, sub], tRT_tCtState[None, 0, sub]
            )
        cute.arch.fence_view_async_tmem_store()

        # --- WG1 chunk loop ---
        clk_wg1_start = cutlass.Int64(0)
        if PROFILE_CLOCKS:
            clk_wg1_start = read_clock()
        for chunk_c in cutlass.range(num_chunks):
            num_state_subs = tRrState.shape[2]
            _sub_tile_size = cute.size(tRrState.shape[0])

            # ---- Part 1: State readout + decay ----
            # Wait previous MMA6 (chunk > 0). Chunk 0: state from init.
            if chunk_c > 0:
                cute.arch.mbarrier_wait(mma6_done_mbar, phase=(chunk_c - 1) % 2)

            # Step A: State TMEM -> RMEM(fp32) -> bf16 -> sST SMEM
            # Per-sub streaming: only 1 sub-tile in RMEM at a time (32 regs, not 128).
            # tRrState[sub] is overwritten each iteration — NOT held across subs.
            for sub in cutlass.range(num_state_subs):
                cute.copy(
                    tiled_state_t2r, tTR_tCtState[None, 0, sub], tRrState[None, 0, sub]
                )
                tRrState_bf16[None, 0, sub].store(
                    tRrState[None, 0, sub].load().to(mma_dtype)
                )
                cute.copy(
                    tiled_state_r2s,
                    tCrState_bf16[None, 0, sub],
                    tCsState_inp[None, 0, sub, 0],
                )
            cute.arch.fence_view_async_shared()
            # Signal MMA: sST bf16 ready for MMA4/MMA3 (non-blocking)
            cute.arch.mbarrier_arrive(st_ready_mbar)

            # Step B: Wait K1 to finish gk_last (current chunk)
            cute.arch.mbarrier_wait(gk_last_ready_mbar, phase=chunk_c % 2)

            # Step C+D: Re-read state from TMEM, decay, write back.
            # tRrState was overwritten per-sub in Step A, so re-read is needed.
            # Cost: 4 extra TMEM reads (~40 cycles total) vs saving 96 spilled regs.
            for sub in cutlass.range(num_state_subs):
                # Re-read this sub-tile from TMEM (only 32 regs live)
                cute.copy(
                    tiled_state_t2r, tTR_tCtState[None, 0, sub], tRrState[None, 0, sub]
                )
                # Decay by gk_last
                for i in cutlass.range(_sub_tile_size):
                    coord = tTR_tCcState[i, 0, sub]
                    k_idx = coord[1]  # N dimension = K in (V,K) convention
                    gk_val = cutlass.Float32(sGkLast[k_idx])
                    tRrState[i, 0, sub] = tRrState[i, 0, sub] * gk_val
                # Write decayed sub-tile back to TMEM immediately
                cute.copy(
                    tiled_state_r2t, tRrState[None, 0, sub], tRT_tCtState[None, 0, sub]
                )
            cute.arch.fence_view_async_tmem_store()
            # Signal MMA: decayed state in TMEM ready for MMA6 (non-blocking)
            cute.arch.mbarrier_arrive(state_decayed_mbar)

            # ---- K4 phase: W/NV/O readout ----
            phase_nbar.arrive_and_wait()  # K123->K4

            tRrR = cute.make_rmem_tensor_like(tTR_cId, acc_dtype)
            tRrR_out = cute.make_rmem_tensor_like(tRrR, mma_dtype)
            tCrR_k = tiled_r2s_k.retile(tRrR_out)
            num_subs = tRrR.shape[2]

            # W readout
            wh = w_cons.wait_and_advance()
            for sub in cutlass.range(num_subs):
                cute.copy(tiled_t2r, tTR_W[None, 0, sub], tRrR[None, 0, sub])
                tRrR_out[None, 0, sub].store(tRrR[None, 0, sub].load().to(mma_dtype))
                cute.copy(tiled_r2s_k, tCrR_k[None, 0, sub], tCsW[None, 0, sub, 0])
            cute.arch.fence_view_async_tmem_load()
            wh.release()
            cute.arch.fence_view_async_shared()
            sW_ready_nbar.arrive_and_wait()

            # NV readout
            nvh = nv_cons.wait_and_advance()
            for sub in cutlass.range(num_subs):
                cute.copy(tiled_t2r, tTR_NV[None, 0, sub], tRrR[None, 0, sub])
                tRrR_out[None, 0, sub].store(tRrR[None, 0, sub].load().to(mma_dtype))
                cute.copy(tiled_r2s_k, tCrR_k[None, 0, sub], tCsNV[None, 0, sub, 0])
            cute.arch.fence_view_async_tmem_load()
            nvh.release()
            cute.arch.fence_view_async_shared()
            sNV_ready_nbar.arrive_and_wait()

            # O readout
            oh = o_cons.wait_and_advance()
            for sub in cutlass.range(num_subs):
                cute.copy(tiled_t2r, tTR_O[None, 0, sub], tRrR[None, 0, sub])
                tRrR_out[None, 0, sub].store(tRrR[None, 0, sub].load().to(mma_dtype))
                cute.copy(tiled_r2s_k, tCrR_k[None, 0, sub], tCsO[None, 0, sub, 0])
            cute.arch.fence_view_async_tmem_load()
            oh.release()
            cute.arch.fence_view_async_shared()
            store_nbar.arrive_and_wait()

            phase_nbar.arrive_and_wait()  # K4->next

        # --- Wait last MMA6, final state store ---
        cute.arch.mbarrier_wait(mma6_done_mbar, phase=(num_chunks - 1) % 2)

        # TMEM -> RMEM fp32
        gS_out = mS_fp32[(None, None, bid)]  # [V=128, K=128] fp32
        tGR_tCgState_out = thr_state_r2g.partition_D(gS_out)
        tGR_tCrState_out = thr_state_r2g.retile(tRrState)
        num_state_subs_final = tRrState.shape[2]
        for sub in cutlass.range(num_state_subs_final):
            cute.copy(
                tiled_state_t2r, tTR_tCtState[None, 0, sub], tRrState[None, 0, sub]
            )
        cute.arch.fence_view_async_tmem_load()
        # RMEM fp32 -> GMEM fp32 (coalesced CopyUniversalOp)
        for sub in cutlass.range(num_state_subs_final):
            cute.copy(
                tiled_state_r2g,
                tGR_tCrState_out[None, 0, sub],
                tGR_tCgState_out[None, 0, sub],
            )

        # Signal MMA warp: safe to dealloc TMEM
        cute.arch.mbarrier_arrive(final_state_done_mbar)

        if PROFILE_CLOCKS:
            clk_wg1_end = read_clock()
            if warpgroup_tidx == 0:
                mClocks[8, bid] = clk_wg1_end - clk_wg1_start

    # ==============================================================
    # K1 (warps 8-11): TMA + gate activation + cumsum + KG/KS/QS
    # ==============================================================
    elif warp_idx >= K1_FIRST_WARP and warp_idx < K1_FIRST_WARP + NUM_K1_WARPS:

        # K1 per-warp setup — each thread handles 1 column, all 64 rows sequentially
        clk_k1_start = cutlass.Int64(0)
        if PROFILE_CLOCKS:
            clk_k1_start = read_clock()
        k1_warp = warp_idx - K1_FIRST_WARP
        my_col = k1_warp * WARP_SIZE + lane_id  # 0-127, one column per thread

        # Per-head constants (invariant across chunks)
        exp_A = cute.exp(mA_log[i_h], fastmath=True)
        cumsum_scale = cutlass.Float32(RCP_LN2)
        rBias_val = cutlass.Float32(0.0)
        if HAS_BIAS:
            rBias_val = mDtBias[i_h, my_col].to(cutlass.Float32)

        # Zero sAQC once (upper triangle stays zero across all chunks)
        _aqc_zero = cutlass.BFloat16(0.0)
        for ri in cutlass.range(BC):
            aqc_row = k1_warp * BC + ri
            c0 = lane_id * 2
            c1 = lane_id * 2 + 1
            sAQC[aqc_row + (c0 % 16) * 64, 0, c0 // 16, 0] = _aqc_zero
            sAQC[aqc_row + (c1 % 16) * 64, 0, c1 // 16, 0] = _aqc_zero
        inv_internal_barrier()

        for chunk_c in cutlass.range(num_chunks):
            c_phase = chunk_c % 2
            chunk_start = i_b * num_chunks * BT + chunk_c * BT
            chunk_start_local = chunk_c * BT

            # TMA load q, k, g for this chunk
            if warp_idx == K1_FIRST_WARP:
                gQ_head = tma_tensor_q_k123[(None, None, i_h)]
                gK_head = tma_tensor_k_k123[(None, None, i_h)]
                gG_head = tma_tensor_g_k123[(None, None, i_h)]

                if lane_id == 0:
                    cute.arch.mbarrier_expect_tx(k123_tma_mbar, k123_tma_bytes)

                sQ_s = sQ_k123
                gQ_s = cute.local_tile(
                    cute.domain_offset((chunk_start, 0), gQ_head), (BT, K_DIM), (0, 0)
                )
                ts_q, tg_q = cpasync.tma_partition(
                    tma_atom_q_k123,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sQ_s, 0, 2),
                    cute.group_modes(gQ_s, 0, 2),
                )
                cute.copy(tma_atom_q_k123, tg_q, ts_q, tma_bar_ptr=k123_tma_mbar)

                sK_s = sK_k123
                gK_s = cute.local_tile(
                    cute.domain_offset((chunk_start, 0), gK_head), (BT, K_DIM), (0, 0)
                )
                ts_k, tg_k = cpasync.tma_partition(
                    tma_atom_k_k123,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK_s, 0, 2),
                    cute.group_modes(gK_s, 0, 2),
                )
                cute.copy(tma_atom_k_k123, tg_k, ts_k, tma_bar_ptr=k123_tma_mbar)

                sG_s = sG_k123
                gG_s = cute.local_tile(
                    cute.domain_offset((chunk_start, 0), gG_head), (BT, K_DIM), (0, 0)
                )
                ts_g, tg_g = cpasync.tma_partition(
                    tma_atom_g_k123,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sG_s, 0, 2),
                    cute.group_modes(gG_s, 0, 2),
                )
                cute.copy(tma_atom_g_k123, tg_g, ts_g, tma_bar_ptr=k123_tma_mbar)

                if lane_id == 0:
                    cute.arch.mbarrier_arrive(k123_tma_mbar)

            # Wait TMA
            cute.arch.mbarrier_wait(k123_tma_mbar, phase=c_phase)

            # Load beta[chunk] into SMEM (first K1 warp, 32 threads load 64 bf16 in 2 iters)
            if warp_idx == K1_FIRST_WARP:
                for bi in cutlass.range_constexpr(2):
                    beta_t = bi * WARP_SIZE + lane_id  # 0..63
                    sBeta[beta_t] = mBeta[i_b, chunk_start_local + beta_t, i_h]

            # ============================================================
            # Fused gate activation + cumsum + KG/KS/QS scaling
            # Each thread processes 1 column across all 64 rows sequentially.
            # No rGact, no sPartialLast, no cross-warp shuffle, no barriers.
            # ============================================================

            # Fused Pass: gate activation → cumsum → sGcum (single pass, 64 rows)
            # 16 dynamic iterations × 4 constexpr unrolled = 64 rows
            running_sum = cutlass.Float32(0.0)
            for row_base in cutlass.range(BT // 4):
                for ri in cutlass.range_constexpr(4):
                    row = row_base * 4 + ri
                    g_val = sG_k123[row, my_col].to(cutlass.Float32)
                    if HAS_BIAS:
                        g_val = g_val + rBias_val
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
                    running_sum = running_sum + g_activated
                    sGcum[row, my_col] = running_sum * cumsum_scale

            # Signal K2: sGcum ready (all 64 rows written)
            cute.arch.mbarrier_arrive(k123_k1_done_mbar)

            # KG/KS/QS scaling (reads sGcum + sK + sQ, writes Zone B)
            # 16 dynamic × 4 constexpr = 64 rows
            gk_last_cs = running_sum * cumsum_scale  # total cumsum for this column
            for row_base in cutlass.range(BT // 4):
                for ri in cutlass.range_constexpr(4):
                    row = row_base * 4 + ri
                    cs = sGcum[row, my_col]
                    k_val = sK_k123[row, my_col].to(cutlass.Float32)
                    q_val = sQ_k123[row, my_col].to(cutlass.Float32)
                    exp2_cs = cute.exp2(cs, fastmath=True)
                    exp2_kg = cute.exp2(gk_last_cs - cs, fastmath=True)
                    # Write directly to Zone B swizzled layout
                    sKG[my_col + (row % 16) * 128, 0, row // 16, 0] = (
                        k_val * exp2_kg
                    ).to(cutlass.BFloat16)
                    sKS[my_col + (row % 16) * 128, 0, row // 16, 0] = (
                        cutlass.Float32(-1.0) * k_val * exp2_cs
                    ).to(cutlass.BFloat16)
                    sQS[row + (my_col % 16) * 64, 0, my_col // 16, 0] = (
                        q_val * exp2_cs * scale
                    ).to(cutlass.BFloat16)

            # gkLast: all threads write their own column
            sGkLast[my_col] = cute.exp2(gk_last_cs, fastmath=True)

            # Signal readout WG: gk_last is in SMEM (K1 arrive, non-blocking)
            cute.arch.mbarrier_arrive(gk_last_ready_mbar)

            # ============================================================
            # K3 phase: wait K2, then inversion + store to Zone B
            # (merged into K1 warps — K1 is idle here, K3 reuses same 4 warps)
            # ============================================================
            store_warp = k1_warp  # k1_warp 0-3 maps directly to store_warp 0-3

            cute.arch.mbarrier_wait(k123_mma_done_mbar, phase=c_phase)

            # Write lower-triangle tiles with actual A_qk values
            # (sAQC zeroed once before chunk loop — upper triangle stays zero)
            for tile_idx in cutlass.range_constexpr(NUM_TILES):
                i_q = _TILE_IQ[tile_idx]
                i_k = _TILE_IK[tile_idx]
                is_diag = _TILE_IQ[tile_idx] == _TILE_IK[tile_idx]
                aqk_col_base = tile_idx * BC

                for ri in cutlass.range(BC // NUM_STORE_WARPS):
                    local_row = store_warp * (BC // NUM_STORE_WARPS) + ri
                    if lane_id < BC:
                        local_col = lane_id
                        aqk_val = sAqk_k123[local_row, aqk_col_base + local_col]
                        if is_diag and local_row < local_col:
                            aqk_val = cutlass.BFloat16(0.0)
                        aqc_m = i_q * BC + local_row
                        aqc_k = i_k * BC + local_col
                        sAQC[aqc_m + (aqc_k % 16) * 64, 0, aqc_k // 16, 0] = aqk_val

            # Akk inversion: 4 stages
            if store_warp == 0:
                _invert_diag(sAkk_k123, lane_id // 16, lane_id)
            if store_warp == 1:
                _invert_diag(sAkk_k123, 2 + lane_id // 16, lane_id)
            inv_internal_barrier()

            # Stage 2: Ai10, Ai21, Ai32
            if store_warp == 0:
                t0, t1, t2, t3, t4, t5, t6, t7 = _matmul_AB(
                    sAkk_k123, 1, 1, 0, 1, lane_id
                )
                r0, r1, r2, r3, r4, r5, r6, r7 = _chain_mma_B(
                    sAkk_k123, 0, 0, t0, t2, t1, t3, t4, t6, t5, t7, lane_id
                )
                _store_neg_C(sAkk_k123, 1, 0, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)
            if store_warp == 1:
                t0, t1, t2, t3, t4, t5, t6, t7 = _matmul_AB(
                    sAkk_k123, 2, 2, 1, 2, lane_id
                )
                r0, r1, r2, r3, r4, r5, r6, r7 = _chain_mma_B(
                    sAkk_k123, 1, 1, t0, t2, t1, t3, t4, t6, t5, t7, lane_id
                )
                _store_neg_C(sAkk_k123, 2, 1, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)
            if store_warp == 2:
                t0, t1, t2, t3, t4, t5, t6, t7 = _matmul_AB(
                    sAkk_k123, 3, 3, 2, 3, lane_id
                )
                r0, r1, r2, r3, r4, r5, r6, r7 = _chain_mma_B(
                    sAkk_k123, 2, 2, t0, t2, t1, t3, t4, t6, t5, t7, lane_id
                )
                _store_neg_C(sAkk_k123, 3, 2, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)
            inv_internal_barrier()

            # Stage 3: Ai20, Ai31
            _zz = cutlass.Float32(0.0)
            t0 = _zz
            t1 = _zz
            t2 = _zz
            t3 = _zz
            t4 = _zz
            t5 = _zz
            t6 = _zz
            t7 = _zz
            if store_warp == 0:
                t0, t1, t2, t3, t4, t5, t6, t7 = _matmul_AB(
                    sAkk_k123, 0, 2, 0, 0, lane_id
                )
            if store_warp == 2:
                s0, s1, s2, s3, s4, s5, s6, s7 = _matmul_AB(
                    sAkk_k123, 1, 2, 1, 0, lane_id
                )
                _store_C_temp(sTemp, 0, s0, s1, s2, s3, s4, s5, s6, s7, lane_id)
            if store_warp == 1:
                t0, t1, t2, t3, t4, t5, t6, t7 = _matmul_AB(
                    sAkk_k123, 1, 3, 1, 1, lane_id
                )
            if store_warp == 3:
                s0, s1, s2, s3, s4, s5, s6, s7 = _matmul_AB(
                    sAkk_k123, 2, 3, 2, 1, lane_id
                )
                _store_C_temp(sTemp, 1, s0, s1, s2, s3, s4, s5, s6, s7, lane_id)
            inv_internal_barrier()
            if store_warp == 0:
                e0, e1, e2, e3, e4, e5, e6, e7 = _load_C_temp(sTemp, 0, lane_id)
                t0 = t0 + e0
                t1 = t1 + e1
                t2 = t2 + e2
                t3 = t3 + e3
                t4 = t4 + e4
                t5 = t5 + e5
                t6 = t6 + e6
                t7 = t7 + e7
                sb = _shuffle_C_to_B(t0, t1, t2, t3, t4, t5, t6, t7, lane_id)
                r0, r1, r2, r3, r4, r5, r6, r7 = _chain_mma_A(
                    sAkk_k123,
                    2,
                    2,
                    sb[0],
                    sb[1],
                    sb[4],
                    sb[5],
                    sb[2],
                    sb[3],
                    sb[6],
                    sb[7],
                    lane_id,
                )
                _store_neg_C(sAkk_k123, 2, 0, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)
            if store_warp == 1:
                e0, e1, e2, e3, e4, e5, e6, e7 = _load_C_temp(sTemp, 1, lane_id)
                t0 = t0 + e0
                t1 = t1 + e1
                t2 = t2 + e2
                t3 = t3 + e3
                t4 = t4 + e4
                t5 = t5 + e5
                t6 = t6 + e6
                t7 = t7 + e7
                sb = _shuffle_C_to_B(t0, t1, t2, t3, t4, t5, t6, t7, lane_id)
                r0, r1, r2, r3, r4, r5, r6, r7 = _chain_mma_A(
                    sAkk_k123,
                    3,
                    3,
                    sb[0],
                    sb[1],
                    sb[4],
                    sb[5],
                    sb[2],
                    sb[3],
                    sb[6],
                    sb[7],
                    lane_id,
                )
                _store_neg_C(sAkk_k123, 3, 1, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)
            inv_internal_barrier()

            # Stage 4: Ai30
            t0 = _zz
            t1 = _zz
            t2 = _zz
            t3 = _zz
            t4 = _zz
            t5 = _zz
            t6 = _zz
            t7 = _zz
            if store_warp == 0:
                t0, t1, t2, t3, t4, t5, t6, t7 = _matmul_AB(
                    sAkk_k123, 0, 3, 0, 0, lane_id
                )
            if store_warp == 1:
                s0, s1, s2, s3, s4, s5, s6, s7 = _matmul_AB(
                    sAkk_k123, 1, 3, 1, 0, lane_id
                )
                _store_C_temp(sTemp, 0, s0, s1, s2, s3, s4, s5, s6, s7, lane_id)
            if store_warp == 2:
                s0, s1, s2, s3, s4, s5, s6, s7 = _matmul_AB(
                    sAkk_k123, 2, 3, 2, 0, lane_id
                )
                _store_C_temp(sTemp, 1, s0, s1, s2, s3, s4, s5, s6, s7, lane_id)
            inv_internal_barrier()
            if store_warp == 0:
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
                sb = _shuffle_C_to_B(t0, t1, t2, t3, t4, t5, t6, t7, lane_id)
                r0, r1, r2, r3, r4, r5, r6, r7 = _chain_mma_A(
                    sAkk_k123,
                    3,
                    3,
                    sb[0],
                    sb[1],
                    sb[4],
                    sb[5],
                    sb[2],
                    sb[3],
                    sb[6],
                    sb[7],
                    lane_id,
                )
                _store_neg_C(sAkk_k123, 3, 0, r0, r1, r2, r3, r4, r5, r6, r7, lane_id)
            inv_internal_barrier()

            # Write inverted Akk * beta to K4's swizzled SMEM (sAB in Zone B)
            inv_row_start = store_warp * 16
            for ri in cutlass.range(BC):
                inv_row = inv_row_start + ri
                c0 = lane_id * 2
                c1 = lane_id * 2 + 1
                beta_c0 = sBeta[c0].to(cutlass.Float32)
                beta_c1 = sBeta[c1].to(cutlass.Float32)
                v0 = (
                    cutlass.Float32(sAkk_k123[inv_row, c0])
                    * cutlass.Float32(inv_row >= c0)
                    * beta_c0
                )
                v1 = (
                    cutlass.Float32(sAkk_k123[inv_row, c1])
                    * cutlass.Float32(inv_row >= c1)
                    * beta_c1
                )
                sAB[inv_row + (c0 % 16) * 64, 0, c0 // 16, 0] = v0.to(cutlass.BFloat16)
                sAB[inv_row + (c1 % 16) * 64, 0, c1 // 16, 0] = v1.to(cutlass.BFloat16)

            # K1+K3 done, idle during K4 phase
            phase_nbar.arrive_and_wait()  # K123->K4
            phase_nbar.arrive_and_wait()  # K4->next

        if PROFILE_CLOCKS:
            clk_k1_end = read_clock()
            if k1_warp == 0 and lane_id == 0:
                mClocks[9, bid] = clk_k1_end - clk_k1_start

    # ==============================================================
    # K2 (warps 12-19): intra sub-chunk attention MMA
    # 8 warps: warps 0-5 do 1 tile each, warps 6-7 do 2 tiles each (row 3)
    # ==============================================================
    elif warp_idx >= K2_FIRST_WARP and warp_idx < K2_FIRST_WARP + NUM_MMA_WARPS:

        # K2 warp decode
        clk_k2_start = cutlass.Int64(0)
        if PROFILE_CLOCKS:
            clk_k2_start = read_clock()
        mma_warp = warp_idx - K2_FIRST_WARP
        # Tile A assignment (all 8 warps)
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
        else:
            # Warps 6-7: first tile of row 3 pair
            # W6: (3,0), W7: (3,2)
            my_i_q = cutlass.Int32(3)
            my_i_k = (mma_warp - 6) * 2
        # Tile B assignment (warps 6-7 only): i_k_b = i_k_a + 1
        # W6: (3,1), W7: (3,3)
        my_i_k_b = my_i_k + cutlass.Int32(1)

        for chunk_c in cutlass.range(num_chunks):
            c_phase = chunk_c % 2

            cute.arch.mbarrier_wait(k123_k1_done_mbar, phase=c_phase)

            # --- Tile A (all 8 warps) ---
            q_row_base = my_i_q * BC
            k_row_base = my_i_k * BC
            # tile_col_base maps to sAqk slot index:
            # warps 0-5: 1:1 mapping (tile slot = mma_warp)
            # warps 6-7: tile slot = 6+(mma_warp-6)*2 = 6 or 8
            tile_col_base = mma_warp * BC
            if mma_warp >= 6:
                tile_col_base = (6 + (mma_warp - 6) * 2) * BC
            akk_row_base = k_row_base
            akk_col_base = q_row_base
            norm_row = q_row_base
            if my_i_q == my_i_k:
                norm_row = q_row_base + cutlass.Int32(BC // 2)

            group_id = lane_id // 4
            tid_in_group = lane_id % 4
            row0, row1 = group_id, group_id + 8

            thr_mma = tiled_mma_k2.get_slice(lane_id)
            thr_copy_A = tiled_copy_mma_A.get_slice(lane_id)
            thr_copy_B = tiled_copy_mma_B.get_slice(lane_id)
            thr_copy_Gn = tiled_copy_Gcum_norm.get_slice(tid_in_group)
            thr_copy_Ggate = tiled_copy_Gcum_gate.get_slice(lane_id)

            beta_row0 = sBeta[q_row_base + row0].to(cutlass.Float32)
            beta_row1 = sBeta[q_row_base + row1].to(cutlass.Float32)

            _z = cutlass.Float32(0.0)
            acc_aqk_n0_0, acc_aqk_n0_1, acc_aqk_n0_2, acc_aqk_n0_3 = _z, _z, _z, _z
            acc_aqk_n1_0, acc_aqk_n1_1, acc_aqk_n1_2, acc_aqk_n1_3 = _z, _z, _z, _z
            acc_akk_n0_0, acc_akk_n0_1, acc_akk_n0_2, acc_akk_n0_3 = _z, _z, _z, _z
            acc_akk_n1_0, acc_akk_n1_1, acc_akk_n1_2, acc_akk_n1_3 = _z, _z, _z, _z

            for k_block in cutlass.range_constexpr(NUM_MMA_K_TILES):
                sQ_tile = cute.local_tile(
                    sQ_k123, tiler=(16, 8), coord=(my_i_q, k_block)
                )
                tCrQ = tiled_mma_k2.make_fragment_A(thr_mma.partition_A(sQ_tile))
                cute.copy(
                    tiled_copy_mma_A,
                    thr_copy_A.partition_S(sQ_tile),
                    thr_copy_A.retile(tCrQ),
                )

                sKq_tile = cute.local_tile(
                    sK_k123, tiler=(16, 8), coord=(my_i_q, k_block)
                )
                tCrKq = tiled_mma_k2.make_fragment_A(thr_mma.partition_A(sKq_tile))
                cute.copy(
                    tiled_copy_mma_A,
                    thr_copy_A.partition_S(sKq_tile),
                    thr_copy_A.retile(tCrKq),
                )

                sGn_tile = cute.local_tile(
                    sGcum, tiler=(1, 8), coord=(norm_row, k_block)
                )
                tCsGn = thr_copy_Gn.partition_S(sGn_tile)
                tCrGn = cute.make_fragment_like(tCsGn, cutlass.Float32)
                cute.copy(tiled_copy_Gcum_norm, tCsGn, thr_copy_Gn.retile(tCrGn))
                g_norm_0 = tCrGn[0]
                g_norm_1 = tCrGn[1]

                sGq_tile = cute.local_tile(
                    sGcum, tiler=(16, 8), coord=(my_i_q, k_block)
                )
                tCrGq = tiled_mma_k2.make_fragment_C(thr_mma.partition_C(sGq_tile))
                cute.copy(
                    tiled_copy_Gcum_gate,
                    thr_copy_Ggate.partition_S(sGq_tile),
                    thr_copy_Ggate.retile(tCrGq),
                )
                gate_q_0 = cute.exp2(tCrGq[0] - g_norm_0, fastmath=True)
                gate_q_1 = cute.exp2(tCrGq[1] - g_norm_1, fastmath=True)
                gate_q_2 = cute.exp2(tCrGq[2] - g_norm_0, fastmath=True)
                gate_q_3 = cute.exp2(tCrGq[3] - g_norm_1, fastmath=True)

                qa0 = tCrQ[0].to(cutlass.Float32) * gate_q_0
                qa1 = tCrQ[2].to(cutlass.Float32) * gate_q_2
                qa2 = tCrQ[1].to(cutlass.Float32) * gate_q_1
                qa3 = tCrQ[3].to(cutlass.Float32) * gate_q_3
                ka0 = tCrKq[0].to(cutlass.Float32) * gate_q_0
                ka1 = tCrKq[2].to(cutlass.Float32) * gate_q_2
                ka2 = tCrKq[1].to(cutlass.Float32) * gate_q_1
                ka3 = tCrKq[3].to(cutlass.Float32) * gate_q_3

                sK_tile_n0 = cute.local_tile(
                    sK_k123, tiler=(8, 8), coord=(my_i_k * 2, k_block)
                )
                tCrK_n0 = tiled_mma_k2.make_fragment_B(thr_mma.partition_B(sK_tile_n0))
                cute.copy(
                    tiled_copy_mma_B,
                    thr_copy_B.partition_S(sK_tile_n0),
                    thr_copy_B.retile(tCrK_n0),
                )

                sK_tile_n1 = cute.local_tile(
                    sK_k123, tiler=(8, 8), coord=(my_i_k * 2 + 1, k_block)
                )
                tCrK_n1 = tiled_mma_k2.make_fragment_B(thr_mma.partition_B(sK_tile_n1))
                cute.copy(
                    tiled_copy_mma_B,
                    thr_copy_B.partition_S(sK_tile_n1),
                    thr_copy_B.retile(tCrK_n1),
                )

                sGk_tile = cute.local_tile(
                    sGcum, tiler=(16, 8), coord=(my_i_k, k_block)
                )
                tCrGk = tiled_mma_k2.make_fragment_C(thr_mma.partition_C(sGk_tile))
                cute.copy(
                    tiled_copy_Gcum_gate,
                    thr_copy_Ggate.partition_S(sGk_tile),
                    thr_copy_Ggate.retile(tCrGk),
                )
                gk_n0_0 = cute.exp2(g_norm_0 - tCrGk[0], fastmath=True)
                gk_n0_1 = cute.exp2(g_norm_1 - tCrGk[1], fastmath=True)
                k_n0_b0 = tCrK_n0[0].to(cutlass.Float32) * gk_n0_0
                k_n0_b1 = tCrK_n0[1].to(cutlass.Float32) * gk_n0_1
                gk_n1_0 = cute.exp2(g_norm_0 - tCrGk[2], fastmath=True)
                gk_n1_1 = cute.exp2(g_norm_1 - tCrGk[3], fastmath=True)
                k_n1_b0 = tCrK_n1[0].to(cutlass.Float32) * gk_n1_0
                k_n1_b1 = tCrK_n1[1].to(cutlass.Float32) * gk_n1_1

                acc_aqk_n0_0, acc_aqk_n0_1, acc_aqk_n0_2, acc_aqk_n0_3 = (
                    mma_tf32_m16n8k8(
                        qa0,
                        qa1,
                        qa2,
                        qa3,
                        k_n0_b0,
                        k_n0_b1,
                        acc_aqk_n0_0,
                        acc_aqk_n0_1,
                        acc_aqk_n0_2,
                        acc_aqk_n0_3,
                    )
                )
                acc_aqk_n1_0, acc_aqk_n1_1, acc_aqk_n1_2, acc_aqk_n1_3 = (
                    mma_tf32_m16n8k8(
                        qa0,
                        qa1,
                        qa2,
                        qa3,
                        k_n1_b0,
                        k_n1_b1,
                        acc_aqk_n1_0,
                        acc_aqk_n1_1,
                        acc_aqk_n1_2,
                        acc_aqk_n1_3,
                    )
                )
                acc_akk_n0_0, acc_akk_n0_1, acc_akk_n0_2, acc_akk_n0_3 = (
                    mma_tf32_m16n8k8(
                        ka0,
                        ka1,
                        ka2,
                        ka3,
                        k_n0_b0,
                        k_n0_b1,
                        acc_akk_n0_0,
                        acc_akk_n0_1,
                        acc_akk_n0_2,
                        acc_akk_n0_3,
                    )
                )
                acc_akk_n1_0, acc_akk_n1_1, acc_akk_n1_2, acc_akk_n1_3 = (
                    mma_tf32_m16n8k8(
                        ka0,
                        ka1,
                        ka2,
                        ka3,
                        k_n1_b0,
                        k_n1_b1,
                        acc_akk_n1_0,
                        acc_akk_n1_1,
                        acc_akk_n1_2,
                        acc_akk_n1_3,
                    )
                )

            # Store tile A
            col0, col1 = tid_in_group * 2, tid_in_group * 2 + 1
            col2, col3 = 8 + tid_in_group * 2, 8 + tid_in_group * 2 + 1

            sAqk_k123[row0, tile_col_base + col0] = (acc_aqk_n0_0 * scale).to(
                cutlass.BFloat16
            )
            sAqk_k123[row0, tile_col_base + col1] = (acc_aqk_n0_1 * scale).to(
                cutlass.BFloat16
            )
            sAqk_k123[row1, tile_col_base + col0] = (acc_aqk_n0_2 * scale).to(
                cutlass.BFloat16
            )
            sAqk_k123[row1, tile_col_base + col1] = (acc_aqk_n0_3 * scale).to(
                cutlass.BFloat16
            )
            sAqk_k123[row0, tile_col_base + col2] = (acc_aqk_n1_0 * scale).to(
                cutlass.BFloat16
            )
            sAqk_k123[row0, tile_col_base + col3] = (acc_aqk_n1_1 * scale).to(
                cutlass.BFloat16
            )
            sAqk_k123[row1, tile_col_base + col2] = (acc_aqk_n1_2 * scale).to(
                cutlass.BFloat16
            )
            sAqk_k123[row1, tile_col_base + col3] = (acc_aqk_n1_3 * scale).to(
                cutlass.BFloat16
            )

            sAkk_k123[akk_row_base + row0, akk_col_base + col0] = (
                acc_akk_n0_0 * beta_row0
            )
            sAkk_k123[akk_row_base + row0, akk_col_base + col1] = (
                acc_akk_n0_1 * beta_row0
            )
            sAkk_k123[akk_row_base + row1, akk_col_base + col0] = (
                acc_akk_n0_2 * beta_row1
            )
            sAkk_k123[akk_row_base + row1, akk_col_base + col1] = (
                acc_akk_n0_3 * beta_row1
            )
            sAkk_k123[akk_row_base + row0, akk_col_base + col2] = (
                acc_akk_n1_0 * beta_row0
            )
            sAkk_k123[akk_row_base + row0, akk_col_base + col3] = (
                acc_akk_n1_1 * beta_row0
            )
            sAkk_k123[akk_row_base + row1, akk_col_base + col2] = (
                acc_akk_n1_2 * beta_row1
            )
            sAkk_k123[akk_row_base + row1, akk_col_base + col3] = (
                acc_akk_n1_3 * beta_row1
            )

            # --- Tile B (warps 6-7 only): second tile of row 3 pair ---
            # W6: (3,1), W7: (3,3). Accumulators reused (zero extra reg pressure).
            if mma_warp >= 6:
                k_row_base_b = my_i_k_b * BC
                tile_col_base_b = tile_col_base + BC  # tile B = tile A + 1 slot
                akk_row_base_b = k_row_base_b
                norm_row_b = q_row_base
                if my_i_q == my_i_k_b:
                    norm_row_b = q_row_base + cutlass.Int32(BC // 2)

                acc_aqk_n0_0, acc_aqk_n0_1, acc_aqk_n0_2, acc_aqk_n0_3 = _z, _z, _z, _z
                acc_aqk_n1_0, acc_aqk_n1_1, acc_aqk_n1_2, acc_aqk_n1_3 = _z, _z, _z, _z
                acc_akk_n0_0, acc_akk_n0_1, acc_akk_n0_2, acc_akk_n0_3 = _z, _z, _z, _z
                acc_akk_n1_0, acc_akk_n1_1, acc_akk_n1_2, acc_akk_n1_3 = _z, _z, _z, _z

                for k_block in cutlass.range_constexpr(NUM_MMA_K_TILES):
                    sQ_tile = cute.local_tile(
                        sQ_k123, tiler=(16, 8), coord=(my_i_q, k_block)
                    )
                    tCrQ = tiled_mma_k2.make_fragment_A(thr_mma.partition_A(sQ_tile))
                    cute.copy(
                        tiled_copy_mma_A,
                        thr_copy_A.partition_S(sQ_tile),
                        thr_copy_A.retile(tCrQ),
                    )

                    sKq_tile = cute.local_tile(
                        sK_k123, tiler=(16, 8), coord=(my_i_q, k_block)
                    )
                    tCrKq = tiled_mma_k2.make_fragment_A(thr_mma.partition_A(sKq_tile))
                    cute.copy(
                        tiled_copy_mma_A,
                        thr_copy_A.partition_S(sKq_tile),
                        thr_copy_A.retile(tCrKq),
                    )

                    sGn_tile = cute.local_tile(
                        sGcum, tiler=(1, 8), coord=(norm_row_b, k_block)
                    )
                    tCsGn = thr_copy_Gn.partition_S(sGn_tile)
                    tCrGn = cute.make_fragment_like(tCsGn, cutlass.Float32)
                    cute.copy(tiled_copy_Gcum_norm, tCsGn, thr_copy_Gn.retile(tCrGn))
                    g_norm_0 = tCrGn[0]
                    g_norm_1 = tCrGn[1]

                    sGq_tile = cute.local_tile(
                        sGcum, tiler=(16, 8), coord=(my_i_q, k_block)
                    )
                    tCrGq = tiled_mma_k2.make_fragment_C(thr_mma.partition_C(sGq_tile))
                    cute.copy(
                        tiled_copy_Gcum_gate,
                        thr_copy_Ggate.partition_S(sGq_tile),
                        thr_copy_Ggate.retile(tCrGq),
                    )
                    gate_q_0 = cute.exp2(tCrGq[0] - g_norm_0, fastmath=True)
                    gate_q_1 = cute.exp2(tCrGq[1] - g_norm_1, fastmath=True)
                    gate_q_2 = cute.exp2(tCrGq[2] - g_norm_0, fastmath=True)
                    gate_q_3 = cute.exp2(tCrGq[3] - g_norm_1, fastmath=True)

                    qa0 = tCrQ[0].to(cutlass.Float32) * gate_q_0
                    qa1 = tCrQ[2].to(cutlass.Float32) * gate_q_2
                    qa2 = tCrQ[1].to(cutlass.Float32) * gate_q_1
                    qa3 = tCrQ[3].to(cutlass.Float32) * gate_q_3
                    ka0 = tCrKq[0].to(cutlass.Float32) * gate_q_0
                    ka1 = tCrKq[2].to(cutlass.Float32) * gate_q_2
                    ka2 = tCrKq[1].to(cutlass.Float32) * gate_q_1
                    ka3 = tCrKq[3].to(cutlass.Float32) * gate_q_3

                    sK_tile_n0 = cute.local_tile(
                        sK_k123, tiler=(8, 8), coord=(my_i_k_b * 2, k_block)
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
                        sK_k123, tiler=(8, 8), coord=(my_i_k_b * 2 + 1, k_block)
                    )
                    tCrK_n1 = tiled_mma_k2.make_fragment_B(
                        thr_mma.partition_B(sK_tile_n1)
                    )
                    cute.copy(
                        tiled_copy_mma_B,
                        thr_copy_B.partition_S(sK_tile_n1),
                        thr_copy_B.retile(tCrK_n1),
                    )

                    sGk_tile = cute.local_tile(
                        sGcum, tiler=(16, 8), coord=(my_i_k_b, k_block)
                    )
                    tCrGk = tiled_mma_k2.make_fragment_C(thr_mma.partition_C(sGk_tile))
                    cute.copy(
                        tiled_copy_Gcum_gate,
                        thr_copy_Ggate.partition_S(sGk_tile),
                        thr_copy_Ggate.retile(tCrGk),
                    )
                    gk_n0_0 = cute.exp2(g_norm_0 - tCrGk[0], fastmath=True)
                    gk_n0_1 = cute.exp2(g_norm_1 - tCrGk[1], fastmath=True)
                    k_n0_b0 = tCrK_n0[0].to(cutlass.Float32) * gk_n0_0
                    k_n0_b1 = tCrK_n0[1].to(cutlass.Float32) * gk_n0_1
                    gk_n1_0 = cute.exp2(g_norm_0 - tCrGk[2], fastmath=True)
                    gk_n1_1 = cute.exp2(g_norm_1 - tCrGk[3], fastmath=True)
                    k_n1_b0 = tCrK_n1[0].to(cutlass.Float32) * gk_n1_0
                    k_n1_b1 = tCrK_n1[1].to(cutlass.Float32) * gk_n1_1

                    acc_aqk_n0_0, acc_aqk_n0_1, acc_aqk_n0_2, acc_aqk_n0_3 = (
                        mma_tf32_m16n8k8(
                            qa0,
                            qa1,
                            qa2,
                            qa3,
                            k_n0_b0,
                            k_n0_b1,
                            acc_aqk_n0_0,
                            acc_aqk_n0_1,
                            acc_aqk_n0_2,
                            acc_aqk_n0_3,
                        )
                    )
                    acc_aqk_n1_0, acc_aqk_n1_1, acc_aqk_n1_2, acc_aqk_n1_3 = (
                        mma_tf32_m16n8k8(
                            qa0,
                            qa1,
                            qa2,
                            qa3,
                            k_n1_b0,
                            k_n1_b1,
                            acc_aqk_n1_0,
                            acc_aqk_n1_1,
                            acc_aqk_n1_2,
                            acc_aqk_n1_3,
                        )
                    )
                    acc_akk_n0_0, acc_akk_n0_1, acc_akk_n0_2, acc_akk_n0_3 = (
                        mma_tf32_m16n8k8(
                            ka0,
                            ka1,
                            ka2,
                            ka3,
                            k_n0_b0,
                            k_n0_b1,
                            acc_akk_n0_0,
                            acc_akk_n0_1,
                            acc_akk_n0_2,
                            acc_akk_n0_3,
                        )
                    )
                    acc_akk_n1_0, acc_akk_n1_1, acc_akk_n1_2, acc_akk_n1_3 = (
                        mma_tf32_m16n8k8(
                            ka0,
                            ka1,
                            ka2,
                            ka3,
                            k_n1_b0,
                            k_n1_b1,
                            acc_akk_n1_0,
                            acc_akk_n1_1,
                            acc_akk_n1_2,
                            acc_akk_n1_3,
                        )
                    )

                # Store tile B
                sAqk_k123[row0, tile_col_base_b + col0] = (acc_aqk_n0_0 * scale).to(
                    cutlass.BFloat16
                )
                sAqk_k123[row0, tile_col_base_b + col1] = (acc_aqk_n0_1 * scale).to(
                    cutlass.BFloat16
                )
                sAqk_k123[row1, tile_col_base_b + col0] = (acc_aqk_n0_2 * scale).to(
                    cutlass.BFloat16
                )
                sAqk_k123[row1, tile_col_base_b + col1] = (acc_aqk_n0_3 * scale).to(
                    cutlass.BFloat16
                )
                sAqk_k123[row0, tile_col_base_b + col2] = (acc_aqk_n1_0 * scale).to(
                    cutlass.BFloat16
                )
                sAqk_k123[row0, tile_col_base_b + col3] = (acc_aqk_n1_1 * scale).to(
                    cutlass.BFloat16
                )
                sAqk_k123[row1, tile_col_base_b + col2] = (acc_aqk_n1_2 * scale).to(
                    cutlass.BFloat16
                )
                sAqk_k123[row1, tile_col_base_b + col3] = (acc_aqk_n1_3 * scale).to(
                    cutlass.BFloat16
                )

                sAkk_k123[akk_row_base_b + row0, akk_col_base + col0] = (
                    acc_akk_n0_0 * beta_row0
                )
                sAkk_k123[akk_row_base_b + row0, akk_col_base + col1] = (
                    acc_akk_n0_1 * beta_row0
                )
                sAkk_k123[akk_row_base_b + row1, akk_col_base + col0] = (
                    acc_akk_n0_2 * beta_row1
                )
                sAkk_k123[akk_row_base_b + row1, akk_col_base + col1] = (
                    acc_akk_n0_3 * beta_row1
                )
                sAkk_k123[akk_row_base_b + row0, akk_col_base + col2] = (
                    acc_akk_n1_0 * beta_row0
                )
                sAkk_k123[akk_row_base_b + row0, akk_col_base + col3] = (
                    acc_akk_n1_1 * beta_row0
                )
                sAkk_k123[akk_row_base_b + row1, akk_col_base + col2] = (
                    acc_akk_n1_2 * beta_row1
                )
                sAkk_k123[akk_row_base_b + row1, akk_col_base + col3] = (
                    acc_akk_n1_3 * beta_row1
                )

            cute.arch.mbarrier_arrive(k123_mma_done_mbar)

            # K2 idle during K4 phase
            phase_nbar.arrive_and_wait()  # K123->K4
            phase_nbar.arrive_and_wait()  # K4->next

        if PROFILE_CLOCKS:
            clk_k2_end = read_clock()
            if mma_warp == 0 and lane_id == 0:
                mClocks[10, bid] = clk_k2_end - clk_k2_start

    # ==============================================================
    # MMA warp (warp 0): 6 MMAs per chunk
    # ==============================================================
    elif warp_idx == K4_MMA_WARP:

        mc = (0, 0, 0, 0)
        ml = cute.make_layout((1, 1, 1, 1))

        # Clock accumulators (thread 0 only, i64)
        clk_k123_acc = cutlass.Int64(0)
        clk_k4_acc = cutlass.Int64(0)
        clk0 = cutlass.Int64(0)
        clk1 = cutlass.Int64(0)
        clk2 = cutlass.Int64(0)
        clk_mma1_acc = cutlass.Int64(0)
        clk_mma2_acc = cutlass.Int64(0)
        clk_wait_s_mma4_acc = cutlass.Int64(0)
        clk_wait_w_mma3_acc = cutlass.Int64(0)
        clk_wait_nv_mma5_acc = cutlass.Int64(0)
        clk_mma6_drain_acc = cutlass.Int64(0)
        clk_k4a = cutlass.Int64(0)
        clk_k4b = cutlass.Int64(0)

        for chunk_c in cutlass.range(num_chunks):
            chunk_start = i_b * num_chunks * BT + chunk_c * BT

            if PROFILE_CLOCKS:
                clk0 = read_clock()

            # Wait for K123 to finish
            phase_nbar.arrive_and_wait()  # K123->K4

            if PROFILE_CLOCKS:
                clk1 = read_clock()
                clk_k123_acc = clk_k123_acc + (clk1 - clk0)

            if DEBUG_K4_LEVEL == 0:
                pass  # Skip K4 entirely for debugging
            else:
                fA_kmn = thr_kmn.make_fragment_A(sAB)
                fB_ks = thr_kmn.make_fragment_B(sKS)
                fB_v = thr_kmn.make_fragment_B(sV_ext)
                fA_w = thr_kmn.make_fragment_A(sW)
                fB_s = thr_kmn.make_fragment_B(sST)
                fA_q = thr_kmn.make_fragment_A(sQS)
                fA_aqc = thr_kmn.make_fragment_A(sAQC)
                fB_nv = thr_kmn.make_fragment_B(sNV_b)
                fA_nv_mn = thr_mn.make_fragment_A(sNV_a)
                fB_kg = thr_mn.make_fragment_B(sKG)

                if PROFILE_CLOCKS:
                    clk_k4a = read_clock()

                # MMA1: W = AB @ KS
                w_h = w_prod.acquire_and_advance()
                tiled_mma_kmn.set(Field.ACCUMULATE, False)
                for k in cutlass.range_constexpr(cute.size(sKS.shape[2])):
                    cute.gemm(
                        tiled_mma_kmn,
                        tCtW,
                        fA_kmn[dice + (0,)][None, None, k],
                        fB_ks[dice + (0,)][None, None, k],
                        tCtW,
                    )
                    if k == 0:
                        tiled_mma_kmn.set(Field.ACCUMULATE, True)
                w_h.commit()

                if PROFILE_CLOCKS:
                    clk_k4b = read_clock()
                    clk_mma1_acc = clk_mma1_acc + (clk_k4b - clk_k4a)
                    clk_k4a = clk_k4b

                # MMA2: U = AB @ V
                vh = v_cons.wait_and_advance()
                tiled_mma_kmn.set(Field.ACCUMULATE, False)
                for k in cutlass.range_constexpr(cute.size(sV_ext.shape[2])):
                    cute.gemm(
                        tiled_mma_kmn,
                        tCtNV,
                        fA_kmn[dice + (0,)][None, None, k],
                        fB_v[dice + (vh.index,)][None, None, k],
                        tCtNV,
                    )
                    if k == 0:
                        tiled_mma_kmn.set(Field.ACCUMULATE, True)
                vh.release()

                if PROFILE_CLOCKS:
                    clk_k4b = read_clock()
                    clk_mma2_acc = clk_mma2_acc + (clk_k4b - clk_k4a)
                    clk_k4a = clk_k4b

                # Wait WG1: sST bf16 ready in SMEM
                cute.arch.mbarrier_wait(st_ready_mbar, phase=chunk_c % 2)

                # MMA4: OI = QS @ S
                tiled_mma_kmn.set(Field.ACCUMULATE, False)
                for k in cutlass.range_constexpr(cute.size(sST.shape[2])):
                    cute.gemm(
                        tiled_mma_kmn,
                        tCtO,
                        fA_q[dice + (0,)][None, None, k],
                        fB_s[dice + (0,)][None, None, k],
                        tCtO,
                    )
                    if k == 0:
                        tiled_mma_kmn.set(Field.ACCUMULATE, True)

                if PROFILE_CLOCKS:
                    clk_k4b = read_clock()
                    clk_wait_s_mma4_acc = clk_wait_s_mma4_acc + (clk_k4b - clk_k4a)
                    clk_k4a = clk_k4b

                # Wait W readout
                sW_ready_nbar.arrive_and_wait()

                # MMA3: NV += sW @ S (accumulate into U)
                nv_h = nv_prod.acquire_and_advance()
                tiled_mma_kmn.set(Field.ACCUMULATE, True)
                for k in cutlass.range_constexpr(cute.size(sST.shape[2])):
                    cute.gemm(
                        tiled_mma_kmn,
                        tCtNV,
                        fA_w[dice + (0,)][None, None, k],
                        fB_s[dice + (0,)][None, None, k],
                        tCtNV,
                    )
                nv_h.commit()

                if PROFILE_CLOCKS:
                    clk_k4b = read_clock()
                    clk_wait_w_mma3_acc = clk_wait_w_mma3_acc + (clk_k4b - clk_k4a)
                    clk_k4a = clk_k4b

                # Wait NV readout
                sNV_ready_nbar.arrive_and_wait()

                # MMA5: O += AQC @ NV
                o_h = o_prod.acquire_and_advance()
                if not DEBUG_SKIP_MMA5:
                    tiled_mma_kmn.set(Field.ACCUMULATE, True)
                    for k in cutlass.range_constexpr(cute.size(sNV_b.shape[2])):
                        cute.gemm(
                            tiled_mma_kmn,
                            tCtO,
                            fA_aqc[dice + (0,)][None, None, k],
                            fB_nv[dice + (0,)][None, None, k],
                            tCtO,
                        )
                o_h.commit()

                if PROFILE_CLOCKS:
                    clk_k4b = read_clock()
                    clk_wait_nv_mma5_acc = clk_wait_nv_mma5_acc + (clk_k4b - clk_k4a)
                    clk_k4a = clk_k4b

                # Wait WG1: decayed state written back to TMEM
                cute.arch.mbarrier_wait(state_decayed_mbar, phase=chunk_c % 2)

                # MMA6: accumulate SU = NV^T @ KG onto decayed state in TMEM@256
                tiled_mma_mn_mn.set(Field.ACCUMULATE, True)
                for k in cutlass.range_constexpr(cute.size(sKG.shape[2])):
                    cute.gemm(
                        tiled_mma_mn_mn,
                        tCtState,
                        fA_nv_mn[dice + (0,)][None, None, k],
                        fB_kg[dice + (0,)][None, None, k],
                        tCtState,
                    )

                # Drain W/NV/O UMMA pipelines, then sync with TMA+Readout
                w_prod.tail()
                nv_prod.tail()
                o_prod.tail()
                store_nbar.arrive_and_wait()

                # Signal WG1: MMA6 done (all chunks, including last)
                cute.arch.mbarrier_arrive(mma6_done_mbar)

                if PROFILE_CLOCKS:
                    clk_k4b = read_clock()
                    clk_mma6_drain_acc = clk_mma6_drain_acc + (clk_k4b - clk_k4a)

            # K4->next chunk
            phase_nbar.arrive_and_wait()

            if PROFILE_CLOCKS:
                clk2 = read_clock()
                clk_k4_acc = clk_k4_acc + (clk2 - clk1)

        # Store accumulated clocks (thread 0 = warp 0 = MMA warp)
        if PROFILE_CLOCKS:
            if tidx == 0:
                mClocks[0, bid] = clk_k123_acc
                mClocks[1, bid] = clk_k4_acc
                mClocks[2, bid] = clk_mma1_acc
                mClocks[3, bid] = clk_mma2_acc
                mClocks[4, bid] = clk_wait_s_mma4_acc
                mClocks[5, bid] = clk_wait_w_mma3_acc
                mClocks[6, bid] = clk_wait_nv_mma5_acc
                mClocks[7, bid] = clk_mma6_drain_acc

        # Wait for WG1 final state store, then dealloc TMEM
        cute.arch.mbarrier_wait(final_state_done_mbar, phase=0)
        cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.dealloc_tmem(tmem_ptr, 512)

    # ==============================================================
    # TMA warp (warp 2): V early load + O store
    # ==============================================================
    elif warp_idx == K4_TMA_WARP:

        mc = (0, 0, 0, 0)
        ml = cute.make_layout((1, 1, 1, 1))
        clk_tma_start = cutlass.Int64(0)
        if PROFILE_CLOCKS:
            clk_tma_start = read_clock()

        for chunk_c in cutlass.range(num_chunks):
            chunk_start = i_b * num_chunks * BT + chunk_c * BT

            # V early load (overlaps K123 phase)
            gV_h = mV_nkl[(None, None, i_h)]
            gV = cute.local_tile(
                cute.domain_offset((0, chunk_start), gV_h),
                tiler=(N4, K4_K),
                coord=(0, 0),
            )
            tBsV, tBgV = cpasync.tma_partition(
                tma_atom_v,
                mc,
                ml,
                cute.group_modes(sV_ext, 0, 3),
                cute.group_modes(thr_kmn.partition_B(gV), 0, 3),
            )
            vh = v_prod.acquire_and_advance()
            cute.copy(tma_atom_v, tBgV, tBsV[None, vh.index], tma_bar_ptr=vh.barrier)

            # Wait for K123 to finish
            phase_nbar.arrive_and_wait()  # K123->K4

            # Store O[c] to GMEM
            gO_h = mOo[(None, None, i_h)]  # [T_total, V_dim]

            # Wait for MMA + Readout to finish (O is in SMEM)
            store_nbar.arrive_and_wait()

            gOo_c = cute.local_tile(
                cute.domain_offset((chunk_start, 0), gO_h), tiler=(M4, N4), coord=(0, 0)
            )
            sOt, gOt = cpasync.tma_partition(
                tma_atom_o_st,
                mc,
                ml,
                cute.group_modes(sO_st, 0, 2),
                cute.group_modes(gOo_c, 0, 2),
            )
            cute.copy(tma_atom_o_st, sOt, gOt)

            # Fence: wait for TMA to finish reading sO from SMEM
            tma_store_fence()

            # K4->next chunk
            phase_nbar.arrive_and_wait()

        # Drain V pipeline
        v_prod.tail()

        if PROFILE_CLOCKS:
            clk_tma_end = read_clock()
            if lane_id == 0:
                mClocks[12, bid] = clk_tma_end - clk_tma_start

    # ==============================================================
    # Idle warps (1, 3): phase barriers only
    # ==============================================================
    else:

        for chunk_c in cutlass.range(num_chunks):
            phase_nbar.arrive_and_wait()  # K123->K4
            phase_nbar.arrive_and_wait()  # K4->next


# =============================================================================
# Host Function
# =============================================================================


def make_host_fn(has_bias=False, use_safe_gate=False, profile_clocks=False):
    """Create the host function for the fused K1234 kernel.

    B, H, NC are all dynamic — one compilation serves all shapes.
    Only mode flags (has_bias, use_safe_gate, profile_clocks) specialize at compile time.
    """
    _HAS_BIAS = 1 if has_bias else 0
    _USE_SAFE_GATE = 1 if use_safe_gate else 0
    _PROFILE_CLOCKS = 1 if profile_clocks else 0

    tile1 = (M4, N4, K4_K)
    tile3 = (M4, N4, K4_K3)
    tile6 = (M6, N6, K6)

    @cute.jit
    def host_fn(
        # K123 raw inputs
        mQ,
        mK,
        mG,
        mA_log,
        mBeta,
        scale_val,
        # K4 inputs (raw pointers — host_fn creates 3D views internally)
        mV_in,  # [B, T, H, V_dim] bf16
        mO_out,  # [B, T, H, V_dim] bf16 output
        mS_fp32,  # [B*H, K, V] fp32 state (init + final output)
        # dt_bias / safe_gate
        mDtBias,  # [H, K] fp32 (or dummy [1,1] if no bias)
        lower_bound_val,  # float (0.0 if unused)
        # Clock profiling
        mClocks,  # [2, B*H] i64 (or dummy if not profiling)
        # Runtime dimensions (dynamic — no recompilation needed)
        num_chunks: cutlass.Int32,
        num_heads: cutlass.Int32,
        batch_size: cutlass.Int32,
        stream: cuda_drv.CUstream,
    ):
        # Derive GMEM shapes from runtime dimensions
        T_total = batch_size * num_chunks * BT
        s_row = num_heads * K_DIM
        s_col = 1
        s_h = K_DIM
        BH = batch_size * num_heads

        # --- Common 3D layout for [T_total, K_DIM, H] tensors ---
        view_layout_3d = cute.make_layout(
            (T_total, K_DIM, num_heads), stride=(s_row, s_col, s_h)
        )

        # --- K4 MMA setup ---
        mma_kmn = sm100_utils.make_trivial_tiled_mma(
            mma_dtype,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            (M4, N4),
            OperandSource.SMEM,
        )
        mma_mn_mn = sm100_utils.make_trivial_tiled_mma(
            mma_dtype,
            OperandMajorMode.MN,
            OperandMajorMode.MN,
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            (M6, N6),
            OperandSource.SMEM,
        )

        sl_ab = sm100_utils.make_smem_layout_a(mma_kmn, tile1, mma_dtype, 1)
        sl_ks = sm100_utils.make_smem_layout_b(mma_kmn, tile1, mma_dtype, 1)
        sl_v = sm100_utils.make_smem_layout_b(mma_kmn, tile1, mma_dtype, 1)
        sl_s = sm100_utils.make_smem_layout_b(mma_kmn, tile3, mma_dtype, 1)
        sl_qs = sm100_utils.make_smem_layout_a(mma_kmn, tile3, mma_dtype, 1)
        sl_aqc = sm100_utils.make_smem_layout_a(mma_kmn, tile1, mma_dtype, 1)
        sl_kg = sm100_utils.make_smem_layout_b(mma_mn_mn, tile6, mma_dtype, 1)

        sl_readout_k = sm100_utils.make_smem_layout_a(mma_kmn, tile3, mma_dtype, 1)
        sl_nv_b = sm100_utils.make_smem_layout_b(mma_kmn, tile1, mma_dtype, 1)
        sl_nv_a = sm100_utils.make_smem_layout_a(mma_mn_mn, tile6, mma_dtype, 1)

        # --- K4 3D GMEM views (V, O, S_fp32 — AB/KS/QS/AQC/KG in Zone B SMEM) ---
        b_view_k4 = cute.make_layout(
            (K_DIM, T_total, num_heads), stride=(1, num_heads * K_DIM, K_DIM)
        )
        mV_k4 = cute.make_tensor(mV_in.iterator, b_view_k4)

        # O output: [T_total, V_dim, H] strides (H*V_dim, 1, V_dim) — same as A K_DIM view
        mO_k4 = cute.make_tensor(mO_out.iterator, view_layout_3d)

        # State S_fp32: [V, K, B*H] view — mode 0=V(stride 1), mode 1=K(stride V_dim)
        # Matches TMEM state [V,K] convention (M=V, N=K from tiled_mma_mn_mn)
        s_fp32_vk_view = cute.make_layout(
            (K_DIM, K_DIM, BH), stride=(1, K_DIM, K_DIM * K_DIM)
        )
        mS_fp32_vk = cute.make_tensor(mS_fp32.iterator, s_fp32_vk_view)

        tma_ld = cpasync.CopyBulkTensorTileG2SOp()
        # Only V loaded via TMA (S now in TMEM, not TMA)
        ta_v, mVt = cute.nvgpu.make_tiled_tma_atom_B(
            tma_ld, mV_k4, cute.select(sl_v, mode=[0, 1, 2]), tile1, mma_kmn
        )

        # TMA store for O (3D view)
        sk = sm100_utils.get_smem_layout_atom_ab(
            OperandMajorMode.K, out_dtype, (M4, N4)
        )
        sl_store = cute.tile_to_shape(
            sm100_utils.make_smem_layout_atom(sk, out_dtype), (M4, N4), order=(0, 1)
        )
        tma_st = cpasync.CopyBulkTensorTileS2GOp()
        ta_o, mOo = cpasync.make_tiled_tma_atom(tma_st, mO_k4, sl_store, (M4, N4))

        # --- K123 TMA setup ---
        mQ_view = cute.make_tensor(mQ.iterator, view_layout_3d)
        mK_view = cute.make_tensor(mK.iterator, view_layout_3d)
        mG_view = cute.make_tensor(mG.iterator, view_layout_3d)

        smem_atom_qk = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        qk_smem_2d = cute.tile_to_shape(smem_atom_qk, (BT, K_DIM), order=(0, 1))

        g_smem_2d = cute.make_layout((BT, K_DIM), stride=(K_DIM, 1))

        tma_op_k123 = cpasync.CopyBulkTensorTileG2SOp(cpasync.CtaGroup.ONE)
        ta_q_k123, tt_q = cpasync.make_tiled_tma_atom(
            tma_op_k123,
            mQ_view,
            qk_smem_2d,
            cute.product_each(qk_smem_2d.shape),
            num_multicast=1,
        )
        ta_k_k123, tt_k = cpasync.make_tiled_tma_atom(
            tma_op_k123,
            mK_view,
            qk_smem_2d,
            cute.product_each(qk_smem_2d.shape),
            num_multicast=1,
        )
        ta_g_k123, tt_g = cpasync.make_tiled_tma_atom(
            tma_op_k123,
            mG_view,
            g_smem_2d,
            cute.product_each(g_smem_2d.shape),
            num_multicast=1,
        )

        g_cumsum_layout = cute.make_layout((BT, K_STRIDE), stride=(K_STRIDE, 1))

        # K123 tiled copies
        copy_atom_qk_k1 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=64
        )
        tiled_copy_qk_k1 = cute.make_tiled_copy_tv(
            copy_atom_qk_k1,
            thr_layout=cute.make_layout((1, 32)),
            val_layout=cute.make_layout((1, 4)),
        )

        # K123 output v2 views removed — k_scaled/q_scaled/kg/gk_last now write to SMEM

        mma_op_k2 = cute.nvgpu.warp.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (16, 8, 8)
        )
        tiled_mma_k2 = cute.make_tiled_mma(
            mma_op_k2, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 8, 8)
        )
        tiled_copy_mma_A = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 2), cutlass.BFloat16
            ),
            tiled_mma_k2,
        )
        tiled_copy_mma_B = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 1), cutlass.BFloat16
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

        fused_k1234_kernel(
            mma_kmn,
            mma_mn_mn,
            # K4 TMA: V only (S in TMEM, not TMA)
            ta_v,
            mVt,
            sl_v,
            sl_s,  # sST SMEM layout (filled from TMEM, not TMA)
            # Zone B swizzle layouts
            sl_ab,
            sl_ks,
            sl_qs,
            sl_aqc,
            sl_kg,
            # K4 TMA store + readout
            ta_o,
            mOo,
            sl_store,
            sl_readout_k,
            sl_nv_b,
            sl_nv_a,
            # K123 TMA
            ta_q_k123,
            tt_q,
            ta_k_k123,
            tt_k,
            ta_g_k123,
            tt_g,
            qk_smem_2d,
            g_smem_2d,
            g_cumsum_layout,
            tiled_copy_qk_k1,
            tiled_mma_k2,
            tiled_copy_mma_A,
            tiled_copy_mma_B,
            tiled_copy_Gcum_norm,
            tiled_copy_Gcum_gate,
            # K123 GMEM
            mA_log,
            mBeta,
            scale_val,
            # K4 GMEM: S_fp32 [V,K,BH] for initial/final state
            mS_fp32_vk,
            mDtBias,
            lower_bound_val,
            _HAS_BIAS,
            _USE_SAFE_GATE,
            mClocks,
            _PROFILE_CLOCKS,
            num_chunks,
            num_heads,
            batch_size,
        ).launch(
            grid=(BH, 1, 1),
            block=(THREADS, 1, 1),
            smem=225 * 1024,
            stream=stream,
        )  # Force high SMEM to prevent >1 block per SM (TMEM conflict)

    return host_fn


# =============================================================================
# Helpers
# =============================================================================


def _wrap(t, leading_dim=1, divisibility=None):
    r = from_dlpack(t, assumed_align=32).mark_layout_dynamic(leading_dim=leading_dim)
    if divisibility is not None:
        r = r.mark_compact_shape_dynamic(mode=leading_dim, divisibility=divisibility)
    return r


def _ct(t, etype=None):
    """Wrap PyTorch tensor as CuTe tensor with explicit element type."""
    r = from_dlpack(t, assumed_align=16)
    if etype is not None:
        r.element_type = etype
    return r


# =============================================================================
# Test (standalone, eqlen only for now)
# =============================================================================


def _run_test(
    NC, B, H, K_dim, V_dim, safe_gate, use_dt_bias, seed=42, use_zero_state=False
):
    """Run a single fused K1234 test with given config."""
    T = NC * BT
    dev = "cuda"
    torch.manual_seed(seed)

    mode = "safe_gate" if safe_gate else "softplus"
    bias_tag = "+dt_bias" if use_dt_bias else ""
    print(f"\n  [{mode}{bias_tag}] B={B}, H={H}, NC={NC}, T={T}, K={K_dim}")

    # Raw inputs
    q = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, T, H, V_dim, dtype=torch.bfloat16, device=dev)
    g = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=dev) * 0.1
    beta = torch.rand(B, T, H, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.5
    scale = float(K_dim**-0.5)

    if use_dt_bias:
        dt_bias = torch.randn(H * K_dim, dtype=torch.float32, device=dev) * 0.1
    else:
        dt_bias = None
    lower_bound = -5.0 if safe_gate else None

    # K4 state buffers (state in TMEM — only need fp32 for init/final)
    BH = B * H
    if use_zero_state:
        S_init = torch.zeros(B, H, K_dim, V_dim, dtype=torch.float32, device=dev)
    else:
        S_init = torch.randn(B, H, K_dim, V_dim, dtype=torch.float32, device=dev) * 0.1
    S_fp32 = S_init.reshape(BH, K_dim, V_dim).clone()
    O_out = torch.zeros(B, T, H, V_dim, dtype=torch.bfloat16, device=dev)

    # dt_bias CuTe tensor (or dummy)
    bf = cutlass.BFloat16
    fp = cutlass.Float32
    has_bias = dt_bias is not None
    if has_bias:
        bias_ct = _ct(dt_bias.float().contiguous().view(H, K_dim), fp)
    else:
        bias_ct = _ct(torch.empty(1, 1, dtype=torch.float32, device=dev), fp)
    lb_val = float(lower_bound) if lower_bound is not None else 0.0

    # Wrap as CuTe tensors
    q_ct = _ct(q, bf)
    k_ct = _ct(k, bf)
    g_ct = _ct(g, bf)
    alog_ct = _ct(A_log, fp)
    beta_ct = _ct(beta, bf)
    v_ct = _ct(v, bf)
    o_ct = _ct(O_out, bf)
    sfp_ct = _ct(S_fp32, fp)

    # Compile
    print("    Compiling...", flush=True)
    # Dummy clocks tensor (no profiling in correctness tests)
    clocks_dummy = torch.zeros(16, BH, dtype=torch.int64, device=dev)
    clocks_ct = _ct(clocks_dummy)
    clocks_ct.element_type = cutlass.Int64

    host_fn = make_host_fn(has_bias=has_bias, use_safe_gate=safe_gate)
    ct_args = (
        q_ct,
        k_ct,
        g_ct,
        alog_ct,
        beta_ct,
        scale,
        v_ct,
        o_ct,
        sfp_ct,
        bias_ct,
        lb_val,
        clocks_ct,
        NC,
        H,
        B,
        cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    compiled = cutlass.cute.compile(host_fn, *ct_args)

    # --- Run standalone K123 to get reference intermediates for Python K4 ref ---
    # (K123 intermediates are now in Zone B SMEM, not accessible from host)
    k_sc_ref = q_sc_ref = kg_ref = gk_ref = aqk_ref = akk_ref = None
    if NC >= 2:
        print("    Running standalone K123 reference...", flush=True)
        try:
            from fuse_kernel123_no_persistent import _compile_fused as _k123_compile

            k_sc_ref = torch.empty(B, T, H, K_dim, dtype=torch.bfloat16, device=dev)
            q_sc_ref = torch.empty(B, T, H, K_dim, dtype=torch.bfloat16, device=dev)
            kg_ref = torch.empty(B, T, H, K_dim, dtype=torch.bfloat16, device=dev)
            gk_ref = torch.empty(B, NC, H, K_dim, dtype=torch.float32, device=dev)
            aqk_ref = torch.zeros(B, T, H, BT, dtype=torch.bfloat16, device=dev)
            akk_ref = torch.zeros(B, T, H, BT, dtype=torch.bfloat16, device=dev)
            compiled_k123, ct_args_k123 = _k123_compile(
                B,
                NC,
                H,
                q.clone(),
                k.clone(),
                g.clone(),
                A_log.clone(),
                beta.clone(),
                scale,
                k_sc_ref,
                kg_ref,
                q_sc_ref,
                gk_ref,
                aqk_ref,
                akk_ref,
                dt_bias=dt_bias,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
            )
            compiled_k123(*ct_args_k123)
            torch.cuda.synchronize()
            print("    Standalone K123 completed OK", flush=True)
        except Exception as e:
            print(f"    Standalone K123 failed: {e}", flush=True)
            k_sc_ref = None
    else:
        print(
            "    Skipping standalone K123 (NC < 2, CPB=2 requires NC >= 2)", flush=True
        )

    # --- Run fused K1234 ---
    print("    Running fused K1234...", flush=True)
    compiled(*ct_args)
    torch.cuda.synchronize()

    # --- Per-batch diagnostics ---
    if B > 1:
        for b_idx in range(min(B, 2)):
            print(f"    Batch {b_idx} diagnostics:")
            print(
                f"      O_out[{b_idx},0,0,:4]: {O_out[b_idx,0,0,:4].float().tolist()}  hasnan={O_out[b_idx].isnan().any().item()}"
            )
            bh = b_idx * H
            print(
                f"      S_fp32[bh={bh},:2,:2]: {S_fp32[bh,:2,:2].tolist()}  hasnan={S_fp32[bh].isnan().any().item()}"
            )

    verbose = B == 1 and H == 1 and NC <= 8

    # --- gk_last now in SMEM — no GMEM comparison available ---

    # --- Python K4 reference using standalone K123 intermediates ---
    # NOTE: In the SMEM version, fused K123 outputs stay in Zone B SMEM (not accessible
    # from host). The pyref uses standalone K123 outputs instead, which may differ slightly
    # from the fused kernel's K123 due to different compilation/SMEM layout. This makes the
    # pyref comparison informational-only. The FLA comparison below is the primary check.
    ok = True
    O_py_chunks = []
    if k_sc_ref is not None and NC <= 8:
        print("    Python K4 reference (b=0, h=0):", flush=True)
        S_py = S_init[0, 0].clone()  # [K, V] fp32 — same random initial state
        O_py_chunks = []
        causal = torch.tril(torch.ones(BT, BT, device=dev))
        for c in range(NC):
            cs, ce = c * BT, (c + 1) * BT
            ks_c = k_sc_ref[0, cs:ce, 0].float()  # standalone K123 output (not negated)
            qs_c = q_sc_ref[0, cs:ce, 0].float()
            kg_c = kg_ref[0, cs:ce, 0].float()
            v_c = v[0, cs:ce, 0].float()
            aqk_c = aqk_ref[0, cs:ce, 0].float()
            akk_c_raw = akk_ref[
                0, cs:ce, 0
            ].float()  # inverted Akk (K2 has beta, no extra beta in store)
            gk_c = gk_ref[0, c, 0].float()
            beta_c = beta[0, cs:ce, 0].float()  # [64] per-position beta

            # cuTile K4 does: ab = ak * be (multiply by per-column beta)
            # Fused kernel also multiplies by beta in Zone B write. Match that here.
            ab = (akk_c_raw * beta_c.unsqueeze(0)).to(torch.bfloat16).float()
            W_py = (
                (ab.to(torch.bfloat16).float() @ (-ks_c).to(torch.bfloat16).float())
                .to(torch.bfloat16)
                .float()
            )
            U_py = ab @ v_c
            S_bf16_py = S_py.to(torch.bfloat16).float()
            WS_py = (
                (W_py.to(torch.bfloat16).float() @ S_bf16_py).to(torch.bfloat16).float()
            )
            NV_py = (U_py - WS_py).to(torch.bfloat16).float()
            OI_py = qs_c @ S_bf16_py
            aqk_masked = (aqk_c * causal).to(torch.bfloat16).float()
            O_py_c_full = (OI_py + aqk_masked @ NV_py).to(torch.bfloat16)
            # When DEBUG_SKIP_MMA5: O = OI = QS @ S only (no AQC@NV correction)
            O_py_c = OI_py.to(torch.bfloat16) if DEBUG_SKIP_MMA5 else O_py_c_full
            O_fused_c = O_out[0, cs:ce, 0]
            d_py = (O_fused_c.float() - O_py_c.float()).abs().max().item()
            if verbose:
                print(f"      chunk {c}: fused_vs_pyref={d_py:.6f}", flush=True)
            O_py_chunks.append(O_py_c)
            SU_py = NV_py.T @ kg_c
            S_py = S_py * gk_c.unsqueeze(1) + SU_py.T

        O_py_cat = torch.cat(O_py_chunks, dim=0)
        O_fused_00 = O_out[0, :, 0]
        o_diff_py = (O_fused_00.float() - O_py_cat.float()).abs()
        o_md_py = o_diff_py.max().item()
        o_absmax = max(O_fused_00.float().abs().max().item(), 1.0)
        o_rel = o_md_py / o_absmax
        o_ok = o_md_py < 3.0 or o_rel < 0.15
        tag = "PASS" if o_ok else "INFO"
        print(
            f"    O(b=0,h=0) vs pyref (standalone K123): {tag} max_diff={o_md_py:.6f} rel={o_rel:.6f}"
        )

        s_opt_00 = S_fp32[0]
        s_diff_py = (s_opt_00 - S_py).abs()
        s_md_py = s_diff_py.max().item()
        s_absmax = max(s_opt_00.abs().max().item(), 1.0)
        s_rel = s_md_py / s_absmax
        s_ok = s_md_py < 0.1 or s_rel < 1e-3
        tag = "PASS" if s_ok else "INFO"
        print(
            f"    S(b=0,h=0) vs pyref (standalone K123): {tag} max_diff={s_md_py:.6f} rel={s_rel:.6f}"
        )

    # --- FLA comparison (all B, H) ---
    try:
        from fla.ops.kda.chunk_fwd import chunk_kda_fwd as fla_fwd
    except Exception as e:
        print(f"    FLA not available ({e}) — skipping FLA comparison")
        return ok

    # Use same random initial state as fused kernel
    # Try disable_recompute=True for intermediates, fall back to basic call
    w_fla = u_fla = kg_fla = v_new_fla = h_fla = None
    g_cumsum_fla = Aqk_fla = Akk_fla = None
    try:
        r_fla = fla_fwd(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            scale=scale,
            initial_state=S_init.clone(),
            output_final_state=True,
            chunk_size=BT,
            safe_gate=safe_gate,
            lower_bound=lower_bound if safe_gate else None,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
            disable_recompute=True,
        )
        o_fla, s_fla = r_fla[0], r_fla[1]
        g_cumsum_fla = r_fla[2]
        Aqk_fla = r_fla[3]
        Akk_fla = r_fla[4]
        w_fla = r_fla[5]
        u_fla = r_fla[6]
        kg_fla = r_fla[8]
        v_new_fla = r_fla[9]
        h_fla = r_fla[10]
        print(f"    FLA with intermediates: OK")
    except Exception as e:
        print(f"    FLA disable_recompute failed ({e}), using basic call")
        r_fla = fla_fwd(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            scale=scale,
            initial_state=S_init.clone(),
            output_final_state=True,
            chunk_size=BT,
            safe_gate=safe_gate,
            lower_bound=lower_bound if safe_gate else None,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
        )
        o_fla, s_fla = r_fla[0], r_fla[1]

    o_diff_fla = (O_out.float() - o_fla.float()).abs()
    o_md_fla = o_diff_fla.max().item()
    o_fla_absmax = max(o_fla.float().abs().max().item(), 1.0)
    o_fla_rel = o_md_fla / o_fla_absmax
    # bf16 MMA chain accumulation error grows with sequence length.
    # At T=8K, O max_diff vs FLA can reach ~0.25 absolute but stays < 1% relative.
    o_fla_ok = o_fla_rel < 0.01
    fla_tag = "PASS" if o_fla_ok else "FAIL"
    print(
        f"    O vs FLA (all B,H): {fla_tag} max_diff={o_md_fla:.6f} rel={o_fla_rel:.6f}"
    )
    if not o_fla_ok:
        ok = False

    # Per-chunk FLA comparison for b=0, h=0 (summary for large NC)
    T = NC * BT
    o_fused_bh = O_out[0, :, 0].float()
    o_fla_bh = o_fla[0, :, 0, :].float()
    chunk_diffs = []
    for c in range(NC):
        cs, ce = c * BT, (c + 1) * BT
        chunk_md = (o_fused_bh[cs:ce] - o_fla_bh[cs:ce]).abs().max().item()
        chunk_diffs.append(chunk_md)
    worst_chunk = max(range(NC), key=lambda c: chunk_diffs[c])
    print(
        f"    Per-chunk O vs FLA (b=0,h=0): worst=chunk {worst_chunk} "
        f"max_diff={chunk_diffs[worst_chunk]:.6f} "
        f"first={chunk_diffs[0]:.6f} last={chunk_diffs[-1]:.6f}"
    )

    s_opt = S_fp32.reshape(B, H, K_dim, V_dim)
    s_diff_fla = (s_opt - s_fla.float()).abs()
    s_md_fla = s_diff_fla.max().item()
    s_fla_absmax = max(s_fla.float().abs().max().item(), 1.0)
    s_fla_rel = s_md_fla / s_fla_absmax
    s_fla_ok = s_fla_rel < 0.01
    fla_s_tag = "PASS" if s_fla_ok else "FAIL"
    print(
        f"    S vs FLA (all B,H): {fla_s_tag} max_diff={s_md_fla:.6f} rel={s_fla_rel:.6f}"
    )
    if not s_fla_ok:
        ok = False

    # --- FLA intermediate comparison: per-chunk W, U, NV, O decomposition ---
    # (Skip for large NC — too verbose)
    if w_fla is not None and h_fla is not None and k_sc_ref is not None and NC <= 8:
        print(f"    --- FLA intermediate comparison (b=0, h=0) ---")
        # h_fla[b, c, h, K, V] = state BEFORE chunk c
        # w_fla[b, t, h, K] = W (per-timestep within chunk), u_fla[b, t, h, V] = U
        # v_new_fla[b, t, h, V] = NV = u - w@S

        causal = torch.tril(torch.ones(BT, BT, device=dev))
        S_fla_running = S_init[0, 0].float()  # track pyref state for comparison

        for c in range(NC):
            cs, ce = c * BT, (c + 1) * BT
            # FLA intermediates for this chunk
            w_fla_c = w_fla[0, cs:ce, 0].float()  # [BT, K] — FLA's W
            u_fla_c = u_fla[0, cs:ce, 0].float()  # [BT, V] — FLA's U
            nv_fla_c = v_new_fla[0, cs:ce, 0].float()  # [BT, V] — FLA's NV
            h_fla_c = h_fla[0, c, 0].float()  # [K, V] — state before chunk c

            # Our pyref K4 intermediates using standalone K123 outputs
            ks_c = k_sc_ref[0, cs:ce, 0].float()
            qs_c = q_sc_ref[0, cs:ce, 0].float()
            kg_c = kg_ref[0, cs:ce, 0].float()
            v_c = v[0, cs:ce, 0].float()
            akk_c = akk_ref[0, cs:ce, 0].float()
            beta_c = beta[0, cs:ce, 0].float()
            aqk_c = aqk_ref[0, cs:ce, 0].float()
            ab_py = (akk_c * beta_c.unsqueeze(0)).to(torch.bfloat16).float()

            # ---- Test A: O computed with FLA's ground-truth state h_fla[c] ----
            # This tells us: if the fused kernel had perfect state, would O be correct?
            h_bf16 = h_fla_c.to(torch.bfloat16).float()
            W_from_fla = (
                (ab_py.to(torch.bfloat16) @ (-ks_c).to(torch.bfloat16))
                .float()
                .to(torch.bfloat16)
                .float()
            )
            U_from_fla = (
                (ab_py.to(torch.bfloat16) @ v_c.to(torch.bfloat16))
                .float()
                .to(torch.bfloat16)
                .float()
            )
            WS_from_fla = (
                (W_from_fla.to(torch.bfloat16) @ h_bf16.to(torch.bfloat16))
                .float()
                .to(torch.bfloat16)
                .float()
            )
            NV_from_fla = (U_from_fla - WS_from_fla).to(torch.bfloat16).float()
            OI_from_fla = (
                (qs_c.to(torch.bfloat16) @ h_bf16.to(torch.bfloat16))
                .float()
                .to(torch.bfloat16)
                .float()
            )
            aqk_masked = (aqk_c * causal).to(torch.bfloat16).float()
            O_ideal = (
                OI_from_fla
                + (aqk_masked.to(torch.bfloat16) @ NV_from_fla.to(torch.bfloat16))
                .float()
                .to(torch.bfloat16)
                .float()
            ).to(torch.bfloat16)

            o_fused_c = O_out[0, cs:ce, 0].float()
            o_fla_c = o_fla[0, cs:ce, 0, :].float()
            o_ideal_c = O_ideal.float()

            fused_vs_fla = (o_fused_c - o_fla_c).abs().max().item()
            ideal_vs_fla = (o_ideal_c - o_fla_c).abs().max().item()
            fused_vs_ideal = (o_fused_c - o_ideal_c).abs().max().item()

            # ---- Test B: Compare W, U with FLA ----
            w_diff = (W_from_fla - w_fla_c).abs().max().item()
            u_diff = (U_from_fla - u_fla_c).abs().max().item()
            nv_diff = (NV_from_fla - nv_fla_c).abs().max().item()

            # ---- Test C: State check ----
            h_diff = (S_fla_running - h_fla_c).abs().max().item()

            print(f"      chunk {c}:")
            print(
                f"        state: pyref_vs_fla={h_diff:.6f} |h_fla|={h_fla_c.abs().max().item():.4f}"
            )
            print(
                f"        W pyref_vs_fla={w_diff:.6f} U pyref_vs_fla={u_diff:.6f} NV pyref_vs_fla={nv_diff:.6f}"
            )
            print(
                f"        O: fused_vs_fla={fused_vs_fla:.6f} ideal_vs_fla={ideal_vs_fla:.6f} fused_vs_ideal={fused_vs_ideal:.6f}"
            )
            if fused_vs_ideal > 0.01:
                print(
                    f"        >>> fused_vs_ideal={fused_vs_ideal:.6f} (state-dependent error, see separate path comparison)"
                )

            # Update pyref state for next chunk (mirror fused kernel's state update)
            SU_py = (NV_from_fla.to(torch.bfloat16).T @ kg_c.to(torch.bfloat16)).float()
            gk_last_c = gk_ref[0, c, 0].float()  # [K] — gk_last for this chunk
            S_fla_running = S_fla_running * gk_last_c.unsqueeze(1) + SU_py.T

    # --- Separate path comparison (K123 + cuTile K4, same random S_init) ---
    print(f"    --- Separate path comparison (same random S_init) ---")
    try:
        from chunk_fwd import chunk_kda_fwd as opt_fwd

        r_sep = opt_fwd(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            scale=scale,
            initial_state=S_init.clone(),
            output_final_state=True,
            chunk_size=BT,
            safe_gate=safe_gate,
            lower_bound=lower_bound if safe_gate else None,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
        )
        o_sep = r_sep[0]  # [B, T, H, V]
        s_sep = r_sep[1]
        o_fused_vs_sep = (O_out.float() - o_sep.float()).abs().max().item()
        s_fused_vs_sep = (
            (S_fp32.reshape(B, H, K_dim, V_dim).float() - s_sep.float())
            .abs()
            .max()
            .item()
        )
        o_sep_vs_fla = (o_sep.float() - o_fla.float()).abs().max().item()
        s_sep_vs_fla = (s_sep.float() - s_fla.float()).abs().max().item()
        # Key metric: fused should match separate path (both use same bf16 MMA chain)
        o_sep_ok = o_fused_vs_sep <= o_sep_vs_fla * 2.0 + 0.01
        s_sep_ok = s_fused_vs_sep <= s_sep_vs_fla * 2.0 + 0.01
        sep_tag = "PASS" if (o_sep_ok and s_sep_ok) else "FAIL"
        print(
            f"      O fused_vs_sep={o_fused_vs_sep:.6f}  sep_vs_fla={o_sep_vs_fla:.6f}  {sep_tag}"
        )
        print(
            f"      S fused_vs_sep={s_fused_vs_sep:.6f}  sep_vs_fla={s_sep_vs_fla:.6f}"
        )
        if not (o_sep_ok and s_sep_ok):
            ok = False
    except Exception as e:
        print(f"      Separate path failed: {e}")

    return ok


def test():
    results = {}

    # Shape configurations: (NC, B, H, K, V, safe_gate, use_dt_bias, label)
    # (NC, B, H, K, V, safe_gate, use_dt_bias, use_zero_state, label)
    configs = [
        # Small: quick sanity
        (2, 1, 1, 128, 128, True, True, False, "nc2_B1H1"),
        # T=8192 (NC=128), various batch/head combos
        (128, 1, 1, 128, 128, True, True, False, "nc128_B1H1"),
        (128, 2, 1, 128, 128, True, True, False, "nc128_B2H1"),
        (128, 1, 4, 128, 128, True, True, False, "nc128_B1H4"),
        (128, 4, 1, 128, 128, True, True, False, "nc128_B4H1"),
        (128, 2, 4, 128, 128, True, True, False, "nc128_B2H4"),
    ]

    for NC, B, H, K_dim, V_dim, safe_gate, use_dt_bias, zero_state, label in configs:
        results[label] = _run_test(
            NC,
            B,
            H,
            K_dim,
            V_dim,
            safe_gate=safe_gate,
            use_dt_bias=use_dt_bias,
            use_zero_state=zero_state,
        )

    all_ok = all(results.values())
    print()
    for name, ok in results.items():
        print(f"  {name:<35s} {'PASS' if ok else 'FAIL'}")
    return all_ok


def _bench_config(NC, B, H, K_dim, V_dim, safe_gate=True, warmup=10, iters=100):
    """Benchmark fused K1234 for a single config. Returns ms/iter."""
    T = NC * BT
    dev = "cuda"
    torch.manual_seed(42)

    q = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, T, H, V_dim, dtype=torch.bfloat16, device=dev)
    g = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=dev) * 0.1
    beta = torch.rand(B, T, H, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.5
    dt_bias = torch.randn(H * K_dim, dtype=torch.float32, device=dev) * 0.1
    scale = float(K_dim**-0.5)
    lower_bound = -5.0

    BH = B * H
    S_init = torch.randn(B, H, K_dim, V_dim, dtype=torch.float32, device=dev) * 0.1
    S_fp32 = S_init.reshape(BH, K_dim, V_dim).clone()
    O_out = torch.zeros(B, T, H, V_dim, dtype=torch.bfloat16, device=dev)

    bf = cutlass.BFloat16
    fp = cutlass.Float32
    bias_ct = _ct(dt_bias.float().contiguous().view(H, K_dim), fp)
    lb_val = float(lower_bound)

    q_ct = _ct(q, bf)
    k_ct = _ct(k, bf)
    g_ct = _ct(g, bf)
    alog_ct = _ct(A_log, fp)
    beta_ct = _ct(beta, bf)
    v_ct = _ct(v, bf)
    o_ct = _ct(O_out, bf)
    sfp_ct = _ct(S_fp32, fp)

    clocks = torch.zeros(16, BH, dtype=torch.int64, device=dev)
    clocks_ct = _ct(clocks)
    clocks_ct.element_type = cutlass.Int64

    host_fn = make_host_fn(has_bias=True, use_safe_gate=safe_gate)
    ct_args = (
        q_ct,
        k_ct,
        g_ct,
        alog_ct,
        beta_ct,
        scale,
        v_ct,
        o_ct,
        sfp_ct,
        bias_ct,
        lb_val,
        clocks_ct,
        NC,
        H,
        B,
        cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    compiled = cutlass.cute.compile(host_fn, *ct_args)

    # Warmup
    for _ in range(warmup):
        S_fp32.copy_(S_init.reshape(BH, K_dim, V_dim))
        compiled(*ct_args)
    torch.cuda.synchronize()

    # Timed
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        S_fp32.copy_(S_init.reshape(BH, K_dim, V_dim))
        compiled(*ct_args)
    end_ev.record()
    torch.cuda.synchronize()
    ms = start_ev.elapsed_time(end_ev) / iters
    return ms


def _bench_separate(NC, B, H, K_dim, V_dim, safe_gate=True, warmup=10, iters=100):
    """Benchmark separate K123+cuTile K4 path. Returns ms/iter or None if unavailable."""
    try:
        from chunk_fwd import chunk_kda_fwd as opt_fwd
    except Exception:
        return None

    T = NC * BT
    dev = "cuda"
    torch.manual_seed(42)

    q = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, T, H, V_dim, dtype=torch.bfloat16, device=dev)
    g = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=dev) * 0.1
    beta = torch.rand(B, T, H, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.5
    dt_bias = torch.randn(H * K_dim, dtype=torch.float32, device=dev) * 0.1
    scale = float(K_dim**-0.5)
    lower_bound = -5.0
    S_init = torch.randn(B, H, K_dim, V_dim, dtype=torch.float32, device=dev) * 0.1

    def run():
        opt_fwd(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            scale=scale,
            initial_state=S_init.clone(),
            output_final_state=True,
            chunk_size=BT,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        run()
    end_ev.record()
    torch.cuda.synchronize()
    return start_ev.elapsed_time(end_ev) / iters


def _bench_fla(NC, B, H, K_dim, V_dim, safe_gate=True, warmup=10, iters=100):
    """Benchmark FLA reference. Returns ms/iter or None if unavailable."""
    try:
        from fla.ops.kda.chunk_fwd import chunk_kda_fwd as fla_fwd
    except Exception:
        return None

    T = NC * BT
    dev = "cuda"
    torch.manual_seed(42)

    q = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, T, H, V_dim, dtype=torch.bfloat16, device=dev)
    g = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=dev) * 0.1
    beta = torch.rand(B, T, H, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.5
    dt_bias = torch.randn(H * K_dim, dtype=torch.float32, device=dev) * 0.1
    scale = float(K_dim**-0.5)
    lower_bound = -5.0
    S_init = torch.randn(B, H, K_dim, V_dim, dtype=torch.float32, device=dev) * 0.1

    def run():
        fla_fwd(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            scale=scale,
            initial_state=S_init.clone(),
            output_final_state=True,
            chunk_size=BT,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        run()
    end_ev.record()
    torch.cuda.synchronize()
    return start_ev.elapsed_time(end_ev) / iters


def benchmark():
    """Benchmark fused K1234 vs separate path vs FLA."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)

    # Benchmark configs: (NC, B, H, K, V, label)
    configs = [
        (128, 1, 1, 128, 128, "B1_H1_T8192"),
        (128, 2, 1, 128, 128, "B2_H1_T8192"),
        (128, 1, 4, 128, 128, "B1_H4_T8192"),
        (128, 4, 1, 128, 128, "B4_H1_T8192"),
        (128, 2, 4, 128, 128, "B2_H4_T8192"),
        (128, 1, 48, 128, 128, "B1_H48_T8192"),
        (128, 4, 48, 128, 128, "B4_H48_T8192"),
        (128, 1, 96, 128, 128, "B1_H96_T8192"),
        (128, 4, 96, 128, 128, "B4_H96_T8192"),
    ]

    warmup, iters = 10, 50

    print(
        f"\n{'Config':<20s} {'Fused':>10s} {'Separate':>10s} {'FLA':>10s} {'Fused/Sep':>10s} {'Fused/FLA':>10s}"
    )
    print("-" * 70)

    for NC, B, H, K_dim, V_dim, label in configs:
        T = NC * BT
        print(f"  Compiling {label} (B={B}, H={H}, T={T})...", flush=True)

        ms_fused = _bench_config(NC, B, H, K_dim, V_dim, warmup=warmup, iters=iters)
        ms_sep = _bench_separate(NC, B, H, K_dim, V_dim, warmup=warmup, iters=iters)
        ms_fla = _bench_fla(NC, B, H, K_dim, V_dim, warmup=warmup, iters=iters)

        sep_str = f"{ms_sep:.3f}ms" if ms_sep is not None else "N/A"
        fla_str = f"{ms_fla:.3f}ms" if ms_fla is not None else "N/A"
        ratio_sep = f"{ms_fused/ms_sep:.3f}x" if ms_sep is not None else "N/A"
        ratio_fla = f"{ms_fused/ms_fla:.3f}x" if ms_fla is not None else "N/A"

        print(
            f"  {label:<20s} {ms_fused:>9.3f}ms {sep_str:>10s} {fla_str:>10s} {ratio_sep:>10s} {ratio_fla:>10s}"
        )

    print()


def profile_clock():
    """Profile K123 vs K4 time using in-kernel globaltimer clock reads."""
    NC, B, H = 128, 1, 96
    K_dim, V_dim = 128, 128
    T = NC * BT
    BH = B * H
    dev = "cuda"
    safe_gate = True
    lower_bound = -5.0

    print(f"\n{'='*60}")
    print(f"CLOCK PROFILING: B={B}, H={H}, NC={NC}, T={T}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, T, H, V_dim, dtype=torch.bfloat16, device=dev)
    g = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=dev) * 0.1
    beta = torch.rand(B, T, H, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.5
    dt_bias = torch.randn(H * K_dim, dtype=torch.float32, device=dev) * 0.1
    scale = float(K_dim**-0.5)
    S_init = torch.randn(B, H, K_dim, V_dim, dtype=torch.float32, device=dev) * 0.1
    S_fp32 = S_init.reshape(BH, K_dim, V_dim).clone()
    O_out = torch.zeros(B, T, H, V_dim, dtype=torch.bfloat16, device=dev)

    bf = cutlass.BFloat16
    fp = cutlass.Float32
    bias_ct = _ct(dt_bias.float().contiguous().view(H, K_dim), fp)
    lb_val = float(lower_bound)
    q_ct = _ct(q, bf)
    k_ct = _ct(k, bf)
    g_ct = _ct(g, bf)
    alog_ct = _ct(A_log, fp)
    beta_ct = _ct(beta, bf)
    v_ct = _ct(v, bf)
    o_ct = _ct(O_out, bf)
    sfp_ct = _ct(S_fp32, fp)

    clocks = torch.zeros(16, BH, dtype=torch.int64, device=dev)
    clocks_ct = _ct(clocks)
    clocks_ct.element_type = cutlass.Int64

    print("  Compiling with profile_clocks=True...", flush=True)
    host_fn = make_host_fn(has_bias=True, use_safe_gate=safe_gate, profile_clocks=True)
    ct_args = (
        q_ct,
        k_ct,
        g_ct,
        alog_ct,
        beta_ct,
        scale,
        v_ct,
        o_ct,
        sfp_ct,
        bias_ct,
        lb_val,
        clocks_ct,
        NC,
        H,
        B,
        cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    compiled = cutlass.cute.compile(host_fn, *ct_args)

    # Warmup
    for _ in range(5):
        S_fp32.copy_(S_init.reshape(BH, K_dim, V_dim))
        clocks.zero_()
        compiled(*ct_args)
    torch.cuda.synchronize()

    # Collect over multiple runs
    n_runs = 10
    acc = torch.zeros(16, dtype=torch.float64)
    for _ in range(n_runs):
        S_fp32.copy_(S_init.reshape(BH, K_dim, V_dim))
        clocks.zero_()
        compiled(*ct_args)
        torch.cuda.synchronize()
        for i in range(16):
            acc[i] += clocks[i].float().mean().item()

    avg = acc / n_runs  # ns per slot, averaged over blocks and runs

    k123_ns = avg[0].item()
    k4_ns = avg[1].item()
    total_ns = k123_ns + k4_ns

    # K4 MMA sub-steps (slots 2-7)
    k4_labels = [
        "MMA1 (W=AB@KS)",
        "MMA2 (U=AB@V)",
        "wait_S + MMA4 (OI=Q@S)",
        "wait_W + MMA3 (NV=U+W@S)",
        "wait_NV + MMA5 (O+=AQC@NV)",
        "MMA6 + drain + sync",
    ]
    k4_vals = [avg[i + 2].item() for i in range(6)]
    k4_sum = sum(k4_vals)

    # Per-role wall-clock (slots 8-12)
    role_labels = [
        "WG1 (readout+decay)",
        "K1 (TMA+gate+cumsum)",
        "K2 (intra MMA)",
        "K3 (inv+store)",
        "K4 TMA (V load+O store)",
    ]
    role_vals = [avg[i + 8].item() for i in range(5)]

    print(f"\n  {'='*72}")
    print(f"  PER-ROLE WALL-CLOCK (start-to-end, includes waits)")
    print(f"  {'='*72}")
    print(f"  {'Role':<30s} {'Time':>10s} {'us/chk':>8s} {'%':>7s}")
    print(f"  {'-'*55}")
    max_role = max(role_vals) if role_vals else 1
    for label, val in zip(role_labels, role_vals):
        pct = val / max_role * 100 if max_role > 0 else 0
        print(f"  {label:<30s} {val/1e6:>9.3f}ms {val/NC/1e3:>7.1f} {pct:>6.1f}%")
    print(f"  {'-'*55}")
    print(
        f"  {'Bottleneck (longest role)':<30s} {max_role/1e6:>9.3f}ms {max_role/NC/1e3:>7.1f}"
    )

    print(f"\n  {'='*72}")
    print(f"  K4 MMA WARP BREAKDOWN (from W0 perspective)")
    print(f"  {'='*72}")
    print(f"  {'Phase':<35s} {'Time':>10s} {'us/chk':>8s} {'%K4':>7s} {'%Tot':>7s}")
    print(f"  {'-'*67}")
    print(
        f"  {'K123 wait':<35s} {k123_ns/1e6:>9.3f}ms {k123_ns/NC/1e3:>7.1f} {'':>7s} {k123_ns/total_ns*100:>6.1f}%"
    )
    print(
        f"  {'K4 (total from barrier)':<35s} {k4_ns/1e6:>9.3f}ms {k4_ns/NC/1e3:>7.1f} {'100.0%':>7s} {k4_ns/total_ns*100:>6.1f}%"
    )
    print(f"  {'-'*67}")
    for label, val in zip(k4_labels, k4_vals):
        pct_k4 = val / k4_sum * 100 if k4_sum > 0 else 0
        pct_tot = val / total_ns * 100 if total_ns > 0 else 0
        print(
            f"    {label:<33s} {val/1e6:>9.3f}ms {val/NC/1e3:>7.1f} {pct_k4:>6.1f}% {pct_tot:>6.1f}%"
        )
    print(f"    {'(MMA warp sum)':<33s} {k4_sum/1e6:>9.3f}ms {k4_sum/NC/1e3:>7.1f}")
    print(f"  {'-'*67}")
    print(f"  {'Total (clock)':<35s} {total_ns/1e6:>9.3f}ms")
    print()


def _sass_analysis():
    """Dump SASS and resource usage from the latest cubin."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cubin_files = glob.glob(os.path.join(script_dir, "*.cubin"))
    if not cubin_files:
        cubin_files = glob.glob("*.cubin")
    if not cubin_files:
        print("  No cubin file found")
        return
    cubin_file = max(cubin_files, key=os.path.getmtime)
    print(f"  Analyzing: {os.path.basename(cubin_file)}")

    cuobjdump = "cuobjdump"
    result = subprocess.run(
        [cuobjdump, "-res-usage", cubin_file], capture_output=True, text=True
    )
    if result.returncode == 0:
        for line in result.stdout.split("\n"):
            if any(kw in line.lower() for kw in ["reg", "smem", "stack", "cmem"]):
                print(f"  {line.strip()}")

    result = subprocess.run(
        [cuobjdump, "-sass", cubin_file], capture_output=True, text=True
    )
    if result.returncode == 0:
        stl = result.stdout.count("STL")
        ldl = result.stdout.count("LDL")
        print(f"  SASS: STL(spill store)={stl}, LDL(spill load)={ldl}")
    else:
        print(f"  cuobjdump -sass failed: {result.stderr[:200]}")


def spill_check():
    """Compile kernel with KeepCUBIN and analyze SASS for register spills."""
    print("\n--- Register Spill Analysis ---")
    NC, B, H = 128, 1, 96
    K_dim = V_dim = 128
    T = NC * BT
    BH = B * H
    dev = "cuda"
    torch.manual_seed(42)

    q = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, T, H, V_dim, dtype=torch.bfloat16, device=dev)
    g = torch.randn(B, T, H, K_dim, dtype=torch.bfloat16, device=dev) * 0.1
    beta = torch.rand(B, T, H, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.5
    dt_bias = torch.randn(H * K_dim, dtype=torch.float32, device=dev) * 0.1
    scale = float(K_dim**-0.5)
    lower_bound = -5.0
    S_init = torch.zeros(BH, K_dim, V_dim, dtype=torch.float32, device=dev)
    O_out = torch.zeros(B, T, H, V_dim, dtype=torch.bfloat16, device=dev)

    bf = cutlass.BFloat16
    fp = cutlass.Float32
    bias_ct = _ct(dt_bias.float().contiguous().view(H, K_dim), fp)
    clocks = torch.zeros(16, BH, dtype=torch.int64, device=dev)
    clocks_ct = _ct(clocks)
    clocks_ct.element_type = cutlass.Int64

    print("  Compiling with KeepPTX + KeepCUBIN...", flush=True)
    host_fn = make_host_fn(has_bias=True, use_safe_gate=True)
    ct_args = (
        _ct(q, bf),
        _ct(k, bf),
        _ct(g, bf),
        _ct(A_log, fp),
        _ct(beta, bf),
        scale,
        _ct(v, bf),
        _ct(O_out, bf),
        _ct(S_init, fp),
        bias_ct,
        float(lower_bound),
        clocks_ct,
        NC,
        H,
        B,
        cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    compiled = cutlass.cute.compile[KeepPTX, KeepCUBIN](host_fn, *ct_args)
    compiled(*ct_args)
    torch.cuda.synchronize()
    _sass_analysis()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-only", action="store_true", help="Skip correctness, run benchmark only"
    )
    parser.add_argument(
        "--no-bench", action="store_true", help="Skip benchmark, run correctness only"
    )
    parser.add_argument(
        "--clock",
        action="store_true",
        help="Profile K123/K4 time split using in-kernel clocks",
    )
    parser.add_argument(
        "--spill", action="store_true", help="Analyze register spills (STL/LDL in SASS)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Fused K1234 Kernel (SMEM optimized)")
    print("=" * 60)

    if args.clock:
        profile_clock()
        return 0

    if args.spill:
        spill_check()
        return 0

    ok = True
    if not args.bench_only:
        ok = test()
        print()
        print("ALL PASS" if ok else "FAILED")

    if not args.no_bench:
        benchmark()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
