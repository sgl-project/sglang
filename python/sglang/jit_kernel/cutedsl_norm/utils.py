"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Shared CuTe DSL Utilities for Norm Kernels
==========================================

Common utilities used by all norm kernels:
- Constants for vectorization and FP8 quantization
- PTX intrinsics for fast reciprocal and FP8 conversion
- Warp and block reduction utilities
- Predicate helpers for bounds checking
- Layout configuration helpers
- Type conversion utilities
"""

import functools
import math
import operator
from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


def get_cutlass_dtype(dtype: str) -> "cutlass.dtype":
    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float32": cutlass.Float32,
        "float8_e5m2": cutlass.Float8E5M2,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
        "float8_e8m0fnu": cutlass.Float8E8M0FNU,
        "float4_e2m1fn": cutlass.Float4E2M1FN,
    }
    return dtype_map[dtype]


def get_num_sm(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


# =============================================================================
# Constants
# =============================================================================

FLOAT8_E4M3_MAX = 448.0  # Maximum value representable in FP8 E4M3
COPY_BITS = 128  # 128-bit vectorized loads


# =============================================================================
# PTX Intrinsics
# =============================================================================


@dsl_user_op
def rcp_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """Fast reciprocal using PTX rcp.approx.ftz.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cvt_and_store_f32_to_e4m3_hw(val: Float32, addr: Int64, *, loc=None, ip=None):
    """Convert float32 to E4M3 and store single byte — hardware path (sm_89+).

    Uses the cvt.rn.satfinite.e4m3x2.f32 PTX instruction for maximum performance.
    """
    llvm.inline_asm(
        None,
        [Float32(val).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 fp8_pair;
            .reg .f32 zero;
            mov.f32 zero, 0f00000000;
            cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $0;
            st.global.b8 [$1], fp8_pair;
        }
        """,
        "f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cvt_and_store_f32_to_e4m3_sw(val: Float32, addr: Int64, *, loc=None, ip=None):
    """Convert float32 to E4M3 and store single byte — software path (all architectures).

    Uses integer bit manipulation mirroring NVIDIA's __nv_cvt_float_to_fp8 from cuda_fp8.hpp.
    The caller must clamp the value to [-448, 448] before calling this function.

    E4M3 format: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits.
    Conversion strategy (operates on f32 bit representation):
      - Normal range (f32 biased exp >= 121): direct exponent/mantissa extraction with RNE
      - Denormal range (f32 biased exp in [118..120]): shift mantissa with implicit bit, RNE
      - Underflow (abs <= 2^-10): flush to zero (2^-10 is the RNE midpoint to min denorm)
    """
    llvm.inline_asm(
        None,
        [Float32(val).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 fbits, sign8, abs_bits, f32_exp, f32_mant;
            .reg .b32 e4m3_exp, e4m3_mant, norm_raw;
            .reg .b32 rbit, sticky, odd_bit, radj;
            .reg .b32 shift, dmant4, denorm_raw;
            .reg .b32 dr_bit, dsticky, dadj;
            .reg .b32 e4m3_raw, tmp, tmp2;
            .reg .pred p_zero, p_denorm;

            // Bitcast float to int and extract sign/exponent/mantissa
            mov.b32 fbits, $0;
            shr.u32 sign8, fbits, 24;
            and.b32 sign8, sign8, 128;
            and.b32 abs_bits, fbits, 0x7FFFFFFF;
            shr.u32 f32_exp, abs_bits, 23;
            and.b32 f32_mant, abs_bits, 0x007FFFFF;

            // === Normal path (f32 biased exp >= 121, i.e. e4m3 exp >= 1) ===
            sub.u32 e4m3_exp, f32_exp, 120;
            shr.u32 e4m3_mant, f32_mant, 20;
            shl.b32 norm_raw, e4m3_exp, 3;
            or.b32  norm_raw, norm_raw, e4m3_mant;

            // Round-to-nearest-even: round up if round_bit AND (sticky OR odd)
            shr.u32 rbit, f32_mant, 19;
            and.b32 rbit, rbit, 1;
            and.b32 sticky, f32_mant, 0x0007FFFF;
            and.b32 odd_bit, e4m3_mant, 1;
            or.b32  tmp, sticky, odd_bit;
            min.u32 tmp, tmp, 1;
            and.b32 radj, tmp, rbit;
            add.u32 norm_raw, norm_raw, radj;
            min.u32 norm_raw, norm_raw, 126;

            // === Denormal path (f32 biased exp in {118,119,120}) ===
            sub.u32 shift, 121, f32_exp;
            shr.u32 tmp, f32_mant, 20;
            or.b32  dmant4, tmp, 8;
            shr.u32 denorm_raw, dmant4, shift;

            // RNE rounding for denormals
            sub.u32 tmp, shift, 1;
            shr.u32 dr_bit, dmant4, tmp;
            and.b32 dr_bit, dr_bit, 1;
            shl.b32 tmp2, 1, tmp;
            sub.u32 tmp2, tmp2, 1;
            and.b32 dsticky, dmant4, tmp2;
            and.b32 tmp, f32_mant, 0x000FFFFF;
            or.b32  dsticky, dsticky, tmp;
            and.b32 odd_bit, denorm_raw, 1;
            or.b32  tmp, dsticky, odd_bit;
            min.u32 tmp, tmp, 1;
            and.b32 dadj, tmp, dr_bit;
            add.u32 denorm_raw, denorm_raw, dadj;

            // Select between normal/denormal, then apply zero flush
            setp.le.u32 p_denorm, f32_exp, 120;
            selp.u32 e4m3_raw, denorm_raw, norm_raw, p_denorm;
            setp.le.u32 p_zero, abs_bits, 0x3A800000;
            selp.u32 e4m3_raw, 0, e4m3_raw, p_zero;

            // Apply sign and store single byte
            or.b32  e4m3_raw, e4m3_raw, sign8;
            st.global.b8 [$1], e4m3_raw;
        }
        """,
        "f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cvt_and_store_8xf32_to_e4m3_hw(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    addr: Int64,
    *,
    loc=None,
    ip=None,
):
    """Convert 8 float32 values to E4M3 and store as one 64-bit global store (sm_89+).

    Uses cvt.rn.satfinite.e4m3x2.f32 to convert pairs, then packs into two b32
    words and issues a single st.global.v2.b32.  ~4x fewer instructions and ~8x
    fewer store transactions compared to 8 scalar st.global.b8 calls.
    """
    llvm.inline_asm(
        None,
        [
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
            Float32(v4).ir_value(loc=loc, ip=ip),
            Float32(v5).ir_value(loc=loc, ip=ip),
            Float32(v6).ir_value(loc=loc, ip=ip),
            Float32(v7).ir_value(loc=loc, ip=ip),
            Int64(addr).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b16 p01, p23, p45, p67;
            .reg .b32 lo, hi;
            cvt.rn.satfinite.e4m3x2.f32 p01, $1, $0;
            cvt.rn.satfinite.e4m3x2.f32 p23, $3, $2;
            cvt.rn.satfinite.e4m3x2.f32 p45, $5, $4;
            cvt.rn.satfinite.e4m3x2.f32 p67, $7, $6;
            mov.b32 lo, {p01, p23};
            mov.b32 hi, {p45, p67};
            st.global.v2.b32 [$8], {lo, hi};
        }
        """,
        "f,f,f,f,f,f,f,f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cvt_and_store_4xf32_to_e4m3_hw(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    addr: Int64,
    *,
    loc=None,
    ip=None,
):
    """Convert 4 float32 values to E4M3 and store as one 32-bit global store (sm_89+)."""
    llvm.inline_asm(
        None,
        [
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
            Int64(addr).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b16 p01, p23;
            .reg .b32 packed;
            cvt.rn.satfinite.e4m3x2.f32 p01, $1, $0;
            cvt.rn.satfinite.e4m3x2.f32 p23, $3, $2;
            mov.b32 packed, {p01, p23};
            st.global.b32 [$4], packed;
        }
        """,
        "f,f,f,f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cvt_and_store_2xf32_to_e4m3_hw(
    v0: Float32, v1: Float32, addr: Int64, *, loc=None, ip=None
):
    """Convert 2 float32 values to E4M3 and store as one 16-bit global store (sm_89+)."""
    llvm.inline_asm(
        None,
        [
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Int64(addr).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b16 packed;
            cvt.rn.satfinite.e4m3x2.f32 packed, $1, $0;
            st.global.b16 [$2], packed;
        }
        """,
        "f,f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def has_hw_fp8_cvt(device: torch.device = None) -> bool:
    """Check if the device supports hardware FP8 conversion (sm_89+)."""
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    return major > 8 or (major == 8 and minor >= 9)


@functools.lru_cache(maxsize=16)
def get_sm_version(device=None) -> int:
    """Get the SM version of a CUDA device (e.g., 100 for SM100)."""
    if not torch.cuda.is_available():
        return 80
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.major * 10 + props.minor


@dsl_user_op
def get_ptr_as_int64(tensor: cute.Tensor, offset: Int64, *, loc=None, ip=None) -> Int64:
    """Get the memory address of tensor[offset] as Int64.

    ``offset`` is a flat element index and is accepted as Int64 so it can
    address tensors whose flat element count exceeds INT32_MAX.
    """
    elem_ptr = tensor.iterator + Int64(offset)
    ptr_int = llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return Int64(ptr_int)


# =============================================================================
# PTX Intrinsics - Cluster Operations (SM90+)
# =============================================================================


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:
    """Map smem pointer to address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote(
    val: Float32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store Float32 value to shared memory on a remote CTA in the cluster."""
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];",
        "r,f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord, *, loc=None, ip=None) -> cute.Pointer:
    """Get pointer to element at coordinate in tensor."""
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


# =============================================================================
# Warp and Block Reduction Utilities
# =============================================================================


@cute.jit
def warp_reduce(val, op, width: cutlass.Constexpr[int] = 32):
    """Reduce across threads in a warp using butterfly shuffle."""
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
        return val


@cute.jit
def block_reduce(
    val: Float32,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: Float32,
) -> Float32:
    """Block reduction across multiple warps using shared memory."""
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    num_warps = cute.size(reduction_buffer.shape)

    if lane_idx == 0:
        reduction_buffer[warp_idx] = val
    cute.arch.barrier()

    block_reduce_val = init_val
    if lane_idx < num_warps:
        block_reduce_val = reduction_buffer[lane_idx]
    return warp_reduce(block_reduce_val, op)


@cute.jit
def row_reduce_sum(
    x: cute.TensorSSA,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor,
) -> Float32:
    """Row reduction for sum operation."""
    local_val = x.reduce(
        cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
    )

    warp_width = min(threads_per_row, 32)
    warp_val = warp_reduce(local_val, operator.add, width=warp_width)

    warps_per_row = max(threads_per_row // 32, 1)

    if cutlass.const_expr(warps_per_row > 1):
        return block_reduce(warp_val, operator.add, reduction_buffer, Float32(0.0))
    else:
        return warp_val


@cute.jit
def block_reduce_multirow(
    val: Float32,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: Float32,
) -> Float32:
    """Block reduction with 2D buffer (rows_per_block, warps_per_row).

    Each warp writes its partial sum to the row it belongs to, then
    lane 0..warps_per_row-1 read back and do a final warp reduction.
    """
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx = warp_idx // warps_per_row
    col_idx = warp_idx % warps_per_row

    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()

    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return warp_reduce(block_reduce_val, op)


@cute.jit
def cluster_reduce_multirow(
    val: Float32,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr,
    cluster_n: cutlass.Constexpr[int],
    init_val: Float32,
) -> Float32:
    """Cluster reduction across multiple CTAs using mbarrier.

    reduction_buffer has shape (rows_per_block, (warps_per_row, cluster_n)).
    Each warp sends its partial result to all CTAs in the cluster via
    st.async.shared::cluster, then every CTA reduces the collected values.
    """
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()

    rows_per_block = reduction_buffer.shape[0]
    warps_per_row = reduction_buffer.shape[1][0]

    row_idx = warp_idx // warps_per_row
    col_idx = warp_idx % warps_per_row

    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            expected_bytes = num_warps * cluster_n * 4
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, expected_bytes)

    if lane_idx < cluster_n:
        store_shared_remote(
            val,
            elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
            mbar_ptr,
            peer_cta_rank_in_cluster=lane_idx,
        )

    cute.arch.mbarrier_wait(mbar_ptr, phase=0)

    num_total = warps_per_row * cluster_n
    num_iter = cute.ceil_div(num_total, 32)

    block_reduce_val = init_val
    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * 32
        if idx < num_total:
            block_reduce_val = op(block_reduce_val, reduction_buffer[row_idx, idx])

    return warp_reduce(block_reduce_val, op)


@cute.jit
def row_reduce_sum_multirow(
    x: cute.TensorSSA,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor,
    mbar_ptr,
    cluster_n: cutlass.Constexpr[int],
) -> Float32:
    """Row reduction for sum with optional cluster support.

    When cluster_n == 1, uses block-level reduction with 2D buffer
    (rows_per_block, warps_per_row). When cluster_n > 1, uses cross-CTA
    cluster reduction with hierarchical buffer
    (rows_per_block, (warps_per_row, cluster_n)).
    """
    local_val = x.reduce(
        cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
    )

    warp_width = min(threads_per_row, 32)
    warp_val = warp_reduce(local_val, operator.add, width=warp_width)

    warps_per_row = max(threads_per_row // 32, 1)

    if cutlass.const_expr(warps_per_row > 1 or cluster_n > 1):
        if cutlass.const_expr(cluster_n == 1):
            return block_reduce_multirow(
                warp_val, operator.add, reduction_buffer, Float32(0.0)
            )
        else:
            return cluster_reduce_multirow(
                warp_val,
                operator.add,
                reduction_buffer,
                mbar_ptr,
                cluster_n,
                Float32(0.0),
            )
    else:
        return warp_val


# =============================================================================
# Predicate Utility
# =============================================================================


@cute.jit
def predicate_k(tXcX: cute.Tensor, limit: int) -> cute.Tensor:
    """Create predicate tensor for bounds checking (2D tensors)."""
    tXpX = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tXcX, mode=[0, 1]),
                cute.size(tXcX, mode=[1]),
                cute.size(tXcX, mode=[2]),
            ),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
            tXpX[rest_v, 0, rest_k] = cute.elem_less(
                tXcX[(0, rest_v), 0, rest_k][1], limit
            )
    return tXpX


@cute.jit
def predicate_k_3d(tXcX: cute.Tensor, limit: int) -> cute.Tensor:
    """Create predicate tensor for bounds checking (3D tensors).

    For 3D tensors after local_tile, the last coordinate [2] is the head_dim dimension.
    """
    tXpX = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tXcX, mode=[0, 1]),
                cute.size(tXcX, mode=[1]),
                cute.size(tXcX, mode=[2]),
            ),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
            # For 3D tensor, coordinate[2] is the head_dim index
            tXpX[rest_v, 0, rest_k] = cute.elem_less(
                tXcX[(0, rest_v), 0, rest_k][2], limit
            )
    return tXpX


# =============================================================================
# Helper Functions for Kernel Configuration
# =============================================================================


def compute_optimal_vec_size(H: int, max_vec_size: int) -> int:
    """Compute vec_size that maximizes warp utilization.

    For small hidden sizes, using max vec_size may result in fewer than 32 threads,
    wasting warp resources. This function finds the largest vec_size that:
    1. Divides H evenly
    2. Results in at least 32 threads (one full warp)

    Examples:
    - H=128, max=8: vec_size=8 gives 16 threads, vec_size=4 gives 32 threads -> return 4
    - H=4096, max=8: vec_size=8 gives 512 threads -> return 8
    - H=111, max=8: no vec_size divides evenly with >=32 threads, use gcd -> return 1
    """
    # Try vec_sizes from largest to smallest
    for vec_size in [
        max_vec_size,
        max_vec_size // 2,
        max_vec_size // 4,
        max_vec_size // 8,
    ]:
        if vec_size < 1:
            continue
        if H % vec_size != 0:
            continue
        threads_needed = H // vec_size
        if threads_needed >= 32:
            return vec_size
    # Fallback: use gcd for correctness (handles odd sizes like 111)
    return math.gcd(max_vec_size, H)


def compute_threads_per_row(H: int, vec_size: int) -> int:
    """Compute optimal threads per row based on hidden size."""
    threads_needed = (H + vec_size - 1) // vec_size
    # Round up to power of 2, capped at 1024
    threads = 32
    while threads < threads_needed and threads < 1024:
        threads *= 2
    return min(threads, 1024)


def make_tv_layout(threads_per_row: int, vec_size: int, num_vec_blocks: int) -> tuple:
    """Create Thread-Value layout for coalesced vectorized memory access.

    This layout distributes work across threads where each thread handles
    vec_size consecutive elements, and threads are arranged for coalesced access.

    Args:
        threads_per_row: Number of threads processing one row
        vec_size: Number of elements each thread processes per vector load
        num_vec_blocks: Number of vector blocks per row

    Returns:
        Tuple of (shape, stride) for creating cute.Layout
    """
    shape = (
        (threads_per_row, 1),
        (vec_size, num_vec_blocks),
    )
    stride = (
        (vec_size, 1),
        (1, vec_size * threads_per_row),
    )
    return shape, stride


# =============================================================================
# Type Conversion Utilities
# =============================================================================


# Module-level dict to avoid recreation on each call
_TORCH_DTYPE_TO_STR_MAP = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float8_e4m3fn: "float8_e4m3fn",
}


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    return _TORCH_DTYPE_TO_STR_MAP[dtype]


# Re-export utilities from cute_dsl.utils for convenience
__all__ = [
    # Constants
    "FLOAT8_E4M3_MAX",
    "COPY_BITS",
    # PTX intrinsics
    "rcp_approx_ftz",
    "cvt_and_store_f32_to_e4m3_hw",
    "cvt_and_store_f32_to_e4m3_sw",
    "cvt_and_store_8xf32_to_e4m3_hw",
    "cvt_and_store_4xf32_to_e4m3_hw",
    "cvt_and_store_2xf32_to_e4m3_hw",
    "has_hw_fp8_cvt",
    "get_ptr_as_int64",
    # PTX intrinsics - Cluster operations
    "set_block_rank",
    "store_shared_remote",
    "elem_pointer",
    # Device utilities
    "get_sm_version",
    # Reduction utilities
    "warp_reduce",
    "block_reduce",
    "row_reduce_sum",
    "block_reduce_multirow",
    "cluster_reduce_multirow",
    "row_reduce_sum_multirow",
    # Predicate utilities
    "predicate_k",
    "predicate_k_3d",
    # Configuration helpers
    "compute_optimal_vec_size",
    "compute_threads_per_row",
    "make_tv_layout",
    # Type utilities
    "_torch_dtype_to_str",
    # Re-exports from cute_dsl.utils
    "get_cutlass_dtype",
    "get_num_sm",
]
