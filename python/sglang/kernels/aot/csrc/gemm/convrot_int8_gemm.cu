/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// ConvRot (arXiv:2512.03673) INT8 W8A8 linear. Activations pass through a
// group-wise Hadamard rotation (Kronecker power of the regular 4x4 matrix, see
// the rotate kernel) and per-row dynamic INT8 quantization;
// weights are rotated and quantized offline with the same row-wise transform
// (convrot_rotate_quantize_activation on the [N, K] weight). The GEMM is a
// CUTLASS dense INT8 kernel whose epilogue applies the per-row (activation) x
// per-column (weight) dequant into BF16: CUTLASS 3.x WGMMA on Sm90, tcgen05 on
// Sm100, and the CUTLASS 2.x mma.sync path on the CC 12.x parts that have neither.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/numeric_types.h>
#include <torch/all.h>

#include <algorithm>
#include <array>
#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <iterator>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "cutlass_extensions/epilogue/epilogue_per_row_per_col_scale.h"
#include "cutlass_extensions/gemm/gemm_universal_base_compat.h"
#include "cutlass_extensions/gemm/gemm_with_epilogue_visitor.h"
#include "utils.h"

// Named rather than anonymous: nvcc reports "reference to '_GLOBAL__N_...' is
// ambiguous" for CUTLASS kernel templates in an anonymous namespace when one
// invocation compiles several -gencode targets.
namespace sgl_kernel_convrot_int8_detail {

using namespace cute;

// ---------------------------- Rotate + quantize ----------------------------
// Register-resident: a warp holds a 1024-element tile of one row as 32 fp32 per
// lane, so there is no K-proportional shared memory and no smem bound on K. Lanes
// are split into GroupSize/32-lane clusters, one group per cluster (q = lane within
// the cluster, r = register 0..31). In-group index bits held in r are {0,1,2,4,6}
// (G >= 128) or {0..4} (G = 64), the rest come from q: bits 0-2 in r make each
// lane's data four runs of 8 contiguous elements (16-B loads, 8-B int8 stores), and
// every radix-4 digit above the first has one bit in r and one in q, so a stage
// costs one shuffle per element and the transform needs no block barrier. A row is
// ceil(K / 1024) warps; up to 32 of them (K <= 32768) keep the rotated row in
// registers across the single abs-max barrier. Wider rows take the two-pass variant
// (rotate for the abs-max, re-read the row from L2 and rotate again to quantize).
constexpr int kRotTileElems = 1024;
constexpr int kRotElemsPerLane = 32;
constexpr int kRotMaxWarps = 32;
constexpr int kRotSmallTiles = 16;  // n_tiles <= 16 (K <= 16384) runs under a 512-thread register cap
constexpr int kRotTargetWarpsPerBlock = 8;
constexpr unsigned kRotFullMask = 0xffffffffu;

// GELU (tanh approximation) matching ATen's CUDA kernel bit-for-bit: same fp32
// constants and association order, libdevice tanhf. The sm90 library build
// passes -use_fast_math, which rewrites ::tanhf into the MUFU approximation
// (measured: 22% of BF16 outputs then shift one INT8 step); calling the
// libdevice symbol directly is exempt from that substitution.
extern "C" __device__ float __nv_tanhf(float);

struct ATenGeluTanh {
  __device__ __forceinline__ float operator()(float x) const {
    const float x_cube = x * x * x;
    const float inner = 0.7978845608028654f * (x + 0.044715f * x_cube);
    return 0.5f * x * (1.0f + __nv_tanhf(inner));
  }
};

template <int GroupSize>
struct RotLayout {
  static_assert(GroupSize == 64 || GroupSize == 128 || GroupSize == 256 || GroupSize == 512, "");
  static constexpr int kLanesPerGroup = GroupSize / 32;
  static constexpr int kGroupsPerTile = 32 / kLanesPerGroup;
  // In-group offset of register run j (registers 8j .. 8j+7) for lane-in-cluster q.
  __device__ static __forceinline__ int run_offset(int q, int j) {
    if constexpr (GroupSize == 64) {
      return (q << 5) | (j << 3);
    } else {
      return ((q & 1) << 3) | ((j & 1) << 4) | (((q >> 1) & 1) << 5) | ((j >> 1) << 6) | (((q >> 2) & 1) << 7) |
             (((q >> 3) & 1) << 8);
    }
  }
};

// Regular (non-Sylvester) Hadamard, the Kronecker power of
//   H4 = [[1,1,1,-1], [1,1,-1,1], [1,-1,1,1], [-1,1,1,1]]:
// one 4-point stage per base-4 digit of the in-group index, y_d = S - 2 x_{d^3}
// with S the sum of the four elements that differ only in that digit. Every row
// of the product sums to +1, so a group's mean stays spread over the
// coefficients. The Sylvester transform (row 0 all ones) concentrates it into
// one coefficient at sqrt(GroupSize) x; that coefficient becomes the row absmax
// and coarsens the INT8 step for the whole row, measured 1.3-1.9x more error on
// GELU outputs and other non-zero-mean inputs. Same matrix as comfy-kitchen's
// ConvRot path. Group sizes with an odd bit count finish with one 2-point stage
// on the top bit.
// S is always formed as (own + low-bit partner) + (high-bit partner + both), which
// is the same fp32 expression for the four members of a quartet (a + b == b + a
// bitwise), so the result does not depend on which lane or register holds an
// element. All v[] indices are compile-time constants after unrolling; a
// lane-dependent index would put the array in local memory.

// Both digit bits in the register index (B0 = low bit, B1 = high bit).
template <int B0, int B1>
__device__ __forceinline__ void rot_stage_regreg(float (&v)[kRotElemsPerLane]) {
  CUTLASS_PRAGMA_UNROLL
  for (int r = 0; r < kRotElemsPerLane; r++) {
    if ((r & (B0 | B1)) == 0) {
      const float a0 = v[r], a1 = v[r | B0], a2 = v[r | B1], a3 = v[r | B0 | B1];
      const float S = (a0 + a1) + (a2 + a3);
      v[r] = S - 2.f * a3;
      v[r | B0] = S - 2.f * a2;
      v[r | B1] = S - 2.f * a1;
      v[r | B0 | B1] = S - 2.f * a0;
    }
  }
}

// Low digit bit in the register index (RB), high digit bit in the lane (xor X).
template <int RB, int X>
__device__ __forceinline__ void rot_stage_split(float (&v)[kRotElemsPerLane]) {
  CUTLASS_PRAGMA_UNROLL
  for (int r = 0; r < kRotElemsPerLane; r++) {
    if ((r & RB) == 0) {
      const float a = v[r], b = v[r | RB];
      const float ap = __shfl_xor_sync(kRotFullMask, a, X);
      const float bp = __shfl_xor_sync(kRotFullMask, b, X);
      const float S = (a + b) + (ap + bp);
      v[r] = S - 2.f * bp;
      v[r | RB] = S - 2.f * ap;
    }
  }
}

// 2-point stage on a register bit (RB) ...
template <int RB>
__device__ __forceinline__ void rot_stage_2pt_reg(float (&v)[kRotElemsPerLane]) {
  CUTLASS_PRAGMA_UNROLL
  for (int r = 0; r < kRotElemsPerLane; r++) {
    if ((r & RB) == 0) {
      const float a = v[r], b = v[r | RB];
      v[r] = a + b;
      v[r | RB] = a - b;
    }
  }
}

// ... or on a lane bit (xor X); `hi` = this lane holds the bit-set element.
template <int X>
__device__ __forceinline__ void rot_stage_2pt_lane(float (&v)[kRotElemsPerLane], bool hi) {
  CUTLASS_PRAGMA_UNROLL
  for (int r = 0; r < kRotElemsPerLane; r++) {
    const float mine = v[r];
    const float p = __shfl_xor_sync(kRotFullMask, mine, X);
    v[r] = hi ? (p - mine) : (mine + p);
  }
}

template <int GroupSize>
__device__ __forceinline__ void rot_rotate_tile(float (&v)[kRotElemsPerLane], int q) {
  if constexpr (GroupSize == 64) {
    rot_stage_regreg<1, 2>(v);  // digit 0: in-group bits 0,1
    rot_stage_regreg<4, 8>(v);  // digit 1: bits 2,3
    rot_stage_split<16, 1>(v);  // digit 2: bit 4 (reg) / bit 5 (lane)
  } else {
    rot_stage_regreg<1, 2>(v);  // digit 0: bits 0,1
    rot_stage_split<4, 1>(v);   // digit 1: bit 2 (reg) / bit 3 (lane)
    rot_stage_split<8, 2>(v);   // digit 2: bit 4 (reg) / bit 5 (lane)
    if constexpr (GroupSize == 128) {
      rot_stage_2pt_reg<16>(v);  // bit 6 (reg)
    } else {
      rot_stage_split<16, 4>(v);  // digit 3: bit 6 (reg) / bit 7 (lane)
      if constexpr (GroupSize == 512) {
        rot_stage_2pt_lane<8>(v, (q & 8) != 0);  // bit 8 (lane)
      }
    }
  }
}

// GeluInput fuses the F.gelu(approximate="tanh") that otherwise precedes an
// FFN down-projection. The GELU result is rounded to BF16 and back before the
// butterfly, so the rotated values are bitwise those of the eager
// store-then-reload path.
template <int GroupSize, bool GeluInput>
__device__ __forceinline__ void
rot_load_tile(const __nv_bfloat16* __restrict__ grp, int q, bool active, float (&v)[kRotElemsPerLane]) {
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < 4; j++) {
    uint4 pk = make_uint4(0u, 0u, 0u, 0u);
    if (active) pk = *reinterpret_cast<const uint4*>(grp + RotLayout<GroupSize>::run_offset(q, j));
    const uint32_t w[4] = {pk.x, pk.y, pk.z, pk.w};
    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e < 4; e++) {
      v[8 * j + 2 * e] = __uint_as_float(w[e] << 16);
      v[8 * j + 2 * e + 1] = __uint_as_float(w[e] & 0xffff0000u);
    }
  }
  if constexpr (GeluInput) {
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < kRotElemsPerLane; r++) {
      v[r] = __bfloat162float(__float2bfloat16(ATenGeluTanh{}(v[r])));
    }
  }
}

template <int GroupSize>
__device__ __forceinline__ void rot_quant_store_tile(
    int8_t* __restrict__ grp_out, int q, bool active, const float (&v)[kRotElemsPerLane], float inv_scale) {
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < 4; j++) {
    uint32_t w[2];
    CUTLASS_PRAGMA_UNROLL
    for (int h = 0; h < 2; h++) {
      int32_t qv[4];
      CUTLASS_PRAGMA_UNROLL
      for (int e = 0; e < 4; e++) {
        float t = v[8 * j + 4 * h + e] * inv_scale;
        t = fmaxf(-127.f, fminf(127.f, t));
        qv[e] = __float2int_rn(t);  // == lrintf on the clamped range
      }
      const uint32_t lo = __byte_perm((uint32_t)qv[0], (uint32_t)qv[1], 0x0040);
      const uint32_t hi = __byte_perm((uint32_t)qv[2], (uint32_t)qv[3], 0x0040);
      w[h] = __byte_perm(lo, hi, 0x5410);
    }
    if (active) *reinterpret_cast<uint2*>(grp_out + RotLayout<GroupSize>::run_offset(q, j)) = make_uint2(w[0], w[1]);
  }
}

__device__ __forceinline__ float rot_warp_max(float a) {
  CUTLASS_PRAGMA_UNROLL
  for (int off = 16; off > 0; off >>= 1) {
    a = fmaxf(a, __shfl_xor_sync(kRotFullMask, a, off));
  }
  return a;
}

// Block = dim3(32 * warps_per_row, rows_per_block): threadIdx.y is the row within
// the block, threadIdx.x >> 5 the tile within the row. Resident: warps_per_row ==
// n_tiles and the rotated row stays in registers across the barrier. Two-pass
// (!Resident, K > 32768): 32 warps stride over the tiles.
template <int GroupSize, bool GeluInput, int MaxThreads, bool Resident>
__global__ void __launch_bounds__(MaxThreads) convrot_rotate_quantize_activation_kernel(
    const __nv_bfloat16* __restrict__ x, int8_t* __restrict__ x_q, float* __restrict__ row_scale, int M, int K) {
  using Layout = RotLayout<GroupSize>;
  __shared__ float warp_amax[kRotMaxWarps];

  const int lane = threadIdx.x & 31;
  const int tile0 = threadIdx.x >> 5;
  const int warps_per_row = blockDim.x >> 5;
  const int row_in_block = threadIdx.y;
  const int block_warp = row_in_block * warps_per_row + tile0;
  const int row = blockIdx.x * blockDim.y + row_in_block;
  const bool row_valid = row < M;               // padding rows compute on zeros and still reach the barrier
  const int c = lane / Layout::kLanesPerGroup;  // group within the tile
  const int q = lane % Layout::kLanesPerGroup;  // lane within the group
  const int num_groups = K / GroupSize;
  const int n_tiles = (K + kRotTileElems - 1) / kRotTileElems;
  const __nv_bfloat16* row_in = x + (int64_t)row * K;
  int8_t* row_out = x_q + (int64_t)row * K;
  // H / sqrt(GroupSize) on both operands keeps (Ux) . (Uw) == x . w exactly.
  const float inv_sqrt_group = rsqrtf((float)GroupSize);

  float v[kRotElemsPerLane];
  float local_amax = 0.f;
  const int g0 = tile0 * Layout::kGroupsPerTile + c;
  const bool active0 = row_valid && (g0 < num_groups);
  if constexpr (Resident) {
    rot_load_tile<GroupSize, GeluInput>(row_in + (int64_t)g0 * GroupSize, q, active0, v);
    rot_rotate_tile<GroupSize>(v, q);
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < kRotElemsPerLane; r++) {
      v[r] *= inv_sqrt_group;
      local_amax = fmaxf(local_amax, fabsf(v[r]));
    }
  } else {
    CUTLASS_PRAGMA_NO_UNROLL
    for (int t = tile0; t < n_tiles; t += warps_per_row) {
      const int g = t * Layout::kGroupsPerTile + c;
      const bool active = row_valid && (g < num_groups);
      rot_load_tile<GroupSize, GeluInput>(row_in + (int64_t)g * GroupSize, q, active, v);
      rot_rotate_tile<GroupSize>(v, q);
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < kRotElemsPerLane; r++) {
        local_amax = fmaxf(local_amax, fabsf(v[r] * inv_sqrt_group));
      }
    }
  }

  local_amax = rot_warp_max(local_amax);
  if (lane == 0) warp_amax[block_warp] = local_amax;
  __syncthreads();

  // Every thread folds its row's warp partials (max is order-independent), so no
  // second barrier is needed; the scale expressions are the same as before.
  const float amax = rot_warp_max(lane < warps_per_row ? warp_amax[row_in_block * warps_per_row + lane] : 0.f);
  const float scale = (amax > 0.f) ? (amax / 127.f) : 1.f;
  const float inv_scale = 1.f / scale;
  if (row_valid && tile0 == 0 && lane == 0) row_scale[row] = scale;

  if constexpr (Resident) {
    rot_quant_store_tile<GroupSize>(row_out + (int64_t)g0 * GroupSize, q, active0, v, inv_scale);
  } else {
    CUTLASS_PRAGMA_NO_UNROLL
    for (int t = tile0; t < n_tiles; t += warps_per_row) {
      const int g = t * Layout::kGroupsPerTile + c;
      const bool active = row_valid && (g < num_groups);
      rot_load_tile<GroupSize, GeluInput>(row_in + (int64_t)g * GroupSize, q, active, v);
      rot_rotate_tile<GroupSize>(v, q);
      CUTLASS_PRAGMA_UNROLL
      for (int r = 0; r < kRotElemsPerLane; r++)
        v[r] *= inv_sqrt_group;
      rot_quant_store_tile<GroupSize>(row_out + (int64_t)g * GroupSize, q, active, v, inv_scale);
    }
  }
}

template <int GroupSize, bool GeluInput>
void launch_rotate_quantize_kernel(
    const __nv_bfloat16* x, int8_t* x_q, float* row_scale, int M, int K, cudaStream_t stream) {
  if (M == 0) return;
  // Rows are K * 2 >= 128 bytes and the in-row offsets multiples of 16 bytes, so
  // the base pointers carry the whole 16-B load / 8-B store alignment requirement.
  TORCH_CHECK(
      (reinterpret_cast<uintptr_t>(x) & 15) == 0 && (reinterpret_cast<uintptr_t>(x_q) & 7) == 0,
      "convrot_int8: x must be 16-byte aligned and x_q 8-byte aligned");
  const int n_tiles = (K + kRotTileElems - 1) / kRotTileElems;
  if (n_tiles <= kRotMaxWarps) {
    const int rows_per_block = std::max(1, std::min(M, kRotTargetWarpsPerBlock / n_tiles));
    const dim3 block(32 * n_tiles, rows_per_block);
    const dim3 grid((M + rows_per_block - 1) / rows_per_block);
    if (n_tiles <= kRotSmallTiles) {
      convrot_rotate_quantize_activation_kernel<GroupSize, GeluInput, 512, true>
          <<<grid, block, 0, stream>>>(x, x_q, row_scale, M, K);
    } else {
      convrot_rotate_quantize_activation_kernel<GroupSize, GeluInput, 1024, true>
          <<<grid, block, 0, stream>>>(x, x_q, row_scale, M, K);
    }
  } else {
    convrot_rotate_quantize_activation_kernel<GroupSize, GeluInput, 1024, false>
        <<<dim3(M), dim3(32 * kRotMaxWarps, 1), 0, stream>>>(x, x_q, row_scale, M, K);
  }
  CHECK_CUDA_SUCCESS(cudaGetLastError());
}

template <int GroupSize>
void launch_rotate_quantize(
    const __nv_bfloat16* x, int8_t* x_q, float* row_scale, int M, int K, bool gelu_input, cudaStream_t stream) {
  if (gelu_input) {
    launch_rotate_quantize_kernel<GroupSize, true>(x, x_q, row_scale, M, K, stream);
  } else {
    launch_rotate_quantize_kernel<GroupSize, false>(x, x_q, row_scale, M, K, stream);
  }
}

void check_group_size(int64_t K, int64_t group_size) {
  TORCH_CHECK(
      group_size == 64 || group_size == 128 || group_size == 256 || group_size == 512,
      "convrot_int8: unsupported group_size ",
      group_size,
      " (supported: 64, 128, 256, 512)");
  TORCH_CHECK(K % group_size == 0, "convrot_int8: K (", K, ") must be a multiple of group_size (", group_size, ")");
}

void rotate_quantize_rowwise(
    const torch::Tensor& x,
    torch::Tensor& x_q,
    torch::Tensor& row_scale,
    int64_t group_size,
    bool gelu_input,
    cudaStream_t stream) {
  const int M = x.size(0);
  const int K = x.size(1);
  const auto* x_ptr = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* x_q_ptr = x_q.data_ptr<int8_t>();
  auto* row_scale_ptr = row_scale.data_ptr<float>();
  switch (group_size) {
    case 64:
      launch_rotate_quantize<64>(x_ptr, x_q_ptr, row_scale_ptr, M, K, gelu_input, stream);
      break;
    case 128:
      launch_rotate_quantize<128>(x_ptr, x_q_ptr, row_scale_ptr, M, K, gelu_input, stream);
      break;
    case 256:
      launch_rotate_quantize<256>(x_ptr, x_q_ptr, row_scale_ptr, M, K, gelu_input, stream);
      break;
    case 512:
      launch_rotate_quantize<512>(x_ptr, x_q_ptr, row_scale_ptr, M, K, gelu_input, stream);
      break;
    default:
      TORCH_CHECK(false, "convrot_int8: unreachable group_size ", group_size);
  }
}

// ----------------------------------- GEMM -----------------------------------
using ElementA = int8_t;
using ElementB = int8_t;
using ElementAccumulator = int32_t;
using ElementCompute = float;
using ElementOutput = cutlass::bfloat16_t;

// x_q is [M, K] row-major; weight_q is [N, K] row-major, i.e. B as K x N column-major.
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementOutput>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementOutput>::value;

// out[m, n] = bf16(x_scale[m] * (w_scale[n] * acc[m, n]) [+ bias[n]]). The Sm90
// EVT nodes alias to their Sm100 counterparts, so one tree serves both archs.
template <class MmaTileShape>
struct ConvRotDequantEpilogue {
  using XScale = cutlass::epilogue::fusion::
      Sm90ColBroadcast<0, MmaTileShape, ElementCompute, ElementCompute, Stride<Int<1>, Int<0>, Int<0>>>;
  using WScale = cutlass::epilogue::fusion::
      Sm90RowBroadcast<0, MmaTileShape, ElementCompute, ElementCompute, Stride<Int<0>, Int<1>, Int<0>>>;
  using Bias = cutlass::epilogue::fusion::
      Sm90RowBroadcast<0, MmaTileShape, ElementOutput, ElementOutput, Stride<Int<0>, Int<1>, Int<0>>>;
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using Compute0 = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, WScale, Accum>;

  using Compute1 = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiplies, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using WithoutBias = cutlass::epilogue::fusion::Sm90EVT<Compute1, XScale, EVTCompute0>;

  using ComputeWithBias = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiply_add, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using WithBias = cutlass::epilogue::fusion::Sm90EVT<ComputeWithBias, XScale, EVTCompute0, Bias>;

  template <bool HasBias>
  using EVT = std::conditional_t<HasBias, WithBias, WithoutBias>;
};

// Builder arguments follow CUTLASS's sm100 s8_s8_void_s32 unit-test config
// (KernelScheduleAuto / EpilogueScheduleAuto).
template <class TileShape_, class ClusterShape_>
struct ConvRotGemmSm100 {
  template <bool WithBias>
  struct Select {
    using EpilogueEVT = typename ConvRotDequantEpilogue<TileShape_>::template EVT<WithBias>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        TileShape_,
        ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementOutput,
        LayoutC,
        AlignmentC,
        ElementOutput,
        LayoutD,
        AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        EpilogueEVT>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        ElementA,
        LayoutA,
        AlignmentA,
        ElementB,
        LayoutB,
        AlignmentB,
        ElementAccumulator,
        TileShape_,
        ClusterShape_,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using GemmKernel =
        cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  };
};

// Mirrors cutlass_int8_scaled_mm_sm90 (int8_gemm_kernel.cu): TMA warp-specialized
// epilogue, caller-chosen mainloop schedule, persistent tile scheduler.
template <class TileShape_, class ClusterShape_, class MainloopSchedule_>
struct ConvRotGemmSm90 {
  template <bool WithBias>
  struct Select {
    using EpilogueEVT = typename ConvRotDequantEpilogue<TileShape_>::template EVT<WithBias>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        TileShape_,
        ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementOutput,
        LayoutC,
        AlignmentC,
        ElementOutput,
        LayoutD,
        AlignmentD,
        cutlass::epilogue::TmaWarpSpecialized,
        EpilogueEVT>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ElementA,
        LayoutA,
        AlignmentA,
        ElementB,
        LayoutB,
        AlignmentB,
        ElementAccumulator,
        TileShape_,
        ClusterShape_,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule_>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  };
};

using ConvRotSm100Default = ConvRotGemmSm100<Shape<_128, _128, _128>, Shape<_2, _1, _1>>;
using ConvRotSm100WideN = ConvRotGemmSm100<Shape<_128, _256, _128>, Shape<_2, _1, _1>>;

// Tile choices follow sm90_dispatch_shape (int8_gemm_kernel.cu).
using ConvRotSm90LargeM =
    ConvRotGemmSm90<Shape<_128, _128, _128>, Shape<_2, _1, _1>, cutlass::gemm::KernelTmaWarpSpecializedPingpong>;
using ConvRotSm90SmallMNarrowN =
    ConvRotGemmSm90<Shape<_64, _64, _128>, Shape<_2, _1, _1>, cutlass::gemm::KernelTmaWarpSpecialized>;
using ConvRotSm90SmallMWideN =
    ConvRotGemmSm90<Shape<_64, _128, _128>, Shape<_2, _1, _1>, cutlass::gemm::KernelTmaWarpSpecialized>;

// CC 12.0 / 12.1 (RTX PRO 6000 Blackwell, RTX 50 series, DGX Spark) have neither
// WGMMA nor tcgen05, so the dequant GEMM runs on the CUTLASS 2.x mma.sync INT8
// path sgl-kernel already uses for its sm89 int8_scaled_mm (int8_gemm_kernel.cu),
// whose per-row x per-column epilogue visitor computes the same
// bf16(x_scale[m] * w_scale[n] * acc + bias[n]).
template <class ThreadblockShape, class WarpShape, int NumStages>
void run_convrot_int8_gemm_mma_sync(
    torch::Tensor& out,
    const torch::Tensor& x_q,
    const torch::Tensor& w_q,
    const torch::Tensor& x_scale,
    const torch::Tensor& w_scale,
    const c10::optional<torch::Tensor>& bias,
    cudaStream_t stream) {
  using ArchTag = cutlass::arch::Sm80;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
  using DefaultGemmConf = cutlass::gemm::device::
      DefaultGemmConfiguration<OperatorClass, ArchTag, ElementA, ElementB, ElementOutput, ElementCompute>;
  using EpilogueOutputOp = typename DefaultGemmConf::EpilogueOutputOp;

  using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<
      ElementA,
      LayoutA,
      DefaultGemmConf::kAlignmentA,
      ElementB,
      LayoutB,
      DefaultGemmConf::kAlignmentB,
      ElementOutput,
      LayoutC,
      ElementAccumulator,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      ThreadblockSwizzle,
      NumStages,
      true,
      typename DefaultGemmConf::Operator>::GemmKernel;

  using AlphaColTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
      cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
          typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Shape,
          typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Count,
          GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::kThreads,
          GemmKernel_::Epilogue::OutputTileIterator::kElementsPerAccess,
          cutlass::sizeof_bits<ElementOutput>::value>,
      ElementCompute>;
  using EpilogueVisitor = typename cutlass::epilogue::threadblock::EpilogueVisitorPerRowPerCol<
      ThreadblockShape,
      GemmKernel_::kThreadCount,
      AlphaColTileIterator,
      typename GemmKernel_::Epilogue::OutputTileIterator,
      ElementAccumulator,
      ElementCompute,
      EpilogueOutputOp>;
  using Epilogue = typename cutlass::epilogue::threadblock::
      EpilogueWithVisitorFromExistingEpilogue<EpilogueVisitor, typename GemmKernel_::Epilogue>::Epilogue;
  using GemmKernel =
      cutlass::gemm::kernel::GemmWithEpilogueVisitor<typename GemmKernel_::Mma, Epilogue, ThreadblockSwizzle>;
  using Gemm = cutlass::gemm::device::GemmUniversalBaseCompat<GemmKernel>;

  const int M = x_q.size(0), K = x_q.size(1), N = w_q.size(0);
  // w_q is [N, K] row-major, i.e. B as K x N column-major with ldb = K; the bias
  // TensorRef with stride 0 broadcasts one row.
  ElementOutput* bias_ptr = bias.has_value() ? static_cast<ElementOutput*>(bias->data_ptr()) : nullptr;
  typename EpilogueOutputOp::Params linear_scaling_params;
  typename EpilogueVisitor::Arguments visitor_args{linear_scaling_params};
  typename Gemm::Arguments args{
      {M, N, K},
      {static_cast<ElementA*>(x_q.data_ptr()), static_cast<int64_t>(K)},
      {static_cast<ElementB*>(w_q.data_ptr()), static_cast<int64_t>(K)},
      {static_cast<ElementCompute*>(w_scale.data_ptr()), 0},
      {static_cast<ElementCompute*>(x_scale.data_ptr()), 0},
      {bias_ptr, 0},
      {static_cast<ElementOutput*>(out.data_ptr()), static_cast<int64_t>(N)},
      visitor_args};

  Gemm gemm_op;
  auto workspace =
      torch::empty({static_cast<int64_t>(gemm_op.get_workspace_size(args))}, x_q.options().dtype(torch::kUInt8));
  const cutlass::Status status = gemm_op.can_implement(args);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess, "convrot_int8: can_implement failed: ", cutlassGetStatusString(status));
  const cutlass::Status run_status = gemm_op(args, workspace.data_ptr(), stream);
  TORCH_CHECK(
      run_status == cutlass::Status::kSuccess, "convrot_int8: run failed: ", cutlassGetStatusString(run_status));
}

// Tile table of sm89_dispatch_shape (int8_gemm_kernel.cu): the 100 KB shared-memory
// class the CC 12.x parts share with Ada.
void dispatch_convrot_int8_gemm_mma_sync(
    torch::Tensor& out,
    const torch::Tensor& x_q,
    const torch::Tensor& w_q,
    const torch::Tensor& x_scale,
    const torch::Tensor& w_scale,
    const c10::optional<torch::Tensor>& bias,
    cudaStream_t stream) {
  using cutlass::gemm::GemmShape;
  const int64_t M = x_q.size(0), N = w_q.size(0);
  if (M <= 16) {
    if (N <= 8192) {
      run_convrot_int8_gemm_mma_sync<GemmShape<16, 64, 128>, GemmShape<16, 64, 64>, 5>(
          out, x_q, w_q, x_scale, w_scale, bias, stream);
    } else {
      run_convrot_int8_gemm_mma_sync<GemmShape<16, 128, 128>, GemmShape<16, 64, 64>, 4>(
          out, x_q, w_q, x_scale, w_scale, bias, stream);
    }
  } else if (M <= 32) {
    if (N <= 8192) {
      run_convrot_int8_gemm_mma_sync<GemmShape<32, 64, 128>, GemmShape<16, 64, 64>, 5>(
          out, x_q, w_q, x_scale, w_scale, bias, stream);
    } else {
      run_convrot_int8_gemm_mma_sync<GemmShape<32, 128, 128>, GemmShape<32, 64, 64>, 4>(
          out, x_q, w_q, x_scale, w_scale, bias, stream);
    }
  } else if (M <= 64) {
    if (N <= 8192) {
      run_convrot_int8_gemm_mma_sync<GemmShape<64, 64, 128>, GemmShape<32, 64, 64>, 5>(
          out, x_q, w_q, x_scale, w_scale, bias, stream);
    } else {
      run_convrot_int8_gemm_mma_sync<GemmShape<64, 128, 128>, GemmShape<64, 64, 64>, 3>(
          out, x_q, w_q, x_scale, w_scale, bias, stream);
    }
  } else if (M <= 128) {
    if (N <= 8192) {
      run_convrot_int8_gemm_mma_sync<GemmShape<64, 128, 128>, GemmShape<32, 64, 64>, 3>(
          out, x_q, w_q, x_scale, w_scale, bias, stream);
    } else if (N <= 16384) {
      run_convrot_int8_gemm_mma_sync<GemmShape<128, 128, 64>, GemmShape<64, 64, 64>, 5>(
          out, x_q, w_q, x_scale, w_scale, bias, stream);
    } else {
      run_convrot_int8_gemm_mma_sync<GemmShape<64, 64, 128>, GemmShape<32, 64, 64>, 5>(
          out, x_q, w_q, x_scale, w_scale, bias, stream);
    }
  } else if (M <= 256) {
    if (N <= 4096) {
      run_convrot_int8_gemm_mma_sync<GemmShape<64, 128, 128>, GemmShape<64, 64, 64>, 3>(
          out, x_q, w_q, x_scale, w_scale, bias, stream);
    } else if (N <= 8192) {
      run_convrot_int8_gemm_mma_sync<GemmShape<128, 128, 64>, GemmShape<64, 64, 64>, 5>(
          out, x_q, w_q, x_scale, w_scale, bias, stream);
    } else if (N <= 16384) {
      run_convrot_int8_gemm_mma_sync<GemmShape<256, 128, 64>, GemmShape<64, 64, 64>, 3>(
          out, x_q, w_q, x_scale, w_scale, bias, stream);
    } else {
      run_convrot_int8_gemm_mma_sync<GemmShape<128, 128, 64>, GemmShape<64, 64, 64>, 5>(
          out, x_q, w_q, x_scale, w_scale, bias, stream);
    }
  } else {
    run_convrot_int8_gemm_mma_sync<GemmShape<128, 128, 64>, GemmShape<64, 64, 64>, 5>(
        out, x_q, w_q, x_scale, w_scale, bias, stream);
  }
}

// The one list of parts these ops run on, as exact CC (major * 10 + minor), not the
// major: sm_90a and sm_100a carry the WGMMA / tcgen05 kernels (the sm_100f family
// pass has the INT8 tcgen05 MMA compiled out, so a CC 10.3 part would trap in a
// stub), CC 12.0 / 12.1 take the mma.sync path. Published to Python through
// convrot_int8_supported_sm_versions, so the quantization method and the tests
// read it from here instead of keeping their own copy.
constexpr int kConvRotSupportedSmVersions[] = {90, 100, 120, 121};

bool is_supported_sm_version(int sm) {
  for (int v : kConvRotSupportedSmVersions) {
    if (v == sm) return true;
  }
  return false;
}

std::string supported_sm_versions_text() {
  std::string text;
  for (int v : kConvRotSupportedSmVersions) {
    text += (text.empty() ? "SM" : ", SM") + std::to_string(v);
  }
  return text;
}

// Cached per device index: a process may drive GPUs of different generations, and
// every linear asks for the SM of the device the caller's CUDA guard selected.
int device_sm_version() {
  static std::array<int, 64> cache{};  // 0 = not queried yet
  int device = 0;
  CHECK_CUDA_SUCCESS(cudaGetDevice(&device));
  if (device < 0 || device >= static_cast<int>(cache.size())) return getSMVersion();
  if (cache[device] == 0) cache[device] = getSMVersion();
  return cache[device];
}

void check_supported_device() {
  const int sm = device_sm_version();
  TORCH_CHECK(
      is_supported_sm_version(sm),
      "convrot_int8: no kernel for SM",
      sm,
      " (supported: ",
      supported_sm_versions_text(),
      ")");
}

template <bool WithBias, class Types>
void run_convrot_int8_gemm(
    torch::Tensor& out,
    const torch::Tensor& x_q,
    const torch::Tensor& w_q,
    const torch::Tensor& x_scale,
    const torch::Tensor& w_scale,
    const c10::optional<torch::Tensor>& bias,
    cudaStream_t stream) {
  using Gemm = typename Types::template Select<WithBias>::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  const int M = x_q.size(0), K = x_q.size(1), N = w_q.size(0);

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
  StrideC stride_C;
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {static_cast<const ElementA*>(x_q.data_ptr()), stride_A, static_cast<const ElementB*>(w_q.data_ptr()), stride_B},
      {{}, nullptr, stride_C, static_cast<ElementOutput*>(out.data_ptr()), stride_D}};

  const float* x_s = static_cast<const float*>(x_scale.data_ptr());
  const float* w_s = static_cast<const float*>(w_scale.data_ptr());
  if constexpr (WithBias) {
    const ElementOutput* bias_ptr = static_cast<const ElementOutput*>(bias->data_ptr());
    args.epilogue.thread = {{x_s}, {{w_s}, {}, {}}, {bias_ptr}, {}};
  } else {
    args.epilogue.thread = {{x_s}, {{w_s}, {}, {}}, {}};
  }

  Gemm gemm_op;
  auto workspace =
      torch::empty({static_cast<int64_t>(Gemm::get_workspace_size(args))}, x_q.options().dtype(torch::kUInt8));

  const cutlass::Status status = gemm_op.can_implement(args);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess, "convrot_int8: can_implement failed: ", cutlassGetStatusString(status));

  const cutlass::Status init_status = gemm_op.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(
      init_status == cutlass::Status::kSuccess,
      "convrot_int8: initialize failed: ",
      cutlassGetStatusString(init_status));
  const cutlass::Status run_status = gemm_op.run(stream);
  TORCH_CHECK(
      run_status == cutlass::Status::kSuccess, "convrot_int8: run failed: ", cutlassGetStatusString(run_status));
}

template <bool WithBias>
void dispatch_convrot_int8_gemm(
    torch::Tensor& out,
    const torch::Tensor& x_q,
    const torch::Tensor& w_q,
    const torch::Tensor& x_scale,
    const torch::Tensor& w_scale,
    const c10::optional<torch::Tensor>& bias,
    cudaStream_t stream) {
  const int64_t M = x_q.size(0), K = x_q.size(1), N = w_q.size(0);
  const int sm = device_sm_version();
  if (sm >= 120) {
    dispatch_convrot_int8_gemm_mma_sync(out, x_q, w_q, x_scale, w_scale, bias, stream);
    return;
  }
  if (sm == 90) {
    if (M > 128) {
      run_convrot_int8_gemm<WithBias, ConvRotSm90LargeM>(out, x_q, w_q, x_scale, w_scale, bias, stream);
    } else if (N <= 4096) {
      run_convrot_int8_gemm<WithBias, ConvRotSm90SmallMNarrowN>(out, x_q, w_q, x_scale, w_scale, bias, stream);
    } else {
      run_convrot_int8_gemm<WithBias, ConvRotSm90SmallMWideN>(out, x_q, w_q, x_scale, w_scale, bias, stream);
    }
    return;
  }
  // Measured on B200-class parts with bias: 128x256 wins at large M except the M >= 4096,
  // K = 12288 down-projection. The no-bias path is unmeasured; constexpr keeps its wide tile uninstantiated.
  if constexpr (WithBias) {
    if (M >= 256 && (K <= 8192 || M <= 2048)) {
      run_convrot_int8_gemm<true, ConvRotSm100WideN>(out, x_q, w_q, x_scale, w_scale, bias, stream);
      return;
    }
  }
  run_convrot_int8_gemm<WithBias, ConvRotSm100Default>(out, x_q, w_q, x_scale, w_scale, bias, stream);
}

void convrot_int8_gemm(
    torch::Tensor& out,
    const torch::Tensor& x_q,
    const torch::Tensor& w_q,
    const torch::Tensor& x_scale,
    const torch::Tensor& w_scale,
    const c10::optional<torch::Tensor>& bias,
    cudaStream_t stream) {
  if (bias.has_value()) {
    dispatch_convrot_int8_gemm<true>(out, x_q, w_q, x_scale, w_scale, bias, stream);
  } else {
    dispatch_convrot_int8_gemm<false>(out, x_q, w_q, x_scale, w_scale, bias, stream);
  }
}

void check_weight(
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_scale,
    const c10::optional<torch::Tensor>& bias,
    int64_t K) {
  CHECK_INPUT(weight_q);
  CHECK_INPUT(weight_scale);
  CHECK_DIM(2, weight_q);
  TORCH_CHECK(weight_q.scalar_type() == torch::kInt8, "convrot_int8: weight_q must be int8");
  TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat32, "convrot_int8: weight_scale must be float32");
  TORCH_CHECK(weight_q.size(1) == K, "convrot_int8: weight_q must be [N, K] with K = ", K, ", got ", weight_q.sizes());
  const int64_t N = weight_q.size(0);
  // Every GEMM path stores the BF16 output 8 elements at a time.
  TORCH_CHECK(N % 8 == 0, "convrot_int8: N (weight rows) must be a multiple of 8, got ", N);
  TORCH_CHECK(weight_scale.numel() == N, "convrot_int8: weight_scale must have N = ", N, " elements");
  if (bias.has_value()) {
    CHECK_INPUT(bias.value());
    TORCH_CHECK(bias->scalar_type() == torch::kBFloat16, "convrot_int8: bias must be BF16");
    TORCH_CHECK(bias->numel() == N, "convrot_int8: bias must have N = ", N, " elements");
  }
}

torch::Tensor make_or_check_out(
    const c10::optional<torch::Tensor>& out_opt, int64_t M, int64_t N, const torch::TensorOptions& options) {
  if (!out_opt.has_value()) {
    return torch::empty({M, N}, options.dtype(torch::kBFloat16));
  }
  const torch::Tensor& out = *out_opt;
  TORCH_CHECK(
      out.is_cuda() && out.is_contiguous() && out.scalar_type() == torch::kBFloat16 && out.dim() == 2 &&
          out.size(0) == M && out.size(1) == N,
      "convrot_int8: out must be a contiguous BF16 [M, N] tensor with M = ",
      M,
      ", N = ",
      N);
  return out;
}

torch::Tensor fused_linear_impl(
    const torch::Tensor& x,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_scale,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size,
    bool gelu_input,
    const c10::optional<torch::Tensor>& out_opt) {
  CHECK_INPUT(x);
  CHECK_DIM(2, x);
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "convrot_int8: x must be BF16");
  const int64_t M = x.size(0), K = x.size(1);
  check_group_size(K, group_size);
  check_weight(weight_q, weight_scale, bias, K);
  const int64_t N = weight_q.size(0);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  check_supported_device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  torch::Tensor out = make_or_check_out(out_opt, M, N, x.options());
  if (M == 0) return out;

  auto x_q = torch::empty({M, K}, x.options().dtype(torch::kInt8));
  auto x_scale = torch::empty({M}, x.options().dtype(torch::kFloat32));
  rotate_quantize_rowwise(x, x_q, x_scale, group_size, gelu_input, stream);
  convrot_int8_gemm(out, x_q, weight_q, x_scale, weight_scale, bias, stream);
  return out;
}

torch::Tensor linear_prequant_impl(
    const torch::Tensor& xq,
    const torch::Tensor& xs,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_scale,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size,
    const c10::optional<torch::Tensor>& out_opt) {
  CHECK_INPUT(xq);
  CHECK_INPUT(xs);
  CHECK_DIM(2, xq);
  CHECK_DIM(1, xs);
  TORCH_CHECK(xq.scalar_type() == torch::kInt8, "convrot_int8: xq must be int8");
  TORCH_CHECK(xs.scalar_type() == torch::kFloat32, "convrot_int8: xs must be float32");
  const int64_t M = xq.size(0), K = xq.size(1);
  TORCH_CHECK(xs.numel() == M, "convrot_int8: xs must have M = ", M, " elements");
  check_group_size(K, group_size);
  check_weight(weight_q, weight_scale, bias, K);
  const int64_t N = weight_q.size(0);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(xq));
  check_supported_device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  torch::Tensor out = make_or_check_out(out_opt, M, N, xq.options());
  if (M == 0) return out;

  convrot_int8_gemm(out, xq, weight_q, xs, weight_scale, bias, stream);
  return out;
}

}  // namespace sgl_kernel_convrot_int8_detail

using namespace sgl_kernel_convrot_int8_detail;

/**
 * \brief Group-wise Hadamard rotation followed by per-row dynamic INT8 quantization.
 *
 * Applies H / sqrt(group_size) to each group_size-wide slice of every row, then
 * quantizes the row symmetrically with scale = absmax / 127 (scale 1 for an
 * all-zero row). This is the activation half of the fused ops below, run by the
 * same kernel, and also the offline transform for the [N, K] weight that yields
 * weight_q / weight_scale.
 *
 * \param x BF16 [M, K], contiguous; K must be a multiple of group_size.
 * \param group_size Hadamard group width: 64, 128, 256 or 512.
 * \return (x_q int8 [M, K], x_scale float32 [M]).
 */
std::tuple<torch::Tensor, torch::Tensor>
convrot_rotate_quantize_activation(const torch::Tensor& x, int64_t group_size) {
  CHECK_INPUT(x);
  CHECK_DIM(2, x);
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "convrot_int8: x must be BF16");
  const int64_t M = x.size(0), K = x.size(1);
  check_group_size(K, group_size);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto x_q = torch::empty({M, K}, x.options().dtype(torch::kInt8));
  auto x_scale = torch::empty({M}, x.options().dtype(torch::kFloat32));
  if (M > 0) {
    rotate_quantize_rowwise(x, x_q, x_scale, group_size, /*gelu_input=*/false, stream);
  }
  return {x_q, x_scale};
}

/**
 * \brief ConvRot INT8 linear: rotate and quantize x, then INT8 GEMM with fused dequant.
 *
 * out[m, n] = bf16(x_scale[m] * (weight_scale[n] * sum_k x_q[m, k] * weight_q[n, k]) + bias[n]).
 * Bitwise equal to convrot_int8_linear_prequant applied to the output of
 * convrot_rotate_quantize_activation(x, group_size).
 *
 * \param x BF16 [M, K], contiguous; K must be a multiple of group_size.
 * \param weight_q int8 [N, K] from convrot_rotate_quantize_activation(weight, group_size).
 * \param weight_scale float32 [N].
 * \param bias Optional BF16 [N].
 * \param group_size Hadamard group width shared by both operands.
 * \return BF16 [M, N].
 */
torch::Tensor convrot_int8_fused_linear(
    const torch::Tensor& x,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_scale,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size) {
  return fused_linear_impl(x, weight_q, weight_scale, bias, group_size, /*gelu_input=*/false, c10::nullopt);
}

/**
 * \brief convrot_int8_fused_linear with GELU(tanh) applied to x inside the rotate kernel.
 *
 * Bitwise equal to convrot_int8_fused_linear(F.gelu(x, approximate="tanh"), ...):
 * the GELU is evaluated with ATen's fp32 formula and rounded to BF16 before the
 * rotation. Intended for FFN down-projections fed by the raw up-projection output.
 *
 * \param x BF16 [M, K] pre-activation, contiguous; K must be a multiple of group_size.
 * \param weight_q int8 [N, K].
 * \param weight_scale float32 [N].
 * \param bias Optional BF16 [N].
 * \param group_size Hadamard group width shared by both operands.
 * \return BF16 [M, N].
 */
torch::Tensor convrot_int8_fused_linear_gelu_input(
    const torch::Tensor& x,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_scale,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size) {
  return fused_linear_impl(x, weight_q, weight_scale, bias, group_size, /*gelu_input=*/true, c10::nullopt);
}

/**
 * \brief convrot_int8_fused_linear writing into a caller-provided output.
 *
 * Bitwise equal to convrot_int8_fused_linear; lets a caller project straight
 * into a slice of a preallocated buffer.
 *
 * \param out BF16 [M, N], contiguous.
 * \return out.
 */
torch::Tensor convrot_int8_fused_linear_out(
    const torch::Tensor& x,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_scale,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size,
    torch::Tensor out) {
  return fused_linear_impl(x, weight_q, weight_scale, bias, group_size, /*gelu_input=*/false, out);
}

/**
 * \brief INT8 GEMM with fused dequant on an already rotated and quantized activation.
 *
 * Bitwise equal to convrot_int8_fused_linear on the x that produced (xq, xs):
 * the tile configuration is a function of (M, K, N, bias) alone and the epilogue
 * is evaluated per element. Lets several linears sharing one input (e.g. a
 * q/k/v projection trio) quantize it once.
 *
 * \param xq int8 [M, K] from convrot_rotate_quantize_activation.
 * \param xs float32 [M] from convrot_rotate_quantize_activation.
 * \param weight_q int8 [N, K].
 * \param weight_scale float32 [N].
 * \param bias Optional BF16 [N].
 * \param group_size Hadamard group width xq was produced with; K must be a multiple of it.
 * \return BF16 [M, N].
 */
torch::Tensor convrot_int8_linear_prequant(
    const torch::Tensor& xq,
    const torch::Tensor& xs,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_scale,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size) {
  return linear_prequant_impl(xq, xs, weight_q, weight_scale, bias, group_size, c10::nullopt);
}

/**
 * \brief convrot_int8_linear_prequant writing into a caller-provided output.
 *
 * \param out BF16 [M, N], contiguous.
 * \return out.
 */
torch::Tensor convrot_int8_linear_prequant_out(
    const torch::Tensor& xq,
    const torch::Tensor& xs,
    const torch::Tensor& weight_q,
    const torch::Tensor& weight_scale,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size,
    torch::Tensor out) {
  return linear_prequant_impl(xq, xs, weight_q, weight_scale, bias, group_size, out);
}

/**
 * \brief Compute capabilities (major * 10 + minor) the convrot_int8_* ops carry code for.
 */
std::vector<int64_t> convrot_int8_supported_sm_versions() {
  return std::vector<int64_t>(std::begin(kConvRotSupportedSmVersions), std::end(kConvRotSupportedSmVersions));
}
