#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <algorithm>
#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sglang {

namespace cg = cooperative_groups;
using namespace host;

constexpr int kMaskedExpert = -1;
constexpr int kInvalidExpert = -2;

template <int kHidden, int kIntermediate, int kTopK>
struct Nvfp4MoeParams {
  const __nv_bfloat16* x;
  const int32_t* topk_ids;
  const float* topk_weights;
  const uint8_t* w13;
  const uint8_t* w2;
  const uint8_t* w13_scale;
  const uint8_t* w2_scale;
  const float* input_scale_1;
  const float* input_scale_2;
  const float* g1_alpha;
  const float* g1_alpha_up;
  const float* g2_alpha;
  uint8_t* x_q;
  uint8_t* x_scale;
  float* fc1;
  float* fc1_split;
  uint8_t* act_q;
  uint8_t* act_scale;
  float* fc2;
  __nv_bfloat16* output;
  int32_t* pair_experts;
  int32_t* group_rows;
  int32_t* group_pairs;
  int32_t* expert_counts;
  int32_t* group_experts;
  int32_t* group_offsets;
  int32_t* num_groups;
  int tokens;
  int global_experts;
  int local_experts;
  int global_routed_experts;
  int local_routed_experts;
  int local_expert_start;
  int w13_scale_stride;
  int w2_scale_stride;
};

__device__ __forceinline__ uint32_t fp32x8_to_e2m1(const float* values) {
  uint32_t packed;
  asm volatile(
      "{\n"
      ".reg .b8 b0;\n"
      ".reg .b8 b1;\n"
      ".reg .b8 b2;\n"
      ".reg .b8 b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}"
      : "=r"(packed)
      : "f"(values[0]),
        "f"(values[1]),
        "f"(values[2]),
        "f"(values[3]),
        "f"(values[4]),
        "f"(values[5]),
        "f"(values[6]),
        "f"(values[7]));
  return packed;
}

__device__ __forceinline__ int map_expert(
    int global_expert,
    int global_experts,
    int local_experts,
    int global_routed_experts,
    int local_routed_experts,
    int local_expert_start) {
  if (global_expert < 0) {
    return kMaskedExpert;
  }
  if (global_expert >= global_experts) {
    return kInvalidExpert;
  }
  const int local_expert = global_expert < global_routed_experts
                               ? global_expert - local_expert_start
                               : global_expert - global_routed_experts + local_routed_experts;
  return local_expert >= 0 && local_expert < local_experts ? local_expert : kMaskedExpert;
}

template <typename Load>
__device__ __forceinline__ void quantize_group(Load load, float global_scale, uint8_t* output, uint8_t* scale_output) {
  float values[16];
  float maximum = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    values[i] = load(i);
    maximum = fmaxf(maximum, fabsf(values[i]));
  }
  if (maximum == 0.0f) {
    *scale_output = 0;
    *reinterpret_cast<uint64_t*>(output) = 0;
    return;
  }
  __nv_fp8_e4m3 fp8_scale(maximum * global_scale * (1.0f / 6.0f));
  const float block_scale = static_cast<float>(fp8_scale);
  if (block_scale == 0.0f) {
    *scale_output = 0;
    *reinterpret_cast<uint64_t*>(output) = 0;
    return;
  }
  *scale_output = fp8_scale.__x;
  const float quant_scale = global_scale / block_scale;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    values[i] *= quant_scale;
  }
  reinterpret_cast<uint32_t*>(output)[0] = fp32x8_to_e2m1(values);
  reinterpret_cast<uint32_t*>(output)[1] = fp32x8_to_e2m1(values + 8);
}

constexpr int kMoeWarps = 4;
constexpr int kMoeThreads = 32 * kMoeWarps;
constexpr int kMoeTileN = 16 * kMoeWarps;
constexpr int kMoeBWords = kMoeTileN * 20;
constexpr int kMoeAOffset = kMoeBWords;
constexpr int kMoeSfaOffset = kMoeAOffset + 16 * 20;
constexpr int kMoeSfbOffset = kMoeSfaOffset + 256;
constexpr int kMoeStageWords = kMoeSfbOffset + kMoeTileN * 2;
constexpr int kMoeStages = 8;

__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
  uint32_t address;
  asm("{ .reg .u64 value; cvta.to.shared.u64 value, %1; cvt.u32.u64 %0, value; }" : "=r"(address) : "l"(pointer));
  return address;
}

__device__ __forceinline__ void copy_async_16(uint32_t destination, const void* source, bool valid) {
  const int bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;\n"
               :
               : "r"(destination), "l"(source), "r"(bytes));
}

__device__ __forceinline__ void copy_async_8(uint32_t destination, const void* source, bool valid) {
  const int bytes = valid ? 8 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" : : "r"(destination), "l"(source), "r"(bytes));
}

__device__ __forceinline__ void copy_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int kPending>
__device__ __forceinline__ void copy_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(kPending));
}

__device__ __forceinline__ void copy_async_wait(int pending) {
  switch (pending) {
    case 0:
      return copy_async_wait<0>();
    case 1:
      return copy_async_wait<1>();
    case 2:
      return copy_async_wait<2>();
    case 3:
      return copy_async_wait<3>();
    case 4:
      return copy_async_wait<4>();
    case 5:
      return copy_async_wait<5>();
    default:
      return copy_async_wait<6>();
  }
}

__device__ __forceinline__ void grid_barrier() {
  if (gridDim.x == 1) {
    __syncthreads();
  } else {
    cg::this_grid().sync();
  }
}

__global__ void nvfp4_moe_cooperative_capture_probe() {
  cg::this_grid().sync();
}

__device__ __forceinline__ void mma_nvfp4(
    float output[4],
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1,
    uint32_t sfa,
    uint32_t sfb) {
  // PTX ISA 9.7.14.6, Table 40 requires every byte-id and thread-id
  // selector to be zero for scale_vec::4X. All four bytes in sfa and sfb
  // contribute scale factors to the MMA.
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(output[0]), "+f"(output[1]), "+f"(output[2]), "+f"(output[3])
      : "r"(a0),
        "r"(a1),
        "r"(a2),
        "r"(a3),
        "r"(b0),
        "r"(b1),
        "r"(sfa),
        "h"(static_cast<uint16_t>(0)),
        "h"(static_cast<uint16_t>(0)),
        "r"(sfb),
        "h"(static_cast<uint16_t>(0)),
        "h"(static_cast<uint16_t>(0)));
}

template <int kK, int kN, int kSplit = 1>
__device__ __forceinline__ void nvfp4_gemm_unit(
    const uint8_t* activation,
    const uint8_t* activation_scale,
    int activation_row_stride,
    int scale_row_stride,
    int pair_divisor,
    int anchor_pair,
    const int32_t* pair_indices,
    int rows,
    int split_index,
    const uint8_t* weight,
    const uint8_t* weight_scale,
    int weight_scale_stride,
    int expert,
    int tile,
    float alpha_first,
    float alpha_second,
    int alpha_split,
    float* output,
    uint32_t* shared) {
  constexpr int kKBlocks = kK / 16;
  constexpr int kKTiles = (kK + 127) / 128;
  static_assert(kKTiles % kSplit == 0);
  static_assert(kK % 16 == 0);
  static_assert(kN % kMoeTileN == 0);
  constexpr int kLocalKTiles = kKTiles / kSplit;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int q = lane >> 2;
  const int t = lane & 3;
  const int row_base = tile * kMoeTileN;

  if (expert < 0) {
    for (int row = tid; row < kMoeTileN; row += kMoeThreads) {
      output[anchor_pair * kN + row_base + row] = 0.0f;
    }
    return;
  }

  float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const uint8_t* expert_scale = weight_scale + static_cast<int64_t>(expert) * weight_scale_stride;

  auto produce = [&](int ktile, int slot) {
    const int global_ktile = split_index * kLocalKTiles + ktile;
    uint32_t* stage = shared + slot * kMoeStageWords;
    const uint32_t stage_address = shared_address(stage);
    const int row = tid >> 1;
    const int chunk = tid & 1;
    const uint8_t* weight_row = weight + static_cast<int64_t>(row_base + row) * (kK / 2);
    const int byte0 = global_ktile * 64 + chunk * 16;
    const int byte1 = byte0 + 32;
    copy_async_16(stage_address + row * 80 + chunk * 16, weight_row + byte0, byte0 < kK / 2);
    copy_async_16(stage_address + row * 80 + (chunk + 2) * 16, weight_row + byte1, byte1 < kK / 2);

    if (tid < 64) {
      const int activation_row = tid >> 2;
      const int activation_chunk = tid & 3;
      if (pair_indices != nullptr || activation_row == 0) {
        const int byte = global_ktile * 64 + activation_chunk * 16;
        const int pair = pair_indices == nullptr || activation_row >= rows ? anchor_pair : pair_indices[activation_row];
        const int row_index = pair / pair_divisor;
        const uint8_t* activation_source = activation + row_index * activation_row_stride;
        copy_async_16(
            stage_address + (kMoeAOffset + activation_row * 20) * 4 + activation_chunk * 16,
            activation_source + byte,
            activation_row < rows && byte < kK / 2);
      }
    }
    if (tid < 32) {
      const int half = tid >> 4;
      const int scale_row = tid & 15;
      if (pair_indices != nullptr || scale_row == 0) {
        uint32_t scales = 0;
        if (scale_row < rows) {
          const int pair = pair_indices == nullptr ? anchor_pair : pair_indices[scale_row];
          const int row_index = pair / pair_divisor;
          const uint8_t* scale_source = activation_scale + row_index * scale_row_stride;
#pragma unroll
          for (int index = 0; index < 4; ++index) {
            const int block = global_ktile * 8 + half * 4 + index;
            if (block < kKBlocks) {
              scales |= static_cast<uint32_t>(scale_source[block]) << (8 * index);
            }
          }
        }
        stage[kMoeSfaOffset + 4 * scale_row + half * 128] = scales;
      }
    }
    if (tid >= 64 && tid < 128) {
      constexpr int kPaddedBlocks = (kKBlocks + 3) & -4;
      const int scale_thread = tid - 64;
      const int scale_block = scale_thread >> 5;
      const int scale_row = scale_thread & 31;
      const int scale_base = (row_base >> 7) * 512 * (kPaddedBlocks >> 2) + global_ktile * 1024 + scale_block * 512 +
                             scale_row * 16 + 4 * ((row_base >> 5) & 3);
      copy_async_8(
          stage_address + kMoeSfbOffset * 4 + scale_block * kMoeTileN * 4 + scale_row * (kMoeTileN / 8),
          expert_scale + scale_base,
          global_ktile * 8 + scale_block * 4 < kKBlocks);
    }
    copy_async_commit();
  };

  constexpr int kPrologue = kLocalKTiles < kMoeStages - 1 ? kLocalKTiles : kMoeStages - 1;
#pragma unroll
  for (int ktile = 0; ktile < kPrologue; ++ktile) {
    produce(ktile, ktile);
  }
#pragma unroll
  for (int ktile = 0; ktile < kLocalKTiles; ++ktile) {
    const int remaining = kLocalKTiles - ktile - 1;
    copy_async_wait(min(kMoeStages - 2, remaining));
    __syncthreads();
    if (ktile + kMoeStages - 1 < kLocalKTiles) {
      const int next = ktile + kMoeStages - 1;
      produce(next, next % kMoeStages);
    }
    const uint32_t* stage = shared + (ktile % kMoeStages) * kMoeStageWords;
    const int arow = q + ((lane & 1) << 3);
    const int aw0 = kMoeAOffset + q * 20 + t;
    const int aw1 = kMoeAOffset + (q + 8) * 20 + t;
    const int w0 = warp * 16 + q;
    const int w1 = w0 + 8;
    const int bw0 = w0 * 20 + t;
    const int bw1 = w1 * 20 + t;
    const int sfa_word = kMoeSfaOffset + 4 * arow;
    const int sfb0 = kMoeSfbOffset + 2 * (w0 & 31) + (w0 >> 5);
    const int sfb1 = kMoeSfbOffset + 2 * (w1 & 31) + (w1 >> 5);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const uint32_t a0 = stage[aw0 + half * 8];
      const uint32_t a2 = stage[aw0 + half * 8 + 4];
      const uint32_t a1 = stage[aw1 + half * 8];
      const uint32_t a3 = stage[aw1 + half * 8 + 4];
      const uint32_t sfa = stage[sfa_word + half * 128];
      mma_nvfp4(
          acc0, a0, a1, a2, a3, stage[bw0 + half * 8], stage[bw0 + half * 8 + 4], sfa, stage[sfb0 + half * kMoeTileN]);
      mma_nvfp4(
          acc1, a0, a1, a2, a3, stage[bw1 + half * 8], stage[bw1 + half * 8 + 4], sfa, stage[sfb1 + half * kMoeTileN]);
    }
  }
  __syncthreads();

  const int row0 = q;
  const int row1 = q + 8;
  const int column0 = row_base + warp * 16 + 2 * t;
  const int column1 = column0 + 8;
  const float scale0 = column0 < alpha_split ? alpha_first : alpha_second;
  const float scale1 = column1 < alpha_split ? alpha_first : alpha_second;
  if (row0 < rows) {
    const int pair = pair_indices == nullptr ? anchor_pair : pair_indices[row0];
    float* row_output = output + pair * kN;
    row_output[column0] = scale0 * acc0[0];
    row_output[column0 + 1] = scale0 * acc0[1];
    row_output[column1] = scale1 * acc1[0];
    row_output[column1 + 1] = scale1 * acc1[1];
  }
  if (row1 < rows) {
    const int pair = pair_indices == nullptr ? anchor_pair : pair_indices[row1];
    float* row_output = output + pair * kN;
    row_output[column0] = scale0 * acc0[2];
    row_output[column0 + 1] = scale0 * acc0[3];
    row_output[column1] = scale1 * acc1[2];
    row_output[column1 + 1] = scale1 * acc1[3];
  }
}

template <int kHidden, int kIntermediate, int kTopK>
__global__ __launch_bounds__(kMoeThreads) void nvfp4_moe_omma_kernel(
    const __grid_constant__ Nvfp4MoeParams<kHidden, kIntermediate, kTopK> params) {
  static_assert(kHidden % 256 == 0);
  static_assert(kIntermediate % 64 == 0);
  static_assert((2 * kIntermediate) % kMoeTileN == 0);
  static_assert(kHidden % kMoeTileN == 0);
  extern __shared__ __align__(16) uint32_t shared[];
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int pairs = params.tokens * kTopK;
  constexpr int kXScaleStride = kHidden / 16;
  constexpr int kActScaleStride = kIntermediate / 16;
  // Direct pairs only use OMMA row zero. Initialize the unused rows once
  // instead of issuing zero-fill copies for every K stage.
  if (params.tokens < 8) {
    constexpr int kDirectZeroWords = kMoeSfbOffset - kMoeAOffset;
    for (int index = threadIdx.x; index < kMoeStages * kDirectZeroWords; index += blockDim.x) {
      const int stage = index / kDirectZeroWords;
      const int offset = index % kDirectZeroWords;
      shared[stage * kMoeStageWords + kMoeAOffset + offset] = 0;
    }
    __syncthreads();
  }
  for (int group = thread; group < params.tokens * (kHidden / 16); group += stride) {
    const int token = group / (kHidden / 16);
    const int column = (group % (kHidden / 16)) * 16;
    quantize_group(
        [&](int lane) { return __bfloat162float(params.x[token * kHidden + column + lane]); },
        *params.input_scale_1,
        params.x_q + token * (kHidden / 2) + column / 2,
        params.x_scale + token * kXScaleStride + column / 16);
  }
  for (int pair = thread; pair < pairs; pair += stride) {
    params.pair_experts[pair] = map_expert(
        params.topk_ids[pair],
        params.global_experts,
        params.local_experts,
        params.global_routed_experts,
        params.local_routed_experts,
        params.local_expert_start);
  }
  if (params.tokens >= 8) {
    for (int expert = thread; expert < params.local_experts; expert += stride) {
      params.expert_counts[expert] = 0;
    }
    if (thread == 0) {
      *params.num_groups = 0;
    }
  }
  grid_barrier();
  if (params.tokens >= 8) {
    // These atomics only assign rows in the grouped packing. group_pairs maps
    // every packed row back to its original pair, all GEMM stores use that
    // pair index, and the final routing sum visits slots in fixed order.
    for (int pair = thread; pair < pairs; pair += stride) {
      const int expert = params.pair_experts[pair];
      if (expert >= 0) {
        const int ordinal = atomicAdd(params.expert_counts + expert, 1);
        params.group_pairs[expert * pairs + ordinal] = pair;
      }
    }
    grid_barrier();
    for (int expert = thread; expert < params.local_experts; expert += stride) {
      const int count = params.expert_counts[expert];
      if (count > 0) {
        const int expert_groups = (count + 15) / 16;
        const int group_base = atomicAdd(params.num_groups, expert_groups);
        for (int group = 0; group < expert_groups; ++group) {
          const int index = group_base + group;
          params.group_experts[index] = expert;
          params.group_offsets[index] = expert * pairs + group * 16;
          const int remaining = count - group * 16;
          params.group_rows[index] = remaining < 16 ? remaining : 16;
        }
      }
    }
    grid_barrier();
  }
  constexpr int kFc1Tiles = 2 * kIntermediate / kMoeTileN;
  // Two fixed K slices expose enough FC1 blocks to fill the SM120 grid. The
  // activation stage adds them in a fixed order before W4A4 requantization.
  constexpr int kFc1Splits = 2;
  const int fc1_groups = params.tokens >= 8 ? *params.num_groups : pairs;
  for (int unit = blockIdx.x; unit < fc1_groups * kFc1Tiles * kFc1Splits; unit += gridDim.x) {
    const int group = unit / (kFc1Tiles * kFc1Splits);
    const int tile = (unit / kFc1Splits) % kFc1Tiles;
    const int split = unit % kFc1Splits;
    int pair = group;
    int expert = params.pair_experts[pair];
    int rows = 1;
    const int32_t* pair_indices = nullptr;
    if (params.tokens >= 8) {
      rows = params.group_rows[group];
      expert = params.group_experts[group];
      pair_indices = params.group_pairs + params.group_offsets[group];
      pair = pair_indices[0];
    }
    const uint8_t* weight =
        expert < 0 ? params.w13 : params.w13 + static_cast<int64_t>(expert) * (2 * kIntermediate) * (kHidden / 2);
    nvfp4_gemm_unit<kHidden, 2 * kIntermediate, kFc1Splits>(
        params.x_q,
        params.x_scale,
        kHidden / 2,
        kXScaleStride,
        kTopK,
        pair,
        pair_indices,
        rows,
        split,
        weight,
        params.w13_scale,
        params.w13_scale_stride,
        expert,
        tile,
        expert < 0 ? 0.0f : params.g1_alpha_up[expert],
        expert < 0 ? 0.0f : params.g1_alpha[expert],
        kIntermediate,
        split == 0 ? params.fc1 : params.fc1_split,
        shared);
  }
  grid_barrier();

  for (int group = thread; group < pairs * (kIntermediate / 16); group += stride) {
    const int pair = group / (kIntermediate / 16);
    const int column = (group % (kIntermediate / 16)) * 16;
    const int expert = params.pair_experts[pair];
    if (expert < 0) {
      *reinterpret_cast<uint64_t*>(params.act_q + pair * (kIntermediate / 2) + column / 2) = 0;
      continue;
    }
    const float* fc1 = params.fc1 + pair * (2 * kIntermediate);
    const float* fc1_split = params.fc1_split + pair * (2 * kIntermediate);
    quantize_group(
        [&](int lane) {
          const float up = fc1[column + lane] + fc1_split[column + lane];
          const float gate = fc1[kIntermediate + column + lane] + fc1_split[kIntermediate + column + lane];
          const float activated = up * gate / (1.0f + expf(-gate));
          return __bfloat162float(__float2bfloat16(activated));
        },
        *params.input_scale_2,
        params.act_q + pair * (kIntermediate / 2) + column / 2,
        params.act_scale + pair * kActScaleStride + column / 16);
  }
  grid_barrier();

  constexpr int kFc2Tiles = kHidden / kMoeTileN;
  const int fc2_groups = params.tokens >= 8 ? *params.num_groups : pairs;
  for (int unit = blockIdx.x; unit < fc2_groups * kFc2Tiles; unit += gridDim.x) {
    const int group = unit / kFc2Tiles;
    const int tile = unit % kFc2Tiles;
    int pair = group;
    int expert = params.pair_experts[pair];
    int rows = 1;
    const int32_t* pair_indices = nullptr;
    if (params.tokens >= 8) {
      rows = params.group_rows[group];
      expert = params.group_experts[group];
      pair_indices = params.group_pairs + params.group_offsets[group];
      pair = pair_indices[0];
    }
    const uint8_t* weight =
        expert < 0 ? params.w2 : params.w2 + static_cast<int64_t>(expert) * kHidden * (kIntermediate / 2);
    const float alpha = expert < 0 ? 0.0f : params.g2_alpha[expert];
    nvfp4_gemm_unit<kIntermediate, kHidden>(
        params.act_q,
        params.act_scale,
        kIntermediate / 2,
        kActScaleStride,
        1,
        pair,
        pair_indices,
        rows,
        0,
        weight,
        params.w2_scale,
        params.w2_scale_stride,
        expert,
        tile,
        alpha,
        alpha,
        kHidden,
        params.fc2,
        shared);
  }
  grid_barrier();

  for (int index = thread; index < params.tokens * kHidden; index += stride) {
    const int token = index / kHidden;
    const int column = index % kHidden;
    float routed = 0.0f;
    bool invalid_route = false;
#pragma unroll
    for (int slot = 0; slot < kTopK; ++slot) {
      const int pair = token * kTopK + slot;
      const int expert = params.pair_experts[pair];
      if (expert == kInvalidExpert) {
        invalid_route = true;
      } else if (expert >= 0) {
        routed += params.topk_weights[pair] * params.fc2[pair * kHidden + column];
      }
    }
    params.output[index] = __float2bfloat16(invalid_route ? CUDART_NAN_F : routed);
  }
}

template <int kHidden, int kIntermediate, int kTopK>
struct Nvfp4MoeKernel {
  static bool graph_capture_supported() {
    int device = 0;
    int cooperative_launch = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&cooperative_launch, cudaDevAttrCooperativeLaunch, device) != cudaSuccess ||
        !cooperative_launch) {
      cudaGetLastError();
      return false;
    }

    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    bool supported = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess;
    if (supported) {
      supported = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
    }
    if (supported) {
      cudaLaunchAttribute cooperative_attribute{};
      cooperative_attribute.id = cudaLaunchAttributeCooperative;
      cooperative_attribute.val.cooperative = 1;
      cudaLaunchConfig_t config{};
      config.gridDim = dim3(1);
      config.blockDim = dim3(1);
      config.stream = stream;
      config.attrs = &cooperative_attribute;
      config.numAttrs = 1;
      supported = cudaLaunchKernelEx(&config, nvfp4_moe_cooperative_capture_probe) == cudaSuccess;
    }
    const cudaError_t capture_status =
        stream == nullptr ? cudaErrorInvalidResourceHandle : cudaStreamEndCapture(stream, &graph);
    supported = supported && capture_status == cudaSuccess && graph != nullptr;
    if (supported) {
      supported = cudaGraphInstantiate(&graph_exec, graph, 0) == cudaSuccess;
    }
    if (supported) {
      supported = cudaGraphLaunch(graph_exec, stream) == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess;
    }

    if (graph_exec != nullptr) {
      cudaGraphExecDestroy(graph_exec);
    }
    if (graph != nullptr) {
      cudaGraphDestroy(graph);
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    cudaGetLastError();
    return supported;
  }

  static bool
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView topk_ids,
      tvm::ffi::TensorView topk_weights,
      tvm::ffi::TensorView w13,
      tvm::ffi::TensorView w2,
      tvm::ffi::TensorView w13_scale,
      tvm::ffi::TensorView w2_scale,
      tvm::ffi::TensorView input_scale_1,
      tvm::ffi::TensorView input_scale_2,
      tvm::ffi::TensorView g1_alpha,
      tvm::ffi::TensorView g1_alpha_up,
      tvm::ffi::TensorView g2_alpha,
      int64_t global_routed_experts,
      int64_t local_routed_experts,
      int64_t local_expert_start,
      tvm::ffi::TensorView x_q,
      tvm::ffi::TensorView x_scale,
      tvm::ffi::TensorView fc1,
      tvm::ffi::TensorView fc1_split,
      tvm::ffi::TensorView act_q,
      tvm::ffi::TensorView act_scale,
      tvm::ffi::TensorView fc2,
      tvm::ffi::TensorView output,
      tvm::ffi::TensorView pair_experts,
      tvm::ffi::TensorView group_rows,
      tvm::ffi::TensorView group_pairs,
      tvm::ffi::TensorView expert_counts,
      tvm::ffi::TensorView group_experts,
      tvm::ffi::TensorView group_offsets,
      tvm::ffi::TensorView num_groups) {
    const DLDevice device = x.device();
    RuntimeCheck(device.device_type == kDLCUDA, "x must be a CUDA tensor");
    const auto check_device = [&](const tvm::ffi::TensorView& tensor, const char* name) {
      RuntimeCheck(tensor.device() == device, name, " must be on the same CUDA device as x");
    };
    const auto check_contiguous = [&](const tvm::ffi::TensorView& tensor, const char* name) {
      int64_t expected_stride = 1;
      for (int dim = tensor.dim() - 1; dim >= 0; --dim) {
        RuntimeCheck(tensor.size(dim) <= 1 || tensor.stride(dim) == expected_stride, name, " must be contiguous");
        expected_stride *= tensor.size(dim);
      }
    };
    check_device(topk_ids, "topk_ids");
    check_device(topk_weights, "topk_weights");
    check_device(w13, "w13");
    check_device(w2, "w2");
    check_device(w13_scale, "w13_scale");
    check_device(w2_scale, "w2_scale");
    check_device(input_scale_1, "input_scale_1");
    check_device(input_scale_2, "input_scale_2");
    check_device(g1_alpha, "g1_alpha");
    check_device(g1_alpha_up, "g1_alpha_up");
    check_device(g2_alpha, "g2_alpha");
    check_device(x_q, "x_q");
    check_device(x_scale, "x_scale");
    check_device(fc1, "fc1");
    check_device(fc1_split, "fc1_split");
    check_device(act_q, "act_q");
    check_device(act_scale, "act_scale");
    check_device(fc2, "fc2");
    check_device(output, "output");
    check_device(pair_experts, "pair_experts");
    check_device(group_rows, "group_rows");
    check_device(group_pairs, "group_pairs");
    check_device(expert_counts, "expert_counts");
    check_device(group_experts, "group_experts");
    check_device(group_offsets, "group_offsets");
    check_device(num_groups, "num_groups");

    RuntimeCheck(host::is_type<bf16_t>(x.dtype()), "x must be bfloat16");
    RuntimeCheck(host::is_type<int32_t>(topk_ids.dtype()), "topk_ids must be int32");
    RuntimeCheck(host::is_type<float>(topk_weights.dtype()), "topk_weights must be float32");
    RuntimeCheck(host::is_type<uint8_t>(w13.dtype()), "w13 must be uint8");
    RuntimeCheck(host::is_type<uint8_t>(w2.dtype()), "w2 must be uint8");
    RuntimeCheck(host::is_type<fp8_e4m3_t>(w13_scale.dtype()), "w13_scale must be float8_e4m3fn");
    RuntimeCheck(host::is_type<fp8_e4m3_t>(w2_scale.dtype()), "w2_scale must be float8_e4m3fn");
    RuntimeCheck(host::is_type<float>(input_scale_1.dtype()), "input_scale_1 must be float32");
    RuntimeCheck(host::is_type<float>(input_scale_2.dtype()), "input_scale_2 must be float32");
    RuntimeCheck(host::is_type<float>(g1_alpha.dtype()), "g1_alpha must be float32");
    RuntimeCheck(host::is_type<float>(g1_alpha_up.dtype()), "g1_alpha_up must be float32");
    RuntimeCheck(host::is_type<float>(g2_alpha.dtype()), "g2_alpha must be float32");
    RuntimeCheck(host::is_type<uint8_t>(x_q.dtype()), "x_q must be uint8");
    RuntimeCheck(host::is_type<uint8_t>(x_scale.dtype()), "x_scale must be uint8");
    RuntimeCheck(host::is_type<float>(fc1.dtype()), "fc1 must be float32");
    RuntimeCheck(host::is_type<float>(fc1_split.dtype()), "fc1_split must be float32");
    RuntimeCheck(host::is_type<uint8_t>(act_q.dtype()), "act_q must be uint8");
    RuntimeCheck(host::is_type<uint8_t>(act_scale.dtype()), "act_scale must be uint8");
    RuntimeCheck(host::is_type<float>(fc2.dtype()), "fc2 must be float32");
    RuntimeCheck(host::is_type<bf16_t>(output.dtype()), "output must be bfloat16");
    RuntimeCheck(host::is_type<int32_t>(pair_experts.dtype()), "pair_experts must be int32");
    RuntimeCheck(host::is_type<int32_t>(group_rows.dtype()), "group_rows must be int32");
    RuntimeCheck(host::is_type<int32_t>(group_pairs.dtype()), "group_pairs must be int32");
    RuntimeCheck(host::is_type<int32_t>(expert_counts.dtype()), "expert_counts must be int32");
    RuntimeCheck(host::is_type<int32_t>(group_experts.dtype()), "group_experts must be int32");
    RuntimeCheck(host::is_type<int32_t>(group_offsets.dtype()), "group_offsets must be int32");
    RuntimeCheck(host::is_type<int32_t>(num_groups.dtype()), "num_groups must be int32");

    check_contiguous(x, "x");
    check_contiguous(topk_ids, "topk_ids");
    check_contiguous(topk_weights, "topk_weights");
    check_contiguous(w13, "w13");
    check_contiguous(w2, "w2");
    check_contiguous(w13_scale, "w13_scale");
    check_contiguous(w2_scale, "w2_scale");
    check_contiguous(input_scale_1, "input_scale_1");
    check_contiguous(input_scale_2, "input_scale_2");
    check_contiguous(g1_alpha, "g1_alpha");
    check_contiguous(g1_alpha_up, "g1_alpha_up");
    check_contiguous(g2_alpha, "g2_alpha");
    check_contiguous(x_q, "x_q");
    check_contiguous(x_scale, "x_scale");
    check_contiguous(fc1, "fc1");
    check_contiguous(fc1_split, "fc1_split");
    check_contiguous(act_q, "act_q");
    check_contiguous(act_scale, "act_scale");
    check_contiguous(fc2, "fc2");
    check_contiguous(output, "output");
    check_contiguous(pair_experts, "pair_experts");
    check_contiguous(group_rows, "group_rows");
    check_contiguous(group_pairs, "group_pairs");
    check_contiguous(expert_counts, "expert_counts");
    check_contiguous(group_experts, "group_experts");
    check_contiguous(group_offsets, "group_offsets");
    check_contiguous(num_groups, "num_groups");

    RuntimeCheck(x.dim() == 2 && x.size(1) == kHidden, "bad x shape");
    RuntimeCheck(topk_ids.dim() == 2 && topk_ids.size(1) == kTopK, "bad topk_ids shape");
    RuntimeCheck(topk_weights.dim() == 2 && topk_weights.size(1) == kTopK, "bad topk_weights shape");
    RuntimeCheck(x.size(0) == topk_ids.size(0) && x.size(0) == topk_weights.size(0), "route row mismatch");
    RuntimeCheck(x.size(0) > 0, "x must contain at least one token");
    RuntimeCheck(w13.dim() == 3 && w13.size(1) == 2 * kIntermediate && w13.size(2) == kHidden / 2, "bad w13 shape");
    RuntimeCheck(w2.dim() == 3 && w2.size(1) == kHidden && w2.size(2) == kIntermediate / 2, "bad w2 shape");
    RuntimeCheck(w13.size(0) == w2.size(0), "expert count mismatch");
    RuntimeCheck(global_routed_experts > 0, "global_routed_experts must be positive");
    RuntimeCheck(local_routed_experts > 0, "local_routed_experts must be positive");
    RuntimeCheck(local_expert_start >= 0, "local_expert_start must be nonnegative");
    RuntimeCheck(local_expert_start + local_routed_experts <= global_routed_experts, "bad local expert range");
    RuntimeCheck(local_routed_experts <= w13.size(0), "too many local routed experts");
    RuntimeCheck(input_scale_1.numel() == 1, "input_scale_1 must be scalar");
    RuntimeCheck(input_scale_2.numel() == 1, "input_scale_2 must be scalar");
    RuntimeCheck(g1_alpha.numel() == w13.size(0), "g1_alpha size mismatch");
    RuntimeCheck(g1_alpha_up.numel() == w13.size(0), "g1_alpha_up size mismatch");
    RuntimeCheck(g2_alpha.numel() == w13.size(0), "g2_alpha size mismatch");
    constexpr int64_t kW13ScaleStride = ((2 * kIntermediate + 127) / 128) * ((kHidden / 16 + 3) / 4) * 512;
    constexpr int64_t kW2ScaleStride = ((kHidden + 127) / 128) * ((kIntermediate / 16 + 3) / 4) * 512;
    RuntimeCheck(w13_scale.numel() == w13.size(0) * kW13ScaleStride, "w13_scale size mismatch");
    RuntimeCheck(w2_scale.numel() == w2.size(0) * kW2ScaleStride, "w2_scale size mismatch");

    const int64_t pairs = topk_ids.numel();
    RuntimeCheck(x_q.numel() >= x.size(0) * kHidden / 2, "x_q workspace is too small");
    RuntimeCheck(x_scale.numel() >= x.size(0) * kHidden / 16, "x_scale workspace is too small");
    RuntimeCheck(fc1.numel() >= pairs * 2 * kIntermediate, "fc1 workspace is too small");
    RuntimeCheck(act_q.numel() >= pairs * kIntermediate / 2, "act_q workspace is too small");
    RuntimeCheck(act_scale.numel() >= pairs * kIntermediate / 16, "act_scale workspace is too small");
    RuntimeCheck(fc2.numel() >= pairs * kHidden, "fc2 workspace is too small");
    RuntimeCheck(output.numel() >= x.size(0) * kHidden, "output is too small");
    RuntimeCheck(pair_experts.numel() >= pairs, "pair expert workspace is too small");
    RuntimeCheck(group_rows.numel() >= pairs, "group row workspace is too small");
    RuntimeCheck(group_experts.numel() >= pairs, "group expert workspace is too small");
    RuntimeCheck(group_offsets.numel() >= pairs, "group offset workspace is too small");
    RuntimeCheck(num_groups.numel() >= 1, "group count workspace is too small");
    RuntimeCheck(expert_counts.numel() >= w13.size(0), "expert count workspace is too small");
    RuntimeCheck(group_pairs.numel() >= w13.size(0) * pairs, "group pair workspace is too small");
    RuntimeCheck(fc1_split.numel() >= pairs * 2 * kIntermediate, "FC1 split workspace is too small");

    using Params = Nvfp4MoeParams<kHidden, kIntermediate, kTopK>;
    Params params{
        static_cast<const __nv_bfloat16*>(x.data_ptr()),
        static_cast<const int32_t*>(topk_ids.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()),
        static_cast<const uint8_t*>(w13.data_ptr()),
        static_cast<const uint8_t*>(w2.data_ptr()),
        static_cast<const uint8_t*>(w13_scale.data_ptr()),
        static_cast<const uint8_t*>(w2_scale.data_ptr()),
        static_cast<const float*>(input_scale_1.data_ptr()),
        static_cast<const float*>(input_scale_2.data_ptr()),
        static_cast<const float*>(g1_alpha.data_ptr()),
        static_cast<const float*>(g1_alpha_up.data_ptr()),
        static_cast<const float*>(g2_alpha.data_ptr()),
        static_cast<uint8_t*>(x_q.data_ptr()),
        static_cast<uint8_t*>(x_scale.data_ptr()),
        static_cast<float*>(fc1.data_ptr()),
        static_cast<float*>(fc1_split.data_ptr()),
        static_cast<uint8_t*>(act_q.data_ptr()),
        static_cast<uint8_t*>(act_scale.data_ptr()),
        static_cast<float*>(fc2.data_ptr()),
        static_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<int32_t*>(pair_experts.data_ptr()),
        static_cast<int32_t*>(group_rows.data_ptr()),
        static_cast<int32_t*>(group_pairs.data_ptr()),
        static_cast<int32_t*>(expert_counts.data_ptr()),
        static_cast<int32_t*>(group_experts.data_ptr()),
        static_cast<int32_t*>(group_offsets.data_ptr()),
        static_cast<int32_t*>(num_groups.data_ptr()),
        static_cast<int>(x.size(0)),
        static_cast<int>(global_routed_experts + (w13.size(0) - local_routed_experts)),
        static_cast<int>(w13.size(0)),
        static_cast<int>(global_routed_experts),
        static_cast<int>(local_routed_experts),
        static_cast<int>(local_expert_start),
        static_cast<int>(kW13ScaleStride),
        static_cast<int>(kW2ScaleStride),
    };

    constexpr int kThreads = kMoeThreads;
    const int routed_pairs = params.tokens * kTopK;
    constexpr int kFc1Tiles = 2 * kIntermediate / kMoeTileN;
    constexpr int kFc2Tiles = kHidden / kMoeTileN;
    const int work = routed_pairs * std::max(kFc1Tiles, kFc2Tiles);
    const auto kernel = nvfp4_moe_omma_kernel<kHidden, kIntermediate, kTopK>;
    constexpr int kSharedBytes = kMoeStages * kMoeStageWords * sizeof(uint32_t);
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    int cooperative_launch = 0;
    if (cudaDeviceGetAttribute(&cooperative_launch, cudaDevAttrCooperativeLaunch, device.device_id) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    int blocks_per_sm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, kThreads, kSharedBytes) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    if (!cooperative_launch || blocks_per_sm == 0) {
      return false;
    }
    int sm_count = 0;
    if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device.device_id) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    const uint32_t resident_blocks = static_cast<uint32_t>(sm_count) * static_cast<uint32_t>(blocks_per_sm);
    const int blocks = std::min<uint32_t>(work, resident_blocks);
    if (blocks <= 0) {
      return false;
    }

    cudaLaunchAttribute cooperative_attribute{};
    cooperative_attribute.id = cudaLaunchAttributeCooperative;
    cooperative_attribute.val.cooperative = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(blocks);
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = kSharedBytes;
    config.stream = LaunchKernel::resolve_device(device);
    config.attrs = &cooperative_attribute;
    config.numAttrs = 1;
    if (cudaLaunchKernelEx(&config, kernel, params) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    return true;
  }
};

}  // namespace sglang
