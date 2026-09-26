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

// SM90 FP8 Tensor Core index-logits kernel for request-major speculative rows.
// One CTA owns one request group and a chunk of compressed positions. Small
// groups keep decoded Q resident across K tiles; each K tile serves all rows.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>

#include "params.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace sglang {
namespace fp4_grouped_indexer_sm90 {

using namespace cute;
using bf16 = cutlass::bfloat16_t;
using fp8 = cutlass::float_e5m2_t;

#define FP4_INDEXER_CUDA_CHECK(call)                                                            \
  do {                                                                                          \
    cudaError_t err = (call);                                                                   \
    if (err != cudaSuccess) {                                                                   \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                                                                  \
    }                                                                                           \
  } while (0)

__host__ __device__ __forceinline__ constexpr int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

__device__ __forceinline__ float ue8m0_to_f32(uint8_t exponent) {
  return __uint_as_float(static_cast<uint32_t>(exponent) << 23);
}

__device__ __forceinline__ uint8_t scaled_e2m1_to_e5m2(uint8_t code, int exponent_delta) {
  // Positive E5M2 encodings of {0, .5, 1, 1.5, 2, 3, 4, 6}. Multiplication
  // by a power of two is an exact exponent-field adjustment. The common scale
  // is selected so production FP4 blocks remain in the normal E5M2 range.
  constexpr uint64_t lut = 0x464442403e3c3800ULL;
  const uint8_t magnitude = code & 7;
  if (magnitude == 0) {
    return 0;
  }
  const int base = static_cast<int>((lut >> (magnitude * 8)) & 0xff);
  return static_cast<uint8_t>(base + exponent_delta * 4) | ((code & 8) << 4);
}

template <typename Kernel>
__global__ void fp4_grouped_indexer_kernel(__grid_constant__ const Sm90Fp4GroupedIndexerParams params);

template <int TILES_PER_CTA = 1, int Q_ROWS = 4, int WARPGROUPS = 4>
struct Sm90Fp4GroupedIndexerKernel {
  static constexpr int HEADS = 64;
  static constexpr int HEAD_DIM = 128;
  static constexpr int BLOCK_L = 64;
  static constexpr int SCALE_GROUPS = 4;
  static constexpr int SCALE_GROUP_SIZE = 32;
  static constexpr int NUM_WARPGROUPS = WARPGROUPS;
  static constexpr int WARPS_PER_WARPGROUP = 4;
  static constexpr int NUM_THREADS = 128 * NUM_WARPGROUPS;
  static constexpr int MIN_BLOCKS = 2;
  static constexpr bool RESIDENT_Q = TILES_PER_CTA > 1;
  static_assert(TILES_PER_CTA >= 1 && Q_ROWS >= NUM_WARPGROUPS);

  using SmemLayout =
      decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<fp8>{}, Shape<Int<64>, Int<128>>{}, Step<_1, _2>{}));
  using TiledMMA = decltype(make_tiled_mma(GMMA::MMA_64x64x32_F32E5M2E5M2_SS_TN<>{}, Layout<Shape<_1, _1, _1>>{}));

  struct QueryStorage {
    array_aligned<fp8, cosize_v<SmemLayout>, 128> q[Q_ROWS];
    array_aligned<float, WARPS_PER_WARPGROUP * BLOCK_L, 128> warp_sums[NUM_WARPGROUPS];
    float q_scales[Q_ROWS][HEADS];
  };

  struct SharedStorage : QueryStorage {
    array_aligned<fp8, cosize_v<SmemLayout>, 128> k;
    int32_t slots[BLOCK_L];
    uint8_t k_exponents[SCALE_GROUPS][BLOCK_L];
    float k_scales[BLOCK_L];
    int64_t group_max_len;
  };

  static __device__ __forceinline__ void
  fill_invisible(const Sm90Fp4GroupedIndexerParams& p, int b0, int group_rows, int start, int count, int tid) {
    float* out = reinterpret_cast<float*>(p.out);
    for (int linear = tid; linear < group_rows * count; linear += NUM_THREADS) {
      const int row = linear / count;
      const int position = start + linear % count;
      if (position < p.width) {
        out[static_cast<int64_t>(b0 + row) * p.out_stride + position] = -INFINITY;
      }
    }
  }

  // Every CTA thread must call this helper, including inactive tail warpgroups.
  template <typename Storage>
  static __device__ __forceinline__ void
  decode_q(const Sm90Fp4GroupedIndexerParams& p, Storage& ss, int b, int q_slot, int wg_tid, bool active = true) {
    const uint8_t* q = reinterpret_cast<const uint8_t*>(p.q);
    const uint32_t* q_scale = reinterpret_cast<const uint32_t*>(p.q_scale);
    Tensor sQ = make_tensor(make_smem_ptr(ss.q[q_slot].data()), SmemLayout{});
    if (active && wg_tid < HEADS) {
      const uint32_t packed_scale = q_scale[static_cast<int64_t>(b) * p.q_scale_stride_b + wg_tid];
      uint8_t max_exponent = 1;
      CUTE_UNROLL
      for (int g = 0; g < SCALE_GROUPS; ++g) {
        max_exponent = max(max_exponent, static_cast<uint8_t>(packed_scale >> (8 * g)));
      }
      const uint8_t common_exponent = max(static_cast<int>(max_exponent) - 12, 1);
      ss.q_scales[q_slot][wg_tid] = ue8m0_to_f32(common_exponent);
    }
    __syncthreads();
    if (!active) return;
    for (int pair = wg_tid; pair < HEADS * (HEAD_DIM / 2); pair += 128) {
      const int head = pair / (HEAD_DIM / 2);
      const int d = (pair % (HEAD_DIM / 2)) * 2;
      const int g = d / SCALE_GROUP_SIZE;
      const uint8_t packed =
          q[static_cast<int64_t>(b) * p.q_stride_b + static_cast<int64_t>(head) * p.q_stride_h + d / 2];
      const uint32_t packed_scale = q_scale[static_cast<int64_t>(b) * p.q_scale_stride_b + head];
      const int exponent = static_cast<uint8_t>(packed_scale >> (8 * g));
      const int common_exponent = (__float_as_uint(ss.q_scales[q_slot][head]) >> 23) & 0xff;
      const int exponent_delta = exponent - common_exponent;
      fp8 v0, v1;
      *reinterpret_cast<uint8_t*>(&v0) = scaled_e2m1_to_e5m2(packed & 0xf, exponent_delta);
      *reinterpret_cast<uint8_t*>(&v1) = scaled_e2m1_to_e5m2(packed >> 4, exponent_delta);
      sQ(head, d) = v0;
      sQ(head, d + 1) = v1;
    }
  }

  template <typename TA, typename TB, typename TC>
  static __device__ __forceinline__ void gemm_k128(TiledMMA& mma, TA const& sQ, TB const& sK, TC& acc, int tid) {
    ThrMMA thr_mma = mma.get_slice(tid);
    Tensor q_frag = thr_mma.partition_fragment_A(sQ);
    Tensor k_frag = thr_mma.partition_fragment_B(sK);
    static_assert(size<2>(q_frag) == 4);
    static_assert(size<2>(k_frag) == 4);
    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    mma.accumulate_ = GMMA::ScaleOut::Zero;
    CUTE_UNROLL
    for (int k = 0; k < size<2>(q_frag); ++k) {
      cute::gemm(mma, q_frag(_, _, k), k_frag(_, _, k), acc);
      mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_fence_operand(acc);
  }

  template <typename Storage, typename Acc>
  static __device__ __forceinline__ void store_logits(
      const Sm90Fp4GroupedIndexerParams& p,
      Storage& ss,
      const float* k_scales,
      const Acc& acc,
      int b,
      int q_slot,
      int l0,
      bool active,
      int tid) {
    const int warpgroup = tid / 128;
    const int wg_tid = tid % 128;
    if (active) {
      const bf16* weights = reinterpret_cast<const bf16*>(p.weights);
      const int warp = wg_tid / 32;
      const int lane = wg_tid % 32;
      const int head_in_warp = lane / 4;
      const int head0 = warp * 16 + head_in_warp;
      const int head1 = head0 + 8;
      const float q_scale0 = ss.q_scales[q_slot][head0];
      const float q_scale1 = ss.q_scales[q_slot][head1];
      const float weight0 = static_cast<float>(weights[static_cast<int64_t>(b) * p.weight_stride_b + head0]);
      const float weight1 = static_cast<float>(weights[static_cast<int64_t>(b) * p.weight_stride_b + head1]);
      CUTE_UNROLL
      for (int j = 0; j < BLOCK_L / 8; ++j) {
        CUTE_UNROLL
        for (int cp = 0; cp < 2; ++cp) {
          const int col = (lane % 4) * 2 + 8 * j + cp;
          const float k_scale = k_scales[col];
          const float score0 = fmaxf(static_cast<float>(bf16(acc(j * 4 + cp) * q_scale0 * k_scale)), 0.0f);
          const float score1 = fmaxf(static_cast<float>(bf16(acc(j * 4 + 2 + cp) * q_scale1 * k_scale)), 0.0f);
          float sum0 = static_cast<float>(bf16(score0 * weight0));
          float sum1 = static_cast<float>(bf16(score1 * weight1));
          sum0 += __shfl_down_sync(0xffffffffu, sum0, 16);
          sum1 += __shfl_down_sync(0xffffffffu, sum1, 16);
          sum0 += __shfl_down_sync(0xffffffffu, sum0, 8);
          sum1 += __shfl_down_sync(0xffffffffu, sum1, 8);
          sum0 += __shfl_down_sync(0xffffffffu, sum0, 4);
          sum1 += __shfl_down_sync(0xffffffffu, sum1, 4);
          if (head_in_warp == 0) {
            ss.warp_sums[warpgroup][warp * BLOCK_L + col] = sum0 + sum1;
          }
        }
      }
    }
    __syncthreads();
    if (active && wg_tid < BLOCK_L) {
      float sum = ss.warp_sums[warpgroup][wg_tid];
      CUTE_UNROLL
      for (int warp = 1; warp < WARPS_PER_WARPGROUP; ++warp) {
        sum += ss.warp_sums[warpgroup][warp * BLOCK_L + wg_tid];
      }
      const int position = l0 + wg_tid;
      if (position < p.width) {
        const bool valid = position < reinterpret_cast<const int64_t*>(p.lens)[b];
        reinterpret_cast<float*>(p.out)[static_cast<int64_t>(b) * p.out_stride + position] =
            valid ? static_cast<float>(bf16(sum)) : -INFINITY;
      }
    }
    __syncthreads();
  }

  static __device__ __forceinline__ void devfunc(const Sm90Fp4GroupedIndexerParams& p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
    const int tid = threadIdx.x;
    const int warpgroup = tid / 128;
    const int wg_tid = tid % 128;
    const int chunk_start = blockIdx.x * (BLOCK_L * TILES_PER_CTA);
    const int b0 = blockIdx.y * p.group_size;
    const int group_rows = min(p.group_size, p.batch_size - b0);

    extern __shared__ char smem_raw[];
    SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_raw);
    Tensor sK = make_tensor(make_smem_ptr(ss.k.data()), SmemLayout{});

    const int64_t* req = reinterpret_cast<const int64_t*>(p.req);
    const int64_t* lens = reinterpret_cast<const int64_t*>(p.lens);
    const int32_t* req_to_token = reinterpret_cast<const int32_t*>(p.req_to_token);
    const uint8_t* table = reinterpret_cast<const uint8_t*>(p.table);
    const int64_t request = req[b0];

    // Group rows can have different visible lengths. Their maximum gives every
    // thread the same skip decision; shorter rows remain masked at the store.
    if (tid == 0) {
      int64_t group_max_len = 0;
      for (int row = 0; row < group_rows; ++row) {
        group_max_len = max(group_max_len, lens[b0 + row]);
      }
      ss.group_max_len = group_max_len;
    }
    __syncthreads();
    const int64_t group_max_len = ss.group_max_len;
    if (chunk_start >= group_max_len) {
      fill_invisible(p, b0, group_rows, chunk_start, BLOCK_L * TILES_PER_CTA, tid);
      return;
    }

    if constexpr (RESIDENT_Q) {
      // Keep the number of barrier arrivals identical across all warpgroups.
      for (int round = 0; round < ceil_div(group_rows, NUM_WARPGROUPS); ++round) {
        const int row = round * NUM_WARPGROUPS + warpgroup;
        const bool active = row < group_rows;
        decode_q(p, ss, b0 + row, active ? row : 0, wg_tid, active);
      }
      __syncthreads();
    }

    for (int tile = 0; tile < TILES_PER_CTA; ++tile) {
      const int l0 = chunk_start + tile * BLOCK_L;
      if (l0 >= p.width) break;
      if (l0 >= group_max_len) {
        fill_invisible(p, b0, group_rows, l0, BLOCK_L, tid);
        continue;
      }
      if (tid < BLOCK_L) {
        const int position = l0 + tid;
        int32_t slot = 0;
        if (position < p.width && position < group_max_len) {
          slot = req_to_token[request * p.req_stride + static_cast<int64_t>(position) * p.ratio] / p.ratio;
        }
        ss.slots[tid] = slot;
      }
      __syncthreads();

      if (tid < BLOCK_L) {
        const int col = tid;
        const int slot = ss.slots[col];
        const int page = slot / p.page_size;
        const int off = slot - page * p.page_size;
        uint8_t max_exponent = 1;
        CUTE_UNROLL
        for (int g = 0; g < SCALE_GROUPS; ++g) {
          const uint8_t exponent = table[static_cast<int64_t>(page) * p.table_stride + p.page_size * 64 + off * 4 + g];
          ss.k_exponents[g][col] = exponent;
          max_exponent = max(max_exponent, exponent);
        }
        const uint8_t common_exponent = max(static_cast<int>(max_exponent) - 12, 1);
        ss.k_scales[col] = ue8m0_to_f32(common_exponent);
      }
      __syncthreads();

      for (int pair = tid; pair < BLOCK_L * (HEAD_DIM / 2); pair += NUM_THREADS) {
        const int col = pair / (HEAD_DIM / 2);
        const int d = (pair % (HEAD_DIM / 2)) * 2;
        const int g = d / SCALE_GROUP_SIZE;
        const int slot = ss.slots[col];
        const int page = slot / p.page_size;
        const int off = slot - page * p.page_size;
        const int packed_col = d / 2;
        const uint8_t packed = table[static_cast<int64_t>(page) * p.table_stride + off * 64 + packed_col];
        const int common_exponent = (__float_as_uint(ss.k_scales[col]) >> 23) & 0xff;
        const int exponent_delta = static_cast<int>(ss.k_exponents[g][col]) - common_exponent;
        fp8 v0, v1;
        *reinterpret_cast<uint8_t*>(&v0) = scaled_e2m1_to_e5m2(packed & 0xf, exponent_delta);
        *reinterpret_cast<uint8_t*>(&v1) = scaled_e2m1_to_e5m2(packed >> 4, exponent_delta);
        sK(col, d) = v0;
        sK(col, d + 1) = v1;
      }
      __syncthreads();

      TiledMMA mma;

      const int rounds = ceil_div(group_rows, NUM_WARPGROUPS);
      for (int round = 0; round < rounds; ++round) {
        const int row_in_group = round * NUM_WARPGROUPS + warpgroup;
        const bool active = row_in_group < group_rows;
        const int b = b0 + row_in_group;

        const int q_slot = RESIDENT_Q ? (active ? row_in_group : 0) : warpgroup;
        Tensor sQ = make_tensor(make_smem_ptr(ss.q[q_slot].data()), SmemLayout{});
        if constexpr (!RESIDENT_Q) {
          decode_q(p, ss, b, q_slot, wg_tid, active);
          __syncthreads();
        }

        Tensor acc = partition_fragment_C(mma, Shape<Int<HEADS>, Int<BLOCK_L>>{});
        if (active) {
          gemm_k128(mma, sQ, sK, acc, wg_tid);
          warpgroup_commit_batch();
          warpgroup_wait<0>();
          warpgroup_fence_operand(acc);
        }

        store_logits(p, ss, ss.k_scales, acc, b, q_slot, l0, active, tid);
      }
    }
#else
    if (cute::thread0()) {
      CUTE_INVALID_CONTROL_PATH("sm90_fp4_grouped_indexer only supports sm90");
    }
#endif
  }

  static void run(const Sm90Fp4GroupedIndexerParams& p) {
    auto kernel = &fp4_grouped_indexer_kernel<Sm90Fp4GroupedIndexerKernel>;
    constexpr size_t smem_size = sizeof(SharedStorage);
    static bool attr_set = [&]() {
      FP4_INDEXER_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      return true;
    }();
    (void)attr_set;
    dim3 grid(ceil_div(p.width, BLOCK_L * TILES_PER_CTA), ceil_div(p.batch_size, p.group_size), 1);
    kernel<<<grid, NUM_THREADS, smem_size, p.stream>>>(p);
    FP4_INDEXER_CUDA_CHECK(cudaGetLastError());
  }
};

// A fixed resident grid walks a prefix of *visible* chunks. The prefix is
// computed on device on every invocation, including graph replay; no host
// length readback, global counter, or request-slot scratch is needed.
template <bool PIPELINED = true, int CHUNK_TILES = 8, int WARPGROUPS = 4>
struct Sm90Fp4PersistentIndexerKernel : Sm90Fp4GroupedIndexerKernel<8, 6, WARPGROUPS> {
  using Base = Sm90Fp4GroupedIndexerKernel<8, 6, WARPGROUPS>;
  using Base::BLOCK_L;
  using Base::decode_q;
  using Base::gemm_k128;
  using Base::HEADS;
  using Base::NUM_THREADS;
  using Base::NUM_WARPGROUPS;
  using Base::store_logits;
  using typename Base::SmemLayout;
  using typename Base::TiledMMA;
  static constexpr int MIN_BLOCKS = 2;
  static constexpr int STAGES = PIPELINED ? 3 : 1;
  static constexpr int MAX_GROUPS = 1024;
  struct Stage {
    array_aligned<uint8_t, BLOCK_L * 64, 128> packed;
    uint8_t exponents[BLOCK_L][4];
    array_aligned<fp8, cosize_v<SmemLayout>, 128> k;
    float scales[BLOCK_L];
  };
  struct SharedStorage : Base::QueryStorage {
    Stage stages[STAGES];
    int visible[MAX_GROUPS];
    int64_t prefix[MAX_GROUPS + 1];
    int chunk_tiles;
  };

  static __device__ __forceinline__ void
  load_packed(const Sm90Fp4GroupedIndexerParams& p, Stage& stage, int64_t request, int l0, int visible, int tid) {
    const auto* map = reinterpret_cast<const int32_t*>(p.req_to_token);
    const auto* table = reinterpret_cast<const uint8_t*>(p.table);
    // Four 16-byte copies per K row. Positions beyond visibility must not
    // dereference req_to_token: padding slots may contain arbitrary sentinels.
    if (tid < BLOCK_L * 4) {
      const int col = tid / 4;
      const int part = tid % 4;
      const bool valid = l0 + col < visible;
      int slot = 0;
      if (valid) slot = map[request * p.req_stride + static_cast<int64_t>(l0 + col) * p.ratio] / p.ratio;
      const int page = slot / p.page_size;
      const int off = slot % p.page_size;
      const uint8_t* src = table + static_cast<int64_t>(page) * p.table_stride + off * 64 + part * 16;
      uint8_t* dst = stage.packed.data() + col * 64 + part * 16;
      if ((reinterpret_cast<uintptr_t>(src) & 15) == 0) {
        const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
        const int bytes = valid ? 16 : 0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(src), "r"(bytes) : "memory");
      } else {
        // Public API accepts a padded, potentially unaligned table stride.
        CUTE_UNROLL
        for (int i = 0; i < 16; ++i)
          dst[i] = valid ? src[i] : 0;
      }
      if (part == 0) {
        CUTE_UNROLL
        for (int g = 0; g < 4; ++g) {
          stage.exponents[col][g] =
              valid ? table[static_cast<int64_t>(page) * p.table_stride + p.page_size * 64 + off * 4 + g] : 127;
        }
      }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }

  static __device__ __forceinline__ void wait_packed() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
  }

  static __device__ __forceinline__ void decode_k(Stage& stage, int tid) {
    if (tid < BLOCK_L) {
      int exponent = 1;
      CUTE_UNROLL
      for (int g = 0; g < 4; ++g)
        exponent = max(exponent, static_cast<int>(stage.exponents[tid][g]));
      stage.scales[tid] = ue8m0_to_f32(max(exponent - 12, 1));
    }
    __syncthreads();
    Tensor sK = make_tensor(make_smem_ptr(stage.k.data()), SmemLayout{});
    for (int pair = tid; pair < BLOCK_L * 64; pair += NUM_THREADS) {
      const int col = pair / 64;
      const int d = pair % 64 * 2;
      const uint8_t packed = stage.packed[pair];
      const int common = (__float_as_uint(stage.scales[col]) >> 23) & 255;
      const int delta = static_cast<int>(stage.exponents[col][d / 32]) - common;
      fp8 v0, v1;
      *reinterpret_cast<uint8_t*>(&v0) = scaled_e2m1_to_e5m2(packed & 15, delta);
      *reinterpret_cast<uint8_t*>(&v1) = scaled_e2m1_to_e5m2(packed >> 4, delta);
      sK(col, d) = v0;
      sK(col, d + 1) = v1;
    }
    __syncthreads();
  }

  static __device__ __forceinline__ void devfunc(const Sm90Fp4GroupedIndexerParams& p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
    const int tid = threadIdx.x;
    const int wg = tid / 128;
    const int wg_tid = tid % 128;
    extern __shared__ char smem_raw[];
    auto& ss = *reinterpret_cast<SharedStorage*>(smem_raw);
    const auto* lens = reinterpret_cast<const int64_t*>(p.lens);
    const auto* req = reinterpret_cast<const int64_t*>(p.req);
    const int groups = ceil_div(p.batch_size, p.group_size);
    for (int g = tid; g < groups; g += NUM_THREADS) {
      int64_t visible = 0;
      for (int b = g * p.group_size; b < min((g + 1) * p.group_size, p.batch_size); ++b) {
        visible = max(visible, lens[b]);
      }
      ss.visible[g] = static_cast<int>(min(visible, static_cast<int64_t>(p.width)));
    }
    __syncthreads();
    if (tid == 0) {
      int64_t total_tiles = 0;
      for (int g = 0; g < groups; ++g)
        total_tiles += ceil_div(ss.visible[g], BLOCK_L);
      ss.chunk_tiles = min(CHUNK_TILES, total_tiles < 128 ? 1 : total_tiles < 256 ? 2 : total_tiles < 1024 ? 4 : 8);
      ss.prefix[0] = 0;
      for (int g = 0; g < groups; ++g) {
        ss.prefix[g + 1] = ss.prefix[g] + ceil_div(ceil_div(ss.visible[g], BLOCK_L), ss.chunk_tiles);
      }
    }
    __syncthreads();
    // Partition mask-only writes among CTAs, independently of compute tasks.
    for (int g = blockIdx.x % groups; g < groups; g += gridDim.x) {
      const int b0 = g * p.group_size;
      const int rows = min(p.group_size, p.batch_size - b0);
      const int tail_start = min(ceil_div(ss.visible[g], BLOCK_L) * BLOCK_L, p.width);
      const int tail_width = p.width - tail_start;
      const int stripes = ceil_div(static_cast<int>(gridDim.x) - g, groups);
      for (int64_t i = static_cast<int64_t>(blockIdx.x / groups) * NUM_THREADS + tid;
           i < static_cast<int64_t>(rows) * tail_width;
           i += static_cast<int64_t>(max(stripes, 1)) * NUM_THREADS) {
        reinterpret_cast<float*>(
            p.out)[static_cast<int64_t>(b0 + i / tail_width) * p.out_stride + tail_start + i % tail_width] = -INFINITY;
      }
    }
    const int chunk_tiles = ss.chunk_tiles;
    for (int g = 0; g < groups; ++g) {
      const int64_t prefix = ss.prefix[g];
      const int64_t chunks = ss.prefix[g + 1] - prefix;
      const int first = (blockIdx.x + gridDim.x - prefix % gridDim.x) % gridDim.x;
      if (first >= chunks) continue;
      const int b0 = g * p.group_size;
      const int rows = min(p.group_size, p.batch_size - b0);
      const int visible = ss.visible[g];
      const int tiles = ceil_div(visible, BLOCK_L);
      for (int round = 0; round < ceil_div(rows, NUM_WARPGROUPS); ++round) {
        const int row = round * NUM_WARPGROUPS + wg;
        decode_q(p, ss, b0 + row, row < rows ? row : 0, wg_tid, row < rows);
      }
      __syncthreads();
      const int64_t request = req[b0];
      for (int chunk = first; chunk < chunks; chunk += gridDim.x) {
        const int tile0 = chunk * chunk_tiles;
        const int count = min(chunk_tiles, tiles - tile0);
        load_packed(p, ss.stages[0], request, tile0 * BLOCK_L, visible, tid);
        wait_packed();
        decode_k(ss.stages[0], tid);
        if constexpr (PIPELINED) {
          if (count > 1) load_packed(p, ss.stages[1], request, (tile0 + 1) * BLOCK_L, visible, tid);
          wait_packed();
        }
        for (int tile = 0; tile < count; ++tile) {
          Stage& current = ss.stages[tile % STAGES];
          Tensor sK = make_tensor(make_smem_ptr(current.k.data()), SmemLayout{});
          TiledMMA mma;
          for (int round = 0; round < ceil_div(rows, NUM_WARPGROUPS); ++round) {
            Tensor acc = partition_fragment_C(mma, Shape<Int<HEADS>, Int<BLOCK_L>>{});
            const int row = round * NUM_WARPGROUPS + wg;
            const bool active = row < rows;
            const int q_slot = active ? row : 0;
            Tensor sQ = make_tensor(make_smem_ptr(ss.q[q_slot].data()), SmemLayout{});
            gemm_k128(mma, sQ, sK, acc, wg_tid);
            warpgroup_commit_batch();
            if constexpr (PIPELINED) {
              // While WGMMA consumes tile i, cp.async loads packed tile i+2
              // and CUDA cores decode tile i+1 into a distinct stage.
              if (round == 0) {
                if (tile + 2 < count) {
                  load_packed(p, ss.stages[(tile + 2) % STAGES], request, (tile0 + tile + 2) * BLOCK_L, visible, tid);
                }
                if (tile + 1 < count) decode_k(ss.stages[(tile + 1) % STAGES], tid);
              }
            }
            warpgroup_wait<0>();
            warpgroup_fence_operand(acc);
            store_logits(p, ss, current.scales, acc, b0 + row, q_slot, (tile0 + tile) * BLOCK_L, active, tid);
          }
          if constexpr (PIPELINED) {
            wait_packed();
          } else if (tile + 1 < count) {
            load_packed(p, current, request, (tile0 + tile + 1) * BLOCK_L, visible, tid);
            wait_packed();
            decode_k(current, tid);
          }
        }
      }
      __syncthreads();
    }
#else
    if (cute::thread0()) CUTE_INVALID_CONTROL_PATH("sm90_fp4_grouped_indexer only supports sm90");
#endif
  }

  static void run(const Sm90Fp4GroupedIndexerParams& p) {
    if (p.group_size > 6 || ceil_div(p.batch_size, p.group_size) > MAX_GROUPS) {
      Sm90Fp4GroupedIndexerKernel<>::run(p);
      return;
    }
    auto kernel = &fp4_grouped_indexer_kernel<Sm90Fp4PersistentIndexerKernel>;
    constexpr size_t smem_size = sizeof(SharedStorage);
    FP4_INDEXER_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    int device, sms;
    FP4_INDEXER_CUDA_CHECK(cudaGetDevice(&device));
    FP4_INDEXER_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const int64_t capacity_chunks =
        static_cast<int64_t>(ceil_div(p.width, BLOCK_L * CHUNK_TILES)) * ceil_div(p.batch_size, p.group_size);
    const int blocks = static_cast<int>(std::min<int64_t>(2 * sms, capacity_chunks));
    if (blocks == 0) return;
    kernel<<<blocks, NUM_THREADS, smem_size, p.stream>>>(p);
    FP4_INDEXER_CUDA_CHECK(cudaGetLastError());
  }
};

template <typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, Kernel::MIN_BLOCKS)
    fp4_grouped_indexer_kernel(__grid_constant__ const Sm90Fp4GroupedIndexerParams params) {
  Kernel::devfunc(params);
}

inline void run_sm90_fp4_grouped_indexer(const Sm90Fp4GroupedIndexerParams& params) {
  // H20 measurements: amortize Q decoding only when there is enough tile
  // parallelism. A fixed eight-tile chunk severely underfills small batches.
  const int tiles_per_group = ceil_div(params.width, 64);
  const int64_t total_tiles = static_cast<int64_t>(tiles_per_group) * ceil_div(params.batch_size, params.group_size);
  if (params.group_size <= 6 && tiles_per_group >= 2 && total_tiles >= 128) {
    if (tiles_per_group >= 8 && total_tiles >= 1024) {
      Sm90Fp4GroupedIndexerKernel<8, 6>::run(params);
    } else if (tiles_per_group >= 4 && total_tiles >= 256) {
      Sm90Fp4GroupedIndexerKernel<4, 6>::run(params);
    } else {
      Sm90Fp4GroupedIndexerKernel<2, 6>::run(params);
    }
    return;
  }
  Sm90Fp4GroupedIndexerKernel<>::run(params);
}

}  // namespace fp4_grouped_indexer_sm90
}  // namespace sglang
