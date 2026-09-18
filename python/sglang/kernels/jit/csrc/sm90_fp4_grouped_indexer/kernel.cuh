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
// One CTA owns one request group and 64 compressed positions. The packed FP4 K
// tile is decoded once and reused by every query row in the group.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>

#include "params.h"
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

#define FP4_INDEXER_CUDA_CHECK(call)                                                        \
  do {                                                                                      \
    cudaError_t err = (call);                                                               \
    if (err != cudaSuccess) {                                                               \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                                                              \
    }                                                                                       \
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

struct Sm90Fp4GroupedIndexerKernel {
  static constexpr int HEADS = 64;
  static constexpr int HEAD_DIM = 128;
  static constexpr int BLOCK_L = 64;
  static constexpr int SCALE_GROUPS = 4;
  static constexpr int SCALE_GROUP_SIZE = 32;
  static constexpr int NUM_WARPGROUPS = 4;
  static constexpr int WARPS_PER_WARPGROUP = 4;
  static constexpr int NUM_THREADS = 128 * NUM_WARPGROUPS;

  using SmemLayout = decltype(
      tile_to_shape(GMMA::Layout_K_SW64_Atom<fp8>{}, Shape<Int<64>, Int<128>>{}, Step<_1, _2>{}));
  using TiledMMA =
      decltype(make_tiled_mma(GMMA::MMA_64x64x32_F32E5M2E5M2_SS_TN<>{}, Layout<Shape<_1, _1, _1>>{}));

  struct SharedStorage {
    array_aligned<fp8, cosize_v<SmemLayout>, 128> q[NUM_WARPGROUPS];
    array_aligned<fp8, cosize_v<SmemLayout>, 128> k;
    array_aligned<float, WARPS_PER_WARPGROUP * BLOCK_L, 128> warp_sums[NUM_WARPGROUPS];
    int32_t slots[BLOCK_L];
    uint8_t k_exponents[SCALE_GROUPS][BLOCK_L];
    float k_scales[BLOCK_L];
    float q_scales[NUM_WARPGROUPS][HEADS];
  };

  template <typename TA, typename TB, typename TC>
  static __device__ __forceinline__ void gemm_k128(
      TiledMMA& mma, TA const& sQ, TB const& sK, TC& acc, int tid) {
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

  static __device__ __forceinline__ void devfunc(const Sm90Fp4GroupedIndexerParams& p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
    const int tid = threadIdx.x;
    const int warpgroup = tid / 128;
    const int wg_tid = tid % 128;
    const int l0 = blockIdx.x * BLOCK_L;
    const int b0 = blockIdx.y * p.group_size;
    const int group_rows = min(p.group_size, p.batch_size - b0);

    extern __shared__ char smem_raw[];
    SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_raw);
    Tensor sQ = make_tensor(make_smem_ptr(ss.q[warpgroup].data()), SmemLayout{});
    Tensor sK = make_tensor(make_smem_ptr(ss.k.data()), SmemLayout{});

    const int64_t* req = reinterpret_cast<const int64_t*>(p.req);
    const int64_t* lens = reinterpret_cast<const int64_t*>(p.lens);
    const int32_t* req_to_token = reinterpret_cast<const int32_t*>(p.req_to_token);
    const uint8_t* table = reinterpret_cast<const uint8_t*>(p.table);
    const int64_t request = req[b0];

    if (tid < BLOCK_L) {
      const int position = l0 + tid;
      int32_t slot = 0;
      if (position < p.width) {
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
        const uint8_t exponent =
            table[static_cast<int64_t>(page) * p.table_stride + p.page_size * 64 + off * 4 + g];
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
      const uint8_t packed =
          table[static_cast<int64_t>(page) * p.table_stride + off * 64 + packed_col];
      const int common_exponent = (__float_as_uint(ss.k_scales[col]) >> 23) & 0xff;
      const int exponent_delta =
          static_cast<int>(ss.k_exponents[g][col]) - common_exponent;
      fp8 v0, v1;
      *reinterpret_cast<uint8_t*>(&v0) =
          scaled_e2m1_to_e5m2(packed & 0xf, exponent_delta);
      *reinterpret_cast<uint8_t*>(&v1) =
          scaled_e2m1_to_e5m2(packed >> 4, exponent_delta);
      sK(col, d) = v0;
      sK(col, d + 1) = v1;
    }
    __syncthreads();

    const uint8_t* q = reinterpret_cast<const uint8_t*>(p.q);
    const uint32_t* q_scale = reinterpret_cast<const uint32_t*>(p.q_scale);
    const bf16* weights = reinterpret_cast<const bf16*>(p.weights);
    float* out = reinterpret_cast<float*>(p.out);
    TiledMMA mma;

    const int rounds = ceil_div(group_rows, NUM_WARPGROUPS);
    for (int round = 0; round < rounds; ++round) {
      const int row_in_group = round * NUM_WARPGROUPS + warpgroup;
      const bool active = row_in_group < group_rows;
      const int b = b0 + row_in_group;

      if (active && wg_tid < HEADS) {
        const uint32_t packed_scale =
            q_scale[static_cast<int64_t>(b) * p.q_scale_stride_b + wg_tid];
        uint8_t max_exponent = 1;
        CUTE_UNROLL
        for (int g = 0; g < SCALE_GROUPS; ++g) {
          max_exponent =
              max(max_exponent, static_cast<uint8_t>(packed_scale >> (8 * g)));
        }
        const uint8_t common_exponent = max(static_cast<int>(max_exponent) - 12, 1);
        ss.q_scales[warpgroup][wg_tid] = ue8m0_to_f32(common_exponent);
      }
      __syncthreads();

      if (active) {
        for (int pair = wg_tid; pair < HEADS * (HEAD_DIM / 2); pair += 128) {
          const int head = pair / (HEAD_DIM / 2);
          const int d = (pair % (HEAD_DIM / 2)) * 2;
          const int g = d / SCALE_GROUP_SIZE;
          const uint8_t packed =
              q[static_cast<int64_t>(b) * p.q_stride_b +
                static_cast<int64_t>(head) * p.q_stride_h + d / 2];
          const uint32_t packed_scale =
              q_scale[static_cast<int64_t>(b) * p.q_scale_stride_b + head];
          const int exponent = static_cast<uint8_t>(packed_scale >> (8 * g));
          const int common_exponent =
              (__float_as_uint(ss.q_scales[warpgroup][head]) >> 23) & 0xff;
          const int exponent_delta = exponent - common_exponent;
          fp8 v0, v1;
          *reinterpret_cast<uint8_t*>(&v0) =
              scaled_e2m1_to_e5m2(packed & 0xf, exponent_delta);
          *reinterpret_cast<uint8_t*>(&v1) =
              scaled_e2m1_to_e5m2(packed >> 4, exponent_delta);
          sQ(head, d) = v0;
          sQ(head, d + 1) = v1;
        }
      }
      __syncthreads();

      Tensor acc = partition_fragment_C(mma, Shape<Int<HEADS>, Int<BLOCK_L>>{});
      if (active) {
        gemm_k128(mma, sQ, sK, acc, wg_tid);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(acc);
      }

      if (active) {
        // Lanes with the same lane % 4 own the same 16 columns. Their two
        // accumulator rows cover one contiguous 16-head slice.
        const int warp = wg_tid / 32;
        const int lane = wg_tid % 32;
        const int head_in_warp = lane / 4;
        const int head0 = warp * 16 + head_in_warp;
        const int head1 = head0 + 8;
        const float q_scale0 = ss.q_scales[warpgroup][head0];
        const float q_scale1 = ss.q_scales[warpgroup][head1];
        const float weight0 =
            static_cast<float>(weights[static_cast<int64_t>(b) * p.weight_stride_b + head0]);
        const float weight1 =
            static_cast<float>(weights[static_cast<int64_t>(b) * p.weight_stride_b + head1]);

        CUTE_UNROLL
        for (int j = 0; j < BLOCK_L / 8; ++j) {
          CUTE_UNROLL
          for (int cp = 0; cp < 2; ++cp) {
            const int col = (lane % 4) * 2 + 8 * j + cp;
            const float k_scale = ss.k_scales[col];
            const float score0 =
                fmaxf(static_cast<float>(bf16(acc(j * 4 + cp) * q_scale0 * k_scale)), 0.0f);
            const float score1 =
                fmaxf(static_cast<float>(bf16(acc(j * 4 + 2 + cp) * q_scale1 * k_scale)), 0.0f);
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
          const bool valid = position < lens[b];
          out[static_cast<int64_t>(b) * p.out_stride + position] =
              valid ? static_cast<float>(bf16(sum)) : -INFINITY;
        }
      }
      __syncthreads();
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
    dim3 grid(ceil_div(p.width, BLOCK_L), ceil_div(p.batch_size, p.group_size), 1);
    kernel<<<grid, NUM_THREADS, smem_size, p.stream>>>(p);
    FP4_INDEXER_CUDA_CHECK(cudaGetLastError());
  }
};

template <typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS)
    fp4_grouped_indexer_kernel(__grid_constant__ const Sm90Fp4GroupedIndexerParams params) {
  Kernel::devfunc(params);
}

inline void run_sm90_fp4_grouped_indexer(const Sm90Fp4GroupedIndexerParams& params) {
  Sm90Fp4GroupedIndexerKernel::run(params);
}

}  // namespace fp4_grouped_indexer_sm90
}  // namespace sglang
