// Adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh
/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <array>
#include <cstdint>
#include <cub/cub.cuh>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <optional>
#include <string>
#include <vector>

#include "trt_fp8_blockscale_mma_utils.cuh"
#include "trt_fp8_blockscale_tma_utils.cuh"
#include "utils.h"

namespace kernel_utils {

inline void find_divisor(uint32_t& mul, uint32_t& shr, int x) {
  auto find_log_2 = [](int x, bool round_up = false) {
    auto clz = [](int x) {
      for (int i = 31; i >= 0; --i) {
        if ((1 << i) & x) {
          return 31 - i;
        }
      }
      return 32;
    };

    int a = 31 - clz(x);
    if (round_up) {
      a += (x & (x - 1)) ? 1 : 0;
    }
    return a;
  };

  assert(x != 0);
  if (x == 1) {
    // If dividing by 1, reduced math doesn't work because mul_coeff would need
    // to be 2^32, which doesn't fit into unsigned int.  the div() routine
    // handles this special case separately.
    mul = 0;
    shr = 0;
  } else {
    // To express the division N/D in terms of a multiplication, what we first
    // imagine is simply N*(1/D).  However, 1/D will always evaluate to 0 (for
    // D>1), so we need another way.  There's nothing that says we have to use
    // exactly the fraction 1/D; instead it could be any X/Y that reduces to 1/D
    // (i.e., Y=X*D), or at least to "close enough" to it.  If we pick Y that is
    // a power of two, then the N*(X/Y) can be N*X followed by a right-shift by
    // some amount. The power of two we should pick should be at least 2^32,
    // because in the div() routine we'll use umulhi(), which returns only the
    // upper 32 bits -- this being equivalent to a right-shift by 32.  But we
    // might want a higher power of two for better accuracy depending on the
    // magnitude of the denominator. Once we've picked Y, then X [our mul_coeff
    // value] is simply Y/D, rounding up, and we save shift_coeff as whatever
    // further shift we have to do beyond what the umulhi() implies.
    uint32_t p = 31 + find_log_2(x, true);
    uint32_t m = (uint32_t)(((1ull << p) + (uint32_t)x - 1) / (uint32_t)x);

    mul = m;
    shr = p - 32;
  }
}

__device__ __forceinline__ void fast_divmod(uint32_t& div, uint32_t& mod, int x, int y, uint32_t mul, uint32_t shr) {
  if (y == 1) {
    div = x;
    mod = 0;
  } else {
    div = __umulhi((uint32_t)x, mul) >> shr;
    mod = x - div * y;
  }
}

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
  constexpr uint32_t FINAL_MASK = 0xffffffff;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

template <>
__inline__ __device__ __nv_bfloat16 warpReduceSum(__nv_bfloat16 val) {
  constexpr uint32_t FINAL_MASK = 0xffffffff;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = __hmax(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

__inline__ __device__ uint32_t elect_one_sync([[maybe_unused]] int lane_id) {
  uint32_t pred = 0;
#if __CUDA_ARCH__ >= 900
  uint32_t laneid = 0;
  asm volatile(
      "\n\
     {\n\
         .reg .b32 %rx;\n\
         .reg .pred %px;\n\
         elect.sync %rx|%px, %2;\n\
         @%px mov.s32 %1, 1;\n\
         mov.s32 %0, %rx;\n\
     }\n\
   "
      : "+r"(laneid), "+r"(pred)
      : "r"(0xFFFFFFFF));
#else
  return lane_id == 0;
#endif
  return pred;
}

}  // namespace kernel_utils

namespace tensorrt_llm::kernels::fp8_blockscale_gemm {

template <typename T>
__device__ __host__ constexpr T div_up(T a, int b) {
  return (a + b - 1) / b;
}

using TileShape = std::tuple<uint32_t, uint32_t, uint32_t>;
enum class Layout { RowMajor, ColMajor };
enum class ScaleType { PerTensor, PerBlock, PerChannel, PerSubChannel };

template <int TILE_M, int TILE_N>
struct GroupedGemmProblemVisitor {
  struct Input {
    int64_t const* problem_m_offsets;
    int64_t const* problem_m_padded_offsets;
  };

  static __host__ __device__ dim3 grid_dim(int shape_m, int shape_n, int num_problems) {
    return dim3(div_up(shape_m, TILE_M), div_up(shape_n, TILE_N), num_problems);
  }

  static __device__ int tile_m_idx() {
    return blockIdx.x;
  }

  static __device__ int tile_n_idx() {
    return blockIdx.y;
  }

  static __device__ int problem_idx() {
    return blockIdx.z;
  }

  static __device__ int m_offset(Input const& input) {
    int problem_idx_ = problem_idx();
    return input.problem_m_offsets[problem_idx_];
  }

  static __device__ int m_padded_offset(Input const& input) {
    int problem_idx_ = problem_idx();
    return input.problem_m_padded_offsets[problem_idx_];
  }

  static __device__ int n_offset(Input const& input) {
    int problem_idx_ = problem_idx();
    return problem_idx_ * TILE_N * gridDim.y;
  }

  static __device__ int m_boundary(Input const& input) {
    int problem_idx_ = problem_idx();
    return input.problem_m_offsets[problem_idx_ + 1] - input.problem_m_offsets[problem_idx_];
  }
};

template <int TILE_M, int TILE_N>
struct PlainGemmProblemVisitor {
  struct Input {
    int shape_m;
  };

  static __host__ __device__ dim3 grid_dim(int shape_m, int shape_n) {
    return dim3(div_up(shape_m, TILE_M), div_up(shape_n, TILE_N));
  }

  static __device__ int tile_m_idx() {
    return blockIdx.x;
  }

  static __device__ int tile_n_idx() {
    return blockIdx.y;
  }

  static __device__ int problem_idx() {
    return 0;
  }

  static __device__ int m_offset(Input const& input) {
    return 0;
  }

  static __device__ int n_offset(Input const& input) {
    return 0;
  }

  static __device__ int m_boundary(Input const& input) {
    return input.shape_m;
  }
};

template <int TILE_M, int TILE_N>
struct StridedBatchedGemmProblemVisitor {
  struct Input {
    int shape_m;
    int ld_a;
    int stride_a;
    int ld_b;
    int stride_b;
    int stride_d;
    int stride_scales_a;
    // stride_a % ld_a must be 0
    // stride_b % ld_b must be 0
  };

  static __host__ __device__ dim3 grid_dim(int shape_m, int shape_n, int num_problems) {
    return dim3(div_up(shape_m, TILE_M), div_up(shape_n, TILE_N), num_problems);
  }

  static __device__ int tile_m_idx() {
    return blockIdx.x;
  }

  static __device__ int tile_n_idx() {
    return blockIdx.y;
  }

  static __device__ int problem_idx() {
    return blockIdx.z;
  }

  static __device__ int m_offset(Input const& input) {
    int problem_idx_ = problem_idx();
    return input.stride_a / input.ld_a * problem_idx_;
  }

  static __device__ int n_offset(Input const& input) {
    int problem_idx_ = problem_idx();
    return input.stride_b / input.ld_b * problem_idx_;
  }

  static __device__ int m_boundary(Input const& input) {
    return input.shape_m;
  }
};

namespace cde = cuda::device::experimental;

template <
    typename ProblemVisitor,
    typename ElementA,
    typename ElementB,
    typename ElementD,
    Layout LayoutD,
    typename WGMMA_OP,
    int TILE_M,
    int TILE_N,
    int TILE_K,
    int NUM_STAGES,
    bool IsPersistentKernel = false>
__global__ void __launch_bounds__(TILE_M == 64 ? 256 : 384, 1) cooperative_1x128_by_128x128_fp8_gemm_kernel(
    ElementD* gmem_d,
    int ld_d,
    float const* scales_b,
    typename ProblemVisitor::Input problem_input,
    int shape_n,
    int shape_k,
    __grid_constant__ const CUtensorMap tensor_map_a,
    __grid_constant__ const CUtensorMap tensor_map_b,
    __grid_constant__ const CUtensorMap tensor_map_scales_a,
    int guessed_m) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  static_assert(sizeof(ElementA) == 1 && sizeof(ElementB) == 1);
  static_assert(TILE_K == 128);
  constexpr int ScaleGranMA = 1;
  constexpr int ScaleGranKA = 128;
  constexpr int ScaleGranNB = 128;
  constexpr int ScaleGranKB = 128;
  static_assert(TILE_K % ScaleGranKA == 0);

  static constexpr int SMEM_A_SIZE_PER_STAGE = TILE_M * TILE_K * sizeof(ElementA);
  static constexpr int SMEM_B_SIZE_PER_STAGE = TILE_N * TILE_K * sizeof(ElementB);
  static constexpr int SMEM_SCALES_A_SIZE_PER_STAGE =
      div_up(TILE_M, ScaleGranMA) * div_up(TILE_K, ScaleGranKA) * sizeof(float);
  static constexpr bool IS_UNIFORM_SCALE_B = ScaleGranNB % TILE_N == 0;

  constexpr int BLOCK_SIZE = TILE_M == 64 ? 256 : 384;
  constexpr int TMA_ISSUE_INTERVAL = 1;
  using Barrier = cuda::barrier<cuda::thread_scope_block>;

  int tile_m_idx = ProblemVisitor::tile_m_idx();
  int m_boundary = ProblemVisitor::m_boundary(problem_input);
  if (tile_m_idx * TILE_M >= m_boundary) return;

  int tile_n_idx = ProblemVisitor::tile_n_idx();
  int problem_idx = ProblemVisitor::problem_idx();
  int problem_m_offset = ProblemVisitor::m_offset(problem_input);
  int problem_m_padded_offset = 0;
  if constexpr (std::is_same_v<ProblemVisitor, GroupedGemmProblemVisitor<TILE_M, TILE_N>>) {
    problem_m_padded_offset = ProblemVisitor::m_padded_offset(problem_input);
  }
  int problem_n_offset = ProblemVisitor::n_offset(problem_input);

  int scales_b_ld = ScaleGranKB != 0 ? div_up(shape_k, ScaleGranKB) : 1;
  scales_b += problem_idx * div_up(shape_n, ScaleGranNB) * scales_b_ld;

  int iters_in_former_scales_b = TILE_N / 8;  // assuming divisible
  if constexpr (ScaleGranNB != 0) {
    scales_b += ((tile_n_idx * TILE_N) / ScaleGranNB) * scales_b_ld;
    iters_in_former_scales_b =
        min(TILE_N, ScaleGranNB - (tile_n_idx * TILE_N) % ScaleGranNB) / 8;  // assuming divisible
  }

  // Align to 1024 byte for swizzle-128B
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  ElementA* smem_a[NUM_STAGES];
  ElementB* smem_b[NUM_STAGES];
  float* smem_scales_a[NUM_STAGES];

  Barrier* full_bars[NUM_STAGES];
  // NUM_EMPTY_BARS must be a const expression, otherwise it will cost too many registers.
  constexpr int NUM_EMPTY_BARS = div_up(NUM_STAGES, TMA_ISSUE_INTERVAL);
  Barrier* empty_bars[NUM_EMPTY_BARS];

  float* smem_scales_b;

  for (int i = 0; i < NUM_STAGES; i++) {
    smem_a[i] = reinterpret_cast<ElementA*>(smem_buffer + i * SMEM_A_SIZE_PER_STAGE);
    smem_b[i] =
        reinterpret_cast<ElementB*>(smem_buffer + NUM_STAGES * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    smem_scales_a[i] = reinterpret_cast<float*>(
        smem_buffer + NUM_STAGES * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) + i * SMEM_SCALES_A_SIZE_PER_STAGE);
    full_bars[i] = reinterpret_cast<Barrier*>(
        smem_buffer + NUM_STAGES * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE) +
        i * sizeof(Barrier));
  }
  for (int i = 0; i < NUM_EMPTY_BARS; i++) {
    empty_bars[i] = i ? empty_bars[i - 1] + 1 : full_bars[NUM_STAGES - 1] + 1;
  }
  smem_scales_b = reinterpret_cast<float*>(empty_bars[NUM_EMPTY_BARS - 1] + 1);

  int lane_predicate = cute::elect_one_sync();
  if (threadIdx.x < 32 && lane_predicate == 1) {
    cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_a));
    cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_b));
    cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_scales_a));

    for (int i = 0; i < NUM_STAGES; i++) {
      init(full_bars[i], 1);
    }
    for (int i = 0; i < NUM_EMPTY_BARS; i++) {
      init(empty_bars[i], BLOCK_SIZE - 128);
    }
    cutlass::arch::fence_view_async_shared();
  }
  int math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128 - 1, 0);

  float scale_b_r, scale_b_r_second_part;
  if constexpr (ScaleGranKB != 0) {
    int end_index = !IS_UNIFORM_SCALE_B && iters_in_former_scales_b < TILE_N / 8 ? scales_b_ld * 2 : scales_b_ld;
#pragma unroll
    for (int i = threadIdx.x; i < end_index; i += BLOCK_SIZE) {
      float gmem_scale_b = __ldg(scales_b + i);
      asm volatile("st.shared.f32 [%0], %1;" ::"l"(smem_scales_b + i), "f"(gmem_scale_b));
    }
  } else {
    scale_b_r = scales_b[0];
  }

  __syncthreads();

  while (true) {
    constexpr int NUM_ACCUMS = WGMMA_OP::NUM_ACCUM;
    float accum[NUM_ACCUMS] = {0};
    float final_accum[NUM_ACCUMS] = {0};
    constexpr int K_PER_ITER = NUM_STAGES * TILE_K;

    if (threadIdx.x < 128) {
      for (int k_iter = 0; k_iter < div_up(shape_k, K_PER_ITER); k_iter++) {
        auto copy_func = [&](Barrier& empty_bar, int stage_range_start, int stage_range_end) {
          empty_bar.wait_parity(k_iter + 1 & 1);
          for (int i = stage_range_start; i < stage_range_end; i++) {
            auto& bar = *full_bars[i];
            int k_idx = k_iter * K_PER_ITER + i * TILE_K;
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                smem_a[i], &tensor_map_a, k_idx, tile_m_idx * TILE_M + problem_m_offset, bar);
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                smem_b[i], &tensor_map_b, k_idx, tile_n_idx * TILE_N + problem_n_offset, bar);
            if constexpr (std::is_same_v<ProblemVisitor, StridedBatchedGemmProblemVisitor<TILE_M, TILE_N>>) {
              int scale_y_offset =
                  problem_idx * (problem_input.stride_scales_a / (div_up(problem_input.shape_m, 4) * 4));
              // The scales has been aligned to 16 bytes
              cde::cp_async_bulk_tensor_2d_global_to_shared(
                  smem_scales_a[i],
                  &tensor_map_scales_a,
                  (tile_m_idx * TILE_M) / ScaleGranMA,
                  scale_y_offset + k_idx / ScaleGranKA,
                  bar);
            } else {
              // The scales has been aligned to 16 bytes
              cde::cp_async_bulk_tensor_2d_global_to_shared(
                  smem_scales_a[i],
                  &tensor_map_scales_a,
                  (problem_m_padded_offset + tile_m_idx * TILE_M) / ScaleGranMA,
                  k_idx / ScaleGranKA,
                  bar);
            }
          }
          for (int i = stage_range_start; i < stage_range_end; i++) {
            auto no_use = mbarrier_arrive_1_expect_tx_cta(
                full_bars[i], SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE);
          }
        };
        if (threadIdx.x == 0) {
          int num_stages = div_up((shape_k - k_iter * K_PER_ITER), TILE_K);
          for (int i = 0; i < NUM_EMPTY_BARS; i++) {
            int range_start = i * TMA_ISSUE_INTERVAL;
            int range_end = (i + 1) * TMA_ISSUE_INTERVAL;
            range_end = range_end > NUM_STAGES ? NUM_STAGES : range_end;
            range_end = range_end > num_stages ? num_stages : range_end;
            copy_func(*empty_bars[i], range_start, range_end);
          }
        }
      }
    } else {
      int thr_id_in_wg = threadIdx.x % 128;
      int base_r = thr_id_in_wg / 32 * 16 + thr_id_in_wg % 32 / 4;
      int r_0 = base_r + math_wg_idx * WGMMA_OP::M;
      int r_1 = base_r + math_wg_idx * WGMMA_OP::M + 8;

      struct DivisibleK {};

      struct NotDivisibleK {};

      auto mma_func = [&](int k_iter, auto type) {
        constexpr bool K_IS_DIVISIBLE = std::is_same_v<decltype(type), DivisibleK> ? true : false;
        int num_stages;
        if constexpr (K_IS_DIVISIBLE) {
          num_stages = NUM_STAGES;
        } else {
          num_stages = div_up(shape_k % K_PER_ITER, TILE_K);
          num_stages = !num_stages ? NUM_STAGES : num_stages;
        }

#pragma unroll
        for (int s = 0; s < num_stages; s++) {
          if constexpr (ScaleGranKB != 0) {
            asm volatile("ld.shared.f32 %0, [%1];" : "=f"(scale_b_r) : "l"(smem_scales_b));
            if (!IS_UNIFORM_SCALE_B && iters_in_former_scales_b < TILE_N / 8) {
              asm volatile("ld.shared.f32 %0, [%1];" : "=f"(scale_b_r_second_part) : "l"(smem_scales_b + scales_b_ld));
            }
            smem_scales_b++;
          }
          (*full_bars[s]).wait_parity(k_iter & 1);
          for (int _ = 0; _ < NUM_ACCUMS; _++) {
            warpgroup_fence_operand(accum[_]);
          }
          warpgroup_arrive();
          for (int k = 0; k < TILE_K / WGMMA_OP::K; k++) {
            auto desc_a = make_smem_desc(smem_a[s] + math_wg_idx * WGMMA_OP::M * TILE_K + k * WGMMA_OP::K, 1);
            auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA_OP::K, 1);
            WGMMA_OP::wgmma(desc_a, desc_b, accum, k);
          }
          warpgroup_commit_batch();
          for (int _ = 0; _ < NUM_ACCUMS; _++) {
            warpgroup_fence_operand(accum[_]);
          }
          warpgroup_wait<0>();

          float scale_0 = smem_scales_a[s][r_0] * scale_b_r;
          float scale_1 = smem_scales_a[s][r_1] * scale_b_r;

          bool cross_0 = tile_m_idx * TILE_M + r_0 >= m_boundary;
          bool cross_1 = tile_m_idx * TILE_M + r_1 >= m_boundary;

          if (cross_0) {
            scale_0 = 0;
          }
          if (cross_1) {
            scale_1 = 0;
          }

          if constexpr (K_IS_DIVISIBLE) {
            if (s % TMA_ISSUE_INTERVAL == TMA_ISSUE_INTERVAL - 1 || s == NUM_STAGES - 1) {
              int tma_group_idx = s / TMA_ISSUE_INTERVAL;
              auto no_use = (*empty_bars[tma_group_idx]).arrive();
            }
          }

          float scale_0_second_part = smem_scales_a[s][r_0] * scale_b_r_second_part;
          float scale_1_second_part = smem_scales_a[s][r_1] * scale_b_r_second_part;

          if (!IS_UNIFORM_SCALE_B && iters_in_former_scales_b < TILE_N / 8) {
            for (int i = 0; i < iters_in_former_scales_b; i++) {
              final_accum[i * 4 + 0] += scale_0 * accum[i * 4];
              final_accum[i * 4 + 1] += scale_0 * accum[i * 4 + 1];
            }
            for (int i = 0; i < iters_in_former_scales_b; i++) {
              final_accum[i * 4 + 2] += scale_1 * accum[i * 4 + 2];
              final_accum[i * 4 + 3] += scale_1 * accum[i * 4 + 3];
            }

            for (int i = iters_in_former_scales_b; i < WGMMA_OP::NUM_ACCUM / 4; i++) {
              final_accum[i * 4 + 0] += scale_0_second_part * accum[i * 4];
              final_accum[i * 4 + 1] += scale_0_second_part * accum[i * 4 + 1];
            }
            for (int i = iters_in_former_scales_b; i < WGMMA_OP::NUM_ACCUM / 4; i++) {
              final_accum[i * 4 + 2] += scale_1_second_part * accum[i * 4 + 2];
              final_accum[i * 4 + 3] += scale_1_second_part * accum[i * 4 + 3];
            }
          } else {
            for (int i = 0; i < WGMMA_OP::NUM_ACCUM / 4; i++) {
              final_accum[i * 4 + 0] += scale_0 * accum[i * 4];
              final_accum[i * 4 + 1] += scale_0 * accum[i * 4 + 1];
            }
            for (int i = 0; i < WGMMA_OP::NUM_ACCUM / 4; i++) {
              final_accum[i * 4 + 2] += scale_1 * accum[i * 4 + 2];
              final_accum[i * 4 + 3] += scale_1 * accum[i * 4 + 3];
            }
          }
        }
      };

      int num_iterations = div_up(shape_k, K_PER_ITER);
      for (int k_iter = 0; k_iter < num_iterations - 1; k_iter++) {
        mma_func(k_iter, DivisibleK{});
      }
      mma_func(num_iterations - 1, NotDivisibleK{});
    }

    if constexpr (LayoutD == Layout::RowMajor) {
      __syncthreads();
      ElementD* smem_c = reinterpret_cast<ElementD*>(smem_buffer);
      constexpr int SMEM_C_PADDING = 8;

      if (threadIdx.x >= 128) {
        int thr_id_in_wg = threadIdx.x % 128;
        int base_r = thr_id_in_wg / 32 * 16 + thr_id_in_wg % 32 / 4;
        int base_c = thr_id_in_wg % 4 * 2;
        int r_0 = base_r;
        int r_1 = base_r + 8;
        int c_0 = base_c;

        for (int i = 0; i < WGMMA_OP::NUM_ACCUM / 4; i++) {
          int c_1 = c_0 + 1;
          smem_c[(r_0 + math_wg_idx * WGMMA_OP::M) * (TILE_N + SMEM_C_PADDING) + c_0] =
              static_cast<ElementD>(final_accum[i * 4]);
          smem_c[(r_0 + math_wg_idx * WGMMA_OP::M) * (TILE_N + SMEM_C_PADDING) + c_1] =
              static_cast<ElementD>(final_accum[i * 4 + 1]);
          smem_c[(r_1 + math_wg_idx * WGMMA_OP::M) * (TILE_N + SMEM_C_PADDING) + c_0] =
              static_cast<ElementD>(final_accum[i * 4 + 2]);
          smem_c[(r_1 + math_wg_idx * WGMMA_OP::M) * (TILE_N + SMEM_C_PADDING) + c_1] =
              static_cast<ElementD>(final_accum[i * 4 + 3]);
          c_0 += 8;
        }
      }
      __syncthreads();
      ElementD* gmem_d_this_block;
      if constexpr (std::is_same_v<ProblemVisitor, StridedBatchedGemmProblemVisitor<TILE_M, TILE_N>>) {
        gmem_d_this_block = gmem_d + problem_idx * problem_input.stride_d + (tile_m_idx * TILE_M) * ld_d;
      } else {
        gmem_d_this_block = gmem_d + (problem_m_offset + tile_m_idx * TILE_M) * ld_d;
      }
      int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
      int lane_idx = threadIdx.x % 32;
      constexpr int int4_per_tile_line = TILE_N * sizeof(ElementD) / sizeof(int4);
      // assert(shape_n * sizeof(ElementD) % sizeof(int4) == 0)
      int int4_per_global_line = shape_n * sizeof(ElementD) / sizeof(int4);
      constexpr int num_lines = TILE_M;
      constexpr int num_warps = BLOCK_SIZE / 32;
      int4* smem_c_int4 = reinterpret_cast<int4*>(smem_c);
      bool is_last_tile_n = (tile_n_idx + 1) * TILE_N > shape_n;
      int int4_per_line = is_last_tile_n ? int4_per_global_line % int4_per_tile_line : int4_per_tile_line;

      for (int line_idx = warp_idx; line_idx < num_lines; line_idx += num_warps) {
        if (tile_m_idx * TILE_M + line_idx >= m_boundary) {
          break;
        }
        for (int elem_idx = lane_idx; elem_idx < int4_per_line; elem_idx += 32) {
          int4* g_data_addr =
              reinterpret_cast<int4*>(&gmem_d_this_block[line_idx * ld_d + tile_n_idx * TILE_N]) + elem_idx;
          int4* s_data_addr =
              &smem_c_int4
                  [line_idx * (int4_per_tile_line + SMEM_C_PADDING * sizeof(ElementD) / sizeof(int4)) + elem_idx];
          *g_data_addr = *s_data_addr;
        }
        __syncwarp();
      }
    } else if constexpr (LayoutD == Layout::ColMajor) {
    }

    if constexpr (!IsPersistentKernel) {
      return;
    }

    tile_m_idx += guessed_m / TILE_M;
    if (tile_m_idx * TILE_M >= m_boundary) return;

    if (threadIdx.x < 32 && lane_predicate == 1) {
      for (int i = 0; i < NUM_STAGES; i++) {
        full_bars[i]->~Barrier();
        init(full_bars[i], 1);
      }
      for (int i = 0; i < NUM_EMPTY_BARS; i++) {
        empty_bars[i]->~Barrier();
        init(empty_bars[i], BLOCK_SIZE - 128);
      }
      cutlass::arch::fence_view_async_shared();
    }
    __syncthreads();
    smem_scales_b = reinterpret_cast<float*>(empty_bars[NUM_EMPTY_BARS - 1] + 1);
  }
#else
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("This kernel requires SM90a\n");
    asm volatile("trap;");
  }
#endif
}

template <
    typename ElementA,
    Layout LayoutA,
    typename ElementB,
    Layout LayoutB,
    typename ElementD,
    Layout LayoutD,
    typename ElementAccumulator,
    typename ElementCompute,
    typename ElementScalar,
    int TILE_M,
    int TILE_N,
    int TILE_K,
    ScaleType ScaleTypeA,
    ScaleType ScaleTypeB,
    int ScaleGranMA = 0,
    int ScaleGranKA = 0,
    int ScaleGranNB = 0,
    int ScaleGranKB = 0,
    int NUM_OF_STAGES = 0>
class Fp8Gemm {
 public:
  static constexpr int MAX_SHAPE_K = 20480;

 private:
  using Barrier = cuda::barrier<cuda::thread_scope_block>;
  static constexpr int SMEM_A_SIZE_PER_STAGE = TILE_M * TILE_K * sizeof(ElementA);
  static constexpr int SMEM_B_SIZE_PER_STAGE = TILE_N * TILE_K * sizeof(ElementB);
  static constexpr bool IS_UNIFORM_SCALE_B = ScaleGranNB % TILE_N == 0;

 public:
  static constexpr int get_smem_size(int num_stages, int max_shape_k = MAX_SHAPE_K) {
    auto smem_size = num_stages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + sizeof(Barrier) + sizeof(Barrier));

    if constexpr (ScaleTypeA == ScaleType::PerSubChannel) {
      auto scale_smem_size =
          num_stages * div_up(TILE_M, ScaleGranMA) * div_up(TILE_K, ScaleGranKA) * sizeof(ElementScalar);
      smem_size += scale_smem_size;
    }
    if constexpr (ScaleTypeB != ScaleType::PerTensor) {
      auto scale_smem_size = (IS_UNIFORM_SCALE_B ? 1 : 2) * div_up(max_shape_k, ScaleGranKB) * sizeof(ElementScalar);
      smem_size += scale_smem_size;
    }
    return smem_size;
  }

 private:
  static constexpr int get_num_stages() {
    constexpr auto sm90_capacity = 232448;

    if constexpr (get_smem_size(8) <= sm90_capacity) return 8;
    if constexpr (get_smem_size(7) <= sm90_capacity) return 7;
    if constexpr (get_smem_size(6) <= sm90_capacity) return 6;
    if constexpr (get_smem_size(5) <= sm90_capacity) return 5;
    static_assert(get_smem_size(4) <= sm90_capacity, "The required shared memory size is too large");
    return 4;
  }

  static constexpr int NUM_STAGES = NUM_OF_STAGES == 0 ? get_num_stages() : NUM_OF_STAGES;
  static constexpr int BLOCK_SIZE = TILE_M == 64 ? 256 : 384;

 public:
  Fp8Gemm() {
    static_assert(!(ScaleTypeA == ScaleType::PerSubChannel && (ScaleGranMA == 0 || ScaleGranKA == 0)));
    static_assert(TILE_M % ScaleGranMA == 0 && TILE_K % ScaleGranKA == 0);
  }

  // GroupedGemm
  static void
  run(ElementA* gmem_a,
      ElementB* gmem_b,
      ElementD* gmem_d,
      ElementScalar* scales_a,
      ElementScalar const* scales_b,
      int num_problems,
      int64_t const* problem_m_offsets,
      int shape_n,
      int shape_k,
      int max_shape_m,
      cudaStream_t stream = 0,
      int guessed_m = TILE_M,
      int64_t const* problem_m_padded_offsets = nullptr,
      int max_shape_m_padded = 0) {
    using ProblemVisitor = GroupedGemmProblemVisitor<TILE_M, TILE_N>;
    // Need a factory for selecting WGMMA_OP, need to add E5M2 op if needed.
    using WGMMA_OP = typename Fp8MmaSelector<ElementA, ElementB, TILE_N>::Type;
#define Kernel                                  \
  cooperative_1x128_by_128x128_fp8_gemm_kernel< \
      ProblemVisitor,                           \
      ElementA,                                 \
      ElementB,                                 \
      ElementD,                                 \
      LayoutD,                                  \
      WGMMA_OP,                                 \
      TILE_M,                                   \
      TILE_N,                                   \
      TILE_K,                                   \
      NUM_STAGES,                               \
      true>
    assert(shape_n % TILE_N == 0);
    auto tma_a_desc = make_2d_tma_a_desc(gmem_a, max_shape_m, shape_k);
    auto tma_b_desc = make_2d_tma_b_desc(gmem_b, shape_k, num_problems * shape_n);
    auto tma_scales_a_desc = make_2d_tma_scales_a_desc(scales_a, max_shape_m_padded, shape_k);
    static_assert(TILE_N == WGMMA_OP::N);
    guessed_m = div_up(guessed_m, TILE_M) * TILE_M;
    int smem_size = get_smem_size(NUM_STAGES, shape_k);
    cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    typename ProblemVisitor::Input problem_input{problem_m_offsets, problem_m_padded_offsets};
    auto grid_size = ProblemVisitor::grid_dim(guessed_m, shape_n, num_problems);

    Kernel<<<grid_size, BLOCK_SIZE, smem_size, stream>>>(
        gmem_d,
        shape_n,
        scales_b,
        problem_input,
        shape_n,
        shape_k,
        tma_a_desc,
        tma_b_desc,
        tma_scales_a_desc,
        guessed_m);
#undef Kernel
  }

  // PlainGemm
  static void
  run(ElementA* gmem_a,
      int ld_a,
      ElementB* gmem_b,
      int ld_b,
      ElementD* gmem_d,
      int ld_d,
      ElementScalar* scales_a,
      ElementScalar const* scales_b,
      int shape_m,
      int shape_n,
      int shape_k,
      cudaStream_t stream = 0,
      int guessed_m = TILE_M) {
    using ProblemVisitor = PlainGemmProblemVisitor<TILE_M, TILE_N>;
    // Need a factory for selecting WGMMA_OP, need to add E5M2 op if needed.
    using WGMMA_OP = typename Fp8MmaSelector<ElementA, ElementB, TILE_N>::Type;
#define Kernel                                  \
  cooperative_1x128_by_128x128_fp8_gemm_kernel< \
      ProblemVisitor,                           \
      ElementA,                                 \
      ElementB,                                 \
      ElementD,                                 \
      LayoutD,                                  \
      WGMMA_OP,                                 \
      TILE_M,                                   \
      TILE_N,                                   \
      TILE_K,                                   \
      NUM_STAGES,                               \
      true>
    assert(shape_n % TILE_N == 0);
    auto tma_a_desc = make_2d_tma_a_desc(gmem_a, shape_m, shape_k, ld_a * sizeof(*gmem_a));
    auto tma_b_desc = make_2d_tma_b_desc(gmem_b, shape_k, shape_n, ld_b * sizeof(*gmem_b));
    auto tma_scales_a_desc = make_2d_tma_scales_a_desc(scales_a, div_up(shape_m, 4) * 4, shape_k);
    static_assert(TILE_N == WGMMA_OP::N);
    guessed_m = div_up(guessed_m, TILE_M) * TILE_M;
    int smem_size = get_smem_size(NUM_STAGES, shape_k);
    cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    typename ProblemVisitor::Input problem_input{shape_m};
    auto grid_size = ProblemVisitor::grid_dim(guessed_m, shape_n);

    Kernel<<<grid_size, BLOCK_SIZE, smem_size, stream>>>(
        gmem_d, ld_d, scales_b, problem_input, shape_n, shape_k, tma_a_desc, tma_b_desc, tma_scales_a_desc, guessed_m);
#undef Kernel
  }

  // StridedBatchedGemm
  static void
  run(ElementA* gmem_a,
      int ld_a,
      int stride_a,
      ElementB* gmem_b,
      int ld_b,
      int stride_b,
      ElementD* gmem_d,
      int ld_d,
      int stride_d,
      ElementScalar* scales_a,
      int stride_scales_a,
      ElementScalar const* scales_b,
      int shape_m,
      int shape_n,
      int shape_k,
      int num_problems,
      cudaStream_t stream = 0) {
    using ProblemVisitor = StridedBatchedGemmProblemVisitor<TILE_M, TILE_N>;
    // Need a factory for selecting WGMMA_OP, need to add E5M2 op if needed.
    using WGMMA_OP = typename Fp8MmaSelector<ElementA, ElementB, TILE_N>::Type;
#define Kernel                                  \
  cooperative_1x128_by_128x128_fp8_gemm_kernel< \
      ProblemVisitor,                           \
      ElementA,                                 \
      ElementB,                                 \
      ElementD,                                 \
      LayoutD,                                  \
      WGMMA_OP,                                 \
      TILE_M,                                   \
      TILE_N,                                   \
      TILE_K,                                   \
      NUM_STAGES,                               \
      true>
    assert(shape_n % TILE_N == 0);
    auto tma_a_desc = make_2d_tma_a_desc(gmem_a, shape_m * num_problems, shape_k, ld_a * sizeof(*gmem_a));
    auto tma_b_desc = make_2d_tma_b_desc(gmem_b, shape_k, shape_n * num_problems, ld_b * sizeof(*gmem_b));
    auto tma_scales_a_desc = make_2d_tma_scales_a_desc(scales_a, shape_m, shape_k, num_problems);
    static_assert(TILE_N == WGMMA_OP::N);
    typename ProblemVisitor::Input problem_input{shape_m, ld_a, stride_a, ld_b, stride_b, stride_d, stride_scales_a};

    int guessed_m = div_up(shape_m, TILE_M) * TILE_M;
    int smem_size = get_smem_size(NUM_STAGES, shape_k);
    cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    auto grid_size = ProblemVisitor::grid_dim(shape_m, shape_n, num_problems);

    Kernel<<<grid_size, BLOCK_SIZE, smem_size, stream>>>(
        gmem_d, ld_d, scales_b, problem_input, shape_n, shape_k, tma_a_desc, tma_b_desc, tma_scales_a_desc, guessed_m);
#undef Kernel
  }

  template <typename T>
  static CUtensorMap
  make_2d_tma_a_desc(T* global_address, uint64_t gmem_rows, uint64_t gmem_cols, uint64_t global_stride_in_bytes = 0) {
    return make_2d_tma_desc(global_address, LayoutA, gmem_rows, gmem_cols, global_stride_in_bytes, TILE_M, TILE_K);
  }

  template <typename T>
  static CUtensorMap
  make_2d_tma_b_desc(T* global_address, uint64_t gmem_rows, uint64_t gmem_cols, uint64_t global_stride_in_bytes = 0) {
    return make_2d_tma_desc(global_address, LayoutB, gmem_rows, gmem_cols, global_stride_in_bytes, TILE_K, TILE_N);
  }

  template <typename T>
  static CUtensorMap make_2d_tma_scales_a_desc(
      T* global_address,
      uint64_t shape_m,
      uint64_t shape_k,
      int num_problems = 1,
      uint64_t global_stride_in_bytes = 0) {
    static_assert(TILE_M % ScaleGranMA == 0);
    static_assert(TILE_K % ScaleGranKA == 0);

    constexpr auto tma_alignment_bytes = 16;
    constexpr auto alignment = tma_alignment_bytes / sizeof(T);
    static_assert(sizeof(T) * alignment == tma_alignment_bytes);

    shape_m = div_up(shape_m, alignment) * alignment;
    return make_2d_tma_desc(
        global_address,
        Layout::ColMajor,
        div_up(shape_m, ScaleGranMA),
        div_up(shape_k, ScaleGranKA) * num_problems,
        global_stride_in_bytes,
        TILE_M / ScaleGranMA,
        TILE_K / ScaleGranKA,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
  }

  template <typename T>
  static CUtensorMap make_2d_tma_desc(
      T* global_address,
      Layout layout,
      uint64_t gmem_rows,
      uint64_t gmem_cols,
      uint64_t global_stride_in_bytes,
      int smem_rows,
      int smem_cols,
      CUtensorMapSwizzle swizzle_type = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      int smem_padding = 0) {
    if (layout == Layout::RowMajor) {
      uint64_t gmem_dim[2] = {gmem_cols, gmem_rows};
      uint32_t smem_dim[2] = {uint32_t(smem_cols), uint32_t(smem_rows)};
      if (!global_stride_in_bytes) {
        global_stride_in_bytes = gmem_cols * sizeof(T);
      }
      return make_2d_tma_copy_desc(global_address, gmem_dim, global_stride_in_bytes, smem_dim, swizzle_type);
    } else {
      uint64_t gmem_dim[2] = {gmem_rows, gmem_cols};
      uint32_t smem_dim[2] = {uint32_t(smem_rows), uint32_t(smem_cols)};

      if (!global_stride_in_bytes) {
        global_stride_in_bytes = gmem_rows * sizeof(T);
      }
      return make_2d_tma_copy_desc(global_address, gmem_dim, global_stride_in_bytes, smem_dim, swizzle_type);
    }
  }
};

template <typename T>
__forceinline__ __device__ T find_max_elem_in_warp(T value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value = T(std::max(float(value), __shfl_down_sync(0xFFFFFFFF, float(value), offset)));
  }
  value = T(__shfl_sync(0xffffffff, float(value), 0));
  return value;
}

template <typename InputType, typename OutputType, typename ScaleType = float>
__global__ void
scale_1x128_kernel(OutputType* output, ScaleType* scales, InputType const* const input, int dim_x, int dim_y) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  size_t scales_along_dim_x = div_up(dim_x, 128);
  size_t scales_along_dim_y = div_up(dim_y, 1);
  size_t stride_scale_dim_y = div_up(dim_y, 4) * 4;

  for (size_t warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
       warp_idx < scales_along_dim_x * scales_along_dim_y;
       warp_idx += gridDim.x * blockDim.x / 32) {
    int scales_idx_y = warp_idx / scales_along_dim_x;
    int scales_idx_x = warp_idx % scales_along_dim_x;

    InputType const* input_line = input + (size_t)scales_idx_y * dim_x + scales_idx_x * 128;
    InputType input_amax = InputType(0);
    int lane_id = threadIdx.x % 32;
    InputType input_frag[4] = {0};

    for (int i = 0; i < 4; i++) {
      if (scales_idx_x * 128 + i * 32 + lane_id >= dim_x) {
        break;
      } else {
        input_frag[i] = input_line[lane_id];
        input_amax = InputType(std::max(float(input_amax), std::fabs(float(input_frag[i]))));
      }
      input_line += 32;
    }

    InputType amax = find_max_elem_in_warp(input_amax);
    ScaleType scale = amax != InputType(0.f) ? 448.f / ScaleType(amax) : 1.f;

    if (lane_id == 0) {
      scales[(size_t)scales_idx_x * stride_scale_dim_y + scales_idx_y] = ScaleType(1.f / scale);
    }

    OutputType* output_line = output + (size_t)scales_idx_y * dim_x + scales_idx_x * 128;
    for (int i = 0; i < 4; i++) {
      if (scales_idx_x * 128 + i * 32 + lane_id >= dim_x) {
        break;
      } else {
        ScaleType value = ScaleType(input_frag[i]) * scale;
        output_line[lane_id] = OutputType(value);
      }
      output_line += 32;
    }
  }
#endif
}

template <int NumThreads>
__global__ void
problem_m_padding_kernel(int64_t const* problem_m_offsets, int64_t* problem_m_padded_offsets, int num_problems) {
  extern __shared__ char shared_memory[];
  int64_t* smem_problem_m_boundaries = reinterpret_cast<int64_t*>(shared_memory);
  using BlockScan = cub::BlockScan<int64_t, NumThreads>;
  using TempStorage = typename BlockScan::TempStorage;
  TempStorage* smem_temp_storage = reinterpret_cast<TempStorage*>(smem_problem_m_boundaries + num_problems);
  // problem_m_offsets[0] is omitted because its value is known to be 0
  for (int i = threadIdx.x; i < num_problems; i += NumThreads) {
    smem_problem_m_boundaries[i] = problem_m_offsets[i + 1];
  }
  __syncthreads();
  int64_t reduce_offset = 0;
  int idx_scan = 0;
  int num_problems_padded = div_up(num_problems, NumThreads) * NumThreads;
  for (int i = threadIdx.x; i < num_problems_padded; i += NumThreads) {
    int64_t problem_m_size = 0;
    if (i < num_problems) {
      problem_m_size = smem_problem_m_boundaries[i] - (i > 0 ? smem_problem_m_boundaries[i - 1] : 0);
    }
    int64_t scan_val = ((problem_m_size + 31) >> 5) << 5;
    int64_t block_aggregate;
    // To reduce the use of __syncthreads, TempStorage is not reused
    BlockScan(smem_temp_storage[idx_scan++]).InclusiveSum(scan_val, scan_val, block_aggregate);
    if (i < num_problems) {
      problem_m_padded_offsets[i + 1] = scan_val + reduce_offset;
    }
    reduce_offset += block_aggregate;
  }
  if (threadIdx.x == 0) {
    problem_m_padded_offsets[0] = 0;
  }
}

template <bool UseBinarySearch, int NumThreads, typename InputType, typename OutputType>
__global__ void scale_1x128_kernel(
    OutputType* output,
    float* scales,
    InputType const* input,
    int64_t const* problem_m_offsets,
    int64_t* problem_m_padded_offsets,
    int num_problems,
    int dim_x,
    int64_t scale_leading_dim,
    uint32_t scale_dim_x_mul,
    uint32_t scale_dim_x_shr) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  extern __shared__ char shared_memory[];
  int64_t* smem_problem_m_boundaries = reinterpret_cast<int64_t*>(shared_memory);
  int64_t* smem_problem_m_padded_boundaries = smem_problem_m_boundaries + num_problems;
  using BlockScan = cub::BlockScan<int64_t, NumThreads>;
  using TempStorage = typename BlockScan::TempStorage;
  TempStorage* smem_temp_storage = reinterpret_cast<TempStorage*>(smem_problem_m_padded_boundaries + num_problems);

  // problem_m_offsets[0] is omitted because its value is known to be 0
  for (int i = threadIdx.x; i < num_problems; i += NumThreads) {
    smem_problem_m_boundaries[i] = problem_m_offsets[i + 1];
  }
  __syncthreads();
  int64_t reduce_offset = 0;
  int idx_scan = 0;
  int num_problems_padded = div_up(num_problems, NumThreads) * NumThreads;
  for (int i = threadIdx.x; i < num_problems_padded; i += NumThreads) {
    int64_t problem_m_size = 0;
    if (i < num_problems) {
      problem_m_size = smem_problem_m_boundaries[i] - (i > 0 ? smem_problem_m_boundaries[i - 1] : 0);
    }
    int64_t scan_val = ((problem_m_size + 31) >> 5) << 5;
    int64_t block_aggregate;
    // To reduce the use of __syncthreads, TempStorage is not reused
    BlockScan(smem_temp_storage[idx_scan++]).InclusiveSum(scan_val, scan_val, block_aggregate);
    if (i < num_problems) {
      smem_problem_m_padded_boundaries[i] = scan_val + reduce_offset;
    }
    reduce_offset += block_aggregate;
  }
  __syncthreads();
  if (blockIdx.x == 0) {
    for (int i = threadIdx.x; i < num_problems; i += NumThreads) {
      problem_m_padded_offsets[i + 1] = smem_problem_m_padded_boundaries[i];
    }
    if (threadIdx.x == 0) {
      problem_m_padded_offsets[0] = 0;
    }
  }

  size_t scales_along_dim_x = div_up(dim_x, 128);
  size_t scales_along_dim_y = smem_problem_m_boundaries[num_problems - 1];
  size_t total_scales = scales_along_dim_x * scales_along_dim_y;

  int problem_idx = 0;
  int64_t padded_offset = 0;
  int64_t boundary_left, boundary_right;
  if constexpr (UseBinarySearch) {
    boundary_left = smem_problem_m_boundaries[0];
    boundary_right = scales_along_dim_y;
  } else {
    boundary_left = 0;
    boundary_right = smem_problem_m_boundaries[0];
  }

  for (size_t warp_idx = (threadIdx.x + blockIdx.x * NumThreads) / 32; warp_idx < total_scales;
       warp_idx += (NumThreads * gridDim.x) / 32) {
    uint32_t scales_idx_y;  // = warp_idx / scales_along_dim_x;
    uint32_t scales_idx_x;  // = warp_idx % scales_along_dim_x;
    kernel_utils::fast_divmod(
        scales_idx_y, scales_idx_x, warp_idx, scales_along_dim_x, scale_dim_x_mul, scale_dim_x_shr);

    if constexpr (UseBinarySearch) {
      int idx_right = num_problems - 1;
      int64_t val_right = boundary_right;
      if (scales_idx_y >= boundary_left) {
        while (problem_idx + 1 < idx_right) {
          int idx_mid = (problem_idx + idx_right) >> 1;
          int64_t val_mid = smem_problem_m_boundaries[idx_mid];
          if (scales_idx_y < val_mid) {
            idx_right = idx_mid;
            val_right = val_mid;
          } else {
            problem_idx = idx_mid;
            boundary_left = val_mid;
          }
        }
        padded_offset = smem_problem_m_padded_boundaries[problem_idx] - boundary_left;
        boundary_left = val_right;
      }
    } else {
      if (boundary_right <= scales_idx_y) {
        while (problem_idx < num_problems - 1) {
          boundary_left = boundary_right;
          boundary_right = smem_problem_m_boundaries[++problem_idx];
          if (scales_idx_y < boundary_right) {
            break;
          }
        }
        padded_offset = smem_problem_m_padded_boundaries[problem_idx - 1] - boundary_left;
      }
    }

    auto warp_offset = (size_t)scales_idx_y * dim_x + scales_idx_x * 128;
    InputType const* input_line = input + warp_offset;
    OutputType* output_line = output + warp_offset;
    auto& scale_output = scales[(size_t)scales_idx_x * scale_leading_dim + scales_idx_y + padded_offset];

    int lane_id = threadIdx.x % 32;
    InputType input_frag[4];

    for (int i = 0; i < 4; i++) {
      input_frag[i] = (scales_idx_x * 128 + i * 32 + lane_id < dim_x) ? input_line[lane_id] : InputType(0);
      input_line += 32;
    }

    InputType amax = kernel_utils::warpReduceSum(
        max(max(fabs(float(input_frag[0])), fabs(float(input_frag[1]))),
            max(fabs(float(input_frag[2])), fabs(float(input_frag[3])))));

    // Half seems to be slower, probably because we need float values below
    // anyway. InputType amax = kernel_utils::warpReduceSum(
    //     __hmax(__hmax(__habs(input_frag[0]), __habs(input_frag[1])),
    //         __hmax(__habs(input_frag[2]), __habs(input_frag[3]))));

    float scale = amax != InputType(0.f) ? 448.f / float(amax) : 1.f;

    if (kernel_utils::elect_one_sync(lane_id)) {
      scale_output = float(1.f / scale);
    }

    for (int i = 0; i < 4; i++) {
      float value = float(input_frag[i]) * scale;
      if (scales_idx_x * 128 + i * 32 + lane_id < dim_x) {
        output_line[lane_id] = OutputType(value);
      }
      output_line += 32;
    }
  }
#endif
}

// input: [dim_y, dim_h, dim_x]
// output: [dim_h, dim_y, dim_x], cs[dim_h, dim_x/128, padding(dim_y)]
template <typename InputType, typename OutputType, typename ScaleType = float>
__global__ void scale_1x128_reshape_kernel(
    OutputType* output,
    ScaleType* scales,
    InputType const* const input,
    int dim_x,
    int dim_h,
    int dim_y,
    int stride_x) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  size_t scales_along_dim_x = div_up(dim_x, 128);
  size_t scales_along_dim_y = div_up(dim_y, 1);
  size_t scales_along_dim_h = div_up(dim_h, 1);
  size_t stride_scale_dim_y = div_up(dim_y, 4) * 4;

  for (size_t warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
       warp_idx < scales_along_dim_x * scales_along_dim_y * scales_along_dim_h;
       warp_idx += gridDim.x * blockDim.x / 32) {
    int scales_idx_y = warp_idx / (scales_along_dim_x * scales_along_dim_h);
    int scales_idx_h = (warp_idx % (scales_along_dim_x * scales_along_dim_h)) / scales_along_dim_x;
    int scales_idx_x = warp_idx % scales_along_dim_x;

    InputType const* input_line =
        input + (size_t)scales_idx_y * stride_x * dim_h + (size_t)scales_idx_h * stride_x + scales_idx_x * 128;
    InputType input_amax = InputType(0);
    int lane_id = threadIdx.x % 32;
    InputType input_frag[4] = {0};

    for (int i = 0; i < 4; i++) {
      if (scales_idx_x * 128 + i * 32 + lane_id >= dim_x) {
        break;
      } else {
        input_frag[i] = input_line[lane_id];
        input_amax = InputType(std::max(float(input_amax), std::fabs(float(input_frag[i]))));
      }
      input_line += 32;
    }

    InputType amax = find_max_elem_in_warp(input_amax);
    ScaleType scale = amax != InputType(0.f) ? 448.f / ScaleType(amax) : 1.f;

    if (lane_id == 0) {
      scales
          [(size_t)scales_idx_h * scales_along_dim_x * stride_scale_dim_y + (size_t)scales_idx_x * stride_scale_dim_y +
           scales_idx_y] = ScaleType(1.f / scale);
    }

    OutputType* output_line =
        output + (size_t)scales_idx_h * dim_y * dim_x + (size_t)scales_idx_y * dim_x + scales_idx_x * 128;
    for (int i = 0; i < 4; i++) {
      if (scales_idx_x * 128 + i * 32 + lane_id >= dim_x) {
        break;
      } else {
        ScaleType value = ScaleType(input_frag[i]) * scale;
        output_line[lane_id] = OutputType(value);
      }
      output_line += 32;
    }
  }
#endif
}

template <typename InputType, typename OutputType, typename ScaleType = float>
__global__ void
scale_128x128_kernel(OutputType* output, ScaleType* scales, InputType const* const input, int dim_x, int dim_y) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  int scales_along_dim_x = div_up(dim_x, 128);
  int scales_along_dim_y = div_up(dim_y, 128);

  for (int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32; warp_idx < scales_along_dim_x * scales_along_dim_y;
       warp_idx += gridDim.x * blockDim.x / 32) {
    int scales_idx_y = warp_idx / scales_along_dim_x;
    int scales_idx_x = warp_idx % scales_along_dim_x;

    InputType const* input_line = input + scales_idx_y * 128 * dim_x + scales_idx_x * 128;
    InputType input_amax = InputType(0);
    int lane_id = threadIdx.x % 32;

    for (int i = 0; i < 128; i++) {
      if (scales_idx_y * 128 + i >= dim_y) {
        break;
      }
      InputType const* input_d = input_line;

      for (int j = 0; j < 4; j++) {
        if (scales_idx_x * 128 + i * 32 + lane_id >= dim_x) {
          break;
        } else {
          input_amax = InputType(std::max(float(input_amax), std::fabs(float(input_d[lane_id]))));
        }
        input_d += 32;
      }
      input_line += dim_x;
    }

    InputType amax = find_max_elem_in_warp(input_amax);
    ScaleType scale = amax != InputType(0.f) ? 448.f / ScaleType(amax) : 1.f;

    if (lane_id == 0) {
      scales[scales_idx_y * scales_along_dim_x + scales_idx_x] = ScaleType(1.f / scale);
    }

    input_line = input + scales_idx_y * 128 * dim_x + scales_idx_x * 128;
    OutputType* output_line = output + scales_idx_y * 128 * dim_x + scales_idx_x * 128;

    for (int i = 0; i < 128; i++) {
      if (scales_idx_y * 128 + i >= dim_y) {
        break;
      }
      InputType const* input_d = input_line;
      OutputType* output_d = output_line;

      for (int j = 0; j < 4; j++) {
        if (scales_idx_x * 128 + j * 32 + lane_id >= dim_x) {
          break;
        } else {
          output_d[lane_id] = OutputType(ScaleType(input_d[lane_id]) * scale);
        }
        input_d += 32;
        output_d += 32;
      }

      input_line += dim_x;
      output_line += dim_x;
    }
  }
#endif
}

template <typename OutputType>
__global__ void fill_kernel(OutputType* output, size_t num_elems, float value) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elems; idx += gridDim.x * blockDim.x) {
    output[idx] = OutputType(value);
  }
}

template <typename InputType, typename OutputType>
__global__ void convert_kernel(OutputType* output, InputType const* const input, size_t num_elems) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elems; idx += gridDim.x * blockDim.x) {
    float value = float(input[idx]);
    if (std::isnan(value)) {
      output[idx] = OutputType(448);
    } else {
      output[idx] = OutputType(value);
    }
  }
}

static int kNumDeviceSMs = -1;

void fp8_1x128_cs(
    __nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_y, cudaStream_t stream) {
  if (kNumDeviceSMs < 0) {
    kNumDeviceSMs = getMultiProcessorCount();
  }
  scale_1x128_kernel<<<kNumDeviceSMs, 256, 0, stream>>>(mat_quant, scales, mat, shape_x, shape_y);
}

void fp8_1x128_cs_reshape(
    __nv_fp8_e4m3* mat_quant,
    float* scales,
    __nv_bfloat16 const* mat,
    int shape_x,
    int shape_h,
    int shape_y,
    int stride_x,
    cudaStream_t stream) {
  if (kNumDeviceSMs < 0) {
    kNumDeviceSMs = getMultiProcessorCount();
  }
  scale_1x128_reshape_kernel<<<kNumDeviceSMs, 256, 0, stream>>>(
      mat_quant, scales, mat, shape_x, shape_h, shape_y, stride_x);
}

void fp8_128x128_cs(
    __nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_y, cudaStream_t stream) {
  if (kNumDeviceSMs < 0) {
    kNumDeviceSMs = getMultiProcessorCount();
  }
  convert_kernel<<<kNumDeviceSMs, 256, 0, stream>>>(mat_quant, mat, shape_x * shape_y);
  fill_kernel<<<kNumDeviceSMs, 256, 0, stream>>>(scales, div_up(shape_x, 128) * div_up(shape_y, 128), 1);
}

void gemm_dispatch_old(
    void* mat_a,
    int ld_a,
    void* mat_b,
    int ld_b,
    void* mat_d,
    int ld_d,
    float* scales_a,
    float* scales_b,
    int shape_m,
    int shape_n,
    int shape_k,
    cudaStream_t stream) {
  if (kNumDeviceSMs < 0) {
    kNumDeviceSMs = getMultiProcessorCount();
  }
  auto get_status = [=](int tile_n) -> std::pair<int, int> {
    int num_blocks = div_up(shape_n, tile_n);
    int num_waves = div_up(num_blocks, kNumDeviceSMs);
    return {num_waves, num_blocks % kNumDeviceSMs};
  };

  auto compare = [=](int tile_n, int old_block_n) -> bool {
    if (old_block_n == 0) return true;

    auto status = get_status(tile_n);
    auto old_status = get_status(old_block_n);
    if (status.first != old_status.first) return status.first < old_status.first;
    if (status.first == 1) return status.second > old_status.second;
    return tile_n > old_block_n;
  };

  int best_tile_m = shape_m <= 64 ? 64 : 128, best_block_n = 0;
  for (auto const& tile_n : {32, 64, 128})
    if (compare(tile_n, best_block_n)) best_block_n = tile_n;

#define DISPATCH_BLOCK_SIZE(TILE_M, TILE_N)      \
  {                                              \
    using GemmType = Fp8Gemm<                    \
        __nv_fp8_e4m3,                           \
        Layout::RowMajor,                        \
        __nv_fp8_e4m3,                           \
        Layout::ColMajor,                        \
        __nv_bfloat16,                           \
        Layout::RowMajor,                        \
        float,                                   \
        float,                                   \
        float,                                   \
        TILE_M,                                  \
        TILE_N,                                  \
        128,                                     \
        ScaleType::PerSubChannel,                \
        ScaleType::PerBlock,                     \
        1,                                       \
        128,                                     \
        128,                                     \
        128>;                                    \
    GemmType::run(                               \
        reinterpret_cast<__nv_fp8_e4m3*>(mat_a), \
        ld_a,                                    \
        reinterpret_cast<__nv_fp8_e4m3*>(mat_b), \
        ld_b,                                    \
        reinterpret_cast<__nv_bfloat16*>(mat_d), \
        ld_d,                                    \
        scales_a,                                \
        scales_b,                                \
        shape_m,                                 \
        shape_n,                                 \
        shape_k,                                 \
        stream                                   \
                                                 \
    );                                           \
  }                                              \
  break

#define DISPATCH_BLOCK_SIZE_M(TILE_N)     \
  {                                       \
    switch (best_tile_m) {                \
      case 64:                            \
        DISPATCH_BLOCK_SIZE(64, TILE_N);  \
      case 128:                           \
        DISPATCH_BLOCK_SIZE(128, TILE_N); \
    }                                     \
  }                                       \
  break

  switch (best_block_n) {
    case 16:
      DISPATCH_BLOCK_SIZE_M(16);
    case 32:
      DISPATCH_BLOCK_SIZE_M(32);
    case 64:
      DISPATCH_BLOCK_SIZE_M(64);
    case 128:
      DISPATCH_BLOCK_SIZE_M(128);
  }
#undef DISPATCH_BLOCK_SIZE
#undef DISPATCH_BLOCK_SIZE_M
}

void gemm_dispatch_old(
    void* mat_a,
    void* mat_b,
    void* mat_d,
    float* scales_a,
    float* scales_b,
    int num_problems,
    int64_t const* problem_m_offsets,
    int max_shape_m,
    int shape_n,
    int shape_k,
    cudaStream_t stream) {
  if (kNumDeviceSMs < 0) {
    kNumDeviceSMs = getMultiProcessorCount();
  }
  auto get_status = [=](int tile_n) -> std::pair<int, int> {
    int num_blocks = div_up(shape_n, tile_n);
    int num_waves = div_up(num_blocks, kNumDeviceSMs);
    return {num_waves, num_blocks % kNumDeviceSMs};
  };

  auto compare = [=](int tile_n, int old_block_n) -> bool {
    if (old_block_n == 0) return true;

    auto status = get_status(tile_n), old_status = get_status(old_block_n);
    if (status.first != old_status.first) return status.first < old_status.first;
    if (status.first == 1) return status.second > old_status.second;
    return tile_n > old_block_n;
  };

  int shape_m = 128;
  int best_tile_m = shape_m <= 64 ? 64 : 128, best_block_n = 0;
  for (auto const& tile_n : {64, 128})
    if (compare(tile_n, best_block_n)) best_block_n = tile_n;

#define DISPATCH_BLOCK_SIZE(TILE_M, TILE_N)      \
  {                                              \
    using GemmType = Fp8Gemm<                    \
        __nv_fp8_e4m3,                           \
        Layout::RowMajor,                        \
        __nv_fp8_e4m3,                           \
        Layout::ColMajor,                        \
        __nv_bfloat16,                           \
        Layout::RowMajor,                        \
        float,                                   \
        float,                                   \
        float,                                   \
        TILE_M,                                  \
        TILE_N,                                  \
        128,                                     \
        ScaleType::PerSubChannel,                \
        ScaleType::PerBlock,                     \
        1,                                       \
        128,                                     \
        128,                                     \
        128>;                                    \
    GemmType::run(                               \
        reinterpret_cast<__nv_fp8_e4m3*>(mat_a), \
        reinterpret_cast<__nv_fp8_e4m3*>(mat_b), \
        reinterpret_cast<__nv_bfloat16*>(mat_d), \
        scales_a,                                \
        scales_b,                                \
        num_problems,                            \
        problem_m_offsets,                       \
        shape_n,                                 \
        shape_k,                                 \
        max_shape_m,                             \
        stream                                   \
                                                 \
    );                                           \
  }                                              \
  break

#define DISPATCH_BLOCK_SIZE_M(TILE_N)     \
  {                                       \
    switch (best_tile_m) {                \
      case 64:                            \
        DISPATCH_BLOCK_SIZE(64, TILE_N);  \
      case 128:                           \
        DISPATCH_BLOCK_SIZE(128, TILE_N); \
    }                                     \
  }                                       \
  break

  switch (best_block_n) {
    case 16:
      DISPATCH_BLOCK_SIZE_M(16);
    case 32:
      DISPATCH_BLOCK_SIZE_M(32);
    case 64:
      DISPATCH_BLOCK_SIZE_M(64);
    case 128:
      DISPATCH_BLOCK_SIZE_M(128);
  }
#undef DISPATCH_BLOCK_SIZE
#undef DISPATCH_BLOCK_SIZE_M
}

void fp8_gemm_run(
    __nv_fp8_e4m3* mat_a,
    int ld_a,
    __nv_fp8_e4m3* mat_b,
    int ld_b,
    __nv_bfloat16* mat_d,
    int ld_d,
    uint32_t shape_m,
    uint32_t shape_n,
    uint32_t shape_k,
    float* scales_a,
    float* scales_b,
    cudaStream_t stream) {
  if (shape_m == 0) {
    return;
  }

  gemm_dispatch_old(
      mat_a,
      ld_a,
      mat_b,
      ld_b,
      mat_d,
      ld_d,
      scales_a,
      scales_b,
      static_cast<int>(shape_m),
      static_cast<int>(shape_n),
      static_cast<int>(shape_k),
      stream);
}

void fp8_gemm_run(
    __nv_bfloat16 const* mat_a,
    __nv_fp8_e4m3* fp8_mat_a,
    int ld_a,
    float* scales_a,
    __nv_bfloat16 const* mat_b,
    __nv_fp8_e4m3* fp8_mat_b,
    int ld_b,
    float* scales_b,
    __nv_bfloat16* mat_d,
    int ld_d,
    uint32_t shape_m,
    uint32_t shape_n,
    uint32_t shape_k,
    cudaStream_t stream,
    bool internal_quantize_a = true,
    bool internal_quantize_b = true) {
  if (shape_m == 0) {
    return;
  }
  if (kNumDeviceSMs < 0) {
    kNumDeviceSMs = getMultiProcessorCount();
  }

  if (internal_quantize_a) {
    scale_1x128_kernel<<<kNumDeviceSMs, 256, 0, stream>>>(fp8_mat_a, scales_a, mat_a, shape_k, shape_m);
  }
  if (internal_quantize_b) {
    scale_128x128_kernel<<<kNumDeviceSMs, 256, 0, stream>>>(fp8_mat_b, scales_b, mat_b, shape_k, shape_n);
  }
  fp8_gemm_run(fp8_mat_a, ld_a, fp8_mat_b, ld_b, mat_d, ld_d, shape_m, shape_n, shape_k, scales_a, scales_b, stream);
}

void fp8_grouped_gemm_run(
    __nv_bfloat16 const* mat_a,
    __nv_fp8_e4m3* fp8_mat_a,
    float* scales_a,
    __nv_bfloat16 const* mat_b,
    __nv_fp8_e4m3* fp8_mat_b,
    float* scales_b,
    __nv_bfloat16* mat_d,
    int64_t const* problem_m_offsets,
    int64_t* problem_m_padded_offsets,
    int num_problems,
    int64_t expected_m,
    int64_t max_shape_m,
    int64_t max_shape_m_padded,
    int shape_n,
    int shape_k,
    cudaStream_t stream,
    bool internal_quantize_a = true,
    bool internal_quantize_b = true) {
  if (kNumDeviceSMs < 0) {
    kNumDeviceSMs = getMultiProcessorCount();
  }

  if (internal_quantize_a) {
    constexpr int NumThreads = 256;
    int scales_dim_x = div_up(shape_k, 128);
    uint32_t scale_dim_x_mul, scale_dim_x_shr;
    kernel_utils::find_divisor(scale_dim_x_mul, scale_dim_x_shr, scales_dim_x);

    using BlockScan = cub::BlockScan<int64_t, NumThreads>;
    int smem_size =
        num_problems * sizeof(int64_t) * 2 + div_up(num_problems, NumThreads) * sizeof(BlockScan::TempStorage);
    int num_blocks = std::min(static_cast<int64_t>(kNumDeviceSMs), div_up(max_shape_m * scales_dim_x, NumThreads / 32));
    // Binary search is expected to have lower complexity when max_shape_m is small
    bool use_binary_search =
        static_cast<double>(max_shape_m) * scales_dim_x / static_cast<double>(NumThreads * num_blocks / 32) <=
        static_cast<double>(num_problems) / std::log2(static_cast<double>(num_problems));
    auto kernel = use_binary_search ? scale_1x128_kernel<true, NumThreads, __nv_bfloat16, __nv_fp8_e4m3>
                                    : scale_1x128_kernel<false, NumThreads, __nv_bfloat16, __nv_fp8_e4m3>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<num_blocks, NumThreads, smem_size, stream>>>(
        fp8_mat_a,
        scales_a,
        mat_a,
        problem_m_offsets,
        problem_m_padded_offsets,
        num_problems,
        shape_k,
        max_shape_m_padded,
        scale_dim_x_mul,
        scale_dim_x_shr);
  } else {
    constexpr int NumThreads = 256;
    using BlockScan = cub::BlockScan<int64_t, NumThreads>;
    int smem_size = num_problems * sizeof(int64_t) + div_up(num_problems, NumThreads) * sizeof(BlockScan::TempStorage);
    auto kernel = problem_m_padding_kernel<NumThreads>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<1, NumThreads, smem_size, stream>>>(problem_m_offsets, problem_m_padded_offsets, num_problems);
  }

  if (internal_quantize_b) {
    __nv_fp8_e4m3* fp8_mat_b_tmp = fp8_mat_b;
    float* scales_b_tmp = scales_b;
    __nv_bfloat16 const* mat_b_tmp = mat_b;

    for (int i = 0; i < num_problems; i++) {
      scale_128x128_kernel<<<kNumDeviceSMs, 256, 0, stream>>>(fp8_mat_b_tmp, scales_b_tmp, mat_b_tmp, shape_k, shape_n);
      fp8_mat_b_tmp += shape_n * shape_k;
      mat_b_tmp += shape_n * shape_k;
      scales_b_tmp += div_up(shape_n, 128) * div_up(shape_k, 128);
    }
  }
  using GemmType = Fp8Gemm<
      __nv_fp8_e4m3,
      Layout::RowMajor,
      __nv_fp8_e4m3,
      Layout::ColMajor,
      __nv_bfloat16,
      Layout::RowMajor,
      float,
      float,
      float,
      128,
      64,
      128,
      ScaleType::PerSubChannel,
      ScaleType::PerBlock,
      1,
      128,
      128,
      128>;
  GemmType::run(
      fp8_mat_a,
      fp8_mat_b,
      mat_d,
      scales_a,
      scales_b,
      num_problems,
      problem_m_offsets,
      shape_n,
      shape_k,
      static_cast<int>(max_shape_m),
      stream,
      128,
      problem_m_padded_offsets,
      static_cast<int>(max_shape_m_padded));
}

void fp8_stride_batch_gemm_run(
    __nv_bfloat16 const* mat_a,
    __nv_fp8_e4m3* fp8_mat_a,
    float* scales_a,
    int ld_a,
    int stride_a,
    int stride_scales_a,
    __nv_bfloat16 const* mat_b,
    __nv_fp8_e4m3* fp8_mat_b,
    float* scales_b,
    int ld_b,
    int stride_b,
    __nv_bfloat16* mat_d,
    int ld_d,
    int stride_d,
    uint32_t num_problems,
    uint32_t shape_m,
    uint32_t shape_n,
    uint32_t shape_k,
    cudaStream_t stream,
    bool internal_quantize_a = true,
    bool internal_quantize_b = true) {
  if (shape_m == 0) {
    return;
  }

  if (kNumDeviceSMs < 0) {
    kNumDeviceSMs = getMultiProcessorCount();
  }
  if (internal_quantize_a) {
    scale_1x128_kernel<<<kNumDeviceSMs, 256, 0, stream>>>(fp8_mat_a, scales_a, mat_a, shape_k, shape_m * num_problems);
  }
  if (internal_quantize_b) {
    scale_128x128_kernel<<<kNumDeviceSMs, 256, 0, stream>>>(
        fp8_mat_b, scales_b, mat_b, shape_k, shape_n * num_problems);
  }
  using GemmType = Fp8Gemm<
      __nv_fp8_e4m3,
      Layout::RowMajor,
      __nv_fp8_e4m3,
      Layout::ColMajor,
      __nv_bfloat16,
      Layout::RowMajor,
      float,
      float,
      float,
      128,
      64,
      128,
      ScaleType::PerSubChannel,
      ScaleType::PerBlock,
      1,
      128,
      128,
      128>;
  GemmType::run(
      fp8_mat_a,
      ld_a,
      stride_a,
      fp8_mat_b,
      ld_b,
      stride_b,
      mat_d,
      ld_d,
      stride_d,
      scales_a,
      stride_scales_a,
      scales_b,
      static_cast<int>(shape_m),
      static_cast<int>(shape_n),
      static_cast<int>(shape_k),
      static_cast<int>(num_problems),
      stream);
}

}  // namespace tensorrt_llm::kernels::fp8_blockscale_gemm
