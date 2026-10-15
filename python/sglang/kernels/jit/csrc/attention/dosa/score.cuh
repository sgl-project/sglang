#pragma once
// Copyright (c) 2025 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Pipeline portions adapted from sgl-deep-gemm v0.1.5.post3 (SM100 H32).
// FP8 payloads, four FP32 reduction chains and unordered top-k are preserved.

#include <ATen/cuda/CUDAContext.h>
#include <cutlass/float8.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>
#include <deep_gemm/impls/sm100_mqa_logits.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <vector>
#include <cutlass/float8.h>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace dosa_score_emit {
template <uint32_t M, uint32_t N, typename Fp8>
__device__ __forceinline__ void issue_h32_mma(Fp8 *keys, Fp8 *queries,
                                              uint32_t key_row, uint32_t tmem,
                                              uint32_t completion_barrier) {
  auto instruction =
      cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                  float, M, N, cute::UMMA::Major::K,
                                  cute::UMMA::Major::K>();
  auto runtime = cute::UMMA::make_runtime_instr_desc(instruction);
#pragma unroll
  for (uint32_t k = 0; k < 128 / 32; ++k) {
    auto ad =
        deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, 128,
                                              128>(keys, key_row, k * 32);
    auto bd =
        deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, 128,
                                              128>(queries, 0, k * 32);
    deep_gemm::ptx::SM100_MMA_F8F6F4_SS::fma(ad, bd, tmem, k, runtime);
  }
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one."
               "shared::cluster.b64 [%0];" ::"r"(completion_barrier));
}
} // namespace dosa_score_emit

#include "geometry.cuh"
// Helper bodies copied verbatim from the validated N45 source.
// Only avoid compiling unrelated N45 entry points during research iteration.
namespace certicascade_streaming_exact {
using torch::Tensor;
constexpr uint32_t kHeads=dosa_geometry::Heads, kHeadDim=128;
__device__ __forceinline__ void named_sync(uint32_t threads, uint32_t id) {
  asm volatile("" ::: "memory");
  cutlass::arch::NamedBarrier(threads, id).arrive_and_wait_unaligned();
  asm volatile("" ::: "memory");
}
__device__ __forceinline__ void tmem_load(const uint32_t address,
                                          float *destination) {
  [&]<size_t... Is>(cute::index_sequence<Is...>) {
    cute::SM100_TMEM_LOAD_32dp32b16x::copy(
        address, reinterpret_cast<uint32_t *>(destination)[Is]...);
  }(cute::make_index_sequence<16>{});
}
__device__ __forceinline__ void tmem_load_heads(uint32_t address, float* destination) {
  tmem_load(address, destination);cutlass::arch::fence_view_async_tmem_load();
  tmem_load(address+16, destination+16);cutlass::arch::fence_view_async_tmem_load();
  if constexpr(kHeads==64) {
    tmem_load(address+32, destination+32);cutlass::arch::fence_view_async_tmem_load();
    tmem_load(address+48, destination+48);cutlass::arch::fence_view_async_tmem_load();
  }
}
__device__ __forceinline__ float
exact_h32_reduce(const float *accum, const float *weights, float scale) {
  float2 sum0 = make_float2(0.0f, 0.0f), sum1 = make_float2(0.0f, 0.0f);
#pragma unroll
  for (uint32_t h = 0; h < kHeads; h += 4) {
    const float2 a0 = make_float2(accum[h], accum[h + 1]);
    const float2 a1 = make_float2(fabsf(accum[h]), fabsf(accum[h + 1]));
    const float2 w0 = make_float2(weights[h], weights[h + 1]);
    sum0 = __ffma2_rn(__fadd2_rn(a0, a1), w0, sum0);
    const float2 a2 = make_float2(accum[h + 2], accum[h + 3]);
    const float2 a3 = make_float2(fabsf(accum[h + 2]), fabsf(accum[h + 3]));
    const float2 w1 = make_float2(weights[h + 2], weights[h + 3]);
    sum1 = __ffma2_rn(__fadd2_rn(a2, a3), w1, sum1);
  }
  const float2 pair = __fadd2_rn(sum0, sum1);
  return __fmul_rn(__fmul_rn(__fadd_rn(pair.x, pair.y), 0.5f), scale);
}

using dosa_score_emit::issue_h32_mma;
inline CUtensorMap make_map(void *pointer, CUtensorMapDataType dtype,
                            std::initializer_list<cuuint64_t> dimensions,
                            std::initializer_list<cuuint64_t> strides,
                            std::initializer_list<cuuint32_t> box,
                            CUtensorMapSwizzle swizzle) {
  CUtensorMap map;
  const cuuint32_t element_strides[3] = {1, 1, 1};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, dtype, dimensions.size(), pointer, dimensions.begin(),
      strides.begin(), box.begin(), element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS,
              "cuTensorMapEncodeTiled failed: ", result);
  return map;
}
// A {NK,1} F32 tensor map still requires a 16-byte outer stride. The second
// coordinate is always zero, so aligning this stride does not read padding.
inline cuuint64_t scale_row_stride(int64_t nk){
  return (static_cast<cuuint64_t>(nk)*sizeof(float)+15)&~cuuint64_t(15);
}
} // namespace certicascade_streaming_exact
