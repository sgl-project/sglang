/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/thop/dsv3RouterGemmOp.cpp
 *
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace dsv3_router_jit {

using bf16_t = __nv_bfloat16;

template <typename T, int kBlockSize, int VPT, int kNumTokens, int kNumExperts, int kHiddenDim, bool kUsePDL>
__global__
__launch_bounds__(kBlockSize, 1) void router_gemm_kernel_float_output(float* out, T const* mat_a, T const* mat_b) {
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  constexpr int kElemsPerKIter = VPT * kBlockSize;
  static_assert(kHiddenDim % kElemsPerKIter == 0, "hidden_dim must be divisible by one K iteration");
  constexpr int kIters = kHiddenDim / kElemsPerKIter;

  int const n_idx = blockIdx.x;
  int const tid = threadIdx.x;
  int const warp_id = tid / kWarpSize;
  int const lane_id = tid % kWarpSize;

  float acc[kNumTokens] = {};
  __shared__ float sm_reduction[kNumTokens][kNumWarps];

  T const* b_col = mat_b + n_idx * kHiddenDim;

  device::PDLWaitPrimary<kUsePDL>();

  int k_base = tid * VPT;
#pragma unroll
  for (int ki = 0; ki < kIters; ++ki, k_base += kElemsPerKIter) {
    device::AlignedVector<bf16_t, VPT> b_vec;
    b_vec.load(b_col + k_base);

#pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; ++m_idx) {
      device::AlignedVector<bf16_t, VPT> a_vec;
      a_vec.load(mat_a + m_idx * kHiddenDim + k_base);

#pragma unroll
      for (int k = 0; k < VPT; ++k) {
        acc[m_idx] += __bfloat162float(a_vec[k]) * __bfloat162float(b_vec[k]);
      }
    }
  }

#pragma unroll
  for (int m_idx = 0; m_idx < kNumTokens; ++m_idx) {
    float sum = device::warp::reduce_sum(acc[m_idx]);
    if (lane_id == 0) {
      sm_reduction[m_idx][warp_id] = sum;
    }
  }

  __syncthreads();

  if (tid == 0) {
#pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; ++m_idx) {
      float final_sum = 0.0f;
#pragma unroll
      for (int w = 0; w < kNumWarps; ++w) {
        final_sum += sm_reduction[m_idx][w];
      }
      out[m_idx * kNumExperts + n_idx] = final_sum;
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <typename T, int kNumTokens, int kNumExperts, int kHiddenDim, bool kUsePDL>
void invokeRouterGemmFloatOutput(float* output, T const* mat_a, T const* mat_b, cudaStream_t stream) {
  constexpr int VPT = 16 / sizeof(T);
  constexpr int kBlockSize = 128;

  constexpr auto kernel =
      router_gemm_kernel_float_output<T, kBlockSize, VPT, kNumTokens, kNumExperts, kHiddenDim, kUsePDL>;

  host::LaunchKernel(dim3(kNumExperts), dim3(kBlockSize), stream).enable_pdl(kUsePDL)(kernel, output, mat_a, mat_b);
}

}  // namespace dsv3_router_jit
