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

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

namespace {

using namespace device;

// Convert VPT bfloat16 values from a uint4 to float array
template <int VPT>
SGL_DEVICE void bf16_uint4_to_float(uint4 const& vec, float* dst) {
  bf16_t* bf16_ptr = reinterpret_cast<bf16_t*>(const_cast<uint4*>(&vec));
#pragma unroll
  for (int i = 0; i < VPT; i++) {
    dst[i] = __bfloat162float(bf16_ptr[i]);
  }
}

// kOutFloat: true = float32 output, false = bfloat16 output
template <int kBlockSize, int VPT, int kNumTokens, int kNumExperts, int kHiddenDim, bool kUsePDL, bool kOutFloat>
__global__
__launch_bounds__(128, 1) void router_gemm_kernel(void* out, bf16_t const* mat_a, bf16_t const* mat_b) {
  int const n_idx = blockIdx.x;
  int const tid = threadIdx.x;
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  constexpr int k_elems_per_k_iteration = VPT * kBlockSize;
  constexpr int k_iterations = kHiddenDim / k_elems_per_k_iteration;

  float acc[kNumTokens] = {};
  __shared__ float sm_reduction[kNumTokens][kNumWarps];

  bf16_t const* b_col = mat_b + n_idx * kHiddenDim;

  int k_bases[k_iterations];
#pragma unroll
  for (int ki = 0; ki < k_iterations; ki++) {
    k_bases[ki] = ki * k_elems_per_k_iteration + tid * VPT;
  }

  PDLWaitPrimary<kUsePDL>();

  for (int ki = 0; ki < k_iterations; ki++) {
    int const k_base = k_bases[ki];

    uint4 b_vec = *reinterpret_cast<uint4 const*>(b_col + k_base);
    float b_float[VPT];
    bf16_uint4_to_float<VPT>(b_vec, b_float);

#pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; m_idx++) {
      uint4 a_vec = *reinterpret_cast<uint4 const*>(mat_a + (m_idx * kHiddenDim) + k_base);
      float a_float[VPT];
      bf16_uint4_to_float<VPT>(a_vec, a_float);
#pragma unroll
      for (int k = 0; k < VPT; k++) {
        acc[m_idx] += a_float[k] * b_float[k];
      }
    }
  }

  int const warpId = tid / kWarpSize;
  int const laneId = tid % kWarpSize;

#pragma unroll
  for (int m = 0; m < kNumTokens; m++) {
    float sum = acc[m];
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    if (laneId == 0) {
      sm_reduction[m][warpId] = sum;
    }
  }

  __syncthreads();

  if (tid == 0) {
#pragma unroll
    for (int m = 0; m < kNumTokens; m++) {
      float final_sum = 0.0f;
#pragma unroll
      for (int w = 0; w < kNumWarps; w++) {
        final_sum += sm_reduction[m][w];
      }
      if constexpr (kOutFloat) {
        static_cast<fp32_t*>(out)[m * kNumExperts + n_idx] = final_sum;
      } else {
        static_cast<bf16_t*>(out)[m * kNumExperts + n_idx] = __float2bfloat16(final_sum);
      }
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

// Dispatch runtime num_tokens to compile-time template parameter [kBegin, kEnd]
template <int kBegin, int kEnd, int kNumExperts, int kHiddenDim, bool kUsePDL, bool kOutFloat>
struct RouterGemmDispatcher {
  static constexpr int kBlockSize = 128;
  static constexpr int VPT = 16 / sizeof(bf16_t);  // 8 elements per thread

  static void run(int num_tokens, void* output, bf16_t const* mat_a, bf16_t const* mat_b, DLDevice device) {
    if (num_tokens == kBegin) {
      constexpr auto kernel =
          router_gemm_kernel<kBlockSize, VPT, kBegin, kNumExperts, kHiddenDim, kUsePDL, kOutFloat>;
      host::LaunchKernel(kNumExperts, kBlockSize, device)
          .enable_pdl(kUsePDL)(kernel, output, mat_a, mat_b);
    } else {
      RouterGemmDispatcher<kBegin + 1, kEnd, kNumExperts, kHiddenDim, kUsePDL, kOutFloat>::run(
          num_tokens, output, mat_a, mat_b, device);
    }
  }
};

// Base case: kBegin == kEnd
template <int kEnd, int kNumExperts, int kHiddenDim, bool kUsePDL, bool kOutFloat>
struct RouterGemmDispatcher<kEnd, kEnd, kNumExperts, kHiddenDim, kUsePDL, kOutFloat> {
  static constexpr int kBlockSize = 128;
  static constexpr int VPT = 16 / sizeof(bf16_t);

  static void run(int num_tokens, void* output, bf16_t const* mat_a, bf16_t const* mat_b, DLDevice device) {
    if (num_tokens == kEnd) {
      constexpr auto kernel =
          router_gemm_kernel<kBlockSize, VPT, kEnd, kNumExperts, kHiddenDim, kUsePDL, kOutFloat>;
      host::LaunchKernel(kNumExperts, kBlockSize, device)
          .enable_pdl(kUsePDL)(kernel, output, mat_a, mat_b);
    } else {
      host::panic({}, "dsv3_router_gemm: num_tokens must be between 1 and 16, got ", num_tokens);
    }
  }
};

// kNumExperts: compile-time 256 or 384
// kHiddenDim: compile-time 7168
// kUsePDL: compile-time bool (true on SM90+)
// kOutFloat: compile-time bool (true = float32 output, false = bfloat16 output)
template <int kNumExperts, int kHiddenDim, bool kUsePDL, bool kOutFloat>
struct DSV3RouterGemmKernel {
  static void run(
      const tvm::ffi::TensorView mat_a,
      const tvm::ffi::TensorView mat_b,
      const tvm::ffi::TensorView output) {
    using namespace host;

    auto M = SymbolicSize{"num_tokens"};
    auto K = SymbolicSize{"hidden_dim"};
    auto N = SymbolicSize{"num_experts"};
    auto device = SymbolicDevice{};
    K.set_value(kHiddenDim);
    N.set_value(kNumExperts);
    device.set_options<kDLCUDA>();

    TensorMatcher({M, K})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(mat_a);
    TensorMatcher({N, K})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(mat_b);
    if constexpr (kOutFloat) {
      TensorMatcher({M, N})
          .with_dtype<fp32_t>()
          .with_device(device)
          .verify(output);
    } else {
      TensorMatcher({M, N})
          .with_dtype<bf16_t>()
          .with_device(device)
          .verify(output);
    }

    const int num_tokens = static_cast<int>(M.unwrap());

    RouterGemmDispatcher<1, 16, kNumExperts, kHiddenDim, kUsePDL, kOutFloat>::run(
        num_tokens,
        output.data_ptr(),
        static_cast<bf16_t const*>(mat_a.data_ptr()),
        static_cast<bf16_t const*>(mat_b.data_ptr()),
        device.unwrap());
  }
};

}  // namespace
