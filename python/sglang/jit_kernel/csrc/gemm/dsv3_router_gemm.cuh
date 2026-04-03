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

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <type_traits>

namespace {

using namespace device;

static constexpr int kDefaultNumExperts = 256;
static constexpr int kKimiK2NumExperts = 384;
static constexpr int kDefaultHiddenDim = 7168;

// kOutFloat: true = float32 output, false = bfloat16 output
template <
    typename T,
    typename OutT,
    int kBlockSize,
    int VPT,
    int kNumTokens,
    int kNumExperts,
    int kHiddenDim,
    bool kUsePDL>
__global__ __launch_bounds__(kBlockSize, 1) void router_gemm_kernel(OutT* out, T const* mat_a, T const* mat_b) {
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  constexpr int kElemsPerKIter = VPT * kBlockSize;
  static_assert(kHiddenDim % kElemsPerKIter == 0, "hidden_dim must be divisible by one K iteration");
  constexpr int kIters = kHiddenDim / kElemsPerKIter;
  // Padding to avoid shared memory bank conflicts when kNumTokens > 8
  constexpr int kSmReductionPad = (kNumTokens > 8) ? 1 : 0;
  static_assert(kSmReductionPad == 0 || kSmReductionPad == 1, "kSmReductionPad only supports 0 or 1");

  int const n_idx = blockIdx.x;
  int const tid = threadIdx.x;
  int const warp_id = tid / kWarpSize;
  int const lane_id = tid % kWarpSize;

  float acc[kNumTokens] = {};
  __shared__ float sm_reduction[kNumTokens][kNumWarps + kSmReductionPad];

  T const* b_col = mat_b + n_idx * kHiddenDim;

  PDLWaitPrimary<kUsePDL>();

  int k_base = tid * VPT;
#pragma unroll
  for (int ki = 0; ki < kIters; ++ki, k_base += kElemsPerKIter) {
    AlignedVector<bf16_t, VPT> b_vec;
    b_vec.load(b_col + k_base);
#pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; ++m_idx) {
      AlignedVector<bf16_t, VPT> a_vec;
      a_vec.load(mat_a + m_idx * kHiddenDim + k_base);
#pragma unroll
      for (int k = 0; k < VPT; ++k) {
        acc[m_idx] += cast<float>(a_vec[k]) * cast<float>(b_vec[k]);
      }
    }
  }

#pragma unroll
  for (int m_idx = 0; m_idx < kNumTokens; ++m_idx) {
    float sum = warp::reduce_sum(acc[m_idx]);
    if (lane_id == 0) {
      sm_reduction[m_idx][warp_id] = sum;
    }
  }

  __syncthreads();

  if (warp_id == 0 && lane_id < kNumTokens) {
    float final_sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kNumWarps; ++w) {
      final_sum += sm_reduction[lane_id][w];
    }
    out[lane_id * kNumExperts + n_idx] = cast<OutT>(final_sum);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <typename T, typename OutT, int kNumTokens, int kNumExperts, int kHiddenDim, bool kUsePDL>
void invokeRouterGemm(OutT* output, T const* mat_a, T const* mat_b, DLDevice device) {
  constexpr int VPT = 16 / sizeof(T);
  constexpr int kBlockSize = 128;
  constexpr auto kernel = router_gemm_kernel<T, OutT, kBlockSize, VPT, kNumTokens, kNumExperts, kHiddenDim, kUsePDL>;
  host::LaunchKernel(kNumExperts, kBlockSize, device).enable_pdl(kUsePDL)(kernel, output, mat_a, mat_b);
}

// Dispatch runtime num_tokens to compile-time template parameter [kBegin, kEnd]
template <int kBegin, int kEnd, typename OutT, int kNumExperts, int kHiddenDim, bool kUsePDL>
struct RouterGemmDispatcher {
  static void run(int num_tokens, OutT* output, bf16_t const* mat_a, bf16_t const* mat_b, DLDevice device) {
    if (num_tokens == kBegin) {
      invokeRouterGemm<bf16_t, OutT, kBegin, kNumExperts, kHiddenDim, kUsePDL>(output, mat_a, mat_b, device);
    } else {
      RouterGemmDispatcher<kBegin + 1, kEnd, OutT, kNumExperts, kHiddenDim, kUsePDL>::run(
          num_tokens, output, mat_a, mat_b, device);
    }
  }
};

// Base case: kBegin == kEnd
template <int kEnd, typename OutT, int kNumExperts, int kHiddenDim, bool kUsePDL>
struct RouterGemmDispatcher<kEnd, kEnd, OutT, kNumExperts, kHiddenDim, kUsePDL> {
  static void run(int num_tokens, OutT* output, bf16_t const* mat_a, bf16_t const* mat_b, DLDevice device) {
    if (num_tokens == kEnd) {
      invokeRouterGemm<bf16_t, OutT, kEnd, kNumExperts, kHiddenDim, kUsePDL>(output, mat_a, mat_b, device);
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
  static_assert(
      kNumExperts == kDefaultNumExperts || kNumExperts == kKimiK2NumExperts,
      "required num_experts == 256 or num_experts == 384");

  using OutT = std::conditional_t<kOutFloat, fp32_t, bf16_t>;

  static void
  run(const tvm::ffi::TensorView mat_a, const tvm::ffi::TensorView mat_b, const tvm::ffi::TensorView output) {
    using namespace host;

    auto M = SymbolicSize{"num_tokens"};
    auto K = SymbolicSize{"hidden_dim"};
    auto N = SymbolicSize{"num_experts"};
    auto device = SymbolicDevice{};
    K.set_value(kHiddenDim);
    N.set_value(kNumExperts);
    device.set_options<kDLCUDA>();

    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(mat_a);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(device).verify(mat_b);
    TensorMatcher({M, N}).with_dtype<OutT>().with_device(device).verify(output);

    const int num_tokens = static_cast<int>(M.unwrap());

    RouterGemmDispatcher<1, 16, OutT, kNumExperts, kHiddenDim, kUsePDL>::run(
        num_tokens,
        static_cast<OutT*>(output.data_ptr()),
        static_cast<bf16_t const*>(mat_a.data_ptr()),
        static_cast<bf16_t const*>(mat_b.data_ptr()),
        device.unwrap());
  }
};

}  // namespace
