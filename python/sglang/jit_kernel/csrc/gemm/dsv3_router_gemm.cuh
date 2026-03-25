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
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cuda_runtime.h>
#include <type_traits>

namespace {

static constexpr int kDefaultNumExperts = 256;
static constexpr int kKimiK2NumExperts = 384;
static constexpr int kDefaultHiddenDim = 7168;

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
  constexpr int kSmReductionPad = (kNumTokens > 8) ? 1 : 0;
  static_assert(kSmReductionPad == 0 || kSmReductionPad == 1, "kSmReductionPad only supports 0 or 1");

  int const n_idx = blockIdx.x;
  int const tid = threadIdx.x;
  int const warp_id = tid / kWarpSize;
  int const lane_id = tid % kWarpSize;

  float acc[kNumTokens] = {};

  __shared__ float sm_reduction[kNumTokens][kNumWarps + kSmReductionPad];

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
        acc[m_idx] += device::cast<float>(a_vec[k]) * device::cast<float>(b_vec[k]);
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

  if (warp_id == 0 && lane_id < kNumTokens) {
    float final_sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kNumWarps; ++w) {
      final_sum += sm_reduction[lane_id][w];
    }
    out[lane_id * kNumExperts + n_idx] = device::cast<OutT>(final_sum);
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <typename T, typename OutT, int kNumTokens, int kNumExperts, int kHiddenDim, bool kUsePDL>
void invokeRouterGemm(OutT* output, T const* mat_a, T const* mat_b, cudaStream_t stream) {
  constexpr int VPT = 16 / sizeof(T);
  constexpr int kBlockSize = 128;

  constexpr auto kernel = router_gemm_kernel<T, OutT, kBlockSize, VPT, kNumTokens, kNumExperts, kHiddenDim, kUsePDL>;

  host::LaunchKernel(dim3(kNumExperts), dim3(kBlockSize), stream).enable_pdl(kUsePDL)(kernel, output, mat_a, mat_b);
}

template <bool kUsePDL, typename OutDType, int kNumExperts, int kNumTokens>
struct DSV3RouterGEMMKernel {
  static_assert(std::is_same_v<OutDType, bf16_t> || std::is_same_v<OutDType, fp32_t>);
  static_assert(
      kNumExperts == kDefaultNumExperts || kNumExperts == kKimiK2NumExperts,
      "required num_experts == 256 or num_experts == 384");
  static_assert(kNumTokens >= 1 && kNumTokens <= 16, "required 1 <= kNumTokens <= 16");

  static void
  run(const tvm::ffi::TensorView output, const tvm::ffi::TensorView mat_a, const tvm::ffi::TensorView mat_b) {
    using namespace host;

    auto num_tokens_sym = SymbolicSize{"num_tokens"};
    auto num_experts_sym = SymbolicSize{"num_experts"};
    auto hidden_dim_sym = SymbolicSize{"hidden_dim"};

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({num_tokens_sym, hidden_dim_sym})  //
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(mat_a);

    TensorMatcher({num_experts_sym, hidden_dim_sym})  //
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(mat_b);

    TensorMatcher({num_tokens_sym, num_experts_sym})  //
        .with_dtype<OutDType>()
        .with_device(device)
        .verify(output);

    const auto num_tokens = static_cast<int>(num_tokens_sym.unwrap());
    const auto num_experts = static_cast<int>(num_experts_sym.unwrap());
    const auto hidden_dim = static_cast<int>(hidden_dim_sym.unwrap());

    RuntimeCheck(num_tokens == kNumTokens, "required num_tokens == ", kNumTokens);
    RuntimeCheck(num_experts == kNumExperts, "required num_experts == ", kNumExperts);
    RuntimeCheck(hidden_dim == kDefaultHiddenDim, "required hidden_dim == 7168");

    auto cc_major = runtime::get_cc_major(device.unwrap().device_id);
    RuntimeCheck(cc_major >= 9, "required CUDA ARCH >= SM_90");

    DLDevice dl_device = device.unwrap();
    cudaStream_t stream = LaunchKernel::resolve_device(dl_device);

    auto* output_ptr = static_cast<OutDType*>(output.data_ptr());
    auto* mat_a_ptr = static_cast<bf16_t const*>(mat_a.data_ptr());
    auto* mat_b_ptr = static_cast<bf16_t const*>(mat_b.data_ptr());

    invokeRouterGemm<bf16_t, OutDType, kNumTokens, kNumExperts, kDefaultHiddenDim, kUsePDL>(
        output_ptr, mat_a_ptr, mat_b_ptr, stream);
  }
};

}  // namespace
