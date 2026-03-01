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

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace dsv3_router_jit {

template <typename T, typename OutT, int kNumTokens, int kNumExperts, int kHiddenDim, bool kUsePDL>
void invokeRouterGemm(OutT* output, T const* mat_a, T const* mat_b, cudaStream_t stream);

}  // namespace dsv3_router_jit

namespace {

using bf16_t = __nv_bfloat16;

static constexpr int kDefaultNumExperts = 256;
static constexpr int kKimiK2NumExperts = 384;
static constexpr int kDefaultHiddenDim = 7168;

template <bool kUsePDL, typename OutDType, int kNumExperts, int kNumTokens>
struct dsv3_router_gemm_kernel {
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

    TensorMatcher({num_tokens_sym, hidden_dim_sym}).with_dtype<bf16_t>().with_device(device).verify(mat_a);
    TensorMatcher({num_experts_sym, hidden_dim_sym}).with_dtype<bf16_t>().with_device(device).verify(mat_b);
    TensorMatcher({num_tokens_sym, num_experts_sym}).with_dtype<OutDType>().with_device(device).verify(output);

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

    dsv3_router_jit::invokeRouterGemm<bf16_t, OutDType, kNumTokens, kNumExperts, kDefaultHiddenDim, kUsePDL>(
        output_ptr, mat_a_ptr, mat_b_ptr, stream);
  }
};

}  // namespace
