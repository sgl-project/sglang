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

template <typename T, int kNumTokens, int kNumExperts, int kHiddenDim, bool kUsePDL>
void invokeRouterGemmFloatOutput(float* output, T const* mat_a, T const* mat_b, cudaStream_t stream);

template <typename T, int kNumTokens, int kNumExperts, int kHiddenDim, bool kUsePDL>
void invokeRouterGemmBf16Output(__nv_bfloat16* output, T const* mat_a, T const* mat_b, cudaStream_t stream);

}  // namespace dsv3_router_jit

namespace {

using bf16_t = __nv_bfloat16;

static constexpr int kDefaultNumExperts = 256;
static constexpr int kKimiK2NumExperts = 384;
static constexpr int kDefaultHiddenDim = 7168;

template <int kBegin, int kEnd, int kNumExperts, int kHiddenDim, bool kUsePDL>
struct LoopUnroller {
  static void
  unroll_float_output(int num_tokens, float* output, bf16_t const* input, bf16_t const* weights, cudaStream_t stream) {
    if (num_tokens == kBegin) {
      dsv3_router_jit::invokeRouterGemmFloatOutput<bf16_t, kBegin, kNumExperts, kHiddenDim, kUsePDL>(
          output, input, weights, stream);
    } else {
      LoopUnroller<kBegin + 1, kEnd, kNumExperts, kHiddenDim, kUsePDL>::unroll_float_output(
          num_tokens, output, input, weights, stream);
    }
  }

  static void
  unroll_bf16_output(int num_tokens, bf16_t* output, bf16_t const* input, bf16_t const* weights, cudaStream_t stream) {
    if (num_tokens == kBegin) {
      dsv3_router_jit::invokeRouterGemmBf16Output<bf16_t, kBegin, kNumExperts, kHiddenDim, kUsePDL>(
          output, input, weights, stream);
    } else {
      LoopUnroller<kBegin + 1, kEnd, kNumExperts, kHiddenDim, kUsePDL>::unroll_bf16_output(
          num_tokens, output, input, weights, stream);
    }
  }
};

template <int kEnd, int kNumExperts, int kHiddenDim, bool kUsePDL>
struct LoopUnroller<kEnd, kEnd, kNumExperts, kHiddenDim, kUsePDL> {
  static void
  unroll_float_output(int num_tokens, float* output, bf16_t const* input, bf16_t const* weights, cudaStream_t stream) {
    if (num_tokens == kEnd) {
      dsv3_router_jit::invokeRouterGemmFloatOutput<bf16_t, kEnd, kNumExperts, kHiddenDim, kUsePDL>(
          output, input, weights, stream);
    }
  }

  static void
  unroll_bf16_output(int num_tokens, bf16_t* output, bf16_t const* input, bf16_t const* weights, cudaStream_t stream) {
    if (num_tokens == kEnd) {
      dsv3_router_jit::invokeRouterGemmBf16Output<bf16_t, kEnd, kNumExperts, kHiddenDim, kUsePDL>(
          output, input, weights, stream);
    }
  }
};

template <bool kUsePDL, typename OutDType>
struct dsv3_router_gemm_kernel {
  static_assert(std::is_same_v<OutDType, bf16_t> || std::is_same_v<OutDType, fp32_t>);

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

    RuntimeCheck(hidden_dim == kDefaultHiddenDim, "required hidden_dim == 7168");
    RuntimeCheck(
        num_experts == kDefaultNumExperts || num_experts == kKimiK2NumExperts,
        "required num_experts == 256 or num_experts == 384");
    RuntimeCheck(num_tokens >= 1 && num_tokens <= 16, "required 1 <= num_tokens <= 16");

    auto cc_major = runtime::get_cc_major(device.unwrap().device_id);
    RuntimeCheck(cc_major >= 9, "required CUDA ARCH >= SM_90");

    DLDevice dl_device = device.unwrap();
    cudaStream_t stream = LaunchKernel::resolve_device(dl_device);

    auto* output_ptr = static_cast<OutDType*>(output.data_ptr());
    auto* mat_a_ptr = static_cast<bf16_t const*>(mat_a.data_ptr());
    auto* mat_b_ptr = static_cast<bf16_t const*>(mat_b.data_ptr());

    if constexpr (std::is_same_v<OutDType, fp32_t>) {
      if (num_experts == kDefaultNumExperts) {
        LoopUnroller<1, 16, kDefaultNumExperts, kDefaultHiddenDim, kUsePDL>::unroll_float_output(
            num_tokens, static_cast<float*>(output_ptr), mat_a_ptr, mat_b_ptr, stream);
      } else {
        LoopUnroller<1, 16, kKimiK2NumExperts, kDefaultHiddenDim, kUsePDL>::unroll_float_output(
            num_tokens, static_cast<float*>(output_ptr), mat_a_ptr, mat_b_ptr, stream);
      }
    } else {
      if (num_experts == kDefaultNumExperts) {
        LoopUnroller<1, 16, kDefaultNumExperts, kDefaultHiddenDim, kUsePDL>::unroll_bf16_output(
            num_tokens, static_cast<bf16_t*>(output_ptr), mat_a_ptr, mat_b_ptr, stream);
      } else {
        LoopUnroller<1, 16, kKimiK2NumExperts, kDefaultHiddenDim, kUsePDL>::unroll_bf16_output(
            num_tokens, static_cast<bf16_t*>(output_ptr), mat_a_ptr, mat_b_ptr, stream);
      }
    }
  }
};

}  // namespace
