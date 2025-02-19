/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#ifndef USE_ROCM

#include <flashinfer/activation.cuh>

#include "pytorch_extension_utils.h"
#include "utils.h"

#else
#include "act_and_mul_internal.cuh"
#endif

// [flashinfer
// activation(https://github.com/flashinfer-ai/flashinfer/blob/4e8eb1879f9c3ba6d75511e5893183bf8f289a62/csrc/activation.cu#L44)

template <typename T>
__device__ __forceinline__ T silu(const T& x) {
#if USE_ROCM
  float f32_val = castToFloat(x);
  return castFromFloat<T>(f32_val / (1.0f + expf(-f32_val)));
#else
  float f32_val = static_cast<float>(x);
  return static_cast<T>(f32_val / (1.0f + expf(-f32_val)));
#endif
}

template <typename T>
__device__ __forceinline__ T gelu(const T& x) {
  constexpr float kAlpha = M_SQRT1_2;
#if USE_ROCM
  float f32_val = castToFloat(x);
  return castFromFloat<T>(f32_val * (0.5f * (1.0f + erf(f32_val * kAlpha))));
#else
  float f32_val = static_cast<float>(x);
  return static_cast<T>(f32_val * (0.5f * (1.0f + erf(f32_val * kAlpha))));
#endif
}

template <typename T>
__device__ __forceinline__ T gelu_tanh(const T& x) {
  constexpr float kAlpha = 0.044715f;
  constexpr float kBeta = 0.7978845608028654f;
#if USE_ROCM
  const float f32_val = castToFloat(x);
#else
  const float f32_val = static_cast<float>(x);
#endif

  const float cdf = 0.5f * (1.0f + tanhf((kBeta * (f32_val + kAlpha * f32_val * f32_val * f32_val))));

#if USE_ROCM
  return castFromFloat<T>(f32_val * cdf);
#else
  return static_cast<T>(f32_val * cdf);
#endif
}

void silu_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
    flashinfer::activation::act_and_mul_kernel<c_type, silu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);

    return true;
  });
}

void gelu_tanh_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
    flashinfer::activation::act_and_mul_kernel<c_type, gelu_tanh>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);

    return true;
  });
}

void gelu_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
    flashinfer::activation::act_and_mul_kernel<c_type, gelu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);

    return true;
  });
}
