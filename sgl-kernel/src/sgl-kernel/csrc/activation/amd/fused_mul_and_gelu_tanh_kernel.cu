/* Copyright 2025 SGLang Team. All Rights Reserved.

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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "act_and_mul_internal.cuh"

namespace flashinfer {
namespace activation {

template <typename T>
__device__ __forceinline__ T gelu_tanh(const T& x) {
  constexpr float kAlpha = 0.044715f;
  constexpr float kBeta = 0.7978845608028654f;
  const float f32_val = castToFloat(x);

  const float f32_val_pow_of_3 = __powf(f32_val, 3.f);
  const float cdf =
    0.5f * (1.0f + tanhf( ( kBeta * ( f32_val + kAlpha  * f32_val_pow_of_3 ) ) ) );

  return castFrom<T>( f32_val * cdf );
}

} // activation
} // flashinfer


void gelu_tanh_and_mul(at::Tensor& out, at::Tensor& input, int64_t cuda_stream) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));

    flashinfer::activation::act_and_mul_kernel<c_type, flashinfer::activation::gelu_tanh><<<grid, block, 0, stream>>>(
        static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);

    return true;
  });
}
