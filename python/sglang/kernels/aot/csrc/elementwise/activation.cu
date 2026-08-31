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

#include <cuda_fp8.h>

#include <flashinfer/activation.cuh>

#include "utils.h"

#else
#include "hip/hip_act_and_mul.cuh"
#endif

// Adapted from flashinfer activation
// https://github.com/flashinfer-ai/flashinfer/blob/4e8eb1879f9c3ba6d75511e5893183bf8f289a62/csrc/activation.cu#L44

namespace detail {

template <typename T>
__device__ __forceinline__ float to_f32(const T& x) {
#if USE_ROCM
  return castToFloat(x);
#else
  return static_cast<float>(x);
#endif
}

template <typename T>
__device__ __forceinline__ T from_f32(float f32) {
#if USE_ROCM
  return castFromFloat<T>(f32);
#else
  return static_cast<T>(f32);
#endif
}

}  // namespace detail

template <typename T>
__device__ __forceinline__ T silu(const T& x) {
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val / (1.0f + expf(-f32_val)));
}

template <typename T>
__device__ __forceinline__ T gelu(const T& x) {
  constexpr float kAlpha = M_SQRT1_2;
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val * (0.5f * (1.0f + erf(f32_val * kAlpha))));
}

// gelu_quick(x) = x * torch.sigmoid(1.702 * x)
template <typename T>
__device__ __forceinline__ T gelu_quick_act(const T& x) {
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val / (1.0f + expf(-f32_val * 1.702f)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh(const T& x) {
  constexpr float kAlpha = 0.044715f;
  constexpr float kBeta = 0.7978845608028654f;
  float f32_val = detail::to_f32(x);
  const float cdf = 0.5f * (1.0f + tanhf((kBeta * (f32_val + kAlpha * f32_val * f32_val * f32_val))));
  return detail::from_f32<T>(f32_val * cdf);
}

void silu_and_mul(at::Tensor& out, at::Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
#if USE_ROCM
    sgl_hip::activation::act_and_mul_kernel<c_type, silu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#else
    flashinfer::activation::act_and_mul_kernel<c_type, silu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#endif
    return true;
  });
}

#ifndef USE_ROCM
template <typename T>
__device__ __forceinline__ T
fused_swiglu_value(T gate_value, T up_value, const float swiglu_limit, const bool has_swiglu_limit) {
  if (has_swiglu_limit) {
    gate_value = static_cast<T>(fminf(static_cast<float>(gate_value), swiglu_limit));
    up_value = static_cast<T>(fmaxf(fminf(static_cast<float>(up_value), swiglu_limit), -swiglu_limit));
  }
  return static_cast<T>(silu(static_cast<float>(gate_value)) * static_cast<float>(up_value));
}

template <typename T>
__global__ void fused_swiglu_quant_fp8_kernel(
    const T* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output_q,
    float* __restrict__ output_s,
    const float* __restrict__ residual,
    const int32_t* __restrict__ expert_offsets,
    const int64_t hidden_dim,
    const int64_t num_tokens,
    const int64_t num_experts,
    const float swiglu_limit,
    const bool has_swiglu_limit) {
  const int64_t token = blockIdx.x;
  if (token >= num_tokens) return;

  const T* gate = input + token * hidden_dim * 2;
  const T* up = gate + hidden_dim;
  float max_value = 0.0f;
  constexpr int64_t kVecSize = 16 / sizeof(T);

  // CUDA allocations are sufficiently aligned, and a full 16-byte vector keeps
  // both gate and up 16-byte aligned for every token. Otherwise, leave the
  // complete row to the scalar path, which also serves as the general tail.
  const int64_t vectorized_dim = hidden_dim % kVecSize == 0 ? hidden_dim : 0;

  // Match flashinfer::activation::act_and_mul_kernel: convert the inputs to
  // FP32, evaluate SiLU and the multiply in FP32, then round the product once
  // to BF16/FP16. The rounded product is what the legacy quantizer observes.
  const uint4* gate_vec = reinterpret_cast<const uint4*>(gate);
  const uint4* up_vec = reinterpret_cast<const uint4*>(up);
  for (int64_t vec = threadIdx.x; vec < vectorized_dim / kVecSize; vec += blockDim.x) {
    const uint4 gate_pack = gate_vec[vec];
    const uint4 up_pack = up_vec[vec];
    const T* gate_values = reinterpret_cast<const T*>(&gate_pack);
    const T* up_values = reinterpret_cast<const T*>(&up_pack);
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      const T value = fused_swiglu_value(gate_values[j], up_values[j], swiglu_limit, has_swiglu_limit);
      max_value = fmaxf(max_value, fabsf(static_cast<float>(value)));
    }
  }
  for (int64_t i = vectorized_dim + threadIdx.x; i < hidden_dim; i += blockDim.x) {
    const T value = fused_swiglu_value(gate[i], up[i], swiglu_limit, has_swiglu_limit);
    max_value = fmaxf(max_value, fabsf(static_cast<float>(value)));
  }

  max_value = blockReduceMax(max_value);

  __shared__ float scale;
  __shared__ float scale_inv;
  if (threadIdx.x == 0) {
    scale = max_value / FP8_E4M3_MAX;
    scale_inv = scale == 0.0f ? 0.0f : 1.0f / scale;

    // upper_bound(expert_offsets, token) - 1. This handles empty experts and
    // both the E=1 and E=256 production configurations without a host read.
    int lo = 0;
    int hi = static_cast<int>(num_experts);
    while (lo + 1 < hi) {
      const int mid = (lo + hi) / 2;
      if (token >= expert_offsets[mid]) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    output_s[token] = scale * residual[lo];
  }
  __syncthreads();

  for (int64_t vec = threadIdx.x; vec < vectorized_dim / kVecSize; vec += blockDim.x) {
    const uint4 gate_pack = gate_vec[vec];
    const uint4 up_pack = up_vec[vec];
    const T* gate_values = reinterpret_cast<const T*>(&gate_pack);
    const T* up_values = reinterpret_cast<const T*>(&up_pack);
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      const int64_t i = vec * kVecSize + j;
      const T value = fused_swiglu_value(gate_values[j], up_values[j], swiglu_limit, has_swiglu_limit);
      float quant_value = static_cast<float>(value) * scale_inv;
      quant_value = fmaxf(fminf(quant_value, FP8_E4M3_MAX), -FP8_E4M3_MAX);
      output_q[token * hidden_dim + i] = static_cast<__nv_fp8_e4m3>(quant_value);
    }
  }
  for (int64_t i = vectorized_dim + threadIdx.x; i < hidden_dim; i += blockDim.x) {
    const T value = fused_swiglu_value(gate[i], up[i], swiglu_limit, has_swiglu_limit);
    float quant_value = static_cast<float>(value) * scale_inv;
    quant_value = fmaxf(fminf(quant_value, FP8_E4M3_MAX), -FP8_E4M3_MAX);
    output_q[token * hidden_dim + i] = static_cast<__nv_fp8_e4m3>(quant_value);
  }
}

void fused_swiglu_quant_fp8(
    const at::Tensor& input,
    at::Tensor& output_q,
    at::Tensor& output_s,
    const at::Tensor& residual,
    const at::Tensor& expert_offsets,
    int64_t num_experts,
    double swiglu_limit,
    bool has_swiglu_limit) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);
  CHECK_INPUT(residual);
  CHECK_INPUT(expert_offsets);
  TORCH_CHECK(input.dim() == 2 && input.size(1) % 2 == 0, "input must have shape [M, 2N]");
  const int64_t num_tokens = input.size(0);
  const int64_t hidden_dim = input.size(1) / 2;
  TORCH_CHECK(output_q.sizes() == at::IntArrayRef({num_tokens, hidden_dim}), "output_q must have shape [M, N]");
  TORCH_CHECK(output_q.scalar_type() == at::kFloat8_e4m3fn, "output_q must be float8_e4m3fn");
  TORCH_CHECK(output_s.numel() == num_tokens && output_s.scalar_type() == at::kFloat, "output_s must be float32 [M]");
  TORCH_CHECK(residual.numel() == num_experts && residual.scalar_type() == at::kFloat, "residual must be float32 [E]");
  TORCH_CHECK(
      expert_offsets.numel() == num_experts + 1 && expert_offsets.scalar_type() == at::kInt,
      "expert_offsets must be int32 [E + 1]");
  TORCH_CHECK(num_experts >= 1 && num_experts <= 256, "num_experts must be in [1, 256]");

  if (num_tokens == 0) return;
  constexpr int kThreads = 256;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    fused_swiglu_quant_fp8_kernel<c_type><<<num_tokens, kThreads, 0, stream>>>(
        static_cast<const c_type*>(input.data_ptr()),
        static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        static_cast<const float*>(residual.data_ptr()),
        static_cast<const int32_t*>(expert_offsets.data_ptr()),
        hidden_dim,
        num_tokens,
        num_experts,
        static_cast<float>(swiglu_limit),
        has_swiglu_limit);
    return true;
  });
}
#endif

void gelu_tanh_and_mul(at::Tensor& out, at::Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
#if USE_ROCM
    sgl_hip::activation::act_and_mul_kernel<c_type, gelu_tanh>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#else
    flashinfer::activation::act_and_mul_kernel<c_type, gelu_tanh>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#endif
    return true;
  });
}

void gelu_and_mul(at::Tensor& out, at::Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
#if USE_ROCM
    sgl_hip::activation::act_and_mul_kernel<c_type, gelu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#else
    flashinfer::activation::act_and_mul_kernel<c_type, gelu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#endif

    return true;
  });
}

#if USE_ROCM
void gelu_quick(at::Tensor& out, const at::Tensor& input) {
  int d = input.size(-1);
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
    sgl_hip::activation::act_only_kernel<c_type, gelu_quick_act>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);

    return true;
  });
}
#endif
