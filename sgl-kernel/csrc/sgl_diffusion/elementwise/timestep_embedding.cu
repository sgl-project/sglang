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
#include <cassert>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/all.h>
#include <torch/extension.h>

template <typename T> __device__ float convert_to_float(T x) {
  if constexpr (std::is_same_v<T, __half>) {
    return __half2float(x);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(x);
  } else if constexpr (std::is_same_v<T, float>) {
    return x;
  } else {
    // int, double
    return static_cast<float>(x);
  }
}

template <typename T> __device__ __nv_bfloat16 convert_to_bfloat16(T x) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return x;
  } else if constexpr (std::is_same_v<T, __half>) {
    return __float2bfloat16(__half2float(x));
  } else if constexpr (std::is_same_v<T, float>) {
    return __float2bfloat16(x);
  } else {
    return __float2bfloat16(static_cast<float>(x));
  }
}

template <typename T> __device__ __half convert_to_float16(T x) {
  if constexpr (std::is_same_v<T, __half>) {
    return x;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __float2half(__bfloat162float(x));
  } else if constexpr (std::is_same_v<T, float>) {
    return __float2half(x);
  } else {
    return __float2half(static_cast<float>(x));
  }
}

template <typename O, typename T> __device__ O cast_to(T x) {
  if constexpr (std::is_same_v<O, float>) {
    return convert_to_float(x);
  } else if constexpr (std::is_same_v<O, __half>) {
    return convert_to_float16(x);
  } else if constexpr (std::is_same_v<O, __nv_bfloat16>) {
    return convert_to_bfloat16(x);
  } else {
    return static_cast<O>(convert_to_float(x));
  }
}

template <typename T_IN>
__global__ void
timestep_embedding_kernel(T_IN* t_ptr, float* output_ptr, int dim, float neg_log_max_period) {
  // Get the timestep for this batch
  float t_val = cast_to<float>(__ldg(&t_ptr[blockIdx.x]));
  float* output_batch_base_ptr = output_ptr + blockIdx.x * dim;

  // Calculate half dimension
  int half_dim = dim / 2;
  float half_dimf = __int2float_rn(half_dim);
  int thread_offset = threadIdx.x;
  while (thread_offset * 4 < half_dim) {
    float4* top_half = reinterpret_cast<float4*>(output_batch_base_ptr + thread_offset * 4);
    float4* bottom_half = reinterpret_cast<float4*>(output_batch_base_ptr + half_dim + thread_offset * 4);

    float4 vals;
    vals.x = t_val * expf(neg_log_max_period * __int2float_rn(thread_offset * 4 + 0) / half_dimf);
    vals.y = t_val * expf(neg_log_max_period * __int2float_rn(thread_offset * 4 + 1) / half_dimf);
    vals.z = t_val * expf(neg_log_max_period * __int2float_rn(thread_offset * 4 + 2) / half_dimf);
    vals.w = t_val * expf(neg_log_max_period * __int2float_rn(thread_offset * 4 + 3) / half_dimf);

    float4 sin_vals;
    sin_vals.x = cosf(vals.x);
    sin_vals.y = cosf(vals.y);
    sin_vals.z = cosf(vals.z);
    sin_vals.w = cosf(vals.w);
    *top_half = sin_vals; // STG.128

    float4 cos_vals;
    cos_vals.x = sinf(vals.x);
    cos_vals.y = sinf(vals.y);
    cos_vals.z = sinf(vals.z);
    cos_vals.w = sinf(vals.w);
    *bottom_half = cos_vals;  // STG.128

    thread_offset += blockDim.x;
  }
}

// NOTE: output always be float32 now. According to python code:
// timestep_embedding
torch::Tensor timestep_embedding_kernel_cuda(torch::Tensor &t,
                                             torch::Tensor &output, int dim,
                                             int max_period) {
  TORCH_CHECK(t.dim() == 1 and t.stride(0) == 1, "t should be 1D");
  TORCH_CHECK(output.dim() == 2 and output.is_contiguous(), "output should be a contiguous 2D tensor.");

  int B = static_cast<int>(t.size(0));
  TORCH_CHECK(output.size(0) == B, "Output batch size doesn't match t");
  TORCH_CHECK(output.size(1) == dim, "Output feature size doesn't match dim");

  TORCH_CHECK(t.device().is_cuda(), "t must be a CUDA tensor");
  TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(t.device() == output.device(),
              "t and output must be on the same device");

  // To align with timestep_embedding python code.
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Float,
              "Output buffer should be float32.");

  TORCH_CHECK(dim % 2 == 0 && (dim / 2) % 4 == 0, "dim should align to 8");
  auto stream = at::cuda::getCurrentCUDAStream();
  
  constexpr int MAX_THREADS_PER_BLOCK = 1024;
  int half_dim = dim / 2;

  dim3 grid(B, 1, 1);
  dim3 block(min(MAX_THREADS_PER_BLOCK, half_dim / 4), 1, 1);
  float neg_log_max_period = std::log(static_cast<float>(max_period)) * (-1.0f);

  if (t.dtype() == torch::kBFloat16) {
    timestep_embedding_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(t.data_ptr()),
      reinterpret_cast<float*>(output.data_ptr()),
      dim,
      neg_log_max_period);
  } else if (t.dtype() == torch::kFloat16) {
    timestep_embedding_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<__half*>(t.data_ptr()),
      reinterpret_cast<float*>(output.data_ptr()),
      dim,
      neg_log_max_period);
  } else if (t.dtype() == torch::kFloat) {
    timestep_embedding_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<float*>(t.data_ptr()),
      reinterpret_cast<float*>(output.data_ptr()),
      dim,
      neg_log_max_period);
  } else {
    TORCH_CHECK(false, "Unsupported dtype.");
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("timestep_embedding_kernel_cuda", &timestep_embedding_kernel_cuda,
        "timestep_embedding_kernel_cuda");
}
