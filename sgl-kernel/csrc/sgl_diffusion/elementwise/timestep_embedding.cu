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
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/all.h>

#include <cassert>
#include <cmath>

#include "utils.h"

template <bool flip_sin_to_cos = false, typename T_IN>
__global__ void timestep_embedding_kernel(
    T_IN* t_ptr, float* output_ptr, int dim, float neg_log_max_period, float scale, int batch_size) {
  // Get the timestep for this batch
  int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
  if (row_idx >= batch_size) {
    return;
  }
  // Use the portable LDG helper (maps to __ldg on CUDA, plain load on ROCm/HIP).
  float t_val = castToFloat(SGLANG_LDG(&t_ptr[row_idx]));
  float* output_batch_base_ptr = output_ptr + row_idx * dim;

  // Calculate half dimension
  int half_dim = dim / 2;
  int thread_offset = threadIdx.x % blockDim.x;
  while (thread_offset * 4 < half_dim) {
    float4* top_half;
    float4* bottom_half;
    if constexpr (flip_sin_to_cos == false) {
      bottom_half = reinterpret_cast<float4*>(output_batch_base_ptr + thread_offset * 4);
      top_half = reinterpret_cast<float4*>(output_batch_base_ptr + half_dim + thread_offset * 4);
    } else {
      top_half = reinterpret_cast<float4*>(output_batch_base_ptr + thread_offset * 4);
      bottom_half = reinterpret_cast<float4*>(output_batch_base_ptr + half_dim + thread_offset * 4);
    }

    float4 vals;
    vals.x = scale * t_val * expf(neg_log_max_period * __int2float_rn(thread_offset * 4 + 0));
    vals.y = scale * t_val * expf(neg_log_max_period * __int2float_rn(thread_offset * 4 + 1));
    vals.z = scale * t_val * expf(neg_log_max_period * __int2float_rn(thread_offset * 4 + 2));
    vals.w = scale * t_val * expf(neg_log_max_period * __int2float_rn(thread_offset * 4 + 3));

    float4 sin_vals;
    sin_vals.x = cosf(vals.x);
    sin_vals.y = cosf(vals.y);
    sin_vals.z = cosf(vals.z);
    sin_vals.w = cosf(vals.w);
    *top_half = sin_vals;  // STG.128

    float4 cos_vals;
    cos_vals.x = sinf(vals.x);
    cos_vals.y = sinf(vals.y);
    cos_vals.z = sinf(vals.z);
    cos_vals.w = sinf(vals.w);
    *bottom_half = cos_vals;  // STG.128

    thread_offset += blockDim.x;
  }
}

torch::Tensor timestep_embedding(
    const torch::Tensor& t,
    torch::Tensor& output,
    int64_t dim,
    bool flip_sin_to_cos,
    double downscale_freq_shift,
    double scale,
    int64_t max_period) {
  TORCH_CHECK(t.dim() == 1 and t.stride(0) == 1, "t should be 1D");
  TORCH_CHECK(output.dim() == 2 and output.is_contiguous(), "output should be a contiguous 2D tensor.");

  const int batch_size = static_cast<int>(t.size(0));
  TORCH_CHECK(output.size(0) == batch_size, "Output batch size doesn't match t");
  TORCH_CHECK(output.size(1) == dim, "Output feature size doesn't match dim");

  TORCH_CHECK(t.device().is_cuda(), "t must be a CUDA tensor");
  TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(t.device() == output.device(), "t and output must be on the same device");

  // To align with timestep_embedding python code.
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Float, "Output buffer should be float32.");

  TORCH_CHECK(dim % 8 == 0, "dim should align to 8");
  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr int MAX_THREADS_PER_BLOCK = 1024;
  constexpr int MIN_THREADS_PER_BLOCK = 128;
  int half_dim = dim / 2;
  int num_threads_per_row = min(MAX_THREADS_PER_BLOCK, half_dim / 4);
  int num_rows = (MIN_THREADS_PER_BLOCK + num_threads_per_row - 1) / num_threads_per_row;

  dim3 grid((batch_size + num_rows - 1) / num_rows);
  // assert float4 vectorize output
  dim3 block(num_threads_per_row, num_rows);
  float neg_log_max_period =
      std::log(static_cast<float>(max_period)) * (-1.0f) / (static_cast<float>(half_dim) - downscale_freq_shift);

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, t.scalar_type(), "timestep_embedding_kernel", [&] {
        if (flip_sin_to_cos == true) {
          timestep_embedding_kernel<true><<<grid, block, 0, stream>>>(
              reinterpret_cast<scalar_t*>(t.data_ptr()),
              reinterpret_cast<float*>(output.data_ptr()),
              static_cast<int>(dim),
              static_cast<float>(neg_log_max_period),
              static_cast<float>(scale),
              static_cast<int>(batch_size));
        } else {
          timestep_embedding_kernel<false><<<grid, block, 0, stream>>>(
              reinterpret_cast<scalar_t*>(t.data_ptr()),
              reinterpret_cast<float*>(output.data_ptr()),
              static_cast<int>(dim),
              static_cast<float>(neg_log_max_period),
              static_cast<float>(scale),
              static_cast<int>(batch_size));
        }
      });

  return output;
}
