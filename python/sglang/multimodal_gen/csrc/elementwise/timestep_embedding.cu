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
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <torch/extension.h>

// TODO: temp include for debug
#include "utils.h"

// // TODO:
// #include "sgl_kernel_ops.h"
// TORCH_LIBRARY_EXPAND(sgl_kernel, m) {
//   m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
//   m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);
// }
// # TODO: namespace
// REGISTER_EXTENSION(common_ops)

// TODO: hard code for now.
// at::vec::convert_to_float instead later
template <typename T>
__device__ float convert_to_float(T x) {
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

template <typename T>
__device__ __nv_bfloat16 convert_to_bfloat16(T x) {
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

template <typename T>
__device__ __half convert_to_float16(T x) {
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

template <typename O, typename T>
__device__ O cast_to(T x) {
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


// TODO: template review
// assert operations is float??
__device__ float calculate_frequency_and_angle(float t_val, int freq_idx, int half, int max_period) {
    float log_max_period = logf(static_cast<float>(max_period));
    float freqs = expf(-log_max_period * static_cast<float>(freq_idx) / static_cast<float>(half));
    return t_val * freqs;
}



template<typename T, typename O>
__global__ void timestep_embedding_kernel(
    T* t_ptr,
    O* output_ptr,
    int B,
    int dim,
    int max_period,
    int stride_t_b,
    int stride_out_b,
    int stride_out_d,
    int BLOCK_SIZE_DIM
) {
    // TODO: add comments
    int pid_b = blockIdx.x;
    int pid_d = blockIdx.y;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Get the timestep for this batch
    // TODO: vect
    float t_val = cast_to<float>(t_ptr[pid_b * stride_t_b]);

    // Calculate half dimension
    int half = dim / 2;

    // Create range of indices for this block
    int d_start = pid_d * BLOCK_SIZE_DIM;
    int d_end = d_start + BLOCK_SIZE_DIM;

    for (int d_idx = d_start + tid; d_idx < min(d_end, half); d_idx += num_threads) {
        // // TODO: remove debug assert later
        // assert(d_idx < half);
        float angles = calculate_frequency_and_angle(t_val, d_idx % half, half, max_period);
        int out_idx_first = pid_b * stride_out_b + d_idx * stride_out_d;
        output_ptr[out_idx_first] = cast_to<O>(cosf(angles));
        int out_idx_second = pid_b * stride_out_b + (d_idx + half) * stride_out_d;
        output_ptr[out_idx_second] = cast_to<O>(sinf(angles));
    }

    // TODO: review, assert output buffer is zero init?
    // if dim % 2:
    //     embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if (dim % 2 != 0) {
        int out_idx_pad = pid_b * stride_out_b + (dim-1) * stride_out_d;
        output_ptr[out_idx_pad] = 0.;
    }
}

// NOTE: output always be float32 now. According to python code: timestep_embedding
torch::Tensor timestep_embedding_kernel_cuda(
    torch::Tensor &t,
    torch::Tensor &output,
    int dim,
    int max_period
    )
{
    TORCH_CHECK(t.dim() == 1, "t should be 1D");
    TORCH_CHECK(output.dim() == 2, "output should be 2D");

    const int B = t.size(0);
    TORCH_CHECK(output.size(0) == B, "Output batch size doesn't match t");
    TORCH_CHECK(output.size(1) == dim, "Output feature size doesn't match dim");

    TORCH_CHECK(t.device().is_cuda(), "t must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(t.device() == output.device(), "t and output must be on the same device");

    // TODO: review
    // To align with timestep_embedding python code.
    TORCH_CHECK(output.scalar_type() == at::ScalarType::Float, "Output buffer should be float32.");

    auto stream = at::cuda::getCurrentCUDAStream();

    // TODO: tuning
    int BLOCK_SIZE_DIM = 256;
    const int num_threads = 256;
    const int half = dim / 2;

    const dim3 grid_size(B, (half + BLOCK_SIZE_DIM - 1) / BLOCK_SIZE_DIM);
    const dim3 block_size(num_threads);

    int stride_t_b = t.stride(0);
    int stride_out_b = output.stride(0);
    int stride_out_d = output.stride(1);

    DISPATCH_FLOAT_TYPES(t.scalar_type(), "timestep_embedding_kernel", [&] {
      using t_type = scalar_t;
      using o_type = float;
      timestep_embedding_kernel<t_type, o_type><<<grid_size, block_size, 0, stream>>>(
          static_cast<t_type*>(t.data_ptr()),
          static_cast<o_type*>(output.data_ptr()),
          B,
          dim,
          max_period,
          stride_t_b,
          stride_out_b,
          stride_out_d,
          BLOCK_SIZE_DIM
      );
    });

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("timestep_embedding_kernel_cuda", &timestep_embedding_kernel_cuda, "timestep_embedding_kernel_cuda");
}

