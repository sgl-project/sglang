/******************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

/**
 * \file
 * \brief cuda kernels to do layernorm on a device memory tensor with RowMajor layout.
 */
 
#include "cutlass/cutlass.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/tensor_ref.h"
#include "device_utils.h"
#include <cfloat>
#include <cassert>
#include <type_traits>
 
namespace cutlass {
 
/** \brief interface to do layernorm on a device memory tensor with RowMajor layout.
 * \tparam T: data type
 */
template <typename T>
void layernorm(cutlass::MatrixCoord tensor_size,
               TensorRef<T, layout::RowMajor> ref_output,
               TensorRef<T, layout::RowMajor> ref_input,
               TensorRef<T, layout::RowMajor> ref_gamma,
               TensorRef<T, layout::RowMajor> ref_beta,
               cudaStream_t stream);

/** \brief interface to do layernorm followed by fused scale/shift on a device memory tensor with RowMajor layout.
 *        Computes: y = (LayerNorm(x; gamma, beta)) * (1 + scale) + shift
 * \tparam T: data type
 */
template <typename T>
void layernorm_fused_scale_shift(cutlass::MatrixCoord tensor_size,
               TensorRef<T, layout::RowMajor> ref_output,
               TensorRef<T, layout::RowMajor> ref_input,
               TensorRef<T, layout::RowMajor> ref_gamma,
               TensorRef<T, layout::RowMajor> ref_beta,
               TensorRef<T, layout::RowMajor> ref_scale,
               TensorRef<T, layout::RowMajor> ref_shift,
               cudaStream_t stream);

               /**
 * output [m, n] row-major
 * input [m, n] row-major
 * gamma [n]
 * beta [n]
 * grid(m)
 * block(block_size) -- each block deals with block_size*ITEM_PER_THREAD*4 elements;
*/
template<typename T4, typename T, int ITEM_PER_THREAD>
__global__ void layernorm_twoPassAlgo_stored_locally_e4(T4* output,
                                                        const T4* input,
                                                        const T4* gamma,
                                                        const T4* beta,
                                                        const int m,
                                                        const int n)
{
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  T4 local_val[ITEM_PER_THREAD];
  const int n_4 = n / 4;
  int offset = m_idx * n_4;
  input += offset;
  output += offset;

  const T4 zero = {T(0.0f), T(0.0f), T(0.0f), T(0.0f)};
  #pragma UNROLL
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i*bdimx + tid;
    local_val[i] = index < n_4 ? input[index] : zero;
    local_sums[0] += static_cast<float>(local_val[i].x) + static_cast<float>(local_val[i].y) +
                     static_cast<float>(local_val[i].z) + static_cast<float>(local_val[i].w);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  }
  else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  #pragma UNROLL
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i*bdimx + tid;
    if (index < n_4){
      const float4 tmp = {static_cast<float>(local_val[i].x) - s_mean,
                          static_cast<float>(local_val[i].y) - s_mean,
                          static_cast<float>(local_val[i].z) - s_mean,
                          static_cast<float>(local_val[i].w) - s_mean};
      local_sums[0] += tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z + tmp.w * tmp.w;
    }
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  }
  else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + 1e-5);
  }
  __syncthreads();

  #pragma UNROLL
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i*bdimx + tid;
    if (index < n_4){
      const T4 gamma_val = gamma[index];
      const T4 beta_val = beta[index];
      T4 tmp;
      tmp.x = T((static_cast<float>(local_val[i].x) - s_mean)*s_variance*static_cast<float>(gamma_val.x) + static_cast<float>(beta_val.x));
      tmp.y = T((static_cast<float>(local_val[i].y) - s_mean)*s_variance*static_cast<float>(gamma_val.y) + static_cast<float>(beta_val.y));
      tmp.z = T((static_cast<float>(local_val[i].z) - s_mean)*s_variance*static_cast<float>(gamma_val.z) + static_cast<float>(beta_val.z));
      tmp.w = T((static_cast<float>(local_val[i].w) - s_mean)*s_variance*static_cast<float>(gamma_val.w) + static_cast<float>(beta_val.w));
      output[index] = tmp;
    }
  }
}

/**
 * output [m, n] row-major
 * input [m, n] row-major
 * gamma [n]
 * beta [n]
 * scale [m, n] row-major (vectorized as n/4)
 * shift [m, n] row-major (vectorized as n/4)
 * grid(m)
 * block(block_size) -- each block deals with block_size*ITEM_PER_THREAD*4 elements;
*/
template<typename T4, typename T, int ITEM_PER_THREAD>
__global__ void layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift(
                                                       T4* output,
                                                       const T4* input,
                                                       const T4* gamma,
                                                       const T4* beta,
                                                       const T4* scale,
                                                       const T4* shift,
                                                       const int m,
                                                       const int n)
{
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  T4 local_val[ITEM_PER_THREAD];
  const int n_4 = n / 4;
  int offset = m_idx * n_4;
  input  += offset;
  output += offset;
  gamma  += 0; // gamma/beta/scale/shift are broadcast per row; index by n_4 below
  beta   += 0;
  scale  += offset;
  shift  += offset;

  const T4 zero = {T(0.0f), T(0.0f), T(0.0f), T(0.0f)};
  #pragma UNROLL
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i*bdimx + tid;
    local_val[i] = index < n_4 ? input[index] : zero;
    local_sums[0] += static_cast<float>(local_val[i].x) + static_cast<float>(local_val[i].y) +
                     static_cast<float>(local_val[i].z) + static_cast<float>(local_val[i].w);
  }

  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  }
  else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_mean = local_sums[0] / n;
  }
  __syncthreads();

  local_sums[0] = 0.0f;
  #pragma UNROLL
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i*bdimx + tid;
    if (index < n_4){
      const float4 tmp = {static_cast<float>(local_val[i].x) - s_mean,
                          static_cast<float>(local_val[i].y) - s_mean,
                          static_cast<float>(local_val[i].z) - s_mean,
                          static_cast<float>(local_val[i].w) - s_mean};
      local_sums[0] += tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z + tmp.w * tmp.w;
    }
  }
  if (blockDim.x <= 32) {
    warpReduceSum<float, 1>(local_sums);
  }
  else {
    blockReduceSum<float, 1>(local_sums);
  }
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_sums[0] / n + 1e-5);
  }
  __syncthreads();

  #pragma UNROLL
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i*bdimx + tid;
    if (index < n_4){
      const T4 gamma_val = gamma[index];
      const T4 beta_val  = beta[index];
      const T4 scale_val = scale[index];
      const T4 shift_val = shift[index];
      T4 tmp;
      // y = ( (x - mean) * inv_std * gamma + beta ) * (1 + scale) + shift
      tmp.x = T(((static_cast<float>(local_val[i].x) - s_mean) * s_variance * static_cast<float>(gamma_val.x) + static_cast<float>(beta_val.x)) * (1.0f + static_cast<float>(scale_val.x)) + static_cast<float>(shift_val.x));
      tmp.y = T(((static_cast<float>(local_val[i].y) - s_mean) * s_variance * static_cast<float>(gamma_val.y) + static_cast<float>(beta_val.y)) * (1.0f + static_cast<float>(scale_val.y)) + static_cast<float>(shift_val.y));
      tmp.z = T(((static_cast<float>(local_val[i].z) - s_mean) * s_variance * static_cast<float>(gamma_val.z) + static_cast<float>(beta_val.z)) * (1.0f + static_cast<float>(scale_val.z)) + static_cast<float>(shift_val.z));
      tmp.w = T(((static_cast<float>(local_val[i].w) - s_mean) * s_variance * static_cast<float>(gamma_val.w) + static_cast<float>(beta_val.w)) * (1.0f + static_cast<float>(scale_val.w)) + static_cast<float>(shift_val.w));
      output[index] = tmp;
    }
  }
}

// Vector-of-4 type for bfloat16
struct alignas(8) bf16_4 {
  cutlass::bfloat16_t x, y, z, w;
};

template <typename T>
void layernorm(cutlass::MatrixCoord tensor_size,
               TensorRef<T, layout::RowMajor> ref_output,
               TensorRef<T, layout::RowMajor> ref_input,
               TensorRef<T, layout::RowMajor> ref_gamma,
               TensorRef<T, layout::RowMajor> ref_beta,
               cudaStream_t stream){
  const int m = tensor_size.row();
  const int n = tensor_size.column();
  T* output = ref_output.data();
  const T* input = ref_input.data();
  const T* gamma = ref_gamma.data();
  const T* beta = ref_beta.data();
  dim3 grid(m);
 // Only support vectorized x4 path
 assert((n % 4) == 0);
 dim3 block(0);
 // Choose ITEM_PER_THREAD based on n to keep block size reasonable, and round to multiples of 32
 if (n <= 4096) {
   block.x = (n/4 + 31)/32*32;
   if (block.x > 1024) block.x = 1024;
   if (std::is_same<T, float>::value) {
     layernorm_twoPassAlgo_stored_locally_e4<float4, float, 1><<<grid, block, 0, stream>>>(
       (float4*)output,
       (const float4*)input,
       (const float4*)gamma,
       (const float4*)beta,
       m,
       n);
   } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
     layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 1><<<grid, block, 0, stream>>>(
       (bf16_4*)output,
       (const bf16_4*)input,
       (const bf16_4*)gamma,
       (const bf16_4*)beta,
       m,
       n);
   } else {
     layernorm_twoPassAlgo_stored_locally_e4<half4, half, 1><<<grid, block, 0, stream>>>(
       (half4*)output,
       (const half4*)input,
       (const half4*)gamma,
       (const half4*)beta,
       m,
       n);
   }
 } else if (n <= 8192) {
   block.x = ((n + 7)/8 + 31)/32*32;
   if (block.x > 1024) block.x = 1024;
   if (std::is_same<T, float>::value) {
     layernorm_twoPassAlgo_stored_locally_e4<float4, float, 8><<<grid, block, 0, stream>>>(
       (float4*)output,
       (const float4*)input,
       (const float4*)gamma,
       (const float4*)beta,
       m,
       n);
   } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
     layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 8><<<grid, block, 0, stream>>>(
       (bf16_4*)output,
       (const bf16_4*)input,
       (const bf16_4*)gamma,
       (const bf16_4*)beta,
       m,
       n);
   } else {
     layernorm_twoPassAlgo_stored_locally_e4<half4, half, 8><<<grid, block, 0, stream>>>(
       (half4*)output,
       (const half4*)input,
       (const half4*)gamma,
       (const half4*)beta,
       m,
       n);
   }
 } else if (n <= 16384) {
   block.x = ((n + 15)/16 + 31)/32*32;
   if (block.x > 1024) block.x = 1024;
   if (std::is_same<T, float>::value) {
     layernorm_twoPassAlgo_stored_locally_e4<float4, float, 4><<<grid, block, 0, stream>>>(
       (float4*)output,
       (const float4*)input,
       (const float4*)gamma,
       (const float4*)beta,
       m,
       n);
   } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
     layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 4><<<grid, block, 0, stream>>>(
       (bf16_4*)output,
       (const bf16_4*)input,
       (const bf16_4*)gamma,
       (const bf16_4*)beta,
       m,
       n);
   } else {
     layernorm_twoPassAlgo_stored_locally_e4<half4, half, 4><<<grid, block, 0, stream>>>(
       (half4*)output,
       (const half4*)input,
       (const half4*)gamma,
       (const half4*)beta,
       m,
       n);
   }
 } else if (n <= 32768) {
   block.x = ((n + 31)/32 + 31)/32*32;
   if (block.x > 1024) block.x = 1024;
   if (std::is_same<T, float>::value) {
     layernorm_twoPassAlgo_stored_locally_e4<float4, float, 8><<<grid, block, 0, stream>>>(
       (float4*)output,
       (const float4*)input,
       (const float4*)gamma,
       (const float4*)beta,
       m,
       n);
   } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
     layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 8><<<grid, block, 0, stream>>>(
       (bf16_4*)output,
       (const bf16_4*)input,
       (const bf16_4*)gamma,
       (const bf16_4*)beta,
       m,
       n);
   } else {
     layernorm_twoPassAlgo_stored_locally_e4<half4, half, 8><<<grid, block, 0, stream>>>(
       (half4*)output,
       (const half4*)input,
       (const half4*)gamma,
       (const half4*)beta,
       m,
       n);
   }
 } else {
   block.x = ((n + 63)/64 + 31)/32*32;
   if (block.x > 1024) block.x = 1024;
   if (std::is_same<T, float>::value) {
     layernorm_twoPassAlgo_stored_locally_e4<float4, float, 16><<<grid, block, 0, stream>>>(
       (float4*)output,
       (const float4*)input,
       (const float4*)gamma,
       (const float4*)beta,
       m,
       n);
   } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
     layernorm_twoPassAlgo_stored_locally_e4<bf16_4, cutlass::bfloat16_t, 16><<<grid, block, 0, stream>>>(
       (bf16_4*)output,
       (const bf16_4*)input,
       (const bf16_4*)gamma,
       (const bf16_4*)beta,
       m,
       n);
   } else {
     layernorm_twoPassAlgo_stored_locally_e4<half4, half, 16><<<grid, block, 0, stream>>>(
       (half4*)output,
       (const half4*)input,
       (const half4*)gamma,
       (const half4*)beta,
       m,
       n);
   }
 }
}

template <typename T>
void layernorm_fused_scale_shift(cutlass::MatrixCoord tensor_size,
               TensorRef<T, layout::RowMajor> ref_output,
               TensorRef<T, layout::RowMajor> ref_input,
               TensorRef<T, layout::RowMajor> ref_gamma,
               TensorRef<T, layout::RowMajor> ref_beta,
               TensorRef<T, layout::RowMajor> ref_scale,
               TensorRef<T, layout::RowMajor> ref_shift,
               cudaStream_t stream){
  const int m = tensor_size.row();
  const int n = tensor_size.column();
  T* output = ref_output.data();
  const T* input = ref_input.data();
  const T* gamma = ref_gamma.data();
  const T* beta = ref_beta.data();
  const T* scale = ref_scale.data();
  const T* shift = ref_shift.data();
  dim3 grid(m);
  // Only support vectorized x4 path for the fused op
  assert((n % 4) == 0);
  dim3 block(0);
  // Choose ITEM_PER_THREAD based on n to keep block size reasonable, and round to multiples of 32
  if (n <= 4096) {
    block.x = (n/4 + 31)/32*32;
    if (block.x > 1024) block.x = 1024;
    if (std::is_same<T, float>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<float4, float, 1><<<grid, block, 0, stream>>>(
        (float4*)output,
        (const float4*)input,
        (const float4*)gamma,
        (const float4*)beta,
        (const float4*)scale,
        (const float4*)shift,
        m,
        n);
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<bf16_4, cutlass::bfloat16_t, 1><<<grid, block, 0, stream>>>(
        (bf16_4*)output,
        (const bf16_4*)input,
        (const bf16_4*)gamma,
        (const bf16_4*)beta,
        (const bf16_4*)scale,
        (const bf16_4*)shift,
        m,
        n);
    } else {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<half4, half, 1><<<grid, block, 0, stream>>>(
        (half4*)output,
        (const half4*)input,
        (const half4*)gamma,
        (const half4*)beta,
        (const half4*)scale,
        (const half4*)shift,
        m,
        n);
    }
  } else if (n <= 8192) {
    block.x = ((n + 7)/8 + 31)/32*32;
    if (block.x > 1024) block.x = 1024;
    if (std::is_same<T, float>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<float4, float, 8><<<grid, block, 0, stream>>>(
        (float4*)output,
        (const float4*)input,
        (const float4*)gamma,
        (const float4*)beta,
        (const float4*)scale,
        (const float4*)shift,
        m,
        n);
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<bf16_4, cutlass::bfloat16_t, 8><<<grid, block, 0, stream>>>(
        (bf16_4*)output,
        (const bf16_4*)input,
        (const bf16_4*)gamma,
        (const bf16_4*)beta,
        (const bf16_4*)scale,
        (const bf16_4*)shift,
        m,
        n);
    } else {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<half4, half, 8><<<grid, block, 0, stream>>>(
        (half4*)output,
        (const half4*)input,
        (const half4*)gamma,
        (const half4*)beta,
        (const half4*)scale,
        (const half4*)shift,
        m,
        n);
    }
  } else if (n <= 16384) {
    block.x = ((n + 15)/16 + 31)/32*32;
    if (block.x > 1024) block.x = 1024;
    if (std::is_same<T, float>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<float4, float, 4><<<grid, block, 0, stream>>>(
        (float4*)output,
        (const float4*)input,
        (const float4*)gamma,
        (const float4*)beta,
        (const float4*)scale,
        (const float4*)shift,
        m,
        n);
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<bf16_4, cutlass::bfloat16_t, 4><<<grid, block, 0, stream>>>(
        (bf16_4*)output,
        (const bf16_4*)input,
        (const bf16_4*)gamma,
        (const bf16_4*)beta,
        (const bf16_4*)scale,
        (const bf16_4*)shift,
        m,
        n);
    } else {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<half4, half, 4><<<grid, block, 0, stream>>>(
        (half4*)output,
        (const half4*)input,
        (const half4*)gamma,
        (const half4*)beta,
        (const half4*)scale,
        (const half4*)shift,
        m,
        n);
    }
  } else if (n <= 32768) {
    block.x = ((n + 31)/32 + 31)/32*32;
    if (block.x > 1024) block.x = 1024;
    if (std::is_same<T, float>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<float4, float, 8><<<grid, block, 0, stream>>>(
        (float4*)output,
        (const float4*)input,
        (const float4*)gamma,
        (const float4*)beta,
        (const float4*)scale,
        (const float4*)shift,
        m,
        n);
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<bf16_4, cutlass::bfloat16_t, 8><<<grid, block, 0, stream>>>(
        (bf16_4*)output,
        (const bf16_4*)input,
        (const bf16_4*)gamma,
        (const bf16_4*)beta,
        (const bf16_4*)scale,
        (const bf16_4*)shift,
        m,
        n);
    } else {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<half4, half, 8><<<grid, block, 0, stream>>>(
        (half4*)output,
        (const half4*)input,
        (const half4*)gamma,
        (const half4*)beta,
        (const half4*)scale,
        (const half4*)shift,
        m,
        n);
    }
  } else {
    block.x = ((n + 63)/64 + 31)/32*32;
    if (block.x > 1024) block.x = 1024;
    if (std::is_same<T, float>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<float4, float, 16><<<grid, block, 0, stream>>>(
        (float4*)output,
        (const float4*)input,
        (const float4*)gamma,
        (const float4*)beta,
        (const float4*)scale,
        (const float4*)shift,
        m,
        n);
    } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<bf16_4, cutlass::bfloat16_t, 16><<<grid, block, 0, stream>>>(
        (bf16_4*)output,
        (const bf16_4*)input,
        (const bf16_4*)gamma,
        (const bf16_4*)beta,
        (const bf16_4*)scale,
        (const bf16_4*)shift,
        m,
        n);
    } else {
      layernorm_twoPassAlgo_stored_locally_e4_fused_scale_shift<half4, half, 16><<<grid, block, 0, stream>>>(
        (half4*)output,
        (const half4*)input,
        (const half4*)gamma,
        (const half4*)beta,
        (const half4*)scale,
        (const half4*)shift,
        m,
        n);
    }
  }
}

} //namespace cutlass
 

