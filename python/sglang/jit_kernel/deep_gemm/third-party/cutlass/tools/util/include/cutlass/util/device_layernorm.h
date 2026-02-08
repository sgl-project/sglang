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

/**
 * output [m, n] row-major
 * input [m, n] row-major
 * gamma [n]
 * beta [n]
 * grid(m)
 * block(block_size) -- each block deals with n elements ; each thread deals with ITEM_PER_THREAD elements
*/
template<typename T, int ITEM_PER_THREAD>
__global__ void layernorm_twoPassAlgo_stored_locally_e1(T* output, 
                                                        const T* input, 
                                                        const T* gamma, 
                                                        const T* beta, 
                                                        const int m, 
                                                        const int n)
{
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  T local_val[ITEM_PER_THREAD];
  float local_sums[1] = {0.0f};
  int offset = m_idx * n;
  input += offset;
  output += offset;

  const T zero = T(0.0f);
  #pragma unroll
  for (int i = 0 ; i < ITEM_PER_THREAD ; i++){ 
    int index = tid + i*bdimx;
    local_val[i] = index < n ? input[index] : zero;   
    local_sums[0] += static_cast<float>(local_val[i]); 
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
  #pragma unroll
  for (int i = 0 ; i < ITEM_PER_THREAD ; i++){
    int index = tid + i*bdimx;
    if (index < n){
      const float tmp = static_cast<float>(local_val[i]) - s_mean;
      local_sums[0] += tmp * tmp;
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

  #pragma unroll
  for (int i = 0 ; i < ITEM_PER_THREAD ; i++){
    int index = tid + i*bdimx;
    if (index < n) {
      const T gamma_val = gamma[index];
      const T beta_val = beta[index];
      output[index] = T((static_cast<float>(local_val[i]) - s_mean) * s_variance * static_cast<float>(gamma_val) + static_cast<float>(beta_val));
    }
  }
}

/**
 * output [m, n] row-major
 * input [m, n] row-major
 * gamma [n]
 * beta [n]
 * grid(m)
 * block(block_size) -- each block deals with block_size*ITEM_PER_THREAD*2 elements;
*/
template<typename T2, typename T, int ITEM_PER_THREAD>
__global__ void layernorm_twoPassAlgo_stored_locally_e2(T2* output,
                                                        const T2* input,
                                                        const T2* gamma,
                                                        const T2* beta,
                                                        const int m,
                                                        const int n)
{
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  T2 local_val[ITEM_PER_THREAD];
  const int n_2 = n / 2;
  int offset = m_idx * n_2;
  input += offset;
  output += offset;

  const T2 zero = {T(0.0f), T(0.0f)};
  #pragma UNROLL
  for (int i = 0; i < ITEM_PER_THREAD; i += 1) {
    const int index = i*bdimx + tid;
    local_val[i] = index < n_2 ? input[index] : zero;
    local_sums[0] += static_cast<float>(local_val[i].x) + static_cast<float>(local_val[i].y);
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
    if (index < n_2){
      const float2 tmp = {static_cast<float>(local_val[i].x) - s_mean,
                          static_cast<float>(local_val[i].y) - s_mean};
      local_sums[0] += tmp.x * tmp.x + tmp.y * tmp.y;
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
    if (index < n_2){
      const T2 gamma_val = gamma[index];
      const T2 beta_val = beta[index];
      T2 tmp;
      tmp.x = T((static_cast<float>(local_val[i].x) - s_mean)*s_variance*static_cast<float>(gamma_val.x) + static_cast<float>(beta_val.x));
      tmp.y = T((static_cast<float>(local_val[i].y) - s_mean)*s_variance*static_cast<float>(gamma_val.y) + static_cast<float>(beta_val.y));
      output[index] = tmp;
    }
  }
}

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
 * grid(m)
 * block(block_size) -- each block deals with n elements ; each thread deals with ITEM_PER_THREAD elements
*/
template<typename T>
__global__ void layernorm_twoPassAlgo_e1(T* output,
                                         const T* input,
                                         const T* gamma,
                                         const T* beta,
                                         const int m,
                                         const int n)
{
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  int offset = m_idx * n;
  input += offset;
  output += offset;

  for (int index = tid ; index < n ; index += bdimx){
    float local_val = static_cast<float>(input[index]);
    local_sums[0] += local_val;
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
  for (int index = tid ; index < n ; index += bdimx){
    float local_val = static_cast<float>(input[index]);
    local_val = local_val - s_mean;
    local_sums[0] += local_val * local_val;
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

  for (int index = tid ; index < n ; index += bdimx){
    const T gamma_val = gamma[index];
    const T beta_val = beta[index];
    const T local_val = input[index];
    output[index] = T((static_cast<float>(local_val) - s_mean) * s_variance * static_cast<float>(gamma_val) + static_cast<float>(beta_val));
  }
}

/**
 * output [m, n] row-major
 * input [m, n] row-major
 * gamma [n]
 * beta [n]
 * grid(m)
 * block(block_size) -- each block deals with block_size*ITEM_PER_THREAD*2 elements;
*/
template<typename T2, typename T>
__global__ void layernorm_twoPassAlgo_e2(T2* output,
                                         const T2* input,
                                         const T2* gamma,
                                         const T2* beta,
                                         const int m,
                                         const int n)
{
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean, s_variance;
  float local_sums[1] = {0.0f};
  const int n_2 = n / 2;
  int offset = m_idx * n_2;
  input += offset;
  output += offset;

  for (int index = tid; index < n_2; index += bdimx) {
    const T2 local_val = input[index];
    local_sums[0] += static_cast<float>(local_val.x) + static_cast<float>(local_val.y);
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
  for (int index = tid; index < n_2; index += bdimx) {
    const T2 local_val = input[index];
    const float2 tmp = {static_cast<float>(local_val.x) - s_mean,
                        static_cast<float>(local_val.y) - s_mean};
    local_sums[0] += tmp.x * tmp.x + tmp.y * tmp.y;
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

  for (int index = tid; index < n_2; index += bdimx) {
    const T2 local_val = input[index];
    const T2 gamma_val = gamma[index];
    const T2 beta_val = beta[index];
    T2 tmp;
    tmp.x = T((static_cast<float>(local_val.x) - s_mean)*s_variance*static_cast<float>(gamma_val.x) + static_cast<float>(beta_val.x));
    tmp.y = T((static_cast<float>(local_val.y) - s_mean)*s_variance*static_cast<float>(gamma_val.y) + static_cast<float>(beta_val.y));
    output[index] = tmp;
  }
}

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
  dim3 block((n + 31)/32*32);
  if (block.x > 1024){
    block.x = 1024;
  }
  // TODO : There should be better configs for different cases, we only use several samples to show how to use here
  // TODO : using registers to store values locally can reduce the loads from global memory and speedup the kernels.
  if ((n % 4 == 0) && (n >= 128) && (n <= 4096)) {
    block.x = (n/4 + 31)/32*32;
    if (std::is_same<T, float>::value) {
      layernorm_twoPassAlgo_stored_locally_e4<float4, float, 1><<<grid, block, 0, stream>>>(
        (float4*)output,
        (const float4*)input,
        (const float4*)gamma,
        (const float4*)beta,
        m,
        n);
    } // if (std::is_same<T, float>::value)
    else {
      layernorm_twoPassAlgo_stored_locally_e4<half4, half, 1><<<grid, block, 0, stream>>>(
        (half4*)output,
        (const half4*)input,
        (const half4*)gamma,
        (const half4*)beta,
        m,
        n);
    }
  } //if ((n % 4 == 0) && (n >= 128) && (n <= 4096))
  else if (n % 2 == 0) {
    if (n / 2 <= 1024) {
      block.x = (n/2 + 31)/32*32;
      if (std::is_same<T, float>::value) {
        layernorm_twoPassAlgo_stored_locally_e2<float2, float, 1><<<grid, block, 0, stream>>>(
          (float2*)output,
          (const float2*)input,
          (const float2*)gamma,
          (const float2*)beta,
          m,
          n);
      } //if (std::is_same<T, float>::value)
      else {
        layernorm_twoPassAlgo_stored_locally_e2<half2, half, 1><<<grid, block, 0, stream>>>(
          (half2*)output,
          (const half2*)input,
          (const half2*)gamma,
          (const half2*)beta,
          m,
          n);
      }
    } // if (n / 2 <= 1024)
    else if (n <= 8192) {
      block.x = ((n + 7)/8 + 31)/32*32;
      if (std::is_same<T, float>::value) {
        layernorm_twoPassAlgo_stored_locally_e2<float2, float, 4><<<grid, block, 0, stream>>>(
          (float2*)output,
          (const float2*)input,
          (const float2*)gamma,
          (const float2*)beta,
          m,
          n);
      } // if (std::is_same<T, float>::value)
      else {
        layernorm_twoPassAlgo_stored_locally_e2<half2, half, 4><<<grid, block, 0, stream>>>(
          (half2*)output,
          (const half2*)input,
          (const half2*)gamma,
          (const half2*)beta,
          m,
          n);
      }
    } // if (n <= 8192)
    else if (n <= 16384) {
      block.x = ((n + 15)/ 16 + 31)/32*32;
      if (std::is_same<T, float>::value) {
        layernorm_twoPassAlgo_stored_locally_e2<float2, float, 8><<<grid, block, 0, stream>>>(
          (float2*)output,
          (const float2*)input,
          (const float2*)gamma,
          (const float2*)beta,
          m,
          n);
      } // if (std::is_same<T, float>::value)
      else {
        layernorm_twoPassAlgo_stored_locally_e2<half2, half, 8><<<grid, block, 0, stream>>>(
          (half2*)output,
          (const half2*)input,
          (const half2*)gamma,
          (const half2*)beta,
          m,
          n);
      }
    } // if (n <= 16384)
    else if (n <= 32768) {
      block.x = ((n + 31)/32 + 31)/32*32;
      if (std::is_same<T, float>::value) {
        layernorm_twoPassAlgo_stored_locally_e2<float2, float, 16><<<grid, block, 0, stream>>>(
          (float2*)output,
          (const float2*)input,
          (const float2*)gamma,
          (const float2*)beta,
          m,
          n);
      } // if (std::is_same<T, float>::value)
      else {
        layernorm_twoPassAlgo_stored_locally_e2<half2, half, 16><<<grid, block, 0, stream>>>(
          (half2*)output,
          (const half2*)input,
          (const half2*)gamma,
          (const half2*)beta,
          m,
          n);
      }
    } // if (n <= 32768)
    else {
      if (block.x > 512)
        block.x = 512;
      if (std::is_same<T, float>::value) {
        layernorm_twoPassAlgo_e2<float2, float><<<grid, block, 0, stream>>>(
          (float2 *)output, 
          (const float2 *)input,
          (const float2 *)gamma, 
          (const float2 *)beta, 
          m, 
          n);
      } // if (std::is_same<T, float>::value)
      else {
        layernorm_twoPassAlgo_e2<half2, half><<<grid, block, 0, stream>>>(
          (half2 *)output,
          (const half2 *)input,
          (const half2 *)gamma,
          (const half2 *)beta,
          m,
          n);
      }
    }
  } // if (n % 2 == 0)
  else {
    if (n <= 1024) {
      layernorm_twoPassAlgo_stored_locally_e1<T, 1><<<grid, block, 0, stream>>>(
        output, 
        input, 
        gamma, 
        beta, 
        m, 
        n);
    } // if (n <= 1024)
    else if (n <= 8192) {
      block.x = ((n + 7)/8 + 31)/32*32;
      layernorm_twoPassAlgo_stored_locally_e1<T, 8><<<grid, block, 0, stream>>>(
        output,
        input,
        gamma,
        beta,
        m,
        n);
    } // if (n <= 8192)
    else if (n <= 16384) {
      block.x = ((n + 15)/16 + 32)/32*32;
      layernorm_twoPassAlgo_stored_locally_e1<T, 16><<<grid, block, 0, stream>>>(
        output,
        input,
        gamma,
        beta,
        m,
        n);
    } // if (n <= 16384)
    else if (n <= 32768) {
      block.x = ((n + 31)/32 + 31)/32*32;
      layernorm_twoPassAlgo_stored_locally_e1<T, 32><<<grid, block, 0, stream>>>(
        output,
        input,
        gamma,
        beta,
        m,
        n);
    } // if (n <= 32768)
    else{
      if (block.x > 512) {
        block.x = 512;
      }
      layernorm_twoPassAlgo_e1<<<grid, block, 0, stream>>>(
        output, 
        input, 
        gamma, 
        beta, 
        m, 
        n);
    }
  } 
}

} //namespace cutlass
