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
 * \brief cuda kernels to do group norm on a device memory tensor with NHWC layout. The tensor will be divided into [N, H, W, G, C'] and then we do normalization on [H, W, C'].
 */

#include "cutlass/cutlass.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/tensor_ref.h"
#include "device_utils.h"
#include <cfloat>

namespace cutlass {

/** \brief interface to do group norm on a device memory tensor with NHWC layout.
 * \tparam T: data type
 */
template <typename T>
void groupnorm(cutlass::Tensor4DCoord input_size,
               const int num_groups,
               const float eps,
               TensorRef<T, layout::TensorNHWC> ref_output,
               TensorRef<T, layout::TensorNHWC> ref_input,
               TensorRef<T, layout::TensorNHWC> ref_gamma,
               TensorRef<T, layout::TensorNHWC> ref_beta,
               cudaStream_t stream);

extern __shared__ char groupnorm_shm[];

// For small prod_dim1_to_last_dim/num_groups, to avoid multiple loads from global memory,
// we store the input in the shared memory.
// grid(num_groups, dim0)
// block(BLOCKSIZE)
// BLOCKSIZE * TVecs_PER_THREAD <= prod_dim1_to_last_dim/num_group
template<typename TVec, typename T, int T_PER_TVec>
__global__ void groupnorm_twopass_store_locally(T*          output,
                                                const T*    input,
                                                const T*    gamma,
                                                const T*    beta,
                                                int         num_groups,
                                                int         prod_dim1_to_last_dim,
                                                int         last_dim,
                                                const float eps,
                                                const int   TVecs_PER_THREAD)
{
    const int   bid               = blockIdx.y;   // index of batch
    const int   gid               = blockIdx.x;   // index of group
    const int   tid               = threadIdx.x;  // index of thread
    const int   bdimx             = blockDim.x;
    const int   s_reduce_elements = prod_dim1_to_last_dim / num_groups;
    const int   v_reduce_elements = s_reduce_elements / T_PER_TVec;
    const int   s_group_stride    = last_dim / num_groups;
    const int   v_group_stride    = s_group_stride / T_PER_TVec;
    const int   offset_of_group   = (bid * prod_dim1_to_last_dim + gid * s_group_stride) / T_PER_TVec;
    const TVec* input_TVec_ptr    = (const TVec*)(input) + offset_of_group;
    TVec*       output_TVec_ptr   = (TVec*)(output) + offset_of_group;
    T*       local_val         = ((T*)groupnorm_shm) + TVecs_PER_THREAD * T_PER_TVec * tid;
    float       local_sum[1]      = {0.0f};

// load from global memory into shared memory
#pragma unroll
    for (int i = 0; i < TVecs_PER_THREAD; i += 1) {
        const int current_load_start_idx = (i * bdimx + tid) * T_PER_TVec;
        const int offset_in_group =
            ((current_load_start_idx / s_group_stride) * last_dim + (current_load_start_idx % s_group_stride))
            / T_PER_TVec;
        if (current_load_start_idx < s_reduce_elements) {
            TVec      tmp_vec          = input_TVec_ptr[offset_in_group];
            T*        tmp_vec_ptr      = (T*)(&tmp_vec);
            const int local_val_offset = i * T_PER_TVec;
#pragma unroll
            for (int j = 0; j < T_PER_TVec; j++) {
                float tmp = static_cast<float>(tmp_vec_ptr[j]);
                local_sum[0] += tmp;
                local_val[local_val_offset + j] = tmp_vec_ptr[j];
            }
        }
    }
    __shared__ float s_mean, s_variance;

    // reduction for mean
    if (bdimx <= 32) {
        warpReduceSum<float, 1>(local_sum);
    }
    else {
        blockReduceSum<float, 1>(local_sum);
    }
    if (tid == 0) {
        s_mean = local_sum[0] / s_reduce_elements;
    }
    __syncthreads();

    // reduction for std
    local_sum[0] = 0.0f;
#pragma unroll
    for (int i = 0; i < TVecs_PER_THREAD; i += 1) {
        const int current_load_start_idx = (i * bdimx + tid) * T_PER_TVec;
        if (current_load_start_idx < s_reduce_elements) {
            const int local_val_offset = i * T_PER_TVec;
#pragma unroll
            for (int j = 0; j < T_PER_TVec; j++) {
                float tmp = static_cast<float>(local_val[local_val_offset + j]);
                tmp -= s_mean;
                local_sum[0] += tmp * tmp;
            }
        }
    }
    if (bdimx <= 32) {
        warpReduceSum<float, 1>(local_sum);
    }
    else {
        blockReduceSum<float, 1>(local_sum);
    }
    if (tid == 0) {
        s_variance = rsqrtf(local_sum[0] / s_reduce_elements + eps);
    }
    __syncthreads();

    // normalize
    const int   gamma_offset_of_group = gid * v_group_stride;
    const TVec* gamma_TVec_ptr        = (const TVec*)gamma + gamma_offset_of_group;
    const TVec* beta_TVec_ptr         = (const TVec*)beta + gamma_offset_of_group;
#pragma unroll
    for (int i = 0; i < TVecs_PER_THREAD; i += 1) {
        const int current_load_start_idx = (i * bdimx + tid) * T_PER_TVec;
        const int offset_in_group =
            ((current_load_start_idx / s_group_stride) * last_dim + (current_load_start_idx % s_group_stride))
            / T_PER_TVec;
        const int gamma_offset_in_group = (current_load_start_idx % s_group_stride) / T_PER_TVec;
        const int local_val_offset      = i * T_PER_TVec;
        if (current_load_start_idx < s_reduce_elements) {
            TVec gamma_val     = gamma_TVec_ptr[gamma_offset_in_group];
            TVec beta_val      = beta_TVec_ptr[gamma_offset_in_group];
            T*   gamma_val_ptr = (T*)(&gamma_val);
            T*   beta_val_ptr  = (T*)(&beta_val);
            TVec tmp_vec;
            T*   tmp_vec_ptr = (T*)(&tmp_vec);
#pragma unroll
            for (int j = 0; j < T_PER_TVec; j++) {
                float tmp = (static_cast<float>(local_val[local_val_offset + j]) - s_mean) * s_variance
                                * static_cast<float>(gamma_val_ptr[j])
                            + static_cast<float>(beta_val_ptr[j]);
                if (sizeof(T) == sizeof(half)) {
                    tmp_vec_ptr[j] = T(__float2half_rn(tmp));
                }
                else {
                    tmp_vec_ptr[j] = T(tmp);
                }
            }
            output_TVec_ptr[offset_in_group] = tmp_vec;
        }
    }
}

// For large prod_dim1_to_last_dim/num_groups,
// in which the data cannot be stored locally,
// we will load from global memory multiple times,
// grid(num_groups, dim0)
// block(BLOCKSIZE)
// BLOCKSIZE * TVecs_PER_THREAD <= prod_dim1_to_last_dim/num_group
template<typename TVec, typename T, int T_PER_TVec>
__global__ void groupnorm_twopass_multiple_load(T*          output,
                                                const T*    input,
                                                const T*    gamma,
                                                const T*    beta,
                                                int         num_groups,
                                                int         prod_dim1_to_last_dim,
                                                int         last_dim,
                                                const float eps,
                                                const int   TVecs_PER_THREAD)
{
    const int   bid               = blockIdx.y;   // index of batch
    const int   gid               = blockIdx.x;   // index of group
    const int   tid               = threadIdx.x;  // index of thread
    const int   bdimx             = blockDim.x;
    const int   s_reduce_elements = prod_dim1_to_last_dim / num_groups;
    const int   v_reduce_elements = s_reduce_elements / T_PER_TVec;
    const int   s_group_stride    = last_dim / num_groups;
    const int   v_group_stride    = s_group_stride / T_PER_TVec;
    const int   offset_of_group   = (bid * prod_dim1_to_last_dim + gid * s_group_stride) / T_PER_TVec;
    const TVec* input_TVec_ptr    = (const TVec*)(input) + offset_of_group;
    TVec*       output_TVec_ptr   = (TVec*)(output) + offset_of_group;
    float       local_sum[1]      = {0.0f};

#pragma unroll
    for (int i = 0; i < TVecs_PER_THREAD; i += 1) {
        const int current_load_start_idx = (i * bdimx + tid) * T_PER_TVec;
        if (current_load_start_idx < s_reduce_elements) {
            const int offset_in_group =
                ((current_load_start_idx / s_group_stride) * last_dim + (current_load_start_idx % s_group_stride))
                / T_PER_TVec;
            TVec tmp_vec     = input_TVec_ptr[offset_in_group];
            T*   tmp_vec_ptr = (T*)(&tmp_vec);
#pragma unroll
            for (int j = 0; j < T_PER_TVec; j++) {
                float tmp = static_cast<float>(tmp_vec_ptr[j]);
                local_sum[0] += tmp;
            }
        }
    }
    __shared__ float s_mean, s_variance;

    // reduction for mean
    if (bdimx <= 32) {
        warpReduceSum<float, 1>(local_sum);
    }
    else {
        blockReduceSum<float, 1>(local_sum);
    }
    if (tid == 0) {
        s_mean = local_sum[0] / s_reduce_elements;
    }
    __syncthreads();

    // reduction for std
    local_sum[0] = 0.0f;
#pragma unroll
    for (int i = 0; i < TVecs_PER_THREAD; i += 1) {
        const int current_load_start_idx = (i * bdimx + tid) * T_PER_TVec;
        if (current_load_start_idx < s_reduce_elements) {
            const int offset_in_group =
                ((current_load_start_idx / s_group_stride) * last_dim + (current_load_start_idx % s_group_stride))
                / T_PER_TVec;
            TVec tmp_vec     = input_TVec_ptr[offset_in_group];
            T*   tmp_vec_ptr = (T*)(&tmp_vec);
#pragma unroll
            for (int j = 0; j < T_PER_TVec; j++) {
                float tmp = static_cast<float>(tmp_vec_ptr[j]);
                tmp -= s_mean;
                local_sum[0] += tmp * tmp;
            }
        }
    }
    if (bdimx <= 32) {
        warpReduceSum<float, 1>(local_sum);
    }
    else {
        blockReduceSum<float, 1>(local_sum);
    }
    if (tid == 0) {
        s_variance = rsqrtf(local_sum[0] / s_reduce_elements + eps);
    }
    __syncthreads();

    // normalize
    const int   gamma_offset_of_group = gid * v_group_stride;
    const TVec* gamma_TVec_ptr        = (const TVec*)gamma + gamma_offset_of_group;
    const TVec* beta_TVec_ptr         = (const TVec*)beta + gamma_offset_of_group;
#pragma unroll
    for (int i = 0; i < TVecs_PER_THREAD; i += 1) {
        const int current_load_start_idx = (i * bdimx + tid) * T_PER_TVec;
        if (current_load_start_idx < s_reduce_elements) {
            const int offset_in_group =
                ((current_load_start_idx / s_group_stride) * last_dim + (current_load_start_idx % s_group_stride))
                / T_PER_TVec;
            const int gamma_offset_in_group = (current_load_start_idx % s_group_stride) / T_PER_TVec;
            TVec      gamma_val             = gamma_TVec_ptr[gamma_offset_in_group];
            TVec      beta_val              = beta_TVec_ptr[gamma_offset_in_group];
            T*        gamma_val_ptr         = (T*)(&gamma_val);
            T*        beta_val_ptr          = (T*)(&beta_val);
            TVec      tmp_vec               = input_TVec_ptr[offset_in_group];
            T*        tmp_vec_ptr           = (T*)(&tmp_vec);
            TVec      output_tmp_vec;
            T*        output_tmp_vec_ptr = (T*)(&output_tmp_vec);
#pragma unroll
            for (int j = 0; j < T_PER_TVec; j++) {
                float tmp =
                    (static_cast<float>(tmp_vec_ptr[j]) - s_mean) * s_variance * static_cast<float>(gamma_val_ptr[j])
                    + static_cast<float>(beta_val_ptr[j]);
                if (sizeof(T) == sizeof(half)) {
                    output_tmp_vec_ptr[j] = T(__float2half_rn(tmp));
                }
                else {
                    output_tmp_vec_ptr[j] = T(tmp);
                }
            }
            output_TVec_ptr[offset_in_group] = output_tmp_vec;
        }
    }
}

//ref_input & ref_output should be [N, H, W, C]
//ref_gamma & ref_beta should be [1, 1, 1, C]
template <typename T>
void groupnorm(cutlass::Tensor4DCoord input_size,
               const int num_groups,
               const float eps,
               TensorRef<T, layout::TensorNHWC> ref_output,
               TensorRef<T, layout::TensorNHWC> ref_input,
               TensorRef<T, layout::TensorNHWC> ref_gamma,
               TensorRef<T, layout::TensorNHWC> ref_beta,
               cudaStream_t stream){
  const int N = input_size.n();
  const int H = input_size.h();
  const int W = input_size.w();
  const int C = input_size.c();
  if (C % num_groups != 0){
    printf("[ERROR] C should be a multiple of num_groups.\n");
  }
  T* output = ref_output.data();
  const T* input = ref_input.data();
  const T* gamma = ref_gamma.data();
  const T* beta = ref_beta.data();

  const int dim0 = N;
  const int last_dim = C;
  const int prod_dim1_to_last_dim = H*W*C;
  const int s_reduce_elements = prod_dim1_to_last_dim / num_groups;
  const int s_group_stride = last_dim / num_groups;
  dim3      grid(num_groups, dim0);
  int       threadblock_size = 32;
  if (s_group_stride % 2 == 0) {
    const int T_PER_TVec = 2;
    while (threadblock_size < 1024) {
      if (s_reduce_elements / T_PER_TVec / threadblock_size <= 8)
        break;
        threadblock_size *= 2;
      }
    dim3      block(threadblock_size);
    const int TVec_PER_THREAD = (s_reduce_elements / T_PER_TVec + threadblock_size - 1) / threadblock_size;
    const int shm_size = T_PER_TVec * TVec_PER_THREAD * threadblock_size * sizeof(T);
    // for small s_reduce_elements, specific case for H=W=22, C=1280, num_groups=32;
    // the size of grid & block may have better choice for different cases.
    // ensure shared memory is smaller than 48KB
    if (std::is_same<T, float>::value){
      if (shm_size < 48 * 1024) {
        groupnorm_twopass_store_locally<float2, T, T_PER_TVec><<<grid, block, shm_size, stream>>>(
          output, input, gamma, beta, num_groups, prod_dim1_to_last_dim, last_dim, eps, TVec_PER_THREAD);
      }
      else {
        groupnorm_twopass_multiple_load<float2, T, T_PER_TVec><<<grid, block, 0, stream>>>(
          output, input, gamma, beta, num_groups, prod_dim1_to_last_dim, last_dim, eps, TVec_PER_THREAD);
      }
    }
    else{
      if (shm_size < 48 * 1024) {
        groupnorm_twopass_store_locally<half2, T, T_PER_TVec><<<grid, block, shm_size, stream>>>(
          output, input, gamma, beta, num_groups, prod_dim1_to_last_dim, last_dim, eps, TVec_PER_THREAD);
      }
      else {
        groupnorm_twopass_multiple_load<half2, T, T_PER_TVec><<<grid, block, 0, stream>>>(
          output, input, gamma, beta, num_groups, prod_dim1_to_last_dim, last_dim, eps, TVec_PER_THREAD);
      }
    }
  }
  else {
    const int T_PER_TVec = 1;
    while (threadblock_size < 1024) {
      if (s_reduce_elements / T_PER_TVec / threadblock_size <= 8)
        break;
        threadblock_size *= 2;
      }
    dim3      block(threadblock_size);
    const int TVec_PER_THREAD = (s_reduce_elements / T_PER_TVec + threadblock_size - 1) / threadblock_size;
    const int shm_size = T_PER_TVec * TVec_PER_THREAD * threadblock_size * sizeof(T);
    if (shm_size < 48 * 1024) {
      groupnorm_twopass_store_locally<T, T, T_PER_TVec><<<grid, block, shm_size, stream>>>(
        output, input, gamma, beta, num_groups, prod_dim1_to_last_dim, last_dim, eps, TVec_PER_THREAD);
    }
    else {
      groupnorm_twopass_multiple_load<T, T, T_PER_TVec><<<grid, block, 0, stream>>>(
        output, input, gamma, beta, num_groups, prod_dim1_to_last_dim, last_dim, eps, TVec_PER_THREAD);
    }
  }

}

} //namespace cutlass
