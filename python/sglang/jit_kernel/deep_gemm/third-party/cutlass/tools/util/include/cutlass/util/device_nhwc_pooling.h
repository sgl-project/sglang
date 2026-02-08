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
 * \brief cuda kernels to do avg/max pooling on a device memory tensor with NHWC layout.
 */

#include "cutlass/cutlass.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/tensor_ref.h"
#include "device_utils.h"
#include <cfloat>

namespace cutlass {

/** \brief interface to do avg/max pooling on a device memory tensor with NHWC layout.
 * \tparam T: data type
 */
template <typename T>
void pooling_nhwc(cutlass::Tensor4DCoord input_tensor_size,
                  cutlass::Tensor4DCoord filter_tensor_size,
                  cutlass::Tensor4DCoord output_tensor_size,
                  cutlass::MatrixCoord padding,
                  cutlass::MatrixCoord stride,
                  TensorRef<T, layout::TensorNHWC> ref_input,
                  TensorRef<T, layout::TensorNHWC> ref_output,
                  int poolingType, //0 for avg pooling ; 1 for max pooling
                  cudaStream_t stream);

/** get the output size of pooling
 */
inline int getOutputSize(int H_W, int padding, int kernel_size, int stride)
{
    return (H_W + 2 * padding - kernel_size) / stride + 1;
}

/**
 * input is [N, H, W, C]
 * assume stride == kernel_size
 * output_h = (H + 2*padding_H - kernel_H)/stride_H
 * output_w = (W + 2*padding_W - kernel_W)/stride_W
 * output is [N, output_h, output_w, C]
 * grid(N, output_h, output_w)
 * block(min(C, 256)) :
 * each block deals with C elements of output when each thread deals with ((C + 255)/256 element of output)
*/
template<typename T, bool IS_AVG_POOLING>
__global__ void pooling_nhwc_element1_kernel(T* output,
                                             const T* input,
                                             const int N,
                                             const int H,
                                             const int W,
                                             const int C,
                                             const int output_H,
                                             const int output_W,
                                             const int kernel_H,
                                             const int kernel_W,
                                             const int stride_H,
                                             const int stride_W,
                                             const int padding_H,
                                             const int padding_W)
{
  const int tid = threadIdx.x;
  const int n_idx = blockIdx.x;
  const int output_h_idx = blockIdx.y;
  const int output_w_idx = blockIdx.z;

  int h_start_idx = output_h_idx * stride_H - padding_H;
  int h_end_idx = h_start_idx + kernel_H;
  h_start_idx = (h_start_idx < 0) ? 0 : h_start_idx;
  h_end_idx = h_end_idx > H ? H : h_end_idx;

  int w_start_idx = output_w_idx * stride_W - padding_W;
  int w_end_idx = w_start_idx + kernel_W;
  w_start_idx = (w_start_idx < 0) ? 0 : w_start_idx;
  w_end_idx = w_end_idx > W ? W : w_end_idx;

  input += n_idx * H * W * C;
  output += ((n_idx * output_H + output_h_idx) * output_W + output_w_idx) * C;
  const int kernel_size2 = kernel_H * kernel_W;
  for (int c_idx = tid; c_idx < C; c_idx += blockDim.x) {
    float pooling;
    if (IS_AVG_POOLING){
      pooling = 0.0f;
    }
    else{
      pooling = -FLT_MAX;
    }
    for (int h = h_start_idx; h < h_end_idx; h++) {
      for (int w = w_start_idx; w < w_end_idx; w++) {
        const int idx = (h * W + w) * C;
        const float tmp = static_cast<float>(input[idx + c_idx]);
        if (IS_AVG_POOLING){
          pooling = pooling + tmp;
        }
        else{
          pooling = pooling > tmp ? pooling : tmp;
        }
      }
    }

    T output_val;
    if (IS_AVG_POOLING){
      output_val = T(pooling/kernel_size2);
    }
    else{
      output_val = T(pooling);
    }
    output[c_idx] = output_val;
  }
}

template<typename T2, typename T, bool IS_AVG_POOLING>
__global__ void pooling_nhwc_element2_kernel(T2* output,
                                             const T2* input,
                                             const int N,
                                             const int H,
                                             const int W,
                                             const int C,
                                             const int output_H,
                                             const int output_W,
                                             const int kernel_H,
                                             const int kernel_W,
                                             const int stride_H,
                                             const int stride_W,
                                             const int padding_H,
                                             const int padding_W)
{
  const int tid = threadIdx.x;
  const int n_idx = blockIdx.x;
  const int output_h_idx = blockIdx.y;
  const int output_w_idx = blockIdx.z;

  int h_start_idx = output_h_idx * stride_H - padding_H;
  int h_end_idx = h_start_idx + kernel_H;
  h_start_idx = (h_start_idx < 0) ? 0 : h_start_idx;
  h_end_idx = h_end_idx > H ? H : h_end_idx;

  int w_start_idx = output_w_idx * stride_W - padding_W;
  int w_end_idx = w_start_idx + kernel_W;
  w_start_idx = (w_start_idx < 0) ? 0 : w_start_idx;
  w_end_idx = w_end_idx > W ? W : w_end_idx;

  input += n_idx * H * W * C;
  output += ((n_idx * output_H + output_h_idx) * output_W + output_w_idx) * C;
  const int kernel_size2 = kernel_H * kernel_W;
  for (int c_idx = tid; c_idx < C; c_idx += blockDim.x) {
    float2 pooling;
    if (IS_AVG_POOLING) { 
      pooling = {0.0f, 0.0f};
    }
    else {
      pooling = {-FLT_MAX, -FLT_MAX};
    }
    for (int h = h_start_idx; h < h_end_idx; h++) {
      for (int w = w_start_idx; w < w_end_idx; w++) {
        const int idx = (h * W + w) * C;
        const T2 tmp = input[idx + c_idx];
        const float2 tmp_flt2 = {static_cast<float>(tmp.x), static_cast<float>(tmp.y)};
        if (IS_AVG_POOLING) {
          pooling.x += tmp_flt2.x;
          pooling.y += tmp_flt2.y;
        }
        else {
          pooling.x = pooling.x > tmp_flt2.x ? pooling.x : tmp_flt2.x;
          pooling.y = pooling.y > tmp_flt2.y ? pooling.y : tmp_flt2.y;
        }
      }
    }

    T2 output_val;
    if (IS_AVG_POOLING) {
      output_val.x = T(pooling.x/kernel_size2);
      output_val.y = T(pooling.y/kernel_size2);
    }
    else {
      output_val.x = T(pooling.x);
      output_val.y = T(pooling.y);
    }
    output[c_idx] = output_val;
  }
}

/**
 * output [N, 1, 1, C]
 * input [N, H, W, C]
 * grid(C, N)
 * block(block_size) -- each block deals with H*W/block_size elements;
*/
template<typename T, bool IS_AVG_POOLING>
__global__ void pooling_nxhTo1x1_element1_kernel(
    T* output, const T* input, const int N, const int HW, const int C)
{
    const int c_idx = blockIdx.x;
    const int n_idx = blockIdx.y;
    float pooling[1];
    if (IS_AVG_POOLING) {
      pooling[0] = 0.0f;
    }
    else {
      pooling[0] = -FLT_MAX;
    }
    const size_t input_offset = n_idx * HW * C + c_idx;
    input += input_offset;
    const size_t output_offset = n_idx * C + c_idx;
    output += output_offset;
    int tid = threadIdx.x;

    for (int index = tid; index < HW; index += blockDim.x) {
        float val = static_cast<float>(input[index * C]);
        if (IS_AVG_POOLING) {
          pooling[0] += val;
        }
        else {
          pooling[0] = pooling[0] > val ? pooling[0] : val;
        }
    }
    if (blockDim.x <= 32) {
        if (IS_AVG_POOLING) {
          warpReduceSum<float, 1>(pooling);
        }
        else {
          warpReduceMax<float, 1>(pooling);
        }
    }
    else {
        if (IS_AVG_POOLING) {
          blockReduceSum<float, 1>(pooling);
        }
        else {
          blockReduceMax<float, 1>(pooling);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        T output_val;
        if (IS_AVG_POOLING) {
          output_val = T(pooling[0] / HW);
        }
        else {
          output_val = T(pooling[0]);
        }
        output[0] = output_val;
    }
}


/**
 * output [N, 1, 1, C]
 * input [N, H, W, C]
 * grid(C/2, N)
 * block(block_size) -- each thread deals with H*W/block_size * 2 elements;
*/
template<typename T2, typename T, bool IS_AVG_POOLING>
__global__ void pooling_nxhTo1x1_element2_kernel(
    T2* output, const T2* input, const int N, const int HW, const int C)
{
    const int c_idx = blockIdx.x;
    const int n_idx = blockIdx.y;
    float pooling[2];
    if (IS_AVG_POOLING) {
      pooling[0] = pooling[1] = 0.0f;
    }
    else {
      pooling[0] = pooling[1] = -FLT_MAX;
    }
    const int C_2 = C / 2;
    const size_t input_offset = n_idx * HW * C_2 + c_idx;
    input += input_offset;
    const size_t output_offset = n_idx * C_2 + c_idx;
    output += output_offset;
    int tid = threadIdx.x;

    for (int index = tid; index < HW; index += blockDim.x) {
        T2 val = input[index * C_2];
        float2 val_flt2 = {static_cast<float>(val.x), static_cast<float>(val.y)};
        if (IS_AVG_POOLING) {
          pooling[0] += val_flt2.x;
          pooling[1] += val_flt2.y;
        }
        else {
          pooling[0] = pooling[0] > val_flt2.x ? pooling[0] : val_flt2.x;
          pooling[1] = pooling[1] > val_flt2.y ? pooling[1] : val_flt2.y;
        }
    }
    if (blockDim.x <= 32) {
        if (IS_AVG_POOLING) {
          warpReduceSum<float, 2>(pooling);
        }
        else {
          warpReduceMax<float, 2>(pooling);
        }
    }
    else {
        if (IS_AVG_POOLING) {
          blockReduceSum<float, 2>(pooling);
        }
        else {
          blockReduceMax<float, 2>(pooling);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        T2 output_val;
        if (IS_AVG_POOLING) {
          output_val.x = T(pooling[0] / HW);
          output_val.y = T(pooling[1] / HW);
        }
        else {
          output_val.x = T(pooling[0]);
          output_val.y = T(pooling[1]);
        }
        output[0] = output_val;
    }
}

template <typename T>
void pooling_nhwc(cutlass::Tensor4DCoord input_tensor_size,
                  cutlass::Tensor4DCoord filter_tensor_size,
                  cutlass::Tensor4DCoord output_tensor_size,
                  cutlass::Tensor4DCoord padding,
                  cutlass::MatrixCoord stride,
                  TensorRef<T, layout::TensorNHWC> ref_input,
                  TensorRef<T, layout::TensorNHWC> ref_output,
                  int poolingType, //0 for avg pooling ; 1 for max pooling
                  cudaStream_t stream) {

  assert(input_tensor_size.n() == output_tensor_size.n() &&
         input_tensor_size.c() == output_tensor_size.c());

  const int N = input_tensor_size.n();
  const int H = input_tensor_size.h();
  const int W = input_tensor_size.w();
  const int C = input_tensor_size.c();
  const int padding_H = padding.h();
  const int padding_W = padding.w();
  const int kernel_H = filter_tensor_size.h();
  const int kernel_W = filter_tensor_size.w();
  const int stride_H = stride.row();
  const int stride_W = stride.column();

  const int output_H = getOutputSize(H, padding_H, kernel_H, stride_H);
  const int output_W = getOutputSize(W, padding_W, kernel_W, stride_W);

  assert(output_tensor_size.h() == output_H &&
         output_tensor_size.w() == output_W);

  if (C % 2 != 0) {
    if ((H == kernel_H && padding_H == 0) && (W == kernel_W && padding_W == 0)) {
      dim3 grid(C, N);
      dim3 block(256);
      if (H*W < block.x){
        block.x = (H*W + 31)/32*32;
      } 
      if (poolingType == 0) {
        pooling_nxhTo1x1_element1_kernel<T, true><<<grid, block, 0, stream>>>(
          ref_output.data(),
          ref_input.data(),
          N,
          H*W,
          C);
      } // if (poolingType == 0)
      else {
        pooling_nxhTo1x1_element1_kernel<T, false><<<grid, block, 0, stream>>>(
          ref_output.data(),
          ref_input.data(),
          N,
          H*W,
          C);
      }
    } // if ((H == kernel_H && padding_H == 0) && (W == kernel_W && padding_W == 0))
    else {
      dim3 grid(N, output_H, output_W);
      dim3 block(256);
      if (C < block.x) {
        block.x = C;
      }
      if (poolingType == 0) {
        pooling_nhwc_element1_kernel<T, true><<<grid, block, 0, stream>>>(
          ref_output.data(), 
          ref_input.data(),
          N,
          H,
          W,
          C,
          output_H,
          output_W,
          kernel_H,
          kernel_W,
          stride_H,
          stride_W,
          padding_H,
          padding_W);
      } // if (poolingType == 0)
      else {
        pooling_nhwc_element1_kernel<T, false><<<grid, block, 0, stream>>>(
          ref_output.data(),
          ref_input.data(),
          N,
          H,
          W,
          C,
          output_H,
          output_W,
          kernel_H,
          kernel_W,
          stride_H,
          stride_W,
          padding_H,
          padding_W);
      }
    }
  } // if (C % 2 != 0))
  else {
    if ((H == kernel_H && padding_H == 0) && (W == kernel_W && padding_W == 0)) {
      dim3 grid(C/2, N);
      dim3 block(256);
      if (H*W < block.x){
        block.x = (H*W + 31)/32*32;
      }
      if (poolingType == 0) {
        if (std::is_same<T, float>::value) {
          pooling_nxhTo1x1_element2_kernel<float2, float, true><<<grid, block, 0, stream>>>(
            (float2*)(ref_output.data()),
            (const float2*)(ref_input.data()),
            N,
            H*W,
            C);
        } // if (std::is_same<T, float>::value)
        else {
          pooling_nxhTo1x1_element2_kernel<half2, half, true><<<grid, block, 0, stream>>>(
            (half2*)(ref_output.data()),
            (const half2*)(ref_input.data()),
            N,
            H*W,
            C);
        }
      } // if (poolingType == 0)
      else {
        if (std::is_same<T, float>::value) {
          pooling_nxhTo1x1_element2_kernel<float2, float, false><<<grid, block, 0, stream>>>(
            (float2*)(ref_output.data()),
            (const float2*)(ref_input.data()),
            N,
            H*W,
            C);
        } // if (std::is_same<T, float>::value)
        else {
          pooling_nxhTo1x1_element2_kernel<half2, half, false><<<grid, block, 0, stream>>>(
            (half2*)(ref_output.data()),
            (const half2*)(ref_input.data()),
            N,
            H*W,
            C);
        }
      }
    } // if ((H == kernel_H && padding_H == 0) && (W == kernel_W && padding_W == 0))
    else {
      dim3 grid(N, output_H, output_W);
      dim3 block(256);
      if (C/2 < block.x) {
        block.x = C/2;
      }
      if (poolingType == 0) {
        if (std::is_same<T, float>::value) {
          pooling_nhwc_element2_kernel<float2, float, true><<<grid, block, 0, stream>>>(
            (float2*)(ref_output.data()),
            (const float2*)(ref_input.data()),
            N,
            H,
            W,
            C/2,
            output_H,
            output_W,
            kernel_H,
            kernel_W,
            stride_H,
            stride_W,
            padding_H,
            padding_W);
        } // if (std::is_same<T, float>::value)
        else {
          pooling_nhwc_element2_kernel<half2, half, true><<<grid, block, 0, stream>>>(
            (half2*)(ref_output.data()),
            (const half2*)(ref_input.data()),
            N,
            H,
            W,
            C/2,
            output_H,
            output_W,
            kernel_H,
            kernel_W,
            stride_H,
            stride_W,
            padding_H,
            padding_W);
        }
      } // if (poolingType == 0)
      else {
        if (std::is_same<T, float>::value) {
          pooling_nhwc_element2_kernel<float2, float, false><<<grid, block, 0, stream>>>(
            (float2*)(ref_output.data()),
            (const float2*)(ref_input.data()),
            N,
            H,
            W,
            C/2,
            output_H,
            output_W,
            kernel_H,
            kernel_W,
            stride_H,
            stride_W,
            padding_H,
            padding_W);
        } // if (std::is_same<T, float>::value)
        else {
          pooling_nhwc_element2_kernel<half2, half, false><<<grid, block, 0, stream>>>(
            (half2*)(ref_output.data()),
            (const half2*)(ref_input.data()),
            N,
            H,
            W,
            C/2,
            output_H,
            output_W,
            kernel_H,
            kernel_W,
            stride_H,
            stride_W,
            padding_H,
            padding_W);
        }
      }
    }
  }
}

} //namespace cutlass
