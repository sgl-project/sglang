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
 * \brief cuda kernels for padding in device memory with NHWC layout.
 */

#include "cutlass/cutlass.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/tensor_ref.h"

namespace cutlass {

/** \brief interface for padding in a device memory tensor with NHWC layout
 * \tparam T: data type
 */
template <typename T>
void nhwc_padding(cutlass::Tensor4DCoord input_tensor_size,
                  cutlass::Tensor4DCoord output_tensor_size,
                  TensorRef<T, layout::TensorNHWC> ref_input,
                  TensorRef<T, layout::TensorNHWC> ref_output,
                  cudaStream_t stream);


template <typename T>
__global__ void nhwc_padding_kernel(const int32_t n,
                                    const int32_t h,
                                    const int32_t w,
                                    const int32_t c_in,
                                    const int32_t c_out,
                                    const T zero,
                                    const T *input,
                                    T *output){

  const int32_t idx_jump       = blockDim.x * gridDim.x;
  const int32_t total_elements = n * h * w * c_out;

  int32_t c_idx, w_idx, h_idx, n_idx, resudial;

  T value;
  for (int32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += idx_jump) {
        
    c_idx = idx%c_out;
    if (c_idx >= c_in){
      value = zero;    
    }
    else{
      resudial = idx/c_out;
      w_idx = resudial%w;
      resudial = resudial/w;
      h_idx = resudial%h;
      n_idx = resudial/h;	
      resudial = ((n_idx * h + h_idx) * w + w_idx) * c_in + c_idx;
      value = input[resudial];
    }
    output[idx] = value;
  }
}


// fast kernel for c_in = 3 & c_out = 4
template <typename Tio, typename Telement, int element_in_Tio>
__global__ void nhwc_padding_channel_3To4_kernel(const int32_t n,
                                                 const int32_t h,
                                                 const int32_t w,
                                                 const Tio *input,
                                                 Tio *output,
                                                 const int32_t max_output_element,
                                                 const int32_t max_input_element,
                                                 const Tio zero_io,
                                                 const Telement zero_element){                                                
  __shared__ Tio shm[192];
  const int tidx = blockIdx.x * 192 + threadIdx.x;  
  const int threadidx = threadIdx.x; 

  shm[threadIdx.x] = tidx >= max_input_element ? zero_io : input[tidx];  
  __syncthreads();
  
  const int output_offset = blockIdx.x * 256;
  const int lower_bound = max_output_element < output_offset + 256 ? max_output_element : output_offset + 256;
  for (int i = output_offset + threadidx, j = threadidx ; i < lower_bound ; i+=192, j+=192)
  {
    const Telement* shm_element = (const Telement*)shm + j*3*element_in_Tio/4;
    Telement array[element_in_Tio];
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0 ; k < element_in_Tio ; k++)
      array[k] = ((k+1)%4 == 0) ? zero_element : shm_element[(k > 3) ? (k - 1) : k];
    output[i] = *((const Tio *)array);
  }
}

// fast kernel for c_in = 3 & c_out = 8
template <typename Tio, typename Telement, int element_in_Tio>
__global__ void nhwc_padding_channel_3To8_kernel(const int32_t n,
                                                 const int32_t h,
                                                 const int32_t w,
                                                 const Tio *input,
                                                 Tio *output,
                                                 const int32_t max_output_element,
                                                 const int32_t max_input_element,
                                                 const Tio zero_io,
                                                 const Telement zero_element){                                                
  __shared__ Tio shm[192];
  const int tidx = blockIdx.x * 192 + threadIdx.x;  
  const int threadidx = threadIdx.x; 

  shm[threadIdx.x] = tidx >= max_input_element ? zero_io : input[tidx];  
  __syncthreads();
  
  const int output_offset = blockIdx.x * 512;
  const int lower_bound = max_output_element < output_offset + 512 ? max_output_element : output_offset + 512;
  for (int i = output_offset + threadidx, j = threadidx ; i < lower_bound ; i+=192, j+=192)
  {
    const Telement* shm_element = (const Telement*)shm + (element_in_Tio == 4 ? j/2 : j)*3;
    Telement array[element_in_Tio];
    //float
    if (element_in_Tio == 4){
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0 ; k < element_in_Tio ; k++)
        array[k] = ((j % 2) == 1) ? zero_element : ((k >= 3) ? zero_element : shm_element[k]);
    }
    //half
    else{
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0 ; k < element_in_Tio ; k++) 
        array[k] = (k >= 3) ? zero_element : shm_element[k];          
    }
    output[i] = *((const Tio *)array);
  }
}

template <typename T>
void nhwc_padding(cutlass::Tensor4DCoord input_tensor_size,
                  cutlass::Tensor4DCoord output_tensor_size,
                  TensorRef<T, layout::TensorNHWC> ref_input,
                  TensorRef<T, layout::TensorNHWC> ref_output,
                  cudaStream_t stream){
  assert(
    input_tensor_size.n() == output_tensor_size.n() &&
    input_tensor_size.h() == output_tensor_size.h() &&
    input_tensor_size.w() == output_tensor_size.w() &&
    input_tensor_size.c() <= output_tensor_size.c()); 
    
  int n = input_tensor_size.n();
  int h = input_tensor_size.h();
  int w = input_tensor_size.w();
  int c_in = input_tensor_size.c();
  int c_out = output_tensor_size.c();
    
  //case 1 : channel == 3 padding to 4 or 8
  if ((c_out == 4 || c_out == 8) && c_in == 3 && (n*h*w % 8 == 0)){
    dim3 block(192);
    const int nhw = n*h*w;
    const int nhwc = nhw*c_in;
    //for half_t
    if (cutlass::sizeof_bits<T>::value == 16){
      const int element_in_Tio = 8;
      const int max_input_element = nhwc/element_in_Tio;
      const int max_output_element = nhw*c_out/element_in_Tio;
      const int4 zero_io = {0, 0, 0, 0};
      const half_t zero_element = static_cast<half_t>(0.0f);
      dim3 grid((nhwc + 192*element_in_Tio - 1)/(192*element_in_Tio));
      if (c_out == 4){
        nhwc_padding_channel_3To4_kernel<int4, half_t, element_in_Tio><<<grid, block, 0, stream>>>
          (n, h, w,
          (const int4 *)ref_input.data(),
          (int4 *)ref_output.data(),
          max_output_element,
          max_input_element,
          zero_io,
          zero_element);
      }
      else if (c_out == 8){
        nhwc_padding_channel_3To8_kernel<int4, half_t, element_in_Tio><<<grid, block, 0, stream>>>
          (n, h, w,
          (const int4 *)ref_input.data(),
          (int4 *)ref_output.data(),
          max_output_element,
          max_input_element,
          zero_io,
          zero_element);
      }
    }
    //for float
    else{
      const int element_in_Tio = 4;
      const int max_input_element = nhwc/element_in_Tio;
      const int max_output_element = nhw*c_out/element_in_Tio;
      const float4 zero_io = {0.0f, 0.0f, 0.0f, 0.0f};
      const float zero_element = 0.0f;
      dim3 grid((nhwc + 192*element_in_Tio - 1)/(192*element_in_Tio));
      if (c_out == 4){
        nhwc_padding_channel_3To4_kernel<float4, float, element_in_Tio><<<grid, block, 0, stream>>>
          (n, h, w,
          (const float4 *)ref_input.data(),
          (float4 *)ref_output.data(),
          max_output_element,
          max_input_element,
          zero_io,
          zero_element);
      }
      else if (c_out == 8){
        nhwc_padding_channel_3To8_kernel<float4, float, element_in_Tio><<<grid, block, 0, stream>>>
          (n, h, w,
          (const float4 *)ref_input.data(),
          (float4 *)ref_output.data(),
          max_output_element,
          max_input_element,
          zero_io,
          zero_element);
      }
    }
  }
  //case 2 : even channel
  else if ((c_out % 2) == 0 && (c_in % 2) == 0){
    int32_t total_elements = n * h * w * c_out / 2;
    int block_size = 256;
    dim3 grid((total_elements + 255)/256);
    dim3 block(block_size);
    //for half_t
    if (cutlass::sizeof_bits<T>::value == 16){
      const __half2 zero  = {0.0f, 0.0f};
      nhwc_padding_kernel<<<grid, block, 0, stream>>>(n, h, w, c_in/2, c_out/2, zero, (const __half2*)ref_input.data(), (__half2*)ref_output.data());
    }
    //for float
    else{
      const float2 zero  = {0.0f, 0.0f};
      nhwc_padding_kernel<<<grid, block, 0, stream>>>(n, h, w, c_in/2, c_out/2, zero, (const float2*)ref_input.data(), (float2*)ref_output.data());
    }
  }
  //case 3 : odd channel
  else{
    int32_t total_elements = n * h * w * c_out;
    int block_size = 256;
    dim3 grid((total_elements + 255)/256);
    dim3 block(block_size);
    const T zero = static_cast<T>(0.0f);
    nhwc_padding_kernel<<<grid, block, 0, stream>>>(n, h, w, c_in, c_out, zero, ref_input.data(), ref_output.data());
  }
}


} //namespace cutlass
