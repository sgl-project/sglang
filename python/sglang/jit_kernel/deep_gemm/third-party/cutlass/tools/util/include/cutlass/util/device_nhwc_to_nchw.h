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
 * \brief cuda kernels to transform a device memory tensor from NHWC layout to NCHW layout.
 */

#include "cutlass/cutlass.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/tensor_ref.h"

namespace cutlass {

/** \brief interface to transform a device memory tensor from NHWC layout to NCHW layout.
 * \tparam T: data type
 */
template <typename T>
void nhwc_to_nchw(cutlass::Tensor4DCoord input_tensor_size,
                  cutlass::Tensor4DCoord output_tensor_size,
                  TensorRef<T, layout::TensorNHWC> ref_input,
                  TensorRef<T, layout::TensorNCHW> ref_output,
                  cudaStream_t stream);


template <typename T>
__global__ void nhwc_to_nchw_kernel(T *output, 
                                    const T *input, 
                                    const int n,
                                    const int h, 
                                    const int w, 
                                    const int c) {
 
  const int hw = h*w;
  const int hwc = hw*c;
  __shared__ T shbuf[32 * (32 + 1)]; 
  const int32_t tid  = threadIdx.y*blockDim.x + threadIdx.x;
  const int32_t wid  = tid / 32; 
  const int32_t lid  = tid % 32; 
  const int32_t ni   = blockIdx.z;
  const int32_t hwi0  = blockIdx.y * 32;  
  const int32_t ci0 = blockIdx.x * 32;  

  const size_t input_idx = ni * hwc + (hwi0 + wid) * c + ci0;
  const T *A = input + input_idx;
  if (ci0 + lid < c) {
    const int lid_x_33 = lid * 33;
    if ((hwi0 + 32) <= hw) {
      int hwi = wid;  // between 0 and 7
      CUTLASS_PRAGMA_UNROLL
      for (int cLoopIdx = 0; cLoopIdx < 4; cLoopIdx++) { 
        shbuf[lid_x_33 + hwi] = A[lid];
        A                     = &A[8 * c];
        hwi += 8;
      }
    } else {
      for (int hwi = wid; hwi < 32; hwi += 8) { 
        if ((hwi + hwi0) < hw) {
          shbuf[lid_x_33 + hwi] = A[lid];
        }
        A = &A[8 * c];
      }
    }
  }
  __syncthreads();

  const int32_t hwiOut = hwi0 + lid;
  output = &output[ni * hwc + hwiOut];
  if (hwiOut < hw) {
    if (ci0 + 32 < c) {
      int cI = wid;
      CUTLASS_PRAGMA_UNROLL
      for (int hwLoopIdx = 0; hwLoopIdx < 4; ++hwLoopIdx) {
        output[(ci0 + cI) * hw] = shbuf[(cI)*33 + lid];
        cI += 8;
      }
    } else {
      for (int cI = wid; cI < 32; cI += 8) {
        if (ci0 + cI < c) {
          output[(ci0 + cI) * hw] = shbuf[(cI)*33 + lid];
        }
      }
    }
  }
}

template <typename T>
void nhwc_to_nchw(cutlass::Tensor4DCoord input_tensor_size,
                  cutlass::Tensor4DCoord output_tensor_size,
                  TensorRef<T, layout::TensorNHWC> ref_input,
                  TensorRef<T, layout::TensorNCHW> ref_output,
                  cudaStream_t stream) {
  
  assert(
    input_tensor_size.n() == output_tensor_size.n() &&
    input_tensor_size.h() == output_tensor_size.c() &&
    input_tensor_size.w() == output_tensor_size.h() &&
    input_tensor_size.c() == output_tensor_size.w());

  int n = input_tensor_size.n();
  int h = input_tensor_size.h();
  int w = input_tensor_size.w();
  int c = input_tensor_size.c();

  dim3 grid((c + 31)/32, (h*w + 31)/32, n);
  dim3 block(32, 8);
  nhwc_to_nchw_kernel<<<grid, block, 0, stream>>>(ref_output.data(), ref_input.data(), 
                                                  n, h, w, c);

}

} //namespace cutlass
