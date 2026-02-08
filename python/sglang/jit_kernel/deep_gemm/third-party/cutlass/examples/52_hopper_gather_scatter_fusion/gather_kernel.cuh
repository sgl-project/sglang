/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 **************************************************************************************************/
#pragma once

#include "cute/numeric/math.hpp"

namespace example
{

// Naive grid-stride loop implementation of gather
template<typename Element, typename Func>
__global__ void
gather_kernel(Element const * __restrict__ input,
              Element       * __restrict__ output,
              Func func,
              int num_elems_input,
              int num_elems_output,
              cutlass::FastDivmod stride_divmod)
{
  Element const * input_b = input + blockIdx.z * num_elems_input;
  Element * output_b = output + blockIdx.z * num_elems_output;
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int k = tidx; k < num_elems_output; k += blockDim.x * gridDim.x) {
    int i,j;
    stride_divmod(j, i, k);
    output_b[k] = input_b[i + func(j) * stride_divmod.divisor];
  }
}

// Gather elements along strided dimension of the tensor according to given indices
template<typename Element, typename Func>
void
gather(Element const * input,
       Element * output,
       Func func,
       int batch_size,
       int num_elems_input,
       int num_elems_output,
       int stride,
       cutlass::KernelHardwareInfo const& hw_info)
{
  // Upcast to uint128_t data type
  int factor = 128 / cutlass::sizeof_bits<Element>::value;
  assert(stride % factor == 0);
  int stride_upcast = stride/factor;
  int num_elems_input_upcast = num_elems_input / factor;
  int num_elems_output_upcast = num_elems_output / factor;

  cutlass::FastDivmod stride_divmod(stride_upcast);
  dim3 blocks(hw_info.sm_count, 1, batch_size);
  gather_kernel<<<blocks, 1024>>>(reinterpret_cast<cute::uint128_t const *>(input),
                                  reinterpret_cast<cute::uint128_t *>(output),
                                  func,
                                  num_elems_input_upcast,
                                  num_elems_output_upcast,
                                  stride_divmod);
}

// Naive grid-stride loop implementation of scatter
template<typename Element, typename Func>
__global__ void
scatter_kernel(Element const * __restrict__ input,
               Element       * __restrict__ output,
               Func func,
               int num_elems_input,
               int num_elems_output,
               cutlass::FastDivmod stride_divmod)
{
  Element const * input_b = input + blockIdx.z * num_elems_input;
  Element * output_b = output + blockIdx.z * num_elems_output;
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int k = tidx; k < num_elems_input; k += blockDim.x * gridDim.x) {
    int i,j;
    stride_divmod(j, i, k);
    output_b[i + func(j) * stride_divmod.divisor] = input_b[k];
  }
}

// Gather elements along strided dimension of the tensor according to given indices
template<typename Element, typename Func>
void
scatter(Element const * input,
        Element * output,
        Func func,
        int batch_size,
        int num_elems_input,
        int num_elems_output,
        int stride,
        cutlass::KernelHardwareInfo const& hw_info)
{
  // Upcast to uint128_t data type
  int factor = 128 / cutlass::sizeof_bits<Element>::value;
  assert(stride % factor == 0);
  int stride_upcast = stride/factor;
  int num_elems_input_upcast = num_elems_input / factor;
  int num_elems_output_upcast = num_elems_output / factor;

  cutlass::FastDivmod stride_divmod(stride_upcast);
  dim3 blocks(hw_info.sm_count, 1, batch_size);
  scatter_kernel<<<blocks, 1024>>>(reinterpret_cast<cute::uint128_t const *>(input),
                                   reinterpret_cast<cute::uint128_t *>(output),
                                   func,
                                   num_elems_input_upcast,
                                   num_elems_output_upcast,
                                   stride_divmod);
}

} // namespace example
