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

/*! \file
    \brief Simple permutation kernel implementation.
*/

#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/tensor_view.h"
#include "cutlass/fast_math.h"
#include "cute/numeric/numeric_types.hpp"

namespace example
{

/**
 * Assumes column-major input (M mode is contiguous, N mode is strided).
 * For row major, the inputs must be switched accordingly.
*/
template<bool Batched, typename Element, typename Permute>
__global__ void
permute_kernel(Element const* __restrict__ input,
               Element* __restrict__ output,
               Permute permute,
               int64_t num_elems,
               cutlass::FastDivmod stride_divmod)
{
  // CUTLASS 2.x batched permute functions assume 0 batch stride for target tensor
  Element const * input_b = input + blockIdx.z * num_elems;
  Element * output_b = output + (Batched ? 0 : blockIdx.z * num_elems);
  for (int64_t k = threadIdx.x + blockIdx.x * blockDim.x; k < num_elems; k += blockDim.x * gridDim.x)
  {
    int i, j;
    stride_divmod(j, i, k);
    output_b[permute(cutlass::PitchLinearCoord(i, j))] = input_b[i + j * stride_divmod.divisor];
  }
}

template<bool Batched, typename Permute, typename Element>
void permute(Element const* input,
             Element * output,
             int64_t num_elems,
             int stride,
             int batch_count,
             cutlass::KernelHardwareInfo const& hw_info)
{
  // Upcast to uint128_t data type
  int factor = 128 / cutlass::sizeof_bits<Element>::value;
  assert(stride % factor == 0);
  int stride_upcast = stride/factor;
  int64_t num_elems_upcast = num_elems / factor;
  Permute permute_upcast(cutlass::PitchLinearCoord(stride_upcast, int(num_elems_upcast/stride_upcast)), stride_upcast);

  cutlass::FastDivmod stride_divmod(stride);
  dim3 blocks(hw_info.sm_count, 1, batch_count);
  permute_kernel<Batched><<<blocks, 1024>>>(reinterpret_cast<cute::uint128_t const *>(input), 
                                            reinterpret_cast<cute::uint128_t *>(output),
                                            permute_upcast,
                                            num_elems_upcast,
                                            stride_upcast);
}

} // namespace example
