/***************************************************************************************************
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
 **************************************************************************************************/
/* \file
   \brief 
*/
  
#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "conv_reference_operation.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_conv3d_reference_operations(Manifest &manifest) {

  make_conv_all<
    3,
    cutlass::half_t, cutlass::layout::TensorNDHWC,
    cutlass::half_t, cutlass::layout::TensorNDHWC,
    cutlass::half_t, cutlass::layout::TensorNDHWC,
    cutlass::half_t,
    cutlass::half_t
  >(manifest);

  make_conv_all<
    3,
    cutlass::half_t, cutlass::layout::TensorNDHWC,
    cutlass::half_t, cutlass::layout::TensorNDHWC,
    cutlass::half_t, cutlass::layout::TensorNDHWC,
    float,
    float
  >(manifest);

  make_conv_all<
    3,
    cutlass::half_t, cutlass::layout::TensorNDHWC,
    cutlass::half_t, cutlass::layout::TensorNDHWC,
    float, cutlass::layout::TensorNDHWC,
    float,
    float
  >(manifest);

  make_conv_all<
    3,
    cutlass::bfloat16_t, cutlass::layout::TensorNDHWC,
    cutlass::bfloat16_t, cutlass::layout::TensorNDHWC,
    cutlass::bfloat16_t, cutlass::layout::TensorNDHWC,
    float,
    float
  >(manifest);

  make_conv_all<
    3,
    cutlass::bfloat16_t, cutlass::layout::TensorNDHWC,
    cutlass::bfloat16_t, cutlass::layout::TensorNDHWC,
    float, cutlass::layout::TensorNDHWC,
    float,
    float
  >(manifest);

  make_conv_all<
    3,
    cutlass::tfloat32_t, cutlass::layout::TensorNDHWC,
    cutlass::tfloat32_t, cutlass::layout::TensorNDHWC,
    cutlass::tfloat32_t, cutlass::layout::TensorNDHWC,
    float,
    float
  >(manifest);

  make_conv_all<
    3,
    cutlass::tfloat32_t, cutlass::layout::TensorNDHWC,
    cutlass::tfloat32_t, cutlass::layout::TensorNDHWC,
    float, cutlass::layout::TensorNDHWC,
    float,
    float
  >(manifest);

  make_conv_all<
    3,
    float, cutlass::layout::TensorNDHWC,
    float, cutlass::layout::TensorNDHWC,
    float, cutlass::layout::TensorNDHWC,
    float,
    float
  >(manifest);

  make_conv_fprop<
    3,
    int8_t, cutlass::layout::TensorNDHWC,
    int8_t, cutlass::layout::TensorNDHWC,
    int32_t, cutlass::layout::TensorNDHWC,
    int32_t,
    int32_t,
    NumericConverterClamp<int32_t, int32_t>
  >(manifest);

  make_conv_fprop<
    3,
    int8_t, cutlass::layout::TensorNDHWC,
    int8_t, cutlass::layout::TensorNDHWC,
    int8_t, cutlass::layout::TensorNDHWC,
    float,
    int32_t,
    NumericConverterClamp<int8_t, float>
  >(manifest);

  make_conv_fprop<
    3,
    uint8_t, cutlass::layout::TensorNDHWC,
    uint8_t, cutlass::layout::TensorNDHWC,
    int32_t, cutlass::layout::TensorNDHWC,
    int32_t,
    int32_t,
    NumericConverterClamp<int32_t, int32_t>
  >(manifest);

  make_conv_fprop<
    3,
    uint8_t, cutlass::layout::TensorNDHWC,
    uint8_t, cutlass::layout::TensorNDHWC,
    int8_t, cutlass::layout::TensorNDHWC,
    float,
    int32_t,
    NumericConverterClamp<int8_t, float>
  >(manifest);

  make_conv_fprop<
    3,
    cutlass::int4b_t, cutlass::layout::TensorNDHWC,
    cutlass::int4b_t, cutlass::layout::TensorNDHWC,
    int32_t, cutlass::layout::TensorNDHWC,
    int32_t,
    int32_t,
    NumericConverterClamp<int32_t, int32_t>
  >(manifest);

  make_conv_fprop<
    3,
    cutlass::int4b_t, cutlass::layout::TensorNDHWC,
    cutlass::int4b_t, cutlass::layout::TensorNDHWC,
    cutlass::int4b_t, cutlass::layout::TensorNDHWC,
    float,
    int32_t,
    NumericConverterClamp<cutlass::int4b_t, float>
  >(manifest);
  
  make_conv_fprop<
    3,
    cutlass::uint4b_t, cutlass::layout::TensorNDHWC,
    cutlass::uint4b_t, cutlass::layout::TensorNDHWC,
    int32_t, cutlass::layout::TensorNDHWC,
    int32_t,
    int32_t,
    NumericConverterClamp<int32_t, int32_t>
  >(manifest);

  make_conv_fprop<
    3,
    cutlass::uint4b_t, cutlass::layout::TensorNDHWC,
    cutlass::uint4b_t, cutlass::layout::TensorNDHWC,
    cutlass::uint4b_t, cutlass::layout::TensorNDHWC,
    float,
    int32_t,
    NumericConverterClamp<cutlass::uint4b_t, float>
  >(manifest);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library 
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

