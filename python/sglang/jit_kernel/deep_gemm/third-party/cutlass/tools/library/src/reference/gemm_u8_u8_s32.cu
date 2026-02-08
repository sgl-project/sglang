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
   \brief Instantiates GEMM reference implementations.
*/

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

#include "gemm_reference_operation.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

// A/B: u8
// Acc : s32
// C/D: some variance

// 1. u8_u8_s32_s32_s32 (s32 epi scalar)
// 2. u8_u8_s32_s32_s32 (f32 epi scalar)
// 3. u8_8_s32_s8_s8 (f32 epi scalar)
// 3. u8_8_s32_s8_s8 (s epi scalar)

void initialize_gemm_reference_operations_u8_u8_s32(Manifest &manifest) {
  // 1.
  make_gemm_real_canonical_layouts<
    uint8_t,                          // ElementA
    uint8_t,                          // ElementB
    int32_t,                          // ElementC
    int32_t,                          // ElementScalar / ElementCompute
    int32_t,                          // ElementAccumulator
    int32_t                           // ElementD
  >(manifest);

  // 2.
  make_gemm_real_canonical_layouts<
    uint8_t,                          // ElementA
    uint8_t,                          // ElementB
    int32_t,                          // ElementC
    float,                            // ElementScalar / ElementCompute
    int32_t,                          // ElementAccumulator
    int32_t,                          // ElementD
    NumericConverterClamp<int32_t, float> // From Scalar to D
  >(manifest);

  // 3.
  make_gemm_real_canonical_layouts<
    uint8_t,                          // ElementA
    uint8_t,                          // ElementB
    int8_t,                           // ElementC
    float,                            // ElementScalar / ElementCompute
    int32_t,                          // ElementAccumulator
    int8_t,                           // ElementD
    NumericConverterClamp<int8_t, float> // From Scalar to D
  >(manifest);

  // 4.
  make_gemm_real_canonical_layouts<
    uint8_t,                          // ElementA
    uint8_t,                          // ElementB
    int8_t,                           // ElementC
    float,                            // ElementScalar / ElementCompute
    int32_t,                          // ElementAccumulator
    uint8_t,                          // ElementD
    NumericConverterClamp<uint8_t, float> // From Scalar to D
  >(manifest);

}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

