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
   \brief Initialize operations for reduction operation in CUTLASS Library.

*/

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

namespace cutlass {
namespace library {
///////////////////////////////////////////////////////////////////////////////////////////////
//                             CUTLASS Reduction Instances                                   //
///////////////////////////////////////////////////////////////////////////////////////////////

void initialize_reduce_add_linear_combination_f16_f16_f16(Manifest &manifest);
void initialize_reduce_add_linear_combination_f32_f32_f16(Manifest &manifest);
void initialize_reduce_add_linear_combination_f32_f32_bf16(Manifest &manifest);
void initialize_reduce_add_linear_combination_f32_f32_f32(Manifest &manifest);
void initialize_reduce_add_linear_combination_f64_f64_f64(Manifest &manifest);
void initialize_reduce_add_linear_combination_cf32_cf32_cf32(Manifest &manifest);

//
// Entry point to construct operations
//
void initialize_all_reduction_op(Manifest &manifest) {

  initialize_reduce_add_linear_combination_f16_f16_f16(manifest);
  initialize_reduce_add_linear_combination_f32_f32_f16(manifest);
  initialize_reduce_add_linear_combination_f32_f32_bf16(manifest);
  initialize_reduce_add_linear_combination_f32_f32_f32(manifest);
  initialize_reduce_add_linear_combination_f64_f64_f64(manifest);
  initialize_reduce_add_linear_combination_cf32_cf32_cf32(manifest);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass
