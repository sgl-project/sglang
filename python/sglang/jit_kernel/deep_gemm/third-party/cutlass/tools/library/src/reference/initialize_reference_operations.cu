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

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

// note: init methods for the same op-class may be split into multiple to parallelize compilation
void initialize_gemm_reference_operations_int4(Manifest &manifest);
void initialize_gemm_reference_operations_int8_interleaved_32(Manifest &manifest);
void initialize_gemm_reference_operations_int8_interleaved_64(Manifest &manifest);
void initialize_gemm_reference_operations_s8_s8_s32(Manifest &manifest);
void initialize_gemm_reference_operations_u8_u8_s32(Manifest &manifest);
void initialize_gemm_reference_operations_e4m3a_e4m3out(Manifest &manifest);
void initialize_gemm_reference_operations_e5m2a_e4m3out(Manifest &manifest);
void initialize_gemm_reference_operations_e4m3a_e5m2out(Manifest &manifest);
void initialize_gemm_reference_operations_e5m2a_e5m2out(Manifest &manifest);

void initialize_gemm_reference_operations_f4_f4_f32(Manifest &manifest);
void initialize_gemm_reference_operations_f4_f6_f32(Manifest &manifest);
void initialize_gemm_reference_operations_f4_f8_f32(Manifest &manifest);
void initialize_gemm_reference_operations_f6_f4_f32(Manifest &manifest);
void initialize_gemm_reference_operations_f6_f6_f32(Manifest &manifest);
void initialize_gemm_reference_operations_f6_f8_f32(Manifest &manifest);
void initialize_gemm_reference_operations_f8_f4_f32(Manifest &manifest);
void initialize_gemm_reference_operations_f8_f6_f32(Manifest &manifest);
void initialize_block_scaled_gemm_reference_operations_fp4a_vs16(Manifest &manifest);
void initialize_block_scaled_gemm_reference_operations_fp4a_vs32(Manifest &manifest);
void initialize_block_scaled_gemm_reference_operations_mixed8bitsa(Manifest &manifest);
void initialize_blockwise_gemm_reference_operations_fp32out(Manifest &manifest);
void initialize_blockwise_gemm_reference_operations_fp16out(Manifest &manifest);
void initialize_blockwise_gemm_reference_operations_bf16out(Manifest &manifest);

void initialize_gemm_reference_operations_fp8in_fp16out(Manifest &manifest);
void initialize_gemm_reference_operations_fp8in_bf16out(Manifest &manifest);
void initialize_gemm_reference_operations_fp8in_fp32out(Manifest &manifest);
void initialize_gemm_reference_operations_fp32out(Manifest &manifest);
void initialize_gemm_reference_operations_fp_other(Manifest &manifest);
void initialize_gemm_reference_operations_fp_mixed_input(Manifest &manifest);
void initialize_gemm_reference_operations_int_mixed_input(Manifest &manifest);

void initialize_conv2d_reference_operations(Manifest &manifest);
void initialize_conv3d_reference_operations(Manifest &manifest);

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_reference_operations(Manifest &manifest) {
  initialize_conv2d_reference_operations(manifest);
  initialize_conv3d_reference_operations(manifest);

  initialize_gemm_reference_operations_int4(manifest);

  initialize_gemm_reference_operations_int8_interleaved_32(manifest);
  initialize_gemm_reference_operations_int8_interleaved_64(manifest);
  initialize_gemm_reference_operations_s8_s8_s32(manifest);
  initialize_gemm_reference_operations_u8_u8_s32(manifest);

  initialize_gemm_reference_operations_e4m3a_e4m3out(manifest);
  initialize_gemm_reference_operations_e5m2a_e4m3out(manifest);
  initialize_gemm_reference_operations_e4m3a_e5m2out(manifest);
  initialize_gemm_reference_operations_e5m2a_e5m2out(manifest);
  initialize_gemm_reference_operations_fp8in_fp16out(manifest);
  initialize_gemm_reference_operations_fp8in_bf16out(manifest);
  initialize_gemm_reference_operations_fp8in_fp32out(manifest);

  initialize_gemm_reference_operations_fp32out(manifest);
  initialize_gemm_reference_operations_fp_other(manifest);
  initialize_gemm_reference_operations_fp_mixed_input(manifest);
  initialize_gemm_reference_operations_int_mixed_input(manifest);

  
  initialize_gemm_reference_operations_f4_f4_f32(manifest);
  initialize_gemm_reference_operations_f4_f6_f32(manifest);
  initialize_gemm_reference_operations_f4_f8_f32(manifest);
  initialize_gemm_reference_operations_f6_f4_f32(manifest);
  initialize_gemm_reference_operations_f6_f6_f32(manifest);
  initialize_gemm_reference_operations_f6_f8_f32(manifest);
  initialize_gemm_reference_operations_f8_f4_f32(manifest);
  initialize_gemm_reference_operations_f8_f6_f32(manifest);
  initialize_block_scaled_gemm_reference_operations_fp4a_vs16(manifest);
  initialize_block_scaled_gemm_reference_operations_fp4a_vs32(manifest);
  initialize_block_scaled_gemm_reference_operations_mixed8bitsa(manifest);
  initialize_blockwise_gemm_reference_operations_fp32out(manifest);
  initialize_blockwise_gemm_reference_operations_fp16out(manifest);
  initialize_blockwise_gemm_reference_operations_bf16out(manifest);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

