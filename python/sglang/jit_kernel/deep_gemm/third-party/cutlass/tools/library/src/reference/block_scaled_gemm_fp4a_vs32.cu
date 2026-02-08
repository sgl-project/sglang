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

#include "block_scaled_gemm_reference_operation.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_block_scaled_gemm_reference_operations_fp4a_vs32(Manifest &manifest) {
  ////////////////////////////////////////////////////////////////////////////////////////////////////////// 
  // SFVectorSize = 32 with MxF4 instructions
  //////////////////////////////////////////////////////////////////////////////////////////////////////////

  // (float_e2m1_t * float_ue8m0_t) * (float_e2m1_t * float_ue8m0_t)
  make_block_scaled_gemm<
    float_e2m1_t /*A*/, float_ue8m0_t /*SFA*/, float_e2m1_t /*B*/, float_ue8m0_t /*SFB*/,
    void  /*C*/, float /*Compute*/, void /*SFD*/, float /*Accum*/, float  /*D*/, 32 /*SFVecSize*/
  >(manifest);

  make_block_scaled_gemm<
    float_e2m1_t /*A*/, float_ue8m0_t /*SFA*/, float_e2m1_t /*B*/, float_ue8m0_t /*SFB*/,
    void  /*C*/, float /*Compute*/, void /*SFD*/, float /*Accum*/, float_e5m2_t /*D*/, 32 /*SFVecSize*/
  >(manifest);

  make_block_scaled_gemm<
    float_e2m1_t /*A*/, float_ue8m0_t /*SFA*/, float_e2m1_t /*B*/, float_ue8m0_t /*SFB*/,
    half_t /*C*/, float /*Compute*/, void /*SFD*/, float /*Accum*/, float_e5m2_t /*D*/, 32 /*SFVecSize*/
  >(manifest);

  make_block_scaled_gemm<
    float_e2m1_t /*A*/, float_ue8m0_t /*SFA*/, float_e2m1_t /*B*/, float_ue8m0_t /*SFB*/,
    half_t /*C*/, float /*Compute*/, void /*SFD*/, float /*Accum*/, float_e3m2_t /*D*/, 32 /*SFVecSize*/
  >(manifest);

  // With SF generation reference
  make_block_scaled_gemm<
    float_e2m1_t /*A*/, float_ue8m0_t /*SFA*/, float_e2m1_t /*B*/, float_ue8m0_t /*SFB*/,
    void  /*C*/, float /*Compute*/, float_ue8m0_t /*SFD*/, float /*Accum*/, float_e2m1_t /*D*/, 32 /*SFVecSize*/,
    16 /*EpiSFVecSize*/
  >(manifest);
  make_block_scaled_gemm<
    float_e2m1_t /*A*/, float_ue8m0_t /*SFA*/, float_e2m1_t /*B*/, float_ue8m0_t /*SFB*/,
    half_t  /*C*/, float /*Compute*/, float_ue8m0_t /*SFD*/, float /*Accum*/, float_e2m1_t /*D*/, 32 /*SFVecSize*/,
    16 /*EpiSFVecSize*/
  >(manifest);

  make_block_scaled_gemm<
    float_e2m1_t /*A*/, float_ue8m0_t /*SFA*/, float_e2m1_t /*B*/, float_ue8m0_t /*SFB*/,
    void  /*C*/, float /*Compute*/, float_ue8m0_t /*SFD*/, float /*Accum*/, float_e2m1_t /*D*/, 32 /*SFVecSize*/,
    32 /*EpiSFVecSize*/
  >(manifest);

  make_block_scaled_gemm<
    float_e2m1_t /*A*/, float_ue8m0_t /*SFA*/, float_e2m1_t /*B*/, float_ue8m0_t /*SFB*/,
    void  /*C*/, float /*Compute*/, float_ue8m0_t /*SFD*/, float /*Accum*/, float_e5m2_t /*D*/, 32 /*SFVecSize*/,
    32 /*EpiSFVecSize*/
  >(manifest);

  make_block_scaled_gemm<
    float_e2m1_t /*A*/, float_ue8m0_t /*SFA*/, float_e2m1_t /*B*/, float_ue8m0_t /*SFB*/,
    void  /*C*/, float /*Compute*/, float_ue8m0_t /*SFD*/, float /*Accum*/, float_e3m2_t /*D*/, 32 /*SFVecSize*/,
    32 /*EpiSFVecSize*/
  >(manifest);

  make_block_scaled_gemm<
    float_e2m1_t /*A*/, float_ue8m0_t /*SFA*/, float_e2m1_t /*B*/, float_ue8m0_t /*SFB*/,
    half_t  /*C*/, float /*Compute*/, float_ue8m0_t /*SFD*/, float /*Accum*/, float_e2m1_t /*D*/, 32 /*SFVecSize*/,
    32 /*EpiSFVecSize*/
  >(manifest);

  make_block_scaled_gemm<
    float_e2m1_t /*A*/, float_ue8m0_t /*SFA*/, float_e2m1_t /*B*/, float_ue8m0_t /*SFB*/,
    half_t  /*C*/, float /*Compute*/, float_ue8m0_t /*SFD*/, float /*Accum*/, float_e5m2_t /*D*/, 32 /*SFVecSize*/,
    32 /*EpiSFVecSize*/
  >(manifest);

  make_block_scaled_gemm<
    float_e2m1_t /*A*/, float_ue8m0_t /*SFA*/, float_e2m1_t /*B*/, float_ue8m0_t /*SFB*/,
    half_t  /*C*/, float /*Compute*/, float_ue8m0_t /*SFD*/, float /*Accum*/, float_e3m2_t /*D*/, 32 /*SFVecSize*/,
    32 /*EpiSFVecSize*/
  >(manifest);

}
///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
