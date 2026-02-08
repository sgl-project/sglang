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
/*! \file
    \brief Tests for device-wide GEMM interface
    
*/

#include <iostream>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"

#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed_universal.h"

////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

TEST(SM75_Device_GemmUniversal_f16n_f16t_f32n_tensor_op_f32, 64x64x32_32x32x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::GemmUniversal<
      cutlass::half_t, 
      cutlass::layout::ColumnMajor, 
      cutlass::half_t,
      cutlass::layout::RowMajor, 
      ElementOutput, cutlass::layout::ColumnMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>, 
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 
      2>;

  EXPECT_TRUE(test::gemm::device::TestAllGemmUniversal<Gemm>());
}


TEST(SM75_Device_GemmUniversal_f16n_f16t_f32n_tensor_op_f32, 64x64x32_32x32x32_updated_batch_count) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::GemmUniversal<
      cutlass::half_t, 
      cutlass::layout::ColumnMajor, 
      cutlass::half_t,
      cutlass::layout::RowMajor, 
      ElementOutput, cutlass::layout::ColumnMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>, 
      cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 
      2,
      1,
      1>;

  EXPECT_TRUE(test::gemm::device::TestGemmUniversal<Gemm>(
    {128, 128, 2}, 
    cutlass::gemm::GemmUniversalMode::kGemm, 
    15));
}

////////////////////////////////////////////////////////////////////////////////

#endif // #if defined(CUTLASS_ARCH_MMA_SM75_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

