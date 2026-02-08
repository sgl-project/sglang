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
#include "cutlass/arch/wmma.h"

#ifdef CUTLASS_ARCH_WMMA_SM72_ENABLED
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "testbed.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////// WMMA Size = 16x16x16, DataType/Instruction = s8*s8+s32=>s32   //////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
TEST(SM75_Device_Gemm_s8t_s8n_s8n_wmma_tensor_op_s32, 128x128x32_64x64x32_16x16x16) {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::RowMajor,
    int8_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassWmmaTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::epilogue::thread::FastLinearCombinationClamp<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(SM75_Device_Gemm_s8t_s8n_s8n_wmma_tensor_op_s32, 64x128x64_32x32x64_16x16x16) {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::RowMajor,
    int8_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassWmmaTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::epilogue::thread::FastLinearCombinationClamp<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////// WMMA Size = 32x8x16, DataType/Instruction = s8*s8+s32=>s32   //////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////   
TEST(SM75_Device_Gemm_s8t_s8n_s8n_wmma_tensor_op_s32, 64x128x64_32x64x64_32x8x16) {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::RowMajor,
    int8_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassWmmaTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<32, 8, 16>,
    cutlass::epilogue::thread::FastLinearCombinationClamp<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////// WMMA Size = 8x32x16, DataType/Instruction = s8*s8+s32=>s32   //////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
TEST(SM75_Device_Gemm_s8t_s8n_s8n_wmma_tensor_op_s32, 64x128x64_32x64x64_8x32x16) {

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;

  using Gemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::RowMajor,
    int8_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassWmmaTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<8, 32, 16>,
    cutlass::epilogue::thread::FastLinearCombinationClamp<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}
#endif //CUTLASS_ARCH_WMMA_SM72_ENABLED
