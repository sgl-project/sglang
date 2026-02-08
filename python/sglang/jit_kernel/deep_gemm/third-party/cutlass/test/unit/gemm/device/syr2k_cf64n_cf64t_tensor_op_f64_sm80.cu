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
    \brief Tests for device-wide SYRK interface
  
*/

#include <iostream>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/blas3.h"
#include "cutlass/gemm/device/rank_2k.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/rank_2k.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed_rank2k_universal.h"

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Syr2k_cf64n_cf64t_l_tensor_op_f64, 32x32x16_16x16x16) {

  using ElementA = cutlass::complex<double>;
  using LayoutA = cutlass::layout::ColumnMajor;

  using ElementB = cutlass::complex<double>;
  using LayoutB = cutlass::layout::ColumnMajor;

  using ElementC = cutlass::complex<double>;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = cutlass::complex<double>;

  using Rank2K = cutlass::gemm::device::Rank2K<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    cutlass::FillMode::kLower,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,     // kStages 
    1,     // AlignmentA
    1,     // AlignmentB
    false, // SplitKSerial
    cutlass::arch::OpMultiplyAddComplex,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone,
    cutlass::BlasMode::kSymmetric
  >;

  EXPECT_TRUE(test::gemm::device::TestAllRank2KUniversal<Rank2K>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Syr2k_cf64n_cf64t_u_tensor_op_f64, 32x32x16_16x16x16) {

  using ElementA = cutlass::complex<double>;
  using LayoutA = cutlass::layout::ColumnMajor;

  using ElementB = cutlass::complex<double>;
  using LayoutB = cutlass::layout::ColumnMajor;

  using ElementC = cutlass::complex<double>;
  using LayoutC = cutlass::layout::RowMajor;
  using ElementAccumulator = cutlass::complex<double>;

  using Rank2K = cutlass::gemm::device::Rank2K<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    cutlass::FillMode::kUpper,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,     // kStages 
    1,     // AlignmentA
    1,     // AlignmentB
    false, // SplitKSerial
    cutlass::arch::OpMultiplyAddComplex,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone,
    cutlass::BlasMode::kSymmetric
  >;

  EXPECT_TRUE(test::gemm::device::TestAllRank2KUniversal<Rank2K>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif // #if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
