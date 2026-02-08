/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Tests for device-wide sparse GEMM interface with:
        A: row major, of type FE4M4 or FE5M2
        B: column major, of type FE4M3 or FE5M2
        C: row major, of type F32
        Accum: F32
*/

#include <iostream>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_sparse_universal.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed_sparse.h"

#if defined(CUTLASS_ARCH_SPARSE_MMA_F32_SM89_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Sparse_Gemm_fe4m3t_fe4m3n_f32t_tensor_op_f32, 128x128x128_64x64x128) {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = float;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static int const kStages = 3;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
      cutlass::gemm::GemmShape<128, 128, 128>, cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<16, 8, 64>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Sparse_Gemm_fe4m3t_fe5m2n_f32t_tensor_op_f32, 128x128x128_64x64x128) {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e5m2_t;
  using ElementOutput = float;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static int const kStages = 3;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
      cutlass::gemm::GemmShape<128, 128, 128>, cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<16, 8, 64>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Sparse_Gemm_fe5m2t_fe4m3n_f32t_tensor_op_f32, 128x128x128_64x64x128) {
  using ElementA = cutlass::float_e5m2_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = float;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static int const kStages = 3;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
      cutlass::gemm::GemmShape<128, 128, 128>, cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<16, 8, 64>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Sparse_Gemm_fe5m2t_fe5m2n_f32t_tensor_op_f32, 128x128x128_64x64x128) {
  using ElementA = cutlass::float_e5m2_t;
  using ElementB = cutlass::float_e5m2_t;
  using ElementOutput = float;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static int const kStages = 3;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
      cutlass::gemm::GemmShape<128, 128, 128>, cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<16, 8, 64>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

////////////////////////////////////////////////////////////////////////////////

#endif  // CUTLASS_ARCH_MMA_F32_SM89_SUPPORTED
