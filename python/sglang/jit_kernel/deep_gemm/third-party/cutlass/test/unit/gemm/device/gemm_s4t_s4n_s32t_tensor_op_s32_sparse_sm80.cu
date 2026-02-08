/**************************************************************************************************
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
#include "cutlass/gemm/device/gemm_sparse_universal.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed_sparse.h"

#if defined(CUTLASS_ARCH_SPARSE_MMA_SM80_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 128x256x256_64x64x256) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 256, 256>,
      cutlass::gemm::GemmShape<64, 64, 256>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 256x128x256_64x64x256) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 256>,
      cutlass::gemm::GemmShape<64, 64, 256>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 128x128x256_64x64x256) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 256>,
      cutlass::gemm::GemmShape<64, 64, 256>,
      cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 256x64x256_64x64x256) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 64, 256>,
      cutlass::gemm::GemmShape<64, 64, 256>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 64x256x256_64x64x256) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 256, 256>,
      cutlass::gemm::GemmShape<64, 64, 256>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 64x128x256_32x64x256) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 128, 256>,
      cutlass::gemm::GemmShape<32, 64, 256>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 6>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 128x64x256_64x32x256) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 64, 256>,
      cutlass::gemm::GemmShape<64, 32, 256>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 6>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 64x64x256_32x32x256) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 256>,
      cutlass::gemm::GemmShape<32, 32, 256>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 10>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 128x128x512_64x64x512) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 512>,
      cutlass::gemm::GemmShape<64, 64, 512>,
      cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 128x64x512_64x32x512) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 64, 512>,
      cutlass::gemm::GemmShape<64, 32, 512>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 64x64x512_32x32x512) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 512>,
      cutlass::gemm::GemmShape<32, 32, 512>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 6>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 32x128x512_32x32x512) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<32, 128, 512>,
      cutlass::gemm::GemmShape<32, 32, 512>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 32x128x256_32x32x256) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<32, 128, 256>,
      cutlass::gemm::GemmShape<32, 32, 256>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 6>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 32x256x256_32x64x256) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<32, 256, 256>,
      cutlass::gemm::GemmShape<32, 64, 256>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 16x128x512_16x32x512) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<16, 128, 512>,
      cutlass::gemm::GemmShape<16, 32, 512>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 128x32x256_32x32x256) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 32, 256>,
      cutlass::gemm::GemmShape<32, 32, 256>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 6>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 128x32x512_32x32x512) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 32, 512>,
      cutlass::gemm::GemmShape<32, 32, 512>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 256x32x256_64x32x256) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 32, 256>,
      cutlass::gemm::GemmShape<64, 32, 256>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 6>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 128x16x256_32x16x256) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 16, 256>,
      cutlass::gemm::GemmShape<32, 16, 256>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 6>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 128x16x512_32x16x512) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 16, 512>,
      cutlass::gemm::GemmShape<32, 16, 512>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 6>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

TEST(SM80_Device_Sparse_Gemm_s4t_s4n_s32t_tensor_op_s32, 256x16x256_16x64x256) {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = cutlass::gemm::device::GemmSparseUniversal<
      cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::int4b_t,
      cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 16, 256>,
      cutlass::gemm::GemmShape<64, 16, 256>, cutlass::gemm::GemmShape<16, 8, 128>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 6>;

  EXPECT_TRUE(test::gemm::device::TestAllSparseGemm<Gemm>());
}

////////////////////////////////////////////////////////////////////////////////

#endif // defined(CUTLASS_ARCH_SPARSE_MMA_SM80_SUPPORTED)
