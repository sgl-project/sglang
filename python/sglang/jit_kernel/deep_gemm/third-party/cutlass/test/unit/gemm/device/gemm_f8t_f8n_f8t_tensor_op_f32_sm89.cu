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
    \brief Tests for device-wide GEMM interface with:
        A: row major, of type FE4M4 or FE5M2
        B: column major, of type FE4M3 or FE5M2
        C: row major, of FE4M3 or FE5M2
        Accum: F32
*/

#include <iostream>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"
#include "cutlass/gemm/device/gemm_universal_with_absmax.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "testbed.h"
#include "testbed_with_absmax.h"

#if defined(CUTLASS_ARCH_MMA_F32_SM89_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Gemm_fe4m3t_fe4m3n_fe4m3t_tensor_op_f32, identity_128x256x64_64x64x64) {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::float_e4m3_t;
  using ElementAuxOutput = ElementOutput;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static int const kStages = 3;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages
  >;

  bool passed = test::gemm::device::TestAllGemmWithAbsmax<Gemm, test::gemm::device::Testbed<Gemm>, cutlass::epilogue::thread::Identity>();
  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Gemm_fe4m3t_fe4m3n_fe4m3t_tensor_op_f32, identity_fastacc_128x256x64_64x64x64) {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::float_e4m3_t;
  using ElementAuxOutput = ElementOutput;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static int const kStages = 3;
  static int const kAlignment = 16;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages,
    kAlignment, kAlignment, cutlass::arch::OpMultiplyAddFastAccum
  >;

  bool passed = test::gemm::device::TestAllGemmWithAbsmax<Gemm, test::gemm::device::Testbed<Gemm>, cutlass::epilogue::thread::Identity>();
  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Gemm_fe4m3t_fe4m3n_fe4m3t_tensor_op_f32, relu_128x256x64_64x64x64) {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::float_e4m3_t;
  using ElementAuxOutput = ElementOutput;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static int const kStages = 3;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::ReLu,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages
  >;

  bool passed = test::gemm::device::TestAllGemmWithAbsmax<Gemm, test::gemm::device::Testbed<Gemm>, cutlass::epilogue::thread::ReLu>();
  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Gemm_fe4m3t_fe5m2n_fe4m3t_tensor_op_f32, identity_128x256x64_64x64x64) {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e5m2_t;
  using ElementOutput = cutlass::float_e4m3_t;
  using ElementAuxOutput = ElementOutput;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static int const kStages = 3;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages
  >;

  bool passed = test::gemm::device::TestAllGemmWithAbsmax<Gemm, test::gemm::device::Testbed<Gemm>, cutlass::epilogue::thread::Identity>();
  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Gemm_fe5m2t_fe4m3n_fe4m3t_tensor_op_f32, identity_128x256x64_64x64x64) {
  using ElementA = cutlass::float_e5m2_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::float_e4m3_t;
  using ElementAuxOutput = ElementOutput;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static int const kStages = 3;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages
  >;

  bool passed = test::gemm::device::TestAllGemmWithAbsmax<Gemm, test::gemm::device::Testbed<Gemm>, cutlass::epilogue::thread::Identity>();
  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Gemm_fe5m2t_fe5m2n_fe4m3t_tensor_op_f32, identity_128x256x64_64x64x64) {
  using ElementA = cutlass::float_e5m2_t;
  using ElementB = cutlass::float_e5m2_t;
  using ElementOutput = cutlass::float_e4m3_t;
  using ElementAuxOutput = ElementOutput;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static int const kStages = 3;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages
  >;

  bool passed = test::gemm::device::TestAllGemmWithAbsmax<Gemm, test::gemm::device::Testbed<Gemm>, cutlass::epilogue::thread::Identity>();
  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Gemm_fe4m3t_fe4m3n_fe5m2t_tensor_op_f32, identity_128x256x64_64x64x64) {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::float_e5m2_t;
  using ElementAuxOutput = ElementOutput;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static int const kStages = 3;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages
  >;

  bool passed = test::gemm::device::TestAllGemmWithAbsmax<Gemm, test::gemm::device::Testbed<Gemm>, cutlass::epilogue::thread::Identity>();
  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Gemm_fe5m2t_fe5m2n_fe5m2t_tensor_op_f32, identity_diff_aux_output_types_128x256x64_64x64x64) {
  using ElementA = cutlass::float_e5m2_t;
  using ElementB = cutlass::float_e5m2_t;
  using ElementOutput = cutlass::float_e4m3_t;
  using ElementAuxOutput = cutlass::float_e5m2_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static int const kStages = 3;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages
  >;

  bool passed = test::gemm::device::TestAllGemmWithAbsmax<Gemm, test::gemm::device::Testbed<Gemm>, cutlass::epilogue::thread::Identity>();
  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Gemm_fe4m3t_fe4m3n_fe4m3t_tensor_op_f32, identity_128x128x64_32x64x64) {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::float_e4m3_t;
  using ElementAuxOutput = ElementOutput;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static int const kStages = 3;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<32, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages
  >;

  bool passed = test::gemm::device::TestAllGemmWithAbsmax<Gemm, test::gemm::device::Testbed<Gemm>, cutlass::epilogue::thread::Identity>();
  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Gemm_fe4m3t_fe4m3n_fe4m3t_tensor_op_f32, identity_noScale_128x256x64_64x64x64) {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::float_e4m3_t;
  using ElementAuxOutput = ElementOutput;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static int const kStages = 3;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages
  >;

  bool passed = test::gemm::device::TestAllGemmWithAbsmax<Gemm, test::gemm::device::Testbed<Gemm>, cutlass::epilogue::thread::Identity>(
    /* scaleA = */false,
    /* scaleB = */false,
    /* scaleC = */false
  );
  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Gemm_fe4m3t_fe4m3n_fe4m3t_tensor_op_f32, identity_noAux_128x256x64_64x64x64) {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::float_e4m3_t;
  using ElementAuxOutput = float;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static int const kStages = 3;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages
  >;

  bool passed = test::gemm::device::TestAllGemmWithAbsmax<Gemm, test::gemm::device::Testbed<Gemm>, cutlass::epilogue::thread::Identity>();
  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////

#endif  // CUTLASS_ARCH_MMA_F32_SM89_SUPPORTED
