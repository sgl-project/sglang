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
    \brief Tests for device-wide Conv2d fprop interface with:
        A: NHWC, of type FE4M4 or FE5M2
        B: NHWC, of type FE4M3 or FE5M2
        C: NHWC, of FE4M3 or FE5M2
        Accum: F32
*/

#include <iostream>

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"
#include "cutlass/conv/kernel/default_conv2d_fprop_with_absmax.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/util/tensor_view_io.h"

#include "conv2d_with_absmax_testbed.h"

#if defined(CUTLASS_ARCH_MMA_F32_SM89_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Conv2d_Fprop_Analytic_ImplicitGemm_fe4m3nhwc_fe4mnhwc_fe4mnhwc_tensor_op_f32,
  identity_128x256x64_64x3_64x64x64) {

  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::float_e4m3_t;
  using ElementAuxOutput = ElementOutput;
  using ElementAccumulator = float;
  static int const kStages = 3;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithAbsMax<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementOutput, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  bool passed = test::conv::device::TestAllConv2dWithAbsmax<Conv2dFprop, cutlass::epilogue::thread::Identity>();
  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Conv2d_Fprop_Analytic_ImplicitGemm_fe5m2nhwc_fe4m3nhwc_fe4m3nhwc_tensor_op_f32,
  identity_128x256x64_64x3_64x64x64) {

  using ElementA = cutlass::float_e5m2_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::float_e4m3_t;
  using ElementAuxOutput = ElementOutput;
  using ElementAccumulator = float;
  static int const kStages = 3;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithAbsMax<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementOutput, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  bool passed = test::conv::device::TestAllConv2dWithAbsmax<Conv2dFprop, cutlass::epilogue::thread::Identity>();
  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Conv2d_Fprop_Analytic_ImplicitGemm_fe5m2nhwc_fe4m3nhwc_fe5m2nhwc_tensor_op_f32,
  identity_128x256x64_64x3_64x64x64) {

  using ElementA = cutlass::float_e5m2_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::float_e5m2_t;
  using ElementAuxOutput = ElementOutput;
  using ElementAccumulator = float;
  static int const kStages = 3;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithAbsMax<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementOutput, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  bool passed = test::conv::device::TestAllConv2dWithAbsmax<Conv2dFprop, cutlass::epilogue::thread::Identity>();
  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Conv2d_Fprop_Optimized_ImplicitGemm_fe4m3nhwc_fe4mnhwc_fe4mnhwc_tensor_op_f32,
  identity_128x256x64_64x3_64x64x64) {

  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::float_e4m3_t;
  using ElementAuxOutput = ElementOutput;
  using ElementAccumulator = float;
  static int const kStages = 3;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithAbsMax<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementOutput, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  bool passed = test::conv::device::TestAllConv2dWithAbsmax<Conv2dFprop, cutlass::epilogue::thread::Identity>();
  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Conv2d_Fprop_Optimized_ImplicitGemm_fe4m3nhwc_fe4mnhwc_fe4mnhwc_tensor_op_f32,
  relu_128x256x64_64x3_64x64x64) {

  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::float_e4m3_t;
  using ElementAuxOutput = ElementOutput;
  using ElementAccumulator = float;
  static int const kStages = 3;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::ReLu,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithAbsMax<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementOutput, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  bool passed = test::conv::device::TestAllConv2dWithAbsmax<Conv2dFprop, cutlass::epilogue::thread::ReLu>();
  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Conv2d_Fprop_Optimized_ImplicitGemm_fe4m3nhwc_fe4mnhwc_fe4mnhwc_tensor_op_f32,
  identity_fastacc_128x256x64_64x3_64x64x64) {

  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::float_e4m3_t;
  using ElementAuxOutput = ElementOutput;
  using ElementAccumulator = float;
  static int const kStages = 3;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithAbsMax<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementOutput, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    cutlass::arch::OpMultiplyAddFastAccum,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  bool passed = test::conv::device::TestAllConv2dWithAbsmax<Conv2dFprop, cutlass::epilogue::thread::Identity>();
  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM89_Device_Conv2d_Fprop_Optimized_ImplicitGemm_fe4m3nhwc_fe4mnhwc_fe4mnhwc_tensor_op_f32,
  identity_noScale_128x256x64_64x3_64x64x64) {

  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::float_e4m3_t;
  using ElementAuxOutput = ElementOutput;
  using ElementAccumulator = float;
  static int const kStages = 3;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementAuxOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithAbsMax<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementOutput, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  bool passed = test::conv::device::TestAllConv2dWithAbsmax<Conv2dFprop, cutlass::epilogue::thread::Identity>(
    /* scaleA = */false,
    /* scaleB = */false,
    /* scaleC = */false
  );
  EXPECT_TRUE(passed);
}

////////////////////////////////////////////////////////////////////////////////

#endif  // CUTLASS_ARCH_MMA_F32_SM89_SUPPORTED
