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
    \brief Tests for device-wide Implicit GEMM interface
*/

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"


#include "cutlass/conv/kernel/default_conv2d_group_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "conv2d_testbed.h"

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Conv2d_Group_Fprop_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32,
  SingleGroupPerCTA_128x128_64x3_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = cutlass::half_t;
  using ElementB           = cutlass::half_t;
  using ElementC           = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;
  using ThreadblockShape   = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape          = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape   = cutlass::gemm::GemmShape<16, 8, 16>;

  /// Device-level Conv2d instance
  using Conv2dGroupFpropKernel = typename cutlass::conv::kernel::DefaultConv2dGroupFprop<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::GroupMode::kSingleGroup,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dGroupFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dGroupFpropKernel>;

  /// Run group conv unit test sizes with device-level Conv2d instance
  test::conv::device::TestbedGroupConv2dProblemSizes problem_sizes(
    ThreadblockShape::kN, ThreadblockShape::kK,
    128/cutlass::sizeof_bits<ElementA>::value
  );
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<Conv2dGroupFprop>(problem_sizes.default_single_group_sizes));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Conv2d_Group_Fprop_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32,
  SingleGroupPerCTA_64x64_64x3_32x32x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = cutlass::half_t;
  using ElementB           = cutlass::half_t;
  using ElementC           = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;
  using ThreadblockShape   = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape          = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape   = cutlass::gemm::GemmShape<16, 8, 16>;

  /// Device-level Conv2d instance
  using Conv2dGroupFpropKernel = typename cutlass::conv::kernel::DefaultConv2dGroupFprop<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::GroupMode::kSingleGroup,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dGroupFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dGroupFpropKernel>;

  /// Run group conv unit test sizes with device-level Conv2d instance
  test::conv::device::TestbedGroupConv2dProblemSizes problem_sizes(
    ThreadblockShape::kN, ThreadblockShape::kK,
    128/cutlass::sizeof_bits<ElementA>::value
  );
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<Conv2dGroupFprop>(problem_sizes.default_single_group_sizes));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Conv2d_Group_Fprop_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32,
  MultipleGroupPerCTA_128x128_64x3_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = cutlass::half_t;
  using ElementB           = cutlass::half_t;
  using ElementC           = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;
  using ThreadblockShape   = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape          = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape   = cutlass::gemm::GemmShape<16, 8, 16>;

  /// Device-level Conv2d instance
  using Conv2dGroupFpropKernel = typename cutlass::conv::kernel::DefaultConv2dGroupFprop<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::GroupMode::kMultipleGroup,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dGroupFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dGroupFpropKernel>;

  /// Run group conv unit test sizes with device-level Conv2d instance
  test::conv::device::TestbedGroupConv2dProblemSizes problem_sizes(
    ThreadblockShape::kN, ThreadblockShape::kK,
    128/cutlass::sizeof_bits<ElementA>::value
  );
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<Conv2dGroupFprop>(problem_sizes.default_multiple_group_sizes));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Conv2d_Group_Fprop_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32,
  MutipleGroupPerCTA_64x64_64x3_32x32x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = cutlass::half_t;
  using ElementB           = cutlass::half_t;
  using ElementC           = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;
  using ThreadblockShape   = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape          = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape   = cutlass::gemm::GemmShape<16, 8, 16>;

  /// Device-level Conv2d instance
  using Conv2dGroupFpropKernel = typename cutlass::conv::kernel::DefaultConv2dGroupFprop<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::GroupMode::kMultipleGroup,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dGroupFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dGroupFpropKernel>;

  /// Run group conv unit test sizes with device-level Conv2d instance
  test::conv::device::TestbedGroupConv2dProblemSizes problem_sizes(
    ThreadblockShape::kN, ThreadblockShape::kK,
    128/cutlass::sizeof_bits<ElementA>::value
  );
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<Conv2dGroupFprop>(problem_sizes.default_multiple_group_sizes));
}

////////////////////////////////////////////////////////////////////////////////

// Analytic 2 stage SingleGroup kernel
TEST(SM80_Device_Conv2d_Group_Fprop_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32,
  SingleGroupPerCTA_128x128_64x2_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = cutlass::half_t;
  using ElementB           = cutlass::half_t;
  using ElementC           = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;
  using ThreadblockShape   = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape          = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape   = cutlass::gemm::GemmShape<16, 8, 16>;

  /// Device-level Conv2d instance
  using Conv2dGroupFpropKernel = typename cutlass::conv::kernel::DefaultConv2dGroupFprop<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::GroupMode::kSingleGroup,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dGroupFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dGroupFpropKernel>;

  /// Run group conv unit test sizes with device-level Conv2d instance
  test::conv::device::TestbedGroupConv2dProblemSizes problem_sizes(
    ThreadblockShape::kN, ThreadblockShape::kK,
    128/cutlass::sizeof_bits<ElementA>::value
  );
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<Conv2dGroupFprop>(problem_sizes.default_single_group_sizes));
}

////////////////////////////////////////////////////////////////////////////////

// Analytic 2 stage MutipleGroup kernel
TEST(SM80_Device_Conv2d_Group_Fprop_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32,
  MutipleGroupPerCTA_64x64_64x2_32x32x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = cutlass::half_t;
  using ElementB           = cutlass::half_t;
  using ElementC           = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;
  using ThreadblockShape   = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape          = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape   = cutlass::gemm::GemmShape<16, 8, 16>;

  /// Device-level Conv2d instance
  using Conv2dGroupFpropKernel = typename cutlass::conv::kernel::DefaultConv2dGroupFprop<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::GroupMode::kMultipleGroup,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dGroupFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dGroupFpropKernel>;

  /// Run group conv unit test sizes with device-level Conv2d instance
  test::conv::device::TestbedGroupConv2dProblemSizes problem_sizes(
    ThreadblockShape::kN, ThreadblockShape::kK,
    128/cutlass::sizeof_bits<ElementA>::value
  );
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<Conv2dGroupFprop>(problem_sizes.default_multiple_group_sizes));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Conv2d_Group_Fprop_Optimized_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32,
  SingleGroupPerCTA_128x128_64x3_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = cutlass::half_t;
  using ElementB           = cutlass::half_t;
  using ElementC           = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;
  using ThreadblockShape   = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape          = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape   = cutlass::gemm::GemmShape<16, 8, 16>;

  /// Device-level Conv2d instance
  using Conv2dGroupFpropKernel = typename cutlass::conv::kernel::DefaultConv2dGroupFprop<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::GroupMode::kSingleGroup,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dGroupFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dGroupFpropKernel>;

  /// Run group conv unit test sizes with device-level Conv2d instance
  test::conv::device::TestbedGroupConv2dProblemSizes problem_sizes(
    ThreadblockShape::kN, ThreadblockShape::kK,
    128/cutlass::sizeof_bits<ElementA>::value
  );
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<Conv2dGroupFprop>(problem_sizes.default_single_group_sizes));
}

////////////////////////////////////////////////////////////////////////////////

// Optimized multistage singleGroup kernel
TEST(SM80_Device_Conv2d_Group_Fprop_Optimized_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32,
  SingleGroupPerCTA_64x64_64x3_32x32x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = cutlass::half_t;
  using ElementB           = cutlass::half_t;
  using ElementC           = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;
  using ThreadblockShape   = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape          = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape   = cutlass::gemm::GemmShape<16, 8, 16>;

  /// Device-level Conv2d instance
  using Conv2dGroupFpropKernel = typename cutlass::conv::kernel::DefaultConv2dGroupFprop<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::GroupMode::kSingleGroup,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dGroupFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dGroupFpropKernel>;

  /// Run group conv unit test sizes with device-level Conv2d instance
  test::conv::device::TestbedGroupConv2dProblemSizes problem_sizes(
    ThreadblockShape::kN, ThreadblockShape::kK,
    128/cutlass::sizeof_bits<ElementA>::value
  );
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<Conv2dGroupFprop>(problem_sizes.default_single_group_sizes));
}

////////////////////////////////////////////////////////////////////////////////

// Optimized 2 stage SingleGroup kernel
TEST(SM80_Device_Conv2d_Group_Fprop_Optimized_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32,
  SingleGroupPerCTA_64x64_64x2_32x32x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = cutlass::half_t;
  using ElementB           = cutlass::half_t;
  using ElementC           = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;
  using ThreadblockShape   = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape          = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape   = cutlass::gemm::GemmShape<16, 8, 16>;

  /// Device-level Conv2d instance
  using Conv2dGroupFpropKernel = typename cutlass::conv::kernel::DefaultConv2dGroupFprop<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::GroupMode::kSingleGroup,
    cutlass::conv::IteratorAlgorithm::kOptimized
  >::Kernel;

  using Conv2dGroupFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dGroupFpropKernel>;

  /// Run group conv unit test sizes with device-level Conv2d instance
  test::conv::device::TestbedGroupConv2dProblemSizes problem_sizes(
    ThreadblockShape::kN, ThreadblockShape::kK,
    128/cutlass::sizeof_bits<ElementA>::value
  );
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<Conv2dGroupFprop>(problem_sizes.default_single_group_sizes));
}

////////////////////////////////////////////////////////////////////////////////

#endif  // CUTLASS_ARCH_MMA_SM80_SUPPORTED

////////////////////////////////////////////////////////////////////////////////
