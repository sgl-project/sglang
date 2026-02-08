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


#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "conv2d_testbed.h"

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

std::vector<cutlass::conv::Conv2dProblemSize> Conv2dFixedChannelProblemSizes(int channels) {

  std::vector<cutlass::conv::Conv2dProblemSize> problems;

  problems.push_back(cutlass::conv::Conv2dProblemSize(
    {1, 8, 8, channels},   // input size  (NHWC)
    {16, 3, 3, channels},   // filter size (KRSC)
    {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
    {2, 2},                            // stride (stride_h, stride_w)
    {1, 1}                             // dilation (dilation_h, dilation_w)
  ));

  problems.push_back(cutlass::conv::Conv2dProblemSize(
    {1, 224, 224, channels},   // input size  (NHWC)
    {32, 7, 7, channels},   // filter size (KRSC)
    {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
    {1, 1},                            // stride (stride_h, stride_w)
    {1, 1}                             // dilation (dilation_h, dilation_w)
  ));

  problems.push_back(cutlass::conv::Conv2dProblemSize(
    {1, 224, 224, channels},   // input size  (NHWC)
    {64, 7, 7, channels},   // filter size (KRSC)
    {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
    {2, 2},                            // stride (stride_h, stride_w)
    {1, 1}                             // dilation (dilation_h, dilation_w)
  ));

  problems.push_back(cutlass::conv::Conv2dProblemSize(
    {1, 224, 224, channels},   // input size  (NHWC)
    {64, 5, 5, channels},   // filter size (KRSC)
    {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
    {1, 1},                            // stride (stride_h, stride_w)
    {1, 1}                             // dilation (dilation_h, dilation_w)
  ));

  problems.push_back(cutlass::conv::Conv2dProblemSize(
    {1, 224, 224, channels},   // input size  (NHWC)
    {64, 5, 5, channels},   // filter size (KRSC)
    {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
    {2, 2},                            // stride (stride_h, stride_w)
    {1, 1}                             // dilation (dilation_h, dilation_w)
  ));

  return problems;
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Conv2d_Fprop_Fixed_Channels_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_channels_8,
  128x128_64x3_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = cutlass::half_t;
  using ElementB           = cutlass::half_t;
  using ElementC           = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  int const kChannelCount = 8;

  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kFixedChannels,
    cutlass::conv::StrideSupport::kStrided,
    kChannelCount,
    kChannelCount
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dFprop>(
    Conv2dFixedChannelProblemSizes(kChannelCount)));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Conv2d_Fprop_Fixed_Channels_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_channels_4,
  128x128_64x3_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = cutlass::half_t;
  using ElementB           = cutlass::half_t;
  using ElementC           = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  int const kChannelCount = 4;

  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kFixedChannels,
    cutlass::conv::StrideSupport::kStrided,
    kChannelCount,
    kChannelCount
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dFprop>(
    Conv2dFixedChannelProblemSizes(kChannelCount)));
}

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Conv2d_Fprop_Fixed_Channels_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_channels_2,
  128x128_64x3_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = cutlass::half_t;
  using ElementB           = cutlass::half_t;
  using ElementC           = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  int const kChannelCount = 2;

  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, cutlass::layout::TensorNHWC,
    ElementB, cutlass::layout::TensorNHWC,
    ElementC, cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kFixedChannels,
    cutlass::conv::StrideSupport::kStrided,
    kChannelCount,
    kChannelCount
  >::Kernel;

  using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dFprop>(
    Conv2dFixedChannelProblemSizes(kChannelCount)));
}

////////////////////////////////////////////////////////////////////////////////

#endif  // CUTLASS_ARCH_MMA_SM80_SUPPORTED

////////////////////////////////////////////////////////////////////////////////
