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
    \brief Tests for Depthwise Direct Conv interface
*/

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"


#include "cutlass/conv/kernel/default_depthwise_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "conv2d_testbed.h"


std::vector<cutlass::conv::Conv2dProblemSize> DepthwiseFpropProblemSizes() {

std::vector<cutlass::conv::Conv2dProblemSize> problems;

for ( int channels = 16; channels < 256 ; channels+=16){
  problems.push_back(cutlass::conv::Conv2dProblemSize(
    {1, 8, 8, channels},                // input size  (NHWC)
    {channels, 3, 3, 1},                // filter size (KRSC)
    {1, 1, 1, 1},                       // padding (pad_h, _, pad_w, _)
    {2, 2},                             // stride (stride_h, stride_w)
    {1, 1},                             // dilation (dilation_h, dilation_w)
    cutlass::conv::Mode::kCrossCorrelation,  // Convolution mode
    1,                                  // split_k_slices
    channels                            // groups
  ));

  problems.push_back(cutlass::conv::Conv2dProblemSize(
    {1, 16, 16, channels},             // input size  (NHWC)
    {channels, 3, 3, 1},              // filter size (KRSC)
    {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
    {2, 2},                            // stride (stride_h, stride_w)
    {2, 2},                             // dilation (dilation_h, dilation_w)
    cutlass::conv::Mode::kCrossCorrelation,  // Convolution mode
    1,                                  // split_k_slices
    channels                            // groups
  ));

  problems.push_back(cutlass::conv::Conv2dProblemSize(
    {1, 16, 16, channels},             // input size  (NHWC)
    {channels, 7, 7, 1},              // filter size (KRSC)
    {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
    {1, 1},                            // stride (stride_h, stride_w)
    {1, 1},                             // dilation (dilation_h, dilation_w)
    cutlass::conv::Mode::kCrossCorrelation,  // Convolution mode
    1,                                  // split_k_slices
    channels                            // groups
  ));

  problems.push_back(cutlass::conv::Conv2dProblemSize(
    {1, 112, 112, channels},   // input size  (NHWC)
    {channels, 7, 7, 1},   // filter size (KRSC)
    {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
    {1, 1},                            // stride (stride_h, stride_w)
    {1, 1},                             // dilation (dilation_h, dilation_w)
    cutlass::conv::Mode::kCrossCorrelation,  // Convolution mode
    1,                                  // split_k_slices
    channels                            // groups
  ));

  problems.push_back(cutlass::conv::Conv2dProblemSize(
    {1, 112, 112, channels},   // input size  (NHWC)
    {channels, 7, 7, 1},   // filter size (KRSC)
    {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
    {2, 2},                            // stride (stride_h, stride_w)
    {2, 2} ,                             // dilation (dilation_h, dilation_w)
    cutlass::conv::Mode::kCrossCorrelation,  // Convolution mode
    1,                                  // split_k_slices
    channels                            // groups
  ));

  problems.push_back(cutlass::conv::Conv2dProblemSize(
    {1, 112, 112, channels},   // input size  (NHWC)
    {channels, 5, 5, 1},   // filter size (KRSC)
    {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
    {1, 1},                            // stride (stride_h, stride_w)
    {1, 1},                             // dilation (dilation_h, dilation_w)
    cutlass::conv::Mode::kCrossCorrelation,  // Convolution mode
    1,                                  // split_k_slices
    channels                            // groups
  ));

  problems.push_back(cutlass::conv::Conv2dProblemSize(
    {1, 112, 112, channels},   // input size  (NHWC)
    {channels, 5, 5, 1},   // filter size (KRSC)
    {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
    {2, 2},                            // stride (stride_h, stride_w)
    {2, 2} ,                             // dilation (dilation_h, dilation_w)
    cutlass::conv::Mode::kCrossCorrelation,  // Convolution mode
    1,                                  // split_k_slices
    channels                            // groups
  ));
}

return problems;
}

////////////////////////////////////////////////////////////////////////////////
TEST(SM60_Device_Depthwise_Fprop_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_simt_f16,
  128x128_8x2_64x64x8) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = cutlass::half_t;
  using ElementB           = cutlass::half_t;
  using ElementC           = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute     = cutlass::half_t;


  /// Device-level depthwiseFpropKernel instance
  using depthwiseFpropKernel = typename cutlass::conv::kernel::DefaultDepthwiseFprop<
    ElementA, 
    cutlass::layout::TensorNHWC,
    ElementB, 
    cutlass::layout::TensorNHWC,
    ElementC, 
    cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm60,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<64, 64, 8>, 
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using DepthwiseFprop = cutlass::conv::device::ImplicitGemmConvolution<depthwiseFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<DepthwiseFprop>(
    DepthwiseFpropProblemSizes()));

}

////////////////////////////////////////////////////////////////////////////////
TEST(SM60_Device_Depthwise_Fprop_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_simt_f16,
  64x64_8x2_32x32x8) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = cutlass::half_t;
  using ElementB           = cutlass::half_t;
  using ElementC           = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute     = cutlass::half_t;


  /// Device-level depthwiseFpropKernel instance
  using depthwiseFpropKernel = typename cutlass::conv::kernel::DefaultDepthwiseFprop<
    ElementA, 
    cutlass::layout::TensorNHWC,
    ElementB, 
    cutlass::layout::TensorNHWC,
    ElementC, 
    cutlass::layout::TensorNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm60,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>, 
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
      ElementC,
      1,
      ElementAccumulator,
      ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using DepthwiseFprop = cutlass::conv::device::ImplicitGemmConvolution<depthwiseFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<DepthwiseFprop>(
    DepthwiseFpropProblemSizes()));

}
