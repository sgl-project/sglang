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
    \brief Tests for device-wide Depthwise Direct Conv interface
*/

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"


#include "cutlass/conv/kernel/default_depthwise_fprop.h"
#include "cutlass/conv/device/direct_convolution.h"

#include "conv2d_testbed.h"
#include "depthwise_conv2d_direct_conv_testbed.h"

std::vector<cutlass::conv::Conv2dProblemSize> DepthwiseFpropProblemSizes_filter3x3() {
  std::vector<cutlass::conv::Conv2dProblemSize> problems;

  for (int channels = 16; channels <= 512; channels += 16) {
    problems.push_back(cutlass::conv::Conv2dProblemSize(
        {1, 8, 8, channels},                     // input size  (NHWC)
        {channels, 3, 3, 1},                     // filter size (KRSC)
        {1, 1, 1, 1},                            // padding (pad_h, _, pad_w, _)
        {1, 1},                                  // stride (stride_h, stride_w)
        {1, 1},                                  // dilation (dilation_h, dilation_w)
        cutlass::conv::Mode::kCrossCorrelation,  // Convolution mode
        16,                                      // split_k_slices
        channels                                 // groups
        ));

    // if(channels == 512 || channels == 16*14)

    problems.push_back(cutlass::conv::Conv2dProblemSize(
        {1, 16, 16, channels},                   // input size  (NHWC)
        {channels, 3, 3, 1},                     // filter size (KRSC)
        {1, 1, 1, 1},                            // padding (pad_h, _, pad_w, _)
        {2, 2},                                  // stride (stride_h, stride_w)
        {2, 2},                                  // dilation (dilation_h, dilation_w)
        cutlass::conv::Mode::kCrossCorrelation,  // Convolution mode
        16,                                      // split_k_slices
        channels                                 // groups
        ));
  }

  return problems;
}

std::vector<cutlass::conv::Conv2dProblemSize> DepthwiseFpropProblemSizes_filter5x5() {
  std::vector<cutlass::conv::Conv2dProblemSize> problems;

  for (int channels = 16; channels < 256; channels += 16) {
    problems.push_back(cutlass::conv::Conv2dProblemSize(
        {1, 16, 16, channels},                   // input size  (NHWC)
        {channels, 5, 5, 1},                     // filter size (KRSC)
        {1, 1, 1, 1},                            // padding (pad_h, _, pad_w, _)
        {1, 1},                                  // stride (stride_h, stride_w)
        {1, 1},                                  // dilation (dilation_h, dilation_w)
        cutlass::conv::Mode::kCrossCorrelation,  // Convolution mode
        16,                                      // split_k_slices
        channels                                 // groups
        ));

    problems.push_back(cutlass::conv::Conv2dProblemSize(
        {1, 112, 112, channels},                 // input size  (NHWC)
        {channels, 5, 5, 1},                     // filter size (KRSC)
        {1, 1, 1, 1},                            // padding (pad_h, _, pad_w, _)
        {1, 1},                                  // stride (stride_h, stride_w)
        {1, 1},                                  // dilation (dilation_h, dilation_w)
        cutlass::conv::Mode::kCrossCorrelation,  // Convolution mode
        16,                                      // split_k_slices
        channels                                 // groups
        ));

    problems.push_back(cutlass::conv::Conv2dProblemSize(
        {1, 112, 112, channels},                 // input size  (NHWC)
        {channels, 5, 5, 1},                     // filter size (KRSC)
        {1, 1, 1, 1},                            // padding (pad_h, _, pad_w, _)
        {2, 2},                                  // stride (stride_h, stride_w)
        {2, 2},                                  // dilation (dilation_h, dilation_w)
        cutlass::conv::Mode::kCrossCorrelation,  // Convolution mode
        16,                                      // split_k_slices
        channels                                 // groups
        ));
  }

  return problems;
}

std::vector<cutlass::conv::Conv2dProblemSize> DepthwiseFpropProblemSizes_filter5x37() {
  std::vector<cutlass::conv::Conv2dProblemSize> problems;

  for (int channels = 16; channels < 256; channels += 16) {
    problems.push_back(cutlass::conv::Conv2dProblemSize(
        {1, 128, 128, channels},                 // input size  (NHWC)
        {channels, 5, 37, 1},                    // filter size (KRSC)
        {1, 1, 1, 1},                            // padding (pad_h, _, pad_w, _)
        {1, 1},                                  // stride (stride_h, stride_w)
        {1, 1},                                  // dilation (dilation_h, dilation_w)
        cutlass::conv::Mode::kCrossCorrelation,  // Convolution mode
        108,                                     // split_k_slices
        channels                                 // groups
        ));
  }

  return problems;
}

////////////////////////////////////////////////////////////////////////////////
TEST(
    SM60_Device_Depthwise_conv2d_Fprop_Direct_Conv_Optimized_f16nhwc_f16nhwc_f16nhwc_simt_f16,
    64x32_4_8x32_3x3) {

  using ElementInputA = cutlass::half_t;
  using ElementInputB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementComputeEpilogue = cutlass::half_t;

  using LayoutInputA = cutlass::layout::TensorNHWC;
  using LayoutInputB = cutlass::layout::TensorNHWC;
  using LayoutOutput = cutlass::layout::TensorNHWC;

  // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU
  // SM
  using MMAOp = cutlass::arch::OpClassSimt;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm60;

  // This code section describes the groups a thread block will compute
  constexpr int groups_per_cta = 32;

  // This code section describes the output tile <N, P, Q, C> a thread block will compute
  using ThreadBlockOutputShape = cutlass::conv::TensorNHWCShape<1, 8, 8, groups_per_cta>;

  // This code section describes the filter shape <R, S>
  using FilterShape = cutlass::MatrixShape<3, 3>;

  // Threadblock tile shape
  using ThreadblockShape =
      cutlass::gemm::GemmShape<ThreadBlockOutputShape::kNHW, groups_per_cta, FilterShape::kCount>;

  // This code section describes tile size a warp will computes
  using WarpShape = cutlass::gemm::GemmShape<8, groups_per_cta, FilterShape::kCount>;

  // This code section describes the size of MMA op
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::conv::threadblock::DepthwiseDirect2dConvIdentityThreadblockSwizzle<
          1,
          ThreadBlockOutputShape::kN,
          ThreadBlockOutputShape::kH,
          ThreadBlockOutputShape::kW>;

  // Number of pipelines you want to use
  constexpr int NumStages = 4;

  // This code section describe iterator algorithm selected is Analytic or Optimized
  static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
      cutlass::conv::IteratorAlgorithm::kOptimized;

  constexpr int kEpilogueElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  // This code section describes the epilogue part of the kernel, we use default value
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,               // Data type of output matrix.
      kEpilogueElementsPerAccess,  // The number of elements per vectorized.
                                   // memory access. This becomes the vector width of
                                   // math instructions in the epilogue too.
      ElementAccumulator,          // Data type of accumulator
      ElementComputeEpilogue,      // Data type for alpha/beta in linear combination
      cutlass::epilogue::thread::ScaleType::Default>;     

  using DepthwiseDirect2dConv = typename cutlass::conv::kernel::DefaultDepthwiseDirect2dConvFprop<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      MMAOp,
      SmArch,
      ThreadblockShape,
      ThreadBlockOutputShape,
      FilterShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      SwizzleThreadBlock,
      NumStages,
      cutlass::arch::OpMultiplyAdd,
      IteratorAlgorithm,
      cutlass::conv::StrideSupport::kStrided>::Kernel;

  using Direct2dConv = cutlass::conv::device::DirectConvolution<DepthwiseDirect2dConv>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestSpecificDepthwiseDirectConv2d<Direct2dConv>(
      DepthwiseFpropProblemSizes_filter3x3()));
}

////////////////////////////////////////////////////////////////////////////////
TEST(
    SM60_Device_Depthwise_conv2d_Fprop_Direct_Conv_Optimized_f16nhwc_f16nhwc_f16nhwc_simt_f16,
    64x64_3_16x64_5x5) {

  using ElementInputA = cutlass::half_t;
  using ElementInputB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementComputeEpilogue = cutlass::half_t;

  using LayoutInputA = cutlass::layout::TensorNHWC;
  using LayoutInputB = cutlass::layout::TensorNHWC;
  using LayoutOutput = cutlass::layout::TensorNHWC;

  // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU
  // SM
  using MMAOp = cutlass::arch::OpClassSimt;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm60;

  // This code section describes the groups a thread block will compute
  constexpr int groups_per_cta = 64;

  // This code section describes the output tile <N, P, Q, C> a thread block will compute
  using ThreadBlockOutputShape = cutlass::conv::TensorNHWCShape<1, 8, 8, groups_per_cta>;

  // This code section describes the filter shape <R, S>
  using FilterShape = cutlass::MatrixShape<5, 5>;

  // Threadblock tile shape
  using ThreadblockShape =
      cutlass::gemm::GemmShape<ThreadBlockOutputShape::kNHW, groups_per_cta, FilterShape::kCount>;

  // This code section describes tile size a warp will computes
  using WarpShape = cutlass::gemm::GemmShape<16, groups_per_cta, FilterShape::kCount>;

  // This code section describes the size of MMA op
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::conv::threadblock::DepthwiseDirect2dConvIdentityThreadblockSwizzle<
          1,
          ThreadBlockOutputShape::kN,
          ThreadBlockOutputShape::kH,
          ThreadBlockOutputShape::kW>;

  // Number of pipelines you want to use
  constexpr int NumStages = 3;

  // This code section describe iterator algorithm selected is Analytic or Optimized
  static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
      cutlass::conv::IteratorAlgorithm::kOptimized;

  constexpr int kEpilogueElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  // This code section describes the epilogue part of the kernel, we use default value
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,               // Data type of output matrix.
      kEpilogueElementsPerAccess,  // The number of elements per vectorized.
                                   // memory access. This becomes the vector width of
                                   // math instructions in the epilogue too.
      ElementAccumulator,          // Data type of accumulator
      ElementComputeEpilogue,      // Data type for alpha/beta in linear combination
      cutlass::epilogue::thread::ScaleType::Default>;  

  using DepthwiseDirect2dConv = typename cutlass::conv::kernel::DefaultDepthwiseDirect2dConvFprop<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      MMAOp,
      SmArch,
      ThreadblockShape,
      ThreadBlockOutputShape,
      FilterShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      SwizzleThreadBlock,
      NumStages,
      cutlass::arch::OpMultiplyAdd,
      IteratorAlgorithm,
      cutlass::conv::StrideSupport::kStrided>::Kernel;

  using Direct2dConv = cutlass::conv::device::DirectConvolution<DepthwiseDirect2dConv>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestSpecificDepthwiseDirectConv2d<Direct2dConv>(
      DepthwiseFpropProblemSizes_filter5x5()));
}

#if 0
////////////////////////////////////////////////////////////////////////////////
TEST(
    SM60_Device_Depthwise_conv2d_Fprop_Direct_Conv_Optimized_f16nhwc_f16nhwc_f16nhwc_simt_f16,
    64x32_3_16x32_5x37) {

  using ElementInputA = cutlass::half_t;
  using ElementInputB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementComputeEpilogue = cutlass::half_t;

  using LayoutInputA = cutlass::layout::TensorNHWC;
  using LayoutInputB = cutlass::layout::TensorNHWC;
  using LayoutOutput = cutlass::layout::TensorNHWC;

  // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU
  // SM
  using MMAOp = cutlass::arch::OpClassSimt;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm60;

  // This code section describes the groups a thread block will compute
  constexpr int groups_per_cta = 32;

  // This code section describes the output tile <N, P, Q, C> a thread block will compute
  using ThreadBlockOutputShape = cutlass::conv::TensorNHWCShape<1, 8, 8, groups_per_cta>;

  // This code section describes the filter shape <R, S>
  using FilterShape = cutlass::MatrixShape<5, 37>;

  // Threadblock tile shape
  using ThreadblockShape =
      cutlass::gemm::GemmShape<ThreadBlockOutputShape::kNHW, groups_per_cta, FilterShape::kCount>;

  // This code section describes tile size a warp will computes
  using WarpShape = cutlass::gemm::GemmShape<16, groups_per_cta, FilterShape::kCount>;

  // This code section describes the size of MMA op
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::conv::threadblock::DepthwiseDirect2dConvIdentityThreadblockSwizzle<
          1,
          ThreadBlockOutputShape::kN,
          ThreadBlockOutputShape::kH,
          ThreadBlockOutputShape::kW>;

  // Number of pipelines you want to use
  constexpr int NumStages = 2;

  // This code section describe iterator algorithm selected is Analytic or Optimized
  static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
      cutlass::conv::IteratorAlgorithm::kOptimized;

  constexpr int kEpilogueElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  // This code section describes the epilogue part of the kernel, we use default value
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,               // Data type of output matrix.
      kEpilogueElementsPerAccess,  // The number of elements per vectorized.
                                   // memory access. This becomes the vector width of
                                   // math instructions in the epilogue too.
      ElementAccumulator,          // Data type of accumulator
      ElementComputeEpilogue,      // Data type for alpha/beta in linear combination
      cutlass::epilogue::thread::ScaleType::Default>;  

  using DepthwiseDirect2dConv = typename cutlass::conv::kernel::DefaultDepthwiseDirect2dConvFprop<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      MMAOp,
      SmArch,
      ThreadblockShape,
      ThreadBlockOutputShape,
      FilterShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      SwizzleThreadBlock,
      NumStages,
      cutlass::arch::OpMultiplyAdd,
      IteratorAlgorithm,
      cutlass::conv::StrideSupport::kStrided>::Kernel;

  using Direct2dConv = cutlass::conv::device::DirectConvolution<DepthwiseDirect2dConv>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestSpecificDepthwiseDirectConv2d<Direct2dConv>(
      DepthwiseFpropProblemSizes_filter5x37()));
}
#endif

