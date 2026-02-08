/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief CUTLASS 3.x Implicit GEMM testbed sizes for ConvNd problem
*/
#pragma once

#include "cutlass/conv/convnd_problem_shape.hpp"
#include <vector>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test::conv::device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<int SpatialDim, cutlass::conv::Operator ConvOp, bool SupportStrides = (ConvOp != cutlass::conv::Operator::kDgrad)>
std::vector<cutlass::conv::ConvProblemShape<ConvOp, SpatialDim>>
inline
get_conv_problem_vector();

/////////////////////////////////////////////////////////////////////////////////////////////////
// Fprop
/////////////////////////////////////////////////////////////////////////////////////////////////

// Specialization for 1D fprop problems
template<>
std::vector<cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kFprop, 1>> inline
get_conv_problem_vector<1, cutlass::conv::Operator::kFprop>() {
  using ProblemShape = cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kFprop, 1>;
  std::vector<ProblemShape> problem_shapes;
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,  8, 64},  // nwc
    {64, 1, 64},  // ksc
    {0},          // padding lower (pad_w)
    {0},          // padding upper (pad_w)
    {1},          // stride (stride_w)
    {1},          // dilation (dilation_w)
    1             // group
  });
  // non-packed input strides.
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,   8,  64},  // nwc
    {800, 80, 1},   // stride (nwc)
    {64,  1,  64},  // ksc
    {64,  64, 1},   // stride (ksc)
    {0},            // padding lower (pad_w)
    {0},            // padding upper (pad_w)
    {1},            // stride (stride_w)
    {1},            // dilation (dilation_w)
    1               // group
  });
  // non-packed output strides.
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,   8,  64},  // nwc
    {512, 64, 1},   // stride (nwc)
    {64,  1,  64},  // ksc
    {64,  64, 1},   // stride (ksc)
    {800, 80, 1},   // stride (nqk)
    {0},            // padding lower (pad_w)
    {0},            // padding upper (pad_w)
    {1},            // stride (stride_w)
    {1},            // dilation (dilation_w)
    1               // group
  });
  // Filter-K = 16 for predication
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1, 8, 64},
    {16,1, 64},
    {0},
    {0},
    {1},
    {1},
    1
  });
  // N = 2 and K = 128 for a larger grid
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  8, 64},
    {96, 1, 64},
    {0},
    {0},
    {1},
    {1},
    1
  });
  // N = 7 and K = 256 for a even larger grid
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {7,   8, 64},
    {256, 1, 64},
    {0},
    {0},
    {1},
    {1},
    1
  });
  // 3 filter, no padding
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   8, 64},
    {256, 3, 64},
    {0},
    {0},
    {1},
    {1},
    1
  });
  // 3 filter, symmetric padding with c % cta_k !=0
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   8, 32},
    {256, 3, 32},
    {1},
    {1},
    {1},
    {1},
    1
  });
  // 4 filter, asymmetric padding
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   8, 64},
    {256, 4, 64},
    {0},
    {1},
    {1},
    {1},
    1
  });
  // 3 filter, asymmetric padding and tstride of 2
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   8, 64},
    {256, 3, 64},
    {0},
    {1},
    {2},
    {1},
    1
  });
  // 3 filter, asymmetric padding and dilation of 2
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   8, 64},
    {256, 3, 64},
    {0},
    {1},
    {1},
    {2},
    1
  });
  return problem_shapes;
}

// Specialization for 2D fprop problems
template<>
std::vector<cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kFprop, 2>> inline
get_conv_problem_vector<2, cutlass::conv::Operator::kFprop>() {
  using ProblemShape = cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kFprop, 2>;
  std::vector<ProblemShape> problem_shapes;
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,  8, 8, 64},  // nhwc
    {64, 1, 1, 64},  // krsc
    {0, 0},          // padding lower (pad_h, pad_w)
    {0, 0},          // padding upper (pad_h, pad_w)
    {1, 1},          // stride (stride_h, stride_w)
    {1, 1},          // dilation (dilation_h, dilation_w)
    1                // group
  });
  // non-packed input strides.
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,    8,   8,  64},  // nhwc
    {8000, 800, 80, 1},   // stride (nhwc)
    {64,   1,   1,  64},  // krsc
    {64,   64,  64, 1},   // stride (krsc)
    {0, 0},               // padding lower (pad_h, pad_w)
    {0, 0},               // padding upper (pad_h, pad_w)
    {1, 1},               // stride (stride_h, stride_w)
    {1, 1},               // dilation (dilation_h, dilation_w)
    1                     // group
  });
  // non-packed output strides.
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,    8,   8,  64},  // nhwc
    {4096, 512, 64, 1},   // stride (nhwc)
    {64,   1,   1,  64},  // krsc
    {64,   64,  64, 1},   // stride (krsc)
    {8000, 800, 80, 1},   // stride (npqk)
    {0, 0},               // padding lower (pad_h, pad_w)
    {0, 0},               // padding upper (pad_h, pad_w)
    {1, 1},               // stride (stride_h, stride_w)
    {1, 1},               // dilation (dilation_h, dilation_w)
    1                     // group
  });
  // Filter-K = 16 for predication
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,  8, 8, 64},
    {16, 1, 1, 64},
    {0, 0},
    {0, 0},
    {1, 1},
    {1, 1},
    1
  });
  // N = 2 and K = 128 for a larger grid
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  8, 8, 64},
    {96, 1, 1, 64},
    {0, 0},
    {0, 0},
    {1, 1},
    {1, 1},
    1
  });
  // N = 7 and K = 256 for a even larger grid
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {7,   8, 8, 64},
    {256, 1, 1, 64},
    {0, 0},
    {0, 0},
    {1, 1},
    {1, 1},
    1
  });
  // 3x3 filter, no padding
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   8, 8, 64},
    {256, 3, 3, 64},
    {0, 0},
    {0, 0},
    {1, 1},
    {1, 1},
    1
  });
  // 3x3 filter, symmetric padding with c % cta_k !=0
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   8, 8, 32},
    {256, 3, 3, 32},
    {1, 1},
    {1, 1},
    {1, 1},
    {1, 1},
    1
  });
  // 2x5 filter, asymmetric padding 1,2/1,2
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   8, 8, 64},
    {256, 2, 5, 64},
    {1, 1},
    {2, 2},
    {1, 1},
    {1, 1},
    1
  });
  // 2x5 filter, asymmetric padding 1,0/1,0, w/ stride
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   7, 7, 64},
    {256, 2, 5, 64},
    {1, 1},
    {0, 0},
    {2, 3},
    {1, 1},
    1
  });
  // 2x5 filter, asymmetric padding 1,0/1,0, w/ dilation
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   16, 16, 64},
    {256, 2,  5,  64},
    {1, 1},
    {0, 0},
    {1, 1},
    {2, 3},
    1
  });
  // 2x5 filter, asymmetric padding 1,0/1,0, w/ stride, w/ dilation
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   16, 15, 64},
    {256, 2,  5,  64},
    {1, 1},
    {0, 0},
    {2, 3},
    {2, 3},
    1
  });
  return problem_shapes;
}

// Specialization for 3D fprop problems
template<>
std::vector<cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kFprop, 3>> inline
get_conv_problem_vector<3, cutlass::conv::Operator::kFprop>() {
  using ProblemShape = cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kFprop, 3>;
  std::vector<ProblemShape> problem_shapes;
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,  1, 8, 8, 64},  // ndhwc
    {64, 1, 1, 1, 64},  // ktrsc
    {0, 0, 0},          // padding lower (pad_d, pad_h, pad_w)
    {0, 0, 0},          // padding upper (pad_d, pad_h, pad_w)
    {1, 1, 1},          // stride (stride_d, stride_h, stride_w)
    {1, 1, 1},          // dilation (dilation_d, dilation_h, dilation_w)
    1                   // group
  });
  // non-packed input output strides.
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,    1,    8,   8,  64},  // ndhwc
    {8000, 8000, 800, 80, 1},   // stride (ndhwc)
    {64,   1,    1,   1,  64},  // ktrsc
    {64,   64,   64,  64, 1},   // stride (ktrsc)
    {8000, 8000, 800, 80, 1},   // stride (nzpqk)
    {0, 0, 0},                  // padding lower (pad_d, pad_h, pad_w)
    {0, 0, 0},                  // padding upper (pad_d, pad_h, pad_w)
    {1, 1, 1},                  // stride (stride_d, stride_h, stride_w)
    {1, 1, 1},                  // dilation (dilation_d, dilation_h, dilation_w)
    1                           // group
  });
  // Filter-K = 16 for predication
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,  1, 8, 8, 64},
    {16, 1, 1, 1, 64},
    {0, 0, 0},
    {0, 0, 0},
    {1, 1, 1},
    {1, 1, 1},
    1
  });
  // N = 7 and K = 256 for a larger grid
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  1, 8, 8, 64},
    {96, 1, 1, 1, 64},
    {0, 0, 0},
    {0, 0, 0},
    {1, 1, 1},
    {1, 1, 1},
    1
  });
  // Filter 3x3x3 + no padding
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  3, 5, 8, 64},
    {96, 3, 3, 3, 64},
    {0, 0, 0},
    {0, 0, 0},
    {1, 1, 1},
    {1, 1, 1},
    1
  });
  // Filter 3x3x3 + symmetric padding with c % cta_k !=0
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  3, 5, 8, 32},
    {96, 3, 3, 3, 32},
    {1, 1, 1},
    {1, 1, 1},
    {1, 1, 1},
    {1, 1, 1},
    1
  });
  // Filter 3x4x5 + symmetric padding 111
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  3, 5, 8, 64},
    {96, 3, 4, 5, 64},
    {1, 1, 1},
    {1, 1, 1},
    {1, 1, 1},
    {1, 1, 1},
    1
  });
  // Filter 3x4x5 + asymmetric padding 102/010
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  3, 5, 8, 64},
    {96, 3, 4, 5, 64},
    {1, 0, 1},
    {0, 2, 0},
    {1, 1, 1},
    {1, 1, 1},
    1
  });
  // Filter 3x4x5 + asymmetric padding 102/010, w/ stride
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  16, 10, 16, 64},
    {96, 3, 4, 5, 64},
    {1, 0, 1},
    {0, 2, 0},
    {2, 2, 3},
    {1, 1, 1},
    1
  });
  // Filter 3x4x5 + asymmetric padding 102/010, w/ dilation
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  16, 10, 16, 64},
    {96, 3,  4,  5,  64},
    {1, 0, 1},
    {0, 2, 0},
    {1, 1, 1},
    {2, 2, 3},
    1
  });
  // Filter 3x4x5 + asymmetric padding 102/010, w/ stride, w/ dilation
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  16, 10, 16, 64},
    {96, 3,  4,  5,  64},
    {1, 0, 1},
    {0, 2, 0},
    {2, 2, 3},
    {2, 2, 3},
    1
  });
  return problem_shapes;
}


/////////////////////////////////////////////////////////////////////////////////////////////////
// Wgrad
/////////////////////////////////////////////////////////////////////////////////////////////////

// Specialization for 1D wgrad problems
template<>
std::vector<cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kWgrad, 1>> inline
get_conv_problem_vector<1, cutlass::conv::Operator::kWgrad>() {
  using ProblemShape = cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kWgrad, 1>;
  std::vector<ProblemShape> problem_shapes;
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,  8, 64},  // nwc
    {64, 1, 64},  // ksc
    {0},          // padding lower (pad_w)
    {0},          // padding upper (pad_w)
    {1},          // stride (stride_w)
    {1},          // dilation (dilation_w)
    1             // group
  });
  // Filter-K = 16 for predication
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1, 8, 64},
    {16,1, 64},
    {0},
    {0},
    {1},
    {1},
    1
  });
  // N = 2 and K = 128 for a larger grid
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  8, 64},
    {96, 1, 64},
    {0},
    {0},
    {1},
    {1},
    1
  });
  // N = 7 and K = 256 for a even larger grid
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {7,   8, 64},
    {256, 1, 64},
    {0},
    {0},
    {1},
    {1},
    1
  });
  // 3 filter, no padding
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   8, 32},
    {256, 3, 32},
    {0},
    {0},
    {1},
    {1},
    1
  });
  // 3 filter, symmetric padding
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   8, 32},
    {256, 3, 32},
    {1},
    {1},
    {1},
    {1},
    1
  });
  // 4 filter, asymmetric padding
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   8, 32},
    {256, 4, 32},
    {0},
    {1},
    {1},
    {1},
    1
  });
  // 3 filter, asymmetric padding and tstride of 2
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   8, 32},
    {256, 3, 32},
    {0},
    {1},
    {2},
    {1},
    1
  });
  // 3 filter, asymmetric padding and dilation of 2
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   8, 32},
    {256, 3, 32},
    {0},
    {1},
    {1},
    {2},
    1
  });
  // To test streamk, equals to gemm-MxNxK size 128x640x2048
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   1024, 128},
    {640, 1,    128},
    {0},
    {0},
    {1},
    {1},
    1
  });
  // To test streamk, equals to gemm-MxNxK size 128x640x2080
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   1040, 128},
    {640, 1,    128},
    {0},
    {0},
    {1},
    {1},
    1
  });
  return problem_shapes;
}

// Specialization for 2D wgrad problems
template<>
std::vector<cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kWgrad, 2>> inline
get_conv_problem_vector<2, cutlass::conv::Operator::kWgrad>() {
  using ProblemShape = cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kWgrad, 2>;
  std::vector<ProblemShape> problem_shapes;
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,  8, 8, 64},  // nhwc
    {64, 1, 1, 64},  // krsc
    {0, 0},          // padding lower (pad_h, pad_w)
    {0, 0},          // padding upper (pad_h, pad_w)
    {1, 1},          // stride (stride_h, stride_w)
    {1, 1},          // dilation (dilation_h, dilation_w)
    1                // group
  });
  // Filter-K = 16 for predication
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,  8, 8, 64},
    {16, 1, 1, 64},
    {0, 0},
    {0, 0},
    {1, 1},
    {1, 1},
    1
  });
  // N = 2 and K = 128 for a larger grid
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  8, 8, 64},
    {96, 1, 1, 64},
    {0, 0},
    {0, 0},
    {1, 1},
    {1, 1},
    1
  });
  // N = 7 and K = 256 for a even larger grid
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {7,   8, 8, 64},
    {256, 1, 1, 64},
    {0, 0},
    {0, 0},
    {1, 1},
    {1, 1},
    1
  });
  // 3x3 filter, no padding
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   8, 8, 32},
    {256, 3, 3, 32},
    {0, 0},
    {0, 0},
    {1, 1},
    {1, 1},
    1
  });
  // 3x3 filter, symmetric padding
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   8, 8, 32},
    {256, 3, 3, 32},
    {1, 1},
    {1, 1},
    {1, 1},
    {1, 1},
    1
  });
  // 2x5 filter, asymmetric padding 1,0/1,0
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   8, 8, 32},
    {256, 2, 5, 32},
    {1, 1},
    {0, 0},
    {1, 1},
    {1, 1},
    1
  });
  // 2x5 filter, asymmetric padding 1,0/1,0, w/ stride
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   15, 16, 32},
    {256, 2,  5,  32},
    {1, 1},
    {0, 0},
    {2, 3},
    {1, 1},
    1
  });
  // 2x5 filter, asymmetric padding 1,0/1,0, w/ dilation
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   16, 16, 32},
    {256, 2,  5,  32},
    {1, 1},
    {0, 0},
    {1, 1},
    {2, 3},
    1
  });
  // 2x5 filter, asymmetric padding 1,0/1,0, w/ stride, w/ dilation
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   16, 15, 32},
    {256, 2,  5,  32},
    {1, 1},
    {0, 0},
    {2, 3},
    {2, 3},
    1
  });
  // To test streamk, equals to gemm-MxNxK size 128x640x2048
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   64, 16, 128},
    {640, 1,  1,  128},
    {0, 0},
    {0, 0},
    {1, 1},
    {1, 1},
    1
  });
  // To test streamk, equals to gemm-MxNxK size 128x640x2080
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   65, 16, 128},
    {640, 1,  1,  128},
    {0, 0},
    {0, 0},
    {1, 1},
    {1, 1},
    1
  });
  return problem_shapes;
}

// Specialization for 3D wgrad problems
template<>
std::vector<cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kWgrad, 3>> inline
get_conv_problem_vector<3, cutlass::conv::Operator::kWgrad>() {
  using ProblemShape = cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kWgrad, 3>;
  std::vector<ProblemShape> problem_shapes;
  problem_shapes.push_back({
     cutlass::conv::Mode::kCrossCorrelation,
     {2,  1, 8, 8, 64},  // ndhwc
     {64, 1, 1, 1, 64},  // ktrsc
     {0, 0, 0},          // padding lower (pad_d, pad_h, pad_w)
     {0, 0, 0},          // padding upper (pad_d, pad_h, pad_w)
     {1, 1, 1},          // stride (stride_d, stride_h, stride_w)
     {1, 1, 1},          // dilation (dilation_d, dilation_h, dilation_w)
     1                   // group
   });
  // Filter 3x3x3 + no padding
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  3, 5, 8, 32},
    {96, 3, 3, 3, 32},
    {0, 0, 0},
    {0, 0, 0},
    {1, 1, 1},
    {1, 1, 1},
    1
  });
  // Filter 3x4x5 + asymmetric padding 102/010
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  3, 5, 8, 32},
    {96, 3, 4, 5, 32},
    {1, 0, 1},
    {0, 2, 0},
    {1, 1, 1},
    {1, 1, 1},
    1
  });
  // Filter 3x4x5 + asymmetric padding 102/010, w/ stride
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  16, 10, 16, 32},
    {96, 3,  4,  5,  32},
    {1, 0, 1},
    {0, 2, 0},
    {2, 2, 3},
    {1, 1, 1},
    1
  });
  // Filter 3x4x5 + asymmetric padding 102/010, w/ dilation
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  16, 10, 16, 32},
    {96, 3,  4,  5,  32},
    {1, 0, 1},
    {0, 2, 0},
    {1, 1, 1},
    {2, 2, 3},
    1
  });
  // To test streamk, equals to gemm-MxNxK size 128x640x2048
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   1, 64, 16, 128},
    {640, 1, 1,  1,  128},
    {0, 0, 0},
    {0, 0, 0},
    {1, 1, 1},
    {1, 1, 1},
    1
  });
  // To test streamk, equals to gemm-MxNxK size 128x640x2080
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   1, 65, 16, 128},
    {640, 1, 1,  1,  128},
    {0, 0, 0},
    {0, 0, 0},
    {1, 1, 1},
    {1, 1, 1},
    1
  });
  return problem_shapes;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Grouped Wgrad
/////////////////////////////////////////////////////////////////////////////////////////////////

// Get problem size vectors for group conv problems
template<int SpatialDim, cutlass::conv::Operator ConvOp>
std::vector<cutlass::conv::ConvProblemShape<ConvOp, SpatialDim>>
inline
get_grouped_conv_problem_vector(int GroupsPerTile);

// Specialization for 3D wgrad problems
template<>
std::vector<cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kWgrad, 3>> inline
get_grouped_conv_problem_vector<3, cutlass::conv::Operator::kWgrad>(int GroupsPerTile) {
  using ProblemShape = cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kWgrad, 3>;
  std::vector<ProblemShape> problem_shapes;

  if (GroupsPerTile == 1) {
    // channel_per_group == 64
    problem_shapes.push_back({
      cutlass::conv::Mode::kCrossCorrelation,
      {1, 1, 16, 16, 2048}, // ndhwc
      {2048, 1, 3, 3, 64},  // ktrsc
      {0, 1, 1},            // padding lower (pad_d, pad_h, pad_w)
      {0, 1, 1},            // padding upper (pad_d, pad_h, pad_w)
      {1, 1, 1},            // stride (stride_d, stride_h, stride_w)
      {1, 1, 1},            // dilation (dilation_d, dilation_h, dilation_w)
      32                    // groups
    });
  }
  else if (GroupsPerTile == 2) {
    // channel_per_group == 32
    problem_shapes.push_back({
      cutlass::conv::Mode::kCrossCorrelation,
      {1, 1, 16, 16, 1024}, // ndhwc
      {1024, 1, 3, 3, 32},  // ktrsc
      {0, 1, 1},            // padding lower (pad_d, pad_h, pad_w)
      {0, 1, 1},            // padding upper (pad_d, pad_h, pad_w)
      {1, 1, 1},            // stride (stride_d, stride_h, stride_w)
      {1, 1, 1},            // dilation (dilation_d, dilation_h, dilation_w)
      32                    // groups
    });
  }
  else if (GroupsPerTile == 4) {
    // channel_per_group == 16
    problem_shapes.push_back({
      cutlass::conv::Mode::kCrossCorrelation,
      {1, 1, 16, 16, 512}, // ndhwc
      {512, 1, 3, 3, 16},  // ktrsc
      {0, 1, 1},           // padding lower (pad_d, pad_h, pad_w)
      {0, 1, 1},           // padding upper (pad_d, pad_h, pad_w)
      {1, 1, 1},           // stride (stride_d, stride_h, stride_w)
      {1, 1, 1},           // dilation (dilation_d, dilation_h, dilation_w)
      32                   // groups
    });
  }
  else if (GroupsPerTile == 8) {
    // channel_per_group == 8
    problem_shapes.push_back({
      cutlass::conv::Mode::kCrossCorrelation,
      {1, 1, 16, 16, 256},  // ndhwc
      {256, 1, 3, 3, 8},    // ktrsc
      {0, 1, 1},            // padding lower (pad_d, pad_h, pad_w)
      {0, 1, 1},            // padding upper (pad_d, pad_h, pad_w)
      {1, 1, 1},            // stride (stride_d, stride_h, stride_w)
      {1, 1, 1},            // dilation (dilation_d, dilation_h, dilation_w)
      32                    // groups
    });
  }
  return problem_shapes;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Unit Stride Dgrad
/////////////////////////////////////////////////////////////////////////////////////////////////

// Specialization for 1D dgrad problems
template<>
std::vector<cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kDgrad, 1>> inline
get_conv_problem_vector<1, cutlass::conv::Operator::kDgrad, false>() {
  using ProblemShape = cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kDgrad, 1>;
  std::vector<ProblemShape> problem_shapes;
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,  8, 64},  // nqk
    {64, 1, 64},  // ksc
    {0},          // padding lower (pad_w)
    {0},          // padding upper (pad_w)
    {1},          // stride (stride_w)
    {1},          // dilation (dilation_w)
    1             // group
  });
  // non-packed input strides.
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,   8,  64},  // nqk
    {800, 80, 1},   // stride (nqk)
    {64,  1,  64},  // ksc
    {64,  64, 1},   // stride (ksc)
    {0},            // padding lower (pad_w)
    {0},            // padding upper (pad_w)
    {1},            // stride (stride_w)
    {1},            // dilation (dilation_w)
    1               // group
  });
  // non-packed output strides.
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,   8,  64},  // nqk
    {512, 64, 1},   // stride (nqk)
    {64,  1,  64},  // ksc
    {64,  64, 1},   // stride (ksc)
    {800, 80, 1},   // stride (nwc)
    {0},            // padding lower (pad_w)
    {0},            // padding upper (pad_w)
    {1},            // stride (stride_w)
    {1},            // dilation (dilation_w)
    1               // group
  });
  // Filter-K = 16 for predication
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,  8, 16},
    {64, 1, 16},
    {0},
    {0},
    {1},
    {1},
    1
  });
  // N = 2 and K = 128 for a larger grid
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  8, 96},
    {64, 1, 96},
    {0},
    {0},
    {1},
    {1},
    1
  });
  // N = 7 and K = 256 for a even larger grid
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {7,  8, 256},
    {64, 1, 256},
    {0},
    {0},
    {1},
    {1},
    1
  });
  // 3 filter, no padding
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  8, 256},
    {64, 3, 256},
    {0},
    {0},
    {1},
    {1},
    1
  });
  // 3 filter, symmetric padding with k % cta_k !=0
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  8, 256},
    {32, 3, 256},
    {1},
    {1},
    {1},
    {1},
    1
  });
  // 4 filter, asymmetric padding
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  8, 256},
    {64, 4, 256},
    {0},
    {1},
    {1},
    {1},
    1
  });
  // 3 filter, asymmetric padding and dilation of 2
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   16, 64},
    {256, 3,  64},
    {0},
    {1},
    {1},
    {2},
    1
  });
  return problem_shapes;
}

// Specialization for 2D dgrad problems
template<>
std::vector<cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kDgrad, 2>> inline
get_conv_problem_vector<2, cutlass::conv::Operator::kDgrad, false>() {
  using ProblemShape = cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kDgrad, 2>;
  std::vector<ProblemShape> problem_shapes;
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,  8, 8, 64},  // npqk
    {64, 1, 1, 64},  // krsc
    {0, 0},          // padding lower (pad_h, pad_w)
    {0, 0},          // padding upper (pad_h, pad_w)
    {1, 1},          // stride (stride_h, stride_w)
    {1, 1},          // dilation (dilation_h, dilation_w)
    1                // group
  });
  // non-packed input strides.
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,    8,   8,  64},  // npqk
    {8000, 800, 80, 1},   // stride (npqk)
    {64,   1,   1,  64},  // krsc
    {64,   64,  64, 1},   // stride (krsc)
    {0, 0},               // padding lower (pad_h, pad_w)
    {0, 0},               // padding upper (pad_h, pad_w)
    {1, 1},               // stride (stride_h, stride_w)
    {1, 1},               // dilation (dilation_h, dilation_w)
    1                     // group
  });
  // non-packed output strides.
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,    8,   8,  64},  // npqk
    {4096, 512, 64, 1},   // stride (npqk)
    {64,   1,   1,  64},  // krsc
    {64,   64,  64, 1},   // stride (krsc)
    {8000, 800, 80, 1},   // stride (nhwc)
    {0, 0},               // padding lower (pad_h, pad_w)
    {0, 0},               // padding upper (pad_h, pad_w)
    {1, 1},               // stride (stride_h, stride_w)
    {1, 1},               // dilation (dilation_h, dilation_w)
    1                     // group
  });
  // Filter-K = 16 for predication
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,  8, 8, 16},
    {64, 1, 1, 16},
    {0, 0},
    {0, 0},
    {1, 1},
    {1, 1},
    1
  });
  // N = 2 and K = 128 for a larger grid
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  8, 8, 96},
    {64, 1, 1, 96},
    {0, 0},
    {0, 0},
    {1, 1},
    {1, 1},
    1
  });
  // N = 7 and K = 256 for a even larger grid
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {7,  8, 8, 256},
    {64, 1, 1, 256},
    {0, 0},
    {0, 0},
    {1, 1},
    {1, 1},
    1
  });
  // 3x3 filter, no padding
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  8, 8, 256},
    {64, 3, 3, 256},
    {0, 0},
    {0, 0},
    {1, 1},
    {1, 1},
    1
  });
  // 3x3 filter, symmetric padding with k % cta_k !=0
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  8, 8, 256},
    {32, 3, 3, 256},
    {1, 1},
    {1, 1},
    {1, 1},
    {1, 1},
    1
  });
  // 2x5 filter, asymmetric padding 1,0/1,0
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  8, 8, 256},
    {64, 2, 5, 256},
    {1, 1},
    {0, 0},
    {1, 1},
    {1, 1},
    1
  });
  // 2x5 filter, asymmetric padding 1,0/1,0, w/ dilation
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,   16, 16, 64},
    {256, 2,  5,  64},
    {1, 1},
    {0, 0},
    {1, 1},
    {2, 3},
    1
  });
  return problem_shapes;
}

// Specialization for 3D dgrad problems
template<>
std::vector<cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kDgrad, 3>> inline
get_conv_problem_vector<3, cutlass::conv::Operator::kDgrad, false>() {
  using ProblemShape = cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kDgrad, 3>;
  std::vector<ProblemShape> problem_shapes;
  // Filter-K = 16 for predication
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,  1, 8, 8, 16},
    {64, 1, 1, 1, 16},
    {0, 0, 0},
    {0, 0, 0},
    {1, 1, 1},
    {1, 1, 1},
    1
  });
  // non-packed input output strides.
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,    1,    8,   8,  64},  // nzpqk
    {8000, 8000, 800, 80, 1},   // stride (nzpqk)
    {64,   1,    1,   1,  64},  // ktrsc
    {64,   64,   64,  64, 1},   // stride (ktrsc)
    {8000, 8000, 800, 80, 1},   // stride (ndhwc)
    {0, 0, 0},                  // padding lower (pad_d, pad_h, pad_w)
    {0, 0, 0},                  // padding upper (pad_d, pad_h, pad_w)
    {1, 1, 1},                  // stride (stride_d, stride_h, stride_w)
    {1, 1, 1},                  // dilation (dilation_d, dilation_h, dilation_w)
    1                           // group
  });
  // N = 7 and K = 256 for a larger grid
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  1, 8, 8, 96},
    {64, 1, 1, 1, 96},
    {0, 0, 0},
    {0, 0, 0},
    {1, 1, 1},
    {1, 1, 1},
    1
  });
  // Filter 3x4x5 + symmetric padding 111
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  3, 5, 8, 96},
    {64, 3, 4, 5, 96},
    {1, 1, 1},
    {1, 1, 1},
    {1, 1, 1},
    {1, 1, 1},
    1
  });
  // Filter 3x4x5 + asymmetric padding 102/010
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  3, 5, 8, 96},
    {64, 3, 4, 5, 96},
    {1, 0, 1},
    {0, 2, 0},
    {1, 1, 1},
    {1, 1, 1},
    1
  });
  // Filter 3x4x5 + asymmetric padding 102/010, w/ dilation
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  16, 10, 16, 64},
    {64, 3,  4,  5,  96},
    {1, 0, 1},
    {0, 2, 0},
    {1, 1, 1},
    {2, 2, 3},
    1
  });
  return problem_shapes;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Strided Dgrad
/////////////////////////////////////////////////////////////////////////////////////////////////

// Specialization for 1D dgrad problems
template<>
std::vector<cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kDgrad, 1>> inline
get_conv_problem_vector<1, cutlass::conv::Operator::kDgrad, true>() {
  using ProblemShape = cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kDgrad, 1>;
  std::vector<ProblemShape> problem_shapes;
  // Test TMA truncation
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,  512, 64},  // nqk
    {64, 1, 64},  // ksc
    {0},          // padding lower (pad_w)
    {0},          // padding upper (pad_w)
    {2},          // stride (stride_w)
    {1},          // dilation (dilation_w)
    1             // group
  });
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,  1024, 64},  // nqk
    {64, 1, 64},  // ksc
    {0},          // padding lower (pad_w)
    {0},          // padding upper (pad_w)
    {4},          // stride (stride_w)
    {1},          // dilation (dilation_w)
    1             // group
  });
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {1,  2048, 64},  // nqk
    {64, 1, 64},  // ksc
    {0},          // padding lower (pad_w)
    {0},          // padding upper (pad_w)
    {8},          // stride (stride_w)
    {1},          // dilation (dilation_w)
    1             // group
  });
  // non-packed input/output strides.
  // stride divides dilation
  // asymmetric padding
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {3,   8,  64},  // nqk
    {800, 80, 1},   // stride (nqk)
    {64,  3,  64},  // ksc
    {64,  64, 1},   // stride (ksc)
    {800, 80, 1},   // stride (nwc)
    {0},            // padding lower (pad_w)
    {1},            // padding upper (pad_w)
    {2},            // stride (stride_w)
    {4},            // dilation (dilation_w)
    1               // group
  });
  // non-packed input/output strides.
  // dilation divides stride
  // asymmetric padding
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {3,   8,  64},  // nqk
    {800, 80, 1},   // stride (nqk)
    {64,  3,  64},  // ksc
    {64,  64, 1},   // stride (ksc)
    {800, 80, 1},   // stride (nwc)
    {1},            // padding lower (pad_w)
    {0},            // padding upper (pad_w)
    {4},            // stride (stride_w)
    {2},            // dilation (dilation_w)
    1               // group
  });
  // non-packed input/output strides.
  // stride dilation dont divide
  // asymmetric padding
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {3,   8,  64},  // nqk
    {800, 80, 1},   // stride (nqk)
    {64,  3,  64},  // ksc
    {64,  64, 1},   // stride (ksc)
    {800, 80, 1},   // stride (nwc)
    {1},            // padding lower (pad_w)
    {2},            // padding upper (pad_w)
    {2},            // stride (stride_w)
    {3},            // dilation (dilation_w)
    1               // group
  });
  return problem_shapes;
}

// Specialization for 2D dgrad problems
template<>
std::vector<cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kDgrad, 2>> inline
get_conv_problem_vector<2, cutlass::conv::Operator::kDgrad, true>() {
  using ProblemShape = cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kDgrad, 2>;
  std::vector<ProblemShape> problem_shapes;
  // 2x5 filter, asymmetric padding 1,0/1,0, w/ dilation
  // mode 0 stride divides dilation
  // mode 1 dilation divides stride
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {3,   16, 16, 64},
    {256, 2, 5, 64},
    {1, 0},
    {0, 1},
    {2, 4},
    {4, 2},
    1
  });
  // 2x5 filter, asymmetric padding 1,0/1,0, w/ dilation
  // mode 0 dilation divides stride
  // mode 1 stride divides dilation
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {3,   16, 16, 64},
    {256, 2, 5, 64},
    {1, 0},
    {0, 1},
    {4, 2},
    {2, 4},
    1
  });
  // 2x5 filter, asymmetric padding 1,0/1,0, w/ dilation
  // stride dilation dont divide
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {3,   16, 16, 64},
    {256, 2, 5, 64},
    {1, 0},
    {0, 1},
    {3, 2},
    {2, 3},
    1
  });
  return problem_shapes;
}

// Specialization for 3D dgrad problems
template<>
std::vector<cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kDgrad, 3>> inline
get_conv_problem_vector<3, cutlass::conv::Operator::kDgrad, true>() {
  using ProblemShape = cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kDgrad, 3>;
  std::vector<ProblemShape> problem_shapes;
  // Filter 3x4x5 + asymmetric padding 102/010, w/ dilation
  problem_shapes.push_back({
    cutlass::conv::Mode::kCrossCorrelation,
    {2,  16, 10, 16, 64},
    {64, 3, 4, 5, 96},
    {1, 0, 1},
    {0, 2, 0},
    {2, 1, 2},
    {4, 2, 3},
    1
  });
  return problem_shapes;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::test
