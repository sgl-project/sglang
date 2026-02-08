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
    \brief

This file contains definitions and utility functions for describing convolution problem sizes in terms of
activation (NHWC), filter (KRSC), output (NPQK), padding (pad_h, pad_w), stride (stride_h, stride_w), and
dilation (dilation_h, dilation_w).  Furthermore, it defines helper functions to map CUTLASS's implicit gemm
tensor extents, sizes, and data types to that of the convolution's extents, sizes, and data types.

                        * Mapping convolutions to Gemm computation *

Cutlass implements convolutions with the Implicit Gemm algorithm.  This algorithm performs a gemm
(general matrix-matrix multiply) on the convolution tensors Activation, Filter, and Output.
The underlying gemm operation follows the standard gemm definition:

                                     C = A * B + C

                               A and B are input matrices
                            C is source and output matrix


For the three convolutional operators (Fprop, Dgrad, Wgrad), ImplicitGemm matrices A, B, and C are mapped
to convolution tensors Activation, Filter and Output as described in the table below.

        ___________________________________________________________________________
         ConvolutionalOperator |        A        |      B         |       C
        ___________________________________________________________________________
        |                      |                 |                |               |
        |       Fprop          |    Activation   |    Filter      |     Output    |
        |       Dgrad          |     Output      |    Filter      |   Activation  |
        |       Wgrad          |     Output      |  Activation    |     Filter    |
        ___________________________________________________________________________

In convolution codebase, DO NOT mix using (A, B, C) with (Activation, Filter, Output).

For example, it's confusing and error prone to document a convolution class or function
as operating on "A, B, Output."  Instead, use the mapping functions below,
and adhere to using either A, B, C or Activation, Filter, Output.

Map elements' data types (ImplicitGemm -> Conv): GemmToConvElementMap
Map elements' data types (Conv -> ImplicitGemm): ConvToGemmElementMap
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm_enumerated_types.h"
#include "cutlass/matrix_coord.h"

namespace cutlass {
namespace conv {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Convolutional operator
enum class Operator {
  kFprop,
  kDgrad,
  kWgrad,
  kDeconv
};

/// Distinguishes convolution from cross correlation
enum class Mode {
  kCrossCorrelation,
  kConvolution
};

/// Selects among several implementation variants trading off performance with simplicity
enum class IteratorAlgorithm {
  kAnalytic,      ///< functionally correct in all cases but lower performance
  kOptimized,     ///< optimized for R <= 32, S <= 32 and unity-stride dgrad
  kFixedChannels, ///< Analytic algorithm optimized for fixed channel count (C == AccessSize)
  kFewChannels,   ///< Analytic algorithm optimized for few channels (C divisible by AccessSize)
  kFixedStrideDilation ///< Optimized for fixed stride and dilation
};

/// Distinguishes among partial specializations that accelerate certain problems where convolution
/// stride is unit.
enum class StrideSupport {
  kStrided,       ///< arbitrary convolution stride
  kUnity,         ///< unit convolution stride
  kFixed          ///< fixed convolution stride
};

/// Identifies split-K mode
enum class SplitKMode {
  kNone,
  kSerial,
  kParallel
};

/// Identifies group mode
enum class GroupMode {
  kNone,
  kSingleGroup,   ///< One CTA calculates one group or less
  kMultipleGroup, ///< One CTA calculates multiple groups
  kDepthwise      ///< One CTA calculates cta_n groups (problem_size.C == problem_size.K == problem_size.groups)
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Shape of a tensor
template <
  int N = 1,
  int H = 1,
  int W = 1,
  int C = 1
>
struct TensorNHWCShape {
  static int const kN = N;
  static int const kH = H;
  static int const kW = W;
  static int const kC = C;

  static int const kHW = H * W;
  static int const kNHW = N * kHW;
  static int const kNHWC = N * H * W * C;

  static int const kCount = kNHWC;

  //
  // Static member functions
  //

  /// Returns a Coord object
  CUTLASS_HOST_DEVICE
  static Coord<4> toCoord() {
    return make_Coord(kN, kH, kW, kC);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Shape of a conv2d stride, which controls how the filter convolves around the input volume
template <
  /// Stride in horizontal direction
  int u = 1,
  /// Stride in vertical direction
  int v = 1
>
struct Stride2D {
  static int const kU = u;
  static int const kV = v;

  //
  // Static member functions
  //

  /// Returns a Coord object
  CUTLASS_HOST_DEVICE
  static Coord<2> toCoord() {
    return make_Coord(kU, kV);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace conv
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////
