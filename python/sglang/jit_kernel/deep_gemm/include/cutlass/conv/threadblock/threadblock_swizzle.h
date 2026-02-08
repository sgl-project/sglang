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
    \brief Implements several possible threadblock-swizzling functions mapping blockIdx to 
      Convolution problems.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/platform/platform.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////
CUTLASS_HOST_DEVICE
static int get_strided_dgrad_tile_m(
  cutlass::conv::Conv2dProblemSize const &problem_size,
  int tile_size_m) {

  // CTAs in M dimension per starting filter position
  int tile_m_per_filter = strided_dgrad_tile_m_per_filter(problem_size, tile_size_m);

  // Inflate number of CTAs in M dimension to cover every strating filter position even those that
  // may fall out of valid MMA (Dy * w) but are needed to apply epilogue (beta * Dx_source) 
  // and point-wise fusion
  int tile_m = tile_m_per_filter * int(problem_size.stride().product());

  // There is a possible performance optimization here that leads up to 2x speeds than the current 
  // CUTLASS strided dgrad performance for stride > filter, i.e., stride={2x2} and filter={1x1})
  //
  // * Optimization * 
  // Only launch CTAs in M dimension which contribute to a row in Dx output
  // 
  // 
  // * Constraints *
  // (A) stride <= filter, for example, stride={2x2} and filter={3x3}: 
  //       - (A.1): There are no constraints for this case and the optimization does 
  //                affect this case functionality or performance. 
  // (B) stride > filter, for example, stride={2x2} and filter={1x1}: 
  //       - (B.1): Dx output tensor should be zero initialized
  //       - (B.2): The kernel epilogue cannot apply beta. Thus, beta should be zero 

  return tile_m;
}
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Threadblock swizzling function for strided dgrad convolution
struct StridedDgradHorizontalThreadblockSwizzle : 
  public gemm::threadblock::GemmHorizontalThreadblockSwizzle {

  using Base = gemm::threadblock::GemmHorizontalThreadblockSwizzle;

  CUTLASS_HOST_DEVICE
  StridedDgradHorizontalThreadblockSwizzle() { }

  /// Returns the shape of the problem in units of logical tiles
  /// For ImplicitGemmConvolution Conv2d problem size: conv_operator(NPQK, NHWC, KRSC)
  CUTLASS_HOST_DEVICE
  static gemm::GemmCoord get_tiled_shape(
    cutlass::conv::Operator conv_operator,
    cutlass::conv::Conv2dProblemSize const &problem_size,
    gemm::GemmCoord tile_size,
    int split_k_slices) {

    gemm::GemmCoord implicit_gemm_problem_size = 
    cutlass::conv::implicit_gemm_problem_size(conv_operator, problem_size);

    // compute number of tiles in m dimension
    int tile_m = get_strided_dgrad_tile_m(problem_size, tile_size.m());

    // compute number of tiles in n dimension 
    int tile_n = (implicit_gemm_problem_size.n() + tile_size.n() - 1) / tile_size.n();

    return gemm::GemmCoord(
      tile_m,
      tile_n,
      split_k_slices);
  }

  /// Returns the shape of the problem in units of logical tiles
  /// For GEMM problem size (MxNxK) (Do not use base class get_tiled_shape())
  private:
    using Base::get_tiled_shape;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Threadblock swizzling function for strided dgrad convolution
template <int N = 1>
struct StridedDgradIdentityThreadblockSwizzle : 
  public gemm::threadblock::GemmIdentityThreadblockSwizzle<N> {

  using Base = gemm::threadblock::GemmIdentityThreadblockSwizzle<N>;

  CUTLASS_HOST_DEVICE
  StridedDgradIdentityThreadblockSwizzle() { }

  /// Returns the shape of the problem in units of logical tiles
  /// For ImplicitGemmConvolution Conv2d problem size: conv_operator(NPQK, NHWC, KRSC)
  CUTLASS_HOST_DEVICE
  static gemm::GemmCoord get_tiled_shape(
    cutlass::conv::Operator conv_operator,
    cutlass::conv::Conv2dProblemSize const &problem_size,
    gemm::GemmCoord tile_size,
    int split_k_slices) {

    gemm::GemmCoord implicit_gemm_problem_size = 
    cutlass::conv::implicit_gemm_problem_size(conv_operator, problem_size);

    // compute number of tiles in m dimension
    int tile_m = get_strided_dgrad_tile_m(problem_size, tile_size.m());

    // compute number of tiles in n dimension 
    int tile_n = (implicit_gemm_problem_size.n() + tile_size.n() - 1) / tile_size.n();

    return gemm::GemmCoord(
      tile_m,
      tile_n,
      split_k_slices);
  }

  /// Returns the shape of the problem in units of logical tiles
  /// For GEMM problem size (MxNxK) (Do not use base class get_tiled_shape())
  private:
    using Base::get_tiled_shape;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for GEMMs
template <int N = 1, int Output_N = 1, int Output_P = 1, int Output_Q = 1>
struct DepthwiseDirect2dConvIdentityThreadblockSwizzle
    : public gemm::threadblock::GemmIdentityThreadblockSwizzle<N> {
  CUTLASS_HOST_DEVICE
  DepthwiseDirect2dConvIdentityThreadblockSwizzle() {}

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  static gemm::GemmCoord get_tiled_shape(cutlass::conv::Operator conv_operator,
                            cutlass::conv::Conv2dProblemSize const &problem_size,
                            gemm::GemmCoord tile_size,
                            int split_k_slices) {
        
    gemm::GemmCoord implicit_gemm_problem_size =
        cutlass::conv::implicit_gemm_problem_size(conv_operator, problem_size);

    return gemm::GemmCoord(1,
                     (implicit_gemm_problem_size.n() + tile_size.n() - 1) / tile_size.n(),
                     split_k_slices);
  }
};

} // namespace threadblock
} // namespace conv
} // namespace cutlass
