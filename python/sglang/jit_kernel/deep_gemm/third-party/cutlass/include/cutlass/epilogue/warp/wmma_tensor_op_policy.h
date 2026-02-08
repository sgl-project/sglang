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
    \brief Defines basic structures needed for implementing the warp-scoped phase of the epilogue.
          These quantities assume a 'column-major' arrangement of TensorOp instructions, of which
          a row-oriented slice is visible per iteration.
*/

#pragma once

#include "cutlass/arch/wmma.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/layout/matrix.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

/// Policy details related to the epilogue
template <
  typename WarpShape,     ///< shape of warp-level GEMM (concept: MatrixShape)
  typename OperatorShape, ///< matrix multiply operation shape (concept: gemm:GemmShape)
  typename Layout         ///< target shared memory layout
>
struct WmmaTensorOpPolicy; 

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for row-major
template <
  typename WarpShape,           ///< shape of warp-level GEMM (concept: MatrixShape)
  typename OperatorShape        ///< matrix multiply operation shape (concept: gemm::GemmShape)
>
struct WmmaTensorOpPolicy<WarpShape, OperatorShape, layout::RowMajor> {

  /// Number of operations
  using OperatorCount = MatrixShape<
    WarpShape::kM / OperatorShape::kM,
    WarpShape::kN / OperatorShape::kN
  >;

  //
  // Hard-coded constants regarding Tensor Operations
  //
  static int const kElementsPerAccess = 2;
  static int const kRowsPerIteration = OperatorShape::kM;
  static int const kWmmaFragmentsPerAccess = 1;

  //
  // Derived quantities
  //

  // Number of externally visible iterations
  static int const kIterations = OperatorCount::kRow;

};

////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////

#endif

