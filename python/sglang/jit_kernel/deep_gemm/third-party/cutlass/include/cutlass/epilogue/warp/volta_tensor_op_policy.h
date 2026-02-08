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

#include "cutlass/matrix_shape.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Policy details related to the epilogue
template <
  typename WarpShape,             ///< shape of warp-level GEMM (concept: MatrixShape)
  typename InterleavedTileShape,  ///< shape of indivisible instruction-level arrangement (concept: GemmShape)
  typename ElementC,              ///< Accumulator layout
  typename Layout                 ///< target shared memory layout
>
struct VoltaTensorOpPolicy; 

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for row-major
template <
  typename WarpShape_          ///< shape of warp-level GEMM (concept: GemmShape)
>
struct VoltaTensorOpPolicy<WarpShape_, gemm::GemmShape<32, 32, 4>, half_t, layout::RowMajor> {

  using WarpShape = WarpShape_;
  using InterleavedTileShape = gemm::GemmShape<32, 32, 4>;
  using ElementC = half_t;
  using Layout = layout::RowMajor;

  /// Shape of one warp-levelinstruction
  using InstructionShape = gemm::GemmShape<16, 16, 4>;

  /// Number of mma operations performed for one 32x32x4 interleaved tile
  using MmaIterations = MatrixShape<
    InterleavedTileShape::kM / InstructionShape::kM,
    InterleavedTileShape::kN / InstructionShape::kN
  >;

  /// Number of 32x32x4 interleaved tiles performed to cover the warp-level GEMM shape
  using TileIterations = MatrixShape<
    WarpShape::kM / InterleavedTileShape::kM,
    WarpShape::kN / InterleavedTileShape::kN
  >;

  /// Number of accumulator elements owned by each thread per Mma
  static int const kElementsPerMma = 8;
  static int const kRowsPerIteration = 16;

  //
  // Hard-coded constants regarding Tensor Operations
  //

  /// Number of accumulator elements stored per memory instruction to shared memory
  static int const kElementsPerAccess = 4;
  
  /// Number of accesses performed per interleaved tile
  static int const kAccessesPerInterleavedTile = 4;

  /// Total number of iterations needed to cover the entire tile
  static int const kIterations = TileIterations::kRow * 2;

  //
  // Derived types
  //

  /// Array type for aligned memory accesses
  using AccessType = AlignedArray<ElementC, kElementsPerAccess>;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<
    ElementC, 
    kElementsPerAccess * kAccessesPerInterleavedTile * TileIterations::kColumn>;

  /// This is the complete warp-level accumulator tile.
  using AccumulatorTile = Array<
    ElementC, 
    TileIterations::kCount * MmaIterations::kCount * kElementsPerMma>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for row-major
template <
  typename WarpShape_          ///< shape of warp-level GEMM (concept: MatrixShape)
>
struct VoltaTensorOpPolicy<WarpShape_, gemm::GemmShape<32, 32, 4>, float, layout::RowMajor> {

  using WarpShape = WarpShape_;
  using InterleavedTileShape = gemm::GemmShape<32, 32, 4>;
  using ElementC = float;
  using Layout = layout::RowMajor;

  /// Shape of one warp-levelinstruction
  using InstructionShape = gemm::GemmShape<16, 16, 4>;

  /// Number of mma operations performed for one 32x32x4 interleaved tile
  using MmaIterations = MatrixShape<
    InterleavedTileShape::kM / InstructionShape::kM,
    InterleavedTileShape::kN / InstructionShape::kN
  >;

  /// Number of 32x32x4 interleaved tiles performed to cover the warp-level GEMM shape
  using TileIterations = MatrixShape<
    WarpShape::kM / InterleavedTileShape::kM,
    WarpShape::kN / InterleavedTileShape::kN
  >;

  /// Number of accumulator elements owned by each thread per Mma
  static int const kElementsPerMma = 8;
  static int const kRowsPerIteration = 16;

  //
  // Hard-coded constants regarding Tensor Operations
  //

  /// Number of accumulator elements stored per memory instruction to shared memory
  static int const kElementsPerAccess = 2;
  
  /// Number of accesses performed per interleaved tile
  static int const kAccessesPerInterleavedTile = 8;

  /// Number of rows per interleaved tile
  static int const kRowsPerMmaTile = 2;

  /// Total number of iterations needed to cover the entire tile
  static int const kIterations = TileIterations::kRow * MmaIterations::kRow;

  //
  // Derived types
  //
  
  /// Array type for aligned memory accesses
  using AccessType = AlignedArray<ElementC, kElementsPerAccess>;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<
    ElementC, 
    kElementsPerAccess * kAccessesPerInterleavedTile * TileIterations::kColumn>;

  /// This is the complete warp-level accumulator tile.
  using AccumulatorTile = Array<
    ElementC, 
    TileIterations::kCount * MmaIterations::kCount * kElementsPerMma>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
