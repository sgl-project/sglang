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
    \brief This defines a "fragment" iterator for visiting the fragments of an accumulator tile
      that participate in one warp-level store operation.

      Typically, the accumulator tile is the largest single block of register-backed storage 
      within the kernel. Storing it to memory is best accomplished by partitioning it into
      smaller tiles and storing these sequentially.

      Round trips through shared memory during the Epilogue phase require partitioning, as
      shared memory capacity is typically insufficient for a threadblock's total accumulator
      size.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/warp/volta_tensor_op_policy.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// 
template <
  typename WarpShape,             ///< shape of warp-level GEMM (concept: MatrixShape)
  typename InterleavedTileShape,  ///< shape of indivisible instruction-level arrangement (concept: GemmShape)
  typename ElementC,              ///< Accumulator layout
  typename Layout                 ///< target shared memory layout
>
class FragmentIteratorVoltaTensorOp;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for row-major shared memory
template <
  typename WarpShape_         ///< shape of warp-level GEMM (concept: MatrixShape)
>
class FragmentIteratorVoltaTensorOp<WarpShape_, gemm::GemmShape<32, 32, 4>, half_t, layout::RowMajor> {
public:

  using WarpShape = WarpShape_;
  using InterleavedTileShape = gemm::GemmShape<32, 32, 4>;
  using ElementC = half_t;
  using Layout = layout::RowMajor;

  /// Policy operator
  using Policy = VoltaTensorOpPolicy<WarpShape, InterleavedTileShape, ElementC, Layout>;

  /// Array type for aligned memory accesses
  using AccessType = typename Policy::AccessType;
  
  /// This is the fragment size produced by one access of the iterator.
  using Fragment = typename Policy::Fragment;

  /// This is the complete warp-level accumulator tile.
  using AccumulatorTile = typename Policy::AccumulatorTile;

  using OutputAccumulatorTile = AccumulatorTile;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

private:

private:

  //
  // Data members
  //

  /// Accumulator tile
  AccessType const *accumulators_;

  /// Internal index
  int index_;

public:

  /// Constructs an iterator
  CUTLASS_HOST_DEVICE
  FragmentIteratorVoltaTensorOp(AccumulatorTile const &accum): 
    accumulators_(reinterpret_cast<AccessType const *>(&accum)), 
    index_(0) {

  }

  /// Increments
  CUTLASS_HOST_DEVICE
  FragmentIteratorVoltaTensorOp &operator++() {
    ++index_;
    return *this;
  }

  /// Decrements
  CUTLASS_HOST_DEVICE
  FragmentIteratorVoltaTensorOp &operator--() {
    --index_;
    return *this;
  }

  /// Loads a fragment from the referenced part of the accumulator tile
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, int index_offset = 0) const {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    static int const kAccessesPerMma = Policy::kElementsPerMma / Policy::kElementsPerAccess;

    CUTLASS_PRAGMA_UNROLL
    for (int tile_n = 0; tile_n < Policy::TileIterations::kColumn; ++tile_n) {
      
      int tile_access_idx = 
        (tile_n * Policy::TileIterations::kRow + (index_ & 2) / 2) * Policy::MmaIterations::kCount * kAccessesPerMma;

      CUTLASS_PRAGMA_UNROLL
      for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn * kAccessesPerMma; ++mma_n) {

        int mma_access_idx = ((mma_n & 1) * 2 + (index_ & 1)) * kAccessesPerMma + (mma_n & 2) / 2;

        frag_ptr[tile_n * Policy::MmaIterations::kColumn * kAccessesPerMma +
          mma_n] = accumulators_[tile_access_idx + mma_access_idx];
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for row-major shared memory
template <
  typename WarpShape_         ///< shape of warp-level GEMM (concept: MatrixShape)
>
class FragmentIteratorVoltaTensorOp<WarpShape_, gemm::GemmShape<32, 32, 4>, float, layout::RowMajor> {
public:

  using WarpShape = WarpShape_;
  using InterleavedTileShape = gemm::GemmShape<32, 32, 4>;
  using ElementC = float;
  using Layout = layout::RowMajor;

  /// Policy operator
  using Policy = VoltaTensorOpPolicy<WarpShape, InterleavedTileShape, ElementC, Layout>;

  /// Array type for aligned memory accesses
  using AccessType = typename Policy::AccessType;
  
  /// This is the fragment size produced by one access of the iterator.
  using Fragment = typename Policy::Fragment;

  /// This is the complete warp-level accumulator tile.
  using AccumulatorTile = typename Policy::AccumulatorTile;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

private:

private:

  //
  // Data members
  //

  /// Accumulator tile
  AccessType const *accumulators_;

  /// Internal index
  int index_;

public:

  /// Constructs an iterator
  CUTLASS_HOST_DEVICE
  FragmentIteratorVoltaTensorOp(AccumulatorTile const &accum): 
    accumulators_(reinterpret_cast<AccessType const *>(&accum)), 
    index_(0) {
  }

  /// Increments
  CUTLASS_HOST_DEVICE
  FragmentIteratorVoltaTensorOp &operator++() {
    ++index_;
    return *this;
  }

  /// Decrements
  CUTLASS_HOST_DEVICE
  FragmentIteratorVoltaTensorOp &operator--() {
    --index_;
    return *this;
  }

  /// Loads a fragment from the referenced part of the accumulator tile
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag, int index_offset = 0) const {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    int const kRegsPerMmaRow = 2;
      
    CUTLASS_PRAGMA_UNROLL
    for (int reg_row = 0; reg_row < Policy::kRowsPerMmaTile; ++reg_row) {

      CUTLASS_PRAGMA_UNROLL
      for (int tile_n = 0; tile_n < Policy::TileIterations::kColumn; ++tile_n) {
    
        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn * 2; ++mma_n) {

          int mma_idx = (index_ & 1) + (index_ & 2) * Policy::MmaIterations::kCount / 2 +
            (tile_n * Policy::TileIterations::kRow) * Policy::MmaIterations::kCount + (mma_n & 1) * 2;

          int reg_offset = reg_row * kRegsPerMmaRow + (mma_n & 2) * 2;
          int reg_idx = mma_idx * Policy::kElementsPerMma + reg_offset;

          *frag_ptr = accumulators_[reg_idx / Policy::kElementsPerAccess];
          ++frag_ptr;
        }
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////


} // namespace warp
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

