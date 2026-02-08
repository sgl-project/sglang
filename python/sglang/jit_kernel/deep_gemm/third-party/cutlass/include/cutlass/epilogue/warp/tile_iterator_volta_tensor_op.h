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
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"

#include "cutlass/epilogue/warp/tensor_op_policy.h"
#include "cutlass/epilogue/warp/volta_tensor_op_policy.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory
template <
  typename WarpShape,             ///< shape of warp-level GEMM (concept: MatrixShape)
  typename InterleavedTileShape,  ///< shape of indivisible instruction-level arrangement (concept: GemmShape)
  typename ElementC,              ///< Accumulator layout
  typename Layout                 ///< target shared memory layout
>
struct TileIteratorVoltaTensorOp; 

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory
template <
  typename WarpShape_         ///< shape of warp-level GEMM (concept: MatrixShape)
>
struct TileIteratorVoltaTensorOp<WarpShape_, gemm::GemmShape<32, 32, 4>, half_t, layout::RowMajor> {
public:

  using WarpShape = WarpShape_;
  using InterleavedTileShape = gemm::GemmShape<32, 32, 4>;
  using Element = half_t;
  using Layout = layout::RowMajor;

  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = VoltaTensorOpPolicy<WarpShape, InterleavedTileShape, Element, Layout>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
    Policy::kRowsPerIteration,
    WarpShape::kN
  >;

  /// Array type for aligned memory accesses
  using AccessType = typename Policy::AccessType;
  
  /// This is the fragment size produced by one access of the iterator.
  using Fragment = typename Policy::Fragment;

  /// This is the complete warp-level accumulator tile.
  using AccumulatorTile = typename Policy::AccumulatorTile;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  /// Number of elements per access
  static int const kElementsPerAccess = Policy::kElementsPerAccess;

  // Internal constants
  struct Detail {
    static int const kLanesInQuad = 4;
    static int const kRowsPerQuad = 4;
    static int const kColumnsPerQuad = 8;
    static int const kAccessesPerQuad = kColumnsPerQuad / Policy::kElementsPerAccess;
    static int const kAccessQuadDelta = 16;
  };

  /// Padding quantity
  using Padding = MatrixShape<
    0,
    Policy::kElementsPerAccess>;

private:

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointer_;

  /// Internal layout object
  Layout layout_;

public:

  /// Default constructor
  CUTLASS_HOST_DEVICE
  TileIteratorVoltaTensorOp(): pointer_(nullptr) { }

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  TileIteratorVoltaTensorOp(
    TensorRef const &ref,
    unsigned lane_id
  ):
    pointer_(reinterpret_cast<AccessType *>(ref.data())),
    layout_(ref.stride()[0] / Policy::kElementsPerAccess) { 

    int quad_id = lane_id / Detail::kLanesInQuad;
    int lane_in_quad = (lane_id % Detail::kLanesInQuad);

    int quad_row_idx = ((quad_id & 4) >> 1) + (quad_id & 1);
    int quad_col_idx = ((quad_id & 2) >> 1);

    int row = quad_row_idx * Detail::kRowsPerQuad + lane_in_quad;
    int column = quad_col_idx * Detail::kColumnsPerQuad;

    pointer_ += layout_({row, column / kElementsPerAccess});
  }

  /// Adds a pointer offset
  CUTLASS_HOST_DEVICE
  TileIteratorVoltaTensorOp & add_pointer_offset(Index pointer_offset) {
    pointer_ += pointer_offset / Policy::kElementsPerAccess;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorVoltaTensorOp & add_tile_offset(TensorCoord const &tile_offset) {

    pointer_ += layout_({
      tile_offset.row() * Shape::kRow, 
      tile_offset.column() * Shape::kColumn / Policy::kElementsPerAccess});

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorVoltaTensorOp & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  /// Store
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int tile_idx = 0; tile_idx < Policy::TileIterations::kColumn; ++tile_idx) {

      CUTLASS_PRAGMA_UNROLL
      for (int access_idx = 0; access_idx < Policy::kAccessesPerInterleavedTile; ++access_idx) {

        int access_quad = access_idx / 2;
        int access = access_idx % 2;

        int ptr_offset = tile_idx * InterleavedTileShape::kN / Policy::kElementsPerAccess +
          access_quad * Detail::kAccessQuadDelta / Policy::kElementsPerAccess + 
          access + pointer_offset / Policy::kElementsPerAccess;

        int frag_idx = tile_idx * Policy::kAccessesPerInterleavedTile + access_idx;

        AccessType access_vector = frag_ptr[frag_idx];

        pointer_[ptr_offset] = access_vector;
      }
    }
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int tile_idx = 0; tile_idx < Policy::TileIterations::kColumn; ++tile_idx) {

      CUTLASS_PRAGMA_UNROLL
      for (int access_idx = 0; access_idx < Policy::kAccessesPerInterleavedTile; ++access_idx) {

        int access_quad = access_idx / 2;
        int access = access_idx % 2;

        int ptr_offset = tile_idx * Detail::kTileDelta + access_quad * Detail::kAccessQuadDelta + 
          access + pointer_offset / Policy::kElementsPerAccess;

        int frag_idx = tile_idx * Policy::kAccessesPerInterleavedTile + access_idx;

        frag_ptr[frag_idx] = pointer_[ptr_offset];
      }
    }
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load(Fragment const &frag) {
    load_with_pointer_offset(frag, 0);
  }
  
  /// Set smem base address
  CUTLASS_HOST_DEVICE
  void set_smem_base_address(Index address) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory
template <
  typename WarpShape_         ///< shape of warp-level GEMM (concept: MatrixShape)
>
struct TileIteratorVoltaTensorOp<WarpShape_, gemm::GemmShape<32, 32, 4>, float, layout::RowMajor> {
public:

  using WarpShape = WarpShape_;
  using InterleavedTileShape = gemm::GemmShape<32, 32, 4>;
  using Element = float;
  using Layout = layout::RowMajor;

  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = VoltaTensorOpPolicy<WarpShape, InterleavedTileShape, Element, Layout>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
    Policy::kRowsPerIteration,
    WarpShape::kN
  >;

  /// Array type for aligned memory accesses
  using AccessType = typename Policy::AccessType;
  
  /// This is the fragment size produced by one access of the iterator.
  using Fragment = typename Policy::Fragment;

  /// This is the complete warp-level accumulator tile.
  using AccumulatorTile = typename Policy::AccumulatorTile;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  /// Number of elements per access
  static int const kElementsPerAccess = Policy::kElementsPerAccess;

  // Internal constants
  struct Detail {
    static int const kLanesInQuad = 4;
    static int const kRowsPerQuad = 4;
    static int const kColumnsPerQuad = 8;
    static int const kAccessesPerQuad = kColumnsPerQuad / Policy::kElementsPerAccess;
    static int const kAccessQuadDelta = 16;
  };

  /// Padding quantity
  using Padding = MatrixShape<
    0,
    Policy::kElementsPerAccess>;

private:

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointer_;

  /// Internal layout object
  Layout layout_;

public:

  /// Default constructor
  CUTLASS_HOST_DEVICE
  TileIteratorVoltaTensorOp(): pointer_(nullptr) { }

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  TileIteratorVoltaTensorOp(
    TensorRef const &ref,
    unsigned lane_id
  ):
    pointer_(reinterpret_cast<AccessType *>(ref.data())),
    layout_(ref.stride()[0] / Policy::kElementsPerAccess) { 

    int quad_id = lane_id / Detail::kLanesInQuad;
    int lane_in_quad = (lane_id % Detail::kLanesInQuad);

    int const kQuadRowDelta = 4;
    int const kQuadColumnDelta = 2 * Policy::MmaIterations::kColumn;

    int quad_row_offset = ((quad_id & 4) / 2 + (quad_id & 1)) * kQuadRowDelta;
    int quad_column_offset = (quad_id & 2) / 2 * kQuadColumnDelta;

    int thread_row_offset = (lane_in_quad & 1);
    int thread_column_offset = (lane_in_quad & 2) / 2;

    int row = quad_row_offset + thread_row_offset;
    int column = quad_column_offset + thread_column_offset;

    pointer_ += layout_({row, column});
  }

  /// Adds a pointer offset
  CUTLASS_HOST_DEVICE
  TileIteratorVoltaTensorOp & add_pointer_offset(Index pointer_offset) {
    pointer_ += pointer_offset / Policy::kElementsPerAccess;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorVoltaTensorOp & add_tile_offset(TensorCoord const &tile_offset) {

    pointer_ += layout_({
      tile_offset.row() * Shape::kRow, 
      tile_offset.column() * Shape::kColumn / Policy::kElementsPerAccess});

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorVoltaTensorOp & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  /// Store
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    int const kAccessesPerRow = Policy::TileIterations::kColumn * Policy::MmaIterations::kColumn * 2;

    CUTLASS_PRAGMA_UNROLL
    for (int row_idx = 0; row_idx < Policy::kRowsPerMmaTile; ++row_idx) {

      CUTLASS_PRAGMA_UNROLL
      for (int access_idx = 0; access_idx < kAccessesPerRow; ++access_idx) {

        int frag_idx = row_idx * kAccessesPerRow + access_idx;

        int ptr_column_offset = (access_idx & 1) * 2 + 
          (access_idx & 2) * Policy::MmaIterations::kColumn * 2 + 
          (access_idx & 4) * Policy::MmaIterations::kColumn * 2;

        int ptr_row_offset = row_idx * 2;

        int ptr_offset = layout_({ptr_row_offset, ptr_column_offset}) + pointer_offset / Policy::kElementsPerAccess;

        pointer_[ptr_offset] = frag_ptr[frag_idx];
      }
    }
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    assert(0);
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load(Fragment const &frag) {
    load_with_pointer_offset(frag, 0);
  }
  
  /// Set smem base address
  CUTLASS_HOST_DEVICE
  void set_smem_base_address(Index address) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
