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

#include "cutlass/cutlass.h"
#include "cutlass/coord.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/layout/pitch_linear.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace layout {

////////////////////////////////////////////////////////////////////////////////

/// Template based on element size (in bits) - defined in terms of pitch-linear
/// memory and Crosswise size (in elements).
/// This one is the base class of all Ampere/Turing fp16/bf16/int8/int4/int1
/// tensor core kernels.  tf32 TN uses this too.
template <int ElementSize, int Crosswise>
struct TensorOpMultiplicand {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = PitchLinearCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Static constants
  //

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = 128;

  static int const kElementSize = ElementSize;
  static int const kElementsPerAccess = kAccessSize / kElementSize;
  static int const kCrosswise = Crosswise;

  /// Contiguous dimension of the tile shape matches one shared memory cache
  /// line - 128B.  For 128bit access size, it equals to 8 accesses.
  static int const kTileShapeContiguous = 128 / (kAccessSize / 8);

  /// Number of kblocks to store PartitionShape::kContiguous Elements
  static int const kFactor =
      kTileShapeContiguous * kElementsPerAccess / kCrosswise;

  static_assert(
      (kFactor > 0),
      "kCrosswise should be no large than one shared memory cache line.");

  /// The strided dimension needs to be at least (WarpSize(32) /
  /// kTileShapeContiguous) for a warp to access.  To ensure conflict free
  /// access, it also needs to be at least (kTileShapeContiguous / kFactor).
  /// See comments below
  static int const kTileShapeStride =
      ((kTileShapeContiguous / kFactor) > (32 / kTileShapeContiguous))
          ? (kTileShapeContiguous / kFactor)
          : (32 / kTileShapeContiguous);

  /// Fundamental tile shape in units of vectors to guarantee bank conflict free
  /// shared memory load/store.
  /// For kFactor = 1, TileShape = <8, 8> 
  /// For kFactor > 1, TileShape = <8, 4>
  using TileShape = PitchLinearShape<kTileShapeContiguous, kTileShapeStride>;

  /// Fundamental partition shape in units of vectors
  using PartitionShape = PitchLinearShape<4, 4>;

  using PartitionCount =
      PitchLinearShape<TileShape::kContiguous / PartitionShape::kContiguous,
                       TileShape::kStrided / PartitionShape::kStrided>;

  using AccessCount =
      PitchLinearShape<PartitionShape::kContiguous, PartitionShape::kStrided>;

 private:
  //
  // Data members
  //

  /// Stride data member. For GEMM, it equals to kCrosswise x stage.
  Stride stride_;

 public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicand(Index ldm = 0) : stride_(ldm) {}

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicand(Stride stride) : stride_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static TensorOpMultiplicand packed(TensorCoord const &extent) {
    return TensorOpMultiplicand(extent[0]);
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    //
    // First, compute c and s of vector within source (in units of vector
    // accesses)
    //

    int vec_contiguous_idx = coord.contiguous() / kElementsPerAccess;
    int vec_strided_idx = coord.strided() / kFactor;

    // Compute the fundamental tile being accessed
    int tile_contiguous_idx =
        vec_contiguous_idx / (TileShape::kContiguous / kFactor);

    int tile_contiguous_residual =
        vec_contiguous_idx % (TileShape::kContiguous / kFactor) +
        ((coord.strided() % kFactor) * (TileShape::kContiguous / kFactor));
    int tile_strided_residual = vec_strided_idx % TileShape::kStrided;

    // Compute the 'partition' within the fundamental tile
    int partition_contiguous_idx =
        tile_contiguous_residual / PartitionShape::kContiguous;
    int partition_strided_idx =
        tile_strided_residual / PartitionShape::kStrided;

    int partition_contiguous_residual =
        tile_contiguous_residual % PartitionShape::kContiguous;
    int partition_strided_residual =
        tile_strided_residual % PartitionShape::kStrided;

    //
    // Then swizzle
    //

    int permuted_vec_contiguous_within_partition =
        partition_contiguous_residual ^ (partition_strided_residual % 4);

    int permuted_partition_contiguous_within_tile =
        partition_contiguous_idx ^ (partition_strided_idx % 2);

    //
    // Compute final element location
    //

    int element_contiguous = (tile_contiguous_idx * TileShape::kContiguous +
                              permuted_partition_contiguous_within_tile *
                                  PartitionShape::kContiguous +
                              permuted_vec_contiguous_within_partition) *
                                 kElementsPerAccess +
                             (coord.contiguous() % kElementsPerAccess);

    int element_strided = vec_strided_idx;

    return element_contiguous + element_strided * stride_[0] * kFactor;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const { return stride_; }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride &stride() { return stride_; }

  /// Compute the number of contiguous elements needed to store a tensor with
  /// the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return extent[1] * stride_[0];
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template based on element size (in bits) - defined in terms of pitch-linear
/// memory and Crosswise size (in elements).
template <int ElementSize, int Crosswise>
struct TensorOpMultiplicandCongruous {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = PitchLinearCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = TensorOpMultiplicand<ElementSize, Crosswise>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;
  using TileShape = typename Base::TileShape;
  using PartitionShape = typename Base::PartitionShape;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;
  static int const kCrosswise = Base::kCrosswise;
  static int const kFactor = Base::kFactor;
  using PartitionCount =  typename Base::PartitionCount;
  using AccessCount = typename Base::AccessCount;

 private:
  //
  // Data members
  //

  Base layout_;

 public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandCongruous(Index ldm = 0) : layout_(ldm) {}

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandCongruous(Stride stride) : layout_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static TensorOpMultiplicandCongruous packed(TensorCoord const &extent) {
    return TensorOpMultiplicandCongruous(extent[0]);
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(coord);
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  CUTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return coord;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const { return layout_.stride(); }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride &stride() { return layout_.stride(); }

  /// Compute the number of contiguous elements needed to store a tensor with
  /// the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(extent);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template based on element size (in bits) - defined in terms of pitch-linear
/// memory and Crosswise size (in elements).
/// This one is just for TF32 NT kernel.
template <int Crosswise>
struct TensorOpMultiplicandCongruous<32, Crosswise> {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = PitchLinearCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = 128;

  /// Fundamental tile shape in units of vectors
  using TileShape = PitchLinearShape<8, 4>;

  /// Partitionshape is the same as TileShape for this layout
  using PartitionShape = PitchLinearShape<8, 4>;

  using PartitionCount =
      PitchLinearShape<TileShape::kContiguous / PartitionShape::kContiguous,
                       TileShape::kStrided / PartitionShape::kStrided>;

  using AccessCount =
      PitchLinearShape<PartitionShape::kContiguous, PartitionShape::kStrided>;

  //
  // Static constants
  //
  static int const kElementSize = 32;
  static int const kElementsPerAccess = kAccessSize / kElementSize;
  static int const kCrosswise = Crosswise;
  static int const kFactor = 1;

 private:
  //
  // Data members
  //

  /// Stride data member.
  Stride stride_;

 public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandCongruous(Index ldm = 0) : stride_(ldm) {}

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandCongruous(Stride stride) : stride_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static TensorOpMultiplicandCongruous packed(TensorCoord const &extent) {
    return TensorOpMultiplicandCongruous(extent[0]);
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    int tc = coord.contiguous() / 32;
    int ts = coord.strided() / 4;

    int c = (coord.contiguous() % 32) / kElementsPerAccess;
    int s = coord.strided() % 4;

    LongIndex offset = (c ^ (2 * s)) * kElementsPerAccess + s * stride_[0] +
                       tc * 32 + ts * stride_[0] * 4 + coord.contiguous() % 4;

    return offset;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const { return stride_; }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride &stride() { return stride_; }

  /// Compute the number of contiguous elements needed to store a tensor with
  /// the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return extent[1] * stride_[0];
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template mapping a column-major view of pitch-linear memory to
/// TensorOpMultiplicand
template <int ElementSize, int Crosswise>
struct ColumnMajorTensorOpMultiplicandCongruous {

  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = MatrixCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = TensorOpMultiplicandCongruous<ElementSize, Crosswise>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;
  using TileShape = typename Base::TileShape;
  using PartitionShape = typename Base::PartitionShape;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;
  static int const kCrosswise = Base::kCrosswise;
  static int const kFactor = Base::kFactor;
  using PartitionCount =  typename Base::PartitionCount;
  using AccessCount = typename Base::AccessCount;

private:

  //
  // Data members
  //

  Base layout_;

public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  ColumnMajorTensorOpMultiplicandCongruous(Index ldm = 0): layout_(ldm) { }

  /// Ctor
  CUTLASS_HOST_DEVICE
  ColumnMajorTensorOpMultiplicandCongruous(Stride stride): layout_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static ColumnMajorTensorOpMultiplicandCongruous packed(TensorCoord const &extent) {
    return ColumnMajorTensorOpMultiplicandCongruous(extent.row());
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(PitchLinearCoord(coord.row(), coord.column()));
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  CUTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return MatrixCoord(coord.contiguous(), coord.strided());    
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const {
    return layout_.stride();
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride & stride() {
    return layout_.stride();
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(PitchLinearCoord(extent.row(), extent.column()));
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template mapping a row-major view of pitch-linear memory to
/// TensorOpMultiplicand
template <int ElementSize, int Crosswise>
struct RowMajorTensorOpMultiplicandCongruous {

  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = MatrixCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = TensorOpMultiplicandCongruous<ElementSize, Crosswise>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;
  using TileShape = typename Base::TileShape;
  using PartitionShape = typename Base::PartitionShape;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;
  static int const kCrosswise = Base::kCrosswise;
  static int const kFactor = Base::kFactor;
  using PartitionCount =  typename Base::PartitionCount;
  using AccessCount = typename Base::AccessCount;

private:

  //
  // Data members
  //

  Base layout_;

public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  RowMajorTensorOpMultiplicandCongruous(Index ldm = 0): layout_(ldm) { }

  /// Ctor
  CUTLASS_HOST_DEVICE
  RowMajorTensorOpMultiplicandCongruous(Stride stride): layout_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static RowMajorTensorOpMultiplicandCongruous packed(TensorCoord const &extent) {
    return RowMajorTensorOpMultiplicandCongruous(extent.column());
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(PitchLinearCoord(coord.column(), coord.row()));
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  CUTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return MatrixCoord(coord.strided(), coord.contiguous());
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const {
    return layout_.stride();
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride & stride() {
    return layout_.stride();
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(PitchLinearCoord(extent.column(), extent.row()));
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template based on element size (in bits) - defined in terms of pitch-linear
/// memory and Crosswise size (in elements).
template <int ElementSize, int Crosswise>
struct TensorOpMultiplicandCrosswise {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = PitchLinearCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = TensorOpMultiplicand<ElementSize, Crosswise>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;
  using TileShape = typename Base::TileShape;
  using PartitionShape = typename Base::PartitionShape;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;
  static int const kCrosswise = Base::kCrosswise;
  static int const kFactor = Base::kFactor;
  using PartitionCount =  typename Base::PartitionCount;
  using AccessCount = typename Base::AccessCount;

 private:
  //
  // Data members
  //

  Base layout_;

 public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandCrosswise(Index ldm = 0) : layout_(ldm) {}

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandCrosswise(Stride stride) : layout_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static TensorOpMultiplicandCrosswise packed(TensorCoord const &extent) {
    return TensorOpMultiplicandCrosswise(extent[0]);
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(coord);
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  CUTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return coord;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const { return layout_.stride(); }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride &stride() { return layout_.stride(); }

  /// Compute the number of contiguous elements needed to store a tensor with
  /// the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(extent);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template mapping a column-major view of pitch-linear memory to
/// TensorOpMultiplicandCrosswise
template <int ElementSize, int Crosswise>
struct ColumnMajorTensorOpMultiplicandCrosswise {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = MatrixCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = TensorOpMultiplicandCrosswise<ElementSize, Crosswise>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;
  using TileShape = typename Base::TileShape;
  using PartitionShape = typename Base::PartitionShape;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;
  using PartitionCount = typename Base::PartitionCount;
  using AccessCount = typename Base::AccessCount;

 private:
  //
  // Data members
  //

  Base layout_;

 public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  ColumnMajorTensorOpMultiplicandCrosswise(Index ldm = 0) : layout_(ldm) {}

  /// Ctor
  CUTLASS_HOST_DEVICE
  ColumnMajorTensorOpMultiplicandCrosswise(Stride stride) : layout_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static ColumnMajorTensorOpMultiplicandCrosswise packed(
      TensorCoord const &extent) {
    return ColumnMajorTensorOpMultiplicandCrosswise(extent.row());
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(PitchLinearCoord(coord.row(), coord.column()));
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  CUTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return MatrixCoord(coord.contiguous(), coord.strided());
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const { return layout_.stride(); }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride &stride() { return layout_.stride(); }

  /// Compute the number of contiguous elements needed to store a tensor with
  /// the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(PitchLinearCoord(extent.row(), extent.column()));
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template mapping a row-major view of pitch-linear memory to
/// TensorOpMultiplicandCrosswise
template <int ElementSize, int Crosswise>
struct RowMajorTensorOpMultiplicandCrosswise {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = MatrixCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = TensorOpMultiplicandCrosswise<ElementSize, Crosswise>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;
  using TileShape = typename Base::TileShape;
  using PartitionShape = typename Base::PartitionShape;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;
  using PartitionCount = typename Base::PartitionCount;
  using AccessCount = typename Base::AccessCount;

 private:
  //
  // Data members
  //

  Base layout_;

 public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  RowMajorTensorOpMultiplicandCrosswise(Index ldm = 0) : layout_(ldm) {}

  /// Ctor
  CUTLASS_HOST_DEVICE
  RowMajorTensorOpMultiplicandCrosswise(Stride stride) : layout_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static RowMajorTensorOpMultiplicandCrosswise packed(
      TensorCoord const &extent) {
    return RowMajorTensorOpMultiplicandCrosswise(extent.column());
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(PitchLinearCoord(coord.column(), coord.row()));
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  CUTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return MatrixCoord(coord.strided(), coord.contiguous());
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const { return layout_.stride(); }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride &stride() { return layout_.stride(); }

  /// Compute the number of contiguous elements needed to store a tensor with
  /// the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(PitchLinearCoord(extent.column(), extent.row()));
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template based on element size (in bits) - defined in terms of pitch-linear memory.
template <int ElementSize, int InterleavedK>
struct TensorOpMultiplicandColumnMajorInterleaved {

  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = PitchLinearCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = 128;

  //
  // Static constants
  //

  static int const kElementSize = ElementSize;
  static int const kElementsPerAccess = kAccessSize / kElementSize;

  //static int const kThreadBlockStrided = ThreadBlockStrided;
  static int const kInterleavedK = InterleavedK;
  
private:

  //
  // Data members
  //

  /// Stride data member
  Stride stride_;

public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandColumnMajorInterleaved(Index ldm = 0): stride_(ldm) { }

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandColumnMajorInterleaved(Stride stride): stride_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static TensorOpMultiplicandColumnMajorInterleaved packed(TensorCoord const &extent) {
    return TensorOpMultiplicandColumnMajorInterleaved(extent[0] * kInterleavedK);
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    int const rows_per_smem_cache_line = 128 / kInterleavedK;

    int row_id = coord.strided() / rows_per_smem_cache_line;
    int col_id = (coord.strided() % rows_per_smem_cache_line) * kInterleavedK + coord.contiguous();

    int access_block_id = col_id >> 4;
    int swizzle_access_block_id = access_block_id ^ (row_id & 1);

    int swizzle_col_id = swizzle_access_block_id << 4;

    return row_id * 128 + swizzle_col_id;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const {
    return stride_;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride & stride() {
    return stride_;
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return (extent[1] / kInterleavedK) * stride_[0];
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template based on element size (in bits) - defined in terms of pitch-linear memory.
template <int ElementSize, int InterleavedK>
struct TensorOpMultiplicandRowMajorInterleaved {

  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = PitchLinearCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = 128;

  //
  // Static constants
  //

  static int const kElementSize = ElementSize;
  static int const kElementsPerAccess = kAccessSize / kElementSize;

  //static int const kThreadBlockStrided = ThreadBlockStrided;
  static int const kInterleavedK = InterleavedK;
  
private:

  //
  // Data members
  //

  /// Stride data member
  Stride stride_;

public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandRowMajorInterleaved(Index ldm = 0): stride_(ldm) { }

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandRowMajorInterleaved(Stride stride): stride_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static TensorOpMultiplicandRowMajorInterleaved packed(TensorCoord const &extent) {
    return TensorOpMultiplicandRowMajorInterleaved(extent[1] * kInterleavedK);
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    int const rows_per_smem_cache_line = 128 / kInterleavedK;

    int row_id = coord.strided() / rows_per_smem_cache_line;
    int col_id = (coord.strided() % rows_per_smem_cache_line) * kInterleavedK + coord.contiguous();

    int access_block_id = col_id >> 4;
    int swizzle_access_block_id = access_block_id ^ (row_id & 1);

    int swizzle_col_id = swizzle_access_block_id << 4;

    return row_id * 128 + swizzle_col_id;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const {
    return stride_;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride & stride() {
    return stride_;
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return (extent[0] / kInterleavedK) * stride_[0];
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace layout
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
