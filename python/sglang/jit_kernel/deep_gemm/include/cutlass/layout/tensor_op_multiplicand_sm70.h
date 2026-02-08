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
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/matrix_coord.h" // cutlass::MatrixCoord

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace layout {

// template <
//   int ElementSize,
//   gemm::Operand Operand
// >
// struct VoltaTensorOpMultiplicandCongruous;

// template <
//   int ElementSize,
//   gemm::Operand Operand
// >
// struct ColumnMajorVoltaTensorOpMultiplicandCongruous;
// template <
//   int ElementSize,
//   gemm::Operand Operand
// >
// struct RowMajorVoltaTensorOpMultiplicandCongruous;
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template based on element size (in bits) - defined in terms of pitch-linear memory.
template <int ElementSize>
struct VoltaTensorOpMultiplicandCongruous {

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

  /// Fundamental partition shape in units of vectors
  using PartitionShape = PitchLinearShape<8, 2>;

  //
  // Static constants
  //

  static int const kElementSize = ElementSize;
  static int const kElementsPerAccess = kAccessSize / kElementSize;
  
  using PartitionCount = PitchLinearShape<
    TileShape::kContiguous / PartitionShape::kContiguous,
    TileShape::kStrided / PartitionShape::kStrided
  >;

  using AccessCount = PitchLinearShape<
    PartitionShape::kContiguous,
    PartitionShape::kStrided
  >;

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
  VoltaTensorOpMultiplicandCongruous(Index ldm = 0): stride_(ldm) { }

  /// Ctor
  CUTLASS_HOST_DEVICE
  VoltaTensorOpMultiplicandCongruous(Stride stride): stride_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static VoltaTensorOpMultiplicandCongruous packed(TensorCoord const &extent) {
    return VoltaTensorOpMultiplicandCongruous(extent[0]);
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    
    // First, compute c and s of vector within source (in units of vector accesses)
    int vec_contiguous_idx = coord.contiguous() / kElementsPerAccess;
    int vec_strided_idx = coord.strided();

    // Compute the fundamental tile being accessed
    int tile_contiguous_idx = vec_contiguous_idx / TileShape::kContiguous;
    int tile_strided_idx = vec_strided_idx / TileShape::kStrided;

    int tile_contiguous_residual = vec_contiguous_idx % TileShape::kContiguous;
    int tile_strided_residual = vec_strided_idx % TileShape::kStrided;

    // Then swizzle in a tile
    // Swizzle pattern is (tid[2:0] << 2)|(tid[4:3] ^ tid[2:1])
    int permuted_strided_within_tile = (tile_contiguous_residual >> 1);
    int permuted_contiguous_within_tile = (tile_strided_residual ^ permuted_strided_within_tile) |
                                       ((tile_contiguous_residual & 1) << 2);
    // Compute final element location
    int element_contiguous = (tile_contiguous_idx * TileShape::kContiguous +
        permuted_contiguous_within_tile) * kElementsPerAccess + (coord.contiguous() % kElementsPerAccess);

    int element_strided = tile_strided_idx * TileShape::kStrided + permuted_strided_within_tile;

    return element_contiguous + element_strided * stride_[0];
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
    return extent[1] * stride_[0];
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template mapping a column-major view of pitch-linear memory to VoltaTensorOpMultiplicandCongruous
template <int ElementSize>
struct ColumnMajorVoltaTensorOpMultiplicandCongruous {

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

  using Base = VoltaTensorOpMultiplicandCongruous<ElementSize>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;
  using TileShape = typename Base::TileShape;
  using PartitionShape = typename Base::PartitionShape;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;
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
  ColumnMajorVoltaTensorOpMultiplicandCongruous(Index ldm = 0): layout_(ldm) { }

  /// Ctor
  CUTLASS_HOST_DEVICE
  ColumnMajorVoltaTensorOpMultiplicandCongruous(Stride stride): layout_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static ColumnMajorVoltaTensorOpMultiplicandCongruous packed(TensorCoord const &extent) {
    return ColumnMajorVoltaTensorOpMultiplicandCongruous(extent.row());
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

/// Template mapping a row-major view of pitch-linear memory to VoltaTensorOpMultiplicandCongruous
template <int ElementSize>
struct RowMajorVoltaTensorOpMultiplicandCongruous {

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

  using Base = VoltaTensorOpMultiplicandCongruous<ElementSize>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;
  using TileShape = typename Base::TileShape;
  using PartitionShape = typename Base::PartitionShape;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;
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
  RowMajorVoltaTensorOpMultiplicandCongruous(Index ldm = 0): layout_(ldm) { }

  /// Ctor
  CUTLASS_HOST_DEVICE
  RowMajorVoltaTensorOpMultiplicandCongruous(Stride stride): layout_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static RowMajorVoltaTensorOpMultiplicandCongruous packed(TensorCoord const &extent) {
    return RowMajorVoltaTensorOpMultiplicandCongruous(extent.column());
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


/// Template based on element size (in bits) - defined in terms of pitch-linear memory.
// template <int ElementSize, Operand Operand>
template <int ElementSize>
struct VoltaTensorOpMultiplicandBCongruous {
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

  /// Fundamental partition shape in units of vectors
  using PartitionShape = PitchLinearShape<4, 4>;

  //
  // Static constants
  //

  static int const kElementSize = ElementSize;
  static int const kElementsPerAccess = kAccessSize / kElementSize;
  
  using PartitionCount = PitchLinearShape<
    TileShape::kContiguous / PartitionShape::kContiguous,
    TileShape::kStrided / PartitionShape::kStrided
  >;

  using AccessCount = PitchLinearShape<
    PartitionShape::kContiguous,
    PartitionShape::kStrided
  >;

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
  VoltaTensorOpMultiplicandBCongruous(Index ldm = 0): stride_(ldm) { }

  /// Ctor
  CUTLASS_HOST_DEVICE
  VoltaTensorOpMultiplicandBCongruous(Stride stride): stride_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static VoltaTensorOpMultiplicandBCongruous packed(TensorCoord const &extent) {
    return VoltaTensorOpMultiplicandBCongruous(extent[0]);
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    
    // First, compute c and s of vector within source (in units of vector accesses)
    int vec_contiguous_idx = coord.contiguous() / kElementsPerAccess;
    int vec_strided_idx = coord.strided();

    // Compute the fundamental tile being accessed
    int tile_contiguous_idx = vec_contiguous_idx / TileShape::kContiguous;
    int tile_strided_idx = vec_strided_idx / TileShape::kStrided;

    int tile_contiguous_residual = vec_contiguous_idx % TileShape::kContiguous;
    int tile_strided_residual = vec_strided_idx % TileShape::kStrided;

    // Then swizzle in a tile
    // Swizzle pattern is (tid[1:0] << 3)|(tid & 0x4)|(tid[1:0])
    int permuted_strided_within_tile = (tile_contiguous_residual & 0x3);
    int permuted_contiguous_within_tile = (tile_strided_residual ^ permuted_strided_within_tile) |
                                       (tile_contiguous_residual & 0x4);
  
    // Compute final element location
    int element_contiguous = (tile_contiguous_idx * TileShape::kContiguous +
        permuted_contiguous_within_tile) * kElementsPerAccess + (coord.contiguous() % kElementsPerAccess);

    int element_strided = tile_strided_idx * TileShape::kStrided + permuted_strided_within_tile;

    return element_contiguous + element_strided * stride_[0];
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
    return extent[1] * stride_[0];
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template mapping a column-major view of pitch-linear memory to VoltaTensorOpMultiplicandCongruous
template <int ElementSize>
struct ColumnMajorVoltaTensorOpMultiplicandBCongruous {

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

  using Base = VoltaTensorOpMultiplicandBCongruous<ElementSize>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;
  using TileShape = typename Base::TileShape;
  using PartitionShape = typename Base::PartitionShape;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;
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
  ColumnMajorVoltaTensorOpMultiplicandBCongruous(Index ldm = 0): layout_(ldm) { }

  /// Ctor
  CUTLASS_HOST_DEVICE
  ColumnMajorVoltaTensorOpMultiplicandBCongruous(Stride stride): layout_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static ColumnMajorVoltaTensorOpMultiplicandBCongruous packed(TensorCoord const &extent) {
    return ColumnMajorVoltaTensorOpMultiplicandBCongruous(extent.row());
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

/// Template mapping a row-major view of pitch-linear memory to VoltaTensorOpMultiplicandCongruous
template <int ElementSize>
struct RowMajorVoltaTensorOpMultiplicandBCongruous {

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

  using Base = VoltaTensorOpMultiplicandBCongruous<ElementSize>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;
  using TileShape = typename Base::TileShape;
  using PartitionShape = typename Base::PartitionShape;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;
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
  RowMajorVoltaTensorOpMultiplicandBCongruous(Index ldm = 0): layout_(ldm) { }

  /// Ctor
  CUTLASS_HOST_DEVICE
  RowMajorVoltaTensorOpMultiplicandBCongruous(Stride stride): layout_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static RowMajorVoltaTensorOpMultiplicandBCongruous packed(TensorCoord const &extent) {
    return RowMajorVoltaTensorOpMultiplicandBCongruous(extent.column());
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

/// Template based on element size (in bits) - defined in terms of pitch-linear
/// memory and KBlock size (in elements).
template <int ElementSize, int KBlock>
struct VoltaTensorOpMultiplicandCrosswise {
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

  /// This layout is optimized for 64b accesses
  static int const kAccessSize = 64;

  //
  // Static constants
  //

  static int const kElementSize = ElementSize;
  static int const kElementsPerAccess = kAccessSize / kElementSize;
  static int const kKBlock = KBlock;

 private:
  //
  // Data members
  //

  /// Stride data member. For GEMM, it equals to KBlock x stage.
  Stride stride_;
 public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  VoltaTensorOpMultiplicandCrosswise(Index ldm = 0) : stride_(ldm) {}

  /// Ctor
  CUTLASS_HOST_DEVICE
  VoltaTensorOpMultiplicandCrosswise(Stride stride) : stride_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static VoltaTensorOpMultiplicandCrosswise packed(TensorCoord const &extent) {
    return VoltaTensorOpMultiplicandCrosswise(extent[1]);
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
    int vec_strided_idx = coord.strided();

    //
    // Then swizzle
    // The mapping is like this:
    // id[1:0]|(id[3]^id[4])|id[2]

    int vec_strided_within_tile = vec_contiguous_idx & 0x7;
    int permuted_vec_contiguous =
        (vec_strided_idx & (~0xF)) + (vec_strided_idx & 0x3) * 4 +
        (((vec_strided_idx >> 2) ^ ((vec_strided_idx & 0x10) >> 3)) & 0x3);

    permuted_vec_contiguous ^= ((vec_strided_within_tile >> 1) & 0x3);

    int permuted_vec_strided = vec_contiguous_idx;

    //
    // Compute final element location
    //

    int element_contiguous = permuted_vec_contiguous *  kElementsPerAccess + 
                             (coord.contiguous() % kElementsPerAccess);
    
    return element_contiguous + permuted_vec_strided * (stride_[0] * kElementsPerAccess);
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
    return extent[0] * stride_[0];
  }
};

/// Template mapping a column-major view of pitch-linear memory to
/// VoltaTensorOpMultiplicandCrosswise
template <int ElementSize, int KBlock>
struct ColumnMajorVoltaTensorOpMultiplicandCrosswise {
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

  using Base = VoltaTensorOpMultiplicandCrosswise<ElementSize, KBlock>;

  /// This layout is optimized for 64b accesses
  static int const kAccessSize = Base::kAccessSize;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;

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
  ColumnMajorVoltaTensorOpMultiplicandCrosswise(Index ldm = 0) : layout_(ldm) {}

  /// Ctor
  CUTLASS_HOST_DEVICE
  ColumnMajorVoltaTensorOpMultiplicandCrosswise(Stride stride) : layout_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static ColumnMajorVoltaTensorOpMultiplicandCrosswise packed(
      TensorCoord const &extent) {
    return ColumnMajorVoltaTensorOpMultiplicandCrosswise(extent.column());
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

/// Template mapping a row-major view of pitch-linear memory to
/// TensorOpMultiplicandCrosswise
template <int ElementSize, int KBlock>
struct RowMajorVoltaTensorOpMultiplicandCrosswise {
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

  using Base = VoltaTensorOpMultiplicandCrosswise<ElementSize, KBlock>;

  /// This layout is optimized for 64b accesses
  static int const kAccessSize = Base::kAccessSize;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;

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
  RowMajorVoltaTensorOpMultiplicandCrosswise(Index ldm = 0) : layout_(ldm) {}

  /// Ctor
  CUTLASS_HOST_DEVICE
  RowMajorVoltaTensorOpMultiplicandCrosswise(Stride stride) : layout_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static RowMajorVoltaTensorOpMultiplicandCrosswise packed(
      TensorCoord const &extent) {
    return RowMajorVoltaTensorOpMultiplicandCrosswise(extent.row());
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

} // namespace layout
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
