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
    \brief Templates implementing computing the addresses of loading small
    vectors from the global memory.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// PredicatedVectorAccessIterator
///
template <
    /// Shape of the vector accessed by the entire threadblock
    typename Shape,
    /// Shape of the vector accessed by the warp
    typename WarpShape,
    /// Type of Element
    typename Element,
    /// Layout of the vector
    typename Layout,
    /// Number of elements for each access
    int ElementsPerAccess,
    /// Support residual tile
    bool EnableResidualAccess = false
>
class PredicatedVectorAccessIterator;

////////////////////////////////////////////////////////////////////////////////

/// Vector access iterator specialized for vectors, e.g. scale and bias
/// Thread arrangements are for TensorOps
///
template <
  typename Shape_, 
  typename WarpShape_, 
  typename Element_, 
  int ElementsPerAccess, 
  bool EnableResidualAccess
>
class PredicatedVectorAccessIterator <
  Shape_,
  WarpShape_,
  Element_,
  layout::PitchLinear,
  ElementsPerAccess,
  EnableResidualAccess
> {
  public:

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ConstPointer = const Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

//  static int const kElementsPerAccess = 128 / sizeof_bits<Element>::value;
  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kThreads = 32;
  static int const kRowsPerIteration = 8;
  static int const kThreadsPerRow = kThreads / kRowsPerIteration;
  static int const kThreadsPerRowMask = 0x3;
  static int const kIterations = WarpShape::kContiguous / (kThreadsPerRow * kElementsPerAccess); 
  static int const kWarpCountStrided = Shape::kStrided / WarpShape::kStrided;

  using AccessType = AlignedArray<Element, kElementsPerAccess>;

 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char *;

 private:
  //
  // Data members
  //

  /// Internal pointer to first access of tile
  BytePointer pointer_;

  /// Extent of tensor
  TensorCoord extent_;

  /// pointer offset of each thread
  TensorCoord thread_offset_;

  /// iteration index
  LongIndex iteration_;

  /// residual access
  bool is_residual_;

  /// residual offset of each thread
  TensorCoord residual_offset_;

 public:
  /// Constructs a vector access iterator
  CUTLASS_HOST_DEVICE
  PredicatedVectorAccessIterator(
    /// Pointer to the start of the vector
    ConstPointer pointer,
    /// Extent of vector
    TensorCoord extent,
    /// ID of each participating thread
    int thread_id,
    /// ID of each participating warp
    int warp_id,
    /// Initial offset of threadblock
    TensorCoord const &threadblock_offset)
    : pointer_(reinterpret_cast<BytePointer>(
                       const_cast<NonConstPointer>(pointer))),
      extent_(extent),
      is_residual_(false) {


    int warp_offset = (warp_id / kWarpCountStrided) * WarpShape::kContiguous;

    // Per-thread offset in logical coordinates of tensor

    thread_offset_ = threadblock_offset + TensorCoord(warp_offset, 0) +
        TensorCoord((thread_id & kThreadsPerRowMask) * kElementsPerAccess, 0);

    set_iteration_index(0);

    if(EnableResidualAccess) {
      // compute residual offset
      typename TensorCoord::Index residual_size = extent_.contiguous() % WarpShape::kContiguous;
      if (residual_size) {
        is_residual_ = true;
        residual_offset_ = make_Coord(residual_size, 0);
      }
    }
  }

  /// Construct a PredicatedVectorAccessIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedVectorAccessIterator(
    /// Pointer to start of vector
    ConstPointer pointer,
    /// Extent of vector
    TensorCoord extent,
    ///< ID of each participating thread
    int thread_id,
    /// ID of each participating warp
    int warp_id)
    : PredicatedVectorAccessIterator(pointer, extent, thread_id, warp_id,
                                     make_Coord(0, 0)) {}


  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iteration_ = index;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_DEVICE
  void add_tile_offset(
      TensorCoord const &tile_offset) {

    thread_offset_ =
        thread_offset_ +
        TensorCoord(WarpShape::kContiguous * tile_offset.contiguous(), 0);
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {

    return reinterpret_cast<AccessType *>(
        pointer_ +
        ((thread_offset_.contiguous() + iteration_ * kThreadsPerRow * kElementsPerAccess) 
        * sizeof_bits<Element>::value / 8));
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  PredicatedVectorAccessIterator &operator++() {
    ++iteration_;
    if(iteration_ >= kIterations)
      iteration_ = 0; 

    return *this;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  void advance() {
    if(EnableResidualAccess && is_residual_) {
      is_residual_ = false;
      thread_offset_ += residual_offset_; 
    }
    else
      add_tile_offset(TensorCoord(1, 0));
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  PredicatedVectorAccessIterator operator++(int) {
    PredicatedVectorAccessIterator self(*this);
    operator++();
    return self;
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return ((thread_offset_.contiguous() + 
              iteration_ * kThreadsPerRow * kElementsPerAccess) < extent_.contiguous());
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedVectorAccessIterator for row-major data.
///
template <
  typename Shape_,
  typename WarpShape_,
  typename Element_,
  int ElementsPerAccess,
  bool EnableResidualAccess
>
class PredicatedVectorAccessIterator<
  Shape_,
  WarpShape_,
  Element_,
  layout::RowMajor,
  ElementsPerAccess,
  EnableResidualAccess
> {
 public:

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using Element = Element_;
  using Layout = layout::RowMajor;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ConstPointer = const Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PredicatedVectorAccessIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, 
      layout::PitchLinearShape<WarpShape::kColumn, WarpShape::kRow>, 
      Element,
      layout::PitchLinear,
      ElementsPerAccess,
      EnableResidualAccess>;

  using AccessType = typename UnderlyingIterator::AccessType;
  static int const kElementsPerAccess = UnderlyingIterator::kElementsPerAccess;
  static int const kRowsPerIteration = UnderlyingIterator::kRowsPerIteration;
  static int const kThreads = UnderlyingIterator::kThreads;
  static int const kIterations = UnderlyingIterator::kIterations;

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedVectorAccessIterator(
      ///< Pointer to the start of the vector
      ConstPointer pointer,
      ///< Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      ///< ID of each participating warp
      int warp_id,
      ///< Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : iterator_(pointer, layout::PitchLinearCoord(extent.column(), extent.row()),
                  thread_id, warp_id,
                  layout::PitchLinearCoord(threadblock_offset.column(),
                                           threadblock_offset.row())) {}

  /// Construct a PredicatedVectorAccessIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedVectorAccessIterator(
      ConstPointer pointer,   ///< Pointer to the start of the vector
      TensorCoord extent,     ///< Extent of tensor
      int thread_id,          ///< ID of each participating thread
      int warp_id             ///< ID of each participating warp
      )
      : PredicatedVectorAccessIterator(pointer, extent, thread_id, warp_id, 
                                        make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedVectorAccessIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedVectorAccessIterator operator++(int) {
    PredicatedVectorAccessIterator self(*this);
    operator++();
    return self;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  void advance() {
    iterator_.advance();
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return iterator_.valid();
  }
};


////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace transform 
}  // namespace cutlass

