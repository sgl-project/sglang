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
    \brief Templates implementing computing the addresses of storing of small
   scale and bias vectors in the shared memory.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// RegularScaleBiasVectorAccessIterator
///
template <typename Shape, typename Element, typename Layout>
class RegularScaleBiasVectorAccessIterator;

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator specialized for congruous arrangements for TensorOps
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_>
class RegularScaleBiasVectorAccessIterator<Shape_, Element_, layout::PitchLinear> {
 public:

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  /// Element type per access
  static int const kElementsPerAccess = 128 / sizeof_bits<Element>::value;
  static int const kThreads = Shape::kContiguous / kElementsPerAccess;
  using AccessType = Array<Element, kElementsPerAccess>;

 private:
  //
  // Data members
  //

  /// Internal pointer 
  AccessType *pointer_;

  /// Internal byte offset
  Index byte_offset_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularScaleBiasVectorAccessIterator(
      TensorRef scale_bias_ref,  ///< Pointer to the start of the scale and bias
                                 ///< vector
      int thread_id              ///< ID of each participating thread
      )
      : byte_offset_(0) {
    // Per-thread offset in logical coordinates of tensor
    int thread_offset = thread_id * kElementsPerAccess;

    // initialize pointer
    pointer_ =
        reinterpret_cast<AccessType *>(scale_bias_ref.data() + thread_offset);

    set_iteration_index(0);
  }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_offset_ += pointer_offset * sizeof(Element);
  }

  /// Returns a pointer
  CUTLASS_DEVICE
  AccessType *get() const {

    char *access_byte_ptr =
        reinterpret_cast<char *>(pointer_);

    return reinterpret_cast<AccessType *>(access_byte_ptr + byte_offset_);
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularScaleBiasVectorAccessIterator &operator++() { return *this; }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularScaleBiasVectorAccessIterator operator++(int) {
    RegularScaleBiasVectorAccessIterator prev(*this);
    this->operator++();

    return prev;
  }

  /// Adds a tile offset in the unit of tile.
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    // Multiply by 2 because we store scale and bias belong to the same stage
    // next to each other.
    add_pointer_offset(coord.contiguous() * Shape::kContiguous * 2);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator specialized for row major layouts
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_>
class RegularScaleBiasVectorAccessIterator<
    Shape_, Element_,
    layout::RowMajor> {
 public:

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajor;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  /// Underlying iterator type
  using UnderlyingIterator = RegularScaleBiasVectorAccessIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
      layout::PitchLinear>;

  using AccessType = typename UnderlyingIterator::AccessType;

 private:

  /// Underlying iterator
  UnderlyingIterator iterator_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularScaleBiasVectorAccessIterator(
      TensorRef scale_bias_ref,  ///< Pointer to the start of the scale and bias
                                 ///< vector
      int thread_id              ///< ID of each participating thread
      )
      : iterator_({scale_bias_ref.data(), scale_bias_ref.stride()}, thread_id) {
  }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularScaleBiasVectorAccessIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularScaleBiasVectorAccessIterator operator++(int) {
    RegularScaleBiasVectorAccessIterator prev(*this);
    ++iterator_;

    return prev;
  }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace transform 
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
