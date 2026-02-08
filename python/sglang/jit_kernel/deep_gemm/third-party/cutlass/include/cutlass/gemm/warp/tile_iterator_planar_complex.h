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
    \brief Templates implementing warp-level matrix multiply-accumulate operations.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/array_planar_complex.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TileIterator_>
class TileIteratorPlanarComplex {
public:

  /// Underlying iterator over real-valued tiles
  using TileIterator = TileIterator_;

  /// Underlying element type
  using Element = typename TileIterator::Element;

  /// Underlying layout type
  using Layout = typename TileIterator::Layout;

  /// TensorRef type for loading element from a tensor
  using TensorRef = typename TileIterator::TensorRef;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Planar complex fragment
  using Fragment = ArrayPlanarComplex<Element, TileIterator::Fragment::kElements>;

public:

  /// Underlying tile iterator
  TileIterator tile_iterator_;

  /// Offset (in units of bytes) to the imaginary part of the planar complex matrix
  LongIndex imaginary_offset_;

public:
    /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  TileIteratorPlanarComplex(): imaginary_offset_(0) { }

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  TileIteratorPlanarComplex(
    TensorRef const &ref, 
    int lane_id,
    LongIndex imaginary_offset
  ):
    tile_iterator_(ref, lane_id),
    imaginary_offset_((imaginary_offset * sizeof_bits<Element>::value) / 8) { }


  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_DEVICE
  TileIteratorPlanarComplex &add_pointer_offset(LongIndex offset) {

    tile_iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  TileIteratorPlanarComplex &add_tile_offset(TensorCoord const &tile_offset) {

    tile_iterator_.add_tile_offset(tile_offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  TileIteratorPlanarComplex & operator++() {
    ++tile_iterator_;
    return *this;
  }

  //
  // WIP
  //

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  TileIteratorPlanarComplex & operator--() {
    --tile_iterator_;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  TileIteratorPlanarComplex & operator+=(TensorCoord const &tile_offset) {
    tile_iterator_.add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  TileIteratorPlanarComplex & operator-=(TensorCoord const &tile_offset) {
    tile_iterator_.add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    tile_iterator_.load_with_byte_offset(frag.real, 0);
    tile_iterator_.load_with_byte_offset(frag.imag, imaginary_offset_);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    tile_iterator_.load_with_byte_offset(frag.real, byte_offset);
    tile_iterator_.load_with_byte_offset(frag.imag, byte_offset + imaginary_offset_);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {

    Index byte_offset = (pointer_offset * sizeof_bits<Element>::value)/8;

    tile_iterator_.load_with_byte_offset(frag.real, byte_offset);
    tile_iterator_.load_with_byte_offset(frag.imag, byte_offset + imaginary_offset_);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {

    tile_iterator_.load_with_byte_offset(frag.real, tile_offset, 0);
    tile_iterator_.load_with_byte_offset(frag.imag, tile_offset, imaginary_offset_);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {

    Index byte_offset = (pointer_offset * sizeof_bits<Element>::value)/8;

    tile_iterator_.load_with_byte_offset(frag.real, tile_offset, byte_offset);
    tile_iterator_.load_with_byte_offset(frag.real, tile_offset, byte_offset + imaginary_offset_);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {

    tile_iterator_.load_with_byte_offset(frag.real, tile_offset, byte_offset);
    tile_iterator_.load_with_byte_offset(frag.imag, tile_offset, byte_offset + imaginary_offset_);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    tile_iterator_.set_kgroup_index(k_group);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
