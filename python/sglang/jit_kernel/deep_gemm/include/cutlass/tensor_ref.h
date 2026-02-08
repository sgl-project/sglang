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
    \brief Defines a structure containing strides, bounds, and a pointer to tensor data.
*/
#pragma once


#include "cutlass/cutlass.h"
#include "cutlass/coord.h"
#include "cutlass/platform/platform.h"
#include "cutlass/subbyte_reference.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Default layout function from coordinates in a tensor's index space into the n-D array held
/// in memory.
///
/// All layout functions must define at least the members shown in IdentityTensorLayout<>.
template <int Rank>
class IdentityTensorLayout {
public:
  /// Logical rank of tensor
  static int const kRank = Rank;

  /// Rank of stride vector
  static int const kStrideRank = Rank;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = Coord<kRank, Index>;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index>;

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

  CUTLASS_HOST_DEVICE
  IdentityTensorLayout(Stride const &stride = Stride()): stride_(stride) { }

  /// Returns the offset of a coordinate in linear memory
  CUTLASS_HOST_DEVICE
  LongIndex operator()(Coord<Rank> const &coord) const {
    return coord.dot(stride_);
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
  LongIndex capacity(TensorCoord const &size) const {
    int idx = stride_.max_dim_index();
    return stride_[idx] * size[idx];
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/* \brief TensorRef is a template for objects pointing to the start of tensors of arbitrary rank
          and layout within memory. A TensorRef combines a pointer and a Layout concept

  Examples:

  (These examples use helpers for matrix layouts defined in cutlass/layout/matrix.h)

  1. Column-major matrix may be represented as a rank=2 tensor:

    TensorRef<float, layout::ColumnMajor> A(ptr_A, ldm);

  2. Row-major matrix may be represented as a rank=2 tensor:

    TensorRef<float, layout::RowMajor> B(ptr_A, ldm);

  3. An interleaved matrix may be represented as a rank=2 tensor:

    TensorRef<int8_t, layout::ColumnMajorInterleaved<32> > C;

  4. A helper exists to define a TensorRef for a contiguous matrix whose layout
     is not known at compile time.

    int ldm;                     // leading dimension
    layout::Matrix kind;         // Could be layout::Matrix::kRowMajor or layout::Matrix::kColumnMajor
    

    TensorRef<int, layout::ContiguousMatrix> E(ptr_E, {ldm, kind});

*/
template <
  /// Data type of element stored within tensor (concept: NumericType)
  typename Element_,
  /// Defines a mapping from logical coordinate to linear memory (concept: Layout)
  typename Layout_
>
class TensorRef {
 public:
  /// Data type of individual access
  using Element = Element_;

  /// Mapping function from logical coordinate to linear memory
  using Layout = Layout_;

  /// Reference type to an element
  using Reference = typename platform::conditional<
    sizeof_bits<Element>::value >= 8,
    Element &,
    SubbyteReference<Element>
    >::type;

  /// Logical rank of tensor index space
  static int const kRank = Layout::kRank;

  /// Index type
  using Index = typename Layout::Index;

  /// Long index used for pointer offsets
  using LongIndex = typename Layout::LongIndex;

  /// Coordinate in logical tensor space
  using TensorCoord = typename Layout::TensorCoord;

  /// Layout's stride vector
  using Stride = typename Layout::Stride;

  /// TensorRef to constant data
  using ConstTensorRef = TensorRef<
    typename platform::remove_const<Element>::type const,
    Layout>;

  /// TensorRef to non-constant data
  using NonConstTensorRef = TensorRef<
    typename platform::remove_const<Element>::type,
    Layout>;

  /// Require at least rank=1. Mathematically, a rank=0 tensor would be considered to be a
  /// scalar, but degenerate cases such as these are difficult to accommodate without
  /// extensive C++ metaprogramming or support for zero-length arrays.
  static_assert(kRank > 0, "Cannot define a zero-rank TensorRef");

 private:

  /// Pointer
  Element* ptr_;

  /// Layout object maps logical coordinates to linear offsets
  Layout layout_;

 public:

  //
  // Methods
  //

  /// Constructs a TensorRef with a pointer and layout object.
  CUTLASS_HOST_DEVICE
  TensorRef(): ptr_(nullptr) {
  
  }

  /// Constructs a TensorRef with a pointer and layout object.
  CUTLASS_HOST_DEVICE
  TensorRef(
    Element *ptr,                   ///< pointer to start of tensor
    Layout const &layout            ///< layout object containing stride and mapping function
  ):
    ptr_(ptr), layout_(layout) {
  
  }

  /// Converting constructor from TensorRef to non-constant data.
  template<typename _Magic = int>
  CUTLASS_HOST_DEVICE
  TensorRef(
    NonConstTensorRef const &ref,              ///< TensorRef to non-const data
    ///SFINAE trick to avoid creating a copy-constructor when Element_ is already non-const
    _Magic magic = (typename platform::enable_if< ! platform::is_same<NonConstTensorRef, TensorRef<Element_, Layout_> >::value, _Magic>::type)0
  ):
    ptr_(ref.data()), layout_(ref.layout()) { }

  /// Returns a reference to constant-valued tensor.
  CUTLASS_HOST_DEVICE
  ConstTensorRef const_ref() const {
    return ConstTensorRef(ptr_, layout_);
  }

  CUTLASS_HOST_DEVICE
  NonConstTensorRef non_const_ref() const {
    return NonConstTensorRef(const_cast<typename platform::remove_const<Element>::type *>(ptr_), layout_);
  }

  /// Updates only the pointer
  CUTLASS_HOST_DEVICE
  void reset(Element* ptr = nullptr) {
    ptr_ = ptr;
  }

  /// Updates the pointer and layout object
  CUTLASS_HOST_DEVICE
  void reset(Element* ptr, Layout const &layout) {
    ptr_ = ptr;
    layout_ = layout;
  }

  /// Returns true if the TensorRef is non-null
  CUTLASS_HOST_DEVICE
  bool good() const {
    return ptr_ != nullptr;
  }

  /// Returns the pointer to referenced data
  CUTLASS_HOST_DEVICE
  Element * data() const { return ptr_; }

  /// Returns a reference to the element at a given linear index
  CUTLASS_HOST_DEVICE
  Reference data(LongIndex idx) const {
    return ReferenceFactory<typename platform::remove_const<Element>::type,
                            (sizeof_bits<Element>::value < 8)>::get(ptr_, idx);
  }

  /// Returns the layout object
  CUTLASS_HOST_DEVICE
  Layout & layout() {
    return layout_;
  }

  /// Returns the layout object
  CUTLASS_HOST_DEVICE
  Layout layout() const {
    return layout_;
  }

  /// Returns the layout object's stride vector
  CUTLASS_HOST_DEVICE
  Stride stride() const {
    return layout_.stride();
  }

  /// Returns the layout object's stride vector
  CUTLASS_HOST_DEVICE
  Stride & stride() {
    return layout_.stride();
  }

  /// Returns the layout object's stride in a given physical dimension
  CUTLASS_HOST_DEVICE
  typename Layout::Stride::Index stride(int dim) const {
    return layout_.stride().at(dim);
  }

  /// Returns the layout object's stride in a given physical dimension
  CUTLASS_HOST_DEVICE
  typename Layout::Stride::Index & stride(int dim) {
    return layout_.stride().at(dim);
  }

  /// Computes the offset of an index from the origin of the tensor
  CUTLASS_HOST_DEVICE
  LongIndex offset(TensorCoord const& coord) const {
    return layout_(coord);
  }

  /// Returns a reference to the element at a given Coord
  CUTLASS_HOST_DEVICE
  Reference at(TensorCoord const& coord) const {
    return data(offset(coord));
  }

  /// Returns a reference to the element at a given Coord
  CUTLASS_HOST_DEVICE
  Reference operator[](TensorCoord const& coord) const {
    return data(offset(coord));
  }

  /// Adds an offset to each pointer
  CUTLASS_HOST_DEVICE
  TensorRef & add_pointer_offset(LongIndex offset_) {
    ptr_ = ReferenceFactory<typename platform::remove_const<Element>::type,
           (sizeof_bits<Element>::value < 8)>::add_pointer_offset(ptr_, offset_);
    return *this;
  }

  /// Adds an offset to each pointer
  CUTLASS_HOST_DEVICE
  TensorRef & add_coord_offset(TensorCoord const &coord) {
    add_pointer_offset(offset(coord));
    return *this;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorRef operator+(TensorCoord const& b) const {
    TensorRef result(*this);
    result.add_coord_offset(b);
    return result;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorRef & operator+=(TensorCoord const& b) {
    add_coord_offset(b);
    return *this;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorRef operator-(TensorCoord const& b) const {
    TensorRef result(*this);
    result.add_pointer_offset(-offset(b));
    return result;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorRef & operator-=(TensorCoord const& b) {
    add_pointer_offset(-offset(b));
    return *this;
  }
};

/// Constructs a TensorRef, deducing types from arguments.
template <
  typename Element,
  typename Layout
>
CUTLASS_HOST_DEVICE
TensorRef<Element, Layout> make_TensorRef(Element *ptr, Layout const &layout) {
  return TensorRef<Element, Layout>(ptr, layout);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations to handle degenerate and sub-byte cases.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Element,
  typename Layout
>
CUTLASS_HOST_DEVICE
bool TensorRef_aligned(TensorRef<Element, Layout> const &ref, int alignment) {

  int const kStrideRank = Layout::kStrideRank;

  if (reinterpret_cast<uintptr_t>(ref.data()) % alignment) {
    return false;
  }

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < kStrideRank; ++i) {
    if (ref.stride(i) % alignment) {
      return false;
    }
  }

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
