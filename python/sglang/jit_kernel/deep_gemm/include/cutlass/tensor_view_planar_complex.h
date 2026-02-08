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
    \brief Defines a structure containing strides and a pointer to tensor data.

    TensorView is derived from TensorRef and contributes bounds to the tensor's index space. Thus,
    it is a complete mathematical object and may be used in tensor algorithms. It is decoupled from
    data storage and is therefore lightweight and may be embedded in larger tensor objects or
    memory structures.

    See cutlass/tensor_ref.h for more details about the mapping of the logical tensor index space to
    linear memory.
*/

#pragma once

#if !defined(__CUDACC_RTC__)
#include <cmath>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref_planar_complex.h"
#include "cutlass/tensor_view.h" // cutlass::TensorView

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  /// Data type of element stored within tensor
  typename Element_,
  /// Maps a Coord<Rank_> in the logical tensor index space to the internal n-D array
  typename Layout_
>
class TensorViewPlanarComplex : public TensorRefPlanarComplex<Element_, Layout_> {
 public:

  /// Base tensor reference
  using Base = cutlass::TensorRefPlanarComplex<Element_, Layout_>;

  /// Mapping function from logical coordinate to internal n-D array
  using Layout = Layout_;

  /// TensorRef pointing to constant memory
  using ConstTensorRef = typename Base::ConstTensorRef;

  /// Underlying TensorRef type
  using TensorRef = Base;

  /// Data type of individual access
  using Element = Element_;

  /// Reference type to an element
  using Reference = Element &;

  /// Logical rank of tensor index space
  static int const kRank = Layout::kRank;

  /// Index type
  using Index = typename Layout::Index;

  /// Long index used for pointer offsets
  using LongIndex = typename Layout::LongIndex;

  /// Coordinate in logical tensor space
  using TensorCoord = typename Layout::TensorCoord;

  /// Coordinate in storage n-D array
  using Stride = typename Layout::Stride;

  /// TensorView pointing to constant memory
  using ConstTensorView = TensorViewPlanarComplex<
    typename platform::remove_const<Element>::type const,
    Layout>;

  /// TensorView pointing to non-constant memory
  using NonConstTensorView = TensorViewPlanarComplex<
    typename platform::remove_const<Element>::type,
    Layout>;

  /// Require at least rank=1. Mathematically, a rank=0 tensor would be considered to be a
  /// scalar, but degenerate cases such as these are difficult to accommodate without
  /// extensive C++ metaprogramming or support for zero-length arrays.
  static_assert(kRank > 0, "Cannot define a zero-rank TensorRef");

 private:

  /// View extent
  TensorCoord extent_;

 public:

  //
  // Methods
  //

  /// Constructs a TensorView object
  CUTLASS_HOST_DEVICE
  TensorViewPlanarComplex(TensorCoord const &extent = TensorCoord()): extent_(extent) {

  }

  /// Constructs a TensorView object
  CUTLASS_HOST_DEVICE
  TensorViewPlanarComplex(
    Element *ptr,                         ///< pointer to start of tensor
    Layout const &layout,                 ///< layout object containing stride and mapping function
    LongIndex imaginary_stride,           ///< stride between real and imaginary part
    TensorCoord const &extent             ///< size of the view in logical coordinates
  ):
    Base(ptr, layout, imaginary_stride), extent_(extent) {
  
  }

  /// Constructs a TensorView object
  CUTLASS_HOST_DEVICE
  TensorViewPlanarComplex(
    TensorRef const &ref,                 ///< pointer and layout object referencing a tensor
    TensorCoord const &extent             ///< logical size of tensor
  ):
    Base(ref), extent_(extent) {
  
  }

  /// Converting constructor from TensorRef to non-constant data.
  CUTLASS_HOST_DEVICE
  TensorViewPlanarComplex(
    NonConstTensorView const &view        ///< TensorView to non-const data
  ):
    Base(view), extent_(view.extent_) { }

  /// Updates the pointer and layout object
  CUTLASS_HOST_DEVICE
  void reset(Element* ptr, Layout const &layout, LongIndex imaginary_stride, TensorCoord size) {
    Base::reset(ptr, layout, imaginary_stride);
    this->resize(extent_);
  }

  /// Changes the size of the view without affecting pointer or layout
  CUTLASS_HOST_DEVICE
  void resize(TensorCoord extent) {
    this->extent_ = extent;
  }

  /// Returns the extent of the view (the size along each logical dimension).
  CUTLASS_HOST_DEVICE
  TensorCoord const& extent() const { return extent_; }

  /// Returns the extent along a particular logical dimension.
  CUTLASS_HOST_DEVICE
  Index extent(int dim) const { return extent_.at(dim); }

  /// Determines whether a location is within a tensor
  CUTLASS_HOST_DEVICE
  bool contains(TensorCoord const& coord) const {
    CUTLASS_PRAGMA_UNROLL
    for (int dim = 0; dim < kRank; ++dim) {
      if (!(coord[dim] >= 0 && coord[dim] < extent(dim))) {
        return false;
      }
    }
    return true;
  }

  /// Returns a TensorRef pointing to the first element of the tensor.
  CUTLASS_HOST_DEVICE
  Base ref() const {
    return Base(this->data(), this->layout(), this->imaginary_stride());
  }

  /// Returns a TensorRef pointing to the first element of the tensor.
  CUTLASS_HOST_DEVICE
  ConstTensorRef const_ref() const {
    return ConstTensorRef(this->data(), this->layout());
  }

  /// Returns a TensorView to const data
  CUTLASS_HOST_DEVICE
  ConstTensorView const_view() const {
    return ConstTensorView(const_ref(), extent_);
  }

  /// Returns a Tensor_view given location and size quantities
  CUTLASS_HOST_DEVICE
  TensorViewPlanarComplex subview(
    TensorCoord extent,                               ///< extent of the resulting view
    TensorCoord const& location = TensorCoord()       ///< resulting view's origin within the old view
  ) const {

    TensorViewPlanarComplex result(this->ref(), extent.clamp(extent_ - location));
    result.add_coord_offset(location);
    return result; 
  }

  /// Returns the number of scalar elements needed to store tensor.
  CUTLASS_HOST_DEVICE
  size_t capacity() const {
    return Base::layout().capacity(extent_);
  }

  /// Returns a TensorView offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorViewPlanarComplex operator+(
    TensorCoord const& b            ///< offset in the logical coordinate space of the tensor
  ) const {

    TensorViewPlanarComplex result(*this);
    result.add_pointer_offset(this->offset(b));
    return result;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorViewPlanarComplex& operator+=(
    TensorCoord const& b            ///< offset in the logical coordinate space of the tensor
  ) {

    this->add_pointer_offset(this->offset(b));
    return *this;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorViewPlanarComplex operator-(
    TensorCoord const& b            ///< offset in the logical coordinate space of the tensor
  ) const {

    TensorRef result(*this);
    result.add_pointer_offset(-this->offset(b));
    return result;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorViewPlanarComplex& operator-=(
    TensorCoord const& b            ///< offset in the logical coordinate space of the tensor
  ) {

    this->add_pointer_offset(-this->offset(b));
    return *this;
  }

  /// TensorRef to real-valued tensor
  CUTLASS_HOST_DEVICE
  cutlass::TensorView<Element, Layout> view_real() const {
    return cutlass::TensorView<Element, Layout>(this->data(), this->layout(), extent_);
  }

  /// TensorRef to real-valued tensor
  CUTLASS_HOST_DEVICE
  cutlass::TensorView<Element, Layout> view_imag() const {
    return cutlass::TensorView<Element, Layout>(this->imaginary_data(), this->layout(), extent_);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Constructs a TensorRef, deducing types from arguments.
template <
  typename Element,
  typename Layout
>
CUTLASS_HOST_DEVICE TensorViewPlanarComplex<Element, Layout> make_TensorViewPlanarComplex(
  Element *ptr, 
  Layout const &layout,
  typename Layout::LongIndex imaginary_stride,
  typename Layout::TensorCoord const &extent) {

  return TensorViewPlanarComplex<Element, Layout>(ptr, layout, imaginary_stride, extent);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
