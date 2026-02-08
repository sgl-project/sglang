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

#include <cstdint>
#include "cutlass/cutlass.h"
#include "cutlass/complex.h"
#include "cutlass/tensor_ref.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element_>
struct PlanarComplexReference {

  //
  // Type definitions
  //

  using Element = Element_;
  using ComplexElement = complex<Element>;

  //
  // Data members
  //

  Element *real;
  Element *imag;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  PlanarComplexReference(
    Element *real_ = nullptr, 
    Element *imag_ = nullptr
  ):
    real(real_), imag(imag_) { }

  /// Loads the complex element
  CUTLASS_HOST_DEVICE
  operator complex<Element>() const {
    return complex<Element>{*real, *imag};
  }

  /// Stores a complex element to the location pointed to by the reference 
  CUTLASS_HOST_DEVICE
  PlanarComplexReference &operator=(complex<Element> const &rhs) {
    *real = rhs.real();
    *imag = rhs.imag();
    return *this;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/* \brief TensorRef is a template for objects pointing to the start of tensors of arbitrary rank
          and layout within memory. A TensorRef combines a pointer and a Layout concept

*/
template <
  /// Data type of element stored within tensor (concept: NumericType)
  typename Element_,
  /// Defines a mapping from logical coordinate to linear memory (concept: Layout)
  typename Layout_
>
class TensorRefPlanarComplex {
 public:
  /// Data type of individual access
  using Element = Element_;

  /// Complex element type
  using ComplexElement = complex<Element>;

  /// Mapping function from logical coordinate to linear memory
  using Layout = Layout_;

  static_assert(sizeof_bits<Element>::value >= 8,
    "Planar complex not suitable for subbyte elements at this time");

  /// Reference type to an element
  using Reference = PlanarComplexReference<Element>;

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
  using ConstTensorRef = TensorRefPlanarComplex<
    typename platform::remove_const<Element>::type const,
    Layout>;

  /// TensorRef to non-constant data
  using NonConstTensorRef = TensorRefPlanarComplex<
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

  /// Offset to imaginary part
  LongIndex imaginary_stride_;

 public:

  //
  // Methods
  //

  /// Constructs a TensorRef with a pointer and layout object.
  CUTLASS_HOST_DEVICE
  TensorRefPlanarComplex(
    Element *ptr = nullptr,                   ///< pointer to start of tensor
    Layout const &layout = Layout(),          ///< layout object containing stride and mapping function
    LongIndex imaginary_stride = 0
  ):
    ptr_(ptr), layout_(layout), imaginary_stride_(imaginary_stride) {
  
  }

  /// Converting constructor from TensorRef to non-constant data.
  CUTLASS_HOST_DEVICE
  TensorRefPlanarComplex(
    NonConstTensorRef const &ref              ///< TensorRef to non-const data
  ):
    ptr_(ref.data()), layout_(ref.layout()), imaginary_stride_(ref.imaginary_stride_) { }

  /// Returns a reference to constant-valued tensor.
  CUTLASS_HOST_DEVICE
  ConstTensorRef const_ref() const {
    return ConstTensorRef(ptr_, layout_, imaginary_stride_);
  }

  CUTLASS_HOST_DEVICE
  NonConstTensorRef non_const_ref() const {
    return NonConstTensorRef(
      const_cast<typename platform::remove_const<Element>::type *>(ptr_), 
      layout_, 
      imaginary_stride_);
  }

  /// Updates only the pointer
  CUTLASS_HOST_DEVICE
  void reset(Element* ptr = nullptr, LongIndex imaginary_stride = 0) {
    ptr_ = ptr;
    imaginary_stride_ = imaginary_stride;
  }

  /// Updates the pointer and layout object
  CUTLASS_HOST_DEVICE
  void reset(Element* ptr, Layout const &layout, LongIndex imaginary_stride) {
    ptr_ = ptr;
    layout_ = layout;
    imaginary_stride_ = imaginary_stride;
  }

  /// Returns true if the TensorRef is non-null
  CUTLASS_HOST_DEVICE
  bool good() const {
    return ptr_ != nullptr;
  }

  /// Returns the pointer to referenced data
  CUTLASS_HOST_DEVICE
  Element * data() const { return ptr_; }

  /// Returns the pointer to referenced data
  CUTLASS_HOST_DEVICE
  Element * imaginary_data() const { return ptr_ + imaginary_stride_; }

  /// Returns a reference to the element at a given linear index
  CUTLASS_HOST_DEVICE
  Reference data(LongIndex idx) const {
    return Reference(ptr_ + idx, ptr_ + idx + imaginary_stride_);
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

  /// Gets the stride to an imaginary element
  LongIndex imaginary_stride() const {
    return imaginary_stride_;
  }

  /// Gets the stride to an imaginary element
  LongIndex &imaginary_stride() {
    return imaginary_stride_;
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
  Index stride(int dim) const {
    return layout_.stride().at(dim);
  }

  /// Returns the layout object's stride in a given physical dimension
  CUTLASS_HOST_DEVICE
  Index & stride(int dim) {
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
  TensorRefPlanarComplex & add_pointer_offset(LongIndex offset_) {
    ptr_ += offset_;
    return *this;
  }

  /// Adds an offset to each pointer
  CUTLASS_HOST_DEVICE
  TensorRefPlanarComplex & add_coord_offset(TensorCoord const &coord) {
    add_pointer_offset(offset(coord));
    return *this;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorRefPlanarComplex operator+(TensorCoord const& b) const {
    TensorRefPlanarComplex result(*this);
    result.add_coord_offset(b);
    return result;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorRefPlanarComplex & operator+=(TensorCoord const& b) {
    add_coord_offset(b);
    return *this;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorRefPlanarComplex operator-(TensorCoord const& b) const {
    TensorRefPlanarComplex result(*this);
    result.add_pointer_offset(-offset(b));
    return result;
  }

  /// Returns a TensorRef offset by a given amount
  CUTLASS_HOST_DEVICE
  TensorRefPlanarComplex & operator-=(TensorCoord const& b) {
    add_pointer_offset(-offset(b));
    return *this;
  }

  /// TensorRef to real-valued tensor
  CUTLASS_HOST_DEVICE
  cutlass::TensorRef<Element, Layout> ref_real() const {
    return cutlass::TensorRef<Element, Layout>(data(), layout());
  }

  /// TensorRef to real-valued tensor
  CUTLASS_HOST_DEVICE
  cutlass::TensorRef<Element, Layout> ref_imag() const {
    return cutlass::TensorRef<Element, Layout>(imaginary_data(), layout());
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Constructs a TensorRef, deducing types from arguments.
template <
  typename Element,
  typename Layout
>
CUTLASS_HOST_DEVICE
TensorRefPlanarComplex<Element, Layout> make_TensorRefPlanarComplex(
  Element *ptr, 
  Layout const &layout, 
  int64_t imaginary_stride) {

  return TensorRefPlanarComplex<Element, Layout>(ptr, layout, imaginary_stride);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
