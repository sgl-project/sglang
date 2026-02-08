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
    \brief Defines a canonical coordinate for rank=4 tensors offering named indices.
*/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/coord.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a canonical 4D coordinate used by tensor operations.
struct Tensor4DCoord : public Coord<4> {

  /// Base class
  using Base = Coord<4>;

  /// Index type
  using Index = typename Base::Index;

  /// LongIndex type
  using LongIndex = typename Base::LongIndex;

  /// Batch dimension
  static int const kN = 0;

  /// Height dimension
  static int const kH = 1;

  /// Width dimension
  static int const kW = 2;

  /// Channels dimension
  static int const kC = 3;

  //
  // Methods
  //

  /// Default ctor
  CUTLASS_HOST_DEVICE
  Tensor4DCoord() { }

  /// Constructs from Coord<4>
  CUTLASS_HOST_DEVICE
  Tensor4DCoord(Coord<4> const &coord): Base(coord) { }

  /// Helper to construct from N, H, W, and C.
  CUTLASS_HOST_DEVICE
  Tensor4DCoord(Index n, Index h, Index w, Index c): Base(make_Coord(n, h, w, c)) { }

  /// Helper to construct from N, H, W, and C, which are LongIndex type
  CUTLASS_HOST_DEVICE
  Tensor4DCoord(LongIndex n, LongIndex h, LongIndex w, LongIndex c)
    : Base(make_Coord(Index(n), Index(h), Index(w), Index(c))) { }

  /// Returns the batch of the coordinate
  CUTLASS_HOST_DEVICE
  Index const & n() const { return this->at(kN); }

  /// Returns the batch of the coordinate
  CUTLASS_HOST_DEVICE
  Index & n() { return this->at(kN); }

  /// Returns the row of the coordinate
  CUTLASS_HOST_DEVICE
  Index const & h() const { return this->at(kH); }

  /// Returns the row of the coordinate
  CUTLASS_HOST_DEVICE
  Index & h() { return this->at(kH); }

  /// Returns the column of the coordinate
  CUTLASS_HOST_DEVICE
  Index const & w() const { return this->at(kW); }

  /// Returns the column of the coordinate
  CUTLASS_HOST_DEVICE
  Index & w() { return this->at(kW); }

  /// Returns the channel of the coordinate
  CUTLASS_HOST_DEVICE
  Index const & c() const { return this->at(kC); }

  /// Returns the channel of the coordinate
  CUTLASS_HOST_DEVICE
  Index & c() { return this->at(kC); }

  //
  // Coord operators
  //

  /// Element-wise addition
  CUTLASS_HOST_DEVICE
  Tensor4DCoord operator+(Base const& b) const {
    return Tensor4DCoord(Base::operator+(b));
  }

  /// Element-wise subtraction
  CUTLASS_HOST_DEVICE
  Tensor4DCoord operator-(Base const& b) const {
    return Tensor4DCoord(Base::operator-(b));
  }

  /// Element-wise multiplication
  CUTLASS_HOST_DEVICE
  Tensor4DCoord operator*(Base const& b) const {
    return Tensor4DCoord(Base::operator*(b));
  }

  /// Element-wise division
  CUTLASS_HOST_DEVICE
  Tensor4DCoord operator/(Base const& b) const {
    return Tensor4DCoord(Base::operator/(b));
  }

  /// In-place addition
  CUTLASS_HOST_DEVICE
  Tensor4DCoord& operator+=(Base const& b) {
    Base::operator+=(b);
    return *this;
  }

  /// In-place subtraction
  CUTLASS_HOST_DEVICE
  Tensor4DCoord& operator-=(Base const& b) {
    Base::operator-=(b);
    return *this;
  }

  /// In-place multiplication
  CUTLASS_HOST_DEVICE
  Tensor4DCoord& operator*=(Base const& b) {
    Base::operator*=(b);
    return *this;
  }

  /// In-place division
  CUTLASS_HOST_DEVICE
  Tensor4DCoord& operator/=(Base const& b) {
    Base::operator/=(b);
    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a canonical 5D coordinate used by tensor operations.
struct Tensor5DCoord : public Coord<5> {

  /// Base class
  using Base = Coord<5>;

  /// Index type
  using Index = typename Base::Index;

  /// LongIndex type
  using LongIndex = typename Base::LongIndex;

  /// Batch dimension
  static int const kN = 0;

  /// Depth dimension
  static int const kD = 1;

  /// Height dimension
  static int const kH = 2;

  /// Width dimension
  static int const kW = 3;

  /// Channels dimension
  static int const kC = 4;

  //
  // Methods
  //

  /// Default ctor
  CUTLASS_HOST_DEVICE
  Tensor5DCoord() { }

  /// Constructs from Coord<5>
  CUTLASS_HOST_DEVICE
  Tensor5DCoord(Coord<5> const &coord): Base(coord) { }

  /// Helper to construct from N, D, H, W, and C.
  CUTLASS_HOST_DEVICE
  Tensor5DCoord(Index n, Index d, Index h, Index w, Index c): Base(make_Coord(n, d, h, w, c)) { }

  /// Helper to construct from N, D, H, W, and C, which are LongIndex type
  CUTLASS_HOST_DEVICE
  Tensor5DCoord(LongIndex n, LongIndex d, LongIndex h, LongIndex w, LongIndex c)
    : Base(make_Coord(Index(n), Index(d), Index(h), Index(w), Index(c))) { }

  /// Returns the batch of the coordinate
  CUTLASS_HOST_DEVICE
  Index const & n() const { return this->at(kN); }

  /// Returns the batch of the coordinate
  CUTLASS_HOST_DEVICE
  Index & n() { return this->at(kN); }

  /// Returns the batch of the coordinate
  CUTLASS_HOST_DEVICE
  Index const & d() const { return this->at(kD); }

  /// Returns the batch of the coordinate
  CUTLASS_HOST_DEVICE
  Index & d() { return this->at(kD); }

  /// Returns the row of the coordinate
  CUTLASS_HOST_DEVICE
  Index const & h() const { return this->at(kH); }

  /// Returns the row of the coordinate
  CUTLASS_HOST_DEVICE
  Index & h() { return this->at(kH); }

  /// Returns the column of the coordinate
  CUTLASS_HOST_DEVICE
  Index const & w() const { return this->at(kW); }

  /// Returns the column of the coordinate
  CUTLASS_HOST_DEVICE
  Index & w() { return this->at(kW); }

  /// Returns the channel of the coordinate
  CUTLASS_HOST_DEVICE
  Index const & c() const { return this->at(kC); }

  /// Returns the channel of the coordinate
  CUTLASS_HOST_DEVICE
  Index & c() { return this->at(kC); }

  //
  // Coord operators
  //

  /// Element-wise addition
  CUTLASS_HOST_DEVICE
  Tensor5DCoord operator+(Base const& b) const {
    return Tensor5DCoord(Base::operator+(b));
  }

  /// Element-wise subtraction
  CUTLASS_HOST_DEVICE
  Tensor5DCoord operator-(Base const& b) const {
    return Tensor5DCoord(Base::operator-(b));
  }

  /// Element-wise multiplication
  CUTLASS_HOST_DEVICE
  Tensor5DCoord operator*(Base const& b) const {
    return Tensor5DCoord(Base::operator*(b));
  }

  /// Element-wise division
  CUTLASS_HOST_DEVICE
  Tensor5DCoord operator/(Base const& b) const {
    return Tensor5DCoord(Base::operator/(b));
  }

  /// In-place addition
  CUTLASS_HOST_DEVICE
  Tensor5DCoord& operator+=(Base const& b) {
    Base::operator+=(b);
    return *this;
  }

  /// In-place subtraction
  CUTLASS_HOST_DEVICE
  Tensor5DCoord& operator-=(Base const& b) {
    Base::operator-=(b);
    return *this;
  }

  /// In-place multiplication
  CUTLASS_HOST_DEVICE
  Tensor5DCoord& operator*=(Base const& b) {
    Base::operator*=(b);
    return *this;
  }

  /// In-place division
  CUTLASS_HOST_DEVICE
  Tensor5DCoord& operator/=(Base const& b) {
    Base::operator/=(b);
    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
