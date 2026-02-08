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
    \brief Defines a canonical coordinate for rank=2 matrices offering named indices.
*/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/coord.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// MatrixCoord wraps Coord<2, int> to provide a helper for accessing named dimensions. Classes
/// expecting a coordinate in the rank=2 index space of a matrix should use MatrixCoord.
struct MatrixCoord : public Coord<2, int> {

public:

  /// Integer-valued index
  using Index = int;

  /// Base type is a Coord of rank=2
  using Base = Coord<2, Index>;

  /// LongIndex type
  using LongIndex = typename Base::LongIndex;

private:

  /// Rows dimension
  static int const kRow = 0;

  /// Columns dimension
  static int const kColumn = 1;

public:

  //
  // Methods
  //

  /// Default ctor
  CUTLASS_HOST_DEVICE
  MatrixCoord() { }

  /// Constructs from Coord<2>
  CUTLASS_HOST_DEVICE
  MatrixCoord(Coord<2, Index> const &coord): Base(coord) { }

  /// Helper to construct from a row and column
  CUTLASS_HOST_DEVICE
  MatrixCoord(Index row, Index column): Base(make_Coord(row, column)) { }

  /// Helper to construct from a row and column, which are LongIndex based
  CUTLASS_HOST_DEVICE
  MatrixCoord(LongIndex row, LongIndex column): Base(make_Coord(Index(row), Index(column))) { }

  /// Returns the row of the coordinate
  CUTLASS_HOST_DEVICE
  Index const & row() const { return this->at(kRow); }

  /// Returns the row of the coordinate
  CUTLASS_HOST_DEVICE
  Index & row() { return this->at(kRow); }

  /// Returns the column of the coordinate
  CUTLASS_HOST_DEVICE
  Index const & column() const { return this->at(kColumn); }

  /// Returns the column of the coordinate
  CUTLASS_HOST_DEVICE
  Index & column() { return this->at(kColumn); }

  //
  // Coord operators
  //

  /// Element-wise addition
  CUTLASS_HOST_DEVICE
  MatrixCoord operator+(Base const& b) const {
    return MatrixCoord(Base::operator+(b));
  }

  /// Element-wise subtraction
  CUTLASS_HOST_DEVICE
  MatrixCoord operator-(Base const& b) const {
    return MatrixCoord(Base::operator-(b));
  }

  /// Element-wise multiplication
  CUTLASS_HOST_DEVICE
  MatrixCoord operator*(Base const& b) const {
    return MatrixCoord(Base::operator*(b));
  }

  /// Element-wise division
  CUTLASS_HOST_DEVICE
  MatrixCoord operator/(Base const& b) const {
    return MatrixCoord(Base::operator/(b));
  }

  /// In-place addition
  CUTLASS_HOST_DEVICE
  MatrixCoord& operator+=(Base const& b) {
    Base::operator+=(b);
    return *this;
  }

  /// In-place subtraction
  CUTLASS_HOST_DEVICE
  MatrixCoord& operator-=(Base const& b) {
    Base::operator-=(b);
    return *this;
  }

  /// In-place multiplication
  CUTLASS_HOST_DEVICE
  MatrixCoord& operator*=(Base const& b) {
    Base::operator*=(b);
    return *this;
  }

  /// In-place division
  CUTLASS_HOST_DEVICE
  MatrixCoord& operator/=(Base const& b) {
    Base::operator/=(b);
    return *this;
  }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
