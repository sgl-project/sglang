/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "cutlass/coord.h"

namespace cutlass {
namespace gemm {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Shape of a matrix multiply-add operation
template <
  /// Rows of matrix product
  int M = 1,
  /// Columns of matrix product
  int N = 1,
  /// Inner dimension of matrix product
  int K = 1
>
struct GemmShape {
  static int const kM = M;
  static int const kN = N;
  static int const kK = K;

  static int const kMN = M * N;
  static int const kMK = M * K;
  static int const kKN = N * K;
  static int const kMNK = M * N * K;

  static int const kCount = kMNK;

  //
  // Static member functions
  //

  /// Returns a Coord object
  CUTLASS_HOST_DEVICE
  static Coord<3> toCoord() {
    return make_Coord(kM, kN, kK);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Type alias of the transpose of a GemmShape
template <
  /// concept: GemmShape
  typename Shape
>
using GemmShapeTranspose = GemmShape<Shape::kN, Shape::kM, Shape::kK>;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// GemmCoord is a structure derived from Coord<3> that specifies a location within the
/// coordinate space of a GEMM problem.
struct GemmCoord : public Coord<3, int> {

  /// Integer-valued index
  typedef int Index;

  /// Base type is a Coord of rank=3
  typedef Coord<3, Index> Base;

  /// GEMM M dimension - rows of the output C matrix
  static int const kM = 0;

  /// GEMM N dimension - columns of the output C matrix
  static int const kN = 1;

  /// GEMM K dimension - inner dimension of the GEMM problem
  static int const kK = 2;

  //
  // Methods
  //

  /// Default ctor
  CUTLASS_HOST_DEVICE
  GemmCoord() { }

  /// Constructs from Coord<3> and a batch
  CUTLASS_HOST_DEVICE
  GemmCoord(Coord<3, Index> const& coord): Base(make_Coord(coord[0], coord[1], coord[2])) { }

  /// Helper to construct from a K, N, M, batch variables
  CUTLASS_HOST_DEVICE
  GemmCoord(Index m, Index n, Index k): Base(make_Coord(m, n, k)) { }

  /// Returns the GEMM M coordinate
  CUTLASS_HOST_DEVICE
  Index const&  m() const { return this->at(kM); }

  /// Returns reference to the GEMM M coordinate
  CUTLASS_HOST_DEVICE
  Index & m() { return this->at(kM); }

  /// Returns the GEMM N coordinate
  CUTLASS_HOST_DEVICE
  Index const&  n() const { return this->at(kN); }

  /// Returns reference to the GEMM N coordinate
  CUTLASS_HOST_DEVICE
  Index & n() { return this->at(kN); }

  /// Returns the GEMM K coordinate
  CUTLASS_HOST_DEVICE
  Index const&  k() const { return this->at(kK); }

  /// Returns reference to the GEMM K coordinate
  CUTLASS_HOST_DEVICE
  Index & k() { return this->at(kK); }

  /// Obtains a Coord<3> from GemmCoord
  CUTLASS_HOST_DEVICE
  Coord<3> mnk() const {
    return make_Coord(m(), n(), k());
  }

  /// Obtains a Coord<3> from GemmCoord
  CUTLASS_HOST_DEVICE
  Coord<3> knm() const {
    return make_Coord(k(), n(), m());
  }

  /// Obtains a Coord<2> from GemmCoord
  CUTLASS_HOST_DEVICE
  Coord<2> nm() const {
    return make_Coord(n(), m());
  }

  /// Obtains a Coord<2> from GemmCoord
  CUTLASS_HOST_DEVICE
  Coord<2> mn() const {
    return make_Coord(m(), n());
  }

  /// Obtains a Coord<2> from GemmCoord
  CUTLASS_HOST_DEVICE
  Coord<2> mk() const {
    return make_Coord(m(), k());
  }

  /// Obtains a Coord<2> from GemmCoord
  CUTLASS_HOST_DEVICE
  Coord<2> km() const {
    return make_Coord(k(), m());
  }

  /// Obtains a Coord<2> from GemmCoord
  CUTLASS_HOST_DEVICE
  Coord<2> nk() const {
    return make_Coord(n(), k());
  }

  /// Obtains a Coord<2> from GemmCoord
  CUTLASS_HOST_DEVICE
  Coord<2> kn() const {
    return make_Coord(k(), n());
  }

  //
  // Coord operators
  //

  /// Element-wise addition
  CUTLASS_HOST_DEVICE
  GemmCoord operator+(Base const& b) const {
    return GemmCoord(Base::operator+(b));
  }

  /// Element-wise subtraction
  CUTLASS_HOST_DEVICE
  GemmCoord operator-(Base const& b) const {
    return GemmCoord(Base::operator-(b));
  }

  /// Element-wise multiplication
  CUTLASS_HOST_DEVICE
  GemmCoord operator*(Base const& b) const {
    return GemmCoord(Base::operator*(b));
  }

  /// Element-wise division
  CUTLASS_HOST_DEVICE
  GemmCoord operator/(Base const& b) const {
    return GemmCoord(Base::operator/(b));
  }

  /// In-place addition
  CUTLASS_HOST_DEVICE
  GemmCoord& operator+=(Base const& b) {
    Base::operator+=(b);
    return *this;
  }

  /// In-place subtraction
  CUTLASS_HOST_DEVICE
  GemmCoord& operator-=(Base const& b) {
    Base::operator-=(b);
    return *this;
  }

  /// In-place multiplication
  CUTLASS_HOST_DEVICE
  GemmCoord& operator*=(Base const& b) {
    Base::operator*=(b);
    return *this;
  }

  /// In-place division
  CUTLASS_HOST_DEVICE
  GemmCoord& operator/=(Base const& b) {
    Base::operator/=(b);
    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// BatchedGemmCoord is a structure derived from Coord<4> that specifies a location within the
/// coordinate space of a batched GEMM problem.
struct BatchedGemmCoord : public Coord<4, int> {

  /// Integer-valued index
  typedef int Index;

  /// Base type is a Coord of rank=4
  typedef Coord<4, Index> Base;

  /// GEMM M dimension - rows of the output C matrix
  static int const kM = 0;

  /// GEMM N dimension - columns of the output C matrix
  static int const kN = 1;

  /// GEMM K dimension - inner dimension of the GEMM problem
  static int const kK = 2;

  /// GEMM Batch dimension - inner dimension of the GEMM problem
  static int const kBatch = 3;

  //
  // Methods
  //

  /// Default ctor
  CUTLASS_HOST_DEVICE
  BatchedGemmCoord() { }

  /// Constructs from Coord<4>
  CUTLASS_HOST_DEVICE
  BatchedGemmCoord(Base const& coord): Base(coord) { }

  /// Helper to construct from a K, N, M, and batch variables
  CUTLASS_HOST_DEVICE
  BatchedGemmCoord(Index m, Index n, Index k, Index b): Base(make_Coord(m, n, k, b)) { }

  /// Returns the GEMM M coordinate
  CUTLASS_HOST_DEVICE
  Index const&  m() const { return this->at(kM); }

  /// Returns reference to the GEMM M coordinate
  CUTLASS_HOST_DEVICE
  Index & m() { return this->at(kM); }

  /// Returns the GEMM N coordinate
  CUTLASS_HOST_DEVICE
  Index const&  n() const { return this->at(kN); }

  /// Returns reference to the GEMM N coordinate
  CUTLASS_HOST_DEVICE
  Index & n() { return this->at(kN); }

  /// Returns the GEMM K coordinate
  CUTLASS_HOST_DEVICE
  Index const&  k() const { return this->at(kK); }

  /// Returns reference to the GEMM K coordinate
  CUTLASS_HOST_DEVICE
  Index & k() { return this->at(kK); }

  /// Returns the GEMM batch coordinate
  CUTLASS_HOST_DEVICE
  Index const&  batch() const { return this->at(kBatch); }

  /// Returns reference to the GEMM batch coordinate
  CUTLASS_HOST_DEVICE
  Index & batch() { return this->at(kBatch); }

  /// Obtains a GemmCoord from BatchedGemmCoord
  CUTLASS_HOST_DEVICE
  GemmCoord mnk() const {
    return GemmCoord(m(), n(), k());
  }

  /// Obtains a Coord<4> from BatchedGemmCoord
  CUTLASS_HOST_DEVICE
  Coord<4> mnkb() const {
    return make_Coord(m(), n(), k(), batch());
  }

  //
  // Coord operators
  //

  /// Element-wise addition
  CUTLASS_HOST_DEVICE
  BatchedGemmCoord operator+(Base const& b) const {
    return BatchedGemmCoord(Base::operator+(b));
  }

  /// Element-wise subtraction
  CUTLASS_HOST_DEVICE
  BatchedGemmCoord operator-(Base const& b) const {
    return BatchedGemmCoord(Base::operator-(b));
  }

  /// Element-wise multiplication
  CUTLASS_HOST_DEVICE
  BatchedGemmCoord operator*(Base const& b) const {
    return BatchedGemmCoord(Base::operator*(b));
  }

  /// Element-wise division
  CUTLASS_HOST_DEVICE
  BatchedGemmCoord operator/(Base const& b) const {
    return BatchedGemmCoord(Base::operator/(b));
  }

  /// In-place addition
  CUTLASS_HOST_DEVICE
  BatchedGemmCoord& operator+=(Base const& b) {
    Base::operator+=(b);
    return *this;
  }

  /// In-place subtraction
  CUTLASS_HOST_DEVICE
  BatchedGemmCoord& operator-=(Base const& b) {
    Base::operator-=(b);
    return *this;
  }

  /// In-place multiplication
  CUTLASS_HOST_DEVICE
  BatchedGemmCoord& operator*=(Base const& b) {
    Base::operator*=(b);
    return *this;
  }

  /// In-place division
  CUTLASS_HOST_DEVICE
  BatchedGemmCoord& operator/=(Base const& b) {
    Base::operator/=(b);
    return *this;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
