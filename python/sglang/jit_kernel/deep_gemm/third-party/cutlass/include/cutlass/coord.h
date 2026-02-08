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
    \brief A Coord is a coordinate of arbitrary rank into a tensor or matrix
*/

#pragma once
#include "cutlass/cutlass.h"
#if defined(__CUDACC_RTC__)
#include CUDA_STD_HEADER(cstdint)
#else
#include <cstdint>
#endif

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Statically-sized array specifying Coords within a tensor
template <
  int Rank_,                          ///< Logical rank of coordinate
  typename Index_ = int,              ///< Index type used for each dimension
  typename LongIndex_ = int64_t       ///< Long index type used for linear offsets
>
struct Coord {

public:

  //
  // Type and constant definitions
  //

  /// Number of elements in Coord
  static int const kRank = Rank_;

  /// Index type used to store elements
  using Index = Index_;

  /// Type used to represent linear offsets
  using LongIndex = LongIndex_;

private:

  //
  // Data members
  //

  /// Indices
  Index idx[kRank];

public:

  //
  // Methods
  //

  /// Default ctor initializes uniformly
  CUTLASS_HOST_DEVICE
  explicit Coord(Index value = Index(0)) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = value;
    }
  }

  /// Constructs from an array of integers
  CUTLASS_HOST_DEVICE
  Coord(Index const (&_idx)[kRank]) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = _idx[i];
    }
  }

  /// Constructs from some other Coord
  template <int R, typename I, typename L>
  CUTLASS_HOST_DEVICE
  Coord(Coord<R, I, L> other) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = other[i];
    }
  }

  /// Returns a slice of the Coord which may be larger or smaller in rank
  /// than this.
  template <int Slice>
  CUTLASS_HOST_DEVICE
  Coord<Slice, Index, LongIndex> slice(int start = 0, Index identity = 0) const {
    Coord<Slice, Index, LongIndex> result;
    for (int i = 0; i < Slice; ++i) {
      if (i + start < kRank) {
        result[i] = idx[i + start];
      }
      else {
        result[i] = identity;
      }
    }
    return result;
  }

  /// Returns the index of the dimension with least value
  CUTLASS_HOST_DEVICE
  int min_dim_index() const {
    int i = 0;
    for (int j = 1; j < kRank; ++j) {
      if (idx[j] < idx[i]) {
        i = j;
      }
    }
    return i;
  }

  /// Returns the index of the dimension with greatest value
  CUTLASS_HOST_DEVICE
  int max_dim_index() const {
    int i = 0;
    for (int j = 1; j < kRank; ++j) {
      if (idx[j] > idx[i]) {
        i = j;
      }
    }
    return i;
  }

  /// Returns true if Coord is non-zero.
  CUTLASS_HOST_DEVICE
  explicit operator bool() const {
    for (int i = 0; i < kRank; ++i) {
      if (idx[i]) {
        return true;
      }
    }
    return false;
  }

  /// Returns true if Coord is uniformly zero.
  CUTLASS_HOST_DEVICE
  bool operator!() const {
    for (int i = 0; i < kRank; ++i) {
      if (idx[i]) {
        return false;
      }
    }
    return true;
  }

  /// Element-wise addition
  CUTLASS_HOST_DEVICE
  Coord operator+(Coord const& b) const {
    Coord c;
    for (int i = 0; i < kRank; ++i) {
      c.idx[i] = idx[i] + b.idx[i];
    }
    return c;
  }

  /// Element-wise subtraction
  CUTLASS_HOST_DEVICE
  Coord operator-(Coord const& b) const {
    Coord c;
    for (int i = 0; i < kRank; ++i) {
      c.idx[i] = idx[i] - b.idx[i];
    }
    return c;
  }

  /// Element-wise multiplication
  CUTLASS_HOST_DEVICE
  Coord operator*(Coord const& b) const {
    Coord c;
    for (int i = 0; i < kRank; ++i) {
      c.idx[i] = idx[i] * b.idx[i];
    }
    return c;
  }

  /// Element-wise division
  CUTLASS_HOST_DEVICE
  Coord operator/(Coord const& b) const {
    Coord c;
    for (int i = 0; i < kRank; ++i) {
      c.idx[i] = idx[i] / b.idx[i];
    }
    return c;
  }

  /// In-place addition
  CUTLASS_HOST_DEVICE
  Coord& operator+=(Coord const& b) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] += b.idx[i];
    }
    return *this;
  }

  /// In-place subtraction
  CUTLASS_HOST_DEVICE
  Coord& operator-=(Coord const& b) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] -= b.idx[i];
    }
    return *this;
  }

  /// In-place multiplication
  CUTLASS_HOST_DEVICE
  Coord& operator*=(Coord const& b) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] *= b.idx[i];
    }
    return *this;
  }

  /// In-place division
  CUTLASS_HOST_DEVICE
  Coord& operator/=(Coord const& b) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] /= b.idx[i];
    }
    return *this;
  }

  /// Member access operator
  CUTLASS_HOST_DEVICE Index& operator[](int dim) { return idx[dim]; }

  /// Member access operator
  CUTLASS_HOST_DEVICE Index const& operator[](int dim) const { return idx[dim]; }

  /// Computes the dot product with anotherCoord object
  CUTLASS_HOST_DEVICE
  LongIndex dot(Coord const& b, LongIndex sum = LongIndex(0)) const {
    for (int i = 0; i < kRank; ++i) {
      sum += idx[i] * b.idx[i];
    }
    return sum;
  }

  /// Gets the index of a given Coord element
  template <int Dim>
  CUTLASS_HOST_DEVICE Index& at() {
    return idx[Dim];
  }

  /// Access via index; may limit unrolling potential
  CUTLASS_HOST_DEVICE
  Index& at(int dim) { return idx[dim]; }

  /// Gets the index of a given Coord element
  template <int Dim>
  CUTLASS_HOST_DEVICE Index const& at() const {
    return idx[Dim];
  }

  /// Access via index; may limit unrolling potential
  CUTLASS_HOST_DEVICE
  Index const& at(int dim) const { return idx[dim]; }

  /// Determines if two Coord<> objects are equal
  CUTLASS_HOST_DEVICE
  bool operator==(Coord const& b) const {
    bool equal = true;
    for (int i = 0; equal && i < kRank; ++i) {
      equal = (idx[i] == b.idx[i]);
    }
    return equal;
  }

  /// Not equal
  CUTLASS_HOST_DEVICE
  bool operator!=(Coord const& b) const { return !(*this == b); }

  /// Clamps a coordinate to a range specified by maximum and minimum values
  CUTLASS_HOST_DEVICE
  Coord& clamp(Coord const& max, Coord const& min = Coord()) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = __NV_STD_MAX(__NV_STD_MIN(idx[i], max.idx[i]), min.idx[i]);
    }
    return *this;
  }

  /// Returns the sum of all elements
  CUTLASS_HOST_DEVICE
  Index sum() const {
    Index sum_(idx[0]);
    for (int i = 1; i < kRank; ++i) {
      sum_ += idx[i];
    }
    return sum_;
  }

  /// Returns the product of all elements
  CUTLASS_HOST_DEVICE
  LongIndex product() const {
    LongIndex product_(idx[0]);
    for (int i = 1; i < kRank; ++i) {
      product_ *= idx[i];
    }
    return product_;
  }

  /// Less than operator
  CUTLASS_HOST_DEVICE
  bool operator<(Coord const &b) const {
    for (int i = 0; i < kRank; ++i) {
      if (!(idx[i] < b[i])) {
        return false;
      }
    }
    return true;
  }

  /// Less than or equals operator
  CUTLASS_HOST_DEVICE
  bool operator<=(Coord const &b) const {
    for (int i = 0; i < kRank; ++i) {
      if (!(idx[i] <= b[i])) {
        return false;
      }
    }
    return true;
  }

  /// Greater than operator
  CUTLASS_HOST_DEVICE
  bool operator>(Coord const &b) const {
    return !(*this <= b);
  }

  /// Greater than or equals operator
  CUTLASS_HOST_DEVICE
  bool operator>=(Coord const &b) const {
    return !(*this < b);
  }
};

} // namespace cutlass 

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {


/// Scalar multiplication
template <int Rank, typename Index>
CUTLASS_HOST_DEVICE
Coord<Rank, Index> operator*(Index s, Coord<Rank, Index> coord) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Rank; ++i) {
    coord[i] *= s;
  }
  return coord;
}

/// Scalar multiplication
template <int Rank, typename Index>
CUTLASS_HOST_DEVICE
Coord<Rank, Index> operator*(Coord<Rank, Index> coord, Index s) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Rank; ++i) {
    coord[i] *= s;
  }
  return coord;
}

/// Scalar division
template <int Rank, typename Index>
CUTLASS_HOST_DEVICE
Coord<Rank, Index> operator/(Index s, Coord<Rank, Index> coord) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Rank; ++i) {
    coord[i] = s / coord[i];
  }
  return coord;
}

/// Scalar division
template <int Rank, typename Index>
CUTLASS_HOST_DEVICE
Coord<Rank, Index> operator/(Coord<Rank, Index> coord, Index s) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < Rank; ++i) {
    coord[i] /= s;
  }
  return coord;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Integer-valued make_Coord
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to make a 1-element coordinate
template <typename T> 
CUTLASS_HOST_DEVICE
Coord<1, T> make_Coord(T _0) {
  T values[1] = {_0};
  return Coord<1, T>(values);
}

/// Helper to make a 2-element coordinate
template <typename T> 
CUTLASS_HOST_DEVICE
Coord<2, T> make_Coord(T _0, T _1) {
  T values[2] = {_0, _1};
  return Coord<2, T>(values);
}

/// Helper to make a 3-element coordinate
template <typename T> 
CUTLASS_HOST_DEVICE
Coord<3, T> make_Coord(T _0, T _1, T _2) {
  T values[3] = {_0, _1, _2};
  return Coord<3, T>(values);
}

/// Helper to make a 4-element coordinate
template <typename T> 
CUTLASS_HOST_DEVICE
Coord<4, T> make_Coord(T _0, T _1, T _2, T _3) {
  T values[4] = {_0, _1, _2, _3};
  return Coord<4, T>(values);
}

/// Helper to make a 5-element coordinate
template <typename T> 
CUTLASS_HOST_DEVICE
Coord<5, T> make_Coord(T _0, T _1, T _2, T _3, T _4) {
  T values[5] = {_0, _1, _2, _3, _4};
  return Coord<5, T>(values);
}

/// Helper to make a 1-element coordinate
template <int N, typename T> 
CUTLASS_HOST_DEVICE
Coord<N, T>make_Coord_with_padding(T _0) {
  Coord<N, T> coord;

  CUTLASS_PRAGMA_UNROLL
  for (int i = N - 1; i > 0; --i) {
    coord[i] = 0;
  }

  coord[0] = _0;

  return coord;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

