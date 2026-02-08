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
    \brief Defines a matrix object intended for storing data in registers and operations within
      a CUDA thread.
*/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/matrix_coord.h"

namespace cutlass {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Per-thread matrix object storing a packed matrix
template <
  typename Element,
  int Rows,
  int Columns,
  typename Layout = layout::RowMajor
>
class Matrix : public Array<Element, Rows * Columns> {
public:
  
  // Verify layout refers to a rank=2 matrix.
  static_assert(
    Layout::kRank == 2,
    "Layout type must refer to a rank=2 matrix");

  /// Base type
  using Base = Array<Element, Rows * Columns>;

  /// Element type
  using Element = Element_;

  /// Number of rows
  static int const kRows = Rows;

  /// Number of columns
  static int const kColumns = Columns;

  /// Layout within the array
  using Layout = Layout_;

  /// Reference type to an element
  using Reference = Element &;

  /// Logical rank of tensor index space
  static int const kRank = 2;

  /// Index type
  using Index = typename Layout::Index;

  /// Long index used for pointer offsets
  using LongIndex = typename Layout::LongIndex;

  /// Coordinate in logical tensor space
  using TensorCoord = typename Layout::TensorCoord;

  /// Stride type
  using Stride = typename Layout::Stride;

  /// TensorRef to matrix object
  using TensorRef = TensorRef<Element, kRank, Layout>;

  /// TensorRef to constant matrix object
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  /// TensorRef to matrix object
  using TensorView = TensorView<Element, kRank, Layout>;

  /// TensorRef to constant matrix object
  using ConstTensorView = typename TensorView::ConstTensorView;

  /// Diagonal vector
  using Diagonal = Vector<Element, __NV_STD_MIN(kRows, kColumns)>;

private:


public:

  //
  // Methods
  //

  /// Returns the size of the object
  CUTLASS_HOST_DEVICE
  static MatrixCoord extent() {
    return make_Coord(kRows, kColumns);
  }

  /// Returns the layout object
  CUTLASS_HOST_DEVICE
  static Layout layout() {
    return Layout::packed(extent());
  }

  /// Ctor
  CUTLASS_HOST_DEVICE
  Matrix() { }

  /// Ctor
  CUTLASS_HOST_DEVICE
  Matrix(Diagonal const &diag) {
  }

  /// Returns a TensorRef pointing to the first element of the tensor.
  CUTLASS_HOST_DEVICE
  TensorRef ref() {
    return TensorRef(this->data(), layout());
  }

  /// Returns a TensorRef pointing to the first element of the tensor.
  CUTLASS_HOST_DEVICE
  ConstTensorRef const_ref() const {
    return ConstTensorRef(this->data(), layout());
  }

  /// Returns a TensorRef pointing to the first element of the tensor.
  CUTLASS_HOST_DEVICE
  TensorView view() {
    return TensorView(ref(), extent());
  }

  /// Returns a TensorView to const data
  CUTLASS_HOST_DEVICE
  ConstTensorView const_view() const {
    return ConstTensorView(const_ref(), extent());
  }

  /// Returns a reference to the element at a given Coord
  CUTLASS_HOST_DEVICE
  Reference at(MatrixCoord const& coord) const {
    typename Base::size_type offset_(layout().offset(coord));
    return Base::at(offset_);
  }

  /// Returns the number of scalar elements needed to store tensor.
  CUTLASS_HOST_DEVICE
  LongIndex capacity() const {
    return LongIndex(Base::size());
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Column vector defined as a matrix with exactly one column
template <
  typename Element,
  int Rows,
  typename Layout = layout::ColumnMajor
>
using ColumnVector = Matrix<Element, Rows, 1, Layout>;

/// Row vector defined as a matrix with exactly one row
template <
  typename Element,
  int Columns,
  typename Layout = layout::RowMajor
>
using RowVector = Matrix<Element, 1, Columns, Layout>;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace cutlass
