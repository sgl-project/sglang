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
    \brief This defines a "fragment" iterator for visiting the fragments of a warp vector
      that participate in one warp-level mma operation.

      Typically, this is used to access the scale/bias fragment of a warp-level mma operation.
      The scale/bias vector is then partitioned into smaller fragments that can be fed into 
      next warp-level mma operation. 

      This iterator is necessary to accomplish warp-level mma fusion where the scale/bias vector is 
      applied to the multiplicand for the next mma.

*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_conversion.h"

namespace cutlass {
namespace transform {
namespace warp {


////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the input fragment tile shape (concept: MatrixShape)
    typename Shape_,
    /// Element type
    typename Element_,
    /// Layout of operand in memory
    typename Layout_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    //// Number of elements per access when loading fragment
    int ElementsPerAccess>
class VectorFragmentIterator;


// Partial specialization for PitchLinear layout tile

template <
    /// Size of the input fragment vector shape (concept: MatrixShape)
    typename Shape_,
    /// Element type
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    //// Number of elements per access when loading fragment
    int ElementsPerAccess>
class VectorFragmentIterator<Shape_, Element_,
                                         cutlass::layout::PitchLinear,
                                         InstructionShape_, ElementsPerAccess> {
 public:
    
  /// Size of the input threadblock tile shape (concept: MatrixShape)
  using Shape = Shape_;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = cutlass::layout::PitchLinear;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = 32;

  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kRowsPerIteration = 8;
  static int const kColumnsPerAccess = 8;
  static int const kElementsPerIteration = kRowsPerIteration * InstructionShape::kK / kThreads;
  static int const kAccessPerIteration = kElementsPerIteration / kElementsPerAccess;
  
  /// Number of iterations
  using Iterations = MatrixShape<InstructionShape::kM / kRowsPerIteration, Shape::kContiguous / kElementsPerIteration>;

public:

  //
  // Derived quantities
  //
  // All fragments have kElementsPerAccess scale followed by bias

  /// Fragment object holding a thread's part of a tile
  /// This is the fragment size produced by one iteration of the iterator.
  using Fragment = Array<Element, kElementsPerIteration * Iterations::kRow>;

  /// Input threadblock fragment tile
  using ThreadblockFragment = Array<Element, Shape::kContiguous >;

private:

  /// Internal access type
  using AccessType = Array<Element, kElementsPerAccess>;

private:
  //
  // Data members
  //

  /// Input threadblock fragment tile
  AccessType const *iterator_;

  /// Internal index
  int index_;

public:
  /// Constructs an iterator
  CUTLASS_HOST_DEVICE
  VectorFragmentIterator(ThreadblockFragment const &threadblock_frag)
      : iterator_(reinterpret_cast<AccessType const *>(&threadblock_frag)),
        index_(0) {}

  /// Add offset
  CUTLASS_HOST_DEVICE
  void add_offset(int index_offset) {
    index_ += index_offset; 

    if(index_ >= Iterations::kColumn)
        index_ = 0;
  }

  /// Increments
  CUTLASS_HOST_DEVICE
  VectorFragmentIterator &operator++() {
    add_offset(1);
    return *this;
  }

  CUTLASS_HOST_DEVICE
  void set_index(int idx) {
    index_ = idx;
  }

  /// Loads a fragment from the referenced part of the accumulator tile
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < Iterations::kRow; r++) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kAccessPerIteration; i++) {
    
          frag_ptr[i * Iterations::kRow + r].clear();
          frag_ptr[i * Iterations::kRow + r] = iterator_[index_ * kAccessPerIteration + i];
        }
    }
  }

};

// Partial specialization for Row-Major layout tile

template <
    /// Size of the input fragment tile shape (concept: MatrixShape)
    typename Shape_,
    /// Element type
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    //// Number of elements per access when loading fragment
    int ElementsPerAccess>
class VectorFragmentIterator<Shape_, Element_,
                                         cutlass::layout::RowMajor,
                                         InstructionShape_, ElementsPerAccess> {
 public:
    
  /// Size of the input threadblock tile shape (concept: MatrixShape)
  using Shape = Shape_;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = cutlass::layout::RowMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Underlying iterator
  using Base = VectorFragmentIterator<
    layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
    layout::PitchLinear, InstructionShape, ElementsPerAccess>;


 public:

  //
  // Derived quantities
  //
  /// Fragment object holding a thread's part of a tile
  /// This is the fragment size produced by one iteration of the iterator.
  using Fragment = typename Base::Fragment;

  /// Input threadblock fragment tile
  using ThreadblockFragment = typename Base::ThreadblockFragment;

 private:
  /// Underlying iterator
  Base iterator_;

public:
  /// Constructs an iterator
  CUTLASS_HOST_DEVICE
  VectorFragmentIterator(ThreadblockFragment const &threadblock_frag)
      : iterator_(threadblock_frag) {}

  /// Add offset
  CUTLASS_HOST_DEVICE
  void add_offset(int index_offset) {
    iterator_.add_offset(index_offset);
  }

  /// Increments
  CUTLASS_HOST_DEVICE
  VectorFragmentIterator &operator++() {
    add_offset(1);
    return *this;
  }

  CUTLASS_HOST_DEVICE
  void set_index(int idx) {
    iterator_.set_index(idx);
  }

  /// Loads a fragment from the referenced part of the accumulator tile
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    iterator_.load(frag);
  }

};


////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace conv
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
