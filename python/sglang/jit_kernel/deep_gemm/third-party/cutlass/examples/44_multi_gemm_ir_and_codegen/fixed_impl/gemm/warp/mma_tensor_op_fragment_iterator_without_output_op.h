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

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_conversion.h"

namespace cutlass {
namespace gemm {
namespace warp {


////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Size of the accumulation tile shape (concept: MatrixShape)
    typename AccumulatorShape_,
    /// KBlocks columns to compute residual
    int KBlocksColumn_,
    /// Accumulator Element type
    typename ElementAccumulator_,    
    /// Element type
    typename Element_,
    /// Layout of operand in memory
    typename Layout_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Whether beta is zero
    bool IsBetaZero_ >
class MmaTensorOpPureFragmentIterator;


// Partial specialization for col-major accumulator tile
// And Element type is the same as Accumulator Element type

template <
    /// Shape of warp tile to load (concept: MatrixShape)
    typename Shape_,
    /// Shape of the warp accumulation tile (concept: MatrixShape)
    typename AccumulatorShape_,
    /// KBlocks columns to compute residual
    int KBlocksColumn_,    
    /// Element type
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_>
class MmaTensorOpPureFragmentIterator<Shape_, AccumulatorShape_, KBlocksColumn_, Element_, Element_,
                                         cutlass::layout::ColumnMajor,
                                         InstructionShape_, true> {
 public:

  /// Shape of warp tile to load (concept: MatrixShape)
  using Shape = Shape_;
    
  /// Shape of the warp accumulation tile (concept: MatrixShape)
  using AccumulatorShape = AccumulatorShape_;

  /// KBlocks columns to compute residual
  static int const kKBlockColumn = KBlocksColumn_;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = cutlass::layout::ColumnMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Whether beta is zero
  static bool const IsBetaZero = true;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN),
        "Shape of warp-level Mma must be divisible by operator shape.");
    static_assert(
        !(AccumulatorShape::kRow % Shape::kRow) &&
            !(AccumulatorShape::kColumn % Shape::kColumn),
        "Shape of Warp Accumulator must be divisible by warp shape.");
    static_assert(
        !(kKBlockColumn % Shape::kColumn),
        "KBlock size must be divisible by warp shape.");

    /// Number of times this iterator can be incremented
    static int const kIterations = AccumulatorShape::kCount / Shape::kCount;
  };

private:

  static int const kElementsPerAccess = InstructionShape::kM * InstructionShape::kN / kThreads;

  /// Number of mma operations performed by a warp
  using MmaIterations = MatrixShape<Shape::kRow / InstructionShape::kM,
                                    Shape::kColumn / InstructionShape::kN>;
  /// Number of mma operations performed by the entire accumulator
  using AccumulatorIterations = MatrixShape<AccumulatorShape::kRow / InstructionShape::kM,
                                              AccumulatorShape::kColumn / InstructionShape::kN>;

  /// Number of K iterations    
  static int const kKBlockIterations = (AccumulatorShape::kColumn + kKBlockColumn - 1) / kKBlockColumn;
  static int const kResidualColumn = AccumulatorShape::kColumn - (kKBlockIterations - 1) * kKBlockColumn;
  static int const kKBlockColumnIterations = kKBlockColumn / Shape::kColumn 
                                     * (AccumulatorShape::kRow / Shape::kRow);
  static int const kResidualIndex = kResidualColumn / Shape::kColumn
                                     * (AccumulatorShape::kRow / Shape::kRow);

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<Element, Shape::kCount / kThreads>;

  /// Accumulator Fragment object
  using AccumulatorFragment = Array<Element, AccumulatorShape::kCount / kThreads>;


private:

  /// Internal access type
  using AccessType = Array<Element, kElementsPerAccess>;

private:
  //
  // Data members
  //

  /// Accumulator tile
  AccessType const *accumulators_;

  /// Internal index
  int index_;

  /// Used to access residual tile first
  bool is_residual_tile_;

public:
  /// Constructs an iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpPureFragmentIterator(AccumulatorFragment const &accum)
      : accumulators_(reinterpret_cast<AccessType const *>(&accum)),
        index_(0), is_residual_tile_(true) {}

  /// Add offset
  CUTLASS_HOST_DEVICE
  void add_offset(int index_offset) {
    index_ += index_offset; 
    if(is_residual_tile_ && index_ >= kKBlockColumnIterations) {
      index_ = index_ - kKBlockColumnIterations + kResidualIndex;
      is_residual_tile_ = false;
    }
  }

  /// Increments
  CUTLASS_HOST_DEVICE
  MmaTensorOpPureFragmentIterator &operator++() {
    add_offset(1);
    return *this;
  }

  /// Decrements
  CUTLASS_HOST_DEVICE
  MmaTensorOpPureFragmentIterator &operator--() {
    add_offset(-1);
    return *this;
  }

  /// Loads a fragment from the referenced part of the accumulator tile
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    AccessType src_fragment;
    src_fragment.clear();


    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    int index_m = (index_ * MmaIterations::kRow) % AccumulatorIterations::kRow;
    int index_n = (index_ * MmaIterations::kRow) / AccumulatorIterations::kRow 
                    * MmaIterations::kColumn;

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < MmaIterations::kColumn; n++) {
      for (int m = 0; m < MmaIterations::kRow; m++) {
        int accumulator_access_offset = 
            (n + index_n) * AccumulatorIterations::kRow + m + index_m;
            
        frag_ptr[n * MmaIterations::kRow + m].clear();
        if(!(is_residual_tile_ && index_ >= kResidualIndex))
            frag_ptr[n * MmaIterations::kRow + m] = accumulators_[accumulator_access_offset];
            // frag_ptr[n * MmaIterations::kRow + m] = output_op(accumulators_[accumulator_access_offset], src_fragment);
      }
    }
  }

};

// Partial specialization for row-major accumulator tile

template <
    /// Shape of warp tile to load (concept: MatrixShape)
    typename Shape_,
    /// Shape of the warp accumulation tile (concept: MatrixShape)
    typename AccumulatorShape_,
    /// KBlocks columns to compute residual
    int KBlocksColumn_,    
    /// Accumulator Element type
    typename ElementAccumulator_,    
    /// Element type
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_>
class MmaTensorOpPureFragmentIterator<Shape_, AccumulatorShape_, KBlocksColumn_, ElementAccumulator_, Element_,
                                         cutlass::layout::RowMajor,
                                         InstructionShape_, true> {
 public:

  /// Shape of warp tile to load (concept: MatrixShape)
  using Shape = Shape_;
    
  /// Shape of the warp accumulation tile (concept: MatrixShape)
  using AccumulatorShape = AccumulatorShape_;

  /// KBlocks columns to compute residual
  static int const kKBlockColumn = KBlocksColumn_;

  /// Accumulator Element type
  using ElementAccumulator = ElementAccumulator_;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = cutlass::layout::RowMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Whether beta is zero
  static bool const IsBetaZero = true;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN),
        "Shape of warp-level Mma must be divisible by operator shape.");
    static_assert(
        !(AccumulatorShape::kRow % Shape::kRow) &&
            !(AccumulatorShape::kColumn % Shape::kColumn),
        "Shape of Warp Accumulator must be divisible by warp shape.");
    static_assert(
        !(kKBlockColumn % Shape::kColumn),
        "KBlock size must be divisible by warp shape.");

    /// Number of times this iterator can be incremented
    static int const kIterations = AccumulatorShape::kCount / Shape::kCount;
  };

private:

  static int const kElementsPerAccess = InstructionShape::kM * InstructionShape::kN / kThreads;

  /// Number of mma operations performed by a warp
  using MmaIterations = MatrixShape<Shape::kRow / InstructionShape::kM,
                                    Shape::kColumn / InstructionShape::kN>;
  /// Number of mma operations performed by the entire accumulator
  using AccumulatorIterations = MatrixShape<AccumulatorShape::kRow / InstructionShape::kM,
                                              AccumulatorShape::kColumn / InstructionShape::kN>;

  /// Number of K iterations    
  static int const kKBlockIterations = (AccumulatorShape::kColumn + kKBlockColumn - 1) / kKBlockColumn;
  static int const kResidualColumn = AccumulatorShape::kColumn - (kKBlockIterations - 1) * kKBlockColumn;
  static int const kKBlockColumnIterations = kKBlockColumn / Shape::kColumn 
                                     * (AccumulatorShape::kRow / Shape::kRow);
  static int const kResidualIndex = kResidualColumn / Shape::kColumn
                                     * (AccumulatorShape::kRow / Shape::kRow);

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<Element, Shape::kCount / kThreads>;

  /// Accumulator Fragment object
  using AccumulatorFragment = Array<ElementAccumulator, AccumulatorShape::kCount / kThreads>;


private:

  /// Internal access type
  using AccessType = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentAccessType = Array<Element, kElementsPerAccess>;

private:
  //
  // Data members
  //

  /// Accumulator tile
  AccessType const *accumulators_;

  /// Internal index
  int index_;

  /// Used to access residual tile first
  bool is_residual_tile_;

public:
  /// Constructs an iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpPureFragmentIterator(AccumulatorFragment const &accum)
      : accumulators_(reinterpret_cast<AccessType const *>(&accum)),
        index_(0), is_residual_tile_(true) {}

  /// Add offset
  CUTLASS_HOST_DEVICE
  void add_offset(int index_offset) {
    index_ += index_offset; 
    if(is_residual_tile_ && index_ >= kKBlockColumnIterations) {
      index_ = index_ - kKBlockColumnIterations + kResidualIndex;
      is_residual_tile_ = false;
    }
  }

  /// Increments
  CUTLASS_HOST_DEVICE
  MmaTensorOpPureFragmentIterator &operator++() {
    add_offset(1);
    return *this;
  }

  /// Decrements
  CUTLASS_HOST_DEVICE
  MmaTensorOpPureFragmentIterator &operator--() {
    add_offset(-1);
    return *this;
  }

  /// Loads a fragment from the referenced part of the accumulator tile
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {


    FragmentAccessType src_fragment;
    src_fragment.clear();

    FragmentAccessType *frag_ptr = reinterpret_cast<FragmentAccessType *>(&frag);

    int index_m = (index_ * MmaIterations::kRow) % AccumulatorIterations::kRow;
    int index_n = (index_ * MmaIterations::kRow) / AccumulatorIterations::kRow 
                    * MmaIterations::kColumn;

    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < MmaIterations::kRow; m++) {
      for (int n = 0; n < MmaIterations::kColumn; n++) {
        int accumulator_access_offset = 
            (m + index_m) * AccumulatorIterations::kColumn + n + index_n;

        frag_ptr[m * MmaIterations::kColumn + n].clear();
        if(!(is_residual_tile_ && index_ >= kResidualIndex))
           frag_ptr[m * MmaIterations::kColumn + n] = (accumulators_[accumulator_access_offset]);
      }
    }
  }

};

////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
