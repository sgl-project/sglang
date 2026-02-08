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
    \brief Defines iterators used by warp-level matrix multiply operations targeting Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm80.h"

#include "cutlass/platform/platform.h"
#include "cutlass/fast_math.h"

#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {


/// Tile access iterator
/// Each iteration access in the tile is
/// used as multiplicand for one
/// warp-level matrix multiplication
template <
    /// Size of the tile (concept: MatrixShape)
    typename Shape_,
    /// Operand identity
    Operand Operand_,
    /// Data type of A elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Shape of one matrix production operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Delta between *MMA operations (in units of *MMA operations, concept:
    /// MatrixShape)
    int OpDelta_,
    /// Number of threads participating in one matrix operation
    int Threads = 32,
    /// Enable Residual Support
    bool EnableResidual = false,
    /// Number of partitions along K dimension
    int PartitionsK_ = 1
>
class MmaTensorOpMultiplicandTileAccessIterator {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  /// Basic check
  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = Layout_;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Number of elements accessed per Shared Memory load
  static int const kElementsPerAccess = 
    (sizeof_bits<Element>::value >= 32 ? 1 : 32 / sizeof_bits<Element>::value);

  using InstructionCount = MatrixShape<
    Shape::kRow / InstructionShape::kRow,
    Shape::kColumn / InstructionShape::kColumn
  >;

  static int const kIterations = (kOperand == Operand::kA) ? 
    InstructionCount::kColumn : InstructionCount::kRow;


public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<
    Element, 
    (kOperand == Operand::kA) ? 
      (Shape::kRow * InstructionShape::kColumn / kThreads) : 
      (Shape::kColumn * InstructionShape::kRow / kThreads)
  >;

  /// Memory access type
  using AccessType = AlignedArray<Element, kElementsPerAccess>;

private:

  /// Underlying tensor reference
  TensorRef ref_;

  /// Extent of tensor
  MatrixCoord extent_;

  /// Origin
  MatrixCoord origin_;

  /// Used to load residual tile
  bool is_residual_;
  
  /// residual offset of each thread
  TensorCoord residual_offset_;

  /// Iterations in a tile
  int iterations_;

public:
  
  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileAccessIterator(
    TensorRef const &ref, 
    TensorCoord extent,
    int lane_id
  ): ref_(ref), extent_(extent), is_residual_(false), iterations_(0) {
  
    if (kOperand == Operand::kA) {
      origin_ = MatrixCoord(lane_id / 4, (lane_id % 4) * kElementsPerAccess);
    }
    else {
      origin_ = MatrixCoord((lane_id % 4) * kElementsPerAccess, lane_id / 4);
    }

    ref_.add_coord_offset(origin_);

    if(EnableResidual) {
      // compute residual offset
      if (kOperand == Operand::kA) {
        typename TensorCoord::Index residual_size = 
          extent_.column() % Shape::kColumn;
        if(residual_size) {
          is_residual_ = true;
          residual_offset_ = make_Coord(0, residual_size);
        }
      }
      else {
        typename TensorCoord::Index residual_size = 
          extent_.row() % Shape::kRow;
        if(residual_size) {
          is_residual_ = true;
          residual_offset_ = make_Coord(residual_size, 0);
        }
      }
    }
  }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileAccessIterator(
    TensorRef const &ref, 
    int lane_id
  ): MmaTensorOpMultiplicandTileAccessIterator(ref,
    {Shape::kRow, Shape::kColumn}, lane_id) {
  }
 
  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileAccessIterator &add_tile_offset(TensorCoord const &tile_offset) {

    TensorCoord coord_offset(tile_offset.row() * Shape::kRow, tile_offset.column() * Shape::kColumn);
    origin_ += coord_offset;

    ref_.add_coord_offset(coord_offset);


    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  void advance() {

    if(EnableResidual && is_residual_) {
      is_residual_ = false;

      origin_ += residual_offset_;
      ref_.add_coord_offset(residual_offset_);

    }

    else {
      if (kOperand == Operand::kA) {
        add_tile_offset({0, 1});
      }
      else {
        add_tile_offset({1, 0});
      }
    }

    iterations_ = 0;
  }

  /// increase iterations in a tile
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileAccessIterator & operator++() {

    iterations_++;

    if(iterations_ >= kIterations)
      advance();
    
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    int const kWarpShapeDivisibleInner =
      (kOperand == Operand::kA ? InstructionShape::kColumn : InstructionShape::kRow);

    // Take advantage of Tensor Op's 8 x 4T access pattern
    int const kAccessesInner = (kWarpShapeDivisibleInner / kElementsPerAccess) / 4;

    AccessType *access_ptr = reinterpret_cast<AccessType *>(&frag);

    if (kOperand == Operand::kA) {
      int const kTilesPerInstruction = InstructionShape::kRow / 8;

      CUTLASS_PRAGMA_UNROLL
      for (int inst_m_idx = 0; inst_m_idx < InstructionCount::kRow; ++inst_m_idx) {

        CUTLASS_PRAGMA_UNROLL
        for (int inner_idx = 0; inner_idx < kAccessesInner; ++inner_idx) {

          CUTLASS_PRAGMA_UNROLL
          for (int access_m_idx = 0; access_m_idx < kTilesPerInstruction; ++access_m_idx) {
            int access_idx = 
              access_m_idx + kTilesPerInstruction * (inner_idx + kAccessesInner * inst_m_idx);
            
            MatrixCoord offset(
              access_m_idx * 8 + inst_m_idx * InstructionShape::kRow, 
              inner_idx * 4 * kElementsPerAccess + iterations_ * InstructionShape::kColumn);

            MatrixCoord access_coord = origin_ + offset;

//            if(access_coord.row() < extent_.row() && access_coord.column() < extent_.column()) {

              access_ptr[access_idx] = *reinterpret_cast<AccessType const *>(
                ref_.data() + ref_.offset(offset));
//            }
//            else {
//              AccessType zero;
//              zero.clear();
//              access_ptr[access_idx] = zero;
//            }
          }
        }
      }
    }
    else {
      CUTLASS_PRAGMA_UNROLL
      for (int inst_n_idx = 0; inst_n_idx < InstructionCount::kColumn; ++inst_n_idx) {

        CUTLASS_PRAGMA_UNROLL
        for (int inner_idx = 0; inner_idx < kAccessesInner; ++inner_idx) {
          int access_idx = inner_idx + kAccessesInner * inst_n_idx;

          MatrixCoord offset(
            inner_idx * 4 * kElementsPerAccess + iterations_ * InstructionShape::kRow,
            inst_n_idx * 8);

          MatrixCoord access_coord = origin_ + offset;

//          if(access_coord.row() < extent_.row() && access_coord.column() < extent_.column()) {
              
            access_ptr[access_idx] = *reinterpret_cast<AccessType const *>(
              ref_.data() + ref_.offset(offset));
//          }
//          else {
//              AccessType zero;
//              zero.clear();
//              access_ptr[access_idx] = zero;
//          }
        }
      } 
    }
  }

};



////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
