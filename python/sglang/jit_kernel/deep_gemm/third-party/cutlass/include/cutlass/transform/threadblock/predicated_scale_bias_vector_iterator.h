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
    \brief Templates calculating the address and predicates to the load of scale and bias vectors.

    This iterator uses masks to guard out-of-bounds accesses.

    This can be used to load var and mean vectors in layernorm which is loop invariant.

    A precomputed "Params" object minimizes the amount of state that must be
   stored in registers, and integer addition is used to advance the pointer
   through memory.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// PredicatedScaleBiasVectorIterator
///
template <typename WarpShape,
          typename Element,
          typename Layout>
class PredicatedScaleBiasVectorIterator;

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator for wgrad pitch-linear data.
///
template <typename WarpShape_, typename Element_>
class PredicatedScaleBiasVectorIterator<WarpShape_,
                                        Element_,
                                        layout::PitchLinear> {
 public:

  using WarpShape = WarpShape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ConstPointer = const Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  static int const kElementsPerAccess = 1;

  using AccessType = AlignedArray<Element, kElementsPerAccess>;

  static int const kIterations = WarpShape::kContiguous / 8;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<__half2, 2 * kIterations * kElementsPerAccess>;

 private:
  //
  // Data members
  //

  /// Internal pointer to first access of tile
  ConstPointer scale_pointer_;
  ConstPointer bias_pointer_;

  /// Size of tensor
  int problem_size_;

  int32_t thread_offset_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedScaleBiasVectorIterator(
      /// Extent of tensor
      int problem_size,
      /// Pointer to the start of the scale vector
      ConstPointer scale_pointer,
      /// Pointer to the start of the bias vector
      ConstPointer bias_pointer,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : problem_size_(problem_size),
        scale_pointer_(scale_pointer),
        bias_pointer_(bias_pointer) {

    thread_offset_ = threadblock_offset.contiguous() + (thread_id % 32) / 4;
  }

  /// Construct a PredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedScaleBiasVectorIterator(
      /// Extent of tensor
      int problem_size,
      /// Pointer to start of scale vector
      ConstPointer scale_pointer,
      /// Pointer to start of scale vector
      ConstPointer bias_pointer,
      ///< ID of each participating thread
      int thread_id)
      : PredicatedScaleBiasVectorIterator(problem_size,
                                          scale_pointer, bias_pointer,
                                          thread_id, make_Coord(0, 0)) {}

  /// Advances an iterator along logical dimensions of matrix in units of whole warp tiles
  CUTLASS_DEVICE
  void add_tile_offset(
      TensorCoord const &tile_offset) {

    thread_offset_ += (WarpShape::kContiguous * tile_offset.contiguous());
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {

    frag.fill(__float2half2_rn(0.0f));
    __half2 *frag_ptr = reinterpret_cast<__half2 *>(&frag);

    // load scale
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kIterations; ++c) {

      cutlass::arch::global_load<
        __half,
        sizeof(AccessType)
      >(
        frag_ptr[c * 2].x,
        scale_pointer_ + thread_offset_ + c * 8,
        (thread_offset_ + c * 8) < problem_size_ 
      );
    }

    // load bias
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kIterations; ++c) {

      cutlass::arch::global_load<
        __half,
        sizeof(AccessType)
      >(
        frag_ptr[c * 2 + 1].x,
        bias_pointer_ + thread_offset_ + c * 8,
        (thread_offset_ + c * 8) < problem_size_ 
      );
    }

    // duplicate scale
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kIterations; ++c) {
      frag_ptr[c * 2].y = frag_ptr[c * 2].x;
    }

    // duplicate bias
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kIterations; ++c) {
      frag_ptr[c * 2 + 1].y = frag_ptr[c * 2 + 1].x;
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator for row-major data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename WarpShape_,
          typename Element_>
class PredicatedScaleBiasVectorIterator<WarpShape_,
                                        Element_,
                                        layout::RowMajor> {
 public:

  using WarpShape = WarpShape_;
  using Element = Element_;
  using Layout = layout::RowMajor;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ConstPointer = const Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PredicatedScaleBiasVectorIterator<
      layout::PitchLinearShape<WarpShape::kColumn, WarpShape::kRow>,
      Element,
      layout::PitchLinear>;

  using AccessType = typename UnderlyingIterator::AccessType;
  static int const kElementsPerAccess = UnderlyingIterator::kElementsPerAccess;
  using Fragment = typename UnderlyingIterator::Fragment;


 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedScaleBiasVectorIterator(
      ///< Extent of tensor
      int problem_size,
      ///< Pointer to the start of the scale vector
      ConstPointer scale_pointer,
      ///< Pointer to the start of the bias vector
      ConstPointer bias_pointer,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : iterator_(problem_size, scale_pointer, bias_pointer,
                  thread_id,
                  layout::PitchLinearCoord(threadblock_offset.column(),
                                           threadblock_offset.row())) {}

  /// Construct a PredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedScaleBiasVectorIterator(
      int problem_size,  ///< Extent of tensor
      ConstPointer scale_pointer,  ///< Pointer to the start of the scale vector
      ConstPointer bias_pointer,   ///< Pointer to the start of the bias vector
      int thread_id                ///< ID of each participating thread
      )
      : PredicatedScaleBiasVectorIterator(problem_size,
                                          scale_pointer, bias_pointer,
                                          thread_id, make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// threadblock tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    iterator_.load(frag);
  }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace transform 
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
