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
#include "cutlass/conv/threadblock/conv2d_params.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// PredicatedScaleBiasVectorAccessIterator
///
template <typename ThreadblockShape,
          typename Element,
          typename Layout>
class PredicatedScaleBiasVectorAccessIterator;

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIterator for fprop pitch-linear data.
///
template <typename ThreadblockShape_, typename Element_>
class PredicatedScaleBiasVectorAccessIterator<ThreadblockShape_,
                                              Element_,
                                              layout::PitchLinear> {
 public:

  using ThreadblockShape = ThreadblockShape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ConstPointer = const Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  static int const kElementsPerAccess = 128 / sizeof_bits<Element>::value;
  static int const kThreads = ThreadblockShape::kContiguous / kElementsPerAccess;

  using AccessType = AlignedArray<Element, kElementsPerAccess>;

  using Params = PredicatedScaleBiasVectorAccessIteratorParams;

 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char *;

 private:
  //
  // Data members
  //

  /// Parameters object with precomputed internal state
  Params const &params_;

  /// Internal pointer to first access of tile
  BytePointer pointer_;

  int problem_size_trs;
  int problem_size_c;
  int filter_trs_;

  TensorCoord thread_offset_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedScaleBiasVectorAccessIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Extent of tensor
      Conv2dProblemSize const &problem_size,
      /// Pointer to the start of the scale vector
      ConstPointer scale_pointer,
      /// Pointer to the start of the bias vector
      ConstPointer bias_pointer,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : params_(params),
        problem_size_trs(problem_size.R * problem_size.S),
        problem_size_c(problem_size.C),
        filter_trs_(0) {
    pointer_ = (thread_id < kThreads)
                   ? reinterpret_cast<BytePointer>(
                         const_cast<NonConstPointer>(scale_pointer))
                   : reinterpret_cast<BytePointer>(
                         const_cast<NonConstPointer>(bias_pointer));

    // Per-thread offset in logical coordinates of tensor
    int thread_base = (thread_id < kThreads) ? 0 : kThreads;

    thread_offset_ =
        threadblock_offset +
        TensorCoord((thread_id - thread_base) * kElementsPerAccess, 0);

    set_iteration_index(0);
  }

  CUTLASS_HOST_DEVICE
  PredicatedScaleBiasVectorAccessIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Extent of tensor
      Conv3dProblemSize const &problem_size,
      /// Pointer to the start of the scale vector
      ConstPointer scale_pointer,
      /// Pointer to the start of the bias vector
      ConstPointer bias_pointer,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : params_(params),
        problem_size_trs(problem_size.T * problem_size.R * problem_size.S),
        problem_size_c(problem_size.C),
        filter_trs_(0) {
    pointer_ = (thread_id < kThreads)
                   ? reinterpret_cast<BytePointer>(
                         const_cast<NonConstPointer>(scale_pointer))
                   : reinterpret_cast<BytePointer>(
                         const_cast<NonConstPointer>(bias_pointer));

    // Per-thread offset in logical coordinates of tensor
    int thread_base = (thread_id < kThreads) ? 0 : kThreads;

    thread_offset_ =
        threadblock_offset +
        TensorCoord((thread_id - thread_base) * kElementsPerAccess, 0);

    set_iteration_index(0);
  }

  /// Construct a PredicatedTileAccessIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedScaleBiasVectorAccessIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Extent of tensor
      Conv2dProblemSize const &problem_size,
      /// Pointer to start of scale vector
      ConstPointer scale_pointer,
      /// Pointer to start of scale vector
      ConstPointer bias_pointer,
      ///< ID of each participating thread
      int thread_id)
      : PredicatedScaleBiasVectorAccessIterator(params, problem_size,
                                                scale_pointer, bias_pointer,
                                                thread_id, make_Coord(0, 0)) {}

  CUTLASS_HOST_DEVICE
  PredicatedScaleBiasVectorAccessIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Extent of tensor
      Conv3dProblemSize const &problem_size,
      /// Pointer to start of scale vector
      ConstPointer scale_pointer,
      /// Pointer to start of scale vector
      ConstPointer bias_pointer,
      ///< ID of each participating thread
      int thread_id)
      : PredicatedScaleBiasVectorAccessIterator(params, problem_size,
                                                scale_pointer, bias_pointer,
                                                thread_id, make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {}

  /// Advances an iterator along logical dimensions of matrix in units of whole threadblock tiles
  CUTLASS_DEVICE
  void add_tile_offset(
      TensorCoord const &tile_offset) {
    thread_offset_ =
        thread_offset_ +
        TensorCoord(ThreadblockShape::kContiguous * tile_offset.contiguous(), 0);
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {

    return reinterpret_cast<AccessType *>(
        pointer_ +
        (thread_offset_.contiguous() * sizeof_bits<Element>::value / 8));
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  PredicatedScaleBiasVectorAccessIterator &operator++() {
    return *this;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  void advance() {
    // moves to the next tile
    ++filter_trs_;
    if (filter_trs_ == problem_size_trs) {
      filter_trs_ = 0;
      add_tile_offset(TensorCoord(1, 0));
    }
  }

  /// Increment and return an instance to self.
  CUTLASS_DEVICE
  PredicatedScaleBiasVectorAccessIterator operator++(int) {
    PredicatedScaleBiasVectorAccessIterator self(*this);
    operator++();
    return self;
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    uint32_t enabled = 0;

#if defined(_MSC_VER) || (__CUDACC_VER_MAJOR__ < 11)
    enabled = threadIdx.x < kThreads * 2;
#else
    asm volatile(
        "{\n"
        "  .reg .u32 tid_reg;\n"
        "  .reg .pred p;\n"
        "  mov.u32 tid_reg, %%tid.x;\n"
        "  setp.lt.u32 p, tid_reg, %1;\n"
        "  selp.u32 %0, 1, 0, p;\n"
        "}\n" : "+r"(enabled) :"n"(kThreads * 2));
#endif

    return ((thread_offset_.contiguous() < problem_size_c) && enabled);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIterator for row-major data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename ThreadblockShape_,
          typename Element_>
class PredicatedScaleBiasVectorAccessIterator<ThreadblockShape_,
                                        Element_,
                                        layout::RowMajor> {
 public:

  using ThreadblockShape = ThreadblockShape_;
  using Element = Element_;
  using Layout = layout::RowMajor;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ConstPointer = const Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PredicatedScaleBiasVectorAccessIterator<
      layout::PitchLinearShape<ThreadblockShape::kColumn, ThreadblockShape::kRow>,
      Element,
      layout::PitchLinear>;

  using AccessType = typename UnderlyingIterator::AccessType;
  static int const kElementsPerAccess = UnderlyingIterator::kElementsPerAccess;

  using Params = PredicatedScaleBiasVectorAccessIteratorParams;

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
  PredicatedScaleBiasVectorAccessIterator(
      ///< Precomputed parameters object
      Params const &params,
      ///< Extent of tensor
      Conv2dProblemSize const &problem_size,
      ///< Pointer to the start of the scale vector
      ConstPointer scale_pointer,
      ///< Pointer to the start of the bias vector
      ConstPointer bias_pointer,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : iterator_(params, problem_size, scale_pointer, bias_pointer,
                  thread_id,
                  layout::PitchLinearCoord(threadblock_offset.column(),
                                           threadblock_offset.row())) {}

  CUTLASS_HOST_DEVICE
  PredicatedScaleBiasVectorAccessIterator(
      ///< Precomputed parameters object
      Params const &params,
      ///< Extent of tensor
      Conv3dProblemSize const &problem_size,
      ///< Pointer to the start of the scale vector
      ConstPointer scale_pointer,
      ///< Pointer to the start of the bias vector
      ConstPointer bias_pointer,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : iterator_(params, problem_size, scale_pointer, bias_pointer,
                  thread_id,
                  layout::PitchLinearCoord(threadblock_offset.column(),
                                           threadblock_offset.row())) {}

  /// Construct a PredicatedTileAccessIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedScaleBiasVectorAccessIterator(
      Params const &params,                   ///< Precomputed parameters object
      Conv2dProblemSize const &problem_size,  ///< Extent of tensor
      ConstPointer scale_pointer,  ///< Pointer to the start of the scale vector
      ConstPointer bias_pointer,   ///< Pointer to the start of the bias vector
      int thread_id                ///< ID of each participating thread
      )
      : PredicatedScaleBiasVectorAccessIterator(params, problem_size,
                                                scale_pointer, bias_pointer,
                                                thread_id, make_Coord(0, 0)) {}

  CUTLASS_HOST_DEVICE
  PredicatedScaleBiasVectorAccessIterator(
      Params const &params,                   ///< Precomputed parameters object
      Conv3dProblemSize const &problem_size,  ///< Extent of tensor
      ConstPointer scale_pointer,  ///< Pointer to the start of the scale vector
      ConstPointer bias_pointer,   ///< Pointer to the start of the bias vector
      int thread_id                ///< ID of each participating thread
      )
      : PredicatedScaleBiasVectorAccessIterator(params, problem_size,
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

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedScaleBiasVectorAccessIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedScaleBiasVectorAccessIterator operator++(int) {
    PredicatedScaleBiasVectorAccessIterator self(*this);
    operator++();
    return self;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  void advance() {
    iterator_.advance();
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return iterator_.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace conv 
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
