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
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/permute.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/epilogue/threadblock/output_tile_thread_map.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_params.h"
#include "cutlass/conv/conv2d_problem_size.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////

namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load and store output tile from global memory in epilogue.
///
/// Satisfies: ReadableTileIterator | PredicatedTileIterator | ForwardTileIterator
///
template <
  typename ThreadMap_,       ///< Thread map (conept: PitchLinearThreadMap)
  typename Element_,         ///< Element data type
  typename ThreadOutputShape_ = cutlass::conv::TensorNHWCShape<1, 1, 1, 1>,
  typename ThreadBlockOutputShape_ = cutlass::conv::TensorNHWCShape<1, 1, 1, 1>
>
class PredicatedTileIteratorDirectConv {
public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;
  using ThreadOutputShape = ThreadOutputShape_;
  using ThreadBlockOutputShape = ThreadBlockOutputShape_;

  using Element = Element_;

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;

  using ConvProblemSize = typename cutlass::conv::Conv2dProblemSize;

  /// Fragment object
  using Fragment = Array<Element, ThreadMap::Iterations::kCount * kElementsPerAccess>;

  /// Memory access size
  using AccessType = AlignedArray<Element, kElementsPerAccess>;

  static int const kLoadsPerAccess = AccessType::kElements / AccessType::kElements;

  using ThreadTileCount = MatrixShape<
    ThreadBlockOutputShape::kH / ThreadOutputShape::kH,
    ThreadBlockOutputShape::kW / ThreadOutputShape::kW
  >;

  //
  // Parameters struct
  //

  /// Uses a non-template class
  struct Params : PredicatedTileIteratorDirect2dConvParams {
    using Base = PredicatedTileIteratorDirect2dConvParams;

    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(Layout const &layout, cutlass::conv::Conv2dProblemSize const &problem_size): 
      PredicatedTileIteratorDirect2dConvParams(
        layout.stride(0) * int(sizeof(AccessType)) / kElementsPerAccess,
        problem_size,
        {ThreadBlockOutputShape::kH, ThreadBlockOutputShape::kW}
      ) 
    { }

    CUTLASS_HOST_DEVICE
    Params(Base const &base) : 
      Base(base) { }
  };

  /// Mask object
  struct Mask {

    static int const kCount = ThreadMap::Iterations::kContiguous;

    /// Predicate state
    bool predicates[kCount];

    //
    // Mask
    //
    CUTLASS_HOST_DEVICE
    Mask() {
      enable();
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_HOST_DEVICE void clear() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = false;
      }
    }

    ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
    CUTLASS_DEVICE void enable() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = true;
      }
    }
  };

private:

  //
  // Data members
  //

  /// Parameters structure containing reference and precomputed state.
  PredicatedTileIteratorDirect2dConvParams params_;

  /// Byte-level pointer
  uint8_t *byte_pointer_;

  ///     
  Element *pointer_;


  /// Array of boolean values to contain steady-state predicates
  Mask mask_;

  /// Extent of the matrix tile in rows
  Index extent_row_;

  /// Extent of the matrix tile in rows
  Index extent_column_;

  /// A thread's starting row position (assuming steady-state predicates have been computed)
  Index thread_start_row_;

  /// A thread's starting column
  Index thread_start_column_;

  /// Initial thread output location
  int thread_start_n_, thread_start_p_, thread_start_q_;

  /// Current threadblock tile index
  int tile_index_;

  //
  // Static asserts about internal strides
  //

  static_assert(sizeof(extent_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(PredicatedTileIteratorDirect2dConvParams::stride) == 8, "Expected 64b strides");

private:

  //
  // Methods
  //



public:

  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE
  PredicatedTileIteratorDirectConv(
    PredicatedTileIteratorDirect2dConvParams const & params,
    Element *pointer,
    TensorCoord extent,
    int thread_idx,
    TensorCoord threadblock_offset = TensorCoord()
  ): 
    params_(params), pointer_(pointer)
  {

    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);

    extent_row_ = extent.row();
    extent_column_ = extent.column();

    // stride dim (PQ)
    thread_start_row_ = thread_offset.column();
    // contiguous dim (Channels)
    thread_start_column_ = threadblock_offset.column() + thread_offset.row();

    tile_index_ = threadblock_offset.row();

    set_tile_index(0);
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void set_tile_index(const int index) { 
   
    int residual;
    params_.pq_divmod(thread_start_n_, residual, tile_index_ + index);
    params_.q_divmod(thread_start_p_, thread_start_q_, residual);

    // Compute the base output coord of ThreadBlock
    thread_start_p_ *= ThreadBlockOutputShape::kH;
    thread_start_q_ *= ThreadBlockOutputShape::kW;

    // Initialize predicates
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
      mask_.predicates[c] = ((thread_start_column_ 
        + c * ThreadMap::Delta::kContiguous) < extent_column_);
    }

    // Null pointer performs no accesses
    if (!pointer_) {
      mask_.clear();
    }

  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, int64_t byte_offset) const {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int frag_base_idx = s * ThreadMap::Iterations::kContiguous + c;

        int current_row = thread_start_row_ + s * ThreadMap::Delta::kStrided;
        int p = current_row / ThreadBlockOutputShape::kW;
        int q = current_row % ThreadBlockOutputShape::kW;

        int current_p = thread_start_p_ + p;
        int current_q = thread_start_q_ + q;

        bool row_guard = (current_p) < params_.P && (current_q) < params_.Q &&
                         (thread_start_n_ < params_.N) && current_row < ThreadMap::Shape::kStrided;

        int output_row_offset =
            thread_start_n_ * params_.stride_n + current_p * params_.stride_p + current_q;

        uint8_t *byte_pointer =
            reinterpret_cast<uint8_t *>(pointer_) +
            LongIndex(output_row_offset) * LongIndex(params_.stride) +
            LongIndex(thread_start_column_ + c * ThreadMap::Delta::kContiguous) *
                sizeof(AccessType) / kElementsPerAccess;

        AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

        AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

        bool guard = row_guard && mask_.predicates[c];

        cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
            frag_ptr[frag_base_idx], (void *)&memory_pointer[0], guard);
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) const {
    load_with_byte_offset(frag, 0);
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, int64_t byte_offset) const {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int frag_base_idx = s * ThreadMap::Iterations::kContiguous + c;

        int current_row = thread_start_row_ + s * ThreadMap::Delta::kStrided;
        int p = current_row / ThreadBlockOutputShape::kW;
        int q = current_row % ThreadBlockOutputShape::kW;

        int current_p = thread_start_p_ + p;
        int current_q = thread_start_q_ + q;

        bool row_guard = (current_p) < params_.P && (current_q) < params_.Q &&
                         (thread_start_n_ < params_.N) && current_row < ThreadMap::Shape::kStrided;

        int output_row_offset =
            thread_start_n_ * params_.stride_n + current_p * params_.stride_p + current_q;

        uint8_t *byte_pointer =
            reinterpret_cast<uint8_t *>(pointer_) +
            LongIndex(output_row_offset) * LongIndex(params_.stride) +
            LongIndex(thread_start_column_ + c * ThreadMap::Delta::kContiguous) *
                sizeof(AccessType) / kElementsPerAccess;

        AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

        AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

        bool guard = row_guard && mask_.predicates[c];

        cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
            frag_ptr[frag_base_idx], (void *)&memory_pointer[0], guard);
      }
    }
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) const {

    store_with_byte_offset(frag, 0);
  }

  CUTLASS_DEVICE
  MatrixCoord thread_start() const {
    return MatrixCoord(thread_start_row_, thread_start_column_);
  }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_row() const {
    return thread_start_row_;
  }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_column() const {
    return thread_start_column_;
  }

  /// Extent of the matrix in rows
  CUTLASS_DEVICE
  Index extent_row() const {
    return extent_row_;
  }

  /// Extent of the matrix in columns
  CUTLASS_DEVICE
  Index extent_column() const {
    return extent_column_;
  }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorDirectConv &operator++() {
    // do nothing

    return *this;
  }

  ///< Efficiently disables all accesses guarded by mask
  CUTLASS_DEVICE void clear_mask() {
    mask_.clear();
  }

  ///< Efficiently enables all accesses guarded by mask
  CUTLASS_DEVICE void enable_mask() {
    mask_.enable();
  }

  ///< Sets the mask
  CUTLASS_DEVICE void get_mask(Mask &mask) const {
    mask = mask_;
  }

  ///< Sets the mask
  CUTLASS_DEVICE void set_mask(Mask const &mask) {
    mask_ = mask;
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
