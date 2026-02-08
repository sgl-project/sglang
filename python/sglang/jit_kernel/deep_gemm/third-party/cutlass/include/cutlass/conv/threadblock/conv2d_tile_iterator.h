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
    \brief Template wraps the tile access iterator concept to load whole tiles from tensors in
      memory used for implicit GEMM convolution.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TileAccessIterator_>
class TileIterator {
public:
  using TileAccessIterator = TileAccessIterator_;

  using Shape = typename TileAccessIterator::Shape;
  using Element = typename TileAccessIterator::Element;
  using Layout = typename TileAccessIterator::Layout;
  using TensorCoord = typename Layout::TensorCoord;
  using ThreadMap = typename TileAccessIterator::ThreadMap;
  using AccessType = typename TileAccessIterator::AccessType;
  using TensorRef = typename TileAccessIterator::TensorRef;
  using Index = typename TileAccessIterator::Index;
  using LongIndex = typename TileAccessIterator::LongIndex;
  static IteratorAlgorithm const kIteratorAlgorithm = TileAccessIterator::kIteratorAlgorithm;
  static StrideSupport const kStrideSupport = TileAccessIterator::kStrideSupport;
  using Params = typename TileAccessIterator::Params;
  static int const kConvDim = TileAccessIterator::kConvDim;
  using ConvProblemSize = typename TileAccessIterator::ConvProblemSize;
  static int const kAccessesPerVector = TileAccessIterator::kAccessesPerVector;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<
    Element, 
    ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

private:

  /// Internal state
  TileAccessIterator tile_access_iterator_;

public:

  /// Constructor
  CUTLASS_HOST_DEVICE
  TileIterator(
    Params const &params,
    ConvProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    tile_access_iterator_(params, problem_size, ptr, thread_idx, threadblock_offset) { }

  CUTLASS_HOST_DEVICE
  static Params getParams(ConvProblemSize const &problem_size, Layout const &layout) {
    return TileAccessIterator::getParams(problem_size, layout);
  }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(Index index) {
    tile_access_iterator_.set_iteration_index(index);
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    tile_access_iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  TileIterator &operator++() {
    tile_access_iterator_.advance();
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  TileIterator operator++(int) {
    TileIterator self(*this);
    operator++();
    return self;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {

    frag.clear();
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {

          int idx = v + kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);

          cutlass::arch::global_load<
            AccessType,
            sizeof(AccessType)
          >(
            frag_ptr[idx],
            tile_access_iterator_.get() + pointer_offset,
            tile_access_iterator_.valid()
          );
  
          ++tile_access_iterator_;
        }
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    tile_access_iterator_.set_iteration_index(0);
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_DEVICE
  void advance() {
    tile_access_iterator_.advance();
  }

  /// Determines whether the Implicit GEMM can execute the given problem.
  CUTLASS_HOST_DEVICE
  static Status can_implement(ConvProblemSize const &problem_size) {

    // dispatch to iterator implementation
    return TileAccessIterator::can_implement(problem_size);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Strided Dgrad Tile Iterator
template <typename TileAccessIterator_>
class TileIteratorStridedDgrad {
public:
  using TileAccessIterator = TileAccessIterator_;

  using Shape = typename TileAccessIterator::Shape;
  using Element = typename TileAccessIterator::Element;
  using Layout = typename TileAccessIterator::Layout;
  using TensorCoord = typename Layout::TensorCoord;
  using ThreadMap = typename TileAccessIterator::ThreadMap;
  using AccessType = typename TileAccessIterator::AccessType;
  using TensorRef = typename TileAccessIterator::TensorRef;
  using Index = typename TileAccessIterator::Index;
  using LongIndex = typename TileAccessIterator::LongIndex;
  static IteratorAlgorithm const kIteratorAlgorithm = TileAccessIterator::kIteratorAlgorithm;
  static StrideSupport const kStrideSupport = TileAccessIterator::kStrideSupport;
  using Params = typename TileAccessIterator::Params;
  static int const kConvDim = TileAccessIterator::kConvDim;
  using ConvProblemSize = typename TileAccessIterator::ConvProblemSize;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<
    Element, 
    ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

private:

  /// Internal state
  TileAccessIterator tile_access_iterator_;

public:

  /// Constructor (output gradient (Dy) OperandA ctor)
  CUTLASS_HOST_DEVICE
  TileIteratorStridedDgrad(
    Params const &params,
    ConvProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    FastDivmod const &stride_h_divmod, FastDivmod const &stride_w_divmod,
    int start_r, int start_s,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    tile_access_iterator_(
      params, 
      problem_size, 
      ptr, 
      thread_idx, 
      stride_h_divmod, stride_w_divmod, 
      start_r, start_s, 
      threadblock_offset) { }

  /// Constructor (filter (w) OperandB ctor)
  CUTLASS_HOST_DEVICE
  TileIteratorStridedDgrad(
    Params const &params,
    ConvProblemSize const &problem_size,
    Element const *ptr,
    int thread_idx,
    int start_r, int start_s,
    MatrixCoord const &threadblock_offset = MatrixCoord()
  ):
    tile_access_iterator_(params, 
      problem_size, 
      ptr, 
      thread_idx, 
      start_r, start_s, 
      threadblock_offset) { }

  CUTLASS_HOST_DEVICE
  static Params getParams(ConvProblemSize const &problem_size, Layout const &layout) {
    return TileAccessIterator::getParams(problem_size, layout);
  }


  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    tile_access_iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  TileIteratorStridedDgrad &operator++() {
    tile_access_iterator_.advance();
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  TileIteratorStridedDgrad operator++(int) {
    TileIteratorStridedDgrad self(*this);
    operator++();
    return self;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {

    frag.clear();
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        cutlass::arch::global_load<
          AccessType,
          sizeof(AccessType)
        >(
          frag_ptr[c + s * ThreadMap::Iterations::kContiguous],
          tile_access_iterator_.get() + pointer_offset,
          tile_access_iterator_.valid()
        );

        ++tile_access_iterator_;
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    tile_access_iterator_.set_iteration_index(0);
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_DEVICE
  void advance() {
    tile_access_iterator_.advance();
  }

  /// Determines whether the Implicit GEMM can execute the given problem.
  CUTLASS_HOST_DEVICE
  static Status can_implement(ConvProblemSize const &problem_size) {

    // dispatch to iterator implementation
    return TileAccessIterator::can_implement(problem_size);
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

