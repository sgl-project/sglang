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
  \brief 
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Predicated tile access iterator descriptor object containing template dependent state
struct PredicatedTileAccessIteratorDesc {

  int element_size_bits = -1;
  int advance_rank = -1;
  layout::PitchLinearCoord threadblock_shape;
  layout::PitchLinearCoord threadmap_iterations;
  layout::PitchLinearCoord threadmap_delta;

  //
  // Methods
  //

  PredicatedTileAccessIteratorDesc() = default;

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorDesc(
    int element_size_bits_,
    int advance_rank_,
    layout::PitchLinearCoord threadblock_shape_,
    layout::PitchLinearCoord threadmap_iterations_,
    layout::PitchLinearCoord threadmap_delta_
  ):
    element_size_bits(element_size_bits_),
    advance_rank(advance_rank_),
    threadblock_shape(threadblock_shape_),
    threadmap_iterations(threadmap_iterations_),
    threadmap_delta(threadmap_delta_)
  {
    #if 0
    printf("PredicatedTileAccessIteratorDesc(%d, %d, {%d, %d}, {%d, %d}, {%d, %d}})\n",
      element_size_bits,
      advance_rank,
      threadblock_shape.contiguous(), threadblock_shape.strided(),
      threadmap_iterations.contiguous(), threadmap_iterations.strided(),
      threadmap_delta.contiguous(), threadmap_delta.strided());
    #endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Helper template to construct an PredicatedTileAccessIteratorDesc from a template 
// dependent state
template <
  typename Shape, typename Element, typename Layout,
  int AdvanceRank, typename ThreadMap>
  struct MakePredicatedTileAccessIteratorDesc;
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIterator for pitch-linear data.
template <
  typename Shape, typename Element, int AdvanceRank, 
  typename ThreadMap>
struct MakePredicatedTileAccessIteratorDesc <
    Shape, Element, layout::PitchLinear, AdvanceRank, ThreadMap> {

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorDesc operator()() {

    return PredicatedTileAccessIteratorDesc(
      sizeof_bits<Element>::value,
      AdvanceRank,
      {Shape::kContiguous, Shape::kStrided},
      {ThreadMap::Iterations::kContiguous, ThreadMap::Iterations::kStrided},
      {ThreadMap::Delta::kContiguous, ThreadMap::Delta::kStrided}
    );
}

};
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIterator for column-major data.
template <
  typename Shape, typename Element, int AdvanceRank, 
  typename ThreadMap>
struct MakePredicatedTileAccessIteratorDesc <
    Shape, Element, layout::ColumnMajor, AdvanceRank, ThreadMap> {

  static int const kAdvanceRank = AdvanceRank;

  using UnderlyingMakeOperator = MakePredicatedTileAccessIteratorDesc<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
      layout::PitchLinear, (kAdvanceRank == 0 ? 0 : 1), ThreadMap>;

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorDesc operator()() {

    return UnderlyingMakeOperator()();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIterator for row-major data.
template <
  typename Shape, typename Element, int AdvanceRank, 
  typename ThreadMap>
struct MakePredicatedTileAccessIteratorDesc <
    Shape, Element, layout::RowMajor, AdvanceRank, ThreadMap> {

  static int const kAdvanceRank = AdvanceRank;

  using UnderlyingMakeOperator = MakePredicatedTileAccessIteratorDesc<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
      layout::PitchLinear, (kAdvanceRank == 0 ? 1 : 0), ThreadMap>;

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorDesc operator()() {

    return UnderlyingMakeOperator()();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIterator for column-major interleaved data.
template <
  typename Shape, typename Element, int AdvanceRank, 
  typename ThreadMap, int InterleavedK>
struct MakePredicatedTileAccessIteratorDesc <
    Shape, Element, layout::ColumnMajorInterleaved<InterleavedK>, AdvanceRank, ThreadMap> {

  static int const kAdvanceRank = AdvanceRank;
  static int const kInterleavedK = InterleavedK;

  using UnderlyingMakeOperator = MakePredicatedTileAccessIteratorDesc<
      layout::PitchLinearShape<Shape::kRow * kInterleavedK, Shape::kColumn / kInterleavedK>, Element,
      layout::PitchLinear, (kAdvanceRank == 0 ? 0 : 1), ThreadMap>;

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorDesc operator()() {

    return UnderlyingMakeOperator()();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileAccessIterator for roww-major interleaved data.
template <
  typename Shape, typename Element, int AdvanceRank, 
  typename ThreadMap, int InterleavedK>
struct MakePredicatedTileAccessIteratorDesc <
    Shape, Element, layout::RowMajorInterleaved<InterleavedK>, AdvanceRank, ThreadMap> {

  static int const kAdvanceRank = AdvanceRank;
  static int const kInterleavedK = InterleavedK;

  using UnderlyingMakeOperator = MakePredicatedTileAccessIteratorDesc<
      layout::PitchLinearShape<Shape::kColumn * kInterleavedK, Shape::kRow / kInterleavedK>, Element,
      layout::PitchLinear, (kAdvanceRank == 0 ? 1 : 0), ThreadMap>;

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorDesc operator()() {

    return UnderlyingMakeOperator()();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Parameters struct
//

struct PredicatedTileAccessIteratorParams {

  using Index = int32_t;
  using LongIndex = int64_t;

  //
  // Data members
  //
  /// stride of pitch-linear layout (units of Element)
  LongIndex stride_ = 0;
  /// amount (in byte) to increment pointer to move to next access along
  /// strided dimension
  LongIndex inc_strided_ = 0;
  /// amount (in byte) to increment pointer from last access to first access
  /// of next tile
  LongIndex inc_next_ = 0;
  /// amount (in byte) to increment pointer from first access of current tile
  /// to first access of next tile
  LongIndex inc_advance_ = 0;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Status initialize(LongIndex stride, PredicatedTileAccessIteratorDesc desc) {
    CUTLASS_ASSERT(desc.element_size_bits > 0);
    CUTLASS_ASSERT(desc.advance_rank == 0 || desc.advance_rank == 1);

    stride_ = stride;

    inc_strided_ = (LongIndex(stride_) * desc.threadmap_delta.strided()) *
                     desc.element_size_bits / 8;

    if (desc.advance_rank) {
      // advance along strided dimension
      inc_advance_ =
          desc.threadblock_shape.strided() * LongIndex(stride_) * desc.element_size_bits / 8;
    } else {
      // advance along contiguous dimension
      inc_advance_ = desc.threadblock_shape.contiguous() * desc.element_size_bits / 8;
    }

    inc_next_ = inc_advance_ - LongIndex(desc.threadmap_iterations.strided() - 1) *
                                   desc.threadmap_delta.strided() * LongIndex(stride_) *
                                   desc.element_size_bits / 8;    

    return Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Status initialize(Index stride, PredicatedTileAccessIteratorDesc desc) {
    return initialize(LongIndex(stride), desc);
  }

  PredicatedTileAccessIteratorParams() = default;

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorParams(Index stride, PredicatedTileAccessIteratorDesc desc) {
    initialize(stride, desc);
  }

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorParams(LongIndex stride, PredicatedTileAccessIteratorDesc desc) {
    initialize(stride, desc);
  }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace transform
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
