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
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops on Volta.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"
#include "cutlass/epilogue/thread/linear_combination_sigmoid.h"
#include "cutlass/epilogue/thread/linear_combination_planar_complex.h"

#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/epilogue/thread/reduction_op.h"

#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_strided_dgrad.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_affine.h"
#include "cutlass/epilogue/threadblock/shared_load_iterator.h"

#include "cutlass/epilogue/warp/fragment_iterator_volta_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_thread_map_volta_tensor_op.h"

#include "cutlass/epilogue/threadblock/epilogue.h"

#include "cutlass/layout/permute.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for TensorOps.
template <
  typename Shape_,
  typename WarpMmaTensorOp_,
  int PartitionsK,
  typename OutputOp_,
  int ElementsPerAccess,
  bool ScatterD = false,
  typename PermuteDLayout = layout::NoPermute
>
struct DefaultEpilogueVoltaTensorOp {

  using Shape = Shape_;
  using WarpMmaTensorOp = WarpMmaTensorOp_;
  static int const kPartitionsK = PartitionsK;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaTensorOp::LayoutC;
  using ElementAccumulator = typename WarpMmaTensorOp::ElementC;

  //
  // Thread map
  //

  using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape,
    typename WarpMmaTensorOp::Shape,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator
  >::Type;

  using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    OutputTileThreadMap,
    ElementOutput,
    ScatterD,
    PermuteDLayout
  >;

  using AccumulatorFragmentIterator = cutlass::epilogue::warp::FragmentIteratorVoltaTensorOp<
    typename WarpMmaTensorOp::Shape,
    gemm::GemmShape<32, 32, 4>,
    ElementAccumulator,
    LayoutC
  >;

  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorVoltaTensorOp<
    typename WarpMmaTensorOp::Shape,
    gemm::GemmShape<32, 32, 4>,
    ElementAccumulator,
    LayoutC
  >;

  static int const kSharedMemAlignment = sizeof_bits<ElementAccumulator>::value * WarpTileIterator::kElementsPerAccess / 8;

  static_assert(kSharedMemAlignment == 8, "Shared memory alignment must be 8B");

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIterator<
    typename OutputTileThreadMap::CompactedThreadMap,
    ElementAccumulator,
    kSharedMemAlignment
  >;

  /// Hard-coded padding elements added 
  using Padding = typename WarpTileIterator::Padding;

  //
  // Define the epilogue
  //
  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    OutputTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    SharedLoadIterator,
    OutputOp,
    Padding
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for TensorOps.
template <
  typename Shape_,
  typename WarpMmaTensorOp_,
  int PartitionsK,
  typename OutputOp_,
  int ElementsPerAccess
>
struct DefaultEpilogueVoltaTensorOpStridedDgrad {

  using Shape = Shape_;
  using WarpMmaTensorOp = WarpMmaTensorOp_;
  static int const kPartitionsK = PartitionsK;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaTensorOp::LayoutC;
  using ElementAccumulator = typename WarpMmaTensorOp::ElementC;

  //
  // Thread map
  //

  using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape,
    typename WarpMmaTensorOp::Shape,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator
  >::Type;

  using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIteratorStridedDgrad<
    OutputTileThreadMap,
    ElementOutput
  >;

  using AccumulatorFragmentIterator = cutlass::epilogue::warp::FragmentIteratorVoltaTensorOp<
    typename WarpMmaTensorOp::Shape,
    gemm::GemmShape<32, 32, 4>,
    ElementAccumulator,
    LayoutC
  >;

  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorVoltaTensorOp<
    typename WarpMmaTensorOp::Shape,
    gemm::GemmShape<32, 32, 4>,
    ElementAccumulator,
    LayoutC
  >;

  static int const kSharedMemAlignment = sizeof_bits<ElementAccumulator>::value * WarpTileIterator::kElementsPerAccess / 8;

  static_assert(kSharedMemAlignment == 8, "Shared memory alignment must be 8B");

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIterator<
    typename OutputTileThreadMap::CompactedThreadMap,
    ElementAccumulator,
    kSharedMemAlignment
  >;

  /// Hard-coded padding elements added 
  using Padding = typename WarpTileIterator::Padding;

  //
  // Define the epilogue
  //
  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    OutputTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    SharedLoadIterator,
    OutputOp,
    Padding
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for TensorOps.
template <
  int Rank,
  typename Shape_,
  typename WarpMmaTensorOp_,
  int PartitionsK,
  typename OutputOp_,
  int ElementsPerAccess
>
struct DefaultEpilogueVoltaTensorOpAffineRankN {

  using Shape = Shape_;
  using WarpMmaTensorOp = WarpMmaTensorOp_;
  static int const kPartitionsK = PartitionsK;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaTensorOp::LayoutC;
  using ElementAccumulator = typename WarpMmaTensorOp::ElementC;

  //
  // Thread map
  //

  using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapVoltaTensorOp<
    Shape,
    typename WarpMmaTensorOp::Shape,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess,
    ElementAccumulator
  >::Type;

  using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIteratorAffineRankN<
    OutputTileThreadMap,
    ElementOutput,
    Rank
  >;

  using AccumulatorFragmentIterator = cutlass::epilogue::warp::FragmentIteratorVoltaTensorOp<
    typename WarpMmaTensorOp::Shape,
    gemm::GemmShape<32, 32, 4>,
    ElementAccumulator,
    LayoutC
  >;

  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorVoltaTensorOp<
    typename WarpMmaTensorOp::Shape,
    gemm::GemmShape<32, 32, 4>,
    ElementAccumulator,
    LayoutC
  >;

  static int const kSharedMemAlignment = sizeof_bits<ElementAccumulator>::value * WarpTileIterator::kElementsPerAccess / 8;

  static_assert(kSharedMemAlignment == 8, "Shared memory alignment must be 8B");

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIterator<
    typename OutputTileThreadMap::CompactedThreadMap,
    ElementAccumulator,
    kSharedMemAlignment
  >;

  /// Hard-coded padding elements added 
  using Padding = typename WarpTileIterator::Padding;

  //
  // Define the epilogue
  //
  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    OutputTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    SharedLoadIterator,
    OutputOp,
    Padding
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
