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

#include "cutlass/platform/platform.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_relu0.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"
#include "cutlass/epilogue/thread/linear_combination_sigmoid.h"
#include "cutlass/epilogue/thread/linear_combination_hardswish.h"
#include "cutlass/epilogue/thread/linear_combination_planar_complex.h"

#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/epilogue/thread/reduction_op.h"

#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"

#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/fragment_iterator_complex_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op_mixed.h"
#include "cutlass/epilogue/threadblock/default_thread_map_tensor_op.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_conv.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_strided_dgrad.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_affine.h"
#include "cutlass/epilogue/threadblock/shared_load_iterator.h"
#include "cutlass/epilogue/threadblock/shared_load_iterator_mixed.h"

#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/threadblock/interleaved_epilogue.h"

#include "cutlass/layout/permute.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <
  typename ElementOutput,
  typename ElementAccumulator,
  int ElementsPerAccess,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename ThreadMap
>
struct DefaultIteratorsTensorOp {
  
  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOp<
    WarpShape,
    InstructionShape,
    ElementAccumulator,
    layout::RowMajor
  >;

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIterator<
    ThreadMap,
    ElementAccumulator
  >;

  static int const kFragmentsPerIteration = 1;
};

/// Partial specialization for float <= float x 4
template <
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename ThreadMap
>
struct DefaultIteratorsTensorOp<float, float, 4, ThreadblockShape, WarpShape, InstructionShape, ThreadMap> {
  
  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOp<
    WarpShape,
    InstructionShape,
    float,
    layout::RowMajor
  >;

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIterator<
    ThreadMap,
    float
  >;

  static int const kFragmentsPerIteration = 2;
};

/// Partial specialization for int32_t <= int32_t
template <
  int ElementsPerAccess,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename ThreadMap
>
struct DefaultIteratorsTensorOp<int32_t, int32_t, ElementsPerAccess, ThreadblockShape, WarpShape, InstructionShape, ThreadMap> {
  
  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOp<
    WarpShape,
    InstructionShape,
    int32_t,
    layout::RowMajor
  >;

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIterator<
    ThreadMap,
    int32_t
  >;

  static int const kFragmentsPerIteration = 1;
};

/// Partial specialization for float <= int32_t
template <
  int ElementsPerAccess,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename ThreadMap
>
struct DefaultIteratorsTensorOp<float, int32_t, ElementsPerAccess, ThreadblockShape, WarpShape, InstructionShape, ThreadMap> {

  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOp<
    WarpShape,
    InstructionShape,
    int32_t,
    layout::RowMajor
  >;

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIterator<
    ThreadMap,
    int32_t
  >;

  static int const kFragmentsPerIteration = 1;
};

/// Partial specialization for half <= float x 8 epilogues avoids shared memory bank conflicts.
template <
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename ThreadMap
>
struct DefaultIteratorsTensorOp<
  half_t, 
  float, 
  8, 
  ThreadblockShape, 
  WarpShape, 
  InstructionShape, 
  ThreadMap> {
  
  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOpMixed<
    WarpShape,
    InstructionShape,
    float,
    32,
    16,
    8,
    8
  >;

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIteratorMixed<
    ThreadMap,
    float,
    32,
    16,
    8,
    8
  >;

  static int const kFragmentsPerIteration = 2;
};

/// Partial specialization for half <= int32_t x 8 epilogues avoids shared memory bank conflicts.
template <
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename ThreadMap
>
struct DefaultIteratorsTensorOp<
  bfloat16_t,
  int32_t,
  8,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  ThreadMap> {

  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOpMixed<
    WarpShape,
    InstructionShape,
    int32_t,
    32,
    16,
    8,
    8
  >;

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIteratorMixed<
    ThreadMap,
    int32_t,
    32,
    16,
    8,
    8
  >;

  static int const kFragmentsPerIteration = 2;
};

/// Partial specialization for half <= int32_t x 8 epilogues avoids shared memory bank conflicts.
template <
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename ThreadMap
>
struct DefaultIteratorsTensorOp<
  half_t, 
  int32_t, 
  8, 
  ThreadblockShape, 
  WarpShape, 
  InstructionShape, 
  ThreadMap> {
  
  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOpMixed<
    WarpShape,
    InstructionShape,
    int32_t,
    32,
    16,
    8,
    8
  >;

  using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIteratorMixed<
    ThreadMap,
    int32_t,
    32,
    16,
    8,
    8
  >;

  static int const kFragmentsPerIteration = 2;
};

/// Partial specialization for int8/int4b_t <= int32 x 16/8 epilogues avoids shared memory bank conflicts.
/// Threadblock::kN = 256 still has bank conflicts.
template <
  typename ElementOutput,
  int ElementsPerAccess,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename ThreadMap
>
struct DefaultIteratorsTensorOp<
  ElementOutput, 
  int32_t, 
  ElementsPerAccess,
  ThreadblockShape, 
  WarpShape, 
  InstructionShape, 
  ThreadMap> {

  static_assert(platform::is_same<ElementOutput, cutlass::int4b_t>::value ||
                platform::is_same<ElementOutput, cutlass::uint4b_t>::value ||
                platform::is_same<ElementOutput, int8_t>::value ||
                platform::is_same<ElementOutput, uint8_t>::value,
                "ElementOutput needs to be 4 or 8 bit (unsigned) int.");

   static_assert((ElementsPerAccess == 16 || ElementsPerAccess == 8 || ElementsPerAccess == 4),
                "ElementsPerAccess needs to be 16 or 8.");
  
  using WarpTileIteratorMixed = cutlass::epilogue::warp::TileIteratorTensorOpMixed<
    WarpShape,
    InstructionShape,
    int32_t,
    32,
    cutlass::sizeof_bits<ElementOutput>::value,
    ElementsPerAccess,
    8
  >;

  using WarpTileIteratorNotMixed =  cutlass::epilogue::warp::TileIteratorTensorOp<
    WarpShape,
    InstructionShape,
    int32_t,
    layout::RowMajor
  >;

  using WarpTileIterator = typename platform::conditional<
                             (ThreadblockShape::kN == 256) || (ThreadblockShape::kN == 128 && ElementsPerAccess == 8) || (ElementsPerAccess == 4),
                             WarpTileIteratorNotMixed,
                             WarpTileIteratorMixed>::type;

  using SharedLoadIteratorMixed = cutlass::epilogue::threadblock::SharedLoadIteratorMixed<
    ThreadMap,
    int32_t,
    32,
    cutlass::sizeof_bits<ElementOutput>::value,
    ElementsPerAccess,
    8
  >;

  using SharedLoadIteratorNotMixed = cutlass::epilogue::threadblock::SharedLoadIterator<
    ThreadMap,
    int32_t
  >;

  using SharedLoadIterator = typename platform::conditional<
                             (ThreadblockShape::kN == 256) || (ThreadblockShape::kN == 128 && ElementsPerAccess == 8) || (ElementsPerAccess == 4),
                             SharedLoadIteratorNotMixed,
                             SharedLoadIteratorMixed>::type;

  static int const kFragmentsPerIteration = 1;
};

/// Partial specialization for float_e4m3_t <= float x 16/8 epilogues avoids shared memory bank conflicts.
/// Threadblock::kN = 256 still has bank conflicts.
template <
  int ElementsPerAccess,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename ThreadMap
>
struct DefaultIteratorsTensorOp<
  cutlass::float_e4m3_t,
  float, 
  ElementsPerAccess,
  ThreadblockShape, 
  WarpShape, 
  InstructionShape, 
  ThreadMap> {

  using ElementOutput = cutlass::float_e4m3_t;

  static_assert((ElementsPerAccess == 16 || ElementsPerAccess == 8 || ElementsPerAccess == 4),
              "ElementsPerAccess needs to be 16 or 8.");
  
  using WarpTileIteratorMixed = cutlass::epilogue::warp::TileIteratorTensorOpMixed<
    WarpShape,
    InstructionShape,
    float,
    32,
    cutlass::sizeof_bits<ElementOutput>::value,
    ElementsPerAccess,
    8
  >;

  using WarpTileIteratorNotMixed =  cutlass::epilogue::warp::TileIteratorTensorOp<
    WarpShape,
    InstructionShape,
    float,
    layout::RowMajor
  >;

  using WarpTileIterator = typename platform::conditional<
                             (ThreadblockShape::kN == 256) || (ThreadblockShape::kN == 128 && ElementsPerAccess == 8) || (ElementsPerAccess == 4),
                             WarpTileIteratorNotMixed,
                             WarpTileIteratorMixed>::type;

  using SharedLoadIteratorMixed = cutlass::epilogue::threadblock::SharedLoadIteratorMixed<
    ThreadMap,
    float,
    32,
    cutlass::sizeof_bits<ElementOutput>::value,
    ElementsPerAccess,
    8
  >;

  using SharedLoadIteratorNotMixed = cutlass::epilogue::threadblock::SharedLoadIterator<
    ThreadMap,
    float
  >;

  using SharedLoadIterator = typename platform::conditional<
                             (ThreadblockShape::kN == 256) || (ThreadblockShape::kN == 128 && ElementsPerAccess == 8) || (ElementsPerAccess == 4),
                             SharedLoadIteratorNotMixed,
                             SharedLoadIteratorMixed>::type;

  static int const kFragmentsPerIteration = 1;
};

/// Partial specialization for float_e5m2_t <= float x 16/8 epilogues avoids shared memory bank conflicts.
/// Threadblock::kN = 256 still has bank conflicts.
template <
  int ElementsPerAccess,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename ThreadMap
>
struct DefaultIteratorsTensorOp<
  cutlass::float_e5m2_t,
  float, 
  ElementsPerAccess,
  ThreadblockShape, 
  WarpShape, 
  InstructionShape, 
  ThreadMap> {

  using ElementOutput = cutlass::float_e5m2_t;

  static_assert((ElementsPerAccess == 16 || ElementsPerAccess == 8 || ElementsPerAccess == 4),
              "ElementsPerAccess needs to be 16 or 8.");
  
  using WarpTileIteratorMixed = cutlass::epilogue::warp::TileIteratorTensorOpMixed<
    WarpShape,
    InstructionShape,
    float,
    32,
    cutlass::sizeof_bits<ElementOutput>::value,
    ElementsPerAccess,
    8
  >;

  using WarpTileIteratorNotMixed =  cutlass::epilogue::warp::TileIteratorTensorOp<
    WarpShape,
    InstructionShape,
    float,
    layout::RowMajor
  >;

  using WarpTileIterator = typename platform::conditional<
                             (ThreadblockShape::kN == 256) || (ThreadblockShape::kN == 128 && ElementsPerAccess == 8) || (ElementsPerAccess == 4),
                             WarpTileIteratorNotMixed,
                             WarpTileIteratorMixed>::type;

  using SharedLoadIteratorMixed = cutlass::epilogue::threadblock::SharedLoadIteratorMixed<
    ThreadMap,
    float,
    32,
    cutlass::sizeof_bits<ElementOutput>::value,
    ElementsPerAccess,
    8
  >;

  using SharedLoadIteratorNotMixed = cutlass::epilogue::threadblock::SharedLoadIterator<
    ThreadMap,
    float
  >;

  using SharedLoadIterator = typename platform::conditional<
                             (ThreadblockShape::kN == 256) || (ThreadblockShape::kN == 128 && ElementsPerAccess == 8) || (ElementsPerAccess == 4),
                             SharedLoadIteratorNotMixed,
                             SharedLoadIteratorMixed>::type;

  static int const kFragmentsPerIteration = 1;
};

} // namespace detail

////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for TensorOps.
template <
  typename Shape_,
  typename WarpMmaTensorOp_,
  int PartitionsK,
  typename OutputOp_,
  int ElementsPerAccess,
  bool ScatterD = false,
  typename PermuteDLayout = layout::NoPermute,
  conv::StrideSupport StrideSupport = conv::StrideSupport::kUnity,
  int Rank = 4
>
struct DefaultEpilogueTensorOp {

  using Shape = Shape_;
  using WarpMmaTensorOp = WarpMmaTensorOp_;
  static int const kPartitionsK = PartitionsK;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaTensorOp::LayoutC;
  using ElementAccumulator = typename WarpMmaTensorOp::ElementC;
  static conv::StrideSupport const kStrideSupport = StrideSupport;
  static int const kRank = Rank;

  //
  // Thread map
  //

  using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
    Shape,
    typename WarpMmaTensorOp::Shape,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess
  >::Type;

  static bool const UseCUDAStore = platform::is_same<ElementOutput, double>::value;

  using PackedOutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    OutputTileThreadMap,
    ElementOutput,
    ScatterD,
    PermuteDLayout,
    UseCUDAStore
  >;

  using StridedOutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIteratorConv<
    OutputTileThreadMap,
    ElementOutput,
    ScatterD,
    PermuteDLayout,
    UseCUDAStore,
    kRank
  >;

  using OutputTileIterator = typename platform::conditional<StrideSupport == cutlass::conv::StrideSupport::kUnity,
                                                            PackedOutputTileIterator,
                                                            StridedOutputTileIterator>::type;

  using AccumulatorFragmentIterator = typename platform::conditional<is_complex<ElementOutput>::value,
                                    cutlass::epilogue::warp::FragmentIteratorComplexTensorOp<
                                        typename WarpMmaTensorOp::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::ElementC,
                                        typename WarpMmaTensorOp::Policy::Operator::FragmentC,
                                        LayoutC>,
                                    cutlass::epilogue::warp::FragmentIteratorTensorOp<
                                        typename WarpMmaTensorOp::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::ElementC,
                                        typename WarpMmaTensorOp::Policy::Operator::FragmentC,
                                        LayoutC> >::type;

  /// Support several implementations depending on structure of epilogue
  using DefaultIterators = detail::DefaultIteratorsTensorOp<
    ElementOutput,
    ElementAccumulator,
    kElementsPerAccess,
    Shape,
    typename WarpMmaTensorOp::Shape,
    typename WarpMmaTensorOp::Policy::Operator::Shape,
    typename OutputTileThreadMap::CompactedThreadMap
  >;

  using WarpTileIterator = typename DefaultIterators::WarpTileIterator;
  using SharedLoadIterator = typename DefaultIterators::SharedLoadIterator;

  /// Hard-coded padding elements added 
  using Padding = cutlass::MatrixShape<0, 64 / sizeof_bits<ElementAccumulator>::value * 4>;

  static int const kFragmentsPerIteration = (kPartitionsK == 1 ? DefaultIterators::kFragmentsPerIteration : 1);

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
    Padding,
    kFragmentsPerIteration
  >;
};

////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for TensorOps.
template <
  typename Shape_,
  typename WarpMmaTensorOp_,
  int PartitionsK,
  typename OutputOp_,
  int ElementsPerAccess
>
struct DefaultEpilogueTensorOpStridedDgrad {

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

  using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
    Shape,
    typename WarpMmaTensorOp::Shape,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess
  >::Type;

  using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIteratorStridedDgrad<
    OutputTileThreadMap,
    ElementOutput
  >;

  using AccumulatorFragmentIterator = typename platform::conditional<is_complex<ElementOutput>::value,
                                    cutlass::epilogue::warp::FragmentIteratorComplexTensorOp<
                                        typename WarpMmaTensorOp::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::ElementC,
                                        typename WarpMmaTensorOp::Policy::Operator::FragmentC,
                                        LayoutC>,
                                    cutlass::epilogue::warp::FragmentIteratorTensorOp<
                                        typename WarpMmaTensorOp::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::ElementC,
                                        typename WarpMmaTensorOp::Policy::Operator::FragmentC,
                                        LayoutC> >::type;

  /// Support several implementations depending on structure of epilogue
  using DefaultIterators = detail::DefaultIteratorsTensorOp<
    ElementOutput,
    ElementAccumulator,
    kElementsPerAccess,
    Shape,
    typename WarpMmaTensorOp::Shape,
    typename WarpMmaTensorOp::Policy::Operator::Shape,
    typename OutputTileThreadMap::CompactedThreadMap
  >;

  using WarpTileIterator = typename DefaultIterators::WarpTileIterator;
  using SharedLoadIterator = typename DefaultIterators::SharedLoadIterator;

  /// Hard-coded padding elements added 
  using Padding = cutlass::MatrixShape<0, 64 / sizeof_bits<ElementAccumulator>::value * 4>;

  static int const kFragmentsPerIteration = (kPartitionsK == 1 ? DefaultIterators::kFragmentsPerIteration : 1);

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
    Padding,
    kFragmentsPerIteration
  >;
};


////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for TensorOps.
template <
  int Rank,
  typename Shape_,
  typename WarpMmaTensorOp_,
  int PartitionsK,
  typename OutputOp_,
  int ElementsPerAccess
>
struct DefaultEpilogueTensorOpAffineRankN {

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

  using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
    Shape,
    typename WarpMmaTensorOp::Shape,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess
  >::Type;

  using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIteratorAffineRankN<
    OutputTileThreadMap,
    ElementOutput,
    Rank
  >;

  // Map to the row major iterator since the iterator selection for affineN is the same.
  using AccumulatorFragmentIterator = typename platform::conditional<is_complex<ElementOutput>::value,
                                    cutlass::epilogue::warp::FragmentIteratorComplexTensorOp<
                                        typename WarpMmaTensorOp::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::ElementC,
                                        typename WarpMmaTensorOp::Policy::Operator::FragmentC,
                                        layout::RowMajor>,
                                    cutlass::epilogue::warp::FragmentIteratorTensorOp<
                                        typename WarpMmaTensorOp::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::ElementC,
                                        typename WarpMmaTensorOp::Policy::Operator::FragmentC,
                                        layout::RowMajor> >::type;

  /// Support several implementations depending on structure of epilogue
  using DefaultIterators = detail::DefaultIteratorsTensorOp<
    ElementOutput,
    ElementAccumulator,
    kElementsPerAccess,
    Shape,
    typename WarpMmaTensorOp::Shape,
    typename WarpMmaTensorOp::Policy::Operator::Shape,
    typename OutputTileThreadMap::CompactedThreadMap
  >;

  using WarpTileIterator = typename DefaultIterators::WarpTileIterator;
  using SharedLoadIterator = typename DefaultIterators::SharedLoadIterator;

  /// Hard-coded padding elements added 
  using Padding = cutlass::MatrixShape<0, 64 / sizeof_bits<ElementAccumulator>::value * 4>;

  static int const kFragmentsPerIteration = (kPartitionsK == 1 ? DefaultIterators::kFragmentsPerIteration : 1);

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
    Padding,
    kFragmentsPerIteration
  >;
};

////////////////////////////////////////////////////////////////////////////////
/// Defines sensible defaults for epilogues for TensorOps which uses
/// intereleaved output layout. For this case, shared memory is not needed.
template <typename Shape_, typename WarpMmaTensorOp_, int PartitionsK,
          typename OutputOp_, int ElementsPerAccess, int InterleavedK,
          bool isSplitK = false>
struct DefaultInterleavedEpilogueTensorOp {
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
  using OutputTileThreadMap = typename cutlass::epilogue::threadblock::
      DefaultInterleavedThreadMapTensorOp<
          Shape, typename WarpMmaTensorOp::Shape, kPartitionsK, ElementOutput,
          kElementsPerAccess, InterleavedK>::Type;

  using OutputTileIterator =
      cutlass::epilogue::threadblock::InterleavedPredicatedTileIterator<
          OutputTileThreadMap, ElementOutput, InterleavedK>;

  using AccumulatorFragmentIterator =
      cutlass::epilogue::warp::FragmentIteratorTensorOp<
          typename WarpMmaTensorOp::Shape,
          typename WarpMmaTensorOp::Policy::Operator::Shape,
          typename WarpMmaTensorOp::Policy::Operator::ElementC,
          typename WarpMmaTensorOp::Policy::Operator::FragmentC,
          LayoutC>;

  //
  // Define the epilogue
  //
  using Epilogue = cutlass::epilogue::threadblock::InterleavedEpilogue<
      Shape, WarpMmaTensorOp, kPartitionsK, OutputTileIterator,
      AccumulatorFragmentIterator, OutputOp, InterleavedK>;
};

////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for TensorOps which uses
/// intereleaved output layout. For this case, shared memory is not needed.
template <typename Shape_, typename WarpMmaTensorOp_, int PartitionsK,
          typename OutputOp_, int ElementsPerAccess, int InterleavedK,
          bool isSplitK = false>
struct DefaultInterleavedConvEpilogue {
  using Shape = Shape_;
  using WarpMmaTensorOp = WarpMmaTensorOp_;
  static int const kPartitionsK = PartitionsK;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;

  using ElementOutput = typename OutputOp::ElementOutput;
  using ElementAccumulator = typename WarpMmaTensorOp::ElementC;

  //
  // Thread map
  //
  using OutputTileThreadMap = typename cutlass::epilogue::threadblock::
      DefaultInterleavedConvThreadMapTensorOp<
          Shape, typename WarpMmaTensorOp::Shape, kPartitionsK, ElementOutput,
          kElementsPerAccess, InterleavedK>::Type;

  using OutputTileIterator =
      cutlass::epilogue::threadblock::InterleavedConvPredicatedTileIterator<
          OutputTileThreadMap, ElementOutput, InterleavedK>;

  using AccumulatorFragmentIterator =
      cutlass::epilogue::warp::FragmentIteratorTensorOp<
          typename WarpMmaTensorOp::Shape,
          typename WarpMmaTensorOp::Policy::Operator::Shape,
          typename WarpMmaTensorOp::Policy::Operator::ElementC,
          typename WarpMmaTensorOp::Policy::Operator::FragmentC,
          // can reuse the gemm version here to do element selection
          layout::ColumnMajorInterleaved<InterleavedK>>;

  //
  // Define the epilogue
  //
  using Epilogue = cutlass::epilogue::threadblock::InterleavedEpilogue<
      Shape, WarpMmaTensorOp, kPartitionsK, OutputTileIterator,
      AccumulatorFragmentIterator, OutputOp, InterleavedK>;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
