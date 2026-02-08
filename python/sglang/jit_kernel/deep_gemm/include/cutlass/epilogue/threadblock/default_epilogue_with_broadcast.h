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

#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/threadblock/epilogue_with_broadcast.h"
#include "cutlass/epilogue/threadblock/epilogue_streamk_with_broadcast.h"

#include "cutlass/layout/permute.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {
////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for SimtOps.
template <
  typename Shape,
  typename WarpMmaSimt,
  typename ElementOutput,
  typename ElementTensor,
  typename ElementVector,
  typename OutputOp,
  int ElementsPerAccess,
  bool ScatterD = false,
  typename PermuteDLayout = layout::NoPermute,
  conv::StrideSupport StrideSupport = conv::StrideSupport::kUnity,
  int Rank = 4
>
struct DefaultEpilogueWithBroadcastSimt {

  static conv::StrideSupport const kStrideSupport = StrideSupport;
  static int const kRank = Rank;

  static bool const UseCUDAStore = platform::is_same<ElementOutput, double>::value;

  /// Use defaults related to the existing epilogue
  using Base = DefaultEpilogueSimt<
    Shape,
    WarpMmaSimt,
    OutputOp,
    ElementsPerAccess
  >;

  using PackedOutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    typename Base::OutputTileThreadMap,
    ElementOutput,
    ScatterD,
    PermuteDLayout,
    UseCUDAStore
  >;

  using StridedOutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIteratorConv<
    typename Base::OutputTileThreadMap,
    ElementOutput,
    ScatterD,
    PermuteDLayout,
    UseCUDAStore,
    kRank
  >;

  //
  // Stores the result z = (y = GEMM(A, B, C), broadcast)
  //
  using OutputTileIterator = typename platform::conditional<StrideSupport == cutlass::conv::StrideSupport::kUnity,
                                                            PackedOutputTileIterator,
                                                            StridedOutputTileIterator>::type;

  //
  // Additional tensor tile iterator - stores t = Elementwise(z)
  //
  using TensorTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    typename Base::OutputTileThreadMap,
    ElementTensor
  >;
  /// Define the epilogue
  using Epilogue = EpilogueWithBroadcast<
    Shape,
    WarpMmaSimt,
    Base::kPartitionsK,
    OutputTileIterator,
    TensorTileIterator,
    ElementVector,
    typename Base::AccumulatorFragmentIterator,
    typename Base::WarpTileIterator,
    typename Base::SharedLoadIterator,
    OutputOp,
    typename Base::Padding
  >;
};
////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for strided dgrad epilogues for SimtOps.
template <
  typename Shape,
  typename WarpMmaSimt,
  typename ElementOutput,
  typename ElementTensor,
  typename ElementVector,
  typename OutputOp,
  int ElementsPerAccess,
  bool ScatterD = false,
  typename PermuteDLayout = layout::NoPermute
>
struct DefaultEpilogueWithBroadcastSimtStridedDgrad {

  /// Use defaults related to the existing epilogue
  using Base = DefaultEpilogueSimtStridedDgrad<
    Shape,
    WarpMmaSimt,
    OutputOp,
    ElementsPerAccess
  >;

  //
  // Stores the result z = (y = GEMM(A, B, C), broadcast)
  //
  using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIteratorStridedDgrad<
    typename Base::OutputTileThreadMap,
    ElementOutput
  >;

  //
  // Additional tensor tile iterator - stores t = Elementwise(z)
  //
  using TensorTileIterator = cutlass::epilogue::threadblock::PredicatedTileIteratorStridedDgrad<
    typename Base::OutputTileThreadMap,
    ElementTensor
  >;

  /// Define the epilogue
  using Epilogue = EpilogueWithBroadcast<
    Shape,
    WarpMmaSimt,
    Base::kPartitionsK,
    OutputTileIterator,
    TensorTileIterator,
    ElementVector,
    typename Base::AccumulatorFragmentIterator,
    typename Base::WarpTileIterator,
    typename Base::SharedLoadIterator,
    OutputOp,
    typename Base::Padding
  >;
};
////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for TensorOps.
template <
  typename Shape,
  typename WarpMmaTensorOp,
  int PartitionsK,
  typename ElementOutput,
  typename ElementTensor,
  typename ElementVector,
  typename OutputOp,
  int ElementsPerAccess,
  bool ScatterD = false,
  typename PermuteDLayout = layout::NoPermute
>
struct DefaultEpilogueWithBroadcastTensorOp {

  /// Use defaults related to the existing epilogue
  using Base = DefaultEpilogueTensorOp<
    Shape,
    WarpMmaTensorOp,
    PartitionsK,
    OutputOp,
    ElementsPerAccess
  >;

  //
  // Stores the result z = (y = GEMM(A, B, C), broadcast)
  //
  using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    typename Base::OutputTileThreadMap,
    ElementOutput,
    ScatterD,
    PermuteDLayout
  >;

  //
  // Additional tensor tile iterator - stores t = Elementwise(z)
  //
  using TensorTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    typename Base::OutputTileThreadMap,
    ElementTensor
  >;

  /// Define the epilogue
  using Epilogue = EpilogueWithBroadcast<
    Shape,
    WarpMmaTensorOp,
    PartitionsK,
    OutputTileIterator,
    TensorTileIterator,
    ElementVector,
    typename Base::AccumulatorFragmentIterator,
    typename Base::WarpTileIterator,
    typename Base::SharedLoadIterator,
    OutputOp,
    typename Base::Padding,
    Base::kFragmentsPerIteration
  >;
};

////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for streamk epilogues for TensorOps.
template <
  typename Shape,
  typename WarpMmaTensorOp,
  int PartitionsK,
  typename ElementOutput,
  typename ElementTensor,
  typename ElementVector,
  typename OutputOp,
  int ElementsPerAccess,
  bool ScatterD = false,
  typename PermuteDLayout = layout::NoPermute
>
struct DefaultStreamkEpilogueWithBroadcastTensorOp {

  /// Use defaults related to the existing epilogue
  using Base = DefaultEpilogueTensorOp<
    Shape,
    WarpMmaTensorOp,
    PartitionsK,
    OutputOp,
    ElementsPerAccess
  >;

  //
  // Stores the result z = (y = GEMM(A, B, C), broadcast)
  //
  using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    typename Base::OutputTileThreadMap,
    ElementOutput,
    ScatterD,
    PermuteDLayout
  >;

  //
  // Additional tensor tile iterator - stores t = Elementwise(z)
  //
  using TensorTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    typename Base::OutputTileThreadMap,
    ElementTensor
  >;

  /// Define the epilogue
  using Epilogue = EpilogueStreamkWithBroadcast<
    Shape,
    WarpMmaTensorOp,
    PartitionsK,
    OutputTileIterator,
    TensorTileIterator,
    ElementVector,
    typename Base::AccumulatorFragmentIterator,
    typename Base::WarpTileIterator,
    typename Base::SharedLoadIterator,
    OutputOp,
    typename Base::Padding,
    Base::kFragmentsPerIteration
  >;
};

////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for VoltaTensorOps.
template <
  typename Shape,
  typename WarpMmaTensorOp,
  int PartitionsK,
  typename ElementOutput,
  typename ElementTensor,
  typename ElementVector,
  typename OutputOp,
  int ElementsPerAccess
>
struct DefaultEpilogueWithBroadcastVoltaTensorOp {

  /// Use defaults related to the existing epilogue
  using Base = DefaultEpilogueVoltaTensorOp<
    Shape,
    WarpMmaTensorOp,
    PartitionsK,
    OutputOp,
    ElementsPerAccess
  >;

  //
  // Stores the result z = (y = GEMM(A, B, C), broadcast)
  //
  using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    typename Base::OutputTileThreadMap,
    ElementOutput
  >;

  //
  // Additional tensor tile iterator - stores t = Elementwise(z)
  //
  using TensorTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    typename Base::OutputTileThreadMap,
    ElementTensor
  >;

  /// Define the epilogue
  using Epilogue = EpilogueWithBroadcast<
    Shape,
    WarpMmaTensorOp,
    PartitionsK,
    OutputTileIterator,
    TensorTileIterator,
    ElementVector,
    typename Base::AccumulatorFragmentIterator,
    typename Base::WarpTileIterator,
    typename Base::SharedLoadIterator,
    OutputOp,
    typename Base::Padding
  >;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
