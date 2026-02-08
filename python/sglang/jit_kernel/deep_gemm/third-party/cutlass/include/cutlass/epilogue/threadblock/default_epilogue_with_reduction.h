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
#include "cutlass/epilogue/threadblock/epilogue_with_reduction.h"

#include "cutlass/layout/permute.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for TensorOps.
template <
  typename Shape,
  typename WarpMmaTensorOp,
  int PartitionsK,
  typename ElementOutput,
  typename OutputOp,
  typename ReductionOp,
  int ElementsPerAccess,
  bool ScatterD = false,
  typename PermuteDLayout = layout::NoPermute
>
struct DefaultEpilogueWithReductionTensorOp {

  /// Use defaults related to the existing epilogue
  using Base = DefaultEpilogueTensorOp<
    Shape,
    WarpMmaTensorOp,
    PartitionsK,
    OutputOp,
    ElementsPerAccess
  >;

  /// Additional tensor tile iterator
  using TensorTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    typename Base::OutputTileThreadMap,
    typename OutputOp::ElementTensor
  >;

  using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    typename Base::OutputTileThreadMap,
    ElementOutput,
    ScatterD,
    PermuteDLayout
  >;

  /// Define the epilogue
  using Epilogue = EpilogueWithReduction<
    Shape,
    WarpMmaTensorOp,
    PartitionsK,
    OutputTileIterator,
    TensorTileIterator,
    typename WarpMmaTensorOp::ElementC,
    typename Base::AccumulatorFragmentIterator,
    typename Base::WarpTileIterator,
    typename Base::SharedLoadIterator,
    typename Base::OutputOp,
    ReductionOp,
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
  typename OutputOp,
  typename ReductionOp,
  int ElementsPerAccess,
  bool ScatterD = false,
  typename PermuteDLayout = layout::NoPermute
>
struct DefaultEpilogueWithReductionVoltaTensorOp {

  /// Use defaults related to the existing epilogue
  using Base = DefaultEpilogueVoltaTensorOp<
    Shape,
    WarpMmaTensorOp,
    PartitionsK,
    OutputOp,
    ElementsPerAccess
  >;

  /// Additional tensor tile iterator
  using TensorTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    typename Base::OutputTileThreadMap,
    typename OutputOp::ElementTensor
  >;

  using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    typename Base::OutputTileThreadMap,
    ElementOutput,
    ScatterD,
    PermuteDLayout
  >;

  /// Define the epilogue
  using Epilogue = EpilogueWithReduction<
    Shape,
    WarpMmaTensorOp,
    PartitionsK,
    OutputTileIterator,
    TensorTileIterator,
    typename WarpMmaTensorOp::ElementC,
    typename Base::AccumulatorFragmentIterator,
    typename Base::WarpTileIterator,
    typename Base::SharedLoadIterator,
    typename Base::OutputOp,
    ReductionOp,
    typename Base::Padding
  >;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
