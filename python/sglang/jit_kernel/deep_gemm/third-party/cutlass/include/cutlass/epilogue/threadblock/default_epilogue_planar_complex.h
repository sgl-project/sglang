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
  \brief Constructs a default epilogue for planar complex outputs.

  This template reuses components for real-valued epilogues and applies them to planar complex
  output matrices.

*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/array_planar_complex.h"

#include "cutlass/arch/arch.h"

#include "cutlass/epilogue/thread/linear_combination_planar_complex.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"

#include "cutlass/epilogue/threadblock/epilogue_planar_complex.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues.
template <
  typename ThreadblockShape_,
  typename WarpMma_,
  typename OpcodeClass_,
  typename ArchTag_,
  int PartitionsK,
  typename OutputOp_,
  int ElementsPerAccess
>
struct DefaultEpiloguePlanarComplex;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues.
template <
  typename ThreadblockShape_,
  typename WarpMmaOperator_,
  int PartitionsK,
  typename OutputOp_,
  int ElementsPerAccess
>
struct DefaultEpiloguePlanarComplex<
  ThreadblockShape_, 
  WarpMmaOperator_, 
  arch::OpClassTensorOp, 
  arch::Sm70,
  PartitionsK, 
  OutputOp_, 
  ElementsPerAccess> {

  using RealEpilogue = DefaultEpilogueVoltaTensorOp<
    ThreadblockShape_,
    WarpMmaOperator_,
    PartitionsK,
    OutputOp_,
    ElementsPerAccess
  >;

  using Epilogue = EpiloguePlanarComplex<
    ThreadblockShape_,
    WarpMmaOperator_,
    PartitionsK,
    typename RealEpilogue::OutputTileIterator,
    typename RealEpilogue::AccumulatorFragmentIterator,
    typename RealEpilogue::WarpTileIterator,
    typename RealEpilogue::SharedLoadIterator,
    OutputOp_,
    typename RealEpilogue::Padding
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues.
template <
  typename ThreadblockShape_,
  typename WarpMmaOperator_,
  int PartitionsK,
  typename OutputOp_,
  int ElementsPerAccess
>
struct DefaultEpiloguePlanarComplex<
  ThreadblockShape_, 
  WarpMmaOperator_, 
  arch::OpClassTensorOp, 
  arch::Sm75,
  PartitionsK, 
  OutputOp_, 
  ElementsPerAccess> {

  using RealEpilogue = DefaultEpilogueTensorOp<
    ThreadblockShape_,
    WarpMmaOperator_,
    PartitionsK,
    OutputOp_,
    ElementsPerAccess
  >;

  using Epilogue = EpiloguePlanarComplex<
    ThreadblockShape_,
    WarpMmaOperator_,
    PartitionsK,
    typename RealEpilogue::OutputTileIterator,
    typename RealEpilogue::AccumulatorFragmentIterator,
    typename RealEpilogue::WarpTileIterator,
    typename RealEpilogue::SharedLoadIterator,
    OutputOp_,
    typename RealEpilogue::Padding
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues.
template <
  typename ThreadblockShape_,
  typename WarpMmaOperator_,
  int PartitionsK,
  typename OutputOp_,
  int ElementsPerAccess
>
struct DefaultEpiloguePlanarComplex<
  ThreadblockShape_, 
  WarpMmaOperator_, 
  arch::OpClassTensorOp, 
  arch::Sm80,
  PartitionsK, 
  OutputOp_, 
  ElementsPerAccess> {

  using RealEpilogue = DefaultEpilogueTensorOp<
    ThreadblockShape_,
    WarpMmaOperator_,
    PartitionsK,
    OutputOp_,
    ElementsPerAccess
  >;

  using Epilogue = EpiloguePlanarComplex<
    ThreadblockShape_,
    WarpMmaOperator_,
    PartitionsK,
    typename RealEpilogue::OutputTileIterator,
    typename RealEpilogue::AccumulatorFragmentIterator,
    typename RealEpilogue::WarpTileIterator,
    typename RealEpilogue::SharedLoadIterator,
    OutputOp_,
    typename RealEpilogue::Padding
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues.
template <
  typename ThreadblockShape_,
  typename WarpMmaOperator_,
  typename ArchTag_,
  int PartitionsK,
  typename OutputOp_,
  int ElementsPerAccess
>
struct DefaultEpiloguePlanarComplex<
  ThreadblockShape_, 
  WarpMmaOperator_, 
  arch::OpClassSimt, 
  ArchTag_,
  PartitionsK, 
  OutputOp_, 
  ElementsPerAccess> {

  using RealEpilogue = DefaultEpilogueSimt<
    ThreadblockShape_,
    WarpMmaOperator_,
    OutputOp_,
    ElementsPerAccess
  >;

  using Epilogue = EpiloguePlanarComplex<
    ThreadblockShape_,
    WarpMmaOperator_,
    PartitionsK,
    typename RealEpilogue::OutputTileIterator,
    typename RealEpilogue::AccumulatorFragmentIterator,
    typename RealEpilogue::WarpTileIterator,
    typename RealEpilogue::SharedLoadIterator,
    OutputOp_,
    typename RealEpilogue::Padding
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
