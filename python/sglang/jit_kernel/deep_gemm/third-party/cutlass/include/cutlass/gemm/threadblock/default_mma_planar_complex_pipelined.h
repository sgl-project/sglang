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
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include "cutlass/gemm/warp/mma_planar_complex.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/mma_planar_complex_pipelined.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template <
  /// Element type for A matrix operand
  typename ElementA_,
  /// Layout type for A matrix operand
  typename LayoutA_,
  /// Access granularity of A matrix in units of elements
  int kAlignmentA,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Layout type for B matrix operand
  typename LayoutB_,
  /// Access granularity of B matrix in units of elements
  int kAlignmentB,
  /// Element type for internal accumulation
  typename ElementAccumulator_,
  /// Layout type for C and D matrix operands
  typename LayoutC_,
  /// Operator class tag
  typename OperatorClass_,
  /// Tag indicating architecture to tune for
  typename ArchTag_,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape_,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape_,
  /// Instruction-level tile size (concept: GemmShape)
  typename InstructionShape_,
  /// Number of stages used in the pipelined mainloop
  int Stages,
  /// Complex transformation on operand A
  ComplexTransform TransformA = ComplexTransform::kNone,
  /// Complex transformation on operand B
  ComplexTransform TransformB = ComplexTransform::kNone,
  /// Math operator tag (e.g. arch::OpMultiplyAdd)
  typename Operator = arch::OpMultiplyAdd
>
struct DefaultMmaPlanarComplexPipelined {

  // Construct a planar complex variant from the real-valued variant
  using RealMma = typename DefaultMma<
    ElementA_,
    LayoutA_,
    kAlignmentA,
    ElementB_,
    LayoutB_,
    kAlignmentB,
    ElementAccumulator_,
    LayoutC_,
    OperatorClass_,
    ArchTag_,
    ThreadblockShape_,
    WarpShape_,
    InstructionShape_,
    Stages,
    Operator
  >::ThreadblockMma;

  using ThreadblockMma = MmaPlanarComplexPipelined<
    ThreadblockShape_,
    typename RealMma::IteratorA,
    typename RealMma::SmemIteratorA,
    typename RealMma::IteratorB,
    typename RealMma::SmemIteratorB,
    ElementAccumulator_,
    LayoutC_,
    typename RealMma::Policy,
    Stages,
    TransformA,
    TransformB
  >;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
