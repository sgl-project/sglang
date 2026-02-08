/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief High-level interface for running a grouped version of a CUTLASS kernel
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/trace.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/gemm_grouped_problem_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// High-level interface for running a grouped version of a CUTLASS kernel
template <
  typename BaseKernel_   ///! Kernel-scoped matrix multiply-accumulate
>
struct GroupedKernel {
public:

  using BaseKernel = BaseKernel_;
  using Epilogue = typename BaseKernel::Epilogue;

  /// Types that need to be exported to work properly with device::BaseGrouped
  using ElementA = typename BaseKernel::ElementA;
  using LayoutA = typename BaseKernel::LayoutA;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  static ComplexTransform const kTransformA = BaseKernel::kTransformA;
  static int const kAlignmentA = BaseKernel::kAlignmentA;

  using ElementB = typename BaseKernel::ElementB;
  using LayoutB = typename BaseKernel::LayoutB;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  static ComplexTransform const kTransformB = BaseKernel::kTransformB;
  static int const kAlignmentB = BaseKernel::kAlignmentB;

  using ElementC = typename BaseKernel::ElementC;
  using LayoutC = typename BaseKernel::LayoutC;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  static int const kAlignmentC = BaseKernel::kAlignmentC;

  using ElementAccumulator = typename BaseKernel::Mma::Policy::Operator::ElementC;

  using EpilogueOutputOp = typename BaseKernel::EpilogueOutputOp;
  using ThreadblockSwizzle = typename BaseKernel::ThreadblockSwizzle;

  using Operator = typename BaseKernel::Operator;
  using WarpMmaOperator = typename BaseKernel::Mma::Policy::Operator;

  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator = typename WarpMmaOperator::MathOperator;
  using OperatorClass = typename WarpMmaOperator::OperatorClass;
  using ArchTag = typename WarpMmaOperator::ArchTag;
  using ThreadblockShape = typename BaseKernel::Mma::Shape;
  using WarpShape = typename BaseKernel::WarpShape;
  using InstructionShape = typename BaseKernel::InstructionShape;
  static int const kStages = BaseKernel::Mma::kStages;

  using Mma = typename BaseKernel::Mma;

  using Arguments = typename BaseKernel::GroupedArguments;
  using Params = typename BaseKernel::GroupedParams;
  using ProblemVisitor = typename ThreadblockSwizzle::ProblemVisitor;

  static int const kThreadCount = BaseKernel::kThreadCount;

  /// Shared memory storage structure
  struct SharedStorage {
    typename BaseKernel::SharedStorage kernel;

    // ProblemVisitor shared storage can't be overlapped with others
    typename ProblemVisitor::SharedStorage problem_visitor;
  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  GroupedKernel() { }

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const & problem_size) {
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return Status::kSuccess;
  }

  /// Executes a kernel-level GEMM in a loop
  CUTLASS_DEVICE
  void operator()(Params &params, SharedStorage &shared_storage) {

    ThreadblockSwizzle swizzle(params.problem_visitor, shared_storage.problem_visitor, blockIdx.x);

    if (ProblemVisitor::kTransposed) {
      params.transpose();
    }

    BaseKernel mma;

    // Outer 'persistent' loop to iterate over tiles
    while (swizzle.problem_visitor.next_tile()) {

      typename BaseKernel::Params mma_params = params.to_single_params(swizzle.problem_visitor);
      mma.run_with_swizzle(mma_params, shared_storage.kernel, swizzle);

      // Next tile
      swizzle.problem_visitor.advance(gridDim.x);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
