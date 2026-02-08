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
    \brief Template for a pipelined Rank2K kernel. Does not compute batching or support split-K.

  
*/

#pragma once

#include "cutlass/blas3.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/rank_2k_universal.h"

#include "cutlass/gemm/kernel/default_rank_2k_universal.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Fill Mode for C (kLower or kUpper)
    FillMode FillModeC,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementC_,
    /// Operator class tag
    typename OperatorClass_ = arch::OpClassTensorOp,
    /// Tag indicating architecture to tune for
    typename ArchTag_ = arch::Sm80,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementA_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementA_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementA_, ElementC_,
        ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementA_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ =
        typename threadblock::GemmIdentityThreadblockSwizzle<>,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementA_,
                                 ElementC_, ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementA_,
                                 ElementC_, ElementAccumulator_>::kAlignmentA,

    /// Access granularity of B matrix in units of elements
    int AlignmentB =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementB_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentB,
    /// If true, kernel supports split-K with serial reduction
    bool SplitKSerial = false,
    /// Operation performed by SYRK
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator,
    /// Complex elementwise transformation 
    ComplexTransform TransformA = ComplexTransform::kNone,
    /// Complex elementwise transformation 
    ComplexTransform TransformB = ComplexTransform::kNone,
    /// Blas3 computation mode (symmetric/hermitian)
    BlasMode BlasMode_ = BlasMode::kSymmetric>
class Rank2K {
 public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static FillMode const kFillModeC = FillModeC;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = SplitKSerial;
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;
  static BlasMode const kBlasMode = BlasMode_;
  static int const kUpdateRank = 2;

  // static asserts for rank 2k update kernel
  static_assert(platform::is_same<LayoutA, LayoutB>::value,
    "Rank 2K update operator support same layouts for operandA and B");

  /// Define the kernel
  using Rank2Kkernel = typename kernel::DefaultRank2KUniversal<
    ElementA,
    LayoutA,
    kTransformA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kTransformB,
    kAlignmentB,
    ElementC,
    LayoutC,
    kFillModeC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    kStages,
    kSplitKSerial,
    Operator,
    kBlasMode
  >::Rank2Kkernel;
  
  using Arguments = typename Rank2Kkernel::Arguments;

private:

  /// Kernel parameters object
  typename Rank2Kkernel::Params params_;
public:

  /// Constructs the SYRK.
  Rank2K() { }

  /// Determines whether the SYRK can execute the given problem.
  static Status can_implement(Arguments const &args) {

    if (!kSplitKSerial && args.batch_count > 1) {
      return Status::kErrorInvalidProblem;
    }

    Status status = Rank2Kkernel::can_implement(args);
   
    if (FillModeC != FillMode::kLower && FillModeC != FillMode::kUpper) {
      return Status::kErrorInvalidProblem;
    }

    if (status != Status::kSuccess) {
      return status;
    }

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    
    size_t bytes = 0;

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord tiled_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size, 
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.batch_count);
    
    if (kSplitKSerial && args.batch_count > 1) {

      bytes += sizeof(int) * size_t(tiled_shape.m()) * size_t(tiled_shape.n());
    }

    return bytes;
  }

  /// Initializes SYRK state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {
    
    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size, 
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.batch_count);

    if (kSplitKSerial) {
      if (args.batch_count > 1) {
        if (!workspace) {
          return Status::kErrorWorkspaceNull;
        }

        size_t bytes = get_workspace_size(args);
      
        cudaError_t result = cudaMemsetAsync(workspace, 0, bytes, stream);

        if (result != cudaSuccess) {
          return Status::kErrorInternal;
        }
      }
    }
    else {

      if (args.batch_count > 1) {
        return Status::kErrorInvalidProblem;
      }
    }
    
    int gemm_k_size = args.problem_size.k();

    // Initialize the Params structure
    params_ = typename Rank2Kkernel::Params{
      args,
      grid_tiled_shape,
      gemm_k_size,
      static_cast<int *>(workspace)
    };
    
    int smem_size = int(sizeof(typename Rank2Kkernel::SharedStorage));
    
    if (smem_size >= (48 << 10)) {
      cudaError_t result = cudaFuncSetAttribute(Kernel<Rank2Kkernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {
    
    if (kSplitKSerial && args.batch_count > 1) {  
      if (!workspace) {
        return Status::kErrorWorkspaceNull;
      }
    }

    size_t workspace_bytes = get_workspace_size(args);

    if (workspace_bytes && !workspace) {
      return Status::kErrorWorkspaceNull;
    }

    params_.update(args, workspace);

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(Rank2Kkernel::kThreadCount, 1, 1);

    int smem_size = int(sizeof(typename Rank2Kkernel::SharedStorage));

    cutlass::arch::synclog_setup();
    cutlass::Kernel<Rank2Kkernel><<<grid, block, smem_size, stream>>>(params_);

    cudaError_t result = cudaGetLastError();

    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr) {
    
    Status status = initialize(args, workspace);
    
    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};
////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for column-major output exchange operand.
template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Fill Mode for C (kLower or kUpper)
    FillMode FillModeC,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for.  This is the minimum SM that
    /// supports the intended feature. The device kernel can be built
    /// targeting any SM larger than this number.
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Epilogue output operator
    typename EpilogueOutputOp_,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB,
    /// If true, kernel supports split-K with serial reduction
    bool SplitKSerial,
    /// Operation performed by Rank2K update kernel
    typename Operator_,
    /// Complex elementwise transformation 
    ComplexTransform TransformA,
    /// Complex elementwise transformation 
    ComplexTransform TransformB,
    /// Blas3 computation mode (symmetric/hermitian)
    BlasMode BlasMode_
    >
class Rank2K<ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_,
           layout::ColumnMajor,  // partially specialized on LayoutC
           FillModeC, ElementAccumulator_, OperatorClass_, ArchTag_, ThreadblockShape_,
           WarpShape_, InstructionShape_, EpilogueOutputOp_,
           ThreadblockSwizzle_, Stages, AlignmentA, AlignmentB,
           SplitKSerial, Operator_, TransformA, TransformB, BlasMode_> {
 public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using ElementC = ElementC_;
  using LayoutC = layout::ColumnMajor;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static FillMode const kFillModeC = FillModeC;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = SplitKSerial;
  static BlasMode const kBlasMode = BlasMode_;
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;
  static int const kUpdateRank = 2;
  
  /// Define the kernel
  using UnderlyingOperator = typename cutlass::gemm::device::Rank2K<
    ElementB,
    LayoutB,
    ElementA,
    LayoutA,
    ElementC,
    layout::RowMajor,
    InvertFillMode<FillModeC>::mode,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    kStages,
    kAlignmentB,
    kAlignmentA,
    kSplitKSerial,
    Operator,
    kTransformA,
    kTransformB,
    kBlasMode
  >;
  

  /// Argument structure
  using Arguments = typename UnderlyingOperator::Arguments;
  using Rank2Kkernel = typename UnderlyingOperator::Rank2Kkernel;

private:

  UnderlyingOperator underlying_operator_;

public:

  /// Constructs the Rank2K.
  Rank2K() { }

  /// Helper to construct a transposed equivalent for the underlying Rank2K operator
  static Arguments to_underlying_arguments(Arguments const &args) {
    return args.transposed_problem();
  }

  /// Determines whether the Rank2K can execute the given problem.
  static Status can_implement(Arguments const &args) {

    return UnderlyingOperator::can_implement(to_underlying_arguments(args));
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    
    return UnderlyingOperator::get_workspace_size(to_underlying_arguments(args));
  }

  /// Computes the grid shape
  static dim3 get_grid_shape(Arguments const &args) { 
    return UnderlyingOperator::get_grid_shape(to_underlying_arguments(args));
  }

  /// Computes the maximum number of active blocks per multiprocessor
  static int maximum_active_blocks(int smem_capacity = -1) {
    return UnderlyingOperator::maximum_active_blocks(smem_capacity);
  }

  /// Initializes Rank2K state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    return underlying_operator_.initialize(to_underlying_arguments(args), workspace, stream);
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {

    return underlying_operator_.update(to_underlying_arguments(args), workspace);
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    return underlying_operator_.run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr) {
    
    Status status = initialize(args, workspace, stream);
    
    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace Rank2K
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
