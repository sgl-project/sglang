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
    \brief Template for a pipelined SYMM and HEMM kernels. Does not compute batching or support split-K.

  
*/

#pragma once

#include "cutlass/blas3.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/symm_universal.h"

#include "cutlass/gemm/kernel/default_symm_universal.h"
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
    /// Side Mode for A (kLeft or kRight)
    SideMode SideModeA,
    /// Fill Mode for A (kLower or kUpper)
    FillMode FillModeA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementC_,
    /// Operator class tag
    typename OperatorClass_ = arch::OpClassTensorOp,
    /// Tag indicating architecture to tune for
    typename ArchTag_ = arch::Sm80,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp_ = epilogue::thread::LinearCombination<
      ElementC_,
      128 / sizeof_bits<ElementC_>::value,
      ElementAccumulator_,
      ElementAccumulator_,
      epilogue::thread::ScaleType::OnlyAlphaScaling
    >,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ = threadblock::GemmIdentityThreadblockSwizzle<>,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentB,
    /// If true, kernel supports split-K with serial reduction
    bool SplitKSerial = false,
    /// Operation performed by SYMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator,
    /// Blas3 computation mode (symmetric/hermitian)
    BlasMode BlasMode_ = BlasMode::kSymmetric>
class Symm {
 public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using ElementAKernel = typename platform::conditional<(SideModeA == SideMode::kRight), ElementB_, ElementA_>::type;
  using LayoutAKernel = typename platform::conditional<(SideModeA == SideMode::kRight), LayoutB_, LayoutA_>::type;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using ElementBKernel = typename platform::conditional<(SideModeA == SideMode::kRight), ElementA_, ElementB_>::type;
  using LayoutBKernel = typename platform::conditional<(SideModeA == SideMode::kRight), LayoutA_, LayoutB_>::type;
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
  static SideMode const kSideModeA = SideModeA;
  static FillMode const kFillModeA = FillModeA;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentAKernel = (SideModeA == SideMode::kRight) ? AlignmentB : AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentBKernel = (SideModeA == SideMode::kRight) ? AlignmentA : AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = SplitKSerial;
  static BlasMode const kBlasMode = BlasMode_;

  // static asserts for symm update kernel
  static_assert(platform::is_same<LayoutA, LayoutB>::value,
    "SYMM update operator support same layouts for operand A and B");

  /// Define the kernel
  using SymmKernel = typename kernel::DefaultSymmUniversal<
    ElementAKernel,
    LayoutAKernel,
    kSideModeA,
    kFillModeA,
    kAlignmentAKernel,
    ElementBKernel,
    LayoutBKernel,
    kAlignmentBKernel,
    ElementC,
    LayoutC,
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
  >::SymmKernel;
  
  using Arguments = typename SymmKernel::Arguments;

private:

  /// Kernel parameters object
  typename SymmKernel::Params params_;
public:

  /// Constructs the SYMM.
  Symm() { }

  /// Determines whether the SYMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    if (!kSplitKSerial && args.batch_count > 1) {
      return Status::kErrorInvalidProblem;
    }

    Status status = SymmKernel::can_implement(args);

    if (SideModeA == SideMode::kInvalid) {
      return Status::kErrorInvalidProblem;
    }
   
    if (FillModeA != FillMode::kLower && FillModeA != FillMode::kUpper) {
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

  /// Initializes SYMM state from arguments.
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

   // Swapping argument for A and B, if A was on the right side (problem size doesn't need to change here).
    if (kSideModeA == SideMode::kRight) {
      // Initialize the Params structure
      params_ = typename SymmKernel::Params{
        args.swapped_matrices(),
        grid_tiled_shape,
        gemm_k_size,
        static_cast<int *>(workspace)
      };

      return Status::kSuccess;
    }

    // Initialize the Params structure
    params_ = typename SymmKernel::Params{
      args,
      grid_tiled_shape,
      gemm_k_size,
      static_cast<int *>(workspace)
    };
    
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
    dim3 block(SymmKernel::kThreadCount, 1, 1);

    int smem_size = int(sizeof(typename SymmKernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      cudaError_t result = cudaFuncSetAttribute(Kernel<SymmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    cutlass::arch::synclog_setup();
    cutlass::Kernel<SymmKernel><<<grid, block, smem_size, stream>>>(params_);

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

/********************************************************************************************************
  SYMM/HEMM has 4 combinations based on Layouts {RowMajor, ColumnMajor} x Side mode {LeftSide, RightSide}
  In templates and arguments to cutlass kernel, `matrix A` is always symmetric/hermitian, and `matrix B` is rectangular. 
  (adhering to the cuBLAS convention)

  Although, cuBLAS SYMM/HEMM only supports ColumnMajor layouts for all matrices (A, B, C/D).

  For the mainloop and symm kernel, `A` and `B` points to left-side and right-side matrices, respectively.
  
  Thus, for LeftSide mode `A` and `B` points to `matrix A` and `matrix B`, respectively. While for 
  the RightSide mode `A` and `B` points to `matrix B` and `matrix A`, respectively. 
  
  Additionally, CUTLASS GEMM epilogue is always RowMajor, and ColumnMajor output is achieved by 
  transposing the GEMM problem. Thus, ColumnMajor output layout for SYMM/HEMM requires:
   - Transposing `matrix A` and `matrix B` layouts
   - Swapping problem size m and n values
   - Swapping LeftSide and RightSide mode
  
  RowMajor output:    D = matrix A x matrix B
  ColumnMajor output: D = matrix A x matrix B -> Transpose (D) = Transpose(matrix B) x Transpose(matrix A)

  {RowMajor, ColumnMajor} x Side Mode {LeftSide, RightSide} 4 cases:
    1.  LeftSide mode and RowMajor output (default template)
    2.  LeftSide mode and ColumnMajor output 
    3.  RightSide mode and RowMajor output
    4.  RightSide mode and ColumnMajor output
  
  Mapping ColumnMajor output layout cases 2 and 4 to RowMajor efficient epilogue implementation:
  
  Case 2 -> Case 3:
      D_col = matrix A x matrix B (LeftSide mode) 
   => Transpose(D_col) = Transpose(matrix B) x Transpose(matrix A) (RightSide mode)

  swap pointers for `A` and `B` call GEMM mainloop with RowMajor efficient-epilogue

  Case 4 -> Case 1:
      D_col = matrix B x matrix A (RightSide mode) 
   => Transpose(D_col) = Transpose(matrix A) x Transpose(matrix B) (LeftSide mode)

   call GEMM mainloop for with RowMajor efficient-epilogue
********************************************************************************************************/

/// Partial specialization for column-major output exchanges problem size and operand.
template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Side Mode for A (kLeft or kRight)
    SideMode SideModeA,
    /// Fill Mode for A (kLower or kUpper)
    FillMode FillModeA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
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
    /// Operation performed by Symm update kernel
    typename Operator_,
    /// Blas3 computation mode (symmetric/hermitian)
    BlasMode BlasMode_
    >
class Symm<ElementA_, LayoutA_, SideModeA, FillModeA, ElementB_, LayoutB_, ElementC_,
           layout::ColumnMajor,  // partially specialized on LayoutC
           ElementAccumulator_, OperatorClass_, ArchTag_, ThreadblockShape_,
           WarpShape_, InstructionShape_, EpilogueOutputOp_,
           ThreadblockSwizzle_, Stages, AlignmentA, AlignmentB,
           SplitKSerial, Operator_, BlasMode_> {
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
  static SideMode const kSideModeA = SideModeA;
  static FillMode const kFillModeA = FillModeA;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = SplitKSerial;
  static BlasMode const kBlasMode = BlasMode_;
  
  /// Define the kernel
  using UnderlyingOperator = typename cutlass::gemm::device::Symm<
    ElementA,
    typename layout::LayoutTranspose<LayoutA>::type,
    InvertSideMode<kSideModeA>::mode,
    InvertFillMode<kFillModeA>::mode,
    ElementB,
    typename layout::LayoutTranspose<LayoutB>::type, 
    ElementC,
    layout::RowMajor,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    kStages,
    kAlignmentA,
    kAlignmentB,
    kSplitKSerial,
    Operator,
    kBlasMode
  >;
  

  /// Argument structure
  using Arguments = typename UnderlyingOperator::Arguments;
  using SymmKernel = typename UnderlyingOperator::SymmKernel;

private:

  UnderlyingOperator underlying_operator_;

public:

  /// Constructs the Symm.
  Symm() { }

  /// Helper to construct a transposed equivalent for the underlying SYMM operator
  static Arguments to_underlying_arguments(Arguments const &args) {
    return args.transposed_problem_size();
  }

  /// Determines whether the Symm can execute the given problem.
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

  /// Initializes Symm state from arguments.
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
} // namespace Symm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
