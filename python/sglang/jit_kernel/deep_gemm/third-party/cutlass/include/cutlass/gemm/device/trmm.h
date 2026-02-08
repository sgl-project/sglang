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
    \brief Template for a TRMM kernel. Does not compute batching or support split-K.

  
*/

#pragma once

#include "cutlass/blas3.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/trmm_universal.h"

#include "cutlass/gemm/kernel/default_trmm_universal.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

/*! Trmm device-level operator. This is an interface to efficient CUTLASS TRMM kernels that may
  be invoked from host code.

  The contributions of this class are:
    
    1. At compile time, it maps data types and high-level structural parameters onto 
       specific CUTLASS components.

    2. At runtime, it maps logical arguments to TRMM problems to kernel parameters.

    3. At runtime, it launches kernels on the device.

  The intent is to provide a convenient mechanism for interacting with most plausible TRMM
  configurations for each supported architecture. Consequently, not all parameters are exposed
  to the top-level interface. Rather, sensible defaults at each level of the CUTLASS hierarchy
  are selected to tradeoff simplicity of the interface with flexibility. We expect 
  most configurations to be specified at this level. Applications with more exotic requirements 
  may construct their kernels of interest using CUTLASS components at the threadblock, warp, 
  and thread levels of abstraction.

  CUTLASS exposes computations using the functor design pattern in which objects compose some
  internal state with an overloaded function call operator. This enables decoupling of
  initialization from execution, possibly reducing overhead during steady state phases of
  application execution.

  CUTLASS device-level operators expose an Arguments structure encompassing each logical
  input to the computation. This is distinct from the kernel-level Params structure pattern
  which contains application-specific precomputed state needed by the device code.

  Example of a CUTLASS TRMM operator implementing the functionality of cuBLAS's STRMM NN
  is as follows:

    //
    // Instantiate the CUTLASS TRMM operator.
    //

    cutlass::gemm::device::Trmm<
      float,
      cutlass::layout::ColumnMajor,
      cutlass::SideMode::kLeft,
      cutlass::FillMode::kLower,
      cutlass::DiagType::kNonUnit,
      float,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::ColumnMajor,
    > trmm_op;

    //
    // Launch the TRMM operation on the device
    //

    cutlass::Status status = trmm_op({
      cutlass::gemm::GemmUniversalMode,   // Trmm Problem Mode
      {m, n, m/n},                        // GemmCoord problem_size (k is based on left- or right-side mode)
      batch_count,
      {alpha},                            // EpilogueOutputOp::Params epilogue_op_params
      void const * ptr_A,
      void const * ptr_B,
      void const * ptr_C,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_C,
      int lda,
      int ldb,
      int ldc
    });

  A simplified view of the template is listed below.

    template <
      /// Element type for A matrix operand
      typename ElementA,
      
      /// Layout type for A matrix operand
      typename LayoutA,
      
      /// Side Mode for A (kLeft or kRight)
      SideMode SideModeA,

      /// Fill Mode for A (kLower or kUpper)
      FillMode FillModeA,

      /// DiagType for A (kNonUnit or kUnit)
      DiagType DiagTypeA,

      /// Element type for B matrix operand
      typename ElementB,
      
      /// Layout type for B matrix operand
      typename LayoutB,
      
      /// Element type for C and D matrix operands
      typename ElementC,
      
      /// Layout type for C and D matrix operands
      typename LayoutC,
      
      /// Element type for internal accumulation
      typename ElementAccumulator,

      /// Operator class tag
      typename OperatorClass,
      
      /// Tag indicating architecture to tune for.  This is the minimum SM that
      /// supports the intended feature. The device kernel can be built
      /// targeting any SM larger than this number.
      typename ArchTag,
      
      /// Threadblock-level tile size (concept: GemmShape)
      typename ThreadblockShape,
      
      /// Warp-level tile size (concept: GemmShape)
      typename WarpShape,
      
      /// Warp-level tile size (concept: GemmShape)
      typename InstructionShape,
      
      /// Epilogue output operator
      typename EpilogueOutputOp,
      
      /// Threadblock-level swizzling operator
      typename ThreadblockSwizzle,
      
      /// Number of stages used in the pipelined mainloop
      int Stages,

      /// Access granularity of A matrix in units of elements
      int AlignmentA,

      /// Access granularity of B matrix in units of elements
      int AlignmentB,

      /// If true, kernel supports split-K with serial reduction
      bool SplitKSerial,

      /// Operation performed by TRMM
      typename Operator,

      /// Complex elementwise transformation on A operand
      ComplexTransform TransformA
    >
    class Trmm;
*/
template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Side Mode for A 
    SideMode SideModeA,
    /// Fill Mode for A
    FillMode FillModeA,
    /// DiagType for A
    DiagType DiagTypeA,
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
    /// Operation performed by TRMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA = ComplexTransform::kNone>
class Trmm {
 public:
  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementAKernel = typename platform::conditional<(SideModeA == SideMode::kRight), ElementB_, ElementA_>::type;
  using LayoutAKernel = typename platform::conditional<(SideModeA == SideMode::kRight), LayoutB_, LayoutA_>::type;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementBKernel = typename platform::conditional<(SideModeA == SideMode::kRight), ElementA_, ElementB_>::type;
  using LayoutBKernel = typename platform::conditional<(SideModeA == SideMode::kRight), LayoutA_, LayoutB_>::type;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static SideMode const kSideMode = SideModeA;
  static FillMode const kFillMode = FillModeA;
  static DiagType const kDiagType = DiagTypeA;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentAKernel = (SideModeA == SideMode::kRight) ? AlignmentB : AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentBKernel = (SideModeA == SideMode::kRight) ? AlignmentA : AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = SplitKSerial;
  // Complex Transform don't apply to B
  static ComplexTransform const kTransformA = TransformA; 
  static ComplexTransform const kTransformB = ComplexTransform::kNone; 
  static ComplexTransform const kTransformAKernel = (SideModeA == SideMode::kRight) ? 
                                              ComplexTransform::kNone : TransformA;
  static ComplexTransform const kTransformBKernel = (SideModeA == SideMode::kRight) ? 
                                              TransformA : ComplexTransform::kNone;

  /// Define the kernel
  using TrmmKernel = typename kernel::DefaultTrmmUniversal<
    ElementAKernel,
    LayoutAKernel,
    kTransformAKernel,
    kAlignmentAKernel,
    ElementBKernel,
    LayoutBKernel,
    kTransformBKernel,
    kAlignmentBKernel,
    kSideMode,
    kFillMode,
    kDiagType,
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
    Operator
  >::TrmmKernel;
  
  using Arguments = typename TrmmKernel::Arguments;

private:

  /// Kernel parameters object
  typename TrmmKernel::Params params_;
public:

  /// Constructs the TRMM.
  Trmm() { }

  /// Determines whether the TRMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    if (!kSplitKSerial && args.batch_count > 1) {
      return Status::kErrorInvalidProblem;
    }

    Status status = TrmmKernel::can_implement(args);
   
    if (SideModeA == SideMode::kInvalid) {
      return Status::kErrorInvalidProblem;
    }

    if (FillModeA == FillMode::kInvalid) {
      return Status::kErrorInvalidProblem;
    }

    if (DiagTypeA == DiagType::kInvalid) {
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

  /// Initializes TRMM state from arguments.
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
    if (kSideMode == SideMode::kRight) {
      // Initialize the Params structure
      params_ = typename TrmmKernel::Params{
        args.swapped_matrices(),
        grid_tiled_shape,
        gemm_k_size,
        static_cast<int *>(workspace)
      };

      return Status::kSuccess;
    }

    // Initialize the Params structure
    params_ = typename TrmmKernel::Params{
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
    dim3 block(TrmmKernel::kThreadCount, 1, 1);

    int smem_size = int(sizeof(typename TrmmKernel::SharedStorage));
    
    if (smem_size >= (48 << 10)) {
      cudaError_t result = cudaFuncSetAttribute(Kernel<TrmmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    cutlass::arch::synclog_setup();
    cutlass::Kernel<TrmmKernel><<<grid, block, smem_size, stream>>>(params_);

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

/********************************************************************************************************
  TRMM has 4 combinations based on Layouts {RowMajor, ColumnMajor} x Side mode {LeftSide, RightSide}
  In templates and arguments to cutlass kernel, `matrix A` is always triangular, and `matrix B` is rectangular. 
  (adhering to the cuBLAS convention)

For the mainloop and trmm kernel, `A` and `B` points to left-side and right-side matrices, respectively.
  
  Thus, for LeftSide mode `A` and `B` points to `matrix A` and `matrix B`, respectively. While for 
  the RightSide mode `A` and `B` points to `matrix B` and `matrix A`, respectively. 
  
  Additionally, CUTLASS GEMM epilogue is always RowMajor, and ColumnMajor output is achieved by 
  transposing the GEMM problem. Thus, ColumnMajor output layout for TRMM requires:
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
    /// Side Mode for A 
    SideMode SideModeA,
    /// Fill Mode for A
    FillMode FillModeA,
    /// DiagType for A
    DiagType DiagTypeA,
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
    /// Tag indicating architecture to tune for
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
    /// If true, kernel supports split-K as a serial reduction
    bool SplitKSerial,
    /// Operation performed by TRMM
    typename Operator_,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA>
class Trmm<ElementA_, LayoutA_, SideModeA, FillModeA, DiagTypeA,
           ElementB_, LayoutB_, ElementC_,
           layout::ColumnMajor,  // partially specialized on LayoutC
           ElementAccumulator_, OperatorClass_, ArchTag_, ThreadblockShape_,
           WarpShape_, InstructionShape_, EpilogueOutputOp_,
           ThreadblockSwizzle_, Stages, AlignmentA, AlignmentB, SplitKSerial,
           Operator_, TransformA> {
 public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_; 
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = layout::ColumnMajor;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static SideMode const kSideMode = SideModeA;
  static FillMode const kFillMode = FillModeA;
  static DiagType const kDiagType = DiagTypeA;
  // Changing SideMode as we change the layout
  static SideMode const kSideModeT = (SideModeA == SideMode::kLeft) ?
                                      SideMode::kRight : SideMode::kLeft;
  // Changing FillMode as we change the layout
  static FillMode const kFillModeT = (FillModeA == FillMode::kLower) ? 
                                      FillMode::kUpper : FillMode::kLower;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static ComplexTransform const kTransformA = TransformA;
  // Complex Transform don't apply to B
  static ComplexTransform const kTransformB = ComplexTransform::kNone; 
  static bool const kSplitKSerial = SplitKSerial;

  using UnderlyingOperator = Trmm<
    ElementA,
    typename layout::LayoutTranspose<LayoutA>::type,
    kSideModeT,
    kFillModeT,
    kDiagType,
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
    TransformA
  >;

  using Arguments = typename UnderlyingOperator::Arguments;
  using TrmmKernel = typename UnderlyingOperator::TrmmKernel;
  static int const kAlignmentC = UnderlyingOperator::kAlignmentC;

private:

  UnderlyingOperator underlying_operator_;

public:

  /// Constructs the TRMM.
  Trmm() { }

  /// Helper to construct a transposed equivalent for the underlying TRMM operator which is identical
  static Arguments to_underlying_arguments(Arguments const &args) {
    return args.transposed_problem_size();
  }

  /// Determines whether the TRMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    return UnderlyingOperator::can_implement(to_underlying_arguments(args));
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    
    return UnderlyingOperator::get_workspace_size(to_underlying_arguments(args));
  }

  /// Initializes TRMM state from arguments.
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
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
