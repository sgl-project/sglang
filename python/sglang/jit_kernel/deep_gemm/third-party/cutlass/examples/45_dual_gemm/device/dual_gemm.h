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
    \brief Performs a dual gemm in one fused kernel:
```
D0 = epilogue0(X @ B0, C0)
D1 = epilogue1(X @ B1, C1)
D2 = element_wise(D0, D1)
```
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"

#include "../kernel/dual_gemm.h"
#include "../dual_gemm_common.h"

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
    /// Layout type for B0 matrix operand
    typename LayoutB0_,
    /// Layout type for B1 matrix operand
    typename LayoutB1_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
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
    typename EpilogueOutputOp0_,
    typename EpilogueOutputOp1_,
    typename EpilogueOutputOp2_,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ = threadblock::GemmIdentityThreadblockSwizzle<>,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    bool StoreD0 = true,
    bool StoreD1 = true,
    /// If true, kernel supports split-K with serial reduction
    bool SplitKSerial = false,
    /// Access granularity of A matrix in units of elements
    int AlignmentA =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentB,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator>
class DualGemm {
 public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB0 = LayoutB0_;
  using LayoutB1 = LayoutB1_;
  using TensorRefB0 = TensorRef<ElementB const, LayoutB0>;
  using TensorRefB1 = TensorRef<ElementB const, LayoutB1>;
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
  using EpilogueOutputOp0 = EpilogueOutputOp0_;
  using EpilogueOutputOp1 = EpilogueOutputOp1_;
  using EpilogueOutputOp2 = EpilogueOutputOp2_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp1::kCount;
  static bool const kSplitKSerial = SplitKSerial;
  static bool constexpr kStoreD0 = StoreD0;
  static bool constexpr kStoreD1 = StoreD1;
  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  using LayoutScaleBias = layout::RowMajor;
  /// Define the kernel
  /// Define the threadblock-scoped matrix multiply-accumulate
  static_assert(ArchTag::kMinComputeCapability >= 80, "Only multistage is implemented");
  static_assert(kStages >= 3, "Only multistage is implemented");
  using Mma0 = typename cutlass::gemm::threadblock::DefaultMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB0, kAlignmentB,
      ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, ArchTag,
      ThreadblockShape, WarpShape,
      InstructionShape, Stages, Operator>::ThreadblockMma;
  using Mma1 = typename cutlass::gemm::threadblock::DefaultMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB1, kAlignmentB,
      ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, ArchTag,
      ThreadblockShape, WarpShape, 
      InstructionShape, Stages, Operator>::ThreadblockMma;
  using DualMma = threadblock::DualMmaMultistage<
    typename Mma0::Shape,
    typename Mma0::IteratorA,
    typename Mma0::SmemIteratorA,
    Mma0::kCacheOpA,
    typename Mma0::IteratorB,
    typename Mma0::SmemIteratorB,
    Mma0::kCacheOpB,
    typename Mma1::IteratorB,
    typename Mma1::SmemIteratorB,
    typename Mma0::ElementC,
    typename Mma0::LayoutC,
    typename Mma0::Policy,
    typename Mma1::Policy,
    Mma0::kStages,
    SharedMemoryClearOption::kNone
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue
  using Epilogue0 =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, typename DualMma::Operator0, kPartitionsK, EpilogueOutputOp0,
          EpilogueOutputOp0::kCount>::Epilogue;
  using Epilogue1 =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, typename DualMma::Operator1, kPartitionsK, EpilogueOutputOp1,
          EpilogueOutputOp1::kCount>::Epilogue;

  /// Define the kernel-level GEMM operator.
  using DualGemmKernel = kernel::DualGemm<
    DualMma,
    Epilogue0, Epilogue1, EpilogueOutputOp2,
    ThreadblockSwizzle, kSplitKSerial,
    kStoreD0, kStoreD1>;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    DualGemmMode mode;
    GemmCoord problem_size;
    TensorRef<ElementA const, LayoutA> ref_A0;
    TensorRef<ElementB const, LayoutB0> ref_B0;
    TensorRef<ElementC const, LayoutC> ref_C0;
    TensorRef<ElementC, LayoutC> ref_D0;
    TensorRef<ElementB const, LayoutB1> ref_B1;
    TensorRef<ElementC const, LayoutC> ref_C1;
    TensorRef<ElementC, LayoutC> ref_D1;
    TensorRef<ElementC, LayoutC> ref_D2;
    typename EpilogueOutputOp0::Params epilogue0;
    typename EpilogueOutputOp1::Params epilogue1;
    typename EpilogueOutputOp2::Params epilogue2;
    int split_k_slices;

    int batch_count;
    int64_t batch_stride_A;
    int64_t batch_stride_B0;
    int64_t batch_stride_B1;
    int64_t batch_stride_C;
    int64_t batch_stride_D;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments(): problem_size(0, 0, 0), split_k_slices(1) {

    }

    /// Constructs an Arguments structure 
    CUTLASS_HOST_DEVICE
    Arguments(
      DualGemmMode mode,
      GemmCoord problem_size_,
      TensorRef<ElementA const, LayoutA> ref_A0_,
      TensorRef<ElementB const, LayoutB0> ref_B0_,
      TensorRef<ElementC const, LayoutC> ref_C0_,
      TensorRef<ElementC, LayoutC> ref_D0_,
      TensorRef<ElementB const, LayoutB1> ref_B1_,
      TensorRef<ElementC const, LayoutC> ref_C1_,
      TensorRef<ElementC, LayoutC> ref_D1_,
      TensorRef<ElementC, LayoutC> ref_D2_,
      typename EpilogueOutputOp0::Params epilogue0_ =
        typename EpilogueOutputOp0::Params(),
      typename EpilogueOutputOp1::Params epilogue1_ =
        typename EpilogueOutputOp1::Params(),
      typename EpilogueOutputOp2::Params epilogue2_ =
        typename EpilogueOutputOp2::Params(),
      int split_k_slices_ = 1,
      int batch_count = 1,
      int64_t batch_stride_A = 0,
      int64_t batch_stride_B0 = 0,
      int64_t batch_stride_B1 = 0,
      int64_t batch_stride_C = 0,
      int64_t batch_stride_D = 0
    ):
      mode(mode),
      problem_size(problem_size_),
      ref_A0(ref_A0_),
      ref_B0(ref_B0_),
      ref_C0(ref_C0_),
      ref_D0(ref_D0_),
      ref_B1(ref_B1_),
      ref_C1(ref_C1_),
      ref_D1(ref_D1_),
      ref_D2(ref_D2_),
      epilogue0(epilogue0_),
      epilogue1(epilogue1_),
      epilogue2(epilogue2_),
      split_k_slices(split_k_slices_),
      batch_count(batch_count),
      batch_stride_A(batch_stride_A),
      batch_stride_B0(batch_stride_B0),
      batch_stride_B1(batch_stride_B1),
      batch_stride_C(batch_stride_C),
      batch_stride_D(batch_stride_D) {

    }
  };

private:

  /// Kernel parameters object
  typename DualGemmKernel::Params params_;

public:

  /// Constructs the GEMM.
  DualGemm() = default;

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    if (args.mode == DualGemmMode::kBatched && kSplitKSerial) {
      return Status::kErrorInvalidProblem;
    }
    if (!kSplitKSerial && args.split_k_slices > 1) {
      return Status::kErrorInvalidProblem;
    }
    if (kStoreD0 != (args.ref_D0.data() != nullptr)) {
      return Status::kErrorInternal;
    }
    if (kStoreD1 != (args.ref_D1.data() != nullptr)) {
      return Status::kErrorInternal;
    }

    Status status = DualGemmKernel::can_implement(
      args.problem_size,
      args.ref_A0.non_const_ref(),
      args.ref_B0.non_const_ref(),
      args.ref_C0.non_const_ref(),
      args.ref_D0,
      args.ref_B1.non_const_ref(),
      args.ref_C1.non_const_ref(),
      args.ref_D1,
      args.ref_D2
    );

    if (status != Status::kSuccess) {
      return status;
    }

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {

    size_t bytes = 0;

    if (kSplitKSerial && args.split_k_slices > 1) {
      // Determine grid shape
      ThreadblockSwizzle threadblock_swizzle;

      cutlass::gemm::GemmCoord tiled_shape = threadblock_swizzle.get_tiled_shape(
        args.problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.split_k_slices);

      bytes += sizeof(int) * size_t(tiled_shape.m()) * size_t(tiled_shape.n());
    }

    return bytes;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size, 
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.mode == DualGemmMode::kBatched ? args.batch_count : args.split_k_slices);

    if (kSplitKSerial) {
      if (args.split_k_slices > 1) {
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

      if (args.split_k_slices > 1) {
        return Status::kErrorInvalidProblem;
      }
    }

    // Initialize the Params structure
    params_ = typename DualGemmKernel::Params{
      args.mode,
      args.problem_size,
      grid_shape,
      args.ref_A0.non_const_ref(),
      args.ref_B0.non_const_ref(),
      args.ref_C0.non_const_ref(),
      args.ref_D0,
      args.ref_B1.non_const_ref(),
      args.ref_C1.non_const_ref(),
      args.ref_D1,
      args.ref_D2,
      args.epilogue0,
      args.epilogue1,
      args.epilogue2,
      reinterpret_cast<int *>(workspace),
      args.batch_stride_A,
      args.batch_stride_B0,
      args.batch_stride_B1,
      args.batch_stride_C,
      args.batch_stride_D,
    };

    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {
    
    if (kSplitKSerial && args.split_k_slices > 1) {  
      if (!workspace) {
        return Status::kErrorWorkspaceNull;
      }
    }

    params_.ref_A0.reset(args.ref_A0.non_const_ref().data());
    params_.ref_B0.reset(args.ref_B0.non_const_ref().data());
    params_.ref_C0.reset(args.ref_C0.non_const_ref().data());
    params_.ref_D0.reset(args.ref_D0.data());
    params_.ref_B1.reset(args.ref_B1.non_const_ref().data());
    params_.ref_C1.reset(args.ref_C1.non_const_ref().data());
    params_.ref_D1.reset(args.ref_D1.data());
    params_.ref_D2.reset(args.ref_D2.data());
    params_.output_op_0 = args.epilogue0;
    params_.output_op_1 = args.epilogue1;
    params_.output_op_2 = args.epilogue2;
    params_.semaphore = reinterpret_cast<int *>(workspace);

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(DualGemmKernel::kThreadCount, 1, 1);

    cudaError_t result;

    int smem_size = int(sizeof(typename DualGemmKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel<DualGemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    cutlass::Kernel<DualGemmKernel><<<grid, block, smem_size, stream>>>(params_);

    result = cudaGetLastError();

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
    
    Status status = initialize(args, workspace, stream);
    
    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
