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
/* \file
   \brief Template for device-level Depthwise Convolution
*/

#pragma once

#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/conv/convolution.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename DirectConvolutionKernel_>
class DirectConvolution {
public:

  using UnderlyingKernel = DirectConvolutionKernel_;

  using ElementA = typename UnderlyingKernel::ElementA;
  using LayoutA = typename UnderlyingKernel::LayoutA;
  using ElementB = typename UnderlyingKernel::ElementB;
  using LayoutB = typename UnderlyingKernel::LayoutB;
  using ElementC = typename UnderlyingKernel::ElementC;
  using LayoutC = typename UnderlyingKernel::LayoutC;
  using ElementAccumulator = typename UnderlyingKernel::ElementAccumulator;
  using ElementCompute = typename UnderlyingKernel::ElementCompute;
  using OperatorClass = typename UnderlyingKernel::OperatorClass;
  using ArchTag = typename UnderlyingKernel::ArchTag;
  using ThreadblockShape = typename UnderlyingKernel::ThreadblockShape;
  using WarpShape = typename UnderlyingKernel::WarpShape;
  using InstructionShape = typename UnderlyingKernel::InstructionShape;
  using ThreadblockSwizzle = typename UnderlyingKernel::ThreadblockSwizzle;
  using EpilogueOutputOp = typename UnderlyingKernel::EpilogueOutputOp;
  static int const kStages = UnderlyingKernel::kStages;
  static int const kConvDim = UnderlyingKernel::kConvDim;
  using WarpMmaOperator = typename UnderlyingKernel::WarpMmaOperator;
  using ArchMmaOperator = typename UnderlyingKernel::ArchMmaOperator;
  using MathOperator = typename UnderlyingKernel::MathOperator; 

  static cutlass::conv::Operator const kConvolutionalOperator = UnderlyingKernel::kConvolutionalOperator;
  static cutlass::conv::IteratorAlgorithm const kIteratorAlgorithm = UnderlyingKernel::kIteratorAlgorithm;
  static cutlass::conv::StrideSupport const kStrideSupport = UnderlyingKernel::kStrideSupport;
  static cutlass::conv::GroupMode const kGroupMode = UnderlyingKernel::kGroupMode;

  static int const kWarpCount = 
    (ThreadblockShape::kM / WarpShape::kM) * 
    (ThreadblockShape::kN / WarpShape::kN) *
    (ThreadblockShape::kK / WarpShape::kK);

  /// Argument structure
  using Arguments = typename UnderlyingKernel::Arguments;

  using ReorderKernel = typename UnderlyingKernel::ReorderKernel;

 private:

  /// Kernel parameters object
  typename UnderlyingKernel::Params params_;

public:

  /// Constructs Implicit GEMM
  DirectConvolution() { }

  /// Determines whether the Implicit GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    // dispatch to iterators
    Status status = UnderlyingKernel::Mma::IteratorA::can_implement(args.problem_size);
    if (Status::kSuccess != status) {
      return status;
    }

    status = UnderlyingKernel::Mma::IteratorB::can_implement(args.problem_size);
    if (Status::kSuccess != status) {
      return status;
    }

    if (kGroupMode != conv::GroupMode::kDepthwise) {
      return Status::kErrorInvalidProblem;
    }

    // C and K should be multiple of groups
    if (args.problem_size.K != args.problem_size.groups &&
      args.problem_size.C != args.problem_size.groups) {
      return Status::kErrorInvalidProblem;
    }
    

    static int const kAlignmentC = UnderlyingKernel::Epilogue::OutputTileIterator::kElementsPerAccess;
    if (kConvolutionalOperator == conv::Operator::kFprop) {
      if (args.problem_size.K % kAlignmentC)
        return Status::kErrorMisalignedOperand;
    } else if (kConvolutionalOperator == conv::Operator::kDgrad) {
       if (args.problem_size.C % kAlignmentC)
        return Status::kErrorMisalignedOperand;
    } else if (kConvolutionalOperator == conv::Operator::kWgrad) {
       if (args.problem_size.C % kAlignmentC)
        return Status::kErrorMisalignedOperand;
    }

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(
      threadblock_swizzle.get_tiled_shape(
        kConvolutionalOperator,
        args.problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices));

    if (!(grid.y <= std::numeric_limits<uint16_t>::max() &&
          grid.z <= std::numeric_limits<uint16_t>::max())) {

      return Status::kErrorInvalidProblem;
    }

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {  
    return 0;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr) {
    
    // initialize the params structure from the arguments
    params_ = typename UnderlyingKernel::Params(
    	args,
    	static_cast<int *>(workspace)
    );
    
    int smem_size = int(sizeof(typename UnderlyingKernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      cudaError_t result = cudaFuncSetAttribute(cutlass::Kernel<UnderlyingKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }
    
    return Status::kSuccess;
  }

  /// Initializes GEMM state from arguments.
  Status update(Arguments const &args, void *workspace = nullptr) {

    // update the params structure from the arguments
    params_.ptr_A = args.ref_A.data();
    params_.ptr_B = args.ref_B.data();
    params_.ptr_C = args.ref_C.data();
    params_.ptr_D = args.ref_D.data();
    params_.output_op = args.output_op;
    params_.ptr_reordered_B = args.ref_reordered_B.data();
    params_.semaphore = static_cast<int *>(workspace);

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    // Launch reorder kernel
    if (params_.ptr_reordered_B != nullptr) {
      dim3 grid = ReorderKernel::get_grid_shape(params_);
      dim3 block = ReorderKernel::get_block_shape();

      cutlass::arch::synclog_setup();
      cutlass::Kernel<ReorderKernel><<<grid, block, 0, stream>>>(params_);
    }

    // Launch main kernel
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(32 * kWarpCount, 1, 1);

    // Dynamic SMEM size based on input params.
    int smem_size = int(params_.get_smem_size());

    // Make sure we can use that much shared memory.
    cudaError_t status = 
        cudaFuncSetAttribute(cutlass::Kernel<UnderlyingKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (status != cudaSuccess)
      return Status::kErrorInternal;

    cutlass::arch::synclog_setup();
    cutlass::Kernel<UnderlyingKernel><<<grid, block, smem_size, stream>>>(params_);

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
    
    Status status = initialize(args, workspace, stream);
    
    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }

  int get_smem_size() { return int(params_.get_smem_size()); }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}
}
}

/////////////////////////////////////////////////////////////////////////////////////////////////
