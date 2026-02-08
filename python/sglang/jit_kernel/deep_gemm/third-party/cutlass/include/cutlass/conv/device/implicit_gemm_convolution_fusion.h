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
   \brief Template for device-level fused activation's scale+bias+relu and Implicit GEMM Convolution
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

template<typename ImplicitGemmFusionKernel_>
class ImplicitGemmConvolutionFusion {
public:

  using ImplicitGemmFusionKernel = ImplicitGemmFusionKernel_;

  using ElementA = typename ImplicitGemmFusionKernel::ElementA;
  using LayoutA = typename ImplicitGemmFusionKernel::LayoutA;
  using ElementB = typename ImplicitGemmFusionKernel::ElementB;
  using LayoutB = typename ImplicitGemmFusionKernel::LayoutB;

//  using ElementScaleBias = typename ImplicitGemmFusionKernel::ElementScaleBias;
//  using LayoutScaleBias = typename ImplicitGemmFusionKernel::LayoutScaleBias;

  using ElementC = typename ImplicitGemmFusionKernel::ElementC;
  using LayoutC = typename ImplicitGemmFusionKernel::LayoutC;
  using ElementAccumulator = typename ImplicitGemmFusionKernel::ElementAccumulator;
  using ElementCompute = typename ImplicitGemmFusionKernel::ElementCompute;
  using OperatorClass = typename ImplicitGemmFusionKernel::OperatorClass;
  using ArchTag = typename ImplicitGemmFusionKernel::ArchTag;
  using ThreadblockShape = typename ImplicitGemmFusionKernel::ThreadblockShape;
  using WarpShape = typename ImplicitGemmFusionKernel::WarpShape;
  using InstructionShape = typename ImplicitGemmFusionKernel::InstructionShape;
  using ThreadblockSwizzle = typename ImplicitGemmFusionKernel::ThreadblockSwizzle;
  using EpilogueOutputOp = typename ImplicitGemmFusionKernel::EpilogueOutputOp;
  static int const kStages = ImplicitGemmFusionKernel::kStages;
  static int const kConvDim = ImplicitGemmFusionKernel::kConvDim;
  using WarpMmaOperator = typename ImplicitGemmFusionKernel::WarpMmaOperator;
  using ArchMmaOperator = typename ImplicitGemmFusionKernel::ArchMmaOperator;
  using MathOperator = typename ImplicitGemmFusionKernel::MathOperator; 

  static cutlass::conv::Operator const kConvolutionalOperator = ImplicitGemmFusionKernel::kConvolutionalOperator;
  static cutlass::conv::IteratorAlgorithm const kIteratorAlgorithm = ImplicitGemmFusionKernel::kIteratorAlgorithm;

  static int const kWarpCount = 
    (ThreadblockShape::kM / WarpShape::kM) * 
    (ThreadblockShape::kN / WarpShape::kN) *
    (ThreadblockShape::kK / WarpShape::kK);

  /// Argument structure
  using Arguments = typename ImplicitGemmFusionKernel::Arguments;

private:

  /// Kernel parameters object
  typename ImplicitGemmFusionKernel::Params params_;

public:

  /// Constructs Implicit GEMM
  ImplicitGemmConvolutionFusion() { }

  /// Determines whether the Implicit GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    // dispatch to iterators
    Status status = ImplicitGemmFusionKernel::Mma::IteratorA::can_implement(args.problem_size);
    if (Status::kSuccess != status) {
      return status;
    }

    status = ImplicitGemmFusionKernel::Mma::IteratorB::can_implement(args.problem_size);
    if (Status::kSuccess != status) {
      return status;
    }

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(
      threadblock_swizzle.get_tiled_shape(
        cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size),
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
  
    size_t workspace_bytes = 0;

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size),
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices);

    if(args.split_k_mode == SplitKMode::kParallel) {

      // Split-K parallel: CTAs in k-dimension write the partial results in a temporary workspace.
      // The user needs to call a reduction operator to optain the final output tensor
      workspace_bytes = 
        sizeof(ElementAccumulator) *
        size_t(cutlass::conv::implicit_gemm_tensor_c_size(kConvolutionalOperator, args.problem_size)) *
        size_t(grid_tiled_shape.k());
    }

    else if(args.split_k_mode == SplitKMode::kSerial && args.problem_size.split_k_slices > 1) {

      // Split-K serial: The user workspace is used to store semaphore and serialize writing the 
      // final reduced output to user's output tensor
      workspace_bytes = sizeof(int) * size_t(grid_tiled_shape.m()) * size_t(grid_tiled_shape.n());
    }

    return workspace_bytes;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr) {
   
    if (args.problem_size.split_k_slices > 1) {

      if (!workspace) {
        return Status::kErrorWorkspaceNull;
      }

      cudaError_t status = cudaMemsetAsync(workspace, 0, get_workspace_size(args), stream);

      if (status != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    // initialize the params structure from the arguments
    params_ = typename ImplicitGemmFusionKernel::Params(
    	args,
    	static_cast<int *>(workspace)
    );
    
    int smem_size = int(sizeof(typename ImplicitGemmFusionKernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      cudaError_t result = cudaFuncSetAttribute(cutlass::Kernel<ImplicitGemmFusionKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }
    
    return Status::kSuccess;
  }

  /// Initializes Impicit GEMM state from arguments.
  Status update(Arguments const &args, void *workspace = nullptr) {

    // update the params structure from the arguments
    params_.ptr_A = args.ref_A.data();
    params_.ptr_B = args.ref_B.data();
    params_.ptr_scale = args.ref_A_scale.data();
    params_.ptr_bias = args.ref_A_bias.data();
    params_.ptr_C = args.ref_C.data();
    params_.ptr_D = args.ref_D.data();
    params_.output_op = args.output_op;
    params_.semaphore = static_cast<int *>(workspace);

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(32 * kWarpCount, 1, 1);

    int smem_size = int(sizeof(typename ImplicitGemmFusionKernel::SharedStorage));

    cutlass::arch::synclog_setup();
    cutlass::Kernel<ImplicitGemmFusionKernel><<<grid, block, smem_size, stream>>>(params_);

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
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}
}
}

/////////////////////////////////////////////////////////////////////////////////////////////////
