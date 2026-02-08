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
   \brief Template for device-level Implicit GEMM
*/

#pragma once

#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/conv/convolution.h"

#include "kernel/b2b_implicit_gemm_convolution.h"
#include "kernel/default_b2b_conv2d_fprop.h"
#include "kernel/default_b2b_conv2d_fprop_sm75.h"
#include "kernel/default_b2b_conv2d_fprop_sm80.h"
#include "kernel/default_b2b_conv2d_fprop_smem_accumulator_sm75.h"
#include "kernel/default_b2b_conv2d_fprop_smem_accumulator_sm80.h"

namespace cutlass {
namespace conv {
namespace device {

template<typename B2bImplicitGemmKernel_>
class B2bImplicitGemmConvolution {
public:

  using B2bImplicitGemmKernel = B2bImplicitGemmKernel_;

  using ElementA = typename B2bImplicitGemmKernel::ElementA;
  using LayoutA = typename B2bImplicitGemmKernel::LayoutA;
  using ElementB = typename B2bImplicitGemmKernel::ElementB;
  using LayoutB = typename B2bImplicitGemmKernel::LayoutB;
  using ElementC = typename B2bImplicitGemmKernel::ElementC;
  using LayoutC = typename B2bImplicitGemmKernel::LayoutC;
  using ElementAccumulator = typename B2bImplicitGemmKernel::ElementAccumulator;
  using ElementCompute = typename B2bImplicitGemmKernel::ElementCompute;
  using ElementScaleBias = typename B2bImplicitGemmKernel::ElementScaleBias;
  using LayoutScaleBias = typename B2bImplicitGemmKernel::LayoutScaleBias;
  using OperatorClass = typename B2bImplicitGemmKernel::OperatorClass;
  using ArchTag = typename B2bImplicitGemmKernel::ArchTag;
  using ThreadblockShape0 = typename B2bImplicitGemmKernel::ThreadblockShape0;
  using ThreadblockShape1 = typename B2bImplicitGemmKernel::ThreadblockShape1;
  using WarpShape0 = typename B2bImplicitGemmKernel::WarpShape0;
  using WarpShape1 = typename B2bImplicitGemmKernel::WarpShape1;
  using InstructionShape = typename B2bImplicitGemmKernel::InstructionShape;
  using ThreadblockSwizzle = typename B2bImplicitGemmKernel::ThreadblockSwizzle;
  using EpilogueOutputOp0 = typename B2bImplicitGemmKernel::EpilogueOutputOp0;
  using EpilogueOutputOp1 = typename B2bImplicitGemmKernel::EpilogueOutputOp1;
  static int const kStages = B2bImplicitGemmKernel::kStages;
  static int const kConvDim = B2bImplicitGemmKernel::kConvDim;
  using WarpMmaOperator0 = typename B2bImplicitGemmKernel::WarpMmaOperator0;
  using WarpMmaOperator1 = typename B2bImplicitGemmKernel::WarpMmaOperator1;
  using ArchMmaOperator = typename B2bImplicitGemmKernel::ArchMmaOperator;
  using MathOperator = typename B2bImplicitGemmKernel::MathOperator; 

  static cutlass::conv::Operator const kConvolutionalOperator = B2bImplicitGemmKernel::kConvolutionalOperator;
  static cutlass::conv::IteratorAlgorithm const kIteratorAlgorithm = B2bImplicitGemmKernel::kIteratorAlgorithm;

  static int const kWarpCount = 
    (ThreadblockShape0::kM / WarpShape0::kM) * 
    (ThreadblockShape0::kN / WarpShape0::kN);

  /// Argument structure
  using Arguments = typename B2bImplicitGemmKernel::Arguments;

private:

  /// Kernel parameters object
  typename B2bImplicitGemmKernel::Params params_;

public:

  /// Constructs Implicit GEMM
  B2bImplicitGemmConvolution() { }

  /// Determines whether the Implicit GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    // dispatch to iterators
    Status status = B2bImplicitGemmKernel::B2bMma::IteratorA0::can_implement(args.problem_size_0);
    if (Status::kSuccess != status) {
      return status;
    }

    status = B2bImplicitGemmKernel::B2bMma::IteratorB0::can_implement(args.problem_size_0);
    if (Status::kSuccess != status) {
      return status;
    }

    status = B2bImplicitGemmKernel::B2bMma::IteratorB1::can_implement(args.problem_size_1);
    if (Status::kSuccess != status) {
      return status;
    }

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(
      threadblock_swizzle.get_tiled_shape(
        cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size_0),
        {ThreadblockShape0::kM, ThreadblockShape0::kN, ThreadblockShape0::kK},
        args.problem_size_0.split_k_slices));

    if (!(grid.y <= std::numeric_limits<uint16_t>::max() &&
          grid.z <= std::numeric_limits<uint16_t>::max())) {

      return Status::kErrorInvalidProblem;
    }

    // Determine if fusion sizes are valid

    cutlass::gemm::GemmCoord problem_size_0 = implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size_0);
    cutlass::gemm::GemmCoord problem_size_1 = implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size_1);

    if(problem_size_0.m() != problem_size_1.m())
      return Status::kErrorInvalidProblem;

    if(problem_size_0.n() != problem_size_1.k())
      return Status::kErrorInvalidProblem;

    if(args.problem_size_1.R != 1 || args.problem_size_1.S != 1)
      return Status::kErrorInvalidProblem;

    if(problem_size_0.n() > ThreadblockShape0::kN)
      return Status::kErrorInvalidProblem;
    
    if(problem_size_1.n() > ThreadblockShape1::kN)
      return Status::kErrorInvalidProblem;

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
  
    size_t workspace_bytes = 0;

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size_0),
        {ThreadblockShape0::kM, ThreadblockShape0::kN, ThreadblockShape0::kK},
        args.problem_size_0.split_k_slices);

    if(args.split_k_mode == SplitKMode::kParallel) {

      // Split-K parallel: CTAs in k-dimension write the partial results in a temporary workspace.
      // The user needs to call a reduction operator to obtain the final output tensor
      workspace_bytes = 
        sizeof(ElementAccumulator) *
        size_t(cutlass::conv::implicit_gemm_tensor_c_size(kConvolutionalOperator, args.problem_size_0)) *
        size_t(grid_tiled_shape.k());
    }

    else if(args.split_k_mode == SplitKMode::kSerial && args.problem_size_0.split_k_slices > 1) {

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
   
    if (args.problem_size_0.split_k_slices > 1) {

      if (!workspace) {
        return Status::kErrorWorkspaceNull;
      }

      cudaError_t status = cudaMemsetAsync(workspace, 0, get_workspace_size(args), stream);

      if (status != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    // initialize the params structure from the arguments
    params_ = typename B2bImplicitGemmKernel::Params(
    	args,
    	static_cast<int *>(workspace)
    );
    
    int smem_size = int(sizeof(typename B2bImplicitGemmKernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      cudaError_t result = cudaFuncSetAttribute(cutlass::Kernel<B2bImplicitGemmKernel>,
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
    params_.ptr_A0 = args.ref_A0.data();
    params_.ptr_B0 = args.ref_B0.data();
    params_.ptr_C0 = args.ref_C0.data();
    params_.ptr_Scale0 = args.ref_Scale0.data();
    params_.ptr_Bias0 = args.ref_Bias0.data();
    params_.ptr_B1 = args.ref_B1.data();
    params_.ptr_C1 = args.ref_C1.data();
    params_.ptr_D1 = args.ref_D1.data();
    params_.output_op_0 = args.output_op_0;
    params_.output_op_1 = args.output_op_1;
    params_.semaphore = static_cast<int *>(workspace);

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(32 * kWarpCount, 1, 1);

    int smem_size = int(sizeof(typename B2bImplicitGemmKernel::SharedStorage));

    cutlass::Kernel<B2bImplicitGemmKernel><<<grid, block, smem_size, stream>>>(params_);

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

/////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace device
} // namespace conv
} // namespace cutlass
/////////////////////////////////////////////////////////////////////////////////////////////////
