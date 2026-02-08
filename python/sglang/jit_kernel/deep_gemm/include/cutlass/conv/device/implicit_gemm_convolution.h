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
   \brief Template for device-level Implicit GEMM Convolution
*/

#pragma once

#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/cuda_host_adapter.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ImplicitGemmKernel_>
class ImplicitGemmConvolution {
public:

  using UnderlyingKernel = GetUnderlyingKernel_t<ImplicitGemmKernel_>;

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

  static bool const kEnableCudaHostAdapter = CUTLASS_ENABLE_CUDA_HOST_ADAPTER;

  static int const kWarpCount = 
    (ThreadblockShape::kM / WarpShape::kM) * 
    (ThreadblockShape::kN / WarpShape::kN) *
    (ThreadblockShape::kK / WarpShape::kK);

  /// Argument structure
  using Arguments = typename UnderlyingKernel::Arguments;

private:

  /// Kernel parameters object
  typename UnderlyingKernel::Params params_;

public:

  /// Constructs Implicit GEMM
  ImplicitGemmConvolution() { }

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

    // Check that tensor sizes don't exceed maximum supported size
    if (kConvolutionalOperator == conv::Operator::kFprop) {
      if (args.problem_size.activation_size() * sizeof(ElementA) >=
              (1ull << 31) ||
          args.problem_size.filter_size() * sizeof(ElementB) >= (1ull << 31) ||
          args.problem_size.output_size() * sizeof(ElementC) >= (1ull << 31)) {
        return Status::kErrorInvalidProblem;
      }
    }
    else if (kConvolutionalOperator == conv::Operator::kDgrad ||
               kConvolutionalOperator == conv::Operator::kDeconv) {
      if (args.problem_size.activation_size() * sizeof(ElementC) >=
              (1ull << 31) ||
          args.problem_size.filter_size() * sizeof(ElementB) >= (1ull << 31) ||
          args.problem_size.output_size() * sizeof(ElementA) >= (1ull << 31)) {
        return Status::kErrorInvalidProblem;
      }
    }
    else if (kConvolutionalOperator == conv::Operator::kWgrad) {
      if (args.problem_size.activation_size() * sizeof(ElementB) >=
              (1ull << 31) ||
          args.problem_size.filter_size() * sizeof(ElementC) >= (1ull << 31) ||
          args.problem_size.output_size() * sizeof(ElementA) >= (1ull << 31)) {
        return Status::kErrorInvalidProblem;
      }
    }

    // check group conv constraint
    if (args.problem_size.groups != 1) {
      if (kGroupMode == conv::GroupMode::kNone) {
        return Status::kErrorInvalidProblem;
      } 

      // C and K should be multiple of groups
      if (args.problem_size.K % args.problem_size.groups ||
        args.problem_size.C % args.problem_size.groups) {
        return Status::kErrorInvalidProblem;
      }

      // split-k is not supported
      if (args.problem_size.split_k_slices != 1) {
        return Status::kErrorInvalidProblem;
      }

      int k_per_group = args.problem_size.K / args.problem_size.groups;
      // k_per_group should be multiple of ThreadblockShape N, one CTA calculate one group
      if (kGroupMode == conv::GroupMode::kSingleGroup && k_per_group % ThreadblockShape::kN) {
        return Status::kErrorInvalidProblem;
      }
      // ThreadblockShape::kN should be divisible by k_per_group, one CTA calculate multiple groups
      if (kGroupMode == conv::GroupMode::kMultipleGroup && ThreadblockShape::kN % k_per_group) {
        return Status::kErrorInvalidProblem;
      }

      // current optimized iterator algo only supports SingleGroup mode
      if (kIteratorAlgorithm == IteratorAlgorithm::kOptimized &&
        kGroupMode != conv::GroupMode::kSingleGroup) {
        return Status::kErrorInvalidProblem;
      }
    }

    static int const kAlignmentC = UnderlyingKernel::Epilogue::OutputTileIterator::kElementsPerAccess;
    if (kConvolutionalOperator == conv::Operator::kFprop) {
      if (args.problem_size.K % kAlignmentC)
        return Status::kErrorMisalignedOperand;
    } else if (kConvolutionalOperator == conv::Operator::kDgrad || kConvolutionalOperator == conv::Operator::kDeconv) {
       if (args.problem_size.C % kAlignmentC)
        return Status::kErrorMisalignedOperand;
    } else if (kConvolutionalOperator == conv::Operator::kWgrad) {
       if (args.problem_size.C % kAlignmentC)
        return Status::kErrorMisalignedOperand;
    }

    // check for unsupported problem sizes for strided dgrad / deconv implementation
    if ((kConvolutionalOperator == conv::Operator::kDgrad || kConvolutionalOperator == conv::Operator::kDeconv) &&
      kStrideSupport == conv::StrideSupport::kStrided) {
      // split-k (serial or parallel) is not supported for strided dgrad / deconv
      if(args.problem_size.split_k_slices > 1 && (args.problem_size.stride().at(args.problem_size.stride().max_dim_index()) > 1)) {
        return Status::kErrorNotSupported;
      }

      // dilation > {1x1} is not supported for strided dgrad / deconv
      if(args.problem_size.dilation_h > 1 || args.problem_size.dilation_w > 1) {
        return Status::kErrorNotSupported;
      }
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
  
    size_t workspace_bytes = 0;

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        kConvolutionalOperator,
        args.problem_size,
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
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {
   
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
    params_ = typename UnderlyingKernel::Params(
    	args,
    	static_cast<int *>(workspace)
    );

    if constexpr (kEnableCudaHostAdapter) {
      CUTLASS_ASSERT(cuda_adapter);
      return Status::kSuccess;
    }
    else {
      int smem_size = int(sizeof(typename UnderlyingKernel::SharedStorage));
  
      if (smem_size >= (48 << 10)) {
        cudaError_t result = cudaFuncSetAttribute(cutlass::Kernel<UnderlyingKernel>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);
  
        if (result != cudaSuccess) {
          return Status::kErrorInternal;
        }
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
    params_.semaphore = static_cast<int *>(workspace);

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr, int32_t kernel_index = 0) {


    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(32 * kWarpCount, 1, 1);

    int smem_size = int(sizeof(typename UnderlyingKernel::SharedStorage));
    cutlass::Status launch_result = cutlass::Status::kSuccess ;

    if constexpr (kEnableCudaHostAdapter) {
        //
        // Use the cuda host adapter
        //
        CUTLASS_ASSERT(cuda_adapter);
        if (cuda_adapter) {

          void* kernel_params[] = {&params_};
          launch_result = cuda_adapter->launch(
              grid, dim3(1,1,1), block, smem_size, stream, kernel_params, kernel_index
              );
        }
        else {
          launch_result = Status::kErrorInternal;
        }
    }
    else {
      cutlass::arch::synclog_setup();
      cutlass::Kernel<UnderlyingKernel><<<grid, block, smem_size, stream>>>(params_);      
    }

    cudaError_t result = cudaGetLastError();
    if (cudaSuccess == result && Status::kSuccess == launch_result) {
      return Status::kSuccess;
    }
    else {
      CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
      return Status::kErrorInternal;
    }
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr, int32_t kernel_index = 0) {
    return run(stream, cuda_adapter, kernel_index);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr, int32_t kernel_index = 0) {
    
    Status status = initialize(args, workspace, stream, cuda_adapter);
    
    if (status == Status::kSuccess) {
      status = run(stream, cuda_adapter, kernel_index);
    }

    return status;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}
}
}

/////////////////////////////////////////////////////////////////////////////////////////////////
