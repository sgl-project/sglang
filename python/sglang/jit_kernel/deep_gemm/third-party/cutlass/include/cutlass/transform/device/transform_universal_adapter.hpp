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
  \brief Transform Kernel Universal adapter
*/

#pragma once

// common
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/detail/mma.hpp"
#include "cutlass/cuda_host_adapter.hpp"

#include "cutlass/kernel_launch.h"
#if !defined(__CUDACC_RTC__)
#include "cutlass/cluster_launch.hpp"
#include "cutlass/trace.h"
#endif // !defined(__CUDACC_RTC__)


////////////////////////////////////////////////////////////////////////////////

namespace cutlass::transform::device {

////////////////////////////////////////////////////////////////////////////////

template <class TransformKernel_>
class TransformUniversalAdapter
{
public:
  using TransformKernel = GetUnderlyingKernel_t<TransformKernel_>;
  using Arguments = typename TransformKernel::Arguments;
  using Params = typename TransformKernel::Params;
  static bool const kEnableCudaHostAdapter = CUTLASS_ENABLE_CUDA_HOST_ADAPTER;


private:

  /// Kernel API parameters object
  Params params_;

public:

  /// Access the Params structure
  Params const& params() const {
    return params_;
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status
  can_implement(Arguments const& args) {
    return TransformKernel::can_implement(args);
  }

  /// Gets the workspace size
  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;
    workspace_bytes += TransformKernel::get_workspace_size(args);

    CUTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);

    return workspace_bytes;
  }

  /// Computes the grid shape
  static dim3
  get_grid_shape(Arguments const& args, void* workspace = nullptr) {
    auto tmp_params = TransformKernel::to_underlying_arguments(args, workspace);
    return TransformKernel::get_grid_shape(tmp_params);
  }

  /// Computes the grid shape
  static dim3
  get_grid_shape(Params const& params) {
    return TransformKernel::get_grid_shape(params);
  }


  /// Initializes GEMM state from arguments.
  Status
  initialize(
    Arguments const& args,
    void* workspace = nullptr,
    cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {

    CUTLASS_TRACE_HOST("TransformUniversalAdapter::initialize() - workspace "
      << workspace << ", stream: " << (stream ? "non-null" : "null")
      << ", EnableCudaHostAdapter: " << (kEnableCudaHostAdapter ? "True" : "false"));

    // Initialize the workspace
    Status status = TransformKernel::initialize_workspace(args, workspace, stream, cuda_adapter);
    if (status != Status::kSuccess) {
      return status;
    }
    // Initialize the Params structure
    params_ = TransformKernel::to_underlying_arguments(args, workspace);
    // Don't set the function attributes - require the CudaHostAdapter to set it.
    if constexpr (kEnableCudaHostAdapter) {
      CUTLASS_ASSERT(cuda_adapter);
      return Status::kSuccess;
    }
    else {
      //
      // Account for dynamic smem capacity if needed
      //
      int smem_size = TransformKernel::SharedStorageSize;

      CUTLASS_ASSERT(cuda_adapter == nullptr);

      if (smem_size >= (48 << 10)) {
        CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
        cudaError_t result = cudaFuncSetAttribute(
            device_kernel<TransformKernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
        if (cudaSuccess != result) {
          result = cudaGetLastError(); // to clear the error bit
          CUTLASS_TRACE_HOST("  cudaFuncSetAttribute() returned error: " << cudaGetErrorString(result));
          return Status::kErrorInternal;
        }
      }
    }
    return Status::kSuccess;
  }

  static Status
  run(Params& params,
      cudaStream_t stream = nullptr,
      CudaHostAdapter *cuda_adapter = nullptr,
      int32_t kernel_index = 0,
      bool launch_with_pdl = false) {
    CUTLASS_TRACE_HOST("TransformUniversalAdapter::run()");
    dim3 const block = TransformKernel::get_block_shape();
    dim3 const grid = get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = TransformKernel::SharedStorageSize;

    Status launch_result{ Status::kSuccess };
    // Use extended launch API only for mainloops that use it
    if constexpr (TransformKernel::ArchTag::kMinComputeCapability >= 90) {
      // Currently only support 1x1x1 for transform kernel.
      dim3 const cluster = {1,1,1};
      void* kernel_params[] = {&params};

      if constexpr (kEnableCudaHostAdapter) {
        //
        // Use the cuda host adapter
        //
        CUTLASS_ASSERT(cuda_adapter);
        if (cuda_adapter) {

          if (launch_with_pdl) {
            CUTLASS_TRACE_HOST(
              "TransformUniversalAdapter::run() does not support launching with PDL and a custom cuda adapter.");
            return Status::kErrorInternal;
          }
          launch_result = cuda_adapter->launch(grid,
                                               cluster,
                                               block,
                                               smem_size,
                                               stream,
                                               kernel_params,
                                               kernel_index);
          CUTLASS_TRACE_HOST("Kernel Launch Result" << cutlassGetStatusString(launch_result));
        }
        else {
          return Status::kErrorInternal;
        }
      }
      else {
        CUTLASS_ASSERT(cuda_adapter == nullptr);
        void const* kernel = (void const*) device_kernel<TransformKernel>;
        if constexpr (TransformKernel::ArchTag::kMinComputeCapability == 90) {
          launch_result = ClusterLauncher::launch(
            grid, cluster, block, smem_size, stream, kernel, kernel_params, launch_with_pdl);
        }
      }
    }
    else {
      launch_result = Status::kSuccess;
      cutlass::arch::synclog_setup();

      if constexpr (kEnableCudaHostAdapter) {
        CUTLASS_ASSERT(cuda_adapter);
        if (cuda_adapter) {
          void* kernel_params[] = {&params};

          launch_result = cuda_adapter->launch(
            grid, block, smem_size, stream, kernel_params, 0
          );

        }
        else {
          return Status::kErrorInternal;
        }
      }
      else {
        CUTLASS_ASSERT(cuda_adapter == nullptr);
        cutlass::kernel_launch<TransformKernel>(grid, block, smem_size, stream, params, launch_with_pdl);
      }
    }

    cudaError_t result = cudaGetLastError();
    if (cudaSuccess == result && Status::kSuccess == launch_result) {
      return Status::kSuccess;
    }
    else if (cudaSuccess != result) {
      CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << cudaGetErrorString(result));
    }
    else if (Status::kSuccess != launch_result) {
      CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << cutlassGetStatusString(launch_result));
    }
    return Status::kErrorInternal;
  }

  //
  // Non-static launch overloads that first create and set the internal params struct of this kernel handle.
  //

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  Status
  run(
    Arguments const& args,
    void* workspace = nullptr,
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr,
    int32_t kernel_index = 0,
    bool launch_with_pdl = false
  ) {
    Status status = initialize(args, workspace, stream, cuda_adapter);

    if (Status::kSuccess == status) {
      status = run(params_, stream, cuda_adapter, kernel_index, launch_with_pdl);
    }
    return status;
  }

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  Status
  operator()(
    Arguments const& args,
    void* workspace = nullptr,
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr,
    bool launch_with_pdl = false) {
    return run(args, workspace, stream, cuda_adapter, 0 /*kernel_index*/, launch_with_pdl);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  run(
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr,
    bool launch_with_pdl = false) {
    return run(params_, stream, cuda_adapter, 0 /*kernel_index*/, launch_with_pdl);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  operator()(cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr, bool launch_with_pdl = false) {
    return run(params_, stream, cuda_adapter, 0 /*kernel_index*/, launch_with_pdl);
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::transform::device

////////////////////////////////////////////////////////////////////////////////
