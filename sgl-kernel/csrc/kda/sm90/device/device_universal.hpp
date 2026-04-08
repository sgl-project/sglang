// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// common
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>

#if !defined(__CUDACC_RTC__)
#include <cutlass/trace.h>

#include <cutlass/cluster_launch.hpp>
#endif  // !defined(__CUDACC_RTC__)

namespace cutlass::device {

template <class Kernel_>
class Universal {
 public:
  using Kernel = Kernel_;

  static int const kThreadCount = Kernel::MaxThreadsPerBlock;

  /// Argument structure: User API
  using Arguments = typename Kernel::Arguments;
  /// Argument structure: Kernel API
  using Params = typename Kernel::Params;

 private:
  /// Kernel API parameters object
  Params params_;

 public:
  /// Access the Params structure
  Params const& params() const {
    return params_;
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const& args) {
    if (Kernel::can_implement(args)) {
      return Status::kSuccess;
    } else {
      return Status::kInvalid;
    }
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;
    workspace_bytes += Kernel::get_workspace_size(args);
    return workspace_bytes;
  }

  /// Computes the grid shape
  static dim3 get_grid_shape(Params const& params) {
    return Kernel::get_grid_shape(params);
  }

  /// Computes the maximum number of active blocks per multiprocessor
  static int maximum_active_blocks(int /* smem_capacity */ = -1) {
    CUTLASS_TRACE_HOST("Universal::maximum_active_blocks()");
    int max_active_blocks = -1;
    int smem_size = Kernel::SharedStorageSize;

    // first, account for dynamic smem capacity if needed
    cudaError_t result;
    if (smem_size >= (48 << 10)) {
      CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
      result = cudaFuncSetAttribute(device_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
      if (cudaSuccess != result) {
        result = cudaGetLastError();  // to clear the error bit
        CUTLASS_TRACE_HOST("  cudaFuncSetAttribute() returned error: " << cudaGetErrorString(result));
        return -1;
      }
    }

    // query occupancy after setting smem size
    result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, device_kernel<Kernel>, Kernel::MaxThreadsPerBlock, smem_size);

    if (cudaSuccess != result) {
      result = cudaGetLastError();  // to clear the error bit
      CUTLASS_TRACE_HOST(
          "  cudaOccupancyMaxActiveBlocksPerMultiprocessor() returned error: " << cudaGetErrorString(result));
      return -1;
    }

    CUTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
    return max_active_blocks;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST(
        "Universal::initialize() - workspace " << workspace << ", stream: " << (stream ? "non-null" : "null"));

    // Initialize the workspace
    Status status = Kernel::initialize_workspace(args, workspace, stream);
    if (status != Status::kSuccess) {
      return status;
    }

    // Initialize the Params structure
    params_ = Kernel::to_underlying_arguments(args, workspace);

    // account for dynamic smem capacity if needed
    int smem_size = Kernel::SharedStorageSize;
    if (smem_size >= (48 << 10)) {
      CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
      cudaError_t result =
          cudaFuncSetAttribute(device_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
      if (cudaSuccess != result) {
        result = cudaGetLastError();  // to clear the error bit
        CUTLASS_TRACE_HOST("  cudaFuncSetAttribute() returned error: " << cudaGetErrorString(result));
        return Status::kErrorInternal;
      }
    }

    return Status::kSuccess;
  }

  /// Update API is preserved in 3.0, but does not guarantee a lightweight update of params.
  Status update(Arguments const& args, void* workspace = nullptr) {
    CUTLASS_TRACE_HOST("Universal()::update() - workspace: " << workspace);

    size_t workspace_bytes = get_workspace_size(args);
    if (workspace_bytes > 0 && nullptr == workspace) {
      return Status::kErrorWorkspaceNull;
    }

    params_ = Kernel::to_underlying_arguments(args, workspace);
    return Status::kSuccess;
  }

  /// Primary run() entry point API that is static allowing users to create and manage their own params.
  /// Supplied params struct must be construct by calling Kernel::to_underling_arguments()
  static Status run(Params& params, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("Universal::run()");
    dim3 const block = Kernel::get_block_shape();
    dim3 const grid = get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = Kernel::SharedStorageSize;

    Status launch_result;
    // Use extended launch API only for mainloops that use it
    if constexpr (Kernel::ArchTag::kMinComputeCapability >= 90) {
      dim3 cluster(
          cute::size<0>(typename Kernel::ClusterShape{}),
          cute::size<1>(typename Kernel::ClusterShape{}),
          cute::size<2>(typename Kernel::ClusterShape{}));
      void const* kernel = (void const*)device_kernel<Kernel>;
      void* kernel_params[] = {&params};
      launch_result = ClusterLauncher::launch(grid, cluster, block, smem_size, stream, kernel, kernel_params);
    } else {
      launch_result = Status::kSuccess;
      cutlass::arch::synclog_setup();
      device_kernel<Kernel><<<grid, block, smem_size, stream>>>(params);
    }

    cudaError_t result = cudaGetLastError();
    if (cudaSuccess == result && Status::kSuccess == launch_result) {
      return Status::kSuccess;
    } else {
      CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
      return Status::kErrorInternal;
    }
  }

  //
  // Non-static launch overloads that first create and set the internal params struct of this kernel handle.
  //

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  Status run(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    Status status = initialize(args, workspace, stream);
    if (Status::kSuccess == status) {
      status = run(params_, stream);
    }
    return status;
  }

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  Status operator()(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    return run(args, workspace, stream);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status run(cudaStream_t stream = nullptr) {
    return run(params_, stream);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(params_, stream);
  }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::device

////////////////////////////////////////////////////////////////////////////////
