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
#pragma once

// common
#include "cutlass/arch/mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/arch/mma.h"
#include "cutlass/trace.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/device_kernel.h"

#include "cutlass/conv/kernel/conv_universal.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/cuda_host_adapter.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv::device {

////////////////////////////////////////////////////////////////////////////////

/*!
  ConvUniversalAdapter is a stateful, reusable handle built around a kernel
  of type cutlass::conv::kernel::ConvUniversal.

  It manages the lifetime of the underlying `kernel::Params` struct, and exposes APIs
  to create it from the host facing arguments. For power users, static methods
  are exposed that bypass the stateful methods or args->params lowering.
*/
template <class ConvKernel_>
class ConvUniversalAdapter
{
public:
  using ConvKernel = GetUnderlyingKernel_t<ConvKernel_>;
  using TileShape = typename ConvKernel::TileShape;
  using ElementA = typename ConvKernel::ElementA;
  using ElementB = typename ConvKernel::ElementB;
  using ElementC = typename ConvKernel::ElementC;
  using ElementD = typename ConvKernel::ElementD;
  using ElementAccumulator = typename ConvKernel::TiledMma::ValTypeC;
  using DispatchPolicy = typename ConvKernel::DispatchPolicy;
  using CollectiveMainloop = typename ConvKernel::CollectiveMainloop;
  using CollectiveEpilogue = typename ConvKernel::CollectiveEpilogue;

  static bool const kEnableCudaHostAdapter = CUTLASS_ENABLE_CUDA_HOST_ADAPTER;

  // Tease out meta-information about the conv algorithm
  static constexpr conv::Operator kConvolutionalOperator = DispatchPolicy::ConvOp;
  static constexpr int NumSpatialDimensions = CollectiveMainloop::NumSpatialDimensions;

  // If our TiledMMA's instruction thread layout size is larger than 1, we know its a tensorop!
  using OperatorClass = cute::conditional_t<
      (cute::size(typename ConvKernel::TiledMma::AtomThrID{}) > 1),
      cutlass::arch::OpClassTensorOp, cutlass::arch::OpClassSimt>;

  using ArchTag = typename ConvKernel::ArchTag;

  // Assume TiledMma's ShapeMNK is the same as 2.x's ThreadblockShape
  using ThreadblockShape = cutlass::gemm::GemmShape<
      cute::size<0>(TileShape{}),
      cute::size<1>(TileShape{}),
      cute::size<2>(TileShape{})>;

  using ClusterShape = cutlass::gemm::GemmShape<
      cute::size<0>(typename ConvKernel::DispatchPolicy::ClusterShape{}),
      cute::size<1>(typename ConvKernel::DispatchPolicy::ClusterShape{}),
      cute::size<2>(typename ConvKernel::DispatchPolicy::ClusterShape{})>;

  // Instruction shape is easy too, since we get that directly from our TiledMma's atom shape
  using InstructionShape = cutlass::gemm::GemmShape<
      cute::size<0>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}),
      cute::size<1>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}),
      cute::size<2>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{})>;

  // Legacy: provide a correct warp count, but no reliable warp shape
  static int const kThreadCount = ConvKernel::MaxThreadsPerBlock;

  // Warp shape is not a primary API type in 3.x
  // But we can best approximate it by inspecting the TiledMma
  // For this, we make the assumption that we always have 4 warps along M, and rest along N, none along K
  // We also always round up the warp count to 4 if the tiled mma is smaller than 128 threads
  static constexpr int WarpsInMma = cute::max(4, CUTE_STATIC_V(cute::size(typename ConvKernel::TiledMma{})) / 32);
  static constexpr int WarpsInMmaM = 4;
  static constexpr int WarpsInMmaN = cute::ceil_div(WarpsInMma, WarpsInMmaM);
  using WarpCount = cutlass::gemm::GemmShape<WarpsInMmaM, WarpsInMmaN, 1>;
  using WarpShape = cutlass::gemm::GemmShape<
      CUTE_STATIC_V(cute::tile_size<0>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaM,
      CUTE_STATIC_V(cute::tile_size<1>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaN,
      CUTE_STATIC_V(cute::tile_size<2>(typename CollectiveMainloop::TiledMma{}))>;

  static int constexpr kStages = CollectiveMainloop::DispatchPolicy::Stages;

  // Inspect TiledCopy for A and B to compute the alignment size
  static int constexpr kAlignmentA = cutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveMainloop::GmemTiledCopyA, ElementA>();
  static int constexpr kAlignmentB = cutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveMainloop::GmemTiledCopyB, ElementB>();
  static int constexpr kAlignmentC = cutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveEpilogue::GmemTiledCopyC, ElementC>();
  static int constexpr kAlignmentD = cutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveEpilogue::GmemTiledCopyD, ElementD>();

  using EpilogueOutputOp = typename CollectiveEpilogue::ThreadEpilogueOp;

  /// Argument structure: User API
  using Arguments = typename ConvKernel::Arguments;
  /// Argument structure: Kernel API
  using Params = typename ConvKernel::Params;

private:

  /// Kernel API parameters object
  Params params_;

public:

  /// Access the Params structure
  Params const& params() const {
    return params_;
  }

  /// Determines whether the conv can execute the given problem.
  static Status
  can_implement(Arguments const& args) {
    if (ConvKernel::can_implement(args)) {
      return Status::kSuccess;
    }
    else {
      return Status::kInvalid;
    }
  }

  /// Gets the workspace size
  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;
    CUTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);

    workspace_bytes += ConvKernel::get_workspace_size(args);
    return workspace_bytes;
  }

  /// Computes the grid shape
  static dim3
  get_grid_shape(Arguments const& args, void* workspace = nullptr) {
    auto tmp_params = ConvKernel::to_underlying_arguments(args, workspace);
    return ConvKernel::get_grid_shape(tmp_params);
  }

  /// Computes the grid shape
  static dim3
  get_grid_shape(Params const& params) {
    return ConvKernel::get_grid_shape(params);
  }

  /// Computes the maximum number of active blocks per multiprocessor
  static int maximum_active_blocks(int /* smem_capacity */ = -1) {
    CUTLASS_TRACE_HOST("ConvUniversal::maximum_active_blocks()");
    int max_active_blocks = -1;
    int smem_size = ConvKernel::SharedStorageSize;

    // first, account for dynamic smem capacity if needed
    cudaError_t result;
    if (smem_size >= (48 << 10)) {
      CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
      result = cudaFuncSetAttribute(
          device_kernel<ConvKernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (cudaSuccess != result) {
        result = cudaGetLastError(); // to clear the error bit
        CUTLASS_TRACE_HOST(
          "  cudaFuncSetAttribute() returned error: "
          << cudaGetErrorString(result));
        return -1;
      }
    }

    // query occupancy after setting smem size
    result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks,
        device_kernel<ConvKernel>,
        ConvKernel::MaxThreadsPerBlock,
        smem_size);

    if (cudaSuccess != result) {
      result = cudaGetLastError(); // to clear the error bit
      CUTLASS_TRACE_HOST(
        "  cudaOccupancyMaxActiveBlocksPerMultiprocessor() returned error: "
        << cudaGetErrorString(result));
      return -1;
    }

    CUTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
    return max_active_blocks;
  }

  /// Initializes conv state from arguments.
  Status
  initialize(
    Arguments const& args,
    void* workspace = nullptr,
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {

    CUTLASS_TRACE_HOST("ConvUniversal::initialize() - workspace "
      << workspace << ", stream: " << (stream ? "non-null" : "null"));

    // Initialize the workspace
    Status status = ConvKernel::initialize_workspace(args, workspace, stream, cuda_adapter);
    if (status != Status::kSuccess) {
      return status;
    }

    // Initialize the Params structure
    params_ = ConvKernel::to_underlying_arguments(args, workspace);

    // Don't set the function attributes - require the CudaHostAdapter to set it.
    if constexpr (kEnableCudaHostAdapter) {
      CUTLASS_ASSERT(cuda_adapter);
      return Status::kSuccess;
    }
    else {
      // account for dynamic smem capacity if needed
      int smem_size = ConvKernel::SharedStorageSize;
      if (smem_size >= (48 << 10)) {
        CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
        cudaError_t result = cudaFuncSetAttribute(
            device_kernel<ConvKernel>,
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

  /// Update API is preserved in 3.0, but does not guarantee a lightweight update of params.
  Status
  update(Arguments const& args, void* workspace = nullptr) {
    CUTLASS_TRACE_HOST("ConvUniversal()::update() - workspace: " << workspace);

    size_t workspace_bytes = get_workspace_size(args);
    if (workspace_bytes > 0 && nullptr == workspace) {
      return Status::kErrorWorkspaceNull;
    }

    params_ = ConvKernel::to_underlying_arguments(args, workspace);
    return Status::kSuccess;
  }

  /// Primary run() entry point API that is static allowing users to create and manage their own params.
  /// Supplied params struct must be construct by calling ConvKernel::to_underling_arguments()
  static Status
  run(Params& params, cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr, int32_t kernel_index = 0) {
    CUTLASS_TRACE_HOST("ConvUniversal::run()");
    dim3 const block = ConvKernel::get_block_shape();
    dim3 const grid = get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = ConvKernel::SharedStorageSize;

    Status launch_result;
    // Use extended launch API only for mainloops that use it
    if constexpr (ConvKernel::ArchTag::kMinComputeCapability >= 90) {
      [[maybe_unused]] constexpr bool is_static_1x1x1 =
        cute::is_static_v<typename ConvKernel::DispatchPolicy::ClusterShape> and
        cute::size(typename ConvKernel::DispatchPolicy::ClusterShape{}) == 1;
      dim3 cluster(cute::size<0>(typename ConvKernel::DispatchPolicy::ClusterShape{}),
                   cute::size<1>(typename ConvKernel::DispatchPolicy::ClusterShape{}),
                   cute::size<2>(typename ConvKernel::DispatchPolicy::ClusterShape{}));
      // Dynamic cluster support
      [[maybe_unused]] dim3 fallback_cluster = dim3{0,0,0};
      if constexpr (ConvKernel::ArchTag::kMinComputeCapability == 100 ||
                    ConvKernel::ArchTag::kMinComputeCapability == 101) {
        if constexpr (!cute::is_static_v<typename ConvKernel::DispatchPolicy::ClusterShape>) {
          fallback_cluster = params.hw_info.cluster_shape_fallback;
          cluster = params.hw_info.cluster_shape;
        }
      }

      void* kernel_params[] = {&params};
      if constexpr (kEnableCudaHostAdapter) {
        //
        // Use the cuda host adapter
        //
        CUTLASS_ASSERT(cuda_adapter);
        if (cuda_adapter) {

          launch_result = cuda_adapter->launch(grid,
                                               cluster, 
                                               fallback_cluster,
                                               block, 
                                               smem_size, 
                                               stream, 
                                               kernel_params,
                                               kernel_index);
        }
        else {
          return Status::kErrorInternal;
        }
      }
      else {
        CUTLASS_ASSERT(cuda_adapter == nullptr);
        void const* kernel = (void const*) device_kernel<ConvKernel>;
        if constexpr (ConvKernel::ArchTag::kMinComputeCapability == 90
                        || ConvKernel::ArchTag::kMinComputeCapability == 100 
                     ) {
          if constexpr (is_static_1x1x1) {
            device_kernel<ConvKernel><<<grid, block, smem_size, stream>>>(params);
            launch_result = Status::kSuccess;
          }
          else {
            launch_result = ClusterLauncher::launch(
                grid, cluster, block, smem_size, stream, kernel, kernel_params);
          }
        }
        else {
          if constexpr (ConvKernel::ArchTag::kMinComputeCapability == 100 ||
                        ConvKernel::ArchTag::kMinComputeCapability == 101) {
            launch_result = ClusterLauncher::launch_with_fallback_cluster(
              grid,
              cluster,
              fallback_cluster,
              block,
              smem_size,
              stream,
              kernel,
              kernel_params);
          }
        }
      }
    }
    else {
      launch_result = Status::kSuccess;

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
        device_kernel<ConvKernel><<<grid, block, smem_size, stream>>>(params);
      }
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
    int32_t kernel_index = 0
  ) {
    Status status = initialize(args, workspace, stream, cuda_adapter);
    if (Status::kSuccess == status) {
      status = run(params_, stream, cuda_adapter, kernel_index);
    }
    return status;
  }

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  Status
  operator()(
    Arguments const& args,
    void* workspace = nullptr,
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {
    return run(args, workspace, stream, cuda_adapter);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  run(cudaStream_t stream = nullptr) {
    return run(params_, stream);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  operator()(cudaStream_t stream = nullptr) {
    return run(params_, stream);
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv::device

////////////////////////////////////////////////////////////////////////////////
