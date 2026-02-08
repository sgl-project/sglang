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
/*!
  \file
  \brief The universal GEMM accommodates serial reductions, parallel reductions, batched strided, and
    batched array variants.
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

// 2.x
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"

// 3.x
#include "cutlass/gemm/kernel/gemm_universal.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::device {

////////////////////////////////////////////////////////////////////////////////

/*!
  GemmUniversalAdapter is a stateful, reusable GEMM handle built around a kernel
  of type cutlass::gemm::kernel::Gemm or cutlass::gemm::kernel::GemmUniversal.

  It manages the lifetime of the underlying `kernel::Params` struct, and exposes APIs
  to create it from the host facing arguments. For power users, new static methods
  are exposed in 3.x APIs that bypass the stateful methods or args->params lowering.

  It supports kernel types that implement both the 2.x and 3.0 APIs,
  however, this is done by specializing the implementation of GemmUniversalAdapter
  on the two kernel API types, and thus, GemmUniversalAdapter's behaviour might
  differ between the two specializations.
*/
template <class GemmKernel_, class Enable = void>
class GemmUniversalAdapter;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// CUTLASS 3.x API /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Work-around for some DispatchPolicy types not having a Stages member.
// In that case, the Stages value is 0.  Most code should static_assert
// that the number of stages is valid.

// Whether DispatchPolicy::Stages is valid.
// It should also be convertible to int, but if not, that will show up
// as a build error when GemmUniversalAdapter attempts to assign it to kStages.
template <class DispatchPolicy, class Enable = void>
struct has_Stages : cute::false_type {};

template <class DispatchPolicy>
struct has_Stages<DispatchPolicy, cute::void_t<decltype(DispatchPolicy::Stages)>> : cute::true_type {};

template<class DispatchPolicy>
constexpr int stages_member(DispatchPolicy) {
  if constexpr (has_Stages<DispatchPolicy>::value) {
    return DispatchPolicy::Stages;
  }
  else {
    return 0;
  }
}

} // namespace detail

template <class GemmKernel_>
class GemmUniversalAdapter<
  GemmKernel_,
  cute::enable_if_t<gemm::detail::IsCutlass3GemmKernel<GetUnderlyingKernel_t<GemmKernel_>>::value>>
{
public:
  using GemmKernel = GetUnderlyingKernel_t<GemmKernel_>;
  using TileShape = typename GemmKernel::TileShape;
  using ElementA = typename GemmKernel::ElementA;
  using ElementB = typename GemmKernel::ElementB;
  using ElementC = typename GemmKernel::ElementC;
  using ElementD = typename GemmKernel::ElementD;
  using ElementAccumulator = typename GemmKernel::ElementAccumulator;
  using DispatchPolicy = typename GemmKernel::DispatchPolicy;
  using CollectiveMainloop = typename GemmKernel::CollectiveMainloop;
  using CollectiveEpilogue = typename GemmKernel::CollectiveEpilogue;

  // Map back to 2.x type as best as possible
  using LayoutA = gemm::detail::StrideToLayoutTagA_t<typename GemmKernel::StrideA>;
  using LayoutB = gemm::detail::StrideToLayoutTagB_t<typename GemmKernel::StrideB>;
  using LayoutC = gemm::detail::StrideToLayoutTagC_t<typename GemmKernel::StrideC>;
  using LayoutD = gemm::detail::StrideToLayoutTagC_t<typename GemmKernel::StrideD>;

  static bool const kEnableCudaHostAdapter = CUTLASS_ENABLE_CUDA_HOST_ADAPTER;

  static ComplexTransform const kTransformA = cute::is_same_v<typename GemmKernel::CollectiveMainloop::TransformA, cute::conjugate> ?
                                              ComplexTransform::kConjugate : ComplexTransform::kNone;
  static ComplexTransform const kTransformB = cute::is_same_v<typename GemmKernel::CollectiveMainloop::TransformB, cute::conjugate> ?
                                              ComplexTransform::kConjugate : ComplexTransform::kNone;

  // Legacy: Assume MultiplyAdd only since we do not use this tag type in 3.0
  using MathOperator = cutlass::arch::OpMultiplyAdd;

  using OperatorClass = cutlass::detail::get_operator_class_t<typename CollectiveMainloop::TiledMma>;

  using ArchTag = typename GemmKernel::ArchTag;

  // NOTE: Assume identity swizzle for now
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  // Assume TiledMma's ShapeMNK is the same as 2.x's ThreadblockShape
  using ThreadblockShape = cutlass::gemm::GemmShape<
      cute::size<0>(TileShape{}),
      cute::size<1>(TileShape{}),
      cute::size<2>(TileShape{})>;

  using ClusterShape = cutlass::gemm::GemmShape<
      cute::size<0>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
      cute::size<1>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
      cute::size<2>(typename GemmKernel::DispatchPolicy::ClusterShape{})>;

  // Instruction shape is easy too, since we get that directly from our TiledMma's atom shape
  using InstructionShape = cutlass::gemm::GemmShape<
      cute::size<0>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}),
      cute::size<1>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}),
      cute::size<2>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{})>;

  // Legacy: provide a correct warp count, but no reliable warp shape
  static int const kThreadCount = GemmKernel::MaxThreadsPerBlock;

  // Warp shape is not a primary API type in 3.x
  // But we can best approximate it by inspecting the TiledMma
  // For this, we make the assumption that we always have 4 warps along M, and rest along N, none along K
  // We also always round up the warp count to 4 if the tiled mma is smaller than 128 threads
  static constexpr int WarpsInMma = cute::max(4, CUTE_STATIC_V(cute::size(typename GemmKernel::TiledMma{})) / 32);
  static constexpr int WarpsInMmaM = 4;
  static constexpr int WarpsInMmaN = cute::ceil_div(WarpsInMma, WarpsInMmaM);
  using WarpCount = cutlass::gemm::GemmShape<WarpsInMmaM, WarpsInMmaN, 1>;
  using WarpShape = cutlass::gemm::GemmShape<
      CUTE_STATIC_V(cute::tile_size<0>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaM,
      CUTE_STATIC_V(cute::tile_size<1>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaN,
      CUTE_STATIC_V(cute::tile_size<2>(typename CollectiveMainloop::TiledMma{}))>;

  static int constexpr kStages = detail::stages_member(typename CollectiveMainloop::DispatchPolicy{});

  // Inspect TiledCopy for A and B to compute the alignment size
  static int constexpr kAlignmentA = cutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveMainloop::GmemTiledCopyA, ElementA, typename CollectiveMainloop::TiledMma::ValTypeA>();
  static int constexpr kAlignmentB = cutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveMainloop::GmemTiledCopyB, ElementB, typename CollectiveMainloop::TiledMma::ValTypeB>();
  static int constexpr kAlignmentC = cutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveEpilogue::GmemTiledCopyC, ElementC>();
  static int constexpr kAlignmentD = cutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveEpilogue::GmemTiledCopyD, ElementD>();

  using EpilogueOutputOp = typename CollectiveEpilogue::ThreadEpilogueOp;

  // Split-K preserves splits that are 128b aligned
  static int constexpr kSplitKAlignment = cute::max(
      128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value);

  /// Argument structure: User API
  using Arguments = typename GemmKernel::Arguments;
  /// Argument structure: Kernel API
  using Params = typename GemmKernel::Params;

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
    if (GemmKernel::can_implement(args)) {
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
    if (args.mode == GemmUniversalMode::kGemmSplitKParallel) {
      workspace_bytes += sizeof(int) * size_t(cute::size<0>(TileShape{})) * size_t(cute::size<1>(TileShape{}));
    }

    workspace_bytes += GemmKernel::get_workspace_size(args);

    CUTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);

    return workspace_bytes;
  }

  /// Computes the grid shape
  static dim3
  get_grid_shape(Arguments const& args, void* workspace = nullptr) {
    auto tmp_params = GemmKernel::to_underlying_arguments(args, workspace);
    return GemmKernel::get_grid_shape(tmp_params);
  }

  /// Computes the grid shape
  static dim3
  get_grid_shape(Params const& params) {
    return GemmKernel::get_grid_shape(params);
  }

  /// Computes the maximum number of active blocks per multiprocessor
  static int maximum_active_blocks(int /* smem_capacity */ = -1) {
    CUTLASS_TRACE_HOST("GemmUniversal::maximum_active_blocks()");
    int max_active_blocks = -1;
    int smem_size = GemmKernel::SharedStorageSize;

    // first, account for dynamic smem capacity if needed
    cudaError_t result;
    if (smem_size >= (48 << 10)) {
      CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
      result = cudaFuncSetAttribute(
          device_kernel<GemmKernel>,
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
        device_kernel<GemmKernel>,
        GemmKernel::MaxThreadsPerBlock,
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

  /// Initializes GEMM state from arguments.
  Status
  initialize(
    Arguments const& args,
    void* workspace = nullptr,
    cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {

    CUTLASS_TRACE_HOST("GemmUniversal::initialize() - workspace "
      << workspace << ", stream: " << (stream ? "non-null" : "null"));

    // Initialize the workspace
    Status status = GemmKernel::initialize_workspace(args, workspace, stream, cuda_adapter);
    if (status != Status::kSuccess) {
      return status;
    }
    // Initialize the Params structure
    params_ = GemmKernel::to_underlying_arguments(args, workspace);
    // Don't set the function attributes - require the CudaHostAdapter to set it.
    if constexpr (kEnableCudaHostAdapter) {
      CUTLASS_ASSERT(cuda_adapter);
      return Status::kSuccess;
    }
    else {
      //
      // Account for dynamic smem capacity if needed
      //
      int smem_size = GemmKernel::SharedStorageSize;

      CUTLASS_ASSERT(cuda_adapter == nullptr);

      if (smem_size >= (48 << 10)) {
        CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
        cudaError_t result = cudaFuncSetAttribute(
            device_kernel<GemmKernel>,
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
    CUTLASS_TRACE_HOST("GemmUniversal()::update() - workspace: " << workspace);

    size_t workspace_bytes = get_workspace_size(args);
    if (workspace_bytes > 0 && nullptr == workspace) {
      return Status::kErrorWorkspaceNull;
    }

    params_ = GemmKernel::to_underlying_arguments(args, workspace);
    return Status::kSuccess;
  }

  /// Primary run() entry point API that is static allowing users to create and manage their own params.
  /// Supplied params struct must be construct by calling GemmKernel::to_underlying_arguments()
  static Status
  run(Params& params,
      cudaStream_t stream = nullptr,
      CudaHostAdapter *cuda_adapter = nullptr,
      bool launch_with_pdl = false) {
    CUTLASS_TRACE_HOST("GemmUniversal::run()");
    dim3 const block = GemmKernel::get_block_shape();
    dim3 const grid = get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = GemmKernel::SharedStorageSize;

    Status launch_result{ Status::kSuccess };
    // Use extended launch API only for mainloops that use it
    if constexpr (GemmKernel::ArchTag::kMinComputeCapability >= 90) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      CUTLASS_TRACE_HOST("GemmUniversal::run: Use extended launch API");
#endif
      [[maybe_unused]] constexpr bool is_static_1x1x1 =
        cute::is_static_v<typename GemmKernel::DispatchPolicy::ClusterShape> and
        cute::size(typename GemmKernel::DispatchPolicy::ClusterShape{}) == 1;
      [[maybe_unused]] dim3 cluster(cute::size<0>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
        cute::size<1>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
        cute::size<2>(typename GemmKernel::DispatchPolicy::ClusterShape{}));
      
      // Dynamic cluster support
      [[maybe_unused]] dim3 fallback_cluster = dim3{0,0,0};
      if constexpr (GemmKernel::ArchTag::kMinComputeCapability == 100 
                    || GemmKernel::ArchTag::kMinComputeCapability == 101
                    || GemmKernel::ArchTag::kMinComputeCapability == 103
                    ) {
        if constexpr (!cute::is_static_v<typename GemmKernel::DispatchPolicy::ClusterShape>) {
          fallback_cluster = params.hw_info.cluster_shape_fallback;
          cluster = params.hw_info.cluster_shape;
        }
      }
      
      [[maybe_unused]] void* kernel_params[] = {&params};

      if constexpr (kEnableCudaHostAdapter) {
        //
        // Use the cuda host adapter
        //
        CUTLASS_ASSERT(cuda_adapter);
        if (cuda_adapter) {
          if (launch_with_pdl) {
            CUTLASS_TRACE_HOST(
              "GemmUniversal::run() does not support launching with PDL and a custom cuda adapter.");
            return Status::kErrorInternal;
          }
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
          CUTLASS_TRACE_HOST("GemmUniversal::run: Launching kernel with CUDA host adapter");
#endif
          if constexpr (is_static_1x1x1) {
            launch_result = cuda_adapter->launch(grid,
                                                block,
                                                smem_size,
                                                stream,
                                                kernel_params,
                                                0);
          }
          else {
            launch_result = cuda_adapter->launch(grid,
                                                cluster,
                                                fallback_cluster, 
                                                block,
                                                smem_size,
                                                stream,
                                                kernel_params,
                                                0);
          }
        }
        else {
          CUTLASS_TRACE_HOST("GemmUniversal::run: kEnableCudaHostAdapter is true, but CUDA host adapter is null");
          return Status::kErrorInternal;
        }
      }
      else {
        CUTLASS_ASSERT(cuda_adapter == nullptr);
        [[maybe_unused]] void const* kernel = (void const*) device_kernel<GemmKernel>;
        static constexpr bool kClusterLaunch = GemmKernel::ArchTag::kMinComputeCapability == 90;
        if constexpr (kClusterLaunch) {
          if constexpr (is_static_1x1x1) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
            CUTLASS_TRACE_HOST("GemmUniversal::run: Launching static 1x1x1 kernel");
#endif
            launch_result = cutlass::kernel_launch<GemmKernel>(
              grid, block, smem_size, stream, params, launch_with_pdl);
            if (launch_result != Status::kSuccess) {
              CUTLASS_TRACE_HOST("GemmUniversal::run: cutlass::kernel_launch reports failure");
            }
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
            else {
              CUTLASS_TRACE_HOST("GemmUniversal::run: cutlass::kernel_launch reports success");
            }
#endif
          }
          else {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
            CUTLASS_TRACE_HOST("GemmUniversal::run: Launching dynamic cluster kernel");
#endif
            launch_result = ClusterLauncher::launch(
              grid, cluster, block, smem_size, stream, kernel, kernel_params, launch_with_pdl);
          }
        }
        
        else {
          if constexpr (GemmKernel::ArchTag::kMinComputeCapability == 100
                        || GemmKernel::ArchTag::kMinComputeCapability == 101
                        || GemmKernel::ArchTag::kMinComputeCapability == 120
                        || GemmKernel::ArchTag::kMinComputeCapability == 103
                       ) {
            if constexpr (is_static_1x1x1) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
              CUTLASS_TRACE_HOST("GemmUniversal::run: Launching static 1x1x1 kernel");
#endif
              launch_result = cutlass::kernel_launch<GemmKernel>(grid, block, smem_size, stream, params, launch_with_pdl);
              if (launch_result != Status::kSuccess) {
                CUTLASS_TRACE_HOST("GemmUniversal::run: cutlass::kernel_launch reports failure");
              }
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
              else {
                CUTLASS_TRACE_HOST("GemmUniversal::run: cutlass::kernel_launch reports success");
              }
#endif
            }
            else {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
              CUTLASS_TRACE_HOST("GemmUniversal::run: Launching kernel with fall-back cluster");
#endif
              launch_result = ClusterLauncher::launch_with_fallback_cluster(
                grid, 
                cluster,
                fallback_cluster,
                block,
                smem_size,
                stream,
                kernel,
                kernel_params,
                launch_with_pdl);
            }
          }
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
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
          CUTLASS_TRACE_HOST("GemmUniversal::run: Launching kernel with CUDA host adapter");
#endif
          launch_result = cuda_adapter->launch(
            grid, block, smem_size, stream, kernel_params, 0
          );

        }
        else {
          CUTLASS_TRACE_HOST("GemmUniversal::run: CUDA host adapter is null");
          return Status::kErrorInternal;
        }
      }
      else {
        CUTLASS_ASSERT(cuda_adapter == nullptr);
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
        CUTLASS_TRACE_HOST("GemmUniversal::run: Launching kernel with cutlass::kernel_launch");
#endif
        launch_result = cutlass::kernel_launch<GemmKernel>(
          grid, block, smem_size, stream, params, launch_with_pdl);
        if (launch_result != Status::kSuccess) {
          CUTLASS_TRACE_HOST("GemmUniversal::run: cutlass::kernel_launch reports failure");
        }
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
        else {
          CUTLASS_TRACE_HOST("GemmUniversal::run: cutlass::kernel_launch reports success");
        }
#endif
      }
    }

    cudaError_t result = cudaGetLastError();
    if (cudaSuccess == result && Status::kSuccess == launch_result) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      CUTLASS_TRACE_HOST("GemmUniversal::run: cudaGetLastError reports success");
#endif
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
    bool launch_with_pdl = false
  ) {
    Status status = initialize(args, workspace, stream, cuda_adapter);

    if (Status::kSuccess == status) {
      status = run(params_, stream, cuda_adapter, launch_with_pdl);
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
    return run(args, workspace, stream, cuda_adapter, launch_with_pdl);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  run(
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr,
    bool launch_with_pdl = false) {
    return run(params_, stream, cuda_adapter, launch_with_pdl);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  operator()(cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr, bool launch_with_pdl = false) {
    return run(params_, stream, cuda_adapter, launch_with_pdl);
  }
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// CUTLASS 2.x API /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <class GemmKernel_>
class GemmUniversalAdapter<
  GemmKernel_,
  cute::enable_if_t<not gemm::detail::IsCutlass3GemmKernel<GetUnderlyingKernel_t<GemmKernel_>>::value>>
{
public:

  using GemmKernel = GetUnderlyingKernel_t<GemmKernel_>;

  static bool const kInternalTranspose =
    !cutlass::epilogue::threadblock::detail::is_2x_evt_v<typename GemmKernel::Epilogue> &&  // 2.x EVT does not require internal transpose
    cute::is_same<typename GemmKernel::LayoutC, cutlass::layout::RowMajor>::value;

  using ThreadblockShape = typename GemmKernel::Mma::Shape;
  using WarpShape = typename GemmKernel::WarpShape;
  using InstructionShape = typename GemmKernel::InstructionShape;

  // warp-level, arch-level (instruction), math operator
  using WarpMmaOperator = typename GemmKernel::Mma::Policy::Operator;
  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator = typename WarpMmaOperator::MathOperator;

  // Operator class and arch tag extract bottom-up
  // set it for top-level gemm device-level template
  using OperatorClass = typename WarpMmaOperator::OperatorClass;
  using ArchTag = typename WarpMmaOperator::ArchTag;

  // Type, layout, and complex transform deliberately exchanged with B
  using MapArguments = kernel::detail::MapArguments<
    typename GemmKernel::ElementA,
    typename GemmKernel::LayoutA,
    GemmKernel::kTransformA,
    GemmKernel::kAlignmentA,
    typename GemmKernel::ElementB,
    typename GemmKernel::LayoutB,
    GemmKernel::kTransformB,
    GemmKernel::kAlignmentB,
    typename GemmKernel::LayoutC,
    kInternalTranspose
  >;

  using ElementA = typename MapArguments::ElementA;
  using LayoutA = typename MapArguments::LayoutA;
  static ComplexTransform const kTransformA = MapArguments::kTransformA;
  static int const kAlignmentA = MapArguments::kAlignmentA;

  using ElementB = typename MapArguments::ElementB;
  using LayoutB = typename MapArguments::LayoutB;
  static ComplexTransform const kTransformB = MapArguments::kTransformB;
  static int const kAlignmentB = MapArguments::kAlignmentB;

  using ElementC = typename GemmKernel::ElementC;
  using LayoutC = typename MapArguments::LayoutC;
  static int const kAlignmentC = GemmKernel::kAlignmentC;

  // C and D same type for 2.x kernel
  using ElementD = ElementC;
  using LayoutD = LayoutC;

  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementD, LayoutD>;

  static int const kStages = GemmKernel::Mma::kStages;

  using EpilogueOutputOp = typename GemmKernel::EpilogueOutputOp;
  using ElementAccumulator = typename EpilogueOutputOp::ElementAccumulator;
  using ThreadblockSwizzle = typename GemmKernel::ThreadblockSwizzle;
  using UnderlyingOperator = GemmUniversalBase<GemmKernel>;
  using Arguments = typename UnderlyingOperator::Arguments;

private:

  UnderlyingOperator underlying_operator_;

public:

  /// Constructs the GEMM.
  GemmUniversalAdapter() { }

  /// Helper to construct a transposed equivalent for the underlying GEMM operator
  static Arguments to_underlying_arguments(Arguments const &args) {
    if (kInternalTranspose) {
      return args.transposed_problem();
    }
    else {
      return args;
    }
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args, CudaHostAdapter *cuda_adapter = nullptr) {

    return UnderlyingOperator::can_implement(to_underlying_arguments(args), cuda_adapter);
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args, CudaHostAdapter *cuda_adapter = nullptr) {

    return UnderlyingOperator::get_workspace_size(to_underlying_arguments(args), cuda_adapter);
  }

  /// Computes the grid shape
  static dim3 get_grid_shape(Arguments const &args) {
    return UnderlyingOperator::get_grid_shape(to_underlying_arguments(args));
  }

  /// Computes the maximum number of active blocks per multiprocessor
  static int maximum_active_blocks(int smem_capacity = -1) {
    return UnderlyingOperator::maximum_active_blocks(smem_capacity);
  }

  /// Initializes GEMM state from arguments.
  Status initialize(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr
  ) {

    return underlying_operator_.initialize(to_underlying_arguments(args), workspace, stream, cuda_adapter);
  }

  /// Lightweight update given a subset of arguments.
  Status update(Arguments const &args) {

    return underlying_operator_.update(to_underlying_arguments(args));
  }

  /// Runs the kernel using initialized state.
  Status run(
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {

    return underlying_operator_.run(stream, cuda_adapter);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {

    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {

    Status status = initialize(args, workspace, stream, cuda_adapter);

    if (status == Status::kSuccess) {
      status = run(stream, cuda_adapter);
    }

    return status;
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::device

////////////////////////////////////////////////////////////////////////////////
