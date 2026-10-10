/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <type_traits>

#include "cute/arch/cluster_sm90.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal_decl.h"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"
#include "cutlass/workspace.h"
#include "cutlass_extensions/gemm/kernel/sm90_gemm_array_tma_warpspecialized_precomputed_decl.hpp"
#include "cutlass_extensions/gemm/kernel/sm90_tile_scheduler_group_precomputed.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

#ifndef CUTLASS_EXTENSIONS_PRECOMPUTED_TENSORMAP_TRAITS_HPP_
#define CUTLASS_EXTENSIONS_PRECOMPUTED_TENSORMAP_TRAITS_HPP_

template <class CollectiveMainloop, class = void>
struct RequiresBatchTensormapUpdate {
  static constexpr bool value = true;
};

template <class CollectiveMainloop>
struct RequiresBatchTensormapUpdate<
    CollectiveMainloop,
    std::void_t<decltype(CollectiveMainloop::RequiresTensormapUpdateOnBatchChange)>> {
  static constexpr bool value = CollectiveMainloop::RequiresTensormapUpdateOnBatchChange;
};

template <class CollectiveMainloop, class = void>
struct RequiresBatchTensormapAcquire {
  static constexpr bool value = false;
};

template <class CollectiveMainloop>
struct RequiresBatchTensormapAcquire<
    CollectiveMainloop,
    std::void_t<decltype(CollectiveMainloop::RequiresPrebuiltTensormapAcquireOnBatchChange)>> {
  static constexpr bool value = CollectiveMainloop::RequiresPrebuiltTensormapAcquireOnBatchChange;
};

template <class CollectiveEpilogue, class = void>
struct PreferMaxMmaRegisters {
  static constexpr bool value = false;
};

template <class CollectiveEpilogue>
struct PreferMaxMmaRegisters<CollectiveEpilogue, std::void_t<decltype(CollectiveEpilogue::PreferMaxMmaRegisters)>> {
  static constexpr bool value = CollectiveEpilogue::PreferMaxMmaRegisters;
};

#endif

///////////////////////////////////////////////////////////////////////////////

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class GemmUniversalPrecomputedScheduler<
    ProblemShape_,
    CollectiveMainloop_,
    CollectiveEpilogue_,
    TileScheduler_,
    cute::enable_if_t<cute::is_base_of_v<
        KernelPtrArrayTmaWarpSpecializedPingpong,
        typename CollectiveMainloop_::DispatchPolicy::Schedule>>> {
 public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(
      rank(typename ProblemShape::UnderlyingProblemShape{}) == 3 or
          rank(typename ProblemShape::UnderlyingProblemShape{}) == 4,
      "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  static_assert(cute::is_base_of_v<
                KernelPtrArrayTmaWarpSpecializedPingpong,
                typename CollectiveMainloop_::DispatchPolicy::Schedule>);

  static constexpr bool IsGdcEnabled = false;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ElementA = typename CollectiveMainloop::ElementA;
  using StrideA = typename CollectiveMainloop::StrideA;
  using InternalStrideA = typename CollectiveMainloop::InternalStrideA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using InternalStrideB = typename CollectiveMainloop::InternalStrideB;
  using StrideB = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using Schedule = typename DispatchPolicy::Schedule;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC = typename CollectiveEpilogue::StrideC;
  using InternalStrideC = typename CollectiveEpilogue::InternalStrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD = typename CollectiveEpilogue::StrideD;
  using InternalStrideD = typename CollectiveEpilogue::InternalStrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static_assert(ArchTag::kMinComputeCapability >= 90);

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;

  static_assert(IsGroupedGemmKernel, "Precomputed grouped scheduler kernel is only for grouped ptr-array GEMM.");

  using SchedulerTag = GroupScheduler;
  using TileScheduler = cute::conditional_t<
      cute::is_void_v<TileScheduler_>,
      detail::PersistentTileSchedulerSm90GroupPrecomputed<ProblemShape, 8>,
      TileScheduler_>;

  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = 2;
  static constexpr uint32_t MaxThreadsPerBlock =
      CUTE_STATIC_V(size(TiledMma{})) + (NumMmaWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumProducerThreads = CollectiveMainloop::NumProducerThreadEvents;

  /// Register requirement for Load and Math WGs
  static constexpr bool UseMaxMmaRegisters = PreferMaxMmaRegisters<CollectiveEpilogue>::value;
  static constexpr uint32_t LoadRegisterRequirement = UseMaxMmaRegisters ? 32 : 40;
  static constexpr uint32_t MmaRegisterRequirement = UseMaxMmaRegisters ? 240 : 232;

  // 1 stage ordered sequence between mainloop and epilogue producer load threads
  using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1, 2>;

  // Order Sequence barrier with two stages: one for Mainloop and one for Epilogue
  static constexpr uint32_t StagesPerMathWarpGroup = 2;
  using MathWarpGroupOrderBarrier = cutlass::OrderedSequenceBarrier<StagesPerMathWarpGroup, NumMmaWarpGroups>;
  using MathWarpGroupOrderBarrierSharedStorage = cutlass::PipelineDetail::OrderedSequenceBarrierSharedStorage<
      MathWarpGroupOrderBarrier::SequenceDepth,
      MathWarpGroupOrderBarrier::SequenceLength>;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _1> {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    } tensors;

    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;
      using MathWarpGroupOrderBarrierStorage = MathWarpGroupOrderBarrierSharedStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) typename LoadWarpOrderBarrier::SharedStorage load_order;
      alignas(16) MathWarpGroupOrderBarrierStorage math_wg_order;
    } pipelines;

    struct TensorMapStorage : cute::aligned_struct<128, _1> {
      using MainloopTensorMapStorage = typename CollectiveMainloop::TensorMapStorage;
      using EpilogueTensorMapStorage = typename CollectiveEpilogue::TensorMapStorage;

      alignas(128) MainloopTensorMapStorage mainloop;
      alignas(128) EpilogueTensorMapStorage epilogue;
    } tensormaps;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
    void* workspace{nullptr};
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    ProblemShape problem_shapes = args.problem_shape;

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST(
          "  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM "
          "count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    // Get maximum number of clusters that could co-exist on the target device
    int max_active_clusters = args.hw_info.max_active_clusters;
    if (max_active_clusters <= 0) {
      max_active_clusters = 0;
      CUTLASS_TRACE_HOST(
          "  WARNING: Arguments do not include a valid max cluster count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the "
          "max_active_clusters.");
    } else {
      CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid cluster count to " << max_active_clusters);
    }

    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count, max_active_clusters};

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(problem_shapes, args.epilogue, sm_count);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);

    void* mainloop_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveMainloop::get_workspace_size(problem_shapes, args.mainloop, sm_count);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);

    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset +=
        TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
            args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);

    // Precompute the sub tiles numbers in epilogue, pass into tile scheduler.  Therefore it will be
    // used in separate reduction scheme for streamk case, NumEpilogueSubTiles default value is 1,
    // which means subtile will not be used, therefore separate reduction will not be enabled.
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    TileSchedulerParams scheduler;
    if constexpr (IsGroupedGemmKernel) {
      scheduler = TileScheduler::to_underlying_arguments(
          problem_shapes,
          TileShape{},
          ClusterShape{},
          hw_info,
          args.scheduler,
          scheduler_workspace,
          NumEpilogueSubTiles);
    } else {
      scheduler = TileScheduler::to_underlying_arguments(
          problem_shapes.get_host_problem_shape(),
          TileShape{},
          ClusterShape{},
          hw_info,
          args.scheduler,
          scheduler_workspace,
          NumEpilogueSubTiles);
    }

    return {
        args.mode,
        problem_shapes,
        CollectiveMainloop::to_underlying_arguments(problem_shapes, args.mainloop, mainloop_workspace),
        CollectiveEpilogue::to_underlying_arguments(problem_shapes, args.epilogue, epilogue_workspace),
        hw_info,
        scheduler,
        workspace};
  }

  static bool can_implement(Arguments const& args) {
    bool implementable = true;
    if constexpr (IsGroupedGemmKernel) {
      // Group GEMM currently only supports rank-3 problem shapes
      implementable &=
          (args.mode == GemmUniversalMode::kGrouped && rank(typename ProblemShape::UnderlyingProblemShape{}) == 3);
    } else {
      implementable &=
          (args.mode == GemmUniversalMode::kArray && rank(typename ProblemShape::UnderlyingProblemShape{}) == 4);
    }
    if (!implementable) {
      CUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements for Ptr Array "
          "Gemm or Grouped Gemm.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);
    return implementable;
  }

  static size_t get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST(
          "  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM "
          "count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue, sm_count);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    workspace_size += CollectiveMainloop::get_workspace_size(args.problem_shape, args.mainloop, sm_count);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    workspace_size +=
        TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
            args.scheduler,
            typename ProblemShape::UnderlyingProblemShape{},
            args.hw_info,
            NumMmaWarpGroups,
            NumEpilogueSubTiles);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    return workspace_size;
  }

  static cutlass::Status initialize_workspace(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr,
      CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    static constexpr uint32_t NumAccumulatorMtxs = 1;

    status = CollectiveEpilogue::initialize_workspace(
        args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset +=
        CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue, args.hw_info.sm_count);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = CollectiveMainloop::initialize_workspace(
        args.problem_shape, args.mainloop, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset +=
        CollectiveMainloop::get_workspace_size(args.problem_shape, args.mainloop, args.hw_info.sm_count);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status =
        TileScheduler::template initialize_workspace<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
            args.scheduler,
            workspace_ptr + workspace_offset,
            stream,
            typename ProblemShape::UnderlyingProblemShape{},
            args.hw_info,
            NumMmaWarpGroups,
            NumEpilogueSubTiles,
            NumAccumulatorMtxs,
            cuda_adapter);
    workspace_offset +=
        TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
            args.scheduler,
            typename ProblemShape::UnderlyingProblemShape{},
            args.hw_info,
            NumMmaWarpGroups,
            NumEpilogueSubTiles);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }
    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3 get_grid_shape(Params const& params) {
    // Given device SM count, set grid size s.t. we do not launch more thread blocks than we can run
    // concurrently
    TileSchedulerArguments args{};
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN
                            ? TileScheduler::RasterOrderOptions::AlongN
                            : TileScheduler::RasterOrderOptions::AlongM;
    dim3 grid_shape;
    if constexpr (IsGroupedGemmKernel) {
      grid_shape = TileScheduler::get_grid_shape(
          params.scheduler, params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, args);
    } else {
      grid_shape = TileScheduler::get_grid_shape(
          params.scheduler,
          params.problem_shape.get_host_problem_shape(),
          TileShape{},
          ClusterShape{},
          params.hw_info,
          args);
    }
    return grid_shape;
  }

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

// Any Tensor Op MMA Atom in the WGMMA ISA is arch conditional to sm90a.
#if !defined(__CUDA_ARCH_FEAT_SM90_ALL)
    printf(
        "ERROR : Arch conditional MMA instruction used without targeting sm90a compute capability. "
        "Aborting.\n");
#else

    // Preconditions
    static_assert(size(TiledMma{}) == 128, "Pingpong kernel must have TiledMMA operating using 128 threads.");
    static_assert(NumMmaWarpGroups == 2, "Pingpong kernels currently only support NumMmaWarpGroups == 2");

    if constexpr (cutlass::epilogue::collective::detail::sm90_is_ptr_array_tma_dispatch_policy_v<
                      typename CollectiveEpilogue::DispatchPolicy>) {
      static_assert(
          NumMmaWarpGroups == CollectiveEpilogue::NumEpilogueWarpGroups,
          "Tiled MmA does not match expected warp groups performing the epilogue");
    }

    static_assert(
        cute::rank(InternalStrideA{}) == 3,
        "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(
        cute::rank(InternalStrideB{}) == 3,
        "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(
        cute::rank(InternalStrideC{}) == 3,
        "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(
        cute::rank(InternalStrideD{}) == 3,
        "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    enum class WarpGroupRole { Producer = 0, Consumer0 = 1, Consumer1 = 2 };
    enum class ProducerWarpRole { Mainloop = 0, Warp1 = 1, Epilogue = 2, Warp3 = 3 };

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = int(threadIdx.x);
    int lane_idx = canonical_lane_idx();
    int warp_idx = canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    int mma_thread_idx = thread_idx % size(TiledMma{});
    auto warp_group_idx = canonical_warp_group_idx();
    auto warp_group_role = WarpGroupRole(warp_group_idx);
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Note: Tma Descriptor Prefetch (from either const or param) is not applicable here

    // Mainloop Load pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Mainloop) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = NumThreadsPerWarpGroup;
    mainloop_pipeline_params.num_producers = NumProducerThreads;
    mainloop_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, mainloop_pipeline_params, ClusterShape{});

    // Epilogue Load pipeline
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Epilogue) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
    epi_load_pipeline_params.consumer_arv_count = NumThreadsPerWarpGroup;
    if constexpr (CollectiveEpilogue::RequiresTransactionBytes) {
      epi_load_pipeline_params.transaction_bytes = params.epilogue.tma_transaction_bytes;
    }
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename LoadWarpOrderBarrier::Params params_load_order_barrier;
    params_load_order_barrier.group_id = producer_warp_role == ProducerWarpRole::Mainloop ? 0 : 1;
    params_load_order_barrier.group_size = NumThreadsPerWarp;
    LoadWarpOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, params_load_order_barrier);

    typename MathWarpGroupOrderBarrier::Params params_math_wg_order_barrier;
    // DMA Load WG will not participate in these Ordered Barrier syncs
    params_math_wg_order_barrier.group_id = warp_group_idx - static_cast<int>(WarpGroupRole::Consumer0);
    params_math_wg_order_barrier.group_size = NumThreadsPerWarpGroup;  // Number of threads / participants in a group
    MathWarpGroupOrderBarrier math_wg_order_barrier(
        shared_storage.pipelines.math_wg_order, params_math_wg_order_barrier);

    // Initialize starting pipeline states for the collectives
    // Epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    // For the DMA Load (producer) we start with an opposite phase
    // i.e., we skip all waits since we know that the buffer is indeed empty
    PipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
    PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    auto cluster_wait_fn = []() {
      // We need this to guarantee that the Pipeline init is visible
      // To all producers and consumer thread blocks in the Cluster
      if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        return []() { cute::cluster_wait(); };
      } else {
        __syncthreads();
        return []() {};  // do nothing
      }
    }();

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    TiledMma tiled_mma;
    const auto blk_shape = TileShape{};  // (BLK_M,BLK_N,BLK_K)
    const auto c_tile_count = CollectiveEpilogue::get_load_pipe_increment(blk_shape);
    const auto d_tile_count = CollectiveEpilogue::get_store_pipe_increment(blk_shape);

    TileScheduler scheduler{params.scheduler};

    // In a warp specialized kernel, collectives expose data movement and compute operations
    // separately
    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();
    auto work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});

    if (not work_tile_info.is_valid()) {
      // When problem shapes are only on device, the grid launched may be larger than the total
      // number of blocks across groups
      return;
    }

    // Optionally append 1s until problem shape is rank-4 in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);

    // Consumer1 is not on the critical path at prologue.
    if (warp_group_role == WarpGroupRole::Consumer1) [[unlikely]] {
      // Advance 2nd Math WG to the next work tile for the startup
      const auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);

      auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
      work_tile_info = next_work_tile_info;
      if (!work_tile_info.is_valid()) {
        return;
      }

      // Advance 2nd Math WG pipeline states to the end of 1st Math WG
      mainloop_pipe_consumer_state.advance(k_tile_count);
      epi_load_pipe_consumer_state.advance(c_tile_count);
      epi_store_pipe_producer_state.advance(d_tile_count);

      problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
    }

    // Prepare and partition the input tensors. Expects a tuple of tensors where:
    // get<0>(load_inputs) is the tma tensor A after local tiling so that it has shape
    // (BLK_M,BLK_K,m,k,l) get<1>(load_inputs) is the tma tensor B after local tiling so that it has
    // shape (BLK_N,BLK_K,n,k,l)
    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
    static_assert(
        cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

    // Extract out partitioned A and B.
    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    // Get pipeline stage increments from tensor shapes
    auto k_tile_count = size<3>(gA_mkl);

    if (warp_group_role == WarpGroupRole::Producer) {
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

      // Mainloop Producer Warp
      if (producer_warp_role == ProducerWarpRole::Mainloop) {
        int32_t curr_batch =
            idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));  // Usually just returns work_tile_info.L_idx;
        int32_t const mock_l_coord = 0;
        int32_t const sm_idx = blockIdx.x + (blockIdx.y * gridDim.x);
        int32_t const sm_count = params.hw_info.sm_count;

        // Fetch a copy of tensormaps for the CTA
        auto input_tensormaps =
            collective_mainloop.tensormaps_init(params.mainloop, shared_storage.tensormaps.mainloop, sm_count, sm_idx);

        // Update tensormap for the initial batch for the CTA
        collective_mainloop.tensormaps_perform_update(
            shared_storage.tensormaps.mainloop, params.mainloop, input_tensormaps, problem_shape_MNKL, curr_batch);
        // Ensure warp is converged before issuing tensormap fence release
        __syncwarp();
        // Entire warp must do this (i.e. it's aligned)
        collective_mainloop.tensormaps_cp_fence_release(shared_storage.tensormaps.mainloop, input_tensormaps);

        bool do_load_order_arrive = true;
        bool did_batch_change = true;
        bool needs_tensormap_acquire = work_tile_info.is_valid();
        while (work_tile_info.is_valid()) {
          if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
            auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
            work_tile_info = next_work_tile_info;
            continue;
          }

          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, mock_l_coord);

          // Get the number of K tiles to compute for this work as well as the starting K tile
          // offset of the work.
          auto work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);
          auto work_k_tile_start = TileScheduler::get_work_k_tile_start(work_tile_info);
          auto k_tile_iter = cute::make_coord_iterator(idx2crd(work_k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl));

          if (did_batch_change) {
            load_inputs = collective_mainloop.tensors_perform_update(
                load_inputs, params.mainloop, problem_shape_MNKL, curr_batch);
            if (needs_tensormap_acquire || RequiresBatchTensormapAcquire<CollectiveMainloop>::value) {
              collective_mainloop.tensormaps_fence_acquire(input_tensormaps);
              needs_tensormap_acquire = false;
            }
          }

          collective_mainloop.load(
              params.mainloop,
              mainloop_pipeline,
              mainloop_pipe_producer_state,
              load_inputs,
              input_tensormaps,
              blk_coord,
              k_tile_iter,
              work_k_tile_count,
              lane_idx,
              block_rank_in_cluster,
              shared_storage.tensors.mainloop);
          // Update starting pipeline state for the next tile
          // Wait for the last TMA stage to complete loading, before issuing tensormap updates
          mainloop_pipe_producer_state.advance(work_k_tile_count - 1);

          // Signal for the epilogue load warp to begin
          if (do_load_order_arrive) {
            load_order_barrier.arrive();
            do_load_order_arrive = false;
          }

          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
          work_tile_info = next_work_tile_info;
          auto next_batch = idx2crd(
              work_tile_info.L_idx,
              shape<4>(gB_nkl));  // Usually just returns work_tile_info.L_idx
          did_batch_change = next_batch != curr_batch;
          if (work_tile_info.is_valid() && did_batch_change) {
            curr_batch = next_batch;
            if constexpr (IsGroupedGemmKernel) {
              problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(curr_batch), 1);
            }
            if constexpr (RequiresBatchTensormapUpdate<CollectiveMainloop>::value) {
              // Purpose of this pipeline state is to make sure TMA loads have finished before doing
              // descriptor updates Since this state is waiting for loads to finish, it must start
              // in the inverted phase.
              typename CollectiveMainloop::PipelineState mainloop_pipe_tma_consumer_state = {
                  mainloop_pipe_producer_state.index(),
                  !mainloop_pipe_producer_state.phase(),
                  mainloop_pipe_producer_state.count()};
              mainloop_pipeline.consumer_wait(mainloop_pipe_tma_consumer_state);
              collective_mainloop.tensormaps_perform_update(
                  shared_storage.tensormaps.mainloop,
                  params.mainloop,
                  input_tensormaps,
                  problem_shape_MNKL,
                  curr_batch);
              // Ensure warp is converged before issuing tensor replace
              __syncwarp();
              // Entire warp must do this (i.e. it's aligned)
              collective_mainloop.tensormaps_cp_fence_release(shared_storage.tensormaps.mainloop, input_tensormaps);
              needs_tensormap_acquire = true;
            }
          }
          // Advance the producer state for the last remaining stage that was being waited for above
          mainloop_pipe_producer_state.advance(1);
        }  // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);
      }  // Mainloop Producer Warp End
      // Epilogue Producer Warp
      else if (producer_warp_role == ProducerWarpRole::Epilogue && collective_epilogue.is_producer_load_needed()) {
        int32_t const sm_idx = blockIdx.x + (blockIdx.y * gridDim.x);
        int32_t const sm_count = params.hw_info.sm_count;

        auto epi_load_tensormap = get<0>(
            collective_epilogue.load_init(params.epilogue, shared_storage.tensormaps.epilogue, sm_count, sm_idx));

        bool did_batch_change = true;
        constexpr bool IsEpiLoad = true;

        if (work_tile_info.is_valid()) {
          collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
              shared_storage.tensormaps.epilogue,
              params.epilogue,
              epi_load_tensormap,
              problem_shape_MNKL,
              work_tile_info.L_idx,
              0);

          // Converge before issuing tensormap fence release since fence is aligned
          __syncwarp();
          collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(
              shared_storage.tensormaps.epilogue, epi_load_tensormap, 0);
        }

        load_order_barrier.wait();

        while (work_tile_info.is_valid()) {
          int32_t curr_batch = work_tile_info.L_idx;

          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);

          if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
            if constexpr (IsGroupedGemmKernel) {
              problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
            }

            // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
            auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
            auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
            auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
            auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

            if (did_batch_change) {
              collective_epilogue.template tensormaps_fence_acquire<IsEpiLoad>(epi_load_tensormap);
            }

            bool wait = work_tile_info.is_valid() && curr_batch != next_work_tile_info.L_idx;

            epi_load_pipe_producer_state = collective_epilogue.load(
                epi_load_pipeline,
                epi_load_pipe_producer_state,
                problem_shape_MNKL,
                blk_shape,
                blk_coord,
                tiled_mma,
                lane_idx,
                shared_storage.tensors.epilogue,
                epi_load_tensormap,
                work_tile_info.reduction_subtile_idx(),
                wait);
          }

          work_tile_info = next_work_tile_info;
          did_batch_change = curr_batch != work_tile_info.L_idx;

          if (work_tile_info.is_valid() && did_batch_change) {
            if constexpr (IsGroupedGemmKernel) {
              problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
            }

            // tensormap update
            {
              collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
                  shared_storage.tensormaps.epilogue,
                  params.epilogue,
                  epi_load_tensormap,
                  problem_shape_MNKL,
                  work_tile_info.L_idx,
                  0);

              // Converge before issuing tensormap fence release since fence is aligned
              __syncwarp();
              collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(
                  shared_storage.tensormaps.epilogue, epi_load_tensormap, 0);
            }
          }

        }  // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
      }  // Epilogue Producer Warp End
    }  // Producer Warp Group End

    else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

      // Index of warp group within consumer warp groups
      int consumer_warp_group_idx = warp_group_role == WarpGroupRole::Consumer0 ? 0 : 1;

      int32_t const sm_idx = blockIdx.x + (blockIdx.y * gridDim.x);
      int32_t const sm_count = params.hw_info.sm_count;
      // Do we potentially issue tail arrives for TMA stores, if epilogue load is waiting for it
      bool do_store_tail = false;
      // Get a copy of tensormaps
      auto epi_store_tensormap = get<0>(collective_epilogue.store_init(
          params.epilogue, shared_storage.tensormaps.epilogue, sm_count, sm_idx, consumer_warp_group_idx));

      bool did_batch_change = true;
      constexpr bool IsEpiLoad = false;

      if (work_tile_info.is_valid()) {
        if (warp_idx_in_warp_group == 0) {
          collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
              shared_storage.tensormaps.epilogue,
              params.epilogue,
              epi_store_tensormap,
              problem_shape_MNKL,
              work_tile_info.L_idx,
              consumer_warp_group_idx);

          // Converge before issuing tensormap fence release since fence is aligned
          __syncwarp();
          collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(
              shared_storage.tensormaps.epilogue, epi_store_tensormap, consumer_warp_group_idx);
        }
      }

      while (work_tile_info.is_valid()) {
        if constexpr (IsGroupedGemmKernel) {
          problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
        }

        int32_t curr_batch = work_tile_info.L_idx;

        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);
        auto work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);

        // Allocate the accumulators for the (M,N) blk_shape
        //
        // MSVC CTAD breaks if we say "Tensor" here, so we use "auto" instead.
        auto accumulators = partition_fragment_C(tiled_mma, take<0, 2>(blk_shape));  // (MMA,MMA_M,MMA_N)

        if (TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
          math_wg_order_barrier.wait();

          collective_mainloop.mma(
              mainloop_pipeline,
              mainloop_pipe_consumer_state,
              accumulators,
              work_k_tile_count,
              mma_thread_idx,
              shared_storage.tensors.mainloop,
              params.mainloop);

          math_wg_order_barrier.arrive();

          // Make sure the math instructions are done and free buffers before entering the epilogue
          collective_mainloop.mma_tail(mainloop_pipeline, mainloop_pipe_consumer_state, work_k_tile_count);

          math_wg_order_barrier.wait();

          // Update starting mainloop pipeline state for the next tile
          mainloop_pipe_consumer_state.advance(work_k_tile_count);
        }

        // Perform reduction across splits, if needed
        TileScheduler::fixup(params.scheduler, work_tile_info, accumulators, NumMmaWarpGroups, consumer_warp_group_idx);

        if (did_batch_change && warp_idx_in_warp_group == 0) {
          collective_epilogue.template tensormaps_fence_acquire<IsEpiLoad>(epi_store_tensormap);
        }

        if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
          // Epilogue and write to gD
          auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next] = collective_epilogue.store(
              epi_load_pipeline,
              epi_load_pipe_consumer_state,
              epi_store_pipeline,
              epi_store_pipe_producer_state,
              problem_shape_MNKL,
              blk_shape,
              blk_coord,
              accumulators,
              tiled_mma,
              mma_thread_idx,
              shared_storage.tensors.epilogue,
              epi_store_tensormap,
              work_tile_info.reduction_subtile_idx());

          epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next;
          epi_store_pipe_producer_state = epi_store_pipe_producer_state_next;
          do_store_tail = true;
        }

        // Get next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);

        work_tile_info = next_work_tile_info;

        // Skip a tile for pingpong
        if (work_tile_info.is_valid()) {
          if constexpr (IsGroupedGemmKernel) {
            problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
          }
          work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);
          mainloop_pipe_consumer_state.advance(work_k_tile_count);

          // Go to next tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
          work_tile_info = next_work_tile_info;
        }

        did_batch_change = curr_batch != work_tile_info.L_idx;
        if (work_tile_info.is_valid() && did_batch_change) {
          if constexpr (IsGroupedGemmKernel) {
            problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
          }
          if (warp_idx_in_warp_group == 0) {
            collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
                shared_storage.tensormaps.epilogue,
                params.epilogue,
                epi_store_tensormap,
                problem_shape_MNKL,
                work_tile_info.L_idx,
                consumer_warp_group_idx);

            // Converge before issuing tensormap fence release since fence is aligned
            __syncwarp();
            collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(
                shared_storage.tensormaps.epilogue, epi_store_tensormap, consumer_warp_group_idx);
          }
        }

        // TMA store pipeline wait is only visible to TMA-issuing warp, so for multiple-consumer
        // kernels we need to wait for all TMA stores to complete before issuing consumer order
        // barrier arrives to ensure next math consumer doesn't overwrite smem of in-flight TMA
        // stores of current consumer.
        auto [epi_load_pipe_consumer_state_next_, epi_store_pipe_producer_state_next_] = collective_epilogue.store_tail(
            epi_load_pipeline, epi_load_pipe_consumer_state, epi_store_pipeline, epi_store_pipe_producer_state);

        // Update starting load/store pipeline states for the next tile
        // state has already been incremented by 1 tile in collective calls, advance once again for
        // ping pong
        epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next_;
        epi_store_pipe_producer_state = epi_store_pipe_producer_state_next_;
        epi_load_pipe_consumer_state.advance(c_tile_count);
        epi_store_pipe_producer_state.advance(d_tile_count);

        // Cue for next Math WG's Epilogue to start
        math_wg_order_barrier.arrive();

      }  // Scheduler work fetch loop
    }  // Consumer Warp Groups End
#endif
  }
};

///////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::kernel
