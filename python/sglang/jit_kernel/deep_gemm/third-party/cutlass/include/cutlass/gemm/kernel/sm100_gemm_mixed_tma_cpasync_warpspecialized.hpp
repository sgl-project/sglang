/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/detail/cluster.hpp"
#include "cutlass/fast_math.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/gemm/kernel/gemm_universal_decl.h"

#include "cutlass/gemm/kernel/tile_scheduler.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileSchedulerTag_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileSchedulerTag_,
  cute::enable_if_t<
    cutlass::detail::is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                KernelMixedTmaCpAsyncWarpSpecializedSm100>>>
{
public:
  using ProblemShape = ProblemShape_;

  static constexpr bool IsGroupedGemmKernel = cutlass::gemm::detail::is_moe_problem_shape<ProblemShape>::value;
  static constexpr bool IsMoEScheduler = false; // stub for MoE scheduler, which accepts a MoEProblemShape instead of GroupProblemShape
  
  CUTLASS_HOST_DEVICE
  static auto get_problem_shape_gemm(ProblemShape const& shape) {
    if constexpr (IsGroupedGemmKernel) {
      return shape.max_problem_shape;
    }
    else {
      return shape;
    }
  }
  CUTLASS_HOST_DEVICE
  static auto get_problem_shape_scheduler(ProblemShape const& shape) {
    if constexpr (IsMoEScheduler) {
      return shape;
    }
    else if constexpr (IsGroupedGemmKernel) {
      return shape.problem_shape;
    }
    else {
      return shape;
    }
  }

  template<class ProblemShape, class WorkTileInfo>
  CUTLASS_HOST_DEVICE
  static auto get_effective_shape(ProblemShape const& shape, WorkTileInfo const& work_tile_info) {
    if constexpr (IsGroupedGemmKernel) {
      return append<4>(shape.problem_shape.get_problem_shape(work_tile_info.L_idx), Int<1>{});
    }
    else {
      return append<4>(shape, Int<1>{});
    }
  }

  using ProblemShapeGemm = decltype(get_problem_shape_gemm(ProblemShape{}));
  using ProblemShapeScheduler = decltype(get_problem_shape_scheduler(ProblemShape{}));

  static_assert(rank(ProblemShapeGemm{}) == 3 or rank(ProblemShapeGemm{}) == 4,
    "ProblemShapeGemm{} should be <M,N,K> or <M,N,K,L>");
  static constexpr bool IsGdcEnabled = false;
  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  static_assert(ArchTag::kMinComputeCapability >= 100);

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueTile = typename CollectiveEpilogue::EpilogueTile;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static constexpr bool IsComplex = CollectiveEpilogue::NumAccumulatorMtxs == 2;

  // CLC pipeline depth
  // determines how many waves (stages-1) a warp can race ahead
  static constexpr uint32_t SchedulerPipelineStageCount = DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  static constexpr bool IsOverlappingAccum = DispatchPolicy::IsOverlappingAccum;
  static_assert(!IsOverlappingAccum, "TMA+CPASYNC kernel currently only supports non-overlapping accum.");

  // TileID scheduler
  // Get Blk and Scheduling tile shapes
  using CtaShape_MNK = typename CollectiveMainloop::CtaShape_MNK;
  using AtomThrShapeMNK = typename CollectiveMainloop::AtomThrShapeMNK;

  static_assert(size(AtomThrShapeMNK{}) == 1, "Lower alignment kernel only supports 1x1x1 cluster shape.");
  using TileSchedulerTag = cute::conditional_t<IsGroupedGemmKernel && !IsMoEScheduler, GroupScheduler, TileSchedulerTag_>;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileSchedulerTag, ArchTag, CtaShape_MNK, ClusterShape, SchedulerPipelineStageCount, ProblemShapeScheduler>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Warp specialization thread count per threadblock
  static constexpr uint32_t NumSchedThreads               = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumMMAThreads                 = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumEmptyThreads               = 0;
  static constexpr uint32_t NumMainloopTMALoadThreads     = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumMainloopCpAsyncLoadThreads = CollectiveMainloop::NumLoadThreadsCpAsync; // 4 warps
  static constexpr uint32_t NumEpilogueLoadThreads        = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumEpilogueThreads            = CollectiveEpilogue::ThreadCount;
  static constexpr uint32_t NumEpilogueWarps              = NumEpilogueThreads / NumThreadsPerWarp;

  static constexpr uint32_t MaxThreadsPerBlock = NumSchedThreads +
                                                 NumMainloopTMALoadThreads + NumMainloopCpAsyncLoadThreads +
                                                 NumMMAThreads +
                                                 NumEpilogueLoadThreads + NumEpilogueThreads + NumEmptyThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  static constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_load_pipe_increment(CtaShape_MNK{});

  static constexpr uint32_t NumFixupBarriers = 1;
  static constexpr uint32_t CLCResponseSize = sizeof(typename TileScheduler::CLCResponse);

  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;

  // Pipelines and pipeline states
  static constexpr uint32_t AccumulatorPipelineStageCount = DispatchPolicy::Schedule::AccumulatorPipelineStageCount;

  // Pipeline and pipeline state types
  using MainloopPipelineTMA = typename CollectiveMainloop::MainloopPipelineTMA;
  using MainloopPipelineTMAState = typename CollectiveMainloop::MainloopPipelineTMAState;
  using MainloopPipelineCpAsync = typename CollectiveMainloop::MainloopPipelineCpAsync;
  using MainloopPipelineCpAsyncState = typename CollectiveMainloop::MainloopPipelineCpAsyncState;

  using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
  using EpiLoadPipelineState = typename CollectiveEpilogue::LoadPipelineState;

  using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
  using EpiStorePipelineState = typename CollectiveEpilogue::StorePipelineState;

  using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount, AtomThrShapeMNK>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;

  // using CLCPipeline = cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape>;
  using CLCPipeline = cute::conditional_t<IsSchedDynamicPersistent,
    cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape>,
    cutlass::PipelineAsync<SchedulerPipelineStageCount>>;
  using CLCPipelineState = typename CLCPipeline::PipelineState;

  using TmemAllocator = cute::TMEM::Allocator1Sm;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;
      using CLCPipelineStorage = typename CLCPipeline::SharedStorage;
      using AccumulatorPipelineStorage = typename AccumulatorPipeline::SharedStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) CLCPipelineStorage clc;
      alignas(16) AccumulatorPipelineStorage accumulator;
      alignas(16) arch::ClusterBarrier tmem_dealloc;
    } pipelines;

    alignas(16) typename TileScheduler::CLCResponse clc_response[SchedulerPipelineStageCount];
    uint32_t tmem_base_ptr;

    struct TensorStorage : cute::aligned_struct<128, _1> {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    } tensors;

  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static_assert(SharedStorageSize <= cutlass::arch::sm100_smem_capacity_bytes, "SMEM usage exceeded capacity.");

  // Host facing host arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel device entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    ProblemShapeGemm problem_shape_gemm{};
    ProblemShapeScheduler problem_shape_scheduler{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
  };

  enum class WarpCategory : int32_t {
    MMA                 = 0,
    Sched               = 1,
    MainloopLoadTMA     = 2,
    EpilogueLoad        = 3,
    Epilogue            = 4,
    MainloopLoadCpAsync = 8
  };

  struct IsParticipant {
    uint32_t mma               = false;
    uint32_t sched             = false;
    uint32_t main_load_tma     = false;
    uint32_t epi_load          = false;
    uint32_t epilogue          = false;
    uint32_t main_load_cpasync = false;
  };

  // Convert to underlying arguments.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;
    // auto problem_shape = args.problem_shape;
    // auto problem_shape_MNKL = append<4>(problem_shape, 1);

    auto problem_shape_gemm = get_problem_shape_gemm(args.problem_shape);
    auto problem_shape_scheduler = get_problem_shape_scheduler(args.problem_shape);

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count != 0) {
      CUTLASS_TRACE_HOST("  WARNING: SM100 tile scheduler does not allow for user specified SM counts.\n"
          "  To restrict a kernel's resource usage, consider using CUDA driver APIs instead (green contexts).");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    // Epilogue
    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;

    // Tile scheduler
    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShapeScheduler, ElementAccumulator>(
      args.scheduler, problem_shape_scheduler, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    TileSchedulerParams scheduler;
    if constexpr (IsGroupedGemmKernel) {
      scheduler = TileScheduler::to_underlying_arguments(
        problem_shape_scheduler, TileShape{}, AtomThrShapeMNK{}, ClusterShape{},
        args.hw_info, args.scheduler, scheduler_workspace);
    }
    else {
      auto problem_shape = args.problem_shape;
      auto problem_shape_MNKL = append<4>(problem_shape, 1);

      scheduler = TileScheduler::to_underlying_arguments(
        problem_shape, TileShape{}, AtomThrShapeMNK{}, ClusterShape{},
        args.hw_info, args.scheduler, scheduler_workspace
      );
    }

    return {
      args.mode,
      args.problem_shape,
      problem_shape_gemm,
      problem_shape_scheduler,
      CollectiveMainloop::to_underlying_arguments(problem_shape_gemm, args.mainloop, mainloop_workspace),
      CollectiveEpilogue::to_underlying_arguments(problem_shape_gemm, args.epilogue, epilogue_workspace),
      hw_info,
      scheduler
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = true;

    if constexpr (IsGroupedGemmKernel) {
      implementable &= args.mode == GemmUniversalMode::kGrouped;
      implementable &= rank(ProblemShapeGemm{}) == 4;
      implementable &= rank(typename ProblemShape::UnderlyingProblemShape::UnderlyingProblemShape{}) == 3;
    }
    else {
      implementable &= (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && rank(ProblemShapeGemm{}) == 4);
    }
    
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    
    auto problem_shape_gemm = get_problem_shape_gemm(args.problem_shape);
    implementable &= CollectiveMainloop::can_implement(problem_shape_gemm, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(problem_shape_gemm, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);
    
    static constexpr int MaxClusterSize = 16;
    implementable &= size(ClusterShape{}) <= MaxClusterSize;

    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;

    auto problem_shape_gemm = get_problem_shape_gemm(args.problem_shape);
    auto problem_shape_scheduler = get_problem_shape_scheduler(args.problem_shape);

    // Epilogue
    workspace_size += CollectiveEpilogue::get_workspace_size(problem_shape_gemm, args.epilogue);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    // Tile scheduler
    workspace_size += TileScheduler::template get_workspace_size<ProblemShapeScheduler, ElementAccumulator>(
      args.scheduler, problem_shape_scheduler, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    auto problem_shape_gemm = get_problem_shape_gemm(args.problem_shape);
    auto problem_shape_scheduler = get_problem_shape_scheduler(args.problem_shape);

    // Epilogue
    status = CollectiveEpilogue::initialize_workspace(problem_shape_gemm, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(problem_shape_gemm, args.epilogue);
    status = cutlass::Status::kSuccess;
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    // Tile scheduler
    status = TileScheduler::template initialize_workspace<ProblemShapeScheduler, ElementAccumulator>(
      args.scheduler, workspace_ptr + workspace_offset, stream, problem_shape_scheduler, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs, cuda_adapter);
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShapeScheduler, ElementAccumulator>(
      args.scheduler, problem_shape_scheduler, args.hw_info, NumFixupBarriers);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  static dim3
  get_grid_shape(Params const& params) {
    auto cluster_shape = ClusterShape{};

    dim3 grid_shape;
    if constexpr (IsGroupedGemmKernel) {
      grid_shape = TileScheduler::get_grid_shape(
        params.scheduler,
        params.problem_shape_scheduler,
        TileShape{},
        AtomThrShapeMNK{},
        cluster_shape,
        params.hw_info);
    }
    else {
      auto problem_shape_MNKL = append<4>(params.problem_shape_scheduler, 1);
      grid_shape = TileScheduler::get_grid_shape(
        params.scheduler,
        problem_shape_MNKL,
        TileShape{},
        AtomThrShapeMNK{},
        cluster_shape,
        params.hw_info);
    }
    return grid_shape;
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {

    using namespace cute;
    using X = Underscore;
    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape_gemm, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // Account for more than one epilogue warp
    int warp_idx = canonical_warp_idx_sync();
    WarpCategory warp_category = warp_idx < static_cast<int>(WarpCategory::Epilogue)            ? WarpCategory(warp_idx)
                               : warp_idx < static_cast<int>(WarpCategory::MainloopLoadCpAsync) ? WarpCategory::Epilogue
                                                                                                : WarpCategory::MainloopLoadCpAsync;
    uint32_t lane_predicate = cute::elect_one_sync();
    auto tile_shape = TileShape{};
    auto cluster_shape = ClusterShape{};
    constexpr int cluster_size = size(ClusterShape{});
    int cta_rank_in_cluster = cute::block_rank_in_cluster();
    bool is_first_cta_in_cluster = cta_rank_in_cluster == 0;
    int cta_coord_v = cta_rank_in_cluster % size<0>(typename TiledMma::AtomThrID{});
    bool is_mma_leader_cta = cta_coord_v == 0;
    int mma_leader_ctas = size(shape_div(cluster_shape, AtomThrShapeMNK{}));
    [[maybe_unused]] uint32_t mma_peer_cta_rank = cta_rank_in_cluster;

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop(params.mainloop);
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    // Do we load source tensor C or other aux inputs
    bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();

    // printf("is_epi_load_needed = %d", (int)is_epi_load_needed);

    IsParticipant is_participant = {
      (warp_category == WarpCategory::MMA)   && is_mma_leader_cta,          // mma
      (warp_category == WarpCategory::Sched) && is_first_cta_in_cluster,    // sched
      (warp_category == WarpCategory::MainloopLoadTMA),                     // main_load_tma
      (warp_category == WarpCategory::EpilogueLoad) && is_epi_load_needed,  // epi_load
      (warp_category == WarpCategory::Epilogue),                            // epilogue
      (warp_category == WarpCategory::MainloopLoadCpAsync)                  // main_load_cpasync
    };

    // Mainloop Load pipeline (TMA)
    typename MainloopPipelineTMA::Params mainloop_pipeline_tma_params;
    if (WarpCategory::MainloopLoadTMA == warp_category) {
      mainloop_pipeline_tma_params.role = MainloopPipelineTMA::ThreadCategory::Producer;
    }
    if (WarpCategory::MMA == warp_category) {
      mainloop_pipeline_tma_params.role = MainloopPipelineTMA::ThreadCategory::Consumer;
    }

    mainloop_pipeline_tma_params.is_leader = lane_predicate && is_mma_leader_cta && is_participant.main_load_tma;
    mainloop_pipeline_tma_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytes;
    mainloop_pipeline_tma_params.initializing_warp = 0;
    MainloopPipelineTMA mainloop_pipeline_tma(shared_storage.pipelines.mainloop.tma,
                                              mainloop_pipeline_tma_params,
                                              cluster_shape,
                                              cute::true_type{},   // Perform barrier init
                                              cute::false_type{}); // Delay mask calculation

    // Mainloop Load pipeline (CpAsync)
    typename MainloopPipelineCpAsync::Params mainloop_pipeline_cpasync_params;
    if (WarpCategory::MainloopLoadCpAsync == warp_category) {
      mainloop_pipeline_cpasync_params.role = MainloopPipelineCpAsync::ThreadCategory::Producer;
    }
    if (WarpCategory::MMA == warp_category) {
      mainloop_pipeline_cpasync_params.role = MainloopPipelineCpAsync::ThreadCategory::Consumer;
    }

    mainloop_pipeline_cpasync_params.producer_arv_count = NumMainloopCpAsyncLoadThreads;
    mainloop_pipeline_cpasync_params.consumer_arv_count = 1; // Only UMMA consumes the A and B buffers
    mainloop_pipeline_cpasync_params.dst_blockid = cta_rank_in_cluster;
    mainloop_pipeline_cpasync_params.initializing_warp = 0;
    MainloopPipelineCpAsync mainloop_pipeline_cpasync(shared_storage.pipelines.mainloop.cpasync, mainloop_pipeline_cpasync_params, cluster_shape);

    // Epilogue Load pipeline
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (WarpCategory::EpilogueLoad == warp_category) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cta_rank_in_cluster;
    epi_load_pipeline_params.producer_arv_count = NumEpilogueLoadThreads;
    epi_load_pipeline_params.consumer_arv_count = NumEpilogueThreads;
    epi_load_pipeline_params.transaction_bytes = CollectiveEpilogue::TmaTransactionBytes;
    epi_load_pipeline_params.initializing_warp = 3;
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    // CLC pipeline
    typename CLCPipeline::Params clc_pipeline_params;
    if (WarpCategory::Sched == warp_category) {
      clc_pipeline_params.role = IsSchedDynamicPersistent ? CLCPipeline::ThreadCategory::ProducerConsumer : CLCPipeline::ThreadCategory::Producer;
    }
    else {
      clc_pipeline_params.role = CLCPipeline::ThreadCategory::Consumer;
    }
    clc_pipeline_params.producer_arv_count = 1;

    if constexpr (IsSchedDynamicPersistent) {
      clc_pipeline_params.producer_blockid = 0;
      clc_pipeline_params.consumer_arv_count = NumSchedThreads + cluster_size *
                                                 (NumMainloopTMALoadThreads + NumMainloopCpAsyncLoadThreads  + NumEpilogueThreads + NumMMAThreads);
      clc_pipeline_params.transaction_bytes = CLCResponseSize;
    }
    else {
      clc_pipeline_params.consumer_arv_count = NumMainloopTMALoadThreads + NumMainloopCpAsyncLoadThreads + NumEpilogueThreads + NumMMAThreads;
    }
    
    clc_pipeline_params.initializing_warp = 1;
    // CLCPipeline clc_pipeline(shared_storage.pipelines.clc, clc_pipeline_params, cluster_shape);
    // Now declare the pipeline outside the if constexpr
    CLCPipeline clc_pipeline = [&]() {
      if constexpr (IsSchedDynamicPersistent) {
        return CLCPipeline(shared_storage.pipelines.clc, clc_pipeline_params, cluster_shape);
      }
      else {
        return CLCPipeline(shared_storage.pipelines.clc, clc_pipeline_params);
      }
    }();

    // Mainloop-Epilogue pipeline
    typename AccumulatorPipeline::Params accumulator_pipeline_params;
    if (WarpCategory::MMA == warp_category) {
      accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) {
      accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Consumer;
    }
    // Only one producer thread arrives on this barrier.
    accumulator_pipeline_params.producer_arv_count = 1;
    accumulator_pipeline_params.consumer_arv_count = size(AtomThrShapeMNK{}) * NumEpilogueThreads;
    accumulator_pipeline_params.initializing_warp = 2;
    AccumulatorPipeline accumulator_pipeline(shared_storage.pipelines.accumulator, accumulator_pipeline_params, cluster_shape);

    // Tmem allocator
    TmemAllocator tmem_allocator{};

    // Sync allocation status between MMA and epilogue warps within CTA
    arch::NamedBarrier tmem_allocation_result_barrier(NumMMAThreads + NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
    // Sync deallocation status between MMA warps of peer CTAs
    arch::ClusterBarrier& tmem_deallocation_result_barrier = shared_storage.pipelines.tmem_dealloc;
    [[maybe_unused]] uint32_t dealloc_barrier_phase = 0;

    MainloopPipelineTMAState mainloop_pipe_tma_consumer_state;
    MainloopPipelineTMAState mainloop_pipe_tma_producer_state = cutlass::make_producer_start_state<MainloopPipelineTMA>();
    MainloopPipelineCpAsyncState mainloop_pipe_cpasync_consumer_state;
    MainloopPipelineCpAsyncState mainloop_pipe_cpasync_producer_state = cutlass::make_producer_start_state<MainloopPipelineCpAsync>();

    EpiLoadPipelineState epi_load_pipe_consumer_state;
    EpiLoadPipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();

    // epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    EpiStorePipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    CLCPipelineState clc_pipe_consumer_state;
    CLCPipelineState clc_pipe_producer_state = cutlass::make_producer_start_state<CLCPipeline>();

    AccumulatorPipelineState accumulator_pipe_consumer_state;
    AccumulatorPipelineState accumulator_pipe_producer_state = cutlass::make_producer_start_state<AccumulatorPipeline>();

    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer threadblocks in the cluster
    pipeline_init_arrive_relaxed(cluster_size);

    dim3 block_id_in_cluster = cute::block_id_in_cluster();
    // TileID scheduler
    TileScheduler scheduler(&shared_storage.clc_response[0], params.scheduler, block_id_in_cluster);
    typename TileScheduler::WorkTileInfo work_tile_info = scheduler.initial_work_tile_info(cluster_shape);
    auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

    //
    // TMEM "Allocation"
    //
    // auto acc_shape = collective_mainloop.partition_accumulator_shape();
    // auto bulk_tmem = TiledMma::make_fragment_C(append(acc_shape,
    //                                                   Int<AccumulatorPipelineStageCount>{}));
    auto tmem_storage = collective_mainloop.template init_tmem_tensors<EpilogueTile, IsOverlappingAccum>(EpilogueTile{});

    //
    // END PROLOGUE
    //

    // Synchronization call. Blocks until barriers are initialized in shared memory.
    pipeline_init_wait(cluster_size);

    // __syncwarp();
    // if (threadIdx.x % 32 == 0) {
    //   printf("warp %d start\n", warp_idx);
    // }

    if (is_participant.main_load_tma) {
      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();

      // bool do_load_order_arrive = is_epi_load_needed;
      bool requires_clc_query = true;

      auto load_inputs = collective_mainloop.load_init_tma(
        problem_shape_MNKL, shared_storage.tensors.mainloop);
      auto k_tiles = cute::get<0>(load_inputs);

      do {
        auto effective_shape = get_effective_shape(params.problem_shape, work_tile_info);

        // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.
        auto k_tile_iter = scheduler.get_k_tile_iterator(work_tile_info, effective_shape, CtaShape_MNK{}, k_tiles);
        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, effective_shape, CtaShape_MNK{});
        // auto k_tile_prologue = min(MainloopPipeline::Stages, k_tile_count);


        auto [mainloop_producer_state_next_, unused_] = collective_mainloop.load_tma(
          mainloop_pipeline_tma,
          mainloop_pipe_tma_producer_state,
          load_inputs,
          cta_coord_mnkl,
          k_tile_iter, k_tile_count      // - k_tile_prologue
        );
        mainloop_pipe_tma_producer_state = mainloop_producer_state_next_;

        // Sync warp to prevent non-participating threads entering next wave early
        __syncwarp();

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );
        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        requires_clc_query = increment_pipe;
        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }
      } while (work_tile_info.is_valid());
      collective_mainloop.load_tail_tma(mainloop_pipeline_tma, mainloop_pipe_tma_producer_state);

    }

    else if (is_participant.main_load_cpasync) {
      auto load_inputs = collective_mainloop.load_init_cpasync(
          problem_shape_MNKL, params.mainloop, shared_storage.tensors.mainloop,
          scheduler, work_tile_info);
      Tensor gA_mkl = get<0>(load_inputs);

      do {
        // Get current work tile and fetch next work tile
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

        auto effective_shape = get_effective_shape(params.problem_shape, work_tile_info);

        // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.
        auto k_tile_iter = scheduler.get_k_tile_iterator(work_tile_info, effective_shape, CtaShape_MNK{}, shape<3>(gA_mkl));
        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, effective_shape, CtaShape_MNK{});

        auto [mainloop_producer_state_next, unused_] = collective_mainloop.load_cpasync(
          params.mainloop,
          mainloop_pipeline_cpasync,
          mainloop_pipe_cpasync_producer_state,
          load_inputs,
          cta_coord_mnkl,
          k_tile_iter, k_tile_count,
          effective_shape
        );
        mainloop_pipe_cpasync_producer_state = mainloop_producer_state_next;

        // Sync warp to prevent non-participating threads entering next wave early
        __syncwarp();

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );
        work_tile_info = next_work_tile_info;

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }
      } while (work_tile_info.is_valid());

      collective_mainloop.load_tail_cpasync(mainloop_pipeline_cpasync, mainloop_pipe_cpasync_producer_state);

    }

    else if (is_participant.sched) {
      
      if constexpr (IsSchedDynamicPersistent) {
        // Whether a new CLC query must be performed.
        // See comment below where this variable is updated for a description of
        // why this variable is needed.
        bool requires_clc_query = true;

        cutlass::arch::wait_on_dependent_grids();

        do {
          if (requires_clc_query) {
            // Query next clcID and update producer state
            clc_pipe_producer_state = scheduler.advance_to_next_work(clc_pipeline, clc_pipe_producer_state);
          }

          // Fetch next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
            work_tile_info,
            clc_pipeline,
            clc_pipe_consumer_state
          );

          // Only perform a new CLC query if we consumed a new CLC query result in
          // `fetch_next_work`. An example of a case in which CLC `fetch_next_work` does
          // not consume a new CLC query response is when processing stream-K units.
          // The current stream-K scheduler uses single WorkTileInfo to track multiple
          // (potentially-partial) tiles to be computed via stream-K. In this case,
          // `fetch_next_work` simply performs in-place updates on the existing WorkTileInfo,
          // rather than consuming a CLC query response.
          requires_clc_query = increment_pipe;
          if (increment_pipe) {
            ++clc_pipe_consumer_state;
          }

          work_tile_info = next_work_tile_info;
        } while (work_tile_info.is_valid());
        clc_pipeline.producer_tail(clc_pipe_producer_state);
      }
      else {

        cutlass::arch::wait_on_dependent_grids();

        do {
          auto [next_work_tile_info, increment_pipe] = scheduler.advance_to_next_work(clc_pipeline, clc_pipe_producer_state);
          work_tile_info = next_work_tile_info;
          if (increment_pipe) {
            ++clc_pipe_producer_state;
          }
        } while (work_tile_info.is_valid());
        clc_pipeline.producer_tail(clc_pipe_producer_state);
      }
    }

    else if (is_participant.mma) {
      // Tmem allocation sequence
      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
      __syncwarp();
      tmem_allocation_result_barrier.arrive();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      // bulk_tmem.data() = tmem_base_ptr;
      collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base_ptr);


      // Pass the acc with tuple type since the bgrad kernel change the mma_init API
      auto mma_inputs = collective_mainloop.mma_init(params.mainloop, 
        tmem_storage, 
        shared_storage.tensors.mainloop);
      do {
        auto effective_shape = get_effective_shape(params.problem_shape, work_tile_info);
        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, effective_shape, CtaShape_MNK{});

        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        // Wait for tmem accumulator buffer to become empty with a flipped phase
        // accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);
        
        int acc_stage = accumulator_pipe_producer_state.index();
        // Tensor accumulators = bulk_tmem(_,_,_,acc_stage);
        auto [mainloop_pipe_tma_consumer_state_next_, mainloop_pipe_cpasync_consumer_state_next_] = collective_mainloop.mma(
          cute::make_tuple(mainloop_pipeline_tma, mainloop_pipeline_cpasync, accumulator_pipeline),
          cute::make_tuple(mainloop_pipe_tma_consumer_state, mainloop_pipe_cpasync_consumer_state, accumulator_pipe_producer_state),
          // Pass the acc with tuple type since the bgrad kernel change the mma API
          // cute::make_tuple(accumulators, accumulators),
          collective_mainloop.slice_accumulator(tmem_storage, acc_stage),
          mma_inputs,
          cta_coord_mnkl,
          k_tile_count
        );
        mainloop_pipe_tma_consumer_state = mainloop_pipe_tma_consumer_state_next_;
        mainloop_pipe_cpasync_consumer_state = mainloop_pipe_cpasync_consumer_state_next_;

        accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);

        ++accumulator_pipe_producer_state;
        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      } while (work_tile_info.is_valid());
      // Release the right to allocate before deallocations so that the next CTA can rasterize
      tmem_allocator.release_allocation_lock();

      accumulator_pipeline.producer_tail(accumulator_pipe_producer_state);

      // Free entire tmem allocation
      tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }

    else if (is_participant.epi_load) {
      bool do_tail_load = false;
      do {
        bool compute_epilogue = TileScheduler::compute_epilogue(work_tile_info, params.scheduler);

        // Get current work tile and fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );
        work_tile_info = next_work_tile_info;

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        if (compute_epilogue) {

          epi_load_pipe_producer_state = collective_epilogue.load(
            epi_load_pipeline,
            epi_load_pipe_producer_state,
            problem_shape_MNKL,
            CtaShape_MNK{},
            cta_coord_mnkl,
            TileShape{},
            TiledMma{},
            shared_storage.tensors.epilogue
          );

          do_tail_load = true;
        }

        // Calculate the cta coordinates of the next work tile
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      } while (work_tile_info.is_valid());

      // Only perform a tail load if one of the work units processed performed
      // an epilogue load. An example of a case in which a tail load should not be
      // performed is in split-K if a cluster is only assigned non-final splits (for which
      // the cluster does not compute the epilogue).
      if (do_tail_load) {
        collective_epilogue.load_tail(
          epi_load_pipeline, epi_load_pipe_producer_state,
          epi_store_pipeline, epi_store_pipe_producer_state);
      }
    }

    else if (is_participant.epilogue) {
      // Wait for tmem allocate here
      tmem_allocation_result_barrier.arrive_and_wait();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base_ptr);
      // bulk_tmem.data() = tmem_base_ptr;

      bool do_tail_store = false;
      do {
        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }
        // Accumulator stage slice
        int acc_stage = accumulator_pipe_consumer_state.index();
        // Tensor accumulators = bulk_tmem(_,_,_,acc_stage);
        auto accumulator = get<0>(collective_mainloop.slice_accumulator(tmem_storage, acc_stage));
        accumulator_pipe_consumer_state = scheduler.template fixup<IsComplex>(
          TiledMma{},
          work_tile_info,
          accumulator,
          accumulator_pipeline,
          accumulator_pipe_consumer_state,
          typename CollectiveEpilogue::CopyOpT2R{}
        );

        //
        // Epilogue and write to gD
        //
        if (scheduler.compute_epilogue(work_tile_info)) {
          auto [load_state_next, store_state_next, acc_state_next] = collective_epilogue.store(
            epi_load_pipeline,
            epi_load_pipe_consumer_state,
            epi_store_pipeline,
            epi_store_pipe_producer_state,
            accumulator_pipeline,
            accumulator_pipe_consumer_state,
            problem_shape_MNKL,
            CtaShape_MNK{},
            cta_coord_mnkl,
            TileShape{},
            TiledMma{},
            accumulator,
            shared_storage.tensors.epilogue
          );
          epi_load_pipe_consumer_state = load_state_next;
          epi_store_pipe_producer_state = store_state_next;
          accumulator_pipe_consumer_state = acc_state_next;
          do_tail_store = true;
        }

        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

      } while (work_tile_info.is_valid());

      // Only perform a tail store if one of the work units processed performed
      // an epilogue. An example of a case in which a tail load should not be
      // performed is in split-K if a cluster is only assigned non-final splits (for which
      // the cluster does not compute the epilogue).
      if (do_tail_store) {
        collective_epilogue.store_tail(
          epi_load_pipeline, epi_load_pipe_consumer_state,
          epi_store_pipeline, epi_store_pipe_producer_state,
          CtaShape_MNK{});
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
