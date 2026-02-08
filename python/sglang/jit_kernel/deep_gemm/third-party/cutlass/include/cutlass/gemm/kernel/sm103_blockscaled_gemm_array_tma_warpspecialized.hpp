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
#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/fast_math.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/detail/sm100_tmem_helper.hpp"

#include "cute/tensor.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/atom/mma_atom.hpp"

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
                                KernelPtrArrayTmaWarpSpecializedBlockScaledSm103>>>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(rank(typename ProblemShape::UnderlyingProblemShape{}) == 3 or rank(typename ProblemShape::UnderlyingProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using InternalStrideA = typename CollectiveMainloop::InternalStrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using InternalStrideB = typename CollectiveMainloop::InternalStrideB;
  using LayoutSFA = typename CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename CollectiveMainloop::LayoutSFB;
  using ElementSF = typename CollectiveMainloop::ElementSF;
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
  using InternalStrideC = typename CollectiveEpilogue::InternalStrideC; 
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using InternalStrideD = typename CollectiveEpilogue::InternalStrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  // CLC pipeline depth
  // determines how many waves (stages-1) a warp can race ahead
  static constexpr uint32_t SchedulerPipelineStageCount = DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  static constexpr uint32_t AccumulatorPipelineStageCount = DispatchPolicy::Schedule::AccumulatorPipelineStageCount;
  static constexpr bool IsOverlappingAccum = DispatchPolicy::IsOverlappingAccum;

  // TileID scheduler
  // Get Blk and Scheduling tile shapes
  using AtomThrShapeMNK = typename CollectiveMainloop::AtomThrShapeMNK;
  using CtaShape_MNK = typename CollectiveMainloop::CtaShape_MNK;

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;
  using TileSchedulerTag = TileSchedulerTag_;
  using TileScheduler = cute::conditional_t<IsGroupedGemmKernel,
      typename detail::TileSchedulerSelector<
        GroupScheduler, ArchTag, CtaShape_MNK, ClusterShape, SchedulerPipelineStageCount, ProblemShape>::Scheduler,
      typename detail::TileSchedulerSelector<
        TileSchedulerTag_, ArchTag, CtaShape_MNK, ClusterShape, SchedulerPipelineStageCount>::Scheduler>;

  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;
  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;
  static constexpr bool IsGdcEnabled = cutlass::arch::IsGdcGloballyEnabled;

  static constexpr uint32_t MinTensorMapWorkspaceAlignment = 64;

  // Warp specialization thread count per threadblock
  static constexpr uint32_t NumSchedThreads          = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumMMAThreads            = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumMainloopABLoadThreads = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumMainloopSFLoadThreads = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumEpilogueThreads       = CollectiveEpilogue::ThreadCount;
  static constexpr uint32_t NumEpilogueWarps         = NumEpilogueThreads / NumThreadsPerWarp;
  static constexpr uint32_t NumEpilogueLoadThreads   = NumThreadsPerWarp; // 1 warp
  static constexpr uint32_t NumEmptyThreads          = 3 * NumThreadsPerWarp; // 3 warp

  static constexpr uint32_t MaxThreadsPerBlock = NumSchedThreads +
                                                 NumMainloopABLoadThreads + NumMainloopSFLoadThreads + NumMMAThreads +
                                                 NumEpilogueLoadThreads + NumEpilogueThreads + NumEmptyThreads;

  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumFixupBarriers = 1;
  static constexpr uint32_t CLCResponseSize = sizeof(typename TileScheduler::CLCResponse);

  // Pipeline and pipeline state types
  using MainloopABPipeline = typename CollectiveMainloop::MainloopABPipeline;
  using MainloopABPipelineState = typename CollectiveMainloop::MainloopABPipelineState;

  using MainloopSFPipeline = typename CollectiveMainloop::MainloopSFPipeline;
  using MainloopSFPipelineState = typename CollectiveMainloop::MainloopSFPipelineState;

  using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
  using EpiLoadPipelineState = typename CollectiveEpilogue::LoadPipelineState;

  using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
  using EpiStorePipelineState = typename CollectiveEpilogue::StorePipelineState;

  using LoadOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;

  using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount, AtomThrShapeMNK>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;

  using CLCPipeline = cute::conditional_t<IsSchedDynamicPersistent,
    cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape>,
    cutlass::PipelineAsync<SchedulerPipelineStageCount>>;
  using CLCPipelineState = typename CLCPipeline::PipelineState;

  using CLCThrottlePipeline = cute::conditional_t<IsSchedDynamicPersistent,
    cutlass::PipelineAsync<SchedulerPipelineStageCount>,
    cutlass::PipelineEmpty>;
  using CLCThrottlePipelineState = typename CLCThrottlePipeline::PipelineState;

  using TmemAllocator = cute::conditional_t<cute::size(cute::shape<0>(typename TiledMma::ThrLayoutVMNK{})) == 1,
      cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

  static constexpr int EpilogueWarpRegs = 248;
  static constexpr int NonEpilogueWarpRegs = 128;

  // Kernel level shared memory storage
  struct SharedStorage {
    // Barriers should be allocated in lower 8KB of SMEM for SM100
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;
      using LoadOrderBarrierStorage = typename LoadOrderBarrier::SharedStorage;
      using CLCPipelineStorage = typename CLCPipeline::SharedStorage;
      using AccumulatorPipelineStorage = typename AccumulatorPipeline::SharedStorage;
      using CLCThrottlePipelineStorage = typename CLCThrottlePipeline::SharedStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) LoadOrderBarrierStorage load_order;
      alignas(16) CLCPipelineStorage clc;
      alignas(16) AccumulatorPipelineStorage accumulator;
      alignas(16) CLCThrottlePipelineStorage clc_throttle;
      alignas(8) arch::ClusterBarrier tmem_dealloc;
    } pipelines;

    alignas(16) typename TileScheduler::CLCResponse clc_response[SchedulerPipelineStageCount];
    uint32_t tmem_base_ptr;

    struct TensorMapStorage : cute::aligned_struct<128, _1> {
      using EpilogueTensorMapStorage = typename CollectiveEpilogue::TensorMapStorage;
      using MainloopTensorMapStorage = typename CollectiveMainloop::TensorMapStorage;
      alignas(128) EpilogueTensorMapStorage epilogue;
      alignas(128) MainloopTensorMapStorage mainloop;
    } tensormaps;

    struct TensorStorage : cute::aligned_struct<128, _1> {
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;

      EpilogueTensorStorage epilogue;
      MainloopTensorStorage mainloop;
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
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    TileSchedulerParams scheduler{};
    KernelHardwareInfo hw_info{};
  };

  enum class WarpCategory : int32_t {
    MMA            = 0,
    Sched          = 1,
    MainloopABLoad = 2,
    MainloopSFLoad = 3,
    Epilogue       = 4,    // Warps [4-8)
    EpilogueLoad   = 8,
    Unused         = 9
  };

  struct IsParticipant {
    uint32_t mma          = false;
    uint32_t sched        = false;
    uint32_t main_ab_load = false;
    uint32_t epi_load     = false;
    uint32_t epilogue     = false;
    uint32_t main_sf_load = false;
    uint32_t unused       = false;
  };

  //
  // Methods
  //

  // Convert to underlying arguments.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    constexpr uint32_t NumEpilogueSubTiles = 1;
    CUTLASS_TRACE_HOST("to_underlying_arguments():");
    ProblemShape problem_shapes = args.problem_shape;
    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (IsGroupedGemmKernel && sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    else if (!IsGroupedGemmKernel && sm_count != 0) {
      CUTLASS_TRACE_HOST("  WARNING: SM100 tile scheduler does not allow for user specified SM counts.\n"
          "  To restrict a kernel's resource usage, consider using CUDA driver APIs instead (green contexts).");
    }
    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    // Epilogue
    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(problem_shapes, args.epilogue, args.hw_info.sm_count);
    workspace_offset = round_nearest(workspace_offset, MinTensorMapWorkspaceAlignment);

    void* mainloop_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveMainloop::get_workspace_size(problem_shapes, args.mainloop, args.hw_info.sm_count);
    workspace_offset = round_nearest(workspace_offset, MinTensorMapWorkspaceAlignment);

    // Tile scheduler
    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset += TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, problem_shapes.get_host_problem_shape(0), args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_offset = round_nearest(workspace_offset, MinTensorMapWorkspaceAlignment);

    TileSchedulerParams scheduler;
    if constexpr (IsGroupedGemmKernel) {
      scheduler = TileScheduler::to_underlying_arguments(
      problem_shapes, TileShape{}, AtomThrShapeMNK{}, ClusterShape{},
      args.hw_info, args.scheduler, scheduler_workspace);
    }
    else {
      scheduler = TileScheduler::to_underlying_arguments(
      problem_shapes.get_host_problem_shape(), TileShape{}, AtomThrShapeMNK{}, ClusterShape{},
      args.hw_info, args.scheduler, scheduler_workspace
      );
    }

    return {
      args.mode,
      problem_shapes,
      CollectiveMainloop::to_underlying_arguments(problem_shapes, args.mainloop, mainloop_workspace, args.hw_info),
      CollectiveEpilogue::to_underlying_arguments(problem_shapes, args.epilogue, epilogue_workspace),
      scheduler,
      args.hw_info
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = true;
    if constexpr (IsGroupedGemmKernel) {
      // Group GEMM currently only supports rank-3 problem shapes
      implementable &= (args.mode == GemmUniversalMode::kGrouped && rank(typename ProblemShape::UnderlyingProblemShape{}) == 3);
    } else {
      implementable &= (args.mode == GemmUniversalMode::kArray && rank(typename ProblemShape::UnderlyingProblemShape{}) == 4);
    }
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements for Ptr Array Gemm or Grouped Gemm.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Mainloop, Epilogue or Scheduler don't meet the requirements for Ptr Array Gemm or Grouped Gemm.\n");
      return implementable;
    }

    if constexpr (IsDynamicCluster) {
      static constexpr int MaxClusterSize = 16;
      implementable &= size(args.hw_info.cluster_shape) <= MaxClusterSize;
      implementable &= size(args.hw_info.cluster_shape_fallback) <= MaxClusterSize;
      implementable &= cutlass::detail::preferred_cluster_can_implement<AtomThrShapeMNK>(args.hw_info.cluster_shape, args.hw_info.cluster_shape_fallback);
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Dynamic Cluster or Preferred Cluster don't meet the requirements for Ptr Array Gemm or Grouped Gemm.\n");
      return implementable;
    }

    constexpr bool IsBlockscaled = !cute::is_void_v<ElementSF>;
    if constexpr (IsBlockscaled) {
      if constexpr (IsDynamicCluster) {
        implementable &= cutlass::detail::preferred_cluster_can_implement<AtomThrShapeMNK>(args.hw_info.cluster_shape, args.hw_info.cluster_shape_fallback);
        // Special cluster check for scale factor multicasts. Due to limited size of scale factors, we can't multicast among
        // more than 4 CTAs
        implementable &= (args.hw_info.cluster_shape.x <= 4 && args.hw_info.cluster_shape.y <= 4 &&
                          args.hw_info.cluster_shape_fallback.x <= 4 && args.hw_info.cluster_shape_fallback.y <= 4);
      }
      else {
        // Special cluster check for scale factor multicasts. Due to limited size of scale factors, we can't multicast among
        // more than 4 CTAs
        implementable &= ((size<0>(ClusterShape{}) <= 4) && (size<1>(ClusterShape{}) <= 4));
      }
    }

    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    constexpr uint32_t NumEpilogueSubTiles = 1;
    size_t workspace_size = 0;

    // Epilogue
    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue, args.hw_info.sm_count);
    workspace_size = round_nearest(workspace_size, MinTensorMapWorkspaceAlignment);

    // Mainloop
    workspace_size += CollectiveMainloop::get_workspace_size(args.problem_shape, args.mainloop, args.hw_info.sm_count);
    workspace_size = round_nearest(workspace_size, MinTensorMapWorkspaceAlignment);

    // Tile scheduler
    workspace_size += TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape.get_host_problem_shape(0), args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_size = round_nearest(workspace_size, MinTensorMapWorkspaceAlignment);

    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    constexpr uint32_t NumEpilogueSubTiles = 1;
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    // Epilogue
    status = CollectiveEpilogue::initialize_workspace(args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue, args.hw_info.sm_count);
    workspace_offset = round_nearest(workspace_offset, MinTensorMapWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    // Mainloop
    status = CollectiveMainloop::initialize_workspace(args.problem_shape, args.mainloop, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveMainloop::get_workspace_size(args.problem_shape, args.mainloop, args.hw_info.sm_count);
    workspace_offset = round_nearest(workspace_offset, MinTensorMapWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    // Tile scheduler
    status = TileScheduler::template initialize_workspace<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, workspace_ptr + workspace_offset, stream, args.problem_shape.get_host_problem_shape(0), args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs, cuda_adapter);
    workspace_offset += TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape.get_host_problem_shape(0), args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_offset = round_nearest(workspace_offset, MinTensorMapWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    // NOTE: cluster_shape here is the major cluster shape, not fallback one
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, params.hw_info.cluster_shape);

    dim3 grid_shape;
    if constexpr (IsGroupedGemmKernel) {
      grid_shape = TileScheduler::get_grid_shape(
        params.scheduler,
        params.problem_shape,
        TileShape{},
        AtomThrShapeMNK{},
        cluster_shape,
        params.hw_info);
    }
    else {
      grid_shape = TileScheduler::get_grid_shape(
        params.scheduler,
        params.problem_shape.get_host_problem_shape(),
        TileShape{},
        AtomThrShapeMNK{},
        cluster_shape,
        params.hw_info);
    }
    return grid_shape;
  }

  static constexpr
  dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

private:

  static constexpr
  CUTLASS_DEVICE
  void set_warpgroup_reg_dealloc() {
    cutlass::arch::warpgroup_reg_dealloc<NonEpilogueWarpRegs>();
  }

  static constexpr
  CUTLASS_DEVICE
  void set_warpgroup_reg_alloc() {
    cutlass::arch::warpgroup_reg_alloc<EpilogueWarpRegs>();
  }

public:

  CUTLASS_DEVICE
  void
  operator() (Params const& params, char* smem_buf) {

    using namespace cute;
    using X = Underscore;

    auto problem_shape = params.problem_shape;

    // Account for more than one epilogue warp
    int warp_idx = canonical_warp_idx_sync();
    WarpCategory warp_category = (warp_idx >= static_cast<int>(WarpCategory::Epilogue) && warp_idx < static_cast<int>(WarpCategory::EpilogueLoad)) ? WarpCategory::Epilogue : 
                                                                                                                     WarpCategory(warp_idx);
    if (warp_idx > static_cast<int>(WarpCategory::EpilogueLoad)) {
      warp_category = WarpCategory::Unused;
    }

    uint32_t lane_predicate = cute::elect_one_sync();
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, cute::cluster_shape());
    int cluster_size = size(cluster_shape);
    uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();
    bool is_first_cta_in_cluster = IsSchedDynamicPersistent ? (cta_rank_in_cluster == 0) : true;
    int cta_coord_v = cta_rank_in_cluster % size<0>(typename TiledMma::AtomThrID{});
    bool is_mma_leader_cta = cta_coord_v == 0;
    constexpr bool has_mma_peer_cta = size(AtomThrShapeMNK{}) == 2;
    [[maybe_unused]] uint32_t mma_peer_cta_rank = has_mma_peer_cta ? cta_rank_in_cluster ^ 1 : cta_rank_in_cluster;

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop(params.mainloop);
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    // Do we load source tensor C or other aux inputs
    bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();
    IsParticipant is_participant = {
      (warp_category == WarpCategory::MMA),                                 // mma
      (warp_category == WarpCategory::Sched) && is_first_cta_in_cluster,    // sched
      (warp_category == WarpCategory::MainloopABLoad),                      // main_ab_load
      (warp_category == WarpCategory::EpilogueLoad) && is_epi_load_needed,  // epi_load
      (warp_category == WarpCategory::Epilogue),                            // epilogue
      (warp_category == WarpCategory::MainloopSFLoad),                      // main_sf_load
      (warp_category == WarpCategory::Unused)                               // empty
    };

    // Mainloop Load pipeline
    typename MainloopABPipeline::Params mainloop_ab_pipeline_params;
    if (WarpCategory::MainloopABLoad == warp_category) {
      mainloop_ab_pipeline_params.role = MainloopABPipeline::ThreadCategory::Producer;
      // Initialize the barrier for TMA load prefetch
    }
    if (WarpCategory::MMA == warp_category) {
      mainloop_ab_pipeline_params.role = MainloopABPipeline::ThreadCategory::Consumer;
    }
    mainloop_ab_pipeline_params.is_leader = lane_predicate && is_mma_leader_cta && is_participant.main_ab_load;
    mainloop_ab_pipeline_params.transaction_bytes = CollectiveMainloop::ABTmaTransactionBytes;
    mainloop_ab_pipeline_params.initializing_warp = 0;
    MainloopABPipeline mainloop_ab_pipeline(shared_storage.pipelines.mainloop.pipeline_ab,
                                       mainloop_ab_pipeline_params,
                                       cluster_shape,
                                       cute::true_type{},   // Perform barrier init
                                       cute::false_type{}); // Delay mask calculation

    // Mainloop SF load pipeline
    typename MainloopSFPipeline::Params mainloop_sf_pipeline_params;
    if (WarpCategory::MainloopSFLoad == warp_category) {
      mainloop_sf_pipeline_params.role = MainloopSFPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::MMA == warp_category) {
      mainloop_sf_pipeline_params.role = MainloopSFPipeline::ThreadCategory::Consumer;
    }
    mainloop_sf_pipeline_params.is_leader = lane_predicate && is_mma_leader_cta && is_participant.main_sf_load;
    mainloop_sf_pipeline_params.transaction_bytes = CollectiveMainloop::SFTransactionBytes;
    mainloop_sf_pipeline_params.initializing_warp = 0;
    MainloopSFPipeline mainloop_sf_pipeline(shared_storage.pipelines.mainloop.pipeline_sf,
                                       mainloop_sf_pipeline_params,
                                       cluster_shape,
                                       cute::true_type{},   // Perform barrier init
                                       cute::false_type{}); // Delay mask calculation

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
    epi_load_pipeline_params.initializing_warp = 4;
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    // Load order barrier
    typename LoadOrderBarrier::Params load_order_barrier_params;
    load_order_barrier_params.group_id = (warp_category == WarpCategory::MainloopABLoad || warp_category == WarpCategory::MainloopSFLoad) ? 0 : 1;
    load_order_barrier_params.group_size = NumMainloopABLoadThreads + NumMainloopSFLoadThreads;
    load_order_barrier_params.initializing_warp = 5;
    LoadOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, load_order_barrier_params);

    // CLC pipeline
    typename CLCPipeline::Params clc_pipeline_params;
    if (WarpCategory::Sched == warp_category) {
      clc_pipeline_params.role = IsSchedDynamicPersistent ? 
        CLCPipeline::ThreadCategory::ProducerConsumer :
        CLCPipeline::ThreadCategory::Producer;
    }
    else {
      clc_pipeline_params.role = CLCPipeline::ThreadCategory::Consumer;
    }

    clc_pipeline_params.initializing_warp = 1;
    clc_pipeline_params.producer_arv_count = 1;

    if constexpr (IsSchedDynamicPersistent) {
      clc_pipeline_params.producer_blockid = 0;
      clc_pipeline_params.consumer_arv_count = NumSchedThreads + cluster_size *
                                                  (NumMainloopABLoadThreads + NumMainloopSFLoadThreads + NumEpilogueThreads + NumMMAThreads);
      if (is_epi_load_needed) {
        clc_pipeline_params.consumer_arv_count += cluster_size * NumEpilogueLoadThreads;
      }
      clc_pipeline_params.transaction_bytes = CLCResponseSize;
    } 
    else {
      clc_pipeline_params.consumer_arv_count = NumMainloopABLoadThreads + NumMainloopSFLoadThreads + NumEpilogueThreads + NumMMAThreads;
      if (is_epi_load_needed) {
        clc_pipeline_params.consumer_arv_count += NumEpilogueLoadThreads;
      }
    }
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
    AccumulatorPipeline accumulator_pipeline(shared_storage.pipelines.accumulator,
                                             accumulator_pipeline_params,
                                             cluster_shape,
                                             cute::true_type{},   // Perform barrier init
                                             cute::false_type{}); // Delay mask calculation

    // CLC throttle pipeline
    typename CLCThrottlePipeline::Params clc_throttle_pipeline_params;
    if constexpr (IsSchedDynamicPersistent) {
      if (WarpCategory::MainloopABLoad == warp_category || WarpCategory::MainloopSFLoad== warp_category) {
        clc_throttle_pipeline_params.role = CLCThrottlePipeline::ThreadCategory::Producer;
      }
      if (WarpCategory::Sched == warp_category) {
        clc_throttle_pipeline_params.role = CLCThrottlePipeline::ThreadCategory::Consumer;
      }
      clc_throttle_pipeline_params.producer_arv_count = NumMainloopSFLoadThreads;
      clc_throttle_pipeline_params.consumer_arv_count = NumSchedThreads;
      clc_throttle_pipeline_params.dst_blockid = 0;
      clc_throttle_pipeline_params.initializing_warp = 3;
    }
    CLCThrottlePipeline clc_throttle_pipeline(shared_storage.pipelines.clc_throttle, clc_throttle_pipeline_params);
    CLCThrottlePipelineState clc_pipe_throttle_consumer_state;
    CLCThrottlePipelineState clc_pipe_throttle_producer_state = cutlass::make_producer_start_state<CLCThrottlePipeline>();

    // Tmem allocator
    TmemAllocator tmem_allocator{};

    // Sync allocation status between MMA and epilogue warps within CTA
    arch::NamedBarrier tmem_allocation_result_barrier(NumMMAThreads + NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
    // Sync deallocation status between MMA warps of peer CTAs
    arch::ClusterBarrier& tmem_deallocation_result_barrier = shared_storage.pipelines.tmem_dealloc;
    [[maybe_unused]] uint32_t dealloc_barrier_phase = 0;
    if constexpr(!IsOverlappingAccum) {
      if (WarpCategory::MMA == warp_category && has_mma_peer_cta && lane_predicate) {
        tmem_deallocation_result_barrier.init(NumMMAThreads);
      }
    }
    else {
      if (WarpCategory::MMA == warp_category && has_mma_peer_cta && lane_predicate) {
        tmem_deallocation_result_barrier.init(NumEpilogueThreads*2);
      }
      else if (WarpCategory::MMA == warp_category && lane_predicate) {
        tmem_deallocation_result_barrier.init(NumEpilogueThreads);
      }
    }

    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer threadblocks in the cluster
    pipeline_init_arrive_relaxed(cluster_size);

    MainloopABPipelineState mainloop_ab_pipe_consumer_state;
    MainloopABPipelineState mainloop_ab_pipe_producer_state = cutlass::make_producer_start_state<MainloopABPipeline>();

    MainloopSFPipelineState mainloop_sf_pipe_consumer_state;
    MainloopSFPipelineState mainloop_sf_pipe_producer_state = cutlass::make_producer_start_state<MainloopSFPipeline>();

    EpiLoadPipelineState epi_load_pipe_consumer_state;
    EpiLoadPipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();

    // epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    EpiStorePipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    CLCPipelineState clc_pipe_consumer_state;
    CLCPipelineState clc_pipe_producer_state = cutlass::make_producer_start_state<CLCPipeline>();

    AccumulatorPipelineState accumulator_pipe_consumer_state;
    AccumulatorPipelineState accumulator_pipe_producer_state = cutlass::make_producer_start_state<AccumulatorPipeline>();

    dim3 block_id_in_cluster = cute::block_id_in_cluster();
    int32_t sm_id = static_cast<int32_t>(cutlass::arch::SmId());

    // Calculate mask after cluster barrier arrival
    mainloop_ab_pipeline.init_masks(cluster_shape);
    mainloop_sf_pipeline.init_masks(cluster_shape);
    accumulator_pipeline.init_masks(cluster_shape);
    // TileID scheduler
    TileScheduler scheduler(&shared_storage.clc_response[0], params.scheduler, block_id_in_cluster);
    typename TileScheduler::WorkTileInfo work_tile_info = scheduler.initial_work_tile_info(cluster_shape);
    auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

    //
    // TMEM "Allocation"
    //
    // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N,ACC_PIPE) where ACC_PIPE=2 so we can double buffer our accumulators for mainloop and epilogue.
    TiledMma tiled_mma;
    ThrMMA cta_mma = tiled_mma.get_slice(cta_coord_v);
    auto acc_shape = partition_shape_C(tiled_mma, take<0,2>(TileShape{}));
    Tensor accumulators = cutlass::detail::make_sm100_accumulator<AccumulatorPipelineStageCount, IsOverlappingAccum>(
        tiled_mma, acc_shape, EpilogueTile{});

    pipeline_init_wait(cluster_size);

    if constexpr (IsGroupedGemmKernel) {
      if (not work_tile_info.is_valid()) {
        // When problem shapes are only on device, the grid launched may be larger than the total number of blocks across groups
        return;
      }
      // In case user wants to engage less SMs than available on device
      sm_id = blockIdx.x + (blockIdx.y * gridDim.x);
    }

    auto problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(work_tile_info.L_idx), 1);

    if (is_participant.main_ab_load) {
      set_warpgroup_reg_dealloc();
      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();

      bool do_load_order_arrive = is_epi_load_needed;
      auto load_inputs = collective_mainloop.load_ab_init(
          problem_shape_MNKL, params.mainloop, shared_storage.tensors.mainloop,
          shared_storage.tensormaps.mainloop,
          params.hw_info.sm_count, sm_id);
      Tensor gA_mkl = get<0>(load_inputs);
      // Fetch a copy of tensormaps for the CTA from Params
      auto input_tensormaps = get<rank(load_inputs) - 1>(load_inputs);

      // Initial batch's tensor address update
      // Even the first tile for a CTA can be from any of the batches.
      // And during initialization of the first TMA descriptor on host, we don't initialize
      bool did_batch_change = true;
      bool requires_clc_query = true;
      // 2cta: 4x4/4x2/2x4 enable the PF
      bool enable_prefetch = shape<0>(AtomThrShapeMNK{}) == 2 and
                             (size<0>(cluster_shape) == 4 and size<1>(cluster_shape) == 4) or 
                             (size<0>(cluster_shape) == 4 and size<1>(cluster_shape) == 2) or
                             (size<0>(cluster_shape) == 2 and size<1>(cluster_shape) == 4);

      do {
        int32_t curr_batch = idx2crd(work_tile_info.L_idx, shape<4>(gA_mkl)); // Usually just returns work_tile_info.L_idx;
        
        if constexpr (IsGroupedGemmKernel) {
          problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(curr_batch), 1);
        }
        if (did_batch_change) {
          collective_mainloop.tensormaps_perform_update_ab(
            shared_storage.tensormaps.mainloop,
            params.mainloop,
            input_tensormaps,
            problem_shape,
            curr_batch
          );
        }

        // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.
        auto k_tile_iter = scheduler.get_k_tile_iterator(work_tile_info, problem_shape_MNKL, CtaShape_MNK{}, shape<3>(gA_mkl));
        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
        auto k_tile_prologue = min(MainloopABPipeline::Stages, k_tile_count);
        // Problem Shape and therefore strides that we construct are [M,N,K,L], but since here for the TMA loads
        // we are managing TMA descriptors to change batches, we need to neglect the L mode 
        auto cta_coord_mnk = append<4>(make_coord(get<0>(cta_coord_mnkl), get<1>(cta_coord_mnkl), get<2>(cta_coord_mnkl)), Int<0>{});

        // Start mainloop prologue loads, arrive on the epilogue residual load barrier, resume mainloop loads
        auto [mainloop_producer_state_next, k_tile_iter_next] = collective_mainloop.load_ab(
          params.mainloop,
          mainloop_ab_pipeline,
          mainloop_ab_pipe_producer_state,
          load_inputs,
          cta_coord_mnk,
          k_tile_iter, k_tile_prologue, 
          did_batch_change,
          enable_prefetch ? k_tile_count : 0
        );
        mainloop_ab_pipe_producer_state = mainloop_producer_state_next;

        if (do_load_order_arrive) {
          load_order_barrier.arrive();
          do_load_order_arrive = false;
        }

        auto [mainloop_producer_state_next_, unused_] = collective_mainloop.load_ab(
          params.mainloop,
          mainloop_ab_pipeline,
          mainloop_ab_pipe_producer_state,
          load_inputs,
          cta_coord_mnk,
          k_tile_iter_next, k_tile_count - k_tile_prologue, 
          false, /* did_batch_change - prologue loads handle tensormap acquire */
          enable_prefetch ? k_tile_count - k_tile_prologue : 0
        );
        mainloop_ab_pipe_producer_state = mainloop_producer_state_next_;

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
        // For subsequent tiles, check if batch changes and therefore, we need tensormap updates
        did_batch_change = curr_batch != idx2crd(work_tile_info.L_idx, shape<4>(gA_mkl));

      } while (work_tile_info.is_valid());
      collective_mainloop.load_tail(mainloop_ab_pipeline, mainloop_ab_pipe_producer_state);

    }

    else if (is_participant.sched) {
      set_warpgroup_reg_dealloc();

      if constexpr (IsSchedDynamicPersistent) {
        // Whether a new CLC query must be performed.
        // See comment below where this variable is updated for a description of
        // why this variable is needed.
        bool requires_clc_query = true;

        do {
          if (requires_clc_query) {
            // Throttle CLC query to mitigate workload imbalance caused by skews among persistent workers.
            clc_throttle_pipeline.consumer_wait(clc_pipe_throttle_consumer_state);
            clc_throttle_pipeline.consumer_release(clc_pipe_throttle_consumer_state);
            ++clc_pipe_throttle_consumer_state;
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

    else if (is_participant.main_sf_load) {
      set_warpgroup_reg_dealloc();
      bool do_load_order_arrive = is_epi_load_needed;
      auto load_inputs = collective_mainloop.load_sf_init(
          problem_shape_MNKL, params.mainloop, shared_storage.tensors.mainloop,
          shared_storage.tensormaps.mainloop,
          params.hw_info.sm_count, sm_id, work_tile_info.L_idx);

      auto gA_mkl = collective_mainloop.get_mkl_shape_tensor(problem_shape_MNKL);
      auto input_tensormaps = get<rank(load_inputs) - 1>(load_inputs);

      // Initial batch's tensor address update
      // Even the first tile for a CTA can be from any of the batches.
      // And during initialization of the first TMA descriptor on host, we don't initialize to the first batch due to that args value being device-only.
      bool did_batch_change = true;

      bool requires_clc_query = true;
      // 2cta: 4x4/4x2/2x4 enable the PF
      bool enable_prefetch = shape<0>(AtomThrShapeMNK{}) == 2 and
                              (size<0>(cluster_shape) == 4 and size<1>(cluster_shape) == 4) or 
                              (size<0>(cluster_shape) == 4 and size<1>(cluster_shape) == 2) or
                              (size<0>(cluster_shape) == 2 and size<1>(cluster_shape) == 4);
      do {
        int32_t curr_batch = idx2crd(work_tile_info.L_idx, shape<4>(gA_mkl)); // Usually just returns work_tile_info.L_idx;
        if constexpr (IsGroupedGemmKernel) {
          problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(curr_batch), 1);
        }
        if (did_batch_change) {
          collective_mainloop.tensormaps_perform_update_sf(
            shared_storage.tensormaps.mainloop,
            params.mainloop,
            input_tensormaps,
            problem_shape,
            curr_batch
          );
        }

        // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.
        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
        auto k_tile_start = TileScheduler::get_work_k_tile_start(work_tile_info);
        auto k_tile_prologue = min(MainloopSFPipeline::Stages/2, k_tile_count);
        auto k_tile_iter = cute::make_coord_iterator(idx2crd(k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl)); // maybe we could use ceil_div(gSFA_mkl, 2);
        auto cta_coord_mnk = append<4>(make_coord(get<0>(cta_coord_mnkl), get<1>(cta_coord_mnkl), get<2>(cta_coord_mnkl)), Int<0>{});
        if constexpr (IsSchedDynamicPersistent) {
          if (is_first_cta_in_cluster && requires_clc_query) {
            clc_throttle_pipeline.producer_acquire(clc_pipe_throttle_producer_state);
            clc_throttle_pipeline.producer_commit(clc_pipe_throttle_producer_state);
            ++clc_pipe_throttle_producer_state;
          }
        }
        // Start mainloop prologue loads, arrive on the epilogue residual load barrier, resume mainloop loads
        auto [mainloop_producer_state_next, k_tile_iter_next] = collective_mainloop.load_sf(
          params.mainloop,
          mainloop_sf_pipeline,
          mainloop_sf_pipe_producer_state,
          load_inputs,
          cta_coord_mnk,
          k_tile_iter, k_tile_prologue, 
          did_batch_change,
          enable_prefetch ? k_tile_count : 0
        );
        mainloop_sf_pipe_producer_state = mainloop_producer_state_next;

        if (do_load_order_arrive) {
          load_order_barrier.arrive();
          do_load_order_arrive = false;
        }

        auto [mainloop_producer_state_next_, unused_] = collective_mainloop.load_sf(
          params.mainloop,
          mainloop_sf_pipeline,
          mainloop_sf_pipe_producer_state,
          load_inputs,
          cta_coord_mnk,
          k_tile_iter_next, k_tile_count - k_tile_prologue, 
          false, /* did_batch_change - prologue loads handle tensormap acquire */
          enable_prefetch ? k_tile_count - k_tile_prologue : 0
        );
        mainloop_sf_pipe_producer_state = mainloop_producer_state_next_;

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
        // For subsequent tiles, check if batch changes and therefore, we need tensormap updates
        did_batch_change = curr_batch != idx2crd(work_tile_info.L_idx, shape<4>(gA_mkl));
      } while (work_tile_info.is_valid());
      collective_mainloop.load_tail(mainloop_sf_pipeline, mainloop_sf_pipe_producer_state);

    }

    else if (is_participant.mma) {
      set_warpgroup_reg_dealloc();
      // Tmem allocation sequence
      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
      __syncwarp();
      tmem_allocation_result_barrier.arrive();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      accumulators.data() = tmem_base_ptr;
      int tmem_non_accumulator_base =  tmem_base_ptr + cutlass::detail::find_tmem_tensor_col_offset(accumulators);
      auto mma_inputs = collective_mainloop.mma_init(params.mainloop,
                                                     shared_storage.tensors.mainloop,
                                                     tmem_non_accumulator_base /*Start SF TMEM allocation after the accumulator*/);

      do {
        if constexpr (IsGroupedGemmKernel) {
          problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
        }
        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
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
        if constexpr (!IsOverlappingAccum) {
          if (is_mma_leader_cta) {
            accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);
          }
        }
        int stage_idx = (IsOverlappingAccum) ? (accumulator_pipe_producer_state.phase() ^ 1) : (accumulator_pipe_producer_state.index());
        Tensor accumulator = accumulators(_,_,_, stage_idx);

        if (is_mma_leader_cta) {
          auto [mainloop_ab_pipe_consumer_state_next, mainloop_sf_pipe_consumer_state_next] = collective_mainloop.mma(
            cute::make_tuple(mainloop_ab_pipeline, mainloop_sf_pipeline, accumulator_pipeline),
            cute::make_tuple(mainloop_ab_pipe_consumer_state, mainloop_sf_pipe_consumer_state, accumulator_pipe_producer_state),
            accumulator,
            mma_inputs,
            cta_coord_mnkl,
            k_tile_count
            );

          mainloop_ab_pipe_consumer_state = mainloop_ab_pipe_consumer_state_next;
          mainloop_sf_pipe_consumer_state = mainloop_sf_pipe_consumer_state_next;
          accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);
        }


        ++accumulator_pipe_producer_state;

        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      } while (work_tile_info.is_valid());

      // Hint on an early release of global memory resources.
      // The timing of calling this function only influences performance,
      // not functional correctness.
      cutlass::arch::launch_dependent_grids();

      // Release the right to allocate before deallocations so that the next CTA can rasterize
      tmem_allocator.release_allocation_lock();

      if constexpr (!IsOverlappingAccum) {
        // Leader MMA waits for leader + peer epilogues to release accumulator stage
        if (is_mma_leader_cta) {
          accumulator_pipeline.producer_tail(accumulator_pipe_producer_state);
        }
        // Signal to peer MMA that entire tmem allocation can be deallocated
        if constexpr (has_mma_peer_cta) {
          // Leader does wait + arrive, follower does arrive + wait
          tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank, not is_mma_leader_cta);
          tmem_deallocation_result_barrier.wait(dealloc_barrier_phase);
          tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank, is_mma_leader_cta);
        }
      }
      else {
        tmem_deallocation_result_barrier.wait(dealloc_barrier_phase);
      }

      // Free entire tmem allocation
      tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }

    else if (is_participant.epi_load) {
      set_warpgroup_reg_dealloc();
      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();

      bool do_load_order_wait = true;
      bool do_tail_load = false;
      int current_wave = 0;

      // Fetch a copy of tensormaps for the CTA from Params
      auto epi_load_tensormap = get<0>(collective_epilogue.load_init(
          params.epilogue, shared_storage.tensormaps.epilogue, params.hw_info.sm_count, sm_id));

      bool did_batch_change = true;
      constexpr bool IsEpiLoad = true;

      do {
        int32_t curr_batch = work_tile_info.L_idx;
        if (did_batch_change) {
          collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
            shared_storage.tensormaps.epilogue,
            params.epilogue,
            epi_load_tensormap,
            problem_shape,
            curr_batch
          );
        }

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
          if (do_load_order_wait) {
            load_order_barrier.wait();
            do_load_order_wait = false;
          }

          if constexpr (IsGroupedGemmKernel) {
            problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(curr_batch), 1);
          }

          bool reverse_epi_n = IsOverlappingAccum && (current_wave % 2 == 0);
          epi_load_pipe_producer_state = collective_epilogue.load<IsOverlappingAccum>(
            epi_load_pipeline,
            epi_load_pipe_producer_state,
            problem_shape_MNKL,
            CtaShape_MNK{},
            cta_coord_mnkl,
            TileShape{},
            TiledMma{},
            shared_storage.tensors.epilogue,
            cute::make_tuple(epi_load_tensormap, did_batch_change),
            reverse_epi_n
          );

          do_tail_load = true;
        }
        current_wave++;

        // Calculate the cta coordinates of the next work tile
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        // For subsequent tiles, check if batch changes and therefore, we need tensormap updates
        did_batch_change = curr_batch != work_tile_info.L_idx;
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
      set_warpgroup_reg_alloc();
      // Wait for tmem allocate here
      tmem_allocation_result_barrier.arrive_and_wait();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      accumulators.data() = tmem_base_ptr;

      auto warp_idx_in_epi = canonical_warp_idx_sync() - static_cast<int>(WarpCategory::Epilogue);
      bool do_tail_store = false;
      // Fetch a copy of tensormaps for the CTA from Params
      auto epi_store_tensormap = get<0>(collective_epilogue.store_init(
          params.epilogue, shared_storage.tensormaps.epilogue, params.hw_info.sm_count, sm_id));
      // Initial batch's tensor address update
      // Even the first tile for a CTA can be from any of the batches.
      // And during initialization of the first TMA descriptor on host, we don't initialize to the first batch due to that args value being device-only.
      bool did_batch_change = true;
      constexpr bool IsEpiLoad = false;
      do {
        int32_t curr_batch = work_tile_info.L_idx;


        if (did_batch_change && warp_idx_in_epi == 0) {
          collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
            shared_storage.tensormaps.epilogue,
            params.epilogue,
            epi_store_tensormap,
            problem_shape,
            curr_batch
          );
        }

        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipe_consumer_state
        );

        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        // Accumulator stage slice after making sure allocation has been performed
        int acc_stage = [&] () {
          if constexpr (IsOverlappingAccum) {
            return accumulator_pipe_consumer_state.phase();
          }
          else {
            return accumulator_pipe_consumer_state.index();
          }
        }();

        // Fusions may need problem shape for the current group
        if constexpr (IsGroupedGemmKernel) {
          problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(curr_batch), 1);
        }

        // Epilogue and write to gD
        //
        auto [load_state_next, store_state_next, acc_state_next] = collective_epilogue.template store<IsOverlappingAccum>(
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
          collective_mainloop.slice_accumulator(accumulators, acc_stage),
          shared_storage.tensors.epilogue,
          cute::make_tuple(epi_store_tensormap, did_batch_change)
        );
        epi_load_pipe_consumer_state = load_state_next;
        epi_store_pipe_producer_state = store_state_next;
        accumulator_pipe_consumer_state = acc_state_next;

        do_tail_store |= TileScheduler::compute_epilogue(work_tile_info, params.scheduler);
        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        // For subsequent tiles, check if batch changes and therefore, we need tensormap updates
        did_batch_change = curr_batch != work_tile_info.L_idx;
      } while (work_tile_info.is_valid());

      if constexpr (IsOverlappingAccum) {
        // Signal to peer MMA that Full TMEM alloc can be deallocated
        if constexpr (has_mma_peer_cta) {
          tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank);
        }
        tmem_deallocation_result_barrier.arrive();
      }

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

    else {
      set_warpgroup_reg_dealloc();
    }

  }


};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
