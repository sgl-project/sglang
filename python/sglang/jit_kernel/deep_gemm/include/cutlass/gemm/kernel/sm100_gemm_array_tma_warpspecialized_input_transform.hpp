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

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/detail/cluster.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/fast_math.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  cute::enable_if_t<
    cutlass::detail::is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, 
                                KernelPtrArrayTmaWarpSpecializedInputTransformSm100>>>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(rank(typename ProblemShape::UnderlyingProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");
  static constexpr bool IsGdcEnabled = cutlass::arch::IsGdcGloballyEnabled;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;

  // Get Blk and Scheduling tile shapes
  using CtaShape_MNK = typename CollectiveMainloop::CtaShape_MNK;
  using AtomThrShapeMNK = typename CollectiveMainloop::AtomThrShapeMNK;

  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using InternalStrideA = typename CollectiveMainloop::InternalStrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using InternalStrideB = typename CollectiveMainloop::InternalStrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  static_assert(ArchTag::kMinComputeCapability >= 100);

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
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
  // TileID scheduler
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileScheduler_, ArchTag, CtaShape_MNK, ClusterShape, SchedulerPipelineStageCount>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;

  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;
  static constexpr uint32_t MinTensorMapWorkspaceAlignment = 64;

  // Warp specialization thread count per threadblock
  static constexpr uint32_t NumSchedThreads           = NumThreadsPerWarp;                             // 1 warp
  static constexpr uint32_t NumMMAThreads             = NumThreadsPerWarp;                             // 1 warp
  static constexpr uint32_t NumMainloopLoadThreads    = NumThreadsPerWarp;                             // 1 warp
  static constexpr uint32_t NumEpilogueLoadThreads    = NumThreadsPerWarp;                             // 1 warp
  static constexpr uint32_t NumEpilogueThreads        = CollectiveMainloop::NumAccumThreads;           // 4 warps
  static constexpr uint32_t NumEpilogueWarps          = NumEpilogueThreads / NumThreadsPerWarp;
  static constexpr uint32_t NumTransformationThreads  = CollectiveMainloop::NumTransformationThreads;  // 4 warps

  static constexpr uint32_t MaxThreadsPerBlock = NumSchedThreads +
                                                 NumMainloopLoadThreads + NumMMAThreads +
                                                 NumEpilogueLoadThreads +
                                                 NumEpilogueThreads + NumTransformationThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  static constexpr uint32_t AccumulatorPipelineStageCount = DispatchPolicy::Schedule::AccumulatorPipelineStageCount;
  static constexpr cutlass::gemm::detail::KernelInputTransformType InputTransformType = DispatchPolicy::InputTransformType;
  static constexpr uint32_t NumFixupBarriers = 1;
  static constexpr uint32_t CLCResponseSize = sizeof(typename TileScheduler::CLCResponse);

  // Transfer registers from regular warps to Accum warps
  static constexpr uint32_t GenericRegisterRequirement = 152;
  static constexpr uint32_t AccumRegisterRequirement = 200;

  // Pipeline and pipeline state types
  using Load2TransformPipeline = typename CollectiveMainloop::Load2TransformPipeline;
  using Load2TransformPipelineState = typename CollectiveMainloop::Load2TransformPipelineState;

  using Transform2MmaPipeline = typename CollectiveMainloop::Transform2MmaPipeline;
  using Transform2MmaPipelineState = typename CollectiveMainloop::Transform2MmaPipelineState;

  using Mma2AccumPipeline = typename CollectiveMainloop::Mma2AccumPipeline;
  using Mma2AccumPipelineState = typename CollectiveMainloop::Mma2AccumPipelineState;

  using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
  using EpiLoadPipelineState = typename CollectiveEpilogue::LoadPipelineState;

  using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
  using EpiStorePipelineState = typename CollectiveEpilogue::StorePipelineState;

  using LoadOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;


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

  // Kernel level shared memory storage
  struct SharedStorage {
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;
      using LoadOrderBarrierStorage = typename LoadOrderBarrier::SharedStorage;
      using CLCPipelineStorage = typename CLCPipeline::SharedStorage;
      using CLCThrottlePipelineStorage = typename CLCThrottlePipeline::SharedStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) LoadOrderBarrierStorage load_order;
      alignas(16) CLCPipelineStorage clc;
      alignas(16) CLCThrottlePipelineStorage clc_throttle;
      alignas(16) arch::ClusterBarrier tmem_dealloc;
      alignas(16) arch::ClusterBarrier epilogue_throttle;
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

  // NOTE: MMA must be on the 0th thread of the warp-group, so make sure pipeline leader is on MainloopLoad warp
  enum class WarpCategory : int32_t {
    MMA           = 0,
    Sched         = 1,
    MainloopLoad  = 2,
    EpilogueLoad  = 3,
    Epilogue      = 4,
    // Transformation starts at 256 thread alignment
    Transformation    = 8
  };

  struct IsParticipant {
    uint32_t mma            = false;
    uint32_t sched          = false;
    uint32_t main_load      = false;
    uint32_t epi_load       = false;
    uint32_t epilogue       = false;
    uint32_t transformation = false;
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    static constexpr uint32_t NumEpilogueSubTiles = 1;
    CUTLASS_TRACE_HOST("to_underlying_arguments():");
    ProblemShape problem_shapes = args.problem_shape;
    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
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

    return {
      args.mode,
      problem_shapes,
      CollectiveMainloop::to_underlying_arguments(problem_shapes, args.mainloop, mainloop_workspace, args.hw_info),
      CollectiveEpilogue::to_underlying_arguments(problem_shapes, args.epilogue, epilogue_workspace),
      TileScheduler::to_underlying_arguments(
        problem_shapes.get_host_problem_shape(), TileShape{}, AtomThrShapeMNK{}, ClusterShape{},
        args.hw_info, args.scheduler, scheduler_workspace
      )
      ,args.hw_info
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kArray && rank(typename ProblemShape::UnderlyingProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);

    if constexpr (IsDynamicCluster) {
      static constexpr int MaxClusterSize = 16;
      implementable &= size(args.hw_info.cluster_shape) <= MaxClusterSize;
      implementable &= size(args.hw_info.cluster_shape_fallback) <= MaxClusterSize;
      implementable &= cutlass::detail::preferred_cluster_can_implement<AtomThrShapeMNK>(args.hw_info.cluster_shape, args.hw_info.cluster_shape_fallback);
    }

    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    static constexpr uint32_t NumEpilogueSubTiles = 1;
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
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    static constexpr uint32_t NumEpilogueSubTiles = 1;

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
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, params.hw_info.cluster_shape);
    return TileScheduler::get_grid_shape(
        params.scheduler,
        params.problem_shape.get_host_problem_shape(),
        TileShape{},
        AtomThrShapeMNK{},
        cluster_shape,
        params.hw_info
       );
}

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator() (Params const& params, char* smem_buf) {

    using namespace cute;
    using X = Underscore;

    auto problem_shape = params.problem_shape;

    // Account for multiple epilogue and transformation warps
    int warp_idx = canonical_warp_idx_sync();
    WarpCategory warp_category = warp_idx < static_cast<int>(WarpCategory::Epilogue)       ? WarpCategory(warp_idx)
                               : warp_idx < static_cast<int>(WarpCategory::Transformation) ? WarpCategory::Epilogue
                                                                                           : WarpCategory::Transformation;
    int thread_idx          = int(threadIdx.x);
    int thread_idx_in_warp  = thread_idx % 32;
    uint32_t lane_predicate = cute::elect_one_sync();
    int cta_rank_in_cluster = cute::block_rank_in_cluster();
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, cute::cluster_shape());
    int cluster_size                = size(cluster_shape);
    bool is_first_cta_in_cluster    = IsSchedDynamicPersistent ? (cta_rank_in_cluster == 0) : true;
    bool is_mma_leader_cta          = (cta_rank_in_cluster % size<0>(TiledMma{}) == 0);
    // Even if this variable is unused, shape_div still performs useful compile-time checks.
    [[maybe_unused]] auto mma_leader_ctas = size(shape_div(cluster_shape, AtomThrShapeMNK{}));
    constexpr bool has_mma_peer_cta = size(AtomThrShapeMNK{}) == 2;
    uint32_t mma_peer_cta_rank = has_mma_peer_cta ? cta_rank_in_cluster ^ 1 : cta_rank_in_cluster;

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    CollectiveMainloop collective_mainloop(params.mainloop, cluster_shape, cta_rank_in_cluster);
    CollectiveEpilogue collective_epilogue{params.epilogue, shared_storage.tensors.epilogue};

    bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();
    IsParticipant is_participant = {
      (warp_category == WarpCategory::MMA),                                               // mma
      (warp_category == WarpCategory::Sched) && (is_first_cta_in_cluster),                // sched
      (warp_category == WarpCategory::MainloopLoad),                                      // main_load
      (warp_category == WarpCategory::EpilogueLoad) && is_epi_load_needed,                // epi_load
      (warp_category == WarpCategory::Epilogue),                                          // epilogue
      (warp_category == WarpCategory::Transformation)                                     // transformation
    };

    // MainloopLoad <--> Transformation Pipeline
    typename Load2TransformPipeline::Params load2transform_pipeline_params;
    if (warp_category == WarpCategory::MainloopLoad) {
      load2transform_pipeline_params.role = Load2TransformPipeline::ThreadCategory::Producer;
    }
    else if (warp_category == WarpCategory::Transformation) {
      load2transform_pipeline_params.role = Load2TransformPipeline::ThreadCategory::Consumer;
    }
    load2transform_pipeline_params.is_leader = (thread_idx_in_warp == 0);
    load2transform_pipeline_params.num_consumers = NumTransformationThreads;
    load2transform_pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytes;
    load2transform_pipeline_params.initializing_warp = 0;
    Load2TransformPipeline load2transform_pipeline(shared_storage.pipelines.mainloop.load2transform_pipeline,
                                                   load2transform_pipeline_params,
                                                   cluster_shape,
                                                   cute::true_type{},  // Perform barrier init
                                                   cute::false_type{}  // Delay mask calculation
                                                   );

    Load2TransformPipelineState load2transform_pipeline_consumer_state;
    Load2TransformPipelineState load2transform_pipeline_producer_state = cutlass::make_producer_start_state<Load2TransformPipeline>();

    // Transformation <--> MMA pipeline
    typename Transform2MmaPipeline::Params transform2mma_pipeline_params;
    if (warp_category == WarpCategory::Transformation) {
      transform2mma_pipeline_params.role = Transform2MmaPipeline::ThreadCategory::Producer;
    }
    else if (warp_category == WarpCategory::MMA) {
      transform2mma_pipeline_params.role = Transform2MmaPipeline::ThreadCategory::Consumer;
    }
    transform2mma_pipeline_params.consumer_arv_count = 1;
    transform2mma_pipeline_params.producer_arv_count = size(AtomThrShapeMNK{}) * NumTransformationThreads;
    transform2mma_pipeline_params.initializing_warp = 2;
    Transform2MmaPipeline transform2mma_pipeline(shared_storage.pipelines.mainloop.transform2mma_pipeline,
                                                 transform2mma_pipeline_params,
                                                 cluster_shape,
                                                 cute::true_type{},  // Perform barrier init
                                                 cute::false_type{}  // Delay mask calculation
                                                 );

    Transform2MmaPipelineState transform2mma_pipeline_consumer_state;
    Transform2MmaPipelineState transform2mma_pipeline_producer_state = cutlass::make_producer_start_state<Transform2MmaPipeline>();

    // MMA <--> Accumulator pipeline
    typename Mma2AccumPipeline::Params mma2accum_pipeline_params;
    if (warp_category == WarpCategory::MMA) {
      mma2accum_pipeline_params.role = Mma2AccumPipeline::ThreadCategory::Producer;
    }
    else if (warp_category == WarpCategory::Epilogue) {
      mma2accum_pipeline_params.role = Mma2AccumPipeline::ThreadCategory::Consumer;
    }
    mma2accum_pipeline_params.producer_arv_count = 1;
    mma2accum_pipeline_params.consumer_arv_count = size(AtomThrShapeMNK{}) * NumEpilogueThreads;
    mma2accum_pipeline_params.initializing_warp = 6;
    Mma2AccumPipeline mma2accum_pipeline(shared_storage.pipelines.mainloop.mma2accum_pipeline, 
                                         mma2accum_pipeline_params,
                                         cluster_shape,
                                         cute::true_type{},  // Perform barrier init
                                         cute::false_type{}  // Delay mask calculation
                                         );

    Mma2AccumPipelineState mma2accum_pipeline_consumer_state;
    Mma2AccumPipelineState mma2accum_pipeline_producer_state = cutlass::make_producer_start_state<Mma2AccumPipeline>();

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
    load_order_barrier_params.group_id = (warp_category == WarpCategory::MainloopLoad) ? 0 : 1;
    load_order_barrier_params.group_size = 1;
    load_order_barrier_params.initializing_warp = 5;
    LoadOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, load_order_barrier_params);

    EpiLoadPipelineState epi_load_pipe_consumer_state;
    EpiLoadPipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();

    // epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    EpiStorePipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    // CLC pipeline
    // Operates Scheduling Warp <--> All Warps
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
                                                  (NumMainloopLoadThreads + NumEpilogueThreads + NumMMAThreads +
                                                   NumTransformationThreads);
      if (is_epi_load_needed) {
        clc_pipeline_params.consumer_arv_count += cluster_size * NumEpilogueLoadThreads;
      }
      clc_pipeline_params.transaction_bytes = CLCResponseSize;
    } 
    else {
      clc_pipeline_params.consumer_arv_count = NumMainloopLoadThreads + NumEpilogueThreads + NumMMAThreads +
                                               NumTransformationThreads;
      if (is_epi_load_needed) {
        clc_pipeline_params.consumer_arv_count += NumEpilogueLoadThreads;
      }
    }
    
    CLCPipeline clc_pipeline = [&]() {
      if constexpr (IsSchedDynamicPersistent) {
        return CLCPipeline(shared_storage.pipelines.clc, clc_pipeline_params, cluster_shape);
      }
      else {
        return CLCPipeline(shared_storage.pipelines.clc, clc_pipeline_params);
      }
    }();

    CLCPipelineState clc_pipeline_consumer_state;
    CLCPipelineState clc_pipeline_producer_state = cutlass::make_producer_start_state<CLCPipeline>();

    // CLC throttle pipeline
    typename CLCThrottlePipeline::Params clc_throttle_pipeline_params;
    if constexpr (IsSchedDynamicPersistent) {
      if (WarpCategory::MainloopLoad == warp_category) {
        clc_throttle_pipeline_params.role = CLCThrottlePipeline::ThreadCategory::Producer;
      }
      if (WarpCategory::Sched == warp_category) {
        clc_throttle_pipeline_params.role = CLCThrottlePipeline::ThreadCategory::Consumer;
      }
      clc_throttle_pipeline_params.producer_arv_count = NumMainloopLoadThreads;
      clc_throttle_pipeline_params.consumer_arv_count = NumSchedThreads;
      clc_throttle_pipeline_params.dst_blockid = 0;
      clc_throttle_pipeline_params.initializing_warp = 3;
    }
    CLCThrottlePipeline clc_throttle_pipeline(shared_storage.pipelines.clc_throttle, clc_throttle_pipeline_params);
    CLCThrottlePipelineState clc_pipe_throttle_consumer_state;
    CLCThrottlePipelineState clc_pipe_throttle_producer_state = cutlass::make_producer_start_state<CLCThrottlePipeline>();

    TmemAllocator tmem_allocator{};

    // Sync allocation status between transform, MMA, and epilogue warps within CTA
    arch::NamedBarrier tmem_allocation_result_barrier(NumTransformationThreads + NumMMAThreads + NumEpilogueThreads,
                                                          cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
    // Sync deallocation status between MMA warps of peer CTAs
    arch::ClusterBarrier& tmem_deallocation_result_barrier = shared_storage.pipelines.tmem_dealloc;
    [[maybe_unused]] uint32_t dealloc_barrier_phase = 0;
    if (WarpCategory::MMA == warp_category && has_mma_peer_cta && lane_predicate) {
      tmem_deallocation_result_barrier.init(NumMMAThreads);
    }

    // Initialize smem barrier for prologue throttling. Epilogue warps are stalled until the prologue finishes.
    arch::ClusterBarrier& epilogue_throttle_barrier = shared_storage.pipelines.epilogue_throttle;
    if (WarpCategory::MMA == warp_category && lane_predicate) {
      epilogue_throttle_barrier.init(                          NumMMAThreads +
                                    (is_first_cta_in_cluster ? NumSchedThreads : 0) +
                                                               NumMainloopLoadThreads +
                                    (is_epi_load_needed      ? NumEpilogueLoadThreads : 0) +
                                                               NumTransformationThreads);
    }

    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer threadblocks in the cluster
    pipeline_init_arrive_relaxed(cluster_size);

    dim3 block_id_in_cluster = cute::block_id_in_cluster();

    // Calculate mask after cluster barrier arrival
    load2transform_pipeline.init_masks(cluster_shape, block_id_in_cluster);
    transform2mma_pipeline.init_masks(cluster_shape);
    mma2accum_pipeline.init_masks(cluster_shape);

    // TileID scheduler
    TileScheduler scheduler(&shared_storage.clc_response[0], params.scheduler, block_id_in_cluster);
    typename TileScheduler::WorkTileInfo work_tile_info = scheduler.initial_work_tile_info(cluster_shape);

    auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

    // Optionally append 1s until problem shape is rank-4 in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape.get_problem_shape(work_tile_info.L_idx), 1);

    // Allocate accumulators
    auto acc_shape = collective_mainloop.partition_accumulator_shape();

    // NOTE: we can assume the tmem buf starts at zero since we allocate all tmem in this kernel
    auto bulk_tmem = TiledMma::make_fragment_C(append(acc_shape,
                                                      Int<AccumulatorPipelineStageCount>{}));

    // Tile transform inputs now to get the k tile count
    auto transform_inputs = collective_mainloop.transform_init(params.mainloop, problem_shape_MNKL, bulk_tmem, shared_storage.tensors.mainloop);
    Tensor gA_mkl = get<0>(transform_inputs);

    // Synchronization call. Blocks until barriers are initialized in shared memory.
    pipeline_init_wait(cluster_size);

    if (is_participant.main_load) {
      // Register reconfiguration
      arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();

      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();

      bool do_load_order_arrive = is_epi_load_needed;
      auto load_inputs = collective_mainloop.load_init(
          problem_shape_MNKL, params.mainloop, shared_storage.tensors.mainloop,
          params.hw_info.sm_count, static_cast<int32_t>(cutlass::arch::SmId()));
      Tensor gA_mkl = get<0>(load_inputs);
      // Fetch a copy of tensormaps for the CTA from Params
      auto input_tensormaps = get<rank(load_inputs) - 1>(load_inputs);

      // Initial batch's tensor address update
      // Even the first tile for a CTA can be from any of the batches.
      // And during initialization of the first TMA descriptor on host, we don't initialize to the first batch due to
      // that args value being device-only.
      bool did_batch_change = true;

      // Signal the epilogue warps to proceed once the prologue is complete
      epilogue_throttle_barrier.arrive();
      bool requires_clc_query = true;

      do {
        int32_t curr_batch = idx2crd(work_tile_info.L_idx, shape<4>(gA_mkl)); // Usually just returns work_tile_info.L_idx;
        if (did_batch_change) {
          collective_mainloop.tensormaps_perform_update(
            shared_storage.tensormaps.mainloop,
            params.mainloop,
            input_tensormaps,
            curr_batch,
            lane_predicate
          );
          // Ensure warp is converged before issuing tensormap fence release
          __syncwarp();
          // Entire warp must do this (i.e. it's aligned)
          collective_mainloop.tensormaps_cp_fence_release(shared_storage.tensormaps.mainloop, input_tensormaps);
        }

        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        auto k_tile_iter = scheduler.get_k_tile_iterator(work_tile_info, problem_shape_MNKL, CtaShape_MNK{}, shape<3>(gA_mkl));
        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
        auto k_tile_prologue = min(Load2TransformPipeline::Stages, k_tile_count);

        // Problem Shape and therefore strides that we construct are [M,N,K,L], but since here for the TMA loads
        // we are managing TMA descriptors to change batches, we need to neglect the L mode
        auto cta_coord_mnk = append<4>(make_coord(get<0>(cta_coord_mnkl), get<1>(cta_coord_mnkl), get<2>(cta_coord_mnkl)), Int<0>{});

        if constexpr (IsSchedDynamicPersistent) {
          if (is_first_cta_in_cluster && requires_clc_query) {
            clc_throttle_pipeline.producer_acquire(clc_pipe_throttle_producer_state);
            clc_throttle_pipeline.producer_commit(clc_pipe_throttle_producer_state);
            ++clc_pipe_throttle_producer_state;
          }
        }

        // Check to see if tensormaps have been replaced in gmem
        if (did_batch_change) {
          collective_mainloop.tensormaps_fence_acquire(input_tensormaps);
        }
        // Start mainloop prologue loads, arrive on the epilogue residual load barrier, resume mainloop loads
        if (lane_predicate) {
          auto [load2transform_pipeline_producer_state_next, k_tile_iter_next] = collective_mainloop.load(
            params.mainloop,
            load2transform_pipeline,
            load2transform_pipeline_producer_state,
            load_inputs,
            cta_coord_mnk,
            k_tile_iter, k_tile_prologue
          );
          load2transform_pipeline_producer_state = load2transform_pipeline_producer_state_next;

          if (do_load_order_arrive) {
            load_order_barrier.arrive();
            do_load_order_arrive = false;
          }

          auto [load2transform_pipeline_producer_state_next_, unused_] = collective_mainloop.load(
            params.mainloop,
            load2transform_pipeline,
            load2transform_pipeline_producer_state,
            load_inputs,
            cta_coord_mnk,
            k_tile_iter_next, k_tile_count - k_tile_prologue
          );
          load2transform_pipeline_producer_state = load2transform_pipeline_producer_state_next_;
        }
        
        // Sync warp to prevent non-participating threads entering next wave early
        __syncwarp();

        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipeline_consumer_state
        );
        requires_clc_query = increment_pipe;
        if (increment_pipe) {
          ++clc_pipeline_consumer_state;
        }
        work_tile_info = next_work_tile_info;
        // For subsequent tiles, check if batch changes and therefore, we need tensormap updates
        did_batch_change = curr_batch != idx2crd(work_tile_info.L_idx, shape<4>(gA_mkl));
      } while (work_tile_info.is_valid());
      if (lane_predicate) {
        load2transform_pipeline.producer_tail(load2transform_pipeline_producer_state);
      }

    }

    else if (is_participant.transformation) {
      // Register reconfiguration
      arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();

      // Signal the epilogue warps to proceed once the prologue is complete
      epilogue_throttle_barrier.arrive();

      // Wait for tmem allocation
      tmem_allocation_result_barrier.arrive_and_wait_unaligned();

      do {
        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
        auto k_tile_start = TileScheduler::get_work_k_tile_start(work_tile_info);
        auto k_tile_iter = cute::make_coord_iterator(idx2crd(k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl));
        auto [load2transform_pipeline_consumer_state_next, transform2mma_pipeline_producer_state_next] = collective_mainloop.transform(
          load2transform_pipeline,
          load2transform_pipeline_consumer_state,
          transform2mma_pipeline,
          transform2mma_pipeline_producer_state,
          bulk_tmem,
          transform_inputs,
          k_tile_iter, k_tile_count
        );
        transform2mma_pipeline_producer_state = transform2mma_pipeline_producer_state_next;
        load2transform_pipeline_consumer_state = load2transform_pipeline_consumer_state_next;

        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipeline_consumer_state
        );
        work_tile_info = next_work_tile_info;

        if (increment_pipe) {
          ++clc_pipeline_consumer_state;
        }
      } while (work_tile_info.is_valid());

      transform2mma_pipeline.producer_tail(transform2mma_pipeline_producer_state);
    }

    else if (is_participant.sched) {
      // Register reconfiguration
      arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();

      // Signal the epilogue warps to proceed once the prologue is complete
      epilogue_throttle_barrier.arrive();

      // Grouped GEMM uses static tile scheduler
      if constexpr (IsSchedDynamicPersistent) {
        // Whether a new CLC query must be performed.
        // See comment below where this variable is updated for a description of
        // why this variable is needed.
        bool requires_clc_query = true;

        cutlass::arch::wait_on_dependent_grids();
        do {
          if (requires_clc_query) {
            // Throttle CLC query to mitigate workload imbalance caused by skews among persistent workers.
            clc_throttle_pipeline.consumer_wait(clc_pipe_throttle_consumer_state);
            clc_throttle_pipeline.consumer_release(clc_pipe_throttle_consumer_state);
            ++clc_pipe_throttle_consumer_state;

            // Query next clcID and update producer state
            clc_pipeline_producer_state = scheduler.advance_to_next_work(
              clc_pipeline, 
              clc_pipeline_producer_state
            );
          }

          // Fetch next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
            work_tile_info,
            clc_pipeline,
            clc_pipeline_consumer_state
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
            ++clc_pipeline_consumer_state;
          }

          work_tile_info = next_work_tile_info;
        } while (work_tile_info.is_valid());
        clc_pipeline.producer_tail(clc_pipeline_producer_state);
      }
      else {
        cutlass::arch::wait_on_dependent_grids();
        do {
          auto [next_work_tile_info, increment_pipe] = scheduler.advance_to_next_work(clc_pipeline, clc_pipeline_producer_state);
          work_tile_info = next_work_tile_info;
          if (increment_pipe) {
            ++clc_pipeline_producer_state;
          }
        } while (work_tile_info.is_valid());
        clc_pipeline.producer_tail(clc_pipeline_producer_state);
      }
    }

    else if (is_participant.mma) {
      // Register reconfiguration
      arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();

      // Allocate all tmem
      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
      __syncwarp();
      tmem_allocation_result_barrier.arrive();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      bulk_tmem.data() = tmem_base_ptr;

      auto mma_input_operands = collective_mainloop.mma_init(bulk_tmem, shared_storage.tensors.mainloop);

      // Signal the epilogue warps to proceed once the prologue is complete
      epilogue_throttle_barrier.arrive();

      do {
        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipeline_consumer_state
        );
        work_tile_info = next_work_tile_info;

        if (increment_pipe) {
          ++clc_pipeline_consumer_state;
        }

        if (is_mma_leader_cta) {
          auto [transform2mma_pipeline_consumer_state_next, mma2accum_pipeline_producer_state_next] = collective_mainloop.mma(
            transform2mma_pipeline,
            transform2mma_pipeline_consumer_state,
            mma2accum_pipeline,
            mma2accum_pipeline_producer_state,
            bulk_tmem,
            mma_input_operands,
            k_tile_count
          );
          // Advance the mm2accum pipe
          transform2mma_pipeline_consumer_state = transform2mma_pipeline_consumer_state_next;
          mma2accum_pipeline_producer_state = mma2accum_pipeline_producer_state_next;
        }
      } while (work_tile_info.is_valid());

      // leader MMA waits for leader + peer epilogues to release accumulator stage
      if (is_mma_leader_cta) {
        mma2accum_pipeline.producer_tail(mma2accum_pipeline_producer_state);
      }

      // Hint on an early release of global memory resources.
      // The timing of calling this function only influences performance,
      // not functional correctness.
      cutlass::arch::launch_dependent_grids();

      // Signal to peer MMA that stage can be deallocated
      if constexpr (has_mma_peer_cta) {
        // Leader does wait + arrive, follower does arrive + wait
        tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank, not is_mma_leader_cta);
        tmem_deallocation_result_barrier.wait(dealloc_barrier_phase);
        tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank, is_mma_leader_cta);
      }

      // Tmem deallocation sequence
      tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }

    else if (is_participant.epi_load) {
      // Register reconfiguration
      arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();

      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();

      bool do_load_order_wait = true;
      bool do_tail_load = false;
      // Fetch a copy of tensormaps for the CTA from Params
      auto epi_load_tensormap = get<0>(collective_epilogue.load_init(
          params.epilogue, shared_storage.tensormaps.epilogue, params.hw_info.sm_count, static_cast<int32_t>(cutlass::arch::SmId())));
      // Initial batch's tensor address update
      // Even the first tile for a CTA can be from any of the batches.
      // And during initialization of the first TMA descriptor on host, we don't initialize to the first batch due to that args value being device-only.
      bool did_batch_change = true;
      constexpr bool IsEpiLoad = true;

      // Signal the epilogue warps to proceed once the prologue is complete
      epilogue_throttle_barrier.arrive();

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
        __syncwarp();
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipeline_consumer_state
        );
        work_tile_info = next_work_tile_info;

        if (increment_pipe) {
          ++clc_pipeline_consumer_state;
        }

        if (compute_epilogue) {
          if (do_load_order_wait) {
            load_order_barrier.wait();
            do_load_order_wait = false;
          }

          epi_load_pipe_producer_state = collective_epilogue.load(
            epi_load_pipeline,
            epi_load_pipe_producer_state,
            problem_shape_MNKL,
            CtaShape_MNK{},
            cta_coord_mnkl,
            TileShape{},
            TiledMma{},
            shared_storage.tensors.epilogue,
            cute::make_tuple(epi_load_tensormap, did_batch_change)
          );

          do_tail_load = true;
        }

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
      // Register reconfiguration
      arch::warpgroup_reg_alloc<AccumRegisterRequirement>();

      // Throttle the epilogue warps to improve prologue performance
      static constexpr int epilogue_throttle_phase_bit = 0;
      epilogue_throttle_barrier.wait(epilogue_throttle_phase_bit);

      // Wait for tmem allocation
      tmem_allocation_result_barrier.arrive_and_wait_unaligned();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      bulk_tmem.data() = tmem_base_ptr;

      auto accum_inputs = collective_mainloop.accum_init(bulk_tmem, typename CollectiveEpilogue::CopyOpT2R{}, typename CollectiveEpilogue::EpilogueTile{});
      bool do_tail_store = false;
      auto warp_idx_in_epi = canonical_warp_idx_sync() - static_cast<int>(WarpCategory::Epilogue);
      // Fetch a copy of tensormaps for the CTA from Params
      auto epi_store_tensormap = get<0>(collective_epilogue.store_init(
          params.epilogue, shared_storage.tensormaps.epilogue, params.hw_info.sm_count, static_cast<int32_t>(cutlass::arch::SmId())));
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
          clc_pipeline_consumer_state
        );

        if (increment_pipe) {
          ++clc_pipeline_consumer_state;
        }

        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});

        if constexpr (InputTransformType == cutlass::gemm::detail::KernelInputTransformType::FastF32) {
          auto [mma2accum_pipeline_consumer_state_next,tTR_rGlobAcc] = collective_mainloop.accum(
            accum_inputs,
            mma2accum_pipeline,
            mma2accum_pipeline_consumer_state,
            k_tile_count);

          // Check to see if tensormaps have been replaced in gmem
          if (did_batch_change && warp_idx_in_epi == 0) {
            collective_epilogue.template tensormaps_fence_acquire<IsEpiLoad>(epi_store_tensormap);
          }
          auto [load_state_next, store_state_next] = collective_epilogue.store(
            epi_load_pipeline,
            epi_load_pipe_consumer_state,
            epi_store_pipeline,
            epi_store_pipe_producer_state,
            problem_shape_MNKL,
            CtaShape_MNK{},
            cta_coord_mnkl,
            TileShape{},
            TiledMma{},
            tTR_rGlobAcc,
            shared_storage.tensors.epilogue,
            epi_store_tensormap,
            get<0>(accum_inputs) // tiled_t2r
          );
          
          do_tail_store |= TileScheduler::compute_epilogue(work_tile_info, params.scheduler);

          epi_load_pipe_consumer_state = load_state_next;
          epi_store_pipe_producer_state = store_state_next;
          // Advance the mm2accum pipe
          mma2accum_pipeline_consumer_state = mma2accum_pipeline_consumer_state_next;
        }
        // Complex kernels use a collective epilogue
        else {
          mma2accum_pipeline.consumer_wait(mma2accum_pipeline_consumer_state);

          // Accumulators (real and imag)
          Tensor accumulators = bulk_tmem(_,_,_,_,mma2accum_pipeline_consumer_state.index()); // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N)

          //
          // Epilogue and write to gD
          //
          // The tile scheduler and current work are passed into the collective epilogue to
          // support fixup operations needed by split-/stream-K. These operations are pushed
          // to the collective layer so that they can reuse the TMEM -> RF copy performed
          // at the collective layer.
          auto [mma2accum_pipeline_state_next] = collective_epilogue(
            mma2accum_pipeline,
            mma2accum_pipeline_consumer_state,
            problem_shape_MNKL,
            CtaShape_MNK{},
            cta_coord_mnkl,
            accumulators,
            shared_storage.tensors.epilogue
          );
          // Advance the mm2accum pipe
          mma2accum_pipeline_consumer_state = mma2accum_pipeline_state_next;
        }

        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        // For subsequent tiles, check if batch changes and therefore, we need tensormap updates
        did_batch_change = curr_batch != work_tile_info.L_idx;
      } while (work_tile_info.is_valid());

      // Only perform a tail load if one of the work units processed performed
      // an epilogue load. An example of a case in which a tail load should not be
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
      // Register reconfiguration
      arch::warpgroup_reg_dealloc<GenericRegisterRequirement>();
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
