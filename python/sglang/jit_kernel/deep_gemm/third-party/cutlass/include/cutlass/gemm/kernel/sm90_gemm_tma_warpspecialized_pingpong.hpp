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
#include "cutlass/workspace.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/fast_math.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/gemm_universal_decl.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"

#include "cute/tensor.hpp"
#include "cutlass/arch/grid_dependency_control.h"

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
  cute::enable_if_t<cute::is_base_of_v<KernelTmaWarpSpecializedPingpong, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");
  static constexpr bool IsGdcEnabled = cutlass::arch::IsGdcGloballyEnabled;

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
  static_assert(ArchTag::kMinComputeCapability >= 90);

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static_assert(!cute::is_same_v<TileScheduler_, StreamKScheduler>, "Ping-pong kernel does not currently support stream-K scheduler.");
  static constexpr uint32_t TileSchedulerPipelineStageCount = DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
                                          TileSchedulerTag, 
                                          ArchTag, 
                                          TileShape,
                                          ClusterShape,
                                          TileSchedulerPipelineStageCount
                                          >::Scheduler;

  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  using TileSchedulerPipeline = typename TileScheduler::Pipeline;
  using TileSchedulerPipelineState = typename TileSchedulerPipeline::PipelineState;
  using TileSchedulerStorage = typename TileScheduler::SharedStorage;

  using TileSchedulerThrottlePipeline = typename TileScheduler::ThrottlePipeline;
  using TileSchedulerThrottlePipelineState = typename TileSchedulerThrottlePipeline::PipelineState;

  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;

  // Warp specialization thread count per threadblock
  static constexpr uint32_t NumSchedThreads        = NumThreadsPerWarp;      // 1 warp
  static constexpr uint32_t NumMainloopLoadThreads = NumThreadsPerWarp;      // 1 warp
  static constexpr uint32_t NumEpilogueLoadThreads = NumThreadsPerWarp;      // 1 warp for C
  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = 2;
  static constexpr uint32_t NumProducerThreads = CollectiveMainloop::NumProducerThreadEvents;
  static constexpr uint32_t NumMMAThreads = size(TiledMma{});                 // 4 warp 
  static constexpr uint32_t MaxThreadsPerBlock = NumMMAThreads * NumMmaWarpGroups + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr bool     IsMainloopAuxiliaryLoadNeeded = detail::HasAuxiliaryLoad_v<typename CollectiveMainloop::DispatchPolicy>;
  
  static_assert(NumMMAThreads == 128, "Pingpong kernel must have TiledMMA operating using 128 threads.");
  static_assert(MaxThreadsPerBlock == 384, "Pingpong kernel must have 384 threads in total.");

  /// Register requirement for Load and Math WGs
  static constexpr int RegsPerThread =
    (size<0>(TileShape{}) * size<1>(TileShape{}) * sizeof(ElementAccumulator))
    / (NumMMAThreads * sizeof(uint32_t));
  static constexpr bool HeavyRegisterPressure = RegsPerThread >= 208;
  static constexpr uint32_t LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;
  static constexpr uint32_t MmaRegisterRequirement = !HeavyRegisterPressure ? 232 : 240;

  // 1 stage ordered sequence between mainloop and epilogue producer load threads
  using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;

  // Order Sequence barrier with two stages: one for Mainloop and one for Epilogue
  static constexpr uint32_t StagesPerMathWarpGroup = 2;
  using MathWarpGroupOrderBarrier = cutlass::OrderedSequenceBarrier<
    StagesPerMathWarpGroup, NumMmaWarpGroups>;
  using MathWarpGroupOrderBarrierSharedStorage =
    cutlass::PipelineDetail::OrderedSequenceBarrierSharedStorage<
      MathWarpGroupOrderBarrier::SequenceDepth,
      MathWarpGroupOrderBarrier::SequenceLength>;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;
      using MathWarpGroupOrderBarrierStorage = MathWarpGroupOrderBarrierSharedStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) MathWarpGroupOrderBarrierStorage math_wg_order;
      alignas(16) typename LoadWarpOrderBarrier::SharedStorage load_order;
    } pipelines;
    
    alignas(16) TileSchedulerStorage scheduler;

    struct TensorStorage : cute::aligned_struct<128, _1> {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      EpilogueTensorStorage epilogue;
      MainloopTensorStorage mainloop;
    } tensors;
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
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    (void) workspace;
    auto problem_shape = args.problem_shape;
    if constexpr (detail::Has_SwapAB_v<CollectiveMainloop>) {
      // swap M/N
      get<0>(problem_shape) = get<1>(args.problem_shape);
      get<1>(problem_shape) = get<0>(args.problem_shape);
    }
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    // Get maximum number of clusters that could co-exist on the target device
    int max_active_clusters = args.hw_info.max_active_clusters;
    if (max_active_clusters <= 0) {
      max_active_clusters = 0;
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid max cluster count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the max_active_clusters.");
    }
    else {
      CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid cluster count to " << max_active_clusters);
    }

    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count, max_active_clusters};

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});

    return {
      args.mode,
      problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, mainloop_workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, epilogue_workspace),
      hw_info,
      TileScheduler::to_underlying_arguments(
        problem_shape_MNKL, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles
      )
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && cute::rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);

    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;

    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    workspace_size += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    static constexpr uint32_t NumEpilogueSubTiles = 1;
    static constexpr uint32_t NumAccumulatorMtxs = 1;

    status = CollectiveEpilogue::initialize_workspace(args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = TileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(
      args.scheduler, workspace_ptr + workspace_offset, stream, args.problem_shape, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles, NumAccumulatorMtxs, cuda_adapter);
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    // Given device SM count, set grid size s.t. we do not launch more thread blocks than we can run concurrently
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN ? TileScheduler::RasterOrderOptions::AlongN : TileScheduler::RasterOrderOptions::AlongM;
    return TileScheduler::get_grid_shape(params.scheduler, params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, args);
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

#  if (defined(__CUDA_ARCH_FEAT_SM90_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) ||\
      CUDA_ARCH_CONDITIONAL_OR_FAMILY(1200) || CUDA_ARCH_CONDITIONAL_OR_FAMILY(1210))
#    define ENABLE_SM90_KERNEL_LEVEL 1
#  endif

// Any Tensor Op MMA Atom in the ISA is arch conditional.
#if ! defined(ENABLE_SM90_KERNEL_LEVEL)
    printf("ERROR : Arch conditional MMA instruction used without targeting appropriate compute capability. Aborting.\n");
#else

    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    enum class WarpGroupRole {
      Producer = 0,
      Consumer0 = 1,
      Consumer1 = 2
    };
    enum class ProducerWarpRole {
      Mainloop = 0,
      Warp1 = 1,
      Epilogue = 2,
      MainloopAux = 3
    };

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = int(threadIdx.x);
    int lane_idx = canonical_lane_idx();
    int warp_idx = canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }


    // TileScheduler pipeline
    typename TileSchedulerPipeline::Params scheduler_pipeline_params;
    typename TileSchedulerThrottlePipeline::Params scheduler_throttle_pipeline_params;
    if constexpr (IsSchedDynamicPersistent) { 
      if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Warp1) {
        scheduler_pipeline_params.role = TileSchedulerPipeline::ThreadCategory::ProducerConsumer;
      }
      else {
        scheduler_pipeline_params.role = TileSchedulerPipeline::ThreadCategory::Consumer;
      }
      scheduler_pipeline_params.producer_blockid = 0;
      scheduler_pipeline_params.producer_arv_count = 1;
      scheduler_pipeline_params.consumer_arv_count = NumSchedThreads + NumMainloopLoadThreads + NumMMAThreads;

      CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);
      bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();

      if (is_epi_load_needed) {
        scheduler_pipeline_params.consumer_arv_count += NumEpilogueLoadThreads;
      } 
      scheduler_pipeline_params.transaction_bytes = sizeof(typename TileScheduler::CLCResponse);

      scheduler_throttle_pipeline_params.producer_arv_count = NumMainloopLoadThreads;
      scheduler_throttle_pipeline_params.consumer_arv_count = NumSchedThreads;
      scheduler_throttle_pipeline_params.dst_blockid = 0;
      if (warp_group_role == WarpGroupRole::Producer &&
          producer_warp_role == ProducerWarpRole::Warp1) {
        scheduler_throttle_pipeline_params.role =
            TileSchedulerThrottlePipeline::ThreadCategory::Consumer;
      }
      // set role when it is for DMA warp in Mainloop
      else if (warp_group_role == WarpGroupRole::Producer &&
               producer_warp_role == ProducerWarpRole::Mainloop) {
        scheduler_throttle_pipeline_params.role =
            TileSchedulerThrottlePipeline::ThreadCategory::Producer;
      }
    }
    TileSchedulerPipeline scheduler_pipeline(shared_storage.scheduler.pipeline(), scheduler_pipeline_params);
    TileSchedulerPipelineState scheduler_pipe_consumer_state;

    TileSchedulerThrottlePipeline scheduler_throttle_pipeline(shared_storage.scheduler.throttle_pipeline(), scheduler_throttle_pipeline_params);
    TileSchedulerThrottlePipelineState scheduler_pipe_throttle_consumer_state;
    TileSchedulerThrottlePipelineState scheduler_pipe_throttle_producer_state = cutlass::make_producer_start_state<TileSchedulerThrottlePipeline>();

    // Mainloop Load pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && (producer_warp_role == ProducerWarpRole::Mainloop 
        || producer_warp_role == ProducerWarpRole::MainloopAux)) {
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
    params_math_wg_order_barrier.group_id = canonical_warp_group_idx() - static_cast<int>(WarpGroupRole::Consumer0);
    params_math_wg_order_barrier.group_size = NumThreadsPerWarpGroup; // Number of threads / participants in a group
    MathWarpGroupOrderBarrier math_wg_order_barrier(shared_storage.pipelines.math_wg_order, params_math_wg_order_barrier);

    // Initialize starting pipeline states for the collectives
    // Epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    // For the DMA Load (producer) we start with an opposite phase
    // i.e., we skip all waits since we know that the buffer is indeed empty
    PipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
    PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    auto cluster_wait_fn = [&] () {
      // We need this to guarantee that the Pipeline init is visible
      // To all producers and consumer thread blocks in the Cluster
      if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        return [] () { cute::cluster_wait(); };
      }
      else {
        __syncthreads();
        return [] () {}; // do nothing
      }
    } ();

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    TiledMma tiled_mma;
    auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)

    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    // Prepare and partition the input tensors. Expects a tuple of tensors where:
    // get<0>(load_inputs) is the tma tensor A after local tiling so that it has shape (BLK_M,BLK_K,m,k,l)
    // get<1>(load_inputs) is the tma tensor B after local tiling so that it has shape (BLK_N,BLK_K,n,k,l)
    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
    static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

    // Extract out partitioned A and B.
    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    // Get pipeline stage increments from tensor shapes
    auto k_tile_count = size<3>(gA_mkl);
    auto c_tile_count = CollectiveEpilogue::get_load_pipe_increment(blk_shape);
    auto d_tile_count = CollectiveEpilogue::get_store_pipe_increment(blk_shape);

    TileScheduler scheduler{params.scheduler};
    if constexpr (IsSchedDynamicPersistent) {
      scheduler.set_data_ptr(shared_storage.scheduler.data());
    }

    if (warp_group_role == WarpGroupRole::Consumer1) {

      if constexpr (not IsSchedDynamicPersistent) {
        // Advance 2nd Math WG to the next work tile for the startup
        scheduler.advance_to_next_work();
      }

      // Advance 2nd Math WG pipeline states to the end of 1st Math WG
      mainloop_pipe_consumer_state.advance(k_tile_count);
      epi_load_pipe_consumer_state.advance(c_tile_count);
      epi_store_pipe_producer_state.advance(d_tile_count);
    }
    auto work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();

    if (warp_group_role == WarpGroupRole::Producer) {
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
    
      // Scheduler Producer Warp
      if (producer_warp_role == ProducerWarpRole::Warp1) {
        if constexpr (IsSchedDynamicPersistent) { 
          bool requires_clc_query = true;
          TileSchedulerPipelineState scheduler_pipe_producer_state = cutlass::make_producer_start_state<TileSchedulerPipeline>();

          while (work_tile_info.is_valid()) {
            
            if (requires_clc_query) {

              // Throttle CLC query to mitigate workload imbalance caused by skews among persistent workers.
              scheduler_throttle_pipeline.consumer_wait(scheduler_pipe_throttle_consumer_state);
              scheduler_throttle_pipeline.consumer_release(scheduler_pipe_throttle_consumer_state);
              ++scheduler_pipe_throttle_consumer_state;

              // Query next work tile
              scheduler_pipe_producer_state = scheduler.advance_to_next_work(scheduler_pipeline, scheduler_pipe_producer_state);
            }

            // Fetch next work tile
            auto [next_work_tile_info, increment_pipe] = 
              scheduler.fetch_next_work(
                  work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);
            
            work_tile_info = next_work_tile_info;
            requires_clc_query = increment_pipe;
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }
          }

          // Terminal condition - if work_tile_info is end-of-grid, produce an extra invalid tile
          scheduler_pipeline.producer_acquire(scheduler_pipe_producer_state);
          scheduler.store_invalid_response(scheduler_pipe_producer_state); // Push invalid tile to smem
          scheduler_pipeline.producer_commit(scheduler_pipe_producer_state); // Manual completion of transaction
          ++scheduler_pipe_producer_state;

          auto [next_work_tile_info, increment_pipe] = 
            scheduler.fetch_next_work(
                work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);

          scheduler_pipeline.producer_tail(scheduler_pipe_producer_state);
        } 
      } // Scheduler Producer Warp End  
      else
      
      // Mainloop Producer Warp
      if (producer_warp_role == ProducerWarpRole::Mainloop) {
        // Ensure that the prefetched kernel does not touch
        // unflushed global memory prior to this instruction
        cutlass::arch::wait_on_dependent_grids();
        bool do_load_order_arrive = true;
        bool requires_clc_query = true;
        while (work_tile_info.is_valid()) {
          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

          auto k_tile_iter  = cute::make_coord_iterator(shape<3>(gA_mkl));

          if (requires_clc_query) {
            scheduler_throttle_pipeline.producer_acquire(scheduler_pipe_throttle_producer_state);
            scheduler_throttle_pipeline.producer_commit(scheduler_pipe_throttle_producer_state);
            ++scheduler_pipe_throttle_producer_state;
          }

          collective_mainloop.load(
            params.mainloop,
            mainloop_pipeline,
            mainloop_pipe_producer_state,
            load_inputs,
            blk_coord,
            k_tile_iter, k_tile_count,
            lane_idx,
            block_rank_in_cluster,
            shared_storage.tensors.mainloop
          );
          // Update starting pipeline state for the next tile
          mainloop_pipe_producer_state.advance(k_tile_count);

          // Signal for the epilogue load warp to begin
          if (do_load_order_arrive) {
            load_order_barrier.arrive();
            do_load_order_arrive = false;
          }

          if constexpr (IsSchedDynamicPersistent) {  
            // Get next work tile
            auto [next_work_tile_info, increment_pipe] =
              scheduler.fetch_next_work(
                  work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);

            work_tile_info = next_work_tile_info;
            requires_clc_query = increment_pipe;
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }
          }
          else {
          // Get next work tile
          scheduler.advance_to_next_work();
          work_tile_info = scheduler.get_current_work();
          }
        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);

        if constexpr (IsSchedDynamicPersistent) {  
          auto [next_work_tile_info, increment_pipe] = 
            scheduler.fetch_next_work(
                work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);
        }
        
      } // Mainloop Producer Warp End

      else if (producer_warp_role == ProducerWarpRole::MainloopAux) {
        if constexpr (IsMainloopAuxiliaryLoadNeeded) {
          // Ensure that the prefetched kernel does not touch
          // unflushed global memory prior to this instruction
          cutlass::arch::wait_on_dependent_grids();
          while (work_tile_info.is_valid()) {
            // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
            auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
            auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
            auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
            auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

            auto k_tile_iter = cute::make_coord_iterator(shape<3>(gA_mkl));
            collective_mainloop.load_auxiliary(
              params.mainloop,
              mainloop_pipeline,
              mainloop_pipe_producer_state,
              load_inputs,
              blk_coord,
              k_tile_iter, k_tile_count,
              lane_idx,
              block_rank_in_cluster,
              shared_storage.tensors.mainloop
            );
            // Update starting pipeline state for the next tile
            mainloop_pipe_producer_state.advance(k_tile_count);

            scheduler.advance_to_next_work();
            work_tile_info = scheduler.get_current_work();
          } // Scheduler work fetch loop

          // Make sure all Consumer Warp Groups have been waited upon
          collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);

          if constexpr (IsSchedDynamicPersistent) {  
            auto [next_work_tile_info, increment_pipe] = 
              scheduler.fetch_next_work(
                work_tile_info,
                scheduler_pipeline,
                scheduler_pipe_consumer_state
              );
          }
          
        }
      }

      // Epilogue Producer Warp
      else if (producer_warp_role == ProducerWarpRole::Epilogue && collective_epilogue.is_producer_load_needed()) {

        // Ensure that the prefetched kernel does not touch
        // unflushed global memory prior to this instruction
        cutlass::arch::wait_on_dependent_grids();

        bool do_load_order_wait = true;
        while (work_tile_info.is_valid()) {
          if (do_load_order_wait) {
            load_order_barrier.wait();
            do_load_order_wait = false;
          }

          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

          epi_load_pipe_producer_state =
          collective_epilogue.load(
            epi_load_pipeline,
            epi_load_pipe_producer_state,
            problem_shape_MNKL,
            blk_shape,
            blk_coord,
            tiled_mma,
            lane_idx,
            shared_storage.tensors.epilogue
          );

          if constexpr (IsSchedDynamicPersistent) {  
            // Get next work tile
            auto [next_work_tile_info, increment_pipe] = 
              scheduler.fetch_next_work(
                  work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);

            work_tile_info = next_work_tile_info;
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }
          }
          else {
          // Get next work tile
          scheduler.advance_to_next_work();
          work_tile_info = scheduler.get_current_work();
          }
        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);

        if constexpr (IsSchedDynamicPersistent) {  
          auto [next_work_tile_info, increment_pipe] = 
            scheduler.fetch_next_work(
                work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);
        }
      } // Epilogue Producer Warp End
    } // Producer Warp Group End

    else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

      #ifdef CUTLASS_ENABLE_GDC_FOR_SM90
      // It is possible to have work tiles start off invalid,
      // so we have to check that first.
      if (not work_tile_info.is_valid()) {
        // Hint on an early release of global memory resources.
        // The timing of calling this function only influences performance,
        // not functional correctness.
        cutlass::arch::launch_dependent_grids();

        return;
      }
      #endif
      
      if constexpr (IsSchedDynamicPersistent) {
        // Consumer0's initial tile is static. It starts consuming the 2nd tile.
        if (warp_group_role == WarpGroupRole::Consumer0) {
            ++scheduler_pipe_consumer_state;
        } 

        if (warp_group_role == WarpGroupRole::Consumer1) {
          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = 
            scheduler.fetch_next_work(
                work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);

          work_tile_info = next_work_tile_info;
          if (increment_pipe) {
            ++scheduler_pipe_consumer_state;
            ++scheduler_pipe_consumer_state;
          }
        } 
      }

      while (work_tile_info.is_valid()) {
        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

        // Allocate the accumulators for the (M,N) blk_shape
        Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));               // (MMA,MMA_M,MMA_N)

        // Order two Math WG's MMA one after the other, helps hide Epilogue
        math_wg_order_barrier.wait();

        collective_mainloop.mma(
          mainloop_pipeline,
          mainloop_pipe_consumer_state,
          accumulators,
          k_tile_count,
          warp_group_thread_idx,
          shared_storage.tensors.mainloop,
          params.mainloop
        );

        // Cue for next Math WG's MMA to start
        math_wg_order_barrier.arrive();

        // Make sure the math instructions are done and free buffers before entering the epilogue
        collective_mainloop.mma_tail(
          mainloop_pipeline,
          mainloop_pipe_consumer_state,
          k_tile_count
        );
        // Update starting mainloop pipeline state for the next tile
        mainloop_pipe_consumer_state.advance(k_tile_count * NumMmaWarpGroups);

        #ifdef CUTLASS_ENABLE_GDC_FOR_SM90
        if (scheduler.is_last_tile(work_tile_info, NumMmaWarpGroups)) {
          // Hint on an early release of global memory resources.
          // The timing of calling this function only influences performance,
          // not functional correctness.
          cutlass::arch::launch_dependent_grids();

        }
        #endif

        // Order two Math WG's Epilogue one after the other
        math_wg_order_barrier.wait();

        // Epilogue and write to gD
        auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next] =
        collective_epilogue.store(
          epi_load_pipeline,
          epi_load_pipe_consumer_state,
          epi_store_pipeline,
          epi_store_pipe_producer_state,
          problem_shape_MNKL,
          blk_shape,
          blk_coord,
          accumulators,
          tiled_mma,
          warp_group_thread_idx,
          shared_storage.tensors.epilogue
        );

        // TMA store pipeline wait is only visible to TMA-issuing warp, so for multiple-consumer kernels
        // we need to wait for all TMA stores to complete before issuing consumer order barrier arrives
        // to ensure next math consumer doesn't overwrite smem of in-flight TMA stores of current consumer.
        auto [epi_load_pipe_consumer_state_next_, epi_store_pipe_producer_state_next_] =
        collective_epilogue.store_tail(
          epi_load_pipeline,
          epi_load_pipe_consumer_state_next,
          epi_store_pipeline,
          epi_store_pipe_producer_state_next
        );

        // Update starting load/store pipeline states for the next tile
        // state has already been incremented by 1 tile in collective calls, advance once again for ping pong
        epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next_;
        epi_store_pipe_producer_state = epi_store_pipe_producer_state_next_;
        epi_load_pipe_consumer_state.advance(c_tile_count);
        epi_store_pipe_producer_state.advance(d_tile_count);

        // Cue for next Math WG's Epilogue to start
        math_wg_order_barrier.arrive();

        if constexpr (IsSchedDynamicPersistent) {  
          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = 
            scheduler.fetch_next_work(
                work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);

          work_tile_info = next_work_tile_info;
          if (increment_pipe) {
            ++scheduler_pipe_consumer_state;
            ++scheduler_pipe_consumer_state;
          }
        }
        else {
        // Get next work tile
        scheduler.advance_to_next_work(NumMmaWarpGroups);
        work_tile_info = scheduler.get_current_work();
        }
      } // Scheduler work fetch loop
    } // Consumer Warp Groups End
#endif
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
