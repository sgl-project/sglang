/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"

#include "cute/tensor.hpp"

#include "../collective/dispatch_policy_extra.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

// GEMM + Prefetch for the A tensor + (optional) split DMA warps
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
    cute::is_same_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, KernelTmaWarpSpecializedFP8FastAccumWithPrefetchAndSplitDMA> || 
    cute::is_same_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, KernelTmaWarpSpecializedFP8FastAccumWithPrefetch>
    >
>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");
  static constexpr bool IsGdcEnabled = cutlass::arch::IsGdcGloballyEnabled;

  static constexpr bool SplitWarps = cute::is_same_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, KernelTmaWarpSpecializedFP8FastAccumWithPrefetchAndSplitDMA>;

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

  static_assert(cute::is_void_v<TileScheduler_> or cute::is_same_v<TileScheduler_, PersistentScheduler>,
    "TMA warp-specialized kernel does not support specializing the tile scheduler.");
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileScheduler_, ArchTag, TileShape, ClusterShape>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;

  // Kernel level shared memory storage
  struct SharedStorage {
    // Mainloop and epilogue don't use smem concurrently since kernel is non-persistent, so we can use a union
    union TensorStorage {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    } tensors;

    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using PrefetcherPipelineStorage = typename CollectiveMainloop::PrefetcherPipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) PrefetcherPipelineStorage prefetcher;
    } pipelines;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{})) + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

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
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;
    auto problem_shape = args.problem_shape;
    if constexpr (detail::Has_SwapAB_v<CollectiveMainloop>) {
      // swap M/N
      get<0>(problem_shape) = get<1>(args.problem_shape);
      get<1>(problem_shape) = get<0>(args.problem_shape);
    }
    return {
      args.mode,
      problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace)
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

  static
  size_t
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  static
  cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    auto cluster_shape = ClusterShape{};
    auto tile_shape = TileShape{};
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    return TileScheduler::get_tiled_cta_shape_mnl(
        problem_shape_MNKL, tile_shape, cluster_shape);
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

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
#  define ENABLE_SM90_KERNEL_LEVEL 1
#endif

// Any Tensor Op MMA Atom in the WGMMA ISA is arch conditional to sm90a.
#if ! defined(ENABLE_SM90_KERNEL_LEVEL)
    printf("ERROR : Arch conditional MMA instruction used without targeting sm90a compute capability. Aborting.\n");
#else

    enum class WarpGroupRole {
      Producer = 0,
      Consumer = 1,
    };
    // Split mode: use Warp0 to load NK and epilogue, Warp2 to load MK.
    // Non-split mode: use Warp0 to load MK, NK and epilogue, Warp2 is unused.
    // Both modes use Warp1 to prefetch.
    enum class ProducerWarpRole {
      Warp0 = 0,
      PrefetchMK = 1,
      Warp2 = 2,
      UnusedWarp = 3
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

    // Mainloop Load pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    if (warp_group_role == WarpGroupRole::Producer && (
          producer_warp_role == ProducerWarpRole::Warp0 ||
          producer_warp_role == ProducerWarpRole::Warp2)) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
      mainloop_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.num_consumers = NumThreadsPerWarpGroup;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, mainloop_pipeline_params, ClusterShape{});
    bool should_prefetch = params.mainloop.prefetch_ratio > 0;
    using PrefetcherPipeline = typename CollectiveMainloop::PrefetcherPipeline;
    typename PrefetcherPipeline::Params prefetcher_pipeline_params;
    prefetcher_pipeline_params.num_prefetchers = 1;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::PrefetchMK) {
      prefetcher_pipeline_params.should_prefetch = should_prefetch;
      prefetcher_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes_mk;
    }
    PrefetcherPipeline prefetcher_pipeline(shared_storage.pipelines.prefetcher, prefetcher_pipeline_params);

    // Epilogue Load pipeline
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Warp0) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
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
        // Non-prefetcher warps arrive and wait,
        // Prefetcher warp can go ahead without waiting.
        cute::cluster_arrive_relaxed();
        if (warp_group_role != WarpGroupRole::Producer ||
            producer_warp_role != ProducerWarpRole::PrefetchMK) {
          cute::cluster_wait();
        }
        return [] () {};
      }
      else {
        // __syncthreads() but only for non prefetcher warps
        if (should_prefetch) {

          // Use a named barrier to let the prefetcher warp start loading into the L2
          // without waiting to sync with all other warps.
          // All other warps need to sync because the mainloop pipeline init
          // should be visible to all of them.
          // Prefetcher has its own barriers, and the only warps it would need to sync
          // with would be the DMA warps.
          using ClusterSyncWithPrefetchBarrier = typename cutlass::arch::NamedBarrier;
          auto prefetcher_arrive_barrier = ClusterSyncWithPrefetchBarrier(
              blockDim.x * blockDim.y * blockDim.z,
              /*id*/ 0);
          // Prefetcher warp doesn't arrive on this barrier.
          auto cluster_arrive_barrier = ClusterSyncWithPrefetchBarrier(
              blockDim.x * blockDim.y * blockDim.z - NumThreadsPerWarp,
              /*id*/ 1);

          if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::PrefetchMK) {
            __syncwarp();
            prefetcher_arrive_barrier.arrive();
          }
          else if (warp_group_role == WarpGroupRole::Producer) {
            prefetcher_arrive_barrier.arrive_and_wait();
            cluster_arrive_barrier.arrive_and_wait();
          }
          else {
            prefetcher_arrive_barrier.arrive();
            cluster_arrive_barrier.arrive_and_wait();
          }
        } else {
        __syncthreads();
        }
        return [] () {};
      }
    } ();

    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Optionally append 1s until problem shape is rank-4 in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)
    TiledMma tiled_mma;

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

    // Compute m_coord, n_coord, and l_coord with their post-tiled shapes
    auto m_coord = idx2crd(int(blockIdx.x), shape<2>(gA_mkl));
    auto n_coord = idx2crd(int(blockIdx.y), shape<2>(gB_nkl));
    auto l_coord = idx2crd(int(blockIdx.z), shape<4>(gB_nkl));
    auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

    // Get pipeline iterators and increments from tensor shapes
    auto k_tile_iter  = cute::make_coord_iterator(shape<3>(gA_mkl));
    auto k_tile_count = size<3>(gA_mkl);

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();

    if (warp_group_role == WarpGroupRole::Producer) {
      if (producer_warp_role == ProducerWarpRole::Warp0) {
        if constexpr(SplitWarps) {
          collective_mainloop.load_NK(
            params.mainloop,
            mainloop_pipeline,
            prefetcher_pipeline,
            mainloop_pipe_producer_state,
            gB_nkl,
            blk_coord,
            k_tile_iter, k_tile_count,
            lane_idx,
            block_rank_in_cluster,
            shared_storage.tensors.mainloop
          );
        }
        else {
          collective_mainloop.load(
            params.mainloop,
            mainloop_pipeline,
            prefetcher_pipeline,
            mainloop_pipe_producer_state,
            gA_mkl, gB_nkl,
            blk_coord,
            k_tile_iter, k_tile_count,
            lane_idx,
            block_rank_in_cluster,
            shared_storage.tensors.mainloop
          );
        }
        // Update starting mainloop pipeline state for the pipeline drain
        mainloop_pipe_producer_state.advance(k_tile_count);
        // Make sure mainloop consumer has been waited upon before issuing epilogue load
        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);

        if (collective_epilogue.is_producer_load_needed()) {
          // Ensure warp is converged before issuing epilogue loads
          __syncwarp();
          epi_load_pipe_producer_state = collective_epilogue.load(
            epi_load_pipeline,
            epi_load_pipe_producer_state,
            problem_shape_MNKL,
            blk_shape,
            blk_coord,
            tiled_mma,
            lane_idx,
            shared_storage.tensors.epilogue
          );
          collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
        }
      }
      else if (SplitWarps && producer_warp_role == ProducerWarpRole::Warp2) {
        collective_mainloop.load_MK(
          params.mainloop,
          mainloop_pipeline,
          prefetcher_pipeline,
          mainloop_pipe_producer_state,
          gA_mkl,
          blk_coord,
          k_tile_iter, k_tile_count,
          lane_idx,
          block_rank_in_cluster,
          shared_storage.tensors.mainloop
        );
        // Update starting mainloop pipeline state for the pipeline drain
        mainloop_pipe_producer_state.advance(k_tile_count);
        // Make sure mainloop consumer has been waited upon before issuing epilogue load
        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);
      } else if (producer_warp_role == ProducerWarpRole::PrefetchMK && should_prefetch) {
        collective_mainloop.prefetch_MK(
          params.mainloop,
          prefetcher_pipeline,
          mainloop_pipe_producer_state,
          gA_mkl,
          blk_coord,
          k_tile_iter, k_tile_count,
          lane_idx,
          block_rank_in_cluster,
          shared_storage.tensors.mainloop
        );
      }
    }
    else if (warp_group_role == WarpGroupRole::Consumer) {
      Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));                 // (MMA,MMA_M,MMA_N)

      collective_mainloop.mma(
        mainloop_pipeline,
        mainloop_pipe_consumer_state,
        accumulators,
        k_tile_count,
        warp_group_thread_idx,
        shared_storage.tensors.mainloop,
        params.mainloop
      );

      // Make sure the math instructions are done and free buffers before entering the epilogue
      collective_mainloop.mma_tail(
        mainloop_pipeline,
        mainloop_pipe_consumer_state,
        k_tile_count
      );

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

      collective_epilogue.store_tail(
        epi_load_pipeline,
        epi_load_pipe_consumer_state_next,
        epi_store_pipeline,
        epi_store_pipe_producer_state_next
      );
    }
#endif
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
