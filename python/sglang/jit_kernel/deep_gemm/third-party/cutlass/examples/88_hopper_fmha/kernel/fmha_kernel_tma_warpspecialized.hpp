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
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/arch/arch.h"

#include "../kernel/fmha_options.hpp"

namespace cutlass::fmha::kernel {

using namespace cute;

template<
  class CollectiveMainloop,
  class CollectiveEpilogue,
  class TileScheduler,
  class... Options
>
struct FmhaKernelTmaWarpSpecialized {

  // Options
  static constexpr bool kIsEpilogueLocked = find_option_t<Tag::kIsEpilogueLocked, false_type, Options...>::value;
  static constexpr bool kLoadsQSeparately = find_option_t<Tag::kLoadsQSeparately, false_type, Options...>::value;


  static const int NumLoadWarpGroups = 1;
  static constexpr int NumMmaWarpGroups = CollectiveMainloop::NumMmaWarpGroups;

  using TileShape = typename CollectiveMainloop::TileShape;
  using ClusterShape = typename CollectiveMainloop::ClusterShape;

  using MainloopPipelineOuter = typename CollectiveMainloop::MainloopPipelineQ;
  using MainloopPipelineInner = typename CollectiveMainloop::MainloopPipeline;
  using MainloopPipelineReducer = cutlass::PipelineAsync<2>;

  static constexpr uint32_t StagesPerMathWarpGroup = 2;
  using MathWarpGroupOrderBarrier = cutlass::OrderedSequenceBarrier<
    StagesPerMathWarpGroup, NumMmaWarpGroups>;

  struct TensorStorageStruct {
    typename CollectiveMainloop::SharedStorage mainloop;
    typename CollectiveEpilogue::TensorStorage epilogue[NumMmaWarpGroups];
  };
  union TensorStorageUnion {
    typename CollectiveMainloop::SharedStorage mainloop;
    typename CollectiveEpilogue::TensorStorage epilogue[NumMmaWarpGroups];
  };
  using TensorStorage = std::conditional_t<CollectiveMainloop::kIsPersistent, TensorStorageStruct, TensorStorageUnion>;

  struct SharedStorage {
    TensorStorage tensors;

    using PipelineStorageInner = typename MainloopPipelineInner::SharedStorage;
    using PipelineStorageOuter = typename MainloopPipelineOuter::SharedStorage;
    using PipelineStorageReducer = typename MainloopPipelineReducer::SharedStorage;

    alignas(16) PipelineStorageInner pipeline_storage_inner;
    alignas(16) PipelineStorageOuter pipeline_storage_outer;
    alignas(16) PipelineStorageReducer pipeline_storage_reducer;

    using MathWarpGroupOrderBarrierStorage = typename MathWarpGroupOrderBarrier::SharedStorage;
    alignas(16) MathWarpGroupOrderBarrierStorage math_wg_order;

    alignas(16) cutlass::arch::ClusterBarrier load_warp_barrier;

    using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;
    alignas(16) EpiLoadPipelineStorage epi_load;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  using ProblemShape = cute::tuple<int, int, int, int, int>;

  struct Arguments {
    ProblemShape problem_size;
    typename CollectiveMainloop::Arguments mainloop;
    typename CollectiveEpilogue::Arguments epilogue;
    KernelHardwareInfo hw_info;
  };

  struct Params {
    ProblemShape problem_size;
    typename CollectiveMainloop::Params mainloop;
    typename CollectiveEpilogue::Params epilogue;
    typename TileScheduler::Params tile_scheduler;
  };

  using PipelineParamsInner = typename MainloopPipelineInner::Params;
  using PipelineStateInner  = typename cutlass::PipelineState<MainloopPipelineInner::Stages>;
  using PipelineParamsOuter = typename MainloopPipelineOuter::Params;
  using PipelineStateOuter  = typename cutlass::PipelineState<MainloopPipelineOuter::Stages>;
  using PipelineParamsReducer = typename MainloopPipelineReducer::Params;
  using PipelineStateReducer  = typename cutlass::PipelineState<MainloopPipelineReducer::Stages>;

  static const int MinBlocksPerMultiprocessor = 1;
  static const int MaxThreadsPerBlock = (NumMmaWarpGroups + NumLoadWarpGroups) * cutlass::NumThreadsPerWarpGroup;
  using ArchTag = cutlass::arch::Sm90;

  static constexpr uint32_t LoadRegisterRequirement = 40 - 2 * 8;
  static constexpr uint32_t TotalRegisterSupply = (64*1024 / MaxThreadsPerBlock / MinBlocksPerMultiprocessor / 8) * 8 * MaxThreadsPerBlock / cutlass::NumThreadsPerWarpGroup;
  static constexpr uint32_t MmaRegisterRequirement = ((TotalRegisterSupply - LoadRegisterRequirement) / NumMmaWarpGroups / 8) * 8;

  static size_t get_workspace_size(Arguments const& args) { return 0; }
  static cutlass::Status initialize_workspace(Arguments const&, void*, cudaStream_t) {
    return cutlass::Status::kSuccess;
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.problem_size, args.mainloop);
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.tile_scheduler);
  }

  static dim3 get_block_shape() {
    dim3 block(MaxThreadsPerBlock, 1, 1);
    return block;
  }

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return Params{
        args.problem_size,
        CollectiveMainloop::to_underlying_arguments(args.problem_size, args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(args.problem_size, args.epilogue, workspace),
        TileScheduler::to_underlying_arguments(args.problem_size, args.hw_info, ClusterShape{}, TileShape{})
    };
  }

  CUTLASS_DEVICE void operator()(const Params &params, char* smem) {
#if ! defined(CUTLASS_ARCH_MMA_SM90A_ENABLED)
    printf("ERROR : Arch conditional MMA instruction used without targeting appropriate compute capability. Aborting.\n");
#else

    enum class WarpGroupRole {
      Producer = 0,
      Consumer0 = 1,
      Consumer1 = 2,
      Consumer2 = 3,
      Consumer3 = 4,
    };
    enum class ProducerWarpRole {
      LoadKV = 1,
      Reducer = 0,
      MaybeLoadQ = 2,  // is kLoadsQSeparately is true, this warp loads Q (otherwise warp 0 does it)
      MainloopEpilogue = 3,
    };

    static constexpr ProducerWarpRole WarpRoleLoadQ = kLoadsQSeparately ? ProducerWarpRole::MaybeLoadQ : ProducerWarpRole::LoadKV;

    TileScheduler tile_scheduler{params.tile_scheduler};
  
    // Shared memory.
    auto& storage = *reinterpret_cast<SharedStorage*>(smem);
  
    int lane_idx = cutlass::canonical_lane_idx();
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    auto warp_group_role = WarpGroupRole(warp_group_idx);
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int consumer_warp_group_idx = warp_group_idx - (int) WarpGroupRole::Consumer0;
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    }

    PipelineParamsOuter pipeline_params_outer;
    pipeline_params_outer.transaction_bytes = CollectiveMainloop::kOuterLoadBytes;
    pipeline_params_outer.is_leader = lane_predicate && (producer_warp_role == WarpRoleLoadQ);
    pipeline_params_outer.num_consumers = cutlass::NumThreadsPerWarpGroup;

    PipelineParamsInner pipeline_params_inner;
    pipeline_params_inner.transaction_bytes = CollectiveMainloop::kInnerLoadBytes;
    pipeline_params_inner.is_leader = lane_predicate && (producer_warp_role == ProducerWarpRole::LoadKV);
    pipeline_params_inner.num_consumers = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;

    PipelineParamsReducer pipeline_params_reducer;
    pipeline_params_reducer.producer_arv_count = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
    pipeline_params_reducer.consumer_arv_count = cutlass::NumThreadsPerWarp;
    
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::LoadKV) {
      pipeline_params_inner.role = MainloopPipelineInner::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == WarpRoleLoadQ) {
      pipeline_params_outer.role = MainloopPipelineOuter::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Reducer) {
      pipeline_params_reducer.role = MainloopPipelineReducer::ThreadCategory::Consumer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || 
        warp_group_role == WarpGroupRole::Consumer1 ||
        warp_group_role == WarpGroupRole::Consumer2 ||
        warp_group_role == WarpGroupRole::Consumer3
    ) {
      pipeline_params_inner.role = MainloopPipelineInner::ThreadCategory::Consumer;
      pipeline_params_outer.role = MainloopPipelineOuter::ThreadCategory::Consumer;
      pipeline_params_reducer.role = MainloopPipelineReducer::ThreadCategory::Producer;
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }

    MainloopPipelineOuter pipeline_outer(storage.pipeline_storage_outer, pipeline_params_outer, Shape<_1, _1, _1>{});
    MainloopPipelineInner pipeline_inner(storage.pipeline_storage_inner, pipeline_params_inner, ClusterShape{});
    MainloopPipelineReducer pipeline_reducer(storage.pipeline_storage_reducer, pipeline_params_reducer);

    // State variables used for iterating the circular buffer
    // smem_pipe_read / release is used by the consumer of SMEM data - i.e MMA
    // smem_pipe_write is used by the producer of SMEM data - i.e TMA
    PipelineStateInner smem_pipe_read_inner;
    PipelineStateInner smem_pipe_write_inner = cutlass::make_producer_start_state<MainloopPipelineInner>();

    PipelineStateOuter smem_pipe_read_outer;
    PipelineStateOuter smem_pipe_write_outer = cutlass::make_producer_start_state<MainloopPipelineOuter>();

    PipelineStateReducer smem_pipe_read_reducer;
    PipelineStateReducer smem_pipe_write_reducer = cutlass::make_producer_start_state<MainloopPipelineReducer>();

    typename MathWarpGroupOrderBarrier::Params params_math_wg_order_barrier;
    // DMA Load WG will not participate in these Ordered Barrier syncs
    params_math_wg_order_barrier.group_id = consumer_warp_group_idx;
    params_math_wg_order_barrier.group_size = cutlass::NumThreadsPerWarpGroup; // Number of threads / participants in a group
    MathWarpGroupOrderBarrier math_wg_order_barrier(storage.math_wg_order, params_math_wg_order_barrier);
    
    // Epilogue Load pipeline
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
    epi_load_pipeline_params.consumer_arv_count = NumThreadsPerWarpGroup;
    epi_load_pipeline_params.transaction_bytes = CollectiveEpilogue::TmaTransactionBytes;
    EpiLoadPipeline epi_load_pipeline(storage.epi_load, epi_load_pipeline_params);

    if constexpr (kLoadsQSeparately) {
      if ((warp_idx == 0) && lane_predicate) {
        storage.load_warp_barrier.init(2 * cutlass::NumThreadsPerWarp);
      }
      cutlass::arch::fence_barrier_init();
    }

    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer blocks in the Cluster
    // and to finish smem init
    if constexpr (size(ClusterShape{}) > 1) {
      cute::cluster_arrive_relaxed();
      cute::cluster_wait();
    }
    else {
      __syncthreads();
    }

    CollectiveMainloop collective_mainloop;

    if (warp_group_role == WarpGroupRole::Producer) {
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
      if (producer_warp_role == ProducerWarpRole::LoadKV) {
        bool do_barrier = kLoadsQSeparately;

        CUTLASS_PRAGMA_NO_UNROLL
        for (; tile_scheduler.is_valid(); ++tile_scheduler) {
          auto blk_coord = tile_scheduler.get_block_coord();
          collective_mainloop.template load_kv_maybe_q<!kLoadsQSeparately>(
            block_rank_in_cluster,
            blk_coord, params.mainloop, params.problem_size,
            pipeline_inner, smem_pipe_write_inner,
            pipeline_outer, smem_pipe_write_outer,
            storage.tensors.mainloop,
            storage.load_warp_barrier, do_barrier
          );
          do_barrier = false;
        }
      }
      else if (kLoadsQSeparately && (producer_warp_role == ProducerWarpRole::MaybeLoadQ)) {
        bool do_barrier = true;

        CUTLASS_PRAGMA_NO_UNROLL
        for (; tile_scheduler.is_valid(); ++tile_scheduler) {
          auto blk_coord = tile_scheduler.get_block_coord();
          collective_mainloop.load_maybe_q(
            blk_coord, params.mainloop, params.problem_size,
            pipeline_outer, smem_pipe_write_outer,
            storage.tensors.mainloop,
            storage.load_warp_barrier, do_barrier
          );
          do_barrier = false;
        }
      } else if (producer_warp_role == ProducerWarpRole::Reducer) {
        for (; tile_scheduler.is_valid(); ++tile_scheduler) {
          auto blk_coord = tile_scheduler.get_block_coord();
          collective_mainloop.reduce(
            blk_coord, params.mainloop, params.problem_size,
            pipeline_reducer, smem_pipe_read_reducer,
            storage.tensors.mainloop
          );
        }
      }
    }
    else if (
      warp_group_role == WarpGroupRole::Consumer0 || 
      warp_group_role == WarpGroupRole::Consumer1 ||
      warp_group_role == WarpGroupRole::Consumer2 ||
      warp_group_role == WarpGroupRole::Consumer3
    ) {
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();
        auto wg_coord = blk_coord;

        constexpr int kOuterLoads = CollectiveMainloop::kOuterLoads;

        if (warp_group_role == WarpGroupRole::Consumer0) {
          smem_pipe_read_outer.advance(0 * kOuterLoads);
        }
        else if (warp_group_role == WarpGroupRole::Consumer1) {
          smem_pipe_read_outer.advance(1 * kOuterLoads);
        }
        else if (warp_group_role == WarpGroupRole::Consumer2) {
          smem_pipe_read_outer.advance(2 * kOuterLoads);
        }
        else if (warp_group_role == WarpGroupRole::Consumer3) {
          smem_pipe_read_outer.advance(3 * kOuterLoads);
        }

        constexpr int wg_dim = is_constant<0, decltype(get<1>(wg_coord))>::value ? 0 : 1;
        auto& wg_block = get<wg_dim>(wg_coord);
        if (warp_group_role == WarpGroupRole::Consumer0) {
          wg_block = NumMmaWarpGroups * wg_block + 0;
        }
        else if (warp_group_role == WarpGroupRole::Consumer1) {
          wg_block = NumMmaWarpGroups * wg_block + 1;
        }
        else if (warp_group_role == WarpGroupRole::Consumer2) {
          wg_block = NumMmaWarpGroups * wg_block + 2;
        }
        else if (warp_group_role == WarpGroupRole::Consumer3) {
          wg_block = NumMmaWarpGroups * wg_block + 3;
        }

        auto result = collective_mainloop.compute(
          blk_coord, wg_coord,
          params.mainloop, params.problem_size,
          pipeline_inner, smem_pipe_read_inner,
          pipeline_outer, smem_pipe_read_outer,
          pipeline_reducer, smem_pipe_write_reducer,
          storage.tensors.mainloop,
          math_wg_order_barrier
        );

        if (warp_group_role == WarpGroupRole::Consumer0) {
          smem_pipe_read_outer.advance(kOuterLoads * (NumMmaWarpGroups - 0));
        }
        if constexpr (NumMmaWarpGroups >= 2) {
          if (warp_group_role == WarpGroupRole::Consumer1) {
            smem_pipe_read_outer.advance(kOuterLoads * (NumMmaWarpGroups - 1));
          }
        }
        if constexpr (NumMmaWarpGroups >= 3) {
          if (warp_group_role == WarpGroupRole::Consumer2) {
            smem_pipe_read_outer.advance(kOuterLoads * (NumMmaWarpGroups - 2));
          }
        }
        if constexpr (NumMmaWarpGroups >= 4) {
          if (warp_group_role == WarpGroupRole::Consumer3) {
            smem_pipe_read_outer.advance(kOuterLoads * (NumMmaWarpGroups - 3));
          }
        }

        if constexpr (kIsEpilogueLocked) ; math_wg_order_barrier.wait();

        CollectiveEpilogue epilogue;
        epilogue(typename CollectiveMainloop::TileShapePV{}, wg_coord,
          result, typename CollectiveMainloop::TiledMmaPV{},
          params.problem_size, params.epilogue,
          epi_load_pipeline, storage.tensors.epilogue[consumer_warp_group_idx]);

        if constexpr (kIsEpilogueLocked) ; math_wg_order_barrier.arrive();
      }
    }
#endif
  }
};

}  // namespace cutlass::fmha::kernel
