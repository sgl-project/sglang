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
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/arch/arch.h"

#include "../kernel/fmha_tile_scheduler.hpp"
#include "../kernel/fmha_options.hpp"

namespace cutlass::fmha::kernel {

template<
  class CollectiveMainloop,
  class CollectiveEpilogue,
  class... Options
>
struct FmhaKernelTma {

  // Options
  static constexpr int kBlocksPerSM = find_option_t<Tag::kBlocksPerSM, Int<2>, Options...>::value;

  using Element = typename CollectiveMainloop::Element;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;

  using TileScheduler = IndividualTileScheduler;

  using StagesQ = typename CollectiveMainloop::StagesQ;
  using Stages = typename CollectiveMainloop::Stages;

  using TileShape = typename CollectiveMainloop::TileShape;
  using ClusterShape = typename CollectiveMainloop::ClusterShape;

  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using MainloopPipelineQ = typename CollectiveMainloop::MainloopPipelineQ;

  using SmemLayoutQ = typename CollectiveMainloop::SmemLayoutQ;
  using SmemLayoutK = typename CollectiveMainloop::SmemLayoutK;

  struct SharedStorage {
    union {
      typename CollectiveMainloop::SharedStorage mainloop;
      typename CollectiveEpilogue::TensorStorage epilogue;
    };

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    using PipelineStorageQ = typename MainloopPipelineQ::SharedStorage;
    alignas(16) PipelineStorage pipeline_storage;
    alignas(16) PipelineStorageQ pipeline_storage_q;

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

  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState  = typename cutlass::PipelineState<MainloopPipeline::Stages>;
  using PipelineParamsQ = typename MainloopPipelineQ::Params;
  using PipelineStateQ  = typename cutlass::PipelineState<MainloopPipelineQ::Stages>;

  static const int MinBlocksPerMultiprocessor = kBlocksPerSM;
  static const int MaxThreadsPerBlock = CollectiveMainloop::MaxThreadsPerBlock;
  using ArchTag = cutlass::arch::Sm90;

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
    TileScheduler tile_scheduler{params.tile_scheduler};

    // Shared memory.
    auto& storage = *reinterpret_cast<SharedStorage*>(smem);
  
    int thread_idx = int(threadIdx.x);
  
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    int warp_idx   = cutlass::canonical_warp_idx_sync();
    int warp_group_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
    int lane_predicate = cute::elect_one_sync();

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    }
  

    PipelineParamsQ pipeline_params_q;
    pipeline_params_q.transaction_bytes = size(SmemLayoutQ{}(_,_,_0{})) * sizeof(Element); // Q
    pipeline_params_q.role = MainloopPipelineQ::ThreadCategory::ProducerConsumer;
    pipeline_params_q.is_leader = warp_group_thread_idx == 0;
    pipeline_params_q.num_consumers = cutlass::NumThreadsPerWarpGroup;

    PipelineParams pipeline_params;
    pipeline_params.transaction_bytes = size(SmemLayoutK{}(_,_,_0{})) * sizeof(Element); // KV
    pipeline_params.role = MainloopPipeline::ThreadCategory::ProducerConsumer;
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = cutlass::NumThreadsPerWarpGroup;

    MainloopPipelineQ pipeline_q(storage.pipeline_storage_q, pipeline_params_q, Shape<_1, _1, _1>{});
    MainloopPipeline pipeline(storage.pipeline_storage, pipeline_params, ClusterShape{});

    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::ProducerConsumer;
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
    epi_load_pipeline_params.consumer_arv_count = NumThreadsPerWarpGroup;
    epi_load_pipeline_params.transaction_bytes = CollectiveEpilogue::TmaTransactionBytes;
    EpiLoadPipeline epi_load_pipeline(storage.epi_load, epi_load_pipeline_params);

    // State variables used for iterating the circular buffer
    // smem_pipe_read / release is used by the consumer of SMEM data - i.e MMA
    // smem_pipe_write is used by the producer of SMEM data - i.e TMA
    PipelineState smem_pipe_read;
    PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();

    PipelineStateQ smem_pipe_read_q;
    PipelineStateQ smem_pipe_write_q = cutlass::make_producer_start_state<MainloopPipelineQ>();

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

    auto blk_coord = tile_scheduler.get_block_coord();

    CollectiveMainloop collective_mainloop;
    auto result = collective_mainloop.compute(
      block_rank_in_cluster,
      blk_coord, params.mainloop, params.problem_size,
      pipeline, smem_pipe_read, smem_pipe_write,
      pipeline_q, smem_pipe_read_q, smem_pipe_write_q,
      storage.mainloop
    );

    CollectiveEpilogue epilogue;
    epilogue(typename CollectiveMainloop::TileShapePV{}, blk_coord,
      result, typename CollectiveMainloop::TiledMmaPV{},
      params.problem_size, params.epilogue,
      epi_load_pipeline, storage.epilogue);
#endif
  }
};

}  // namespace cutlass::fmha::kernel
