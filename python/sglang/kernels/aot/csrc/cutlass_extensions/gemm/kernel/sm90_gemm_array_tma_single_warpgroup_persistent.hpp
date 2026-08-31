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

#include "cutlass_extensions/gemm/kernel/sm90_gemm_array_tma_warpspecialized_pingpong_precomputed.hpp"

namespace cutlass::gemm::kernel {

enum class SingleWarpgroupPipelineMode { PrefillAll, RollingRefill };

template <class ProblemShape, class CollectiveMainloop, class CollectiveEpilogue>
using SingleWarpgroupPersistentBase = GemmUniversalPrecomputedScheduler<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue,
    detail::PersistentTileSchedulerSm90GroupPrecomputed<ProblemShape, 8, true>>;

// Small-K persistent kernel that keeps the existing mixed-input collectives
// intact while one warpgroup overlaps the next output tile with current MMA.
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_,
          int MinCtasPerMultiprocessor_, int PrefetchNextTileStages_,
          SingleWarpgroupPipelineMode PipelineMode_ = SingleWarpgroupPipelineMode::PrefillAll>
class SingleWarpgroupPersistentGemm
    : public SingleWarpgroupPersistentBase<ProblemShape_, CollectiveMainloop_,
                                           CollectiveEpilogue_> {
 private:
  using Base =
      SingleWarpgroupPersistentBase<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_>;

 public:
  using ProblemShape = typename Base::ProblemShape;
  using CollectiveMainloop = typename Base::CollectiveMainloop;
  using CollectiveEpilogue = typename Base::CollectiveEpilogue;
  using TileShape = typename Base::TileShape;
  using TiledMma = typename Base::TiledMma;
  using ArchTag = typename Base::ArchTag;
  using InternalStrideA = typename Base::InternalStrideA;
  using InternalStrideB = typename Base::InternalStrideB;
  using InternalStrideC = typename Base::InternalStrideC;
  using InternalStrideD = typename Base::InternalStrideD;
  using ClusterShape = typename Base::ClusterShape;
  using TileScheduler = typename Base::TileScheduler;
  using Arguments = typename Base::Arguments;
  using Params = typename Base::Params;
  using SharedStorage = typename Base::SharedStorage;

  static constexpr uint32_t MaxThreadsPerBlock = NumThreadsPerWarpGroup;
  static constexpr uint32_t MinBlocksPerMultiprocessor = MinCtasPerMultiprocessor_;
  static constexpr int PrefetchNextTileStages = PrefetchNextTileStages_;
  static constexpr SingleWarpgroupPipelineMode PipelineMode = PipelineMode_;
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  static_assert(MinCtasPerMultiprocessor_ > 0,
                "Single-warpgroup persistent GEMM requires a positive CTA/SM target.");

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  static bool can_implement(Arguments const& args) {
    bool implementable = Base::can_implement(args);
    if constexpr (PipelineMode == SingleWarpgroupPipelineMode::PrefillAll) {
      auto problem_shape = args.problem_shape;
      if (problem_shape.is_host_problem_shape_available()) {
        constexpr int MaxPrefillK =
            CollectiveMainloop::DispatchPolicy::Stages * cute::size<2>(TileShape{});
        for (int group = 0; group < problem_shape.groups(); ++group) {
          implementable &= cute::get<2>(problem_shape.get_host_problem_shape(group)) <= MaxPrefillK;
        }
      }
    }
    return implementable;
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

#if !defined(__CUDA_ARCH_FEAT_SM90_ALL)
    printf(
        "ERROR : Arch conditional MMA instruction used without targeting sm90a compute capability. "
        "Aborting.\n");
#else
    static_assert(size(TiledMma{}) == NumThreadsPerWarpGroup,
                  "Single-warpgroup persistent GEMM requires a 128-thread TiledMma.");
    static_assert(size(ClusterShape{}) == 1,
                  "The single-warpgroup kernel supports only a 1x1x1 cluster.");
    static_assert(Base::IsGroupedGemmKernel,
                  "The single-warpgroup kernel supports grouped GEMM only.");
    static_assert(
        PrefetchNextTileStages > 0 &&
            PrefetchNextTileStages <= CollectiveMainloop::DispatchPolicy::Stages,
        "Cross-tile prefetch depth must be positive and fit inside the mainloop stage ring.");
    static_assert(rank(InternalStrideA{}) == 3 && rank(InternalStrideB{}) == 3,
                  "Mainloop strides must be rank-3.");
    static_assert(rank(InternalStrideC{}) == 3 && rank(InternalStrideD{}) == 3,
                  "Epilogue strides must be rank-3.");

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    int const thread_idx = int(threadIdx.x);
    int const lane_idx = canonical_lane_idx();
    int const warp_idx = canonical_warp_idx_sync();
    int const mma_thread_idx = thread_idx;
    uint32_t const block_rank_in_cluster = cute::block_rank_in_cluster();

    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::ProducerConsumer;
    mainloop_pipeline_params.is_leader = thread_idx == 0;
    mainloop_pipeline_params.num_consumers = NumThreadsPerWarpGroup;
    mainloop_pipeline_params.num_producers = CollectiveMainloop::NumProducerThreadEvents;
    mainloop_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, mainloop_pipeline_params,
                                       ClusterShape{});

    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    epi_load_pipeline_params.dst_blockid = block_rank_in_cluster;
    epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
    epi_load_pipeline_params.consumer_arv_count = NumThreadsPerWarpGroup;
    if constexpr (CollectiveEpilogue::RequiresTransactionBytes) {
      epi_load_pipeline_params.transaction_bytes = params.epilogue.tma_transaction_bytes;
    }
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;
    PipelineState mainloop_pipe_producer_state =
        cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState epi_store_pipe_producer_state =
        cutlass::make_producer_start_state<EpiStorePipeline>();

    __syncthreads();

    TiledMma tiled_mma;
    auto const blk_shape = TileShape{};
    TileScheduler scheduler{params.scheduler};
    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    // This wrapper intentionally has no epilogue producer warp. The Humming-style
    // token-scale callback reads its small row scale directly during store.
    if (collective_epilogue.is_producer_load_needed()) {
      return;
    }

    auto work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});
    if (!work_tile_info.is_valid()) {
      return;
    }

    auto problem_shape_MNKL =
        append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
    static_assert(tuple_size_v<decltype(load_inputs)> >= 2,
                  "load_init must return at least A and B tensors.");
    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    int32_t const logical_sm_idx = int32_t(blockIdx.x + blockIdx.y * gridDim.x);
    int32_t const logical_sm_count = params.hw_info.sm_count;
    auto input_tensormaps = collective_mainloop.tensormaps_init(
        params.mainloop, shared_storage.tensormaps.mainloop, logical_sm_count, logical_sm_idx);

    constexpr int EpilogueDescriptorSlot = 0;
    auto epi_store_tensormap = get<0>(
        collective_epilogue.store_init(params.epilogue, shared_storage.tensormaps.epilogue,
                                       logical_sm_count, logical_sm_idx, EpilogueDescriptorSlot));

    int32_t current_group = -1;
    constexpr bool IsEpiLoad = false;
    int current_prefetched_stages = 0;

    while (work_tile_info.is_valid()) {
      int32_t const next_group = work_tile_info.L_idx;
      bool const did_group_change = next_group != current_group;
      if (did_group_change) {
        problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(next_group), 1);
      }
      if (did_group_change && warp_idx == 0) {
        load_inputs = collective_mainloop.tensors_perform_update(load_inputs, params.mainloop,
                                                                 problem_shape_MNKL, next_group);
        collective_mainloop.tensormaps_fence_acquire(input_tensormaps);

        collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
            shared_storage.tensormaps.epilogue, params.epilogue, epi_store_tensormap,
            problem_shape_MNKL, next_group, EpilogueDescriptorSlot);
        __syncwarp();
        collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(
            shared_storage.tensormaps.epilogue, epi_store_tensormap, EpilogueDescriptorSlot);
      }
      current_group = next_group;

      auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
      auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
      auto producer_blk_coord = make_coord(m_coord, n_coord, _, Int<0>{});
      auto epilogue_blk_coord =
          make_coord(m_coord, n_coord, _, idx2crd(next_group, shape<4>(gB_nkl)));

      int const work_k_tile_count =
          TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);
      auto work_k_tile_start = TileScheduler::get_work_k_tile_start(work_tile_info);
      auto k_tile_iter =
          make_coord_iterator(idx2crd(work_k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl));

      auto accumulators = partition_fragment_C(tiled_mma, take<0, 2>(blk_shape));
      auto next_work_tile_info = work_tile_info;
      auto next_load_inputs = load_inputs;
      auto next_problem_shape_MNKL = problem_shape_MNKL;
      int next_prefetched_stages = 0;

      CUTLASS_PRAGMA_UNROLL
      for (int stage = 0; stage < current_prefetched_stages; ++stage) {
        ++k_tile_iter;
      }

      int current_prefill_stage_count = work_k_tile_count;
      int current_k_tiles_to_refill = 0;
      if constexpr (PipelineMode == SingleWarpgroupPipelineMode::RollingRefill) {
        current_prefill_stage_count = work_k_tile_count < CollectiveMainloop::DispatchPolicy::Stages
                                          ? work_k_tile_count
                                          : CollectiveMainloop::DispatchPolicy::Stages;
        current_k_tiles_to_refill = work_k_tile_count - current_prefill_stage_count;
      }
      int const current_k_tiles_to_produce =
          current_prefill_stage_count - current_prefetched_stages;
      CUTLASS_ASSERT(current_k_tiles_to_produce >= 0);
      if (current_k_tiles_to_produce > 0 && warp_idx == 0) {
        collective_mainloop.load(params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state,
                                 load_inputs, input_tensormaps, producer_blk_coord, k_tile_iter,
                                 current_k_tiles_to_produce, lane_idx, block_rank_in_cluster,
                                 shared_storage.tensors.mainloop);
        mainloop_pipe_producer_state.advance(current_k_tiles_to_produce);
      }
      auto current_refill_k_tile_iter = k_tile_iter;
      CUTLASS_PRAGMA_UNROLL
      for (int stage = 0; stage < current_k_tiles_to_produce; ++stage) {
        ++current_refill_k_tile_iter;
      }

      auto next_work = scheduler.fetch_next_work(work_tile_info);
      next_work_tile_info = get<0>(next_work);

      auto next_producer_blk_coord = producer_blk_coord;
      auto next_work_k_tile_start = work_k_tile_start;
      int next_work_k_tile_count = 0;
      bool next_group_change = false;
      bool next_group_mainloop_state_ready = true;

      if (next_work_tile_info.is_valid()) {
        next_group_change = next_work_tile_info.L_idx != current_group;
        next_group_mainloop_state_ready = !next_group_change;
        if (next_group_change) {
          next_problem_shape_MNKL =
              append<4>(params.problem_shape.get_problem_shape(next_work_tile_info.L_idx), 1);
        }
        auto next_m_coord = idx2crd(next_work_tile_info.M_idx, shape<2>(gA_mkl));
        auto next_n_coord = idx2crd(next_work_tile_info.N_idx, shape<2>(gB_nkl));
        next_producer_blk_coord = make_coord(next_m_coord, next_n_coord, _, Int<0>{});
        if (next_group_change) {
          next_work_k_tile_count = TileScheduler::get_work_k_tile_count(
              next_work_tile_info, next_problem_shape_MNKL, blk_shape);
          next_work_k_tile_start = TileScheduler::get_work_k_tile_start(next_work_tile_info);
        } else {
          next_work_k_tile_count = work_k_tile_count;
          next_work_k_tile_start = work_k_tile_start;
        }

        if constexpr (PipelineMode == SingleWarpgroupPipelineMode::PrefillAll) {
          if (next_group_change && warp_idx == 0) {
            next_load_inputs = collective_mainloop.tensors_perform_update(
                next_load_inputs, params.mainloop, next_problem_shape_MNKL,
                next_work_tile_info.L_idx);
            collective_mainloop.tensormaps_fence_acquire(input_tensormaps);
          }
          next_group_mainloop_state_ready = true;
        }
      }

      auto next_k_tile_iter =
          make_coord_iterator(idx2crd(next_work_k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl));

      int const available_prefetch_stages =
          next_work_k_tile_count < work_k_tile_count ? next_work_k_tile_count : work_k_tile_count;
      int const next_prefetch_stage_count =
          next_work_tile_info.is_valid()
              ? (available_prefetch_stages < PrefetchNextTileStages_ ? available_prefetch_stages
                                                                     : PrefetchNextTileStages_)
              : 0;
      int next_k_tiles_to_produce = next_prefetch_stage_count;
      auto produce_released_stage = [&] {
        if constexpr (PipelineMode == SingleWarpgroupPipelineMode::RollingRefill) {
          if (current_k_tiles_to_refill > 0) {
            if (warp_idx == 0) {
              collective_mainloop.load(params.mainloop, mainloop_pipeline,
                                       mainloop_pipe_producer_state, load_inputs, input_tensormaps,
                                       producer_blk_coord, current_refill_k_tile_iter, 1, lane_idx,
                                       block_rank_in_cluster, shared_storage.tensors.mainloop);
              ++current_refill_k_tile_iter;
              ++mainloop_pipe_producer_state;
            }
            --current_k_tiles_to_refill;
            return;
          }
        }

        if (next_k_tiles_to_produce > 0) {
          if constexpr (PipelineMode == SingleWarpgroupPipelineMode::RollingRefill) {
            if (!next_group_mainloop_state_ready) {
              if (warp_idx == 0) {
                next_load_inputs = collective_mainloop.tensors_perform_update(
                    next_load_inputs, params.mainloop, next_problem_shape_MNKL,
                    next_work_tile_info.L_idx);
                collective_mainloop.tensormaps_fence_acquire(input_tensormaps);
              }
              next_group_mainloop_state_ready = true;
            }
          }
          if (warp_idx == 0) {
            collective_mainloop.load(
                params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state, next_load_inputs,
                input_tensormaps, next_producer_blk_coord, next_k_tile_iter, 1, lane_idx,
                block_rank_in_cluster, shared_storage.tensors.mainloop);
            ++next_k_tile_iter;
            ++mainloop_pipe_producer_state;
          }
          --next_k_tiles_to_produce;
        }
      };

      collective_mainloop.mma_with_released_stage_producer(
          mainloop_pipeline, mainloop_pipe_consumer_state, accumulators, work_k_tile_count,
          mma_thread_idx, shared_storage.tensors.mainloop, params.mainloop, produce_released_stage);
      collective_mainloop.mma_tail(mainloop_pipeline, mainloop_pipe_consumer_state,
                                   work_k_tile_count);
      CUTLASS_ASSERT(current_k_tiles_to_refill == 0);
      produce_released_stage();

      if constexpr (PipelineMode == SingleWarpgroupPipelineMode::RollingRefill) {
        if (next_work_tile_info.is_valid() && !next_group_mainloop_state_ready) {
          if (warp_idx == 0) {
            next_load_inputs = collective_mainloop.tensors_perform_update(
                next_load_inputs, params.mainloop, next_problem_shape_MNKL,
                next_work_tile_info.L_idx);
            collective_mainloop.tensormaps_fence_acquire(input_tensormaps);
          }
          next_group_mainloop_state_ready = true;
        }
      }
      next_prefetched_stages = next_prefetch_stage_count - next_k_tiles_to_produce;
      mainloop_pipe_consumer_state.advance(work_k_tile_count);

      TileScheduler::fixup(params.scheduler, work_tile_info, accumulators, 1, 0);

      if (did_group_change && warp_idx == 0) {
        collective_epilogue.template tensormaps_fence_acquire<IsEpiLoad>(epi_store_tensormap);
      }

      auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next] =
          collective_epilogue.store(epi_load_pipeline, epi_load_pipe_consumer_state,
                                    epi_store_pipeline, epi_store_pipe_producer_state,
                                    problem_shape_MNKL, blk_shape, epilogue_blk_coord, accumulators,
                                    tiled_mma, mma_thread_idx, shared_storage.tensors.epilogue,
                                    epi_store_tensormap, work_tile_info.reduction_subtile_idx());
      epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next;
      epi_store_pipe_producer_state = epi_store_pipe_producer_state_next;

      auto store_tail_states =
          collective_epilogue.store_tail(epi_load_pipeline, epi_load_pipe_consumer_state,
                                         epi_store_pipeline, epi_store_pipe_producer_state);
      epi_load_pipe_consumer_state = get<0>(store_tail_states);
      epi_store_pipe_producer_state = get<1>(store_tail_states);

      if (next_work_tile_info.is_valid()) {
        load_inputs = next_load_inputs;
        problem_shape_MNKL = next_problem_shape_MNKL;
        current_group = next_work_tile_info.L_idx;
      }
      work_tile_info = next_work_tile_info;
      current_prefetched_stages = next_prefetched_stages;
    }

    if (warp_idx == 0) {
      collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);
    }
#endif
  }
};

}  // namespace cutlass::gemm::kernel
