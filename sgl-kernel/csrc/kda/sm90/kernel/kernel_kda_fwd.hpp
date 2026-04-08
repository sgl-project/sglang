// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cutlass/arch/arch.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cutlass.h>

#include <cutlass/pipeline/pipeline.hpp>

#include "kda/sm90/kernel/options.hpp"
#include "kda/sm90/utils/common.hpp"
#include "kda/sm90/utils/unused.hpp"

namespace kda::sm90::kernel {

using namespace cute;

template <typename T1, typename T2>
constexpr T1 round_down(T1 a, T2 b) {
  return (a / b) * b;
}

constexpr std::tuple<uint32_t, uint32_t, uint32_t> get_register_requirements(
    uint32_t max_threads_per_block,
    uint32_t min_blocks_per_multiprocessor,
    uint32_t num_state_mma_warp_groups  // state related mma
) {
  uint32_t reg_alloc_granularity = 8;

#if !defined(FLAT_DEBUG_PRINT) || !FLAT_DEBUG_PRINT
  uint32_t load_registers = 40 - 2 * reg_alloc_granularity;
#else
  uint32_t load_registers = 40;
#endif
  // TODO: better tuning
  uint32_t total_aux_load_budget = 176;
  uint32_t aux_registers = total_aux_load_budget - load_registers;  // (24 + X) or (40 + X)

  uint32_t total_registers =
      round_down(64 * 1024 / min_blocks_per_multiprocessor, max_threads_per_block * reg_alloc_granularity) /
      cutlass::NumThreadsPerWarpGroup;
  uint32_t mma_registers =
      round_down((total_registers - load_registers - aux_registers) / num_state_mma_warp_groups, reg_alloc_granularity);

  // max reg is 255, 248 round to multiple of reg_alloc_granularity;
  return {cute::min(248, load_registers), cute::min(248, mma_registers), cute::min(248, aux_registers)};
}

template <class CollectiveMainloop, class TileScheduler, class Options>
struct FlatKernelTmaWarpSpecializedKdaFwd {
  using ArchTag = cutlass::arch::Sm90;

  static const int NumLoadWarpGroups = 1;
  static constexpr int NumStateMmaWarpGroups = CollectiveMainloop::NumStateMmaWarpGroups;
  static constexpr int NumAuxMmaWarpGroups = CollectiveMainloop::NumAuxMmaWarpGroups;

  static constexpr int NeedsAlpha = CollectiveMainloop::NeedsAlpha;
  static constexpr int NeedsBeta = CollectiveMainloop::NeedsBeta;
  static constexpr int SafeGate = CollectiveMainloop::SafeGate;

  using TileShape = typename CollectiveMainloop::TileShape;
  using ClusterShape = typename CollectiveMainloop::ClusterShape;

  using MainloopQPipeline = typename CollectiveMainloop::MainloopQPipeline;
  using MainloopKPipeline = typename CollectiveMainloop::MainloopKPipeline;
  using MainloopVPipeline = typename CollectiveMainloop::MainloopVPipeline;
  using MainloopOPipeline = typename CollectiveMainloop::MainloopOPipeline;

  using MainloopQKPipeline = typename CollectiveMainloop::MainloopQKPipeline;
  using MainloopKKPipeline = typename CollectiveMainloop::MainloopKKPipeline;

  using MainloopAlphaLastPipeline = typename CollectiveMainloop::MainloopAlphaLastPipeline;

  using MainloopAlphaPipeline = typename CollectiveMainloop::MainloopAlphaPipeline;
  using MainloopBetaPipeline = typename CollectiveMainloop::MainloopBetaPipeline;

  using OrderedMathBarriers = typename CollectiveMainloop::OrderedMathBarriers;

  static constexpr uint32_t StagesPerMathWarpGroup = 2;

  using MathWarpGroupOrderBarrier = cutlass::OrderedSequenceBarrier<StagesPerMathWarpGroup, NumStateMmaWarpGroups>;

  struct TensorStorage {
    typename CollectiveMainloop::SharedStorage mainloop;
  };

  struct SharedStorage {
    TensorStorage tensors;

    using QPipelineStorage = typename MainloopQPipeline::SharedStorage;
    using KPipelineStorage = typename MainloopKPipeline::SharedStorage;
    using VPipelineStorage = typename MainloopVPipeline::SharedStorage;
    using OPipelineStorage = typename MainloopOPipeline::SharedStorage;

    alignas(16) QPipelineStorage q_pipeline_storage;
    alignas(16) KPipelineStorage k_pipeline_storage;
    alignas(16) VPipelineStorage v_pipeline_storage;
    alignas(16) OPipelineStorage o_pipeline_storage;

    using QKPipelineStorage = typename MainloopQKPipeline::SharedStorage;
    using KKPipelineStorage = typename MainloopKKPipeline::SharedStorage;

    alignas(16) QKPipelineStorage qk_pipeline_storage;
    alignas(16) KKPipelineStorage kk_pipeline_storage;

    using AlphaLastPipelineStorage = typename MainloopAlphaLastPipeline::SharedStorage;

    alignas(16) AlphaLastPipelineStorage alpha_last_pipeline_storage;

    using AlphaPipelineStorage = typename MainloopAlphaPipeline::SharedStorage;
    using BetaPipelineStorage = typename MainloopBetaPipeline::SharedStorage;
    alignas(16) AlphaPipelineStorage alpha_pipeline_storage;
    alignas(16) BetaPipelineStorage beta_pipeline_storage;

    alignas(16) cutlass::arch::ClusterBarrier load_warp_barrier;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct VarlenProblemShape {
    int32_t const* cu_seqlens;
    int64_t total_seqlen;
    int32_t num_seqs;
    int32_t num_heads;  // Q, K, V, O all share the same head count in KDA
    int32_t head_size;  // d
  };
  using ProblemShape = VarlenProblemShape;

  struct Arguments {
    ProblemShape problem_size;
    typename CollectiveMainloop::Arguments mainloop;
    cutlass::KernelHardwareInfo hw_info;
  };

  struct Params {
    ProblemShape problem_size;
    typename CollectiveMainloop::Params mainloop;
    typename TileScheduler::Params scheduler;
  };

  using QPipelineParams = typename MainloopQPipeline::Params;
  using QPipelineState = typename cutlass::PipelineState<MainloopQPipeline::Stages>;

  using KPipelineParams = typename MainloopKPipeline::Params;
  using KPipelineState = typename cutlass::PipelineState<MainloopKPipeline::Stages>;

  using VPipelineParams = typename MainloopVPipeline::Params;
  using VPipelineState = typename cutlass::PipelineState<MainloopVPipeline::Stages>;

  using OPipelineParams = typename MainloopOPipeline::Params;
  using OPipelineState = typename cutlass::PipelineState<MainloopOPipeline::Stages>;

  using QKPipelineParams = typename MainloopQKPipeline::Params;
  using QKPipelineState = typename cutlass::PipelineState<MainloopQKPipeline::Stages>;

  using KKPipelineParams = typename MainloopKKPipeline::Params;
  using KKPipelineState = typename cutlass::PipelineState<MainloopKKPipeline::Stages>;

  using AlphaLastPipelineParams = std::conditional_t<NeedsAlpha, typename MainloopAlphaLastPipeline::Params, Unused>;
  using AlphaLastPipelineState =
      std::conditional_t<NeedsAlpha, cutlass::PipelineState<MainloopAlphaLastPipeline::Stages>, Unused>;

  using AlphaPipelineParams = std::conditional_t<NeedsAlpha, typename MainloopAlphaPipeline::Params, Unused>;
  using AlphaPipelineState =
      std::conditional_t<NeedsAlpha, cutlass::PipelineState<MainloopAlphaPipeline::Stages>, Unused>;

  using BetaPipelineParams = std::conditional_t<NeedsBeta, typename MainloopBetaPipeline::Params, Unused>;
  using BetaPipelineState = std::conditional_t<NeedsBeta, cutlass::PipelineState<MainloopBetaPipeline::Stages>, Unused>;

  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int MaxThreadsPerBlock =
      (NumLoadWarpGroups + NumStateMmaWarpGroups + NumAuxMmaWarpGroups) * cutlass::NumThreadsPerWarpGroup;

  static constexpr auto RegisterRequirements =
      get_register_requirements(MaxThreadsPerBlock, MinBlocksPerMultiprocessor, NumStateMmaWarpGroups);
  static constexpr uint32_t LdStRegisterRequirement = get<0>(RegisterRequirements);
  static constexpr uint32_t StateMmaRegisterRequirement = get<1>(RegisterRequirements);
  static constexpr uint32_t AuxMmaRegisterRequirement = get<2>(RegisterRequirements);

  static size_t get_workspace_size(Arguments const& args) {
    return CollectiveMainloop::get_workspace_size(args.mainloop, args.hw_info.sm_count);
  }

  static cutlass::Status initialize_workspace(Arguments const& args, void* workspace, cudaStream_t stream) {
    return CollectiveMainloop::initialize_workspace(args.problem_size, args.mainloop, workspace, stream);
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.problem_size, args.mainloop);
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.scheduler);
  }

  static dim3 get_block_shape() {
    dim3 block(MaxThreadsPerBlock, 1, 1);
    return block;
  }

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return Params{
        args.problem_size,
        CollectiveMainloop::to_underlying_arguments(args.problem_size, args.mainloop, workspace),
        TileScheduler::to_underlying_arguments(args.problem_size, args.hw_info, ClusterShape{}, TileShape{})};
  }

  CUTE_DEVICE void operator()(const Params& params, char* smem) {
    enum class WarpGroupRole {
      LdSt = 0,
      Math0 = 1,
      Math1 = 2,
      MathA = 3,  // auxiliary math WG
    };

    // NOTE: CollectiveInverse will have more utilization on warp 0&1
    //       so we put beta and alpha preprocessing on warp 2&3
    enum class LdStWarpRole {
      LoadQKV = 0,
      StoreO = 1,
      LoadBeta = 2,
      LoadAlpha = 3,
    };

    TileScheduler scheduler{params.scheduler};

    // Shared memory.
    auto& storage = *reinterpret_cast<SharedStorage*>(smem);

    int lane_idx = cutlass::canonical_lane_idx();
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int warp_idx_in_wg = warp_idx % cutlass::NumWarpsPerWarpGroup;
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    auto warp_group_role = WarpGroupRole(warp_group_idx);
    auto ldst_warp_role = LdStWarpRole(warp_idx_in_wg);

    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    }

    constexpr int NumStateMathThreads = NumStateMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
    constexpr int NumAuxMathThreads = NumAuxMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;

    QPipelineParams q_pipeline_params;
    q_pipeline_params.transaction_bytes = CollectiveMainloop::LoadQBytes;
    q_pipeline_params.is_leader = lane_predicate && (ldst_warp_role == LdStWarpRole::LoadQKV);
    q_pipeline_params.num_consumers = NumStateMathThreads + NumAuxMathThreads;

    KPipelineParams k_pipeline_params;
    k_pipeline_params.transaction_bytes = CollectiveMainloop::LoadKBytes;
    k_pipeline_params.is_leader = lane_predicate && (ldst_warp_role == LdStWarpRole::LoadQKV);
    k_pipeline_params.num_consumers = NumStateMathThreads + NumAuxMathThreads;

    VPipelineParams v_pipeline_params;
    v_pipeline_params.transaction_bytes = CollectiveMainloop::LoadVBytes;
    v_pipeline_params.is_leader = lane_predicate && (ldst_warp_role == LdStWarpRole::LoadQKV);
    v_pipeline_params.num_consumers = NumStateMathThreads;

    AlphaPipelineParams alpha_pipeline_params;
    if constexpr (NeedsAlpha) {
      alpha_pipeline_params.transaction_bytes = CollectiveMainloop::LoadAlphaBytes;
      alpha_pipeline_params.is_leader = lane_predicate && (ldst_warp_role == LdStWarpRole::LoadQKV);
      alpha_pipeline_params.num_consumers = NumStateMathThreads + NumAuxMathThreads + cutlass::NumThreadsPerWarp;
    }

    OPipelineParams o_pipeline_params;
    o_pipeline_params.producer_arv_count = NumStateMathThreads;
    o_pipeline_params.consumer_arv_count = cutlass::NumThreadsPerWarp;

    QKPipelineParams qk_pipeline_params;
    qk_pipeline_params.producer_arv_count = NumAuxMathThreads;
    qk_pipeline_params.consumer_arv_count = NumStateMathThreads;

    KKPipelineParams kk_pipeline_params;
    kk_pipeline_params.producer_arv_count = NumAuxMathThreads;
    kk_pipeline_params.consumer_arv_count = NumStateMathThreads;

    AlphaLastPipelineParams alpha_last_pipeline_params;
    if constexpr (NeedsAlpha) {
      alpha_last_pipeline_params.producer_arv_count = cutlass::NumThreadsPerWarp;
      alpha_last_pipeline_params.consumer_arv_count = NumStateMathThreads;
    }

    BetaPipelineParams beta_pipeline_params;
    if constexpr (NeedsBeta) {
      beta_pipeline_params.producer_arv_count = cutlass::NumThreadsPerWarp;
      beta_pipeline_params.consumer_arv_count = NumAuxMathThreads + NumStateMathThreads;
    }

    OrderedMathBarriers math_barriers;

    if (warp_group_role == WarpGroupRole::LdSt && ldst_warp_role == LdStWarpRole::LoadQKV) {
      DPRINTF0_W("ldst_warp_role: LoadQKV Alpha\n");
      q_pipeline_params.role = MainloopQPipeline::ThreadCategory::Producer;
      k_pipeline_params.role = MainloopKPipeline::ThreadCategory::Producer;
      v_pipeline_params.role = MainloopVPipeline::ThreadCategory::Producer;
      if constexpr (NeedsAlpha) {
        alpha_pipeline_params.role = MainloopAlphaPipeline::ThreadCategory::Producer;
      }
    }
    if (warp_group_role == WarpGroupRole::LdSt && ldst_warp_role == LdStWarpRole::StoreO) {
      DPRINTF0_W("ldst_warp_role: StoreO\n");
      o_pipeline_params.role = MainloopOPipeline::ThreadCategory::Consumer;
    }
    if (warp_group_role == WarpGroupRole::LdSt && ldst_warp_role == LdStWarpRole::LoadBeta) {
      if constexpr (NeedsBeta) {
        beta_pipeline_params.role = MainloopBetaPipeline::ThreadCategory::Producer;
      }
    }
    if (warp_group_role == WarpGroupRole::LdSt && ldst_warp_role == LdStWarpRole::LoadAlpha) {
      // LoadAlpha warp consumes alpha_pipeline (reads last row) and produces alpha_last_pipeline
      if constexpr (NeedsAlpha) {
        alpha_pipeline_params.role = MainloopAlphaPipeline::ThreadCategory::Consumer;
      }
      alpha_last_pipeline_params.role = MainloopAlphaLastPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Math0 || warp_group_role == WarpGroupRole::Math1) {
      DPRINTF0_WG("warp_group_role: MathX\n");
      q_pipeline_params.role = MainloopQPipeline::ThreadCategory::Consumer;
      k_pipeline_params.role = MainloopKPipeline::ThreadCategory::Consumer;
      v_pipeline_params.role = MainloopVPipeline::ThreadCategory::Consumer;
      o_pipeline_params.role = MainloopOPipeline::ThreadCategory::Producer;

      qk_pipeline_params.role = MainloopQKPipeline::ThreadCategory::Consumer;
      kk_pipeline_params.role = MainloopKKPipeline::ThreadCategory::Consumer;

      if constexpr (NeedsAlpha) {
        alpha_pipeline_params.role = MainloopAlphaPipeline::ThreadCategory::Consumer;
        alpha_last_pipeline_params.role = MainloopAlphaLastPipeline::ThreadCategory::Consumer;
      }
      if constexpr (NeedsBeta) {
        beta_pipeline_params.role = MainloopBetaPipeline::ThreadCategory::Consumer;
      }

      math_barriers.init(warp_group_idx - 1);
    }
    if (warp_group_role == WarpGroupRole::MathA) {
      DPRINTF0_WG("warp_group_role: MathA\n");
      q_pipeline_params.role = MainloopQPipeline::ThreadCategory::Consumer;
      k_pipeline_params.role = MainloopKPipeline::ThreadCategory::Consumer;

      qk_pipeline_params.role = MainloopQKPipeline::ThreadCategory::Producer;
      kk_pipeline_params.role = MainloopKKPipeline::ThreadCategory::Producer;

      if constexpr (NeedsAlpha) {
        alpha_pipeline_params.role = MainloopAlphaPipeline::ThreadCategory::Consumer;
      }
      if constexpr (NeedsBeta) {
        beta_pipeline_params.role = MainloopBetaPipeline::ThreadCategory::Consumer;
      }
    }

    MainloopQPipeline q_pipeline(storage.q_pipeline_storage, q_pipeline_params, ClusterShape{});
    MainloopKPipeline k_pipeline(storage.k_pipeline_storage, k_pipeline_params, ClusterShape{});
    MainloopVPipeline v_pipeline(storage.v_pipeline_storage, v_pipeline_params, ClusterShape{});
    MainloopAlphaPipeline alpha_pipeline(storage.alpha_pipeline_storage, alpha_pipeline_params, ClusterShape{});
    MainloopOPipeline o_pipeline(storage.o_pipeline_storage, o_pipeline_params, /*InitBarriers=*/cute::true_type{});

    MainloopAlphaLastPipeline alpha_last_pipeline(
        storage.alpha_last_pipeline_storage,
        alpha_last_pipeline_params,
        /*InitBarriers=*/cute::true_type{});

    MainloopQKPipeline qk_pipeline(
        storage.qk_pipeline_storage,
        qk_pipeline_params,
        /*InitBarriers=*/cute::true_type{});
    MainloopKKPipeline kk_pipeline(
        storage.kk_pipeline_storage,
        kk_pipeline_params,
        /*InitBarriers=*/cute::true_type{});

    // MainloopAlphaPipeline alpha_pipeline(storage.alpha_pipeline_storage, alpha_pipeline_params,
    // /*InitBarriers=*/cute::true_type{});
    MainloopBetaPipeline beta_pipeline(
        storage.beta_pipeline_storage,
        beta_pipeline_params,
        /*InitBarriers=*/cute::true_type{});

    QPipelineState q_smem_pipe_read;
    QPipelineState q_smem_pipe_write = cutlass::make_producer_start_state<MainloopQPipeline>();
    KPipelineState k_smem_pipe_read;
    KPipelineState k_smem_pipe_write = cutlass::make_producer_start_state<MainloopKPipeline>();
    VPipelineState v_smem_pipe_read;
    VPipelineState v_smem_pipe_write = cutlass::make_producer_start_state<MainloopVPipeline>();
    OPipelineState o_smem_pipe_read;
    OPipelineState o_smem_pipe_write = cutlass::make_producer_start_state<MainloopOPipeline>();

    AlphaLastPipelineState alpha_last_smem_pipe_read;
    AlphaLastPipelineState alpha_last_smem_pipe_write;
    if constexpr (NeedsAlpha) {
      alpha_last_smem_pipe_write = cutlass::make_producer_start_state<MainloopAlphaLastPipeline>();
    }

    QKPipelineState qk_smem_pipe_read;
    QKPipelineState qk_smem_pipe_write = cutlass::make_producer_start_state<MainloopQKPipeline>();
    KKPipelineState kk_smem_pipe_read;
    KKPipelineState kk_smem_pipe_write = cutlass::make_producer_start_state<MainloopKKPipeline>();

    AlphaPipelineState alpha_smem_pipe_read;
    AlphaPipelineState alpha_smem_pipe_write;
    if constexpr (NeedsAlpha) {
      alpha_smem_pipe_write = cutlass::make_producer_start_state<MainloopAlphaPipeline>();
    }
    BetaPipelineState beta_smem_pipe_read;
    BetaPipelineState beta_smem_pipe_write;
    if constexpr (NeedsBeta) {
      beta_smem_pipe_write = cutlass::make_producer_start_state<MainloopBetaPipeline>();
    }

    // barrier sm or cluster level for initialization
    if constexpr (size(ClusterShape{}) > 1) {
      cute::cluster_arrive_relaxed();
      cute::cluster_wait();
    } else {
      __syncthreads();
    }
    DPRINTF0_WG("warpspecialized grid initialized\n");

    CollectiveMainloop collective_mainloop;

    if (warp_group_role == WarpGroupRole::LdSt) {
      DPRINTF0_WG("LsSt warp_group_idx:%d, RegisterRequirement:%d\n", warp_group_idx, LdStRegisterRequirement);
      cutlass::arch::warpgroup_reg_dealloc<LdStRegisterRequirement>();
      if (ldst_warp_role == LdStWarpRole::LoadQKV) {
        auto work_desc = scheduler.get_next_work(params.scheduler, params.problem_size);
        CUTE_NO_UNROLL
        for (; work_desc.is_valid(params.scheduler);
             work_desc = scheduler.get_next_work(params.scheduler, params.problem_size)) {
          DPRINTF0_WG(
              "LsSt working on LoadQ/K/V, seq_idx:%d, q/k/v_head_idx:(%d,%d,%d), seq_len:%lld)\n",
              work_desc.seq_idx,
              work_desc.q_head_idx(),
              work_desc.k_head_idx(),
              work_desc.v_head_idx(),
              work_desc.seq_len);
          auto tile_shape = typename CollectiveMainloop::TileShape{};
          collective_mainloop.load_qkv(
              params.mainloop,
              params.problem_size,
              tile_shape,
              work_desc,
              q_pipeline,
              q_smem_pipe_write,
              k_pipeline,
              k_smem_pipe_write,
              v_pipeline,
              v_smem_pipe_write,
              alpha_pipeline,
              alpha_smem_pipe_write,
              storage.tensors.mainloop);
        }
      } else if (ldst_warp_role == LdStWarpRole::LoadBeta) {
        if constexpr (NeedsBeta) {
          auto work_desc = scheduler.get_next_work(params.scheduler, params.problem_size);
          CUTE_NO_UNROLL
          for (; work_desc.is_valid(params.scheduler);
               work_desc = scheduler.get_next_work(params.scheduler, params.problem_size)) {
            DPRINTF0_WG(
                "LsSt working on LoadBeta, seq_idx:%d, sab_head_idx:%d, seq_len:%lld)\n",
                work_desc.seq_idx,
                work_desc.o_head_idx(),
                work_desc.seq_len);
            auto tile_shape = typename CollectiveMainloop::TileShape{};
            collective_mainloop.load_beta(
                params.mainloop,
                params.problem_size,
                tile_shape,
                work_desc,
                beta_pipeline,
                beta_smem_pipe_write,
                storage.tensors.mainloop);
          }
        }
      } else if (ldst_warp_role == LdStWarpRole::LoadAlpha) {
        // produce the last row of Alpha
        if constexpr (NeedsAlpha) {
          auto work_desc = scheduler.get_next_work(params.scheduler, params.problem_size);
          CUTE_NO_UNROLL
          for (; work_desc.is_valid(params.scheduler);
               work_desc = scheduler.get_next_work(params.scheduler, params.problem_size)) {
            DPRINTF0_WG(
                "LsSt working on LoadAlpha+ExtractLast, seq_idx:%d, sab_head_idx:%d, seq_len:%lld)\n",
                work_desc.seq_idx,
                work_desc.o_head_idx(),
                work_desc.seq_len);
            auto tile_shape = typename CollectiveMainloop::TileShape{};
            collective_mainloop.extract_alpha_last(
                params.mainloop,
                params.problem_size,
                tile_shape,
                work_desc,
                alpha_pipeline,
                alpha_smem_pipe_read,
                alpha_last_pipeline,
                alpha_last_smem_pipe_write,
                storage.tensors.mainloop);
          }
        }
      } else if (ldst_warp_role == LdStWarpRole::StoreO) {
        auto work_desc = scheduler.get_next_work(params.scheduler, params.problem_size);
        DPRINTF0_WG(
            "LsSt working on StoreO, seq_idx:%d, o_head_idx:%d, seq_len:%lld)\n",
            work_desc.seq_idx,
            work_desc.o_head_idx(),
            work_desc.seq_len);
        auto tile_shape = typename CollectiveMainloop::TileShape{};
        collective_mainloop.store(
            params.mainloop.tma_store_o,
            params.mainloop.tensormaps,
            params.problem_size,
            tile_shape,
            work_desc,
            o_pipeline,
            o_smem_pipe_read,
            storage.tensors.mainloop.smem_o);
      }
    } else if (warp_group_role == WarpGroupRole::Math0 || warp_group_role == WarpGroupRole::Math1) {
      DPRINTF0_WG(
          "Compute[state]: warp_group_idx:%d, RegisterRequirement:%d\n", warp_group_idx, StateMmaRegisterRequirement);
      cutlass::arch::warpgroup_reg_alloc<StateMmaRegisterRequirement>();
      auto work_desc = scheduler.get_next_work(params.scheduler, params.problem_size);
      CUTE_NO_UNROLL
      for (; work_desc.is_valid(params.scheduler);
           work_desc = scheduler.get_next_work(params.scheduler, params.problem_size)) {
        DPRINTF0_WG(
            "Compute[state]: seq_idx:%d, qk/v/o_head_idx:(%d,%d,%d,%d), seq_len:%lld)\n",
            work_desc.seq_idx,
            work_desc.q_head_idx(),
            work_desc.k_head_idx(),
            work_desc.v_head_idx(),
            work_desc.o_head_idx(),
            work_desc.seq_len);
        collective_mainloop.compute(
            params.mainloop,
            params.problem_size,
            work_desc,
            q_pipeline,
            q_smem_pipe_read,
            k_pipeline,
            k_smem_pipe_read,
            v_pipeline,
            v_smem_pipe_read,
            o_pipeline,
            o_smem_pipe_write,
            qk_pipeline,
            qk_smem_pipe_read,
            kk_pipeline,
            kk_smem_pipe_read,
            alpha_pipeline,
            alpha_smem_pipe_read,
            beta_pipeline,
            beta_smem_pipe_read,
            alpha_last_pipeline,
            alpha_last_smem_pipe_read,
            math_barriers,
            storage.tensors.mainloop);
      }
    } else if (warp_group_role == WarpGroupRole::MathA) {
      DPRINTF0_WG(
          "Compute[aux]: warp_group_idx:%d, RegisterRequirement:%d\n", warp_group_idx, AuxMmaRegisterRequirement);
      cutlass::arch::warpgroup_reg_alloc<AuxMmaRegisterRequirement>();
      auto work_desc = scheduler.get_next_work(params.scheduler, params.problem_size);
      CUTE_NO_UNROLL
      for (; work_desc.is_valid(params.scheduler);
           work_desc = scheduler.get_next_work(params.scheduler, params.problem_size)) {
        DPRINTF0_WG(
            "Compute[aux]: seq_idx:%d, qk/v/o_head_idx:(%d,%d,%d,%d), seq_len:%lld)\n",
            work_desc.seq_idx,
            work_desc.q_head_idx(),
            work_desc.k_head_idx(),
            work_desc.v_head_idx(),
            work_desc.o_head_idx(),
            work_desc.seq_len);
        collective_mainloop.compute_aux_safe(
            params.mainloop,
            params.problem_size,
            work_desc,
            q_pipeline,
            q_smem_pipe_read,
            k_pipeline,
            k_smem_pipe_read,
            qk_pipeline,
            qk_smem_pipe_write,
            kk_pipeline,
            kk_smem_pipe_write,
            alpha_pipeline,
            alpha_smem_pipe_read,
            beta_pipeline,
            beta_smem_pipe_read,
            alpha_last_pipeline,
            alpha_last_smem_pipe_write,
            storage.tensors.mainloop);
      }
    } else {
      DPRINTF0_WG("Unknown warp role, warp_group_idx:%d\n", warp_group_idx);
    }

    __syncthreads();
  }
};

}  // namespace kda::sm90::kernel
