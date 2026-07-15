/*
 * Copyright (c) 2026 SGLang Team
 *
 * Portions copyright (c) 2025 DeepSeek.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * The compute, softmax, cluster exchange, and epilogue in this file are
 * adapted from the SM90 sparse FP8 decode kernel in FlashMLA, pinned by
 * sgl-kernel/cmake/flashmla.cmake at commit
 * be055fb7df0090fde45f08e9cb5b8b4c0272da73.  The producer warpgroup was
 * replaced with a loader for SGLang's 416-byte NVFP4 latent-cache row.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/barrier.h>

#include <cutlass/cluster_launch.hpp>

#include "dequant.h"
#include "flashmla_utils.h"
#include "legacy_flashmla_compat.h"
#include "params.h"
#include "splitkv_mla.h"

using namespace cute;

namespace sm90 {

static constexpr float MAX_INIT_VAL = -1e30f;
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using decode::sparse_fp8::L1CacheHint;
using decode::sparse_fp8::L2PrefetchHint;
using decode::sparse_fp8::load_128b_from_gmem;

#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)

struct StageTimingAccumulator {
  uint64_t timed_tiles = 0;
  uint64_t load_cycles = 0;
  uint64_t dequant_cycles = 0;
  uint64_t handoff_cycles = 0;
  uint64_t consumer_cycles = 0;
  uint64_t consumer_ready_wait_cycles = 0;
  uint64_t producer_available_wait_cycles = 0;
  uint64_t consumer_sync_wait_cycles = 0;
};

__forceinline__ __device__ void stage_timing_depend(uint64_t value) {
  // A volatile dependent instruction prevents the end clock from moving in
  // front of the global load or arithmetic whose result is being measured.
  asm volatile("mov.b64 %0, %0;" : "+l"(value) : : "memory");
}

__forceinline__ __device__ void write_stage_timing_record(
    uint64_t* stage_timing_ptr, int cta_idx, int record_idx, const StageTimingAccumulator& timing) {
  uint64_t* record =
      stage_timing_ptr + (cta_idx * kStageTimingRecordsPerCta + record_idx) * kStageTimingMetricsPerRecord;
  record[kTimedTileCount] = timing.timed_tiles;
  record[kLoadCycles] = timing.load_cycles;
  record[kDequantCycles] = timing.dequant_cycles;
  record[kHandoffCycles] = timing.handoff_cycles;
  record[kConsumerCycles] = timing.consumer_cycles;
  record[kConsumerReadyWaitCycles] = timing.consumer_ready_wait_cycles;
  record[kProducerAvailableWaitCycles] = timing.producer_available_wait_cycles;
  record[kConsumerSyncWaitCycles] = timing.consumer_sync_wait_cycles;
}

#endif

template <typename Tensor0, typename Tensor1>
__forceinline__ __device__ void save_rPb_to_sP_nvfp4(Tensor0 const& rPb, Tensor1 const& sP, int idx_in_warpgroup) {
  auto r2s_copy = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, bf16>{}, TiledMMA_QK{});
  ThrCopy thr_copy = r2s_copy.get_slice(idx_in_warpgroup);
  Tensor thr_copy_rPb = thr_copy.retile_S(rPb);
  Tensor thr_copy_sP = thr_copy.partition_D(sP);
  cute::copy(r2s_copy, thr_copy_rPb, thr_copy_sP);
}

template <typename Tensor0, typename Tensor1, typename Tensor2>
__forceinline__ __device__ void scale_softmax_nvfp4(
    Tensor0& rP,
    Tensor1& rS,
    Tensor2& rO,
    float scale_softmax_log2,
    float sScale[],
    float rM[2],
    float rL[2],
    bool is_kv_valid[],
    int idx_in_warpgroup) {
  float scale_for_olds[2];
  CUTE_UNROLL
  for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
    Tensor cur_rP = flatten(rP(make_coord(_, local_row_idx, _), _, _));
    Tensor cur_rS = flatten(rS(make_coord(_, local_row_idx, _), _, _));
    Tensor cur_rO = flatten(rO(make_coord(_, local_row_idx, _), _, _));

    float cur_max = -INFINITY;
    CUTE_UNROLL
    for (int i = 0; i < size(cur_rP); ++i) {
      if (!is_kv_valid[(i & 1) + (i / 2) * 8 + (idx_in_warpgroup % 4) * 2]) {
        cur_rP(i) = -INFINITY;
      }
      cur_max = max(cur_max, cur_rP(i));
    }
    cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
    cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

    cur_max *= scale_softmax_log2;
    const float old_max = rM[local_row_idx];
    rM[local_row_idx] = max(cur_max, old_max);
    const float scale_for_old = exp2f(old_max - rM[local_row_idx]);
    scale_for_olds[local_row_idx] = scale_for_old;

    CUTE_UNROLL
    for (int i = 0; i < size(cur_rO); ++i) {
      cur_rO(i) *= scale_for_old;
    }

    float cur_sum = 0.0f;
    CUTE_UNROLL
    for (int i = 0; i < size(cur_rP); ++i) {
      cur_rP(i) = exp2f(cur_rP(i) * scale_softmax_log2 - rM[local_row_idx]);
      cur_rS(i) = static_cast<bf16>(cur_rP(i));
      cur_sum += cur_rP(i);
    }
    rL[local_row_idx] = rL[local_row_idx] * scale_for_old + cur_sum;
  }
  if (idx_in_warpgroup % 4 == 0) {
    *reinterpret_cast<float2*>(sScale + 2 * (idx_in_warpgroup / 4)) = *reinterpret_cast<float2*>(scale_for_olds);
  }
}

#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
template <bool kEnableStageTiming, typename TmaParams>
#else
template <typename TmaParams>
#endif
__global__ void __launch_bounds__(NUM_THREADS, 1, 2) flash_fwd_splitkv_mla_nvfp4_sparse_kernel(
    __grid_constant__ const DecodingParams params,
    __grid_constant__ const TmaParams tma_params,
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
    const float* const kv_global_scale_ptr,
    uint64_t* const stage_timing_ptr
#else
    const float* const kv_global_scale_ptr
#endif
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
  const int head_block_idx = blockIdx.x;
  const int s_q_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int idx_in_cluster = head_block_idx % 2;
  const int warpgroup_idx = cutlass::canonical_warp_group_idx();
  const int idx_in_warpgroup = threadIdx.x % 128;
  const int warp_idx = cutlass::canonical_warp_idx_sync();
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
  const int stage_timing_cta_idx = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
#endif

  extern __shared__ char wksp_buf[];
  SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
  Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQ{});
  Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()), SmemLayoutOBuf{});
  Tensor sOAccumBuf = make_tensor(make_smem_ptr(plan.u.oAccumBuf.data()), SmemLayoutOAccumBuf{});
  Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
  float* sM = plan.sM;
  float* sL = plan.sL;
  float* sScale = plan.sScale;

  if (warp_idx == 0 && elect_one_sync()) {
    cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
  }

  if (warp_idx == 0 && elect_one_sync()) {
    plan.bar_q.init(1);
    CUTE_UNROLL
    for (int i = 0; i < NUM_K_BUFS; ++i) {
      plan.bar_k_local_ready[i].init(128);
      plan.bar_k_remote_ready[i].init(1);
      plan.bar_k_avail[i].init(4);
    }
    fence_view_async_shared();
  }
  cute::cluster_arrive();

  bool bar_phase_q = 0;
  int bar_phase_k = 0;

  const DecodingSchedMeta sched_meta =
      reinterpret_cast<const DecodingSchedMeta*>(params.tile_scheduler_metadata_ptr)[partition_idx];
  const int begin_idx = sched_meta.begin_req_idx;
  const int sched_begin_block_idx = sched_meta.begin_block_idx;
  const int end_idx = sched_meta.end_req_idx;
  const int sched_end_block_idx = sched_meta.end_block_idx;
  if (begin_idx >= params.b || begin_idx < 0) {
    return;
  }
  const int begin_n_split_idx = sched_meta.begin_split_idx;

  if (warp_idx == 0 && elect_one_sync()) {
    Tensor gQ = flat_divide(
        tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, begin_idx),
        Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{})(_, _, head_block_idx, _0{});
    launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
    plan.bar_q.arrive_and_expect_tx(BLOCK_M * HEAD_DIM_K * sizeof(bf16));
  }

  cute::cluster_wait();

  auto get_cur_req_info = [&](int batch_idx) -> std::tuple<int, int, bool> {
    constexpr int kBlockN = TOPK_BLOCK_SIZE;
    const int start_block_idx = batch_idx == begin_idx ? sched_begin_block_idx : 0;
    const int end_block_idx = batch_idx == end_idx ? sched_end_block_idx : cute::ceil_div(params.topk, kBlockN);
    const bool is_no_split = batch_idx == begin_idx ? !sched_meta.is_first_req_splitted
                                                    : (batch_idx == end_idx ? !sched_meta.is_last_req_splitted : true);
    return {start_block_idx, end_block_idx, is_no_split};
  };

  if (warpgroup_idx == 0) {
    cutlass::arch::warpgroup_reg_alloc<192>();

    TiledMMA tiled_mma_QK = TiledMMA_QK{};
    ThrMMA thr_mma_QK = tiled_mma_QK.get_slice(idx_in_warpgroup);
    TiledMMA tiled_mma_PV = TiledMMA_PV_LocalP{};
    ThrMMA thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);

    float rL[2], rM[2];
    Tensor rO = partition_fragment_C(TiledMMA_PV_LocalP{}, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V / 2>>{});
    Tensor rP = partition_fragment_C(TiledMMA_QK{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{});
    Tensor rS = make_tensor<bf16>(partition_shape_A(TiledMMA_PV_LocalP{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}));
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
    StageTimingAccumulator stage_timing = {};
    bool stage_timing_seen_first_tile = false;
#endif

#pragma unroll 1
    for (int batch_idx = begin_idx; batch_idx <= end_idx; ++batch_idx) {
      auto [start_block_idx, end_block_idx, is_no_split] = get_cur_req_info(batch_idx);

      rL[0] = rL[1] = 0.0f;
      rM[0] = rM[1] = MAX_INIT_VAL;
      cute::fill(rO, 0.0f);

      plan.bar_q.wait(bar_phase_q);
      bar_phase_q ^= 1;

      CUTE_NO_UNROLL
      for (int block_idx = start_block_idx; block_idx < end_block_idx; ++block_idx) {
        const int buf_idx = (block_idx - start_block_idx) % NUM_K_BUFS;
        Tensor sK = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutHalfV{});

#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
        const bool stage_timing_measure_tile = stage_timing_seen_first_tile;
        uint64_t stage_start = 0;
        if constexpr (kEnableStageTiming) {
          if (idx_in_warpgroup == 0) {
            stage_start = clock64();
          }
        }
#endif
        plan.bar_k_local_ready[buf_idx].wait(bar_phase_k >> buf_idx & 1);
        plan.bar_k_remote_ready[buf_idx].wait(bar_phase_k >> buf_idx & 1);
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
        if constexpr (kEnableStageTiming) {
          if (idx_in_warpgroup == 0 && stage_timing_measure_tile) {
            stage_timing.consumer_ready_wait_cycles += clock64() - stage_start;
          }
          if (idx_in_warpgroup == 0) {
            stage_start = clock64();
          }
        }
#endif

        gemm<true, -1>(tiled_mma_QK, thr_mma_QK.partition_fragment_A(sQ), thr_mma_QK.partition_fragment_B(sK), rP);

        bar_phase_k ^= 1 << buf_idx;
        cute::warpgroup_wait<0>();
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
        if constexpr (kEnableStageTiming) {
          if (idx_in_warpgroup == 0 && stage_timing_measure_tile) {
            stage_timing.consumer_cycles += clock64() - stage_start;
          }
          if (idx_in_warpgroup == 0) {
            stage_start = clock64();
          }
        }
#endif

        if (block_idx != start_block_idx) {
          NamedBarrier::arrive_and_wait(256, NamedBarriers::sScale_and_sS_free);
        }
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
        if constexpr (kEnableStageTiming) {
          if (idx_in_warpgroup == 0 && stage_timing_measure_tile && block_idx != start_block_idx) {
            stage_timing.consumer_sync_wait_cycles += clock64() - stage_start;
          }
          if (idx_in_warpgroup == 0) {
            stage_start = clock64();
          }
        }
#endif

        scale_softmax_nvfp4(
            rP, rS, rO, params.scale_softmax_log2, sScale, rM, rL, plan.is_kv_valid[buf_idx], idx_in_warpgroup);

        save_rPb_to_sP_nvfp4(rS, sS, idx_in_warpgroup);
        fence_view_async_shared();
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
        if constexpr (kEnableStageTiming) {
          if (idx_in_warpgroup == 0 && stage_timing_measure_tile) {
            stage_timing.consumer_cycles += clock64() - stage_start;
          }
          if (idx_in_warpgroup == 0) {
            stage_start = clock64();
          }
        }
#endif

        gemm<false, -1>(tiled_mma_PV, rS, thr_mma_PV.partition_fragment_B(sV), rO);

        NamedBarrier::arrive(256, NamedBarriers::sScale_and_sS_ready);
        cute::warpgroup_wait<0>();
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
        if constexpr (kEnableStageTiming) {
          if (idx_in_warpgroup == 0 && stage_timing_measure_tile) {
            stage_timing.consumer_cycles += clock64() - stage_start;
            ++stage_timing.timed_tiles;
          }
        }
        stage_timing_seen_first_tile = true;
#endif

        plan.bar_k_avail[buf_idx].arrive(0, idx_in_warpgroup == 32);
        plan.bar_k_avail[buf_idx].arrive(1, idx_in_warpgroup == 64);
      }

      if (warp_idx == 0 && elect_one_sync()) {
        if (batch_idx != end_idx) {
          Tensor gQ = flat_divide(
              tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, batch_idx + 1),
              Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{})(_, _, head_block_idx, _0{});
          launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
          plan.bar_q.arrive_and_expect_tx(BLOCK_M * HEAD_DIM_K * sizeof(bf16));
        } else {
          cudaTriggerProgrammaticLaunchCompletion();
        }
      }

      rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
      rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
      rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
      rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);
      if (idx_in_warpgroup % 4 == 0) {
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i) {
          const int row = get_AorC_row_idx(i, idx_in_warpgroup);
          sL[row] = rL[i];
          sM[row] = rM[i];
        }
      }

      NamedBarrier::arrive_and_wait(256, NamedBarriers::oBuf_free_and_sL_ready);

      CUTE_UNROLL
      for (int i = 0; i < 2; ++i) {
        rL[i] = rL[i] == 0.0f ? 1.0f : rL[i];
      }

      const int num_valid_seq_q = min(params.q_head_per_hk - head_block_idx * BLOCK_M, BLOCK_M);
      const int start_seq_idx = s_q_idx * params.q_head_per_hk + head_block_idx * BLOCK_M;
      if (is_no_split) {
        bf16* o_ptr =
            static_cast<bf16*>(params.o_ptr) + batch_idx * params.o_batch_stride + start_seq_idx * params.o_row_stride;
        Tensor gO = make_tensor(
            make_gmem_ptr(o_ptr),
            make_layout(Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{}, make_stride(params.o_row_stride, _1{})));
        float* gSoftmaxLse =
            static_cast<float*>(params.softmax_lse_ptr) + batch_idx * params.q_seq_per_hk + start_seq_idx;

        store_o<true>(
            rO,
            gO,
            sOBuf,
            sOAccumBuf,
            rL,
            tma_params,
            batch_idx,
            s_q_idx,
            head_block_idx,
            num_valid_seq_q,
            warpgroup_idx,
            idx_in_warpgroup);

        const int i = threadIdx.x;
        if (i < num_valid_seq_q) {
          const float cur_L = sL[i];
          gSoftmaxLse[i] = cur_L == 0.0f ? INFINITY : logf(cur_L) + sM[i] / static_cast<float>(M_LOG2E);
        }
        cute::tma_store_wait<0>();
      } else {
        const int n_split_idx = batch_idx == begin_idx ? begin_n_split_idx : 0;
        const int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
        float* oaccum_ptr =
            static_cast<float*>(params.oaccum_ptr) + (split_idx * params.q_seq_per_hk + start_seq_idx) * HEAD_DIM_V;
        float* gSoftmaxLseAccum =
            static_cast<float*>(params.softmax_lseaccum_ptr) + split_idx * params.q_seq_per_hk + start_seq_idx;
        Tensor gOAccum = make_tensor(
            make_gmem_ptr(oaccum_ptr), Layout<Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>, Stride<Int<HEAD_DIM_V>, _1>>{});
        store_o<false>(
            rO,
            gOAccum,
            sOBuf,
            sOAccumBuf,
            rL,
            tma_params,
            batch_idx,
            s_q_idx,
            head_block_idx,
            num_valid_seq_q,
            warpgroup_idx,
            idx_in_warpgroup);

        const int i = threadIdx.x;
        if (i < num_valid_seq_q) {
          const float cur_L = sL[i];
          gSoftmaxLseAccum[i] = cur_L == 0.0f ? -INFINITY : log2f(cur_L) + sM[i];
        }
        cute::tma_store_wait<0>();
      }

      cute::cluster_sync();
    }
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
    if constexpr (kEnableStageTiming) {
      if (idx_in_warpgroup == 0) {
        write_stage_timing_record(stage_timing_ptr, stage_timing_cta_idx, kConsumerLocalRecord, stage_timing);
      }
    }
#endif
  } else if (warpgroup_idx == 1) {
    cutlass::arch::warpgroup_reg_dealloc<160>();

    TiledMMA tiled_mma_PV = TiledMMA_PV_RemoteP{};
    ThrMMA thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);
    Tensor rO = partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V / 2>>{});
    float rL[2];
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
    StageTimingAccumulator stage_timing = {};
    bool stage_timing_seen_first_tile = false;
#endif

#pragma unroll 1
    for (int batch_idx = begin_idx; batch_idx <= end_idx; ++batch_idx) {
      auto [start_block_idx, end_block_idx, is_no_split] = get_cur_req_info(batch_idx);
      cute::fill(rO, 0.0f);

      CUTE_NO_UNROLL
      for (int block_idx = start_block_idx; block_idx < end_block_idx; ++block_idx) {
        const int buf_idx = (block_idx - start_block_idx) % NUM_K_BUFS;
        Tensor sV =
            make_tensor(make_smem_ptr(plan.u.k[buf_idx].data() + SmemLayoutV{}(_256{}, _0{})), SmemLayoutHalfV{});

#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
        const bool stage_timing_measure_tile = stage_timing_seen_first_tile;
        uint64_t stage_start = 0;
        if constexpr (kEnableStageTiming) {
          if (idx_in_warpgroup == 0) {
            stage_start = clock64();
          }
        }
#endif
        NamedBarrier::arrive_and_wait(256, NamedBarriers::sScale_and_sS_ready);
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
        if constexpr (kEnableStageTiming) {
          if (idx_in_warpgroup == 0 && stage_timing_measure_tile) {
            stage_timing.consumer_sync_wait_cycles += clock64() - stage_start;
          }
          if (idx_in_warpgroup == 0) {
            stage_start = clock64();
          }
        }
#endif

        float cur_scales[2];
        *reinterpret_cast<float2*>(cur_scales) = *reinterpret_cast<float2*>(sScale + (idx_in_warpgroup / 4) * 2);
        CUTE_UNROLL
        for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
          Tensor cur_rO = flatten(rO(make_coord(_, local_row_idx, _), _, _));
          CUTE_UNROLL
          for (int i = 0; i < size(cur_rO); ++i) {
            cur_rO(i) *= cur_scales[local_row_idx];
          }
        }

        gemm<false, -1>(tiled_mma_PV, thr_mma_PV.partition_fragment_A(sS), thr_mma_PV.partition_fragment_B(sV), rO);
        cute::warpgroup_wait<0>();
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
        if constexpr (kEnableStageTiming) {
          if (idx_in_warpgroup == 0 && stage_timing_measure_tile) {
            stage_timing.consumer_cycles += clock64() - stage_start;
            ++stage_timing.timed_tiles;
          }
        }
        stage_timing_seen_first_tile = true;
#endif

        plan.bar_k_avail[buf_idx].arrive(0, idx_in_warpgroup == 32);
        plan.bar_k_avail[buf_idx].arrive(1, idx_in_warpgroup == 64);

        if (block_idx != end_block_idx - 1) {
          NamedBarrier::arrive(256, NamedBarriers::sScale_and_sS_free);
        }
      }

      NamedBarrier::arrive_and_wait(256, NamedBarriers::oBuf_free_and_sL_ready);
      CUTE_UNROLL
      for (int i = 0; i < 2; ++i) {
        const int row = get_AorC_row_idx(i, idx_in_warpgroup);
        rL[i] = sL[row];
      }
      CUTE_UNROLL
      for (int i = 0; i < 2; ++i) {
        rL[i] = rL[i] == 0.0f ? 1.0f : rL[i];
      }

      const int num_valid_seq_q = min(params.q_head_per_hk - head_block_idx * BLOCK_M, BLOCK_M);
      const int start_seq_idx = s_q_idx * params.q_head_per_hk + head_block_idx * BLOCK_M;
      if (is_no_split) {
        bf16* o_ptr =
            static_cast<bf16*>(params.o_ptr) + batch_idx * params.o_batch_stride + start_seq_idx * params.o_row_stride;
        Tensor gO = make_tensor(
            make_gmem_ptr(o_ptr),
            make_layout(Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{}, make_stride(params.o_row_stride, _1{})));
        store_o<true>(
            rO,
            gO,
            sOBuf,
            sOAccumBuf,
            rL,
            tma_params,
            batch_idx,
            s_q_idx,
            head_block_idx,
            num_valid_seq_q,
            warpgroup_idx,
            idx_in_warpgroup);
        cute::tma_store_wait<0>();
      } else {
        const int n_split_idx = batch_idx == begin_idx ? begin_n_split_idx : 0;
        const int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
        float* oaccum_ptr =
            static_cast<float*>(params.oaccum_ptr) + (split_idx * params.q_seq_per_hk + start_seq_idx) * HEAD_DIM_V;
        Tensor gOAccum = make_tensor(
            make_gmem_ptr(oaccum_ptr), Layout<Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>, Stride<Int<HEAD_DIM_V>, _1>>{});
        store_o<false>(
            rO,
            gOAccum,
            sOBuf,
            sOAccumBuf,
            rL,
            tma_params,
            batch_idx,
            s_q_idx,
            head_block_idx,
            num_valid_seq_q,
            warpgroup_idx,
            idx_in_warpgroup);
        cute::tma_store_wait<0>();
      }
      cute::cluster_sync();
    }
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
    if constexpr (kEnableStageTiming) {
      if (idx_in_warpgroup == 0) {
        write_stage_timing_record(stage_timing_ptr, stage_timing_cta_idx, kConsumerRemoteRecord, stage_timing);
      }
    }
#endif
  } else {
    // Producer warpgroup.  Each group of four lanes owns one selected token;
    // each lane dequantizes one aligned 16-element latent block at a time.
    cutlass::arch::warpgroup_reg_dealloc<152>();

    const int producer_warp_idx = __shfl_sync(0xffffffff, idx_in_warpgroup / 32, 0);
    const int lane_idx = idx_in_warpgroup % 32;
    const int my_token_idx = producer_warp_idx * 8 + lane_idx % 8;
    const float kv_global_scale = __ldg(kv_global_scale_ptr);
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
    StageTimingAccumulator stage_timing = {};
    bool stage_timing_seen_first_tile = false;
    const bool stage_timing_lane_leader = lane_idx == 0;
#endif

    CUTE_NO_UNROLL
    for (int batch_idx = begin_idx; batch_idx <= end_idx; ++batch_idx) {
      auto [start_block_idx, end_block_idx, is_no_split] = get_cur_req_info(batch_idx);
      int* gIndices =
          params.indices_ptr + batch_idx * params.indices_batch_stride + s_q_idx * params.indices_row_stride;

#define GET_TOKEN_INDEX(block_idx) \
  __ldg(gIndices + (block_idx) * TOPK_BLOCK_SIZE + idx_in_cluster * (TOPK_BLOCK_SIZE / 2) + my_token_idx)

#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
      uint64_t initial_index_start = 0;
      if constexpr (kEnableStageTiming) {
        if (stage_timing_lane_leader) {
          initial_index_start = clock64();
        }
      }
#endif
      int nxt_token_index = GET_TOKEN_INDEX(start_block_idx);
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
      if constexpr (kEnableStageTiming) {
        stage_timing_depend(static_cast<uint32_t>(nxt_token_index));
        if (stage_timing_lane_leader && stage_timing_seen_first_tile) {
          stage_timing.load_cycles += clock64() - initial_index_start;
        }
      }
#endif

      CUTE_NO_UNROLL
      for (int block_idx = start_block_idx; block_idx < end_block_idx; ++block_idx) {
        const int buf_idx = (block_idx - start_block_idx) % NUM_K_BUFS;
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
        const bool stage_timing_measure_tile = stage_timing_seen_first_tile;
#endif

        bf16* sK_nope_base = plan.u.k[buf_idx].data() + (idx_in_cluster * (TOPK_BLOCK_SIZE / 2) + my_token_idx) * 8 +
                             ((lane_idx / 8) * 16) * TOPK_BLOCK_SIZE;
        bf16* sK_nope_peer_base = get_peer_addr(sK_nope_base);
        transac_bar_t* peer_bar_k_remote_ready = get_peer_addr(&plan.bar_k_remote_ready[buf_idx]);

        const int token_index = nxt_token_index;
        if (block_idx + 1 != end_block_idx) {
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
          uint64_t next_index_start = 0;
          if constexpr (kEnableStageTiming) {
            if (stage_timing_lane_leader) {
              next_index_start = clock64();
            }
          }
#endif
          nxt_token_index = GET_TOKEN_INDEX(block_idx + 1);
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
          if constexpr (kEnableStageTiming) {
            stage_timing_depend(static_cast<uint32_t>(nxt_token_index));
            if (stage_timing_lane_leader) {
              // This prefetch belongs to the following tile.  It is always a
              // measured tile because only the CTA's very first tile is
              // discarded.
              stage_timing.load_cycles += clock64() - next_index_start;
            }
          }
#endif
        }
        const bool token_is_valid = token_index >= 0 && token_index < params.num_blocks * PAGE_BLOCK_SIZE;
        const int safe_token_index = token_is_valid ? token_index : 0;
        const int block_index = safe_token_index / PAGE_BLOCK_SIZE;
        const int rel_idx_in_block = safe_token_index % PAGE_BLOCK_SIZE;
        const uint8_t* gK_base = static_cast<const uint8_t*>(params.k_ptr) + block_index * params.k_batch_stride +
                                 rel_idx_in_block * params.k_row_stride;

#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
        uint64_t producer_wait_start = 0;
        if constexpr (kEnableStageTiming) {
          if (stage_timing_lane_leader) {
            producer_wait_start = clock64();
          }
        }
#endif
        plan.bar_k_avail[buf_idx].wait((bar_phase_k >> buf_idx & 1) ^ 1);
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
        if constexpr (kEnableStageTiming) {
          if (stage_timing_lane_leader && stage_timing_measure_tile) {
            stage_timing.producer_available_wait_cycles += clock64() - producer_wait_start;
          }
        }
#endif
        bar_phase_k ^= 1 << buf_idx;

#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
        uint64_t remote_ready_start = 0;
        if constexpr (kEnableStageTiming) {
          if (idx_in_warpgroup == 0) {
            remote_ready_start = clock64();
          }
        }
#endif
        if (idx_in_warpgroup == 0) {
          plan.bar_k_remote_ready[buf_idx].arrive_and_expect_tx(
              (TOPK_BLOCK_SIZE / 2) * (HEAD_DIM_NOPE + HEAD_DIM_ROPE) * sizeof(bf16));
        }
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
        if constexpr (kEnableStageTiming) {
          if (idx_in_warpgroup == 0 && stage_timing_measure_tile) {
            stage_timing.handoff_cycles += clock64() - remote_ready_start;
          }
        }
#endif

        CUTE_UNROLL
        for (int dim_idx = 0; dim_idx < HEAD_DIM_NOPE / 64; ++dim_idx) {
          const int logical_dim = dim_idx * 64 + (lane_idx / 8) * 16;
          uint64_t packed = 0;
          uint8_t block_scale_bits = 0;
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
          uint64_t stage_start = 0;
          if constexpr (kEnableStageTiming) {
            if (stage_timing_lane_leader) {
              stage_start = clock64();
            }
          }
#endif
          if (token_is_valid) {
            packed = nvfp4::load_packed_e2m1x16(gK_base + logical_dim / 2);
            block_scale_bits =
                nvfp4::load_scale_e4m3_bits(gK_base + nvfp4::kPackedLatentBytes + logical_dim / nvfp4::kBlockSize);
          }
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
          if constexpr (kEnableStageTiming) {
            stage_timing_depend(packed ^ static_cast<uint64_t>(block_scale_bits));
            if (stage_timing_lane_leader && stage_timing_measure_tile) {
              stage_timing.load_cycles += clock64() - stage_start;
            }
            if (stage_timing_lane_leader) {
              stage_start = clock64();
            }
          }
#endif

          float effective_scale = 0.0f;
          if (token_is_valid) {
            effective_scale = nvfp4::e4m3_bits_to_float(block_scale_bits) * kv_global_scale;
          }

          const nvfp4::bf16x16 dequant = nvfp4::dequant_e2m1x16(packed, effective_scale);
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
          if constexpr (kEnableStageTiming) {
            const uint64_t* dequant_bits = reinterpret_cast<const uint64_t*>(&dequant);
            stage_timing_depend(dequant_bits[0] ^ dequant_bits[1] ^ dequant_bits[2] ^ dequant_bits[3]);
            if (stage_timing_lane_leader && stage_timing_measure_tile) {
              stage_timing.dequant_cycles += clock64() - stage_start;
            }
            if (stage_timing_lane_leader) {
              stage_start = clock64();
            }
          }
#endif
          const int smem_offset = dim_idx * 64 * TOPK_BLOCK_SIZE;
          *reinterpret_cast<__int128_t*>(sK_nope_base + smem_offset) =
              *reinterpret_cast<const __int128_t*>(&dequant.lo);
          *reinterpret_cast<__int128_t*>(sK_nope_base + smem_offset + 8 * TOPK_BLOCK_SIZE) =
              *reinterpret_cast<const __int128_t*>(&dequant.hi);
          st_async_128b(sK_nope_peer_base + smem_offset, dequant.lo, peer_bar_k_remote_ready);
          st_async_128b(sK_nope_peer_base + smem_offset + 8 * TOPK_BLOCK_SIZE, dequant.hi, peer_bar_k_remote_ready);
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
          if constexpr (kEnableStageTiming) {
            if (stage_timing_lane_leader && stage_timing_measure_tile) {
              stage_timing.handoff_cycles += clock64() - stage_start;
            }
          }
#endif
        }

        const bf16* gK_rope = reinterpret_cast<const bf16*>(gK_base + nvfp4::kPackedLatentBytes + nvfp4::kScaleBytes) +
                              (lane_idx / 8) * 8;
        bf16* sK_rope_base = plan.u.k[buf_idx].data() + (idx_in_cluster * (TOPK_BLOCK_SIZE / 2) + my_token_idx) * 8 +
                             ((lane_idx / 8) * 8) * TOPK_BLOCK_SIZE;
        bf16* sK_rope_peer_base = get_peer_addr(sK_rope_base);

        CUTE_UNROLL
        for (int dim_idx = 0; dim_idx < HEAD_DIM_ROPE / 32; ++dim_idx) {
          bf16x8 rope = {};
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
          uint64_t stage_start = 0;
          if constexpr (kEnableStageTiming) {
            if (stage_timing_lane_leader) {
              stage_start = clock64();
            }
          }
#endif
          if (token_is_valid) {
            rope = load_128b_from_gmem<bf16x8, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(gK_rope + dim_idx * 32);
          }
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
          if constexpr (kEnableStageTiming) {
            const uint64_t* rope_bits = reinterpret_cast<const uint64_t*>(&rope);
            stage_timing_depend(rope_bits[0] ^ rope_bits[1]);
            if (stage_timing_lane_leader && stage_timing_measure_tile) {
              stage_timing.load_cycles += clock64() - stage_start;
            }
            if (stage_timing_lane_leader) {
              stage_start = clock64();
            }
          }
#endif
          const int smem_offset = (HEAD_DIM_NOPE + dim_idx * 32) * TOPK_BLOCK_SIZE;
          *reinterpret_cast<__int128_t*>(sK_rope_base + smem_offset) = *reinterpret_cast<const __int128_t*>(&rope);
          st_async_128b(sK_rope_peer_base + smem_offset, rope, peer_bar_k_remote_ready);
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
          if constexpr (kEnableStageTiming) {
            if (stage_timing_lane_leader && stage_timing_measure_tile) {
              stage_timing.handoff_cycles += clock64() - stage_start;
            }
          }
#endif
        }

#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
        uint64_t handoff_start = 0;
        if constexpr (kEnableStageTiming) {
          if (stage_timing_lane_leader) {
            handoff_start = clock64();
          }
        }
#endif
        fence_view_async_shared();
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
        if constexpr (kEnableStageTiming) {
          if (stage_timing_lane_leader && stage_timing_measure_tile) {
            stage_timing.handoff_cycles += clock64() - handoff_start;
          }
        }
#endif

        if (idx_in_warpgroup < 32) {
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
          uint64_t selected_load_start = 0;
          if constexpr (kEnableStageTiming) {
            if (stage_timing_lane_leader) {
              selected_load_start = clock64();
            }
          }
#endif
          const int2 selected = __ldg(reinterpret_cast<int2*>(gIndices + block_idx * TOPK_BLOCK_SIZE + lane_idx * 2));
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
          if constexpr (kEnableStageTiming) {
            stage_timing_depend(
                static_cast<uint32_t>(selected.x) | (static_cast<uint64_t>(static_cast<uint32_t>(selected.y)) << 32));
            if (stage_timing_lane_leader && stage_timing_measure_tile) {
              stage_timing.load_cycles += clock64() - selected_load_start;
              handoff_start = clock64();
            }
          }
#endif
          const int token_capacity = params.num_blocks * PAGE_BLOCK_SIZE;
          *reinterpret_cast<char2*>(&plan.is_kv_valid[buf_idx][lane_idx * 2]) = {
              selected.x >= 0 && selected.x < token_capacity, selected.y >= 0 && selected.y < token_capacity};
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
          if constexpr (kEnableStageTiming) {
            if (stage_timing_lane_leader && stage_timing_measure_tile) {
              stage_timing.handoff_cycles += clock64() - handoff_start;
            }
          }
#endif
        }

#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
        if constexpr (kEnableStageTiming) {
          if (stage_timing_lane_leader) {
            handoff_start = clock64();
          }
        }
#endif
        plan.bar_k_local_ready[buf_idx].arrive();
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
        if constexpr (kEnableStageTiming) {
          if (stage_timing_lane_leader && stage_timing_measure_tile) {
            stage_timing.handoff_cycles += clock64() - handoff_start;
            ++stage_timing.timed_tiles;
          }
        }
        stage_timing_seen_first_tile = true;
#endif
      }

#undef GET_TOKEN_INDEX
      cute::cluster_sync();
    }
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
    if constexpr (kEnableStageTiming) {
      if (stage_timing_lane_leader) {
        write_stage_timing_record(
            stage_timing_ptr, stage_timing_cta_idx, kProducerWarp0Record + producer_warp_idx, stage_timing);
      }
    }
#endif
  }

  if (begin_idx > end_idx) {
    cute::cluster_sync();
  }
#else
  if (cute::thread0()) {
    CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
  }
#endif
}

void run_flash_splitkv_mla_nvfp4_sparse_kernel(
    DecodingParams& params, const float* kv_global_scale_ptr, cudaStream_t stream) {
  FLASH_ASSERT(params.h_k == 1);
  FLASH_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);

  auto shape_Q = make_shape(params.q_head_per_hk, params.d, params.s_q, params.b);
  auto tma_Q = cute::make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(
          make_gmem_ptr(static_cast<bf16*>(params.q_ptr)),
          make_layout(
              shape_Q,
              make_stride(
                  params.q_row_stride, _1{}, params.q_head_per_hk * params.q_row_stride, params.q_batch_stride))),
      SmemLayoutQ{});

  auto shape_O = make_shape(params.q_head_per_hk, params.d_v, params.s_q, params.b);
  auto tma_O = cute::make_tma_copy(
      SM90_TMA_STORE{},
      make_tensor(
          make_gmem_ptr(static_cast<bf16*>(params.o_ptr)),
          make_layout(
              shape_O,
              make_stride(
                  params.o_row_stride, _1{}, params.q_head_per_hk * params.o_row_stride, params.o_batch_stride))),
      SmemLayoutOBuf{});

  TmaParams<decltype(shape_Q), decltype(tma_Q), decltype(shape_O), decltype(tma_O)> tma_params = {
      shape_Q, tma_Q, shape_O, tma_O};
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
  auto mla_kernel = &flash_fwd_splitkv_mla_nvfp4_sparse_kernel<false, decltype(tma_params)>;
#else
  auto mla_kernel = &flash_fwd_splitkv_mla_nvfp4_sparse_kernel<decltype(tma_params)>;
#endif

  constexpr size_t smem_size = sizeof(SharedMemoryPlan);
  CHECK_CUDA(cudaFuncSetAttribute(mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  const int num_m_block = cute::ceil_div(params.q_head_per_hk, 2 * BLOCK_M) * 2;
  cutlass::ClusterLaunchParams launch_params = {
      dim3(num_m_block, params.s_q, params.num_sm_parts), dim3(NUM_THREADS, 1, 1), dim3(2, 1, 1), smem_size, stream};
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
  cutlass::launch_kernel_on_cluster(
      launch_params, reinterpret_cast<void*>(mla_kernel), params, tma_params, kv_global_scale_ptr, nullptr);
#else
  cutlass::launch_kernel_on_cluster(
      launch_params, reinterpret_cast<void*>(mla_kernel), params, tma_params, kv_global_scale_ptr);
#endif
  CHECK_CUDA_KERNEL_LAUNCH();
}

#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)

int get_flash_splitkv_mla_nvfp4_stage_timing_num_ctas(const DecodingParams& params) {
  const int num_m_block = cute::ceil_div(params.q_head_per_hk, 2 * BLOCK_M) * 2;
  return num_m_block * params.s_q * params.num_sm_parts;
}

void run_flash_splitkv_mla_nvfp4_sparse_profile_kernel(
    DecodingParams& params, const float* kv_global_scale_ptr, uint64_t* stage_timing_ptr, cudaStream_t stream) {
  FLASH_ASSERT(params.h_k == 1);
  FLASH_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);
  FLASH_ASSERT(stage_timing_ptr != nullptr);

  auto shape_Q = make_shape(params.q_head_per_hk, params.d, params.s_q, params.b);
  auto tma_Q = cute::make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(
          make_gmem_ptr(static_cast<bf16*>(params.q_ptr)),
          make_layout(
              shape_Q,
              make_stride(
                  params.q_row_stride, _1{}, params.q_head_per_hk * params.q_row_stride, params.q_batch_stride))),
      SmemLayoutQ{});

  auto shape_O = make_shape(params.q_head_per_hk, params.d_v, params.s_q, params.b);
  auto tma_O = cute::make_tma_copy(
      SM90_TMA_STORE{},
      make_tensor(
          make_gmem_ptr(static_cast<bf16*>(params.o_ptr)),
          make_layout(
              shape_O,
              make_stride(
                  params.o_row_stride, _1{}, params.q_head_per_hk * params.o_row_stride, params.o_batch_stride))),
      SmemLayoutOBuf{});

  TmaParams<decltype(shape_Q), decltype(tma_Q), decltype(shape_O), decltype(tma_O)> tma_params = {
      shape_Q, tma_Q, shape_O, tma_O};
  auto mla_kernel = &flash_fwd_splitkv_mla_nvfp4_sparse_kernel<true, decltype(tma_params)>;

  constexpr size_t smem_size = sizeof(SharedMemoryPlan);
  CHECK_CUDA(cudaFuncSetAttribute(mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  const int num_m_block = cute::ceil_div(params.q_head_per_hk, 2 * BLOCK_M) * 2;
  cutlass::ClusterLaunchParams launch_params = {
      dim3(num_m_block, params.s_q, params.num_sm_parts), dim3(NUM_THREADS, 1, 1), dim3(2, 1, 1), smem_size, stream};
  cutlass::launch_kernel_on_cluster(
      launch_params, reinterpret_cast<void*>(mla_kernel), params, tma_params, kv_global_scale_ptr, stage_timing_ptr);
  CHECK_CUDA_KERNEL_LAUNCH();
}

#endif

}  // namespace sm90
