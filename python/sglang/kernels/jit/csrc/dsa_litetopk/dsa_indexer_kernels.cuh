// Vendored from LiteTopK (https://github.com/Heisenberg-Yin/LiteTopK), whose
// vLLM integration is PR #48726. sglang deviations are marked SGLANG DEVIATION.
//
// LiteTopK DSA scoring kernel:
//   * fp8 MQA scoring on tcgen05 (DeepGEMM-style TMA/UMMA/math warp
//     specialization). SGLANG DEVIATION: the two math warpgroups split the
//     q-block's rows instead of the KV block's halves (each holds two rows of
//     weights, so both halves' accumulators stay in flight and a TMEM stage is
//     handed back as soon as its loads land), KV scales are read straight from
//     global, one UMMA warp per TMEM stage, and the KV block loop is unrolled
//     over the two stages so every TMEM address and barrier is static;
//   * NON-persistent KV-split scheduling: blockIdx.x = q-block, blockIdx.y =
//     KV split window, which keeps all SMs busy on tiny-Q chunks at long
//     context;
//   * LiteTopK sparse epilogue (batched-vote emit, strided gate reload,
//     warp-local candidate queues) and the spare-warp threshold-refresh daemon
//     (one q-block per CTA, fixed rows).
//
// Ragged Q handled by forcing an empty KV range on padded rows.

#pragma once

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
// SGLANG DEVIATION: DeepGEMM headers come from the installed package, not a
// vendored copy.
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace dsa_litetopk {

using namespace deep_gemm;

// Two TMEM stages: block k+1's MMAs land while block k is still being
// reduced. A KV block is two M=128 MMAs (KV halves), one slot each, and both
// math warpgroups read every slot (each takes its own rows = accumulator
// columns). Two stages x two halves x 128 columns fills TMEM.
constexpr uint32_t kNumTmemStagesPerWG = 2;

#define DSA_WARP_QUEUE_CAP 128  // candidates per (warp, row) smem queue between drains
// Heads whose ReLU runs as (a+|a|)/2 on the FMA pipe instead of FMNMX on the
// ALU pipe (the two pipes run in parallel; this split balances them).
#define DSA_ABS_HEADS 24
#define DSA_REFRESH_STRIDE 32  // KV blocks between refresh-daemon progress updates
#define DSA_GATE_STRIDE 16     // KV blocks between gate threshold reloads

#define DSA_ST_CAND_VAL(dst, v) __stcs(&(dst), (v))
#define DSA_ST_CAND_IDX(dst, v) __stcs(&(dst), (v))

template <
    uint32_t kNumHeads,
    uint32_t kHeadDim,
    uint32_t BLOCK_Q,
    uint32_t BLOCK_KV,
    uint32_t kNumQStages,
    uint32_t kNumKVStages,
    uint32_t kNumSMs,
    uint32_t kNumSpecializedThreads,
    uint32_t kNumMathThreads,
    uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1) void sm100_dsa_litetopk(
    const uint32_t seq_len,
    const uint32_t seq_len_kv,
    uint32_t* cu_seq_len_k_start,
    uint32_t* cu_seq_len_k_end,
    const float* __restrict__ origin,     // [seq_len]
    const float* __restrict__ inv_delta,  // [seq_len]
    int32_t* __restrict__ th_bucket,      // [seq_len]
    int32_t* __restrict__ bcount,         // [seq_len,
                                          // num_buckets]
    const uint32_t num_buckets,
    const uint32_t topk,
    const uint32_t refresh_every,
    const uint32_t num_kv_splits,
    const uint32_t probe_group,      // compacted-space group size
                                     // (pstp-1)*64; 0 = no probe
                                     // compaction (identity map)
    const uint64_t probe_magic,      // ceil(2^42/probe_group):
                                     // exact div via mul-shift
    const uint32_t probe_add_max,    // npage*64 cap for the map
    float* __restrict__ cand_val,    // [seq_len,
                                     // cand_cap]
    int32_t* __restrict__ cand_idx,  // [seq_len,
                                     // cand_cap]
    int32_t* __restrict__ cand_cnt,  // [seq_len]
    const uint32_t cand_cap,
    const int32_t* __restrict__ qblock_mask,  // [num_q_blocks] or null: 0 skips the q-block
    const float* __restrict__ kv_scales,      // [kv_scales_len], read directly by the math warps
    const uint32_t kv_scales_len,
    const __grid_constant__ cute::TmaDescriptor tensor_map_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
  const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);
  if (qblock_mask != nullptr and qblock_mask[blockIdx.x] == 0) return;

  using Barrier = cutlass::arch::ClusterTransactionBarrier;

  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto warpgroup_idx = warp_idx / 4;
  const auto lane_idx = ptx::get_lane_idx();
  constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;
  constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;

  DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");

  if (warp_idx == kSpecWarpStart) {
    cute::prefetch_tma_descriptor(&tensor_map_q);
    cute::prefetch_tma_descriptor(&tensor_map_kv);
    cute::prefetch_tma_descriptor(&tensor_map_weights);
  }

  static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
  static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
  static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);

  extern __shared__ __align__(512) uint8_t smem_buffer[];
  DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
  DG_STATIC_ASSERT(SMEM_WEIGHT_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
  DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");

  constexpr uint32_t kNumKVHalves = BLOCK_KV / 128;  // M=128 MMAs per KV block, one TMEM slot each
  constexpr uint32_t kNumUmmaSlots = kNumKVHalves * kNumTmemStagesPerWG;
  constexpr uint32_t kNumTmemCols = BLOCK_Q * kNumHeads * kNumUmmaSlots;
  DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

  auto smem_q = utils::PatternVisitor(
      [&](const uint32_t& i) { return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * i); });
  auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<float*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
  });
  auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<__nv_fp8_e4m3*>(
        smem_buffer +
        (SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i));
  });
  auto barrier_ptr = reinterpret_cast<Barrier*>(
      smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
      SMEM_KV_SIZE_PER_STAGE * kNumKVStages);
  auto full_q_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
  auto empty_q_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
  auto full_kv_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
  auto empty_kv_barriers =
      utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });
  auto full_umma_barriers =
      utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + i); });
  auto empty_umma_barriers = utils::PatternVisitor(
      [&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + kNumUmmaSlots + i); });

  auto tmem_ptr_in_smem =
      reinterpret_cast<uint32_t*>(barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + kNumUmmaSlots * 2);
  auto scan_done_flag = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 1);
  auto kv_progress_ptr = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 2);
  // Per-row candidate counters for the single-scanner case (num_kv_splits ==
  // 1): drains allocate their output slots from smem instead of a global
  // atomic round trip.
  auto smem_cnt = reinterpret_cast<int32_t*>(tmem_ptr_in_smem + 4);
  // (value bits, kv index) pairs: one 8-byte store per candidate.
  auto warpq = reinterpret_cast<uint2*>(smem_cnt + kNumMathWarps * BLOCK_Q);
  // Per-CTA refresh histogram (BLOCK_Q x num_buckets). When this CTA is the
  // ONLY scanner of its rows (num_kv_splits == 1, i.e. all large-Q shapes),
  // the per-candidate histogram feed goes to smem instead of RED.GLOBAL:
  // cheaper atomic, no 64-bit address math, no L2 pressure. The daemon then
  // reads global bcount (seed counts) + this smem part. Counts and totals
  // are identical to the global path, so thresholds and recall are
  // unchanged; a racing read can only UNDERcount -> looser gate -> safe.
  auto smem_hist =
      reinterpret_cast<int32_t*>(warpq + kNumMathWarps * (BLOCK_Q / kNumMathWarpGroups) * DSA_WARP_QUEUE_CAP);

  DG_STATIC_ASSERT(kNumSpecializedThreads % 128 == 0 and kNumSpecializedThreads >= 64, "Invalid threads");
  if (warp_idx == kSpecWarpStart and cute::elect_one_sync()) {
#pragma unroll
    for (uint32_t i = 0; i < kNumQStages; ++i) {
      full_q_barriers[i]->init(1);
      empty_q_barriers[i]->init(kNumMathThreads + 64);  // math threads + both UMMA warps
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumKVStages; ++i) {
      full_kv_barriers[i]->init(1);
      empty_kv_barriers[i]->init(1);  // released by the UMMA commit
    }
    *scan_done_flag = 0;
    *kv_progress_ptr = 0;
    cutlass::arch::fence_barrier_init();
  }
  if (warp_idx == kSpecWarpStart + 1) {
    if (cute::elect_one_sync()) {
      // One full/empty pair per TMEM stage (both KV halves): one tensor-core
      // commit fills it, every math thread releases it once per block.
#pragma unroll
      for (uint32_t i = 0; i < kNumTmemStagesPerWG; ++i) {
        full_umma_barriers[i]->init(1);
        empty_umma_barriers[i]->init(kNumMathThreads);
      }
      cutlass::arch::fence_barrier_init();
    }
    cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
  }
  const bool hist_in_smem = (num_kv_splits == 1) && (refresh_every > 0) && (refresh_every != 0x7fffffff);
  if (hist_in_smem) {
    for (uint32_t idx = threadIdx.x; idx < BLOCK_Q * num_buckets; idx += blockDim.x)
      smem_hist[idx] = 0;
  }
  if (threadIdx.x < BLOCK_Q) smem_cnt[threadIdx.x] = 0;
  __syncthreads();

  constexpr uint32_t kNumSpecializedRegisters = 40;
  constexpr uint32_t kNumMathRegisters = 232;

  // V1 KV-split scheduling: blockIdx.x = q-block (one per CTA), blockIdx.y =
  // contiguous KV sub-window. Split boundaries are BLOCK_KV-aligned.
  const uint32_t block_q_idx = blockIdx.x;
  const uint32_t kv_split = blockIdx.y;
  uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
  const auto load_schedule = [&](const uint32_t block_q_idx) -> cute::tuple<uint32_t, uint32_t> {
    uint32_t start = cute::numeric_limits<uint32_t>::max();
    uint32_t end = cute::numeric_limits<uint32_t>::min();

#pragma unroll
    for (uint32_t i = 0; i < BLOCK_Q; ++i) {
      const auto q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
      seq_k_start[i] = cu_seq_len_k_start[q_idx];
      seq_k_end[i] = cu_seq_len_k_end[q_idx];
      if (block_q_idx * BLOCK_Q + i >= seq_len) {
        // Padded row of a ragged final q-block: empty, aggregation-neutral.
        seq_k_start[i] = seq_len_kv;
        seq_k_end[i] = 0;
      }
      start = min(start, min(seq_k_start[i], seq_len_kv));
      end = max(end, min(seq_k_end[i], seq_len_kv));
    }
    const uint32_t total_blocks = math::ceil_div(seq_len_kv, BLOCK_KV);
    const uint32_t blocks_per_split = math::ceil_div(total_blocks, num_kv_splits);
    const uint32_t split_lo = kv_split * blocks_per_split * BLOCK_KV;
    const uint32_t split_hi = min((kv_split + 1) * blocks_per_split * BLOCK_KV, seq_len_kv);
    start = start / 4 * 4;  // keeps KV block starts 16B-aligned in the scale array
    if (start < split_lo) start = split_lo;
    if (end > split_hi) end = split_hi;
    const uint32_t nkv = (end > start) ? math::ceil_div(end - start, BLOCK_KV) : 0;
    return {start, nkv};
  };

  const auto get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
    return {kv_block_idx % kNumKVStages, (kv_block_idx / kNumKVStages) & 1};
  };

  constexpr uint32_t UMMA_M = 128;
  constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
  constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;

  if (warp_idx == kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    if (cute::elect_one_sync()) {
      if (block_q_idx < num_q_blocks) {
        // Q + weights once for this q-block.
        tma::copy<kHeadDim, BLOCK_Q * kNumHeads, kHeadDim>(
            &tensor_map_q, full_q_barriers[0], smem_q[0], 0, block_q_idx * BLOCK_Q * kNumHeads);
        tma::copy<kNumHeads, BLOCK_Q, 0>(
            &tensor_map_weights, full_q_barriers[0], smem_weights[0], 0, block_q_idx * BLOCK_Q);
        full_q_barriers[0]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);

        CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
        for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
          CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
          empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

          tma::copy<kHeadDim, BLOCK_KV, kHeadDim>(
              &tensor_map_kv,
              full_kv_barriers[kv_stage_idx],
              smem_kv[kv_stage_idx],
              0,
              kv_start + kv_block_idx * BLOCK_KV);
          full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE);
        }
      }
    }
  } else if (warp_idx == kSpecWarpStart + 1 or warp_idx == kSpecWarpStart + 2) {
    // One UMMA warp per TMEM stage: warp +1 issues the even KV blocks, warp
    // +2 the odd ones, so the issue cost is spread over two SM sub-partitions.
    // A KV block is two M=128 MMAs (KV halves), each into its own slot of the
    // stage, committed together.
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    const uint32_t tmem_stage = warp_idx - (kSpecWarpStart + 1);
    DG_STATIC_ASSERT(kNumTmemStagesPerWG == 2, "One UMMA warp per TMEM stage");

    DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

    auto instr_desc = cute::UMMA::make_instr_desc<
        cutlass::float_e4m3_t,
        cutlass::float_e4m3_t,
        float,
        UMMA_M,
        UMMA_N,
        cute::UMMA::Major::K,
        cute::UMMA::Major::K>();
    auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

    if (block_q_idx < num_q_blocks) {
      CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
      full_q_barriers[0]->wait(0);

      for (uint32_t kvg = tmem_stage; kvg < num_kv_blocks; kvg += kNumTmemStagesPerWG) {
        CUTE_TIE_DECL(get_kv_pipeline(kvg), kv_stage_idx, kv_phase);
        full_kv_barriers[kv_stage_idx]->wait(kv_phase);

        DG_STATIC_ASSERT(BLOCK_KV == kNumKVHalves * UMMA_M, "Invalid block size");
        DG_STATIC_ASSERT(kHeadDim % UMMA_K == 0, "Invalid head dim");
        const uint32_t tmem_phase = (kvg / kNumTmemStagesPerWG) & 1;
        empty_umma_barriers[tmem_stage]->wait(tmem_phase ^ 1);
        ptx::tcgen05_after_thread_sync();
#pragma unroll
        for (uint32_t h = 0; h < kNumKVHalves; ++h) {
          const uint32_t slot = tmem_stage * kNumKVHalves + h;
#pragma unroll
          for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++k) {
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                smem_kv[kv_stage_idx], h * UMMA_M, k * UMMA_K);
            auto b_desc =
                mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(smem_q[0], 0, k * UMMA_K);
            cute::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, slot * UMMA_N, k, runtime_instr_desc);
          }
        }
        // One commit covers both halves; the KV stage is consumed by the MMAs
        // alone (scales come from global), so the tensor core releases it too.
        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(full_umma_barriers[tmem_stage]));
        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_kv_barriers[kv_stage_idx]));
      }
      empty_q_barriers[0]->arrive();
    }
  } else if (warp_idx == kSpecWarpStart + 3) {
    // Spare-warp threshold-refresh daemon (V1 semantics: fixed rows).
    // NOTE: moving this refresh into the math warps' gate-reload point
    // (tidal-style C2) measured 12-13% SLOWER at 256K/512K here: unlike
    // tidal, this kernel has no register spill (setmaxnreg 232/40) and
    // sleeping was only 0.28 cyc/issue — the daemon overlaps well, while
    // inline refresh puts bcount global-read latency on the math warps'
    // critical path (a warpgroup hiccup every GATE_STRIDE blocks).
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    const bool in_scan_refresh = (refresh_every > 0 && refresh_every != 0x7fffffff);
    if (in_scan_refresh && block_q_idx < num_q_blocks) {
      const auto refresh_row = [&](const uint32_t row) {
        if (row >= seq_len) return;
        const int32_t* brow = bcount + static_cast<uint64_t>(row) * num_buckets;
        const int32_t* srow = smem_hist + (row - block_q_idx * BLOCK_Q) * num_buckets;
        int carry = 0;
        int found = static_cast<int>(num_buckets) - 1;
        bool done = false;
        for (uint32_t base = 0; base < num_buckets && !done; base += 32) {
          uint32_t b = base + lane_idx;
          int v = (b < num_buckets) ? brow[b] : 0;
          if (hist_in_smem && b < num_buckets) v += srow[b];
          int prefix = v;
#pragma unroll
          for (int off = 1; off < 32; off <<= 1) {
            int nsh = __shfl_up_sync(0xffffffffu, prefix, off);
            if (static_cast<int>(lane_idx) >= off) prefix += nsh;
          }
          int incl = carry + prefix;
          bool hit = (b < num_buckets) && (incl >= static_cast<int>(topk)) && (incl - v < static_cast<int>(topk));
          unsigned hm = __ballot_sync(0xffffffffu, hit);
          if (hm) {
            found = static_cast<int>(base) + (__ffs(hm) - 1);
            done = true;
          } else {
            carry += __shfl_sync(0xffffffffu, prefix, 31);
          }
        }
        if (lane_idx == 0 && found < th_bucket[row]) th_bucket[row] = found;
      };
      int last_prog = 0;
      while (true) {
        const int done = *scan_done_flag;
        const int prog = *kv_progress_ptr;
        if (prog > last_prog) {
#pragma unroll 1
          for (uint32_t r = 0; r < BLOCK_Q; ++r)
            refresh_row(block_q_idx * BLOCK_Q + r);
          last_prog = prog;
        } else if (done) {
#pragma unroll 1
          for (uint32_t r = 0; r < BLOCK_Q; ++r)
            refresh_row(block_q_idx * BLOCK_Q + r);
          break;
        } else {
          __nanosleep(2000);
        }
      }
    }
  } else if (warp_idx < kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

    // Row-split warpgroups: warpgroup g scores rows [g*kRowsPerWG, +kRowsPerWG)
    // of the q-block for every KV position of the block. A KV block is two
    // M=128 MMAs (KV halves) in two TMEM slots; a math warp reads its 32 KV
    // lanes x its warpgroup's kColsPerWG accumulator columns from each. Holding
    // two rows of weights instead of four leaves room to keep both halves'
    // accumulators in flight, so one half's TMEM load latency hides under the
    // other half's reduction.
    constexpr uint32_t kRowsPerWG = BLOCK_Q / kNumMathWarpGroups;
    constexpr uint32_t kColsPerWG = kRowsPerWG * kNumHeads;
    constexpr uint32_t kNumAbsHeads = DSA_ABS_HEADS;
    DG_STATIC_ASSERT(kNumAbsHeads % 2 == 0 and kNumAbsHeads <= kNumHeads, "Head pairs");
    DG_STATIC_ASSERT(BLOCK_Q % kNumMathWarpGroups == 0, "Rows must split evenly over the math warpgroups");
    DG_STATIC_ASSERT(
        kNumKVHalves == 2 and kNumTmemStagesPerWG == 2 and kRowsPerWG == 2,
        "Two KV halves, two TMEM stages, two rows per warpgroup");
    const uint32_t quad = warp_idx % 4;  // TMEM lane quadrant = KV sub-range of a half
    const uint32_t row0_local = warpgroup_idx * kRowsPerWG;
    const auto math_thread_idx = warp_idx * 32 + lane_idx;

    auto tmem_load_async = [](const uint32_t& tmem_addr, float* accum) {
      DG_STATIC_ASSERT(kColsPerWG == 64, "Unsupported TMEM load size");
      [&]<size_t... Is>(cute::index_sequence<Is...>) {
        cute::SM100_TMEM_LOAD_32dp32b64x::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
      }(cute::make_index_sequence<kColsPerWG>{});
    };
    // tcgen05.wait::ld, pinned after the value it is passed so the reduction
    // it should overlap cannot be scheduled below it.
    const auto tmem_wait_after = [](const float dep) {
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::"f"(dep) : "memory");
    };

    float weights[kRowsPerWG][kNumHeads];
    float o_reg[kRowsPerWG], inv_reg[kRowsPerWG], vth_reg[kRowsPerWG];
    const unsigned FULL = 0xffffffffu;

    if (block_q_idx < num_q_blocks) {
      CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
      full_q_barriers[0]->wait(0);

      // This warpgroup's rows of the weights, into registers, with the
      // per-row bucket scale folded in: the ReLU-weighted sum then lands
      // directly in bucket units.
#pragma unroll
      for (uint32_t i = 0; i < kRowsPerWG; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < kNumHeads; ++j)
          weights[i][j] = ptx::ld_shared(smem_weights[0] + (row0_local + i) * kNumHeads + j);
      }
      // Queue fill counts are warp-uniform: every lane tracks them
      // redundantly in registers, so the hot emit path needs no smem
      // bookkeeping and no shfl broadcast.
      int qn_reg[kRowsPerWG];
      uint32_t ks_reg[kRowsPerWG], klen_reg[kRowsPerWG];
#pragma unroll
      for (uint32_t i = 0; i < kRowsPerWG; ++i) {
        const uint32_t rq = min(block_q_idx * BLOCK_Q + row0_local + i, seq_len - 1);
        o_reg[i] = origin[rq];
        inv_reg[i] = inv_delta[rq];
        // Bucket-space gate: bq = fmaf(scale_kv, sum, c0) with c0 = -origin*inv;
        // the gate compares bq's bits against the edge float(g+1) (>= 1, so
        // every negative pattern passes). cand_val stores bq itself.
        vth_reg[i] = -o_reg[i] * inv_reg[i];
        o_reg[i] = 0.0f;  // gate closed until the first consume
        qn_reg[i] = 0;
        ks_reg[i] = seq_k_start[row0_local + i];
        klen_reg[i] = seq_k_end[row0_local + i] > seq_k_start[row0_local + i]
                          ? seq_k_end[row0_local + i] - seq_k_start[row0_local + i]
                          : 0u;
#pragma unroll
        for (uint32_t j = 0; j < kNumHeads; ++j)
          weights[i][j] *= (j < kNumAbsHeads ? -0.5f : -1.0f) * inv_reg[i];
      }

      // Gate prefetch: th_bucket is tightened concurrently by the refresh
      // daemon; consume the value fetched one window earlier and issue the
      // next window's load. A one-window-stale gate is recall-safe: refresh
      // only tightens.
      int th_pf[kRowsPerWG];
#pragma unroll
      for (uint32_t i = 0; i < kRowsPerWG; ++i)
        th_pf[i] = __ldcg(th_bucket + min(block_q_idx * BLOCK_Q + row0_local + i, seq_len - 1));

      // KV scales straight from global (L1/L2 resident), two blocks ahead:
      // the math warps never touch the KV smem stages.
      const uint32_t lane_kv = quad * 32 + lane_idx;  // this lane's KV row within a half
      const auto load_scale = [&](const uint32_t blk, const uint32_t half) {
        const uint32_t idx = min(kv_start + blk * BLOCK_KV + half * UMMA_M + lane_kv, kv_scales_len - 1);
        return __ldg(kv_scales + idx);
      };
      float scale_pf[2][kNumKVHalves];
#pragma unroll
      for (uint32_t h = 0; h < kNumKVHalves; ++h) {
        scale_pf[0][h] = load_scale(0, h);
        scale_pf[1][h] = load_scale(1, h);
      }
      const bool single_cta = (num_kv_splits == 1);

      // Drain a (warp,row) queue segment to the global candidate buffer.
      const auto drain_queue = [&](const uint32_t row_q, const uint32_t queue_base, const int qn, const int base) {
        const uint64_t out_base = static_cast<uint64_t>(row_q) * cand_cap;
        for (int t = static_cast<int>(lane_idx); t < qn; t += 32) {
          const uint2 e = warpq[queue_base + t];
          const float x = __uint_as_float(e.x);
          uint32_t kvo = e.y;
          if (probe_group != 0) {
            // compacted -> original position; exact c/probe_group via
            // magic mul-shift
            const uint32_t sup = (uint32_t)(((uint64_t)kvo * probe_magic) >> 42);
            kvo += min((sup + 1) * 64u, probe_add_max);
          }
          const int w = base + t;
          if (w < static_cast<int>(cand_cap)) {
            DSA_ST_CAND_VAL(cand_val[out_base + w], x);
            DSA_ST_CAND_IDX(cand_idx[out_base + w], static_cast<int32_t>(kvo));
          }
        }
      };

      // Hand a TMEM stage back to its UMMA warp.
      const auto release_slot = [&](const uint32_t stage) {
        ptx::tcgen05_before_thread_sync();
        empty_umma_barriers[stage]->arrive();
      };

      // ---- KV block loop ---------------------------------------------------
      // Block kb lives in TMEM stage (kb & 1), slots (stage*2 + half), with
      // barrier phase (kb >> 1) & 1. The loop is unrolled by two so the stage
      // -- and with it every TMEM address and barrier -- is a compile-time
      // constant; the gate reload and the progress update run once per
      // DSA_GATE_STRIDE blocks around it.
      const uint32_t tmem_col_wg = warpgroup_idx * kColsPerWG;
      float acc_a[kColsPerWG], acc_b[kColsPerWG];  // KV half 0 / half 1 of the current block
      float v_row[kNumKVHalves * kRowsPerWG];      // [half*kRowsPerWG + row]
      uint32_t pass_bits = 0;

      const auto score_row = [&](const uint32_t i,
                                 const float* accum,
                                 const float scale_kv,
                                 const uint32_t kv_offset,
                                 const uint32_t bit) {
        // ReLU-weighted head sum, balanced over the two 32-wide pipes: the
        // first kNumAbsHeads heads use relu(a)*w = (a+|a|)*(w/2) (a packed
        // FADD2 on the FMA pipe, the 1/2 folded into the weights), the rest
        // FMNMX on the ALU pipe; one FFMA2 per head pair either way, two
        // independent partial sums.
        auto sum_0 = make_float2(0, 0);
        auto sum_1 = make_float2(0, 0);
        const auto transform = [&](const uint32_t& j, const float2& sum) {
          const auto a = make_float2(accum[j], accum[j + 1]);
          const auto r = j < kNumAbsHeads ? __fadd2_rn(a, make_float2(fabsf(a.x), fabsf(a.y)))
                                          : make_float2(fmaxf(a.x, 0.0f), fmaxf(a.y, 0.0f));
          const auto b = make_float2(weights[i][j], weights[i][j + 1]);
          return __ffma2_rn(r, b, sum);
        };
#pragma unroll
        for (uint32_t j = 0; j < kNumHeads; j += 4) {
          sum_0 = transform(j, sum_0);
          sum_1 = transform(j + 2, sum_1);
        }
        const auto sum = __fadd2_rn(sum_0, sum_1);
        // NaN maps to a large positive pattern and is dropped.
        const float bq = fmaf(scale_kv, sum.x + sum.y, vth_reg[i]);
        v_row[bit] = bq;
        const bool g = (__float_as_int(bq) < __float_as_int(o_reg[i])) and (kv_offset - ks_reg[i] < klen_reg[i]);
        pass_bits |= g ? (1u << bit) : 0u;
        return bq;
      };

      // Insert this warp's hits for row `i` (both KV halves) into its queue.
      // `m` is the ballot of lanes that insert one entry, `sel_h1` says which
      // half a lane's entry comes from.
      const auto insert_hits =
          [&](const uint32_t i, const unsigned m, const bool take, const bool sel_h1, const uint32_t kv_off0) {
            const uint32_t row_q = block_q_idx * BLOCK_Q + row0_local + i;
            const uint32_t queue_base = (warp_idx * kRowsPerWG + i) * DSA_WARP_QUEUE_CAP;
            const int cnt = __popc(m);
            int qn = qn_reg[i];
            if (qn + cnt > static_cast<int>(DSA_WARP_QUEUE_CAP)) {
              // Drain: one returning atomic per ~DSA_WARP_QUEUE_CAP candidates,
              // against a smem counter when this CTA is the row's only scanner.
              int base = 0;
              if (lane_idx == 0)
                base = single_cta ? atomicAdd(smem_cnt + row0_local + i, qn) : atomicAdd(cand_cnt + row_q, qn);
              base = __shfl_sync(FULL, base, 0);
              drain_queue(row_q, queue_base, qn, base);
              qn = 0;
              __syncwarp(FULL);  // queue slots reusable
            }
            if (take) {
              const float x = sel_h1 ? v_row[kRowsPerWG + i] : v_row[i];  // bucket-space value IS the payload
              const uint32_t kvo = kv_off0 + (sel_h1 ? UMMA_M : 0u);
              const unsigned below = (1u << lane_idx) - 1u;
              const int pos = qn + __popc(m & below);
              warpq[queue_base + pos] = make_uint2(__float_as_uint(x), kvo);
              if (refresh_every > 0) {
                // Histogram feed for the refresh daemon: one F2I off the stored
                // bucket float, bucket-identical to the gate.
                const int braw = static_cast<int>(x);
                const int b =
                    braw < 0 ? 0
                             : (braw > static_cast<int>(num_buckets) - 1 ? static_cast<int>(num_buckets) - 1 : braw);
                if (hist_in_smem) {
                  atomicAdd(smem_hist + (row0_local + i) * num_buckets + b, 1);
                } else {
                  atomicAdd(&bcount[static_cast<uint64_t>(row_q) * num_buckets + b], 1);
                }
              }
            }
            qn_reg[i] = qn + cnt;
          };

      const auto emit_block = [&](const uint32_t kv_off0) {
        // redux pruning: inside an active block, one redux.sync.or gives the
        // warp-wide union of hit (row, half) pairs, so the ballots and queue
        // bookkeeping run only for rows with hits. A lane's two KV positions
        // of a row go through one ballot (a lane with hits in both halves,
        // rare, takes a second one). Warp-uniform branches throughout.
        if (__any_sync(FULL, pass_bits)) {
          const uint32_t union_bits = __reduce_or_sync(FULL, pass_bits);
#pragma unroll
          for (uint32_t i = 0; i < kRowsPerWG; ++i) {
            if (union_bits & ((1u << i) | (1u << (kRowsPerWG + i)))) {
              const bool g0 = (pass_bits >> i) & 1u;
              const bool g1 = (pass_bits >> (kRowsPerWG + i)) & 1u;
              const unsigned m = __ballot_sync(FULL, g0 | g1);
              insert_hits(i, m, g0 | g1, !g0, kv_off0);
              const unsigned m2 = __ballot_sync(FULL, g0 & g1);
              if (m2) insert_hits(i, m2, g0 & g1, true, kv_off0);
            }
          }
        }
      };

      // One KV block in TMEM stage STAGE. Half 0 is already in acc_a (its
      // wait::ld outstanding); with prefetch_next, half 0 of block kb + 1 is
      // loaded from the other stage before this block's second half is reduced.
      const auto process_block = [&](auto stage_c, const uint32_t kb, const uint32_t phase, const bool prefetch_next) {
        constexpr uint32_t STAGE = decltype(stage_c)::value;
        constexpr uint32_t NSTAGE = STAGE ^ 1u;
        constexpr uint32_t SLOT1 = STAGE * kNumKVHalves + 1;
        constexpr uint32_t NSLOT0 = NSTAGE * kNumKVHalves;
        const uint32_t kv_off0 = kv_start + kb * BLOCK_KV + lane_kv;
        const float scale0 = scale_pf[0][0], scale1 = scale_pf[0][1];
        scale_pf[0][0] = scale_pf[1][0];
        scale_pf[0][1] = scale_pf[1][1];
        scale_pf[1][0] = load_scale(kb + 2, 0);
        scale_pf[1][1] = load_scale(kb + 2, 1);
        pass_bits = 0;
        // Half 0 landed (the stage's single commit covered half 1 too, so its
        // load can go straight out; it lands while half 0 is reduced).
        cutlass::arch::fence_view_async_tmem_load();
        tmem_load_async(SLOT1 * UMMA_N + tmem_col_wg, acc_b);
        float last = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < kRowsPerWG; ++i)
          last = score_row(i, acc_a + i * kNumHeads, scale0, kv_off0, i);
        // Keep the half-0 reduction above the release: sunk below it, the
        // next TMEM load into acc_a would stall on its pending register reads.
        asm volatile("" ::"r"(pass_bits), "f"(v_row[0]), "f"(v_row[1]));
        tmem_wait_after(last);
        // Both halves consumed from TMEM: hand the stage back (one arrive).
        release_slot(STAGE);
        if (prefetch_next) {
          full_umma_barriers[NSTAGE]->wait(STAGE == 1 ? phase ^ 1u : phase);
          ptx::tcgen05_after_thread_sync();
          tmem_load_async(NSLOT0 * UMMA_N + tmem_col_wg, acc_a);
        }
#pragma unroll
        for (uint32_t i = 0; i < kRowsPerWG; ++i)
          score_row(i, acc_b + i * kNumHeads, scale1, kv_off0 + UMMA_M, kRowsPerWG + i);
        emit_block(kv_off0);
      };

      if (num_kv_blocks > 0) {
        full_umma_barriers[0]->wait(0);
        ptx::tcgen05_after_thread_sync();
        tmem_load_async(tmem_col_wg, acc_a);
      }
      for (uint32_t kb0 = 0; kb0 < num_kv_blocks; kb0 += DSA_GATE_STRIDE) {
        // Gate window: consume the threshold prefetched a window ago (edge =
        // float(g+1), exact for small ints) and issue the next window's load.
#pragma unroll
        for (uint32_t i = 0; i < kRowsPerWG; ++i) {
          o_reg[i] = static_cast<float>(th_pf[i] + 1);
          th_pf[i] = __ldcg(th_bucket + min(block_q_idx * BLOCK_Q + row0_local + i, seq_len - 1));
        }
        const uint32_t kend = min(kb0 + DSA_GATE_STRIDE, num_kv_blocks);
        for (uint32_t kb = kb0; kb < kend; kb += 2) {
          const uint32_t phase = (kb >> 1) & 1u;
          process_block(cute::Int<0>{}, kb, phase, kb + 1 < num_kv_blocks);
          if (kb + 1 < kend) process_block(cute::Int<1>{}, kb + 1, phase, kb + 2 < num_kv_blocks);
        }
        if (math_thread_idx == 0 && ((kend % DSA_REFRESH_STRIDE) == 0 || kend == num_kv_blocks)) {
          __threadfence_block();
          *kv_progress_ptr = static_cast<int>(kend);
        }
      }

      // Flush this CTA's warp queues (counts live in qn_reg).
#pragma unroll
      for (uint32_t i = 0; i < kRowsPerWG; ++i) {
        const uint32_t row_q = block_q_idx * BLOCK_Q + row0_local + i;
        const int qn = qn_reg[i];
        if (row_q < seq_len && qn > 0) {
          const uint32_t queue_base = (warp_idx * kRowsPerWG + i) * DSA_WARP_QUEUE_CAP;
          int base = 0;
          if (lane_idx == 0)
            base = single_cta ? atomicAdd(smem_cnt + row0_local + i, qn) : atomicAdd(cand_cnt + row_q, qn);
          base = __shfl_sync(FULL, base, 0);
          drain_queue(row_q, queue_base, qn, base);
        }
      }

      empty_q_barriers[0]->arrive();
    }

    // Signal the refresh daemon, then free tensor memory.
    cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
    if (block_q_idx < num_q_blocks && num_kv_splits == 1 && threadIdx.x < BLOCK_Q) {
      const uint32_t row_q = block_q_idx * BLOCK_Q + threadIdx.x;
      if (row_q < seq_len) cand_cnt[row_q] = smem_cnt[threadIdx.x];
    }
    if (threadIdx.x == 0) {
      __threadfence_block();
      *scan_done_flag = 1;
    }
    if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
  }
}

}  // namespace dsa_litetopk
