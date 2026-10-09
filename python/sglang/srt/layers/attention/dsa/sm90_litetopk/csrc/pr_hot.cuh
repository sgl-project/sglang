// SPDX-License-Identifier: Apache-2.0
// vLLM PR48726 c7184685b20282848e73d6528f7bb49b9f6d5114
// Planner and carry copied verbatim. Seed changes only candidate ABI to full FP32.
#pragma once
#include <cooperative_groups.h>
namespace pair_swap_gather {

namespace cg = cooperative_groups;

constexpr int kPlanThreads = 256;
constexpr int kGatherBlockY = 32;
constexpr int kGatherThreadsX = 8;
constexpr int kGatherVecBytes = 16;
constexpr int kHotSize = 12288;

// Restore the previous epoch's swaps, mark this epoch's HOT set, collect both
// sides of the bijection, and publish the new swaps in one cooperative launch.
// HOT12288 produces exactly 48 resident CTAs on the qualified B200 path.
template <typename HotIndexT>
__global__ __launch_bounds__(kPlanThreads, 1) void cooperative_plan_kernel(
    const HotIndexT* __restrict__ hot, int* __restrict__ hot_epoch,
    int* __restrict__ permutation, int* __restrict__ swap_a,
    int* __restrict__ swap_b, int* __restrict__ counts, int hot_size,
    int window_start, int common_end, int epoch) {
  cg::grid_group grid = cg::this_grid();
  const int i = static_cast<int>(blockIdx.x) * kPlanThreads + threadIdx.x;
  const int lane = threadIdx.x & 31;

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    counts[1] = 0;
    counts[2] = 0;
  }
  const int old_count = max(0, min(counts[0], hot_size));
  if (i < old_count) {
    const int a = swap_a[i];
    const int b = swap_b[i];
    permutation[a] = a;
    permutation[b] = b;
  }
  if (i < hot_size) {
    const int64_t value = static_cast<int64_t>(hot[i]);
    if (value >= window_start && value < common_end) {
      const int previous =
          atomicExch(hot_epoch + static_cast<int>(value), epoch);
      if (previous == epoch) {
        atomicOr(counts + 3, 1);
      }
    } else {
      atomicOr(counts + 3, 2);
    }
  }
  grid.sync();

  const int64_t hot_value64 = static_cast<int64_t>(hot[i]);
  const bool hot_in_range =
      hot_value64 >= window_start && hot_value64 < common_end;
  const int hot_value =
      hot_in_range ? static_cast<int>(hot_value64) : window_start;
  const int window_value = window_start + i;
  const bool take_a = hot_in_range && hot_value >= window_start + hot_size;
  const bool take_b = hot_epoch[window_value] != epoch;

  const unsigned a_mask = __ballot_sync(0xffffffffu, take_a);
  int a_base = 0;
  if (lane == 0 && a_mask != 0) {
    a_base = atomicAdd(counts + 1, __popc(a_mask));
  }
  a_base = __shfl_sync(0xffffffffu, a_base, 0);
  if (take_a) {
    const int rank = __popc(a_mask & ((1u << lane) - 1u));
    swap_a[a_base + rank] = hot_value;
  }

  const unsigned b_mask = __ballot_sync(0xffffffffu, take_b);
  int b_base = 0;
  if (lane == 0 && b_mask != 0) {
    b_base = atomicAdd(counts + 2, __popc(b_mask));
  }
  b_base = __shfl_sync(0xffffffffu, b_base, 0);
  if (take_b) {
    const int rank = __popc(b_mask & ((1u << lane) - 1u));
    swap_b[b_base + rank] = window_value;
  }
  grid.sync();

  const int pair_count = min(counts[1], counts[2]);
  const int metadata_error = counts[3];
  if (metadata_error != 0 || counts[1] != counts[2]) {
    if (i == 0 && counts[1] != counts[2]) {
      atomicOr(counts + 3, 8);
    }
    asm volatile("trap;");
    return;
  }
  if (i < pair_count) {
    const int a = swap_a[i];
    const int b = swap_b[i];
    permutation[a] = b;
    permutation[b] = a;
  }
  if (i == 0) {
    counts[0] = pair_count;
  }
}


} // namespace pair_swap_gather
template <bool kEmitCandidates, int kRetainedHead, int BT>
__global__ void seed_prep_kernel(
    const float* __restrict__ slog, const int64_t slog_stride, const int head,
    const int NB, const int K,
    const float headroom,  // extend the bucket scale ABOVE the sample max by
                           // headroom*span (absolute, resolution-preserving
                           // when NB is scaled up with it): drifted scores
                           // land in real buckets instead of clamping to
                           // bucket 0 where refresh can never resolve them
    float* __restrict__ origin, float* __restrict__ inv_delta,
    int32_t* __restrict__ th_bucket, float* __restrict__ cand_val,
    int32_t* __restrict__ cand_idx, int32_t* __restrict__ cand_cnt,
    const int cand_cap, const int physical_index_base,
    int32_t* __restrict__ bcount_out) {
  constexpr int NSUB = 4;  // sub-histograms to spread smem atomic conflicts
  static_assert(kRetainedHead == 8192 || kRetainedHead == 12288,
                "production seed supports only the qualified 8K/12K layouts");
  constexpr int kRetainVecs = kRetainedHead / (BT * 4);
  const int row = gridDim.x - 1 - blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const float* srow = slog + (size_t)row * slog_stride;
  extern __shared__ int s_hist[];  // NSUB * NB ints

  // pass 1: min/max of the row's FINITE scores (vectorized). Ignore any -inf
  // padding in diagnostic full-row logits so it cannot poison the range.
  __shared__ float s_mx[BT / 32];
  __shared__ float s_mn[BT / 32];
  float mx = -INFINITY, mn = INFINITY;
  const auto acc = [&](const float s) {
    if (isfinite(s)) {
      mx = fmaxf(mx, s);
      mn = fminf(mn, s);
    }
  };
  // The production 8K and 12K specializations retain respectively eight
  // and twelve float4 values per thread. Keep them live across the CTA
  // reduction so histogram construction and emission never reread the
  // materialized prefix logits. Missing tail lanes carry -inf and are
  // ignored by the generic <=8K compatibility path.
  static_assert(BT == 256 || BT == 384 || BT == 512,
                "retained HOT seed requires a qualified CTA size");
  static_assert(BT % (NSUB * 32) == 0,
                "each seed sub-histogram must own whole warps");
  float4 retained[kRetainVecs];
  if (head == kRetainedHead) {
#pragma unroll
    for (int it = 0; it < kRetainVecs; ++it) {
      const int j = tid * 4 + it * BT * 4;
      const float4 s4 = *reinterpret_cast<const float4*>(srow + j);
      retained[it] = s4;
      acc(s4.x);
      acc(s4.y);
      acc(s4.z);
      acc(s4.w);
    }
  } else {
#pragma unroll
    for (int it = 0; it < kRetainVecs; ++it) {
      const int j = tid * 4 + it * BT * 4;
      float4 s4 = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
      if (j + 3 < head) {
        s4 = *reinterpret_cast<const float4*>(srow + j);
      } else {
        if (j < head) s4.x = srow[j];
        if (j + 1 < head) s4.y = srow[j + 1];
        if (j + 2 < head) s4.z = srow[j + 2];
      }
      retained[it] = s4;
      acc(s4.x);
      acc(s4.y);
      acc(s4.z);
      acc(s4.w);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, off));
  }
  if (lane == 0) {
    s_mx[tid >> 5] = mx;
    s_mn[tid >> 5] = mn;
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int wgi = 1; wgi < BT / 32; ++wgi) {
      s_mx[0] = fmaxf(s_mx[0], s_mx[wgi]);
      s_mn[0] = fminf(s_mn[0], s_mn[wgi]);
    }
  }
  __syncthreads();
  float o = -s_mx[0];         // min over x = -score
  const float hi = -s_mn[0];  // max over x
  const float span = fmaxf(hi - o, 1e-20f);
  o -= headroom * span;  // forward (above-max) drift headroom
  float inv = (NB - 1) / (span * (1.0f + headroom));
  const float vth = -o * inv;

  // pass 2: histogram in [o, inv] bucket space, NSUB sub-histograms to cut
  // smem atomic conflicts, vectorized loads.
  for (int b = tid; b < NSUB * NB; b += BT) s_hist[b] = 0;
  __syncthreads();
  int* my_hist = s_hist + (tid / (BT / NSUB)) * NB;
  const auto bucket_of = [&](const float s) -> int {
    // Use the byte-for-byte arithmetic contract consumed by both the
    // seed emitter and the main scan.  Computing (-s - o) * inv as two
    // rounded operations can put a boundary value one bucket below its
    // FMA result: the histogram would then certify K records while the
    // emitter rejects one of them, producing a silent underfill.
    const float bq = fmaf(-s, inv, vth);
    int b = static_cast<int>(bq);
    return b < 0 ? 0 : (b > NB - 1 ? NB - 1 : b);
  };
#pragma unroll
  for (int it = 0; it < kRetainVecs; ++it) {
    const float4 s4 = retained[it];
    if (isfinite(s4.x)) atomicAdd(&my_hist[bucket_of(s4.x)], 1);
    if (isfinite(s4.y)) atomicAdd(&my_hist[bucket_of(s4.y)], 1);
    if (isfinite(s4.z)) atomicAdd(&my_hist[bucket_of(s4.z)], 1);
    if (isfinite(s4.w)) atomicAdd(&my_hist[bucket_of(s4.w)], 1);
  }
  __syncthreads();
  // merge sub-histograms into s_hist[0..NB)
  for (int b = tid; b < NB; b += BT) {
    int c = s_hist[b];
#pragma unroll
    for (int g = 1; g < NSUB; ++g) c += s_hist[g * NB + b];
    s_hist[b] = c;
  }
  __syncthreads();
  if (bcount_out != nullptr) {
    // Full-row overwrite of the sample histogram. The ring daemon warm-
    // starts its refresh base from these counts: they are genuine row
    // records in the final (origin, inv) bucket space, so adding them to
    // the daemon's subset cum can only tighten the published edge safely
    // — provided the main scan starts after the sampled prefix (the
    // exact-once contract), or the same records would count twice.
    for (int b = tid; b < NB; b += BT)
      bcount_out[(size_t)row * NB + b] = s_hist[b];
  }
  // Coarse K-th estimate on the single (o, inv) scale built above. There
  // is deliberately NO scale rebuild: th_bucket, origin/inv, the emitted
  // candidates, and the exported bcount histogram must all share one
  // bucket space — the ring warm-start base is only sound under that
  // identity. Headroom above the sample max keeps drifted scores out of
  // bucket 0 where refresh could never resolve them.
  // The production U16 contract uses emit_limit==0 and a single KV split.
  // Its scan covers the complete KV range and initializes the CTA-local
  // histogram itself, so writing Q*NB zeros to global memory is dead work.
  // Find the first histogram prefix that reaches K in parallel.  The old
  // single-thread walk serialized 256 dependent shared-memory loads while
  // the other 1023 threads waited.  NB <= BT gives every bin one owner;
  // the half-open prefix ranges are disjoint, so exactly one thread writes
  // the same threshold as the serial "first cumulative sum >= K" rule.
  __shared__ int s_th;
  __shared__ int s_wsum[BT / 32];
  if (tid == 0) s_th = NB - 1;
  const int h = (tid < NB) ? s_hist[tid] : 0;
  int x = h;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, off);
    if ((tid & 31) >= off) x += y;
  }
  if ((tid & 31) == 31) s_wsum[tid >> 5] = x;
  __syncthreads();
  int base = 0;
#pragma unroll
  for (int w = 0; w < BT / 32; ++w)
    if (w < (tid >> 5)) base += s_wsum[w];
  const int incl = base + x;
  const int excl = incl - h;
  if (tid < NB && excl < K && K <= incl) s_th = tid;
  __syncthreads();
  if (tid == 0) {
    th_bucket[row] = s_th;
    origin[row] = o;
    inv_delta[row] = inv;
  }
  __syncthreads();
  if constexpr (kEmitCandidates) {
    // Large exact-once mode: the HOT scores retained above are the physical
    // prefix [physical_index_base, physical_index_base + head).  Emit their
    // passing records now, then let the main producer start after `head`.
    //
    // One iteration covers 256 * float4 == 1024 ordered columns.  A warp
    // scan plus eight warp totals gives every thread a deterministic CTA
    // prefix; this reserves no global counter per candidate.  The one CTA
    // owning the row publishes the true (possibly over-cap) total once.
    int emitted_before = 0;
    const float gate_edge = static_cast<float>(s_th + 1);
    const uint64_t row_base = static_cast<uint64_t>(row) * cand_cap;
#pragma unroll
    for (int it = 0; it < kRetainVecs; ++it) {
      const int j0 = tid * 4 + it * BT * 4;
      const float4 s4 = retained[it];
      const float score[4] = {s4.x, s4.y, s4.z, s4.w};
      float bq[4];
      bool pass[4];
      int local_count = 0;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        bq[k] = fmaf(-score[k], inv, vth);
        pass[k] = j0 + k < head && isfinite(score[k]) &&
                  __float_as_int(bq[k]) < __float_as_int(gate_edge);
        local_count += pass[k] ? 1 : 0;
      }

      int warp_inclusive = local_count;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, warp_inclusive, off);
        if (lane >= off) warp_inclusive += other;
      }
      if (lane == 31) s_wsum[tid >> 5] = warp_inclusive;
      __syncthreads();

      int warp_before = 0;
#pragma unroll
      for (int w = 0; w < BT / 32; ++w) {
        if (w < (tid >> 5)) warp_before += s_wsum[w];
      }
      const int thread_base =
          emitted_before + warp_before + warp_inclusive - local_count;
      int local_rank = 0;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        if (pass[k]) {
          const int out = thread_base + local_rank++;
          if (out < cand_cap) {
            const uint32_t physical_idx =
                static_cast<uint32_t>(physical_index_base + j0 + k);
            // SM90 exact-FP32 ABI: retain the original score, not high24 bucket bits.
            cand_val[row_base + out] = score[k];
            cand_idx[row_base + out] = physical_idx;
          }
        }
      }

      int block_total = 0;
#pragma unroll
      for (int w = 0; w < BT / 32; ++w) block_total += s_wsum[w];
      emitted_before += block_total;
      // Do not let an early warp overwrite s_wsum for the next retained
      // group while a slower warp still consumes this group's totals.
      __syncthreads();
    }
    if (tid == 0) cand_cnt[row] = emitted_before;
  } else {
    if (tid == 0) cand_cnt[row] = 0;
  }
}

constexpr int kCarryTileItems = 8192;
constexpr int kCarryMaxItems = 1 << 20;
constexpr int kCarryMaxK = 12288;
constexpr int kCarryMaxVote = 8192;
constexpr int kCarryMaxBlocks =
    (kCarryMaxItems + kCarryTileItems - 1) / kCarryTileItems;
constexpr int kCarryThreads = 256;
constexpr int kCarryWarps = kCarryThreads / 32;

enum CarryStateOffset : int {
  kCarryTicket = 0,
  kCarryThreshold = 1,
  kCarryTieBlock = 2,
  kCarryTieTake = 3,
  kCarryOutK = 4,
  kCarryNumBlocks = 5,
  kCarryBlockOffsets = 6,
};
constexpr int kCarryStateInts = kCarryBlockOffsets + kCarryMaxBlocks + 1;

__device__ __forceinline__ int carry_warp_sum(int value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ void carry_votes_plan_litetopk_kernel(
    const int32_t* __restrict__ votes, int count, int min_index, int out_k,
    int max_vote, volatile int16_t* __restrict__ partial, int partial_stride,
    int32_t* __restrict__ state) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int block = blockIdx.x;
  const int begin = block * kCarryTileItems;
  const int end = min(begin + kCarryTileItems, count);
  const int bins = max_vote + 1;

  extern __shared__ uint32_t s_freq[];
  __shared__ int s_warp_sum[kCarryWarps];
  __shared__ int s_last;
  __shared__ int s_scan_base;
  __shared__ int s_found;
  __shared__ int s_threshold;
  __shared__ int s_count_gt;
  __shared__ int s_tie_block;
  __shared__ int s_tie_take;
  __shared__ int s_block_count[kCarryMaxBlocks];

  for (int bin = tid; bin < bins; bin += kCarryThreads) {
    s_freq[bin] = 0;
  }
  __syncthreads();

  // Zero votes dominate most corpora. Count them in registers and reduce
  // once instead of serializing every zero through one shared atomic.
  int local_zero = 0;
  for (int index = begin + tid; index < end; index += kCarryThreads) {
    if (index < min_index) {
      continue;
    }
    int value = votes[index];
    // The selector emits unique winners per sampled row, so this clamp is
    // unreachable under the public ABI. Keep release builds memory-safe
    // if an upstream invariant is violated.
    value = value < 0 ? 0 : (value > max_vote ? max_vote : value);
    if (value == 0) {
      ++local_zero;
    } else {
      atomicAdd(&s_freq[value], 1u);
    }
  }
  local_zero = carry_warp_sum(local_zero);
  if (lane == 0) {
    s_warp_sum[warp] = local_zero;
  }
  __syncthreads();
  if (warp == 0) {
    int value = lane < kCarryWarps ? s_warp_sum[lane] : 0;
    value = carry_warp_sum(value);
    if (lane == 0) {
      s_freq[0] = static_cast<uint32_t>(value);
    }
  }
  __syncthreads();

  volatile int16_t* block_partial =
      partial + static_cast<size_t>(block) * partial_stride;
  for (int bin = tid; bin < bins; bin += kCarryThreads) {
    // A CTA owns at most 8192 positions, safely inside signed int16.
    block_partial[bin] = static_cast<int16_t>(s_freq[bin]);
  }
  // Every thread publishes its own global stores. A fence in tid0 alone
  // would not release the other 255 writers before the completion ticket.
  __threadfence();
  __syncthreads();

  // CUDA's canonical "last block" reduction pattern. No CTA spins: every
  // non-last block exits, while the last ticket holder sees all partial
  // writes made visible before the atomic increment.
  if (tid == 0) {
    const int old = atomicAdd(&state[kCarryTicket], 1);
    s_last = old == gridDim.x - 1;
  }
  __syncthreads();
  if (!s_last) {
    return;
  }

  for (int bin = tid; bin < bins; bin += kCarryThreads) {
    int total = 0;
    for (int source_block = 0; source_block < gridDim.x; ++source_block) {
      total += static_cast<int>(
          partial[static_cast<size_t>(source_block) * partial_stride + bin]);
    }
    s_freq[bin] = static_cast<uint32_t>(total);
  }
  __syncthreads();

  // Descending 256-bin tiles. This is the seed-prep parallel prefix in the
  // opposite direction, extended to the dynamic [0,max_vote] domain.
  if (tid == 0) {
    s_scan_base = 0;
    s_found = 0;
    s_threshold = 0;
    s_count_gt = 0;
  }
  __syncthreads();
  for (int tile = 0; tile < bins; tile += kCarryThreads) {
    const int bin = max_vote - tile - tid;
    const int count_here = bin >= 0 ? static_cast<int>(s_freq[bin]) : 0;
    int inclusive = count_here;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
      if (lane >= offset) {
        inclusive += other;
      }
    }
    if (lane == 31) {
      s_warp_sum[warp] = inclusive;
    }
    __syncthreads();
    int warp_base = 0;
#pragma unroll
    for (int source_warp = 0; source_warp < kCarryWarps; ++source_warp) {
      if (source_warp < warp) {
        warp_base += s_warp_sum[source_warp];
      }
    }
    const int exclusive = s_scan_base + warp_base + inclusive - count_here;
    const int inclusive_global = exclusive + count_here;
    if (bin >= 0 && exclusive < out_k && out_k <= inclusive_global) {
      s_threshold = bin;
      s_count_gt = exclusive;
      s_found = 1;
    }
    __syncthreads();
    if (s_found) {
      break;
    }
    if (tid == 0) {
      int tile_total = 0;
#pragma unroll
      for (int source_warp = 0; source_warp < kCarryWarps; ++source_warp) {
        tile_total += s_warp_sum[source_warp];
      }
      s_scan_base += tile_total;
    }
    __syncthreads();
  }

  if (tid == 0) {
    const int need_equal = out_k - s_count_gt;
    int equal_before = 0;
    s_tie_block = gridDim.x - 1;
    s_tie_take = 0;
    for (int source_block = 0; source_block < gridDim.x; ++source_block) {
      const int equal_here = static_cast<int>(
          partial[static_cast<size_t>(source_block) * partial_stride +
                  s_threshold]);
      if (equal_before < need_equal &&
          need_equal <= equal_before + equal_here) {
        s_tie_block = source_block;
        s_tie_take = need_equal - equal_before;
        break;
      }
      equal_before += equal_here;
    }
  }
  __syncthreads();

  // Compute each block's exact stable-output size. Warps read one partial
  // row at a time so the second partial pass remains coalesced.
  for (int source_block = warp; source_block < gridDim.x;
       source_block += kCarryWarps) {
    int selected = 0;
    for (int bin = s_threshold + 1 + lane; bin < bins; bin += 32) {
      selected += static_cast<int>(
          partial[static_cast<size_t>(source_block) * partial_stride + bin]);
    }
    selected = carry_warp_sum(selected);
    if (lane == 0) {
      int equal_take = 0;
      if (source_block < s_tie_block) {
        equal_take = static_cast<int>(
            partial[static_cast<size_t>(source_block) * partial_stride +
                    s_threshold]);
      } else if (source_block == s_tie_block) {
        equal_take = s_tie_take;
      }
      s_block_count[source_block] = selected + equal_take;
    }
  }
  __syncthreads();

  if (tid == 0) {
    int offset = 0;
    for (int source_block = 0; source_block < gridDim.x; ++source_block) {
      state[kCarryBlockOffsets + source_block] = offset;
      offset += s_block_count[source_block];
    }
    state[kCarryBlockOffsets + gridDim.x] = offset;
    state[kCarryThreshold] = s_threshold;
    state[kCarryTieBlock] = s_tie_block;
    state[kCarryTieTake] = s_tie_take;
    state[kCarryOutK] = out_k;
    state[kCarryNumBlocks] = gridDim.x;
    __threadfence();
    atomicExch(&state[kCarryTicket], 0);
  }
}

__global__ void carry_votes_emit_reset_litetopk_kernel(
    int32_t* __restrict__ votes, int count, int min_index, int max_vote,
    int64_t* __restrict__ out_idx, const int32_t* __restrict__ state) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int block = blockIdx.x;
  const int begin = block * kCarryTileItems;
  const int threshold = state[kCarryThreshold];
  const int tie_block = state[kCarryTieBlock];
  const int tie_take = state[kCarryTieTake];
  const int output_base = state[kCarryBlockOffsets + block];

  __shared__ int s_warp_count[kCarryWarps];
  __shared__ int s_warp_prefix[kCarryWarps];
  __shared__ int s_tile_output_base;
  __shared__ int s_tie_seen;
  __shared__ int s_tile_total;
  if (tid == 0) {
    s_tile_output_base = 0;
    s_tie_seen = 0;
  }
  __syncthreads();

  constexpr unsigned kFullMask = 0xffffffffu;
  const unsigned lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
  for (int tile = 0; tile < kCarryTileItems; tile += kCarryThreads) {
    const int index = begin + tile + tid;
    const bool valid = index < count;
    const int raw_value = valid ? votes[index] : 0;
    const int value =
        raw_value < 0 ? 0 : (raw_value > max_vote ? max_vote : raw_value);
    if (valid) {
      votes[index] = 0;
    }
    const bool eligible = valid && index >= min_index;
    const bool is_equal = eligible && value == threshold;

    bool take_equal = is_equal && block < tie_block;
    if (block == tie_block) {
      const unsigned equal_mask = __ballot_sync(kFullMask, is_equal);
      if (lane == 0) {
        s_warp_count[warp] = __popc(equal_mask);
      }
      __syncthreads();
      if (tid == 0) {
        int prefix = 0;
        for (int source_warp = 0; source_warp < kCarryWarps; ++source_warp) {
          s_warp_prefix[source_warp] = prefix;
          prefix += s_warp_count[source_warp];
        }
        s_tile_total = prefix;
      }
      __syncthreads();
      const int equal_rank =
          s_tie_seen + s_warp_prefix[warp] + __popc(equal_mask & lane_mask);
      take_equal = is_equal && equal_rank < tie_take;
      __syncthreads();
      if (tid == 0) {
        s_tie_seen += s_tile_total;
      }
      __syncthreads();
    }

    const bool selected = eligible && (value > threshold || take_equal);
    const unsigned selected_mask = __ballot_sync(kFullMask, selected);
    if (lane == 0) {
      s_warp_count[warp] = __popc(selected_mask);
    }
    __syncthreads();
    if (tid == 0) {
      int prefix = 0;
      for (int source_warp = 0; source_warp < kCarryWarps; ++source_warp) {
        s_warp_prefix[source_warp] = prefix;
        prefix += s_warp_count[source_warp];
      }
      s_tile_total = prefix;
    }
    __syncthreads();
    const int local_rank =
        s_warp_prefix[warp] + __popc(selected_mask & lane_mask);
    if (selected) {
      out_idx[output_base + s_tile_output_base + local_rank] =
          static_cast<int64_t>(index);
    }
    __syncthreads();
    if (tid == 0) {
      s_tile_output_base += s_tile_total;
    }
    __syncthreads();
  }
}

void carry_votes_topk_reset_litetopk_(torch::Tensor votes,
                                      torch::Tensor out_idx,
                                      torch::Tensor partial,
                                      torch::Tensor state, int64_t k64,
                                      int64_t max_vote64, int64_t min_index64) {
  TORCH_CHECK(votes.is_cuda() && out_idx.is_cuda() && partial.is_cuda() &&
                  state.is_cuda(),
              "votes/out_idx/partial/state must be CUDA tensors");
  TORCH_CHECK(votes.is_contiguous() && out_idx.is_contiguous() &&
                  partial.is_contiguous() && state.is_contiguous(),
              "votes/out_idx/partial/state must be contiguous");
  TORCH_CHECK(votes.scalar_type() == torch::kInt, "votes must be int32");
  TORCH_CHECK(out_idx.scalar_type() == torch::kLong, "out_idx must be int64");
  TORCH_CHECK(partial.scalar_type() == torch::kShort, "partial must be int16");
  TORCH_CHECK(state.scalar_type() == torch::kInt, "state must be int32");
  TORCH_CHECK(votes.device() == out_idx.device() &&
                  votes.device() == partial.device() &&
                  votes.device() == state.device(),
              "votes/out_idx/partial/state must be on the same CUDA device");
  TORCH_CHECK(votes.dim() == 1, "votes must be a 1-D histogram");
  TORCH_CHECK(out_idx.dim() == 1, "out_idx must be 1-D");
  TORCH_CHECK(partial.dim() == 2, "partial must be [blocks,bins]");
  TORCH_CHECK(state.dim() == 1 && state.numel() >= kCarryStateInts,
              "state is too small for the carry top-k ABI");

  const int64_t count64 = votes.numel();
  TORCH_CHECK(count64 >= 1 && count64 <= kCarryMaxItems,
              "votes length must be in [1,1048576]");
  TORCH_CHECK(k64 >= 1 && k64 <= kCarryMaxK, "k must be in [1,12288]");
  TORCH_CHECK(max_vote64 >= 1 && max_vote64 <= kCarryMaxVote,
              "max_vote must be in [1,8192]");
  TORCH_CHECK(min_index64 >= 0 && min_index64 < count64,
              "min_index must be in [0,votes.numel())");
  const int count = static_cast<int>(count64);
  const int min_index = static_cast<int>(min_index64);
  const int eligible = count - min_index;
  const int out_k = static_cast<int>(min(k64, static_cast<int64_t>(eligible)));
  const int max_vote = static_cast<int>(max_vote64);
  const int bins = max_vote + 1;
  const int blocks = (count + kCarryTileItems - 1) / kCarryTileItems;
  TORCH_CHECK(out_idx.numel() == out_k,
              "out_idx must have min(k,votes.numel()-min_index) elements");
  TORCH_CHECK(partial.size(0) >= blocks && partial.size(1) >= bins,
              "partial must provide at least [ceil(N/8192),max_vote+1]");

  const int partial_stride = static_cast<int>(partial.size(1));
  const size_t dynamic_smem = static_cast<size_t>(bins) * sizeof(uint32_t);
  const c10::cuda::CUDAGuard device_guard(votes.device());
  auto stream = c10::cuda::getCurrentCUDAStream();
  carry_votes_plan_litetopk_kernel<<<blocks, kCarryThreads, dynamic_smem,
                                     stream>>>(
      votes.data_ptr<int32_t>(), count, min_index, out_k, max_vote,
      partial.data_ptr<int16_t>(), partial_stride, state.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  carry_votes_emit_reset_litetopk_kernel<<<blocks, kCarryThreads, 0, stream>>>(
      votes.data_ptr<int32_t>(), count, min_index, max_vote,
      out_idx.data_ptr<int64_t>(), state.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

