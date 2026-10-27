// SPDX-License-Identifier: MIT
// Exact recovery for the blocks the persistent kernel cannot select from its
// gated candidates (a row's candidate count leaves [K, kCandidatesPerRow], or
// its weights are too small to halve exactly): the block is rescanned densely,
// and every kSegmentKeys-key segment of scores is merged into a running
// per-row top set (kept <= kRetainedPerRow, compacted to K).
#pragma once

#include "common.cuh"
#include "dense_scan.cuh"
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sglang::fused_mqa_topk::rescan {

using dense::Scheduler;

// Keys whose scores are merged into the retained sets at a time.
inline constexpr uint32_t kSegmentKeys = 1024;
// Per-row segment histograms: half_bin<kSegmentBinShift> of a score (1024 bins).
inline constexpr uint32_t kSegmentBinShift = 6;

// output: [rows, K] int32 (K = the row top-k); retained / indices: the per-row
// retained sets, addressed by absolute row (callers offset them).
struct RescanBuffers {
  float* retained;
  int32_t* indices;
  int32_t* output;
};

struct alignas(16) RescanShared {
  float ring[2][kBlockRows][kSegmentKeys];
  uint32_t hist[kBlockRows][padded_bins(1024)], groups[kBlockRows][32], gates[kBlockRows], kept[kBlockRows];
  uint32_t work[padded_bins(256)], prefix, rank, written, added, emit_counts[kBlockRows];
  __device__ void init(uint32_t tid) {
    for (uint32_t j = tid; j < kBlockRows * padded_bins(1024); j += kNumThreads)
      reinterpret_cast<uint32_t*>(hist)[j] = 0;
    if (tid < kMathThreads) reinterpret_cast<uint32_t*>(groups)[tid] = 0;
    if (tid < kBlockRows) {
      gates[tid] = __float_as_uint(-CUDART_INF_F);
      kept[tid] = 0;
    }
  }
};

// Compacts row `local`'s retained set: to the values >= its gate when that
// keeps between K and kRetainedPerRow - kSegmentKeys of them, otherwise (and
// always when `final`) to its exact top K by radix levels over the fp32 codes.
__device__ void compact_retained(
    RescanShared& s, RescanBuffers b, uint32_t K, uint32_t local, uint32_t row, uint32_t tid, bool final = false) {
  constexpr uint32_t T = kMathThreads, Items = kRetainedPerRow / T;
  uint32_t lane = tid & 31, count = s.kept[local];
  uint64_t base = (uint64_t)row * kRetainedPerRow;
  float values[Items];
  int32_t indices[Items];
#pragma unroll
  for (uint32_t i = 0; i < Items; ++i) {
    uint32_t j = tid + i * T;
    values[i] = j < count ? b.retained[base + j] : -CUDART_INF_F;
    indices[i] = j < count ? b.indices[base + j] : -1;
  }
  // Writes the values >= gate (mode 0), the codes above threshold (1) or the
  // codes equal to it (2, ties up to K) from base_offset on; returns how many
  // passed.
  auto emit = [&](float gate, uint32_t threshold, int32_t mode, uint32_t base_offset) {
    uint32_t total = 0;
#pragma unroll
    for (uint32_t i = 0; i < Items; ++i) {
      float v = values[i];
      bool pass = isfinite(v) &&
                  (mode == 0 ? v >= gate : (mode == 1 ? ordered_code(v) > threshold : ordered_code(v) == threshold));
      total += pass;
    }
    uint32_t warp_total = __reduce_add_sync(kFullMask, total);
    if (lane == 0) s.emit_counts[tid / 32] = warp_total;
    math_sync();
    uint32_t start = base_offset;
    for (uint32_t w = 0; w < tid / 32; ++w)
      start += s.emit_counts[w];
    uint32_t all = 0;
    for (uint32_t w = 0; w < 8; ++w)
      all += s.emit_counts[w];
#pragma unroll
    for (uint32_t i = 0; i < Items; ++i) {
      float v = values[i];
      bool pass = isfinite(v) &&
                  (mode == 0 ? v >= gate : (mode == 1 ? ordered_code(v) > threshold : ordered_code(v) == threshold));
      uint32_t mask = __ballot_sync(kFullMask, pass), pos = start + __popc(mask & ((1u << lane) - 1));
      if (pass && (mode != 2 || pos < K)) {
        b.retained[base + pos] = v;
        b.indices[base + pos] = indices[i];
      }
      start += __popc(mask);
    }
    math_sync();
    return all;
  };
  if (!final) {
    if (tid == 0) {
      s.rank = 0;
      s.written = 0;
    }
    math_sync();
    float gate = __uint_as_float(s.gates[local]);
    uint32_t kept = 0;
#pragma unroll
    for (uint32_t i = 0; i < Items; ++i)
      kept += isfinite(values[i]) && values[i] >= gate;
    kept = __reduce_add_sync(kFullMask, kept);
    if (lane == 0) atomicAdd(&s.rank, kept);
    math_sync();
    kept = s.rank;
    if (kept >= K && kept <= kRetainedPerRow - kSegmentKeys) {
      emit(gate, 0, 0, 0);
      math_sync();
      if (tid == 0) s.kept[local] = kept;
      math_sync();
      return;
    }
  }
  if (tid == 0) {
    s.prefix = 0;
    s.rank = K;
    s.written = 0;
  }
  math_sync();
  for (int32_t shift = 24; shift >= 0; shift -= 8) {
    for (uint32_t j = tid; j < padded_bins(256); j += T)
      s.work[j] = 0;
    math_sync();
    uint32_t prefix = s.prefix;
#pragma unroll
    for (uint32_t i = 0; i < Items; ++i) {
      uint32_t code = ordered_code(values[i]);
      bool valid = isfinite(values[i]) && (shift == 24 || (code >> (shift + 8)) == prefix);
      uint32_t mask = __ballot_sync(kFullMask, valid);
      if (valid) {
        uint32_t bin = (code >> shift) & 255u, peers = __match_any_sync(mask, bin);
        if (lane == uint32_t(__ffs(peers) - 1)) atomicAdd(s.work + padded_bin(bin), __popc(peers));
      }
    }
    math_sync();
    if (tid < 32) {
      uint32_t sum = 0;
      for (uint32_t j = 0; j < 8; ++j) {
        uint32_t bin = 255 - lane * 8 - j;
        sum += s.work[padded_bin(bin)];
      }
      uint32_t inclusive = sum;
      inclusive = warp_inclusive_scan(inclusive, lane);
      uint32_t before = inclusive - sum, rank = s.rank,
               crossing = __ballot_sync(kFullMask, before < rank && inclusive >= rank),
               chosen_lane = __ffs(crossing) - 1, chosen = 0, above = before;
      if (lane == chosen_lane) {
        for (uint32_t j = 0; j < 8; ++j) {
          uint32_t bin = 255 - lane * 8 - j;
          if (above + s.work[padded_bin(bin)] >= rank) {
            chosen = bin;
            break;
          }
          above += s.work[padded_bin(bin)];
        }
      }
      chosen = __shfl_sync(kFullMask, chosen, chosen_lane);
      above = __shfl_sync(kFullMask, above, chosen_lane);
      __syncwarp();  // every lane read s.rank above
      if (lane == 0) {
        s.prefix = (prefix << 8) | chosen;
        s.rank = rank - above;
      }
    }
    math_sync();
  }
  uint32_t threshold = s.prefix;
  uint32_t greater = emit(0, threshold, 1, 0);
  emit(0, threshold, 2, greater);
  if (tid == 0) {
    s.kept[local] = K;
    float gate = fmaxf(__uint_as_float(s.gates[local]), ordered_float(threshold));
    atomicExch(s.gates + local, __float_as_uint(gate));
  }
  math_sync();
}

// One warp merges row `local`'s segment (the first n ring entries) into its
// retained set: the segment's histogram raises the row's gate to the k-th
// largest score seen so far, and the values above the gate are appended.
__device__ void merge_segment(
    RescanShared& s,
    RescanBuffers b,
    uint32_t K,
    uint32_t local,
    uint32_t row,
    uint32_t epoch,
    uint32_t first,
    uint32_t n,
    uint32_t tid,
    bool final) {
  constexpr uint32_t T = 32, Items = kSegmentKeys / T;
  uint32_t lane = tid, bank = epoch & 1;
  float values[Items], gate = __uint_as_float(s.gates[local]);
  uint32_t count = 0;
#pragma unroll
  for (uint32_t i = 0; i < Items; ++i) {
    uint32_t j = tid + i * T;
    float v = j < n ? s.ring[bank][local][j] : -CUDART_INF_F;
    if (v <= gate) v = -CUDART_INF_F;
    values[i] = v;
    bool valid = isfinite(v);
    count += valid;
    uint32_t mask = __ballot_sync(kFullMask, valid);
    if (valid) {
      uint32_t bin = half_bin<kSegmentBinShift>(v), peers = __match_any_sync(mask, bin);
      if (lane == uint32_t(__ffs(peers) - 1)) atomicAdd(&s.hist[local][padded_bin(bin)], __popc(peers));
    }
  }
  count = __reduce_add_sync(kFullMask, count);
  __syncwarp();
  if (count) {
    uint32_t sum = 0;
#pragma unroll
    for (uint32_t j = 0; j < 32; ++j)
      sum += s.hist[local][(31 - lane) * 33 + j];
    uint32_t inclusive = sum;
    inclusive = warp_inclusive_scan(inclusive, lane);
    uint32_t before = inclusive - sum, mask = __ballot_sync(kFullMask, before < K && inclusive >= K);
    if (mask) {
      uint32_t chosen = __ffs(mask) - 1, group = 31 - chosen, rank = K - __shfl_sync(kFullMask, before, chosen),
               bin = group * 32 + 31 - lane;
      sum = s.hist[local][padded_bin(bin)];
      inclusive = sum;
      inclusive = warp_inclusive_scan(inclusive, lane);
      before = inclusive - sum;
      chosen = __ffs(__ballot_sync(kFullMask, before < rank && inclusive >= rank)) - 1;
      bin = __shfl_sync(kFullMask, bin, chosen);
      gate = fmaxf(gate, half_bin_floor<kSegmentBinShift>(bin));
      if (lane == 0) atomicExch(s.gates + local, __float_as_uint(gate));
    }
    uint32_t written = s.kept[local];
    uint64_t base = (uint64_t)row * kRetainedPerRow;
#pragma unroll
    for (uint32_t i = 0; i < Items; ++i) {
      float v = values[i];
      bool pass = isfinite(v) && v >= gate;
      uint32_t mask = __ballot_sync(kFullMask, pass), pos = written + __popc(mask & ((1u << lane) - 1));
      if (pass) {
        b.retained[base + pos] = v;
        b.indices[base + pos] = first + tid + i * T;
      }
      written += __popc(mask);
    }
    __syncwarp();
    if (lane == 0) s.kept[local] = written;
    __syncwarp();
  }
}

// Streams each row's scores (above its running gate) into the segment ring and
// merges every full segment into the row's retained top set; the final set is
// written to the output.
struct RescanPolicy {
  static constexpr float kWeightScale = 0.5f, kScoreScale = 1.0f;
  static constexpr uint32_t kSplitsPerSegment = kSegmentKeys / kTileKeys;
  RescanShared& stream;
  RescanBuffers buffers;
  uint32_t topk, query_base, row_base, epoch = 0;
  uint32_t seq_k_start[4], seq_k_end[4];
  float gate[4];
  CUTLASS_DEVICE void begin_block(const Scheduler& s, uint32_t q_block, uint32_t base) {
    query_base = base;
    row_base = s.get_logits_row(q_block, 0);
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const uint32_t row = s.get_logits_row(q_block, base + i);
      gate[i] = -CUDART_INF_F;
      seq_k_start[i] = min(s.cu_seq_len_k_start[row], s.num_kv_tokens);
      seq_k_end[i] = min(s.cu_seq_len_k_end[row], s.num_kv_tokens);
    }
  }
  CUTLASS_DEVICE void begin_split(uint32_t split, uint32_t) {
    if (split % kSplitsPerSegment == 0) {
      __syncwarp();
      for (uint32_t i = 0; i < 4; ++i)
        gate[i] = __uint_as_float(atomicAdd(stream.gates + query_base + i, 0u));
    }
  }
  CUTLASS_DEVICE void score(uint32_t i, uint32_t kv_offset, uint32_t kv_lane, uint32_t split, float bq) {
    const bool valid = kv_offset >= seq_k_start[i] && kv_offset < seq_k_end[i] && isfinite(bq) && bq > gate[i];
    stream.ring[epoch & 1][query_base + i][(split % kSplitsPerSegment) * kTileKeys + kv_lane] =
        valid ? bq : -CUDART_INF_F;
  }
  CUTLASS_DEVICE void end_split(uint32_t split, uint32_t num_splits, uint32_t kv_base) {
    if ((split + 1) % kSplitsPerSegment != 0 && split + 1 != num_splits) return;
    __threadfence_block();
    math_sync();
    const uint32_t tid = threadIdx.x;
    for (uint32_t local = 0; local < kBlockRows; ++local)
      if (stream.kept[local] > kRetainedPerRow - kSegmentKeys)
        compact_retained(stream, buffers, topk, local, row_base + local, tid);
    math_sync();
    const uint32_t local = tid / 32;
    merge_segment(
        stream,
        buffers,
        topk,
        local,
        row_base + local,
        epoch,
        kv_base + (split / kSplitsPerSegment) * kSegmentKeys,
        (split % kSplitsPerSegment + 1) * kTileKeys,
        tid % 32,
        false);
    math_sync();
    if (split + 1 == num_splits)
      for (uint32_t l = 0; l < kBlockRows; ++l)
        if (stream.kept[l] > topk) compact_retained(stream, buffers, topk, l, row_base + l, tid, true);
    math_sync();
    ++epoch;
  }
  CUTLASS_DEVICE void end_block(const Scheduler& s, uint32_t q_block) {
    const uint32_t local = threadIdx.x / 32, lane = threadIdx.x % 32;
    const uint32_t row = s.get_logits_row(q_block, local), count = stream.kept[local];
    if (row < s.num_q_tokens)
      for (uint32_t j = lane; j < topk; j += 32)
        buffers.output[(uint64_t)row * topk + j] =
            j < count ? buffers.indices[(uint64_t)row * kRetainedPerRow + j] : -1;
  }
};

// Rescans one block exactly. Out of line: the rare rescan must not add
// register pressure to the caller's hot loops.
__device__ __noinline__ void rescan_block(
    const Scheduler& block,
    RescanBuffers buffers,
    uint32_t topk,
    const cute::TmaDescriptor& tq,
    const cute::TmaDescriptor& tk,
    const cute::TmaDescriptor& ts,
    const cute::TmaDescriptor& tw) {
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  auto& stream = *reinterpret_cast<RescanShared*>(smem_buffer + sizeof(dense::SharedStorage));
  stream.init(threadIdx.x);
  RescanPolicy policy{stream, buffers, topk};
  dense::score_block(block, policy, tq, tk, ts, tw);
}

}  // namespace sglang::fused_mqa_topk::rescan
