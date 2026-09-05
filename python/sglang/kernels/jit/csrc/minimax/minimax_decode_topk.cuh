#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>
#ifdef USE_ROCM
static constexpr unsigned long long kWarpSyncMask = 0xFFFFFFFFFFFFFFFFull;
#else
#include <math_constants.h>
static constexpr unsigned int kWarpSyncMask = 0xFFFFFFFFu;
#endif

namespace sglang {

// Block top-k selection over a per-(head, batch) row of block scores, run by one
// CTA of TopKTrait::kCTASize threads. Picks the `topk` highest-scoring block ids
// (k_eff = min(topk, num_blocks)). Three size regimes, chosen by num_blocks:
//   * <= kSmallThreshold : O(n^2) rank-by-compare (no radix).
//   * <= kCTASize        : 4-pass 8-bit radix, one element per thread in a reg.
//   * <= kMaxNumBlocks   : 4-pass 8-bit radix, kIters elements per thread cached
//                          in registers (row read from global exactly once);
//                          liveness is a uint32_t bitmask, selection is an
//                          in-loop scatter -- nothing is cached in shared memory.
// The trivial case num_blocks <= topk (every block selected) is handled by the
// kernels below, outside the Trait.
//
// On ROCm every regime is replaced by rocm_hist_select over a packed
// (score, ~id) u64 total order: one coarse histogram round (256 bins for
// n <= 1024, else 2048), 11-bit refine rounds while > kRocmCap candidates
// remain, then a wave-0 bitonic-sort finish.
template <uint32_t kNumThreads = 512>
struct TopKTraitImpl {
  // Also sizes the kernels' smem staging for the ascending-order emit; the
  // block-id path's test contract goes up to topk == 64.
  static constexpr uint32_t kMaxTopK = 64;
  static constexpr uint32_t kCTASize = kNumThreads;
  static constexpr uint32_t kNumWarps = kCTASize / device::kWarpThreads;
  // Must match _MAX_NUM_BLOCKS in ops/attention/minimax_decode_topk.py.
#ifdef USE_ROCM
  static constexpr uint32_t kMaxNumBlocks = 16384;  // block topk
#else
  static constexpr uint32_t kMaxNumBlocks = 4096;  // block topk
#endif
  static constexpr uint32_t kSmallThreshold = 8 * kNumWarps;
  static constexpr uint32_t kRadixBits = 8;
  static constexpr uint32_t kRadixSize = 1 << kRadixBits;
  static constexpr float kNegInf = -std::numeric_limits<float>::infinity();
#ifdef USE_ROCM
  static constexpr uint32_t kWaveSize = 64;  // physical wave64, not the logical-32 kWarpThreads
  static constexpr uint32_t kNumWaves = kCTASize / kWaveSize;
  static constexpr uint32_t kRocmBins = 2048;  // refine rounds (11-bit); round 1 may use 256
  static constexpr uint32_t kRocmCap = 128;    // max candidates for the bitonic-sort finish
#endif

  struct Smem {
    uint32_t warp_sum[kNumWarps];
    alignas(128) uint32_t counter;
    alignas(128) uint32_t counter_final;
    alignas(128) uint32_t threshold_bin;
    uint32_t equal_count;
    uint32_t above_count;
    uint32_t histogram[2][kRadixSize];    // 8 bit radix
    float small_scores[kSmallThreshold];  // small (O(n^2)) path only
#ifdef USE_ROCM
    // Histogram-select path only.
    uint32_t rocm_hist[kRocmBins];
    uint32_t rocm_wave_sums[kNumWaves];
    uint32_t rocm_thr_bin;
    uint32_t rocm_above;  // count strictly above the threshold bin this round
    uint32_t rocm_equal;  // count inside the threshold bin this round
    uint32_t rocm_emit;   // output ticket counter for early (above-bin) emits
    uint32_t rocm_stage_count;
    uint64_t rocm_stage[kRocmCap];
#endif
  };

#ifdef USE_ROCM
  // Packed (score desc, block id asc) total order. High half: monotone
  // fp32->u32 key (NaN -> -inf); key(-inf) > 0, so packed == 0 is the "no
  // element" sentinel. The ~id low half makes packed values unique and lower
  // block ids win score ties.
  SGL_DEVICE static uint64_t pack_score_id(float x, uint32_t idx) {
    if (x != x) x = kNegInf;
    const uint32_t b = __float_as_uint(x);
    const uint32_t key = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return (static_cast<uint64_t>(key) << 32) | (0xFFFFFFFFu - idx);
  }

  // Find the bin holding the topk_remain-th largest element counted in
  // smem->rocm_hist[0..2^kBits) (thread tx owns kPer consecutive bins);
  // publishes the bin, the count strictly above it, and the count inside it.
  template <uint32_t kBits>
  SGL_DEVICE static void rocm_find_threshold(const uint32_t topk_remain, Smem* smem) {
    constexpr uint32_t kB = 1u << kBits;
    constexpr uint32_t kPer = kB >= kCTASize ? kB / kCTASize : 1;
    const uint32_t tx = threadIdx.x;
    const uint32_t wave = tx / kWaveSize;
    const uint32_t lane = tx % kWaveSize;
    const bool own = (kB >= kCTASize) || (tx < kB);

    uint32_t loc[kPer];
    uint32_t lsum = 0;
#pragma unroll
    for (uint32_t j = 0; j < kPer; ++j) {
      loc[j] = own ? smem->rocm_hist[tx * kPer + j] : 0;
      lsum += loc[j];
    }
    uint32_t inc = lsum;
#pragma unroll
    for (uint32_t offset = 1; offset < kWaveSize; offset <<= 1) {
      const uint32_t n = __shfl_up(inc, offset, kWaveSize);
      if (lane >= offset) inc += n;
    }
    if (lane == kWaveSize - 1) smem->rocm_wave_sums[wave] = inc;
    __syncthreads();
    uint32_t wave_prefix = 0;
    uint32_t total = 0;
#pragma unroll
    for (uint32_t w = 0; w < kNumWaves; ++w) {
      const uint32_t v = smem->rocm_wave_sums[w];
      total += v;
      wave_prefix += w < wave ? v : 0;
    }
    uint32_t prefix = wave_prefix + inc - lsum;  // count in bins before tx's first bin
#pragma unroll
    for (uint32_t j = 0; j < kPer; ++j) {
      const uint32_t above = total - (prefix + loc[j]);  // strictly above bin tx*kPer+j
      if (own && above < topk_remain && above + loc[j] >= topk_remain) {
        smem->rocm_thr_bin = tx * kPer + j;
        smem->rocm_above = above;
        smem->rocm_equal = loc[j];
      }
      prefix += loc[j];
    }
    __syncthreads();
  }

  // One histogram round over the live elements' key bits [shift, shift+kBits):
  // bins above the threshold bin are emitted immediately (ticket order; callers
  // re-sort), bins below die, the threshold bin survives to the next round.
  // Ends with a barrier: rocm_emit is re-armed next round.
  template <uint32_t kBits, uint32_t kSlots>
  SGL_DEVICE static void rocm_hist_round(
      uint64_t (&packed)[kSlots],
      bool (&live)[kSlots],
      const int shift,
      const uint32_t topk,
      uint32_t& topk_remain,
      uint32_t& cand_count,
      int32_t* __restrict__ topk_out,
      Smem* smem) {
    constexpr uint32_t kB = 1u << kBits;
    const uint32_t tx = threadIdx.x;
    for (uint32_t i = tx; i < kB; i += kCTASize)
      smem->rocm_hist[i] = 0;
    if (tx == 0) smem->rocm_emit = topk - topk_remain;
    __syncthreads();
#pragma unroll
    for (uint32_t s = 0; s < kSlots; ++s)
      if (live[s]) atomicAdd(&smem->rocm_hist[(packed[s] >> shift) & (kB - 1)], 1);
    __syncthreads();

    rocm_find_threshold<kBits>(topk_remain, smem);
    const uint32_t thr = smem->rocm_thr_bin;
    const uint32_t above = smem->rocm_above;
    const uint32_t equal = smem->rocm_equal;

#pragma unroll
    for (uint32_t s = 0; s < kSlots; ++s) {
      if (live[s]) {
        const uint32_t bin = (packed[s] >> shift) & (kB - 1);
        if (bin > thr) {
          const uint32_t p = atomicAdd(&smem->rocm_emit, 1);
          topk_out[p] = static_cast<int32_t>(0xFFFFFFFFu - static_cast<uint32_t>(packed[s]));
          live[s] = false;
        } else if (bin < thr) {
          live[s] = false;
        }
      }
    }
    topk_remain -= above;
    cand_count = equal;
    __syncthreads();
  }

  // Stage the <= kRocmCap survivors to LDS; wave 0 bitonic-sorts all 128
  // slots (element e = lane + 64*r held as v0/v1, descending, sentinel-0
  // padded) and emits the first topk_remain elements in parallel.
  template <uint32_t kSlots>
  SGL_DEVICE static void rocm_sort_finish(
      uint64_t (&packed)[kSlots],
      bool (&live)[kSlots],
      const uint32_t topk,
      const uint32_t topk_remain,
      int32_t* __restrict__ topk_out,
      Smem* smem) {
    const uint32_t tx = threadIdx.x;
    const uint32_t wave = tx / kWaveSize;
    const uint32_t lane = tx % kWaveSize;
    if (tx == 0) smem->rocm_stage_count = 0;
    __syncthreads();
#pragma unroll
    for (uint32_t s = 0; s < kSlots; ++s) {
      if (live[s]) {
        const uint32_t p = atomicAdd(&smem->rocm_stage_count, 1);
        if (p < kRocmCap) smem->rocm_stage[p] = packed[s];
      }
    }
    __syncthreads();
    const uint32_t nc = smem->rocm_stage_count < kRocmCap ? smem->rocm_stage_count : kRocmCap;
    if (wave == 0) {
      static_assert(kRocmCap == 2 * kWaveSize);
      uint64_t v0 = lane < nc ? smem->rocm_stage[lane] : 0;
      uint64_t v1 = lane + kWaveSize < nc ? smem->rocm_stage[lane + kWaveSize] : 0;
#pragma unroll
      for (uint32_t kk = 2; kk <= 2 * kWaveSize; kk <<= 1) {
#pragma unroll
        for (uint32_t j = kWaveSize; j >= 1; j >>= 1) {
          if (j >= kk) continue;
          // element e is in a descending run iff (e & kk) == 0 (final pass kk=128: all descending)
          if (j == kWaveSize) {  // stride-64 exchange pairs v0 (e=lane) with v1 (e=lane+64)
            const bool desc = ((lane & kk) == 0) || kk == 2 * kWaveSize;
            const uint64_t hi = v0 > v1 ? v0 : v1;
            const uint64_t lo = v0 > v1 ? v1 : v0;
            v0 = desc ? hi : lo;
            v1 = desc ? lo : hi;
          } else {
            const uint64_t o0 = __shfl_xor(v0, j, kWaveSize);
            const uint64_t o1 = __shfl_xor(v1, j, kWaveSize);
            const bool lower = (lane & j) == 0;
            const bool desc0 = ((lane & kk) == 0) || kk == 2 * kWaveSize;
            const bool desc1 = (((lane + kWaveSize) & kk) == 0) || kk == 2 * kWaveSize;
            const bool take_max0 = (lower == desc0);
            const bool take_max1 = (lower == desc1);
            v0 = (take_max0 == (o0 > v0)) ? o0 : v0;
            v1 = (take_max1 == (o1 > v1)) ? o1 : v1;
          }
        }
      }
      // descending order: e-th largest at e = lane (r = 0); topk_remain <= 64
      const uint32_t base = topk - topk_remain;
      if (lane < topk_remain) topk_out[base + lane] = static_cast<int32_t>(0xFFFFFFFFu - static_cast<uint32_t>(v0));
    }
  }

  // Round-1 bin count for a dispatch bucket: 256 bins up to 1024 rows, else 2048.
  static constexpr uint32_t round1_bits(uint32_t bucket_capacity) {
    return bucket_capacity <= 1024 ? 8 : 11;
  }

  // Load the row once into registers (wave-contiguous), run histogram rounds
  // until <= kRocmCap candidates remain (the shift ladder covers the whole
  // 64-bit key and keys are unique, so the loop terminates), then sort the
  // survivors on wave 0. topk_out[0, topk) holds the exact top-k set in
  // unspecified order. Callers guarantee num_blocks > topk.
  template <uint32_t kSlots, uint32_t kBits1>
  SGL_DEVICE static void rocm_hist_select(
      const float* __restrict__ scores,
      const uint32_t num_blocks,
      int32_t* __restrict__ topk_out,
      const uint32_t topk,
      Smem* smem) {
    const uint32_t tx = threadIdx.x;
    const uint32_t wave = tx / kWaveSize;
    const uint32_t lane = tx % kWaveSize;

    uint64_t packed[kSlots];
    bool live[kSlots];
#pragma unroll
    for (uint32_t s = 0; s < kSlots; ++s) {
      const uint32_t idx = (s * kNumWaves + wave) * kWaveSize + lane;
      const bool in = idx < num_blocks;
      packed[s] = in ? pack_score_id(scores[idx], idx) : 0;
      live[s] = in;
    }

    uint32_t topk_remain = topk;
    uint32_t cand_count = num_blocks;
    if (cand_count > kRocmCap) {
      rocm_hist_round<kBits1>(
          packed, live, 64 - static_cast<int>(kBits1), topk, topk_remain, cand_count, topk_out, smem);
      int shift = 64 - static_cast<int>(kBits1);
      // The shift ladder covers all 64 key bits in kMaxRefineRounds, and keys
      // are unique, so cand_count <= 1 by the last round; the explicit bound
      // makes a broken-invariant failure degrade to the sort-finish clamp
      // instead of a device hang.
      constexpr int kMaxRefineRounds = (64 + 10) / 11 + 1;
      for (int round = 0; round < kMaxRefineRounds && cand_count > kRocmCap; ++round) {
        shift -= 11;
        if (shift < 0) shift = 0;
        rocm_hist_round<11>(packed, live, shift, topk, topk_remain, cand_count, topk_out, smem);
      }
    }
    rocm_sort_finish(packed, live, topk, topk_remain, topk_out, smem);
  }
#endif  // USE_ROCM

  SGL_DEVICE static void forward(
      const float* __restrict__ scores,
      const uint32_t num_blocks,
      int32_t* __restrict__ topk_out,
      const uint32_t topk,
      Smem* smem) {
#ifdef USE_ROCM
    // Smallest register footprint that covers the row.
    if (num_blocks <= 1 * kCTASize) {
      rocm_hist_select<1, round1_bits(1 * kCTASize)>(scores, num_blocks, topk_out, topk, smem);
    } else if (num_blocks <= 2 * kCTASize) {
      rocm_hist_select<2, round1_bits(2 * kCTASize)>(scores, num_blocks, topk_out, topk, smem);
    } else if (num_blocks <= 4 * kCTASize) {
      rocm_hist_select<4, 11>(scores, num_blocks, topk_out, topk, smem);
    } else if (num_blocks <= 8 * kCTASize) {
      rocm_hist_select<8, 11>(scores, num_blocks, topk_out, topk, smem);
    } else if (num_blocks <= 16 * kCTASize) {
      rocm_hist_select<16, 11>(scores, num_blocks, topk_out, topk, smem);
    } else {
      // 32 slots are only needed (and only instantiated) when 16 * kCTASize
      // cannot reach kMaxNumBlocks.
      static_assert(16 * kCTASize >= kMaxNumBlocks || 32 * kCTASize >= kMaxNumBlocks);
      if constexpr (16 * kCTASize < kMaxNumBlocks) {
        rocm_hist_select<32, 11>(scores, num_blocks, topk_out, topk, smem);
      }
    }
    return;
#else
    using namespace device;
    const auto tx = threadIdx.x;
    __builtin_assume(tx < kCTASize);
    const auto warp_id = tx / kWarpThreads;
    const auto lane_id = tx % kWarpThreads;

    constexpr auto is_greater = [](float x, float y, int32_t delta) {
      return (x > y) || ((x == y) && delta < 0);  // lower block id wins
    };
    constexpr auto warp_inclusive_sum = [](uint32_t lane_id, uint32_t val) {
#pragma unroll
      for (uint32_t offset = 1; offset < device::kWarpThreads; offset *= 2) {
        // Width-32 up-shuffle. On wave64 HIP the un-suffixed __shfl_up takes the
        // logical-warp width directly; CUDA needs the active mask.
#ifdef USE_ROCM
        uint32_t n = __shfl_up(val, offset, device::kWarpThreads);
#else
        uint32_t n = __shfl_up_sync(kWarpSyncMask, val, offset, device::kWarpThreads);
#endif
        if (lane_id >= offset) val += n;
      }
      return val;
    };
    constexpr auto clip_nan = [](float x) { return x != x ? kNegInf : x; };
    constexpr auto score_to_key = [](float x) {
      uint32_t b = __float_as_uint(x);
      return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    };
    // Find the radix bin holding the topk_remain-th largest of `total_active`
    // elements currently counted in `histogram`. Writes threshold_bin (the bin),
    // above_count (elements strictly above it), equal_count (elements in it).
    const auto find_threshold = [&](uint32_t* histogram, uint32_t total_active, uint32_t topk_remain) {
      using namespace device;
      uint32_t hist_val = 0;
      uint32_t warp_inc = 0;
      if (tx < kRadixSize) {
        hist_val = histogram[tx];
        warp_inc = warp_inclusive_sum(lane_id, hist_val);
        if (lane_id == kWarpThreads - 1) smem->warp_sum[warp_id] = warp_inc;
      }
      __syncthreads();
      if (tx < kRadixSize) {
        const auto inter = warp::reduce_sum(lane_id < warp_id ? smem->warp_sum[lane_id] : 0);
        const auto prefix = inter + warp_inc;      // count in bins [0, tx]
        const auto above = total_active - prefix;  // count in bins ABOVE tx
        if (above < topk_remain && above + hist_val >= topk_remain) {
          smem->threshold_bin = tx;
          smem->above_count = above;
          smem->equal_count = hist_val;
        }
      }
      __syncthreads();
    };

    if (num_blocks <= kSmallThreshold) {
      // O(n^2) compare: each block's rank = #blocks that outrank it; the ones
      // with rank < topk are selected (rank is its position in topk_out).
      static_assert(kSmallThreshold <= kCTASize);
      if (tx < num_blocks) smem->small_scores[tx] = clip_nan(scores[tx]);
      __syncthreads();
      constexpr uint32_t kNumCandidates = kSmallThreshold / kNumWarps;
      constexpr uint32_t kNumTargets = kSmallThreshold / kWarpThreads;
      float candidates[kNumCandidates];
      float target[kNumTargets];
#pragma unroll
      for (uint32_t i = 0; i < kNumTargets; ++i) {
        const auto idx = lane_id + i * kWarpThreads;
        target[i] = (idx < num_blocks) ? smem->small_scores[idx] : kNegInf;
      }
#pragma unroll
      for (uint32_t i = 0; i < kNumCandidates; ++i) {
        const auto idx = warp_id + i * kNumWarps;
        candidates[i] = (idx < num_blocks) ? smem->small_scores[idx] : kNegInf;
      }

#pragma unroll
      for (uint32_t i = 0; i < kNumCandidates; ++i) {
        const int32_t idx = warp_id + i * kNumWarps;
        if (idx >= static_cast<int32_t>(num_blocks)) break;
        uint32_t rank = 0;
#pragma unroll
        for (uint32_t j = 0; j < kNumTargets; ++j) {
          const int32_t delta = lane_id + j * kWarpThreads - idx;
          // partial rank = how many of this lane's targets outrank the candidate
          rank += is_greater(target[j], candidates[i], delta);
        }
        // full rank = sum of the per-lane partial ranks across the warp
        rank = warp::reduce_sum(rank);
        if (rank < topk) topk_out[rank] = idx;
      }
    } else if (num_blocks <= kCTASize) {
      // 4-pass 8-bit radix select, one element per thread held in a register.
      bool active = tx < num_blocks;
      const auto value = active ? clip_nan(scores[tx]) : kNegInf;
      const auto key = score_to_key(value);
      uint32_t topk_remain = topk;
      uint32_t write_pos = topk;  // sentinel: not selected
      if (tx < kRadixSize) smem->histogram[0][tx] = 0;
      if (tx == kRadixSize) smem->counter = smem->counter_final = 0;
      __syncthreads();
      uint32_t total_active = num_blocks;

#pragma unroll
      for (int round = 0; round < 4; round++) {
        const uint32_t shift = 24 - round * 8;
        const uint32_t bin = (key >> shift) & 0xFFu;
        const auto hist_idx = round % 2;
        const auto histogram = smem->histogram[hist_idx];

        if (active) atomicAdd(&histogram[bin], 1);
        if (round < 3 && tx < kRadixSize) smem->histogram[hist_idx ^ 1][tx] = 0;
        __syncthreads();

        find_threshold(histogram, total_active, topk_remain);

        const auto threshold_bin = smem->threshold_bin;
        const auto above_count = smem->above_count;
        const auto equal_count = smem->equal_count;

        if (round < 3) total_active = equal_count;
        topk_remain -= above_count;

        // scatter: above -> selected now; equal at the last pass -> keep the rest
        if (active) {
          if (bin > threshold_bin) {
            write_pos = atomicAdd(&smem->counter, 1);
            active = false;
          } else if (bin < threshold_bin) {
            active = false;
          } else if (round == 3) {
            write_pos = topk - topk_remain + atomicAdd(&smem->counter_final, 1);
          }
          // bin == threshold && round < 3: stay active for the next pass
        }

        if (round == 3 || topk_remain == 0) break;
      }

      if (write_pos < topk) topk_out[write_pos] = tx;
    } else {
      // num_blocks in (kCTASize, kMaxNumBlocks]: each thread caches its (up to
      // kIters) slice of the row in registers -- read from global exactly ONCE --
      // then runs the same 4-pass radix select as the single-element path looped
      // over those slots. Liveness is a uint32_t bitmask (bit i = slot i still in
      // the running set), so there is no per-element flag array; selection is an
      // in-loop scatter, so there is no per-element position array. Nothing is
      // cached in shared memory beyond the histogram.
      constexpr uint32_t kIters = kMaxNumBlocks / kCTASize;
      static_assert(kIters <= 32, "active liveness is packed into a uint32_t");
      uint32_t key[kIters];
      uint32_t active = 0;
#pragma unroll
      for (uint32_t i = 0; i < kIters; ++i) {
        const uint32_t idx = i * kCTASize + tx;
        if (idx < num_blocks) {
          key[i] = score_to_key(clip_nan(scores[idx]));
          active |= 1u << i;
        }
      }
      if (tx < kRadixSize) smem->histogram[0][tx] = 0;
      if (tx == kRadixSize) smem->counter = smem->counter_final = 0;
      __syncthreads();

      uint32_t topk_remain = topk;
      uint32_t total_active = num_blocks;

#pragma unroll
      for (int round = 0; round < 4; ++round) {
        const uint32_t shift = 24 - round * 8;
        const auto hb = round & 1;

#pragma unroll
        for (uint32_t i = 0; i < kIters; ++i)
          if (active & (1u << i)) atomicAdd(&smem->histogram[hb][(key[i] >> shift) & 0xFFu], 1);
        if (round < 3 && tx < kRadixSize) smem->histogram[hb ^ 1][tx] = 0;
        __syncthreads();

        find_threshold(smem->histogram[hb], total_active, topk_remain);
        const auto threshold_bin = smem->threshold_bin;
        const auto above_count = smem->above_count;
        const auto equal_count = smem->equal_count;

        if (round < 3) total_active = equal_count;
        topk_remain -= above_count;

#pragma unroll
        for (uint32_t i = 0; i < kIters; ++i) {
          if (active & (1u << i)) {
            const uint32_t bin = (key[i] >> shift) & 0xFFu;
            if (bin > threshold_bin) {
              topk_out[atomicAdd(&smem->counter, 1)] = i * kCTASize + tx;
              active &= ~(1u << i);
            } else if (bin < threshold_bin) {
              active &= ~(1u << i);
            } else if (round == 3) {
              const auto pos = topk - topk_remain + atomicAdd(&smem->counter_final, 1);
              if (pos < topk) topk_out[pos] = i * kCTASize + tx;
            }
            // bin == threshold && round < 3: slot stays live for the next pass
          }
        }

        if (round == 3 || topk_remain == 0) break;
      }
    }
#endif  // !USE_ROCM
  }
};

// CUDA uses only the 512-thread instantiation; ROCm launchers pick 1024
// threads when max_seqblock > 1024.
using TopKTrait = TopKTraitImpl<512>;
#ifdef USE_ROCM
using TopKTraitWide = TopKTraitImpl<1024>;
#endif

// -------------------------------------------------------------------------
// Kernels: one CTA (kCTASize threads) per (head, batch) row. The trivial case
// num_blocks <= topk (every block selected) is special-judged here, outside the
// Trait; otherwise the Trait selects the top-k block ids.
// -------------------------------------------------------------------------

// Block-id output: topk_idx[h, b, 0:k_eff) = selected block ids (sorted
// ascending), [k_eff:topk) = -1. Ascending order is a hard requirement of the
// MSA fmha_sm100 consumer (kv_block_indexes must be strictly ascending; its
// sorted-order early-exit otherwise mis-masks the partial last block).
template <typename SeqLenT, bool kUsePDL, typename TopKTrait>
__global__ void minimax_decode_topk_block_kernel(
    const float* __restrict__ score,
    const SeqLenT* __restrict__ seq_lens,
    int32_t* __restrict__ topk_idx,
    int batch,
    int num_heads,
    int max_seqblock,
    int block_size,
    int topk) {
  const int b = blockIdx.x;  // grid.x = batch
  const int h = blockIdx.y;  // grid.y = num_heads
  const int tx = threadIdx.x;

  // seq_lens is from an earlier kernel; prefetch it (and the cheap setup) before
  // waiting on the score producer so the prologue overlaps its tail (PDL).
  const int64_t seq_len = static_cast<int64_t>(seq_lens[b]);
  const int num_blocks_raw = static_cast<int>((seq_len + block_size - 1) / block_size);
  // Never scan past the materialized score columns.
  const int num_blocks = num_blocks_raw < max_seqblock ? num_blocks_raw : max_seqblock;
  int32_t* __restrict__ out = topk_idx + (static_cast<int64_t>(h) * batch + b) * topk;
  device::PDLWaitPrimary<kUsePDL>();

  if (num_blocks <= topk) {  // trivial: identity, -1 padded
    for (int i = tx; i < topk; i += TopKTrait::kCTASize)
      out[i] = (i < num_blocks) ? i : -1;
    return;
  }

  const float* __restrict__ row = score + (static_cast<int64_t>(h) * batch + b) * max_seqblock;
  __shared__ typename TopKTrait::Smem smem;
  __shared__ int32_t s_topk[TopKTrait::kMaxTopK];
  TopKTrait::forward(row, static_cast<uint32_t>(num_blocks), s_topk, static_cast<uint32_t>(topk), &smem);
  __syncthreads();  // s_topk fully written before the sort reads it

  // Emit ascending: num_blocks > topk here, so all topk slots hold distinct
  // ids and rank(v) = |{x : x < v}| is a permutation. deepseek_v4
  // topk_impl.cuh warp sort (32x32 / 64x64 branches; topk <= kMaxTopK = 64):
  // lanes hold the elements in registers (INT32_MAX sentinel past topk), warp
  // w ranks targets {w, w + kNumWarps, ...} via ballot+popc, lane 0 emits.
  static_assert(TopKTrait::kMaxTopK <= 2 * device::kWarpThreads);
#ifdef USE_ROCM
  // One 64-lane ballot covers topk <= kMaxTopK = 64; physical waves (not
  // logical-32 warps) stride the targets.
  static_assert(TopKTrait::kMaxTopK <= 64);
  {
    const uint32_t wave = tx / 64u;
    const uint32_t lane64 = tx % 64u;
    const int32_t tie = (lane64 < static_cast<uint32_t>(topk)) ? s_topk[lane64] : INT32_MAX;
    for (uint32_t t = wave; t < static_cast<uint32_t>(topk); t += TopKTrait::kCTASize / 64u) {
      const int32_t target = s_topk[t];
      const auto rank = __popcll(__ballot(tie < target));
      if (lane64 == 0) out[rank] = target;
    }
  }
#else
  const auto warp_id = tx / device::kWarpThreads;
  const auto lane_id = tx % device::kWarpThreads;
  const auto count_lt = [](int32_t x, int32_t v) { return __popc(__ballot_sync(kWarpSyncMask, x < v)); };
  if (topk <= static_cast<int>(device::kWarpThreads)) {  // 32 x 32
    const int32_t tie = (lane_id < static_cast<uint32_t>(topk)) ? s_topk[lane_id] : INT32_MAX;
    for (uint32_t t = warp_id; t < static_cast<uint32_t>(topk); t += TopKTrait::kNumWarps) {
      const int32_t target = s_topk[t];
      const auto rank = count_lt(tie, target);
      if (lane_id == 0) out[rank] = target;
    }
  } else {  // 64 x 64: each lane takes 2 elements
    const int32_t tie_0 = s_topk[lane_id];
    const int32_t tie_1 = (lane_id + device::kWarpThreads < static_cast<uint32_t>(topk))
                              ? s_topk[lane_id + device::kWarpThreads]
                              : INT32_MAX;
    for (uint32_t t = warp_id; t < static_cast<uint32_t>(topk); t += TopKTrait::kNumWarps) {
      const int32_t target = s_topk[t];
      const auto rank = count_lt(tie_0, target) + count_lt(tie_1, target);
      if (lane_id == 0) out[rank] = target;
    }
  }
#endif  // USE_ROCM
}

// Page-table output: for each (batch b, kv-head h) pseudo-request emit the
// trtllm/fa3 page table -- selected blocks sorted ascending (so the final partial
// block's pages land last), each expanded to its ppb = block_size/page_size pages
// via req_to_token -- plus the effective KV length seq_lens_out.
//
// DP attention (num_kv_heads > 1): each kv head selects its OWN blocks, so the
// per-request page table can't be shared across heads. We flatten (b, h) into
// num_heads*batch pseudo-requests laid out batch-major (row = b*num_heads + h,
// matching q.view(bs, nkv, gqa, d).reshape(bs*nkv, gqa, d)). seq_lens / slot_ids /
// req_to_token are per-batch (head-independent: a token's cache slot is the same
// for every head). The page index is head-encoded (head-minor) as
// base_page*num_heads + h, which is exactly the page index into an HND cache
// [num_pages, nkv, page_size, D] reshaped to [num_pages*nkv, 1, page_size, D] (a
// free view when the cache is contiguous HND). num_heads == 1 (h == 0) reproduces
// the single-kv-head TP>=4 behavior (page index == base_page).
template <typename SeqLenT, bool kUsePDL, typename TopKTrait>
__global__ void minimax_decode_topk_page_table_kernel(
    const float* __restrict__ score,
    const SeqLenT* __restrict__ seq_lens,
    const int32_t* __restrict__ req_to_token,
    const int64_t* __restrict__ slot_ids,
    int32_t* __restrict__ page_table,
    int32_t* __restrict__ seq_lens_out,
    int batch,
    int num_heads,
    int max_seqblock,
    int block_size,
    int topk,
    int page_size,
    int r2t_stride,
    int max_kv_len,
    int max_sparse_pages) {
  const int b = blockIdx.x;  // grid.x = batch
  const int h = blockIdx.y;  // grid.y = num_heads (kv head)
  const int tx = threadIdx.x;

  // Prefetch seq_lens / slot_ids (from earlier kernels) and the cheap setup
  // before waiting on the score producer, so the prologue overlaps its tail (PDL).
  const int64_t seq_len = static_cast<int64_t>(seq_lens[b]);
  const int num_blocks_raw = static_cast<int>((seq_len + block_size - 1) / block_size);
  const int num_blocks = num_blocks_raw < max_seqblock ? num_blocks_raw : max_seqblock;
  const int ppb = block_size / page_size;
  const int64_t out_row = static_cast<int64_t>(b) * num_heads + h;  // flattened pseudo-request
  int32_t* __restrict__ pt_row = page_table + out_row * max_sparse_pages;
  const int64_t r2t_base = static_cast<int64_t>(slot_ids[b]) * r2t_stride;
  device::PDLWaitPrimary<kUsePDL>();

  if (num_blocks <= topk) {  // trivial: every block selected, all tokens valid
    if (tx == 0) seq_lens_out[out_row] = static_cast<int>(seq_len);
    // block id == ascending slot, so the partial final block's pages land last
    const int total = num_blocks * ppb;
    for (int e = tx; e < total; e += TopKTrait::kCTASize) {
      const int slot = e / ppb;
      const int pp = e % ppb;
      int tok = slot * block_size + pp * page_size;
      if (tok >= max_kv_len) tok = max_kv_len - 1;
      pt_row[e] = req_to_token[r2t_base + tok] / page_size * num_heads + h;
    }
    return;
  }

  const int k_eff = topk;                                                                        // num_blocks > topk
  const float* __restrict__ row = score + (static_cast<int64_t>(h) * batch + b) * max_seqblock;  // head-major score
  __shared__ typename TopKTrait::Smem smem;
  __shared__ int32_t s_topk[TopKTrait::kMaxTopK];
  TopKTrait::forward(row, static_cast<uint32_t>(num_blocks), s_topk, static_cast<uint32_t>(topk), &smem);
  __syncthreads();  // s_topk fully written before the transform reads it

  // Sort the selected block ids ascending (k_eff <= kMaxTopK is tiny) so the
  // partial final block lands last, accumulating the effective KV length in the
  // same pass: each selected block contributes min(block_size, seq_len - c*block)
  // valid tokens (only the final block can be partial).
  __shared__ int32_t s_sorted[TopKTrait::kMaxTopK];
  __shared__ int s_eff_kv;
  if (tx == 0) s_eff_kv = 0;
  __syncthreads();
  for (int slot = tx; slot < k_eff; slot += TopKTrait::kCTASize) {
    const int32_t v = s_topk[slot];
    int rank = 0;
    for (int j = 0; j < k_eff; ++j)
      rank += (s_topk[j] < v);
    s_sorted[rank] = v;
    const int rem = static_cast<int>(seq_len - static_cast<int64_t>(v) * block_size);
    atomicAdd(&s_eff_kv, rem < block_size ? rem : block_size);
  }
  __syncthreads();
  if (tx == 0) seq_lens_out[out_row] = s_eff_kv;

  // Parallel page emit: one thread per output page.
  const int total = k_eff * ppb;
  for (int e = tx; e < total; e += TopKTrait::kCTASize) {
    const int slot = e / ppb;
    const int pp = e % ppb;
    int tok = s_sorted[slot] * block_size + pp * page_size;
    if (tok >= max_kv_len) tok = max_kv_len - 1;
    pt_row[e] = req_to_token[r2t_base + tok] / page_size * num_heads + h;
  }
}

// -------------------------------------------------------------------------
// Launchers
// -------------------------------------------------------------------------
template <typename SeqLenT, bool kUsePDL>
void minimax_decode_topk(
    tvm::ffi::TensorView score,     // [H, B, S] fp32
    tvm::ffi::TensorView seq_lens,  // [B] int32/int64
    tvm::ffi::TensorView topk_idx,  // [H, B, T] int32
    int64_t block_size,
    int64_t topk) {
  using namespace host;

  SymbolicSize H = {"num_heads"};
  SymbolicSize B = {"batch"};
  SymbolicSize S = {"max_seqblock"};
  SymbolicSize T = {"topk"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({H, B, S}).with_dtype<fp32_t>().with_device(device_).verify(score);
  TensorMatcher({B}).with_dtype<SeqLenT>().with_device(device_).verify(seq_lens);
  TensorMatcher({H, B, T}).with_dtype<int32_t>().with_device(device_).verify(topk_idx);

  const int num_heads = static_cast<int>(H.unwrap());
  const int batch = static_cast<int>(B.unwrap());
  const int max_seqblock = static_cast<int>(S.unwrap());
  const int topk_i = static_cast<int>(T.unwrap());
  const DLDevice device = device_.unwrap();

  RuntimeCheck(
      static_cast<int64_t>(topk_i) == topk,
      "minimax_decode_topk: topk arg (",
      topk,
      ") must match topk_idx last dim (",
      topk_i,
      ")");
  RuntimeCheck(block_size > 0, "block_size must be > 0, got ", block_size);
  RuntimeCheck(
      topk >= 1 && topk <= static_cast<int64_t>(TopKTrait::kMaxTopK),
      "topk must be in [1, kMaxTopK] (ascending-sort smem buffer)");
  RuntimeCheck(
      max_seqblock <= static_cast<int>(TopKTrait::kMaxNumBlocks),
      "max_seqblock (",
      max_seqblock,
      ") exceeds kMaxNumBlocks (",
      TopKTrait::kMaxNumBlocks,
      ")");
  if (batch == 0 || num_heads == 0) return;

  const dim3 grid(static_cast<unsigned>(batch), static_cast<unsigned>(num_heads));
#ifdef USE_ROCM
  if (max_seqblock > 1024) {
    LaunchKernel(grid, TopKTraitWide::kCTASize, device, 0)
        .enable_pdl(kUsePDL)(
            minimax_decode_topk_block_kernel<SeqLenT, kUsePDL, TopKTraitWide>,
            static_cast<const float*>(score.data_ptr()),
            static_cast<const SeqLenT*>(seq_lens.data_ptr()),
            static_cast<int32_t*>(topk_idx.data_ptr()),
            batch,
            num_heads,
            max_seqblock,
            static_cast<int>(block_size),
            topk_i);
    return;
  }
#endif
  LaunchKernel(grid, TopKTrait::kCTASize, device, 0)
      .enable_pdl(kUsePDL)(
          minimax_decode_topk_block_kernel<SeqLenT, kUsePDL, TopKTrait>,
          static_cast<const float*>(score.data_ptr()),
          static_cast<const SeqLenT*>(seq_lens.data_ptr()),
          static_cast<int32_t*>(topk_idx.data_ptr()),
          batch,
          num_heads,
          max_seqblock,
          static_cast<int>(block_size),
          topk_i);
}

// Page-table variant: emit the per-(batch, kv-head) paged page table consumed by
// the dense backend (trtllm_mha / fa3) plus the effective KV length, instead of
// block ids. For DP attention (num_kv_heads > 1) each kv head selects its own
// blocks, so (b, h) pseudo-requests are flattened batch-major into the output
// (B*num_heads rows); num_heads == 1 is the TP>=4 single-kv-head case. The page
// index is head-encoded (head-minor) as base_page*num_heads + h -- the index into
// an HND cache [num_pages, nkv, ps, D] reshaped to [num_pages*nkv, 1, ps, D].
// page_table and seq_lens_out are allocated by the caller.
template <typename SeqLenT, bool kUsePDL>
void minimax_decode_topk_page_table(
    tvm::ffi::TensorView score,         // [H, B, S] fp32 (H = num_kv_heads)
    tvm::ffi::TensorView seq_lens,      // [B] int32/int64
    tvm::ffi::TensorView req_to_token,  // [max_reqs, max_kv_len] int32
    tvm::ffi::TensorView slot_ids,      // [B] int64 (req_pool_indices)
    tvm::ffi::TensorView page_table,    // [B*H, max_sparse_pages] int32 (out)
    tvm::ffi::TensorView seq_lens_out,  // [B*H] int32 (effective KV length, out)
    int64_t block_size,
    int64_t topk,
    int64_t page_size) {
  using namespace host;

  SymbolicSize H = {"num_heads"};
  SymbolicSize B = {"batch"};
  SymbolicSize S = {"max_seqblock"};
  SymbolicSize R = {"max_reqs"};
  SymbolicSize KV = {"max_kv_len"};
  SymbolicSize BH = {"batch_heads"};
  SymbolicSize P = {"max_sparse_pages"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({H, B, S}).with_dtype<fp32_t>().with_device(device_).verify(score);
  TensorMatcher({B}).with_dtype<SeqLenT>().with_device(device_).verify(seq_lens);
  TensorMatcher({R, KV}).with_dtype<int32_t>().with_device(device_).verify(req_to_token);
  TensorMatcher({B}).with_dtype<int64_t>().with_device(device_).verify(slot_ids);
  TensorMatcher({BH, P}).with_dtype<int32_t>().with_device(device_).verify(page_table);
  TensorMatcher({BH}).with_dtype<int32_t>().with_device(device_).verify(seq_lens_out);

  const int num_heads = static_cast<int>(H.unwrap());
  const int batch = static_cast<int>(B.unwrap());
  const int max_seqblock = static_cast<int>(S.unwrap());
  const int max_kv_len = static_cast<int>(KV.unwrap());
  const int max_sparse_pages = static_cast<int>(P.unwrap());
  const int r2t_stride = static_cast<int>(req_to_token.stride(0));
  const DLDevice device = device_.unwrap();

  RuntimeCheck(
      BH.unwrap() == static_cast<int64_t>(batch) * num_heads,
      "page_table rows (",
      BH.unwrap(),
      ") must equal batch*num_heads (",
      static_cast<int64_t>(batch) * num_heads,
      ")");
  RuntimeCheck(
      block_size > 0 && page_size > 0 && block_size % page_size == 0,
      "block_size must be a positive multiple of page_size");
  RuntimeCheck(
      topk >= 1 && topk <= static_cast<int64_t>(TopKTrait::kMaxTopK),
      "topk must be in [1, kMaxTopK] for page-table mode");
  RuntimeCheck(
      max_seqblock <= static_cast<int>(TopKTrait::kMaxNumBlocks),
      "max_seqblock (",
      max_seqblock,
      ") exceeds kMaxNumBlocks (",
      TopKTrait::kMaxNumBlocks,
      ")");
  if (batch == 0 || num_heads == 0) return;

  const dim3 grid(static_cast<unsigned>(batch), static_cast<unsigned>(num_heads));
#ifdef USE_ROCM
  if (max_seqblock > 1024) {
    LaunchKernel(grid, TopKTraitWide::kCTASize, device, 0)
        .enable_pdl(kUsePDL)(
            minimax_decode_topk_page_table_kernel<SeqLenT, kUsePDL, TopKTraitWide>,
            static_cast<const float*>(score.data_ptr()),
            static_cast<const SeqLenT*>(seq_lens.data_ptr()),
            static_cast<const int32_t*>(req_to_token.data_ptr()),
            static_cast<const int64_t*>(slot_ids.data_ptr()),
            static_cast<int32_t*>(page_table.data_ptr()),
            static_cast<int32_t*>(seq_lens_out.data_ptr()),
            batch,
            num_heads,
            max_seqblock,
            static_cast<int>(block_size),
            static_cast<int>(topk),
            static_cast<int>(page_size),
            r2t_stride,
            max_kv_len,
            max_sparse_pages);
    return;
  }
#endif
  LaunchKernel(grid, TopKTrait::kCTASize, device, 0)
      .enable_pdl(kUsePDL)(
          minimax_decode_topk_page_table_kernel<SeqLenT, kUsePDL, TopKTrait>,
          static_cast<const float*>(score.data_ptr()),
          static_cast<const SeqLenT*>(seq_lens.data_ptr()),
          static_cast<const int32_t*>(req_to_token.data_ptr()),
          static_cast<const int64_t*>(slot_ids.data_ptr()),
          static_cast<int32_t*>(page_table.data_ptr()),
          static_cast<int32_t*>(seq_lens_out.data_ptr()),
          batch,
          num_heads,
          max_seqblock,
          static_cast<int>(block_size),
          static_cast<int>(topk),
          static_cast<int>(page_size),
          r2t_stride,
          max_kv_len,
          max_sparse_pages);
}

}  // namespace sglang
