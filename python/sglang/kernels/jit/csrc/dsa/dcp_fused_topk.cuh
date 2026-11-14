// Fused global top-k of the DSA indexer for decode context parallelism over 8
// ranks (DCP8): global position p lives on rank p % 8 at local position p / 8.
//
// One CTA per row on every rank, all CTAs of all ranks resident at once (rows
// <= SM count). The row is not staged in shared memory: each thread loads its
// 16 B vectors into registers right after the PDL wait (one batch of up to
// 32768 keys; longer rows batch by batch), and only candidates go to shared
// memory:
//   1. sampled coarse histogram (2048 keys from global memory, 8192 for rows
//      over 64K keys) -> filter threshold; filter the registers into a candidate
//      buffer (every key whose local rank is about 2 hi or better);
//   2. fine histogram of the candidates -> the window of local ranks [lo, hi]
//      at bin granularity: A = #keys above it, W = its keys (never splits ties);
//   3. one Lamport round: the 768 B message {A + 1, |W| + 1 | flags, min W,
//      max W, W keys} is multicast into this row's slot on every rank, and each
//      of 384 threads spins on one 16 B vector of the local copy (every word
//      nonzero). While the messages fly, the A keys above the window go out at
//      [0, A) (they are > t* whenever the certificate holds) and the window keys
//      in descending order at [A, A + W);
//   4. decision, identical on every rank, by warps 0..3 only (the other warps
//      leave, or join the rare paths): t* = the (k - sum A)-th key of the union
//      of the windows, certified exact when t* lies inside every window; ties
//      at t* are split by quotas in rank order;
//   5. this rank's selection is then a prefix of what went out: keys > t* plus
//      its tie quota; the rest of [A, A + W) becomes -1, and the count.
// A failed certificate (rank-uniform) runs an exact fallback: three histogram
// all-gather rounds of 11 + 11 + 10 key bits through a symmetric buffer, each
// closed by a release / acquire counter on every rank.
//
// Contract: context <= 1M (131072 local keys, whose page ids fit in shared
// memory once the filter is done); a longer row is cut to its first 131072 keys.
//
// Exchange buffers alternate by the parity of a per-row exchange counter: exchange
// e zeroes parity (e - 1) & 1 of its row, which peers write again only at
// exchange e + 1, after they have received this rank's exchange-e message. A call
// without an exchange (L <= k) leaves the counter and the buffers alone: it waits
// on no peer, so a zeroing there could wipe a faster peer's next message. The
// zeroing is plain stores, ordered before the peers' next writes only by this
// rank's message that follows them (a full NVLink round trip), not by a fence.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/ptx.cuh>
#include <sgl_kernel/ptx/barrier.cuh>  // ptx::to_shared, ptx::bar_sync, ptx::bar_arrive

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp16.h>

namespace sglang {

namespace device::ptx {

// Lamport store: a peer polls these words with relaxed loads, so the store is
// .relaxed.sys (a weak multimem store would race with them).
SGL_DEVICE void multimem_st_relaxed_v4(uint32_t* mc_ptr, uint4 v) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(mc_ptr),
               "r"(v.x),
               "r"(v.y),
               "r"(v.z),
               "r"(v.w)
               : "memory");
}

// Shared histogram increments: atomicAdd in their place measured about 0.1 us slower.
SGL_DEVICE void red_shared_add1(uint32_t addr) {
  asm volatile("red.shared.add.u32 [%0], 1;" ::"r"(addr) : "memory");
}

SGL_DEVICE void red_shared_add1_if(uint32_t addr, bool pred) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %1, 0;\n\t@p red.shared.add.u32 [%0], 1;\n\t}" ::"r"(addr),
               "r"(static_cast<uint32_t>(pred))
               : "memory");
}

// Predicated store: in the early emit loop, between warp ballots and shuffles, a
// guarded C++ store compiles to BSSY / BRA (+30 instructions, +5 BSSY).
SGL_DEVICE void st_global_if(int32_t* ptr, int32_t v, bool pred) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %2, 0;\n\t@p st.global.b32 [%0], %1;\n\t}" ::"l"(ptr),
               "r"(v),
               "r"(static_cast<uint32_t>(pred))
               : "memory");
}

// mask |= bit when x >= t (false for NaN), as a predicated OR. Every C++ spelling
// becomes one SEL per key plus an add tree (6 ops per 4 keys instead of 4).
SGL_DEVICE void or_if_ge(uint32_t& mask, float x, float t, uint32_t bit) {
  asm("{\n\t.reg .pred p;\n\tsetp.ge.f32 p, %1, %2;\n\t@p or.b32 %0, %0, %3;\n\t}"
      : "+r"(mask)
      : "f"(x), "f"(t), "r"(bit));
}

}  // namespace device::ptx

namespace device::dcp_fused_topk {

constexpr uint32_t kThreads = 512;
constexpr uint32_t kWarps = kThreads / 32;
constexpr uint32_t kTopK = 2048;
constexpr uint32_t kRanks = 8;
constexpr uint32_t kPageSize = 64;
// message slot: 4 header words + 188 window keys = 48 vectors of 16 B
constexpr uint32_t kSlotWords = 192;
constexpr uint32_t kSlotKeys = 188;
constexpr uint32_t kSlotVecs = 48;
constexpr uint32_t kPollThreads = kRanks * kSlotVecs;
constexpr uint32_t kCandCap = 4096;   // candidate buffer
constexpr uint32_t kMaxPages = 2048;  // 131072 local keys (1M context)
constexpr uint32_t kCoarseBins = 4096;
constexpr uint32_t kRadixBins = 2048;  // an 11-bit round of the slow decision and of the fallback
// with the pad words of hist_index
constexpr uint32_t kCoarseWords = kCoarseBins + kCoarseBins / 4;
constexpr uint32_t kRadixWords = kRadixBins + kRadixBins / 2;
constexpr uint32_t kWinBins = 512;  // 512 never overflowed a slot on real rows, 256 did in 0.9 %
constexpr uint32_t kDecBins = 512;
constexpr uint32_t kFallbackWords = 2048;  // fallback histogram words per rank
constexpr uint32_t kScanThreads = 256;
// The decision runs on warps 0..3, one per scheduler: work done by all 16 warps
// costs 4 issue slots per instruction.
constexpr uint32_t kWorkers = 128;
constexpr uint32_t kTStarCap = 128;  // keys of the t* bin the fast decision gathers

// message header {A + 1, (|W| + 1) | flags, min W, max W}
constexpr uint32_t kFlagTop = 1u << 16, kFlagBottom = 1u << 17, kFlagBad = 1u << 18;
constexpr uint32_t kFlagsMask = 0x00FF0000u;

// Named barriers; 0 is __syncthreads.
enum Barrier : uint32_t {
  BAR_SCAN = 1,    // threads [0, kScanThreads) of a 256-thread search
  BAR_PARK = 2,    // the whole CTA: warps 4..15 wait for the verdict
  BAR_GATHER = 3,  // the workers: t* bin keys gathered
  BAR_WINDOW = 4,  // the workers: warp totals of the window search
};

// How the decision ended, the same on every rank.
enum Verdict : uint32_t {
  CERTIFIED = 0,  // fast decision, certificate held
  FALLBACK = 1,   // exact fallback
  SLOW = 2,       // slow decision (radix select over the union), then its certificate
};

struct Params {
  const float* logits;
  int64_t logits_stride;
  const int32_t* local_lens;
  const int32_t* global_lens;
  const int32_t* page_table;
  int64_t page_table_stride;
  int32_t* out_local;    // [rows, kTopK] local slots, -1 after
  int32_t* out_slots;    // [rows, kTopK] widened slots, -1 after
  int32_t* out_len;      // [rows]
  uint32_t* msg_buf;     // [2][max_rows][kRanks][kSlotWords] message slots
  uint32_t* msg_mc;      // its multicast address
  uint32_t* hist_buf;    // [2][max_rows][kRanks][kFallbackWords] fallback histograms
  uint32_t* hist_mc;     // its multicast address
  uint32_t* arrive_buf;  // [max_rows][32] fallback arrival counters
  uint32_t* arrive_mc;   // its multicast address
  int32_t* calls;        // [max_rows] exchanges per row
  int32_t* fallbacks;    // [max_rows] fallbacks per row
  uint32_t rank;
  uint32_t max_rows;
  uint32_t lo;  // window: local ranks [lo, hi]
  uint32_t hi;
};

// This rank's window: A keys above it, W keys in it, its message flags, and its
// fine-bin range [bottom_bin, top_bin] (top_bin holds local rank lo).
struct Window {
  uint32_t above;
  uint32_t size;
  uint32_t flags;
  uint32_t top_bin;
  uint32_t bottom_bin;
};

// A rank search result: the bin holding rank k and the count above that bin
// (the window search also stores the bin's count).
struct RankHit {
  uint32_t bin;
  uint32_t above;
};

struct SearchResult {
  RankHit hit[2];
  uint32_t count[2];
};

// Scalars thread 0 publishes to the CTA through a barrier.
struct Broadcast {
  Window window;  // read again by the final emit
  uint32_t verdict;
  uint32_t union_min;  // min and max key over all windows
  uint32_t union_max;
  uint32_t union_rank;  // rank of t* in the union of the windows
  uint32_t slow_tstar;  // the slow decision's t* and the ties at it still to take
  uint32_t slow_need;
  uint32_t fallback_tstar;  // the fallback's emission rule for this rank
  uint32_t fallback_quota;
  uint32_t range_lo;  // a key range from retune_bound / rerange
  uint32_t range_hi;
};

// Members read or written as 16 B vectors are alignas(16).
struct alignas(16) Smem {
  // sampled coarse histogram; the decision's bins; the slow decision's two radix histograms
  alignas(16) uint32_t hist_sample[2 * kRadixWords];
  alignas(16) uint32_t hist_coarse[kCoarseWords];  // exact coarse histogram; the landed keys during the poll
  alignas(16) uint32_t hist_win[kRadixWords];      // window bins; the fallback's radix bins
  uint32_t group_totals[2][32];                    // group totals (64 bins each) of the slow decision's histograms
  uint2 cand[kCandCap];                            // {local position | window bin << kPosBits, key}
  alignas(16) uint32_t msg[kSlotWords];
  uint32_t win_cand[kSlotWords];             // candidate index of window key msg[4 + j]
  uint32_t win_bin[kSlotWords];              // its window bin
  alignas(16) uint32_t win_above[kWinBins];  // window bins: candidates in higher bins
  alignas(16) uint32_t headers[kRanks][4];
  alignas(16) uint32_t warp_sums[kWarps];
  SearchResult found;
  uint32_t rank_ties[kRanks];                  // keys equal to t* per rank (slow decision, fallback)
  alignas(16) uint32_t tstar_keys[kTStarCap];  // fast decision: keys of the t* bin and their rank
  uint32_t tstar_ranks[kTStarCap];
  uint32_t tstar_count;
  uint32_t cand_count;
  uint32_t out_count;
  uint32_t tie_claims;  // tie quota claims; the window slots when the window is the whole row
  uint32_t win_min;
  uint32_t win_max;
  Broadcast bcast;
};

SGL_DEVICE Smem& get_smem() {
  extern __shared__ __align__(128) uint8_t smem_raw[];
  return *reinterpret_cast<Smem*>(smem_raw);
}

// ---------------------------------------------------------------- keys and bins

// Order-preserving u32 key of a canonical (non-NaN) score; -inf maps to
// 0x007FFFFF, so no key is 0 and 0 marks an empty exchange word.
SGL_DEVICE uint32_t to_key(float f) {
  const uint32_t b = __float_as_uint(f);
  return b ^ (static_cast<uint32_t>(static_cast<int32_t>(b) >> 31) | 0x80000000u);
}

SGL_DEVICE float canonical(float x) {
  return fmaxf(x, -INFINITY);  // NaN -> -inf
}

// topk_v2's 12-bit coarse bin (fp16-ordered)
SGL_DEVICE uint32_t coarse_bin(float f) {
  const uint32_t h = __half_as_ushort(__float2half_rn(f));
  const uint32_t o = h ^ ((h & 0x8000u) ? 0xFFFFu : 0x8000u);
  return o >> 4;
}

SGL_DEVICE float code16_to_float(uint32_t c) {
  const uint32_t h = (c & 0x8000u) ? (c ^ 0x8000u) : (~c & 0xFFFFu);
  return __half2float(__ushort_as_half(static_cast<unsigned short>(h)));
}

// a key bound below every key whose coarse bin is >= b (may admit up to half an fp16 ulp of bin b - 1)
SGL_DEVICE uint32_t coarse_lo_key(uint32_t b) {
  if (b <= 0x40u) return 0u;
  if (b > 0xFC0u) return 0xFFFFFFFFu;
  return to_key(code16_to_float((b << 4) - 1u)) + 1u;
}

// a key bound above every key whose coarse bin is <= b
SGL_DEVICE uint32_t coarse_hi_key(uint32_t b) {
  if (b + 1u > 0xFC0u) return 0xFFFFFFFFu;
  if (b < 0x40u) return 0x00800000u;
  return to_key(code16_to_float((b + 1u) << 4));
}

// inverse of to_key; keys at or below -inf map to -inf, keys above +inf to NaN (compares false)
SGL_DEVICE float key_to_float(uint32_t k) {
  if (k <= 0x007FFFFFu) return -INFINITY;
  return __uint_as_float((k & 0x80000000u) ? (k ^ 0x80000000u) : ~k);
}

SGL_DEVICE uint32_t bit_length(uint32_t x) {
  return 32u - __clz(x);
}

// Window bins are linear in the score over [score(lo_key), score(hi_key)]:
// linear in the key bits failed when the range crossed zero or several binades.
struct WinMap {
  float base;
  float scale;
  SGL_DEVICE uint32_t bin(uint32_t k) const {
    const float v = (key_to_float(k) - base) * scale;
    return v <= 0.f ? 0u : (v >= static_cast<float>(kWinBins - 1) ? kWinBins - 1 : static_cast<uint32_t>(v));
  }
};

SGL_DEVICE WinMap make_win_map(uint32_t lo_key, uint32_t hi_key) {
  const float lo = key_to_float(lo_key), hi = key_to_float(hi_key);
  WinMap map;
  map.base = lo == -INFINITY ? -3.0e38f : lo;
  const float span = (hi == INFINITY || isnan(hi) ? 3.0e38f : hi) - map.base;
  // any monotone map works: one map per call
  map.scale = span > 0.f ? __fdividef(static_cast<float>(kWinBins), span) : 0.f;
  return map;
}

// ---------------------------------------------------------------- rank searches

// Histograms scanned by 256 threads keep each thread's kPer bins contiguous; for
// kPer >= 8 every group of kPer bins is followed by 4 pad words, so the 16 B
// reads (pitch an odd number of 16 B units) are bank-conflict free.
template <uint32_t kBins>
SGL_DEVICE constexpr uint32_t hist_index(uint32_t b) {
  constexpr uint32_t kPer = kBins / kScanThreads;
  if constexpr (kPer >= 8) {
    return b + (b / kPer) * 4;
  } else {
    return b;
  }
}

// The decision histogram (512 bins) is searched by each worker warp on its own:
// lane l owns the 16 bins 512 - 16 (l + 1) .. 512 - 16 l - 1, and every 16 bins
// are followed by 4 pad words (16 B reads conflict free).
constexpr uint32_t kDecWords = kDecBins + kDecBins / 4;
SGL_DEVICE constexpr uint32_t dec_index(uint32_t b) {
  return b + (b / 16) * 4;
}

// Load kPer contiguous counters starting at hist[0] (16 B aligned) into registers.
template <uint32_t kPer>
SGL_DEVICE void load_counts(const uint32_t* hist, uint32_t (&counts)[kPer]) {
  if constexpr (kPer >= 4) {
#pragma unroll
    for (uint32_t q = 0; q < kPer / 4; ++q) {
      const uint4 y = reinterpret_cast<const uint4*>(hist)[q];
      counts[4 * q] = y.x;
      counts[4 * q + 1] = y.y;
      counts[4 * q + 2] = y.z;
      counts[4 * q + 3] = y.w;
    }
  } else if constexpr (kPer == 2) {
    const uint2 y = *reinterpret_cast<const uint2*>(hist);
    counts[0] = y.x;
    counts[1] = y.y;
  } else {
    counts[0] = hist[0];
  }
}

template <uint32_t kPer>
SGL_DEVICE uint32_t sum_counts(const uint32_t (&counts)[kPer]) {
  uint32_t total = 0;
#pragma unroll
  for (uint32_t i = 0; i < kPer; ++i) {
    total += counts[i];
  }
  return total;
}

// Run by threads [0, 256) only, over a padded histogram. For ranks r0, r1
// (1-based from the top, 1 <= r <= total): the bin holding it and the count
// above it, in found.hit[0], found.hit[1]. The caller publishes the result with
// a barrier.
template <uint32_t kBins>
SGL_DEVICE void half_cta_find(const uint32_t* hist, uint32_t r0, uint32_t r1, Smem& smem, uint32_t tid) {
  constexpr uint32_t kPer = kBins / kScanThreads;
  static_assert(kPer % 4 == 0);
  const uint32_t lane_id = tid % 32, warp_id = tid / 32;
  const uint32_t first_bin = kBins - kPer * (tid + 1);
  uint32_t counts[kPer];
  load_counts(hist + hist_index<kBins>(first_bin), counts);
  const uint32_t total = sum_counts(counts);
  const uint32_t incl = warp::inclusive_sum(total, lane_id);
  if (lane_id == 31) smem.warp_sums[warp_id] = incl;
  ptx::bar_sync(BAR_SCAN, kScanThreads);
  const uint32_t warp_total = lane_id < kScanThreads / 32 ? smem.warp_sums[lane_id] : 0;
  const uint32_t warp_incl = warp::inclusive_sum<32, 1, kScanThreads / 32>(warp_total, lane_id);
  // keys above this thread's bins
  const uint32_t above0 = __shfl_sync(0xffffffffu, warp_incl - warp_total, warp_id) + incl - total;
  const bool has_r0 = above0 < r0 && r0 <= above0 + total;
  const bool has_r1 = above0 < r1 && r1 <= above0 + total;
  if (has_r0 || has_r1) {
    uint32_t above = above0;
#pragma unroll
    for (int i = kPer - 1; i >= 0; --i) {
      const uint32_t count = counts[i];
      if (has_r0 && above < r0 && r0 <= above + count) {
        smem.found.hit[0].bin = first_bin + i;
        smem.found.hit[0].above = above;
      }
      if (has_r1 && above < r1 && r1 <= above + count) {
        smem.found.hit[1].bin = first_bin + i;
        smem.found.hit[1].above = above;
      }
      above += count;
    }
  }
}

// One warp over a histogram kept with group totals (bin b also counted in
// groups[b / 64]): for ranks r0, r1 the bin and the count above it, in
// found.hit[0], found.hit[1].
template <uint32_t kBins>
SGL_DEVICE void grouped_warp_find(
    const uint32_t* hist, const uint32_t* groups, uint32_t r0, uint32_t r1, Smem& smem, uint32_t lane_id) {
  constexpr uint32_t kGroups = kBins / 64, kPer = kGroups / 32;
  static_assert(kPer == 1 || kPer == 2);
  // lane l owns groups kGroups - kPer (l + 1) .. kGroups - kPer l - 1, scanned from the top
  uint32_t group_counts[kPer];
  if constexpr (kPer == 2) {
    const uint2 y = *reinterpret_cast<const uint2*>(groups + kGroups - 2 * (lane_id + 1));
    group_counts[0] = y.x;
    group_counts[1] = y.y;
  } else {
    group_counts[0] = groups[kGroups - 1 - lane_id];
  }
  const uint32_t total = sum_counts(group_counts);
  const uint32_t above_lane = warp::inclusive_sum(total, lane_id) - total;
  const uint32_t ranks[2] = {r0, r1};
#pragma unroll
  for (uint32_t t = 0; t < 2; ++t) {
    const uint32_t r = ranks[t];
    const uint32_t who = __ffs(__ballot_sync(0xffffffffu, above_lane < r && r <= above_lane + total)) - 1;
    uint32_t group = kGroups - kPer * (who + 1);
    uint32_t above = __shfl_sync(0xffffffffu, above_lane, who);
    if constexpr (kPer == 2) {
      const uint32_t top = __shfl_sync(0xffffffffu, group_counts[1], who);  // the higher group of that lane
      if (r > above + top) {
        above += top;
      } else {
        group += 1;
      }
    }
    // the 64 bins of the group: lane l holds bins group * 64 + 63 - 2 l and the one below
    const uint2 y = *reinterpret_cast<const uint2*>(hist + group * 64 + 62 - 2 * lane_id);
    const uint32_t pair = y.x + y.y;
    const uint32_t above_pair = above + warp::inclusive_sum(pair, lane_id) - pair;
    const bool in_hi = above_pair < r && r <= above_pair + y.y;
    const bool in_lo = above_pair + y.y < r && r <= above_pair + pair;
    if (in_hi) {
      smem.found.hit[t].bin = group * 64 + 63 - 2 * lane_id;
      smem.found.hit[t].above = above_pair;
    }
    if (in_lo) {
      smem.found.hit[t].bin = group * 64 + 62 - 2 * lane_id;
      smem.found.hit[t].above = above_pair + y.y;
    }
  }
}

// The whole CTA over kBins unpadded bins (thread t owns kBins - kPer (t + 1) ..
// kBins - kPer t - 1): for kK ranks the bin and the count above it, in
// found.hit[k]. Two CTA barriers; the result is published when it returns.
template <uint32_t kBins, uint32_t kK>
SGL_DEVICE void cta_find(const uint32_t* hist, const uint32_t (&ranks)[kK], Smem& smem, uint32_t tid) {
  constexpr uint32_t kPer = kBins / kThreads;
  static_assert(kPer == 1 || kPer == 2 || kPer == 4 || kPer == 8);
  const uint32_t lane_id = tid % 32, warp_id = tid / 32;
  const uint32_t first_bin = kBins - kPer * (tid + 1);
  uint32_t counts[kPer];
  load_counts(hist + first_bin, counts);
  const uint32_t total = sum_counts(counts);
  const uint32_t incl = warp::inclusive_sum(total, lane_id);
  if (lane_id == 31) smem.warp_sums[warp_id] = incl;
  __syncthreads();
  const uint32_t warp_total = lane_id < kWarps ? smem.warp_sums[lane_id] : 0u;
  const uint32_t warp_above = __shfl_sync(0xffffffffu, warp::inclusive_sum(warp_total, lane_id) - warp_total, warp_id);
  const uint32_t above0 = warp_above + incl - total;
#pragma unroll
  for (uint32_t k = 0; k < kK; ++k) {
    if (above0 < ranks[k] && ranks[k] <= above0 + total) {
      uint32_t above = above0;
#pragma unroll
      for (int i = kPer - 1; i >= 0; --i) {
        if (above < ranks[k] && ranks[k] <= above + counts[i]) {
          smem.found.hit[k].bin = first_bin + i;
          smem.found.hit[k].above = above;
        }
        above += counts[i];
      }
    }
  }
  __syncthreads();
}

// One warp over the padded decision histogram, without a barrier: for kK ranks
// r[k] (1 <= r[k] <= total) the bin holding it, the count above it and the
// bin's count, in registers of every lane.
template <uint32_t kK>
SGL_DEVICE void warp_find(
    const uint32_t* hist,
    const uint32_t (&ranks)[kK],
    uint32_t (&bin)[kK],
    uint32_t (&above)[kK],
    uint32_t (&count)[kK],
    uint32_t lane_id) {
  constexpr uint32_t kPer = 16;
  static_assert(kDecBins == 32 * kPer);
  const uint32_t first_bin = kDecBins - kPer * (lane_id + 1);
  uint32_t counts[kPer];
  load_counts(hist + dec_index(first_bin), counts);
  const uint32_t total = sum_counts(counts);
  const uint32_t above0 = warp::inclusive_sum(total, lane_id) - total;
#pragma unroll
  for (uint32_t k = 0; k < kK; ++k) {
    uint32_t a = above0, hit_bin = 0, hit_above = 0, hit_count = 0;
#pragma unroll
    for (int i = kPer - 1; i >= 0; --i) {
      const bool hit = a < ranks[k] && ranks[k] <= a + counts[i];
      hit_bin = hit ? first_bin + i : hit_bin;
      hit_above = hit ? a : hit_above;
      hit_count = hit ? counts[i] : hit_count;
      a += counts[i];
    }
    const uint32_t who = __ffs(__ballot_sync(0xffffffffu, above0 < ranks[k] && ranks[k] <= above0 + total)) - 1;
    bin[k] = __shfl_sync(0xffffffffu, hit_bin, who);
    above[k] = __shfl_sync(0xffffffffu, hit_above, who);
    count[k] = __shfl_sync(0xffffffffu, hit_count, who);
  }
}

// The window search on the workers only (the other warps go straight to the
// caller's barrier): thread t < 128 owns the 4 bins 508 - 4 t .. 511 - 4 t. For
// the ranks r[k] (1 <= r <= total) the bin, the count above it and the bin's
// count in found.hit[k], found.count[k]; per bin the count above it in
// win_above[]. One named barrier among the workers; the caller publishes the
// result with a CTA barrier.
template <uint32_t kK>
SGL_DEVICE void window_find(const uint32_t* hist, const uint32_t (&ranks)[kK], Smem& smem, uint32_t tid) {
  constexpr uint32_t kPer = kWinBins / kWorkers;
  static_assert(kPer == 4);
  const uint32_t lane_id = tid % 32, warp_id = tid / 32;
  const uint32_t first_bin = kWinBins - kPer * (tid + 1);
  const uint4 y = *reinterpret_cast<const uint4*>(hist + first_bin);
  const uint32_t counts[kPer] = {y.x, y.y, y.z, y.w};
  const uint32_t total = y.x + y.y + y.z + y.w;
  const uint32_t incl = warp::inclusive_sum(total, lane_id);
  if (lane_id == 31) smem.warp_sums[warp_id] = incl;
  ptx::bar_sync(BAR_WINDOW, kWorkers);
  const uint4 sums = *reinterpret_cast<const uint4*>(smem.warp_sums);
  const uint32_t warp_above = (warp_id > 0 ? sums.x : 0u) + (warp_id > 1 ? sums.y : 0u) + (warp_id > 2 ? sums.z : 0u);
  const uint32_t above0 = warp_above + incl - total;
  *reinterpret_cast<uint4*>(smem.win_above + first_bin) = make_uint4(
      above0 + counts[3] + counts[2] + counts[1], above0 + counts[3] + counts[2], above0 + counts[3], above0);
#pragma unroll
  for (uint32_t k = 0; k < kK; ++k) {
    if (above0 < ranks[k] && ranks[k] <= above0 + total) {
      uint32_t above = above0;
#pragma unroll
      for (int i = kPer - 1; i >= 0; --i) {
        if (above < ranks[k] && ranks[k] <= above + counts[i]) {
          smem.found.hit[k].bin = first_bin + i;
          smem.found.hit[k].above = above;
          smem.found.count[k] = counts[i];
        }
        above += counts[i];
      }
    }
  }
}

// ---------------------------------------------------------------- the row

// This rank's keys of one row in global memory (not staged).
struct Row {
  const float* scores;  // key i at scores[i]
  uint32_t n;           // keys
  uint32_t head;        // keys before the first 16 B vector
  uint32_t body_vecs;   // 16 B vectors after the head
};

SGL_DEVICE Row make_row(const float* scores, uint32_t n) {
  Row row;
  row.scores = scores;
  row.n = n;
  row.head = min(n, static_cast<uint32_t>((4 - ((reinterpret_cast<uintptr_t>(scores) >> 2) & 3)) & 3));
  row.body_vecs = (n - row.head) / 4;
  return row;
}

// The last warp handles the keys outside the 16 B vectors: lanes 0..2 the head,
// lanes 4..7 the tail. Sets pos to the key of `lane` when it has one.
constexpr uint32_t kEdgeWarpFirst = kThreads - 32;
SGL_DEVICE void find_edge(const Row& row, uint32_t lane, uint32_t& pos) {
  const uint32_t tail = row.head + 4 * row.body_vecs;
  if (lane < row.head) {
    pos = lane;
  } else if (lane >= 4 && lane < 8 && tail + (lane - 4) < row.n) {
    pos = tail + (lane - 4);
  }
}

// Visit every key of the row from global memory as f(x4, pos) / f.edge(x, pos)
// (rare paths: the hot path reads the row once, into registers).
template <typename F>
SGL_DEVICE void for_row(const Row& row, uint32_t tid, F&& f) {
  const float4* vecs = reinterpret_cast<const float4*>(row.scores + row.head);
  for (uint32_t u = tid; u < row.body_vecs; u += kThreads) {
    const float4 x = vecs[u];
    f(x, row.head + 4 * u);
  }
  if (tid >= kEdgeWarpFirst) {
    uint32_t i = ~0u;
    find_edge(row, tid - kEdgeWarpFirst, i);
    if (i != ~0u) f.edge(row.scores[i], i);
  }
}

// The row body lives in registers: vector u = tid + j kThreads of the batch in
// vec[j], loaded right after the PDL wait (no shared-memory staging). Rows longer
// than one batch are filtered batch by batch, later batches loaded in the filter.
constexpr uint32_t kBatchVecs = 16;              // vectors per thread per batch
constexpr uint32_t kMaskWords = kBatchVecs / 4;  // 16-key hit masks per thread per batch

SGL_DEVICE void load_batch(const Row& row, float4 (&vec)[kBatchVecs], uint32_t batch, uint32_t tid) {
  const float4* vecs = reinterpret_cast<const float4*>(row.scores + row.head);
#pragma unroll
  for (uint32_t j = 0; j < kBatchVecs; ++j) {
    const uint32_t u = batch * kBatchVecs * kThreads + tid + j * kThreads;
    // __ldg: ld.global (a generic LD otherwise)
    vec[j] = u < row.body_vecs ? __ldg(vecs + u) : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
  }
}

// The sample: up to 256 pieces of 32 B (8 keys) at a uniform stride from the
// first 32 B-aligned key, each piece loaded by a thread pair (one LDG.128 per
// thread); rows longer than kShortRowKeys take kSampleLoads times as many. On
// real GLM-5.2 rows, 256 pieces never needed a retune up to 65536 local keys but
// did on 27 % of 122880-key shards; 1024 pieces on none.
constexpr uint32_t kSampleLoads = 4;
constexpr uint32_t kShortRowKeys = 65536;
constexpr uint32_t kSamplePieces = kThreads / 2;

struct SampleGeometry {
  uint32_t first;   // first 32 B-aligned key
  uint32_t span;    // keys from there
  uint32_t pieces;  // pieces at most
  uint32_t stride;  // keys between pieces, a multiple of 8
};

SGL_DEVICE SampleGeometry sample_geometry(const Row& row) {
  SampleGeometry g;
  g.first = static_cast<uint32_t>((8 - ((reinterpret_cast<uintptr_t>(row.scores) >> 2) & 7)) & 7);
  g.span = row.n > g.first ? row.n - g.first : 0;
  g.pieces = row.n > kShortRowKeys ? kSampleLoads * kSamplePieces : kSamplePieces;
  g.stride = max(8u, (g.span / g.pieces) / 8 * 8);
  return g;
}

// This thread's half of piece (tid / 2 + load * kSamplePieces); false when the
// piece is not in the row.
SGL_DEVICE bool load_sample(const Row& row, const SampleGeometry& g, uint32_t tid, uint32_t load, float4& keys) {
  const uint32_t piece = (tid >> 1) + load * kSamplePieces;
  const uint32_t i0 = g.first + piece * g.stride + 4 * (tid & 1);
  const bool valid = piece < g.pieces && i0 + 4 <= row.n && piece * g.stride + 8 <= g.span;
  if (valid) keys = __ldg(reinterpret_cast<const float4*>(row.scores + i0));
  return valid;
}

// Keys of the whole sample (thread pairs whose piece is in the row), in closed form.
SGL_DEVICE uint32_t sample_size(const SampleGeometry& g) {
  return g.span >= 8 ? 8 * min(g.pieces, (g.span - 8) / g.stride + 1) : 0u;
}

// Filter bound and window range.
struct Bounds {
  uint32_t threshold_key;  // candidates: keys >= threshold_key (0: every key)
  float threshold;         // threshold_key as a score
  uint32_t base_key;       // window range [base_key, top_key]
  uint32_t top_key;
};

SGL_DEVICE void set_threshold(Bounds& b, uint32_t key) {
  b.threshold_key = key;
  b.threshold = key_to_float(key);
  if (b.threshold == -INFINITY) b.threshold_key = 0;
}

// Filter one batch: pass 1 builds one 16-key hit mask per step (vectors j = 4 w +
// q) for all of the thread's vectors, so the slot range is claimed once per
// thread (one warp scan, one shared atomic per warp); pass 2 appends the hits'
// positions word by word (the values are gathered afterwards from global memory,
// L1-resident). `all` takes every valid key. The hit count accumulates in
// cand_count (may exceed kCandCap).
SGL_DEVICE void filter_batch(
    const Row& row,
    const float4 (&vec)[kBatchVecs],
    uint32_t first_vec,
    Smem& smem,
    uint32_t tid,
    float threshold,
    bool all,
    uint32_t edge_pos,
    float edge_score) {
  const uint32_t lane_id = tid % 32;
  const float compare_to = all ? INFINITY : threshold;  // all: the compare selects nothing, the valid bits are ORed in
  uint32_t hit_mask[kMaskWords];
  uint32_t hits = 0;
#pragma unroll
  for (uint32_t w = 0; w < kMaskWords; ++w) {
    hit_mask[w] = 0;
#pragma unroll
    for (uint32_t q = 0; q < 4; ++q) {
      const uint32_t j = 4 * w + q;
      if (first_vec + j * kThreads >= row.body_vecs) continue;  // CTA-uniform: no thread holds vector j
      const float x[4] = {vec[j].x, vec[j].y, vec[j].z, vec[j].w};
      uint32_t quad = 0;
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        ptx::or_if_ge(quad, x[e], compare_to, 1u << (4 * q + e));
      }
      const uint32_t valid = first_vec + tid + j * kThreads < row.body_vecs ? 0xFu << (4 * q) : 0u;
      hit_mask[w] |= (quad & valid) | (all ? valid : 0u);  // every valid key, NaN included
    }
    hits += __popc(hit_mask[w]);
  }
  const bool edge_hit = edge_pos != ~0u && (all || edge_score >= threshold);
  hits += edge_hit;
  const uint32_t incl = warp::inclusive_sum(hits, lane_id);
  uint32_t base = 0;
  if (lane_id == 31 && incl) base = atomicAdd(&smem.cand_count, incl);
  uint32_t slot = __shfl_sync(0xffffffffu, base, 31) + incl - hits;
  if (edge_hit && slot < kCandCap) smem.cand[slot].x = edge_pos;
  slot += edge_hit;
  // one hit loop over the 64 mask bits of the four steps (bit b: vector b / 4 of
  // the thread, element b % 4): one warp max and one loop instead of four
  static_assert(kMaskWords == 4 && kThreads == 512);
  uint32_t mask_lo = hit_mask[0] | hit_mask[1] << 16;
  uint32_t mask_hi = hit_mask[2] | hit_mask[3] << 16;
  const uint32_t max_hits = __reduce_max_sync(0xffffffffu, __popc(mask_lo) + __popc(mask_hi));
  const uint32_t first_pos = row.head + 4 * (first_vec + tid);
  for (uint32_t k = 0; k < max_hits; ++k) {
    const bool active = (mask_lo | mask_hi) != 0, in_lo = mask_lo != 0;
    const uint32_t b = (__ffs(in_lo ? mask_lo : mask_hi) - 1) + (in_lo ? 0u : 32u);
    mask_lo = in_lo ? mask_lo & (mask_lo - 1) : mask_lo;
    mask_hi = in_lo ? mask_hi : mask_hi & (mask_hi - 1);
    const uint32_t pos = first_pos + ((b >> 2) << 11) + (b & 3);
    if (active && slot < kCandCap) smem.cand[slot].x = pos;
    slot += active;
  }
}

SGL_DEVICE void zero_words(uint32_t* p, uint32_t words, uint32_t tid) {
  for (uint32_t i = tid; i < words / 4; i += kThreads) {
    reinterpret_cast<uint4*>(p)[i] = make_uint4(0, 0, 0, 0);
  }
}

// The kRanks message slots of one row and parity.
SGL_DEVICE uint32_t* row_slots(uint32_t* buf, const Params& p, uint32_t parity, uint32_t row_id) {
  return buf + (static_cast<int64_t>(parity) * p.max_rows + row_id) * (kRanks * kSlotWords);
}

// Candidate word: the local position in the low kPosBits bits, its window bin above.
constexpr uint32_t kPosBits = 17;
static_assert(kMaxPages * kPageSize <= (1u << kPosBits) && kWinBins <= (1u << (32 - kPosBits)));

SGL_DEVICE uint32_t cand_word(uint32_t pos, uint32_t bin) {
  return pos | bin << kPosBits;
}

SGL_DEVICE uint32_t cand_pos(uint32_t word) {
  return word & ((1u << kPosBits) - 1u);
}

SGL_DEVICE uint32_t cand_bin(uint32_t word) {
  return word >> kPosBits;
}

// The row's page ids, stored after the filter into the tail of hist_coarse: the
// filter's rare paths are its last full users, the poll and the fallback stay below.
constexpr uint32_t kPageIdsOffset = 2 * 4 * kPollThreads;
static_assert(kPageIdsOffset + kMaxPages <= kCoarseWords);

SGL_DEVICE int32_t* page_ids(Smem& smem) {
  return reinterpret_cast<int32_t*>(smem.hist_coarse + kPageIdsOffset);
}

SGL_DEVICE int32_t local_slot(Smem& smem, uint32_t pos) {
  return page_ids(smem)[pos / kPageSize] * kPageSize + static_cast<int32_t>(pos % kPageSize);
}

// ---------------------------------------------------------------- rare paths
// Rare paths are noinline, out of the hot instruction stream.

SGL_DEVICE void win_hist_pass(const WinMap& map, Smem& smem, uint32_t count, uint32_t tid) {
  for (uint32_t i = tid; i < count; i += kThreads) {
    const uint32_t b = map.bin(smem.cand[i].y);
    smem.cand[i].x = cand_word(cand_pos(smem.cand[i].x), b);
    atomicAdd(&smem.hist_win[b], 1u);
  }
}

// threshold too high or too low: exact coarse histogram of the row; the new filter
// bound (local rank t, or just above that bin when it would overflow the buffer)
// and window top (rank t2) in bcast.range_lo, range_hi
__device__ __noinline__ void retune_bound(Row row, uint32_t tid, uint32_t t, uint32_t t2) {
  auto& smem = get_smem();
  zero_words(smem.hist_coarse, kCoarseWords, tid);
  zero_words(smem.hist_win, kWinBins, tid);
  __syncthreads();
  struct CoarseHist {
    Smem* smem;
    SGL_DEVICE void operator()(const float4& x, uint32_t) const {
      edge(x.x, 0);
      edge(x.y, 0);
      edge(x.z, 0);
      edge(x.w, 0);
    }
    SGL_DEVICE void edge(float x, uint32_t) const {
      atomicAdd(&smem->hist_coarse[hist_index<kCoarseBins>(coarse_bin(canonical(x)))], 1u);
    }
  };
  for_row(row, tid, CoarseHist{&smem});
  __syncthreads();
  if (tid < kScanThreads) half_cta_find<kCoarseBins>(smem.hist_coarse, t, t2, smem, tid);
  __syncthreads();
  if (tid == 0) {
    const RankHit bound = smem.found.hit[0];
    const bool fits = bound.above + smem.hist_coarse[hist_index<kCoarseBins>(bound.bin)] <= kCandCap;
    smem.bcast.range_lo = fits ? coarse_lo_key(bound.bin) : coarse_hi_key(bound.bin);
    smem.bcast.range_hi = coarse_hi_key(smem.found.hit[1].bin);
    smem.cand_count = 0;
  }
  __syncthreads();
}

// ranks lo / hi fall into a clamped end bin (the sample misjudged the range):
// exact coarse histogram of the m candidates, range from the bins of local ranks
// lo_t / 4 and min(m, 2 hi), window histogram again; the range in
// bcast.range_lo, range_hi
__device__ __noinline__ void rerange(uint32_t m, uint32_t lo_t, uint32_t hi2, uint32_t threshold_key, uint32_t tid) {
  auto& smem = get_smem();
  zero_words(smem.hist_coarse, kCoarseWords, tid);
  __syncthreads();
  zero_words(smem.hist_win, kWinBins, tid);
  for (uint32_t i = tid; i < m; i += kThreads) {
    atomicAdd(&smem.hist_coarse[hist_index<kCoarseBins>(coarse_bin(key_to_float(smem.cand[i].y)))], 1u);
  }
  __syncthreads();
  if (tid < kScanThreads) half_cta_find<kCoarseBins>(smem.hist_coarse, hi2, max(1u, lo_t / 4), smem, tid);
  __syncthreads();
  const uint32_t base_key = max(threshold_key, coarse_lo_key(smem.found.hit[0].bin));
  const uint32_t top_key = coarse_hi_key(smem.found.hit[1].bin);
  win_hist_pass(make_win_map(base_key, top_key), smem, m, tid);
  if (tid == 0) {
    smem.bcast.range_lo = base_key;
    smem.bcast.range_hi = top_key;
  }
  __syncthreads();
}

// Thread 0: the window holds more keys than a slot; drop the edge bin farther
// from the expected share until it fits. In and out: bcast.window.
__device__ __noinline__ void trim_window(uint32_t lo, uint32_t hi) {
  auto& smem = get_smem();
  Window& win = smem.bcast.window;
  uint32_t above = win.above, size = win.size, top_bin = win.top_bin, bottom_bin = win.bottom_bin;
  const int center = (static_cast<int>(lo) + static_cast<int>(hi)) / 2;
  while (size > kSlotKeys && bottom_bin <= top_bin) {
    const int d_up = center - static_cast<int>(above + 1), d_dn = static_cast<int>(above + size) - center;
    if (d_dn >= d_up) {
      size -= smem.hist_win[bottom_bin];
      ++bottom_bin;
    } else {
      above += smem.hist_win[top_bin];
      size -= smem.hist_win[top_bin];
      --top_bin;
    }
  }
  win.above = above;
  win.size = size;
  win.flags = size == 0 ? kFlagBad : 0;
  win.top_bin = top_bin;
  win.bottom_bin = bottom_bin;
}

// Radix select over the union of the windows, two or three 2048-bin rounds;
// t* and the ties still to take in bcast.slow_tstar, slow_need, per-rank counts
// of t* in rank_ties. `vec` is this thread's landed vector `vec_idx` of rank
// src_rank's slot.
__device__ __noinline__ void slow_decision(uint4 vec, uint32_t src_rank, uint32_t vec_idx, uint32_t tid) {
  auto& smem = get_smem();
  const uint32_t my_size = tid < kPollThreads ? (smem.headers[src_rank][1] & 0xFFFFu) - 1 : 0;
  zero_words(smem.hist_sample, 2 * kRadixWords, tid);
  if (tid < 32) {
    smem.group_totals[0][tid] = 0;
    smem.group_totals[1][tid] = 0;
  }
  __syncthreads();
  const uint32_t union_min = smem.bcast.union_min, union_max = smem.bcast.union_max;
  uint32_t rank_left = smem.bcast.union_rank, base = union_min, span = union_max - union_min;
  uint32_t shift = bit_length(span) > 11 ? bit_length(span) - 11 : 0;
  for (uint32_t round = 0;; ++round) {
    uint32_t* const hist = smem.hist_sample + (round & 1) * kRadixWords;
    uint32_t* const groups = smem.group_totals[round & 1];
    if (round >= 1) {
      zero_words(smem.hist_sample + ((round + 1) & 1) * kRadixWords, kRadixBins, tid);
      if (tid < 32) smem.group_totals[(round + 1) & 1][tid] = 0;
    }
    if (tid < kPollThreads && vec_idx > 0) {
      const uint32_t first_word = 4 * (vec_idx - 1);
      const uint32_t keys[4] = {vec.x, vec.y, vec.z, vec.w};
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t d = keys[i] - base;
        if (first_word + i < my_size && keys[i] >= base && d <= span) {
          atomicAdd(&hist[d >> shift], 1u);
          atomicAdd(&groups[(d >> shift) >> 6], 1u);
        }
      }
    }
    __syncthreads();
    if (tid < 32) grouped_warp_find<kRadixBins>(hist, groups, rank_left, rank_left, smem, tid);
    __syncthreads();
    const uint32_t b = smem.found.hit[0].bin;
    rank_left -= smem.found.hit[0].above;
    base += b << shift;
    if (shift == 0) break;
    span = (1u << shift) - 1;
    shift = shift > 11 ? shift - 11 : 0;
  }
  if (tid < kPollThreads && vec_idx > 0) {
    const uint32_t first_word = 4 * (vec_idx - 1);
    const uint32_t ties = (first_word < my_size && vec.x == base) + (first_word + 1 < my_size && vec.y == base) +
                          (first_word + 2 < my_size && vec.z == base) + (first_word + 3 < my_size && vec.w == base);
    if (ties) atomicAdd(&smem.rank_ties[src_rank], ties);
  }
  if (tid == 0) {
    smem.bcast.slow_tstar = base;
    smem.bcast.slow_need = rank_left;
  }
  __syncthreads();
}

// Exact fallback (certificate failed): 3 histogram all-gather rounds (11 + 11 +
// 10 key bits) from the row; this rank's emission rule in bcast.fallback_tstar,
// fallback_quota.
__device__ __noinline__ void exact_fallback(const Params p, Row row, uint32_t row_id, uint32_t me, uint32_t tid) {
  auto& smem = get_smem();
  const int32_t fallbacks = p.fallbacks[row_id];
  uint32_t prefix = 0, rank_left = kTopK;
  uint32_t* const arrive = p.arrive_buf + row_id * 32;
  for (uint32_t round = 0; round < 3; ++round) {
    const uint32_t shift = round == 0 ? 21 : (round == 1 ? 10 : 0), bits = round == 2 ? 10 : 11;
    zero_words(smem.hist_coarse, kFallbackWords, tid);
    __syncthreads();
    struct RadixHist {
      Smem* smem;
      uint32_t round, shift, bits, prefix;
      SGL_DEVICE void one(float x) const {
        const uint32_t k = to_key(canonical(x));
        if (round == 0 || (k >> (shift + bits)) == prefix) {
          atomicAdd(&smem->hist_coarse[(k >> shift) & ((1u << bits) - 1)], 1u);
        }
      }
      SGL_DEVICE void operator()(const float4& x, uint32_t) const {
        one(x.x);
        one(x.y);
        one(x.z);
        one(x.w);
      }
      SGL_DEVICE void edge(float x, uint32_t) const {
        one(x);
      }
    };
    for_row(row, tid, RadixHist{&smem, round, shift, bits, prefix});
    __syncthreads();
    const int64_t region = (static_cast<int64_t>(round & 1) * p.max_rows + row_id) * (kRanks * kFallbackWords);
    for (uint32_t i = tid; i < kFallbackWords / 4; i += kThreads) {
      ptx::st_multimem_16B(
          reinterpret_cast<const uint4*>(smem.hist_coarse)[i], p.hist_mc + region + me * kFallbackWords, i);
    }
    __syncthreads();
    if (tid == 0) {
      // the release orders this CTA's histogram stores (made cumulative by the barrier above)
      ptx::multimem_red_add_release(p.arrive_mc + row_id * 32, 1u);
      const auto target = static_cast<uint32_t>(fallbacks * 3 + round + 1) * kRanks;
      while (static_cast<int32_t>(ptx::load_acquire_sys(arrive) - target) < 0)
        ;
    }
    __syncthreads();
    for (uint32_t i = tid; i < (1u << bits) / 2; i += kThreads) {
      uint2 acc = make_uint2(0, 0);
      // by 4, not 8: fully unrolled, this function spills, and the kernel (which
      // takes its callees' registers and stack) gets a stack frame and a slower schedule
#pragma unroll 4
      for (uint32_t q = 0; q < kRanks; ++q) {
        uint2 y;
        ptx::ld_relaxed_8B(y, p.hist_buf + region + q * kFallbackWords, i);
        acc.x += y.x;
        acc.y += y.y;
      }
      *reinterpret_cast<uint2*>(smem.hist_win + (bits == 11 ? hist_index<kRadixBins>(2 * i) : 2 * i)) = acc;
    }
    __syncthreads();
    if (tid < kScanThreads) {
      if (bits == 11) {
        half_cta_find<kRadixBins>(smem.hist_win, rank_left, rank_left, smem, tid);
      } else {
        half_cta_find<1024>(smem.hist_win, rank_left, rank_left, smem, tid);
      }
    }
    __syncthreads();
    const uint32_t b = smem.found.hit[0].bin;
    rank_left -= smem.found.hit[0].above;
    prefix = (prefix << bits) | b;
    if (round == 2 && tid < kRanks) {
      smem.rank_ties[tid] = ptx::load_acquire_sys(p.hist_buf + region + tid * kFallbackWords + b);
    }
    __syncthreads();
  }
  if (tid == 0) {
    p.fallbacks[row_id] = fallbacks + 1;
    uint32_t before = 0;
    for (uint32_t q = 0; q < me; ++q) {
      before += smem.rank_ties[q];
    }
    smem.bcast.fallback_tstar = prefix;
    smem.bcast.fallback_quota = rank_left > before ? min(smem.rank_ties[me], rank_left - before) : 0;
  }
  __syncthreads();
}

struct RowEmitter {
  Smem* smem;
  int32_t* out_local;
  int32_t* out_slots;
  uint32_t tstar, quota, rank;
  SGL_DEVICE void key(uint32_t k, uint32_t pos) const {
    bool sel = k > tstar;
    if (!sel && quota && k == tstar) sel = atomicAdd(&smem->tie_claims, 1u) < quota;
    if (sel) {
      const uint32_t at = atomicAdd(&smem->out_count, 1u);
      if (at < kTopK) {
        const int32_t slot = local_slot(*smem, pos);
        out_local[at] = slot;
        out_slots[at] = slot * static_cast<int32_t>(kRanks) + static_cast<int32_t>(rank);
      }
    }
  }
  SGL_DEVICE void operator()(const float4& v, uint32_t pos) const {
    key(to_key(canonical(v.x)), pos);
    key(to_key(canonical(v.y)), pos + 1);
    key(to_key(canonical(v.z)), pos + 2);
    key(to_key(canonical(v.w)), pos + 3);
  }
  SGL_DEVICE void edge(float x, uint32_t pos) const {
    key(to_key(canonical(x)), pos);
  }
};

// emit from the row (fallback, L <= k)
__device__ __noinline__ void emit_row(Row row, RowEmitter emitter, uint32_t tid) {
  for_row(row, tid, emitter);
}

// ---------------------------------------------------------------- kernel stages

// Emission rule: keys > tstar plus the first `quota` keys equal to it; from the
// row, or as a prefix of what went out before the poll.
struct Rule {
  uint32_t tstar;
  uint32_t quota;
  bool from_row;
};

// Before the PDL wait: nothing the producer writes.
SGL_DEVICE void init_smem(Smem& smem, uint32_t tid) {
  zero_words(smem.hist_sample, kCoarseBins, tid);  // the sample histogram (rare paths zero what else they use)
  zero_words(smem.hist_win, kWinBins, tid);
  if (tid < 32) {
    smem.group_totals[0][tid] = 0;
    smem.group_totals[1][tid] = 0;
  }
  if (tid < kSlotWords) smem.msg[tid] = 1u;  // filler: nonzero, below every real key
  if (tid < kRanks) smem.rank_ties[tid] = 0;
  if (tid < kTStarCap) smem.tstar_keys[tid] = 0;
  if (tid == 0) {
    smem.cand_count = 0;
    smem.out_count = 0;
    smem.tie_claims = 0;
    smem.win_min = 0xFFFFFFFFu;
    smem.win_max = 0;
    smem.tstar_count = 0;
  }
}

// Sampled coarse histogram -> filter threshold at the scaled sample rank 2 hi,
// window range [base_key, top_key] with top_key above the scaled rank lo / 8
// (keys outside clamp into the end bins; the filter stage rebuilds the range when
// ranks lo..hi fall outside).
SGL_DEVICE Bounds sample_bounds(
    Smem& smem,
    const float4 (&sample)[kSampleLoads],
    uint32_t sample_mask,
    uint32_t sample_size,
    bool exchange,
    bool take_all,
    uint32_t rank_bound,
    uint32_t rank_top,
    uint32_t tid) {
  const uint32_t hist = ptx::to_shared(smem.hist_sample);
#pragma unroll
  for (uint32_t j = 0; j < kSampleLoads; ++j) {
    if (exchange && (sample_mask >> j & 1u)) {
      ptx::red_shared_add1(hist + 4 * coarse_bin(canonical(sample[j].x)));
      ptx::red_shared_add1(hist + 4 * coarse_bin(canonical(sample[j].y)));
      ptx::red_shared_add1(hist + 4 * coarse_bin(canonical(sample[j].z)));
      ptx::red_shared_add1(hist + 4 * coarse_bin(canonical(sample[j].w)));
    }
  }
  __syncthreads();
  Bounds bounds;
  bounds.threshold_key = 0;
  bounds.top_key = 0xFFFFFFFFu;
  bounds.base_key = 0;
  if (exchange && sample_size > 0) {
    const uint32_t ranks[2] = {rank_top, rank_bound};
    cta_find<kCoarseBins>(smem.hist_sample, ranks, smem, tid);
    bounds.base_key = coarse_lo_key(smem.found.hit[1].bin);
    bounds.threshold_key = take_all ? 0u : bounds.base_key;
    bounds.top_key = coarse_hi_key(smem.found.hit[0].bin);
  }
  return bounds;
}

// One filter pass over the row: candidates = keys >= threshold (every key when
// the threshold is -inf), then their keys and window bins (kept in cand[].x above kPosBits) and the
// window histogram. Returns the candidate count (may exceed kCandCap).
SGL_DEVICE uint32_t filter_pass(
    const Row& row,
    float4 (&vec)[kBatchVecs],
    uint32_t& vec_batch,
    Bounds& b,
    bool take_all,
    uint32_t edge_pos,
    float edge_score,
    Smem& smem,
    uint32_t tid) {
  if (b.top_key == 0xFFFFFFFFu) b.top_key = 0xFF800000u;  // no sample (fewer than 8 keys): up to +inf
  b.base_key = max(b.base_key, b.threshold_key);
  const WinMap map = make_win_map(b.base_key, b.top_key);
  constexpr uint32_t kBatchItems = kBatchVecs * kThreads;
  const uint32_t batches = max(1u, (row.body_vecs + kBatchItems - 1) / kBatchItems);
  for (uint32_t batch = 0; batch < batches; ++batch) {
    // rows of more than one batch (or a retune of one)
    if (batch != vec_batch) {
      load_batch(row, vec, batch, tid);
      vec_batch = batch;
    }
    filter_batch(
        row,
        vec,
        batch * kBatchItems,
        smem,
        tid,
        b.threshold,
        take_all || b.threshold == -INFINITY,
        batch == 0 ? edge_pos : ~0u,
        edge_score);
  }
  __syncthreads();
  const uint32_t m = smem.cand_count;
  // two candidates per thread per step (loads issued together)
  const uint32_t stored = min(m, kCandCap);
  for (uint32_t i = tid; i < stored; i += 2 * kThreads) {
    const uint32_t i2 = i + kThreads;
    const bool has2 = i2 < stored;
    const uint32_t pos1 = smem.cand[i].x, pos2 = smem.cand[has2 ? i2 : i].x;
    const uint32_t k1 = to_key(canonical(__ldg(row.scores + pos1)));
    const uint32_t k2 = to_key(canonical(__ldg(row.scores + pos2)));
    const uint32_t b1 = map.bin(k1), b2 = map.bin(k2);
    smem.cand[i] = make_uint2(cand_word(pos1, b1), k1);
    atomicAdd(&smem.hist_win[b1], 1u);
    if (has2) {
      smem.cand[i2] = make_uint2(cand_word(pos2, b2), k2);
      atomicAdd(&smem.hist_win[b2], 1u);
    }
  }
  __syncthreads();
  return m;
}

// Filter the row into candidates. Rare: threshold too high (m < hi while keys
// were left out) or too low (m > kCandCap): exact coarse histogram of the row,
// new bound at local rank 2 hi (or just above that bin when the bin would
// overflow the buffer). One filter instance in the code (a loop around it, not a
// second inlined copy).
SGL_DEVICE uint32_t filter_candidates(
    const Row& row,
    float4 (&vec)[kBatchVecs],
    uint32_t& vec_batch,
    Bounds b,
    bool take_all,
    uint32_t edge_pos,
    float edge_score,
    uint32_t lo,
    uint32_t hi,
    Smem& smem,
    uint32_t tid) {
  const uint32_t n = row.n;
  const uint32_t hi_eff = min(hi, n);
  uint32_t m = 0;
  for (uint32_t tries = 0;; ++tries) {
    m = filter_pass(row, vec, vec_batch, b, take_all, edge_pos, edge_score, smem, tid);
    if (!(tries < 3 && ((m < hi_eff && b.threshold > -INFINITY) || m > kCandCap))) break;
    const uint32_t t = min(n, 2 * hi);
    retune_bound(row, tid, t, max(1u, min(t, lo / 4)));
    set_threshold(b, smem.bcast.range_lo);
    b.base_key = b.threshold_key;
    b.top_key = smem.bcast.range_hi;
  }
  // rare: ranks lo / hi fall into a clamped end bin (the sample misjudged the
  // range): exact coarse histogram of the candidates, range from the bins of
  // local ranks lo / 4 and min(m, 2 hi), window histogram again
  if (m <= kCandCap && m > kSlotKeys) {
    const uint32_t lo_t = min(lo, m), hi_t = min(hi, m);
    if (smem.hist_win[kWinBins - 1] >= lo_t || m - smem.hist_win[0] < hi_t) {
      rerange(m, lo_t, min(m, 2 * hi), b.threshold_key, tid);
    }
  }
  return m;
}

// The window keys into msg[4..], ordered by bin (from the top): higher bins
// first, then a slot within the bin (the bin count is consumed); one atomic per
// key, measured faster than warp-aggregated ballot slots.
SGL_DEVICE void assemble_window(
    Smem& smem, uint32_t above, uint32_t top_bin, uint32_t bottom_bin, uint32_t cands, bool all, uint32_t tid) {
  for (uint32_t i = tid; i < cands; i += kThreads) {
    const uint2 cand = smem.cand[i];
    const uint32_t k = cand.y, bin = cand_bin(cand.x);
    if (all || (bin >= bottom_bin && bin <= top_bin)) {
      const uint32_t at =
          all ? atomicAdd(&smem.tie_claims, 1u) : smem.win_above[bin] - above + atomicSub(&smem.hist_win[bin], 1u) - 1u;
      if (at < kSlotKeys) {
        smem.msg[4 + at] = k;
        smem.win_cand[at] = i;
        smem.win_bin[at] = all ? 0u : bin;
      }
      atomicMin(&smem.win_min, k);
      atomicMax(&smem.win_max, k);
    }
  }
}

// Multicast the message into this row's slot on every rank: the header vector by
// thread 0, the key vectors by threads 1..47.
SGL_DEVICE void
push_window(Smem& smem, uint32_t* slot, uint32_t above, uint32_t size, uint32_t flags, uint32_t n, uint32_t tid) {
  if (tid == 0) {
    if (above == 0) flags |= kFlagTop;
    if (above + size == n) flags |= kFlagBottom;
    const uint4 header = make_uint4(above + 1, (size + 1) | flags, size ? smem.win_min : 1u, size ? smem.win_max : 1u);
    *reinterpret_cast<uint4*>(smem.msg) = header;
    ptx::multimem_st_relaxed_v4(slot, header);
  } else if (tid < kSlotVecs) {
    ptx::multimem_st_relaxed_v4(slot + 4 * tid, reinterpret_cast<const uint4*>(smem.msg)[tid]);
  }
}

// While the messages fly: -1 beyond A + W (the largest count this rank emits
// when the certificate holds), the keys above the window at [0, A) and the window
// keys in descending order at [A, A + W). After the poll the selection is a prefix
// of this, so only the count and the -1 tail remain.
SGL_DEVICE void early_emit(
    Smem& smem,
    uint32_t above,
    uint32_t size,
    uint32_t top_bin,
    uint32_t cands,
    bool all,
    int32_t* out_local,
    int32_t* out_slots,
    uint32_t me,
    uint32_t tid,
    uint32_t lane_id) {
  // 16 B stores after the unaligned head
  const uint32_t pad_first = above + size, pad_vec = min((pad_first + 3) & ~3u, kTopK);
  if (tid < pad_vec - pad_first) {
    out_local[pad_first + tid] = -1;
    out_slots[pad_first + tid] = -1;
  }
  for (uint32_t v = pad_vec / 4 + tid; v < kTopK / 4; v += kThreads) {
    reinterpret_cast<int4*>(out_local)[v] = make_int4(-1, -1, -1, -1);
    reinterpret_cast<int4*>(out_slots)[v] = make_int4(-1, -1, -1, -1);
  }
  // the keys above the window are > t* whenever the certificate holds (a failed
  // one re-emits from the row)
  for (uint32_t i0 = 0; i0 < cands; i0 += kThreads) {  // warp-uniform trip count
    const uint32_t i = i0 + tid;
    const bool in = i < cands;
    const uint2 cand = smem.cand[in ? i : 0];
    const bool sel = in && !all && cand_bin(cand.x) > top_bin;
    const uint32_t ballot = __ballot_sync(0xffffffffu, sel);
    uint32_t base = 0;
    if (lane_id == 0 && ballot) base = atomicAdd(&smem.out_count, __popc(ballot));
    const uint32_t at = __shfl_sync(0xffffffffu, base, 0) + __popc(ballot & ((1u << lane_id) - 1));
    const int32_t slot_id = local_slot(smem, cand_pos(cand.x));
    const bool store = sel && at < kTopK;
    ptx::st_global_if(out_local + at, slot_id, store);
    ptx::st_global_if(out_slots + at, slot_id * static_cast<int32_t>(kRanks) + static_cast<int32_t>(me), store);
  }
  // window key j: rank = its bin's offset + its rank within the bin (ties by index)
  if (tid < size) {
    const uint32_t j = tid, k = smem.msg[4 + j], bin = smem.win_bin[j];
    const uint32_t bin_first = all ? 0u : smem.win_above[bin] - above;
    uint32_t rank = bin_first;
    for (uint32_t i = bin_first; i < size && smem.win_bin[i] == bin; ++i) {
      const uint32_t v = smem.msg[4 + i];
      rank += (v > k) | (v == k && i < j);
    }
    const int32_t slot_id = local_slot(smem, cand_pos(smem.cand[smem.win_cand[j]].x));
    out_local[above + rank] = slot_id;
    out_slots[above + rank] = slot_id * static_cast<int32_t>(kRanks) + static_cast<int32_t>(me);
  }
}

// The window of local ranks [lo, hi] from the window histogram (every thread
// derives it from the search result), then assembly, push and early emit. The
// window derivation stays inline here: as a separate function it compiles to
// different predicate code.
SGL_DEVICE void send_window(
    const Params& p,
    Smem& smem,
    uint32_t row_id,
    uint32_t parity,
    uint32_t m,
    uint32_t n,
    uint32_t lo,
    uint32_t hi,
    uint32_t tid,
    uint32_t lane_id) {
  const bool bad = m > kCandCap;
  const bool all = m == n && m <= kSlotKeys;
  uint32_t above = 0;  // A, W, the message flags and the window's bin range
  uint32_t size = bad ? 0 : m;
  uint32_t flags = bad ? kFlagBad : 0;
  uint32_t top_bin = kWinBins - 1;
  uint32_t bottom_bin = 0;
  // no candidates (retune gave up): the window would not cover the row
  if (!bad && !all && m == 0) flags = kFlagBad;
  if (!bad && !all && m > 0) {
    const uint32_t lo_t = min(lo, m), hi_t = min(hi, m);
    const uint32_t ranks[2] = {lo_t, hi_t};
    if (tid < kWorkers) window_find<2>(smem.hist_win, ranks, smem, tid);
    __syncthreads();
    top_bin = smem.found.hit[0].bin;
    bottom_bin = smem.found.hit[1].bin;
    above = smem.found.hit[0].above;
    // hist_win[] is consumed by the assembly
    size = smem.found.hit[1].above + smem.found.count[1] - smem.found.hit[0].above;
    if (size > kSlotKeys) {  // rare
      if (tid == 0) {
        smem.bcast.window.above = above;
        smem.bcast.window.size = size;
        smem.bcast.window.top_bin = top_bin;
        smem.bcast.window.bottom_bin = bottom_bin;
        trim_window(lo, hi);
      }
      __syncthreads();
      above = smem.bcast.window.above;
      size = smem.bcast.window.size;
      top_bin = smem.bcast.window.top_bin;
      bottom_bin = smem.bcast.window.bottom_bin;
    }
    flags = size == 0 ? kFlagBad : 0;
  }
  // read after the emit (padding range)
  if (tid == 0) {
    smem.bcast.window.above = above;
    smem.bcast.window.size = size;
  }
  // decision histogram (published by the assembly barrier)
  if (tid < kDecWords / 4) reinterpret_cast<uint4*>(smem.hist_sample)[tid] = make_uint4(0, 0, 0, 0);
  const uint32_t cands = bad ? 0 : m;
  assemble_window(smem, above, top_bin, bottom_bin, cands, all, tid);
  __syncthreads();
  push_window(smem, row_slots(p.msg_mc, p, parity, row_id) + p.rank * kSlotWords, above, size, flags, n, tid);
  early_emit(
      smem,
      above,
      size,
      top_bin,
      cands,
      all,
      p.out_local + row_id * kTopK,
      p.out_slots + row_id * kTopK,
      p.rank,
      tid,
      lane_id);
}

// Thread t < 384 spins on vector t % 48 of rank t / 48's slot; as it lands, its
// keys go to the decision histogram (512 padded bins linear in the score) and,
// with their bins, to shared memory (hist_coarse is free until the fallback).
SGL_DEVICE void poll_messages(Smem& smem, const uint32_t* slots, uint32_t tid) {
  uint32_t* const landed_keys = smem.hist_coarse;
  uint32_t* const landed_bins = smem.hist_coarse + 4 * kPollThreads;
  {
    float dec_base = 0.f, dec_scale = 0.f;
    {
      const float vmin = key_to_float(smem.win_min), vmax = key_to_float(smem.win_max);
      const float width = vmax - vmin;
      if (smem.bcast.window.size && isfinite(vmin) && isfinite(vmax) && width > 0.f) {
        dec_base = vmin - 0.5f * width;
        dec_scale = 256.f / width;
      }
    }
    if (tid < kPollThreads) {
      const uint32_t src_rank = tid / kSlotVecs, vec_idx = tid % kSlotVecs;
      const uint32_t* src = slots + src_rank * kSlotWords + 4 * vec_idx;
      uint4 x;
      do {
        ptx::ld_relaxed_16B(x, src, 0);
      } while (!(x.x && x.y && x.z && x.w));
      if (vec_idx == 0) {
        *reinterpret_cast<uint4*>(smem.headers[src_rank]) = x;
      } else {
        const uint32_t hist = ptx::to_shared(smem.hist_sample);
        const uint32_t keys[4] = {x.x, x.y, x.z, x.w};
        uint32_t bins[4];
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
          // branch-free clamp (NaN and v <= 0 -> 0)
          const float v = (key_to_float(keys[i]) - dec_base) * dec_scale;
          bins[i] = static_cast<uint32_t>(fminf(fmaxf(v, 0.f), kDecBins - 1.f));
          ptx::red_shared_add1_if(hist + 4 * dec_index(bins[i]), keys[i] != 1u);
          bins[i] = keys[i] != 1u ? bins[i] : 0xFFFFFFFFu;  // fillers match no bin
        }
        reinterpret_cast<uint4*>(landed_keys)[tid] = x;
        reinterpret_cast<uint4*>(landed_bins)[tid] = make_uint4(bins[0], bins[1], bins[2], bins[3]);
      }
    }
  }
}

// Workers only: the fast decision. Every worker warp decodes the headers and
// searches the t* bin on its own; the keys of that bin are gathered with their
// rank, then every worker warp resolves t* and the tie quota and checks the
// certificate.
SGL_DEVICE Verdict fast_decision(Smem& smem, uint32_t me, Rule& rule, uint32_t tid, uint32_t lane_id) {
  constexpr uint32_t kWorkerVecs = kPollThreads / kWorkers;
  static_assert(kPollThreads % kWorkers == 0);
  uint32_t* const landed_keys = smem.hist_coarse;
  uint32_t* const landed_bins = smem.hist_coarse + 4 * kPollThreads;
  // header decode by every worker warp (lane q < 8 = rank q)
  bool in_range;
  int union_rank;
  {
    const uint32_t q = lane_id % kRanks;
    const uint4 h = *reinterpret_cast<const uint4*>(smem.headers[q]);
    const uint32_t above_q = h.x - 1, size_q = (h.y & 0xFFFFu) - 1, flags_q = h.y & kFlagsMask;
    const bool lead = lane_id < kRanks;
    const uint32_t above = __reduce_add_sync(0xffffffffu, lead ? above_q : 0);
    const uint32_t size = __reduce_add_sync(0xffffffffu, lead ? size_q : 0);
    const uint32_t union_min = __reduce_min_sync(0xffffffffu, lead && size_q ? h.z : 0xFFFFFFFFu);
    const uint32_t union_max = __reduce_max_sync(0xffffffffu, lead && size_q ? h.w : 0u);
    const uint32_t any_bad = __reduce_or_sync(0xffffffffu, lead ? flags_q & kFlagBad : 0);
    union_rank = static_cast<int>(kTopK) - static_cast<int>(above);
    in_range = !any_bad && union_rank >= 1 && union_rank <= static_cast<int>(size);
    if (tid == 0) {
      smem.bcast.union_min = union_min;
      smem.bcast.union_max = union_max;
      smem.bcast.union_rank = static_cast<uint32_t>(union_rank);
    }
  }
  // the t* bin: every worker warp searches the decision histogram on its own
  uint32_t tstar_bin = 0, above = 0;
  bool fast = false, certified = false;
  if (in_range) {
    const uint32_t ranks[1] = {static_cast<uint32_t>(union_rank)};
    uint32_t bin[1], bin_above[1], bin_count[1];
    warp_find<1>(smem.hist_sample, ranks, bin, bin_above, bin_count, lane_id);
    tstar_bin = bin[0];
    above = bin_above[0];
    fast = tstar_bin != 0 && tstar_bin != kDecBins - 1 && bin_count[0] <= kTStarCap;
  }
  if (fast) {
    // gather the keys of the t* bin with their rank: worker t looks at vectors
    // t + 128 j; one warp scan and one shared atomic per warp claim the slots
    uint4 keys[kWorkerVecs];
    uint32_t hit_mask = 0;
#pragma unroll
    for (uint32_t j = 0; j < kWorkerVecs; ++j) {
      const uint32_t v = tid + j * kWorkers;
      const uint4 b = reinterpret_cast<const uint4*>(landed_bins)[v];
      keys[j] = reinterpret_cast<const uint4*>(landed_keys)[v];
      const bool ok = v % kSlotVecs != 0;  // the header vector holds no keys
      const uint32_t quad = (ok && b.x == tstar_bin) | (ok && b.y == tstar_bin) << 1 | (ok && b.z == tstar_bin) << 2 |
                            (ok && b.w == tstar_bin) << 3;
      hit_mask |= quad << (4 * j);
    }
    const uint32_t hits = __popc(hit_mask);
    const uint32_t incl = warp::inclusive_sum(hits, lane_id);
    uint32_t base = 0;
    if (lane_id == 31 && incl) base = atomicAdd(&smem.tstar_count, incl);
    uint32_t slot = __shfl_sync(0xffffffffu, base, 31) + incl - hits;
#pragma unroll
    for (uint32_t j = 0; j < kWorkerVecs; ++j) {
      const uint32_t src_rank = (tid + j * kWorkers) / kSlotVecs;
      const uint32_t k4[4] = {keys[j].x, keys[j].y, keys[j].z, keys[j].w};
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        const bool hit = (hit_mask >> (4 * j + i)) & 1u;
        if (hit && slot < kTStarCap) {
          smem.tstar_keys[slot] = k4[i];
          smem.tstar_ranks[slot] = src_rank;
        }
        slot += hit;
      }
    }
    ptx::bar_sync(BAR_GATHER, kWorkers);
    const uint32_t gathered = smem.tstar_count;
    const uint32_t need = static_cast<uint32_t>(union_rank) - above;  // rank of t* within the bin, from the top
    // every worker warp resolves t* on its own: t* is the gathered key with
    // gt < need <= ge; the ties at t* of the ranks before this one and of this one
    uint32_t tstar = 0, need_ties = 0, ties_before = 0, ties_mine = 0;
    for (uint32_t j0 = 0; j0 < gathered; j0 += 32) {
      const uint32_t j = j0 + lane_id;
      const uint32_t k = smem.tstar_keys[j], src_rank = smem.tstar_ranks[j];
      uint32_t gt = 0, ge = 0;  // tstar_keys is zeroed beyond the gathered keys (0 is below every key)
#pragma unroll 1
      for (uint32_t i = 0; i < gathered; i += 4) {
        const uint4 v = *reinterpret_cast<const uint4*>(smem.tstar_keys + i);
        gt += (v.x > k) + (v.y > k) + (v.z > k) + (v.w > k);
        ge += (v.x >= k) + (v.y >= k) + (v.z >= k) + (v.w >= k);
      }
      const bool is_tstar = j < gathered && gt < need && need <= ge;
      const uint32_t ballot = __ballot_sync(0xffffffffu, is_tstar);
      if (ballot) {
        const uint32_t w = __ffs(ballot) - 1;
        tstar = __shfl_sync(0xffffffffu, k, w);
        need_ties = need - __shfl_sync(0xffffffffu, gt, w);
      }
      ties_before += __popc(__ballot_sync(0xffffffffu, is_tstar && src_rank < me));
      ties_mine += __popc(__ballot_sync(0xffffffffu, is_tstar && src_rank == me));
    }
    // certificate: t* inside every window (an open end when a window reaches the
    // top or the bottom of its row); no short circuit
    bool ok = true;
#pragma unroll
    for (uint32_t q = 0; q < kRanks; ++q) {
      const uint4 h = *reinterpret_cast<const uint4*>(smem.headers[q]);
      const uint32_t flags = h.y & kFlagsMask;
      ok &= ((flags & kFlagBottom) != 0 | h.z <= tstar) & ((flags & kFlagTop) != 0 | tstar <= h.w);
    }
    rule.tstar = tstar;
    rule.quota = need_ties > ties_before ? min(ties_mine, need_ties - ties_before) : 0u;
    certified = ok;
  }
  return fast ? (certified ? CERTIFIED : FALLBACK) : (in_range ? SLOW : FALLBACK);
}

// Rare, all 512 threads: the slow decision and its certificate, and the exact
// fallback when either certificate fails; both re-read the landed messages.
SGL_DEVICE void rare_decision(
    const Params& p,
    const Row& row,
    uint32_t row_id,
    const uint32_t* slots,
    uint32_t me,
    Verdict verdict,
    Smem& smem,
    Rule& rule,
    uint32_t tid) {
  const uint32_t src_rank = tid / kSlotVecs, vec_idx = tid % kSlotVecs;
  uint4 x = make_uint4(0, 0, 0, 0);
  if (tid < kPollThreads) ptx::ld_relaxed_16B(x, slots + src_rank * kSlotWords + 4 * vec_idx, 0);
  bool ok = false;
  if (verdict == SLOW) {
    slow_decision(x, src_rank, vec_idx, tid);
    const uint32_t tstar = smem.bcast.slow_tstar, need = smem.bcast.slow_need;
    uint32_t before = 0, mine = 0;
    ok = true;
#pragma unroll
    for (uint32_t q = 0; q < kRanks; ++q) {
      const uint4 h = *reinterpret_cast<const uint4*>(smem.headers[q]);
      const uint32_t flags = h.y & kFlagsMask, ties = smem.rank_ties[q];
      ok &= ((flags & kFlagBottom) != 0 | h.z <= tstar) & ((flags & kFlagTop) != 0 | tstar <= h.w);
      before += q < me ? ties : 0u;
      mine = q == me ? ties : mine;
    }
    rule.tstar = tstar;
    rule.quota = need > before ? min(mine, need - before) : 0u;
  }
  if (!ok) {
    exact_fallback(p, row, row_id, me, tid);
    rule.tstar = smem.bcast.fallback_tstar;
    rule.quota = smem.bcast.fallback_quota;
    rule.from_row = true;
  }
}

// Poll and decision: the workers decide; the other warps wait at a named barrier
// and leave (returns false), or join the rare paths.
SGL_DEVICE bool decide(
    const Params& p,
    const Row& row,
    uint32_t row_id,
    const uint32_t* slots,
    Smem& smem,
    Rule& rule,
    uint32_t tid,
    uint32_t lane_id) {
  const uint32_t me = p.rank;
  Verdict verdict;
  poll_messages(smem, slots, tid);
  __syncthreads();
  if (tid < kWorkers) {
    verdict = fast_decision(smem, me, rule, tid, lane_id);
    if (tid == 0) smem.bcast.verdict = verdict;
    if (verdict == CERTIFIED) {
      ptx::bar_arrive(BAR_PARK, kThreads);  // release the parked warps
    } else {
      ptx::bar_sync(BAR_PARK, kThreads);
    }
  } else {
    ptx::bar_sync(BAR_PARK, kThreads);
    verdict = static_cast<Verdict>(smem.bcast.verdict);
    if (verdict == CERTIFIED) return false;  // nothing left for this warp
  }
  rule.from_row = false;
  if (verdict != CERTIFIED) rare_decision(p, row, row_id, slots, me, verdict, smem, rule, tid);
  return true;
}

// After the fallback (or without an exchange): emit from the row. The
// early emit, if any, is overwritten or padded.
SGL_DEVICE void emit_from_row(
    const Row& row,
    const Rule& rule,
    bool exchange,
    int32_t* out_local,
    int32_t* out_slots,
    uint32_t me,
    int32_t* out_len,
    Smem& smem,
    uint32_t tid) {
  if (tid == 0) {
    smem.tie_claims = 0;
    smem.out_count = 0;
  }
  __syncthreads();
  const RowEmitter emitter{&smem, out_local, out_slots, rule.tstar, rule.quota, me};
  emit_row(row, emitter, tid);
  __syncthreads();
  const uint32_t count = min(smem.out_count, kTopK);
  // -1 already written from here on
  const uint32_t padded = exchange ? smem.bcast.window.above + smem.bcast.window.size : kTopK;
  for (uint32_t i = count + tid; i < padded; i += kThreads) {
    out_local[i] = -1;
    out_slots[i] = -1;
  }
  if (tid == 0) *out_len = static_cast<int32_t>(count);
}

// Workers: the A keys above the window and the W window keys in
// descending order went out before the poll; the selection is their first A +
// (window keys > t*) + quota. msg beyond the W window keys holds the filler 1,
// below every t*.
SGL_DEVICE void cut_to_selection(
    const Rule& rule,
    int32_t* out_local,
    int32_t* out_slots,
    int32_t* out_len,
    Smem& smem,
    uint32_t tid,
    uint32_t lane_id) {
  const uint32_t warp_id = tid / 32;
  const uint32_t above = smem.bcast.window.above, size = smem.bcast.window.size;
  constexpr uint32_t kChunks = (kSlotKeys + 31) / 32;
  if (warp_id < kWorkers / 32) {
    uint32_t greater = 0;
#pragma unroll
    for (uint32_t c = 0; c < kChunks; ++c) {
      const uint32_t k = 32 * c + lane_id < kSlotKeys ? smem.msg[4 + 32 * c + lane_id] : 1u;
      greater += __popc(__ballot_sync(0xffffffffu, k > rule.tstar));
    }
    const uint32_t count = min(above + greater + rule.quota, kTopK);
    for (uint32_t i = count + tid; i < above + size; i += kWorkers) {
      out_local[i] = -1;
      out_slots[i] = -1;
    }
    if (tid == 0) *out_len = static_cast<int32_t>(count);
  }
}

__global__ void __launch_bounds__(kThreads, 1) fused_topk_kernel(const Params p) {
  auto& smem = get_smem();
  const uint32_t tid = threadIdx.x, lane_id = tid % 32, row_id = blockIdx.x;

  // before the PDL wait: nothing the producer writes
  // contract: context <= 1M; a longer row is cut to its first kMaxPages pages
  const auto n = min(static_cast<uint32_t>(p.local_lens[row_id]), kMaxPages * kPageSize);
  const auto global_len = static_cast<uint32_t>(p.global_lens[row_id]);
  const Row row = make_row(p.logits + row_id * p.logits_stride, n);
  init_smem(smem, tid);
  const uint32_t pages = (n + kPageSize - 1) / kPageSize;

  PDLWaitPrimary<true>();
  const SampleGeometry geometry = sample_geometry(row);
  float4 sample[kSampleLoads];
  uint32_t sample_mask = 0;
#pragma unroll
  for (uint32_t j = 0; j < kSampleLoads; ++j) {
    sample[j] = make_float4(0.f, 0.f, 0.f, 0.f);
    if (load_sample(row, geometry, tid, j, sample[j])) sample_mask |= 1u << j;
  }
  // the row body into registers, all loads issued at once (consumed after the
  // filter bound is known; the sample is consumed first)
  float4 vec[kBatchVecs];
  load_batch(row, vec, 0, tid);
  uint32_t vec_batch = 0;
  // sample ranks of the filter bound and the window top, computed while the
  // sample loads are in flight
  const uint32_t lo = p.lo, hi = p.hi;
  const uint32_t samples = sample_size(geometry);
  const float scale = static_cast<float>(samples) / static_cast<float>(n);
  const uint32_t rank_bound = min(samples, static_cast<uint32_t>(ceilf(2.0f * hi * scale)));
  const uint32_t rank_top = max(1u, min(rank_bound, static_cast<uint32_t>(0.125f * lo * scale)));
  // early loads, consumed later: call index (parity), head / tail keys (the
  // lanes of find_edge, written out: the helper's reference output compiles to a
  // branch here)
  const int32_t call = p.calls[row_id];
  uint32_t edge_pos = ~0u;
  float edge_score = 0.f;
  if (tid >= kEdgeWarpFirst) {
    const uint32_t lane = tid - kEdgeWarpFirst, tail = row.head + 4 * row.body_vecs;
    if (lane < row.head) {
      edge_pos = lane;
    } else if (lane >= 4 && lane < 8 && tail + (lane - 4) < n) {
      edge_pos = tail + (lane - 4);
    }
    if (edge_pos != ~0u) edge_score = __ldg(row.scores + edge_pos);
  }
  // L <= k: every rank keeps all its keys, no exchange (all ranks agree on L)
  const bool exchange = global_len > kTopK;
  const bool take_all = !exchange || n <= 2 * hi;

  Bounds bounds = sample_bounds(smem, sample, sample_mask, samples, exchange, take_all, rank_bound, rank_top, tid);
  // page ids: first needed by the early emit, so their load hides behind the filter
  constexpr uint32_t kPageLoads = kMaxPages / kThreads;
  static_assert(kMaxPages % kThreads == 0);
  const int32_t* const page_row = p.page_table + row_id * p.page_table_stride;
  int32_t page_id[kPageLoads];
#pragma unroll
  for (uint32_t j = 0; j < kPageLoads; ++j) {
    page_id[j] = tid + j * kThreads < pages ? __ldg(page_row + tid + j * kThreads) : 0;
  }

  set_threshold(bounds, bounds.threshold_key);
  const uint32_t m =
      exchange ? filter_candidates(row, vec, vec_batch, bounds, take_all, edge_pos, edge_score, lo, hi, smem, tid) : 0;
  // published by the barrier ahead of the first lookup (window assembly, row emit)
#pragma unroll
  for (uint32_t j = 0; j < kPageLoads; ++j) {
    if (tid + j * kThreads < pages) page_ids(smem)[tid + j * kThreads] = page_id[j];
  }

  const uint32_t parity = static_cast<uint32_t>(call) & 1u;
  const uint32_t* const slots = row_slots(p.msg_buf, p, parity, row_id);
  if (exchange) {
    // zero this row's previous parity (the previous exchange, complete on this rank)
    if (tid >= kThreads - kPollThreads) {
      uint32_t* const prev = row_slots(p.msg_buf, p, parity ^ 1u, row_id);
      reinterpret_cast<uint4*>(prev)[tid - (kThreads - kPollThreads)] = make_uint4(0, 0, 0, 0);
    }
    send_window(p, smem, row_id, parity, m, n, lo, hi, tid, lane_id);
    if (tid == 0) p.calls[row_id] = call + 1;
  }

  Rule rule{0, 0, true};  // L <= k: every key
  if (exchange && !decide(p, row, row_id, slots, smem, rule, tid, lane_id)) return;

  int32_t* const out_local = p.out_local + row_id * kTopK;
  int32_t* const out_slots = p.out_slots + row_id * kTopK;
  if (rule.from_row) {
    emit_from_row(row, rule, exchange, out_local, out_slots, p.rank, p.out_len + row_id, smem, tid);
  } else {
    cut_to_selection(rule, out_local, out_slots, p.out_len + row_id, smem, tid, lane_id);
  }
  PDLTriggerSecondary<true>();
}

constexpr size_t kSmemBase = (sizeof(Smem) + 127) / 128 * 128;

}  // namespace device::dcp_fused_topk

struct DcpFusedTopKKernel {
  static constexpr auto kernel = device::dcp_fused_topk::fused_topk_kernel;

  // One batch of decode rows, the same rows in the same order on every rank; the
  // argument contract is in dcp_fused_topk.py.
  static void
  run(tvm::ffi::TensorView logits,
      tvm::ffi::TensorView local_lens,
      tvm::ffi::TensorView global_lens,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out_local,
      tvm::ffi::TensorView out_slots,
      tvm::ffi::TensorView out_len,
      tvm::ffi::TensorView buffer,
      int64_t mc_buffer,
      tvm::ffi::TensorView counters,
      tvm::ffi::TensorView fallbacks,
      int64_t rank,
      int64_t lo,
      int64_t hi) {
    using namespace host;
    using namespace device::dcp_fused_topk;

    auto R = SymbolicSize{"rows"};
    auto M = SymbolicSize{"max_rows"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({R, -1})  //
        .with_strides({-1, 1})
        .with_dtype<fp32_t>()
        .with_device(device_)
        .verify(logits);
    TensorMatcher({R})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(local_lens)
        .verify(global_lens);
    TensorMatcher({-1, -1})  //
        .with_strides({-1, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(page_table);
    TensorMatcher({R, kTopK})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .ensure_alignment(16)  // 16 B stores
        .verify(out_local)
        .verify(out_slots);
    TensorMatcher({R})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(out_len);
    TensorMatcher({-1})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(buffer);
    TensorMatcher({M})  //
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(counters)
        .verify(fallbacks);

    const auto rows = static_cast<uint32_t>(R.unwrap());
    const auto max_rows = static_cast<uint32_t>(M.unwrap());
    const auto device_id = device_.unwrap().device_id;
    // every CTA of every rank must be resident at once: one CTA per SM
    CHECK_HOST(rows <= max_rows && rows <= runtime::get_sm_count(device_id)) << "dcp_fused_topk: rows " << rows;
    CHECK_HOST(page_table.size(0) >= rows) << "dcp_fused_topk: page_table has " << page_table.size(0) << " rows";
    CHECK_HOST(0 <= rank && rank < kRanks) << "dcp_fused_topk: rank " << rank;
    CHECK_HOST(buffer.size(0) >= buffer_words(max_rows)) << "dcp_fused_topk: buffer too small";
    // trim_window drops a top bin only while above + size < lo + hi, so the keys
    // that go out stay below 1.5 (lo + hi) + kSlotKeys
    CHECK_HOST(1 <= lo && lo <= hi && hi - lo + 1 <= kSlotKeys && 3 * (lo + hi) / 2 + kSlotKeys <= kTopK)
        << "dcp_fused_topk: window [" << lo << ", " << hi << "]";
    if (rows == 0) return;

    const size_t smem = kSmemBase;
    runtime::set_max_dynamic_smem_per_device<kernel>(device_id, smem);
    const auto base = static_cast<uint32_t*>(buffer.data_ptr());
    const auto mc = reinterpret_cast<uint32_t*>(mc_buffer);
    const int64_t msg_words = 2LL * max_rows * kRanks * kSlotWords;
    const int64_t hist_words = 2LL * max_rows * kRanks * kFallbackWords;
    const auto params = Params{
        .logits = static_cast<const float*>(logits.data_ptr()),
        .logits_stride = logits.stride(0),
        .local_lens = static_cast<const int32_t*>(local_lens.data_ptr()),
        .global_lens = static_cast<const int32_t*>(global_lens.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .page_table_stride = page_table.stride(0),
        .out_local = static_cast<int32_t*>(out_local.data_ptr()),
        .out_slots = static_cast<int32_t*>(out_slots.data_ptr()),
        .out_len = static_cast<int32_t*>(out_len.data_ptr()),
        .msg_buf = base,
        .msg_mc = mc,
        .hist_buf = base + msg_words,
        .hist_mc = mc + msg_words,
        .arrive_buf = base + msg_words + hist_words,
        .arrive_mc = mc + msg_words + hist_words,
        .calls = static_cast<int32_t*>(counters.data_ptr()),
        .fallbacks = static_cast<int32_t*>(fallbacks.data_ptr()),
        .rank = static_cast<uint32_t>(rank),
        .max_rows = max_rows,
        .lo = static_cast<uint32_t>(lo),
        .hi = static_cast<uint32_t>(hi),
    };
    LaunchKernel(rows, kThreads, device_.unwrap(), smem).enable_pdl(true)(kernel, params);
  }

  // int32 words of the symmetric buffer: message slots, fallback histograms and arrival counters.
  static int64_t buffer_words(int64_t max_rows) {
    using namespace device::dcp_fused_topk;
    return 2 * max_rows * kRanks * (kSlotWords + kFallbackWords) + max_rows * 32;
  }
};

}  // namespace sglang
