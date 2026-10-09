// SPDX-License-Identifier: Apache-2.0
// Exact top-K decode selection from the 1024-bin score histogram (bin 0 = largest scores) of DeepGEMM's paged MQA
// logits kernels: Fp32Top2048 for the DSA indexers (FP8 logits), Bf16Top512 for DeepSeek-V4.1 (BF16 MXFP4 logits).
// The bin where the running count reaches K splits a row: the scores above it are selected outright, the bin itself
// is ranked exactly. Long rows are cut into parts (one CTA each) that reserve output and candidate positions with one
// atomic per row; the part arriving last ranks the candidates. Rows up to Cfg::kLocalScores stay on one CTA.
// Any count mismatch falls back to an exact radix select of the row, so the result never depends on the histogram.
// The histogram, the hand-off words and the candidate buffers are left zeroed.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace litetopk {

constexpr uint32_t kCoarseBins = 1024;
constexpr uint32_t kVectors = 4;          // 16-byte loads per thread per tile
constexpr uint32_t kSlicePages = 2048;    // a segment's page-table slice is copied to shared memory when it fits
constexpr uint32_t kMaxCount = 1u << 21;  // above any live row length: bounds the histogram sums

// Ordered unsigned key: a < b as floats (with -0 < +0) <=> ordered_key(a) < ordered_key(b)
__device__ __forceinline__ uint32_t ordered_key(float x) {
  const uint32_t bits = __float_as_uint(x);
  return bits ^ (static_cast<uint32_t>(static_cast<int32_t>(bits) >> 31) | 0x80000000u);
}

__device__ __forceinline__ float half_bits_to_float(uint32_t h) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(h)));
}

// Smallest score of DeepGEMM's coarse bins 0..t (bin 0 holds the largest scores).
// Positive bins t = 511 - c hold the FP16-RN magnitude code c = |h| >> 6 for c < 304, and [c - 288, c - 287) above;
// negative bins t = 512 + c hold the same codes, with lower-inclusive unit bins [-(c - 287), -(c - 288)).
__device__ __forceinline__ float coarse_lower_edge(uint32_t t) {
  if (t < 512) {
    const uint32_t c = 511 - t;
    if (c == 0) return 0.0f;
    if (c <= 304)  // first score rounding to FP16 code c << 6: ties go to the even code
      return 0.5f * (half_bits_to_float(c * 64 - 1) + half_bits_to_float(c * 64));
    return static_cast<float>(c - 288);
  }
  const uint32_t c = t - 512;
  if (c == 511) return -__int_as_float(0x7f800000);
  if (c >= 304) return -static_cast<float>(c - 287);
  // Largest magnitude rounding to at most code (c << 6) | 63: just below the midpoint, whose tie rounds up
  const float mid = 0.5f * (half_bits_to_float(c * 64 + 63) + half_bits_to_float(c * 64 + 64));
  return -__uint_as_float(__float_as_uint(mid) - 1);
}

// Score loads of both configurations: plain ld.global after griddepcontrol.wait (read-only cache loads may miss the
// PDL producer's writes); vector loads stay in L1 (staging reads hits back) with evict_first. Measured (FP32):
// no-allocate slows staging, plain allocation slows long DRAM-bound rows.
struct Fp32Top2048 {
  using Score = float;
  using Vector = float4;
  static constexpr uint32_t kTopK = 2048;
  static constexpr uint32_t kPerVector = 4;
  static constexpr uint32_t kKeyBits = 32;      // ordered key width; composites hold (key, ~slot)
  static constexpr uint32_t kUnit = 256;        // partition granularity in scores (1 KiB)
  static constexpr uint32_t kMinPartUnits = 8;  // measured: parts under 2048 scores only add hand-off traffic
  static constexpr uint32_t kFineBins = 2048;
  static constexpr uint32_t kFineBits = 11;
  static constexpr uint32_t kStage = 2048;         // the crossing fine bin's list
  static constexpr uint32_t kStaged = 2 * kStage;  // hits a CTA stages per segment
  static constexpr uint32_t kRankCount = 128;      // crossing bins up to this size are ranked directly
  static constexpr uint32_t kDirectRank = 512;     // crossing fine bins up to this size are ranked directly
  static constexpr uint32_t kKeyAboveInf = 0xff800001u;
  static constexpr uint32_t kPageBits = 6;
  // Measured on B200: 5 float4 per thread keep 8K rows (the multi-part path's slowest) local; 8 make
  // select_kernel<3> spill.
  static constexpr uint32_t kLocalScores = 10240;

  static __device__ __forceinline__ float load(const float* ptr) {
    float value;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(value) : "l"(ptr) : "memory");
    return value;
  }

  static __device__ __forceinline__ float4 load_vector(const float* ptr) {
    float4 v;
    asm volatile("ld.global.L1::evict_first.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(ptr)
                 : "memory");
    return v;
  }

  static __device__ __forceinline__ float element(const float4& v, uint32_t q) {
    return q == 0 ? v.x : q == 1 ? v.y : q == 2 ? v.z : v.w;
  }

  static __device__ __forceinline__ uint32_t key(float x) {
    return ordered_key(x);
  }

  // Smallest key of a float >= e; a zero edge also admits -0.0
  static __device__ __forceinline__ uint32_t edge(float e) {
    return e == 0.0f ? 0x7fffffffu : ordered_key(e);
  }
};

// BF16 scores, top-512, 128-slot pages (DeepSeek-V4.1's index-K pool). kUnit keeps a unit (and so the minimum part,
// kWideTiles and the tile) at the FP32 byte size; kFineBins covers the 16-bit key range of every coarse bin but the
// four at +-0 and +-inf, so the fine histogram ranks exactly (shift 0).
struct Bf16Top512 {
  using Score = __nv_bfloat16;
  using Vector = uint4;
  static constexpr uint32_t kTopK = 512;
  static constexpr uint32_t kPerVector = 8;
  static constexpr uint32_t kKeyBits = 16;
  static constexpr uint32_t kUnit = 512;
  static constexpr uint32_t kMinPartUnits = 8;
  static constexpr uint32_t kFineBins = 1024;
  static constexpr uint32_t kFineBits = 10;
  static constexpr uint32_t kStage = 2048;
  static constexpr uint32_t kStaged = 2048;
  static constexpr uint32_t kRankCount = 128;
  static constexpr uint32_t kDirectRank = 512;
  static constexpr uint32_t kKeyAboveInf = 0xff81u;
  static constexpr uint32_t kPageBits = 7;
  // One 512-thread tile; local_row's 16-bit place counters hold <= 16384 scores per kind. Measured on B200:
  // 1.20-1.48x over the multi-part path for such rows.
  static constexpr uint32_t kLocalScores = 16384;

  static __device__ __forceinline__ float load(const __nv_bfloat16* ptr) {
    unsigned short bits;
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(bits) : "l"(ptr) : "memory");
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
  }

  static __device__ __forceinline__ uint4 load_vector(const __nv_bfloat16* ptr) {
    uint4 v;
    asm volatile("ld.global.L1::evict_first.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr)
                 : "memory");
    return v;
  }

  // Score q of a vector: the low half of word q / 2 holds score q = 2 (q / 2)
  static __device__ __forceinline__ float element(const uint4& v, uint32_t q) {
    const uint32_t w = (q >> 1) == 0 ? v.x : (q >> 1) == 1 ? v.y : (q >> 1) == 2 ? v.z : v.w;
    return __uint_as_float(q & 1 ? w & 0xffff0000u : w << 16);
  }

  // 16-bit ordered key of a BF16 value held in a float (its low 16 bits are zero)
  static __device__ __forceinline__ uint32_t key(float x) {
    return ordered_key(x) >> 16;
  }

  // Smallest key of a BF16 value >= e; a zero edge also admits -0.0
  static __device__ __forceinline__ uint32_t edge(float e) {
    return e == 0.0f ? 0x7fffu : ordered_key(__bfloat162float(__float2bfloat16_ru(e))) >> 16;
  }

  // Both halves = the smallest BF16 >= t, so that a BF16 x satisfies x >= t (as floats) iff x >= that value
  static __device__ __forceinline__ uint32_t threshold(float t) {
    const auto h = __bfloat16_as_ushort(__float2bfloat16_ru(t));
    return static_cast<uint32_t>(h) * 0x10001u;
  }

  // Bit q of the result: score q of the vector is >= the packed threshold
  static __device__ __forceinline__ uint32_t at_least(const uint4& v, uint32_t t2) {
    const auto t = *reinterpret_cast<const __nv_bfloat162*>(&t2);
    const uint32_t m0 = __hge2_mask(*reinterpret_cast<const __nv_bfloat162*>(&v.x), t);
    const uint32_t m1 = __hge2_mask(*reinterpret_cast<const __nv_bfloat162*>(&v.y), t);
    const uint32_t m2 = __hge2_mask(*reinterpret_cast<const __nv_bfloat162*>(&v.z), t);
    const uint32_t m3 = __hge2_mask(*reinterpret_cast<const __nv_bfloat162*>(&v.w), t);
    // One flag byte per score (0xff / 0x00), then signed dot products weight them 1, 2, 4, ... 128
    const uint32_t lo = __byte_perm(m0, m1, 0x7531), hi = __byte_perm(m2, m3, 0x7531);
    const int low = __dp4a(static_cast<int>(lo), static_cast<int>(0xf8fcfeffu), 0);
    return static_cast<uint32_t>(__dp4a(static_cast<int>(hi), static_cast<int>(0x80c0e0f0u), low));
  }
};

// Per-row hand-off; at rest `word` is zero and `done` equals `done_base`
struct alignas(16) RowState {
  unsigned long long word;  // [63:48] parts arrived, [47:24] scores above the crossing bin, [23:0] inside it
  uint32_t done;            // kPartSignal per part whose writes have landed, counted across calls
  uint32_t done_base;       // `done` at the start of the call: each finalizer advances it by the other parts' signals
};

template <typename Score>
struct Params {
  const Score* scores;
  int64_t score_stride;
  uint32_t score_width;
  const int32_t* lengths;
  int32_t* histogram;
  const int32_t* table;
  int64_t table_stride;
  uint32_t rows_per_table_row;
  int32_t* out;
  RowState* state;
  unsigned long long* candidates;
  uint32_t capacity;
  uint32_t rows;
};

// A row's split from the histogram, identical in every CTA that touches the row
struct Split {
  bool certified;
  bool whole;  // every score of the crossing bin is selected
  int32_t bin;
  uint32_t strict;
  uint32_t count;
  float lo;  // scores >= lo lie in bins 0..bin
  float hi;  // scores >= hi lie above the crossing bin (NaN for bin 0)
  uint32_t key_lo;
  uint32_t shift;         // fine bin = (ordered_key - key_lo) >> shift
  uint32_t bucket_shift;  // short rows: bucket = (ordered_key - key_lo) >> bucket_shift, one of 32
};

__device__ __forceinline__ uint32_t lane_id() {
  return threadIdx.x & 31;
}

// The shuffle's predicate marks the lanes that have a source: two instructions per step
__device__ __forceinline__ uint32_t warp_inclusive_sum(uint32_t v) {
  asm volatile(
      "{\n.reg .pred p;\n.reg .b32 t;\n"
      "shfl.sync.up.b32 t|p, %0, 1, 0, -1;\n@p add.u32 %0, %0, t;\n"
      "shfl.sync.up.b32 t|p, %0, 2, 0, -1;\n@p add.u32 %0, %0, t;\n"
      "shfl.sync.up.b32 t|p, %0, 4, 0, -1;\n@p add.u32 %0, %0, t;\n"
      "shfl.sync.up.b32 t|p, %0, 8, 0, -1;\n@p add.u32 %0, %0, t;\n"
      "shfl.sync.up.b32 t|p, %0, 16, 0, -1;\n@p add.u32 %0, %0, t;\n}"
      : "+r"(v));
  return v;
}

// Relaxed: the finalizer reads only the word itself and candidate entries it waits for
__device__ __forceinline__ unsigned long long arrive(unsigned long long* word, unsigned long long add) {
  unsigned long long old;
  asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], %2;" : "=l"(old) : "l"(word), "l"(add) : "memory");
  return old;
}

// Marks a part's writes (ordered before it by a barrier) as landed
__device__ __forceinline__ void signal_done(uint32_t* done, uint32_t weight) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(done), "r"(weight) : "memory");
}

// Every part's warps signal kPartSignal in total, whatever the width of the CTA that ran it
constexpr uint32_t kPartSignal = 32;

__device__ __forceinline__ uint32_t load_acquire(const uint32_t* ptr) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
  return v;
}

__device__ __forceinline__ unsigned long long load_relaxed(const unsigned long long* ptr) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(ptr) : "memory");
  return v;
}

// A candidate entry is never zero; its writer may still be in flight after the arrival
__device__ __forceinline__ unsigned long long await_entry(const unsigned long long* ptr, unsigned long long v) {
  while (v == 0)
    v = load_relaxed(ptr);
  return v;
}

template <typename Cfg>
__device__ __forceinline__ int32_t physical_slot(const int32_t* table_row, uint32_t i) {
  return (table_row[i >> Cfg::kPageBits] << Cfg::kPageBits) | static_cast<int32_t>(i & ((1u << Cfg::kPageBits) - 1));
}

// Ordering composite over a score's key and its physical KV slot: larger is better, ties go to the lower slot
__device__ __forceinline__ unsigned long long composite(uint32_t key, int32_t slot) {
  return (static_cast<unsigned long long>(key) << 32) | ~static_cast<uint32_t>(slot);
}

__device__ __forceinline__ int32_t composite_slot(unsigned long long v) {
  return static_cast<int32_t>(~static_cast<uint32_t>(v));
}

__device__ __forceinline__ uint32_t composite_key(unsigned long long v) {
  return static_cast<uint32_t>(v >> 32);
}

template <typename Score>
__device__ __forceinline__ uint32_t row_length(const Params<Score>& p, uint32_t row) {
  return min(static_cast<uint32_t>(max(p.lengths[row], 0)), p.score_width);
}

// Parts a row of `length` scores uses of its `slots`: none shorter than Cfg::kMinPartUnits units
template <typename Cfg>
__device__ __forceinline__ uint32_t parts_for(uint32_t slots, uint32_t length) {
  if (length <= Cfg::kLocalScores) return 1;  // Selector::local_row: the whole row on one CTA
  return min(slots, max(1u, (length + Cfg::kUnit - 1) / Cfg::kUnit / Cfg::kMinPartUnits));
}

// Score range [x, y) of part `part` of `parts`: the parts share the row's kUnit-score units evenly
template <typename Cfg>
__device__ __forceinline__ uint2 segment(uint32_t length, uint32_t part, uint32_t parts) {
  const uint32_t units = (length + Cfg::kUnit - 1) / Cfg::kUnit;
  return make_uint2(part * units / parts * Cfg::kUnit, min((part + 1) * units / parts * Cfg::kUnit, length));
}

extern __shared__ __align__(16) unsigned char dynamic_smem[];

// A 1024-thread CTA runs Selector<Cfg, 1024> or, with its upper half exited, Selector<Cfg, 512>: barriers count the
// participating threads explicitly.
template <typename Cfg, uint32_t kThreads>
struct Selector {
  using Score = typename Cfg::Score;
  using Vector = typename Cfg::Vector;
  static constexpr bool kFp32 = std::is_same_v<Score, float>;
  static constexpr uint32_t kTopK = Cfg::kTopK;
  static constexpr uint32_t kFineBins = Cfg::kFineBins;
  static constexpr uint32_t kFineBits = Cfg::kFineBits;
  static constexpr uint32_t kStage = Cfg::kStage;
  static constexpr uint32_t kStaged = Cfg::kStaged;
  static constexpr uint32_t kRankCount = Cfg::kRankCount;
  static constexpr uint32_t kPerVector = Cfg::kPerVector;
  static constexpr uint32_t kWarps = kThreads / 32;
  static constexpr uint32_t kTile = kThreads * kVectors * kPerVector;
  static constexpr uint32_t kCoarsePerThread = kCoarseBins / kThreads;  // 2 or 1
  static constexpr uint32_t kFinePerThread = kFineBins / kThreads;
  static_assert(kFineBins == 1u << kFineBits);
  static_assert(kStaged >= kTopK, "a CTA's scores above the crossing bin always fit its stage");
  static_assert(kStage <= kStaged, "the crossing fine bin's list shares the staging memory");
  static_assert(kVectors * kPerVector <= 32, "a thread's tile scores index one 32-bit mask");
  static_assert(kCoarseBins % kThreads == 0 and kFineBins % kThreads == 0 and kStaged % kThreads == 0);
  static_assert(kPartSignal % kWarps == 0);
  using P = Params<Score>;

  struct Shared {
    uint32_t scan[kWarps];
    uint32_t radix[256];
    union {
      unsigned long long staged[kStaged];  // (score bits, index) of the segment's hits
      uint32_t fine[kFineBins];            // the crossing bin's fine histogram
      unsigned long long list[kStage];     // the crossing fine bin's composites
    };
    alignas(16) int32_t pages[kSlicePages];  // the segment's page-table slice (a short row's crossing bucket, after it)
    uint32_t staged_count;                   // hits of the segment, possibly beyond kStaged
    uint32_t strict_count;                   // those of them above the crossing bin
    unsigned long long reserved;
    unsigned long long row_word;  // the finalizer's view of its row's hand-off word
    int32_t bin;
    uint32_t strict;
    uint32_t count;
    uint32_t emitted;
    uint32_t listed;
    Split local;  // a short row's split, derived by warp 0 (Selector::local_row)
  };

  static __device__ __forceinline__ Shared& sm() {
    return *reinterpret_cast<Shared*>(dynamic_smem);
  }

  static __device__ __forceinline__ void sync() {
    asm volatile("bar.sync 1, %0;" ::"n"(kThreads) : "memory");
  }

  // Exclusive prefix in thread order. Contains one barrier; `scratch` is free again after the next barrier.
  static __device__ __forceinline__ uint32_t block_exclusive_sum(uint32_t v, uint32_t* scratch, uint32_t& total) {
    const uint32_t inclusive = warp_inclusive_sum(v);
    if (lane_id() == 31) scratch[threadIdx.x >> 5] = inclusive;
    sync();
    const uint32_t warp_total = lane_id() < kWarps ? scratch[lane_id()] : 0u;
    const uint32_t warp_inclusive = warp_inclusive_sum(warp_total);
    total = __shfl_sync(0xffffffffu, warp_inclusive, kWarps - 1);
    return __shfl_sync(0xffffffffu, warp_inclusive - warp_total, threadIdx.x >> 5) + inclusive - v;
  }

  // Page-table slice of a segment: copied asynchronously to shared memory when it fits, read from global otherwise
  struct PageSlice {
    const int32_t* table_row;
    uint32_t first_page;
    bool shared;

    __device__ __forceinline__ PageSlice(const int32_t* table_row, uint32_t begin, uint32_t end)
        : table_row(table_row), first_page(begin >> Cfg::kPageBits) {
      const uint32_t count = end > begin ? ((end - 1) >> Cfg::kPageBits) - first_page + 1 : 0u;
      shared = count <= kSlicePages;
      if (shared) {
        for (uint32_t i = threadIdx.x; i < count; i += kThreads) {
          const auto dst = static_cast<uint32_t>(__cvta_generic_to_shared(&sm().pages[i]));
          asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(table_row + first_page + i)
                       : "memory");
        }
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    }

    // Call before the barrier that precedes the first slot lookup
    __device__ __forceinline__ static void wait() {
      asm volatile("cp.async.wait_all;" ::: "memory");
    }

    __device__ __forceinline__ int32_t slot(uint32_t i) const {
      const int32_t page = shared ? sm().pages[(i >> Cfg::kPageBits) - first_page] : table_row[i >> Cfg::kPageBits];
      return (page << Cfg::kPageBits) | static_cast<int32_t>(i & ((1u << Cfg::kPageBits) - 1));
    }
  };

  // The loads stay in flight until the scores are compared; vectors past `end` keep stale values `live_bits` masks
  static __device__ __forceinline__ void
  load_tile(Vector (&v)[kVectors], const Score* row, uint32_t base, uint32_t end) {
    if (base + kTile <= end) {
#pragma unroll
      for (uint32_t u = 0; u < kVectors; ++u)
        v[u] = Cfg::load_vector(row + base + (u * kThreads + threadIdx.x) * kPerVector);
      return;
    }
#pragma unroll
    for (uint32_t u = 0; u < kVectors; ++u) {
      const uint32_t first = base + (u * kThreads + threadIdx.x) * kPerVector;
      if (first < end) v[u] = Cfg::load_vector(row + first);
    }
  }

  // Bit u * kPerVector + q is set when score q of this thread's vector u lies before `end`
  static __device__ __forceinline__ uint32_t live_bits(uint32_t base, uint32_t end) {
    uint32_t bits = 0;
#pragma unroll
    for (uint32_t u = 0; u < kVectors; ++u) {
      const uint32_t first = base + (u * kThreads + threadIdx.x) * kPerVector;
      const uint32_t live = end > first ? min(end - first, kPerVector) : 0u;
      bits |= ((1u << live) - 1) << (u * kPerVector);
    }
    return bits;
  }

  // Bit u * kPerVector + q: score q of vector u is >= the BF16 threshold `t2` (Bf16Top512::threshold)
  static __device__ __forceinline__ uint32_t tile_at_least(const Vector (&v)[kVectors], uint32_t t2) {
    uint32_t bits = 0;
#pragma unroll
    for (uint32_t u = 0; u < kVectors; ++u)
      bits |= Cfg::at_least(v[u], t2) << (u * kPerVector);
    return bits;
  }

  // This thread's coarse bins (kCoarsePerThread of them, in .x then .y)
  static __device__ __forceinline__ int2 load_histogram(const P& p, uint32_t row) {
    const int32_t* bins = p.histogram + static_cast<size_t>(row) * kCoarseBins;
    if constexpr (kCoarsePerThread == 2) return __ldcg(reinterpret_cast<const int2*>(bins) + threadIdx.x);
    return make_int2(__ldcg(bins + threadIdx.x), 0);
  }

  static __device__ __forceinline__ void clear_histogram(const P& p, uint32_t row) {
    int32_t* bins = p.histogram + static_cast<size_t>(row) * kCoarseBins;
    if constexpr (kCoarsePerThread == 2) {
      reinterpret_cast<int2*>(bins)[threadIdx.x] = make_int2(0, 0);
    } else {
      bins[threadIdx.x] = 0;
    }
  }

  static __device__ Split derive_split(int2 bins, uint32_t length) {
    if (threadIdx.x == 0) sm().bin = -1;
    const uint32_t a = min(static_cast<uint32_t>(bins.x), kMaxCount);
    const uint32_t b = min(static_cast<uint32_t>(bins.y), kMaxCount);
    uint32_t total;
    const uint32_t before = block_exclusive_sum(a + b, sm().scan, total);
    // The crossing bin holds the K-th largest score: before < K <= before + count
    if (total == length) {
      if (before < kTopK and kTopK <= before + a) {
        sm().bin = kCoarsePerThread * threadIdx.x, sm().strict = before, sm().count = a;
      } else if (before + a < kTopK and kTopK <= before + a + b) {
        sm().bin = kCoarsePerThread * threadIdx.x + 1, sm().strict = before + a, sm().count = b;
      }
    }
    sync();
    Split s;
    s.bin = sm().bin;
    s.certified = s.bin >= 0;
    s.strict = sm().strict;
    s.count = sm().count;
    s.whole = s.certified and s.count == kTopK - s.strict;
    const float nan = __int_as_float(0x7fc00000);
    s.lo = s.certified ? coarse_lower_edge(s.bin) : nan;
    s.hi = s.bin > 0 ? coarse_lower_edge(s.bin - 1) : nan;
    s.key_lo = Cfg::edge(s.lo);
    const uint32_t key_hi = s.bin > 0 ? Cfg::edge(s.hi) : Cfg::kKeyAboveInf;  // just above +inf
    const uint32_t width = key_hi - s.key_lo;
    s.shift = width > kFineBins ? 32 - __clz(width - 1) - kFineBits : 0;
    return s;
  }

  // Score j of this thread's tile vectors, without indexing registers dynamically
  static __device__ __forceinline__ float pick(const Vector (&v)[kVectors], uint32_t j) {
    static_assert(kVectors == 4);
    const uint32_t u = j / kPerVector, q = (j % kPerVector) / (kPerVector / 4);  // the word holding score j
    Vector w = v[0];
    w = u == 1 ? v[1] : w;
    w = u == 2 ? v[2] : w;
    w = u == 3 ? v[3] : w;
    auto x = w.x;
    x = q == 1 ? w.y : x;
    x = q == 2 ? w.z : x;
    x = q == 3 ? w.w : x;
    if constexpr (kFp32) {
      return x;
    } else {
      return __uint_as_float(j & 1 ? x & 0xffff0000u : x << 16);
    }
  }

  // Stages a warp's hits (scores >= lo) as (score bits, index); their scores are read back from L1
  static __device__ __forceinline__ void
  stage_hits(const Score* x, uint32_t hits, uint32_t base, float hi, uint32_t& strict_mine) {
    const uint32_t mine = __popc(hits);
    const uint32_t inclusive = warp_inclusive_sum(mine);
    uint32_t first = 0;
    if (lane_id() == 31) first = atomicAdd(&sm().staged_count, inclusive);
    uint32_t pos = __shfl_sync(0xffffffffu, first, 31) + inclusive - mine;
    for (uint32_t left = hits; left != 0; left &= left - 1, ++pos) {
      const uint32_t j = __ffs(left) - 1;
      const uint32_t i = base + ((j / kPerVector) * kThreads + threadIdx.x) * kPerVector + (j % kPerVector);
      const float score = Cfg::load(x + i);
      strict_mine += score >= hi;
      if (pos < kStaged) sm().staged[pos] = (static_cast<unsigned long long>(__float_as_uint(score)) << 32) | i;
    }
  }

  // Returns whether this part arrived last. Candidates are written after the arrival (the finalizer waits for each
  // entry to turn nonzero); positions past the certified counts mean a histogram mismatch the finalizer detects.
  static __device__ bool
  publish_segment(const P& p, const Split& s, uint32_t row, uint32_t parts, int32_t* out_row, const PageSlice& pages) {
    constexpr uint32_t kPerThread = kStaged / kThreads;
    PageSlice::wait();
    sync();
    const uint32_t staged = min(sm().staged_count, kStaged);
    unsigned long long before = 0, add = 0;
    if (threadIdx.x == 0) {
      // A part that staged more hits than it holds adds K to the strict count: the finalizer then sees a mismatch
      const bool overflow = sm().staged_count > kStaged;
      add = (1ull << 48) | (static_cast<unsigned long long>(sm().strict_count + (overflow ? kTopK : 0u)) << 24) |
            (sm().staged_count - sm().strict_count);
      before = arrive(&p.state[row].word, add);  // read only after the split below: the round trip overlaps it
    }
    // Each thread splits its staged hits at the crossing bin, as staging counted them, and looks up their pages
    uint32_t live = 0, strict = 0;
    int32_t slot[kPerThread];
#pragma unroll
    for (uint32_t k = 0; k < kPerThread; ++k) {
      const uint32_t i = k * kThreads + threadIdx.x;
      if (i < staged) {
        const unsigned long long entry = sm().staged[i];
        live |= 1u << k;
        strict |= static_cast<uint32_t>(__uint_as_float(static_cast<uint32_t>(entry >> 32)) >= s.hi) << k;
        slot[k] = pages.slot(static_cast<uint32_t>(entry));
      }
    }
    uint32_t total;
    const uint32_t offset = block_exclusive_sum(__popc(strict) | (__popc(live ^ strict) << 16), sm().scan, total);
    if (threadIdx.x == 0) sm().reserved = before, sm().row_word = before + add;
    sync();
    before = sm().reserved;
    uint32_t strict_pos = (static_cast<uint32_t>(before >> 24) & 0xffffffu) + (offset & 0xffffu);
    uint32_t inside_pos = (static_cast<uint32_t>(before) & 0xffffffu) + (offset >> 16);
    const bool last = (before >> 48) + 1 == parts;
    unsigned long long* candidates = p.candidates + static_cast<size_t>(row) * p.capacity;
#pragma unroll
    for (uint32_t k = 0; k < kPerThread; ++k) {
      if (not(live >> k & 1)) continue;
      if (strict >> k & 1) {
        if (strict_pos < s.strict) out_row[strict_pos] = slot[k];
        ++strict_pos;
      } else {
        if (s.whole) {
          if (inside_pos < s.count) out_row[s.strict + inside_pos] = slot[k];
        } else if (inside_pos < p.capacity) {
          const auto bits = static_cast<uint32_t>(sm().staged[k * kThreads + threadIdx.x] >> 32);
          candidates[inside_pos] = composite(Cfg::key(__uint_as_float(bits)), slot[k]);
        }
        ++inside_pos;
      }
    }
    if (threadIdx.x == 0) sm().staged_count = 0, sm().strict_count = 0;
    // The finalizer reuses the stage's shared memory
    if (last) sync();
    // Each warp's writes land before its own completion signal
    __syncwarp();
    if (lane_id() == 0 and not last) signal_done(&p.state[row].done, kPartSignal / kWarps);
    return last;
  }

  // A one-tile part without staging: a block scan places every thread's hits. Returns whether this part arrived last.
  static __device__ bool publish_tile(
      const P& p,
      const Split& s,
      uint32_t row,
      uint32_t parts,
      int32_t* out_row,
      const PageSlice& pages,
      const Vector (&v)[kVectors],
      uint32_t base,
      uint32_t end) {
    uint32_t hits = 0, strict = 0;
    if constexpr (kFp32) {
#pragma unroll
      for (uint32_t u = 0; u < kVectors; ++u) {
#pragma unroll
        for (uint32_t q = 0; q < 4; ++q) {
          hits |= static_cast<uint32_t>(Cfg::element(v[u], q) >= s.lo) << (u * 4 + q);
          strict |= static_cast<uint32_t>(Cfg::element(v[u], q) >= s.hi) << (u * 4 + q);
        }
      }
    } else {
      hits = tile_at_least(v, Cfg::threshold(s.lo));
      strict = tile_at_least(v, Cfg::threshold(s.hi));
    }
    const uint32_t live = live_bits(base, end);
    hits &= live, strict &= live;
    // A tile holds at most kTile hits of each kind: both counts fit 16 bits
    const uint32_t mine = __popc(strict) | (__popc(hits ^ strict) << 16);
    uint32_t total;
    const uint32_t offset = block_exclusive_sum(mine, sm().scan, total);
    PageSlice::wait();
    if (threadIdx.x == 0) {
      const unsigned long long add =
          (1ull << 48) | (static_cast<unsigned long long>(total & 0xffffu) << 24) | (total >> 16);
      const unsigned long long before = arrive(&p.state[row].word, add);
      sm().reserved = before;
      sm().row_word = before + add;
    }
    sync();
    const unsigned long long before = sm().reserved;
    uint32_t strict_pos = (static_cast<uint32_t>(before >> 24) & 0xffffffu) + (offset & 0xffffu);
    uint32_t inside_pos = (static_cast<uint32_t>(before) & 0xffffffu) + (offset >> 16);
    const bool last = (before >> 48) + 1 == parts;
    const auto index = [&](uint32_t j) {
      return base + ((j / kPerVector) * kThreads + threadIdx.x) * kPerVector + (j % kPerVector);
    };
    for (uint32_t left = strict; left != 0; left &= left - 1, ++strict_pos) {
      if (strict_pos < s.strict) out_row[strict_pos] = pages.slot(index(__ffs(left) - 1));
    }
    unsigned long long* candidates = p.candidates + static_cast<size_t>(row) * p.capacity;
    for (uint32_t left = hits ^ strict; left != 0; left &= left - 1, ++inside_pos) {
      const uint32_t j = __ffs(left) - 1;
      const int32_t slot = pages.slot(index(j));
      if (s.whole) {
        if (inside_pos < s.count) out_row[s.strict + inside_pos] = slot;
      } else if (inside_pos < p.capacity) {
        candidates[inside_pos] = composite(Cfg::key(pick(v, j)), slot);
      }
    }
    __syncwarp();
    if (lane_id() == 0 and not last) signal_done(&p.state[row].done, kPartSignal / kWarps);
    return last;
  }

  // Exact top-`need` of `n` distinct composites `load(i)`, 8 bits per pass from the top of the key; their slots go to
  // out[0, need) in no particular order
  template <typename Load>
  static __device__ void radix_select(uint32_t n, uint32_t need, const Load& load, int32_t* out) {
    if (threadIdx.x == 0) sm().emitted = 0;
    sync();
    unsigned long long prefix = 0, mask = 0;
    for (int shift = 24 + Cfg::kKeyBits; shift >= 0; shift -= 8) {
      if (threadIdx.x < 256) sm().radix[threadIdx.x] = 0;
      sync();
      for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const unsigned long long v = load(i);
        if ((v & mask) == prefix) atomicAdd(&sm().radix[(v >> shift) & 255], 1u);
      }
      sync();
      // Thread t holds digit 255 - t, so the prefix runs from the best digit down
      const uint32_t count = threadIdx.x < 256 ? sm().radix[(255 - threadIdx.x) & 255] : 0u;
      uint32_t total;
      const uint32_t above = block_exclusive_sum(count, sm().scan, total);
      if (threadIdx.x < 256 and above < need and need <= above + count)
        sm().bin = 255 - threadIdx.x, sm().strict = above, sm().count = count;
      sync();
      const uint32_t digit = sm().bin, digit_above = sm().strict, digit_count = sm().count;
      const bool take = digit_count == need - digit_above;
      for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const unsigned long long v = load(i);
        if ((v & mask) == prefix) {
          const uint32_t d = (v >> shift) & 255;
          if (d > digit or (take and d == digit)) out[atomicAdd(&sm().emitted, 1u)] = composite_slot(v);
        }
      }
      if (take) return;
      need -= digit_above;
      prefix |= static_cast<unsigned long long>(digit) << shift;
      mask |= 0xffull << shift;
      sync();
    }
  }

  // Exact top-K of a whole row from its scores
  static __device__ __noinline__ void
  select_row_exact(const P& p, uint32_t row, uint32_t length, int32_t* out_row, const int32_t* table_row) {
    const Score* x = p.scores + static_cast<size_t>(row) * p.score_stride;
    const auto load = [&](uint32_t i) {
      return composite(Cfg::key(Cfg::load(x + i)), physical_slot<Cfg>(table_row, i));
    };
    radix_select(length, kTopK, load, out_row);
  }

  // Ranks the crossing bin's candidates through a fine histogram of their keys. Every candidate entry is read
  // (waiting for late writers) and returned to zero.
  static __device__ bool resolve_bin(const P& p, const Split& s, uint32_t row, int32_t* out_row) {
    constexpr uint32_t kHeld = 4;  // candidates per thread kept in registers between the two passes
    unsigned long long* candidates = p.candidates + static_cast<size_t>(row) * p.capacity;
    const uint32_t need = kTopK - s.strict;
    unsigned long long held[kHeld];
#pragma unroll
    for (uint32_t k = 0; k < kHeld; ++k) {
      const uint32_t i = k * kThreads + threadIdx.x;
      held[k] = i < s.count ? load_relaxed(candidates + i) : 1ull;
    }
    if (s.count <= kRankCount) {
      // Small bin: every candidate ranks itself against the others; composites are distinct
      if (threadIdx.x < s.count) {
        held[0] = await_entry(candidates + threadIdx.x, held[0]);
        candidates[threadIdx.x] = 0;
        sm().list[threadIdx.x] = held[0];
      }
      sync();
      if (threadIdx.x < s.count) {
        uint32_t rank = 0;
        for (uint32_t j = 0; j < s.count; ++j)
          rank += sm().list[j] > held[0];
        if (rank < need) out_row[s.strict + rank] = composite_slot(held[0]);
      }
      return true;
    }
    uint32_t* fine = sm().fine;
    for (uint32_t i = threadIdx.x; i < kFineBins; i += kThreads)
      fine[i] = 0;
    if (threadIdx.x == 0) sm().bin = -1, sm().emitted = 0, sm().listed = 0;
    sync();
    const auto fine_bin = [&](unsigned long long v) {
      return min((composite_key(v) - s.key_lo) >> s.shift, kFineBins - 1);
    };
#pragma unroll
    for (uint32_t k = 0; k < kHeld; ++k) {
      const uint32_t i = k * kThreads + threadIdx.x;
      if (i < s.count) {
        held[k] = await_entry(candidates + i, held[k]);
        candidates[i] = 0;
        atomicAdd(&fine[fine_bin(held[k])], 1u);
      }
    }
    for (uint32_t i = kHeld * kThreads + threadIdx.x; i < s.count; i += kThreads)
      atomicAdd(&fine[fine_bin(await_entry(candidates + i, load_relaxed(candidates + i)))], 1u);
    sync();
    // Thread t holds the fine bins just below kFineBins - kFinePerThread * t, so the prefix runs from the best bin down
    uint32_t counts[kFinePerThread], sum = 0;
#pragma unroll
    for (uint32_t j = 0; j < kFinePerThread; ++j)
      sum += counts[j] = fine[kFineBins - 1 - kFinePerThread * threadIdx.x - j];
    uint32_t total;
    uint32_t acc = block_exclusive_sum(sum, sm().scan, total);
#pragma unroll
    for (uint32_t j = 0; j < kFinePerThread; ++j) {
      if (acc < need and need <= acc + counts[j])
        sm().bin = kFineBins - 1 - kFinePerThread * threadIdx.x - j, sm().strict = acc, sm().count = counts[j];
      acc += counts[j];
    }
    sync();
    const bool found = sm().bin >= 0;
    const uint32_t crossing = sm().bin, above = sm().strict, in_bin = sm().count;
    const uint32_t rest = need - above;
    const bool take_bin = in_bin == rest;
    const auto place = [&](unsigned long long v) {
      const uint32_t fb = fine_bin(v);
      if (fb > crossing or (take_bin and fb == crossing)) {
        out_row[s.strict + atomicAdd(&sm().emitted, 1u)] = composite_slot(v);
      } else if (fb == crossing) {
        const uint32_t j = atomicAdd(&sm().listed, 1u);
        if (j < kStage) sm().list[j] = v;
      }
    };
    // The second pass reads the entries beyond the held ones again, then restores them to zero
    for (uint32_t i = kHeld * kThreads + threadIdx.x; i < s.count; i += kThreads) {
      const unsigned long long v = load_relaxed(candidates + i);
      candidates[i] = 0;
      if (found) place(v);
    }
    if (not found) return false;
#pragma unroll
    for (uint32_t k = 0; k < kHeld; ++k) {
      if (k * kThreads + threadIdx.x < s.count) place(held[k]);
    }
    sync();
    if (take_bin) return true;
    if (sm().listed != in_bin or in_bin > kStage) return false;
    int32_t* out_bin = out_row + s.strict + above;
    const unsigned long long* list = sm().list;
    if (in_bin <= Cfg::kDirectRank) {
      // Composites are distinct, so ranks are a permutation of 0..in_bin-1
      if (threadIdx.x < in_bin) {
        const unsigned long long mine = list[threadIdx.x];
        uint32_t rank = 0;
        for (uint32_t j = 0; j < in_bin; ++j)
          rank += list[j] > mine;
        if (rank < rest) out_bin[rank] = composite_slot(mine);
      }
      return true;
    }
    radix_select(in_bin, rest, [&](uint32_t i) { return list[i]; }, out_bin);
    return true;
  }

  // Waits until the row's other parts have finished writing
  static __device__ __forceinline__ void await_parts(const P& p, uint32_t row, uint32_t parts) {
    if (threadIdx.x == 0) {
      const uint32_t base = p.state[row].done_base;
      while (load_acquire(&p.state[row].done) - base != (parts - 1) * kPartSignal) {
      }
    }
    sync();
  }

  static __device__ void finalize_row(
      const P& p,
      const Split& s,
      uint32_t row,
      uint32_t length,
      uint32_t parts,
      int32_t* out_row,
      const int32_t* table_row) {
    const unsigned long long word = sm().row_word;
    const auto strict_total = static_cast<uint32_t>(word >> 24) & 0xffffffu;
    const auto inside_total = static_cast<uint32_t>(word) & 0xffffffu;
    bool ok =
        s.certified and strict_total == s.strict and inside_total == s.count and (s.whole or s.count <= p.capacity);
    const bool drained = ok and not s.whole;  // resolve_bin reads and zeroes every candidate
    if (drained) ok = resolve_bin(p, s, row, out_row);
    // Every thread has read its shared state; the fallback reuses it
    sync();
    if (not ok) {
      // Late writes of the other parts must land first: candidates are restored to zero, outputs rewritten
      await_parts(p, row, parts);
      if (s.certified and not s.whole and not drained) {
        unsigned long long* candidates = p.candidates + static_cast<size_t>(row) * p.capacity;
        for (uint32_t i = threadIdx.x; i < min(inside_total, p.capacity); i += kThreads)
          candidates[i] = 0;
      }
      select_row_exact(p, row, length, out_row, table_row);
    }
    clear_histogram(p, row);
    if (threadIdx.x == 0) {
      // Every part has arrived, so the word can be reset; their `done` signals may still be in flight
      p.state[row].word = 0;
      atomicAdd(&p.state[row].done_base, (parts - 1) * kPartSignal);  // no return value: a fire-and-forget reduction
    }
  }

  // `v` holds the first tile; the next tile's loads reuse its registers once its scores are compared
  static __device__ __forceinline__ void
  scan_segment(const Split& s, const Score* x, uint32_t begin, uint32_t end, Vector (&v)[kVectors]) {
    uint32_t strict_mine = 0;
    uint32_t lo2 = 0;
    if constexpr (not kFp32) lo2 = Cfg::threshold(s.lo);
    for (uint32_t base = begin; base < end; base += kTile) {
      uint32_t hits = 0;
      if constexpr (kFp32) {
#pragma unroll
        for (uint32_t u = 0; u < kVectors; ++u) {
#pragma unroll
          for (uint32_t q = 0; q < 4; ++q)
            hits |= static_cast<uint32_t>(Cfg::element(v[u], q) >= s.lo) << (u * 4 + q);
        }
      } else {
        hits = tile_at_least(v, lo2);
      }
      if (base + kTile > end) hits &= live_bits(base, end);
      if (base + kTile < end) load_tile(v, x, base + kTile, end);
      if (__any_sync(0xffffffffu, hits != 0)) stage_hits(x, hits, base, s.hi, strict_mine);
    }
    // The counts let publishing reserve before it has split the staged hits
    const uint32_t warp_strict = __reduce_add_sync(0xffffffffu, strict_mine);
    if (lane_id() == 0 and warp_strict != 0) atomicAdd(&sm().strict_count, warp_strict);
  }

  // ------------------------------------------------------------------------------------------------ short rows
  // Vector u of thread t holds scores (u * kThreads + t) * kPerVector + q (mask bit u * kPerVector + q) of a short row
  // (K < length <= Cfg::kLocalScores); a vector never straddles a page.
  // A short row's crossing bin is counted into kBuckets buckets of its keys (one per lane of a warp)
  static constexpr uint32_t kBucketBits = 5;
  static constexpr uint32_t kBuckets = 1u << kBucketBits;
  // Capacity of a short row's crossing-bucket list (in the histogram copy's shared memory)
  static constexpr uint32_t kCrossCap = kSlicePages * sizeof(int32_t) / sizeof(unsigned long long);
  static constexpr uint32_t kLocalVectors = (Cfg::kLocalScores + kThreads * kPerVector - 1) / (kThreads * kPerVector);
  static_assert(kLocalVectors * kPerVector <= 32, "a thread's short-row scores index one 32-bit mask");

  // Entries of list[0, n) ranked below `need` go to out[rank]: lanes hold the entries as columns, one ballot per
  // column (missing entries are zero, which no composite is).
  static __device__ __forceinline__ void
  local_ballot_rank(const unsigned long long* list, uint32_t n, uint32_t need, int32_t* out) {
    static_assert(kRankCount == 128, "four 32-entry columns");
    const uint32_t lane = lane_id();
    unsigned long long column[4];
#pragma unroll
    for (uint32_t c = 0; c < 4; ++c)
      column[c] = c * 32 + lane < n ? list[c * 32 + lane] : 0ull;
#pragma unroll 1
    for (uint32_t i = threadIdx.x >> 5; i < n; i += kWarps) {
      const unsigned long long target = list[i];
      uint32_t rank = 0;
#pragma unroll
      for (uint32_t c = 0; c < 4; ++c)
        rank += __popc(__ballot_sync(0xffffffffu, column[c] > target));
      if (lane == 0 and rank < need) out[rank] = composite_slot(target);
    }
  }

  // Out of line: rare (n > 32), and kept out of the short-row path's code
  static __device__ __noinline__ void
  rank_gathered(const unsigned long long* gathered, uint32_t n, uint32_t rest, int32_t* out_bin) {
    if (n <= kRankCount) {
      local_ballot_rank(gathered, n, rest, out_bin);
    } else {
      radix_select(n, rest, [&](uint32_t i) { return gathered[i]; }, out_bin);
    }
  }

  // Score j of this thread's short-row vectors, without indexing registers dynamically
  static __device__ __forceinline__ float pick_local(const Vector (&v)[kLocalVectors], uint32_t j) {
    const uint32_t u = j / kPerVector;
    Vector w = v[0];
#pragma unroll
    for (uint32_t k = 1; k < kLocalVectors; ++k)
      w = u == k ? v[k] : w;
    return Cfg::element(w, j % kPerVector);
  }

  // A short row on this CTA alone, its scores in registers: no arrival atomic or global candidates. The crossing bin's
  // scores are listed in shared memory and counted into kBuckets key buckets; only the crossing bucket is ranked.
  static __device__ void local_row(const P& p, uint32_t row, uint32_t length, int2 bins) {
    static_assert(Cfg::kLocalScores <= kLocalVectors * kThreads * kPerVector, "a short row fits the local vectors");
    static_assert(kCoarsePerThread == 2, "two coarse bins per thread");
    int32_t* out_row = p.out + static_cast<size_t>(row) * kTopK;
    const uint32_t table_index = p.rows_per_table_row == 1 ? row : row / p.rows_per_table_row;
    const int32_t* table_row = p.table + static_cast<size_t>(table_index) * p.table_stride;
    const Score* x = p.scores + static_cast<size_t>(row) * p.score_stride;
    Vector v[kLocalVectors];
#pragma unroll
    for (uint32_t u = 0; u < kLocalVectors; ++u) {
      const uint32_t first = (u * kThreads + threadIdx.x) * kPerVector;
      v[u] = Vector{};
      if (u * kThreads * kPerVector < length and first < length) v[u] = Cfg::load_vector(x + first);
    }
    const PageSlice pages(table_row, 0, length);
    // Edges of coarse bins 2t - 1 .. 2t + 1, computed after the length arrived while the histogram is in flight
    const float nan = __int_as_float(0x7fc00000);
    uint32_t b0 = 2 * threadIdx.x;
    asm volatile("" : "+r"(b0) : "r"(length));
    const float edge_above = b0 > 0 ? coarse_lower_edge(b0 - 1) : nan;
    const float edge0 = coarse_lower_edge(b0), edge1 = coarse_lower_edge(b0 + 1);
    const uint32_t key_above = b0 > 0 ? Cfg::edge(edge_above) : Cfg::kKeyAboveInf;  // just above +inf
    const uint32_t key0 = Cfg::edge(edge0), key1 = Cfg::edge(edge1);
    // Split, first half: the warps' running counts of their bin pairs and their totals
    const uint32_t lane = lane_id(), warp = threadIdx.x >> 5;
    const uint32_t a = min(static_cast<uint32_t>(bins.x), kMaxCount), b = min(static_cast<uint32_t>(bins.y), kMaxCount);
    const uint32_t inclusive = warp_inclusive_sum(a + b);
    if (lane == 31) sm().scan[warp] = inclusive;
    if (threadIdx.x == 0) {
      sm().local = Split{false, false, -1, 0, 0, nan, nan, 0, 0, 0};
      sm().staged_count = 0;
    }
    if (threadIdx.x < kBuckets + 2) sm().radix[threadIdx.x] = 0;  // bucket counts, then the two resolve counters
    PageSlice::wait();
    sync();
    // Every thread has read its bins: the row's histogram can be returned to zero now
    clear_histogram(p, row);
    {
      // Split, second half: the thread whose bin pair reaches K publishes
      static_assert(kWarps % 4 == 0);
      uint32_t warps_before = 0, total = 0;
#pragma unroll
      for (uint32_t k = 0; k < kWarps / 4; ++k) {
        const uint4 q = reinterpret_cast<const uint4*>(sm().scan)[k];
        warps_before += (4 * k < warp ? q.x : 0u) + (4 * k + 1 < warp ? q.y : 0u) + (4 * k + 2 < warp ? q.z : 0u) +
                        (4 * k + 3 < warp ? q.w : 0u);
        total += (q.x + q.y) + (q.z + q.w);
      }
      const uint32_t before = warps_before + inclusive - (a + b);
      // The crossing bin holds the K-th largest score: before < K <= before + count
      if (total == length and before < kTopK and kTopK <= before + a + b) {
        const bool first = kTopK <= before + a;
        Split split;
        split.certified = true;
        split.bin = static_cast<int32_t>(first ? b0 : b0 + 1);
        split.strict = first ? before : before + a;
        split.count = first ? a : b;
        split.whole = split.count == kTopK - split.strict;
        split.lo = first ? edge0 : edge1;
        split.hi = first ? edge_above : edge0;
        split.key_lo = first ? key0 : key1;
        const uint32_t width = (first ? key_above : key0) - split.key_lo;
        split.shift = width > kFineBins ? 32 - __clz(width - 1) - kFineBits : 0;
        split.bucket_shift = width > kBuckets ? 32 - __clz(width - 1) - kBucketBits : 0;
        sm().local = split;
      }
    }
    sync();
    const Split s = sm().local;
    // `hits`: scores >= lo (in or above the crossing bin); `strict`: above it
    uint32_t hits = 0, strict = 0;
    if constexpr (kFp32) {
#pragma unroll
      for (uint32_t u = 0; u < kLocalVectors; ++u) {
        if (u * kThreads * kPerVector < length) {  // the whole CTA skips vectors past the row
#pragma unroll
          for (uint32_t q = 0; q < 4; ++q) {
            hits |= static_cast<uint32_t>(Cfg::element(v[u], q) >= s.lo) << (u * 4 + q);
            strict |= static_cast<uint32_t>(Cfg::element(v[u], q) >= s.hi) << (u * 4 + q);
          }
        }
      }
    } else {
      const uint32_t lo2 = Cfg::threshold(s.lo), hi2 = Cfg::threshold(s.hi);
#pragma unroll
      for (uint32_t u = 0; u < kLocalVectors; ++u) {
        if (u * kThreads * kPerVector < length) {
          hits |= Cfg::at_least(v[u], lo2) << (u * kPerVector);
          strict |= Cfg::at_least(v[u], hi2) << (u * kPerVector);
        }
      }
    }
    {
      // Dead scores only lie in the last live vector
      const uint32_t last = (length - 1) / (kThreads * kPerVector);
      const uint32_t first = (last * kThreads + threadIdx.x) * kPerVector;
      const uint32_t n = length > first ? min(length - first, kPerVector) : 0u;
      const uint32_t live = ~((((1u << kPerVector) - 1) & ~((1u << n) - 1)) << (last * kPerVector));
      hits &= live, strict &= live;
    }
    // One shared atomic per warp reserves its places; a warp whose places pass the histogram's counts writes nothing
    // (the counts check then falls back), so no write needs a bound.
    const uint32_t inside = hits ^ strict;
    const uint32_t mine = __popc(strict) | (__popc(inside) << 16);
    const uint32_t warp_inclusive = warp_inclusive_sum(mine);
    uint32_t warp_base = 0;
    bool in_bounds = false;
    if (lane == 31) {
      warp_base = atomicAdd(&sm().staged_count, warp_inclusive);
      const uint32_t end = warp_base + warp_inclusive;
      in_bounds = (end & 0xffffu) <= s.strict and (end >> 16) <= (s.whole ? s.count : min(s.count, kStage));
    }
    const uint32_t offset = __shfl_sync(0xffffffffu, warp_base, 31) + warp_inclusive - mine;
    const uint32_t need = kTopK - s.strict;
    unsigned long long* list = sm().list;
    uint32_t* bucket = sm().radix;  // [kBuckets] counts of the crossing bin's keys, then two place counters
    const auto bucket_of = [&](uint32_t key) { return min((key - s.key_lo) >> s.bucket_shift, kBuckets - 1); };
    // This thread's vector j / kPerVector lies at `page_offset` in one page, kVectorPages after the previous vector's
    static_assert(((kThreads * kPerVector) >> Cfg::kPageBits) << Cfg::kPageBits == kThreads * kPerVector);
    constexpr uint32_t kVectorPages = (kThreads * kPerVector) >> Cfg::kPageBits;
    const int32_t* slice = sm().pages + ((threadIdx.x * kPerVector) >> Cfg::kPageBits);  // the slice starts at page 0
    const int32_t page_offset = static_cast<int32_t>((threadIdx.x * kPerVector) & ((1u << Cfg::kPageBits) - 1));
    const auto slot = [&](uint32_t j) {
      return (slice[(j / kPerVector) * kVectorPages] << Cfg::kPageBits) + page_offset +
             static_cast<int32_t>(j % kPerVector);
    };
    if (s.certified and __ballot_sync(0xffffffffu, in_bounds) != 0) {
      uint32_t strict_pos = offset & 0xffffu, inside_pos = offset >> 16;
      // Two hits per step: their page lookups and stores overlap
      uint32_t left = strict;
      for (; __popc(left) >= 2; left &= left - 1, left &= left - 1, strict_pos += 2) {
        const uint32_t j0 = __ffs(left) - 1, j1 = __ffs(left & (left - 1)) - 1;
        const int32_t s0 = slot(j0), s1 = slot(j1);
        out_row[strict_pos] = s0, out_row[strict_pos + 1] = s1;
      }
      if (left != 0) out_row[strict_pos] = slot(__ffs(left) - 1);
      if (__builtin_expect(not s.whole, 1)) {
        for (uint32_t rest_bits = inside; rest_bits != 0; rest_bits &= rest_bits - 1) {
          const uint32_t j = __ffs(rest_bits) - 1;
          const uint32_t key = Cfg::key(pick_local(v, j));
          list[inside_pos++] = composite(key, slot(j));
          atomicAdd(&bucket[bucket_of(key)], 1u);
        }
      } else {
        for (uint32_t rest_bits = inside; rest_bits != 0; rest_bits &= rest_bits - 1)
          out_row[s.strict + inside_pos++] = slot(__ffs(rest_bits) - 1);
      }
    }
    sync();
    // The same in every thread: the counts must be the histogram's
    const uint32_t placed = sm().staged_count;
    bool done = false;
    if (s.certified and (placed & 0xffffu) == s.strict and (placed >> 16) == s.count) {
      if (s.whole) {
        done = true;
      } else if (s.count <= kStage) {
        // The crossing bucket's entries are gathered in the page slice's shared memory (no longer needed); the warps
        // holding list entries find the bucket, warp 0 publishes it.
        unsigned long long* gathered = reinterpret_cast<unsigned long long*>(sm().pages);
        if ((threadIdx.x & ~31u) < s.count) {
          const uint32_t c = bucket[kBuckets - 1 - lane];
          const uint32_t counted = warp_inclusive_sum(c);
          const uint32_t at = __ffs(__ballot_sync(0xffffffffu, counted >= need)) - 1;  // the counts add up to >= need
          const uint32_t above = __shfl_sync(0xffffffffu, counted - c, at);
          const uint32_t in_bucket = __shfl_sync(0xffffffffu, c, at);
          const uint32_t crossing = kBuckets - 1 - at;
          const bool take = in_bucket == need - above;
          for (uint32_t i = threadIdx.x; i < s.count; i += kThreads) {
            const unsigned long long e = list[i];
            const uint32_t eb = bucket_of(composite_key(e));
            if (eb > crossing or (take and eb == crossing)) {
              out_row[s.strict + atomicAdd(&bucket[kBuckets], 1u)] = composite_slot(e);
            } else if (eb == crossing) {
              const uint32_t at_gather = atomicAdd(&bucket[kBuckets + 1], 1u);
              if (at_gather < kCrossCap) gathered[at_gather] = e;
            }
          }
          if (threadIdx.x == 0) sm().strict = above, sm().count = take ? 0u : in_bucket, sm().emitted = 0;
        }
        sync();
        const uint32_t above = sm().strict, in_bucket = sm().count;  // in_bucket 0: the bucket was taken whole
        int32_t* out_bin = out_row + s.strict + above;
        const uint32_t rest = need - above;
        if (in_bucket == 0) {
          done = true;
        } else if (in_bucket <= 32) {
          // One column: warp w ranks entries w and w + kWarps with a ballot each
          const unsigned long long column = lane < in_bucket ? gathered[lane] : 0ull;
          for (uint32_t i = warp; i < in_bucket; i += kWarps) {
            const unsigned long long target = gathered[i];
            const uint32_t rank = __popc(__ballot_sync(0xffffffffu, column > target));
            if (lane == 0 and rank < rest) out_bin[rank] = composite_slot(target);
          }
          done = true;
        } else if (in_bucket <= kCrossCap) {
          rank_gathered(gathered, in_bucket, rest, out_bin);
          done = true;
        }
      }
    }
    if (not done) {
      // A histogram that disagrees with the scores, or a bin beyond the lists: exact select of the whole row
      sync();
      select_row_exact(p, row, length, out_row, table_row);
    }
    sync();  // the next row of this CTA reuses the shared memory
  }

  // Scores [begin, end) of a row; `early` holds the row's histogram bins when the caller loaded them (`use_early`)
  static __device__ void run_part(
      const P& p,
      uint32_t row,
      uint32_t length,
      uint32_t begin,
      uint32_t end,
      uint32_t part,
      uint32_t parts,
      int2 early = make_int2(0, 0),
      bool use_early = false) {
    if (threadIdx.x == 0) sm().staged_count = 0, sm().strict_count = 0, sm().reserved = 0;
    sync();
    int32_t* out_row = p.out + static_cast<size_t>(row) * kTopK;
    const int32_t* table_row = p.table + static_cast<size_t>(row / p.rows_per_table_row) * p.table_stride;
    if (length <= kTopK) {
      // Every live score is selected: no split, no hand-off. Part 0 pads and clears the histogram.
      for (uint32_t i = begin + threadIdx.x; i < end; i += kThreads)
        out_row[i] = physical_slot<Cfg>(table_row, i);
      if (part == 0) {
        for (uint32_t i = length + threadIdx.x; i < kTopK; i += kThreads)
          out_row[i] = -1;
        clear_histogram(p, row);
      }
      return;
    }
    const Score* x = p.scores + static_cast<size_t>(row) * p.score_stride;
    Vector current[kVectors] = {};
    load_tile(current, x, begin, end);  // in flight while the split is derived
    const PageSlice pages(table_row, begin, end);
    const int2 bins = use_early ? early : load_histogram(p, row);
    const Split s = derive_split(bins, length);
    bool last;
    if (s.certified and end - begin <= kTile) {
      last = publish_tile(p, s, row, parts, out_row, pages, current, begin, end);
    } else {
      if (s.certified) scan_segment(s, x, begin, end, current);
      last = publish_segment(p, s, row, parts, out_row, pages);
    }
    if (last) {
      finalize_row(p, s, row, length, parts, out_row, table_row);
      sync();
    }
    sync();
  }

  // Each row owns gridDim / rows slots, numbered part-major: the live parts of all rows take distinct SMs before any
  // SM hosts two. With more rows than CTAs, rows are taken whole, round robin.
  static __device__ void run(const P& p) {
    const uint32_t slots = max(gridDim.x / p.rows, 1u);
    const uint32_t part = blockIdx.x / p.rows;
    // Slots no row could fill leave before reading the length words
    if (part >= parts_for<Cfg>(slots, p.score_width)) return;
    for (uint32_t row = blockIdx.x % p.rows; row < p.rows; row += gridDim.x) {
      // Part 0 (which alone selects a short row) loads the histogram with the length, and so does every part of a row
      // with at most kEarlySlots slots; more parts loading at once queue on the same lines.
      constexpr uint32_t kEarlySlots = 64;
      const bool early = part == 0 or slots <= kEarlySlots;
      int2 bins = make_int2(0, 0);
      if (early) bins = load_histogram(p, row);
      const uint32_t length = row_length(p, row);
      const uint32_t parts = parts_for<Cfg>(slots, length);
      if (part < parts) {
        // The longer rows' code comes first (its place in the kernel's code matters for cold launches)
        if (__builtin_expect(length <= kTopK or length > Cfg::kLocalScores, 1)) {
          const uint2 seg = segment<Cfg>(length, part, parts);
          run_part(p, row, length, seg.x, seg.y, part, parts, bins, early);
        } else {
          local_row(p, row, length, bins);
        }
      }
    }
  }
};

// kMinBlocks 512-thread CTAs per SM: one CTA per SM may use twice the registers, which spares the spills
template <typename Cfg, uint32_t kMinBlocks>
__global__ void __launch_bounds__(512, kMinBlocks)
    select_kernel(const __grid_constant__ Params<typename Cfg::Score> p) {
  // Every reader waits before any histogram, score, length or persistent-state access.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  // Dependents may launch now; they wait for this grid's completion before reading its outputs
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  Selector<Cfg, 512>::run(p);
}

// Measured on B200 (FP32): parts over 12 tiles of a 512-thread CTA stream faster with 32 warps (BF16 tiles have the
// same bytes)
constexpr uint32_t kWideTiles = 12;

// Rows 16..63 on one 1024-thread CTA per SM, in Selector<Cfg, 512>::run's slots; only parts over kWideTiles use all
// 32 warps. The narrow path comes first: its code is the hot one.
template <typename Cfg>
__global__ void __launch_bounds__(1024, 1) select_kernel_wide(const __grid_constant__ Params<typename Cfg::Score> p) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const uint32_t slots = gridDim.x / p.rows;
  const uint32_t row = blockIdx.x % p.rows, part = blockIdx.x / p.rows;
  if (part >= parts_for<Cfg>(slots, p.score_width)) return;
  int2 bins = make_int2(0, 0);
  // FP32 loads a local row's histogram only once the row is known local: measured, an early load (unused by the
  // multi-part paths) slowed their part 0 by 0.3-0.6 us.
  constexpr bool kEarlyBins = not std::is_same_v<typename Cfg::Score, float>;
  if constexpr (kEarlyBins) {
    if (part == 0 and threadIdx.x < 512) bins = Selector<Cfg, 512>::load_histogram(p, row);
  }
  const uint32_t length = row_length(p, row);
  const uint32_t parts = parts_for<Cfg>(slots, length);
  if (part >= parts) return;
  const uint2 seg = segment<Cfg>(length, part, parts);
  if (length > Cfg::kTopK and length <= Cfg::kLocalScores) {
    if constexpr (not kEarlyBins) {
      if (threadIdx.x < 512) bins = Selector<Cfg, 512>::load_histogram(p, row);
    }
    if (threadIdx.x < 512) Selector<Cfg, 512>::local_row(p, row, length, bins);
    return;
  }
  const bool wide = __builtin_expect(length > Cfg::kTopK and seg.y - seg.x > kWideTiles * Selector<Cfg, 512>::kTile, 0);
  if (not wide) {
    if (threadIdx.x < 512) Selector<Cfg, 512>::run_part(p, row, length, seg.x, seg.y, part, parts);
  } else {
    Selector<Cfg, 1024>::run_part(p, row, length, seg.x, seg.y, part, parts);
  }
}

// scores [R, >= max length] with 16B-aligned rows; lengths/histogram/out per row; row r maps pages through
// table row r / rows_per_table_row. The workspace is zero before the first call and stays reusable afterwards.
template <typename Cfg>
inline void select(
    tvm::ffi::TensorView scores,
    tvm::ffi::TensorView lengths,
    tvm::ffi::TensorView histogram,
    tvm::ffi::TensorView table,
    int64_t rows_per_table_row,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView workspace,
    int64_t capacity) {
  using namespace sglang::host;
  using Score = typename Cfg::Score;
  auto R = SymbolicSize{"rows"};
  auto W = SymbolicSize{"score_width"};
  auto S = SymbolicSize{"score_stride"};
  auto T = SymbolicSize{"table_rows"};
  auto P = SymbolicSize{"table_cols"};
  auto TS = SymbolicSize{"table_stride"};
  auto WS = SymbolicSize{"workspace_bytes"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();
  TensorMatcher({R, W}).with_strides({S, 1}).with_dtype<Score>().with_device(device_).verify(scores);
  TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(lengths);
  TensorMatcher({R, kCoarseBins}).with_dtype<int32_t>().with_device(device_).verify(histogram);
  TensorMatcher({T, P}).with_strides({TS, 1}).with_dtype<int32_t>().with_device(device_).verify(table);
  TensorMatcher({R, Cfg::kTopK}).with_dtype<int32_t>().with_device(device_).verify(out);
  TensorMatcher({WS}).with_dtype<uint8_t>().with_device(device_).verify(workspace);
  const auto rows = static_cast<uint32_t>(R.unwrap());
  RuntimeCheck(rows >= 1, "rows must be positive");
  RuntimeCheck(S.unwrap() % Cfg::kPerVector == 0, "score_stride must be a multiple of ", Cfg::kPerVector);
  RuntimeCheck(reinterpret_cast<uintptr_t>(scores.data_ptr()) % 16 == 0, "scores must be 16-byte aligned");
  // RowState counts a row's scores in 24-bit fields
  RuntimeCheck(W.unwrap() < (1 << 24), "score_width must be below 2^24");
  RuntimeCheck(rows_per_table_row >= 1 and T.unwrap() * rows_per_table_row >= rows, "table has too few rows");
  RuntimeCheck(capacity >= 1 and capacity <= (1 << 24), "capacity out of range: ", capacity);
  const size_t workspace_bytes = rows * (sizeof(RowState) + capacity * sizeof(unsigned long long));
  RuntimeCheck(
      static_cast<size_t>(WS.unwrap()) == workspace_bytes,
      "workspace must be exactly ",
      workspace_bytes,
      " bytes: its layout depends on rows");
  RuntimeCheck(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0, "workspace must be 16-byte aligned");
  auto* base = static_cast<uint8_t*>(workspace.data_ptr());
  auto* state = reinterpret_cast<RowState*>(base);
  auto* candidates = reinterpret_cast<unsigned long long*>(base + rows * sizeof(RowState));
  const Params<Score> params{
      .scores = static_cast<const Score*>(scores.data_ptr()),
      .score_stride = S.unwrap(),
      .score_width = static_cast<uint32_t>(W.unwrap()),
      .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
      .histogram = static_cast<int32_t*>(histogram.data_ptr()),
      .table = static_cast<const int32_t*>(table.data_ptr()),
      .table_stride = TS.unwrap(),
      .rows_per_table_row = static_cast<uint32_t>(rows_per_table_row),
      .out = static_cast<int32_t*>(out.data_ptr()),
      .state = state,
      .candidates = candidates,
      .capacity = static_cast<uint32_t>(capacity),
      .rows = rows,
  };
  const DLDevice device = device_.unwrap();
  // Measured on B200 (FP32): the 1024-thread kernel pays off from 16 rows, two 512-thread CTAs per SM from 64
  constexpr uint32_t kSms = 148;
  constexpr uint32_t kWideRows = 16;
  constexpr uint32_t kTwoCtaRows = 64;
  constexpr size_t kSmem =
      std::max(sizeof(typename Selector<Cfg, 512>::Shared), sizeof(typename Selector<Cfg, 1024>::Shared));
  static_assert(kSmem <= 48 * 1024, "dynamic shared memory beyond 48 KiB needs a function attribute");
  static_assert(kWideRows < kTwoCtaRows and kTwoCtaRows <= kSms, "the wide kernel expects a slot per row");
  if (rows < kWideRows) {
    LaunchKernel(kSms, 512, device, kSmem).enable_pdl(true)(select_kernel<Cfg, 1>, params);
  } else if (rows < kTwoCtaRows) {
    LaunchKernel(kSms, 1024, device, kSmem).enable_pdl(true)(select_kernel_wide<Cfg>, params);
  } else if ((rows > kSms and rows <= 192) or (rows > 2 * kSms and rows <= 3 * kSms)) {
    // Measured on B200: extra slots help 149..192 rows; three CTAs per SM shorten the 297..444-row tail.
    LaunchKernel(3 * kSms, 512, device, kSmem).enable_pdl(true)(select_kernel<Cfg, 3>, params);
  } else {
    LaunchKernel(2 * kSms, 512, device, kSmem).enable_pdl(true)(select_kernel<Cfg, 2>, params);
  }
}

}  // namespace litetopk
