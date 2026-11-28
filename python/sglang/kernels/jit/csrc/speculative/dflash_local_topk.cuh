// DFlash2 K1 (local_topk): the top-K of each logits row [N, W] bf16 (one TP rank's shard), sorted (value desc,
// position asc), as packed words (order16(value) << 16 | position + 1; -0.0 counts as +0.0). One CTA per row: a
// threshold T from the lanes' maxima (at least K elements are >= T), the elements >= T compacted as distinct keys,
// each key's rank = the number of larger keys. Rows with more candidates than the cap (heavy ties) are recomputed by
// an exact radix select.
//
// Publish: the producer half of a push plane round (communicator.cuh). Rank r stores row n's words at [n * K, n * K +
// K) of slot (phase, src = r) in every rank's workspace with one multicast store. K2 is the consumer that polls,
// refills the zeros and commits the phase. The position is stored + 1, so no word is 0 (the empty
// marker).

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/algorithm.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace device::dflash_local_topk {

namespace ptx {

using namespace device::ptx;

// count + the number of the four words of v that are <= key (the carry of key - v_j).
SGL_DEVICE uint32_t count_le4(uint32_t count, uint32_t key, uint4 v) {
  [[maybe_unused]] uint32_t t;  // the borrow-out scratch of each sub.cc
  asm("sub.cc.u32 %1, %2, %3;\n\t"
      "addc.cc.u32 %0, %0, 0;\n\t"
      "sub.cc.u32 %1, %2, %4;\n\t"
      "addc.cc.u32 %0, %0, 0;\n\t"
      "sub.cc.u32 %1, %2, %5;\n\t"
      "addc.cc.u32 %0, %0, 0;\n\t"
      "sub.cc.u32 %1, %2, %6;\n\t"
      "addc.u32 %0, %0, 0;"
      : "+r"(count), "=&r"(t)
      : "r"(key), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
  return count;
}

}  // namespace ptx

// The raw bits of a bf16 pair (__nv_bfloat162 is not trivially copyable, so no std::bit_cast).
SGL_DEVICE uint32_t bits32(bf16x2_t x) {
  return reinterpret_cast<const uint32_t&>(x);
}

// bf16 bits -> u16 key, larger value -> larger key; -0.0 maps to +0.0's key.
SGL_DEVICE uint32_t order16(uint32_t bits) {
  bits = bits == 0x8000u ? 0u : bits;
  return bits ^ ((bits & 0x8000u) ? 0xFFFFu : 0x8000u);
}

SGL_DEVICE uint32_t unorder16(uint32_t key) {
  return key ^ ((key & 0x8000u) ? 0x8000u : 0xFFFFu);
}

// u32 key of an element: value first, then the lower position. Never 0 (the low half is 0xFFFF - pos > 0).
SGL_DEVICE uint32_t element_key(uint32_t bits, uint32_t pos) {
  return order16(bits) << 16 | (0xFFFFu - pos);
}

// Published word of an element key: the value key stays, the low half becomes position + 1 (never 0).
SGL_DEVICE uint32_t key_to_word(uint32_t key) {
  return (key & 0xFFFF0000u) | (0x10000u - (key & 0xFFFFu));
}

struct LocalTopKParams {
  const bf16_t* __restrict__ logits;  // [N, W], row stride logits_stride
  int64_t logits_stride;
  // Multicast VA of slot (phase 0, src = this rank): rows of [N][K] u32 words; phase 1 is phase_stride bytes on.
  uint8_t* publish_mc;
  const distributed::EpochCounter* counter;  // the plane's phase
  uint32_t phase_stride;                     // world_size * slot_bytes
};

// K1 shapes: a row of kWidth elements is kNumWarps warps x 32 lanes x kNumVecs 32 B vectors; per warp slice the
// vectors are laid out [v][lane][16] (a coalesced 1 KB per vector index). Vector v of a lane is chunks 2v and 2v + 1
// (16 B, 8 elements each): chunk c starts at lane * 16 + (c / 2) * 512 + (c % 2) * 8 of the warp slice.
template <uint32_t K, uint32_t kWidth>
struct LocalTopKShape {
  static constexpr uint32_t kNumVecs = 2;  // per lane
  static constexpr uint32_t kNumChunks = 2 * kNumVecs;
  static constexpr uint32_t kElemsPerWarp = 32 * 16 * kNumVecs;
  static_assert(kWidth % kElemsPerWarp == 0, "the shard width must be a multiple of 1024");
  static constexpr uint32_t kNumWarps = kWidth / kElemsPerWarp;
  static constexpr uint32_t kNumThreads = kNumWarps * 32;
  static constexpr uint32_t kElemsPerRow = kWidth;
  // Candidates ranked by counting (a half-warp per key, 16 keys per warp); the rank reads up to 2 x 64 slots,
  // zero-padded past n.
  static constexpr uint32_t kCountCap = 128;
  static_assert(K == 16 || K == 32);
  // 8: kCountCap keys ranked at once, K / kNumWarps <= 4; 32: the CTA's 1024 threads.
  static_assert(kNumWarps >= 8 && kNumWarps <= 32, "the shard width must be in [8192, 32768]");
  static_assert(kElemsPerRow <= 0xFFFF);  // position + 1 fits the word's 16 bits
  static_assert(kNumVecs <= 16);          // the chunk mask: vector v's chunks at bits v and 16 + v
};

// The exact top-K keys of a row into s_cand[0, K) (the rest of s_cand zeroed), for rows whose chunks or candidates
// overflow kCountCap: a radix select of the K-th largest element key, 8 bits per pass from the top, until every key
// of the selected bin is in the top K; then the K keys at or above that bin's lower edge. Slow (a pass over the row
// per step) but exact for any row.
template <typename S, uint32_t K>
SGL_DEVICE void select_exact(const bf16_t* row_src, uint32_t* s_cand, uint32_t* s_hist, uint32_t* s_select) {
  constexpr uint32_t kNumRowChunks = S::kElemsPerRow / 8;
  const uint32_t lane_id = threadIdx.x % 32;
  const auto for_each_key = [&](auto&& fn) {
#pragma unroll 1
    for (uint32_t c = threadIdx.x; c < kNumRowChunks; c += S::kNumThreads) {
      AlignedVector<bf16x2_t, 4> chunk;  // positions [8c, 8c + 8)
      chunk.load(row_src, c);
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t bits = bits32(chunk[i]);
        fn(element_key(bits & 0xFFFFu, c * 8 + 2 * i));
        fn(element_key(bits >> 16, c * 8 + 2 * i + 1));
      }
    }
  };
  // The target is the rem-th largest (1-based) of the keys whose bits under `mask` equal `prefix`.
  uint32_t prefix = 0, mask = 0, rem = K;
#pragma unroll 1
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (uint32_t i = threadIdx.x; i < 256; i += S::kNumThreads) {
      s_hist[i] = 0;
    }
    __syncthreads();
    for_each_key([&](uint32_t key) {
      if (((key ^ prefix) & mask) == 0) atomicAdd(&s_hist[(key >> shift) & 0xFFu], 1u);
    });
    __syncthreads();
    if (threadIdx.x < 32) {
      // lane l: bins 255 - 8l - j for j < 8, highest first; incl = the keys in its bins and all higher ones
      uint32_t cnt[8], sum = 0;
#pragma unroll
      for (uint32_t j = 0; j < 8; ++j) {
        cnt[j] = s_hist[255 - 8 * lane_id - j];
        sum += cnt[j];
      }
      const uint32_t incl = warp::inclusive_sum(sum, lane_id);
      const uint32_t owner = __ffs(__ballot_sync(0xFFFFFFFFu, incl >= rem)) - 1;
      if (lane_id == owner) {
        uint32_t above = incl - sum;
        for (uint32_t j = 0; j < 8; ++j) {
          if (above + cnt[j] >= rem) {
            s_select[0] = 255 - 8 * lane_id - j, s_select[1] = rem - above, s_select[2] = cnt[j];
            break;
          }
          above += cnt[j];
        }
      }
    }
    __syncthreads();
    prefix |= s_select[0] << shift;
    mask |= 0xFFu << shift;
    rem = s_select[1];
    const bool done = rem == s_select[2];
    __syncthreads();  // s_hist and s_select are rewritten by the next pass
    if (done) break;
  }
  // Exactly K keys are >= prefix: those above the selected bins and the whole last bin.
  for (uint32_t i = threadIdx.x; i < S::kCountCap; i += S::kNumThreads) {
    s_cand[i] = 0;
  }
  if (threadIdx.x == 0) s_select[3] = 0;
  __syncthreads();
  for_each_key([&](uint32_t key) {
    if (key >= prefix) s_cand[atomicAdd(&s_select[3], 1u)] = key;
  });
  __syncthreads();
}

// Local top-K, one CTA per row (kMinBlocks: CTAs per SM the register allocation must allow; the caller picks the
// fewest that keep every row's CTA resident):
//  1. every lane's maximum as a 16-bit order key (the lane key), and the maxima of its 16 B chunks;
//  2. T: a lower bound of the K-th largest lane key, so at least K elements are >= T;
//  3. the chunks whose maximum reaches T (they hold every element >= T) are listed in shared memory; then thread t
//     reloads chunk t (an L1 hit) and writes its elements >= T as distinct u32 keys (element_key) to a shared list;
//  4. a key's rank is the number of larger keys, ranks < K are stored.
// A row with more than kCountCap chunks or candidates runs steps 3a / 3b on clipped lists, then select_exact
// recomputes its K keys before step 4; so does a row with fewer than K candidates, which only a NaN lane maximum
// causes (no element compares >= a NaN T), so that every row publishes K words.
template <uint32_t K, uint32_t kWidth, uint32_t kMinBlocks, bool kUsePDL>
__global__ __launch_bounds__(LocalTopKShape<K, kWidth>::kNumThreads, kMinBlocks)  //
    void local_topk_kernel(const __grid_constant__ LocalTopKParams p) {
  using S = LocalTopKShape<K, kWidth>;
  constexpr uint32_t kNumWarps = S::kNumWarps;
  constexpr uint32_t kCountCap = S::kCountCap;
  constexpr uint32_t kNumVecs = S::kNumVecs;
  constexpr uint32_t kNumChunks = S::kNumChunks;
  constexpr uint32_t kBinBits = 5;  // the T window: 32 bins, one per lane of the suffix scan
  constexpr uint32_t kNumBins = 1u << kBinBits;
  // Each warp's bound is its kWarpBoundRank-th largest lane key: kNumWarps * kWarpBoundRank >= K, so at least K lane
  // keys of the CTA reach the smallest bound.
  constexpr uint32_t kWarpBoundRank = (K + kNumWarps - 1) / kNumWarps;
  static_assert(kWarpBoundRank <= 4);
  // Every listed chunk holds a candidate, so kCountCap bounds the list too; one thread per listed chunk.
  static_assert(S::kNumThreads >= kCountCap);

  __shared__ alignas(16) uint32_t s_cand[kCountCap];  // candidate keys, zero-padded
  // Row offsets of the chunks whose maximum reaches T; stores past kCountCap land in the padding (the row falls back).
  __shared__ uint32_t s_chunk[kCountCap + kNumChunks];
  __shared__ alignas(16) uint32_t s_cnt[kNumBins];
  __shared__ uint32_t s_warp_max_bound[kNumWarps];  // per warp: its largest lane key << 16 | its bound
  __shared__ uint32_t s_num_chunks, s_num_extra;
  __shared__ uint32_t s_hist[256], s_select[4];  // select_exact only

  const uint32_t lane_id = threadIdx.x % 32;
  const uint32_t warp_id = warp::reduce_max(threadIdx.x / 32);
  const uint32_t row = blockIdx.x;

  const auto row_src = assume_global(p.logits) + p.logits_stride * row;
  PDLWaitPrimary<kUsePDL>();
  // The phase after the PDL wait: an earlier grid may trigger its dependents before it commits the phase.
  const auto publish = reinterpret_cast<uint32_t*>(p.publish_mc + p.counter->load() * p.phase_stride) + row * K;
  AlignedVector<bf16x2_t, 8> v[kNumVecs];  // 32 B each
#pragma unroll
  for (uint32_t i = 0; i < kNumVecs; ++i) {
    v[i].load(row_src + (threadIdx.x / 32) * S::kElemsPerWarp, i * 32 + lane_id);
  }
  // Shared state, while the loads are in flight (ordered before its uses by the first barrier).
  for (uint32_t i = threadIdx.x; i < kCountCap; i += S::kNumThreads) {
    s_cand[i] = 0;
  }
  if (threadIdx.x < kNumBins) s_cnt[threadIdx.x] = 0;
  if (threadIdx.x == 0) s_num_chunks = 0, s_num_extra = 0;

  // 1. Each chunk's maximum by a serial bf16x2 max over its 4 pairs; vector v's two chunk maxima merged into
  //    vec_max[v] = (chunk 2v, chunk 2v + 1); then the lane maximum as a 16-bit order key.
  bf16x2_t vec_max[kNumVecs];
  uint32_t lane_key;
  {
#pragma unroll
    for (uint32_t i = 0; i < kNumVecs; ++i) {
      auto lo = v[i][0], hi = v[i][4];
#pragma unroll
      for (uint32_t j = 1; j < 4; ++j) {
        lo = __hmax2(lo, v[i][j + 0]);
        hi = __hmax2(hi, v[i][j + 4]);
      }
      vec_max[i] = __hmax2(__lows2bfloat162(lo, hi), __highs2bfloat162(lo, hi));
    }
    auto m = vec_max[0];
#pragma unroll
    for (uint32_t i = 1; i < kNumVecs; ++i) {
      m = __hmax2(m, vec_max[i]);
    }
    const auto lane_max = __hmax2(m, __lowhigh2highlow(m));  // both halves
    lane_key = order16(bits32(lane_max) & 0xFFFFu);
  }

  // 2. T, a 16-bit order key that at least K lane keys (so K elements) reach. Each warp's largest lane key and its
  //    bound (its kWarpBoundRank-th largest, by redux rounds over keys made distinct by the lane id) give a window
  //    [smallest bound, largest key] that holds the CTA's K-th largest lane key; every lane key in the window counts
  //    into kNumBins bins of 2^shift keys (red.add 1: same-address lanes merge in ATOMS.POPC.INC), and T is the lower
  //    edge of the highest bin that K lane keys reach.
  uint32_t T;
  {
    uint32_t lane_key_id = lane_key << 16 | lane_id;
    uint32_t popped = warp::reduce_max(lane_key_id);
    const uint32_t warp_max = popped >> 16;
#pragma unroll
    for (uint32_t r = 1; r < kWarpBoundRank; ++r) {
      lane_key_id = lane_key_id == popped ? 0u : lane_key_id;
      popped = warp::reduce_max(lane_key_id);
    }
    const uint32_t warp_bound = popped >> 16;
    if (lane_id == 0) s_warp_max_bound[warp_id] = warp_max << 16 | warp_bound;
    __syncthreads();
    const uint32_t max_bound = s_warp_max_bound[lane_id < kNumWarps ? lane_id : 0];
    const uint32_t window_hi = warp::reduce_max(max_bound >> 16);
    const uint32_t window_lo = warp::reduce_min(max_bound & 0xFFFFu);
    const uint32_t span = window_hi - window_lo;
    uint32_t shift = span < kNumBins ? 0u : (32u - __clz(span)) - kBinBits;
    shift += ((window_hi >> shift) - (window_lo >> shift)) >= kNumBins ? 1u : 0u;
    const uint32_t bin_base = window_lo >> shift;
    const bool in_window = lane_key >= window_lo;
    if (in_window) atomicAdd(&s_cnt[(lane_key >> shift) - bin_base], 1u);
    __syncthreads();
    // lane l: the number of lane keys in bins >= l
    const uint32_t at_or_above = warp::inclusive_sum<32, 32, 1>(s_cnt[lane_id], lane_id);
    const uint32_t top_bin = 31 - __clz(__ballot_sync(0xFFFFFFFFu, at_or_above >= K));
    T = top_bin == 0 ? window_lo : (bin_base + top_bin) << shift;
  }
  const auto thr2 = __bfloat162bfloat162(__ushort_as_bfloat16(static_cast<uint16_t>(unorder16(T))));

  // 3a. List the chunks whose maximum reaches T: bit v / 16 + v of `active` for chunk 2v / 2v + 1.
  {
    uint32_t active = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumVecs; ++i) {
      active |= __hge2_mask(vec_max[i], thr2) & (0x00010001u << i);
    }
    // One atomic per lane with a variable count: a literal 1 becomes a leader atomic plus a SHFL broadcast for any
    // active mask; this stays a per-lane ATOMS.ADD unless the whole warp is active.
    if (active != 0) {
      /// NOTE: for out of bound write, always write to a padded area
      uint32_t slot = min(atomicAdd(&s_num_chunks, __popc(active)), kCountCap);
      const uint32_t lane_offset = warp_id * S::kElemsPerWarp + lane_id * 16;
#pragma unroll
      for (uint32_t c = 0; c < kNumChunks; ++c) {
        const uint32_t bit = (active >> (c / 2 + (c % 2) * 16)) & 1u;
        if (bit) s_chunk[slot] = lane_offset + (c / 2) * 512 + (c % 2) * 8;
        slot += bit;
      }
    }
  }
  __syncthreads();
  const uint32_t num_chunks = s_num_chunks;
  const uint32_t num_listed = min(num_chunks, kCountCap);

  // 3b. Thread t: chunk t's elements >= T as keys. The first takes slot t, the rest (rare) append past the chunks.
  if (threadIdx.x < num_listed) {
    const uint32_t offset = s_chunk[threadIdx.x];
    AlignedVector<bf16x2_t, 4> chunk;  // 16 B
    chunk.load(row_src + offset);
    uint32_t pairs[4], ge[4];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      pairs[i] = bits32(chunk[i]);
      ge[i] = __hge2_mask(chunk[i], thr2);
    }
    // m: bit i = the low element of pair i is >= T, bit 16 + i = its high element (a merge tree of bit selects:
    // each __hge2_mask word is all ones or zeros per half). Not 0: the chunk's maximum reaches T.
    ge[0] = (ge[0] & ~0x00020002u) | (ge[1] & 0x00020002u);
    ge[2] = (ge[2] & ~0x00080008u) | (ge[3] & 0x00080008u);
    uint32_t m = ((ge[0] & ~0x000C000Cu) | (ge[2] & 0x000C000Cu)) & 0x000F000Fu;
    const auto pop_key = [&] {
      const uint32_t bit = 31 - __clz(m);
      m ^= 1u << bit;
      const uint32_t i = bit & 3u, hi = bit >> 4;
      const uint32_t pair = (i & 2u) ? ((i & 1u) ? pairs[3] : pairs[2]) : ((i & 1u) ? pairs[1] : pairs[0]);
      return element_key((pair >> (16 * hi)) & 0xFFFFu, offset + 2 * i + hi);
    };
    s_cand[threadIdx.x] = pop_key();
    if (m != 0) {
      uint32_t slot = num_listed + atomicAdd(&s_num_extra, __popc(m));
      do {
        const uint32_t key = pop_key();
        if (slot < kCountCap) s_cand[slot] = key;
        ++slot;
      } while (m != 0);
    }
  }
  __syncthreads();
  // The secondary's CTAs may land on this SM now: nothing below reads the row through L1 again (but select_exact).
  // K2 reads the GEMM's hp without a griddepcontrol.wait of its own: it relies on this trigger following the
  // PDLWaitPrimary above in every thread.
  PDLTriggerSecondary<kUsePDL>();
  uint32_t n = num_chunks + s_num_extra;
  // Uniform: the lists overflowed (n > kCountCap), or fewer than K candidates (a NaN T); one unsigned compare.
  static_assert(kCountCap >= K);
  if (n - K > kCountCap - K) {
    [[unlikely]] select_exact<S, K>(row_src, s_cand, s_hist, s_select);
    n = K;
  }

  // 4. Rank by counting. Half-warps: lanes l and l + 16 take the same key and half of the slots each (the zero
  //    padding counts as keys <= this one, so rank = slots - the count of keys <= it, the key itself included).
  static_assert(kNumWarps * 16 >= kCountCap);
  if (warp_id * 16 >= n) return;
  const uint32_t key = s_cand[warp_id * 16 + lane_id % 16];
  const uint32_t half = n <= 32 ? 16u : n <= 64 ? 32u : 64u;  // slots per half-warp, 2 x half <= kCountCap
  uint32_t le[8] = {};                                        // independent carry chains
  const uint4* q = reinterpret_cast<const uint4*>(s_cand + (lane_id / 16) * half);
  if (n <= 32) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      le[j] = ptx::count_le4(le[j], key, q[j]);
    }
  } else if (n <= 64) {
#pragma unroll
    for (uint32_t j = 0; j < 8; ++j) {
      le[j] = ptx::count_le4(le[j], key, q[j]);
    }
  } else {
#pragma unroll 1
    for (uint32_t c = 0; c < half / 4; c += 8) {
#pragma unroll
      for (uint32_t j = 0; j < 8; ++j) {
        le[j] = ptx::count_le4(le[j], key, q[c + j]);
      }
    }
  }
  uint32_t at_most = ((le[0] + le[1]) + (le[2] + le[3])) + ((le[4] + le[5]) + (le[6] + le[7]));
  at_most += __shfl_xor_sync(0xFFFFFFFFu, at_most, 16);
  const uint32_t rank = 2 * half - at_most;
  if (lane_id < 16 && key != 0 && rank < K) {
    ptx::multimem_store_relaxed(publish + rank, key_to_word(key));
  }
}

}  // namespace device::dflash_local_topk

template <uint32_t K, uint32_t kWidth, uint32_t kMinBlocks, bool kUsePDL>
struct DFlashLocalTopKKernel {
  static constexpr auto kernel = device::dflash_local_topk::local_topk_kernel<K, kWidth, kMinBlocks, kUsePDL>;

  static void run(tvm::ffi::TensorView logits, host::distributed::PushPlaneRef plane) {
    using namespace host;
    using Shape = device::dflash_local_topk::LocalTopKShape<K, kWidth>;
    const auto& push = *plane.get();
    auto N = SymbolicSize{"num_rows"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, kWidth})  //
        .with_strides({-1, 1})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .ensure_alignment(32)
        .verify(logits);
    const auto n = static_cast<uint32_t>(N.unwrap());
    if (n == 0) return;
    const auto world_size = push.world_size;
    const auto slot_bytes = push.slot_bytes;
    const auto required_bytes = static_cast<int64_t>(n) * K * 4;
    CHECK_HOST(required_bytes <= slot_bytes) << required_bytes << " > " << slot_bytes;
    CHECK_HOST(push.mc_workspace != nullptr) << "no multicast support";
    const auto params = device::dflash_local_topk::LocalTopKParams{
        .logits = static_cast<const bf16_t*>(logits.data_ptr()),
        .logits_stride = logits.stride(0),
        .publish_mc = push.mc_workspace + push.rank * slot_bytes,
        .counter = push.counter,
        .phase_stride = static_cast<uint32_t>(world_size * slot_bytes),
    };
    LaunchKernel(n, Shape::kNumThreads, device_.unwrap()).config({.use_pdl = kUsePDL})(kernel, params);
  }
};

}  // namespace sglang
