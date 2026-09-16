/**
 * \brief DeepSeek-V4.1's bf16 top-k kernel for short rows (<= 16384 scores)
 * Adapted from https://github.com/deepseek-ai/DeepSelect
 * Plain SIMT (no tensor cores or clusters), tuned for 16384-wide rows with k = 512.
 */
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>

namespace sglang {

/**
 * \brief bf16 top-k of one row that fits in registers (rows of at most 16384 scores: the
 *        DeepSeek-V4.1 sparse indexer's consumer rows), fused with a page-table transform.
 *
 * One CTA of 512 threads per row. The row is split into contiguous per-thread slices of up to
 * 32 scores held in registers for the whole kernel. Two radix passes (the raw high byte, then
 * the raw low byte among the elements sharing the pivot's high byte) locate the k-th largest
 * value exactly, a census then tells every thread how many of its elements are above / equal
 * to it and where they go, and the selected indices are staged in shared memory before one
 * coalesced, page-transformed copy to the output. This is DeepSelect's init-window select.
 *
 * \note The value order used everywhere is the "distorted" order of the raw bf16 bits
 *       (`x ^ (x < 0 ? 0xFFFF : 0x8000)`, negatives below positives, -0 below +0). The
 *       histograms are indexed by the *raw* byte instead, and the pivot search undoes the
 *       permutation once per lane, so no element pays the distortion.
 * \note NaN scores are not selected: the ordered compares never match them, so a row with n
 *       positive NaNs yields its top (k - n) real scores and -1 in the remaining slots (a
 *       negative NaN orders below -inf and is simply never picked).
 */
struct TopKBF16Config {
  static constexpr uint32_t kBlockSize = 512;
  static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
  static constexpr uint32_t kOccupancy = 3;
  static constexpr uint32_t kVecSize = 8;
  static constexpr uint32_t kMaxVecs = 4;
  static constexpr uint32_t kElemsPerThread = kVecSize * kMaxVecs;
  static constexpr uint32_t kMaxSeqLen = kBlockSize * kElemsPerThread;
  static constexpr uint32_t kMaxTopK = 2048;
  static constexpr uint32_t kNumBins = 256;
  static constexpr uint32_t kSinkBin = kNumBins;  // LSB pass sends out-of-bucket elements here
  /// NOTE: in the MSB row the negative half lives 16 words further up. Raw bytes 128 apart share
  /// a bank, so without it +x and -x with the same exponent (the common case for centered data)
  /// collide on every histogram update; measured as half of all atomic wavefronts.
  static constexpr uint32_t kNegShift = 16;
  static constexpr uint32_t kHistStride = kNumBins + kNegShift + 4;  // keeps both rows 16 B aligned
  /// NOTE: a negative NaN. In the distorted order it sits below -inf, and every ordered bf16
  /// comparison against it is false, so padding is never counted nor selected.
  static constexpr uint32_t kPadElem = 0xFFFFu;
  static constexpr uint32_t kNegZeroBits = 0x8000u;
  using vec_t = device::AlignedVector<bf16x2_t, kVecSize / 2>;
  static_assert(kMaxSeqLen == 16384 && kMaxSeqLen <= 0xFFFF);  // the census counters pack in 16 bits
  // one census bit per element of the slice, in a uint32_t
  static_assert(kElemsPerThread == 32);

  struct Smem {
    uint32_t count_gt_eq;  // packed (gt << 16 | eq), the block-wide census prefix
    uint32_t pivot_bin;
    uint32_t pivot_remain;
    union {
      alignas(16) uint32_t histogram[2][kHistStride];
      alignas(16) uint32_t stage[kMaxTopK];
    };
  };
};

struct TopKBF16Params {
  const bf16_t* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int64_t score_stride;
  int64_t page_table_stride;
  int64_t page_indices_stride;
  uint32_t topk;
  uint32_t page_bits;
};

SGL_DEVICE uint32_t get_ptx_lane_id() {
  uint32_t lane_id;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
  return lane_id;
}

/// \brief Exclusive suffix scan: lane `L` gets the sum over lanes `> L`.
SGL_DEVICE uint32_t warp_exclusive_suffix_sum(uint32_t x, uint32_t lane_id) {
  uint32_t inc = x;
#pragma unroll
  for (uint32_t offset = 1; offset < device::kWarpThreads; offset <<= 1) {
    const auto t = __shfl_down_sync(device::kFullMask, inc, offset);
    if (lane_id + offset < device::kWarpThreads) inc += t;
  }
  return inc - x;
}

template <typename To, typename From>
SGL_DEVICE To bitcast(const From& f) {
  static_assert(sizeof(From) == sizeof(To));
  return reinterpret_cast<const To&>(f);
}

struct TopKBF16Pivot {
  uint32_t bin;     // in distorted (value-ascending) order, [0, 256)
  uint32_t remain;  // how many elements of `bin` still have to be taken
};

/**
 * \brief Locate the bin holding the k-th largest element in a 256-bin histogram indexed by a
 *        raw byte. Called by one whole warp; exactly one lane finds it and writes the answer
 *        to `smem.pivot_*` (the block reads it behind the caller's barrier, so there is no
 *        point in broadcasting it inside the warp first).
 * \param msb_mode  The raw byte is the high byte: lanes < 16 cover raw 0xFF..0x80 (negatives,
 *                  reversed), lanes >= 16 cover raw 0x00..0x7F.
 * \param negative  LSB mode only: the pivot bucket is negative, so the whole byte is reversed.
 */
SGL_DEVICE void topk_bf16_find_pivot_warp(
    const uint32_t* hist, uint32_t k, bool msb_mode, bool negative, uint32_t lane_id, TopKBF16Config::Smem& smem) {
  using C = TopKBF16Config;
  // lane L owns distorted bins [8L, 8L + 8)
  const bool reverse = msb_mode ? lane_id < 16 : negative;
  uint32_t raw_base = reverse ? 0xF8 - 8 * lane_id : 8 * lane_id - (msb_mode ? 0x80 : 0);
  if (msb_mode && reverse) raw_base += C::kNegShift;
  device::AlignedVector<uint32_t, 4> lo, hi;
  lo.load(hist + raw_base);
  hi.load(hist + raw_base + 4);
  uint32_t count[8];
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const auto fwd = i < 4 ? lo[i] : hi[i - 4];
    const auto rev = i < 4 ? hi[3 - i] : lo[7 - i];
    count[i] = reverse ? rev : fwd;
  }
  uint32_t local = 0;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    local += count[i];
  }
  // suffix[j] = number of elements in bins >= 8L + j
  uint32_t suffix[9];
  suffix[8] = warp_exclusive_suffix_sum(local, lane_id);
#pragma unroll
  for (int32_t j = 7; j >= 0; --j) {
    suffix[j] = suffix[j + 1] + count[j];
  }
  // exactly one lane satisfies suffix[8] < k <= suffix[0]; inside it, the pivot is the largest
  // offset j with suffix[j] >= k
  const bool found = suffix[8] < k && k <= suffix[0];
  uint32_t offset = 0;
  uint32_t next = suffix[1];
#pragma unroll
  for (uint32_t j = 1; j < 8; ++j) {
    if (suffix[j] >= k) {
      offset = j;
      next = suffix[j + 1];
    }
  }
  if (found) {
    smem.pivot_bin = 8 * lane_id + offset;
    smem.pivot_remain = k - next;
  }
}

/// \brief One byte of hit bits for the 8 elements of a vector, element `e` at bit `e`.
/// \param m Per-pair 16-bit masks (0xFFFF / 0) as produced by `__hgt2_mask` and friends.
SGL_DEVICE uint32_t topk_bf16_pack_hits(const uint32_t (&m)[4]) {
  // one flag byte per element (0xFF / 0x00), then signed dot products turn them into bits
  const auto lo = __byte_perm(m[0], m[1], 0x7531);
  const auto hi = __byte_perm(m[2], m[3], 0x7531);
  const auto nib = __dp4a(static_cast<int>(lo), static_cast<int>(0xF8FCFEFFu), 0);  // -1,-2,-4,-8
  return __dp4a(static_cast<int>(hi), static_cast<int>(0x80C0E0F0u), nib);          // -16..-128
}

template <bool kUsePDL>
__global__ __launch_bounds__(TopKBF16Config::kBlockSize, TopKBF16Config::kOccupancy)  //
    void topk_bf16_small_kernel(const __grid_constant__ TopKBF16Params params) {
  using namespace device;
  using C = TopKBF16Config;
  using vec_t = C::vec_t;
  __shared__ C::Smem smem;

  const auto bx = blockIdx.x;
  const auto tx = threadIdx.x;
  const auto lane_id = get_ptx_lane_id();
  const auto warp_id = tx / kWarpThreads;
  const auto topk = params.topk;
  // a selected index i maps through this row's table to slot
  // table[i >> page_bits] << page_bits | (i & mask); -1 past what the row has
  const auto* __restrict__ table = params.page_table + bx * params.page_table_stride;
  auto* __restrict__ out = params.page_indices + bx * params.page_indices_stride;
  const auto page_bits = params.page_bits;
  const auto page_mask = (1u << page_bits) - 1;
  const auto transform = [&](uint32_t idx) -> int32_t {
    return (table[idx >> page_bits] << page_bits) | static_cast<int32_t>(idx & page_mask);
  };

  {
    using zero_vec_t = AlignedVector<uint32_t, 4>;
    static_assert(sizeof(smem.histogram) % sizeof(zero_vec_t) == 0);
    constexpr uint32_t kZeroVecs = sizeof(smem.histogram) / sizeof(zero_vec_t);
    zero_vec_t zeros;
    zeros.fill(0);
#pragma unroll
    for (uint32_t idx = tx; idx < kZeroVecs; idx += C::kBlockSize) {
      zeros.store(smem.histogram, idx);
    }
    if (tx == 0) smem.count_gt_eq = 0;
  }

  // NOTE: we prefetch metadata like seq_len
  const auto seq_len = static_cast<uint32_t>(params.seq_lens[bx]);
  const auto* __restrict__ scores_row = params.scores + bx * params.score_stride;
  if (seq_len <= topk) {  // every element is selected, -1 past the row
    PDLWaitPrimary<kUsePDL>();
    for (uint32_t t = tx; t < topk; t += C::kBlockSize) {
      out[t] = t < seq_len ? transform(t) : -1;
    }
    return PDLTriggerSecondary<kUsePDL>();
  }
  PDLWaitPrimary<kUsePDL>();

  // Contiguous slices of whole vectors, balanced so short rows still spread over the block.
  // Only the last vector of a row can be partial; it is padded with NaNs (see kPadElem).
  const uint32_t num_vecs = div_ceil(seq_len, C::kVecSize);
  const uint32_t num_full = seq_len / C::kVecSize;
  const uint32_t vecs_per_thread = num_vecs / C::kBlockSize;
  const uint32_t vecs_rem = num_vecs % C::kBlockSize;
  const uint32_t vec_start = tx * vecs_per_thread + min(tx, vecs_rem);
  const uint32_t num_my = vecs_per_thread + (tx < vecs_rem ? 1 : 0);
  vec_t vecs[C::kMaxVecs];
#pragma unroll
  for (uint32_t i = 0; i < C::kMaxVecs; ++i) {
    if (i >= num_my) break;
    const auto v = vec_start + i;
    if (v < num_full) {
      vecs[i].load(scores_row, v);
    } else {
      const auto* ptr = reinterpret_cast<const uint16_t*>(scores_row) + v * C::kVecSize;
      const auto n = seq_len - v * C::kVecSize;  // in [1, kVecSize)
#pragma unroll
      for (uint32_t j = 0; j < C::kVecSize / 2; ++j) {
        vecs[i][j].x = bitcast<bf16_t>(2 * j + 0 < n ? ptr[2 * j + 0] : static_cast<uint16_t>(C::kPadElem));
        vecs[i][j].y = bitcast<bf16_t>(2 * j + 1 < n ? ptr[2 * j + 1] : static_cast<uint16_t>(C::kPadElem));
      }
    }
  }

  __syncthreads();

  // Pass 1: histogram of the raw high byte (sign + 7 exponent bits)
  const auto hist_msb = smem.histogram[0];
#pragma unroll
  for (uint32_t i = 0; i < C::kMaxVecs; ++i) {
    if (i >= num_my) break;
#pragma unroll
    for (uint32_t j = 0; j < C::kVecSize / 2; ++j) {
      const auto raw = bitcast<uint32_t>(vecs[i][j]);
      /// NOTE: spelled as byte extraction so the address is one PRMT + one LEA per element
      const auto b0 = __byte_perm(raw, 0, 0x4441);
      const auto b1 = __byte_perm(raw, 0, 0x4443);
      atomicAdd(hist_msb + b0 + (b0 >> 7) * C::kNegShift, 1);
      atomicAdd(hist_msb + b1 + (b1 >> 7) * C::kNegShift, 1);
    }
  }
  __syncthreads();

  const auto pivot_of = [&](const uint32_t* hist, uint32_t k, bool msb_mode, bool neg) -> TopKBF16Pivot {
    if (warp_id == 0) topk_bf16_find_pivot_warp(hist, k, msb_mode, neg, lane_id, smem);
    __syncthreads();
    return {smem.pivot_bin, smem.pivot_remain};
  };
  const auto msb = pivot_of(hist_msb, topk, true, false);
  const bool negative = msb.bin < 0x80;
  const auto pivot_hi = negative ? 0xFF - msb.bin : msb.bin - 0x80;  // raw high byte

  // Pass 2: among elements sharing the pivot's high byte, histogram the raw low byte. The high
  // bytes are compared as tiny positive bf16 values (exact), the others land in the sink bin.
  const auto hist_lsb = smem.histogram[1];
  const auto pivot_hi_x2 = bitcast<bf16x2_t>(pivot_hi << 16 | pivot_hi);
  constexpr uint32_t kSinkBinX2 = C::kSinkBin << 16 | C::kSinkBin;
#pragma unroll
  for (uint32_t i = 0; i < C::kMaxVecs; ++i) {
    if (i >= num_my) break;
#pragma unroll
    for (uint32_t j = 0; j < C::kVecSize / 2; ++j) {
      const auto raw = bitcast<uint32_t>(vecs[i][j]);
      const auto hi = __byte_perm(raw, 0, 0x5341);  // {byte1, 0, byte3, 0}
      const auto sel = __heq2_mask(bitcast<bf16x2_t>(hi), pivot_hi_x2);
      const auto lo = raw & 0x00FF00FFu;
      const auto bins = (sel & lo) | (~sel & kSinkBinX2);     // sel ? lo : kSinkBin
      atomicAdd(hist_lsb + __byte_perm(bins, 0, 0x4410), 1);  // bins & 0xFFFF
      atomicAdd(hist_lsb + __byte_perm(bins, 0, 0x4432), 1);  // bins >> 16
    }
  }
  __syncthreads();

  const auto lsb = pivot_of(hist_lsb, msb.remain, false, negative);
  const uint32_t pivot_lo = negative ? 0xFF - lsb.bin : lsb.bin;
  const uint32_t pivot_bits = pivot_hi << 8 | pivot_lo;
  const auto pivot_x2 = bitcast<bf16x2_t>(pivot_bits << 16 | pivot_bits);

  // Census: one bit per element of the slice, in element order (vector i fills byte i)
  uint32_t gt_mask = 0;
  uint32_t eq_mask = 0;
#pragma unroll
  for (uint32_t i = 0; i < C::kMaxVecs; ++i) {
    if (i >= num_my) break;
    uint32_t gt[4], eq[4];
#pragma unroll
    for (uint32_t j = 0; j < C::kVecSize / 2; ++j) {
      gt[j] = __hgt2_mask(vecs[i][j], pivot_x2);
      eq[j] = __heq2_mask(vecs[i][j], pivot_x2);
    }
    // drop the new byte into slot i, keeping the other three
    constexpr uint32_t kInsert[4] = {0x3214, 0x3240, 0x3410, 0x4210};
    gt_mask = __byte_perm(gt_mask, topk_bf16_pack_hits(gt), kInsert[i]);
    eq_mask = __byte_perm(eq_mask, topk_bf16_pack_hits(eq), kInsert[i]);
  }
  const uint32_t cnt_gt = __popc(gt_mask);
  const uint32_t cnt_eq = __popc(eq_mask);

  // Block-wide exclusive prefix of (gt, eq), packed: one warp scan plus one shared atomic per
  // warp. Warps land in arrival order, which is fine since the output is unordered.
  const uint32_t local = cnt_gt << 16 | cnt_eq;
  const uint32_t warp_inc = warp::inclusive_sum(lane_id, local);
  uint32_t warp_base = 0;
  if (lane_id == kWarpThreads - 1) warp_base = atomicAdd(&smem.count_gt_eq, warp_inc);
  warp_base = __shfl_sync(kFullMask, warp_base, kWarpThreads - 1);
  const uint32_t before = warp_base + warp_inc - local;

  // Everything above the pivot is taken, plus `remain` of the elements equal to it.
  uint32_t eq_total = lsb.remain;
  if (pivot_bits == C::kNegZeroBits) {
    /// NOTE: the census compares as floats, so a -0 pivot also sees +0 as equal while the
    /// histogram ranked +0 above it. Both are worth the same, so let the equal quota absorb
    /// them: the quota then has to come from the census total (one extra barrier, rare).
    __syncthreads();
    eq_total = topk - (smem.count_gt_eq >> 16);
  }
  const uint32_t gt_before = before >> 16;
  const uint32_t eq_before = before & 0xFFFF;
  const uint32_t eq_start = min(eq_before, eq_total);
  const uint32_t eq_quota = min(eq_before + cnt_eq, eq_total) - eq_start;
  // keep only `eq_quota` of the equal bits (which ones does not matter)
  if (eq_quota == 0) {
    eq_mask = 0;
  } else {
#pragma unroll 1
    for (uint32_t n = cnt_eq; n > eq_quota; --n) {
      eq_mask &= eq_mask - 1;
    }
  }

  uint32_t hits = gt_mask | eq_mask;
  auto* dst = smem.stage + gt_before + eq_start;
  const uint32_t elem_base = vec_start * C::kVecSize;
  while (hits != 0) {
    const auto e = __ffs(hits) - 1;
    hits &= hits - 1;
    *dst++ = elem_base + e;
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();

  // Slots past the census total were never staged. That only happens with NaN scores (the
  // histogram counts them, no ordered compare ever selects them); write -1 there rather than
  // whatever shared memory held before.
  const uint32_t totals = smem.count_gt_eq;
  const uint32_t num_staged = (totals >> 16) + min(totals & 0xFFFFu, eq_total);
  // TODO(perf): unroll once k regularly exceeds 512.
  for (uint32_t t = tx; t < topk; t += C::kBlockSize) {
    out[t] = t < num_staged ? transform(smem.stage[t]) : -1;
  }
}

/// Host entry: bf16 top-k over rows of at most kMaxSeqLen, selected indices
/// written through a per-row table as `table[i >> log2(page_size)] << log2(page_size)
/// | (i & mask)`, -1 past min(topk, seq_len).
template <bool kPDL>
struct TopKBF16Kernel {
  static void transform(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView page_indices,
      const uint32_t page_size) {
    using namespace host;
    using C = TopKBF16Config;
    auto B = SymbolicSize{"batch_size"};
    auto L = SymbolicSize{"max_seq_len"};
    auto S = SymbolicSize{"score_stride"};
    auto K = SymbolicSize{"topk"};
    auto O = SymbolicSize{"page_indices_stride"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();
    TensorMatcher({B, L})  // scores
        .with_strides({S, 1})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .verify(scores);
    TensorMatcher({B})  // seq_lens
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    TensorMatcher({B, -1})  // page_table
        .with_strides({-1, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(page_table);
    TensorMatcher({B, K})  // page_indices
        .with_strides({O, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(page_indices);
    CHECK_HOST(std::has_single_bit(page_size)) << "page_size must be a power of 2";
    CHECK_HOST(L.unwrap() <= C::kMaxSeqLen) << "rows longer than kMaxSeqLen take the streaming top-k";
    /// NOTE: a row base must stay aligned to the vector width, not just the tensor base.
    CHECK_HOST(S.unwrap() % C::kVecSize == 0) << "score_stride must keep every row vector-aligned";
    const auto topk = static_cast<uint32_t>(K.unwrap());
    CHECK_HOST(topk > 0 && topk <= C::kMaxTopK) << "topk must be in (0, " << C::kMaxTopK << "]";
    const auto params = TopKBF16Params{
        .scores = static_cast<const bf16_t*>(scores.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .score_stride = S.unwrap(),
        .page_table_stride = page_table.stride(0),
        .page_indices_stride = O.unwrap(),
        .topk = topk,
        .page_bits = static_cast<uint32_t>(std::countr_zero(page_size)),
    };
    LaunchKernel(static_cast<uint32_t>(B.unwrap()), C::kBlockSize, device_.unwrap())
        .config({.use_pdl = kPDL})
        .launch(topk_bf16_small_kernel<kPDL>, params);
  }
};

}  // namespace sglang
