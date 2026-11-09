// QSA top-512 (prefill and target-verify rows) with fast_topk's exact output:
// the same indices in the same slots for every row.
//
// fast_topk (fast_topk.cuh, deterministic_topk.cuh) returns, for a row longer
// than 4096 keys, the 512 largest ordered_key()s with ties at the cutoff taken
// by smallest index, emitted by its 1024-thread warp / lane / word-round / bit
// walk of the winner bitmap. This kernel finds the same winners from one
// vectorized pass over the row. A 2048-key sample, counted into bins of a
// monotone map of the key (hist_bin), places a floor below the 512th key as
// qsa_prefill_topk.cuh's sample does. The pass collects every key at or above
// it. The candidates' own bin counts give the boundary bin: keys of higher bins
// all win, and the boundary bin's keys are ranked by (key descending, index
// ascending), which is fast_topk's tie rule.
//
// A row whose candidates or boundary keys do not fit (heavy ties) is flagged,
// and the second launch runs qsa_prefill_topk_half.cuh's 1024-thread code (the
// sampled select, then fast_topk's own device function) on exactly those rows,
// as on rows of 4096 or fewer keys and rows past one 131072-key tile.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include "deterministic_topk.cuh"
#include "qsa_prefill_topk_half.cuh"

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

namespace sglang {

namespace qsa_topk_onepass_detail {

constexpr int kThreads = 512;
constexpr int kMinBlocks = 2;  // two CTAs per SM: 64 registers for kUnroll float4 loads in flight
constexpr int kUnroll = 4;     // VR200 (mb-h559-a1): 2 CTAs x 4 loads beat 4 CTAs x 2 at 192-3328 rows
constexpr int kWarps = kThreads / 32;
constexpr int kStockThreads = 1024;  // fast_topk's block: its emission order's unit
constexpr int kStockWarps = kStockThreads / 32;
constexpr int kVirtual = kStockWarps / kWarps;  // virtual warps walked per warp
constexpr int kTopK = 512;
constexpr int kCap = 2048;
constexpr int kBoundaryCap = 1024;
constexpr int kSample = 2048;
constexpr int kTile = 8192 * 16;  // deterministic_topk's single-tile limit
constexpr int kWords = kTile / 32;
constexpr int kWordRounds = kWords / kStockThreads;

// Bins of 1/64 octave over positive keys in [2^-16, 2^16); zero and negative keys
// share bin 0, larger keys bin kBins - 1. Arbitrary: any monotone map is exact,
// finer bins only keep the boundary bin small.
constexpr int kBins = 2048;
constexpr int kShift = 17;
constexpr int kBase = int(0x80000000u >> kShift) + (111 << 6);
static_assert(kBins == 4 * kThreads);

struct Smem {
  uint32_t cand_keys[kCap];
  uint32_t cand_ids[kCap];
  uint32_t bitmap[kWords];  // first the sample's, then the candidates' bins (kBins ints)
  uint32_t bnd_keys[kBoundaryCap];
  uint32_t bnd_ids[kBoundaryCap];
};
constexpr size_t kSmemBytes = sizeof(Smem);
static_assert(kBins * sizeof(int) <= sizeof(Smem::bitmap));

struct Scratch {
  int warp_total[kWarps];
  uint32_t warp_sel[kStockWarps];
  int bin, count, n_above, n_bnd;
};

SGL_DEVICE int hist_bin(uint32_t key) {
  return min(kBins - 1, max(0, int(key >> kShift) - kBase));
}

// The smallest key of a bin.
SGL_DEVICE uint32_t bin_floor(int bin) {
  return bin == 0 ? 0u : uint32_t(bin + kBase) << kShift;
}

SGL_DEVICE void zero_bins(int* bins) {
  for (int b = threadIdx.x; b < kBins; b += kThreads) bins[b] = 0;
  __syncthreads();
}

// s.bin = the bin holding the k-th largest key (1-based) by the counts bins[0, kBins),
// or -1 when they total less than k. Thread t reads bins 4t .. 4t + 3.
SGL_DEVICE void find_bin(const int* bins, int k, Scratch& s) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
  const int4 v = reinterpret_cast<const int4*>(bins)[tid];
  const int c[4] = {v.x, v.y, v.z, v.w};
  const int mine = c[0] + c[1] + c[2] + c[3];
  int suffix = mine;  // this lane's and the higher lanes' counts
#pragma unroll
  for (int off = 1; off < 32; off *= 2) {
    const int other = __shfl_down_sync(0xffffffffu, suffix, off);
    if (lane + off < 32) suffix += other;
  }
  if (lane == 0) s.warp_total[warp] = suffix;
  if (tid == 0) s.bin = -1;
  __syncthreads();
  int higher = suffix - mine;
  for (int w = warp + 1; w < kWarps; ++w) higher += s.warp_total[w];
  if (higher < k && k <= higher + mine) {
#pragma unroll
    for (int j = 3; j >= 0; --j) {
      if (higher < k && k <= higher + c[j]) s.bin = 4 * tid + j;
      higher += c[j];
    }
  }
  __syncthreads();
}

// Every key of row[0, length) at or above floor into the candidate arrays (the
// first kCap of them); s.count ends as the number found. s.count is zero on entry.
SGL_DEVICE void collect(const float* __restrict__ row, int length, uint32_t floor, Smem& m, Scratch& s) {
  using deterministic_topk_detail::ordered_key;
  const int tid = threadIdx.x, lane = tid & 31;
  // A scalar head up to 16-byte alignment and a scalar tail around the float4 body.
  const int head = min(length, int(((16 - (reinterpret_cast<uintptr_t>(row) & 15)) & 15) / 4));
  const int nvec = (length - head) / 4;
  const int tail = head + 4 * nvec;
  if (tid < head + (length - tail)) {
    const int id = tid < head ? tid : tail + (tid - head);
    const uint32_t key = ordered_key(row[id]);
    if (key >= floor) {
      const int at = atomicAdd(&s.count, 1);
      if (at < kCap) {
        m.cand_keys[at] = key;
        m.cand_ids[at] = uint32_t(id);
      }
    }
  }
  const float4* vec = reinterpret_cast<const float4*>(row + head);
  for (int base = 0; base < nvec; base += kUnroll * kThreads) {
    float4 v[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int i = base + u * kThreads + tid;
      v[u] = i < nvec ? __ldcg(vec + i) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int i = base + u * kThreads + tid;
      const bool ok = i < nvec;
      const float f[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
      uint32_t keys[4];
      unsigned masks[4];
      int total = 0;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        keys[e] = ordered_key(f[e]);
        masks[e] = __ballot_sync(0xffffffffu, ok && keys[e] >= floor);
        total += __popc(masks[e]);
      }
      if (total == 0) continue;  // warp-uniform
      int slot = 0;
      if (lane == 0) slot = atomicAdd(&s.count, total);
      slot = __shfl_sync(0xffffffffu, slot, 0);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        if ((masks[e] >> lane) & 1u) {
          const int at = slot + __popc(masks[e] & ((1u << lane) - 1u));
          if (at < kCap) {
            m.cand_keys[at] = keys[e];
            m.cand_ids[at] = uint32_t(head + 4 * i + e);
          }
        }
        slot += __popc(masks[e]);
      }
    }
  }
}

// The winners into the bitmap: candidates at or above hi all win, and of the
// boundary keys [lo, hi) the ones fewer than `need` keys beat by (key
// descending, index ascending). False (CTA-uniform) when they do not fit.
SGL_DEVICE bool mark_winners(int found, int length, uint32_t lo, bool has_hi, uint32_t hi, Smem& m, Scratch& s) {
  const int tid = threadIdx.x, lane = tid & 31;
  const int words = (length + 31) / 32;
  for (int w = tid; w < words; w += kThreads) m.bitmap[w] = 0u;
  if (tid == 0) s.n_above = s.n_bnd = 0;
  __syncthreads();
  for (int base = 0; base < found; base += kThreads) {
    const int i = base + tid;
    const bool valid = i < found;
    const uint32_t key = valid ? m.cand_keys[i] : 0u, id = valid ? m.cand_ids[i] : 0u;
    const bool above = valid && has_hi && key >= hi;
    const bool bnd = valid && !above && key >= lo;
    if (above) atomicOr(&m.bitmap[id >> 5], 1u << (id & 31u));
    const unsigned a = __ballot_sync(0xffffffffu, above), b = __ballot_sync(0xffffffffu, bnd);
    int slot = 0;
    if (lane == 0) {
      if (a) atomicAdd(&s.n_above, __popc(a));
      if (b) slot = atomicAdd(&s.n_bnd, __popc(b));
    }
    slot = __shfl_sync(0xffffffffu, slot, 0) + __popc(b & ((1u << lane) - 1u));
    if (bnd && slot < kBoundaryCap) {
      m.bnd_keys[slot] = key;
      m.bnd_ids[slot] = id;
    }
  }
  __syncthreads();
  const int need = kTopK - s.n_above, n_bnd = s.n_bnd;
  if (need < 0 || n_bnd < need || n_bnd > kBoundaryCap) return false;
  for (int i = tid; i < n_bnd; i += kThreads) {
    const uint32_t key = m.bnd_keys[i], id = m.bnd_ids[i];
    int better = 0;
    if (n_bnd > need) {
      for (int j = 0; j < n_bnd; ++j) {
        const uint32_t other = m.bnd_keys[j];
        better += (other > key || (other == key && m.bnd_ids[j] < id)) ? 1 : 0;
      }
    }
    if (better < need) atomicOr(&m.bitmap[id >> 5], 1u << (id & 31u));
  }
  __syncthreads();
  return true;
}

// fast_topk's 1024-thread emission of the winner bitmap: virtual warp v reads words
// v * rounds * 32 + r * 32 + lane and emits after all lower virtual warps, lane by
// lane, round by round (qsa_prefill_topk_half.cuh's walk).
SGL_DEVICE void emit(int length, int32_t* __restrict__ output, const Smem& m, Scratch& s) {
  using deterministic_topk_detail::warp_exclusive;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
  const int words = (length + 31) / 32;
  const int rounds = (words + kStockThreads - 1) / kStockThreads;
  uint32_t lane_base[kVirtual];
#pragma unroll
  for (int j = 0; j < kVirtual; ++j) {
    const int v = warp * kVirtual + j;
    uint32_t count = 0;
#pragma unroll
    for (int r = 0; r < kWordRounds; ++r) {
      const int word = v * rounds * 32 + r * 32 + lane;
      if (r < rounds && word < words) count += __popc(m.bitmap[word]);
    }
    lane_base[j] = warp_exclusive(count);
    if (lane == 31) s.warp_sel[v] = lane_base[j] + count;
  }
  __syncthreads();
  if (tid < 32) {
    const uint32_t count = s.warp_sel[lane];
    s.warp_sel[lane] = warp_exclusive(count);
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < kVirtual; ++j) {
    const int v = warp * kVirtual + j;
    uint32_t slot = s.warp_sel[v] + lane_base[j];
#pragma unroll
    for (int r = 0; r < kWordRounds; ++r) {
      const int word = v * rounds * 32 + r * 32 + lane;
      uint32_t bits = (r < rounds && word < words) ? m.bitmap[word] : 0u;
      while (bits) {
        const int bit = __ffs(bits) - 1;
        output[slot++] = word * 32 + bit;
        bits &= bits - 1;
      }
    }
  }
}

// One row with 4096 < length <= kTile; false (CTA-uniform, output untouched) when
// the row needs the 1024-thread pass.
SGL_DEVICE bool select_row(const float* __restrict__ row, int32_t* __restrict__ output, int length, Smem& m,
                           Scratch& s) {
  using deterministic_topk_detail::ordered_key;
  const int tid = threadIdx.x;
  int* bins = reinterpret_cast<int*>(m.bitmap);
  // qsa_prefill_topk.cuh's sample and rank (twice the expected rank of the 512th key,
  // plus slack), taken at bin resolution: the floor is that bin's.
  zero_bins(bins);
  for (int i = tid; i < kSample; i += kThreads) {
    atomicAdd(&bins[hist_bin(ordered_key(row[static_cast<int64_t>(i) * length / kSample]))], 1);
  }
  __syncthreads();
  const int rank = min(kSample, static_cast<int>((2LL * kTopK * kSample) / length) + 9);
  find_bin(bins, rank, s);
  const uint32_t floor = bin_floor(s.bin);  // s.bin >= 0: the sample totals kSample >= rank
  if (tid == 0) s.count = 0;
  __syncthreads();
  collect(row, length, floor, m, s);
  __syncthreads();
  const int found = s.count;
  if (found < kTopK || found > kCap) return false;
  // The candidates' own bins: exact counts for every bin at or above the floor.
  zero_bins(bins);
  for (int i = tid; i < found; i += kThreads) atomicAdd(&bins[hist_bin(m.cand_keys[i])], 1);
  __syncthreads();
  find_bin(bins, kTopK, s);
  const int bin = s.bin;  // >= 0: found >= kTopK
  const bool has_hi = bin < kBins - 1;
  if (!mark_winners(found, length, bin_floor(bin), has_hi, has_hi ? bin_floor(bin + 1) : 0u, m, s)) return false;
  emit(length, output, m, s);
  return true;
}

template <bool kUsePDL>
__global__ __launch_bounds__(kThreads, kMinBlocks) void qsa_topk_onepass_kernel(
    const qsa_prefill_topk_half_detail::Params __grid_constant__ p) {
  extern __shared__ __align__(16) uint8_t smem[];
  __shared__ Scratch scratch;
  device::PDLWaitPrimary<kUsePDL>();
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const int row_start = p.row_starts[bid];
  const int length = p.lengths[bid];
  int32_t* indice = p.indices + bid * kTopK;
  bool rest = false;
  if (length <= kTopK) {
    for (int i = threadIdx.x; i < kTopK; i += kThreads) indice[i] = (i < length) ? i : -1;
  } else if (length <= 4096 || length > kTile) {
    rest = true;
  } else {
    rest = !select_row(p.input + bid * p.input_stride + row_start, indice, length, *reinterpret_cast<Smem*>(smem),
                       scratch);
  }
  if (threadIdx.x == 0) p.fallback[bid] = rest ? 1 : 0;
  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace qsa_topk_onepass_detail

/**
 * \brief fast_topk<512>'s output (same indices, same slots) from one pass over each row
 * longer than 4096 keys, plus a 1024-thread pass over the rows it leaves (flagged in
 * fallback[b], which is scratch).
 *
 * Row b selects from score[b, row_starts[b] : row_starts[b] + lengths[b]) and writes
 * indices relative to row_starts[b] into indices[b] (-1 past lengths[b] < 512).
 */
template <bool kUsePDL>
struct QsaTopKOnePassKernel {
  static void
  run(const tvm::ffi::TensorView score,
      const tvm::ffi::TensorView row_starts,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView fallback) {
    using namespace host;
    namespace d = qsa_topk_onepass_detail;
    namespace half = qsa_prefill_topk_half_detail;
    auto B = SymbolicSize{"batch"};
    auto L = SymbolicSize{"length"};
    auto S = SymbolicSize{"input_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({B, L}).with_strides({S, 1}).with_dtype<fp32_t>().with_device(device).verify(score);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(row_starts);
    TensorMatcher({B, d::kTopK}).with_dtype<int32_t>().with_device(device).verify(indices);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(fallback);
    const auto params = half::Params{
        .input = static_cast<const float*>(score.data_ptr()),
        .row_starts = static_cast<const int32_t*>(row_starts.data_ptr()),
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .fallback = static_cast<int32_t*>(fallback.data_ptr()),
        .input_stride = S.unwrap(),
    };
    const auto num_rows = static_cast<uint32_t>(B.unwrap());
    LaunchKernel(num_rows, d::kThreads, device.unwrap(), d::kSmemBytes)
        .enable_pdl(kUsePDL)(d::qsa_topk_onepass_kernel<kUsePDL>, params);
    LaunchKernel(num_rows, half::kStockThreads, device.unwrap(), half::kSmemBytes)
        .enable_pdl(kUsePDL)(half::qsa_prefill_topk_rest_kernel<kUsePDL>, params);
  }
};

}  // namespace sglang
