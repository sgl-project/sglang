// DCP sparse-attention top-2048 on AMD (wave64), one 1024-thread block per query row.
//
// Local: rank r's indexer logits row (only its first length entries are scores) -> its top-2048
// fp32 scores, and their global positions i * world + r kept on the rank (-1 padded). Global: the
// all-gathered scores of every rank -> the global top-2048, of which a rank emits only its own
// winners, as KV slots through the page table. The global top-k of the union is the top-k of the
// per-rank top-k sets, so both steps are exact; every rank selects over the same gathered scores
// with the same tie order, so the ranks' winners are disjoint and together form that one set.
//
// Selection is an exact radix select on key - row_min (key = order-preserving uint32 of the score):
// the digits start at the highest bit that varies within the row, so scores sharing an exponent
// still spread over the histogram instead of contending for a few LDS bins. Ties at the threshold
// resolve in a fixed thread order (deterministic, not by position). Lengths are read on the device,
// so one launch serves every row length under graph replay.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace dcp_topk_hip_detail {

constexpr int kThreads = 1024;
constexpr int kWaves = kThreads / 64;
constexpr int kTopK = 2048;
constexpr int kDigitBits = 11;
constexpr int kBins = 1 << kDigitBits;
// -inf: a padded score loses to every real one
constexpr uint32_t kNegInfBits = 0xFF800000u;

struct Smem {
  uint32_t hist[2][kBins];
  uint32_t wave_sum[kWaves];
  uint32_t wave_min[kWaves];
  uint32_t wave_max[kWaves];
  uint32_t wave_cnt[kWaves];
  uint32_t wave_sum_b[kWaves];  // emit's second count: wave_cnt may still be read when the search returns early
  uint32_t pick[3];
};

// Larger float -> larger key; NaNs with the sign bit set sort below -inf.
SGL_DEVICE uint32_t ordered_key(float x) {
  const uint32_t b = __float_as_uint(x);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

SGL_DEVICE uint32_t key_to_bits(uint32_t k) {
  return (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
}

// u >> shift for shift in [0, 32].
SGL_DEVICE uint32_t high_bits(uint32_t u, int shift) {
  return shift >= 32 ? 0u : (u >> shift);
}

// Barrier for LDS hand-offs only: __syncthreads() would also drain outstanding global loads.
SGL_DEVICE void lds_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

SGL_DEVICE uint32_t wave_min(uint32_t v) {
#pragma unroll
  for (int o = 32; o > 0; o >>= 1)
    v = min(v, static_cast<uint32_t>(__shfl_xor(v, o)));
  return v;
}

SGL_DEVICE uint32_t wave_max(uint32_t v) {
#pragma unroll
  for (int o = 32; o > 0; o >>= 1)
    v = max(v, static_cast<uint32_t>(__shfl_xor(v, o)));
  return v;
}

SGL_DEVICE uint32_t wave_sum(uint32_t v) {
#pragma unroll
  for (int o = 32; o > 0; o >>= 1)
    v += __shfl_xor(v, o);
  return v;
}

// Inclusive scan of one value per thread over the block, in thread order; *excl gets the exclusive prefix.
SGL_DEVICE uint32_t block_inclusive_scan(uint32_t v, Smem& sm, uint32_t* excl) {
  const int lane = threadIdx.x & 63;
  const int wave = threadIdx.x >> 6;
  uint32_t x = v;
#pragma unroll
  for (int o = 1; o < 64; o <<= 1) {
    const uint32_t y = __shfl_up(x, o);
    if (lane >= o) x += y;
  }
  if (lane == 63) sm.wave_sum[wave] = x;
  lds_barrier();
  uint32_t base = 0;
#pragma unroll
  for (int w = 0; w < kWaves; ++w)
    base += w < wave ? sm.wave_sum[w] : 0u;
  *excl = base + x - v;
  return base + x;
}

// Exclusive scans of two counts at once (packed in 64 bits, so one pass over the waves); totals in *ta, *tb.
SGL_DEVICE void
block_exclusive_scan2(uint32_t a, uint32_t b, Smem& sm, uint32_t* oa, uint32_t* ob, uint32_t* ta, uint32_t* tb) {
  const int lane = threadIdx.x & 63;
  const int wave = threadIdx.x >> 6;
  uint64_t v = (static_cast<uint64_t>(a) << 32) | b;
  uint64_t x = v;
#pragma unroll
  for (int o = 1; o < 64; o <<= 1) {
    const uint64_t y = __shfl_up(x, o);
    if (lane >= o) x += y;
  }
  if (lane == 63) {
    sm.wave_sum[wave] = static_cast<uint32_t>(x >> 32);
    sm.wave_sum_b[wave] = static_cast<uint32_t>(x);
  }
  lds_barrier();
  uint32_t base_a = 0, base_b = 0, sum_a = 0, sum_b = 0;
#pragma unroll
  for (int w = 0; w < kWaves; ++w) {
    const uint32_t s_a = sm.wave_sum[w], s_b = sm.wave_sum_b[w];
    base_a += w < wave ? s_a : 0u;
    base_b += w < wave ? s_b : 0u;
    sum_a += s_a;
    sum_b += s_b;
  }
  const uint64_t ex = x - v;
  *oa = base_a + static_cast<uint32_t>(ex >> 32);
  *ob = base_b + static_cast<uint32_t>(ex);
  *ta = sum_a;
  *tb = sum_b;
}

// The digit whose descending suffix count first reaches `remaining`, by the whole block: thread t owns
// bins bins-1-2t and bins-2-2t. Writes pick = {digit, count above it, count in it}; the caller barriers.
SGL_DEVICE void pick_digit(const uint32_t* hist, Smem& sm, int bins, uint32_t remaining) {
  const int hi = bins - 1 - 2 * static_cast<int>(threadIdx.x);
  const uint32_t c_hi = hi >= 0 ? hist[hi] : 0u;
  const uint32_t c_lo = hi >= 1 ? hist[hi - 1] : 0u;
  uint32_t excl;
  const uint32_t incl = block_inclusive_scan(c_hi + c_lo, sm, &excl);
  if (hi >= 0 && excl < remaining && incl >= remaining) {
    if (excl + c_hi >= remaining) {
      sm.pick[0] = hi;
      sm.pick[1] = excl;
      sm.pick[2] = c_hi;
    } else {
      sm.pick[0] = hi - 1;
      sm.pick[1] = excl + c_hi;
      sm.pick[2] = c_lo;
    }
  }
}

struct Threshold {
  uint32_t kmin;
  int shift;
  uint32_t prefix;
  uint32_t remaining;
};

// Top-kTopK set of the valid keys `keys.for_each` enumerates: keys whose high_bits(key - kmin,
// shift) exceed `prefix` are in, and the first `remaining` keys equal to it complete the set.
// With at most kTopK valid keys every key is in (shift 32, prefix 0).
template <class Keys>
SGL_DEVICE Threshold select_threshold(Keys& keys, uint32_t& nvalid, Smem& sm) {
  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int wave = tid >> 6;
  uint32_t lo = 0xFFFFFFFFu, hi = 0, cnt = 0;
  keys.for_each([&](uint32_t k, int) {
    lo = min(lo, k);
    hi = max(hi, k);
    ++cnt;
  });
  lo = wave_min(lo);
  hi = wave_max(hi);
  cnt = wave_sum(cnt);
  if (lane == 0) {
    sm.wave_min[wave] = lo;
    sm.wave_max[wave] = hi;
    sm.wave_cnt[wave] = cnt;
  }
  for (int b = tid; b < kBins; b += kThreads)
    sm.hist[0][b] = 0;
  lds_barrier();
  lo = 0xFFFFFFFFu;
  hi = 0;
  cnt = 0;
#pragma unroll
  for (int w = 0; w < kWaves; ++w) {
    lo = min(lo, sm.wave_min[w]);
    hi = max(hi, sm.wave_max[w]);
    cnt += sm.wave_cnt[w];
  }
  nvalid = cnt;
  Threshold t{lo, 32, 0u, min(cnt, static_cast<uint32_t>(kTopK))};
  if (cnt <= static_cast<uint32_t>(kTopK)) return t;

  const uint32_t range = hi - lo;
  t.shift = range == 0 ? 0 : 32 - __clz(range);
  for (int pass = 0; t.shift > 0; ++pass) {
    const int width = min(kDigitBits, t.shift);
    const int s = t.shift - width;
    const uint32_t mask = (1u << width) - 1u;
    uint32_t* hist = sm.hist[pass & 1];
    // the next pass's buffer was last read by the previous pass's pick, which two barriers retired
    for (int b = tid; b < kBins; b += kThreads)
      sm.hist[(pass + 1) & 1][b] = 0;
    keys.for_each([&](uint32_t k, int) {
      const uint32_t u = k - t.kmin;
      if (high_bits(u, t.shift) == t.prefix) atomicAdd(&hist[(u >> s) & mask], 1u);
    });
    lds_barrier();
    pick_digit(hist, sm, static_cast<int>(mask) + 1, t.remaining);
    lds_barrier();
    const uint32_t digit = sm.pick[0];
    const uint32_t above = sm.pick[1];
    const uint32_t in_digit = sm.pick[2];
    t.prefix = (t.prefix << width) | digit;
    t.remaining -= above;
    t.shift = s;
    // the threshold bin holds exactly the keys still needed: take it whole
    if (in_digit == t.remaining || s == 0) break;
  }
  return t;
}

// Writes slot e < min(nvalid, kTopK) of the selected set through emit(e, key, id) and pads the rest
// through pad(e).
template <class Keys, class Emit, class Pad>
SGL_DEVICE void emit_selected(Keys& keys, const Threshold& t, uint32_t nvalid, Smem& sm, Emit emit, Pad pad) {
  uint32_t greater = 0, equal = 0;
  keys.for_each([&](uint32_t k, int) {
    const uint32_t m = high_bits(k - t.kmin, t.shift);
    greater += m > t.prefix;
    equal += m == t.prefix;
  });
  uint32_t total_greater, total_equal, og, oe;
  block_exclusive_scan2(greater, equal, sm, &og, &oe, &total_greater, &total_equal);
  keys.for_each([&](uint32_t k, int id) {
    const uint32_t m = high_bits(k - t.kmin, t.shift);
    if (m > t.prefix) {
      emit(og++, k, id);
    } else if (m == t.prefix) {
      if (oe < t.remaining) emit(total_greater + oe, k, id);
      ++oe;
    }
  });
  const int filled = static_cast<int>(min(nvalid, static_cast<uint32_t>(kTopK)));
  for (int e = filled + static_cast<int>(threadIdx.x); e < kTopK; e += kThreads)
    pad(e);
}

// The first n scores of a logits row: kPer per thread in registers, the rest re-read per pass.
template <int kPer>
struct RowKeys {
  uint32_t key[kPer];
  const float* s;
  int n;

  SGL_DEVICE void load() {
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int i = j * kThreads + static_cast<int>(threadIdx.x);
      key[j] = i < n ? ordered_key(s[i]) : 0u;
    }
  }

  template <class F>
  SGL_DEVICE void for_each(F&& f) {
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int i = j * kThreads + static_cast<int>(threadIdx.x);
      if (i < n) f(key[j], i);
    }
    for (int i = kPer * kThreads + static_cast<int>(threadIdx.x); i < n; i += kThreads)
      f(ordered_key(s[i]), i);
  }
};

// The gathered scores of one row: kWorld ranks x kTopK.
template <int kWorld>
struct CandidateKeys {
  static constexpr int kPer = kWorld * kTopK / kThreads;
  static_assert(kPer <= 32, "validity is one 32-bit mask per thread");
  uint32_t key[kPer];
  uint32_t valid;

  template <class F>
  SGL_DEVICE void for_each(F&& f) {
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      if ((valid >> j) & 1u) f(key[j], j);
    }
  }
};

template <int kPer>
__global__ __launch_bounds__(kThreads) void dcp_local_candidates_kernel(
    const float* __restrict__ logits,
    int64_t stride,
    const int32_t* __restrict__ lengths,
    float* __restrict__ scores,
    int32_t* __restrict__ positions,
    int world,
    int rank) {
  __shared__ Smem sm;
  const int64_t row = blockIdx.x;
  float* os = scores + row * kTopK;
  int32_t* op = positions + row * kTopK;
  RowKeys<kPer> keys;
  keys.s = logits + row * stride;
  keys.n = lengths[row];
  if (keys.n <= kTopK) {
    for (int e = threadIdx.x; e < kTopK; e += kThreads) {
      const bool in = e < keys.n;
      os[e] = in ? keys.s[e] : __uint_as_float(kNegInfBits);
      op[e] = in ? e * world + rank : -1;
    }
    return;
  }
  keys.load();
  uint32_t nvalid;
  const Threshold t = select_threshold(keys, nvalid, sm);
  emit_selected(
      keys,
      t,
      nvalid,
      sm,
      [&](uint32_t e, uint32_t k, int i) {
        os[e] = __uint_as_float(key_to_bits(k));
        op[e] = i * world + rank;
      },
      [&](int e) {
        os[e] = __uint_as_float(kNegInfBits);
        op[e] = -1;
      });
}

// Chunked local step for rows longer than one block holds in registers: block (row, c) takes the
// top-2048 of chunk c (kPer * kThreads scores; the last chunk takes the rest of the row). A row of
// one chunk writes its result directly; longer rows write per-chunk candidates for the merge kernel.
// The top-k of the union of the chunks' top-k sets is the row's top-k, so the result is exact.
template <int kPer>
__global__ __launch_bounds__(kThreads) void dcp_local_chunk_kernel(
    const float* __restrict__ logits,
    int64_t stride,
    const int32_t* __restrict__ lengths,
    float* __restrict__ scores,
    int32_t* __restrict__ positions,
    float* __restrict__ chunk_scores,
    int32_t* __restrict__ chunk_positions,
    int max_chunks,
    int world,
    int rank) {
  constexpr int kChunk = kPer * kThreads;
  __shared__ Smem sm;
  const int64_t row = blockIdx.x;
  const int c = static_cast<int>(blockIdx.y);
  const int len = lengths[row];
  const int nch = min(max((len + kChunk - 1) / kChunk, 1), max_chunks);
  if (c >= nch) return;
  const int begin = c * kChunk;
  const int end = c == nch - 1 ? len : begin + kChunk;
  float* os;
  int32_t* op;
  if (nch == 1) {
    os = scores + row * kTopK;
    op = positions + row * kTopK;
  } else {
    os = chunk_scores + (row * max_chunks + c) * kTopK;
    op = chunk_positions + (row * max_chunks + c) * kTopK;
  }
  RowKeys<kPer> keys;
  keys.s = logits + row * stride + begin;
  keys.n = max(end - begin, 0);
  if (keys.n <= kTopK) {
    for (int e = threadIdx.x; e < kTopK; e += kThreads) {
      const bool in = e < keys.n;
      os[e] = in ? keys.s[e] : __uint_as_float(kNegInfBits);
      op[e] = in ? (begin + e) * world + rank : -1;
    }
    return;
  }
  keys.load();
  uint32_t nvalid;
  const Threshold t = select_threshold(keys, nvalid, sm);
  emit_selected(
      keys,
      t,
      nvalid,
      sm,
      [&](uint32_t e, uint32_t k, int i) {
        os[e] = __uint_as_float(key_to_bits(k));
        op[e] = (begin + i) * world + rank;
      },
      [&](int e) {
        os[e] = __uint_as_float(kNegInfBits);
        op[e] = -1;
      });
}

// Merge for rows of more than one chunk: the top-2048 of the chunks' candidates (nch * 2048 scores,
// contiguous per row), carrying each winner's global position.
template <int kPer>
__global__ __launch_bounds__(kThreads) void dcp_local_merge_kernel(
    const int32_t* __restrict__ lengths,
    const float* __restrict__ chunk_scores,
    const int32_t* __restrict__ chunk_positions,
    float* __restrict__ scores,
    int32_t* __restrict__ positions,
    int max_chunks) {
  constexpr int kChunk = kPer * kThreads;
  __shared__ Smem sm;
  const int64_t row = blockIdx.x;
  const int len = lengths[row];
  const int nch = min(max((len + kChunk - 1) / kChunk, 1), max_chunks);
  if (nch == 1) return;
  const int64_t base = row * max_chunks * kTopK;
  float* os = scores + row * kTopK;
  int32_t* op = positions + row * kTopK;
  RowKeys<kPer> keys;
  keys.s = chunk_scores + base;
  keys.n = nch * kTopK;
  keys.load();
  uint32_t nvalid;
  const Threshold t = select_threshold(keys, nvalid, sm);
  emit_selected(
      keys,
      t,
      nvalid,
      sm,
      [&](uint32_t e, uint32_t k, int i) {
        os[e] = __uint_as_float(key_to_bits(k));
        op[e] = chunk_positions[base + i];
      },
      [&](int e) {
        os[e] = __uint_as_float(kNegInfBits);
        op[e] = -1;
      });
}

template <int kWorld>
__global__ __launch_bounds__(kThreads) void dcp_global_slots_kernel(
    const float* __restrict__ gathered,
    const int32_t* __restrict__ positions,
    int rows,
    int rank,
    const int32_t* __restrict__ global_lens,
    const int32_t* __restrict__ page_table,
    int64_t page_table_stride,
    int span,
    int32_t* __restrict__ out) {
  using Keys = CandidateKeys<kWorld>;
  __shared__ Smem sm;
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int len = global_lens[row];
  Keys keys;
  keys.valid = 0;
#pragma unroll
  for (int j = 0; j < Keys::kPer; ++j) {
    const int c = j * kThreads + tid;
    const int r = c / kTopK;
    const int e = c % kTopK;
    // a rank holds positions r, r + world, ...; gathers leave entries past its count undefined
    const int n = min((len + kWorld - 1 - r) / kWorld, kTopK);
    keys.key[j] = 0;
    if (e < n) {
      keys.key[j] = ordered_key(gathered[(static_cast<int64_t>(r) * rows + row) * kTopK + e]);
      keys.valid |= 1u << j;
    }
  }
  uint32_t nvalid;
  const Threshold t = select_threshold(keys, nvalid, sm);
  const int32_t* pt = page_table + static_cast<int64_t>(row) * page_table_stride;
  const int32_t* pos = positions + static_cast<int64_t>(row) * kTopK;
  int32_t* o = out + static_cast<int64_t>(row) * kTopK;
  emit_selected(
      keys,
      t,
      nvalid,
      sm,
      [&](uint32_t e, uint32_t, int j) {
        const int c = j * kThreads + tid;
        if (c / kTopK == rank) {
          const int p = pos[c % kTopK];
          o[e] = pt[p / span] * span + p % span;
        } else {
          o[e] = -1;
        }
      },
      [&](int e) { o[e] = -1; });
}

}  // namespace dcp_topk_hip_detail

template <int kPer>
struct DcpLocalCandidatesHip {
  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView positions,
      const int64_t world,
      const int64_t rank) {
    using namespace host;
    using namespace dcp_topk_hip_detail;
    const int64_t rows = lengths.size(0);
    RuntimeCheck(logits.stride(1) == 1, "logits rows must be contiguous");
    RuntimeCheck(logits.size(0) >= rows, "one logits row per length");
    RuntimeCheck(
        scores.size(0) == rows && scores.size(1) == kTopK && scores.is_contiguous(),
        "scores must be a dense [rows, 2048]");
    RuntimeCheck(
        positions.size(0) == rows && positions.size(1) == kTopK && positions.is_contiguous(),
        "positions must be a dense [rows, 2048]");
    if (rows == 0) return;
    LaunchKernel(static_cast<uint32_t>(rows), kThreads, logits.device())(
        dcp_local_candidates_kernel<kPer>,
        static_cast<const float*>(logits.data_ptr()),
        logits.stride(0),
        static_cast<const int32_t*>(lengths.data_ptr()),
        static_cast<float*>(scores.data_ptr()),
        static_cast<int32_t*>(positions.data_ptr()),
        static_cast<int>(world),
        static_cast<int>(rank));
  }
};

template <int kPer>
struct DcpLocalCandidatesChunkedHip {
  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView chunk_scores,
      const tvm::ffi::TensorView chunk_positions,
      const int64_t world,
      const int64_t rank) {
    using namespace host;
    using namespace dcp_topk_hip_detail;
    const int64_t rows = lengths.size(0);
    const int64_t max_chunks = chunk_scores.size(1);
    RuntimeCheck(logits.stride(1) == 1, "logits rows must be contiguous");
    RuntimeCheck(logits.size(0) >= rows, "one logits row per length");
    RuntimeCheck(
        scores.size(0) == rows && scores.size(1) == kTopK && scores.is_contiguous(),
        "scores must be a dense [rows, 2048]");
    RuntimeCheck(
        positions.size(0) == rows && positions.size(1) == kTopK && positions.is_contiguous(),
        "positions must be a dense [rows, 2048]");
    RuntimeCheck(
        chunk_scores.size(0) == rows && chunk_scores.size(2) == kTopK && chunk_scores.is_contiguous(),
        "chunk_scores must be a dense [rows, max_chunks, 2048]");
    RuntimeCheck(
        chunk_positions.size(0) == rows && chunk_positions.size(1) == max_chunks && chunk_positions.size(2) == kTopK &&
            chunk_positions.is_contiguous(),
        "chunk_positions must be a dense [rows, max_chunks, 2048]");
    // the merge holds every candidate of a row in registers
    RuntimeCheck(max_chunks >= 1 && max_chunks * kTopK <= kPer * kThreads, "max_chunks out of range");
    if (rows == 0) return;
    LaunchKernel(dim3(static_cast<uint32_t>(rows), static_cast<uint32_t>(max_chunks)), kThreads, logits.device())(
        dcp_local_chunk_kernel<kPer>,
        static_cast<const float*>(logits.data_ptr()),
        logits.stride(0),
        static_cast<const int32_t*>(lengths.data_ptr()),
        static_cast<float*>(scores.data_ptr()),
        static_cast<int32_t*>(positions.data_ptr()),
        static_cast<float*>(chunk_scores.data_ptr()),
        static_cast<int32_t*>(chunk_positions.data_ptr()),
        static_cast<int>(max_chunks),
        static_cast<int>(world),
        static_cast<int>(rank));
    if (max_chunks == 1) return;
    LaunchKernel(static_cast<uint32_t>(rows), kThreads, logits.device())(
        dcp_local_merge_kernel<kPer>,
        static_cast<const int32_t*>(lengths.data_ptr()),
        static_cast<const float*>(chunk_scores.data_ptr()),
        static_cast<const int32_t*>(chunk_positions.data_ptr()),
        static_cast<float*>(scores.data_ptr()),
        static_cast<int32_t*>(positions.data_ptr()),
        static_cast<int>(max_chunks));
  }
};

template <int kWorld>
struct DcpGlobalSlotsHip {
  static void
  run(const tvm::ffi::TensorView gathered,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView global_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView out,
      const int64_t span,
      const int64_t rank) {
    using namespace host;
    using namespace dcp_topk_hip_detail;
    const int64_t rows = out.size(0);
    RuntimeCheck(
        gathered.size(0) == kWorld * rows && gathered.size(1) == kTopK && gathered.is_contiguous(),
        "gathered must be a dense rank-major [world * rows, 2048]");
    RuntimeCheck(
        positions.size(0) == rows && positions.size(1) == kTopK && positions.is_contiguous(),
        "positions must be this rank's dense [rows, 2048]");
    RuntimeCheck(rank >= 0 && rank < kWorld, "rank out of range");
    RuntimeCheck(global_lens.size(0) >= rows, "one length per row");
    RuntimeCheck(page_table.stride(1) == 1 && page_table.size(0) >= rows, "page table rows must be contiguous");
    RuntimeCheck(out.size(1) == kTopK && out.is_contiguous(), "out must be a dense [rows, 2048]");
    if (rows == 0) return;
    LaunchKernel(static_cast<uint32_t>(rows), kThreads, out.device())(
        dcp_global_slots_kernel<kWorld>,
        static_cast<const float*>(gathered.data_ptr()),
        static_cast<const int32_t*>(positions.data_ptr()),
        static_cast<int>(rows),
        static_cast<int>(rank),
        static_cast<const int32_t*>(global_lens.data_ptr()),
        static_cast<const int32_t*>(page_table.data_ptr()),
        page_table.stride(0),
        static_cast<int>(span),
        static_cast<int32_t*>(out.data_ptr()));
  }
};

}  // namespace sglang
