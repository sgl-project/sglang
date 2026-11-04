// Exact global top-k merge for DSA decode context parallelism (DCP).
//
// Every rank holds the local top-k candidates of its token shard (the v2
// candidate epilogue: fp32 score bits, global position; position -1 is empty).
// One CTA per row:
//   1. pushes the rank's best kCandK candidates to every rank through NVLink
//      multicast, and keeps its full local top-k for peers' fallback reads;
//   2. polls the kNumRanks x kCandK candidates of the row;
//   3. selects the k-th key t* among them (score descending, then position
//      ascending: a strict total order over 64-bit keys);
//   4. checks the certificate: a rank may hold more of the global top-k only if
//      it sent kCandK candidates, has more keys, and all of them reach t*. Then
//      its full local top-k replaces its candidates and t* is selected again,
//      which is exact because a rank holds at most k global top-k keys;
//   5. writes this rank's keys >= t* as widened KV slots, -1 after them.
//
// Exchange: every 64-bit entry carries an 8-bit epoch tag next to its payload
// (score bits << 32 | position << 8 | tag), written with one 64-bit multimem
// store and read with one 64-bit relaxed load, so a reader that sees the tag of
// the current call also sees its payload; no flag or release is needed. Each
// row slot keeps a call counter; the tag is counter % 255 + 1 and two buffers
// alternate by its parity. A rank overwrites a parity buffer only after it has
// read every peer's entries of the call in between, which each peer wrote after
// it had finished reading that buffer.
//
// All ranks see the same candidates, so t* and the certificate agree on every
// rank and the fallback needs no extra communication.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/ptx.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace dcp_topk_merge {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr uint32_t kEmptyPos = 0xFFFFFFu;  // 24-bit position field of an empty entry

struct Smem {
  int hist[2][256];
  int warp_sum[kWarps];
  int digit, above, cnt;
  uint64_t min_key;
  int sent[kWarps], kept[kWarps];
};

SGL_DEVICE uint32_t convert_to_uint32(float x) {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// (score bits, position) -> 64-bit key, larger is better; 0 for an empty slot
SGL_DEVICE uint64_t make_topk_key(int32_t score_bits, int32_t pos) {
  if (pos < 0) return 0;
  return (static_cast<uint64_t>(convert_to_uint32(__int_as_float(score_bits))) << 32) | ~static_cast<uint32_t>(pos);
}

SGL_DEVICE uint64_t pack_entry(int32_t score_bits, int32_t pos, uint64_t tag) {
  const uint32_t p = pos >= 0 ? static_cast<uint32_t>(pos) : kEmptyPos;
  return (static_cast<uint64_t>(static_cast<uint32_t>(score_bits)) << 32) | (static_cast<uint64_t>(p) << 8) | tag;
}

SGL_DEVICE uint64_t entry_key(uint64_t e) {
  const uint32_t p = static_cast<uint32_t>(e >> 8) & kEmptyPos;
  return p == kEmptyPos ? 0 : make_topk_key(static_cast<int32_t>(e >> 32), static_cast<int32_t>(p));
}

// A visitor over a per-thread key array, in the form select_kth takes.
template <int kN>
SGL_DEVICE auto visit_keys(const uint64_t (&keys)[kN]) {
  return [&keys](auto f) {
#pragma unroll
    for (int i = 0; i < kN; ++i)
      f(keys[i]);
  };
}

// Exclusive prefix sum over the CTA (one value per thread); `total` is the CTA sum.
SGL_DEVICE int block_exclusive_sum(int v, int& total, Smem& s) {
  const uint32_t lane = device::get_lane_id();
  const int warp = threadIdx.x / 32;
  const int incl = device::warp::inclusive_sum(v, lane);
  if (lane == 31) s.warp_sum[warp] = incl;
  __syncthreads();
  int before = 0;
  total = 0;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    before += w < warp ? s.warp_sum[w] : 0;
    total += s.warp_sum[w];
  }
  __syncthreads();
  return before + incl - v;
}

// Writes the CTA's selected entries in thread order to slots [0, n) through
// emit(slot, i), i the entry within its thread, and pad(slot) to [n, kWidth).
template <int kWidth, int kN, class Emit, class Pad>
SGL_DEVICE void compact(const bool (&sel)[kN], Emit emit, Pad pad, Smem& s) {
  int mine = 0;
#pragma unroll
  for (int i = 0; i < kN; ++i)
    mine += sel[i];
  int n;
  int at = block_exclusive_sum(mine, n, s);
#pragma unroll
  for (int i = 0; i < kN; ++i)
    if (sel[i]) emit(at++, i);
  for (int e = n + threadIdx.x; e < kWidth; e += kThreads)
    pad(e);
}

// The rem-th largest key (rem >= 1) among the keys visited by `visit(f)`; every
// thread visits its own share and calls f the same number of times (k = 0 for
// an empty slot). 8-bit radix rounds from the top byte with double-buffered
// histograms and a one-warp scan; stops once the chosen bucket is exactly the
// remaining quota and returns that bucket's smallest key. 0 when rem < 1.
template <class Visit>
SGL_DEVICE uint64_t select_kth(Visit visit, int rem, Smem& s) {
  if (rem < 1) return 0;
  const int tid = threadIdx.x, warp = tid / 32;
  const uint32_t lane = device::get_lane_id();
  s.hist[0][tid] = 0;
  s.hist[1][tid] = 0;
  __syncthreads();
  uint64_t prefix = 0;
  int b = 0;
  for (int shift = 56; shift >= 0; shift -= 8, b ^= 1) {
    int* h = s.hist[b];
    visit([&](uint64_t k) {
      if (k && (shift == 56 || (k >> (shift + 8)) == prefix)) atomicAdd(&h[(k >> shift) & 255], 1);
    });
    s.hist[b ^ 1][tid] = 0;  // its last reader finished before the previous round's last barrier
    __syncthreads();
    if (warp == 0) {
      int c[8], tot = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        c[i] = h[255 - (lane * 8 + i)];
        tot += c[i];
      }
      int run = device::warp::inclusive_sum(tot, lane) - tot;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int excl = run;
        run += c[i];
        if (run >= rem && excl < rem) {
          s.digit = 255 - (lane * 8 + i);
          s.above = excl;
          s.cnt = c[i];
          s.min_key = ~uint64_t{0};
        }
      }
    }
    __syncthreads();
    rem -= s.above;
    prefix = (prefix << 8) | static_cast<uint64_t>(s.digit);
    if (s.cnt == rem) {
      const uint64_t pre = prefix;
      uint64_t mk = ~uint64_t{0};
      visit([&](uint64_t k) {
        if (k && (k >> shift) == pre) mk = k < mk ? k : mk;
      });
      mk = device::warp::reduce_min(mk);
      if (lane == 0) atomicMin(reinterpret_cast<unsigned long long*>(&s.min_key), mk);
      __syncthreads();
      return s.min_key;
    }
  }
  return prefix;
}

// sel[i] = keys[i] is at least t; nothing is selected when t is 0
template <int kN>
SGL_DEVICE void select_at_least(const uint64_t (&keys)[kN], uint64_t t, bool (&sel)[kN]) {
#pragma unroll
  for (int i = 0; i < kN; ++i)
    sel[i] = t != 0 && keys[i] >= t;
}

// Poll kPer entries (stride `step` from `src`) until all carry `tag`; every
// load of a round is in flight together.
template <int kPer>
SGL_DEVICE void poll_tagged(const uint64_t* src, int step, uint64_t tag, uint64_t (&v)[kPer]) {
  unsigned pending = (1u << kPer) - 1;
  while (pending) {
#pragma unroll
    for (int j = 0; j < kPer; ++j)
      if (pending >> j & 1) v[j] = device::ptx::load_relaxed_sys(src + j * step);
#pragma unroll
    for (int j = 0; j < kPer; ++j)
      if ((v[j] & 255) == tag) pending &= ~(1u << j);
  }
}

struct Params {
  const int2* __restrict__ candidates;    // [rows, kTopK] (score bits, global position)
  uint64_t* __restrict__ buffer;          // this rank's symmetric buffer
  uint64_t* mc_buffer;                    // its multicast address
  const int64_t* __restrict__ peer_ptrs;  // [kNumRanks] symmetric buffer of each rank
  int64_t* __restrict__ counters;         // [max_rows] calls per row slot
  const int32_t* __restrict__ global_lens;
  const int32_t* __restrict__ page_table;
  int64_t page_table_stride;
  int32_t* __restrict__ out;             // [rows, kTopK] widened slots, -1 after
  int32_t* __restrict__ fallback;        // [rows] 1 when the certificate failed
  uint64_t* __restrict__ union_scratch;  // [max_rows, kNumRanks * kTopK] fallback union; global, so smem stays small
  int32_t rank;
  int32_t span;  // page_size * kNumRanks
  int32_t max_rows;
};

// Fallback of a row whose certificate failed for the ranks in `fail`: the
// k-th key of the union in union_scratch, where segment q holds rank q's full
// local top-k if it failed (polled over NVLink), else its candidates.
template <int kNumRanks, int kTopK, int kCandK>
SGL_DEVICE uint64_t fallback_kth(
    const Params& p,
    int row,
    unsigned fail,
    int64_t full_off,
    uint64_t tag,
    const uint64_t (&cand)[kCandK / 32],
    Smem& s) {
  constexpr int kPerThread = kTopK / kThreads;
  constexpr int kPerLane = kCandK / 32;
  const int tid = threadIdx.x, warp = tid / 32;
  const uint32_t lane = device::get_lane_id();
  uint64_t* const keys = p.union_scratch + static_cast<int64_t>(row) * kNumRanks * kTopK;
#pragma unroll 1
  for (int q = 0; q < kNumRanks; ++q) {
    if (fail >> q & 1) {
      const auto* src = reinterpret_cast<const uint64_t*>(p.peer_ptrs[q]) + full_off;
      uint64_t tmp[kPerThread];
      poll_tagged<kPerThread>(src + tid, kThreads, tag, tmp);
#pragma unroll
      for (int i = 0; i < kPerThread; ++i)
        keys[q * kTopK + tid + i * kThreads] = entry_key(tmp[i]);
    } else if (warp == q) {
#pragma unroll
      for (int i = 0; i < kPerLane; ++i)
        keys[q * kTopK + lane + 32 * i] = cand[i];
      for (int e = kCandK + lane; e < kTopK; e += 32)
        keys[q * kTopK + e] = 0;
    }
  }
  __syncthreads();
  auto visit = [&](auto f) {
#pragma unroll 4
    for (int e = tid; e < kNumRanks * kTopK; e += kThreads)
      f(keys[e]);
  };
  int mine = 0, n;
  visit([&](uint64_t k) { mine += k != 0; });
  block_exclusive_sum(mine, n, s);
  return select_kth(visit, min(n, kTopK), s);
}

// Buffer layout (uint64_t entries): [2 parities][max_rows][kNumRanks * kCandK + kTopK],
// the candidate block of every rank, then this rank's full local top-k.
template <int kNumRanks, int kTopK, int kCandK>
__global__ void __launch_bounds__(kThreads) merge_kernel(const __grid_constant__ Params p) {
  static_assert(kNumRanks == kWarps, "warp q polls rank q's candidates");
  static_assert(kCandK % 32 == 0 && kTopK % kThreads == 0 && kCandK <= kTopK);
  constexpr int kPerThread = kTopK / kThreads;  // local top-k entries per thread
  constexpr int kPerLane = kCandK / 32;         // candidates per lane
  constexpr int64_t kSlot = static_cast<int64_t>(kNumRanks) * kCandK + kTopK;
  __shared__ Smem s;

  const int row = blockIdx.x, tid = threadIdx.x, warp = tid / 32;
  const int64_t c = p.counters[row];
  const uint64_t tag = static_cast<uint64_t>(c % 255) + 1;
  const int64_t base = ((c & 1) * p.max_rows + row) * kSlot;
  const int64_t full_off = base + static_cast<int64_t>(kNumRanks) * kCandK;

  // ---- this rank's local top-k, kept tagged for peers' fallback reads ----
  int2 v[kPerThread];
  uint64_t own[kPerThread];
  int n_valid = 0, n_own;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    v[i] = p.candidates[static_cast<int64_t>(row) * kTopK + tid + i * kThreads];
    own[i] = make_topk_key(v[i].x, v[i].y);
    n_valid += own[i] != 0;
    device::ptx::store_relaxed_sys(p.buffer + full_off + tid + i * kThreads, pack_entry(v[i].x, v[i].y, tag));
  }
  block_exclusive_sum(n_valid, n_own, s);

  // ---- push the best kCandK to every rank ----
  bool sel[kPerThread];
  select_at_least(own, select_kth(visit_keys(own), min(n_own, kCandK), s), sel);
  uint64_t* const dst = p.mc_buffer + base + static_cast<int64_t>(p.rank) * kCandK;
  compact<kCandK>(
      sel,
      [&](int e, int i) { device::ptx::multimem_store_relaxed(dst + e, pack_entry(v[i].x, v[i].y, tag)); },
      [&](int e) { device::ptx::multimem_store_relaxed(dst + e, pack_entry(0, -1, tag)); },
      s);

  // widened slots of the own keys while the pushes travel
  int slot[kPerThread];
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    const int pos = v[i].y;
    const int page = pos >= 0 ? p.page_table[row * p.page_table_stride + pos / p.span] : 0;
    slot[i] = pos >= 0 ? page * p.span + pos % p.span : -1;
  }

  // ---- threshold over the kNumRanks x kCandK candidates ----
  uint64_t cand[kPerLane];
  poll_tagged<kPerLane>(p.buffer + base + warp * kCandK + device::get_lane_id(), 32, tag, cand);
  int sent = 0;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    cand[i] = entry_key(cand[i]);
    sent += __popc(__ballot_sync(0xffffffff, cand[i] != 0));
  }
  if (device::get_lane_id() == 0) s.sent[warp] = sent;
  __syncthreads();
  int n_cand = 0;
#pragma unroll
  for (int q = 0; q < kNumRanks; ++q)
    n_cand += s.sent[q];
  const int want = min(n_cand, kTopK);
  uint64_t t = select_kth(visit_keys(cand), want, s);
  int kept = 0;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i)
    kept += __popc(__ballot_sync(0xffffffff, cand[i] != 0 && cand[i] >= t));
  if (device::get_lane_id() == 0) s.kept[warp] = kept;
  __syncthreads();

  // ---- certificate and exact fallback ----
  const int L = p.global_lens[row];
  unsigned fail = 0;
#pragma unroll
  for (int q = 0; q < kNumRanks; ++q) {
    const int local_len = max((L - q + kNumRanks - 1) / kNumRanks, 0);
    if (want > 0 && s.kept[q] == s.sent[q] && local_len > s.sent[q]) fail |= 1u << q;
  }
  if (tid == 0) p.fallback[row] = fail != 0;
  if (fail) t = fallback_kth<kNumRanks, kTopK, kCandK>(p, row, fail, full_off, tag, cand, s);

  // ---- this rank's keys >= t*, compacted ----
  select_at_least(own, t, sel);
  int32_t* const out = p.out + static_cast<int64_t>(row) * kTopK;
  compact<kTopK>(sel, [&](int e, int i) { out[e] = slot[i]; }, [&](int e) { out[e] = -1; }, s);
  __syncthreads();
  if (tid == 0) p.counters[row] = c + 1;
}

}  // namespace dcp_topk_merge

template <int64_t kNumRanks, int64_t kTopK, int64_t kCandK>
struct DcpTopKMergeKernel {
  static constexpr auto kernel = dcp_topk_merge::merge_kernel<kNumRanks, kTopK, kCandK>;

  /**
   * \brief Merges one batch of decode rows; every rank of the group calls it
   * with the same rows in the same order. See dcp_topk_merge.py for the
   * tensor contract.
   */
  static void
  run(tvm::ffi::TensorView candidates,
      tvm::ffi::TensorView buffer,
      int64_t mc_buffer,
      tvm::ffi::TensorView peer_ptrs,
      tvm::ffi::TensorView counters,
      tvm::ffi::TensorView global_lens,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out_slots,
      tvm::ffi::TensorView out_fallback,
      tvm::ffi::TensorView union_scratch,
      int64_t rank,
      int64_t span) {
    using namespace host;

    auto R = SymbolicSize{"rows"};
    auto M = SymbolicSize{"max_rows"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({R, kTopK, 2}).with_dtype<int32_t>().with_device(device_).verify(candidates);
    TensorMatcher({-1}).with_dtype<int64_t>().with_device(device_).verify(buffer);
    TensorMatcher({kNumRanks}).with_dtype<int64_t>().with_device(device_).verify(peer_ptrs);
    TensorMatcher({M}).with_dtype<int64_t>().with_device(device_).verify(counters);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(global_lens);
    TensorMatcher({-1, -1}).with_strides({-1, 1}).with_dtype<int32_t>().with_device(device_).verify(page_table);
    TensorMatcher({R, kTopK}).with_dtype<int32_t>().with_device(device_).verify(out_slots);
    TensorMatcher({R}).with_dtype<int32_t>().with_device(device_).verify(out_fallback);
    TensorMatcher({M, kNumRanks * kTopK}).with_dtype<int64_t>().with_device(device_).verify(union_scratch);

    const auto rows = static_cast<int32_t>(R.unwrap());
    const auto max_rows = static_cast<int32_t>(M.unwrap());
    RuntimeCheck(rows <= max_rows, "dcp_topk_merge: rows ", rows, " > max_rows ", max_rows);
    RuntimeCheck(
        page_table.size(0) >= rows, "dcp_topk_merge: page_table has ", page_table.size(0), " rows < rows ", rows);
    const int64_t need = 2LL * max_rows * (kNumRanks * kCandK + kTopK);
    RuntimeCheck(buffer.size(0) >= need, "dcp_topk_merge: buffer numel ", buffer.size(0), " < ", need);
    // an entry carries the position in 24 bits, all ones meaning empty
    const int64_t max_pos = page_table.size(1) * span;
    RuntimeCheck(max_pos <= dcp_topk_merge::kEmptyPos, "dcp_topk_merge: positions up to ", max_pos, " exceed 24 bits");
    if (rows == 0) return;

    const auto params = dcp_topk_merge::Params{
        .candidates = static_cast<const int2*>(candidates.data_ptr()),
        .buffer = static_cast<uint64_t*>(buffer.data_ptr()),
        .mc_buffer = reinterpret_cast<uint64_t*>(mc_buffer),
        .peer_ptrs = static_cast<const int64_t*>(peer_ptrs.data_ptr()),
        .counters = static_cast<int64_t*>(counters.data_ptr()),
        .global_lens = static_cast<const int32_t*>(global_lens.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .page_table_stride = page_table.stride(0),
        .out = static_cast<int32_t*>(out_slots.data_ptr()),
        .fallback = static_cast<int32_t*>(out_fallback.data_ptr()),
        .union_scratch = static_cast<uint64_t*>(union_scratch.data_ptr()),
        .rank = static_cast<int32_t>(rank),
        .span = static_cast<int32_t>(span),
        .max_rows = max_rows,
    };
    LaunchKernel(rows, dcp_topk_merge::kThreads, device_.unwrap())(kernel, params);
  }
};

}  // namespace sglang
