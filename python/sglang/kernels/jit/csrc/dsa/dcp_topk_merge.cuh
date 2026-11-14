// Exact global top-k merge for DSA decode context parallelism (DCP), one CTA per
// row. Ranks exchange their best kCandK local candidates over NVLink multicast
// and select the k-th key t* among them. A rank whose candidates all reach t*
// and that has more keys may hold more of the top-k; its full local top-k then
// replaces its candidates (exact: a rank holds at most k top-k keys). Every rank
// sees the same candidates, so all agree on t* without more traffic.
//
// Each 64-bit entry carries an 8-bit call tag beside its payload and is written
// and read in one access, so seeing the tag means seeing the payload. Two buffers
// alternate by call parity; a rank reuses one only after reading every peer's
// entries of the call in between.

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

// one warp per rank
constexpr int kMaxWarps = 16;
constexpr uint32_t kEmptyPos = 0xFFFFFFu;  // 24-bit position field of an empty entry

struct Smem {
  int hist[2][256];
  int warp_sum[kMaxWarps];
  int digit, above, cnt;
  uint64_t min_key;
  int sent[kMaxWarps], kept[kMaxWarps];
};

// ordered fp32 -> uint32 with -0 equal to +0, as top-k mode 2 orders the local candidates
SGL_DEVICE uint32_t extract_tie_key(float x) {
  const uint32_t bits = __float_as_uint(x == 0.0f ? 0.0f : x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// (score bits, position) -> 64-bit key, larger is better; 0 for an empty slot
SGL_DEVICE uint64_t make_topk_key(int32_t score_bits, int32_t pos) {
  if (pos < 0) return 0;
  return (static_cast<uint64_t>(extract_tie_key(__int_as_float(score_bits))) << 32) | ~static_cast<uint32_t>(pos);
}

SGL_DEVICE uint64_t pack_entry(int32_t score_bits, int32_t pos, uint64_t tag) {
  const uint32_t p = pos >= 0 ? static_cast<uint32_t>(pos) : kEmptyPos;
  return (static_cast<uint64_t>(static_cast<uint32_t>(score_bits)) << 32) | (static_cast<uint64_t>(p) << 8) | tag;
}

SGL_DEVICE uint64_t entry_key(uint64_t e) {
  const uint32_t p = static_cast<uint32_t>(e >> 8) & kEmptyPos;
  return p == kEmptyPos ? 0 : make_topk_key(static_cast<int32_t>(e >> 32), static_cast<int32_t>(p));
}

template <int kN>
SGL_DEVICE auto visit_keys(const uint64_t (&keys)[kN]) {
  return [&keys](auto f) {
#pragma unroll
    for (int i = 0; i < kN; ++i)
      f(keys[i]);
  };
}

template <int kWarps>
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

// selected entries to slots [0, n) in thread order via emit(slot, i); pad(slot) fills [n, kWidth)
template <int kWidth, int kThreads, int kN, class Emit, class Pad>
SGL_DEVICE void compact(const bool (&sel)[kN], Emit emit, Pad pad, Smem& s) {
  int mine = 0;
#pragma unroll
  for (int i = 0; i < kN; ++i)
    mine += sel[i];
  int n;
  int at = block_exclusive_sum<kThreads / 32>(mine, n, s);
#pragma unroll
  for (int i = 0; i < kN; ++i)
    if (sel[i]) emit(at++, i);
  for (int e = n + threadIdx.x; e < kWidth; e += kThreads)
    pad(e);
}

// The rem-th largest nonzero key that `visit` yields across the CTA (0 when rem < 1),
// by 8-bit radix rounds from the top byte.
template <class Visit>
SGL_DEVICE uint64_t select_kth(Visit visit, int rem, Smem& s) {
  if (rem < 1) return 0;
  const int tid = threadIdx.x, warp = tid / 32;
  const uint32_t lane = device::get_lane_id();
  for (int i = tid; i < 256; i += blockDim.x) {
    s.hist[0][i] = 0;
    s.hist[1][i] = 0;
  }
  __syncthreads();
  uint64_t prefix = 0;
  int b = 0;
  for (int shift = 56; shift >= 0; shift -= 8, b ^= 1) {
    int* h = s.hist[b];
    visit([&](uint64_t k) {
      if (k && (shift == 56 || (k >> (shift + 8)) == prefix)) atomicAdd(&h[(k >> shift) & 255], 1);
    });
    for (int i = tid; i < 256; i += blockDim.x)
      s.hist[b ^ 1][i] = 0;  // last read before the previous round's final barrier
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

template <int kN>
SGL_DEVICE void select_at_least(const uint64_t (&keys)[kN], uint64_t t, bool (&sel)[kN]) {
#pragma unroll
  for (int i = 0; i < kN; ++i)
    sel[i] = t != 0 && keys[i] >= t;
}

// v[j] = *addr(j) for every bit j of `pending`, once each carries `tag`
template <int kPer, typename Addr>
SGL_DEVICE void poll_tagged_at(Addr addr, uint64_t tag, uint64_t (&v)[kPer], unsigned pending = (1u << kPer) - 1) {
  while (pending) {
#pragma unroll
    for (int j = 0; j < kPer; ++j)
      if (pending >> j & 1) v[j] = device::ptx::load_relaxed_sys(addr(j));
#pragma unroll
    for (int j = 0; j < kPer; ++j)
      if ((v[j] & 255) == tag) pending &= ~(1u << j);
  }
}

template <int kPer>
SGL_DEVICE void poll_tagged(const uint64_t* src, int step, uint64_t tag, uint64_t (&v)[kPer]) {
  poll_tagged_at<kPer>([=](int j) { return src + j * step; }, tag, v);
}

// k-th largest of the nonzero keys `visit` yields across the CTA, capped at kTopK
template <int kWarps, int kTopK, typename Visit>
SGL_DEVICE uint64_t kth_of_nonzero(Visit visit, Smem& s) {
  int mine = 0, n;
  visit([&](uint64_t k) { mine += k != 0; });
  block_exclusive_sum<kWarps>(mine, n, s);
  return select_kth(visit, min(n, kTopK), s);
}

struct Params {
  const int2* __restrict__ candidates;  // [rows, kTopK] (score bits, global position)
  uint64_t* __restrict__ buffer;        // this rank's symmetric buffer
  uint64_t* multicast_ptr;
  const int64_t* __restrict__ peer_ptrs;  // [kNumRanks] symmetric buffer of each rank
  int64_t* __restrict__ counters;         // [max_rows] calls per row slot
  const int32_t* __restrict__ global_lens;
  const int32_t* __restrict__ page_table;
  int64_t page_table_stride;
  int32_t* __restrict__ out_slots;       // [rows, kTopK] widened slots, -1 after
  int32_t* __restrict__ fallback;        // [rows] 1 when the certificate failed
  uint64_t* __restrict__ union_scratch;  // [max_rows, kNumRanks * kTopK] fallback union
  int32_t dcp_rank;
  int32_t span;  // page_size * kNumRanks
  int32_t max_rows;
};

// k-th key of the union of each failed rank's full local top-k and the others' candidates
template <int kNumRanks, int kTopK, int kCandK>
SGL_DEVICE uint64_t fallback_kth(
    const Params& p,
    int row,
    unsigned fail,
    int64_t full_off,
    uint64_t tag,
    const uint64_t (&cand)[kCandK / 32],
    Smem& s) {
  constexpr int kThreads = 32 * kNumRanks;
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
  return kth_of_nonzero<kNumRanks, kTopK>(visit, s);
}

// fallback_kth for at most two failed ranks without the union scratch: the
// failed ranks' full local top-k (polled into registers) and the candidates of
// every other rank, selected in registers.
template <int kNumRanks, int kTopK, int kCandK>
SGL_DEVICE uint64_t fallback_kth_regs(
    const Params& p, unsigned fail, int64_t full_off, uint64_t tag, const uint64_t (&cand)[kCandK / 32], Smem& s) {
  constexpr int kThreads = 32 * kNumRanks;
  constexpr int kPerThread = kTopK / kThreads;
  constexpr int kPerLane = kCandK / 32;
  const int tid = threadIdx.x, warp = tid / 32;
  const int q0 = __ffs(fail) - 1, q1 = 31 - __clz(fail);
  const bool two = q1 != q0;
  const auto* src0 = reinterpret_cast<const uint64_t*>(p.peer_ptrs[q0]) + full_off + tid;
  const auto* src1 = reinterpret_cast<const uint64_t*>(p.peer_ptrs[q1]) + full_off + tid;
  uint64_t f[2 * kPerThread] = {};
  poll_tagged_at<2 * kPerThread>(
      [&](int j) { return (j < kPerThread ? src0 : src1) + (j % kPerThread) * kThreads; },
      tag,
      f,
      (1u << (two ? 2 * kPerThread : kPerThread)) - 1);
#pragma unroll
  for (int j = 0; j < 2 * kPerThread; ++j)
    f[j] = j < kPerThread || two ? entry_key(f[j]) : 0;
  const bool drop = fail >> warp & 1;
  auto visit = [&](auto fn) {
#pragma unroll
    for (int i = 0; i < kPerLane; ++i)
      fn(drop ? uint64_t{0} : cand[i]);
#pragma unroll
    for (int j = 0; j < 2 * kPerThread; ++j)
      fn(f[j]);
  };
  return kth_of_nonzero<kNumRanks, kTopK>(visit, s);
}

// Keys of rank q among the first L positions under the token interleave.
template <int kNumRanks>
SGL_DEVICE int local_len(int L, int q) {
  return max((L - q + kNumRanks - 1) / kNumRanks, 0);
}

// Buffer layout (uint64_t entries): [2 parities][max_rows][kNumRanks * kCandK + kTopK],
// the candidate block of every rank, then this rank's full local top-k.
template <int kNumRanks, int kTopK, int kCandK>
__global__ void __launch_bounds__(32 * kNumRanks, kNumRanks == 8 ? 3 : 1)
    merge_kernel(const __grid_constant__ Params p) {
  static_assert(kNumRanks == 8 || kNumRanks == 16, "one warp per rank: 8 or 16 ranks");
  constexpr int kThreads = 32 * kNumRanks;
  static_assert(kCandK % 32 == 0 && kTopK % kThreads == 0 && kCandK <= kTopK);
  constexpr int kPerThread = kTopK / kThreads;
  constexpr int kPerLane = kCandK / 32;
  constexpr int64_t kSlot = static_cast<int64_t>(kNumRanks) * kCandK + kTopK;
  __shared__ Smem s;
  __shared__ uint64_t stage[kCandK];  // this rank's candidate block, pushed coalesced

  const int row = blockIdx.x, tid = threadIdx.x, warp = tid / 32;
  const int64_t c = p.counters[row];
  const uint64_t tag = static_cast<uint64_t>(c % 255) + 1;
  const int64_t base = ((c & 1) * p.max_rows + row) * kSlot;
  const int64_t full_off = base + static_cast<int64_t>(kNumRanks) * kCandK;

  // ---- this rank's local top-k ----
  int2 v[kPerThread];
  uint64_t own[kPerThread];
  int n_valid = 0, n_own;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    v[i] = p.candidates[static_cast<int64_t>(row) * kTopK + tid + i * kThreads];
    own[i] = make_topk_key(v[i].x, v[i].y);
    n_valid += own[i] != 0;
  }
  block_exclusive_sum<kNumRanks>(n_valid, n_own, s);

  // ---- push the best kCandK to every rank; keep the full top-k tagged for peers' fallback reads ----
  bool sel[kPerThread];
  select_at_least(own, select_kth(visit_keys(own), min(n_own, kCandK), s), sel);
  uint64_t* const dst = p.multicast_ptr + base + static_cast<int64_t>(p.dcp_rank) * kCandK;
  compact<kCandK, kThreads>(
      sel,
      [&](int e, int i) { stage[e] = pack_entry(v[i].x, v[i].y, tag); },
      [&](int e) { stage[e] = pack_entry(0, -1, tag); },
      s);
  __syncthreads();
  for (int e = tid; e < kCandK; e += kThreads)
    device::ptx::multimem_store_relaxed(dst + e, stage[e]);
#pragma unroll
  for (int i = 0; i < kPerThread; ++i)
    device::ptx::store_relaxed_sys(p.buffer + full_off + tid + i * kThreads, pack_entry(v[i].x, v[i].y, tag));

  // widened slots of the own keys, while the pushes travel
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
    if (want > 0 && s.kept[q] == s.sent[q] && local_len<kNumRanks>(L, q) > s.sent[q]) fail |= 1u << q;
  }
  if (tid == 0) p.fallback[row] = fail != 0;
  const int n_fail = __popc(fail);
  if (n_fail > 2) {
    t = fallback_kth<kNumRanks, kTopK, kCandK>(p, row, fail, full_off, tag, cand, s);
  } else if (n_fail > 0) {
    t = fallback_kth_regs<kNumRanks, kTopK, kCandK>(p, fail, full_off, tag, cand, s);
  }

  int32_t* const out = p.out_slots + static_cast<int64_t>(row) * kTopK;
  const auto pad = [&](int e) { out[e] = -1; };
  // ---- this rank's keys >= t*, compacted ----
  select_at_least(own, t, sel);
  compact<kTopK, kThreads>(sel, [&](int e, int i) { out[e] = slot[i]; }, pad, s);
  __syncthreads();
  if (tid == 0) p.counters[row] = c + 1;
}

}  // namespace dcp_topk_merge

template <int64_t kNumRanks, int64_t kTopK, int64_t kCandK>
struct DcpTopKMergeKernel {
  static constexpr auto kernel = dcp_topk_merge::merge_kernel<kNumRanks, kTopK, kCandK>;

  /// \brief Merges one batch of decode rows; dcp_topk_merge.py holds the contract.
  static void
  run(tvm::ffi::TensorView candidates,
      tvm::ffi::TensorView buffer,
      int64_t multicast_ptr,
      tvm::ffi::TensorView peer_ptrs,
      tvm::ffi::TensorView counters,
      tvm::ffi::TensorView global_lens,
      tvm::ffi::TensorView page_table,
      tvm::ffi::TensorView out_slots,
      tvm::ffi::TensorView out_fallback,
      tvm::ffi::TensorView union_scratch,
      int64_t dcp_rank,
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
    RuntimeCheck(0 <= dcp_rank && dcp_rank < kNumRanks, "dcp_topk_merge: dcp_rank ", dcp_rank, " out of range");
    RuntimeCheck(
        page_table.size(0) >= rows, "dcp_topk_merge: page_table has ", page_table.size(0), " rows < rows ", rows);
    const int64_t need = 2LL * max_rows * (kNumRanks * kCandK + kTopK);
    RuntimeCheck(buffer.size(0) >= need, "dcp_topk_merge: buffer numel ", buffer.size(0), " < ", need);
    const int64_t max_pos = page_table.size(1) * span;
    RuntimeCheck(max_pos <= dcp_topk_merge::kEmptyPos, "dcp_topk_merge: positions up to ", max_pos, " exceed 24 bits");
    if (rows == 0) return;

    const auto params = dcp_topk_merge::Params{
        .candidates = static_cast<const int2*>(candidates.data_ptr()),
        .buffer = static_cast<uint64_t*>(buffer.data_ptr()),
        .multicast_ptr = reinterpret_cast<uint64_t*>(multicast_ptr),
        .peer_ptrs = static_cast<const int64_t*>(peer_ptrs.data_ptr()),
        .counters = static_cast<int64_t*>(counters.data_ptr()),
        .global_lens = static_cast<const int32_t*>(global_lens.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .page_table_stride = page_table.stride(0),
        .out_slots = static_cast<int32_t*>(out_slots.data_ptr()),
        .fallback = static_cast<int32_t*>(out_fallback.data_ptr()),
        .union_scratch = static_cast<uint64_t*>(union_scratch.data_ptr()),
        .dcp_rank = static_cast<int32_t>(dcp_rank),
        .span = static_cast<int32_t>(span),
        .max_rows = max_rows,
    };
    LaunchKernel(rows, static_cast<int>(32 * kNumRanks), device_.unwrap())(kernel, params);
  }
};

}  // namespace sglang
