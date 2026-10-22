// DeepSeek-V4.1 Engram lookup for EP decode as a two-kernel push.
//
// Every TP rank owns a row shard of each Engram table (whole hash columns when
// 24 % R == 0, else the equal-row split). Instead of all-gathering the hash
// ids, gathering on every rank and reduce-scattering BF16 rows, each rank sends
// its ids to the owner, and the owner pushes the raw MXFP8 rows straight into
// the requester's wkv input buffer. Rank d's receive buffer is u64
// recv[R][1 + max_ids]: recv[s][0] is the number of ids rank s sent, its
// packets {hi = t * 24 + col, lo = global row} follow from recv[s][1].
//
//   engram_id_push    plain CTAs on every rank (grid from T): bucket this
//                     rank's T x 24 ids by owner, write the packets and the
//                     count of every owner (0 included) into owner.recv[me].
//                     No cross-rank sync.
//   engram_scatter    persistent: a rank-level head barrier (all peers' id
//                     pushes have landed), then copy the 256 B FP8 row + 8 B
//                     E8M0 scales of every packet from the local shard into
//                     requester.out_fp8 / requester.out_sf (the 128x4 swizzled
//                     layout of the MXFP8 GEMM), tail barrier.
//
// One recv / out set serves both Engram layers, with no double buffering.
// Safety:
//   - recv reuse: the scatter of call n ends with a cross-rank barrier joined
//     by every CTA after its last read of recv, so no owner still reads recv
//     when any rank starts the id push of call n + 1, which rewrites every count.
//   - out reuse: an owner writes requester.out only after the head barrier of
//     the same step, which the requester joins after its previous wkv GEMM has
//     consumed out. This holds ONLY IF the GEMM that reads out runs on the
//     same stream between two lookups.
// Above 2 ranks both barriers signal through one multicast red per CTA.
// `EngramPush::lookup` launches both kernels; every rank calls it every step
// (an idle rank with T = 0 included): it still owns rows that the others ask for.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/atomic.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/engram.cuh>
#include <sgl_kernel/distributed/communicator.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>

namespace sglang {

namespace device::engram::mega {

using namespace device::engram::config;

template <uint32_t kWorldSize_>
struct IdPushTrait {
  static constexpr uint32_t kWorldSize = kWorldSize_;
  static constexpr uint32_t kNumThreads = 256;
  static constexpr uint32_t kVecSize = 4;                      // ids per 16 B load
  static constexpr uint32_t kMaxVecs = 4;                      // loads per thread, so <= 4096 ids per CTA
  static constexpr uint32_t kCtaIds = kNumThreads * kVecSize;  // ids per CTA and vec
  // 24 % R == 0: shards are whole hash columns. Else the equal-row split.
  static constexpr bool kByColumn = kNumCols % kWorldSize == 0;
  using vec_t = AlignedVector<uint32_t, kVecSize>;

  static_assert(kWorldSize <= kNumThreads && kNumCols % kVecSize == 0);
};

inline constexpr uint32_t kSortBuckets = 256;
inline constexpr uint32_t kSortBucketShift = 20;  // 1 Mi rows = 256 MiB of fp8 per bucket
inline constexpr uint32_t kSortSrcShift = 26;     // entry = src << 26 | packet index

/// This rank's private state, zero-filled once. The id push re-zeroes its
/// counters itself and resets the sort state before every scatter.
template <uint32_t kWorldSize>
struct LocalBuffer {
  uint64_t counter[kWorldSize];                  // id push: {arrived CTAs, ids} per owner
  alignas(16) uint32_t histogram[kSortBuckets];  // sorted visit: packets per row bucket
  // Sorted visit: one Event per grid sync, reset by the id push before each scatter.
  atomic::Event grid_sync[2];
  uint32_t sort_entries[];  // sorted visit: [R * max_ids] packets in row order
};

template <uint32_t kWorldSize>
struct IdPushParams {
  const int32_t* __restrict__ ids;  // [T, 24] global rows, contiguous, 16 B aligned
  LocalBuffer<kWorldSize>* local;
  uint64_t* workspace[kWorldSize];  // rank d's recv buffer for this rank: [0] count, [1 + slot] packets
  uint32_t num_ids;
  uint32_t num_vecs;  // loads per thread, <= kMaxVecs
  // Equal-row split only: ceil(2^64 / N); umulhi(x, magic) == x / N for x < R * N (checked on host).
  uint64_t owner_magic;
};

template <uint32_t N, uint32_t M, typename Fn>
SGL_DEVICE void static_for_each(uint32_t idx, Fn fn) {
  constexpr uint32_t kNumLoops = N / M;
  for (uint32_t i = 0; i < kNumLoops; ++i) {
    fn(idx + i * M);
  }
  if constexpr (N % M != 0) {
    const auto last_idx = idx + kNumLoops * M;
    if (last_idx < N) fn(last_idx);
  }
}

/**
 * \brief Bucket this rank's hash ids by owner and push them as packets.
 *
 * Each thread keeps its ids and their owners in registers across both passes.
 * Pass 1 counts ids per owner in smem; a multi-CTA grid turns the counts into
 * per-CTA bases with one atomic per owner, and the last CTA to arrive publishes
 * the total and re-zeroes its counter. Pass 2 writes packets
 * {hi = t * 24 + col, lo = global row} into owner.workspace[1 + slot].
 * No cross-rank barrier: the next kernel's head barrier publishes the writes.
 */
template <uint32_t kWorldSize, typename Trait = IdPushTrait<kWorldSize>>
__global__ __launch_bounds__(Trait::kNumThreads, 1) void engram_id_push_kernel(
    const __grid_constant__ IdPushParams<kWorldSize> params) {
  using vec_t = typename Trait::vec_t;
  constexpr uint32_t kVecSize = Trait::kVecSize;
  constexpr uint32_t kMaxVecs = Trait::kMaxVecs;
  constexpr uint32_t kNumThreads = Trait::kNumThreads;
  // Same-owner lanes take consecutive slots, so a warp's packet stores to one owner coalesce.
  __shared__ uint32_t s_counter[kWorldSize];

  const auto tx = threadIdx.x;
  const auto num_ids = params.num_ids;
  const auto num_vecs = params.num_vecs;
  const auto cta_begin = blockIdx.x * num_vecs * Trait::kCtaIds;
  const auto offset_of = [&](uint32_t v) { return cta_begin + (v * Trait::kNumThreads + tx) * kVecSize; };
  // Loads of this thread inside [0, num_ids): offset_of grows with v.
  const auto first = cta_begin + tx * kVecSize;
  const auto num_local_vecs = first < num_ids ? min(num_vecs, div_ceil(num_ids - first, Trait::kCtaIds)) : 0u;
  const auto find_owner = [&](uint32_t offset, uint32_t id) -> uint32_t {
    if constexpr (Trait::kByColumn) {
      return offset % kNumCols / (kNumCols / kWorldSize);
    } else {
      // Rank r owns [N r / R, N (r + 1) / R), so owner = (R (id + 1) - 1) / N.
      return static_cast<uint32_t>(__umul64hi((uint64_t{id} + 1) * kWorldSize - 1, params.owner_magic));
    }
  };

  static_for_each<kWorldSize, kNumThreads>(tx, [&](uint32_t d) { s_counter[d] = 0; });

  PDLWaitPrimary<true>();
  // The previous scatter is done (the stream orders it before this grid): get
  // the next scatter's sort state ready, so its main loop never resets it.
  if (blockIdx.x == 0) {
    auto& local = *params.local;
    static_for_each<kSortBuckets, kNumThreads>(tx, [&](uint32_t b) { local.histogram[b] = 0; });
    if (tx < 2) local.grid_sync[tx].unsafe_reset();
  }

  vec_t ids[kMaxVecs];
#pragma unroll
  for (uint32_t v = 0; v < kMaxVecs; ++v) {
    if (v < num_local_vecs) ids[v].load(params.ids + offset_of(v));
  }
  __syncthreads();

  uint32_t owner[kMaxVecs][kVecSize];
#pragma unroll
  for (uint32_t v = 0; v < kMaxVecs; ++v) {
    if (v >= num_local_vecs) break;
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      owner[v][j] = find_owner(offset_of(v) + j, ids[v][j]);
      atomicAdd(&s_counter[owner[v][j]], 1u);
    }
  }
  __syncthreads();

  if (tx < kWorldSize) {
    const uint32_t n = s_counter[tx];
    if (gridDim.x > 1) {
      const auto payload = static_cast<uint64_t>(1) << 32 | n;
      const auto old = atomicAdd(reinterpret_cast<unsigned long long*>(&params.local->counter[tx]), payload);
      s_counter[tx] = static_cast<uint32_t>(old);
      if ((old >> 32) + 1 == gridDim.x) {
        params.workspace[tx][0] = static_cast<uint32_t>(old) + n;
        params.local->counter[tx] = 0;  // every CTA has arrived
      }
    } else {
      params.workspace[tx][0] = n;
      s_counter[tx] = 0;
    }
  }
  __syncthreads();

#pragma unroll
  for (uint32_t v = 0; v < kMaxVecs; ++v) {
    if (v >= num_local_vecs) break;
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      const auto owner_id = owner[v][j];
      const auto slot = atomicAdd(&s_counter[owner_id], 1u);
      const auto packed = static_cast<uint64_t>(offset_of(v) + j) << 32 | ids[v][j];
      params.workspace[owner_id][1 + slot] = packed;
    }
  }

  PDLTriggerSecondary<true>();
}

template <uint32_t kWorldSize>
struct ScatterParams {
  // Above 2 ranks one multicast signal per barrier replaces the R-wide unicast fan-out.
  static constexpr bool kMulticast = kWorldSize > 2;
  const uint8_t* __restrict__ weight;  // local shard [rows, 256] fp8 (device or registered host memory)
  const uint8_t* __restrict__ scale;   // local shard [rows, 8] e8m0
  const uint64_t* recv;                // local [R, stride]: [s][0] count, [s][1 + i] packets
  uint32_t stride;                     // 1 + max_ids
  uint32_t row_start;                  // first global row of the local shard
  uint32_t rank;
  uint32_t sort_min_packets;  // row-ordered visit from this many packets (UINT32_MAX: never)
  LocalBuffer<kWorldSize>* local;
  uint8_t* out[kWorldSize];  // rank s's out_fp8 [T_max, 6144]
  uint8_t* sf[kWorldSize];   // rank s's swizzled scales
  // Multicast: this rank's semaphores. Else every rank's.
  // The array of semaphores for the current rank.
  distributed::Semaphore* semaphores[kMulticast ? 2 : kWorldSize];

  SGL_DEVICE auto make_barrier(uint32_t num_arrives) const {
    if constexpr (kMulticast) {
      return distributed::McBarrier{semaphores[0], semaphores[1], kWorldSize, num_arrives};
    } else {
      return distributed::Barrier<kWorldSize>{semaphores, rank, num_arrives};
    }
  }
};

/// Byte offset of scale column tile `k_tile` of row `m` in the 128x4 layout.
SGL_DEVICE uint32_t sf_offset(uint32_t m, uint32_t k_tile) {
  return ((m / 128) * kSfTilesPerRow + k_tile) * kSfTileBytes + (m % 32) * 16 + (m % 128 / 32) * 4;
}

// Row-ordered visit: a host shard on 64 KiB base pages is far beyond the GPU's
// address translation reach, so rows read in random order miss on almost every
// access (4-8x slower than on huge pages). Bucketing the packets by row and
// giving each CTA one contiguous slice keeps every SM inside a ~1/G region.

/// All CTAs of the grid; it is persistent with at most one CTA per SM, which the
/// cross-rank barriers already rely on. The Event is used once per call.
SGL_DEVICE void grid_sync(atomic::Event& event) {
  __syncthreads();
  if (threadIdx.x < kWarpSize && warp::elect_one_lane()) {
    event.arrive_async();
    event.wait_no_reset(/*num_producers=*/gridDim.x);
  }
  __syncthreads();
}

/// Counting-sort the packets by row bucket (two passes, two grid syncs), then
/// CTA b copies the b-th contiguous slice of the sorted packets.
template <uint32_t kWorldSize, uint32_t kNumThreads, uint32_t kUnroll>
SGL_DEVICE bool scatter_sorted(const ScatterParams<kWorldSize>& params, const uint32_t* s_count, uint32_t total) {
  constexpr uint32_t kGroupLanes = kRowBytes / 16;
  constexpr uint32_t kNumGroups = kNumThreads / kGroupLanes;
  constexpr uint32_t kTile = kNumGroups * kUnroll;
  constexpr uint32_t kIdxMask = (1u << kSortSrcShift) - 1;
  using vec_t = AlignedVector<uint32_t, 4>;
  static_assert(kSortBuckets <= kNumThreads && kSortBuckets % kWarpThreads == 0);
  __shared__ uint32_t s_prefix[kWorldSize + 1];
  alignas(16) __shared__ uint32_t s_hist[kSortBuckets];  // pass 1 counts, pass 2 cursors
  __shared__ uint32_t s_start[kSortBuckets];             // this CTA's first slot per bucket

  const auto tx = threadIdx.x;
  auto& local = *params.local;
  if (tx == 0) {
    uint32_t acc = 0;
    for (uint32_t src = 0; src < kWorldSize; ++src) {
      s_prefix[src] = acc;
      acc += s_count[src];
    }
    s_prefix[kWorldSize] = acc;
  }
  if (tx < kSortBuckets) s_hist[tx] = 0;
  __syncthreads();

  const auto lo = static_cast<uint32_t>(uint64_t{total} * blockIdx.x / gridDim.x);
  const auto hi = static_cast<uint32_t>(uint64_t{total} * (blockIdx.x + 1) / gridDim.x);
  const auto packet = [&](uint32_t src, uint32_t idx) {
    return params.recv[src * params.stride + 1 + idx];  //
  };
  const auto bucket_of = [](uint64_t pkt, uint32_t row_start) {
    return min((static_cast<uint32_t>(pkt) - row_start) >> kSortBucketShift, kSortBuckets - 1);
  };
  // Packet e of the concatenated sources -> (source, index in its slot).
  const auto visit = [&](auto fn) {
    uint32_t src = 0;
    for (uint32_t e = lo + tx; e < hi; e += kNumThreads) {
      while (e >= s_prefix[src + 1]) {
        ++src;
      }
      fn(src, e - s_prefix[src]);
    }
  };

  // Pass 1: count this CTA's slice, reserve its run in every bucket.
  visit([&](uint32_t src, uint32_t idx) {  //
    atomicAdd(&s_hist[bucket_of(packet(src, idx), params.row_start)], 1u);
  });
  __syncthreads();
  if (tx < kSortBuckets) {
    const auto n = s_hist[tx];
    s_start[tx] = n ? atomicAdd(&local.histogram[tx], n) : 0;
  }
  grid_sync(local.grid_sync[0]);

  // Bucket starts: exclusive scan of the bucket totals by warp 0.
  if (tx < kWarpThreads) {
    constexpr uint32_t kPer = kSortBuckets / kWarpThreads;
    static_assert(kPer == 8);
    AlignedVector<uint32_t, 4> vec[2];
    vec[0].load(local.histogram + tx * kPer + 0);
    vec[1].load(local.histogram + tx * kPer + 4);
    uint32_t v[9];
    v[0] = 0;
#pragma unroll
    for (uint32_t j = 0; j < kPer; ++j) {
      v[j + 1] = v[j] + vec[j / 4][j % 4];
    }
    const auto sum = v[8];
    const auto exc_sum = warp::inclusive_sum(sum) - sum;
#pragma unroll
    for (uint32_t j = 0; j < kPer; ++j) {
      vec[j / 4][j % 4] = v[j] + exc_sum;
    }
    vec[0].store(s_hist + tx * kPer + 0);
    vec[1].store(s_hist + tx * kPer + 4);
  }
  __syncthreads();
  // Pass 2: place the entries.
  visit([&](uint32_t src, uint32_t idx) {
    const uint32_t b = bucket_of(packet(src, idx), params.row_start);
    local.sort_entries[s_start[b] + atomicAdd(&s_hist[b], 1u)] = src << kSortSrcShift | idx;
  });

  grid_sync(local.grid_sync[1]);

  // Copy the sorted slice [lo, hi); the next tile's packets are fetched while
  // this tile's rows are in flight.
  const uint32_t group_id = tx / kGroupLanes;
  const uint32_t lane_id = tx % kGroupLanes;
  uint32_t src[kUnroll];
  uint64_t pkt[kUnroll];
  const auto fetch = [&](uint32_t begin) {
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      const uint32_t p = begin + u * kNumGroups + group_id;
      const uint32_t entry = p < hi ? local.sort_entries[p] : 0;
      src[u] = entry >> kSortSrcShift;
      pkt[u] = p < hi ? packet(src[u], entry & kIdxMask) : 0;
    }
  };
  fetch(lo);
  for (uint32_t begin = lo; begin < hi; begin += kTile) {
    const uint32_t end = min(begin + kTile, hi);
    vec_t row[kUnroll];
    uint32_t scale[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      if (begin + u * kNumGroups + group_id >= end) break;
      const uint32_t local_row = static_cast<uint32_t>(pkt[u]) - params.row_start;
      row[u].load(params.weight + uint64_t{local_row} * kRowBytes, lane_id);
      if (lane_id < 2) {
        scale[u] = reinterpret_cast<const uint32_t*>(params.scale + uint64_t{local_row} * kScaleBytes)[lane_id];
      }
    }
    uint32_t cur_src[kUnroll], th[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      cur_src[u] = src[u];
      th[u] = static_cast<uint32_t>(pkt[u] >> 32);
    }
    fetch(begin + kTile);
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      if (begin + u * kNumGroups + group_id >= end) break;
      row[u].store(params.out[cur_src[u]] + uint64_t{th[u]} * kRowBytes, lane_id);
      if (lane_id < 2) {
        const uint32_t t = th[u] / kNumCols;
        const uint32_t h = th[u] - t * kNumCols;
        *reinterpret_cast<uint32_t*>(params.sf[cur_src[u]] + sf_offset(t, 2 * h + lane_id)) = scale[u];
      }
    }
  }
  return lo < hi;
}

/**
 * \brief Scatter every packet in the local recv buffer: push the MXFP8 row and
 *        its scales into the requester's GEMM input.
 *
 * Persistent grid of identical size on every rank (the tail barrier pairs CTA
 * b with CTA b of each peer). Head: CTA 0 signals every peer once and waits for
 * all R signals (the id pushes of every rank are visible), then releases the
 * other CTAs through a local word; all CTAs polling one sys-scope flag instead
 * is ~2x slower on GB300.
 * Work is (source s, tile i) items of kTile packets, item = i * R + s, dealt
 * round-robin over a grid coprime with R, so every CTA cycles through all
 * sources and a source's tiles spread over the CTAs; or, on a large host shard
 * from `sort_min_packets` packets, the row-ordered visit of `scatter_sorted`. A
 * row is moved by a group of 16 lanes, 16 B each; `kUnroll` packets per group
 * are in flight.
 */
template <uint32_t kWorldSize, uint32_t kNumThreads, uint32_t kUnroll>
__global__ __launch_bounds__(kNumThreads, 1) void engram_scatter_kernel(  //
    const __grid_constant__ ScatterParams<kWorldSize> params) {
  constexpr uint32_t kGroupLanes = kRowBytes / 16;
  constexpr uint32_t kNumGroups = kNumThreads / kGroupLanes;
  constexpr uint32_t kTile = kNumGroups * kUnroll;  // packets per item
  using vec_t = AlignedVector<uint32_t, 4>;
  static_assert(kWorldSize <= kNumThreads && kNumThreads % kWarpThreads == 0);

  __shared__ uint32_t s_count[kWorldSize];
  __shared__ uint32_t s_num_items;
  __shared__ uint32_t s_total;

  PDLWaitPrimary<true>();
  const auto barrier = params.make_barrier(/*num_arrives=*/2);
  barrier.arrive_relaxed(/*n=*/0);

  const auto tx = threadIdx.x;
  // The barrier's polling thread is in warp 0 (thread 0, or thread `rank` for R = 2).
  if (tx < kWarpThreads) {
    __syncwarp();
    const auto lane_id = tx;
    uint32_t max_count = 0, total = 0;
    for (uint32_t src = lane_id; src < kWorldSize; src += kWarpThreads) {
      const auto count = static_cast<uint32_t>(params.recv[src * params.stride]);
      s_count[src] = count;
      max_count = max(max_count, count);
      total += count;
    }
    max_count = warp::reduce_max(max_count);
    total = warp::reduce_sum(total);
    if (lane_id == 0) {
      s_num_items = div_ceil(max_count, kTile) * kWorldSize;
      s_total = total;
    }
  }
  __syncthreads();

  // The same decision in every CTA: the sorted visit syncs the whole grid.
  const bool sorted = s_total > params.sort_min_packets;
  bool pushed = sorted && scatter_sorted<kWorldSize, kNumThreads, kUnroll>(params, s_count, s_total);

  const uint32_t group_id = tx / kGroupLanes;
  const uint32_t lane_id = tx % kGroupLanes;
  const uint32_t num_items = sorted ? 0 : s_num_items;
  for (uint32_t item = blockIdx.x; item < num_items; item += gridDim.x) {
    const uint32_t src = item % kWorldSize;
    const uint32_t begin = item / kWorldSize * kTile;
    const uint32_t count = s_count[src];
    if (begin >= count) continue;
    pushed = true;
    const uint32_t end = min(begin + kTile, count);
    const uint64_t* packets = params.recv + src * params.stride + 1;

    uint64_t pkt[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      const uint32_t p = begin + u * kNumGroups + group_id;
      pkt[u] = p < end ? packets[p] : 0;
    }
    vec_t row[kUnroll];
    uint32_t scale[kUnroll];  // lanes 0 and 1: the two 4-scale halves of the row
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      if (begin + u * kNumGroups + group_id >= end) break;
      const uint32_t local_row = static_cast<uint32_t>(pkt[u]) - params.row_start;
      row[u].load(params.weight + uint64_t{local_row} * kRowBytes, lane_id);
      if (lane_id < 2) {
        scale[u] = reinterpret_cast<const uint32_t*>(params.scale + uint64_t{local_row} * kScaleBytes)[lane_id];
      }
    }
    const auto out = params.out[src];
    const auto sf = params.sf[src];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      if (begin + u * kNumGroups + group_id >= end) break;
      const auto th = static_cast<uint32_t>(pkt[u] >> 32);
      row[u].store(out + uint64_t{th} * kRowBytes, lane_id);
      if (lane_id < 2) {
        const uint32_t t = th / kNumCols;
        const uint32_t h = th - t * kNumCols;
        *reinterpret_cast<uint32_t*>(sf + sf_offset(t, 2 * h + lane_id)) = scale[u];
      }
    }
  }

  __syncthreads();
  // A CTA that pushed nothing has nothing to publish.
  PDLTriggerSecondary<true>();
  if (pushed) {
    barrier.arrive_rel_acq(/*n=*/1);
  } else {
    barrier.arrive_relaxed(/*n=*/1);
  }
}

}  // namespace device::engram::mega

using host::distributed::CommunicatorRef;

template <uint32_t kWorldSize, uint32_t kScatterThreads, uint32_t kUnroll>
struct MegaEngram {
  using TensorView = tvm::ffi::TensorView;
  using Trait = device::engram::mega::IdPushTrait<kWorldSize>;
  static constexpr uint32_t kNumCols = device::engram::config::kNumCols;
  static constexpr bool kMulticast = device::engram::mega::ScatterParams<kWorldSize>::kMulticast;
  /// Row-ordered visit from this many packets per owner (~700 tokens per rank at
  /// R = 2): it costs ~6 us and saves 4-8x on a host shard without huge pages...
  static constexpr uint32_t kSortMinPackets = 16384;
  /// ... on shards larger than the GPU's address translation reach (~16 GiB).
  static constexpr int64_t kSortMinShardBytes = int64_t{16} << 30;

  using LocalBuffer = device::engram::mega::LocalBuffer<kWorldSize>;

  /// Bytes of the LocalBuffer for receive slots of `max_ids` ids.
  static int64_t local_bytes(int64_t max_ids) {
    return sizeof(LocalBuffer) + sizeof(uint32_t) * kWorldSize * max_ids;
  }

  /// Peer address of the byte at `local` in this rank's workspace.
  static uint8_t* peer_ptr(const host::distributed::PullPlaneObj& pull, const void* local, uint32_t peer) {
    const auto offset = static_cast<const uint8_t*>(local) - pull.workspaces[pull.rank];
    CHECK_HOST(offset >= 0 && offset < pull.num_bytes) << "buffer is not inside the pull workspace";
    return pull.workspaces[peer] + offset;
  }

  /**
   * \brief One Engram lookup: push this rank's ids to their owners, then serve
   *        the rows the peers asked for into their wkv inputs.
   *
   * Every rank calls this every step, idle (T = 0) or not, on workspaces with
   * the same number of semaphores; the GEMM reading `out` must run on the same
   * stream before the next call.
   *
   * \param hash_ids    [T, 24] int32, contiguous, 16 B aligned (T may be 0)
   * \param weight_ptr  local shard [rows, 256] uint8 (device or registered host)
   * \param scale_ptr   local shard [rows, 8] uint8
   * \param recv        [R, 1 + max_ids] uint64, this rank's receive buffer
   * \param out         [T_max, 6144] uint8, this rank's GEMM input
   * \param sf          [ceil(T_max / 128) * 128 * 192] uint8, this rank's swizzled scales
   * \param owner_magic 0 for whole-column shards, else ceil(2^64 / N), exact for the table (checked by the caller)
   * \param row_start   first global row of the local shard
   * \param num_rows    rows of the local shard; a shard of at least kSortMinShardBytes
   *                    takes the row-ordered visit unless `no_sort`
   * \param no_sort     never take the row-ordered visit (e.g. a device shard)
   * \param local       [local_bytes(max_ids)] uint8, this rank's LocalBuffer, zero before the first call
   */
  static void lookup(
      CommunicatorRef ref,
      TensorView hash_ids,
      int64_t weight_ptr,
      int64_t scale_ptr,
      TensorView recv,
      TensorView out,
      TensorView sf,
      uint64_t owner_magic,
      uint32_t row_start,
      uint32_t num_rows,
      bool no_sort,
      TensorView local) {
    using namespace host;
    using device::engram::config::kRowBytes, device::engram::config::kTokenScales;
    const auto& pull = ref.get()->get_pull_obj();
    CHECK_HOST(pull.world_size == kWorldSize);
    auto T = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    // Any strides: an empty [0, 24] view need not report contiguous ones.
    TensorMatcher({T, kNumCols})  //
        .with_strides({-1, -1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(hash_ids);
    TensorMatcher({kWorldSize, -1})  //
        .with_dtype<uint64_t>()
        .with_device(device)
        .verify(recv);
    TensorMatcher({-1, kNumCols * kRowBytes})  //
        .with_dtype<uint8_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(out);
    const int64_t sf_bytes = div_ceil(out.size(0), int64_t{128}) * 128 * kTokenScales;
    TensorMatcher({sf_bytes})  //
        .with_dtype<uint8_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(sf);
    TensorMatcher({-1})  //
        .with_dtype<uint8_t>()
        .with_device(device)
        .ensure_alignment(alignof(LocalBuffer))
        .verify(local);
    const auto stride = recv.stride(0);  // since contiguous, == recv.size(1)
    const auto num_tokens = static_cast<uint32_t>(T.unwrap());
    const auto num_ids = num_tokens * kNumCols;
    CHECK_HOST(num_ids == 0 || ((num_tokens == 1 || hash_ids.stride(0) == kNumCols) && hash_ids.stride(1) == 1))
        << "hash_ids must be contiguous";
    CHECK_HOST(num_ids == 0 || reinterpret_cast<uintptr_t>(hash_ids.data_ptr()) % 16 == 0);
    CHECK_HOST(num_ids < stride) << num_tokens << " tokens exceed the receive buffer";
    CHECK_HOST(weight_ptr % 16 == 0 && scale_ptr % 8 == 0);
    CHECK_HOST(int64_t{row_start} + num_rows <= int64_t{1} << 31);
    CHECK_HOST(local.numel() >= local_bytes(stride - 1));
    CHECK_HOST(int64_t{kWorldSize} * (stride - 1) < int64_t{1} << device::engram::mega::kSortSrcShift);
    CHECK_HOST(!kMulticast || pull.mc_semaphore != nullptr) << "R = " << kWorldSize << " needs multicast semaphores";
    // Id push: one load per thread until the grid would pass the SM count, then up to kMaxVecs.
    const auto local_buffer = static_cast<LocalBuffer*>(local.data_ptr());
    const auto num_sms = runtime::get_sm_count(device.unwrap().device_id);
    const auto num_vecs = std::clamp<uint32_t>(div_ceil(num_ids, Trait::kCtaIds * num_sms), 1, Trait::kMaxVecs);
    // Scatter: one CTA per semaphore, shrunk until coprime with R so every CTA cycles through all sources.
    uint32_t num_scatter_blocks = pull.num_blocks;
    while (std::gcd(num_scatter_blocks, kWorldSize) != 1) {
      --num_scatter_blocks;
    }
    const auto num_id_push_blocks = std::max<uint32_t>(1, div_ceil(num_ids, Trait::kCtaIds * num_vecs));

    device::engram::mega::IdPushParams<kWorldSize> id_push_params{};
    id_push_params.ids = static_cast<const int32_t*>(hash_ids.data_ptr());
    id_push_params.local = local_buffer;
    const auto mine = static_cast<uint64_t*>(recv.data_ptr()) + int64_t{pull.rank} * stride;
    for (uint32_t d = 0; d < kWorldSize; ++d) {
      id_push_params.workspace[d] = reinterpret_cast<uint64_t*>(peer_ptr(pull, mine, d));
    }
    id_push_params.num_ids = num_ids;
    id_push_params.num_vecs = num_vecs;
    id_push_params.owner_magic = owner_magic;

    device::engram::mega::ScatterParams<kWorldSize> scatter_params{};
    scatter_params.weight = reinterpret_cast<const uint8_t*>(weight_ptr);
    scatter_params.scale = reinterpret_cast<const uint8_t*>(scale_ptr);
    scatter_params.recv = static_cast<const uint64_t*>(recv.data_ptr());
    scatter_params.stride = stride;
    scatter_params.row_start = row_start;
    scatter_params.rank = pull.rank;
    scatter_params.local = local_buffer;
    const bool sort_rows = !no_sort && int64_t{num_rows} * kRowBytes >= kSortMinShardBytes;
    static_assert(kSortMinPackets > 0);
    scatter_params.sort_min_packets = sort_rows ? kSortMinPackets - 1 : UINT32_MAX;
    for (uint32_t s = 0; s < kWorldSize; ++s) {
      scatter_params.out[s] = peer_ptr(pull, out.data_ptr(), s);
      scatter_params.sf[s] = peer_ptr(pull, sf.data_ptr(), s);
    }
    if constexpr (kMulticast) {
      scatter_params.semaphores[0] = pull.semaphores[pull.rank];
      scatter_params.semaphores[1] = pull.mc_semaphore;
    } else {
      for (uint32_t s = 0; s < kWorldSize; ++s) {
        scatter_params.semaphores[s] = pull.semaphores[s];
      }
    }
    const auto id_push_kernel = device::engram::mega::engram_id_push_kernel<kWorldSize>;
    const auto scatter_kernel = device::engram::mega::engram_scatter_kernel<kWorldSize, kScatterThreads, kUnroll>;
    LaunchKernel(num_id_push_blocks, Trait::kNumThreads, device.unwrap())  //
        .enable_pdl(true)(id_push_kernel, id_push_params);
    LaunchKernel(num_scatter_blocks, kScatterThreads, device.unwrap())  //
        .enable_pdl(true)(scatter_kernel, scatter_params);
  }
};

}  // namespace sglang
