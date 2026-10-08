// MHC sequence-parallel sublayer boundary over peer-to-peer NVLink load/store, fused with
// its communication. The NVLink-multicast (NVLS) variant lives in mhc_sp_fusion_nvls.cuh.
//
// Every TP rank holds a partial sublayer output y_r [T_pad, kHidden] (bf16) in symmetric
// memory and owns M = T_pad / world rows of the hc residual. One kernel does
//
//   RS:   y[row] = sum_r y_r[row]          (row in this rank's slice, peer loads)
//   post: R_new = post * y + comb^T R_old  (local rows)
//   norm: x = RMSNorm(pre . R_new)         (local rows)
//   AG:   x -> every rank's x buffer       (peer stores)
//
// With kStats the remaining CTAs also compute this boundary's own coefficients from
// R_old on tensor cores and publish them per tile; the comm CTAs wait on that before
// they consume a row. R_old is in memory before the launch, so the statistics start
// immediately and run under the reduce-scatter.
//
// y and x may be the same symmetric buffer, which is what the caller does to reduce in
// place: a row is reduced by exactly one rank, and that rank reads every copy of the row
// before writing its result over them. That does not hold under kFP8AG -- the scale plane
// sits past the values and would land on top of a peer's rows.
//
// Why a pull for the reduce-scatter and a push for the all-gather: those are the only
// directions that keep both halves of a rank's NVLink busy at once. A rank reads
// (world - 1) / world of the partial y it owns and writes (world - 1) / world of the x
// it produced, so ingress and egress carry the same bytes and overlap.
//
// A group of kGroupThreads owns one row at a time, assigned by a static stride: the
// per-tile ready gate already serializes consumption to tile order, so ticket scheduling
// has nothing left to equalize, and a static row sequence lets the ring prefetch
// arbitrarily far ahead. A row's rank partials arrive through a 2-deep TMA ring per
// group, so the NVLink reads of row n+1 are in flight while row n computes and stores.
//
// Without kEpilogue the comm role stops after writing R': no norm fold_collapse, no next-input
// peer stores, nothing written to x. That serves a seam whose next norm cannot fold_collapse --
// the next combine computes from the sharded R' and pays the gather itself.
#pragma once
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/atomic.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>
#include <sgl_kernel/dsv41/mhc.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/stl.h>

#include <cstdint>
#include <vector>

namespace sglang {

namespace mhc_p2p {

using namespace device::mhc;

// A bare `using namespace device` would make `ptx` ambiguous with device::ptx.
namespace ptx = device::mhc::ptx;

inline constexpr uint32_t kGroupThreads = 128;
inline constexpr uint32_t kRowVecs = kHidden / 8;
inline constexpr uint32_t kVecsPerThread = kRowVecs / kGroupThreads;  // 5
inline constexpr uint32_t kWarpsPerGroup = kGroupThreads / device::kWarpThreads;
static_assert(kRowVecs % kGroupThreads == 0);

using vec_t = device::AlignedVector<bf16x2_t, 4>;

// Nesting vec_t in an AlignedVector gives 32B accesses: one 256-bit load/store on Blackwell.
using vec_pair_t = device::AlignedVector<vec_t, 2>;

inline constexpr uint32_t kRowBytes = kHidden * sizeof(bf16_t);  // one row of y / x / weight

// The comm role's shared: a 2-deep ring the row's rank partials are TMA-copied into.
// `warp_sums` is double buffered so a row's read cannot race the next row's write,
// which saves the second group barrier a single buffer would need.
template <uint32_t kRowsPerCTA, uint32_t kWorldSize>
struct CommSmem {
  __align__(128) bf16_t norm_weight[kHidden];
  float warp_sums[2][kRowsPerCTA][kWarpsPerGroup];
  __align__(128) bf16_t y_stage[2][kRowsPerCTA][kWorldSize][kHidden];
  uint64_t weight_arrived;
  uint64_t stage_bar[2][kRowsPerCTA];
};

// The two roles never share a CTA, so they share the allocation: the comm role's ring
// costs nothing next to the statistics working set.
template <uint32_t kRowsPerCTA, uint32_t kWorldSize, typename StatTrait>
union BoundarySmem {
  CommSmem<kRowsPerCTA, kWorldSize> comm;
  MHCStatSmem<StatTrait> stats;
};

template <uint32_t kWorldSize>
struct Params {
  // ---- communication ----
  // Symmetric buffers, indexed by rank: y is read from every peer, x written to every
  // peer. The semaphores come from the all-reduce pull plane; this kernel only barriers
  // on them and never touches its buffers.
  const bf16_t* y_peer[kWorldSize];  // [T_pad, kHidden] bf16
  bf16_t* x_peer[kWorldSize];        // [T_pad, kHidden] bf16
  device::distributed::Semaphore* semaphores[kWorldSize];
  const bf16_t* residual;          // [M, kNumStreams, kHidden] local rows
  bf16_t* residual_out;            // [M, kNumStreams, kHidden]
  const float *post, *comb, *pre;  // [M, 4], [M, 4, 4], [M, 4]
  const bf16_t* norm_weight;       // [kHidden]
  uint32_t rank;
  uint32_t num_rows;    // M
  uint32_t row_offset;  // rank * M
  uint32_t total_rows;  // T_pad, for the MXFP8 scale offsets
  float eps;
  uint32_t num_comm_blocks;  // blocks [0, this) run the comm role
  // ---- this boundary's own statistics, when kStats ----
  // Its pre / post / comb alias the comm role's, which is the point: the statistics CTAs
  // produce exactly what the comm CTAs of this same launch consume.
  MHCStatParams stats;
};

template <uint32_t kWorldSize>
SGL_DEVICE void
store_mxfp8(const Params<kWorldSize>& params, const vec_t& out, uint64_t global_row, uint32_t vec_index) {
  using namespace device;
  const auto q = mhc::quantize_mxfp8(out);
  const uint32_t lane = threadIdx.x % kWarpThreads;
  // Join the halves so the warp's 8 scales leave as one 8B store rather than two 4B
  // ones: a 4B store to a peer is the worst shape this datapath has.
  const uint32_t scales_hi = __shfl_sync(0xffffffff, q.half_warp_scales, 16);
  const uint64_t value_offset = mhc::mxfp8_value_offset(global_row, vec_index);
  // Lane 0 owns the scale store; (kHidden / 32) and its own (vec_index / 16) * 4 are
  // both multiples of 8, so the address is 8B aligned.
  const uint64_t scale_offset = mhc::mxfp8_scale_offset(params.total_rows, global_row, vec_index);
  // Fixed destination order: at world 2 there is no many-writer contention to shape (measured flat).
#pragma unroll
  for (uint32_t r = 0; r < kWorldSize; ++r) {
    auto* base = reinterpret_cast<uint8_t*>(params.x_peer[r]);
    *reinterpret_cast<uint2*>(base + value_offset) = make_uint2(q.values[0], q.values[1]);
    if (lane == 0) *reinterpret_cast<uint2*>(base + scale_offset) = make_uint2(q.half_warp_scales, scales_hi);
  }
}

// The pair-map MXFP8 all-gather (kStore256b): 16B value store per lane; one 16B scale
// store per warp, lane 0 joining the four 8-lane segment words. Both offsets are 16B
// aligned: values at 16 * slot, scales at 16 * warp within multiples-of-16 row strides.
template <uint32_t kWorldSize>
SGL_DEVICE void
store_mxfp8_pair(const Params<kWorldSize>& params, const vec_t& o0, const vec_t& o1, uint64_t global_row, uint32_t v0) {
  using namespace device;
  const auto q = mhc::quantize_mxfp8_pair(o0, o1);
  const uint32_t lane = threadIdx.x % kWarpThreads;
  const uint4 scales = {
      q.seg_scales,
      __shfl_sync(0xffffffff, q.seg_scales, 8),
      __shfl_sync(0xffffffff, q.seg_scales, 16),
      __shfl_sync(0xffffffff, q.seg_scales, 24),
  };
  const uint64_t value_offset = mhc::mxfp8_value_offset(global_row, v0);
  const uint64_t scale_offset = mhc::mxfp8_scale_offset(params.total_rows, global_row, v0);
#pragma unroll
  for (uint32_t r = 0; r < kWorldSize; ++r) {
    auto* base = reinterpret_cast<uint8_t*>(params.x_peer[r]);
    *reinterpret_cast<uint4*>(base + value_offset) = make_uint4(q.values[0], q.values[1], q.values[2], q.values[3]);
    if (lane == 0) *reinterpret_cast<uint4*>(base + scale_offset) = scales;
  }
}

// ---------------------------------------------------------------------------
// Comm role: RS + post / combine / norm + AG for this rank's rows.
// ---------------------------------------------------------------------------
// kStore256b: the write side (R_new and the all-gather) issues 32B stores; the per-thread
// vector map becomes adjacent pairs (v = 2*slot + up*256) plus a 16B tail (512 + slot),
// all warp-coalesced. Per-element values are bitwise-unchanged; only the sum_squares
// reduction order follows the ownership map, so x is tolerance-level (~2 ulp) against the
// 16B build while R and the coefficients stay bitwise.
template <uint32_t kWorldSize, uint32_t kRowsPerCTA, bool kStats, bool kFP8AG, bool kEpilogue, bool kStore256b>
SGL_DEVICE void run_comm(
    const Params<kWorldSize>& params, uint32_t comm_bx, uint32_t num_blocks, CommSmem<kRowsPerCTA, kWorldSize>& smem) {
  using namespace device;
  static_assert(kRowsPerCTA + 1 <= 16, "one named barrier per group");
  const auto tx = threadIdx.x;
  const auto group = tx / kGroupThreads;
  const auto slot = tx % kGroupThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto warp_id_in_group = slot / kWarpThreads;
  const auto warp_id = tx / kWarpThreads;
  if constexpr (kEpilogue) {
    if (warp_id == 1 && warp::elect_one_lane()) {
      ptx::mbar_init(&smem.weight_arrived, 1);
      ptx::mbar_arrive_expect_tx(&smem.weight_arrived, kRowBytes);
      ptx::cp_async_bulk_g2s(smem.norm_weight, params.norm_weight, kRowBytes, &smem.weight_arrived);
      ptx::mbar_wait_parity(&smem.weight_arrived, 0);
    }
  }
  if (warp_id == 2) {
    static_assert(kRowsPerCTA < kWarpThreads);
    if (const auto g = lane_id; g < kRowsPerCTA) {
      ptx::mbar_init(&smem.stage_bar[0][g], 1);
      ptx::mbar_init(&smem.stage_bar[1][g], 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }

  const distributed::Barrier<kWorldSize> barrier{params.semaphores, params.rank, 2, comm_bx};
  barrier.arrive_relaxed(0);
  __syncthreads();

  const auto issue = [&](uint32_t row, uint32_t stage) {
    ptx::mbar_arrive_expect_tx(&smem.stage_bar[stage][group], kWorldSize * kRowBytes);
#pragma unroll
    for (uint32_t r = 0; r < kWorldSize; ++r) {
      ptx::cp_async_bulk_g2s(
          smem.y_stage[stage][group][r],
          params.y_peer[r] + (uint64_t(params.row_offset) + row) * kHidden,
          kHidden * sizeof(bf16_t),
          &smem.stage_bar[stage][group]);
    }
  };
  const uint32_t row_base = comm_bx * kRowsPerCTA + group;
  const uint32_t row_stride = num_blocks * kRowsPerCTA;

  if (slot == 0) {
#pragma unroll
    for (uint32_t s = 0; s < 2; ++s) {
      if (row_base + s * row_stride < params.num_rows) issue(row_base + s * row_stride, s);
    }
  }

  const auto get_normed = [&](const vec_t& coll, const vec_t& weight, float inv_rms) {
    using namespace device;
    vec_t out;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const auto v = cast<fp32x2_t>(coll[k]);
      const auto w = cast<fp32x2_t>(weight[k]);
      out[k] = cast<bf16x2_t>(fp32x2_t{v.x * inv_rms * w.x, v.y * inv_rms * w.y});
    }
    return out;
  };
  const auto normalize = [&](const vec_t(&collapsed)[kVecsPerThread], float inv_rms, uint64_t global_row) {
    const auto push = [&](const vec_t& out, uint32_t v) {
#pragma unroll
      for (uint32_t r = 0; r < kWorldSize; ++r) {
        out.store(params.x_peer[r], global_row * kRowVecs + v);
      }
    };
    if constexpr (kStore256b) {
#pragma unroll
      for (uint32_t up = 0; up < 2; ++up) {
        const uint32_t v0 = 2 * slot + up * 2 * kGroupThreads;
        vec_t w0, w1;
        w0.load(smem.norm_weight, v0);
        w1.load(smem.norm_weight, v0 + 1);
        if constexpr (kFP8AG) {
          store_mxfp8_pair(
              params,
              get_normed(collapsed[2 * up], w0, inv_rms),
              get_normed(collapsed[2 * up + 1], w1, inv_rms),
              global_row,
              v0);
        } else {
          vec_pair_t out;
          out[0] = get_normed(collapsed[2 * up], w0, inv_rms);
          out[1] = get_normed(collapsed[2 * up + 1], w1, inv_rms);
#pragma unroll
          for (uint32_t r = 0; r < kWorldSize; ++r) {
            out.store(params.x_peer[r], (global_row * kRowVecs + v0) / 2);
          }
        }
      }
      const uint32_t vt = 4 * kGroupThreads + slot;
      vec_t wt;
      wt.load(smem.norm_weight, vt);
      const auto out = get_normed(collapsed[4], wt, inv_rms);
      // The tail is on the stock map, so it quantizes with the stock block geometry.
      if constexpr (kFP8AG) {
        store_mxfp8(params, out, global_row, vt);
      } else {
        push(out, vt);
      }
    } else {
#pragma unroll
      for (uint32_t j = 0; j < kVecsPerThread; ++j) {
        const auto v = slot + j * kGroupThreads;
        vec_t weight;
        weight.load(smem.norm_weight, v);
        const auto out = get_normed(collapsed[j], weight, inv_rms);
        if constexpr (kFP8AG) {
          store_mxfp8(params, out, global_row, v);
        } else {
          push(out, v);
        }
      }
    }
  };

  // One counter carries the whole schedule: iteration k consumes ring slot k % 2,
  // whose mbarrier is on phase (k / 2) % 2, and writes warp_sums buffer k % 2.
  for (uint32_t k = 0;; ++k) {
    const uint32_t row = row_base + k * row_stride;
    if (row >= params.num_rows) break;
    const uint32_t stage = k & 1;
    const uint64_t global_row = params.row_offset + row;
    const bf16_t* old_row = params.residual + static_cast<uint64_t>(row) * K;

    if constexpr (kStats) {
      if (slot == 0) {
        const auto tile = row / M_TILE;
        const auto rows_in_tile = min(M_TILE, params.num_rows - tile * M_TILE);
        params.stats.ready[tile].wait(1, rows_in_tile, 1, kPollSleepNanoSecond);
      }
    }
    if (slot == kWarpThreads) {
      // One thread spins the ring mbarrier, on a different warp from slot 0's ready
      // spin; the group barrier below publishes the arrival.
      static_assert(kWarpThreads < kGroupThreads);
      ptx::mbar_wait_parity(&smem.stage_bar[stage][group], (k >> 1) & 1);
    }
    // Publishes both waits: the ring arrival to every thread, and under kStats the
    // coefficient reads behind slot 0's ready-wait.
    ptx::bar_sync(group + 1, kGroupThreads);

    const auto coeff = load_row_coeff(params.pre + row * 4, params.post + row * 4, params.comb + row * 16, lane_id);
    float post[kNumStreams], pre[kNumStreams], comb[kNumStreams][kNumStreams];
#pragma unroll
    for (uint32_t to = 0; to < kNumStreams; ++to) {
      post[to] = coeff_post(coeff, to);
      pre[to] = coeff_pre(coeff, to);
#pragma unroll
      for (uint32_t from = 0; from < kNumStreams; ++from) {
        comb[from][to] = coeff_comb(coeff, from, to);
      }
    }

    bf16_t* new_row = params.residual_out + static_cast<uint64_t>(row) * K;
    vec_t collapsed[kVecsPerThread];
    float sum_squares = 0.f;
    // The rank sum, out of the ring instead of straight off the wire: fp32 in rank
    // order and rounded once, so every rank rounds identically.
    const auto get_rank_sum = [&](uint32_t vec_index) {
      vec_t part[kWorldSize], y;
#pragma unroll
      for (uint32_t r = 0; r < kWorldSize; ++r)
        part[r].load(smem.y_stage[stage][group][r], vec_index);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        auto acc = cast<fp32x2_t>(part[0][k]);
#pragma unroll
        for (uint32_t r = 1; r < kWorldSize; ++r) {
          const auto v = cast<fp32x2_t>(part[r][k]);
          acc.x += v.x;
          acc.y += v.y;
        }
        y[k] = cast<bf16x2_t>(acc);
      }
      return y;
    };
    const auto get_post_stream = [&](const vec_t& y,
                                     const vec_t(&old_streams)[kNumStreams],
                                     uint32_t to,
                                     fp32x2_t(&collapse_acc)[4]) {  //
      vec_t updated;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const auto contribution = cast<fp32x2_t>(y[k]);
        auto acc = fp32x2_t{post[to] * contribution.x, post[to] * contribution.y};
#pragma unroll
        for (uint32_t from = 0; from < kNumStreams; ++from) {
          const auto old = cast<fp32x2_t>(old_streams[from][k]);
          acc = math::fma_f32x2(fp32x2_t{comb[from][to], comb[from][to]}, old, acc);
        }
        updated[k] = cast<bf16x2_t>(acc);
        if constexpr (kEpilogue) {
          // The next combine consumes the rounded residual, as the reference does.
          const auto rounded = cast<fp32x2_t>(updated[k]);
          collapse_acc[k] = math::fma_f32x2(fp32x2_t{pre[to], pre[to]}, rounded, collapse_acc[k]);
        }
      }
      return updated;
    };
    const auto fold_collapse = [&](const fp32x2_t(&ca)[4], vec_t& coll) {
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        coll[k] = cast<bf16x2_t>(ca[k]);
        const auto v = cast<fp32x2_t>(coll[k]);
        sum_squares = fmaf(v.x, v.x, sum_squares);
        sum_squares = fmaf(v.y, v.y, sum_squares);
      }
    };
    const auto process_one = [&](uint32_t v, vec_t& coll) {
      const auto y = get_rank_sum(v);
      vec_t old_streams[kNumStreams];
#pragma unroll
      for (uint32_t from = 0; from < kNumStreams; ++from)
        old_streams[from].load(old_row + from * kHidden, v);
      fp32x2_t ca[4] = {};
#pragma unroll
      for (uint32_t to = 0; to < kNumStreams; ++to) {
        const auto u = get_post_stream(y, old_streams, to, ca);
        u.store(new_row + to * kHidden, v);
      }
      if constexpr (kEpilogue) fold_collapse(ca, coll);
    };

    if constexpr (kStore256b) {
      static_assert(kVecsPerThread == 5, "the pair map is shaped for 5 vectors per thread");
      const auto process_pair = [&](uint32_t v0, vec_t& coll0, vec_t& coll1) {
        const auto y0 = get_rank_sum(v0 + 0), y1 = get_rank_sum(v0 + 1);
        vec_t old0[kNumStreams], old1[kNumStreams];
#pragma unroll
        for (uint32_t from = 0; from < kNumStreams; ++from) {
          vec_pair_t old;
          old.load(old_row + from * kHidden, v0 / 2);
          old0[from] = old[0];
          old1[from] = old[1];
        }
        fp32x2_t ca0[4] = {}, ca1[4] = {};
#pragma unroll
        for (uint32_t to = 0; to < kNumStreams; ++to) {
          vec_pair_t updated;
          updated[0] = get_post_stream(y0, old0, to, ca0);
          updated[1] = get_post_stream(y1, old1, to, ca1);
          updated.store(new_row + to * kHidden, v0 / 2);
        }
        if constexpr (kEpilogue) {
          fold_collapse(ca0, coll0);
          fold_collapse(ca1, coll1);
        }
      };
#pragma unroll
      for (uint32_t up = 0; up < 2; ++up) {
        process_pair(2 * slot + up * 2 * kGroupThreads, collapsed[2 * up], collapsed[2 * up + 1]);
      }
      process_one(4 * kGroupThreads + slot, collapsed[4]);
    } else {
#pragma unroll
      for (uint32_t j = 0; j < kVecsPerThread; ++j) {
        process_one(slot + j * kGroupThreads, collapsed[j]);
      }
    }

    if constexpr (kEpilogue) {
      // broadcast write same value to same smem dst, ok
      smem.warp_sums[stage][group][warp_id_in_group] = warp::reduce_sum(sum_squares);
    }
    // One barrier closes the ring slot (every thread has read its y vectors) and
    // publishes warp_sums; only then may slot 0 refill the slot.
    ptx::bar_sync(group + 1, kGroupThreads);
    if (slot == 0) {
      const uint32_t next = row + 2 * row_stride;
      if (next < params.num_rows) issue(next, stage);
    }
    if constexpr (kEpilogue) {
      float row_total = 0.f;
#pragma unroll
      for (uint32_t w = 0; w < kWarpsPerGroup; ++w)
        row_total += smem.warp_sums[stage][group][w];
      normalize(collapsed, math::rsqrt(row_total / float(kHidden) + params.eps), global_row);
    }
  }

  __syncthreads();
  barrier.arrive_rel_acq(1);
}

}  // namespace mhc_p2p

extern __shared__ __align__(16) char smem_base[];

template <
    uint32_t kWorldSize,
    uint32_t kRowsPerCTA,
    bool kStats,
    typename StatTrait,
    bool kFP8AG,
    bool kEpilogue,
    bool kStore256b>
__global__ __launch_bounds__(mhc_p2p::kGroupThreads* kRowsPerCTA, 1) void mhc_sp_fusion_p2p_kernel(
    const __grid_constant__ mhc_p2p::Params<kWorldSize> params) {
  auto& smem = *reinterpret_cast<mhc_p2p::BoundarySmem<kRowsPerCTA, kWorldSize, StatTrait>*>(smem_base);
  const auto comm = [&](uint32_t comm_bx, uint32_t num_blocks) {
    mhc_p2p::run_comm<kWorldSize, kRowsPerCTA, kStats, kFP8AG, kEpilogue, kStore256b>(
        params, comm_bx, num_blocks, smem.comm);
  };
  if constexpr (!kStats) {
    comm(blockIdx.x, gridDim.x);
  } else {
    static_assert(mhc_p2p::kGroupThreads * kRowsPerCTA == StatTrait::kBlockSize, "the roles share a block shape");
    // Roles are laid out in dependency order: mma, then the reduction that consumes it,
    // then the communication that consumes the published tiles. Blocks are dispatched by
    // increasing blockIdx, so a grid that cannot be seated all at once still makes
    // progress -- a producer is never left behind a consumer that is waiting for it.
    const auto bx = blockIdx.x;
    if (bx < StatTrait::kNumMMABlocks) {
      device::mhc::run_mma<StatTrait>(params.stats, bx, smem.stats);
    } else if (bx < StatTrait::kNumBlocks) {
      device::mhc::run_reduction<StatTrait, true>(params.stats, bx - StatTrait::kNumMMABlocks, smem.stats);
    } else {
      comm(bx - StatTrait::kNumBlocks, params.num_comm_blocks);
    }
  }
}

template <
    uint32_t kWorldSize,
    uint32_t kRowsPerCTA,
    bool kStats,
    uint32_t kNumWeightParts,
    uint32_t SPLIT_K,
    uint32_t kNumStatMMABlocks,
    uint32_t kNumStatReduceBlocks,
    bool kFP8AG,
    bool kEpilogue,
    bool kStore256b = false>
struct MHCSPFusionP2P {
  using StatTrait = device::mhc::MHCStatTrait<SPLIT_K, kNumWeightParts, kNumStatMMABlocks, kNumStatReduceBlocks>;

  static constexpr auto kernel =
      mhc_sp_fusion_p2p_kernel<kWorldSize, kRowsPerCTA, kStats, StatTrait, kFP8AG, kEpilogue, kStore256b>;
  static constexpr uint32_t kBlockSize = mhc_p2p::kGroupThreads * kRowsPerCTA;
  using CommSmemT = mhc_p2p::CommSmem<kRowsPerCTA, kWorldSize>;
  static constexpr size_t kSmemBytes =
      kStats ? sizeof(mhc_p2p::BoundarySmem<kRowsPerCTA, kWorldSize, StatTrait>) : sizeof(CommSmemT);
  static_assert(kWorldSize == 2, "the ring is sized for world 2; p2p transport is only picked there");
  static_assert(kSmemBytes <= device::mhc::kSmemCliffBytes, "past the shared cliff; raise SPLIT_K");
  static_assert(sizeof(device::atomic::Event) == sizeof(uint32_t), "counters is one int32 per Event");
  static_assert(kEpilogue || !kFP8AG, "the MXFP8 all-gather is epilogue output");

  static void
  run(const host::distributed::CommunicatorRef communicator,  // barriered on; its buffers are untouched
      const std::vector<int64_t> y_ptrs,                      // [kWorldSize] every rank's symmetric y
      const std::vector<int64_t> x_ptrs,                      // [kWorldSize] every rank's symmetric x
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView residual_out,
      const tvm::ffi::TensorView post,
      const tvm::ffi::TensorView comb,
      const tvm::ffi::TensorView pre,
      const tvm::ffi::Optional<tvm::ffi::TensorView> weight,  // RMSNorm weight; fold_collapseed only under kEpilogue
      const int64_t row_offset,
      const int64_t total_rows,
      const float eps,
      const int64_t num_blocks,
      // statistics (ignored unless kStats)
      const tvm::ffi::Optional<tvm::ffi::TensorView> hc_w,
      const tvm::ffi::Optional<tvm::ffi::TensorView> hc_scale,
      const tvm::ffi::Optional<tvm::ffi::TensorView> hc_base,
      const tvm::ffi::Optional<tvm::ffi::TensorView> workspace,  // fp32 partial mixes
      const tvm::ffi::Optional<tvm::ffi::TensorView> counters,   // int32 [2 * tiles], zeroed once
      const float rms_eps,
      const float hc_eps) {
    using namespace host;
    using namespace device::mhc;

    constexpr int64_t S = kNumStreams;
    constexpr int64_t D = kHidden;
    auto M = SymbolicSize{"num_rows"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({M, S, D}).with_dtype<bf16_t>().with_device(dev).verify(residual).verify(residual_out);
    TensorMatcher({M, S}).with_dtype<float>().with_device(dev).verify(post).verify(pre);
    TensorMatcher({M, S, S}).with_dtype<float>().with_device(dev).verify(comb);
    if constexpr (kEpilogue) {
      TensorMatcher({D}).with_dtype<bf16_t>().with_device(dev).verify(weight.value());
    }
    if constexpr (kStats) {
      TensorMatcher({kNumWeightParts, N, K}).with_dtype<bf16_t>().with_device(dev).verify(hc_w.value());
      TensorMatcher({3}).with_dtype<float>().with_device(dev).verify(hc_scale.value());
      TensorMatcher({N}).with_dtype<float>().with_device(dev).verify(hc_base.value());
    }

    // The semaphores are the all-reduce pull plane's: this kernel only barriers on them
    // and reduces in place on the caller's own symmetric y / x.
    const auto& pull = communicator.get()->get_pull_obj();
    const auto num_rows = static_cast<uint32_t>(M.unwrap());
    const auto num_comm_blocks = static_cast<uint32_t>(num_blocks);
    const auto num_tiles = div_ceil(num_rows, M_TILE);
    const auto split = row_split(static_cast<uint32_t>(total_rows), pull.rank, kWorldSize);

    CHECK_HOST(pull.world_size == kWorldSize);
    CHECK_HOST(num_comm_blocks <= pull.num_blocks);
    CHECK_HOST(y_ptrs.size() == kWorldSize && x_ptrs.size() == kWorldSize);
    // A null peer pointer faults deep inside the kernel; say so here instead.
    for (uint32_t r = 0; r < kWorldSize; ++r) {
      CHECK_HOST(y_ptrs[r] && (x_ptrs[r] || !kEpilogue)) << "peer " << r << " has a null symmetric buffer";
    }
    if constexpr (kFP8AG) {
      CHECK_HOST(y_ptrs[0] != x_ptrs[0])
          << "MXFP8 all-gather writes a scale plane past the values, so y and x cannot alias";
    }
    CHECK_HOST(static_cast<uint32_t>(row_offset) == split.offset && num_rows == split.count)
        << "the caller's row split disagrees: rank " << pull.rank << " passed [" << row_offset << ", +" << num_rows
        << ") of " << total_rows << ", this kernel owns [" << split.offset << ", +" << split.count << ")";
    if constexpr (kStats) {
      CHECK_HOST(workspace.value().numel() >= int64_t(num_tiles) * SPLIT_K * M_TILE * StatTrait::kPartialStride);
      CHECK_HOST(counters.value().numel() >= int64_t(2 * num_tiles));
    }

    const auto residual_ptr = static_cast<const bf16_t*>(residual.data_ptr());
    const auto post_ptr = static_cast<float*>(post.data_ptr());
    const auto comb_ptr = static_cast<float*>(comb.data_ptr());
    const auto pre_ptr = static_cast<float*>(pre.data_ptr());
    const auto norm_weight = [&]() -> const bf16_t* {
      if constexpr (kEpilogue) {
        return static_cast<const bf16_t*>(weight.value().data_ptr());
      } else {
        return nullptr;
      }
    }();
    const auto stat_params = [&]() -> MHCStatParams {
      if constexpr (!kStats) {
        return {};
      } else {
        auto* const counter_base = static_cast<device::atomic::Event*>(counters.value().data_ptr());
        return {
            .residual = residual_ptr,
            .weight = static_cast<const bf16_t*>(hc_w.value().data_ptr()),
            .scale = static_cast<const float*>(hc_scale.value().data_ptr()),
            .base = static_cast<const float*>(hc_base.value().data_ptr()),
            .pre = pre_ptr,
            .post = post_ptr,
            .comb = comb_ptr,
            .partial = static_cast<float*>(workspace.value().data_ptr()),
            .done = counter_base,
            .ready = counter_base + num_tiles,
            .num_rows = num_rows,
            .rms_eps = rms_eps,
            .hc_eps = hc_eps,
        };
      }
    }();
    const auto params = [&] {
      mhc_p2p::Params<kWorldSize> p{
          .residual = residual_ptr,
          .residual_out = static_cast<bf16_t*>(residual_out.data_ptr()),
          .post = post_ptr,
          .comb = comb_ptr,
          .pre = pre_ptr,
          .norm_weight = norm_weight,
          .rank = pull.rank,
          .num_rows = num_rows,
          .row_offset = static_cast<uint32_t>(row_offset),
          .total_rows = static_cast<uint32_t>(total_rows),
          .eps = eps,
          .num_comm_blocks = num_comm_blocks,
          .stats = stat_params,
      };
      for (uint32_t r = 0; r < kWorldSize; ++r) {
        p.y_peer[r] = reinterpret_cast<const bf16_t*>(y_ptrs[r]);
        p.x_peer[r] = reinterpret_cast<bf16_t*>(x_ptrs[r]);
        p.semaphores[r] = pull.semaphores[r];
      }
      return p;
    }();

    CHECK_CUDA(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    const auto device = dev.unwrap();
    const auto grid = num_comm_blocks + (kStats ? StatTrait::kNumBlocks : 0u);
    const auto num_sm = runtime::get_sm_count(device.device_id);
    CHECK_HOST(grid <= num_sm) << "the grid must be co-resident: " << grid << " blocks, " << num_sm << " SM";
    LaunchKernel(grid, kBlockSize, dev.unwrap(), kSmemBytes).launch(kernel, params);
  }
};

}  // namespace sglang
