// DeepSeek-V4.1 Engram: the table gather and the gate + mHC-combine seam, one
// launch each (SP / DP / TP variants of the seam).
//
// The table is sharded over the TP ranks on hash-column boundaries
// (`shard_range` in srt/layers/engram.py), so an id's owner follows from its
// column alone; shards live in symmetric memory and peers read them in place.
// A table row is already an MXFP8 row, so gather_mxfp8 copies the bytes
// verbatim into wkv's A operand (128x4-swizzled scales): no dequant, no
// requant. The seam kernels consume the previous boundary's lagged `pre`, so
// nothing waits on a coefficient and R' never round-trips through memory.
//
// The SP seam barriers on the pull plane like the mhc boundary kernels: a
// relaxed entry arrival before the first peer store, a rel_acq exit so every
// rank's next read of x sees every peer's rows. All ranks must launch the
// same grid or the semaphore rounds diverge.
#pragma once
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/dsv41/engram.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/stl.h>

#include <cstdint>
#include <ios>
#include <vector>

namespace sglang {

namespace engram_fusion {

using namespace device::engram;
using device::mhc::Vec4;

// ---------------------------------------------------------------------------
// gather_mxfp8
// ---------------------------------------------------------------------------

template <uint32_t kWarpsPerCTA_, uint32_t kRowsPerWarp_>
struct GatherShape {
  static constexpr uint32_t kBytesPerLane = sizeof(table_vec_t);
  static constexpr uint32_t kLanesPerRow = kHeadDim / kBytesPerLane;
  static constexpr uint32_t kRowsPerSweep = device::kWarpThreads / kLanesPerRow;
  static constexpr uint32_t kRowsPerWarp = kRowsPerWarp_;
  static constexpr uint32_t kNumSweeps = kRowsPerWarp / kRowsPerSweep;
  static constexpr uint32_t kWarpsPerToken = kCols / kRowsPerWarp;
  static_assert(kRowsPerWarp % kRowsPerSweep == 0 && kCols % kRowsPerWarp == 0);
  static constexpr uint32_t kWarpsPerCTA = kWarpsPerCTA_;
  static constexpr uint32_t kCTASize = kWarpsPerCTA * device::kWarpThreads;
};

using GatherDecode = GatherShape<1, 4>;
using GatherPrefill = GatherShape<2, 8>;

struct ScalePlane {
  uint8_t* __restrict__ base;  // [m_pad / 128, kSfKTiles, 512] ue8m0 bytes

  SGL_DEVICE void store(uint32_t m, uint32_t col, const device::AlignedVector<uint32_t, 2>& scales) const {
#pragma unroll
    for (uint32_t half = 0; half < 2; ++half) {
      const uint32_t tile = (m >> 7) * kSfKTiles + col * (kScalesPerRow / 4) + half;
      const uint32_t within = (m & 31u) * 16u + ((m >> 5) & 3u) * 4u;
      *reinterpret_cast<uint32_t*>(base + tile * 512u + within) = scales[half];
    }
  }
};

template <uint32_t kWorldSize>
struct GatherParams {
  const int32_t* __restrict__ ids;  // [num_tokens, 24], row stride ids_stride
  uint8_t* __restrict__ out_a;      // [m_pad, 6144] e4m3 bytes
  ScalePlane out_sf;
  // Rank r's shard base, pre-biased by its first global row, so a global id
  // indexes it directly; the owner follows from the column (kColsPerRank each).
  const uint8_t* w[kWorldSize];
  const uint8_t* s[kWorldSize];
  int64_t ids_stride;
  uint32_t num_tokens;
  uint32_t m_pad;
};

template <uint32_t kWorldSize, typename Shape>
__global__ __launch_bounds__(Shape::kCTASize)  //
    void gather_mxfp8_kernel(const __grid_constant__ GatherParams<kWorldSize> params) {
  using namespace device;
  constexpr uint32_t kColsPerRank = kCols / kWorldSize;
  constexpr uint32_t kRowsPerWarp = Shape::kRowsPerWarp;
  constexpr uint32_t kWarpsPerToken = Shape::kWarpsPerToken;
  constexpr uint32_t kWarpsPerCTA = Shape::kWarpsPerCTA;
  constexpr uint32_t kLanesPerRow = Shape::kLanesPerRow;
  constexpr uint32_t kRowsPerSweep = Shape::kRowsPerSweep;
  constexpr uint32_t kBytesPerLane = Shape::kBytesPerLane;
  constexpr uint32_t kNumSweeps = Shape::kNumSweeps;
  static_assert(kCols % kWorldSize == 0, "the shards cut on hash-column boundaries");
  const auto bx = blockIdx.x;
  const auto tx = threadIdx.x;
  const auto lane_id = tx % kWarpThreads;
  const auto global_warp_id = bx * kWarpsPerCTA + tx / kWarpThreads;
  const auto m = global_warp_id / kWarpsPerToken;
  const auto col = (global_warp_id % kWarpsPerToken) * kRowsPerWarp;
  // This lane's slice of a row, and which warp-local row it serves on pass i.
  const auto byte_of_row = (lane_id % kLanesPerRow) * kBytesPerLane;
  const auto row_lane_id = lane_id / kLanesPerRow;
  const auto row_of = [&](uint32_t i) { return i * kRowsPerSweep + row_lane_id; };
  if (m >= params.m_pad) return;
  const auto out_row = params.out_a + static_cast<uint64_t>(m) * K + col * kHeadDim;
  if (m >= params.num_tokens) {
    // padding rows of the 128-row tile: zero payload, 2^-127 scales
    table_vec_t zero;
    zero.fill(0u);
#pragma unroll
    for (uint32_t i = 0; i < kNumSweeps; ++i) {
      zero.store(out_row + (i * kRowsPerSweep + row_lane_id) * kHeadDim + byte_of_row);
    }
    if (lane_id < kRowsPerWarp) {
      AlignedVector<uint32_t, 2> zero_sf;
      zero_sf.fill(0u);
      params.out_sf.store(m, col + lane_id, zero_sf);
    }
    return;
  }
  uint32_t my_id = 0;
  if (lane_id < kRowsPerWarp) {
    my_id = params.ids[static_cast<int64_t>(m) * params.ids_stride + col + lane_id];
  }
  table_vec_t payload[kNumSweeps];
#pragma unroll
  for (uint32_t i = 0; i < kNumSweeps; ++i) {
    const uint32_t row = row_of(i);
    const uint32_t id = __shfl_sync(0xffffffffu, my_id, row);
    payload[i].load(params.w[(col + row) / kColsPerRank] + static_cast<uint64_t>(id) * kHeadDim + byte_of_row);
  }
  AlignedVector<uint32_t, 2> scales;
  scales.fill(0u);
  if (lane_id < kRowsPerWarp) {
    scales.load(params.s[(col + lane_id) / kColsPerRank] + static_cast<uint64_t>(my_id) * kScalesPerRow);
  }
#pragma unroll
  for (uint32_t i = 0; i < kNumSweeps; ++i) {
    payload[i].store(out_row + row_of(i) * kHeadDim + byte_of_row);
  }
  if (lane_id < kRowsPerWarp) {
    params.out_sf.store(m, col + lane_id, scales);
  }
}

template <uint32_t kWorldSize>
struct GatherBf16Params {
  const int32_t* __restrict__ ids;  // [num_tokens, 24], row stride ids_stride
  bf16_t* __restrict__ out;         // [num_tokens, 6144]
  // Rank r's shard bases, pre-biased as in GatherParams.
  const uint8_t* w[kWorldSize];
  const uint8_t* s[kWorldSize];
  int64_t ids_stride;
  uint32_t num_tokens;
};

/// The gather dequantized to bf16
template <uint32_t kWorldSize, typename Shape>
__global__ __launch_bounds__(Shape::kCTASize)  //
    void gather_bf16_kernel(const __grid_constant__ GatherBf16Params<kWorldSize> params) {
  using namespace device;
  constexpr uint32_t kColsPerRank = kCols / kWorldSize;
  constexpr uint32_t kLanesPerRow = Shape::kLanesPerRow;
  constexpr uint32_t kBytesPerLane = Shape::kBytesPerLane;
  const auto tx = threadIdx.x;
  const auto lane_id = tx % kWarpThreads;
  const auto global_warp_id = blockIdx.x * Shape::kWarpsPerCTA + tx / kWarpThreads;
  const auto m = global_warp_id / Shape::kWarpsPerToken;
  const auto col = (global_warp_id % Shape::kWarpsPerToken) * Shape::kRowsPerWarp;
  const auto byte_of_row = (lane_id % kLanesPerRow) * kBytesPerLane;
  const auto row_lane_id = lane_id / kLanesPerRow;
  if (m >= params.num_tokens) return;
  uint32_t my_id = 0;
  if (lane_id < Shape::kRowsPerWarp) {
    my_id = params.ids[static_cast<int64_t>(m) * params.ids_stride + col + lane_id];
  }
  const auto out_row = params.out + static_cast<uint64_t>(m) * K + col * kHeadDim;
#pragma unroll
  for (uint32_t i = 0; i < Shape::kNumSweeps; ++i) {
    const uint32_t row = i * Shape::kRowsPerSweep + row_lane_id;
    const uint32_t id = __shfl_sync(0xffffffffu, my_id, row);
    const auto owner = (col + row) / kColsPerRank;
    table_vec_t payload;
    payload.load(params.w[owner] + static_cast<uint64_t>(id) * kHeadDim + byte_of_row);
    const float scale = e8m0_to_float(params.s[owner][static_cast<uint64_t>(id) * kScalesPerRow + (byte_of_row >> 5)]);
    const auto bytes = reinterpret_cast<const fp8x2_e4m3_t*>(&payload);
    vec_t lo, hi;
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      const auto a = cast<fp32x2_t>(bytes[e]);
      const auto b = cast<fp32x2_t>(bytes[e + 4]);
      lo[e] = cast<bf16x2_t>(fp32x2_t{a.x * scale, a.y * scale});
      hi[e] = cast<bf16x2_t>(fp32x2_t{b.x * scale, b.y * scale});
    }
    lo.store(out_row + row * kHeadDim + byte_of_row);
    hi.store(out_row + row * kHeadDim + byte_of_row + 8);
  }
}

// ---------------------------------------------------------------------------
// the seam: gate + apply + combine / norm (+ all-gather)
// ---------------------------------------------------------------------------
// The boundary kernel's shape (mhc_sp_fusion_p2p's comm role, minus the ring):
// a group of kGroupThreads owns one row at a time, assigned by a static stride.

inline constexpr uint32_t kGroupThreads = 128;
// Flat shape for every seam kernel: strided rows already spread small-M work,
// so the 2-row small-M variant is retired (engram-dp-order / engram-sp-order).
inline constexpr uint32_t kSeamRowsPerCTA = 4;
inline constexpr uint32_t kRowVecs = D / 8;
inline constexpr uint32_t kVecsPerThread = kRowVecs / kGroupThreads;  // 5
inline constexpr uint32_t kWarpsPerGroup = kGroupThreads / device::kWarpThreads;
static_assert(kRowVecs % kGroupThreads == 0);

template <uint32_t kWorldSize>
struct GateCombineParams {
  // Symmetric next-input buffers, indexed by rank; the semaphores are the
  // all-reduce pull plane's, only barriered on.
  bf16_t* x_peer[kWorldSize];  // [T_pad, 5120] bf16
  device::distributed::Semaphore* semaphores[kWorldSize];
  bf16_t* residual;                        // [M, 4, 5120] this rank's rows, updated in place
  const __nv_bfloat16* __restrict__ kv;    // [M, 25600]
  const __nv_bfloat16* __restrict__ qw;    // [4, 5120]
  const __nv_bfloat16* __restrict__ kw;    // [4, 5120]
  const float* __restrict__ pre;           // [M, 4], the previous boundary's lagged pre
  const bf16_t* __restrict__ norm_weight;  // [5120]
  const uint8_t* __restrict__ skip;        // [M] nonzero keeps the row's residual; null for none
  uint32_t rank;
  uint32_t num_rows;    // M
  uint32_t row_offset;  // this rank's first row of T_pad
  float eps;            // the gate's rms eps
  float clamp;
  float rms_eps;  // the next norm's eps
};

// The three weights every row re-reads, staged once per CTA over TMA; the rest
// is the groups' reduction scratch.
template <uint32_t kRowsPerCTA>
struct GateCombineSmem {
  __align__(128) bf16_t norm_weight[D];
  __align__(128) bf16_t q_weight[kHC * D];
  __align__(128) bf16_t k_weight[kHC * D];
  float red[kRowsPerCTA][kWarpsPerGroup][3 * kHC];
  float gates[kRowsPerCTA][kHC];
  float warp_sums[kRowsPerCTA][kWarpsPerGroup];
  uint64_t weights_arrived;
};

extern __shared__ __align__(16) char smem_base[];

/// A group's first row: rows spread one per CTA before a CTA takes its second,
/// so small batches wake as many SMs as they have rows (B300 A/B, -19..25% at
/// decode sizes, tied at 1K-8K; the stride is gridDim.x * kRowsPerCTA).
template <uint32_t kRowsPerCTA>
SGL_DEVICE uint32_t first_row_of(uint32_t group) {
  return blockIdx.x + group * gridDim.x;
}

/// The per-row body shared bitwise by the SP and DP kernels; `store_x(v, out)`
/// is the only seam-specific piece. `skip_update` keeps the row's residual
/// (image tokens) while the combine still folds the old row.
template <uint32_t kRowsPerCTA, typename StoreX>
SGL_DEVICE void gate_combine_norm_row(
    GateCombineSmem<kRowsPerCTA>& smem,
    const uint32_t group,
    const uint32_t slot,
    const uint32_t warp_in_group,
    bf16_t* const x_row,
    const __nv_bfloat16* const kv_row,
    const float* const pre_row,
    const bool skip_update,
    const float eps,
    const float clamp,
    const float rms_eps,
    StoreX store_x) {
  using namespace device;
  namespace ptx = device::mhc::ptx;

  fp32x2_t x_sq[kHC] = {}, key_sq[kHC] = {}, dot[kHC] = {};
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    const auto v = slot + j * kGroupThreads;
#pragma unroll
    for (uint32_t h = 0; h < kHC; ++h) {
      vec_t x_vec, key_vec, q_weight, k_weight;
      x_vec.load(x_row + h * D, v);
      key_vec.load(kv_row + h * D, v);
      q_weight.load(smem.q_weight + h * D, v);
      k_weight.load(smem.k_weight + h * D, v);
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        const auto x = cast<fp32x2_t>(x_vec[e]);
        const auto k = cast<fp32x2_t>(key_vec[e]);
        const auto q = cast<fp32x2_t>(q_weight[e]);
        const auto w = cast<fp32x2_t>(k_weight[e]);
        x_sq[h] = math::fma_f32x2(x, x, x_sq[h]);
        key_sq[h] = math::fma_f32x2(k, k, key_sq[h]);
        const fp32x2_t x_weighted{x.x * (q.x * w.x), x.y * (q.y * w.y)};
        dot[h] = math::fma_f32x2(x_weighted, k, dot[h]);
      }
    }
  }
#pragma unroll
  for (uint32_t h = 0; h < kHC; ++h) {
    const float folded[3] = {x_sq[h].x + x_sq[h].y, key_sq[h].x + key_sq[h].y, dot[h].x + dot[h].y};
#pragma unroll
    for (uint32_t i = 0; i < 3; ++i) {
      smem.red[group][warp_in_group][3 * h + i] = warp::reduce_sum(folded[i]);
    }
  }
  ptx::bar_sync(group + 1, kGroupThreads);
  if (slot < kHC) {
    float x_sq_total = 0.f, key_sq_total = 0.f, dot_total = 0.f;
#pragma unroll
    for (uint32_t w = 0; w < kWarpsPerGroup; ++w) {
      x_sq_total += smem.red[group][w][3 * slot];
      key_sq_total += smem.red[group][w][3 * slot + 1];
      dot_total += smem.red[group][w][3 * slot + 2];
    }
    smem.gates[group][slot] = gate_from_sums(x_sq_total, key_sq_total, dot_total, eps, clamp);
  }
  ptx::bar_sync(group + 1, kGroupThreads);

  float gates[kHC];
#pragma unroll
  for (uint32_t h = 0; h < kHC; ++h)
    gates[h] = smem.gates[group][h];
  // Every thread reads the same 16B of pre: one broadcast load, no staging.
  Vec4 pre;
  pre.load(pre_row);

  vec_t collapsed[kVecsPerThread];
  float sum_squares = 0.f;
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    const uint32_t v = slot + j * kGroupThreads;
    vec_t value;
    value.load(kv_row + kHC * D, v);
    fp32x2_t collapse_acc[4] = {};
#pragma unroll
    for (uint32_t h = 0; h < kHC; ++h) {
      vec_t old, updated;
      old.load(x_row + h * D, v);
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        const auto gated = math::fma_f32x2({gates[h], gates[h]}, cast<fp32x2_t>(value[e]), cast<fp32x2_t>(old[e]));
        // A select, not gate = 0: fma(0, inf, old) would poison the kept row.
        updated[e] = skip_update ? old[e] : cast<bf16x2_t>(gated);
        // The next combine consumes the rounded residual, as the boundary kernel does.
        collapse_acc[e] = math::fma_f32x2({pre[h], pre[h]}, cast<fp32x2_t>(updated[e]), collapse_acc[e]);
      }
      updated.store(x_row + h * D, v);
    }
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      collapsed[j][e] = cast<bf16x2_t>(collapse_acc[e]);
      sum_squares = math::fma_chain2_f32_bf16(collapsed[j][e], collapsed[j][e], sum_squares);
    }
  }
  smem.warp_sums[group][warp_in_group] = warp::reduce_sum(sum_squares);
  ptx::bar_sync(group + 1, kGroupThreads);
  float row_total = 0.f;
#pragma unroll
  for (uint32_t w = 0; w < kWarpsPerGroup; ++w)
    row_total += smem.warp_sums[group][w];
  const auto inv_rms = math::rsqrt(row_total / float(D) + rms_eps);
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    const uint32_t v = slot + j * kGroupThreads;
    vec_t weight, out;
    weight.load(smem.norm_weight, v);
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      const auto scaled = math::fma_f32x2_bf16x2(collapsed[j][e], weight[e], {0.f, 0.f});
      out[e] = cast<bf16x2_t>(fp32x2_t{scaled.x * inv_rms, scaled.y * inv_rms});
    }
    store_x(v, out);
  }
}

template <uint32_t kWorldSize, uint32_t kRowsPerCTA>
__global__ __launch_bounds__(kGroupThreads* kRowsPerCTA, 1)  //
    void sp_gate_mhc_combine_norm_kernel(const __grid_constant__ GateCombineParams<kWorldSize> params) {
  using namespace device;
  namespace ptx = device::mhc::ptx;
  static_assert(kRowsPerCTA + 1 <= 16, "one named barrier per group");
  auto& smem = *reinterpret_cast<GateCombineSmem<kRowsPerCTA>*>(smem_base);
  const auto tx = threadIdx.x;
  const auto group = tx / kGroupThreads;
  const auto slot = tx % kGroupThreads;
  const auto warp_in_group = slot / kWarpThreads;

  // Warp 1, so the copies overlap warp 0's barrier arrival. Initializing the
  // mbarrier on the thread that uses it needs no fence (the exchange lowering
  // carries an implicit one); the __syncthreads below publishes the weights.
  if (tx / kWarpThreads == 1 && warp::elect_one_lane()) {
    constexpr uint32_t kWeightBytes = sizeof(smem.norm_weight) + sizeof(smem.q_weight) + sizeof(smem.k_weight);
    ptx::mbar_init(&smem.weights_arrived, 1);
    ptx::mbar_arrive_expect_tx(&smem.weights_arrived, kWeightBytes);
    ptx::cp_async_bulk_g2s(smem.norm_weight, params.norm_weight, sizeof(smem.norm_weight), &smem.weights_arrived);
    ptx::cp_async_bulk_g2s(smem.q_weight, params.qw, sizeof(smem.q_weight), &smem.weights_arrived);
    ptx::cp_async_bulk_g2s(smem.k_weight, params.kw, sizeof(smem.k_weight), &smem.weights_arrived);
    ptx::mbar_wait_parity(&smem.weights_arrived, 0);
  }
  const distributed::Barrier<kWorldSize> barrier{params.semaphores, params.rank, 2};
  barrier.arrive_relaxed(0);
  __syncthreads();

  const auto row_stride = gridDim.x * kRowsPerCTA;
#pragma unroll 1
  for (uint32_t row = first_row_of<kRowsPerCTA>(group); row < params.num_rows; row += row_stride) {
    const auto x_row = params.residual + static_cast<uint64_t>(row) * kHC * D;
    const auto kv_row = params.kv + static_cast<uint64_t>(row) * KV;
    const auto global_row = static_cast<uint64_t>(params.row_offset + row);
    gate_combine_norm_row<kRowsPerCTA>(
        smem,
        group,
        slot,
        warp_in_group,
        x_row,
        kv_row,
        params.pre + static_cast<uint64_t>(row) * kHC,
        params.skip != nullptr && params.skip[row] != 0,
        params.eps,
        params.clamp,
        params.rms_eps,
        [&](uint32_t v, const vec_t& out) {
#pragma unroll
          for (uint32_t r = 0; r < kWorldSize; ++r) {
            out.store(params.x_peer[r], global_row * kRowVecs + v);
          }
        });
  }

  __syncthreads();
  barrier.arrive_rel_acq(1);
}

// ---------------------------------------------------------------------------
// DP mode: every rank owns its tokens entirely -- the same seam, no comm
// ---------------------------------------------------------------------------
// The SP body minus the plane: no semaphores, no row_offset, a local x_out,
// and a free grid (nothing barriers).

struct DpGateCombineParams {
  bf16_t* residual;                        // [M, 4, 5120] updated in place
  const __nv_bfloat16* __restrict__ kv;    // [M, 25600]
  const __nv_bfloat16* __restrict__ qw;    // [4, 5120]
  const __nv_bfloat16* __restrict__ kw;    // [4, 5120]
  const float* __restrict__ pre;           // [M, 4], the previous boundary's lagged pre
  const bf16_t* __restrict__ norm_weight;  // [5120]
  const uint8_t* __restrict__ skip;        // [M] nonzero keeps the row's residual; null for none
  bf16_t* __restrict__ x_out;              // [M, 5120]
  uint32_t num_rows;                       // M
  float eps;                               // the gate's rms eps
  float clamp;
  float rms_eps;  // the next norm's eps
};

template <uint32_t kRowsPerCTA>
__global__ __launch_bounds__(kGroupThreads* kRowsPerCTA, 1)  //
    void dp_gate_mhc_combine_norm_kernel(const __grid_constant__ DpGateCombineParams params) {
  using namespace device;
  namespace ptx = device::mhc::ptx;
  static_assert(kRowsPerCTA + 1 <= 16, "one named barrier per group");
  auto& smem = *reinterpret_cast<GateCombineSmem<kRowsPerCTA>*>(smem_base);
  const auto tx = threadIdx.x;
  const auto group = tx / kGroupThreads;
  const auto slot = tx % kGroupThreads;
  const auto warp_in_group = slot / kWarpThreads;

  // Warp 1, exactly as the SP kernel stages its weights.
  if (tx / kWarpThreads == 1 && warp::elect_one_lane()) {
    constexpr uint32_t kWeightBytes = sizeof(smem.norm_weight) + sizeof(smem.q_weight) + sizeof(smem.k_weight);
    ptx::mbar_init(&smem.weights_arrived, 1);
    ptx::mbar_arrive_expect_tx(&smem.weights_arrived, kWeightBytes);
    ptx::cp_async_bulk_g2s(smem.norm_weight, params.norm_weight, sizeof(smem.norm_weight), &smem.weights_arrived);
    ptx::cp_async_bulk_g2s(smem.q_weight, params.qw, sizeof(smem.q_weight), &smem.weights_arrived);
    ptx::cp_async_bulk_g2s(smem.k_weight, params.kw, sizeof(smem.k_weight), &smem.weights_arrived);
    ptx::mbar_wait_parity(&smem.weights_arrived, 0);
  }
  __syncthreads();

  const auto row_stride = gridDim.x * kRowsPerCTA;
#pragma unroll 1
  for (uint32_t row = first_row_of<kRowsPerCTA>(group); row < params.num_rows; row += row_stride) {
    const auto x_row = params.residual + static_cast<uint64_t>(row) * kHC * D;
    const auto kv_row = params.kv + static_cast<uint64_t>(row) * KV;
    gate_combine_norm_row<kRowsPerCTA>(
        smem,
        group,
        slot,
        warp_in_group,
        x_row,
        kv_row,
        params.pre + static_cast<uint64_t>(row) * kHC,
        params.skip != nullptr && params.skip[row] != 0,
        params.eps,
        params.clamp,
        params.rms_eps,
        [&](uint32_t v, const vec_t& out) { out.store(params.x_out, static_cast<uint64_t>(row) * kRowVecs + v); });
  }
}

// ---------------------------------------------------------------------------
// TP mode: gate on this rank's token share, then a purely local seam
// ---------------------------------------------------------------------------
// Split in two so only a small kernel touches the pull plane: tp_gate_push
// gates this rank's token share and pushes (value, gates) to every rank's
// staging plane; tp_update_combine_norm then runs all T rows purely locally,
// so its grid is free. A staging plane is [T, 5120] bf16 values then [T, 4]
// fp32 gates, split only in get_staging_ptr.

template <uint32_t kWorldSize>
struct TpGatePushParams {
  // Every rank's staging plane, pre-split by the host: values then gates.
  bf16_t* values[kWorldSize];  // [T, 5120] bf16
  float* gates[kWorldSize];    // [T, 4] fp32
  device::distributed::Semaphore* semaphores[kWorldSize];
  const bf16_t* residual;                // [T, 4, 5120] replicated; only our share is read
  const __nv_bfloat16* __restrict__ kv;  // [M, 25600] this rank's share, local
  const __nv_bfloat16* __restrict__ qw;  // [4, 5120]
  const __nv_bfloat16* __restrict__ kw;  // [4, 5120]
  uint32_t rank;
  uint32_t num_rows;    // M = T / kWorldSize
  uint32_t row_offset;  // rank * M
  float eps;
  float clamp;
};

template <uint32_t kRowsPerCTA>
struct TpGatePushSmem {
  float red[kRowsPerCTA][kWarpsPerGroup][3 * kHC];
  float gates[kRowsPerCTA][kHC];
};

// Not the hot path (SP covers extend), so the gate weights read straight from
// global per row instead of a TMA stage: static shared, no attribute dance.
template <uint32_t kWorldSize, uint32_t kRowsPerCTA>
__global__ __launch_bounds__(kGroupThreads* kRowsPerCTA, 1)  //
    void tp_gate_push_kernel(const __grid_constant__ TpGatePushParams<kWorldSize> params) {
  using namespace device;
  namespace ptx = device::mhc::ptx;
  static_assert(kRowsPerCTA + 1 <= 16, "one named barrier per group");
  __shared__ TpGatePushSmem<kRowsPerCTA> smem;
  const auto tx = threadIdx.x;
  const auto group = tx / kGroupThreads;
  const auto slot = tx % kGroupThreads;
  const auto lane_id = tx % kWarpThreads;
  const auto warp_in_group = slot / kWarpThreads;

  const distributed::Barrier<kWorldSize> barrier{params.semaphores, params.rank, 2};
  barrier.arrive_relaxed(0);
  __syncthreads();

  const uint32_t row_stride = gridDim.x * kRowsPerCTA;
#pragma unroll 1
  for (uint32_t row = blockIdx.x * kRowsPerCTA + group; row < params.num_rows; row += row_stride) {
    const uint64_t global_row = params.row_offset + row;
    const bf16_t* x_row = params.residual + global_row * kHC * D;
    const bf16_t* kv_row = params.kv + static_cast<uint64_t>(row) * KV;

    fp32x2_t x_sq[kHC] = {}, key_sq[kHC] = {}, dot[kHC] = {};
#pragma unroll
    for (uint32_t j = 0; j < kVecsPerThread; ++j) {
      const uint32_t v = slot + j * kGroupThreads;
#pragma unroll
      for (uint32_t h = 0; h < kHC; ++h) {
        vec_t x_vec, key_vec, q_weight, k_weight;
        x_vec.load(x_row + h * D, v);
        key_vec.load(kv_row + h * D, v);
        q_weight.load(params.qw + h * D, v);
        k_weight.load(params.kw + h * D, v);
#pragma unroll
        for (uint32_t e = 0; e < 4; ++e) {
          const auto x = cast<fp32x2_t>(x_vec[e]);
          const auto k = cast<fp32x2_t>(key_vec[e]);
          const auto q = cast<fp32x2_t>(q_weight[e]);
          const auto w = cast<fp32x2_t>(k_weight[e]);
          x_sq[h] = math::fma_f32x2(x, x, x_sq[h]);
          key_sq[h] = math::fma_f32x2(k, k, key_sq[h]);
          const fp32x2_t x_weighted{x.x * (q.x * w.x), x.y * (q.y * w.y)};
          dot[h] = math::fma_f32x2(x_weighted, k, dot[h]);
        }
      }
    }
#pragma unroll
    for (uint32_t h = 0; h < kHC; ++h) {
      const float folded[3] = {x_sq[h].x + x_sq[h].y, key_sq[h].x + key_sq[h].y, dot[h].x + dot[h].y};
#pragma unroll
      for (uint32_t i = 0; i < 3; ++i) {
        const float total = warp::reduce_sum(folded[i]);
        if (lane_id == 0) smem.red[group][warp_in_group][3 * h + i] = total;
      }
    }
    ptx::bar_sync(group + 1, kGroupThreads);
    if (slot < kHC) {
      float x_sq_total = 0.f, key_sq_total = 0.f, dot_total = 0.f;
#pragma unroll
      for (uint32_t w = 0; w < kWarpsPerGroup; ++w) {
        x_sq_total += smem.red[group][w][3 * slot];
        key_sq_total += smem.red[group][w][3 * slot + 1];
        dot_total += smem.red[group][w][3 * slot + 2];
      }
      smem.gates[group][slot] = gate_from_sums(x_sq_total, key_sq_total, dot_total, params.eps, params.clamp);
    }
    ptx::bar_sync(group + 1, kGroupThreads);

#pragma unroll
    for (uint32_t j = 0; j < kVecsPerThread; ++j) {
      const uint32_t v = slot + j * kGroupThreads;
      vec_t value;
      value.load(kv_row + kHC * D, v);
#pragma unroll
      for (uint32_t r = 0; r < kWorldSize; ++r) {
        value.store(params.values[r], global_row * kRowVecs + v);
      }
    }
    if (slot < kHC) {
      const float gate = smem.gates[group][slot];
#pragma unroll
      for (uint32_t r = 0; r < kWorldSize; ++r)
        params.gates[r][global_row * kHC + slot] = gate;
    }
  }

  __syncthreads();
  barrier.arrive_rel_acq(1);
}

struct TpCombineParams {
  bf16_t* residual;                        // [T, 4, 5120] replicated, updated in place
  const bf16_t* __restrict__ values;       // [T, 5120] staged by every rank's tp_gate_push
  const float* __restrict__ gates;         // [T, 4] staged
  const float* __restrict__ pre;           // [T, 4], the previous boundary's lagged pre
  const bf16_t* __restrict__ norm_weight;  // [5120]
  const uint8_t* __restrict__ skip;        // [T] nonzero keeps the row's residual; null for none
  bf16_t* __restrict__ x_out;              // [T, 5120]
  uint32_t num_rows;                       // T
  float rms_eps;
};

template <uint32_t kRowsPerCTA>
struct TpCombineSmem {
  __align__(16) bf16_t norm_weight[D];
  float warp_sums[kRowsPerCTA][kWarpsPerGroup];
};

template <uint32_t kRowsPerCTA>
__global__ __launch_bounds__(kGroupThreads* kRowsPerCTA) void tp_update_combine_norm_kernel(
    const __grid_constant__ TpCombineParams params) {
  using namespace device;
  namespace ptx = device::mhc::ptx;
  static_assert(kRowsPerCTA + 1 <= 16, "one named barrier per group");
  __shared__ TpCombineSmem<kRowsPerCTA> smem;
  const auto tx = threadIdx.x;
  const auto group = tx / kGroupThreads;
  const auto slot = tx % kGroupThreads;
  const auto warp_in_group = (tx % kGroupThreads) / kWarpThreads;

  for (uint32_t v = tx; v < kRowVecs; v += kGroupThreads * kRowsPerCTA) {
    vec_t w;
    w.load(params.norm_weight, v);
    w.store(smem.norm_weight, v);
  }
  __syncthreads();

  const uint32_t row_stride = gridDim.x * kRowsPerCTA;
#pragma unroll 1
  for (uint32_t row = blockIdx.x * kRowsPerCTA + group; row < params.num_rows; row += row_stride) {
    bf16_t* x_row = params.residual + static_cast<uint64_t>(row) * kHC * D;
    const bool skip_update = params.skip != nullptr && params.skip[row] != 0;
    // Broadcast loads: every thread of the group reads the same 16B.
    Vec4 gates, pre;
    gates.load(params.gates + static_cast<uint64_t>(row) * kHC);
    pre.load(params.pre + static_cast<uint64_t>(row) * kHC);

    vec_t collapsed[kVecsPerThread];
    float sum_squares = 0.f;
#pragma unroll
    for (uint32_t j = 0; j < kVecsPerThread; ++j) {
      const uint32_t v = slot + j * kGroupThreads;
      vec_t value;
      value.load(params.values + static_cast<uint64_t>(row) * D, v);
      fp32x2_t collapse_acc[4] = {};
#pragma unroll
      for (uint32_t h = 0; h < kHC; ++h) {
        vec_t old, updated;
        old.load(x_row + h * D, v);
#pragma unroll
        for (uint32_t e = 0; e < 4; ++e) {
          const auto gated = math::fma_f32x2({gates[h], gates[h]}, cast<fp32x2_t>(value[e]), cast<fp32x2_t>(old[e]));
          // A select, not gate = 0: fma(0, inf, old) would poison the kept row.
          updated[e] = skip_update ? old[e] : cast<bf16x2_t>(gated);
          // The next combine consumes the rounded residual, as the boundary kernel does.
          collapse_acc[e] = math::fma_f32x2({pre[h], pre[h]}, cast<fp32x2_t>(updated[e]), collapse_acc[e]);
        }
        updated.store(x_row + h * D, v);
      }
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        collapsed[j][e] = cast<bf16x2_t>(collapse_acc[e]);
        sum_squares = math::fma_chain2_f32_bf16(collapsed[j][e], collapsed[j][e], sum_squares);
      }
    }
    smem.warp_sums[group][warp_in_group] = warp::reduce_sum(sum_squares);
    ptx::bar_sync(group + 1, kGroupThreads);
    float row_total = 0.f;
#pragma unroll
    for (uint32_t w = 0; w < kWarpsPerGroup; ++w)
      row_total += smem.warp_sums[group][w];
    // The second barrier orders this read against the next row's write; the SP
    // kernel gets that for free from its gate barriers.
    ptx::bar_sync(group + 1, kGroupThreads);
    const float inv_rms = math::rsqrt(row_total / float(D) + params.rms_eps);

#pragma unroll
    for (uint32_t j = 0; j < kVecsPerThread; ++j) {
      const uint32_t v = slot + j * kGroupThreads;
      vec_t weight, out;
      weight.load(smem.norm_weight, v);
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        const auto scaled = math::fma_f32x2_bf16x2(collapsed[j][e], weight[e], {0.f, 0.f});
        out[e] = cast<bf16x2_t>(fp32x2_t{scaled.x * inv_rms, scaled.y * inv_rms});
      }
      out.store(params.x_out, static_cast<uint64_t>(row) * kRowVecs + v);
    }
  }
}

}  // namespace engram_fusion

// ---------------------------------------------------------------------------
// host entry points
// ---------------------------------------------------------------------------

template <uint32_t kWorldSize>
struct EngramFusion {
  static_assert(kWorldSize >= 1 && kWorldSize <= device::engram::kMaxWorld);

  static void gather_mxfp8(
      const tvm::ffi::TensorView ids,
      const tvm::ffi::TensorView out_a,
      const tvm::ffi::TensorView out_sf,
      const std::vector<int64_t> w_ptrs,          // [kWorldSize] rank r's weight shard base
      const std::vector<int64_t> s_ptrs,          // [kWorldSize] rank r's scale shard base
      const std::vector<int64_t> shard_starts) {  // [kWorldSize] first table row of rank r's shard
    using namespace host;
    using namespace engram_fusion;

    auto M = SymbolicSize{"num_tokens"};
    auto P = SymbolicSize{"m_pad"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({M, static_cast<int64_t>(kCols)})  //
        .with_strides({-1, 1})
        .with_dtype<int32_t>()
        .with_device(dev)
        .verify(ids);
    TensorMatcher({P, K})  //
        .with_dtype<uint8_t>()
        .with_device(dev)
        .ensure_alignment(32)
        .verify(out_a);

    const auto m_pad = static_cast<uint32_t>(P.unwrap());
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    TensorMatcher({m_pad / 128 * kSfKTiles * 512})  // sf
        .with_dtype<uint8_t>()
        .with_device(dev)
        .verify(out_sf);

    CHECK_HOST(m_pad % 128 == 0 && m_pad >= num_tokens);
    CHECK_HOST(w_ptrs.size() == kWorldSize && s_ptrs.size() == kWorldSize && shard_starts.size() == kWorldSize);
    CHECK_HOST(shard_starts[0] == 0);
    for (uint32_t r = 0; r < kWorldSize; ++r) {
      CHECK_HOST(w_ptrs[r] != 0 && w_ptrs[r] % 32 == 0 && s_ptrs[r] && s_ptrs[r] % 8 == 0)
          << "peer " << r << " " << std::hex << w_ptrs[r] << " " << s_ptrs[r] << std::dec;
    }
    const auto params = [&] {
      GatherParams<kWorldSize> p{
          .ids = static_cast<const int32_t*>(ids.data_ptr()),
          .out_a = static_cast<uint8_t*>(out_a.data_ptr()),
          .out_sf = {static_cast<uint8_t*>(out_sf.data_ptr())},
          .ids_stride = ids.stride(0),
          .num_tokens = num_tokens,
          .m_pad = m_pad,
      };
      // The bias is a whole number of rows, so the 32B / 8B alignment survives.
      for (uint32_t r = 0; r < kWorldSize; ++r) {
        p.w[r] = reinterpret_cast<const uint8_t*>(w_ptrs[r]) - shard_starts[r] * kHeadDim;
        p.s[r] = reinterpret_cast<const uint8_t*>(s_ptrs[r]) - shard_starts[r] * kScalesPerRow;
      }
      return p;
    }();
    const auto device = dev.unwrap();
    if (m_pad <= 128) {
      using Shape = GatherDecode;
      const auto num_warps = m_pad * Shape::kWarpsPerToken;
      host::LaunchKernel(host::div_ceil(num_warps, Shape::kWarpsPerCTA), Shape::kCTASize, device)(
          gather_mxfp8_kernel<kWorldSize, Shape>, params);  //
    } else {
      using Shape = GatherPrefill;
      const auto num_warps = m_pad * Shape::kWarpsPerToken;
      host::LaunchKernel(host::div_ceil(num_warps, Shape::kWarpsPerCTA), Shape::kCTASize, device)(
          gather_mxfp8_kernel<kWorldSize, Shape>, params);  //
    }
  }

  /// The gather dequantized to bf16 ([M, 6144]), for a wkv without an MXFP8 view.
  static void gather_bf16(
      const tvm::ffi::TensorView ids,
      const tvm::ffi::TensorView out,
      const std::vector<int64_t> w_ptrs,
      const std::vector<int64_t> s_ptrs,
      const std::vector<int64_t> shard_starts) {
    using namespace host;
    using namespace engram_fusion;

    auto M = SymbolicSize{"num_tokens"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({M, static_cast<int64_t>(kCols)})
        .with_strides({-1, 1})
        .with_dtype<int32_t>()
        .with_device(dev)
        .verify(ids);
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(dev).verify(out);
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    CHECK_HOST(w_ptrs.size() == kWorldSize && s_ptrs.size() == kWorldSize && shard_starts.size() == kWorldSize);
    CHECK_HOST(shard_starts[0] == 0);
    for (uint32_t r = 0; r < kWorldSize; ++r) {
      CHECK_HOST(w_ptrs[r] != 0 && w_ptrs[r] % 32 == 0 && s_ptrs[r] && s_ptrs[r] % 8 == 0)
          << "peer " << r << " " << std::hex << w_ptrs[r] << " " << s_ptrs[r] << std::dec;
    }
    if (num_tokens == 0) return;
    const auto params = [&] {
      GatherBf16Params<kWorldSize> p{
          .ids = static_cast<const int32_t*>(ids.data_ptr()),
          .out = static_cast<bf16_t*>(out.data_ptr()),
          .ids_stride = ids.stride(0),
          .num_tokens = num_tokens,
      };
      for (uint32_t r = 0; r < kWorldSize; ++r) {
        p.w[r] = reinterpret_cast<const uint8_t*>(w_ptrs[r]) - shard_starts[r] * kHeadDim;
        p.s[r] = reinterpret_cast<const uint8_t*>(s_ptrs[r]) - shard_starts[r] * kScalesPerRow;
      }
      return p;
    }();
    const auto device = dev.unwrap();
    // One shape for every M: on H200 the 4-row warp never loses and wins
    // 7-14% at 1K-16K rows over the mxfp8 gather's prefill shape.
    using Shape = GatherDecode;
    const auto num_warps = num_tokens * Shape::kWarpsPerToken;
    host::LaunchKernel(host::div_ceil(num_warps, Shape::kWarpsPerCTA), Shape::kCTASize, device)(
        gather_bf16_kernel<kWorldSize, Shape>, params);
  }

  /// The optional per-row skip mask's device pointer, shape-checked against M.
  static const uint8_t*
  skip_ptr(const tvm::ffi::Optional<tvm::ffi::TensorView>& skip, int64_t num_rows, host::SymbolicDevice& dev) {
    using namespace host;
    if (!skip.has_value()) return nullptr;
    TensorMatcher({num_rows}).with_dtype<uint8_t>().with_device(dev).verify(skip.value());
    return static_cast<const uint8_t*>(skip.value().data_ptr());
  }

  /// The whole engram seam in one launch. Every rank must
  /// launch it; the grid is derived from the world size on every rank alike.
  static void sp_gate_mhc_combine_norm(
      const host::distributed::CommunicatorRef communicator,  // barriered on; its buffers are untouched
      const std::vector<int64_t> x_ptrs,                      // [kWorld] every rank's symmetric next input
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView kv,
      const tvm::ffi::TensorView qw,
      const tvm::ffi::TensorView kw,
      const tvm::ffi::TensorView pre,
      const tvm::ffi::TensorView weight,                    // the next norm's weight
      const tvm::ffi::Optional<tvm::ffi::TensorView> skip,  // [M] uint8, nonzero keeps the row
      const int64_t row_offset,
      const int64_t total_rows,
      const float eps,
      const float clamp,
      const float rms_eps) {
    using namespace host;
    using namespace engram_fusion;

    constexpr int64_t H = kHC;
    auto M = SymbolicSize{"num_rows"};
    auto KM = SymbolicSize{"kv_rows"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({M, H, D}).with_dtype<bf16_t>().with_device(dev).verify(residual);
    TensorMatcher({KM, static_cast<int64_t>(KV)}).with_dtype<bf16_t>().with_device(dev).verify(kv);
    TensorMatcher({H, D}).with_dtype<bf16_t>().with_device(dev).verify(qw).verify(kw);
    TensorMatcher({M, H}).with_dtype<float>().with_device(dev).verify(pre);
    TensorMatcher({D}).with_dtype<bf16_t>().with_device(dev).verify(weight);

    const auto& pull = communicator.get()->get_pull_obj();
    const auto num_rows = static_cast<uint32_t>(M.unwrap());
    const auto device = dev.unwrap();
    // World 2 scales with the grid up to the full machine; past that the seam is
    // ingress-bound and 64 blocks measured the same as 128 (B300, M 1K-8K).
    const uint32_t want = kWorldSize == 2 ? runtime::get_sm_count(device.device_id) : 64u;
    const auto grid = std::min(want, pull.num_blocks);
    const auto split = device::mhc::row_split(static_cast<uint32_t>(total_rows), pull.rank, kWorldSize);

    CHECK_HOST(pull.world_size == kWorldSize);
    CHECK_HOST(KM.unwrap() >= M.unwrap());
    CHECK_HOST(x_ptrs.size() == kWorldSize);
    // A null peer pointer faults deep inside the kernel; say so here instead.
    for (uint32_t r = 0; r < kWorldSize; ++r) {
      CHECK_HOST(x_ptrs[r] && x_ptrs[r] % 16 == 0) << "peer " << r << " has a null or misaligned symmetric buffer";
    }
    CHECK_HOST(static_cast<uint32_t>(row_offset) == split.offset && num_rows == split.count)
        << "the caller's row split disagrees: rank " << pull.rank << " passed [" << row_offset << ", +" << num_rows
        << ") of " << total_rows << ", this kernel owns [" << split.offset << ", +" << split.count << ")";

    const auto params = [&] {
      GateCombineParams<kWorldSize> p{
          .residual = static_cast<bf16_t*>(residual.data_ptr()),
          .kv = static_cast<const __nv_bfloat16*>(kv.data_ptr()),
          .qw = static_cast<const __nv_bfloat16*>(qw.data_ptr()),
          .kw = static_cast<const __nv_bfloat16*>(kw.data_ptr()),
          .pre = static_cast<const float*>(pre.data_ptr()),
          .norm_weight = static_cast<const bf16_t*>(weight.data_ptr()),
          .skip = skip_ptr(skip, M.unwrap(), dev),
          .rank = pull.rank,
          .num_rows = num_rows,
          .row_offset = static_cast<uint32_t>(row_offset),
          .eps = eps,
          .clamp = clamp,
          .rms_eps = rms_eps,
      };
      for (uint32_t r = 0; r < kWorldSize; ++r) {
        p.x_peer[r] = reinterpret_cast<bf16_t*>(x_ptrs[r]);
        p.semaphores[r] = pull.semaphores[r];
      }
      return p;
    }();
    // The grid launches in full: every rank must make every barrier arrival.
    // Strided rows wake more SMs at small M and tie at large M, retiring the
    // 2-row variant (engram-sp-order; accepted regret 1.4-2.5us W4 mid-M).
    constexpr auto kernel = sp_gate_mhc_combine_norm_kernel<kWorldSize, kSeamRowsPerCTA>;
    constexpr size_t kSmemBytes = sizeof(GateCombineSmem<kSeamRowsPerCTA>);
    CHECK_CUDA(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    LaunchKernel(grid, kGroupThreads * kSeamRowsPerCTA, device, kSmemBytes).launch(kernel, params);
  }

  /// The DP seam: the SP math on this rank's rows only, purely local -- no
  /// symmetric buffers, no barriers, x lands in ``x_out``.
  static void dp_gate_mhc_combine_norm(
      const tvm::ffi::TensorView residual,  // [M, 4, 5120] updated in place
      const tvm::ffi::TensorView kv,
      const tvm::ffi::TensorView qw,
      const tvm::ffi::TensorView kw,
      const tvm::ffi::TensorView pre,                       // [M, 4], the previous boundary's lagged pre
      const tvm::ffi::TensorView weight,                    // the next norm's weight
      const tvm::ffi::Optional<tvm::ffi::TensorView> skip,  // [M] uint8, nonzero keeps the row
      const tvm::ffi::TensorView x_out,                     // [M, 5120]
      const float eps,
      const float clamp,
      const float rms_eps) {
    using namespace host;
    using namespace engram_fusion;

    constexpr int64_t H = kHC;
    auto M = SymbolicSize{"num_rows"};
    auto KM = SymbolicSize{"kv_rows"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({M, H, D}).with_dtype<bf16_t>().with_device(dev).verify(residual);
    TensorMatcher({KM, static_cast<int64_t>(KV)}).with_dtype<bf16_t>().with_device(dev).verify(kv);
    TensorMatcher({H, D}).with_dtype<bf16_t>().with_device(dev).verify(qw).verify(kw);
    TensorMatcher({M, H}).with_dtype<float>().with_device(dev).verify(pre);
    TensorMatcher({D}).with_dtype<bf16_t>().with_device(dev).verify(weight);
    TensorMatcher({M, D}).with_dtype<bf16_t>().with_device(dev).verify(x_out);
    CHECK_HOST(KM.unwrap() >= M.unwrap());

    const auto num_rows = static_cast<uint32_t>(M.unwrap());
    if (num_rows == 0) return;  // nothing barriers, so an empty rank just skips
    const auto params = DpGateCombineParams{
        .residual = static_cast<bf16_t*>(residual.data_ptr()),
        .kv = static_cast<const __nv_bfloat16*>(kv.data_ptr()),
        .qw = static_cast<const __nv_bfloat16*>(qw.data_ptr()),
        .kw = static_cast<const __nv_bfloat16*>(kw.data_ptr()),
        .pre = static_cast<const float*>(pre.data_ptr()),
        .norm_weight = static_cast<const bf16_t*>(weight.data_ptr()),
        .skip = skip_ptr(skip, M.unwrap(), dev),
        .x_out = static_cast<bf16_t*>(x_out.data_ptr()),
        .num_rows = num_rows,
        .eps = eps,
        .clamp = clamp,
        .rms_eps = rms_eps,
    };
    // Grid: min(num_sm, M) -- past one CTA per SM the free grid never helps
    // (2 CTAs/SM costs 2-5%), and trimming the idle CTAs at M < num_sm saves
    // another ~1 us (B300 study, engram-dp-order REPORT.md).
    const auto device = dev.unwrap();
    const auto grid = std::min(runtime::get_sm_count(device.device_id), num_rows);
    constexpr auto kernel = dp_gate_mhc_combine_norm_kernel<kSeamRowsPerCTA>;
    constexpr size_t kSmemBytes = sizeof(GateCombineSmem<kSeamRowsPerCTA>);
    CHECK_CUDA(::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    LaunchKernel(grid, kGroupThreads * kSeamRowsPerCTA, device, kSmemBytes).launch(kernel, params);
  }

  static void tp_gate_mhc_combine_norm(
      const host::distributed::CommunicatorRef communicator,  // barriered on; its buffers are untouched
      const std::vector<int64_t> staging_ptrs,                // [kWorld] every rank's symmetric staging plane
      const tvm::ffi::TensorView staging,                     // this rank's plane, tp_staging_bytes(T) uint8
      const tvm::ffi::TensorView residual,                    // [T, 4, 5120] replicated, updated in place
      const tvm::ffi::TensorView kv,                          // [M, 25600] this rank's share
      const tvm::ffi::TensorView qw,
      const tvm::ffi::TensorView kw,
      const tvm::ffi::TensorView pre,                       // [T, 4] fp32, the previous boundary's lagged pre
      const tvm::ffi::TensorView weight,                    // the next norm's weight
      const tvm::ffi::Optional<tvm::ffi::TensorView> skip,  // [T] uint8, nonzero keeps the row
      const tvm::ffi::TensorView x_out,                     // [T, 5120] bf16
      const float eps,
      const float clamp,
      const float rms_eps) {
    using namespace host;
    using namespace engram_fusion;

    constexpr int64_t H = kHC;
    auto T = SymbolicSize{"total_rows"};
    auto KM = SymbolicSize{"kv_rows"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({T, H, D}).with_dtype<bf16_t>().with_device(dev).verify(residual);
    TensorMatcher({KM, static_cast<int64_t>(KV)}).with_dtype<bf16_t>().with_device(dev).verify(kv);
    TensorMatcher({H, D}).with_dtype<bf16_t>().with_device(dev).verify(qw).verify(kw);
    TensorMatcher({T, H}).with_dtype<float>().with_device(dev).verify(pre);
    TensorMatcher({D}).with_dtype<bf16_t>().with_device(dev).verify(weight);
    TensorMatcher({T, D}).with_dtype<bf16_t>().with_device(dev).verify(x_out);
    const auto total_rows = static_cast<uint32_t>(T.unwrap());
    TensorMatcher({static_cast<int64_t>(total_rows) * (D * 2 + 16)})
        .with_dtype<uint8_t>()
        .with_device(dev)
        .ensure_alignment(16)
        .verify(staging);

    constexpr auto get_staging_ptr = [](int64_t base, uint32_t total_rows) {
      const auto values = reinterpret_cast<bf16_t*>(base);
      const auto gates = reinterpret_cast<float*>(values + static_cast<uint64_t>(total_rows) * engram_fusion::D);
      return std::make_pair(values, gates);
    };

    const auto& pull = communicator.get()->get_pull_obj();
    const auto device = dev.unwrap();
    CHECK_HOST(pull.world_size == kWorldSize);
    // The token shares split ragged exactly like the boundary's rows.
    const auto split = device::mhc::row_split(total_rows, pull.rank, kWorldSize);
    const uint32_t num_rows = split.count;
    CHECK_HOST(KM.unwrap() >= num_rows);
    CHECK_HOST(staging_ptrs.size() == kWorldSize);
    for (uint32_t r = 0; r < kWorldSize; ++r) {
      CHECK_HOST(staging_ptrs[r] && staging_ptrs[r] % 16 == 0)
          << "peer " << r << " has a null or misaligned staging plane";
    }
    CHECK_HOST(staging_ptrs[pull.rank] == reinterpret_cast<int64_t>(staging.data_ptr()))
        << "the staging tensor is not this rank's entry of staging_ptrs";

    constexpr uint32_t kRowsPerCTA = kSeamRowsPerCTA;
    const auto num_sm = runtime::get_sm_count(device.device_id);
    {
      const auto max_blocks = kWorldSize == 2 ? num_sm : 64u;  // based on profile
      const auto num_blocks = std::min(max_blocks, pull.num_blocks);
      const auto params = [&] {
        auto p = TpGatePushParams<kWorldSize>{
            .residual = static_cast<const bf16_t*>(residual.data_ptr()),
            .kv = static_cast<const __nv_bfloat16*>(kv.data_ptr()),
            .qw = static_cast<const __nv_bfloat16*>(qw.data_ptr()),
            .kw = static_cast<const __nv_bfloat16*>(kw.data_ptr()),
            .rank = pull.rank,
            .num_rows = num_rows,
            .row_offset = split.offset,
            .eps = eps,
            .clamp = clamp,
        };
        for (uint32_t r = 0; r < kWorldSize; ++r) {
          std::tie(p.values[r], p.gates[r]) = get_staging_ptr(staging_ptrs[r], total_rows);
          p.semaphores[r] = pull.semaphores[r];
        }
        return p;
      }();
      LaunchKernel(num_blocks, engram_fusion::kGroupThreads * kRowsPerCTA, device)(
          engram_fusion::tp_gate_push_kernel<kWorldSize, kRowsPerCTA>, params);
    }
    {
      const auto [values, gates] = get_staging_ptr(staging_ptrs[pull.rank], total_rows);
      const auto params = TpCombineParams{
          .residual = static_cast<bf16_t*>(residual.data_ptr()),
          .values = values,
          .gates = gates,
          .pre = static_cast<const float*>(pre.data_ptr()),
          .norm_weight = static_cast<const bf16_t*>(weight.data_ptr()),
          .skip = skip_ptr(skip, T.unwrap(), dev),
          .x_out = static_cast<bf16_t*>(x_out.data_ptr()),
          .num_rows = total_rows,
          .rms_eps = rms_eps,
      };
      const auto num_blocks = std::min(host::div_ceil(total_rows, kRowsPerCTA), 2 * num_sm);
      LaunchKernel(num_blocks, engram_fusion::kGroupThreads * kRowsPerCTA, device)(
          engram_fusion::tp_update_combine_norm_kernel<kRowsPerCTA>, params);
    }
  }
};

}  // namespace sglang
