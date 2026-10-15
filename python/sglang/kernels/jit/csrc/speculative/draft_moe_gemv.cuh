// Single-token (M = 1) MoE for a bf16 draft layer: routed top-k experts plus
// the shared expert, as two bandwidth-bound GEMV passes.
//
// up pass:   act[s][i] = bf16(silu(g) * u) with g, u = bf16(dot(x, W13[e_s] rows i, I + i));
//            slot kTopK is the shared expert. One extra CTA computes the
//            shared-expert gate sigmoid(dot(x, gate_w)).
// down pass: out[j] = bf16(gate * bf16(dot(act[kTopK], Ws[j]))
//                          + bf16(sum_s bf16(dot(act[s], W2[e_s][j]) * w_s)))
// The rounding points are the ones of the unfused path (up GEMM -> bf16,
// act -> bf16, down GEMM * routing weight -> bf16, fp32 top-k sum in slot
// order -> bf16, gate/shared/add in fp32 -> bf16); only the summation order
// inside each dot product differs.
//
// Inputs written by earlier grids (x, topk ids/weights, act, gate) are read
// through plain pointers after griddepcontrol.wait; only static weights use
// the read-only path, and those may be loaded before the wait.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace draft_moe {

/// \brief Same expression as the SiLU of the act-and-mul kernel.
SGL_DEVICE float silu_f32(float x) {
  return x / (1.0f + expf(-x));
}

SGL_DEVICE float round_bf16(float x) {
  return device::cast<fp32_t>(device::cast<bf16_t>(x));
}

}  // namespace draft_moe

struct DraftMoeUpParams {
  const bf16_t* x;                        // [H], written by an earlier grid
  const bf16_t* __restrict__ w13;         // [E, 2I, H]
  const bf16_t* __restrict__ w13_shared;  // [2I, H]
  const bf16_t* __restrict__ gate_w;      // [H]
  const int32_t* topk_ids;                // [topk], written by an earlier grid
  bf16_t* act;                            // [topk + 1, I]
  float* gate_out;                        // [1]
};

struct DraftMoeDownParams {
  const bf16_t* act;                     // [topk + 1, I], written by the up pass
  const bf16_t* __restrict__ w2;         // [E, H, I]
  const bf16_t* __restrict__ w2_shared;  // [H, I]
  const int32_t* topk_ids;               // [topk]
  const float* topk_weights;             // [topk]
  const float* gate;                     // [1], written by the up pass
  bf16_t* out;                           // [H]
};

/**
 * \brief Up pass: one warp per weight row, two warps (gate, up) per output.
 *
 * \tparam kH     hidden size (reduction length)
 * \tparam kI     per-rank intermediate size of one expert
 * \tparam kTopK  routed experts per token
 * \tparam kPairs act outputs per CTA (2 * kPairs warps)
 */
template <int kH, int kI, int kTopK, int kPairs, bool kUsePDL>
__global__ void __launch_bounds__(kPairs * 64) draft_moe_up_kernel(const DraftMoeUpParams __grid_constant__ p) {
  using namespace device;
  constexpr int kThreads = kPairs * 64;
  constexpr int kVecN = static_cast<int>(kMaxVecBytes / sizeof(bf16_t));
  using vec_t = AlignedVector<bf16_t, kVecN>;
  constexpr int kChunks = kH / kVecN;
  static_assert(kH % kVecN == 0 && kChunks % kWarpThreads == 0, "hidden size must tile the warp");
  constexpr int kPerLane = kChunks / kWarpThreads;

  __shared__ alignas(32) bf16_t s_x[kH];
  __shared__ float s_dot[kThreads / kWarpThreads];

  const int warp = static_cast<int>(threadIdx.x / kWarpThreads);
  const int lane = static_cast<int>(get_lane_id());

  if (blockIdx.x == gridDim.x - 1) {
    // Shared-expert gate.
    PDLWaitPrimary<kUsePDL>();
    float acc = 0.0f;
    for (int c = static_cast<int>(threadIdx.x); c < kChunks; c += kThreads) {
      vec_t xv, wv;
      xv.load(p.x, c);
      wv.load(p.gate_w, c);
#pragma unroll
      for (int k = 0; k < kVecN; ++k) {
        acc = fmaf(cast<fp32_t>(xv[k]), cast<fp32_t>(wv[k]), acc);
      }
    }
    acc = warp::reduce_sum(acc);
    if (lane == 0) s_dot[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
      float t = 0.0f;
#pragma unroll
      for (int w = 0; w < kThreads / kWarpThreads; ++w) t += s_dot[w];
      p.gate_out[0] = 1.0f / (1.0f + expf(-t));
    }
    PDLTriggerSecondary<kUsePDL>();
    return;
  }

  const int pair = static_cast<int>(blockIdx.x) * kPairs + warp / 2;
  const int slot = pair / kI;
  const int i = pair % kI;
  const int64_t row_in_expert = ((warp & 1) ? kI : 0) + i;

  vec_t w[kPerLane];
  if (slot == kTopK) {
    const bf16_t* row = p.w13_shared + row_in_expert * kH;
#pragma unroll
    for (int k = 0; k < kPerLane; ++k) w[k].load(row, k * kWarpThreads + lane);
  }

  PDLWaitPrimary<kUsePDL>();

  if (slot < kTopK) {
    const int64_t e = p.topk_ids[slot];
    const bf16_t* row = p.w13 + (e * 2 * kI + row_in_expert) * kH;
#pragma unroll
    for (int k = 0; k < kPerLane; ++k) w[k].load(row, k * kWarpThreads + lane);
  }
  for (int c = static_cast<int>(threadIdx.x); c < kChunks; c += kThreads) {
    vec_t v;
    v.load(p.x, c);
    v.store(s_x, c);
  }
  __syncthreads();

  float acc = 0.0f;
#pragma unroll
  for (int k = 0; k < kPerLane; ++k) {
    vec_t xv;
    xv.load(s_x, k * kWarpThreads + lane);
#pragma unroll
    for (int j = 0; j < kVecN; ++j) {
      acc = fmaf(cast<fp32_t>(w[k][j]), cast<fp32_t>(xv[j]), acc);
    }
  }
  acc = warp::reduce_sum(acc);
  if (lane == 0) s_dot[warp] = acc;
  __syncthreads();
  PDLTriggerSecondary<kUsePDL>();
  if (threadIdx.x < kPairs) {
    const float g = draft_moe::round_bf16(s_dot[2 * threadIdx.x]);
    const float u = draft_moe::round_bf16(s_dot[2 * threadIdx.x + 1]);
    p.act[static_cast<int64_t>(blockIdx.x) * kPairs + threadIdx.x] = cast<bf16_t>(draft_moe::silu_f32(g) * u);
  }
}

/**
 * \brief Down pass: one warp per output column; all kTopK + 1 slots per warp.
 */
template <int kH, int kI, int kTopK, int kWarps, bool kUsePDL>
__global__ void __launch_bounds__(kWarps * 32) draft_moe_down_kernel(const DraftMoeDownParams __grid_constant__ p) {
  using namespace device;
  constexpr int kSlots = kTopK + 1;
  constexpr int kLaneElems = kI / static_cast<int>(kWarpThreads);
  static_assert(kI % kWarpThreads == 0, "intermediate size must tile the warp");
  using rvec_t = AlignedVector<bf16_t, kLaneElems>;
  constexpr int kCopyN = 8;  // 16-byte staging copies
  using cvec_t = AlignedVector<bf16_t, kCopyN>;
  static_assert((kSlots * kI) % kCopyN == 0);

  __shared__ alignas(32) bf16_t s_act[kSlots * kI];
  __shared__ int32_t s_eid[kTopK];
  __shared__ float s_w[kTopK];
  __shared__ float s_gate;

  const int warp = static_cast<int>(threadIdx.x / kWarpThreads);
  const int lane = static_cast<int>(get_lane_id());
  const int64_t j = static_cast<int64_t>(blockIdx.x) * kWarps + warp;

  rvec_t ws;
  ws.load(p.w2_shared + j * kI, lane);

  PDLWaitPrimary<kUsePDL>();

  for (int c = static_cast<int>(threadIdx.x); c < kSlots * kI / kCopyN; c += kWarps * 32) {
    cvec_t v;
    v.load(p.act, c);
    v.store(s_act, c);
  }
  if (threadIdx.x < kTopK) {
    s_eid[threadIdx.x] = p.topk_ids[threadIdx.x];
    s_w[threadIdx.x] = p.topk_weights[threadIdx.x];
  }
  if (threadIdx.x == 0) s_gate = p.gate[0];
  __syncthreads();

  rvec_t wv[kTopK];
#pragma unroll
  for (int s = 0; s < kTopK; ++s) {
    wv[s].load(p.w2 + (static_cast<int64_t>(s_eid[s]) * kH + j) * kI, lane);
  }

  float d[kSlots];
#pragma unroll
  for (int s = 0; s < kTopK; ++s) {
    rvec_t a;
    a.load(s_act + s * kI, lane);
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < kLaneElems; ++k) acc = fmaf(cast<fp32_t>(wv[s][k]), cast<fp32_t>(a[k]), acc);
    d[s] = acc;
  }
  {
    rvec_t a;
    a.load(s_act + kTopK * kI, lane);
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < kLaneElems; ++k) acc = fmaf(cast<fp32_t>(ws[k]), cast<fp32_t>(a[k]), acc);
    d[kTopK] = acc;
  }
#pragma unroll
  for (int s = 0; s < kSlots; ++s) d[s] = warp::reduce_sum(d[s]);

  PDLTriggerSecondary<kUsePDL>();
  if (lane == 0) {
    float sum = 0.0f;
#pragma unroll
    for (int s = 0; s < kTopK; ++s) sum += draft_moe::round_bf16(d[s] * s_w[s]);
    const float routed = draft_moe::round_bf16(sum);
    const float shared = draft_moe::round_bf16(d[kTopK]);
    float r = s_gate * shared;
    r += routed;
    p.out[j] = cast<bf16_t>(r);
  }
}

namespace draft_moe {

// global -> shared bulk copy completed on an smem mbarrier (PTX ISA cp.async.bulk);
// size and both addresses must be 16-byte aligned.
SGL_DEVICE void bulk_g2s(void* smem_dst, const void* gmem_src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          device::ptx::to_shared(smem_dst)),
      "l"(gmem_src),
      "r"(bytes),
      "r"(device::ptx::to_shared(bar))
      : "memory");
}

SGL_DEVICE void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

}  // namespace draft_moe

/**
 * \brief Up pass, persistent: one CTA per SM, a producer warp streams (gate, up)
 * row pairs into a shared-memory ring with bulk copies; 8 consumer warps each
 * reduce one quarter of the gate or up row against x held in registers.
 *
 * Pair q < kI is shared-expert row i = q (static, issued before the PDL wait);
 * pair q >= kI is routed slot (q - kI) / kI, row (q - kI) % kI.
 */
template <int kH, int kI, int kTopK, int kStages, bool kContig, bool kUsePDL>
__global__ void __launch_bounds__(288, 1) draft_moe_up_tma_kernel(const DraftMoeUpParams __grid_constant__ p) {
  using namespace device;
  constexpr int kRowBytes = kH * static_cast<int>(sizeof(bf16_t));
  constexpr int kStageBytes = 2 * kRowBytes;
  constexpr int kPairsTotal = (kTopK + 1) * kI;
  constexpr int kConsumers = 8;
  constexpr int kQuarter = kH / 4;
  using cvec_t = AlignedVector<bf16_t, 8>;
  static_assert(kQuarter % (8 * kWarpThreads) == 0, "hidden size must tile 4 warps x 16 bytes");
  constexpr int kChunksPerLane = kQuarter / (8 * kWarpThreads);
  constexpr int kMaxPairsPerCta = 64;

  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ uint64_t full_bar[kStages];
  __shared__ uint64_t empty_bar[kStages];
  __shared__ float s_part[kMaxPairsPerCta][kConsumers];
  __shared__ float s_gate_part[4];

  const int warp = static_cast<int>(threadIdx.x / kWarpThreads);
  const int lane = static_cast<int>(get_lane_id());
  const int G = static_cast<int>(gridDim.x);
  const int c = static_cast<int>(blockIdx.x);
  // kContig: CTA c owns a contiguous pair range; otherwise pairs c, c + G, ...
  const int base = kPairsTotal / G;
  const int rem = kPairsTotal % G;
  const int q0 = kContig ? c * base + min(c, rem) : c;
  const int qstep = kContig ? 1 : G;
  const int npairs = kContig ? base + (c < rem ? 1 : 0) : (kPairsTotal - c + G - 1) / G;

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      ptx::mbar_init(&full_bar[s], 1);
      ptx::mbar_init(&empty_bar[s], kConsumers);
    }
    draft_moe::fence_mbarrier_init();
  }
  __syncthreads();

  if (warp == kConsumers) {
    if (lane == 0) {
      bool waited = false;
      int32_t ids[kTopK];
      int stage = 0;
      uint32_t phase = 0;
      for (int t = 0; t < npairs; ++t) {
        const int q = q0 + t * qstep;
        const bf16_t* src;
        if (q < kI) {
          src = p.w13_shared + static_cast<int64_t>(q) * kH;
        } else {
          if (!waited) {
            PDLWaitPrimary<kUsePDL>();
#pragma unroll
            for (int k = 0; k < kTopK; ++k) ids[k] = p.topk_ids[k];
            waited = true;
          }
          const int s = (q - kI) / kI;
          const int i = (q - kI) % kI;
          src = p.w13 + (static_cast<int64_t>(ids[s]) * 2 * kI + i) * kH;
        }
        ptx::mbar_wait_parity(&empty_bar[stage], phase ^ 1);
        ptx::mbar_arrive_expect_tx(&full_bar[stage], kStageBytes);
        draft_moe::bulk_g2s(smem + stage * kStageBytes, src, kRowBytes, &full_bar[stage]);
        draft_moe::bulk_g2s(
            smem + stage * kStageBytes + kRowBytes, src + static_cast<int64_t>(kI) * kH, kRowBytes, &full_bar[stage]);
        if (++stage == kStages) {
          stage = 0;
          phase ^= 1;
        }
      }
      if (!waited) PDLWaitPrimary<kUsePDL>();
    }
  } else {
    PDLWaitPrimary<kUsePDL>();
    const int quarter = warp & 3;
    const bool is_up = warp >= 4;
    cvec_t xr[kChunksPerLane];
#pragma unroll
    for (int k = 0; k < kChunksPerLane; ++k) xr[k].load(p.x + quarter * kQuarter, k * kWarpThreads + lane);
    int stage = 0;
    uint32_t phase = 0;
    for (int t = 0; t < npairs; ++t) {
      ptx::mbar_wait_parity(&full_bar[stage], phase);
      const auto* row =
          reinterpret_cast<const bf16_t*>(smem + stage * kStageBytes + (is_up ? kRowBytes : 0)) + quarter * kQuarter;
      float acc = 0.0f;
#pragma unroll
      for (int k = 0; k < kChunksPerLane; ++k) {
        cvec_t w;
        w.load(row, k * kWarpThreads + lane);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc = fmaf(cast<fp32_t>(w[j]), cast<fp32_t>(xr[k][j]), acc);
      }
      acc = warp::reduce_sum(acc);
      if (lane == 0) {
        s_part[t][warp] = acc;
        ptx::mbar_arrive(&empty_bar[stage]);
      }
      if (++stage == kStages) {
        stage = 0;
        phase ^= 1;
      }
    }
    if (c == 0 && !is_up) {
      float acc = 0.0f;
#pragma unroll
      for (int k = 0; k < kChunksPerLane; ++k) {
        cvec_t gw;
        gw.load(p.gate_w + quarter * kQuarter, k * kWarpThreads + lane);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc = fmaf(cast<fp32_t>(xr[k][j]), cast<fp32_t>(gw[j]), acc);
      }
      acc = warp::reduce_sum(acc);
      if (lane == 0) s_gate_part[quarter] = acc;
    }
  }
  __syncthreads();
  PDLTriggerSecondary<kUsePDL>();
  if (static_cast<int>(threadIdx.x) < npairs) {
    const int t = static_cast<int>(threadIdx.x);
    const int q = q0 + t * qstep;
    const float g = draft_moe::round_bf16(((s_part[t][0] + s_part[t][1]) + s_part[t][2]) + s_part[t][3]);
    const float u = draft_moe::round_bf16(((s_part[t][4] + s_part[t][5]) + s_part[t][6]) + s_part[t][7]);
    const int64_t dst = q < kI ? static_cast<int64_t>(kTopK) * kI + q : static_cast<int64_t>(q - kI);
    p.act[dst] = cast<bf16_t>(draft_moe::silu_f32(g) * u);
  }
  if (c == 0 && threadIdx.x == 0) {
    const float t = ((s_gate_part[0] + s_gate_part[1]) + s_gate_part[2]) + s_gate_part[3];
    p.gate_out[0] = 1.0f / (1.0f + expf(-t));
  }
}

/**
 * \brief Down pass, one shot per CTA: CTA b owns output columns [b * kCols, ...);
 * all kTopK + 1 weight slices (kCols contiguous rows each) land in shared memory
 * through bulk copies, then each warp reduces whole columns.
 */
template <int kH, int kI, int kTopK, int kCols, bool kUsePDL>
__global__ void __launch_bounds__(256, 1) draft_moe_down_tma_kernel(const DraftMoeDownParams __grid_constant__ p) {
  using namespace device;
  constexpr int kSlots = kTopK + 1;
  constexpr int kWarps = 8;
  constexpr int kLaneElems = kI / static_cast<int>(kWarpThreads);
  static_assert(kI % kWarpThreads == 0, "intermediate size must tile the warp");
  using rvec_t = AlignedVector<bf16_t, kLaneElems>;
  constexpr int kSliceBytes = kCols * kI * static_cast<int>(sizeof(bf16_t));
  constexpr int kCopyN = 8;
  using cvec_t = AlignedVector<bf16_t, kCopyN>;

  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ alignas(32) bf16_t s_act[kSlots * kI];
  __shared__ uint64_t bar;
  __shared__ float s_w[kTopK];
  __shared__ float s_gate;

  const int warp = static_cast<int>(threadIdx.x / kWarpThreads);
  const int lane = static_cast<int>(get_lane_id());
  const int j0 = static_cast<int>(blockIdx.x) * kCols;
  const int ncols = min(kCols, kH - j0);
  const uint32_t bytes = static_cast<uint32_t>(ncols) * kI * sizeof(bf16_t);

  if (threadIdx.x == 0) {
    ptx::mbar_init(&bar, 1);
    draft_moe::fence_mbarrier_init();
    ptx::mbar_arrive_expect_tx(&bar, kSlots * bytes);
    draft_moe::bulk_g2s(smem + kTopK * kSliceBytes, p.w2_shared + static_cast<int64_t>(j0) * kI, bytes, &bar);
  }

  PDLWaitPrimary<kUsePDL>();

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kTopK; ++s) {
      const int64_t e = p.topk_ids[s];
      draft_moe::bulk_g2s(smem + s * kSliceBytes, p.w2 + (e * kH + j0) * kI, bytes, &bar);
    }
  }
  for (int c = static_cast<int>(threadIdx.x); c < kSlots * kI / kCopyN; c += kWarps * kWarpThreads) {
    cvec_t v;
    v.load(p.act, c);
    v.store(s_act, c);
  }
  if (threadIdx.x < kTopK) s_w[threadIdx.x] = p.topk_weights[threadIdx.x];
  if (threadIdx.x == 0) s_gate = p.gate[0];
  __syncthreads();
  PDLTriggerSecondary<kUsePDL>();

  rvec_t a[kSlots];
#pragma unroll
  for (int s = 0; s < kSlots; ++s) a[s].load(s_act + s * kI, lane);
  ptx::mbar_wait_parity(&bar, 0);

  for (int jj = warp; jj < ncols; jj += kWarps) {
    float d[kSlots];
#pragma unroll
    for (int s = 0; s < kSlots; ++s) {
      rvec_t w;
      w.load(reinterpret_cast<const bf16_t*>(smem + s * kSliceBytes) + jj * kI, lane);
      float acc = 0.0f;
#pragma unroll
      for (int k = 0; k < kLaneElems; ++k) acc = fmaf(cast<fp32_t>(w[k]), cast<fp32_t>(a[s][k]), acc);
      d[s] = warp::reduce_sum(acc);
    }
    if (lane == 0) {
      float sum = 0.0f;
#pragma unroll
      for (int s = 0; s < kTopK; ++s) sum += draft_moe::round_bf16(d[s] * s_w[s]);
      const float routed = draft_moe::round_bf16(sum);
      const float shared = draft_moe::round_bf16(d[kTopK]);
      float r = s_gate * shared;
      r += routed;
      p.out[j0 + jj] = cast<bf16_t>(r);
    }
  }
}

/**
 * \brief Down pass with one mbarrier per weight slice: slices are reduced in
 * landing order (shared slice first). Each 8-lane group owns one output column,
 * so a warp reduces four columns per pass with three shuffles.
 */
template <int kH, int kI, int kTopK, int kCols, bool kUsePDL>
__global__ void __launch_bounds__(256, 1) draft_moe_down_v3_kernel(const DraftMoeDownParams __grid_constant__ p) {
  using namespace device;
  constexpr int kSlots = kTopK + 1;
  constexpr int kWarps = 8;
  constexpr int kGroups = kWarps * 4;
  constexpr int kChunks = kI / 8;
  static_assert(kI % 64 == 0, "intermediate size must tile 8 lanes x 16 bytes");
  constexpr int kPerLane = kChunks / 8;
  constexpr int kRounds = (kCols + kGroups - 1) / kGroups;
  using cvec_t = AlignedVector<bf16_t, 8>;
  constexpr int kSliceBytes = kCols * kI * static_cast<int>(sizeof(bf16_t));

  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ alignas(16) bf16_t s_act[kSlots * kI];
  __shared__ uint64_t bar[kSlots];
  __shared__ float s_d[kSlots][kCols];
  __shared__ float s_w[kTopK];
  __shared__ float s_gate;

  const int warp = static_cast<int>(threadIdx.x / kWarpThreads);
  const int lane = static_cast<int>(get_lane_id());
  const int grp = warp * 4 + lane / 8;
  const int gl = lane % 8;
  const int j0 = static_cast<int>(blockIdx.x) * kCols;
  const int ncols = min(kCols, kH - j0);
  const uint32_t bytes = static_cast<uint32_t>(ncols) * kI * sizeof(bf16_t);

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kSlots; ++s) ptx::mbar_init(&bar[s], 1);
    draft_moe::fence_mbarrier_init();
    ptx::mbar_arrive_expect_tx(&bar[kTopK], bytes);
    draft_moe::bulk_g2s(smem + kTopK * kSliceBytes, p.w2_shared + static_cast<int64_t>(j0) * kI, bytes, &bar[kTopK]);
  }

  PDLWaitPrimary<kUsePDL>();

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kTopK; ++s) {
      const int64_t e = p.topk_ids[s];
      ptx::mbar_arrive_expect_tx(&bar[s], bytes);
      draft_moe::bulk_g2s(smem + s * kSliceBytes, p.w2 + (e * kH + j0) * kI, bytes, &bar[s]);
    }
  }
  for (int c = static_cast<int>(threadIdx.x); c < kSlots * kChunks; c += kWarps * kWarpThreads) {
    cvec_t v;
    v.load(p.act, c);
    v.store(s_act, c);
  }
  if (threadIdx.x < kTopK) s_w[threadIdx.x] = p.topk_weights[threadIdx.x];
  if (threadIdx.x == 0) s_gate = p.gate[0];
  __syncthreads();
  PDLTriggerSecondary<kUsePDL>();

  for (int it = 0; it < kSlots; ++it) {
    const int s = it == 0 ? kTopK : it - 1;
    ptx::mbar_wait_parity(&bar[s], 0);
    const auto* slice = reinterpret_cast<const bf16_t*>(smem + s * kSliceBytes);
#pragma unroll
    for (int r = 0; r < kRounds; ++r) {
      const int jj = r * kGroups + grp;
      const int jr = min(jj, ncols - 1);
      float acc = 0.0f;
#pragma unroll
      for (int k = 0; k < kPerLane; ++k) {
        cvec_t w, a;
        w.load(slice + jr * kI, gl + 8 * k);
        a.load(s_act + s * kI, gl + 8 * k);
#pragma unroll
        for (int e = 0; e < 8; ++e) acc = fmaf(cast<fp32_t>(w[e]), cast<fp32_t>(a[e]), acc);
      }
      acc += __shfl_xor_sync(0xffffffffu, acc, 4);
      acc += __shfl_xor_sync(0xffffffffu, acc, 2);
      acc += __shfl_xor_sync(0xffffffffu, acc, 1);
      if (gl == 0 && jj < ncols) s_d[s][jj] = acc;
    }
  }
  __syncthreads();
  if (static_cast<int>(threadIdx.x) < ncols) {
    const int jj = static_cast<int>(threadIdx.x);
    float sum = 0.0f;
#pragma unroll
    for (int s = 0; s < kTopK; ++s) sum += draft_moe::round_bf16(s_d[s][jj] * s_w[s]);
    const float routed = draft_moe::round_bf16(sum);
    const float shared = draft_moe::round_bf16(s_d[kTopK][jj]);
    float r = s_gate * shared;
    r += routed;
    p.out[j0 + jj] = cast<bf16_t>(r);
  }
}

template <int kH, int kI, int kTopK, int kPairs, int kDownWarps, int kStages, int kCols, bool kContig, bool kUsePDL>
struct DraftMoeGemv {
  static_assert(((kTopK + 1) * kI) % kPairs == 0);
  static_assert(kH % kDownWarps == 0);

  static void up(
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w13,
      const tvm::ffi::TensorView w13_shared,
      const tvm::ffi::TensorView gate_w,
      const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView act,
      const tvm::ffi::TensorView gate_out) {
    using namespace host;
    auto E = SymbolicSize{"num_experts"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(32).verify(x);
    TensorMatcher({E, 2 * kI, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(32).verify(w13);
    TensorMatcher({2 * kI, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(32).verify(w13_shared);
    TensorMatcher({kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(32).verify(gate_w);
    TensorMatcher({1, kTopK}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({kTopK + 1, kI}).with_dtype<bf16_t>().with_device(device).verify(act);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(gate_out);
    const auto params = DraftMoeUpParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .w13 = static_cast<const bf16_t*>(w13.data_ptr()),
        .w13_shared = static_cast<const bf16_t*>(w13_shared.data_ptr()),
        .gate_w = static_cast<const bf16_t*>(gate_w.data_ptr()),
        .topk_ids = static_cast<const int32_t*>(topk_ids.data_ptr()),
        .act = static_cast<bf16_t*>(act.data_ptr()),
        .gate_out = static_cast<float*>(gate_out.data_ptr()),
    };
    const uint32_t grid = (kTopK + 1) * kI / kPairs + 1;
    LaunchKernel(grid, kPairs * 64, device.unwrap())
        .enable_pdl(kUsePDL)(draft_moe_up_kernel<kH, kI, kTopK, kPairs, kUsePDL>, params);
  }

  static void down(
      const tvm::ffi::TensorView act,
      const tvm::ffi::TensorView w2,
      const tvm::ffi::TensorView w2_shared,
      const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView gate,
      const tvm::ffi::TensorView out) {
    using namespace host;
    auto E = SymbolicSize{"num_experts"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({kTopK + 1, kI}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(act);
    TensorMatcher({E, kH, kI}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(w2);
    TensorMatcher({kH, kI}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(w2_shared);
    TensorMatcher({1, kTopK}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({1, kTopK}).with_dtype<fp32_t>().with_device(device).verify(topk_weights);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(gate);
    TensorMatcher({1, kH}).with_dtype<bf16_t>().with_device(device).verify(out);
    const auto params = DraftMoeDownParams{
        .act = static_cast<const bf16_t*>(act.data_ptr()),
        .w2 = static_cast<const bf16_t*>(w2.data_ptr()),
        .w2_shared = static_cast<const bf16_t*>(w2_shared.data_ptr()),
        .topk_ids = static_cast<const int32_t*>(topk_ids.data_ptr()),
        .topk_weights = static_cast<const float*>(topk_weights.data_ptr()),
        .gate = static_cast<const float*>(gate.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
    };
    LaunchKernel(kH / kDownWarps, kDownWarps * 32, device.unwrap())
        .enable_pdl(kUsePDL)(draft_moe_down_kernel<kH, kI, kTopK, kDownWarps, kUsePDL>, params);
  }

  static void up_tma(
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w13,
      const tvm::ffi::TensorView w13_shared,
      const tvm::ffi::TensorView gate_w,
      const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView act,
      const tvm::ffi::TensorView gate_out) {
    using namespace host;
    auto E = SymbolicSize{"num_experts"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(x);
    TensorMatcher({E, 2 * kI, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(w13);
    TensorMatcher({2 * kI, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(w13_shared);
    TensorMatcher({kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(gate_w);
    TensorMatcher({1, kTopK}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({kTopK + 1, kI}).with_dtype<bf16_t>().with_device(device).verify(act);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(gate_out);
    const auto params = DraftMoeUpParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .w13 = static_cast<const bf16_t*>(w13.data_ptr()),
        .w13_shared = static_cast<const bf16_t*>(w13_shared.data_ptr()),
        .gate_w = static_cast<const bf16_t*>(gate_w.data_ptr()),
        .topk_ids = static_cast<const int32_t*>(topk_ids.data_ptr()),
        .act = static_cast<bf16_t*>(act.data_ptr()),
        .gate_out = static_cast<float*>(gate_out.data_ptr()),
    };
    const DLDevice dev = device.unwrap();
    const uint32_t grid = runtime::get_sm_count(dev.device_id);
    CHECK_HOST(((kTopK + 1) * kI + grid - 1) / grid <= 64) << "too many row pairs per CTA";
    constexpr std::size_t kSmem = static_cast<std::size_t>(kStages) * 2 * kH * sizeof(bf16_t);
    auto kernel = draft_moe_up_tma_kernel<kH, kI, kTopK, kStages, kContig, kUsePDL>;
    static const bool attr_set = [&] {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
      return true;
    }();
    (void)attr_set;
    LaunchKernel(grid, 288, dev, kSmem).enable_pdl(kUsePDL)(kernel, params);
  }

  static void down_tma(
      const tvm::ffi::TensorView act,
      const tvm::ffi::TensorView w2,
      const tvm::ffi::TensorView w2_shared,
      const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView gate,
      const tvm::ffi::TensorView out) {
    using namespace host;
    auto E = SymbolicSize{"num_experts"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({kTopK + 1, kI}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(act);
    TensorMatcher({E, kH, kI}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(w2);
    TensorMatcher({kH, kI}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(w2_shared);
    TensorMatcher({1, kTopK}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({1, kTopK}).with_dtype<fp32_t>().with_device(device).verify(topk_weights);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(gate);
    TensorMatcher({1, kH}).with_dtype<bf16_t>().with_device(device).verify(out);
    const auto params = DraftMoeDownParams{
        .act = static_cast<const bf16_t*>(act.data_ptr()),
        .w2 = static_cast<const bf16_t*>(w2.data_ptr()),
        .w2_shared = static_cast<const bf16_t*>(w2_shared.data_ptr()),
        .topk_ids = static_cast<const int32_t*>(topk_ids.data_ptr()),
        .topk_weights = static_cast<const float*>(topk_weights.data_ptr()),
        .gate = static_cast<const float*>(gate.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
    };
    constexpr std::size_t kSmem = static_cast<std::size_t>(kTopK + 1) * kCols * kI * sizeof(bf16_t);
    auto kernel = draft_moe_down_tma_kernel<kH, kI, kTopK, kCols, kUsePDL>;
    static const bool attr_set = [&] {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
      return true;
    }();
    (void)attr_set;
    LaunchKernel((kH + kCols - 1) / kCols, 256, device.unwrap(), kSmem).enable_pdl(kUsePDL)(kernel, params);
  }

  static void down_v3(
      const tvm::ffi::TensorView act,
      const tvm::ffi::TensorView w2,
      const tvm::ffi::TensorView w2_shared,
      const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView gate,
      const tvm::ffi::TensorView out) {
    using namespace host;
    auto E = SymbolicSize{"num_experts"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({kTopK + 1, kI}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(act);
    TensorMatcher({E, kH, kI}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(w2);
    TensorMatcher({kH, kI}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(w2_shared);
    TensorMatcher({1, kTopK}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({1, kTopK}).with_dtype<fp32_t>().with_device(device).verify(topk_weights);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(gate);
    TensorMatcher({1, kH}).with_dtype<bf16_t>().with_device(device).verify(out);
    const auto params = DraftMoeDownParams{
        .act = static_cast<const bf16_t*>(act.data_ptr()),
        .w2 = static_cast<const bf16_t*>(w2.data_ptr()),
        .w2_shared = static_cast<const bf16_t*>(w2_shared.data_ptr()),
        .topk_ids = static_cast<const int32_t*>(topk_ids.data_ptr()),
        .topk_weights = static_cast<const float*>(topk_weights.data_ptr()),
        .gate = static_cast<const float*>(gate.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
    };
    constexpr std::size_t kSmem = static_cast<std::size_t>(kTopK + 1) * kCols * kI * sizeof(bf16_t);
    auto kernel = draft_moe_down_v3_kernel<kH, kI, kTopK, kCols, kUsePDL>;
    static const bool attr_set = [&] {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
      return true;
    }();
    (void)attr_set;
    LaunchKernel((kH + kCols - 1) / kCols, 256, device.unwrap(), kSmem).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
