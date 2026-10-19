// Single-launch M = 1 bf16 draft MoE: router logits, softmax top-k, routed and
// shared up pass, act, down pass and output in one grid of one CTA per SM.
//
// Every dot product keeps the lane partition and reduction order of
// draft_router_kernel, draft_moe_up_tma_kernel and draft_moe_down_v3_kernel,
// and every rounding point of that path, so ids, weights and the output are
// bitwise equal to it; only the schedule differs.
//
// Phases hand over through tagged words in global scratch. A word carries its
// value and the call tag (epoch + 1): logits and the shared-expert gate as
// tag << 32 | fp32 bits, act values as (tag & 0xffff) << 16 | bf16 bits. Every
// call rewrites every word, so a reader polls until each word it needs carries
// this call's tag and nothing is ever reset. CTA 0 advances the epoch once it
// has seen every CTA's logits, which is after every CTA has read the old epoch.
// Writers store every word into kFusedReplicas copies and CTA c polls copy
// c % kFusedReplicas, so no cache line is polled by the whole grid.

#include "draft_moe_gemv.cuh"
#include "sgl_kernel/distributed/q35_hop.cuh"

namespace sglang {

namespace draft_moe_fused {

SGL_DEVICE void st_relaxed(uint64_t* ptr, uint64_t v) {
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(ptr), "l"(v) : "memory");
}

SGL_DEVICE void st_relaxed(uint32_t* ptr, uint32_t v) {
  asm volatile("st.relaxed.gpu.global.b32 [%0], %1;" ::"l"(ptr), "r"(v) : "memory");
}

SGL_DEVICE uint64_t ld_relaxed(const uint64_t* ptr) {
  uint64_t v;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(ptr) : "memory");
  return v;
}

// Four tagged 32-bit words; each word is single-copy atomic on its own.
SGL_DEVICE uint4 ld_relaxed_v4(const uint32_t* ptr) {
  uint4 v;
  asm volatile("ld.relaxed.gpu.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr)
               : "memory");
  return v;
}

SGL_DEVICE uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// A cross-CTA wait this long means the grid was not co-resident; trap rather
// than wedge the device.
inline constexpr uint64_t kSpinLimitNs = 2'000'000'000ull;

SGL_DEVICE void check_spin(uint64_t t0) {
  if (globaltimer() - t0 > kSpinLimitNs) __trap();
}

SGL_DEVICE void split(int total, int parts, int idx, int& begin, int& count) {
  const int base = total / parts;
  const int rem = total % parts;
  begin = idx * base + min(idx, rem);
  count = base + (idx < rem ? 1 : 0);
}

// mbarrier wait that sleeps between tries, so idle warps leave the issue slots to the
// warps of the same SM that are computing.
SGL_DEVICE void mbar_wait_sleep(uint64_t* bar, uint32_t parity) {
  uint32_t done;
  while (true) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n\tselp.b32 %0, 1, 0, p;\n\t}\n"
        : "=r"(done)
        : "r"(device::ptx::to_shared(bar)), "r"(parity)
        : "memory");
    if (done) return;
    __nanosleep(64);
  }
}

SGL_DEVICE void named_sync(uint32_t id, uint32_t threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

// Bitonic sort of one lane's keys, descending.
template <int kN>
SGL_DEVICE void sort_desc(uint32_t (&key)[kN]) {
  static_assert((kN & (kN - 1)) == 0);
#pragma unroll
  for (int size = 2; size <= kN; size <<= 1) {
#pragma unroll
    for (int stride = size / 2; stride > 0; stride >>= 1) {
#pragma unroll
      for (int i = 0; i < kN; ++i) {
        const int j = i ^ stride;
        if (j > i) {
          const uint32_t a = key[i], b = key[j];
          const bool desc = (i & size) == 0;
          key[i] = desc ? max(a, b) : min(a, b);
          key[j] = desc ? min(a, b) : max(a, b);
        }
      }
    }
  }
}

}  // namespace draft_moe_fused

struct DraftMoeFusedParams {
  const bf16_t* x;                        // [H], written by an earlier grid
  const bf16_t* __restrict__ router_w;    // [E, H]
  const bf16_t* __restrict__ w13;         // [E, 2I, H]
  const bf16_t* __restrict__ w2;          // [E, H, I]
  const bf16_t* __restrict__ w13_shared;  // [2I, H]
  const bf16_t* __restrict__ w2_shared;   // [H, I]
  const bf16_t* __restrict__ gate_w;      // [H]
  uint64_t* logit_buf;                    // [kFusedReplicas, kFusedLogitWords(E)] tagged logits, then the gate
  uint32_t* act_buf;                      // [kFusedReplicas, (K + 1) * I] tagged act values
  uint32_t* epoch;                        // [1] completed calls
  float* topk_weights;                    // [K]
  int32_t* topk_ids;                      // [K]
  bf16_t* out;                            // [H]
  int64_t* trace;     // [grid, kFusedTraceSlots] clock64 stamps, globaltimer at 0 and 9 (kTrace only)
  q35hop::Site hop;   // kPush: the q35 hop mailbox the output is also pushed into
  int32_t hop_token;  // kPush: the output's row in the hop
};

inline constexpr int kFusedTraceSlots = 32;
inline constexpr int kFusedReplicas = 152;

// E logits and the gate, padded to whole 128-byte lines per replica.
SGL_DEVICE_HOST constexpr int kFusedLogitWords(int experts) {
  return (experts + 1 + 15) / 16 * 16;
}

/**
 * \brief Fused M = 1 draft MoE, one CTA per SM, all CTAs co-resident.
 *
 * Warps 0-7 consume a shared-memory ring of 16 KB stages (two router rows, or
 * one gate/up row pair); warp 8 issues the bulk copies and, once every CTA's
 * logits are published, repeats the top-k selection itself. Router rows and
 * shared-expert weights are requested before the PDL wait; the routed up rows
 * and routed down slices right after the selection. Down slices are the last
 * ring items and stay resident until every act value of the call is published.
 *
 * \tparam kH      hidden size
 * \tparam kI      per-rank intermediate size of one expert
 * \tparam kE      routed experts
 * \tparam kTopK   routed experts per token
 * \tparam kCols   output columns per CTA (grid = ceil(kH / kCols))
 * \tparam kStages ring stages of 16 KB; at least kTopK
 * \tparam kWindow routed pairs a CTA keeps in flight (at most kStages)
 * \tparam kTrace  record per-phase clock64 stamps
 * \tparam kPush   also push the output into the q35 hop (columns split in whole quads)
 */
template <int kH, int kI, int kE, int kTopK, int kCols, int kStages, int kWindow, bool kTrace, bool kUsePDL, bool kPush>
__global__ void __launch_bounds__(288, 1) draft_moe_fused_kernel(const DraftMoeFusedParams __grid_constant__ p) {
  using namespace device;
  namespace fz = draft_moe_fused;
  constexpr int kConsumers = 8;
  constexpr int kConsumerThreads = kConsumers * static_cast<int>(kWarpThreads);
  constexpr int kSlots = kTopK + 1;
  constexpr int kRowBytes = kH * static_cast<int>(sizeof(bf16_t));
  constexpr int kStageBytes = 2 * kRowBytes;
  constexpr int kQuarter = kH / 4;
  using cvec_t = AlignedVector<bf16_t, 8>;
  static_assert(kQuarter % (8 * kWarpThreads) == 0, "hidden size must tile 4 warps x 16 bytes");
  constexpr int kChunksPerLane = kQuarter / (8 * kWarpThreads);
  constexpr int kVecN = static_cast<int>(kMaxVecBytes / sizeof(bf16_t));
  using vec_t = AlignedVector<bf16_t, kVecN>;
  static_assert(kH % (kVecN * kWarpThreads) == 0 && kE % kWarpThreads == 0);
  constexpr int kRouterPerLane = kH / kVecN / static_cast<int>(kWarpThreads);
  constexpr int kPerLaneE = kE / static_cast<int>(kWarpThreads);
  static_assert(kI % 64 == 0, "intermediate size must tile 8 lanes x 16 bytes");
  constexpr int kDownPerLane = kI / 64;
  // A hop push is at least 32 bits wide, so the pushing kernel gives each CTA whole quads of columns.
  constexpr int kColsCap = kPush ? (kCols + 3) / 4 * 4 : kCols;
  static_assert(!kPush || kColsCap <= static_cast<int>(kWarpThreads), "warp 0 pushes every column of its CTA");
  constexpr int kSliceBytes = kColsCap * kI * static_cast<int>(sizeof(bf16_t));
  constexpr int kSliceAlloc = (kSliceBytes + 127) / 128 * 128;
  static_assert(kSliceBytes <= kStageBytes && kStages >= kTopK, "every down slice must own a ring stage");
  constexpr int kMaxPairs = 32;
  constexpr int kActWords = kSlots * kI;
  constexpr int kLogitWords = kFusedLogitWords(kE);

  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* const ring = smem;
  uint8_t* const s_wsd = smem + kStages * kStageBytes;
  bf16_t* const s_x = reinterpret_cast<bf16_t*>(s_wsd + kSliceAlloc);
  bf16_t* const s_act = s_x + kH;
  __shared__ uint64_t full_bar[kStages];
  __shared__ uint64_t empty_bar[kStages];
  __shared__ uint64_t wsd_bar;
  __shared__ uint64_t sel_bar;
  __shared__ float s_part[kMaxPairs][kConsumers];
  __shared__ float s_d[kSlots][kColsCap];
  __shared__ float s_gate_part[4];
  __shared__ float s_v[kE];
  __shared__ float s_w[kTopK];
  __shared__ int32_t s_ids[kTopK];
  __shared__ float s_gate;
  __shared__ uint32_t s_tag;
  __shared__ uint32_t s_stage;
  __shared__ uint32_t s_word[kMaxPairs];

  const int warp = static_cast<int>(threadIdx.x / kWarpThreads);
  const int lane = static_cast<int>(get_lane_id());
  const int G = static_cast<int>(gridDim.x);
  const int c = static_cast<int>(blockIdx.x);
  int row0, nrows, sh0, nsh, rq0, nrq;
  fz::split(kE, G, c, row0, nrows);
  fz::split(kI, G, c, sh0, nsh);
  // Reversed, so the CTAs with an extra router row do not also get an extra routed pair.
  fz::split(kTopK * kI, G, G - 1 - c, rq0, nrq);
  const int nrouter = (nrows + 1) / 2;
  const int nstatic = nrouter + nsh;
  const int nup = nstatic + nrq;
  const int npairs = nsh + nrq;
  int j0, ncols;
  if constexpr (kPush) {
    int q0, nq;
    fz::split(kH / 4, G, c, q0, nq);
    j0 = 4 * q0;
    ncols = 4 * nq;
  } else {
    j0 = c * kCols;
    ncols = min(kCols, kH - j0);
  }
  const uint32_t slice_bytes = static_cast<uint32_t>(ncols) * kI * sizeof(bf16_t);
  const bool gate_cta = c == G - 1;
  const uint64_t* const my_logits = p.logit_buf + (c % kFusedReplicas) * kLogitWords;
  const uint32_t* const my_acts = p.act_buf + (c % kFusedReplicas) * kActWords;
  int64_t* const trace = p.trace + c * kFusedTraceSlots;
  // After a CTA barrier the clock is read behind a dependent shared load: the barrier does
  // not hold back the clock read itself.
  auto stamp_after_sync = [&](int slot) {
    if constexpr (kTrace) {
      if (threadIdx.x == 0 && *reinterpret_cast<volatile uint32_t*>(&s_tag) != 0xffffffffu) trace[slot] = clock64();
    }
  };

  if (threadIdx.x == 0) {
    if constexpr (kTrace) {
      trace[0] = static_cast<int64_t>(fz::globaltimer());
      trace[23] = clock64();
    }
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      ptx::mbar_init(&full_bar[s], 1);
      ptx::mbar_init(&empty_bar[s], kConsumers);
    }
    ptx::mbar_init(&wsd_bar, 1);
    ptx::mbar_init(&sel_bar, 1);
    draft_moe::fence_mbarrier_init();
  }
  __syncthreads();
  PDLTriggerSecondary<kUsePDL>();

  // Ring item n: two router rows, then shared-expert pairs, routed pairs, routed down slices.
  auto issue = [&](int n) {
    const int stage = n % kStages;
    uint8_t* dst = ring + stage * kStageBytes;
    uint64_t* bar = &full_bar[stage];
    if (n < nrouter) {
      const int r = row0 + 2 * n;
      const uint32_t bytes = static_cast<uint32_t>(min(2, row0 + nrows - r)) * kRowBytes;
      ptx::mbar_arrive_expect_tx(bar, bytes);
      draft_moe::bulk_g2s(dst, p.router_w + static_cast<int64_t>(r) * kH, bytes, bar);
      return;
    }
    if (n >= nup) {
      const int64_t e = s_ids[n - nup];
      ptx::mbar_arrive_expect_tx(bar, slice_bytes);
      draft_moe::bulk_g2s(dst, p.w2 + (e * kH + j0) * kI, slice_bytes, bar);
      return;
    }
    const bf16_t* src;
    if (n < nstatic) {
      src = p.w13_shared + static_cast<int64_t>(sh0 + n - nrouter) * kH;
    } else {
      const int rq = rq0 + n - nstatic;
      src = p.w13 + (static_cast<int64_t>(s_ids[rq / kI]) * 2 * kI + rq % kI) * kH;
    }
    ptx::mbar_arrive_expect_tx(bar, kStageBytes);
    draft_moe::bulk_g2s(dst, src, kRowBytes, bar);
    draft_moe::bulk_g2s(dst + kRowBytes, src + static_cast<int64_t>(kI) * kH, kRowBytes, bar);
  };

  if (warp == kConsumers && lane == 0) {
    for (int n = 0; n < nstatic; ++n)
      issue(n);
    ptx::mbar_arrive_expect_tx(&wsd_bar, slice_bytes);
    draft_moe::bulk_g2s(s_wsd, p.w2_shared + static_cast<int64_t>(j0) * kI, slice_bytes, &wsd_bar);
    if constexpr (kTrace) trace[19] = clock64();
  }
  // Warp w < nrows computes router row row0 + w out of ring item w / 2.
  const bool router_warp = warp < nrows;
  cvec_t gw[kChunksPerLane];
  if (gate_cta && warp < 4) {
#pragma unroll
    for (int k = 0; k < kChunksPerLane; ++k)
      gw[k].load(p.gate_w + warp * kQuarter, k * kWarpThreads + lane);
  }

  PDLWaitPrimary<kUsePDL>();
  for (int i = static_cast<int>(threadIdx.x); i < kH / 8; i += static_cast<int>(blockDim.x)) {
    cvec_t v;
    v.load(p.x, i);
    v.store(s_x, i);
  }
  if (threadIdx.x == 0) {
    s_tag = *p.epoch + 1u;
    if constexpr (kPush) s_stage = q35hop::stage_peek(p.hop);
  }
  __syncthreads();
  const uint32_t tag = s_tag;
  stamp_after_sync(1);

  if (warp == kConsumers) {
    // Every CTA's logits, in the lane layout of the router's selection warp.
    uint64_t lw[kPerLaneE];
#pragma unroll
    for (int i = 0; i < kPerLaneE; ++i)
      lw[i] = fz::ld_relaxed(my_logits + i * kWarpThreads + lane);
    const uint64_t t0 = fz::globaltimer();
    while (true) {
      bool ready = true;
#pragma unroll
      for (int i = 0; i < kPerLaneE; ++i)
        ready &= static_cast<uint32_t>(lw[i] >> 32) == tag;
      if (__all_sync(0xffffffffu, ready)) break;
      fz::check_spin(t0);
#pragma unroll
      for (int i = 0; i < kPerLaneE; ++i) {
        if (static_cast<uint32_t>(lw[i] >> 32) != tag) lw[i] = fz::ld_relaxed(my_logits + i * kWarpThreads + lane);
      }
    }
    if constexpr (kTrace) {
      if (lane == 0) {
        trace[2] = clock64();
        trace[28] = static_cast<int64_t>(fz::globaltimer());
      }
    }

    // Keys, max and sum exactly as in the selection warp of draft_router_kernel.
    float v[kPerLaneE];
    uint32_t key[kPerLaneE];
    float row_max = draft_moe::kNegInf;
#pragma unroll
    for (int i = 0; i < kPerLaneE; ++i) {
      const float l = __uint_as_float(static_cast<uint32_t>(lw[i]));
      v[i] = l == l ? l : -1e30f;
      s_v[i * kWarpThreads + lane] = v[i];
      row_max = fmaxf(row_max, v[i]);
      uint32_t b = __float_as_uint(cast<fp32_t>(cast<bf16_t>(v[i]))) >> 16;
      b = b == 0x8000u ? 0u : b;
      const uint32_t ord = (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
      key[i] = (ord << 16) | (0xffffu - static_cast<uint32_t>(i * kWarpThreads + lane));
    }
    // Round k of the router's warp max picks the k-th largest (unique) key; with
    // each lane's keys sorted, a round is one warp max and a shift.
    fz::sort_desc(key);
    uint32_t my_key = 0;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      const uint32_t best = __reduce_max_sync(0xffffffffu, key[0]);
      if (lane == k) my_key = best;
      const bool pop = key[0] == best;
#pragma unroll
      for (int i = 0; i + 1 < kPerLaneE; ++i)
        key[i] = pop ? key[i + 1] : key[i];
      key[kPerLaneE - 1] = pop ? 0u : key[kPerLaneE - 1];
    }
    const int32_t id = lane < kTopK ? static_cast<int32_t>(0xffffu - (my_key & 0xffffu)) : 0;
    if (lane < kTopK) s_ids[lane] = id;
    __syncwarp();
    // The routed copies need only the ids; the weights are only read by the final combine.
    int next = nstatic;
    if (lane == 0) {
      for (; next < min(nstatic + kWindow, nup); ++next) {
        ptx::mbar_wait_parity(&empty_bar[next % kStages], ((next / kStages) & 1) ^ 1);
        issue(next);
      }
    }
    row_max = warp::reduce_max(row_max);
    float lsum = 0.0f;
#pragma unroll
    for (int i = 0; i < kPerLaneE; ++i)
      lsum += expf(v[i] - row_max);
    const float row_sum = warp::reduce_sum(lsum);
    const float pw = expf(s_v[id] - row_max) / row_sum;
    float routed_sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k)
      routed_sum += __shfl_sync(0xffffffffu, pw, k);
    const float norm = routed_sum > 0.0f ? routed_sum : 1.0f;
    if (lane < kTopK) {
      const float w = pw / norm;
      s_w[lane] = w;
      if (c == 0) {
        p.topk_weights[lane] = w;
        p.topk_ids[lane] = id;
      }
    }
    __syncwarp();
    if (lane == 0) {
      if (c == 0) *p.epoch = tag;
      ptx::mbar_arrive(&sel_bar);
      if constexpr (kTrace) trace[3] = clock64();
      for (int n = next; n < nup + kTopK; ++n) {
        // A short window keeps every CTA's rows arriving at the same pace.
        const int m = n - kWindow;
        if (n < nup && m >= nstatic) ptx::mbar_wait_parity(&empty_bar[m % kStages], (m / kStages) & 1);
        ptx::mbar_wait_parity(&empty_bar[n % kStages], ((n / kStages) & 1) ^ 1);
        issue(n);
        if constexpr (kTrace) {
          if (n == nup - 1) trace[12] = clock64();
        }
      }
      if constexpr (kTrace) trace[11] = clock64();
    }
    return;
  }

  if (router_warp) {
    const int item = warp / 2;
    fz::mbar_wait_sleep(&full_bar[item], 0);
    const auto* wrow = reinterpret_cast<const bf16_t*>(ring + item * kStageBytes + (warp & 1) * kRowBytes);
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < kRouterPerLane; ++k) {
      vec_t w, xv;
      w.load(wrow, k * kWarpThreads + lane);
      xv.load(s_x, k * kWarpThreads + lane);
#pragma unroll
      for (int j = 0; j < kVecN; ++j)
        acc = fmaf(cast<fp32_t>(w[j]), cast<fp32_t>(xv[j]), acc);
    }
    acc = warp::reduce_sum(acc);
    if constexpr (kTrace) {
      if (threadIdx.x == 0) trace[20] = clock64();
    }
    const uint64_t word = (static_cast<uint64_t>(tag) << 32) | __float_as_uint(draft_moe::round_bf16(acc));
    for (int r = lane; r < kFusedReplicas; r += kWarpThreads) {
      fz::st_relaxed(p.logit_buf + r * kLogitWords + row0 + warp, word);
    }
    if constexpr (kTrace) {
      if (lane == 0) trace[24 + warp] = static_cast<int64_t>(fz::globaltimer());
      if (threadIdx.x == 0) trace[4] = clock64();
    }
  }

  // Consumers: x quarter in registers, as in the up_tma consumers.
  const int quarter = warp & 3;
  const bool is_up = warp >= 4;
  cvec_t xr[kChunksPerLane];
#pragma unroll
  for (int k = 0; k < kChunksPerLane; ++k)
    xr[k].load(s_x + quarter * kQuarter, k * kWarpThreads + lane);
  if (gate_cta && warp < 4) {
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < kChunksPerLane; ++k) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        acc = fmaf(cast<fp32_t>(xr[k][j]), cast<fp32_t>(gw[k][j]), acc);
    }
    acc = warp::reduce_sum(acc);
    if (lane == 0) s_gate_part[warp] = acc;
  }

  for (int n = 0; n < nup; ++n) {
    const int stage = n % kStages;
    fz::mbar_wait_sleep(&full_bar[stage], (n / kStages) & 1);
    if (n < nrouter) {
      __syncwarp();
      if (lane == 0) ptx::mbar_arrive(&empty_bar[stage]);
      continue;
    }
    if constexpr (kTrace) {
      if (n == nstatic && threadIdx.x == 0) trace[5] = clock64();
    }
    const auto* row =
        reinterpret_cast<const bf16_t*>(ring + stage * kStageBytes + (is_up ? kRowBytes : 0)) + quarter * kQuarter;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < kChunksPerLane; ++k) {
      cvec_t w;
      w.load(row, k * kWarpThreads + lane);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        acc = fmaf(cast<fp32_t>(w[j]), cast<fp32_t>(xr[k][j]), acc);
    }
    acc = warp::reduce_sum(acc);
    if (lane == 0) {
      s_part[n - nrouter][warp] = acc;
      ptx::mbar_arrive(&empty_bar[stage]);
    }
  }

  fz::named_sync(1, kConsumerThreads);
  stamp_after_sync(6);
  if (static_cast<int>(threadIdx.x) < npairs) {
    const int t = static_cast<int>(threadIdx.x);
    const float g = draft_moe::round_bf16(((s_part[t][0] + s_part[t][1]) + s_part[t][2]) + s_part[t][3]);
    const float u = draft_moe::round_bf16(((s_part[t][4] + s_part[t][5]) + s_part[t][6]) + s_part[t][7]);
    const bf16_t a = cast<bf16_t>(draft_moe::silu_f32(g) * u);
    s_word[t] = ((tag & 0xffffu) << 16) | static_cast<uint32_t>(__bfloat16_as_ushort(a));
  }
  fz::named_sync(1, kConsumerThreads);
  // One warp store per replica: the routed words of a CTA are contiguous, so a replica
  // costs one or two lines instead of one line per word.
  if (lane < npairs) {
    const int idx = lane < nsh ? kTopK * kI + sh0 + lane : rq0 + lane - nsh;
    const uint32_t word = s_word[lane];
    for (int r = warp; r < kFusedReplicas; r += kConsumers)
      fz::st_relaxed(p.act_buf + r * kActWords + idx, word);
  }
  if constexpr (kTrace) {
    if (threadIdx.x == 0) trace[29] = static_cast<int64_t>(fz::globaltimer());
  }
  if (gate_cta) {
    const float t = ((s_gate_part[0] + s_gate_part[1]) + s_gate_part[2]) + s_gate_part[3];
    const float gate = 1.0f / (1.0f + expf(-t));
    const uint64_t word = (static_cast<uint64_t>(tag) << 32) | __float_as_uint(gate);
    for (int r = static_cast<int>(threadIdx.x); r < kFusedReplicas; r += kConsumerThreads) {
      fz::st_relaxed(p.logit_buf + r * kLogitWords + kE, word);
    }
  }

  // Act values of the whole call: the first loads go out before the down slices are waited
  // on, and with few enough registers each group lifts its down weights out of shared memory
  // while the act values are still on the way.
  constexpr int kGroups = kConsumers * 4;
  constexpr int kRounds = (kColsCap + kGroups - 1) / kGroups;
  constexpr int kActVecs = kActWords / 4;
  constexpr int kActVecsPerThread = (kActVecs + kConsumerThreads - 1) / kConsumerThreads;
  const uint32_t tag16 = tag & 0xffffu;
  uint4 av[kActVecsPerThread];
#pragma unroll
  for (int i = 0; i < kActVecsPerThread; ++i) {
    const int vi = static_cast<int>(threadIdx.x) + i * kConsumerThreads;
    if (vi < kActVecs) av[i] = fz::ld_relaxed_v4(my_acts + 4 * vi);
  }
  const int grp = warp * 4 + lane / 8;
  const int gl = lane % 8;
  const bf16_t* slices[kSlots];
  fz::mbar_wait_sleep(&wsd_bar, 0);
  slices[kTopK] = reinterpret_cast<const bf16_t*>(s_wsd);
#pragma unroll
  for (int s = 0; s < kTopK; ++s) {
    const int n = nup + s;
    fz::mbar_wait_sleep(&full_bar[n % kStages], (n / kStages) & 1);
    slices[s] = reinterpret_cast<const bf16_t*>(ring + (n % kStages) * kStageBytes);
  }
  constexpr bool kPreload = kRounds * kSlots * kDownPerLane <= 24;
  cvec_t wpre[kPreload ? kRounds : 1][kPreload ? kSlots : 1][kPreload ? kDownPerLane : 1];
  if constexpr (kPreload) {
#pragma unroll
    for (int r = 0; r < kRounds; ++r) {
      const int jr = min(r * kGroups + grp, ncols - 1);
#pragma unroll
      for (int s = 0; s < kSlots; ++s) {
#pragma unroll
        for (int k = 0; k < kDownPerLane; ++k)
          wpre[r][s][k].load(slices[s] + jr * kI, gl + 8 * k);
      }
    }
  }
  {
    const uint64_t t0 = fz::globaltimer();
    while (true) {
      bool ready = true;
#pragma unroll
      for (int i = 0; i < kActVecsPerThread; ++i) {
        const int vi = static_cast<int>(threadIdx.x) + i * kConsumerThreads;
        if (vi >= kActVecs) continue;
        const bool ok = (av[i].x >> 16) == tag16 && (av[i].y >> 16) == tag16 && (av[i].z >> 16) == tag16 &&
                        (av[i].w >> 16) == tag16;
        if (!ok) {
          ready = false;
          av[i] = fz::ld_relaxed_v4(my_acts + 4 * vi);
        }
      }
      if (ready) break;
      fz::check_spin(t0);
    }
#pragma unroll
    for (int i = 0; i < kActVecsPerThread; ++i) {
      const int vi = static_cast<int>(threadIdx.x) + i * kConsumerThreads;
      if (vi < kActVecs) {
        uint2 packed;
        packed.x = (av[i].x & 0xffffu) | (av[i].y << 16);
        packed.y = (av[i].z & 0xffffu) | (av[i].w << 16);
        *reinterpret_cast<uint2*>(s_act + 4 * vi) = packed;
      }
    }
  }
  if (threadIdx.x == 0) {
    uint64_t gword = fz::ld_relaxed(my_logits + kE);
    const uint64_t t0 = fz::globaltimer();
    while (static_cast<uint32_t>(gword >> 32) != tag) {
      fz::check_spin(t0);
      gword = fz::ld_relaxed(my_logits + kE);
    }
    s_gate = __uint_as_float(static_cast<uint32_t>(gword));
    if constexpr (kTrace) trace[30] = static_cast<int64_t>(fz::globaltimer());
  }
  fz::mbar_wait_sleep(&sel_bar, 0);
  fz::named_sync(1, kConsumerThreads);
  stamp_after_sync(7);

  // Down pass of down_v3: one 8-lane group per (slot, column), the kSlots independent
  // dots of a group interleaved.
#pragma unroll
  for (int r = 0; r < kRounds; ++r) {
    const int jj = r * kGroups + grp;
    const int jr = min(jj, ncols - 1);
    float d[kSlots];
#pragma unroll
    for (int s = 0; s < kSlots; ++s) {
      float acc = 0.0f;
#pragma unroll
      for (int k = 0; k < kDownPerLane; ++k) {
        cvec_t w, a;
        if constexpr (kPreload) {
          w = wpre[r][s][k];
        } else {
          w.load(slices[s] + jr * kI, gl + 8 * k);
        }
        a.load(s_act + s * kI, gl + 8 * k);
#pragma unroll
        for (int e = 0; e < 8; ++e)
          acc = fmaf(cast<fp32_t>(w[e]), cast<fp32_t>(a[e]), acc);
      }
      d[s] = acc;
    }
#pragma unroll
    for (int s = 0; s < kSlots; ++s)
      d[s] += __shfl_xor_sync(0xffffffffu, d[s], 4);
#pragma unroll
    for (int s = 0; s < kSlots; ++s)
      d[s] += __shfl_xor_sync(0xffffffffu, d[s], 2);
#pragma unroll
    for (int s = 0; s < kSlots; ++s)
      d[s] += __shfl_xor_sync(0xffffffffu, d[s], 1);
    if (gl == 0 && jj < ncols) {
#pragma unroll
      for (int s = 0; s < kSlots; ++s)
        s_d[s][jj] = d[s];
    }
  }
  fz::named_sync(1, kConsumerThreads);
  stamp_after_sync(8);
  auto column = [&](int jj) {
    float sum = 0.0f;
#pragma unroll
    for (int s = 0; s < kTopK; ++s)
      sum += draft_moe::round_bf16(s_d[s][jj] * s_w[s]);
    const float routed = draft_moe::round_bf16(sum);
    const float shared = draft_moe::round_bf16(s_d[kTopK][jj]);
    float r = s_gate * shared;
    r += routed;
    return cast<bf16_t>(r);
  };
  if constexpr (kPush) {
    if (warp == 0) {
      const bf16_t v = column(min(lane, ncols - 1));
      const uint32_t b0 = __bfloat16_as_ushort(v);
      const uint32_t b1 = __shfl_down_sync(0xffffffffu, b0, 1);
      const uint32_t b2 = __shfl_down_sync(0xffffffffu, b0, 2);
      const uint32_t b3 = __shfl_down_sync(0xffffffffu, b0, 3);
      // Push before the store: the push starts the flight to every rank.
      if ((lane & 3) == 0 && lane < ncols)
        q35hop::push_quad(p.hop, s_stage, p.hop_token, (j0 + lane) >> 2, make_uint2(b0 | (b1 << 16), b2 | (b3 << 16)));
      if (lane < ncols) p.out[j0 + lane] = v;
    }
  } else if (static_cast<int>(threadIdx.x) < ncols) {
    const int jj = static_cast<int>(threadIdx.x);
    p.out[j0 + jj] = column(jj);
  }
  if constexpr (kTrace) {
    if (threadIdx.x == 0) {
      trace[15] = clock64();
      trace[9] = static_cast<int64_t>(fz::globaltimer());
    }
  }
}

template <int kH, int kI, int kE, int kTopK, int kCols, int kStages, int kWindow, bool kTrace, bool kUsePDL, bool kPush>
struct DraftMoeFused {
  static_assert(kWindow >= 1 && kWindow <= kStages);
  static constexpr int kThreads = 288;
  static constexpr int kColsCap = kPush ? (kCols + 3) / 4 * 4 : kCols;
  static constexpr std::size_t kSmem = static_cast<std::size_t>(kStages) * 4 * kH +
                                       (static_cast<std::size_t>(kColsCap) * kI * 2 + 127) / 128 * 128 + 2 * kH +
                                       2 * static_cast<std::size_t>(kTopK + 1) * kI;

  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView router_w,
      const tvm::ffi::TensorView w13,
      const tvm::ffi::TensorView w2,
      const tvm::ffi::TensorView w13_shared,
      const tvm::ffi::TensorView w2_shared,
      const tvm::ffi::TensorView gate_w,
      const tvm::ffi::TensorView logit_buf,
      const tvm::ffi::TensorView act_buf,
      const tvm::ffi::TensorView epoch,
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView trace,
      const tvm::ffi::TensorView hop_site,
      int64_t hop_token) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(x);
    TensorMatcher({kE, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(router_w);
    TensorMatcher({kE, 2 * kI, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(w13);
    TensorMatcher({kE, kH, kI}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(w2);
    TensorMatcher({2 * kI, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(w13_shared);
    TensorMatcher({kH, kI}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(w2_shared);
    TensorMatcher({kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(gate_w);
    TensorMatcher({kFusedReplicas * kFusedLogitWords(kE)}).with_dtype<int64_t>().with_device(device).verify(logit_buf);
    TensorMatcher({kFusedReplicas * (kTopK + 1) * kI})
        .with_dtype<int32_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(act_buf);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(epoch);
    TensorMatcher({1, kTopK}).with_dtype<fp32_t>().with_device(device).verify(topk_weights);
    TensorMatcher({1, kTopK}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({1, kH}).with_dtype<bf16_t>().with_device(device).verify(out);
    const DLDevice dev = device.unwrap();
    const int grid = (kH + kCols - 1) / kCols;
    if constexpr (kTrace) {
      TensorMatcher({grid * kFusedTraceSlots}).with_dtype<int64_t>().with_device(device).verify(trace);
    }
    auto params = DraftMoeFusedParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .router_w = static_cast<const bf16_t*>(router_w.data_ptr()),
        .w13 = static_cast<const bf16_t*>(w13.data_ptr()),
        .w2 = static_cast<const bf16_t*>(w2.data_ptr()),
        .w13_shared = static_cast<const bf16_t*>(w13_shared.data_ptr()),
        .w2_shared = static_cast<const bf16_t*>(w2_shared.data_ptr()),
        .gate_w = static_cast<const bf16_t*>(gate_w.data_ptr()),
        .logit_buf = static_cast<uint64_t*>(logit_buf.data_ptr()),
        .act_buf = static_cast<uint32_t*>(act_buf.data_ptr()),
        .epoch = static_cast<uint32_t*>(epoch.data_ptr()),
        .topk_weights = static_cast<float*>(topk_weights.data_ptr()),
        .topk_ids = static_cast<int32_t*>(topk_ids.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .trace = static_cast<int64_t*>(trace.data_ptr()),
        .hop = {},
        .hop_token = static_cast<int32_t>(hop_token),
    };
    if constexpr (kPush) {
      // [mc, uc, ctrl, rank, world, m_cap, timeout_ns] of Q35Hop.site_fields(), host int64.
      CHECK_HOST(hop_site.device().device_type == kDLCPU && hop_site.numel() == 7) << "hop_site must be 7 host int64";
      const auto* f = static_cast<const int64_t*>(hop_site.data_ptr());
      params.hop = q35hop::Site{
          reinterpret_cast<void*>(f[0]),
          reinterpret_cast<void*>(f[1]),
          reinterpret_cast<q35hop::Ctrl*>(f[2]),
          static_cast<int>(f[3]),
          static_cast<int>(f[4]),
          static_cast<int>(f[5]),
          static_cast<unsigned long long>(f[6])};
      CHECK_HOST(hop_token >= 0 && hop_token < params.hop.m_cap) << "hop row out of range";
    }
    auto kernel = draft_moe_fused_kernel<kH, kI, kE, kTopK, kCols, kStages, kWindow, kTrace, kUsePDL, kPush>;
    static const bool attr_set = [&] {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
      return true;
    }();
    (void)attr_set;
    // CTAs spin on each other's results, so the whole grid has to be resident at once.
    static const uint32_t per_sm = runtime::get_blocks_per_sm(kernel, kThreads, kSmem);
    const auto sms = runtime::get_sm_count(dev.device_id);
    CHECK_HOST(per_sm >= 1 && static_cast<uint32_t>(grid) <= per_sm * sms)
        << "fused draft MoE grid " << grid << " does not fit: " << per_sm << " CTA/SM x " << sms << " SMs";
    const int max_static = ((kE + grid - 1) / grid + 1) / 2 + (kI + grid - 1) / grid;
    const int max_pairs = (kI + grid - 1) / grid + (kTopK * kI + grid - 1) / grid;
    CHECK_HOST(max_static <= kStages && max_pairs <= 32 && (kE + grid - 1) / grid <= 8)
        << "too much work per CTA for grid " << grid;
    LaunchKernel(grid, kThreads, dev, kSmem).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
