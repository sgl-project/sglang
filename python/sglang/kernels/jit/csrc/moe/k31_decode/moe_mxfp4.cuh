// Kimi-K3.1 decode routed MoE (TP-MoE rank: all 896 experts, intermediate shard 384, top-16), MXFP4 weights x MXFP8
// activations, SiTU, on SM100 tcgen05. Two kernels chained with PDL:
//   route_kernel (2 CTAs): CTA 0 builds the permutation tables (expert histogram + scan of the precomputed top-k),
//       CTA 1 resets the MoE kernel's counters;
//   moe_kernel (persistent): FC1 + SiTU + MXFP8 requant, then FC2, writing the engine's deferred-finalize rows
//       (gemm2_out in a permuted row order + expanded_idx); the engine applies the top-k weights later.
//
// Weights are consumed in the layout the engine prepares for flashinfer's TRT-LLM-gen MXFP4 MoE (no repack):
//   W13 [E, 768, 1792] u8: row 2i = up_i, 2i+1 = gate_i, then within each 32-row block stored row s = source row
//       4 (s % 8) + s / 8. W2 [E, 3584, 192] u8: the same 32-row shuffle. Scales: 128x4 atoms (512 B: row m of a
//       128-row tile, scale column k: (m % 32) * 16 + (m / 32) * 4 + k % 4), tiles row-major over (m / 128, k / 4).
//
// Work: virtual experts v = (expert e, token chunk c of <= kNCap tokens). FC1 item (v, j): 128 W13 rows (64 channels)
// x K 3584 (14 stages of K 256); FC2 item (v, j): two 128-row W2 tiles sharing the token tile x K 384 (3 stages of
// K 128). Every CTA claims FC1 items from one global queue until it is empty, then FC2 items from a second one; each
// role runs an FC1 loop, then an FC2 loop, on one continuous stage ring. FC2 of v waits for v's 6 FC1 items (per-v
// counter, release / acquire), any CTA's.
//
// moe_kernel warp roles (256 threads): 0 TMA producer (one lane), 1 MMA issuer, 2 scheduler (claims, token lists),
// 3 TMEM allocator, then token scales, 4-7 epilogue (one TMEM lane quadrant each).
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/atomic.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/ptx/async.cuh>
#include <sgl_kernel/ptx/barrier.cuh>
#include <sgl_kernel/ptx/tcgen05.cuh>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace sglang {

namespace device::k31_moe {

inline constexpr uint32_t kNumExperts = 896;
inline constexpr uint32_t kTopK = 16;
inline constexpr uint32_t kHidden = 3584;
inline constexpr uint32_t kInter = 384;
inline constexpr uint32_t kMaxT = 256;
inline constexpr uint32_t kMaxWords = kMaxT / 32;  // 32-token words of a token bitmask
// virtual experts: sum over touched experts of ceil(n_e / kNCap) <= 896 + T * 16 / 32 at kNCap = 32
inline constexpr uint32_t kMaxV = 1024;
// counters (zeroed by the routing kernel): [0, kMaxV) FC1 items done per virtual expert, then the two claim cursors
inline constexpr uint32_t kClaim1Word = kMaxV;
inline constexpr uint32_t kClaim2Word = kMaxV + 1;
inline constexpr uint32_t kCounterWords = kMaxV + 2;

inline constexpr uint32_t kFc1Tiles = 2 * kInter / 128;                   // 6
inline constexpr uint32_t kFc2Items = kHidden / 256;                      // 14: two 128-row M tiles each
inline constexpr uint32_t kFc1Stages = kHidden / 256;                     // K 256 per stage (2 K-atoms of 128)
inline constexpr uint32_t kFc2Stages = kInter / 128;                      // K 128 per stage, 2 M tiles
inline constexpr uint32_t kW13SfPerExpert = 2 * kInter * (kHidden / 32);  // 86016
inline constexpr uint32_t kW2SfPerExpert = kHidden * (kInter / 32);       // 43008
inline constexpr uint32_t kAtomBytes = 512;                               // one 128x4 scale atom
inline constexpr uint32_t kWeightBoxTx = 128 * 128 / 2;  // a 128 x 128 fp4 box: tx counts the packed global bytes
inline constexpr uint32_t kInfoDepth = 4;  // also the claim lookahead: deeper hoards items and unbalances the tail
inline constexpr uint32_t kNumThreads = 256;
inline constexpr uint32_t kRouteThreads = 1024;

using tcgen05 = ptx::TCGen05<1>;

namespace ptx {

using namespace device::ptx;

SGL_DEVICE uint64_t make_evict_first_policy() {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}

SGL_DEVICE void
tma_load_2d_hint(uint32_t dst, const CUtensorMap* tmap, int32_t x, int32_t y, uint64_t* bar, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, "
      "%3}], [%4], %5;" ::"r"(dst),
      "l"(tmap),
      "r"(x),
      "r"(y),
      "r"(to_shared(bar)),
      "l"(policy)
      : "memory");
}

SGL_DEVICE void bulk_load_hint(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(
          to_shared(dst)),
      "l"(src),
      "r"(bytes),
      "r"(to_shared(bar)),
      "l"(policy)
      : "memory");
}

SGL_DEVICE void tma_gather4(
    uint32_t dst, const CUtensorMap* tmap, int32_t x, int32_t r0, int32_t r1, int32_t r2, int32_t r3, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, "
      "%5, %6}], [%7];" ::"r"(dst),
      "l"(tmap),
      "r"(x),
      "r"(r0),
      "r"(r1),
      "r"(r2),
      "r"(r3),
      "r"(to_shared(bar))
      : "memory");
}

SGL_DEVICE void fence_proxy_async_global() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}

SGL_DEVICE void red_async_release_add(uint32_t* addr, uint32_t value) {
  asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;" ::"l"(addr), "r"(value) : "memory");
}

SGL_DEVICE float tanh_approx(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

}  // namespace ptx

// UE8M0 scale of a 32-block with amax `amax`: 2^ceil(log2(amax / 448)) (round up), as the biased exponent byte.
SGL_DEVICE uint32_t ue8m0_round_up(float amax) {
  const uint32_t bits = __float_as_uint(amax * (1.0f / 448.0f));
  const uint32_t e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0);  // not a power of two: round the exponent up
  return min(e, 254u);
}

// beta * tanh(gate / beta) * sigmoid(gate) * lb * tanh(up / lb), beta = 4, lb = 25, sigmoid(x) = (1 + tanh(x / 2)) / 2;
// MUFU approximations (rel. error ~2^-11) ahead of an E4M3 quantization (2^-4).
SGL_DEVICE float situ(float gate, float up) {
  const float g = 2.0f * ptx::tanh_approx(gate * 0.25f) * (1.0f + ptx::tanh_approx(gate * 0.5f));
  const float u = 25.0f * ptx::tanh_approx(up * 0.04f);
  return g * u;
}

template <uint32_t N>
struct Ring {
  uint32_t stage = 0, phase = 0;
  SGL_DEVICE void advance() {
    if (++stage == N) {
      stage = 0;
      phase ^= 1;
    }
  }
};

// vinfo[v] of virtual expert v (tokens <= 32, first row < 4096, chunk < 8); vinfo[kMaxV] is their count.
struct VInfo {
  uint32_t bits;
  SGL_DEVICE static uint32_t pack(uint32_t e, uint32_t row_base, uint32_t n, uint32_t chunk) {
    return e | (row_base << 10) | (n << 22) | (chunk << 28);
  }
  SGL_DEVICE uint32_t expert() const {
    return bits & 1023;
  }
  SGL_DEVICE uint32_t row_base() const {
    return (bits >> 10) & 4095;
  }
  SGL_DEVICE uint32_t num_tokens() const {
    return (bits >> 22) & 63;
  }
  SGL_DEVICE uint32_t chunk() const {
    return bits >> 28;
  }
};

// ---- routing ---------------------------------------------------------------------------------------------------

struct RouteParams {
  const int32_t* __restrict__ topk_ids;  // [T, 16]: expert ids, or (id << 16) | bf16(weight) when kPacked
  uint16_t* __restrict__ topk_weights;   // kPacked only: [T, 16] bf16, the weights of the packed form
  int32_t* __restrict__ expanded_idx;    // [T * 16] -> permuted row
  uint32_t* __restrict__ token_bits;     // [kMaxV, kMaxWords]: v's expert's bitmask, bit t % 32 of word t / 32
  uint32_t* __restrict__ vinfo;          // [kMaxV + 1]
  uint32_t* __restrict__ counters;       // [kCounterWords], zeroed here for the MoE kernel
  uint32_t num_tokens;
};

// The rows of an expert are its tokens in token order: the rank of token t in expert e is the number of set bits
// below t in e's token bitmask. The histogram builds the bitmask with atomicOr, which is order-independent, so the
// tables are deterministic.
template <uint32_t kNCap, bool kPacked>
__global__ void __launch_bounds__(kRouteThreads, 1) route_kernel(const RouteParams p) {
  __shared__ alignas(16) uint32_t s_bits[kMaxWords][kNumExperts];  // word-major: conflict-free per expert
  __shared__ uint16_t s_row[kMaxWords][kNumExperts];               // first row of the tokens of word w of expert e
  __shared__ uint32_t s_scan[32];
  const uint32_t tid = threadIdx.x;
  if (blockIdx.x == 1) {
    // CTA 1 only resets the MoE kernel's counters, once the previous kernels of the stream have completed
    PDLWaitPrimary<true>();
    for (uint32_t i = tid; i < kCounterWords; i += kRouteThreads) {
      p.counters[i] = 0;
    }
    return;
  }
  // early trigger: the MoE kernel's PDL wait covers this grid's completion, so the trigger only lets it launch and
  // run its prologue (TMEM alloc, barrier init) while the tables are built
  PDLTriggerSecondary<true>();
  const uint32_t lane_id = tid % 32, warp_id = tid / 32;
  const uint32_t num_words = div_ceil(p.num_tokens, 32u);
  AlignedVector<uint32_t, 4> zero;
  zero.fill(0);
  for (uint32_t i = tid; i < num_words * kNumExperts / 4; i += kRouteThreads) {
    zero.store(s_bits, i);
  }
  PDLWaitPrimary<true>();
  // thread i: routes [4 i, 4 i + 4) = token i / 4, k in [4 (i % 4), 4 (i % 4) + 4)
  const uint32_t t = tid / 4;
  const bool active = t < p.num_tokens;
  AlignedVector<uint32_t, 4> ids;
  ids.fill(0);
  if (active) ids.load(p.topk_ids, tid);
  if constexpr (kPacked) {
    // the packed form of the route + quant handoff: keep the ids, pass the weights on to the engine's finalize
    AlignedVector<uint16_t, 4> weights;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      weights[j] = static_cast<uint16_t>(ids[j] & 0xFFFF);
      ids[j] >>= 16;
    }
    if (active) weights.store(p.topk_weights, tid);
  }
  __syncthreads();
  if (active) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      atomicOr(&s_bits[t / 32][ids[j]], 1u << (t % 32));
    }
  }
  __syncthreads();
  // thread e: its bitmask, row count and virtual expert count; exclusive scan of both over the experts
  const uint32_t e = tid;
  AlignedVector<uint32_t, kMaxWords> bits;
  uint32_t cnt = 0;
#pragma unroll
  for (uint32_t w = 0; w < kMaxWords; ++w) {
    bits[w] = (e < kNumExperts && w < num_words) ? s_bits[w][e] : 0u;
    cnt += __popc(bits[w]);
  }
  const uint32_t nv = div_ceil(cnt, kNCap);
  const uint32_t packed = nv | (cnt << 16);  // sums: nv <= 1024, cnt <= 4096
  const uint32_t incl = warp::inclusive_sum(packed, lane_id);
  if (lane_id == 31) s_scan[warp_id] = incl;
  __syncthreads();
  const uint32_t excl = warp::reduce_sum(lane_id < warp_id ? s_scan[lane_id] : 0u) + incl - packed;
  const uint32_t v0 = excl & 0xFFFF, r0 = excl >> 16;
  if (e < kNumExperts) {
    uint32_t r = r0;
#pragma unroll
    for (uint32_t w = 0; w < kMaxWords; ++w) {
      if (w < num_words) s_row[w][e] = static_cast<uint16_t>(r);
      r += __popc(bits[w]);
    }
  }
  __syncthreads();
  if (e < kNumExperts) {
    for (uint32_t c = 0; c < nv; ++c) {
      p.vinfo[v0 + c] = VInfo::pack(e, r0 + c * kNCap, min(kNCap, cnt - c * kNCap), c);
      bits.store(p.token_bits, v0 + c);
    }
  }
  if (e == kNumExperts - 1) p.vinfo[kMaxV] = v0 + nv;
  if (active) {
    AlignedVector<uint32_t, 4> rows;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      const uint32_t below = s_bits[t / 32][ids[j]] & ((1u << (t % 32)) - 1);
      rows[j] = s_row[t / 32][ids[j]] + __popc(below);
    }
    rows.store(p.expanded_idx, tid);
  }
}

// ---- the MoE kernel ----------------------------------------------------------------------------------------------

struct Params {
  const uint8_t* __restrict__ w13_sf;       // [E, 86016]
  const uint8_t* __restrict__ w2_sf;        // [E, 43008]
  const uint8_t* __restrict__ x_sf;         // [T, 112] e8m0
  const uint32_t* __restrict__ token_bits;  // [kMaxV, kMaxWords]
  const uint32_t* __restrict__ vinfo;       // [kMaxV + 1]
  uint8_t* __restrict__ act;                // [T * 16 + kNCap, 384] e4m3, permuted rows
  uint8_t* __restrict__ act_sf;             // [kMaxV * 3, 512] 128x4 atoms per virtual expert
  bf16_t* __restrict__ gemm2_out;           // [T * 16, 3584] permuted rows
  uint32_t* __restrict__ counters;          // [kCounterWords]
};

template <uint32_t kNCap, uint32_t kStages>
struct Smem {
  // A: two boxes of 128 rows x 128 K, unpacked fp4 (FC1: K-atoms 2s, 2s + 1 of one M tile; FC2: K-atom s of two
  // M tiles). B: token boxes of kNCap rows x 128 K (FC1: 2, FC2: 1). Weight / token scale atoms: 2 + 2.
  static constexpr uint32_t kABoxBytes = 128 * 128;
  static constexpr uint32_t kBBoxBytes = kNCap * 128;
  static constexpr uint32_t kSlotBytes = 2 * kABoxBytes + 2 * kBBoxBytes + 4 * kAtomBytes;
  static_assert(kSlotBytes % 1024 == 0);
  static_assert(kNCap <= 32, "token scale atoms use rows < 32");
  struct Info {
    uint16_t tok[kNCap];  // FC1: token of each column
    uint32_t v, tile, e, row_base;
    // tokens of an FC1 / FC2 item, 0 in the other field. A phase's loops stop at a 0 without consuming the slot, so
    // an FC2 item hands its slot to the FC2 loops and no slot carries a mid-way end marker (a marker needs no loads,
    // so MMA and the epilogue could release it before the TMA and scale warps read it). Both 0: no more work.
    uint32_t n_fc1, n_fc2;
  };

  uint8_t slots[kStages][kSlotBytes];
  Info info[kInfoDepth];
  float amax[4][kNCap];  // FC1 epilogue: per warp quadrant, per token
  uint64_t full[kStages], empty[kStages];
  uint64_t tmem_full[2], tmem_empty[2];
  uint64_t info_full[kInfoDepth], info_empty[kInfoDepth];
  uint32_t tmem_base;

  SGL_DEVICE uint8_t* a(uint32_t s) {
    return slots[s];
  }
  SGL_DEVICE uint8_t* b(uint32_t s) {
    return slots[s] + 2 * kABoxBytes;
  }
  SGL_DEVICE uint8_t* sfa(uint32_t s) {
    return slots[s] + 2 * kABoxBytes + 2 * kBBoxBytes;
  }
  SGL_DEVICE uint8_t* sfb(uint32_t s) {
    return sfa(s) + 2 * kAtomBytes;
  }
};

enum Role { WARP_TMA = 0, WARP_MMA = 1, WARP_SCHED = 2, WARP_SF = 3, WARP_EPI_START = 4 };

template <uint32_t kNCap, uint32_t kStages>
__global__ __launch_bounds__(kNumThreads, 1) void moe_kernel(
    const __grid_constant__ CUtensorMap tmap_w13,
    const __grid_constant__ CUtensorMap tmap_w2,
    const __grid_constant__ CUtensorMap tmap_x,
    const __grid_constant__ CUtensorMap tmap_act,
    const Params p) {
  using S = Smem<kNCap, kStages>;
  using Desc = ptx::TCGen05Common;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  // offset from the shared array itself (not through an integer), so smem accesses stay LDS / STS
  S& smem = *reinterpret_cast<S*>(smem_raw + ((1024u - (ptx::to_shared(smem_raw) & 1023u)) & 1023u));
  const uint32_t warp_id = __shfl_sync(warp::kFullMask, threadIdx.x / 32, 0);
  const uint32_t lane_id = get_lane_id();

  // TMEM: 2 accumulator slots of 2 x kNCap columns (FC2 uses both halves); per stage 2 weight + 2 token scale atoms.
  constexpr uint32_t kSfCol = 4 * kNCap;
  constexpr uint32_t kTmemCols = get_tmem_cols(kSfCol + 16 * kStages);

  // prologue: reads nothing the routing kernel writes, so it runs ahead of the PDL wait
  if (warp_id == 3) {
    tcgen05::alloc(&smem.tmem_base, kTmemCols);
    tcgen05::fence_before_thread_sync();
  } else if (warp_id == 1 && lane_id == 0) {
    for (uint32_t s = 0; s < kStages; ++s) {
      ptx::mbar_init(&smem.full[s], 2);  // the producer and the token scale warp
      ptx::mbar_init(&smem.empty[s], 1);
    }
    for (uint32_t i = 0; i < 2; ++i) {
      ptx::mbar_init(&smem.tmem_full[i], 1);
      ptx::mbar_init(&smem.tmem_empty[i], 4);
    }
    for (uint32_t i = 0; i < kInfoDepth; ++i) {
      ptx::mbar_init(&smem.info_full[i], 1);
      ptx::mbar_init(&smem.info_empty[i], 2);  // MMA once the item's loads have landed; the epilogue
    }
    ptx::fence_barrier_init();
  } else if (warp_id == 0 && lane_id == 0) {
    ptx::prefetch_tensormap(&tmap_w13);
    ptx::prefetch_tensormap(&tmap_w2);
    ptx::prefetch_tensormap(&tmap_x);
    ptx::prefetch_tensormap(&tmap_act);
  }
  PDLWaitPrimary<true>();
  __syncthreads();
  tcgen05::fence_after_thread_sync();
  const uint32_t tmem_base = smem.tmem_base;
  const uint32_t num_v = p.vinfo[kMaxV];

  if (warp_id == WARP_TMA) {
    // producer (one lane): weight tiles + weight scale atoms + token tiles per stage
    if (warp::elect_one_lane()) {
      const uint64_t policy = ptx::make_evict_first_policy();
      Ring<kStages> ring;
      Ring<kInfoDepth> iring;
      // FC1 phase
      while (true) {
        ptx::mbar_wait_parity(&smem.info_full[iring.stage], iring.phase);
        const auto& inf = smem.info[iring.stage];
        const uint32_t n_tok = inf.n_fc1;
        if (n_tok == 0) break;
        const uint32_t e = inf.e, tile = inf.tile;
        const uint32_t num_groups = div_ceil(n_tok, 4u);  // gather4 groups of token rows
        const auto y = static_cast<int32_t>(e * 2 * kInter + tile * 128);
        const uint8_t* sfa_src = p.w13_sf + size_t(e) * kW13SfPerExpert + tile * 128 * (kHidden / 32);
        for (uint32_t st = 0; st < kFc1Stages; ++st) {
          ptx::mbar_wait_parity(&smem.empty[ring.stage], ring.phase ^ 1);
          uint64_t* bar = &smem.full[ring.stage];
          const uint32_t a = ptx::to_shared(smem.a(ring.stage));
          const uint32_t b = ptx::to_shared(smem.b(ring.stage));
          ptx::mbar_arrive_expect_tx(bar, 2 * (kWeightBoxTx + kAtomBytes + num_groups * 4 * 128));
#pragma unroll
          for (uint32_t box = 0; box < 2; ++box) {
            ptx::tma_load_2d_hint(
                a + box * S::kABoxBytes, &tmap_w13, static_cast<int32_t>((2 * st + box) * 128), y, bar, policy);
          }
          ptx::bulk_load_hint(smem.sfa(ring.stage), sfa_src + st * 2 * kAtomBytes, 2 * kAtomBytes, bar, policy);
          for (uint32_t g = 0; g < num_groups; ++g) {
            int32_t r[4];
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i) {
              r[i] = inf.tok[min(4 * g + i, n_tok - 1)];
            }
#pragma unroll
            for (uint32_t box = 0; box < 2; ++box) {
              ptx::tma_gather4(
                  b + box * S::kBBoxBytes + g * 4 * 128,
                  &tmap_x,
                  static_cast<int32_t>((2 * st + box) * 128),
                  r[0],
                  r[1],
                  r[2],
                  r[3],
                  bar);
            }
          }
          ring.advance();
        }
        iring.advance();
      }
      // FC2 phase. The intermediate (other CTAs' epilogues) is read through the async proxy: its writers issue
      // fence.proxy.async.global before the release the scheduler acquires.
      while (true) {
        ptx::mbar_wait_parity(&smem.info_full[iring.stage], iring.phase);
        const auto& inf = smem.info[iring.stage];
        if (inf.n_fc2 == 0) break;
        const uint32_t e = inf.e, tile = inf.tile, row_base = inf.row_base;
        iring.advance();
        for (uint32_t st = 0; st < kFc2Stages; ++st) {
          ptx::mbar_wait_parity(&smem.empty[ring.stage], ring.phase ^ 1);
          uint64_t* bar = &smem.full[ring.stage];
          const uint32_t a = ptx::to_shared(smem.a(ring.stage));
          ptx::mbar_arrive_expect_tx(bar, 2 * (kWeightBoxTx + kAtomBytes) + S::kBBoxBytes);
#pragma unroll
          for (uint32_t m = 0; m < 2; ++m) {
            const uint32_t mt = 2 * tile + m;  // M tile of 128 W2 rows
            ptx::tma_load_2d_hint(
                a + m * S::kABoxBytes,
                &tmap_w2,
                static_cast<int32_t>(st * 128),
                static_cast<int32_t>(e * kHidden + mt * 128),
                bar,
                policy);
            const uint8_t* sfa_src = p.w2_sf + size_t(e) * kW2SfPerExpert + mt * 128 * (kInter / 32);
            ptx::bulk_load_hint(
                smem.sfa(ring.stage) + m * kAtomBytes, sfa_src + st * kAtomBytes, kAtomBytes, bar, policy);
          }
          ptx::cp_async_bulk_tensor_2d(
              ptx::to_shared(smem.b(ring.stage)),
              &tmap_act,
              static_cast<int32_t>(st * 128),
              static_cast<int32_t>(row_base),
              bar);
          ring.advance();
        }
      }
    }
  } else if (warp_id == WARP_SCHED) {
    // scheduler: claims items one at a time and publishes them on the info ring (FC1: with the token list)
    Ring<kInfoDepth> iring;
    {
      // FC1 items (v, j), expert-major. The first item of a CTA is its blockIdx, the later ones come from the claim
      // counter (offset by the grid size), claimed one item ahead.
      const uint32_t num_items = kFc1Tiles * num_v;
      uint32_t claim_next = blockIdx.x;
      while (true) {
        const uint32_t i = __shfl_sync(warp::kFullMask, claim_next, 0);
        const uint32_t v = i / kFc1Tiles, tile = i % kFc1Tiles;
        // issued ahead of the bound check, so the first item's loads start right after the PDL wait
        const VInfo info{p.vinfo[min(v, kMaxV - 1)]};
        AlignedVector<uint32_t, kMaxWords> words;
        words.load(p.token_bits, min(v, kMaxV - 1));
        if (i >= num_items) break;
        const uint32_t n = info.num_tokens(), first = info.chunk() * kNCap;
        ptx::mbar_wait_parity(&smem.info_empty[iring.stage], iring.phase ^ 1);
        auto& inf = smem.info[iring.stage];
        if (lane_id == 0) {
          inf.v = v;
          inf.tile = tile;
          inf.e = info.expert();
          inf.row_base = info.row_base();
          inf.n_fc1 = n;
          inf.n_fc2 = 0;
        }
        // token list: lane l tests token 32 w + l of the expert's bitmask; ranks [first, first + n) are this chunk's
        uint32_t seen = 0;
#pragma unroll
        for (uint32_t w = 0; w < kMaxWords; ++w) {
          const uint32_t rank = seen + __popc(words[w] & ((1u << lane_id) - 1)) - first;
          if (((words[w] >> lane_id) & 1) && rank < n) inf.tok[rank] = static_cast<uint16_t>(32 * w + lane_id);
          seen += __popc(words[w]);
        }
        __syncwarp();
        if (lane_id == 0) {
          ptx::mbar_arrive(&smem.info_full[iring.stage]);
          // after the arrive: a release arrive would wait for the atomic's round trip
          claim_next = gridDim.x + atomicAdd(p.counters + kClaim1Word, 1u);
        }
        iring.advance();
      }
    }
    if (warp::elect_one_lane()) {
      // FC2 items (v, j), one lane: groups of 8 virtual experts, tile-major inside a group
      const uint32_t num_items = kFc2Items * num_v;
      const uint32_t num_full = num_v / 8;
      uint32_t claim_next = atomicAdd(p.counters + kClaim2Word, 1u);
      while (true) {
        const uint32_t j = claim_next;
        if (j >= num_items) break;
        claim_next = atomicAdd(p.counters + kClaim2Word, 1u);
        uint32_t v, tile;
        if (j < num_full * 8 * kFc2Items) {
          const uint32_t r = j % (8 * kFc2Items);
          v = (j / (8 * kFc2Items)) * 8 + r % 8;
          tile = r / 8;
        } else {
          const uint32_t r = j - num_full * 8 * kFc2Items, num_rest = num_v % 8;
          v = num_full * 8 + r % num_rest;
          tile = r / num_rest;
        }
        const VInfo info{p.vinfo[v]};
        // this virtual expert's 6 FC1 items (any CTA) must be done
        while (atomic::ptx::load_acquire_u32(p.counters + v) < kFc1Tiles)
          ;
        ptx::mbar_wait_parity(&smem.info_empty[iring.stage], iring.phase ^ 1);
        auto& inf = smem.info[iring.stage];
        inf.v = v;
        inf.tile = tile;
        inf.e = info.expert();
        inf.row_base = info.row_base();
        inf.n_fc1 = 0;
        inf.n_fc2 = info.num_tokens();
        ptx::mbar_arrive(&smem.info_full[iring.stage]);
        iring.advance();
      }
      ptx::mbar_wait_parity(&smem.info_empty[iring.stage], iring.phase ^ 1);
      smem.info[iring.stage].n_fc1 = 0;
      smem.info[iring.stage].n_fc2 = 0;
      ptx::mbar_arrive(&smem.info_full[iring.stage]);
    }
  } else if (warp_id == WARP_SF) {
    // token scales: one 128x4 atom per K-atom (row n = token column n) into each stage's slot; this warp is the
    // second arrival on the stage's full barrier
    Ring<kStages> ring;
    Ring<kInfoDepth> iring;
    // FC1: lane n keeps the 112 scale bytes of token n of the item and stores row n of the stage's two atoms
    uint32_t cur_v = 0xFFFFFFFF;
    AlignedVector<uint32_t, 4> sf[7];
    while (true) {
      ptx::mbar_wait_parity(&smem.info_full[iring.stage], iring.phase);
      const auto& inf = smem.info[iring.stage];
      const uint32_t n_tok = inf.n_fc1, v = inf.v;
      if (n_tok == 0) break;
      if (v != cur_v) {
        // lanes past the item's tokens load a valid row; their columns are not stored
        const uint32_t tok = inf.tok[min(lane_id, n_tok - 1)];
        const uint8_t* src = p.x_sf + size_t(tok) * (kHidden / 32);
#pragma unroll
        for (uint32_t a = 0; a < 7; ++a) {
          sf[a].load(src, a);
        }
        cur_v = v;
      }
      iring.advance();
#pragma unroll
      for (uint32_t st = 0; st < kFc1Stages; ++st) {
        ptx::mbar_wait_parity(&smem.empty[ring.stage], ring.phase ^ 1);
        // K-atoms 2 st, 2 st + 1: scale bytes [8 st, 8 st + 8) of the token row; one 16-byte row per token
        AlignedVector<uint32_t, 4> row0, row1;
        row0.fill(0);
        row1.fill(0);
        row0[0] = sf[st / 2][st % 2 == 0 ? 0 : 2];
        row1[0] = sf[st / 2][st % 2 == 0 ? 1 : 3];
        row0.store(smem.sfb(ring.stage) + lane_id * 16);
        row1.store(smem.sfb(ring.stage) + kAtomBytes + lane_id * 16);
        ptx::fence_proxy_async_shared();
        __syncwarp();
        if (lane_id == 0) ptx::mbar_arrive(&smem.full[ring.stage]);
        ring.advance();
      }
    }
    // FC2 (one lane): the item's atom of the intermediate's scales (written by the FC1 epilogues) per stage
    if (warp::elect_one_lane()) {
      while (true) {
        ptx::mbar_wait_parity(&smem.info_full[iring.stage], iring.phase);
        const auto& inf = smem.info[iring.stage];
        if (inf.n_fc2 == 0) break;
        const uint32_t v = inf.v;
        iring.advance();
        for (uint32_t st = 0; st < kFc2Stages; ++st) {
          ptx::mbar_wait_parity(&smem.empty[ring.stage], ring.phase ^ 1);
          uint64_t* bar = &smem.full[ring.stage];
          ptx::mbar_arrive_expect_tx(bar, kAtomBytes);
          ptx::cp_async_bulk_g2s(smem.sfb(ring.stage), p.act_sf + (size_t(v) * 3 + st) * kAtomBytes, kAtomBytes, bar);
          ring.advance();
        }
      }
    }
  } else if (warp_id == WARP_MMA) {
    // MMA issuer: converged waits, one elected lane per stage issues (a single-lane loop measured slower: its ring
    // state moves to vector registers and every issue needs R2UR)
    Ring<kStages> ring;
    Ring<kInfoDepth> iring;
    Ring<2> acc;
    Desc::MxInstrDesc idesc{};
    idesc.a_format = 5;  // e2m1
    idesc.b_format = 0;  // e4m3
    idesc.n_dim = kNCap >> 3;
    idesc.scale_format = 1;
    idesc.m_dim = 128 >> 4;
    // FC1: per stage 2 K-atoms of one M tile into one accumulator
    while (true) {
      ptx::mbar_wait_parity(&smem.info_full[iring.stage], iring.phase);
      if (smem.info[iring.stage].n_fc1 == 0) break;
      ptx::mbar_wait_parity(&smem.tmem_empty[acc.stage], acc.phase ^ 1);
      tcgen05::fence_after_thread_sync();
      const uint32_t taddr_d = tmem_base + acc.stage * 2 * kNCap;
      for (uint32_t st = 0; st < kFc1Stages; ++st) {
        ptx::mbar_wait_parity(&smem.full[ring.stage], ring.phase);
        tcgen05::fence_after_thread_sync();
        if (warp::elect_one_lane()) {
          const uint32_t sf_col = tmem_base + kSfCol + ring.stage * 16;  // weight atoms [0, 8), token atoms [8, 16)
#pragma unroll
          for (uint32_t i = 0; i < 2; ++i) {
            const auto sfa = Desc::make_desc_sf_chunk(smem.sfa(ring.stage) + i * kAtomBytes);
            const auto sfb = Desc::make_desc_sf_chunk(smem.sfb(ring.stage) + i * kAtomBytes);
            tcgen05::cp_32x128b_warpx4(sf_col + i * 4, sfa.bits);
            tcgen05::cp_32x128b_warpx4(sf_col + 8 + i * 4, sfb.bits);
          }
          const auto a0 = Desc::make_desc_kmajor_sw128(smem.a(ring.stage));
          const auto b0 = Desc::make_desc_kmajor_sw128(smem.b(ring.stage));
#pragma unroll
          for (uint32_t box = 0; box < 2; ++box) {
#pragma unroll
            for (uint32_t kk = 0; kk < 4; ++kk) {
              const auto a = a0.advance_bytes(box * S::kABoxBytes + kk * 32);
              const auto b = b0.advance_bytes(box * S::kBBoxBytes + kk * 32);
              tcgen05::mma_mxf8_ss(
                  taddr_d,
                  a.bits,
                  b.bits,
                  idesc.with_sf_id(kk).bits,
                  sf_col + box * 4,
                  sf_col + 8 + box * 4,
                  (st | box | kk) != 0);
            }
          }
          tcgen05::commit(&smem.empty[ring.stage]);
        }
        ring.advance();
      }
      // every load of the item has landed: its info slot may be reused
      if (lane_id == 0) ptx::mbar_arrive(&smem.info_empty[iring.stage]);
      iring.advance();
      if (warp::elect_one_lane()) tcgen05::commit(&smem.tmem_full[acc.stage]);
      acc.advance();
    }
    // FC2: per stage one K-atom of two M tiles (two accumulator halves) sharing the token box
    while (true) {
      ptx::mbar_wait_parity(&smem.info_full[iring.stage], iring.phase);
      if (smem.info[iring.stage].n_fc2 == 0) break;
      ptx::mbar_wait_parity(&smem.tmem_empty[acc.stage], acc.phase ^ 1);
      tcgen05::fence_after_thread_sync();
      const uint32_t taddr_d = tmem_base + acc.stage * 2 * kNCap;
      for (uint32_t st = 0; st < kFc2Stages; ++st) {
        ptx::mbar_wait_parity(&smem.full[ring.stage], ring.phase);
        tcgen05::fence_after_thread_sync();
        if (warp::elect_one_lane()) {
          const uint32_t sf_col = tmem_base + kSfCol + ring.stage * 16;
#pragma unroll
          for (uint32_t i = 0; i < 2; ++i) {
            const auto sfa = Desc::make_desc_sf_chunk(smem.sfa(ring.stage) + i * kAtomBytes);
            tcgen05::cp_32x128b_warpx4(sf_col + i * 4, sfa.bits);
          }
          tcgen05::cp_32x128b_warpx4(sf_col + 8, Desc::make_desc_sf_chunk(smem.sfb(ring.stage)).bits);
          const auto a0 = Desc::make_desc_kmajor_sw128(smem.a(ring.stage));
          const auto b0 = Desc::make_desc_kmajor_sw128(smem.b(ring.stage));
#pragma unroll
          for (uint32_t m = 0; m < 2; ++m) {
#pragma unroll
            for (uint32_t kk = 0; kk < 4; ++kk) {
              const auto a = a0.advance_bytes(m * S::kABoxBytes + kk * 32);
              const auto b = b0.advance_bytes(kk * 32);
              tcgen05::mma_mxf8_ss(
                  taddr_d + m * kNCap,
                  a.bits,
                  b.bits,
                  idesc.with_sf_id(kk).bits,
                  sf_col + m * 4,
                  sf_col + 8,
                  (st | kk) != 0);
            }
          }
          tcgen05::commit(&smem.empty[ring.stage]);
        }
        ring.advance();
      }
      if (lane_id == 0) ptx::mbar_arrive(&smem.info_empty[iring.stage]);
      iring.advance();
      if (warp::elect_one_lane()) tcgen05::commit(&smem.tmem_full[acc.stage]);
      acc.advance();
    }
  } else if (warp_id >= WARP_EPI_START) {
    // epilogue: warp q = warp_id % 4 reads TMEM lanes [32 q, 32 q + 32)
    const uint32_t q = warp_id % 4;
    Ring<2> acc;
    Ring<kInfoDepth> iring;
    // FC1: SiTU + MXFP8 requant into the intermediate rows and their scale atoms, then publish the item
    while (true) {
      ptx::mbar_wait_parity(&smem.info_full[iring.stage], iring.phase);
      const auto& inf = smem.info[iring.stage];
      const uint32_t v = inf.v, tile = inf.tile, row_base = inf.row_base, n_tok = inf.n_fc1;
      if (n_tok == 0) break;
      ptx::bar_sync(3, 128);  // all four warps have read the slot
      if (q == 0 && lane_id == 0) ptx::mbar_arrive(&smem.info_empty[iring.stage]);
      iring.advance();
      ptx::mbar_wait_parity(&smem.tmem_full[acc.stage], acc.phase);
      tcgen05::fence_after_thread_sync();
      const uint32_t taddr = tmem_base + acc.stage * 2 * kNCap + ((q * 32) << 16);
      const uint32_t num_chunks = div_ceil(n_tok, 8u);
      // the item's token columns into registers in one go, then the accumulator goes back to the MMA warp
      float d[kNCap];
      tcgen05::ld_32x32b<kNCap>(taddr, d);
      tcgen05::wait_ld();
      tcgen05::fence_before_thread_sync();
      if (lane_id == 0) ptx::mbar_arrive(&smem.tmem_empty[acc.stage]);
      // lane l (l / 8 even) holds up of channel 16 q + 2 (l % 8) + l / 16 of this tile; lane l + 8 its gate
      const bool is_up = ((lane_id >> 3) & 1) == 0;
      const uint32_t ch = 16 * q + 2 * (lane_id % 8) + (lane_id >> 4);
      const uint32_t blk = q / 2;  // 32-channel quant block of this tile: warps (0, 1) and (2, 3)
      // pass 1: SiTU (in place, kept for pass 2) and the per-token amax of this warp's 16 channels
#pragma unroll
      for (uint32_t c = 0; c < kNCap / 8; ++c) {
        if (c < num_chunks) {
#pragma unroll
          for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t n = c * 8 + i;
            const float gate = __shfl_down_sync(warp::kFullMask, d[n], 8);
            d[n] = is_up ? situ(gate, d[n]) : 0.0f;
            // |a| as bits orders like the float; one REDUX per token
            const uint32_t m = warp::reduce_max(__float_as_uint(fabsf(d[n])));
            if (lane_id == n) smem.amax[q][n] = __uint_as_float(m);
          }
        }
      }
      ptx::bar_sync(1 + blk, 64);
      uint32_t my_sbyte = 0;  // lane n: scale byte of token n
      if (lane_id < n_tok) {
        my_sbyte = ue8m0_round_up(fmaxf(smem.amax[2 * blk][lane_id], smem.amax[2 * blk + 1][lane_id]));
      }
      ptx::bar_sync(1 + blk, 64);  // amax reads done before the next item overwrites it

      // pass 2: quantize, store
#pragma unroll
      for (uint32_t c = 0; c < kNCap / 8; ++c) {
        if (c < num_chunks) {
#pragma unroll
          for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t n = c * 8 + i;
            const uint32_t sb = __shfl_sync(warp::kFullMask, my_sbyte, n);
            if (is_up && n < n_tok) {
              const float inv = __uint_as_float((254u - sb) << 23);  // 2^-(e - 127)
              p.act[size_t(row_base + n) * kInter + 64 * tile + ch] = __nv_fp8_e4m3(d[n] * inv).__x;
            }
          }
        }
      }
      // token n's scale of K-block kb = 2 tile + blk: atom kb / 4, byte n * 16 + kb % 4
      if ((q & 1) == 0 && lane_id < n_tok) {
        const uint32_t kb = 2 * tile + blk;
        p.act_sf[(size_t(v) * 3 + kb / 4) * kAtomBytes + lane_id * 16 + kb % 4] = static_cast<uint8_t>(my_sbyte);
      }
      // publish: each thread orders its stores before FC2's async-proxy reads, then one release increment after the
      // barrier (cumulative over the 4 warps' stores)
      ptx::fence_proxy_async_global();
      ptx::bar_sync(3, 128);
      if (q == 0 && warp::elect_one_lane()) ptx::red_async_release_add(p.counters + v, 1u);
      acc.advance();
    }
    // FC2: M tile m, lane l of quadrant q holds hidden row 128 (2 tile + m) + 32 q + 4 (l % 8) + l / 8
    while (true) {
      ptx::mbar_wait_parity(&smem.info_full[iring.stage], iring.phase);
      const auto& inf = smem.info[iring.stage];
      const uint32_t tile = inf.tile, row_base = inf.row_base, n_tok = inf.n_fc2;
      ptx::bar_sync(3, 128);
      if (q == 0 && lane_id == 0) ptx::mbar_arrive(&smem.info_empty[iring.stage]);
      iring.advance();
      if (n_tok == 0) break;
      ptx::mbar_wait_parity(&smem.tmem_full[acc.stage], acc.phase);
      tcgen05::fence_after_thread_sync();
      const uint32_t taddr = tmem_base + acc.stage * 2 * kNCap + ((q * 32) << 16);
      const uint32_t num_chunks = div_ceil(n_tok, 8u);
      float d[2][kNCap];
      tcgen05::ld_32x32b<2 * kNCap>(taddr, d);  // the two M tiles' adjacent column ranges
      tcgen05::wait_ld();
      tcgen05::fence_before_thread_sync();
      if (lane_id == 0) ptx::mbar_arrive(&smem.tmem_empty[acc.stage]);
#pragma unroll
      for (uint32_t m = 0; m < 2; ++m) {
        const uint32_t o = 128 * (2 * tile + m) + 32 * q + 4 * (lane_id % 8) + lane_id / 8;
#pragma unroll
        for (uint32_t c = 0; c < kNCap / 8; ++c) {
          if (c < num_chunks) {
#pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
              const uint32_t n = c * 8 + i;
              if (n < n_tok) p.gemm2_out[size_t(row_base + n) * kHidden + o] = __float2bfloat16_rn(d[m][n]);
            }
          }
        }
      }
      acc.advance();
    }
  }

  // teardown: the next kernel's PDL wait covers this grid's completion, so the trigger only lets it launch
  PDLTriggerSecondary<true>();
  __syncthreads();
  if (warp_id == 3) {
    tcgen05::fence_after_thread_sync();
    tcgen05::dealloc(tmem_base, kTmemCols);
  }
}

}  // namespace device::k31_moe

// Routing tables of the precomputed top-k (two CTAs). PDL secondary of the top-k producer, primary of the MoE kernel.
// With `topk_weights`, `topk_ids` is the packed (id << 16) | bf16(weight) form and the weights are written there.
template <uint32_t kNCap>
void k31_moe_route(
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView expanded_idx,
    tvm::ffi::TensorView token_bits,
    tvm::ffi::TensorView vinfo,
    tvm::ffi::TensorView counters,
    tvm::ffi::Optional<tvm::ffi::TensorView> topk_weights) {
  using namespace host;
  using namespace device::k31_moe;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({-1, kTopK})  // topk_ids
      .with_dtype<int32_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(topk_ids);
  TensorMatcher({-1})  // expanded_idx
      .with_dtype<int32_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(expanded_idx);
  TensorMatcher({kMaxV * kMaxWords})  // token_bits
      .with_dtype<int32_t>()
      .with_device(device)
      .ensure_alignment(32)
      .verify(token_bits);
  TensorMatcher({kMaxV + 1})  // vinfo
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(vinfo);
  TensorMatcher({kCounterWords})  // counters
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(counters);
  const auto num_tokens = static_cast<uint32_t>(topk_ids.size(0));
  CHECK_HOST(num_tokens >= 1 && num_tokens <= kMaxT) << "num_tokens = " << num_tokens;
  CHECK_HOST(expanded_idx.size(0) >= num_tokens * kTopK) << "expanded_idx holds " << expanded_idx.size(0) << " rows";
  const bool packed = topk_weights.has_value();
  if (packed) {
    TensorMatcher({static_cast<int64_t>(num_tokens), kTopK})  // topk_weights
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(8)
        .verify(topk_weights.value());
  }
  const RouteParams params{
      .topk_ids = static_cast<const int32_t*>(topk_ids.data_ptr()),
      .topk_weights = packed ? static_cast<uint16_t*>(topk_weights.value().data_ptr()) : nullptr,
      .expanded_idx = static_cast<int32_t*>(expanded_idx.data_ptr()),
      .token_bits = static_cast<uint32_t*>(token_bits.data_ptr()),
      .vinfo = static_cast<uint32_t*>(vinfo.data_ptr()),
      .counters = static_cast<uint32_t*>(counters.data_ptr()),
      .num_tokens = num_tokens,
  };
  const auto kernel = packed ? route_kernel<kNCap, true> : route_kernel<kNCap, false>;
  LaunchKernel(2, kRouteThreads, device.unwrap()).enable_pdl(true)(kernel, params);
}

template <uint32_t kNCap, uint32_t kStages>
void k31_moe_mxfp4(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView x_sf,
    tvm::ffi::TensorView w13,
    tvm::ffi::TensorView w13_sf,
    tvm::ffi::TensorView w2,
    tvm::ffi::TensorView w2_sf,
    tvm::ffi::TensorView token_bits,
    tvm::ffi::TensorView vinfo,
    tvm::ffi::TensorView act,
    tvm::ffi::TensorView act_sf,
    tvm::ffi::TensorView gemm2_out,
    tvm::ffi::TensorView counters) {
  using namespace host;
  using namespace device::k31_moe;
  auto T = SymbolicSize{"num_tokens"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({T, kHidden})  // x (e4m3 bytes)
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(x);
  TensorMatcher({T, kHidden / 32})  // x_sf
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(x_sf);
  TensorMatcher({kNumExperts, 2 * kInter, kHidden / 2})  // w13
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(w13);
  TensorMatcher({kNumExperts, 2 * kInter, kHidden / 32})  // w13_sf
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(w13_sf);
  TensorMatcher({kNumExperts, kHidden, kInter / 2})  // w2
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(w2);
  TensorMatcher({kNumExperts, kHidden, kInter / 32})  // w2_sf: rows of 12 bytes, read as 128x4 atoms per expert
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(w2_sf);
  TensorMatcher({kMaxV * kMaxWords})  // token_bits
      .with_dtype<int32_t>()
      .with_device(device)
      .ensure_alignment(32)
      .verify(token_bits);
  TensorMatcher({kMaxV + 1})  // vinfo
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(vinfo);
  TensorMatcher({-1, kInter})  // act
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(act);
  TensorMatcher({kMaxV * 3 * kAtomBytes})  // act_sf
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(act_sf);
  TensorMatcher({-1, kHidden})  // gemm2_out
      .with_dtype<bf16_t>()
      .with_device(device)
      .verify(gemm2_out);
  TensorMatcher({kCounterWords})  // counters
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(counters);
  // w2_sf rows are 12 bytes, so only its base can be 16-byte aligned
  CHECK_HOST(reinterpret_cast<uintptr_t>(w2_sf.data_ptr()) % 16 == 0) << "w2_sf is not 16-byte aligned";
  const auto num_tokens = static_cast<uint32_t>(T.unwrap());
  CHECK_HOST(num_tokens >= 1 && num_tokens <= kMaxT) << "num_tokens = " << num_tokens;
  CHECK_HOST(act.size(0) >= num_tokens * kTopK + kNCap) << "act holds " << act.size(0) << " rows";
  CHECK_HOST(gemm2_out.size(0) >= num_tokens * kTopK) << "gemm2_out holds " << gemm2_out.size(0) << " rows";
  constexpr uint32_t kSmem = sizeof(Smem<kNCap, kStages>) + 1024;  // + the runtime 1024-byte alignment
  static_assert(kSmem <= 227 * 1024, "smem");
  constexpr auto kernel = moe_kernel<kNCap, kStages>;
  const DLDevice dev = device.unwrap();
  runtime::set_max_dynamic_smem_per_device<kernel>(dev, kSmem);
  // persistent: SMs x occupancy (1). Any grid size is correct: work is claimed dynamically, and an FC2 item only
  // waits for FC1 items that a running CTA has claimed
  const uint32_t grid = runtime::get_sm_count(dev.device_id) * runtime::get_blocks_per_sm(kernel, kNumThreads, kSmem);

  const auto map_w13 = tma::make_map(
      w13.data_ptr(),
      CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
      {kHidden, uint64_t(kNumExperts) * 2 * kInter},
      {kHidden / 2},
      {128, 128},
      {.swizzle = CU_TENSOR_MAP_SWIZZLE_128B});
  const auto map_w2 = tma::make_map(
      w2.data_ptr(),
      CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
      {kInter, uint64_t(kNumExperts) * kHidden},
      {kInter / 2},
      {128, 128},
      {.swizzle = CU_TENSOR_MAP_SWIZZLE_128B});
  const auto map_x = tma::make_map(
      x.data_ptr(),
      CU_TENSOR_MAP_DATA_TYPE_UINT8,
      {kHidden, num_tokens},
      {kHidden},
      {128, 1},
      {.swizzle = CU_TENSOR_MAP_SWIZZLE_128B});
  const auto map_act = tma::make_map(
      act.data_ptr(),
      CU_TENSOR_MAP_DATA_TYPE_UINT8,
      {kInter, static_cast<uint64_t>(act.size(0))},
      {kInter},
      {128, kNCap},
      {.swizzle = CU_TENSOR_MAP_SWIZZLE_128B});
  const Params params{
      .w13_sf = static_cast<const uint8_t*>(w13_sf.data_ptr()),
      .w2_sf = static_cast<const uint8_t*>(w2_sf.data_ptr()),
      .x_sf = static_cast<const uint8_t*>(x_sf.data_ptr()),
      .token_bits = static_cast<const uint32_t*>(token_bits.data_ptr()),
      .vinfo = static_cast<const uint32_t*>(vinfo.data_ptr()),
      .act = static_cast<uint8_t*>(act.data_ptr()),
      .act_sf = static_cast<uint8_t*>(act_sf.data_ptr()),
      .gemm2_out = static_cast<bf16_t*>(gemm2_out.data_ptr()),
      .counters = static_cast<uint32_t*>(counters.data_ptr()),
  };
  LaunchKernel(grid, kNumThreads, dev, kSmem).enable_pdl(true)(kernel, map_w13, map_w2, map_x, map_act, params);
}

}  // namespace sglang
