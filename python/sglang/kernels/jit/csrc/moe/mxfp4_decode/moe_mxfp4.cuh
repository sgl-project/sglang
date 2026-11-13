// Decode routed MoE (TP-MoE rank, or one EP rank's local experts), MXFP4 weights x MXFP8 activations, SiTU or
// clamped SwiGLU, on SM100 tcgen05. Two kernels chained with PDL, launched by one host call:
//   route_kernel (2 CTAs): CTA 0 builds the permutation tables (expert histogram + scan of the precomputed top-k),
//       CTA 1 resets the MoE kernel's counters;
//   moe_kernel (persistent): FC1 + activation + MXFP8 requant, then FC2, writing the engine's deferred-finalize rows
//       (gemm2_out in a permuted row order + expanded_idx); the engine applies the top-k weights later.
//
// Shape (template): E local experts, top-k, hidden H, intermediate shard I, kBlockN tokens per virtual expert.
// Weights are consumed in the layout the engine prepares for flashinfer's TRT-LLM-gen MXFP4 MoE (no repack):
//   W13 [E, 2 I, H / 2] u8: row 2i = up_i, 2i+1 = gate_i, then within each 32-row block stored row s = source row
//       4 (s % 8) + s / 8. W2 [E, H, I / 2] u8: the same 32-row shuffle. Scales: 128x4 atoms (512 B: row m of a
//       128-row tile, scale column k: (m % 32) * 16 + (m / 32) * 4 + k % 4), tiles row-major over (m / 128, k / 4).
//
// Work: virtual experts v = (expert e, token chunk c of <= kBlockN tokens). FC1 item (v, j): 128 W13 rows (64
// channels) x K H (H / 256 stages of K 256); FC2 item (v, j): two 128-row W2 tiles sharing the token tile x K I
// (I / 128 stages of K 128). Every CTA claims FC1 items from one global queue until it is empty, then FC2 items from
// a second one; each role runs an FC1 loop, then an FC2 loop, on one continuous stage ring. FC2 of v waits for v's
// 2 I / 128 FC1 items (per-v counter, release / acquire), any CTA's.
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

#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace sglang {

namespace device::mxfp4_moe {

inline constexpr uint32_t kMaxM = 256;                   // tokens per call
inline constexpr uint32_t kMaxWords = kMaxM / 32;        // 32-token words of a token bitmask
inline constexpr uint32_t kAtomBytes = 512;              // one 128x4 scale atom
inline constexpr uint32_t kWeightBoxTx = 128 * 128 / 2;  // a 128 x 128 fp4 box: tx counts the packed global bytes
inline constexpr uint32_t kInfoDepth = 4;  // item info ring; also bounds the claim prefetch (see moe_kernel)
inline constexpr uint32_t kNumThreads = 256;
inline constexpr uint32_t kRouteThreads = 1024;
inline constexpr uint32_t kSmemBudget = 227 * 1024;  // dynamic smem per CTA

template <uint32_t kNumExperts_, uint32_t kTopK_, uint32_t kHidden_, uint32_t kInter_, uint32_t kBlockN_>
struct Shape {
  static constexpr uint32_t kNumExperts = kNumExperts_;  // local experts
  static constexpr uint32_t kTopK = kTopK_;
  static constexpr uint32_t kHidden = kHidden_;
  static constexpr uint32_t kInter = kInter_;    // intermediate shard (after padding to 128)
  static constexpr uint32_t kBlockN = kBlockN_;  // tokens per virtual expert = MMA N
  static constexpr uint32_t kMaxRows = kMaxM * kTopK;
  // virtual experts: sum over touched experts of ceil(n_e / kBlockN) <= E + T * top-k / kBlockN
  static constexpr uint32_t kMaxV = kNumExperts + kMaxRows / kBlockN;
  // counters (zeroed by the routing kernel): [0, kMaxV) FC1 items done per virtual expert, then the two claim cursors
  static constexpr uint32_t kClaim1Word = kMaxV;
  static constexpr uint32_t kClaim2Word = kMaxV + 1;
  static constexpr uint32_t kCounterWords = kMaxV + 2;

  static constexpr uint32_t kFc1Tiles = 2 * kInter / 128;  // FC1 items per v: 128 W13 rows = 64 channels each
  static constexpr uint32_t kFc2Items = kHidden / 256;     // FC2 items per v: two 128-row M tiles each
  static constexpr uint32_t kFc1Stages = kHidden / 256;    // K 256 per stage (2 K-atoms of 128)
  static constexpr uint32_t kFc2Stages = kInter / 128;     // K 128 per stage, 2 M tiles
  static constexpr uint32_t kActSfAtoms = kInter / 128;    // intermediate scale atoms per v (one per K-atom)
  static constexpr uint32_t kTokenSfVecs = kHidden / 512;  // 16-byte vectors of a token's H / 32 scale bytes
  static constexpr uint32_t kW13SfPerExpert = 2 * kInter * (kHidden / 32);
  static constexpr uint32_t kW2SfPerExpert = kHidden * (kInter / 32);

  static_assert(kHidden % 512 == 0, "FC1 / FC2 K 256 stages, token scale rows in 16-byte vectors");
  static_assert(kInter % 128 == 0, "FC2 K-atoms of 128, intermediate scale atoms of 4 K-blocks");
  static_assert(kBlockN == 16 || kBlockN == 32, "MMA N % 16 at M 128; token scale atom rows < 32 (one per lane)");
  static_assert(kNumExperts % 4 == 0, "the routing kernel zeroes its bitmask in 16-byte vectors");
  static_assert(kNumExperts <= kRouteThreads, "routing: one thread per expert");
  static_assert(kMaxRows <= 4 * kRouteThreads, "routing: four ids per thread");
  // VInfo fields: expert 10 bits, first row 12 bits, tokens 6 bits, chunk 4 bits
  static_assert(kNumExperts <= 1024 && kMaxRows <= 4096 && kBlockN < 64 && kMaxM / kBlockN <= 16);
  static_assert(kMaxV < (1u << 16), "routing scan packs (virtual experts, rows) into 16-bit halves");

  // workspace (one byte tensor, carved here): counters, vinfo, token bitmasks, intermediate scales, intermediate
  static constexpr uint64_t align_up(uint64_t x) {
    return (x + 255) / 256 * 256;
  }
  static constexpr uint32_t kActRows = kMaxRows + kBlockN;  // the last virtual expert's token box may read past
  static constexpr uint64_t kCountersOffset = 0;
  static constexpr uint64_t kVInfoOffset = align_up(kCountersOffset + kCounterWords * 4);
  static constexpr uint64_t kTokenBitsOffset = align_up(kVInfoOffset + (kMaxV + 1) * 4);
  static constexpr uint64_t kActSfOffset = align_up(kTokenBitsOffset + uint64_t(kMaxV) * kMaxWords * 4);
  static constexpr uint64_t kActOffset = align_up(kActSfOffset + uint64_t(kMaxV) * kActSfAtoms * kAtomBytes);
  static constexpr uint64_t kWorkspaceBytes = align_up(kActOffset + uint64_t(kActRows) * kInter);
};

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

// vinfo[v] of virtual expert v (tokens <= 32, first row < 4096, chunk < 16); vinfo[kMaxV] is their count.
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
  const int32_t* __restrict__ topk_ids;  // [T, top-k]: global expert ids, or (id << 16) | bf16(weight) when kPacked
  uint16_t* __restrict__ topk_weights;   // kPacked only: [T, top-k] bf16, the weights of the packed form
  int32_t* __restrict__ expanded_idx;    // [T * top-k] -> permuted row, -1 for an id that is not a local expert
  uint32_t* __restrict__ token_bits;     // [kMaxV, kMaxWords]: v's expert's bitmask, bit t % 32 of word t / 32
  uint32_t* __restrict__ vinfo;          // [kMaxV + 1]
  uint32_t* __restrict__ counters;       // [kCounterWords], zeroed here for the MoE kernel
  uint32_t num_tokens;
  uint32_t expert_offset;  // global id of local expert 0
};

// The rows of an expert are its tokens in token order: the rank of token t in expert e is the number of set bits
// below t in e's token bitmask. The histogram builds the bitmask with atomicOr, which is order-independent, so the
// tables are deterministic. Ids outside [expert_offset, expert_offset + E) (other EP ranks, -1) get no row.
// kWords: 32-token words of the bitmask this call needs (ceil(T / 32) rounded up to a power of two); the routing
// kernel is issue-bound on its one SM, so small T skips the unused words
template <class S, bool kPacked, uint32_t kWords>
__global__ void __launch_bounds__(kRouteThreads, 1) route_kernel(const RouteParams p) {
  constexpr uint32_t kNumExperts = S::kNumExperts, kTopK = S::kTopK, kBlockN = S::kBlockN;
  static_assert(kWords <= kMaxWords && (kWords & (kWords - 1)) == 0);
  static_assert(kNumExperts * kWords * 6 + 32 * 4 <= 48 * 1024, "static smem");
  static_assert(kNumExperts >= 32, "invalid ids OR 0 into word lane_id of the first row");
  __shared__ alignas(16) uint32_t s_bits[kWords][kNumExperts];  // word-major: conflict-free per expert
  __shared__ uint16_t s_row[kWords][kNumExperts];               // first row of the tokens of word w of expert e
  __shared__ uint32_t s_scan[32];
  const uint32_t tid = threadIdx.x;
  if (blockIdx.x == 1) {
    // CTA 1 only resets the MoE kernel's counters, once the previous kernels of the stream have completed
    PDLWaitPrimary<true>();
    for (uint32_t i = tid; i < S::kCounterWords; i += kRouteThreads) {
      p.counters[i] = 0;
    }
    return;
  }
  // early trigger: the MoE kernel's PDL wait covers this grid's completion, so the trigger only lets it launch and
  // run its prologue (TMEM alloc, barrier init) while the tables are built
  PDLTriggerSecondary<true>();
  const uint32_t lane_id = tid % 32, warp_id = tid / 32;
  AlignedVector<uint32_t, 4> zero;
  zero.fill(0);
  for (uint32_t i = tid; i < kWords * kNumExperts / 4; i += kRouteThreads) {
    zero.store(s_bits, i);
  }
  PDLWaitPrimary<true>();
  // thread i: ids [4 i, 4 i + 4) of the flat [T, top-k] tensor; id j belongs to token (4 i + j) / top-k
  const uint32_t num_ids = p.num_tokens * kTopK;
  uint32_t t[4];
  bool in[4];
  AlignedVector<uint32_t, 4> ids;
  ids.fill(0);
  if constexpr (kTopK % 4 == 0) {
    // the four ids share one token, and T * top-k has no partial chunk
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      t[j] = tid / (kTopK / 4);
      in[j] = t[j] < p.num_tokens;
    }
    if (in[0]) ids.load(p.topk_ids, tid);
  } else {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      t[j] = (4 * tid + j) / kTopK;
      in[j] = 4 * tid + j < num_ids;
    }
    if (in[3]) {
      ids.load(p.topk_ids, tid);
    } else {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        if (in[j]) ids[j] = static_cast<uint32_t>(p.topk_ids[4 * tid + j]);
      }
    }
  }
  if constexpr (kPacked) {
    // the packed form of the route + quant handoff: keep the ids, pass the weights on to the engine's finalize
    AlignedVector<uint16_t, 4> weights;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      weights[j] = static_cast<uint16_t>(ids[j] & 0xFFFF);
      ids[j] >>= 16;
    }
    if (in[3]) {
      weights.store(p.topk_weights, tid);
    } else {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        if (in[j]) p.topk_weights[4 * tid + j] = weights[j];
      }
    }
  }
  // local expert ids; one unsigned compare rejects other ranks' experts and -1 (packed: 0xFFFF)
  bool valid[4];
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    ids[j] -= p.expert_offset;
    valid[j] = in[j] && ids[j] < kNumExperts;
  }
  __syncthreads();
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    // ptxas branches around a predicated ATOMS (`@p red.shared` too), so an invalid id ORs 0 into a lane-distinct
    // word instead: one branch-free ATOMS per id, no same-address serialization among the invalid lanes
    uint32_t* const word = valid[j] ? &s_bits[t[j] / 32][ids[j]] : &s_bits[0][lane_id];
    atomicOr(word, valid[j] ? 1u << (t[j] % 32) : 0u);
  }
  __syncthreads();
  // thread e: its bitmask, row count and virtual expert count; exclusive scan of both over the experts
  const uint32_t e = tid;
  AlignedVector<uint32_t, kMaxWords> bits;  // stored full width: the scheduler reads kMaxWords words
  bits.fill(0);
  uint32_t cnt = 0;
#pragma unroll
  for (uint32_t w = 0; w < kWords; ++w) {
    bits[w] = e < kNumExperts ? s_bits[w][e] : 0u;
    cnt += __popc(bits[w]);
  }
  const uint32_t nv = div_ceil(cnt, kBlockN);
  const uint32_t packed = nv | (cnt << 16);  // sums: nv <= kMaxV, cnt <= kMaxRows
  const uint32_t incl = warp::inclusive_sum(packed, lane_id);
  if (lane_id == 31) s_scan[warp_id] = incl;
  __syncthreads();
  const uint32_t excl = warp::reduce_sum(lane_id < warp_id ? s_scan[lane_id] : 0u) + incl - packed;
  const uint32_t v0 = excl & 0xFFFF, r0 = excl >> 16;
  if (e < kNumExperts) {
    uint32_t r = r0;
#pragma unroll
    for (uint32_t w = 0; w < kWords; ++w) {
      s_row[w][e] = static_cast<uint16_t>(r);
      r += __popc(bits[w]);
    }
  }
  __syncthreads();
  if (e < kNumExperts) {
    for (uint32_t c = 0; c < nv; ++c) {
      p.vinfo[v0 + c] = VInfo::pack(e, r0 + c * kBlockN, min(kBlockN, cnt - c * kBlockN), c);
      bits.store(p.token_bits, v0 + c);
    }
  }
  if (e == kNumExperts - 1) p.vinfo[S::kMaxV] = v0 + nv;
  AlignedVector<uint32_t, 4> rows;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    // in-bounds operands for an invalid id (no branch), its row is replaced by -1
    const uint32_t w = valid[j] ? t[j] / 32 : 0u, id = valid[j] ? ids[j] : 0u;
    const uint32_t below = s_bits[w][id] & ((1u << (t[j] % 32)) - 1);
    rows[j] = valid[j] ? s_row[w][id] + __popc(below) : ~0u;
  }
  if (in[3]) {
    rows.store(p.expanded_idx, tid);
  } else if constexpr (kTopK % 4 != 0) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      if (in[j]) p.expanded_idx[4 * tid + j] = static_cast<int32_t>(rows[j]);
    }
  }
}

// ---- the MoE kernel ----------------------------------------------------------------------------------------------

struct Params {
  const uint8_t* __restrict__ w13_sf;       // [E, 2 I * H / 32]
  const uint8_t* __restrict__ w2_sf;        // [E, H * I / 32]
  const uint8_t* __restrict__ x_sf;         // [T, H / 32] e8m0
  const uint32_t* __restrict__ token_bits;  // [kMaxV, kMaxWords]
  const uint32_t* __restrict__ vinfo;       // [kMaxV + 1]
  uint8_t* __restrict__ act;                // [kActRows, I] e4m3, permuted rows
  uint8_t* __restrict__ act_sf;             // [kMaxV * I / 128, 512] 128x4 atoms per virtual expert
  bf16_t* __restrict__ gemm2_out;           // [T * top-k, H] permuted rows
  uint32_t* __restrict__ counters;          // [kCounterWords]
};

template <uint32_t kBlockN, uint32_t kStages>
struct Smem {
  // A: two boxes of 128 rows x 128 K, unpacked fp4 (FC1: K-atoms 2s, 2s + 1 of one M tile; FC2: K-atom s of two
  // M tiles). B: token boxes of kBlockN rows x 128 K (FC1: 2, FC2: 1). Weight / token scale atoms: 2 + 2.
  static constexpr uint32_t kABoxBytes = 128 * 128;
  static constexpr uint32_t kBBoxBytes = kBlockN * 128;
  static constexpr uint32_t kSlotBytes = 2 * kABoxBytes + 2 * kBBoxBytes + 4 * kAtomBytes;
  static_assert(kSlotBytes % 1024 == 0);
  struct Info {
    uint16_t tok[kBlockN];  // FC1: token of each column
    uint32_t v, tile, e, row_base;
    // tokens of an FC1 / FC2 item, 0 in the other field. A phase's loops stop at a 0 without consuming the slot, so
    // an FC2 item hands its slot to the FC2 loops and no slot carries a mid-way end marker (a marker needs no loads,
    // so MMA and the epilogue could release it before the TMA and scale warps read it). Both 0: no more work.
    uint32_t n_fc1, n_fc2;
  };

  uint8_t slots[kStages][kSlotBytes];
  Info info[kInfoDepth];
  float amax[4][kBlockN];  // FC1 epilogue: per warp quadrant, per token
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

// smem ring depth: as many stages as fit next to the fixed part (+ the runtime 1024-byte alignment)
template <uint32_t kBlockN>
constexpr uint32_t num_stages() {
  using S1 = Smem<kBlockN, 1>;
  constexpr uint32_t kPerStage = S1::kSlotBytes + 2 * sizeof(uint64_t);  // slot + full / empty barriers
  constexpr uint32_t kStages = (kSmemBudget - 1024 - (sizeof(S1) - kPerStage)) / kPerStage;
  static_assert(sizeof(Smem<kBlockN, kStages>) + 1024 <= kSmemBudget);
  static_assert(sizeof(Smem<kBlockN, kStages + 1>) + 1024 > kSmemBudget);
  return kStages;
}

enum Role { WARP_TMA = 0, WARP_MMA = 1, WARP_SCHED = 2, WARP_SF = 3, WARP_EPI_START = 4 };
// named barriers of the epilogue warps (0 is __syncthreads): one per quant block pair, one for all four
enum Barrier { BAR_EPI_PAIR = 1, BAR_EPI_ALL = 3 };
static_assert(kNumThreads == 32 * (WARP_EPI_START + 4), "four epilogue warps, one per TMEM lane quadrant");

// kFc1Prefetch / kFc2Prefetch: items a CTA may hold in that phase, claimed and not yet released (the one in flight
// included). A claim waits for the release of the item that would exceed it. FC1: 1..kInfoDepth + 1, FC2 (it claims
// one item ahead of publishing): 2..kInfoDepth + 2; the largest value is the info ring's own bound (no gate). The first
// FC2 claim is not gated.
template <class Sh, uint32_t kFc1Prefetch, uint32_t kFc2Prefetch, auto kAct>
__global__ __launch_bounds__(kNumThreads, 1) void moe_kernel(
    const __grid_constant__ CUtensorMap tmap_w13,
    const __grid_constant__ CUtensorMap tmap_w2,
    const __grid_constant__ CUtensorMap tmap_x,
    const __grid_constant__ CUtensorMap tmap_act,
    const Params p) {
  constexpr uint32_t kHidden = Sh::kHidden, kInter = Sh::kInter;
  constexpr uint32_t kBlockN = Sh::kBlockN, kMaxV = Sh::kMaxV;
  constexpr uint32_t kFc1Tiles = Sh::kFc1Tiles, kFc2Items = Sh::kFc2Items;
  constexpr uint32_t kFc1Stages = Sh::kFc1Stages, kFc2Stages = Sh::kFc2Stages;
  constexpr uint32_t kStages = num_stages<kBlockN>();
  static_assert(kStages >= 2);
  using S = Smem<kBlockN, kStages>;
  using Desc = ptx::TCGen05Common;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  // offset from the shared array itself (not through an integer), so smem accesses stay LDS / STS
  S& smem = *reinterpret_cast<S*>(smem_raw + ((1024u - (ptx::to_shared(smem_raw) & 1023u)) & 1023u));
  const uint32_t warp_id = __shfl_sync(warp::kFullMask, threadIdx.x / 32, 0);
  const uint32_t lane_id = get_lane_id();

  // TMEM: 2 accumulator slots of 2 x kBlockN columns (FC2 uses both halves); per stage 2 weight + 2 token scale atoms.
  constexpr uint32_t kSfCol = 4 * kBlockN;
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
        const uint8_t* sfa_src = p.w13_sf + size_t(e) * Sh::kW13SfPerExpert + tile * 128 * (kHidden / 32);
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
            const uint8_t* sfa_src = p.w2_sf + size_t(e) * Sh::kW2SfPerExpert + mt * 128 * (kInter / 32);
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
    static_assert(1 <= kFc1Prefetch && kFc1Prefetch <= kInfoDepth + 1);
    static_assert(2 <= kFc2Prefetch && kFc2Prefetch <= kInfoDepth + 2);
    constexpr bool kFc1Gated = kFc1Prefetch >= 2 && kFc1Prefetch <= kInfoDepth;
    constexpr bool kFc2Gated = kFc2Prefetch <= kInfoDepth + 1;
    Ring<kInfoDepth> iring;
    // ring positions of the latest published items, newest first (the claim gates wait on their release)
    constexpr uint32_t kHist = std::max({kFc1Prefetch - 1, kFc2Gated ? kFc2Prefetch - 1 : 0u, 1u});
    Ring<kInfoDepth> hist[kHist];
    bool has_hist[kHist] = {};
    const auto remember = [&](const Ring<kInfoDepth>& r) {
#pragma unroll
      for (uint32_t h = kHist - 1; h > 0; --h) {
        hist[h] = hist[h - 1];
        has_hist[h] = has_hist[h - 1];
      }
      hist[0] = r;
      has_hist[0] = true;
    };
    {
      // FC1 items (v, j), expert-major. The first item of a CTA is its blockIdx, the later ones come from the claim
      // counter (offset by the grid size). A small prefetch keeps early CTAs from hoarding the queue at small T.
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
        const uint32_t n = info.num_tokens(), first = info.chunk() * kBlockN;
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
          if constexpr (kFc1Prefetch == 1) {
            ptx::mbar_wait_parity(&smem.info_empty[iring.stage], iring.phase);
          } else if constexpr (kFc1Gated) {
            const auto& h = hist[kFc1Prefetch - 2];
            if (has_hist[kFc1Prefetch - 2]) ptx::mbar_wait_parity(&smem.info_empty[h.stage], h.phase);
          }
          // after the arrive: a release arrive would wait for the atomic's round trip
          claim_next = gridDim.x + atomicAdd(p.counters + Sh::kClaim1Word, 1u);
        }
        remember(iring);
        iring.advance();
      }
    }
    if (warp::elect_one_lane()) {
      // FC2 items (v, j), one lane: groups of 8 virtual experts, tile-major inside a group
      const uint32_t num_items = kFc2Items * num_v;
      const uint32_t num_full = num_v / 8;
      uint32_t claim_next = atomicAdd(p.counters + Sh::kClaim2Word, 1u);
      while (true) {
        const uint32_t j = claim_next;
        if (j >= num_items) break;
        if constexpr (kFc2Gated) {
          const auto& h = hist[kFc2Prefetch - 2];
          if (has_hist[kFc2Prefetch - 2]) ptx::mbar_wait_parity(&smem.info_empty[h.stage], h.phase);
        }
        claim_next = atomicAdd(p.counters + Sh::kClaim2Word, 1u);
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
        // this virtual expert's FC1 items (any CTA) must be done
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
        if constexpr (kFc2Gated) remember(iring);
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
    AlignedVector<uint32_t, 4> sf[Sh::kTokenSfVecs];
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
        for (uint32_t a = 0; a < Sh::kTokenSfVecs; ++a) {
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
          ptx::cp_async_bulk_g2s(
              smem.sfb(ring.stage), p.act_sf + (size_t(v) * Sh::kActSfAtoms + st) * kAtomBytes, kAtomBytes, bar);
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
    idesc.n_dim = kBlockN >> 3;
    idesc.scale_format = 1;
    idesc.m_dim = 128 >> 4;
    // FC1: per stage 2 K-atoms of one M tile into one accumulator
    while (true) {
      ptx::mbar_wait_parity(&smem.info_full[iring.stage], iring.phase);
      if (smem.info[iring.stage].n_fc1 == 0) break;
      ptx::mbar_wait_parity(&smem.tmem_empty[acc.stage], acc.phase ^ 1);
      tcgen05::fence_after_thread_sync();
      const uint32_t taddr_d = tmem_base + acc.stage * 2 * kBlockN;
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
      const uint32_t taddr_d = tmem_base + acc.stage * 2 * kBlockN;
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
                  taddr_d + m * kBlockN,
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
    // FC1: activation + MXFP8 requant into the intermediate rows and their scale atoms, then publish the item
    while (true) {
      ptx::mbar_wait_parity(&smem.info_full[iring.stage], iring.phase);
      const auto& inf = smem.info[iring.stage];
      const uint32_t v = inf.v, tile = inf.tile, row_base = inf.row_base, n_tok = inf.n_fc1;
      if (n_tok == 0) break;
      ptx::bar_sync(BAR_EPI_ALL, 128);  // all four warps have read the slot
      if (q == 0 && lane_id == 0) ptx::mbar_arrive(&smem.info_empty[iring.stage]);
      iring.advance();
      ptx::mbar_wait_parity(&smem.tmem_full[acc.stage], acc.phase);
      tcgen05::fence_after_thread_sync();
      const uint32_t taddr = tmem_base + acc.stage * 2 * kBlockN + ((q * 32) << 16);
      const uint32_t num_chunks = div_ceil(n_tok, 8u);
      // the item's token columns into registers in one go, then the accumulator goes back to the MMA warp
      float d[kBlockN];
      tcgen05::ld_32x32b<kBlockN>(taddr, d);
      tcgen05::wait_ld();
      tcgen05::fence_before_thread_sync();
      if (lane_id == 0) ptx::mbar_arrive(&smem.tmem_empty[acc.stage]);
      // lane l (l / 8 even) holds up of channel 16 q + 2 (l % 8) + l / 16 of this tile; lane l + 8 its gate
      const bool is_up = ((lane_id >> 3) & 1) == 0;
      const uint32_t ch = 16 * q + 2 * (lane_id % 8) + (lane_id >> 4);
      const uint32_t blk = q / 2;  // 32-channel quant block of this tile: warps (0, 1) and (2, 3)
      // pass 1: activation (in place, kept for pass 2) and the per-token amax of this warp's 16 channels
#pragma unroll
      for (uint32_t c = 0; c < kBlockN / 8; ++c) {
        if (c < num_chunks) {
#pragma unroll
          for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t n = c * 8 + i;
            const float gate = __shfl_down_sync(warp::kFullMask, d[n], 8);
            d[n] = is_up ? kAct(gate, d[n]) : 0.0f;
            // |a| as bits orders like the float; one REDUX per token
            const uint32_t m = warp::reduce_max(__float_as_uint(fabsf(d[n])));
            if (lane_id == n) smem.amax[q][n] = __uint_as_float(m);
          }
        }
      }
      ptx::bar_sync(BAR_EPI_PAIR + blk, 64);
      uint32_t my_sbyte = 0;  // lane n: scale byte of token n
      if (lane_id < n_tok) {
        my_sbyte = ue8m0_round_up(fmaxf(smem.amax[2 * blk][lane_id], smem.amax[2 * blk + 1][lane_id]));
      }
      ptx::bar_sync(BAR_EPI_PAIR + blk, 64);  // amax reads done before the next item overwrites it

      // pass 2: quantize, store
#pragma unroll
      for (uint32_t c = 0; c < kBlockN / 8; ++c) {
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
        p.act_sf[(size_t(v) * Sh::kActSfAtoms + kb / 4) * kAtomBytes + lane_id * 16 + kb % 4] =
            static_cast<uint8_t>(my_sbyte);
      }
      // publish: each thread orders its stores before FC2's async-proxy reads, then one release increment after the
      // barrier (cumulative over the 4 warps' stores)
      ptx::fence_proxy_async_global();
      ptx::bar_sync(BAR_EPI_ALL, 128);
      if (q == 0 && warp::elect_one_lane()) ptx::red_async_release_add(p.counters + v, 1u);
      acc.advance();
    }
    // FC2: M tile m, lane l of quadrant q holds hidden row 128 (2 tile + m) + 32 q + 4 (l % 8) + l / 8
    while (true) {
      ptx::mbar_wait_parity(&smem.info_full[iring.stage], iring.phase);
      const auto& inf = smem.info[iring.stage];
      const uint32_t tile = inf.tile, row_base = inf.row_base, n_tok = inf.n_fc2;
      ptx::bar_sync(BAR_EPI_ALL, 128);
      if (q == 0 && lane_id == 0) ptx::mbar_arrive(&smem.info_empty[iring.stage]);
      iring.advance();
      if (n_tok == 0) break;
      ptx::mbar_wait_parity(&smem.tmem_full[acc.stage], acc.phase);
      tcgen05::fence_after_thread_sync();
      const uint32_t taddr = tmem_base + acc.stage * 2 * kBlockN + ((q * 32) << 16);
      const uint32_t num_chunks = div_ceil(n_tok, 8u);
      float d[2][kBlockN];
      tcgen05::ld_32x32b<2 * kBlockN>(taddr, d);  // the two M tiles' adjacent column ranges
      tcgen05::wait_ld();
      tcgen05::fence_before_thread_sync();
      if (lane_id == 0) ptx::mbar_arrive(&smem.tmem_empty[acc.stage]);
#pragma unroll
      for (uint32_t m = 0; m < 2; ++m) {
        const uint32_t o = 128 * (2 * tile + m) + 32 * q + 4 * (lane_id % 8) + lane_id / 8;
#pragma unroll
        for (uint32_t c = 0; c < kBlockN / 8; ++c) {
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

}  // namespace device::mxfp4_moe

// Kimi-K3.1 SiTU: beta tanh(gate / beta) sigmoid(gate) lb tanh(up / lb) (K3.1: beta 4, lb 25),
// sigmoid(x) = (1 + tanh(x / 2)) / 2
template <float kBeta, float kLinearBeta>
SGL_DEVICE float situ(float gate, float up) {
  namespace ptx = device::mxfp4_moe::ptx;
  static_assert(kBeta > 0.0f && kLinearBeta > 0.0f);
  constexpr float kHalfBeta = 0.5f * kBeta, kInvBeta = 1.0f / kBeta, kInvLinearBeta = 1.0f / kLinearBeta;
  const float g = kHalfBeta * ptx::tanh_approx(gate * kInvBeta) * (1.0f + ptx::tanh_approx(gate * 0.5f));
  const float u = kLinearBeta * ptx::tanh_approx(up * kInvLinearBeta);
  return g * u;
}

// DeepSeek-V4 clamped SwiGLU: silu(min(gate, L)) clamp(up, -L, L) on the fp32 accumulator, as flashinfer's fused FC1
// epilogue (no bf16 rounding before the clamp)
template <float kLimit>
SGL_DEVICE float swiglu(float gate, float up) {
  static_assert(kLimit > 0.0f);
  const float g = fminf(gate, kLimit);
  const float u = fminf(fmaxf(up, -kLimit), kLimit);
  return __fdividef(g, 1.0f + __expf(-g)) * u;
}

// One call: the routing kernel (two CTAs; PDL secondary of the top-k producer) then the MoE kernel (PDL secondary of
// the routing kernel). Outputs: gemm2_out [>= T * top-k, H] bf16 permuted FC2 rows and expanded_idx [T * top-k]
// (row of slot (t, k), -1 when the id is not one of the E local experts [expert_offset, expert_offset + E)). With
// `topk_weights`, `topk_ids` is the packed (id << 16) | bf16(weight) form and the weights are written there.
// `workspace` holds no state across calls (the routing kernel resets the counters).
template <
    uint32_t kNumExperts,
    uint32_t kTopK,
    uint32_t kHidden,
    uint32_t kInter,
    uint32_t kBlockN,
    uint32_t kFc1Prefetch,  // see moe_kernel
    uint32_t kFc2Prefetch,
    auto kAct>  // FC1 activation
struct MXFP4DecodeMoE {
  using Sh = device::mxfp4_moe::Shape<kNumExperts, kTopK, kHidden, kInter, kBlockN>;

  static int64_t workspace_bytes() {
    return static_cast<int64_t>(Sh::kWorkspaceBytes);
  }

  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView x_sf,
      tvm::ffi::TensorView topk_ids,
      tvm::ffi::TensorView w13,
      tvm::ffi::TensorView w13_sf,
      tvm::ffi::TensorView w2,
      tvm::ffi::TensorView w2_sf,
      int64_t expert_offset,
      tvm::ffi::TensorView workspace,
      tvm::ffi::TensorView gemm2_out,
      tvm::ffi::TensorView expanded_idx,
      tvm::ffi::Optional<tvm::ffi::TensorView> topk_weights) {
    using namespace host;
    using namespace device::mxfp4_moe;
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
    TensorMatcher({T, kTopK})  // topk_ids: read flat in 16-byte vectors, so only the base needs the alignment
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(topk_ids);
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
    TensorMatcher({kNumExperts, kHidden, kInter / 32})  // w2_sf: rows of I / 32 bytes, read as 128x4 atoms
        .with_dtype<uint8_t>()
        .with_device(device)
        .verify(w2_sf);
    TensorMatcher({static_cast<int64_t>(Sh::kWorkspaceBytes)})  // workspace
        .with_dtype<uint8_t>()
        .with_device(device)
        .ensure_alignment(256)
        .verify(workspace);
    TensorMatcher({-1, kHidden})  // gemm2_out
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(gemm2_out);
    TensorMatcher({-1})  // expanded_idx
        .with_dtype<int32_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(expanded_idx);
    // w2_sf rows are I / 32 bytes, so only its base can be 16-byte aligned
    CHECK_HOST(reinterpret_cast<uintptr_t>(w2_sf.data_ptr()) % 16 == 0) << "w2_sf is not 16-byte aligned";
    CHECK_HOST(reinterpret_cast<uintptr_t>(topk_ids.data_ptr()) % 16 == 0) << "topk_ids is not 16-byte aligned";
    const auto num_tokens = static_cast<uint32_t>(T.unwrap());
    CHECK_HOST(num_tokens >= 1 && num_tokens <= kMaxM) << "num_tokens = " << num_tokens;
    CHECK_HOST(expanded_idx.size(0) == num_tokens * kTopK) << "expanded_idx holds " << expanded_idx.size(0);
    CHECK_HOST(gemm2_out.size(0) >= num_tokens * kTopK) << "gemm2_out holds " << gemm2_out.size(0) << " rows";
    CHECK_HOST(expert_offset >= 0 && expert_offset < (int64_t{1} << 30)) << "expert_offset = " << expert_offset;
    const bool packed = topk_weights.has_value();
    if (packed) {
      TensorMatcher({T, kTopK})  // topk_weights: written flat in 8-byte vectors
          .with_dtype<bf16_t>()
          .with_device(device)
          .verify(topk_weights.value());
      CHECK_HOST(reinterpret_cast<uintptr_t>(topk_weights.value().data_ptr()) % 8 == 0);
    }
    const DLDevice dev = device.unwrap();

    const auto ws = static_cast<uint8_t*>(workspace.data_ptr());
    const auto counters = reinterpret_cast<uint32_t*>(ws + Sh::kCountersOffset);
    const auto vinfo = reinterpret_cast<uint32_t*>(ws + Sh::kVInfoOffset);
    const auto token_bits = reinterpret_cast<uint32_t*>(ws + Sh::kTokenBitsOffset);
    const auto act_sf = ws + Sh::kActSfOffset;
    const auto act = ws + Sh::kActOffset;

    const RouteParams route_params{
        .topk_ids = static_cast<const int32_t*>(topk_ids.data_ptr()),
        .topk_weights = packed ? static_cast<uint16_t*>(topk_weights.value().data_ptr()) : nullptr,
        .expanded_idx = static_cast<int32_t*>(expanded_idx.data_ptr()),
        .token_bits = token_bits,
        .vinfo = vinfo,
        .counters = counters,
        .num_tokens = num_tokens,
        .expert_offset = static_cast<uint32_t>(expert_offset),
    };
    const uint32_t num_words = div_ceil(num_tokens, 32u);
    const auto pick = [&]<uint32_t kWords>() {
      return packed ? route_kernel<Sh, true, kWords> : route_kernel<Sh, false, kWords>;
    };
    const auto route = num_words <= 1   ? pick.template operator()<1>()
                       : num_words <= 2 ? pick.template operator()<2>()
                       : num_words <= 4 ? pick.template operator()<4>()
                                        : pick.template operator()<kMaxWords>();
    LaunchKernel(2, kRouteThreads, dev).enable_pdl(true)(route, route_params);

    constexpr uint32_t kSmem = sizeof(Smem<kBlockN, num_stages<kBlockN>()>) + 1024;  // + the runtime alignment
    static_assert(kSmem <= kSmemBudget, "smem");
    constexpr auto kernel = moe_kernel<Sh, kFc1Prefetch, kFc2Prefetch, kAct>;
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
        act,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        {kInter, num_tokens * kTopK + kBlockN},
        {kInter},
        {128, kBlockN},
        {.swizzle = CU_TENSOR_MAP_SWIZZLE_128B});
    const Params params{
        .w13_sf = static_cast<const uint8_t*>(w13_sf.data_ptr()),
        .w2_sf = static_cast<const uint8_t*>(w2_sf.data_ptr()),
        .x_sf = static_cast<const uint8_t*>(x_sf.data_ptr()),
        .token_bits = token_bits,
        .vinfo = vinfo,
        .act = act,
        .act_sf = act_sf,
        .gemm2_out = static_cast<bf16_t*>(gemm2_out.data_ptr()),
        .counters = counters,
    };
    LaunchKernel(grid, kNumThreads, dev, kSmem).enable_pdl(true)(kernel, map_w13, map_w2, map_x, map_act, params);
  }
};

}  // namespace sglang
