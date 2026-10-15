// MoE router front end for decode token counts in one launch: router GEMV
// (512 experts x 4096 hidden, bf16 weights, fp32 accumulation, bf16 logits),
// softmax / top-10 / renormalize, NVFP4 quantization of the activations
// (16-element blocks, e4m3 scales, linear layout) and the routing tables the
// cute-dsl grouped GEMMs consume.
//
// Every CTA computes the logits of kExpertsPerCta experts for all tokens on
// tensor cores; CTA b < M also quantizes token row b. The last CTA to arrive
// (acq_rel counter, no spinning) runs top-k and builds the routing tables.
//
// Given the same bf16 logits, the softmax / top-k / renormalize arithmetic
// follows the operation order of the Triton router kernel
// (moe_fused_gate._router_triton_kernel, softmax, renormalize, 512 experts), the
// quantization follows flashinfer's NVFP4 linear-layout quantizer, and the
// routing tables equal flashinfer moe_sort's output on every entry the GEMMs
// read.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace qwen_router_front {

inline constexpr uint32_t kHidden = 4096;
inline constexpr uint32_t kExperts = 512;
inline constexpr uint32_t kTopK = 10;
inline constexpr uint32_t kMaxTokens = 16;
inline constexpr uint32_t kThreads = 256;
inline constexpr uint32_t kWarps = kThreads / 32;
inline constexpr uint32_t kExpertsPerCta = 4;
inline constexpr uint32_t kNumCtas = kExperts / kExpertsPerCta;
inline constexpr uint32_t kKPerWarp = kHidden / kWarps;  // 512
inline constexpr uint32_t kKPerLane = kKPerWarp / 32;    // 16: one NVFP4 block
inline constexpr uint32_t kSfVec = 16;
inline constexpr uint32_t kBlocksPerRow = kHidden / kSfVec;  // 256
static_assert(kKPerLane == kSfVec, "each lane must own exactly one quantization block");
static_assert(kExperts == 2 * 256, "the top-k lane layout assumes 2 x 256 experts");

struct Params {
  const bf16_t* x;            // [M, kHidden]
  const bf16_t* weight;       // [kExperts, kHidden]
  const float* global_scale;  // [1], quantization-domain global scale
  bf16_t* logits;             // [M, kExperts]
  float* topk_weights;        // [M, kTopK]
  int32_t* topk_ids;          // [M, kTopK]
  uint8_t* x_fp4;             // [M, kHidden / 2]
  uint8_t* x_sf;              // [M, kBlocksPerRow]
  int32_t* tile_idx_to_expert_idx;
  int32_t* tile_idx_to_mn_limit;
  int32_t* expanded_idx_to_permuted_idx;  // [M, kTopK]
  int32_t* permuted_idx_to_expanded_idx;
  int32_t* total_num_padded_tokens;  // [1]
  int32_t* num_non_exiting_tiles;    // [1]
  uint32_t* arrival;                 // monotonic arrival counter, one per call site
  uint64_t* trace;                   // optional %globaltimer stamps (instrumented build only)
  uint32_t num_tokens;
  uint32_t tile_log2;
};

SGL_DEVICE float bf16_lo(uint32_t v) {
  return __uint_as_float(v << 16);
}
SGL_DEVICE float bf16_hi(uint32_t v) {
  return __uint_as_float(v & 0xFFFF0000u);
}

struct U32x8 {
  uint4 lo, hi;
};

SGL_DEVICE U32x8 ld_weight(const bf16_t* p) {
  // Static weights, 32 B per lane: read-only path, not kept in L1, evict-first in L2.
  U32x8 v;
  asm volatile(
      "ld.global.nc.L1::no_allocate.L2::evict_first.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(v.lo.x), "=r"(v.lo.y), "=r"(v.lo.z), "=r"(v.lo.w), "=r"(v.hi.x), "=r"(v.hi.y), "=r"(v.hi.z), "=r"(v.hi.w)
      : "l"(p));
  return v;
}

SGL_DEVICE U32x8 ld_produced32(const void* p) {
  // Data written by an earlier grid: coherent load, issued after the PDL wait.
  U32x8 v;
  asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=r"(v.lo.x), "=r"(v.lo.y), "=r"(v.lo.z), "=r"(v.lo.w), "=r"(v.hi.x), "=r"(v.hi.y), "=r"(v.hi.z),
                 "=r"(v.hi.w)
               : "l"(p));
  return v;
}

SGL_DEVICE uint4 ld_produced(const void* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SGL_DEVICE uint64_t global_timer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

SGL_DEVICE float ex2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

SGL_DEVICE float div_full(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

SGL_DEVICE float rcp_approx_ftz(float a) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(a));
  return y;
}

SGL_DEVICE uint32_t cvt_e4m3(float a) {
  uint32_t out;
  asm("{\n"
      ".reg .b16 pair;\n"
      ".reg .f32 zero;\n"
      "mov.f32 zero, 0f00000000;\n"
      "cvt.rn.satfinite.e4m3x2.f32 pair, zero, %1;\n"
      "cvt.u32.u16 %0, pair;\n"
      "}"
      : "=r"(out)
      : "f"(a));
  return out & 0xFFu;
}

SGL_DEVICE float e4m3_to_f32(uint32_t v) {
  float out;
  asm("{\n"
      ".reg .b16 pair;\n"
      ".reg .b32 h2;\n"
      ".reg .b16 lo, hi;\n"
      "cvt.u16.u32 pair, %1;\n"
      "cvt.rn.f16x2.e4m3x2 h2, pair;\n"
      "mov.b32 {lo, hi}, h2;\n"
      "cvt.f32.f16 %0, lo;\n"
      "}"
      : "=f"(out)
      : "r"(v));
  return out;
}

SGL_DEVICE uint32_t cvt_e2m1x8(const float (&v)[8]) {
  uint32_t out;
  asm("{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}"
      : "=r"(out)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return out;
}

SGL_DEVICE uint32_t bf16x2_abs_max(uint32_t a, uint32_t b) {
  uint32_t out;
  asm("{\n"
      ".reg .b32 ta, tb;\n"
      "and.b32 ta, %1, 0x7FFF7FFF;\n"
      "and.b32 tb, %2, 0x7FFF7FFF;\n"
      "max.bf16x2 %0, ta, tb;\n"
      "}"
      : "=r"(out)
      : "r"(a), "r"(b));
  return out;
}

SGL_DEVICE uint32_t bf16x2_max(uint32_t a, uint32_t b) {
  uint32_t out;
  asm("max.bf16x2 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
  return out;
}

// One 16-element NVFP4 block: e4m3 scale byte and 8 packed bytes (element 2i in
// the low nibble of byte i).
SGL_DEVICE void quantize_block(const uint4& a, const uint4& b, float gs, uint32_t& sf, uint2& packed) {
  const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  const uint32_t m01 = bf16x2_abs_max(w[0], w[1]);
  const uint32_t m23 = bf16x2_abs_max(w[2], w[3]);
  const uint32_t m45 = bf16x2_abs_max(w[4], w[5]);
  const uint32_t m67 = bf16x2_abs_max(w[6], w[7]);
  const uint32_t m = bf16x2_max(bf16x2_max(m01, m23), bf16x2_max(m45, m67));
  const float amax = fmaxf(bf16_lo(m), bf16_hi(m));
  const float scale = __fmul_rn(gs, __fmul_rn(amax, rcp_approx_ftz(6.0f)));
  sf = cvt_e4m3(scale);
  const float sf_f = e4m3_to_f32(sf);
  const float out_scale = (sf_f == 0.0f) ? 0.0f : rcp_approx_ftz(__fmul_rn(sf_f, rcp_approx_ftz(gs)));
  float lo[8], hi[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    lo[2 * i] = __fmul_rn(bf16_lo(w[i]), out_scale);
    lo[2 * i + 1] = __fmul_rn(bf16_hi(w[i]), out_scale);
    hi[2 * i] = __fmul_rn(bf16_lo(w[4 + i]), out_scale);
    hi[2 * i + 1] = __fmul_rn(bf16_hi(w[4 + i]), out_scale);
  }
  packed.x = cvt_e2m1x8(lo);
  packed.y = cvt_e2m1x8(hi);
}

// Monotonic 16-bit key of a bf16 value (the canonical +0 for -0 is applied by
// the caller's +0.0 add).
SGL_DEVICE uint32_t bf16_order_key(uint32_t bits16) {
  return (bits16 & 0x8000u) ? (~bits16 & 0xFFFFu) : (bits16 | 0x8000u);
}

SGL_DEVICE float bf16_from_key(uint32_t key16) {
  const uint32_t bits = (key16 & 0x8000u) ? (key16 & 0x7FFFu) : (~key16 & 0xFFFFu);
  return __uint_as_float(bits << 16);
}

template <bool kTrace>
SGL_DEVICE void trace_stamp(const Params& p, uint32_t slot) {
  if constexpr (kTrace) {
    if (threadIdx.x == 0) p.trace[slot] = global_timer();
  }
}

// D += A * B for one m16n8k16 bf16 tile, fp32 accumulators.
SGL_DEVICE void mma_bf16_16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                               uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

SGL_DEVICE uint32_t u32x8_at(const U32x8& v, int i) {
  switch (i) {
    case 0: return v.lo.x;
    case 1: return v.lo.y;
    case 2: return v.lo.z;
    case 3: return v.lo.w;
    case 4: return v.hi.x;
    case 5: return v.hi.y;
    case 6: return v.hi.z;
    default: return v.hi.w;
  }
}

// Softmax / top-10 / renormalize for kNT tokens of one warp, interleaved for ILP.
// Lane layout of the reference kernel: lane holds experts [8l, 8l+8) and [256+8l, 256+8l+8).
template <bool kTrace>
SGL_DEVICE void cycle_stamp(const Params& p, uint32_t slot, uint64_t t0) {
  if constexpr (kTrace) {
    if (threadIdx.x == 0) p.trace[kNumCtas * 8 + slot] = clock64() - t0;
  }
}

template <int kNT, bool kTrace>
SGL_DEVICE void topk_tokens(const Params& p, const uint32_t (&m)[kNT], uint32_t* s_mask, int16_t* s_ids) {
  const uint32_t lane = threadIdx.x % 32;
  const uint64_t c0 = kTrace ? clock64() : 0;
  constexpr float kLog2e = 1.44269502162933349609375f;
  float a[kNT][16];
  uint4 r[kNT][2];
#pragma unroll
  for (int t = 0; t < kNT; ++t) {
    const bf16_t* row = p.logits + m[t] * kExperts;
    r[t][0] = ld_produced(row + 8 * lane);
    r[t][1] = ld_produced(row + 256 + 8 * lane);
  }
  if (r[0][1].w != 0x12345678u) cycle_stamp<kTrace>(p, 4, c0);
  float row_max[kNT], row_sum[kNT];
#pragma unroll
  for (int t = 0; t < kNT; ++t) {
    const uint32_t w[8] = {r[t][0].x, r[t][0].y, r[t][0].z, r[t][0].w, r[t][1].x, r[t][1].y, r[t][1].z, r[t][1].w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      a[t][2 * i] = __fadd_rn(bf16_lo(w[i]), 0.0f);
      a[t][2 * i + 1] = __fadd_rn(bf16_hi(w[i]), 0.0f);
    }
    float mx = a[t][0];
#pragma unroll
    for (int i = 1; i < 16; ++i) mx = fmaxf(mx, a[t][i]);
    row_max[t] = mx;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
    for (int t = 0; t < kNT; ++t) row_max[t] = fmaxf(row_max[t], __shfl_xor_sync(0xFFFFFFFFu, row_max[t], off));
  }
#pragma unroll
  for (int t = 0; t < kNT; ++t) {
    float e[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) e[i] = ex2_approx(__fmul_rn(__fsub_rn(a[t][i], row_max[t]), kLog2e));
    // e[0..7] are experts 8l..8l+7, e[8..15] are 256+8l..: the reference's in-thread tree.
    const float lo = __fadd_rn(
        __fadd_rn(__fadd_rn(e[0], e[2]), __fadd_rn(e[4], e[6])),
        __fadd_rn(__fadd_rn(e[8], e[10]), __fadd_rn(e[12], e[14])));
    const float hi = __fadd_rn(
        __fadd_rn(__fadd_rn(e[1], e[3]), __fadd_rn(e[5], e[7])),
        __fadd_rn(__fadd_rn(e[9], e[11]), __fadd_rn(e[13], e[15])));
    row_sum[t] = __fadd_rn(lo, hi);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
    for (int t = 0; t < kNT; ++t) row_sum[t] = __fadd_rn(row_sum[t], __shfl_xor_sync(0xFFFFFFFFu, row_sum[t], off));
  }

  if (row_sum[0] != 12345.0f) cycle_stamp<kTrace>(p, 5, c0);
  // Keys: monotonic value in the high half, (0xFFFF - expert) in the low half, so
  // the warp max is the largest logit with the lowest expert id on ties.
  uint32_t key[kNT][16];
#pragma unroll
  for (int t = 0; t < kNT; ++t) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint32_t idx = (i < 8) ? (8 * lane + i) : (256 + 8 * lane + (i - 8));
      const uint32_t bits = __float_as_uint(a[t][i]) >> 16;
      key[t][i] = (a[t][i] == a[t][i]) ? ((bf16_order_key(bits) << 16) | (0xFFFFu - idx)) : (0xFFFFu - idx);
    }
    // Lane-local descending bitonic sort.
#pragma unroll
    for (int k = 2; k <= 16; k <<= 1) {
#pragma unroll
      for (int j = k >> 1; j > 0; j >>= 1) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const int l = i ^ j;
          if (l > i) {
            const uint32_t x = key[t][i], y = key[t][l];
            if ((i & k) == 0) {
              key[t][i] = max(x, y);
              key[t][l] = min(x, y);
            } else {
              key[t][i] = min(x, y);
              key[t][l] = max(x, y);
            }
          }
        }
      }
    }
  }

  if (key[0][0] != 0x12345678u) cycle_stamp<kTrace>(p, 6, c0);
  float sel[kNT][kTopK];
  uint32_t ids[kNT][kTopK];
#pragma unroll
  for (int rr = 0; rr < static_cast<int>(kTopK); ++rr) {
#pragma unroll
    for (int t = 0; t < kNT; ++t) {
      const uint32_t winner = __reduce_max_sync(0xFFFFFFFFu, key[t][0]);
      const bool owner = key[t][0] == winner;
#pragma unroll
      for (int i = 0; i < 15; ++i) key[t][i] = owner ? key[t][i + 1] : key[t][i];
      key[t][15] = owner ? 0u : key[t][15];
      ids[t][rr] = 0xFFFFu - (winner & 0xFFFFu);
      // The winner's softmax probability, recomputed from its logit with the
      // per-expert operations instead of shuffled from its owner lane.
      const float v = bf16_from_key(winner >> 16);
      sel[t][rr] = div_full(ex2_approx(__fmul_rn(__fsub_rn(v, row_max[t]), kLog2e)), row_sum[t]);
    }
  }
  if (sel[0][kTopK - 1] != 12345.0f) cycle_stamp<kTrace>(p, 7, c0);
#pragma unroll
  for (int t = 0; t < kNT; ++t) {
    // Reference reduction over the 16 top-k slots (10 used): even lanes hold slots 0..7, odd 8..15.
    const float even = __fadd_rn(
        __fadd_rn(__fadd_rn(sel[t][0], sel[t][2]), __fadd_rn(sel[t][4], sel[t][6])),
        __fadd_rn(__fadd_rn(sel[t][1], sel[t][3]), __fadd_rn(sel[t][5], sel[t][7])));
    const float odd = __fadd_rn(sel[t][8], sel[t][9]);
    const float routed_sum = __fadd_rn(even, odd);
    const float norm = routed_sum > 0.0f ? routed_sum : 1.0f;
    if (lane < kTopK) {
      float w = 0.0f;
      uint32_t id = 0;
#pragma unroll
      for (int rr = 0; rr < static_cast<int>(kTopK); ++rr) {
        if (lane == static_cast<uint32_t>(rr)) {
          w = div_full(sel[t][rr], norm);
          id = ids[t][rr];
        }
      }
      p.topk_weights[m[t] * kTopK + lane] = w;
      p.topk_ids[m[t] * kTopK + lane] = static_cast<int32_t>(id);
      s_ids[m[t] * kTopK + lane] = static_cast<int16_t>(id);
      atomicOr(&s_mask[id], 1u << m[t]);
    }
  }
}

template <bool kTrace>
SGL_DEVICE void topk_and_tables(const Params& p, uint32_t* s_mask, int16_t* s_tile, int16_t* s_ids, uint32_t* s_warp) {
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t M = p.num_tokens;

  for (uint32_t i = threadIdx.x; i < kExperts; i += kThreads) s_mask[i] = 0;
  __syncthreads();
  trace_stamp<kTrace>(p, kNumCtas * 8 + 1);
  if (M > kWarps) {
    if (warp + kWarps < M) {
      const uint32_t m2[2] = {warp, warp + kWarps};
      topk_tokens<2, kTrace>(p, m2, s_mask, s_ids);
    } else {
      const uint32_t m1[1] = {warp};
      topk_tokens<1, kTrace>(p, m1, s_mask, s_ids);
    }
  } else if (warp < M) {
    const uint32_t m1[1] = {warp};
    topk_tokens<1, kTrace>(p, m1, s_mask, s_ids);
  }
  __syncthreads();
  trace_stamp<kTrace>(p, kNumCtas * 8 + 2);

  // Tiles in ascending expert order; with M <= 16 every active expert owns one tile.
  const uint32_t e0 = 2 * threadIdx.x;
  const uint32_t mask0 = s_mask[e0], mask1 = s_mask[e0 + 1];
  const bool act0 = mask0 != 0, act1 = mask1 != 0;
  const uint32_t bal0 = __ballot_sync(0xFFFFFFFFu, act0);
  const uint32_t bal1 = __ballot_sync(0xFFFFFFFFu, act1);
  const uint32_t lt = (1u << lane) - 1u;
  const uint32_t in_warp = __popc(bal0 & lt) + __popc(bal1 & lt);
  if (lane == 0) s_warp[warp] = __popc(bal0) + __popc(bal1);
  __syncthreads();
  uint32_t base = 0, total = 0;
#pragma unroll
  for (uint32_t w = 0; w < kWarps; ++w) {
    const uint32_t c = s_warp[w];
    base += (w < warp) ? c : 0u;
    total += c;
  }
  const uint32_t tile0 = base + in_warp;
  const uint32_t tile1 = tile0 + (act0 ? 1u : 0u);
  if (act0) {
    s_tile[e0] = static_cast<int16_t>(tile0);
    p.tile_idx_to_expert_idx[tile0] = static_cast<int32_t>(e0);
    p.tile_idx_to_mn_limit[tile0] = static_cast<int32_t>((tile0 << p.tile_log2) + __popc(mask0));
  }
  if (act1) {
    s_tile[e0 + 1] = static_cast<int16_t>(tile1);
    p.tile_idx_to_expert_idx[tile1] = static_cast<int32_t>(e0 + 1);
    p.tile_idx_to_mn_limit[tile1] = static_cast<int32_t>((tile1 << p.tile_log2) + __popc(mask1));
  }
  if (threadIdx.x == 0) {
    p.total_num_padded_tokens[0] = static_cast<int32_t>(total << p.tile_log2);
    p.num_non_exiting_tiles[0] = static_cast<int32_t>(total);
  }
  __syncthreads();
  for (uint32_t i = threadIdx.x; i < M * kTopK; i += kThreads) {
    const uint32_t m = i / kTopK;
    const uint32_t e = static_cast<uint32_t>(s_ids[i]);
    const uint32_t rank = __popc(s_mask[e] & ((1u << m) - 1u));
    const uint32_t permuted = (static_cast<uint32_t>(s_tile[e]) << p.tile_log2) + rank;
    p.expanded_idx_to_permuted_idx[i] = static_cast<int32_t>(permuted);
    p.permuted_idx_to_expanded_idx[permuted] = static_cast<int32_t>(i);
  }
  trace_stamp<kTrace>(p, kNumCtas * 8 + 3);
}

// Router GEMV on tensor cores. Per CTA: kExpertsPerCta experts as the n8 tile
// (columns 4..7 zero), tokens as the m16 tile, each warp a 512-wide K slice in
// 8 groups of 64. Inside a group, lane (g, c) loads the 32 B chunk
// [base + 16c, base + 16c + 16) of token row g and expert row g, and step t uses
// its elements [4t, 4t + 4): a fixed K permutation shared by A and B.
template <uint32_t kMaxM>
SGL_DEVICE void gemv_tiles(const Params& p, uint32_t expert0, const U32x8 (&w)[8], float (&acc)[4][4]) {
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t g = lane >> 2, c = lane & 3;
  const uint32_t M = p.num_tokens;
  const bool row_lo = g < M;
  const bool row_hi = (kMaxM > 8) && (g + 8 < M);
  const bf16_t* x_lo = p.x + static_cast<size_t>(g) * kHidden + warp * kKPerWarp + 16 * c;
  const bf16_t* x_hi = x_lo + 8 * static_cast<size_t>(kHidden);
#pragma unroll
  for (int i = 0; i < 4; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    U32x8 xl{make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    U32x8 xh{make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    if (row_lo) xl = ld_produced32(x_lo + 64 * q);
    if (row_hi) xh = ld_produced32(x_hi + 64 * q);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      mma_bf16_16816(
          acc[t],
          u32x8_at(xl, 2 * t),
          u32x8_at(xh, 2 * t),
          u32x8_at(xl, 2 * t + 1),
          u32x8_at(xh, 2 * t + 1),
          u32x8_at(w[q], 2 * t),
          u32x8_at(w[q], 2 * t + 1));
    }
  }
  (void)expert0;
}

template <uint32_t kMaxM, bool kUsePDL, bool kTrace>
__global__ __launch_bounds__(kThreads, 1) void router_front_kernel(const __grid_constant__ Params p) {
  __shared__ float s_red[kWarps][kMaxTokens][kExpertsPerCta];
  __shared__ uint32_t s_mask[kExperts];
  __shared__ int16_t s_tile[kExperts];
  __shared__ int16_t s_ids[kMaxTokens * kTopK];
  __shared__ uint32_t s_warp[kWarps];
  __shared__ uint32_t s_last;

  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t g = lane >> 2, c = lane & 3;
  const uint32_t expert0 = blockIdx.x * kExpertsPerCta;
  const uint32_t M = p.num_tokens;
  const uint32_t tb = blockIdx.x * 8;
  trace_stamp<kTrace>(p, tb + 0);

  // Router weights are static: load them before waiting on the producer grid.
  U32x8 w[8];
  {
    const bf16_t* src = p.weight + static_cast<size_t>(expert0 + (g & 3)) * kHidden + warp * kKPerWarp + 16 * c;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      w[q] = U32x8{make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
      if (g < kExpertsPerCta) w[q] = ld_weight(src + 64 * q);
    }
  }
  if (u32x8_at(w[7], 7) != 0x12345678u) trace_stamp<kTrace>(p, tb + 1);

  device::PDLWaitPrimary<kUsePDL>();
  trace_stamp<kTrace>(p, tb + 2);

  // Row blockIdx.x is quantized by this CTA, one 16-element block per thread,
  // loaded now so the latency overlaps the GEMV.
  const bool quant_row = blockIdx.x < M;
  const uint32_t qk = warp * kKPerWarp + lane * kKPerLane;
  U32x8 qv{make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
  if (quant_row) qv = ld_produced32(p.x + static_cast<size_t>(blockIdx.x) * kHidden + qk);

  float acc[4][4];
  gemv_tiles<kMaxM>(p, expert0, w, acc);
  if (acc[0][0] != 12345.0f) trace_stamp<kTrace>(p, tb + 3);

  if (quant_row) {
    uint32_t sf;
    uint2 packed;
    quantize_block(qv.lo, qv.hi, p.global_scale[0], sf, packed);
    const uint32_t blk = qk / kSfVec;
    *reinterpret_cast<uint2*>(p.x_fp4 + static_cast<size_t>(blockIdx.x) * (kHidden / 2) + blk * 8) = packed;
    p.x_sf[static_cast<size_t>(blockIdx.x) * kBlocksPerRow + blk] = static_cast<uint8_t>(sf);
  }

  // C fragment: d0/d1 = (token g, experts 2c, 2c+1), d2/d3 = token g + 8.
  if (c < kExpertsPerCta / 2) {
    float d[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) d[i] = __fadd_rn(__fadd_rn(acc[0][i], acc[1][i]), __fadd_rn(acc[2][i], acc[3][i]));
    s_red[warp][g][2 * c] = d[0];
    s_red[warp][g][2 * c + 1] = d[1];
    if constexpr (kMaxM > 8) {
      s_red[warp][g + 8][2 * c] = d[2];
      s_red[warp][g + 8][2 * c + 1] = d[3];
    }
  }
  __syncthreads();
  if (threadIdx.x < M * kExpertsPerCta) {
    const uint32_t m = threadIdx.x / kExpertsPerCta;
    const uint32_t e = threadIdx.x % kExpertsPerCta;
    float s = s_red[0][m][e];
#pragma unroll
    for (uint32_t ww = 1; ww < kWarps; ++ww) s = __fadd_rn(s, s_red[ww][m][e]);
    p.logits[m * kExperts + expert0 + e] = __float2bfloat16_rn(s);
  }

  // Publish: bar.sync orders this CTA's stores before the arriving thread's
  // gpu-scope release; the last arrival acquires and runs top-k.
  __syncthreads();
  trace_stamp<kTrace>(p, tb + 4);
  if (threadIdx.x == 0) {
    uint32_t prev;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;" : "=r"(prev) : "l"(p.arrival) : "memory");
    s_last = ((prev + 1u) % kNumCtas) == 0u ? 1u : 0u;
  }
  __syncthreads();
  trace_stamp<kTrace>(p, tb + 5);
  if (s_last) {
    trace_stamp<kTrace>(p, kNumCtas * 8 + 0);
    topk_and_tables<kTrace>(p, s_mask, s_tile, s_ids, s_warp);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <uint32_t kMaxM, bool kUsePDL, bool kTrace>
struct RouterFrontKernel {
  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView global_scale,
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView x_fp4,
      const tvm::ffi::TensorView x_sf,
      const tvm::ffi::TensorView tile_idx_to_expert_idx,
      const tvm::ffi::TensorView tile_idx_to_mn_limit,
      const tvm::ffi::TensorView expanded_idx_to_permuted_idx,
      const tvm::ffi::TensorView permuted_idx_to_expanded_idx,
      const tvm::ffi::TensorView total_num_padded_tokens,
      const tvm::ffi::TensorView num_non_exiting_tiles,
      const tvm::ffi::TensorView arrival,
      const tvm::ffi::TensorView trace,
      int64_t tile_log2) {
    using namespace host;
    auto M_ = SymbolicSize{"num_tokens"};
    auto T_ = SymbolicSize{"max_tiles"};
    auto P_ = SymbolicSize{"max_permuted"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M_, kHidden}).with_dtype<bf16_t>().with_device(device).ensure_alignment(32).verify(x);
    TensorMatcher({kExperts, kHidden}).with_dtype<bf16_t>().with_device(device).ensure_alignment(32).verify(weight);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(global_scale);
    TensorMatcher({M_, kExperts}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(logits);
    TensorMatcher({M_, kTopK}).with_dtype<fp32_t>().with_device(device).verify(topk_weights);
    TensorMatcher({M_, kTopK}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({M_, kHidden / 2}).with_dtype<uint8_t>().with_device(device).ensure_alignment(8).verify(x_fp4);
    TensorMatcher({M_, kBlocksPerRow}).with_dtype<uint8_t>().with_device(device).verify(x_sf);
    TensorMatcher({T_}).with_dtype<int32_t>().with_device(device).verify(tile_idx_to_expert_idx);
    TensorMatcher({T_}).with_dtype<int32_t>().with_device(device).verify(tile_idx_to_mn_limit);
    TensorMatcher({M_, kTopK}).with_dtype<int32_t>().with_device(device).verify(expanded_idx_to_permuted_idx);
    TensorMatcher({P_}).with_dtype<int32_t>().with_device(device).verify(permuted_idx_to_expanded_idx);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(total_num_padded_tokens);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(num_non_exiting_tiles);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(arrival);
    if constexpr (kTrace) {
      TensorMatcher({(kNumCtas + 1) * 8}).with_dtype<int64_t>().with_device(device).verify(trace);
    }
    const auto M = static_cast<uint32_t>(M_.unwrap());
    CHECK_HOST(M <= kMaxM && kMaxM <= kMaxTokens) << "router_front: num_tokens " << M << " exceeds " << kMaxM;
    CHECK_HOST(tile_log2 >= 4 && tile_log2 <= 10) << "router_front: tile_log2 " << tile_log2;
    CHECK_HOST(static_cast<uint64_t>(T_.unwrap()) >= M * kTopK) << "router_front: tile table too small";
    CHECK_HOST(static_cast<uint64_t>(P_.unwrap()) >= (static_cast<uint64_t>(M) * kTopK) << tile_log2)
        << "router_front: permutation table too small";
    if (M == 0) return;
    const Params params{
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(weight.data_ptr()),
        static_cast<const float*>(global_scale.data_ptr()),
        static_cast<bf16_t*>(logits.data_ptr()),
        static_cast<float*>(topk_weights.data_ptr()),
        static_cast<int32_t*>(topk_ids.data_ptr()),
        static_cast<uint8_t*>(x_fp4.data_ptr()),
        static_cast<uint8_t*>(x_sf.data_ptr()),
        static_cast<int32_t*>(tile_idx_to_expert_idx.data_ptr()),
        static_cast<int32_t*>(tile_idx_to_mn_limit.data_ptr()),
        static_cast<int32_t*>(expanded_idx_to_permuted_idx.data_ptr()),
        static_cast<int32_t*>(permuted_idx_to_expanded_idx.data_ptr()),
        static_cast<int32_t*>(total_num_padded_tokens.data_ptr()),
        static_cast<int32_t*>(num_non_exiting_tiles.data_ptr()),
        static_cast<uint32_t*>(arrival.data_ptr()),
        kTrace ? static_cast<uint64_t*>(trace.data_ptr()) : nullptr,
        M,
        static_cast<uint32_t>(tile_log2),
    };
    LaunchKernel(kNumCtas, kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(router_front_kernel<kMaxM, kUsePDL, kTrace>, params);
  }
};

}  // namespace qwen_router_front

using qwen_router_front::RouterFrontKernel;

}  // namespace sglang
