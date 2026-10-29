// Q38 experiment: H2560 router/quant only. Numerical parity is NOT yet validated.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// MoE router front end for decode token counts in one launch: router GEMV
// (512 experts x 4096 hidden, bf16 weights, fp32 accumulation, bf16 logits),
// softmax / top-10 / renormalize, NVFP4 quantization of the activations
// (16-element blocks, e4m3 scales, linear layout) and the routing tables the
// cute-dsl grouped GEMMs consume.
//
// Every CTA computes the logits of kExpertsPerCta experts for all tokens on
// tensor cores, bitwise equal to the cuBLAS router GEMM (see gemv_chain); CTA
// b < M also quantizes token row b. The last CTA to arrive (acq_rel counter, no
// spinning) runs top-k and builds the routing tables.
// router_tail_kernel runs the same top-k, quantization and tables behind a
// router GEMM computed elsewhere (bitwise to the stock chain given its logits).
//
// Given the same bf16 logits, the softmax / top-k / renormalize arithmetic
// follows the operation order of the Triton router kernel
// (moe_fused_gate._router_triton_kernel, softmax, renormalize, 512 experts), the
// quantization follows flashinfer's NVFP4 linear-layout quantizer, and the
// routing tables equal flashinfer moe_sort's output on every entry the GEMMs
// read.
//
// Built with NVRTC: the kernel is a device function, and the extern "C" entries at the
// end instantiate its router GEMM + NVFP4 input quantization (mode kModeQuant), without and
// (up to 8 rows) with the shared expert's gate/up projection.

#include <cuda_bf16.h>

// NVRTC has no <cstdint>
typedef short int16_t;
typedef int int32_t;
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define SGL_DEVICE __forceinline__ __device__

namespace sglang {

using bf16_t = __nv_bfloat16;

namespace device {

template <bool kUsePDL>
SGL_DEVICE void PDLWaitPrimary() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
}

template <bool kUsePDL>
SGL_DEVICE void PDLTriggerSecondary() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
}

}  // namespace device

}  // namespace sglang

namespace sglang {

namespace qwen_router_front {

inline constexpr uint32_t kHidden = FN_HIDDEN;
inline constexpr uint32_t kExperts = FN_EXPERTS;
inline constexpr uint32_t kTopK = FN_TOP_K;
inline constexpr uint32_t kMaxTokens = 16;
inline constexpr uint32_t kThreads = 256;
inline constexpr uint32_t kWarps = kThreads / 32;
// 64 CTAs: leaves SMs free for the shared expert running concurrently on the side stream.
inline constexpr uint32_t kExpertsPerCta = 8;
inline constexpr uint32_t kNumCtas = kExperts / kExpertsPerCta;
inline constexpr uint32_t kSfVec = 16;
inline constexpr uint32_t kBlocksPerRow = kHidden / kSfVec;  // 160; quantizer loops own blocks independently of GEMV warps
static_assert(kHidden % kSfVec == 0, "complete quantization blocks");
static_assert(kExpertsPerCta <= 8, "a CTA's experts form one n8 MMA tile");
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
  uint32_t mode;  // kMode* bits: input quantization, top-k, routing tables
  // Shared-expert gate/up projection (static per-tensor FP8), computed alongside the router GEMV.
  const uint8_t* sh_weight;  // [sh_n, kHidden] e4m3
  const float* sh_in_scale;  // [1] activation scale
  const float* sh_w_scale;   // [1] weight scale
  bf16_t* sh_out;            // [M, sh_n]
  uint32_t sh_n;
};

inline constexpr uint32_t kModeQuant = 1u;
inline constexpr uint32_t kModeTopk = 2u;
inline constexpr uint32_t kModeTables = 4u;

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

template <bool kTrace>
SGL_DEVICE void cycle_stamp(const Params& p, uint32_t slot, uint64_t t0) {
  if constexpr (kTrace) {
    if (threadIdx.x == 0) p.trace[kNumCtas * 8 + slot] = clock64() - t0;
  }
}

// Softmax / top-10 / renormalize for kNT tokens of one warp with a full per-lane
// sort (the exact fallback of topk_tokens_fast). Lane layout of the reference
// kernel: lane holds experts [8l, 8l+8) and [256+8l, 256+8l+8).
template <int kNT, bool kTrace>
SGL_DEVICE void topk_tokens(const Params& p, const uint32_t (&m)[kNT], uint32_t* s_mask, int16_t* s_ids) {
  const uint32_t lane = threadIdx.x % 32;
  constexpr float kLog2e = 1.44269502162933349609375f;
  float a[kNT][16];
  uint4 r[kNT][2];
#pragma unroll
  for (int t = 0; t < kNT; ++t) {
    const bf16_t* row = p.logits + m[t] * kExperts;
    r[t][0] = ld_produced(row + 8 * lane);
    r[t][1] = ld_produced(row + 256 + 8 * lane);
  }
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

// Out of line: rarely taken, and the tail CTA runs this code cold.
__device__ __noinline__ void topk_token_full(const Params& p, uint32_t m, uint32_t* s_mask, int16_t* s_ids) {
  const uint32_t m1[1] = {m};
  topk_tokens<1, false>(p, m1, s_mask, s_ids);
}

SGL_DEVICE void cmpx_desc(uint32_t& a, uint32_t& b) {
  const uint32_t hi = max(a, b), lo = min(a, b);
  a = hi;
  b = lo;
}

// Descending sort of 4 keys (5 comparators).
SGL_DEVICE void sort4_desc(uint32_t& x0, uint32_t& x1, uint32_t& x2, uint32_t& x3) {
  cmpx_desc(x0, x1);
  cmpx_desc(x2, x3);
  cmpx_desc(x0, x2);
  cmpx_desc(x1, x3);
  cmpx_desc(x1, x2);
}

// Top 4 of two descending 4-lists, descending (bitonic split + clean).
SGL_DEVICE void merge_top4_desc(uint32_t (&a)[4], const uint32_t (&b)[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = max(a[i], b[3 - i]);
  cmpx_desc(a[0], a[2]);
  cmpx_desc(a[1], a[3]);
  cmpx_desc(a[0], a[1]);
  cmpx_desc(a[2], a[3]);
}

// Key of a canonical (bf16-exact) fp32 logit: monotonic value bits in the high
// half, (0xFFFF - expert) in the low half, so the max key is the largest logit
// with the lowest expert id on ties. NaN ranks below every number.
SGL_DEVICE uint32_t logit_key(float v, uint32_t expert) {
  const uint32_t u = __float_as_uint(v);
  const uint32_t ord = u ^ (static_cast<uint32_t>(static_cast<int32_t>(u) >> 31) | 0x80000000u);
  return (v == v) ? ((ord & 0xFFFF0000u) | (0xFFFFu - expert)) : (0xFFFFu - expert);
}

SGL_DEVICE float key_value(uint32_t key) {
  const uint32_t hi = key & 0xFFFF0000u;
  return __uint_as_float((hi & 0x80000000u) ? (hi & 0x7FFF0000u) : ~hi & 0xFFFF0000u);
}

// Softmax / top-10 / renormalize for kNT tokens of one warp. Each lane keeps the
// top 4 of its 16 keys; a token where some lane runs out before the last round
// is recomputed with the full per-lane sort (exact either way). Arithmetic on
// the selected values follows the reference kernel's operations and order.
template <int kNT, bool kTrace>
SGL_DEVICE void topk_tokens_fast(
    const Params& p, const uint32_t (&m)[kNT], const bool (&valid)[kNT], uint32_t* s_mask, int16_t* s_ids) {
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
  uint32_t top[kNT][4];
#pragma unroll
  for (int t = 0; t < kNT; ++t) {
    const uint32_t w[8] = {r[t][0].x, r[t][0].y, r[t][0].z, r[t][0].w, r[t][1].x, r[t][1].y, r[t][1].z, r[t][1].w};
    uint32_t k[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      a[t][2 * i] = __fadd_rn(bf16_lo(w[i]), 0.0f);
      a[t][2 * i + 1] = __fadd_rn(bf16_hi(w[i]), 0.0f);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint32_t idx = (i < 8) ? (8 * lane + i) : (256 + 8 * lane + (i - 8));
      k[i] = logit_key(a[t][i], idx);
    }
#pragma unroll
    for (int g = 0; g < 4; ++g) sort4_desc(k[4 * g], k[4 * g + 1], k[4 * g + 2], k[4 * g + 3]);
    uint32_t q0[4] = {k[0], k[1], k[2], k[3]}, q1[4] = {k[4], k[5], k[6], k[7]};
    uint32_t q2[4] = {k[8], k[9], k[10], k[11]}, q3[4] = {k[12], k[13], k[14], k[15]};
    merge_top4_desc(q0, q1);
    merge_top4_desc(q2, q3);
    merge_top4_desc(q0, q2);
#pragma unroll
    for (int i = 0; i < 4; ++i) top[t][i] = q0[i];
  }
  if (top[0][0] != 0x12345678u) cycle_stamp<kTrace>(p, 5, c0);

  float row_max[kNT], row_sum[kNT];
  uint32_t mine[kNT];
  bool bad[kNT];
  uint32_t taken[kNT];
#pragma unroll
  for (int t = 0; t < kNT; ++t) {
    mine[t] = 0;
    bad[t] = false;
    taken[t] = 0;
  }
#pragma unroll
  for (int rr = 0; rr < static_cast<int>(kTopK); ++rr) {
#pragma unroll
    for (int t = 0; t < kNT; ++t) {
      const uint32_t winner = __reduce_max_sync(0xFFFFFFFFu, top[t][0]);
      const bool owner = top[t][0] == winner;
      top[t][0] = owner ? top[t][1] : top[t][0];
      top[t][1] = owner ? top[t][2] : top[t][1];
      top[t][2] = owner ? top[t][3] : top[t][2];
      top[t][3] = owner ? 0u : top[t][3];
      taken[t] += owner ? 1u : 0u;
      bad[t] = bad[t] || (owner && taken[t] == 4u && rr + 1 < static_cast<int>(kTopK));
      mine[t] = (lane == static_cast<uint32_t>(rr)) ? winner : mine[t];
      if (rr == 0) {
        // The largest key is the row max; the reference's exp/sum tree follows.
        row_max[t] = key_value(winner);
        float e[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) e[i] = ex2_approx(__fmul_rn(__fsub_rn(a[t][i], row_max[t]), kLog2e));
        const float lo = __fadd_rn(
            __fadd_rn(__fadd_rn(e[0], e[2]), __fadd_rn(e[4], e[6])),
            __fadd_rn(__fadd_rn(e[8], e[10]), __fadd_rn(e[12], e[14])));
        const float hi = __fadd_rn(
            __fadd_rn(__fadd_rn(e[1], e[3]), __fadd_rn(e[5], e[7])),
            __fadd_rn(__fadd_rn(e[9], e[11]), __fadd_rn(e[13], e[15])));
        row_sum[t] = __fadd_rn(lo, hi);
      }
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
    for (int t = 0; t < kNT; ++t) row_sum[t] = __fadd_rn(row_sum[t], __shfl_xor_sync(0xFFFFFFFFu, row_sum[t], off));
  }
  if (row_sum[0] != 12345.0f) cycle_stamp<kTrace>(p, 6, c0);

#pragma unroll
  for (int t = 0; t < kNT; ++t) {
    if (!valid[t]) continue;
    if (__any_sync(0xFFFFFFFFu, bad[t])) {
      topk_token_full(p, m[t], s_mask, s_ids);
      continue;
    }
    // Lane rr < 10 owns slot rr: its probability, then the reference's 16-slot sum tree.
    const float s = div_full(ex2_approx(__fmul_rn(__fsub_rn(key_value(mine[t]), row_max[t]), kLog2e)), row_sum[t]);
    float w[kTopK];
#pragma unroll
    for (int j = 0; j < static_cast<int>(kTopK); ++j) w[j] = __shfl_sync(0xFFFFFFFFu, s, j);
    const float even = __fadd_rn(
        __fadd_rn(__fadd_rn(w[0], w[2]), __fadd_rn(w[4], w[6])), __fadd_rn(__fadd_rn(w[1], w[3]), __fadd_rn(w[5], w[7])));
    const float routed_sum = __fadd_rn(even, __fadd_rn(w[8], w[9]));
    const float norm = routed_sum > 0.0f ? routed_sum : 1.0f;
    if (lane < kTopK) {
      const uint32_t id = 0xFFFFu - (mine[t] & 0xFFFFu);
      p.topk_weights[m[t] * kTopK + lane] = div_full(s, norm);
      p.topk_ids[m[t] * kTopK + lane] = static_cast<int32_t>(id);
      s_ids[m[t] * kTopK + lane] = static_cast<int16_t>(id);
      atomicOr(&s_mask[id], 1u << m[t]);
    }
  }
  cycle_stamp<kTrace>(p, 7, c0);
}

template <uint32_t kMaxM, bool kTrace>
SGL_DEVICE void topk_and_tables(const Params& p, uint32_t* s_mask, int16_t* s_tile, int16_t* s_ids, uint32_t* s_warp) {
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t M = p.num_tokens;

  for (uint32_t i = threadIdx.x; i < kExperts; i += kThreads) s_mask[i] = 0;
  __syncthreads();
  trace_stamp<kTrace>(p, kNumCtas * 8 + 1);
  if (warp < M) {
    if constexpr (kMaxM > kWarps) {
      // Warp w handles tokens w and w + 8; a missing second token repeats the first unwritten.
      const bool second = warp + kWarps < M;
      const uint32_t m2[2] = {warp, second ? warp + kWarps : warp};
      const bool v2[2] = {true, second};
      topk_tokens_fast<2, kTrace>(p, m2, v2, s_mask, s_ids);
    } else {
      const uint32_t m1[1] = {warp};
      const bool v1[1] = {true};
      topk_tokens_fast<1, kTrace>(p, m1, v1, s_mask, s_ids);
    }
  }
  __syncthreads();
  trace_stamp<kTrace>(p, kNumCtas * 8 + 2);
  if (!(p.mode & kModeTables)) return;

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

// Router GEMV in a fixed accumulation order, bitwise equal to the router GEMM it
// replaces for this shape: K split into two halves, each half a sequential chain of
// 16-wide MMAs in increasing K, the two fp32 halves added, rounded to bf16.
// Per CTA: kExpertsPerCta experts as the n8 tile, tokens as the m16 tile (rows
// >= M repeat token 0); warp 0 runs the chain of the first half, warp 1 the
// second, operands from shared memory.
// Q35_ROUTER_KSPLIT > 2 (numeric): K split into that many chains, one per warp 0 .. KSPLIT - 1, their fp32 partials
// added in split order (another order than cuBLAS's); the CTAs of the token rows quantize after the chains at 8.
#ifndef Q35_ROUTER_KSPLIT
#define Q35_ROUTER_KSPLIT 2
#endif
inline constexpr uint32_t kKSplit = Q35_ROUTER_KSPLIT;
static_assert(kKSplit == 2 || kKSplit == 4 || kKSplit == 8, "router K splits");
static_assert(kHidden % (kKSplit * 16) == 0, "complete MMA K chains");
inline constexpr uint32_t kSplitK = kHidden / kKSplit;
inline constexpr uint32_t kChainSteps = kSplitK / 16;
// Shared-memory rows are padded by 16 B so the 8 rows of an ldmatrix hit distinct banks.
inline constexpr uint32_t kRowBytes = kHidden * 2 + 16;

// Shared gate/up stage: FP8 rows padded by 16 B like the bf16 ones; K split in kShSplits
// chains of m16n8k32 MMAs, one per warp 2 .. 2 + kShSplits - 1.
inline constexpr uint32_t kShRowBytes = kHidden + 16;
inline constexpr uint32_t kShSplits = 4;
inline constexpr uint32_t kShSteps = kHidden / kShSplits / 32;

template <uint32_t kMaxM, bool kShared = false>
struct GemvSmem {
  static constexpr uint32_t kW = 0;
  static constexpr uint32_t kX = kW + kExpertsPerCta * kRowBytes;
  static constexpr uint32_t kPart = kX + kMaxM * kRowBytes;  // fp32 [kKSplit][16][kExpertsPerCta]
  static constexpr uint32_t kShW = kPart + kKSplit * 16 * kExpertsPerCta * 4;
  static constexpr uint32_t kXq = kShW + (kShared ? 8 * kShRowBytes : 0);
  static constexpr uint32_t kShPart = kXq + (kShared ? kMaxM * kShRowBytes : 0);  // fp32 [kShSplits][16][8]
  static constexpr uint32_t kBarW = kShPart + (kShared ? kShSplits * 16 * 8 * 4 : 0);
  static constexpr uint32_t kBarX = kBarW + 8;
  static constexpr uint32_t kBarQ = kBarX + 8;  // e4m3 token rows written (shared stage only)
  static constexpr uint32_t kBytes = kBarQ + (kShared ? 8 : 0);
};

SGL_DEVICE uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

SGL_DEVICE void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

SGL_DEVICE void ldmatrix_x2(uint32_t (&r)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

// One K half of the GEMV for this CTA's experts; writes the fp32 partials to s_part.
template <uint32_t kMaxM>
SGL_DEVICE void gemv_chain(uint32_t half, uint32_t M, const uint8_t* smem, float* s_part) {
  using L = GemvSmem<kMaxM>;
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t j = lane >> 3, r = lane & 7;
  // ldmatrix.x4 for A: matrix j covers tokens (j & 1) * 8 + r and K offset (j >> 1) * 8.
  const uint32_t tok = (j & 1) * 8 + r;
  const uint32_t a_row = tok < M ? tok : 0;
  const uint32_t a_addr = smem_addr(smem + L::kX + a_row * kRowBytes + (half * kSplitK + (j >> 1) * 8) * 2);
  // ldmatrix.x2 for B: matrix (j & 1) covers experts r and K offset (j & 1) * 8.
  const uint32_t b_row = r < kExpertsPerCta ? r : 0;
  const uint32_t b_addr = smem_addr(smem + L::kW + b_row * kRowBytes + (half * kSplitK + (j & 1) * 8) * 2);
  // Fragments are loaded kDepth steps ahead so shared-memory latency stays off the MMA chain.
  constexpr uint32_t kDepth = 4;
  uint32_t a[kDepth][4], b[kDepth][2];
#pragma unroll
  for (uint32_t i = 0; i < kDepth; ++i) {
    ldmatrix_x4(a[i], a_addr + i * 32);
    ldmatrix_x2(b[i], b_addr + i * 32);
  }
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (uint32_t step = 0; step < kChainSteps; ++step) {
    const uint32_t slot = step % kDepth;
    mma_bf16_16816(d, a[slot][0], a[slot][1], a[slot][2], a[slot][3], b[slot][0], b[slot][1]);
    if (step + kDepth < kChainSteps) {
      ldmatrix_x4(a[slot], a_addr + (step + kDepth) * 32);
      ldmatrix_x2(b[slot], b_addr + (step + kDepth) * 32);
    }
  }
  // C fragment: d0/d1 = (token g, experts 2c, 2c+1), d2/d3 = token g + 8.
  const uint32_t g = lane >> 2, c = lane & 3;
  if (c < kExpertsPerCta / 2) {
    float* part = s_part + half * 16 * kExpertsPerCta;
    part[g * kExpertsPerCta + 2 * c] = d[0];
    part[g * kExpertsPerCta + 2 * c + 1] = d[1];
    if constexpr (kMaxM > 8) {
      part[(g + 8) * kExpertsPerCta + 2 * c] = d[2];
      part[(g + 8) * kExpertsPerCta + 2 * c + 1] = d[3];
    }
  }
}

SGL_DEVICE void mma_e4m3_16832(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Two values -> two e4m3 bytes (low byte = first), saturating, round to nearest even.
SGL_DEVICE uint32_t cvt_e4m3x2(float lo, float hi) {
  uint32_t out;
  asm("{\n"
      ".reg .b16 pair;\n"
      "cvt.rn.satfinite.e4m3x2.f32 pair, %2, %1;\n"
      "cvt.u32.u16 %0, pair;\n"
      "}"
      : "=r"(out)
      : "f"(lo), "f"(hi));
  return out;
}

// One K split of the shared gate/up projection for this CTA's 8 rows; fp32 partials to s_part.
template <uint32_t kMaxM>
SGL_DEVICE void shared_chain(uint32_t split, uint32_t M, const uint8_t* smem, float* s_part) {
  using L = GemvSmem<kMaxM, true>;
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t j = lane >> 3, r = lane & 7;
  const uint32_t tok = (j & 1) * 8 + r;
  const uint32_t a_row = tok < M ? tok : 0;
  const uint32_t kbase = split * (kHidden / kShSplits);
  // ldmatrix on byte pairs: a 16 B row segment holds 16 e4m3 values (one m16n8k32 half-K).
  const uint32_t a_addr = smem_addr(smem + L::kXq + a_row * kShRowBytes + kbase + (j >> 1) * 16);
  const uint32_t b_addr = smem_addr(smem + L::kShW + r * kShRowBytes + kbase + (j & 1) * 16);
  constexpr uint32_t kDepth = 4;
  uint32_t a[kDepth][4], b[kDepth][2];
#pragma unroll
  for (uint32_t i = 0; i < kDepth; ++i) {
    ldmatrix_x4(a[i], a_addr + i * 32);
    ldmatrix_x2(b[i], b_addr + i * 32);
  }
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (uint32_t step = 0; step < kShSteps; ++step) {
    const uint32_t slot = step % kDepth;
    mma_e4m3_16832(d, a[slot], b[slot]);
    if (step + kDepth < kShSteps) {
      ldmatrix_x4(a[slot], a_addr + (step + kDepth) * 32);
      ldmatrix_x2(b[slot], b_addr + (step + kDepth) * 32);
    }
  }
  const uint32_t g = lane >> 2, c = lane & 3;
  float* part = s_part + split * 16 * 8;
  part[g * 8 + 2 * c] = d[0];
  part[g * 8 + 2 * c + 1] = d[1];
  if constexpr (kMaxM > 8) {
    part[(g + 8) * 8 + 2 * c] = d[2];
    part[(g + 8) * 8 + 2 * c + 1] = d[3];
  }
}

SGL_DEVICE void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

SGL_DEVICE void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}

SGL_DEVICE void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

SGL_DEVICE void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred done;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
      "@!done bra WAIT_%=;\n"
      "}" ::"r"(bar),
      "r"(parity)
      : "memory");
}

SGL_DEVICE void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
      "l"(src),
      "r"(bytes),
      "r"(bar)
      : "memory");
}

// Warps 2..7 of a CTA with shared rows: quantize the token rows to e4m3 the way
// static_quant_fp8 does (x * (1 / scale), saturate, round to nearest even), then
// warps 2..5 run the gate/up K splits (after all 192 threads arrive on the kBarQ
// mbarrier) while warps 6..7 quantize row b < M to NVFP4.
template <uint32_t kMaxM>
SGL_DEVICE void shared_gate_up(const Params& p, uint32_t M, uint8_t* smem, uint32_t bar_w, uint32_t bar_x) {
  using L = GemvSmem<kMaxM, true>;
  const uint32_t t = threadIdx.x - 64;  // 0 .. 191
  const uint32_t warp = threadIdx.x / 32;
  mbar_wait(bar_x, 0);
  const float inv = div_full(1.0f, p.sh_in_scale[0]);
  for (uint32_t i = t; i < M * (kHidden / 16); i += kThreads - 64) {
    const uint32_t row = i / (kHidden / 16), blk = i % (kHidden / 16);
    const uint4* src = reinterpret_cast<const uint4*>(smem + L::kX + row * kRowBytes + blk * 32);
    const uint4 lo = src[0], hi = src[1];
    const uint32_t w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    uint32_t q[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float a0 = fminf(fmaxf(__fmul_rn(bf16_lo(w[2 * k]), inv), -448.0f), 448.0f);
      const float a1 = fminf(fmaxf(__fmul_rn(bf16_hi(w[2 * k]), inv), -448.0f), 448.0f);
      const float a2 = fminf(fmaxf(__fmul_rn(bf16_lo(w[2 * k + 1]), inv), -448.0f), 448.0f);
      const float a3 = fminf(fmaxf(__fmul_rn(bf16_hi(w[2 * k + 1]), inv), -448.0f), 448.0f);
      q[k] = cvt_e4m3x2(a0, a1) | (cvt_e4m3x2(a2, a3) << 16);
    }
    *reinterpret_cast<uint4*>(smem + L::kXq + row * kShRowBytes + blk * 16) = make_uint4(q[0], q[1], q[2], q[3]);
  }
  const uint32_t bar_q = smem_addr(smem + L::kBarQ);
  mbar_arrive(bar_q);
  if (warp < 2 + kShSplits) {
    mbar_wait(bar_q, 0);
    mbar_wait(bar_w, 0);
    shared_chain<kMaxM>(warp - 2, M, smem, reinterpret_cast<float*>(smem + L::kShPart));
  } else if ((p.mode & kModeQuant) && blockIdx.x < M) {
    const uint8_t* row = smem + L::kX + blockIdx.x * kRowBytes;
    for (uint32_t blk = threadIdx.x - 64 - 32 * kShSplits; blk < kBlocksPerRow; blk += kThreads - 64 - 32 * kShSplits) {
      const uint4* src = reinterpret_cast<const uint4*>(row + blk * kSfVec * 2);
      uint32_t sf;
      uint2 packed;
      quantize_block(src[0], src[1], p.global_scale[0], sf, packed);
      *reinterpret_cast<uint2*>(p.x_fp4 + static_cast<size_t>(blockIdx.x) * (kHidden / 2) + blk * 8) = packed;
      p.x_sf[static_cast<size_t>(blockIdx.x) * kBlocksPerRow + blk] = static_cast<uint8_t>(sf);
    }
  }
}

template <uint32_t kMaxM, bool kUsePDL, bool kTrace, bool kShared = false>
SGL_DEVICE void router_front_kernel(const Params& p) {
  using L = GemvSmem<kMaxM, kShared>;
  static_assert(!kShared || kMaxM <= 8, "the shared gate/up stage fits shared memory only for M <= 8");
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ uint32_t s_mask[kExperts];
  __shared__ int16_t s_tile[kExperts];
  __shared__ int16_t s_ids[kMaxTokens * kTopK];
  __shared__ uint32_t s_warp[kWarps];
  __shared__ uint32_t s_last;

  const uint32_t warp = threadIdx.x / 32;
  const uint32_t expert0 = blockIdx.x * kExpertsPerCta;
  // token tile blockIdx.y: rows [t0, t0 + M) of x, logits and the NVFP4 input (one tile without route_gemm)
  const uint32_t t0 = blockIdx.y * kMaxM;
  const uint32_t M = min(p.num_tokens - t0, kMaxM);
  const uint32_t tb = blockIdx.x * 8;
  float* s_part = reinterpret_cast<float*>(smem + L::kPart);
  const uint32_t bar_w = smem_addr(smem + L::kBarW);
  const uint32_t bar_x = smem_addr(smem + L::kBarX);
  trace_stamp<kTrace>(p, tb + 0);

  // This CTA's 8 rows of the shared gate/up projection, when it has any.
  const uint32_t sh_row0 = blockIdx.x * 8;
  const bool has_sh = kShared && sh_row0 < p.sh_n;

  // Router (and shared) weights are static: their bulk copies are issued before waiting on the producer grid.
  if (threadIdx.x == 0) {
    mbar_init(bar_w, 1);
    mbar_init(bar_x, 1);
    if (has_sh) mbar_init(smem_addr(smem + L::kBarQ), kThreads - 64);
    mbar_expect_tx(bar_w, kExpertsPerCta * kHidden * 2 + (has_sh ? 8 * kHidden : 0));
#pragma unroll
    for (uint32_t e = 0; e < kExpertsPerCta; ++e) {
      bulk_load(smem_addr(smem + L::kW + e * kRowBytes), p.weight + static_cast<size_t>(expert0 + e) * kHidden,
                kHidden * 2, bar_w);
    }
    if (has_sh) {
#pragma unroll
      for (uint32_t r = 0; r < 8; ++r) {
        bulk_load(smem_addr(smem + L::kShW + r * kShRowBytes), p.sh_weight + static_cast<size_t>(sh_row0 + r) * kHidden,
                  kHidden, bar_w);
      }
    }
  }
  __syncthreads();

  device::PDLWaitPrimary<kUsePDL>();
  trace_stamp<kTrace>(p, tb + 1);
  if (!(p.mode & kModeTopk)) {
    // The dependent grid reads our outputs only after its own griddepcontrol.wait.
    device::PDLTriggerSecondary<kUsePDL>();
  }
  if (threadIdx.x == 0) {
    mbar_expect_tx(bar_x, M * kHidden * 2);
#pragma unroll
    for (uint32_t row = 0; row < kMaxM; ++row) {
      if (row < M) {
        bulk_load(smem_addr(smem + L::kX + row * kRowBytes), p.x + static_cast<size_t>(t0 + row) * kHidden, kHidden * 2,
                  bar_x);
      }
    }
  }

  if (warp < kKSplit) {
    mbar_wait(bar_w, 0);
    mbar_wait(bar_x, 0);
    trace_stamp<kTrace>(p, tb + 2);
    gemv_chain<kMaxM>(warp, M, smem, s_part);
  } else if (has_sh) {
    shared_gate_up<kMaxM>(p, M, smem, bar_w, bar_x);
  } else if ((p.mode & kModeQuant) && blockIdx.x < M) {
    // CTA b < M quantizes token row b while warps 0 .. kKSplit - 1 run the GEMV chains.
    mbar_wait(bar_x, 0);
    const uint8_t* row = smem + L::kX + blockIdx.x * kRowBytes;
    for (uint32_t blk = threadIdx.x - 32 * kKSplit; blk < kBlocksPerRow; blk += kThreads - 32 * kKSplit) {
      const uint4* src = reinterpret_cast<const uint4*>(row + blk * kSfVec * 2);
      uint32_t sf;
      uint2 packed;
      quantize_block(src[0], src[1], p.global_scale[0], sf, packed);
      *reinterpret_cast<uint2*>(p.x_fp4 + static_cast<size_t>(t0 + blockIdx.x) * (kHidden / 2) + blk * 8) = packed;
      p.x_sf[static_cast<size_t>(t0 + blockIdx.x) * kBlocksPerRow + blk] = static_cast<uint8_t>(sf);
    }
  }
  __syncthreads();
  trace_stamp<kTrace>(p, tb + 3);
  if (threadIdx.x < M * kExpertsPerCta) {
    const uint32_t m = threadIdx.x / kExpertsPerCta;
    const uint32_t e = threadIdx.x % kExpertsPerCta;
    float acc = s_part[m * kExpertsPerCta + e];
#pragma unroll
    for (uint32_t k = 1; k < kKSplit; ++k) acc = __fadd_rn(acc, s_part[(k * 16 + m) * kExpertsPerCta + e]);
    p.logits[(t0 + m) * kExperts + expert0 + e] = __float2bfloat16_rn(acc);
  }
  if (kKSplit == 8 && !kShared && (p.mode & kModeQuant) && blockIdx.x < M) {
    // every warp ran a GEMV chain: CTA b < M quantizes token row b now
    const uint8_t* row = smem + L::kX + blockIdx.x * kRowBytes;
    for (uint32_t blk = threadIdx.x; blk < kBlocksPerRow; blk += kThreads) {
      const uint4* src = reinterpret_cast<const uint4*>(row + blk * kSfVec * 2);
      uint32_t sf;
      uint2 packed;
      quantize_block(src[0], src[1], p.global_scale[0], sf, packed);
      *reinterpret_cast<uint2*>(p.x_fp4 + static_cast<size_t>(t0 + blockIdx.x) * (kHidden / 2) + blk * 8) = packed;
      p.x_sf[static_cast<size_t>(t0 + blockIdx.x) * kBlocksPerRow + blk] = static_cast<uint8_t>(sf);
    }
  }
  if (has_sh && threadIdx.x < M * 8) {
    const uint32_t m = threadIdx.x / 8;
    const uint32_t n = threadIdx.x % 8;
    const float* sp = reinterpret_cast<const float*>(smem + L::kShPart);
    float acc = sp[m * 8 + n];
#pragma unroll
    for (uint32_t k = 1; k < kShSplits; ++k) acc = __fadd_rn(acc, sp[(k * 16 + m) * 8 + n]);
    const float alpha = __fmul_rn(p.sh_in_scale[0], p.sh_w_scale[0]);
    p.sh_out[static_cast<size_t>(m) * p.sh_n + sh_row0 + n] = __float2bfloat16_rn(__fmul_rn(acc, alpha));
  }
  if (!(p.mode & kModeTopk)) return;

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
  // The dependent grid reads our outputs only after its own griddepcontrol.wait,
  // which waits for this grid to complete; triggering here lets it launch during the tail.
  device::PDLTriggerSecondary<kUsePDL>();
  if (s_last) {
    trace_stamp<kTrace>(p, kNumCtas * 8 + 0);
    topk_and_tables<kMaxM, kTrace>(p, s_mask, s_tile, s_ids, s_warp);
  }
}


}  // namespace qwen_router_front

}  // namespace sglang

// Router logits and the NVFP4 input of up to Q35_ROUTER_MAX_M rows (1, 8 or 16, set at
// compile time) in one launch, with PDL.
extern "C" __global__ __launch_bounds__(sglang::qwen_router_front::kThreads, 1) void q38_router_gemm_quant(
    const __grid_constant__ sglang::qwen_router_front::Params p) {
  sglang::qwen_router_front::router_front_kernel<Q35_ROUTER_MAX_M, true, false, false>(p);
}

