// Portions adapted from https://github.com/moonmath-ai/amd-kernels (MIT License).
//
// Copyright (c) 2026 The MoonMath.ai
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/// Kimi-K3 routed-expert grouped GEMMs for gfx950: MXFP8 activations x MXFP4 weights (A8W4) on the
/// scaled MFMA (v_mfma_scale_f32_16x16x128_f8f6f4), reading AITER's preshuffled K3 weights in place.
///
/// Structure ported from moonmath-ai/amd-kernels csrc/gfx942/mxfp4_moe_{gateup,down}.hip (MIT):
/// weights stream from global straight into MFMA operand registers and never touch LDS; in gate/up
/// every steady-state load is inline asm and every vmcnt wait is a literal placed by hand.
///
///   gateup: inter[p] = MXFP8(SiTUv2(x[tok(p)] @ Wg[e]^T, x[tok(p)] @ Wu[e]^T))   p = sorted row
///   down:   y[p] = bf16(inter[p] @ W2[e]^T), y as [N / 512][sorted rows][512]
///   reduce: out[tok] = bf16(sum_slot w[tok][slot] * y[pos[tok * topk + slot]])   (fp32 sum)
///
/// Operand layouts of the 16x16x128 scaled MFMA, as used by AITER's FlyDSL A8W4 MoE:
///   weights (A, fp4): lane l holds row l%16, k 32*(l/16) .. +31, 16 contiguous bytes
///   activations (B, fp8): lane l holds column l%16, k 16*(l/16) .. +15 and 64 + 16*(l/16) .. +15
///   scales: lane l supplies the E8M0 of its row/column and K block l/16; opsel picks the byte
///   result: lane l holds C[4*(l/16) + v][l%16], v = 0..3 -> 4 consecutive weight rows of one token

#pragma once

#ifndef USE_ROCM
#error "k3_a8w4_gfx950.cuh targets gfx950: it uses the f8f6f4 scaled MFMA"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <stdint.h>
#include <type_traits>
#include <utility>

namespace sglang::k3_a8w4 {

typedef unsigned int u32;
typedef unsigned short u16;
typedef u32 u32x2 __attribute__((ext_vector_type(2)));
typedef u32 u32x4 __attribute__((ext_vector_type(4)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

#define K3A8W4_DEV __device__ __forceinline__

constexpr int kKStep = 128;             // k of one MFMA
constexpr int kTileBytes = 1024;        // one 16-row x 128-k MXFP4 tile in AITER's (16, 16) preshuffle
constexpr u32 kSrdWord3 = 0x00020000u;  // raw 32-bit data format, as AITER/CK use on gfx9
constexpr u32 kOffDrop = 0xFFFFFFFFu;   // an offset every SRD range check rejects

// ---------------------------------------------------------------------------------------------
// Buffer-resource memory ops. A voffset at or past num_records loads 0 or drops the store, so
// padded rows cost no branch. The soffset is added to the address but not range-checked.
// ---------------------------------------------------------------------------------------------
K3A8W4_DEV __amdgpu_buffer_rsrc_t make_rsrc(const void* p, uint64_t bytes) {
  const u32 n = bytes > 0xFFFFFFFFull ? 0xFFFFFFFFu : (u32)bytes;
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), 0, n, kSrdWord3);
}

K3A8W4_DEV u32x4 ld_b128(__amdgpu_buffer_rsrc_t r, u32 voff, u32 soff) {
  u32x4 v;
  asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen"  // codespell:ignore offen

               : "=v"(v)
               : "v"(voff), "s"(r), "s"(soff)
               : "memory");
  return v;
}

K3A8W4_DEV u32 ld_b32(__amdgpu_buffer_rsrc_t r, u32 voff, u32 soff) {
  u32 v;
  asm volatile("buffer_load_dword %0, %1, %2, %3 offen"  // codespell:ignore offen

               : "=v"(v)
               : "v"(voff), "s"(r), "s"(soff)
               : "memory");
  return v;
}

K3A8W4_DEV void st_b32(__amdgpu_buffer_rsrc_t r, u32 v, u32 voff) {
  asm volatile("buffer_store_dword %0, %1, %2, 0 offen"  // codespell:ignore offen
               ::"v"(v),
               "v"(voff),
               "s"(r)
               : "memory");
}

K3A8W4_DEV void st_b8(__amdgpu_buffer_rsrc_t r, u32 v, u32 voff) {
  asm volatile("buffer_store_byte %0, %1, %2, 0 offen"  // codespell:ignore offen
               ::"v"(v),
               "v"(voff),
               "s"(r)
               : "memory");
}

template <int N>
K3A8W4_DEV void wait_vmcnt() {
  static_assert(N >= 0 && N <= 63, "vmcnt is a 6-bit field");
  asm volatile("s_waitcnt vmcnt(%c0)" ::"i"(N) : "memory");
}

// vmcnt waits that name the register they guard as a read-write operand, so no use of it can be
// scheduled above the wait. Repeating one for each register of a group costs a satisfied
// s_waitcnt apiece.
template <int N>
K3A8W4_DEV void guard(u32x4& v) {
  asm volatile("s_waitcnt vmcnt(%c1)" : "+v"(v) : "n"(N) : "memory");
}

template <int N>
K3A8W4_DEV void guard(u32& v) {
  asm volatile("s_waitcnt vmcnt(%c1)" : "+v"(v) : "n"(N) : "memory");
}

// Only the LDS half of __syncthreads: the weight stream keeps flying through it.
K3A8W4_DEV void lds_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

K3A8W4_DEV u32 to_sgpr(u32 v) {
  return __builtin_amdgcn_readfirstlane(v);
}

// MI300-class parts deal workgroups to their eight XCDs round-robin. Re-linearising gives each
// die a contiguous run of blocks, so one expert's m-blocks share one L2. A bijection on [0, live).
K3A8W4_DEV int xcd_linear(int live) {
  constexpr u32 kXcd = 8;
  const u32 lin = blockIdx.x;
  if (lin >= (u32)live) return -1;
  const u32 per = (u32)live / kXcd, rem = (u32)live % kXcd;
  const u32 die = lin % kXcd, seq = lin / kXcd;
  return (int)(die * per + (die < rem ? die : rem) + seq);
}

// Weights are the MFMA A operand (fp4, cbsz 4), activations the B operand (fp8, blgp 0).
template <int OPSEL_W, int OPSEL_A>
K3A8W4_DEV f32x4 mfma_a8w4(u32x4 w, u32x4 a_lo, u32x4 a_hi, f32x4 c, u32 scale_w, u32 scale_a) {
  const i32x4 w4 = __builtin_bit_cast(i32x4, w);
  const i32x8 wv = __builtin_shufflevector(w4, w4, 0, 1, 2, 3, -1, -1, -1, -1);
  const i32x4 lo = __builtin_bit_cast(i32x4, a_lo), hi = __builtin_bit_cast(i32x4, a_hi);
  const i32x8 a8 = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
  return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
      wv, a8, c, 4, 0, OPSEL_W, (int)scale_w, OPSEL_A, (int)scale_a);
}

// ---------------------------------------------------------------------------------------------
// Numerics shared with AITER's A8W4 SiTUv2 path (sgl-amd-mono's situ_aiter matches it).
// ---------------------------------------------------------------------------------------------
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kFp8Max = 448.0f;

K3A8W4_DEV float tanh_aiter(float x) {
  const float e = __builtin_amdgcn_exp2f(__builtin_fabsf(x) * (float)(-2.0 * 1.4426950408889634));
  const float t = (1.0f - e) * __builtin_amdgcn_rcpf(1.0f + e);
  return x > 0.0f ? t : -t;
}

K3A8W4_DEV float situ_v2(float g, float u, float beta, float inv_beta, float lbeta, float inv_lbeta) {
  const float sg = __builtin_amdgcn_rcpf(1.0f + __builtin_amdgcn_exp2f(g * -kLog2e));
  const float a = (beta * tanh_aiter(g * inv_beta)) * sg;
  return (a * lbeta) * tanh_aiter(u * inv_lbeta);
}

// Biased exponent of the smallest power of two >= v, clamped to [1, 254].
K3A8W4_DEV u32 e8m0_ceil(float v) {
  const u32 b = __builtin_bit_cast(u32, v);
  int e = (int)((b >> 23) & 0xFF) + ((b & 0x7FFFFF) != 0 ? 1 : 0);
  e = e < 1 ? 1 : (e > 254 ? 254 : e);
  return (u32)e;
}

K3A8W4_DEV float rcp_pow2(u32 code) {
  return __builtin_bit_cast(float, (254u - code) << 23);
}

K3A8W4_DEV float clamp_fp8(float v) {
  return __builtin_fminf(__builtin_fmaxf(v, -kFp8Max), kFp8Max);
}

// Four in-range f32 -> four OCP E4M3 bytes, RNE.
K3A8W4_DEV u32 fp8x4(float a, float b, float c, float d) {
  const int lo = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
  return (u32)__builtin_amdgcn_cvt_pk_fp8_f32(c, d, lo, true);
}

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef float f32x2 __attribute__((ext_vector_type(2)));

// v_cvt_pk_bf16_f32: RNE under the default round mode.
K3A8W4_DEV u32 bf16x2_rne(float lo, float hi) {
  return __builtin_bit_cast(u32, __builtin_convertvector((f32x2){lo, hi}, bf16x2));
}

// =============================================================================================
// gate/up: one workgroup = BM sorted rows of one expert x BN inter columns (gate and up both).
// Wave w owns NT consecutive 16-column tiles; NT is even so a wave holds whole 32-column MXFP8
// groups of the output and quantizes them without leaving registers.
// =============================================================================================
struct GateUpParams {
  const uint8_t* a;        // [T, K] e4m3
  const uint8_t* a_scale;  // [T, K/32] e8m0
  const uint8_t* w;        // [E, 2I, K/2]: AITER shuffle_weight(is_guinterleave, gate_up=True)
  const uint8_t* w_scale;  // [E * 2I, K/32]: AITER shuffle_scale(is_guinterleave, gate_up=True)
  const int* sorted_ids;   // (slot << 24) | token, padding = token >= T
  const int* expert_ids;   // per BM block
  const int* num_valid;    // [0]: sorted rows including padding
  uint8_t* inter;          // [max_sorted, I] e4m3
  uint8_t* inter_scale;    // [max_sorted, I/32] e8m0
  int T, K, I, E, max_blocks;
  float beta, inv_beta, lbeta, inv_lbeta;
};

template <int BM, int NW, int NT>
struct GateUpCfg {
  static_assert(BM % 16 == 0 && NT % 2 == 0, "whole row tiles, whole 32-column quant groups");
  static constexpr int NTHR = NW * 64;
  static constexpr int RT = BM / 16;
  static constexpr int BN = NW * NT * 16;
  static constexpr int CHUNKS = BM * (kKStep / 16);  // 16-byte A chunks per k-step slab
  static constexpr int GPT = (CHUNKS + NTHR - 1) / NTHR;
  static constexpr int A_OPS = GPT;
  static constexpr int B_OPS = NT * 3;  // gate + up granule and one scale dword per column tile
  static constexpr int SLAB_OPS = A_OPS + B_OPS;
  static constexpr int SLAB_BYTES = BM * kKStep;
  static_assert(SLAB_OPS + B_OPS <= 63, "vmcnt rungs");
};

// LDS byte offset of 16-byte chunk c of row r in a 128-byte-per-row slab. XOR on the row's low
// bits spreads the 16 rows an MFMA operand read touches over distinct bank groups.
K3A8W4_DEV int a_slab_off(int r, int c) {
  return r * kKStep + ((c ^ (r & 7)) << 4);
}

template <int BM, int NW, int NT>
__global__ __launch_bounds__(NW * 64, 1) void gateup_kernel(const GateUpParams p) {
  using C = GateUpCfg<BM, NW, NT>;
  constexpr int RT = C::RT, GPT = C::GPT;

  const int tid = threadIdx.x, lane = tid & 63, wave = (int)to_sgpr((u32)tid >> 6);
  const int lrow = lane & 15, lgrp = lane >> 4;
  const int nblk_n = p.I / C::BN;
  // Metadata the sort did not fill reads as a block past the grid or an expert past E: skip it.
  const int live_m = min((p.num_valid[0] + BM - 1) / BM, p.max_blocks);
  const int lin = xcd_linear(live_m * nblk_n);
  if (lin < 0) return;
  const int pid_m = lin / nblk_n;
  const int pid_n = lin - pid_m * nblk_n;
  const int expert = p.expert_ids[pid_m];
  if (expert < 0 || expert >= p.E) return;

  const int KS = p.K / kKStep;
  extern __shared__ __attribute__((aligned(16))) uint8_t lds[];
  uint8_t* a_lds = lds;                                           // [2][BM][128]
  u32* as_lds = reinterpret_cast<u32*>(lds + 2 * C::SLAB_BYTES);  // [KS/4][BM][4]

  // A scales, once: dword (q, row, j) holds K blocks 4*(4q + i) + j for i = 0..3, so a lane
  // reads one dword per four k-steps and the k-step picks the byte through opsel.
  {
    const int ngrp = KS / 4;
    for (int idx = tid; idx < ngrp * BM; idx += C::NTHR) {
      const int q = idx / BM, row = idx - q * BM;
      const int tok = p.sorted_ids[pid_m * BM + row] & 0xFFFFFF;
      u32x4 d = {0, 0, 0, 0};
      if (tok < p.T) d = *reinterpret_cast<const u32x4*>(p.a_scale + (size_t)tok * (p.K / 32) + q * 16);
      u32x4 o;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        o[j] = ((d[0] >> (8 * j)) & 0xFF) | (((d[1] >> (8 * j)) & 0xFF) << 8) | (((d[2] >> (8 * j)) & 0xFF) << 16) |
               (((d[3] >> (8 * j)) & 0xFF) << 24);
      *reinterpret_cast<u32x4*>(as_lds + (q * BM + row) * 4) = o;
    }
  }

  // A staging: thread owns chunk (tid + g * NTHR) of every slab.
  int st_lds[GPT];
  u32 st_voff[GPT];
  bool st_live[GPT];
#pragma unroll
  for (int g = 0; g < GPT; ++g) {
    const int c = tid + g * C::NTHR;
    st_live[g] = c < C::CHUNKS;
    const int row = st_live[g] ? c / 8 : 0, ch = c % 8;
    const int tok = p.sorted_ids[pid_m * BM + row] & 0xFFFFFF;
    st_voff[g] = (st_live[g] && tok < p.T) ? (u32)tok * (u32)p.K + (u32)(ch * 16) : kOffDrop;
    st_lds[g] = a_slab_off(row, ch);
  }

  const auto a_rsrc = make_rsrc(p.a, (uint64_t)p.T * p.K);
  const uint64_t w_bytes = (uint64_t)2 * p.I * (p.K / 2);
  const uint64_t ws_bytes = (uint64_t)2 * p.I * (p.K / 32);
  const auto w_rsrc = make_rsrc(p.w + (size_t)expert * w_bytes, w_bytes);
  const auto ws_rsrc = make_rsrc(p.w_scale + (size_t)expert * ws_bytes, ws_bytes);

  // Column tile n of the expert: gate block (n, 0) and up block (n, 1), each KS tiles of 1 KB;
  // its scale dwords: (n, k-step pair) blocks of 64 lanes.
  const int ntile0 = pid_n * (C::BN / 16) + wave * NT;
  u32 b_voff[NT][2], s_voff[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
#pragma unroll
    for (int gu = 0; gu < 2; ++gu)
      b_voff[t][gu] = (u32)(((ntile0 + t) * 2 + gu) * KS) * kTileBytes + lane * 16;
    s_voff[t] = (u32)((ntile0 + t) * (KS / 2)) * 256 + lane * 4;
  }

  f32x4 acc[RT][NT][2];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int t = 0; t < NT; ++t)
      acc[r][t][0] = acc[r][t][1] = (f32x4){0.f, 0.f, 0.f, 0.f};

  u32x4 a_ring[2][GPT];
  u32x4 b_ring[2][NT][2];
  u32 s_ring[2][NT];

  auto issue_a = [&](int slot, int slab) {
    const u32 cur = to_sgpr((u32)slab * kKStep);
#pragma unroll
    for (int g = 0; g < GPT; ++g)
      a_ring[slot][g] = ld_b128(a_rsrc, st_voff[g], cur);
  };
  auto issue_b = [&](int slot, int slab) {
    const u32 bcur = to_sgpr((u32)slab * kTileBytes);
    const u32 scur = to_sgpr((u32)(slab >> 1) * 256);
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      b_ring[slot][t][0] = ld_b128(w_rsrc, b_voff[t][0], bcur);
      b_ring[slot][t][1] = ld_b128(w_rsrc, b_voff[t][1], bcur);
      s_ring[slot][t] = ld_b32(ws_rsrc, s_voff[t], scur);
    }
  };
  auto stage_a = [&](int slot) {
#pragma unroll
    for (int g = 0; g < GPT; ++g)
      if (st_live[g]) *reinterpret_cast<u32x4*>(a_lds + slot * C::SLAB_BYTES + st_lds[g]) = a_ring[slot][g];
  };

  auto wait_b = [&](auto slot_tag) {
    constexpr int S = decltype(slot_tag)::value;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      guard<C::SLAB_OPS>(b_ring[S][t][0]);
      guard<C::SLAB_OPS>(b_ring[S][t][1]);
      guard<C::SLAB_OPS>(s_ring[S][t]);
    }
  };
  auto wait_a = [&](auto slot_tag, auto cnt_tag) {
    constexpr int S = decltype(slot_tag)::value;
    constexpr int N = decltype(cnt_tag)::value;
#pragma unroll
    for (int g = 0; g < GPT; ++g)
      guard<N>(a_ring[S][g]);
  };

  u32 asc[RT];
  // One k-step: slab s is in LDS slot s % 2 and B ring slot s % 2. Publishes A(s + 1) and
  // issues A(s + 2) / B(s + 2) into the slots this step frees.
  auto body = [&](auto slot_tag, auto q_tag, int s) {
    constexpr int S = decltype(slot_tag)::value;
    constexpr int Q = decltype(q_tag)::value;
    wait_b(slot_tag);
    if constexpr (Q == 0) {
#pragma unroll
      for (int r = 0; r < RT; ++r)
        asc[r] = as_lds[((s >> 2) * BM + r * 16 + lrow) * 4 + lgrp];
    }
    const uint8_t* slab = a_lds + S * C::SLAB_BYTES;
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      const int row = r * 16 + lrow;
      const u32x4 lo = *reinterpret_cast<const u32x4*>(slab + a_slab_off(row, lgrp));
      const u32x4 hi = *reinterpret_cast<const u32x4*>(slab + a_slab_off(row, lgrp + 4));
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        acc[r][t][0] = mfma_a8w4<S * 2 + 0, Q>(b_ring[S][t][0], lo, hi, acc[r][t][0], s_ring[S][t], asc[r]);
        acc[r][t][1] = mfma_a8w4<S * 2 + 1, Q>(b_ring[S][t][1], lo, hi, acc[r][t][1], s_ring[S][t], asc[r]);
      }
    }
    wait_a(std::integral_constant<int, S ^ 1>{}, std::integral_constant<int, C::B_OPS>{});
    stage_a(S ^ 1);
    // The last two steps re-read slab KS - 1, so every step issues the same ops and no cursor
    // addresses past K: the k-step rides the soffset, which the SRD range check is not relied on for.
    const int next = s + 2 < KS ? s + 2 : KS - 1;
    issue_a(S, next);
    issue_b(S, next);
    lds_barrier();
  };

  // Prime. Issue order must match the body's (A then B per slab): the rungs are FIFO positions.
  issue_a(0, 0);
  issue_b(0, 0);
  issue_a(1, 1);
  issue_b(1, 1);
  wait_a(std::integral_constant<int, 0>{}, std::integral_constant<int, C::B_OPS + C::SLAB_OPS>{});
  stage_a(0);
  __syncthreads();

  for (int s = 0; s < KS; s += 4) {
    body(std::integral_constant<int, 0>{}, std::integral_constant<int, 0>{}, s);
    body(std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{}, s + 1);
    body(std::integral_constant<int, 0>{}, std::integral_constant<int, 2>{}, s + 2);
    body(std::integral_constant<int, 1>{}, std::integral_constant<int, 3>{}, s + 3);
  }
  // Drain naming every ring register: the last re-reads are dead values, and without a use after
  // their landing the allocator hands their registers to the epilogue while the loads are in flight.
#pragma unroll
  for (int sl = 0; sl < 2; ++sl) {
#pragma unroll
    for (int g = 0; g < GPT; ++g)
      guard<0>(a_ring[sl][g]);
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      guard<0>(b_ring[sl][t][0]);
      guard<0>(b_ring[sl][t][1]);
      guard<0>(s_ring[sl][t]);
    }
  }

  // Epilogue: SiTUv2, then MXFP8 per 32 inter columns (amax over the wave's tile pair, which
  // lanes l, l^16, l^32, l^48 share), as AITER's stage-1 fp8 output quant.
  const auto i_rsrc = make_rsrc(p.inter, (uint64_t)live_m * BM * p.I);
  const auto is_rsrc = make_rsrc(p.inter_scale, (uint64_t)live_m * BM * (p.I / 32));
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    const u32 prow = (u32)(pid_m * BM + r * 16 + lrow);
#pragma unroll
    for (int g = 0; g < NT / 2; ++g) {
      float h[8];
#pragma unroll
      for (int tt = 0; tt < 2; ++tt)
#pragma unroll
        for (int v = 0; v < 4; ++v)
          h[tt * 4 + v] =
              situ_v2(acc[r][2 * g + tt][0][v], acc[r][2 * g + tt][1][v], p.beta, p.inv_beta, p.lbeta, p.inv_lbeta);
      float amax = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        amax = __builtin_fmaxf(amax, __builtin_fabsf(h[i]));
      amax = __builtin_fmaxf(amax, __shfl_xor(amax, 16));
      amax = __builtin_fmaxf(amax, __shfl_xor(amax, 32));
      const u32 code = e8m0_ceil(__builtin_fmaxf(amax, 1e-10f) * (1.0f / kFp8Max));
      const float inv = rcp_pow2(code);
      const int col = (ntile0 + 2 * g) * 16 + 4 * lgrp;
#pragma unroll
      for (int tt = 0; tt < 2; ++tt) {
        const u32 q = fp8x4(
            clamp_fp8(h[tt * 4 + 0] * inv),
            clamp_fp8(h[tt * 4 + 1] * inv),
            clamp_fp8(h[tt * 4 + 2] * inv),
            clamp_fp8(h[tt * 4 + 3] * inv));
        st_b32(i_rsrc, q, prow * (u32)p.I + (u32)(col + tt * 16));
      }
      if (lgrp == 0) st_b8(is_rsrc, code, prow * (u32)(p.I / 32) + (u32)((ntile0 + 2 * g) / 2));
    }
  }
}

// =============================================================================================
// down, weight-stationary: one workgroup = up to seg_blocks BM-row blocks of one expert x NW * 64 output
// columns. Each wave holds its 64 columns of W2 over all of K in registers, read from HBM once per item,
// and walks the item's blocks with A double-buffered through LDS. A wave writes 128 contiguous bytes
// of each row.
// Memory ops here are compiler-visible builtins: the loop has no weight stream to pipeline by hand.
// =============================================================================================
struct DownParams {
  const uint8_t* inter;        // [max_sorted, I] e4m3
  const uint8_t* inter_scale;  // [max_sorted, I/32] e8m0
  const uint8_t* w;            // [E, N, I/2]: AITER shuffle_weight(is_guinterleave, gate_up=False)
  const uint8_t* w_scale;      // [E * N, pad8(I/32)]: AITER shuffle_scale(is_guinterleave, False)
  const int* sorted_ids;
  const int* expert_ids;
  const int* num_valid;
  int E, max_blocks;
  uint16_t* y;  // [N / BN][y_rows][BN] bf16, sorted rows
  int* pos;     // [T * topk]: sorted row of (tok, slot), for the reduce
  int y_rows;   // = sorted_ids length
  // work list from down_items_kernel: int4 {items, 0, 0, 0}, then per item
  // int4 {expert, first block, end block, 0} covering at most seg_blocks blocks of one expert
  const int4* items;
  int seg_blocks, max_items;
  int T, topk, N, I;
};

// y keeps sorted rows split by the workgroups' column chunks, [N / BN][rows][BN], so a workgroup writes each
// block as one contiguous run; a per-wave LDS transpose makes each store 8 rows x 128 B of whole lines.
template <int BM, int NW, int KS>
struct DownWsCfg {
  static constexpr int NTHR = NW * 64;
  static constexpr int RT = BM / 16;
  static constexpr int NTILE = 4;  // 16-column tiles a wave owns
  static constexpr int BN = NW * NTILE * 16;
  static constexpr int K1 = (KS + 1) / 2;
  static constexpr int A_BYTES = BM * KS * kKStep;  // the block's inter tile, [s][row][128]
  // A arrives by LDS DMA in 1 KB wave loads (64 lanes x 16 B = 8 rows of one k-step)
  static constexpr int A_WLOADS = A_BYTES / 1024;
  static constexpr int WLOADS_PER_WAVE = (A_WLOADS + NW - 1) / NW;
  static constexpr int SLOT_BYTES = A_BYTES + BM * 16 + BM * 4;  // + scale dwords + output row offsets
  static constexpr int XPOSE_BYTES = NW * 16 * NTILE * 16 * 2;   // a wave's 16 rows x 64 bf16
  static constexpr int LDS_BYTES = 2 * SLOT_BYTES + XPOSE_BYTES;
  // whole-line dwordx4 stores a wave issues per block, two per row tile
  static constexpr int STORES = RT * 2;
  static_assert(BM % 8 == 0 && BM <= NTHR, "whole 8-row wave loads, one thread a row");
};

// Builds the weight-stationary down's work list: one item per seg_blocks blocks of an expert, so a hot
// expert's blocks spread over many workgroups instead of streaming through one. One workgroup, E <= 1024.
__global__ __launch_bounds__(1024) void down_items_kernel(
    const int* __restrict__ expert_ids,
    const int* __restrict__ num_valid,
    int bm,
    int max_blocks,
    int E,
    int seg_blocks,
    int4* __restrict__ items) {
  __shared__ int first[1024], last[1024], scan[1024];
  const int t = threadIdx.x;
  const int live_m = min((num_valid[0] + bm - 1) / bm, max_blocks);
  first[t] = 0;
  last[t] = 0;
  __syncthreads();
  // expert ids ascend along the blocks: record where each expert's run starts and ends
  for (int b = t; b < live_m; b += 1024) {
    const int e = expert_ids[b];
    if (e < 0 || e >= E) continue;
    if (b == 0 || expert_ids[b - 1] != e) first[e] = b;
    if (b == live_m - 1 || expert_ids[b + 1] != e) last[e] = b + 1;
  }
  __syncthreads();
  const int lo = first[t], hi = last[t];
  const int segs = (hi - lo + seg_blocks - 1) / seg_blocks;
  scan[t] = segs;
  __syncthreads();
  for (int off = 1; off < 1024; off <<= 1) {
    const int v = t >= off ? scan[t - off] : 0;
    __syncthreads();
    scan[t] += v;
    __syncthreads();
  }
  int4* out = items + 1 + (scan[t] - segs);
  for (int s = 0; s < segs; ++s)
    out[s] = make_int4(t, lo + s * seg_blocks, min(lo + (s + 1) * seg_blocks, hi), 0);
  if (t == 0) items[0] = make_int4(scan[1023], 0, 0, 0);
}

template <int BM, int NW, int KS>
__global__ __launch_bounds__(NW * 64, 4) void down_ws_kernel(const DownParams p) {
  using C = DownWsCfg<BM, NW, KS>;
  constexpr int RT = C::RT, NTILE = C::NTILE, K1 = C::K1;

  const int tid = threadIdx.x, lane = tid & 63, wave = (int)to_sgpr((u32)tid >> 6);
  const int lrow = lane & 15, lgrp = lane >> 4;
  const int nchunk = p.N / C::BN;
  // Item-major order, so one item's column chunks (and an expert's neighbouring segments) share a die
  // and its inter rows and W2 in L2.
  const int lin = xcd_linear(min(p.items[0].x, p.max_items) * nchunk);
  if (lin < 0) return;
  const int item = lin / nchunk, chunk = lin - item * nchunk;
  const int live_m = min((p.num_valid[0] + BM - 1) / BM, p.max_blocks);
  const int4 work = p.items[1 + item];
  const int expert = (int)to_sgpr((u32)work.x);
  const int blk_lo = (int)to_sgpr((u32)work.y), blk_hi = min((int)to_sgpr((u32)work.z), live_m);
  if (blk_lo >= blk_hi) return;

  extern __shared__ __attribute__((aligned(16))) uint8_t lds[];

  // This wave's W2 columns, all K, scales per tile pair.
  const uint64_t w_bytes = (uint64_t)p.N * (p.I / 2);
  const int ks_pad = ((p.I / 32) + 7) / 8 * 8;
  const uint64_t ws_bytes = (uint64_t)p.N * ks_pad;
  const auto w_rsrc = make_rsrc(p.w + (size_t)expert * w_bytes, w_bytes);
  const auto ws_rsrc = make_rsrc(p.w_scale + (size_t)expert * ws_bytes, ws_bytes);
  const int tile0 = chunk * (C::BN / 16) + wave * NTILE;
  const u32 k1_stride = (u32)(ks_pad / 8);
  u32x4 b[NTILE][KS];
  u32 sc[NTILE / 2][K1];
#pragma unroll
  for (int t = 0; t < NTILE; ++t)
#pragma unroll
    for (int s = 0; s < KS; ++s)
      b[t][s] = __builtin_bit_cast(
          u32x4,
          __builtin_amdgcn_raw_buffer_load_b128(w_rsrc, (u32)(((tile0 + t) * KS + s) * kTileBytes + lane * 16), 0, 0));
#pragma unroll
  for (int q = 0; q < NTILE / 2; ++q)
#pragma unroll
    for (int k1 = 0; k1 < K1; ++k1)
      sc[q][k1] = __builtin_amdgcn_raw_buffer_load_b32(
          ws_rsrc, (((u32)(tile0 / 2 + q) * k1_stride + k1) * 64 + lane) * 4, 0, 0);

  const auto a_rsrc = make_rsrc(p.inter, (uint64_t)live_m * BM * p.I);
  const auto as_rsrc = make_rsrc(p.inter_scale, (uint64_t)live_m * BM * (p.I / 32));
  const auto y_rsrc = make_rsrc(p.y, (uint64_t)p.y_rows * p.N * 2);

  // A of block blk -> LDS slot by DMA: wave load g covers k-step s, rows 8 * rg .. + 7; lane l lands at
  // byte 16 l of it = row 8 rg + l / 8, swizzled chunk l % 8, so it fetches chunk (l % 8) ^ (row & 7).
  // The DMA is inline asm: as a builtin the waitcnt pass cannot tell the slots apart and would wait for
  // the next block's A before reading this one's. It writes LDS only, so no register can be reused early.
  u32 st_sc[KS];
  int st_sid = 0;
  const u32 lds_base = (u32)(uintptr_t)(__attribute__((address_space(3))) uint8_t*)lds;
  auto load_block = [&](int blk, int slot) {
#pragma unroll
    for (int i = 0; i < C::WLOADS_PER_WAVE; ++i) {
      const int g = wave + i * NW;
      if (C::A_WLOADS % NW != 0 && g >= C::A_WLOADS) break;  // wave-uniform
      const int s = g / (BM / 8), rg = g - s * (BM / 8);
      const int row = rg * 8 + (lane >> 3), cc = (lane & 7) ^ (row & 7);
      const u32 off = (u32)(blk * BM + row) * (u32)p.I + (u32)(s * kKStep + cc * 16);
      const u32 dst = to_sgpr(lds_base + (u32)(slot * C::SLOT_BYTES + s * BM * kKStep + rg * 1024));
      asm volatile("s_mov_b32 m0, %1\n\tbuffer_load_dwordx4 %0, %2, 0 offen lds"  // codespell:ignore offen
                   ::"v"(off),
                   "s"(dst),
                   "s"(a_rsrc)
                   : "memory", "m0");
    }
    if (tid < BM) {
      const int prow = blk * BM + tid;
#pragma unroll
      for (int i = 0; i < KS; ++i)  // the row's 4 * KS scale bytes, one dword a k-step
        st_sc[i] = __builtin_amdgcn_raw_buffer_load_b32(as_rsrc, (u32)prow * (u32)(p.I / 32) + 4 * i, 0, 0);
      st_sid = p.sorted_ids[prow];  // decoded at staging, so the loop does not wait on it early
    }
  };
  // Scales and output rows of block blk (registers -> LDS); the A DMA must have landed before the barrier.
  auto stage_block = [&](int slot, int blk) {
    uint8_t* base = lds + slot * C::SLOT_BYTES;
    if (tid < BM) {
      // dword j of the row: the scales of K blocks j, 4 + j, 8 + j(, 12 + j) -> opsel = k-step
      u32x4 o;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        u32 v = 0;
#pragma unroll
        for (int i = 0; i < KS; ++i)
          v |= ((st_sc[i] >> (8 * j)) & 0xFF) << (8 * i);
        o[j] = v;
      }
      *reinterpret_cast<u32x4*>(base + C::A_BYTES + tid * 16) = o;
      const int tok = st_sid & 0xFFFFFF, slot = (int)((u32)st_sid >> 24);
      const int prow = blk * BM + tid;
      const u32 yoff = ((u32)chunk * (u32)p.y_rows + (u32)prow) * (u32)C::BN * 2u;
      *reinterpret_cast<u32*>(base + C::A_BYTES + BM * 16 + tid * 4) = tok < p.T ? yoff : kOffDrop;
      if (chunk == 0 && tok < p.T) p.pos[tok * p.topk + slot] = prow;
    }
  };

  auto compute_block = [&](int slot) {
    const uint8_t* base = lds + slot * C::SLOT_BYTES;
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      const int row = r * 16 + lrow;
      const u32 asc = *reinterpret_cast<const u32*>(base + C::A_BYTES + row * 16 + lgrp * 4);
      f32x4 acc[NTILE];
#pragma unroll
      for (int t = 0; t < NTILE; ++t)
        acc[t] = (f32x4){0.f, 0.f, 0.f, 0.f};
      // one k-step of A at a time: 8 VGPRs live instead of 8 * KS
      auto kstep = [&](auto s_tag) {
        constexpr int s = decltype(s_tag)::value;
        const u32x4 a_lo = *reinterpret_cast<const u32x4*>(base + s * BM * kKStep + a_slab_off(row, lgrp));
        const u32x4 a_hi = *reinterpret_cast<const u32x4*>(base + s * BM * kKStep + a_slab_off(row, lgrp + 4));
#pragma unroll
        for (int t = 0; t < NTILE; ++t)
          acc[t] = (t & 1) ? mfma_a8w4<(s & 1) * 2 + 1, s>(b[t][s], a_lo, a_hi, acc[t], sc[t >> 1][s >> 1], asc)
                           : mfma_a8w4<(s & 1) * 2 + 0, s>(b[t][s], a_lo, a_hi, acc[t], sc[t >> 1][s >> 1], asc);
      };
      [&]<int... Ss>(std::integer_sequence<int, Ss...>) {
        (kstep(std::integral_constant<int, Ss>{}), ...);
      }(std::make_integer_sequence<int, KS>{});
      // lane (row, j) holds columns 16 t + 4 j .. +3: into the wave's [16][128 B] tile (16 B chunks
      // XOR-swizzled by row), then back out as 16 B of one row a lane, 8 whole rows a store
      uint8_t* xs = lds + 2 * C::SLOT_BYTES + wave * (16 * NTILE * 16 * 2);
#pragma unroll
      for (int t = 0; t < NTILE; ++t) {
        const u32x2 v = {bf16x2_rne(acc[t][0], acc[t][1]), bf16x2_rne(acc[t][2], acc[t][3])};
        const int ch = t * 2 + (lgrp >> 1);
        *reinterpret_cast<u32x2*>(xs + lrow * 128 + ((ch ^ (lrow & 7)) << 4) + (lgrp & 1) * 8) = v;
      }
      __builtin_amdgcn_wave_barrier();
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int c = k * 64 + lane, rr = c >> 3, cc = c & 7;
        const u32x4 v = *reinterpret_cast<const u32x4*>(xs + rr * 128 + ((cc ^ (rr & 7)) << 4));
        const u32 yr = *reinterpret_cast<const u32*>(base + C::A_BYTES + BM * 16 + (r * 16 + rr) * 4);
        const u32 off = yr == kOffDrop ? kOffDrop : yr + (u32)(wave * NTILE * 16 + cc * 8) * 2u;
        __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(u32x4, v), y_rsrc, off, 0, 0);
      }
      __builtin_amdgcn_wave_barrier();  // the next row tile reuses the wave's tile
    }
  };

  load_block(blk_lo, 0);
  // vmcnt(0) through the builtin, so the waitcnt pass knows the weights have landed and stops
  // re-waiting for them (and for the previous block's stores) at the top of every block.
  __builtin_amdgcn_s_waitcnt(0x0F70);
  stage_block(0, blk_lo);
  __syncthreads();
  for (int blk = blk_lo; blk < blk_hi; ++blk) {
    const int slot = (blk - blk_lo) & 1;
    const bool more = blk + 1 < blk_hi;
    if (more) load_block(blk + 1, slot ^ 1);
    compute_block(slot);
    if (more) {
      stage_block(slot ^ 1, blk + 1);
      // the next block's A DMA: issued before this block's STORES stores (and, in the first wave,
      // its scale loads), so at most that many newer ops may still be in flight
      wait_vmcnt<C::STORES>();
    }
    __syncthreads();
  }
}

// =============================================================================================
// reduce: out[tok][n] = bf16(sum_slot w[tok][slot] * y[pos[tok * topk + slot]][n]), fp32 in slot order.
// =============================================================================================
template <int TOPK>
__global__ __launch_bounds__(256) void reduce_kernel(
    const uint16_t* __restrict__ y,
    uint64_t y_bytes,
    const float* __restrict__ w,
    const int* __restrict__ pos,
    uint16_t* __restrict__ out,
    int N,
    long ldo,
    int chunk_cols,
    int y_rows) {
  const int tok = blockIdx.x;
  const float* wt = w + (size_t)tok * TOPK;
  const auto y_rsrc = make_rsrc(y, y_bytes);
  float ws[TOPK];
  u32 row_off[TOPK];  // byte offset of each slot's row: y stays under 4 GB
#pragma unroll
  for (int s = 0; s < TOPK; ++s) {
    ws[s] = wt[s];
    // chunked y is [N / chunk_cols][y_rows][chunk_cols]. The host always passes pos and chunk_cols;
    // the unused fallbacks stay because dropping them reschedules this kernel (130 -> 164 VGPRs).
    row_off[s] = (pos ? (u32)pos[tok * TOPK + s] : (u32)(tok * TOPK + s)) * (u32)(chunk_cols ? chunk_cols : N) * 2u;
  }
  for (int c = threadIdx.x; c < N / 8; c += blockDim.x) {
    const u32 col_off =
        chunk_cols ? ((u32)(c * 8 / chunk_cols) * (u32)y_rows * (u32)chunk_cols + (u32)(c * 8 % chunk_cols)) * 2u
                   : (u32)c * 16u;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < TOPK; ++s) {
      // aux 2 = nt on gfx950: y is read once
      const u32x4 v =
          __builtin_bit_cast(u32x4, __builtin_amdgcn_raw_buffer_load_b128(y_rsrc, row_off[s] + col_off, 0, 2));
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc[2 * i] += ws[s] * __builtin_bit_cast(float, v[i] << 16);
        acc[2 * i + 1] += ws[s] * __builtin_bit_cast(float, v[i] & 0xFFFF0000u);
      }
    }
    const u32x4 o = {
        bf16x2_rne(acc[0], acc[1]), bf16x2_rne(acc[2], acc[3]), bf16x2_rne(acc[4], acc[5]), bf16x2_rne(acc[6], acc[7])};
    *reinterpret_cast<u32x4*>(out + (size_t)tok * ldo + c * 8) = o;
  }
}

// ---------------------------------------------------------------------------------------------
// Host launchers.
// ---------------------------------------------------------------------------------------------
template <int BM, int NW, int NT>
inline void launch_gateup(const GateUpParams& p, int max_blocks, hipStream_t stream) {
  using C = GateUpCfg<BM, NW, NT>;
  const int nblk_n = p.I / C::BN;
  const size_t lds = 2 * C::SLAB_BYTES + (size_t)(p.K / 512) * BM * 16;
  hipLaunchKernelGGL((gateup_kernel<BM, NW, NT>), dim3(max_blocks * nblk_n), dim3(C::NTHR), lds, stream, p);
}

template <int BM, int NW, int KS>
inline void launch_down_ws(const DownParams& p, hipStream_t stream) {
  using C = DownWsCfg<BM, NW, KS>;
  hipLaunchKernelGGL(
      down_items_kernel,
      dim3(1),
      dim3(1024),
      0,
      stream,
      p.expert_ids,
      p.num_valid,
      BM,
      p.max_blocks,
      p.E,
      p.seg_blocks,
      const_cast<int4*>(p.items));
  hipLaunchKernelGGL(
      (down_ws_kernel<BM, NW, KS>), dim3(p.max_items * (p.N / C::BN)), dim3(C::NTHR), C::LDS_BYTES, stream, p);
}

template <int TOPK>
inline void launch_reduce(
    const uint16_t* y,
    uint64_t y_bytes,
    const float* w,
    const int* pos,
    uint16_t* out,
    int T,
    int N,
    long ldo,
    int chunk_cols,
    int y_rows,
    hipStream_t stream) {
  hipLaunchKernelGGL(
      (reduce_kernel<TOPK>), dim3(T), dim3(256), 0, stream, y, y_bytes, w, pos, out, N, ldo, chunk_cols, y_rows);
}

}  // namespace sglang::k3_a8w4

namespace sglang {

/// Host entry points of the K3 A8W4 MoE. Byte tensors (e4m3, e8m0, packed e2m1) arrive as uint8.
struct K3A8W4MoE {
  /// \brief inter[p] = MXFP8(SiTUv2(gate, up)) for every sorted row p of a BM block.
  static void gateup(
      const tvm::ffi::TensorView a,
      const tvm::ffi::TensorView a_scale,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView w_scale,
      const tvm::ffi::TensorView sorted_ids,
      const tvm::ffi::TensorView expert_ids,
      const tvm::ffi::TensorView num_valid,
      const tvm::ffi::TensorView inter,
      const tvm::ffi::TensorView inter_scale,
      const int64_t block_m,
      const double beta,
      const double linear_beta) {
    using namespace host;
    auto T_ = SymbolicSize{"num_tokens"};
    auto K_ = SymbolicSize{"hidden"};
    auto KG_ = SymbolicSize{"hidden / 32"};
    auto KP_ = SymbolicSize{"hidden / 2"};
    auto E_ = SymbolicSize{"num_experts"};
    auto N2_ = SymbolicSize{"2 * inter"};
    auto EN_ = SymbolicSize{"num_experts * 2 * inter"};
    auto S_ = SymbolicSize{"max_sorted"};
    auto B_ = SymbolicSize{"max_blocks"};
    auto V_ = SymbolicSize{"num_valid"};
    auto I_ = SymbolicSize{"inter"};
    auto IG_ = SymbolicSize{"inter / 32"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();
    TensorMatcher({T_, K_}).with_dtype<uint8_t>().with_device(device_).verify(a);
    TensorMatcher({T_, KG_}).with_dtype<uint8_t>().with_device(device_).verify(a_scale);
    TensorMatcher({E_, N2_, KP_}).with_dtype<uint8_t>().with_device(device_).verify(w);
    TensorMatcher({EN_, KG_}).with_dtype<uint8_t>().with_device(device_).verify(w_scale);
    TensorMatcher({S_}).with_dtype<int32_t>().with_device(device_).verify(sorted_ids);
    TensorMatcher({B_}).with_dtype<int32_t>().with_device(device_).verify(expert_ids);
    TensorMatcher({V_}).with_dtype<int32_t>().with_device(device_).verify(num_valid);
    TensorMatcher({S_, I_}).with_dtype<uint8_t>().with_device(device_).verify(inter);
    TensorMatcher({S_, IG_}).with_dtype<uint8_t>().with_device(device_).verify(inter_scale);

    k3_a8w4::GateUpParams p;
    p.T = static_cast<int>(T_.unwrap());
    p.K = static_cast<int>(K_.unwrap());
    p.I = static_cast<int>(I_.unwrap());
    p.E = static_cast<int>(E_.unwrap());
    p.max_blocks = static_cast<int>(B_.unwrap());
    RuntimeCheck(KP_.unwrap() * 2 == K_.unwrap() && KG_.unwrap() * 32 == K_.unwrap(), "hidden packing");
    RuntimeCheck(N2_.unwrap() == 2 * I_.unwrap() && IG_.unwrap() * 32 == I_.unwrap(), "inter packing");
    RuntimeCheck(EN_.unwrap() == E_.unwrap() * N2_.unwrap(), "w_scale rows");
    RuntimeCheck(p.K % 512 == 0, "hidden must be a multiple of 512, got ", p.K);
    RuntimeCheck(S_.unwrap() >= B_.unwrap() * block_m, "max_sorted below max_blocks * block_m");
    p.a = static_cast<const uint8_t*>(a.data_ptr());
    p.a_scale = static_cast<const uint8_t*>(a_scale.data_ptr());
    p.w = static_cast<const uint8_t*>(w.data_ptr());
    p.w_scale = static_cast<const uint8_t*>(w_scale.data_ptr());
    p.sorted_ids = static_cast<const int*>(sorted_ids.data_ptr());
    p.expert_ids = static_cast<const int*>(expert_ids.data_ptr());
    p.num_valid = static_cast<const int*>(num_valid.data_ptr());
    p.inter = static_cast<uint8_t*>(inter.data_ptr());
    p.inter_scale = static_cast<uint8_t*>(inter_scale.data_ptr());
    p.beta = static_cast<float>(beta);
    p.inv_beta = static_cast<float>(1.0 / beta);
    p.lbeta = static_cast<float>(linear_beta);
    p.inv_lbeta = static_cast<float>(1.0 / linear_beta);
    const auto stream = LaunchKernel::resolve_device(device_.unwrap());
    const int blocks = static_cast<int>(B_.unwrap());
    // A workgroup covers 4 waves x 2 tiles x 16 inter columns.
    RuntimeCheck(p.I % 128 == 0, "inter must be a multiple of 128, got ", p.I);
    switch (block_m) {
      case 16:
        k3_a8w4::launch_gateup<16, 4, 2>(p, blocks, stream);
        break;
      case 32:
        k3_a8w4::launch_gateup<32, 4, 2>(p, blocks, stream);
        break;
      case 64:
        k3_a8w4::launch_gateup<64, 4, 2>(p, blocks, stream);
        break;
      default:
        RuntimeCheck(false, "block_m must be 16, 32 or 64, got ", block_m);
    }
    RuntimeDeviceCheck();
  }

  /// \brief y[p] = bf16(inter[p] @ W2[e]^T) for every valid sorted row p, as [hidden / 512][rows][512];
  /// pos[tok * topk + slot] = p for the reduce.
  static void down(
      const tvm::ffi::TensorView inter,
      const tvm::ffi::TensorView inter_scale,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView w_scale,
      const tvm::ffi::TensorView sorted_ids,
      const tvm::ffi::TensorView expert_ids,
      const tvm::ffi::TensorView num_valid,
      const tvm::ffi::TensorView y,
      const tvm::ffi::TensorView pos,
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView work,
      const int64_t block_m,
      const int64_t seg_blocks) {
    using namespace host;
    auto S_ = SymbolicSize{"max_sorted"};
    auto I_ = SymbolicSize{"inter"};
    auto IG_ = SymbolicSize{"inter / 32"};
    auto IP_ = SymbolicSize{"inter / 2"};
    auto E_ = SymbolicSize{"num_experts"};
    auto N_ = SymbolicSize{"hidden"};
    auto EN_ = SymbolicSize{"num_experts * hidden"};
    auto GP_ = SymbolicSize{"pad8(inter / 32)"};
    auto B_ = SymbolicSize{"max_blocks"};
    auto V_ = SymbolicSize{"num_valid"};
    auto T_ = SymbolicSize{"num_tokens"};
    auto K_ = SymbolicSize{"topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();
    TensorMatcher({S_, I_}).with_dtype<uint8_t>().with_device(device_).verify(inter);
    TensorMatcher({S_, IG_}).with_dtype<uint8_t>().with_device(device_).verify(inter_scale);
    TensorMatcher({E_, N_, IP_}).with_dtype<uint8_t>().with_device(device_).verify(w);
    TensorMatcher({EN_, GP_}).with_dtype<uint8_t>().with_device(device_).verify(w_scale);
    TensorMatcher({S_}).with_dtype<int32_t>().with_device(device_).verify(sorted_ids);
    TensorMatcher({B_}).with_dtype<int32_t>().with_device(device_).verify(expert_ids);
    TensorMatcher({V_}).with_dtype<int32_t>().with_device(device_).verify(num_valid);
    TensorMatcher({T_, K_}).with_dtype<float>().with_device(device_).verify(topk_weights);
    // y rows are sorted rows; pos maps (tok, slot) to one for the reduce
    auto R_ = SymbolicSize{"num_tokens * topk"};
    TensorMatcher({S_, N_}).with_dtype<bf16_t>().with_device(device_).verify(y);
    TensorMatcher({R_}).with_dtype<int32_t>().with_device(device_).verify(pos);
    RuntimeCheck(R_.unwrap() == T_.unwrap() * K_.unwrap(), "pos must hold num_tokens * topk entries");

    k3_a8w4::DownParams p{};
    p.I = static_cast<int>(I_.unwrap());
    p.N = static_cast<int>(N_.unwrap());
    p.T = static_cast<int>(T_.unwrap());
    p.topk = static_cast<int>(K_.unwrap());
    p.y = static_cast<uint16_t*>(y.data_ptr());
    p.pos = static_cast<int*>(pos.data_ptr());
    p.y_rows = static_cast<int>(S_.unwrap());
    p.E = static_cast<int>(E_.unwrap());
    p.max_blocks = static_cast<int>(B_.unwrap());
    RuntimeCheck(E_.unwrap() <= 1024, "the work list builder takes at most 1024 experts");
    RuntimeCheck(seg_blocks >= 1, "seg_blocks must be >= 1");
    // sum over experts of ceil(blocks / seg_blocks) < max_blocks / seg_blocks + num_experts
    p.seg_blocks = static_cast<int>(seg_blocks);
    p.max_items = static_cast<int>((B_.unwrap() + seg_blocks - 1) / seg_blocks + E_.unwrap());
    auto W_ = SymbolicSize{"work ints"};
    TensorMatcher({W_}).with_dtype<int32_t>().with_device(device_).verify(work);
    RuntimeCheck(W_.unwrap() >= 4 * (1 + p.max_items), "work must hold 4 * (1 + max_items) ints");
    RuntimeCheck(reinterpret_cast<uintptr_t>(work.data_ptr()) % 16 == 0, "work must be 16 B aligned");
    p.items = static_cast<const int4*>(work.data_ptr());
    RuntimeCheck(IP_.unwrap() * 2 == I_.unwrap() && IG_.unwrap() * 32 == I_.unwrap(), "inter packing");
    RuntimeCheck(GP_.unwrap() == (IG_.unwrap() + 7) / 8 * 8, "w_scale columns must be pad8(inter / 32)");
    RuntimeCheck(EN_.unwrap() == E_.unwrap() * N_.unwrap(), "w_scale rows");
    // 8 waves x 64 columns a workgroup: hidden splits into 512-column chunks
    RuntimeCheck(p.N % 512 == 0, "hidden must be a multiple of 512, got ", p.N);
    RuntimeCheck(S_.unwrap() >= B_.unwrap() * block_m, "max_sorted below max_blocks * block_m");
    p.inter = static_cast<const uint8_t*>(inter.data_ptr());
    p.inter_scale = static_cast<const uint8_t*>(inter_scale.data_ptr());
    p.w = static_cast<const uint8_t*>(w.data_ptr());
    p.w_scale = static_cast<const uint8_t*>(w_scale.data_ptr());
    p.sorted_ids = static_cast<const int*>(sorted_ids.data_ptr());
    p.expert_ids = static_cast<const int*>(expert_ids.data_ptr());
    p.num_valid = static_cast<const int*>(num_valid.data_ptr());
    const auto stream = LaunchKernel::resolve_device(device_.unwrap());
    // KS = inter / 128 k-steps, instantiated for 3 and 4
    RuntimeCheck(p.I == 384 || p.I == 512, "inter must be 384 or 512, got ", p.I);
    auto go = [&](auto ks) {
      constexpr int KS = decltype(ks)::value;
      switch (block_m) {
        case 16:
          k3_a8w4::launch_down_ws<16, 8, KS>(p, stream);
          break;
        case 32:
          k3_a8w4::launch_down_ws<32, 8, KS>(p, stream);
          break;
        case 64:
          k3_a8w4::launch_down_ws<64, 8, KS>(p, stream);
          break;
        default:
          RuntimeCheck(false, "block_m must be 16, 32 or 64, got ", block_m);
      }
    };
    if (p.I == 384)
      go(std::integral_constant<int, 3>{});
    else
      go(std::integral_constant<int, 4>{});
    RuntimeDeviceCheck();
  }

  /// \brief out[tok] = bf16(sum_slot topk_weights[tok][slot] * y[pos[tok * topk + slot]]), fp32 sum;
  /// y is [hidden / chunk_cols][rows][chunk_cols], as down writes it.
  static void reduce(
      const tvm::ffi::TensorView y,
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView pos,
      const tvm::ffi::TensorView out,
      const int64_t chunk_cols) {
    using namespace host;
    auto T_ = SymbolicSize{"num_tokens"};
    auto K_ = SymbolicSize{"topk"};
    auto R_ = SymbolicSize{"y rows"};
    auto N_ = SymbolicSize{"hidden"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();
    TensorMatcher({R_, N_}).with_dtype<bf16_t>().with_device(device_).verify(y);
    TensorMatcher({T_, K_}).with_dtype<float>().with_device(device_).verify(topk_weights);
    TensorMatcher({T_, N_}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device_).verify(out);
    auto P_ = SymbolicSize{"num_tokens * topk"};
    TensorMatcher({P_}).with_dtype<int32_t>().with_device(device_).verify(pos);
    RuntimeCheck(P_.unwrap() == T_.unwrap() * K_.unwrap(), "pos must hold num_tokens * topk entries");
    RuntimeCheck(N_.unwrap() % 8 == 0 && out.stride(0) % 8 == 0, "16-byte rows");
    RuntimeCheck(K_.unwrap() == 16, "topk must be 16, got ", K_.unwrap());
    RuntimeCheck(chunk_cols > 0 && chunk_cols % 8 == 0 && N_.unwrap() % chunk_cols == 0, "chunk_cols");
    const auto stream = LaunchKernel::resolve_device(device_.unwrap());
    if (T_.unwrap() == 0) return;
    RuntimeCheck(R_.unwrap() * N_.unwrap() * 2 < (int64_t{1} << 32), "y must stay under 4 GB");
    k3_a8w4::launch_reduce<16>(
        static_cast<const uint16_t*>(y.data_ptr()),
        static_cast<uint64_t>(R_.unwrap() * N_.unwrap() * 2),
        static_cast<const float*>(topk_weights.data_ptr()),
        static_cast<const int*>(pos.data_ptr()),
        static_cast<uint16_t*>(out.data_ptr()),
        static_cast<int>(T_.unwrap()),
        static_cast<int>(N_.unwrap()),
        out.stride(0),
        static_cast<int>(chunk_cols),
        static_cast<int>(R_.unwrap()),
        stream);
    RuntimeDeviceCheck();
  }
};

}  // namespace sglang
