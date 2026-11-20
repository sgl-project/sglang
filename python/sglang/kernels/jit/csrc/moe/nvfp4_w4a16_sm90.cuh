/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/* decode4's pair-3 multiply (below) uses the one-IMAD decode of onnxruntime's moe_gemv_fp4.cu, which carries this
license:

MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Grouped NVFP4 x BF16 routed-expert GEMMs for Hopper (sm_90a): y[route] = W2 (silu(W1 x) * (W3 x)) for every route
// whose expert is local, with runtime hidden size H, expert intermediate size I, local expert count E and top-k, and
// optionally the weighted sum over each token's routes. python/sglang/kernels/ops/moe/nvfp4_w4a16_sm90.py owns the
// workspace and python/sglang/kernels/ops/quantization/nvfp4_sm90_repack.py packs the weights. Per call:
//   count_k   per-expert route counts of the local routes (ids outside [0, E) are other ranks' routes)
//   plan_k    per expert its token tiling, the experts by route count (most first), both GEMMs' tile offsets
//   scatter_k route ids into expert-sorted order, other ranks' routes after the local ones
//             (front_small_k does these three in one block up to kSmallRoutes routes)
//   gather_k  the routed token rows into sorted order, applying the K permutation the packed weights expect, so the
//             caller passes hidden states in their natural column order. Without the weighted sum it also writes
//             zeros to the output rows of other ranks' routes.
//   grouped_k (or pair_k, CTA pairs sharing token rows by multicast, at 16384+ routes) both GEMMs in one persistent
//             launch: GEMM1 tiles first, each GEMM2 tile waits for its token tile's GEMM1 row blocks.
//   combine_k out[t] = scale * sum over t's local routes k of w[t, k] y[t top_k + k], in increasing k
// Weights are 16-row x 64-k units of 576B (512B of E2M1 codes in decode4's bit layout, then the 64 E4M3 block scales
// as sc_bf16 codes), 256 rows per GEMM1 row block (for each half of its 128 intermediate columns, 64 gate rows then
// the same 64 up rows). I is a multiple of 64 (the repack pads); when it is an odd multiple, GEMM1's last row block
// holds only one half and GEMM2 has an odd number of 64-wide K chunks. One FP32 global scale per expert per GEMM is
// applied in the epilogue. Differing gate and up scales are folded exactly at pack time (up / gate into GEMM2's
// scale). GEMM1's copy-out optionally clamps SwiGLU's inputs (swiglu_limit).
// Every wgmma here takes A from registers, and ptxas through CUDA 13.2 can overwrite those registers while the wgmma
// is in flight, so the loader refuses a build whose SASS shows that (sglang.kernels.jit.utils.wgmma_rs_hazard).
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/ptx/async.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>

namespace sglang::nvfp4_w4a16_sm90 {

#define GDC_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")

constexpr int NW = 8, UNIT2 = 576;       // consumer warps; bytes per 16-row x 64-k weight unit
constexpr int WB2 = NW * 2 * UNIT2;      // weight bytes per 64-k chunk of a 256-row tile (9216)
constexpr int WROWS = WB2 / 512;         // the same in 512B rows of the weight tensor maps (18)
constexpr int MX_SB = 37888, MX_NS = 4;  // ring stage bytes (1KB aligned), stages
constexpr int MX_PAIR = 18432;           // narrow tiles: x0 at 0, w0 at 8192, x1 at MX_PAIR, w1 at MX_PAIR + 8192
constexpr uint32_t PAIR_W1 = 8192 + WB2, PAIR_X1 = 28672;  // pair_k narrow stage: x0, w0 w1 (one box), x1
constexpr int XOFF = 16384;           // full-width stage: token rows at 0 (up to 128 x 128B), weights here
constexpr int ST = 264;               // epilogue staging stride (BF16) per token row
constexpr int EBYTES = 128 * ST * 2;  // epilogue staging: [128 tokens][264]
constexpr int SMEM = 1024 + MX_NS * MX_SB + EBYTES;
constexpr int TOKMAX = 128;        // tokens per tile
constexpr int MX_MS = 16;          // tiles of at most this many tokens use mma.sync m16n8k16
constexpr int INFO_NTT_BITS = 24;  // plan info word: token tile count below, tsz / 32 above

// Test builds only (tests/test_races.py builds with -DNVFP4_MOE_DEBUG_DELAY_NS=<ns>): stretch the windows of the
// global-memory hand-offs so a missing wait or a reused buffer fails deterministically instead of rarely. gather_k
// lets the GEMM kernel launch at its start and then sleeps before writing its rows (a missing PDL wait reads them
// unwritten); the copiers sleep before writing the last GEMM1 row block of a token tile (a GEMM2 tile that starts
// early reads it unwritten), and one copier warp sleeps before its GEMM2 stores (a slot buffer reused too early is
// overwritten under it). The host fills the gathered rows and the intermediate with NaN first.
#ifndef NVFP4_MOE_DEBUG_DELAY_NS
#define NVFP4_MOE_DEBUG_DELAY_NS 0
#endif
__device__ __forceinline__ void debug_delay() {
  if constexpr (NVFP4_MOE_DEBUG_DELAY_NS > 0) {
    uint64_t t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
      __nanosleep(1000);
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < (uint64_t)NVFP4_MOE_DEBUG_DELAY_NS);
  }
}

// ---- per-call plan (int32 meta buffer, zeroed by the caller), offsets by local expert count E ----
// cnt[E] | start[E] | toff1[E + 1] | toff2[E + 1] | counter | nonlocal cursor | cursor[E] | perm[E] | info[E] | jt0[E]
// | cnt1[...]. perm, info, jt0 and the tile offsets are by plan position (experts by route count, most first); info =
// ntt | tsz / 32 << 24 (ntt token tiles of tsz tokens, the last one shorter; R < 2^30 keeps ntt below 2^23); cnt1
// counts the GEMM1 row blocks written per token tile. The non-local cursor places other ranks' routes after the local
// ones in the sorted order, so gather_k can zero their output rows.
__host__ __device__ __forceinline__ int m_start(int E) {
  return E;
}
__host__ __device__ __forceinline__ int m_toff1(int E) {
  return 2 * E;
}
__host__ __device__ __forceinline__ int m_toff2(int E) {
  return 3 * E + 1;
}
__host__ __device__ __forceinline__ int m_ctr(int E) {
  return 4 * E + 2;
}
__host__ __device__ __forceinline__ int m_nlcur(int E) {
  return 4 * E + 3;
}
__host__ __device__ __forceinline__ int m_cur(int E) {
  return 4 * E + 4;
}
__host__ __device__ __forceinline__ int m_perm(int E) {
  return 5 * E + 4;
}
__host__ __device__ __forceinline__ int m_info(int E) {
  return 6 * E + 4;
}
__host__ __device__ __forceinline__ int m_jt0(int E) {
  return 7 * E + 4;
}
__host__ __device__ __forceinline__ int m_cnt1(int E) {
  return 8 * E + 4;
}

struct Shape {
  int E;         // local experts
  int RB1, RB2;  // 256-row blocks per expert: GEMM1 (ceil(2 I / 256), the last one half high when I % 128 == 64), GEMM2
                 // (H / 256)
  int HB1;       // 64-column halves of the intermediate (I / 64): RB1 full blocks, or RB1 - 1 and a half block if odd
  int C1, C2;    // 64-wide K chunks: GEMM1 (H / 64), GEMM2 (I / 64, odd when GEMM1 ends on a half block)
  int I, H;      // row strides of the intermediate and of the output
  float lim;     // SwiGLU clamp (gate <= lim, -lim <= up <= lim), or +inf for none
};

// Token row maps of a (gathered input, [R][H]) and inter ([R][I]): boxes of 64 k x {64, 128, 8, 16, 32} rows; the
// weights as rows of 512B, boxes of one chunk (18 rows) and, for pair_k's narrow tiles, two chunks (36 rows). A
// grouped_k launch with a half block (never a pair_k one) holds GEMM1's half-chunk boxes (9 rows) in w2c[0] instead.
struct Maps {
  CUtensorMap x64[2], x128[2], xn[2][3], w[2], w2c[2];
};

__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect(uint32_t bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ bool elect_one() {  // elect.sync over the full (converged) warp: true in one lane
  uint32_t p;
  asm volatile("{\n .reg .pred P;\n elect.sync _|P, 0xffffffff;\n selp.u32 %0, 1, 0, P;\n}" : "=r"(p));
  return p != 0;
}
__device__ __forceinline__ void stsm4t(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr),
               "r"(a),
               "r"(b),
               "r"(c),
               "r"(d)
               : "memory");
}
__device__ __forceinline__ uint32_t pkbf(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
          dst),
      "l"(map),
      "r"(c0),
      "r"(c1),
      "r"(bar)
      : "memory");
}
__device__ __forceinline__ void
tma_2d_mc(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar, uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, "
      "%3}], [%4], %5;" ::"r"(dst),
      "l"(map),
      "r"(c0),
      "r"(c1),
      "r"(bar),
      "h"(mask)
      : "memory");
}
__device__ __forceinline__ void
tma_2d_h(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, "
      "%3}], [%4], %5;" ::"r"(dst),
      "l"(map),
      "r"(c0),
      "r"(c1),
      "r"(bar),
      "l"(pol)
      : "memory");
}
__device__ __forceinline__ uint64_t pol_evict_first() {
  uint64_t p;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
// CTA-pair helpers for pair_k: a shared address in CTA r of the cluster, remote arrives, release / acquire arrives for
// the claim hand-off, a remote shared store and a cluster-wide barrier.
__device__ __forceinline__ uint32_t cluster_map(uint32_t a, uint32_t r) {
  uint32_t o;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(r));
  return o;
}
__device__ __forceinline__ void mbar_arrive_cluster(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_arrive_release_cluster(uint32_t a) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_wait_acquire_cluster(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n W: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}" ::
          "r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void st_shared_cluster(uint32_t a, int v) {
  asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void cluster_sync() {  // every thread of both CTAs (divergent call sites: no .aligned)
  asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
  return r;
}
__device__ __forceinline__ uint64_t wg_desc(uint32_t addr) {  // K-major, 128B swizzle, 8-row groups 1KB apart
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void wg_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wg_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
template <int K_>
__device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(K_) : "memory");
}

// clang-format off
// m64nNk16 BF16 wgmma with A from registers and FP32 accumulators: wgmma_n<N> on float[N / 2], and
// wgmma_part<N, FULL> on the first N / 2 of a float[FULL / 2] (a tile with fewer tokens than its accumulators).
template <int N>
__device__ __forceinline__ void wgmma_n(float (&d)[N / 2], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc);
template <int N, int FULL>
__device__ __forceinline__ void wgmma_part(float (&d)[FULL / 2], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                           uint64_t desc);
template <>
__device__ __forceinline__ void wgmma_n<32>(float (&d)[16], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                            uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15"
      "}, {%16,%17,%18,%19}, %20, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_n<64>(float (&d)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                            uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31"
      "}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_n<128>(float (&d)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63"
      "}, {%64,%65,%66,%67}, %68, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
        "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]),
        "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<72, 128>(float (&d)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                    uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n72k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35"
      "}, {%36,%37,%38,%39}, %40, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<80, 128>(float (&d)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                    uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %45, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n80k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39"
      "}, {%40,%41,%42,%43}, %44, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<88, 128>(float (&d)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                    uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %49, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n88k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43"
      "}, {%44,%45,%46,%47}, %48, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<96, 128>(float (&d)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                    uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %53, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47"
      "}, {%48,%49,%50,%51}, %52, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<104, 128>(float (&d)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                     uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %57, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n104k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51"
      "}, {%52,%53,%54,%55}, %56, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
        "+f"(d[49]), "+f"(d[50]), "+f"(d[51])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<112, 128>(float (&d)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                     uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %61, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55"
      "}, {%56,%57,%58,%59}, %60, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
        "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<120, 128>(float (&d)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                     uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %65, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n120k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59"
      "}, {%60,%61,%62,%63}, %64, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
        "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]),
        "+f"(d[57]), "+f"(d[58]), "+f"(d[59])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<24, 32>(float (&d)[16], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                   uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %17, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11"
      "}, {%12,%13,%14,%15}, %16, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<40, 64>(float (&d)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                   uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %25, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19"
      "}, {%20,%21,%22,%23}, %24, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<48, 64>(float (&d)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                   uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %29, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23"
      "}, {%24,%25,%26,%27}, %28, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

template <>
__device__ __forceinline__ void wgmma_part<56, 64>(float (&d)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                   uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %33, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {"
      "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27"
      "}, {%28,%29,%30,%31}, %32, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}

// clang-format on

__device__ __forceinline__ void
mma16816(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

// Exact decode with no table: an E2M1 code c (sign bit 3) placed as BF16 bits (c & 7) << 6 | (c & 8) << 12 is
// exactly e2m1(c) * 2^-126. Multiplying by the block scale times 2^63 (exact) gives e2m1 * scale * 2^-63, a normal
// BF16; the 2^63 goes on the output with the global scale. decode4 does all four A pairs of a word in 6 ALU ops and 3
// IMADs; the pack places each code's bits so every pair lands under the mask M. Pair 3's single multiply by 0x1084 is
// the one-IMAD trick of onnxruntime's FP4 MoE GEMV decode (moe_gemv_fp4.cu; its MIT notice is at the top).
__device__ __forceinline__ void decode4(uint32_t q, uint32_t* o) {
  const uint32_t M = 0x81C081C0u;
  o[0] = q & M;
  o[1] = (q << 3) & M;
  o[2] = (q << 6) & M;
  o[3] = ((__funnelshift_l(q, q, 6) & 0x001B001Bu) * 0x1084u) & M;
}
// Two block scales (bytes 0, 1 of v) -> BF16 bits of scale * 2^63, each in both halves. The pack stores each E4M3
// scale losslessly as the code 8 E + m with scale = 1.m * 2^(E - 9) (subnormals normalized), so its BF16 bits are
// code * 16 + (181 << 7).
__device__ __forceinline__ uint32_t sc_bf16(uint32_t v, uint32_t sel) {
  return prmt(v, 0, sel) * 16u + 0x5A805A80u;
}
__device__ __forceinline__ uint32_t bmulh(uint32_t a, uint32_t s, int h) {
  const __nv_bfloat162 sv = *reinterpret_cast<__nv_bfloat162*>(&s);
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), h ? __high2bfloat162(sv) : __low2bfloat162(sv));
  return *reinterpret_cast<uint32_t*>(&r);
}

// One 64-k chunk: this warp's two units (rows ubase and ubase + 4 of the stage's 16) decoded and issued as W-wide
// wgmmas on accumulators of width TN: the whole chunk is decoded first, then one fence and the eight wgmmas.
template <int W, int TN>
__device__ __forceinline__ void
chunk_mma(float (&acc)[2][TN / 2], uint32_t (&r)[2][16], const uint8_t* wb, int ubase, int lane, uint32_t xa) {
  constexpr int MB = 2;
  const uint64_t d0 = wg_desc(xa);  // k16 step kk is 32B further: + 2 in the 16B address field
  auto mma = [&](int b, int kk) {
    if constexpr (W == TN)
      wgmma_n<TN>(acc[b], r[b][4 * kk], r[b][4 * kk + 1], r[b][4 * kk + 2], r[b][4 * kk + 3], d0 + 2 * kk);
    else
      wgmma_part<W, TN>(acc[b], r[b][4 * kk], r[b][4 * kk + 1], r[b][4 * kk + 2], r[b][4 * kk + 3], d0 + 2 * kk);
  };
  uint4 nb[MB];
  uint32_t S[MB];  // one scale pair (rows g | g + 8, this lane's block) per unit per chunk
#pragma unroll
  for (int b = 0; b < MB; b++) {
    const uint8_t* up = wb + (ubase + 4 * b) * UNIT2;
    nb[b] = *reinterpret_cast<const uint4*>(up + lane * 16);
    S[b] = sc_bf16(*reinterpret_cast<const uint16_t*>(up + 512 + lane * 2), 0x4140);
  }
#pragma unroll
  for (int kk = 0; kk < 4; kk++) {
#pragma unroll
    for (int b = 0; b < MB; b++) {
      const uint32_t w = kk == 0 ? nb[b].x : kk == 1 ? nb[b].y : kk == 2 ? nb[b].z : nb[b].w;
      uint32_t o[4];
      decode4(w, o);
      r[b][4 * kk + 0] = bmulh(o[0], S[b], 0);
      r[b][4 * kk + 1] = bmulh(o[1], S[b], 1);
      r[b][4 * kk + 2] = bmulh(o[2], S[b], 0);
      r[b][4 * kk + 3] = bmulh(o[3], S[b], 1);
    }
  }
  wg_fence();
#pragma unroll
  for (int kk = 0; kk < 4; kk++)
#pragma unroll
    for (int b = 0; b < MB; b++)
      mma(b, kk);
}
template <int V>
using IC = std::integral_constant<int, V>;

// MMA width of a tile with nv tokens: 8 / 16 (mma.sync), 32, 64, 128.
__host__ __device__ __forceinline__ int mx_w(int nv) {
  return nv <= MX_MS ? (nv + 7) / 8 * 8 : nv <= 32 ? 32 : nv <= 64 ? 64 : 128;
}

struct TileInfo {
  int t, mode, e, rb, row0, nv, jt;
};

// Locate tile t (warp-uniform, 0 <= t < total) from the plan: GEMM1 tiles first, then GEMM2's, each numbered over the
// experts in plan order; within an expert, row block outer and token tile inner. crank >= 0 (pair_k): t is even and
// the pair's two CTAs take row blocks 2 q and 2 q + 1. The binary search over the tile offsets is one ballot per 32
// experts (the offsets are monotone, so the expert is the last one whose offset is <= t).
__device__ __forceinline__ TileInfo locate_warp(const int* __restrict__ meta, const Shape& S, int t, int crank) {
  const int E = S.E, lane = threadIdx.x & 31;
  const int total1 = __ldg(meta + m_toff1(E) + E);
  TileInfo o;
  o.t = t;
  o.mode = t < total1 ? 1 : 2;
  const int tq = o.mode == 1 ? t : t - total1;
  const int* to = meta + (o.mode == 1 ? m_toff1(E) : m_toff2(E));
  int cnt = 0;
  for (int b = 0; b < E; b += 32) {
    const int i = b + lane;
    cnt += __popc(__ballot_sync(0xffffffffu, i < E && __ldg(to + i) <= tq));
  }
  const int pos = cnt - 1, lt = tq - __ldg(to + pos);
  o.e = __ldg(meta + m_perm(E) + pos);
  const int inf = __ldg(meta + m_info(E) + pos), ntt = inf & ((1 << INFO_NTT_BITS) - 1);
  const int tsz = (inf >> INFO_NTT_BITS) * 32;
  int tt;
  if (crank < 0) {
    o.rb = lt / ntt;
    tt = lt - o.rb * ntt;
  } else {
    const int qr = (lt >> 1) / ntt;
    o.rb = 2 * qr + crank;
    tt = (lt >> 1) - qr * ntt;
  }
  o.row0 = __ldg(meta + m_start(E) + o.e) + tt * tsz;
  o.nv = min(tsz, __ldg(meta + o.e) - tt * tsz);
  o.jt = __ldg(meta + m_jt0(E) + pos) + tt;
  return o;
}
// Queue entry: x = expert, y = row block | GEMM << 16 | MMA width << 20, z = first row, w = tokens.
__device__ __forceinline__ int4 tl_pack(const TileInfo& o, int W) {
  return make_int4(o.e, o.rb | o.mode << 16 | W << 20, o.row0, o.nv);
}

// Copy-out of tile k by the 96 copier threads (ct 0 .. 95) of grouped_k and pair_k, between the consumers' staging
// and the next tile. Named barrier 2: staging written (consumers arrive, copiers sync); barrier 3: staging free again
// (copiers arrive, consumers sync); barrier 4: the copiers alone. GEMM1 applies SiLU(gate) * up to the staged BF16
// gate and up rows, writes its 128 intermediate columns and then releases the row block to GEMM2's TMA reads (cnt1);
// GEMM2 writes each token's 256 output columns to its route's row. zs is the tile's slot buffer, double-buffered by
// tile parity: a copier loading tile k + 1's rows has passed tile k + 1's barrier, so every copier is done with tile
// k - 1's buffer. TAIL: the intermediate is an odd number of 64-column halves (see grouped_k). CLAMP: SwiGLU clamps its
// inputs at S.lim. Both are template flags, so the plain shapes compile to the same code as without them.
template <bool TAIL, bool CLAMP>
__device__ __forceinline__ void copy_out(
    const int4 tl,
    int jt,
    int ct,
    int* zs,
    const __nv_bfloat16* obuf,
    __nv_bfloat16* out1,
    __nv_bfloat16* out2,
    const int64_t* slot_of,
    int* cnt1,
    const float* gsr,
    const Shape& S) {
  constexpr int NC = NW * 32;
  const int rb = tl.y & 0xFFFF, mode = (tl.y >> 16) & 15, row0 = tl.z, nv = tl.w;
  if (mode == 2) {
    for (int i = ct; i < nv; i += 96)
      zs[i] = (int)slot_of[row0 + i];
    asm volatile("bar.sync 4, 96;" ::: "memory");
  }
  asm volatile("bar.sync 2, %0;" ::"n"(NC + 96) : "memory");
  if (mode == 1) {
    if (rb == S.RB1 - 1) debug_delay();
    // A half block (the last one when HB1 is odd) has only its first 64 columns: 8 uint4 per token, not 16.
    const int sh = TAIL && rb == S.RB1 - 1 ? 3 : 4;
    // The staged up rows carry the gate's global scale, and GEMM2's scale holds the up / gate ratio gsr[e], so the
    // clamp on the true up value is a clamp at lim / |gsr[e]| here (the clamp is symmetric, so a negative ratio works
    // too; gsr[e] is 0 only when up is all zero, and then the limit is inf).
    float ulim = 0.f;
    if constexpr (CLAMP) ulim = S.lim / fabsf(__ldg(gsr + tl.x));
    for (int i = ct; i < nv << sh; i += 96) {
      const int tok = i >> sh, c8 = (i & ((1 << sh) - 1)) * 8;
      const uint4 gq = *reinterpret_cast<const uint4*>(obuf + tok * ST + c8);
      const uint4 uq = *reinterpret_cast<const uint4*>(obuf + tok * ST + 128 + c8);
      const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gq);
      const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&uq);
      __align__(16) __nv_bfloat16 o[8];
#pragma unroll
      for (int x = 0; x < 8; x++) {
        float gv = __bfloat162float(gp[x]), uv = __bfloat162float(up[x]);
        if constexpr (CLAMP) {  // as SGLang's clamped SiLU: gate from above, up on both sides
          gv = gv > S.lim ? S.lim : gv;
          uv = uv > ulim ? ulim : (uv < -ulim ? -ulim : uv);
        }
        const float sl = __bfloat162float(__float2bfloat16(__fdividef(gv, 1.f + __expf(-gv))));
        o[x] = __float2bfloat16(sl * uv);
      }
      *reinterpret_cast<uint4*>(out1 + (size_t)(row0 + tok) * S.I + rb * 128 + c8) = *reinterpret_cast<const uint4*>(o);
    }
    // Release: the stores reach the async proxy and the GPU scope before any copier's count does.
    asm volatile("fence.proxy.async.global;" ::: "memory");
    __threadfence();
    asm volatile("bar.sync 4, 96;" ::: "memory");
    if (ct == 0) atomicAdd(&cnt1[jt], 1);
  } else {
    if (ct >= 64) debug_delay();
    for (int i = ct; i < nv * 32; i += 96) {
      const int tok = i >> 5, c8 = (i & 31) * 8;
      *reinterpret_cast<uint4*>(out2 + (size_t)zs[tok] * S.H + rb * 256 + c8) =
          *reinterpret_cast<const uint4*>(obuf + tok * ST + c8);
    }
  }
  asm volatile("bar.arrive 3, %0;" ::"n"(NC + 96) : "memory");
}

// Epilogue of tile k in grouped_k and pair_k, by each consumer warp: once the copiers are done with the staging
// (named barrier 3, from tile 1 on), scale the accumulators by the global scale times 2^63 (decode4's exponent offset)
// and stage them [token][ST] as BF16 by stmatrix.x4.trans, then tell the copiers (barrier 2). Columns: GEMM1 gate
// (warp / 4) * 64 + (warp % 4) * 16 (+ 8), up the same + 128 (block 1 holds the up rows of block 0's gate rows); GEMM2
// the tile's weight row. Per stmatrix: blocks (tokens 8j.., rows g), (8j.., g + 8), (8j + 8.., g), (8j + 8.., g + 8);
// lane 8i + t gives row t of block i.
template <int AW>
__device__ __forceinline__ void
stage_tile(const float (&acc)[2][AW / 2], float gscale, int k, int mode, int warp, int lane, __nv_bfloat16* obuf) {
  constexpr int NC = NW * 32;
  const float sc = gscale * 9.223372036854775808e18f;
  if (k > 0) asm volatile("bar.sync 3, %0;" ::"n"(NC + 96) : "memory");
  const int i4 = lane >> 3, t8 = lane & 7;
  const uint32_t ob = (uint32_t)__cvta_generic_to_shared(obuf);
#pragma unroll
  for (int b = 0; b < 2; b++) {
    const int cb = (mode == 1 ? 128 * b + (warp / 4) * 64 : 64 * b + (warp / 4) * 128) + (warp % 4) * 16 + 8 * (i4 & 1);
    const uint32_t base = ob + ((8 * (i4 >> 1) + t8) * ST + cb) * 2;
#pragma unroll
    for (int j = 0; j < AW / 8; j += 2)
      stsm4t(
          base + j * 8 * ST * 2,
          pkbf(acc[b][4 * j] * sc, acc[b][4 * j + 1] * sc),
          pkbf(acc[b][4 * j + 2] * sc, acc[b][4 * j + 3] * sc),
          pkbf(acc[b][4 * j + 4] * sc, acc[b][4 * j + 5] * sc),
          pkbf(acc[b][4 * j + 6] * sc, acc[b][4 * j + 7] * sc));
  }
  asm volatile("bar.arrive 2, %0;" ::"n"(NC + 96) : "memory");
}

// ---- the persistent GEMM kernel. Warp roles: 8 consumer warps (two warpgroups) decode and multiply; warp NW is the
// producer (TMA loads into a ring of MX_NS stages); warps NW + 1 .. NW + 3 copy finished tiles out, and the first of
// them claims and locates the next tile (scheduler). Every tile, a CTA's first included, is claimed from a counter, so
// only resident CTAs take tiles: a CTA that is still queued behind another kernel holds none, and the GEMM1 tiles a
// GEMM2 tile waits for were all claimed (by running CTAs) before it. TAIL: GEMM1's last row block is half high (its
// weights come through mp.w2c[0], which holds half-chunk boxes in such a launch) and GEMM2 has an odd chunk count.
template <bool TAIL, bool CLAMP>
__global__ void __launch_bounds__((NW + 4) * 32, 1) grouped_k(
    const __grid_constant__ Maps mp,
    int* __restrict__ meta,
    const float* __restrict__ gs1,
    const float* __restrict__ gs2,
    const float* __restrict__ gsr,
    __nv_bfloat16* __restrict__ out1,
    __nv_bfloat16* __restrict__ out2,
    const int64_t* __restrict__ slot_of,
    const Shape S) {
  constexpr int NC = NW * 32, SB = MX_SB, NS = MX_NS;
  const int E = S.E;
  int* const counter = meta + m_ctr(E);
  int* const cnt1 = meta + m_cnt1(E);
  extern __shared__ uint8_t mraw[];
  __shared__ __align__(8) uint64_t bars[2 * NS];
  __shared__ int s_tq[8];
  __shared__ int4 s_tl[8];
  __shared__ int s_tj[8];
  __shared__ __align__(8) uint64_t tbars[8];  // queue slot published (producer / scheduler -> all roles)
  __shared__ __align__(8) uint64_t sbars[8];  // queue slot k filled (scheduler -> producer)
  __shared__ __align__(8) uint64_t gbars[8];  // claim the next tile now (producer -> scheduler)
  __shared__ int z0_slot[2][TOKMAX];          // a GEMM2 tile's destination rows (route slots), double-buffered
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(mraw);
  const uint32_t sb = (raw + 1023) & ~1023u;
  uint8_t* const sp0 = mraw + (sb - raw);
  __nv_bfloat16* const obuf = reinterpret_cast<__nv_bfloat16*>(sp0 + NS * SB);
  const uint32_t bfull = (uint32_t)__cvta_generic_to_shared(bars), bempty = bfull + 8 * NS;
  if (tid == 0) {
    for (int s = 0; s < NS; s++) {
      mbar_init(bfull + 8 * s, 1);
      mbar_init(bempty + 8 * s, NW);
    }
    for (int s = 0; s < 8; s++)
      mbar_init((uint32_t)__cvta_generic_to_shared(&tbars[s]), 1);
    for (int s = 0; s < 8; s++)
      mbar_init((uint32_t)__cvta_generic_to_shared(&sbars[s]), 1);
    for (int s = 0; s < 8; s++)
      mbar_init((uint32_t)__cvta_generic_to_shared(&gbars[s]), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (tid >= NC && tid < NC + (TAIL ? 13 : 12)) {  // tensor maps (kernel parameters): before the PDL wait
    const CUtensorMap* pm = reinterpret_cast<const CUtensorMap*>(&mp) + (tid - NC);
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(pm)) : "memory");
  }
  __syncthreads();
  const uint32_t btq = (uint32_t)__cvta_generic_to_shared(tbars);
  const uint32_t bsl = (uint32_t)__cvta_generic_to_shared(sbars);
  const uint32_t bgo = (uint32_t)__cvta_generic_to_shared(gbars);
  if (warp >= NW) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    GDC_WAIT();
    const int total = __ldg(meta + m_toff1(E) + E) + __ldg(meta + m_toff2(E) + E);
    if (warp == NW) {    // producer warp: all 32 lanes stay converged; one elected lane issues each copy
      uint32_t gch = 0;  // chunks (stages) issued; unsigned so % NS and / NS are a mask and a shift
      const uint64_t pol = pol_evict_first();
      for (int k = 0;; k++) {
        int t, mode, e, rb, row0, nv, jt, W;
        if (k == 0) {
          TileInfo o{-1, 0, 0, 0, 0, 0, 0};
          int t0 = 0;
          if (lane == 0) t0 = atomicAdd(counter, 1);
          t0 = __shfl_sync(0xffffffffu, t0, 0);
          if (t0 < total) o = locate_warp(meta, S, t0, -1);
          t = o.t;
          mode = o.mode;
          e = o.e;
          rb = o.rb;
          row0 = o.row0;
          nv = o.nv;
          jt = o.jt;
          W = t >= 0 ? mx_w(nv) : 0;
          if (lane == 0) {
            s_tq[0] = t;
            s_tl[0] = tl_pack(o, W);
            s_tj[0] = jt;
          }
        } else {
          mbar_wait(bsl + 8 * (k & 7), ((k >> 3) - ((k & 7) == 0 ? 1 : 0)) & 1);
          t = s_tq[k & 7];
          const int4 tl0 = s_tl[k & 7];
          e = tl0.x;
          rb = tl0.y & 0xFFFF;
          mode = (tl0.y >> 16) & 15;
          W = tl0.y >> 20;
          row0 = tl0.z;
          nv = tl0.w;
          jt = s_tj[k & 7];
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(btq + 8 * (k & 7));
        if (t < 0) break;
        const int C = mode == 1 ? S.C1 : S.C2, RBm = mode == 1 ? S.RB1 : S.RB2;
        // GEMM1's half block (the last one when HB1 is odd) stores 8 units per chunk: half the rows and bytes. Its
        // consumers in the second warpgroup multiply whatever the stage held before, and copy_out drops their columns.
        const bool half = TAIL && mode == 1 && rb == S.RB1 - 1;
        const CUtensorMap* wm = half ? &mp.w2c[0] : &mp.w[mode - 1];
        const int wr0 = TAIL ? e * (mode == 1 ? S.HB1 : 2 * S.RB2) * C * (WROWS / 2) + rb * C * WROWS
                             : ((e * RBm + rb) * C) * WROWS;
        const int cr = half ? WROWS / 2 : WROWS, wbytes = half ? WB2 / 2 : WB2;
        if (mode == 2) {  // this token tile's intermediate rows: wait for its RB1 GEMM1 row blocks
          while (ld_acquire(&cnt1[jt]) < S.RB1)
            __nanosleep(128);
          asm volatile("fence.proxy.async.global;" ::: "memory");
        }
        if (W <= 64) {
          // Narrow tile: two chunks per stage, so the ring holds twice as many of these short, load-heavy chunks. An
          // odd chunk count (GEMM2 after a half block) leaves one chunk in the last stage.
          const CUtensorMap* bm = W <= 32 ? &mp.xn[mode - 1][W == 8 ? 0 : W == 16 ? 1 : 2] : &mp.x64[mode - 1];
          const int cgo =
              (C >= 8 ? C - 8 : 0) & ~1;  // 4 stages from the end (or the first): the scheduler may claim tile k + 1
          for (int c = 0; c < C; c += 2, gch++) {
            const int s = gch % NS, nj = TAIL && c + 1 == C ? 1 : 2;
            if (gch >= NS) mbar_wait(bempty + 8 * s, ((gch / NS) - 1) & 1);
            if (elect_one()) {
              mbar_expect(bfull + 8 * s, nj * (W * 128 + wbytes));
#pragma unroll
              for (int j = 0; j < 2; j++) {
                if (j == nj) break;
                tma_2d_h(sb + s * SB + j * MX_PAIR + 8192, wm, 0, wr0 + (c + j) * cr, bfull + 8 * s, pol);
                tma_2d(sb + s * SB + j * MX_PAIR, bm, (c + j) * 64, row0, bfull + 8 * s);
              }
              if (c == cgo) mbar_arrive(bgo + 8 * (k & 7));
            }
          }
        } else {
          // W 128: one 128-row box of token rows and one chunk of weights per stage.
          const CUtensorMap* bmw = &mp.x128[mode - 1];
          const int ex = 128 * 128 + wbytes;
          if ((gch % NS) == 0 && (C % NS) == 0 && !half) {  // stage slots known at compile time: unroll by NS
            for (int c = 0; c < C; c += NS, gch += NS) {
              const uint32_t par = ((gch / NS) - 1) & 1;
#pragma unroll
              for (int s = 0; s < NS; s++) {
                if (gch >= NS) mbar_wait(bempty + 8 * s, par);
                if (elect_one()) {
                  mbar_expect(bfull + 8 * s, ex);
                  tma_2d_h(sb + s * SB + XOFF, wm, 0, wr0 + (c + s) * WROWS, bfull + 8 * s, pol);
                  tma_2d(sb + s * SB, bmw, (c + s) * 64, row0, bfull + 8 * s);
                  if (s == 0 && c == C - NS) mbar_arrive(bgo + 8 * (k & 7));
                }
              }
            }
          } else {
            const int cgo = C >= 4 ? C - 4 : 0;
            for (int c = 0; c < C; c++, gch++) {
              const int s = gch % NS;
              if (gch >= NS) mbar_wait(bempty + 8 * s, ((gch / NS) - 1) & 1);
              if (elect_one()) {
                mbar_expect(bfull + 8 * s, ex);
                tma_2d_h(sb + s * SB + XOFF, wm, 0, wr0 + c * cr, bfull + 8 * s, pol);
                tma_2d(sb + s * SB, bmw, c * 64, row0, bfull + 8 * s);
                if (c == cgo) mbar_arrive(bgo + 8 * (k & 7));
              }
            }
          }
        }
      }
    } else {  // copy-out warps (copy_out); the first of them is also the scheduler
      const int ct = tid - (NC + 32);
      for (int k = 0;; k++) {
        mbar_wait(btq + 8 * (k & 7), (k >> 3) & 1);
        const int t = s_tq[k & 7];
        if (t < 0) break;
        if (warp == NW + 1) {  // the scheduler warp claims, locates and publishes tile k + 1
          const int kk = k + 1;
          mbar_wait(bgo + 8 * (k & 7), (k >> 3) & 1);  // claim late: once the producer is 4 stages from tile k's end
          int t_ = 0;
          if (lane == 0) t_ = atomicAdd(counter, 1);
          t_ = __shfl_sync(0xffffffffu, t_, 0);
          TileInfo o{-1, 0, 0, 0, 0, 0, 0};
          if (t_ < total) o = locate_warp(meta, S, t_, -1);
          if (lane == 0) {
            s_tq[kk & 7] = o.t;
            s_tl[kk & 7] = tl_pack(o, o.t >= 0 ? mx_w(o.nv) : 0);
            s_tj[kk & 7] = o.jt;
            mbar_arrive(bsl + 8 * (kk & 7));
          }
          __syncwarp();
        }
        copy_out<TAIL, CLAMP>(s_tl[k & 7], s_tj[k & 7], ct, z0_slot[k & 1], obuf, out1, out2, slot_of, cnt1, gsr, S);
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  GDC_WAIT();
  uint32_t gch = 0;
  for (int k = 0;; k++) {
    mbar_wait(btq + 8 * (k & 7), (k >> 3) & 1);
    const int t = s_tq[k & 7];
    if (t < 0) break;
    const int4 tl = s_tl[k & 7];
    const int e = tl.x, mode = (tl.y >> 16) & 15, wv = tl.y >> 20, ntok = tl.w;
    const int C = mode == 1 ? S.C1 : S.C2;
    const int Ce = TAIL ? C & ~1 : C;  // chunks in whole stages (pairs); a TAIL launch handles an odd last one after
    // One tile at MMA width W on its own accumulator array (wgmmas of different shapes writing one array are
    // serialized by ptxas, warning C7512).
    auto tile = [&](auto Wc) {
      constexpr int W = decltype(Wc)::value, MB = 2;
      const int ubase = (warp < 4 ? 0 : 8) + (warp % 4);
      constexpr int AW = (W + 15) / 16 * 16;  // W = 8 stages a pair of n8 blocks, the extra one stays 0
      float acc[MB][AW / 2];
      uint32_t ra[MB][16], rb_[MB][16];
#pragma unroll
      for (int b = 0; b < MB; b++)
#pragma unroll
        for (int i = 0; i < AW / 2; i++)
          acc[b][i] = 0.f;
      // Full-width (W 128) chunk at MMA width NWc on the W 128 accumulators. At full width the stage is freed one
      // chunk late (wait 1); a narrower MMA waits for its own chunk and frees its stage at once.
      auto step = [&](auto NWc, int c, uint32_t (&r)[MB][16]) {
        constexpr int NWv = decltype(NWc)::value;
        const int s = gch % NS;
        mbar_wait(bfull + 8 * s, (gch / NS) & 1);
        if constexpr (W == 128) chunk_mma<NWv, 128>(acc, r, sp0 + s * SB + XOFF, ubase, lane, sb + s * SB);
        wg_commit();
        if constexpr (NWv == W) {
          wg_wait<1>();
          if (c > 0) {
            __syncwarp();
            if (lane == 0) mbar_arrive(bempty + 8 * ((gch - 1) % NS));
          }
        } else {
          wg_wait<0>();
          __syncwarp();
          if (lane == 0) mbar_arrive(bempty + 8 * s);
        }
        gch++;
      };
      // Narrow pair loop (W 32 / 64) at MMA width NWc: both chunks of a stage, then wait 0 and release the stage. An
      // odd chunk count ends on a stage with one chunk.
      auto pairs = [&](auto NWc) {
        constexpr int NWv = decltype(NWc)::value;
        for (int c = 0; c < Ce; c += 2, gch++) {
          const int s = gch % NS;
          mbar_wait(bfull + 8 * s, (gch / NS) & 1);
          chunk_mma<NWv, W>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
          wg_commit();
          chunk_mma<NWv, W>(acc, rb_, sp0 + s * SB + MX_PAIR + 8192, ubase, lane, sb + s * SB + MX_PAIR);
          wg_commit();
          wg_wait<0>();
          __syncwarp();
          if (lane == 0) mbar_arrive(bempty + 8 * s);
        }
        if (TAIL && (C & 1)) {
          const int s = gch % NS;
          mbar_wait(bfull + 8 * s, (gch / NS) & 1);
          chunk_mma<NWv, W>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
          wg_commit();
          wg_wait<0>();
          __syncwarp();
          if (lane == 0) mbar_arrive(bempty + 8 * s);
          gch++;
        }
      };
      if constexpr (W < 32) {
        // At most MX_MS tokens: mma.sync m16n8k16 per n8 block on the same decoded A fragments, B fragments by
        // ldmatrix from the TMA-swizzled token rows.
        constexpr int NT = W / 8;
        const int lrow = lane & 7, lblk = lane >> 3;
        for (int c = 0; c < C; c += 2, gch++) {
          const int s = gch % NS;
          mbar_wait(bfull + 8 * s, (gch / NS) & 1);
#pragma unroll
          for (int j = 0; j < 2; j++) {
            if (TAIL && j == 1 && c + 1 == C) break;  // an odd chunk count: the last stage holds one chunk
            const uint32_t xr = sb + s * SB + j * MX_PAIR + lrow * 128;
            uint32_t bf[NT][8];
#pragma unroll
            for (int n = 0; n < NT; n++) {
              const uint32_t xn = xr + n * 1024;  // token rows 8 n + lrow: the 128B swizzle repeats every 8 rows
              ldsm4(xn + ((lblk ^ lrow) << 4), bf[n][0], bf[n][1], bf[n][2], bf[n][3]);
              ldsm4(xn + (((lblk + 4) ^ lrow) << 4), bf[n][4], bf[n][5], bf[n][6], bf[n][7]);
            }
            const uint8_t* wb = sp0 + s * SB + j * MX_PAIR + 8192;
#pragma unroll
            for (int b = 0; b < 2; b++) {
              const uint8_t* up = wb + (ubase + 4 * b) * UNIT2;
              const uint4 nb = *reinterpret_cast<const uint4*>(up + lane * 16);
              const uint32_t Sc = sc_bf16(*reinterpret_cast<const uint16_t*>(up + 512 + lane * 2), 0x4140);
#pragma unroll
              for (int kk = 0; kk < 4; kk++) {
                const uint32_t w = kk == 0 ? nb.x : kk == 1 ? nb.y : kk == 2 ? nb.z : nb.w;
                uint32_t o[4];
                decode4(w, o);
                const uint32_t a0 = bmulh(o[0], Sc, 0), a1 = bmulh(o[1], Sc, 1), a2 = bmulh(o[2], Sc, 0),
                               a3 = bmulh(o[3], Sc, 1);
#pragma unroll
                for (int n = 0; n < NT; n++)
                  mma16816(&acc[b][4 * n], a0, a1, a2, a3, bf[n][2 * kk], bf[n][2 * kk + 1]);
              }
            }
          }
          __syncwarp();  // mma.sync read this stage synchronously, so it is free now
          // The last ldmatrix can still be outstanding here (ptxas may issue the arrive before the HMMA that
          // consumes it); the proxy fence makes the release wait for the reads.
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          if (lane == 0) mbar_arrive(bempty + 8 * s);
        }
      } else if constexpr (W == 64) {
        if (ntok <= 40)
          pairs(IC<40>{});
        else if (ntok <= 48)
          pairs(IC<48>{});
        else if (ntok <= 56)
          pairs(IC<56>{});
        else
          pairs(IC<64>{});
      } else if constexpr (W == 32) {
        if (ntok <= 24)
          pairs(IC<24>{});
        else
          pairs(IC<32>{});
      } else {
        const int nw = ntok <= 72    ? 72
                       : ntok <= 80  ? 80
                       : ntok <= 88  ? 88
                       : ntok <= 96  ? 96
                       : ntok <= 104 ? 104
                       : ntok <= 112 ? 112
                       : ntok <= 120 ? 120
                                     : 128;
        if (nw == 80) {
          for (int c = 0; c < Ce; c += 2) {
            step(IC<80>{}, c, ra);
            step(IC<80>{}, c + 1, rb_);
          }
          if (TAIL && (C & 1)) step(IC<80>{}, C - 1, ra);
        } else if (nw == 96) {
          for (int c = 0; c < Ce; c += 2) {
            step(IC<96>{}, c, ra);
            step(IC<96>{}, c + 1, rb_);
          }
          if (TAIL && (C & 1)) step(IC<96>{}, C - 1, ra);
        } else if (nw == 112) {
          for (int c = 0; c < Ce; c += 2) {
            step(IC<112>{}, c, ra);
            step(IC<112>{}, c + 1, rb_);
          }
          if (TAIL && (C & 1)) step(IC<112>{}, C - 1, ra);
        } else if (nw == 72) {
          for (int c = 0; c < Ce; c += 2) {
            step(IC<72>{}, c, ra);
            step(IC<72>{}, c + 1, rb_);
          }
          if (TAIL && (C & 1)) step(IC<72>{}, C - 1, ra);
        } else if (nw == 88) {
          for (int c = 0; c < Ce; c += 2) {
            step(IC<88>{}, c, ra);
            step(IC<88>{}, c + 1, rb_);
          }
          if (TAIL && (C & 1)) step(IC<88>{}, C - 1, ra);
        } else if (nw == 104) {
          for (int c = 0; c < Ce; c += 2) {
            step(IC<104>{}, c, ra);
            step(IC<104>{}, c + 1, rb_);
          }
          if (TAIL && (C & 1)) step(IC<104>{}, C - 1, ra);
        } else if (nw == 120) {
          for (int c = 0; c < Ce; c += 2) {
            step(IC<120>{}, c, ra);
            step(IC<120>{}, c + 1, rb_);
          }
          if (TAIL && (C & 1)) step(IC<120>{}, C - 1, ra);
        } else {
          for (int c = 0; c < Ce; c += 2) {
            step(IC<128>{}, c, ra);
            step(IC<128>{}, c + 1, rb_);
          }
          if (TAIL && (C & 1)) step(IC<128>{}, C - 1, ra);
          // only the full-width step's lagged release leaves the tile's last stage to free here
          wg_wait<0>();
          __syncwarp();
          if (lane == 0) mbar_arrive(bempty + 8 * ((gch - 1) % NS));
        }
      }
      wg_wait<0>();
      __syncwarp();
      stage_tile<AW>(acc, __ldg((mode == 1 ? gs1 : gs2) + e), k, mode, warp, lane, obuf);
    };
    if (wv == 8)
      tile(IC<8>{});
    else if (wv == 16)
      tile(IC<16>{});
    else if (wv == 32)
      tile(IC<32>{});
    else if (wv == 64)
      tile(IC<64>{});
    else
      tile(IC<128>{});
  }
}

// grouped_k on clusters of two CTAs (launched for 16384+ routes). The pair works on the same token tile: CTA rank r
// takes weight row blocks 2 q + r, so every claim covers two tiles (both GEMMs' row block counts are even). Rank 0
// claims pairs from the counter and forwards each claim to rank 1. The token rows are multicast: each CTA issues half
// of a stage's token boxes (or rank 0 the single box) to both CTAs, and loads its own weights. A stage is free again
// only when the consumers of both CTAs are done with it (empty barriers count 2 NW arrives).
template <bool CLAMP>
__global__ void __launch_bounds__((NW + 4) * 32, 1) pair_k(
    const __grid_constant__ Maps mp,
    int* __restrict__ meta,
    const float* __restrict__ gs1,
    const float* __restrict__ gs2,
    const float* __restrict__ gsr,
    __nv_bfloat16* __restrict__ out1,
    __nv_bfloat16* __restrict__ out2,
    const int64_t* __restrict__ slot_of,
    const Shape S) {
  constexpr int NC = NW * 32, SB = MX_SB, NS = MX_NS;
  const int E = S.E;
  int* const counter = meta + m_ctr(E);
  int* const cnt1 = meta + m_cnt1(E);
  extern __shared__ uint8_t mraw[];
  __shared__ __align__(8) uint64_t bars[2 * NS];
  __shared__ int s_tq[8];
  __shared__ int4 s_tl[8];
  __shared__ int s_tj[8];
  __shared__ __align__(8) uint64_t tbars[8];
  __shared__ int z0_slot[2][TOKMAX];
  __shared__ int s_claim[8];                  // claims forwarded by rank 0 (rank 1 reads them)
  __shared__ __align__(8) uint64_t cbars[8];  // claim slot filled
  uint32_t crank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(crank));
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(mraw);
  const uint32_t sb = (raw + 1023) & ~1023u;
  uint8_t* const sp0 = mraw + (sb - raw);
  __nv_bfloat16* const obuf = reinterpret_cast<__nv_bfloat16*>(sp0 + NS * SB);
  const uint32_t bfull = (uint32_t)__cvta_generic_to_shared(bars), bempty = bfull + 8 * NS;
  if (tid == 0) {
    for (int s = 0; s < NS; s++) {
      mbar_init(bfull + 8 * s, 1);
      mbar_init(bempty + 8 * s, 2 * NW);
    }
    for (int s = 0; s < 8; s++)
      mbar_init((uint32_t)__cvta_generic_to_shared(&tbars[s]), 1);
    for (int i = 0; i < 8; i++)
      mbar_init((uint32_t)__cvta_generic_to_shared(&cbars[i]), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (tid >= NC && tid < NC + 14) {
    const CUtensorMap* pm = reinterpret_cast<const CUtensorMap*>(&mp) + (tid - NC);
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(pm)) : "memory");
  }
  __syncthreads();
  cluster_sync();
  GDC_WAIT();
  const uint32_t peer_empty = cluster_map(bempty, crank ^ 1u);
  const int total = __ldg(meta + m_toff1(E) + E) + __ldg(meta + m_toff2(E) + E);
  const uint32_t btq = (uint32_t)__cvta_generic_to_shared(tbars);
  if (warp >= NW) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    if (warp == NW) {  // producer warp, converged; it also claims (rank 0) or receives (rank 1) the tile pairs
      uint32_t gch = 0;
      const uint32_t pclaim = cluster_map((uint32_t)__cvta_generic_to_shared(s_claim), crank ^ 1u);
      const uint32_t pcbar = cluster_map((uint32_t)__cvta_generic_to_shared(cbars), crank ^ 1u);
      const uint32_t lcbar = (uint32_t)__cvta_generic_to_shared(cbars);
      int next_pair = -1;
      for (int k = 0;; k++) {
        int pair;
        if (crank == 0) {
          auto claim = [&](int kk) {
            int p = 0;
            if (lane == 0) {
              p = atomicAdd(counter, 1);
              if (2 * p >= total) p = -1;
              st_shared_cluster(pclaim + 4 * (kk & 7), p);
              mbar_arrive_release_cluster(pcbar + 8 * (kk & 7));
            }
            return __shfl_sync(0xffffffffu, p, 0);
          };
          if (k == 0) next_pair = claim(0);  // one claim ahead, so rank 1 never waits on rank 0's issue loop
          pair = next_pair;
          if (pair >= 0) next_pair = claim(k + 1);
        } else {
          mbar_wait_acquire_cluster(lcbar + 8 * (k & 7), (k >> 3) & 1);
          pair = *reinterpret_cast<volatile int*>(&s_claim[k & 7]);
        }
        TileInfo o{-1, 0, 0, 0, 0, 0, 0};
        if (pair >= 0) o = locate_warp(meta, S, 2 * pair, (int)crank);
        const int W = o.t >= 0 ? mx_w(o.nv) : 0;
        if (lane == 0) {
          s_tq[k & 7] = o.t;
          s_tl[k & 7] = tl_pack(o, W);
          s_tj[k & 7] = o.jt;
          mbar_arrive(btq + 8 * (k & 7));
        }
        __syncwarp();
        if (o.t < 0) break;
        const int mode = o.mode, C = mode == 1 ? S.C1 : S.C2, RBm = mode == 1 ? S.RB1 : S.RB2;
        const int wr0 = ((o.e * RBm + o.rb) * C) * WROWS;
        if (mode == 2) {
          while (ld_acquire(&cnt1[o.jt]) < S.RB1)
            __nanosleep(128);
          asm volatile("fence.proxy.async.global;" ::: "memory");
        }
        if (W <= 64) {
          // Narrow tile: two chunks per stage (x0 at 0, w0 w1 at 8192 as one box, x1 at PAIR_X1). Rank r multicasts
          // chunk c + r's token rows.
          const CUtensorMap* bm = W <= 32 ? &mp.xn[mode - 1][W == 8 ? 0 : W == 16 ? 1 : 2] : &mp.x64[mode - 1];
          const int ex = 2 * (W * 128 + WB2);
          for (int c = 0; c < C; c += 2, gch++) {
            const int s = gch % NS;
            if (gch >= NS) mbar_wait(bempty + 8 * s, ((gch / NS) - 1) & 1);
            if (elect_one()) {
              mbar_expect(bfull + 8 * s, ex);
              tma_2d_mc(
                  sb + s * SB + (crank ? PAIR_X1 : 0u), bm, (c + (int)crank) * 64, o.row0, bfull + 8 * s, (uint16_t)3);
              tma_2d(sb + s * SB + 8192, &mp.w2c[mode - 1], 0, wr0 + c * WROWS, bfull + 8 * s);  // both chunks' weights
            }
          }
        } else {
          for (int c = 0; c < C; c++, gch++) {
            const int s = gch % NS;
            if (gch >= NS) mbar_wait(bempty + 8 * s, ((gch / NS) - 1) & 1);
            if (elect_one()) {
              mbar_expect(bfull + 8 * s, 128 * 128 + WB2);
              tma_2d(sb + s * SB + XOFF, &mp.w[mode - 1], 0, wr0 + c * WROWS, bfull + 8 * s);  // own weights first
              if (crank == 0) tma_2d_mc(sb + s * SB, &mp.x128[mode - 1], c * 64, o.row0, bfull + 8 * s, (uint16_t)3);
            }
          }
        }
      }
    } else {
      const int ct = tid - (NC + 32);
      for (int k = 0;; k++) {
        mbar_wait(btq + 8 * (k & 7), (k >> 3) & 1);
        const int t = s_tq[k & 7];
        if (t < 0) break;
        copy_out<false, CLAMP>(s_tl[k & 7], s_tj[k & 7], ct, z0_slot[k & 1], obuf, out1, out2, slot_of, cnt1, gsr, S);
      }
    }
    cluster_sync();
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
  uint32_t gch = 0;
  for (int k = 0;; k++) {
    mbar_wait(btq + 8 * (k & 7), (k >> 3) & 1);
    const int t = s_tq[k & 7];
    if (t < 0) break;
    const int4 tl = s_tl[k & 7];
    const int e = tl.x, mode = (tl.y >> 16) & 15, wv = tl.y >> 20, ntok = tl.w;
    const int C = mode == 1 ? S.C1 : S.C2;
    auto tile = [&](auto Wc) {
      constexpr int W = decltype(Wc)::value, MB = 2;
      const int ubase = (warp < 4 ? 0 : 8) + (warp % 4);
      constexpr int AW = (W + 15) / 16 * 16;
      float acc[MB][AW / 2];
      uint32_t ra[MB][16], rb_[MB][16];
#pragma unroll
      for (int b = 0; b < MB; b++)
#pragma unroll
        for (int i = 0; i < AW / 2; i++)
          acc[b][i] = 0.f;
      auto release = [&](int s) {
        if (lane == 0) {
          mbar_arrive(bempty + 8 * s);
          mbar_arrive_cluster(peer_empty + 8 * s);
        }
      };
      auto step = [&](auto NWc, int c, uint32_t (&r)[MB][16]) {
        constexpr int NWv = decltype(NWc)::value;
        const int s = gch % NS;
        mbar_wait(bfull + 8 * s, (gch / NS) & 1);
        if constexpr (W == 128) chunk_mma<NWv, 128>(acc, r, sp0 + s * SB + XOFF, ubase, lane, sb + s * SB);
        wg_commit();
        if constexpr (NWv == W) {
          wg_wait<1>();
          if (c > 0) {
            __syncwarp();
            release((gch - 1) % NS);
          }
        } else {
          wg_wait<0>();
          __syncwarp();
          release(s);
        }
        gch++;
      };
      auto pairs = [&](auto NWc) {
        constexpr int NWv = decltype(NWc)::value;
        for (int c = 0; c < C; c += 2, gch++) {
          const int s = gch % NS;
          mbar_wait(bfull + 8 * s, (gch / NS) & 1);
          chunk_mma<NWv, W>(acc, ra, sp0 + s * SB + 8192, ubase, lane, sb + s * SB);
          wg_commit();
          chunk_mma<NWv, W>(acc, rb_, sp0 + s * SB + PAIR_W1, ubase, lane, sb + s * SB + PAIR_X1);
          wg_commit();
          wg_wait<0>();
          __syncwarp();
          release(s);
        }
      };
      if constexpr (W < 32) {
        constexpr int NT = W / 8;
        const int lrow = lane & 7, lblk = lane >> 3;
        for (int c = 0; c < C; c += 2, gch++) {
          const int s = gch % NS;
          mbar_wait(bfull + 8 * s, (gch / NS) & 1);
#pragma unroll
          for (int j = 0; j < 2; j++) {
            const uint32_t xr = sb + s * SB + j * PAIR_X1 + lrow * 128;  // chunk j's token rows
            uint32_t bf[NT][8];
#pragma unroll
            for (int n = 0; n < NT; n++) {
              const uint32_t xn = xr + n * 1024;
              ldsm4(xn + ((lblk ^ lrow) << 4), bf[n][0], bf[n][1], bf[n][2], bf[n][3]);
              ldsm4(xn + (((lblk + 4) ^ lrow) << 4), bf[n][4], bf[n][5], bf[n][6], bf[n][7]);
            }
            const uint8_t* wb = sp0 + s * SB + 8192 + j * (PAIR_W1 - 8192);  // chunk j's weights
#pragma unroll
            for (int b = 0; b < 2; b++) {
              const uint8_t* up = wb + (ubase + 4 * b) * UNIT2;
              const uint4 nb = *reinterpret_cast<const uint4*>(up + lane * 16);
              const uint32_t Sc = sc_bf16(*reinterpret_cast<const uint16_t*>(up + 512 + lane * 2), 0x4140);
#pragma unroll
              for (int kk = 0; kk < 4; kk++) {
                const uint32_t w = kk == 0 ? nb.x : kk == 1 ? nb.y : kk == 2 ? nb.z : nb.w;
                uint32_t o[4];
                decode4(w, o);
                const uint32_t a0 = bmulh(o[0], Sc, 0), a1 = bmulh(o[1], Sc, 1), a2 = bmulh(o[2], Sc, 0),
                               a3 = bmulh(o[3], Sc, 1);
#pragma unroll
                for (int n = 0; n < NT; n++)
                  mma16816(&acc[b][4 * n], a0, a1, a2, a3, bf[n][2 * kk], bf[n][2 * kk + 1]);
              }
            }
          }
          __syncwarp();
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          release(s);
        }
      } else if constexpr (W == 64) {
        if (ntok <= 40)
          pairs(IC<40>{});
        else if (ntok <= 48)
          pairs(IC<48>{});
        else if (ntok <= 56)
          pairs(IC<56>{});
        else
          pairs(IC<64>{});
      } else if constexpr (W == 32) {
        if (ntok <= 24)
          pairs(IC<24>{});
        else
          pairs(IC<32>{});
      } else {
        const int nw = ntok <= 80 ? 80 : ntok <= 96 ? 96 : ntok <= 112 ? 112 : 128;
        if (nw == 80) {
          for (int c = 0; c < C; c += 2) {
            step(IC<80>{}, c, ra);
            step(IC<80>{}, c + 1, rb_);
          }
        } else if (nw == 96) {
          for (int c = 0; c < C; c += 2) {
            step(IC<96>{}, c, ra);
            step(IC<96>{}, c + 1, rb_);
          }
        } else if (nw == 112) {
          for (int c = 0; c < C; c += 2) {
            step(IC<112>{}, c, ra);
            step(IC<112>{}, c + 1, rb_);
          }
        } else {
          for (int c = 0; c < C; c += 2) {
            step(IC<128>{}, c, ra);
            step(IC<128>{}, c + 1, rb_);
          }
          wg_wait<0>();
          __syncwarp();
          release((gch - 1) % NS);
        }
      }
      wg_wait<0>();
      __syncwarp();
      stage_tile<AW>(acc, __ldg((mode == 1 ? gs1 : gs2) + e), k, mode, warp, lane, obuf);
    };
    if (wv == 8)
      tile(IC<8>{});
    else if (wv == 16)
      tile(IC<16>{});
    else if (wv == 32)
      tile(IC<32>{});
    else if (wv == 64)
      tile(IC<64>{});
    else
      tile(IC<128>{});
  }
  cluster_sync();
}

// ---- route list kernels ----
// Per-CTA histograms of the local routes' experts (ids in [0, E)), added into meta's counts (meta comes zeroed).
__global__ void __launch_bounds__(1024) count_k(const int* __restrict__ ids, int R, int E, int* __restrict__ meta) {
  extern __shared__ int c[];
  const int tid = threadIdx.x, lane = tid % 32;
  for (int i = tid; i < E; i += blockDim.x)
    c[i] = 0;
  __syncthreads();
  for (int base = blockIdx.x * blockDim.x; base < R; base += gridDim.x * blockDim.x) {
    const int i = base + tid;
    const int e = i < R ? ids[i] : -1;
    const bool loc = e >= 0 && e < E;
    const unsigned m = __match_any_sync(0xffffffffu, loc ? e : -1);
    if (loc && lane == __ffs(m) - 1) atomicAdd(&c[e], __popc(m));
  }
  __syncthreads();
  for (int i = tid; i < E; i += blockDim.x)
    if (c[i]) atomicAdd(&meta[i], c[i]);
}
// Exclusive prefix sum over the block (every thread calls it); total gets the block's sum.
__device__ __forceinline__ int block_excl_scan(int v, int* sw, int& total) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) sw[wid] = x;
  __syncthreads();
  if (wid == 0) {
    int w = lane < nw ? sw[lane] : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, w, o);
      if (lane >= o) w += y;
    }
    if (lane < nw) sw[lane] = w;
  }
  __syncthreads();
  const int pre = wid ? sw[wid - 1] : 0;
  total = sw[nw - 1];
  __syncthreads();
  return pre + x - v;
}
// The plan: per expert its token tiling (ntt tiles of tsz tokens, tsz a multiple of 32 and at most 128), the experts
// by route count (most first, ties by id: the longest tiles are claimed first, the last claims are short), starts and
// scatter cursors by id, and by plan position the info word, both GEMMs' tile offsets and the token tile base.
// One block of at least E threads (a multiple of 32), with sh = [2 E + 32] ints whose first E hold the route counts.
__device__ __forceinline__ void plan_block(int* __restrict__ meta, int* sh, int E, int rb1, int rb2) {
  int* const scnt = sh;        // [E] route counts by id
  int* const sperm = sh + E;   // [E] experts by plan position
  int* const sw = sh + 2 * E;  // [32] scan scratch
  const int e = threadIdx.x;
  const bool act = e < E;
  const int n = act ? scnt[e] : 0;
  if (act) {
    int pos = 0;
    for (int j = 0; j < E; j++) {
      const int cj = scnt[j];
      pos += cj > n || (cj == n && j < e);
    }
    sperm[pos] = e;
    meta[m_perm(E) + pos] = e;
  }
  int tot;
  const int st = block_excl_scan(n, sw, tot);
  if (act) {
    meta[m_start(E) + e] = st;
    meta[m_cur(E) + e] = st;
  }
  // by plan position e: the expert pe
  int pn = 0, info = 0;
  if (act) {
    const int pc = scnt[sperm[e]];
    if (pc > 0) {
      pn = (pc + TOKMAX - 1) / TOKMAX;
      const int tsz = ((pc + pn - 1) / pn + 31) / 32 * 32;
      info = pn | tsz / 32 << INFO_NTT_BITS;
    }
  }
  int tot1, tot2, totj;
  const int t1 = block_excl_scan(pn * rb1, sw, tot1);
  const int t2 = block_excl_scan(pn * rb2, sw, tot2);
  const int j0 = block_excl_scan(pn, sw, totj);
  if (act) {
    meta[m_toff1(E) + e] = t1;
    meta[m_toff2(E) + e] = t2;
    meta[m_info(E) + e] = info;
    meta[m_jt0(E) + e] = j0;
  }
  if (e == 0) {
    meta[m_toff1(E) + E] = tot1;
    meta[m_toff2(E) + E] = tot2;
    meta[m_ctr(E)] = 0;
    meta[m_nlcur(E)] = tot;  // other ranks' routes go after the tot local ones
  }
}
__global__ void plan_k(int* __restrict__ meta, int E, int rb1, int rb2) {
  extern __shared__ int sh[];
  if (threadIdx.x < E) sh[threadIdx.x] = meta[threadIdx.x];
  __syncthreads();
  plan_block(meta, sh, E, rb1, rb2);
}
// Route ids to their expert-sorted positions (warp-aggregated atomics; the order within an expert is arbitrary, each
// route is computed alone), and other ranks' routes after all local ones.
__device__ __forceinline__ void
scatter_routes(const int* __restrict__ ids, int R, int E, int64_t* __restrict__ order, int* __restrict__ meta) {
  const int lane = threadIdx.x % 32;
  for (int base = blockIdx.x * blockDim.x; base < R; base += gridDim.x * blockDim.x) {
    const int i = base + threadIdx.x;
    const int e = i < R ? ids[i] : -1;
    const bool loc = e >= 0 && e < E;
    const int key = loc ? e : i < R ? -1 : -2;  // an expert, another rank, or past the end
    const unsigned m = __match_any_sync(0xffffffffu, key);
    const int leader = __ffs(m) - 1;
    int p = 0;
    if (key != -2 && lane == leader) p = atomicAdd(&meta[loc ? m_cur(E) + e : m_nlcur(E)], __popc(m));
    p = __shfl_sync(0xffffffffu, p, leader);
    if (key != -2) order[p + __popc(m & ((1u << lane) - 1))] = i;
  }
}
__global__ void __launch_bounds__(1024)
    scatter_k(const int* __restrict__ ids, int R, int E, int64_t* __restrict__ order, int* __restrict__ meta) {
  scatter_routes(ids, R, E, order, meta);
}
// count_k, plan_k and scatter_k in one block for small calls (decode), zeroing meta itself: one launch instead of a
// memset and three.
__global__ void __launch_bounds__(1024) front_small_k(
    const int* __restrict__ ids,
    int R,
    int E,
    int rb1,
    int rb2,
    int64_t* __restrict__ order,
    int* __restrict__ meta,
    int meta_words) {
  extern __shared__ int sh[];
  for (int i = threadIdx.x; i < meta_words; i += blockDim.x)
    meta[i] = 0;
  for (int i = threadIdx.x; i < E; i += blockDim.x)
    sh[i] = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < R; i += blockDim.x) {
    const int e = ids[i];
    if (e >= 0 && e < E) atomicAdd(&sh[e], 1);
  }
  __syncthreads();
  if (threadIdx.x < E) meta[threadIdx.x] = sh[threadIdx.x];
  plan_block(meta, sh, E, rb1, rb2);
  __syncthreads();
  scatter_routes(ids, R, E, order, meta);
}
// The routed rows of x in sorted order (token = route / topk), each 64-column chunk in the K order of the packed
// weights: position p = 16 kk + 8 half + 2 q + j holds column 16 q + 4 kk + 2 half + j, so (in 32-bit words of two
// columns) output word 8 kk + 4 half + q is input word 8 q + 2 kk + half, and output uint4 i of a chunk is input
// words i, 8 + i, 16 + i, 24 + i. One CTA per row, one thread per output uint4: the stores are contiguous and each
// load instruction reads whole 32B sectors (8 lanes x 4B per chunk). Sorted positions from n on are other ranks'
// routes: their output rows get zeros, as SGLang's Triton MoE writes for routes whose expert is not local, so the
// caller's weighted sum over a token's routes never reads stale bytes. combine_k needs no zeros (it skips those
// routes). (Zeroing them in the GEMM kernel's idle copier warps instead hides part of these stores at 2,048+ tokens but
// slows the GEMM kernel itself by 2% to 12%.)
__global__ void __launch_bounds__(256) gather_k(
    const uint32_t* __restrict__ x,
    const int64_t* __restrict__ order,
    const int* __restrict__ meta,
    int E,
    int R,
    uint4* __restrict__ a,
    uint4* __restrict__ y,
    int H,
    int topk) {
  GDC_WAIT();                                    // launched as a programmatic dependent of the plan
  if constexpr (NVFP4_MOE_DEBUG_DELAY_NS > 0) {  // test builds: let the GEMM kernel start before the rows exist
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    debug_delay();
  }
  const int n = meta[m_start(E) + E - 1] + meta[E - 1];
  const int H8 = H / 8, H2 = H / 2;
  // Without y (the caller takes the weighted sum, which skips other ranks' routes) only the local rows are visited.
  const int rows = y != nullptr ? R : n;
  for (int r = blockIdx.x; r < rows; r += gridDim.x) {
    if (r < n) {
      const uint32_t* in = x + order[r] / topk * H2;
      for (int c = threadIdx.x; c < H8; c += blockDim.x) {
        const uint32_t* p = in + (c >> 3) * 32 + (c & 7);
        a[(int64_t)r * H8 + c] = make_uint4(__ldg(p), __ldg(p + 8), __ldg(p + 16), __ldg(p + 24));
      }
    } else {
      uint4* const yr = y + order[r] * H8;
      for (int c = threadIdx.x; c < H8; c += blockDim.x)
        yr[c] = make_uint4(0, 0, 0, 0);
    }
  }
}

// out[t] = scale * sum over the local routes k of t (ids in [0, E)) of w[t, k] y[t top_k + k], in increasing k. Rows
// of y for other routes are never read. One CTA per token, one thread per 8 columns.
__global__ void __launch_bounds__(256) combine_k(
    const uint4* __restrict__ y,
    const int* __restrict__ ids,
    const float* __restrict__ w,
    int T,
    int topk,
    int E,
    int H8,
    float scale,
    uint4* __restrict__ out) {
  GDC_WAIT();  // launched as a programmatic dependent of the GEMMs
  for (int t = blockIdx.x; t < T; t += gridDim.x) {
    for (int c = threadIdx.x; c < H8; c += blockDim.x) {
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      for (int k0 = 0; k0 < topk; k0 += 8) {  // eight routes' loads in flight
        uint4 v[8];
        float wk[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const int k = k0 + j, e = k < topk ? ids[t * topk + k] : -1;
          wk[j] = e >= 0 && e < E ? w[t * topk + k] : 0.f;
        }
#pragma unroll
        for (int j = 0; j < 8; j++)
          v[j] = wk[j] != 0.f ? y[(int64_t)(t * topk + k0 + j) * H8 + c] : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const uint32_t u[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
#pragma unroll
          for (int i = 0; i < 4; i++) {
            const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u[i]));
            acc[2 * i] = fmaf(wk[j], f.x, acc[2 * i]);
            acc[2 * i + 1] = fmaf(wk[j], f.y, acc[2 * i + 1]);
          }
        }
      }
      out[(int64_t)t * H8 + c] = make_uint4(
          pkbf(acc[0] * scale, acc[1] * scale),
          pkbf(acc[2] * scale, acc[3] * scale),
          pkbf(acc[4] * scale, acc[5] * scale),
          pkbf(acc[6] * scale, acc[7] * scale));
    }
  }
}

// A kernel as a type, so one launch helper takes any instantiation.
template <auto K_>
using KernelC = std::integral_constant<decltype(K_), K_>;

// Route count from which the CTA-pair kernel is used (at fewer routes one CTA per tile is faster on the H200).
inline constexpr int64_t kPairMinRoutes = 16384;
// Route count up to which one block plans the call (the multi-block count and scatter win above it).
inline constexpr int64_t kSmallRoutes = 1024;

// Token rows: BF16 [rows][cols], boxes of 64 columns (128B, swizzled) x box_rows rows.
inline CUtensorMap row_map(void* base, int64_t cols, int64_t rows, uint32_t box_rows) {
  return host::tma::make_map(
      base,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      {(cuuint64_t)cols, (cuuint64_t)rows},
      {(cuuint64_t)cols * 2},
      {64, box_rows},
      {.swizzle = CU_TENSOR_MAP_SWIZZLE_128B, .l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B});
}
// Packed weights as rows of 512B (256 x uint16), boxes of box_rows rows.
inline CUtensorMap weight_map(void* base, int64_t bytes, uint32_t box_rows, CUtensorMapL2promotion l2) {
  return host::tma::make_map(
      base,
      CU_TENSOR_MAP_DATA_TYPE_UINT16,
      {256, (cuuint64_t)(bytes / 512)},
      {512},
      {256, box_rows},
      {.l2_promotion = l2});
}

// y[r] = W2 (silu(W1 x[r / top_k]) * (W3 x[r / top_k])) * scales for every route r = token * top_k + slot whose id
// is in [0, E). With out, out = scale * the weighted sum by topk_weights, and y is workspace. Without it, the rows of
// y for other routes are zeros. meta (int32), order, a and inter are the caller's workspace. variant is 0 for auto, 1
// for one CTA per tile and 2 for CTA pairs. ctas is 0 for one CTA per SM, else that many (tests reach each path).
inline void
run(tvm::ffi::TensorView x,
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView w13,
    tvm::ffi::TensorView w2,
    tvm::ffi::TensorView gs13,
    tvm::ffi::TensorView gs2,
    tvm::ffi::TensorView gsr,
    tvm::ffi::TensorView meta,
    tvm::ffi::TensorView order,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView inter,
    tvm::ffi::TensorView y,
    tvm::ffi::Optional<tvm::ffi::TensorView> topk_weights,
    tvm::ffi::Optional<tvm::ffi::TensorView> out,
    double scale,
    double swiglu_limit,
    int64_t variant,
    int64_t ctas) {
  using namespace host;
  auto T = SymbolicSize{"tokens"};
  auto H = SymbolicSize{"hidden"};
  auto K = SymbolicSize{"top_k"};
  auto R = SymbolicSize{"routes"};
  auto I = SymbolicSize{"intermediate"};
  auto E = SymbolicSize{"local_experts"};
  auto device = SymbolicDevice{};
  TensorMatcher({T, H}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).ensure_alignment(16).verify(x);
  TensorMatcher({T, K}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
  TensorMatcher({E}).with_dtype<float>().with_device(device).verify(gs13);
  TensorMatcher({E}).with_dtype<float>().with_device(device).verify(gs2);
  TensorMatcher({E}).with_dtype<float>().with_device(device).verify(gsr);
  TensorMatcher({R, H}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(a);
  TensorMatcher({R, H}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(y);
  TensorMatcher({R, I}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(inter);
  TensorMatcher({R}).with_dtype<int64_t>().with_device(device).verify(order);
  TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(meta);
  const int64_t t = T.unwrap(), h = H.unwrap(), topk = K.unwrap(), r = R.unwrap(), i = I.unwrap(), e = E.unwrap();
  RuntimeCheck(r == t * topk, "nvfp4_w4a16_sm90: workspace rows ", r, " != tokens * top_k ", t * topk);
  RuntimeCheck(topk >= 1 && r < (int64_t{1} << 30), "nvfp4_w4a16_sm90: need 1 <= top_k and tokens * top_k < 2^30");
  RuntimeCheck(h >= 256 && h % 256 == 0, "nvfp4_w4a16_sm90: hidden size must be a multiple of 256, got ", h);
  RuntimeCheck(i >= 64 && i % 64 == 0, "nvfp4_w4a16_sm90: padded intermediate size must be a multiple of 64, got ", i);
  RuntimeCheck(e >= 1 && e <= 1024, "nvfp4_w4a16_sm90: 1 to 1024 local experts, got ", e);
  RuntimeCheck(swiglu_limit > 0, "nvfp4_w4a16_sm90: swiglu_limit must be positive (inf for none)");
  const int HB1 = (int)(i / 64), RB1 = (HB1 + 1) / 2, RB2 = (int)(h / 256), C1 = (int)(h / 64), C2 = (int)(i / 64);
  TensorMatcher({e, (int64_t)HB1 * C1 * 8, 576})
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(w13);
  TensorMatcher({e, RB2, C2, 16, 576}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w2);
  RuntimeCheck(meta.size(0) >= 8 * e + 4 + r / TOKMAX + e + 64, "nvfp4_w4a16_sm90: meta workspace too small");
  RuntimeCheck(out.has_value() == topk_weights.has_value(), "nvfp4_w4a16_sm90: out and topk_weights go together");
  if (out.has_value()) {
    TensorMatcher({T, H}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(out.value());
    TensorMatcher({T, K}).with_dtype<float>().with_device(device).verify(topk_weights.value());
  }
  RuntimeCheck(variant >= 0 && variant <= 2, "nvfp4_w4a16_sm90: variant must be 0 (auto), 1 or 2");
  if (r == 0) return;
  const DLDevice dev = device.unwrap();
  RuntimeCheck(runtime::get_sm_version(dev.device_id) == 90, "nvfp4_w4a16_sm90: needs an sm_90 (Hopper) GPU");
  const int sms = (int)runtime::get_sm_count(dev.device_id);
  const bool pair_ok = HB1 % 2 == 0 && RB1 % 2 == 0 && RB2 % 2 == 0;
  const bool pair = variant == 2 || (variant == 0 && pair_ok && r >= kPairMinRoutes);
  RuntimeCheck(
      !pair || pair_ok, "nvfp4_w4a16_sm90: CTA pairs need an even number of full 256-row blocks in both GEMMs");
  int grid = ctas > 0 ? (int)std::min<int64_t>(ctas, sms) : sms;
  if (pair) grid &= ~1;
  RuntimeCheck(grid >= (pair ? 2 : 1), "nvfp4_w4a16_sm90: too few CTAs");

  cudaStream_t stream = LaunchKernel::resolve_device(dev);
  int* m = static_cast<int*>(meta.data_ptr());
  const int* ids = static_cast<const int*>(topk_ids.data_ptr());
  auto* ord = static_cast<int64_t*>(order.data_ptr());
  const int Ei = (int)e, Ri = (int)r;
  const int pb = (Ei + 31) / 32 * 32;
  const size_t plan_smem = (2 * Ei + 32) * sizeof(int);
  if (r <= kSmallRoutes) {
    LaunchKernel(1, std::max(pb, 256), dev, plan_smem)(front_small_k, ids, Ri, Ei, RB1, RB2, ord, m, (int)meta.size(0));
  } else {
    const int g = (int)std::min<int64_t>(2L * sms, (r + 1023) / 1024);
    CHECK_CUDA(cudaMemsetAsync(m, 0, meta.size(0) * sizeof(int), stream));
    LaunchKernel(g, 1024, dev, Ei * sizeof(int))(count_k, ids, Ri, Ei, m);
    LaunchKernel(1, pb, dev, plan_smem)(plan_k, m, Ei, RB1, RB2);
    LaunchKernel(g, 1024, dev)(scatter_k, ids, Ri, Ei, ord, m);
  }
  if constexpr (NVFP4_MOE_DEBUG_DELAY_NS > 0) {  // test builds: rows read before they are written come out NaN
    CHECK_CUDA(cudaMemsetAsync(a.data_ptr(), 0xFF, r * h * 2, stream));
    CHECK_CUDA(cudaMemsetAsync(inter.data_ptr(), 0xFF, r * i * 2, stream));
  }
  LaunchKernel((int)std::min<int64_t>(2048, r), 256, dev)
      .enable_pdl(true)(
          gather_k,
          static_cast<const uint32_t*>(x.data_ptr()),
          (const int64_t*)ord,
          (const int*)m,
          Ei,
          Ri,
          static_cast<uint4*>(a.data_ptr()),
          out.has_value() ? nullptr : static_cast<uint4*>(y.data_ptr()),
          (int)h,
          (int)topk);

  Maps mp;
  for (int k = 0; k < 2; k++) {
    void* b = k == 0 ? a.data_ptr() : inter.data_ptr();
    const int64_t cols = k == 0 ? h : i;
    mp.x64[k] = row_map(b, cols, r, 64);
    mp.x128[k] = row_map(b, cols, r, 128);
    for (int j = 0; j < 3; j++)
      mp.xn[k][j] = row_map(b, cols, r, 8u << j);
    const tvm::ffi::TensorView& wt = k == 0 ? w13 : w2;
    const int64_t bytes = k == 0 ? e * HB1 * C1 * 8 * 576 : e * RB2 * C2 * 16 * 576;
    mp.w[k] = weight_map(wt.data_ptr(), bytes, WROWS, CU_TENSOR_MAP_L2_PROMOTION_NONE);
    mp.w2c[k] = weight_map(wt.data_ptr(), bytes, 2 * WROWS, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    if (k == 0 && HB1 % 2) mp.w2c[0] = weight_map(wt.data_ptr(), bytes, WROWS / 2, CU_TENSOR_MAP_L2_PROMOTION_NONE);
  }
  const Shape S{Ei, RB1, RB2, HB1, C1, C2, (int)i, (int)h, (float)swiglu_limit};
  const bool clamp = swiglu_limit < INFINITY;
  const auto* g1 = static_cast<const float*>(gs13.data_ptr());
  const auto* g2 = static_cast<const float*>(gs2.data_ptr());
  const auto* gr = static_cast<const float*>(gsr.data_ptr());
  auto* o1 = static_cast<__nv_bfloat16*>(inter.data_ptr());
  auto* o2 = static_cast<__nv_bfloat16*>(y.data_ptr());
  if (pair) {
    auto launch = [&](auto kernel) {
      runtime::set_max_dynamic_smem_per_device<decltype(kernel)::value>(dev, SMEM);
      LaunchKernel(grid, (NW + 4) * 32, dev, SMEM)
          .enable_cluster(dim3{2, 1, 1})(decltype(kernel)::value, mp, m, g1, g2, gr, o1, o2, (const int64_t*)ord, S);
    };
    if (clamp)
      launch(KernelC<pair_k<true>>{});
    else
      launch(KernelC<pair_k<false>>{});
  } else {  // programmatic dependent launch: the prologue overlaps the gather's tail
    auto launch = [&](auto kernel) {
      runtime::set_max_dynamic_smem_per_device<decltype(kernel)::value>(dev, SMEM);
      LaunchKernel(grid, (NW + 4) * 32, dev, SMEM)
          .enable_pdl(true)(decltype(kernel)::value, mp, m, g1, g2, gr, o1, o2, (const int64_t*)ord, S);
    };
    const bool tail = HB1 % 2;
    if (tail && clamp)
      launch(KernelC<grouped_k<true, true>>{});
    else if (tail)
      launch(KernelC<grouped_k<true, false>>{});
    else if (clamp)
      launch(KernelC<grouped_k<false, true>>{});
    else
      launch(KernelC<grouped_k<false, false>>{});
  }
  if (out.has_value()) {
    LaunchKernel((int)std::min<int64_t>(t, 4 * sms), 256, dev)
        .enable_pdl(true)(
            combine_k,
            static_cast<const uint4*>(y.data_ptr()),
            ids,
            static_cast<const float*>(topk_weights.value().data_ptr()),
            (int)t,
            (int)topk,
            Ei,
            (int)(h / 8),
            (float)scale,
            static_cast<uint4*>(out.value().data_ptr()));
  }
}

}  // namespace sglang::nvfp4_w4a16_sm90
