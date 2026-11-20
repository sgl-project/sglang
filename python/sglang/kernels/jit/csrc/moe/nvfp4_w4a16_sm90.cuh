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

// ---- shared by the decode kernels ----
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, int bytes, uint32_t bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::
          "r"(dst),
      "l"(src),
      "r"(bytes),
      "r"(bar),
      "l"(pol)
      : "memory");
}
__device__ __forceinline__ uint4 ld_cg4(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ int atom_add_acq_rel(int* p, int v) {
  int o;
  asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], %2;" : "=r"(o) : "l"(p), "r"(v) : "memory");
  return o;
}
__device__ __forceinline__ uint32_t u4w(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}
// Natural intermediate column of gate row i (0 .. 63) of a 128-row half (pack.kperm_index).
__device__ __forceinline__ int dec_kp(int i) {
  return 16 * ((i >> 1) & 3) + 4 * (i >> 4) + 2 * ((i >> 3) & 1) + (i & 1);
}
__device__ __forceinline__ void
tma_4d_h(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, int c3, uint32_t bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, "
      "%4, %5}], [%6], %7;" ::"r"(dst),
      "l"(map),
      "r"(c0),
      "r"(c1),
      "r"(c2),
      "r"(c3),
      "r"(bar),
      "l"(pol)
      : "memory");
}

// ---- decode, one launch (decf_k): both GEMMs in one persistent kernel, a tagged intermediate exchange ----
// For one to a few tokens the call costs the touched experts' weight bytes. Every CTA computes the routing (slots:
// experts with local routes in increasing id), owns rows over the full K (no K sum crosses CTAs) and streams its share
// of each GEMM through a two-stage ring. GEMM2 needs intermediates that other CTAs write, and on H100 any global round
// trip during the weight stream waits several us, so the exchange is built to need one:
//   * GEMM1 writes the intermediate as 8-byte words, two BF16 values with this call's epoch above them, so a reader
//     needs no counter and no fence: a word with this call's epoch holds this call's values. The epoch and the words
//     live in one workspace that only grows (zeroed), so it is never newer than its epoch.
//   * Every local route's intermediate row gets its own shared-memory row when they fit, fetched once for all of the
//     CTA's GEMM2 pieces (otherwise slots rotate through as many buffers as fit). Three pairs of fetcher warps take
//     16B-word chunks of those rows; a chunk starts once this CTA has written its GEMM1 items up to the round in which
//     the chunk's last slot finishes (every CTA runs the same schedule) plus DF_FDLY_NS, and then reloads only the
//     words whose epoch is still old.
//   * Scales and top-k weights are staged in shared memory with the routing, before the stream starts, and every
//     item is decoded once into shared memory: the hot paths stay short, because after an L2 flush each role's first
//     pass through its code fetches instructions from HBM.
//   * GEMM1 items are 32 gate and 32 up rows, or 16 and 16 when that lowers the busiest CTA's stages. A GEMM2 item is
//     one slot, or up to 8 slots of one piece when a slot's weights are one small block: one ring stage and one
//     hand-off for all of them.
constexpr int DF_CW = 8;                                  // consumer warps
constexpr int DF_NP = 3;                                  // fetcher warp pairs
constexpr int DF_THREADS = (DF_CW + 2 + 2 * DF_NP) * 32;  // + producer, epilogue, fetchers
constexpr int DF_STB = 36864;                             // bytes per ring stage
constexpr int DF_NS = 2;                                  // ring stages: fewer bytes in flight, shorter round trips
constexpr int DF_TMAX = 4;
constexpr int DF_RMAX = 256;
constexpr int DF_EMAX = 1024;
constexpr int DF_RED = DF_CW * 64 * DF_TMAX * 4;  // bytes per hand-off buffer
constexpr int DF_NBMAX = 16;                      // intermediate row buffers / fetch chunks
constexpr int DF_NBAR = 2 * DF_NS + 1 + 2 * DF_NBMAX + 4;
constexpr int DF_RL = 8;           // 16B loads per fetcher lane per chunk
constexpr int DF_CH = 64 * DF_RL;  // 16B loads per chunk (a pair of warps)
constexpr int DF_IMAX = 1024;      // items decoded into shared memory
constexpr int DF_FDLY_NS = 1000;   // a chunk's first loads after its slot's writes
constexpr int DF_WAIT_NS = 2048;   // suspend hint of the mbarrier waits
constexpr int DF_FIXED =
    ((2 * DF_RED + DF_NBAR * 8 + (9 * DF_RMAX + 3 * DF_EMAX + 16 + DF_IMAX) * 4) + 1023) / 1024 * 1024;
constexpr int DF_SMEM = 232448;

struct DFMaps {
  CUtensorMap w13, w13h, w2;  // w13h: boxes of one 576B unit (16 rows) x nc chunks
};
struct DFArgs {
  const __nv_bfloat16* x;
  const int* ids;
  const float* tw;  // null: y per route (zeros for other ranks' routes); else the weighted sum into out
  const float* gs13;
  const float* gs2;
  int* state;                 // [0] CTAs done, [1] the last call's epoch
  unsigned long long* inter;  // [routes][I / 2] tagged words
  __nv_bfloat16* y;
  __nv_bfloat16* out;
  float scale;
  int T, topk, E, H, I, RB1, RB2, C1, C2, nc1, nc1h, nc2, off_x;
};
// decf_k's clamp and workspace size, a kernel parameter of their own. With them inside DFArgs, ptxas emits 5,368
// instead of 6,616 instructions for decf_k<false> and it runs about 1us slower (H100, Muse and DeepSeek EP8, 1 and 2
// tokens), even unused.
struct DFClamp {
  const float* gsr;  // up / gate global scale per expert, for the clamp on up (CLAMP only)
  int64_t nwords;    // tagged words in the workspace (all of them are zeroed when the epoch wraps)
  float lim;         // the SwiGLU clamp (CLAMP only)
};

__device__ __forceinline__ void st_relaxed_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
// two 8-byte words, each single-copy atomic: a word's epoch and its two values are seen together
__device__ __forceinline__ uint4 ld_relaxed_2u64(const unsigned long long* p) {
  unsigned long long a, c;
  asm volatile("ld.relaxed.gpu.global.v2.b64 {%0, %1}, [%2];" : "=l"(a), "=l"(c) : "l"(p) : "memory");
  return make_uint4((uint32_t)a, (uint32_t)(a >> 32), (uint32_t)c, (uint32_t)(c >> 32));
}
__device__ __forceinline__ void df_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n @!p bra W;\n}" ::"r"(bar),
      "r"(parity),
      "n"(DF_WAIT_NS)
      : "memory");
}
__device__ __forceinline__ uint64_t df_time() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// One stage for this consumer warp: chunks warp, warp + 8, ... (< nv), chunk stride cs, U units of 16 rows at uoff[u].
template <int U>
__device__ __forceinline__ void df_stage(
    float (&acc)[4][4],
    const uint8_t* sb,
    const int (&uoff)[4],
    int cs,
    int nv,
    const uint8_t* brow,
    int k0,
    int warp,
    int lane) {
  const int q = lane & 3;
  for (int c = warp; c < nv; c += DF_CW) {
    uint4 b0 = make_uint4(0, 0, 0, 0), b1 = b0;
    if (brow != nullptr) {
      const uint8_t* bp = brow + (k0 + c) * 128 + q * 32;
      b0 = *reinterpret_cast<const uint4*>(bp);
      b1 = *reinterpret_cast<const uint4*>(bp + 16);
    }
    const uint8_t* cb = sb + c * cs;
#pragma unroll
    for (int u = 0; u < U; u++) {
      const uint8_t* up = cb + uoff[u];
      const uint4 nb = *reinterpret_cast<const uint4*>(up + lane * 16);
      const uint32_t Sc = sc_bf16(*reinterpret_cast<const uint16_t*>(up + 512 + lane * 2), 0x4140);
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        uint32_t o[4];
        decode4(u4w(nb, kk), o);
        const uint32_t a0 = bmulh(o[0], Sc, 0), a1 = bmulh(o[1], Sc, 1), a2 = bmulh(o[2], Sc, 0),
                       a3 = bmulh(o[3], Sc, 1);
        const uint4& bv = kk < 2 ? b0 : b1;
        mma16816(acc[u], a0, a1, a2, a3, u4w(bv, 2 * (kk & 1)), u4w(bv, 2 * (kk & 1) + 1));
      }
    }
  }
}

// CLAMP: SwiGLU clamps its inputs at C.lim exactly as grouped_k's copy_out does (gate from above, up on both sides at
// lim / |gsr[e]|, since up carries the gate's global scale here and GEMM2's scale holds up / gate).
template <bool CLAMP>
__global__ void __launch_bounds__(DF_THREADS, 1)
    decf_k(const __grid_constant__ DFMaps mp, const DFArgs A, const __grid_constant__ DFClamp C) {
  extern __shared__ __align__(1024) uint8_t dsm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, G = gridDim.x, b = blockIdx.x;
  float* redb = reinterpret_cast<float*>(dsm);  // two hand-off buffers
  uint64_t* bars = reinterpret_cast<uint64_t*>(dsm + 2 * DF_RED);
  int* sids = reinterpret_cast<int*>(bars + DF_NBAR);
  int* slot_e = sids + DF_RMAX;
  int* slot_m = slot_e + DF_RMAX;
  int* slot_r0 = slot_m + DF_RMAX;
  int* routes = slot_r0 + DF_RMAX;                            // route indices by slot, increasing within a slot
  float* sgs13 = reinterpret_cast<float*>(routes + DF_RMAX);  // per slot
  float* sgs2 = sgs13 + DF_RMAX;
  float* stw = sgs2 + DF_RMAX;                            // per route
  float* egs = stw + DF_RMAX;                             // per expert: gs13 x 2^63, then gs2 x 2^63 at + DF_EMAX
  int* scnt = reinterpret_cast<int*>(egs + 2 * DF_EMAX);  // per expert: route count, then slot
  int* misc = scnt + DF_EMAX;  // [0] slots, [1] own GEMM1 items written, [2] row buffers (0: one row per route),
                               // [3] ring offset, [4] rows offset, [5] rows per buffer, [6] 16-row GEMM1 items
  int* itab = misc + 16;       // decoded items (see item)
  const uint32_t bar0 = (uint32_t)__cvta_generic_to_shared(bars);
  // barriers: stage full [DF_NS] / free [DF_NS], x rows, intermediate rows full [DF_NBMAX] / free [DF_NBMAX],
  // hand-off full [2] / free [2]
  const uint32_t bfull = bar0, bempty = bar0 + 8 * DF_NS, bx = bar0 + 16 * DF_NS, bif = bx + 8,
                 bie = bif + 8 * DF_NBMAX, brf = bie + 8 * DF_NBMAX, bre = brf + 16;
  const int R = A.T * A.topk, E = A.E;
  uint32_t ep = (uint32_t)A.state[1] + 1u;
  if (ep == 0) ep = 1;

  if (tid == 0) {
    for (int s = 0; s < DF_NS; s++) {
      mbar_init(bfull + 8 * s, 1);
      mbar_init(bempty + 8 * s, DF_CW);
    }
    mbar_init(bx, 1);
    for (int s = 0; s < DF_NBMAX; s++) {
      mbar_init(bif + 8 * s, 2);
      mbar_init(bie + 8 * s, DF_CW);
    }
    for (int s = 0; s < 2; s++) {
      mbar_init(brf + 8 * s, DF_CW);
      mbar_init(bre + 8 * s, 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < R; i += DF_THREADS) {
    sids[i] = A.ids[i];
    stw[i] = A.tw != nullptr ? A.tw[i] : 0.f;
  }
  for (int e = tid; e < E; e += DF_THREADS) {
    scnt[e] = 0;
    egs[e] = A.gs13[e] * 9.223372036854775808e18f;
    egs[DF_EMAX + e] = A.gs2[e] * 9.223372036854775808e18f;
  }
  __syncthreads();
  if (tid == DF_CW * 32) {  // every token's x row (padded by 16B); it needs no routing
    mbar_expect(bx, A.T * 2 * A.H);
    for (int t = 0; t < A.T; t++)
      bulk_g2s(
          (uint32_t)__cvta_generic_to_shared(dsm + A.off_x + t * (2 * A.H + 16)),
          A.x + (int64_t)t * A.H,
          2 * A.H,
          bx,
          pol_evict_first());
  }
  for (int i = tid; i < R; i += DF_THREADS)
    if ((unsigned)sids[i] < (unsigned)E) atomicAdd(&scnt[sids[i]], 1);
  __syncthreads();
  if (warp == 0) {
    int n = 0;
    for (int e0 = 0; e0 < E; e0 += 32) {
      const int e = e0 + lane, c = e < E ? scnt[e] : 0;
      const uint32_t bal = __ballot_sync(0xffffffffu, c > 0);
      if (c > 0) {
        const int pos = n + __popc(bal & ((1u << lane) - 1));
        slot_e[pos] = e;
        slot_m[pos] = c;
        scnt[e] = pos;
      }
      n += __popc(bal);
    }
    __syncwarp();
    for (int s = lane; s < n; s += 32) {
      sgs13[s] = egs[slot_e[s]];
      sgs2[s] = egs[DF_EMAX + slot_e[s]];
    }
    if (lane == 0) {
      int r0 = 0, mmax = 1;
      for (int s = 0; s < n; s++) {
        slot_r0[s] = r0;
        r0 += slot_m[s];
        mmax = max(mmax, slot_m[s]);
      }
      misc[0] = n;
      misc[1] = 0;
      // past the x rows: the ring, then the intermediate rows, one per local route when they fit
      const int rowi = 2 * A.I + 16;
      const int oring = (A.off_x + A.T * (2 * A.H + 16) + 1023) / 1024 * 1024;
      const int avail = DF_SMEM - oring - DF_NS * DF_STB;
      int nb = 0;
      if (r0 * rowi > avail || (r0 * (A.I / 4) + DF_CH - 1) / DF_CH > DF_NBMAX)
        nb = min(DF_NBMAX, avail / (mmax * rowi));
      misc[2] = nb;
      misc[3] = oring;
      misc[4] = oring + DF_NS * DF_STB;
      misc[5] = mmax;
      // 16-row GEMM1 items when they lower this grid's most GEMM1 stages per CTA
      const int k1 = (A.C1 + A.nc1 - 1) / A.nc1, k1h = (A.C1 + A.nc1h - 1) / A.nc1h;
      const int N1 = n * A.RB1 * 4;
      misc[6] = (2 * N1 + G - 1) / G * k1h < (N1 + G - 1) / G * k1;
    }
  }
  __syncthreads();
  for (int i = tid; i < R; i += DF_THREADS) {
    const int e = sids[i];
    if ((unsigned)e < (unsigned)E) {
      int rank = 0;
      for (int j = 0; j < i; j++)
        rank += sids[j] == e;
      routes[slot_r0[scnt[e]] + rank] = i;
    }
  }
  __syncthreads();
  const int n = misc[0], NB = misc[2], mmax = misc[5], g1h = misc[6];
  uint8_t* const ring = dsm + misc[3];
  uint8_t* const rows = dsm + misc[4];
  const int per1 = A.RB1 * (g1h ? 8 : 4);
  const int N1 = n * per1, P2 = A.RB2 * 8;
  const int n1c = N1 > b ? (N1 - 1 - b) / G + 1 : 0, n2c = P2 > b ? (P2 - 1 - b) / G + 1 : 0;
  const int nc1 = g1h ? A.nc1h : A.nc1;
  const int k1 = (A.C1 + nc1 - 1) / nc1, k2 = (A.C2 + A.nc2 - 1) / A.nc2;
  const int rowi = 2 * A.I + 16, rowx = 2 * A.H + 16;
  // A GEMM2 item is G2 consecutive slots of one piece when a slot's weights are one small block (k2 == 1). The
  // hand-off holds 8 routes, so G2 x (most routes of a slot) <= 8.
  const int G2 = k2 == 1 && NB == 0 ? max(1, min(min(8, DF_STB / (1152 * A.nc2)), 8 / mmax)) : 1;
  const int nG = (n + G2 - 1) / G2;
  const int nitems = n1c + n2c * nG;
  // item it: GEMM1 (it < n1c; slot s, row block rb, part hp) or GEMM2 (piece (it - n1c) / nG, slots [s, s + gsz)),
  // decoded once here and read by every role as kind << 31 | slot << 20 | row block << 6 | (gsz - 1) << 3 | part
  auto item_dec = [&](int it, int& kind, int& s, int& rb, int& hp, int& gsz) {
    if (it < n1c) {
      const int i = b + it * G;
      kind = 0, s = i / per1, rb = (i % per1) / (per1 / A.RB1), hp = i % (per1 / A.RB1), gsz = 1;
    } else {
      const int j = b + ((it - n1c) / nG) * G;
      kind = 1, s = ((it - n1c) % nG) * G2, rb = j >> 3, hp = j & 7, gsz = min(G2, n - s);
    }
  };
  for (int it = tid; it < min(nitems, DF_IMAX); it += DF_THREADS) {
    int kind, s, rb, hp, gsz;
    item_dec(it, kind, s, rb, hp, gsz);
    itab[it] = (int)(((uint32_t)kind << 31) | ((uint32_t)s << 20) | ((uint32_t)rb << 6) | ((uint32_t)(gsz - 1) << 3) |
                     (uint32_t)hp);
  }
  __syncthreads();
  auto item = [&](int it, int& kind, int& s, int& rb, int& hp, int& gsz) {
    if (it < DF_IMAX) {
      const uint32_t w = (uint32_t)itab[it];
      kind = (int)(w >> 31), s = (int)((w >> 20) & 0x7ff), rb = (int)((w >> 6) & 0x3fff);
      gsz = (int)((w >> 3) & 7) + 1, hp = (int)(w & 7);
    } else {
      item_dec(it, kind, s, rb, hp, gsz);
    }
  };
  // GEMM2 item j2's rows of slot s: the slot's own rows (NB == 0) or buffer j2 % NB
  auto rowbuf = [&](int j2, int s) { return NB == 0 ? rows + slot_r0[s] * rowi : rows + (j2 % NB) * mmax * rowi; };

  if (warp == DF_CW) {  // producer
    if (lane == 0) {
      const uint64_t pol = pol_evict_first();
      const uint32_t rs = (uint32_t)__cvta_generic_to_shared(ring);
      int k = 0;
      for (int it = 0; it < nitems; it++) {
        int kind, s, rb, hp, gsz;
        item(it, kind, s, rb, hp, gsz);
        const int e = slot_e[s];
        const int ks = kind ? k2 : k1;
        for (int j = 0; j < ks; j++, k++) {
          const int st = k % DF_NS;
          if (k >= DF_NS) df_wait(bempty + 8 * st, ((k / DF_NS) + 1) & 1);
          const uint32_t dst = rs + st * DF_STB, bar = bfull + 8 * st;
          if (kind == 0 && g1h) {  // unit hp & 1 of gate group 4 (hp >> 2) + ((hp >> 1) & 1) and of its up group
            const int h = hp >> 2, p = (hp >> 1) & 1, u = hp & 1, row = e * A.RB1 + rb;
            mbar_expect(bar, 2 * 576 * nc1);
            tma_4d_h(dst, &mp.w13h, 72 * u, 4 * h + p, j * nc1, row, bar, pol);
            tma_4d_h(dst + 576 * nc1, &mp.w13h, 72 * u, 4 * h + 2 + p, j * nc1, row, bar, pol);
          } else if (kind == 0) {
            const int h = hp >> 1, p = hp & 1, row = e * A.RB1 + rb;
            mbar_expect(bar, 2 * 1152 * nc1);
            tma_4d_h(dst, &mp.w13, 0, 4 * h + p, j * nc1, row, bar, pol);
            tma_4d_h(dst + 1152 * nc1, &mp.w13, 0, 4 * h + 2 + p, j * nc1, row, bar, pol);
          } else {
            mbar_expect(bar, gsz * 1152 * A.nc2);
            for (int u = 0; u < gsz; u++)
              tma_4d_h(dst + u * 1152 * A.nc2, &mp.w2, 0, hp, j * A.nc2, slot_e[s + u] * A.RB2 + rb, bar, pol);
          }
        }
      }
    }
  } else if (warp >= DF_CW + 2) {  // fetchers: pair fp takes chunks fp, fp + DF_NP, ...
    // A task is one chunk of DF_CH 16B loads: of every local route's row in slot order (one row per route; a chunk
    // can span slots), or, when rotating, of one GEMM2 item's rows. A rotating task t refills buffer t % NB after a
    // parity wait on its free barrier, which cannot tell phase c from c - 2, so no pair may reach a buffer two phases
    // ahead of the consumers. With NB >= DF_NP a pair at task t has passed task t - DF_NP >= t - NB; with fewer
    // buffers, pair b alone takes buffer b's tasks (pairs NB .. DF_NP - 1 idle), so it has passed t - NB.
    const int fw = warp - DF_CW - 2, fp = fw >> 1, wi = fw & 1;
    const int fstep = NB > 0 && NB < DF_NP ? NB : DF_NP;
    const int per = A.I / 4;
    const int nr = n > 0 ? slot_r0[n - 1] + slot_m[n - 1] : 0;
    const int ntask = n2c > 0 ? (NB == 0 ? (nr * per + DF_CH - 1) / DF_CH : n2c * n) : 0;
    for (int t = fp < fstep ? fp : ntask; t < ntask; t += fstep) {
      int pos0, f0, f1, s;  // first route position, flat range [f0, f1) of 16B words from it, the last slot
      uint8_t* bb;
      if (NB == 0) {
        f0 = t * DF_CH, f1 = min(nr * per, f0 + DF_CH), pos0 = 0, bb = rows;
        const int lastpos = (f1 - 1) / per;
        s = 0;
        while (s + 1 < n && slot_r0[s + 1] <= lastpos)
          s++;
      } else {
        s = t % n, pos0 = slot_r0[s], f0 = 0, f1 = slot_m[s] * per, bb = rowbuf(t, s);
        if (t >= NB) df_wait(bie + 8 * (t % NB), ((t / NB) + 1) & 1);
      }
      const int need = min(n1c, ((s + 1) * per1 + G - 1) / G);
      if (lane == 0) {
        while (*reinterpret_cast<volatile int*>(misc + 1) < need)
          __nanosleep(64);
        const uint64_t t0 = df_time();
        while (df_time() - t0 < (uint64_t)DF_FDLY_NS)
          __nanosleep(64);
      }
      __syncwarp();
      for (int c0 = f0; c0 < f1; c0 += DF_CH) {
        // one division per chunk: a lane's words are 64 apart, so its (row, column) advance by 64 with a carry
        uint4 v[DF_RL];
        const unsigned long long* sp[DF_RL];
        int doff[DF_RL];
        uint32_t pend = 0;
        {
          const int i0 = c0 + wi * 32 + lane;
          int row = i0 / per, col = i0 - row * per;
#pragma unroll
          for (int u = 0; u < DF_RL; u++) {
            const bool ok = i0 + u * 64 < f1;
            sp[u] = ok ? A.inter + (int64_t)routes[pos0 + row] * (A.I / 2) + col * 2 : A.inter;
            doff[u] = row * rowi + col * 8;
            pend |= (ok ? 1u : 0u) << u;
            col += 64;
            while (col >= per)
              col -= per, row++;
          }
        }
#pragma unroll
        for (int u = 0; u < DF_RL; u++)
          v[u] = ld_relaxed_2u64(sp[u]);
        while (true) {
#pragma unroll
          for (int u = DF_RL - 1; u >= 0; u--) {
            if (((pend >> u) & 1u) && v[u].y == ep && v[u].w == ep) {
              *reinterpret_cast<uint2*>(bb + doff[u]) = make_uint2(v[u].x, v[u].z);
              pend &= ~(1u << u);
            }
          }
          if (!__any_sync(0xffffffffu, pend != 0)) break;
#pragma unroll
          for (int u = 0; u < DF_RL; u++)
            if ((pend >> u) & 1u) v[u] = ld_relaxed_2u64(sp[u]);
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(bif + 8 * (NB == 0 ? t : t % NB));
    }
  } else if (warp == DF_CW + 1) {  // epilogue
    float wsum[DF_TMAX];
#pragma unroll
    for (int t = 0; t < DF_TMAX; t++)
      wsum[t] = 0.f;
    for (int it = 0; it < nitems; it++) {
      int kind, s, rb, hp, gsz;
      item(it, kind, s, rb, hp, gsz);
      const int hb = it & 1, m = slot_m[s], r0 = slot_r0[s];
      df_wait(brf + 8 * hb, (it >> 1) & 1);
      const float* red = redb + hb * (DF_RED / 4);
      const int jr = lane;
      if (kind == 0) {
        const float sc = sgs13[s];
        float ulim = 0.f;
        if constexpr (CLAMP) ulim = C.lim / fabsf(__ldg(C.gsr + slot_e[s]));
        // 32-row items: gate rows jr at red rows jr, up at 32 + jr; 16-row items: gate rows at jr, up at 16 + jr
        // (lanes 16 .. 31 idle)
        const int h = g1h ? hp >> 2 : hp >> 1, p = g1h ? (hp >> 1) & 1 : hp & 1;
        const int gi = g1h ? 16 * (hp & 1) + (jr & 15) : jr, uo = g1h ? 16 : 32;
        const int col = 128 * rb + 64 * h + dec_kp(32 * p + gi);
        for (int r = 0; r < m; r++) {
          float gt = 0.f, ut = 0.f;
#pragma unroll
          for (int w = 0; w < DF_CW; w++)
            gt += red[(w * 64 + (jr & (uo - 1))) * DF_TMAX + r],
                ut += red[(w * 64 + uo + (jr & (uo - 1))) * DF_TMAX + r];
          gt *= sc;
          float uv = ut * sc;
          if constexpr (CLAMP) {  // as SGLang's clamped SiLU: gate from above, up on both sides
            gt = gt > C.lim ? C.lim : gt;
            uv = uv > ulim ? ulim : (uv < -ulim ? -ulim : uv);
          }
          // lanes 2c and 2c + 1 hold adjacent intermediate columns: one tagged word per pair
          const uint32_t hv = __bfloat16_as_ushort(__float2bfloat16(gt / (1.f + __expf(-gt)) * uv));
          const uint32_t hn = __shfl_down_sync(0xffffffffu, hv, 1);
          if (!(lane & 1) && jr < uo)
            st_relaxed_u64(
                A.inter + (int64_t)routes[r0 + r] * (A.I / 2) + col / 2,
                ((unsigned long long)ep << 32) | (hv | (hn << 16)));
        }
        __syncwarp();
        if (lane == 0) {
          mbar_arrive(bre + 8 * hb);
          *reinterpret_cast<volatile int*>(misc + 1) = it + 1;
        }
      } else {
        // the group's partial sums: [warp][32 rows][8 routes], its routes in slot order
        const int row0 = 256 * rb + 32 * hp;
        int cum = 0;
        for (int sub = 0; sub < gsz; sub++) {
          const int ss = s + sub, ms = slot_m[ss], rs = slot_r0[ss];
          const float sc = sgs2[ss];
          for (int r = 0; r < ms; r++) {
            float v = 0.f;
#pragma unroll
            for (int w = 0; w < DF_CW; w++)
              v += red[(w * 32 + jr) * 8 + cum + r];
            const int route = routes[rs + r];
            if (A.tw == nullptr) {
              A.y[(int64_t)route * A.H + row0 + jr] = __float2bfloat16(v * sc);
            } else {
              const float c = stw[route] * (v * sc);
              const int t = route / A.topk;
#pragma unroll
              for (int tt = 0; tt < DF_TMAX; tt++)
                if (tt == t) wsum[tt] += c;
            }
          }
          cum += ms;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(bre + 8 * hb);
        if (s + gsz == n) {  // the piece's last group: its rows of out (or the zeros of other ranks' routes in y)
          if (A.tw != nullptr) {
#pragma unroll
            for (int t = 0; t < DF_TMAX; t++) {
              if (t < A.T) A.out[(int64_t)t * A.H + row0 + jr] = __float2bfloat16(wsum[t] * A.scale);
              wsum[t] = 0.f;
            }
          } else {
            for (int r = 0; r < R; r++)
              if ((unsigned)sids[r] >= (unsigned)E) A.y[(int64_t)r * A.H + row0 + jr] = __float2bfloat16(0.f);
          }
        }
      }
    }
    if (n == 0) {  // no local route: this CTA's pieces are zeros
      for (int it = 0; it < n2c; it++) {
        const int j = b + it * G, row0 = 256 * (j >> 3) + 32 * (j & 7);
        const int nr = A.tw != nullptr ? A.T : R;
        __nv_bfloat16* dst = A.tw != nullptr ? A.out : A.y;
        for (int r = 0; r < nr; r++)
          dst[(int64_t)r * A.H + row0 + lane] = __float2bfloat16(0.f);
      }
    }
  } else {  // consumers
    const int g = lane >> 2, q = lane & 3;
    int k = 0;
    if (n1c > 0) df_wait(bx, 0);  // else after the items: the x copies must land before this CTA exits
    for (int it = 0; it < nitems; it++) {
      int kind, s, rb, hp, gsz;
      item(it, kind, s, rb, hp, gsz);
      const int hb = it & 1;
      float acc[4][4];
      if (kind == 1) {  // GEMM2: the group's slots one after another from one stage (k2 stages when gsz is 1)
        const int j2 = it - n1c;
        if (it >= 2) df_wait(bre + 8 * hb, ((it >> 1) + 1) & 1);
        float* red = redb + hb * (DF_RED / 4);
        int cum = 0;
        for (int sub = 0; sub < gsz; sub++) {
          const int ss = s + sub, ms = slot_m[ss], rs = slot_r0[ss];
          if (NB == 0) {
            const int per = A.I / 4;
            for (int c = rs * per / DF_CH; c <= ((rs + ms) * per - 1) / DF_CH; c++)
              df_wait(bif + 8 * c, 0);
          } else {
            df_wait(bif + 8 * (j2 % NB), (j2 / NB) & 1);
          }
          const uint8_t* brow = g < ms ? rowbuf(j2, ss) + g * rowi : nullptr;
#pragma unroll
          for (int u = 0; u < 4; u++)
#pragma unroll
            for (int i = 0; i < 4; i++)
              acc[u][i] = 0.f;
          for (int j = 0; j < k2; j++) {
            const int st = k % DF_NS;
            if (sub == 0) df_wait(bfull + 8 * st, (k / DF_NS) & 1);
            const int nv = min(A.nc2, A.C2 - j * A.nc2);
            const int uo[4] = {0, 576, 0, 0};
            df_stage<2>(acc, ring + st * DF_STB + sub * 1152 * A.nc2, uo, 1152, nv, brow, j * A.nc2, warp, lane);
            if (sub == gsz - 1) {
              __syncwarp();
              if (lane == 0) mbar_arrive(bempty + 8 * st);
              k++;
            }
          }
          if (NB != 0) {
            __syncwarp();
            if (lane == 0) mbar_arrive(bie + 8 * (j2 % NB));
          }
#pragma unroll
          for (int u = 0; u < 2; u++) {
            float* rp = red + (warp * 32 + 16 * u + g) * 8 + cum;
            if (2 * q < ms) rp[2 * q] = acc[u][0], rp[64 + 2 * q] = acc[u][2];
            if (2 * q + 1 < ms) rp[2 * q + 1] = acc[u][1], rp[64 + 2 * q + 1] = acc[u][3];
          }
          cum += ms;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(brf + 8 * hb);
        continue;
      }
      // GEMM1
      const int m = slot_m[s], r0 = slot_r0[s];
      const uint8_t* brow = g < m ? dsm + A.off_x + (routes[r0 + g] / A.topk) * rowx : nullptr;
#pragma unroll
      for (int u = 0; u < 4; u++)
#pragma unroll
        for (int i = 0; i < 4; i++)
          acc[u][i] = 0.f;
      for (int j = 0; j < k1; j++, k++) {
        const int st = k % DF_NS;
        df_wait(bfull + 8 * st, (k / DF_NS) & 1);
        const uint8_t* sb = ring + st * DF_STB;
        const int nv = min(nc1, A.C1 - j * nc1);
        if (g1h) {
          const int uo[4] = {0, 576 * nc1, 0, 0};
          df_stage<2>(acc, sb, uo, 576, nv, brow, j * nc1, warp, lane);
        } else {
          const int uo[4] = {0, 576, 1152 * nc1, 1152 * nc1 + 576};
          df_stage<4>(acc, sb, uo, 1152, nv, brow, j * nc1, warp, lane);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(bempty + 8 * st);
      }
      if (it >= 2) df_wait(bre + 8 * hb, ((it >> 1) + 1) & 1);
      float* red = redb + hb * (DF_RED / 4);
      const int U = g1h ? 2 : 4;
#pragma unroll
      for (int u = 0; u < 4; u++) {
        if (u < U) {
          float* rp = red + (warp * 64 + 16 * u + g) * DF_TMAX;
          if (2 * q < m) rp[2 * q] = acc[u][0], rp[8 * DF_TMAX + 2 * q] = acc[u][2];
          if (2 * q + 1 < m) rp[2 * q + 1] = acc[u][1], rp[8 * DF_TMAX + 2 * q + 1] = acc[u][3];
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(brf + 8 * hb);
    }
    if (n1c == 0) df_wait(bx, 0);
  }
  __syncthreads();
  // The last CTA to finish advances the epoch (the next call's reads of it are ordered by the kernel boundary). When
  // the epoch would wrap, it zeroes every tagged word first and the next call starts again at epoch 1, so a word
  // written 2^32 - 1 calls ago is never taken as current (one thread, once in 2^32 - 1 calls, so no other thread
  // waits).
  if (tid == 0) {
    __threadfence();
    if (atom_add_acq_rel(A.state, 1) == G - 1) {
      if (ep == 0xFFFFFFFFu) {
        for (int64_t i = 0; i < C.nwords; i++)
          A.inter[i] = 0ull;
        ep = 0;
      }
      A.state[1] = (int)ep;
      A.state[0] = 0;
    }
  }
}

// The fused decode path: one decf_k launch (cooperative: every CTA must be resident), 1 to DF_TMAX tokens. ws (int64)
// holds the done count and epoch (word 0), then the tagged intermediate (routes x I / 2 words); it must be zero before
// its first call and is reused unchanged by later calls on the same stream.
inline bool decode_fused_fits(int64_t t, int64_t h, int64_t i) {
  const int64_t oring = (DF_FIXED + t * (2 * h + 16) + 1023) / 1024 * 1024;
  return oring + DF_NS * DF_STB + 2 * t * (2 * i + 16) <= DF_SMEM;
}
inline void run_decode_fused(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView w13,
    tvm::ffi::TensorView w2,
    tvm::ffi::TensorView gs13,
    tvm::ffi::TensorView gs2,
    tvm::ffi::TensorView gsr,
    tvm::ffi::TensorView ws,
    tvm::ffi::Optional<tvm::ffi::TensorView> y,
    tvm::ffi::Optional<tvm::ffi::TensorView> topk_weights,
    tvm::ffi::Optional<tvm::ffi::TensorView> out,
    double scale,
    double swiglu_limit,
    int64_t ctas) {
  using namespace host;
  auto T = SymbolicSize{"tokens"};
  auto H = SymbolicSize{"hidden"};
  auto K = SymbolicSize{"top_k"};
  auto R = SymbolicSize{"routes"};
  auto E = SymbolicSize{"local_experts"};
  auto device = SymbolicDevice{};
  TensorMatcher({T, H}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).ensure_alignment(16).verify(x);
  TensorMatcher({T, K}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
  TensorMatcher({E}).with_dtype<float>().with_device(device).verify(gs13);
  TensorMatcher({E}).with_dtype<float>().with_device(device).verify(gs2);
  TensorMatcher({E}).with_dtype<float>().with_device(device).verify(gsr);
  if (y.has_value())
    TensorMatcher({R, H}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(y.value());
  TensorMatcher({-1}).with_dtype<int64_t>().with_device(device).ensure_alignment(16).verify(ws);
  const int64_t t = T.unwrap(), h = H.unwrap(), topk = K.unwrap(), e = E.unwrap();
  const int64_t r = y.has_value() ? R.unwrap() : t * topk;
  RuntimeCheck(
      w13.dim() == 3 && h >= 64 && w13.size(1) % (h / 64 * 8) == 0,
      "nvfp4_w4a16_sm90: packed w13 must be [E, I / 64 * H / 64 * 8, 576]");
  // The padded width; the decode kernels read whole 256-row GEMM1 blocks, so a half block (I % 128 == 64) is refused
  // below and stays on the grouped kernels.
  const int64_t i = w13.size(1) / (h / 64 * 8) * 64;
  RuntimeCheck(r == t * topk, "nvfp4_w4a16_sm90: workspace rows ", r, " != tokens * top_k ", t * topk);
  RuntimeCheck(
      t >= 1 && t <= DF_TMAX && r <= DF_RMAX,
      "nvfp4_w4a16_sm90 decode_fused: at most ",
      DF_TMAX,
      " tokens and ",
      DF_RMAX,
      " routes");
  RuntimeCheck(h >= 256 && h % 256 == 0, "nvfp4_w4a16_sm90: hidden size must be a multiple of 256, got ", h);
  RuntimeCheck(
      i >= 128 && i % 128 == 0,
      "nvfp4_w4a16_sm90 decode: padded intermediate size must be a multiple of 128 (no half block), got ",
      i);
  RuntimeCheck(swiglu_limit > 0, "nvfp4_w4a16_sm90: swiglu_limit must be positive (inf for none)");
  RuntimeCheck(e >= 1 && e <= DF_EMAX, "nvfp4_w4a16_sm90: 1 to 1024 local experts, got ", e);
  const int RB1 = (int)(2 * i / 256), RB2 = (int)(h / 256), C1 = (int)(h / 64), C2 = (int)(i / 64);
  TensorMatcher({e, (int64_t)RB1 * C1 * 16, 576})
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(w13);
  TensorMatcher({e, RB2, C2, 16, 576}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w2);
  RuntimeCheck(out.has_value() == topk_weights.has_value(), "nvfp4_w4a16_sm90: out and topk_weights go together");
  RuntimeCheck(out.has_value() != y.has_value(), "nvfp4_w4a16_sm90 decode: y (per route) or out (weighted), not both");
  if (out.has_value()) {
    TensorMatcher({T, H}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(out.value());
    TensorMatcher({T, K}).with_dtype<float>().with_device(device).verify(topk_weights.value());
  }
  const DLDevice dev = device.unwrap();
  RuntimeCheck(runtime::get_sm_version(dev.device_id) == 90, "nvfp4_w4a16_sm90: needs an sm_90 (Hopper) GPU");
  const int sms = (int)runtime::get_sm_count(dev.device_id);
  RuntimeCheck(ws.size(0) >= 2 + r * i / 2, "nvfp4_w4a16_sm90 decode_fused: workspace too small");
  RuntimeCheck(decode_fused_fits(t, h, i), "nvfp4_w4a16_sm90 decode_fused: tokens x width too large for shared memory");
  const int nc1 = std::min(16, C1), nc1h = std::min(32, C1), nc2 = std::min(32, C2);
  auto wmap = [&](void* base, int C, int64_t blocks, int nc, int box0) {
    return host::tma::make_map(
        base,
        CU_TENSOR_MAP_DATA_TYPE_UINT64,
        {144, 8, (cuuint64_t)C, (cuuint64_t)blocks},
        {1152, 9216, (cuuint64_t)C * 9216},
        {(cuuint32_t)box0, 1, (cuuint32_t)nc, 1},
        {.l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_256B});
  };
  DFMaps mp;
  mp.w13 = wmap(w13.data_ptr(), C1, e * RB1, nc1, 144);
  mp.w13h = wmap(w13.data_ptr(), C1, e * RB1, nc1h, 72);
  mp.w2 = wmap(w2.data_ptr(), C2, e * RB2, nc2, 144);
  DFArgs A;
  A.x = static_cast<const __nv_bfloat16*>(x.data_ptr());
  A.ids = static_cast<const int*>(topk_ids.data_ptr());
  A.tw = topk_weights.has_value() ? static_cast<const float*>(topk_weights.value().data_ptr()) : nullptr;
  A.gs13 = static_cast<const float*>(gs13.data_ptr());
  A.gs2 = static_cast<const float*>(gs2.data_ptr());
  A.state = static_cast<int*>(ws.data_ptr());
  A.inter = static_cast<unsigned long long*>(ws.data_ptr()) + 2;
  A.y = y.has_value() ? static_cast<__nv_bfloat16*>(y.value().data_ptr()) : nullptr;
  A.out = out.has_value() ? static_cast<__nv_bfloat16*>(out.value().data_ptr()) : nullptr;
  A.scale = (float)scale;
  A.T = (int)t, A.topk = (int)topk, A.E = (int)e, A.H = (int)h, A.I = (int)i;
  A.RB1 = RB1, A.RB2 = RB2, A.C1 = C1, A.C2 = C2, A.nc1 = nc1, A.nc1h = nc1h, A.nc2 = nc2;
  A.off_x = DF_FIXED;
  const DFClamp C{static_cast<const float*>(gsr.data_ptr()), ws.size(0) - 2, (float)swiglu_limit};
  const int grid = ctas > 0 ? (int)std::min<int64_t>(ctas, sms) : sms;
  auto launch = [&](auto kernel) {
    runtime::set_max_dynamic_smem_per_device<decltype(kernel)::value>(dev, DF_SMEM);
    LaunchKernel(grid, DF_THREADS, dev, DF_SMEM).enable_cooperative(true)(decltype(kernel)::value, mp, A, C);
  };
  if (swiglu_limit < INFINITY)
    launch(KernelC<decf_k<true>>{});
  else
    launch(KernelC<decf_k<false>>{});
}

// The id of the CUDA graph capture on x's device's current stream, 0 when it is not capturing: the op keys
// decode_fused's workspace by it (see _fused_workspace).
inline int64_t capture_id(tvm::ffi::TensorView x) {
  using namespace host;
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  unsigned long long id = 0;
  RuntimeDeviceCheck(cudaStreamGetCaptureInfo(LaunchKernel::resolve_device(x.device()), &st, &id));
  return st == cudaStreamCaptureStatusActive ? (int64_t)id : 0;
}

// ---- decode, two launches (decp1_k, then decp2_k): no exchange between CTAs inside a launch ----
// decp1_k computes GEMM1 and SwiGLU into the BF16 intermediate. decp2_k is its programmatic dependent: both fit on
// one SM together (shared memory split between them, at most 2 x DP_THREADS x 88 registers), so decp2_k's CTAs start
// while decp1_k streams and fill their ring with GEMM2 weights (they need no GEMM1 output). Only its row loader waits
// for decp1_k to finish (griddepcontrol.wait), then reads every local route's intermediate row once, so the only
// global round trip on the critical path runs while no weight copy is queued. Items own rows over the full K (no K
// sum crosses CTAs), at 16 or 32 rows: GEMM1 picks per call the size that balances the slots' row groups over the CTAs,
// GEMM2 the size that balances H's rows.
constexpr int DP_CW = 8;
constexpr int DP_THREADS = (DP_CW + 3) * 32;  // consumers + producer, row loader, epilogue
constexpr int DP_TMAX = 4;
constexpr int DP_RMAX = 256;
constexpr int DP_EMAX = 1024;
constexpr int DP_NSMAX = 48;          // ring stages at most
constexpr int DP_SMEM_MAX = 232448;   // one CTA's dynamic shared memory at most on sm_90
constexpr int DP_SMEM_PAIR = 231424;  // both kernels' dynamic shared memory on one SM (228KB less 1KB per CTA)

struct DPMaps {
  CUtensorMap w13[2], w2;  // w13: boxes of 576B (uh 1) and 1152B (uh 2) x nc1 chunks; w2: 576 uh2 B x nc2 chunks
};
struct DPArgs {
  const __nv_bfloat16* x;
  const int* ids;
  const float* tw;  // null: y per route (zeros for other ranks' routes); else the weighted sum into out
  const float* gs13;
  const float* gs2;
  __nv_bfloat16* inter;  // [routes][I]
  __nv_bfloat16* y;
  __nv_bfloat16* out;
  float scale;
  int T, topk, E, H, I, RB1, RB2, C1, C2;
  int nc1a, nc1b, nc2, uh2;       // nc1a, nc1b: GEMM1 chunks per stage at uh 1, 2 (decp1_k picks uh per call)
  int ns, stb, off_b, off_fixed;  // ring stages and bytes per stage; B rows (x or intermediate); the rest
  const uint8_t* w2raw;           // the packed w2 (decp2_k reads small pieces of it with 16B copies)
  int cpa;                        // decp2_k: 16B copies (stages under 9KB) instead of tensor copies
  const float* gsr;               // up / gate global scale per expert, for the clamp on up (decp1_k with CLAMP only)
  float lim;                      // SwiGLU clamp, read only when CLAMP
};

// Routing, the same in both kernels: slots are experts with local routes in increasing id, each slot's routes in route
// order. rt holds sids[R], routes[R], slot_e, slot_m, slot_r0 [R each], scnt[E], misc[4]; returns the slot count.
__device__ __forceinline__ int dp_route(const int* ids, int R, int E, int* rt) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int *sids = rt, *routes = rt + R, *slot_e = rt + 2 * R, *slot_m = rt + 3 * R, *slot_r0 = rt + 4 * R,
      *scnt = rt + 5 * R, *misc = scnt + E;
  for (int i = tid; i < R; i += blockDim.x)
    sids[i] = ids[i];
  for (int e = tid; e < E; e += blockDim.x)
    scnt[e] = 0;
  __syncthreads();
  for (int i = tid; i < R; i += blockDim.x)
    if ((unsigned)sids[i] < (unsigned)E) atomicAdd(&scnt[sids[i]], 1);
  __syncthreads();
  if (warp == 0) {
    int n = 0;
    for (int e0 = 0; e0 < E; e0 += 32) {
      const int e = e0 + lane, c = e < E ? scnt[e] : 0;
      const uint32_t bal = __ballot_sync(0xffffffffu, c > 0);
      if (c > 0) {
        const int pos = n + __popc(bal & ((1u << lane) - 1));
        slot_e[pos] = e;
        slot_m[pos] = c;
        scnt[e] = pos;
      }
      n += __popc(bal);
    }
    if (lane == 0) {
      int r0 = 0;
      for (int s = 0; s < n; s++) {
        slot_r0[s] = r0;
        r0 += slot_m[s];
      }
      misc[0] = n;
    }
  }
  __syncthreads();
  for (int i = tid; i < R; i += blockDim.x) {
    const int e = sids[i];
    if ((unsigned)e < (unsigned)E) {
      int rank = 0;
      for (int j = 0; j < i; j++)
        rank += sids[j] == e;
      routes[slot_r0[scnt[e]] + rank] = i;
    }
  }
  __syncthreads();
  return misc[0];
}

// 16B global -> shared copy through the load path (L2 only), and an arrive on bar once this thread's earlier ones land.
// Bulk and tensor copies run about one at a time per SM (a 9KB copy and a 36KB copy each take about 1.2us under
// load), so small pieces go this way.
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_arrive(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}

// One stage for this consumer warp: chunks c0, c0 + cstep, ... (< nv valid) of cs bytes each, nu units of 16 rows (576B
// apart in a chunk) into accumulators ub .. ub + nu - 1; brow is this lane's B row.
__device__ __forceinline__ void dp_stage(
    float (&acc)[4][4],
    const uint8_t* sb,
    int ub,
    int nu,
    int cs,
    int nv,
    const uint8_t* brow,
    int k0,
    int c0,
    int cstep,
    int lane) {
  const int q = lane & 3;
  for (int c = c0; c < nv; c += cstep) {
    uint4 b0 = make_uint4(0, 0, 0, 0), b1 = b0;
    if (brow != nullptr) {
      const uint8_t* bp = brow + (k0 + c) * 128 + q * 32;
      b0 = *reinterpret_cast<const uint4*>(bp);
      b1 = *reinterpret_cast<const uint4*>(bp + 16);
    }
    const uint8_t* cb = sb + c * cs;
#pragma unroll
    for (int u = 0; u < 4; u++) {
      if (u < ub || u >= ub + nu) continue;
      const uint8_t* up = cb + 576 * (u - ub);
      const uint4 nb = *reinterpret_cast<const uint4*>(up + lane * 16);
      const uint32_t Sc = sc_bf16(*reinterpret_cast<const uint16_t*>(up + 512 + lane * 2), 0x4140);
#pragma unroll
      for (int kk = 0; kk < 4; kk++) {
        uint32_t o[4];
        decode4(u4w(nb, kk), o);
        const uint32_t a0 = bmulh(o[0], Sc, 0), a1 = bmulh(o[1], Sc, 1), a2 = bmulh(o[2], Sc, 0),
                       a3 = bmulh(o[3], Sc, 1);
        const uint4& bv = kk < 2 ? b0 : b1;
        mma16816(acc[u], a0, a1, a2, a3, u4w(bv, 2 * (kk & 1)), u4w(bv, 2 * (kk & 1) + 1));
      }
    }
  }
}

// This warp's K share of an item (nu units of 16 rows, m routes) into hand-off buffer red [warp][16 nu][T].
__device__ __forceinline__ void
dp_handoff(float* red, const float (&acc)[4][4], int nu, int m, int T, int warp, int lane) {
  const int g = lane >> 2, q = lane & 3, RR = 16 * nu;
#pragma unroll
  for (int u = 0; u < 4; u++) {
    if (u >= nu) break;
    float* rp = red + (warp * RR + 16 * u + g) * T;
    if (2 * q < m) rp[2 * q] = acc[u][0], rp[8 * T + 2 * q] = acc[u][2];
    if (2 * q + 1 < m) rp[2 * q + 1] = acc[u][1], rp[8 * T + 2 * q + 1] = acc[u][3];
  }
}

// GEMM1 row-group size for n slots: 32 rows (uh 2) unless 16-row items spread the work over the CTAs better.
__device__ __forceinline__ int dp_uh1(int n, int RB1, int G) {
  const int w2 = (n * RB1 * 4 + G - 1) / G * 2, w1 = (n * RB1 * 8 + G - 1) / G;
  return w1 < w2 ? 1 : 2;
}

// CLAMP: SwiGLU clamps its inputs at A.lim, as in decf_k.
template <bool CLAMP>
__global__ void __launch_bounds__(DP_THREADS, 2) decp1_k(const __grid_constant__ DPMaps mp, const DPArgs A) {
  extern __shared__ __align__(1024) uint8_t dsm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, G = gridDim.x, b = blockIdx.x;
  // decp2_k may start now: it reads no output of this grid before its griddepcontrol.wait
  if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int R = A.T * A.topk, E = A.E, NS = A.ns;
  constexpr int RRMAX = 64;
  const int redf = DP_CW * RRMAX * A.T;  // floats per hand-off buffer
  float* redb = reinterpret_cast<float*>(dsm + A.off_fixed);
  uint64_t* bars = reinterpret_cast<uint64_t*>(redb + 2 * redf);
  int* rt = reinterpret_cast<int*>(bars + 2 * DP_NSMAX + 8);
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(dsm);
  const uint32_t bar0 = (uint32_t)__cvta_generic_to_shared(bars);
  const uint32_t bfull = bar0, bempty = bar0 + 8 * DP_NSMAX, bx = bar0 + 16 * DP_NSMAX, brf = bx + 8, bre = bx + 24;
  if (tid == 0) {
    for (int s = 0; s < NS; s++) {
      mbar_init(bfull + 8 * s, 1);
      mbar_init(bempty + 8 * s, DP_CW);
    }
    mbar_init(bx, 1);
    for (int s = 0; s < 2; s++) {
      mbar_init(brf + 8 * s, DP_CW);
      mbar_init(bre + 8 * s, 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid == (DP_CW + 1) * 32) {  // every token's x row (padded by 16B); it needs no routing
    mbar_expect(bx, A.T * 2 * A.H);
    for (int t = 0; t < A.T; t++)
      bulk_g2s(ring + A.off_b + t * (2 * A.H + 16), A.x + (int64_t)t * A.H, 2 * A.H, bx, pol_evict_first());
  }
  // every expert's global scale in shared memory, loaded with the ids (one round trip before the weight stream)
  float* gsa = reinterpret_cast<float*>(rt + 5 * R + E + 4);
  for (int e = tid; e < E; e += blockDim.x)
    gsa[e] = A.gs13[e];
  const int n = dp_route(A.ids, R, E, rt);
  const int *routes = rt + R, *slot_e = rt + 2 * R, *slot_m = rt + 3 * R, *slot_r0 = rt + 4 * R;
  const int uh = dp_uh1(n, A.RB1, G);
  const int gpb = 8 / uh;  // items per 256-row block
  const int per = A.RB1 * gpb, N1 = n * per;
  const int n1c = N1 > b ? (N1 - 1 - b) / G + 1 : 0;
  const int nc = uh == 1 ? A.nc1a : A.nc1b, cs = 576 * uh, ks = (A.C1 + nc - 1) / nc, RR = 32 * uh;
  // item it: slot s, row block rb, half h, pair p, 16-row sub-unit u (uh 1)
  auto item = [&](int it, int& s, int& rb, int& h, int& p, int& u) {
    const int i = b + it * G, q = i % per, gq = q % gpb;
    s = i / per, rb = q / gpb;
    if (uh == 2)
      h = gq >> 1, p = gq & 1, u = 0;
    else
      h = gq >> 2, p = (gq >> 1) & 1, u = gq & 1;
  };

  if (warp == DP_CW) {  // producer
    if (lane == 0) {
      const uint64_t pol = pol_evict_first();
      const CUtensorMap* map = &mp.w13[uh - 1];
      int k = 0;
      for (int it = 0; it < n1c; it++) {
        int s, rb, h, p, u;
        item(it, s, rb, h, p, u);
        const int row = slot_e[s] * A.RB1 + rb;
        for (int j = 0; j < ks; j++, k++) {  // a stage: the gate rows, then the up rows, of nc chunks
          const int st = k % NS;
          if (k >= NS) mbar_wait(bempty + 8 * st, ((k / NS) + 1) & 1);
          const uint32_t dst = ring + st * A.stb, bar = bfull + 8 * st;
          mbar_expect(bar, 2 * cs * nc);
          tma_4d_h(dst, map, 72 * u, 4 * h + p, j * nc, row, bar, pol);
          tma_4d_h(dst + cs * nc, map, 72 * u, 4 * h + 2 + p, j * nc, row, bar, pol);
        }
      }
    }
  } else if (warp == DP_CW + 2) {  // epilogue: SwiGLU into the intermediate
    for (int it = 0; it < n1c; it++) {
      int s, rb, h, p, u;
      item(it, s, rb, h, p, u);
      const int hb = it & 1, e = slot_e[s], m = slot_m[s], r0 = slot_r0[s];
      mbar_wait(brf + 8 * hb, (it >> 1) & 1);
      const float* red = redb + hb * redf;
      const int jr = lane, half = 16 * uh;
      const float sc = gsa[e] * 9.223372036854775808e18f;
      float ulim = 0.f;
      if constexpr (CLAMP) ulim = A.lim / fabsf(__ldg(A.gsr + e));
      const int col = 128 * rb + 64 * h + dec_kp(32 * p + 16 * u + (jr & (half - 1)));
      for (int r = 0; r < m; r++) {
        float gt = 0.f, ut = 0.f;
        if (jr < half) {
#pragma unroll
          for (int w = 0; w < DP_CW; w++)
            gt += red[(w * RR + jr) * A.T + r], ut += red[(w * RR + half + jr) * A.T + r];
        }
        gt *= sc;
        float uv = ut * sc;
        if constexpr (CLAMP) {
          gt = gt > A.lim ? A.lim : gt;
          uv = uv > ulim ? ulim : (uv < -ulim ? -ulim : uv);
        }
        const uint32_t hv = __bfloat16_as_ushort(__float2bfloat16(gt / (1.f + __expf(-gt)) * uv));
        const uint32_t hn = __shfl_down_sync(0xffffffffu, hv, 1);
        if (jr < half && !(jr & 1))
          *reinterpret_cast<uint32_t*>(A.inter + (int64_t)routes[r0 + r] * A.I + col) = hv | (hn << 16);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(bre + 8 * hb);
    }
  } else if (warp < DP_CW) {  // consumers
    int k = 0;
    const int rowx = 2 * A.H + 16, g = lane >> 2;
    mbar_wait(bx, 0);  // always: the x copies are in flight even with no item
    for (int it = 0; it < n1c; it++) {
      int s, rb, h, p, u;
      item(it, s, rb, h, p, u);
      const int m = slot_m[s], r0 = slot_r0[s];
      const uint8_t* brow = g < m ? dsm + A.off_b + (routes[r0 + g] / A.topk) * rowx : nullptr;
      float acc[4][4];
#pragma unroll
      for (int a = 0; a < 4; a++)
#pragma unroll
        for (int i = 0; i < 4; i++)
          acc[a][i] = 0.f;
      for (int j = 0; j < ks; j++, k++) {
        const int st = k % NS;
        mbar_wait(bfull + 8 * st, (k / NS) & 1);
        const int nv = min(nc, A.C1 - j * nc);
        const uint8_t* sb = dsm + st * A.stb;
        dp_stage(acc, sb, 0, uh, cs, nv, brow, j * nc, warp, DP_CW, lane);
        dp_stage(acc, sb + cs * nc, uh, uh, cs, nv, brow, j * nc, warp, DP_CW, lane);
        __syncwarp();
        if (lane == 0) mbar_arrive(bempty + 8 * st);
      }
      const int hb = it & 1;
      if (it >= 2) mbar_wait(bre + 8 * hb, ((it >> 1) + 1) & 1);
      dp_handoff(redb + hb * redf, acc, 2 * uh, m, A.T, warp, lane);
      __syncwarp();
      if (lane == 0) mbar_arrive(brf + 8 * hb);
    }
  }
}

__global__ void __launch_bounds__(DP_THREADS, 2) decp2_k(const __grid_constant__ DPMaps mp, const DPArgs A) {
  // Work units are (piece of 16 uh output rows, slot), window by window, then piece, then slot. The eight consumer
  // warps form 8 / F groups of F warps; group i takes every (8 / F)-th unit and its F warps split the unit's chunks.
  // Each warp adds its share of route r's sum, times the slot's scale (and r's top-k weight), into wacc with shared
  // atomics, so no warp waits on another until the end. Every unit's addresses are tabulated once, in parallel: the
  // loops do no integer division (a runtime division is a chain of about 20 dependent instructions, and at these
  // sizes the dependent chains, not the bytes, set the time).
  extern __shared__ __align__(1024) uint8_t dsm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, G = gridDim.x, b = blockIdx.x;
  const int R = A.T * A.topk, E = A.E, NS = A.ns, uh = A.uh2, RR = 16 * uh;
  const int ppb = 16 / uh, P2 = A.RB2 * ppb;  // pieces of 16 uh output rows
  const int n2c = P2 > b ? (P2 - 1 - b) / G + 1 : 0, n2max = (P2 + G - 1) / G, UM = n2max * R;
  const int W = A.tw != nullptr ? A.T : R;  // wacc columns: tokens (weighted sum) or routes (y per route)
  // weighted sum: per consumer warp [piece][T][32], plain adds; y per route: [piece][R][32], shared atomics
  const int WS = n2max * W * 32, NWS = A.tw != nullptr ? DP_THREADS / 32 : 1;
  float* wacc = reinterpret_cast<float*>(dsm + A.off_fixed);
  long long* usrc = reinterpret_cast<long long*>(wacc + max(DP_THREADS / 32 * n2max * A.T, n2max * R) * 32);
  uint64_t* bars = reinterpret_cast<uint64_t*>(usrc + UM);
  int* rt = reinterpret_cast<int*>(bars + 2 * DP_NSMAX + 2);
  float* gsa = reinterpret_cast<float*>(rt + 5 * R + E + 4);  // per expert: gs2 x 2^63
  float* rtw = gsa + E;                                       // per route: its top-k weight
  int* rcol = reinterpret_cast<int*>(rtw + R);                // per route: its wacc column (token or route)
  int* wins = rcol + R;                                       // window starts (slot index), [nw + 1]
  int* woff = wins + DP_RMAX + 1;                             // window starts (unit index), [nw + 1]
  int* urow = woff + DP_RMAX + 1;  // per unit: first row in the window's rows, routes, first route, piece, map coords
  int* um = urow + UM;
  int* ur0 = um + UM;
  int* upi = ur0 + UM;
  int* uc = upi + UM;   // 72 u | hp << 8
  int* uerb = uc + UM;  // e RB2 + rb
  float* usc = reinterpret_cast<float*>(uerb + UM);
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(dsm);
  const uint32_t bar0 = (uint32_t)__cvta_generic_to_shared(bars);
  const uint32_t bfull = bar0, bempty = bar0 + 8 * DP_NSMAX;
  for (int i = tid; i < NWS * WS; i += blockDim.x)
    wacc[i] = 0.f;
  for (int i = tid; i < R; i += blockDim.x)
    rcol[i] = A.tw != nullptr ? i / A.topk : i;
  // ids, topk weights and scales come from kernels that finished before decp1_k started, so they are read before the
  // wait, all at once: one round trip before decp1_k's weight stream fills the memory queues
  if (A.tw != nullptr)
    for (int i = tid; i < R; i += blockDim.x)
      rtw[i] = A.tw[i];
  for (int e = tid; e < E; e += blockDim.x)
    gsa[e] = A.gs2[e] * 9.223372036854775808e18f;
  const int n = dp_route(A.ids, R, E, rt);
  const int *routes = rt + R, *slot_e = rt + 2 * R, *slot_m = rt + 3 * R, *slot_r0 = rt + 4 * R;
  // Windows: consecutive slots whose routes fit the rows region (one window unless the call has more local routes
  // than the host sized for).
  const int rowi = 2 * A.I + 16, rcap = (A.off_fixed - A.off_b) / rowi;
  const int nc = A.nc2, cs = 576 * uh, ks = (A.C2 + nc - 1) / nc, U = n2c * n;
  // F: warps per unit, the power of two that minimizes (units per group) x (chunks per warp); on ties the smallest,
  // since each unit also costs a fixed chain of shared-memory reads per warp
  // With the ring reused (not one shot), a group's next unit is (8 / F) ks stages on, so (8 / F) ks <= NS keeps every
  // wait within one phase of its stage (a group a full ring ahead would read a stale phase's parity as done).
  const bool fits = A.cpa && U * ks <= NS && (n == 0 || slot_r0[n - 1] + slot_m[n - 1] <= rcap);  // one shot
  int F = 8;
  for (int f = 4; f >= 1; f >>= 1)
    if ((fits || (8 / f) * ks <= NS) &&
        (U + 8 / f - 1) / (8 / f) * ((A.C2 + f - 1) / f) <= (U + 8 / F - 1) / (8 / F) * ((A.C2 + F - 1) / F))
      F = f;
  if (tid == 0) {
    int nw = 0, r0 = 0;
    wins[0] = 0;
    for (int s = 0; s < n; s++)
      if (slot_r0[s] + slot_m[s] - r0 > rcap) wins[++nw] = s, r0 = slot_r0[s];
    wins[++nw] = n;
    rt[5 * R + E + 1] = nw;
    woff[0] = 0;
    for (int w = 0; w < nw; w++)
      woff[w + 1] = woff[w] + n2c * (wins[w + 1] - wins[w]);
    for (int s = 0; s < NS; s++) {
      mbar_init(bfull + 8 * s, A.cpa ? 32 : 1);
      mbar_init(bempty + 8 * s, F);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int nw = n > 0 ? rt[5 * R + E + 1] : 0;
  for (int q = tid; q < U; q += blockDim.x) {
    int w = 0;
    while (q >= woff[w + 1])
      w++;
    const int l = q - woff[w], nsw = wins[w + 1] - wins[w], pi = l / nsw, s = wins[w] + l % nsw;
    const int j = b + pi * G, rb = j / ppb, qq = j % ppb, hp = qq / (2 / uh), u = qq % (2 / uh), e = slot_e[s];
    usrc[q] = ((long long)(e * A.RB2 + rb) * A.C2) * 9216 + hp * 1152 + u * 576;
    urow[q] = slot_r0[s] - slot_r0[wins[w]];
    um[q] = slot_m[s];
    ur0[q] = slot_r0[s];
    upi[q] = pi;
    uc[q] = 72 * u | hp << 8;
    uerb[q] = e * A.RB2 + rb;
    usc[q] = gsa[e];
  }
  __syncthreads();
  const int pc = cs / 16;  // 16B pieces per chunk

  // 16B copies (small stages): every warp issues a share of the first NS stages now, a warp per stage (one warp
  // issuing them all takes about 0.75us per stage). Later stages are issued by the warp that frees the slot when a
  // unit has one warp (F == 1), else by the producer warp.
  const bool multi = A.cpa && nw == 1, total_fit = U * ks <= NS, crefill = multi && F == 1;
  // With no producer warp needed (16B copies, one warp per unit), every warp of the CTA consumes, as long as a group's
  // next unit stays within one ring of its last (the same rule as for F above, now with 11 groups).
  const int NCW = (crefill || (multi && total_fit && F == 1)) && (total_fit || (DP_THREADS / 32) * ks <= NS)
                      ? DP_THREADS / 32
                      : DP_CW;
  // Stages in use. A reused ring is safe only when one group fills and consumes every slot: a group's parity wait on a
  // slot then always follows its own wait on that slot's previous fill. Groups take units round robin, so that holds
  // when the stages in use are a multiple of (groups x stages per unit). Otherwise a group can wait on a slot whose
  // previous fill belongs to another group and is still in flight (copies complete out of order), read the parity of
  // the fill two back as done, and free the slot early; the producer then arrives twice on one phase (a device
  // exception, seen with another kernel holding most SMs).
  const int gNc = NCW / F;
  const int NU = total_fit || gNc == 1 ? NS : NS - NS % (gNc * ks);
  auto issue_cp = [&](int k) {  // stage k into its slot by this warp's 32 lanes, then their arrives
    const int q = k / ks, j = k - q * ks, nv = min(nc, A.C2 - j * nc);
    const uint8_t* src = A.w2raw + usrc[q] + (long long)j * nc * 9216;
    const uint32_t dst = ring + (k % NU) * A.stb;
    for (int c = 0; c < nv; c++)
      for (int o = lane; o < pc; o += 32)
        cp_async16(dst + c * cs + o * 16, src + (long long)c * 9216 + o * 16);
    cp_async_arrive(bfull + 8 * (k % NU));
  };
  if (multi)
    for (int k = warp; k < min(NU, U * ks); k += DP_THREADS / 32)
      issue_cp(k);
  if (warp == DP_CW && !crefill && !(multi && total_fit)) {  // producer for the stages not issued above
    const uint64_t pol = pol_evict_first();
    for (int k = multi ? NU : 0; k < U * ks; k++) {
      const int st = k % NU, q = k / ks, j = k - q * ks;
      if (k >= NU) mbar_wait(bempty + 8 * st, ((k / NU) + 1) & 1);
      if (A.cpa) {
        issue_cp(k);
      } else if (lane == 0) {
        mbar_expect(bfull + 8 * st, cs * nc);
        tma_4d_h(ring + st * A.stb, &mp.w2, uc[q] & 255, uc[q] >> 8, j * nc, uerb[q], bfull + 8 * st, pol);
      }
    }
  }
  if (warp < NCW) {  // consumers; they also load each window's intermediate rows once decp1_k has finished
    const int g = lane >> 2, qd = lane & 3, gN = NCW / F, grp = warp / F, wg = warp % F;
    GDC_WAIT();
    for (int w = 0; w < nw && n2c > 0; w++) {
      if (w > 0) asm volatile("bar.sync 1, %0;" ::"r"(NCW * 32) : "memory");  // every warp is done with the last rows
      {
        const int per = A.I / 8, wr0 = slot_r0[wins[w]];  // 16B per load
        constexpr int L = 4;
        const int nl = slot_r0[wins[w + 1] - 1] + slot_m[wins[w + 1] - 1] - wr0;
        for (int i0 = 0; i0 < nl * per; i0 += NCW * 32 * L) {
          uint4 v[L];
#pragma unroll
          for (int u = 0; u < L; u++) {
            const int i = min(i0 + u * NCW * 32 + tid, nl * per - 1);
            v[u] = ld_cg4(A.inter + (int64_t)routes[wr0 + i / per] * A.I + (i % per) * 8);
          }
#pragma unroll
          for (int u = 0; u < L; u++) {
            const int i = i0 + u * NCW * 32 + tid;
            if (i < nl * per) *reinterpret_cast<uint4*>(dsm + A.off_b + (i / per) * rowi + (i % per) * 16) = v[u];
          }
        }
      }
      asm volatile("bar.sync 1, %0;" ::"r"(NCW * 32) : "memory");
      const int q0 = woff[w] + ((grp - woff[w] % gN) + gN) % gN;
      for (int q = q0; q < woff[w + 1]; q += gN) {
        const int m = um[q];
        const uint8_t* brow = g < m ? dsm + A.off_b + (urow[q] + g) * rowi : nullptr;
        float acc[4][4];
#pragma unroll
        for (int a = 0; a < 4; a++)
#pragma unroll
          for (int i = 0; i < 4; i++)
            acc[a][i] = 0.f;
        for (int j = 0; j < ks; j++) {
          const int k = q * ks + j, st = k % NU;
          mbar_wait(bfull + 8 * st, (k / NU) & 1);
          const int nv = min(nc, A.C2 - j * nc);
          dp_stage(acc, dsm + st * A.stb, 0, uh, cs, nv, brow, j * nc, wg, F, lane);
          __syncwarp();
          if (crefill) {
            if (k + NU < U * ks) issue_cp(k + NU);
          } else if (lane == 0 && !(multi && total_fit)) {
            mbar_arrive(bempty + 8 * st);
          }
        }
        // this warp's share: rows 16 u + g (+ 8), routes 2 qd, 2 qd + 1 of the slot
        const float sc = usc[q];
        const int r0 = ur0[q], wbase = upi[q] * W;
#pragma unroll
        for (int u = 0; u < 2; u++) {
          if (u >= uh) break;
#pragma unroll
          for (int c = 0; c < 2; c++) {
            if (2 * qd + c >= m) continue;
            const int route = routes[r0 + 2 * qd + c];
            float* wp = wacc + (wbase + rcol[route]) * 32 + 16 * u + g;
            if (A.tw != nullptr) {  // this warp's own sums: no other warp writes them
              const float f = sc * rtw[route];
              wp += warp * WS;
              wp[0] += f * acc[u][c];
              wp[8] += f * acc[u][2 + c];
            } else {
              atomicAdd(wp, sc * acc[u][c]);
              atomicAdd(wp + 8, sc * acc[u][2 + c]);
            }
          }
        }
      }
    }
    asm volatile("bar.sync 1, %0;" ::"r"(NCW * 32) : "memory");
    // each piece's rows of out, or of y for every route (zeros for other ranks' routes)
    for (int i = tid; i < n2c * W * RR; i += NCW * 32) {
      const int pi = i / (W * RR), c = (i / RR) % W, jr = i % RR;
      const int j = b + pi * G, qq = j % ppb;
      const int row = 256 * (j / ppb) + 32 * (qq / (2 / uh)) + 16 * (qq % (2 / uh)) + jr;
      float v = 0.f;
      for (int ww = 0; ww < NWS; ww++)
        v += wacc[ww * WS + (pi * W + c) * 32 + jr];
      if (A.tw != nullptr)
        A.out[(int64_t)c * A.H + row] = __float2bfloat16(v * A.scale);
      else
        A.y[(int64_t)c * A.H + row] = __float2bfloat16(v);
    }
  }
}

struct DPCarveout {};
// The two-launch decode path: decp1_k, then decp2_k as its programmatic dependent, 1 to DP_TMAX tokens. inter
// ([routes][I] BF16) needs no initial value.
inline void run_decode_pdl(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView w13,
    tvm::ffi::TensorView w2,
    tvm::ffi::TensorView gs13,
    tvm::ffi::TensorView gs2,
    tvm::ffi::TensorView gsr,
    tvm::ffi::TensorView inter,
    tvm::ffi::Optional<tvm::ffi::TensorView> y,
    tvm::ffi::Optional<tvm::ffi::TensorView> topk_weights,
    tvm::ffi::Optional<tvm::ffi::TensorView> out,
    double scale,
    double swiglu_limit,
    int64_t ctas) {
  using namespace host;
  auto T = SymbolicSize{"tokens"};
  auto H = SymbolicSize{"hidden"};
  auto K = SymbolicSize{"top_k"};
  auto R = SymbolicSize{"routes"};
  auto E = SymbolicSize{"local_experts"};
  auto II = SymbolicSize{"intermediate"};
  auto device = SymbolicDevice{};
  TensorMatcher({T, H}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).ensure_alignment(16).verify(x);
  TensorMatcher({T, K}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
  TensorMatcher({E}).with_dtype<float>().with_device(device).verify(gs13);
  TensorMatcher({E}).with_dtype<float>().with_device(device).verify(gs2);
  TensorMatcher({E}).with_dtype<float>().with_device(device).verify(gsr);
  if (y.has_value())
    TensorMatcher({R, H}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(y.value());
  TensorMatcher({R, II}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(inter);
  const int64_t t = T.unwrap(), h = H.unwrap(), topk = K.unwrap(), r = R.unwrap(), e = E.unwrap();
  RuntimeCheck(
      w13.dim() == 3 && h >= 64 && w13.size(1) % (h / 64 * 8) == 0,
      "nvfp4_w4a16_sm90: packed w13 must be [E, I / 64 * H / 64 * 8, 576]");
  // The padded width; the decode kernels read whole 256-row GEMM1 blocks, so a half block (I % 128 == 64) is refused
  // below and stays on the grouped kernels.
  const int64_t i = w13.size(1) / (h / 64 * 8) * 64;
  RuntimeCheck(II.unwrap() == i, "nvfp4_w4a16_sm90 decode_pdl: intermediate width");
  RuntimeCheck(r == t * topk, "nvfp4_w4a16_sm90: workspace rows ", r, " != tokens * top_k ", t * topk);
  RuntimeCheck(
      t >= 1 && t <= DP_TMAX && r <= DP_RMAX,
      "nvfp4_w4a16_sm90 decode_pdl: at most ",
      DP_TMAX,
      " tokens and ",
      DP_RMAX,
      " routes");
  RuntimeCheck(h >= 256 && h % 256 == 0, "nvfp4_w4a16_sm90: hidden size must be a multiple of 256, got ", h);
  RuntimeCheck(
      i >= 128 && i % 128 == 0,
      "nvfp4_w4a16_sm90 decode: padded intermediate size must be a multiple of 128 (no half block), got ",
      i);
  RuntimeCheck(swiglu_limit > 0, "nvfp4_w4a16_sm90: swiglu_limit must be positive (inf for none)");
  RuntimeCheck(e >= 1 && e <= DP_EMAX, "nvfp4_w4a16_sm90: 1 to 1024 local experts, got ", e);
  const int RB1 = (int)(2 * i / 256), RB2 = (int)(h / 256), C1 = (int)(h / 64), C2 = (int)(i / 64);
  TensorMatcher({e, (int64_t)RB1 * C1 * 16, 576})
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(16)
      .verify(w13);
  TensorMatcher({e, RB2, C2, 16, 576}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w2);
  RuntimeCheck(out.has_value() == topk_weights.has_value(), "nvfp4_w4a16_sm90: out and topk_weights go together");
  RuntimeCheck(out.has_value() != y.has_value(), "nvfp4_w4a16_sm90 decode: y (per route) or out (weighted), not both");
  if (out.has_value()) {
    TensorMatcher({T, H}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(out.value());
    TensorMatcher({T, K}).with_dtype<float>().with_device(device).verify(topk_weights.value());
  }
  const DLDevice dev = device.unwrap();
  RuntimeCheck(runtime::get_sm_version(dev.device_id) == 90, "nvfp4_w4a16_sm90: needs an sm_90 (Hopper) GPU");
  const int sms = (int)runtime::get_sm_count(dev.device_id);
  const int G = ctas > 0 ? (int)std::min<int64_t>(ctas, sms) : sms;
  // GEMM2 piece size: 32 rows unless 16-row pieces balance H's rows over the CTAs better
  const int P32 = (int)(h / 32);
  const int uh2 = ((2 * P32 + G - 1) / G) < ((P32 + G - 1) / G) * 2 ? 1 : 2;
  const int nc1a = (int)std::min<int64_t>(32, C1), nc1b = (int)std::min<int64_t>(16, C1);
  const int nc2 = (int)std::min<int64_t>(64 / uh2, C2);
  const int stba = 2 * 1152 * nc1b, stbb = 576 * uh2 * nc2;  // gate and up copies; nc1a at uh 1: the same bytes
  auto rnd = [](int64_t v) { return (int)((v + 127) / 128 * 128); };
  const int route_b = (int)(6 * r + 2 * e + 8) * 4, bars_b = (2 * DP_NSMAX + 8) * 8;
  const int xrows = rnd(t * (2 * h + 16));
  const int n2max = (int)((h / 16 / uh2 + G - 1) / G);
  const int fixa = 2 * DP_CW * 64 * (int)t * 4 + bars_b + route_b;
  const int fixb = (int)std::max<int64_t>(DP_THREADS / 32 * n2max * t, n2max * r) * 32 * 4 + (int)r * 4 +
                   n2max * (int)r * (8 + 7 * 4) + (2 * DP_NSMAX + 2) * 8 + route_b + 2 * (DP_RMAX + 1) * 4;
  const int nsa = 2;  // GEMM1 ring stages (more measured slower)
  const int smem_a = nsa * stba + xrows + fixa;
  // decp2_k's intermediate rows: room for rcap routes at once (more local routes run in windows of rcap). Default:
  // all routes if three ring stages still fit, else as many as leave three stages, at least one slot's routes.
  const int rowb = (int)(2 * i + 16), budget_b = DP_SMEM_PAIR - smem_a - fixb - 128;
  const int rcap = (int)std::min<int64_t>(r, std::max<int64_t>(t, std::min<int64_t>(r, (budget_b - 3 * stbb) / rowb)));
  RuntimeCheck(rcap >= t, "nvfp4_w4a16_sm90 decode_pdl: intermediate rows under one slot");
  const int irows = rnd((int64_t)rcap * rowb);
  const int nsb_fit = std::min<int>(DP_NSMAX, (budget_b + 128 - irows) / stbb);
  RuntimeCheck(nsb_fit >= 2, "nvfp4_w4a16_sm90 decode_pdl: tokens x width too large for shared memory");
  const int smem_b = nsb_fit * stbb + irows + fixb;
  auto wmap = [&](void* base, int C, int64_t blocks, int nc, int uh) {
    return host::tma::make_map(
        base,
        CU_TENSOR_MAP_DATA_TYPE_UINT64,
        {144, 8, (cuuint64_t)C, (cuuint64_t)blocks},
        {1152, 9216, (cuuint64_t)C * 9216},
        {(cuuint32_t)(72 * uh), 1, (cuuint32_t)nc, 1},
        {.l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_256B});
  };
  DPMaps mp;
  mp.w13[0] = wmap(w13.data_ptr(), C1, e * RB1, nc1a, 1);
  mp.w13[1] = wmap(w13.data_ptr(), C1, e * RB1, nc1b, 2);
  mp.w2 = wmap(w2.data_ptr(), C2, e * RB2, nc2, uh2);
  DPArgs A;
  A.x = static_cast<const __nv_bfloat16*>(x.data_ptr());
  A.ids = static_cast<const int*>(topk_ids.data_ptr());
  A.tw = topk_weights.has_value() ? static_cast<const float*>(topk_weights.value().data_ptr()) : nullptr;
  A.gs13 = static_cast<const float*>(gs13.data_ptr());
  A.gs2 = static_cast<const float*>(gs2.data_ptr());
  A.inter = static_cast<__nv_bfloat16*>(inter.data_ptr());
  A.y = y.has_value() ? static_cast<__nv_bfloat16*>(y.value().data_ptr()) : nullptr;
  A.out = out.has_value() ? static_cast<__nv_bfloat16*>(out.value().data_ptr()) : nullptr;
  A.scale = (float)scale;
  A.T = (int)t, A.topk = (int)topk, A.E = (int)e, A.H = (int)h, A.I = (int)i;
  A.RB1 = RB1, A.RB2 = RB2, A.C1 = C1, A.C2 = C2;
  A.nc1a = nc1a, A.nc1b = nc1b, A.nc2 = nc2, A.uh2 = uh2;
  A.w2raw = static_cast<const uint8_t*>(w2.data_ptr());
  A.cpa = stbb < 9216;
  A.gsr = static_cast<const float*>(gsr.data_ptr());
  A.lim = (float)swiglu_limit;
  const bool clamp = swiglu_limit < INFINITY;
  runtime::init_per_device<DPCarveout>(dev, [] {
    RuntimeDeviceCheck(cudaFuncSetAttribute(decp1_k<false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    RuntimeDeviceCheck(cudaFuncSetAttribute(decp1_k<true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    RuntimeDeviceCheck(cudaFuncSetAttribute(decp2_k, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  });
  if (clamp)
    runtime::set_max_dynamic_smem_per_device<decp1_k<true>>(dev, DP_SMEM_MAX);
  else
    runtime::set_max_dynamic_smem_per_device<decp1_k<false>>(dev, DP_SMEM_MAX);
  runtime::set_max_dynamic_smem_per_device<decp2_k>(dev, DP_SMEM_MAX);
  DPArgs B = A;
  A.ns = nsa, A.stb = stba, A.off_b = nsa * stba, A.off_fixed = A.off_b + xrows;
  B.ns = nsb_fit, B.stb = stbb, B.off_b = nsb_fit * stbb, B.off_fixed = B.off_b + irows;
  if (clamp)
    LaunchKernel(G, DP_THREADS, dev, smem_a)(decp1_k<true>, mp, A);
  else
    LaunchKernel(G, DP_THREADS, dev, smem_a)(decp1_k<false>, mp, A);
  LaunchKernel(G, DP_THREADS, dev, smem_b).enable_pdl(true)(decp2_k, mp, B);
}

}  // namespace sglang::nvfp4_w4a16_sm90
