// Copyright 2026 SGLang Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Ported from the KDA GLM-4.5 fused-MoE artifact at commit 9a86621.
// Native CUDA fused MoE for GLM-4.5 FP8 (H200 / sm_90a).
//
// Pipeline (all on the caller's stream):
//   k_quant_hidden : per-token dynamic FP8 quant of hidden [M,5120]
//   k_count/k_scan/k_scatter : expert-sorted, BLOCK_M-padded pair index
//   k_gemm1 : grouped GEMM1 (wgmma e4m3) + fused SiLU*up + per-row FP8 requant
//   k_gemm2 : grouped GEMM2 (wgmma) + dequant * routing weight, scatter bf16
//   k_combine : sum 9 pair rows * 2.5 -> bf16 hidden, in place
//
// Reference semantics: SGLang fused_experts_impl with per-channel weight scales,
// dynamic per-token activation scales (amax/448, no eps), gate = cols [0,192),
// up = cols [192,384), fast-math silu in fp32. Both GEMMs accumulate the whole
// K inside the fp8 tensor core (ascending k32 wgmma chain) to reproduce the
// baseline triton tl.dot numerics on Hopper (reduced-precision accumulation).

#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define DEV __device__ __forceinline__
#include "wgmma.cuh"

// ---------------------------------------------------------------------------
// mbarrier + TMA helpers (sm_90a).
DEV void mbar_init(uint64_t* m, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(m)), "r"(count));
}
DEV void mbar_arrive_expect_tx(uint64_t* m, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(m)), "r"(bytes));
}
DEV void mbar_wait(uint64_t* m, uint32_t phase) {
  asm volatile(
      "{\n.reg .pred P1;\nLAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE;\n"
      "bra LAB_WAIT;\nDONE:\n}\n" ::"r"(smem_u32(m)),
      "r"(phase));
}
DEV void tma_load_2d(const CUtensorMap* desc, uint32_t dst, int32_t c0, int32_t c1, uint64_t* mbar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(desc)),
      "r"(c0),
      "r"(c1),
      "r"(smem_u32(mbar))
      : "memory");
}

// Programmatic dependent launch: k_gemm1 signals early so the (PDL-launched)
// GEMM2 grid becomes schedulable and gap-fills SMs freed by finished GEMM1
// CTAs; per-block ready flags provide the actual data dependency.
DEV void gdc_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;");
}
DEV void flag_release(int* f) {
  asm volatile("st.release.gpu.global.b32 [%0], 1;" ::"l"(f) : "memory");
}
DEV void flag_acquire_spin(const int* f) {
  int v;
  do {
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(f) : "memory");
  } while (v == 0);
}

namespace {

constexpr int HID = 5120;  // hidden size == GEMM1 K
constexpr int N1 = 384;    // gate+up width
constexpr int IMD = 192;   // intermediate width == GEMM2 K
constexpr int N2 = 5120;   // GEMM2 output width
constexpr int BM = 64;     // rows per sorted block
constexpr int BK1 = 128;   // GEMM1 K bytes per stage (SW128 atom)
constexpr int TOPK = 9;

// 3 stages x (A 64x128 + B 384x128): the 3-deep ring is the minimum that
// legalizes a deferred wgmma_wait<1> (buffer recycled two iterations after its
// chain was committed). smX fp32[64][192] and the epilogue h-stash reuse the
// staging smem after the final drain.
constexpr int K1_STAGES = 4;
constexpr int K1_SMEM = K1_STAGES * (BM * BK1 + N1 * BK1);  // 229376 B
constexpr int K1W_SMEM = 2 * (BM * BK1 + N1 * BK1);         // 114688 B

DEV void cp_async16(uint32_t dst, const void* src, bool full) {
  const int sz = full ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}

DEV void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int N>
DEV void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// SW128 K-major smem: 16B chunk (row r, chunk c of its 128B row) lands at
// r*128 + ((c ^ (r & 7)) * 16) -- the canonical swizzle wgmma descriptors expect.
DEV uint32_t sw128_off(int r, int c) {
  return r * 128 + ((c ^ (r & 7)) << 4);
}

DEV uint16_t pack2_fp8(float x, float y) {
  return static_cast<uint16_t>(__nv_cvt_float2_to_fp8x2(make_float2(x, y), __NV_SATFINITE, __NV_E4M3));
}

// The reference quant/activation kernels are compiled with --use_fast_math:
// their float divisions lower to div.full.f32, whose last-ulp differences vs
// IEEE div.rn propagate through the stored per-token scales into every
// dequantized element. Replicate the exact instruction.
DEV float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

DEV float bf16r(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

// ---------------------------------------------------------------------------
// Grid-stride over tokens with a bounded grid (<=3 CTAs/SM): the kernel is
// DRAM-bound, and leaving SM slots free lets the side-stream sort chain
// (M>=512) or the PDL route kernel (M<512) co-schedule instead of queueing
// behind thousands of quant blocks.
__global__ void k_quant_hidden(
    const __nv_bfloat16* __restrict__ hidden,
    uint8_t* __restrict__ Aq,
    float* __restrict__ a1s,
    int* __restrict__ tokcnt,
    int M) {
  const int tid = threadIdx.x;  // 128 threads
  gdc_launch_dependents();      // k_route is independent: let it run concurrently
  for (int t = blockIdx.x; t < M; t += gridDim.x) {
    if (tid == 0) tokcnt[t] = 0;
    const __nv_bfloat16* row = hidden + static_cast<size_t>(t) * HID;

    float v[40];
    float m = 0.f;
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      const int vec = tid + i * 128;  // 0..639, 8 bf16 each
      const uint4 raw = *(reinterpret_cast<const uint4*>(row) + vec);
      const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float f = __bfloat162float(pb[j]);
        v[i * 8 + j] = f;
        m = fmaxf(m, fabsf(f));
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    __shared__ float wmax[4];
    if ((tid & 31) == 0) wmax[tid >> 5] = m;
    __syncthreads();
    const float bm = fmaxf(fmaxf(wmax[0], wmax[1]), fmaxf(wmax[2], wmax[3]));
    const float scale = div_full(bm, 448.f);
    const float inv = scale == 0.f ? 0.f : div_full(1.f, scale);
    if (tid == 0) a1s[t] = scale;

#pragma unroll
    for (int i = 0; i < 5; ++i) {
      const int vec = tid + i * 128;
      uint16_t q[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        q[j] = pack2_fp8(v[i * 8 + 2 * j] * inv, v[i * 8 + 2 * j + 1] * inv);
      *(reinterpret_cast<uint2*>(Aq + static_cast<size_t>(t) * HID) + vec) = *reinterpret_cast<uint2*>(q);
    }
    __syncthreads();  // wmax reuse across the token loop
  }                   // token loop
}

// ---------------------------------------------------------------------------
// Fused small-M routing: one CTA does count + prefix + scatter + padding in a
// single launch (replaces both memsets, k_count, k_scan and k_scatter). Only
// used when P is small enough that one CTA beats four launch latencies.
__global__ void k_route(
    const int* __restrict__ tki,
    int P,
    int E,
    int maxRows,
    int* __restrict__ rowOff,
    int* __restrict__ ebk,
    int* __restrict__ nbt,
    int* __restrict__ sorted,
    int* __restrict__ flags,
    int maxBlocks) {
  // Signal immediately: the PDL-launched k_quant_hidden is fully independent
  // (disjoint buffers), so it may start on the other SMs while this single
  // CTA runs the routing critical path.
  gdc_launch_dependents();
  for (int i = threadIdx.x; i < maxBlocks; i += blockDim.x)
    flags[i] = 0;
  __shared__ int sCnt[1024];
  __shared__ int sFill[1024];
  __shared__ int sOff[1025];
  __shared__ int sWs[8];
  const int tid = threadIdx.x;  // 1024 threads
  if (tid < E) {
    sCnt[tid] = 0;
    sFill[tid] = 0;
  }
  __syncthreads();
  for (int i = tid; i < P; i += blockDim.x)
    atomicAdd(sCnt + tki[i], 1);
  __syncthreads();
  // Padded-count exclusive prefix over experts via warp-shuffle scan
  // (requires E < 256; integer math, bitwise-identical to the serial loop).
  int excl = 0;
  if (tid < 256) {
    const int pad = tid < E ? ((sCnt[tid] + BM - 1) / BM) * BM : 0;
    int incl = pad;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, incl, d);
      if ((tid & 31) >= d) incl += y;
    }
    if ((tid & 31) == 31) sWs[tid >> 5] = incl;
    excl = incl - pad;
  }
  __syncthreads();
  if (tid < 8) {
    int w = sWs[tid];
#pragma unroll
    for (int d = 1; d < 8; d <<= 1) {
      const int y = __shfl_up_sync(0xffu, w, d);
      if (tid >= d) w += y;
    }
    sWs[tid] = w;
  }
  __syncthreads();
  if (tid < 256) {
    const int wbase = tid >= 32 ? sWs[(tid >> 5) - 1] : 0;
    const int off = wbase + excl;
    if (tid <= E) sOff[tid] = off;
    if (tid == E) *nbt = off / BM;
  }
  __syncthreads();
  const int total = sOff[E];
  for (int e = tid; e <= E; e += blockDim.x)
    rowOff[e] = sOff[e];
  for (int b = tid; b < total / BM; b += blockDim.x) {
    const int r = b * BM;
    int lo = 0, hi = E - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (sOff[mid] <= r)
        lo = mid;
      else
        hi = mid - 1;
    }
    ebk[b] = lo;
  }
  // Padding tails: slots [sOff[e]+cnt[e], sOff[e+1]) get -1, filled per
  // expert (at most 63 writes each) instead of a per-slot binary search.
  for (int e = tid; e < E; e += blockDim.x) {
    const int end = sOff[e + 1];
    for (int s = sOff[e] + sCnt[e]; s < end; ++s)
      sorted[s] = -1;
  }
  __syncthreads();
  for (int i = tid; i < P; i += blockDim.x) {
    const int e = tki[i];
    const int pos = sOff[e] + atomicAdd(sFill + e, 1);
    sorted[pos] = i;
  }
}

__global__ void k_count(const int* __restrict__ tki, int P, int* __restrict__ cnt) {
  // Hierarchical histogram: per-CTA smem bins (E < 256), one global add per
  // (CTA, expert) -- 67K same-address global atomics were 61% drain-stalled.
  __shared__ int sCnt[256];
  const int tid = threadIdx.x;  // 256 threads
  sCnt[tid] = 0;
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + tid; i < P; i += gridDim.x * blockDim.x)
    atomicAdd(sCnt + tki[i], 1);
  __syncthreads();
  const int v = sCnt[tid];
  if (v != 0) atomicAdd(cnt + tid, v);
}

__global__ void k_scan(
    const int* __restrict__ cnt,
    int E,
    int* __restrict__ rowOff,
    int* __restrict__ ebk,
    int* __restrict__ nbt,
    int* __restrict__ flags,
    int maxBlocks) {
  for (int i = threadIdx.x; i < maxBlocks; i += blockDim.x)
    flags[i] = 0;
  __shared__ int sOff[1025];
  __shared__ int sWs[8];
  const int tid = threadIdx.x;  // 256 threads
  // Padded-count exclusive prefix via warp-shuffle scan (requires E < 256).
  int excl = 0;
  {
    const int pad = tid < E ? ((cnt[tid] + BM - 1) / BM) * BM : 0;
    int incl = pad;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, incl, d);
      if ((tid & 31) >= d) incl += y;
    }
    if ((tid & 31) == 31) sWs[tid >> 5] = incl;
    excl = incl - pad;
  }
  __syncthreads();
  if (tid < 8) {
    int w = sWs[tid];
#pragma unroll
    for (int d = 1; d < 8; d <<= 1) {
      const int y = __shfl_up_sync(0xffu, w, d);
      if (tid >= d) w += y;
    }
    sWs[tid] = w;
  }
  __syncthreads();
  {
    const int wbase = tid >= 32 ? sWs[(tid >> 5) - 1] : 0;
    const int off = wbase + excl;
    if (tid <= E) sOff[tid] = off;
    if (tid == E) {
      *nbt = off / BM;
      nbt[1] = 0;  // gemm2 work-stealing counter (nbt slot is 16B)
    }
  }
  __syncthreads();
  for (int e = threadIdx.x; e <= E; e += blockDim.x)
    rowOff[e] = sOff[e];
  const int total = sOff[E] / BM;
  for (int b = threadIdx.x; b < total; b += blockDim.x) {
    const int r = b * BM;
    int lo = 0, hi = E - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (sOff[mid] <= r)
        lo = mid;
      else
        hi = mid - 1;
    }
    ebk[b] = lo;
  }
}

__global__ void k_scatter(
    const int* __restrict__ tki,
    int P,
    const int* __restrict__ rowOff,
    int* __restrict__ fill,
    int* __restrict__ sorted) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < P) {
    const int e = tki[i];
    const int pos = rowOff[e] + atomicAdd(fill + e, 1);
    sorted[pos] = i;
  }
}

// ---------------------------------------------------------------------------
// GEMM1: C[64,384] = Aq[rows,5120] x w1[e]^T via wgmma e4m3 (QGMMA), fused
// silu(gate)*up + per-row FP8 requant. Accumulation matches the baseline's
// triton tl.dot(acc=...) on sm90: whole-K in-tensor-core chain, ascending k32
// (max_num_imprecise_acc default 2^30 => no fp32 promotion).
// Persistent grid: CTAs loop over row-blocks so partial tail waves do not
// idle whole SMs (at M~129 the second wave would be 22% occupied).
__global__ __launch_bounds__(256, 1) void k_gemm1(
    const uint8_t* __restrict__ Aq,
    const float* __restrict__ a1s,
    const __grid_constant__ CUtensorMap w1map,
    const float* __restrict__ w1s,
    const int* __restrict__ sorted,
    const int* __restrict__ ebk,
    const int* __restrict__ nbt,
    uint8_t* __restrict__ interq,
    float* __restrict__ a2s,
    int* __restrict__ flags) {
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* smA = smem;                         // [3][64][128]
  uint8_t* smB = smem + K1_STAGES * BM * BK1;  // [3][384][128]
  // Epilogue staging [64][IMDP] fp32: row stride padded 192->196 floats so
  // rows rotate across banks (192%32==0 made same-column accesses from the
  // 8 gid-rows of a warp an 8-way bank conflict on every smX/smH op).
  constexpr int IMDP = IMD + 4;
  float* smX = reinterpret_cast<float*>(smem);
  float* smH = reinterpret_cast<float*>(smem + 64 * IMDP * 4);  // h stash

  __shared__ int sPair[BM];
  __shared__ int sTok[BM];
  __shared__ float sA1s[BM];
  __shared__ __align__(8) uint64_t mFull[K1_STAGES];

  const int tid = threadIdx.x;
  if (tid < K1_STAGES) mbar_init(&mFull[tid], 1);
  int srun = 0;  // global stage counter: mbarrier buf/phase continue across mb

  const int nblocks = *nbt;
  gdc_launch_dependents();  // let the GEMM2 grid start gap-filling freed SMs
  for (int mb = blockIdx.x; mb < nblocks; mb += gridDim.x) {
    const int e = ebk[mb];
    if (tid < BM) {
      const int p = sorted[mb * BM + tid];
      sPair[tid] = p;
      const int tok = p >= 0 ? p / TOPK : 0;
      sTok[tid] = tok;
      sA1s[tid] = p >= 0 ? a1s[tok] : 0.f;
    }
    __syncthreads();

    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp & 3, wn = warp >> 2;  // wn = warpgroup: 0 gate, 1 up
    const int gid = lane >> 2, tig = lane & 3;

    float acc[96];
#pragma unroll
    for (int i = 0; i < 96; ++i)
      acc[i] = 0.f;

    auto issue = [&](int kstep, int buf) {
      const int k0 = kstep * BK1;
      // A: 64 rows x 8 chunks of 16B = 512 -> 2 per thread (gathered rows).
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int lin = i * 256 + tid;
        const int r = lin >> 3, c = lin & 7;
        const uint32_t dst = smem_u32(smA + buf * BM * BK1 + sw128_off(r, c));
        const uint8_t* src = Aq + static_cast<size_t>(sTok[r]) * HID + k0 + c * 16;
        cp_async16(dst, src, sPair[r] >= 0);
      }
      cp_async_commit();
      // B: whole 384x128B tile via 2 TMA loads (box outer dim max is 256);
      // the tensor map's 128B swizzle writes the same SW128 layout the wgmma
      // descriptors expect. One elected thread issues; mbarrier counts bytes.
      if (tid == 0) {
        mbar_arrive_expect_tx(&mFull[buf], N1 * BK1);
        const uint32_t d = smem_u32(smB + buf * N1 * BK1);
        tma_load_2d(&w1map, d, k0, e * N1, &mFull[buf]);
        tma_load_2d(&w1map, d + 192 * BK1, k0, e * N1 + 192, &mFull[buf]);
      }
    };

    constexpr int KSTEPS = HID / BK1;  // 40
    const int sbase = srun;
    srun += KSTEPS;
    issue(0, sbase % K1_STAGES);
    for (int it = 0; it < KSTEPS; ++it) {
      const int s = sbase + it;
      if (it + 1 < KSTEPS) {
        issue(it + 1, (s + 1) % K1_STAGES);
        cp_async_wait<1>();
      } else {
        cp_async_wait<0>();
      }
      mbar_wait(&mFull[s % K1_STAGES], (s / K1_STAGES) & 1);
      __syncthreads();
      fence_proxy_async();
      const uint32_t a0 = smem_u32(smA + (s % K1_STAGES) * BM * BK1);
      const uint32_t b0 = smem_u32(smB + (s % K1_STAGES) * N1 * BK1 + wn * 192 * BK1);
      wgmma_fence();
#pragma unroll
      for (int kc = 0; kc < 4; ++kc)
        wgmma_n192(acc, make_desc_sw128(a0 + kc * 32), make_desc_sw128(b0 + kc * 32), (it == 0 && kc == 0) ? 0 : 1);
      wgmma_commit();
      // Keep one group in flight. Single-barrier WAR safety (4-stage ring):
      // the buffer issue(it+1) overwrites was last read by wgmma group it-3,
      // which wait<1> at the end of iteration it-2 retires; barrier #1 of
      // iteration it-1 (passed by every thread before issuing at it) certifies
      // all warps completed iteration it-2, so no trailing in-loop barrier is
      // needed. The final iteration drains fully and the post-loop barrier
      // orders cross-warpgroup reads before smX/smH overwrite the staging ring.
      if (it + 1 < KSTEPS) {
        wgmma_wait<1>();
      } else {
        wgmma_wait<0>();
      }
    }
    __syncthreads();

    // Epilogue: dequant, silu(gate)*up, per-row amax, fp8 requant.
    const int r0 = wm * 16 + gid, r1 = r0 + 8;
    const float a10 = sA1s[r0], a11 = sA1s[r1];
    // Match reference numerics: GEMM1 output and silu*up are both rounded to
    // bf16 before the FP8 requant, and silu uses the baseline's fast-math
    // forms (__expf + approximate division from --use_fast_math).
    if (wn == 0) {
#pragma unroll
      for (int nt = 0; nt < 24; ++nt) {
        const int n = nt * 8 + tig * 2;
        const float s0 = w1s[static_cast<size_t>(e) * N1 + n];
        const float s1 = w1s[static_cast<size_t>(e) * N1 + n + 1];
        const float g00 = bf16r(acc[4 * nt + 0] * (a10 * s0));
        const float g01 = bf16r(acc[4 * nt + 1] * (a10 * s1));
        const float g10 = bf16r(acc[4 * nt + 2] * (a11 * s0));
        const float g11 = bf16r(acc[4 * nt + 3] * (a11 * s1));
        smX[r0 * IMDP + n] = div_full(g00, 1.f + __expf(-g00));
        smX[r0 * IMDP + n + 1] = div_full(g01, 1.f + __expf(-g01));
        smX[r1 * IMDP + n] = div_full(g10, 1.f + __expf(-g10));
        smX[r1 * IMDP + n + 1] = div_full(g11, 1.f + __expf(-g11));
      }
    }
    __syncthreads();
    if (wn == 1) {
      // h values are stashed in smem (each thread reads back only its own
      // slots) to keep the register count under the 255 cap alongside acc[96].
      float m0 = 0.f, m1 = 0.f;
#pragma unroll
      for (int nt = 0; nt < 24; ++nt) {
        const int i = nt * 8 + tig * 2;
        const int n = IMD + i;
        const float s0 = w1s[static_cast<size_t>(e) * N1 + n];
        const float s1 = w1s[static_cast<size_t>(e) * N1 + n + 1];
        const float u00 = bf16r(acc[4 * nt + 0] * (a10 * s0));
        const float u01 = bf16r(acc[4 * nt + 1] * (a10 * s1));
        const float u10 = bf16r(acc[4 * nt + 2] * (a11 * s0));
        const float u11 = bf16r(acc[4 * nt + 3] * (a11 * s1));
        const float h00 = bf16r(smX[r0 * IMDP + i] * u00);
        const float h01 = bf16r(smX[r0 * IMDP + i + 1] * u01);
        const float h10 = bf16r(smX[r1 * IMDP + i] * u10);
        const float h11 = bf16r(smX[r1 * IMDP + i + 1] * u11);
        smH[r0 * IMDP + i] = h00;
        smH[r0 * IMDP + i + 1] = h01;
        smH[r1 * IMDP + i] = h10;
        smH[r1 * IMDP + i + 1] = h11;
        m0 = fmaxf(m0, fmaxf(fabsf(h00), fabsf(h01)));
        m1 = fmaxf(m1, fmaxf(fabsf(h10), fabsf(h11)));
      }
      m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
      m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
      m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
      m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
      const float sc0 = div_full(m0, 448.f), sc1 = div_full(m1, 448.f);
      const float inv0 = sc0 == 0.f ? 0.f : div_full(1.f, sc0);
      const float inv1 = sc1 == 0.f ? 0.f : div_full(1.f, sc1);
      const size_t gr0 = static_cast<size_t>(mb) * BM + r0;
      const size_t gr1 = static_cast<size_t>(mb) * BM + r1;
      if (tig == 0) {
        a2s[gr0] = sc0;
        a2s[gr1] = sc1;
      }
#pragma unroll
      for (int nt = 0; nt < 24; ++nt) {
        const int i = nt * 8 + tig * 2;
        *reinterpret_cast<uint16_t*>(interq + gr0 * IMD + i) =
            pack2_fp8(smH[r0 * IMDP + i] * inv0, smH[r0 * IMDP + i + 1] * inv0);
        *reinterpret_cast<uint16_t*>(interq + gr1 * IMD + i) =
            pack2_fp8(smH[r1 * IMDP + i] * inv1, smH[r1 * IMDP + i + 1] * inv1);
      }
    }
    __syncthreads();                         // epilogue smem reads done before the next block stages
    if (tid == 0) flag_release(&flags[mb]);  // interq/a2s for mb published
  }                                          // mb loop
}

// ---------------------------------------------------------------------------
// GEMM1, single-warpgroup variant: one 128-thread warpgroup owns the whole
// [64,384] tile (accG/accU chains for the gate and up halves), so the silu
// pairing of column j with j+192 is thread-local (same fragment slot in both
// accumulators) and the smem epilogue handoff disappears. A 2-stage ring
// (114688 B) fits TWO CTAs per SM: two independent barrier domains cover
// each other's mbarrier/TMA and wgmma-drain bubbles, which a single fat CTA
// cannot (its barrier convergence stalls were the measured limiter).
__global__ __launch_bounds__(128, 2) void k_gemm1_w1(
    const uint8_t* __restrict__ Aq,
    const float* __restrict__ a1s,
    const __grid_constant__ CUtensorMap w1map,
    const float* __restrict__ w1s,
    const int* __restrict__ sorted,
    const int* __restrict__ ebk,
    const int* __restrict__ nbt,
    uint8_t* __restrict__ interq,
    float* __restrict__ a2s,
    int* __restrict__ flags) {
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* smA = smem;                 // [2][64][128]
  uint8_t* smB = smem + 2 * BM * BK1;  // [2][384][128]

  __shared__ int sPair[BM];
  __shared__ int sTok[BM];
  __shared__ float sA1s[BM];
  __shared__ __align__(8) uint64_t mFull[2];

  const int tid = threadIdx.x;
  if (tid < 2) mbar_init(&mFull[tid], 1);
  int srun = 0;  // global stage counter: mbarrier buf/phase continue across mb

  const int nblocks = *nbt;
  gdc_launch_dependents();  // let the GEMM2 grid start gap-filling freed SMs
  for (int mb = blockIdx.x; mb < nblocks; mb += gridDim.x) {
    const int e = ebk[mb];
    if (tid < BM) {
      const int p = sorted[mb * BM + tid];
      sPair[tid] = p;
      const int tok = p >= 0 ? p / TOPK : 0;
      sTok[tid] = tok;
      sA1s[tid] = p >= 0 ? a1s[tok] : 0.f;
    }
    __syncthreads();

    const int warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, tig = lane & 3;

    float accG[96], accU[96];
#pragma unroll
    for (int i = 0; i < 96; ++i)
      accG[i] = accU[i] = 0.f;

    auto issue = [&](int kstep, int buf) {
      const int k0 = kstep * BK1;
      // A: 64 rows x 8 chunks of 16B = 512 -> 4 per thread (gathered rows).
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int lin = i * 128 + tid;
        const int r = lin >> 3, c = lin & 7;
        const uint32_t dst = smem_u32(smA + buf * BM * BK1 + sw128_off(r, c));
        const uint8_t* src = Aq + static_cast<size_t>(sTok[r]) * HID + k0 + c * 16;
        cp_async16(dst, src, sPair[r] >= 0);
      }
      cp_async_commit();
      if (tid == 0) {
        mbar_arrive_expect_tx(&mFull[buf], N1 * BK1);
        const uint32_t d = smem_u32(smB + buf * N1 * BK1);
        tma_load_2d(&w1map, d, k0, e * N1, &mFull[buf]);
        tma_load_2d(&w1map, d + 192 * BK1, k0, e * N1 + 192, &mFull[buf]);
      }
    };

    constexpr int KSTEPS = HID / BK1;  // 40
    const int sbase = srun;
    srun += KSTEPS;
    issue(0, sbase & 1);
    issue(1, (sbase + 1) & 1);
    for (int it = 0; it < KSTEPS; ++it) {
      const int s = sbase + it;
      if (it + 1 < KSTEPS) {
        cp_async_wait<1>();
      } else {
        cp_async_wait<0>();
      }
      mbar_wait(&mFull[s & 1], (s >> 1) & 1);
      __syncthreads();
      fence_proxy_async();
      const uint32_t a0 = smem_u32(smA + (s & 1) * BM * BK1);
      const uint32_t b0 = smem_u32(smB + (s & 1) * N1 * BK1);
      wgmma_fence();
#pragma unroll
      for (int kc = 0; kc < 4; ++kc) {
        const int sd = (it == 0 && kc == 0) ? 0 : 1;
        wgmma_n192(accG, make_desc_sw128(a0 + kc * 32), make_desc_sw128(b0 + kc * 32), sd);
        wgmma_n192(accU, make_desc_sw128(a0 + kc * 32), make_desc_sw128(b0 + IMD * BK1 + kc * 32), sd);
      }
      wgmma_commit();
      // 2-stage ring: the aligned wait<0> retires this stage's collective
      // chains for the whole (single) warpgroup, so every thread may refill
      // this buffer without a barrier; the re-armed mbarrier cannot be seen by
      // a stale spinner because the top-of-iteration syncthreads follows the
      // mbar_wait. Bubble at the chain tail is covered by the co-resident CTA.
      wgmma_wait<0>();
      if (it + 2 < KSTEPS) issue(it + 2, s & 1);
    }

    // Epilogue: dequant, silu(gate)*up, per-row amax, fp8 requant -- all
    // thread-local (gate col c and up col c share a fragment slot). Numerics
    // match k_gemm1 exactly: bf16 round of each GEMM output, fast-math silu
    // (__expf + div.full), bf16 round of silu*up, div.full quant scales.
    const int r0 = warp * 16 + gid, r1 = r0 + 8;
    const float a10 = sA1s[r0], a11 = sA1s[r1];
    float m0 = 0.f, m1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < 24; ++nt) {
      const int n = nt * 8 + tig * 2;
      const float s0 = w1s[static_cast<size_t>(e) * N1 + n];
      const float s1 = w1s[static_cast<size_t>(e) * N1 + n + 1];
      const float su0 = w1s[static_cast<size_t>(e) * N1 + IMD + n];
      const float su1 = w1s[static_cast<size_t>(e) * N1 + IMD + n + 1];
      const float g00 = bf16r(accG[4 * nt + 0] * (a10 * s0));
      const float g01 = bf16r(accG[4 * nt + 1] * (a10 * s1));
      const float g10 = bf16r(accG[4 * nt + 2] * (a11 * s0));
      const float g11 = bf16r(accG[4 * nt + 3] * (a11 * s1));
      const float u00 = bf16r(accU[4 * nt + 0] * (a10 * su0));
      const float u01 = bf16r(accU[4 * nt + 1] * (a10 * su1));
      const float u10 = bf16r(accU[4 * nt + 2] * (a11 * su0));
      const float u11 = bf16r(accU[4 * nt + 3] * (a11 * su1));
      const float h00 = bf16r(div_full(g00, 1.f + __expf(-g00)) * u00);
      const float h01 = bf16r(div_full(g01, 1.f + __expf(-g01)) * u01);
      const float h10 = bf16r(div_full(g10, 1.f + __expf(-g10)) * u10);
      const float h11 = bf16r(div_full(g11, 1.f + __expf(-g11)) * u11);
      accU[4 * nt + 0] = h00;
      accU[4 * nt + 1] = h01;
      accU[4 * nt + 2] = h10;
      accU[4 * nt + 3] = h11;
      m0 = fmaxf(m0, fmaxf(fabsf(h00), fabsf(h01)));
      m1 = fmaxf(m1, fmaxf(fabsf(h10), fabsf(h11)));
    }
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    const float sc0 = div_full(m0, 448.f), sc1 = div_full(m1, 448.f);
    const float inv0 = sc0 == 0.f ? 0.f : div_full(1.f, sc0);
    const float inv1 = sc1 == 0.f ? 0.f : div_full(1.f, sc1);
    const size_t gr0 = static_cast<size_t>(mb) * BM + r0;
    const size_t gr1 = static_cast<size_t>(mb) * BM + r1;
    if (tig == 0) {
      a2s[gr0] = sc0;
      a2s[gr1] = sc1;
    }
#pragma unroll
    for (int nt = 0; nt < 24; ++nt) {
      const int n = nt * 8 + tig * 2;
      *reinterpret_cast<uint16_t*>(interq + gr0 * IMD + n) =
          pack2_fp8(accU[4 * nt + 0] * inv0, accU[4 * nt + 1] * inv0);
      *reinterpret_cast<uint16_t*>(interq + gr1 * IMD + n) =
          pack2_fp8(accU[4 * nt + 2] * inv1, accU[4 * nt + 3] * inv1);
    }
    __syncthreads();                         // epilogue sA1s reads done before the next block stages
    if (tid == 0) flag_release(&flags[mb]);  // interq/a2s for mb published
  }                                          // mb loop
}

// ---------------------------------------------------------------------------
// Tiny-M GEMM1 (M <= 16): swap_ab orientation. The expert's w1 slab is the
// wgmma A/m-side operand (6 m64 tiles over the 384 gate+up cols, 3 per
// warpgroup) and the <=16 gathered token rows are the B/n-side (one n16
// tile), so a cnt<=16 block costs ~1/24 the tensor-core work of the padded
// [64x384] tile. The per-output-element ascending-k32 whole-K in-TC chain is
// orientation- and N-tiling-invariant (swapped-n64 rig == real triton ic1,
// bitwise). Output fragments are transposed: [w1 col j][token c].
template <bool kDeep>
__global__ __launch_bounds__(256, 1) void k_gemm1_tiny(
    const uint8_t* __restrict__ Aq,
    const float* __restrict__ a1s,
    const __grid_constant__ CUtensorMap w1map,
    const float* __restrict__ w1s,
    const int* __restrict__ sorted,
    const int* __restrict__ ebk,
    const int* __restrict__ nbt,
    uint8_t* __restrict__ interq,
    float* __restrict__ a2s,
    int* __restrict__ flags) {
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* smT = smem;                                          // [4][16][128] token tiles
  uint8_t* smB = smem + K1_STAGES * BM * BK1;                   // [4][384][128] w1 tiles
  constexpr int IMDP = IMD + 4;                                 // bank-rotating row stride
  float* smG = reinterpret_cast<float*>(smem);                  // [16][IMDP] silu(gate) fp32
  float* smH = reinterpret_cast<float*>(smem + 16 * IMDP * 4);  // [16][IMDP]

  __shared__ int sPair[16];
  __shared__ int sTok[16];
  __shared__ float sA1s[16];
  __shared__ __align__(8) uint64_t mFull[K1_STAGES];

  const int tid = threadIdx.x;
  if (tid < K1_STAGES) mbar_init(&mFull[tid], 1);
  int srun = 0;  // global stage counter: mbarrier buf/phase continue across mb

  const int nblocks = *nbt;
  gdc_launch_dependents();
  for (int mb = blockIdx.x; mb < nblocks; mb += gridDim.x) {
    const int e = ebk[mb];
    if (tid < 16) {
      const int p = sorted[mb * BM + tid];
      sPair[tid] = p;
      const int tok = p >= 0 ? p / TOPK : 0;
      sTok[tid] = tok;
      sA1s[tid] = p >= 0 ? a1s[tok] : 0.f;
    }
    __syncthreads();

    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp & 3, wn = warp >> 2;  // wn = warpgroup: 0 gate, 1 up
    const int gid = lane >> 2, tig = lane & 3;

    float acc[3][8];
#pragma unroll
    for (int t = 0; t < 3; ++t)
#pragma unroll
      for (int i = 0; i < 8; ++i)
        acc[t][i] = 0.f;

    auto issue = [&](int kstep, int buf) {
      const int k0 = kstep * BK1;
      // Tokens: 16 rows x 8 chunks of 16B = 128 -> tid<128, one each.
      if (tid < 128) {
        const int r = tid >> 3, c = tid & 7;
        const uint32_t dst = smem_u32(smT + buf * BM * BK1 + sw128_off(r, c));
        const uint8_t* src = Aq + static_cast<size_t>(sTok[r]) * HID + k0 + c * 16;
        cp_async16(dst, src, sPair[r] >= 0);
      }
      cp_async_commit();
      if (tid == 0) {
        mbar_arrive_expect_tx(&mFull[buf], N1 * BK1);
        const uint32_t d = smem_u32(smB + buf * N1 * BK1);
        tma_load_2d(&w1map, d, k0, e * N1, &mFull[buf]);
        tma_load_2d(&w1map, d + 192 * BK1, k0, e * N1 + 192, &mFull[buf]);
      }
    };

    // kDeep (M<=4): 3-ahead issue + per-stage full drain -- with almost no TC
    // work per stage the 1-ahead schedule exposes a TMA round trip per stage
    // (~1us x 40 at M=1); the pre-issue barrier certifies every warp retired
    // stage it-1, whose buffer (s+3)%4 the new issue overwrites. At M in
    // (4,16] the machine is BW-fed by ~120 concurrent blocks and the per-stage
    // drain costs more than it hides: keep the 1-ahead wait<1> schedule.
    constexpr int KSTEPS = HID / BK1;  // 40
    const int sbase = srun;
    srun += KSTEPS;
    issue(0, sbase % K1_STAGES);
    if (kDeep) {
      issue(1, (sbase + 1) % K1_STAGES);
      issue(2, (sbase + 2) % K1_STAGES);
    }
    for (int it = 0; it < KSTEPS; ++it) {
      const int s = sbase + it;
      if (kDeep) {
        cp_async_wait<2>();  // token group of stage it retired (in-order)
        mbar_wait(&mFull[s % K1_STAGES], (s / K1_STAGES) & 1);
        __syncthreads();
        if (it + 3 < KSTEPS) issue(it + 3, (s + 3) % K1_STAGES);
      } else {
        if (it + 1 < KSTEPS) {
          issue(it + 1, (s + 1) % K1_STAGES);
          cp_async_wait<1>();
        } else {
          cp_async_wait<0>();
        }
        mbar_wait(&mFull[s % K1_STAGES], (s / K1_STAGES) & 1);
        __syncthreads();
      }
      fence_proxy_async();
      const uint32_t t0 = smem_u32(smT + (s % K1_STAGES) * BM * BK1);
      const uint32_t b0 = smem_u32(smB + (s % K1_STAGES) * N1 * BK1 + wn * 192 * BK1);
      wgmma_fence();
#pragma unroll
      for (int t = 0; t < 3; ++t)
#pragma unroll
        for (int kc = 0; kc < 4; ++kc)
          wgmma_n16(
              acc[t],
              make_desc_sw128(b0 + t * 64 * BK1 + kc * 32),
              make_desc_sw128(t0 + kc * 32),
              (it == 0 && kc == 0) ? 0 : 1);
      wgmma_commit();
      if (kDeep) {
        wgmma_wait<0>();
      } else if (it + 1 < KSTEPS) {
        wgmma_wait<1>();
      } else {
        wgmma_wait<0>();
      }
    }
    __syncthreads();

    // Transposed epilogue. Fragment (warp wm, lane): weight-col rows
    // r0=wm*16+gid, r1=r0+8 within m64 tile t (global gate col j=t*64+r), token
    // cols c=i*8+tig*2 and c+1 for the two n8 groups i. Reference numerics:
    // gate/up bf16-rounded before use with acc*(a1s*w1s) grouping, silu kept
    // fp32 via div_full+__expf, one bf16 round on silu*up.
    const int r0 = wm * 16 + gid, r1 = r0 + 8;
    if (wn == 0) {
#pragma unroll
      for (int t = 0; t < 3; ++t) {
        const int j0 = t * 64 + r0, j1 = t * 64 + r1;
        const float ws0 = w1s[static_cast<size_t>(e) * N1 + j0];
        const float ws1 = w1s[static_cast<size_t>(e) * N1 + j1];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int c = i * 8 + tig * 2;
          const float sa0 = sA1s[c], sa1 = sA1s[c + 1];
          const float g00 = bf16r(acc[t][4 * i + 0] * (sa0 * ws0));
          const float g01 = bf16r(acc[t][4 * i + 1] * (sa1 * ws0));
          const float g10 = bf16r(acc[t][4 * i + 2] * (sa0 * ws1));
          const float g11 = bf16r(acc[t][4 * i + 3] * (sa1 * ws1));
          smG[c * IMDP + j0] = div_full(g00, 1.f + __expf(-g00));
          smG[(c + 1) * IMDP + j0] = div_full(g01, 1.f + __expf(-g01));
          smG[c * IMDP + j1] = div_full(g10, 1.f + __expf(-g10));
          smG[(c + 1) * IMDP + j1] = div_full(g11, 1.f + __expf(-g11));
        }
      }
    }
    __syncthreads();
    if (wn == 1) {
#pragma unroll
      for (int t = 0; t < 3; ++t) {
        const int j0 = t * 64 + r0, j1 = t * 64 + r1;
        const float ws0 = w1s[static_cast<size_t>(e) * N1 + IMD + j0];
        const float ws1 = w1s[static_cast<size_t>(e) * N1 + IMD + j1];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int c = i * 8 + tig * 2;
          const float sa0 = sA1s[c], sa1 = sA1s[c + 1];
          const float u00 = bf16r(acc[t][4 * i + 0] * (sa0 * ws0));
          const float u01 = bf16r(acc[t][4 * i + 1] * (sa1 * ws0));
          const float u10 = bf16r(acc[t][4 * i + 2] * (sa0 * ws1));
          const float u11 = bf16r(acc[t][4 * i + 3] * (sa1 * ws1));
          smH[c * IMDP + j0] = bf16r(smG[c * IMDP + j0] * u00);
          smH[(c + 1) * IMDP + j0] = bf16r(smG[(c + 1) * IMDP + j0] * u01);
          smH[c * IMDP + j1] = bf16r(smG[c * IMDP + j1] * u10);
          smH[(c + 1) * IMDP + j1] = bf16r(smG[(c + 1) * IMDP + j1] * u11);
        }
      }
    }
    __syncthreads();

    // Per-token amax + fp8 requant: warp w owns token cols w and w+8.
    for (int cc = warp; cc < 16; cc += 8) {
      const int p = sPair[cc];
      float mx = 0.f;
#pragma unroll
      for (int i = 0; i < 6; ++i)
        mx = fmaxf(mx, fabsf(smH[cc * IMDP + lane + 32 * i]));
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
      const size_t gr = static_cast<size_t>(mb) * BM + cc;
      if (p >= 0) {
        const float sc = div_full(mx, 448.f);
        const float inv = sc == 0.f ? 0.f : div_full(1.f, sc);
        if (lane == 0) a2s[gr] = sc;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
          const int jj = 2 * (lane + 32 * i);
          *reinterpret_cast<uint16_t*>(interq + gr * IMD + jj) =
              pack2_fp8(smH[cc * IMDP + jj] * inv, smH[cc * IMDP + jj + 1] * inv);
        }
      } else if (lane == 0) {
        a2s[gr] = 0.f;  // matches k_gemm1's zero output for padded rows
      }
    }
    __syncthreads();                         // epilogue smem reads done before the next block stages
    if (tid == 0) flag_release(&flags[mb]);  // interq/a2s for mb published
  }                                          // mb loop
}

// ---------------------------------------------------------------------------
// GEMM2: C[64,256] per CTA = interq[rows,192] x w2[e][cols,192]^T via wgmma
// (6 ascending k32 chunks, whole-K in-TC chain), scaled + routed weight.
// Two epilogues: at large M the C tile is staged in smem so C3 rows are
// written as contiguous 512B runs (the direct fragment scatter degenerates
// to 4B stores and C3 volume dominates); at small M the direct scatter wins
// because the staging barrier + smem round-trip is pure per-CTA overhead.
// K=192 = stage0 128B (4 chunks) + stage1 64B (2 chunks); 2 CTAs/SM.
constexpr int K2_SMEM = 2 * BM * 128 + 2 * 256 * 128;  // 81920 B

DEV void tok_count_release(int* c) {
  asm volatile("red.release.gpu.global.add.s32 [%0], 1;" ::"l"(c) : "memory");
}

template <bool kSmemEpi, bool kTok>
DEV void gemm2_body(
    const uint8_t* __restrict__ interq,
    const float* __restrict__ a2s,
    const uint8_t* __restrict__ w2,
    const float* __restrict__ w2s,
    const int* __restrict__ sorted,
    const int* __restrict__ ebk,
    const float* __restrict__ tkw,
    __nv_bfloat16* __restrict__ C3,
    int* __restrict__ tokcnt,
    int mb,
    int nb) {
  const int e = ebk[mb];

  extern __shared__ __align__(1024) uint8_t smem2[];
  uint8_t* smA = smem2;                 // [2 k-stages][64][128]
  uint8_t* smB = smem2 + 2 * BM * 128;  // [2 k-stages][256][128]
  // C tile staging reuses the (consumed) B region after the wgmma chain.
  __nv_bfloat162* smC = reinterpret_cast<__nv_bfloat162*>(smB);  // [64][128]

  __shared__ int sPair[BM];
  __shared__ float sW[BM];
  __shared__ float sS[BM];

  const int tid = threadIdx.x;
  if (tid < BM) {
    const int p = sorted[mb * BM + tid];
    sPair[tid] = p;
    sW[tid] = p >= 0 ? tkw[p] : 0.f;
    sS[tid] = a2s[mb * BM + tid];
  }
  // A: 64 rows x 12 chunks (192B) = 768 -> 3 per thread.
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int lin = j * 256 + tid;
    const int r = lin / 12, c = lin % 12;
    const int st = c >> 3, cc = c & 7;
    const uint32_t dst = smem_u32(smA + st * BM * 128 + sw128_off(r, cc));
    const uint8_t* src = interq + (static_cast<size_t>(mb) * BM + r) * IMD + c * 16;
    cp_async16(dst, src, true);
  }
  // B: 256 rows x 12 chunks = 3072 -> 12 per thread.
  const uint8_t* w2e = w2 + (static_cast<size_t>(e) * N2 + nb * 256) * IMD;
#pragma unroll
  for (int j = 0; j < 12; ++j) {
    const int lin = j * 256 + tid;
    const int r = lin / 12, c = lin % 12;
    const int st = c >> 3, cc = c & 7;
    const uint32_t dst = smem_u32(smB + st * 256 * 128 + sw128_off(r, cc));
    cp_async16(dst, w2e + static_cast<size_t>(r) * IMD + c * 16, true);
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();
  fence_proxy_async();

  const int warp = tid >> 5, lane = tid & 31;
  const int wm = warp & 3, wn = warp >> 2;  // wn = warpgroup: cols wn*128..+128
  const int gid = lane >> 2, tig = lane & 3;

  float acc[64];
#pragma unroll
  for (int i = 0; i < 64; ++i)
    acc[i] = 0.f;

  wgmma_fence();
#pragma unroll
  for (int st = 0; st < 2; ++st) {
    const uint32_t a0 = smem_u32(smA + st * BM * 128);
    const uint32_t b0 = smem_u32(smB + st * 256 * 128 + wn * 128 * 128);
    const int nkc = st == 0 ? 4 : 2;
#pragma unroll
    for (int kc = 0; kc < nkc; ++kc)
      wgmma_n128(acc, make_desc_sw128(a0 + kc * 32), make_desc_sw128(b0 + kc * 32), (st == 0 && kc == 0) ? 0 : 1);
  }
  wgmma_commit();
  wgmma_wait<0>();

  // Reference order: acc *= a2_scale * w2_scale, then *= routing weight,
  // then a single bf16 round.
  const int r0 = wm * 16 + gid, r1 = r0 + 8;
  const int p0 = sPair[r0], p1 = sPair[r1];
  const float q0 = sS[r0], q1 = sS[r1];
  const float wt0 = sW[r0], wt1 = sW[r1];
  if constexpr (kSmemEpi) {
    __syncthreads();  // B fully consumed by both warpgroups; smC may reuse it
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const int c = wn * 128 + nt * 8 + tig * 2;
      const int gcol = nb * 256 + c;
      const float s0 = w2s[static_cast<size_t>(e) * N2 + gcol];
      const float s1 = w2s[static_cast<size_t>(e) * N2 + gcol + 1];
      __nv_bfloat162 o0, o1;
      o0.x = __float2bfloat16_rn((acc[4 * nt + 0] * (q0 * s0)) * wt0);
      o0.y = __float2bfloat16_rn((acc[4 * nt + 1] * (q0 * s1)) * wt0);
      o1.x = __float2bfloat16_rn((acc[4 * nt + 2] * (q1 * s0)) * wt1);
      o1.y = __float2bfloat16_rn((acc[4 * nt + 3] * (q1 * s1)) * wt1);
      smC[r0 * 128 + (c >> 1)] = o0;
      smC[r1 * 128 + (c >> 1)] = o1;
    }
    __syncthreads();

    // 4 threads per row stream 512B rows out as 16B chunks (64B/row/instr).
    const int r = tid >> 2, jq = tid & 3;
    const int p = sPair[r];
    if (p >= 0) {
      const uint4* srcv = reinterpret_cast<const uint4*>(smC + r * 128);
      uint4* dstv = reinterpret_cast<uint4*>(C3 + static_cast<size_t>(p) * N2 + nb * 256);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        dstv[i * 4 + jq] = srcv[i * 4 + jq];
    }
  } else {
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const int c = wn * 128 + nt * 8 + tig * 2;
      const int gcol = nb * 256 + c;
      const float s0 = w2s[static_cast<size_t>(e) * N2 + gcol];
      const float s1 = w2s[static_cast<size_t>(e) * N2 + gcol + 1];
      if (p0 >= 0) {
        __nv_bfloat162 o;
        o.x = __float2bfloat16_rn((acc[4 * nt + 0] * (q0 * s0)) * wt0);
        o.y = __float2bfloat16_rn((acc[4 * nt + 1] * (q0 * s1)) * wt0);
        *reinterpret_cast<__nv_bfloat162*>(C3 + static_cast<size_t>(p0) * N2 + gcol) = o;
      }
      if (p1 >= 0) {
        __nv_bfloat162 o;
        o.x = __float2bfloat16_rn((acc[4 * nt + 2] * (q1 * s0)) * wt1);
        o.y = __float2bfloat16_rn((acc[4 * nt + 3] * (q1 * s1)) * wt1);
        *reinterpret_cast<__nv_bfloat162*>(C3 + static_cast<size_t>(p1) * N2 + gcol) = o;
      }
    }
  }
  // Publish this (row-block, col-slice)'s C3 rows: each real pair row bumps
  // its token's counter; the combine tail waits for TOPK*20 per token.
  if constexpr (kTok) {
    __syncthreads();
    if (tid < BM) {
      const int p = sPair[tid];
      if (p >= 0) tok_count_release(&tokcnt[p / TOPK]);
    }
  }
}

// Token combine, folded into the GEMM2 grid as a tail phase: a CTA that has
// finished its item slice reduces its tokens as soon as their 180 col-slices
// (9 pairs x 20) are published, overlapping the reduction with the GEMM2
// drain. Same-grid release/acquire counters carry the data dependency.
DEV void combine_tokens(
    const __nv_bfloat16* __restrict__ C3, __nv_bfloat16* __restrict__ hidden, const int* __restrict__ tokcnt, int M) {
  const int tid = threadIdx.x;
  for (int t = blockIdx.x; t < M; t += gridDim.x) {
    if (tid == 0) {
      int v;
      do {
        asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(tokcnt + t) : "memory");
      } while (v < TOPK * (N2 / 256));
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      const int vec = tid + i * 256;  // 0..1279, 4 bf16 each
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        const uint2 raw = *(reinterpret_cast<const uint2*>(C3 + (static_cast<size_t>(t) * TOPK + k) * N2) + vec);
        const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&raw);
        acc[0] += __bfloat162float(pb[0]);
        acc[1] += __bfloat162float(pb[1]);
        acc[2] += __bfloat162float(pb[2]);
        acc[3] += __bfloat162float(pb[3]);
      }
      __nv_bfloat16 outv[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        outv[j] = __float2bfloat16_rn(2.5f * acc[j]);
      *(reinterpret_cast<uint2*>(hidden + static_cast<size_t>(t) * N2) + vec) = *reinterpret_cast<uint2*>(outv);
    }
  }
}

// Persistent wrappers: <= 264 resident CTAs sweep the (row-block, col-slice)
// item list, then flow into the token-combine tail.
__global__ __launch_bounds__(256, 2) void k_gemm2_direct(
    const uint8_t* __restrict__ interq,
    const float* __restrict__ a2s,
    const uint8_t* __restrict__ w2,
    const float* __restrict__ w2s,
    const int* __restrict__ sorted,
    const int* __restrict__ ebk,
    const int* __restrict__ nbt,
    const float* __restrict__ tkw,
    __nv_bfloat16* __restrict__ C3,
    const int* __restrict__ flags,
    int* __restrict__ tokcnt,
    __nv_bfloat16* __restrict__ hidden,
    int M) {
  const int nblocks = *nbt;
  const int items = nblocks * (N2 / 256);
  int lastmb = -1;
  for (int i = blockIdx.x; i < items; i += gridDim.x) {
    const int mb = i % nblocks;  // mb-fast rasterization
    const int nb = i / nblocks;
    if (mb != lastmb) {
      if (threadIdx.x == 0) flag_acquire_spin(&flags[mb]);
      lastmb = mb;
    }
    __syncthreads();
    gemm2_body<false, true>(interq, a2s, w2, w2s, sorted, ebk, tkw, C3, tokcnt, mb, nb);
    __syncthreads();
  }
  combine_tokens(C3, hidden, tokcnt, M);
}

// Flat per-CTA direct variant (one item per CTA, separate combine): best for
// mid-size M where the persistent loop + counter overhead outweighs the
// launch savings of the folded-combine path.
__global__ __launch_bounds__(256, 2) void k_gemm2_direct_flat(
    const uint8_t* __restrict__ interq,
    const float* __restrict__ a2s,
    const uint8_t* __restrict__ w2,
    const float* __restrict__ w2s,
    const int* __restrict__ sorted,
    const int* __restrict__ ebk,
    const int* __restrict__ nbt,
    const float* __restrict__ tkw,
    __nv_bfloat16* __restrict__ C3,
    const int* __restrict__ flags,
    int* __restrict__ tokcnt,
    __nv_bfloat16* __restrict__ hidden,
    int M) {
  const int mb = blockIdx.x;
  if (mb >= *nbt) return;
  if (threadIdx.x == 0) flag_acquire_spin(&flags[mb]);
  __syncthreads();
  gemm2_body<false, false>(interq, a2s, w2, w2s, sorted, ebk, tkw, C3, tokcnt, mb, blockIdx.y);
}

// Pipelined persistent variant: <=264 resident CTAs (2/SM) sweep the item
// list in the same nb-fast order the non-persistent grid rasterized. The C
// tile stages through a DEDICATED smC buffer (not aliasing smB), so item
// i+1's 60KB of cp.async loads issue right after wgmma_wait<0> and overlap
// the whole epilogue of item i. smem = 16K A + 64K B + 32K C = 112KB still
// fits 2 CTA/SM (2 x (114688 + ~1.8K static/driver) <= 233472).
constexpr int K2P_SMEM = 2 * BM * 128 + 2 * 256 * 128 + BM * 256 * 2;

__global__ __launch_bounds__(256, 2) void k_gemm2_staged(
    const uint8_t* __restrict__ interq,
    const float* __restrict__ a2s,
    const uint8_t* __restrict__ w2,
    const float* __restrict__ w2s,
    const __grid_constant__ CUtensorMap w2map,
    const __grid_constant__ CUtensorMap w2mapT,
    const __grid_constant__ CUtensorMap iqmap,
    const __grid_constant__ CUtensorMap iqmapT,
    const int* __restrict__ sorted,
    const int* __restrict__ ebk,
    int* __restrict__ nbt,
    const float* __restrict__ tkw,
    __nv_bfloat16* __restrict__ C3,
    const int* __restrict__ flags,
    int* __restrict__ tokcnt,
    __nv_bfloat16* __restrict__ hidden,
    int M) {
  const int nblocks = *nbt;
  const int items = nblocks * (N2 / 256);
  int* const ctr = nbt + 1;  // work-stealing item counter (zeroed by k_scan)

  extern __shared__ __align__(1024) uint8_t smem2[];
  uint8_t* smA = smem2;                 // [2 k-stages][64][128]
  uint8_t* smB = smem2 + 2 * BM * 128;  // [2 k-stages][256][128]
  // C staging tile [64 rows][512B], 16B chunks XOR-swizzled by row: without
  // the swizzle both the fill (8 lanes sharing a column-pair across rows)
  // and the readback (8 rows sharing a chunk index) are 8-way bank conflicts
  // because the 512B row stride is bank-aligned.
  uint8_t* smCb = smB + 2 * 256 * 128;
  auto smc_off = [](int r, int chunk) -> uint32_t { return r * 512 + ((chunk ^ (r & 31)) << 4); };

  __shared__ int sPair[BM];
  __shared__ float sW[BM];
  __shared__ float sS[BM];
  __shared__ __align__(8) uint64_t mFull;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int wm = warp & 3, wn = warp >> 2;
  const int gid = lane >> 2, tig = lane & 3;
  const int r0 = wm * 16 + gid, r1 = r0 + 8;

  if (tid == 0) mbar_init(&mFull, 1);

  // B stage-0 (256 rows x 128B, the L1TEX-heavy 2/3): one TMA load; the
  // tensor map's 128B swizzle writes the same SW128 layout cp.async did.
  // tid0-only, so it can issue early (right after the post-wgmma barrier)
  // and hide its latency under the whole smC fill + C3 store.
  auto issue_tma = [&](int i) {
    const int mb = i / (N2 / 256), nb = i % (N2 / 256);
    const int e = ebk[mb];
    mbar_arrive_expect_tx(&mFull, (256 + BM) * (128 + 64));
    tma_load_2d(&w2map, smem_u32(smB), 0, e * N2 + nb * 256, &mFull);
    tma_load_2d(&w2mapT, smem_u32(smB + 256 * 128), 128, e * N2 + nb * 256, &mFull);
    tma_load_2d(&iqmap, smem_u32(smA), 0, mb * BM, &mFull);
    tma_load_2d(&iqmapT, smem_u32(smA + BM * 128), 128, mb * BM, &mFull);
  };

  // w2s scale stash: the epilogue's 32 per-thread w2s loads were ~20% of
  // warp time as an L2-latency chain feeding the smC fill. Prefetch the
  // item's 256 scales (1KB) with the routing metadata, double-buffered in
  // the DEAD pad chunks of the A stage-1 sub-atom TMA region: its 64B boxes
  // scatter data to chunk positions (c ^ (r&7)) for c<4, leaving positions
  // (r&7)^(4+s), s<4, of every 128B row untouched. Buffer b uses rows
  // [b*16, b*16+16); chunk j of the 64 lives at row j>>2, slot j&3.
  uint8_t* const smA1 = smA + BM * 128;
  // Byte offset of 16B chunk g within a stash buffer (b*16 rows == 0 mod 8,
  // so the row-XOR term only sees g's bits; buffer b adds a flat b*2048).
  auto w2s_slot = [](int g) -> uint32_t {
    const uint32_t r = static_cast<uint32_t>(g) >> 2;
    return r * 128 + (((r & 7) ^ (4 + (g & 3))) << 4);
  };
  auto issue_rest = [&](int i, int buf) {
    const int mb = i / (N2 / 256), nb = i % (N2 / 256);
    if (tid < BM) {
      const int p = sorted[mb * BM + tid];
      sPair[tid] = p;
      sW[tid] = p >= 0 ? tkw[p] : 0.f;
      sS[tid] = a2s[mb * BM + tid];
      const int e = ebk[mb];
      cp_async16(
          smem_u32(smA1 + buf * 2048 + w2s_slot(tid)),
          reinterpret_cast<const uint8_t*>(w2s + static_cast<size_t>(e) * N2 + nb * 256 + tid * 4),
          true);
    }
    cp_async_commit();
  };

  __shared__ int sNext;
  int spunmb = -1;  // tid0-local flag-spin tracker
  if (tid == 0) {
    const int i0 = atomicAdd(ctr, 1);
    sNext = i0;
    if (i0 < items) {
      const int mb0 = i0 / (N2 / 256);
      flag_acquire_spin(&flags[mb0]);
      spunmb = mb0;
    }
  }
  __syncthreads();
  int i = sNext;
  uint32_t phase = 0;
  int seq = 0;  // local item sequence: selects the w2s stash buffer
  if (i < items) {
    if (tid == 0) issue_tma(i);
    issue_rest(i, 0);
  }
  while (i < items) {
    const int nb = i % (N2 / 256);

    mbar_wait(&mFull, phase);
    phase ^= 1;
    cp_async_wait<0>();  // this item's w2s stash chunks landed
    __syncthreads();
    fence_proxy_async();

    float acc[64];
#pragma unroll
    for (int j = 0; j < 64; ++j)
      acc[j] = 0.f;

    wgmma_fence();
#pragma unroll
    for (int st = 0; st < 2; ++st) {
      const uint32_t a0 = smem_u32(smA + st * BM * 128);
      const uint32_t b0 = smem_u32(smB + st * 256 * 128 + wn * 128 * 128);
      const int nkc = st == 0 ? 4 : 2;
#pragma unroll
      for (int kc = 0; kc < nkc; ++kc)
        wgmma_n128(acc, make_desc_sw128(a0 + kc * 32), make_desc_sw128(b0 + kc * 32), (st == 0 && kc == 0) ? 0 : 1);
    }
    wgmma_commit();
    wgmma_wait<0>();
    __syncthreads();  // both warpgroups retired: smA/smB reusable

    // Pop the next item (work-conserving during the gemm1->gemm2 ramp) and
    // issue its TMA immediately: smB st0 is dead past the barrier above and
    // the load hides under the smC fill + C3 store.
    if (tid == 0) {
      const int inx = atomicAdd(ctr, 1);
      sNext = inx;
      if (inx < items) {
        const int mbn = inx / (N2 / 256);
        if (mbn != spunmb) {
          flag_acquire_spin(&flags[mbn]);
          spunmb = mbn;
        }
        issue_tma(inx);
      }
    }
    const int pst = sPair[tid >> 2];  // before issue_rest overwrites sPair

    // Fill smC: consuming acc here frees its 64 registers before the
    // cp.async prefetch issues, keeping the loop spill-free. Reference
    // order: acc *= a2_scale * w2_scale, then *= routing weight, one round.
    const float q0 = sS[r0], q1 = sS[r1];
    const float wt0 = sW[r0], wt1 = sW[r1];
    // Per-thread stash base for this item's buffer; c&3 is constant per
    // thread (tig*2 & 3), folded into the base so the unrolled loop's
    // addresses are base + f(nt) with small immediate parts.
    const uint8_t* const w2sb = smA1 + (seq & 1) * 2048 + ((wn * 128 + tig * 2) & 3) * 4;
    const int g0 = wn * 32 + (tig >> 1);
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const int c = wn * 128 + nt * 8 + tig * 2;
      // Bit-identical w2s values via the stash (pairs are 8B-aligned: c&3
      // is 0 or 2, so both floats sit in the same 16B chunk).
      const float2 sp = *reinterpret_cast<const float2*>(w2sb + w2s_slot(g0 + 2 * nt));
      const float s0 = sp.x, s1 = sp.y;
      __nv_bfloat162 o0, o1;
      o0.x = __float2bfloat16_rn((acc[4 * nt + 0] * (q0 * s0)) * wt0);
      o0.y = __float2bfloat16_rn((acc[4 * nt + 1] * (q0 * s1)) * wt0);
      o1.x = __float2bfloat16_rn((acc[4 * nt + 2] * (q1 * s0)) * wt1);
      o1.y = __float2bfloat16_rn((acc[4 * nt + 3] * (q1 * s1)) * wt1);
      const int ch = (c >> 1) >> 2, sub = ((c >> 1) & 3) << 2;
      *reinterpret_cast<__nv_bfloat162*>(smCb + smc_off(r0, ch) + sub) = o0;
      *reinterpret_cast<__nv_bfloat162*>(smCb + smc_off(r1, ch) + sub) = o1;
    }
    __syncthreads();  // smC filled; sNext published; sPair/sW/sS reads settled
    const int inext = sNext;
    if (inext < items) issue_rest(inext, (seq + 1) & 1);

    // 4 threads per row stream 512B rows out as 16B chunks (64B/row/instr).
    if (pst >= 0) {
      const int r = tid >> 2, jq = tid & 3;
      uint4* dstv = reinterpret_cast<uint4*>(C3 + static_cast<size_t>(pst) * N2 + nb * 256);
#pragma unroll
      for (int q = 0; q < 8; ++q)
        dstv[q * 4 + jq] = *reinterpret_cast<const uint4*>(smCb + smc_off(r, q * 4 + jq));
    }
    i = inext;
    ++seq;
  }
}

// ---------------------------------------------------------------------------
__global__ void k_combine(const __nv_bfloat16* __restrict__ C3, __nv_bfloat16* __restrict__ hidden) {
  const int t = blockIdx.x;
  const int tid = threadIdx.x;  // 256 threads
#pragma unroll
  for (int i = 0; i < 5; ++i) {
    const int vec = tid + i * 256;  // 0..1279, 4 bf16 each
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      const uint2 raw = *(reinterpret_cast<const uint2*>(C3 + (static_cast<size_t>(t) * TOPK + k) * N2) + vec);
      const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&raw);
      acc[0] += __bfloat162float(pb[0]);
      acc[1] += __bfloat162float(pb[1]);
      acc[2] += __bfloat162float(pb[2]);
      acc[3] += __bfloat162float(pb[3]);
    }
    __nv_bfloat16 outv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      outv[j] = __float2bfloat16_rn(2.5f * acc[j]);
    *(reinterpret_cast<uint2*>(hidden + static_cast<size_t>(t) * N2) + vec) = *reinterpret_cast<uint2*>(outv);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
extern "C" void launch_fused_moe(
    void* hidden,
    const void* w1map,
    const void* w2map,
    const void* iqmap,
    const void* w2,
    const float* tkw,
    const int* tki,
    const float* w1s,
    const float* w2s,
    int M,
    int E,
    int maxBlocks,
    uint8_t* Aq,
    float* a1s,
    int* cnt,
    int* fill,
    int* rowOff,
    int* ebk,
    int* nbt,
    int* sorted,
    float* a2s,
    uint8_t* interq,
    void* C3,
    int* flags,
    int* tokcnt,
    cudaStream_t stream) {
  const int P = M * TOPK;
  const int maxRows = maxBlocks * BM;

  if (M < 512) {
    // k_route is the long pole of the pre-GEMM phase and runs on one SM:
    // launch it FIRST (it signals dependents at entry) with k_quant_hidden
    // as the PDL secondary overlapping on the remaining SMs; the buffers are
    // disjoint and k_gemm1 is a normal launch that still waits for both.
    k_route<<<1, 1024, 0, stream>>>(
        const_cast<const int*>(tki), P, E, maxRows, rowOff, ebk, nbt, sorted, flags, maxBlocks);
    cudaLaunchAttribute qattr[1];
    qattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    qattr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t qcfg = {};
    qcfg.gridDim = dim3(M);
    qcfg.blockDim = dim3(128);
    qcfg.stream = stream;
    qcfg.attrs = qattr;
    qcfg.numAttrs = 1;
    qcfg.gridDim = dim3(M);
    cudaLaunchKernelEx(&qcfg, k_quant_hidden, static_cast<const __nv_bfloat16*>(hidden), Aq, a1s, tokcnt, M);
  } else {
    // Overlap the sort chain (reads only tki) with k_quant_hidden on a side
    // stream. evtHead is recorded at the CURRENT main-stream tail so the side
    // chain (a) orders after every prior consumer of sorted/ebk/nbt/flags and
    // (b) does not start under the harness's cold-L2 flush, which would
    // inflate the measured kernel-timeline span.
    static cudaStream_t sideStream = nullptr;
    static cudaEvent_t evtHead = nullptr, evtSort = nullptr;
    if (sideStream == nullptr) {
      int loPrio, hiPrio;
      cudaDeviceGetStreamPriorityRange(&loPrio, &hiPrio);
      cudaStreamCreateWithPriority(&sideStream, cudaStreamNonBlocking, hiPrio);
      cudaEventCreateWithFlags(&evtHead, cudaEventDisableTiming);
      cudaEventCreateWithFlags(&evtSort, cudaEventDisableTiming);
    }
    cudaEventRecord(evtHead, stream);
    cudaStreamWaitEvent(sideStream, evtHead, 0);
    cudaMemsetAsync(cnt, 0, sizeof(int) * E * 2, sideStream);  // cnt+fill
    cudaMemsetAsync(sorted, 0xFF, sizeof(int) * maxRows, sideStream);
    const int tb = 256;
    const int cblk = (P + tb - 1) / tb;
    const int count_blocks = cblk < 264 ? cblk : 264;
    k_count<<<count_blocks, tb, 0, sideStream>>>(tki, P, cnt);
    k_scan<<<1, 256, 0, sideStream>>>(cnt, E, rowOff, ebk, nbt, flags, maxBlocks);
    k_scatter<<<cblk, tb, 0, sideStream>>>(tki, P, rowOff, fill, sorted);
    cudaEventRecord(evtSort, sideStream);
    k_quant_hidden<<<M, 128, 0, stream>>>(static_cast<const __nv_bfloat16*>(hidden), Aq, a1s, tokcnt, M);
    cudaStreamWaitEvent(stream, evtSort, 0);
  }

  static bool smem_set = false;
  if (!smem_set) {
    cudaFuncSetAttribute(k_gemm1, cudaFuncAttributeMaxDynamicSharedMemorySize, K1_SMEM);
    cudaFuncSetAttribute(k_gemm1_w1, cudaFuncAttributeMaxDynamicSharedMemorySize, K1W_SMEM);
    cudaFuncSetAttribute(k_gemm1_tiny<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, K1_SMEM);
    cudaFuncSetAttribute(k_gemm1_tiny<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, K1_SMEM);
    cudaFuncSetAttribute(k_gemm2_direct, cudaFuncAttributeMaxDynamicSharedMemorySize, K2_SMEM);
    cudaFuncSetAttribute(k_gemm2_direct_flat, cudaFuncAttributeMaxDynamicSharedMemorySize, K2_SMEM);
    cudaFuncSetAttribute(k_gemm2_staged, cudaFuncAttributeMaxDynamicSharedMemorySize, K2P_SMEM);
    smem_set = true;
  }
  const int g1 = maxBlocks < 132 ? maxBlocks : 132;
  if (M <= 4) {
    k_gemm1_tiny<true><<<g1, 256, K1_SMEM, stream>>>(
        Aq, a1s, *static_cast<const CUtensorMap*>(w1map), w1s, sorted, ebk, nbt, interq, a2s, flags);
  } else if (M <= 16) {
    k_gemm1_tiny<false><<<g1, 256, K1_SMEM, stream>>>(
        Aq, a1s, *static_cast<const CUtensorMap*>(w1map), w1s, sorted, ebk, nbt, interq, a2s, flags);
  } else {
    const int g1w = maxBlocks < 264 ? maxBlocks : 264;
    k_gemm1_w1<<<g1w, 128, K1W_SMEM, stream>>>(
        Aq, a1s, *static_cast<const CUtensorMap*>(w1map), w1s, sorted, ebk, nbt, interq, a2s, flags);
  }
  // GEMM2 launches as a programmatic dependent of GEMM1: its CTAs become
  // schedulable once GEMM1's CTAs signal, land on SMs GEMM1 has vacated
  // (the smem footprints forbid co-residency), and spin on per-block flags.
  {
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = K2_SMEM;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    if (M < 48) {
      const int items = maxBlocks * (N2 / 256);
      cfg.gridDim = dim3(items < 264 ? items : 264);
      cudaLaunchKernelEx(
          &cfg,
          k_gemm2_direct,
          interq,
          a2s,
          static_cast<const uint8_t*>(w2),
          w2s,
          const_cast<const int*>(sorted),
          const_cast<const int*>(ebk),
          const_cast<const int*>(nbt),
          tkw,
          static_cast<__nv_bfloat16*>(C3),
          const_cast<const int*>(flags),
          tokcnt,
          static_cast<__nv_bfloat16*>(hidden),
          M);
    } else if (M < 512) {
      cfg.gridDim = dim3(maxBlocks, N2 / 256);
      cudaLaunchKernelEx(
          &cfg,
          k_gemm2_direct_flat,
          interq,
          a2s,
          static_cast<const uint8_t*>(w2),
          w2s,
          const_cast<const int*>(sorted),
          const_cast<const int*>(ebk),
          const_cast<const int*>(nbt),
          tkw,
          static_cast<__nv_bfloat16*>(C3),
          const_cast<const int*>(flags),
          tokcnt,
          static_cast<__nv_bfloat16*>(hidden),
          M);
      k_combine<<<M, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(C3), static_cast<__nv_bfloat16*>(hidden));
    } else {
      const int items = maxBlocks * (N2 / 256);
      cfg.gridDim = dim3(items < 264 ? items : 264);
      cfg.dynamicSmemBytes = K2P_SMEM;
      cudaLaunchKernelEx(
          &cfg,
          k_gemm2_staged,
          interq,
          a2s,
          static_cast<const uint8_t*>(w2),
          w2s,
          static_cast<const CUtensorMap*>(w2map)[0],
          static_cast<const CUtensorMap*>(w2map)[1],
          static_cast<const CUtensorMap*>(iqmap)[0],
          static_cast<const CUtensorMap*>(iqmap)[1],
          const_cast<const int*>(sorted),
          const_cast<const int*>(ebk),
          nbt,
          tkw,
          static_cast<__nv_bfloat16*>(C3),
          const_cast<const int*>(flags),
          tokcnt,
          static_cast<__nv_bfloat16*>(hidden),
          M);
      k_combine<<<M, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(C3), static_cast<__nv_bfloat16*>(hidden));
    }
  }
}
