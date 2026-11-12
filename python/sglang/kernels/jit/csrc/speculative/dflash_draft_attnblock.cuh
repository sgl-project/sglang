// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SGLang project
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

// DFlash MLA draft attention block (SGLANG_OPT_DFLASH_DRAFT_ATTNBLOCK, default off): the draft block's split-KV extend attention
// (P1 partial + P1 combine) and the absorbed w_vc projection in ONE clustered launch.
//
// Per (request, head): one cluster of C CTAs. Rank 0 is the combiner; ranks 1..C-1 are workers. Worker w takes rail tiles w, w + (C-1), ...
// (64 keys each); the block's own 8 keys go to the first worker with one rail tile fewer (rem = n_rail % (C-1); rem == 0: the last worker).
// Each worker runs an online softmax over its slots (QK and PV on mma.sync with F16 operands: q BF16 -> F16, FP8 K / V -> F16 exactly; FP32
// accumulators; own block in FP32 SIMT from BF16 k / v), projects its UNNORMALIZED partial through w_vc[head] (BF16 mma.sync, FP32 accumulate;
// w_vc staged in shared memory before griddepcontrol.wait: it is a static weight) and sends Y [8 x 128] FP32 plus its row max and row sum
// (4,160 B) to rank 0 with st.async + mbarrier complete_tx. Rank 0 combines the C-1 packets in fixed rank order:
// out = sum_k exp(m_k - M) Y_k / sum_k exp(m_k - M) l_k (denominator 0 -> 1, as the served combine), one BF16 rounding, stored in the
// [rows, heads * 128] layout o_proj reads. The projection is linear, so projecting before the combine is exact in real arithmetic.
//
// Memory ordering:
//  E1 rank 0: mbarrier.init; fence.mbarrier_init.release.cluster; every CTA barrier.cluster.arrive.relaxed before griddepcontrol.wait,
//     barrier.cluster.wait (acquire) before any st.async to rank 0 (the CUTLASS fence_barrier_init + cluster_arrive_relaxed + cluster_wait form).
//  E2 workers: st.async ... mbarrier::complete_tx (release at cluster scope per PTX) into rank 0's slots; rank 0: every thread
//     mbarrier.try_wait.parity.acquire.cluster, then generic ld.shared of the slots (no async-proxy reader).
//  E3 lifetime: only rank 0's shared memory is accessed remotely, and rank 0 exits after every byte arrived.
//  E4 PDL: every read of predecessor output (q, k_ext, v_ext, the KV pool, indptrs, kv_indices) follows griddepcontrol.wait; w_vc is static.
//     launch_dependents follows the wait and has no memory effect; the dependent waits on this grid itself.
//  E5 w_vc: cp.async.bulk (async proxy) completing on the CTA-local mbarrier wbar; every thread waits wbar (parity 0) before reading it.
//  E6 cp.async (CTA-local, generic): cp.async.wait_group + __syncthreads before the generic reads.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace sglang {
namespace dflash_attnblock {

constexpr int kThreads = 256;
constexpr int kWarps = 8;
constexpr int kRows = 8;      // query rows per request (the draft block)
constexpr int kD = 576;       // key width (latent 512 + rope 64)
constexpr int kDV = 512;      // value width (latent)
constexpr int kVO = 128;      // v_head_dim: w_vc output columns per head
constexpr int kTile = 64;     // keys per rail tile
constexpr int kMaxTiles = 4;  // rail tiles per worker (rail <= 17 tiles, C >= 6)
constexpr int kKPitch = 592;  // bytes per FP8 key row in shared memory (576 + 16: conflict-free B fragments)
constexpr int kQPitch = 584;  // F16 elements per q row (576 + 8)
constexpr int kPPitch = 72;   // F16 elements per P row (64 + 8)
constexpr int kWPitch = kDV * 2 + 16;                     // 1,040 B per w_vc row (n): conflict-free 32-bit B-fragment loads
constexpr int kPacket = kRows * kVO * 4 + 2 * kRows * 4;  // 4,160 B: Y [8 x 128] FP32 | m [8] | l [8]
constexpr int kObK = kRows * kD * 2;                      // own-block k rows staged in a tile buffer (9,216 B), then v rows (8,192 B)

// shared memory map (bytes). Rank 0 uses only kOffSlots (aliases kOffW: rank 0 stages no weight) and the mbarrier.
constexpr int kOffW = 0;                                   // w_vc[head] as [128 n][512 k] BF16 rows of kWPitch B (one bulk copy per row)
constexpr int kOffSlots = 0;                               // rank 0: (C - 1) packets
constexpr int kOffK = kOffW + kVO * kWPitch;               // 133,120: two slot buffers [64][592] FP8 (or the own block's BF16 k / v rows)
constexpr int kOffYred = kOffK;                            // workers after the slot loop: [8 warps][8 rows][128] FP32 (32 KB)
constexpr int kOffQ = kOffK + 2 * kTile * kKPitch;         // 208,896: q [8][584] F16
constexpr int kOffP = kOffQ + kRows * kQPitch * 2;         // 218,240: P [8][72] F16
constexpr int kOffRed = kOffP + kRows * kPPitch * 2;       // 219,392: row max / row sum per warp [2][8][8] FP32
constexpr int kOffSob = kOffRed + 2 * kWarps * kRows * 4;  // 219,904: own-block scores [8][8] FP32
constexpr int kOffML = kOffSob + kRows * kRows * 4;        // 220,160: m [8] | l [8] FP32
constexpr int kOffIdx = kOffML + 2 * kRows * 4;            // 220,224: the worker's rail slots [kMaxTiles][64] int64 (-1 = masked key)
constexpr int kOffBar = kOffIdx + kMaxTiles * kTile * 8;   // 221,760: rank 0 packet mbarrier
constexpr int kOffWBar = kOffBar + 8;                      // workers: w_vc bulk-copy mbarrier
constexpr int kSmemBytes = kOffBar + 16;                   // 221,776 (<= 232,448 opt-in)

static_assert(kOffYred + kWarps * kRows * kVO * 4 <= kOffQ, "Yred must fit in the slot buffers");
static_assert(kObK + kRows * kDV * 2 <= kTile * kKPitch, "own block must fit in one slot buffer");
static_assert(kSmemBytes <= 232448, "dynamic shared memory above the opt-in limit");

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void fence_mbarrier_init_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait_cta(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAITC_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra WAITC_%=;\n\t}" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}" ::"r"(bar),
      "r"(parity)
      : "memory");
}
// one 1-D bulk copy global -> this CTA's shared memory, completing bytes on a local mbarrier (async proxy writer)
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src), "r"(bytes),
               "r"(bar)
               : "memory");
}
__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ void st_async_v4(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(addr), "r"(a), "r"(b),
               "r"(c), "r"(d), "r"(bar)
               : "memory");
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0, uint32_t& r1, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
}
__device__ __forceinline__ void mma_f16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// two e4m3 bytes (low byte = first element) -> F16x2 (low half = first element); exact
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint16_t v) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v), __NV_E4M3);
  return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
}
__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

template <int C>
__global__ void __launch_bounds__(kThreads, 1) attnblock_kernel(
    const __nv_bfloat16* q,        // [T, heads, 576] (predecessor output: plain loads after griddepcontrol.wait)
    const __nv_bfloat16* k_ext,    // [T, 576] row pitch k_stride
    const __nv_bfloat16* v_ext,    // [T, 512] row pitch v_stride
    const uint8_t* pool,           // FP8 e4m3 KV rows, row pitch pool_stride bytes (k = 576 B, v = first 512 B)
    const int32_t* qo_indptr,      // [bs + 1]
    const int32_t* kv_indptr,      // [bs + 1]
    const int64_t* kv_indices,     // rail slots
    const __nv_bfloat16* __restrict__ wvc,  // [heads, 128, 512] contiguous (static)
    __nv_bfloat16* out,            // [T, heads * 128] row pitch out_stride
    int64_t q_stride,
    int64_t k_stride,
    int64_t v_stride,
    int64_t pool_stride,
    int64_t out_stride,
    float qk_scale,  // sm_scale * k_scale (rail)
    float sm_scale,  // own block
    int window) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, c = lane & 3;
  const uint32_t rank = cluster_ctarank();
  const int head = blockIdx.y, req = blockIdx.z;
  const uint32_t bar = smem_u32(smem + kOffBar);
  const uint32_t wbar = smem_u32(smem + kOffWBar);

  // ---- pre-wait: static work only ----
  if (rank != 0) {
    // w_vc[head] (static weight) -> shared: 128 bulk copies of one 1 KB row each (16 per warp) on the local mbarrier wbar
    if (tid == 0) {
      mbar_init(wbar, 1);
      fence_mbarrier_init_cluster();
      mbar_arrive_expect_tx(wbar, kVO * kDV * 2);
    }
    __syncthreads();
    const __nv_bfloat16* wsrc = wvc + static_cast<int64_t>(head) * kVO * kDV;
    if (lane < 16) {
      const int n = warp * 16 + lane;
      bulk_g2s(smem_u32(smem + kOffW) + n * kWPitch, wsrc + n * kDV, kDV * 2, wbar);
    }
  } else if (tid == 0) {
    mbar_init(bar, 1);
    fence_mbarrier_init_cluster();
  }
  cluster_arrive_relaxed();
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  const int q_start = qo_indptr[req];
  const int q_len = qo_indptr[req + 1] - q_start;

  if (rank == 0) {
    // ---- phase C: combine the C - 1 packets in fixed rank order ----
    if (tid == 0) mbar_arrive_expect_tx(bar, (C - 1) * kPacket);
    cluster_wait();
    mbar_wait_cluster(bar, 0);
    const float* slots = reinterpret_cast<const float*>(smem + kOffSlots);
    const int r = tid >> 5, n0 = (tid & 31) * 4;
    float M = -INFINITY;
#pragma unroll
    for (int k = 0; k < C - 1; ++k) {
      M = fmaxf(M, slots[k * (kPacket / 4) + kRows * kVO + r]);
    }
    float L = 0.f, acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
#pragma unroll
    for (int k = 0; k < C - 1; ++k) {
      const float* s = slots + k * (kPacket / 4);
      const float wk = __expf(s[kRows * kVO + r] - M);
      L = fmaf(wk, s[kRows * kVO + kRows + r], L);
      const float4 y = *reinterpret_cast<const float4*>(s + r * kVO + n0);
      acc0 = fmaf(wk, y.x, acc0);
      acc1 = fmaf(wk, y.y, acc1);
      acc2 = fmaf(wk, y.z, acc2);
      acc3 = fmaf(wk, y.w, acc3);
    }
    const float inv = 1.f / (L == 0.f ? 1.f : L);
    if (r < q_len) {
      uint2 pk;
      pk.x = pack_bf16(acc0 * inv, acc1 * inv);
      pk.y = pack_bf16(acc2 * inv, acc3 * inv);
      *reinterpret_cast<uint2*>(out + (q_start + r) * out_stride + head * kVO + n0) = pk;
    }
    return;
  }

  // ---- workers: slot schedule ----
  const int w = static_cast<int>(rank) - 1, NW = C - 1;
  const int kv_start = kv_indptr[req];
  const int L = kv_indptr[req + 1] - kv_start;
  const int n_rail = (L + kTile - 1) / kTile;
  const int rem = n_rail % NW;
  const int own_w = rem == 0 ? NW - 1 : rem;  // first worker with one rail tile fewer
  int nr_w = (w < n_rail) ? (n_rail - 1 - w) / NW + 1 : 0;
  if (nr_w > kMaxTiles) __trap();       // rail longer than this build covers: fail loud (host checks first)
  const bool has_own = (w == own_w);
  const int n_my = nr_w + (has_own ? 1 : 0);

  __half* sq = reinterpret_cast<__half*>(smem + kOffQ);
  __half* sP = reinterpret_cast<__half*>(smem + kOffP);
  float* red = reinterpret_cast<float*>(smem + kOffRed);  // [0]: max [8 warps][8], [1]: sum
  float* sob = reinterpret_cast<float*>(smem + kOffSob);
  int64_t* sidx = reinterpret_cast<int64_t*>(smem + kOffIdx);

  // the worker's rail-slot table and q[head] (BF16 -> F16, exact for BF16 values in F16 range) in one pass of independent loads
  for (int i = tid; i < nr_w * kTile; i += kThreads) {
    const int kidx = (w + (i / kTile) * NW) * kTile + (i % kTile);
    sidx[i] = kidx < L ? kv_indices[kv_start + kidx] : int64_t(-1);
  }
  for (int i = tid; i < kRows * (kD / 8); i += kThreads) {
    const int r = i / (kD / 8), ch = i % (kD / 8);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < q_len) v = *reinterpret_cast<const uint4*>(q + (q_start + r) * q_stride + head * kD + ch * 8);
    const uint32_t src[4] = {v.x, v.y, v.z, v.w};
    uint32_t dst[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      __half2 h = __floats2half2_rn(bf16_lo(src[j]), bf16_hi(src[j]));
      dst[j] = *reinterpret_cast<uint32_t*>(&h);
    }
    *reinterpret_cast<uint4*>(sq + r * kQPitch + ch * 8) = make_uint4(dst[0], dst[1], dst[2], dst[3]);
  }
  __syncthreads();

  // slot i < nr_w: rail tile w + i * NW (FP8 rows by kv_indices); slot nr_w: the own block (BF16 k rows | v rows)
  auto issue = [&](int i, int buf) {
    const uint32_t base = smem_u32(smem + kOffK + buf * kTile * kKPitch);
    if (i < nr_w) {
      for (int e = tid; e < kTile * (kD / 16); e += kThreads) {
        const int key = e / (kD / 16), ch = e % (kD / 16);
        const int64_t loc = sidx[i * kTile + key];
        cp_async_16(base + key * kKPitch + ch * 16, pool + (loc < 0 ? 0 : loc) * pool_stride + ch * 16, loc < 0 ? 0u : 16u);
      }
    } else {
      for (int e = tid; e < kRows * (kD / 8); e += kThreads) {  // k rows: 72 chunks of 16 B
        const int j = e / (kD / 8), ch = e % (kD / 8);
        cp_async_16(base + j * (kD * 2) + ch * 16, k_ext + (q_start + (j < q_len ? j : 0)) * k_stride + ch * 8, j < q_len ? 16u : 0u);
      }
      for (int e = tid; e < kRows * (kDV / 8); e += kThreads) {  // v rows: 64 chunks of 16 B
        const int j = e / (kDV / 8), ch = e % (kDV / 8);
        cp_async_16(base + kObK + j * (kDV * 2) + ch * 16, v_ext + (q_start + (j < q_len ? j : 0)) * v_stride + ch * 8, j < q_len ? 16u : 0u);
      }
    }
    cp_async_commit();
  };

  float m_run = -1e20f, l_run = 0.f;
  float acc[8][4];
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
  const bool row_ok = g < q_len;
  const uint32_t sq_u32 = smem_u32(sq), sP_u32 = smem_u32(sP);

  int issued = 0;
  if (n_my > 0) { issue(0, 0); issued = 1; }
  if (n_my > 1) { issue(1, 1); issued = 2; }
  for (int it = 0; it < n_my; ++it) {
    const int buf = it & 1;
    if (issued > it + 1) {
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    const uint8_t* kb = smem + kOffK + buf * kTile * kKPitch;
    if (it < nr_w) {
      // ---- rail tile: QK (warp = keys 8 warp .. +7), four independent accumulator chains summed in fixed order ----
      const int t = w + it * NW;
      const uint8_t* krow = kb + (8 * warp + g) * kKPitch;
      float sc4[4][4];
#pragma unroll
      for (int j = 0; j < 4; ++j) sc4[j][0] = sc4[j][1] = sc4[j][2] = sc4[j][3] = 0.f;
      const uint32_t a_addr = sq_u32 + ((lane & 7) * kQPitch + ((lane >> 3) & 1) * 8) * 2;
#pragma unroll 3
      for (int ks4 = 0; ks4 < kD / 64; ++ks4) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int ks = ks4 * 4 + j;
          uint32_t a0, a2;
          ldmatrix_x2(a0, a2, a_addr + ks * 32);
          const uint32_t b0 = e4m3x2_to_f16x2(*reinterpret_cast<const uint16_t*>(krow + ks * 16 + 2 * c));
          const uint32_t b1 = e4m3x2_to_f16x2(*reinterpret_cast<const uint16_t*>(krow + ks * 16 + 8 + 2 * c));
          mma_f16(sc4[j], a0, a0, a2, a2, b0, b1);
        }
      }
      const int key0 = t * kTile + 8 * warp + 2 * c;
      float v[2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int key = key0 + e;
        bool ok = row_ok && key < L;
        if (window > 0) ok = ok && (L + g) <= (key + window);
        const float sv = ((sc4[0][e] + sc4[1][e]) + sc4[2][e]) + sc4[3][e];
        v[e] = ok ? sv * qk_scale : -INFINITY;
      }
      float mx = fmaxf(v[0], v[1]);
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      if (c == 0) red[warp * kRows + g] = mx;
      __syncthreads();
      float m_t = -INFINITY;
#pragma unroll
      for (int k = 0; k < kWarps; ++k) m_t = fmaxf(m_t, red[k * kRows + g]);
      m_t = (m_t == -INFINITY) ? -1e20f : m_t;
      const float m_new = fmaxf(m_run, m_t);
      const float alpha = __expf(m_run - m_new);
      const float p0 = __expf(v[0] - m_new), p1 = __expf(v[1] - m_new);
      float ps = p0 + p1;
      ps += __shfl_xor_sync(0xffffffffu, ps, 1);
      ps += __shfl_xor_sync(0xffffffffu, ps, 2);
      if (c == 0) red[kWarps * kRows + warp * kRows + g] = ps;
      __half2 ph = __floats2half2_rn(p0, p1);
      *reinterpret_cast<__half2*>(sP + g * kPPitch + 8 * warp + 2 * c) = ph;
      __syncthreads();
      float l_t = 0.f;
#pragma unroll
      for (int k = 0; k < kWarps; ++k) l_t += red[kWarps * kRows + k * kRows + g];
      l_run = l_run * alpha + l_t;
      m_run = m_new;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        acc[nt][0] *= alpha;
        acc[nt][1] *= alpha;
      }
      // ---- PV (warp = DV columns 64 warp .. +63), V = first 512 B of each key row ----
      const uint32_t pa_addr = sP_u32 + ((lane & 7) * kPPitch + ((lane >> 3) & 1) * 8) * 2;
#pragma unroll
      for (int ks = 0; ks < kTile / 16; ++ks) {
        uint32_t a0, a2;
        ldmatrix_x2(a0, a2, pa_addr + ks * 32);
        const uint8_t* vr0 = kb + (ks * 16 + 2 * c) * kKPitch;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const int dv = 64 * warp + 8 * nt + g;
          const uint16_t lo = static_cast<uint16_t>(vr0[dv]) | (static_cast<uint16_t>(vr0[kKPitch + dv]) << 8);
          const uint16_t hi = static_cast<uint16_t>(vr0[8 * kKPitch + dv]) | (static_cast<uint16_t>(vr0[9 * kKPitch + dv]) << 8);
          mma_f16(acc[nt], a0, a0, a2, a2, e4m3x2_to_f16x2(lo), e4m3x2_to_f16x2(hi));
        }
      }
    } else {
      // ---- own block (8 keys, BF16 k / v staged in this buffer, causal + window), FP32 SIMT ----
      const uint8_t* kob = kb;
      const uint8_t* vob = kb + kObK;
      {
        const int r = warp;  // warp r: row r against the 8 own keys
        float sv[kRows];
#pragma unroll
        for (int j = 0; j < kRows; ++j) {
          float part = 0.f;
          for (int ch = lane; ch < kD / 8; ch += 32) {
            const uint4 qv = *reinterpret_cast<const uint4*>(sq + r * kQPitch + ch * 8);
            const uint4 kv = *reinterpret_cast<const uint4*>(kob + j * (kD * 2) + ch * 16);
            const uint32_t qa[4] = {qv.x, qv.y, qv.z, qv.w}, ka[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const float2 qf = __half22float2(*reinterpret_cast<const __half2*>(&qa[e]));
              part = fmaf(qf.x, bf16_lo(ka[e]), part);
              part = fmaf(qf.y, bf16_hi(ka[e]), part);
            }
          }
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
          sv[j] = part;
        }
        if (lane == 0) {
#pragma unroll
          for (int j = 0; j < kRows; ++j) {
            bool ok = r < q_len && j < q_len && j <= r;
            if (window > 0) ok = ok && r <= j + window;
            sob[r * kRows + j] = ok ? sv[j] * sm_scale : -INFINITY;
          }
        }
      }
      __syncthreads();
      float mt = -INFINITY;
#pragma unroll
      for (int j = 0; j < kRows; ++j) mt = fmaxf(mt, sob[g * kRows + j]);
      mt = (mt == -INFINITY) ? -1e20f : mt;
      const float m_new = fmaxf(m_run, mt);
      const float alpha = __expf(m_run - m_new);
      float pj[kRows];
      float l_t = 0.f;
#pragma unroll
      for (int j = 0; j < kRows; ++j) {
        pj[j] = __expf(sob[g * kRows + j] - m_new);
        l_t += pj[j];
      }
      l_run = l_run * alpha + l_t;
      m_run = m_new;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int dv = 64 * warp + 8 * nt + 2 * c;
        float o0 = acc[nt][0] * alpha, o1 = acc[nt][1] * alpha;
#pragma unroll
        for (int j = 0; j < kRows; ++j) {
          const uint32_t vv = *reinterpret_cast<const uint32_t*>(vob + j * (kDV * 2) + dv * 2);
          o0 = fmaf(pj[j], bf16_lo(vv), o0);
          o1 = fmaf(pj[j], bf16_hi(vv), o1);
        }
        acc[nt][0] = o0;
        acc[nt][1] = o1;
      }
    }
    __syncthreads();  // slot buffer, P and the reduction scratch are free for the next slot
    if (it + 2 < n_my) {
      issue(it + 2, buf);
      issued = it + 3;
    }
  }

  // ---- phase V: Y_warp = bf16(acc[:, 64 warp .. +63]) @ w_vc[head][64 warp .. +63, :] (BF16 mma, FP32 accumulate) ----
  mbar_wait_cta(wbar, 0);  // w_vc rows landed (bulk copies, complete_tx)
  float y[16][4];
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) y[nt][0] = y[nt][1] = y[nt][2] = y[nt][3] = 0.f;
  const uint8_t* wsm = smem + kOffW;
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    // k16 step kk covers DV 64 warp + 16 kk .. +15 = accumulator n-tiles 2 kk (k 0..7) and 2 kk + 1 (k 8..15)
    const uint32_t a0 = pack_bf16(acc[2 * kk][0], acc[2 * kk][1]);
    const uint32_t a2 = pack_bf16(acc[2 * kk + 1][0], acc[2 * kk + 1][1]);
    const int kb0 = (64 * warp + 16 * kk + 2 * c) * 2;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const uint8_t* wr = wsm + (8 * nt + g) * kWPitch + kb0;
      mma_bf16(y[nt], a0, 0u, a2, 0u, *reinterpret_cast<const uint32_t*>(wr), *reinterpret_cast<const uint32_t*>(wr + 16));
    }
  }
  float* yred = reinterpret_cast<float*>(smem + kOffYred);
#pragma unroll
  for (int nt = 0; nt < 16; ++nt)
    *reinterpret_cast<float2*>(yred + (warp * kRows + g) * kVO + 8 * nt + 2 * c) = make_float2(y[nt][0], y[nt][1]);
  float* ml = reinterpret_cast<float*>(smem + kOffML);
  if (warp == 0 && c == 0) {
    ml[g] = m_run;
    ml[kRows + g] = l_run;
  }
  __syncthreads();
  const int r = tid >> 5, n0 = (tid & 31) * 4;
  float4 ysum = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int k = 0; k < kWarps; ++k) {
    const float4 t = *reinterpret_cast<const float4*>(yred + (k * kRows + r) * kVO + n0);
    ysum.x += t.x;
    ysum.y += t.y;
    ysum.z += t.z;
    ysum.w += t.w;
  }

  // ---- send the packet to rank 0 ----
  cluster_wait();  // rank 0's mbarrier init is visible (E1)
  const uint32_t rslot = mapa(smem_u32(smem + kOffSlots + (rank - 1) * kPacket), 0);
  const uint32_t rbar = mapa(bar, 0);
  st_async_v4(rslot + tid * 16, __float_as_uint(ysum.x), __float_as_uint(ysum.y), __float_as_uint(ysum.z), __float_as_uint(ysum.w), rbar);
  if (tid < 4) {
    const float4 t = *reinterpret_cast<const float4*>(ml + tid * 4);
    st_async_v4(rslot + kRows * kVO * 4 + tid * 16, __float_as_uint(t.x), __float_as_uint(t.y), __float_as_uint(t.z), __float_as_uint(t.w),
                rbar);
  }
}

template <int C>
struct AttnBlockKernel {
  // q BF16 [T, heads, 576]; k_ext BF16 [T, >=576] (row pitch k_stride); v_ext BF16 [T, >=512]; pool FP8 (uint8 view) [slots, ..] row pitch
  // pool_stride B; indptrs int32 [bs + 1]; kv_indices int64; wvc BF16 [heads, 128, 512] contiguous; out BF16 [T, heads * 128] row pitch out_stride.
  // max_kv: the rail bound the caller sized for (the kernel traps on a rail with more than kMaxTiles tiles per worker).
  static void run(tvm::ffi::TensorView q, tvm::ffi::TensorView k_ext, tvm::ffi::TensorView v_ext, tvm::ffi::TensorView pool,
                  tvm::ffi::TensorView qo_indptr, tvm::ffi::TensorView kv_indptr, tvm::ffi::TensorView kv_indices, tvm::ffi::TensorView wvc,
                  tvm::ffi::TensorView out, int64_t q_stride, int64_t k_stride, int64_t v_stride, int64_t pool_stride, int64_t out_stride,
                  double qk_scale, double sm_scale, int64_t window, int64_t bs, int64_t max_kv) {
    using namespace host;
    static_assert(C >= 2 && C <= 24 && (C - 1) * kPacket <= kOffK, "cluster size");
    const int heads = static_cast<int>(q.size(1));
    RuntimeCheck(q.size(2) == kD && wvc.size(0) == heads && wvc.size(1) == kVO && wvc.size(2) == kDV, "attnblock: shapes");
    RuntimeCheck((max_kv + kTile - 1) / kTile <= kMaxTiles * (C - 1), "attnblock: rail longer than this cluster size covers");
    RuntimeCheck(k_stride % 8 == 0 && v_stride % 8 == 0 && q_stride % 8 == 0 && pool_stride % 16 == 0, "attnblock: 16 B row alignment");
    auto kernel = attnblock_kernel<C>;
    static const bool attr_set = [&] {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
      if (C > 8) RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      return true;
    }();
    (void)attr_set;
    LaunchKernel(dim3(C, heads, static_cast<uint32_t>(bs)), dim3(kThreads), q.device(), kSmemBytes)
        .config({.use_pdl = true, .cluster_dim = dim3(C, 1, 1)})(
            kernel, static_cast<const __nv_bfloat16*>(q.data_ptr()), static_cast<const __nv_bfloat16*>(k_ext.data_ptr()),
            static_cast<const __nv_bfloat16*>(v_ext.data_ptr()), static_cast<const uint8_t*>(pool.data_ptr()),
            static_cast<const int32_t*>(qo_indptr.data_ptr()), static_cast<const int32_t*>(kv_indptr.data_ptr()),
            static_cast<const int64_t*>(kv_indices.data_ptr()), static_cast<const __nv_bfloat16*>(wvc.data_ptr()),
            static_cast<__nv_bfloat16*>(out.data_ptr()), q_stride, k_stride, v_stride, pool_stride, out_stride, static_cast<float>(qk_scale),
            static_cast<float>(sm_scale), static_cast<int>(window));
  }
};

}  // namespace dflash_attnblock
}  // namespace sglang
