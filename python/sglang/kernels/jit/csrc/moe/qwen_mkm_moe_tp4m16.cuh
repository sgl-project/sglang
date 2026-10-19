// SPDX-License-Identifier: Apache-2.0
// Fused small-M NVFP4 MoE expert phase for Qwen3.5 TP4 decode verify (M <= 16 tokens, top-10, 512 experts,
// hidden 4096, intermediate 256 per rank): DynBlock-exact top-10 from the router logits, FC1, SwiGLU, NVFP4
// requantization and FC2 in one launch over the trtllm-gen weight layout. Writes the RT deferred-finalize rows
// (row t * 10 + k, identity permutation) and the expert weights.
// Device code generated from the engine source (source md5 ef91b74d33508ed7ff3c201374256593,
// PTX helpers md5 b4d0b7dea5606da866a21d8e1a373dd9); qwen_mkm_moe.py checks the source md5 against source_md5_lo().
//
// The router top-k key (topk_key / topk_unkey) and softmax (softmax_topk) follow FlashInfer 0.7.1 trtllm-gen MoE
// routing (TopKRedType in RoutingKernelTopK.cuh, calcSoftmax in RoutingKernel.cuh):
// Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
// Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
// Modified by NVIDIA: explicit approximate instructions reproduce FlashInfer's -use_fast_math routing bit for bit.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

// Minimal SM100 (tcgen05 / TMA / mbarrier) PTX helpers for the small-M NVFP4 MoE engine.
#include <cstdint>
#include <cuda_runtime.h>

namespace sm100_ptx {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---------------------------------------------------------------- mbarrier
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, uint32_t phase) {
  uint32_t ok;
  asm volatile(
      "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
      : "=r"(ok)
      : "r"(smem_u32(bar)), "r"(phase)
      : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  while (!mbar_try_wait(bar, phase)) {
  }
}

// ---------------------------------------------------------------- TMA / bulk copies
__device__ __forceinline__ void tma_load_2d(void* dst, const void* tmap, uint64_t* bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::
          "r"(smem_u32(dst)),
      "l"(tmap), "r"(smem_u32(bar)), "r"(c0), "r"(c1)
      : "memory");
}
__device__ __forceinline__ void bulk_load(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   smem_u32(dst)),
               "l"(src), "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}
__device__ __forceinline__ void prefetch_tmap(const void* tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}
// generic-proxy smem writes -> visible to the async proxy (UMMA / UTCCP operand reads)
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// ---------------------------------------------------------------- tcgen05
__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(dst_smem)),
               "r"(ncols)
               : "memory");
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}
__device__ __forceinline__ void tc_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
// arrive on `bar` when all prior tcgen05 async ops issued by this thread complete
__device__ __forceinline__ void tc_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar))
               : "memory");
}
// smem -> TMEM copy of one 512-byte scale-factor chunk (32 rows x 16 B), broadcast to the 4 lane quarters
__device__ __forceinline__ void utccp_32x128b_warpx4(uint32_t taddr, uint64_t sdesc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(sdesc) : "memory");
}
// D(tmem) (+)= A(smem) * B(smem)^T with NVFP4 block-16 scales from TMEM
__device__ __forceinline__ void umma_nvf4(uint32_t tmem_d, uint64_t adesc, uint64_t bdesc, uint32_t idesc,
                                          uint32_t tmem_sfa, uint32_t tmem_sfb, uint32_t accumulate) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
      " tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p;\n}\n" ::"r"(
          tmem_d),
      "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accumulate), "r"(tmem_sfa), "r"(tmem_sfb)
      : "memory");
}
// each warp reads its 32-lane quarter: 32 lanes x 16 consecutive 32-bit columns
__device__ __forceinline__ void tmem_ld_32x32b_x16(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
        "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr)
      : "memory");
}
// L2 prefetch of a contiguous global range (bytes multiple of 16, 16-byte aligned)
__device__ __forceinline__ void bulk_prefetch_l2(const void* gptr, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(gptr), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
// each warp writes its 32-lane quarter: 32 lanes x 16 consecutive 32-bit columns
__device__ __forceinline__ void tmem_st_32x32b_x16(uint32_t taddr, const uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(
          taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
      "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15])
      : "memory");
}
__device__ __forceinline__ void tmem_st_wait() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

// ---------------------------------------------------------------- descriptors
// SM100 shared-memory matrix descriptor (version 1). layout: 0 none, 2 SW128, 4 SW64, 6 SW32.
__device__ __forceinline__ uint64_t make_sdesc(uint32_t saddr, uint32_t lbo, uint32_t sbo, uint32_t layout) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 46;  // version
  d |= (uint64_t)(layout & 0x7) << 61;
  return d;
}
// Block-scaled instruction descriptor: E2M1 x E2M1, UE4M3 scales, both K-major, F32 accumulate.
__host__ __device__ constexpr uint32_t make_idesc_nvf4(uint32_t M, uint32_t N) {
  return (1u << 7)                 // a_format E2M1
         | (1u << 10)              // b_format E2M1
         | ((N >> 3) << 17)        // n_dim
         | (0u << 23)              // scale format UE4M3
         | ((M >> 4) << 24);       // m_dim
}

}  // namespace sm100_ptx

namespace sglang {

namespace qwen_mkm_moe_tp4m16 {

using namespace sm100_ptx;


constexpr int kH = 4096, kE = 512, kTopK = 10, kMaxTok = 16;
constexpr int kI = 256;             // intermediates per expert per rank (TP4)
constexpr int kN1 = 2 * kI;         // W13 rows per expert (gate + up)
constexpr int kStageBytes = 36864;  // 32 KB weights + 4 KB scale factors
constexpr int kStages = 4;
constexpr int kN1h = 256;           // W13 rows per half
constexpr int kW13BoxRows = 128;    // rows of one W13 TMA box; the host tensor map must use the same
constexpr int kHalfStages = 16;     // FC1 stages per half: 4 k-blocks each (full K = 4096)
constexpr int kFc1Stages = 2 * kHalfStages;
constexpr int kFc2Stages = 16;      // one tile of 256 W2 rows x 128 B each
constexpr int kTotStages = kFc1Stages + kFc2Stages;
constexpr int kThreads = 256;
// TMEM columns. FC1: accumulator [0,256), x SFA ring [256,320) (16 slots x 4), SFB ring [384,512) (16 slots x 8).
// FC2: accumulator [0,256), h SFA [256,384) (32 atoms x 4), SFB ring [384,512) (16 slots x 8).
constexpr uint32_t kColSfaX = 256, kColSfa = 256, kColSfb = 384;
constexpr int kSfbSlots = 16, kSfaSlots = 16;
constexpr int kARow = 260;  // FC1 accumulator row stride in floats: 16 B aligned, rows on distinct banks
#ifndef L2_EVICT_FIRST  // stream expert weights with an L2 evict-first policy (they are read once per layer)
#define L2_EVICT_FIRST 1
#endif
#ifndef TOPK_RANK
#define TOPK_RANK 0
#endif
#ifndef HOIST_SCALES
#define HOIST_SCALES 1
#endif
#ifndef SF_UTCCP_FC1
#define SF_UTCCP_FC1 0
#endif
#ifndef SPEC_STAGES  // ring stages issued for the speculated expert before the routing verdict (rest: after it)
#define SPEC_STAGES 4
#endif
#ifndef STAGE_STAMPS  // DIAG: per-stage issue (slots 64 + st) and landing (16 + st, seen by warp 4) stamps
#define STAGE_STAMPS 0
#endif
constexpr int kTraceStride = STAGE_STAMPS ? 128 : 16;
#ifndef SMEM_SPACE  // 1: shared-memory accesses through Smem compile to shared-space instructions
#define SMEM_SPACE 1
#endif
#ifndef PF_SET  // L2 prefetch of the stages that wait for a ring slot at: 1 = start, 2 = FC1 half 1, 4 = FC2, 8 = FC2 pass 1
#define PF_SET 5
#endif
#ifndef PF_N1  // stages prefetched at the start trigger
#define PF_N1 4
#endif
#ifndef PF_N2  // stages prefetched at the FC2 trigger
#define PF_N2 4
#endif
#ifndef TAILV  // FC2 drains: bit 0 = last pass on all 8 warps; bit 1 = TMEM loads double-buffered; bit 2 = coalesced stores
#define TAILV 3
#endif
#define TAIL8 (TAILV & 1)
#define DRAIN_PIPE ((TAILV >> 1) & 1)
#define DRAIN_SMEM ((TAILV >> 2) & 1)
#ifndef EARLY_SF2  // FC2 stages whose scale factors are staged before the h epilogue (old order: 2)
#define EARLY_SF2 0
#endif
#ifndef FC2_PF  // FC2 tiles (beyond the first kStages) prefetched to L2 ahead of the ring; 0 = off
#define FC2_PF 0
#endif
#ifndef FC2_PF_LEAD  // issue the FC2 prefetch when the producer reaches FC1 stage kFc1Stages - FC2_PF_LEAD
#define FC2_PF_LEAD 4
#endif

struct Params {
  const uint16_t* logits;    // [M, 512] bf16 router logits (router_gemm_quant output)
  uint16_t* expert_weights;  // [M, 10] bf16 out: DynBlock softmax-over-top-10 weights (CTA 0 writes)
  int32_t* topk_ids;         // [M, 10] out (CTA 0 writes)
  const uint8_t* x_sf;       // [M, 256] linear e4m3
  const uint8_t* w13_sf;     // [E, 4 row blocks x 64 k blocks x 512 B] (128x4 interleaved)
  const uint8_t* w2_sf;      // [E, 32 row blocks x 4 k blocks x 512 B]
  const float* g1_alpha;     // [E] gate dequant
  const float* g1_scale_c;   // [E] up dequant * FC2 input quant scale
  const float* g2_alpha;     // [E]
  __nv_bfloat16* out;        // [M * 10, 4096]
  long long* trace;          // optional [grid, 16] %globaltimer stamps
  int num_tokens;
};

struct __align__(1024) Smem {
  uint8_t ring[kStages][kStageBytes];
  alignas(1024) uint8_t xa[32 * 1024];  // x: per 128-B K chunk c, 2 KB = SW128 atoms of rows 0..7 | rows 8..15
  alignas(1024) uint8_t ha[16 * 1024];  // h: 8 copies of (rows 0..7 | rows 8..15) x 128 B (K = 256), SW128
  float acc1[16 * kARow];               // FC1 accumulator rows 0..15 (tokens) x 256 columns of one half
  uint8_t xsf[16][256];                 // x scale bytes, full K (row t = token t), by bulk copy
  uint32_t hsf[16][4];                  // h scale words: token t, k-block kb (4 e4m3 bytes)
  uint64_t full[kStages], empty[kStages], sf_full[kStages];
  uint64_t x_full, fc1_half, acc_free, acc_ready[2], acc_read, h_ready, fc2_pass[2], fc2_free, route_done;
  uint32_t tmem;
  int expert;
  uint32_t tokmask;
  int kslot[16];
  alignas(16) uint32_t cand[8][32];  // read as uint4 by the rank-based top-k
  int32_t ids[16][kTopK];
};

__device__ __forceinline__ uint64_t gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
  return t;
}
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}
// no release: the phase only orders DSMEM traffic that mbarriers already completed, so pending global stores
// (stamps, routing outputs, output rows) need not drain before the arrive
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t mapa(uint32_t saddr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(saddr), "r"(rank));
  return r;
}
// smem (this CTA) -> smem (peer CTA in the cluster), completes tx bytes on the peer's mbarrier
__device__ __forceinline__ void bulk_s2s(uint32_t dst_cluster, uint32_t src, uint32_t bytes, uint32_t bar_cluster) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   dst_cluster),
               "r"(src), "r"(bytes), "r"(bar_cluster)
               : "memory");
}
// registers -> the peer CTA's shared memory; completes the bytes on the peer's mbarrier
__device__ __forceinline__ void st_async_v4(uint32_t dst, const uint4& v, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(dst),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(bar)
               : "memory");
}
__device__ __forceinline__ void st_async_b32(uint32_t dst, uint32_t v, uint32_t bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(dst), "r"(v), "r"(bar)
               : "memory");
}
__device__ __forceinline__ void tmem_st_x4(uint32_t taddr, const uint4& v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};" ::"r"(taddr), "r"(v.x), "r"(v.y),
               "r"(v.z), "r"(v.w)
               : "memory");
}
// bar.sync counts arrivals per warp execution: reconverge first, so a warp whose lane 0 just ran a stamp or an
// mbarrier arrive cannot arrive twice and release the barrier early
__device__ __forceinline__ void named_bar(int id, int n) {
  __syncwarp();
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void tmem_ld_x32(uint32_t taddr, uint32_t (&r)[32]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
      "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_ld_x64(uint32_t taddr, uint32_t (&r)[64]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
      "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]),
        "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]),
        "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]),
        "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]),
        "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_st_x8(uint32_t taddr, const uint32_t (&r)[8]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"r"(taddr), "r"(r[0]),
               "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7])
               : "memory");
}
__device__ __forceinline__ void tmem_st_x16(uint32_t taddr, const uint32_t (&r)[16]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::
                   "r"(taddr),
               "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
               "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15])
               : "memory");
}
__device__ __forceinline__ void tmem_st_x32(uint32_t taddr, const uint32_t (&r)[32]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,"
      "%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
      "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]),
      "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]),
      "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31])
      : "memory");
}
__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float mul_ftz(float a, float b) {
  float y;
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float rcp_approx(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t cvt_e4m3(float a) {
  uint32_t out;
  asm("{\n .reg .b16 pair;\n .reg .f32 zero;\n mov.f32 zero, 0f00000000;\n"
      " cvt.rn.satfinite.e4m3x2.f32 pair, zero, %1;\n cvt.u32.u16 %0, pair;\n}"
      : "=r"(out)
      : "f"(a));
  return out & 0xFFu;
}
__device__ __forceinline__ float e4m3_to_f32(uint32_t v) {
  float out;
  asm("{\n .reg .b16 pair;\n .reg .b32 h2;\n .reg .b16 lo, hi;\n cvt.u16.u32 pair, %1;\n"
      " cvt.rn.f16x2.e4m3x2 h2, pair;\n mov.b32 {lo, hi}, h2;\n cvt.f32.f16 %0, lo;\n}"
      : "=f"(out)
      : "r"(v));
  return out;
}
__device__ __forceinline__ uint32_t cvt_e2m1x8(const float* v) {
  uint32_t out;
  asm("{\n .reg .b8 b0, b1, b2, b3;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      " mov.b32 %0, {b0, b1, b2, b3};\n}"
      : "=r"(out)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return out;
}
// ---- router top-k, bit-exact with FlashInfer's single-block routing (routingCustom, RenormalizeNaive on bf16):
// key = bf16 order (cub float twiddle) above 65535 - expert, i.e. the higher score first, the smaller expert on ties.
__device__ __forceinline__ uint32_t topk_key(uint32_t bits, uint32_t expert) {
  const uint32_t twiddled = bits ^ ((bits & 0x8000u) ? 0xFFFFu : 0x8000u);
  return (twiddled << 16) | (65535u - expert);
}
__device__ __forceinline__ void topk_unkey(uint32_t key, uint32_t& bits, uint32_t& expert) {
  expert = 65535u - (key & 0xFFFFu);
  const uint32_t twiddled = key >> 16;
  bits = twiddled ^ ((twiddled & 0x8000u) ? 0x8000u : 0xFFFFu);
}
__device__ __forceinline__ uint32_t max16(const uint32_t (&k)[16]) {
  const uint32_t a = max(max(max(k[0], k[1]), max(k[2], k[3])), max(max(k[4], k[5]), max(k[6], k[7])));
  const uint32_t b = max(max(max(k[8], k[9]), max(k[10], k[11])), max(max(k[12], k[13]), max(k[14], k[15])));
  return max(a, b);
}
__device__ __forceinline__ uint32_t warp_max_u32(uint32_t v) {
  uint32_t r;
  asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
}
__device__ __forceinline__ uint32_t warp_sum_u32(uint32_t v) {
  uint32_t r;
  asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
  return r;
}
__device__ __forceinline__ uint32_t warp_topk_of_lanes(uint32_t v, uint32_t lane) {
  uint32_t mine = 0;
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    const uint32_t top = warp_max_u32(v);
    v = v == top ? 0u : v;
    mine = lane == k ? top : mine;
  }
  return mine;
}
__device__ __forceinline__ uint32_t count_greater(const uint32_t* s_cand, uint32_t v) {
  uint32_t r = 0;
#pragma unroll
  for (int j = 0; j < 32; j += 4) {
    const uint4 q = *reinterpret_cast<const uint4*>(s_cand + j);
    r += (q.x > v) + (q.y > v) + (q.z > v) + (q.w > v);
  }
  return r;
}
// Top-10 keys of 512 (16 per lane), lane k < 10 holding the k-th largest (keys unique and nonzero). Ranks come from
// all-pairs compares over shared memory rather than chains of warp reductions.
__device__ __forceinline__ uint32_t warp_top10(uint32_t (&keys)[16], uint32_t lane, uint32_t* s_cand) {
  // bound = the 10th largest lane maximum; every top-10 key is >= bound
  const uint32_t m = max16(keys);
  s_cand[lane] = m;
  __syncwarp();
  const uint32_t r = count_greater(s_cand, m);
  const uint32_t bound = __shfl_sync(0xffffffffu, m, __ffs(__ballot_sync(0xffffffffu, r == kTopK - 1)) - 1);
  uint32_t count = 0;
#pragma unroll
  for (uint32_t i = 0; i < 16; ++i) count += keys[i] >= bound;
  // exclusive prefix of count over the lanes, from the ballots of its bits (count <= 16)
  const uint32_t lt = (1u << lane) - 1u;
  uint32_t offset = 0, total = 0;
#pragma unroll
  for (int b = 0; b < 5; ++b) {
    const uint32_t bb = __ballot_sync(0xffffffffu, (count >> b) & 1u);
    offset += (uint32_t)__popc(bb & lt) << b;
    total += (uint32_t)__popc(bb) << b;
  }
  if (total <= 32) {
    __syncwarp();  // every lane has read the lane maxima
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
      if (keys[i] >= bound) s_cand[offset++] = keys[i];
    }
    if (lane >= total) s_cand[lane] = 0u;
    __syncwarp();
    const uint32_t cand = s_cand[lane];
    const uint32_t rk = count_greater(s_cand, cand);
    __syncwarp();
    if (lane < total && rk < kTopK) s_cand[rk] = cand;
    __syncwarp();
    return lane < kTopK ? s_cand[lane] : 0u;
  }
  uint32_t mine = 0;
#pragma unroll 1
  for (uint32_t k = 0; k < kTopK; ++k) {
    const uint32_t top = warp_max_u32(max16(keys));
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) keys[i] = keys[i] == top ? 0u : keys[i];
    mine = lane == k ? top : mine;
  }
  return mine;
}
// Top-10 keys of 512 (16 per lane), lane k holding the k-th largest (keys unique and nonzero).
__device__ __forceinline__ uint32_t warp_top10_redux(uint32_t (&keys)[16], uint32_t lane, uint32_t* s_cand) {
  uint32_t v = max16(keys);
  uint32_t bound = 0;
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    bound = warp_max_u32(v);
    v = v == bound ? 0u : v;
  }
  uint32_t count = 0;
#pragma unroll
  for (uint32_t i = 0; i < 16; ++i) count += keys[i] >= bound;
  const uint32_t total = warp_sum_u32(count);
  if (total <= 32) {
    uint32_t offset = count;
#pragma unroll
    for (uint32_t d = 1; d < 32; d <<= 1) {
      const uint32_t up = __shfl_up_sync(0xffffffffu, offset, d);
      offset += lane >= d ? up : 0u;
    }
    offset -= count;
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
      if (keys[i] >= bound) s_cand[offset++] = keys[i];
    }
    __syncwarp();
    const uint32_t cand = lane < total ? s_cand[lane] : 0u;
    __syncwarp();
    return warp_topk_of_lanes(cand, lane);
  }
  uint32_t mine = 0;
#pragma unroll 1
  for (uint32_t k = 0; k < kTopK; ++k) {
    const uint32_t top = warp_max_u32(max16(keys));
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) keys[i] = keys[i] == top ? 0u : keys[i];
    mine = lane == k ? top : mine;
  }
  return mine;
}
// FlashInfer calcSoftmax (bf16 scores) as built with -use_fast_math: expf -> ex2.approx.ftz of x * log2(e),
// division -> div.full.ftz; reductions are cooperative_groups::reduce like the reference.
__device__ __forceinline__ float fast_expf(float x) {
  float y;
  asm("{\n .reg .f32 t;\n mul.ftz.f32 t, %1, 0f3FB8AA3B;\n ex2.approx.ftz.f32 %0, t;\n}" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float fast_div(float a, float b) {
  float y;
  asm("div.full.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ uint16_t softmax_topk(uint16_t score_bits, uint32_t lane) {
  namespace cg = cooperative_groups;
  const auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  const float si = __uint_as_float((uint32_t)score_bits << 16);
  float max_score = -INFINITY;
  if (lane < kTopK) max_score = si >= max_score ? si : max_score;
  max_score = cg::reduce(warp, max_score, cg::greater<float>());
  float sum_score = 0.f, new_score = 0.f;
  if (lane < kTopK) {
    new_score = fast_expf(__fsub_rn(si, max_score));
    sum_score = __fadd_rn(sum_score, new_score);
  }
  sum_score = cg::reduce(warp, sum_score, cg::plus<float>());
  uint16_t out = score_bits;
  if (lane < kTopK) {
    const __nv_bfloat16 b = __float2bfloat16_rn(fast_div(new_score, sum_score));
    out = *reinterpret_cast<const uint16_t*>(&b);
  }
  return out;
}

__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void tma_load_2d_w(void* dst, const void* tmap, uint64_t* bar, int c0, int c1, uint64_t pol) {
  if (L2_EVICT_FIRST)
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(smem_u32(dst)),
        "l"(tmap), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "l"(pol)
        : "memory");
  else
    tma_load_2d(dst, tmap, bar, c0, c1);
}
__device__ __forceinline__ void bulk_load_w(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t pol) {
  if (L2_EVICT_FIRST)
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(
                     smem_u32(dst)),
                 "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(pol)
                 : "memory");
  else
    bulk_load(dst, src, bytes, bar);
}

__device__ __forceinline__ void stamp(const Params& p, int slot) {
  if (p.trace) p.trace[blockIdx.x * kTraceStride + slot] = (long long)gtime();
}
// stamp after `dep` is available (the branch needs its value; keys are never 0)
__device__ __forceinline__ void stamp_dep(const Params& p, int slot, uint32_t dep) {
  if (dep == 0u) __trap();
  stamp(p, slot);
}
__device__ __forceinline__ void tma_prefetch_2d(const void* tmap, int c0, int c1) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" ::"l"(tmap), "r"(c0), "r"(c1) : "memory");
}
// Physical W13 column (accumulator column) holding reordered row q: block-32 shuffle of FlashInfer's
// get_shuffle_matrix_a_row_indices (epilogue_tile_m = 128); reordered row 2j is up_j, 2j + 1 is gate_j.
__device__ __forceinline__ int w13_col(int q) { return (q & ~31) + ((q & 31) % 4) * 8 + (q & 31) / 4; }
__host__ __device__ constexpr int w13_logical(int p) {
  return ((p & ~31) + (p % 8) * 4 + (p & 31) / 8) % 2 ? ((p & ~31) + (p % 8) * 4 + (p & 31) / 8) / 2
                                                     : 128 + ((p & ~31) + (p % 8) * 4 + (p & 31) / 8) / 2;
}


// FC2 drain of one pass: TMEM lane quarter q (tiles 2 q, 2 q + 1 of the pass), 256 accumulator columns
__device__ __forceinline__ void reg_fence64(uint32_t (&r)[64]) {  // keep uses of r after the tcgen05.wait::ld
#pragma unroll
  for (int i = 0; i < 64; ++i) asm volatile("" : "+r"(r[i]));
}

// FC2 accumulator columns [64 C0, 64 C1) of this warp's lane quarter -> bf16 gemm2_out rows (row = tile token)
template <int C0, int C1>
__device__ __forceinline__ void fc2_drain16(const Params& p, Smem& s, uint32_t tm, int q, int lane, int pass, float a2) {
  const int row = q * 32 + lane, tile = pass * 8 + row / 16, t = row % 16;
  const bool valid = t < p.num_tokens && ((s.tokmask >> t) & 1u);
  __nv_bfloat16* orow = valid ? p.out + (size_t)(t * kTopK + s.kslot[t]) * kH + tile * 256 : nullptr;
  const uint32_t base = tm + ((uint32_t)(q * 32) << 16);
  __nv_bfloat16* orow4[4];  // DRAIN_SMEM readback: token rows (k * 4 + lane / 8) % 16 of tile 2 q (+ 256 for 2 q + 1)
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int tr = (k * 4 + lane / 8) % 16;
    orow4[k] = (DRAIN_SMEM && tr < p.num_tokens && ((s.tokmask >> tr) & 1u))
                   ? p.out + (size_t)(tr * kTopK + s.kslot[tr]) * kH + (pass * 8 + 2 * q) * 256
                   : nullptr;
  }
  auto put = [&](const uint32_t (&r)[64], int c) {
    if (DRAIN_SMEM) {
      // rows go through a per-warp 4 KB buffer in x's smem (dead after FC1): 32 rows x 128 B, 16-B chunks XOR-swizzled
      // by row, read back so that 8 lanes store one row's 128 B (full lines instead of 16-B pieces of 32 rows).
      // Shared-space accesses for the staging buffer (tunable).
      const uint32_t stg = smem_u32(s.xa) + (threadIdx.x / 32) * 4096;
#pragma unroll
      for (int b32 = 0; b32 < 2; ++b32) {
        uint32_t pk[16];
#pragma unroll
        for (int l = 0; l < 32; l += 2) {
          const float v0 = __uint_as_float(r[b32 * 32 + (l % 4) * 8 + l / 4]) * a2;
          const float v1 = __uint_as_float(r[b32 * 32 + ((l + 1) % 4) * 8 + (l + 1) / 4]) * a2;
          const __nv_bfloat162 b2 = __floats2bfloat162_rn(v0, v1);
          pk[l / 2] = *reinterpret_cast<const uint32_t*>(&b2);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
          asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(stg + lane * 128 + (((b32 * 4 + i) ^ (lane & 7)) * 16)),
                       "r"(pk[4 * i]), "r"(pk[4 * i + 1]), "r"(pk[4 * i + 2]), "r"(pk[4 * i + 3]) : "memory");
      }
      __syncwarp();
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int rr = k * 4 + lane / 8, ch = lane % 8;
        uint4 v;
        asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "r"(stg + rr * 128 + ((ch ^ (rr & 7)) * 16)) : "memory");
        if (orow4[k % 4]) *reinterpret_cast<uint4*>(orow4[k % 4] + (k / 4) * 256 + c * 64 + ch * 8) = v;
      }
      __syncwarp();
    } else if (valid) {
#pragma unroll
      for (int b32 = 0; b32 < 2; ++b32) {
        uint32_t pk[16];
#pragma unroll
        for (int l = 0; l < 32; l += 2) {
          const float v0 = __uint_as_float(r[b32 * 32 + (l % 4) * 8 + l / 4]) * a2;
          const float v1 = __uint_as_float(r[b32 * 32 + ((l + 1) % 4) * 8 + (l + 1) / 4]) * a2;
          const __nv_bfloat162 b2 = __floats2bfloat162_rn(v0, v1);
          pk[l / 2] = *reinterpret_cast<const uint32_t*>(&b2);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
          *reinterpret_cast<uint4*>(orow + c * 64 + b32 * 32 + i * 8) =
              make_uint4(pk[4 * i], pk[4 * i + 1], pk[4 * i + 2], pk[4 * i + 3]);
      }
    }
  };
  if (DRAIN_PIPE) {  // the next chunk's TMEM load is in flight while this chunk converts and stores
    uint32_t ra[64], rb[64];
    tmem_ld_x64(base + C0 * 64, ra);
    tmem_ld_wait();
    reg_fence64(ra);
#pragma unroll
    for (int c = C0; c < C1; c += 2) {
      if (c + 1 < C1) tmem_ld_x64(base + (uint32_t)((c + 1) * 64), rb);
      put(ra, c);
      if (c + 1 < C1) {
        tmem_ld_wait();
        reg_fence64(rb);
        if (c + 2 < C1) tmem_ld_x64(base + (uint32_t)((c + 2) * 64), ra);
        put(rb, c + 1);
        if (c + 2 < C1) { tmem_ld_wait(); reg_fence64(ra); }
      }
    }
  } else {
#pragma unroll 1
    for (int c = C0; c < C1; ++c) {
      uint32_t r[64];
      tmem_ld_x64(base + (uint32_t)(c * 64), r);
      tmem_ld_wait();
      put(r, c);
    }
  }
}

__global__ void __launch_bounds__(kThreads, 1)
fused_expert_kernel(const __grid_constant__ CUtensorMap tm13, const __grid_constant__ CUtensorMap tm2,
                    const __grid_constant__ CUtensorMap tmx, const Params p) {
  extern __shared__ __align__(1024) uint8_t raw[];
#if SMEM_SPACE  // pointer arithmetic on the __shared__ array keeps the shared address space
  Smem& s = *reinterpret_cast<Smem*>(raw + ((1024u - (smem_u32(raw) & 1023u)) & 1023u));
#else
  Smem& s = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(raw) + 1023) & ~uintptr_t(1023));
#endif
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int slot = blockIdx.x;  // (token, top-k slot) served by this CTA
  const int M = p.num_tokens;
  if (threadIdx.x == 0) {
    stamp(p, 0);
    for (int i = 0; i < kStages; ++i) {
      mbar_init(&s.full[i], 1);
      mbar_init(&s.empty[i], 1);
      mbar_init(&s.sf_full[i], 4);
    }
    mbar_init(&s.x_full, 1);
    mbar_init(&s.fc1_half, 1);
    mbar_init(&s.acc_free, 2);   // warps 0 and 4 drained half 0
    mbar_init(&s.acc_ready[0], 2);  // per half (one barrier each, single phase): warps 0 and 4 drained
    mbar_init(&s.acc_ready[1], 2);  // their columns into acc1
    mbar_init(&s.acc_read, 2);   // warps 0 and 3 finished the half-0 epilogue reads of acc1
    mbar_init(&s.h_ready, 4);
    mbar_init(&s.fc2_pass[0], 1);  // one single-phase barrier per FC2 pass
    mbar_init(&s.fc2_pass[1], 1);
    mbar_init(&s.fc2_free, 4);
    mbar_init(&s.route_done, 1);
    fence_mbar_init();
  }
  if (warp == 2) { tmem_alloc(&s.tmem, 512); tmem_relinquish(); tc_fence_before(); }
  if (warp == 1 && lane == 0) { prefetch_tmap(&tm13); prefetch_tmap(&tm2); prefetch_tmap(&tmx); }
  __syncthreads();
  tc_fence_after();
  const uint32_t tm = s.tmem;
  const int my_t = slot / kTopK, my_k = slot % kTopK;
  if (my_t >= M) {
    if (warp == 2) tmem_dealloc(tm, 512);
    return;
  }
  auto issue_stage = [&](int st, int e, uint64_t pol) {
    const int b = st % kStages;
    mbar_expect_tx(&s.full[b], kStageBytes);
    uint8_t* dst = s.ring[b];
    if (st < kFc1Stages) {
      const int h = st / kHalfStages, ks = st % kHalfStages;
      // rows [256 h, 256 h + 256) of this expert as two boxes of 128 rows, K bytes [128 ks, 128 ks + 128)
      tma_load_2d_w(dst, &tm13, &s.full[b], ks * 128, e * kN1 + h * kN1h, pol);
      tma_load_2d_w(dst + 16384, &tm13, &s.full[b], ks * 128, e * kN1 + h * kN1h + 128, pol);
      // scale atoms of row blocks 2 h and 2 h + 1, k-blocks [4 ks, 4 ks + 4)
      const uint8_t* sf = p.w13_sf + (size_t)e * (kN1 * kH / 16);
      bulk_load_w(dst + 32768, sf + (size_t)(2 * h * 64 + ks * 4) * 512, 2048, &s.full[b], pol);
      bulk_load_w(dst + 32768 + 2048, sf + (size_t)((2 * h + 1) * 64 + ks * 4) * 512, 2048, &s.full[b], pol);
    } else {
      const int j = st - kFc1Stages;
      const int row0 = j * 256;
      tma_load_2d_w(dst, &tm2, &s.full[b], 0, e * kH + row0, pol);
      bulk_load_w(dst + 32768, p.w2_sf + (size_t)e * (kH * 16) + (size_t)(row0 / 128) * 4 * 512, 4096, &s.full[b], pol);
    }
  };
  auto prefetch_stage = [&](int st, int e) {  // the same bytes as issue_stage, to L2 only
    if (st < kFc1Stages) {
      const int h = st / kHalfStages, ks = st % kHalfStages;
      tma_prefetch_2d(&tm13, ks * 128, e * kN1 + h * kN1h);
      tma_prefetch_2d(&tm13, ks * 128, e * kN1 + h * kN1h + 128);
      const uint8_t* sf = p.w13_sf + (size_t)e * (kN1 * kH / 16);
      bulk_prefetch_l2(sf + (size_t)(2 * h * 64 + ks * 4) * 512, 2048);
      bulk_prefetch_l2(sf + (size_t)((2 * h + 1) * 64 + ks * 4) * 512, 2048);
    } else {
      const int row0 = (st - kFc1Stages) * 256;
      tma_prefetch_2d(&tm2, 0, e * kH + row0);
      bulk_prefetch_l2(p.w2_sf + (size_t)e * (kH * 16) + (size_t)(row0 / 128) * 4 * 512, 4096);
    }
  };
  // routing: warps 0, 2..7 take tokens 0..15 (DynBlock-exact top-10 + softmax), up to 3 rounds; warp 1 recomputes token
  // my_t's top-10 and starts this CTA's weight stream at once (a duplicate slot only wastes its first ring fill)
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (threadIdx.x == 32) stamp(p, 10);
  if (warp == 3 && lane == 0) {
    // x (16 chunks of 16 rows x 128 B, rows >= M zero-filled) and its scale bytes; independent of the routing
    mbar_expect_tx(&s.x_full, 32 * 1024 + M * 256);
#pragma unroll
    for (int a = 0; a < 16; ++a) tma_load_2d(s.xa + a * 2048, &tmx, &s.x_full, a * 128, 0);
    for (int t = 0; t < M; ++t) bulk_load(s.xsf[t], p.x_sf + (size_t)t * (kH / 16), 256, &s.x_full);
  }
  auto top10v = [&](const uint4 v0, const uint4 v1) -> uint32_t {
    uint32_t keys[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t w = i < 4 ? (&v0.x)[i] : (&v1.x)[i - 4];
      keys[2 * i] = topk_key(w & 0xFFFFu, (uint32_t)(lane * 16 + 2 * i));
      keys[2 * i + 1] = topk_key(w >> 16, (uint32_t)(lane * 16 + 2 * i + 1));
    }
    return TOPK_RANK ? warp_top10(keys, (uint32_t)lane, s.cand[warp]) : warp_top10_redux(keys, (uint32_t)lane, s.cand[warp]);
  };
  auto top10 = [&](int rtok) -> uint32_t {
    const uint4* src = reinterpret_cast<const uint4*>(p.logits + (size_t)rtok * kE + lane * 16);
    return top10v(src[0], src[1]);
  };
  int e;
  if (warp == 1) {
    const uint32_t key = top10(my_t);
    uint32_t bits, espec;
    topk_unkey(__shfl_sync(0xffffffffu, key, my_k), bits, espec);
    if (lane == 0) {
      stamp(p, 9);
      const uint64_t pol = policy_evict_first();
      for (int st = 0; st < SPEC_STAGES; ++st) {
        issue_stage(st, (int)espec, pol);
        if (STAGE_STAMPS) stamp(p, 64 + st);
      }
      stamp(p, 2);
    }
    mbar_wait(&s.route_done, 0);
    e = s.expert;
  } else {
    const int base = warp == 0 ? 0 : warp - 1;
    // every round's logits are requested before the first ranking, so later rounds do not pay the load latency again
    uint4 lv[3][2];
#pragma unroll
    for (int r = 0; r < 3; ++r) {
      const int rtok = base + 7 * r;
      if (rtok < M) {
        const uint4* src = reinterpret_cast<const uint4*>(p.logits + (size_t)rtok * kE + lane * 16);
        lv[r][0] = src[0];
        lv[r][1] = src[1];
      }
    }
#pragma unroll
    for (int r = 0; r < 3; ++r) {
      const int rtok = base + 7 * r;
      if (rtok >= M) break;
      const uint32_t key = top10v(lv[r][0], lv[r][1]);
      uint32_t bits, expert;
      topk_unkey(key, bits, expert);
      const uint16_t wbits = softmax_topk((uint16_t)bits, (uint32_t)lane);
      if (lane < kTopK) {
        s.ids[rtok][lane] = (int32_t)expert;
        if (blockIdx.x == 0) {
          p.expert_weights[rtok * kTopK + lane] = wbits;
          p.topk_ids[rtok * kTopK + lane] = (int32_t)expert;
        }
      }
      __syncwarp();
    }
    named_bar(4, 224);
    if (warp == 0) {
      // this slot's expert, whether an earlier slot picked it, and which tokens use it (top-k slot per token)
      const int ee = s.ids[my_t][my_k];
      bool dup = false;
      for (int q = lane; q < slot; q += 32) dup |= s.ids[q / kTopK][q % kTopK] == ee;
      dup = __any_sync(0xffffffffu, dup);
      uint32_t mbit = 0;
      int ks = 0;
      if (lane < M) {
#pragma unroll
        for (int k = 0; k < kTopK; ++k)
          if (s.ids[lane][k] == ee) { mbit = 1u << lane; ks = k; }
      }
      const uint32_t mask = __reduce_or_sync(0xffffffffu, mbit);
      if (lane < 16) s.kslot[lane] = ks;
      if (lane == 0) {
        s.expert = dup ? -1 : ee;
        s.tokmask = mask;
        stamp(p, 1);
        mbar_arrive(&s.route_done);
      }
    }
    named_bar(4, 224);
    e = s.expert;
  }
  if (e < 0) {  // duplicate slot: let the speculative loads land, then leave
    if (warp == 1) {
      if (lane == 0)
        for (int b = 0; b < SPEC_STAGES; ++b) mbar_wait(&s.full[b], 0);
      __syncwarp();
    }
    mbar_wait(&s.x_full, 0);
    named_bar(7, kThreads);
    if (warp == 2) tmem_dealloc(tm, 512);
    return;
  }
  tc_fence_after();
  float ag = 0.f, sc = 0.f, a2 = 0.f;
  if (HOIST_SCALES) { ag = p.g1_alpha[e]; sc = p.g1_scale_c[e]; a2 = p.g2_alpha[e]; }
  // FC1 drain of quarter-0 accumulator columns [64 c0, 64 c0 + 128) (TMEM lanes 0..15 hold the token rows) into acc1
  auto drain_fc1 = [&](int c0) {
#pragma unroll 1
    for (int c = c0; c < c0 + 2; ++c) {
      uint32_t r[64];
      tmem_ld_x64(tm + (uint32_t)(c * 64), r);
      tmem_ld_wait();
      if (lane < 16) {
#pragma unroll
        for (int i = 0; i < 16; ++i)
          *reinterpret_cast<uint4*>(s.acc1 + lane * kARow + c * 64 + 4 * i) =
              make_uint4(r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]);
      }
    }
    tc_fence_before();
    __syncwarp();
  };
  // SwiGLU + NVFP4 epilogue of half h, tasks (token row, 16-wide block, half) = 16 x 8 x 2 = 256, task = et + k * nt.
  // Accumulator columns of each 32-block: [0, 8) up 2m, [8, 16) gate 2m, [16, 24) up 2m + 1, [24, 32) gate 2m + 1
  auto epi_tasks = [&](int h, int et, int nt) {
    if (!HOIST_SCALES) { ag = p.g1_alpha[e]; sc = p.g1_scale_c[e]; }
#pragma unroll 1
    for (int task = et; task < 256; task += nt) {
      const int row = task / 16, blk = (task / 2) % 8, hf = task % 2;
      const float4* ar = reinterpret_cast<const float4*>(s.acc1 + row * kARow + 32 * blk + 4 * hf);
      const float4 ue = ar[0], ge = ar[2], uo = ar[4], go = ar[6];
      const float ga[8] = {ge.x, go.x, ge.y, go.y, ge.z, go.z, ge.w, go.w};
      const float ua[8] = {ue.x, uo.x, ue.y, uo.y, ue.z, uo.z, ue.w, uo.w};
      // Same SwiGLU + NVFP4 requantization results as trtllm-gen's FC1 epilogue (bit-equal FC2 inputs);
      // a = FC1 dequant, c = a x FC2 input scale:
      // h' = (u a)(g sigmoid(g a)), sigmoid = rcp(ex2(-(g (a log2e))) + 1), block scale e4m3(max|h'| c / 6), FP4 (h' / scale) c
      const float a_l2e = mul_ftz(ag, 1.4426950216293334961f);
      float hv[8];
      float amax = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float ex = ex2_approx(-(ga[i] * a_l2e));
        const float sg = rcp_approx(ex + 1.0f);
        const float v = fmaf(ua[i], ag, 0.0f) * (ga[i] * sg);
        hv[i] = v;
        amax = fmaxf(amax, fabsf(v));
      }
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
      const uint32_t sf = cvt_e4m3(fmaxf(0.f, fabsf(amax * sc)) * 0.16666667163372039795f);
      const float sff = e4m3_to_f32(sf);
      const float osc = sff == 0.f ? 0.f : rcp_approx(sff);
      float qv[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) qv[i] = mul_ftz(hv[i], osc) * sc;
      const uint32_t packed = cvt_e2m1x8(qv);
      const int gblk = 8 * h + blk;  // 16-wide block over the 256 intermediates; k-blocks 2 h, 2 h + 1
      const int chunk = gblk / 2, r8 = row % 8;
      const uint32_t off = (uint32_t)((row / 8) * 1024 + r8 * 128 + ((chunk ^ r8) * 16) + (gblk % 2) * 8 + hf * 4);
#pragma unroll
      for (int cp = 0; cp < 8; ++cp) *reinterpret_cast<uint32_t*>(s.ha + cp * 2048 + off) = packed;
      if (hf == 0) reinterpret_cast<uint8_t*>(&s.hsf[row][gblk / 4])[gblk % 4] = (uint8_t)sf;
    }
  };

  if (warp == 1) {
    if (lane == 0) {
      // ---------------- TMA producer: the rest of the weight stream (stages 0 .. kStages - 1 went out above)
      const uint64_t pol = policy_evict_first();
      for (int st = SPEC_STAGES; st < kTotStages; ++st) {
        if (FC2_PF > 0 && st == kFc1Stages - FC2_PF_LEAD) {
          // the FC2 tiles beyond the ring go to L2 now, while the last FC1 stages and h are in progress
          for (int j = kStages; j < kStages + FC2_PF; ++j) {
            const int row0 = j * 256;
            tma_prefetch_2d(&tm2, 0, e * kH + row0);
            bulk_prefetch_l2(p.w2_sf + (size_t)e * (kH * 16) + (size_t)(row0 / 128) * 4 * 512, 4096);
          }
        }
        if (PF_SET != 0 && ((st == SPEC_STAGES && (PF_SET & 1)) || (st == kHalfStages + 4 && (PF_SET & 2)) ||
                            (st == kFc1Stages + 4 && (PF_SET & 4)) || (st == kFc1Stages + 12 && (PF_SET & 8)))) {
          // prefetch these stages to L2 ahead of their ring slots (PF_SET; tunable)
          const int n = st == SPEC_STAGES ? PF_N1 : st == kFc1Stages + 4 ? PF_N2 : 4;
          for (int k = st; k < st + n && k < kTotStages; ++k) prefetch_stage(k, e);
        }
        if (st >= kStages) mbar_wait(&s.empty[st % kStages], ((st / kStages) - 1) & 1);
        issue_stage(st, e, pol);
        if (STAGE_STAMPS) stamp(p, 64 + st);
      }
    }
    __syncwarp();
  } else if (warp == 2) {
    if (lane == 0) {
      // ---------------- MMA issuer (tensor pipe: MMAs only)
      const uint32_t idesc = make_idesc_nvf4(128u, 256u);
      mbar_wait(&s.x_full, 0);
      tc_fence_after();
      int n = 0;
      for (int st = 0; st < kFc1Stages; ++st) {
        const int b = st % kStages, ks = st % kHalfStages;
        if (st == kHalfStages) { mbar_wait(&s.acc_free, 0); tc_fence_after(); }
        mbar_wait(&s.full[b], (st / kStages) & 1);
        mbar_wait(&s.sf_full[b], (st / kStages) & 1);
        tc_fence_after();
        if (st == 0) stamp(p, 3);
#pragma unroll
        for (int q = 0; q < 4; ++q, ++n) {
          const int kb = ks * 4 + q;  // k-blocks in K order: one accumulator per half, the stock summation order
          umma_nvf4(tm, make_sdesc(smem_u32(s.xa + (kb / 4) * 2048) + (kb % 4) * 32, 0, 1024, 2),
                    make_sdesc(smem_u32(s.ring[b]) + q * 32, 0, 1024, 2), idesc,
                    tm + kColSfaX + (uint32_t)(kb % kSfaSlots) * 4, tm + kColSfb + (uint32_t)(n % kSfbSlots) * 8, kb > 0);
        }
        tc_commit(&s.empty[b]);
        if (ks == kHalfStages - 1) tc_commit(&s.fc1_half);
      }
      mbar_wait(&s.h_ready, 0);
      tc_fence_after();
      stamp(p, 6);
      for (int st = kFc1Stages; st < kTotStages; ++st) {
        const int b = st % kStages, j = st - kFc1Stages, jt = j % 8;
        if (j == 8) { mbar_wait(&s.fc2_free, 0); tc_fence_after(); }
        mbar_wait(&s.full[b], (st / kStages) & 1);
        mbar_wait(&s.sf_full[b], (st / kStages) & 1);
        tc_fence_after();
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          const int n2 = 128 + j * 4 + kb;  // scale-factor slot, numbered on from FC1's 128 k-blocks
          umma_nvf4(tm, make_sdesc(smem_u32(s.ha) + kb * 32, 0, 1024, 2),
                    make_sdesc(smem_u32(s.ring[b]) + kb * 32, 0, 1024, 2), idesc,
                    tm + kColSfa + (uint32_t)(jt * 4 + kb) * 4, tm + kColSfb + (uint32_t)(n2 % kSfbSlots) * 8, (jt | kb) != 0);
        }
        tc_commit(&s.empty[b]);
        if (jt == 7) tc_commit(&s.fc2_pass[j / 8]);
      }
    }
    __syncwarp();
  } else if (warp == 0 || warp == 3) {
    // ---------------- FC1 drains (warp 0: columns [0, 128); warp 4 drains [128, 256)) and the SwiGLU + NVFP4 epilogue
    const int et = (warp == 0 ? 0 : 32) + lane;  // 0..63
    // half 0: warps 0 and 3 only (the TMEM warpgroup is staging half-1 scale factors meanwhile)
    if (warp == 0) {
      mbar_wait(&s.fc1_half, 0);
      tc_fence_after();
      if (lane == 0) stamp(p, 4);
      drain_fc1(0);
      if (lane == 0) { mbar_arrive(&s.acc_free); mbar_arrive(&s.acc_ready[0]); }
    }
    mbar_wait(&s.acc_ready[0], 0);
    epi_tasks(0, et, 64);
    __syncwarp();
    if (lane == 0) mbar_arrive(&s.acc_read);
    // half 1: with the TMEM warpgroup (192 threads)
    if (warp == 0) {
      mbar_wait(&s.fc1_half, 1);
      mbar_wait(&s.acc_read, 0);
      tc_fence_after();
      if (lane == 0) stamp(p, 11);
      drain_fc1(0);
      if (lane == 0) { mbar_arrive(&s.acc_ready[1]); stamp(p, 13); }
    }
    mbar_wait(&s.acc_ready[1], 0);
    epi_tasks(1, et, 192);
    if (warp == 3 && lane == 0) stamp(p, 14);
    fence_proxy_async_smem();
    named_bar(6, 192);  // h and its scale words are complete for the TMEM warpgroup
  } else {
    // ---------------- TMEM-side warpgroup (warps 4..7): q = lane quarter (TMEM lanes 32 q .. 32 q + 31)
    const int q = warp - 4;
    const uint32_t lane_addr = (uint32_t)(q * 32) << 16;
    mbar_wait(&s.x_full, 0);
    auto stage_sf2 = [&](int st) {  // FC2 tile st - kFc1Stages: k-blocks 0..3 of row blocks 2 j, 2 j + 1
      const int b = st % kStages;
      mbar_wait(&s.full[b], (st / kStages) & 1);
      if (STAGE_STAMPS && threadIdx.x == 128) stamp(p, 16 + st);
      if (st >= 4) mbar_wait(&s.empty[(st - 4) % kStages], ((st - 4) / kStages) & 1);
      const uint8_t* sf = s.ring[b] + 32768;
      const int n0 = 128 + (st - kFc1Stages) * 4;
      for (int i = 0; i < 4; ++i) {
        const uint4 v0 = *reinterpret_cast<const uint4*>(sf + i * 512 + lane * 16);
        const uint4 v1 = *reinterpret_cast<const uint4*>(sf + 2048 + i * 512 + lane * 16);
        const uint32_t r[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        tmem_st_x8(tm + lane_addr + kColSfb + (uint32_t)((n0 + i) % kSfbSlots) * 8, r);
      }
      tmem_st_wait();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&s.sf_full[b]);
    };
    // FC1 stage st: weight scale factors of k-blocks [4 ks, 4 ks + 4) (8 columns each: row blocks 2 h | 2 h + 1) and
    // the x scale factors of the same k-blocks (row m -> lane m % 32, column m / 32 of each 4-column atom, every
    // quarter holds a copy; rows >= M, including the A rows read past the 16 real ones, get zero scales)
    for (int st = 0; st < kFc1Stages; ++st) {
      const int b = st % kStages, ks = st % kHalfStages;
      if (st == kHalfStages && q == 0) {  // half 0 is done: drain its upper 128 columns so half 1 can start
        mbar_wait(&s.fc1_half, 0);
        tc_fence_after();
        drain_fc1(2);
        if (lane == 0) { mbar_arrive(&s.acc_free); mbar_arrive(&s.acc_ready[0]); }
      }
      mbar_wait(&s.full[b], (st / kStages) & 1);
      if (STAGE_STAMPS && threadIdx.x == 128) stamp(p, 16 + st);
      if (st >= 4) mbar_wait(&s.empty[(st - 4) % kStages], ((st - 4) / kStages) & 1);
      const uint8_t* sf = s.ring[b] + 32768;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 v0 = *reinterpret_cast<const uint4*>(sf + i * 512 + lane * 16);
        const uint4 v1 = *reinterpret_cast<const uint4*>(sf + 2048 + i * 512 + lane * 16);
        const uint32_t r[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        tmem_st_x8(tm + lane_addr + kColSfb + (uint32_t)((st * 4 + i) % kSfbSlots) * 8, r);
      }
      uint32_t xr[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int kb = ks * 4 + i / 4;
        xr[i] = (i % 4 == 0 && lane < M) ? *reinterpret_cast<const uint32_t*>(&s.xsf[lane][kb * 4]) : 0u;
      }
      tmem_st_x16(tm + lane_addr + kColSfaX + (uint32_t)((ks * 4) % kSfaSlots) * 4, xr);
      tmem_st_wait();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&s.sf_full[b]);
    }
    // EARLY_SF2: stage the first FC2 scale factors before h (they wait for FC2 data to land, which delays h)
    for (int st = kFc1Stages; st < kFc1Stages + EARLY_SF2; ++st) stage_sf2(st);
    if (q == 0) {  // half 1: drain the upper 128 columns once the half-0 epilogue has read acc1
      mbar_wait(&s.fc1_half, 1);
      mbar_wait(&s.acc_read, 0);
      tc_fence_after();
      drain_fc1(2);
      if (lane == 0) mbar_arrive(&s.acc_ready[1]);
    }
    mbar_wait(&s.acc_ready[1], 0);  // phase-1 parity waits on one barrier would hang once both phases are done
    epi_tasks(1, 64 + (warp - 4) * 32 + lane, 192);
    fence_proxy_async_smem();
    named_bar(6, 192);  // h complete
    if (threadIdx.x == 128) stamp(p, 5);
    // h scale factors: 32 atoms (tile jt, k-block kb) of 4 columns; row m = 32 c + lane is nonzero only in its tile
    {
      uint32_t r[32];
#pragma unroll 1
      for (int half = 0; half < 4; ++half) {
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          const int a = half * 8 + i / 4, col = i % 4;  // atom a = 4 jt + kb
          const int jt = a / 4, kb = a % 4;
          const int row = 32 * col + lane;
          r[i] = (row / 16 == jt && row % 16 < M) ? s.hsf[row % 16][kb] : 0u;
        }
        tmem_st_x32(tm + lane_addr + kColSfa + half * 32, r);
      }
      tmem_st_wait();
    }
    tc_fence_before();
    __syncwarp();
    if (lane == 0) mbar_arrive(&s.h_ready);
    if (q == 0 && lane == 0) stamp(p, 15);
    for (int st = kFc1Stages + EARLY_SF2; st < kTotStages; ++st) {
      if (st == kFc1Stages + 8) {
        // pass 0 is fully staged; drain it once its MMAs are done, then free the accumulator for pass 1
        mbar_wait(&s.fc2_pass[0], 0);
        tc_fence_after();
        if (q == 0 && lane == 0) stamp(p, 7);
        fc2_drain16<0, 4>(p, s, tm, q, lane, 0, a2);
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&s.fc2_free);
      }
      stage_sf2(st);
    }
    mbar_wait(&s.fc2_pass[1], 0);
    tc_fence_after();
    if (q == 0 && lane == 0) stamp(p, 12);
    fc2_drain16<0, (TAIL8 ? 2 : 4)>(p, s, tm, q, lane, 1, a2);
  }
  if (TAIL8 && warp < 4) {  // the last pass's upper columns: every warp is free now (TMEM lane quarter = warp)
    mbar_wait(&s.fc2_pass[1], 0);
    tc_fence_after();
    fc2_drain16<2, 4>(p, s, tm, warp, lane, 1, HOIST_SCALES ? a2 : p.g2_alpha[e]);
  }
  tc_fence_before();
  named_bar(7, kThreads);  // every warp, including the drains, before TMEM is released
  if (threadIdx.x == 128) stamp(p, 8);
  if (warp == 2) tmem_dealloc(tm, 512);
}

inline PFN_cuTensorMapEncodeTiled_v12000 get_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    CHECK_CUDA(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", reinterpret_cast<void**>(&fn), cudaEnableDefault, &q));
    CHECK_HOST(fn != nullptr) << "qwen_mkm_moe_tp4m16: cuTensorMapEncodeTiled unavailable";
  }
  return fn;
}

inline CUtensorMap make_tmap(const void* base, uint64_t rows, uint64_t row_bytes, uint32_t box_bytes,
                             uint32_t box_rows, CUtensorMapSwizzle swz) {
  CUtensorMap m;
  cuuint64_t dims[2] = {row_bytes, rows};
  cuuint64_t strides[1] = {row_bytes};
  cuuint32_t box[2] = {box_bytes, box_rows};
  cuuint32_t estr[2] = {1, 1};
  const CUresult rc = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box,
                                   estr, CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CHECK_HOST(rc == CUDA_SUCCESS) << "qwen_mkm_moe_tp4m16: tensor map encode failed " << static_cast<int>(rc);
  return m;
}

/// \brief Launcher of the fused expert phase; one CTA per (token, top-k) slot; a slot whose expert an earlier slot took exits.
struct MkmMoeTp4M16 {
  /// Low 60 bits of the engine source md5 this module was compiled from (stale-build check).
  static int64_t source_md5_lo() { return 0xef91b74d33508edLL; }

  /// \param logits         [M, 512] bf16 router logits (router_gemm_quant output), M <= 16.
  /// \param expert_weights [M, 10] bf16 out, DynBlock softmax-over-top-10 weights.
  /// \param topk_ids       [M, 10] int32 out, DynBlock top-10 order.
  /// \param x_fp4          [M, 2048] uint8 NVFP4 activations, x_sf [M, 256] uint8 (linear e4m3 scales).
  /// \param w13            [512, 512, 2048] uint8 (trtllm-gen shuffled), w13_sf [512, 512, 256] uint8 (e4m3).
  /// \param w2             [512, 4096, 128] uint8, w2_sf [512, 4096, 16] uint8 (e4m3).
  /// \param g1_alpha       [512] fp32 gate dequant, g1_scale_c [512] up dequant times the FC2 input scale,
  ///                        g2_alpha [512] fp32.
  /// \param out            [>= M * 10, 4096] bf16 out: row t * 10 + k is expert topk_ids[t, k] on token t.
  static void run(const tvm::ffi::TensorView logits,
                  const tvm::ffi::TensorView expert_weights,
                  const tvm::ffi::TensorView topk_ids,
                  const tvm::ffi::TensorView x_fp4,
                  const tvm::ffi::TensorView x_sf,
                  const tvm::ffi::TensorView w13,
                  const tvm::ffi::TensorView w13_sf,
                  const tvm::ffi::TensorView w2,
                  const tvm::ffi::TensorView w2_sf,
                  const tvm::ffi::TensorView g1_alpha,
                  const tvm::ffi::TensorView g1_scale_c,
                  const tvm::ffi::TensorView g2_alpha,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto M_ = SymbolicSize{"num_tokens"};
    auto R_ = SymbolicSize{"out_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M_, kE}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(logits);
    TensorMatcher({M_, kTopK}).with_dtype<bf16_t>().with_device(device).verify(expert_weights);
    TensorMatcher({M_, kTopK}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({M_, kH / 2}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(x_fp4);
    TensorMatcher({M_, kH / 16}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(x_sf);
    TensorMatcher({kE, kN1, kH / 2}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w13);
    TensorMatcher({kE, kN1, kH / 16}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w13_sf);
    TensorMatcher({kE, kH, kI / 2}).with_dtype<uint8_t>().with_device(device).ensure_alignment(16).verify(w2);
    TensorMatcher({kE, kH, kI / 16}).with_dtype<uint8_t>().with_device(device).verify(w2_sf);
    // Scale-factor blocks are bulk-copied in 2 KB pieces from 512 B aligned offsets of the base.
    CHECK_HOST(reinterpret_cast<uintptr_t>(w2_sf.data_ptr()) % 16 == 0) << "qwen_mkm_moe_tp4m16: w2_sf base not 16 B aligned";
    TensorMatcher({kE}).with_dtype<fp32_t>().with_device(device).verify(g1_alpha).verify(g1_scale_c).verify(g2_alpha);
    TensorMatcher({R_, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(out);
    const auto M = static_cast<int64_t>(M_.unwrap());
    CHECK_HOST(M >= 1 && M <= kMaxTok) << "qwen_mkm_moe_tp4m16: num_tokens " << M << " outside 1.." << kMaxTok;
    CHECK_HOST(R_.unwrap() >= M * kTopK) << "qwen_mkm_moe_tp4m16: out has " << R_.unwrap() << " rows, needs " << M * kTopK;
    const CUtensorMap m13 = make_tmap(w13.data_ptr(), static_cast<uint64_t>(kE) * kN1, kH / 2, 128, kW13BoxRows,
                                      CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap m2 = make_tmap(w2.data_ptr(), static_cast<uint64_t>(kE) * kH, kI / 2, kI / 2, 256,
                                     CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap mx = make_tmap(x_fp4.data_ptr(), static_cast<uint64_t>(M), kH / 2, 128, 16,
                                     CU_TENSOR_MAP_SWIZZLE_128B);
    Params p{};
    p.logits = static_cast<const uint16_t*>(logits.data_ptr());
    p.expert_weights = static_cast<uint16_t*>(expert_weights.data_ptr());
    p.topk_ids = static_cast<int32_t*>(topk_ids.data_ptr());
    p.x_sf = static_cast<const uint8_t*>(x_sf.data_ptr());
    p.w13_sf = static_cast<const uint8_t*>(w13_sf.data_ptr());
    p.w2_sf = static_cast<const uint8_t*>(w2_sf.data_ptr());
    p.g1_alpha = static_cast<const float*>(g1_alpha.data_ptr());
    p.g1_scale_c = static_cast<const float*>(g1_scale_c.data_ptr());
    p.g2_alpha = static_cast<const float*>(g2_alpha.data_ptr());
    p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    p.trace = nullptr;
    p.num_tokens = static_cast<int>(M);
    constexpr size_t kSmem = sizeof(Smem) + 1024;
    static const bool smem_attr_set = [] {
      CHECK_CUDA(cudaFuncSetAttribute(fused_expert_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem)));
      return true;
    }();
    (void)smem_attr_set;
    // One CTA per (token, top-k) slot, no clusters.
    LaunchKernel(dim3(static_cast<unsigned>(M * kTopK)), dim3(kThreads), device.unwrap(), kSmem)
        .enable_pdl(true)(fused_expert_kernel, m13, m2, mx, p);
  }
};

}  // namespace qwen_mkm_moe_tp4m16

using qwen_mkm_moe_tp4m16::MkmMoeTp4M16;

}  // namespace sglang
