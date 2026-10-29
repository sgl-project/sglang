// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Fused small-M NVFP4 MoE expert phase for Qwen3.5 TP4 decode verify (M <= 7 tokens, top-10, 512 experts,
// hidden 2560, intermediate 192 per rank): the top-10 of the router logits in the trtllm-gen routing's order and
// weights, FC1, SwiGLU, the NVFP4 requantization and FC2 in one launch over the trtllm-gen weight layout. Writes the
// deferred-finalize rows (row t * 10 + k, identity permutation) and the expert weights.
//
// Built with NVRTC: the kernel has C linkage (q35_mkm_moe_tp4), and the launcher's dynamic shared memory size
// (Q35_MKM_SMEM_BYTES) is checked against the layout at compile time.

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>

// NVRTC has no <cstdint>, <cuda.h> or <math.h>
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned long long uintptr_t;
typedef struct __align__(64) {
  uint64_t opaque[16];
} CUtensorMap;
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

// Minimal SM100 (tcgen05 / TMA / mbarrier) PTX helpers for the small-M NVFP4 MoE engine.

namespace qwen35 {

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

}  // namespace qwen35

namespace sglang {

namespace qwen_mkm_moe_tp4 {

using namespace qwen35;


constexpr int kH = 2560, kE = 512, kTopK = 10, kMaxTok = 7;
constexpr int kI = 192;             // intermediates per expert per rank (TP4)
constexpr int kN1 = 2 * kI;         // W13 rows per expert (gate + up)
constexpr int kStageBytes = 36864;  // 32 KB weights + 4 KB scale factors
constexpr int kStages = 5;
constexpr int kN1c = kI;            // 192 W13 rows = 96 intermediates per CTA
constexpr int kW13BoxRows = 128;    // rows of one W13 TMA box; the host tensor map must use the same
constexpr int kFc1Stages = kH / 256;      // 4 K64 blocks each (full K = 2560); two original 128-row TMA boxes
constexpr int kFc2Stages = kH / 512;       // one tile of 256 W2 rows x 128 B each
constexpr int kTotStages = kFc1Stages + kFc2Stages;
constexpr int kThreads = 256;
// TMEM columns. FC1: accumulator [0,256), x SFA ring [256,320) (16 slots x 4), SFB ring [384,512) (16 slots x 8).
// FC2: accumulator [0,256), h SFA [256,384) (32 atoms x 4), SFB ring [384,512) (16 slots x 8).
constexpr uint32_t kColSfaX = 256, kColSfa = 256, kColSfb = 384;
constexpr int kSfbSlots = 16, kSfaSlots = 16;
constexpr int kARow = 260;  // FC1 accumulator row stride in floats: 16 B aligned, rows on distinct banks
#ifndef SF_UTCCP_FC1
#define SF_UTCCP_FC1 0
#endif
#ifndef DIAG_NO_SFWAIT  // timing diagnostic only: the MMA issuer does not wait for staged scale factors
#define DIAG_NO_SFWAIT 0
#endif
#ifndef L2_EVICT_FIRST  // stream expert weights with an L2 evict-first policy (they are read once per layer)
#define L2_EVICT_FIRST 1
#endif
#ifndef TOPK_RANK  // 1: all-pairs rank top-10; 0: chains of warp max reductions (faster: 13.14 vs 13.32 us)
#define TOPK_RANK 0
#endif
#ifndef PDL_TRIGGER  // 1: let dependent grids launch once the FC2 MMAs are queued (only safe if they all wait)
#define PDL_TRIGGER 0
#endif
#ifndef HOIST_SCALES  // 1: per-expert dequant scales loaded right after routing; 0: at first use
#define HOIST_SCALES 1
#endif
#ifndef FC2SF_PRE  // 1: stage FC2 scale factors of stages 8, 9 during the partial exchange; 0: after h
#define FC2SF_PRE 1
#endif
#ifndef DIAG_NO_MMA  // timing diagnostic only: no MMAs, stages are released as soon as they land
#define DIAG_NO_MMA 0
#endif  // partial row stride in floats: 16 B aligned, rows land on distinct banks

struct Params {
  const uint16_t* logits;    // [M, 512] bf16 router logits (router_gemm_quant output)
  uint16_t* expert_weights;  // [M, 10] bf16 out: DynBlock softmax-over-top-10 weights (CTA 0 writes)
  int32_t* topk_ids;         // [M, 10] out (CTA 0 writes)
  const uint8_t* x_sf;       // [M, 160] linear e4m3
  const uint8_t* w13_sf;     // [E, 3 row blocks x 40 K-scale atoms x 512 B] (128x4 interleaved)
  const uint8_t* w2_sf;      // [E, 20 row blocks x 3 K-scale atoms x 512 B]
  const float* g1_alpha;     // [E] gate dequant
  const float* g1_scale_c;   // [E] up dequant * FC2 input quant scale
  const float* g2_alpha;     // [E]
  __nv_bfloat16* out;        // [M * 10, 2560]
  long long* trace;          // optional [grid, 16] %globaltimer stamps
  int num_tokens;
};

struct __align__(1024) Smem {
  uint8_t ring[kStages][kStageBytes];
  alignas(1024) uint8_t xa[16 * 1024];  // x operand: 8 rows x full K (2048 B), SW128 (16 atoms of 8 rows x 128 B)
  float acc1[8 * kARow];        // FC1 accumulator rows 0..7 (tokens) x 256 accumulator columns
  alignas(1024) uint8_t ha[1024];  // h operand: 8 rows x 128 B (K = 256), SW128 (atom must be 1024 B aligned)
  uint8_t xsf[8][256];          // x scale bytes, full K (row t = token t), by bulk copy
  uint32_t hsf[8][4];           // h scale words: token t, k-block kb (4 e4m3 bytes)
  uint64_t full[kStages], empty[kStages], sf_full[kStages];
  uint64_t x_full, fc1_done, h_peer, h_ready, fc2_done, route_done;
  uint32_t tmem;
  int expert;
  uint32_t tokmask;
  int kslot[8];
  alignas(16) uint32_t cand[8][32];  // read as uint4 by the rank-based top-k
  int32_t ids[8][kTopK];
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
__device__ __forceinline__ void named_bar(int id, int n) {
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
  if (p.trace) p.trace[blockIdx.x * 16 + slot] = (long long)gtime();
}
// stamp after `dep` is available (the branch needs its value; keys are never 0)
__device__ __forceinline__ void stamp_dep(const Params& p, int slot, uint32_t dep) {
  if (dep == 0u) __trap();
  stamp(p, slot);
}
// Physical W13 column (accumulator column) holding reordered row q: block-32 shuffle of FlashInfer's
// get_shuffle_matrix_a_row_indices (epilogue_tile_m = 128); reordered row 2j is up_j, 2j + 1 is gate_j.
__device__ __forceinline__ int w13_col(int q) { return (q & ~31) + ((q & 31) % 4) * 8 + (q & 31) / 4; }
__host__ __device__ constexpr int w13_logical(int p) {
  return ((p & ~31) + (p % 8) * 4 + (p & 31) / 8) % 2 ? ((p & ~31) + (p % 8) * 4 + (p & 31) / 8) / 2
                                                     : 128 + ((p & ~31) + (p % 8) * 4 + (p & 31) / 8) / 2;
}

// FC2 drain of TMEM lane quarter q (tiles 4 q .. 4 q + 3), column half `half` (128 accumulator columns)
__device__ __forceinline__ void fc2_drain(const Params& p, Smem& s, uint32_t tm, int q, int lane, int half, float a2,
                                          uint32_t rank, int e) {
  mbar_wait(&s.fc2_done, 0);
  if (!HOIST_SCALES) a2 = p.g2_alpha[e];
  tc_fence_after();
  if (q == 0 && half == 0 && lane == 0) stamp(p, 7);
  const int row = q * 32 + lane, tile = row / 8, t = row % 8;
  const bool valid = tile < kFc2Stages && t < p.num_tokens && ((s.tokmask >> t) & 1u);
  __nv_bfloat16* orow = p.out + (size_t)(t * kTopK + s.kslot[t]) * kH + rank * (kH / 2) + tile * 256 + half * 128;
#pragma unroll 1
  for (int c = 0; c < 2; ++c) {
    uint32_t r[64];
    tmem_ld_x64(tm + ((uint32_t)(q * 32) << 16) + (uint32_t)(half * 128 + c * 64), r);
    tmem_ld_wait();
    if (valid) {
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
  }
}

extern "C" __global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(kThreads, 1)
q35_mkm_moe_tp4(const __grid_constant__ CUtensorMap tm13, const __grid_constant__ CUtensorMap tm2,
          const __grid_constant__ CUtensorMap tmx, const Params p) {
  extern __shared__ __align__(1024) uint8_t raw[];
  Smem& s = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(raw) + 1023) & ~uintptr_t(1023));
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t rank = cluster_rank();
  const int pair = blockIdx.x / 2;
  if (threadIdx.x == 0) {
    stamp(p, 0);
    for (int i = 0; i < kStages; ++i) {
      mbar_init(&s.full[i], 1);
      mbar_init(&s.empty[i], 1);
      mbar_init(&s.sf_full[i], 4);
    }
    mbar_init(&s.x_full, 1);
    mbar_init(&s.fc1_done, 1);
    mbar_init(&s.h_peer, 1);
    mbar_init(&s.h_ready, 4);
    mbar_init(&s.fc2_done, 1);
    mbar_init(&s.route_done, 1);
    mbar_expect_tx(&s.h_peer, 24 * 16 + 16 * 4);  // peer h: 8 rows x 48 B; SF: 8 rows x 8 B
    fence_mbar_init();
  }
  if (warp == 2) { tmem_alloc(&s.tmem, 512); tmem_relinquish(); tc_fence_before(); }
  if (warp == 1 && lane == 0) { prefetch_tmap(&tm13); prefetch_tmap(&tm2); prefetch_tmap(&tmx); }
  __syncthreads();
  cluster_arrive();  // waited on before the first DSMEM transfer and at exit
  tc_fence_after();
  const uint32_t tm = s.tmem;
  // cluster `pair` serves (token my_t, top-k slot my_k) unless an earlier pair (in t * 10 + k order) has its expert
  const int my_t = pair / kTopK, my_k = pair % kTopK;
  if (my_t >= p.num_tokens) {  // both CTAs of the pair agree; nothing was issued
    cluster_wait();
    if (warp == 2) tmem_dealloc(tm, 512);
    return;
  }
  auto issue_stage = [&](int st, int e, uint64_t pol) {
    const int b = st % kStages;
    // Shape-only tail: FC2 has three K64 SF atoms, while the TMA box
    // still fills the original 256x128B SW128 shared-memory footprint.
    mbar_expect_tx(&s.full[b], st < kFc1Stages ? kStageBytes : 32768 + 3072);
    uint8_t* dst = s.ring[b];
    if (st < kFc1Stages) {
      // Useful rows [192r,192r+192), in the original two 128-row boxes.
      // The last 64 loaded rows are unused by the N192 MMA.
      tma_load_2d_w(dst, &tm13, &s.full[b], st * 128, e * kN1 + (int)rank * kN1c, pol);
      tma_load_2d_w(dst + 16384, &tm13, &s.full[b], st * 128, e * kN1 + (int)rank * kN1c + 128, pol);
      // Native SF row atoms r and r+1, K-scale atoms [4st,4st+4).
      const uint8_t* sf = p.w13_sf + (size_t)e * (kN1 * kH / 16);
      bulk_load_w(dst + 32768, sf + (size_t)((int)rank * (kH / 64) + st * 4) * 512, 2048, &s.full[b], pol);
      bulk_load_w(dst + 32768 + 2048, sf + (size_t)(((int)rank + 1) * (kH / 64) + st * 4) * 512, 2048, &s.full[b], pol);
    } else {
      const int j = st - kFc1Stages;
      const int row0 = (int)rank * (kH / 2) + j * 256;
      tma_load_2d_w(dst, &tm2, &s.full[b], 0, e * kH + row0, pol);
      bulk_load_w(dst + 32768, p.w2_sf + (size_t)e * (kH * kI / 16) + (size_t)(row0 / 128) * 3 * 512, 3072, &s.full[b], pol);
    }
  };
  // routing: warps 0, 2..7 take tokens 0..6 (DynBlock-exact top-10 + softmax); warp 1 recomputes token my_t's top-10
  // and starts this pair's weight stream at once (a duplicate pair only wastes its first ring fill, mostly L2 hits)
  const int rtok = warp == 1 ? my_t : (warp == 0 ? 0 : warp - 1);
  uint32_t key = 0;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (threadIdx.x == 32) stamp(p, 10);
  if (warp == 3 && lane == 0) {
    // Full x (10 atoms, rows >= M zero-filled) and SF; independent of routing
    mbar_expect_tx(&s.x_full, (kH / 256) * 1024 + p.num_tokens * (kH / 16));
#pragma unroll
    for (int a = 0; a < kH / 256; ++a) tma_load_2d(s.xa + a * 1024, &tmx, &s.x_full, a * 128, 0);
    for (int t = 0; t < p.num_tokens; ++t) bulk_load(s.xsf[t], p.x_sf + (size_t)t * (kH / 16), kH / 16, &s.x_full);
  }
  if (rtok < p.num_tokens) {
    const uint4* src = reinterpret_cast<const uint4*>(p.logits + (size_t)rtok * kE + lane * 16);
    const uint4 v0 = src[0], v1 = src[1];
    uint32_t keys[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t w = i < 4 ? (&v0.x)[i] : (&v1.x)[i - 4];
      keys[2 * i] = topk_key(w & 0xFFFFu, (uint32_t)(lane * 16 + 2 * i));
      keys[2 * i + 1] = topk_key(w >> 16, (uint32_t)(lane * 16 + 2 * i + 1));
    }
    if (threadIdx.x == 32) stamp_dep(p, 11, keys[0]);
    key = TOPK_RANK ? warp_top10(keys, (uint32_t)lane, s.cand[warp]) : warp_top10_redux(keys, (uint32_t)lane, s.cand[warp]);
  }
  int e;
  if (warp == 1) {
    uint32_t bits, espec;
    topk_unkey(__shfl_sync(0xffffffffu, key, my_k), bits, espec);
    if (lane == 0) {
      stamp(p, 9);
      const uint64_t pol = policy_evict_first();
      for (int st = 0; st < kStages; ++st) issue_stage(st, (int)espec, pol);
      stamp(p, 2);
    }
    mbar_wait(&s.route_done, 0);
    e = s.expert;
  } else {
    if (rtok < p.num_tokens) {
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
    }
    named_bar(4, 224);
    if (warp == 0) {
      // this pair's expert, whether an earlier pair picked it, and which tokens use it (top-k slot per token)
      const int ee = s.ids[my_t][my_k];
      bool dup = false;
      for (int q = lane; q < pair; q += 32) dup |= s.ids[q / kTopK][q % kTopK] == ee;
      dup = __any_sync(0xffffffffu, dup);
      uint32_t mbit = 0;
      int ks = 0;
      if (lane < p.num_tokens) {
#pragma unroll
        for (int k = 0; k < kTopK; ++k)
          if (s.ids[lane][k] == ee) { mbit = 1u << lane; ks = k; }
      }
      const uint32_t mask = __reduce_or_sync(0xffffffffu, mbit);
      if (lane < 8) s.kslot[lane] = ks;
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
  if (e < 0) {  // duplicate pair (both CTAs agree): let the speculative loads land; nothing was sent over DSMEM
    if (warp == 1) {
      if (lane == 0)
        for (int b = 0; b < kStages; ++b) mbar_wait(&s.full[b], 0);
      __syncwarp();
    }
    mbar_wait(&s.x_full, 0);
    cluster_wait();
    if (warp == 2) tmem_dealloc(tm, 512);
    return;
  }
  tc_fence_after();
  // per-expert dequant scales, loaded now so that no global read sits behind the FC1 -> h -> FC2 chain
  float ag = 0.f, sc = 0.f, a2 = 0.f;
  if (HOIST_SCALES) { ag = p.g1_alpha[e]; sc = p.g1_scale_c[e]; a2 = p.g2_alpha[e]; }

  if (warp == 1) {
    if (lane == 0) {
      // ---------------- TMA producer: the rest of the weight stream (stages 0 .. kStages - 1 went out above)
      const uint64_t pol = policy_evict_first();
      for (int st = kStages; st < kTotStages; ++st) {
        mbar_wait(&s.empty[st % kStages], ((st / kStages) - 1) & 1);
        issue_stage(st, e, pol);
      }
    }
    __syncwarp();
  } else if (warp == 2) {
    if (lane == 0) {
      // ---------------- MMA issuer (tensor pipe: MMAs only)
      const uint32_t idesc = make_idesc_nvf4(128u, kN1c);
      const uint32_t idesc2 = make_idesc_nvf4(128u, 256u);
      mbar_wait(&s.x_full, 0);
      tc_fence_after();
      int n = 0;
      for (int st = 0; st < kFc1Stages; ++st) {
        const int b = st % kStages;
        mbar_wait(&s.full[b], (st / kStages) & 1);
        if (!SF_UTCCP_FC1 && !DIAG_NO_SFWAIT) mbar_wait(&s.sf_full[b], (st / kStages) & 1);
        tc_fence_after();
        if (st == 0) stamp(p, 3);
        if (DIAG_NO_MMA) { n += 4; tc_commit(&s.empty[b]); continue; }
#pragma unroll
        for (int q = 0; q < 4; ++q, ++n) {
          const int kb = st * 4 + q;  // k-blocks in K order: one accumulator, the stock summation order
          umma_nvf4(tm, make_sdesc(smem_u32(s.xa + (kb / 4) * 1024) + (kb % 4) * 32, 0, 0, 2),
                    make_sdesc(smem_u32(s.ring[b]) + q * 32, 0, 1024, 2), idesc,
                    tm + kColSfaX + (uint32_t)(kb % kSfaSlots) * 4, tm + kColSfb + (uint32_t)(n % kSfbSlots) * 8, kb > 0);
        }
        tc_commit(&s.empty[b]);
      }
      tc_commit(&s.fc1_done);
      mbar_wait(&s.h_ready, 0);
      tc_fence_after();
      stamp(p, 6);
      for (int st = kFc1Stages; st < kTotStages; ++st) {
        const int b = st % kStages;
        mbar_wait(&s.full[b], (st / kStages) & 1);
        if (!DIAG_NO_SFWAIT) mbar_wait(&s.sf_full[b], (st / kStages) & 1);
        tc_fence_after();
        if (DIAG_NO_MMA) { tc_commit(&s.empty[b]); continue; }
        const int j = st - kFc1Stages;
#pragma unroll
        for (int kb = 0; kb < kI / 64; ++kb) {
          const int n2 = kFc1Stages * 4 + j * (kI / 64) + kb;  // SF ring slots continue after FC1's 40 K64 blocks
          umma_nvf4(tm, make_sdesc(smem_u32(s.ha) + kb * 32, 0, 0, 2),
                    make_sdesc(smem_u32(s.ring[b]) + kb * 32, 0, 1024, 2), idesc2,
                    tm + kColSfa + (uint32_t)(j * (kI / 64) + kb) * 4, tm + kColSfb + (uint32_t)(n2 % kSfbSlots) * 8, (j | kb) != 0);
        }
        tc_commit(&s.empty[b]);
      }
      tc_commit(&s.fc2_done);
      if (PDL_TRIGGER) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    __syncwarp();
  } else if (warp >= 4) {
    // ---------------- TMEM-side warpgroup: q = lane quarter (TMEM lanes 32 q .. 32 q + 31)
    const int q = warp - 4;
    const uint32_t lane_addr = (uint32_t)(q * 32) << 16;
    mbar_wait(&s.x_full, 0);
    // weight scale factors of every stage: k-block slot (8 columns) = [atom rows 0..127 | atom rows 128..255]
    auto stage_sf = [&](int st, int n0, int nkb, int atom_stride, int half_off) {
      const int b = st % kStages;
      mbar_wait(&s.full[b], (st / kStages) & 1);
      // the slots about to be overwritten were last read by the MMAs of stage st - 4
      if (st >= 4) mbar_wait(&s.empty[(st - 4) % kStages], ((st - 4) / kStages) & 1);
      const uint8_t* sf = s.ring[b] + 32768;
      for (int i = 0; i < nkb; ++i) {
        const int tl = i / 2, kb = i % 2;
        const uint8_t* a0 = atom_stride ? sf + tl * 2048 + kb * 512 : sf + i * 512;
        const uint8_t* a1 = a0 + half_off;
        const uint4 v0 = *reinterpret_cast<const uint4*>(a0 + lane * 16);
        const uint4 v1 = *reinterpret_cast<const uint4*>(a1 + lane * 16);
        const uint32_t r[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        tmem_st_x8(tm + lane_addr + kColSfb + (uint32_t)((n0 + i) % kSfbSlots) * 8, r);
      }
      tmem_st_wait();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&s.sf_full[b]);
    };
    // FC1 stage st: weight scale factors of k-blocks [4 st, 4 st + 4) (8 columns each: row blocks 2 r | 2 r + 1) and
    // the x scale factors of the same k-blocks (row m -> lane m % 32, column m / 32; every quarter holds a copy)
    for (int st = 0; st < kFc1Stages; ++st) {
      const int b = st % kStages;
      mbar_wait(&s.full[b], (st / kStages) & 1);
      // the slots about to be overwritten were last read by the MMAs of stage st - 4
      if (st >= 4) mbar_wait(&s.empty[(st - 4) % kStages], ((st - 4) / kStages) & 1);
      const uint8_t* sf = s.ring[b] + 32768;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        // A 96-intermediate half starts at physical W13 row 192*rank.
        // Rank 1 therefore starts halfway through a 128-row SF atom. Gather
        // each local SF row from the two native atoms; rows >=192 are padding.
        uint32_t r[8];
#pragma unroll
        for (int w = 0; w < 8; ++w) {
          const int local_row = lane + 32 * w;
          const int native_row = (int)rank * kN1c + local_row;
          const int atom = native_row / 128 - (int)rank;
          const int offset = atom * 2048 + i * 512 + (native_row % 32) * 16 + ((native_row % 128) / 32) * 4;
          r[w] = local_row < kN1c ? *reinterpret_cast<const uint32_t*>(sf + offset) : 0u;
        }
        tmem_st_x8(tm + lane_addr + kColSfb + (uint32_t)((st * 4 + i) % kSfbSlots) * 8, r);
      }
      uint32_t xr[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int kb = st * 4 + i / 4;
        xr[i] = (i % 4 == 0 && lane < p.num_tokens) ? *reinterpret_cast<const uint32_t*>(&s.xsf[lane][kb * 4]) : 0u;
      }
      tmem_st_x16(tm + lane_addr + kColSfaX + (uint32_t)((st * 4) % kSfaSlots) * 4, xr);
      tmem_st_wait();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&s.sf_full[b]);
    }

    // FC1 accumulator: TMEM lanes 0..7 (quarter 0) hold the token rows; quarter 0 drains all 256 columns
    mbar_wait(&s.fc1_done, 0);
    tc_fence_after();
    if (q == 0) {
      if (lane == 0) stamp(p, 4);
#pragma unroll 1
      for (int c = 0; c < 4; ++c) {
        uint32_t r[64];
        tmem_ld_x64(tm + (uint32_t)(c * 64), r);
        tmem_ld_wait();
        if (lane < 8) {
#pragma unroll
          for (int i = 0; i < 16; ++i)
            *reinterpret_cast<uint4*>(s.acc1 + lane * kARow + c * 64 + 4 * i) =
                make_uint4(r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]);
        }
      }
    }
    // FC2 scale factors of the first two FC2 stages while h is computed (all FC1 MMAs are done)
    if (FC2SF_PRE) {
      stage_sf(kFc1Stages, kFc1Stages * 4, 3, 0, 1536);
      stage_sf(kFc1Stages + 1, kFc1Stages * 4 + 3, 3, 0, 1536);
    }
    if (threadIdx.x == 128) stamp(p, 15);
    tc_fence_before();
    named_bar(2, 128);  // quarter 0's accumulator rows are visible to the whole warpgroup

    if (threadIdx.x == 128) stamp(p, 5);
    if (!HOIST_SCALES) { ag = p.g1_alpha[e]; sc = p.g1_scale_c[e]; }
    {
      // this CTA's 96 intermediates: (token row, 16-wide block, half). Accumulator columns of each 32-block:
      // [0, 8) up 2m, [8, 16) gate 2m, [16, 24) up 2m + 1, [24, 32) gate 2m + 1 (m = 0..7 within the 16 indices)
      const int it = threadIdx.x - 128;  // 0..127
      const int row = it / 12, blk = (it / 2) % 6, hf = it % 2;
      if (row < 8) {  // 8 rows x 96 intermediates; remaining warp stays in every original barrier
      const float4* ar = reinterpret_cast<const float4*>(s.acc1 + row * kARow + 32 * blk + 4 * hf);
      const float4 u_even = ar[0], g_even = ar[2], u_odd = ar[4], g_odd = ar[6];
      const float ga[8] = {g_even.x, g_odd.x, g_even.y, g_odd.y, g_even.z, g_odd.z, g_even.w, g_odd.w};
      const float ua[8] = {u_even.x, u_odd.x, u_even.y, u_odd.y, u_even.z, u_odd.z, u_even.w, u_odd.w};
      // SwiGLU + NVFP4 epilogue, bitwise equal to the reference MoE path; a = FC1 dequant,
      // c = a x FC2 input scale:
      // h' = (u a)(g sigmoid(g a)), sigmoid = rcp(ex2(-(g (a log2e))) + 1), block scale e4m3(max|h'| c / 6), FP4 (h' / scale) c
      const float a_l2e = mul_ftz(ag, 1.4426950216293334961f);
      float h[8];
      float amax = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float e = ex2_approx(-(ga[i] * a_l2e));
        const float sg = rcp_approx(e + 1.0f);
        const float v = fmaf(ua[i], ag, 0.0f) * (ga[i] * sg);
        h[i] = v;
        amax = fmaxf(amax, fabsf(v));
      }
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
      const uint32_t sf = cvt_e4m3(fmaxf(0.f, fabsf(amax * sc)) * 0.16666667163372039795f);
      const float sff = e4m3_to_f32(sf);
      const float osc = sff == 0.f ? 0.f : rcp_approx(sff);
      float qv[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) qv[i] = mul_ftz(h[i], osc) * sc;
      const uint32_t packed = cvt_e2m1x8(qv);
      const int gblk = 6 * (int)rank + blk;  // 16-wide block over the 192 intermediates
      const int chunk = gblk / 2;
      *reinterpret_cast<uint32_t*>(s.ha + row * 128 + ((chunk ^ row) * 16) + (gblk % 2) * 8 + hf * 4) = packed;
      // Each CTA owns an aligned 8-byte metadata half containing six SF
      // bytes. This avoids two CTAs writing halves of the same uint32 word.
      if (hf == 0) reinterpret_cast<uint8_t*>(&s.hsf[row][2 * rank])[blk] = (uint8_t)sf;
      }
    }
    named_bar(1, 128);
    if (q == 0) {
      // Send the 96-wide h half (3 chunks/row) and its aligned SF half to the peer.
      cluster_wait();  // phase 1: the peer's barriers are initialized
      const uint32_t pbar = mapa(smem_u32(&s.h_peer), rank ^ 1u);
      {
        if (lane < 24) {  // eight token rows x three 16-byte chunks per 96-wide half
          const int row = lane / 3, chunk = 3 * (int)rank + lane % 3;
          const uint32_t off = (uint32_t)(row * 128 + ((chunk ^ row) * 16));
          st_async_v4(mapa(smem_u32(s.ha + off), rank ^ 1u), *reinterpret_cast<const uint4*>(s.ha + off), pbar);
        }
      }
      if (lane < 16) {
        const int row = lane / 2, w = 2 * (int)rank + lane % 2;
        st_async_b32(mapa(smem_u32(&s.hsf[row][w]), rank ^ 1u), s.hsf[row][w], pbar);
      }
      __syncwarp();
    }
    mbar_wait(&s.h_peer, 0);  // the peer's half of h and its scale words
    if (threadIdx.x == 128) stamp(p, 5);
    // h SF: 5 output tiles x 3 K64 atoms. Retain the original TMEM staging
    // footprint, zeroing the unused atoms; each live tile owns eight TMEM rows.
    {
      uint32_t r[32];
#pragma unroll 1
      for (int half = 0; half < 4; ++half) {
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          const int a = half * 8 + i / 4, col = i % 4;  // atom a = 3 jt + kb
          const int jt = a / 3, kb = a % 3;
          const int row = 32 * col + lane;
          uint32_t sfword = 0;
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            const int group = 4 * kb + b;
            const auto* half_sf = reinterpret_cast<const uint8_t*>(&s.hsf[row % 8][2 * (group / 6)]);
            sfword |= (uint32_t)half_sf[group % 6] << (8 * b);
          }
          r[i] = (jt < kFc2Stages && row / 8 == jt) ? sfword : 0u;
        }
        tmem_st_x32(tm + lane_addr + kColSfa + half * 32, r);
      }
      tmem_st_wait();
    }
    fence_proxy_async_smem();
    tc_fence_before();
    __syncwarp();
    if (lane == 0) mbar_arrive(&s.h_ready);
    // FC2 weight scale factors (the MMA issuer waits for each stage's)
    for (int st = kFc1Stages + (FC2SF_PRE ? 2 : 0); st < kTotStages; ++st)
      stage_sf(st, kFc1Stages * 4 + (st - kFc1Stages) * 3, 3, 0, 1536);

    if (q < 2) fc2_drain(p, s, tm, q, lane, 0, a2, rank, e);
  }
  if (warp < 2) fc2_drain(p, s, tm, warp, lane, 1, a2, rank, e);
  tc_fence_before();
  __syncthreads();
  if (threadIdx.x == 0) stamp(p, 13);
  // every warp but warp 4 still owes its wait on the first cluster phase; phase 2 then ends the DSMEM exchange
  if (warp != 4) cluster_wait();
  cluster_arrive_relaxed();
  cluster_wait();
  if (threadIdx.x == 0) stamp(p, 8);
  if (warp == 2) tmem_dealloc(tm, 512);
}

static_assert(sizeof(Smem) + 1024 == Q35_MKM_SMEM_BYTES, "the launcher's dynamic shared memory size");

}  // namespace qwen_mkm_moe_tp4

}  // namespace sglang
