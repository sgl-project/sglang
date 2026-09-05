// Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
// (fastvideo-kernel/csrc/attention/primitives.cuh, Apache-2.0). Inference-only
// forward; the sm_103a device pass is admitted alongside sm_100a.

// primitives.cuh -- device primitives for the sm_100a VSA block-sparse attention
// forward: tcgen05 (alloc / mma / ld / st / commit / wait / fence), TMA load / store /
// tensormap, mbarrier, cluster launch control, setmaxnreg, fast math, and the FMHA helpers.
//
// Generated and pruned to what the kernel reaches -- do not edit by hand.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cassert>
#include <cstring>
#include <vector_types.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(stmt) do {                                                 \
    cudaError_t _e = (stmt);                                                  \
    if (_e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s -> %s\n",                         \
              __FILE__, __LINE__, #stmt, cudaGetErrorString(_e));             \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)
#endif

__device__ __forceinline__
uint64_t mbarrier_arrive(uint32_t mbar_smem) {
  uint64_t state;
  asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n"
               : "=l"(state) : "r"(mbar_smem) : "memory");
  return state;
}

__device__ __forceinline__
void mbarrier_arrive_nostate(uint32_t mbar_smem) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n"
               :: "r"(mbar_smem) : "memory");
}

__device__ __forceinline__
void mbarrier_arrive_cluster_default(uint32_t cluster_smem_addr) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];\n"
               :: "r"(cluster_smem_addr) : "memory");
}

__device__ __forceinline__
void mbarrier_arrive_expect_tx(uint32_t mbar_smem, uint32_t expected_bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
               :: "r"(mbar_smem), "r"(expected_bytes) : "memory");
}

__device__ __forceinline__
void mbarrier_wait_parity_suspend(uint32_t mbar_smem, uint32_t phase_parity) {
  asm volatile(
    "{\n"
    ".reg .pred P1;\n"
    "LAB_WAIT:\n"
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, 10000000;\n"
    "@!P1 bra.uni LAB_WAIT;\n"
    "}\n"
    :: "r"(mbar_smem), "r"(phase_parity) : "memory");
}

__device__ __forceinline__
void mbarrier_wait_parity(uint32_t mbar_smem, uint32_t phase_parity) {
  asm volatile(
    "{\n"
    ".reg .pred P1;\n"
    "LAB_WAIT_HOT:\n"
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
    "@!P1 bra.uni LAB_WAIT_HOT;\n"
    "}\n"
    :: "r"(mbar_smem), "r"(phase_parity) : "memory");
}

__device__ __forceinline__
void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__
void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void clc_try_cancel_async(
    uint32_t smem_dst, uint32_t mbar_smem) {
  asm volatile(
    "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128"
    " [%0], [%1];\n"
    :: "r"(smem_dst), "r"(mbar_smem) : "memory");
}

__device__ __forceinline__ void clc_load_response(
    uint32_t smem_slot, uint32_t& r0, uint32_t& r1,
    uint32_t& r2, uint32_t& r3) {
  asm volatile("ld.shared::cta.v4.b32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(smem_slot));
}

template <int NUM_STAGES>
struct MbarrierPhaseTracker {

  uint32_t phase[NUM_STAGES];
  int idx;

  __device__ __forceinline__
  void init() {
    for (int i = 0; i < NUM_STAGES; ++i) phase[i] = 0;
    idx = 0;
  }

  __device__ __forceinline__
  uint32_t current_phase() const { return phase[idx]; }

  __device__ __forceinline__
  void advance() {
    phase[idx] ^= 1u;
    idx = (idx + 1) % NUM_STAGES;
  }

  __device__ __forceinline__
  int stage() const { return idx; }
};

template <int NUM_STAGES>
struct PhaseTracker {
  int stage;
  uint32_t phase;

  __device__ __forceinline__
  PhaseTracker() : stage(0), phase(0) {}

  __device__ __forceinline__
  void advance() {
    stage++;
    if (stage == NUM_STAGES) {
      stage = 0;
      phase ^= 1;
    }
  }

  __device__ __forceinline__
  int get_stage() const { return stage; }

  __device__ __forceinline__
  uint32_t get_phase() const { return phase; }
};

template <int NUM_STAGES>
struct EmptyPhaseTracker {
  int stage;
  uint32_t phase;

  __device__ __forceinline__
  EmptyPhaseTracker() : stage(0), phase(1) {}

  __device__ __forceinline__
  void advance() {
    stage++;
    if (stage == NUM_STAGES) {
      stage = 0;
      phase ^= 1;
    }
  }

  __device__ __forceinline__
  int get_stage() const { return stage; }

  __device__ __forceinline__
  uint32_t get_phase() const { return phase; }
};

template <int STAGES>
__device__ __forceinline__
void advance_stage_phase(int& stage, uint32_t& phase) {
  ++stage;
  if (stage == STAGES) {
    stage = 0;
    phase ^= 1u;
  }
}

static constexpr uint32_t SM100_CLC_PEER_MASK = 0xFEFFFFFF;

struct ClcTileInfo {
  int  m_tile;
  int  n_tile;
  bool valid;
};

enum class ClcRasterOrder { AlongN, AlongM };

__device__ __forceinline__
void clc_arrive_expect_tx_cta(uint32_t clc_full_local_addr, uint32_t tx_bytes) {
  if ((threadIdx.x & 31) == 0) {
    mbarrier_arrive_expect_tx(clc_full_local_addr, tx_bytes);
  }
}

__device__ __forceinline__
void clc_consumer_release(uint32_t clc_empty_local_addr) {
  uint32_t peer0_addr = clc_empty_local_addr & SM100_CLC_PEER_MASK;
  mbarrier_arrive_cluster_default(peer0_addr);
}

__device__ __forceinline__
void clc_consumer_release_cta(uint32_t clc_empty_local_addr) {
  mbarrier_arrive_nostate(clc_empty_local_addr);
}

template <int CLUSTER_SHAPE_M, int CLUSTER_SHAPE_N, ClcRasterOrder ORDER>
__device__ __forceinline__
ClcTileInfo clc_parse_response(uint32_t resp_smem_addr) {
  uint32_t d0, d1, d2, d3;
  fence_proxy_async_shared_cta();
  clc_load_response(resp_smem_addr, d0, d1, d2, d3);
  const int  ctaid_x = static_cast<int>(d0);
  const int  ctaid_y = static_cast<int>(d1 & 0xFFFFu);
  const bool valid   = (d2 & 1u) != 0u;
  (void)d3;

  ClcTileInfo info;
  info.valid = valid;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    info.m_tile = ctaid_y / CLUSTER_SHAPE_M;
    info.n_tile = ctaid_x / CLUSTER_SHAPE_N;
  } else {
    info.m_tile = ctaid_x / CLUSTER_SHAPE_M;
    info.n_tile = ctaid_y / CLUSTER_SHAPE_N;
  }
  return info;
}

template <int CLUSTER_SHAPE_M, int CLUSTER_SHAPE_N, ClcRasterOrder ORDER,
          int CTA_GROUP = 2, bool SUSPEND = false>
__device__ __forceinline__
ClcTileInfo clc_fetch_next_tile(
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    int clc_cons_stage, uint32_t clc_cons_phase, bool do_release) {
  uint32_t full_addr = static_cast<uint32_t>(
      __cvta_generic_to_shared(&clc_full_bar[clc_cons_stage]));
  if constexpr (SUSPEND) mbarrier_wait_parity_suspend(full_addr, clc_cons_phase);
  else                   mbarrier_wait_parity(full_addr, clc_cons_phase);
  uint32_t resp_addr = static_cast<uint32_t>(
      __cvta_generic_to_shared(&clc_response[clc_cons_stage * 4]));
  ClcTileInfo t = clc_parse_response<
      CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(resp_addr);
  if (do_release) {
    uint32_t empty_local = static_cast<uint32_t>(
        __cvta_generic_to_shared(&clc_empty_bar[clc_cons_stage]));
    if constexpr (CTA_GROUP == 1) {
      clc_consumer_release_cta(empty_local);
    } else {
      clc_consumer_release(empty_local);
    }
  }
  return t;
}

template <int STAGES = 2>
__device__ __forceinline__
void clc_fetch_next_tile_advance(int& clc_cons_stage,
                                 uint32_t& clc_cons_phase) {
  advance_stage_phase<STAGES>(clc_cons_stage, clc_cons_phase);
}

__device__ __forceinline__ unsigned fdiv(unsigned n, unsigned long long pk) {
  unsigned M = (unsigned)pk;
  if (M == 0u) return n;
  return __umulhi(n, M) >> (unsigned)(pk >> 32);
}
__host__ inline unsigned long long make_magic(unsigned d) {
  if (d <= 1u) return 0ULL;
  unsigned l = 0; while ((1u << (l + 1)) <= d) ++l;
  unsigned p = 31u + l;
  unsigned long long m = ((1ull << p) + (unsigned long long)d - 1ull) / d;
  return (m & 0xffffffffULL) | ((unsigned long long)(p - 32u) << 32);
}

template <int BARRIER_ID>
__device__ __forceinline__
void bar_sync(uint32_t thread_count) {
  static_assert(BARRIER_ID >= 0 && BARRIER_ID <= 15,
                "bar.sync: BARRIER_ID must be in [0, 15]");
  asm volatile("bar.sync %0, %1;\n"
               :: "n"(BARRIER_ID), "r"(thread_count) : "memory");
}

template <int BARRIER_ID>
__device__ __forceinline__
void bar_arrive(uint32_t thread_count) {
  static_assert(BARRIER_ID >= 0 && BARRIER_ID <= 15,
                "bar.arrive: BARRIER_ID must be in [0, 15]");
  asm volatile("bar.arrive %0, %1;\n"
               :: "n"(BARRIER_ID), "r"(thread_count) : "memory");
}

__device__ __forceinline__ void full_bar_arrive(int m_tile, int band) {
  switch (1 + m_tile * 4 + band) {
    case 1: bar_arrive<1>(64); break;
    case 2: bar_arrive<2>(64); break;
    case 3: bar_arrive<3>(64); break;
    case 4: bar_arrive<4>(64); break;
    case 5: bar_arrive<5>(64); break;
    case 6: bar_arrive<6>(64); break;
    case 7: bar_arrive<7>(64); break;
    case 8: bar_arrive<8>(64); break;
  }
}
__device__ __forceinline__ void full_bar_wait(int m_tile, int band) {
  switch (1 + m_tile * 4 + band) {
    case 1: bar_sync<1>(64); break;
    case 2: bar_sync<2>(64); break;
    case 3: bar_sync<3>(64); break;
    case 4: bar_sync<4>(64); break;
    case 5: bar_sync<5>(64); break;
    case 6: bar_sync<6>(64); break;
    case 7: bar_sync<7>(64); break;
    case 8: bar_sync<8>(64); break;
  }
}

template <bool IS_CAUSAL, int K_TILE>
__device__ __forceinline__ void mask_s_row_r2p(float* scores, int k_offset, int q_pos, int seqlen_k) {
  int n_keep = seqlen_k - k_offset;
  if constexpr (IS_CAUSAL) {
    const int causal = q_pos - k_offset + 1;
    n_keep = n_keep < causal ? n_keep : causal;
  }
  #pragma unroll
  for (int s = 0; s < K_TILE / 32; ++s) {
    int m = (s + 1) * 32 - n_keep;
    m = m < 0 ? 0 : (m > 32 ? 32 : m);
    const uint32_t keep = (m >= 32) ? 0u : (0xFFFFFFFFu >> m);
    #pragma unroll
    for (int i = 0; i < 32; ++i)
      if (!(keep & (1u << i))) scores[s * 32 + i] = -INFINITY;
  }
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_alloc(uint32_t smem_dst_ptr,
                                              uint32_t n_cols) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2,
                "tcgen05_alloc: CTA_GROUP must be 1 or 2");
  if constexpr (CTA_GROUP == 1) {
    asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
      :: "r"(smem_dst_ptr), "r"(n_cols));
  } else {
    asm volatile(
      "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;\n"
      :: "r"(smem_dst_ptr), "r"(n_cols));
  }
}

__device__ __forceinline__ void tcgen05_st_32x32b_x16(
    uint32_t tmem_addr, const uint32_t (&r)[16]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 "
               "[%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,"
               "%11,%12,%13,%14,%15,%16};\n"
               :: "r"(tmem_addr),
                  "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),
                  "r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),
                  "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),
                  "r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]));
}

__device__ __forceinline__ void tcgen05_st_32x32b_x32(
    uint32_t tmem_addr, const uint32_t (&r)[32]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 "
               "[%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,"
               "%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,"
               "%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,"
               "%31,%32};\n"
               :: "r"(tmem_addr),
                  "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),
                  "r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),
                  "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),
                  "r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]),
                  "r"(r[16]),"r"(r[17]),"r"(r[18]),"r"(r[19]),
                  "r"(r[20]),"r"(r[21]),"r"(r[22]),"r"(r[23]),
                  "r"(r[24]),"r"(r[25]),"r"(r[26]),"r"(r[27]),
                  "r"(r[28]),"r"(r[29]),"r"(r[30]),"r"(r[31]));
}

__device__ __forceinline__ void tcgen05_commit1_lead(uint32_t lead, uint32_t mbar_smem_addr) {
  asm volatile(
    "{\n\t"
    ".reg .pred q;\n\t"
    "setp.ne.b32 q, %0, 0;\n\t"
    "@q tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%1];\n\t"
    "}\n"
    :: "r"(lead), "r"(mbar_smem_addr));
}

__device__ __forceinline__ void tcgen05_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
}

__device__ __forceinline__
void tma_load_2d(uint32_t smem_dst, const void* tensormap_ptr,
                 uint32_t mbar_smem, int coord_x, int coord_y) {
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes"
    " [%0], [%1, {%3, %4}], [%2];\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem), "r"(coord_x), "r"(coord_y)
    : "memory");
}

__device__ __forceinline__
void tma_load_3d(uint32_t smem_dst, const void* tensormap_ptr,
                 uint32_t mbar_smem, int c0, int c1, int c2) {
  asm volatile(
    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes"
    " [%0], [%1, {%3, %4, %5}], [%2];\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem), "r"(c0), "r"(c1), "r"(c2)
    : "memory");
}

__device__ __forceinline__
void tma_load_4d(uint32_t smem_dst, const void* tensormap_ptr,
                 uint32_t mbar_smem, int c0, int c1, int c2, int c3) {
  asm volatile(
    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes"
    " [%0], [%1, {%3, %4, %5, %6}], [%2];\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem), "r"(c0), "r"(c1), "r"(c2), "r"(c3)
    : "memory");
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_dealloc(uint32_t tmem_addr,
                                                uint32_t n_cols) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2,
                "tcgen05_dealloc: CTA_GROUP must be 1 or 2");
  if constexpr (CTA_GROUP == 1) {
    asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
      :: "r"(tmem_addr), "r"(n_cols));
  } else {
    asm volatile(
      "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;\n"
      :: "r"(tmem_addr), "r"(n_cols));
  }
}

__device__ __forceinline__
void tma_store_2d(const void* tensormap_ptr, int coord_x, int coord_y,
                  uint32_t smem_src) {
  asm volatile(
    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
    " [%0, {%1, %2}], [%3];\n"
    :: "l"(tensormap_ptr), "r"(coord_x), "r"(coord_y),
       "r"(smem_src)
    : "memory");
}

__device__ __forceinline__
void tma_store_3d(const void* tensormap_ptr, int c0, int c1, int c2,
                  uint32_t smem_src) {
  asm volatile(
    "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
    " [%0, {%1, %2, %3}], [%4];\n"
    :: "l"(tensormap_ptr), "r"(c0), "r"(c1), "r"(c2), "r"(smem_src)
    : "memory");
}

__device__ __forceinline__
void tma_store_4d(const void* tensormap_ptr, int c0, int c1, int c2, int c3,
                  uint32_t smem_src) {
  asm volatile(
    "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
    " [%0, {%1, %2, %3, %4}], [%5];\n"
    :: "l"(tensormap_ptr), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(smem_src)
    : "memory");
}

inline cudaError_t make_tma_2d_tiled(
    CUtensorMap* out,
    const void* ptr, int rows, int cols, int box_rows, int box_cols,
    int elem_bytes, CUtensorMapDataType dtype,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B,
    CUtensorMapL2promotion l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CUtensorMapFloatOOBfill oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) {
  uint64_t globalDim[2]     = { (uint64_t)cols, (uint64_t)rows };
  uint64_t globalStrides[1] = { (uint64_t)cols * (uint64_t)elem_bytes };
  uint32_t boxDim[2]        = { (uint32_t)box_cols, (uint32_t)box_rows };
  uint32_t elemStrides[2]   = { 1u, 1u };

  CUresult r = cuTensorMapEncodeTiled(
      out, dtype, 2,
      const_cast<void*>(ptr), globalDim, globalStrides,
      boxDim, elemStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle, l2, oob);
  return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
}

__device__ __forceinline__
void cp_async_bulk_commit_group() {
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__
void cp_async_bulk_wait_group_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__
void mbarrier_init(uint32_t mbar_smem, uint32_t arrive_count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
               :: "r"(mbar_smem), "r"(arrive_count) : "memory");
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_relinquish_alloc_permit() {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2,
                "tcgen05_relinquish_alloc_permit: CTA_GROUP must be 1 or 2");
  if constexpr (CTA_GROUP == 1) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::);
  } else {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n" ::);
  }
}

__device__ __forceinline__
void fence_mbarrier_init_release_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_mma_f16_ss_lead(uint32_t lead,
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p, q;\n\t"
    "setp.ne.b32 q, %0, 0;\n\t"
    "setp.ne.b32 p, %5, 0;\n\t"
    "@q tcgen05.mma.cta_group::1.kind::f16 [%1], %2, %3, %4, {%6, %7, %8, %9}, p;\n\t"
    "}\n"
    :: "r"(lead), "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u), "r"(0u), "r"(0u), "r"(0u), "r"(0u));
}

__device__ __forceinline__ void tcgen05_mma_f16_ts_1sm_lead(uint32_t lead,
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p, q;\n\t"
    "setp.ne.b32 q, %0, 0;\n\t"
    "setp.ne.b32 p, %5, 0;\n\t"
    "@q tcgen05.mma.cta_group::1.kind::f16 [%1], [%2], %3, %4, {%6, %7, %8, %9}, p;\n\t"
    "}\n"
    :: "r"(lead), "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u), "r"(0u), "r"(0u), "r"(0u), "r"(0u));
}

enum class SmemSwizzleBlackwell : uint32_t {
  None = 0,
  B128_32atom = 1,
  B128 = 2,
  B64  = 4,
  B32  = 6,
};

__device__ __host__ __forceinline__ uint64_t build_smem_desc_blackwell(
    uint32_t smem_addr,
    uint32_t stride_byte_offset,
    uint32_t leading_byte_offset,
    SmemSwizzleBlackwell swizzle = SmemSwizzleBlackwell::B128,
    uint32_t base_offset = 0) {
  uint64_t d = 0;
  d |= static_cast<uint64_t>((smem_addr >> 4) & 0x3FFF);
  d |= static_cast<uint64_t>((leading_byte_offset >> 4) & 0x3FFF) << 16;
  d |= static_cast<uint64_t>((stride_byte_offset >> 4) & 0x3FFF) << 32;
  d |= static_cast<uint64_t>(1) << 46;
  d |= (static_cast<uint64_t>(base_offset) & 0x7) << 49;
  d |= static_cast<uint64_t>(static_cast<uint32_t>(swizzle) & 0x7) << 61;
  return d;
}

__device__ __forceinline__
uint32_t elect_one_sync() {
  uint32_t elected;
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "elect.sync %0|p, 0xffffffff;\n\t"
    "selp.b32 %0, 1, 0, p;\n\t"
    "}\n"
    : "=r"(elected));
  return elected;
}

__device__ __forceinline__
uint32_t elect_one_sync(uint32_t membermask) {
  uint32_t elected;
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "elect.sync %0|p, %1;\n\t"
    "selp.b32 %0, 1, 0, p;\n\t"
    "}\n"
    : "=r"(elected) : "r"(membermask));
  return elected;
}

template <int N>
__device__ __forceinline__
void setmaxnreg_dec() {
  static_assert(N >= 24 && N <= 256 && N % 8 == 0,
                "setmaxnreg_dec: N must be in [24, 256] and a multiple of 8");
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n"
               :: "n"(N) : "memory");
}

template <int N>
__device__ __forceinline__
void setmaxnreg_inc() {
  static_assert(N >= 24 && N <= 256 && N % 8 == 0,
                "setmaxnreg_inc: N must be in [24, 256] and a multiple of 8");
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n"
               :: "n"(N) : "memory");
}

__device__ __forceinline__
uint32_t cvt_f32x2_to_bf16x2(float a, float b) {
  uint32_t r;
  asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;\n"
               : "=r"(r) : "f"(a), "f"(b));
  return r;
}

namespace {
__device__ __forceinline__ uint64_t f32x2_bits(float2 v) {
  uint64_t b; __builtin_memcpy(&b, &v, 8); return b;
}
__device__ __forceinline__ float2 f32x2_make(uint64_t b) {
  float2 v; __builtin_memcpy(&v, &b, 8); return v;
}
}

__device__ __forceinline__ float2 fmul2(float2 a, float2 b) {
  uint64_t d;
  asm volatile("mul.f32x2 %0, %1, %2;\n" : "=l"(d) : "l"(f32x2_bits(a)), "l"(f32x2_bits(b)));
  return f32x2_make(d);
}

__device__ __forceinline__ float2 fadd2(float2 a, float2 b) {
  uint64_t d;
  asm volatile("add.f32x2 %0, %1, %2;\n" : "=l"(d) : "l"(f32x2_bits(a)), "l"(f32x2_bits(b)));
  return f32x2_make(d);
}

__device__ __forceinline__ float2 ffma2(float2 a, float2 b, float2 c) {
  uint64_t d;
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
               : "=l"(d) : "l"(f32x2_bits(a)), "l"(f32x2_bits(b)), "l"(f32x2_bits(c)));
  return f32x2_make(d);
}

__device__ __forceinline__ float2 f32x2_splat(float s) { return make_float2(s, s); }

__device__ __forceinline__ float ex2_approx_f32(float z) {
  float d;
  asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(d) : "f"(z));
  return d;
}

__device__ __forceinline__ float2 ex2_emu_f32x2(float x, float y) {
  uint32_t ox, oy;
  asm volatile(
    "{\n\t"
    ".reg .f32 f1,f2,f3,f4,f5,f6,f7;\n\t"
    ".reg .b64 l1,l2,l3,l4,l5,l6,l7,l8,l9,l10;\n\t"
    ".reg .s32 r1,r2,r3,r4,r5,r6,r7,r8;\n\t"
    "max.f32 f1, %2, 0fC2FE0000;\n\t"
    "max.f32 f2, %3, 0fC2FE0000;\n\t"
    "mov.b64 l1, {f1, f2};\n\t"
    "mov.f32 f3, 0f4B400000;\n\t"
    "mov.b64 l2, {f3, f3};\n\t"
    "add.rm.f32x2 l7, l1, l2;\n\t"
    "sub.rn.f32x2 l8, l7, l2;\n\t"
    "sub.rn.f32x2 l9, l1, l8;\n\t"
    "mov.f32 f7, 0f3D9DF09D;\n\t"
    "mov.b64 l6, {f7, f7};\n\t"
    "mov.f32 f6, 0f3E6906A4;\n\t"
    "mov.b64 l5, {f6, f6};\n\t"
    "mov.f32 f5, 0f3F31F519;\n\t"
    "mov.b64 l4, {f5, f5};\n\t"
    "mov.f32 f4, 0f3F800000;\n\t"
    "mov.b64 l3, {f4, f4};\n\t"
    "fma.rn.f32x2 l10, l9, l6, l5;\n\t"
    "fma.rn.f32x2 l10, l10, l9, l4;\n\t"
    "fma.rn.f32x2 l10, l10, l9, l3;\n\t"
    "mov.b64 {r1, r2}, l7;\n\t"
    "mov.b64 {r3, r4}, l10;\n\t"
    "shl.b32 r5, r1, 23;\n\t"
    "add.s32 r7, r5, r3;\n\t"
    "shl.b32 r6, r2, 23;\n\t"
    "add.s32 r8, r6, r4;\n\t"
    "mov.b32 %0, r7;\n\t"
    "mov.b32 %1, r8;\n\t"
    "}\n"
    : "=r"(ox), "=r"(oy) : "f"(x), "f"(y));
  float2 r; __builtin_memcpy(&r.x, &ox, 4); __builtin_memcpy(&r.y, &oy, 4);
  return r;
}

__device__ __forceinline__ float rcp_approx_ftz_f32(float x) {
  float d;
  asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(d) : "f"(x));
  return d;
}

__device__ __forceinline__ uint32_t make_idesc_table44(
    int M, int N,
    uint32_t dtype, uint32_t atype, uint32_t btype,
    bool transpose_a = false, bool transpose_b = false,
    bool negate_a = false, bool negate_b = false) {
  uint32_t idesc = 0;
  idesc |= (dtype & 0x3) << 4;
  idesc |= (atype & 0x7) << 7;
  idesc |= (btype & 0x7) << 10;
  idesc |= (negate_a ? 1u : 0u) << 13;
  idesc |= (negate_b ? 1u : 0u) << 14;
  idesc |= (transpose_a ? 1u : 0u) << 15;
  idesc |= (transpose_b ? 1u : 0u) << 16;
  idesc |= ((static_cast<uint32_t>(N) >> 3) & 0x3F) << 17;
  idesc |= ((static_cast<uint32_t>(M) >> 4) & 0x1F) << 24;
  return idesc;
}

__device__ __forceinline__ uint32_t make_idesc_bf16_f32(
    int M, int N, bool ta = false, bool tb = false) {
  return make_idesc_table44(M, N,  1,
                              1,  1, ta, tb);
}

__device__ __forceinline__ void tcgen05_ld_32x32b_x16(
    uint32_t tmem_addr, uint32_t (&r)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,"
               "%10,%11,%12,%13,%14,%15}, [%16];\n"
               : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
                 "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                 "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
                 "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
               : "r"(tmem_addr));
}

__device__ __forceinline__ void tcgen05_ld_32x32b_x32(
    uint32_t tmem_addr, uint32_t (&r)[32]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,"
               "%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
               "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,"
               "%30,%31}, [%32];\n"
               : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
                 "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                 "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
                 "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
                 "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),
                 "=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
                 "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),
                 "=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
               : "r"(tmem_addr));
}

__device__ __forceinline__ void tcgen05_ld_32x32b_x64(
    uint32_t tmem_addr, uint32_t (&r)[64]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,"
               "%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
               "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,"
               "%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,"
               "%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,"
               "%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,"
               "%60,%61,%62,%63}, [%64];\n"
               : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
                 "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                 "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
                 "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
                 "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),
                 "=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
                 "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),
                 "=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31]),
                 "=r"(r[32]),"=r"(r[33]),"=r"(r[34]),"=r"(r[35]),
                 "=r"(r[36]),"=r"(r[37]),"=r"(r[38]),"=r"(r[39]),
                 "=r"(r[40]),"=r"(r[41]),"=r"(r[42]),"=r"(r[43]),
                 "=r"(r[44]),"=r"(r[45]),"=r"(r[46]),"=r"(r[47]),
                 "=r"(r[48]),"=r"(r[49]),"=r"(r[50]),"=r"(r[51]),
                 "=r"(r[52]),"=r"(r[53]),"=r"(r[54]),"=r"(r[55]),
                 "=r"(r[56]),"=r"(r[57]),"=r"(r[58]),"=r"(r[59]),
                 "=r"(r[60]),"=r"(r[61]),"=r"(r[62]),"=r"(r[63])
               : "r"(tmem_addr));
}

__device__ __forceinline__ void tcgen05_ld_32x32b_x128(
    uint32_t tmem_addr, uint32_t (&r)[128]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x128.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,"
               "%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"
               "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,"
               "%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,"
               "%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,"
               "%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,"
               "%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,"
               "%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,"
               "%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,"
               "%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,"
               "%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,"
               "%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,"
               "%120,%121,%122,%123,%124,%125,%126,%127}, [%128];\n"
               : "=r"(r[  0]),"=r"(r[  1]),"=r"(r[  2]),"=r"(r[  3]),
                 "=r"(r[  4]),"=r"(r[  5]),"=r"(r[  6]),"=r"(r[  7]),
                 "=r"(r[  8]),"=r"(r[  9]),"=r"(r[ 10]),"=r"(r[ 11]),
                 "=r"(r[ 12]),"=r"(r[ 13]),"=r"(r[ 14]),"=r"(r[ 15]),
                 "=r"(r[ 16]),"=r"(r[ 17]),"=r"(r[ 18]),"=r"(r[ 19]),
                 "=r"(r[ 20]),"=r"(r[ 21]),"=r"(r[ 22]),"=r"(r[ 23]),
                 "=r"(r[ 24]),"=r"(r[ 25]),"=r"(r[ 26]),"=r"(r[ 27]),
                 "=r"(r[ 28]),"=r"(r[ 29]),"=r"(r[ 30]),"=r"(r[ 31]),
                 "=r"(r[ 32]),"=r"(r[ 33]),"=r"(r[ 34]),"=r"(r[ 35]),
                 "=r"(r[ 36]),"=r"(r[ 37]),"=r"(r[ 38]),"=r"(r[ 39]),
                 "=r"(r[ 40]),"=r"(r[ 41]),"=r"(r[ 42]),"=r"(r[ 43]),
                 "=r"(r[ 44]),"=r"(r[ 45]),"=r"(r[ 46]),"=r"(r[ 47]),
                 "=r"(r[ 48]),"=r"(r[ 49]),"=r"(r[ 50]),"=r"(r[ 51]),
                 "=r"(r[ 52]),"=r"(r[ 53]),"=r"(r[ 54]),"=r"(r[ 55]),
                 "=r"(r[ 56]),"=r"(r[ 57]),"=r"(r[ 58]),"=r"(r[ 59]),
                 "=r"(r[ 60]),"=r"(r[ 61]),"=r"(r[ 62]),"=r"(r[ 63]),
                 "=r"(r[ 64]),"=r"(r[ 65]),"=r"(r[ 66]),"=r"(r[ 67]),
                 "=r"(r[ 68]),"=r"(r[ 69]),"=r"(r[ 70]),"=r"(r[ 71]),
                 "=r"(r[ 72]),"=r"(r[ 73]),"=r"(r[ 74]),"=r"(r[ 75]),
                 "=r"(r[ 76]),"=r"(r[ 77]),"=r"(r[ 78]),"=r"(r[ 79]),
                 "=r"(r[ 80]),"=r"(r[ 81]),"=r"(r[ 82]),"=r"(r[ 83]),
                 "=r"(r[ 84]),"=r"(r[ 85]),"=r"(r[ 86]),"=r"(r[ 87]),
                 "=r"(r[ 88]),"=r"(r[ 89]),"=r"(r[ 90]),"=r"(r[ 91]),
                 "=r"(r[ 92]),"=r"(r[ 93]),"=r"(r[ 94]),"=r"(r[ 95]),
                 "=r"(r[ 96]),"=r"(r[ 97]),"=r"(r[ 98]),"=r"(r[ 99]),
                 "=r"(r[100]),"=r"(r[101]),"=r"(r[102]),"=r"(r[103]),
                 "=r"(r[104]),"=r"(r[105]),"=r"(r[106]),"=r"(r[107]),
                 "=r"(r[108]),"=r"(r[109]),"=r"(r[110]),"=r"(r[111]),
                 "=r"(r[112]),"=r"(r[113]),"=r"(r[114]),"=r"(r[115]),
                 "=r"(r[116]),"=r"(r[117]),"=r"(r[118]),"=r"(r[119]),
                 "=r"(r[120]),"=r"(r[121]),"=r"(r[122]),"=r"(r[123]),
                 "=r"(r[124]),"=r"(r[125]),"=r"(r[126]),"=r"(r[127])
               : "r"(tmem_addr));
}

__device__ __forceinline__
uint32_t smem_ptr_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__
void sts_f32(uint32_t smem_addr, float val) {
  asm volatile("st.shared.f32 [%0], %1;" :: "r"(smem_addr), "f"(val) : "memory");
}

