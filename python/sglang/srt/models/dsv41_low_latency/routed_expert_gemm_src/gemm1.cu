// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// DeepSeek-V4.1 low-batch routed-expert GEMM1: h = MXFP8(clamped SwiGLU(x
// W13^T)) per routed expert, 128 x 8 tiles, one 8-row token tile per expert.
// Machine-generated CUDA for SM100-family GPUs (tcgen05 block-scaled MMA, TMEM,
// cluster launch control). Launch contract, operands and numerics:
// ../routed_expert_gemm.py and launch.json in this directory.

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "rx requires an LP64 CUDA host ABI");
typedef signed int int32_t;
typedef short int int16_t;
struct __align__(128) kxTensorMap {
  uint64_t opaque[16];
};
struct __align__(64) kxTensorMap64 {
  uint64_t opaque[16];
};
static_assert(sizeof(kxTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(kxTensorMap64) == 64,
              "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) {
  uint64_t opaque[16];
} CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128,
              "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(kxTensorMap) >= alignof(CUtensorMap),
              "kxTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define kx_INF CUDART_INF_F
#define TMEM_NCOLS 88
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SFA_OFFSET 16
#define TMEM_SFB_OFFSET 64
#define NUM_K_PIPE_STAGES 3
#define NUM_MMA_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define NUM_DRAIN_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 65536
#define SMEM_SMEM_A_STRIDE 65536
#define SMEM_SMEM_B_OFF 197632
#define SMEM_SMEM_B_STAGE_BYTES 4096
#define SMEM_SMEM_B_STRIDE 4096
#define SMEM_SMEM_SFA_OFF 209920
#define SMEM_SMEM_SFA_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_STRIDE 2048
#define SMEM_SMEM_SFB_OFF 216064
#define SMEM_SMEM_SFB_STAGE_BYTES 128
#define SMEM_SMEM_SFB_STRIDE 128
#define SMEM_AMAX_X_OFF 216448
#define SMEM_AMAX_X_STAGE_BYTES 256
#define SMEM_AMAX_X_STRIDE 256
#define SMEM_WORK_RESPONSE_OFF 216704
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_DRAIN_RESPONSE_OFF 216752
#define SMEM_DRAIN_RESPONSE_STAGE_BYTES 64
#define SMEM_DRAIN_RESPONSE_STRIDE 64
#define SMEM_TOTAL 216832
#define THREADS 512

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile("{\n\t"
               ".reg .pred %%px;\n\t"
               "elect.sync _|%%px, %1;\n\t"
               "@%%px mov.s32 %0, 1;\n\t"
               "}\n"
               : "+r"(pred)
               : "r"(0xFFFFFFFF));
  return pred;
}

__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar_addr),
               "r"(count)
               : "memory");
}

__device__ __forceinline__ void mbarrier_init_generic(void *mbar_addr,
                                                      int count) {
  asm volatile("mbarrier.init.b64 [%0], %1;" ::"l"(mbar_addr), "r"(count)
               : "memory");
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_plain(int mbar_addr,
                                                            int phase) {
  uint32_t token;
  asm volatile("{\n\t"
               ".reg .pred P1;\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
               "selp.u32 %0, 1, 0, P1;\n\t"
               "}\n"
               : "=r"(token)
               : "r"(mbar_addr), "r"(phase)
               : "memory");
  return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr,
                                                      int phase) {
  uint32_t token;
  asm volatile("{\n\t"
               ".reg .pred P1;\n\t"
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
               " P1, [%1], %2;\n\t"
               "selp.u32 %0, 1, 0, P1;\n\t"
               "}\n"
               : "=r"(token)
               : "r"(mbar_addr), "r"(phase)
               : "memory");
  return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr,
                                                              int phase) {
  uint32_t token;
  asm volatile("{\n\t"
               ".reg .pred P1;\n\t"
               "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
               " P1, [%1], %2;\n\t"
               "selp.u32 %0, 1, 0, P1;\n\t"
               "}\n"
               : "=r"(token)
               : "r"(mbar_addr), "r"(phase)
               : "memory");
  return token;
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
  asm volatile("{\n\t"
               ".reg .pred P1;\n\t"
               "LAB_WAIT:\n\t"
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
               " P1, [%0], %1;\n\t"
               "@P1 bra.uni DONE;\n\t"
               "bra.uni LAB_WAIT;\n\t"
               "DONE:\n\t"
               "}\n" ::"r"(mbar_addr),
               "r"(phase)
               : "memory");
}

__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr,
                                                      int phase) {
  asm volatile("{\n\t"
               ".reg .pred P1;\n\t"
               "LAB_WAIT_RELAXED:\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64"
               " P1, [%0], %1, 10000000;\n\t"
               "@P1 bra.uni DONE_RELAXED;\n\t"
               "bra.uni LAB_WAIT_RELAXED;\n\t"
               "DONE_RELAXED:\n\t"
               "}\n" ::"r"(mbar_addr),
               "r"(phase)
               : "memory");
}

__device__ __forceinline__ void
mbarrier_wait_suspend(int mbar_addr, int phase, uint32_t suspend_time_hint) {
  asm volatile("{\n\t"
               ".reg .pred P1;\n\t"
               "LAB_WAIT_SUSPEND:\n\t"
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
               " P1, [%0], %1, %2;\n\t"
               "@P1 bra.uni DONE_SUSPEND;\n\t"
               "bra.uni LAB_WAIT_SUSPEND;\n\t"
               "DONE_SUSPEND:\n\t"
               "}\n" ::"r"(mbar_addr),
               "r"(phase), "r"(suspend_time_hint)
               : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr,
                                                      int phase) {
  asm volatile("{\n\t"
               ".reg .pred P1;\n\t"
               "LAB_WAIT_CLUSTER:\n\t"
               "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
               " P1, [%0], %1;\n\t"
               "@P1 bra.uni DONE_CLUSTER;\n\t"
               "bra.uni LAB_WAIT_CLUSTER;\n\t"
               "DONE_CLUSTER:\n\t"
               "}\n" ::"r"(mbar_addr),
               "r"(phase)
               : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(int mbar_addr, int phase,
                                                   uint32_t suspend_time_hint) {
  asm volatile("{\n\t"
               ".reg .pred P1;\n\t"
               ".reg .u32 WAIT_ADDR;\n\t"
               "mov.u32 WAIT_ADDR, %0;\n\t"
               "LAB_WAIT_HINT:\n\t"
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
               " P1, [WAIT_ADDR], %1, %2;\n\t"
               "@P1 bra.uni DONE_HINT;\n\t"
               "bra.uni LAB_WAIT_HINT;\n\t"
               "DONE_HINT:\n\t"
               "}\n" ::"r"(mbar_addr),
               "r"(phase), "r"(suspend_time_hint)
               : "memory");
}

__device__ __forceinline__ void
mbarrier_wait_relaxed_hint(int mbar_addr, int phase,
                           uint32_t suspend_time_hint) {
  asm volatile("{\n\t"
               ".reg .pred P1;\n\t"
               "LAB_WAIT_RELAXED_HINT:\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64"
               " P1, [%0], %1, %2;\n\t"
               "@P1 bra DONE_RELAXED_HINT;\n\t"
               "bra LAB_WAIT_RELAXED_HINT;\n\t"
               "DONE_RELAXED_HINT:\n\t"
               "}\n" ::"r"(mbar_addr),
               "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void
mbarrier_wait_cluster_hint(int mbar_addr, int phase,
                           uint32_t suspend_time_hint) {
  asm volatile("{\n\t"
               ".reg .pred P1;\n\t"
               "LAB_WAIT_CLUSTER_HINT:\n\t"
               "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
               " P1, [%0], %1, %2;\n\t"
               "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
               "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
               "DONE_CLUSTER_HINT:\n\t"
               "}\n" ::"r"(mbar_addr),
               "r"(phase), "r"(suspend_time_hint)
               : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase,
                                                    uint32_t token) {
  if (token == 0) {
    mbarrier_wait(mbar_addr, phase);
  }
}

__device__ __forceinline__ void
mbarrier_wait_token_suspend(int mbar_addr, int phase, uint32_t token,
                            uint32_t suspend_time_hint) {
  if (token == 0) {
    mbarrier_wait_suspend(mbar_addr, phase, suspend_time_hint);
  }
}

__device__ __forceinline__ void
mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
  if (token == 0) {
    mbarrier_wait_cluster(mbar_addr, phase);
  }
}

__device__ __forceinline__ void
mbarrier_wait_token_hint(int mbar_addr, int phase, uint32_t token,
                         uint32_t suspend_time_hint) {
  if (token == 0) {
    mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
  }
}

__device__ __forceinline__ void
mbarrier_wait_token_cluster_hint(int mbar_addr, int phase, uint32_t token,
                                 uint32_t suspend_time_hint) {
  if (token == 0) {
    mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
  }
}

__device__ __forceinline__ void
tcgen05_mma_mxf8_bs(int taddr, uint64_t a_desc, uint64_t b_desc,
                    uint32_t i_desc, int sfa_taddr, int sfb_taddr,
                    int enable_input_d) {
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "setp.ne.b32 p, %6, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale"
               " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
               "}\n" ::"r"(taddr),
               "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(sfa_taddr),
               "r"(sfb_taddr), "r"(enable_input_d));
}

__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
  return (x & 0x3FFFFULL) >> 4ULL;
}

union MmaSmemDesc {
  uint64_t u64;
  uint32_t u32[2];
};

__device__ __forceinline__ void incr_smem_desc_lo(uint64_t &smem_desc,
                                                  uint32_t offset) {
  MmaSmemDesc tmp;
  tmp.u64 = smem_desc;
  tmp.u32[0] += offset;
  smem_desc = tmp.u64;
}

__device__ __forceinline__ void elect_commit(int mbar_addr) {
  asm volatile("{\n\t"
               ".reg .pred leader;\n\t"
               "elect.sync _|leader, 0xFFFFFFFF;\n\t"
               "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
               ".shared::cluster.b64 [%0];\n\t"
               "}\n" ::"r"(mbar_addr));
}

__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
  asm volatile(
      "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(mbar_addr)
      : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr,
                                                          uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::
          "r"(mbar_addr),
      "r"(bytes)
      : "memory");
}

__device__ __forceinline__ float approx_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float approx_rcp(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ void elect_commit2(int mbar_addr0, int mbar_addr1) {
  asm volatile("{\n\t"
               ".reg .pred leader;\n\t"
               "elect.sync _|leader, 0xFFFFFFFF;\n\t"
               "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
               ".shared::cluster.b64 [%0];\n\t"
               "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
               ".shared::cluster.b64 [%1];\n\t"
               "}\n" ::"r"(mbar_addr0),
               "r"(mbar_addr1)
               : "memory");
}

__device__ __forceinline__ void tcgen05_commit2(int mbar_addr0,
                                                int mbar_addr1) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one"
               ".shared::cluster.b64 [%0];\n\t"
               "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
               ".shared::cluster.b64 [%1];\n\t" ::"r"(mbar_addr0),
               "r"(mbar_addr1)
               : "memory");
}

__device__ __forceinline__ void fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
  const int SBO = 1024;
  return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) |
         (2ULL << 61ULL);
}

__device__ __forceinline__ void tma_2d_gmem2smem(int dst, const void *tmap_ptr,
                                                 int x, int y, int mbar_addr) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global"
               ".mbarrier::complete_tx::bytes"
               " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
               "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr)
               : "memory");
}

__device__ __forceinline__ void tma_4d_gmem2smem(int dst, const void *tmap_ptr,
                                                 int x, int y, int z, int w,
                                                 int mbar_addr) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cta.global"
               ".mbarrier::complete_tx::bytes"
               " [%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(dst),
               "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(mbar_addr)
               : "memory");
}

__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one"
               ".shared::cluster.b64 [%0];" ::"r"(mbar_addr)
               : "memory");
}

__device__ __forceinline__ void tmem_ld_x4(float *dst, int tmem_addr) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32"
               " {%0, %1, %2, %3}, [%4];"
               : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
               : "r"(tmem_addr));
}

__device__ __forceinline__ void tmem_ld_x4_wait(float *dst, int addr) {
  tmem_ld_x4(dst, addr);
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
  uint32_t result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
               : "=r"(result)
               : "r"(val));
  return result;
}

__device__ __forceinline__ unsigned int __as_u32(float v) {
  unsigned int u;
  asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
  return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
  return *reinterpret_cast<const unsigned int *>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
  unsigned int u;
  asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
  return u;
}

extern "C" {

__global__ __launch_bounds__(512, 1) void dsv41_routed_expert_gemm1(
    const __grid_constant__ CUtensorMap A,
    const __grid_constant__ CUtensorMap SFA, uint8_t *__restrict__ X,
    uint8_t *__restrict__ XS, int *route, float *clamp,
    uint16_t *__restrict__ H, uint8_t *__restrict__ HS, int cta_words) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int mbar_base = smem;
#define a_full_addr (mbar_base + 0)
#define b_full_addr (mbar_base + 24)
#define sfa_full_addr (mbar_base + 48)
#define sfb_full_addr (mbar_base + 72)
#define sfa_free_addr (mbar_base + 96)
#define sfb_free_addr (mbar_base + 120)
#define tmem_sfa_full_addr (mbar_base + 144)
#define tmem_sfb_full_addr (mbar_base + 168)
#define k_done_addr (mbar_base + 192)
#define mma_full_addr (mbar_base + 216)
#define mma_free_addr (mbar_base + 232)
#define work_full_addr (mbar_base + 248)
#define work_empty_addr (mbar_base + 272)
#define throttle_full_addr (mbar_base + 296)
#define throttle_empty_addr (mbar_base + 320)
#define drain_full_addr (mbar_base + 344)

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  uint8_t *smem_a = reinterpret_cast<uint8_t *>(smem_raw + 1024);
  const int smem_a_addr = smem + 1024;
  uint8_t *smem_b = reinterpret_cast<uint8_t *>(smem_raw + 197632);
  const int smem_b_addr = smem + 197632;
  uint8_t *smem_sfa = reinterpret_cast<uint8_t *>(smem_raw + 209920);
  const int smem_sfa_addr = smem + 209920;
  uint8_t *smem_sfb = reinterpret_cast<uint8_t *>(smem_raw + 216064);
  const int smem_sfb_addr = smem + 216064;
  float *amax_x = reinterpret_cast<float *>(smem_raw + 216448);
  const int amax_x_addr = smem + 216448;
  unsigned int *work_response =
      reinterpret_cast<unsigned int *>(smem_raw + 216704);
  const int work_response_addr = smem + 216704;
  unsigned int *drain_response =
      reinterpret_cast<unsigned int *>(smem_raw + 216752);
  const int drain_response_addr = smem + 216752;

  if (warp == 0) {
    uint32_t leader = elect_sync();
    if (leader) {

      mbarrier_init(smem + 0, 1);
      mbarrier_init(smem + 8, 1);
      mbarrier_init(smem + 16, 1);

      mbarrier_init(smem + 24, 1);
      mbarrier_init(smem + 32, 1);
      mbarrier_init(smem + 40, 1);

      mbarrier_init(smem + 48, 1);
      mbarrier_init(smem + 56, 1);
      mbarrier_init(smem + 64, 1);

      mbarrier_init(smem + 72, 1);
      mbarrier_init(smem + 80, 1);
      mbarrier_init(smem + 88, 1);

      mbarrier_init(smem + 96, 1);
      mbarrier_init(smem + 104, 1);
      mbarrier_init(smem + 112, 1);

      mbarrier_init(smem + 120, 1);
      mbarrier_init(smem + 128, 1);
      mbarrier_init(smem + 136, 1);

      mbarrier_init(smem + 144, 1);
      mbarrier_init(smem + 152, 1);
      mbarrier_init(smem + 160, 1);

      mbarrier_init(smem + 168, 1);
      mbarrier_init(smem + 176, 1);
      mbarrier_init(smem + 184, 1);

      mbarrier_init(smem + 192, 1);
      mbarrier_init(smem + 200, 1);
      mbarrier_init(smem + 208, 1);

      mbarrier_init(smem + 216, 1);
      mbarrier_init(smem + 224, 1);

      mbarrier_init(smem + 232, 4);
      mbarrier_init(smem + 240, 4);

      mbarrier_init(smem + 248, 1);
      mbarrier_init(smem + 256, 1);
      mbarrier_init(smem + 264, 1);

      mbarrier_init(smem + 272, 512);
      mbarrier_init(smem + 280, 512);
      mbarrier_init(smem + 288, 512);

      mbarrier_init(smem + 296, 32);
      mbarrier_init(smem + 304, 32);
      mbarrier_init(smem + 312, 32);

      mbarrier_init(smem + 320, 32);
      mbarrier_init(smem + 328, 32);
      mbarrier_init(smem + 336, 32);

      mbarrier_init(smem + 344, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }

  __syncwarp();

  volatile int *tmem_addr_storage = (volatile int *)(smem_raw + 352);
  if (warp == 0) {
    int _tmem_hold = smem + 352;
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
            "r"(_tmem_hold),
        "r"(128)
        : "memory");
    __syncwarp();
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }

  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  const int taddr = tmem_addr_storage[0];

  const int tmem_accum = taddr;
  const int tmem_sfa = taddr + 16;
  const int tmem_sfb = taddr + 64;

  if (warp >= 8 && warp <= 15) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
  }

  if (warp <= 3) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
    {
      asm volatile("griddepcontrol.wait;" ::: "memory");
      const int warp_0 = warp;
      const int lane_1 = lane;
      unsigned int acc_stage = 0;
      unsigned int work_stage = 0;
      unsigned int m_tile = blockIdx.x;
      unsigned int n_tile = blockIdx.y;
      int pair = warp_0 * 16 + lane_1 / 4 * 2;
      int token0 = lane_1 % 4 * 2;
      unsigned int value = 1065353216;
      unsigned int value_2 = 1065353216;
      float2 _f2_0 =
          make_float2(__uint_as_float(value), __uint_as_float(value_2));
      float2 one2 = _f2_0;
      unsigned int value_3 = 3212836864;
      unsigned int value_4 = 3212836864;
      float2 _f2_1 =
          make_float2(__uint_as_float(value_3), __uint_as_float(value_4));
      float2 neg_one2 = _f2_1;
      float2 _f2_2 = make_float2(0.0f, 0.0f);
      float2 zero2 = _f2_2;
      unsigned int value_5 = 1069066811;
      unsigned int value_6 = 1069066811;
      float2 _f2_3 =
          make_float2(__uint_as_float(value_5), __uint_as_float(value_6));
      float2 log2e2 = _f2_3;
      unsigned int value_7 = 991054117;
      unsigned int value_8 = 991054117;
      float2 _f2_4 =
          make_float2(__uint_as_float(value_7), __uint_as_float(value_8));
      float2 rcp448_2 = _f2_4;
      unsigned int value_9 = 4286578687;
      float neg_max = __uint_as_float(value_9);
      unsigned int _phase_mma_full = 0;
      unsigned int _phase_work_full = 0;
#pragma unroll 1
      for (unsigned int _unit = 0; _unit < 576; _unit++) {
        if (n_tile >= (unsigned int)route[1]) {
          break;
        }
        int expert = route[4 + n_tile];
        int valid_rows =
            (unsigned int)route[(unsigned int)(4 + cta_words) + n_tile] -
            n_tile * 8;
        float limit = clamp[expert];
        float neg_limit = -limit;
        mbarrier_wait(mma_full_addr + (acc_stage) * 8, _phase_mma_full);
        asm volatile("tcgen05.fence::after_thread_sync;");
        int acc_offset = acc_stage * 8;
        float _tmem_load_0[4];
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32"
                     " {%0, %1, %2, %3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t *>(&_tmem_load_0[0])),
                       "=r"(*reinterpret_cast<uint32_t *>(&_tmem_load_0[1])),
                       "=r"(*reinterpret_cast<uint32_t *>(&_tmem_load_0[2])),
                       "=r"(*reinterpret_cast<uint32_t *>(&_tmem_load_0[3]))
                     : "r"(taddr + (unsigned int)acc_offset));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        float _tmem_load_1[4];
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32"
                     " {%0, %1, %2, %3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t *>(&_tmem_load_1[0])),
                       "=r"(*reinterpret_cast<uint32_t *>(&_tmem_load_1[1])),
                       "=r"(*reinterpret_cast<uint32_t *>(&_tmem_load_1[2])),
                       "=r"(*reinterpret_cast<uint32_t *>(&_tmem_load_1[3]))
                     : "r"(taddr + 1048576 + (unsigned int)acc_offset));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        asm volatile("tcgen05.fence::before_thread_sync;");
        if (elect_sync()) {
          mbarrier_arrive(mma_free_addr + (acc_stage) * 8);
        }
        float u = _tmem_load_0[0];
        float cu = u;
        if (u > limit) {
          cu = limit;
        }
        if (u < neg_limit) {
          cu = neg_limit;
        }
        float g = _tmem_load_0[2];
        float cg = g;
        if (g > limit) {
          cg = limit;
        }
        if (g < neg_max) {
          cg = neg_max;
        }
        float u_0 = _tmem_load_0[1];
        float cu_1 = u_0;
        if (u_0 > limit) {
          cu_1 = limit;
        }
        if (u_0 < neg_limit) {
          cu_1 = neg_limit;
        }
        float g_2 = _tmem_load_0[3];
        float cg_3 = g_2;
        if (g_2 > limit) {
          cg_3 = limit;
        }
        if (g_2 < neg_max) {
          cg_3 = neg_max;
        }
        float2 _f2_5 = make_float2(cu, cu_1);
        float2 up2 = _f2_5;
        float2 _f2_6 = make_float2(cg, cg_3);
        float2 gate2 = _f2_6;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
#error "Packed FP32x2 arithmetic requires SM100 or newer"
#endif
        float2 _packed_fma_f32x2_0;
        {
          float2 _packed_f32x2_0_0 = up2;
          float2 _packed_f32x2_0_1 = one2;
          float2 _packed_f32x2_0_2 = zero2;
          asm volatile(
              "fma.rn.f32x2 %0, %1, %2, %3;"
              : "=l"(reinterpret_cast<uint64_t &>(_packed_fma_f32x2_0))
              : "l"(reinterpret_cast<const uint64_t &>(_packed_f32x2_0_0)),
                "l"(reinterpret_cast<const uint64_t &>(_packed_f32x2_0_1)),
                "l"(reinterpret_cast<const uint64_t &>(_packed_f32x2_0_2)));
        }
        float2 _mul_f32x2_0;
        asm("mul.rn.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long *)&_mul_f32x2_0)
            : "l"(*(const unsigned long long *)&gate2),
              "l"(*(const unsigned long long *)&log2e2));
        float2 _mul_f32x2_1;
        asm("mul.rn.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long *)&_mul_f32x2_1)
            : "l"(*(const unsigned long long *)&neg_one2),
              "l"(*(const unsigned long long *)&_mul_f32x2_0));
        float _exp2_0 = approx_exp2(_mul_f32x2_1.x);
        float e0 = _exp2_0;
        float _exp2_1 = approx_exp2(_mul_f32x2_1.y);
        float e1 = _exp2_1;
        float2 _f2_7 = make_float2(e0, e1);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
#error "Packed FP32x2 arithmetic requires SM100 or newer"
#endif
        float2 _packed_add_f32x2_0;
        {
          float2 _packed_f32x2_1_0 = one2;
          float2 _packed_f32x2_1_1 = _f2_7;
          asm volatile(
              "add.f32x2 %0, %1, %2;"
              : "=l"(reinterpret_cast<uint64_t &>(_packed_add_f32x2_0))
              : "l"(reinterpret_cast<const uint64_t &>(_packed_f32x2_1_0)),
                "l"(reinterpret_cast<const uint64_t &>(_packed_f32x2_1_1)));
        }
        float _rcp_0 = approx_rcp(_packed_add_f32x2_0.x);
        float r0 = _rcp_0;
        float _rcp_1 = approx_rcp(_packed_add_f32x2_0.y);
        float r1 = _rcp_1;
        float2 _f2_8 = make_float2(r0, r1);
        float2 _mul_f32x2_2;
        asm("mul.rn.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long *)&_mul_f32x2_2)
            : "l"(*(const unsigned long long *)&gate2),
              "l"(*(const unsigned long long *)&_f2_8));
        float2 _mul_f32x2_3;
        asm("mul.rn.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long *)&_mul_f32x2_3)
            : "l"(*(const unsigned long long *)&_packed_fma_f32x2_0),
              "l"(*(const unsigned long long *)&_mul_f32x2_2));
        float u_4 = _tmem_load_1[0];
        float cu_5 = u_4;
        if (u_4 > limit) {
          cu_5 = limit;
        }
        if (u_4 < neg_limit) {
          cu_5 = neg_limit;
        }
        float g_6 = _tmem_load_1[2];
        float cg_7 = g_6;
        if (g_6 > limit) {
          cg_7 = limit;
        }
        if (g_6 < neg_max) {
          cg_7 = neg_max;
        }
        float u_8 = _tmem_load_1[1];
        float cu_9 = u_8;
        if (u_8 > limit) {
          cu_9 = limit;
        }
        if (u_8 < neg_limit) {
          cu_9 = neg_limit;
        }
        float g_10 = _tmem_load_1[3];
        float cg_11 = g_10;
        if (g_10 > limit) {
          cg_11 = limit;
        }
        if (g_10 < neg_max) {
          cg_11 = neg_max;
        }
        float2 _f2_9 = make_float2(cu_5, cu_9);
        float2 up2_12 = _f2_9;
        float2 _f2_10 = make_float2(cg_7, cg_11);
        float2 gate2_13 = _f2_10;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
#error "Packed FP32x2 arithmetic requires SM100 or newer"
#endif
        float2 _packed_fma_f32x2_1;
        {
          float2 _packed_f32x2_2_0 = up2_12;
          float2 _packed_f32x2_2_1 = one2;
          float2 _packed_f32x2_2_2 = zero2;
          asm volatile(
              "fma.rn.f32x2 %0, %1, %2, %3;"
              : "=l"(reinterpret_cast<uint64_t &>(_packed_fma_f32x2_1))
              : "l"(reinterpret_cast<const uint64_t &>(_packed_f32x2_2_0)),
                "l"(reinterpret_cast<const uint64_t &>(_packed_f32x2_2_1)),
                "l"(reinterpret_cast<const uint64_t &>(_packed_f32x2_2_2)));
        }
        float2 _mul_f32x2_4;
        asm("mul.rn.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long *)&_mul_f32x2_4)
            : "l"(*(const unsigned long long *)&gate2_13),
              "l"(*(const unsigned long long *)&log2e2));
        float2 _mul_f32x2_5;
        asm("mul.rn.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long *)&_mul_f32x2_5)
            : "l"(*(const unsigned long long *)&neg_one2),
              "l"(*(const unsigned long long *)&_mul_f32x2_4));
        float _exp2_2 = approx_exp2(_mul_f32x2_5.x);
        float e0_14 = _exp2_2;
        float _exp2_3 = approx_exp2(_mul_f32x2_5.y);
        float e1_15 = _exp2_3;
        float2 _f2_11 = make_float2(e0_14, e1_15);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
#error "Packed FP32x2 arithmetic requires SM100 or newer"
#endif
        float2 _packed_add_f32x2_1;
        {
          float2 _packed_f32x2_3_0 = one2;
          float2 _packed_f32x2_3_1 = _f2_11;
          asm volatile(
              "add.f32x2 %0, %1, %2;"
              : "=l"(reinterpret_cast<uint64_t &>(_packed_add_f32x2_1))
              : "l"(reinterpret_cast<const uint64_t &>(_packed_f32x2_3_0)),
                "l"(reinterpret_cast<const uint64_t &>(_packed_f32x2_3_1)));
        }
        float _rcp_2 = approx_rcp(_packed_add_f32x2_1.x);
        float r0_16 = _rcp_2;
        float _rcp_3 = approx_rcp(_packed_add_f32x2_1.y);
        float r1_17 = _rcp_3;
        float2 _f2_12 = make_float2(r0_16, r1_17);
        float2 _mul_f32x2_6;
        asm("mul.rn.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long *)&_mul_f32x2_6)
            : "l"(*(const unsigned long long *)&gate2_13),
              "l"(*(const unsigned long long *)&_f2_12));
        float2 _mul_f32x2_7;
        asm("mul.rn.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long *)&_mul_f32x2_7)
            : "l"(*(const unsigned long long *)&_packed_fma_f32x2_1),
              "l"(*(const unsigned long long *)&_mul_f32x2_6));
        float v_lo = ((1) ? _mul_f32x2_3.x : _mul_f32x2_3.y);
        float v_hi = ((1) ? _mul_f32x2_7.x : _mul_f32x2_7.y);
        float _fabs_0 = fabsf(v_lo);
        float _fabs_1 = fabsf(v_hi);
        float _fmax_0 = fmaxf(_fabs_0, _fabs_1);
        float m = _fmax_0;
        float _fabs_2 = fabsf(m);
        float _fmax_1 = fmaxf(_fabs_2, 0.0f);
        m = _fmax_1;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, m, 16);
        float _fmax_2 = fmaxf(m, _shfl_xor_0);
        m = _fmax_2;
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, m, 8);
        float _fmax_3 = fmaxf(m, _shfl_xor_1);
        m = _fmax_3;
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, m, 4);
        float _fmax_4 = fmaxf(m, _shfl_xor_2);
        m = _fmax_4;
        float v_lo_18 = ((0) ? _mul_f32x2_3.x : _mul_f32x2_3.y);
        float v_hi_19 = ((0) ? _mul_f32x2_7.x : _mul_f32x2_7.y);
        float _fabs_3 = fabsf(v_lo_18);
        float _fabs_4 = fabsf(v_hi_19);
        float _fmax_5 = fmaxf(_fabs_3, _fabs_4);
        float m_20 = _fmax_5;
        float _fabs_5 = fabsf(m_20);
        float _fmax_6 = fmaxf(_fabs_5, 0.0f);
        m_20 = _fmax_6;
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, m_20, 16);
        float _fmax_7 = fmaxf(m_20, _shfl_xor_3);
        m_20 = _fmax_7;
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, m_20, 8);
        float _fmax_8 = fmaxf(m_20, _shfl_xor_4);
        m_20 = _fmax_8;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, m_20, 4);
        float _fmax_9 = fmaxf(m_20, _shfl_xor_5);
        m_20 = _fmax_9;
        int amax_base = acc_stage * 32;
        if (lane_1 < 4) {
          amax_x[amax_base + warp_0 * 8 + token0] = m;
          amax_x[amax_base + warp_0 * 8 + token0 + 1] = m_20;
        }
        asm volatile("barrier.sync 7, 128;" ::: "memory");
        int partner = warp_0 ^ 1;
        float _fmax_10 = fmaxf(m, amax_x[amax_base + partner * 8 + token0]);
        float block_max = _fmax_10;
        float numerator = block_max;
        if (block_max == 0.0f) {
          unsigned int value_0 = 1138753536;
          numerator = __uint_as_float(value_0);
        }
        float _fmax_11 =
            fmaxf(m_20, amax_x[amax_base + partner * 8 + token0 + 1]);
        float block_max_21 = _fmax_11;
        float numerator_22 = block_max_21;
        if (block_max_21 == 0.0f) {
          unsigned int value_0_1 = 1138753536;
          numerator_22 = __uint_as_float(value_0_1);
        }
        float2 _f2_13 = make_float2(numerator, numerator_22);
        float2 _mul_f32x2_8;
        asm("mul.rn.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long *)&_mul_f32x2_8)
            : "l"(*(const unsigned long long *)&_f2_13),
              "l"(*(const unsigned long long *)&rcp448_2));
        unsigned int bits = __as_u32(((1) ? _mul_f32x2_8.x : _mul_f32x2_8.y));
        unsigned int exponent = bits >> 23 & 255;
        unsigned int mantissa = bits & 8388607;
        unsigned int code = exponent;
        if (mantissa != 0) {
          if (exponent != 254) {
            if (exponent != 0) {
              code = exponent + 1;
            }
            if (exponent == 0) {
              if (mantissa > 4194304) {
                code = 1;
              }
            }
          }
        }
        if (exponent == 255) {
          code = 255;
        }
        unsigned int inverse = 254 - code << 23;
        if (code == 254) {
          inverse = 4194304;
        }
        unsigned int bits_23 =
            __as_u32(((0) ? _mul_f32x2_8.x : _mul_f32x2_8.y));
        unsigned int exponent_24 = bits_23 >> 23 & 255;
        unsigned int mantissa_25 = bits_23 & 8388607;
        unsigned int code_26 = exponent_24;
        if (mantissa_25 != 0) {
          if (exponent_24 != 254) {
            if (exponent_24 != 0) {
              code_26 = exponent_24 + 1;
            }
            if (exponent_24 == 0) {
              if (mantissa_25 > 4194304) {
                code_26 = 1;
              }
            }
          }
        }
        if (exponent_24 == 255) {
          code_26 = 255;
        }
        unsigned int inverse_27 = 254 - code_26 << 23;
        if (code_26 == 254) {
          inverse_27 = 4194304;
        }
        float v_lo_28 =
            ((1) ? _mul_f32x2_3.x : _mul_f32x2_3.y) * __uint_as_float(inverse);
        float v_hi_29 =
            ((1) ? _mul_f32x2_7.x : _mul_f32x2_7.y) * __uint_as_float(inverse);
        uint16_t _e4m3x2_f32_0;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
            : "=h"(_e4m3x2_f32_0)
            : "f"(v_hi_29), "f"(v_lo_28));
        uint16_t packed = _e4m3x2_f32_0;
        int row = token0;
        if (row < valid_rows) {
          H[((n_tile * 8 + (unsigned int)row) * 640 + m_tile * 64 +
             (unsigned int)pair) /
            2] = packed;
        }
        float v_lo_30 = ((0) ? _mul_f32x2_3.x : _mul_f32x2_3.y) *
                        __uint_as_float(inverse_27);
        float v_hi_31 = ((0) ? _mul_f32x2_7.x : _mul_f32x2_7.y) *
                        __uint_as_float(inverse_27);
        uint16_t _e4m3x2_f32_1;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
            : "=h"(_e4m3x2_f32_1)
            : "f"(v_hi_31), "f"(v_lo_30));
        uint16_t packed_32 = _e4m3x2_f32_1;
        int row_33 = token0 + 1;
        if (row_33 < valid_rows) {
          H[((n_tile * 8 + (unsigned int)row_33) * 640 + m_tile * 64 +
             (unsigned int)pair) /
            2] = packed_32;
        }
        if (warp_0 % 2 == 0) {
          if (lane_1 < 8) {
            int token = lane_1 % 4 * 2 + lane_1 / 4;
            int block = m_tile * 2 + (unsigned int)(warp_0 / 2);
            unsigned int code_sel = code;
            if (lane_1 >= 4) {
              code_sel = code_26;
            }
            if (token < valid_rows) {
              HS[n_tile * 160 + (unsigned int)(block / 4 * 32) +
                 (unsigned int)(token * 4) + (unsigned int)(block % 4)] =
                  (uint8_t)code_sel;
            }
          }
        }
        if (m_tile == 8) {
          int thread = warp_0 * 32 + lane_1;
          int pad_row = thread / 16;
          if (pad_row < valid_rows) {
            H[((n_tile * 8 + (unsigned int)pad_row) * 640 + 576 +
               (unsigned int)(thread % 16 * 4)) /
              2] = 0;
            H[((n_tile * 8 + (unsigned int)pad_row) * 640 + 576 +
               (unsigned int)(thread % 16 * 4)) /
                  2 +
              1] = 0;
          }
          if (thread < 16) {
            int pad_token = thread / 2;
            int pad_block = 18 + thread % 2;
            if (pad_token < valid_rows) {
              HS[n_tile * 160 + (unsigned int)(pad_block / 4 * 32) +
                 (unsigned int)(pad_token * 4) +
                 (unsigned int)(pad_block % 4)] = 127;
            }
          }
        }
        acc_stage += 1;
        if (acc_stage == 2) {
          acc_stage = 0;
          _phase_mma_full ^= 1;
        }
        mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
        uint32_t _clc_valid_7 = 0;
        asm volatile("{\n\t"
                     ".reg .pred p1;\n\t"
                     ".reg .b128 clc_r;\n\t"
                     "ld.shared.b128 clc_r, [%1];\n\t"
                     "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 "
                     "p1, clc_r;\n\t"
                     "selp.u32 %0, 1, 0, p1;\n\t"
                     "}\n"
                     : "=r"(_clc_valid_7)
                     : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                     : "memory");
        unsigned int valid = _clc_valid_7;
        uint32_t _clc_ctaid_14 = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p1;\n\t"
            ".reg .b128 clc_r;\n\t"
            "ld.shared.b128 clc_r, [%1];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
            "clc_r;\n\t"
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 "
            "%0, clc_r;\n\t"
            "}\n"
            : "=r"(_clc_ctaid_14)
            : "r"(work_response_addr + work_stage * 16 + 0 * 16)
            : "memory");
        unsigned int next_x = _clc_ctaid_14;
        uint32_t _clc_ctaid_15 = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p1;\n\t"
            ".reg .b128 clc_r;\n\t"
            "ld.shared.b128 clc_r, [%1];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
            "clc_r;\n\t"
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 "
            "%0, clc_r;\n\t"
            "}\n"
            : "=r"(_clc_ctaid_15)
            : "r"(work_response_addr + work_stage * 16 + 0 * 16)
            : "memory");
        unsigned int next_y = _clc_ctaid_15;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbarrier_arrive(work_empty_addr + (work_stage) * 8);
        work_stage += 1;
        if (work_stage == 3) {
          work_stage = 0;
          _phase_work_full ^= 1;
        }
        unsigned int valid_34 = valid;
        m_tile = next_x;
        n_tile = next_y;
        if (valid_34 == 0) {
          break;
        }
      }
    }
  }

  if (warp >= 4 && warp <= 7) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    {
      asm volatile("griddepcontrol.wait;" ::: "memory");
      unsigned int stage = 0;
      unsigned int work_stage_1 = 0;
      unsigned int m_tile_1 = blockIdx.x;
      unsigned int n_tile_1 = blockIdx.y;
      const int lane_0 = lane;
      unsigned int word[1];
      unsigned int _phase_sfb_full = 0;
      unsigned int _phase_k_done = 1;
      unsigned int _phase_work_full_1 = 0;
#pragma unroll 1
      for (unsigned int _unit_1 = 0; _unit_1 < 576; _unit_1++) {
        if (n_tile_1 >= (unsigned int)route[1]) {
          break;
        }
#pragma unroll 1
        for (int _iter_k = 0; _iter_k < 10; _iter_k++) {
          mbarrier_wait(sfb_full_addr + (stage) * 8, _phase_sfb_full);
          mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
          asm volatile("tcgen05.fence::after_thread_sync;");
#pragma unroll
          for (int group = 0; group < 4; group++) {
            word[0] = 0;
            if (lane_0 < 8) {
              asm volatile("ld.shared.b32 %0, [%1];"
                           : "=r"(*reinterpret_cast<uint32_t *>(&word[0]))
                           : "r"(smem_sfb_addr + stage * 128 +
                                 (unsigned int)(group * 32) +
                                 (unsigned int)(lane_0 * 4)));
            }
            asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32"
                         " [%0], {%1};" ::"r"(taddr + 16 + 48 + stage * 8 +
                                              (unsigned int)(group * 2)),
                         "r"(word[0]));
          }
          asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
          asm volatile("tcgen05.fence::before_thread_sync;");
          asm volatile("barrier.sync 10, 128;" ::: "memory");
          if (warp == 4) {
            if (elect_sync()) {
              mbarrier_arrive(tmem_sfb_full_addr + (stage) * 8);
              mbarrier_arrive(sfb_free_addr + (stage) * 8);
            }
          }
          stage += 1;
          if (stage == 3) {
            stage = 0;
            _phase_sfb_full ^= 1;
            _phase_k_done ^= 1;
          }
        }
        mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
        uint32_t _clc_valid_5 = 0;
        asm volatile("{\n\t"
                     ".reg .pred p1;\n\t"
                     ".reg .b128 clc_r;\n\t"
                     "ld.shared.b128 clc_r, [%1];\n\t"
                     "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 "
                     "p1, clc_r;\n\t"
                     "selp.u32 %0, 1, 0, p1;\n\t"
                     "}\n"
                     : "=r"(_clc_valid_5)
                     : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                     : "memory");
        unsigned int valid_1 = _clc_valid_5;
        uint32_t _clc_ctaid_10 = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p1;\n\t"
            ".reg .b128 clc_r;\n\t"
            "ld.shared.b128 clc_r, [%1];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
            "clc_r;\n\t"
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 "
            "%0, clc_r;\n\t"
            "}\n"
            : "=r"(_clc_ctaid_10)
            : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
            : "memory");
        unsigned int next_x_1 = _clc_ctaid_10;
        uint32_t _clc_ctaid_11 = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p1;\n\t"
            ".reg .b128 clc_r;\n\t"
            "ld.shared.b128 clc_r, [%1];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
            "clc_r;\n\t"
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 "
            "%0, clc_r;\n\t"
            "}\n"
            : "=r"(_clc_ctaid_11)
            : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
            : "memory");
        unsigned int next_y_1 = _clc_ctaid_11;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbarrier_arrive(work_empty_addr + (work_stage_1) * 8);
        work_stage_1 += 1;
        if (work_stage_1 == 3) {
          work_stage_1 = 0;
          _phase_work_full_1 ^= 1;
        }
        unsigned int valid_0 = valid_1;
        m_tile_1 = next_x_1;
        n_tile_1 = next_y_1;
        if (valid_0 == 0) {
          break;
        }
      }
    }
  }

  if (warp >= 8 && warp <= 9) {
    {
      asm volatile("griddepcontrol.wait;" ::: "memory");
      unsigned int stage_1 = 0;
      unsigned int work_stage_2 = 0;
      unsigned int m_tile_2 = blockIdx.x;
      unsigned int n_tile_2 = blockIdx.y;
      int thread_1 = (warp - 8) * 32 + lane;
      int row_1 = thread_1 / 8;
      int chunk16 = thread_1 % 8;
      unsigned int _phase_k_done_1 = 1;
      unsigned int _phase_work_full_2 = 0;
#pragma unroll 1
      for (unsigned int _unit_2 = 0; _unit_2 < 576; _unit_2++) {
        if (n_tile_2 >= (unsigned int)route[1]) {
          break;
        }
        int valid_rows_1 =
            (unsigned int)route[(unsigned int)(4 + cta_words) + n_tile_2] -
            n_tile_2 * 8;
        int routed = 0;
        if (row_1 < valid_rows_1) {
          routed = route[(unsigned int)(4 + 2 * cta_words) + n_tile_2 * 8 +
                         (unsigned int)row_1];
        }
#pragma unroll 1
        for (int iter_k = 0; iter_k < 10; iter_k++) {
          mbarrier_wait(k_done_addr + (stage_1) * 8, _phase_k_done_1);
          int dst_base = smem_b_addr + stage_1 * 4096;
          for (int atom = 0; atom < 4; atom++) {
            asm volatile("{\n\t"
                         ".reg .pred p;\n\t"
                         "setp.ne.b32 p, %0, 0;\n\t"
                         "@p cp.async.ca.shared::cta.global [%1], [%2], 16;\n\t"
                         "}" ::"r"((row_1 < valid_rows_1) ? 1 : 0),
                         "r"(dst_base + atom * 1024 + row_1 * 128 +
                             (chunk16 ^ row_1) * 16),
                         "l"(X + (routed * 5120 + iter_k * 512 + atom * 128 +
                                  chunk16 * 16)));
          }
          asm volatile("{\n\t"
                       "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                       "}" ::"r"(b_full_addr + (stage_1) * 8)
                       : "memory");
          asm volatile("barrier.sync 8, 64;" ::: "memory");
          if (warp == 8) {
            if (elect_sync()) {
              mbarrier_arrive(b_full_addr + (stage_1) * 8);
            }
          }
          stage_1 += 1;
          if (stage_1 == 3) {
            stage_1 = 0;
            _phase_k_done_1 ^= 1;
          }
        }
        mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
        uint32_t _clc_valid_1 = 0;
        asm volatile("{\n\t"
                     ".reg .pred p1;\n\t"
                     ".reg .b128 clc_r;\n\t"
                     "ld.shared.b128 clc_r, [%1];\n\t"
                     "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 "
                     "p1, clc_r;\n\t"
                     "selp.u32 %0, 1, 0, p1;\n\t"
                     "}\n"
                     : "=r"(_clc_valid_1)
                     : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                     : "memory");
        unsigned int valid_2 = _clc_valid_1;
        uint32_t _clc_ctaid_2 = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p1;\n\t"
            ".reg .b128 clc_r;\n\t"
            "ld.shared.b128 clc_r, [%1];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
            "clc_r;\n\t"
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 "
            "%0, clc_r;\n\t"
            "}\n"
            : "=r"(_clc_ctaid_2)
            : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
            : "memory");
        unsigned int next_x_2 = _clc_ctaid_2;
        uint32_t _clc_ctaid_3 = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p1;\n\t"
            ".reg .b128 clc_r;\n\t"
            "ld.shared.b128 clc_r, [%1];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
            "clc_r;\n\t"
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 "
            "%0, clc_r;\n\t"
            "}\n"
            : "=r"(_clc_ctaid_3)
            : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
            : "memory");
        unsigned int next_y_2 = _clc_ctaid_3;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbarrier_arrive(work_empty_addr + (work_stage_2) * 8);
        work_stage_2 += 1;
        if (work_stage_2 == 3) {
          work_stage_2 = 0;
          _phase_work_full_2 ^= 1;
        }
        unsigned int valid_0_1 = valid_2;
        m_tile_2 = next_x_2;
        n_tile_2 = next_y_2;
        if (valid_0_1 == 0) {
          break;
        }
      }
    }
  }

  if (warp == 10) {
    {
      asm volatile("griddepcontrol.wait;" ::: "memory");
      unsigned int stage_2 = 0;
      unsigned int work_stage_3 = 0;
      unsigned int m_tile_3 = blockIdx.x;
      unsigned int n_tile_3 = blockIdx.y;
      int sf_row = lane / 4;
      int sf_group = lane % 4;
      unsigned int _phase_sfb_free = 1;
      unsigned int _phase_work_full_3 = 0;
#pragma unroll 1
      for (unsigned int _unit_3 = 0; _unit_3 < 576; _unit_3++) {
        if (n_tile_3 >= (unsigned int)route[1]) {
          break;
        }
        int valid_rows_2 =
            (unsigned int)route[(unsigned int)(4 + cta_words) + n_tile_3] -
            n_tile_3 * 8;
        int routed_1 = 0;
        if (sf_row < valid_rows_2) {
          routed_1 = route[(unsigned int)(4 + 2 * cta_words) + n_tile_3 * 8 +
                           (unsigned int)sf_row];
        }
#pragma unroll 1
        for (int iter_k_1 = 0; iter_k_1 < 10; iter_k_1++) {
          mbarrier_wait(sfb_free_addr + (stage_2) * 8, _phase_sfb_free);
          asm volatile(
              "{\n\t"
              ".reg .pred p;\n\t"
              "setp.ne.b32 p, %0, 0;\n\t"
              "@p cp.async.ca.shared::cta.global [%1], [%2], 4;\n\t"
              "}" ::"r"((sf_row < valid_rows_2) ? 1 : 0),
              "r"(smem_sfb_addr + stage_2 * 128 +
                  (unsigned int)(sf_group * 32) + (unsigned int)(sf_row * 4)),
              "l"(XS + (routed_1 * 160 + iter_k_1 * 16 + sf_group * 4)));
          asm volatile("{\n\t"
                       "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                       "}" ::"r"(sfb_full_addr + (stage_2) * 8)
                       : "memory");
          asm volatile("barrier.sync 9, 32;" ::: "memory");
          if (elect_sync()) {
            mbarrier_arrive(sfb_full_addr + (stage_2) * 8);
          }
          stage_2 += 1;
          if (stage_2 == 3) {
            stage_2 = 0;
            _phase_sfb_free ^= 1;
          }
        }
        mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
        uint32_t _clc_valid_3 = 0;
        asm volatile("{\n\t"
                     ".reg .pred p1;\n\t"
                     ".reg .b128 clc_r;\n\t"
                     "ld.shared.b128 clc_r, [%1];\n\t"
                     "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 "
                     "p1, clc_r;\n\t"
                     "selp.u32 %0, 1, 0, p1;\n\t"
                     "}\n"
                     : "=r"(_clc_valid_3)
                     : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                     : "memory");
        unsigned int valid_3 = _clc_valid_3;
        uint32_t _clc_ctaid_6 = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p1;\n\t"
            ".reg .b128 clc_r;\n\t"
            "ld.shared.b128 clc_r, [%1];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
            "clc_r;\n\t"
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 "
            "%0, clc_r;\n\t"
            "}\n"
            : "=r"(_clc_ctaid_6)
            : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
            : "memory");
        unsigned int next_x_3 = _clc_ctaid_6;
        uint32_t _clc_ctaid_7 = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p1;\n\t"
            ".reg .b128 clc_r;\n\t"
            "ld.shared.b128 clc_r, [%1];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
            "clc_r;\n\t"
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 "
            "%0, clc_r;\n\t"
            "}\n"
            : "=r"(_clc_ctaid_7)
            : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
            : "memory");
        unsigned int next_y_3 = _clc_ctaid_7;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbarrier_arrive(work_empty_addr + (work_stage_3) * 8);
        work_stage_3 += 1;
        if (work_stage_3 == 3) {
          work_stage_3 = 0;
          _phase_work_full_3 ^= 1;
        }
        unsigned int valid_0_2 = valid_3;
        m_tile_3 = next_x_3;
        n_tile_3 = next_y_3;
        if (valid_0_2 == 0) {
          break;
        }
      }
    }
  }

  if (warp == 11) {
    {
      asm volatile("griddepcontrol.wait;" ::: "memory");
      unsigned int stage_3 = 0;
      unsigned int work_stage_4 = 0;
      unsigned int throttle_stage = 0;
      unsigned int m_tile_4 = blockIdx.x;
      unsigned int n_tile_4 = blockIdx.y;
      unsigned int _phase_throttle_empty = 1;
      unsigned int _phase_k_done_2 = 1;
      unsigned int _phase_work_full_4 = 0;
#pragma unroll 1
      for (unsigned int _unit_4 = 0; _unit_4 < 576; _unit_4++) {
        if (n_tile_4 >= (unsigned int)route[1]) {
          break;
        }
        int expert_1 = route[4 + n_tile_4];
        mbarrier_wait(throttle_empty_addr + (throttle_stage) * 8,
                      _phase_throttle_empty);
        mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
        throttle_stage += 1;
        if (throttle_stage == 3) {
          throttle_stage = 0;
          _phase_throttle_empty ^= 1;
        }
#pragma unroll 1
        for (int iter_k_2 = 0; iter_k_2 < 10; iter_k_2++) {
          mbarrier_wait(k_done_addr + (stage_3) * 8, _phase_k_done_2);
          if (elect_sync()) {
            for (int half = 0; half < 2; half++) {
              asm volatile("cp.async.bulk.tensor.4d.shared::cta.global."
                           "mbarrier::complete_tx::bytes.L2::cache_hint"
                           " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(
                               smem_a_addr + stage_3 * 65536 +
                               (unsigned int)(half * 32768)),
                           "l"((&A)), "r"(0), "r"(m_tile_4 * 128),
                           "r"(iter_k_2 * 4 + half * 2), "r"(expert_1),
                           "r"(a_full_addr + (stage_3) * 8),
                           "l"(0x12F0000000000000ULL)
                           : "memory");
            }
            mbarrier_arrive_expect_tx(a_full_addr + (stage_3) * 8, 32768);
          }
          stage_3 += 1;
          if (stage_3 == 3) {
            stage_3 = 0;
            _phase_k_done_2 ^= 1;
          }
        }
        mbarrier_wait(work_full_addr + (work_stage_4) * 8, _phase_work_full_4);
        uint32_t _clc_valid_0 = 0;
        asm volatile("{\n\t"
                     ".reg .pred p1;\n\t"
                     ".reg .b128 clc_r;\n\t"
                     "ld.shared.b128 clc_r, [%1];\n\t"
                     "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 "
                     "p1, clc_r;\n\t"
                     "selp.u32 %0, 1, 0, p1;\n\t"
                     "}\n"
                     : "=r"(_clc_valid_0)
                     : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                     : "memory");
        unsigned int valid_4 = _clc_valid_0;
        uint32_t _clc_ctaid_0 = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p1;\n\t"
            ".reg .b128 clc_r;\n\t"
            "ld.shared.b128 clc_r, [%1];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
            "clc_r;\n\t"
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 "
            "%0, clc_r;\n\t"
            "}\n"
            : "=r"(_clc_ctaid_0)
            : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
            : "memory");
        unsigned int next_x_4 = _clc_ctaid_0;
        uint32_t _clc_ctaid_1 = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p1;\n\t"
            ".reg .b128 clc_r;\n\t"
            "ld.shared.b128 clc_r, [%1];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
            "clc_r;\n\t"
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 "
            "%0, clc_r;\n\t"
            "}\n"
            : "=r"(_clc_ctaid_1)
            : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
            : "memory");
        unsigned int next_y_4 = _clc_ctaid_1;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbarrier_arrive(work_empty_addr + (work_stage_4) * 8);
        work_stage_4 += 1;
        if (work_stage_4 == 3) {
          work_stage_4 = 0;
          _phase_work_full_4 ^= 1;
        }
        unsigned int valid_0_3 = valid_4;
        m_tile_4 = next_x_4;
        n_tile_4 = next_y_4;
        if (valid_0_3 == 0) {
          break;
        }
      }
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
  }

  if (warp == 12) {
    {
      asm volatile("griddepcontrol.wait;" ::: "memory");
      unsigned int stage_4 = 0;
      unsigned int work_stage_5 = 0;
      unsigned int m_tile_5 = blockIdx.x;
      unsigned int n_tile_5 = blockIdx.y;
      unsigned int _phase_sfa_free = 1;
      unsigned int _phase_work_full_5 = 0;
#pragma unroll 1
      for (unsigned int _unit_5 = 0; _unit_5 < 576; _unit_5++) {
        if (n_tile_5 >= (unsigned int)route[1]) {
          break;
        }
        int row_tile = (unsigned int)(route[4 + n_tile_5] * 10) + m_tile_5;
#pragma unroll 1
        for (int iter_k_3 = 0; iter_k_3 < 10; iter_k_3++) {
          mbarrier_wait(sfa_free_addr + (stage_4) * 8, _phase_sfa_free);
          if (elect_sync()) {
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_"
                "tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(smem_sfa_addr +
                                                         stage_4 * 2048),
                "l"((&SFA)), "r"(0), "r"((row_tile * 40 + iter_k_3 * 4) * 2),
                "r"(sfa_full_addr + (stage_4) * 8), "l"(0x12F0000000000000ULL)
                : "memory");
            mbarrier_arrive_expect_tx(sfa_full_addr + (stage_4) * 8, 2048);
          }
          stage_4 += 1;
          if (stage_4 == 3) {
            stage_4 = 0;
            _phase_sfa_free ^= 1;
          }
        }
        mbarrier_wait(work_full_addr + (work_stage_5) * 8, _phase_work_full_5);
        uint32_t _clc_valid_2 = 0;
        asm volatile("{\n\t"
                     ".reg .pred p1;\n\t"
                     ".reg .b128 clc_r;\n\t"
                     "ld.shared.b128 clc_r, [%1];\n\t"
                     "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 "
                     "p1, clc_r;\n\t"
                     "selp.u32 %0, 1, 0, p1;\n\t"
                     "}\n"
                     : "=r"(_clc_valid_2)
                     : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                     : "memory");
        unsigned int valid_5 = _clc_valid_2;
        uint32_t _clc_ctaid_4 = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p1;\n\t"
            ".reg .b128 clc_r;\n\t"
            "ld.shared.b128 clc_r, [%1];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
            "clc_r;\n\t"
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 "
            "%0, clc_r;\n\t"
            "}\n"
            : "=r"(_clc_ctaid_4)
            : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
            : "memory");
        unsigned int next_x_5 = _clc_ctaid_4;
        uint32_t _clc_ctaid_5 = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p1;\n\t"
            ".reg .b128 clc_r;\n\t"
            "ld.shared.b128 clc_r, [%1];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
            "clc_r;\n\t"
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 "
            "%0, clc_r;\n\t"
            "}\n"
            : "=r"(_clc_ctaid_5)
            : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
            : "memory");
        unsigned int next_y_5 = _clc_ctaid_5;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbarrier_arrive(work_empty_addr + (work_stage_5) * 8);
        work_stage_5 += 1;
        if (work_stage_5 == 3) {
          work_stage_5 = 0;
          _phase_work_full_5 ^= 1;
        }
        unsigned int valid_0_4 = valid_5;
        m_tile_5 = next_x_5;
        n_tile_5 = next_y_5;
        if (valid_0_4 == 0) {
          break;
        }
      }
    }
  }

  if (warp == 13) {
    {
      asm volatile("griddepcontrol.wait;" ::: "memory");
      unsigned int stage_5 = 0;
      unsigned int work_stage_6 = 0;
      unsigned int m_tile_6 = blockIdx.x;
      unsigned int n_tile_6 = blockIdx.y;
      unsigned int _phase_sfa_full = 0;
      unsigned int _phase_k_done_3 = 1;
      unsigned int _phase_work_full_6 = 0;
#pragma unroll 1
      for (unsigned int _unit_6 = 0; _unit_6 < 576; _unit_6++) {
        if (n_tile_6 >= (unsigned int)route[1]) {
          break;
        }
#pragma unroll 1
        for (int _iter_k_1 = 0; _iter_k_1 < 10; _iter_k_1++) {
          mbarrier_wait(sfa_full_addr + (stage_5) * 8, _phase_sfa_full);
          mbarrier_wait(k_done_addr + (stage_5) * 8, _phase_k_done_3);
          asm volatile("tcgen05.fence::after_thread_sync;");
          if (elect_sync()) {
            for (int group_1 = 0; group_1 < 4; group_1++) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
#error "Tcgen05Cp requires Blackwell tcgen05.cp support"
#endif
              {
                uint64_t _tcgen05_cp_desc_0 =
                    ((((uint64_t)(smem_sfa_addr + stage_5 * 2048 +
                                  (unsigned int)(group_1 * 512))) &
                      0x3FFFFULL) >>
                     4ULL) |
                    (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) |
                    (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) |
                    (1ULL << 46ULL) | (0ULL << 61ULL);
                asm volatile(
                    "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(
                        (uint32_t)((unsigned int)tmem_sfa +
                                   (stage_5 * 16 +
                                    (unsigned int)(group_1 * 4)))),
                    "l"(_tcgen05_cp_desc_0)
                    : "memory");
              }
            }
          }
          elect_commit2(tmem_sfa_full_addr + (stage_5) * 8,
                        sfa_free_addr + (stage_5) * 8);
          stage_5 += 1;
          if (stage_5 == 3) {
            stage_5 = 0;
            _phase_sfa_full ^= 1;
            _phase_k_done_3 ^= 1;
          }
        }
        mbarrier_wait(work_full_addr + (work_stage_6) * 8, _phase_work_full_6);
        uint32_t _clc_valid_4 = 0;
        asm volatile("{\n\t"
                     ".reg .pred p1;\n\t"
                     ".reg .b128 clc_r;\n\t"
                     "ld.shared.b128 clc_r, [%1];\n\t"
                     "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 "
                     "p1, clc_r;\n\t"
                     "selp.u32 %0, 1, 0, p1;\n\t"
                     "}\n"
                     : "=r"(_clc_valid_4)
                     : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                     : "memory");
        unsigned int valid_6 = _clc_valid_4;
        uint32_t _clc_ctaid_8 = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p1;\n\t"
            ".reg .b128 clc_r;\n\t"
            "ld.shared.b128 clc_r, [%1];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
            "clc_r;\n\t"
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 "
            "%0, clc_r;\n\t"
            "}\n"
            : "=r"(_clc_ctaid_8)
            : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
            : "memory");
        unsigned int next_x_6 = _clc_ctaid_8;
        uint32_t _clc_ctaid_9 = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p1;\n\t"
            ".reg .b128 clc_r;\n\t"
            "ld.shared.b128 clc_r, [%1];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
            "clc_r;\n\t"
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 "
            "%0, clc_r;\n\t"
            "}\n"
            : "=r"(_clc_ctaid_9)
            : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
            : "memory");
        unsigned int next_y_6 = _clc_ctaid_9;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbarrier_arrive(work_empty_addr + (work_stage_6) * 8);
        work_stage_6 += 1;
        if (work_stage_6 == 3) {
          work_stage_6 = 0;
          _phase_work_full_6 ^= 1;
        }
        unsigned int valid_0_5 = valid_6;
        m_tile_6 = next_x_6;
        n_tile_6 = next_y_6;
        if (valid_0_5 == 0) {
          break;
        }
      }
    }
  }

  if (warp == 14) {
    {
      asm volatile("griddepcontrol.wait;" ::: "memory");
      unsigned int stage_6 = 0;
      unsigned int acc_stage_1 = 0;
      unsigned int work_stage_7 = 0;
      unsigned int m_tile_7 = blockIdx.x;
      unsigned int n_tile_7 = blockIdx.y;
      unsigned int _phase_mma_free = 1;
      unsigned int _phase_a_full = 0;
      unsigned int _phase_b_full = 0;
      unsigned int _phase_tmem_sfa_full = 0;
      unsigned int _phase_tmem_sfb_full = 0;
      unsigned int _phase_work_full_7 = 0;
#pragma unroll 1
      for (unsigned int _unit_7 = 0; _unit_7 < 576; _unit_7++) {
        if (n_tile_7 >= (unsigned int)route[1]) {
          break;
        }
        mbarrier_wait(mma_free_addr + (acc_stage_1) * 8, _phase_mma_free);
#pragma unroll 1
        for (int iter_k_4 = 0; iter_k_4 < 10; iter_k_4++) {
          mbarrier_wait(a_full_addr + (stage_6) * 8, _phase_a_full);
          mbarrier_wait(b_full_addr + (stage_6) * 8, _phase_b_full);
          mbarrier_wait(tmem_sfa_full_addr + (stage_6) * 8,
                        _phase_tmem_sfa_full);
          mbarrier_wait(tmem_sfb_full_addr + (stage_6) * 8,
                        _phase_tmem_sfb_full);
          asm volatile("tcgen05.fence::after_thread_sync;");
          int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) +
                                              (stage_6) * 4096);
          int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) +
                                              (stage_6) * 256);
          if (elect_sync()) {
            {
              uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) |
                                ((uint64_t)0x40004040 << 32);
              uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) |
                                ((uint64_t)0x40004040 << 32);

              tcgen05_mma_mxf8_bs(
                  (tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                  0x8820280U, (unsigned int)tmem_sfa + stage_6 * 16,
                  (unsigned int)tmem_sfb + stage_6 * 8,
                  ((((1) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
              tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 2,
                                  b_desc + 2, 0x28820290U,
                                  (unsigned int)tmem_sfa + stage_6 * 16,
                                  (unsigned int)tmem_sfb + stage_6 * 8, 1);
              tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 4,
                                  b_desc + 4, 0x488202a0U,
                                  (unsigned int)tmem_sfa + stage_6 * 16,
                                  (unsigned int)tmem_sfb + stage_6 * 8, 1);
              tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 6,
                                  b_desc + 6, 0x688202b0U,
                                  (unsigned int)tmem_sfa + stage_6 * 16,
                                  (unsigned int)tmem_sfb + stage_6 * 8, 1);
            }
          }
          int _mma_a_lo_1 = make_warp_uniform(
              (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (stage_6) * 4096);
          int _mma_b_lo_1 = make_warp_uniform(
              (((smem_b_addr + 1024) >> 4) & 0x3FFF) + (stage_6) * 256);
          if (elect_sync()) {
            {
              uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) |
                                ((uint64_t)0x40004040 << 32);
              uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) |
                                ((uint64_t)0x40004040 << 32);

              tcgen05_mma_mxf8_bs(
                  (tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                  0x8820280U, (unsigned int)tmem_sfa + (stage_6 * 16 + 4),
                  (unsigned int)tmem_sfb + (stage_6 * 8 + 2),
                  ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
              tcgen05_mma_mxf8_bs(
                  (tmem_accum + (acc_stage_1 * 8)), a_desc + 2, b_desc + 2,
                  0x28820290U, (unsigned int)tmem_sfa + (stage_6 * 16 + 4),
                  (unsigned int)tmem_sfb + (stage_6 * 8 + 2), 1);
              tcgen05_mma_mxf8_bs(
                  (tmem_accum + (acc_stage_1 * 8)), a_desc + 4, b_desc + 4,
                  0x488202a0U, (unsigned int)tmem_sfa + (stage_6 * 16 + 4),
                  (unsigned int)tmem_sfb + (stage_6 * 8 + 2), 1);
              tcgen05_mma_mxf8_bs(
                  (tmem_accum + (acc_stage_1 * 8)), a_desc + 6, b_desc + 6,
                  0x688202b0U, (unsigned int)tmem_sfa + (stage_6 * 16 + 4),
                  (unsigned int)tmem_sfb + (stage_6 * 8 + 2), 1);
            }
          }
          int _mma_a_lo_2 = make_warp_uniform(
              (((smem_a_addr + 32768) >> 4) & 0x3FFF) + (stage_6) * 4096);
          int _mma_b_lo_2 = make_warp_uniform(
              (((smem_b_addr + 2048) >> 4) & 0x3FFF) + (stage_6) * 256);
          if (elect_sync()) {
            {
              uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) |
                                ((uint64_t)0x40004040 << 32);
              uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) |
                                ((uint64_t)0x40004040 << 32);

              tcgen05_mma_mxf8_bs(
                  (tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                  0x8820280U, (unsigned int)tmem_sfa + (stage_6 * 16 + 8),
                  (unsigned int)tmem_sfb + (stage_6 * 8 + 4),
                  ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
              tcgen05_mma_mxf8_bs(
                  (tmem_accum + (acc_stage_1 * 8)), a_desc + 2, b_desc + 2,
                  0x28820290U, (unsigned int)tmem_sfa + (stage_6 * 16 + 8),
                  (unsigned int)tmem_sfb + (stage_6 * 8 + 4), 1);
              tcgen05_mma_mxf8_bs(
                  (tmem_accum + (acc_stage_1 * 8)), a_desc + 4, b_desc + 4,
                  0x488202a0U, (unsigned int)tmem_sfa + (stage_6 * 16 + 8),
                  (unsigned int)tmem_sfb + (stage_6 * 8 + 4), 1);
              tcgen05_mma_mxf8_bs(
                  (tmem_accum + (acc_stage_1 * 8)), a_desc + 6, b_desc + 6,
                  0x688202b0U, (unsigned int)tmem_sfa + (stage_6 * 16 + 8),
                  (unsigned int)tmem_sfb + (stage_6 * 8 + 4), 1);
            }
          }
          int _mma_a_lo_3 = make_warp_uniform(
              (((smem_a_addr + 49152) >> 4) & 0x3FFF) + (stage_6) * 4096);
          int _mma_b_lo_3 = make_warp_uniform(
              (((smem_b_addr + 3072) >> 4) & 0x3FFF) + (stage_6) * 256);
          if (elect_sync()) {
            {
              uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) |
                                ((uint64_t)0x40004040 << 32);
              uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) |
                                ((uint64_t)0x40004040 << 32);

              tcgen05_mma_mxf8_bs(
                  (tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                  0x8820280U, (unsigned int)tmem_sfa + (stage_6 * 16 + 12),
                  (unsigned int)tmem_sfb + (stage_6 * 8 + 6),
                  ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
              tcgen05_mma_mxf8_bs(
                  (tmem_accum + (acc_stage_1 * 8)), a_desc + 2, b_desc + 2,
                  0x28820290U, (unsigned int)tmem_sfa + (stage_6 * 16 + 12),
                  (unsigned int)tmem_sfb + (stage_6 * 8 + 6), 1);
              tcgen05_mma_mxf8_bs(
                  (tmem_accum + (acc_stage_1 * 8)), a_desc + 4, b_desc + 4,
                  0x488202a0U, (unsigned int)tmem_sfa + (stage_6 * 16 + 12),
                  (unsigned int)tmem_sfb + (stage_6 * 8 + 6), 1);
              tcgen05_mma_mxf8_bs(
                  (tmem_accum + (acc_stage_1 * 8)), a_desc + 6, b_desc + 6,
                  0x688202b0U, (unsigned int)tmem_sfa + (stage_6 * 16 + 12),
                  (unsigned int)tmem_sfb + (stage_6 * 8 + 6), 1);
            }
          }
          if (iter_k_4 + 1 == 10) {
            elect_commit2(k_done_addr + (stage_6) * 8,
                          mma_full_addr + (acc_stage_1) * 8);
          } else {
            elect_commit(k_done_addr + (stage_6) * 8);
          }
          stage_6 += 1;
          if (stage_6 == 3) {
            stage_6 = 0;
            _phase_a_full ^= 1;
            _phase_b_full ^= 1;
            _phase_tmem_sfa_full ^= 1;
            _phase_tmem_sfb_full ^= 1;
          }
        }
        acc_stage_1 += 1;
        if (acc_stage_1 == 2) {
          acc_stage_1 = 0;
          _phase_mma_free ^= 1;
        }
        mbarrier_wait(work_full_addr + (work_stage_7) * 8, _phase_work_full_7);
        uint32_t _clc_valid_6 = 0;
        asm volatile("{\n\t"
                     ".reg .pred p1;\n\t"
                     ".reg .b128 clc_r;\n\t"
                     "ld.shared.b128 clc_r, [%1];\n\t"
                     "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 "
                     "p1, clc_r;\n\t"
                     "selp.u32 %0, 1, 0, p1;\n\t"
                     "}\n"
                     : "=r"(_clc_valid_6)
                     : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                     : "memory");
        unsigned int valid_7 = _clc_valid_6;
        uint32_t _clc_ctaid_12 = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p1;\n\t"
            ".reg .b128 clc_r;\n\t"
            "ld.shared.b128 clc_r, [%1];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
            "clc_r;\n\t"
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 "
            "%0, clc_r;\n\t"
            "}\n"
            : "=r"(_clc_ctaid_12)
            : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
            : "memory");
        unsigned int next_x_7 = _clc_ctaid_12;
        uint32_t _clc_ctaid_13 = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p1;\n\t"
            ".reg .b128 clc_r;\n\t"
            "ld.shared.b128 clc_r, [%1];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
            "clc_r;\n\t"
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 "
            "%0, clc_r;\n\t"
            "}\n"
            : "=r"(_clc_ctaid_13)
            : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
            : "memory");
        unsigned int next_y_7 = _clc_ctaid_13;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbarrier_arrive(work_empty_addr + (work_stage_7) * 8);
        work_stage_7 += 1;
        if (work_stage_7 == 3) {
          work_stage_7 = 0;
          _phase_work_full_7 ^= 1;
        }
        unsigned int valid_0_6 = valid_7;
        m_tile_7 = next_x_7;
        n_tile_7 = next_y_7;
        if (valid_0_6 == 0) {
          break;
        }
      }
    }
  }

  if (warp == 15) {
    {
      asm volatile("griddepcontrol.wait;" ::: "memory");
      unsigned int work_stage_8 = 0;
      unsigned int throttle_stage_1 = 0;
      unsigned int m_tile_8 = blockIdx.x;
      unsigned int n_tile_8 = blockIdx.y;
      unsigned int _phase_drain_full = 0;
      unsigned int _phase_throttle_full = 0;
      unsigned int _phase_work_empty = 1;
      unsigned int _phase_work_full_8 = 0;
#pragma unroll 1
      for (unsigned int _unit_8 = 0; _unit_8 < 576; _unit_8++) {
        if (n_tile_8 >= (unsigned int)route[1]) {
          {
            int rounds = (gridDim.x * gridDim.y + 4 - 1) / 4 + 1;
            unsigned int drain_stage = 0;
#pragma unroll 1
            for (unsigned int _round = 0; _round < rounds; _round++) {
              if (elect_sync()) {
                mbarrier_arrive_expect_tx(drain_full_addr + (drain_stage) * 8,
                                          64);
                asm volatile("fence.proxy.async.shared::cta;\n\t"
                             "clusterlaunchcontrol.try_cancel.async.shared::cta"
                             ".mbarrier::complete_tx::bytes.b128"
                             " [%0], [%1];" ::"r"(drain_response_addr +
                                                  drain_stage * 64 + 0 * 16),
                             "r"(drain_full_addr + drain_stage * 8)
                             : "memory");
                asm volatile("fence.proxy.async.shared::cta;\n\t"
                             "clusterlaunchcontrol.try_cancel.async.shared::cta"
                             ".mbarrier::complete_tx::bytes.b128"
                             " [%0], [%1];" ::"r"(drain_response_addr +
                                                  drain_stage * 64 + 1 * 16),
                             "r"(drain_full_addr + drain_stage * 8)
                             : "memory");
                asm volatile("fence.proxy.async.shared::cta;\n\t"
                             "clusterlaunchcontrol.try_cancel.async.shared::cta"
                             ".mbarrier::complete_tx::bytes.b128"
                             " [%0], [%1];" ::"r"(drain_response_addr +
                                                  drain_stage * 64 + 2 * 16),
                             "r"(drain_full_addr + drain_stage * 8)
                             : "memory");
                asm volatile("fence.proxy.async.shared::cta;\n\t"
                             "clusterlaunchcontrol.try_cancel.async.shared::cta"
                             ".mbarrier::complete_tx::bytes.b128"
                             " [%0], [%1];" ::"r"(drain_response_addr +
                                                  drain_stage * 64 + 3 * 16),
                             "r"(drain_full_addr + drain_stage * 8)
                             : "memory");
              }
              mbarrier_wait(drain_full_addr + (drain_stage) * 8,
                            _phase_drain_full);
              unsigned int canceled = 0;
              uint32_t _clc_valid_8 = 0;
              asm volatile(
                  "{\n\t"
                  ".reg .pred p1;\n\t"
                  ".reg .b128 clc_r;\n\t"
                  "ld.shared.b128 clc_r, [%1];\n\t"
                  "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
                  "clc_r;\n\t"
                  "selp.u32 %0, 1, 0, p1;\n\t"
                  "}\n"
                  : "=r"(_clc_valid_8)
                  : "r"(drain_response_addr + drain_stage * 64 + 0 * 16)
                  : "memory");
              canceled += _clc_valid_8;
              uint32_t _clc_valid_9 = 0;
              asm volatile(
                  "{\n\t"
                  ".reg .pred p1;\n\t"
                  ".reg .b128 clc_r;\n\t"
                  "ld.shared.b128 clc_r, [%1];\n\t"
                  "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
                  "clc_r;\n\t"
                  "selp.u32 %0, 1, 0, p1;\n\t"
                  "}\n"
                  : "=r"(_clc_valid_9)
                  : "r"(drain_response_addr + drain_stage * 64 + 1 * 16)
                  : "memory");
              canceled += _clc_valid_9;
              uint32_t _clc_valid_10 = 0;
              asm volatile(
                  "{\n\t"
                  ".reg .pred p1;\n\t"
                  ".reg .b128 clc_r;\n\t"
                  "ld.shared.b128 clc_r, [%1];\n\t"
                  "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
                  "clc_r;\n\t"
                  "selp.u32 %0, 1, 0, p1;\n\t"
                  "}\n"
                  : "=r"(_clc_valid_10)
                  : "r"(drain_response_addr + drain_stage * 64 + 2 * 16)
                  : "memory");
              canceled += _clc_valid_10;
              uint32_t _clc_valid_11 = 0;
              asm volatile(
                  "{\n\t"
                  ".reg .pred p1;\n\t"
                  ".reg .b128 clc_r;\n\t"
                  "ld.shared.b128 clc_r, [%1];\n\t"
                  "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
                  "clc_r;\n\t"
                  "selp.u32 %0, 1, 0, p1;\n\t"
                  "}\n"
                  : "=r"(_clc_valid_11)
                  : "r"(drain_response_addr + drain_stage * 64 + 3 * 16)
                  : "memory");
              canceled += _clc_valid_11;
              asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
              _phase_drain_full ^= 1;
              if (canceled == 0) {
                break;
              }
            }
          }
          break;
        }
        mbarrier_wait(throttle_full_addr + (throttle_stage_1) * 8,
                      _phase_throttle_full);
        mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
        throttle_stage_1 += 1;
        if (throttle_stage_1 == 3) {
          throttle_stage_1 = 0;
          _phase_throttle_full ^= 1;
        }
        if (elect_sync()) {
          mbarrier_wait(work_empty_addr + (work_stage_8) * 8,
                        _phase_work_empty);
          mbarrier_arrive_expect_tx(work_full_addr + (work_stage_8) * 8, 16);
          asm volatile("fence.proxy.async.shared::cta;\n\t"
                       "clusterlaunchcontrol.try_cancel.async.shared::cta"
                       ".mbarrier::complete_tx::bytes.b128"
                       " [%0], [%1];" ::"r"(work_response_addr +
                                            work_stage_8 * 16 + 0 * 16),
                       "r"(work_full_addr + work_stage_8 * 8)
                       : "memory");
        }
        __syncwarp();
        mbarrier_wait(work_full_addr + (work_stage_8) * 8, _phase_work_full_8);
        uint32_t _clc_valid_12 = 0;
        asm volatile("{\n\t"
                     ".reg .pred p1;\n\t"
                     ".reg .b128 clc_r;\n\t"
                     "ld.shared.b128 clc_r, [%1];\n\t"
                     "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 "
                     "p1, clc_r;\n\t"
                     "selp.u32 %0, 1, 0, p1;\n\t"
                     "}\n"
                     : "=r"(_clc_valid_12)
                     : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                     : "memory");
        unsigned int valid_8 = _clc_valid_12;
        uint32_t _clc_ctaid_16 = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p1;\n\t"
            ".reg .b128 clc_r;\n\t"
            "ld.shared.b128 clc_r, [%1];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
            "clc_r;\n\t"
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 "
            "%0, clc_r;\n\t"
            "}\n"
            : "=r"(_clc_ctaid_16)
            : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
            : "memory");
        unsigned int next_x_8 = _clc_ctaid_16;
        uint32_t _clc_ctaid_17 = 0;
        asm volatile(
            "{\n\t"
            ".reg .pred p1;\n\t"
            ".reg .b128 clc_r;\n\t"
            "ld.shared.b128 clc_r, [%1];\n\t"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
            "clc_r;\n\t"
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 "
            "%0, clc_r;\n\t"
            "}\n"
            : "=r"(_clc_ctaid_17)
            : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
            : "memory");
        unsigned int next_y_8 = _clc_ctaid_17;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbarrier_arrive(work_empty_addr + (work_stage_8) * 8);
        work_stage_8 += 1;
        if (work_stage_8 == 3) {
          work_stage_8 = 0;
          _phase_work_empty ^= 1;
          _phase_work_full_8 ^= 1;
        }
        unsigned int valid_0_7 = valid_8;
        m_tile_8 = next_x_8;
        n_tile_8 = next_y_8;
        if (valid_0_7 == 0) {
          break;
        }
      }
    }
  }

  __syncthreads();

  if (warp == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(
                     tmem_addr_storage[0]),
                 "r"(128));
  }
}
}
