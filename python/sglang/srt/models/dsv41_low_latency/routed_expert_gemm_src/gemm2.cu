// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// DeepSeek-V4.1 low-batch routed-expert GEMM2: rows = h W2^T (BF16) per routed
// expert, 128 x 8 tiles, 5-stage weight ring streamed before the grid wait.
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
#define TMEM_NCOLS 76
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SFA_OFFSET 16
#define TMEM_SFB_OFFSET 56
#define NUM_K_PIPE_STAGES 5
#define NUM_MMA_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define NUM_DRAIN_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 164864
#define SMEM_SMEM_B_STAGE_BYTES 2048
#define SMEM_SMEM_B_STRIDE 2048
#define SMEM_SMEM_SFA_OFF 175104
#define SMEM_SMEM_SFA_STAGE_BYTES 1024
#define SMEM_SMEM_SFA_STRIDE 1024
#define SMEM_SMEM_SFB_OFF 180224
#define SMEM_SMEM_SFB_STAGE_BYTES 64
#define SMEM_SMEM_SFB_STRIDE 64
#define SMEM_WORK_RESPONSE_OFF 180544
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_DRAIN_RESPONSE_OFF 180592
#define SMEM_DRAIN_RESPONSE_STAGE_BYTES 64
#define SMEM_DRAIN_RESPONSE_STRIDE 64
#define SMEM_TOTAL 180736
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

__global__ __launch_bounds__(512, 1) void dsv41_routed_expert_gemm2(
    const __grid_constant__ CUtensorMap A,
    const __grid_constant__ CUtensorMap SFA, uint8_t *__restrict__ H,
    uint8_t *__restrict__ HS, int *route, unsigned int *__restrict__ Y,
    int cta_words) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int mbar_base = smem;
#define a_full_addr (mbar_base + 0)
#define b_full_addr (mbar_base + 40)
#define sfa_full_addr (mbar_base + 80)
#define sfb_full_addr (mbar_base + 120)
#define sfa_free_addr (mbar_base + 160)
#define sfb_free_addr (mbar_base + 200)
#define tmem_sfa_full_addr (mbar_base + 240)
#define tmem_sfb_full_addr (mbar_base + 280)
#define k_done_addr (mbar_base + 320)
#define mma_full_addr (mbar_base + 360)
#define mma_free_addr (mbar_base + 376)
#define work_full_addr (mbar_base + 392)
#define work_empty_addr (mbar_base + 416)
#define throttle_full_addr (mbar_base + 440)
#define throttle_empty_addr (mbar_base + 464)
#define drain_full_addr (mbar_base + 488)

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  uint8_t *smem_a = reinterpret_cast<uint8_t *>(smem_raw + 1024);
  const int smem_a_addr = smem + 1024;
  uint8_t *smem_b = reinterpret_cast<uint8_t *>(smem_raw + 164864);
  const int smem_b_addr = smem + 164864;
  uint8_t *smem_sfa = reinterpret_cast<uint8_t *>(smem_raw + 175104);
  const int smem_sfa_addr = smem + 175104;
  uint8_t *smem_sfb = reinterpret_cast<uint8_t *>(smem_raw + 180224);
  const int smem_sfb_addr = smem + 180224;
  unsigned int *work_response =
      reinterpret_cast<unsigned int *>(smem_raw + 180544);
  const int work_response_addr = smem + 180544;
  unsigned int *drain_response =
      reinterpret_cast<unsigned int *>(smem_raw + 180592);
  const int drain_response_addr = smem + 180592;

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
      mbarrier_init(smem + 232, 1);

      mbarrier_init(smem + 240, 1);
      mbarrier_init(smem + 248, 1);
      mbarrier_init(smem + 256, 1);
      mbarrier_init(smem + 264, 1);
      mbarrier_init(smem + 272, 1);

      mbarrier_init(smem + 280, 1);
      mbarrier_init(smem + 288, 1);
      mbarrier_init(smem + 296, 1);
      mbarrier_init(smem + 304, 1);
      mbarrier_init(smem + 312, 1);

      mbarrier_init(smem + 320, 1);
      mbarrier_init(smem + 328, 1);
      mbarrier_init(smem + 336, 1);
      mbarrier_init(smem + 344, 1);
      mbarrier_init(smem + 352, 1);

      mbarrier_init(smem + 360, 1);
      mbarrier_init(smem + 368, 1);

      mbarrier_init(smem + 376, 4);
      mbarrier_init(smem + 384, 4);

      mbarrier_init(smem + 392, 1);
      mbarrier_init(smem + 400, 1);
      mbarrier_init(smem + 408, 1);

      mbarrier_init(smem + 416, 512);
      mbarrier_init(smem + 424, 512);
      mbarrier_init(smem + 432, 512);

      mbarrier_init(smem + 440, 32);
      mbarrier_init(smem + 448, 32);
      mbarrier_init(smem + 456, 32);

      mbarrier_init(smem + 464, 32);
      mbarrier_init(smem + 472, 32);
      mbarrier_init(smem + 480, 32);

      mbarrier_init(smem + 488, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }

  __syncwarp();

  volatile int *tmem_addr_storage = (volatile int *)(smem_raw + 496);
  if (warp == 0) {
    int _tmem_hold = smem + 496;
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
  const int tmem_sfb = taddr + 56;

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
      int hidden0 = warp_0 * 32 + lane_1 / 4 * 4;
      int token0 = lane_1 % 4 * 2;
      unsigned int _phase_mma_full = 0;
      unsigned int _phase_work_full = 0;
#pragma unroll 1
      for (unsigned int _unit = 0; _unit < 2560; _unit++) {
        if (n_tile >= (unsigned int)route[1]) {
          break;
        }
        int valid_rows =
            (unsigned int)route[(unsigned int)(4 + cta_words) + n_tile] -
            n_tile * 8;
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
        int row = token0;
        if (row < valid_rows) {
          int word = ((n_tile * 8 + (unsigned int)row) * 5120 + m_tile * 128 +
                      (unsigned int)hidden0) /
                     2;
          __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(
              make_float2(_tmem_load_0[0], _tmem_load_0[2]));
          Y[word] = __as_u32(_bf16x2_0);
          __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(
              make_float2(_tmem_load_1[0], _tmem_load_1[2]));
          Y[word + 1] = __as_u32(_bf16x2_1);
        }
        int row_0 = token0 + 1;
        if (row_0 < valid_rows) {
          int word_1 = ((n_tile * 8 + (unsigned int)row_0) * 5120 +
                        m_tile * 128 + (unsigned int)hidden0) /
                       2;
          __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(
              make_float2(_tmem_load_0[1], _tmem_load_0[3]));
          Y[word_1] = __as_u32(_bf16x2_2);
          __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(
              make_float2(_tmem_load_1[1], _tmem_load_1[3]));
          Y[word_1 + 1] = __as_u32(_bf16x2_3);
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
        unsigned int valid_1 = valid;
        m_tile = next_x;
        n_tile = next_y;
        if (valid_1 == 0) {
          break;
        }
      }
    }
  }

  if (warp >= 4 && warp <= 7) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    {
      unsigned int stage = 0;
      unsigned int work_stage_1 = 0;
      unsigned int m_tile_1 = blockIdx.x;
      unsigned int n_tile_1 = blockIdx.y;
      const int lane_0 = lane;
      unsigned int word_2[1];
      unsigned int _phase_sfb_full = 0;
      unsigned int _phase_k_done = 1;
      unsigned int _phase_work_full_1 = 0;
#pragma unroll 1
      for (unsigned int _unit_1 = 0; _unit_1 < 2560; _unit_1++) {
        if (n_tile_1 >= (unsigned int)route[1]) {
          break;
        }
        mbarrier_wait(sfb_full_addr + (stage) * 8, _phase_sfb_full);
        mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
        asm volatile("tcgen05.fence::after_thread_sync;");
        word_2[0] = 0;
        if (lane_0 < 8) {
          asm volatile(
              "ld.shared.b32 %0, [%1];"
              : "=r"(*reinterpret_cast<uint32_t *>(&word_2[0]))
              : "r"(smem_sfb_addr + stage * 64 + (unsigned int)(lane_0 * 4)));
        }
        asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32"
                     " [%0], {%1};" ::"r"(taddr + 56 + stage * 4),
                     "r"(word_2[0]));
        word_2[0] = 0;
        if (lane_0 < 8) {
          asm volatile("ld.shared.b32 %0, [%1];"
                       : "=r"(*reinterpret_cast<uint32_t *>(&word_2[0]))
                       : "r"(smem_sfb_addr + stage * 64 + 32 +
                             (unsigned int)(lane_0 * 4)));
        }
        asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32"
                     " [%0], {%1};" ::"r"(taddr + 56 + stage * 4 + 2),
                     "r"(word_2[0]));
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
        if (stage == 5) {
          stage = 0;
          _phase_sfb_full ^= 1;
          _phase_k_done ^= 1;
        }
        mbarrier_wait(sfb_full_addr + (stage) * 8, _phase_sfb_full);
        mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
        asm volatile("tcgen05.fence::after_thread_sync;");
        word_2[0] = 0;
        if (lane_0 < 8) {
          asm volatile(
              "ld.shared.b32 %0, [%1];"
              : "=r"(*reinterpret_cast<uint32_t *>(&word_2[0]))
              : "r"(smem_sfb_addr + stage * 64 + (unsigned int)(lane_0 * 4)));
        }
        asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32"
                     " [%0], {%1};" ::"r"(taddr + 56 + stage * 4),
                     "r"(word_2[0]));
        word_2[0] = 0;
        if (lane_0 < 8) {
          asm volatile("ld.shared.b32 %0, [%1];"
                       : "=r"(*reinterpret_cast<uint32_t *>(&word_2[0]))
                       : "r"(smem_sfb_addr + stage * 64 + 32 +
                             (unsigned int)(lane_0 * 4)));
        }
        asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32"
                     " [%0], {%1};" ::"r"(taddr + 56 + stage * 4 + 2),
                     "r"(word_2[0]));
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
        if (stage == 5) {
          stage = 0;
          _phase_sfb_full ^= 1;
          _phase_k_done ^= 1;
        }
        mbarrier_wait(sfb_full_addr + (stage) * 8, _phase_sfb_full);
        mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
        asm volatile("tcgen05.fence::after_thread_sync;");
        word_2[0] = 0;
        if (lane_0 < 8) {
          asm volatile(
              "ld.shared.b32 %0, [%1];"
              : "=r"(*reinterpret_cast<uint32_t *>(&word_2[0]))
              : "r"(smem_sfb_addr + stage * 64 + (unsigned int)(lane_0 * 4)));
        }
        asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32"
                     " [%0], {%1};" ::"r"(taddr + 56 + stage * 4),
                     "r"(word_2[0]));
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
        if (stage == 5) {
          stage = 0;
          _phase_sfb_full ^= 1;
          _phase_k_done ^= 1;
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
        unsigned int valid_2 = _clc_valid_5;
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
        unsigned int valid_0 = valid_2;
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
      int thread = (warp - 8) * 32 + lane;
      int row_1 = thread / 8;
      int chunk16 = thread % 8;
      unsigned int _phase_k_done_1 = 1;
      unsigned int _phase_work_full_2 = 0;
#pragma unroll 1
      for (unsigned int _unit_2 = 0; _unit_2 < 2560; _unit_2++) {
        if (n_tile_2 >= (unsigned int)route[1]) {
          break;
        }
        int valid_rows_1 =
            (unsigned int)route[(unsigned int)(4 + cta_words) + n_tile_2] -
            n_tile_2 * 8;
        mbarrier_wait(k_done_addr + (stage_1) * 8, _phase_k_done_1);
        int dst_base = smem_b_addr + stage_1 * 2048;
        asm volatile("{\n\t"
                     ".reg .pred p;\n\t"
                     "setp.ne.b32 p, %0, 0;\n\t"
                     "@p cp.async.ca.shared::cta.global [%1], [%2], 16;\n\t"
                     "}" ::"r"((row_1 < valid_rows_1) ? 1 : 0),
                     "r"(dst_base + row_1 * 128 + (chunk16 ^ row_1) * 16),
                     "l"(H + ((n_tile_2 * 8 + (unsigned int)row_1) * 640 +
                              (unsigned int)(chunk16 * 16))));
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %0, 0;\n\t"
            "@p cp.async.ca.shared::cta.global [%1], [%2], 16;\n\t"
            "}" ::"r"((row_1 < valid_rows_1) ? 1 : 0),
            "r"(dst_base + 1024 + row_1 * 128 + (chunk16 ^ row_1) * 16),
            "l"(H + ((n_tile_2 * 8 + (unsigned int)row_1) * 640 + 128 +
                     (unsigned int)(chunk16 * 16))));
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
        if (stage_1 == 5) {
          stage_1 = 0;
          _phase_k_done_1 ^= 1;
        }
        mbarrier_wait(k_done_addr + (stage_1) * 8, _phase_k_done_1);
        int dst_base_0 = smem_b_addr + stage_1 * 2048;
        asm volatile("{\n\t"
                     ".reg .pred p;\n\t"
                     "setp.ne.b32 p, %0, 0;\n\t"
                     "@p cp.async.ca.shared::cta.global [%1], [%2], 16;\n\t"
                     "}" ::"r"((row_1 < valid_rows_1) ? 1 : 0),
                     "r"(dst_base_0 + row_1 * 128 + (chunk16 ^ row_1) * 16),
                     "l"(H + ((n_tile_2 * 8 + (unsigned int)row_1) * 640 + 256 +
                              (unsigned int)(chunk16 * 16))));
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %0, 0;\n\t"
            "@p cp.async.ca.shared::cta.global [%1], [%2], 16;\n\t"
            "}" ::"r"((row_1 < valid_rows_1) ? 1 : 0),
            "r"(dst_base_0 + 1024 + row_1 * 128 + (chunk16 ^ row_1) * 16),
            "l"(H + ((n_tile_2 * 8 + (unsigned int)row_1) * 640 + 256 + 128 +
                     (unsigned int)(chunk16 * 16))));
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
        if (stage_1 == 5) {
          stage_1 = 0;
          _phase_k_done_1 ^= 1;
        }
        mbarrier_wait(k_done_addr + (stage_1) * 8, _phase_k_done_1);
        int dst_base_1 = smem_b_addr + stage_1 * 2048;
        asm volatile("{\n\t"
                     ".reg .pred p;\n\t"
                     "setp.ne.b32 p, %0, 0;\n\t"
                     "@p cp.async.ca.shared::cta.global [%1], [%2], 16;\n\t"
                     "}" ::"r"((row_1 < valid_rows_1) ? 1 : 0),
                     "r"(dst_base_1 + row_1 * 128 + (chunk16 ^ row_1) * 16),
                     "l"(H + ((n_tile_2 * 8 + (unsigned int)row_1) * 640 + 512 +
                              (unsigned int)(chunk16 * 16))));
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
        if (stage_1 == 5) {
          stage_1 = 0;
          _phase_k_done_1 ^= 1;
        }
        mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
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
                     : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                     : "memory");
        unsigned int valid_3 = _clc_valid_2;
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
            : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
            : "memory");
        unsigned int next_x_2 = _clc_ctaid_4;
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
            : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
            : "memory");
        unsigned int next_y_2 = _clc_ctaid_5;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbarrier_arrive(work_empty_addr + (work_stage_2) * 8);
        work_stage_2 += 1;
        if (work_stage_2 == 3) {
          work_stage_2 = 0;
          _phase_work_full_2 ^= 1;
        }
        unsigned int valid_2_1 = valid_3;
        m_tile_2 = next_x_2;
        n_tile_2 = next_y_2;
        if (valid_2_1 == 0) {
          break;
        }
      }
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
  }

  if (warp == 10) {
    {
      asm volatile("griddepcontrol.wait;" ::: "memory");
      unsigned int stage_2 = 0;
      unsigned int work_stage_3 = 0;
      unsigned int m_tile_3 = blockIdx.x;
      unsigned int n_tile_3 = blockIdx.y;
      const int lane_0_1 = lane;
      int sf_row = lane_0_1 % 8;
      unsigned int _phase_sfb_free = 1;
      unsigned int _phase_work_full_3 = 0;
#pragma unroll 1
      for (unsigned int _unit_3 = 0; _unit_3 < 2560; _unit_3++) {
        if (n_tile_3 >= (unsigned int)route[1]) {
          break;
        }
        int valid_rows_2 =
            (unsigned int)route[(unsigned int)(4 + cta_words) + n_tile_3] -
            n_tile_3 * 8;
        mbarrier_wait(sfb_free_addr + (stage_2) * 8, _phase_sfb_free);
        if (lane_0_1 < 16) {
          asm volatile(
              "{\n\t"
              ".reg .pred p;\n\t"
              "setp.ne.b32 p, %0, 0;\n\t"
              "@p cp.async.ca.shared::cta.global [%1], [%2], 4;\n\t"
              "}" ::"r"((sf_row < valid_rows_2) ? 1 : 0),
              "r"(smem_sfb_addr + stage_2 * 64 + (unsigned int)(lane_0_1 * 4)),
              "l"(HS + (n_tile_3 * 160 + (unsigned int)(lane_0_1 * 4))));
        }
        asm volatile("{\n\t"
                     "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                     "}" ::"r"(sfb_full_addr + (stage_2) * 8)
                     : "memory");
        asm volatile("barrier.sync 9, 32;" ::: "memory");
        if (elect_sync()) {
          mbarrier_arrive(sfb_full_addr + (stage_2) * 8);
        }
        stage_2 += 1;
        if (stage_2 == 5) {
          stage_2 = 0;
          _phase_sfb_free ^= 1;
        }
        mbarrier_wait(sfb_free_addr + (stage_2) * 8, _phase_sfb_free);
        if (lane_0_1 < 16) {
          asm volatile(
              "{\n\t"
              ".reg .pred p;\n\t"
              "setp.ne.b32 p, %0, 0;\n\t"
              "@p cp.async.ca.shared::cta.global [%1], [%2], 4;\n\t"
              "}" ::"r"((sf_row < valid_rows_2) ? 1 : 0),
              "r"(smem_sfb_addr + stage_2 * 64 + (unsigned int)(lane_0_1 * 4)),
              "l"(HS + (n_tile_3 * 160 + 64 + (unsigned int)(lane_0_1 * 4))));
        }
        asm volatile("{\n\t"
                     "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                     "}" ::"r"(sfb_full_addr + (stage_2) * 8)
                     : "memory");
        asm volatile("barrier.sync 9, 32;" ::: "memory");
        if (elect_sync()) {
          mbarrier_arrive(sfb_full_addr + (stage_2) * 8);
        }
        stage_2 += 1;
        if (stage_2 == 5) {
          stage_2 = 0;
          _phase_sfb_free ^= 1;
        }
        mbarrier_wait(sfb_free_addr + (stage_2) * 8, _phase_sfb_free);
        if (lane_0_1 < 8) {
          asm volatile(
              "{\n\t"
              ".reg .pred p;\n\t"
              "setp.ne.b32 p, %0, 0;\n\t"
              "@p cp.async.ca.shared::cta.global [%1], [%2], 4;\n\t"
              "}" ::"r"((sf_row < valid_rows_2) ? 1 : 0),
              "r"(smem_sfb_addr + stage_2 * 64 + (unsigned int)(lane_0_1 * 4)),
              "l"(HS + (n_tile_3 * 160 + 128 + (unsigned int)(lane_0_1 * 4))));
        }
        asm volatile("{\n\t"
                     "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                     "}" ::"r"(sfb_full_addr + (stage_2) * 8)
                     : "memory");
        asm volatile("barrier.sync 9, 32;" ::: "memory");
        if (elect_sync()) {
          mbarrier_arrive(sfb_full_addr + (stage_2) * 8);
        }
        stage_2 += 1;
        if (stage_2 == 5) {
          stage_2 = 0;
          _phase_sfb_free ^= 1;
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
        unsigned int valid_4 = _clc_valid_3;
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
        unsigned int valid_0_1 = valid_4;
        m_tile_3 = next_x_3;
        n_tile_3 = next_y_3;
        if (valid_0_1 == 0) {
          break;
        }
      }
    }
  }

  if (warp == 11) {
    {
      unsigned int stage_3 = 0;
      unsigned int work_stage_4 = 0;
      unsigned int throttle_stage = 0;
      unsigned int m_tile_4 = blockIdx.x;
      unsigned int n_tile_4 = blockIdx.y;
      unsigned int _phase_throttle_empty = 1;
      unsigned int _phase_k_done_2 = 1;
      unsigned int _phase_work_full_4 = 0;
#pragma unroll 1
      for (unsigned int _unit_4 = 0; _unit_4 < 2560; _unit_4++) {
        if (n_tile_4 >= (unsigned int)route[1]) {
          break;
        }
        int expert = route[4 + n_tile_4];
        mbarrier_wait(throttle_empty_addr + (throttle_stage) * 8,
                      _phase_throttle_empty);
        mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
        throttle_stage += 1;
        if (throttle_stage == 3) {
          throttle_stage = 0;
          _phase_throttle_empty ^= 1;
        }
        mbarrier_wait(k_done_addr + (stage_3) * 8, _phase_k_done_2);
        if (elect_sync()) {
          asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::"
                       "complete_tx::bytes.L2::cache_hint"
                       " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(
                           smem_a_addr + stage_3 * 32768),
                       "l"((&A)), "r"(0), "r"(m_tile_4 * 128), "r"(0),
                       "r"(expert), "r"(a_full_addr + (stage_3) * 8),
                       "l"(0x12F0000000000000ULL)
                       : "memory");
          mbarrier_arrive_expect_tx(a_full_addr + (stage_3) * 8, 16384);
        }
        stage_3 += 1;
        if (stage_3 == 5) {
          stage_3 = 0;
          _phase_k_done_2 ^= 1;
        }
        mbarrier_wait(k_done_addr + (stage_3) * 8, _phase_k_done_2);
        if (elect_sync()) {
          asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::"
                       "complete_tx::bytes.L2::cache_hint"
                       " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(
                           smem_a_addr + stage_3 * 32768),
                       "l"((&A)), "r"(0), "r"(m_tile_4 * 128), "r"(2),
                       "r"(expert), "r"(a_full_addr + (stage_3) * 8),
                       "l"(0x12F0000000000000ULL)
                       : "memory");
          mbarrier_arrive_expect_tx(a_full_addr + (stage_3) * 8, 16384);
        }
        stage_3 += 1;
        if (stage_3 == 5) {
          stage_3 = 0;
          _phase_k_done_2 ^= 1;
        }
        mbarrier_wait(k_done_addr + (stage_3) * 8, _phase_k_done_2);
        if (elect_sync()) {
          asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::"
                       "complete_tx::bytes.L2::cache_hint"
                       " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(
                           smem_a_addr + stage_3 * 32768),
                       "l"((&A)), "r"(0), "r"(m_tile_4 * 128), "r"(4),
                       "r"(expert), "r"(a_full_addr + (stage_3) * 8),
                       "l"(0x12F0000000000000ULL)
                       : "memory");
          mbarrier_arrive_expect_tx(a_full_addr + (stage_3) * 8, 16384);
        }
        stage_3 += 1;
        if (stage_3 == 5) {
          stage_3 = 0;
          _phase_k_done_2 ^= 1;
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
        unsigned int valid_5 = _clc_valid_0;
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
        unsigned int valid_0_2 = valid_5;
        m_tile_4 = next_x_4;
        n_tile_4 = next_y_4;
        if (valid_0_2 == 0) {
          break;
        }
      }
    }
  }

  if (warp == 12) {
    {
      unsigned int stage_4 = 0;
      unsigned int work_stage_5 = 0;
      unsigned int m_tile_5 = blockIdx.x;
      unsigned int n_tile_5 = blockIdx.y;
      unsigned int _phase_sfa_free = 1;
      unsigned int _phase_work_full_5 = 0;
#pragma unroll 1
      for (unsigned int _unit_5 = 0; _unit_5 < 2560; _unit_5++) {
        if (n_tile_5 >= (unsigned int)route[1]) {
          break;
        }
        int row_tile = (unsigned int)(route[4 + n_tile_5] * 40) + m_tile_5;
        mbarrier_wait(sfa_free_addr + (stage_4) * 8, _phase_sfa_free);
        if (elect_sync()) {
          asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::"
                       "complete_tx::bytes.L2::cache_hint"
                       " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(smem_sfa_addr +
                                                                stage_4 * 1024),
                       "l"((&SFA)), "r"(0), "r"(row_tile * 5 * 2),
                       "r"(sfa_full_addr + (stage_4) * 8),
                       "l"(0x12F0000000000000ULL)
                       : "memory");
          asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::"
                       "complete_tx::bytes.L2::cache_hint"
                       " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(
                           smem_sfa_addr + stage_4 * 1024 + 512),
                       "l"((&SFA)), "r"(0), "r"((row_tile * 5 + 1) * 2),
                       "r"(sfa_full_addr + (stage_4) * 8),
                       "l"(0x12F0000000000000ULL)
                       : "memory");
          mbarrier_arrive_expect_tx(sfa_full_addr + (stage_4) * 8, 1024);
        }
        stage_4 += 1;
        if (stage_4 == 5) {
          stage_4 = 0;
          _phase_sfa_free ^= 1;
        }
        mbarrier_wait(sfa_free_addr + (stage_4) * 8, _phase_sfa_free);
        if (elect_sync()) {
          asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::"
                       "complete_tx::bytes.L2::cache_hint"
                       " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(smem_sfa_addr +
                                                                stage_4 * 1024),
                       "l"((&SFA)), "r"(0), "r"((row_tile * 5 + 2) * 2),
                       "r"(sfa_full_addr + (stage_4) * 8),
                       "l"(0x12F0000000000000ULL)
                       : "memory");
          asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::"
                       "complete_tx::bytes.L2::cache_hint"
                       " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(
                           smem_sfa_addr + stage_4 * 1024 + 512),
                       "l"((&SFA)), "r"(0), "r"((row_tile * 5 + 2 + 1) * 2),
                       "r"(sfa_full_addr + (stage_4) * 8),
                       "l"(0x12F0000000000000ULL)
                       : "memory");
          mbarrier_arrive_expect_tx(sfa_full_addr + (stage_4) * 8, 1024);
        }
        stage_4 += 1;
        if (stage_4 == 5) {
          stage_4 = 0;
          _phase_sfa_free ^= 1;
        }
        mbarrier_wait(sfa_free_addr + (stage_4) * 8, _phase_sfa_free);
        if (elect_sync()) {
          asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::"
                       "complete_tx::bytes.L2::cache_hint"
                       " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(smem_sfa_addr +
                                                                stage_4 * 1024),
                       "l"((&SFA)), "r"(0), "r"((row_tile * 5 + 4) * 2),
                       "r"(sfa_full_addr + (stage_4) * 8),
                       "l"(0x12F0000000000000ULL)
                       : "memory");
          mbarrier_arrive_expect_tx(sfa_full_addr + (stage_4) * 8, 512);
        }
        stage_4 += 1;
        if (stage_4 == 5) {
          stage_4 = 0;
          _phase_sfa_free ^= 1;
        }
        mbarrier_wait(work_full_addr + (work_stage_5) * 8, _phase_work_full_5);
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
                     : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                     : "memory");
        unsigned int valid_6 = _clc_valid_1;
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
            : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
            : "memory");
        unsigned int next_x_5 = _clc_ctaid_2;
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
            : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
            : "memory");
        unsigned int next_y_5 = _clc_ctaid_3;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbarrier_arrive(work_empty_addr + (work_stage_5) * 8);
        work_stage_5 += 1;
        if (work_stage_5 == 3) {
          work_stage_5 = 0;
          _phase_work_full_5 ^= 1;
        }
        unsigned int valid_0_3 = valid_6;
        m_tile_5 = next_x_5;
        n_tile_5 = next_y_5;
        if (valid_0_3 == 0) {
          break;
        }
      }
    }
  }

  if (warp == 13) {
    {
      unsigned int stage_5 = 0;
      unsigned int work_stage_6 = 0;
      unsigned int m_tile_6 = blockIdx.x;
      unsigned int n_tile_6 = blockIdx.y;
      unsigned int _phase_sfa_full = 0;
      unsigned int _phase_k_done_3 = 1;
      unsigned int _phase_work_full_6 = 0;
#pragma unroll 1
      for (unsigned int _unit_6 = 0; _unit_6 < 2560; _unit_6++) {
        if (n_tile_6 >= (unsigned int)route[1]) {
          break;
        }
        mbarrier_wait(sfa_full_addr + (stage_5) * 8, _phase_sfa_full);
        mbarrier_wait(k_done_addr + (stage_5) * 8, _phase_k_done_3);
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (elect_sync()) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
#error "Tcgen05Cp requires Blackwell tcgen05.cp support"
#endif
          {
            uint64_t _tcgen05_cp_desc_0 =
                ((((uint64_t)(smem_sfa_addr + stage_5 * 1024)) & 0x3FFFFULL) >>
                 4ULL) |
                (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) |
                (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) |
                (1ULL << 46ULL) | (0ULL << 61ULL);
            asm volatile(
                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(
                    (uint32_t)((unsigned int)tmem_sfa + stage_5 * 8)),
                "l"(_tcgen05_cp_desc_0)
                : "memory");
          }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
#error "Tcgen05Cp requires Blackwell tcgen05.cp support"
#endif
          {
            uint64_t _tcgen05_cp_desc_1 =
                ((((uint64_t)(smem_sfa_addr + stage_5 * 1024 + 512)) &
                  0x3FFFFULL) >>
                 4ULL) |
                (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) |
                (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) |
                (1ULL << 46ULL) | (0ULL << 61ULL);
            asm volatile(
                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(
                    (uint32_t)((unsigned int)tmem_sfa + (stage_5 * 8 + 4))),
                "l"(_tcgen05_cp_desc_1)
                : "memory");
          }
        }
        elect_commit2(tmem_sfa_full_addr + (stage_5) * 8,
                      sfa_free_addr + (stage_5) * 8);
        stage_5 += 1;
        if (stage_5 == 5) {
          stage_5 = 0;
          _phase_sfa_full ^= 1;
          _phase_k_done_3 ^= 1;
        }
        mbarrier_wait(sfa_full_addr + (stage_5) * 8, _phase_sfa_full);
        mbarrier_wait(k_done_addr + (stage_5) * 8, _phase_k_done_3);
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (elect_sync()) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
#error "Tcgen05Cp requires Blackwell tcgen05.cp support"
#endif
          {
            uint64_t _tcgen05_cp_desc_2 =
                ((((uint64_t)(smem_sfa_addr + stage_5 * 1024)) & 0x3FFFFULL) >>
                 4ULL) |
                (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) |
                (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) |
                (1ULL << 46ULL) | (0ULL << 61ULL);
            asm volatile(
                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(
                    (uint32_t)((unsigned int)tmem_sfa + stage_5 * 8)),
                "l"(_tcgen05_cp_desc_2)
                : "memory");
          }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
#error "Tcgen05Cp requires Blackwell tcgen05.cp support"
#endif
          {
            uint64_t _tcgen05_cp_desc_3 =
                ((((uint64_t)(smem_sfa_addr + stage_5 * 1024 + 512)) &
                  0x3FFFFULL) >>
                 4ULL) |
                (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) |
                (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) |
                (1ULL << 46ULL) | (0ULL << 61ULL);
            asm volatile(
                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(
                    (uint32_t)((unsigned int)tmem_sfa + (stage_5 * 8 + 4))),
                "l"(_tcgen05_cp_desc_3)
                : "memory");
          }
        }
        elect_commit2(tmem_sfa_full_addr + (stage_5) * 8,
                      sfa_free_addr + (stage_5) * 8);
        stage_5 += 1;
        if (stage_5 == 5) {
          stage_5 = 0;
          _phase_sfa_full ^= 1;
          _phase_k_done_3 ^= 1;
        }
        mbarrier_wait(sfa_full_addr + (stage_5) * 8, _phase_sfa_full);
        mbarrier_wait(k_done_addr + (stage_5) * 8, _phase_k_done_3);
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (elect_sync()) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
#error "Tcgen05Cp requires Blackwell tcgen05.cp support"
#endif
          {
            uint64_t _tcgen05_cp_desc_4 =
                ((((uint64_t)(smem_sfa_addr + stage_5 * 1024)) & 0x3FFFFULL) >>
                 4ULL) |
                (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) |
                (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) |
                (1ULL << 46ULL) | (0ULL << 61ULL);
            asm volatile(
                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(
                    (uint32_t)((unsigned int)tmem_sfa + stage_5 * 8)),
                "l"(_tcgen05_cp_desc_4)
                : "memory");
          }
        }
        elect_commit2(tmem_sfa_full_addr + (stage_5) * 8,
                      sfa_free_addr + (stage_5) * 8);
        stage_5 += 1;
        if (stage_5 == 5) {
          stage_5 = 0;
          _phase_sfa_full ^= 1;
          _phase_k_done_3 ^= 1;
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
        unsigned int valid_7 = _clc_valid_4;
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
        unsigned int valid_0_4 = valid_7;
        m_tile_6 = next_x_6;
        n_tile_6 = next_y_6;
        if (valid_0_4 == 0) {
          break;
        }
      }
    }
  }

  if (warp == 14) {
    {
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
      for (unsigned int _unit_7 = 0; _unit_7 < 2560; _unit_7++) {
        if (n_tile_7 >= (unsigned int)route[1]) {
          break;
        }
        mbarrier_wait(mma_free_addr + (acc_stage_1) * 8, _phase_mma_free);
        mbarrier_wait(a_full_addr + (stage_6) * 8, _phase_a_full);
        mbarrier_wait(b_full_addr + (stage_6) * 8, _phase_b_full);
        mbarrier_wait(tmem_sfa_full_addr + (stage_6) * 8, _phase_tmem_sfa_full);
        mbarrier_wait(tmem_sfb_full_addr + (stage_6) * 8, _phase_tmem_sfb_full);
        asm volatile("tcgen05.fence::after_thread_sync;");
        int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) +
                                            (stage_6) * 2048);
        int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) +
                                            (stage_6) * 128);
        if (elect_sync()) {
          {
            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) |
                              ((uint64_t)0x40004040 << 32);
            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) |
                              ((uint64_t)0x40004040 << 32);

            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 0,
                                b_desc + 0, 0x8820280U,
                                (unsigned int)tmem_sfa + stage_6 * 8,
                                (unsigned int)tmem_sfb + stage_6 * 4,
                                ((((1) ? 1 : 0)) ? 0 : 1));
            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 2,
                                b_desc + 2, 0x28820290U,
                                (unsigned int)tmem_sfa + stage_6 * 8,
                                (unsigned int)tmem_sfb + stage_6 * 4, 1);
            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 4,
                                b_desc + 4, 0x488202a0U,
                                (unsigned int)tmem_sfa + stage_6 * 8,
                                (unsigned int)tmem_sfb + stage_6 * 4, 1);
            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 6,
                                b_desc + 6, 0x688202b0U,
                                (unsigned int)tmem_sfa + stage_6 * 8,
                                (unsigned int)tmem_sfb + stage_6 * 4, 1);
          }
        }
        int _mma_a_lo_1 = make_warp_uniform(
            (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (stage_6) * 2048);
        int _mma_b_lo_1 = make_warp_uniform(
            (((smem_b_addr + 1024) >> 4) & 0x3FFF) + (stage_6) * 128);
        if (elect_sync()) {
          {
            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) |
                              ((uint64_t)0x40004040 << 32);
            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) |
                              ((uint64_t)0x40004040 << 32);

            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 0,
                                b_desc + 0, 0x8820280U,
                                (unsigned int)tmem_sfa + (stage_6 * 8 + 4),
                                (unsigned int)tmem_sfb + (stage_6 * 4 + 2),
                                ((((0) ? 1 : 0)) ? 0 : 1));
            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 2,
                                b_desc + 2, 0x28820290U,
                                (unsigned int)tmem_sfa + (stage_6 * 8 + 4),
                                (unsigned int)tmem_sfb + (stage_6 * 4 + 2), 1);
            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 4,
                                b_desc + 4, 0x488202a0U,
                                (unsigned int)tmem_sfa + (stage_6 * 8 + 4),
                                (unsigned int)tmem_sfb + (stage_6 * 4 + 2), 1);
            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 6,
                                b_desc + 6, 0x688202b0U,
                                (unsigned int)tmem_sfa + (stage_6 * 8 + 4),
                                (unsigned int)tmem_sfb + (stage_6 * 4 + 2), 1);
          }
        }
        {
          elect_commit(k_done_addr + (stage_6) * 8);
        }
        stage_6 += 1;
        if (stage_6 == 5) {
          stage_6 = 0;
          _phase_a_full ^= 1;
          _phase_b_full ^= 1;
          _phase_tmem_sfa_full ^= 1;
          _phase_tmem_sfb_full ^= 1;
        }
        mbarrier_wait(a_full_addr + (stage_6) * 8, _phase_a_full);
        mbarrier_wait(b_full_addr + (stage_6) * 8, _phase_b_full);
        mbarrier_wait(tmem_sfa_full_addr + (stage_6) * 8, _phase_tmem_sfa_full);
        mbarrier_wait(tmem_sfb_full_addr + (stage_6) * 8, _phase_tmem_sfb_full);
        asm volatile("tcgen05.fence::after_thread_sync;");
        int _mma_a_lo_2 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) +
                                            (stage_6) * 2048);
        int _mma_b_lo_2 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) +
                                            (stage_6) * 128);
        if (elect_sync()) {
          {
            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) |
                              ((uint64_t)0x40004040 << 32);
            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) |
                              ((uint64_t)0x40004040 << 32);

            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 0,
                                b_desc + 0, 0x8820280U,
                                (unsigned int)tmem_sfa + stage_6 * 8,
                                (unsigned int)tmem_sfb + stage_6 * 4,
                                ((((0) ? 1 : 0)) ? 0 : 1));
            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 2,
                                b_desc + 2, 0x28820290U,
                                (unsigned int)tmem_sfa + stage_6 * 8,
                                (unsigned int)tmem_sfb + stage_6 * 4, 1);
            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 4,
                                b_desc + 4, 0x488202a0U,
                                (unsigned int)tmem_sfa + stage_6 * 8,
                                (unsigned int)tmem_sfb + stage_6 * 4, 1);
            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 6,
                                b_desc + 6, 0x688202b0U,
                                (unsigned int)tmem_sfa + stage_6 * 8,
                                (unsigned int)tmem_sfb + stage_6 * 4, 1);
          }
        }
        int _mma_a_lo_3 = make_warp_uniform(
            (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (stage_6) * 2048);
        int _mma_b_lo_3 = make_warp_uniform(
            (((smem_b_addr + 1024) >> 4) & 0x3FFF) + (stage_6) * 128);
        if (elect_sync()) {
          {
            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) |
                              ((uint64_t)0x40004040 << 32);
            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) |
                              ((uint64_t)0x40004040 << 32);

            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 0,
                                b_desc + 0, 0x8820280U,
                                (unsigned int)tmem_sfa + (stage_6 * 8 + 4),
                                (unsigned int)tmem_sfb + (stage_6 * 4 + 2),
                                ((((0) ? 1 : 0)) ? 0 : 1));
            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 2,
                                b_desc + 2, 0x28820290U,
                                (unsigned int)tmem_sfa + (stage_6 * 8 + 4),
                                (unsigned int)tmem_sfb + (stage_6 * 4 + 2), 1);
            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 4,
                                b_desc + 4, 0x488202a0U,
                                (unsigned int)tmem_sfa + (stage_6 * 8 + 4),
                                (unsigned int)tmem_sfb + (stage_6 * 4 + 2), 1);
            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 6,
                                b_desc + 6, 0x688202b0U,
                                (unsigned int)tmem_sfa + (stage_6 * 8 + 4),
                                (unsigned int)tmem_sfb + (stage_6 * 4 + 2), 1);
          }
        }
        {
          elect_commit(k_done_addr + (stage_6) * 8);
        }
        stage_6 += 1;
        if (stage_6 == 5) {
          stage_6 = 0;
          _phase_a_full ^= 1;
          _phase_b_full ^= 1;
          _phase_tmem_sfa_full ^= 1;
          _phase_tmem_sfb_full ^= 1;
        }
        mbarrier_wait(a_full_addr + (stage_6) * 8, _phase_a_full);
        mbarrier_wait(b_full_addr + (stage_6) * 8, _phase_b_full);
        mbarrier_wait(tmem_sfa_full_addr + (stage_6) * 8, _phase_tmem_sfa_full);
        mbarrier_wait(tmem_sfb_full_addr + (stage_6) * 8, _phase_tmem_sfb_full);
        asm volatile("tcgen05.fence::after_thread_sync;");
        int _mma_a_lo_4 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) +
                                            (stage_6) * 2048);
        int _mma_b_lo_4 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) +
                                            (stage_6) * 128);
        if (elect_sync()) {
          {
            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) |
                              ((uint64_t)0x40004040 << 32);
            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) |
                              ((uint64_t)0x40004040 << 32);

            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 0,
                                b_desc + 0, 0x8820280U,
                                (unsigned int)tmem_sfa + stage_6 * 8,
                                (unsigned int)tmem_sfb + stage_6 * 4,
                                ((((0) ? 1 : 0)) ? 0 : 1));
            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 2,
                                b_desc + 2, 0x28820290U,
                                (unsigned int)tmem_sfa + stage_6 * 8,
                                (unsigned int)tmem_sfb + stage_6 * 4, 1);
            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 4,
                                b_desc + 4, 0x488202a0U,
                                (unsigned int)tmem_sfa + stage_6 * 8,
                                (unsigned int)tmem_sfb + stage_6 * 4, 1);
            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage_1 * 8)), a_desc + 6,
                                b_desc + 6, 0x688202b0U,
                                (unsigned int)tmem_sfa + stage_6 * 8,
                                (unsigned int)tmem_sfb + stage_6 * 4, 1);
          }
        }
        {
          elect_commit2(k_done_addr + (stage_6) * 8,
                        mma_full_addr + (acc_stage_1) * 8);
        }
        stage_6 += 1;
        if (stage_6 == 5) {
          stage_6 = 0;
          _phase_a_full ^= 1;
          _phase_b_full ^= 1;
          _phase_tmem_sfa_full ^= 1;
          _phase_tmem_sfb_full ^= 1;
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
        unsigned int valid_8 = _clc_valid_6;
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
        unsigned int valid_0_5 = valid_8;
        m_tile_7 = next_x_7;
        n_tile_7 = next_y_7;
        if (valid_0_5 == 0) {
          break;
        }
      }
    }
  }

  if (warp == 15) {
    {
      unsigned int work_stage_8 = 0;
      unsigned int throttle_stage_1 = 0;
      unsigned int m_tile_8 = blockIdx.x;
      unsigned int n_tile_8 = blockIdx.y;
      unsigned int _phase_drain_full = 0;
      unsigned int _phase_throttle_full = 0;
      unsigned int _phase_work_empty = 1;
      unsigned int _phase_work_full_8 = 0;
#pragma unroll 1
      for (unsigned int _unit_8 = 0; _unit_8 < 2560; _unit_8++) {
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
        unsigned int valid_9 = _clc_valid_12;
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
        unsigned int valid_0_6 = valid_9;
        m_tile_8 = next_x_8;
        n_tile_8 = next_y_8;
        if (valid_0_6 == 0) {
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
