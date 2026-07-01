// MoE LoRA-A "shrink" grouped GEMM (SPLIT_K == 1): warp-level m16n8k16
// tensor cores + cp.async multi-stage pipeline. CUDA port of
// _moe_lora_shrink_splitk_kernel (sglang/srt/lora/triton_ops/virtual_experts.py),
// specialized for rank 16/32/64 and K divisible by 256.
//
//     output[t, n] = sum_k  hidden_states[t // top_k, k] * lora_a[expert(t), n, k]
//
// One block owns one expert token-block (BLOCK_M=16). Each active warp computes
// one rank-16 x token-8 tile, mapping rank to MMA-M and routed tokens to MMA-N
// so decode-like blocks with 2-8 tokens/expert avoid the old 16-token WMMA
// granularity. The gathered hidden rows and LoRA-A rows are streamed through a
// KSTAGES-deep cp.async ring buffer with 128-bit copies.
//
// Layout (row-major, K-contiguous): stride_ak == stride_bk == 1.
//   hidden_states : [num_tokens_M, K]
//   lora_a        : [num_virtual_experts, N, K]
//   output        : [num_tokens_M * top_k, N]   written in place, routed rows only

#include <sgl_kernel/utils.h>  // For RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, fp16_t/bf16_t

#include <tvm/ffi/container/tensor.h>  // For tvm::ffi::TensorView

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <type_traits>

using namespace nvcuda;

namespace {

constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int BLOCK_M = 16;     // one WMMA M-tile per block (routing block_size must be 16)
constexpr int BLOCK_K = 256;    // K streamed per stage (multiple of WMMA_K); few tiles -> few serial stages
constexpr int BLOCK_WARPS = 4;  // 128 threads; warp w (< num_n_tiles) owns N-tile w
constexpr int KSTAGES = 3;      // cp.async ring-buffer depth
constexpr int VEC = 8;          // 8 x 16-bit = 16B vectorized copy
constexpr int MAX_N = BLOCK_WARPS * WMMA_N;  // 64
constexpr int SWAP_TOKEN_N = 8;
constexpr int SWAP_WARPS = 8;
constexpr int SWAP_THREADS = SWAP_WARPS * 32;
constexpr int TCGEN_THREADS = 128;
constexpr int TCGEN_M = 64;
constexpr int TCGEN_N = 8;
constexpr int TCGEN_BLOCK_K = 256;
constexpr int TCGEN_GROUP_KTILES = 2;
constexpr int TCGEN_ALLOC_COLS = 32;

__host__ __device__ __forceinline__ size_t align16(size_t n) {
  return (n + 15) & ~size_t(15);
}
__host__ __device__ __forceinline__ size_t align1024(size_t n) {
  return (n + 1023) & ~size_t(1023);
}

__device__ __forceinline__ void cp_async_16(void* dst_smem, const void* src_global, int src_bytes) {
#if __CUDA_ARCH__ >= 800
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst_smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src_global), "r"(src_bytes));
#else
  char* d = static_cast<char*>(dst_smem);
  const char* g = static_cast<const char*>(src_global);
#pragma unroll
  for (int i = 0; i < 16; ++i)
    d[i] = (i < src_bytes) ? g[i] : char(0);
#endif
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&regs)[4], const void* smem_ptr) {
#if __CUDA_ARCH__ >= 800
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
               : "r"(s));
#endif
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t (&regs)[2], const void* smem_ptr) {
#if __CUDA_ARCH__ >= 800
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(regs[0]), "=r"(regs[1]) : "r"(s));
#endif
}

template <typename scalar_t>
__device__ __forceinline__ void mma_m16n8k16(const uint32_t (&a_frag)[4], const uint32_t (&b_frag)[2], float (&c)[4]) {
#if __CUDA_ARCH__ >= 800
  if constexpr (std::is_same_v<scalar_t, fp16_t>) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a_frag[0]),
          "r"(a_frag[1]),
          "r"(a_frag[2]),
          "r"(a_frag[3]),
          "r"(b_frag[0]),
          "r"(b_frag[1]),
          "f"(c[0]),
          "f"(c[1]),
          "f"(c[2]),
          "f"(c[3]));
  } else if constexpr (std::is_same_v<scalar_t, bf16_t>) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a_frag[0]),
          "r"(a_frag[1]),
          "r"(a_frag[2]),
          "r"(a_frag[3]),
          "r"(b_frag[0]),
          "r"(b_frag[1]),
          "f"(c[0]),
          "f"(c[1]),
          "f"(c[2]),
          "f"(c[3]));
  }
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n");
#endif
}

template <int kKeep>
__device__ __forceinline__ void cp_async_wait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kKeep));
#endif
}

__device__ __forceinline__ uint64_t tcgen_desc_encode(uint32_t smem_addr) {
  return static_cast<uint64_t>((smem_addr >> 4) & 0x3fff);
}

__device__ __forceinline__ uint64_t tcgen_make_smem_desc(uint32_t smem_addr) {
  constexpr uint64_t kSbo = 8 * 128;
  return tcgen_desc_encode(smem_addr) | (tcgen_desc_encode(kSbo) << 32) | (1ULL << 46) | (2ULL << 61);
}

__device__ __forceinline__ uint32_t tcgen_swizzle_128b_offset(int row, int k_elem, int rows) {
  const int k64 = k_elem >> 6;
  const int seg = (k_elem >> 3) & 7;
  const int row_group = row >> 3;
  const int row_in_group = row & 7;
  return static_cast<uint32_t>(
      k64 * rows * 128 + row_group * 8 * 128 + row_in_group * 128 + ((seg ^ row_in_group) << 4));
}

__device__ __forceinline__ uint32_t elect_sync_u32() {
  uint32_t pred = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "{\n\t"
      ".reg .pred %%p;\n\t"
      "elect.sync _|%%p, %1;\n\t"
      "@%%p mov.u32 %0, 1;\n\t"
      "}\n"
      : "+r"(pred)
      : "r"(0xffffffff));
#endif
  return pred;
}

__device__ __forceinline__ void mbarrier_init_shared(uint64_t* mbar, uint32_t arrives) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_addr), "r"(arrives) : "memory");
#endif
}

__device__ __forceinline__ void mbarrier_init_fence() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void mbarrier_wait_shared(uint64_t* mbar, uint32_t phase) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  uint32_t complete = 0;
  do {
    asm volatile(
        "{\n\t"
        ".reg .pred %%p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 %%p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, %%p;\n\t"
        "}\n"
        : "=r"(complete)
        : "r"(smem_addr), "r"(phase)
        : "memory");
  } while (!complete);
#endif
}

__device__ __forceinline__ void tcgen_alloc(uint32_t* tmem_addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(tmem_addr));
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(smem_addr),
               "r"(TCGEN_ALLOC_COLS)
               : "memory");
#endif
}

__device__ __forceinline__ void tcgen_relinquish_alloc_permit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void tcgen_dealloc(uint32_t tmem_addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tmem_addr), "r"(TCGEN_ALLOC_COLS)
               : "memory");
#endif
}

__device__ __forceinline__ void tcgen_fence_after_thread_sync() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void
tcgen_mma_f16(uint32_t tmem_addr, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, uint32_t enable_input_d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile(
      "{\n\t"
      ".reg .pred %%p;\n\t"
      "setp.ne.u32 %%p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %%p;\n\t"
      "}\n"
      :
      : "r"(tmem_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(enable_input_d)
      : "memory");
#endif
}

__device__ __forceinline__ void tcgen_commit(uint64_t* mbar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(smem_addr)
               : "memory");
#endif
}

__device__ __forceinline__ void tcgen_ld_32x32b_x8(float (&vals)[8], uint32_t tmem_addr, int row, int col) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  const uint32_t addr = tmem_addr + (static_cast<uint32_t>(row) << 16) + static_cast<uint32_t>(col);
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
      "tcgen05.wait::ld.sync.aligned;\n"
      : "=f"(vals[0]),
        "=f"(vals[1]),
        "=f"(vals[2]),
        "=f"(vals[3]),
        "=f"(vals[4]),
        "=f"(vals[5]),
        "=f"(vals[6]),
        "=f"(vals[7])
      : "r"(addr)
      : "memory");
#endif
}

__device__ __forceinline__ void atomic_add_bf16(bf16_t* addr, float val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  atomicAdd(addr, __float2bfloat16(val));
#endif
}

// Programmatic Dependent Launch (PDL, sm_90+). When the launch sets
// cudaLaunchAttributeProgrammaticStreamSerialization, the dependent grid may
// begin its preamble (grid setup, arg loads) while the prior grid drains.
// pdl_wait waits for the prior grid's trigger before this grid's dependent work;
// pdl_trigger releases the next grid. No-ops if the launch didn't opt in.
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
}

__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Issue cp.async for one K-tile (offset k0) into a stage's A/B buffers.
//   As: [BLOCK_M][BLOCK_K] gathered hidden rows (invalid token -> zero-fill)
//   Bs: [N][BLOCK_K]       lora_a[expert] rows (contiguous in k)
template <typename scalar_t>
__device__ __forceinline__ void issue_stage(
    scalar_t* As,
    scalar_t* Bs,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const int32_t* tok,
    int64_t b_base,
    int N,
    int N_PAD,
    int num_valid_tokens,
    int top_k,
    int64_t stride_am,
    int64_t stride_bn,
    int k0,
    int tid,
    int nthreads) {
  const int nA = BLOCK_M * (BLOCK_K / VEC);
  for (int cc = tid; cc < nA; cc += nthreads) {
    const int m = cc / (BLOCK_K / VEC);
    const int kc = (cc % (BLOCK_K / VEC)) * VEC;
    const int t = tok[m];
    const scalar_t* src = a;
    int src_bytes = 0;
    if (t < num_valid_tokens) {
      const int64_t row = static_cast<int64_t>(t) / top_k;
      src = a + row * stride_am + (k0 + kc);
      src_bytes = 16;
    }
    cp_async_16(&As[m * BLOCK_K + kc], src, src_bytes);
  }
  const int nB = N_PAD * (BLOCK_K / VEC);
  for (int cc = tid; cc < nB; cc += nthreads) {
    const int n = cc / (BLOCK_K / VEC);
    const int kc = (cc % (BLOCK_K / VEC)) * VEC;
    const scalar_t* src = b;
    int src_bytes = 0;
    if (n < N) {
      src = b + b_base + static_cast<int64_t>(n) * stride_bn + (k0 + kc);
      src_bytes = 16;
    }
    cp_async_16(&Bs[n * BLOCK_K + kc], src, src_bytes);
  }
}

template <typename scalar_t, int Rank, int Threads>
__device__ __forceinline__ void issue_swap_stage(
    scalar_t* Ash,
    scalar_t* Bsh,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const int32_t* tok,
    int64_t b_base,
    int num_valid_tokens,
    int top_k,
    int64_t stride_am,
    int64_t stride_bn,
    int k0,
    int tid) {
  constexpr int kKVecs = BLOCK_K / VEC;
  constexpr int nA = Rank * kKVecs;
  for (int cc = tid; cc < nA; cc += Threads) {
    const int n = cc / (BLOCK_K / VEC);
    const int kc = (cc % (BLOCK_K / VEC)) * VEC;
    cp_async_16(&Ash[n * BLOCK_K + kc], b + b_base + static_cast<int64_t>(n) * stride_bn + (k0 + kc), 16);
  }
  constexpr int nB = BLOCK_M * kKVecs;
  for (int cc = tid; cc < nB; cc += Threads) {
    const int m = cc / (BLOCK_K / VEC);
    const int kc = (cc % (BLOCK_K / VEC)) * VEC;
    const int t = tok[m];
    const scalar_t* src = a;
    int src_bytes = 0;
    if (t < num_valid_tokens) {
      const int64_t row = static_cast<int64_t>(t) / top_k;
      src = a + row * stride_am + (k0 + kc);
      src_bytes = 16;
    }
    cp_async_16(&Bsh[m * BLOCK_K + kc], src, src_bytes);
  }
}

template <typename scalar_t, int Rank, int Threads>
__device__ __forceinline__ void issue_swap_half_stage(
    scalar_t* Ash,
    scalar_t* Bsh,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const int32_t* tok,
    int token_base,
    int64_t b_base,
    int num_valid_tokens,
    int top_k,
    int64_t stride_am,
    int64_t stride_bn,
    int k0,
    int tid) {
  constexpr int kKVecs = TCGEN_BLOCK_K / VEC;
  constexpr int nA = Rank * kKVecs;
  for (int cc = tid; cc < nA; cc += Threads) {
    const int n = cc / kKVecs;
    const int kc = (cc % kKVecs) * VEC;
    cp_async_16(&Ash[n * BLOCK_K + kc], b + b_base + static_cast<int64_t>(n) * stride_bn + (k0 + kc), 16);
  }
  constexpr int nB = SWAP_TOKEN_N * kKVecs;
  for (int cc = tid; cc < nB; cc += Threads) {
    const int local_m = cc / kKVecs;
    const int kc = (cc % kKVecs) * VEC;
    const int t = tok[token_base + local_m];
    const scalar_t* src = a;
    int src_bytes = 0;
    if (t < num_valid_tokens) {
      const int64_t row = static_cast<int64_t>(t) / top_k;
      src = a + row * stride_am + (k0 + kc);
      src_bytes = 16;
    }
    cp_async_16(&Bsh[local_m * BLOCK_K + kc], src, src_bytes);
  }
}

template <typename scalar_t>
__global__ void __launch_bounds__(BLOCK_WARPS * 32) moe_lora_shrink_wmma_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    int N,
    int K,
    int num_valid_tokens,
    int top_k,
    int64_t stride_am,
    int64_t stride_be,
    int64_t stride_bn,
    int64_t stride_bk,
    int64_t stride_cm,
    int64_t stride_cn) {
  const int pid_m = blockIdx.x;
  if (pid_m * BLOCK_M >= num_tokens_post_padded[0]) return;
  const int expert = expert_ids[pid_m];
  if (expert == -1) return;

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int nthreads = BLOCK_WARPS * 32;
  const int N_PAD = static_cast<int>(align16(static_cast<size_t>(N)));
  const int num_n_tiles = N_PAD / WMMA_N;

  // Dynamic shared (sized by padded N host-side): tok | Ash ring | Bsh ring | Csh.
  extern __shared__ __align__(16) char smem[];
  const int a_stage = BLOCK_M * BLOCK_K;
  const int b_stage = N_PAD * BLOCK_K;
  char* p = smem;
  int32_t* tok = reinterpret_cast<int32_t*>(p);
  p += align16(BLOCK_M * sizeof(int32_t));
  scalar_t* Ash = reinterpret_cast<scalar_t*>(p);
  p += align16(static_cast<size_t>(KSTAGES) * a_stage * sizeof(scalar_t));
  scalar_t* Bsh = reinterpret_cast<scalar_t*>(p);
  p += align16(static_cast<size_t>(KSTAGES) * b_stage * sizeof(scalar_t));
  float* Csh = reinterpret_cast<float*>(p);

  for (int i = tid; i < BLOCK_M; i += nthreads)
    tok[i] = sorted_token_ids[pid_m * BLOCK_M + i];
  __syncthreads();

  const int64_t b_base = static_cast<int64_t>(expert) * stride_be;
  const int num_k_tiles = K / BLOCK_K;  // K % BLOCK_K == 0 (checked host-side)

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  // Prologue: prefetch the first (KSTAGES - 1) K-tiles.
#pragma unroll
  for (int s = 0; s < KSTAGES - 1; ++s) {
    if (s < num_k_tiles) {
      issue_stage(
          Ash + s * a_stage,
          Bsh + s * b_stage,
          a,
          b,
          tok,
          b_base,
          N,
          N_PAD,
          num_valid_tokens,
          top_k,
          stride_am,
          stride_bn,
          s * BLOCK_K,
          tid,
          nthreads);
    }
    cp_async_commit();
  }

  for (int kt = 0; kt < num_k_tiles; ++kt) {
    cp_async_wait<KSTAGES - 2>();
    __syncthreads();

    const int cur = kt % KSTAGES;
    if (warp < num_n_tiles) {
      const int n0 = warp * WMMA_N;
      const scalar_t* As_cur = Ash + cur * a_stage;
      const scalar_t* Bs_cur = Bsh + cur * b_stage;
#pragma unroll
      for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, scalar_t, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, scalar_t, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag, As_cur + kk, BLOCK_K);
        wmma::load_matrix_sync(b_frag, Bs_cur + n0 * BLOCK_K + kk, BLOCK_K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
    }
    __syncthreads();  // done reading `cur`; safe to refill it

    const int load_idx = kt + KSTAGES - 1;
    if (load_idx < num_k_tiles) {
      const int nb = load_idx % KSTAGES;
      issue_stage(
          Ash + nb * a_stage,
          Bsh + nb * b_stage,
          a,
          b,
          tok,
          b_base,
          N,
          N_PAD,
          num_valid_tokens,
          top_k,
          stride_am,
          stride_bn,
          load_idx * BLOCK_K,
          tid,
          nthreads);
    }
    cp_async_commit();
  }

  if (warp < num_n_tiles) {
    wmma::store_matrix_sync(&Csh[warp * WMMA_M * WMMA_N], c_frag, WMMA_N, wmma::mem_row_major);
  }
  __syncthreads();

  for (int w = 0; w < num_n_tiles; ++w) {
    const int wn0 = w * WMMA_N;
    for (int idx = tid; idx < WMMA_M * WMMA_N; idx += nthreads) {
      const int m = idx / WMMA_N;
      const int n = idx % WMMA_N;
      const int t = tok[m];
      if (t < num_valid_tokens && (wn0 + n) < N) {
        c[static_cast<int64_t>(t) * stride_cm + static_cast<int64_t>(wn0 + n) * stride_cn] =
            static_cast<scalar_t>(Csh[w * WMMA_M * WMMA_N + m * WMMA_N + n]);
      }
    }
  }
}

template <typename scalar_t, int Rank, int Threads, int NumKTiles>
__global__ void __launch_bounds__(Threads) moe_lora_shrink_m16n8_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    int K,
    int num_valid_tokens,
    int top_k,
    int64_t stride_am,
    int64_t stride_be,
    int64_t stride_bn,
    int64_t stride_bk,
    int64_t stride_cm,
    int64_t stride_cn) {
  const int pid_m = blockIdx.x;
  if (pid_m * BLOCK_M >= num_tokens_post_padded[0]) return;
  const int expert = expert_ids[pid_m];
  if (expert == -1) return;

  const int warp = threadIdx.x >> 5;
  const int rank_tile = warp >> 1;
  const int token_tile = warp & 1;
  const int n0 = rank_tile * WMMA_M;
  const int m0 = token_tile * SWAP_TOKEN_N;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  constexpr int RankTiles = Rank / WMMA_M;
  const bool compute_warp = warp < RankTiles * 2;

  extern __shared__ __align__(16) char smem[];
  int32_t* tok = reinterpret_cast<int32_t*>(smem);
  char* p = smem + align16(BLOCK_M * sizeof(int32_t));
  constexpr int a_stage = Rank * BLOCK_K;
  constexpr int b_stage = BLOCK_M * BLOCK_K;
  scalar_t* Ash = reinterpret_cast<scalar_t*>(p);  // [stage][N rank][BLOCK_K]
  p += align16(static_cast<size_t>(KSTAGES) * a_stage * sizeof(scalar_t));
  scalar_t* Bsh = reinterpret_cast<scalar_t*>(p);  // [stage][16 token][BLOCK_K]

  for (int i = tid; i < BLOCK_M; i += Threads)
    tok[i] = sorted_token_ids[pid_m * BLOCK_M + i];
  __syncthreads();

  // If the first token in this 8-wide half is padding, all later tokens in the
  // routed block are padding too.
  if (tok[0] >= num_valid_tokens) return;
  const bool compute_tile = compute_warp;

  const int64_t b_base = static_cast<int64_t>(expert) * stride_be;

  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int a_ld_row = (lane & 7) + ((lane & 8) ? 8 : 0);
  const int a_ld_col = (lane & 16) ? 8 : 0;
  const int b_ld_row = lane & 7;
  const int b_ld_col = (lane & 8) ? 8 : 0;

  const int num_k_tiles = NumKTiles > 0 ? NumKTiles : K / BLOCK_K;
#pragma unroll
  for (int s = 0; s < KSTAGES - 1; ++s) {
    if constexpr (NumKTiles > 0) {
      issue_swap_stage<scalar_t, Rank, Threads>(
          Ash + s * a_stage,
          Bsh + s * b_stage,
          a,
          b,
          tok,
          b_base,
          num_valid_tokens,
          top_k,
          stride_am,
          stride_bn,
          s * BLOCK_K,
          tid);
    } else {
      if (s < num_k_tiles) {
        issue_swap_stage<scalar_t, Rank, Threads>(
            Ash + s * a_stage,
            Bsh + s * b_stage,
            a,
            b,
            tok,
            b_base,
            num_valid_tokens,
            top_k,
            stride_am,
            stride_bn,
            s * BLOCK_K,
            tid);
      }
    }
    cp_async_commit();
  }

#pragma unroll
  for (int kt = 0; kt < num_k_tiles; ++kt) {
    cp_async_wait<KSTAGES - 2>();
    __syncthreads();

    const int cur = kt % KSTAGES;
    const scalar_t* As_cur = Ash + cur * a_stage;
    const scalar_t* Bs_cur = Bsh + cur * b_stage;
    if (compute_tile) {
#pragma unroll
      for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
        uint32_t a_frag[4];
        uint32_t b_frag[2];
        ldmatrix_x4(a_frag, As_cur + (n0 + a_ld_row) * BLOCK_K + kk + a_ld_col);
        ldmatrix_x2(b_frag, Bs_cur + (m0 + b_ld_row) * BLOCK_K + kk + b_ld_col);
        mma_m16n8k16<scalar_t>(a_frag, b_frag, acc);
      }
    }
    __syncthreads();

    const int load_idx = kt + KSTAGES - 1;
    if constexpr (NumKTiles > 0) {
      if (load_idx < NumKTiles) {
        const int nb = load_idx % KSTAGES;
        issue_swap_stage<scalar_t, Rank, Threads>(
            Ash + nb * a_stage,
            Bsh + nb * b_stage,
            a,
            b,
            tok,
            b_base,
            num_valid_tokens,
            top_k,
            stride_am,
            stride_bn,
            load_idx * BLOCK_K,
            tid);
      }
    } else if (load_idx < num_k_tiles) {
      const int nb = load_idx % KSTAGES;
      issue_swap_stage<scalar_t, Rank, Threads>(
          Ash + nb * a_stage,
          Bsh + nb * b_stage,
          a,
          b,
          tok,
          b_base,
          num_valid_tokens,
          top_k,
          stride_am,
          stride_bn,
          load_idx * BLOCK_K,
          tid);
    }
    cp_async_commit();
  }

  if (compute_tile) {
    const int rank_row0 = n0 + (lane >> 2);
    const int rank_row1 = rank_row0 + 8;
    const int token_col = m0 + ((lane & 3) << 1);

    const int t0 = tok[token_col];
    if (rank_row0 < Rank && token_col < BLOCK_M && t0 < num_valid_tokens) {
      c[static_cast<int64_t>(t0) * stride_cm + static_cast<int64_t>(rank_row0) * stride_cn] =
          static_cast<scalar_t>(acc[0]);
    }
    const int t1 = tok[token_col + 1];
    if (rank_row0 < Rank && token_col + 1 < BLOCK_M && t1 < num_valid_tokens) {
      c[static_cast<int64_t>(t1) * stride_cm + static_cast<int64_t>(rank_row0) * stride_cn] =
          static_cast<scalar_t>(acc[1]);
    }
    if (rank_row1 < Rank && token_col < BLOCK_M && t0 < num_valid_tokens) {
      c[static_cast<int64_t>(t0) * stride_cm + static_cast<int64_t>(rank_row1) * stride_cn] =
          static_cast<scalar_t>(acc[2]);
    }
    if (rank_row1 < Rank && token_col + 1 < BLOCK_M && t1 < num_valid_tokens) {
      c[static_cast<int64_t>(t1) * stride_cm + static_cast<int64_t>(rank_row1) * stride_cn] =
          static_cast<scalar_t>(acc[3]);
    }
  }
}

template <typename scalar_t, int Rank, int Threads, int NumKTiles>
__global__ void __launch_bounds__(Threads) moe_lora_shrink_m16n8_half_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    int K,
    int num_valid_tokens,
    int top_k,
    int64_t stride_am,
    int64_t stride_be,
    int64_t stride_bn,
    int64_t stride_bk,
    int64_t stride_cm,
    int64_t stride_cn) {
  const int pid_m = blockIdx.x >> 1;
  if (pid_m * BLOCK_M >= num_tokens_post_padded[0]) return;
  const int token_base = (blockIdx.x & 1) * SWAP_TOKEN_N;

  const int tid = threadIdx.x;
  extern __shared__ __align__(16) char smem[];
  int32_t* tok = reinterpret_cast<int32_t*>(smem);
  for (int i = tid; i < BLOCK_M; i += Threads)
    tok[i] = sorted_token_ids[pid_m * BLOCK_M + i];
  __syncthreads();

  if (tok[token_base] >= num_valid_tokens) return;
  const int expert = expert_ids[pid_m];
  if (expert == -1) return;

  const int warp = threadIdx.x >> 5;
  const int lane = tid & 31;
  constexpr int RankTiles = Rank / WMMA_M;
  const bool compute_tile = warp < RankTiles;
  const int n0 = warp * WMMA_M;

  char* p = smem + align16(BLOCK_M * sizeof(int32_t));
  constexpr int a_stage = Rank * BLOCK_K;
  constexpr int b_stage = SWAP_TOKEN_N * BLOCK_K;
  scalar_t* Ash = reinterpret_cast<scalar_t*>(p);  // [stage][N rank][BLOCK_K]
  p += align16(static_cast<size_t>(KSTAGES) * a_stage * sizeof(scalar_t));
  scalar_t* Bsh = reinterpret_cast<scalar_t*>(p);  // [stage][8 token][BLOCK_K]

  const int64_t b_base = static_cast<int64_t>(expert) * stride_be;

  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int a_ld_row = (lane & 7) + ((lane & 8) ? 8 : 0);
  const int a_ld_col = (lane & 16) ? 8 : 0;
  const int b_ld_row = lane & 7;
  const int b_ld_col = (lane & 8) ? 8 : 0;

  const int num_k_tiles = NumKTiles > 0 ? NumKTiles : K / BLOCK_K;
#pragma unroll
  for (int s = 0; s < KSTAGES - 1; ++s) {
    if constexpr (NumKTiles > 0) {
      issue_swap_half_stage<scalar_t, Rank, Threads>(
          Ash + s * a_stage,
          Bsh + s * b_stage,
          a,
          b,
          tok,
          token_base,
          b_base,
          num_valid_tokens,
          top_k,
          stride_am,
          stride_bn,
          s * BLOCK_K,
          tid);
    } else {
      if (s < num_k_tiles) {
        issue_swap_half_stage<scalar_t, Rank, Threads>(
            Ash + s * a_stage,
            Bsh + s * b_stage,
            a,
            b,
            tok,
            token_base,
            b_base,
            num_valid_tokens,
            top_k,
            stride_am,
            stride_bn,
            s * BLOCK_K,
            tid);
      }
    }
    cp_async_commit();
  }

#pragma unroll
  for (int kt = 0; kt < num_k_tiles; ++kt) {
    cp_async_wait<KSTAGES - 2>();
    __syncthreads();

    const int cur = kt % KSTAGES;
    const scalar_t* As_cur = Ash + cur * a_stage;
    const scalar_t* Bs_cur = Bsh + cur * b_stage;
    if (compute_tile) {
#pragma unroll
      for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
        uint32_t a_frag[4];
        uint32_t b_frag[2];
        ldmatrix_x4(a_frag, As_cur + (n0 + a_ld_row) * BLOCK_K + kk + a_ld_col);
        ldmatrix_x2(b_frag, Bs_cur + b_ld_row * BLOCK_K + kk + b_ld_col);
        mma_m16n8k16<scalar_t>(a_frag, b_frag, acc);
      }
    }
    __syncthreads();

    const int load_idx = kt + KSTAGES - 1;
    if constexpr (NumKTiles > 0) {
      if (load_idx < NumKTiles) {
        const int nb = load_idx % KSTAGES;
        issue_swap_half_stage<scalar_t, Rank, Threads>(
            Ash + nb * a_stage,
            Bsh + nb * b_stage,
            a,
            b,
            tok,
            token_base,
            b_base,
            num_valid_tokens,
            top_k,
            stride_am,
            stride_bn,
            load_idx * BLOCK_K,
            tid);
      }
    } else if (load_idx < num_k_tiles) {
      const int nb = load_idx % KSTAGES;
      issue_swap_half_stage<scalar_t, Rank, Threads>(
          Ash + nb * a_stage,
          Bsh + nb * b_stage,
          a,
          b,
          tok,
          token_base,
          b_base,
          num_valid_tokens,
          top_k,
          stride_am,
          stride_bn,
          load_idx * BLOCK_K,
          tid);
    }
    cp_async_commit();
  }

  if (compute_tile) {
    const int rank_row0 = n0 + (lane >> 2);
    const int rank_row1 = rank_row0 + 8;
    const int token_col = token_base + ((lane & 3) << 1);

    const int t0 = tok[token_col];
    if (rank_row0 < Rank && t0 < num_valid_tokens) {
      c[static_cast<int64_t>(t0) * stride_cm + static_cast<int64_t>(rank_row0) * stride_cn] =
          static_cast<scalar_t>(acc[0]);
    }
    const int t1 = tok[token_col + 1];
    if (rank_row0 < Rank && t1 < num_valid_tokens) {
      c[static_cast<int64_t>(t1) * stride_cm + static_cast<int64_t>(rank_row0) * stride_cn] =
          static_cast<scalar_t>(acc[1]);
    }
    if (rank_row1 < Rank && t0 < num_valid_tokens) {
      c[static_cast<int64_t>(t0) * stride_cm + static_cast<int64_t>(rank_row1) * stride_cn] =
          static_cast<scalar_t>(acc[2]);
    }
    if (rank_row1 < Rank && t1 < num_valid_tokens) {
      c[static_cast<int64_t>(t1) * stride_cm + static_cast<int64_t>(rank_row1) * stride_cn] =
          static_cast<scalar_t>(acc[3]);
    }
  }
}

// --------------------------------------------------------------------------
// Triton-style pipelined shrink: one block per expert token-block (16 tokens),
// 4 warps each owning an 8-wide rank N-tile (rank padded to PIPE_BLOCK_N=32),
// streaming K through a KStages-deep cp.async ring buffer. Mirrors the legacy
// _moe_lora_shrink_splitk_kernel (SPLIT_K==1, BLOCK_N=32, num_stages=4): plain
// m16n8k16 HMMA with tokens in the MMA-M slot and rank in the MMA-N slot so all
// four warps stay busy and the K-loop pipeline hides global-load latency.
// --------------------------------------------------------------------------
constexpr int PIPE_BLOCK_N = 32;  // rank padded to 32 -> 4 n-tiles of 8
constexpr int PIPE_WARPS = 4;     // one 8-wide rank N-tile per warp
constexpr int PIPE_THREADS = PIPE_WARPS * 32;

// 128-byte XOR swizzle for the K-contiguous smem tiles. Each row stores K in
// 8-element (16-byte) segments; within every 128-byte window (8 segments) the
// segment index is XORed by (row & 7) so the 8 rows an ldmatrix reads land in
// distinct shared-memory banks (otherwise stride-BlockK rows alias bank 0 and
// serialize -> the short-scoreboard stalls that dominate the naive layout).
template <int BlockK>
__device__ __forceinline__ int pipe_swz(int row, int k) {  // k a multiple of 8
  return row * BlockK + (((k >> 3) ^ (row & 7)) << 3);
}

template <typename scalar_t, int BlockK, int Rank, int NWarps>
__device__ __forceinline__ void issue_pipe_stage(
    scalar_t* As,
    scalar_t* Bs,
    const scalar_t* __restrict__ a,
    const scalar_t* const* arow,  // per-token A row base ptr (nullptr -> padding)
    const scalar_t* const* brow,  // per-rank B row base ptr
    int k0,
    int tid) {
  // Cheap addressing: warp g owns a strided set of rows, lane owns a K-vector.
  // No per-element divide/modulo or 64-bit pointer rebuild -- those integer ops
  // (IMAD/LOP3/SHF) dominated the naive cc/kKVecs gather.
  constexpr int kKVecs = BlockK / VEC;
  const int g = tid >> 5;
  const int lane = tid & 31;
  for (int kv = lane; kv < kKVecs; kv += 32) {
    const int kc = kv * VEC;
#pragma unroll
    for (int r = g; r < BLOCK_M; r += NWarps) {
      const scalar_t* base = arow[r];
      cp_async_16(&As[pipe_swz<BlockK>(r, kc)], base ? base + (k0 + kc) : a, base ? 16 : 0);
    }
#pragma unroll
    for (int r = g; r < Rank; r += NWarps) {
      cp_async_16(&Bs[pipe_swz<BlockK>(r, kc)], brow[r] + (k0 + kc), 16);
    }
  }
}

template <typename scalar_t, int KStages, int BlockK, int Rank, int NWarps, int SplitK>
__global__ void __launch_bounds__(NWarps * 32) moe_lora_shrink_pipe_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    float* __restrict__ workspace,  // [SplitK][num_valid_tokens][Rank] fp32 partials (SplitK>1)
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    int N,
    int K,
    int num_valid_tokens,
    int top_k,
    int64_t stride_am,
    int64_t stride_be,
    int64_t stride_bn,
    int64_t stride_bk,
    int64_t stride_cm,
    int64_t stride_cn) {
  const int pid_m = blockIdx.x;
  const int pid_sk = SplitK > 1 ? static_cast<int>(blockIdx.y) : 0;
  if (pid_m * BLOCK_M >= num_tokens_post_padded[0]) return;
  const int expert = expert_ids[pid_m];
  if (expert == -1) return;

  constexpr int kThreads = NWarps * 32;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  // Every warp computes ALL kNTiles rank N-tiles over its slice of the K-substeps
  // (the K-reduction is split across warps, not the N-tiles). This (a) keeps all
  // warps balanced at the per-tile __syncthreads -- idle load-helper warps were
  // the dominant barrier stall; and (b) lets the A fragment (token x k, shared
  // across N-tiles) be loaded ONCE per substep and reused, instead of reloaded
  // per N-tile -- redundant A ldmatrix was the dominant short-scoreboard stall.
  // The kNTiles independent MMAs per substep also supply the ILP that hides
  // ldmatrix latency. Per-warp partials are summed once in smem at the end.
  constexpr int kNTiles = Rank / 8;           // 2 (r16) / 4 (r32)
  constexpr int kSubsteps = BlockK / WMMA_K;  // K-substeps per tile (16 @ bk256)
  constexpr int kSubPerWarp = (kSubsteps >= NWarps) ? (kSubsteps / NWarps) : 1;

  extern __shared__ __align__(16) char smem[];
  int32_t* tok = reinterpret_cast<int32_t*>(smem);
  char* p = smem + align16(BLOCK_M * sizeof(int32_t));
  const scalar_t** arow = reinterpret_cast<const scalar_t**>(p);  // [16] A row base ptrs
  p += align16(BLOCK_M * sizeof(const scalar_t*));
  const scalar_t** brow = reinterpret_cast<const scalar_t**>(p);  // [Rank] B row base ptrs
  p += align16(Rank * sizeof(const scalar_t*));
  constexpr int a_stage = BLOCK_M * BlockK;
  constexpr int b_stage = Rank * BlockK;
  scalar_t* Ash = reinterpret_cast<scalar_t*>(p);  // [stage][16 token][BlockK]
  p += align16(static_cast<size_t>(KStages) * a_stage * sizeof(scalar_t));
  scalar_t* Bsh = reinterpret_cast<scalar_t*>(p);  // [stage][Rank][BlockK]
  p += align16(static_cast<size_t>(KStages) * b_stage * sizeof(scalar_t));
  float* redsmem = reinterpret_cast<float*>(p);  // [NWarps][32][kNTiles][4] partial-acc reduction

  // Load routing and precompute per-token A row base pointers ONCE (the t/top_k
  // map is loop-invariant; recomputing the runtime 64-bit divide per k-element
  // dominated the issue cost). Padding slots get a nullptr -> zero-filled.
  const int64_t b_base = static_cast<int64_t>(expert) * stride_be;
  for (int i = tid; i < BLOCK_M; i += kThreads) {
    const int t = sorted_token_ids[pid_m * BLOCK_M + i];
    tok[i] = t;
    arow[i] = (t < num_valid_tokens) ? (a + static_cast<int64_t>(t) / top_k * stride_am) : nullptr;
  }
  for (int i = tid; i < Rank; i += kThreads) {
    brow[i] = b + b_base + static_cast<int64_t>(i) * stride_bn;
  }
  __syncthreads();
  if (tok[0] >= num_valid_tokens) return;

  // PDL: none of this block's loads (hidden/lora_a/routing) depend on the prior
  // grid, so wait here only to honor stream ordering; the real win is letting the
  // next launch's preamble overlap this grid's tail (released by pdl_trigger).
  pdl_wait();

  // Split-K: this block owns a contiguous run of K-tiles [kt0, kt0+num_k_tiles).
  // num_k_tiles is kept RUNTIME on purpose: making it compile-time lets nvcc
  // unroll the K-loop, which spills the per-stage gather addresses and regresses
  // ~12%. The rolled loop reuses those registers across K-tiles.
  const int total_k_tiles = K / BlockK;
  const int num_k_tiles = total_k_tiles / SplitK;
  const int kt0 = pid_sk * num_k_tiles;
  const int kk_lo = warp * kSubPerWarp * WMMA_K;  // this warp's K-substep slice
  const bool active = kk_lo < BlockK;             // false only if NWarps > kSubsteps

  float acc[kNTiles][4];
#pragma unroll
  for (int nt = 0; nt < kNTiles; ++nt)
    acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
  const int a_ld_row = (lane & 7) + ((lane & 8) ? 8 : 0);
  const int a_ld_col = (lane & 16) ? 8 : 0;
  const int b_ld_row = lane & 7;
  const int b_ld_col = (lane & 8) ? 8 : 0;

#pragma unroll
  for (int s = 0; s < KStages - 1; ++s) {
    if (s < num_k_tiles) {
      issue_pipe_stage<scalar_t, BlockK, Rank, NWarps>(
          Ash + s * a_stage, Bsh + s * b_stage, a, arow, brow, (kt0 + s) * BlockK, tid);
    }
    cp_async_commit();
  }

  // Cooperative-load software pipeline: all warps fill the whole smem tile (best
  // global-load locality), the per-tile __syncthreads makes it visible, and the
  // next K-tile is issued BEFORE the MMA so cp.async overlaps the tensor cores.
  for (int kt = 0; kt < num_k_tiles; ++kt) {
    cp_async_wait<KStages - 2>();
    __syncthreads();

    const int load_idx = kt + KStages - 1;
    if (load_idx < num_k_tiles) {
      const int nb = load_idx % KStages;
      issue_pipe_stage<scalar_t, BlockK, Rank, NWarps>(
          Ash + nb * a_stage, Bsh + nb * b_stage, a, arow, brow, (kt0 + load_idx) * BlockK, tid);
    }
    cp_async_commit();

    if (active) {
      const scalar_t* As_cur = Ash + (kt % KStages) * a_stage;
      const scalar_t* Bs_cur = Bsh + (kt % KStages) * b_stage;
#pragma unroll
      for (int j = 0; j < kSubPerWarp; ++j) {
        const int kk = kk_lo + j * WMMA_K;
        uint32_t a_frag[4];
        ldmatrix_x4(a_frag, As_cur + pipe_swz<BlockK>(a_ld_row, kk + a_ld_col));  // shared across N-tiles
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
          uint32_t b_frag[2];
          ldmatrix_x2(b_frag, Bs_cur + pipe_swz<BlockK>(nt * 8 + b_ld_row, kk + b_ld_col));
          mma_m16n8k16<scalar_t>(a_frag, b_frag, acc[nt]);
        }
      }
    }
  }

  // The K-loop (all global loads + MMAs) is done; release the next grid so its
  // launch/preamble overlaps this grid's smem-reduction + store tail. All active
  // threads reach this before the warp>=kNTiles early-return below.
  pdl_trigger();

  // Sum each N-tile's per-substep partials across all NWarps warps (once, in smem).
  {
#pragma unroll
    for (int nt = 0; nt < kNTiles; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i)
        redsmem[((warp * 32 + lane) * kNTiles + nt) * 4 + i] = acc[nt][i];
    __syncthreads();
    if (warp >= kNTiles) return;  // kNTiles warps own the kNTiles N-tile stores
#pragma unroll
    for (int i = 0; i < 4; ++i)
      acc[0][i] = 0.f;
#pragma unroll
    for (int w = 0; w < NWarps; ++w)
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc[0][i] += redsmem[((w * 32 + lane) * kNTiles + warp) * 4 + i];
  }

  // C fragment of m16n8 (M=token, N=rank): rows {lane>>2, (lane>>2)+8}, cols {2*(lane&3), +1}.
  const int n0 = warp * 8;  // warp `w` (< kNTiles) stores N-tile w
  const int token_row0 = lane >> 2;
  const int token_row1 = token_row0 + 8;
  const int rank_col = n0 + ((lane & 3) << 1);
  const int t0 = tok[token_row0];
  const int t1 = tok[token_row1];
  auto store = [&](int t, int rank, float val) {
    if (t >= num_valid_tokens) return;
    if constexpr (SplitK == 1) {
      c[static_cast<int64_t>(t) * stride_cm + static_cast<int64_t>(rank) * stride_cn] = static_cast<scalar_t>(val);
    } else {
      // Split-K: write this split's fp32 partial; a tiny reduction kernel sums
      // the SplitK partials -> output. Inline bf16 atomic-add was measured ~2x
      // SLOWER here (emulated CAS + heavy contention: SplitK splits hammer only
      // num_valid_tokens*Rank locations, adjacent ranks sharing 32-bit words).
      // The contention-free fp32 staging + sequential reduce wins.
      workspace[(static_cast<int64_t>(pid_sk) * num_valid_tokens + t) * Rank + rank] = val;
    }
  };
  store(t0, rank_col, acc[0][0]);
  store(t1, rank_col, acc[0][2]);
  store(t0, rank_col + 1, acc[0][1]);
  store(t1, rank_col + 1, acc[0][3]);
}

// Sum the SplitK fp32 partials -> bf16/fp16 output. Tiny (num_valid_tokens*Rank).
template <typename scalar_t, int Rank, int SplitK>
__global__ void __launch_bounds__(256) moe_lora_shrink_splitk_reduce_kernel(
    const float* __restrict__ workspace,
    scalar_t* __restrict__ c,
    int num_valid_tokens,
    int64_t stride_cm,
    int64_t stride_cn) {
  const int total = num_valid_tokens * Rank;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
    const int t = i / Rank;
    const int rank = i - t * Rank;
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < SplitK; ++s)
      acc += workspace[(static_cast<int64_t>(s) * num_valid_tokens + t) * Rank + rank];
    c[static_cast<int64_t>(t) * stride_cm + static_cast<int64_t>(rank) * stride_cn] = static_cast<scalar_t>(acc);
  }
}

// Process-persistent fp32 scratch for split-K partials. Grown (cudaMalloc) on
// the host path during warm-up, so it is graph-capture safe at replay time.
inline float* get_shrink_workspace(size_t n_floats) {
  static float* ptr = nullptr;
  static size_t cap = 0;
  if (n_floats > cap) {
    if (ptr) cudaFree(ptr);
    host::RuntimeDeviceCheck(cudaMalloc(&ptr, n_floats * sizeof(float)));
    cap = n_floats;
  }
  return ptr;
}

template <typename scalar_t, int Rank, int NumKTiles>
__device__ __forceinline__ void issue_tcgen05_stage(
    char* Ash,
    char* Bsh,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const int32_t* tok,
    int token_base,
    int rank_base,
    int64_t b_base,
    int num_valid_tokens,
    int top_k,
    int64_t stride_am,
    int64_t stride_bn,
    int k0,
    int tid) {
  constexpr int kKVecs = BLOCK_K / VEC;
  for (int cc = tid; cc < TCGEN_M * kKVecs; cc += TCGEN_THREADS) {
    const int n = cc / kKVecs;
    const int kc = (cc % kKVecs) * VEC;
    const uint32_t dst = tcgen_swizzle_128b_offset(n, kc, TCGEN_M);
    if (n < WMMA_M && rank_base + n < Rank) {
      cp_async_16(Ash + dst, b + b_base + static_cast<int64_t>(rank_base + n) * stride_bn + (k0 + kc), 16);
    } else {
      cp_async_16(Ash + dst, b, 0);
    }
  }

  for (int cc = tid; cc < TCGEN_N * kKVecs; cc += TCGEN_THREADS) {
    const int local_m = cc / kKVecs;
    const int kc = (cc % kKVecs) * VEC;
    const int m = token_base + local_m;
    const int t = tok[m];
    const scalar_t* src = a;
    int src_bytes = 0;
    if (t < num_valid_tokens) {
      const int64_t row = static_cast<int64_t>(t) / top_k;
      src = a + row * stride_am + (k0 + kc);
      src_bytes = 16;
    }
    const uint32_t dst = tcgen_swizzle_128b_offset(local_m, kc, TCGEN_N);
    cp_async_16(Bsh + dst, src, src_bytes);
  }
}

template <typename scalar_t, int Rank, int NumKTiles>
__global__ void __launch_bounds__(TCGEN_THREADS) moe_lora_shrink_tcgen05_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    int K,
    int num_valid_tokens,
    int top_k,
    int64_t stride_am,
    int64_t stride_be,
    int64_t stride_bn,
    int64_t stride_bk,
    int64_t stride_cm,
    int64_t stride_cn) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  static_assert(Rank == 16 || Rank == 32 || Rank == 64);
  static_assert(NumKTiles > 0);

  const int pid_m = blockIdx.x;
  const int token_base = static_cast<int>(blockIdx.y) * TCGEN_N;
  const int rank_base = static_cast<int>(blockIdx.z) * WMMA_M;
  if (pid_m * BLOCK_M >= num_tokens_post_padded[0]) return;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  extern __shared__ __align__(16) char smem[];
  char* p = smem;
  int32_t* tok = reinterpret_cast<int32_t*>(p);
  p += align16(BLOCK_M * sizeof(int32_t));
  uint64_t* mma_mbar = reinterpret_cast<uint64_t*>(p);
  p += align16(sizeof(uint64_t));
  uint32_t* tmem_addr_smem = reinterpret_cast<uint32_t*>(p);
  p += align16(sizeof(uint32_t));
  p = reinterpret_cast<char*>(align1024(reinterpret_cast<size_t>(p)));
  char* Ash = p;
  p += TCGEN_GROUP_KTILES * TCGEN_M * TCGEN_BLOCK_K * sizeof(scalar_t);
  char* Bsh = p;

  for (int i = tid; i < BLOCK_M; i += TCGEN_THREADS)
    tok[i] = sorted_token_ids[pid_m * BLOCK_M + i];
  if (tid == 0) {
    *tmem_addr_smem = 0;
    mbarrier_init_shared(mma_mbar, 1);
    mbarrier_init_fence();
  }
  __syncthreads();

  if (tok[token_base] >= num_valid_tokens) return;
  const int expert = expert_ids[pid_m];
  if (expert == -1) return;

  if (tid < 32) {
    tcgen_alloc(tmem_addr_smem);
  }
  __syncthreads();
  if (tid < 32) {
    tcgen_relinquish_alloc_permit();
  }
  __syncthreads();

  const uint32_t tmem_addr = *tmem_addr_smem;
  constexpr uint32_t idesc =
      (1U << 4U) | (1U << 7U) | (1U << 10U) | ((uint32_t(TCGEN_N) >> 3U) << 17U) | ((uint32_t(TCGEN_M) >> 4U) << 24U);
  const int64_t b_base = static_cast<int64_t>(expert) * stride_be;
  uint32_t phase = 0;

  constexpr int a_tcgen_stage = TCGEN_M * TCGEN_BLOCK_K * sizeof(scalar_t);
  constexpr int b_tcgen_stage = TCGEN_N * TCGEN_BLOCK_K * sizeof(scalar_t);

#pragma unroll
  for (int group = 0; group < NumKTiles; group += TCGEN_GROUP_KTILES) {
#pragma unroll
    for (int s = 0; s < TCGEN_GROUP_KTILES; ++s) {
      const int kt = group + s;
      char* As_cur = Ash + s * a_tcgen_stage;
      char* Bs_cur = Bsh + s * b_tcgen_stage;
      issue_tcgen05_stage<scalar_t, Rank, NumKTiles>(
          As_cur,
          Bs_cur,
          a,
          b,
          tok,
          token_base,
          rank_base,
          b_base,
          num_valid_tokens,
          top_k,
          stride_am,
          stride_bn,
          kt * TCGEN_BLOCK_K,
          tid);
      cp_async_commit();
    }
    cp_async_wait<0>();
    __syncthreads();
    tcgen_fence_after_thread_sync();

#pragma unroll
    for (int s = 0; s < TCGEN_GROUP_KTILES; ++s) {
      const int kt = group + s;
      char* As_cur = Ash + s * a_tcgen_stage;
      char* Bs_cur = Bsh + s * b_tcgen_stage;
      if (warp == 0 && elect_sync_u32()) {
        const uint32_t a_base = static_cast<uint32_t>(__cvta_generic_to_shared(As_cur));
        const uint32_t b_base_s = static_cast<uint32_t>(__cvta_generic_to_shared(Bs_cur));
#pragma unroll
        for (int k1 = 0; k1 < TCGEN_BLOCK_K / 64; ++k1) {
#pragma unroll
          for (int k2 = 0; k2 < 64 / WMMA_K; ++k2) {
            const uint32_t koff = k1 * TCGEN_M * 128 + k2 * 32;
            const uint32_t boff = k1 * TCGEN_N * 128 + k2 * 32;
            const uint64_t a_desc = tcgen_make_smem_desc(a_base + koff);
            const uint64_t b_desc = tcgen_make_smem_desc(b_base_s + boff);
            tcgen_mma_f16(tmem_addr, a_desc, b_desc, idesc, (kt != 0 || k1 != 0 || k2 != 0) ? 1U : 0U);
          }
        }
      }
    }

    __syncthreads();
    if (warp == 0 && elect_sync_u32()) {
      tcgen_commit(mma_mbar);
    }
    mbarrier_wait_shared(mma_mbar, phase);
    phase ^= 1;
    __syncthreads();
  }

  tcgen_fence_after_thread_sync();
  if (tid < 32) {
    float vals0[8];
    tcgen_ld_32x32b_x8(vals0, tmem_addr, tid, 0);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int t = tok[token_base + i];
      if (tid < WMMA_M && rank_base + tid < Rank && t < num_valid_tokens) {
        c[static_cast<int64_t>(t) * stride_cm + static_cast<int64_t>(rank_base + tid) * stride_cn] =
            static_cast<scalar_t>(vals0[i]);
      }
    }
  }
  __syncthreads();
  if (tid < 32) {
    tcgen_dealloc(tmem_addr);
  }
#endif
}

template <typename scalar_t, int Rank>
__global__ void __launch_bounds__(256)
    zero_rank_output_kernel(scalar_t* __restrict__ c, int num_valid_tokens, int64_t stride_cm, int64_t stride_cn) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_valid_tokens * Rank;
  for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
    const int t = i / Rank;
    const int n = i - t * Rank;
    c[static_cast<int64_t>(t) * stride_cm + static_cast<int64_t>(n) * stride_cn] = scalar_t(0.0f);
  }
}

template <typename scalar_t, int SplitK>
__global__ void __launch_bounds__(TCGEN_THREADS) moe_lora_shrink_tcgen05_splitk_rank16_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    int K,
    int num_valid_tokens,
    int top_k,
    int64_t stride_am,
    int64_t stride_be,
    int64_t stride_bn,
    int64_t stride_bk,
    int64_t stride_cm,
    int64_t stride_cn) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  static_assert(std::is_same_v<scalar_t, bf16_t>);
  static_assert(SplitK >= 1 && SplitK <= 8);
  const int pid_m = blockIdx.x;
  const int token_base = static_cast<int>(blockIdx.y) * TCGEN_N;
  const int pid_sk = static_cast<int>(blockIdx.z);
  if (pid_m * BLOCK_M >= num_tokens_post_padded[0]) return;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  extern __shared__ __align__(16) char smem[];
  char* p = smem;
  int32_t* tok = reinterpret_cast<int32_t*>(p);
  p += align16(BLOCK_M * sizeof(int32_t));
  uint64_t* mma_mbar = reinterpret_cast<uint64_t*>(p);
  p += align16(sizeof(uint64_t));
  uint32_t* tmem_addr_smem = reinterpret_cast<uint32_t*>(p);
  p += align16(sizeof(uint32_t));
  p = reinterpret_cast<char*>(align1024(reinterpret_cast<size_t>(p)));
  char* Ash = p;
  p += TCGEN_M * TCGEN_BLOCK_K * sizeof(scalar_t);
  char* Bsh = p;

  for (int i = tid; i < BLOCK_M; i += TCGEN_THREADS)
    tok[i] = sorted_token_ids[pid_m * BLOCK_M + i];
  if (tid == 0) {
    *tmem_addr_smem = 0;
    mbarrier_init_shared(mma_mbar, 1);
    mbarrier_init_fence();
  }
  __syncthreads();

  if (tok[token_base] >= num_valid_tokens) return;
  const int expert = expert_ids[pid_m];
  if (expert == -1) return;

  if (tid < 32) {
    tcgen_alloc(tmem_addr_smem);
  }
  __syncthreads();
  if (tid < 32) {
    tcgen_relinquish_alloc_permit();
  }
  __syncthreads();

  const uint32_t tmem_addr = *tmem_addr_smem;
  constexpr uint32_t idesc =
      (1U << 4U) | (1U << 7U) | (1U << 10U) | ((uint32_t(TCGEN_N) >> 3U) << 17U) | ((uint32_t(TCGEN_M) >> 4U) << 24U);
  const int64_t b_base = static_cast<int64_t>(expert) * stride_be;

  issue_tcgen05_stage<scalar_t, 16, 1>(
      Ash,
      Bsh,
      a,
      b,
      tok,
      token_base,
      0,
      b_base,
      num_valid_tokens,
      top_k,
      stride_am,
      stride_bn,
      pid_sk * TCGEN_BLOCK_K,
      tid);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();
  tcgen_fence_after_thread_sync();

  if (warp == 0 && elect_sync_u32()) {
    const uint32_t a_base = static_cast<uint32_t>(__cvta_generic_to_shared(Ash));
    const uint32_t b_base_s = static_cast<uint32_t>(__cvta_generic_to_shared(Bsh));
#pragma unroll
    for (int k1 = 0; k1 < TCGEN_BLOCK_K / 64; ++k1) {
#pragma unroll
      for (int k2 = 0; k2 < 64 / WMMA_K; ++k2) {
        const uint32_t koff = k1 * TCGEN_M * 128 + k2 * 32;
        const uint32_t boff = k1 * TCGEN_N * 128 + k2 * 32;
        const uint64_t a_desc = tcgen_make_smem_desc(a_base + koff);
        const uint64_t b_desc = tcgen_make_smem_desc(b_base_s + boff);
        tcgen_mma_f16(tmem_addr, a_desc, b_desc, idesc, (k1 != 0 || k2 != 0) ? 1U : 0U);
      }
    }
    tcgen_commit(mma_mbar);
  }
  mbarrier_wait_shared(mma_mbar, 0);
  __syncthreads();

  tcgen_fence_after_thread_sync();
  if (tid < 32) {
    float vals[8];
    tcgen_ld_32x32b_x8(vals, tmem_addr, tid, 0);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int t = tok[token_base + i];
      if (tid < WMMA_M && t < num_valid_tokens) {
        atomic_add_bf16(
            reinterpret_cast<bf16_t*>(c + static_cast<int64_t>(t) * stride_cm + static_cast<int64_t>(tid) * stride_cn),
            vals[i]);
      }
    }
  }
  __syncthreads();
  if (tid < 32) {
    tcgen_dealloc(tmem_addr);
  }
#endif
}

template <typename scalar_t, int SplitK>
__global__ void __launch_bounds__(TCGEN_THREADS) moe_lora_shrink_tcgen05_splitk_grouped_rank16_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    int K,
    int num_valid_tokens,
    int top_k,
    int64_t stride_am,
    int64_t stride_be,
    int64_t stride_bn,
    int64_t stride_bk,
    int64_t stride_cm,
    int64_t stride_cn) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  static_assert(std::is_same_v<scalar_t, bf16_t>);
  static_assert(SplitK >= 1 && SplitK <= 8);
  constexpr int NumKTiles = 2048 / TCGEN_BLOCK_K;
  const int pid_m = blockIdx.x;
  const int token_base = static_cast<int>(blockIdx.y) * TCGEN_N;
  const int pid_sk = static_cast<int>(blockIdx.z);
  if (pid_m * BLOCK_M >= num_tokens_post_padded[0]) return;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  extern __shared__ __align__(16) char smem[];
  char* p = smem;
  int32_t* tok = reinterpret_cast<int32_t*>(p);
  p += align16(BLOCK_M * sizeof(int32_t));
  uint64_t* mma_mbar = reinterpret_cast<uint64_t*>(p);
  p += align16(sizeof(uint64_t));
  uint32_t* tmem_addr_smem = reinterpret_cast<uint32_t*>(p);
  p += align16(sizeof(uint32_t));
  p = reinterpret_cast<char*>(align1024(reinterpret_cast<size_t>(p)));
  char* Ash = p;
  p += TCGEN_GROUP_KTILES * TCGEN_M * TCGEN_BLOCK_K * sizeof(scalar_t);
  char* Bsh = p;

  for (int i = tid; i < BLOCK_M; i += TCGEN_THREADS)
    tok[i] = sorted_token_ids[pid_m * BLOCK_M + i];
  if (tid == 0) {
    *tmem_addr_smem = 0;
    mbarrier_init_shared(mma_mbar, 1);
    mbarrier_init_fence();
  }
  __syncthreads();

  if (tok[token_base] >= num_valid_tokens) return;
  const int expert = expert_ids[pid_m];
  if (expert == -1) return;

  if (tid < 32) {
    tcgen_alloc(tmem_addr_smem);
  }
  __syncthreads();
  if (tid < 32) {
    tcgen_relinquish_alloc_permit();
  }
  __syncthreads();

  const uint32_t tmem_addr = *tmem_addr_smem;
  constexpr uint32_t idesc =
      (1U << 4U) | (1U << 7U) | (1U << 10U) | ((uint32_t(TCGEN_N) >> 3U) << 17U) | ((uint32_t(TCGEN_M) >> 4U) << 24U);
  const int64_t b_base = static_cast<int64_t>(expert) * stride_be;
  uint32_t phase = 0;

  constexpr int a_tcgen_stage = TCGEN_M * TCGEN_BLOCK_K * sizeof(scalar_t);
  constexpr int b_tcgen_stage = TCGEN_N * TCGEN_BLOCK_K * sizeof(scalar_t);
  for (int group = pid_sk; group < NumKTiles; group += SplitK * TCGEN_GROUP_KTILES) {
#pragma unroll
    for (int s = 0; s < TCGEN_GROUP_KTILES; ++s) {
      const int kt = group + s * SplitK;
      if (kt >= NumKTiles) continue;
      char* As_cur = Ash + s * a_tcgen_stage;
      char* Bs_cur = Bsh + s * b_tcgen_stage;
      issue_tcgen05_stage<scalar_t, 16, 1>(
          As_cur,
          Bs_cur,
          a,
          b,
          tok,
          token_base,
          0,
          b_base,
          num_valid_tokens,
          top_k,
          stride_am,
          stride_bn,
          kt * TCGEN_BLOCK_K,
          tid);
      cp_async_commit();
      cp_async_wait<0>();
      __syncthreads();
      tcgen_fence_after_thread_sync();

      if (warp == 0 && elect_sync_u32()) {
        const uint32_t a_base = static_cast<uint32_t>(__cvta_generic_to_shared(As_cur));
        const uint32_t b_base_s = static_cast<uint32_t>(__cvta_generic_to_shared(Bs_cur));
#pragma unroll
        for (int k1 = 0; k1 < TCGEN_BLOCK_K / 64; ++k1) {
#pragma unroll
          for (int k2 = 0; k2 < 64 / WMMA_K; ++k2) {
            const uint32_t koff = k1 * TCGEN_M * 128 + k2 * 32;
            const uint32_t boff = k1 * TCGEN_N * 128 + k2 * 32;
            const uint64_t a_desc = tcgen_make_smem_desc(a_base + koff);
            const uint64_t b_desc = tcgen_make_smem_desc(b_base_s + boff);
            tcgen_mma_f16(tmem_addr, a_desc, b_desc, idesc, (kt != pid_sk || k1 != 0 || k2 != 0) ? 1U : 0U);
          }
        }
      }
    }

    __syncthreads();
    if (warp == 0 && elect_sync_u32()) {
      tcgen_commit(mma_mbar);
    }
    mbarrier_wait_shared(mma_mbar, phase);
    phase ^= 1;
    __syncthreads();
  }

  tcgen_fence_after_thread_sync();
  if (tid < 32) {
    float vals[8];
    tcgen_ld_32x32b_x8(vals, tmem_addr, tid, 0);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int t = tok[token_base + i];
      if (tid < WMMA_M && t < num_valid_tokens) {
        bf16_t* dst =
            reinterpret_cast<bf16_t*>(c + static_cast<int64_t>(t) * stride_cm + static_cast<int64_t>(tid) * stride_cn);
        if constexpr (SplitK == 1) {
          *dst = __float2bfloat16(vals[i]);
        } else {
          atomic_add_bf16(dst, vals[i]);
        }
      }
    }
  }
  __syncthreads();
  if (tid < 32) {
    tcgen_dealloc(tmem_addr);
  }
#endif
}

template <typename scalar_t, int Rank, int NumKTiles>
void launch_m16n8_rank(
    cudaStream_t stream,
    int num_m_blocks,
    const scalar_t* hidden_states,
    const scalar_t* lora_a,
    scalar_t* output,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded,
    int K,
    int num_valid_tokens,
    int top_k,
    int64_t stride_am,
    int64_t stride_be,
    int64_t stride_bn,
    int64_t stride_bk,
    int64_t stride_cm,
    int64_t stride_cn) {
  static_assert(Rank == 16 || Rank == 32 || Rank == 64);
  constexpr int Threads = SWAP_THREADS;
  const size_t smem_bytes = align16(BLOCK_M * sizeof(int32_t)) +
                            align16(static_cast<size_t>(KSTAGES) * Rank * BLOCK_K * sizeof(scalar_t)) +
                            align16(static_cast<size_t>(KSTAGES) * BLOCK_M * BLOCK_K * sizeof(scalar_t));

  auto kernel = moe_lora_shrink_m16n8_kernel<scalar_t, Rank, Threads, NumKTiles>;
  if (smem_bytes > 48 * 1024) {
    host::RuntimeDeviceCheck(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
  }

  host::LaunchKernel(dim3(num_m_blocks), dim3(Threads), stream, smem_bytes)(
      kernel,
      hidden_states,
      lora_a,
      output,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_padded,
      K,
      num_valid_tokens,
      top_k,
      stride_am,
      stride_be,
      stride_bn,
      stride_bk,
      stride_cm,
      stride_cn);
}

template <typename scalar_t, int Rank, int NumKTiles>
void launch_m16n8_half_rank(
    cudaStream_t stream,
    int num_m_blocks,
    const scalar_t* hidden_states,
    const scalar_t* lora_a,
    scalar_t* output,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded,
    int K,
    int num_valid_tokens,
    int top_k,
    int64_t stride_am,
    int64_t stride_be,
    int64_t stride_bn,
    int64_t stride_bk,
    int64_t stride_cm,
    int64_t stride_cn) {
  static_assert(Rank == 16 || Rank == 32 || Rank == 64);
  constexpr int Threads = SWAP_THREADS;
  const size_t smem_bytes = align16(BLOCK_M * sizeof(int32_t)) +
                            align16(static_cast<size_t>(KSTAGES) * Rank * BLOCK_K * sizeof(scalar_t)) +
                            align16(static_cast<size_t>(KSTAGES) * SWAP_TOKEN_N * BLOCK_K * sizeof(scalar_t));

  auto kernel = moe_lora_shrink_m16n8_half_kernel<scalar_t, Rank, Threads, NumKTiles>;
  if (smem_bytes > 48 * 1024) {
    host::RuntimeDeviceCheck(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
  }

  host::LaunchKernel(dim3(num_m_blocks * 2), dim3(Threads), stream, smem_bytes)(
      kernel,
      hidden_states,
      lora_a,
      output,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_padded,
      K,
      num_valid_tokens,
      top_k,
      stride_am,
      stride_be,
      stride_bn,
      stride_bk,
      stride_cm,
      stride_cn);
}

template <typename scalar_t, int KStages, int BlockK, int Rank, int NWarps, int SplitK>
void launch_pipe(
    cudaStream_t stream,
    int num_m_blocks,
    int N,
    const scalar_t* hidden_states,
    const scalar_t* lora_a,
    scalar_t* output,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded,
    int K,
    int num_valid_tokens,
    int top_k,
    int64_t stride_am,
    int64_t stride_be,
    int64_t stride_bn,
    int64_t stride_bk,
    int64_t stride_cm,
    int64_t stride_cn) {
  constexpr int kNTiles = Rank / 8;
  const size_t smem_bytes = align16(BLOCK_M * sizeof(int32_t)) + align16(BLOCK_M * sizeof(const scalar_t*)) +
                            align16(Rank * sizeof(const scalar_t*)) +
                            align16(static_cast<size_t>(KStages) * BLOCK_M * BlockK * sizeof(scalar_t)) +
                            align16(static_cast<size_t>(KStages) * Rank * BlockK * sizeof(scalar_t)) +
                            align16(static_cast<size_t>(NWarps) * 32 * kNTiles * 4 * sizeof(float));

  // Split-K writes fp32 partials into a persistent scratch (contention-free), then
  // a tiny reduction kernel sums them into the output. Inline atomic-add (bf16)
  // was measured ~2x slower at bs=1 -- see the store() comment in the kernel.
  float* workspace = nullptr;
  if constexpr (SplitK > 1) {
    workspace = get_shrink_workspace(static_cast<size_t>(SplitK) * num_valid_tokens * Rank);
  }

  auto kernel = moe_lora_shrink_pipe_kernel<scalar_t, KStages, BlockK, Rank, NWarps, SplitK>;
  if (smem_bytes > 48 * 1024) {
    host::RuntimeDeviceCheck(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
  }

  // PDL only for the single-pass path: the split-K path reuses a persistent fp32
  // workspace across calls, so overlapping consecutive launches could race the
  // following reduce kernel's read of it. SGLANG_SHRINK_PIPE_PDL=0 disables it.
  static const bool s_pipe_pdl = []() {
    const char* v = std::getenv("SGLANG_SHRINK_PIPE_PDL");
    return v == nullptr || v[0] != '0';
  }();
  constexpr bool kPdlSafe = (SplitK == 1);

  host::LaunchKernel(dim3(num_m_blocks, SplitK), dim3(NWarps * 32), stream, smem_bytes)
      .enable_pdl(s_pipe_pdl && kPdlSafe)(
          kernel,
          hidden_states,
          lora_a,
          output,
          workspace,
          sorted_token_ids,
          expert_ids,
          num_tokens_post_padded,
          N,
          K,
          num_valid_tokens,
          top_k,
          stride_am,
          stride_be,
          stride_bn,
          stride_bk,
          stride_cm,
          stride_cn);

  if constexpr (SplitK > 1) {
    const int total = num_valid_tokens * Rank;
    const int red_blocks = std::min(1024, std::max(1, (total + 255) / 256));
    host::LaunchKernel(dim3(red_blocks), dim3(256), stream)(
        moe_lora_shrink_splitk_reduce_kernel<scalar_t, Rank, SplitK>,
        workspace,
        output,
        num_valid_tokens,
        stride_cm,
        stride_cn);
  }
}

template <typename scalar_t, int Rank, int NumKTiles>
void launch_tcgen05_rank(
    cudaStream_t stream,
    int num_m_blocks,
    const scalar_t* hidden_states,
    const scalar_t* lora_a,
    scalar_t* output,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded,
    int K,
    int num_valid_tokens,
    int top_k,
    int64_t stride_am,
    int64_t stride_be,
    int64_t stride_bn,
    int64_t stride_bk,
    int64_t stride_cm,
    int64_t stride_cn) {
  static_assert(Rank == 16 || Rank == 32 || Rank == 64);
  const size_t smem_bytes =
      align1024(align16(BLOCK_M * sizeof(int32_t)) + align16(sizeof(uint64_t)) + align16(sizeof(uint32_t))) +
      TCGEN_GROUP_KTILES * TCGEN_M * TCGEN_BLOCK_K * sizeof(scalar_t) +
      TCGEN_GROUP_KTILES * TCGEN_N * TCGEN_BLOCK_K * sizeof(scalar_t);

  auto kernel = moe_lora_shrink_tcgen05_kernel<scalar_t, Rank, NumKTiles>;
  if (smem_bytes > 48 * 1024) {
    host::RuntimeDeviceCheck(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
  }

  host::LaunchKernel(dim3(num_m_blocks, BLOCK_M / TCGEN_N, Rank / WMMA_M), dim3(TCGEN_THREADS), stream, smem_bytes)(
      kernel,
      hidden_states,
      lora_a,
      output,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_padded,
      K,
      num_valid_tokens,
      top_k,
      stride_am,
      stride_be,
      stride_bn,
      stride_bk,
      stride_cm,
      stride_cn);
}

template <typename scalar_t>
void launch_tcgen05_splitk_rank16(
    cudaStream_t stream,
    int split_k,
    int num_m_blocks,
    const scalar_t* hidden_states,
    const scalar_t* lora_a,
    scalar_t* output,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded,
    int K,
    int num_valid_tokens,
    int top_k,
    int64_t stride_am,
    int64_t stride_be,
    int64_t stride_bn,
    int64_t stride_bk,
    int64_t stride_cm,
    int64_t stride_cn) {
  static_assert(std::is_same_v<scalar_t, bf16_t>);
  constexpr int Rank = 16;
  if (split_k > 1) {
    const int zero_blocks = std::min(1024, std::max(1, (num_valid_tokens * Rank + 255) / 256));
    host::LaunchKernel(dim3(zero_blocks), dim3(256), stream)(
        zero_rank_output_kernel<scalar_t, Rank>, output, num_valid_tokens, stride_cm, stride_cn);
  }

  if (split_k == 8) {
    const size_t smem_bytes =
        align1024(align16(BLOCK_M * sizeof(int32_t)) + align16(sizeof(uint64_t)) + align16(sizeof(uint32_t))) +
        TCGEN_M * TCGEN_BLOCK_K * sizeof(scalar_t) + TCGEN_N * TCGEN_BLOCK_K * sizeof(scalar_t);

    auto kernel = moe_lora_shrink_tcgen05_splitk_rank16_kernel<scalar_t, 8>;
    if (smem_bytes > 48 * 1024) {
      host::RuntimeDeviceCheck(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }

    host::LaunchKernel(dim3(num_m_blocks, BLOCK_M / TCGEN_N, 8), dim3(TCGEN_THREADS), stream, smem_bytes)(
        kernel,
        hidden_states,
        lora_a,
        output,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        K,
        num_valid_tokens,
        top_k,
        stride_am,
        stride_be,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn);
    return;
  }

  const size_t smem_bytes =
      align1024(align16(BLOCK_M * sizeof(int32_t)) + align16(sizeof(uint64_t)) + align16(sizeof(uint32_t))) +
      TCGEN_GROUP_KTILES * TCGEN_M * TCGEN_BLOCK_K * sizeof(scalar_t) +
      TCGEN_GROUP_KTILES * TCGEN_N * TCGEN_BLOCK_K * sizeof(scalar_t);

#define SGL_LAUNCH_TCGEN05_SPLITK(SPLIT_K_VALUE)                                                                       \
  do {                                                                                                                 \
    auto kernel = moe_lora_shrink_tcgen05_splitk_grouped_rank16_kernel<scalar_t, SPLIT_K_VALUE>;                       \
    if (smem_bytes > 48 * 1024) {                                                                                      \
      host::RuntimeDeviceCheck(                                                                                        \
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));    \
    }                                                                                                                  \
    host::LaunchKernel(dim3(num_m_blocks, BLOCK_M / TCGEN_N, SPLIT_K_VALUE), dim3(TCGEN_THREADS), stream, smem_bytes)( \
        kernel,                                                                                                        \
        hidden_states,                                                                                                 \
        lora_a,                                                                                                        \
        output,                                                                                                        \
        sorted_token_ids,                                                                                              \
        expert_ids,                                                                                                    \
        num_tokens_post_padded,                                                                                        \
        K,                                                                                                             \
        num_valid_tokens,                                                                                              \
        top_k,                                                                                                         \
        stride_am,                                                                                                     \
        stride_be,                                                                                                     \
        stride_bn,                                                                                                     \
        stride_bk,                                                                                                     \
        stride_cm,                                                                                                     \
        stride_cn);                                                                                                    \
  } while (0)

  switch (split_k) {
    case 8:
      SGL_LAUNCH_TCGEN05_SPLITK(8);
      break;
    case 7:
      SGL_LAUNCH_TCGEN05_SPLITK(7);
      break;
    case 6:
      SGL_LAUNCH_TCGEN05_SPLITK(6);
      break;
    case 5:
      SGL_LAUNCH_TCGEN05_SPLITK(5);
      break;
    case 4:
      SGL_LAUNCH_TCGEN05_SPLITK(4);
      break;
    case 3:
      SGL_LAUNCH_TCGEN05_SPLITK(3);
      break;
    case 2:
      SGL_LAUNCH_TCGEN05_SPLITK(2);
      break;
    default:
      SGL_LAUNCH_TCGEN05_SPLITK(1);
      break;
  }

#undef SGL_LAUNCH_TCGEN05_SPLITK
}

template <typename scalar_t>
struct MoeLoraShrinkKernel {
  static void
  run(tvm::ffi::TensorView output,
      tvm::ffi::TensorView hidden_states,
      tvm::ffi::TensorView lora_a,
      tvm::ffi::TensorView sorted_token_ids,
      tvm::ffi::TensorView expert_ids,
      tvm::ffi::TensorView num_tokens_post_padded,
      int64_t top_k,
      int64_t block_size_m) {
    using namespace host;

    RuntimeCheck(hidden_states.ndim() == 2, "hidden_states must be 2D [num_tokens, K]");
    RuntimeCheck(lora_a.ndim() == 3, "lora_a must be 3D [num_virtual_experts, N, K]");
    RuntimeCheck(output.ndim() == 2, "output must be 2D [num_tokens * top_k, N]");
    RuntimeCheck(block_size_m == BLOCK_M, "m16n8 shrink requires routing block_size_m == 16");
    RuntimeCheck(hidden_states.stride(1) == 1, "hidden_states must be K-contiguous (stride[1] == 1)");
    RuntimeCheck(lora_a.stride(2) == 1, "lora_a must be K-contiguous (stride[2] == 1)");

    const int N = static_cast<int>(lora_a.size(1));
    const int K = static_cast<int>(lora_a.size(2));
    RuntimeCheck(N == 16 || N == 32 || N == 64, "m16n8 shrink supports rank N in {16, 32, 64}");
    RuntimeCheck(K % BLOCK_K == 0, "m16n8 shrink requires K divisible by BLOCK_K");
    RuntimeCheck(hidden_states.size(1) == K, "hidden_states K must match lora_a K");
    RuntimeCheck(output.size(1) == N, "output N must match lora_a N");

    const int num_m_blocks = static_cast<int>(expert_ids.size(0));
    const int num_valid_tokens = static_cast<int>(output.size(0));
    if (num_m_blocks == 0 || N == 0 || num_valid_tokens == 0) return;

    const auto device = hidden_states.device();
    const cudaStream_t stream = LaunchKernel::resolve_device(device);

    auto launch_rank = [&]<int Rank, int NumKTiles>() {
      launch_m16n8_rank<scalar_t, Rank, NumKTiles>(
          stream,
          num_m_blocks,
          static_cast<const scalar_t*>(hidden_states.data_ptr()),
          static_cast<const scalar_t*>(lora_a.data_ptr()),
          static_cast<scalar_t*>(output.data_ptr()),
          static_cast<const int32_t*>(sorted_token_ids.data_ptr()),
          static_cast<const int32_t*>(expert_ids.data_ptr()),
          static_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
          K,
          num_valid_tokens,
          static_cast<int>(top_k),
          hidden_states.stride(0),
          lora_a.stride(0),
          lora_a.stride(1),
          lora_a.stride(2),
          output.stride(0),
          output.stride(1));
    };

    auto launch_tcgen = [&]<int Rank, int NumKTiles>() {
      launch_tcgen05_rank<scalar_t, Rank, NumKTiles>(
          stream,
          num_m_blocks,
          static_cast<const scalar_t*>(hidden_states.data_ptr()),
          static_cast<const scalar_t*>(lora_a.data_ptr()),
          static_cast<scalar_t*>(output.data_ptr()),
          static_cast<const int32_t*>(sorted_token_ids.data_ptr()),
          static_cast<const int32_t*>(expert_ids.data_ptr()),
          static_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
          K,
          num_valid_tokens,
          static_cast<int>(top_k),
          hidden_states.stride(0),
          lora_a.stride(0),
          lora_a.stride(1),
          lora_a.stride(2),
          output.stride(0),
          output.stride(1));
    };

    auto launch_pipe_call = [&]<int KStages, int BlockK, int Rank, int NWarps, int SplitK>() {
      launch_pipe<scalar_t, KStages, BlockK, Rank, NWarps, SplitK>(
          stream,
          num_m_blocks,
          N,
          static_cast<const scalar_t*>(hidden_states.data_ptr()),
          static_cast<const scalar_t*>(lora_a.data_ptr()),
          static_cast<scalar_t*>(output.data_ptr()),
          static_cast<const int32_t*>(sorted_token_ids.data_ptr()),
          static_cast<const int32_t*>(expert_ids.data_ptr()),
          static_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
          K,
          num_valid_tokens,
          static_cast<int>(top_k),
          hidden_states.stride(0),
          lora_a.stride(0),
          lora_a.stride(1),
          lora_a.stride(2),
          output.stride(0),
          output.stride(1));
    };

    // Triton-style pipelined shrink (moe_lora_shrink_pipe_kernel) is the default
    // for the rank-16 K==2048 hot path. Wins, in order of impact: 128B smem
    // swizzle (kills ldmatrix bank conflicts), rank-aware no-padding, hoisted
    // token/top_k divide, cheap warp/lane gather, 8 warps (occupancy hides the
    // latency), and a K-split layout where every warp computes ALL N-tiles so the
    // A fragment is loaded once and reused and all warps stay barrier-balanced.
    //
    // BlockK defaults to 512 (not 256): K==2048 then streams in 4 tiles instead
    // of 8, halving the serial cp.async-commit/__syncthreads cycles on the
    // latency-bound single-block path. Measured flat ~5.16us across bs=1..128 on
    // B200 (vs ~5.67us at BlockK=256) -- beats cuBLAS (nvjet skinny GEMM + splitK
    // reduce, ~5.5us GPU / ~8.4us wall) at every bs, and beats production Triton
    // for bs>=16. See benchmark/moe_lora_shrink_optimization.md.
    //
    // Low-occupancy fill: when only a handful of expert token-blocks are active
    // (bs=1 decode -> ~8 m-blocks on a 148-SM B200) the single-pass K-loop leaves
    // the machine idle and is purely latency-bound. Split K to fan out ~128 blocks
    // (same idea as the Triton 128/base_grid heuristic). Our split-K writes fp32
    // partials + a separate reduce launch (vs Triton's inline atomics), so it only
    // pays off when occupancy is genuinely low -- gate it at num_m_blocks <= 16.
    // BlockK drops to 256 there so split-K can reach 8 (capped at K/BlockK).
    // SGLANG_SHRINK_PIPE=0 disables it; =1 forces it on for any rank-16/32
    // K%256==0 case. STAGES/BLOCKK/WARPS/SPLITK env overrides win over the defaults.
    if ((N == 16 || N == 32) && K % 256 == 0) {
      bool use_pipe = (N == 16 && K == 2048);
      if (const char* v = std::getenv("SGLANG_SHRINK_PIPE")) use_pipe = (v[0] == '1');
      if (use_pipe) {
        {
          int kstages = 5, blockk = 512, warps = 8, splitk = 1;
          if (num_m_blocks <= 16) {
            blockk = 256;
            splitk = 8;  // 128/8 m-blocks; min(8, K/blockk). Macro handles {1,2,4,8}.
          }
          if (const char* s = std::getenv("SGLANG_SHRINK_PIPE_STAGES")) kstages = atoi(s);
          if (const char* bk = std::getenv("SGLANG_SHRINK_PIPE_BLOCKK")) blockk = atoi(bk);
          if (const char* w = std::getenv("SGLANG_SHRINK_PIPE_WARPS")) warps = atoi(w);
          if (const char* sk = std::getenv("SGLANG_SHRINK_PIPE_SPLITK")) splitk = atoi(sk);
#define SGL_PIPE_KS(BK, RANK, NW, SK)                                \
  do {                                                               \
    switch (kstages) {                                               \
      case 2:                                                        \
        launch_pipe_call.template operator()<2, BK, RANK, NW, SK>(); \
        break;                                                       \
      case 3:                                                        \
        launch_pipe_call.template operator()<3, BK, RANK, NW, SK>(); \
        break;                                                       \
      case 5:                                                        \
        launch_pipe_call.template operator()<5, BK, RANK, NW, SK>(); \
        break;                                                       \
      case 6:                                                        \
        launch_pipe_call.template operator()<6, BK, RANK, NW, SK>(); \
        break;                                                       \
      case 7:                                                        \
        launch_pipe_call.template operator()<7, BK, RANK, NW, SK>(); \
        break;                                                       \
      case 8:                                                        \
        launch_pipe_call.template operator()<8, BK, RANK, NW, SK>(); \
        break;                                                       \
      case 9:                                                        \
        launch_pipe_call.template operator()<9, BK, RANK, NW, SK>(); \
        break;                                                       \
      default:                                                       \
        launch_pipe_call.template operator()<4, BK, RANK, NW, SK>(); \
        break;                                                       \
    }                                                                \
  } while (0)
#define SGL_PIPE_SK(BK, RANK, NW)     \
  do {                                \
    switch (splitk) {                 \
      case 2:                         \
        SGL_PIPE_KS(BK, RANK, NW, 2); \
        break;                        \
      case 4:                         \
        SGL_PIPE_KS(BK, RANK, NW, 4); \
        break;                        \
      case 8:                         \
        SGL_PIPE_KS(BK, RANK, NW, 8); \
        break;                        \
      default:                        \
        SGL_PIPE_KS(BK, RANK, NW, 1); \
        break;                        \
    }                                 \
  } while (0)
#define SGL_PIPE_WARPS(BK, RANK)   \
  do {                             \
    switch (warps) {               \
      case 2:                      \
        SGL_PIPE_SK(BK, RANK, 2);  \
        break;                     \
      case 8:                      \
        SGL_PIPE_SK(BK, RANK, 8);  \
        break;                     \
      case 16:                     \
        SGL_PIPE_SK(BK, RANK, 16); \
        break;                     \
      default:                     \
        SGL_PIPE_SK(BK, RANK, 4);  \
        break;                     \
    }                              \
  } while (0)
#define SGL_PIPE_DISPATCH(RANK)  \
  do {                           \
    if (blockk == 128) {         \
      SGL_PIPE_WARPS(128, RANK); \
    } else if (blockk == 512) {  \
      SGL_PIPE_WARPS(512, RANK); \
    } else {                     \
      SGL_PIPE_WARPS(256, RANK); \
    }                            \
  } while (0)
          if (N == 16) {
            SGL_PIPE_DISPATCH(16);
          } else {
            SGL_PIPE_DISPATCH(32);
          }
#undef SGL_PIPE_DISPATCH
#undef SGL_PIPE_WARPS
#undef SGL_PIPE_SK
#undef SGL_PIPE_KS
          return;
        }
      }
    }

    const bool use_k2048 = K == 2048;
    bool use_tcgen05 = false;
    if constexpr (std::is_same_v<scalar_t, bf16_t>) {
      int current_device = 0;
      cudaDeviceProp prop{};
      RuntimeDeviceCheck(cudaGetDevice(&current_device));
      RuntimeDeviceCheck(cudaGetDeviceProperties(&prop, current_device));
      use_tcgen05 = use_k2048 && N == 16 && prop.major >= 10;
    }
    switch (N) {
      case 16:
        if constexpr (std::is_same_v<scalar_t, bf16_t>) {
          if (use_tcgen05) {
            if (num_m_blocks <= 16) {
              launch_tcgen05_splitk_rank16<scalar_t>(
                  stream,
                  8,
                  num_m_blocks,
                  static_cast<const scalar_t*>(hidden_states.data_ptr()),
                  static_cast<const scalar_t*>(lora_a.data_ptr()),
                  static_cast<scalar_t*>(output.data_ptr()),
                  static_cast<const int32_t*>(sorted_token_ids.data_ptr()),
                  static_cast<const int32_t*>(expert_ids.data_ptr()),
                  static_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
                  K,
                  num_valid_tokens,
                  static_cast<int>(top_k),
                  hidden_states.stride(0),
                  lora_a.stride(0),
                  lora_a.stride(1),
                  lora_a.stride(2),
                  output.stride(0),
                  output.stride(1));
            } else {
              launch_tcgen.template operator()<16, 8>();
            }
            break;
          }
        }
        if (use_k2048) {
          launch_m16n8_half_rank<scalar_t, 16, 8>(
              stream,
              num_m_blocks,
              static_cast<const scalar_t*>(hidden_states.data_ptr()),
              static_cast<const scalar_t*>(lora_a.data_ptr()),
              static_cast<scalar_t*>(output.data_ptr()),
              static_cast<const int32_t*>(sorted_token_ids.data_ptr()),
              static_cast<const int32_t*>(expert_ids.data_ptr()),
              static_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
              K,
              num_valid_tokens,
              static_cast<int>(top_k),
              hidden_states.stride(0),
              lora_a.stride(0),
              lora_a.stride(1),
              lora_a.stride(2),
              output.stride(0),
              output.stride(1));
        } else {
          launch_rank.template operator()<16, 0>();
        }
        break;
      case 32:
        if (use_tcgen05) {
          launch_tcgen.template operator()<32, 8>();
        } else if (use_k2048) {
          launch_m16n8_half_rank<scalar_t, 32, 8>(
              stream,
              num_m_blocks,
              static_cast<const scalar_t*>(hidden_states.data_ptr()),
              static_cast<const scalar_t*>(lora_a.data_ptr()),
              static_cast<scalar_t*>(output.data_ptr()),
              static_cast<const int32_t*>(sorted_token_ids.data_ptr()),
              static_cast<const int32_t*>(expert_ids.data_ptr()),
              static_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
              K,
              num_valid_tokens,
              static_cast<int>(top_k),
              hidden_states.stride(0),
              lora_a.stride(0),
              lora_a.stride(1),
              lora_a.stride(2),
              output.stride(0),
              output.stride(1));
        } else {
          launch_rank.template operator()<32, 0>();
        }
        break;
      case 64:
        if (use_tcgen05) {
          launch_tcgen.template operator()<64, 8>();
        } else if (use_k2048) {
          launch_m16n8_half_rank<scalar_t, 64, 8>(
              stream,
              num_m_blocks,
              static_cast<const scalar_t*>(hidden_states.data_ptr()),
              static_cast<const scalar_t*>(lora_a.data_ptr()),
              static_cast<scalar_t*>(output.data_ptr()),
              static_cast<const int32_t*>(sorted_token_ids.data_ptr()),
              static_cast<const int32_t*>(expert_ids.data_ptr()),
              static_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
              K,
              num_valid_tokens,
              static_cast<int>(top_k),
              hidden_states.stride(0),
              lora_a.stride(0),
              lora_a.stride(1),
              lora_a.stride(2),
              output.stride(0),
              output.stride(1));
        } else {
          launch_rank.template operator()<64, 0>();
        }
        break;
    }
  }
};

}  // namespace
