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

// SM90 (Hopper) Q8KV8 born-fp8 q-prep kernel.
//
// Computes, per head h:
//   out[:, h, :512]    = fp8_e4m3(bf16(fp32_accum(q_nope[:, h, :] @ w_kc[h])))
//   out[:, h, 512:576] = fp8_e4m3(q_rope[:, h, :])
//
// This is the CUDA replacement for the Triton absorbed_bmm_concat_cast_q_fp8
// kernel (triton_ops/cache_ops.py).  The epilogue keeps the exact rounding
// chain of the Triton variants: fp32 WGMMA accumulate -> bf16 round-to-nearest
// (cublas-equivalent output rounding) -> fp8_e4m3 rn/satfinite on store.  The
// K dimension is consumed as one in-order chain of k=16 WGMMA steps into a
// single fp32 accumulator, i.e. the same fp32 add order as the Triton
// "two_dot"/"grouped" variants (128+64 chained tl.dot), so the nope half can
// come out bitwise identical to them.
//
// Phase-1 design (correctness first):
//   grid = (ceil(T / 128), H); one CTA = two WGMMA warpgroups (256 threads)
//   owning a 128-row m-tile of one head (warpgroup w computes rows
//   [64w, 64w+64)).  A tile [128, K] bf16 is cp.async'd to smem once; the
//   N=512 output is produced in four BN=128 n-slabs, each slab loading B
//   [128, K] bf16 to smem (single-buffered, shared by both warpgroups) and
//   running an SS WGMMA m64n128k16 chain per warpgroup.  The B-slab load of
//   slab nb+1 overlaps the fp8 store epilogue of slab nb; the rope path
//   overlaps the initial cp.async.  smem = (128 + 128) * K * 2 bytes = 96 KB
//   at K=192, so 2 CTAs/SM cover the remaining load/compute serialization
//   across CTAs.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>

#include "params.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>

namespace qprep_sm90 {

using namespace cute;
using bf16 = cutlass::bfloat16_t;

#define QPREP_ASSERT(cond)                                                             \
  do {                                                                                 \
    if (!(cond)) {                                                                     \
      fprintf(stderr, "QPREP_ASSERT failed (%s:%d): %s\n", __FILE__, __LINE__, #cond); \
      exit(1);                                                                         \
    }                                                                                  \
  } while (0)

#define QPREP_CUDA_CHECK(call)                                                                  \
  do {                                                                                          \
    cudaError_t err = (call);                                                                   \
    if (err != cudaSuccess) {                                                                   \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                                                                  \
    }                                                                                           \
  } while (0)

__host__ __device__ __forceinline__ constexpr int ceil_div_i(int a, int b) {
  return (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// 16-byte cp.async.cg with zero-fill when pred is false (same instruction
// family as the sparse-prefill producer; no L2 policy in phase 1).
__device__ __forceinline__ void cp_async_16_zfill(void* smem_dst, const void* gmem_src, bool pred) {
  uint32_t dst_addr = cute::cast_smem_ptr_to_uint(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst_addr), "l"(gmem_src), "r"(pred ? 16 : 0));
}

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
  uint32_t dst_addr = cute::cast_smem_ptr_to_uint(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_addr), "l"(gmem_src));
}

// Pack two fp32 into two fp8_e4m3 bytes with round-to-nearest + satfinite.
// PTX: cvt.rn.satfinite.e4m3x2.f32 d, a, b -> d[7:0] = cvt(b), d[15:8] = cvt(a).
__device__ __forceinline__ uint16_t f32x2_to_e4m3x2_rn_satfinite(float f_lo, float f_hi) {
  uint16_t v;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(v) : "f"(f_hi), "f"(f_lo));
  return v;
}

// The exact Triton epilogue rounding chain for the nope half:
// fp32 accum -> bf16 (rn) -> fp32 (exact) -> fp8_e4m3 (rn, satfinite).
__device__ __forceinline__ uint16_t f32x2_to_bf16x2_to_e4m3x2(float f0, float f1) {
  const __nv_bfloat162 b = __float22bfloat162_rn(make_float2(f0, f1));
  return f32x2_to_e4m3x2_rn_satfinite(__low2float(b), __high2float(b));
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <typename Kernel>
__global__ void qprep_bf16_fp8_kernel(__grid_constant__ const QprepBf16Fp8Sm90Params params);

template <int K_DIM>
struct QprepBf16Fp8Kernel {
  static constexpr int BM = 128;     // m-tile rows (two WGMMA warpgroups)
  static constexpr int BN = 128;     // n-slab width (WGMMA m64n128k16)
  static constexpr int N_OUT = 512;  // kv_lora_rank
  static constexpr int ROPE = 64;    // qk_rope_head_dim
  static constexpr int NUM_THREADS = 256;
  static constexpr int N_SLABS = N_OUT / BN;
  static constexpr int LOAD_ROWS_PER_PASS = NUM_THREADS / 8;  // 16B-chunk loaders

  static_assert(K_DIM % 64 == 0, "K must tile the SW128 bf16 GMMA atom (64 cols)");
  static_assert(N_OUT % BN == 0);

  // K-major SW128 smem layouts for the SS WGMMA operands (bf16 atom = 8x64).
  using SmemLayoutA =
      decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<BM>, Int<K_DIM>>{}, Step<_1, _2>{}));
  using SmemLayoutB =
      decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<BN>, Int<K_DIM>>{}, Step<_1, _2>{}));

  // Two warpgroups stacked along M: threads [128w, 128w+128) own rows
  // [64w, 64w+64) of the m-tile.
  using TiledMMA_t = decltype(make_tiled_mma(
      SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}, Layout<Shape<_2, _1, _1>>{}));

  struct SharedStorage {
    array_aligned<bf16, cosize_v<SmemLayoutA>, 128> a;
    array_aligned<bf16, cosize_v<SmemLayoutB>, 128> b;
    // fp8 output staging for one n-slab: scattered per-thread u16 epilogue
    // writes land here, then leave as coalesced 16B global stores (the direct
    // u16 global stores 4x-amplify the store sectors and throttle the LSU).
    array_aligned<uint8_t, BM * BN, 16> c_stage;
  };

  // -------------------------------------------------------------------------
  // Loads: NUM_THREADS as (NUM_THREADS/8) row-threads x 8 chunk-threads, 16B
  // per cp.async.  Smem addresses go through the CUTE tensor so the SW128
  // swizzle is applied (16B chunks stay contiguous under the swizzle).
  // -------------------------------------------------------------------------
  template <typename SmemT>
  static __device__ __forceinline__ void load_a_tile(SmemT& sA, const bf16* gA, int64_t a_s0, int m_residue, int tid) {
    const int cthr = tid % 8, rthr = tid / 8;
    CUTE_UNROLL
    for (int mi = 0; mi < BM / LOAD_ROWS_PER_PASS; ++mi) {
      const int row = rthr + LOAD_ROWS_PER_PASS * mi;
      const bool pred = row < m_residue;  // zfill OOB rows: 0 * w == 0, never stored
      const bf16* g = gA + (int64_t)row * a_s0;
      CUTE_UNROLL
      for (int ki = 0; ki < K_DIM / 64; ++ki) {
        const int col = cthr * 8 + 64 * ki;
        cp_async_16_zfill(&sA(row, col), g + col, pred);
      }
    }
  }

  template <typename SmemT>
  static __device__ __forceinline__ void load_b_slab(SmemT& sB, const bf16* gB_head, int64_t b_s2, int nb, int tid) {
    const int cthr = tid % 8, rthr = tid / 8;
    const bf16* g0 = gB_head + (int64_t)nb * BN * b_s2;
    CUTE_UNROLL
    for (int ni = 0; ni < BN / LOAD_ROWS_PER_PASS; ++ni) {
      const int nrow = rthr + LOAD_ROWS_PER_PASS * ni;
      const bf16* g = g0 + (int64_t)nrow * b_s2;
      CUTE_UNROLL
      for (int ki = 0; ki < K_DIM / 64; ++ki) {
        const int col = cthr * 8 + 64 * ki;
        cp_async_16(&sB(nrow, col), g + col);
      }
    }
  }

  // -------------------------------------------------------------------------
  // SS WGMMA over the whole K extent as one in-order k=16 chain (clears the
  // accumulator on the first step).  Adapted from the sparse-prefill gemm_ss.
  // -------------------------------------------------------------------------
  template <typename TA, typename TB, typename TC>
  static __device__ __forceinline__ void gemm_ss(TiledMMA_t& tiled_mma, TA const& sA, TB const& sB, TC& acc, int tid) {
    ThrMMA thr_mma = tiled_mma.get_slice(tid);
    Tensor sA_frag = thr_mma.partition_fragment_A(sA);
    Tensor sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(sA_frag) == size<2>(sB_frag));

    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    CUTE_UNROLL
    for (int k = 0; k < size<2>(sA_frag); ++k) {
      cute::gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), acc);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_fence_operand(acc);
  }

  // -------------------------------------------------------------------------
  // Epilogue for one n-slab: fp32 acc -> bf16 -> fp8, 2 adjacent columns per
  // 16-bit store.  WGMMA m64nN C layout: within its warpgroup, thread t holds
  // rows (t/32)*16 + (t%32)/4 + {0,8} (plus 64 * warpgroup_idx here) and
  // columns (t%4)*2 + 8j + {0,1}; fragment linear index
  // i = 4j + 2*row_parity + col_parity.
  // -------------------------------------------------------------------------
  template <typename TC>
  static __device__ __forceinline__ void
  store_slab_direct(TC const& acc, uint8_t* gO, int64_t o_s0, int n0, int row_base, int col_base, int m_residue) {
    CUTE_UNROLL
    for (int rp = 0; rp < 2; ++rp) {
      const int row = row_base + 8 * rp;
      if (row >= m_residue) continue;
      uint8_t* orow = gO + (int64_t)row * o_s0 + n0 + col_base;
      CUTE_UNROLL
      for (int j = 0; j < BN / 8; ++j) {
        const float f0 = acc(j * 4 + rp * 2 + 0);
        const float f1 = acc(j * 4 + rp * 2 + 1);
        *reinterpret_cast<uint16_t*>(orow + 8 * j) = f32x2_to_bf16x2_to_e4m3x2(f0, f1);
      }
    }
  }

  // Staged variant: XOR-swizzle the 16B chunk index by the row so the u16
  // stage writes (8 distinct rows per warp) land in distinct banks, while the
  // 16B copy-out reads stay conflict-free row segments.
  static __device__ __forceinline__ int stage_off(int r, int c) {
    return r * BN + ((((c >> 4) ^ (r & 7)) << 4) | (c & 15));
  }

  template <typename TC>
  static __device__ __forceinline__ void store_slab_staged(
      TC const& acc,
      uint8_t* stage,
      uint8_t* gO,
      int64_t o_s0,
      int n0,
      int row_base,
      int col_base,
      int m_residue,
      int tid) {
    CUTE_UNROLL
    for (int rp = 0; rp < 2; ++rp) {
      const int row = row_base + 8 * rp;  // OOB rows staged but never copied out
      CUTE_UNROLL
      for (int j = 0; j < BN / 8; ++j) {
        const float f0 = acc(j * 4 + rp * 2 + 0);
        const float f1 = acc(j * 4 + rp * 2 + 1);
        *reinterpret_cast<uint16_t*>(stage + stage_off(row, col_base + 8 * j)) = f32x2_to_bf16x2_to_e4m3x2(f0, f1);
      }
    }
    __syncthreads();
    constexpr int CHUNKS_PER_ROW = BN / 16;
    constexpr int NUM_CHUNKS = BM * BN / 16;
    CUTE_UNROLL
    for (int i = 0; i < NUM_CHUNKS / NUM_THREADS; ++i) {
      const int chunk = tid + i * NUM_THREADS;
      const int r = chunk / CHUNKS_PER_ROW;
      const int c = (chunk % CHUNKS_PER_ROW) * 16;
      if (r >= m_residue) continue;
      const uint4 v = *reinterpret_cast<const uint4*>(stage + stage_off(r, c));
      *reinterpret_cast<uint4*>(gO + (int64_t)r * o_s0 + n0 + c) = v;
    }
  }

  // -------------------------------------------------------------------------
  // Rope path: out[:, h, 512:576] = fp8(q_rope[:, h, :]).  bf16 -> fp32
  // (exact) -> fp8 rn/satfinite == the Triton store conversion, so this half
  // is bit-exact vs concat_and_cast_q_fp8_pad.  8 bf16 per thread-chunk.
  // -------------------------------------------------------------------------
  static __device__ __forceinline__ void
  rope_path(const bf16* gR, uint8_t* gO, const QprepBf16Fp8Sm90Params& p, int m_residue, int tid) {
    const int cthr = tid % 8, rthr = tid / 8;
    CUTE_UNROLL
    for (int mi = 0; mi < BM / LOAD_ROWS_PER_PASS; ++mi) {
      const int row = rthr + LOAD_ROWS_PER_PASS * mi;
      if (row >= m_residue) continue;
      const bf16* g = gR + (int64_t)row * p.r_s0 + cthr * 8;
      uint8_t* o = gO + (int64_t)row * p.o_s0 + N_OUT + cthr * 8;
      __nv_bfloat162 v[4];
      if (p.rope_vec16) {
        const uint4 raw = *reinterpret_cast<const uint4*>(g);
        v[0] = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
        v[1] = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
        v[2] = *reinterpret_cast<const __nv_bfloat162*>(&raw.z);
        v[3] = *reinterpret_cast<const __nv_bfloat162*>(&raw.w);
      } else {
        const __nv_bfloat16* gh = reinterpret_cast<const __nv_bfloat16*>(g);
        CUTE_UNROLL
        for (int i = 0; i < 4; ++i) {
          v[i] = __nv_bfloat162(gh[2 * i], gh[2 * i + 1]);
        }
      }
      uint16_t packed[4];
      CUTE_UNROLL
      for (int i = 0; i < 4; ++i) {
        packed[i] = f32x2_to_e4m3x2_rn_satfinite(__low2float(v[i]), __high2float(v[i]));
      }
      if (p.out_vec16) {
        // out_vec16 guarantees 16B-aligned rows; N_OUT + 8*cthr keeps 8B
        // alignment, so the 8-byte chunk goes out as one coalesced store.
        *reinterpret_cast<uint64_t*>(o) = *reinterpret_cast<const uint64_t*>(packed);
      } else {
        CUTE_UNROLL
        for (int i = 0; i < 4; ++i) {
          *reinterpret_cast<uint16_t*>(o + 2 * i) = packed[i];
        }
      }
    }
  }

  // -------------------------------------------------------------------------
  // Main device function
  // -------------------------------------------------------------------------
  static __device__ __forceinline__ void devfunc(const QprepBf16Fp8Sm90Params& p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
    const int m0 = blockIdx.x * BM;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int m_residue = p.num_tokens - m0;  // > 0 by grid construction

    extern __shared__ char smem_raw[];
    SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_raw);
    Tensor sA = make_tensor(make_smem_ptr(ss.a.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(ss.b.data()), SmemLayoutB{});

    const bf16* gA = reinterpret_cast<const bf16*>(p.q_nope) + (int64_t)m0 * p.a_s0 + (int64_t)h * p.a_s1;
    const bf16* gB = reinterpret_cast<const bf16*>(p.w_kc) + (int64_t)h * p.b_s0;
    const bf16* gR = reinterpret_cast<const bf16*>(p.q_rope) + (int64_t)m0 * p.r_s0 + (int64_t)h * p.r_s1;
    uint8_t* gO = reinterpret_cast<uint8_t*>(p.out) + (int64_t)m0 * p.o_s0 + (int64_t)h * p.o_s1;

    // Issue A tile + first B slab, then run the rope path over the async loads.
    load_a_tile(sA, gA, p.a_s0, m_residue, tid);
    load_b_slab(sB, gB, p.b_s2, 0, tid);
    cp_async_fence();

    rope_path(gR, gO, p, m_residue, tid);

    cp_async_wait<0>();
    __syncthreads();

    TiledMMA_t tiled_mma;
    Tensor acc = partition_fragment_C(tiled_mma, Shape<Int<BM>, Int<BN>>{});
    const int row_base = (tid / 128) * 64 + ((tid % 128) / 32) * 16 + ((tid % 32) / 4);
    const int col_base = (tid % 4) * 2;

    CUTE_UNROLL
    for (int nb = 0; nb < N_SLABS; ++nb) {
      gemm_ss(tiled_mma, sA, sB, acc, tid);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      __syncthreads();  // every thread's WGMMA drained -> sB reusable (WAR)
      if (nb + 1 < N_SLABS) {
        load_b_slab(sB, gB, p.b_s2, nb + 1, tid);
        cp_async_fence();
      }
      // fp8 store epilogue overlaps the next slab's cp.async.
      if (p.out_vec16) {
        store_slab_staged(acc, ss.c_stage.data(), gO, p.o_s0, nb * BN, row_base, col_base, m_residue, tid);
      } else {
        store_slab_direct(acc, gO, p.o_s0, nb * BN, row_base, col_base, m_residue);
      }
      if (nb + 1 < N_SLABS) {
        cp_async_wait<0>();
        __syncthreads();
      }
    }
#else
    if (cute::thread0()) {
      CUTE_INVALID_CONTROL_PATH("qprep_bf16_fp8_sm90 only supports sm90");
    }
#endif
  }

  // -------------------------------------------------------------------------
  // Host-side launch
  // -------------------------------------------------------------------------
  static void run(const QprepBf16Fp8Sm90Params& p) {
    QPREP_ASSERT(p.num_tokens > 0);
    QPREP_ASSERT(p.num_heads > 0);

    auto kernel = &qprep_bf16_fp8_kernel<QprepBf16Fp8Kernel<K_DIM>>;
    constexpr size_t smem_size = sizeof(SharedStorage);
    static bool attr_set = [&]() {
      QPREP_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      return true;
    }();
    (void)attr_set;

    dim3 grid(ceil_div_i(p.num_tokens, BM), p.num_heads, 1);
    kernel<<<grid, NUM_THREADS, smem_size, p.stream>>>(p);
    QPREP_CUDA_CHECK(cudaGetLastError());
  }
};

template <typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1)
    qprep_bf16_fp8_kernel(__grid_constant__ const QprepBf16Fp8Sm90Params params) {
  Kernel::devfunc(params);
}

template <int K_DIM>
void run_qprep_bf16_fp8_sm90(const QprepBf16Fp8Sm90Params& params) {
  QprepBf16Fp8Kernel<K_DIM>::run(params);
}

}  // namespace qprep_sm90
