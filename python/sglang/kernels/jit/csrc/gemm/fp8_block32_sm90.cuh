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

// SM90 M=1 block-[32, 32] FP8 GEMM.
//
// y[n] = sum_k dequant(x[k]) * dequant(W[n, k]), with one fp32 scale per
// 32-wide K group and per 32 output rows (the UE8M0 / block-32 contract).
//
// This is not the stock CUTLASS SM90 blockwise collective. That mainloop
// requires ScalePromotionInterval % 4 == 0, so the finest scale it can apply
// is one per K=128 (four FP8 WGMMA atoms). Here each K=32 WGMMA is scaled on
// its own and accumulated into a separate fp32 register, which is the same
// algorithm as the Triton block-[32, 32] fallback.
//
// Orientation matches a SWAP_AB GEMM: the weight tile is the WGMMA M=64
// operand and up to 8 activation rows fill the atom's N. One CTA owns 64
// output rows and 8 tokens; grid z walks the remaining tokens. A token past M
// stays zero and is not stored.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace sglang {

namespace fp8_block32_sm90_impl {

using namespace cute;

using fp8_t = cutlass::float_e4m3_t;

constexpr int kRows = 64;
constexpr int kAtomN = 8;
constexpr int kTileK = 64;  // two K=32 WGMMA atoms
constexpr int kThreads = 128;

using SmemLayoutA =
    decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<fp8_t>{}, Shape<Int<kRows>, Int<kTileK>>{}, Step<_1, _2>{}));
using SmemLayoutB =
    decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<fp8_t>{}, Shape<Int<kAtomN>, Int<kTileK>>{}, Step<_1, _2>{}));
using TiledMma = decltype(make_tiled_mma(GMMA::MMA_64x8x32_F32E4M3E4M3_SS_TN<>{}, Layout<Shape<_1, _1, _1>>{}));

struct SharedStorage {
  array_aligned<fp8_t, cosize_v<SmemLayoutA>, 128> a[2];
  array_aligned<fp8_t, cosize_v<SmemLayoutB>, 128> b[2];
};

struct Params {
  const fp8_t* __restrict__ act;
  const fp8_t* __restrict__ weight;
  const float* __restrict__ act_scale;
  const float* __restrict__ weight_scale;
  float* __restrict__ partial;
  bf16_t* __restrict__ out;
  int32_t m;
  int32_t n;
  int32_t k;
  int32_t scale_k;
  int32_t split_k;
};

__device__ __forceinline__ int64_t evict_first_policy() {
  int64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(policy) :);
  return policy;
}

__device__ __forceinline__ int64_t evict_last_policy() {
  int64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(policy) :);
  return policy;
}

__device__ __forceinline__ void cp_async_16_zfill(void* smem_dst, const void* gmem_src, bool pred, int64_t policy) {
  uint32_t dst = cute::cast_smem_ptr_to_uint(smem_dst);
  asm volatile(
      "cp.async.cg.shared.global.L2::cache_hint.L2::256B [%0], [%1], 16, %2, %3;\n" ::"r"(dst),
      "l"(gmem_src),
      "r"(pred ? 16 : 0),
      "l"(policy));
}

__device__ __forceinline__ void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

/// Load one 64x64 fp8 weight tile (logical columns past `k_end` are zero)
/// and 8 activation rows. Rows past M are zero-filled.
template <typename SmemA, typename SmemB>
__device__ __forceinline__ void load_tile(
    SmemA& sA,
    SmemB& sB,
    const fp8_t* __restrict__ weight,
    const fp8_t* __restrict__ act,
    int32_t n0,
    int32_t n,
    int32_t m0,
    int32_t m,
    int32_t k,
    int32_t k_base,
    int32_t k_end,
    int tid,
    int64_t policy_w,
    int64_t policy_a) {
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const int chunk = tid + pass * kThreads;
    const int row = chunk >> 2;
    const int col = (chunk & 3) << 4;
    const bool pred = (n0 + row) < n && (k_base + col + 16) <= k_end;
    const int32_t src_row = pred ? (n0 + row) : n0;
    const int32_t src_col = pred ? (k_base + col) : 0;
    cp_async_16_zfill(&sA(row, col), weight + static_cast<int64_t>(src_row) * k + src_col, pred, policy_w);
  }
  // 8 token rows x 4 chunks of 16 bytes. cp.async zfill clears tokens past M.
  if (tid < 32) {
    const int brow = tid >> 2;
    const int col = (tid & 3) << 4;
    const int32_t token = m0 + brow;
    const bool pred = token < m && (k_base + col + 16) <= k_end;
    const int32_t src_col = pred ? (k_base + col) : 0;
    const fp8_t* src = act + (pred ? static_cast<int64_t>(token) * k + src_col : 0);
    cp_async_16_zfill(&sB(brow, col), src, pred, policy_a);
  }
  cp_async_fence();
}

/// One K=32 WGMMA, then scale that atom into the running fp32 rows.
/// Thread t owns two token columns `(t%4)*2 + {0,1}` and two output rows.
template <typename SmemA, typename SmemB, typename Acc>
__device__ __forceinline__ void mma_scaled(
    TiledMma& tiled_mma,
    Acc& acc,
    SmemA const& sA,
    SmemB const& sB,
    int ki,
    int32_t m0,
    int32_t m,
    const float* __restrict__ act_scale,
    int32_t scale_k,
    int32_t kb,
    float weight_s,
    float run[4],
    int tid) {
  auto thr = tiled_mma.get_slice(tid);
  Tensor frag_a = thr.partition_fragment_A(sA);
  Tensor frag_b = thr.partition_fragment_B(sB);
  static_assert(rank(frag_a) == 3, "A fragment must be (atom, rest, K)");
  warpgroup_fence_operand(acc);
  warpgroup_arrive();
  tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
  cute::gemm(tiled_mma, frag_a(_, _, ki), frag_b(_, _, ki), acc);
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(acc);
  // m64n8 C layout: thread t owns rows (t/32)*16 + (t%32)/4 + {0,8} and
  // columns (t%4)*2 + {0,1}. Fragment index = 2*row_parity + col_parity.
  const int col0 = (tid & 3) * 2;
  const int32_t tok0 = m0 + col0;
  const int32_t tok1 = tok0 + 1;
  float s0 = 0.f;
  float s1 = 0.f;
  if (tok0 < m) {
    s0 = __ldg(act_scale + static_cast<int64_t>(tok0) * scale_k + kb) * weight_s;
  }
  if (tok1 < m) {
    s1 = __ldg(act_scale + static_cast<int64_t>(tok1) * scale_k + kb) * weight_s;
  }
  run[0] = fmaf(acc(0), s0, run[0]);
  run[1] = fmaf(acc(1), s1, run[1]);
  run[2] = fmaf(acc(2), s0, run[2]);
  run[3] = fmaf(acc(3), s1, run[3]);
}

__global__ void __launch_bounds__(kThreads) fp8_block32_sm90_kernel(Params p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
  const int tid = threadIdx.x;
  const int32_t n0 = static_cast<int32_t>(blockIdx.x) * kRows;
  const int32_t m0 = static_cast<int32_t>(blockIdx.z) * kAtomN;
  if (n0 >= p.n || m0 >= p.m) {
    return;
  }
  const int32_t chunk = p.k / p.split_k;
  const int32_t k0 = static_cast<int32_t>(blockIdx.y) * chunk;
  const int32_t k1 = k0 + chunk;

  extern __shared__ __align__(128) char smem_raw[];
  SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_raw);
  Tensor sA0 = make_tensor(make_smem_ptr(ss.a[0].data()), SmemLayoutA{});
  Tensor sA1 = make_tensor(make_smem_ptr(ss.a[1].data()), SmemLayoutA{});
  Tensor sB0 = make_tensor(make_smem_ptr(ss.b[0].data()), SmemLayoutB{});
  Tensor sB1 = make_tensor(make_smem_ptr(ss.b[1].data()), SmemLayoutB{});

  const int64_t policy_w = evict_first_policy();
  const int64_t policy_a = evict_last_policy();

  const int row_base = (tid >> 5) * 16 + ((tid & 31) >> 2);
  const int32_t g0 = n0 + row_base;
  const int32_t scale_row = (n0 >> 5) + ((tid >> 5) >= 2 ? 1 : 0);
  const int col0 = (tid & 3) * 2;
  const int32_t tok0 = m0 + col0;
  const int32_t tok1 = tok0 + 1;
  float run[4] = {0.f, 0.f, 0.f, 0.f};

  TiledMma tiled_mma;
  Tensor acc = partition_fragment_C(tiled_mma, Shape<Int<kRows>, Int<kAtomN>>{});

  auto consume = [&](int buf, int32_t k_base, int32_t valid) {
    Tensor sA = (buf == 0) ? sA0 : sA1;
    Tensor sB = (buf == 0) ? sB0 : sB1;
    const int32_t kb = k_base >> 5;
    const float ws0 = __ldg(p.weight_scale + static_cast<int64_t>(scale_row) * p.scale_k + kb);
    mma_scaled(tiled_mma, acc, sA, sB, 0, m0, p.m, p.act_scale, p.scale_k, kb, ws0, run, tid);
    if (valid == kTileK) {
      const float ws1 = __ldg(p.weight_scale + static_cast<int64_t>(scale_row) * p.scale_k + kb + 1);
      mma_scaled(tiled_mma, acc, sA, sB, 1, m0, p.m, p.act_scale, p.scale_k, kb + 1, ws1, run, tid);
    }
  };

  int buf = 0;
  {
    const int32_t valid0 = k1 - k0 < kTileK ? (k1 - k0) : kTileK;
    if (buf == 0) {
      load_tile(sA0, sB0, p.weight, p.act, n0, p.n, m0, p.m, p.k, k0, k0 + valid0, tid, policy_w, policy_a);
    }
  }
  cp_async_wait<0>();
  __syncthreads();

  for (int32_t k_base = k0; k_base < k1;) {
    const int32_t valid = k1 - k_base < kTileK ? (k1 - k_base) : kTileK;
    const int32_t next = k_base + valid;
    if (next < k1) {
      const int32_t valid_n = k1 - next < kTileK ? (k1 - next) : kTileK;
      if (buf == 0) {
        load_tile(sA1, sB1, p.weight, p.act, n0, p.n, m0, p.m, p.k, next, next + valid_n, tid, policy_w, policy_a);
      } else {
        load_tile(sA0, sB0, p.weight, p.act, n0, p.n, m0, p.m, p.k, next, next + valid_n, tid, policy_w, policy_a);
      }
    }
    consume(buf, k_base, valid);
    if (next < k1) {
      cp_async_wait<0>();
      __syncthreads();
      buf ^= 1;
    }
    k_base = next;
  }

  auto store_one = [&](int32_t token, int32_t row, float value) {
    if (token >= p.m || row >= p.n) {
      return;
    }
    if (p.split_k == 1) {
      p.out[static_cast<int64_t>(token) * p.n + row] = __float2bfloat16(value);
    } else {
      float* dst = p.partial + (static_cast<int64_t>(blockIdx.y) * p.m + token) * p.n;
      dst[row] = value;
    }
  };
  store_one(tok0, g0, run[0]);
  store_one(tok1, g0, run[1]);
  store_one(tok0, g0 + 8, run[2]);
  store_one(tok1, g0 + 8, run[3]);
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("fp8_block32_sm90 requires sm90\n");
  }
#endif
}

// Large-M tile. The M=8 kernel reloads the weight once per 8 tokens, so a
// prefill GEMM (M in the thousands) is bandwidth-bound on H20 even when the
// arithmetic intensity of the whole product is high. Triton uses 64x128;
// these tiles match that and, at M>=128, stack two warpgroups on M so one
// weight K-slice feeds 128 tokens.
//
// One K step is one FP8 WGMMA atom (K=32) and one fp32 scale. The running
// sum stays in registers beside the MMA accumulator: block scales change
// every atom, so they cannot be folded into a single hardware accumulator.
template <int kBM, int kStages, int kGroup>
struct LargeTile {
  static constexpr int kBN = 128;
  static constexpr int kBK = 32;
  static constexpr int kThreads = kBM * 2;
  static constexpr int kAcc = 64;  // 64 rows x 128 cols / 128 threads
  static_assert(kBM == 64 || kBM == 128, "BM is one or two warpgroups");
  static_assert(kStages >= 2 && kStages <= 5, "pipeline depth");

  using LayoutA =
      decltype(tile_to_shape(GMMA::Layout_K_SW32_Atom<fp8_t>{}, Shape<Int<kBM>, Int<kBK>>{}, Step<_1, _2>{}));
  using LayoutB =
      decltype(tile_to_shape(GMMA::Layout_K_SW32_Atom<fp8_t>{}, Shape<Int<kBN>, Int<kBK>>{}, Step<_1, _2>{}));
  using Atom = GMMA::MMA_64x128x32_F32E4M3E4M3_SS_TN<>;
  using TiledMma = std::conditional_t<
      kBM == 128,
      decltype(make_tiled_mma(Atom{}, Layout<Shape<_2, _1, _1>>{})),
      decltype(make_tiled_mma(Atom{}, Layout<Shape<_1, _1, _1>>{}))>;

  struct SharedStorage {
    array_aligned<fp8_t, cosize_v<LayoutA>, 128> a[kStages];
    array_aligned<fp8_t, cosize_v<LayoutB>, 128> b[kStages];
  };

  static constexpr size_t kPipeBytes = sizeof(SharedStorage);
  static constexpr size_t kEpiBytes = sizeof(bf16_t) * kBM * kBN;
  static constexpr size_t kSmem = kPipeBytes > kEpiBytes ? kPipeBytes : kEpiBytes;
  static_assert(kSmem <= 48 * 1024, "large tile must fit the default 48KB smem");
};

struct LargeParams {
  const fp8_t* __restrict__ act;
  const fp8_t* __restrict__ weight;
  const float* __restrict__ act_scale;
  const float* __restrict__ weight_scale;
  bf16_t* __restrict__ out;
  int32_t m;
  int32_t n;
  int32_t k;
  int32_t scale_k;
};

template <int kBM, int kStages, int kGroup>
__global__ void __launch_bounds__(LargeTile<kBM, kStages, kGroup>::kThreads)
    fp8_block32_sm90_large_kernel(LargeParams p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
  using Tile = LargeTile<kBM, kStages, kGroup>;
  constexpr int kBN = Tile::kBN;
  constexpr int kThreads = Tile::kThreads;
  const int tid = threadIdx.x;
  const int tiles_m = (p.m + kBM - 1) / kBM;
  const int tiles_n = (p.n + kBN - 1) / kBN;
  const int total = tiles_m * tiles_n;
  const int pid = static_cast<int>(blockIdx.x);
  if (pid >= total) {
    return;
  }

  // Group M tiles so a weight N-slab stays in L2 across several tokens.
  const int span = kGroup * tiles_n;
  const int group = pid / span;
  const int first_m = group * kGroup;
  int group_m = tiles_m - first_m;
  if (group_m > kGroup) {
    group_m = kGroup;
  }
  const int local = pid - group * span;
  const int pid_m = first_m + (local % group_m);
  const int pid_n = local / group_m;
  const int32_t m0 = pid_m * kBM;
  const int32_t n0 = pid_n * kBN;
  const int k_tiles = p.k >> 5;

  extern __shared__ __align__(128) char smem_raw[];
  typename Tile::SharedStorage& ss = *reinterpret_cast<typename Tile::SharedStorage*>(smem_raw);

  const int64_t policy = evict_last_policy();
  const int wg = tid >> 7;
  const int lane = tid & 127;
  const int row0 = wg * 64 + (lane >> 5) * 16 + ((lane & 31) >> 2);
  const int row1 = row0 + 8;
  const int col_base = (lane & 3) * 2;
  const int32_t tok0 = m0 + row0;
  const int32_t tok1 = m0 + row1;

  typename Tile::TiledMma tiled_mma;
  Tensor acc = partition_fragment_C(tiled_mma, Shape<Int<kBM>, Int<kBN>>{});
  float run[Tile::kAcc];
#pragma unroll
  for (int i = 0; i < Tile::kAcc; ++i) {
    run[i] = 0.f;
  }

  auto issue = [&](int kt) {
    const int stage = kt % kStages;
    const int32_t k_base = kt << 5;
    Tensor sA = make_tensor(make_smem_ptr(ss.a[stage].data()), typename Tile::LayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(ss.b[stage].data()), typename Tile::LayoutB{});
    {
      const int row = tid >> 1;
      const int col = (tid & 1) << 4;
      const bool pred = (m0 + row) < p.m;
      const fp8_t* src = p.act + (pred ? (static_cast<int64_t>(m0 + row) * p.k + k_base + col) : 0);
      cp_async_16_zfill(&sA(row, col), src, pred, policy);
    }
#pragma unroll
    for (int it = 0; it < kBN / kBM; ++it) {
      const int idx = tid + it * kThreads;
      const int row = idx >> 1;
      const int col = (idx & 1) << 4;
      const bool pred = (n0 + row) < p.n;
      const fp8_t* src = p.weight + (pred ? (static_cast<int64_t>(n0 + row) * p.k + k_base + col) : 0);
      cp_async_16_zfill(&sB(row, col), src, pred, policy);
    }
    cp_async_fence();
  };

  auto consume = [&](int kt) {
    const int stage = kt % kStages;
    float as0 = 0.f;
    float as1 = 0.f;
    if (tok0 < p.m) {
      as0 = __ldg(p.act_scale + static_cast<int64_t>(tok0) * p.scale_k + kt);
    }
    if (tok1 < p.m) {
      as1 = __ldg(p.act_scale + static_cast<int64_t>(tok1) * p.scale_k + kt);
    }
    const int scale_n0 = n0 >> 5;
    float ws[4];
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const int32_t col = n0 + (g << 5);
      ws[g] = col < p.n ? __ldg(p.weight_scale + static_cast<int64_t>(scale_n0 + g) * p.scale_k + kt) : 0.f;
    }

    Tensor sA = make_tensor(make_smem_ptr(ss.a[stage].data()), typename Tile::LayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(ss.b[stage].data()), typename Tile::LayoutB{});
    auto thr = tiled_mma.get_slice(tid);
    Tensor frag_a = thr.partition_fragment_A(sA);
    Tensor frag_b = thr.partition_fragment_B(sB);
    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    cute::gemm(tiled_mma, frag_a(_, _, 0), frag_b(_, _, 0), acc);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(acc);

#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float s0 = as0 * ws[j >> 2];
      const float s1 = as1 * ws[j >> 2];
      run[4 * j + 0] = fmaf(acc(4 * j + 0), s0, run[4 * j + 0]);
      run[4 * j + 1] = fmaf(acc(4 * j + 1), s0, run[4 * j + 1]);
      run[4 * j + 2] = fmaf(acc(4 * j + 2), s1, run[4 * j + 2]);
      run[4 * j + 3] = fmaf(acc(4 * j + 3), s1, run[4 * j + 3]);
    }
  };

#pragma unroll
  for (int i = 0; i < kStages - 1; ++i) {
    if (i < k_tiles) {
      issue(i);
    }
  }
  for (int kt = 0; kt < k_tiles; ++kt) {
    const int nxt = kt + (kStages - 1);
    if (nxt < k_tiles) {
      issue(nxt);
      cp_async_wait<kStages - 1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    consume(kt);
    __syncthreads();
  }

  // Stage the fragmented WGMMA layout, then flush 16-byte stores. Direct
  // pair stores scatter 4-byte sectors and lose the short-K shapes.
  bf16_t* tile_out = reinterpret_cast<bf16_t*>(smem_raw);
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int col = col_base + 8 * j;
    tile_out[row0 * kBN + col] = __float2bfloat16(run[4 * j]);
    tile_out[row0 * kBN + col + 1] = __float2bfloat16(run[4 * j + 1]);
    tile_out[row1 * kBN + col] = __float2bfloat16(run[4 * j + 2]);
    tile_out[row1 * kBN + col + 1] = __float2bfloat16(run[4 * j + 3]);
  }
  __syncthreads();

  constexpr int kVec = 8;  // 8 bf16 = 16 bytes
  constexpr int kVecs = (kBM * kBN) / kVec;
  for (int v = tid; v < kVecs; v += kThreads) {
    const int elem = v * kVec;
    const int row = elem / kBN;
    const int col = elem - row * kBN;
    const int32_t gm = m0 + row;
    const int32_t gn = n0 + col;
    if (gm >= p.m) {
      continue;
    }
    bf16_t* src = tile_out + elem;
    bf16_t* dst = p.out + static_cast<int64_t>(gm) * p.n + gn;
    if (gn + kVec <= p.n) {
      *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    } else {
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        if (gn + i < p.n) {
          dst[i] = src[i];
        }
      }
    }
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("fp8_block32_sm90_large requires sm90\n");
  }
#endif
}

__global__ void fp8_block32_sm90_reduce(
    bf16_t* __restrict__ out, const float* __restrict__ partial, int32_t m, int32_t n, int32_t split_k) {
  const int64_t mn = static_cast<int64_t>(m) * n;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= mn) {
    return;
  }
  float sum = 0.f;
  for (int32_t s = 0; s < split_k; ++s) {
    sum += partial[static_cast<int64_t>(s) * mn + i];
  }
  out[i] = __float2bfloat16(sum);
}

}  // namespace fp8_block32_sm90_impl

/// \brief Block-[32, 32] FP8 GEMM on SM90. Each CTA consumes 8 tokens.
///
/// \param act `[M, K]` fp8 e4m3, K contiguous.
/// \param weight `[N, K]` fp8 e4m3, K contiguous.
/// \param act_scale `[M, K/32]` fp32 decoded UE8M0 (or any fp32 group scale).
/// \param weight_scale `[ceil(N/32), K/32]` fp32, same layout as the Triton path.
/// \param partial `[split_k, M, N]` fp32 scratch, written by the mainloop.
/// \param out `[M, N]` bf16.
/// \param split_k K-split. `K % split_k == 0` and `(K / split_k) % 32 == 0`.
void fp8_block32_sm90(
    tvm::ffi::TensorView act,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView act_scale,
    tvm::ffi::TensorView weight_scale,
    tvm::ffi::TensorView partial,
    tvm::ffi::TensorView out,
    int64_t split_k) {
  using namespace host;
  using namespace fp8_block32_sm90_impl;

  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  auto M = SymbolicSize{"M"};
  auto N = SymbolicSize{"N"};
  auto K = SymbolicSize{"K"};
  auto SK = SymbolicSize{"SK"};
  auto SN = SymbolicSize{"SN"};
  auto SP = SymbolicSize{"SP"};

  TensorMatcher({M, K}).with_dtype<fp8_e4m3_t>().with_strides({K, 1}).with_device(device).ensure_alignment(16).verify(
      act);
  TensorMatcher({N, K}).with_dtype<fp8_e4m3_t>().with_strides({K, 1}).with_device(device).ensure_alignment(16).verify(
      weight);
  TensorMatcher({M, SK}).with_dtype<fp32_t>().with_strides({SK, 1}).with_device(device).verify(act_scale);
  TensorMatcher({SN, SK}).with_dtype<fp32_t>().with_strides({SK, 1}).with_device(device).verify(weight_scale);
  // split_k == 1 writes bf16 directly and never touches `partial`.
  if (split_k > 1) {
    TensorMatcher({SP, M, N}).with_dtype<fp32_t>().with_device(device).verify(partial);
  }
  TensorMatcher({M, N}).with_dtype<bf16_t>().with_strides({N, 1}).with_device(device).verify(out);

  const int32_t m = static_cast<int32_t>(M.unwrap());
  const int32_t n = static_cast<int32_t>(N.unwrap());
  const int32_t k = static_cast<int32_t>(K.unwrap());
  const int32_t scale_k = static_cast<int32_t>(SK.unwrap());
  const int32_t scale_n = static_cast<int32_t>(SN.unwrap());
  const int32_t sp = split_k > 1 ? static_cast<int32_t>(SP.unwrap()) : 1;
  CHECK_HOST(split_k >= 1) << "split_k must be >= 1";
  CHECK_HOST(k > 0 && k % 32 == 0) << "K must be a positive multiple of 32, got " << k;
  CHECK_HOST(k % split_k == 0) << "split_k must divide K";
  CHECK_HOST((k / split_k) % 32 == 0) << "each K split must be a multiple of 32";
  CHECK_HOST(scale_k * 32 == k) << "act_scale K groups do not match K";
  CHECK_HOST(scale_n == static_cast<int32_t>(div_ceil(static_cast<uint64_t>(n), 32)))
      << "weight_scale rows do not match N";
  CHECK_HOST(sp == split_k) << "partial split dim must equal split_k";
  CHECK_HOST(n > 0 && m > 0) << "M and N must be positive";

  Params params{
      .act = reinterpret_cast<const fp8_t*>(act.data_ptr()),
      .weight = reinterpret_cast<const fp8_t*>(weight.data_ptr()),
      .act_scale = static_cast<const float*>(act_scale.data_ptr()),
      .weight_scale = static_cast<const float*>(weight_scale.data_ptr()),
      .partial = split_k > 1 ? static_cast<float*>(partial.data_ptr()) : nullptr,
      .out = static_cast<bf16_t*>(out.data_ptr()),
      .m = m,
      .n = n,
      .k = k,
      .scale_k = scale_k,
      .split_k = static_cast<int32_t>(split_k),
  };

  constexpr size_t kSmem = sizeof(SharedStorage);
  static_assert(kSmem <= 48 * 1024, "fp8_block32_sm90 smem must fit the default 48KB limit");
  const uint32_t tiles_n = static_cast<uint32_t>(div_ceil(static_cast<uint64_t>(n), kRows));
  const uint32_t tiles_m = static_cast<uint32_t>(div_ceil(static_cast<uint64_t>(m), kAtomN));
  LaunchKernel(dim3(tiles_n, static_cast<uint32_t>(split_k), tiles_m), dim3(kThreads), device.unwrap(), kSmem)(
      fp8_block32_sm90_kernel, params);

  if (split_k > 1) {
    const uint32_t reduce_threads = 256;
    const int64_t mn = static_cast<int64_t>(m) * n;
    const uint32_t reduce_blocks = static_cast<uint32_t>(div_ceil(static_cast<uint64_t>(mn), reduce_threads));
    LaunchKernel(reduce_blocks, reduce_threads, device.unwrap())(
        fp8_block32_sm90_reduce,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const float*>(partial.data_ptr()),
        m,
        n,
        static_cast<int32_t>(split_k));
  }
}

/// \brief 64x128 block-[32, 32] FP8 GEMM for prefill-sized M.
/// Four cp.async stages, L2-grouped raster of 16 M-tiles.
void fp8_block32_sm90_large(
    tvm::ffi::TensorView act,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView act_scale,
    tvm::ffi::TensorView weight_scale,
    tvm::ffi::TensorView out) {
  using namespace host;
  using namespace fp8_block32_sm90_impl;

  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  auto M = SymbolicSize{"M"};
  auto N = SymbolicSize{"N"};
  auto K = SymbolicSize{"K"};
  auto SK = SymbolicSize{"SK"};
  auto SN = SymbolicSize{"SN"};

  TensorMatcher({M, K}).with_dtype<fp8_e4m3_t>().with_strides({K, 1}).with_device(device).ensure_alignment(16).verify(
      act);
  TensorMatcher({N, K}).with_dtype<fp8_e4m3_t>().with_strides({K, 1}).with_device(device).ensure_alignment(16).verify(
      weight);
  TensorMatcher({M, SK}).with_dtype<fp32_t>().with_strides({SK, 1}).with_device(device).verify(act_scale);
  TensorMatcher({SN, SK}).with_dtype<fp32_t>().with_strides({SK, 1}).with_device(device).verify(weight_scale);
  TensorMatcher({M, N}).with_dtype<bf16_t>().with_strides({N, 1}).with_device(device).ensure_alignment(16).verify(out);

  const int32_t m = static_cast<int32_t>(M.unwrap());
  const int32_t n = static_cast<int32_t>(N.unwrap());
  const int32_t k = static_cast<int32_t>(K.unwrap());
  const int32_t scale_k = static_cast<int32_t>(SK.unwrap());
  const int32_t scale_n = static_cast<int32_t>(SN.unwrap());
  CHECK_HOST(k > 0 && k % 32 == 0) << "K must be a positive multiple of 32, got " << k;
  CHECK_HOST(scale_k * 32 == k) << "act_scale K groups do not match K";
  CHECK_HOST(scale_n == static_cast<int32_t>(div_ceil(static_cast<uint64_t>(n), 32)))
      << "weight_scale rows do not match N";
  CHECK_HOST(n > 0 && m > 0) << "M and N must be positive";

  LargeParams params{
      .act = reinterpret_cast<const fp8_t*>(act.data_ptr()),
      .weight = reinterpret_cast<const fp8_t*>(weight.data_ptr()),
      .act_scale = static_cast<const float*>(act_scale.data_ptr()),
      .weight_scale = static_cast<const float*>(weight_scale.data_ptr()),
      .out = static_cast<bf16_t*>(out.data_ptr()),
      .m = m,
      .n = n,
      .k = k,
      .scale_k = scale_k,
  };

  using Tile = LargeTile<64, 4, 16>;
  const uint32_t tiles_m = static_cast<uint32_t>(div_ceil(static_cast<uint64_t>(m), 64));
  const uint32_t tiles_n = static_cast<uint32_t>(div_ceil(static_cast<uint64_t>(n), 128));
  LaunchKernel(tiles_m * tiles_n, static_cast<uint32_t>(Tile::kThreads), device.unwrap(), Tile::kSmem)(
      fp8_block32_sm90_large_kernel<64, 4, 16>, params);
}

}  // namespace sglang
