/*
 * Adapted from Tencent HPC-Ops (MIT License):
 * https://github.com/Tencent/hpc-ops/blob/main/src/gemm/sm90/gemm_bf16xfp32.cu
 * https://github.com/Tencent/hpc-ops/blob/main/src/gemm/sm90/entry.cc
 * https://github.com/Tencent/hpc-ops/blob/main/src/utils/utils.cuh
 * Copyright (C) 2026 Tencent. All rights reserved.
 *
 * SM90 warp-specialized GEMM computing y[m, n] = x[m, k] @ w[n, k]^T where x
 * is bf16 and w is fp32. The fp32 weight is pre-split on the Python side into
 * two bf16 tensors:
 *   w_high = w.to(bf16)
 *   w_low  = ((w - w_high.float()) / scale).to(bf16)
 * and each K-tile issues two GMMAs (low then high) that the epilogue combines
 * as y = y_high + scale * y_low, recovering near-fp32 accuracy at bf16 GMMA
 * throughput. Optionally splits K across CTAs (split-k) with an in-kernel
 * flag-based reduction.
 *
 * The launch configuration (tile sizes, stages, warpgroups, split-k) is
 * selected by the shape heuristic in sglang/jit_kernel/gemm_bf16xfp32.py and
 * baked into this translation unit as template arguments.
 */
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>

#include <tvm/ffi/container/tensor.h>

#include "cute/tensor.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/fast_math.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_bf16.h>
#include <type_traits>

namespace {

namespace gemm_bf16xfp32 {

// ---------------------------------------------------------------------------
// Device helpers (subset of hpc-ops src/utils/utils.cuh)
// ---------------------------------------------------------------------------

template <typename T, int N>
struct vec_t {
  T data[N];

  __device__ __forceinline__ constexpr T& operator[](int idx) {
    return data[idx];
  }

  __device__ __forceinline__ constexpr const T& operator[](int idx) const {
    return data[idx];
  }
};

template <typename U, typename T, int N>
__device__ __forceinline__ constexpr auto to(const vec_t<T, N>& v) {
  if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __nv_bfloat16>) {
    vec_t<__nv_bfloat16, N> o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      o[i] = __float2bfloat16(v[i]);
    }
    return o;
  } else if constexpr (std::is_same_v<T, U>) {
    return v;
  }
}

template <typename T, int N>
__device__ __forceinline__ constexpr auto load(const void* ptr) {
  vec_t<T, N> v;

  constexpr int kBytes = sizeof(T) * N;
  static_assert(kBytes == 4 || kBytes == 8 || kBytes == 16, "not support for T x N");

  if constexpr (kBytes == 4) {
    *reinterpret_cast<uint32_t*>(&v) = *reinterpret_cast<const uint32_t*>(ptr);
  } else if constexpr (kBytes == 8) {
    *reinterpret_cast<uint64_t*>(&v) = *reinterpret_cast<const uint64_t*>(ptr);
  } else if constexpr (kBytes == 16) {
    *reinterpret_cast<uint4*>(&v) = *reinterpret_cast<const uint4*>(ptr);
  }

  return v;
}

template <typename T, int N>
__device__ __forceinline__ constexpr void store(void* ptr, const vec_t<T, N>& v) {
  constexpr int kBytes = sizeof(T) * N;
  static_assert(kBytes == 4 || kBytes == 8 || kBytes == 16, "not support for T x N");

  if constexpr (kBytes == 4) {
    *reinterpret_cast<uint32_t*>(ptr) = *reinterpret_cast<const uint32_t*>(&v);
  } else if constexpr (kBytes == 8) {
    *reinterpret_cast<uint64_t*>(ptr) = *reinterpret_cast<const uint64_t*>(&v);
  } else if constexpr (kBytes == 16) {
    *reinterpret_cast<uint4*>(ptr) = *reinterpret_cast<const uint4*>(&v);
  }
}

__device__ int __forceinline__ load_global_volatile(int* ptr) {
  int val;
  asm volatile("ld.volatile.global.s32 {%0}, [%1];\n" : "=r"(val) : "l"(ptr));
  return val;
}

__device__ __forceinline__ void syncwarpgroup(int barrier_id) {
  asm volatile("barrier.cta.sync %0, 128;\n" ::"r"(barrier_id) : "memory");
}

template <int N>
__device__ __forceinline__ void bar_sync(int barrier_id) {
  asm volatile("barrier.cta.sync %0, %1;\n" ::"r"(barrier_id), "n"(N) : "memory");
}

__device__ __forceinline__ void fence_async_global() {
  asm volatile("fence.proxy.async.global;\n");
}

// ---------------------------------------------------------------------------
// Kernel (hpc-ops src/gemm/sm90/gemm_bf16xfp32.cu)
// ---------------------------------------------------------------------------

template <int kBlockSwizzle, int kSplitK>
__device__ __forceinline__ auto get_next_tile(
    int iblock, int num_tile_m, int num_tile_n, cutlass::FastDivmod swizzle_divider, cutlass::FastDivmod flat_divider) {
  int itile_m, itile_n;
  int num_tile_bxn = kBlockSwizzle * num_tile_n * kSplitK;
  int total_sizzle_blocks = num_tile_m / kBlockSwizzle * num_tile_bxn;

  if (iblock >= total_sizzle_blocks) {
    flat_divider(itile_m, itile_n, iblock);
  } else {
    int i_bxn, i_bxn_res;
    swizzle_divider(i_bxn, i_bxn_res, iblock);

    itile_m = i_bxn * kBlockSwizzle + i_bxn_res % kBlockSwizzle;
    itile_n = i_bxn_res / kBlockSwizzle;
  }

  int ichunk = itile_n % kSplitK;
  itile_n = itile_n / kSplitK;

  return cute::make_tuple(itile_m, itile_n, ichunk);
}

template <typename Tout, int kTileM, int kTileN, int kSplitK, int kWarpCount>
__device__ __forceinline__ void
splitk_reduce(Tout* y_ptr, float* splitk_y_ptr, int m, int n, int itile_m, int itile_n) {
  int iwarp = threadIdx.x / 32;
  int ilane = threadIdx.x % 32;

  if (itile_m * kTileM + iwarp >= m) {
    return;
  }

  if (ilane * 4 >= kTileN || itile_n * kTileN + ilane * 4 >= n) {
    return;
  }

  auto* y_tile = y_ptr + (itile_m * kTileM + iwarp) * n + itile_n * kTileN + ilane * 4;
  auto* splitk_y_tile = splitk_y_ptr + (itile_m * kTileM + iwarp) * n + itile_n * kTileN + ilane * 4;

  int local_m = m - (itile_m * kTileM + iwarp);

#pragma unroll
  for (int irow = 0; irow < kTileM; irow += kWarpCount) {
    if (irow >= local_m) {
      return;
    }
    auto y = load<float, 4>(splitk_y_tile + irow * n);

#pragma unroll
    for (int ichunk = 1; ichunk < kSplitK; ++ichunk) {
      auto split_y = load<float, 4>(splitk_y_tile + ichunk * m * n + irow * n);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        y[i] += split_y[i];
      }
    }

    store(y_tile + irow * n, to<Tout>(y));
  }
}

template <
    typename Tin,
    typename TY,
    typename Tout,
    typename TiledMma,
    typename TmaX,
    typename TmaWH,
    typename TmaWL,
    typename TmaY,
    int kTileM,
    int kTileN,
    int kTileK,
    int kStage,
    int kWarpGroupN,
    typename SLayoutX,
    typename SLayoutW,
    typename SLayoutY,
    int kBlockSwizzle,
    int kSplitK>
__global__ void __launch_bounds__(128 * (kWarpGroupN + 1), 1) gemm_bf16xfp32_kernel(
    const __grid_constant__ TmaX tma_x,
    const __grid_constant__ TmaWH tma_wh,
    const __grid_constant__ TmaWL tma_wl,
    const __grid_constant__ TmaY tma_y,
    Tout* y_ptr,
    float* splitk_y_ptr,
    int* split_flag_ptr,
    int m,
    int n,
    int k,
    float scale,
    cutlass::FastDivmod swizzle_divider,
    cutlass::FastDivmod flat_divider,
    cutlass::FastDivmod reduce_flat_divider) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;

  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  int elected = cute::elect_one_sync();
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  constexpr int kWLIdx = 0;
  constexpr int kWHIdx = 1;

  __shared__ uint64_t writable_x[kStage];
  __shared__ uint64_t readable_x[kStage];

  __shared__ uint64_t writable_w[kStage][kWarpGroupN][2];
  __shared__ uint64_t readable_w[kStage][kWarpGroupN][2];

  extern __shared__ uint8_t shm_data[] alignas(128);
  auto* shm_x = (Tin*)shm_data;
  auto* shm_w = (Tin*)shm_x + cosize(SLayoutX{});
  auto* shm_y = (TY*)(shm_w + cosize(SLayoutW{}));

  auto sX = make_tensor(make_smem_ptr(shm_x), SLayoutX{});
  auto sW = make_tensor(make_smem_ptr(shm_w), SLayoutW{});

  auto gX = tma_x.get_tma_tensor(make_shape(m, k));
  auto gWH = tma_wh.get_tma_tensor(make_shape(n, k));
  auto gWL = tma_wl.get_tma_tensor(make_shape(n, k));

  auto gY = make_tensor(
      make_gmem_ptr((float*)(nullptr)), make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));

  auto btma_x = tma_x.get_slice(0);
  auto btma_wh = tma_wh.get_slice(0);
  auto btma_wl = tma_wl.get_slice(0);

  auto tXg = btma_x.partition_S(gX);  // (TMA, TMA_M, TMA_K)
  auto tXs = btma_x.partition_D(sX);  // (TMA, _1, _1, kStage)

  auto tWHg = btma_wh.partition_S(gWH);  // (TMA, TMA_N, TMA_K)
  auto tWHs = btma_wh.partition_D(sW);   // (TMA, _1, _1, kStage)

  auto tWLg = btma_wl.partition_S(gWL);  // (TMA, TMA_N, TMA_K)
  auto tWLs = btma_wl.partition_D(sW);   // (TMA, _1, _1, kStage)

  int num_tile_m = size<1>(tXg);
  int num_tile_n = (size<1>(tWHg) + kWarpGroupN - 1) / kWarpGroupN;

  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable_x[i], 1);
      initialize_barrier(writable_x[i], kWarpGroupN);
    }
#pragma unroll
    for (int istage = 0; istage < kStage; ++istage) {
#pragma unroll
      for (int j = 0; j < kWarpGroupN; ++j) {
        initialize_barrier(readable_w[istage][j][kWLIdx], 1);
        initialize_barrier(readable_w[istage][j][kWHIdx], 1);
        initialize_barrier(writable_w[istage][j][kWLIdx], 1);
        initialize_barrier(writable_w[istage][j][kWHIdx], 1);
      }
    }
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  // load warpgroup
  if (idx >= kWarpGroupN * 128) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= kWarpGroupN * 128;
    constexpr int kTransactionBytesX = sizeof(Tin) * kTileK * kTileM;
    constexpr int kTransactionBytesW = sizeof(Tin) * kTileK * kTileN;

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    if (is_leader_in_load) {
      int phase = 1;  // start with ok
      int ismem_write = __shfl_sync(0xFFFFFFFF, 0, 0);
      int iblock = blockIdx.x;
      int ntile_k = size<2>(tXg);

      while (true) {
        auto [itile_m, itile_n, ichunk] =
            get_next_tile<kBlockSwizzle, kSplitK>(iblock, num_tile_m, num_tile_n, swizzle_divider, flat_divider);

        if (itile_m >= num_tile_m) {
          break;
        }

        iblock += gridDim.x;

#pragma unroll 1
        for (int itile_k = ichunk; itile_k < ntile_k; itile_k += kSplitK) {
          // load a
          wait_barrier(writable_x[ismem_write], phase);
          cute::copy(tma_x.with(readable_x[ismem_write]), tXg(_, itile_m, itile_k), tXs(_, 0, 0, ismem_write));
          set_barrier_transaction_bytes(readable_x[ismem_write], kTransactionBytesX);
          // load wgX low
#pragma unroll
          for (int wg = 0; wg < kWarpGroupN; ++wg) {
            wait_barrier(writable_w[ismem_write][wg][kWLIdx], phase);
            cute::copy(
                tma_wl.with(readable_w[ismem_write][wg][kWLIdx]),
                tWLg(_, kWarpGroupN * itile_n + wg, itile_k),
                tWLs(_, 0, 0, wg, kWLIdx, ismem_write));
            set_barrier_transaction_bytes(readable_w[ismem_write][wg][kWLIdx], kTransactionBytesW);
          }
          // load wgX high
#pragma unroll
          for (int wg = 0; wg < kWarpGroupN; ++wg) {
            wait_barrier(writable_w[ismem_write][wg][kWHIdx], phase);
            cute::copy(
                tma_wh.with(readable_w[ismem_write][wg][kWHIdx]),
                tWHg(_, kWarpGroupN * itile_n + wg, itile_k),
                tWHs(_, 0, 0, wg, kWHIdx, ismem_write));
            set_barrier_transaction_bytes(readable_w[ismem_write][wg][kWHIdx], kTransactionBytesW);
          }

          ++ismem_write;
          if (ismem_write == kStage) {
            ismem_write = 0;
            phase ^= 1;
          }
        }
      }
    }
  } else {
    // math warpgroup
    cutlass::arch::warpgroup_reg_alloc<168>();

    int idx_in_warpgroup = idx % 128;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    TiledMma tiled_mma;

    auto thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    auto tWs4r = thr_mma.partition_A(sW);
    auto tXs4r = thr_mma.partition_B(sX);

    auto tWr = thr_mma.make_fragment_A(tWs4r);  // (MMA, MMA_M, MMA_K, kStage)
    auto tXr = thr_mma.make_fragment_B(tXs4r);  // (MMA, MMA_N, MMA_K, kStage)

    auto tYr_low = thr_mma.partition_fragment_C(gY);
    auto tYr_high = make_tensor_like(tYr_low);

    int ismem_read = 0;
    int phase = 0;

    int iblock = blockIdx.x;
    int last_tile_m = -1;
    int last_tile_n = -1;
    while (true) {
      auto [itile_m, itile_n, ichunk] =
          get_next_tile<kBlockSwizzle, kSplitK>(iblock, num_tile_m, num_tile_n, swizzle_divider, flat_divider);
      if (itile_m >= num_tile_m) {
        break;
      }
      iblock += gridDim.x;

      clear(tYr_low);
      clear(tYr_high);

      int ntile_k = size<2>(tXg);

      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
#pragma unroll 1
      for (int itilek = ichunk; itilek < ntile_k; itilek += kSplitK) {
        wait_barrier(readable_x[ismem_read], phase);

        // mma low
        wait_barrier(readable_w[ismem_read][iwarpgroup][kWLIdx], phase);
        warpgroup_fence_operand(tYr_low);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tXr); ++ik) {
          cute::gemm(
              tiled_mma, tWr(_, _, ik, iwarpgroup, kWLIdx, ismem_read), tXr(_, _, ik, ismem_read), tYr_low(_, _, _));
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tYr_low);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_w[ismem_read][iwarpgroup][kWLIdx]);
        }

        // mma high
        wait_barrier(readable_w[ismem_read][iwarpgroup][kWHIdx], phase);
        warpgroup_fence_operand(tYr_high);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tXr); ++ik) {
          cute::gemm(
              tiled_mma, tWr(_, _, ik, iwarpgroup, kWHIdx, ismem_read), tXr(_, _, ik, ismem_read), tYr_high(_, _, _));
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tYr_high);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_x[ismem_read]);
          arrive_barrier(writable_w[ismem_read][iwarpgroup][kWHIdx]);
        }

        ++ismem_read;
        if (ismem_read == kStage) {
          phase ^= 1;
          ismem_read = 0;
        }
      }

      // float32 -> bfloat16
      auto tYrh = make_tensor_like<TY>(tYr_low);

#pragma unroll
      for (int i = 0; i < size(tYr_low); ++i) {
        tYrh(i) = (TY)(tYr_low(i) * scale + tYr_high(i));
      }

      using STSM_ATOM = std::conditional_t<kTileM == 8, cute::SM90_U16x4_STSM_T, cute::SM90_U16x8_STSM_T>;
      using STS_ATOM = std::conditional_t<std::is_same_v<TY, float>, UniversalCopy<uint32_t>, STSM_ATOM>;
      // Epilogue
      auto sY = make_tensor(make_smem_ptr((TY*)shm_y), SLayoutY{});  // (M, N)
      using R2SCopyAtomY = Copy_Atom<STS_ATOM, TY>;
      auto tiled_copy_y = make_tiled_copy_C(R2SCopyAtomY{}, tiled_mma);
      auto thr_copy_y = tiled_copy_y.get_slice(idx_in_warpgroup);

      auto tYr4s = thr_copy_y.retile_S(tYrh);
      auto tYs4r = thr_copy_y.partition_D(sY);

      cute::tma_store_wait<0>();
      syncwarpgroup(iwarpgroup);

      cute::copy(tiled_copy_y, tYr4s, tYs4r(_, _, _, iwarpgroup));

      if constexpr (kSplitK > 1) {
        if (is_leader_in_warpgroup) {
          if (last_tile_m != -1 && last_tile_n != -1) {
            auto* split_flag = split_flag_ptr + last_tile_m * num_tile_n + last_tile_n;
            atomicAdd(split_flag, 1);
          }
          last_tile_m = itile_m;
          last_tile_n = itile_n;
        }
      }

      syncwarpgroup(iwarpgroup);
      cute::tma_store_fence();

      if (is_leader_in_warpgroup) {
        auto gYY = tma_y.get_tma_tensor(make_shape(n, m, kSplitK));
        auto btma_y = tma_y.get_slice(0);

        auto tYs = btma_y.partition_S(sY);   // (TMA, _2, _1)
        auto tYg = btma_y.partition_D(gYY);  // (TMA, TMA_M, TMA_N)

        cute::copy(tma_y, tYs(_, 0, 0, iwarpgroup), tYg(_, kWarpGroupN * itile_n + iwarpgroup, itile_m, ichunk));
        tma_store_arrive();
      }
    }

    if constexpr (kSplitK > 1) {
      cute::tma_store_wait<0>();

      fence_async_global();
      __threadfence();
      syncwarpgroup(iwarpgroup);

      if (is_leader_in_warpgroup) {
        if (last_tile_m != -1 && last_tile_n != -1) {
          auto* split_flag = split_flag_ptr + last_tile_m * num_tile_n + last_tile_n;
          atomicAdd(split_flag, 1);
        }
      }

      bar_sync<128 * kWarpGroupN>(kWarpGroupN);

      iblock = blockIdx.x;
      __threadfence();
      using NVTout = std::conditional_t<std::is_same_v<Tout, float>, float, __nv_bfloat16>;
      while (true) {
        int itile_m, itile_n;
        reduce_flat_divider(itile_m, itile_n, iblock);

        if (itile_m >= num_tile_m) {
          break;
        }
        iblock += gridDim.x;
        auto* split_flag = split_flag_ptr + itile_m * num_tile_n + itile_n;
        while (load_global_volatile(split_flag) != kSplitK * kWarpGroupN) {
        }
        splitk_reduce<NVTout, kTileM, kTileN * kWarpGroupN, kSplitK, 128 * kWarpGroupN / 32>(
            reinterpret_cast<NVTout*>(y_ptr), splitk_y_ptr, m, n, itile_m, itile_n);
        bar_sync<128 * kWarpGroupN>(kWarpGroupN);
        // reset flag
        if (is_leader_in_warpgroup && iwarpgroup == 0) {
          *split_flag = 0;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Host launcher (hpc-ops launch_gemm_bf16xfp32_kernel)
// ---------------------------------------------------------------------------

template <typename Tin, typename Tout, int kTileM, int kTileN, int kTileK, int kStage, int kWarpGroupN, int kSplitK>
void launch_gemm_bf16xfp32_kernel(
    void* y_ptr,
    void* splitk_y_ptr,
    void* split_flag_ptr,
    const void* x_ptr,
    const void* w_high_ptr,
    const void* w_low_ptr,
    int m,
    int n,
    int k,
    float scale,
    int sm_count,
    cudaStream_t stream) {
  using namespace cute;  // NOLINT

  constexpr int kBlockSwizzle = 4;

  using TY = std::conditional_t<(kSplitK > 1), float, Tout>;

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(x_ptr)), make_shape(m, k), make_stride(k, Int<1>{}));
  auto W_HIGH =
      make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(w_high_ptr)), make_shape(n, k), make_stride(k, Int<1>{}));
  auto W_LOW =
      make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(w_low_ptr)), make_shape(n, k), make_stride(k, Int<1>{}));
  auto Y = make_tensor(
      make_gmem_ptr(reinterpret_cast<TY*>(kSplitK > 1 ? splitk_y_ptr : y_ptr)),
      make_shape(n, m, kSplitK),
      make_stride(Int<1>{}, n, n * m));

  auto slayout_x =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{}));
  auto slayout_w = tile_to_shape(
      GMMA::Layout_K_SW128_Atom<Tin>{},
      make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kWarpGroupN>{}, Int<2>{}, Int<kStage>{}));
  auto slayout_y =
      tile_to_shape(GMMA::Layout_MN_SW128_Atom<TY>{}, make_shape(Int<kTileN>{}, Int<kTileM>{}, Int<kWarpGroupN>{}));

  int shm_xw = sizeof(Tin) * (cosize(slayout_x) + cosize(slayout_w));
  int shm_y = sizeof(TY) * cosize(slayout_y);
  int shm_size = shm_xw + shm_y;

  auto tma_x = make_tma_copy(SM90_TMA_LOAD{}, X, take<0, 2>(slayout_x));
  auto tma_wh = make_tma_copy(SM90_TMA_LOAD{}, W_HIGH, take<0, 2>(slayout_w));
  auto tma_wl = make_tma_copy(SM90_TMA_LOAD{}, W_LOW, take<0, 2>(slayout_w));
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, take<0, 2>(slayout_y));

  using MMA_ATOM = std::conditional_t<
      kTileM == 64,
      SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>,
      SM90_64x16x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>>;

  auto tiled_mma = make_tiled_mma(MMA_ATOM{});

  auto kernel = gemm_bf16xfp32_kernel<
      Tin,
      TY,
      Tout,
      decltype(tiled_mma),
      decltype(tma_x),
      decltype(tma_wh),
      decltype(tma_wl),
      decltype(tma_y),
      kTileM,
      kTileN,
      kTileK,
      kStage,
      kWarpGroupN,
      decltype(slayout_x),
      decltype(slayout_w),
      decltype(slayout_y),
      kBlockSwizzle,
      kSplitK>;
  host::RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));

  int num_tile_m = (m + kTileM - 1) / kTileM;
  int num_tile_n = (n + (kTileN * kWarpGroupN) - 1) / (kTileN * kWarpGroupN) * kSplitK;
  int num_tile = num_tile_m * num_tile_n;
  int num_tile_bxn = kBlockSwizzle * num_tile_n;
  cutlass::FastDivmod swizzle_divider(num_tile_bxn);
  cutlass::FastDivmod flat_divider(num_tile_n);
  cutlass::FastDivmod reduce_flat_divider(num_tile_n / kSplitK);

  dim3 block(size(tiled_mma) * kWarpGroupN + 128);
  dim3 grid(std::min(sm_count, num_tile));

  host::LaunchKernel(grid, block, stream, shm_size)(
      kernel,
      tma_x,
      tma_wh,
      tma_wl,
      tma_y,
      reinterpret_cast<Tout*>(y_ptr),
      reinterpret_cast<float*>(splitk_y_ptr),
      reinterpret_cast<int*>(split_flag_ptr),
      m,
      n,
      k,
      scale,
      swizzle_divider,
      flat_divider,
      reduce_flat_divider);
}

}  // namespace gemm_bf16xfp32

// ---------------------------------------------------------------------------
// TVM-FFI entry
// ---------------------------------------------------------------------------

template <int kTileM, int kTileN, int kTileK, int kStage, int kWGN, int kSplitK, bool kFp32Out>
struct GemmBf16xFp32Kernel {
  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w_high,
      const tvm::ffi::TensorView w_low,
      const tvm::ffi::TensorView y,
      const tvm::ffi::Optional<tvm::ffi::TensorView> split_y,
      const tvm::ffi::Optional<tvm::ffi::TensorView> split_flag,
      double scale) {
    using namespace host;

    using bf16 = cute::bfloat16_t;
    using OutT = std::conditional_t<kFp32Out, fp32_t, bf16_t>;

    auto M = SymbolicSize{"m"};
    auto K = SymbolicSize{"k"};
    auto N = SymbolicSize{"n"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(device).verify(w_high);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(device).verify(w_low);
    TensorMatcher({M, N}).with_dtype<OutT>().with_device(device).verify(y);

    const int m = static_cast<int>(M.unwrap());
    const int n = static_cast<int>(N.unwrap());
    const int k = static_cast<int>(K.unwrap());

    RuntimeCheck(n % 64 == 0, "gemm_bf16xfp32: n must be divisible by 64, got ", n);

    void* split_y_ptr = nullptr;
    void* split_flag_ptr = nullptr;
    if constexpr (kSplitK > 1) {
      RuntimeCheck(split_y.has_value(), "gemm_bf16xfp32: split_y workspace is required when split_k > 1");
      RuntimeCheck(split_flag.has_value(), "gemm_bf16xfp32: split_flag workspace is required when split_k > 1");
      TensorMatcher({kSplitK, M, N}).with_dtype<fp32_t>().with_device(device).verify(split_y.value());
      const int num_tile_m = (m + kTileM - 1) / kTileM;
      const int num_tile_n = (n + kTileN * kWGN - 1) / (kTileN * kWGN);
      TensorMatcher({num_tile_m, num_tile_n}).with_dtype<int32_t>().with_device(device).verify(split_flag.value());
      split_y_ptr = split_y.value().data_ptr();
      split_flag_ptr = split_flag.value().data_ptr();
    }

    const DLDevice dev = device.unwrap();
    auto stream = LaunchKernel::resolve_device(dev);
    const int sm_count = static_cast<int>(runtime::get_sm_count(dev.device_id));

    gemm_bf16xfp32::launch_gemm_bf16xfp32_kernel<
        bf16,
        std::conditional_t<kFp32Out, float, bf16>,
        kTileM,
        kTileN,
        kTileK,
        kStage,
        kWGN,
        kSplitK>(
        y.data_ptr(),
        split_y_ptr,
        split_flag_ptr,
        x.data_ptr(),
        w_high.data_ptr(),
        w_low.data_ptr(),
        m,
        n,
        k,
        static_cast<float>(scale),
        sm_count,
        stream);
  }
};

}  // namespace
