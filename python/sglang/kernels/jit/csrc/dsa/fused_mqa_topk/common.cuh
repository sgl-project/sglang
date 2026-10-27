// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek
// Device helpers shared by the persistent kernel, its gate sample, the
// selection and the exact rescan. The TMA loads are DeepGEMM's tma_copy for
// unswizzled boxes.
#pragma once

#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <deep_gemm/common/ring_pipeline.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

#include "constants.h"
#include <cstdint>
#include <cuda_fp16.h>

namespace sglang::fused_mqa_topk {

namespace ptx = ::deep_gemm::ptx;
using ::deep_gemm::RingPipeline;
using Barrier = cutlass::arch::ClusterTransactionBarrier;

// Mask of all 32 lanes for the warp-synchronous intrinsics.
inline constexpr uint32_t kFullMask = 0xffffffffu;

// Warp roles of both pipelines (persistent kernel and dense rescan): warps 0-7
// are the math warps (two warpgroups of kWgRows query rows each), warps 8-11
// the producer / UMMA / drain warps.
inline constexpr uint32_t kMathThreads = 256, kMathWgThreads = 128, kMathWarps = kMathThreads / 32;
inline constexpr uint32_t kWgRows = kBlockRows / 2;
// Register split: ptxas gives each of the 384 threads 168 registers;
// setmaxnreg moves them from the producer / UMMA / drain warpgroup to the two
// math warpgroups.
inline constexpr uint32_t kMathRegs = 224, kSpecRegs = 168 * 3 - 2 * kMathRegs;

// Named barrier ids (cutlass::arch::NamedBarrier offsets them past the ids it
// reserves). kBarDenseMath: the dense rescan's math threads; kBarMath /
// kBarSelect: the math threads; kBarDrain: the math threads and the two drain
// warps (kDrainThreads); kBarWarpgroup + g: math warpgroup g.
inline constexpr uint32_t kBarDenseMath = 0, kBarMath = 2, kBarDrain = 4, kBarSelect = 5, kBarWarpgroup = 6;
inline constexpr uint32_t kDrainThreads = kMathThreads + 2 * 32;

// Barrier of the math threads; the compiler barriers keep shared-memory
// accesses on their side of it.
__device__ __forceinline__ void math_sync() {
  asm volatile("" ::: "memory");
  cutlass::arch::NamedBarrier(kMathThreads, kBarMath).sync();
  asm volatile("" ::: "memory");
}

// Inclusive prefix sum of v over the warp's lanes.
__device__ __forceinline__ uint32_t warp_inclusive_scan(uint32_t v, uint32_t lane) {
#pragma unroll
  for (uint32_t d = 1; d < 32; d *= 2) {
    const uint32_t u = __shfl_up_sync(kFullMask, v, d);
    if (lane >= d) v += u;
  }
  return v;
}

// Shared-memory histograms pad one word per 32 (bank conflicts): word w lives
// at padded_bin(w), a histogram of n words takes padded_bins(n).
__host__ __device__ constexpr uint32_t padded_bins(uint32_t n) {
  return n + n / 32;
}
__device__ __forceinline__ uint32_t padded_bin(uint32_t w) {
  return w + w / 32;
}

// Reads `x` with a volatile load, so the compiler re-reads it from (shared)
// memory instead of keeping it live in a register.
template <class T>
__device__ __forceinline__ T volatile_load(const T& x) {
  return *reinterpret_cast<const volatile T*>(&x);
}

// Order-preserving uint32 code of a float (a higher code is a larger value)
// and its inverse.
__device__ __forceinline__ uint32_t ordered_code(float x) {
  const uint32_t u = __float_as_uint(x);
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}
__device__ __forceinline__ float ordered_float(uint32_t code) {
  return __uint_as_float((code & 0x80000000u) ? (code ^ 0x80000000u) : ~code);
}

// Histogram bin of a score: its fp16 value (rounded down) as an
// order-preserving 16-bit code, >> kShift; half_bin_floor is the smallest value
// of bin `b`.
template <uint32_t kShift>
__device__ __forceinline__ uint32_t half_bin(float x) {
  const uint32_t u = __half_as_ushort(__float2half_rd(x));
  return ((u & 0x8000u) ? ((~u) & 65535u) : (u ^ 0x8000u)) >> kShift;
}
template <uint32_t kShift>
__device__ __forceinline__ float half_bin_floor(uint32_t b) {
  const uint32_t o = b << kShift, u = (o & 0x8000u) ? (o ^ 0x8000u) : ((~o) & 65535u);
  return __half2float(__ushort_as_half(u));
}

// One TMA load of the box at (x, y) (3D: at (x, y, z)) into `smem`, completing
// on `barrier`.
template <typename T>
CUTLASS_DEVICE void tma_load(const void* desc, Barrier* barrier, T* smem, uint32_t x, uint32_t y) {
  constexpr auto kCacheHint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
  cute::SM90_TMA_LOAD_2D::copy(desc, reinterpret_cast<uint64_t*>(barrier), kCacheHint, smem, x, y);
}
template <typename T>
CUTLASS_DEVICE void tma_load(const void* desc, Barrier* barrier, T* smem, uint32_t x, uint32_t y, uint32_t z) {
  constexpr auto kCacheHint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
  cute::SM90_TMA_LOAD_3D::copy(desc, reinterpret_cast<uint64_t*>(barrier), kCacheHint, smem, x, y, z);
}

}  // namespace sglang::fused_mqa_topk
