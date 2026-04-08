// Copyright 2025-2026 Ant Group Co., Ltd.
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

/*
 * Copyright (c) 2025 DeepSeek. All Rights Reserved.
 *
 * Licensed under the MIT License.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <cute/tensor.hpp>

#include "kerutils/device/common.h"

// Adapted from
// https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh
namespace kerutils {

// ============================================================
// TMA gather4 intrinsics (SM100)
// ============================================================

// tma gather4
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
// Please pay attention that the coordinates of TMA gather4 are int32, which may lead to overflow under some scenarios
CUTE_DEVICE
void tma_gather4(
    const void* desc_ptr, transac_bar_t& mbar_ptr, void* smem_ptr, int col_idx, int4 row_idxs, int64_t cache_hint) {
  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
  uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar_ptr);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_"
      "hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
      :
      : "r"(smem_addr),
        "l"(desc_ptr),
        "r"(col_idx),
        "r"(row_idxs.x),
        "r"(row_idxs.y),
        "r"(row_idxs.z),
        "r"(row_idxs.w),
        "r"(mbar_addr),
        "l"(cache_hint)
      : "memory");
}

// tma gather4 prefetch
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor)
// Please pay attention that the coordinates of TMA gather4 are int32, which may lead to overflow under some scenarios
CUTE_DEVICE
void tma_gather4_prefetch(const void* desc_ptr, int col_idx, int4 row_idxs, int64_t cache_hint) {
  asm volatile(
      "cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint [%0, {%1, %2, %3, %4, %5}], %6;\n"
      :
      : "l"(desc_ptr),
        "r"(col_idx),
        "r"(row_idxs.x),
        "r"(row_idxs.y),
        "r"(row_idxs.z),
        "r"(row_idxs.w),
        "l"(cache_hint));
}

// tma gather4 with cta_group::2, allowing for synchronization across CTAs within a pair of CTAs
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
template <bool USE_CTA0_MBAR = false>
CUTE_DEVICE void tma_gather4_cta_group_2(
    const void* desc_ptr, transac_bar_t& mbar_ptr, void* smem_ptr, int col_idx, int4 row_idxs, int64_t cache_hint) {
  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
  uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar_ptr);
  if constexpr (USE_CTA0_MBAR) {
    mbar_addr &= cute::Sm100MmaPeerBitMask;
  }
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_"
      "hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
      :
      : "r"(smem_addr),
        "l"(desc_ptr),
        "r"(col_idx),
        "r"(row_idxs.x),
        "r"(row_idxs.y),
        "r"(row_idxs.z),
        "r"(row_idxs.w),
        "r"(mbar_addr),
        "l"(cache_hint)
      : "memory");
}

// ============================================================
// Vectorized float2 arithmetic
// ============================================================

// Vectorized addition for float32
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add)
CUTE_DEVICE
float2 float2_add(const float2& a, const float2& b) {
  float2 c;
  asm volatile("add.f32x2 %0, %1, %2;\n"
               : "=l"(reinterpret_cast<uint64_t&>(c))
               : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
  return c;
}

// Vectorized subtraction for float32
CUTE_DEVICE
float2 float2_sub(const float2& a, const float2& b) {
  float2 c;
  asm volatile("sub.f32x2 %0, %1, %2;\n"
               : "=l"(reinterpret_cast<uint64_t&>(c))
               : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
  return c;
}

// Vectorized multiplication for float32
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul)
CUTE_DEVICE
float2 float2_mul(const float2& a, const float2& b) {
  float2 c;
  asm volatile("mul.f32x2 %0, %1, %2;\n"
               : "=l"(reinterpret_cast<uint64_t&>(c))
               : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
  return c;
}

// Vectorized fused multiply-add for float32
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma)
CUTE_DEVICE
float2 float2_fma(const float2& a, const float2& b, const float2& c) {
  // return a*b+c
  float2 d;
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
               : "=l"(reinterpret_cast<uint64_t&>(d))
               : "l"(reinterpret_cast<uint64_t const&>(a)),
                 "l"(reinterpret_cast<uint64_t const&>(b)),
                 "l"(reinterpret_cast<uint64_t const&>(c)));
  return d;
}

// Vectorized negation for float32
CUTE_DEVICE
float2 float2_neg(const float2& a) {
  float2 t = {-1.0f, -1.0f};
  return float2_mul(a, t);
}

// ============================================================
// tcgen05 fence intrinsics (SM100)
// ============================================================

// tcgen05.fence::before_thread_sync
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-special-sync-operations-fence)
__device__ __forceinline__ void tcgen05_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;");
}

// tcgen05.fence::after_thread_sync
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-special-sync-operations-fence)
__device__ __forceinline__ void tcgen05_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}

// ============================================================
// Tensor memory (TMEM) load/store intrinsics (SM100)
// ============================================================

// Load from tensor memory, 32 data path lanes, 32-bit pattern, repeated N times.
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld)
template <int kNumElements>
__device__ __forceinline__ void tmem_ld_32dp32bNx(uint32_t tmem_start, void* data_) {
  uint32_t* data = (uint32_t*)data_;
  static_assert(
      kNumElements == 1 || kNumElements == 2 || kNumElements == 4 || kNumElements == 8 || kNumElements == 16 ||
          kNumElements == 32 || kNumElements == 64 || kNumElements == 128,
      "Invalid kNumElements");
  // NOTE The following code crashes VSCode intellisense engine, so we disable it
#ifndef __VSCODE_IDE__
  [&]<size_t... Is>(cute::index_sequence<Is...>) {
    if constexpr (kNumElements == 1) {
      cute::SM100_TMEM_LOAD_32dp32b1x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 2) {
      cute::SM100_TMEM_LOAD_32dp32b2x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 4) {
      cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 8) {
      cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 16) {
      cute::SM100_TMEM_LOAD_32dp32b16x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 32) {
      cute::SM100_TMEM_LOAD_32dp32b32x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 64) {
      cute::SM100_TMEM_LOAD_32dp32b64x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 128) {
      cute::SM100_TMEM_LOAD_32dp32b128x::copy(tmem_start, data[Is]...);
    }
  }(cute::make_index_sequence<kNumElements>{});
#endif
}

// Store into tensor memory, 32 data path lanes, 32-bit pattern, repeated N times.
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st)
template <int kNumElements>
__device__ __forceinline__ void tmem_st_32dp32bNx(uint32_t tmem_start, void const* data_) {
  uint32_t const* data = (uint32_t const*)data_;
  static_assert(
      kNumElements == 1 || kNumElements == 2 || kNumElements == 4 || kNumElements == 8 || kNumElements == 16 ||
          kNumElements == 32 || kNumElements == 64 || kNumElements == 128,
      "Invalid kNumElements");
#ifndef __VSCODE_IDE__
  [&]<size_t... Is>(cute::index_sequence<Is...>) {
    if constexpr (kNumElements == 1) {
      cute::SM100_TMEM_STORE_32dp32b1x::copy(data[Is]..., tmem_start);
    } else if constexpr (kNumElements == 2) {
      cute::SM100_TMEM_STORE_32dp32b2x::copy(data[Is]..., tmem_start);
    } else if constexpr (kNumElements == 4) {
      cute::SM100_TMEM_STORE_32dp32b4x::copy(data[Is]..., tmem_start);
    } else if constexpr (kNumElements == 8) {
      cute::SM100_TMEM_STORE_32dp32b8x::copy(data[Is]..., tmem_start);
    } else if constexpr (kNumElements == 16) {
      cute::SM100_TMEM_STORE_32dp32b16x::copy(data[Is]..., tmem_start);
    } else if constexpr (kNumElements == 32) {
      cute::SM100_TMEM_STORE_32dp32b32x::copy(data[Is]..., tmem_start);
    } else if constexpr (kNumElements == 64) {
      cute::SM100_TMEM_STORE_32dp32b64x::copy(data[Is]..., tmem_start);
    } else if constexpr (kNumElements == 128) {
      cute::SM100_TMEM_STORE_32dp32b128x::copy(data[Is]..., tmem_start);
    }
  }(cute::make_index_sequence<kNumElements>{});
#endif
}

}  // namespace kerutils
