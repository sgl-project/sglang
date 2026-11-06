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

// SM100 PTX helpers; TMA gather4 / TMEM wrappers follow DeepSeek FlashMLA kerutils.
#pragma once

#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>

namespace sglang::q8kv8_sm100 {

using bf16 = cutlass::bfloat16_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;

// TMA
// Gather coordinates are int32; negative or >= s_kv rows are zero-filled.
__device__ __forceinline__ void
tma_gather4(const void* desc_ptr, transac_bar_t& bar, void* smem_ptr, int col, int4 rows, int64_t cache_hint) {
  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
  uint32_t bar_addr = cute::cast_smem_ptr_to_uint(&bar);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1"
      ".L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
      :
      : "r"(smem_addr),
        "l"(desc_ptr),
        "r"(col),
        "r"(rows.x),
        "r"(rows.y),
        "r"(rows.z),
        "r"(rows.w),
        "r"(bar_addr),
        "l"(cache_hint)
      : "memory");
}

// UMMA descriptors
// Swizzled K-major: LBO is ignored, SBO is the 8-row group stride.
// Swizzled MN-major: LBO strides MN atoms, SBO strides 8-deep K atoms.
__device__ __forceinline__ uint64_t
make_smem_desc(const void* smem_ptr, cute::UMMA::LayoutType layout, uint32_t lbo_bytes, uint32_t sbo_bytes) {
  cute::UMMA::SmemDescriptor desc;
  desc.version_ = 1;
  desc.lbo_mode_ = 0;
  desc.base_offset_ = 0;
  desc.layout_type_ = uint8_t(layout);
  desc.start_address_ = static_cast<uint16_t>(cute::cast_smem_ptr_to_uint(smem_ptr) >> 4);
  desc.leading_byte_offset_ = static_cast<uint16_t>(lbo_bytes >> 4);
  desc.stride_byte_offset_ = static_cast<uint16_t>(sbo_bytes >> 4);
  return desc.desc_;
}

// D=F32; format 0 is E4M3 (kind::f8f6f4).
constexpr uint32_t make_idesc(uint32_t a_fmt, uint32_t b_fmt, bool b_mn_major, int m, int n) {
  return (1u << 4) | (a_fmt << 7) | (b_fmt << 10) | ((b_mn_major ? 1u : 0u) << 16) | ((uint32_t(n) >> 3) << 17) |
         ((uint32_t(m) >> 4) << 24);
}

// tcgen05
// .ws M=64 splits N across lane halves: lanes [0,64) hold n < N/2, [64,128) hold n >= N/2.
__device__ __forceinline__ void
mma_ws_ts_f8(uint32_t tmem_d, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p, 0;\n\t}\n"
      :
      : "r"(tmem_d), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accumulate));
}

__device__ __forceinline__ void
mma_ws_ss_f8(uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p, 0;\n\t}\n"
      :
      : "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accumulate));
}

// 64 smem rows x 16B; rows 0-31 land in lanes 0-31 and 64-95, rows 32-63 in 32-63 and 96-127.
__device__ __forceinline__ void utccp_64x128b_warpx2_0213(uint32_t tmem_dst, uint64_t desc_src) {
  asm volatile("tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;" : : "r"(tmem_dst), "l"(desc_src));
}

// 32B global store (STG.256); `src` holds 8 packed 32-bit words.
__device__ __forceinline__ void stg_256(void* gptr, const uint32_t* src) {
  asm volatile("st.global.L1::no_allocate.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n"
               :
               : "l"(gptr),
                 "r"(src[0]),
                 "r"(src[1]),
                 "r"(src[2]),
                 "r"(src[3]),
                 "r"(src[4]),
                 "r"(src[5]),
                 "r"(src[6]),
                 "r"(src[7])
               : "memory");
}

__device__ __forceinline__ void umma_commit(transac_bar_t& bar) {
  uint32_t bar_addr = cute::cast_smem_ptr_to_uint(&bar);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" : : "r"(bar_addr));
}

__device__ __forceinline__ void tcgen05_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// 32 lanes x 32 columns of 32-bit; the lane field must name the warp's own quadrant.
__device__ __forceinline__ void tmem_ld_32x32b_x32(uint32_t tmem_addr, float (&v)[32]) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  cute::SM100_TMEM_LOAD_32dp32b32x::copy(
      tmem_addr,
      r[0],
      r[1],
      r[2],
      r[3],
      r[4],
      r[5],
      r[6],
      r[7],
      r[8],
      r[9],
      r[10],
      r[11],
      r[12],
      r[13],
      r[14],
      r[15],
      r[16],
      r[17],
      r[18],
      r[19],
      r[20],
      r[21],
      r[22],
      r[23],
      r[24],
      r[25],
      r[26],
      r[27],
      r[28],
      r[29],
      r[30],
      r[31]);
}

__device__ __forceinline__ void tmem_st_32x32b_x32(uint32_t tmem_addr, const float (&v)[32]) {
  const uint32_t* r = reinterpret_cast<const uint32_t*>(v);
  cute::SM100_TMEM_STORE_32dp32b32x::copy(
      r[0],
      r[1],
      r[2],
      r[3],
      r[4],
      r[5],
      r[6],
      r[7],
      r[8],
      r[9],
      r[10],
      r[11],
      r[12],
      r[13],
      r[14],
      r[15],
      r[16],
      r[17],
      r[18],
      r[19],
      r[20],
      r[21],
      r[22],
      r[23],
      r[24],
      r[25],
      r[26],
      r[27],
      r[28],
      r[29],
      r[30],
      r[31],
      tmem_addr);
}

__device__ __forceinline__ void tmem_wait_ld() {
  cutlass::arch::fence_view_async_tmem_load();
}

__device__ __forceinline__ void tmem_wait_st() {
  cutlass::arch::fence_view_async_tmem_store();
}

// packed FP32x2 math
__device__ __forceinline__ float2 float2_fma(const float2& a, const float2& b, const float2& c) {
  float2 d;
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
               : "=l"(reinterpret_cast<uint64_t&>(d))
               : "l"(reinterpret_cast<uint64_t const&>(a)),
                 "l"(reinterpret_cast<uint64_t const&>(b)),
                 "l"(reinterpret_cast<uint64_t const&>(c)));
  return d;
}

__device__ __forceinline__ float2 float2_add(const float2& a, const float2& b) {
  float2 c;
  asm volatile("add.f32x2 %0, %1, %2;\n"
               : "=l"(reinterpret_cast<uint64_t&>(c))
               : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
  return c;
}

__device__ __forceinline__ float2 float2_mul(const float2& a, const float2& b) {
  float2 c;
  asm volatile("mul.f32x2 %0, %1, %2;\n"
               : "=l"(reinterpret_cast<uint64_t&>(c))
               : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
  return c;
}

}  // namespace sglang::q8kv8_sm100
