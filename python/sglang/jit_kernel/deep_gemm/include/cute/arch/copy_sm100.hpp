/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cute/arch/config.hpp>

#include <cute/arch/mma_sm100.hpp>
#include <cute/arch/copy.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cute {

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Global Memory Load and Store PTX definitions
//
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM100_LOAD_256bit_CACHE_NOALLOCATION
{
  using SRegisters = uint256_t[1];
  using DRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void
  copy(uint256_t const& gmem_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3,
       uint32_t& dst4, uint32_t& dst5, uint32_t& dst6, uint32_t& dst7)
  {
    #if defined(CUTE_ARCH_LOAD256_SM100A_ENABLED)
      asm volatile("ld.global.L1::no_allocate.v8.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
              : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3), "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7)
              : "l"(&gmem_addr) );
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use LOAD.256 without CUTE_ARCH_LOAD256_SM100A_ENABLED.");
    #endif
  }
};

struct SM100_STORE_256bit_CACHE_NOALLOCATION
{
  using SRegisters = uint32_t[8];
  using DRegisters = uint256_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& src4, uint32_t const& src5, uint32_t const& src6, uint32_t const& src7,
       uint256_t& gmem_addr)
  {
    #if defined(CUTE_ARCH_STORE256_SM100A_ENABLED)
      asm volatile("st.global.L1::no_allocate.v8.f32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n"
              :: "l"(&gmem_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3), "r"(src4), "r"(src5), "r"(src6), "r"(src7));
    #else
      CUTE_INVALID_CONTROL_PATH("Trying to use stg.256 without CUTE_ARCH_STORE256_SM100A_ENABLED.");
    #endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// LDSM PTX definitions
//
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM100_U8x8_LDSM_T
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1)
  {
#if defined(CUTE_ARCH_LDSM_SM100A_ENABLED)
    uint32_t tmp0, tmp1;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0, %1}, [%2];\n"
        : "=r"(reinterpret_cast<uint32_t &>(tmp0)), "=r"(reinterpret_cast<uint32_t &>(tmp1))
        :  "r"(smem_int_ptr));
    // RefLayout of ldmatrix.m16n16.x1.trans won't match stmatrix.m16n8.x2.trans without additional transformations
    // Do this here so we don't need to add an additional reg to reg copy at the collective layer
    uchar4& tmp0_ = reinterpret_cast<uchar4&>(tmp0);
    uchar4& tmp1_ = reinterpret_cast<uchar4&>(tmp1);
    uchar4 dst0_{tmp0_.x, tmp0_.y, tmp1_.x, tmp1_.y};
    uchar4 dst1_{tmp0_.z, tmp0_.w, tmp1_.z, tmp1_.w};
    dst0 = reinterpret_cast<uint32_t&>(dst0_);
    dst1 = reinterpret_cast<uint32_t&>(dst1_);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM100A_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM100_U8x16_LDSM_T
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_LDSM_SM100A_ENABLED)
    uint32_t tmp0, tmp1, tmp2, tmp3;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(reinterpret_cast<uint32_t &>(tmp0)), "=r"(reinterpret_cast<uint32_t &>(tmp1)),
          "=r"(reinterpret_cast<uint32_t &>(tmp2)), "=r"(reinterpret_cast<uint32_t &>(tmp3))
        :  "r"(smem_int_ptr));
    uchar4& tmp0_ = reinterpret_cast<uchar4&>(tmp0);
    uchar4& tmp1_ = reinterpret_cast<uchar4&>(tmp1);
    uchar4& tmp2_ = reinterpret_cast<uchar4&>(tmp2);
    uchar4& tmp3_ = reinterpret_cast<uchar4&>(tmp3);
    uchar4 dst0_{tmp0_.x, tmp0_.y, tmp1_.x, tmp1_.y};
    uchar4 dst1_{tmp0_.z, tmp0_.w, tmp1_.z, tmp1_.w};
    uchar4 dst2_{tmp2_.x, tmp2_.y, tmp3_.x, tmp3_.y};
    uchar4 dst3_{tmp2_.z, tmp2_.w, tmp3_.z, tmp3_.w};
    dst0 = reinterpret_cast<uint32_t&>(dst0_);
    dst1 = reinterpret_cast<uint32_t&>(dst1_);
    dst2 = reinterpret_cast<uint32_t&>(dst2_);
    dst3 = reinterpret_cast<uint32_t&>(dst3_);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM100A_ENABLED.");
#endif
  }
};

struct SM100_SU4_DU8x16_x1_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[1];

  CUTE_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0)
  {
#if defined(CUTE_ARCH_LDSM_SM100A_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64  {%0}, [%1];\n"
        : "=r"(dst0)
        :  "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM100A_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM100_SU6_DU8x16_x1_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[1];

  CUTE_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0)
  {
#if defined(CUTE_ARCH_LDSM_SM100A_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32  {%0}, [%1];\n"
        : "=r"(dst0)
        :  "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM100A_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM100_SU4_DU8x16_x2_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[2];

  CUTE_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1)
  {
#if defined(CUTE_ARCH_LDSM_SM100A_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64  {%0, %1}, [%2];\n"
        : "=r"(dst0), "=r"(dst1)
        :  "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM100A_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM100_SU6_DU8x16_x2_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[2];

  CUTE_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1)
  {
#if defined(CUTE_ARCH_LDSM_SM100A_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b6x16_p32  {%0, %1}, [%2];\n"
        : "=r"(dst0), "=r"(dst1)
        :  "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM100A_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM100_SU4_DU8x16_x4_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_LDSM_SM100A_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64  {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM100A_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM100_SU6_DU8x16_x4_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_LDSM_SM100A_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b6x16_p32  {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM100A_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// STSM PTX definitions
//
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM100_U8x4_STSM_T
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint128_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src,
       uint128_t& smem_dst)
  {
#if defined(CUTE_ARCH_STSM_SM100A_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    asm volatile ("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};\n"
        :: "r"(smem_int_ptr),
           "r"(src));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use stmatrix without CUTE_ARCH_STSM_SM100A_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM100_U8x8_STSM_T
{
  using SRegisters = uint32_t[2];
  using DRegisters = uint128_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1,
       uint128_t& smem_dst)
  {
#if defined(CUTE_ARCH_STSM_SM100A_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    asm volatile ("stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1, %2};\n"
        :: "r"(smem_int_ptr),
           "r"(src0), "r"(src1));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use stmatrix without CUTE_ARCH_STSM_SM100A_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM100_U8x16_STSM_T
{
  using SRegisters = uint32_t[4];
  using DRegisters = uint128_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint128_t& smem_dst)
  {
#if defined(CUTE_ARCH_STSM_SM100A_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    asm volatile ("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
        :: "r"(smem_int_ptr),
           "r"(src0), "r"(src1), "r"(src2), "r"(src3));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use stmatrix without CUTE_ARCH_STSM_SM100A_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// UTCCP PTX definitions
//
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace SM100::TMEM::UTCCP {

// 128 data path lanes, 256-bit pattern, 1cta mode
struct SM100_UTCCP_128dp256bit_1cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::1.128x256b [%0], %1;"
    :
    : "r"(dst_addr)  "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

// 128 data path lanes, 256-bit pattern, 2cta mode
struct SM100_UTCCP_128dp256bit_2cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::2.128x256b [%0], %1;"
    :
    : "r"(dst_addr)  "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

struct SM100_UTCCP_128dp128bit_1cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::1.128x128b [%0], %1;"
    :
    : "r"(dst_addr)  "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

struct SM100_UTCCP_128dp128bit_2cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::2.128x128b [%0], %1;"
    :
    : "r"(dst_addr)  "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};


// 4 data path lanes, 256-bit pattern, 1cta mode
struct SM100_UTCCP_4dp256bit_1cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::1.4x256b [%0], %1;"
    :
    : "r"(dst_addr)  "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

// 4 data path lanes, 256-bit pattern, 2cta mode
struct SM100_UTCCP_4dp256bit_2cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
        asm volatile ("tcgen05.cp.cta_group::2.4x256b [%0], %1;"
    :
    : "r"(dst_addr)  "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

// 4x32 data path lanes (broadcast), 128-bit pattern, 1cta mode
struct SM100_UTCCP_4x32dp128bit_1cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
    :
    : "r"(dst_addr)  "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

// 4x32 data path lanes (broadcast), 128-bit pattern, 2cta mode
struct SM100_UTCCP_4x32dp128bit_2cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
    :
    : "r"(dst_addr)  "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

// 2x64 data path lanes (broadcast like 4x32dp), 128-bit pattern, 1cta mode
struct SM100_UTCCP_2x64dp128bitlw0213_1cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::1.64x128b.warpx2::02_13  [%0], %1;"
    :
    : "r"(dst_addr)  "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

// 2x64 data path lanes (broadcast like 4x32dp), 128-bit pattern, 2cta mode
struct SM100_UTCCP_2x64dp128bitlw0213_2cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::2.64x128b.warpx2::02_13  [%0], %1;"
    :
    : "r"(dst_addr)  "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

// 2x64 data path lanes (broadcast seperately in upper and lower 64dp), 128-bit pattern, 1cta mode
// data_row[0:31] -> DP[0:63]
// data_row[32:63] -> DP[64:127]
struct SM100_UTCCP_2x64dp128bitlw0123_1cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::1.64x128b.warpx2::01_23 [%0], %1;"
    :
    : "r"(dst_addr)  "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

// 2x64 data path lanes (broadcast seperately in upper and lower 64dp), 128-bit pattern, 2cta mode
// data_row[0:31] -> DP[0:63]
// data_row[32:63] -> DP[64:127]
struct SM100_UTCCP_2x64dp128bitlw0123_2cta
{
  using SRegisters = uint64_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& src_addr, uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;"
    :
    : "r"(dst_addr)  "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

} // end namespace SM100::TMEM::UTCCP

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace SM100::TMEM::LOAD {

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMEM LOAD PTX definitions
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 1 times
struct SM100_TMEM_LOAD_16dp256b1x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    "{%0, %1, %2, %3},"
                    "[%4];\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 1 times, packed 16b read
struct SM100_TMEM_LOAD_16dp256b1x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x256b.x1.pack::16b.b32"
                    "{%0, %1, %2, %3},"
                    "[%4];\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 2 times
struct SM100_TMEM_LOAD_16dp256b2x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3,
       uint32_t& dst4, uint32_t& dst5, uint32_t& dst6, uint32_t& dst7)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x256b.x2.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7},"
                    "[%8];\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3),
       "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 2 times, packed 16b read
struct SM100_TMEM_LOAD_16dp256b2x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3,
       uint32_t& dst4, uint32_t& dst5, uint32_t& dst6, uint32_t& dst7)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x256b.x2.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7},"
                    "[%8];\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3),
       "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 4 times
struct SM100_TMEM_LOAD_16dp256b4x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15},"
                    "[%16];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 4 times, packed 16b read
struct SM100_TMEM_LOAD_16dp256b4x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x256b.x4.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15},"
                    "[%16];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 8 times
struct SM100_TMEM_LOAD_16dp256b8x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31},"
                    "[%32];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 8 times, packed 16b read
struct SM100_TMEM_LOAD_16dp256b8x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x256b.x8.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31},"
                    "[%32];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 16 times
struct SM100_TMEM_LOAD_16dp256b16x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31,
       uint32_t& dst32, uint32_t& dst33, uint32_t& dst34, uint32_t& dst35,
       uint32_t& dst36, uint32_t& dst37, uint32_t& dst38, uint32_t& dst39,
       uint32_t& dst40, uint32_t& dst41, uint32_t& dst42, uint32_t& dst43,
       uint32_t& dst44, uint32_t& dst45, uint32_t& dst46, uint32_t& dst47,
       uint32_t& dst48, uint32_t& dst49, uint32_t& dst50, uint32_t& dst51,
       uint32_t& dst52, uint32_t& dst53, uint32_t& dst54, uint32_t& dst55,
       uint32_t& dst56, uint32_t& dst57, uint32_t& dst58, uint32_t& dst59,
       uint32_t& dst60, uint32_t& dst61, uint32_t& dst62, uint32_t& dst63)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x256b.x16.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63},"
                    "[%64];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31),
       "=r"(dst32), "=r"(dst33), "=r"(dst34), "=r"(dst35),
       "=r"(dst36), "=r"(dst37), "=r"(dst38), "=r"(dst39),
       "=r"(dst40), "=r"(dst41), "=r"(dst42), "=r"(dst43),
       "=r"(dst44), "=r"(dst45), "=r"(dst46), "=r"(dst47),
       "=r"(dst48), "=r"(dst49), "=r"(dst50), "=r"(dst51),
       "=r"(dst52), "=r"(dst53), "=r"(dst54), "=r"(dst55),
       "=r"(dst56), "=r"(dst57), "=r"(dst58), "=r"(dst59),
       "=r"(dst60), "=r"(dst61), "=r"(dst62), "=r"(dst63)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 16 times, packed 16b read
struct SM100_TMEM_LOAD_16dp256b16x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31,
       uint32_t& dst32, uint32_t& dst33, uint32_t& dst34, uint32_t& dst35,
       uint32_t& dst36, uint32_t& dst37, uint32_t& dst38, uint32_t& dst39,
       uint32_t& dst40, uint32_t& dst41, uint32_t& dst42, uint32_t& dst43,
       uint32_t& dst44, uint32_t& dst45, uint32_t& dst46, uint32_t& dst47,
       uint32_t& dst48, uint32_t& dst49, uint32_t& dst50, uint32_t& dst51,
       uint32_t& dst52, uint32_t& dst53, uint32_t& dst54, uint32_t& dst55,
       uint32_t& dst56, uint32_t& dst57, uint32_t& dst58, uint32_t& dst59,
       uint32_t& dst60, uint32_t& dst61, uint32_t& dst62, uint32_t& dst63)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x256b.x16.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63},"
                    "[%64];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31),
       "=r"(dst32), "=r"(dst33), "=r"(dst34), "=r"(dst35),
       "=r"(dst36), "=r"(dst37), "=r"(dst38), "=r"(dst39),
       "=r"(dst40), "=r"(dst41), "=r"(dst42), "=r"(dst43),
       "=r"(dst44), "=r"(dst45), "=r"(dst46), "=r"(dst47),
       "=r"(dst48), "=r"(dst49), "=r"(dst50), "=r"(dst51),
       "=r"(dst52), "=r"(dst53), "=r"(dst54), "=r"(dst55),
       "=r"(dst56), "=r"(dst57), "=r"(dst58), "=r"(dst59),
       "=r"(dst60), "=r"(dst61), "=r"(dst62), "=r"(dst63)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 32 times
struct SM100_TMEM_LOAD_16dp256b32x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst000, uint32_t& dst001, uint32_t& dst002, uint32_t& dst003,
       uint32_t& dst004, uint32_t& dst005, uint32_t& dst006, uint32_t& dst007,
       uint32_t& dst008, uint32_t& dst009, uint32_t& dst010, uint32_t& dst011,
       uint32_t& dst012, uint32_t& dst013, uint32_t& dst014, uint32_t& dst015,
       uint32_t& dst016, uint32_t& dst017, uint32_t& dst018, uint32_t& dst019,
       uint32_t& dst020, uint32_t& dst021, uint32_t& dst022, uint32_t& dst023,
       uint32_t& dst024, uint32_t& dst025, uint32_t& dst026, uint32_t& dst027,
       uint32_t& dst028, uint32_t& dst029, uint32_t& dst030, uint32_t& dst031,
       uint32_t& dst032, uint32_t& dst033, uint32_t& dst034, uint32_t& dst035,
       uint32_t& dst036, uint32_t& dst037, uint32_t& dst038, uint32_t& dst039,
       uint32_t& dst040, uint32_t& dst041, uint32_t& dst042, uint32_t& dst043,
       uint32_t& dst044, uint32_t& dst045, uint32_t& dst046, uint32_t& dst047,
       uint32_t& dst048, uint32_t& dst049, uint32_t& dst050, uint32_t& dst051,
       uint32_t& dst052, uint32_t& dst053, uint32_t& dst054, uint32_t& dst055,
       uint32_t& dst056, uint32_t& dst057, uint32_t& dst058, uint32_t& dst059,
       uint32_t& dst060, uint32_t& dst061, uint32_t& dst062, uint32_t& dst063,
       uint32_t& dst064, uint32_t& dst065, uint32_t& dst066, uint32_t& dst067,
       uint32_t& dst068, uint32_t& dst069, uint32_t& dst070, uint32_t& dst071,
       uint32_t& dst072, uint32_t& dst073, uint32_t& dst074, uint32_t& dst075,
       uint32_t& dst076, uint32_t& dst077, uint32_t& dst078, uint32_t& dst079,
       uint32_t& dst080, uint32_t& dst081, uint32_t& dst082, uint32_t& dst083,
       uint32_t& dst084, uint32_t& dst085, uint32_t& dst086, uint32_t& dst087,
       uint32_t& dst088, uint32_t& dst089, uint32_t& dst090, uint32_t& dst091,
       uint32_t& dst092, uint32_t& dst093, uint32_t& dst094, uint32_t& dst095,
       uint32_t& dst096, uint32_t& dst097, uint32_t& dst098, uint32_t& dst099,
       uint32_t& dst100, uint32_t& dst101, uint32_t& dst102, uint32_t& dst103,
       uint32_t& dst104, uint32_t& dst105, uint32_t& dst106, uint32_t& dst107,
       uint32_t& dst108, uint32_t& dst109, uint32_t& dst110, uint32_t& dst111,
       uint32_t& dst112, uint32_t& dst113, uint32_t& dst114, uint32_t& dst115,
       uint32_t& dst116, uint32_t& dst117, uint32_t& dst118, uint32_t& dst119,
       uint32_t& dst120, uint32_t& dst121, uint32_t& dst122, uint32_t& dst123,
       uint32_t& dst124, uint32_t& dst125, uint32_t& dst126, uint32_t& dst127)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x256b.x32.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63,"
                    "%64, %65, %66, %67,"
                    "%68, %69, %70, %71,"
                    "%72, %73, %74, %75,"
                    "%76, %77, %78, %79,"
                    "%80, %81, %82, %83,"
                    "%84, %85, %86, %87,"
                    "%88, %89, %90, %91,"
                    "%92, %93, %94, %95,"
                    "%96, %97, %98, %99,"
                    "%100, %101, %102, %103,"
                    "%104, %105, %106, %107,"
                    "%108, %109, %110, %111,"
                    "%112, %113, %114, %115,"
                    "%116, %117, %118, %119,"
                    "%120, %121, %122, %123,"
                    "%124, %125, %126, %127},"
                    "[%128];\n"
    :  "=r"(dst000), "=r"(dst001), "=r"(dst002), "=r"(dst003),
       "=r"(dst004), "=r"(dst005), "=r"(dst006), "=r"(dst007),
       "=r"(dst008), "=r"(dst009), "=r"(dst010), "=r"(dst011),
       "=r"(dst012), "=r"(dst013), "=r"(dst014), "=r"(dst015),
       "=r"(dst016), "=r"(dst017), "=r"(dst018), "=r"(dst019),
       "=r"(dst020), "=r"(dst021), "=r"(dst022), "=r"(dst023),
       "=r"(dst024), "=r"(dst025), "=r"(dst026), "=r"(dst027),
       "=r"(dst028), "=r"(dst029), "=r"(dst030), "=r"(dst031),
       "=r"(dst032), "=r"(dst033), "=r"(dst034), "=r"(dst035),
       "=r"(dst036), "=r"(dst037), "=r"(dst038), "=r"(dst039),
       "=r"(dst040), "=r"(dst041), "=r"(dst042), "=r"(dst043),
       "=r"(dst044), "=r"(dst045), "=r"(dst046), "=r"(dst047),
       "=r"(dst048), "=r"(dst049), "=r"(dst050), "=r"(dst051),
       "=r"(dst052), "=r"(dst053), "=r"(dst054), "=r"(dst055),
       "=r"(dst056), "=r"(dst057), "=r"(dst058), "=r"(dst059),
       "=r"(dst060), "=r"(dst061), "=r"(dst062), "=r"(dst063),
       "=r"(dst064), "=r"(dst065), "=r"(dst066), "=r"(dst067),
       "=r"(dst068), "=r"(dst069), "=r"(dst070), "=r"(dst071),
       "=r"(dst072), "=r"(dst073), "=r"(dst074), "=r"(dst075),
       "=r"(dst076), "=r"(dst077), "=r"(dst078), "=r"(dst079),
       "=r"(dst080), "=r"(dst081), "=r"(dst082), "=r"(dst083),
       "=r"(dst084), "=r"(dst085), "=r"(dst086), "=r"(dst087),
       "=r"(dst088), "=r"(dst089), "=r"(dst090), "=r"(dst091),
       "=r"(dst092), "=r"(dst093), "=r"(dst094), "=r"(dst095),
       "=r"(dst096), "=r"(dst097), "=r"(dst098), "=r"(dst099),
       "=r"(dst100), "=r"(dst101), "=r"(dst102), "=r"(dst103),
       "=r"(dst104), "=r"(dst105), "=r"(dst106), "=r"(dst107),
       "=r"(dst108), "=r"(dst109), "=r"(dst110), "=r"(dst111),
       "=r"(dst112), "=r"(dst113), "=r"(dst114), "=r"(dst115),
       "=r"(dst116), "=r"(dst117), "=r"(dst118), "=r"(dst119),
       "=r"(dst120), "=r"(dst121), "=r"(dst122), "=r"(dst123),
       "=r"(dst124), "=r"(dst125), "=r"(dst126), "=r"(dst127)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 32 times, packed 16b read
struct SM100_TMEM_LOAD_16dp256b32x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst000, uint32_t& dst001, uint32_t& dst002, uint32_t& dst003,
       uint32_t& dst004, uint32_t& dst005, uint32_t& dst006, uint32_t& dst007,
       uint32_t& dst008, uint32_t& dst009, uint32_t& dst010, uint32_t& dst011,
       uint32_t& dst012, uint32_t& dst013, uint32_t& dst014, uint32_t& dst015,
       uint32_t& dst016, uint32_t& dst017, uint32_t& dst018, uint32_t& dst019,
       uint32_t& dst020, uint32_t& dst021, uint32_t& dst022, uint32_t& dst023,
       uint32_t& dst024, uint32_t& dst025, uint32_t& dst026, uint32_t& dst027,
       uint32_t& dst028, uint32_t& dst029, uint32_t& dst030, uint32_t& dst031,
       uint32_t& dst032, uint32_t& dst033, uint32_t& dst034, uint32_t& dst035,
       uint32_t& dst036, uint32_t& dst037, uint32_t& dst038, uint32_t& dst039,
       uint32_t& dst040, uint32_t& dst041, uint32_t& dst042, uint32_t& dst043,
       uint32_t& dst044, uint32_t& dst045, uint32_t& dst046, uint32_t& dst047,
       uint32_t& dst048, uint32_t& dst049, uint32_t& dst050, uint32_t& dst051,
       uint32_t& dst052, uint32_t& dst053, uint32_t& dst054, uint32_t& dst055,
       uint32_t& dst056, uint32_t& dst057, uint32_t& dst058, uint32_t& dst059,
       uint32_t& dst060, uint32_t& dst061, uint32_t& dst062, uint32_t& dst063,
       uint32_t& dst064, uint32_t& dst065, uint32_t& dst066, uint32_t& dst067,
       uint32_t& dst068, uint32_t& dst069, uint32_t& dst070, uint32_t& dst071,
       uint32_t& dst072, uint32_t& dst073, uint32_t& dst074, uint32_t& dst075,
       uint32_t& dst076, uint32_t& dst077, uint32_t& dst078, uint32_t& dst079,
       uint32_t& dst080, uint32_t& dst081, uint32_t& dst082, uint32_t& dst083,
       uint32_t& dst084, uint32_t& dst085, uint32_t& dst086, uint32_t& dst087,
       uint32_t& dst088, uint32_t& dst089, uint32_t& dst090, uint32_t& dst091,
       uint32_t& dst092, uint32_t& dst093, uint32_t& dst094, uint32_t& dst095,
       uint32_t& dst096, uint32_t& dst097, uint32_t& dst098, uint32_t& dst099,
       uint32_t& dst100, uint32_t& dst101, uint32_t& dst102, uint32_t& dst103,
       uint32_t& dst104, uint32_t& dst105, uint32_t& dst106, uint32_t& dst107,
       uint32_t& dst108, uint32_t& dst109, uint32_t& dst110, uint32_t& dst111,
       uint32_t& dst112, uint32_t& dst113, uint32_t& dst114, uint32_t& dst115,
       uint32_t& dst116, uint32_t& dst117, uint32_t& dst118, uint32_t& dst119,
       uint32_t& dst120, uint32_t& dst121, uint32_t& dst122, uint32_t& dst123,
       uint32_t& dst124, uint32_t& dst125, uint32_t& dst126, uint32_t& dst127)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x256b.x32.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63,"
                    "%64, %65, %66, %67,"
                    "%68, %69, %70, %71,"
                    "%72, %73, %74, %75,"
                    "%76, %77, %78, %79,"
                    "%80, %81, %82, %83,"
                    "%84, %85, %86, %87,"
                    "%88, %89, %90, %91,"
                    "%92, %93, %94, %95,"
                    "%96, %97, %98, %99,"
                    "%100, %101, %102, %103,"
                    "%104, %105, %106, %107,"
                    "%108, %109, %110, %111,"
                    "%112, %113, %114, %115,"
                    "%116, %117, %118, %119,"
                    "%120, %121, %122, %123,"
                    "%124, %125, %126, %127},"
                    "[%128];\n"
    :  "=r"(dst000), "=r"(dst001), "=r"(dst002), "=r"(dst003),
       "=r"(dst004), "=r"(dst005), "=r"(dst006), "=r"(dst007),
       "=r"(dst008), "=r"(dst009), "=r"(dst010), "=r"(dst011),
       "=r"(dst012), "=r"(dst013), "=r"(dst014), "=r"(dst015),
       "=r"(dst016), "=r"(dst017), "=r"(dst018), "=r"(dst019),
       "=r"(dst020), "=r"(dst021), "=r"(dst022), "=r"(dst023),
       "=r"(dst024), "=r"(dst025), "=r"(dst026), "=r"(dst027),
       "=r"(dst028), "=r"(dst029), "=r"(dst030), "=r"(dst031),
       "=r"(dst032), "=r"(dst033), "=r"(dst034), "=r"(dst035),
       "=r"(dst036), "=r"(dst037), "=r"(dst038), "=r"(dst039),
       "=r"(dst040), "=r"(dst041), "=r"(dst042), "=r"(dst043),
       "=r"(dst044), "=r"(dst045), "=r"(dst046), "=r"(dst047),
       "=r"(dst048), "=r"(dst049), "=r"(dst050), "=r"(dst051),
       "=r"(dst052), "=r"(dst053), "=r"(dst054), "=r"(dst055),
       "=r"(dst056), "=r"(dst057), "=r"(dst058), "=r"(dst059),
       "=r"(dst060), "=r"(dst061), "=r"(dst062), "=r"(dst063),
       "=r"(dst064), "=r"(dst065), "=r"(dst066), "=r"(dst067),
       "=r"(dst068), "=r"(dst069), "=r"(dst070), "=r"(dst071),
       "=r"(dst072), "=r"(dst073), "=r"(dst074), "=r"(dst075),
       "=r"(dst076), "=r"(dst077), "=r"(dst078), "=r"(dst079),
       "=r"(dst080), "=r"(dst081), "=r"(dst082), "=r"(dst083),
       "=r"(dst084), "=r"(dst085), "=r"(dst086), "=r"(dst087),
       "=r"(dst088), "=r"(dst089), "=r"(dst090), "=r"(dst091),
       "=r"(dst092), "=r"(dst093), "=r"(dst094), "=r"(dst095),
       "=r"(dst096), "=r"(dst097), "=r"(dst098), "=r"(dst099),
       "=r"(dst100), "=r"(dst101), "=r"(dst102), "=r"(dst103),
       "=r"(dst104), "=r"(dst105), "=r"(dst106), "=r"(dst107),
       "=r"(dst108), "=r"(dst109), "=r"(dst110), "=r"(dst111),
       "=r"(dst112), "=r"(dst113), "=r"(dst114), "=r"(dst115),
       "=r"(dst116), "=r"(dst117), "=r"(dst118), "=r"(dst119),
       "=r"(dst120), "=r"(dst121), "=r"(dst122), "=r"(dst123),
       "=r"(dst124), "=r"(dst125), "=r"(dst126), "=r"(dst127)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 1 times
struct SM100_TMEM_LOAD_16dp128b1x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x128b.x1.b32"
                    "{%0, %1},"
                    "[%2];\n"
    :  "=r"(dst0), "=r"(dst1)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 1 times, packed 16b read
struct SM100_TMEM_LOAD_16dp128b1x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x128b.x1.pack::16b.b32"
                    "{%0, %1},"
                    "[%2];\n"
    :  "=r"(dst0), "=r"(dst1)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 2 times
struct SM100_TMEM_LOAD_16dp128b2x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x128b.x2.b32"
                    "{%0, %1, %2, %3},"
                    "[%4];\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 2 times, packed 16b read
struct SM100_TMEM_LOAD_16dp128b2x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32"
                    "{%0, %1, %2, %3},"
                    "[%4];\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 4 times
struct SM100_TMEM_LOAD_16dp128b4x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3,
       uint32_t& dst4, uint32_t& dst5, uint32_t& dst6, uint32_t& dst7)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x128b.x4.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7},"
                    "[%8];\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3),
       "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 4 times, packed 16b read
struct SM100_TMEM_LOAD_16dp128b4x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3,
       uint32_t& dst4, uint32_t& dst5, uint32_t& dst6, uint32_t& dst7)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7},"
                    "[%8];\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3),
       "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 8 times
struct SM100_TMEM_LOAD_16dp128b8x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x128b.x8.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15},"
                    "[%16];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 8 times, packed 16b read
struct SM100_TMEM_LOAD_16dp128b8x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x128b.x8.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15},"
                    "[%16];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 16 times
struct SM100_TMEM_LOAD_16dp128b16x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x128b.x16.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31},"
                    "[%32];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 16 times, packed 16b read
struct SM100_TMEM_LOAD_16dp128b16x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x128b.x16.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31},"
                    "[%32];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 32 times
struct SM100_TMEM_LOAD_16dp128b32x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31,
       uint32_t& dst32, uint32_t& dst33, uint32_t& dst34, uint32_t& dst35,
       uint32_t& dst36, uint32_t& dst37, uint32_t& dst38, uint32_t& dst39,
       uint32_t& dst40, uint32_t& dst41, uint32_t& dst42, uint32_t& dst43,
       uint32_t& dst44, uint32_t& dst45, uint32_t& dst46, uint32_t& dst47,
       uint32_t& dst48, uint32_t& dst49, uint32_t& dst50, uint32_t& dst51,
       uint32_t& dst52, uint32_t& dst53, uint32_t& dst54, uint32_t& dst55,
       uint32_t& dst56, uint32_t& dst57, uint32_t& dst58, uint32_t& dst59,
       uint32_t& dst60, uint32_t& dst61, uint32_t& dst62, uint32_t& dst63)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x128b.x32.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63},"
                    "[%64];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31),
       "=r"(dst32), "=r"(dst33), "=r"(dst34), "=r"(dst35),
       "=r"(dst36), "=r"(dst37), "=r"(dst38), "=r"(dst39),
       "=r"(dst40), "=r"(dst41), "=r"(dst42), "=r"(dst43),
       "=r"(dst44), "=r"(dst45), "=r"(dst46), "=r"(dst47),
       "=r"(dst48), "=r"(dst49), "=r"(dst50), "=r"(dst51),
       "=r"(dst52), "=r"(dst53), "=r"(dst54), "=r"(dst55),
       "=r"(dst56), "=r"(dst57), "=r"(dst58), "=r"(dst59),
       "=r"(dst60), "=r"(dst61), "=r"(dst62), "=r"(dst63)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 32 times, packed 16b read
struct SM100_TMEM_LOAD_16dp128b32x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31,
       uint32_t& dst32, uint32_t& dst33, uint32_t& dst34, uint32_t& dst35,
       uint32_t& dst36, uint32_t& dst37, uint32_t& dst38, uint32_t& dst39,
       uint32_t& dst40, uint32_t& dst41, uint32_t& dst42, uint32_t& dst43,
       uint32_t& dst44, uint32_t& dst45, uint32_t& dst46, uint32_t& dst47,
       uint32_t& dst48, uint32_t& dst49, uint32_t& dst50, uint32_t& dst51,
       uint32_t& dst52, uint32_t& dst53, uint32_t& dst54, uint32_t& dst55,
       uint32_t& dst56, uint32_t& dst57, uint32_t& dst58, uint32_t& dst59,
       uint32_t& dst60, uint32_t& dst61, uint32_t& dst62, uint32_t& dst63)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x128b.x32.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63},"
                    "[%64];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31),
       "=r"(dst32), "=r"(dst33), "=r"(dst34), "=r"(dst35),
       "=r"(dst36), "=r"(dst37), "=r"(dst38), "=r"(dst39),
       "=r"(dst40), "=r"(dst41), "=r"(dst42), "=r"(dst43),
       "=r"(dst44), "=r"(dst45), "=r"(dst46), "=r"(dst47),
       "=r"(dst48), "=r"(dst49), "=r"(dst50), "=r"(dst51),
       "=r"(dst52), "=r"(dst53), "=r"(dst54), "=r"(dst55),
       "=r"(dst56), "=r"(dst57), "=r"(dst58), "=r"(dst59),
       "=r"(dst60), "=r"(dst61), "=r"(dst62), "=r"(dst63)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 64 times
struct SM100_TMEM_LOAD_16dp128b64x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst000, uint32_t& dst001, uint32_t& dst002, uint32_t& dst003,
       uint32_t& dst004, uint32_t& dst005, uint32_t& dst006, uint32_t& dst007,
       uint32_t& dst008, uint32_t& dst009, uint32_t& dst010, uint32_t& dst011,
       uint32_t& dst012, uint32_t& dst013, uint32_t& dst014, uint32_t& dst015,
       uint32_t& dst016, uint32_t& dst017, uint32_t& dst018, uint32_t& dst019,
       uint32_t& dst020, uint32_t& dst021, uint32_t& dst022, uint32_t& dst023,
       uint32_t& dst024, uint32_t& dst025, uint32_t& dst026, uint32_t& dst027,
       uint32_t& dst028, uint32_t& dst029, uint32_t& dst030, uint32_t& dst031,
       uint32_t& dst032, uint32_t& dst033, uint32_t& dst034, uint32_t& dst035,
       uint32_t& dst036, uint32_t& dst037, uint32_t& dst038, uint32_t& dst039,
       uint32_t& dst040, uint32_t& dst041, uint32_t& dst042, uint32_t& dst043,
       uint32_t& dst044, uint32_t& dst045, uint32_t& dst046, uint32_t& dst047,
       uint32_t& dst048, uint32_t& dst049, uint32_t& dst050, uint32_t& dst051,
       uint32_t& dst052, uint32_t& dst053, uint32_t& dst054, uint32_t& dst055,
       uint32_t& dst056, uint32_t& dst057, uint32_t& dst058, uint32_t& dst059,
       uint32_t& dst060, uint32_t& dst061, uint32_t& dst062, uint32_t& dst063,
       uint32_t& dst064, uint32_t& dst065, uint32_t& dst066, uint32_t& dst067,
       uint32_t& dst068, uint32_t& dst069, uint32_t& dst070, uint32_t& dst071,
       uint32_t& dst072, uint32_t& dst073, uint32_t& dst074, uint32_t& dst075,
       uint32_t& dst076, uint32_t& dst077, uint32_t& dst078, uint32_t& dst079,
       uint32_t& dst080, uint32_t& dst081, uint32_t& dst082, uint32_t& dst083,
       uint32_t& dst084, uint32_t& dst085, uint32_t& dst086, uint32_t& dst087,
       uint32_t& dst088, uint32_t& dst089, uint32_t& dst090, uint32_t& dst091,
       uint32_t& dst092, uint32_t& dst093, uint32_t& dst094, uint32_t& dst095,
       uint32_t& dst096, uint32_t& dst097, uint32_t& dst098, uint32_t& dst099,
       uint32_t& dst100, uint32_t& dst101, uint32_t& dst102, uint32_t& dst103,
       uint32_t& dst104, uint32_t& dst105, uint32_t& dst106, uint32_t& dst107,
       uint32_t& dst108, uint32_t& dst109, uint32_t& dst110, uint32_t& dst111,
       uint32_t& dst112, uint32_t& dst113, uint32_t& dst114, uint32_t& dst115,
       uint32_t& dst116, uint32_t& dst117, uint32_t& dst118, uint32_t& dst119,
       uint32_t& dst120, uint32_t& dst121, uint32_t& dst122, uint32_t& dst123,
       uint32_t& dst124, uint32_t& dst125, uint32_t& dst126, uint32_t& dst127)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x128b.x64.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63,"
                    "%64, %65, %66, %67,"
                    "%68, %69, %70, %71,"
                    "%72, %73, %74, %75,"
                    "%76, %77, %78, %79,"
                    "%80, %81, %82, %83,"
                    "%84, %85, %86, %87,"
                    "%88, %89, %90, %91,"
                    "%92, %93, %94, %95,"
                    "%96, %97, %98, %99,"
                    "%100, %101, %102, %103,"
                    "%104, %105, %106, %107,"
                    "%108, %109, %110, %111,"
                    "%112, %113, %114, %115,"
                    "%116, %117, %118, %119,"
                    "%120, %121, %122, %123,"
                    "%124, %125, %126, %127},"
                    "[%128];\n"
    :  "=r"(dst000), "=r"(dst001), "=r"(dst002), "=r"(dst003),
       "=r"(dst004), "=r"(dst005), "=r"(dst006), "=r"(dst007),
       "=r"(dst008), "=r"(dst009), "=r"(dst010), "=r"(dst011),
       "=r"(dst012), "=r"(dst013), "=r"(dst014), "=r"(dst015),
       "=r"(dst016), "=r"(dst017), "=r"(dst018), "=r"(dst019),
       "=r"(dst020), "=r"(dst021), "=r"(dst022), "=r"(dst023),
       "=r"(dst024), "=r"(dst025), "=r"(dst026), "=r"(dst027),
       "=r"(dst028), "=r"(dst029), "=r"(dst030), "=r"(dst031),
       "=r"(dst032), "=r"(dst033), "=r"(dst034), "=r"(dst035),
       "=r"(dst036), "=r"(dst037), "=r"(dst038), "=r"(dst039),
       "=r"(dst040), "=r"(dst041), "=r"(dst042), "=r"(dst043),
       "=r"(dst044), "=r"(dst045), "=r"(dst046), "=r"(dst047),
       "=r"(dst048), "=r"(dst049), "=r"(dst050), "=r"(dst051),
       "=r"(dst052), "=r"(dst053), "=r"(dst054), "=r"(dst055),
       "=r"(dst056), "=r"(dst057), "=r"(dst058), "=r"(dst059),
       "=r"(dst060), "=r"(dst061), "=r"(dst062), "=r"(dst063),
       "=r"(dst064), "=r"(dst065), "=r"(dst066), "=r"(dst067),
       "=r"(dst068), "=r"(dst069), "=r"(dst070), "=r"(dst071),
       "=r"(dst072), "=r"(dst073), "=r"(dst074), "=r"(dst075),
       "=r"(dst076), "=r"(dst077), "=r"(dst078), "=r"(dst079),
       "=r"(dst080), "=r"(dst081), "=r"(dst082), "=r"(dst083),
       "=r"(dst084), "=r"(dst085), "=r"(dst086), "=r"(dst087),
       "=r"(dst088), "=r"(dst089), "=r"(dst090), "=r"(dst091),
       "=r"(dst092), "=r"(dst093), "=r"(dst094), "=r"(dst095),
       "=r"(dst096), "=r"(dst097), "=r"(dst098), "=r"(dst099),
       "=r"(dst100), "=r"(dst101), "=r"(dst102), "=r"(dst103),
       "=r"(dst104), "=r"(dst105), "=r"(dst106), "=r"(dst107),
       "=r"(dst108), "=r"(dst109), "=r"(dst110), "=r"(dst111),
       "=r"(dst112), "=r"(dst113), "=r"(dst114), "=r"(dst115),
       "=r"(dst116), "=r"(dst117), "=r"(dst118), "=r"(dst119),
       "=r"(dst120), "=r"(dst121), "=r"(dst122), "=r"(dst123),
       "=r"(dst124), "=r"(dst125), "=r"(dst126), "=r"(dst127)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 64 times, packed 16b read
struct SM100_TMEM_LOAD_16dp128b64x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst000, uint32_t& dst001, uint32_t& dst002, uint32_t& dst003,
       uint32_t& dst004, uint32_t& dst005, uint32_t& dst006, uint32_t& dst007,
       uint32_t& dst008, uint32_t& dst009, uint32_t& dst010, uint32_t& dst011,
       uint32_t& dst012, uint32_t& dst013, uint32_t& dst014, uint32_t& dst015,
       uint32_t& dst016, uint32_t& dst017, uint32_t& dst018, uint32_t& dst019,
       uint32_t& dst020, uint32_t& dst021, uint32_t& dst022, uint32_t& dst023,
       uint32_t& dst024, uint32_t& dst025, uint32_t& dst026, uint32_t& dst027,
       uint32_t& dst028, uint32_t& dst029, uint32_t& dst030, uint32_t& dst031,
       uint32_t& dst032, uint32_t& dst033, uint32_t& dst034, uint32_t& dst035,
       uint32_t& dst036, uint32_t& dst037, uint32_t& dst038, uint32_t& dst039,
       uint32_t& dst040, uint32_t& dst041, uint32_t& dst042, uint32_t& dst043,
       uint32_t& dst044, uint32_t& dst045, uint32_t& dst046, uint32_t& dst047,
       uint32_t& dst048, uint32_t& dst049, uint32_t& dst050, uint32_t& dst051,
       uint32_t& dst052, uint32_t& dst053, uint32_t& dst054, uint32_t& dst055,
       uint32_t& dst056, uint32_t& dst057, uint32_t& dst058, uint32_t& dst059,
       uint32_t& dst060, uint32_t& dst061, uint32_t& dst062, uint32_t& dst063,
       uint32_t& dst064, uint32_t& dst065, uint32_t& dst066, uint32_t& dst067,
       uint32_t& dst068, uint32_t& dst069, uint32_t& dst070, uint32_t& dst071,
       uint32_t& dst072, uint32_t& dst073, uint32_t& dst074, uint32_t& dst075,
       uint32_t& dst076, uint32_t& dst077, uint32_t& dst078, uint32_t& dst079,
       uint32_t& dst080, uint32_t& dst081, uint32_t& dst082, uint32_t& dst083,
       uint32_t& dst084, uint32_t& dst085, uint32_t& dst086, uint32_t& dst087,
       uint32_t& dst088, uint32_t& dst089, uint32_t& dst090, uint32_t& dst091,
       uint32_t& dst092, uint32_t& dst093, uint32_t& dst094, uint32_t& dst095,
       uint32_t& dst096, uint32_t& dst097, uint32_t& dst098, uint32_t& dst099,
       uint32_t& dst100, uint32_t& dst101, uint32_t& dst102, uint32_t& dst103,
       uint32_t& dst104, uint32_t& dst105, uint32_t& dst106, uint32_t& dst107,
       uint32_t& dst108, uint32_t& dst109, uint32_t& dst110, uint32_t& dst111,
       uint32_t& dst112, uint32_t& dst113, uint32_t& dst114, uint32_t& dst115,
       uint32_t& dst116, uint32_t& dst117, uint32_t& dst118, uint32_t& dst119,
       uint32_t& dst120, uint32_t& dst121, uint32_t& dst122, uint32_t& dst123,
       uint32_t& dst124, uint32_t& dst125, uint32_t& dst126, uint32_t& dst127)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x128b.x64.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63,"
                    "%64, %65, %66, %67,"
                    "%68, %69, %70, %71,"
                    "%72, %73, %74, %75,"
                    "%76, %77, %78, %79,"
                    "%80, %81, %82, %83,"
                    "%84, %85, %86, %87,"
                    "%88, %89, %90, %91,"
                    "%92, %93, %94, %95,"
                    "%96, %97, %98, %99,"
                    "%100, %101, %102, %103,"
                    "%104, %105, %106, %107,"
                    "%108, %109, %110, %111,"
                    "%112, %113, %114, %115,"
                    "%116, %117, %118, %119,"
                    "%120, %121, %122, %123,"
                    "%124, %125, %126, %127},"
                    "[%128];\n"
    :  "=r"(dst000), "=r"(dst001), "=r"(dst002), "=r"(dst003),
       "=r"(dst004), "=r"(dst005), "=r"(dst006), "=r"(dst007),
       "=r"(dst008), "=r"(dst009), "=r"(dst010), "=r"(dst011),
       "=r"(dst012), "=r"(dst013), "=r"(dst014), "=r"(dst015),
       "=r"(dst016), "=r"(dst017), "=r"(dst018), "=r"(dst019),
       "=r"(dst020), "=r"(dst021), "=r"(dst022), "=r"(dst023),
       "=r"(dst024), "=r"(dst025), "=r"(dst026), "=r"(dst027),
       "=r"(dst028), "=r"(dst029), "=r"(dst030), "=r"(dst031),
       "=r"(dst032), "=r"(dst033), "=r"(dst034), "=r"(dst035),
       "=r"(dst036), "=r"(dst037), "=r"(dst038), "=r"(dst039),
       "=r"(dst040), "=r"(dst041), "=r"(dst042), "=r"(dst043),
       "=r"(dst044), "=r"(dst045), "=r"(dst046), "=r"(dst047),
       "=r"(dst048), "=r"(dst049), "=r"(dst050), "=r"(dst051),
       "=r"(dst052), "=r"(dst053), "=r"(dst054), "=r"(dst055),
       "=r"(dst056), "=r"(dst057), "=r"(dst058), "=r"(dst059),
       "=r"(dst060), "=r"(dst061), "=r"(dst062), "=r"(dst063),
       "=r"(dst064), "=r"(dst065), "=r"(dst066), "=r"(dst067),
       "=r"(dst068), "=r"(dst069), "=r"(dst070), "=r"(dst071),
       "=r"(dst072), "=r"(dst073), "=r"(dst074), "=r"(dst075),
       "=r"(dst076), "=r"(dst077), "=r"(dst078), "=r"(dst079),
       "=r"(dst080), "=r"(dst081), "=r"(dst082), "=r"(dst083),
       "=r"(dst084), "=r"(dst085), "=r"(dst086), "=r"(dst087),
       "=r"(dst088), "=r"(dst089), "=r"(dst090), "=r"(dst091),
       "=r"(dst092), "=r"(dst093), "=r"(dst094), "=r"(dst095),
       "=r"(dst096), "=r"(dst097), "=r"(dst098), "=r"(dst099),
       "=r"(dst100), "=r"(dst101), "=r"(dst102), "=r"(dst103),
       "=r"(dst104), "=r"(dst105), "=r"(dst106), "=r"(dst107),
       "=r"(dst108), "=r"(dst109), "=r"(dst110), "=r"(dst111),
       "=r"(dst112), "=r"(dst113), "=r"(dst114), "=r"(dst115),
       "=r"(dst116), "=r"(dst117), "=r"(dst118), "=r"(dst119),
       "=r"(dst120), "=r"(dst121), "=r"(dst122), "=r"(dst123),
       "=r"(dst124), "=r"(dst125), "=r"(dst126), "=r"(dst127)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 1 times
struct SM100_TMEM_LOAD_16dp64b1x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x64b.x1.b32"
                    "{%0},"
                    "[%1];\n"
    :  "=r"(dst0)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 1 times, packed 16b read
struct SM100_TMEM_LOAD_16dp64b1x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x64b.x1.pack::16b.b32"
                    "{%0},"
                    "[%1];\n"
    :  "=r"(dst0)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 2 times
struct SM100_TMEM_LOAD_16dp64b2x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x64b.x2.b32"
                    "{%0, %1},"
                    "[%2];\n"
    :  "=r"(dst0), "=r"(dst1)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 2 times, packed 16b read
struct SM100_TMEM_LOAD_16dp64b2x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x64b.x2.pack::16b.b32"
                    "{%0, %1},"
                    "[%2];\n"
    :  "=r"(dst0), "=r"(dst1)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 4 times
struct SM100_TMEM_LOAD_16dp64b4x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x64b.x4.b32"
                    "{%0, %1, %2, %3},"
                    "[%4];\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 4 times, packed 16b read
struct SM100_TMEM_LOAD_16dp64b4x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x64b.x4.pack::16b.b32"
                    "{%0, %1, %2, %3},"
                    "[%4];\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 8 times
struct SM100_TMEM_LOAD_16dp64b8x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3,
       uint32_t& dst4, uint32_t& dst5, uint32_t& dst6, uint32_t& dst7)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x64b.x8.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7},"
                    "[%8];\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3),
       "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 8 times, packed 16b read
struct SM100_TMEM_LOAD_16dp64b8x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3,
       uint32_t& dst4, uint32_t& dst5, uint32_t& dst6, uint32_t& dst7)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x64b.x8.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7},"
                    "[%8];\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3),
       "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 16 times
struct SM100_TMEM_LOAD_16dp64b16x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x64b.x16.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15},"
                    "[%16];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 16 times, packed 16b read
struct SM100_TMEM_LOAD_16dp64b16x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x64b.x16.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15},"
                    "[%16];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 32 times
struct SM100_TMEM_LOAD_16dp64b32x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x64b.x32.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31},"
                    "[%32];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 32 times, packed 16b read
struct SM100_TMEM_LOAD_16dp64b32x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x64b.x32.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31},"
                    "[%32];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 64 times
struct SM100_TMEM_LOAD_16dp64b64x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31,
       uint32_t& dst32, uint32_t& dst33, uint32_t& dst34, uint32_t& dst35,
       uint32_t& dst36, uint32_t& dst37, uint32_t& dst38, uint32_t& dst39,
       uint32_t& dst40, uint32_t& dst41, uint32_t& dst42, uint32_t& dst43,
       uint32_t& dst44, uint32_t& dst45, uint32_t& dst46, uint32_t& dst47,
       uint32_t& dst48, uint32_t& dst49, uint32_t& dst50, uint32_t& dst51,
       uint32_t& dst52, uint32_t& dst53, uint32_t& dst54, uint32_t& dst55,
       uint32_t& dst56, uint32_t& dst57, uint32_t& dst58, uint32_t& dst59,
       uint32_t& dst60, uint32_t& dst61, uint32_t& dst62, uint32_t& dst63)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x64b.x64.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63},"
                    "[%64];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31),
       "=r"(dst32), "=r"(dst33), "=r"(dst34), "=r"(dst35),
       "=r"(dst36), "=r"(dst37), "=r"(dst38), "=r"(dst39),
       "=r"(dst40), "=r"(dst41), "=r"(dst42), "=r"(dst43),
       "=r"(dst44), "=r"(dst45), "=r"(dst46), "=r"(dst47),
       "=r"(dst48), "=r"(dst49), "=r"(dst50), "=r"(dst51),
       "=r"(dst52), "=r"(dst53), "=r"(dst54), "=r"(dst55),
       "=r"(dst56), "=r"(dst57), "=r"(dst58), "=r"(dst59),
       "=r"(dst60), "=r"(dst61), "=r"(dst62), "=r"(dst63)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 64 times, packed 16b read
struct SM100_TMEM_LOAD_16dp64b64x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31,
       uint32_t& dst32, uint32_t& dst33, uint32_t& dst34, uint32_t& dst35,
       uint32_t& dst36, uint32_t& dst37, uint32_t& dst38, uint32_t& dst39,
       uint32_t& dst40, uint32_t& dst41, uint32_t& dst42, uint32_t& dst43,
       uint32_t& dst44, uint32_t& dst45, uint32_t& dst46, uint32_t& dst47,
       uint32_t& dst48, uint32_t& dst49, uint32_t& dst50, uint32_t& dst51,
       uint32_t& dst52, uint32_t& dst53, uint32_t& dst54, uint32_t& dst55,
       uint32_t& dst56, uint32_t& dst57, uint32_t& dst58, uint32_t& dst59,
       uint32_t& dst60, uint32_t& dst61, uint32_t& dst62, uint32_t& dst63)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x64b.x64.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63},"
                    "[%64];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31),
       "=r"(dst32), "=r"(dst33), "=r"(dst34), "=r"(dst35),
       "=r"(dst36), "=r"(dst37), "=r"(dst38), "=r"(dst39),
       "=r"(dst40), "=r"(dst41), "=r"(dst42), "=r"(dst43),
       "=r"(dst44), "=r"(dst45), "=r"(dst46), "=r"(dst47),
       "=r"(dst48), "=r"(dst49), "=r"(dst50), "=r"(dst51),
       "=r"(dst52), "=r"(dst53), "=r"(dst54), "=r"(dst55),
       "=r"(dst56), "=r"(dst57), "=r"(dst58), "=r"(dst59),
       "=r"(dst60), "=r"(dst61), "=r"(dst62), "=r"(dst63)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 128 times
struct SM100_TMEM_LOAD_16dp64b128x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst000, uint32_t& dst001, uint32_t& dst002, uint32_t& dst003,
       uint32_t& dst004, uint32_t& dst005, uint32_t& dst006, uint32_t& dst007,
       uint32_t& dst008, uint32_t& dst009, uint32_t& dst010, uint32_t& dst011,
       uint32_t& dst012, uint32_t& dst013, uint32_t& dst014, uint32_t& dst015,
       uint32_t& dst016, uint32_t& dst017, uint32_t& dst018, uint32_t& dst019,
       uint32_t& dst020, uint32_t& dst021, uint32_t& dst022, uint32_t& dst023,
       uint32_t& dst024, uint32_t& dst025, uint32_t& dst026, uint32_t& dst027,
       uint32_t& dst028, uint32_t& dst029, uint32_t& dst030, uint32_t& dst031,
       uint32_t& dst032, uint32_t& dst033, uint32_t& dst034, uint32_t& dst035,
       uint32_t& dst036, uint32_t& dst037, uint32_t& dst038, uint32_t& dst039,
       uint32_t& dst040, uint32_t& dst041, uint32_t& dst042, uint32_t& dst043,
       uint32_t& dst044, uint32_t& dst045, uint32_t& dst046, uint32_t& dst047,
       uint32_t& dst048, uint32_t& dst049, uint32_t& dst050, uint32_t& dst051,
       uint32_t& dst052, uint32_t& dst053, uint32_t& dst054, uint32_t& dst055,
       uint32_t& dst056, uint32_t& dst057, uint32_t& dst058, uint32_t& dst059,
       uint32_t& dst060, uint32_t& dst061, uint32_t& dst062, uint32_t& dst063,
       uint32_t& dst064, uint32_t& dst065, uint32_t& dst066, uint32_t& dst067,
       uint32_t& dst068, uint32_t& dst069, uint32_t& dst070, uint32_t& dst071,
       uint32_t& dst072, uint32_t& dst073, uint32_t& dst074, uint32_t& dst075,
       uint32_t& dst076, uint32_t& dst077, uint32_t& dst078, uint32_t& dst079,
       uint32_t& dst080, uint32_t& dst081, uint32_t& dst082, uint32_t& dst083,
       uint32_t& dst084, uint32_t& dst085, uint32_t& dst086, uint32_t& dst087,
       uint32_t& dst088, uint32_t& dst089, uint32_t& dst090, uint32_t& dst091,
       uint32_t& dst092, uint32_t& dst093, uint32_t& dst094, uint32_t& dst095,
       uint32_t& dst096, uint32_t& dst097, uint32_t& dst098, uint32_t& dst099,
       uint32_t& dst100, uint32_t& dst101, uint32_t& dst102, uint32_t& dst103,
       uint32_t& dst104, uint32_t& dst105, uint32_t& dst106, uint32_t& dst107,
       uint32_t& dst108, uint32_t& dst109, uint32_t& dst110, uint32_t& dst111,
       uint32_t& dst112, uint32_t& dst113, uint32_t& dst114, uint32_t& dst115,
       uint32_t& dst116, uint32_t& dst117, uint32_t& dst118, uint32_t& dst119,
       uint32_t& dst120, uint32_t& dst121, uint32_t& dst122, uint32_t& dst123,
       uint32_t& dst124, uint32_t& dst125, uint32_t& dst126, uint32_t& dst127)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x64b.x128.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63,"
                    "%64, %65, %66, %67,"
                    "%68, %69, %70, %71,"
                    "%72, %73, %74, %75,"
                    "%76, %77, %78, %79,"
                    "%80, %81, %82, %83,"
                    "%84, %85, %86, %87,"
                    "%88, %89, %90, %91,"
                    "%92, %93, %94, %95,"
                    "%96, %97, %98, %99,"
                    "%100, %101, %102, %103,"
                    "%104, %105, %106, %107,"
                    "%108, %109, %110, %111,"
                    "%112, %113, %114, %115,"
                    "%116, %117, %118, %119,"
                    "%120, %121, %122, %123,"
                    "%124, %125, %126, %127},"
                    "[%128];\n"
    :  "=r"(dst000), "=r"(dst001), "=r"(dst002), "=r"(dst003),
       "=r"(dst004), "=r"(dst005), "=r"(dst006), "=r"(dst007),
       "=r"(dst008), "=r"(dst009), "=r"(dst010), "=r"(dst011),
       "=r"(dst012), "=r"(dst013), "=r"(dst014), "=r"(dst015),
       "=r"(dst016), "=r"(dst017), "=r"(dst018), "=r"(dst019),
       "=r"(dst020), "=r"(dst021), "=r"(dst022), "=r"(dst023),
       "=r"(dst024), "=r"(dst025), "=r"(dst026), "=r"(dst027),
       "=r"(dst028), "=r"(dst029), "=r"(dst030), "=r"(dst031),
       "=r"(dst032), "=r"(dst033), "=r"(dst034), "=r"(dst035),
       "=r"(dst036), "=r"(dst037), "=r"(dst038), "=r"(dst039),
       "=r"(dst040), "=r"(dst041), "=r"(dst042), "=r"(dst043),
       "=r"(dst044), "=r"(dst045), "=r"(dst046), "=r"(dst047),
       "=r"(dst048), "=r"(dst049), "=r"(dst050), "=r"(dst051),
       "=r"(dst052), "=r"(dst053), "=r"(dst054), "=r"(dst055),
       "=r"(dst056), "=r"(dst057), "=r"(dst058), "=r"(dst059),
       "=r"(dst060), "=r"(dst061), "=r"(dst062), "=r"(dst063),
       "=r"(dst064), "=r"(dst065), "=r"(dst066), "=r"(dst067),
       "=r"(dst068), "=r"(dst069), "=r"(dst070), "=r"(dst071),
       "=r"(dst072), "=r"(dst073), "=r"(dst074), "=r"(dst075),
       "=r"(dst076), "=r"(dst077), "=r"(dst078), "=r"(dst079),
       "=r"(dst080), "=r"(dst081), "=r"(dst082), "=r"(dst083),
       "=r"(dst084), "=r"(dst085), "=r"(dst086), "=r"(dst087),
       "=r"(dst088), "=r"(dst089), "=r"(dst090), "=r"(dst091),
       "=r"(dst092), "=r"(dst093), "=r"(dst094), "=r"(dst095),
       "=r"(dst096), "=r"(dst097), "=r"(dst098), "=r"(dst099),
       "=r"(dst100), "=r"(dst101), "=r"(dst102), "=r"(dst103),
       "=r"(dst104), "=r"(dst105), "=r"(dst106), "=r"(dst107),
       "=r"(dst108), "=r"(dst109), "=r"(dst110), "=r"(dst111),
       "=r"(dst112), "=r"(dst113), "=r"(dst114), "=r"(dst115),
       "=r"(dst116), "=r"(dst117), "=r"(dst118), "=r"(dst119),
       "=r"(dst120), "=r"(dst121), "=r"(dst122), "=r"(dst123),
       "=r"(dst124), "=r"(dst125), "=r"(dst126), "=r"(dst127)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 128 times, packed 16b read
struct SM100_TMEM_LOAD_16dp64b128x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst000, uint32_t& dst001, uint32_t& dst002, uint32_t& dst003,
       uint32_t& dst004, uint32_t& dst005, uint32_t& dst006, uint32_t& dst007,
       uint32_t& dst008, uint32_t& dst009, uint32_t& dst010, uint32_t& dst011,
       uint32_t& dst012, uint32_t& dst013, uint32_t& dst014, uint32_t& dst015,
       uint32_t& dst016, uint32_t& dst017, uint32_t& dst018, uint32_t& dst019,
       uint32_t& dst020, uint32_t& dst021, uint32_t& dst022, uint32_t& dst023,
       uint32_t& dst024, uint32_t& dst025, uint32_t& dst026, uint32_t& dst027,
       uint32_t& dst028, uint32_t& dst029, uint32_t& dst030, uint32_t& dst031,
       uint32_t& dst032, uint32_t& dst033, uint32_t& dst034, uint32_t& dst035,
       uint32_t& dst036, uint32_t& dst037, uint32_t& dst038, uint32_t& dst039,
       uint32_t& dst040, uint32_t& dst041, uint32_t& dst042, uint32_t& dst043,
       uint32_t& dst044, uint32_t& dst045, uint32_t& dst046, uint32_t& dst047,
       uint32_t& dst048, uint32_t& dst049, uint32_t& dst050, uint32_t& dst051,
       uint32_t& dst052, uint32_t& dst053, uint32_t& dst054, uint32_t& dst055,
       uint32_t& dst056, uint32_t& dst057, uint32_t& dst058, uint32_t& dst059,
       uint32_t& dst060, uint32_t& dst061, uint32_t& dst062, uint32_t& dst063,
       uint32_t& dst064, uint32_t& dst065, uint32_t& dst066, uint32_t& dst067,
       uint32_t& dst068, uint32_t& dst069, uint32_t& dst070, uint32_t& dst071,
       uint32_t& dst072, uint32_t& dst073, uint32_t& dst074, uint32_t& dst075,
       uint32_t& dst076, uint32_t& dst077, uint32_t& dst078, uint32_t& dst079,
       uint32_t& dst080, uint32_t& dst081, uint32_t& dst082, uint32_t& dst083,
       uint32_t& dst084, uint32_t& dst085, uint32_t& dst086, uint32_t& dst087,
       uint32_t& dst088, uint32_t& dst089, uint32_t& dst090, uint32_t& dst091,
       uint32_t& dst092, uint32_t& dst093, uint32_t& dst094, uint32_t& dst095,
       uint32_t& dst096, uint32_t& dst097, uint32_t& dst098, uint32_t& dst099,
       uint32_t& dst100, uint32_t& dst101, uint32_t& dst102, uint32_t& dst103,
       uint32_t& dst104, uint32_t& dst105, uint32_t& dst106, uint32_t& dst107,
       uint32_t& dst108, uint32_t& dst109, uint32_t& dst110, uint32_t& dst111,
       uint32_t& dst112, uint32_t& dst113, uint32_t& dst114, uint32_t& dst115,
       uint32_t& dst116, uint32_t& dst117, uint32_t& dst118, uint32_t& dst119,
       uint32_t& dst120, uint32_t& dst121, uint32_t& dst122, uint32_t& dst123,
       uint32_t& dst124, uint32_t& dst125, uint32_t& dst126, uint32_t& dst127)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x64b.x128.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63,"
                    "%64, %65, %66, %67,"
                    "%68, %69, %70, %71,"
                    "%72, %73, %74, %75,"
                    "%76, %77, %78, %79,"
                    "%80, %81, %82, %83,"
                    "%84, %85, %86, %87,"
                    "%88, %89, %90, %91,"
                    "%92, %93, %94, %95,"
                    "%96, %97, %98, %99,"
                    "%100, %101, %102, %103,"
                    "%104, %105, %106, %107,"
                    "%108, %109, %110, %111,"
                    "%112, %113, %114, %115,"
                    "%116, %117, %118, %119,"
                    "%120, %121, %122, %123,"
                    "%124, %125, %126, %127},"
                    "[%128];\n"
    :  "=r"(dst000), "=r"(dst001), "=r"(dst002), "=r"(dst003),
       "=r"(dst004), "=r"(dst005), "=r"(dst006), "=r"(dst007),
       "=r"(dst008), "=r"(dst009), "=r"(dst010), "=r"(dst011),
       "=r"(dst012), "=r"(dst013), "=r"(dst014), "=r"(dst015),
       "=r"(dst016), "=r"(dst017), "=r"(dst018), "=r"(dst019),
       "=r"(dst020), "=r"(dst021), "=r"(dst022), "=r"(dst023),
       "=r"(dst024), "=r"(dst025), "=r"(dst026), "=r"(dst027),
       "=r"(dst028), "=r"(dst029), "=r"(dst030), "=r"(dst031),
       "=r"(dst032), "=r"(dst033), "=r"(dst034), "=r"(dst035),
       "=r"(dst036), "=r"(dst037), "=r"(dst038), "=r"(dst039),
       "=r"(dst040), "=r"(dst041), "=r"(dst042), "=r"(dst043),
       "=r"(dst044), "=r"(dst045), "=r"(dst046), "=r"(dst047),
       "=r"(dst048), "=r"(dst049), "=r"(dst050), "=r"(dst051),
       "=r"(dst052), "=r"(dst053), "=r"(dst054), "=r"(dst055),
       "=r"(dst056), "=r"(dst057), "=r"(dst058), "=r"(dst059),
       "=r"(dst060), "=r"(dst061), "=r"(dst062), "=r"(dst063),
       "=r"(dst064), "=r"(dst065), "=r"(dst066), "=r"(dst067),
       "=r"(dst068), "=r"(dst069), "=r"(dst070), "=r"(dst071),
       "=r"(dst072), "=r"(dst073), "=r"(dst074), "=r"(dst075),
       "=r"(dst076), "=r"(dst077), "=r"(dst078), "=r"(dst079),
       "=r"(dst080), "=r"(dst081), "=r"(dst082), "=r"(dst083),
       "=r"(dst084), "=r"(dst085), "=r"(dst086), "=r"(dst087),
       "=r"(dst088), "=r"(dst089), "=r"(dst090), "=r"(dst091),
       "=r"(dst092), "=r"(dst093), "=r"(dst094), "=r"(dst095),
       "=r"(dst096), "=r"(dst097), "=r"(dst098), "=r"(dst099),
       "=r"(dst100), "=r"(dst101), "=r"(dst102), "=r"(dst103),
       "=r"(dst104), "=r"(dst105), "=r"(dst106), "=r"(dst107),
       "=r"(dst108), "=r"(dst109), "=r"(dst110), "=r"(dst111),
       "=r"(dst112), "=r"(dst113), "=r"(dst114), "=r"(dst115),
       "=r"(dst116), "=r"(dst117), "=r"(dst118), "=r"(dst119),
       "=r"(dst120), "=r"(dst121), "=r"(dst122), "=r"(dst123),
       "=r"(dst124), "=r"(dst125), "=r"(dst126), "=r"(dst127)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 1 times
struct SM100_TMEM_LOAD_16dp32b1x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x32bx2.x1.b32"
                    "{%0},"
                    "[%1], 1;\n"
    :  "=r"(dst0)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 1 times, packed 16b read
struct SM100_TMEM_LOAD_16dp32b1x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x32bx2.x1.pack::16b.b32"
                    "{%0},"
                    "[%1], 2;\n"
    :  "=r"(dst0)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 2 times
struct SM100_TMEM_LOAD_16dp32b2x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x32bx2.x2.b32"
                    "{%0, %1},"
                    "[%2], 2;\n"
    :  "=r"(dst0), "=r"(dst1)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 2 times, packed 16b read
struct SM100_TMEM_LOAD_16dp32b2x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x32bx2.x2.pack::16b.b32"
                    "{%0, %1},"
                    "[%2], 4;\n"
    :  "=r"(dst0), "=r"(dst1)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 4 times
struct SM100_TMEM_LOAD_16dp32b4x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x32bx2.x4.b32"
                    "{%0, %1, %2, %3},"
                    "[%4], 4;\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 4 times, packed 16b read
struct SM100_TMEM_LOAD_16dp32b4x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x32bx2.x4.pack::16b.b32"
                    "{%0, %1, %2, %3},"
                    "[%4], 8;\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 8 times
struct SM100_TMEM_LOAD_16dp32b8x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3,
       uint32_t& dst4, uint32_t& dst5, uint32_t& dst6, uint32_t& dst7)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7},"
                    "[%8], 8;\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3),
       "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 8 times, packed 16b read
struct SM100_TMEM_LOAD_16dp32b8x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3,
       uint32_t& dst4, uint32_t& dst5, uint32_t& dst6, uint32_t& dst7)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x32bx2.x8.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7},"
                    "[%8], 16;\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3),
       "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 16 times
struct SM100_TMEM_LOAD_16dp32b16x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15},"
                    "[%16], 16;\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 16 times, packed 16b read
struct SM100_TMEM_LOAD_16dp32b16x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x32bx2.x16.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15},"
                    "[%16], 32;\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 32 times
struct SM100_TMEM_LOAD_16dp32b32x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31},"
                    "[%32], 32;\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 32 times, packed 16b read
struct SM100_TMEM_LOAD_16dp32b32x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x32bx2.x32.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31},"
                    "[%32], 64;\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 64 times
struct SM100_TMEM_LOAD_16dp32b64x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31,
       uint32_t& dst32, uint32_t& dst33, uint32_t& dst34, uint32_t& dst35,
       uint32_t& dst36, uint32_t& dst37, uint32_t& dst38, uint32_t& dst39,
       uint32_t& dst40, uint32_t& dst41, uint32_t& dst42, uint32_t& dst43,
       uint32_t& dst44, uint32_t& dst45, uint32_t& dst46, uint32_t& dst47,
       uint32_t& dst48, uint32_t& dst49, uint32_t& dst50, uint32_t& dst51,
       uint32_t& dst52, uint32_t& dst53, uint32_t& dst54, uint32_t& dst55,
       uint32_t& dst56, uint32_t& dst57, uint32_t& dst58, uint32_t& dst59,
       uint32_t& dst60, uint32_t& dst61, uint32_t& dst62, uint32_t& dst63)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x32bx2.x64.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63},"
                    "[%64], 64;\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31),
       "=r"(dst32), "=r"(dst33), "=r"(dst34), "=r"(dst35),
       "=r"(dst36), "=r"(dst37), "=r"(dst38), "=r"(dst39),
       "=r"(dst40), "=r"(dst41), "=r"(dst42), "=r"(dst43),
       "=r"(dst44), "=r"(dst45), "=r"(dst46), "=r"(dst47),
       "=r"(dst48), "=r"(dst49), "=r"(dst50), "=r"(dst51),
       "=r"(dst52), "=r"(dst53), "=r"(dst54), "=r"(dst55),
       "=r"(dst56), "=r"(dst57), "=r"(dst58), "=r"(dst59),
       "=r"(dst60), "=r"(dst61), "=r"(dst62), "=r"(dst63)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 64 times, packed 16b read
struct SM100_TMEM_LOAD_16dp32b64x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31,
       uint32_t& dst32, uint32_t& dst33, uint32_t& dst34, uint32_t& dst35,
       uint32_t& dst36, uint32_t& dst37, uint32_t& dst38, uint32_t& dst39,
       uint32_t& dst40, uint32_t& dst41, uint32_t& dst42, uint32_t& dst43,
       uint32_t& dst44, uint32_t& dst45, uint32_t& dst46, uint32_t& dst47,
       uint32_t& dst48, uint32_t& dst49, uint32_t& dst50, uint32_t& dst51,
       uint32_t& dst52, uint32_t& dst53, uint32_t& dst54, uint32_t& dst55,
       uint32_t& dst56, uint32_t& dst57, uint32_t& dst58, uint32_t& dst59,
       uint32_t& dst60, uint32_t& dst61, uint32_t& dst62, uint32_t& dst63)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x32bx2.x64.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63},"
                    "[%64], 128;\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31),
       "=r"(dst32), "=r"(dst33), "=r"(dst34), "=r"(dst35),
       "=r"(dst36), "=r"(dst37), "=r"(dst38), "=r"(dst39),
       "=r"(dst40), "=r"(dst41), "=r"(dst42), "=r"(dst43),
       "=r"(dst44), "=r"(dst45), "=r"(dst46), "=r"(dst47),
       "=r"(dst48), "=r"(dst49), "=r"(dst50), "=r"(dst51),
       "=r"(dst52), "=r"(dst53), "=r"(dst54), "=r"(dst55),
       "=r"(dst56), "=r"(dst57), "=r"(dst58), "=r"(dst59),
       "=r"(dst60), "=r"(dst61), "=r"(dst62), "=r"(dst63)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 128 times
struct SM100_TMEM_LOAD_16dp32b128x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst000, uint32_t& dst001, uint32_t& dst002, uint32_t& dst003,
       uint32_t& dst004, uint32_t& dst005, uint32_t& dst006, uint32_t& dst007,
       uint32_t& dst008, uint32_t& dst009, uint32_t& dst010, uint32_t& dst011,
       uint32_t& dst012, uint32_t& dst013, uint32_t& dst014, uint32_t& dst015,
       uint32_t& dst016, uint32_t& dst017, uint32_t& dst018, uint32_t& dst019,
       uint32_t& dst020, uint32_t& dst021, uint32_t& dst022, uint32_t& dst023,
       uint32_t& dst024, uint32_t& dst025, uint32_t& dst026, uint32_t& dst027,
       uint32_t& dst028, uint32_t& dst029, uint32_t& dst030, uint32_t& dst031,
       uint32_t& dst032, uint32_t& dst033, uint32_t& dst034, uint32_t& dst035,
       uint32_t& dst036, uint32_t& dst037, uint32_t& dst038, uint32_t& dst039,
       uint32_t& dst040, uint32_t& dst041, uint32_t& dst042, uint32_t& dst043,
       uint32_t& dst044, uint32_t& dst045, uint32_t& dst046, uint32_t& dst047,
       uint32_t& dst048, uint32_t& dst049, uint32_t& dst050, uint32_t& dst051,
       uint32_t& dst052, uint32_t& dst053, uint32_t& dst054, uint32_t& dst055,
       uint32_t& dst056, uint32_t& dst057, uint32_t& dst058, uint32_t& dst059,
       uint32_t& dst060, uint32_t& dst061, uint32_t& dst062, uint32_t& dst063,
       uint32_t& dst064, uint32_t& dst065, uint32_t& dst066, uint32_t& dst067,
       uint32_t& dst068, uint32_t& dst069, uint32_t& dst070, uint32_t& dst071,
       uint32_t& dst072, uint32_t& dst073, uint32_t& dst074, uint32_t& dst075,
       uint32_t& dst076, uint32_t& dst077, uint32_t& dst078, uint32_t& dst079,
       uint32_t& dst080, uint32_t& dst081, uint32_t& dst082, uint32_t& dst083,
       uint32_t& dst084, uint32_t& dst085, uint32_t& dst086, uint32_t& dst087,
       uint32_t& dst088, uint32_t& dst089, uint32_t& dst090, uint32_t& dst091,
       uint32_t& dst092, uint32_t& dst093, uint32_t& dst094, uint32_t& dst095,
       uint32_t& dst096, uint32_t& dst097, uint32_t& dst098, uint32_t& dst099,
       uint32_t& dst100, uint32_t& dst101, uint32_t& dst102, uint32_t& dst103,
       uint32_t& dst104, uint32_t& dst105, uint32_t& dst106, uint32_t& dst107,
       uint32_t& dst108, uint32_t& dst109, uint32_t& dst110, uint32_t& dst111,
       uint32_t& dst112, uint32_t& dst113, uint32_t& dst114, uint32_t& dst115,
       uint32_t& dst116, uint32_t& dst117, uint32_t& dst118, uint32_t& dst119,
       uint32_t& dst120, uint32_t& dst121, uint32_t& dst122, uint32_t& dst123,
       uint32_t& dst124, uint32_t& dst125, uint32_t& dst126, uint32_t& dst127)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x32bx2.x128.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63,"
                    "%64, %65, %66, %67,"
                    "%68, %69, %70, %71,"
                    "%72, %73, %74, %75,"
                    "%76, %77, %78, %79,"
                    "%80, %81, %82, %83,"
                    "%84, %85, %86, %87,"
                    "%88, %89, %90, %91,"
                    "%92, %93, %94, %95,"
                    "%96, %97, %98, %99,"
                    "%100, %101, %102, %103,"
                    "%104, %105, %106, %107,"
                    "%108, %109, %110, %111,"
                    "%112, %113, %114, %115,"
                    "%116, %117, %118, %119,"
                    "%120, %121, %122, %123,"
                    "%124, %125, %126, %127},"
                    "[%128], 128;\n"
    :  "=r"(dst000), "=r"(dst001), "=r"(dst002), "=r"(dst003),
       "=r"(dst004), "=r"(dst005), "=r"(dst006), "=r"(dst007),
       "=r"(dst008), "=r"(dst009), "=r"(dst010), "=r"(dst011),
       "=r"(dst012), "=r"(dst013), "=r"(dst014), "=r"(dst015),
       "=r"(dst016), "=r"(dst017), "=r"(dst018), "=r"(dst019),
       "=r"(dst020), "=r"(dst021), "=r"(dst022), "=r"(dst023),
       "=r"(dst024), "=r"(dst025), "=r"(dst026), "=r"(dst027),
       "=r"(dst028), "=r"(dst029), "=r"(dst030), "=r"(dst031),
       "=r"(dst032), "=r"(dst033), "=r"(dst034), "=r"(dst035),
       "=r"(dst036), "=r"(dst037), "=r"(dst038), "=r"(dst039),
       "=r"(dst040), "=r"(dst041), "=r"(dst042), "=r"(dst043),
       "=r"(dst044), "=r"(dst045), "=r"(dst046), "=r"(dst047),
       "=r"(dst048), "=r"(dst049), "=r"(dst050), "=r"(dst051),
       "=r"(dst052), "=r"(dst053), "=r"(dst054), "=r"(dst055),
       "=r"(dst056), "=r"(dst057), "=r"(dst058), "=r"(dst059),
       "=r"(dst060), "=r"(dst061), "=r"(dst062), "=r"(dst063),
       "=r"(dst064), "=r"(dst065), "=r"(dst066), "=r"(dst067),
       "=r"(dst068), "=r"(dst069), "=r"(dst070), "=r"(dst071),
       "=r"(dst072), "=r"(dst073), "=r"(dst074), "=r"(dst075),
       "=r"(dst076), "=r"(dst077), "=r"(dst078), "=r"(dst079),
       "=r"(dst080), "=r"(dst081), "=r"(dst082), "=r"(dst083),
       "=r"(dst084), "=r"(dst085), "=r"(dst086), "=r"(dst087),
       "=r"(dst088), "=r"(dst089), "=r"(dst090), "=r"(dst091),
       "=r"(dst092), "=r"(dst093), "=r"(dst094), "=r"(dst095),
       "=r"(dst096), "=r"(dst097), "=r"(dst098), "=r"(dst099),
       "=r"(dst100), "=r"(dst101), "=r"(dst102), "=r"(dst103),
       "=r"(dst104), "=r"(dst105), "=r"(dst106), "=r"(dst107),
       "=r"(dst108), "=r"(dst109), "=r"(dst110), "=r"(dst111),
       "=r"(dst112), "=r"(dst113), "=r"(dst114), "=r"(dst115),
       "=r"(dst116), "=r"(dst117), "=r"(dst118), "=r"(dst119),
       "=r"(dst120), "=r"(dst121), "=r"(dst122), "=r"(dst123),
       "=r"(dst124), "=r"(dst125), "=r"(dst126), "=r"(dst127)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 128 times, packed 16b read
struct SM100_TMEM_LOAD_16dp32b128x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst000, uint32_t& dst001, uint32_t& dst002, uint32_t& dst003,
       uint32_t& dst004, uint32_t& dst005, uint32_t& dst006, uint32_t& dst007,
       uint32_t& dst008, uint32_t& dst009, uint32_t& dst010, uint32_t& dst011,
       uint32_t& dst012, uint32_t& dst013, uint32_t& dst014, uint32_t& dst015,
       uint32_t& dst016, uint32_t& dst017, uint32_t& dst018, uint32_t& dst019,
       uint32_t& dst020, uint32_t& dst021, uint32_t& dst022, uint32_t& dst023,
       uint32_t& dst024, uint32_t& dst025, uint32_t& dst026, uint32_t& dst027,
       uint32_t& dst028, uint32_t& dst029, uint32_t& dst030, uint32_t& dst031,
       uint32_t& dst032, uint32_t& dst033, uint32_t& dst034, uint32_t& dst035,
       uint32_t& dst036, uint32_t& dst037, uint32_t& dst038, uint32_t& dst039,
       uint32_t& dst040, uint32_t& dst041, uint32_t& dst042, uint32_t& dst043,
       uint32_t& dst044, uint32_t& dst045, uint32_t& dst046, uint32_t& dst047,
       uint32_t& dst048, uint32_t& dst049, uint32_t& dst050, uint32_t& dst051,
       uint32_t& dst052, uint32_t& dst053, uint32_t& dst054, uint32_t& dst055,
       uint32_t& dst056, uint32_t& dst057, uint32_t& dst058, uint32_t& dst059,
       uint32_t& dst060, uint32_t& dst061, uint32_t& dst062, uint32_t& dst063,
       uint32_t& dst064, uint32_t& dst065, uint32_t& dst066, uint32_t& dst067,
       uint32_t& dst068, uint32_t& dst069, uint32_t& dst070, uint32_t& dst071,
       uint32_t& dst072, uint32_t& dst073, uint32_t& dst074, uint32_t& dst075,
       uint32_t& dst076, uint32_t& dst077, uint32_t& dst078, uint32_t& dst079,
       uint32_t& dst080, uint32_t& dst081, uint32_t& dst082, uint32_t& dst083,
       uint32_t& dst084, uint32_t& dst085, uint32_t& dst086, uint32_t& dst087,
       uint32_t& dst088, uint32_t& dst089, uint32_t& dst090, uint32_t& dst091,
       uint32_t& dst092, uint32_t& dst093, uint32_t& dst094, uint32_t& dst095,
       uint32_t& dst096, uint32_t& dst097, uint32_t& dst098, uint32_t& dst099,
       uint32_t& dst100, uint32_t& dst101, uint32_t& dst102, uint32_t& dst103,
       uint32_t& dst104, uint32_t& dst105, uint32_t& dst106, uint32_t& dst107,
       uint32_t& dst108, uint32_t& dst109, uint32_t& dst110, uint32_t& dst111,
       uint32_t& dst112, uint32_t& dst113, uint32_t& dst114, uint32_t& dst115,
       uint32_t& dst116, uint32_t& dst117, uint32_t& dst118, uint32_t& dst119,
       uint32_t& dst120, uint32_t& dst121, uint32_t& dst122, uint32_t& dst123,
       uint32_t& dst124, uint32_t& dst125, uint32_t& dst126, uint32_t& dst127)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.16x32bx2.x128.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63,"
                    "%64, %65, %66, %67,"
                    "%68, %69, %70, %71,"
                    "%72, %73, %74, %75,"
                    "%76, %77, %78, %79,"
                    "%80, %81, %82, %83,"
                    "%84, %85, %86, %87,"
                    "%88, %89, %90, %91,"
                    "%92, %93, %94, %95,"
                    "%96, %97, %98, %99,"
                    "%100, %101, %102, %103,"
                    "%104, %105, %106, %107,"
                    "%108, %109, %110, %111,"
                    "%112, %113, %114, %115,"
                    "%116, %117, %118, %119,"
                    "%120, %121, %122, %123,"
                    "%124, %125, %126, %127},"
                    "[%128], 256;\n"
    :  "=r"(dst000), "=r"(dst001), "=r"(dst002), "=r"(dst003),
       "=r"(dst004), "=r"(dst005), "=r"(dst006), "=r"(dst007),
       "=r"(dst008), "=r"(dst009), "=r"(dst010), "=r"(dst011),
       "=r"(dst012), "=r"(dst013), "=r"(dst014), "=r"(dst015),
       "=r"(dst016), "=r"(dst017), "=r"(dst018), "=r"(dst019),
       "=r"(dst020), "=r"(dst021), "=r"(dst022), "=r"(dst023),
       "=r"(dst024), "=r"(dst025), "=r"(dst026), "=r"(dst027),
       "=r"(dst028), "=r"(dst029), "=r"(dst030), "=r"(dst031),
       "=r"(dst032), "=r"(dst033), "=r"(dst034), "=r"(dst035),
       "=r"(dst036), "=r"(dst037), "=r"(dst038), "=r"(dst039),
       "=r"(dst040), "=r"(dst041), "=r"(dst042), "=r"(dst043),
       "=r"(dst044), "=r"(dst045), "=r"(dst046), "=r"(dst047),
       "=r"(dst048), "=r"(dst049), "=r"(dst050), "=r"(dst051),
       "=r"(dst052), "=r"(dst053), "=r"(dst054), "=r"(dst055),
       "=r"(dst056), "=r"(dst057), "=r"(dst058), "=r"(dst059),
       "=r"(dst060), "=r"(dst061), "=r"(dst062), "=r"(dst063),
       "=r"(dst064), "=r"(dst065), "=r"(dst066), "=r"(dst067),
       "=r"(dst068), "=r"(dst069), "=r"(dst070), "=r"(dst071),
       "=r"(dst072), "=r"(dst073), "=r"(dst074), "=r"(dst075),
       "=r"(dst076), "=r"(dst077), "=r"(dst078), "=r"(dst079),
       "=r"(dst080), "=r"(dst081), "=r"(dst082), "=r"(dst083),
       "=r"(dst084), "=r"(dst085), "=r"(dst086), "=r"(dst087),
       "=r"(dst088), "=r"(dst089), "=r"(dst090), "=r"(dst091),
       "=r"(dst092), "=r"(dst093), "=r"(dst094), "=r"(dst095),
       "=r"(dst096), "=r"(dst097), "=r"(dst098), "=r"(dst099),
       "=r"(dst100), "=r"(dst101), "=r"(dst102), "=r"(dst103),
       "=r"(dst104), "=r"(dst105), "=r"(dst106), "=r"(dst107),
       "=r"(dst108), "=r"(dst109), "=r"(dst110), "=r"(dst111),
       "=r"(dst112), "=r"(dst113), "=r"(dst114), "=r"(dst115),
       "=r"(dst116), "=r"(dst117), "=r"(dst118), "=r"(dst119),
       "=r"(dst120), "=r"(dst121), "=r"(dst122), "=r"(dst123),
       "=r"(dst124), "=r"(dst125), "=r"(dst126), "=r"(dst127)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 1 times
struct SM100_TMEM_LOAD_32dp32b1x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    "{%0},"
                    "[%1];\n"
    :  "=r"(dst0)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 1 times, packed 16b read
struct SM100_TMEM_LOAD_32dp32b1x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32"
                    "{%0},"
                    "[%1];\n"
    :  "=r"(dst0)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 2 times
struct SM100_TMEM_LOAD_32dp32b2x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x2.b32"
                    "{%0, %1},"
                    "[%2];\n"
    :  "=r"(dst0), "=r"(dst1)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 2 times, packed 16b read
struct SM100_TMEM_LOAD_32dp32b2x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x2.pack::16b.b32"
                    "{%0, %1},"
                    "[%2];\n"
    :  "=r"(dst0), "=r"(dst1)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 4 times
struct SM100_TMEM_LOAD_32dp32b4x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x4.b32"
                    "{%0, %1, %2, %3},"
                    "[%4];\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 4 times, packed 16b read
struct SM100_TMEM_LOAD_32dp32b4x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x4.pack::16b.b32"
                    "{%0, %1, %2, %3},"
                    "[%4];\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 8 times
struct SM100_TMEM_LOAD_32dp32b8x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3,
       uint32_t& dst4, uint32_t& dst5, uint32_t& dst6, uint32_t& dst7)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x8.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7},"
                    "[%8];\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3),
       "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 8 times, packed 16b read
struct SM100_TMEM_LOAD_32dp32b8x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3,
       uint32_t& dst4, uint32_t& dst5, uint32_t& dst6, uint32_t& dst7)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x8.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7},"
                    "[%8];\n"
    :  "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3),
       "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 16 times
struct SM100_TMEM_LOAD_32dp32b16x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x16.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15},"
                    "[%16];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 16 times, packed 16b read
struct SM100_TMEM_LOAD_32dp32b16x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x16.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15},"
                    "[%16];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 32 times
struct SM100_TMEM_LOAD_32dp32b32x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31},"
                    "[%32];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 32 times, packed 16b read
struct SM100_TMEM_LOAD_32dp32b32x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x32.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31},"
                    "[%32];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 64 times
struct SM100_TMEM_LOAD_32dp32b64x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31,
       uint32_t& dst32, uint32_t& dst33, uint32_t& dst34, uint32_t& dst35,
       uint32_t& dst36, uint32_t& dst37, uint32_t& dst38, uint32_t& dst39,
       uint32_t& dst40, uint32_t& dst41, uint32_t& dst42, uint32_t& dst43,
       uint32_t& dst44, uint32_t& dst45, uint32_t& dst46, uint32_t& dst47,
       uint32_t& dst48, uint32_t& dst49, uint32_t& dst50, uint32_t& dst51,
       uint32_t& dst52, uint32_t& dst53, uint32_t& dst54, uint32_t& dst55,
       uint32_t& dst56, uint32_t& dst57, uint32_t& dst58, uint32_t& dst59,
       uint32_t& dst60, uint32_t& dst61, uint32_t& dst62, uint32_t& dst63)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x64.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63},"
                    "[%64];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31),
       "=r"(dst32), "=r"(dst33), "=r"(dst34), "=r"(dst35),
       "=r"(dst36), "=r"(dst37), "=r"(dst38), "=r"(dst39),
       "=r"(dst40), "=r"(dst41), "=r"(dst42), "=r"(dst43),
       "=r"(dst44), "=r"(dst45), "=r"(dst46), "=r"(dst47),
       "=r"(dst48), "=r"(dst49), "=r"(dst50), "=r"(dst51),
       "=r"(dst52), "=r"(dst53), "=r"(dst54), "=r"(dst55),
       "=r"(dst56), "=r"(dst57), "=r"(dst58), "=r"(dst59),
       "=r"(dst60), "=r"(dst61), "=r"(dst62), "=r"(dst63)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 64 times, packed 16b read
struct SM100_TMEM_LOAD_32dp32b64x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst00, uint32_t& dst01, uint32_t& dst02, uint32_t& dst03,
       uint32_t& dst04, uint32_t& dst05, uint32_t& dst06, uint32_t& dst07,
       uint32_t& dst08, uint32_t& dst09, uint32_t& dst10, uint32_t& dst11,
       uint32_t& dst12, uint32_t& dst13, uint32_t& dst14, uint32_t& dst15,
       uint32_t& dst16, uint32_t& dst17, uint32_t& dst18, uint32_t& dst19,
       uint32_t& dst20, uint32_t& dst21, uint32_t& dst22, uint32_t& dst23,
       uint32_t& dst24, uint32_t& dst25, uint32_t& dst26, uint32_t& dst27,
       uint32_t& dst28, uint32_t& dst29, uint32_t& dst30, uint32_t& dst31,
       uint32_t& dst32, uint32_t& dst33, uint32_t& dst34, uint32_t& dst35,
       uint32_t& dst36, uint32_t& dst37, uint32_t& dst38, uint32_t& dst39,
       uint32_t& dst40, uint32_t& dst41, uint32_t& dst42, uint32_t& dst43,
       uint32_t& dst44, uint32_t& dst45, uint32_t& dst46, uint32_t& dst47,
       uint32_t& dst48, uint32_t& dst49, uint32_t& dst50, uint32_t& dst51,
       uint32_t& dst52, uint32_t& dst53, uint32_t& dst54, uint32_t& dst55,
       uint32_t& dst56, uint32_t& dst57, uint32_t& dst58, uint32_t& dst59,
       uint32_t& dst60, uint32_t& dst61, uint32_t& dst62, uint32_t& dst63)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x64.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63},"
                    "[%64];\n"
    :  "=r"(dst00), "=r"(dst01), "=r"(dst02), "=r"(dst03),
       "=r"(dst04), "=r"(dst05), "=r"(dst06), "=r"(dst07),
       "=r"(dst08), "=r"(dst09), "=r"(dst10), "=r"(dst11),
       "=r"(dst12), "=r"(dst13), "=r"(dst14), "=r"(dst15),
       "=r"(dst16), "=r"(dst17), "=r"(dst18), "=r"(dst19),
       "=r"(dst20), "=r"(dst21), "=r"(dst22), "=r"(dst23),
       "=r"(dst24), "=r"(dst25), "=r"(dst26), "=r"(dst27),
       "=r"(dst28), "=r"(dst29), "=r"(dst30), "=r"(dst31),
       "=r"(dst32), "=r"(dst33), "=r"(dst34), "=r"(dst35),
       "=r"(dst36), "=r"(dst37), "=r"(dst38), "=r"(dst39),
       "=r"(dst40), "=r"(dst41), "=r"(dst42), "=r"(dst43),
       "=r"(dst44), "=r"(dst45), "=r"(dst46), "=r"(dst47),
       "=r"(dst48), "=r"(dst49), "=r"(dst50), "=r"(dst51),
       "=r"(dst52), "=r"(dst53), "=r"(dst54), "=r"(dst55),
       "=r"(dst56), "=r"(dst57), "=r"(dst58), "=r"(dst59),
       "=r"(dst60), "=r"(dst61), "=r"(dst62), "=r"(dst63)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 128 times
struct SM100_TMEM_LOAD_32dp32b128x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst000, uint32_t& dst001, uint32_t& dst002, uint32_t& dst003,
       uint32_t& dst004, uint32_t& dst005, uint32_t& dst006, uint32_t& dst007,
       uint32_t& dst008, uint32_t& dst009, uint32_t& dst010, uint32_t& dst011,
       uint32_t& dst012, uint32_t& dst013, uint32_t& dst014, uint32_t& dst015,
       uint32_t& dst016, uint32_t& dst017, uint32_t& dst018, uint32_t& dst019,
       uint32_t& dst020, uint32_t& dst021, uint32_t& dst022, uint32_t& dst023,
       uint32_t& dst024, uint32_t& dst025, uint32_t& dst026, uint32_t& dst027,
       uint32_t& dst028, uint32_t& dst029, uint32_t& dst030, uint32_t& dst031,
       uint32_t& dst032, uint32_t& dst033, uint32_t& dst034, uint32_t& dst035,
       uint32_t& dst036, uint32_t& dst037, uint32_t& dst038, uint32_t& dst039,
       uint32_t& dst040, uint32_t& dst041, uint32_t& dst042, uint32_t& dst043,
       uint32_t& dst044, uint32_t& dst045, uint32_t& dst046, uint32_t& dst047,
       uint32_t& dst048, uint32_t& dst049, uint32_t& dst050, uint32_t& dst051,
       uint32_t& dst052, uint32_t& dst053, uint32_t& dst054, uint32_t& dst055,
       uint32_t& dst056, uint32_t& dst057, uint32_t& dst058, uint32_t& dst059,
       uint32_t& dst060, uint32_t& dst061, uint32_t& dst062, uint32_t& dst063,
       uint32_t& dst064, uint32_t& dst065, uint32_t& dst066, uint32_t& dst067,
       uint32_t& dst068, uint32_t& dst069, uint32_t& dst070, uint32_t& dst071,
       uint32_t& dst072, uint32_t& dst073, uint32_t& dst074, uint32_t& dst075,
       uint32_t& dst076, uint32_t& dst077, uint32_t& dst078, uint32_t& dst079,
       uint32_t& dst080, uint32_t& dst081, uint32_t& dst082, uint32_t& dst083,
       uint32_t& dst084, uint32_t& dst085, uint32_t& dst086, uint32_t& dst087,
       uint32_t& dst088, uint32_t& dst089, uint32_t& dst090, uint32_t& dst091,
       uint32_t& dst092, uint32_t& dst093, uint32_t& dst094, uint32_t& dst095,
       uint32_t& dst096, uint32_t& dst097, uint32_t& dst098, uint32_t& dst099,
       uint32_t& dst100, uint32_t& dst101, uint32_t& dst102, uint32_t& dst103,
       uint32_t& dst104, uint32_t& dst105, uint32_t& dst106, uint32_t& dst107,
       uint32_t& dst108, uint32_t& dst109, uint32_t& dst110, uint32_t& dst111,
       uint32_t& dst112, uint32_t& dst113, uint32_t& dst114, uint32_t& dst115,
       uint32_t& dst116, uint32_t& dst117, uint32_t& dst118, uint32_t& dst119,
       uint32_t& dst120, uint32_t& dst121, uint32_t& dst122, uint32_t& dst123,
       uint32_t& dst124, uint32_t& dst125, uint32_t& dst126, uint32_t& dst127)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x128.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63,"
                    "%64, %65, %66, %67,"
                    "%68, %69, %70, %71,"
                    "%72, %73, %74, %75,"
                    "%76, %77, %78, %79,"
                    "%80, %81, %82, %83,"
                    "%84, %85, %86, %87,"
                    "%88, %89, %90, %91,"
                    "%92, %93, %94, %95,"
                    "%96, %97, %98, %99,"
                    "%100, %101, %102, %103,"
                    "%104, %105, %106, %107,"
                    "%108, %109, %110, %111,"
                    "%112, %113, %114, %115,"
                    "%116, %117, %118, %119,"
                    "%120, %121, %122, %123,"
                    "%124, %125, %126, %127},"
                    "[%128];\n"
    :  "=r"(dst000), "=r"(dst001), "=r"(dst002), "=r"(dst003),
       "=r"(dst004), "=r"(dst005), "=r"(dst006), "=r"(dst007),
       "=r"(dst008), "=r"(dst009), "=r"(dst010), "=r"(dst011),
       "=r"(dst012), "=r"(dst013), "=r"(dst014), "=r"(dst015),
       "=r"(dst016), "=r"(dst017), "=r"(dst018), "=r"(dst019),
       "=r"(dst020), "=r"(dst021), "=r"(dst022), "=r"(dst023),
       "=r"(dst024), "=r"(dst025), "=r"(dst026), "=r"(dst027),
       "=r"(dst028), "=r"(dst029), "=r"(dst030), "=r"(dst031),
       "=r"(dst032), "=r"(dst033), "=r"(dst034), "=r"(dst035),
       "=r"(dst036), "=r"(dst037), "=r"(dst038), "=r"(dst039),
       "=r"(dst040), "=r"(dst041), "=r"(dst042), "=r"(dst043),
       "=r"(dst044), "=r"(dst045), "=r"(dst046), "=r"(dst047),
       "=r"(dst048), "=r"(dst049), "=r"(dst050), "=r"(dst051),
       "=r"(dst052), "=r"(dst053), "=r"(dst054), "=r"(dst055),
       "=r"(dst056), "=r"(dst057), "=r"(dst058), "=r"(dst059),
       "=r"(dst060), "=r"(dst061), "=r"(dst062), "=r"(dst063),
       "=r"(dst064), "=r"(dst065), "=r"(dst066), "=r"(dst067),
       "=r"(dst068), "=r"(dst069), "=r"(dst070), "=r"(dst071),
       "=r"(dst072), "=r"(dst073), "=r"(dst074), "=r"(dst075),
       "=r"(dst076), "=r"(dst077), "=r"(dst078), "=r"(dst079),
       "=r"(dst080), "=r"(dst081), "=r"(dst082), "=r"(dst083),
       "=r"(dst084), "=r"(dst085), "=r"(dst086), "=r"(dst087),
       "=r"(dst088), "=r"(dst089), "=r"(dst090), "=r"(dst091),
       "=r"(dst092), "=r"(dst093), "=r"(dst094), "=r"(dst095),
       "=r"(dst096), "=r"(dst097), "=r"(dst098), "=r"(dst099),
       "=r"(dst100), "=r"(dst101), "=r"(dst102), "=r"(dst103),
       "=r"(dst104), "=r"(dst105), "=r"(dst106), "=r"(dst107),
       "=r"(dst108), "=r"(dst109), "=r"(dst110), "=r"(dst111),
       "=r"(dst112), "=r"(dst113), "=r"(dst114), "=r"(dst115),
       "=r"(dst116), "=r"(dst117), "=r"(dst118), "=r"(dst119),
       "=r"(dst120), "=r"(dst121), "=r"(dst122), "=r"(dst123),
       "=r"(dst124), "=r"(dst125), "=r"(dst126), "=r"(dst127)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 128 times, packed 16b read
struct SM100_TMEM_LOAD_32dp32b128x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst000, uint32_t& dst001, uint32_t& dst002, uint32_t& dst003,
       uint32_t& dst004, uint32_t& dst005, uint32_t& dst006, uint32_t& dst007,
       uint32_t& dst008, uint32_t& dst009, uint32_t& dst010, uint32_t& dst011,
       uint32_t& dst012, uint32_t& dst013, uint32_t& dst014, uint32_t& dst015,
       uint32_t& dst016, uint32_t& dst017, uint32_t& dst018, uint32_t& dst019,
       uint32_t& dst020, uint32_t& dst021, uint32_t& dst022, uint32_t& dst023,
       uint32_t& dst024, uint32_t& dst025, uint32_t& dst026, uint32_t& dst027,
       uint32_t& dst028, uint32_t& dst029, uint32_t& dst030, uint32_t& dst031,
       uint32_t& dst032, uint32_t& dst033, uint32_t& dst034, uint32_t& dst035,
       uint32_t& dst036, uint32_t& dst037, uint32_t& dst038, uint32_t& dst039,
       uint32_t& dst040, uint32_t& dst041, uint32_t& dst042, uint32_t& dst043,
       uint32_t& dst044, uint32_t& dst045, uint32_t& dst046, uint32_t& dst047,
       uint32_t& dst048, uint32_t& dst049, uint32_t& dst050, uint32_t& dst051,
       uint32_t& dst052, uint32_t& dst053, uint32_t& dst054, uint32_t& dst055,
       uint32_t& dst056, uint32_t& dst057, uint32_t& dst058, uint32_t& dst059,
       uint32_t& dst060, uint32_t& dst061, uint32_t& dst062, uint32_t& dst063,
       uint32_t& dst064, uint32_t& dst065, uint32_t& dst066, uint32_t& dst067,
       uint32_t& dst068, uint32_t& dst069, uint32_t& dst070, uint32_t& dst071,
       uint32_t& dst072, uint32_t& dst073, uint32_t& dst074, uint32_t& dst075,
       uint32_t& dst076, uint32_t& dst077, uint32_t& dst078, uint32_t& dst079,
       uint32_t& dst080, uint32_t& dst081, uint32_t& dst082, uint32_t& dst083,
       uint32_t& dst084, uint32_t& dst085, uint32_t& dst086, uint32_t& dst087,
       uint32_t& dst088, uint32_t& dst089, uint32_t& dst090, uint32_t& dst091,
       uint32_t& dst092, uint32_t& dst093, uint32_t& dst094, uint32_t& dst095,
       uint32_t& dst096, uint32_t& dst097, uint32_t& dst098, uint32_t& dst099,
       uint32_t& dst100, uint32_t& dst101, uint32_t& dst102, uint32_t& dst103,
       uint32_t& dst104, uint32_t& dst105, uint32_t& dst106, uint32_t& dst107,
       uint32_t& dst108, uint32_t& dst109, uint32_t& dst110, uint32_t& dst111,
       uint32_t& dst112, uint32_t& dst113, uint32_t& dst114, uint32_t& dst115,
       uint32_t& dst116, uint32_t& dst117, uint32_t& dst118, uint32_t& dst119,
       uint32_t& dst120, uint32_t& dst121, uint32_t& dst122, uint32_t& dst123,
       uint32_t& dst124, uint32_t& dst125, uint32_t& dst126, uint32_t& dst127)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x128.pack::16b.b32"
                    "{%0, %1, %2, %3,"
                    "%4, %5, %6, %7,"
                    "%8, %9, %10, %11,"
                    "%12, %13, %14, %15,"
                    "%16, %17, %18, %19,"
                    "%20, %21, %22, %23,"
                    "%24, %25, %26, %27,"
                    "%28, %29, %30, %31,"
                    "%32, %33, %34, %35,"
                    "%36, %37, %38, %39,"
                    "%40, %41, %42, %43,"
                    "%44, %45, %46, %47,"
                    "%48, %49, %50, %51,"
                    "%52, %53, %54, %55,"
                    "%56, %57, %58, %59,"
                    "%60, %61, %62, %63,"
                    "%64, %65, %66, %67,"
                    "%68, %69, %70, %71,"
                    "%72, %73, %74, %75,"
                    "%76, %77, %78, %79,"
                    "%80, %81, %82, %83,"
                    "%84, %85, %86, %87,"
                    "%88, %89, %90, %91,"
                    "%92, %93, %94, %95,"
                    "%96, %97, %98, %99,"
                    "%100, %101, %102, %103,"
                    "%104, %105, %106, %107,"
                    "%108, %109, %110, %111,"
                    "%112, %113, %114, %115,"
                    "%116, %117, %118, %119,"
                    "%120, %121, %122, %123,"
                    "%124, %125, %126, %127},"
                    "[%128];\n"
    :  "=r"(dst000), "=r"(dst001), "=r"(dst002), "=r"(dst003),
       "=r"(dst004), "=r"(dst005), "=r"(dst006), "=r"(dst007),
       "=r"(dst008), "=r"(dst009), "=r"(dst010), "=r"(dst011),
       "=r"(dst012), "=r"(dst013), "=r"(dst014), "=r"(dst015),
       "=r"(dst016), "=r"(dst017), "=r"(dst018), "=r"(dst019),
       "=r"(dst020), "=r"(dst021), "=r"(dst022), "=r"(dst023),
       "=r"(dst024), "=r"(dst025), "=r"(dst026), "=r"(dst027),
       "=r"(dst028), "=r"(dst029), "=r"(dst030), "=r"(dst031),
       "=r"(dst032), "=r"(dst033), "=r"(dst034), "=r"(dst035),
       "=r"(dst036), "=r"(dst037), "=r"(dst038), "=r"(dst039),
       "=r"(dst040), "=r"(dst041), "=r"(dst042), "=r"(dst043),
       "=r"(dst044), "=r"(dst045), "=r"(dst046), "=r"(dst047),
       "=r"(dst048), "=r"(dst049), "=r"(dst050), "=r"(dst051),
       "=r"(dst052), "=r"(dst053), "=r"(dst054), "=r"(dst055),
       "=r"(dst056), "=r"(dst057), "=r"(dst058), "=r"(dst059),
       "=r"(dst060), "=r"(dst061), "=r"(dst062), "=r"(dst063),
       "=r"(dst064), "=r"(dst065), "=r"(dst066), "=r"(dst067),
       "=r"(dst068), "=r"(dst069), "=r"(dst070), "=r"(dst071),
       "=r"(dst072), "=r"(dst073), "=r"(dst074), "=r"(dst075),
       "=r"(dst076), "=r"(dst077), "=r"(dst078), "=r"(dst079),
       "=r"(dst080), "=r"(dst081), "=r"(dst082), "=r"(dst083),
       "=r"(dst084), "=r"(dst085), "=r"(dst086), "=r"(dst087),
       "=r"(dst088), "=r"(dst089), "=r"(dst090), "=r"(dst091),
       "=r"(dst092), "=r"(dst093), "=r"(dst094), "=r"(dst095),
       "=r"(dst096), "=r"(dst097), "=r"(dst098), "=r"(dst099),
       "=r"(dst100), "=r"(dst101), "=r"(dst102), "=r"(dst103),
       "=r"(dst104), "=r"(dst105), "=r"(dst106), "=r"(dst107),
       "=r"(dst108), "=r"(dst109), "=r"(dst110), "=r"(dst111),
       "=r"(dst112), "=r"(dst113), "=r"(dst114), "=r"(dst115),
       "=r"(dst116), "=r"(dst117), "=r"(dst118), "=r"(dst119),
       "=r"(dst120), "=r"(dst121), "=r"(dst122), "=r"(dst123),
       "=r"(dst124), "=r"(dst125), "=r"(dst126), "=r"(dst127)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace SM100::TMEM::LOAD

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace SM100::TMEM::STORE {

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMEM STORE PTX definitions
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 1 times
struct SM100_TMEM_STORE_16dp256b1x
{
  using SRegisters = uint32_t[4];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x256b.x1.b32"
                    "[%0],"
                    "{%1, %2, %3, %4};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 1 times, expand 16b write
struct SM100_TMEM_STORE_16dp256b1x_16b
{
  using SRegisters = uint32_t[4];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x256b.x1.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 2 times
struct SM100_TMEM_STORE_16dp256b2x
{
  using SRegisters = uint32_t[8];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& src4, uint32_t const& src5, uint32_t const& src6, uint32_t const& src7,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x256b.x2.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3),
       "r"(src4), "r"(src5), "r"(src6), "r"(src7) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 2 times, expand 16b write
struct SM100_TMEM_STORE_16dp256b2x_16b
{
  using SRegisters = uint32_t[8];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& src4, uint32_t const& src5, uint32_t const& src6, uint32_t const& src7,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x256b.x2.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3),
       "r"(src4), "r"(src5), "r"(src6), "r"(src7) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 4 times
struct SM100_TMEM_STORE_16dp256b4x
{
  using SRegisters = uint32_t[16];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x256b.x4.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 4 times, expand 16b write
struct SM100_TMEM_STORE_16dp256b4x_16b
{
  using SRegisters = uint32_t[16];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x256b.x4.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 8 times
struct SM100_TMEM_STORE_16dp256b8x
{
  using SRegisters = uint32_t[32];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x256b.x8.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 8 times, expand 16b write
struct SM100_TMEM_STORE_16dp256b8x_16b
{
  using SRegisters = uint32_t[32];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x256b.x8.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 16 times
struct SM100_TMEM_STORE_16dp256b16x
{
  using SRegisters = uint32_t[64];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& src32, uint32_t const& src33, uint32_t const& src34, uint32_t const& src35,
       uint32_t const& src36, uint32_t const& src37, uint32_t const& src38, uint32_t const& src39,
       uint32_t const& src40, uint32_t const& src41, uint32_t const& src42, uint32_t const& src43,
       uint32_t const& src44, uint32_t const& src45, uint32_t const& src46, uint32_t const& src47,
       uint32_t const& src48, uint32_t const& src49, uint32_t const& src50, uint32_t const& src51,
       uint32_t const& src52, uint32_t const& src53, uint32_t const& src54, uint32_t const& src55,
       uint32_t const& src56, uint32_t const& src57, uint32_t const& src58, uint32_t const& src59,
       uint32_t const& src60, uint32_t const& src61, uint32_t const& src62, uint32_t const& src63,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x256b.x16.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31),
       "r"(src32), "r"(src33), "r"(src34), "r"(src35),
       "r"(src36), "r"(src37), "r"(src38), "r"(src39),
       "r"(src40), "r"(src41), "r"(src42), "r"(src43),
       "r"(src44), "r"(src45), "r"(src46), "r"(src47),
       "r"(src48), "r"(src49), "r"(src50), "r"(src51),
       "r"(src52), "r"(src53), "r"(src54), "r"(src55),
       "r"(src56), "r"(src57), "r"(src58), "r"(src59),
       "r"(src60), "r"(src61), "r"(src62), "r"(src63) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 16 times, expand 16b write
struct SM100_TMEM_STORE_16dp256b16x_16b
{
  using SRegisters = uint32_t[64];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& src32, uint32_t const& src33, uint32_t const& src34, uint32_t const& src35,
       uint32_t const& src36, uint32_t const& src37, uint32_t const& src38, uint32_t const& src39,
       uint32_t const& src40, uint32_t const& src41, uint32_t const& src42, uint32_t const& src43,
       uint32_t const& src44, uint32_t const& src45, uint32_t const& src46, uint32_t const& src47,
       uint32_t const& src48, uint32_t const& src49, uint32_t const& src50, uint32_t const& src51,
       uint32_t const& src52, uint32_t const& src53, uint32_t const& src54, uint32_t const& src55,
       uint32_t const& src56, uint32_t const& src57, uint32_t const& src58, uint32_t const& src59,
       uint32_t const& src60, uint32_t const& src61, uint32_t const& src62, uint32_t const& src63,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x256b.x16.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31),
       "r"(src32), "r"(src33), "r"(src34), "r"(src35),
       "r"(src36), "r"(src37), "r"(src38), "r"(src39),
       "r"(src40), "r"(src41), "r"(src42), "r"(src43),
       "r"(src44), "r"(src45), "r"(src46), "r"(src47),
       "r"(src48), "r"(src49), "r"(src50), "r"(src51),
       "r"(src52), "r"(src53), "r"(src54), "r"(src55),
       "r"(src56), "r"(src57), "r"(src58), "r"(src59),
       "r"(src60), "r"(src61), "r"(src62), "r"(src63) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 32 times
struct SM100_TMEM_STORE_16dp256b32x
{
  using SRegisters = uint32_t[128];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src000, uint32_t const& src001, uint32_t const& src002, uint32_t const& src003,
       uint32_t const& src004, uint32_t const& src005, uint32_t const& src006, uint32_t const& src007,
       uint32_t const& src008, uint32_t const& src009, uint32_t const& src010, uint32_t const& src011,
       uint32_t const& src012, uint32_t const& src013, uint32_t const& src014, uint32_t const& src015,
       uint32_t const& src016, uint32_t const& src017, uint32_t const& src018, uint32_t const& src019,
       uint32_t const& src020, uint32_t const& src021, uint32_t const& src022, uint32_t const& src023,
       uint32_t const& src024, uint32_t const& src025, uint32_t const& src026, uint32_t const& src027,
       uint32_t const& src028, uint32_t const& src029, uint32_t const& src030, uint32_t const& src031,
       uint32_t const& src032, uint32_t const& src033, uint32_t const& src034, uint32_t const& src035,
       uint32_t const& src036, uint32_t const& src037, uint32_t const& src038, uint32_t const& src039,
       uint32_t const& src040, uint32_t const& src041, uint32_t const& src042, uint32_t const& src043,
       uint32_t const& src044, uint32_t const& src045, uint32_t const& src046, uint32_t const& src047,
       uint32_t const& src048, uint32_t const& src049, uint32_t const& src050, uint32_t const& src051,
       uint32_t const& src052, uint32_t const& src053, uint32_t const& src054, uint32_t const& src055,
       uint32_t const& src056, uint32_t const& src057, uint32_t const& src058, uint32_t const& src059,
       uint32_t const& src060, uint32_t const& src061, uint32_t const& src062, uint32_t const& src063,
       uint32_t const& src064, uint32_t const& src065, uint32_t const& src066, uint32_t const& src067,
       uint32_t const& src068, uint32_t const& src069, uint32_t const& src070, uint32_t const& src071,
       uint32_t const& src072, uint32_t const& src073, uint32_t const& src074, uint32_t const& src075,
       uint32_t const& src076, uint32_t const& src077, uint32_t const& src078, uint32_t const& src079,
       uint32_t const& src080, uint32_t const& src081, uint32_t const& src082, uint32_t const& src083,
       uint32_t const& src084, uint32_t const& src085, uint32_t const& src086, uint32_t const& src087,
       uint32_t const& src088, uint32_t const& src089, uint32_t const& src090, uint32_t const& src091,
       uint32_t const& src092, uint32_t const& src093, uint32_t const& src094, uint32_t const& src095,
       uint32_t const& src096, uint32_t const& src097, uint32_t const& src098, uint32_t const& src099,
       uint32_t const& src100, uint32_t const& src101, uint32_t const& src102, uint32_t const& src103,
       uint32_t const& src104, uint32_t const& src105, uint32_t const& src106, uint32_t const& src107,
       uint32_t const& src108, uint32_t const& src109, uint32_t const& src110, uint32_t const& src111,
       uint32_t const& src112, uint32_t const& src113, uint32_t const& src114, uint32_t const& src115,
       uint32_t const& src116, uint32_t const& src117, uint32_t const& src118, uint32_t const& src119,
       uint32_t const& src120, uint32_t const& src121, uint32_t const& src122, uint32_t const& src123,
       uint32_t const& src124, uint32_t const& src125, uint32_t const& src126, uint32_t const& src127,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x256b.x32.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64,"
                    "%65, %66, %67, %68,"
                    "%69, %70, %71, %72,"
                    "%73, %74, %75, %76,"
                    "%77, %78, %79, %80,"
                    "%81, %82, %83, %84,"
                    "%85, %86, %87, %88,"
                    "%89, %90, %91, %92,"
                    "%93, %94, %95, %96,"
                    "%97, %98, %99, %100,"
                    "%101, %102, %103, %104,"
                    "%105, %106, %107, %108,"
                    "%109, %110, %111, %112,"
                    "%113, %114, %115, %116,"
                    "%117, %118, %119, %120,"
                    "%121, %122, %123, %124,"
                    "%125, %126, %127, %128};\n"
    :
    :  "r"(dst_addr), "r"(src000), "r"(src001), "r"(src002), "r"(src003),
       "r"(src004), "r"(src005), "r"(src006), "r"(src007),
       "r"(src008), "r"(src009), "r"(src010), "r"(src011),
       "r"(src012), "r"(src013), "r"(src014), "r"(src015),
       "r"(src016), "r"(src017), "r"(src018), "r"(src019),
       "r"(src020), "r"(src021), "r"(src022), "r"(src023),
       "r"(src024), "r"(src025), "r"(src026), "r"(src027),
       "r"(src028), "r"(src029), "r"(src030), "r"(src031),
       "r"(src032), "r"(src033), "r"(src034), "r"(src035),
       "r"(src036), "r"(src037), "r"(src038), "r"(src039),
       "r"(src040), "r"(src041), "r"(src042), "r"(src043),
       "r"(src044), "r"(src045), "r"(src046), "r"(src047),
       "r"(src048), "r"(src049), "r"(src050), "r"(src051),
       "r"(src052), "r"(src053), "r"(src054), "r"(src055),
       "r"(src056), "r"(src057), "r"(src058), "r"(src059),
       "r"(src060), "r"(src061), "r"(src062), "r"(src063),
       "r"(src064), "r"(src065), "r"(src066), "r"(src067),
       "r"(src068), "r"(src069), "r"(src070), "r"(src071),
       "r"(src072), "r"(src073), "r"(src074), "r"(src075),
       "r"(src076), "r"(src077), "r"(src078), "r"(src079),
       "r"(src080), "r"(src081), "r"(src082), "r"(src083),
       "r"(src084), "r"(src085), "r"(src086), "r"(src087),
       "r"(src088), "r"(src089), "r"(src090), "r"(src091),
       "r"(src092), "r"(src093), "r"(src094), "r"(src095),
       "r"(src096), "r"(src097), "r"(src098), "r"(src099),
       "r"(src100), "r"(src101), "r"(src102), "r"(src103),
       "r"(src104), "r"(src105), "r"(src106), "r"(src107),
       "r"(src108), "r"(src109), "r"(src110), "r"(src111),
       "r"(src112), "r"(src113), "r"(src114), "r"(src115),
       "r"(src116), "r"(src117), "r"(src118), "r"(src119),
       "r"(src120), "r"(src121), "r"(src122), "r"(src123),
       "r"(src124), "r"(src125), "r"(src126), "r"(src127) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 256-bit pattern, repeated 32 times, expand 16b write
struct SM100_TMEM_STORE_16dp256b32x_16b
{
  using SRegisters = uint32_t[128];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src000, uint32_t const& src001, uint32_t const& src002, uint32_t const& src003,
       uint32_t const& src004, uint32_t const& src005, uint32_t const& src006, uint32_t const& src007,
       uint32_t const& src008, uint32_t const& src009, uint32_t const& src010, uint32_t const& src011,
       uint32_t const& src012, uint32_t const& src013, uint32_t const& src014, uint32_t const& src015,
       uint32_t const& src016, uint32_t const& src017, uint32_t const& src018, uint32_t const& src019,
       uint32_t const& src020, uint32_t const& src021, uint32_t const& src022, uint32_t const& src023,
       uint32_t const& src024, uint32_t const& src025, uint32_t const& src026, uint32_t const& src027,
       uint32_t const& src028, uint32_t const& src029, uint32_t const& src030, uint32_t const& src031,
       uint32_t const& src032, uint32_t const& src033, uint32_t const& src034, uint32_t const& src035,
       uint32_t const& src036, uint32_t const& src037, uint32_t const& src038, uint32_t const& src039,
       uint32_t const& src040, uint32_t const& src041, uint32_t const& src042, uint32_t const& src043,
       uint32_t const& src044, uint32_t const& src045, uint32_t const& src046, uint32_t const& src047,
       uint32_t const& src048, uint32_t const& src049, uint32_t const& src050, uint32_t const& src051,
       uint32_t const& src052, uint32_t const& src053, uint32_t const& src054, uint32_t const& src055,
       uint32_t const& src056, uint32_t const& src057, uint32_t const& src058, uint32_t const& src059,
       uint32_t const& src060, uint32_t const& src061, uint32_t const& src062, uint32_t const& src063,
       uint32_t const& src064, uint32_t const& src065, uint32_t const& src066, uint32_t const& src067,
       uint32_t const& src068, uint32_t const& src069, uint32_t const& src070, uint32_t const& src071,
       uint32_t const& src072, uint32_t const& src073, uint32_t const& src074, uint32_t const& src075,
       uint32_t const& src076, uint32_t const& src077, uint32_t const& src078, uint32_t const& src079,
       uint32_t const& src080, uint32_t const& src081, uint32_t const& src082, uint32_t const& src083,
       uint32_t const& src084, uint32_t const& src085, uint32_t const& src086, uint32_t const& src087,
       uint32_t const& src088, uint32_t const& src089, uint32_t const& src090, uint32_t const& src091,
       uint32_t const& src092, uint32_t const& src093, uint32_t const& src094, uint32_t const& src095,
       uint32_t const& src096, uint32_t const& src097, uint32_t const& src098, uint32_t const& src099,
       uint32_t const& src100, uint32_t const& src101, uint32_t const& src102, uint32_t const& src103,
       uint32_t const& src104, uint32_t const& src105, uint32_t const& src106, uint32_t const& src107,
       uint32_t const& src108, uint32_t const& src109, uint32_t const& src110, uint32_t const& src111,
       uint32_t const& src112, uint32_t const& src113, uint32_t const& src114, uint32_t const& src115,
       uint32_t const& src116, uint32_t const& src117, uint32_t const& src118, uint32_t const& src119,
       uint32_t const& src120, uint32_t const& src121, uint32_t const& src122, uint32_t const& src123,
       uint32_t const& src124, uint32_t const& src125, uint32_t const& src126, uint32_t const& src127,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x256b.x32.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64,"
                    "%65, %66, %67, %68,"
                    "%69, %70, %71, %72,"
                    "%73, %74, %75, %76,"
                    "%77, %78, %79, %80,"
                    "%81, %82, %83, %84,"
                    "%85, %86, %87, %88,"
                    "%89, %90, %91, %92,"
                    "%93, %94, %95, %96,"
                    "%97, %98, %99, %100,"
                    "%101, %102, %103, %104,"
                    "%105, %106, %107, %108,"
                    "%109, %110, %111, %112,"
                    "%113, %114, %115, %116,"
                    "%117, %118, %119, %120,"
                    "%121, %122, %123, %124,"
                    "%125, %126, %127, %128};\n"
    :
    :  "r"(dst_addr), "r"(src000), "r"(src001), "r"(src002), "r"(src003),
       "r"(src004), "r"(src005), "r"(src006), "r"(src007),
       "r"(src008), "r"(src009), "r"(src010), "r"(src011),
       "r"(src012), "r"(src013), "r"(src014), "r"(src015),
       "r"(src016), "r"(src017), "r"(src018), "r"(src019),
       "r"(src020), "r"(src021), "r"(src022), "r"(src023),
       "r"(src024), "r"(src025), "r"(src026), "r"(src027),
       "r"(src028), "r"(src029), "r"(src030), "r"(src031),
       "r"(src032), "r"(src033), "r"(src034), "r"(src035),
       "r"(src036), "r"(src037), "r"(src038), "r"(src039),
       "r"(src040), "r"(src041), "r"(src042), "r"(src043),
       "r"(src044), "r"(src045), "r"(src046), "r"(src047),
       "r"(src048), "r"(src049), "r"(src050), "r"(src051),
       "r"(src052), "r"(src053), "r"(src054), "r"(src055),
       "r"(src056), "r"(src057), "r"(src058), "r"(src059),
       "r"(src060), "r"(src061), "r"(src062), "r"(src063),
       "r"(src064), "r"(src065), "r"(src066), "r"(src067),
       "r"(src068), "r"(src069), "r"(src070), "r"(src071),
       "r"(src072), "r"(src073), "r"(src074), "r"(src075),
       "r"(src076), "r"(src077), "r"(src078), "r"(src079),
       "r"(src080), "r"(src081), "r"(src082), "r"(src083),
       "r"(src084), "r"(src085), "r"(src086), "r"(src087),
       "r"(src088), "r"(src089), "r"(src090), "r"(src091),
       "r"(src092), "r"(src093), "r"(src094), "r"(src095),
       "r"(src096), "r"(src097), "r"(src098), "r"(src099),
       "r"(src100), "r"(src101), "r"(src102), "r"(src103),
       "r"(src104), "r"(src105), "r"(src106), "r"(src107),
       "r"(src108), "r"(src109), "r"(src110), "r"(src111),
       "r"(src112), "r"(src113), "r"(src114), "r"(src115),
       "r"(src116), "r"(src117), "r"(src118), "r"(src119),
       "r"(src120), "r"(src121), "r"(src122), "r"(src123),
       "r"(src124), "r"(src125), "r"(src126), "r"(src127) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 1 times
struct SM100_TMEM_STORE_16dp128b1x
{
  using SRegisters = uint32_t[2];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x128b.x1.b32"
                    "[%0],"
                    "{%1, %2};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 1 times, expand 16b write
struct SM100_TMEM_STORE_16dp128b1x_16b
{
  using SRegisters = uint32_t[2];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x128b.x1.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 2 times
struct SM100_TMEM_STORE_16dp128b2x
{
  using SRegisters = uint32_t[4];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x128b.x2.b32"
                    "[%0],"
                    "{%1, %2, %3, %4};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 2 times, expand 16b write
struct SM100_TMEM_STORE_16dp128b2x_16b
{
  using SRegisters = uint32_t[4];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x128b.x2.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 4 times
struct SM100_TMEM_STORE_16dp128b4x
{
  using SRegisters = uint32_t[8];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& src4, uint32_t const& src5, uint32_t const& src6, uint32_t const& src7,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x128b.x4.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3),
       "r"(src4), "r"(src5), "r"(src6), "r"(src7) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 4 times, expand 16b write
struct SM100_TMEM_STORE_16dp128b4x_16b
{
  using SRegisters = uint32_t[8];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& src4, uint32_t const& src5, uint32_t const& src6, uint32_t const& src7,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x128b.x4.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3),
       "r"(src4), "r"(src5), "r"(src6), "r"(src7) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 8 times
struct SM100_TMEM_STORE_16dp128b8x
{
  using SRegisters = uint32_t[16];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x128b.x8.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 8 times, expand 16b write
struct SM100_TMEM_STORE_16dp128b8x_16b
{
  using SRegisters = uint32_t[16];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x128b.x8.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 16 times
struct SM100_TMEM_STORE_16dp128b16x
{
  using SRegisters = uint32_t[32];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x128b.x16.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 16 times, expand 16b write
struct SM100_TMEM_STORE_16dp128b16x_16b
{
  using SRegisters = uint32_t[32];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x128b.x16.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 32 times
struct SM100_TMEM_STORE_16dp128b32x
{
  using SRegisters = uint32_t[64];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& src32, uint32_t const& src33, uint32_t const& src34, uint32_t const& src35,
       uint32_t const& src36, uint32_t const& src37, uint32_t const& src38, uint32_t const& src39,
       uint32_t const& src40, uint32_t const& src41, uint32_t const& src42, uint32_t const& src43,
       uint32_t const& src44, uint32_t const& src45, uint32_t const& src46, uint32_t const& src47,
       uint32_t const& src48, uint32_t const& src49, uint32_t const& src50, uint32_t const& src51,
       uint32_t const& src52, uint32_t const& src53, uint32_t const& src54, uint32_t const& src55,
       uint32_t const& src56, uint32_t const& src57, uint32_t const& src58, uint32_t const& src59,
       uint32_t const& src60, uint32_t const& src61, uint32_t const& src62, uint32_t const& src63,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x128b.x32.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31),
       "r"(src32), "r"(src33), "r"(src34), "r"(src35),
       "r"(src36), "r"(src37), "r"(src38), "r"(src39),
       "r"(src40), "r"(src41), "r"(src42), "r"(src43),
       "r"(src44), "r"(src45), "r"(src46), "r"(src47),
       "r"(src48), "r"(src49), "r"(src50), "r"(src51),
       "r"(src52), "r"(src53), "r"(src54), "r"(src55),
       "r"(src56), "r"(src57), "r"(src58), "r"(src59),
       "r"(src60), "r"(src61), "r"(src62), "r"(src63) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 32 times, expand 16b write
struct SM100_TMEM_STORE_16dp128b32x_16b
{
  using SRegisters = uint32_t[64];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& src32, uint32_t const& src33, uint32_t const& src34, uint32_t const& src35,
       uint32_t const& src36, uint32_t const& src37, uint32_t const& src38, uint32_t const& src39,
       uint32_t const& src40, uint32_t const& src41, uint32_t const& src42, uint32_t const& src43,
       uint32_t const& src44, uint32_t const& src45, uint32_t const& src46, uint32_t const& src47,
       uint32_t const& src48, uint32_t const& src49, uint32_t const& src50, uint32_t const& src51,
       uint32_t const& src52, uint32_t const& src53, uint32_t const& src54, uint32_t const& src55,
       uint32_t const& src56, uint32_t const& src57, uint32_t const& src58, uint32_t const& src59,
       uint32_t const& src60, uint32_t const& src61, uint32_t const& src62, uint32_t const& src63,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x128b.x32.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31),
       "r"(src32), "r"(src33), "r"(src34), "r"(src35),
       "r"(src36), "r"(src37), "r"(src38), "r"(src39),
       "r"(src40), "r"(src41), "r"(src42), "r"(src43),
       "r"(src44), "r"(src45), "r"(src46), "r"(src47),
       "r"(src48), "r"(src49), "r"(src50), "r"(src51),
       "r"(src52), "r"(src53), "r"(src54), "r"(src55),
       "r"(src56), "r"(src57), "r"(src58), "r"(src59),
       "r"(src60), "r"(src61), "r"(src62), "r"(src63) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 64 times
struct SM100_TMEM_STORE_16dp128b64x
{
  using SRegisters = uint32_t[128];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src000, uint32_t const& src001, uint32_t const& src002, uint32_t const& src003,
       uint32_t const& src004, uint32_t const& src005, uint32_t const& src006, uint32_t const& src007,
       uint32_t const& src008, uint32_t const& src009, uint32_t const& src010, uint32_t const& src011,
       uint32_t const& src012, uint32_t const& src013, uint32_t const& src014, uint32_t const& src015,
       uint32_t const& src016, uint32_t const& src017, uint32_t const& src018, uint32_t const& src019,
       uint32_t const& src020, uint32_t const& src021, uint32_t const& src022, uint32_t const& src023,
       uint32_t const& src024, uint32_t const& src025, uint32_t const& src026, uint32_t const& src027,
       uint32_t const& src028, uint32_t const& src029, uint32_t const& src030, uint32_t const& src031,
       uint32_t const& src032, uint32_t const& src033, uint32_t const& src034, uint32_t const& src035,
       uint32_t const& src036, uint32_t const& src037, uint32_t const& src038, uint32_t const& src039,
       uint32_t const& src040, uint32_t const& src041, uint32_t const& src042, uint32_t const& src043,
       uint32_t const& src044, uint32_t const& src045, uint32_t const& src046, uint32_t const& src047,
       uint32_t const& src048, uint32_t const& src049, uint32_t const& src050, uint32_t const& src051,
       uint32_t const& src052, uint32_t const& src053, uint32_t const& src054, uint32_t const& src055,
       uint32_t const& src056, uint32_t const& src057, uint32_t const& src058, uint32_t const& src059,
       uint32_t const& src060, uint32_t const& src061, uint32_t const& src062, uint32_t const& src063,
       uint32_t const& src064, uint32_t const& src065, uint32_t const& src066, uint32_t const& src067,
       uint32_t const& src068, uint32_t const& src069, uint32_t const& src070, uint32_t const& src071,
       uint32_t const& src072, uint32_t const& src073, uint32_t const& src074, uint32_t const& src075,
       uint32_t const& src076, uint32_t const& src077, uint32_t const& src078, uint32_t const& src079,
       uint32_t const& src080, uint32_t const& src081, uint32_t const& src082, uint32_t const& src083,
       uint32_t const& src084, uint32_t const& src085, uint32_t const& src086, uint32_t const& src087,
       uint32_t const& src088, uint32_t const& src089, uint32_t const& src090, uint32_t const& src091,
       uint32_t const& src092, uint32_t const& src093, uint32_t const& src094, uint32_t const& src095,
       uint32_t const& src096, uint32_t const& src097, uint32_t const& src098, uint32_t const& src099,
       uint32_t const& src100, uint32_t const& src101, uint32_t const& src102, uint32_t const& src103,
       uint32_t const& src104, uint32_t const& src105, uint32_t const& src106, uint32_t const& src107,
       uint32_t const& src108, uint32_t const& src109, uint32_t const& src110, uint32_t const& src111,
       uint32_t const& src112, uint32_t const& src113, uint32_t const& src114, uint32_t const& src115,
       uint32_t const& src116, uint32_t const& src117, uint32_t const& src118, uint32_t const& src119,
       uint32_t const& src120, uint32_t const& src121, uint32_t const& src122, uint32_t const& src123,
       uint32_t const& src124, uint32_t const& src125, uint32_t const& src126, uint32_t const& src127,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x128b.x64.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64,"
                    "%65, %66, %67, %68,"
                    "%69, %70, %71, %72,"
                    "%73, %74, %75, %76,"
                    "%77, %78, %79, %80,"
                    "%81, %82, %83, %84,"
                    "%85, %86, %87, %88,"
                    "%89, %90, %91, %92,"
                    "%93, %94, %95, %96,"
                    "%97, %98, %99, %100,"
                    "%101, %102, %103, %104,"
                    "%105, %106, %107, %108,"
                    "%109, %110, %111, %112,"
                    "%113, %114, %115, %116,"
                    "%117, %118, %119, %120,"
                    "%121, %122, %123, %124,"
                    "%125, %126, %127, %128};\n"
    :
    :  "r"(dst_addr), "r"(src000), "r"(src001), "r"(src002), "r"(src003),
       "r"(src004), "r"(src005), "r"(src006), "r"(src007),
       "r"(src008), "r"(src009), "r"(src010), "r"(src011),
       "r"(src012), "r"(src013), "r"(src014), "r"(src015),
       "r"(src016), "r"(src017), "r"(src018), "r"(src019),
       "r"(src020), "r"(src021), "r"(src022), "r"(src023),
       "r"(src024), "r"(src025), "r"(src026), "r"(src027),
       "r"(src028), "r"(src029), "r"(src030), "r"(src031),
       "r"(src032), "r"(src033), "r"(src034), "r"(src035),
       "r"(src036), "r"(src037), "r"(src038), "r"(src039),
       "r"(src040), "r"(src041), "r"(src042), "r"(src043),
       "r"(src044), "r"(src045), "r"(src046), "r"(src047),
       "r"(src048), "r"(src049), "r"(src050), "r"(src051),
       "r"(src052), "r"(src053), "r"(src054), "r"(src055),
       "r"(src056), "r"(src057), "r"(src058), "r"(src059),
       "r"(src060), "r"(src061), "r"(src062), "r"(src063),
       "r"(src064), "r"(src065), "r"(src066), "r"(src067),
       "r"(src068), "r"(src069), "r"(src070), "r"(src071),
       "r"(src072), "r"(src073), "r"(src074), "r"(src075),
       "r"(src076), "r"(src077), "r"(src078), "r"(src079),
       "r"(src080), "r"(src081), "r"(src082), "r"(src083),
       "r"(src084), "r"(src085), "r"(src086), "r"(src087),
       "r"(src088), "r"(src089), "r"(src090), "r"(src091),
       "r"(src092), "r"(src093), "r"(src094), "r"(src095),
       "r"(src096), "r"(src097), "r"(src098), "r"(src099),
       "r"(src100), "r"(src101), "r"(src102), "r"(src103),
       "r"(src104), "r"(src105), "r"(src106), "r"(src107),
       "r"(src108), "r"(src109), "r"(src110), "r"(src111),
       "r"(src112), "r"(src113), "r"(src114), "r"(src115),
       "r"(src116), "r"(src117), "r"(src118), "r"(src119),
       "r"(src120), "r"(src121), "r"(src122), "r"(src123),
       "r"(src124), "r"(src125), "r"(src126), "r"(src127) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 128-bit pattern, repeated 64 times, expand 16b write
struct SM100_TMEM_STORE_16dp128b64x_16b
{
  using SRegisters = uint32_t[128];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src000, uint32_t const& src001, uint32_t const& src002, uint32_t const& src003,
       uint32_t const& src004, uint32_t const& src005, uint32_t const& src006, uint32_t const& src007,
       uint32_t const& src008, uint32_t const& src009, uint32_t const& src010, uint32_t const& src011,
       uint32_t const& src012, uint32_t const& src013, uint32_t const& src014, uint32_t const& src015,
       uint32_t const& src016, uint32_t const& src017, uint32_t const& src018, uint32_t const& src019,
       uint32_t const& src020, uint32_t const& src021, uint32_t const& src022, uint32_t const& src023,
       uint32_t const& src024, uint32_t const& src025, uint32_t const& src026, uint32_t const& src027,
       uint32_t const& src028, uint32_t const& src029, uint32_t const& src030, uint32_t const& src031,
       uint32_t const& src032, uint32_t const& src033, uint32_t const& src034, uint32_t const& src035,
       uint32_t const& src036, uint32_t const& src037, uint32_t const& src038, uint32_t const& src039,
       uint32_t const& src040, uint32_t const& src041, uint32_t const& src042, uint32_t const& src043,
       uint32_t const& src044, uint32_t const& src045, uint32_t const& src046, uint32_t const& src047,
       uint32_t const& src048, uint32_t const& src049, uint32_t const& src050, uint32_t const& src051,
       uint32_t const& src052, uint32_t const& src053, uint32_t const& src054, uint32_t const& src055,
       uint32_t const& src056, uint32_t const& src057, uint32_t const& src058, uint32_t const& src059,
       uint32_t const& src060, uint32_t const& src061, uint32_t const& src062, uint32_t const& src063,
       uint32_t const& src064, uint32_t const& src065, uint32_t const& src066, uint32_t const& src067,
       uint32_t const& src068, uint32_t const& src069, uint32_t const& src070, uint32_t const& src071,
       uint32_t const& src072, uint32_t const& src073, uint32_t const& src074, uint32_t const& src075,
       uint32_t const& src076, uint32_t const& src077, uint32_t const& src078, uint32_t const& src079,
       uint32_t const& src080, uint32_t const& src081, uint32_t const& src082, uint32_t const& src083,
       uint32_t const& src084, uint32_t const& src085, uint32_t const& src086, uint32_t const& src087,
       uint32_t const& src088, uint32_t const& src089, uint32_t const& src090, uint32_t const& src091,
       uint32_t const& src092, uint32_t const& src093, uint32_t const& src094, uint32_t const& src095,
       uint32_t const& src096, uint32_t const& src097, uint32_t const& src098, uint32_t const& src099,
       uint32_t const& src100, uint32_t const& src101, uint32_t const& src102, uint32_t const& src103,
       uint32_t const& src104, uint32_t const& src105, uint32_t const& src106, uint32_t const& src107,
       uint32_t const& src108, uint32_t const& src109, uint32_t const& src110, uint32_t const& src111,
       uint32_t const& src112, uint32_t const& src113, uint32_t const& src114, uint32_t const& src115,
       uint32_t const& src116, uint32_t const& src117, uint32_t const& src118, uint32_t const& src119,
       uint32_t const& src120, uint32_t const& src121, uint32_t const& src122, uint32_t const& src123,
       uint32_t const& src124, uint32_t const& src125, uint32_t const& src126, uint32_t const& src127,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x128b.x64.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64,"
                    "%65, %66, %67, %68,"
                    "%69, %70, %71, %72,"
                    "%73, %74, %75, %76,"
                    "%77, %78, %79, %80,"
                    "%81, %82, %83, %84,"
                    "%85, %86, %87, %88,"
                    "%89, %90, %91, %92,"
                    "%93, %94, %95, %96,"
                    "%97, %98, %99, %100,"
                    "%101, %102, %103, %104,"
                    "%105, %106, %107, %108,"
                    "%109, %110, %111, %112,"
                    "%113, %114, %115, %116,"
                    "%117, %118, %119, %120,"
                    "%121, %122, %123, %124,"
                    "%125, %126, %127, %128};\n"
    :
    :  "r"(dst_addr), "r"(src000), "r"(src001), "r"(src002), "r"(src003),
       "r"(src004), "r"(src005), "r"(src006), "r"(src007),
       "r"(src008), "r"(src009), "r"(src010), "r"(src011),
       "r"(src012), "r"(src013), "r"(src014), "r"(src015),
       "r"(src016), "r"(src017), "r"(src018), "r"(src019),
       "r"(src020), "r"(src021), "r"(src022), "r"(src023),
       "r"(src024), "r"(src025), "r"(src026), "r"(src027),
       "r"(src028), "r"(src029), "r"(src030), "r"(src031),
       "r"(src032), "r"(src033), "r"(src034), "r"(src035),
       "r"(src036), "r"(src037), "r"(src038), "r"(src039),
       "r"(src040), "r"(src041), "r"(src042), "r"(src043),
       "r"(src044), "r"(src045), "r"(src046), "r"(src047),
       "r"(src048), "r"(src049), "r"(src050), "r"(src051),
       "r"(src052), "r"(src053), "r"(src054), "r"(src055),
       "r"(src056), "r"(src057), "r"(src058), "r"(src059),
       "r"(src060), "r"(src061), "r"(src062), "r"(src063),
       "r"(src064), "r"(src065), "r"(src066), "r"(src067),
       "r"(src068), "r"(src069), "r"(src070), "r"(src071),
       "r"(src072), "r"(src073), "r"(src074), "r"(src075),
       "r"(src076), "r"(src077), "r"(src078), "r"(src079),
       "r"(src080), "r"(src081), "r"(src082), "r"(src083),
       "r"(src084), "r"(src085), "r"(src086), "r"(src087),
       "r"(src088), "r"(src089), "r"(src090), "r"(src091),
       "r"(src092), "r"(src093), "r"(src094), "r"(src095),
       "r"(src096), "r"(src097), "r"(src098), "r"(src099),
       "r"(src100), "r"(src101), "r"(src102), "r"(src103),
       "r"(src104), "r"(src105), "r"(src106), "r"(src107),
       "r"(src108), "r"(src109), "r"(src110), "r"(src111),
       "r"(src112), "r"(src113), "r"(src114), "r"(src115),
       "r"(src116), "r"(src117), "r"(src118), "r"(src119),
       "r"(src120), "r"(src121), "r"(src122), "r"(src123),
       "r"(src124), "r"(src125), "r"(src126), "r"(src127) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 1 times
struct SM100_TMEM_STORE_16dp64b1x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x64b.x1.b32"
                    "[%0],"
                    "{%1};\n"
    :
    :  "r"(dst_addr), "r"(src0) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 1 times, expand 16b write
struct SM100_TMEM_STORE_16dp64b1x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x64b.x1.unpack::16b.b32"
                    "[%0],"
                    "{%1};\n"
    :
    :  "r"(dst_addr), "r"(src0) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 2 times
struct SM100_TMEM_STORE_16dp64b2x
{
  using SRegisters = uint32_t[2];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x64b.x2.b32"
                    "[%0],"
                    "{%1, %2};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 2 times, expand 16b write
struct SM100_TMEM_STORE_16dp64b2x_16b
{
  using SRegisters = uint32_t[2];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x64b.x2.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 4 times
struct SM100_TMEM_STORE_16dp64b4x
{
  using SRegisters = uint32_t[4];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x64b.x4.b32"
                    "[%0],"
                    "{%1, %2, %3, %4};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 4 times, expand 16b write
struct SM100_TMEM_STORE_16dp64b4x_16b
{
  using SRegisters = uint32_t[4];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x64b.x4.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 8 times
struct SM100_TMEM_STORE_16dp64b8x
{
  using SRegisters = uint32_t[8];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& src4, uint32_t const& src5, uint32_t const& src6, uint32_t const& src7,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x64b.x8.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3),
       "r"(src4), "r"(src5), "r"(src6), "r"(src7) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 8 times, expand 16b write
struct SM100_TMEM_STORE_16dp64b8x_16b
{
  using SRegisters = uint32_t[8];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& src4, uint32_t const& src5, uint32_t const& src6, uint32_t const& src7,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x64b.x8.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3),
       "r"(src4), "r"(src5), "r"(src6), "r"(src7) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 16 times
struct SM100_TMEM_STORE_16dp64b16x
{
  using SRegisters = uint32_t[16];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x64b.x16.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 16 times, expand 16b write
struct SM100_TMEM_STORE_16dp64b16x_16b
{
  using SRegisters = uint32_t[16];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x64b.x16.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 32 times
struct SM100_TMEM_STORE_16dp64b32x
{
  using SRegisters = uint32_t[32];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x64b.x32.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 32 times, expand 16b write
struct SM100_TMEM_STORE_16dp64b32x_16b
{
  using SRegisters = uint32_t[32];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x64b.x32.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 64 times
struct SM100_TMEM_STORE_16dp64b64x
{
  using SRegisters = uint32_t[64];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& src32, uint32_t const& src33, uint32_t const& src34, uint32_t const& src35,
       uint32_t const& src36, uint32_t const& src37, uint32_t const& src38, uint32_t const& src39,
       uint32_t const& src40, uint32_t const& src41, uint32_t const& src42, uint32_t const& src43,
       uint32_t const& src44, uint32_t const& src45, uint32_t const& src46, uint32_t const& src47,
       uint32_t const& src48, uint32_t const& src49, uint32_t const& src50, uint32_t const& src51,
       uint32_t const& src52, uint32_t const& src53, uint32_t const& src54, uint32_t const& src55,
       uint32_t const& src56, uint32_t const& src57, uint32_t const& src58, uint32_t const& src59,
       uint32_t const& src60, uint32_t const& src61, uint32_t const& src62, uint32_t const& src63,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x64b.x64.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31),
       "r"(src32), "r"(src33), "r"(src34), "r"(src35),
       "r"(src36), "r"(src37), "r"(src38), "r"(src39),
       "r"(src40), "r"(src41), "r"(src42), "r"(src43),
       "r"(src44), "r"(src45), "r"(src46), "r"(src47),
       "r"(src48), "r"(src49), "r"(src50), "r"(src51),
       "r"(src52), "r"(src53), "r"(src54), "r"(src55),
       "r"(src56), "r"(src57), "r"(src58), "r"(src59),
       "r"(src60), "r"(src61), "r"(src62), "r"(src63) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 64 times, expand 16b write
struct SM100_TMEM_STORE_16dp64b64x_16b
{
  using SRegisters = uint32_t[64];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& src32, uint32_t const& src33, uint32_t const& src34, uint32_t const& src35,
       uint32_t const& src36, uint32_t const& src37, uint32_t const& src38, uint32_t const& src39,
       uint32_t const& src40, uint32_t const& src41, uint32_t const& src42, uint32_t const& src43,
       uint32_t const& src44, uint32_t const& src45, uint32_t const& src46, uint32_t const& src47,
       uint32_t const& src48, uint32_t const& src49, uint32_t const& src50, uint32_t const& src51,
       uint32_t const& src52, uint32_t const& src53, uint32_t const& src54, uint32_t const& src55,
       uint32_t const& src56, uint32_t const& src57, uint32_t const& src58, uint32_t const& src59,
       uint32_t const& src60, uint32_t const& src61, uint32_t const& src62, uint32_t const& src63,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x64b.x64.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31),
       "r"(src32), "r"(src33), "r"(src34), "r"(src35),
       "r"(src36), "r"(src37), "r"(src38), "r"(src39),
       "r"(src40), "r"(src41), "r"(src42), "r"(src43),
       "r"(src44), "r"(src45), "r"(src46), "r"(src47),
       "r"(src48), "r"(src49), "r"(src50), "r"(src51),
       "r"(src52), "r"(src53), "r"(src54), "r"(src55),
       "r"(src56), "r"(src57), "r"(src58), "r"(src59),
       "r"(src60), "r"(src61), "r"(src62), "r"(src63) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 128 times
struct SM100_TMEM_STORE_16dp64b128x
{
  using SRegisters = uint32_t[128];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src000, uint32_t const& src001, uint32_t const& src002, uint32_t const& src003,
       uint32_t const& src004, uint32_t const& src005, uint32_t const& src006, uint32_t const& src007,
       uint32_t const& src008, uint32_t const& src009, uint32_t const& src010, uint32_t const& src011,
       uint32_t const& src012, uint32_t const& src013, uint32_t const& src014, uint32_t const& src015,
       uint32_t const& src016, uint32_t const& src017, uint32_t const& src018, uint32_t const& src019,
       uint32_t const& src020, uint32_t const& src021, uint32_t const& src022, uint32_t const& src023,
       uint32_t const& src024, uint32_t const& src025, uint32_t const& src026, uint32_t const& src027,
       uint32_t const& src028, uint32_t const& src029, uint32_t const& src030, uint32_t const& src031,
       uint32_t const& src032, uint32_t const& src033, uint32_t const& src034, uint32_t const& src035,
       uint32_t const& src036, uint32_t const& src037, uint32_t const& src038, uint32_t const& src039,
       uint32_t const& src040, uint32_t const& src041, uint32_t const& src042, uint32_t const& src043,
       uint32_t const& src044, uint32_t const& src045, uint32_t const& src046, uint32_t const& src047,
       uint32_t const& src048, uint32_t const& src049, uint32_t const& src050, uint32_t const& src051,
       uint32_t const& src052, uint32_t const& src053, uint32_t const& src054, uint32_t const& src055,
       uint32_t const& src056, uint32_t const& src057, uint32_t const& src058, uint32_t const& src059,
       uint32_t const& src060, uint32_t const& src061, uint32_t const& src062, uint32_t const& src063,
       uint32_t const& src064, uint32_t const& src065, uint32_t const& src066, uint32_t const& src067,
       uint32_t const& src068, uint32_t const& src069, uint32_t const& src070, uint32_t const& src071,
       uint32_t const& src072, uint32_t const& src073, uint32_t const& src074, uint32_t const& src075,
       uint32_t const& src076, uint32_t const& src077, uint32_t const& src078, uint32_t const& src079,
       uint32_t const& src080, uint32_t const& src081, uint32_t const& src082, uint32_t const& src083,
       uint32_t const& src084, uint32_t const& src085, uint32_t const& src086, uint32_t const& src087,
       uint32_t const& src088, uint32_t const& src089, uint32_t const& src090, uint32_t const& src091,
       uint32_t const& src092, uint32_t const& src093, uint32_t const& src094, uint32_t const& src095,
       uint32_t const& src096, uint32_t const& src097, uint32_t const& src098, uint32_t const& src099,
       uint32_t const& src100, uint32_t const& src101, uint32_t const& src102, uint32_t const& src103,
       uint32_t const& src104, uint32_t const& src105, uint32_t const& src106, uint32_t const& src107,
       uint32_t const& src108, uint32_t const& src109, uint32_t const& src110, uint32_t const& src111,
       uint32_t const& src112, uint32_t const& src113, uint32_t const& src114, uint32_t const& src115,
       uint32_t const& src116, uint32_t const& src117, uint32_t const& src118, uint32_t const& src119,
       uint32_t const& src120, uint32_t const& src121, uint32_t const& src122, uint32_t const& src123,
       uint32_t const& src124, uint32_t const& src125, uint32_t const& src126, uint32_t const& src127,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x64b.x128.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64,"
                    "%65, %66, %67, %68,"
                    "%69, %70, %71, %72,"
                    "%73, %74, %75, %76,"
                    "%77, %78, %79, %80,"
                    "%81, %82, %83, %84,"
                    "%85, %86, %87, %88,"
                    "%89, %90, %91, %92,"
                    "%93, %94, %95, %96,"
                    "%97, %98, %99, %100,"
                    "%101, %102, %103, %104,"
                    "%105, %106, %107, %108,"
                    "%109, %110, %111, %112,"
                    "%113, %114, %115, %116,"
                    "%117, %118, %119, %120,"
                    "%121, %122, %123, %124,"
                    "%125, %126, %127, %128};\n"
    :
    :  "r"(dst_addr), "r"(src000), "r"(src001), "r"(src002), "r"(src003),
       "r"(src004), "r"(src005), "r"(src006), "r"(src007),
       "r"(src008), "r"(src009), "r"(src010), "r"(src011),
       "r"(src012), "r"(src013), "r"(src014), "r"(src015),
       "r"(src016), "r"(src017), "r"(src018), "r"(src019),
       "r"(src020), "r"(src021), "r"(src022), "r"(src023),
       "r"(src024), "r"(src025), "r"(src026), "r"(src027),
       "r"(src028), "r"(src029), "r"(src030), "r"(src031),
       "r"(src032), "r"(src033), "r"(src034), "r"(src035),
       "r"(src036), "r"(src037), "r"(src038), "r"(src039),
       "r"(src040), "r"(src041), "r"(src042), "r"(src043),
       "r"(src044), "r"(src045), "r"(src046), "r"(src047),
       "r"(src048), "r"(src049), "r"(src050), "r"(src051),
       "r"(src052), "r"(src053), "r"(src054), "r"(src055),
       "r"(src056), "r"(src057), "r"(src058), "r"(src059),
       "r"(src060), "r"(src061), "r"(src062), "r"(src063),
       "r"(src064), "r"(src065), "r"(src066), "r"(src067),
       "r"(src068), "r"(src069), "r"(src070), "r"(src071),
       "r"(src072), "r"(src073), "r"(src074), "r"(src075),
       "r"(src076), "r"(src077), "r"(src078), "r"(src079),
       "r"(src080), "r"(src081), "r"(src082), "r"(src083),
       "r"(src084), "r"(src085), "r"(src086), "r"(src087),
       "r"(src088), "r"(src089), "r"(src090), "r"(src091),
       "r"(src092), "r"(src093), "r"(src094), "r"(src095),
       "r"(src096), "r"(src097), "r"(src098), "r"(src099),
       "r"(src100), "r"(src101), "r"(src102), "r"(src103),
       "r"(src104), "r"(src105), "r"(src106), "r"(src107),
       "r"(src108), "r"(src109), "r"(src110), "r"(src111),
       "r"(src112), "r"(src113), "r"(src114), "r"(src115),
       "r"(src116), "r"(src117), "r"(src118), "r"(src119),
       "r"(src120), "r"(src121), "r"(src122), "r"(src123),
       "r"(src124), "r"(src125), "r"(src126), "r"(src127) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 64-bit pattern, repeated 128 times, expand 16b write
struct SM100_TMEM_STORE_16dp64b128x_16b
{
  using SRegisters = uint32_t[128];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src000, uint32_t const& src001, uint32_t const& src002, uint32_t const& src003,
       uint32_t const& src004, uint32_t const& src005, uint32_t const& src006, uint32_t const& src007,
       uint32_t const& src008, uint32_t const& src009, uint32_t const& src010, uint32_t const& src011,
       uint32_t const& src012, uint32_t const& src013, uint32_t const& src014, uint32_t const& src015,
       uint32_t const& src016, uint32_t const& src017, uint32_t const& src018, uint32_t const& src019,
       uint32_t const& src020, uint32_t const& src021, uint32_t const& src022, uint32_t const& src023,
       uint32_t const& src024, uint32_t const& src025, uint32_t const& src026, uint32_t const& src027,
       uint32_t const& src028, uint32_t const& src029, uint32_t const& src030, uint32_t const& src031,
       uint32_t const& src032, uint32_t const& src033, uint32_t const& src034, uint32_t const& src035,
       uint32_t const& src036, uint32_t const& src037, uint32_t const& src038, uint32_t const& src039,
       uint32_t const& src040, uint32_t const& src041, uint32_t const& src042, uint32_t const& src043,
       uint32_t const& src044, uint32_t const& src045, uint32_t const& src046, uint32_t const& src047,
       uint32_t const& src048, uint32_t const& src049, uint32_t const& src050, uint32_t const& src051,
       uint32_t const& src052, uint32_t const& src053, uint32_t const& src054, uint32_t const& src055,
       uint32_t const& src056, uint32_t const& src057, uint32_t const& src058, uint32_t const& src059,
       uint32_t const& src060, uint32_t const& src061, uint32_t const& src062, uint32_t const& src063,
       uint32_t const& src064, uint32_t const& src065, uint32_t const& src066, uint32_t const& src067,
       uint32_t const& src068, uint32_t const& src069, uint32_t const& src070, uint32_t const& src071,
       uint32_t const& src072, uint32_t const& src073, uint32_t const& src074, uint32_t const& src075,
       uint32_t const& src076, uint32_t const& src077, uint32_t const& src078, uint32_t const& src079,
       uint32_t const& src080, uint32_t const& src081, uint32_t const& src082, uint32_t const& src083,
       uint32_t const& src084, uint32_t const& src085, uint32_t const& src086, uint32_t const& src087,
       uint32_t const& src088, uint32_t const& src089, uint32_t const& src090, uint32_t const& src091,
       uint32_t const& src092, uint32_t const& src093, uint32_t const& src094, uint32_t const& src095,
       uint32_t const& src096, uint32_t const& src097, uint32_t const& src098, uint32_t const& src099,
       uint32_t const& src100, uint32_t const& src101, uint32_t const& src102, uint32_t const& src103,
       uint32_t const& src104, uint32_t const& src105, uint32_t const& src106, uint32_t const& src107,
       uint32_t const& src108, uint32_t const& src109, uint32_t const& src110, uint32_t const& src111,
       uint32_t const& src112, uint32_t const& src113, uint32_t const& src114, uint32_t const& src115,
       uint32_t const& src116, uint32_t const& src117, uint32_t const& src118, uint32_t const& src119,
       uint32_t const& src120, uint32_t const& src121, uint32_t const& src122, uint32_t const& src123,
       uint32_t const& src124, uint32_t const& src125, uint32_t const& src126, uint32_t const& src127,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x64b.x128.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64,"
                    "%65, %66, %67, %68,"
                    "%69, %70, %71, %72,"
                    "%73, %74, %75, %76,"
                    "%77, %78, %79, %80,"
                    "%81, %82, %83, %84,"
                    "%85, %86, %87, %88,"
                    "%89, %90, %91, %92,"
                    "%93, %94, %95, %96,"
                    "%97, %98, %99, %100,"
                    "%101, %102, %103, %104,"
                    "%105, %106, %107, %108,"
                    "%109, %110, %111, %112,"
                    "%113, %114, %115, %116,"
                    "%117, %118, %119, %120,"
                    "%121, %122, %123, %124,"
                    "%125, %126, %127, %128};\n"
    :
    :  "r"(dst_addr), "r"(src000), "r"(src001), "r"(src002), "r"(src003),
       "r"(src004), "r"(src005), "r"(src006), "r"(src007),
       "r"(src008), "r"(src009), "r"(src010), "r"(src011),
       "r"(src012), "r"(src013), "r"(src014), "r"(src015),
       "r"(src016), "r"(src017), "r"(src018), "r"(src019),
       "r"(src020), "r"(src021), "r"(src022), "r"(src023),
       "r"(src024), "r"(src025), "r"(src026), "r"(src027),
       "r"(src028), "r"(src029), "r"(src030), "r"(src031),
       "r"(src032), "r"(src033), "r"(src034), "r"(src035),
       "r"(src036), "r"(src037), "r"(src038), "r"(src039),
       "r"(src040), "r"(src041), "r"(src042), "r"(src043),
       "r"(src044), "r"(src045), "r"(src046), "r"(src047),
       "r"(src048), "r"(src049), "r"(src050), "r"(src051),
       "r"(src052), "r"(src053), "r"(src054), "r"(src055),
       "r"(src056), "r"(src057), "r"(src058), "r"(src059),
       "r"(src060), "r"(src061), "r"(src062), "r"(src063),
       "r"(src064), "r"(src065), "r"(src066), "r"(src067),
       "r"(src068), "r"(src069), "r"(src070), "r"(src071),
       "r"(src072), "r"(src073), "r"(src074), "r"(src075),
       "r"(src076), "r"(src077), "r"(src078), "r"(src079),
       "r"(src080), "r"(src081), "r"(src082), "r"(src083),
       "r"(src084), "r"(src085), "r"(src086), "r"(src087),
       "r"(src088), "r"(src089), "r"(src090), "r"(src091),
       "r"(src092), "r"(src093), "r"(src094), "r"(src095),
       "r"(src096), "r"(src097), "r"(src098), "r"(src099),
       "r"(src100), "r"(src101), "r"(src102), "r"(src103),
       "r"(src104), "r"(src105), "r"(src106), "r"(src107),
       "r"(src108), "r"(src109), "r"(src110), "r"(src111),
       "r"(src112), "r"(src113), "r"(src114), "r"(src115),
       "r"(src116), "r"(src117), "r"(src118), "r"(src119),
       "r"(src120), "r"(src121), "r"(src122), "r"(src123),
       "r"(src124), "r"(src125), "r"(src126), "r"(src127) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 1 times
struct SM100_TMEM_STORE_16dp32b1x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x32bx2.x1.b32"
                    "[%0] , 1,"
                    "{%1};\n"
    :
    :  "r"(dst_addr), "r"(src0) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 1 times, expand 16b write
struct SM100_TMEM_STORE_16dp32b1x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x32bx2.x1.unpack::16b.b32"
                    "[%0] , 2,"
                    "{%1};\n"
    :
    :  "r"(dst_addr), "r"(src0) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 2 times
struct SM100_TMEM_STORE_16dp32b2x
{
  using SRegisters = uint32_t[2];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x32bx2.x2.b32"
                    "[%0] , 2,"
                    "{%1, %2};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 2 times, expand 16b write
struct SM100_TMEM_STORE_16dp32b2x_16b
{
  using SRegisters = uint32_t[2];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x32bx2.x2.unpack::16b.b32"
                    "[%0] , 4,"
                    "{%1, %2};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 4 times
struct SM100_TMEM_STORE_16dp32b4x
{
  using SRegisters = uint32_t[4];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x32bx2.x4.b32"
                    "[%0] , 4,"
                    "{%1, %2, %3, %4};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 4 times, expand 16b write
struct SM100_TMEM_STORE_16dp32b4x_16b
{
  using SRegisters = uint32_t[4];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x32bx2.x4.unpack::16b.b32"
                    "[%0] , 8,"
                    "{%1, %2, %3, %4};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 8 times
struct SM100_TMEM_STORE_16dp32b8x
{
  using SRegisters = uint32_t[8];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& src4, uint32_t const& src5, uint32_t const& src6, uint32_t const& src7,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                    "[%0] , 8,"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3),
       "r"(src4), "r"(src5), "r"(src6), "r"(src7) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 8 times, expand 16b write
struct SM100_TMEM_STORE_16dp32b8x_16b
{
  using SRegisters = uint32_t[8];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& src4, uint32_t const& src5, uint32_t const& src6, uint32_t const& src7,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x32bx2.x8.unpack::16b.b32"
                    "[%0] , 16,"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3),
       "r"(src4), "r"(src5), "r"(src6), "r"(src7) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 16 times
struct SM100_TMEM_STORE_16dp32b16x
{
  using SRegisters = uint32_t[16];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                    "[%0] , 16,"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 16 times, expand 16b write
struct SM100_TMEM_STORE_16dp32b16x_16b
{
  using SRegisters = uint32_t[16];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x32bx2.x16.unpack::16b.b32"
                    "[%0] , 32,"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 32 times
struct SM100_TMEM_STORE_16dp32b32x
{
  using SRegisters = uint32_t[32];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x32bx2.x32.b32"
                    "[%0] , 32,"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 32 times, expand 16b write
struct SM100_TMEM_STORE_16dp32b32x_16b
{
  using SRegisters = uint32_t[32];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x32bx2.x32.unpack::16b.b32"
                    "[%0] , 64,"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 64 times
struct SM100_TMEM_STORE_16dp32b64x
{
  using SRegisters = uint32_t[64];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& src32, uint32_t const& src33, uint32_t const& src34, uint32_t const& src35,
       uint32_t const& src36, uint32_t const& src37, uint32_t const& src38, uint32_t const& src39,
       uint32_t const& src40, uint32_t const& src41, uint32_t const& src42, uint32_t const& src43,
       uint32_t const& src44, uint32_t const& src45, uint32_t const& src46, uint32_t const& src47,
       uint32_t const& src48, uint32_t const& src49, uint32_t const& src50, uint32_t const& src51,
       uint32_t const& src52, uint32_t const& src53, uint32_t const& src54, uint32_t const& src55,
       uint32_t const& src56, uint32_t const& src57, uint32_t const& src58, uint32_t const& src59,
       uint32_t const& src60, uint32_t const& src61, uint32_t const& src62, uint32_t const& src63,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x32bx2.x64.b32"
                    "[%0] , 64,"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31),
       "r"(src32), "r"(src33), "r"(src34), "r"(src35),
       "r"(src36), "r"(src37), "r"(src38), "r"(src39),
       "r"(src40), "r"(src41), "r"(src42), "r"(src43),
       "r"(src44), "r"(src45), "r"(src46), "r"(src47),
       "r"(src48), "r"(src49), "r"(src50), "r"(src51),
       "r"(src52), "r"(src53), "r"(src54), "r"(src55),
       "r"(src56), "r"(src57), "r"(src58), "r"(src59),
       "r"(src60), "r"(src61), "r"(src62), "r"(src63) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 64 times, expand 16b write
struct SM100_TMEM_STORE_16dp32b64x_16b
{
  using SRegisters = uint32_t[64];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& src32, uint32_t const& src33, uint32_t const& src34, uint32_t const& src35,
       uint32_t const& src36, uint32_t const& src37, uint32_t const& src38, uint32_t const& src39,
       uint32_t const& src40, uint32_t const& src41, uint32_t const& src42, uint32_t const& src43,
       uint32_t const& src44, uint32_t const& src45, uint32_t const& src46, uint32_t const& src47,
       uint32_t const& src48, uint32_t const& src49, uint32_t const& src50, uint32_t const& src51,
       uint32_t const& src52, uint32_t const& src53, uint32_t const& src54, uint32_t const& src55,
       uint32_t const& src56, uint32_t const& src57, uint32_t const& src58, uint32_t const& src59,
       uint32_t const& src60, uint32_t const& src61, uint32_t const& src62, uint32_t const& src63,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x32bx2.x64.unpack::16b.b32"
                    "[%0] , 128,"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31),
       "r"(src32), "r"(src33), "r"(src34), "r"(src35),
       "r"(src36), "r"(src37), "r"(src38), "r"(src39),
       "r"(src40), "r"(src41), "r"(src42), "r"(src43),
       "r"(src44), "r"(src45), "r"(src46), "r"(src47),
       "r"(src48), "r"(src49), "r"(src50), "r"(src51),
       "r"(src52), "r"(src53), "r"(src54), "r"(src55),
       "r"(src56), "r"(src57), "r"(src58), "r"(src59),
       "r"(src60), "r"(src61), "r"(src62), "r"(src63) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 128 times
struct SM100_TMEM_STORE_16dp32b128x
{
  using SRegisters = uint32_t[128];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src000, uint32_t const& src001, uint32_t const& src002, uint32_t const& src003,
       uint32_t const& src004, uint32_t const& src005, uint32_t const& src006, uint32_t const& src007,
       uint32_t const& src008, uint32_t const& src009, uint32_t const& src010, uint32_t const& src011,
       uint32_t const& src012, uint32_t const& src013, uint32_t const& src014, uint32_t const& src015,
       uint32_t const& src016, uint32_t const& src017, uint32_t const& src018, uint32_t const& src019,
       uint32_t const& src020, uint32_t const& src021, uint32_t const& src022, uint32_t const& src023,
       uint32_t const& src024, uint32_t const& src025, uint32_t const& src026, uint32_t const& src027,
       uint32_t const& src028, uint32_t const& src029, uint32_t const& src030, uint32_t const& src031,
       uint32_t const& src032, uint32_t const& src033, uint32_t const& src034, uint32_t const& src035,
       uint32_t const& src036, uint32_t const& src037, uint32_t const& src038, uint32_t const& src039,
       uint32_t const& src040, uint32_t const& src041, uint32_t const& src042, uint32_t const& src043,
       uint32_t const& src044, uint32_t const& src045, uint32_t const& src046, uint32_t const& src047,
       uint32_t const& src048, uint32_t const& src049, uint32_t const& src050, uint32_t const& src051,
       uint32_t const& src052, uint32_t const& src053, uint32_t const& src054, uint32_t const& src055,
       uint32_t const& src056, uint32_t const& src057, uint32_t const& src058, uint32_t const& src059,
       uint32_t const& src060, uint32_t const& src061, uint32_t const& src062, uint32_t const& src063,
       uint32_t const& src064, uint32_t const& src065, uint32_t const& src066, uint32_t const& src067,
       uint32_t const& src068, uint32_t const& src069, uint32_t const& src070, uint32_t const& src071,
       uint32_t const& src072, uint32_t const& src073, uint32_t const& src074, uint32_t const& src075,
       uint32_t const& src076, uint32_t const& src077, uint32_t const& src078, uint32_t const& src079,
       uint32_t const& src080, uint32_t const& src081, uint32_t const& src082, uint32_t const& src083,
       uint32_t const& src084, uint32_t const& src085, uint32_t const& src086, uint32_t const& src087,
       uint32_t const& src088, uint32_t const& src089, uint32_t const& src090, uint32_t const& src091,
       uint32_t const& src092, uint32_t const& src093, uint32_t const& src094, uint32_t const& src095,
       uint32_t const& src096, uint32_t const& src097, uint32_t const& src098, uint32_t const& src099,
       uint32_t const& src100, uint32_t const& src101, uint32_t const& src102, uint32_t const& src103,
       uint32_t const& src104, uint32_t const& src105, uint32_t const& src106, uint32_t const& src107,
       uint32_t const& src108, uint32_t const& src109, uint32_t const& src110, uint32_t const& src111,
       uint32_t const& src112, uint32_t const& src113, uint32_t const& src114, uint32_t const& src115,
       uint32_t const& src116, uint32_t const& src117, uint32_t const& src118, uint32_t const& src119,
       uint32_t const& src120, uint32_t const& src121, uint32_t const& src122, uint32_t const& src123,
       uint32_t const& src124, uint32_t const& src125, uint32_t const& src126, uint32_t const& src127,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x32bx2.x128.b32"
                    "[%0] , 128,"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64,"
                    "%65, %66, %67, %68,"
                    "%69, %70, %71, %72,"
                    "%73, %74, %75, %76,"
                    "%77, %78, %79, %80,"
                    "%81, %82, %83, %84,"
                    "%85, %86, %87, %88,"
                    "%89, %90, %91, %92,"
                    "%93, %94, %95, %96,"
                    "%97, %98, %99, %100,"
                    "%101, %102, %103, %104,"
                    "%105, %106, %107, %108,"
                    "%109, %110, %111, %112,"
                    "%113, %114, %115, %116,"
                    "%117, %118, %119, %120,"
                    "%121, %122, %123, %124,"
                    "%125, %126, %127, %128};\n"
    :
    :  "r"(dst_addr), "r"(src000), "r"(src001), "r"(src002), "r"(src003),
       "r"(src004), "r"(src005), "r"(src006), "r"(src007),
       "r"(src008), "r"(src009), "r"(src010), "r"(src011),
       "r"(src012), "r"(src013), "r"(src014), "r"(src015),
       "r"(src016), "r"(src017), "r"(src018), "r"(src019),
       "r"(src020), "r"(src021), "r"(src022), "r"(src023),
       "r"(src024), "r"(src025), "r"(src026), "r"(src027),
       "r"(src028), "r"(src029), "r"(src030), "r"(src031),
       "r"(src032), "r"(src033), "r"(src034), "r"(src035),
       "r"(src036), "r"(src037), "r"(src038), "r"(src039),
       "r"(src040), "r"(src041), "r"(src042), "r"(src043),
       "r"(src044), "r"(src045), "r"(src046), "r"(src047),
       "r"(src048), "r"(src049), "r"(src050), "r"(src051),
       "r"(src052), "r"(src053), "r"(src054), "r"(src055),
       "r"(src056), "r"(src057), "r"(src058), "r"(src059),
       "r"(src060), "r"(src061), "r"(src062), "r"(src063),
       "r"(src064), "r"(src065), "r"(src066), "r"(src067),
       "r"(src068), "r"(src069), "r"(src070), "r"(src071),
       "r"(src072), "r"(src073), "r"(src074), "r"(src075),
       "r"(src076), "r"(src077), "r"(src078), "r"(src079),
       "r"(src080), "r"(src081), "r"(src082), "r"(src083),
       "r"(src084), "r"(src085), "r"(src086), "r"(src087),
       "r"(src088), "r"(src089), "r"(src090), "r"(src091),
       "r"(src092), "r"(src093), "r"(src094), "r"(src095),
       "r"(src096), "r"(src097), "r"(src098), "r"(src099),
       "r"(src100), "r"(src101), "r"(src102), "r"(src103),
       "r"(src104), "r"(src105), "r"(src106), "r"(src107),
       "r"(src108), "r"(src109), "r"(src110), "r"(src111),
       "r"(src112), "r"(src113), "r"(src114), "r"(src115),
       "r"(src116), "r"(src117), "r"(src118), "r"(src119),
       "r"(src120), "r"(src121), "r"(src122), "r"(src123),
       "r"(src124), "r"(src125), "r"(src126), "r"(src127) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 16 data path lanes, 32-bit pattern, repeated 128 times, expand 16b write
struct SM100_TMEM_STORE_16dp32b128x_16b
{
  using SRegisters = uint32_t[128];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src000, uint32_t const& src001, uint32_t const& src002, uint32_t const& src003,
       uint32_t const& src004, uint32_t const& src005, uint32_t const& src006, uint32_t const& src007,
       uint32_t const& src008, uint32_t const& src009, uint32_t const& src010, uint32_t const& src011,
       uint32_t const& src012, uint32_t const& src013, uint32_t const& src014, uint32_t const& src015,
       uint32_t const& src016, uint32_t const& src017, uint32_t const& src018, uint32_t const& src019,
       uint32_t const& src020, uint32_t const& src021, uint32_t const& src022, uint32_t const& src023,
       uint32_t const& src024, uint32_t const& src025, uint32_t const& src026, uint32_t const& src027,
       uint32_t const& src028, uint32_t const& src029, uint32_t const& src030, uint32_t const& src031,
       uint32_t const& src032, uint32_t const& src033, uint32_t const& src034, uint32_t const& src035,
       uint32_t const& src036, uint32_t const& src037, uint32_t const& src038, uint32_t const& src039,
       uint32_t const& src040, uint32_t const& src041, uint32_t const& src042, uint32_t const& src043,
       uint32_t const& src044, uint32_t const& src045, uint32_t const& src046, uint32_t const& src047,
       uint32_t const& src048, uint32_t const& src049, uint32_t const& src050, uint32_t const& src051,
       uint32_t const& src052, uint32_t const& src053, uint32_t const& src054, uint32_t const& src055,
       uint32_t const& src056, uint32_t const& src057, uint32_t const& src058, uint32_t const& src059,
       uint32_t const& src060, uint32_t const& src061, uint32_t const& src062, uint32_t const& src063,
       uint32_t const& src064, uint32_t const& src065, uint32_t const& src066, uint32_t const& src067,
       uint32_t const& src068, uint32_t const& src069, uint32_t const& src070, uint32_t const& src071,
       uint32_t const& src072, uint32_t const& src073, uint32_t const& src074, uint32_t const& src075,
       uint32_t const& src076, uint32_t const& src077, uint32_t const& src078, uint32_t const& src079,
       uint32_t const& src080, uint32_t const& src081, uint32_t const& src082, uint32_t const& src083,
       uint32_t const& src084, uint32_t const& src085, uint32_t const& src086, uint32_t const& src087,
       uint32_t const& src088, uint32_t const& src089, uint32_t const& src090, uint32_t const& src091,
       uint32_t const& src092, uint32_t const& src093, uint32_t const& src094, uint32_t const& src095,
       uint32_t const& src096, uint32_t const& src097, uint32_t const& src098, uint32_t const& src099,
       uint32_t const& src100, uint32_t const& src101, uint32_t const& src102, uint32_t const& src103,
       uint32_t const& src104, uint32_t const& src105, uint32_t const& src106, uint32_t const& src107,
       uint32_t const& src108, uint32_t const& src109, uint32_t const& src110, uint32_t const& src111,
       uint32_t const& src112, uint32_t const& src113, uint32_t const& src114, uint32_t const& src115,
       uint32_t const& src116, uint32_t const& src117, uint32_t const& src118, uint32_t const& src119,
       uint32_t const& src120, uint32_t const& src121, uint32_t const& src122, uint32_t const& src123,
       uint32_t const& src124, uint32_t const& src125, uint32_t const& src126, uint32_t const& src127,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.16x32bx2.x128.unpack::16b.b32"
                    "[%0] , 256,"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64,"
                    "%65, %66, %67, %68,"
                    "%69, %70, %71, %72,"
                    "%73, %74, %75, %76,"
                    "%77, %78, %79, %80,"
                    "%81, %82, %83, %84,"
                    "%85, %86, %87, %88,"
                    "%89, %90, %91, %92,"
                    "%93, %94, %95, %96,"
                    "%97, %98, %99, %100,"
                    "%101, %102, %103, %104,"
                    "%105, %106, %107, %108,"
                    "%109, %110, %111, %112,"
                    "%113, %114, %115, %116,"
                    "%117, %118, %119, %120,"
                    "%121, %122, %123, %124,"
                    "%125, %126, %127, %128};\n"
    :
    :  "r"(dst_addr), "r"(src000), "r"(src001), "r"(src002), "r"(src003),
       "r"(src004), "r"(src005), "r"(src006), "r"(src007),
       "r"(src008), "r"(src009), "r"(src010), "r"(src011),
       "r"(src012), "r"(src013), "r"(src014), "r"(src015),
       "r"(src016), "r"(src017), "r"(src018), "r"(src019),
       "r"(src020), "r"(src021), "r"(src022), "r"(src023),
       "r"(src024), "r"(src025), "r"(src026), "r"(src027),
       "r"(src028), "r"(src029), "r"(src030), "r"(src031),
       "r"(src032), "r"(src033), "r"(src034), "r"(src035),
       "r"(src036), "r"(src037), "r"(src038), "r"(src039),
       "r"(src040), "r"(src041), "r"(src042), "r"(src043),
       "r"(src044), "r"(src045), "r"(src046), "r"(src047),
       "r"(src048), "r"(src049), "r"(src050), "r"(src051),
       "r"(src052), "r"(src053), "r"(src054), "r"(src055),
       "r"(src056), "r"(src057), "r"(src058), "r"(src059),
       "r"(src060), "r"(src061), "r"(src062), "r"(src063),
       "r"(src064), "r"(src065), "r"(src066), "r"(src067),
       "r"(src068), "r"(src069), "r"(src070), "r"(src071),
       "r"(src072), "r"(src073), "r"(src074), "r"(src075),
       "r"(src076), "r"(src077), "r"(src078), "r"(src079),
       "r"(src080), "r"(src081), "r"(src082), "r"(src083),
       "r"(src084), "r"(src085), "r"(src086), "r"(src087),
       "r"(src088), "r"(src089), "r"(src090), "r"(src091),
       "r"(src092), "r"(src093), "r"(src094), "r"(src095),
       "r"(src096), "r"(src097), "r"(src098), "r"(src099),
       "r"(src100), "r"(src101), "r"(src102), "r"(src103),
       "r"(src104), "r"(src105), "r"(src106), "r"(src107),
       "r"(src108), "r"(src109), "r"(src110), "r"(src111),
       "r"(src112), "r"(src113), "r"(src114), "r"(src115),
       "r"(src116), "r"(src117), "r"(src118), "r"(src119),
       "r"(src120), "r"(src121), "r"(src122), "r"(src123),
       "r"(src124), "r"(src125), "r"(src126), "r"(src127) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 1 times
struct SM100_TMEM_STORE_32dp32b1x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.32x32b.x1.b32"
                    "[%0],"
                    "{%1};\n"
    :
    :  "r"(dst_addr), "r"(src0) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 1 times, expand 16b write
struct SM100_TMEM_STORE_32dp32b1x_16b
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.32x32b.x1.unpack::16b.b32"
                    "[%0],"
                    "{%1};\n"
    :
    :  "r"(dst_addr), "r"(src0) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 2 times
struct SM100_TMEM_STORE_32dp32b2x
{
  using SRegisters = uint32_t[2];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.32x32b.x2.b32"
                    "[%0],"
                    "{%1, %2};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 2 times, expand 16b write
struct SM100_TMEM_STORE_32dp32b2x_16b
{
  using SRegisters = uint32_t[2];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.32x32b.x2.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 4 times
struct SM100_TMEM_STORE_32dp32b4x
{
  using SRegisters = uint32_t[4];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.32x32b.x4.b32"
                    "[%0],"
                    "{%1, %2, %3, %4};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 4 times, expand 16b write
struct SM100_TMEM_STORE_32dp32b4x_16b
{
  using SRegisters = uint32_t[4];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.32x32b.x4.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 8 times
struct SM100_TMEM_STORE_32dp32b8x
{
  using SRegisters = uint32_t[8];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& src4, uint32_t const& src5, uint32_t const& src6, uint32_t const& src7,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.32x32b.x8.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3),
       "r"(src4), "r"(src5), "r"(src6), "r"(src7) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 8 times, expand 16b write
struct SM100_TMEM_STORE_32dp32b8x_16b
{
  using SRegisters = uint32_t[8];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint32_t const& src4, uint32_t const& src5, uint32_t const& src6, uint32_t const& src7,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.32x32b.x8.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8};\n"
    :
    :  "r"(dst_addr), "r"(src0), "r"(src1), "r"(src2), "r"(src3),
       "r"(src4), "r"(src5), "r"(src6), "r"(src7) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 16 times
struct SM100_TMEM_STORE_32dp32b16x
{
  using SRegisters = uint32_t[16];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.32x32b.x16.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 16 times, expand 16b write
struct SM100_TMEM_STORE_32dp32b16x_16b
{
  using SRegisters = uint32_t[16];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.32x32b.x16.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 32 times
struct SM100_TMEM_STORE_32dp32b32x
{
  using SRegisters = uint32_t[32];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.32x32b.x32.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 32 times, expand 16b write
struct SM100_TMEM_STORE_32dp32b32x_16b
{
  using SRegisters = uint32_t[32];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.32x32b.x32.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 64 times
struct SM100_TMEM_STORE_32dp32b64x
{
  using SRegisters = uint32_t[64];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& src32, uint32_t const& src33, uint32_t const& src34, uint32_t const& src35,
       uint32_t const& src36, uint32_t const& src37, uint32_t const& src38, uint32_t const& src39,
       uint32_t const& src40, uint32_t const& src41, uint32_t const& src42, uint32_t const& src43,
       uint32_t const& src44, uint32_t const& src45, uint32_t const& src46, uint32_t const& src47,
       uint32_t const& src48, uint32_t const& src49, uint32_t const& src50, uint32_t const& src51,
       uint32_t const& src52, uint32_t const& src53, uint32_t const& src54, uint32_t const& src55,
       uint32_t const& src56, uint32_t const& src57, uint32_t const& src58, uint32_t const& src59,
       uint32_t const& src60, uint32_t const& src61, uint32_t const& src62, uint32_t const& src63,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.32x32b.x64.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31),
       "r"(src32), "r"(src33), "r"(src34), "r"(src35),
       "r"(src36), "r"(src37), "r"(src38), "r"(src39),
       "r"(src40), "r"(src41), "r"(src42), "r"(src43),
       "r"(src44), "r"(src45), "r"(src46), "r"(src47),
       "r"(src48), "r"(src49), "r"(src50), "r"(src51),
       "r"(src52), "r"(src53), "r"(src54), "r"(src55),
       "r"(src56), "r"(src57), "r"(src58), "r"(src59),
       "r"(src60), "r"(src61), "r"(src62), "r"(src63) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 64 times, expand 16b write
struct SM100_TMEM_STORE_32dp32b64x_16b
{
  using SRegisters = uint32_t[64];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src00, uint32_t const& src01, uint32_t const& src02, uint32_t const& src03,
       uint32_t const& src04, uint32_t const& src05, uint32_t const& src06, uint32_t const& src07,
       uint32_t const& src08, uint32_t const& src09, uint32_t const& src10, uint32_t const& src11,
       uint32_t const& src12, uint32_t const& src13, uint32_t const& src14, uint32_t const& src15,
       uint32_t const& src16, uint32_t const& src17, uint32_t const& src18, uint32_t const& src19,
       uint32_t const& src20, uint32_t const& src21, uint32_t const& src22, uint32_t const& src23,
       uint32_t const& src24, uint32_t const& src25, uint32_t const& src26, uint32_t const& src27,
       uint32_t const& src28, uint32_t const& src29, uint32_t const& src30, uint32_t const& src31,
       uint32_t const& src32, uint32_t const& src33, uint32_t const& src34, uint32_t const& src35,
       uint32_t const& src36, uint32_t const& src37, uint32_t const& src38, uint32_t const& src39,
       uint32_t const& src40, uint32_t const& src41, uint32_t const& src42, uint32_t const& src43,
       uint32_t const& src44, uint32_t const& src45, uint32_t const& src46, uint32_t const& src47,
       uint32_t const& src48, uint32_t const& src49, uint32_t const& src50, uint32_t const& src51,
       uint32_t const& src52, uint32_t const& src53, uint32_t const& src54, uint32_t const& src55,
       uint32_t const& src56, uint32_t const& src57, uint32_t const& src58, uint32_t const& src59,
       uint32_t const& src60, uint32_t const& src61, uint32_t const& src62, uint32_t const& src63,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.32x32b.x64.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64};\n"
    :
    :  "r"(dst_addr), "r"(src00), "r"(src01), "r"(src02), "r"(src03),
       "r"(src04), "r"(src05), "r"(src06), "r"(src07),
       "r"(src08), "r"(src09), "r"(src10), "r"(src11),
       "r"(src12), "r"(src13), "r"(src14), "r"(src15),
       "r"(src16), "r"(src17), "r"(src18), "r"(src19),
       "r"(src20), "r"(src21), "r"(src22), "r"(src23),
       "r"(src24), "r"(src25), "r"(src26), "r"(src27),
       "r"(src28), "r"(src29), "r"(src30), "r"(src31),
       "r"(src32), "r"(src33), "r"(src34), "r"(src35),
       "r"(src36), "r"(src37), "r"(src38), "r"(src39),
       "r"(src40), "r"(src41), "r"(src42), "r"(src43),
       "r"(src44), "r"(src45), "r"(src46), "r"(src47),
       "r"(src48), "r"(src49), "r"(src50), "r"(src51),
       "r"(src52), "r"(src53), "r"(src54), "r"(src55),
       "r"(src56), "r"(src57), "r"(src58), "r"(src59),
       "r"(src60), "r"(src61), "r"(src62), "r"(src63) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 128 times
struct SM100_TMEM_STORE_32dp32b128x
{
  using SRegisters = uint32_t[128];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src000, uint32_t const& src001, uint32_t const& src002, uint32_t const& src003,
       uint32_t const& src004, uint32_t const& src005, uint32_t const& src006, uint32_t const& src007,
       uint32_t const& src008, uint32_t const& src009, uint32_t const& src010, uint32_t const& src011,
       uint32_t const& src012, uint32_t const& src013, uint32_t const& src014, uint32_t const& src015,
       uint32_t const& src016, uint32_t const& src017, uint32_t const& src018, uint32_t const& src019,
       uint32_t const& src020, uint32_t const& src021, uint32_t const& src022, uint32_t const& src023,
       uint32_t const& src024, uint32_t const& src025, uint32_t const& src026, uint32_t const& src027,
       uint32_t const& src028, uint32_t const& src029, uint32_t const& src030, uint32_t const& src031,
       uint32_t const& src032, uint32_t const& src033, uint32_t const& src034, uint32_t const& src035,
       uint32_t const& src036, uint32_t const& src037, uint32_t const& src038, uint32_t const& src039,
       uint32_t const& src040, uint32_t const& src041, uint32_t const& src042, uint32_t const& src043,
       uint32_t const& src044, uint32_t const& src045, uint32_t const& src046, uint32_t const& src047,
       uint32_t const& src048, uint32_t const& src049, uint32_t const& src050, uint32_t const& src051,
       uint32_t const& src052, uint32_t const& src053, uint32_t const& src054, uint32_t const& src055,
       uint32_t const& src056, uint32_t const& src057, uint32_t const& src058, uint32_t const& src059,
       uint32_t const& src060, uint32_t const& src061, uint32_t const& src062, uint32_t const& src063,
       uint32_t const& src064, uint32_t const& src065, uint32_t const& src066, uint32_t const& src067,
       uint32_t const& src068, uint32_t const& src069, uint32_t const& src070, uint32_t const& src071,
       uint32_t const& src072, uint32_t const& src073, uint32_t const& src074, uint32_t const& src075,
       uint32_t const& src076, uint32_t const& src077, uint32_t const& src078, uint32_t const& src079,
       uint32_t const& src080, uint32_t const& src081, uint32_t const& src082, uint32_t const& src083,
       uint32_t const& src084, uint32_t const& src085, uint32_t const& src086, uint32_t const& src087,
       uint32_t const& src088, uint32_t const& src089, uint32_t const& src090, uint32_t const& src091,
       uint32_t const& src092, uint32_t const& src093, uint32_t const& src094, uint32_t const& src095,
       uint32_t const& src096, uint32_t const& src097, uint32_t const& src098, uint32_t const& src099,
       uint32_t const& src100, uint32_t const& src101, uint32_t const& src102, uint32_t const& src103,
       uint32_t const& src104, uint32_t const& src105, uint32_t const& src106, uint32_t const& src107,
       uint32_t const& src108, uint32_t const& src109, uint32_t const& src110, uint32_t const& src111,
       uint32_t const& src112, uint32_t const& src113, uint32_t const& src114, uint32_t const& src115,
       uint32_t const& src116, uint32_t const& src117, uint32_t const& src118, uint32_t const& src119,
       uint32_t const& src120, uint32_t const& src121, uint32_t const& src122, uint32_t const& src123,
       uint32_t const& src124, uint32_t const& src125, uint32_t const& src126, uint32_t const& src127,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.32x32b.x128.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64,"
                    "%65, %66, %67, %68,"
                    "%69, %70, %71, %72,"
                    "%73, %74, %75, %76,"
                    "%77, %78, %79, %80,"
                    "%81, %82, %83, %84,"
                    "%85, %86, %87, %88,"
                    "%89, %90, %91, %92,"
                    "%93, %94, %95, %96,"
                    "%97, %98, %99, %100,"
                    "%101, %102, %103, %104,"
                    "%105, %106, %107, %108,"
                    "%109, %110, %111, %112,"
                    "%113, %114, %115, %116,"
                    "%117, %118, %119, %120,"
                    "%121, %122, %123, %124,"
                    "%125, %126, %127, %128};\n"
    :
    :  "r"(dst_addr), "r"(src000), "r"(src001), "r"(src002), "r"(src003),
       "r"(src004), "r"(src005), "r"(src006), "r"(src007),
       "r"(src008), "r"(src009), "r"(src010), "r"(src011),
       "r"(src012), "r"(src013), "r"(src014), "r"(src015),
       "r"(src016), "r"(src017), "r"(src018), "r"(src019),
       "r"(src020), "r"(src021), "r"(src022), "r"(src023),
       "r"(src024), "r"(src025), "r"(src026), "r"(src027),
       "r"(src028), "r"(src029), "r"(src030), "r"(src031),
       "r"(src032), "r"(src033), "r"(src034), "r"(src035),
       "r"(src036), "r"(src037), "r"(src038), "r"(src039),
       "r"(src040), "r"(src041), "r"(src042), "r"(src043),
       "r"(src044), "r"(src045), "r"(src046), "r"(src047),
       "r"(src048), "r"(src049), "r"(src050), "r"(src051),
       "r"(src052), "r"(src053), "r"(src054), "r"(src055),
       "r"(src056), "r"(src057), "r"(src058), "r"(src059),
       "r"(src060), "r"(src061), "r"(src062), "r"(src063),
       "r"(src064), "r"(src065), "r"(src066), "r"(src067),
       "r"(src068), "r"(src069), "r"(src070), "r"(src071),
       "r"(src072), "r"(src073), "r"(src074), "r"(src075),
       "r"(src076), "r"(src077), "r"(src078), "r"(src079),
       "r"(src080), "r"(src081), "r"(src082), "r"(src083),
       "r"(src084), "r"(src085), "r"(src086), "r"(src087),
       "r"(src088), "r"(src089), "r"(src090), "r"(src091),
       "r"(src092), "r"(src093), "r"(src094), "r"(src095),
       "r"(src096), "r"(src097), "r"(src098), "r"(src099),
       "r"(src100), "r"(src101), "r"(src102), "r"(src103),
       "r"(src104), "r"(src105), "r"(src106), "r"(src107),
       "r"(src108), "r"(src109), "r"(src110), "r"(src111),
       "r"(src112), "r"(src113), "r"(src114), "r"(src115),
       "r"(src116), "r"(src117), "r"(src118), "r"(src119),
       "r"(src120), "r"(src121), "r"(src122), "r"(src123),
       "r"(src124), "r"(src125), "r"(src126), "r"(src127) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// 32 data path lanes, 32-bit pattern, repeated 128 times, expand 16b write
struct SM100_TMEM_STORE_32dp32b128x_16b
{
  using SRegisters = uint32_t[128];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src000, uint32_t const& src001, uint32_t const& src002, uint32_t const& src003,
       uint32_t const& src004, uint32_t const& src005, uint32_t const& src006, uint32_t const& src007,
       uint32_t const& src008, uint32_t const& src009, uint32_t const& src010, uint32_t const& src011,
       uint32_t const& src012, uint32_t const& src013, uint32_t const& src014, uint32_t const& src015,
       uint32_t const& src016, uint32_t const& src017, uint32_t const& src018, uint32_t const& src019,
       uint32_t const& src020, uint32_t const& src021, uint32_t const& src022, uint32_t const& src023,
       uint32_t const& src024, uint32_t const& src025, uint32_t const& src026, uint32_t const& src027,
       uint32_t const& src028, uint32_t const& src029, uint32_t const& src030, uint32_t const& src031,
       uint32_t const& src032, uint32_t const& src033, uint32_t const& src034, uint32_t const& src035,
       uint32_t const& src036, uint32_t const& src037, uint32_t const& src038, uint32_t const& src039,
       uint32_t const& src040, uint32_t const& src041, uint32_t const& src042, uint32_t const& src043,
       uint32_t const& src044, uint32_t const& src045, uint32_t const& src046, uint32_t const& src047,
       uint32_t const& src048, uint32_t const& src049, uint32_t const& src050, uint32_t const& src051,
       uint32_t const& src052, uint32_t const& src053, uint32_t const& src054, uint32_t const& src055,
       uint32_t const& src056, uint32_t const& src057, uint32_t const& src058, uint32_t const& src059,
       uint32_t const& src060, uint32_t const& src061, uint32_t const& src062, uint32_t const& src063,
       uint32_t const& src064, uint32_t const& src065, uint32_t const& src066, uint32_t const& src067,
       uint32_t const& src068, uint32_t const& src069, uint32_t const& src070, uint32_t const& src071,
       uint32_t const& src072, uint32_t const& src073, uint32_t const& src074, uint32_t const& src075,
       uint32_t const& src076, uint32_t const& src077, uint32_t const& src078, uint32_t const& src079,
       uint32_t const& src080, uint32_t const& src081, uint32_t const& src082, uint32_t const& src083,
       uint32_t const& src084, uint32_t const& src085, uint32_t const& src086, uint32_t const& src087,
       uint32_t const& src088, uint32_t const& src089, uint32_t const& src090, uint32_t const& src091,
       uint32_t const& src092, uint32_t const& src093, uint32_t const& src094, uint32_t const& src095,
       uint32_t const& src096, uint32_t const& src097, uint32_t const& src098, uint32_t const& src099,
       uint32_t const& src100, uint32_t const& src101, uint32_t const& src102, uint32_t const& src103,
       uint32_t const& src104, uint32_t const& src105, uint32_t const& src106, uint32_t const& src107,
       uint32_t const& src108, uint32_t const& src109, uint32_t const& src110, uint32_t const& src111,
       uint32_t const& src112, uint32_t const& src113, uint32_t const& src114, uint32_t const& src115,
       uint32_t const& src116, uint32_t const& src117, uint32_t const& src118, uint32_t const& src119,
       uint32_t const& src120, uint32_t const& src121, uint32_t const& src122, uint32_t const& src123,
       uint32_t const& src124, uint32_t const& src125, uint32_t const& src126, uint32_t const& src127,
       uint32_t const& dst_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.st.sync.aligned.32x32b.x128.unpack::16b.b32"
                    "[%0],"
                    "{%1, %2, %3, %4,"
                    "%5, %6, %7, %8,"
                    "%9, %10, %11, %12,"
                    "%13, %14, %15, %16,"
                    "%17, %18, %19, %20,"
                    "%21, %22, %23, %24,"
                    "%25, %26, %27, %28,"
                    "%29, %30, %31, %32,"
                    "%33, %34, %35, %36,"
                    "%37, %38, %39, %40,"
                    "%41, %42, %43, %44,"
                    "%45, %46, %47, %48,"
                    "%49, %50, %51, %52,"
                    "%53, %54, %55, %56,"
                    "%57, %58, %59, %60,"
                    "%61, %62, %63, %64,"
                    "%65, %66, %67, %68,"
                    "%69, %70, %71, %72,"
                    "%73, %74, %75, %76,"
                    "%77, %78, %79, %80,"
                    "%81, %82, %83, %84,"
                    "%85, %86, %87, %88,"
                    "%89, %90, %91, %92,"
                    "%93, %94, %95, %96,"
                    "%97, %98, %99, %100,"
                    "%101, %102, %103, %104,"
                    "%105, %106, %107, %108,"
                    "%109, %110, %111, %112,"
                    "%113, %114, %115, %116,"
                    "%117, %118, %119, %120,"
                    "%121, %122, %123, %124,"
                    "%125, %126, %127, %128};\n"
    :
    :  "r"(dst_addr), "r"(src000), "r"(src001), "r"(src002), "r"(src003),
       "r"(src004), "r"(src005), "r"(src006), "r"(src007),
       "r"(src008), "r"(src009), "r"(src010), "r"(src011),
       "r"(src012), "r"(src013), "r"(src014), "r"(src015),
       "r"(src016), "r"(src017), "r"(src018), "r"(src019),
       "r"(src020), "r"(src021), "r"(src022), "r"(src023),
       "r"(src024), "r"(src025), "r"(src026), "r"(src027),
       "r"(src028), "r"(src029), "r"(src030), "r"(src031),
       "r"(src032), "r"(src033), "r"(src034), "r"(src035),
       "r"(src036), "r"(src037), "r"(src038), "r"(src039),
       "r"(src040), "r"(src041), "r"(src042), "r"(src043),
       "r"(src044), "r"(src045), "r"(src046), "r"(src047),
       "r"(src048), "r"(src049), "r"(src050), "r"(src051),
       "r"(src052), "r"(src053), "r"(src054), "r"(src055),
       "r"(src056), "r"(src057), "r"(src058), "r"(src059),
       "r"(src060), "r"(src061), "r"(src062), "r"(src063),
       "r"(src064), "r"(src065), "r"(src066), "r"(src067),
       "r"(src068), "r"(src069), "r"(src070), "r"(src071),
       "r"(src072), "r"(src073), "r"(src074), "r"(src075),
       "r"(src076), "r"(src077), "r"(src078), "r"(src079),
       "r"(src080), "r"(src081), "r"(src082), "r"(src083),
       "r"(src084), "r"(src085), "r"(src086), "r"(src087),
       "r"(src088), "r"(src089), "r"(src090), "r"(src091),
       "r"(src092), "r"(src093), "r"(src094), "r"(src095),
       "r"(src096), "r"(src097), "r"(src098), "r"(src099),
       "r"(src100), "r"(src101), "r"(src102), "r"(src103),
       "r"(src104), "r"(src105), "r"(src106), "r"(src107),
       "r"(src108), "r"(src109), "r"(src110), "r"(src111),
       "r"(src112), "r"(src113), "r"(src114), "r"(src115),
       "r"(src116), "r"(src117), "r"(src118), "r"(src119),
       "r"(src120), "r"(src121), "r"(src122), "r"(src123),
       "r"(src124), "r"(src125), "r"(src126), "r"(src127) );
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_STORE without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace SM100::TMEM::STORE

////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cute
