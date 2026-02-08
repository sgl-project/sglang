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

#include <cute/config.hpp>      // CUTE_HOST_DEVICE
#include <cute/arch/config.hpp> // CUTE_ARCH_TMA_SMxx_ENABLED
#include <cute/arch/copy.hpp>

namespace cute
{

struct SM90_U32x1_STSM_N
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint128_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src,
       uint128_t     & smem_dst)
  {
#if defined(CUTE_ARCH_STSM_SM90_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    asm volatile ("stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n"
        :: "r"(smem_int_ptr),
           "r"(src));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use stmatrix without CUTE_ARCH_STSM_SM90_ENABLED.");
#endif
  }
};

struct SM90_U32x2_STSM_N
{
  using SRegisters = uint32_t[2];
  using DRegisters = uint128_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1,
       uint128_t& smem_dst)
  {
#if defined(CUTE_ARCH_STSM_SM90_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    asm volatile ("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
        :: "r"(smem_int_ptr),
           "r"(src0), "r"(src1));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use stmatrix without CUTE_ARCH_STSM_SM90_ENABLED.");
#endif
  }
};

struct SM90_U32x4_STSM_N
{
  using SRegisters = uint32_t[4];
  using DRegisters = uint128_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint128_t& smem_dst)
  {
#if defined(CUTE_ARCH_STSM_SM90_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    asm volatile ("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(smem_int_ptr),
          "r"(src0), "r"(src1), "r"(src2), "r"(src3));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use stmatrix without CUTE_ARCH_STSM_SM90_ENABLED.");
#endif
  }
};

struct SM90_U16x2_STSM_T
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint128_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src,
       uint128_t& smem_dst)
  {
#if defined(CUTE_ARCH_STSM_SM90_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    asm volatile ("stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0], {%1};\n"
        :: "r"(smem_int_ptr),
           "r"(src));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use stmatrix without CUTE_ARCH_STSM_SM90_ENABLED.");
#endif
  }
};

struct SM90_U16x4_STSM_T
{
  using SRegisters = uint32_t[2];
  using DRegisters = uint128_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1,
       uint128_t& smem_dst)
  {
#if defined(CUTE_ARCH_STSM_SM90_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    asm volatile ("stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1, %2};\n"
        :: "r"(smem_int_ptr),
           "r"(src0), "r"(src1));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use stmatrix without CUTE_ARCH_STSM_SM90_ENABLED.");
#endif
  }
};

struct SM90_U16x8_STSM_T
{
  using SRegisters = uint32_t[4];
  using DRegisters = uint128_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint128_t& smem_dst)
  {
#if defined(CUTE_ARCH_STSM_SM90_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    asm volatile ("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(smem_int_ptr),
          "r"(src0), "r"(src1), "r"(src2), "r"(src3));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use stmatrix without CUTE_ARCH_STSM_SM90_ENABLED.");
#endif
  }
};

//
// Legacy STSM interfaces that aren't very useful
//

template <class T>
CUTE_HOST_DEVICE
void
copy_stsm(T const* const rmem_ptr,
          uint128_t* const smem_ptr)
{
  uint32_t const* reg_ptr = reinterpret_cast<uint32_t const*>(rmem_ptr);

  // if constexpr
  if (sizeof(T) == 4) {
    SM90_U32x1_STSM_N::copy(reg_ptr[0], smem_ptr[0]);
  }
  else if (sizeof(T) == 8) {
    SM90_U32x2_STSM_N::copy(reg_ptr[0], reg_ptr[1], smem_ptr[0]);
  }
  else if (sizeof(T) == 16) {
    SM90_U32x4_STSM_N::copy(reg_ptr[0], reg_ptr[1], reg_ptr[2], reg_ptr[3], smem_ptr[0]);
  }
  else {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16, "sizeof(T) is not supported");
  }
}

template <class T>
CUTE_HOST_DEVICE
void
copy_stsm_trans(T const* const rmem_ptr,
                uint128_t* const smem_ptr)
{
  uint32_t const* reg_ptr = reinterpret_cast<uint32_t const*>(rmem_ptr);

  // if constexpr
  if (sizeof(T) == 4) {
    SM90_U16x2_STSM_T::copy(reg_ptr[0], smem_ptr[0]);
  }
  else if (sizeof(T) == 8) {
    SM90_U16x4_STSM_T::copy(reg_ptr[0], reg_ptr[1], smem_ptr[0]);
  }
  else if (sizeof(T) == 16) {
    SM90_U16x8_STSM_T::copy(reg_ptr[0], reg_ptr[1], reg_ptr[2], reg_ptr[3], smem_ptr[0]);
  }
  else {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16, "sizeof(T) is not supported");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cute

////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////
