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

#include <cute/config.hpp>

#include <cute/arch/copy.hpp>

// Config
#if defined(__clang__) && defined(__CUDA__)
  // ldmatrix PTX instructions added in Clang 14: https://reviews.llvm.org/D107046
  // ... but will not work until Clang 15:
  //   * https://reviews.llvm.org/D121666
  //   * https://reviews.llvm.org/D126846
  #define CUTE_ARCH_CLANG_SUPPORTS_LDSM_SM75 (__clang_major__ >= 15)
  #define CUTE_ARCH_CLANG_SUPPORTS_MOVM_SM75 (__clang_major__ >= 15)
#endif

#if defined(__NVCC__) || defined(__CUDACC_RTC__)
  // ldmatrix PTX instruction added in CUDA 10.2+
  #define CUTE_ARCH_NVCC_SUPPORTS_LDSM_SM75 ((__CUDACC_VER_MAJOR__  == 10 && __CUDACC_VER_MINOR__ >= 2) || __CUDACC_VER_MAJOR__ >= 11)
  #define CUTE_ARCH_NVCC_SUPPORTS_MOVM_SM75 ((__CUDACC_VER_MAJOR__  == 10 && __CUDACC_VER_MINOR__ >= 2) || __CUDACC_VER_MAJOR__ >= 11)
#endif

#if ! defined(CUTE_ARCH_LDSM_SM75_SUPPORTED)
  #define CUTE_ARCH_LDSM_SM75_SUPPORTED (CUTE_ARCH_NVCC_SUPPORTS_LDSM_SM75 || CUTE_ARCH_CLANG_SUPPORTS_LDSM_SM75)
#endif

#if ! defined(CUTE_ARCH_LDSM_SM75_ENABLED)
  #define CUTE_ARCH_LDSM_SM75_ENABLED (CUTE_ARCH_LDSM_SM75_SUPPORTED)
#endif

#if (CUTE_ARCH_LDSM_SM75_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  #define CUTE_ARCH_LDSM_SM75_ACTIVATED 1
#endif

#if ! defined(CUTE_ARCH_MOVM_SM75_SUPPORTED)
  #define CUTE_ARCH_MOVM_SM75_SUPPORTED (CUTE_ARCH_NVCC_SUPPORTS_MOVM_SM75 || CUTE_ARCH_CLANG_SUPPORTS_MOVM_SM75)
#endif

#if ! defined(CUTE_ARCH_MOVM_SM75_ENABLED)
  #define CUTE_ARCH_MOVM_SM75_ENABLED (CUTE_ARCH_MOVM_SM75_SUPPORTED)
#endif

#if (CUTE_ARCH_MOVM_SM75_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  #define CUTE_ARCH_MOVM_SM75_ACTIVATED 1
#endif


namespace cute
{

struct SM75_U32x1_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
        : "=r"(dst)
        :  "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
  }
};

struct SM75_U32x2_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst0), "=r"(dst1)
        :  "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
  }
};

struct SM75_U32x4_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
  }
};

struct SM75_U16x2_LDSM_T
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"
        : "=r"(dst)
        :  "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
  }
};

struct SM75_U16x4_LDSM_T
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst0), "=r"(dst1)
        :  "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
  }
};

struct SM75_U16x8_LDSM_T
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
  }
};

struct SM75_U32x1_MOVM_T 
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void 
  copy(uint32_t src, 
       uint32_t &dst) 
  {
#if CUTE_ARCH_MOVM_SM75_ACTIVATED
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
        : "=r"(dst)
        :  "r"(src));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use movmatrix without CUTE_ARCH_MOVM_SM75_ACTIVATED.");
#endif
  }
};
//
// Legacy LDSM interfaces that aren't very useful
//

template <class T>
CUTE_HOST_DEVICE
void
copy_ldsm(uint128_t const* const smem_ptr,
          T* rmem_ptr)
{
  uint32_t* reg_ptr = reinterpret_cast<uint32_t*>(rmem_ptr);

  // if constexpr
  if (sizeof(T) == 4) {
    SM75_U32x1_LDSM_N::copy(smem_ptr[0], reg_ptr[0]);
  }
  else if (sizeof(T) == 8) {
    SM75_U32x2_LDSM_N::copy(smem_ptr[0], reg_ptr[0], reg_ptr[1]);
  }
  else if (sizeof(T) == 16) {
    SM75_U32x4_LDSM_N::copy(smem_ptr[0], reg_ptr[0], reg_ptr[1], reg_ptr[2], reg_ptr[3]);
  }
  else {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16, "sizeof(T) is not supported");
  }
}

template <class T>
CUTE_HOST_DEVICE
void
copy_ldsm_trans(uint128_t const* const smem_ptr,
                T* rmem_ptr)
{
  uint32_t* reg_ptr = reinterpret_cast<uint32_t*>(rmem_ptr);

  // if constexpr
  if (sizeof(T) == 4) {
    SM75_U16x2_LDSM_T::copy(smem_ptr[0], reg_ptr[0]);
  }
  else if (sizeof(T) == 8) {
    SM75_U16x4_LDSM_T::copy(smem_ptr[0], reg_ptr[0], reg_ptr[1]);
  }
  else if (sizeof(T) == 16) {
    SM75_U16x8_LDSM_T::copy(smem_ptr[0], reg_ptr[0], reg_ptr[1], reg_ptr[2], reg_ptr[3]);
  }
  else {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16, "sizeof(T) is not supported");
  }
}

} // end namespace cute
