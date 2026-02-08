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

#include <cute/arch/config.hpp> // CUTE_ARCH_TMA_SMxx_ENABLED
#include <cute/arch/copy.hpp>
#include <cute/arch/copy_sm90.hpp>
#include "cutlass/arch/synclog.hpp"

namespace cute
{

////////////////////////////////////////////////////////////////////////////////////////////////////
/// TMA_LOAD : Initiates a TMA copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_LOAD_1D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
    asm volatile (
      "cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3}], [%2], %4;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "l"(cache_hint)
      : "memory");
#else
    asm volatile (
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3}], [%2], %4;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "l"(cache_hint)
      : "memory");
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  struct PREFETCH
  {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& crd0)
    {
  #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      asm volatile (
        "cp.async.bulk.prefetch.tensor.1d.L2.global"
        " [%0, {%1}];"
        :
        : "l"(gmem_int_desc),
          "r"(crd0)
        : "memory");
  #else
      CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
  #endif
    }
  };
};

struct SM90_TMA_LOAD_2D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
    asm volatile (
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "l"(cache_hint)
      : "memory");
#else
    asm volatile (
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "l"(cache_hint)
      : "memory");
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  struct PREFETCH
  {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& crd0, int32_t const& crd1)
    {
  #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      asm volatile (
        "cp.async.bulk.prefetch.tensor.2d.L2.global"
        " [%0, {%1, %2}];"
        :
        : "l"(gmem_int_desc),
          "r"(crd0), "r"(crd1)
        : "memory");
  #else
      CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
  #endif
    }
  };
};

struct SM90_TMA_LOAD_3D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
    asm volatile (
      "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5}], [%2], %6;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
      : "memory");
#else
    asm volatile (
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5}], [%2], %6;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
      : "memory");
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  struct PREFETCH
  {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
    {
  #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      asm volatile (
        "cp.async.bulk.prefetch.tensor.3d.L2.global"
        " [%0, {%1, %2, %3}];"
        :
        : "l"(gmem_int_desc),
          "r"(crd0), "r"(crd1), "r"(crd2)
        : "memory");
  #else
      CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
  #endif
    }
  };
};

struct SM90_TMA_LOAD_4D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
    asm volatile (
      "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "l"(cache_hint)
      : "memory");
#else
    asm volatile (
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "l"(cache_hint)
      : "memory");
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  struct PREFETCH
  {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
    {
  #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      asm volatile (
        "cp.async.bulk.prefetch.tensor.4d.L2.global"
        " [%0, {%1, %2, %3, %4}];"
        :
        : "l"(gmem_int_desc),
          "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3)
        : "memory");
  #else
      CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
  #endif
    }
  };
};

struct SM90_TMA_LOAD_5D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
    asm volatile (
      "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4), "l"(cache_hint)
      : "memory");
#else
    asm volatile (
      "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4), "l"(cache_hint)
      : "memory");
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  struct PREFETCH
  {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
    {
  #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      asm volatile (
        "cp.async.bulk.prefetch.tensor.5d.L2.global"
        " [%0, {%1, %2, %3, %4, %5}];"
        :
        : "l"(gmem_int_desc),
          "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
        : "memory");
  #else
      CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
  #endif
    }
  };
};

struct SM90_TMA_LOAD
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0)
  {
    return SM90_TMA_LOAD_1D::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
    return SM90_TMA_LOAD_2D::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, crd1);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
    return SM90_TMA_LOAD_3D::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, crd1, crd2);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
    return SM90_TMA_LOAD_4D::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, crd1, crd2, crd3);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
    return SM90_TMA_LOAD_5D::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, crd1, crd2, crd3, crd4);
  }

  struct PREFETCH
  {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& crd0)
    {
      return SM90_TMA_LOAD_1D::PREFETCH::copy(desc_ptr, crd0);
    }
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& crd0, int32_t const& crd1)
    {
      return SM90_TMA_LOAD_2D::PREFETCH::copy(desc_ptr, crd0, crd1);
    }
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
    {
      return SM90_TMA_LOAD_3D::PREFETCH::copy(desc_ptr, crd0, crd1, crd2);
    }
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
    {
      return SM90_TMA_LOAD_4D::PREFETCH::copy(desc_ptr, crd0, crd1, crd2, crd3);
    }
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
    {
      return SM90_TMA_LOAD_5D::PREFETCH::copy(desc_ptr, crd0, crd1, crd2, crd3, crd4);
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// TMA_LOAD im2col: Initiates a TMA copy, in im2col mode, from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_LOAD_IM2COL_3D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr,
       void      * smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_n,
       uint16_t const& offset_w)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
    // Copy from global to shared::cluster.
    asm volatile (
      "cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5}], [%2], {%6};"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(coord_c), "r"(coord_w), "r"(coord_n),
        "h"(offset_w)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  struct PREFETCH
  {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_n,
         uint16_t const& offset_w)
    {
  #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      asm volatile (
        "cp.async.bulk.prefetch.tensor.3d.L2.global.im2col"
        " [%0, {%1, %2, %3}], {%4};"
        :
        : "l"(gmem_int_desc),
          "r"(coord_c), "r"(coord_w), "r"(coord_n),
          "h"(offset_w)
        : "memory");
  #else
      CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
  #endif
    }
  };
};

struct SM90_TMA_LOAD_IM2COL_4D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr,
       void      * smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_n,
       uint16_t const& offset_w, uint16_t const& offset_h)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
    // Copy from global to shared::cluster.
    asm volatile (
      "cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6}], [%2], {%7, %8};"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_n),
        "h"(offset_w), "h"(offset_h)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  struct PREFETCH
  {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_n,
         uint16_t const& offset_w, uint16_t const& offset_h)
    {
  #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      asm volatile (
        "cp.async.bulk.prefetch.tensor.4d.L2.global.im2col"
        " [%0, {%1, %2, %3, %4}], {%5, %6};"
        :
        : "l"(gmem_int_desc),
          "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_n),
          "h"(offset_w), "h"(offset_h)
        : "memory");
  #else
      CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
  #endif
    }
  };
};

struct SM90_TMA_LOAD_IM2COL_5D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr,
       void      * smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_d, int32_t const& coord_n,
       uint16_t const& offset_w, uint16_t const& offset_h, uint16_t const& offset_d)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
    // Copy from global to shared::cluster.
    asm volatile (
      "cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], {%8, %9, %10};"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_d), "r"(coord_n),
        "h"(offset_w), "h"(offset_h), "h"(offset_d)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  struct PREFETCH
  {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_d, int32_t const& coord_n,
         uint16_t const& offset_w, uint16_t const& offset_h, uint16_t const& offset_d)
    {
  #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
      uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
      asm volatile (
        "cp.async.bulk.prefetch.tensor.5d.L2.global.im2col"
        " [%0, {%1, %2, %3, %4, %5}], {%6, %7, %8};"
        :
        : "l"(gmem_int_desc),
          "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_d), "r"(coord_n),
          "h"(offset_w), "h"(offset_h), "h"(offset_d)
        : "memory");
  #else
      CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
  #endif
    }
  };
};

struct SM90_TMA_LOAD_IM2COL
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr,
       void      * smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_n,
       uint16_t const& offset_w)
  {
    return SM90_TMA_LOAD_IM2COL_3D::copy(desc_ptr, mbar_ptr, smem_ptr,
                                         coord_c, coord_w, coord_n,
                                         offset_w);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr,
       void      * smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_n,
       uint16_t const& offset_w, uint16_t const& offset_h)
  {
    return SM90_TMA_LOAD_IM2COL_4D::copy(desc_ptr, mbar_ptr, smem_ptr,
                                         coord_c, coord_w, coord_h, coord_n,
                                         offset_w, offset_h);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr,
       void      * smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_d, int32_t const& coord_n,
       uint16_t const& offset_w, uint16_t const& offset_h, uint16_t const& offset_d)
  {
    return SM90_TMA_LOAD_IM2COL_5D::copy(desc_ptr, mbar_ptr, smem_ptr,
                                         coord_c, coord_w, coord_h, coord_d, coord_n,
                                         offset_w, offset_h, offset_d);
  }

  struct PREFETCH
  {
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_n,
         uint16_t const& offset_w)
    {
      return SM90_TMA_LOAD_IM2COL_3D::PREFETCH::copy(desc_ptr,
                                                     coord_c, coord_w, coord_n,
                                                     offset_w);
    }
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_n,
         uint16_t const& offset_w, uint16_t const& offset_h)
    {
      return SM90_TMA_LOAD_IM2COL_4D::PREFETCH::copy(desc_ptr,
                                                     coord_c, coord_w, coord_h, coord_n,
                                                     offset_w, offset_h);
    }
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr,
         int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_d, int32_t const& coord_n,
         uint16_t const& offset_w, uint16_t const& offset_h, uint16_t const& offset_d)
    {
      return SM90_TMA_LOAD_IM2COL_5D::PREFETCH::copy(desc_ptr,
                                                     coord_c, coord_w, coord_h, coord_d, coord_n,
                                                     offset_w, offset_h, offset_d);
    }
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// TMA_LOAD_MULTICAST: Initiates a TMA copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_LOAD_MULTICAST_1D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
    asm volatile (
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
      " [%0], [%1, {%4}], [%2], %3, %5;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "h"(multicast_mask),
        "r"(crd0), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_MULTICAST_2D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
    asm volatile (
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
      " [%0], [%1, {%4, %5}], [%2], %3, %6;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "h"(multicast_mask),
        "r"(crd0), "r"(crd1), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_MULTICAST_3D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
    asm volatile (
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
      " [%0], [%1, {%4, %5, %6}], [%2], %3, %7;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "h"(multicast_mask),
        "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_MULTICAST_4D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
    asm volatile (
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
      " [%0], [%1, {%4, %5, %6, %7}], [%2], %3, %8;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "h"(multicast_mask),
        "r"(crd0), "r"(crd1), "r"(crd2),  "r"(crd3), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_MULTICAST_5D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
    asm volatile (
      "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
      " [%0], [%1, {%4, %5, %6, %7, %8}], [%2], %3, %9;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "h"(multicast_mask),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4), "l"(cache_hint)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_MULTICAST
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0)
  {
    return SM90_TMA_LOAD_MULTICAST_1D::copy(desc_ptr, mbar_ptr, multicast_mask, cache_hint, smem_ptr, crd0);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
    return SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, mbar_ptr, multicast_mask, cache_hint, smem_ptr, crd0, crd1);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
    return SM90_TMA_LOAD_MULTICAST_3D::copy(desc_ptr, mbar_ptr, multicast_mask, cache_hint, smem_ptr, crd0, crd1, crd2);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
    return SM90_TMA_LOAD_MULTICAST_4D::copy(desc_ptr, mbar_ptr, multicast_mask, cache_hint, smem_ptr, crd0, crd1, crd2, crd3);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
    return SM90_TMA_LOAD_MULTICAST_5D::copy(desc_ptr, mbar_ptr, multicast_mask, cache_hint, smem_ptr, crd0, crd1, crd2, crd3, crd4);
  }

  using PREFETCH = typename SM90_TMA_LOAD::PREFETCH;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// TMA_LOAD_MULTICAST im2col: Initiates a TMA copy, in im2col mode, from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_LOAD_IM2COL_MULTICAST_3D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask,
       void      * smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_n,
       uint16_t const& offset_w)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
    // Copy from global to shared::cluster.
    asm volatile (
      "cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%3, %4, %5}], [%2], {%6}, %7;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(coord_c), "r"(coord_w), "r"(coord_n),
        "h"(offset_w),
        "h"(multicast_mask)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_IM2COL_MULTICAST_4D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask,
       void      * smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_n,
       uint16_t const& offset_w, uint16_t const& offset_h)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
    // Copy from global to shared::cluster.
    asm volatile (
      "cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%3, %4, %5, %6}], [%2], {%7, %8}, %9;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_n),
        "h"(offset_w), "h"(offset_h),
        "h"(multicast_mask)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_IM2COL_MULTICAST_5D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask,
       void      * smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_d, int32_t const& coord_n,
       uint16_t const& offset_w, uint16_t const& offset_h, uint16_t const& offset_d)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
    // Copy from global to shared::cluster.
    asm volatile (
      "cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], {%8, %9, %10}, %11;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_d), "r"(coord_n),
        "h"(offset_w), "h"(offset_h), "h"(offset_d),
        "h"(multicast_mask)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_LOAD_IM2COL_MULTICAST
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask,
       void      * smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_n,
       uint16_t const& offset_w)
  {
    return SM90_TMA_LOAD_IM2COL_MULTICAST_3D::copy(desc_ptr, mbar_ptr, multicast_mask,
                                                   smem_ptr,
                                                   coord_c, coord_w, coord_n,
                                                   offset_w);
  }

  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask,
       void      * smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_n,
       uint16_t const& offset_w, uint16_t const& offset_h)
  {
    return SM90_TMA_LOAD_IM2COL_MULTICAST_4D::copy(desc_ptr, mbar_ptr, multicast_mask,
                                                   smem_ptr,
                                                   coord_c, coord_w, coord_h, coord_n,
                                                   offset_w, offset_h);
  }

  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask,
       void      * smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_d, int32_t const& coord_n,
       uint16_t const& offset_w, uint16_t const& offset_h, uint16_t const& offset_d)
  {
    return SM90_TMA_LOAD_IM2COL_MULTICAST_5D::copy(desc_ptr, mbar_ptr, multicast_mask,
                                                   smem_ptr,
                                                   coord_c, coord_w, coord_h, coord_d, coord_n,
                                                   offset_w, offset_h, offset_d);
  }

  using PREFETCH = typename SM90_TMA_LOAD_IM2COL::PREFETCH;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// TMA_STORE : Initiates a TMA copy from shared memory to global memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_STORE_1D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_store(__LINE__, gmem_int_desc, smem_int_ptr);
    asm volatile (
      "cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%0, {%2}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(crd0)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_2D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_store(__LINE__, gmem_int_desc, smem_int_ptr);
    asm volatile (
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(crd0), "r"(crd1)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_3D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_store(__LINE__, gmem_int_desc, smem_int_ptr);
    asm volatile (
      "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(crd0), "r"(crd1), "r"(crd2)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_4D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_store(__LINE__, gmem_int_desc, smem_int_ptr);
    asm volatile (
      "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_5D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_store(__LINE__, gmem_int_desc, smem_int_ptr);
    asm volatile (
      "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5, %6}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0)
  {
    return SM90_TMA_STORE_1D::copy(desc_ptr, smem_ptr, crd0);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
    return SM90_TMA_STORE_2D::copy(desc_ptr, smem_ptr, crd0, crd1);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
    return SM90_TMA_STORE_3D::copy(desc_ptr, smem_ptr, crd0, crd1, crd2);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
    return SM90_TMA_STORE_4D::copy(desc_ptr, smem_ptr, crd0, crd1, crd2, crd3);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
    return SM90_TMA_STORE_5D::copy(desc_ptr, smem_ptr, crd0, crd1, crd2, crd3, crd4);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// TMA_STORE im2col: Initiates a TMA copy, in im2col mode, from shared memory to global memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_STORE_IM2COL_3D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_n)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_store(__LINE__, gmem_int_desc, smem_int_ptr);
    asm volatile (
      "cp.async.bulk.tensor.3d.global.shared::cta.im2col_no_offs.bulk_group"
      " [%0, {%2, %3, %4}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(coord_c), "r"(coord_w), "r"(coord_n)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_IM2COL_4D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_n)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_store(__LINE__, gmem_int_desc, smem_int_ptr);
    asm volatile (
      "cp.async.bulk.tensor.4d.global.shared::cta.im2col_no_offs.bulk_group"
      " [%0, {%2, %3, %4, %5}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_n)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_IM2COL_5D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_d, int32_t const& coord_n)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_store(__LINE__, gmem_int_desc, smem_int_ptr);
    asm volatile (
      "cp.async.bulk.tensor.5d.global.shared::cta.im2col_no_offs.bulk_group"
      " [%0, {%2, %3, %4, %5, %6}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_d), "r"(coord_n)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_IM2COL
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_n)
  {
    return SM90_TMA_STORE_IM2COL_3D::copy(desc_ptr, smem_ptr, coord_c, coord_w, coord_n);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_n)
  {
    return SM90_TMA_STORE_IM2COL_4D::copy(desc_ptr, smem_ptr, coord_c, coord_w, coord_h, coord_n);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& coord_c, int32_t const& coord_w, int32_t const& coord_h, int32_t const& coord_d, int32_t const& coord_n)
  {
    return SM90_TMA_STORE_IM2COL_5D::copy(desc_ptr, smem_ptr, coord_c, coord_w, coord_h, coord_d, coord_n);
  }
};

// Fence for smem stores for subsequent TMA_STORE
CUTE_HOST_DEVICE static void
tma_store_fence() {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    cutlass::arch::synclog_emit_fence_view_async_shared(__LINE__);
    asm volatile ("fence.proxy.async.shared::cta;");
#elif defined(__CUDA_ARCH__)
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}

// Indicate arrival of warp issuing TMA_STORE
CUTE_HOST_DEVICE static void
tma_store_arrive() {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    cutlass::arch::synclog_emit_tma_store_arrive(__LINE__);
    asm volatile("cp.async.bulk.commit_group;");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}


CUTE_HOST_DEVICE static void
tma_desc_commit_group() {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    asm volatile("cp.async.bulk.commit_group;");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}


// Wait until at most Count committed TMA_STOREs are pending and all prior commits are complete
template <int Count>
CUTE_HOST_DEVICE static void
tma_store_wait() {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    asm volatile(
      "cp.async.bulk.wait_group.read %0;"
      :
      : "n"(Count)
      : "memory");
    cutlass::arch::synclog_emit_tma_store_wait(__LINE__, Count);
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}


// Wait until all TMA descriptor previously issued are safe to be modified after tma_desc_commit_group()
CUTE_HOST_DEVICE static void
tma_desc_wait_group() {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    asm volatile(
      "cp.async.bulk.wait_group.read %0;"
      :
      : "n"(0)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// TMA_REDUCE_ADD : Initiates a TMA reduce-add from shared memory to global memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_REDUCE_ADD_1D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_store(__LINE__, gmem_int_desc, smem_int_ptr);
    asm volatile (
      "cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.bulk_group [%0, {%2}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(crd0)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_REDUCE_ADD_2D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_store(__LINE__, gmem_int_desc, smem_int_ptr);
    asm volatile (
      "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group [%0, {%2, %3}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(crd0), "r"(crd1)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_REDUCE_ADD_3D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_store(__LINE__, gmem_int_desc, smem_int_ptr);
    asm volatile (
      "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.bulk_group [%0, {%2, %3, %4}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(crd0), "r"(crd1), "r"(crd2)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_REDUCE_ADD_4D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_store(__LINE__, gmem_int_desc, smem_int_ptr);
    asm volatile (
      "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_REDUCE_ADD_5D
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_store(__LINE__, gmem_int_desc, smem_int_ptr);
    asm volatile (
      "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.bulk_group [%0, {%2, %3, %4, %5, %6}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_int_ptr),
        "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
      : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_REDUCE_ADD
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0)
  {
    return SM90_TMA_REDUCE_ADD_1D::copy(desc_ptr, smem_ptr, crd0);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
    return SM90_TMA_REDUCE_ADD_2D::copy(desc_ptr, smem_ptr, crd0, crd1);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
  {
    return SM90_TMA_REDUCE_ADD_3D::copy(desc_ptr, smem_ptr, crd0, crd1, crd2);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3)
  {
    return SM90_TMA_REDUCE_ADD_4D::copy(desc_ptr, smem_ptr, crd0, crd1, crd2, crd3);
  }
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
    return SM90_TMA_REDUCE_ADD_5D::copy(desc_ptr, smem_ptr, crd0, crd1, crd2, crd3, crd4);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// BULK_COPY : Copy a bulk of memory between shared memory and global memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_BULK_COPY_G2S
{
  CUTE_HOST_DEVICE static void
  copy(void const* gmem_ptr, uint64_t* mbar_ptr,
       void      * smem_ptr, int32_t load_bytes)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :
                     : "r"(smem_int_ptr), "l"(gmem_ptr), "r"(load_bytes), "r"(smem_int_mbar)
                     : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use BULK_COPY without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  struct PREFETCH
  {
    CUTE_HOST_DEVICE static void
    copy(void const* gmem_ptr, int32_t load_bytes)
    {
  #if defined(CUTE_ARCH_TMA_SM90_ENABLED)
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n"
                      :
                      : "l"(gmem_ptr), "r"(load_bytes)
                      : "memory");
  #else
      CUTE_INVALID_CONTROL_PATH("Trying to use BULK_COPY without CUTE_ARCH_TMA_SM90_ENABLED.");
  #endif
    }
  };
};

struct SM90_BULK_COPY_S2G
{
  CUTE_HOST_DEVICE static void
  copy(void const* smem_ptr,
       void      * gmem_ptr, int32_t store_bytes)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                     :
                     : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes)
                     : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use BULK_COPY without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_BULK_COPY_AUTO {};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cute
