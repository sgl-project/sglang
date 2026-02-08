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

#include <cute/arch/mma.hpp>

// Config
#if ((__CUDACC_VER_MAJOR__ > 10) || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1))
#  define CUTE_ARCH_MMA_SM70_SUPPORTED
#  if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))
#    define CUTE_ARCH_MMA_SM70_ENABLED
#  endif
#endif

namespace cute
{

//
// SM70 MMA 884 F16F16F16
//

struct SM70_8x8x4_F16F16F16F16_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM70_ENABLED)
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16"
                 "{%0, %1,  %2,  %3},"
                 "{%4, %5},"
                 "{%6, %7},"
                 "{%8, %9, %10, %11};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        :  "r"(a0),  "r"(a1),
           "r"(b0),  "r"(b1),
           "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM70_8x8x4_F16F16F16F16_TN without CUTE_ARCH_MMA_SM70_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM70_8x8x4_F16F16F16F16_NT
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM70_ENABLED)
    asm volatile("mma.sync.aligned.m8n8k4.col.row.f16.f16.f16.f16"
                 "{%0, %1,  %2,  %3},"
                 "{%4, %5},"
                 "{%6, %7},"
                 "{%8, %9, %10, %11};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        :  "r"(a0),  "r"(a1),
           "r"(b0),  "r"(b1),
           "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM70_8x8x4_F16F16F16F16_NT without CUTE_ARCH_MMA_SM70_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM70_8x8x4_F16F16F16F16_NN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM70_ENABLED)
    asm volatile("mma.sync.aligned.m8n8k4.col.col.f16.f16.f16.f16"
                 "{%0, %1,  %2,  %3},"
                 "{%4, %5},"
                 "{%6, %7},"
                 "{%8, %9, %10, %11};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        :  "r"(a0),  "r"(a1),
           "r"(b0),  "r"(b1),
           "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM70_8x8x4_F16F16F16F16_NN without CUTE_ARCH_MMA_SM70_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM70_8x8x4_F16F16F16F16_TT
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM70_ENABLED)
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16"
                 "{%0, %1,  %2,  %3},"
                 "{%4, %5},"
                 "{%6, %7},"
                 "{%8, %9, %10, %11};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        :  "r"(a0),  "r"(a1),
           "r"(b0),  "r"(b1),
           "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM70_8x8x4_F16F16F16F16_TT without CUTE_ARCH_MMA_SM70_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// SM70 MMA 884 F16F16F32
//

struct SM70_8x8x4_F32F16F16F32_TN
{
  using DRegisters = float[8];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[8];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float      & d2, float      & d3,
      float         & d4, float         & d5, float      & d6, float      & d7,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      float    const& c0, float    const& c1, float const& c2, float const& c3,
      float    const& c4, float    const& c5, float const& c6, float const& c7)
  {
#if defined(CUTE_ARCH_MMA_SM70_ENABLED)
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"
                 "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11},"
                 "{%12, %13, %14, %15, %16, %17, %18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3),
          "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        :  "r"(a0),  "r"(a1),
           "r"(b0),  "r"(b1),
           "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
           "f"(c4),  "f"(c5),  "f"(c6),  "f"(c7));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM70_8x8x4_F32F16F16F32_TN without CUTE_ARCH_MMA_SM70_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM70_8x8x4_F32F16F16F32_NT
{
  using DRegisters = float[8];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[8];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float      & d2, float      & d3,
      float         & d4, float         & d5, float      & d6, float      & d7,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      float    const& c0, float    const& c1, float const& c2, float const& c3,
      float    const& c4, float    const& c5, float const& c6, float const& c7)
  {
#if defined(CUTE_ARCH_MMA_SM70_ENABLED)
    asm volatile("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32"
                 "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11},"
                 "{%12, %13, %14, %15, %16, %17, %18, %19};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3),
          "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        :  "r"(a0),  "r"(a1),
           "r"(b0),  "r"(b1),
           "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
           "f"(c4),  "f"(c5),  "f"(c6),  "f"(c7));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM70_8x8x4_F32F16F16F32_NT without CUTE_ARCH_MMA_SM70_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM70_8x8x4_F32F16F16F32_NN
{
  using DRegisters = float[8];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[8];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float      & d2, float      & d3,
      float         & d4, float         & d5, float      & d6, float      & d7,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      float    const& c0, float    const& c1, float const& c2, float const& c3,
      float    const& c4, float    const& c5, float const& c6, float const& c7)
  {
#if defined(CUTE_ARCH_MMA_SM70_ENABLED)
    asm volatile("mma.sync.aligned.m8n8k4.col.col.f32.f16.f16.f32"
                 "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11},"
                 "{%12, %13, %14, %15, %16, %17, %18, %19};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3),
          "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        :  "r"(a0),  "r"(a1),
           "r"(b0),  "r"(b1),
           "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
           "f"(c4),  "f"(c5),  "f"(c6),  "f"(c7));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM70_8x8x4_F32F16F16F32_NN without CUTE_ARCH_MMA_SM70_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM70_8x8x4_F32F16F16F32_TT
{
  using DRegisters = float[8];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[8];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float      & d2, float      & d3,
      float         & d4, float         & d5, float      & d6, float      & d7,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      float    const& c0, float    const& c1, float const& c2, float const& c3,
      float    const& c4, float    const& c5, float const& c6, float const& c7)
  {
#if defined(CUTE_ARCH_MMA_SM70_ENABLED)
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32"
                 "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11},"
                 "{%12, %13, %14, %15, %16, %17, %18, %19};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3),
          "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        :  "r"(a0),  "r"(a1),
           "r"(b0),  "r"(b1),
           "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
           "f"(c4),  "f"(c5),  "f"(c6),  "f"(c7));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM70_8x8x4_F32F16F16F32_TT without CUTE_ARCH_MMA_SM70_ENABLED");
#endif
  }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cute
