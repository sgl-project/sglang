/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cute/arch/mma.hpp>
#include <cute/numeric/numeric_types.hpp> // cute::float_e4m3_t, etc
#include <cutlass/detail/dependent_false.hpp>

namespace cute {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <class a_type, class b_type, class c_type>
struct SM120_16x8x32_TN
{
  static_assert(cutlass::detail::dependent_false<a_type>, "No MMA matches SM120_16x8x32_TN for given data types.");
};


////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E2M1 x E2M1
template <>
struct SM120_16x8x32_TN<float_e2m1_t, float_e2m1_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E2M1 x E3M2
template <>
struct SM120_16x8x32_TN<float_e2m1_t, float_e3m2_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e3m2.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E2M1 x E2M3
template <>
struct SM120_16x8x32_TN<float_e2m1_t, float_e2m3_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m3.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E2M1 x E4M3
template <>
struct SM120_16x8x32_TN<float_e2m1_t, float_e4m3_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e4m3.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E2M1 x E5M2
template <>
struct SM120_16x8x32_TN<float_e2m1_t, float_e5m2_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e5m2.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// MMA 16x8x32 TN E3M2 x E2M1
template <>
struct SM120_16x8x32_TN<float_e3m2_t, float_e2m1_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e3m2.e2m1.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E3M2 x E3M2
template <>
struct SM120_16x8x32_TN<float_e3m2_t, float_e3m2_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e3m2.e3m2.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E3M2 x E2M3
template <>
struct SM120_16x8x32_TN<float_e3m2_t, float_e2m3_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e3m2.e2m3.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E3M2 x E4M3
template <>
struct SM120_16x8x32_TN<float_e3m2_t, float_e4m3_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e3m2.e4m3.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E3M2 x E5M2
template <>
struct SM120_16x8x32_TN<float_e3m2_t, float_e5m2_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e3m2.e5m2.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////



// MMA 16x8x32 TN E2M3 x E2M1
template <>
struct SM120_16x8x32_TN<float_e2m3_t, float_e2m1_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m3.e2m1.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E2M3 x E3M2
template <>
struct SM120_16x8x32_TN<float_e2m3_t, float_e3m2_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m3.e3m2.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E2M3 x E2M3
template <>
struct SM120_16x8x32_TN<float_e2m3_t, float_e2m3_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m3.e2m3.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E2M3 x E4M3
template <>
struct SM120_16x8x32_TN<float_e2m3_t, float_e4m3_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m3.e4m3.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E2M3 x E5M2
template <>
struct SM120_16x8x32_TN<float_e2m3_t, float_e5m2_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m3.e5m2.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////



// MMA 16x8x32 TN E4M3 x E2M1
template <>
struct SM120_16x8x32_TN<float_e4m3_t, float_e2m1_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e2m1.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E4M3 x E3M2
template <>
struct SM120_16x8x32_TN<float_e4m3_t, float_e3m2_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e3m2.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E4M3 x E2M3
template <>
struct SM120_16x8x32_TN<float_e4m3_t, float_e2m3_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e2m3.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E4M3 x E4M3
template <>
struct SM120_16x8x32_TN<float_e4m3_t, float_e4m3_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E4M3 x E5M2
template <>
struct SM120_16x8x32_TN<float_e4m3_t, float_e5m2_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////



// MMA 16x8x32 TN E5M2 x E2M1
template <>
struct SM120_16x8x32_TN<float_e5m2_t, float_e2m1_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e5m2.e2m1.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E5M2 x E3M2
template <>
struct SM120_16x8x32_TN<float_e5m2_t, float_e3m2_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e5m2.e3m2.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E5M2 x E2M3
template <>
struct SM120_16x8x32_TN<float_e5m2_t, float_e2m3_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e5m2.e2m3.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E5M2 x E4M3
template <>
struct SM120_16x8x32_TN<float_e5m2_t, float_e4m3_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e5m2.e4m3.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN E5M2 x E5M2
template <>
struct SM120_16x8x32_TN<float_e5m2_t, float_e5m2_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////


// MMA.F16 16x8x32 TN E2M1 x E2M1
template <>
struct SM120_16x8x32_TN<float_e2m1_t, float_e2m1_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e2m1.e2m1.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E2M1 x E3M2
template <>
struct SM120_16x8x32_TN<float_e2m1_t, float_e3m2_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e2m1.e3m2.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E2M1 x E2M3
template <>
struct SM120_16x8x32_TN<float_e2m1_t, float_e2m3_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e2m1.e2m3.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E2M1 x E4M3
template <>
struct SM120_16x8x32_TN<float_e2m1_t, float_e4m3_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e2m1.e4m3.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E2M1 x E5M2
template <>
struct SM120_16x8x32_TN<float_e2m1_t, float_e5m2_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e2m1.e5m2.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E3M2 x E2M1
template <>
struct SM120_16x8x32_TN<float_e3m2_t, float_e2m1_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e3m2.e2m1.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E3M2 x E3M2
template <>
struct SM120_16x8x32_TN<float_e3m2_t, float_e3m2_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e3m2.e3m2.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E3M2 x E2M3
template <>
struct SM120_16x8x32_TN<float_e3m2_t, float_e2m3_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e3m2.e2m3.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E3M2 x E4M3
template <>
struct SM120_16x8x32_TN<float_e3m2_t, float_e4m3_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e3m2.e4m3.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E3M2 x E5M2
template <>
struct SM120_16x8x32_TN<float_e3m2_t, float_e5m2_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e3m2.e5m2.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E2M3 x E2M1
template <>
struct SM120_16x8x32_TN<float_e2m3_t, float_e2m1_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e2m3.e2m1.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E2M3 x E3M2
template <>
struct SM120_16x8x32_TN<float_e2m3_t, float_e3m2_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e2m3.e3m2.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E2M3 x E2M3
template <>
struct SM120_16x8x32_TN<float_e2m3_t, float_e2m3_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e2m3.e2m3.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E2M3 x E4M3
template <>
struct SM120_16x8x32_TN<float_e2m3_t, float_e4m3_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e2m3.e4m3.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E2M3 x E5M2
template <>
struct SM120_16x8x32_TN<float_e2m3_t, float_e5m2_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e2m3.e5m2.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////


// MMA.F16 16x8x32 TN E4M3 x E2M1
template <>
struct SM120_16x8x32_TN<float_e4m3_t, float_e2m1_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e4m3.e2m1.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E4M3 x E3M2
template <>
struct SM120_16x8x32_TN<float_e4m3_t, float_e3m2_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e4m3.e3m2.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E4M3 x E2M3
template <>
struct SM120_16x8x32_TN<float_e4m3_t, float_e2m3_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e4m3.e2m3.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E4M3 x E4M3
template <>
struct SM120_16x8x32_TN<float_e4m3_t, float_e4m3_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E4M3 x E5M2
template <>
struct SM120_16x8x32_TN<float_e4m3_t, float_e5m2_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e4m3.e5m2.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E5M2 x E2M1
template <>
struct SM120_16x8x32_TN<float_e5m2_t, float_e2m1_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e5m2.e2m1.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E5M2 x E3M2
template <>
struct SM120_16x8x32_TN<float_e5m2_t, float_e3m2_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e5m2.e3m2.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E5M2 x E2M3
template <>
struct SM120_16x8x32_TN<float_e5m2_t, float_e2m3_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e5m2.e2m3.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E5M2 x E4M3
template <>
struct SM120_16x8x32_TN<float_e5m2_t, float_e4m3_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e5m2.e4m3.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.F16 16x8x32 TN E5M2 x E5M2
template <>
struct SM120_16x8x32_TN<float_e5m2_t, float_e5m2_t, half_t>
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e5m2.e5m2.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace SM120::BLOCKSCALED {
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class a_type, class b_type, class c_type, class sf_type, int VS>
struct SM120_16x8x32_TN_VS
{
  static_assert(cutlass::detail::dependent_false<a_type>, "No MMA matches SM120_16x8x32_TN_VS for given data types.");
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <class a_type, class b_type, class c_type, class sf_type, int VS>
struct SM120_16x8x64_TN_VS
{
  static_assert(cutlass::detail::dependent_false<a_type>, "No MMA matches SM120_16x8x64_TN_VS for given data types.");
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E2M1 x E2M1 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)), "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E2M1 x E3M2 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e2m1_t, float_e3m2_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];
 
  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e3m2.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E2M1 x E2M3 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e2m1_t, float_e2m3_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];
 
  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m3.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E2M1 x E4M3 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e2m1_t, float_e4m3_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];
 
  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E2M1 x E5M2 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e2m1_t, float_e5m2_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e5m2.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E3M2 x E2M1 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e3m2_t, float_e2m1_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e3m2.e2m1.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E3M2 x E3M2 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e3m2_t, float_e3m2_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e3m2.e3m2.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E3M2 x E2M3 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e3m2_t, float_e2m3_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e3m2.e2m3.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E3M2 x E4M3 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e3m2_t, float_e4m3_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e3m2.e4m3.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E3M2 x E5M2 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e3m2_t, float_e5m2_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e3m2.e5m2.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E2M3 x E2M1 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e2m3_t, float_e2m1_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m3.e2m1.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E2M3 x E3M2 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e2m3_t, float_e3m2_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m3.e3m2.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E2M3 x E2M3 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e2m3_t, float_e2m3_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m3.e2m3.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E2M3 x E4M3 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e2m3_t, float_e4m3_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m3.e4m3.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E2M3 x E5M2 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e2m3_t, float_e5m2_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m3.e5m2.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E4M3 x E2M1 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e4m3_t, float_e2m1_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E4M3 x E3M2 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e4m3_t, float_e3m2_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e3m2.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E4M3 x E2M3 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e4m3_t, float_e2m3_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m3.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E4M3 x E4M3 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e4m3_t, float_e4m3_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E4M3 x E5M2 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e4m3_t, float_e5m2_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e5m2.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E5M2 x E2M1 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e5m2_t, float_e2m1_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e5m2.e2m1.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E5M2 x E3M2 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e5m2_t, float_e3m2_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e5m2.e3m2.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E5M2 x E2M3 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e5m2_t, float_e2m3_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e5m2.e2m3.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E5M2 x E4M3 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e5m2_t, float_e4m3_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e5m2.e4m3.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x32 TN E5M2 x E5M2 with SF UE8M0
template <int VS>
struct SM120_16x8x32_TN_VS<float_e5m2_t, float_e5m2_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      uint8_t const& sfa0,
      uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e5m2.e5m2.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x32_TN_VS without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA.SF 16x8x64 TN E2M1 x E2M1 with SF UE8M0
template <int VS>
struct SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  static constexpr int SFBits = (VS == 32) ? 16 : 32;
  using RegTypeSF = uint_bit_t<SFBits>;

  using SFARegisters = RegTypeSF[1];
  using SFBRegisters = RegTypeSF[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      RegTypeSF const& sfa0,
      RegTypeSF const& sfb0)
  {
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;
    CUTE_STATIC_ASSERT(VS == 16 || VS == 32, "Scaling factor vector size has to be 16 or 32 for MXF4NVF4 MMA.");
    
#if defined(CUTE_ARCH_MXF4NVF4_2X_UE8M0_MMA_ENABLED)
      asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13},"
      "{%14},"
      "{%15, %16},"
      "{%17},"
      "{%18, %19};\n"
      :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
      :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
          "r"(b0),   "r"(b1),
          "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
          "r"(uint32_t(sfa0)),  "h"(bidA), "h"(tidA),
          "r"(uint32_t(sfb0)),  "h"(bidB), "h"(tidB));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x64_TN_VS without CUTE_ARCH_MXF4NVF4_2X_UE8M0_MMA_ENABLED");
#endif
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////


// MMA.SF 16x8x64 TN E2M1 x E2M1 with SF E4M3
template <int VS>
struct SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue4m3_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  static constexpr int SFBits = (VS == 32) ? 16 : 32;
  using RegTypeSF = uint_bit_t<SFBits>;

  using SFARegisters = RegTypeSF[1];
  using SFBRegisters = RegTypeSF[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      RegTypeSF const& sfa0,
      RegTypeSF const& sfb0)
  {
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

 CUTE_STATIC_ASSERT(VS == 16 || VS == 32, "Scaling factor vector size has to be 16 or 32 for MXF4NVF4 MMA.");
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
    asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x64_TN_VS without CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED");
#endif

  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace SM120::BLOCKSCALED


////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class ElementA,
  class ElementB,
  class ElementC
>
CUTE_HOST_DEVICE constexpr
auto
rr_op_selector_sm120()
{
  return SM120_16x8x32_TN<ElementA, ElementB, ElementC>{};
}

template <
  class ElementA,
  class ElementB,
  class ElementC,
  class ElementSF,
  int   SFVecSize,
  bool  UseF8F6F4
>
CUTE_HOST_DEVICE constexpr
auto
rr_blockscaled_op_selector_sm120()
{
  if constexpr (UseF8F6F4) {
    return SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<ElementA, ElementB, ElementC, ElementSF, SFVecSize>{};
  }
  else{
    return SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<ElementA, ElementB, ElementC, ElementSF, SFVecSize>{};
  }
}

} // namespace cute
