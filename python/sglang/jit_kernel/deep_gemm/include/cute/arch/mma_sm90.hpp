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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#    define CUTE_ARCH_MMA_SM90_ENABLED
#    define CUTE_ARCH_MMA_F64_SM90_ENABLED
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cute {

namespace SM90 {

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x4 TN
struct MMA_16x8x4_F64F64F64F64_TN
{
  using DRegisters = double[4];
  using ARegisters = double[2];
  using BRegisters = double[1];
  using CRegisters = double[4];

  CUTE_HOST_DEVICE static void
  fma(double      & d0, double      & d1, double      & d2, double      & d3,
      double const& a0, double const& a1,
      double const& b0,
      double const& c0, double const& c1, double const& c2, double const& c3)
  {
#if defined(CUTE_ARCH_MMA_F64_SM90_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64"
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=d"(d0), "=d"(d1), "=d"(d2), "=d"(d3)
      :  "d"(a0),  "d"(a1),
         "d"(b0),
         "d"(c0),  "d"(c1),  "d"(c2),  "d"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use MMA_16x8x4_F64F64F64F64_TN without CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x8 TN
struct MMA_16x8x8_F64F64F64F64_TN
{
  using DRegisters = double[4];
  using ARegisters = double[4];
  using BRegisters = double[2];
  using CRegisters = double[4];

  CUTE_HOST_DEVICE static void
  fma(double      & d0, double      & d1, double      & d2, double      & d3,
      double const& a0, double const& a1, double const& a2, double const& a3,
      double const& b0, double const& b1,
      double const& c0, double const& c1, double const& c2, double const& c3)
  {
#if defined(CUTE_ARCH_MMA_F64_SM90_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64"
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=d"(d0), "=d"(d1), "=d"(d2), "=d"(d3)
      :  "d"(a0),  "d"(a1),  "d"(a2),  "d"(a3),
         "d"(b0),  "d"(b1),
         "d"(c0),  "d"(c1),  "d"(c2),  "d"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use MMA_16x8x8_F64F64F64F64_TN without CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct MMA_16x8x16_F64F64F64F64_TN
{
  using DRegisters = double[4];
  using ARegisters = double[8];
  using BRegisters = double[4];
  using CRegisters = double[4];

  CUTE_HOST_DEVICE static void
  fma(double      & d0, double      & d1, double      & d2, double      & d3,
      double const& a0, double const& a1, double const& a2, double const& a3,
      double const& a4, double const& a5, double const& a6, double const& a7,
      double const& b0, double const& b1, double const& b2, double const& b3,
      double const& c0, double const& c1, double const& c2, double const& c3)
  {
#if defined(CUTE_ARCH_MMA_F64_SM90_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64"
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7,  %8,  %9,  %10, %11},"
      "{%12, %13, %14, %15},"
      "{%16, %17, %18, %19};\n"
      : "=d"(d0), "=d"(d1), "=d"(d2), "=d"(d3)
      :  "d"(a0),  "d"(a1),  "d"(a2),  "d"(a3),
         "d"(a4),  "d"(a5),  "d"(a6),  "d"(a7),
         "d"(b0),  "d"(b1),  "d"(b2),  "d"(b3),
         "d"(c0),  "d"(c1),  "d"(c2),  "d"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use MMA_16x8x16_F64F64F64F64_TN without CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x4 TN
struct MMA_16x8x4_C64C64C64C64_TN
{
  using DRegisters = complex<double>[4];
  using ARegisters = complex<double>[2];
  using BRegisters = complex<double>[1];
  using CRegisters = complex<double>[4];

  CUTE_HOST_DEVICE static void
  fma(complex<double>      & d0, complex<double>      & d1,
      complex<double>      & d2, complex<double>      & d3,
      complex<double> const& a0, complex<double> const& a1,
      complex<double> const& b0,
      complex<double> const& c0, complex<double> const& c1,
      complex<double> const& c2, complex<double> const& c3)
  {
    // Because thrust::complex does not provide a mutable ref
    double& rd0 = reinterpret_cast<double(&)[2]>(d0)[0];
    double& id0 = reinterpret_cast<double(&)[2]>(d0)[1];
    double& rd1 = reinterpret_cast<double(&)[2]>(d1)[0];
    double& id1 = reinterpret_cast<double(&)[2]>(d1)[1];
    double& rd2 = reinterpret_cast<double(&)[2]>(d2)[0];
    double& id2 = reinterpret_cast<double(&)[2]>(d2)[1];
    double& rd3 = reinterpret_cast<double(&)[2]>(d3)[0];
    double& id3 = reinterpret_cast<double(&)[2]>(d3)[1];

    // d.real() =  a.real() * b.real() + c.real();
    MMA_16x8x4_F64F64F64F64_TN::fma(
      rd0, rd1, rd2, rd3,
      a0.real(), a1.real(),
      b0.real(),
      c0.real(), c1.real(), c2.real(), c3.real());

    // d.imag() =  a.imag() * b.real() + c.imag();
    MMA_16x8x4_F64F64F64F64_TN::fma(
      id0, id1, id2, id3,
      a0.imag(), a1.imag(),
      b0.real(),
      c0.imag(), c1.imag(), c2.imag(), c3.imag());

    // d.real() = -a.imag() * b.imag() + d.real();
    MMA_16x8x4_F64F64F64F64_TN::fma(
      rd0, rd1, rd2, rd3,
      -a0.imag(), -a1.imag(),
      b0.imag(),
      d0.real(), d1.real(), d2.real(), d3.real());

    // d.imag() =  a.real() * b.imag() + d.imag();
    MMA_16x8x4_F64F64F64F64_TN::fma(
      id0, id1, id2, id3,
      a0.real(), a1.real(),
      b0.imag(),
      d0.imag(), d1.imag(), d2.imag(), d3.imag());
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x8 TN
struct MMA_16x8x8_C64C64C64C64_TN
{
  using DRegisters = complex<double>[4];
  using ARegisters = complex<double>[4];
  using BRegisters = complex<double>[2];
  using CRegisters = complex<double>[4];

  CUTE_HOST_DEVICE static void
  fma(complex<double>      & d0, complex<double>      & d1,
      complex<double>      & d2, complex<double>      & d3,
      complex<double> const& a0, complex<double> const& a1,
      complex<double> const& a2, complex<double> const& a3,
      complex<double> const& b0, complex<double> const& b1,
      complex<double> const& c0, complex<double> const& c1,
      complex<double> const& c2, complex<double> const& c3)
  {
    // Because thrust::complex does not provide a mutable ref
    double& rd0 = reinterpret_cast<double(&)[2]>(d0)[0];
    double& id0 = reinterpret_cast<double(&)[2]>(d0)[1];
    double& rd1 = reinterpret_cast<double(&)[2]>(d1)[0];
    double& id1 = reinterpret_cast<double(&)[2]>(d1)[1];
    double& rd2 = reinterpret_cast<double(&)[2]>(d2)[0];
    double& id2 = reinterpret_cast<double(&)[2]>(d2)[1];
    double& rd3 = reinterpret_cast<double(&)[2]>(d3)[0];
    double& id3 = reinterpret_cast<double(&)[2]>(d3)[1];

    // d.real() =  a.real() * b.real() + c.real();
    MMA_16x8x8_F64F64F64F64_TN::fma(
      rd0, rd1, rd2, rd3,
      a0.real(), a1.real(), a2.real(), a3.real(),
      b0.real(), b1.real(),
      c0.real(), c1.real(), c2.real(), c3.real());

    // d.imag() =  a.imag() * b.real() + c.imag();
    MMA_16x8x8_F64F64F64F64_TN::fma(
      id0, id1, id2, id3,
      a0.imag(), a1.imag(), a2.imag(), a3.imag(),
      b0.real(), b1.real(),
      c0.imag(), c1.imag(), c2.imag(), c3.imag());

    // d.real() = -a.imag() * b.imag() + d.real();
    MMA_16x8x8_F64F64F64F64_TN::fma(
      rd0, rd1, rd2, rd3,
      -a0.imag(), -a1.imag(), -a2.imag(), -a3.imag(),
      b0.imag(), b1.imag(),
      d0.real(), d1.real(), d2.real(), d3.real());

    // d.imag() =  a.real() * b.imag() + d.imag();
    MMA_16x8x8_F64F64F64F64_TN::fma(
      id0, id1, id2, id3,
      a0.real(), a1.real(), a2.real(), a3.real(),
      b0.imag(), b1.imag(),
      d0.imag(), d1.imag(), d2.imag(), d3.imag());
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct MMA_16x8x16_C64C64C64C64_TN
{
  using DRegisters = complex<double>[4];
  using ARegisters = complex<double>[8];
  using BRegisters = complex<double>[4];
  using CRegisters = complex<double>[4];

  CUTE_HOST_DEVICE static void
  fma(complex<double>      & d0, complex<double>      & d1,
      complex<double>      & d2, complex<double>      & d3,
      complex<double> const& a0, complex<double> const& a1,
      complex<double> const& a2, complex<double> const& a3,
      complex<double> const& a4, complex<double> const& a5,
      complex<double> const& a6, complex<double> const& a7,
      complex<double> const& b0, complex<double> const& b1,
      complex<double> const& b2, complex<double> const& b3,
      complex<double> const& c0, complex<double> const& c1,
      complex<double> const& c2, complex<double> const& c3)
  {
    // Because thrust::complex does not provide a mutable ref
    double& rd0 = reinterpret_cast<double(&)[2]>(d0)[0];
    double& id0 = reinterpret_cast<double(&)[2]>(d0)[1];
    double& rd1 = reinterpret_cast<double(&)[2]>(d1)[0];
    double& id1 = reinterpret_cast<double(&)[2]>(d1)[1];
    double& rd2 = reinterpret_cast<double(&)[2]>(d2)[0];
    double& id2 = reinterpret_cast<double(&)[2]>(d2)[1];
    double& rd3 = reinterpret_cast<double(&)[2]>(d3)[0];
    double& id3 = reinterpret_cast<double(&)[2]>(d3)[1];

    // d.real() =  a.real() * b.real() + c.real();
    MMA_16x8x16_F64F64F64F64_TN::fma(
      rd0, rd1, rd2, rd3,
      a0.real(), a1.real(), a2.real(), a3.real(),
      a4.real(), a5.real(), a6.real(), a7.real(),
      b0.real(), b1.real(), b2.real(), b3.real(),
      c0.real(), c1.real(), c2.real(), c3.real());

    // d.imag() =  a.imag() * b.real() + c.imag();
    MMA_16x8x16_F64F64F64F64_TN::fma(
      id0, id1, id2, id3,
      a0.imag(), a1.imag(), a2.imag(), a3.imag(),
      a4.imag(), a5.imag(), a6.imag(), a7.imag(),
      b0.real(), b1.real(), b2.real(), b3.real(),
      c0.imag(), c1.imag(), c2.imag(), c3.imag());

    // d.real() = -a.imag() * b.imag() + d.real();
    MMA_16x8x16_F64F64F64F64_TN::fma(
      rd0, rd1, rd2, rd3,
      -a0.imag(), -a1.imag(), -a2.imag(), -a3.imag(),
      -a4.imag(), -a5.imag(), -a6.imag(), -a7.imag(),
      b0.imag(), b1.imag(), b2.imag(), b3.imag(),
      d0.real(), d1.real(), d2.real(), d3.real());

    // d.imag() =  a.real() * b.imag() + d.imag();
    MMA_16x8x16_F64F64F64F64_TN::fma(
      id0, id1, id2, id3,
      a0.real(), a1.real(), a2.real(), a3.real(),
      a4.real(), a5.real(), a6.real(), a7.real(),
      b0.imag(), b1.imag(), b2.imag(), b3.imag(),
      d0.imag(), d1.imag(), d2.imag(), d3.imag());
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}

} // namespace cute

////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_gmma_sparse.hpp>
#include <cute/layout.hpp>                     // cute::size
#include <cute/numeric/integral_constant.hpp>  // cute::is_static
#include <cute/numeric/numeric_types.hpp>      // cute::half_t, cute::float_e4m3_t, cute::tfloat32_t, etc
#include <cute/util/type_traits.hpp>           // cute::is_same_v

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cute {
namespace SM90::GMMA {

template <
  class ElementA,
  class ElementB,
  class ElementC,
  class TileShape_MNK,
  GMMA::Major MajorA = GMMA::Major::K,
  GMMA::Major MajorB = GMMA::Major::K,
  auto... Args                         // e.g. GMMA::ScaleOut::One, [GMMA::ScaleIn::One, GMMA::ScaleIn::One]
                                       // But most commonly leave empty for defaults
>
CUTE_HOST_DEVICE constexpr
auto
ss_op_selector()
{
  static_assert(is_static<TileShape_MNK>::value, "TileShape_MNK must be static.");
  static_assert(rank(TileShape_MNK{}) == 3, "TileShape_MNK must be rank 3.");
  static_assert(size<0>(TileShape_MNK{}) % 64 == 0, "Tile_M must be a multiple of 64.");
  auto Tile_N = size<1>(TileShape_MNK{});

  // F16 accumulator
  if constexpr (is_same_v<ElementC, half_t>) {

    // Input A: half_t ; Input B: half_t
    if constexpr (is_same_v<ElementA, half_t> && is_same_v<ElementB, half_t>) {
      static_assert(size<2>(TileShape_MNK{}) % 16 == 0, "Tile_K must be a multiple of 16.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x16_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e4m3_t ; Input B: float_e4m3_t
    else if constexpr (is_same_v<ElementA, float_e4m3_t> && is_same_v<ElementB, float_e4m3_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_F16E4M3E4M3_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e4m3_t ; Input B: float_e5m2_t
    else if constexpr (is_same_v<ElementA, float_e4m3_t> && is_same_v<ElementB, float_e5m2_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_F16E4M3E5M2_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e5m2_t ; Input B: float_e4m3_t
    else if constexpr (is_same_v<ElementA, float_e5m2_t> && is_same_v<ElementB, float_e4m3_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_F16E5M2E4M3_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e5m2_t ; Input B: float_e5m2_t
    else if constexpr (is_same_v<ElementA, float_e5m2_t> && is_same_v<ElementB, float_e5m2_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_F16E5M2E5M2_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    else {
      static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
    }
  }

  // F32 accumulator
  else if constexpr (is_same_v<ElementC, float>) {

    // Input A: half_t ; Input B: half_t
    if constexpr (is_same_v<ElementA, half_t> && is_same_v<ElementB, half_t>) {
      static_assert(size<2>(TileShape_MNK{}) % 16 == 0, "Tile_K must be a multiple of 16.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x16_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: bfloat16_t ; Input B: bfloat16_t
    else if constexpr (is_same_v<ElementA, bfloat16_t> && is_same_v<ElementB, bfloat16_t>) {
      static_assert(size<2>(TileShape_MNK{}) % 16 == 0, "Tile_K must be a multiple of 16.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x16_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: tfloat32_t ; Input B: tfloat32_t
    else if constexpr (is_same_v<ElementA, tfloat32_t> && is_same_v<ElementB, tfloat32_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 8 == 0, "Tile_K must be a multiple of 8.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x8_F32TF32TF32_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x8_F32TF32TF32_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x8_F32TF32TF32_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x8_F32TF32TF32_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x8_F32TF32TF32_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x8_F32TF32TF32_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x8_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x8_F32TF32TF32_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x8_F32TF32TF32_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e4m3_t ; Input B: float_e4m3_t
    else if constexpr (is_same_v<ElementA, float_e4m3_t> && is_same_v<ElementB, float_e4m3_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_F32E4M3E4M3_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e4m3_t ; Input B: float_e5m2_t
    else if constexpr (is_same_v<ElementA, float_e4m3_t> && is_same_v<ElementB, float_e5m2_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_F32E4M3E5M2_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e5m2_t ; Input B: float_e4m3_t
    else if constexpr (is_same_v<ElementA, float_e5m2_t> && is_same_v<ElementB, float_e4m3_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_F32E5M2E4M3_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e5m2_t ; Input B: float_e5m2_t
    else if constexpr (is_same_v<ElementA, float_e5m2_t> && is_same_v<ElementB, float_e5m2_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_F32E5M2E5M2_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    else {
      static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
    }
  }

  // S32 accumulator
  else if constexpr (is_same_v<ElementC, int32_t>) {

    // Input A: int8_t ; Input B: int8_t
    if constexpr (is_same_v<ElementA, int8_t> && is_same_v<ElementB, int8_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_S32S8S8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_S32S8S8_SS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_S32S8S8_SS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_S32S8S8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_S32S8S8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_S32S8S8_SS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_S32S8S8_SS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_S32S8S8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_S32S8S8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_S32S8S8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_S32S8S8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_S32S8S8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_S32S8S8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_S32S8S8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_S32S8S8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_S32S8S8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_S32S8S8_SS_TN{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_S32S8S8_SS_TN{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: int8_t ; Input B: uint8_t
    else if constexpr (is_same_v<ElementA, int8_t> && is_same_v<ElementB, uint8_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_S32S8U8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_S32S8U8_SS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_S32S8U8_SS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_S32S8U8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_S32S8U8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_S32S8U8_SS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_S32S8U8_SS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_S32S8U8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_S32S8U8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_S32S8U8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_S32S8U8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_S32S8U8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_S32S8U8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_S32S8U8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_S32S8U8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_S32S8U8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_S32S8U8_SS_TN{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_S32S8U8_SS_TN{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: uint8_t ; Input B: int8_t
    else if constexpr (is_same_v<ElementA, uint8_t> && is_same_v<ElementB, int8_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_S32U8S8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_S32U8S8_SS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_S32U8S8_SS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_S32U8S8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_S32U8S8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_S32U8S8_SS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_S32U8S8_SS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_S32U8S8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_S32U8S8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_S32U8S8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_S32U8S8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_S32U8S8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_S32U8S8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_S32U8S8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_S32U8S8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_S32U8S8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_S32U8S8_SS_TN{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_S32U8S8_SS_TN{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: uint8_t ; Input B: uint8_t
    else if constexpr (is_same_v<ElementA, uint8_t> && is_same_v<ElementB, uint8_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_S32U8U8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_S32U8U8_SS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_S32U8U8_SS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_S32U8U8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_S32U8U8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_S32U8U8_SS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_S32U8U8_SS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_S32U8U8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_S32U8U8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_S32U8U8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_S32U8U8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_S32U8U8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_S32U8U8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_S32U8U8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_S32U8U8_SS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_S32U8U8_SS_TN{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_S32U8U8_SS_TN{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_S32U8U8_SS_TN{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    else {
      static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
    }
  }

  // Unknown accumulator type
  else {
    static_assert(sizeof(ElementC) == 0, "Unknown ElementC accumulator type.");
  }
}

template <
  class ElementA,
  class ElementB,
  class ElementC,
  class TileShape_MNK,
  GMMA::Major MajorA = GMMA::Major::K,
  GMMA::Major MajorB = GMMA::Major::K,
  auto... Args                         // e.g. GMMA::ScaleOut::One, [GMMA::ScaleIn::One, GMMA::ScaleIn::One]
                                       // But most commonly leave empty for defaults
>
CUTE_HOST_DEVICE constexpr
auto
ss_op_selector_sparse()
{
  static_assert(is_static<TileShape_MNK>::value, "TileShape_MNK must be static.");
  static_assert(rank(TileShape_MNK{}) == 3, "TileShape_MNK must be rank 3.");
  static_assert(size<0>(TileShape_MNK{}) % 64 == 0, "Tile_M must be a multiple of 64.");
  auto Tile_N = size<1>(TileShape_MNK{});

  // F16 accumulator
  if constexpr (is_same_v<ElementC, half_t>) {

    // Input A: half_t ; Input B: half_t
    if constexpr (is_same_v<ElementA, half_t> && is_same_v<ElementB, half_t>) {
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x32_F16F16F16_SS<MajorA, MajorB, Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e4m3_t ; Input B: float_e4m3_t
    else if constexpr (is_same_v<ElementA, float_e4m3_t> && is_same_v<ElementB, float_e4m3_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_F16E4M3E4M3_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e4m3_t ; Input B: float_e5m2_t
    else if constexpr (is_same_v<ElementA, float_e4m3_t> && is_same_v<ElementB, float_e5m2_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_F16E4M3E5M2_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e5m2_t ; Input B: float_e4m3_t
    else if constexpr (is_same_v<ElementA, float_e5m2_t> && is_same_v<ElementB, float_e4m3_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_F16E5M2E4M3_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e5m2_t ; Input B: float_e5m2_t
    else if constexpr (is_same_v<ElementA, float_e5m2_t> && is_same_v<ElementB, float_e5m2_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_F16E5M2E5M2_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    else {
      static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
    }
  }

  // F32 accumulator
  else if constexpr (is_same_v<ElementC, float>) {

    // Input A: half_t ; Input B: half_t
    if constexpr (is_same_v<ElementA, half_t> && is_same_v<ElementB, half_t>) {
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x32_F32F16F16_SS<MajorA, MajorB, Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: bfloat16_t ; Input B: bfloat16_t
    else if constexpr (is_same_v<ElementA, bfloat16_t> && is_same_v<ElementB, bfloat16_t>) {
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x32_F32BF16BF16_SS<MajorA, MajorB, Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: tfloat32_t ; Input B: tfloat32_t
    else if constexpr (is_same_v<ElementA, tfloat32_t> && is_same_v<ElementB, tfloat32_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 16 == 0, "Tile_K must be a multiple of 16.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x16_F32TF32TF32_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x16_F32TF32TF32_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x16_F32TF32TF32_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x16_F32TF32TF32_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x16_F32TF32TF32_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x16_F32TF32TF32_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x16_F32TF32TF32_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x16_F32TF32TF32_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x16_F32TF32TF32_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e4m3_t ; Input B: float_e4m3_t
    else if constexpr (is_same_v<ElementA, float_e4m3_t> && is_same_v<ElementB, float_e4m3_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_F32E4M3E4M3_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e4m3_t ; Input B: float_e5m2_t
    else if constexpr (is_same_v<ElementA, float_e4m3_t> && is_same_v<ElementB, float_e5m2_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_F32E4M3E5M2_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e5m2_t ; Input B: float_e4m3_t
    else if constexpr (is_same_v<ElementA, float_e5m2_t> && is_same_v<ElementB, float_e4m3_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_F32E5M2E4M3_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e5m2_t ; Input B: float_e5m2_t
    else if constexpr (is_same_v<ElementA, float_e5m2_t> && is_same_v<ElementB, float_e5m2_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_F32E5M2E5M2_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    else {
      static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
    }
  }

  // S32 accumulator
  else if constexpr (is_same_v<ElementC, int32_t>) {

    // Input A: int8_t ; Input B: int8_t
    if constexpr (is_same_v<ElementA, int8_t> && is_same_v<ElementB, int8_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_S32S8S8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_S32S8S8_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_S32S8S8_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_S32S8S8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_S32S8S8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_S32S8S8_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_S32S8S8_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_S32S8S8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_S32S8S8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_S32S8S8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_S32S8S8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_S32S8S8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_S32S8S8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_S32S8S8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_S32S8S8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_S32S8S8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_S32S8S8_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_S32S8S8_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: int8_t ; Input B: uint8_t
    else if constexpr (is_same_v<ElementA, int8_t> && is_same_v<ElementB, uint8_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_S32S8U8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_S32S8U8_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_S32S8U8_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_S32S8U8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_S32S8U8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_S32S8U8_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_S32S8U8_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_S32S8U8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_S32S8U8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_S32S8U8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_S32S8U8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_S32S8U8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_S32S8U8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_S32S8U8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_S32S8U8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_S32S8U8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_S32S8U8_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_S32S8U8_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: uint8_t ; Input B: int8_t
    else if constexpr (is_same_v<ElementA, uint8_t> && is_same_v<ElementB, int8_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_S32U8S8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_S32U8S8_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_S32U8S8_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_S32U8S8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_S32U8S8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_S32U8S8_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_S32U8S8_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_S32U8S8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_S32U8S8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_S32U8S8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_S32U8S8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_S32U8S8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_S32U8S8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_S32U8S8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_S32U8S8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_S32U8S8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_S32U8S8_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_S32U8S8_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: uint8_t ; Input B: uint8_t
    else if constexpr (is_same_v<ElementA, uint8_t> && is_same_v<ElementB, uint8_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_S32U8U8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_S32U8U8_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_S32U8U8_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_S32U8U8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_S32U8U8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_S32U8U8_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_S32U8U8_SS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_S32U8U8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_S32U8U8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_S32U8U8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_S32U8U8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_S32U8U8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_S32U8U8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_S32U8U8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_S32U8U8_SS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_S32U8U8_SS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_S32U8U8_SS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_S32U8U8_SS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    else {
      static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
    }
  }

  // Unknown accumulator type
  else {
    static_assert(sizeof(ElementC) == 0, "Unknown ElementC accumulator type.");
  }
}

template <
  class ElementA,
  class ElementB,
  class ElementC,
  class TileShape_MNK,
  GMMA::Major MajorA = GMMA::Major::K,
  GMMA::Major MajorB = GMMA::Major::K,
  auto... Args                         // e.g. GMMA::ScaleOut::One, [GMMA::ScaleIn::One, GMMA::ScaleIn::One]
                                       // But most commonly leave empty for defaults
>
CUTE_HOST_DEVICE constexpr
auto
rs_op_selector()
{
  static_assert(is_static<TileShape_MNK>::value, "TileShape_MNK must be static.");
  static_assert(rank(TileShape_MNK{}) == 3, "TileShape_MNK must be rank 3.");
  static_assert(size<0>(TileShape_MNK{}) % 64 == 0, "Tile_M must be a multiple of 64.");
  static_assert(MajorA == GMMA::Major::K, "Register source A operand GMMAs must have K-major A layout.");
  auto Tile_N = size<1>(TileShape_MNK{});

  // F16 accumulator
  if constexpr (is_same_v<ElementC, half_t>) {

    // Input A: half_t ; Input B: half_t
    if constexpr (is_same_v<ElementA, half_t> && is_same_v<ElementB, half_t>) {
      static_assert(size<2>(TileShape_MNK{}) % 16 == 0, "Tile_K must be a multiple of 16.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x16_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e4m3_t ; Input B: float_e4m3_t
    else if constexpr (is_same_v<ElementA, float_e4m3_t> && is_same_v<ElementB, float_e4m3_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_F16E4M3E4M3_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e4m3_t ; Input B: float_e5m2_t
    else if constexpr (is_same_v<ElementA, float_e4m3_t> && is_same_v<ElementB, float_e5m2_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_F16E4M3E5M2_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e5m2_t ; Input B: float_e4m3_t
    else if constexpr (is_same_v<ElementA, float_e5m2_t> && is_same_v<ElementB, float_e4m3_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_F16E5M2E4M3_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e5m2_t ; Input B: float_e5m2_t
    else if constexpr (is_same_v<ElementA, float_e5m2_t> && is_same_v<ElementB, float_e5m2_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_F16E5M2E5M2_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    else {
      static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
    }
  }

  // F32 accumulator
  else if constexpr (is_same_v<ElementC, float>) {

    // Input A: half_t ; Input B: half_t
    if constexpr (is_same_v<ElementA, half_t> && is_same_v<ElementB, half_t>) {
      static_assert(size<2>(TileShape_MNK{}) % 16 == 0, "Tile_K must be a multiple of 16.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x16_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: bfloat16_t ; Input B: bfloat16_t
    else if constexpr (is_same_v<ElementA, bfloat16_t> && is_same_v<ElementB, bfloat16_t>) {
      static_assert(size<2>(TileShape_MNK{}) % 16 == 0, "Tile_K must be a multiple of 16.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x16_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: tfloat32_t ; Input B: tfloat32_t
    else if constexpr (is_same_v<ElementA, tfloat32_t> && is_same_v<ElementB, tfloat32_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 8 == 0, "Tile_K must be a multiple of 8.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x8_F32TF32TF32_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x8_F32TF32TF32_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x8_F32TF32TF32_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x8_F32TF32TF32_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x8_F32TF32TF32_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x8_F32TF32TF32_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x8_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x8_F32TF32TF32_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x8_F32TF32TF32_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e4m3_t ; Input B: float_e4m3_t
    else if constexpr (is_same_v<ElementA, float_e4m3_t> && is_same_v<ElementB, float_e4m3_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_F32E4M3E4M3_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e4m3_t ; Input B: float_e5m2_t
    else if constexpr (is_same_v<ElementA, float_e4m3_t> && is_same_v<ElementB, float_e5m2_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_F32E4M3E5M2_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e5m2_t ; Input B: float_e4m3_t
    else if constexpr (is_same_v<ElementA, float_e5m2_t> && is_same_v<ElementB, float_e4m3_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_F32E5M2E4M3_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e5m2_t ; Input B: float_e5m2_t
    else if constexpr (is_same_v<ElementA, float_e5m2_t> && is_same_v<ElementB, float_e5m2_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::MMA_64x248x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::MMA_64x232x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::MMA_64x216x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::MMA_64x200x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::MMA_64x184x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::MMA_64x168x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::MMA_64x152x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::MMA_64x136x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::MMA_64x120x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::MMA_64x104x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::MMA_64x88x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::MMA_64x72x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::MMA_64x56x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::MMA_64x40x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_F32E5M2E5M2_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    else {
      static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
    }
  }

  // S32 accumulator
  else if constexpr (is_same_v<ElementC, int32_t>) {

    // Input A: int8_t ; Input B: int8_t
    if constexpr (is_same_v<ElementA, int8_t> && is_same_v<ElementB, int8_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_S32S8S8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_S32S8S8_RS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_S32S8S8_RS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_S32S8S8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_S32S8S8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_S32S8S8_RS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_S32S8S8_RS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_S32S8S8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_S32S8S8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_S32S8S8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_S32S8S8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_S32S8S8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_S32S8S8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_S32S8S8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_S32S8S8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_S32S8S8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_S32S8S8_RS_TN{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_S32S8S8_RS_TN{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: int8_t ; Input B: uint8_t
    else if constexpr (is_same_v<ElementA, int8_t> && is_same_v<ElementB, uint8_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_S32S8U8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_S32S8U8_RS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_S32S8U8_RS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_S32S8U8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_S32S8U8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_S32S8U8_RS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_S32S8U8_RS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_S32S8U8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_S32S8U8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_S32S8U8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_S32S8U8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_S32S8U8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_S32S8U8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_S32S8U8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_S32S8U8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_S32S8U8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_S32S8U8_RS_TN{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_S32S8U8_RS_TN{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: uint8_t ; Input B: int8_t
    else if constexpr (is_same_v<ElementA, uint8_t> && is_same_v<ElementB, int8_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_S32U8S8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_S32U8S8_RS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_S32U8S8_RS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_S32U8S8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_S32U8S8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_S32U8S8_RS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_S32U8S8_RS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_S32U8S8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_S32U8S8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_S32U8S8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_S32U8S8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_S32U8S8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_S32U8S8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_S32U8S8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_S32U8S8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_S32U8S8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_S32U8S8_RS_TN{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_S32U8S8_RS_TN{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: uint8_t ; Input B: uint8_t
    else if constexpr (is_same_v<ElementA, uint8_t> && is_same_v<ElementB, uint8_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::MMA_64x256x32_S32U8U8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::MMA_64x240x32_S32U8U8_RS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::MMA_64x224x32_S32U8U8_RS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::MMA_64x208x32_S32U8U8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::MMA_64x192x32_S32U8U8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::MMA_64x176x32_S32U8U8_RS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::MMA_64x160x32_S32U8U8_RS_TN{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::MMA_64x144x32_S32U8U8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::MMA_64x128x32_S32U8U8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::MMA_64x112x32_S32U8U8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::MMA_64x96x32_S32U8U8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::MMA_64x80x32_S32U8U8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::MMA_64x64x32_S32U8U8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::MMA_64x48x32_S32U8U8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::MMA_64x32x32_S32U8U8_RS_TN{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::MMA_64x24x32_S32U8U8_RS_TN{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::MMA_64x16x32_S32U8U8_RS_TN{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::MMA_64x8x32_S32U8U8_RS_TN{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    else {
      static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
    }
  }

  // Unknown accumulator type
  else {
    static_assert(sizeof(ElementC) == 0, "Unknown ElementC accumulator type.");
  }
}

template <
  class ElementA,
  class ElementB,
  class ElementC,
  class TileShape_MNK,
  GMMA::Major MajorA = GMMA::Major::K,
  GMMA::Major MajorB = GMMA::Major::K,
  auto... Args                         // e.g. GMMA::ScaleOut::One, [GMMA::ScaleIn::One, GMMA::ScaleIn::One]
                                       // But most commonly leave empty for defaults
>
CUTE_HOST_DEVICE constexpr
auto
rs_op_selector_sparse()
{
  static_assert(is_static<TileShape_MNK>::value, "TileShape_MNK must be static.");
  static_assert(rank(TileShape_MNK{}) == 3, "TileShape_MNK must be rank 3.");
  static_assert(size<0>(TileShape_MNK{}) % 64 == 0, "Tile_M must be a multiple of 64.");
  static_assert(MajorA == GMMA::Major::K, "Register source A operand GMMAs must have K-major A layout.");
  auto Tile_N = size<1>(TileShape_MNK{});

  // F16 accumulator
  if constexpr (is_same_v<ElementC, half_t>) {

    // Input A: half_t ; Input B: half_t
    if constexpr (is_same_v<ElementA, half_t> && is_same_v<ElementB, half_t>) {
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x32_F16F16F16_RS<MajorA, MajorB, Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e4m3_t ; Input B: float_e4m3_t
    else if constexpr (is_same_v<ElementA, float_e4m3_t> && is_same_v<ElementB, float_e4m3_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_F16E4M3E4M3_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e4m3_t ; Input B: float_e5m2_t
    else if constexpr (is_same_v<ElementA, float_e4m3_t> && is_same_v<ElementB, float_e5m2_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_F16E4M3E5M2_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e5m2_t ; Input B: float_e4m3_t
    else if constexpr (is_same_v<ElementA, float_e5m2_t> && is_same_v<ElementB, float_e4m3_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_F16E5M2E4M3_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e5m2_t ; Input B: float_e5m2_t
    else if constexpr (is_same_v<ElementA, float_e5m2_t> && is_same_v<ElementB, float_e5m2_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_F16E5M2E5M2_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    else {
      static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
    }
  }

  // F32 accumulator
  else if constexpr (is_same_v<ElementC, float>) {

    // Input A: half_t ; Input B: half_t
    if constexpr (is_same_v<ElementA, half_t> && is_same_v<ElementB, half_t>) {
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x32_F32F16F16_RS<MajorA, MajorB, Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: bfloat16_t ; Input B: bfloat16_t
    else if constexpr (is_same_v<ElementA, bfloat16_t> && is_same_v<ElementB, bfloat16_t>) {
      static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x32_F32BF16BF16_RS<MajorA, MajorB, Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: tfloat32_t ; Input B: tfloat32_t
    else if constexpr (is_same_v<ElementA, tfloat32_t> && is_same_v<ElementB, tfloat32_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 16 == 0, "Tile_K must be a multiple of 16.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x16_F32TF32TF32_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x16_F32TF32TF32_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x16_F32TF32TF32_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x16_F32TF32TF32_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x16_F32TF32TF32_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x16_F32TF32TF32_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x16_F32TF32TF32_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x16_F32TF32TF32_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x16_F32TF32TF32_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e4m3_t ; Input B: float_e4m3_t
    else if constexpr (is_same_v<ElementA, float_e4m3_t> && is_same_v<ElementB, float_e4m3_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_F32E4M3E4M3_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e4m3_t ; Input B: float_e5m2_t
    else if constexpr (is_same_v<ElementA, float_e4m3_t> && is_same_v<ElementB, float_e5m2_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_F32E4M3E5M2_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e5m2_t ; Input B: float_e4m3_t
    else if constexpr (is_same_v<ElementA, float_e5m2_t> && is_same_v<ElementB, float_e4m3_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_F32E5M2E4M3_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: float_e5m2_t ; Input B: float_e5m2_t
    else if constexpr (is_same_v<ElementA, float_e5m2_t> && is_same_v<ElementB, float_e5m2_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 248 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x248x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 232 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x232x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 216 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x216x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 200 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x200x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 184 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x184x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 168 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x168x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 152 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x152x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 136 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x136x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 120 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x120x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 104 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x104x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 88 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x88x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 72 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x72x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 56 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x56x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 40 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x40x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_F32E5M2E5M2_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    else {
      static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
    }
  }

  // S32 accumulator
  else if constexpr (is_same_v<ElementC, int32_t>) {

    // Input A: int8_t ; Input B: int8_t
    if constexpr (is_same_v<ElementA, int8_t> && is_same_v<ElementB, int8_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_S32S8S8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_S32S8S8_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_S32S8S8_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_S32S8S8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_S32S8S8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_S32S8S8_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_S32S8S8_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_S32S8S8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_S32S8S8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_S32S8S8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_S32S8S8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_S32S8S8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_S32S8S8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_S32S8S8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_S32S8S8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_S32S8S8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_S32S8S8_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_S32S8S8_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: int8_t ; Input B: uint8_t
    else if constexpr (is_same_v<ElementA, int8_t> && is_same_v<ElementB, uint8_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_S32S8U8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_S32S8U8_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_S32S8U8_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_S32S8U8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_S32S8U8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_S32S8U8_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_S32S8U8_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_S32S8U8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_S32S8U8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_S32S8U8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_S32S8U8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_S32S8U8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_S32S8U8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_S32S8U8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_S32S8U8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_S32S8U8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_S32S8U8_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_S32S8U8_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: uint8_t ; Input B: int8_t
    else if constexpr (is_same_v<ElementA, uint8_t> && is_same_v<ElementB, int8_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_S32U8S8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_S32U8S8_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_S32U8S8_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_S32U8S8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_S32U8S8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_S32U8S8_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_S32U8S8_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_S32U8S8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_S32U8S8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_S32U8S8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_S32U8S8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_S32U8S8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_S32U8S8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_S32U8S8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_S32U8S8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_S32U8S8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_S32U8S8_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_S32U8S8_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    // Input A: uint8_t ; Input B: uint8_t
    else if constexpr (is_same_v<ElementA, uint8_t> && is_same_v<ElementB, uint8_t>) {
      static_assert(MajorA == GMMA::Major::K, "MajorA must be GMMA::Major::K for this config.");
      static_assert(MajorB == GMMA::Major::K, "MajorB must be GMMA::Major::K for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 64 == 0, "Tile_K must be a multiple of 64.");

      if constexpr (Tile_N % 256 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x256x64_S32U8U8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 240 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x240x64_S32U8U8_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 224 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x224x64_S32U8U8_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 208 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x208x64_S32U8U8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 192 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x192x64_S32U8U8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 176 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x176x64_S32U8U8_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 160 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x160x64_S32U8U8_RS_TN<Args...>{};
      }
#endif
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 144 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x144x64_S32U8U8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 128 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x128x64_S32U8U8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 112 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x112x64_S32U8U8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 96 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x96x64_S32U8U8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 80 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x80x64_S32U8U8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 64 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x64x64_S32U8U8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 48 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x48x64_S32U8U8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 32 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x32x64_S32U8U8_RS_TN<Args...>{};
      }
#if defined(CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED)
      else if constexpr (Tile_N % 24 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x24x64_S32U8U8_RS_TN<Args...>{};
      }
#endif
      else if constexpr (Tile_N % 16 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x16x64_S32U8U8_RS_TN<Args...>{};
      }
      else if constexpr (Tile_N % 8 == 0) {
        return SM90::GMMA::SPARSE::GMMA_64x8x64_S32U8U8_RS_TN<Args...>{};
      }
      else {
        static_assert(Tile_N % 8 == 0, "Tile_N must be a multiple of 8.");
      }
    }

    else {
      static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
    }
  }

  // Unknown accumulator type
  else {
    static_assert(sizeof(ElementC) == 0, "Unknown ElementC accumulator type.");
  }
}

} // end namespace SM90::GMMA
} // end namespace cute

////////////////////////////////////////////////////////////////////////////////////////////////////
