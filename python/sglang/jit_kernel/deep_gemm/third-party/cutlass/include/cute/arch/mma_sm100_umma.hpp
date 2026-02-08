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
#include <cute/arch/mma.hpp>
#include <cute/arch/mma_sm100.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/cluster_sm90.hpp>

namespace cute
{

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_TF32_SS
{
  static_assert(M == 64 || M == 128, "SM100_MMA_TF32 M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert((N % 8 == 0) && (8 <= N) && (N <= 256),
                "SM100_MMA_TF32 N-mode size should be a multiple of 8 between 8 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_TF32_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[4] = {0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, {%5, %6, %7, %8}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_TF32_SS without CUTE_ARCH_TCGEN05_TF32_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_SS
{
  static_assert(M == 64 || M == 128, "SM100_MMA_F16BF16 M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert((N % 8 == 0)  && (8 <= N)  && (N <= 256),
                "SM100_MMA_F16BF16 N-mode size should be a multiple of 8 between 8 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[4] = {0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F16BF16_SS without CUTE_ARCH_MMA_SM100A_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_TF32_TS
{
  static_assert(M == 64 || M == 128, "SM100_MMA_TF32 M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert((M == 64  && (N % 8 == 0)  && (8 <= N)  && (N <= 256)) ||
                (M == 128 && (N % 16 == 0) && (16 <= N) && (N <= 256)),
                "SM100_MMA_TF32 N-mode size should be a multiple of 8 between 8 and 256 for M=64,\
                 or a multiple of 16 between 16 and 256 for M=128.");
  static_assert(a_major == UMMA::Major::K, "SM100_MMA_TF32 A from TMEM can't be transposed");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& tmem_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_TF32_MMA_ENABLED)
    uint32_t mask[4] = {0, 0, 0, 0};
    if (cute::elect_one_sync()) {
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, {%5, %6, %7, %8}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_TF32_TS without CUTE_ARCH_TCGEN05_TF32_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_F16BF16_TS
{
  static_assert(M == 64 || M == 128, "SM100_MMA_F16BF16 M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert((M == 64  && (N % 8 == 0)  && (8 <= N)  && (N <= 256)) ||
                (M == 128 && (N % 16 == 0) && (16 <= N) && (N <= 256)),
                "SM100_MMA_F16BF16 N-mode size should be a multiple of 8 between 8 and 256 for M=64,\
                 or a multiple of 16 between 16 and 256 for M=128.");
  static_assert(a_major == UMMA::Major::K, "SM100_MMA_F16BF16 A from TMEM can't be transposed");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& tmem_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
    uint32_t mask[4] = {0, 0, 0, 0};
    if (cute::elect_one_sync()) {
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%5, %6, %7, %8}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F16BF16_TS without CUTE_ARCH_MMA_SM100A_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major, uint32_t ScaleC,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_SS_SCALED
{
  static_assert(M == 64 || M == 128, "SM100_MMA_F16BF16_SS_SCALED M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert((N % 8 == 0) && (8 <= N) && (N <= 256),
                "SM100_MMA_F16BF16_SS_SCALED N-mode size should be a multiple of 8 between 8 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& accumulate,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_F16BF16_MMA_SCALED_ENABLED)
    if (cute::elect_one_sync()) {
      // ScaleC input should be a literal or compile time constant
      uint32_t mask[4] = {0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p, %9; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(accumulate),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]), "n"(ScaleC));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F16BF16_SS without CUTE_ARCH_TCGEN05_F16BF16_MMA_SCALED_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major, uint32_t ScaleC,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_F16BF16_TS_SCALED
{
  static_assert(M == 64 || M == 128, "SM100_MMA_F16BF16_TS_SCALED M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert((M == 64  && (N % 8 == 0)  && (8 <= N)  && (N <= 256)) ||
                (M == 128 && (N % 16 == 0) && (16 <= N) && (N <= 256)),
                "SM100_MMA_F16BF16_TS_SCALED N-mode size should be a multiple of 8 between 8 and 256 for M=64,\
                 or a multiple of 16 between 16 and 256 for M=128.");
  static_assert(a_major == UMMA::Major::K, "SM100_MMA_F16BF16_TS_SCALED A from TMEM can't be transposed");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& tmem_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& accumulate,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_F16BF16_MMA_SCALED_ENABLED)
    if (cute::elect_one_sync()) {
      // ScaleC input should be a literal or compile time constant
      uint32_t mask[4] = {0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%5, %6, %7, %8}, p, %9; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(accumulate),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]), "n"(ScaleC));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F16BF16_TS_SCALED without CUTE_ARCH_TCGEN05_F16BF16_MMA_SCALED_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_TF32_SS_SPARSE
{
  static_assert(M == 64 || M == 128, "SM100_MMA_TF32_SS_SPARSE M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert((N % 8 == 0) && (8 <= N) && (N <= 256),
                "SM100_MMA_TF32_SS_SPARSE N-mode size should be a multiple of 8 between 8 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tmem_e)
  {
#if defined(CUTE_ARCH_TCGEN05_TF32_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[4] = {0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.sp.cta_group::1.kind::tf32 [%0], %1, %2, [%9], %3, {%5, %6, %7, %8}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]), "r"(tmem_e));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_TF32_SS_SPARSE without CUTE_ARCH_TCGEN05_TF32_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_SS_SPARSE
{
  static_assert(M == 64 || M == 128, "SM100_MMA_F16BF16_SS_SPARSE M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert((N % 8 == 0) && (8 <= N) && (N <= 256),
                "SM100_MMA_F16BF16_SS_SPARSE N-mode size should be a multiple of 8 between 8 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tmem_e)
  {
#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[4] = {0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.sp.cta_group::1.kind::f16 [%0], %1, %2, [%9], %3, {%5, %6, %7, %8}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]), "r"(tmem_e));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F16BF16_SS_SPARSE without CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_TF32_2x1SM_SS
{
  static_assert(M == 128 || M == 256, "SM100_MMA_TF32_2x1SM_SS M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256), "SM100_MMA_TF32_2x1SM_SS N-mode size should be a multiple of 16 between 16 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_TF32_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::tf32 [%0], %1, %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
          "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_TF32_2x1SM_SS without CUTE_ARCH_TCGEN05_TF32_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_2x1SM_SS
{
  static_assert(M == 128 || M == 256, "SM100_MMA_F16BF16_2x1SM_SS M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256), "SM100_MMA_F16BF16_2x1SM_SS N-mode size should be a multiple of 16 between 16 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
          "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F16BF16_2x1SM_SS without CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_TF32_2x1SM_TS
{
  static_assert(M == 128 || M == 256, "SM100_MMA_TF32_2x1SM_TS M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 32 == 0) && (32 <= N) && (N <= 256), "SM100_MMA_TF32_2x1SM_TS N-mode size should be a multiple of 32 between 32 and 256.");
  static_assert(a_major == UMMA::Major::K, "SM100_MMA_TF32_2x1SM_TS A from TMEM can't be transposed");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& tmem_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_TF32_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::tf32 [%0], [%1], %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
          "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_TF32_2x1SM_TS without CUTE_ARCH_TCGEN05_TF32_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_F16BF16_2x1SM_TS
{
  static_assert(M == 128 || M == 256, "SM100_MMA_F16BF16_2x1SM_TS M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 32 == 0) && (32 <= N) && (N <= 256), "SM100_MMA_F16BF16_2x1SM_TS N-mode size should be a multiple of 32 between 32 and 256.");
  static_assert(a_major == UMMA::Major::K, "SM100_MMA_F16BF16_2x1SM_TS A from TMEM can't be transposed");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& tmem_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
          "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F16BF16_2x1SM_TS without CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major, uint32_t ScaleC,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_2x1SM_SS_SCALED
{
  static_assert(M == 128 || M == 256, "SM100_MMA_F16BF16_2x1SM_SS_SCALED M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256), "SM100_MMA_F16BF16_2x1SM_SS_SCALED N-mode size should be a multiple of 16 between 16 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& accumulate,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_F16BF16_MMA_SCALED_ENABLED)
    if (cute::elect_one_sync()) {
      // ScaleC input should be a literal or compile time constant
      uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p, %13; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(accumulate),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
          "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]), "n"(ScaleC));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F16BF16_2x1SM_SS_SCALED without CUTE_ARCH_TCGEN05_F16BF16_MMA_SCALED_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major, uint32_t ScaleC,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_F16BF16_2x1SM_TS_SCALED
{
  static_assert(M == 128 || M == 256, "SM100_MMA_F16BF16_2x1SM_TS_SCALED M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 32 == 0) && (32 <= N) && (N <= 256), "SM100_MMA_F16BF16_2x1SM_TS_SCALED N-mode size should be a multiple of 32 between 32 and 256.");
  static_assert(a_major == UMMA::Major::K, "SM100_MMA_F16BF16_2x1SM_TS_SCALED A from TMEM can't be transposed");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& tmem_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& accumulate,
      uint64_t idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_F16BF16_MMA_SCALED_ENABLED)
    if (cute::elect_one_sync()) {
      // ScaleC input should be a literal or compile time constant
      uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p, %13; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(accumulate),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
          "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]), "n"(ScaleC));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F16BF16_2x1SM_TS_SCALED without CUTE_ARCH_TCGEN05_F16BF16_MMA_SCALED_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_TF32_2x1SM_SS_SPARSE
{
  static_assert(M == 128 || M == 256, "SM100_MMA_TF32_2x1SM_SS_SPARSE M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256), "SM100_MMA_TF32_2x1SM_SS_SPARSE N-mode size should be a multiple of 16 between 16 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tmem_e)
  {
#if defined(CUTE_ARCH_TCGEN05_TF32_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.sp.cta_group::2.kind::tf32 [%0], %1, %2, [%13], %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
          "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]), "r"(tmem_e));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_TF32_2x1SM_SS_SPARSE without CUTE_ARCH_TCGEN05_TF32_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_2x1SM_SS_SPARSE
{
  static_assert(M == 128 || M == 256, "SM100_MMA_F16BF16_2x1SM_SS_SPARSE M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256), "SM100_MMA_F16BF16_2x1SM_SS_SPARSE N-mode size should be a multiple of 16 between 16 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tmem_e)
  {
#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.sp.cta_group::2.kind::f16 [%0], %1, %2, [%13], %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
          "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]), "r"(tmem_e));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F16BF16_2x1SM_SS_SPARSE without CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_S8_SS
{
  static_assert(is_same_v<c_type, int32_t>, "SM100_MMA_S8_SS result type can only be int32_t.");
  static_assert(M == 64 || M == 128, "SM100_MMA_S8_SS M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert(N == 8 || ((N % 16 == 0) && (16 <= N) && (N <= 256)), "SM100_MMA_S8_SS N-mode size should be 8 or a multiple of 16 between 16 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_S8_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[4] = {0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, {%5, %6, %7, %8}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_S8_SS without CUTE_ARCH_TCGEN05_S8_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_S8_TS
{
  static_assert(M == 64 || M == 128, "SM100_MMA_S8_TS M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert(N == 8 || ((N % 16 == 0) && (16 <= N) && (N <= 256)), "SM100_MMA_S8_TS N-mode size should be 8 or a multiple of 16 between 16 and 256.");
  static_assert(a_major == UMMA::Major::K, "SM100_MMA_S8_TS A from TMEM can't be transposed");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& tmem_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_S8_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[4] = {0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::i8 [%0], [%1], %2, %3, {%5, %6, %7, %8}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_S8_TS without CUTE_ARCH_TCGEN05_S8_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_S8_SS_SPARSE
{
  static_assert(is_same_v<c_type, int32_t>, "SM100_MMA_S8_SS_SPARSE result type can only be int32_t.");
  static_assert(M == 64 || M == 128, "SM100_MMA_S8_SS_SPARSE M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert(N == 8 || ((N % 16 == 0) && (16 <= N) && (N <= 256)), "SM100_MMA_S8_SS_SPARSE N-mode size should be 8 or a multiple of 16 between 16 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tmem_e)
  {
#if defined(CUTE_ARCH_TCGEN05_S8_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[4] = {0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.sp.cta_group::1.kind::i8 [%0], %1, %2, [%9], %3, {%5, %6, %7, %8}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]), "r"(tmem_e));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_S8_SS_SPARSE without CUTE_ARCH_TCGEN05_S8_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_S8_2x1SM_SS
{
  static_assert(M == 128 || M == 256, "SM100_MMA_S8_2x1SM_SS M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 32 == 0) && (32 <= N) && (N <= 256), "SM100_MMA_S8_2x1SM_SS N-mode size should be a multiple of 32 between 32 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_S8_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::i8 [%0], %1, %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
          "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_S8_2x1SM_SS without CUTE_ARCH_TCGEN05_S8_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_S8_2x1SM_TS
{
  static_assert(M == 128 || M == 256, "SM100_MMA_S8_2x1SM_TS M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 32 == 0) && (32 <= N) && (N <= 256), "SM100_MMA_S8_2x1SM_TS N-mode size should be a multiple of 32 between 32 and 256.");
  static_assert(a_major == UMMA::Major::K, "SM100_MMA_S8_2x1SM_TS A from TMEM can't be transposed");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& tmem_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_S8_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::i8 [%0], [%1], %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
          "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_S8_2x1SM_TS without CUTE_ARCH_TCGEN05_S8_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_S8_2x1SM_SS_SPARSE
{
  static_assert(M == 128 || M == 256, "SM100_MMA_S8 M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 32 == 0) && (32 <= N) && (N <= 256), "SM100_MMA_S8 N-mode size should be a multiple of 32 between 32 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tmem_e)
  {
#if defined(CUTE_ARCH_TCGEN05_S8_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.sp.cta_group::2.kind::i8 [%0], %1, %2, [%13], %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
          "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]), "r"(tmem_e));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_S8_2x1SM_SS_SPARSE without CUTE_ARCH_TCGEN05_S8_MMA_ENABLED");
#endif
  }
};

struct SM100_MMA_F8F6F4_SS
{
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[4] = {0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%5, %6, %7, %8}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F8F6F4_SS without CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_MXF8F6F4_SS
{
  static_assert(M == 128, "SM100_MMA_MXF8F6F4_SS M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert((N % 8 == 0) && (8 <= N) && (N <= 256), "SM100_MMA_MXF8F6F4_SS N-mode size should be a multiple of 8 between 8 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tsfa_addr,
      uint32_t const& tsfb_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC), "r"(tsfa_addr), "r"(tsfb_addr));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F8F6F4_SS without CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_F8F6F4_TS
{
  static_assert(M == 64 || M == 128, "SM100_MMA_F8F6F4_TS M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert((M == 64  && (N % 8 == 0)  && (8 <= N)  && (N <= 256)) ||
                (M == 128 && (N % 16 == 0) && (16 <= N) && (N <= 256)),
                "SM100_MMA_F8F6F4_TS N-mode size should be a multiple of 8 between 8 and 256 for M=64,\
                 or a multiple of 16 between 16 and 256 for M=128.");
  static_assert(a_major == UMMA::Major::K, "SM100_MMA_F8F6F4_TS A from TMEM can't be transposed");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& tmem_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[4] = {0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, {%5, %6, %7, %8}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F8F6F4_TS without CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One,
          UMMA::Saturate c_sat = UMMA::Saturate::False>
struct SM100_MMA_F8F6F4_2x1SM_TS
{
  static_assert(M == 128 || M == 256, "SM100_MMA_F8F6F4_2x1SM_TS M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert((N % 32 == 0) && (32 <= N) && (N <= 256), "SM100_MMA_F8F6F4_2x1SM_TS N-mode size should be a multiple of 32 between 32 and 256.");
  static_assert(a_major == UMMA::Major::K, "SM100_MMA_F8F6F4_2x1SM_TS A from TMEM can't be transposed");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& tmem_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%1], %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
          "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F8F6F4_TS without CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F8F6F4_SS_SPARSE
{
  static_assert(M == 64 || M == 128, "SM100_MMA_F8F6F4_SS_SPARSE M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
  static_assert((M == 64  && (N % 8 == 0)  && (8 <= N)  && (N <= 256)) ||
                (M == 128 && (N % 16 == 0) && (16 <= N) && (N <= 256)),
                "SM100_MMA_F8F6F4_SS_SPARSE N-mode size should be a multiple of 8 between 8 and 256 for M=64,\
                 or a multiple of 16 between 16 and 256 for M=128.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tmem_e)
  {
#if defined(CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[4] = {0, 0, 0, 0}; // %5, %6, %7, %8
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [%0], %1, %2, [%9], %3, {%5, %6, %7, %8}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]), "r"(tmem_e));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F8F6F4_SS_SPARSE without CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_MXF8F6F4_SS_SPARSE
{
  static_assert(M == 128, "SM100_MMA_MXF8F6F4_SS_SPARSE M-mode size should be 128 for 1 CTA cluster MMA.");
  static_assert((N % 8 == 0) && (8 <= N) && (N <= 256), "SM100_MMA_MXF8F6F4_SS_SPARSE N-mode size should be a multiple of 8 between 8 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tsfa_addr,
      uint32_t const& tsfb_addr,
      uint32_t const& tmem_e)
  {
#if defined(CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, [%7], %3, [%5], [%6], p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC), "r"(tsfa_addr), "r"(tsfb_addr), "r"(tmem_e));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_MXF8F6F4_SS_SPARSE without CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

struct SM100_MMA_F8F6F4_2x1SM_SS
{  
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE)
  {
#if defined(CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
          "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F8F6F4_2x1SM_SS without CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_MXF8F6F4_2x1SM_SS_SPARSE
{
  static_assert(M == 256, "SM100_MMA_MXF8F6F4_2x1SM_SS_SPARSE M-mode size should be 256 for 2 CTA cluster MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256), "SM100_MMA_MXF8F6F4_2x1SM_SS_SPARSE N-mode size should be a multiple of 16 between 16 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tsfa_addr,
      uint32_t const& tsfb_addr,
      uint32_t const& tmem_e)
  {
#if defined(CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, [%7], %3, [%5], [%6], p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(tsfa_addr), "r"(tsfb_addr), "r"(tmem_e));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_MXF8F6F4_2x1SM_SS_SPARSE without CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_MXF8F6F4_2x1SM_SS
{
  static_assert(M == 256, "SM100_MMA_MXF8F6F4_2x1SM_SS M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256), "SM100_MMA_MXF8F6F4_2x1SM_SS N-mode size should be a multiple of 16 between 16 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tsfa_addr,
      uint32_t const& tsfb_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(tsfa_addr), "r"(tsfb_addr));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_MXF8F6F4_2x1SM_SS without CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type,
          int M, int N, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F8F6F4_2x1SM_SS_SPARSE
{
  static_assert(M == 128 || M == 256, "SM100_MMA_F8F6F4_2x1SM_SS_SPARSE M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 32 == 0) && (32 <= N) && (N <= 256), "SM100_MMA_F8F6F4_2x1SM_SS_SPARSE N-mode size should be a multiple of 32 between 32 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tmem_e)
  {
#if defined(CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.sp.cta_group::2.kind::f8f6f4 [%0], %1, %2, [%13], %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
          "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]), "r"(tmem_e));
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_F8F6F4_2x1SM_SS_SPARSE without CUTE_ARCH_TCGEN05_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, int VS, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_MXF4_SS
{
  static_assert(M == 128, "SM100_MMA_MXF4_SS M-mode size should be 128 for 1 CTA cluster MMA.");
  static_assert((N % 8 == 0) && (8 <= N) && (N <= 256), "SM100_MMA_MXF4_SS N-mode size should be a multiple of 8 between 8 and 256.");
  static_assert((VS == 16) || (VS == 32), "SM100_MMA_MXF4_SS Vector size can only be 16 or 32.");

  using DRegisters   = void;
  using ARegisters   = uint64_t[1];
  using BRegisters   = uint64_t[1];
  using CRegisters   = uint32_t[1];
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tsfa_addr,
      uint32_t const& tsfb_addr)
  {
    if constexpr (VS == 16) {
#if defined(CUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ENABLED)
      if (cute::elect_one_sync()) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
          "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
          "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
            "r"(tsfa_addr), "r"(tsfb_addr));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_MXF4_SS (VS = 16) without CUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ENABLED");
#endif
    }
    if constexpr (VS == 32) {
#if defined(CUTE_ARCH_TCGEN05_MXF4_MMA_ENABLED)
      if (cute::elect_one_sync()) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
          "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
          "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
            "r"(tsfa_addr), "r"(tsfb_addr));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_MXF4_SS (VS = 32) without CUTE_ARCH_TCGEN05_MXF4_MMA_ENABLED");
#endif
    }
  }

};

template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, int VS, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_MXF4NVF4_SS_SPARSE
{
  static_assert(M == 128, "SM100_MMA_MXF4NVF4_SS_SPARSE M-mode size should be 128 for 1 CTA cluster MMA.");
  static_assert((N % 8 == 0) && (8 <= N) && (N <= 256), "SM100_MMA_MXF4NVF4_SS_SPARSE N-mode size should be a multiple of 8 between 8 and 256.");

  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tsfa_addr,
      uint32_t const& tsfb_addr,
      uint32_t const& tmem_e)
  {
    if constexpr (VS == 32) {
#if defined(CUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ENABLED)
      if (cute::elect_one_sync()) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
          "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, [%7], %3, [%5], [%6], p; \n\t"
#else
          "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, [%7], %3, [%5], [%6], p; \n\t"
#endif
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
            "r"(tsfa_addr), "r"(tsfb_addr), "r"(tmem_e));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_MXF4NVF4_SS_SPARSE (VS = 32) without CUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ENABLED");
#endif
    }

    if constexpr (VS == 64) {
#if defined(CUTE_ARCH_TCGEN05_MXF4_MMA_ENABLED)
      if (cute::elect_one_sync()) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
          "tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32 [%0], %1, %2, [%7], %3, [%5], [%6], p; \n\t"
#else
          "tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, [%7], %3, [%5], [%6], p; \n\t"
#endif
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
            "r"(tsfa_addr), "r"(tsfb_addr), "r"(tmem_e));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_MXF4NVF4_SS_SPARSE (VS = 64) without CUTE_ARCH_TCGEN05_MXF4_MMA_ENABLED");
#endif
    }
  }
};

template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, int VS, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_MXF4_2x1SM_SS
{
  static_assert(M == 128 || M == 256, "SM100_MMA_MXF4_2x1SM_SS M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256), "SM100_MMA_MXF4_2x1SM_SS N-mode size should be a multiple of 16 between 16 and 256.");
  static_assert((VS == 16) || (VS == 32), "SM100_MMA_MXF4_2x1SM_SS Vector size can only be 16 or 32.");

  using DRegisters   = void;
  using ARegisters   = uint64_t[1];
  using BRegisters   = uint64_t[1];
  using CRegisters   = uint32_t[1];
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tsfa_addr,
      uint32_t const& tsfb_addr)
  {
    if constexpr (VS == 16) {
#if defined(CUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ENABLED)
      if (cute::elect_one_sync()) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
          "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
          "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
            "r"(tsfa_addr), "r"(tsfb_addr));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_MXF4_2x1SM_SS (VS = 16) without CUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ENABLED");
#endif
    }
    if constexpr (VS == 32) {
#if defined(CUTE_ARCH_TCGEN05_MXF4_MMA_ENABLED)
      if (cute::elect_one_sync()) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
          "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
          "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
            "r"(tsfa_addr), "r"(tsfb_addr));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_MXF4_2x1SM_SS (VS = 32) without CUTE_ARCH_TCGEN05_MXF4_MMA_ENABLED");
#endif
    }
  }
};

template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, int VS, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_MXF4NVF4_2x1SM_SS_SPARSE
{
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256), "SM100_MMA_MXF4NVF4_2x1SM_SS_SPARSE N-mode size should be a multiple of 16 between 16 and 256.");
  static_assert((VS == 32) || (VS == 64), "SM100_MMA_MXF4NVF4_2x1SM_SS_SPARSE Vector size can only be 32 or 64.");

  using DRegisters   = void;
  using ARegisters   = uint64_t[1];
  using BRegisters   = uint64_t[1];
  using CRegisters   = uint32_t[1];
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tsfa_addr,
      uint32_t const& tsfb_addr,
      uint32_t const& tmem_e)
  {
    if constexpr (VS == 32) {
#if defined(CUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ENABLED)
      if (cute::elect_one_sync()) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
          "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, [%7], %3, [%5], [%6], p; \n\t"
#else
          "tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, [%7], %3, [%5], [%6], p; \n\t"
#endif
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
            "r"(tsfa_addr), "r"(tsfb_addr), "r"(tmem_e));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_MXF4NVF4_2x1SM_SS_SPARSE (VS = 32) without CUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ENABLED");
#endif
    }

    if constexpr (VS == 64) {
#if defined(CUTE_ARCH_TCGEN05_MXF4_MMA_ENABLED)
      if (cute::elect_one_sync()) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
          "tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32 [%0], %1, %2, [%7], %3, [%5], [%6], p; \n\t"
#else
          "tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, [%7], %3, [%5], [%6], p; \n\t"
#endif
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
            "r"(tsfa_addr), "r"(tsfb_addr), "r"(tmem_e));
      }
#else
      CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_MMA_MXF4NVF4_2x1SM_SS_SPARSE (VS = 64) without CUTE_ARCH_TCGEN05_MXF4_MMA_ENABLED");
#endif
    }
  }
};

namespace SM103 {
template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, int VS, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM103_MXF4_ULTRA_SS_VS
{
  static_assert(M == 128, "MMA M-mode size should be 128 for 1 CTA cluster MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256), "MMA N-mode size should be a multiple of 16 between 16 and 256.");
  static_assert(((VS == 32) & (is_same_v<a_type, cutlass::float_e2m1_t> && is_same_v<sf_type, cutlass::float_ue8m0_t>)) || (VS == 16),
    "Vector size can only be 4x mode (VS=16) or 2x mode (VS=32) for MMA. 2x mode only supports float_e2m1_t for a/b types and ue8m0_t for sf type");

  using DRegisters   = void;
  using ARegisters   = uint64_t[1];
  using BRegisters   = uint64_t[1];
  using CRegisters   = uint32_t[1];
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tsfa_addr,
      uint32_t const& tsfb_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ULTRA_ENABLED)
    if constexpr (VS == 16) {
      if (cute::elect_one_sync()) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
          "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
          "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
            "r"(tsfa_addr), "r"(tsfb_addr));
      }
    }
    else if constexpr (VS == 32) {
      if (cute::elect_one_sync()) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
          "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
          "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
            "r"(tsfa_addr), "r"(tsfb_addr));
      }
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM103_MXF4_ULTRA_SS_VS without CUTE_ARCH_MMA_SM103A_ENABLED");
#endif
  }

};


template <class a_type, class b_type, class c_type, class sf_type,
          int M, int N, int VS, UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One, UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM103_MXF4_ULTRA_2x1SM_SS_VS
{
  static_assert(M == 128 || M == 256, "MMA M-mode size should be 128 or 256 for 2 CTA cluster MMA.");
  static_assert((N % 16 == 0) && (16 <= N) && (N <= 256), "MMA N-mode size should be a multiple of 16 between 16 and 256.");
  static_assert(((VS == 32) & (is_same_v<a_type, cutlass::float_e2m1_t> && is_same_v<sf_type, cutlass::float_ue8m0_t>)) || (VS == 16),
    "Vector size can only be 4x mode (VS=16) or 2x mode (VS=32) for MMA. 2x mode only supports float_e2m1_t for a/b types and ue8m0_t for sf type");

  using DRegisters   = void;
  using ARegisters   = uint64_t[1];
  using BRegisters   = uint64_t[1];
  using CRegisters   = uint32_t[1];
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t const& tmem_c,
      uint32_t const& scaleC,
      uint64_t const& idescE,
      uint32_t const& tsfa_addr,
      uint32_t const& tsfb_addr)
  {
#if defined(CUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ULTRA_ENABLED)
    if constexpr (VS == 16) {
      if (cute::elect_one_sync()) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
          "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
          "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
            "r"(tsfa_addr), "r"(tsfb_addr));
      }
    }
    else if constexpr (VS == 32) {
      if (cute::elect_one_sync()) {
        asm volatile(
          "{\n\t"
          ".reg .pred p;\n\t"
          "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
          "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
          "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idescE>>32)), "r"(scaleC),
            "r"(tsfa_addr), "r"(tsfb_addr));
      }
    }
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM103_MXF4_ULTRA_2x1SM_SS_VS without CUTE_ARCH_MMA_SM103A_ENABLED");
#endif
  }

};
} // namespace SM103

} // end namespace cute
