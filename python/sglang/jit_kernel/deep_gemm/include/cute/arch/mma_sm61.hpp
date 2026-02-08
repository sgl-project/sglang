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
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610))
#  define CUTE_ARCH_MMA_SM61_ENABLED
#endif

namespace cute
{

struct SM61_DP4A
{
  using DRegisters = int32_t[1];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = int32_t[1];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(int32_t& d, uint32_t const& a, uint32_t const& b, int32_t const& c)
  {
#if defined(CUTE_ARCH_MMA_SM61_ENABLED)
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
                 : "=r"(d)
                 : "r"(a), "r"(b), "r"(c));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM61_DP4A without CUTE_ARCH_MMA_SM61_ENABLED");
#endif
  }
};

struct SM61_DP2A
{
  using DRegisters = int32_t[1];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = int32_t[1];

  // Register asm fma
  CUTE_HOST_DEVICE static void
  fma(int32_t& d, uint32_t const& a, uint32_t const& b, int32_t const& c)
  {
#if defined(CUTE_ARCH_MMA_SM61_ENABLED)
    asm volatile("dp2a.s32.s32 %0, %1, %2, %3;"
                 : "=r"(d)
                 : "r"(a), "r"(b), "r"(c));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM61_DP2A without CUTE_ARCH_MMA_SM61_ENABLED");
#endif
  }
};

} // namespace cute
