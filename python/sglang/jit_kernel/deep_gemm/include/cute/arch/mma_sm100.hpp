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
//
//

#pragma once

#include <cute/arch/config.hpp>
#include <cute/arch/mma.hpp>

#include <cute/arch/simd_sm100.hpp>

namespace cute {

struct SM100_2x1x1_F32F32F32F32 {
  using DRegisters = float2[1];
  using ARegisters = float2[1];
  using BRegisters = float[1];
  using CRegisters = float2[1];

  CUTE_HOST_DEVICE static void
  fma(float2       &  d01,
      float2  const&  a01,
      float   const&  b0,
      float2  const&  c01)
  {
#if defined(CUTE_ARCH_FFMA2_SM100_ENABLED)
  cute::fma(d01, a01, make_float2(b0, b0), c01);
#else
  CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_2x1x1_F32F32F32F32 without CUTE_ARCH_FLOAT2_MATH_ENABLED");
#endif
  }
};

struct SM100_1x2x1_F32F32F32F32 {
  using DRegisters = float2[1];
  using ARegisters = float[1];
  using BRegisters = float2[1];
  using CRegisters = float2[1];

  CUTE_HOST_DEVICE static void
  fma(float2       &  d01,
      float   const&  a0,
      float2  const&  b01,
      float2  const&  c01)
  {
#if defined(CUTE_ARCH_FFMA2_SM100_ENABLED)
  cute::fma(d01, make_float2(a0, a0), b01, c01);
#else
  CUTE_INVALID_CONTROL_PATH("Attempting to use SM100_1x2x1_F32F32F32F32 without CUTE_ARCH_FFMA2_SM100_ENABLED");
#endif
  }
};

} // namespace cute
