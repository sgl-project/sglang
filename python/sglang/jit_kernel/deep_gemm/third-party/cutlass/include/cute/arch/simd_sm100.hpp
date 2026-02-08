/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cute/config.hpp>
#include <cute/arch/config.hpp>
#include <cute/numeric/real.hpp>

namespace cute {

CUTE_HOST_DEVICE
void 
add(float2      & c,
    float2 const& a, 
    float2 const& b) 
{
#if defined(CUTE_ARCH_FLOAT2_MATH_ENABLED)
  asm volatile("add.f32x2 %0, %1, %2;\n"
    : "=l"(reinterpret_cast<uint64_t      &>(c))
    :  "l"(reinterpret_cast<uint64_t const&>(a)),
       "l"(reinterpret_cast<uint64_t const&>(b)));
#else
  add(c.x, a.x, b.x);
  add(c.y, a.y, b.y);
#endif
}

CUTE_HOST_DEVICE
void 
mul(float2      & c,
    float2 const& a, 
    float2 const& b) 
{
#if defined(CUTE_ARCH_FLOAT2_MATH_ENABLED)
  asm volatile("mul.f32x2 %0, %1, %2;\n"
    : "=l"(reinterpret_cast<uint64_t      &>(c))
    :  "l"(reinterpret_cast<uint64_t const&>(a)),
       "l"(reinterpret_cast<uint64_t const&>(b)));
#else
  mul(c.x, a.x, b.x);
  mul(c.y, a.y, b.y);
#endif
}

CUTE_HOST_DEVICE
void 
fma(float2      & d,
    float2 const& a, 
    float2 const& b, 
    float2 const& c) 
{
#if defined(CUTE_ARCH_FLOAT2_MATH_ENABLED)
  asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
    : "=l"(reinterpret_cast<uint64_t      &>(d))
    :  "l"(reinterpret_cast<uint64_t const&>(a)),
       "l"(reinterpret_cast<uint64_t const&>(b)),
       "l"(reinterpret_cast<uint64_t const&>(c)));
#else
  fma(d.x, a.x, b.x, c.x);
  fma(d.y, a.y, b.y, c.y);
#endif
}

} // namespace cute
