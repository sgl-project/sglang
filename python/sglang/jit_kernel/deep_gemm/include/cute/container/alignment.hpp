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

#include <cute/numeric/numeric_types.hpp>
#include <cute/numeric/math.hpp>

namespace cute
{

// Test if a pointer is aligned to N bytes
template <int N>
CUTE_HOST_DEVICE constexpr
bool
is_byte_aligned(void const* const ptr)
{
  static_assert(has_single_bit(N), "N must be a power of 2 in alignment check");
  return (reinterpret_cast<uintptr_t>(ptr) & (N-1)) == 0;
}

#if defined(__CUDACC__)
#  define CUTE_ALIGNAS(n) __align__(n)
#else
#  define CUTE_ALIGNAS(n) alignas(n)
#endif

template <size_t Alignment, class Child = void>
struct aligned_struct {};

template <class Child> struct CUTE_ALIGNAS(  1) aligned_struct<  1, Child> {};
template <class Child> struct CUTE_ALIGNAS(  2) aligned_struct<  2, Child> {};
template <class Child> struct CUTE_ALIGNAS(  4) aligned_struct<  4, Child> {};
template <class Child> struct CUTE_ALIGNAS(  8) aligned_struct<  8, Child> {};
template <class Child> struct CUTE_ALIGNAS( 16) aligned_struct< 16, Child> {};
template <class Child> struct CUTE_ALIGNAS( 32) aligned_struct< 32, Child> {};
template <class Child> struct CUTE_ALIGNAS( 64) aligned_struct< 64, Child> {};
template <class Child> struct CUTE_ALIGNAS(128) aligned_struct<128, Child> {};
template <class Child> struct CUTE_ALIGNAS(256) aligned_struct<256, Child> {};

} // end namespace cute
