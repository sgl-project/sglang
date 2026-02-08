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

#include <cute/config.hpp>                     // CUTE_HOST_DEVICE, CUTE_GCC_UNREACHABLE
#include <cute/numeric/integral_constant.hpp>  // cute::integral_constant

namespace cute
{

//
// dim3
//

using dim3 = ::dim3;

// MSVC doesn't define its C++ version macro to match
// its C++ language version.  This means that when
// building with MSVC, dim3 isn't constexpr-friendly.
template <size_t I>
CUTE_HOST_DEVICE
#if ! defined(_MSC_VER)
constexpr
#endif
uint32_t& get(dim3& a)
{
  static_assert(I < 3, "Index out of range");
  if constexpr (I == 0) {
    return a.x;
  } else if constexpr (I == 1) {
    return a.y;
  } else if constexpr (I == 2) {
    return a.z;
  }

  CUTE_GCC_UNREACHABLE;
}

template <size_t I>
CUTE_HOST_DEVICE
#if ! defined(_MSC_VER)
constexpr
#endif
uint32_t const& get(dim3 const& a)
{
  static_assert(I < 3, "Index out of range");
  if constexpr (I == 0) {
    return a.x;
  } else if constexpr (I == 1) {
    return a.y;
  } else if constexpr (I == 2) {
    return a.z;
  }

  CUTE_GCC_UNREACHABLE;
}

template <size_t I>
CUTE_HOST_DEVICE
#if ! defined(_MSC_VER)
constexpr
#endif
uint32_t&& get(dim3&& a)
{
  static_assert(I < 3, "Index out of range");
  if constexpr (I == 0) {
    return cute::move(a.x);
  } else if constexpr (I == 1) {
    return cute::move(a.y);
  } else if constexpr (I == 2) {
    return cute::move(a.z);
  }

  CUTE_GCC_UNREACHABLE;
}

// Specialize cute::tuple-traits for external types
template <>
struct tuple_size<dim3>
    : integral_constant<size_t, 3>
{};

template <size_t I>
struct tuple_element<I, dim3>
{
  using type = uint32_t;
};

//
// uint3
//

using uint3 = ::uint3;

template <size_t I>
CUTE_HOST_DEVICE constexpr
uint32_t& get(uint3& a)
{
  static_assert(I < 3, "Index out of range");
  if constexpr (I == 0) {
    return a.x;
  } else if constexpr (I == 1) {
    return a.y;
  } else if constexpr (I == 2) {
    return a.z;
  }

  CUTE_GCC_UNREACHABLE;
}

template <size_t I>
CUTE_HOST_DEVICE constexpr
uint32_t const& get(uint3 const& a)
{
  static_assert(I < 3, "Index out of range");
  if constexpr (I == 0) {
    return a.x;
  } else if constexpr (I == 1) {
    return a.y;
  } else if constexpr (I == 2) {
    return a.z;
  }

  CUTE_GCC_UNREACHABLE;
}

template <size_t I>
CUTE_HOST_DEVICE constexpr
uint32_t&& get(uint3&& a)
{
  static_assert(I < 3, "Index out of range");
  if constexpr (I == 0) {
    return cute::move(a.x);
  } else if constexpr (I == 1) {
    return cute::move(a.y);
  } else if constexpr (I == 2) {
    return cute::move(a.z);
  }

  CUTE_GCC_UNREACHABLE;
}

// Specialize cute::tuple-traits for external types
template <>
struct tuple_size<uint3>
    : integral_constant<size_t, 3>
{};

template <size_t I>
struct tuple_element<I, uint3>
{
  using type = uint32_t;
};

} // end namespace cute
