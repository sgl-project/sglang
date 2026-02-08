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

#include <cute/config.hpp>          // CUTE_HOST_DEVICE
#include <cute/numeric/int.hpp>     // cute::int2_t, cute::int4_t, etc

#include <cutlass/numeric_size.h>   // cutlass::sizeof_bits
#include <cutlass/numeric_types.h>  // cutlass::float_e4m3_t, cutlass::float_e5m2_t, etc

namespace cute {

template <class T>
struct sizeof_bits : cutlass::sizeof_bits<T> {};

template <class T>
struct sizeof_bits<T const> : sizeof_bits<T> {};

template <class T>
struct sizeof_bits<T volatile> : sizeof_bits<T> {};

template <class T>
struct sizeof_bits<T const volatile> : sizeof_bits<T> {};

// DO NOT change auto to int, sizeof_bits<sparse_elem> use integral_ratio instead of int
template <class T>
static constexpr auto sizeof_bits_v = sizeof_bits<T>::value;

using cutlass::bits_to_bytes;
using cutlass::bytes_to_bits;

using cutlass::is_subbyte;

template <class T>
static constexpr auto is_subbyte_v = is_subbyte<T>::value;

//
// Integral
//

using cutlass::bin1_t;
using cutlass::uint1b_t;
using cutlass::int2b_t;
using cutlass::uint2b_t;
using cutlass::int4b_t;
using cutlass::uint4b_t;
using cutlass::int6b_t;
using cutlass::uint6b_t;

//
// Floating Point
//

using cutlass::half_t;
using cutlass::bfloat16_t;

using cutlass::tfloat32_t;

// Umbrella floating-point 8-bit data type : type_erased_dynamic_float8_t
// This umbrella datatype can be enabled when a user provides a specific
// datatype in runtime argument list.
using cutlass::type_erased_dynamic_float8_t;
using cutlass::float_e4m3_t;
using cutlass::float_e5m2_t;




using cutlass::float_ue4m3_t;
using cutlass::float_ue8m0_t;

using cutlass::float_e2m1_t;
using cutlass::float_e2m3_t;
using cutlass::float_e3m2_t;

using cutlass::type_erased_dynamic_float6_t;
using cutlass::type_erased_dynamic_float4_t;

namespace detail {
using cutlass::detail::float_e2m1_unpacksmem_t;
using cutlass::detail::float_e2m3_unpacksmem_t;
using cutlass::detail::float_e3m2_unpacksmem_t;
using cutlass::detail::float_e2m3_unpack8bits_t;
using cutlass::detail::float_e3m2_unpack8bits_t;
using cutlass::detail::type_erased_dynamic_float4_unpacksmem_t;
using cutlass::detail::type_erased_dynamic_float6_unpacksmem_t;
};

//
// Print utility
//

CUTE_HOST_DEVICE
void
print(half_t a) {
  printf("%f", static_cast<float>(a));
}

CUTE_HOST_DEVICE
void
print(bfloat16_t a) {
  printf("%f", static_cast<float>(a));
}

CUTE_HOST_DEVICE
void
print(tfloat32_t a) {
  printf("%f", static_cast<float>(a));
}

CUTE_HOST_DEVICE
void
print(float_e4m3_t a) {
  printf("%f", static_cast<float>(a));
}

CUTE_HOST_DEVICE
void
print(float_e5m2_t a) {
  printf("%f", static_cast<float>(a));
}

template <cutlass::detail::FpEncoding Encoding, class Derived>
CUTE_HOST_DEVICE
void
print(cutlass::float_exmy_base<Encoding, Derived> a) {
  printf("%f", static_cast<float>(a));
}

// Pretty Print utility

CUTE_HOST_DEVICE void
pretty_print(bfloat16_t v) {
  printf("%*.2f", 8, float(v));
}

CUTE_HOST_DEVICE void
pretty_print(half_t v) {
  printf("%*.2f", 8, float(v));
}

CUTE_HOST_DEVICE void
pretty_print(tfloat32_t v) {
  printf("%*.2e", 10, static_cast<float>(v));
}

CUTE_HOST_DEVICE void
pretty_print(float_e4m3_t t) {
  printf("%*.2f", 8, static_cast<float>(t));
}

CUTE_HOST_DEVICE void
pretty_print(float_e5m2_t t) {
  printf("%*.2f", 8, static_cast<float>(t));
}

template <cutlass::detail::FpEncoding Encoding, class Derived>
CUTE_HOST_DEVICE
void
pretty_print_float_exmy_base(cutlass::float_exmy_base<Encoding, Derived> t) {
  printf("%*.2f", 8, static_cast<float>(t));
}

} // namespace cute
