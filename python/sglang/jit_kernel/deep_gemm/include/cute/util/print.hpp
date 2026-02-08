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

#include <cute/config.hpp>           // CUTE_HOST_DEVICE
#include <cute/util/type_traits.hpp> // cute::is_valid
#include <cute/numeric/numeric_types.hpp> 

//
// CUDA compatible print and printf
//

namespace cute
{

CUTE_HOST_DEVICE
int
num_digits(int x)
{
  return (x < 10 ? 1 :
          (x < 100 ? 2 :
           (x < 1000 ? 3 :
            (x < 10000 ? 4 :
             (x < 100000 ? 5 :
              (x < 1000000 ? 6 :
               (x < 10000000 ? 7 :
                (x < 100000000 ? 8 :
                 (x < 1000000000 ? 9 :
                  10)))))))));
}

//
// print dispatcher
//

CUTE_HOST_DEVICE
void
print(char c) {
  printf("%c", c);
}

CUTE_HOST_DEVICE
void
print(signed char a) {
  printf("%d", static_cast<int>(a));
}

CUTE_HOST_DEVICE
void
print(unsigned char a) {
  printf("%u", static_cast<unsigned int>(a));
}

CUTE_HOST_DEVICE
void
print(short a) {
  printf("%hd", a);
}

CUTE_HOST_DEVICE
void
print(unsigned short a) {
  printf("%hu", a);
}

CUTE_HOST_DEVICE
void
print(int a) {
  printf("%d", a);
}

CUTE_HOST_DEVICE
void
print(uint1b_t a) {
  printf("%d", int(a));
}

CUTE_HOST_DEVICE
void
print(int2b_t a) {
  printf("%d", int(a));
}

CUTE_HOST_DEVICE
void
print(uint2b_t a) {
  printf("%d", int(a));
}

CUTE_HOST_DEVICE
void
print(int4b_t a) {
  printf("%d", int(a));
}

CUTE_HOST_DEVICE
void
print(uint4b_t a) {
  printf("%d", int(a));
}

CUTE_HOST_DEVICE
void
print(bin1_t a) {
  printf("%d", int(a));
}

CUTE_HOST_DEVICE
void
print(unsigned int a) {
  printf("%u", a);
}

CUTE_HOST_DEVICE
void
print(long a) {
  printf("%ld", a);
}

CUTE_HOST_DEVICE
void
print(unsigned long a) {
  printf("%lu", a);
}

CUTE_HOST_DEVICE
void
print(long long a) {
  printf("%lld", a);
}

CUTE_HOST_DEVICE
void
print(unsigned long long a) {
  printf("%llu", a);
}

CUTE_HOST_DEVICE
void
print(float a) {
  printf("%f", a);
}

CUTE_HOST_DEVICE
void
print(double a) {
  printf("%f", a);
}

template <class... T>
CUTE_HOST_DEVICE
void
print(char const* format, T const&... t) {
  printf(format, t...);
}

CUTE_HOST_DEVICE
void
print(char const* format) {
  printf("%s", format);
}

//
// pretty printing
//

CUTE_HOST_DEVICE void
pretty_print(uint1b_t a) {
  printf("%*d", 3, int(a));
}

CUTE_HOST_DEVICE void
pretty_print(int2b_t a) {
  printf("%*d", 5, int(a));
}

CUTE_HOST_DEVICE void
pretty_print(uint2b_t a) {
  printf("%*d", 5, int(a));
}

CUTE_HOST_DEVICE void
pretty_print(int4b_t a) {
  printf("%*d", 5, int(a));
}

CUTE_HOST_DEVICE void
pretty_print(uint4b_t a) {
  printf("%*d", 5, int(a));
}

CUTE_HOST_DEVICE void
pretty_print(bool v) {
  printf("%*d", 3, int(v));
}

CUTE_HOST_DEVICE void
pretty_print(int32_t v) {
  printf("%*d", 5, v);
}

CUTE_HOST_DEVICE void
pretty_print(uint32_t v) {
  printf("%*d", 5, v);
}

CUTE_HOST_DEVICE void
pretty_print(int64_t v) {
  printf("%*lld", 5, static_cast<long long>(v));
}

CUTE_HOST_DEVICE void
pretty_print(uint64_t v) {
  printf("%*llu", 5, static_cast<unsigned long long>(v));
}

CUTE_HOST_DEVICE void
pretty_print(float v) {
  printf("%*.2e", 10, v);
}

CUTE_HOST_DEVICE void
pretty_print(double v) {
  printf("%*.3e", 11, v);
}

template <class T>
CUTE_HOST_DEVICE void
pretty_print(T t) {
  constexpr auto has_print_exmy_base = cute::is_valid([](auto t) -> decltype(pretty_print_float_exmy_base(t)) {}, t);  
  if constexpr (has_print_exmy_base) {   
  pretty_print_float_exmy_base(t);       
  } else {                               
  printf("  "); print(t);
  }                                      
}

} // end namespace cute
