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
#include "cutlass/cutlass.h"
#if defined(__CUDACC_RTC__)
#include CUDA_STD_HEADER(cstdint)
#else
#include <cstdint>
#endif

#include <cute/config.hpp>          // CUTE_STL_NAMESPACE

#include <cutlass/numeric_types.h>  // cutlass::int2b_t, cutlass::int4b_t

namespace cute
{

//
// Signed integers
//

using int2_t = cutlass::int2b_t;
using int4_t = cutlass::int4b_t;
using int6_t = cutlass::int6b_t;
using CUTE_STL_NAMESPACE::int8_t;
using CUTE_STL_NAMESPACE::int16_t;
using CUTE_STL_NAMESPACE::int32_t;
using CUTE_STL_NAMESPACE::int64_t;

template <int N> struct int_bit;
template <> struct int_bit<  2>  { using type = int2_t; };
template <> struct int_bit<  4>  { using type = int4_t; };
template <> struct int_bit<  8>  { using type = int8_t;  };
template <> struct int_bit< 16>  { using type = int16_t; };
template <> struct int_bit< 32>  { using type = int32_t; };
template <> struct int_bit< 64>  { using type = int64_t; };

template <int N>
using int_bit_t = typename int_bit<N>::type;

template <int N>
using int_byte = int_bit<8*N>;

template <int N>
using int_byte_t = typename int_byte<N>::type;

//
// Unsigned integers
//

using uint1_t = cutlass::uint1b_t;
using uint2_t = cutlass::uint2b_t;
using uint4_t = cutlass::uint4b_t;
using uint6_t = cutlass::uint6b_t;
using CUTE_STL_NAMESPACE::uint8_t;
using CUTE_STL_NAMESPACE::uint16_t;
using CUTE_STL_NAMESPACE::uint32_t;
using CUTE_STL_NAMESPACE::uint64_t;
using cutlass::uint128_t;
using cutlass::uint256_t;

template <int N> struct uint_bit;
template <> struct uint_bit<  1> { using type = uint1_t; };
template <> struct uint_bit<  2> { using type = uint2_t; };
template <> struct uint_bit<  4> { using type = uint4_t; };
template <> struct uint_bit<  6> { using type = uint6_t; };
template <> struct uint_bit<  8> { using type = uint8_t;  };
template <> struct uint_bit< 16> { using type = uint16_t; };
template <> struct uint_bit< 32> { using type = uint32_t; };
template <> struct uint_bit< 64> { using type = uint64_t; };
template <> struct uint_bit<128> { using type = cutlass::uint128_t; };
template <> struct uint_bit<256> { using type = cutlass::uint256_t; };

template <int N>
using uint_bit_t = typename uint_bit<N>::type;

template <int N>
using uint_byte = uint_bit<8*N>;

template <int N>
using uint_byte_t = typename uint_byte<N>::type;

} // namespace cute
