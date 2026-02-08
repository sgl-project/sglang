/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/* \file
  \brief Performs comparison between two elements with support for floating-point comparisons.
*/

#pragma once

#include "numeric_types.h"
#include "complex.h"

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename U = T>
CUTLASS_HOST_DEVICE
bool relatively_equal(T a, T b, U epsilon, U nonzero_floor);

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// This floating-point comparison function implements the method described in
//
// https://floating-point-gui.de/errors/comparison/
//
template <typename T>
CUTLASS_HOST_DEVICE
bool relatively_equal_float(T a, T b, T epsilon, T nonzero_floor) {
  
#if defined(__CUDACC_RTC__)
  using cuda::std::abs;
#else
  using std::abs;
#endif

  T abs_A = abs(a);
  T abs_B = abs(b);
  T diff = abs(a - b);
  T zero = T(0);

  if (a == b) {
    return true;
  }
  else if (a == zero || b == zero || (abs_A + abs_B) < nonzero_floor) {
    return diff < epsilon * nonzero_floor;
  }
  
  return diff < epsilon * (abs_A + abs_B);
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<bool>(bool a, bool b, bool, bool) {
  return (a == b);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<uint1b_t>(uint1b_t a, uint1b_t b, uint1b_t, uint1b_t) {
  return (a == b);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<int2b_t>(int2b_t a, int2b_t b, int2b_t, int2b_t) {
  return (a == b);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<uint2b_t>(uint2b_t a, uint2b_t b, uint2b_t, uint2b_t) {
  return (a == b);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<int4b_t>(int4b_t a, int4b_t b, int4b_t, int4b_t) {
  return (a == b);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<uint4b_t>(uint4b_t a, uint4b_t b, uint4b_t, uint4b_t) {
  return (a == b);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<int8_t>(int8_t a, int8_t b, int8_t, int8_t) {
  return (a == b);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<uint8_t>(uint8_t a, uint8_t b, uint8_t, uint8_t) {
  return (a == b);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<int16_t>(int16_t a, int16_t b, int16_t, int16_t) {
  return (a == b);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<uint16_t>(uint16_t a, uint16_t b, uint16_t, uint16_t) {
  return (a == b);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<int32_t>(int32_t a, int32_t b, int32_t, int32_t) {
  return (a == b);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<uint32_t>(uint32_t a, uint32_t b, uint32_t, uint32_t) {
  return (a == b);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<int64_t>(int64_t a, int64_t b, int64_t, int64_t) {
  return (a == b);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<uint64_t>(uint64_t a, uint64_t b, uint64_t, uint64_t) {
  return (a == b);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<float_e4m3_t>(float_e4m3_t a, float_e4m3_t b, float_e4m3_t epsilon, float_e4m3_t nonzero_floor) {
  return detail::relatively_equal_float<float>(a, b, epsilon, nonzero_floor);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<float_e5m2_t>(float_e5m2_t a, float_e5m2_t b, float_e5m2_t epsilon, float_e5m2_t nonzero_floor) {
  return detail::relatively_equal_float<float>(a, b, epsilon, nonzero_floor);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<half_t>(half_t a, half_t b, half_t epsilon, half_t nonzero_floor) {
  return detail::relatively_equal_float(a, b, epsilon, nonzero_floor);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<bfloat16_t>(
  bfloat16_t a, 
  bfloat16_t b, 
  bfloat16_t epsilon, 
  bfloat16_t nonzero_floor) {
  
  return detail::relatively_equal_float(a, b, epsilon, nonzero_floor);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<tfloat32_t>(
  tfloat32_t a, 
  tfloat32_t b, 
  tfloat32_t epsilon, 
  tfloat32_t nonzero_floor) {
  
  return detail::relatively_equal_float(a, b, epsilon, nonzero_floor);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<float>(float a, float b, float epsilon, float nonzero_floor) {
  return detail::relatively_equal_float(a, b, epsilon, nonzero_floor);
}


template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<double>(double a, double b, double epsilon, double nonzero_floor) {
  return detail::relatively_equal_float(a, b, epsilon, nonzero_floor);
}

template<typename T>
CUTLASS_HOST_DEVICE
bool relatively_equal(complex<T> a, complex<T> b, T epsilon, T nonzero_floor) {
#if defined(__CUDACC_RTC__)
  using cuda::std::abs;
#else
  using std::abs;
#endif

  T abs_A = abs(a);
  T abs_B = abs(b);
  T diff = abs(a - b);
  complex<T> zero = complex<T>{T{}, T{}};

  if (a == b) {
    return true;
  }
  else if (a == zero || b == zero || diff < nonzero_floor) {
    return diff < epsilon * nonzero_floor;
  }

  return diff < epsilon * (abs_A + abs_B);
}

template <typename T>
CUTLASS_HOST_DEVICE 
bool relatively_equal(complex<T> a,  complex<T> b, complex<T> epsilon, complex<T> nonzero_floor) {
#if defined(__CUDACC_RTC__)
  using cuda::std::abs;
#else
  using std::abs;
#endif

  T abs_A = abs(a);
  T abs_B = abs(b);
  complex<T> diff = a - b;
  T abs_diff = abs(diff);
  complex<T> zero = complex<T>{T{}, T{}};

  if (a == b) {
    return true;
  }
  else if (a == zero || b == zero || abs_diff < abs(nonzero_floor)) {
    return abs_diff < abs(epsilon * nonzero_floor);
  }

  return abs_diff < abs(epsilon) * (abs_A + abs_B);
}


template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<float_e2m3_t>(float_e2m3_t a, float_e2m3_t b, float_e2m3_t epsilon, float_e2m3_t nonzero_floor) {
  return detail::relatively_equal_float<float>(a, b, epsilon, nonzero_floor);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<float_e3m2_t>(float_e3m2_t a, float_e3m2_t b, float_e3m2_t epsilon, float_e3m2_t nonzero_floor) {
  return detail::relatively_equal_float<float>(a, b, epsilon, nonzero_floor);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<float_e2m1_t>(float_e2m1_t a, float_e2m1_t b, float_e2m1_t epsilon, float_e2m1_t nonzero_floor) {
  return detail::relatively_equal_float<float>(a, b, epsilon, nonzero_floor);
}
template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<float_ue8m0_t>(float_ue8m0_t a, float_ue8m0_t b, float_ue8m0_t epsilon, float_ue8m0_t nonzero_floor) {
  return detail::relatively_equal_float<float>(a, b, epsilon, nonzero_floor);
}

template <>
CUTLASS_HOST_DEVICE
bool relatively_equal<float_ue4m3_t>(float_ue4m3_t a, float_ue4m3_t b, float_ue4m3_t epsilon, float_ue4m3_t nonzero_floor) {
  return detail::relatively_equal_float<float>(a, b, epsilon, nonzero_floor);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
