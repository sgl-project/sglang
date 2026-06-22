/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector_types.h>

#if defined(__CUDACC_VER_MAJOR__)
#if __CUDACC_VER_MAJOR__ > 11 || \
    (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 8)
#include <cuda_fp8.h>
#define OMNIDREAMS_NATIVE_HAS_FP8 1
#endif
#endif
#ifndef OMNIDREAMS_NATIVE_HAS_FP8
#define OMNIDREAMS_NATIVE_HAS_FP8 0
#endif

#include "native_common/macros.h"

namespace omnidreams_native {

using float16_t = __half;
using bfloat16_t = __nv_bfloat16;
using float32_t = float;

#if OMNIDREAMS_NATIVE_HAS_FP8
using float8_e4m3_t = __nv_fp8_e4m3;
#endif

template <typename T>
struct ScalarTraits;

template <>
struct ScalarTraits<float16_t> {
  using type = float16_t;
  using vec2_type = __half2;
  using accumulator_type = float;
  static constexpr int size_bytes = 2;
  static constexpr bool is_float16 = true;
  static constexpr bool is_bfloat16 = false;
  static constexpr bool is_float8 = false;
  static constexpr bool is_integer = false;
  static constexpr const char* name = "float16";
};

template <>
struct ScalarTraits<bfloat16_t> {
  using type = bfloat16_t;
  using vec2_type = __nv_bfloat162;
  using accumulator_type = float;
  static constexpr int size_bytes = 2;
  static constexpr bool is_float16 = false;
  static constexpr bool is_bfloat16 = true;
  static constexpr bool is_float8 = false;
  static constexpr bool is_integer = false;
  static constexpr const char* name = "bfloat16";
};

template <>
struct ScalarTraits<float32_t> {
  using type = float32_t;
  using vec2_type = float2;
  using accumulator_type = float;
  static constexpr int size_bytes = 4;
  static constexpr bool is_float16 = false;
  static constexpr bool is_bfloat16 = false;
  static constexpr bool is_float8 = false;
  static constexpr bool is_integer = false;
  static constexpr const char* name = "float32";
};

template <>
struct ScalarTraits<int8_t> {
  using type = int8_t;
  using accumulator_type = int32_t;
  static constexpr int size_bytes = 1;
  static constexpr bool is_float16 = false;
  static constexpr bool is_bfloat16 = false;
  static constexpr bool is_float8 = false;
  static constexpr bool is_integer = true;
  static constexpr const char* name = "int8";
};

#if OMNIDREAMS_NATIVE_HAS_FP8
template <>
struct ScalarTraits<float8_e4m3_t> {
  using type = float8_e4m3_t;
  using accumulator_type = float;
  static constexpr int size_bytes = 1;
  static constexpr bool is_float16 = false;
  static constexpr bool is_bfloat16 = false;
  static constexpr bool is_float8 = true;
  static constexpr bool is_integer = false;
  static constexpr const char* name = "float8_e4m3";
};
#endif

template <typename T>
constexpr bool is_float16_v = ScalarTraits<T>::is_float16;
template <typename T>
constexpr bool is_bfloat16_v = ScalarTraits<T>::is_bfloat16;
template <typename T>
constexpr bool is_float8_v = ScalarTraits<T>::is_float8;
template <typename T>
constexpr bool is_integer_v = ScalarTraits<T>::is_integer;
template <typename T>
constexpr bool is_16bit_float_v =
    ScalarTraits<T>::is_float16 || ScalarTraits<T>::is_bfloat16;

OMNIDREAMS_NATIVE_HOST_DEVICE inline float to_float(float16_t x) {
  return __half2float(x);
}

OMNIDREAMS_NATIVE_HOST_DEVICE inline float to_float(bfloat16_t x) {
  return __bfloat162float(x);
}

OMNIDREAMS_NATIVE_HOST_DEVICE inline float to_float(float x) {
  return x;
}

template <typename T>
OMNIDREAMS_NATIVE_HOST_DEVICE inline T from_float(float x);

template <>
OMNIDREAMS_NATIVE_HOST_DEVICE inline float16_t from_float<float16_t>(float x) {
  return __float2half(x);
}

template <>
OMNIDREAMS_NATIVE_HOST_DEVICE inline bfloat16_t from_float<bfloat16_t>(float x) {
  return __float2bfloat16(x);
}

template <>
OMNIDREAMS_NATIVE_HOST_DEVICE inline float from_float<float>(float x) {
  return x;
}

template <typename To, typename From>
OMNIDREAMS_NATIVE_HOST_DEVICE inline To scalar_cast(From x) {
  static_assert(sizeof(From) == sizeof(To), "scalar_cast requires same-size types");
  return reinterpret_cast<const To&>(x);
}

template <typename To, typename From>
OMNIDREAMS_NATIVE_HOST_DEVICE inline To* ptr_cast(From* p) {
  static_assert(sizeof(From) == sizeof(To), "ptr_cast requires same-size element types");
  return reinterpret_cast<To*>(p);
}

template <typename To, typename From>
OMNIDREAMS_NATIVE_HOST_DEVICE inline const To* ptr_cast(const From* p) {
  static_assert(sizeof(From) == sizeof(To), "ptr_cast requires same-size element types");
  return reinterpret_cast<const To*>(p);
}

}  // namespace omnidreams_native

#if defined(CUTLASS_VERSION)
static_assert(
    sizeof(omnidreams_native::float16_t) == sizeof(cutlass::half_t),
    "omnidreams_native::float16_t and cutlass::half_t must be same size");
static_assert(
    sizeof(omnidreams_native::bfloat16_t) == sizeof(cutlass::bfloat16_t),
    "omnidreams_native::bfloat16_t and cutlass::bfloat16_t must be same size");
#if OMNIDREAMS_NATIVE_HAS_FP8
static_assert(
    sizeof(omnidreams_native::float8_e4m3_t) == sizeof(cutlass::float_e4m3_t),
    "omnidreams_native::float8_e4m3_t and cutlass::float_e4m3_t must be same size");
#endif
#endif
