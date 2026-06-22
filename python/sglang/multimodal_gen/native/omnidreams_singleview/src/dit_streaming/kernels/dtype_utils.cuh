// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// dtype_utils.cuh - Unified dtype support for fp16 and bf16
// Provides type traits, conversion utilities, and common operations for both dtypes

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cutlass/numeric_types.h>
#include <cutlass/half.h>
#include <cutlass/bfloat16.h>

namespace omnidreams_singleview {

// =============================================================================
// Type Traits - Map between CUDA native types and CUTLASS types
// =============================================================================

template <typename T>
struct DTypeTraits;

// FP16 specialization
template <>
struct DTypeTraits<cutlass::half_t> {
    using cuda_type = __half;
    using cuda_vec2 = __half2;
    using cutlass_type = cutlass::half_t;
    using accumulator_type = float;
    static constexpr bool is_fp16 = true;
    static constexpr bool is_bf16 = false;
    static constexpr const char* name = "fp16";
};

template <>
struct DTypeTraits<__half> : DTypeTraits<cutlass::half_t> {};

// BF16 specialization
template <>
struct DTypeTraits<cutlass::bfloat16_t> {
    using cuda_type = __nv_bfloat16;
    using cuda_vec2 = __nv_bfloat162;
    using cutlass_type = cutlass::bfloat16_t;
    using accumulator_type = float;
    static constexpr bool is_fp16 = false;
    static constexpr bool is_bf16 = true;
    static constexpr const char* name = "bf16";
};

template <>
struct DTypeTraits<__nv_bfloat16> : DTypeTraits<cutlass::bfloat16_t> {};

// =============================================================================
// Conversion Utilities - Device functions for type conversion
// =============================================================================

// To float conversions
__device__ __forceinline__ float to_float(cutlass::half_t x) {
    return __half2float(reinterpret_cast<const __half&>(x));
}

__device__ __forceinline__ float to_float(cutlass::bfloat16_t x) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16&>(x));
}

__device__ __forceinline__ float to_float(__half x) {
    return __half2float(x);
}

__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// From float conversions
template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ cutlass::half_t from_float<cutlass::half_t>(float x) {
    __half h = __float2half(x);
    return reinterpret_cast<cutlass::half_t&>(h);
}

template <>
__device__ __forceinline__ cutlass::bfloat16_t from_float<cutlass::bfloat16_t>(float x) {
    __nv_bfloat16 b = __float2bfloat16(x);
    return reinterpret_cast<cutlass::bfloat16_t&>(b);
}

template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
    return __float2half(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

// =============================================================================
// Vector Operations - For vectorized memory access
// =============================================================================

// Load 2 elements as vector
template <typename T>
__device__ __forceinline__ float2 load_vec2_as_float2(const T* ptr);

template <>
__device__ __forceinline__ float2 load_vec2_as_float2<cutlass::half_t>(const cutlass::half_t* ptr) {
    __half2 v = *reinterpret_cast<const __half2*>(ptr);
    return __half22float2(v);
}

template <>
__device__ __forceinline__ float2 load_vec2_as_float2<cutlass::bfloat16_t>(const cutlass::bfloat16_t* ptr) {
    __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(ptr);
    return __bfloat1622float2(v);
}

// Store 2 floats as vector type
template <typename T>
__device__ __forceinline__ void store_float2_as_vec2(T* ptr, float2 v);

template <>
__device__ __forceinline__ void store_float2_as_vec2<cutlass::half_t>(cutlass::half_t* ptr, float2 v) {
    __half2 h2 = __float22half2_rn(v);
    *reinterpret_cast<__half2*>(ptr) = h2;
}

template <>
__device__ __forceinline__ void store_float2_as_vec2<cutlass::bfloat16_t>(cutlass::bfloat16_t* ptr, float2 v) {
    __nv_bfloat162 b2 = __float22bfloat162_rn(v);
    *reinterpret_cast<__nv_bfloat162*>(ptr) = b2;
}

// =============================================================================
// Make vec2 from two floats
// =============================================================================

template <typename T>
__device__ __forceinline__ typename DTypeTraits<T>::cuda_vec2 make_vec2(float a, float b);

template <>
__device__ __forceinline__ __half2 make_vec2<cutlass::half_t>(float a, float b) {
    return __floats2half2_rn(a, b);
}

template <>
__device__ __forceinline__ __nv_bfloat162 make_vec2<cutlass::bfloat16_t>(float a, float b) {
    return __floats2bfloat162_rn(a, b);
}

// =============================================================================
// Fused multiply-add for vec2 types
// =============================================================================

__device__ __forceinline__ __half2 fma_vec2(__half2 a, __half2 b, __half2 c) {
    return __hfma2(a, b, c);
}

__device__ __forceinline__ __nv_bfloat162 fma_vec2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
    return __hfma2(a, b, c);  // BF16 uses same intrinsic name in CUDA 11+
}

// =============================================================================
// Numeric Constants
// =============================================================================

template <typename T>
__device__ __forceinline__ T zero();

template <>
__device__ __forceinline__ cutlass::half_t zero<cutlass::half_t>() {
    return cutlass::half_t(0.0f);
}

template <>
__device__ __forceinline__ cutlass::bfloat16_t zero<cutlass::bfloat16_t>() {
    return cutlass::bfloat16_t(0.0f);
}

template <typename T>
__device__ __forceinline__ T one();

template <>
__device__ __forceinline__ cutlass::half_t one<cutlass::half_t>() {
    return cutlass::half_t(1.0f);
}

template <>
__device__ __forceinline__ cutlass::bfloat16_t one<cutlass::bfloat16_t>() {
    return cutlass::bfloat16_t(1.0f);
}

} // namespace omnidreams_singleview
