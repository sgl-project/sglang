/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/cudaBf16Fallbacks.cuh"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#if ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace tensorrt_llm
{
namespace common
{

template <typename T>
inline __device__ T ldg(T const* val)
{
    return __ldg(val);
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 ldg(__nv_bfloat162 const* val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return val[0];
#else
    return __ldg(val);
#endif
}

template <>
inline __device__ __nv_bfloat16 ldg(__nv_bfloat16 const* val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return val[0];
#else
    return __ldg(val);
#endif
}
#endif // ENABLE_BF16

// Get type2 from type or vice versa (applied to half and bfloat16)
template <typename T>
struct TypeConverter
{
    using Type = half2;
}; // keep for generality

template <>
struct TypeConverter<half2>
{
    using Type = half;
};

template <>
struct TypeConverter<half>
{
    using Type = half2;
};

#if ENABLE_BF16
template <>
struct TypeConverter<__nv_bfloat162>
{
    using Type = __nv_bfloat16;
};

template <>
struct TypeConverter<__nv_bfloat16>
{
    using Type = __nv_bfloat162;
};
#endif // ENABLE_BF16

// Defined math operations (bfloat16 fallback to fp32 when it is not supported)
template <typename T>
inline __device__ T hadd2(T a, T b)
{
    return __hadd2(a, b);
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 hadd2(__nv_bfloat162 a, __nv_bfloat162 b)
{
    return bf16hadd2(a, b);
}
#endif // ENABLE_BF16

template <typename T>
inline __device__ T add(T a, T b)
{
    return a + b;
}

template <>
inline __device__ half2 add(half2 a, half2 b)
{
    return __hadd2(a, b);
}

template <>
inline __device__ half add(half a, half b)
{
    return __hadd(a, b);
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 add(__nv_bfloat162 a, __nv_bfloat162 b)
{
    return bf16hadd2(a, b);
}

template <>
inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b)
{
    return bf16hadd(a, b);
}

inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, float b)
{
    return bf16hadd(a, __float2bfloat16(b));
}
#endif // ENABLE_BF16

// applies to all 4 values addition
template <typename T>
inline __device__ T add(T a, T b, T c)
{
    return a + b + c;
}

#if ENABLE_BF16
inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c)
{
    return bf16hadd(a, b, c);
}

inline __device__ __nv_bfloat162 add(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c)
{
    return bf16hadd2(a, b, c);
}
#endif // ENABLE_BF16

// applies to all 4 values addition
template <typename T>
inline __device__ T add(T a, T b, T c, T d)
{
    return (T) ((float) a + (float) b + (float) c + (float) d);
}

#if ENABLE_BF16
inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c, __nv_bfloat16 d)
{
    return bf16hadd(a, b, c, d);
}
#endif // ENABLE_BF16

template <typename T>
inline __device__ T hsub2(T a, T b)
{
    return __hsub2(a, b);
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 hsub2(__nv_bfloat162 a, __nv_bfloat162 b)
{
    return bf16hsub2(a, b);
}
#endif // ENABLE_BF16

template <typename T>
inline __device__ T hmul2(T a, T b)
{
    return __hmul2(a, b);
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 hmul2(__nv_bfloat162 a, __nv_bfloat162 b)
{
    return bf16hmul2(a, b);
}
#endif // ENABLE_BF16

template <typename T>
inline __device__ T hmul2(T a, T b, T c)
{
    return a * b * c;
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 hmul2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c)
{
    return bf16hmul2(a, b, c);
}
#endif // ENABLE_BF16

template <typename T>
inline __device__ T mul(T a, T b, T c)
{
    return a * b * c;
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c)
{
    return bf16hmul(a, b, c);
}

inline __device__ __nv_bfloat162 mul(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c)
{
    return bf16hmul2(a, b, c);
}
#endif // ENABLE_BF16

template <typename T>
inline __device__ T fma(T a, T b, T c, T d)
{
    return a * b * c + d;
}

#if ENABLE_BF16
inline __device__ __nv_bfloat162 fma(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c, __nv_bfloat162 d)
{
    return bf16hfma2(a, b, c, d);
}
#endif // ENABLE_BF16

template <typename T>
inline __device__ T fma(T a, T b, T c)
{
    return a * b + c;
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 fma(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c)
{
    return bf16hfma2(a, b, c);
}

template <>
inline __device__ __nv_bfloat16 fma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c)
{
    return bf16hfma(a, b, c);
}
#endif // ENABLE_BF16

template <typename T>
inline __device__ T hexp2(T a)
{
    return h2exp(a);
}

#if ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 hexp2(__nv_bfloat162 a)
{
    return bf16exp2(a);
}
#endif // ENABLE_BF16

template <typename T_OUT, typename T_IN>
__device__ inline T_OUT cuda_cast(T_IN val)
{
    return val;
}

template <>
__device__ inline float2 cuda_cast<float2, int2>(int2 val)
{
    return make_float2(val.x, val.y);
}

template <>
__device__ inline float2 cuda_cast<float2, float>(float val)
{
    return make_float2(val, val);
}

template <>
__device__ inline float2 cuda_cast<float2, half2>(half2 val)
{
    return __half22float2(val);
}

template <>
__device__ inline half2 cuda_cast<half2, float2>(float2 val)
{
    return __float22half2_rn(val);
}

template <>
__device__ inline half2 cuda_cast<half2, float>(float val)
{
    return __float2half2_rn(val);
}

template <>
__device__ inline half2 cuda_cast<half2, half>(half val)
{
    return __half2half2(val);
}

template <>
__device__ inline int8_t cuda_cast<int8_t, half>(half val)
{
    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    union
    {
        half fp16;
        int16_t int16_in;
    };

    fp16 = val;
    asm volatile("cvt.rni.sat.s8.f16 %0, %1;" : "=h"(int16) : "h"(int16_in));
    return int8[0];
}

template <>
__device__ inline int16_t cuda_cast<int16_t, half2>(half2 val)
{
    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    int8[0] = cuda_cast<int8_t>(val.x);
    int8[1] = cuda_cast<int8_t>(val.y);
    return int16;
}

template <>
__device__ inline int8_t cuda_cast<int8_t, float>(float val)
{
    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=h"(int16) : "f"(val));
    return int8[0];
}

template <>
__device__ inline int16_t cuda_cast<int16_t, float2>(float2 val)
{
    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    int8[0] = cuda_cast<int8_t>(val.x);
    int8[1] = cuda_cast<int8_t>(val.y);
    return int16;
}

template <>
__device__ inline half2 cuda_cast<half2, int16_t>(int16_t val)
{
    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    int16 = val;
    return make_half2(int8[0], int8[1]);
}

template <>
__device__ inline float2 cuda_cast<float2, int16_t>(int16_t val)
{
    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    int16 = val;
    return make_float2(int8[0], int8[1]);
}

#ifdef ENABLE_BF16
template <>
__device__ inline __nv_bfloat16 cuda_cast(int32_t val)
{
    return static_cast<float>(val);
}

template <>
__device__ inline __nv_bfloat16 cuda_cast(int8_t val)
{
    return static_cast<float>(val);
}

template <>
__device__ inline int8_t cuda_cast(__nv_bfloat16 val)
{
    return static_cast<float>(val);
}

template <>
__device__ inline float cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 val)
{
    return __bfloat162float(val);
}

template <>
__device__ inline float2 cuda_cast<float2, __nv_bfloat162>(__nv_bfloat162 val)
{
    return bf1622float2(val);
}

template <>
__device__ inline half cuda_cast<half, __nv_bfloat16>(__nv_bfloat16 val)
{
    return __float2half(__bfloat162float(val));
}

template <>
__device__ inline int16_t cuda_cast<int16_t, __nv_bfloat162>(__nv_bfloat162 val)
{
    return bf1622int16(val);
}

template <>
__device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, float>(float val)
{
    return __float2bfloat16(val);
}

template <>
__device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, half>(half val)
{
    return __float2bfloat16(__half2float(val));
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, __nv_bfloat16>(__nv_bfloat16 val)
{
    return bf162bf162(val);
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, float>(float val)
{
    return __float2bfloat162_rn(val);
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, float2>(float2 val)
{
    return float22bf162(val);
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, int16_t>(int16_t val)
{
    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    int16 = val;
    __nv_bfloat162 res;
    res.x = cuda_cast<__nv_bfloat16>(int8[0]);
    res.y = cuda_cast<__nv_bfloat16>(int8[1]);
    return res;
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, half2>(half2 val)
{
    return float22bf162(__half22float2(val));
}

#endif // ENABLE BF16

template <typename T>
__device__ inline T cuda_abs(T val)
{
    assert(false);
    return {};
}

template <>
__device__ inline float cuda_abs(float val)
{
    return fabs(val);
}

template <>
__device__ inline float2 cuda_abs(float2 val)
{
    return make_float2(fabs(val.x), fabs(val.y));
}

template <>
__device__ inline half cuda_abs(half val)
{
    return __habs(val);
}

template <>
__device__ inline half2 cuda_abs(half2 val)
{
    return __habs2(val);
}

#ifdef ENABLE_BF16

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template <>
__device__ inline __nv_bfloat16 cuda_abs(__nv_bfloat16 val)
{
    return __habs(val);
}

template <>
__device__ inline __nv_bfloat162 cuda_abs(__nv_bfloat162 val)
{
    return __habs2(val);
}
#endif

#endif // ENABLE_FP16

template <typename To, typename Ti>
__device__ inline To cuda_sum(Ti val)
{
    return cuda_cast<To>(val);
};

template <typename To>
__device__ inline To cuda_sum(float2 val)
{
    return cuda_cast<To>(val.x + val.y);
};

// Unary maximum: compute the max of a vector type
template <typename To, typename Ti>
__device__ inline To cuda_max(Ti val)
{
    return cuda_cast<To>(val);
};

template <>
__device__ inline float cuda_max(float2 val)
{
    return fmaxf(val.x, val.y);
}

template <>
__device__ inline half cuda_max(half2 val)
{
    return __hmax(val.x, val.y);
}

#ifdef ENABLE_BF16
template <>
__device__ inline __nv_bfloat16 cuda_max(__nv_bfloat162 val)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    return __hmax(val.x, val.y);
#else
    assert(0);
    asm volatile("brkpt;\n" ::);
    return __nv_bfloat16(0);
#endif
}
#endif

// Binary maximum: compute the max of two values.
template <typename T>
__device__ inline T cuda_max(T val1, T val2)
{
    return (val1 > val2) ? val1 : val2;
}

template <>
__device__ inline float2 cuda_max(float2 val1, float2 val2)
{
    float2 out;
    out.x = fmaxf(val1.x, val2.x);
    out.y = fmaxf(val1.y, val2.y);
    return out;
}

template <>
__device__ inline half2 cuda_max(half2 val1, half2 val2)
{
    return __hmax2(val1, val2);
}

#ifdef ENABLE_BF16
template <>
__device__ inline __nv_bfloat162 cuda_max(__nv_bfloat162 val1, __nv_bfloat162 val2)
{
    return __hmax2(val1, val2);
}
#endif // ENABLE_BF16

// Binary maximum: compute the min of two values.
template <typename T>
__device__ inline T cuda_min(T val1, T val2)
{
    return (val1 < val2) ? val1 : val2;
}

template <>
__device__ inline float2 cuda_min(float2 val1, float2 val2)
{
    float2 out;
    out.x = fminf(val1.x, val2.x);
    out.y = fminf(val1.y, val2.y);
    return out;
}

template <>
__device__ inline half2 cuda_min(half2 val1, half2 val2)
{
    return __hmin2(val1, val2);
}

#ifdef ENABLE_BF16
template <>
__device__ inline __nv_bfloat162 cuda_min(__nv_bfloat162 val1, __nv_bfloat162 val2)
{
    return __hmin2(val1, val2);
}
#endif // ENABLE_BF16

// Helper function of clamping the val into the given range.
template <typename T>
inline __device__ T cuda_clamp(T val, T minVal, T maxVal)
{
    return cuda_min(cuda_max(val, minVal), maxVal);
}

#ifdef ENABLE_FP8
template <>
__device__ inline float2 cuda_cast<float2, __nv_fp8x2_e4m3>(__nv_fp8x2_e4m3 val)
{
    return bf1622float2(fp8x2_e4m3_to_bfloat2(&val));
}

template <>
__device__ inline half2 cuda_cast<half2, __nv_fp8x2_e4m3>(__nv_fp8x2_e4m3 val)
{
    return fp8x2_e4m3_to_half2(&val);
}

template <>
__device__ inline __nv_fp8x2_e4m3 cuda_cast<__nv_fp8x2_e4m3, float2>(float2 val)
{
    return __nv_fp8x2_e4m3(bf1622float2(float22bf162(val)));
}

template <>
__device__ inline __nv_fp8x2_e4m3 cuda_cast<__nv_fp8x2_e4m3, half2>(half2 val)
{
    return __nv_fp8x2_e4m3(cuda_cast<float2>(val));
}

template <>
__device__ inline __nv_fp8x2_e4m3 cuda_cast<__nv_fp8x2_e4m3, __nv_bfloat162>(__nv_bfloat162 val)
{
    return __nv_fp8x2_e4m3(cuda_cast<float2>(val));
}

template <>
__device__ inline __nv_fp8_e4m3 cuda_cast<__nv_fp8_e4m3, half>(half val)
{
    return __nv_fp8_e4m3(val);
}

template <>
__device__ inline __nv_fp8_e4m3 cuda_cast<__nv_fp8_e4m3, __nv_bfloat16>(__nv_bfloat16 val)
{
    return __nv_fp8_e4m3(val);
}

template <>
__device__ inline __nv_fp8_e4m3 cuda_cast<__nv_fp8_e4m3, float>(float val)
{
    return __nv_fp8_e4m3(val);
}

template <>
__device__ inline float cuda_cast<float, __nv_fp8_e4m3>(__nv_fp8_e4m3 val)
{
    return (float) val;
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, __nv_fp8x2_e4m3>(__nv_fp8x2_e4m3 val)
{
    return fp8x2_e4m3_to_bfloat2(&val);
}

template <>
__device__ inline int8_t cuda_cast<int8_t, __nv_fp8_e4m3>(__nv_fp8_e4m3 val)
{
    // no impl
    return 0;
}

template <>
__device__ inline __nv_fp8_e4m3 cuda_cast<__nv_fp8_e4m3, int8_t>(int8_t val)
{
    return cuda_cast<__nv_fp8_e4m3>(cuda_cast<__nv_bfloat16>(cuda_cast<float>(val)));
}

#endif // ENABLE_FP8

} // namespace common
} // namespace tensorrt_llm
