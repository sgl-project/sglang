/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace tensorrt_llm
{
namespace common
{

#ifdef ENABLE_BF16
inline __device__ float2 bf1622float2(const __nv_bfloat162 val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float2 f_val;
    f_val.x = __low2float(val);
    f_val.y = __high2float(val);
    return f_val;
#else
    return __bfloat1622float2(val);
#endif
}

inline __device__ int16_t bf1622int16(__nv_bfloat162 val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float2 f_val;
    f_val.x = max(min(__low2float(val), 127.f), -128.f);
    f_val.y = max(min(__high2float(val), 127.f), -128.f);

    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    int8[0] = static_cast<int8_t>(static_cast<short>(f_val.x));
    int8[1] = static_cast<int8_t>(static_cast<short>(f_val.y));
    return int16;
#else
    val = __hmin2(val, make_bfloat162(127., 127.));
    val = __hmax2(val, make_bfloat162(-128., -128.));

    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    int8[0] = static_cast<int8_t>(static_cast<short>(val.x));
    int8[1] = static_cast<int8_t>(static_cast<short>(val.y));
    return int16;
#endif
}

inline __device__ __nv_bfloat162 float22bf162(const float2 val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __floats2bfloat162_rn(val.x, val.y);
#else
    return __float22bfloat162_rn(val);
#endif
}

inline __device__ __nv_bfloat162 bf162bf162(const __nv_bfloat16 val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    __nv_bfloat162 val2;
    val2.x = val;
    val2.y = val;
    return val2;
#else
    return __bfloat162bfloat162(val);
#endif
}

inline __device__ __nv_bfloat162 bf16hadd2(const __nv_bfloat162 x, const __nv_bfloat162 y)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(fxl + fyl, fxh + fyh);
#else
    return __hadd2(x, y);
#endif
}

inline __device__ __nv_bfloat16 bf16hadd(const __nv_bfloat16 x, const __nv_bfloat16 y)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __float2bfloat16(__bfloat162float(x) + __bfloat162float(y));
#else
    return __hadd(x, y);
#endif
}

inline __device__ __nv_bfloat162 bf16hsub2(const __nv_bfloat162 x, const __nv_bfloat162 y)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(fxl - fyl, fxh - fyh);
#else
    return __hsub2(x, y);
#endif
}

inline __device__ __nv_bfloat16 bf16hsub(const __nv_bfloat16 x, const __nv_bfloat16 y)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __float2bfloat16(__bfloat162float(x) - __bfloat162float(y));
#else
    return __hsub(x, y);
#endif
}

inline __device__ __nv_bfloat162 bf16hmul2(const __nv_bfloat162 x, const __nv_bfloat162 y)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(fxl * fyl, fxh * fyh);
#else
    return __hmul2(x, y);
#endif
}

inline __device__ __nv_bfloat16 bf16hmul(const __nv_bfloat16 x, const __nv_bfloat16 y)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __float2bfloat16(__bfloat162float(x) * __bfloat162float(y));
#else
    return __hmul(x, y);
#endif
}

inline __device__ __nv_bfloat162 bf16hfma2(const __nv_bfloat162 x, const __nv_bfloat162 y, const __nv_bfloat162 z)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fxl, fxh, fyl, fyh, fzl, fzh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    fzl = __low2float(z);
    fzh = __high2float(z);
    return __floats2bfloat162_rn(fxl * fyl + fzl, fxh * fyh + fzh);
#else
    return __hfma2(x, y, z);
#endif
}

inline __device__ __nv_bfloat16 bf16hfma(const __nv_bfloat16 x, const __nv_bfloat16 y, const __nv_bfloat16 z)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __float2bfloat16(__bfloat162float(x) * __bfloat162float(y) + __bfloat162float(z));
#else
    return __hfma(x, y, z);
#endif
}

inline __device__ __nv_bfloat162 bf16exp2(const __nv_bfloat162 x)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fxl, fxh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    ;
    return __floats2bfloat162_rn(expf(fxl), expf(fxh));
#else
    return h2exp(x);
#endif
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
#if defined(CUDART_VERSION) && (CUDART_VERSION < 12020)

inline __device__ __nv_bfloat162 make_bfloat162(const __nv_bfloat16 x, const __nv_bfloat16 y)
{
    __nv_bfloat162 t;
    t.x = x;
    t.y = y;
    return t;
}
#endif
#endif

inline __device__ __nv_bfloat16 bf16hadd(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b) + __bfloat162float(c));
#else
    return a + b + c;
#endif
}

inline __device__ __nv_bfloat16 bf16hadd(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c, __nv_bfloat16 d)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b) + __bfloat162float(c) + __bfloat162float(d));
#else
    return (__nv_bfloat16) ((float) a + (float) b + (float) c + (float) d);
#endif
}

inline __device__ __nv_bfloat162 bf16hadd2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fal, fah, fbl, fbh, fcl, fch;
    fal = __low2float(a);
    fah = __high2float(a);
    fbl = __low2float(b);
    fbh = __high2float(b);
    fcl = __low2float(c);
    fch = __high2float(c);
    return __floats2bfloat162_rn(fal + fbl + fcl, fah + fbh + fch);
#else
    return a + b + c;
#endif
}

inline __device__ __nv_bfloat16 bf16hmul(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b) * __bfloat162float(c));
#else
    return a * b * c;
#endif
}

inline __device__ __nv_bfloat162 bf16hmul2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fal, fah, fbl, fbh, fcl, fch;
    fal = __low2float(a);
    fah = __high2float(a);
    fbl = __low2float(b);
    fbh = __high2float(b);
    fcl = __low2float(c);
    fch = __high2float(c);
    return __floats2bfloat162_rn(fal * fbl * fcl, fah * fbh * fch);
#else
    return a * b * c;
#endif
}

inline __device__ __nv_bfloat162 bf16hfma2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c, __nv_bfloat162 d)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fal, fah, fbl, fbh, fcl, fch, fdl, fdh;
    fal = __low2float(a);
    fah = __high2float(a);
    fbl = __low2float(b);
    fbh = __high2float(b);
    fcl = __low2float(c);
    fch = __high2float(c);
    fdl = __low2float(d);
    fdh = __high2float(d);
    return __floats2bfloat162_rn(fal * fbl * fcl + fdl, fah * fbh * fch + fdh);
#else
    return a * b * c + d;
#endif
}

#endif // ENABLE_BF16

} // namespace common
} // namespace tensorrt_llm

// Operator definitions intentionally in global namespace
namespace
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
#if defined(CUDART_VERSION) && (CUDART_VERSION < 12020)

inline __device__ __nv_bfloat162 operator*(const __nv_bfloat162 x, const __nv_bfloat162 y)
{
    return tensorrt_llm::common::bf16hmul2(x, y);
};

inline __device__ __nv_bfloat162 operator+(const __nv_bfloat162 x, const __nv_bfloat162 y)
{
    return tensorrt_llm::common::bf16hadd2(x, y);
};
#endif
#endif
} // namespace
