#pragma once

#include <cuda_fp8.h>
#include <cuda_bf16.h>

#include "../defines.h"

namespace sm90::decode::sparse_fp8 {

struct fp8x8 {
    __nv_fp8x4_e4m3 lo;
    __nv_fp8x4_e4m3 hi;
};

struct fp8x16 {
    fp8x8 lo;
    fp8x8 hi;
};

__device__ __forceinline__
bf16x8 cvt_fp8x8_bf16x8(const fp8x8 &inputs, const __nv_bfloat162 &scale_bf162) {
    #define DEQUANT_FP8x4(OUTPUT_BF16_LO, OUTPUT_BF16_HI, FP8x4) \
    { \
        float4 fp32x4 = (float4)(FP8x4); \
        OUTPUT_BF16_LO = __float22bfloat162_rn({fp32x4.x, fp32x4.y})*scale_bf162; \
        OUTPUT_BF16_HI = __float22bfloat162_rn({fp32x4.z, fp32x4.w})*scale_bf162; \
    }

    bf16x8 result;
    DEQUANT_FP8x4(result.a01, result.a23, inputs.lo);
    DEQUANT_FP8x4(result.a45, result.a67, inputs.hi);

    return result;
}

enum class L1CacheHint {
    NO_ALLOCATE,
    EVICT_FIRST,
    EVICT_NORMAL,
    EVICT_LAST
};

enum class L2PrefetchHint {
    B64,
    B128,
    B256
};

template<
    typename T,
    L1CacheHint l1_cache_hint,
    L2PrefetchHint l2_prefetch_hint
>
__device__ __forceinline__
T load_128b_from_gmem(const void* addr) {
    static_assert(sizeof(T) == 128/8);
    int4 ret;

    #define EXEC(L1_HINT_STR, L2_HINT_STR) { \
        asm volatile("ld.global.nc.L1::" L1_HINT_STR ".L2::" L2_HINT_STR ".v4.s32 {%0, %1, %2, %3}, [%4];" \
            : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) \
            : "l"(addr)); \
    }

    #define DISPATCH_L2(L1_HINT_STR) { \
        if constexpr(l2_prefetch_hint == L2PrefetchHint::B64) \
            EXEC(L1_HINT_STR, "64B") \
        else if constexpr(l2_prefetch_hint == L2PrefetchHint::B128) \
            EXEC(L1_HINT_STR, "128B") \
        else if constexpr(l2_prefetch_hint == L2PrefetchHint::B256) \
            EXEC(L1_HINT_STR, "256B") \
    }

    if constexpr(l1_cache_hint == L1CacheHint::NO_ALLOCATE)
        DISPATCH_L2("no_allocate")
    else if constexpr(l1_cache_hint == L1CacheHint::EVICT_FIRST)
        DISPATCH_L2("evict_first")
    else if constexpr(l1_cache_hint == L1CacheHint::EVICT_NORMAL)
        DISPATCH_L2("evict_normal")
    else if constexpr(l1_cache_hint == L1CacheHint::EVICT_LAST)
        DISPATCH_L2("evict_last")

    #undef EXEC
    #undef DISPATCH_L2
    return *reinterpret_cast<T*>(&ret);
}

template<
    typename T,
    L1CacheHint l1_cache_hint,
    L2PrefetchHint l2_prefetch_hint
>
__device__ __forceinline__
T load_64b_from_gmem(const void* addr) {
    static_assert(sizeof(T) == 64/8);
    int2 ret;

    #define EXEC(L1_HINT_STR, L2_HINT_STR) { \
        asm volatile("ld.global.nc.L1::" L1_HINT_STR ".L2::" L2_HINT_STR ".v2.s32 {%0, %1}, [%2];" \
            : "=r"(ret.x), "=r"(ret.y) \
            : "l"(addr)); \
    }

    #define DISPATCH_L2(L1_HINT_STR) { \
        if constexpr(l2_prefetch_hint == L2PrefetchHint::B64) \
            EXEC(L1_HINT_STR, "64B") \
        else if constexpr(l2_prefetch_hint == L2PrefetchHint::B128) \
            EXEC(L1_HINT_STR, "128B") \
        else if constexpr(l2_prefetch_hint == L2PrefetchHint::B256) \
            EXEC(L1_HINT_STR, "256B") \
    }

    if constexpr(l1_cache_hint == L1CacheHint::NO_ALLOCATE)
        DISPATCH_L2("no_allocate")
    else if constexpr(l1_cache_hint == L1CacheHint::EVICT_FIRST)
        DISPATCH_L2("evict_first")
    else if constexpr(l1_cache_hint == L1CacheHint::EVICT_NORMAL)
        DISPATCH_L2("evict_normal")
    else if constexpr(l1_cache_hint == L1CacheHint::EVICT_LAST)
        DISPATCH_L2("evict_last")

    #undef EXEC
    #undef DISPATCH_L2
    return *reinterpret_cast<T*>(&ret);
}

}
