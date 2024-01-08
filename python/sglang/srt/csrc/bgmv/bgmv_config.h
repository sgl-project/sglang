#pragma once

template <int feat_in, int feat_out, typename T>
void bgmv_kernel(T* __restrict__ Y, const T* __restrict__ X,
                 const T* __restrict__ W, const int64_t* __restrict__ start_indicies,
                 const int64_t* __restrict__ lora_ranks, const int64_t* __restrict__ loc_indicies,
                 const int64_t* __restrict__ indicies, int64_t qkvo, int64_t batch_size,
                 const T* __restrict__ lora_scales);

// clang-format off

#define FOR_BGMV_WIDE(f, T, narrow) \
    f(T, narrow, 768) \
    f(T, narrow, 1024) \
    f(T, narrow, 1664) \
    f(T, narrow, 2048) \
    f(T, narrow, 2560) \
    f(T, narrow, 3072) \
    f(T, narrow, 3328) \
    f(T, narrow, 4096) \
    f(T, narrow, 5120) \
    f(T, narrow, 6656) \
    f(T, narrow, 7168) \
    f(T, narrow, 8192) \
    f(T, narrow, 9216) \
    f(T, narrow, 10240) \
    f(T, narrow, 11008) \
    f(T, narrow, 12288) \
    f(T, narrow, 13824) \
    f(T, narrow, 16384) \
    f(T, narrow, 20480) \
    f(T, narrow, 28672) \
    f(T, narrow, 36864) \
    f(T, narrow, 49152) \

#define FOR_BGMV_WIDE_NARROW(f, T) \
    FOR_BGMV_WIDE(f, T, 8) \
    FOR_BGMV_WIDE(f, T, 16) \
    FOR_BGMV_WIDE(f, T, 32) \
    FOR_BGMV_WIDE(f, T, 64) \
    FOR_BGMV_WIDE(f, T, 128)

// clang-format on
