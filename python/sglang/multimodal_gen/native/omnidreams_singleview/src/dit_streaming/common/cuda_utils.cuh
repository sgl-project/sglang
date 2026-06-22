// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

// Signal to backends that common provides build_past_shifted helpers
#ifndef OMNIDREAMS_SINGLEVIEW_BUILD_PAST_IN_COMMON
#define OMNIDREAMS_SINGLEVIEW_BUILD_PAST_IN_COMMON 1
#endif

// Forward declaration for trim kernel (defined later in this header)
__global__ void ntchw_trim_front_half_kernel(
    const half* __restrict__ in,
    half* __restrict__ out,
    int N_, int T_, int C_, int H_, int W_, int K_);

namespace omnidreams_singleview {

// Debug/assert helpers
#if !defined(OMNIDREAMS_SINGLEVIEW_DEBUG)
#define OMNIDREAMS_SINGLEVIEW_DEBUG 0
#endif

// Variadic macro to support both OMNIDREAMS_SINGLEVIEW_CUDA_CHECK() and OMNIDREAMS_SINGLEVIEW_CUDA_CHECK(cudaCall())
#if OMNIDREAMS_SINGLEVIEW_DEBUG
#define OMNIDREAMS_SINGLEVIEW_ASSERT(cond) do { if (!(cond)) { fprintf(stderr, "[OMNIDREAMS_SINGLEVIEW_ASSERT] %s:%d: assertion failed: %s\n", __FILE__, __LINE__, #cond); throw std::runtime_error("OMNIDREAMS_SINGLEVIEW_ASSERT failed"); } } while (0)
#define OMNIDREAMS_SINGLEVIEW_CUDA_CHECK(...) do { \
    __VA_ARGS__; \
    cudaError_t _e_sync = cudaDeviceSynchronize(); \
    if (_e_sync != cudaSuccess) { \
        fprintf(stderr, "[OMNIDREAMS_SINGLEVIEW_CUDA_CHECK] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e_sync)); \
        throw std::runtime_error("CUDA kernel error"); \
    } \
} while (0)
#else
#define OMNIDREAMS_SINGLEVIEW_ASSERT(cond) ((void)0)
#define OMNIDREAMS_SINGLEVIEW_CUDA_CHECK(...) do { __VA_ARGS__; } while (0)
#endif

// Activations
enum class Activation : uint32_t {
    None = 0,
    ReLU = 1,
    ClampTanh3 = 2
};

// Helpers
__device__ __forceinline__ float clamp_tanh3(float x) {
    return tanhf(x * (1.0f / 3.0f)) * 3.0f;
}

// Shape helpers
inline int conv3x3_out_dim(int in, int stride, int pad) {
    OMNIDREAMS_SINGLEVIEW_ASSERT(in > 0 && stride > 0);
    return (int)((in + 2 * pad - 3) / stride) + 1;
}

// Transpose kernels declarations (defined inline here for reuse)
__global__ void transpose_hwc_to_nchw_half_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int N, int H, int W, int C)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;
    if (x >= W || y >= H || n >= N) return;
    int hwc_base = ((n * H + y) * W + x) * C;
    for (int c = 0; c < C; ++c) {
        output[((n * C + c) * H + y) * W + x] = input[hwc_base + c];
    }
}

__global__ void transpose_nchw_to_hwc_half_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int N, int H, int W, int C)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;
    if (x >= W || y >= H || n >= N) return;
    int hwc_base = ((n * H + y) * W + x) * C;
    for (int c = 0; c < C; ++c) {
        output[hwc_base + c] = input[((n * C + c) * H + y) * W + x];
    }
}

__global__ void transpose_nthwc_to_ntchw_half_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int N, int T, int H, int W, int C)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * T * H * W * C;
    if (idx >= total) return;
    int c = idx % C; idx /= C;
    int w = idx % W; idx /= W;
    int h = idx % H; idx /= H;
    int t = idx % T; idx /= T;
    int n = (int)idx;
    long long in_idx  = (((((long long)n * T + t) * H + h) * W + w) * C + c);
    long long out_idx = (((((long long)(n * T + t) * C + c) * H + h) * W + w));
    output[out_idx] = input[in_idx];
}

__global__ void transpose_ntchw_to_nthwc_half_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int N, int T, int H, int W, int C)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * T * H * W * C;
    if (idx >= total) return;
    int c = idx % C; idx /= C;
    int w = idx % W; idx /= W;
    int h = idx % H; idx /= H;
    int t = idx % T; idx /= T;
    int n = (int)idx;
    long long out_idx = (((((long long)n * T + t) * H + h) * W + w) * C + c);
    long long in_idx  = (((((long long)(n * T + t) * C + c) * H + h) * W + w));
    output[out_idx] = input[in_idx];
}

__global__ void reorder_mn_to_nchw_half_kernel(
    const half* __restrict__ src_mn,
    half* __restrict__ dst_nchw,
    long long M, int N, int C, int H, int W)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = M * C;
    if (idx >= total) return;
    int oc = (int)(idx % C);
    long long row = idx / C;
    int n = (int)(row / (H * W));
    int hw = (int)(row % (H * W));
    long long out_index = (long long)n * (C * H * W) + (long long)oc * (H * W) + hw;
    dst_nchw[out_index] = src_mn[row * C + oc];
}

// Build past by shifting time in flattened NT batches: for each (n, t), past(n,t)=x(n,t-1), and zero for t=0
__global__ void build_past_shifted_nchw_half_kernel(
    const half* __restrict__ x, half* __restrict__ past,
    int N, int T, int C, int H, int W)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * T * C * H * W;
    if (idx >= total) return;
    int w = idx % W; idx /= W;
    int h = idx % H; idx /= H;
    int c = idx % C; idx /= C;
    int t = idx % T; idx /= T;
    int n = (int)idx;
    long long base = (((((long long)n * T + t) * C + c) * H + h) * W + w);
    if (t == 0) {
        past[base] = __float2half(0.0f);
    } else {
        int t_prev = t - 1;
        long long base_prev = (((((long long)n * T + t_prev) * C + c) * H + h) * W + w);
        past[base] = x[base_prev];
    }
}

inline void build_past_shifted_nchw(const half* x, half* past, int N, int T, int C, int H, int W, cudaStream_t stream)
{
    OMNIDREAMS_SINGLEVIEW_ASSERT(x != nullptr && past != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N > 0 && T > 0 && C > 0 && H > 0 && W > 0);
    long long total = (long long)N * T * C * H * W;
    int blk = 256;
    int grd = (int)((total + blk - 1) / blk);
    build_past_shifted_nchw_half_kernel<<<grd, blk, 0, stream>>>(x, past, N, T, C, H, W);
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
}

// Build past by shifting time in flattened NT batches for NHWC layout: past(n,t,y,x,c)=x(n,t-1,y,x,c), zero for t=0
__global__ void build_past_shifted_nhwc_half_kernel(
    const half* __restrict__ x, half* __restrict__ past,
    int N, int T, int H, int W, int C)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * T * H * W * C;
    if (idx >= total) return;
    int c = (int)(idx % C); idx /= C;
    int w = (int)(idx % W); idx /= W;
    int h = (int)(idx % H); idx /= H;
    int t = (int)(idx % T); idx /= T;
    int n = (int)idx;
    long long base = (((((long long)n * T + t) * H + h) * W + w) * C + c);
    if (t == 0) {
        past[base] = __float2half(0.0f);
    } else {
        int t_prev = t - 1;
        long long base_prev = (((((long long)n * T + t_prev) * H + h) * W + w) * C + c);
        past[base] = x[base_prev];
    }
}

inline void build_past_shifted_nhwc(const half* x, half* past, int N, int T, int H, int W, int C, cudaStream_t stream)
{
    OMNIDREAMS_SINGLEVIEW_ASSERT(x != nullptr && past != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N > 0 && T > 0 && H > 0 && W > 0 && C > 0);
    long long total = (long long)N * T * H * W * C;
    int blk = 256;
    int grd = (int)((total + blk - 1) / blk);
    build_past_shifted_nhwc_half_kernel<<<grd, blk, 0, stream>>>(x, past, N, T, H, W, C);
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
}

// Trim first K frames from NTHWC buffer: in [N*T, H, W, C] with known N and T.
__global__ void nthwc_trim_front_half_kernel(
    const half* __restrict__ in, half* __restrict__ out,
    int N, int T, int H, int W, int C, int K)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * (T - K) * H * W * C;
    if (idx >= total) return;
    int c = idx % C; idx /= C;
    int w = idx % W; idx /= W;
    int h = idx % H; idx /= H;
    int t = idx % (T - K); idx /= (T - K);
    int n = (int)idx;
    long long in_base  = (((((long long)n * T + (t + K)) * H + h) * W + w) * C + c);
    long long out_base = (((((long long)n * (T - K) + t) * H + h) * W + w) * C + c);
    out[out_base] = in[in_base];
}

inline void nthwc_trim_front_half(
    const half* in, half* out,
    int N, int T, int H, int W, int C, int K,
    cudaStream_t stream)
{
    OMNIDREAMS_SINGLEVIEW_ASSERT(in != nullptr && out != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N > 0 && T > 0 && H > 0 && W > 0 && C > 0);
    OMNIDREAMS_SINGLEVIEW_ASSERT(K >= 0 && K < T);
    long long total = (long long)N * (T - K) * H * W * C;
    int blk = 256;
    int grd = (int)((total + blk - 1) / blk);
    nthwc_trim_front_half_kernel<<<grd, blk, 0, stream>>>(in, out, N, T, H, W, C, K);
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
}

// Wrappers
inline void hwc_to_nchw_half(const half* in_hwc, half* out_nchw, int N, int H, int W, int C, cudaStream_t stream)
{
    OMNIDREAMS_SINGLEVIEW_ASSERT(in_hwc != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(out_nchw != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N > 0 && H > 0 && W > 0 && C > 0);
    dim3 block(16, 16, 1);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, N);
    transpose_hwc_to_nchw_half_kernel<<<grid, block, 0, stream>>>(in_hwc, out_nchw, N, H, W, C);
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
}

inline void nchw_to_hwc_half(const half* in_nchw, half* out_hwc, int N, int H, int W, int C, cudaStream_t stream)
{
    OMNIDREAMS_SINGLEVIEW_ASSERT(in_nchw != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(out_hwc != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N > 0 && H > 0 && W > 0 && C > 0);
    dim3 block(16, 16, 1);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, N);
    transpose_nchw_to_hwc_half_kernel<<<grid, block, 0, stream>>>(in_nchw, out_hwc, N, H, W, C);
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
}

inline void nthwc_to_ntchw_half(const half* in_nthwc, half* out_ntchw, int N, int T, int H, int W, int C, cudaStream_t stream)
{
    OMNIDREAMS_SINGLEVIEW_ASSERT(in_nthwc != nullptr && out_ntchw != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N > 0 && T > 0 && H > 0 && W > 0 && C > 0);
    long long total = (long long)N * T * H * W * C;
    int blk = 256;
    int grd = (int)((total + blk - 1) / blk);
    transpose_nthwc_to_ntchw_half_kernel<<<grd, blk, 0, stream>>>(in_nthwc, out_ntchw, N, T, H, W, C);
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
}

inline void ntchw_to_nthwc_half(const half* in_ntchw, half* out_nthwc, int N, int T, int H, int W, int C, cudaStream_t stream)
{
    OMNIDREAMS_SINGLEVIEW_ASSERT(in_ntchw != nullptr && out_nthwc != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N > 0 && T > 0 && H > 0 && W > 0 && C > 0);
    long long total = (long long)N * T * H * W * C;
    int blk = 256;
    int grd = (int)((total + blk - 1) / blk);
    transpose_ntchw_to_nthwc_half_kernel<<<grd, blk, 0, stream>>>(in_ntchw, out_nthwc, N, T, H, W, C);
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
}

inline void ntchw_trim_front_half(const half* in_ntchw, half* out_ntchw, int N, int T, int C, int H, int W, int K, cudaStream_t stream)
{
    OMNIDREAMS_SINGLEVIEW_ASSERT(in_ntchw != nullptr && out_ntchw != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N > 0 && T > K && C > 0 && H > 0 && W > 0);
    long long total = (long long)N * (T - K) * C * H * W;
    int blk = 256;
    int grd = (int)((total + blk - 1) / blk);
    ::ntchw_trim_front_half_kernel<<<grd, blk, 0, stream>>>(in_ntchw, out_ntchw, N, T, C, H, W, K);
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
}

} // namespace omnidreams_singleview

// Define the trim kernel after the namespace to avoid accidental internal linkage issues
__global__ void ntchw_trim_front_half_kernel(
    const half* __restrict__ in,
    half* __restrict__ out,
    int N_, int T_, int C_, int H_, int W_, int K_)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total2 = (long long)N_ * (T_ - K_) * C_ * H_ * W_;
    if (idx >= total2) return;
    int w = idx % W_; idx /= W_;
    int h = idx % H_; idx /= H_;
    int c = idx % C_; idx /= C_;
    int t = idx % (T_ - K_); idx /= (T_ - K_);
    int n = (int)idx;
    long long in_base  = (((((long long)n * T_ + (t + K_)) * C_ + c) * H_ + h) * W_ + w);
    long long out_base = (((((long long)n * (T_ - K_) + t) * C_ + c) * H_ + h) * W_ + w);
    out[out_base] = in[in_base];
}
