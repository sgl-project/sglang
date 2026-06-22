// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <algorithm>

#include "kernels.cuh"

namespace omnidreams_singleview {

// Forward declaration so it can be used earlier in this TU
inline void cat_channels_nhwc(const half* a, const half* b, half* out,
                              int N, int H, int W, int Ca, int Cb, cudaStream_t stream);

// Elementwise ClampTanh3: y = tanh(x/3) * 3
__global__ void clamp_tanh3_inplace_half_kernel(half* __restrict__ x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]);
    float r = tanhf(v * (1.0f/3.0f)) * 3.0f;
    x[i] = __float2half(r);
}

inline void clamp_tanh3_inplace(half* x, int N, int C, int H, int W, cudaStream_t stream)
{
    OMNIDREAMS_SINGLEVIEW_ASSERT(x != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N > 0 && C > 0 && H > 0 && W > 0);
    int n = N * C * H * W;
    int blk = 256;
    int grd = (n + blk - 1) / blk;
    clamp_tanh3_inplace_half_kernel<<<grd, blk, 0, stream>>>(x, n);
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
}

// Build past by shifting time in flattened NT batches: for each (n, t), past(n,t)=x(n,t-1), and zero for t=0
#if !OMNIDREAMS_SINGLEVIEW_BUILD_PAST_IN_COMMON
#error "build_past_shifted must be provided by common/cuda_utils.cuh as NHWC/NTHWC variants"
#endif

// Contracts & invariants for ops (Phase 1):
// - All tensors are contiguous NHWC unless explicitly noted.
// - Elementwise ops require matching element counts and non-null pointers.
// - cat_channels_nhwc: inputs [N,H,W,Ca], [N,H,W,Cb], output [N,H,W,Ca+Cb].
// - upsample2x_nearest_nhwc: output has H*2,W*2; caller must allocate.
// - memblock_forward_nhwc: follows Python MemBlock ordering with three 3x3 convs
//   and an optional 1x1 skip; uses FP32 accum, bias+act in FP32, then cast.
//   Temporal semantics (past) are defined by the executor; the kernel assumes
//   shapes match and does not manage time.

// Elementwise add: y = a + b
__global__ void ew_add_half_kernel(const half* __restrict__ a, const half* __restrict__ b, half* __restrict__ y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float va = __half2float(a[i]);
    float vb = __half2float(b[i]);
    y[i] = __float2half(va + vb);
}

// Elementwise ReLU in-place
__global__ void relu_inplace_half_kernel(half* __restrict__ x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]);
    x[i] = __float2half(v > 0.0f ? v : 0.0f);
}

// NCHW channel concat kernel removed in NHWC-only backend

// Nearest 2x upsample (spatial) for NCHW removed in NHWC-only backend

// (Removed) fused NCHW tiled conv3x3-cat+ReLU kernel and wrapper

// MemBlock forward (NCHW):
// y = ReLU( Conv3x3(ReLU(Conv3x3(ReLU(Conv3x3(cat(x, past))))) ) + Skip(x) )
// - x: [N, Cin, H, W], past: [N, Cin, H, W], Cin may differ from Cout
// - weights: w1[Co1,Ccat,3,3], w2[Co2,Co1,3,3], w3[Cout,Co2,3,3]
// - skip: 1x1 if Cin!=Cout else identity
inline void memblock_forward_nchw(
    const half* x, const half* past,
    const half* w1, const half* b1,
    const half* w2, const half* b2,
    const half* w3, const half* b3,
    const half* wskip, // null if Cin==Cout
    half* tmp_cat, half* tmp1, half* tmp2, half* y,
    int N, int Cin, int H, int W, int Cout,
    cudaStream_t stream)
{
    OMNIDREAMS_SINGLEVIEW_ASSERT(x != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(past != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(w1 != nullptr && w2 != nullptr && w3 != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(y != nullptr && tmp_cat != nullptr && tmp1 != nullptr && tmp2 != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N > 0 && H > 0 && W > 0 && Cin > 0 && Cout > 0);
    // Always use NHWC convs with local reorders
    {
        half* x_nhwc = nullptr;
        half* p_nhwc = nullptr;
        half* cat_nhwc = nullptr;
        half* t1_nhwc = nullptr;
        half* t2_nhwc = nullptr;
        half* y_nhwc = nullptr;
        const size_t elems_in   = (size_t)N * H * W * Cin;
        const size_t elems_cat  = (size_t)N * H * W * (size_t)(2 * Cin);
        const size_t elems_cout = (size_t)N * H * W * Cout;

        auto check_malloc = [](cudaError_t err, const char* name, size_t bytes) {
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("cudaMalloc failed for memblock ") + name + " (" +
                    std::to_string(bytes / 1048576.0) + " MB): " + cudaGetErrorString(err));
            }
        };

        check_malloc(cudaMalloc(&x_nhwc,   elems_in   * sizeof(half)), "x_nhwc",   elems_in   * sizeof(half));
        check_malloc(cudaMalloc(&p_nhwc,   elems_in   * sizeof(half)), "p_nhwc",   elems_in   * sizeof(half));
        check_malloc(cudaMalloc(&cat_nhwc, elems_cat  * sizeof(half)), "cat_nhwc", elems_cat  * sizeof(half));
        check_malloc(cudaMalloc(&t1_nhwc,  elems_cout * sizeof(half)), "t1_nhwc",  elems_cout * sizeof(half));
        check_malloc(cudaMalloc(&t2_nhwc,  elems_cout * sizeof(half)), "t2_nhwc",  elems_cout * sizeof(half));
        check_malloc(cudaMalloc(&y_nhwc,   elems_cout * sizeof(half)), "y_nhwc",   elems_cout * sizeof(half));
        OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
        nchw_to_hwc_half(x,    x_nhwc,   N, H, W, Cin, stream);
        nchw_to_hwc_half(past, p_nhwc,   N, H, W, Cin, stream);
        OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
        cat_channels_nhwc(x_nhwc, p_nhwc, cat_nhwc, N, H, W, Cin, Cin, stream);
        OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
        conv2d_3x3_nhwc_half(cat_nhwc, w1, b1, t1_nhwc, N, H, W, /*Cin=*/Cin*2, /*Cout=*/Cout, /*groups=*/1, Activation::ReLU, /*stride=*/1, /*pad=*/1, /*H_out=*/H, /*W_out=*/W, stream);
        conv2d_3x3_nhwc_half(t1_nhwc,  w2, b2, t2_nhwc, N, H, W, /*Cin=*/Cout,  /*Cout=*/Cout, /*groups=*/1, Activation::ReLU, /*stride=*/1, /*pad=*/1, /*H_out=*/H, /*W_out=*/W, stream);
        conv2d_3x3_nhwc_half(t2_nhwc,  w3, b3, y_nhwc,  N, H, W, /*Cin=*/Cout,  /*Cout=*/Cout, /*groups=*/1, Activation::None, /*stride=*/1, /*pad=*/1, /*H_out=*/H, /*W_out=*/W, stream);
        hwc_to_nchw_half(y_nhwc, y, N, H, W, Cout, stream);
        OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
        cudaFree(x_nhwc); cudaFree(p_nhwc); cudaFree(cat_nhwc);
        cudaFree(t1_nhwc); cudaFree(t2_nhwc); cudaFree(y_nhwc);
        OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
    }

    // skip path + final ReLU
    if (wskip) {
        // 1x1 project Cin->Cout and add (via NHWC path)
        {
            half* x_nhwc = nullptr; half* y_nhwc = nullptr;
            const size_t elems_in = (size_t)N * H * W * Cin;
            const size_t elems_out= (size_t)N * H * W * Cout;
            cudaError_t err1 = cudaMalloc(&x_nhwc, elems_in * sizeof(half));
            if (err1 != cudaSuccess) {
                throw std::runtime_error(std::string("cudaMalloc failed for skip x_nhwc (") +
                    std::to_string(elems_in * sizeof(half) / 1048576.0) + " MB): " + cudaGetErrorString(err1));
            }
            cudaError_t err2 = cudaMalloc(&y_nhwc, elems_out * sizeof(half));
            if (err2 != cudaSuccess) {
                cudaFree(x_nhwc);
                throw std::runtime_error(std::string("cudaMalloc failed for skip y_nhwc (") +
                    std::to_string(elems_out * sizeof(half) / 1048576.0) + " MB): " + cudaGetErrorString(err2));
            }
            OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
            nchw_to_hwc_half(x, x_nhwc, N, H, W, Cin, stream);
            conv2d_1x1_nhwc_half(x_nhwc, wskip, nullptr, y_nhwc, N, H, W, Cin, Cout, /*groups=*/1, Activation::None, stream);
            hwc_to_nchw_half(y_nhwc, tmp1, N, H, W, Cout, stream);
            cudaFree(x_nhwc); cudaFree(y_nhwc);
            OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
        }
        int n_elems = N * Cout * H * W;
        int blk = 256;
        int grd = (n_elems + blk - 1) / blk;
        ew_add_half_kernel<<<grd, blk, 0, stream>>>(y, tmp1, y, n_elems);
    } else {
        // in-place add with identity
        // nothing to add since channels match and skip is identity; y already holds conv output, just add x
        int n_elems = N * Cout * H * W;
        int blk = 256;
        int grd = (n_elems + blk - 1) / blk;
        ew_add_half_kernel<<<grd, blk, 0, stream>>>(y, x, y, n_elems);
    }
    // final ReLU
    {
        int n_elems = N * Cout * H * W;
        int blk = 256;
        int grd = (n_elems + blk - 1) / blk;
        relu_inplace_half_kernel<<<grd, blk, 0, stream>>>(y, n_elems);
    }
}

// --- Native NHWC implementations ---

// Simple channel concatenation kernel for memblock use case with debug support
__global__ void cat_channels_nhwc_simple_kernel(
    const half* __restrict__ a, const half* __restrict__ b, half* __restrict__ out,
    long long total_spatial, int channels_per_tensor)
{
    long long spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (spatial_idx >= total_spatial) return;

    // Calculate base offsets for this spatial position
    long long in_base_a = spatial_idx * channels_per_tensor;
    long long in_base_b = spatial_idx * channels_per_tensor;
    long long out_base = spatial_idx * (channels_per_tensor * 2);

    #if OMNIDREAMS_SINGLEVIEW_DEBUG
    // Debug output for first few threads to track memory access patterns
    if (spatial_idx < 4 && threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[KERNEL DEBUG] spatial_idx=%lld, in_base_a=%lld, in_base_b=%lld, out_base=%lld\n",
               spatial_idx, in_base_a, in_base_b, out_base);
        printf("[KERNEL DEBUG] channels_per_tensor=%d, total_spatial=%lld\n",
               channels_per_tensor, total_spatial);
    }
    #endif


    #if OMNIDREAMS_SINGLEVIEW_DEBUG
    if (spatial_idx < 2) {
        // Bounds checking with debug output
        long long max_input_idx_a = in_base_a + channels_per_tensor - 1;
        long long max_input_idx_b = in_base_b + channels_per_tensor - 1;
        long long max_output_idx = out_base + (channels_per_tensor * 2) - 1;

        printf("[KERNEL DEBUG] Thread %lld: max_input_idx_a=%lld, max_input_idx_b=%lld, max_output_idx=%lld\n",
               spatial_idx, max_input_idx_a, max_input_idx_b, max_output_idx);
    }
    #endif

    // Copy all channels from tensor a
    for (int c = 0; c < channels_per_tensor; c++) {
        long long src_idx = in_base_a + c;
        long long dst_idx = out_base + c;

        #if OMNIDREAMS_SINGLEVIEW_DEBUG
        if (spatial_idx == 0 && c < 2) {
            printf("[KERNEL DEBUG] Copy A: a[%lld] -> out[%lld]\n", src_idx, dst_idx);
        }
        #endif

        out[dst_idx] = a[src_idx];
    }

    // Copy all channels from tensor b
    for (int c = 0; c < channels_per_tensor; c++) {
        long long src_idx = in_base_b + c;
        long long dst_idx = out_base + channels_per_tensor + c;

        #if OMNIDREAMS_SINGLEVIEW_DEBUG
        if (spatial_idx == 0 && c < 2) {
            printf("[KERNEL DEBUG] Copy B: b[%lld] -> out[%lld]\n", src_idx, dst_idx);
        }
        #endif

        out[dst_idx] = b[src_idx];
    }
}

// Channel concat: y[N, H, W, Cx+Cy] = concat(a, b) along C (NHWC)
__global__ void cat_channels_nhwc_half_kernel(
    const half* __restrict__ a, const half* __restrict__ b, half* __restrict__ out,
    int N, int H, int W, int Ca, int Cb)
{
    // Extremely conservative approach - process one element at a time
    long long total_elements_out = (long long)N * H * W * (Ca + Cb);
    long long global_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= total_elements_out) return;

    // Decompose to spatial position and channel
    long long spatial_pos = global_idx / (Ca + Cb);
    int channel = global_idx % (Ca + Cb);

    // Safety bounds check
    if (spatial_pos >= (long long)N * H * W) return;
    if (channel >= (Ca + Cb)) return;

    if (channel < Ca) {
        // Copy from tensor a
        long long src_idx = spatial_pos * Ca + channel;
        if (src_idx < (long long)N * H * W * Ca) {
            out[global_idx] = a[src_idx];
        }
    } else {
        // Copy from tensor b
        int b_channel = channel - Ca;
        long long src_idx = spatial_pos * Cb + b_channel;
        if (src_idx < (long long)N * H * W * Cb && b_channel >= 0 && b_channel < Cb) {
            out[global_idx] = b[src_idx];
        }
    }
}

inline void cat_channels_nhwc(const half* a, const half* b, half* out,
                              int N, int H, int W, int Ca, int Cb, cudaStream_t stream)
{
    OMNIDREAMS_SINGLEVIEW_ASSERT(a != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(b != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(out != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N > 0 && H > 0 && W > 0 && Ca > 0 && Cb > 0);

    // Launch one thread per output element
    long long total_elements = (long long)N * H * W * (Ca + Cb);
    int threads = 256;

    // For very large tensors, split into multiple kernel launches
    while (total_elements > 0) {
        long long chunk_size = min(total_elements, (long long)65535 * threads);
        int blocks = (int)((chunk_size + threads - 1) / threads);

        cat_channels_nhwc_half_kernel<<<blocks, threads, 0, stream>>>(a, b, out, N, H, W, Ca, Cb);
        OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();

        // For simplicity, process everything in one go for typical sizes
        // Multi-chunk support can be added if needed for very large tensors
        break;
    }
}

// Nearest 2x upsample (spatial) for NHWC - vectorized over channels
__global__ void upsample2x_nearest_nhwc_half_kernel(
    const half* __restrict__ in, half* __restrict__ out,
    int N, int H, int W, int C)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;
    int H2 = H * 2, W2 = W * 2;
    if (x >= W2 || y >= H2 || n >= N) return;
    int ix = x >> 1;
    int iy = y >> 1;
    long long base_in  = (((((long long)n * H) + iy) * W + ix) * (long long)C);
    long long base_out = (((((long long)n * H2) + y) * W2 + x) * (long long)C);

    // Vectorize across channels when possible
    int c2 = C / 2;
    const half2* src_h2 = reinterpret_cast<const half2*>(in + base_in);
    half2* dst_h2 = reinterpret_cast<half2*>(out + base_out);
    for (int i = 0; i < c2; ++i) {
        half2 v = src_h2[i];
        dst_h2[i] = v;
    }
    // Handle odd tail channel
    if ((C & 1) != 0) {
        out[base_out + (C - 1)] = in[base_in + (C - 1)];
    }
}

// Flat NHWC concat kernel removed - use cat_channels_nhwc() function instead

inline void upsample2x_nearest_nhwc(const half* in, half* out, int N, int H, int W, int C, cudaStream_t stream)
{
    OMNIDREAMS_SINGLEVIEW_ASSERT(in != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(out != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N > 0 && H > 0 && W > 0 && C > 0);
    // Favor wider x for better global load coalescing
    dim3 block(32, 8, 1);
    dim3 grid((W * 2 + block.x - 1) / block.x,
              (H * 2 + block.y - 1) / block.y,
              N);
    upsample2x_nearest_nhwc_half_kernel<<<grid, block, 0, stream>>>(in, out, N, H, W, C);
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
}

#if !OMNIDREAMS_SINGLEVIEW_BUILD_PAST_IN_COMMON
// Build past by shifting time in flattened NT batches for NHWC:
// past(n,t,y,x,c) = x(n,t-1,y,x,c), and zero for t=0
__global__ void build_past_shifted_nhwc_half_kernel(
    const half* __restrict__ x, half* __restrict__ past,
    int N, int T, int H, int W, int C)
{
    long long total = (long long)N * T * H * W * C;
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = (int)(idx % C); idx /= C;
    int w = (int)(idx % W); idx /= W;
    int h = (int)(idx % H); idx /= H;
    int t = (int)(idx % T); idx /= T;
    int n = (int)idx;
    long long base = ((((((long long)n * T + t) * H + h) * W + w) * C) + c);
    if (t == 0) {
        past[base] = __float2half(0.0f);
    } else {
        int t_prev = t - 1;
        long long base_prev = ((((((long long)n * T + t_prev) * H + h) * W + w) * C) + c);
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
#endif

// MemBlock forward (NHWC):
// y = ReLU( Conv3x3(ReLU(Conv3x3(ReLU(Conv3x3(cat(x, past))))) ) + Skip(x) )
// - x: [N, H, W, Cin], past: [N, H, W, Cin], Cin may differ from Cout
// - weights: w1[Co1,Ccat,3,3], w2[Co2,Co1,3,3], w3[Cout,Co2,3,3]
// - skip: 1x1 if Cin!=Cout else identity
inline void memblock_forward_nhwc(
    const half* x, const half* past,
    const half* w1, const half* b1,
    const half* w2, const half* b2,
    const half* w3, const half* b3,
    const half* wskip, // null if Cin==Cout
    half* tmp_cat, half* tmp1, half* tmp2, half* y,
    int N, int H, int W, int Cin, int Cout,
    cudaStream_t stream)
{
    OMNIDREAMS_SINGLEVIEW_ASSERT(x != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(past != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(w1 != nullptr && w2 != nullptr && w3 != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(y != nullptr && tmp_cat != nullptr && tmp1 != nullptr && tmp2 != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N > 0 && H > 0 && W > 0 && Cin > 0 && Cout > 0);

    #if OMNIDREAMS_SINGLEVIEW_DEBUG
    printf("[DEBUG] MemBlock NHWC Entry: N=%d, H=%d, W=%d, Cin=%d, Cout=%d\n", N, H, W, Cin, Cout);
    printf("[DEBUG] MemBlock NHWC Pointers: x=%p, past=%p, tmp_cat=%p, tmp1=%p, tmp2=%p, y=%p\n",
           (void*)x, (void*)past, (void*)tmp_cat, (void*)tmp1, (void*)tmp2, (void*)y);
    printf("[DEBUG] MemBlock NHWC Stream: %p\n", (void*)stream);

    // Calculate expected tensor sizes for validation
    long long spatial_size = (long long)N * H * W;
    long long x_size = spatial_size * Cin;
    long long past_size = spatial_size * Cin;
    long long tmp_cat_size = spatial_size * (Cin + Cin);
    long long tmp1_size = spatial_size * Cout;
    long long tmp2_size = spatial_size * Cout;
    long long y_size = spatial_size * Cout;

    printf("[DEBUG] Expected tensor element counts: x=%lld, past=%lld, tmp_cat=%lld, tmp1=%lld, tmp2=%lld, y=%lld\n",
           x_size, past_size, tmp_cat_size, tmp1_size, tmp2_size, y_size);

    // Check for dangerous pointer relationships
    if (tmp_cat == x || tmp_cat == past) {
        printf("[ERROR] MemBlock NHWC: tmp_cat shares pointer with input tensor!\n");
    }
    if (y == x || y == past || y == tmp_cat) {
        printf("[ERROR] MemBlock NHWC: Output y shares pointer with input/intermediate tensor!\n");
    }
    #endif

    // Concatenate channels along C in NHWC - use coalesced element mapping
    // Switch to the coalesced concat implementation to avoid uncoalesced per-thread inner loops
    {
        int threads = 256;
        long long total_spatial = (long long)N * H * W;
        int blocks = (int)min((total_spatial + threads - 1) / threads, 65535LL);
        (void)threads; (void)total_spatial; (void)blocks;

        #if OMNIDREAMS_SINGLEVIEW_DEBUG
        printf("[DEBUG] HWC Concat: N=%d, H=%d, W=%d, Cin=%d, Cout=%d\n", N, H, W, Cin, Cout);
        printf("[DEBUG] HWC Concat: total_spatial=%lld, blocks=%d, threads=%d\n", total_spatial, blocks, threads);
        printf("[DEBUG] HWC Concat: x=%p, past=%p, tmp_cat=%p\n", (void*)x, (void*)past, (void*)tmp_cat);

        // Validate pointer alignment and basic sanity
        if (x == nullptr || past == nullptr || tmp_cat == nullptr) {
            printf("[ERROR] HWC Concat: Null pointer detected! x=%p, past=%p, tmp_cat=%p\n",
                   (void*)x, (void*)past, (void*)tmp_cat);
        }

        // Check for pointer overlap that could cause corruption
        char* x_end = (char*)x + total_spatial * Cin * sizeof(half);
        char* past_end = (char*)past + total_spatial * Cin * sizeof(half);
        char* tmp_cat_end = (char*)tmp_cat + total_spatial * (Cin + Cin) * sizeof(half);

        if ((char*)tmp_cat < x_end && (char*)x < tmp_cat_end) {
            printf("[WARNING] HWC Concat: Potential overlap between x and tmp_cat!\n");
        }
        if ((char*)tmp_cat < past_end && (char*)past < tmp_cat_end) {
            printf("[WARNING] HWC Concat: Potential overlap between past and tmp_cat!\n");
        }

        printf("[DEBUG] HWC Concat: Expected tensor sizes - x:%lld bytes, past:%lld bytes, tmp_cat:%lld bytes\n",
               total_spatial * Cin * sizeof(half),
               total_spatial * Cin * sizeof(half),
               total_spatial * (Cin + Cin) * sizeof(half));
        #endif

        // Use coalesced concat implementation
        cat_channels_nhwc(x, past, tmp_cat, N, H, W, Cin, Cin, stream);
        OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();

        #if OMNIDREAMS_SINGLEVIEW_DEBUG
        printf("[DEBUG] HWC Concat: Coalesced concat completed successfully\n");
        #endif
    }

    #if OMNIDREAMS_SINGLEVIEW_DEBUG
    printf("[DEBUG] MemBlock NHWC: Starting 3x3 convolutions sequence\n");
    #endif

    // Three 3x3 convs with ReLU, ReLU, None
    conv2d_3x3_nhwc_half(tmp_cat, w1, b1, tmp1,
                         N, /*H_in=*/H, /*W_in=*/W, /*Cin=*/Cin * 2, /*Cout=*/Cout, /*groups=*/1,
                         Activation::ReLU,
                         /*stride=*/1, /*pad=*/1,
                         /*H_out=*/H, /*W_out=*/W,
                         stream);
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();

    #if OMNIDREAMS_SINGLEVIEW_DEBUG
    printf("[DEBUG] MemBlock NHWC: First 3x3 conv completed\n");
    #endif

    conv2d_3x3_nhwc_half(tmp1, w2, b2, tmp2,
                         N, /*H_in=*/H, /*W_in=*/W, /*Cin=*/Cout, /*Cout=*/Cout, /*groups=*/1,
                         Activation::ReLU,
                         /*stride=*/1, /*pad=*/1,
                         /*H_out=*/H, /*W_out=*/W,
                         stream);
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();

    #if OMNIDREAMS_SINGLEVIEW_DEBUG
    printf("[DEBUG] MemBlock NHWC: Second 3x3 conv completed\n");
    #endif

    conv2d_3x3_nhwc_half(tmp2, w3, b3, y,
                         N, /*H_in=*/H, /*W_in=*/W, /*Cin=*/Cout, /*Cout=*/Cout, /*groups=*/1,
                         Activation::None,
                         /*stride=*/1, /*pad=*/1,
                         /*H_out=*/H, /*W_out=*/W,
                         stream);
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();

    #if OMNIDREAMS_SINGLEVIEW_DEBUG
    printf("[DEBUG] MemBlock NHWC: Third 3x3 conv completed\n");
    #endif

    // skip path + final ReLU
    long long n_elems_ll = (long long)N * (long long)H * (long long)W * (long long)Cout;
    OMNIDREAMS_SINGLEVIEW_ASSERT(n_elems_ll > 0);
    OMNIDREAMS_SINGLEVIEW_ASSERT(n_elems_ll <= (long long)std::numeric_limits<int>::max());
    int n_elems = (int)n_elems_ll;
    if (wskip != nullptr) {
        // Compute skip projection into tmp1, then accumulate in-place into y
        conv2d_1x1_nhwc_half(x, wskip, nullptr, tmp1,
                             N, H, W, Cin, Cout, /*groups=*/1, Activation::None, stream);
        OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
        int blk = 256;
        int grd = (n_elems + blk - 1) / blk;
        ew_add_half_kernel<<<grd, blk, 0, stream>>>(y, tmp1, y, n_elems);
        OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
    } else if (Cin == Cout) {
        // Identity skip: accumulate x into y in-place
        int blk = 256;
        int grd = (n_elems + blk - 1) / blk;
        ew_add_half_kernel<<<grd, blk, 0, stream>>>(y, x, y, n_elems);
        OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
    } else {
        // No skip provided and channels differ: do nothing
    }
    {
        int blk = 256;
        int grd = (n_elems + blk - 1) / blk;
        relu_inplace_half_kernel<<<grd, blk, 0, stream>>>(y, n_elems);
        OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
    }
}

// Unpack TGrow for NHWC: input [NT, H, W, stride*Cin] -> output [NT*stride, H, W, Cin]
__global__ void tgrow_unpack_nhwc_half_kernel(
    const half* __restrict__ in, half* __restrict__ out,
    int NT, int H, int W, int Cin, int stride)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)NT * stride * H * W * Cin;
    if (idx >= total) return;
    int c = (int)(idx % Cin); idx /= Cin;
    int w = (int)(idx % W); idx /= W;
    int h = (int)(idx % H); idx /= H;
    int nt_new = (int)(idx % (NT * stride));
    int s = nt_new % stride;
    int nt_old = nt_new / stride;
    long long in_base  = (((((long long)nt_old * H) + h) * W + w) * (long long)(stride * Cin)) + (long long)s * Cin + c;
    long long out_base = (((((long long)nt_new * H) + h) * W + w) * (long long)Cin) + c;
    out[out_base] = in[in_base];
}

inline void tgrow_unpack_nhwc(const half* in, half* out,
                              int NT, int H, int W, int Cin, int stride, cudaStream_t stream)
{
    OMNIDREAMS_SINGLEVIEW_ASSERT(in != nullptr && out != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(NT > 0 && H > 0 && W > 0 && Cin > 0 && stride > 0);
    long long total = (long long)NT * stride * H * W * Cin;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    tgrow_unpack_nhwc_half_kernel<<<blocks, threads, 0, stream>>>(in, out, NT, H, W, Cin, stride);
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
}

} // namespace omnidreams_singleview
