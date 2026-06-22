// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <assert.h>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>
#include <mutex>
#include <sstream>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// Backend selector and common utilities
#include "../common/attn_backend.h"
#include "../common/cuda_utils.cuh"

#include <cuda_fp16.h>

// Provide missing __half comparison operators for CUTLASS compatibility
#if defined(__CUDA_ARCH__) && !defined(__CUDA_NO_HALF_OPERATORS__)
// Already have operators, do nothing
#elif defined(__CUDACC__)
// Define minimal operators needed by CUTLASS ArrayMaximum
__device__ __forceinline__ bool operator<(const __half& lhs, const __half& rhs) {
    #if __CUDA_ARCH__ >= 530
    return __hlt(lhs, rhs);
    #else
    return __half2float(lhs) < __half2float(rhs);
    #endif
}
#endif

#include <cutlass/epilogue/thread/linear_combination_bias_relu.h>
#include <cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>


namespace omnidreams_singleview {

// Global workspace pool for conv3x3 to eliminate dynamic allocations
namespace {
    struct Conv3x3WorkspaceArena {
        struct Entry {
            at::Tensor storage;
            size_t capacity;
        };

        std::vector<Entry> entries;
        mutable std::mutex mtx;

        static size_t round_capacity(size_t bytes) {
            constexpr size_t kAlign = 128;
            if (bytes == 0) return 0;
            size_t rounded = (bytes + kAlign - 1) / kAlign * kAlign;
            return rounded;
        }

        Entry* find_or_create(size_t required) {
            size_t rounded = round_capacity(required);
            std::lock_guard<std::mutex> lock(mtx);
            Entry* best = nullptr;
            for (auto& e : entries) {
                if (e.capacity >= rounded) {
                    if (!best || e.capacity < best->capacity) best = &e;
                }
            }
            if (!best) {
                Entry fresh;
                if (rounded > 0) {
                    auto options = at::TensorOptions().dtype(at::kByte).device(at::kCUDA);
                    fresh.storage = at::empty({static_cast<long long>(rounded)}, options);
                    fresh.capacity = rounded;
                    entries.push_back(fresh);
                    best = &entries.back();
                } else {
                    entries.push_back({});
                    best = &entries.back();
                }
            }
            return best;
        }

        void* get(size_t required_size, cudaStream_t stream, bool zero_fill) {
            Entry* e = find_or_create(required_size);
            if (!e || required_size == 0) return e ? e->storage.data_ptr() : nullptr;
            void* ptr = e->storage.data_ptr();
            if (zero_fill) {
                OMNIDREAMS_SINGLEVIEW_CUDA_CHECK(cudaMemsetAsync(ptr, 0, required_size, stream));
            }
            return ptr;
        }

        void reset() {
            std::lock_guard<std::mutex> lock(mtx);
            entries.clear();
        }

        void pre_reserve(size_t size) {
            if (size == 0) return;
            size_t rounded = round_capacity(size);
            std::lock_guard<std::mutex> lock(mtx);
            Entry* best = nullptr;
            for (auto& e : entries) {
                if (e.capacity >= rounded) {
                    best = &e;
                    break;
                }
            }
            if (!best) {
                Entry fresh;
                auto options = at::TensorOptions().dtype(at::kByte).device(at::kCUDA);
                fresh.storage = at::empty({static_cast<long long>(rounded)}, options);
                fresh.capacity = rounded;
                entries.push_back(std::move(fresh));
            }
        }

        size_t capacity() const {
            std::lock_guard<std::mutex> lock(mtx);
            size_t max_cap = 0;
            for (const auto& e : entries) {
                if (e.capacity > max_cap) max_cap = e.capacity;
            }
            return max_cap;
        }
    };

    static Conv3x3WorkspaceArena g_conv3x3_workspace_arena;
}

// Reset all CUTLASS workspaces for reproducibility
inline void reset_cutlass_workspaces() {
    // Reset the global workspace pool
    g_conv3x3_workspace_arena.reset();
}

// Pre-reserve conv3x3 workspace for maximum expected size
// Call this once before decode to eliminate all malloc overhead during execution
inline void pre_reserve_conv3x3_workspace(size_t workspace_bytes) {
    g_conv3x3_workspace_arena.pre_reserve(workspace_bytes);
}

inline size_t get_conv3x3_workspace_capacity() {
    return g_conv3x3_workspace_arena.capacity();
}

inline size_t estimate_conv3x3_workspace_bytes(
    int N, int H_in, int W_in, int Cin, int Cout, int stride, int pad) {

    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementComputeEpilogue = float;

    using LayoutInputA = cutlass::layout::TensorNHWC;
    using LayoutInputB = cutlass::layout::TensorNHWC;
    using LayoutOutput = cutlass::layout::TensorNHWC;

    using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    constexpr int NumStages = 3;
    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementInputB>::value;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementComputeEpilogue
    >;

    using Conv2dFpropKernelOpt = typename cutlass::conv::kernel::DefaultConv2dFprop<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        NumStages,
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kOptimized,
        cutlass::conv::StrideSupport::kStrided,
        AlignmentA,
        AlignmentB
    >::Kernel;
    using ImplicitGemmOpt = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernelOpt>;

    using Conv2dFpropKernelAna = typename cutlass::conv::kernel::DefaultConv2dFprop<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        NumStages,
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kAnalytic,
        cutlass::conv::StrideSupport::kStrided,
        AlignmentA,
        AlignmentB
    >::Kernel;
    using ImplicitGemmAna = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernelAna>;

    int H_out = conv3x3_out_dim(H_in, stride, pad);
    int W_out = conv3x3_out_dim(W_in, stride, pad);

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;
    int split_k_slices = 1;

    cutlass::conv::Conv2dProblemSize problem_size(
        {N, H_in, W_in, Cin},
        {Cout, 3, 3, Cin},
        {pad, pad, pad, pad},
        {stride, stride},
        {1, 1},
        {N, H_out, W_out, Cout},
        mode,
        split_k_slices
    );

    LayoutInputA layout_a(LayoutInputA::packed({N, H_in, W_in, Cin}));
    LayoutInputB layout_b(LayoutInputB::packed({Cout, 3, 3, Cin}));
    LayoutOutput layout_c(LayoutOutput::packed({N, H_out, W_out, Cout}));

    typename Conv2dFpropKernelOpt::TensorRefA tensor_a{nullptr, layout_a};
    typename Conv2dFpropKernelOpt::TensorRefB tensor_b{nullptr, layout_b};
    typename Conv2dFpropKernelOpt::TensorRefC tensor_c{nullptr, layout_c};
    typename Conv2dFpropKernelOpt::TensorRefC tensor_d{nullptr, layout_c};

    typename EpilogueOp::Params epilogue_params(
        ElementComputeEpilogue(1.0f),
        ElementComputeEpilogue(0.0f)
    );

    typename ImplicitGemmOpt::Arguments arguments_opt(
        problem_size,
        tensor_a,
        tensor_b,
        tensor_c,
        tensor_d,
        epilogue_params,
        cutlass::conv::SplitKMode::kSerial
    );
    typename ImplicitGemmAna::Arguments arguments_ana(
        problem_size,
        tensor_a,
        tensor_b,
        tensor_c,
        tensor_d,
        epilogue_params,
        cutlass::conv::SplitKMode::kSerial
    );

    ImplicitGemmOpt implicit_gemm_opt;
    ImplicitGemmAna implicit_gemm_ana;

    size_t ws_opt = implicit_gemm_opt.get_workspace_size(arguments_opt);
    size_t ws_ana = implicit_gemm_ana.get_workspace_size(arguments_ana);
    return std::max(ws_opt, ws_ana);
}

// Forward declarations for layout conversion functions
inline void nchw_to_hwc_half(const half* in_nchw, half* out_hwc, int N, int H, int W, int C, cudaStream_t stream);
inline void hwc_to_nchw_half(const half* in_hwc, half* out_nchw, int N, int H, int W, int C, cudaStream_t stream);

#if defined(OMNIDREAMS_SINGLEVIEW_USE_CUTLASS)
// Forward declaration for CUTLASS-backed NHWC 1x1
inline void conv2d_1x1_nhwc_cutlass(
    const half* input, const half* weight, const half* bias, half* output,
    int N, int H, int W, int Cin, int Cout, Activation act, cudaStream_t stream);
#endif

#if defined(OMNIDREAMS_SINGLEVIEW_USE_CUTLASS)
// Forward declaration for CUTLASS-backed NHWC 3x3 implicit convolution
inline void conv2d_3x3_nhwc_cutlass_implicit(
    const half* input, const half* weight, const half* bias, half* output,
    int N, int H_in, int W_in, int Cin, int Cout, Activation act,
    int stride, int pad, int H_out, int W_out, cudaStream_t stream,
    int chunks);
// (NCHW conv forward declarations removed)
#endif

// (cuDNN implementation removed)

// apply_activation template kept here for CUTLASS kernels that use it
template <Activation ACT>
__device__ __forceinline__ float apply_activation(float x) {
    if constexpr (ACT == Activation::ReLU) {
        return x > 0.0f ? x : 0.0f;
    } else if constexpr (ACT == Activation::ClampTanh3) {
        return clamp_tanh3(x);
    } else {
        return x;
    }
}

// CUTLASS-only implementation - no naive kernels

// Transpose and reorder kernels moved to common; CUTLASS keeps wrappers using them

// Runtime activation (for fused reorder path)
__device__ __forceinline__ float apply_activation_runtime(float x, int act_code) {
    // 0=None, 1=ReLU, 2=ClampTanh3
    if (act_code == 1) return x > 0.0f ? x : 0.0f;
    if (act_code == 2) return clamp_tanh3(x);
    return x;
}

// (NCHW reorder+bias kernel removed)

// CUTLASS-only implementation - no naive kernels

// Trim first K frames from NTCHW buffer: in [N*T, C, H, W] with known N and T.
__global__ void ntchw_trim_front_half_kernel(
    const half* __restrict__ in, half* __restrict__ out,
    int N, int T, int C, int H, int W, int K)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * (T - K) * C * H * W;
    if (idx >= total) return;
    int w = idx % W; idx /= W;
    int h = idx % H; idx /= H;
    int c = idx % C; idx /= C;
    int t = idx % (T - K); idx /= (T - K);
    int n = (int)idx;
    long long in_base  = (((((long long)n * T + (t + K)) * C + c) * H + h) * W + w);
    long long out_base = (((((long long)n * (T - K) + t) * C + c) * H + h) * W + w);
    out[out_base] = in[in_base];
}

// CUTLASS-only implementation - no naive kernels

// Public launchers

// (NCHW tiled 3x3 kernel and helpers removed)

// (NCHW 3x3 tiled host wrapper removed)
// (NCHW 1x1 host wrapper removed)

// (NCHW 3x3 host wrapper removed)

inline void conv2d_3x3_nhwc_half(
    const half* input, const half* weight, const half* bias, half* output,
    int N, int H_in, int W_in, int Cin, int Cout, int groups, Activation act,
    int stride, int pad,
    int H_out, int W_out,
    cudaStream_t stream,
    int chunks = 0)
{
    // CUTLASS-only implementation
    OMNIDREAMS_SINGLEVIEW_ASSERT(input != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(weight != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(output != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N > 0 && Cin > 0 && Cout > 0 && H_in > 0 && W_in > 0);
    OMNIDREAMS_SINGLEVIEW_ASSERT(groups > 0);
    OMNIDREAMS_SINGLEVIEW_ASSERT((Cin % groups) == 0);
    OMNIDREAMS_SINGLEVIEW_ASSERT((Cout % groups) == 0);
    OMNIDREAMS_SINGLEVIEW_ASSERT(stride > 0);
    OMNIDREAMS_SINGLEVIEW_ASSERT(H_out > 0 && W_out > 0);

    if (groups != 1) {
        throw std::runtime_error("conv2d_3x3_nhwc_half: Only groups=1 is supported with CUTLASS backend");
    }

    size_t required_ws = 0;
    if (chunks <= 1) {
        required_ws = estimate_conv3x3_workspace_bytes(N, H_in, W_in, Cin, Cout, stride, pad);
    }
    if (required_ws > 0) {
        // Pre-allocate workspace (caches internally, return value not needed)
        (void)g_conv3x3_workspace_arena.get(required_ws, stream, /*zero_fill=*/true);
    }

    conv2d_3x3_nhwc_cutlass_implicit(
        input, weight, bias, output,
        N, H_in, W_in, Cin, Cout, act, stride, pad, H_out, W_out, stream, chunks);
}

inline void conv2d_1x1_nhwc_half(
    const half* input, const half* weight, const half* bias, half* output,
    int N, int H, int W, int Cin, int Cout, int groups, Activation act,
    cudaStream_t stream)
{
    // CUTLASS-only implementation
    OMNIDREAMS_SINGLEVIEW_ASSERT(input != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(weight != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(output != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N > 0 && Cin > 0 && Cout > 0 && H > 0 && W > 0);
    OMNIDREAMS_SINGLEVIEW_ASSERT(groups > 0);
    OMNIDREAMS_SINGLEVIEW_ASSERT((Cin % groups) == 0);
    OMNIDREAMS_SINGLEVIEW_ASSERT((Cout % groups) == 0);

    conv2d_1x1_nhwc_cutlass(input, weight, bias, output, N, H, W, Cin, Cout, act, stream);
}

// Wrappers now provided by common/cuda_utils.cuh

// Convenience wrappers for single-image CHW/HWC using N=1
// Removed CHW single-image wrappers; use boundary permutes at call sites

// NTHWC <-> NTCHW wrappers (for video inputs with time dimension), often with N=1
// Wrappers now provided by common/cuda_utils.cuh

// Convenience wrappers for TPool/TGrow semantics without data copies.
// TPool: input NTCHW flattened as (N*T, C, H, W). After pooling with stride s,
// pass N' = (N*T)/s and Cin' = s*C into 1x1 conv. Result is (N', C_out, H, W).
inline void tpool_1x1_conv(
    const half* input, const half* weight, const half* bias, half* output,
    int N_mul_T, int C, int H, int W, int stride, int C_out,
    Activation act, cudaStream_t stream)
{
    OMNIDREAMS_SINGLEVIEW_ASSERT(input != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(weight != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(output != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N_mul_T > 0 && C > 0 && H > 0 && W > 0 && stride > 0 && C_out > 0);
    OMNIDREAMS_SINGLEVIEW_ASSERT((N_mul_T % stride) == 0);
    int Nprime = N_mul_T / stride; // assume divisible
    int CinPrime = stride * C;
    // Reorder to NHWC, run NHWC 1x1, reorder back
    static thread_local half* s_in_nhwc = nullptr;
    static thread_local size_t s_in_elems = 0;
    static thread_local half* s_out_nhwc = nullptr;
    static thread_local size_t s_out_elems = 0;
    size_t elems_in = (size_t)Nprime * H * W * CinPrime;
    size_t elems_out = (size_t)Nprime * H * W * C_out;
    if (elems_in > s_in_elems) {
        if (s_in_nhwc) cudaFree(s_in_nhwc);
        cudaError_t err = cudaMalloc(&s_in_nhwc, elems_in * sizeof(half));
        if (err != cudaSuccess) {
            s_in_nhwc = nullptr;
            s_in_elems = 0;
            throw std::runtime_error(std::string("cudaMalloc failed for clamp input (") +
                std::to_string(elems_in * sizeof(half) / 1048576.0) + " MB): " + cudaGetErrorString(err));
        }
        s_in_elems = elems_in;
    }
    if (elems_out > s_out_elems) {
        if (s_out_nhwc) cudaFree(s_out_nhwc);
        cudaError_t err = cudaMalloc(&s_out_nhwc, elems_out * sizeof(half));
        if (err != cudaSuccess) {
            s_out_nhwc = nullptr;
            s_out_elems = 0;
            throw std::runtime_error(std::string("cudaMalloc failed for clamp output (") +
                std::to_string(elems_out * sizeof(half) / 1048576.0) + " MB): " + cudaGetErrorString(err));
        }
        s_out_elems = elems_out;
    }
    half* in_nhwc = s_in_nhwc; half* out_nhwc = s_out_nhwc;
    if (!in_nhwc || !out_nhwc) {
        throw std::runtime_error("Internal error: null workspace pointers in clamp_1x1_conv");
    }
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
    nchw_to_hwc_half(input, in_nhwc, Nprime, H, W, CinPrime, stream);
    conv2d_1x1_nhwc_half(in_nhwc, weight, bias, out_nhwc, Nprime, H, W, CinPrime, C_out, /*groups*/1, act, stream);
    hwc_to_nchw_half(out_nhwc, output, Nprime, H, W, C_out, stream);
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
}

// TGrow: perform 1x1 conv to expand channels by stride, then reinterpret as (N*T*stride, C, H, W)
inline void tgrow_1x1_conv(
    const half* input, const half* weight, const half* bias, half* output,
    int N_mul_T, int C, int H, int W, int stride,
    Activation act, cudaStream_t stream)
{
    OMNIDREAMS_SINGLEVIEW_ASSERT(input != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(weight != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(output != nullptr);
    OMNIDREAMS_SINGLEVIEW_ASSERT(N_mul_T > 0 && C > 0 && H > 0 && W > 0 && stride > 0);
    int C_out = stride * C;
    // Reorder to NHWC, run NHWC 1x1, reorder back
    static thread_local half* s_in_nhwc = nullptr;
    static thread_local size_t s_in_elems = 0;
    static thread_local half* s_out_nhwc = nullptr;
    static thread_local size_t s_out_elems = 0;
    size_t elems_in = (size_t)N_mul_T * H * W * C;
    size_t elems_out = (size_t)N_mul_T * H * W * C_out;
    if (elems_in > s_in_elems) {
        if (s_in_nhwc) cudaFree(s_in_nhwc);
        cudaError_t err = cudaMalloc(&s_in_nhwc, elems_in * sizeof(half));
        if (err != cudaSuccess) {
            s_in_nhwc = nullptr;
            s_in_elems = 0;
            throw std::runtime_error(std::string("cudaMalloc failed for tgrow input (") +
                std::to_string(elems_in * sizeof(half) / 1048576.0) + " MB): " + cudaGetErrorString(err));
        }
        s_in_elems = elems_in;
    }
    if (elems_out > s_out_elems) {
        if (s_out_nhwc) cudaFree(s_out_nhwc);
        cudaError_t err = cudaMalloc(&s_out_nhwc, elems_out * sizeof(half));
        if (err != cudaSuccess) {
            s_out_nhwc = nullptr;
            s_out_elems = 0;
            throw std::runtime_error(std::string("cudaMalloc failed for tgrow output (") +
                std::to_string(elems_out * sizeof(half) / 1048576.0) + " MB): " + cudaGetErrorString(err));
        }
        s_out_elems = elems_out;
    }
    half* in_nhwc = s_in_nhwc; half* out_nhwc = s_out_nhwc;
    if (!in_nhwc || !out_nhwc) {
        throw std::runtime_error("Internal error: null workspace pointers in tgrow_1x1_conv");
    }
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
    nchw_to_hwc_half(input, in_nhwc, N_mul_T, H, W, C, stream);
    conv2d_1x1_nhwc_half(in_nhwc, weight, bias, out_nhwc, N_mul_T, H, W, C, C_out, /*groups*/1, act, stream);
    hwc_to_nchw_half(out_nhwc, output, N_mul_T, H, W, C_out, stream);
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
    // consumer should reinterpret output as (N*T*stride, C, H, W) if desired
}

// Optional CUTLASS/CuTe backends (disabled by default). Provide wrapper signatures
// CUTLASS is required for all operations.

#if defined(OMNIDREAMS_SINGLEVIEW_USE_CUTLASS)
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
// Note: A production-grade implementation would dispatch to cutlass convolution
// (implicit GEMM) or GEMM for 1x1 with proper tensor layouts. Here we keep the
// public API stable and, if CUTLASS is enabled later, these can be replaced.
// CUTLASS implementation for 1x1 convolutions.
__global__ void nhwc_add_bias_act_half_kernel_linear(
    half* out, const half* bias, long long total_elems, int N, Activation act)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;
    int n = (int)(idx % N);
    float v = __half2float(out[idx]);
    if (bias) v += __half2float(bias[n]);
    // 0=None, 1=ReLU, 2=ClampTanh3 (match apply_activation equivalents)
    if (act == Activation::ReLU) v = v > 0.0f ? v : 0.0f;
    else if (act == Activation::ClampTanh3) v = clamp_tanh3(v);
    out[idx] = __float2half(v);
}

inline void nhwc_add_bias_act_half(half* out, const half* bias, int M, int N, Activation act, cudaStream_t stream) {
    // Process [M, N] as a flat 1D buffer to avoid gridDim.y > 65535 invalid configs
    long long total = (long long)M * (long long)N;
    int threads = 256;
    // Chunk to keep gridDim.x within conservative limits across devices
    long long max_chunk = (long long)threads * 65535; // ~16.7M elements per launch
    for (long long processed = 0; processed < total; processed += max_chunk) {
        long long chunk = std::min(max_chunk, total - processed);
        int blocks = (int)((chunk + threads - 1) / threads);
        nhwc_add_bias_act_half_kernel_linear<<<blocks, threads, 0, stream>>>(out + processed, bias, chunk, N, act);
        OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
    }
}

// Forward declarations for inline fusion kernels (defined later in file)
__global__ void fused_residual_add_relu_nhwc_kernel(half* output, const half* residual, long long total);

// Phase 5: Fused residual Add+ReLU helper (for MemBlock skip connections and other residual ops)
// output = ReLU(output + residual)
inline void nhwc_residual_add_relu_half(half* output, const half* residual, long long total_elements, cudaStream_t stream) {
    int threads = 256;
    long long max_chunk = (long long)threads * 65535;

    for (long long processed = 0; processed < total_elements; processed += max_chunk) {
        long long chunk = std::min(max_chunk, total_elements - processed);
        int blocks = (int)((chunk + threads - 1) / threads);
        fused_residual_add_relu_nhwc_kernel<<<blocks, threads, 0, stream>>>(
            output + processed, residual + processed, chunk);
        OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
    }
}

inline void conv2d_1x1_nhwc_cutlass(
    const half* input, const half* weight, const half* bias, half* output,
    int N, int H, int W, int Cin, int Cout, Activation act, cudaStream_t stream)
{
    // Treat NHWC as RowMajor [M, K], M=N*H*W, K=Cin
    // 1x1 convolution is just a matrix multiply: [M,K] × [K,N] = [M,N]
    int M = N * H * W;
    using Element = cutlass::half_t;
    using LayoutA = cutlass::layout::RowMajor;      // [M,K]
    using LayoutB = cutlass::layout::ColumnMajor;   // [K,N] via view of row-major [N,K]
    using LayoutC = cutlass::layout::RowMajor;      // [M,N]
    using Acc = float;
    using Gemm = cutlass::gemm::device::Gemm<
        Element, LayoutA,
        Element, LayoutB,
        Element, LayoutC,
        Acc
    >;

    typename Gemm::Arguments args(
        {M, Cout, Cin},   // problem size: (m, n, k)
        {reinterpret_cast<const Element*>(input), Cin},   // A ptr, lda=K
        {reinterpret_cast<const Element*>(weight), Cin},  // B ptr as ColumnMajor [K,N], ldb=K
        {reinterpret_cast<Element*>(output), Cout},
        {reinterpret_cast<Element*>(output), Cout},
        {1.0f, 0.0f}  // alpha=1, beta=0
    );
    Gemm gemm;
    cutlass::Status st = gemm.initialize(args);
    if (st != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM initialization failed");
    }
    st = gemm(stream);
    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();

    // Phase 4 extension: Apply bias/activation separately
    // TODO: Fuse bias into GEMM epilogue for 1x1 convolutions (lower priority - GEMM is already fast)
    if (bias || act != Activation::None) {
        nhwc_add_bias_act_half(output, bias, M, Cout, act, stream);
    }
}

// Utility: Compute minimum number of chunks needed to keep tensors under CUTLASS 2GB limit
inline int compute_optimal_conv_chunks(int N, int H, int W, int C, int elem_bytes = 2) {
    // CUTLASS device wrapper rejects tensors >= 2^31 bytes
    // For NHWC: activation_size = N*H*W*C elements
    unsigned long long total_elems = (unsigned long long)N * H * W * C;
    unsigned long long total_bytes = total_elems * elem_bytes;
    unsigned long long limit = (1ULL << 31);  // 2GB

    if (total_bytes < limit) {
        return 1;  // No chunking needed
    }

    // Find minimum chunks to bring each chunk under limit
    // chunk_bytes = (total_bytes / chunks), need chunk_bytes < limit
    // => chunks > total_bytes / limit
    int min_chunks = (int)((total_bytes + limit - 1) / limit);

    // Round up to next power of 2 for cleaner splits
    int chunks = 1;
    while (chunks < min_chunks) {
        chunks *= 2;
    }

    // Cap at N to avoid empty chunks
    if (chunks > N) chunks = N;

    return chunks;
}

// --- Phase 3: Regular CUTLASS conv3x3 (no fusion) ---
// Used for cases where bias is null OR activation is not ReLU
// chunks: number of N-dimension splits (default 1 = no chunking; auto-computed if 0)
static void conv2d_3x3_regular(
    const half* input, const half* weight, half* output,
    int N, int H_in, int W_in, int Cin, int Cout,
    int stride, int pad, int H_out, int W_out, cudaStream_t stream,
    int chunks = 1)
{
    // Enforce Tensor Core alignment requirements
    OMNIDREAMS_SINGLEVIEW_ASSERT((Cin % 8) == 0 && "Cin must be a multiple of 8 for Tensor Core alignment");
    OMNIDREAMS_SINGLEVIEW_ASSERT((Cout % 8) == 0 && "Cout must be a multiple of 8 for Tensor Core alignment");
    // Follow CUTLASS example approach with explicit Tensor Core configuration
    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementComputeEpilogue = float;

    using LayoutInputA = cutlass::layout::TensorNHWC;
    using LayoutInputB = cutlass::layout::TensorNHWC;  // Weights in NHWC [Cout, R, S, Cin] (transposed at load time)
    using LayoutOutput = cutlass::layout::TensorNHWC;

    // Tile shapes chosen to match profiler-optimal kernels
    using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 32>;  // M x N x K
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;  // Tensor Core MMA shape

    constexpr int NumStages = 3;  // Multi-stage pipeline
    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementInputB>::value;

    // Regular epilogue: just alpha * conv_result + beta * source
    // Bias and activation applied separately
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementComputeEpilogue
    >;

    using Conv2dFpropKernelOpt = typename cutlass::conv::kernel::DefaultConv2dFprop<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        NumStages,
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kOptimized,
        cutlass::conv::StrideSupport::kStrided,
        AlignmentA,
        AlignmentB
    >::Kernel;
    using ImplicitGemmOpt = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernelOpt>;

    // Analytic fallback (same shapes) for permissive support when optimized rejects
    using Conv2dFpropKernelAna = typename cutlass::conv::kernel::DefaultConv2dFprop<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        NumStages,
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kAnalytic,
        cutlass::conv::StrideSupport::kStrided,
        AlignmentA,
        AlignmentB
    >::Kernel;
    using ImplicitGemmAna = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernelAna>;

    // Setup problem size
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;
    int split_k_slices = 1;

    cutlass::conv::Conv2dProblemSize problem_size(
        {N, H_in, W_in, Cin},       // input size NHWC
        {Cout, 3, 3, Cin},          // filter size KRSC/NHWC (transposed from PyTorch at load time)
        {pad, pad, pad, pad},       // padding (top, bottom, left, right)
        {stride, stride},           // conv stride
        {1, 1},                     // dilation
        {N, H_out, W_out, Cout},    // output size NHWC
        mode,
        split_k_slices
    );

    // Construct kernel-specific TensorRef objects for raw pointers
    // All tensors use NHWC layout - weights are transposed at model load time
    auto stride_a = LayoutInputA::packed({N, H_in, W_in, Cin});          // NHWC
    auto stride_b = LayoutInputB::packed({Cout, 3, 3, Cin});             // NHWC (KRSC)
    auto stride_c = LayoutOutput::packed({N, H_out, W_out, Cout});       // NHWC

    LayoutInputA layout_a(stride_a);
    LayoutInputB layout_b(stride_b);
    LayoutOutput layout_c(stride_c);

    typename Conv2dFpropKernelOpt::TensorRefA tensor_a{
        const_cast<ElementInputA*>(reinterpret_cast<ElementInputA const*>(input)),
        layout_a
    };

    typename Conv2dFpropKernelOpt::TensorRefB tensor_b{
        const_cast<ElementInputB*>(reinterpret_cast<ElementInputB const*>(weight)),
        layout_b
    };

    typename Conv2dFpropKernelOpt::TensorRefC tensor_c{
        reinterpret_cast<ElementOutput*>(output),
        layout_c
    };

    typename Conv2dFpropKernelOpt::TensorRefC tensor_d{
        reinterpret_cast<ElementOutput*>(output),
        layout_c
    };

    // Regular epilogue parameters
    typename EpilogueOp::Params epilogue_params(
        ElementComputeEpilogue(1.0f),  // alpha
        ElementComputeEpilogue(0.0f)   // beta
    );

    // Regular arguments (no broadcast)
    typename ImplicitGemmOpt::Arguments arguments_opt(
        problem_size,
        tensor_a,
        tensor_b,
        tensor_c,
        tensor_d,
        epilogue_params,
        cutlass::conv::SplitKMode::kSerial
    );
    typename ImplicitGemmAna::Arguments arguments_ana(
        problem_size,
        tensor_a,
        tensor_b,
        tensor_c,
        tensor_d,
        epilogue_params,
        cutlass::conv::SplitKMode::kSerial
    );

    ImplicitGemmOpt implicit_gemm_opt;
    ImplicitGemmAna implicit_gemm_ana;

    // Note: Output is zero-initialized by torch::zeros in the caller
    // The epilogue with beta=0.0 will overwrite all output elements

    // Validate input dimensions before CUTLASS
    if (Cin % 8 != 0 || Cout % 8 != 0) {
        char err_msg[512];
        snprintf(err_msg, sizeof(err_msg),
                "CUTLASS conv3x3 requires Cin and Cout to be multiples of 8 for FP16: "
                "N=%d Cin=%d Cout=%d H=%d W=%d (need padding)",
                N, Cin, Cout, H_in, W_in);
        throw std::runtime_error(err_msg);
    }

    if (N <= 0 || Cin <= 0 || Cout <= 0 || H_in <= 0 || W_in <= 0) {
        char err_msg[512];
        snprintf(err_msg, sizeof(err_msg),
                "Invalid conv3x3 dimensions: N=%d Cin=%d Cout=%d H=%d W=%d",
                N, Cin, Cout, H_in, W_in);
        throw std::runtime_error(err_msg);
    }

    // Auto-compute chunks if requested (chunks==0)
    if (chunks == 0) {
        chunks = compute_optimal_conv_chunks(N, H_in, W_in, Cin, sizeof(half));
        // Also check output size
        int out_chunks = compute_optimal_conv_chunks(N, H_out, W_out, Cout, sizeof(half));
        chunks = std::max(chunks, out_chunks);
    }
    if (chunks < 1) chunks = 1;

    // Helper: run optimized kernel in 'num_chunks' along N (batch) dimension
    auto run_optimized_in_chunks = [&](int num_chunks) -> bool {
        if (num_chunks < 1) num_chunks = 1;
        bool all_ok = true;
        for (int i = 0; i < num_chunks; ++i) {
            int n0 = (i * N) / num_chunks;
            int n1 = ((i + 1) * N) / num_chunks;
            int N_chunk = n1 - n0;
            if (N_chunk <= 0) continue;

            // Offset base pointers for chunk start
            long long in_off_e = layout_a(typename LayoutInputA::TensorCoord(n0, 0, 0, 0));
            long long out_off_e = layout_c(typename LayoutOutput::TensorCoord(n0, 0, 0, 0));

            typename Conv2dFpropKernelOpt::TensorRefA ta{
                const_cast<ElementInputA*>(reinterpret_cast<ElementInputA const*>(input)) + in_off_e,
                layout_a
            };
            typename Conv2dFpropKernelOpt::TensorRefB tb{
                const_cast<ElementInputB*>(reinterpret_cast<ElementInputB const*>(weight)),
                layout_b
            };
            typename Conv2dFpropKernelOpt::TensorRefC tc{
                reinterpret_cast<ElementOutput*>(output) + out_off_e,
                layout_c
            };
            typename Conv2dFpropKernelOpt::TensorRefC td{
                reinterpret_cast<ElementOutput*>(output) + out_off_e,
                layout_c
            };

            cutlass::conv::Conv2dProblemSize ps(
                {N_chunk, H_in, W_in, Cin},
                {Cout, 3, 3, Cin},
                {pad, pad, pad, pad},
                {stride, stride},
                {1, 1},
                {N_chunk, H_out, W_out, Cout},
                mode,
                split_k_slices
            );

            typename ImplicitGemmOpt::Arguments args_opt(
                ps, ta, tb, tc, td, epilogue_params, cutlass::conv::SplitKMode::kSerial
            );

            cutlass::Status st_ci = implicit_gemm_opt.can_implement(args_opt);
            if (st_ci != cutlass::Status::kSuccess) { all_ok = false; break; }

            size_t ws = implicit_gemm_opt.get_workspace_size(args_opt);
            void* workspace = nullptr;
            if (ws > 0) {
                workspace = g_conv3x3_workspace_arena.get(ws, stream, true);
                if (workspace == nullptr) { all_ok = false; break; }
            }
            cutlass::Status st = implicit_gemm_opt.initialize(args_opt, workspace);
            if (st != cutlass::Status::kSuccess) { all_ok = false; break; }
            st = implicit_gemm_opt(stream);
            cudaError_t err = cudaPeekAtLastError();
            if (st != cutlass::Status::kSuccess || err == cudaErrorInvalidConfiguration || err == cudaErrorInvalidValue) { all_ok = false; break; }
        }
        return all_ok;
    };

    // Helper: run analytic kernel in 'num_chunks' along N
    auto run_analytic_in_chunks = [&](int num_chunks) -> bool {
        if (num_chunks < 1) num_chunks = 1;
        bool all_ok = true;
        for (int i = 0; i < num_chunks; ++i) {
            int n0 = (i * N) / num_chunks;
            int n1 = ((i + 1) * N) / num_chunks;
            int N_chunk = n1 - n0;
            if (N_chunk <= 0) continue;
            long long in_off_e = layout_a(typename LayoutInputA::TensorCoord(n0, 0, 0, 0));
            long long out_off_e = layout_c(typename LayoutOutput::TensorCoord(n0, 0, 0, 0));
            typename Conv2dFpropKernelAna::TensorRefA ta{
                const_cast<ElementInputA*>(reinterpret_cast<ElementInputA const*>(input)) + in_off_e,
                layout_a
            };
            typename Conv2dFpropKernelAna::TensorRefB tb{
                const_cast<ElementInputB*>(reinterpret_cast<ElementInputB const*>(weight)),
                layout_b
            };
            typename Conv2dFpropKernelAna::TensorRefC tc{
                reinterpret_cast<ElementOutput*>(output) + out_off_e,
                layout_c
            };
            typename Conv2dFpropKernelAna::TensorRefC td{
                reinterpret_cast<ElementOutput*>(output) + out_off_e,
                layout_c
            };
            cutlass::conv::Conv2dProblemSize ps(
                {N_chunk, H_in, W_in, Cin}, {Cout, 3, 3, Cin}, {pad, pad, pad, pad}, {stride, stride}, {1, 1}, {N_chunk, H_out, W_out, Cout}, mode, split_k_slices
            );
            typename ImplicitGemmAna::Arguments args_ana(
                ps, ta, tb, tc, td, epilogue_params, cutlass::conv::SplitKMode::kSerial
            );
            cutlass::Status st_ci = implicit_gemm_ana.can_implement(args_ana);
            if (st_ci != cutlass::Status::kSuccess) { all_ok = false; break; }
            size_t ws = implicit_gemm_ana.get_workspace_size(args_ana);
            void* workspace = nullptr;
            if (ws > 0) { workspace = g_conv3x3_workspace_arena.get(ws, stream, true); if (workspace == nullptr) { all_ok = false; break; } }
            cutlass::Status st = implicit_gemm_ana.initialize(args_ana, workspace);
            if (st != cutlass::Status::kSuccess) { all_ok = false; break; }
            st = implicit_gemm_ana(stream);
            cudaError_t err = cudaPeekAtLastError();
            if (st != cutlass::Status::kSuccess || err == cudaErrorInvalidConfiguration || err == cudaErrorInvalidValue) { all_ok = false; break; }
        }
        return all_ok;
    };

    // Try optimized with requested chunking; escalate if needed
    int current_chunks = chunks;
    while (current_chunks <= std::max(1, N)) {
        if (run_optimized_in_chunks(current_chunks)) {
            OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
            return;
        }
        // can_implement or execution failed; try more chunks
        current_chunks *= 2;
    }

    // Optimized failed even with max chunking; try analytic with requested chunks
    if (run_analytic_in_chunks(chunks)) {
        OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
        return;
    }

    // Both paths failed; construct diagnostic error
    cutlass::Status can_impl_status = implicit_gemm_opt.can_implement(arguments_opt);
    if (can_impl_status != cutlass::Status::kSuccess) {
        // Derive helpful diagnostics without changing behavior
        int const gemm_M = N * H_out * W_out;            // M tile dimension
        int const gemm_N = Cout;                          // N tile dimension
        int const gemm_K = 3 * 3 * Cin;                   // K tile dimension
        // Vectorization assumptions (in elements) for FP16 with 128-bit accesses
        int const vec128_elems = 128 / cutlass::sizeof_bits<ElementInputA>::value; // expected 8 for fp16
        int const vec64_elems  =  64 / cutlass::sizeof_bits<ElementInputA>::value; // expected 4 for fp16
        // Pointer alignment info
        auto in_addr  = reinterpret_cast<uintptr_t>(input);
        auto w_addr   = reinterpret_cast<uintptr_t>(weight);
        auto out_addr = reinterpret_cast<uintptr_t>(output);
        // Compile-time tile shapes
        int const tb_m = ThreadblockShape::kM;
        int const tb_n = ThreadblockShape::kN;
        int const tb_k = ThreadblockShape::kK;
        int const wp_m = WarpShape::kM;
        int const wp_n = WarpShape::kN;
        int const wp_k = WarpShape::kK;
        int const inst_m = InstructionShape::kM;
        int const inst_n = InstructionShape::kN;
        int const inst_k = InstructionShape::kK;
        // Epilogue vector length (elements)
        int const epilogue_vec = 128 / cutlass::sizeof_bits<ElementOutput>::value;

        char err_msg[1024];
        snprintf(err_msg, sizeof(err_msg),
                "CUTLASS implicit conv2d (Optimized) can_implement failed (status=%d '%s')\n"
                "  Problem: N=%d Cin=%d Cout=%d H=%d W=%d H_out=%d W_out=%d stride=%d pad=%d\n"
                "  GEMM dims: M=%d N=%d K=%d (K=R*S*Cin)\n"
                "  TB shape:  M=%d N=%d K=%d; Warp: M=%d N=%d K=%d; MMA: %d x %d x %d\n"
                "  AlignA/B (elements): %d/%d; epilogue_vec=%d (128-bit); alt_vec64=%d\n"
                "  Remainders: K%%vec128=%d N%%vec128=%d; K%%vec64=%d N%%vec64=%d\n"
                "  Ptr align (bytes): input%%16=%llu weight%%16=%llu output%%16=%llu\n"
                "  Note: Failure may be due to iterator/tile constraints in kOptimized, not 8-multiplicity.",
                (int)can_impl_status, cutlass::cutlassGetStatusString(can_impl_status),
                N, Cin, Cout, H_in, W_in, H_out, W_out, stride, pad,
                gemm_M, gemm_N, gemm_K,
                tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, inst_m, inst_n, inst_k,
                AlignmentA, AlignmentB, epilogue_vec, vec64_elems,
                gemm_K % vec128_elems, gemm_N % vec128_elems,
                gemm_K % vec64_elems,  gemm_N % vec64_elems,
                (unsigned long long)(in_addr % 16ULL),
                (unsigned long long)(w_addr % 16ULL),
                (unsigned long long)(out_addr % 16ULL));
        // Try analytic fallback
        cutlass::Status can_impl_status_ana = implicit_gemm_ana.can_implement(arguments_ana);
        if (can_impl_status_ana != cutlass::Status::kSuccess) {
            // Both paths reject: throw with optimized diagnostics (includes tile details)
            throw std::runtime_error(err_msg);
        }
        // Use analytic path
        size_t workspace_size = implicit_gemm_ana.get_workspace_size(arguments_ana);
        void* workspace = nullptr;
        if (workspace_size > 0) {
            workspace = g_conv3x3_workspace_arena.get(workspace_size, stream, true);
            if (workspace == nullptr) {
                char ws_err[512];
                snprintf(ws_err, sizeof(ws_err),
                        "CUTLASS (Analytic) workspace allocation failed: required=%zu bytes. ",
                        workspace_size);
                throw std::runtime_error(ws_err);
            }
        }
        cutlass::Status st = implicit_gemm_ana.initialize(arguments_ana, workspace);
        if (st != cutlass::Status::kSuccess) {
            char ini_err[512];
            snprintf(ini_err, sizeof(ini_err),
                    "CUTLASS (Analytic) initialize failed (status=%d)", (int)st);
            throw std::runtime_error(ini_err);
        }
        st = implicit_gemm_ana(stream);
        cudaError_t err = cudaPeekAtLastError();
        if (st != cutlass::Status::kSuccess || err == cudaErrorInvalidConfiguration || err == cudaErrorInvalidValue) {
            char run_err[512];
            snprintf(run_err, sizeof(run_err),
                    "CUTLASS (Analytic) kernel launch failed (status=%d, cuda_err=%d '%s')",
                    (int)st, (int)err, cudaGetErrorName(err));
            throw std::runtime_error(run_err);
        }
        OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
        return; // analytic path completed successfully
    }

    // Optional success diagnostics gated by env var (no behavior change)
    if (const char* dbg = std::getenv("OMNIDREAMS_DIT_CUTLASS_DEBUG")) {
        (void)dbg; // suppress unused warning
        int const gemm_M = N * H_out * W_out;
        int const gemm_N = Cout;
        int const gemm_K = 3 * 3 * Cin;
        int const tb_m = ThreadblockShape::kM;
        int const tb_n = ThreadblockShape::kN;
        int const tb_k = ThreadblockShape::kK;
        int const wp_m = WarpShape::kM;
        int const wp_n = WarpShape::kN;
        int const wp_k = WarpShape::kK;
        int const inst_m = InstructionShape::kM;
        int const inst_n = InstructionShape::kN;
        int const inst_k = InstructionShape::kK;
        fprintf(stderr,
                "[CUTLASS] can_implement OK | N=%d Cin=%d Cout=%d H=%d W=%d -> H_out=%d W_out=%d stride=%d pad=%d\n"
                "          GEMM M=%d N=%d K=%d | TB %dx%dx%d Warp %dx%dx%d MMA %dx%dx%d | AlignA/B=%d/%d\n",
                N, Cin, Cout, H_in, W_in, H_out, W_out, stride, pad,
                gemm_M, gemm_N, gemm_K,
                tb_m, tb_n, tb_k, wp_m, wp_n, wp_k, inst_m, inst_n, inst_k,
                AlignmentA, AlignmentB);
    }

    // Get workspace from global pool (optimized path)
    size_t workspace_size = implicit_gemm_opt.get_workspace_size(arguments_opt);
    void* workspace = nullptr;

    if (workspace_size > 0) {
        workspace = g_conv3x3_workspace_arena.get(workspace_size, stream, true);
        // Check for allocation failure if workspace was needed
        if (workspace == nullptr) {
            char err_msg[512];
            snprintf(err_msg, sizeof(err_msg),
                    "CUTLASS workspace allocation failed: "
                    "N=%d Cin=%d Cout=%d H=%d W=%d required=%zu bytes. "
                    "Pool capacity=%zu. Try reducing batch size or pre-reserving workspace.",
                    N, Cin, Cout, H_in, W_in, workspace_size,
                    g_conv3x3_workspace_arena.capacity());
            throw std::runtime_error(err_msg);
        }
    } else {
        // workspace_size==0 means CUTLASS doesn't need workspace for this config
        // Try using pre-allocated workspace if available (might help with some CUTLASS versions)
        if (g_conv3x3_workspace_arena.capacity() > 0) {
            workspace = g_conv3x3_workspace_arena.get(1, stream, true);  // Request minimal workspace
        }
    }

    // Initialize
    cutlass::Status st = implicit_gemm_opt.initialize(arguments_opt, workspace);
    if (st != cutlass::Status::kSuccess) {
        char err_msg[512];
        snprintf(err_msg, sizeof(err_msg),
                "CUTLASS implicit conv2d initialize failed (status=%d): "
                "N=%d Cin=%d Cout=%d H=%d W=%d H_out=%d W_out=%d stride=%d pad=%d "
                "workspace=%p workspace_size=%zu pool_capacity=%zu. "
                "This may indicate unsupported tensor dimensions or alignment issues. "
                "Ensure Cin and Cout are multiples of 8 for FP16 tensor cores.",
                (int)st, N, Cin, Cout, H_in, W_in, H_out, W_out, stride, pad,
                workspace, workspace_size, g_conv3x3_workspace_arena.capacity());
        throw std::runtime_error(err_msg);
    }

    // Launch kernel
    st = implicit_gemm_opt(stream);
    cudaError_t err = cudaPeekAtLastError();
    if (st != cutlass::Status::kSuccess || err == cudaErrorInvalidConfiguration || err == cudaErrorInvalidValue) {
        char err_msg[512];
        snprintf(err_msg, sizeof(err_msg),
                "CUTLASS implicit conv2d kernel launch failed (status=%d, cuda_err=%d '%s'): "
                "N=%d Cin=%d Cout=%d H=%d W=%d",
                (int)st, (int)err, cudaGetErrorName(err),
                N, Cin, Cout, H_in, W_in);
        throw std::runtime_error(err_msg);
    }

    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
}

// --- Phase 3-7: Inline fusion kernels ---
// Applied immediately after CUTLASS convolution for true fusion
// Avoids CUTLASS broadcast epilogue compatibility issues

// Phase 3: Fused bias+ReLU kernel
__global__ void fused_bias_relu_nhwc_kernel(
    half* output, const half* bias, int M, int N)
{
    // M = N_batch * H * W (spatial dimensions)
    // N = Cout (channels)
    // Layout: NHWC, so output[m, n] = output[m * N + n]

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)M * N;

    if (idx < total) {
        int channel = idx % N;
        float val = __half2float(output[idx]);
        val += __half2float(bias[channel]);  // Add bias
        val = fmaxf(val, 0.0f);              // ReLU
        output[idx] = __float2half(val);
    }
}

// Phase 4: Fused bias-only kernel (no activation)
__global__ void fused_bias_nhwc_kernel(
    half* output, const half* bias, int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)M * N;

    if (idx < total) {
        int channel = idx % N;
        float val = __half2float(output[idx]);
        val += __half2float(bias[channel]);  // Add bias only
        output[idx] = __float2half(val);
    }
}

// Phase 5: Fused residual Add+ReLU kernel (for MemBlock skip connections)
__global__ void fused_residual_add_relu_nhwc_kernel(
    half* output, const half* residual, long long total)
{
    // output = ReLU(output + residual)
    // Both tensors are same shape [N, H, W, C] in NHWC layout

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        float val = __half2float(output[idx]);
        val += __half2float(residual[idx]);  // Add residual
        val = fmaxf(val, 0.0f);              // ReLU
        output[idx] = __float2half(val);
    }
}

// Phase 7: Fused bias+ClampTanh3 kernel (for final decoder layer)
__global__ void fused_bias_clamptanh3_nhwc_kernel(
    half* output, const half* bias, int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)M * N;

    if (idx < total) {
        int channel = idx % N;
        float val = __half2float(output[idx]);
        val += __half2float(bias[channel]);  // Add bias
        val = clamp_tanh3(val);              // ClampTanh3
        output[idx] = __float2half(val);
    }
}

// --- Phase 3: Fused bias+ReLU via CUTLASS + inline kernel ---
// Uses regular CUTLASS conv followed by fused bias+ReLU kernel
// This avoids DefaultConv2dFpropWithBroadcast compatibility issues
static void conv2d_3x3_fused_bias_relu(
    const half* input, const half* weight, const half* bias, half* output,
    int N, int H_in, int W_in, int Cin, int Cout,
    int stride, int pad, int H_out, int W_out, cudaStream_t stream,
    int chunks = 0)
{
    // Run regular CUTLASS convolution (no epilogue fusion)
    conv2d_3x3_regular(input, weight, output,
                      N, H_in, W_in, Cin, Cout,
                      stride, pad, H_out, W_out, stream, chunks);

    // Immediately apply fused bias+ReLU kernel in chunks (like nhwc_add_bias_act_half)
    int M = N * H_out * W_out;
    long long total = (long long)M * Cout;
    int threads = 256;
    long long max_chunk = (long long)threads * 65535;  // Max elements per launch

    for (long long processed = 0; processed < total; processed += max_chunk) {
        long long chunk = std::min(max_chunk, total - processed);
        int blocks = (int)((chunk + threads - 1) / threads);
        fused_bias_relu_nhwc_kernel<<<blocks, threads, 0, stream>>>(
            output + processed, bias, M, Cout);
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            char err_msg[512];
            snprintf(err_msg, sizeof(err_msg),
                    "Fused bias+ReLU post-kernel failed (cuda_err=%d '%s')",
                    (int)err, cudaGetErrorName(err));
            throw std::runtime_error(err_msg);
        }
    }

    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
}

// --- Phase 4: Fused bias-only via CUTLASS + inline kernel ---
static void conv2d_3x3_fused_bias_only(
    const half* input, const half* weight, const half* bias, half* output,
    int N, int H_in, int W_in, int Cin, int Cout,
    int stride, int pad, int H_out, int W_out, cudaStream_t stream,
    int chunks = 0)
{
    // Run regular CUTLASS convolution
    conv2d_3x3_regular(input, weight, output,
                      N, H_in, W_in, Cin, Cout,
                      stride, pad, H_out, W_out, stream, chunks);

    // Immediately apply fused bias kernel in chunks
    int M = N * H_out * W_out;
    long long total = (long long)M * Cout;
    int threads = 256;
    long long max_chunk = (long long)threads * 65535;

    for (long long processed = 0; processed < total; processed += max_chunk) {
        long long chunk = std::min(max_chunk, total - processed);
        int blocks = (int)((chunk + threads - 1) / threads);
        fused_bias_nhwc_kernel<<<blocks, threads, 0, stream>>>(
            output + processed, bias, M, Cout);
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            char err_msg[512];
            snprintf(err_msg, sizeof(err_msg),
                    "Fused bias post-kernel failed (cuda_err=%d '%s')",
                    (int)err, cudaGetErrorName(err));
            throw std::runtime_error(err_msg);
        }
    }

    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
}

// --- Phase 7: Fused bias+ClampTanh3 via CUTLASS + inline kernel ---
static void conv2d_3x3_fused_bias_clamptanh3(
    const half* input, const half* weight, const half* bias, half* output,
    int N, int H_in, int W_in, int Cin, int Cout,
    int stride, int pad, int H_out, int W_out, cudaStream_t stream,
    int chunks = 0)
{
    // Run regular CUTLASS convolution
    conv2d_3x3_regular(input, weight, output,
                      N, H_in, W_in, Cin, Cout,
                      stride, pad, H_out, W_out, stream, chunks);

    // Immediately apply fused bias+ClampTanh3 kernel in chunks
    int M = N * H_out * W_out;
    long long total = (long long)M * Cout;
    int threads = 256;
    long long max_chunk = (long long)threads * 65535;

    for (long long processed = 0; processed < total; processed += max_chunk) {
        long long chunk = std::min(max_chunk, total - processed);
        int blocks = (int)((chunk + threads - 1) / threads);
        fused_bias_clamptanh3_nhwc_kernel<<<blocks, threads, 0, stream>>>(
            output + processed, bias, M, Cout);
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            char err_msg[512];
            snprintf(err_msg, sizeof(err_msg),
                    "Fused bias+ClampTanh3 post-kernel failed (cuda_err=%d '%s')",
                    (int)err, cudaGetErrorName(err));
            throw std::runtime_error(err_msg);
        }
    }

    OMNIDREAMS_SINGLEVIEW_CUDA_CHECK();
}

// --- Main conv3x3 dispatcher ---
// IMPORTANT: This function expects weights in NHWC format [Cout, R, S, Cin]
// PyTorch weights [Cout, Cin, R, S] must be transposed to [Cout, R, S, Cin] at model load time
//
// ALIGNMENT REQUIREMENTS for Tensor Cores:
// - Cin and Cout must be multiples of 8 (for 128-bit alignment with fp16)
// chunks: 0 = auto-compute, 1 = no chunking, >1 = explicit N-splits
inline void conv2d_3x3_nhwc_cutlass_implicit(
    const half* input, const half* weight, const half* bias, half* output,
    int N, int H_in, int W_in, int Cin, int Cout, Activation act,
    int stride, int pad, int H_out, int W_out, cudaStream_t stream,
    int chunks = 0)
{
    // Phase 3: Fused bias+ReLU (inline kernel approach - always works)
    if (bias != nullptr && act == Activation::ReLU) {
        conv2d_3x3_fused_bias_relu(input, weight, bias, output,
                                     N, H_in, W_in, Cin, Cout,
                                     stride, pad, H_out, W_out, stream, chunks);
        return;
    }

    // Phase 4: Fused bias-only (no activation)
    if (bias != nullptr && act == Activation::None) {
        conv2d_3x3_fused_bias_only(input, weight, bias, output,
                                     N, H_in, W_in, Cin, Cout,
                                     stride, pad, H_out, W_out, stream, chunks);
        return;
    }

    // Phase 7: Fused bias+ClampTanh3 (for final decoder layer)
    if (bias != nullptr && act == Activation::ClampTanh3) {
        conv2d_3x3_fused_bias_clamptanh3(input, weight, bias, output,
                                          N, H_in, W_in, Cin, Cout,
                                          stride, pad, H_out, W_out, stream, chunks);
        return;
    }

    // Regular path: no bias or unsupported activation
    conv2d_3x3_regular(input, weight, output,
                      N, H_in, W_in, Cin, Cout,
                      stride, pad, H_out, W_out, stream, chunks);

    // Apply bias and/or activation separately if needed
    if (bias || act != Activation::None) {
        nhwc_add_bias_act_half(output, bias, N * H_out * W_out, Cout, act, stream);
    }
}

inline void reset_conv3x3_workspace() {
    g_conv3x3_workspace_arena.reset();
}

#endif // OMNIDREAMS_SINGLEVIEW_USE_CUTLASS

} // namespace omnidreams_singleview

// Backward-compat namespace alias during migration
namespace taehv = native;
