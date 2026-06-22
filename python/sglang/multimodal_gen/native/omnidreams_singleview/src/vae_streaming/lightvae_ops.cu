// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "lightvae_ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <string>

#include <cutlass/conv/conv2d_problem_size.h>
#include <cutlass/conv/conv3d_problem_size.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/kernel/default_conv3d_fprop.h>
#include <cutlass/conv/kernel/default_conv3d_fprop_with_broadcast.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/layout/tensor.h>

namespace py = pybind11;

namespace omnidreams_singleview {
namespace {

#define OMNIDREAMS_SINGLEVIEW_VAE_PACK_INPUT_LAYOUT_3D(problem) \
    cutlass::layout::TensorNDHWC::packed({ \
        (problem).N, (problem).D, (problem).H, (problem).W, (problem).C})

#define OMNIDREAMS_SINGLEVIEW_VAE_PACK_WEIGHT_LAYOUT_3D(problem) \
    cutlass::layout::TensorNDHWC::packed({ \
        (problem).K, (problem).T, (problem).R, (problem).S, (problem).C})

#define OMNIDREAMS_SINGLEVIEW_VAE_PACK_OUTPUT_LAYOUT_3D(problem) \
    cutlass::layout::TensorNDHWC::packed({ \
        (problem).N, (problem).Z, (problem).P, (problem).Q, (problem).K})

#define OMNIDREAMS_SINGLEVIEW_VAE_PACK_INPUT_LAYOUT_2D(problem) \
    cutlass::layout::TensorNHWC::packed({ \
        (problem).N, (problem).H, (problem).W, (problem).C})

#define OMNIDREAMS_SINGLEVIEW_VAE_PACK_WEIGHT_LAYOUT_2D(problem) \
    cutlass::layout::TensorNHWC::packed({ \
        (problem).K, (problem).R, (problem).S, (problem).C})

#define OMNIDREAMS_SINGLEVIEW_VAE_PACK_OUTPUT_LAYOUT_2D(problem) \
    cutlass::layout::TensorNHWC::packed({ \
        (problem).N, (problem).P, (problem).Q, (problem).K})

void check_half_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, name, " must be torch.float16");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device_as_input(const torch::Tensor& tensor, const char* name, const torch::Tensor& input) {
    TORCH_CHECK(tensor.device() == input.device(),
                name, " must be on the same CUDA device as input; input=",
                input.device(), " ", name, "=", tensor.device());
}

int ceil_multiple(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

int conv_output_dim_or_zero(int extent, int pad, int kernel, int stride) {
    if (extent <= 0 || kernel <= 0 || stride <= 0) {
        return 0;
    }
    const int padded = extent + 2 * pad;
    if (padded < kernel) {
        return 0;
    }
    return (padded - kernel) / stride + 1;
}

const half* half_ptr_const(const torch::Tensor& tensor) {
    return reinterpret_cast<const half*>(tensor.data_ptr<at::Half>());
}

half* half_ptr(torch::Tensor& tensor) {
    return reinterpret_cast<half*>(tensor.data_ptr<at::Half>());
}

__global__ void bcthw_to_ndhwc_padded_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int bsz,
    int cin,
    int t,
    int h,
    int w,
    int cpad,
    int tpad_left,
    int hpad,
    int wpad,
    int t_total,
    int h_total,
    int w_total) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(bsz) * cin * t * h * w;
    if (idx >= total) {
        return;
    }
    int x = static_cast<int>(idx % w);
    idx /= w;
    int y = static_cast<int>(idx % h);
    idx /= h;
    int ti = static_cast<int>(idx % t);
    idx /= t;
    int c = static_cast<int>(idx % cin);
    idx /= cin;
    int b = static_cast<int>(idx);

    const int to = ti + tpad_left;
    const int yo = y + hpad;
    const int xo = x + wpad;
    const long long out_idx =
        (((static_cast<long long>(b) * t_total + to) * h_total + yo) * w_total + xo) * cpad + c;
    const long long in_idx =
        (((static_cast<long long>(b) * cin + c) * t + ti) * h + y) * w + x;
    output[out_idx] = input[in_idx];
}

__global__ void kctrs_to_ktrsc_padded_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int kout,
    int cin,
    int kt,
    int kh,
    int kw,
    int kpad,
    int cpad) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(kout) * cin * kt * kh * kw;
    if (idx >= total) {
        return;
    }
    int s = static_cast<int>(idx % kw);
    idx /= kw;
    int r = static_cast<int>(idx % kh);
    idx /= kh;
    int tt = static_cast<int>(idx % kt);
    idx /= kt;
    int c = static_cast<int>(idx % cin);
    idx /= cin;
    int k = static_cast<int>(idx);

    const long long in_idx =
        (((static_cast<long long>(k) * cin + c) * kt + tt) * kh + r) * kw + s;
    const long long out_idx =
        ((((static_cast<long long>(k) * kt + tt) * kh + r) * kw + s) * cpad + c);
    output[out_idx] = input[in_idx];
}

__global__ void add_bias_ndhwc_kernel(
    half* __restrict__ data,
    const half* __restrict__ bias,
    long long total,
    int channels,
    int real_channels) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (; idx < total; idx += stride) {
        int c = static_cast<int>(idx % channels);
        if (c < real_channels) {
            data[idx] = __float2half(__half2float(data[idx]) + __half2float(bias[c]));
        }
    }
}

int launch_blocks_linear(long long total) {
    constexpr int threads = 256;
    return static_cast<int>(std::min((total + threads - 1) / threads, 65535LL));
}

__global__ void ndhwc_to_bcthw_crop_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int bsz,
    int kout,
    int t,
    int h,
    int w,
    int kpad) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(bsz) * kout * t * h * w;
    if (idx >= total) {
        return;
    }
    int x = static_cast<int>(idx % w);
    idx /= w;
    int y = static_cast<int>(idx % h);
    idx /= h;
    int ti = static_cast<int>(idx % t);
    idx /= t;
    int c = static_cast<int>(idx % kout);
    idx /= kout;
    int b = static_cast<int>(idx);

    const long long in_idx =
        (((static_cast<long long>(b) * t + ti) * h + y) * w + x) * kpad + c;
    const long long out_idx =
        (((static_cast<long long>(b) * kout + c) * t + ti) * h + y) * w + x;
    output[out_idx] = input[in_idx];
}

cudaError_t cutlass_conv2d_nhwc(
    const half* input,
    const half* weight,
    half* output,
    int n,
    int cin,
    int cout,
    int h,
    int w,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    cudaStream_t stream) {
    const int h_out = conv_output_dim_or_zero(h, 0, kh, stride_h);
    const int w_out = conv_output_dim_or_zero(w, 0, kw, stride_w);
    if (!input || !weight || !output || n <= 0 || cin <= 0 || cout <= 0 ||
        h <= 0 || w <= 0 || h_out <= 0 || w_out <= 0) {
        return cudaSuccess;
    }

    using Element = cutlass::half_t;
    using Accum = float;
    using Conv2dKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        Element, cutlass::layout::TensorNHWC,
        Element, cutlass::layout::TensorNHWC,
        Element, cutlass::layout::TensorNHWC,
        Accum,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            Element, 128 / cutlass::sizeof_bits<Element>::value, Accum, Accum>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2,
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kOptimized,
        cutlass::conv::StrideSupport::kStrided,
        4,
        4>::Kernel;
    using Conv2dOp = cutlass::conv::device::ImplicitGemmConvolution<Conv2dKernel>;

    cutlass::conv::Conv2dProblemSize problem(
        n, h, w, cin, cout, kh, kw, h_out, w_out,
        0, 0, stride_h, stride_w, 1, 1, cutlass::conv::Mode::kCrossCorrelation);
    auto layout_a = OMNIDREAMS_SINGLEVIEW_VAE_PACK_INPUT_LAYOUT_2D(problem);
    auto layout_b = OMNIDREAMS_SINGLEVIEW_VAE_PACK_WEIGHT_LAYOUT_2D(problem);
    auto layout_c = OMNIDREAMS_SINGLEVIEW_VAE_PACK_OUTPUT_LAYOUT_2D(problem);
    typename Conv2dKernel::TensorRefA ref_a(
        const_cast<Element*>(reinterpret_cast<const Element*>(input)), layout_a);
    typename Conv2dKernel::TensorRefB ref_b(
        const_cast<Element*>(reinterpret_cast<const Element*>(weight)), layout_b);
    typename Conv2dKernel::TensorRefC ref_c(reinterpret_cast<Element*>(output), layout_c);
    typename Conv2dKernel::TensorRefC ref_d(reinterpret_cast<Element*>(output), layout_c);
    typename Conv2dKernel::Epilogue::OutputOp::Params epilogue(Accum(1.0f), Accum(0.0f));
    typename Conv2dOp::Arguments args(problem, ref_a, ref_b, ref_c, ref_d, epilogue, cutlass::conv::SplitKMode::kSerial);
    Conv2dOp op;
    cutlass::Status status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorNotSupported;
    }
    status = op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInitializationError;
    }
    status = op(stream);
    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

cudaError_t cutlass_conv3d_ndhwc(
    const half* input,
    const half* weight,
    half* output,
    int n,
    int cin,
    int cout,
    int t,
    int h,
    int w,
    int kt,
    int kh,
    int kw,
    int pad_d,
    int pad_h,
    int pad_w,
    int stride_t,
    int stride_h,
    int stride_w,
    cudaStream_t stream) {
    const int t_out = conv_output_dim_or_zero(t, pad_d, kt, stride_t);
    const int h_out = conv_output_dim_or_zero(h, pad_h, kh, stride_h);
    const int w_out = conv_output_dim_or_zero(w, pad_w, kw, stride_w);
    if (!input || !weight || !output || n <= 0 || cin <= 0 || cout <= 0 ||
        t <= 0 || h <= 0 || w <= 0 || t_out <= 0 || h_out <= 0 || w_out <= 0) {
        return cudaSuccess;
    }
    if (kt == 1 && stride_t == 1 && pad_d == 0 && pad_h == 0 && pad_w == 0) {
        return cutlass_conv2d_nhwc(
            input,
            weight,
            output,
            n * t,
            cin,
            cout,
            h,
            w,
            kh,
            kw,
            stride_h,
            stride_w,
            stream);
    }

    using Element = cutlass::half_t;
    using Accum = float;
    using Conv3dKernel = typename cutlass::conv::kernel::DefaultConv3dFprop<
        Element, cutlass::layout::TensorNDHWC,
        Element, cutlass::layout::TensorNDHWC,
        Element, cutlass::layout::TensorNDHWC,
        Accum,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            Element, 128 / cutlass::sizeof_bits<Element>::value, Accum, Accum>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4,
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kOptimized,
        cutlass::conv::StrideSupport::kStrided>::Kernel;
    using Conv3dOp = cutlass::conv::device::ImplicitGemmConvolution<Conv3dKernel>;

    cutlass::conv::Conv3dProblemSize problem(
        n, t, h, w, cin, cout, kt, kh, kw, t_out, h_out, w_out,
        pad_d, pad_h, pad_w, stride_t, stride_h, stride_w, 1, 1, 1, cutlass::conv::Mode::kCrossCorrelation);
    auto layout_a = OMNIDREAMS_SINGLEVIEW_VAE_PACK_INPUT_LAYOUT_3D(problem);
    auto layout_b = OMNIDREAMS_SINGLEVIEW_VAE_PACK_WEIGHT_LAYOUT_3D(problem);
    auto layout_c = OMNIDREAMS_SINGLEVIEW_VAE_PACK_OUTPUT_LAYOUT_3D(problem);
    typename Conv3dKernel::TensorRefA ref_a(
        const_cast<Element*>(reinterpret_cast<const Element*>(input)), layout_a);
    typename Conv3dKernel::TensorRefB ref_b(
        const_cast<Element*>(reinterpret_cast<const Element*>(weight)), layout_b);
    typename Conv3dKernel::TensorRefC ref_c(reinterpret_cast<Element*>(output), layout_c);
    typename Conv3dKernel::TensorRefC ref_d(reinterpret_cast<Element*>(output), layout_c);
    typename Conv3dKernel::Epilogue::OutputOp::Params epilogue(Accum(1.0f), Accum(0.0f));
    typename Conv3dOp::Arguments args(problem, ref_a, ref_b, ref_c, ref_d, epilogue, cutlass::conv::SplitKMode::kSerial);
    Conv3dOp op;
    cutlass::Status status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorNotSupported;
    }
    status = op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInitializationError;
    }
    status = op(stream);
    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

template <
    int ThreadblockM,
    int ThreadblockN,
    int ThreadblockK,
    int WarpM,
    int WarpN,
    int WarpK,
    int Stages>
cudaError_t cutlass_conv3d_ndhwc_bias(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    int n,
    int cin,
    int cout,
    int t,
    int h,
    int w,
    int kt,
    int kh,
    int kw,
    int pad_d,
    int pad_h,
    int pad_w,
    int stride_t,
    int stride_h,
    int stride_w,
    cudaStream_t stream) {
    const int t_out = conv_output_dim_or_zero(t, pad_d, kt, stride_t);
    const int h_out = conv_output_dim_or_zero(h, pad_h, kh, stride_h);
    const int w_out = conv_output_dim_or_zero(w, pad_w, kw, stride_w);
    if (!input || !weight || !bias || !output || n <= 0 || cin <= 0 || cout <= 0 ||
        t <= 0 || h <= 0 || w <= 0 || t_out <= 0 || h_out <= 0 || w_out <= 0) {
        return cudaSuccess;
    }

    using Element = cutlass::half_t;
    using Accum = float;
    using Compute = float;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
        Element,
        Accum,
        Compute,
        Element,
        Element,
        128 / cutlass::sizeof_bits<Element>::value,
        cutlass::epilogue::thread::Identity<Compute>,
        cutlass::plus<Compute>,
        false>;
    using Conv3dKernel = typename cutlass::conv::kernel::DefaultConv3dFpropWithBroadcast<
        Element, cutlass::layout::TensorNDHWC,
        Element, cutlass::layout::TensorNDHWC,
        Element, cutlass::layout::TensorNDHWC,
        Accum,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
        cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Stages,
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kOptimized,
        cutlass::conv::StrideSupport::kStrided>::Kernel;
    using Conv3dOp = cutlass::conv::device::ImplicitGemmConvolution<Conv3dKernel>;

    cutlass::conv::Conv3dProblemSize problem(
        n, t, h, w, cin, cout, kt, kh, kw, t_out, h_out, w_out,
        pad_d, pad_h, pad_w, stride_t, stride_h, stride_w, 1, 1, 1, cutlass::conv::Mode::kCrossCorrelation);
    auto layout_a = OMNIDREAMS_SINGLEVIEW_VAE_PACK_INPUT_LAYOUT_3D(problem);
    auto layout_b = OMNIDREAMS_SINGLEVIEW_VAE_PACK_WEIGHT_LAYOUT_3D(problem);
    auto layout_c = OMNIDREAMS_SINGLEVIEW_VAE_PACK_OUTPUT_LAYOUT_3D(problem);
    typename Conv3dKernel::TensorRefA ref_a(
        const_cast<Element*>(reinterpret_cast<const Element*>(input)), layout_a);
    typename Conv3dKernel::TensorRefB ref_b(
        const_cast<Element*>(reinterpret_cast<const Element*>(weight)), layout_b);
    typename Conv3dKernel::TensorRefC ref_c(reinterpret_cast<Element*>(output), layout_c);
    typename Conv3dKernel::TensorRefC ref_d(reinterpret_cast<Element*>(output), layout_c);
    typename EpilogueOutputOp::Params epilogue(Compute(1.0f), Compute(0.0f));
    typename Conv3dOp::Arguments args(
        problem,
        ref_a,
        ref_b,
        ref_c,
        ref_d,
        epilogue,
        cutlass::conv::SplitKMode::kSerial,
        const_cast<Element*>(reinterpret_cast<const Element*>(bias)),
        nullptr,
        0,
        cout);
    Conv3dOp op;
    cutlass::Status status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorNotSupported;
    }
    status = op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInitializationError;
    }
    status = op(stream);
    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

cudaError_t cutlass_conv3d_ndhwc_bias(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    int n,
    int cin,
    int cout,
    int t,
    int h,
    int w,
    int kt,
    int kh,
    int kw,
    int pad_d,
    int pad_h,
    int pad_w,
    int stride_t,
    int stride_h,
    int stride_w,
    cudaStream_t stream) {
    if (bias == nullptr || kt == 1) {
        return cudaErrorNotSupported;
    }

    const char* env_tile = std::getenv("OMNIDREAMS_SINGLEVIEW_VAE_CONV3D_TILE");
    const std::string tile = env_tile == nullptr ? "auto" : std::string(env_tile);
    auto run_current = [&]() {
        return cutlass_conv3d_ndhwc_bias<128, 128, 32, 64, 64, 32, 4>(
            input, weight, bias, output,
            n, cin, cout, t, h, w, kt, kh, kw,
            pad_d, pad_h, pad_w, stride_t, stride_h, stride_w, stream);
    };
    auto run_128x64 = [&]() {
        return cutlass_conv3d_ndhwc_bias<128, 64, 32, 64, 32, 32, 4>(
            input, weight, bias, output,
            n, cin, cout, t, h, w, kt, kh, kw,
            pad_d, pad_h, pad_w, stride_t, stride_h, stride_w, stream);
    };
    auto run_128x32 = [&]() {
        return cutlass_conv3d_ndhwc_bias<128, 32, 32, 64, 32, 32, 4>(
            input, weight, bias, output,
            n, cin, cout, t, h, w, kt, kh, kw,
            pad_d, pad_h, pad_w, stride_t, stride_h, stride_w, stream);
    };
    auto run_current_s3 = [&]() {
        return cutlass_conv3d_ndhwc_bias<128, 128, 32, 64, 64, 32, 3>(
            input, weight, bias, output,
            n, cin, cout, t, h, w, kt, kh, kw,
            pad_d, pad_h, pad_w, stride_t, stride_h, stride_w, stream);
    };
    auto run_128x64_s3 = [&]() {
        return cutlass_conv3d_ndhwc_bias<128, 64, 32, 64, 32, 32, 3>(
            input, weight, bias, output,
            n, cin, cout, t, h, w, kt, kh, kw,
            pad_d, pad_h, pad_w, stride_t, stride_h, stride_w, stream);
    };
    auto run_128x32_s3 = [&]() {
        return cutlass_conv3d_ndhwc_bias<128, 32, 32, 64, 32, 32, 3>(
            input, weight, bias, output,
            n, cin, cout, t, h, w, kt, kh, kw,
            pad_d, pad_h, pad_w, stride_t, stride_h, stride_w, stream);
    };
    auto run_256x64_s3 = [&]() {
        return cutlass_conv3d_ndhwc_bias<256, 64, 32, 64, 32, 32, 3>(
            input, weight, bias, output,
            n, cin, cout, t, h, w, kt, kh, kw,
            pad_d, pad_h, pad_w, stride_t, stride_h, stride_w, stream);
    };
    auto run_256x32_s3 = [&]() {
        return cutlass_conv3d_ndhwc_bias<256, 32, 32, 64, 32, 32, 3>(
            input, weight, bias, output,
            n, cin, cout, t, h, w, kt, kh, kw,
            pad_d, pad_h, pad_w, stride_t, stride_h, stride_w, stream);
    };
    auto run_64x64_s3 = [&]() {
        return cutlass_conv3d_ndhwc_bias<64, 64, 32, 32, 32, 32, 3>(
            input, weight, bias, output,
            n, cin, cout, t, h, w, kt, kh, kw,
            pad_d, pad_h, pad_w, stride_t, stride_h, stride_w, stream);
    };
    auto run_64x32_s3 = [&]() {
        return cutlass_conv3d_ndhwc_bias<64, 32, 32, 32, 32, 32, 3>(
            input, weight, bias, output,
            n, cin, cout, t, h, w, kt, kh, kw,
            pad_d, pad_h, pad_w, stride_t, stride_h, stride_w, stream);
    };

    if (tile == "128x128x32" || tile == "current") {
        return run_current();
    }
    if (tile == "128x64x32") {
        return run_128x64();
    }
    if (tile == "128x32x32") {
        return run_128x32();
    }
    if (tile == "128x128x32_s3" || tile == "current_s3") {
        return run_current_s3();
    }
    if (tile == "128x64x32_s3") {
        return run_128x64_s3();
    }
    if (tile == "128x32x32_s3") {
        return run_128x32_s3();
    }
    if (tile == "256x64x32_s3") {
        return run_256x64_s3();
    }
    if (tile == "256x32x32_s3") {
        return run_256x32_s3();
    }
    if (tile == "64x64x32_s3") {
        return run_64x64_s3();
    }
    if (tile == "64x32x32_s3") {
        return run_64x32_s3();
    }
    if (tile == "auto_s3") {
        cudaError_t err = cout <= 32 ? run_128x32_s3() : (cout <= 64 ? run_128x64_s3() : run_current_s3());
        if (err == cudaErrorNotSupported || err == cudaErrorInitializationError) {
            return run_current();
        }
        return err;
    }
    if (tile == "auto" || tile == "auto_256_s3") {
        cudaError_t err = cout <= 32 ? run_256x32_s3() : (cout <= 64 ? run_256x64_s3() : run_current_s3());
        if (err == cudaErrorNotSupported || err == cudaErrorInitializationError) {
            return run_current();
        }
        return err;
    }
    if (tile == "auto_64_s3") {
        cudaError_t err = cout <= 32 ? run_64x32_s3() : (cout <= 64 ? run_64x64_s3() : run_current_s3());
        if (err == cudaErrorNotSupported || err == cudaErrorInitializationError) {
            return run_current();
        }
        return err;
    }
    if (tile != "auto_legacy") {
        return cudaErrorNotSupported;
    }

    cudaError_t err = cout <= 32 ? run_128x32() : (cout <= 64 ? run_128x64() : run_current());
    if (err == cudaErrorNotSupported || err == cudaErrorInitializationError) {
        return run_current();
    }
    return err;
}

}  // namespace

torch::Tensor lightvae_causal_conv3d_bcthw(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int pad_t_left,
    int pad_h,
    int pad_w,
    int stride_t,
    int stride_h,
    int stride_w) {
    check_half_cuda_contiguous(input, "input");
    check_half_cuda_contiguous(weight, "weight");
    check_same_device_as_input(weight, "weight", input);
    TORCH_CHECK(input.dim() == 5, "input must be [B, C, T, H, W]");
    TORCH_CHECK(weight.dim() == 5, "weight must be [K, C, Kt, Kh, Kw]");
    TORCH_CHECK(pad_t_left >= 0 && pad_h >= 0 && pad_w >= 0, "padding must be non-negative");
    TORCH_CHECK(stride_t > 0 && stride_h > 0 && stride_w > 0, "stride must be positive");

    const int bsz = static_cast<int>(input.size(0));
    const int cin = static_cast<int>(input.size(1));
    const int t = static_cast<int>(input.size(2));
    const int h = static_cast<int>(input.size(3));
    const int w = static_cast<int>(input.size(4));
    const int kout = static_cast<int>(weight.size(0));
    const int wcin = static_cast<int>(weight.size(1));
    const int kt = static_cast<int>(weight.size(2));
    const int kh = static_cast<int>(weight.size(3));
    const int kw = static_cast<int>(weight.size(4));
    TORCH_CHECK(wcin == cin, "weight input channels ", wcin, " do not match input channels ", cin);

    torch::Tensor bias_tensor;
    if (bias.has_value()) {
        bias_tensor = bias.value();
        check_half_cuda_contiguous(bias_tensor, "bias");
        check_same_device_as_input(bias_tensor, "bias", input);
        TORCH_CHECK(bias_tensor.dim() == 1 && bias_tensor.size(0) == kout,
                    "bias must be [", kout, "]");
    }

    const int cpad = ceil_multiple(cin, 8);
    const int kpad = ceil_multiple(kout, 8);
    const int t_total = t + pad_t_left;
    const int h_total = h + 2 * pad_h;
    const int w_total = w + 2 * pad_w;
    at::cuda::CUDAGuard guard(input.device());
    auto opts = input.options();
    const int t_out = conv_output_dim_or_zero(t_total, 0, kt, stride_t);
    const int h_out = conv_output_dim_or_zero(h_total, 0, kh, stride_h);
    const int w_out = conv_output_dim_or_zero(w_total, 0, kw, stride_w);
    if (bsz <= 0 || cin <= 0 || kout <= 0 || t <= 0 || h <= 0 || w <= 0 ||
        t_out <= 0 || h_out <= 0 || w_out <= 0) {
        return torch::empty(
            {bsz, kout, std::max(0, t_out), std::max(0, h_out), std::max(0, w_out)},
            opts);
    }
    torch::Tensor input_ndhwc = torch::zeros({bsz, t_total, h_total, w_total, cpad}, opts);
    torch::Tensor weight_ktrsc = torch::zeros({kpad, kt, kh, kw, cpad}, opts);
    torch::Tensor output_ndhwc = torch::empty({bsz, t_out, h_out, w_out, kpad}, opts);
    torch::Tensor output_bcthw = torch::empty({bsz, kout, t_out, h_out, w_out}, opts);
    torch::Tensor bias_padded;
    if (bias_tensor.defined()) {
        if (kpad == kout) {
            bias_padded = bias_tensor;
        } else {
            bias_padded = torch::zeros({kpad}, opts);
            bias_padded.narrow(0, 0, kout).copy_(bias_tensor);
        }
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int threads = 256;

    long long input_total = static_cast<long long>(bsz) * cin * t * h * w;
    if (input_total > 0) {
        bcthw_to_ndhwc_padded_kernel<<<static_cast<int>((input_total + threads - 1) / threads), threads, 0, stream>>>(
        half_ptr_const(input),
        half_ptr(input_ndhwc),
        bsz,
        cin,
        t,
        h,
        w,
        cpad,
        pad_t_left,
        pad_h,
        pad_w,
        t_total,
        h_total,
        w_total);
    }

    long long weight_total = static_cast<long long>(kout) * cin * kt * kh * kw;
    if (weight_total > 0) {
        kctrs_to_ktrsc_padded_kernel<<<static_cast<int>((weight_total + threads - 1) / threads), threads, 0, stream>>>(
        half_ptr_const(weight),
        half_ptr(weight_ktrsc),
        kout,
        cin,
        kt,
        kh,
        kw,
        kpad,
        cpad);
    }

    bool fused_bias = false;
    cudaError_t err = cudaErrorNotSupported;
    if (bias_tensor.defined()) {
        err = cutlass_conv3d_ndhwc_bias(
            half_ptr_const(input_ndhwc),
            half_ptr_const(weight_ktrsc),
            half_ptr_const(bias_padded),
            half_ptr(output_ndhwc),
            bsz,
            cpad,
            kpad,
            t_total,
            h_total,
            w_total,
            kt,
            kh,
            kw,
            0,
            0,
            0,
            stride_t,
            stride_h,
            stride_w,
            stream);
        if (err == cudaSuccess) {
            fused_bias = true;
        } else {
            TORCH_CHECK(err == cudaErrorNotSupported || err == cudaErrorInitializationError,
                        "CUTLASS LightVAE Conv3D fused-bias failed: ", cudaGetErrorString(err));
        }
    }
    if (!fused_bias) {
        err = cutlass_conv3d_ndhwc(
            half_ptr_const(input_ndhwc),
            half_ptr_const(weight_ktrsc),
            half_ptr(output_ndhwc),
            bsz,
            cpad,
            kpad,
            t_total,
            h_total,
            w_total,
            kt,
            kh,
            kw,
            0,
            0,
            0,
            stride_t,
            stride_h,
            stride_w,
            stream);
        TORCH_CHECK(err == cudaSuccess, "CUTLASS LightVAE Conv3D failed: ", cudaGetErrorString(err));
    }
    if (bias_tensor.defined()) {
        long long out_total = static_cast<long long>(bsz) * t_out * h_out * w_out * kpad;
        if (!fused_bias && out_total > 0) {
            add_bias_ndhwc_kernel<<<launch_blocks_linear(out_total), threads, 0, stream>>>(
                half_ptr(output_ndhwc),
                half_ptr_const(bias_tensor),
                out_total,
                kpad,
                kout);
        }
    }

    long long crop_total = static_cast<long long>(bsz) * kout * t_out * h_out * w_out;
    if (crop_total > 0) {
        ndhwc_to_bcthw_crop_kernel<<<static_cast<int>((crop_total + threads - 1) / threads), threads, 0, stream>>>(
        half_ptr_const(output_ndhwc),
        half_ptr(output_bcthw),
        bsz,
        kout,
        t_out,
        h_out,
        w_out,
        kpad);
    }
    return output_bcthw;
}

torch::Tensor cutlass_conv3d_nthwc(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int pad_d,
    int pad_h,
    int pad_w,
    int stride_d,
    int stride_h,
    int stride_w) {
    check_half_cuda_contiguous(input, "input");
    check_half_cuda_contiguous(weight, "weight");
    check_same_device_as_input(weight, "weight", input);
    TORCH_CHECK(input.dim() == 5, "input must be NDHWC [B,T,H,W,C]");
    TORCH_CHECK(weight.dim() == 5, "weight must be [K,T,R,S,C]");
    TORCH_CHECK(pad_d >= 0 && pad_h >= 0 && pad_w >= 0, "padding must be non-negative");
    TORCH_CHECK(stride_d > 0 && stride_h > 0 && stride_w > 0, "stride must be positive");

    const int bsz = static_cast<int>(input.size(0));
    const int t = static_cast<int>(input.size(1));
    const int h = static_cast<int>(input.size(2));
    const int w = static_cast<int>(input.size(3));
    const int cin = static_cast<int>(input.size(4));
    const int cout = static_cast<int>(weight.size(0));
    const int kt = static_cast<int>(weight.size(1));
    const int kh = static_cast<int>(weight.size(2));
    const int kw = static_cast<int>(weight.size(3));
    TORCH_CHECK(weight.size(4) == cin, "weight Cin=", weight.size(4), " does not match input C=", cin);

    torch::Tensor bias_tensor;
    if (bias.has_value()) {
        bias_tensor = bias.value();
        check_half_cuda_contiguous(bias_tensor, "bias");
        check_same_device_as_input(bias_tensor, "bias", input);
        TORCH_CHECK(bias_tensor.dim() == 1 && bias_tensor.size(0) == cout,
                    "bias must be [", cout, "]");
    }

    at::cuda::CUDAGuard guard(input.device());
    const int t_out = conv_output_dim_or_zero(t, pad_d, kt, stride_d);
    const int h_out = conv_output_dim_or_zero(h, pad_h, kh, stride_h);
    const int w_out = conv_output_dim_or_zero(w, pad_w, kw, stride_w);
    if (bsz <= 0 || cin <= 0 || cout <= 0 || t <= 0 || h <= 0 || w <= 0 ||
        t_out <= 0 || h_out <= 0 || w_out <= 0) {
        return torch::empty(
            {bsz, std::max(0, t_out), std::max(0, h_out), std::max(0, w_out), cout},
            input.options());
    }
    auto output = torch::empty({bsz, t_out, h_out, w_out, cout}, input.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bool fused_bias = false;
    cudaError_t err = cudaErrorNotSupported;
    if (bias_tensor.defined()) {
        err = cutlass_conv3d_ndhwc_bias(
            half_ptr_const(input),
            half_ptr_const(weight),
            half_ptr_const(bias_tensor),
            half_ptr(output),
            bsz,
            cin,
            cout,
            t,
            h,
            w,
            kt,
            kh,
            kw,
            pad_d,
            pad_h,
            pad_w,
            stride_d,
            stride_h,
            stride_w,
            stream);
        if (err == cudaSuccess) {
            fused_bias = true;
        } else {
            TORCH_CHECK(err == cudaErrorNotSupported || err == cudaErrorInitializationError,
                        "CUTLASS Conv3D NDHWC fused-bias failed: ", cudaGetErrorString(err));
        }
    }
    if (!fused_bias) {
        err = cutlass_conv3d_ndhwc(
            half_ptr_const(input),
            half_ptr_const(weight),
            half_ptr(output),
            bsz,
            cin,
            cout,
            t,
            h,
            w,
            kt,
            kh,
            kw,
            pad_d,
            pad_h,
            pad_w,
            stride_d,
            stride_h,
            stride_w,
            stream);
        TORCH_CHECK(err == cudaSuccess, "CUTLASS Conv3D NDHWC failed: ", cudaGetErrorString(err));
    }
    if (bias_tensor.defined()) {
        constexpr int threads = 256;
        long long out_total = static_cast<long long>(bsz) * t_out * h_out * w_out * cout;
        if (!fused_bias && out_total > 0) {
            add_bias_ndhwc_kernel<<<launch_blocks_linear(out_total), threads, 0, stream>>>(
                half_ptr(output),
                half_ptr_const(bias_tensor),
                out_total,
                cout,
                cout);
        }
    }
    return output;
}

void bind_lightvae_ops(py::module_& m) {
    m.def(
        "lightvae_causal_conv3d_bcthw",
        &lightvae_causal_conv3d_bcthw,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("pad_t_left"),
        py::arg("pad_h"),
        py::arg("pad_w"),
        py::arg("stride_t") = 1,
        py::arg("stride_h") = 1,
        py::arg("stride_w") = 1,
        "TIN3-free CUTLASS Conv3D helper for LightVAE BCTHW tensors.");
    m.def(
        "cutlass_conv3d_nthwc",
        &cutlass_conv3d_nthwc,
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("pad_d") = 0,
        py::arg("pad_h") = 0,
        py::arg("pad_w") = 0,
        py::arg("stride_d") = 1,
        py::arg("stride_h") = 1,
        py::arg("stride_w") = 1,
        "Prepared CUTLASS Conv3D helper for NDHWC tensors.");
    m.def(
        "lightvae_fp8_quantize_bcthw_to_tin16",
        &lightvae_fp8_quantize_bcthw_to_tin16,
        py::arg("input"),
        py::arg("scale"),
        py::arg("padded_height"),
        py::arg("padded_width"),
        py::arg("padded_channels"),
        "Quantize [1,C,T,H,W] fp16 LightVAE activations to raw E4M3 TIN16 layout.");
    m.def(
        "lightvae_fp8_dequantize_tin16_to_bcthw",
        &lightvae_fp8_dequantize_tin16_to_bcthw,
        py::arg("input"),
        py::arg("scale"),
        py::arg("real_channels"),
        py::arg("real_height"),
        py::arg("real_width"),
        "Dequantize raw E4M3 TIN16 LightVAE activations to [1,C,T,H,W] fp16.");
    m.def(
        "lightvae_fp8_rmsnorm_tin16",
        &lightvae_fp8_rmsnorm_tin16,
        py::arg("input"),
        py::arg("input_scale"),
        py::arg("gamma"),
        py::arg("output_scale"),
        py::arg("real_channels"),
        py::arg("silu") = true,
        "FP8 TIN16 RMSNorm with optional SiLU and FP8 output.");
    m.def(
        "lightvae_fp8_add_tin16",
        &lightvae_fp8_add_tin16,
        py::arg("a"),
        py::arg("b"),
        py::arg("a_scale"),
        py::arg("b_scale"),
        py::arg("output_scale"),
        py::arg("real_channels") = -1,
        "Scaled FP8 TIN16 residual add with FP8 output.");
    m.def(
        "lightvae_fp8_add_relu_tin16",
        &lightvae_fp8_add_relu_tin16,
        py::arg("a"),
        py::arg("b"),
        py::arg("a_scale"),
        py::arg("b_scale"),
        py::arg("output_scale"),
        py::arg("real_channels") = -1,
        "Scaled FP8 TIN16 residual add followed by ReLU with FP8 output.");
    m.def(
        "lightvae_fp8_upsample2x_tin16",
        &lightvae_fp8_upsample2x_tin16,
        py::arg("input"),
        "Nearest-neighbor 2x spatial upsample for raw FP8 TIN16 tensors.");
    m.def(
        "lightvae_extract_mu_normalize_tin16",
        &lightvae_fp8_extract_mu_normalize_tin16,
        py::arg("input"),
        py::arg("input_scale"),
        py::arg("mean"),
        py::arg("inv_std"),
        py::arg("real_height"),
        py::arg("real_width"),
        "Dequantize TIN16 FP8 input, extract the first 16 channels, apply LightVAE latent normalization, and return fp16.");
    m.def(
        "cutlass_conv3x3_nhwc_fp8",
        &cutlass_conv3x3_nhwc_fp8,
        py::arg("input"),
        py::arg("weight"),
        py::arg("stride") = 1,
        py::arg("pad") = 1,
        py::arg("relu") = false,
        "TIN3-free CUTLASS FP8 Tensor Core 3x3 Conv2D primitive for NHWC raw E4M3 bytes.");
    m.def(
        "lightvae_fp8_prepare_conv2d_weight_krsc",
        &lightvae_fp8_prepare_conv2d_weight_krsc,
        py::arg("weight_krsc"),
        py::arg("input_scale"),
        py::arg("output_scale"),
        "Prepare KRSC fp16 weights as raw E4M3 bytes scaled by input_scale/output_scale.");
    m.def(
        "cutlass_conv2d_nhwc_fp8_prepared",
        &cutlass_conv2d_nhwc_fp8_prepared,
        py::arg("input"),
        py::arg("weight"),
        py::arg("output_scale"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("pad") = 1,
        py::arg("relu") = false,
        "TIN3-free CUTLASS FP8 Conv2D over prepared KRSC raw E4M3 weights with optional bias.");
    m.def(
        "lightvae_fp8_tin16_to_nhwc",
        &lightvae_fp8_tin16_to_nhwc,
        py::arg("input"),
        "Convert raw FP8 TIN16 [T,C/16,H,W,16] bytes to raw FP8 NHWC bytes.");
    m.def(
        "lightvae_fp8_nhwc_to_tin16",
        &lightvae_fp8_nhwc_to_tin16,
        py::arg("input"),
        "Convert raw FP8 NHWC bytes to raw FP8 TIN16 [T,C/16,H,W,16] bytes.");
    m.def(
        "lightvae_fp8_nhwc_rescale_to_tin16",
        &lightvae_fp8_nhwc_rescale_to_tin16,
        py::arg("input"),
        py::arg("source_scale"),
        py::arg("output_scale"),
        "Convert raw FP8 NHWC bytes to TIN16 while rescaling from source_scale to output_scale.");
    m.def(
        "lightvae_fp8_pack_causal3_tin16",
        &lightvae_fp8_pack_causal3_tin16,
        py::arg("input"),
        py::arg("cache") = py::none(),
        "Pack current/cache TIN16 frames into a 3-frame causal TIN16 channel bundle.");
    m.def(
        "lightvae_fp8_pack_causal3_tin16_to_nhwc",
        &lightvae_fp8_pack_causal3_tin16_to_nhwc,
        py::arg("input"),
        py::arg("cache") = py::none(),
        "Pack current/cache TIN16 frames directly into raw FP8 NHWC bytes for CUTLASS FP8 Conv2D.");
    m.def(
        "lightvae_fp8_update_tail_cache_tin16",
        &lightvae_fp8_update_tail_cache_tin16,
        py::arg("input"),
        py::arg("cache") = py::none(),
        py::arg("cache_frames") = 2,
        "Update a raw FP8 TIN16 tail cache without leaving TIN16 layout.");
    m.def(
        "lightvae_fp8_spatial_pad_right_bottom_tin16",
        &lightvae_fp8_spatial_pad_right_bottom_tin16,
        py::arg("input"),
        "Append zero bottom/right padding to a raw FP8 TIN16 tensor.");
    m.def(
        "lightvae_fp8_pack_temporal3_tin16",
        &lightvae_fp8_pack_temporal3_tin16,
        py::arg("input"),
        py::arg("cache") = py::none(),
        "Pack temporal-downsample current/cache TIN16 frames into a 3-frame channel bundle.");
    m.def(
        "lightvae_fp8_pack_temporal3_tin16_to_nhwc",
        &lightvae_fp8_pack_temporal3_tin16_to_nhwc,
        py::arg("input"),
        py::arg("cache") = py::none(),
        "Pack temporal-downsample current/cache TIN16 frames directly into raw FP8 NHWC bytes.");
    m.def(
        "lightvae_fp8_causal_conv3_tin16_prepared",
        &lightvae_fp8_causal_conv3_tin16_prepared,
        py::arg("input"),
        py::arg("cache"),
        py::arg("weight"),
        py::arg("output_scale"),
        py::arg("bias") = py::none(),
        py::arg("relu") = false,
        "Run a true-FP8 LightVAE causal 3x3 stage from TIN16 input/cache to TIN16 output.");
    m.def(
        "lightvae_fp8_spatial_conv3_tin16_prepared",
        &lightvae_fp8_spatial_conv3_tin16_prepared,
        py::arg("input"),
        py::arg("weight"),
        py::arg("output_scale"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("pad") = 1,
        py::arg("relu") = false,
        py::arg("pad_right_bottom") = false,
        "Run a true-FP8 LightVAE spatial 3x3 stage from TIN16 input to TIN16 output.");
    m.def(
        "lightvae_fp8_conv1_tin16_prepared",
        &lightvae_fp8_conv1_tin16_prepared,
        py::arg("input"),
        py::arg("weight"),
        py::arg("output_scale"),
        py::arg("bias") = py::none(),
        py::arg("relu") = false,
        "Run a true-FP8 LightVAE 1x1 stage from TIN16 input to TIN16 output.");
    m.def(
        "lightvae_fp8_temporal_conv1_tin16_prepared",
        &lightvae_fp8_temporal_conv1_tin16_prepared,
        py::arg("input"),
        py::arg("cache"),
        py::arg("weight"),
        py::arg("output_scale"),
        py::arg("bias") = py::none(),
        py::arg("relu") = false,
        "Run a true-FP8 LightVAE temporal-downsample 1x1 stage from packed TIN16 input/cache to TIN16 output.");
    m.def(
        "lightvae_fp8_qkv_tin16_to_bmhd",
        &lightvae_fp8_qkv_tin16_to_bmhd,
        py::arg("qkv"),
        "Split raw FP8 TIN16 [T,288/16,H,W,16] QKV into cuDNN BMHD Q/K/V byte tensors.");
    m.def(
        "lightvae_fp8_bmhd_to_tin16",
        &lightvae_fp8_bmhd_to_tin16,
        py::arg("input"),
        py::arg("frames"),
        py::arg("height"),
        py::arg("width"),
        "Pack cuDNN BMHD raw FP8 attention output back into TIN16 [T,96/16,H,W,16].");
    m.def(
        "lightvae_fp8_sdpa_bmhd",
        &lightvae_fp8_sdpa_bmhd,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("qkv_scale"),
        py::arg("sdpa_inverse_scale"),
        py::arg("unit_scale"),
        py::arg("batch"),
        py::arg("seq"),
        py::arg("attn_scale"),
        "Run cuDNN FP8 SDPA for LightVAE middle attention over raw FP8 BMHD bytes.");
    m.def(
        "lightvae_fp8_causal_conv3_tin16_direct_prepared",
        &lightvae_fp8_causal_conv3_tin16_direct_prepared,
        py::arg("input"),
        py::arg("cache"),
        py::arg("weight"),
        py::arg("output_scale"),
        py::arg("bias") = py::none(),
        py::arg("relu") = false,
        "Run the experimental direct-TIN16 FP8 causal 3x3 stage without pack/NHWC bridge launches.");
    m.def(
        "lightvae_fp8_spatial_conv3_tin16_direct_prepared",
        &lightvae_fp8_spatial_conv3_tin16_direct_prepared,
        py::arg("input"),
        py::arg("weight"),
        py::arg("output_scale"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("pad") = 1,
        py::arg("relu") = false,
        py::arg("pad_right_bottom") = false,
        "Run the experimental direct-TIN16 FP8 spatial 3x3 stage without pad/NHWC bridge launches.");
    m.def(
        "lightvae_fp8_conv1_tin16_direct_prepared",
        &lightvae_fp8_conv1_tin16_direct_prepared,
        py::arg("input"),
        py::arg("weight"),
        py::arg("output_scale"),
        py::arg("bias") = py::none(),
        py::arg("relu") = false,
        "Run the experimental direct-TIN16 FP8 1x1 stage without NHWC bridge launches.");
    m.def(
        "lightvae_fp8_temporal_conv1_tin16_direct_prepared",
        &lightvae_fp8_temporal_conv1_tin16_direct_prepared,
        py::arg("input"),
        py::arg("cache"),
        py::arg("weight"),
        py::arg("output_scale"),
        py::arg("bias") = py::none(),
        py::arg("relu") = false,
        "Run the experimental direct-TIN16 FP8 temporal 1x1 stage without temporal-pack/NHWC bridge launches.");
    m.def(
        "lightvae_fp8_conv1_tin16_warp_mma_prepared",
        &lightvae_fp8_conv1_tin16_warp_mma_prepared,
        py::arg("input"),
        py::arg("weight"),
        py::arg("output_scale"),
        py::arg("bias") = py::none(),
        py::arg("relu") = false,
        "Run the warp-MMA Tensor Core TIN16 FP8 1x1 stage.");
    m.def(
        "lightvae_fp8_spatial_conv3_tin16_warp_mma_prepared",
        &lightvae_fp8_spatial_conv3_tin16_warp_mma_prepared,
        py::arg("input"),
        py::arg("weight"),
        py::arg("output_scale"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("pad") = 1,
        py::arg("relu") = false,
        py::arg("pad_right_bottom") = false,
        "Run the experimental warp-MMA Tensor Core TIN16 FP8 spatial 3x3 stage.");
    m.def(
        "lightvae_fp8_causal_conv3_tin16_warp_mma_prepared",
        &lightvae_fp8_causal_conv3_tin16_warp_mma_prepared,
        py::arg("input"),
        py::arg("cache"),
        py::arg("weight"),
        py::arg("output_scale"),
        py::arg("bias") = py::none(),
        py::arg("relu") = false,
        "Run the experimental warp-MMA Tensor Core TIN16 FP8 causal 3x3 stage.");
    m.def(
        "lightvae_fp8_spatial_conv3_tin16_warp_mma_scaled_prepared",
        &lightvae_fp8_spatial_conv3_tin16_warp_mma_scaled_prepared,
        py::arg("input"),
        py::arg("weight"),
        py::arg("epilogue_scale"),
        py::arg("bias_scaled") = py::none(),
        py::arg("stride") = 1,
        py::arg("pad") = 1,
        py::arg("relu") = false,
        py::arg("pad_right_bottom") = false,
        "Run a warp-MMA FP8 spatial 3x3 stage with per-output epilogue scaling.");
    m.def(
        "lightvae_fp8_causal_conv3_tin16_warp_mma_scaled_prepared",
        &lightvae_fp8_causal_conv3_tin16_warp_mma_scaled_prepared,
        py::arg("input"),
        py::arg("cache"),
        py::arg("weight"),
        py::arg("epilogue_scale"),
        py::arg("bias_scaled") = py::none(),
        py::arg("relu") = false,
        "Run a warp-MMA FP8 causal 3x3 stage with per-output epilogue scaling.");
}

}  // namespace omnidreams_singleview
