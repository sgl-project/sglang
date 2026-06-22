// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "lightvae_ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>

#include <algorithm>

#include <cutlass/numeric_conversion.h>

namespace omnidreams_singleview {
namespace {

constexpr int kFp8ChannelsPerSlice = 16;

int launch_blocks(long long total) {
    constexpr int threads = 256;
    return static_cast<int>(std::min((total + threads - 1) / threads, 65535LL));
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

void check_u8_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    check_cuda_tensor(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == at::kByte, name, " must be torch.uint8");
}

void check_half_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    check_cuda_tensor(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, name, " must be torch.float16");
}

void check_scale_cuda_tensor(const torch::Tensor& tensor, const char* name) {
    check_cuda_tensor(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kHalf,
                name, " must be torch.float32 or torch.float16");
    TORCH_CHECK(tensor.dim() == 0 || tensor.dim() == 1, name, " must be a scalar or 1D");
}

void check_u8_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    check_cuda_contiguous(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == at::kByte, name, " must be torch.uint8");
}

void check_half_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    check_cuda_contiguous(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, name, " must be torch.float16");
}

torch::Tensor ensure_device_dtype_contiguous(
    const torch::Tensor& tensor,
    const c10::Device& device,
    at::ScalarType dtype) {
    if (tensor.device() == device && tensor.scalar_type() == dtype && tensor.is_contiguous()) {
        return tensor;
    }
    return tensor.to(device, dtype, false, false).contiguous();
}

void check_tin16_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    check_u8_cuda_contiguous(tensor, name);
    TORCH_CHECK(tensor.dim() == 5 && tensor.size(4) == kFp8ChannelsPerSlice,
                name, " must be [T,C/16,H,W,16], got ", tensor.sizes());
}

const half* half_ptr_const(const torch::Tensor& tensor) {
    return reinterpret_cast<const half*>(tensor.data_ptr<at::Half>());
}

template <typename ScaleT>
__device__ __forceinline__ float scale_to_float(const ScaleT* scale, int idx) {
    return static_cast<float>(scale[idx]);
}

template <>
__device__ __forceinline__ float scale_to_float<half>(const half* scale, int idx) {
    return __half2float(scale[idx]);
}

__device__ __forceinline__ uint8_t encode_e4m3(float value) {
    cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;
    cutlass::float_e4m3_t fp8 = to_fp8(value);
    return *reinterpret_cast<uint8_t*>(&fp8);
}

__device__ __forceinline__ float decode_e4m3(uint8_t value) {
    cutlass::float_e4m3_t fp8;
    *reinterpret_cast<uint8_t*>(&fp8) = value;
    cutlass::NumericConverter<float, cutlass::float_e4m3_t> to_float;
    return to_float(fp8);
}

template <typename ScaleT>
__device__ __forceinline__ uint8_t finalize_fp8_stage_value(
    float accum,
    const half* __restrict__ bias,
    const ScaleT* __restrict__ output_scale,
    int output_channel,
    int output_scale_count,
    bool relu) {
    if (bias != nullptr) {
        const int scale_idx = output_scale_count == 1 ? 0 : output_channel;
        const float out_s = fmaxf(scale_to_float(output_scale, scale_idx), 1.0e-12f);
        accum += __half2float(bias[output_channel]) / out_s;
    }
    if (relu) {
        accum = fmaxf(accum, 0.0f);
    }
    return encode_e4m3(accum);
}

template <typename ScaleT>
__global__ void conv1_tin16_direct_kernel(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ weight,
    uint8_t* __restrict__ output,
    const ScaleT* __restrict__ output_scale,
    const half* __restrict__ bias,
    long long total,
    int frames,
    int in_slices,
    int height,
    int width,
    int out_channels,
    int output_scale_count,
    bool relu) {
    const int in_channels = in_slices * kFp8ChannelsPerSlice;
    const int out_slices = out_channels / kFp8ChannelsPerSlice;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long rem = idx;
        const int lane = static_cast<int>(rem % kFp8ChannelsPerSlice);
        rem /= kFp8ChannelsPerSlice;
        const int x = static_cast<int>(rem % width);
        rem /= width;
        const int y = static_cast<int>(rem % height);
        rem /= height;
        const int out_slice = static_cast<int>(rem % out_slices);
        rem /= out_slices;
        const int frame = static_cast<int>(rem);
        const int k = out_slice * kFp8ChannelsPerSlice + lane;

        float accum = 0.0f;
        for (int c = 0; c < in_channels; ++c) {
            const int in_slice = c / kFp8ChannelsPerSlice;
            const int in_lane = c - in_slice * kFp8ChannelsPerSlice;
            const size_t in_idx =
                (((static_cast<size_t>(frame) * in_slices + in_slice) * height + y) * width + x) *
                    kFp8ChannelsPerSlice + in_lane;
            const size_t w_idx = static_cast<size_t>(k) * in_channels + c;
            accum += decode_e4m3(input[in_idx]) * decode_e4m3(weight[w_idx]);
        }
        output[idx] = finalize_fp8_stage_value(accum, bias, output_scale, k, output_scale_count, relu);
    }
}

template <typename ScaleT>
__global__ void spatial3_tin16_direct_kernel(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ weight,
    uint8_t* __restrict__ output,
    const ScaleT* __restrict__ output_scale,
    const half* __restrict__ bias,
    long long total,
    int frames,
    int in_slices,
    int height,
    int width,
    int effective_height,
    int effective_width,
    int out_height,
    int out_width,
    int out_channels,
    int stride,
    int pad,
    int output_scale_count,
    bool relu) {
    const int in_channels = in_slices * kFp8ChannelsPerSlice;
    const int out_slices = out_channels / kFp8ChannelsPerSlice;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long rem = idx;
        const int lane = static_cast<int>(rem % kFp8ChannelsPerSlice);
        rem /= kFp8ChannelsPerSlice;
        const int ox = static_cast<int>(rem % out_width);
        rem /= out_width;
        const int oy = static_cast<int>(rem % out_height);
        rem /= out_height;
        const int out_slice = static_cast<int>(rem % out_slices);
        rem /= out_slices;
        const int frame = static_cast<int>(rem);
        const int k = out_slice * kFp8ChannelsPerSlice + lane;

        float accum = 0.0f;
        for (int r = 0; r < 3; ++r) {
            const int iy = oy * stride + r - pad;
            if (iy < 0 || iy >= effective_height || iy >= height) {
                continue;
            }
            for (int s = 0; s < 3; ++s) {
                const int ix = ox * stride + s - pad;
                if (ix < 0 || ix >= effective_width || ix >= width) {
                    continue;
                }
                for (int c = 0; c < in_channels; ++c) {
                    const int in_slice = c / kFp8ChannelsPerSlice;
                    const int in_lane = c - in_slice * kFp8ChannelsPerSlice;
                    const size_t in_idx =
                        (((static_cast<size_t>(frame) * in_slices + in_slice) * height + iy) * width + ix) *
                            kFp8ChannelsPerSlice + in_lane;
                    const size_t w_idx =
                        (((static_cast<size_t>(k) * 3 + r) * 3 + s) * in_channels + c);
                    accum += decode_e4m3(input[in_idx]) * decode_e4m3(weight[w_idx]);
                }
            }
        }
        output[idx] = finalize_fp8_stage_value(accum, bias, output_scale, k, output_scale_count, relu);
    }
}

template <typename ScaleT>
__global__ void causal3_tin16_direct_kernel(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ cache,
    const uint8_t* __restrict__ weight,
    uint8_t* __restrict__ output,
    const ScaleT* __restrict__ output_scale,
    const half* __restrict__ bias,
    long long total,
    int frames,
    int cache_frames,
    int in_slices,
    int height,
    int width,
    int out_channels,
    int output_scale_count,
    bool relu) {
    const int in_channels = in_slices * kFp8ChannelsPerSlice;
    const int packed_channels = in_channels * 3;
    const int out_slices = out_channels / kFp8ChannelsPerSlice;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long rem = idx;
        const int lane = static_cast<int>(rem % kFp8ChannelsPerSlice);
        rem /= kFp8ChannelsPerSlice;
        const int ox = static_cast<int>(rem % width);
        rem /= width;
        const int oy = static_cast<int>(rem % height);
        rem /= height;
        const int out_slice = static_cast<int>(rem % out_slices);
        rem /= out_slices;
        const int frame = static_cast<int>(rem);
        const int k = out_slice * kFp8ChannelsPerSlice + lane;

        float accum = 0.0f;
        for (int r = 0; r < 3; ++r) {
            const int iy = oy + r - 1;
            if (iy < 0 || iy >= height) {
                continue;
            }
            for (int s = 0; s < 3; ++s) {
                const int ix = ox + s - 1;
                if (ix < 0 || ix >= width) {
                    continue;
                }
                for (int dt = 0; dt < 3; ++dt) {
                    const int src_ext_t = cache_frames + frame + dt - 2;
                    const bool from_cache = src_ext_t >= 0 && src_ext_t < cache_frames && cache != nullptr;
                    const int src_t = from_cache ? src_ext_t : (src_ext_t - cache_frames);
                    const uint8_t* src_ptr = from_cache ? cache : input;
                    const int src_frames = from_cache ? cache_frames : frames;
                    if (src_t < 0 || src_t >= src_frames) {
                        continue;
                    }
                    for (int c = 0; c < in_channels; ++c) {
                        const int in_slice = c / kFp8ChannelsPerSlice;
                        const int in_lane = c - in_slice * kFp8ChannelsPerSlice;
                        const size_t in_idx =
                            (((static_cast<size_t>(src_t) * in_slices + in_slice) * height + iy) * width + ix) *
                                kFp8ChannelsPerSlice + in_lane;
                        const int packed_c = dt * in_channels + c;
                        const size_t w_idx =
                            (((static_cast<size_t>(k) * 3 + r) * 3 + s) * packed_channels + packed_c);
                        accum += decode_e4m3(src_ptr[in_idx]) * decode_e4m3(weight[w_idx]);
                    }
                }
            }
        }
        output[idx] = finalize_fp8_stage_value(accum, bias, output_scale, k, output_scale_count, relu);
    }
}

template <typename ScaleT>
__global__ void temporal1_tin16_direct_kernel(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ cache,
    const uint8_t* __restrict__ weight,
    uint8_t* __restrict__ output,
    const ScaleT* __restrict__ output_scale,
    const half* __restrict__ bias,
    long long total,
    int frames,
    int t_out,
    int in_slices,
    int height,
    int width,
    int out_channels,
    int output_scale_count,
    bool has_cache,
    bool relu) {
    const int in_channels = in_slices * kFp8ChannelsPerSlice;
    const int packed_channels = in_channels * 3;
    const int out_slices = out_channels / kFp8ChannelsPerSlice;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long rem = idx;
        const int lane = static_cast<int>(rem % kFp8ChannelsPerSlice);
        rem /= kFp8ChannelsPerSlice;
        const int x = static_cast<int>(rem % width);
        rem /= width;
        const int y = static_cast<int>(rem % height);
        rem /= height;
        const int out_slice = static_cast<int>(rem % out_slices);
        rem /= out_slices;
        const int to = static_cast<int>(rem);
        const int k = out_slice * kFp8ChannelsPerSlice + lane;

        float accum = 0.0f;
        for (int dt = 0; dt < 3; ++dt) {
            const int src_ext_t = to * 2 + dt;
            const bool from_cache = has_cache && src_ext_t == 0 && cache != nullptr;
            const int src_t = from_cache ? 0 : (has_cache ? src_ext_t - 1 : src_ext_t);
            if (!from_cache && (src_t < 0 || src_t >= frames)) {
                continue;
            }
            const uint8_t* src_ptr = from_cache ? cache : input;
            for (int c = 0; c < in_channels; ++c) {
                const int in_slice = c / kFp8ChannelsPerSlice;
                const int in_lane = c - in_slice * kFp8ChannelsPerSlice;
                const size_t in_idx =
                    (((static_cast<size_t>(src_t) * in_slices + in_slice) * height + y) * width + x) *
                        kFp8ChannelsPerSlice + in_lane;
                const int packed_c = dt * in_channels + c;
                const size_t w_idx = static_cast<size_t>(k) * packed_channels + packed_c;
                accum += decode_e4m3(src_ptr[in_idx]) * decode_e4m3(weight[w_idx]);
            }
        }
        output[idx] = finalize_fp8_stage_value(accum, bias, output_scale, k, output_scale_count, relu);
    }
}

template <typename ScaleT>
torch::Tensor dispatch_conv1_direct(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output_scale,
    const half* bias,
    bool relu) {
    const int frames = static_cast<int>(input.size(0));
    const int in_slices = static_cast<int>(input.size(1));
    const int height = static_cast<int>(input.size(2));
    const int width = static_cast<int>(input.size(3));
    const int out_channels = static_cast<int>(weight.size(0));
    auto output = torch::empty(
        {frames, out_channels / kFp8ChannelsPerSlice, height, width, kFp8ChannelsPerSlice},
        input.options());
    const long long total = output.numel();
    if (total == 0) {
        return output;
    }
    conv1_tin16_direct_kernel<ScaleT><<<launch_blocks(total), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.template data_ptr<uint8_t>(),
        weight.template data_ptr<uint8_t>(),
        output.template data_ptr<uint8_t>(),
        reinterpret_cast<const ScaleT*>(output_scale.data_ptr()),
        bias,
        total,
        frames,
        in_slices,
        height,
        width,
        out_channels,
        static_cast<int>(output_scale.numel()),
        relu);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <typename ScaleT>
torch::Tensor dispatch_spatial3_direct(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output_scale,
    const half* bias,
    int stride,
    int pad,
    bool relu,
    bool pad_right_bottom) {
    const int frames = static_cast<int>(input.size(0));
    const int in_slices = static_cast<int>(input.size(1));
    const int height = static_cast<int>(input.size(2));
    const int width = static_cast<int>(input.size(3));
    const int effective_height = height + (pad_right_bottom ? 1 : 0);
    const int effective_width = width + (pad_right_bottom ? 1 : 0);
    const int out_channels = static_cast<int>(weight.size(0));
    const int out_height = (effective_height + 2 * pad - 3) / stride + 1;
    const int out_width = (effective_width + 2 * pad - 3) / stride + 1;
    TORCH_CHECK(out_height > 0 && out_width > 0, "invalid output shape");
    auto output = torch::empty(
        {frames, out_channels / kFp8ChannelsPerSlice, out_height, out_width, kFp8ChannelsPerSlice},
        input.options());
    const long long total = output.numel();
    if (total == 0) {
        return output;
    }
    spatial3_tin16_direct_kernel<ScaleT><<<launch_blocks(total), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.template data_ptr<uint8_t>(),
        weight.template data_ptr<uint8_t>(),
        output.template data_ptr<uint8_t>(),
        reinterpret_cast<const ScaleT*>(output_scale.data_ptr()),
        bias,
        total,
        frames,
        in_slices,
        height,
        width,
        effective_height,
        effective_width,
        out_height,
        out_width,
        out_channels,
        stride,
        pad,
        static_cast<int>(output_scale.numel()),
        relu);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <typename ScaleT>
torch::Tensor dispatch_causal3_direct(
    torch::Tensor input,
    torch::Tensor cache,
    const uint8_t* cache_ptr,
    int cache_frames,
    torch::Tensor weight,
    torch::Tensor output_scale,
    const half* bias,
    bool relu) {
    const int frames = static_cast<int>(input.size(0));
    const int in_slices = static_cast<int>(input.size(1));
    const int height = static_cast<int>(input.size(2));
    const int width = static_cast<int>(input.size(3));
    const int out_channels = static_cast<int>(weight.size(0));
    auto output = torch::empty(
        {frames, out_channels / kFp8ChannelsPerSlice, height, width, kFp8ChannelsPerSlice},
        input.options());
    const long long total = output.numel();
    if (total == 0) {
        return output;
    }
    causal3_tin16_direct_kernel<ScaleT><<<launch_blocks(total), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.template data_ptr<uint8_t>(),
        cache_ptr,
        weight.template data_ptr<uint8_t>(),
        output.template data_ptr<uint8_t>(),
        reinterpret_cast<const ScaleT*>(output_scale.data_ptr()),
        bias,
        total,
        frames,
        cache_frames,
        in_slices,
        height,
        width,
        out_channels,
        static_cast<int>(output_scale.numel()),
        relu);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <typename ScaleT>
torch::Tensor dispatch_temporal1_direct(
    torch::Tensor input,
    torch::Tensor cache,
    const uint8_t* cache_ptr,
    bool has_cache,
    torch::Tensor weight,
    torch::Tensor output_scale,
    const half* bias,
    bool relu) {
    const int frames = static_cast<int>(input.size(0));
    const int in_slices = static_cast<int>(input.size(1));
    const int height = static_cast<int>(input.size(2));
    const int width = static_cast<int>(input.size(3));
    const int out_channels = static_cast<int>(weight.size(0));
    const int available = frames + (has_cache ? 1 : 0);
    const int t_out = available < 3 ? 0 : ((available - 3) / 2 + 1);
    TORCH_CHECK(t_out >= 0, "temporal direct stage produced invalid T_out=", t_out, " from T=", frames);
    auto output = torch::empty(
        {t_out, out_channels / kFp8ChannelsPerSlice, height, width, kFp8ChannelsPerSlice},
        input.options());
    const long long total = output.numel();
    if (total == 0) {
        return output;
    }
    temporal1_tin16_direct_kernel<ScaleT><<<launch_blocks(total), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.template data_ptr<uint8_t>(),
        cache_ptr,
        weight.template data_ptr<uint8_t>(),
        output.template data_ptr<uint8_t>(),
        reinterpret_cast<const ScaleT*>(output_scale.data_ptr()),
        bias,
        total,
        frames,
        t_out,
        in_slices,
        height,
        width,
        out_channels,
        static_cast<int>(output_scale.numel()),
        has_cache,
        relu);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor normalize_optional_cache(
    c10::optional<torch::Tensor> cache,
    const torch::Tensor& input,
    int expected_cache_frames,
    const char* name,
    const uint8_t** cache_ptr,
    int* cache_frames,
    bool* has_cache) {
    *cache_ptr = nullptr;
    *cache_frames = 0;
    *has_cache = false;
    if (!(cache.has_value() && cache.value().defined())) {
        return torch::Tensor();
    }
    auto cache_c = cache.value().contiguous();
    check_tin16_cuda_contiguous(cache_c, name);
    TORCH_CHECK(cache_c.device() == input.device(), name, " must be on the same device as input");
    TORCH_CHECK(cache_c.size(1) == input.size(1) && cache_c.size(2) == input.size(2) &&
                    cache_c.size(3) == input.size(3),
                name, " shape must match input channel/spatial dimensions");
    if (expected_cache_frames >= 0) {
        TORCH_CHECK(cache_c.size(0) == expected_cache_frames,
                    name, " must have ", expected_cache_frames, " frame(s)");
    }
    *cache_ptr = cache_c.template data_ptr<uint8_t>();
    *cache_frames = static_cast<int>(cache_c.size(0));
    *has_cache = true;
    return cache_c;
}

}  // namespace

torch::Tensor lightvae_fp8_conv1_tin16_direct_prepared(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    bool relu) {
    check_tin16_cuda_contiguous(input, "input");
    check_u8_cuda_tensor(weight, "weight");
    check_scale_cuda_tensor(output_scale, "output_scale");
    TORCH_CHECK(weight.dim() == 4 && weight.size(1) == 1 && weight.size(2) == 1,
                "weight must be [K,1,1,C], got ", weight.sizes());
    TORCH_CHECK(weight.size(3) == input.size(1) * kFp8ChannelsPerSlice,
                "weight C must match input channels");
    TORCH_CHECK(weight.size(0) % kFp8ChannelsPerSlice == 0,
                "weight K must be divisible by ", kFp8ChannelsPerSlice);
    TORCH_CHECK(output_scale.numel() == 1 || output_scale.numel() == weight.size(0),
                "output_scale must be scalar or have one entry per output channel");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    auto weight_c = ensure_device_dtype_contiguous(weight, input.device(), weight.scalar_type());
    auto out_s = ensure_device_dtype_contiguous(output_scale, input.device(), output_scale.scalar_type());
    torch::Tensor bias_c;
    const half* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        check_half_cuda_tensor(bias.value(), "bias");
        bias_c = ensure_device_dtype_contiguous(bias.value(), input.device(), bias.value().scalar_type());
        check_half_cuda_contiguous(bias_c, "bias");
        TORCH_CHECK(bias_c.numel() == weight_c.size(0), "bias must have one entry per output channel");
        bias_ptr = half_ptr_const(bias_c);
    }
    return out_s.scalar_type() == at::kHalf
        ? dispatch_conv1_direct<half>(input_c, weight_c, out_s, bias_ptr, relu)
        : dispatch_conv1_direct<float>(input_c, weight_c, out_s, bias_ptr, relu);
}

torch::Tensor lightvae_fp8_spatial_conv3_tin16_direct_prepared(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    int stride,
    int pad,
    bool relu,
    bool pad_right_bottom) {
    check_tin16_cuda_contiguous(input, "input");
    check_u8_cuda_tensor(weight, "weight");
    check_scale_cuda_tensor(output_scale, "output_scale");
    TORCH_CHECK(weight.dim() == 4 && weight.size(1) == 3 && weight.size(2) == 3,
                "weight must be [K,3,3,C], got ", weight.sizes());
    TORCH_CHECK(weight.size(3) == input.size(1) * kFp8ChannelsPerSlice,
                "weight C must match input channels");
    TORCH_CHECK(weight.size(0) % kFp8ChannelsPerSlice == 0,
                "weight K must be divisible by ", kFp8ChannelsPerSlice);
    TORCH_CHECK(output_scale.numel() == 1 || output_scale.numel() == weight.size(0),
                "output_scale must be scalar or have one entry per output channel");
    TORCH_CHECK(stride > 0 && pad >= 0, "invalid stride/pad");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    auto weight_c = ensure_device_dtype_contiguous(weight, input.device(), weight.scalar_type());
    auto out_s = ensure_device_dtype_contiguous(output_scale, input.device(), output_scale.scalar_type());
    torch::Tensor bias_c;
    const half* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        check_half_cuda_tensor(bias.value(), "bias");
        bias_c = ensure_device_dtype_contiguous(bias.value(), input.device(), bias.value().scalar_type());
        check_half_cuda_contiguous(bias_c, "bias");
        TORCH_CHECK(bias_c.numel() == weight_c.size(0), "bias must have one entry per output channel");
        bias_ptr = half_ptr_const(bias_c);
    }
    return out_s.scalar_type() == at::kHalf
        ? dispatch_spatial3_direct<half>(input_c, weight_c, out_s, bias_ptr, stride, pad, relu, pad_right_bottom)
        : dispatch_spatial3_direct<float>(input_c, weight_c, out_s, bias_ptr, stride, pad, relu, pad_right_bottom);
}

torch::Tensor lightvae_fp8_causal_conv3_tin16_direct_prepared(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    bool relu) {
    check_tin16_cuda_contiguous(input, "input");
    check_u8_cuda_tensor(weight, "weight");
    check_scale_cuda_tensor(output_scale, "output_scale");
    TORCH_CHECK(weight.dim() == 4 && weight.size(1) == 3 && weight.size(2) == 3,
                "weight must be [K,3,3,3*C], got ", weight.sizes());
    TORCH_CHECK(weight.size(3) == input.size(1) * kFp8ChannelsPerSlice * 3,
                "weight C must match packed causal input channels");
    TORCH_CHECK(weight.size(0) % kFp8ChannelsPerSlice == 0,
                "weight K must be divisible by ", kFp8ChannelsPerSlice);
    TORCH_CHECK(output_scale.numel() == 1 || output_scale.numel() == weight.size(0),
                "output_scale must be scalar or have one entry per output channel");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    const uint8_t* cache_ptr = nullptr;
    int cache_frames = 0;
    bool has_cache = false;
    auto cache_c = normalize_optional_cache(cache, input_c, -1, "cache", &cache_ptr, &cache_frames, &has_cache);
    TORCH_CHECK(cache_frames >= 0 && cache_frames <= 2, "causal cache must have 0, 1, or 2 frames");
    auto weight_c = ensure_device_dtype_contiguous(weight, input.device(), weight.scalar_type());
    auto out_s = ensure_device_dtype_contiguous(output_scale, input.device(), output_scale.scalar_type());
    torch::Tensor bias_c;
    const half* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        check_half_cuda_tensor(bias.value(), "bias");
        bias_c = ensure_device_dtype_contiguous(bias.value(), input.device(), bias.value().scalar_type());
        check_half_cuda_contiguous(bias_c, "bias");
        TORCH_CHECK(bias_c.numel() == weight_c.size(0), "bias must have one entry per output channel");
        bias_ptr = half_ptr_const(bias_c);
    }
    return out_s.scalar_type() == at::kHalf
        ? dispatch_causal3_direct<half>(input_c, cache_c, cache_ptr, cache_frames, weight_c, out_s, bias_ptr, relu)
        : dispatch_causal3_direct<float>(input_c, cache_c, cache_ptr, cache_frames, weight_c, out_s, bias_ptr, relu);
}

torch::Tensor lightvae_fp8_temporal_conv1_tin16_direct_prepared(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    bool relu) {
    check_tin16_cuda_contiguous(input, "input");
    check_u8_cuda_tensor(weight, "weight");
    check_scale_cuda_tensor(output_scale, "output_scale");
    TORCH_CHECK(weight.dim() == 4 && weight.size(1) == 1 && weight.size(2) == 1,
                "weight must be [K,1,1,3*C], got ", weight.sizes());
    TORCH_CHECK(weight.size(3) == input.size(1) * kFp8ChannelsPerSlice * 3,
                "weight C must match packed temporal input channels");
    TORCH_CHECK(weight.size(0) % kFp8ChannelsPerSlice == 0,
                "weight K must be divisible by ", kFp8ChannelsPerSlice);
    TORCH_CHECK(output_scale.numel() == 1 || output_scale.numel() == weight.size(0),
                "output_scale must be scalar or have one entry per output channel");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    const uint8_t* cache_ptr = nullptr;
    int cache_frames = 0;
    bool has_cache = false;
    auto cache_c = normalize_optional_cache(cache, input_c, 1, "cache", &cache_ptr, &cache_frames, &has_cache);
    auto weight_c = ensure_device_dtype_contiguous(weight, input.device(), weight.scalar_type());
    auto out_s = ensure_device_dtype_contiguous(output_scale, input.device(), output_scale.scalar_type());
    torch::Tensor bias_c;
    const half* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        check_half_cuda_tensor(bias.value(), "bias");
        bias_c = ensure_device_dtype_contiguous(bias.value(), input.device(), bias.value().scalar_type());
        check_half_cuda_contiguous(bias_c, "bias");
        TORCH_CHECK(bias_c.numel() == weight_c.size(0), "bias must have one entry per output channel");
        bias_ptr = half_ptr_const(bias_c);
    }
    return out_s.scalar_type() == at::kHalf
        ? dispatch_temporal1_direct<half>(input_c, cache_c, cache_ptr, has_cache, weight_c, out_s, bias_ptr, relu)
        : dispatch_temporal1_direct<float>(input_c, cache_c, cache_ptr, has_cache, weight_c, out_s, bias_ptr, relu);
}

}  // namespace omnidreams_singleview
