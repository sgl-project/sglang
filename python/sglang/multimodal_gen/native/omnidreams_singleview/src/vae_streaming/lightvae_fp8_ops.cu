// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "lightvae_ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>

#include <algorithm>

#include <cutlass/conv/conv2d_problem_size.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/kernel/default_conv2d_fprop_with_absmax.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_generic_with_scaling.h>
#include <cutlass/layout/tensor.h>
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

void check_half_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    check_cuda_contiguous(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, name, " must be torch.float16");
}

void check_u8_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    check_cuda_contiguous(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == at::kByte, name, " must be torch.uint8");
}

void check_tin16_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    check_u8_cuda_contiguous(tensor, name);
    TORCH_CHECK(tensor.dim() == 5 && tensor.size(4) == kFp8ChannelsPerSlice,
                name, " must be [N,C/16,H,W,16], got ", tensor.sizes());
}

void check_scale_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    check_cuda_contiguous(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kHalf,
                name, " must be torch.float32 or torch.float16");
    TORCH_CHECK(tensor.dim() == 1, name, " must be 1D");
}

const half* half_ptr_const(const torch::Tensor& tensor) {
    return reinterpret_cast<const half*>(tensor.data_ptr<at::Half>());
}

half* half_ptr(torch::Tensor& tensor) {
    return reinterpret_cast<half*>(tensor.data_ptr<at::Half>());
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
__global__ void quantize_bcthw_to_tin16_kernel(
    const half* __restrict__ input,
    uint8_t* __restrict__ output,
    const ScaleT* __restrict__ scale,
    int channels,
    int frames,
    int height,
    int width,
    int padded_channels,
    int padded_height,
    int padded_width,
    int scale_count) {
    const int slices = padded_channels / kFp8ChannelsPerSlice;
    const long long total =
        static_cast<long long>(frames) * slices * padded_height * padded_width * kFp8ChannelsPerSlice;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long rem = idx;
        const int lane = static_cast<int>(rem % kFp8ChannelsPerSlice);
        rem /= kFp8ChannelsPerSlice;
        const int x = static_cast<int>(rem % padded_width);
        rem /= padded_width;
        const int y = static_cast<int>(rem % padded_height);
        rem /= padded_height;
        const int slice = static_cast<int>(rem % slices);
        const int frame = static_cast<int>(rem / slices);
        const int c = slice * kFp8ChannelsPerSlice + lane;

        float value = 0.0f;
        if (c < channels && y < height && x < width) {
            const size_t in_idx =
                (((static_cast<size_t>(c) * frames + frame) * height + y) * width + x);
            const int scale_idx = scale_count == 1 ? 0 : c;
            const float s = fmaxf(scale_to_float(scale, scale_idx), 1.0e-12f);
            value = __half2float(input[in_idx]) / s;
        }
        output[idx] = encode_e4m3(value);
    }
}

template <typename ScaleT>
__global__ void quantize_rgb_bcthw_to_tin16_kernel(
    const half* __restrict__ input,
    uint8_t* __restrict__ output,
    const ScaleT* __restrict__ scale,
    int frames,
    int height,
    int width,
    int padded_height,
    int padded_width,
    int scale_count) {
    constexpr int padded_slices = 2;
    const long long total = static_cast<long long>(frames) * padded_height * padded_width;
    const uint4 zeros = make_uint4(0u, 0u, 0u, 0u);
    const float s0 = fmaxf(scale_to_float(scale, 0), 1.0e-12f);
    const float s1 = fmaxf(scale_to_float(scale, scale_count == 1 ? 0 : 1), 1.0e-12f);
    const float s2 = fmaxf(scale_to_float(scale, scale_count == 1 ? 0 : 2), 1.0e-12f);
    const size_t channel_stride = static_cast<size_t>(frames) * height * width;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long rem = idx;
        const int x = static_cast<int>(rem % padded_width);
        rem /= padded_width;
        const int y = static_cast<int>(rem % padded_height);
        const int frame = static_cast<int>(rem / padded_height);

        uint4 slice0 = zeros;
        if (y < height && x < width) {
            const size_t frame_offset = (static_cast<size_t>(frame) * height + y) * width + x;
            const uint32_t q0 = static_cast<uint32_t>(encode_e4m3(__half2float(input[frame_offset]) / s0));
            const uint32_t q1 = static_cast<uint32_t>(
                encode_e4m3(__half2float(input[channel_stride + frame_offset]) / s1));
            const uint32_t q2 = static_cast<uint32_t>(
                encode_e4m3(__half2float(input[channel_stride * 2 + frame_offset]) / s2));
            slice0.x = q0 | (q1 << 8) | (q2 << 16);
        }

        const size_t base0 =
            (((static_cast<size_t>(frame) * padded_slices) * padded_height + y) * padded_width + x) *
            kFp8ChannelsPerSlice;
        const size_t base1 =
            (((static_cast<size_t>(frame) * padded_slices + 1) * padded_height + y) * padded_width + x) *
            kFp8ChannelsPerSlice;
        reinterpret_cast<uint4*>(output + base0)[0] = slice0;
        reinterpret_cast<uint4*>(output + base1)[0] = zeros;
    }
}

template <typename ScaleT>
__global__ void dequantize_tin16_to_bcthw_kernel(
    const uint8_t* __restrict__ input,
    half* __restrict__ output,
    const ScaleT* __restrict__ scale,
    int real_channels,
    int frames,
    int padded_channels,
    int padded_height,
    int padded_width,
    int real_height,
    int real_width,
    int scale_count) {
    const int slices = padded_channels / kFp8ChannelsPerSlice;
    const long long total = static_cast<long long>(real_channels) * frames * real_height * real_width;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long rem = idx;
        const int x = static_cast<int>(rem % real_width);
        rem /= real_width;
        const int y = static_cast<int>(rem % real_height);
        rem /= real_height;
        const int frame = static_cast<int>(rem % frames);
        const int c = static_cast<int>(rem / frames);
        const int slice = c / kFp8ChannelsPerSlice;
        const int lane = c - slice * kFp8ChannelsPerSlice;
        const size_t in_idx =
            (((static_cast<size_t>(frame) * slices + slice) * padded_height + y) * padded_width + x) *
                kFp8ChannelsPerSlice + lane;
        const int scale_idx = scale_count == 1 ? 0 : c;
        output[idx] = __float2half_rn(decode_e4m3(input[in_idx]) * scale_to_float(scale, scale_idx));
    }
}

template <int Channels, typename InScaleT, typename OutScaleT>
__global__ void rmsnorm_tin16_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const InScaleT* __restrict__ input_scale,
    const half* __restrict__ gamma,
    const OutScaleT* __restrict__ output_scale,
    int frames,
    int height,
    int width,
    int real_channels,
    int input_scale_count,
    int output_scale_count,
    bool silu) {
    constexpr int warps_per_block = (Channels <= 96) ? 4 : 2;
    constexpr int slices = Channels / kFp8ChannelsPerSlice;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int spatial_idx = blockIdx.x * warps_per_block + warp;
    const int frame = blockIdx.y;
    const int spatial_count = height * width;
    if (spatial_idx >= spatial_count || frame >= frames) {
        return;
    }

    const int y = spatial_idx / width;
    const int x = spatial_idx - y * width;
    float vals[(Channels + 31) / 32];
    int count = 0;
    float sum_sq = 0.0f;

    for (int c = lane; c < Channels; c += 32) {
        const int slice = c / kFp8ChannelsPerSlice;
        const int sub = c - slice * kFp8ChannelsPerSlice;
        const size_t idx =
            (((static_cast<size_t>(frame) * slices + slice) * height + y) * width + x) *
                kFp8ChannelsPerSlice + sub;
        float v = 0.0f;
        if (c < real_channels) {
            const int scale_idx = input_scale_count == 1 ? 0 : c;
            v = decode_e4m3(input[idx]) * scale_to_float(input_scale, scale_idx);
            sum_sq += v * v;
        }
        vals[count++] = v;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset);
    }
    const float total_sum_sq = __shfl_sync(0xffffffffu, sum_sq, 0);
    const float inv_rms = rsqrtf(total_sum_sq / static_cast<float>(real_channels) + 1.0e-8f);

    int store_idx = 0;
    for (int c = lane; c < Channels; c += 32) {
        const int slice = c / kFp8ChannelsPerSlice;
        const int sub = c - slice * kFp8ChannelsPerSlice;
        const size_t idx =
            (((static_cast<size_t>(frame) * slices + slice) * height + y) * width + x) *
                kFp8ChannelsPerSlice + sub;
        float result = 0.0f;
        if (c < real_channels) {
            result = vals[store_idx] * inv_rms * __half2float(gamma[c]);
            if (silu) {
                result = result / (1.0f + expf(-result));
            }
        }
        const int scale_idx = output_scale_count == 1 ? 0 : c;
        const float out_s = fmaxf(scale_to_float(output_scale, scale_idx), 1.0e-12f);
        output[idx] = encode_e4m3(result / out_s);
        ++store_idx;
    }
}

template <typename AScaleT, typename BScaleT, typename OutScaleT>
__global__ void add_tin16_pack_kernel(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    uint8_t* __restrict__ output,
    const AScaleT* __restrict__ a_scale,
    const BScaleT* __restrict__ b_scale,
    const OutScaleT* __restrict__ output_scale,
    long long count,
    int slices,
    int height,
    int width,
    int real_channels,
    int a_scale_count,
    int b_scale_count,
    int output_scale_count,
    bool relu) {
    const long long pack_count = count / kFp8ChannelsPerSlice;
    for (long long pack_idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         pack_idx < pack_count;
         pack_idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long rem = pack_idx;
        rem /= width;
        rem /= height;
        const int slice = static_cast<int>(rem % slices);
        const int channel_base = slice * kFp8ChannelsPerSlice;
        const size_t base = static_cast<size_t>(pack_idx) * kFp8ChannelsPerSlice;
        const uint4 a_pack = reinterpret_cast<const uint4*>(a + base)[0];
        const uint4 b_pack = reinterpret_cast<const uint4*>(b + base)[0];
        const uint32_t* a_words = reinterpret_cast<const uint32_t*>(&a_pack);
        const uint32_t* b_words = reinterpret_cast<const uint32_t*>(&b_pack);
        uint4 out_pack = make_uint4(0u, 0u, 0u, 0u);
        uint32_t* out_words = reinterpret_cast<uint32_t*>(&out_pack);
        #pragma unroll
        for (int group = 0; group < 4; ++group) {
            uint32_t word = 0u;
            const uint32_t aw = a_words[group];
            const uint32_t bw = b_words[group];
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                const int lane = group * 4 + sub;
                const int channel = channel_base + lane;
                uint8_t out = encode_e4m3(0.0f);
                if (real_channels < 0 || channel < real_channels) {
                    const int a_idx = a_scale_count == 1 ? 0 : channel;
                    const int b_idx = b_scale_count == 1 ? 0 : channel;
                    const int out_idx = output_scale_count == 1 ? 0 : channel;
                    float result = decode_e4m3(static_cast<uint8_t>((aw >> (sub * 8)) & 0xffu)) *
                                       scale_to_float(a_scale, a_idx) +
                                   decode_e4m3(static_cast<uint8_t>((bw >> (sub * 8)) & 0xffu)) *
                                       scale_to_float(b_scale, b_idx);
                    if (relu) {
                        result = fmaxf(result, 0.0f);
                    }
                    const float out_s = fmaxf(scale_to_float(output_scale, out_idx), 1.0e-12f);
                    out = encode_e4m3(result / out_s);
                }
                word |= static_cast<uint32_t>(out) << (sub * 8);
            }
            out_words[group] = word;
        }
        reinterpret_cast<uint4*>(output + base)[0] = out_pack;
    }
}

template <typename AScaleT, typename BScaleT, typename OutScaleT>
__global__ void add_relu_nhwc_tin16_to_tin16_kernel(
    const uint8_t* __restrict__ a_nhwc,
    const uint8_t* __restrict__ b_tin16,
    uint8_t* __restrict__ output,
    const AScaleT* __restrict__ a_scale,
    const BScaleT* __restrict__ b_scale,
    const OutScaleT* __restrict__ output_scale,
    long long count,
    int slices,
    int height,
    int width,
    int real_channels,
    int a_scale_count,
    int b_scale_count,
    int output_scale_count) {
    const int channels = slices * kFp8ChannelsPerSlice;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long rem = idx;
        const int lane = static_cast<int>(rem % kFp8ChannelsPerSlice);
        rem /= kFp8ChannelsPerSlice;
        const int x = static_cast<int>(rem % width);
        rem /= width;
        const int y = static_cast<int>(rem % height);
        rem /= height;
        const int slice = static_cast<int>(rem % slices);
        rem /= slices;
        const int frame = static_cast<int>(rem);
        const int c = slice * kFp8ChannelsPerSlice + lane;
        float result = 0.0f;
        if (real_channels < 0 || c < real_channels) {
            const int a_idx = a_scale_count == 1 ? 0 : c;
            const int b_idx = b_scale_count == 1 ? 0 : c;
            const size_t a_offset =
                (((static_cast<size_t>(frame) * height + y) * width + x) * channels + c);
            result = decode_e4m3(a_nhwc[a_offset]) * scale_to_float(a_scale, a_idx) +
                     decode_e4m3(b_tin16[idx]) * scale_to_float(b_scale, b_idx);
            result = fmaxf(result, 0.0f);
        }
        const int out_idx = output_scale_count == 1 ? 0 : c;
        const float out_s = fmaxf(scale_to_float(output_scale, out_idx), 1.0e-12f);
        output[idx] = encode_e4m3(result / out_s);
    }
}

__global__ void upsample2x_tin16_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    long long total,
    int frames,
    int slices,
    int height,
    int width) {
    const int out_h = height * 2;
    const int out_w = width * 2;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long rem = idx;
        const int lane = static_cast<int>(rem % kFp8ChannelsPerSlice);
        rem /= kFp8ChannelsPerSlice;
        const int ox = static_cast<int>(rem % out_w);
        rem /= out_w;
        const int oy = static_cast<int>(rem % out_h);
        rem /= out_h;
        const int slice = static_cast<int>(rem % slices);
        rem /= slices;
        const int frame = static_cast<int>(rem);
        const int iy = oy >> 1;
        const int ix = ox >> 1;
        const size_t in_idx =
            (((static_cast<size_t>(frame) * slices + slice) * height + iy) * width + ix) *
                kFp8ChannelsPerSlice + lane;
        output[idx] = input[in_idx];
    }
}

template <typename ScaleT>
__global__ void extract_mu_normalize_kernel(
    const uint8_t* __restrict__ input,
    half* __restrict__ output,
    const ScaleT* __restrict__ input_scale,
    const half* __restrict__ mean,
    const half* __restrict__ inv_std,
    int frames,
    int padded_height,
    int padded_width,
    int real_height,
    int real_width,
    int scale_count) {
    constexpr int channels = 16;
    constexpr int slices = 32 / kFp8ChannelsPerSlice;
    const long long total = static_cast<long long>(channels) * frames * real_height * real_width;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long rem = idx;
        const int x = static_cast<int>(rem % real_width);
        rem /= real_width;
        const int y = static_cast<int>(rem % real_height);
        rem /= real_height;
        const int frame = static_cast<int>(rem % frames);
        const int c = static_cast<int>(rem / frames);
        const int slice = c / kFp8ChannelsPerSlice;
        const int lane = c - slice * kFp8ChannelsPerSlice;
        const size_t in_idx =
            (((static_cast<size_t>(frame) * slices + slice) * padded_height + y) * padded_width + x) *
                kFp8ChannelsPerSlice + lane;
        const int scale_idx = scale_count == 1 ? 0 : c;
        const float mu = decode_e4m3(input[in_idx]) * scale_to_float(input_scale, scale_idx);
        const float out = (mu - __half2float(mean[c])) * __half2float(inv_std[c]);
        output[idx] = __float2half_rn(out);
    }
}

template <typename InScaleT, typename OutScaleT>
__global__ void prepare_conv2d_weight_krsc_kernel(
    const half* __restrict__ weight,
    uint8_t* __restrict__ output,
    const InScaleT* __restrict__ input_scale,
    const OutScaleT* __restrict__ output_scale,
    long long count,
    int channels,
    int output_channels,
    int r_size,
    int s_size,
    int input_scale_count,
    int output_scale_count) {
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        const int c = static_cast<int>(idx % channels);
        long long tmp = idx / channels;
        tmp /= s_size;
        tmp /= r_size;
        const int k = static_cast<int>(tmp % output_channels);
        const int in_idx = input_scale_count == 1 ? 0 : c;
        const int out_idx = output_scale_count == 1 ? 0 : k;
        const float in_s = scale_to_float(input_scale, in_idx);
        const float out_s = fmaxf(scale_to_float(output_scale, out_idx), 1.0e-12f);
        output[idx] = encode_e4m3(__half2float(weight[idx]) * in_s / out_s);
    }
}

template <typename OutScaleT>
__global__ void add_bias_act_fp8_nhwc_kernel(
    uint8_t* __restrict__ output,
    const half* __restrict__ bias,
    const OutScaleT* __restrict__ output_scale,
    long long count,
    int channels,
    int output_scale_count,
    bool relu) {
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        const int c = static_cast<int>(idx % channels);
        const int scale_idx = output_scale_count == 1 ? 0 : c;
        const float out_s = fmaxf(scale_to_float(output_scale, scale_idx), 1.0e-12f);
        float value = decode_e4m3(output[idx]) + __half2float(bias[c]) / out_s;
        if (relu) {
            value = fmaxf(value, 0.0f);
        }
        output[idx] = encode_e4m3(value);
    }
}

__global__ void pack_causal3_tin16_kernel(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ cache,
    uint8_t* __restrict__ output,
    int frames,
    int cache_frames,
    int slices,
    int height,
    int width) {
    const int out_slices = slices * 3;
    const long long total =
        static_cast<long long>(frames) * out_slices * height * width * kFp8ChannelsPerSlice;
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
        const int dt = out_slice / slices;
        const int slice = out_slice - dt * slices;
        const int src_ext_t = cache_frames + frame + dt - 2;
        uint8_t value = encode_e4m3(0.0f);
        if (src_ext_t >= 0 && src_ext_t < cache_frames && cache != nullptr) {
            const size_t src =
                (((static_cast<size_t>(src_ext_t) * slices + slice) * height + y) * width + x) *
                    kFp8ChannelsPerSlice + lane;
            value = cache[src];
        } else {
            const int src_t = src_ext_t - cache_frames;
            if (src_t >= 0 && src_t < frames) {
                const size_t src =
                    (((static_cast<size_t>(src_t) * slices + slice) * height + y) * width + x) *
                        kFp8ChannelsPerSlice + lane;
                value = input[src];
            }
        }
        output[idx] = value;
    }
}

__global__ void update_tail_cache_tin16_kernel(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ cache,
    uint8_t* __restrict__ output,
    int frames,
    int old_cache_frames,
    int new_cache_frames,
    int slices,
    int height,
    int width) {
    const long long total =
        static_cast<long long>(new_cache_frames) * slices * height * width * kFp8ChannelsPerSlice;
    const int tail_start = old_cache_frames + frames - new_cache_frames;
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
        const int slice = static_cast<int>(rem % slices);
        rem /= slices;
        const int ct = static_cast<int>(rem);
        const int src_ext_t = tail_start + ct;
        uint8_t value = encode_e4m3(0.0f);
        if (src_ext_t >= 0 && src_ext_t < old_cache_frames && cache != nullptr) {
            const size_t src =
                (((static_cast<size_t>(src_ext_t) * slices + slice) * height + y) * width + x) *
                    kFp8ChannelsPerSlice + lane;
            value = cache[src];
        } else {
            const int src_t = src_ext_t - old_cache_frames;
            if (src_t >= 0 && src_t < frames) {
                const size_t src =
                    (((static_cast<size_t>(src_t) * slices + slice) * height + y) * width + x) *
                        kFp8ChannelsPerSlice + lane;
                value = input[src];
            }
        }
        output[idx] = value;
    }
}

__global__ void spatial_pad_right_bottom_tin16_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int frames,
    int slices,
    int height,
    int width) {
    const int out_h = height + 1;
    const int out_w = width + 1;
    const long long total =
        static_cast<long long>(frames) * slices * out_h * out_w * kFp8ChannelsPerSlice;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long rem = idx;
        const int lane = static_cast<int>(rem % kFp8ChannelsPerSlice);
        rem /= kFp8ChannelsPerSlice;
        const int x = static_cast<int>(rem % out_w);
        rem /= out_w;
        const int y = static_cast<int>(rem % out_h);
        rem /= out_h;
        const int slice = static_cast<int>(rem % slices);
        rem /= slices;
        const int frame = static_cast<int>(rem);
        uint8_t value = encode_e4m3(0.0f);
        if (y < height && x < width) {
            const size_t src =
                (((static_cast<size_t>(frame) * slices + slice) * height + y) * width + x) *
                    kFp8ChannelsPerSlice + lane;
            value = input[src];
        }
        output[idx] = value;
    }
}

__global__ void pack_temporal3_tin16_kernel(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ cache,
    uint8_t* __restrict__ output,
    int frames,
    int t_out,
    int slices,
    int height,
    int width,
    bool has_cache) {
    const int out_slices = slices * 3;
    const long long total =
        static_cast<long long>(t_out) * out_slices * height * width * kFp8ChannelsPerSlice;
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
        const int dt = out_slice / slices;
        const int slice = out_slice - dt * slices;
        const int src_ext_t = to * 2 + dt;
        uint8_t value = encode_e4m3(0.0f);
        if (has_cache && src_ext_t == 0 && cache != nullptr) {
            const size_t src =
                ((static_cast<size_t>(slice) * height + y) * width + x) * kFp8ChannelsPerSlice + lane;
            value = cache[src];
        } else {
            const int src_t = has_cache ? src_ext_t - 1 : src_ext_t;
            if (src_t >= 0 && src_t < frames) {
                const size_t src =
                    (((static_cast<size_t>(src_t) * slices + slice) * height + y) * width + x) *
                        kFp8ChannelsPerSlice + lane;
                value = input[src];
            }
        }
        output[idx] = value;
    }
}

__global__ void tin16_to_nhwc_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int frames,
    int slices,
    int height,
    int width) {
    const int channels = slices * kFp8ChannelsPerSlice;
    const long long total = static_cast<long long>(frames) * height * width * channels;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long rem = idx;
        const int c = static_cast<int>(rem % channels);
        rem /= channels;
        const int x = static_cast<int>(rem % width);
        rem /= width;
        const int y = static_cast<int>(rem % height);
        rem /= height;
        const int frame = static_cast<int>(rem);
        const int slice = c / kFp8ChannelsPerSlice;
        const int lane = c - slice * kFp8ChannelsPerSlice;
        const size_t src =
            (((static_cast<size_t>(frame) * slices + slice) * height + y) * width + x) *
                kFp8ChannelsPerSlice + lane;
        output[idx] = input[src];
    }
}

__global__ void nhwc_to_tin16_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int frames,
    int slices,
    int height,
    int width) {
    const int channels = slices * kFp8ChannelsPerSlice;
    const long long total =
        static_cast<long long>(frames) * slices * height * width * kFp8ChannelsPerSlice;
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
        const int slice = static_cast<int>(rem % slices);
        rem /= slices;
        const int frame = static_cast<int>(rem);
        const int c = slice * kFp8ChannelsPerSlice + lane;
        const size_t src = (((static_cast<size_t>(frame) * height + y) * width + x) * channels + c);
        output[idx] = input[src];
    }
}

template <typename SrcScaleT, typename OutScaleT>
__global__ void nhwc_rescale_to_tin16_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const SrcScaleT* __restrict__ source_scale,
    const OutScaleT* __restrict__ output_scale,
    int frames,
    int slices,
    int height,
    int width,
    int source_scale_count,
    int output_scale_count) {
    const int channels = slices * kFp8ChannelsPerSlice;
    const long long total =
        static_cast<long long>(frames) * slices * height * width * kFp8ChannelsPerSlice;
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
        const int slice = static_cast<int>(rem % slices);
        rem /= slices;
        const int frame = static_cast<int>(rem);
        const int c = slice * kFp8ChannelsPerSlice + lane;
        const size_t src = (((static_cast<size_t>(frame) * height + y) * width + x) * channels + c);
        const int src_scale_idx = source_scale_count == 1 ? 0 : c;
        const int out_scale_idx = output_scale_count == 1 ? 0 : c;
        const float src_s = scale_to_float(source_scale, src_scale_idx);
        const float out_s = fmaxf(scale_to_float(output_scale, out_scale_idx), 1.0e-12f);
        output[idx] = encode_e4m3(decode_e4m3(input[src]) * src_s / out_s);
    }
}

__global__ void pack_causal3_tin16_to_nhwc_kernel(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ cache,
    uint8_t* __restrict__ output,
    int frames,
    int cache_frames,
    int slices,
    int height,
    int width) {
    const int channels = slices * 3 * kFp8ChannelsPerSlice;
    const long long total = static_cast<long long>(frames) * height * width * channels;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long rem = idx;
        const int c = static_cast<int>(rem % channels);
        rem /= channels;
        const int x = static_cast<int>(rem % width);
        rem /= width;
        const int y = static_cast<int>(rem % height);
        rem /= height;
        const int frame = static_cast<int>(rem);
        const int out_slice = c / kFp8ChannelsPerSlice;
        const int lane = c - out_slice * kFp8ChannelsPerSlice;
        const int dt = out_slice / slices;
        const int slice = out_slice - dt * slices;
        const int src_ext_t = cache_frames + frame + dt - 2;
        uint8_t value = encode_e4m3(0.0f);
        if (src_ext_t >= 0 && src_ext_t < cache_frames && cache != nullptr) {
            const size_t src =
                (((static_cast<size_t>(src_ext_t) * slices + slice) * height + y) * width + x) *
                    kFp8ChannelsPerSlice + lane;
            value = cache[src];
        } else {
            const int src_t = src_ext_t - cache_frames;
            if (src_t >= 0 && src_t < frames) {
                const size_t src =
                    (((static_cast<size_t>(src_t) * slices + slice) * height + y) * width + x) *
                        kFp8ChannelsPerSlice + lane;
                value = input[src];
            }
        }
        output[idx] = value;
    }
}

__global__ void pack_temporal3_tin16_to_nhwc_kernel(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ cache,
    uint8_t* __restrict__ output,
    int frames,
    int t_out,
    int slices,
    int height,
    int width,
    bool has_cache) {
    const int channels = slices * 3 * kFp8ChannelsPerSlice;
    const long long total = static_cast<long long>(t_out) * height * width * channels;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long rem = idx;
        const int c = static_cast<int>(rem % channels);
        rem /= channels;
        const int x = static_cast<int>(rem % width);
        rem /= width;
        const int y = static_cast<int>(rem % height);
        rem /= height;
        const int to = static_cast<int>(rem);
        const int out_slice = c / kFp8ChannelsPerSlice;
        const int lane = c - out_slice * kFp8ChannelsPerSlice;
        const int dt = out_slice / slices;
        const int slice = out_slice - dt * slices;
        const int src_ext_t = to * 2 + dt;
        uint8_t value = encode_e4m3(0.0f);
        if (has_cache && src_ext_t == 0 && cache != nullptr) {
            const size_t src =
                ((static_cast<size_t>(slice) * height + y) * width + x) * kFp8ChannelsPerSlice + lane;
            value = cache[src];
        } else {
            const int src_t = has_cache ? src_ext_t - 1 : src_ext_t;
            if (src_t >= 0 && src_t < frames) {
                const size_t src =
                    (((static_cast<size_t>(src_t) * slices + slice) * height + y) * width + x) *
                        kFp8ChannelsPerSlice + lane;
                value = input[src];
            }
        }
        output[idx] = value;
    }
}

template <typename Kernel>
cudaError_t run_fp8_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor zeros,
    torch::Tensor ones,
    torch::Tensor amax,
    int stride,
    int pad,
    cudaStream_t stream) {
    using Element = cutlass::float_e4m3_t;
    using Conv2dOp = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;

    const int n = static_cast<int>(input.size(0));
    const int h = static_cast<int>(input.size(1));
    const int w = static_cast<int>(input.size(2));
    const int c = static_cast<int>(input.size(3));
    const int k = static_cast<int>(weight.size(0));
    const int r = static_cast<int>(weight.size(1));
    const int s = static_cast<int>(weight.size(2));
    const int p = static_cast<int>(output.size(1));
    const int q = static_cast<int>(output.size(2));

    cutlass::conv::Conv2dProblemSize problem(
        n, h, w, c, k, r, s, p, q,
        pad, pad, stride, stride, 1, 1, cutlass::conv::Mode::kCrossCorrelation);
    auto layout_a = cutlass::layout::TensorNHWC::packed({problem.N, problem.H, problem.W, problem.C});
    auto layout_b = cutlass::layout::TensorNHWC::packed({problem.K, problem.R, problem.S, problem.C});
    auto layout_c = cutlass::layout::TensorNHWC::packed({problem.N, problem.P, problem.Q, problem.K});
    typename Kernel::TensorRefA ref_a(reinterpret_cast<Element*>(input.template data_ptr<uint8_t>()), layout_a);
    typename Kernel::TensorRefB ref_b(reinterpret_cast<Element*>(weight.template data_ptr<uint8_t>()), layout_b);
    typename Kernel::TensorRefC ref_c(reinterpret_cast<Element*>(zeros.template data_ptr<uint8_t>()), layout_c);
    typename Kernel::TensorRefC ref_d(reinterpret_cast<Element*>(output.template data_ptr<uint8_t>()), layout_c);
    typename Kernel::Epilogue::OutputOp::Params::ActivationParams activation_params(1.0f, 0.0f);
    typename Kernel::Epilogue::OutputOp::Params epilogue(
        activation_params,
        ones.data_ptr<float>(),
        ones.data_ptr<float>(),
        ones.data_ptr<float>(),
        ones.data_ptr<float>(),
        ones.data_ptr<float>(),
        amax.data_ptr<float>(),
        amax.data_ptr<float>());

    typename Conv2dOp::Arguments args(
        problem,
        ref_a,
        ref_b,
        ref_c,
        ref_d,
        ref_d,
        epilogue,
        cutlass::conv::SplitKMode::kSerial,
        reinterpret_cast<Element*>(zeros.template data_ptr<uint8_t>()),
        0);
    Conv2dOp op;
    cutlass::Status status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorNotSupported;
    }
    size_t workspace_size = Conv2dOp::get_workspace_size(args);
    torch::Tensor workspace = torch::empty(
        {static_cast<long long>(workspace_size)},
        input.options().dtype(torch::kUInt8));
    status = op.initialize(args, workspace.template data_ptr<uint8_t>(), stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInitializationError;
    }
    status = op(stream);
    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

template <template <typename T> class ActivationFunctor>
cudaError_t cutlass_conv2d_fp8_impl(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int stride,
    int pad,
    cudaStream_t stream) {
    using Element = cutlass::float_e4m3_t;
    using ElementAccumulator = float;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
        ActivationFunctor,
        Element,
        Element,
        128 / cutlass::sizeof_bits<Element>::value,
        ElementAccumulator,
        ElementAccumulator>;
    using Conv2dKernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithAbsMax<
        Element, cutlass::layout::TensorNHWC,
        Element, cutlass::layout::TensorNHWC,
        Element, cutlass::layout::TensorNHWC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm89,
        cutlass::gemm::GemmShape<128, 256, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3,
        cutlass::arch::OpMultiplyAddFastAccum,
        cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;

    const auto u8_opts = input.options().dtype(torch::kUInt8);
    torch::Tensor zeros = torch::zeros({std::max<int64_t>(output.numel(), weight.size(0))}, u8_opts);
    torch::Tensor ones = torch::ones({1}, input.options().dtype(torch::kFloat32));
    torch::Tensor amax = torch::zeros({1}, input.options().dtype(torch::kFloat32));
    return run_fp8_conv2d<Conv2dKernel>(input, weight, output, zeros, ones, amax, stride, pad, stream);
}

cudaError_t cutlass_conv2d_fp8_plain_impl(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int stride,
    int pad,
    cudaStream_t stream) {
    using Element = cutlass::float_e4m3_t;
    using ElementAccumulator = float;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        Element,
        128 / cutlass::sizeof_bits<Element>::value,
        ElementAccumulator,
        ElementAccumulator>;
    using Conv2dKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        Element, cutlass::layout::TensorNHWC,
        Element, cutlass::layout::TensorNHWC,
        Element, cutlass::layout::TensorNHWC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm89,
        cutlass::gemm::GemmShape<128, 256, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3,
        cutlass::arch::OpMultiplyAddFastAccum,
        cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
    using Conv2dOp = cutlass::conv::device::ImplicitGemmConvolution<Conv2dKernel>;

    const int n = static_cast<int>(input.size(0));
    const int h = static_cast<int>(input.size(1));
    const int w = static_cast<int>(input.size(2));
    const int c = static_cast<int>(input.size(3));
    const int k = static_cast<int>(weight.size(0));
    const int r = static_cast<int>(weight.size(1));
    const int s = static_cast<int>(weight.size(2));
    const int p = static_cast<int>(output.size(1));
    const int q = static_cast<int>(output.size(2));

    cutlass::conv::Conv2dProblemSize problem(
        n, h, w, c, k, r, s, p, q,
        pad, pad, stride, stride, 1, 1, cutlass::conv::Mode::kCrossCorrelation);
    auto layout_a = cutlass::layout::TensorNHWC::packed({problem.N, problem.H, problem.W, problem.C});
    auto layout_b = cutlass::layout::TensorNHWC::packed({problem.K, problem.R, problem.S, problem.C});
    auto layout_c = cutlass::layout::TensorNHWC::packed({problem.N, problem.P, problem.Q, problem.K});
    typename Conv2dKernel::TensorRefA ref_a(reinterpret_cast<Element*>(input.template data_ptr<uint8_t>()), layout_a);
    typename Conv2dKernel::TensorRefB ref_b(reinterpret_cast<Element*>(weight.template data_ptr<uint8_t>()), layout_b);
    typename Conv2dKernel::TensorRefC ref_c(reinterpret_cast<Element*>(output.template data_ptr<uint8_t>()), layout_c);
    typename Conv2dKernel::TensorRefC ref_d(reinterpret_cast<Element*>(output.template data_ptr<uint8_t>()), layout_c);
    typename Conv2dKernel::Epilogue::OutputOp::Params epilogue(
        ElementAccumulator(1.0f),
        ElementAccumulator(0.0f));

    typename Conv2dOp::Arguments args(
        problem,
        ref_a,
        ref_b,
        ref_c,
        ref_d,
        epilogue,
        cutlass::conv::SplitKMode::kSerial);
    Conv2dOp op;
    cutlass::Status status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorNotSupported;
    }
    size_t workspace_size = Conv2dOp::get_workspace_size(args);
    torch::Tensor workspace;
    void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        workspace = torch::empty(
            {static_cast<long long>(workspace_size)},
            input.options().dtype(torch::kUInt8));
        workspace_ptr = workspace.template data_ptr<uint8_t>();
    }
    status = op.initialize(args, workspace_ptr, stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInitializationError;
    }
    status = op(stream);
    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

template <typename ScaleT>
torch::Tensor dispatch_quantize_bcthw_to_tin16(
    torch::Tensor input,
    torch::Tensor scale,
    int padded_height,
    int padded_width,
    int padded_channels) {
    const int channels = static_cast<int>(input.size(1));
    const int frames = static_cast<int>(input.size(2));
    const int height = static_cast<int>(input.size(3));
    const int width = static_cast<int>(input.size(4));
    auto output = torch::empty(
        {frames, padded_channels / kFp8ChannelsPerSlice, padded_height, padded_width, kFp8ChannelsPerSlice},
        input.options().dtype(torch::kUInt8));
    if (channels == 3 && padded_channels == 32) {
        const long long pixels = static_cast<long long>(frames) * padded_height * padded_width;
        quantize_rgb_bcthw_to_tin16_kernel<ScaleT><<<
            launch_blocks(pixels), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            half_ptr_const(input),
            output.template data_ptr<uint8_t>(),
            reinterpret_cast<const ScaleT*>(scale.data_ptr()),
            frames,
            height,
            width,
            padded_height,
            padded_width,
            static_cast<int>(scale.numel()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    }
    const long long total = output.numel();
    quantize_bcthw_to_tin16_kernel<ScaleT><<<launch_blocks(total), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        half_ptr_const(input),
        output.template data_ptr<uint8_t>(),
        reinterpret_cast<const ScaleT*>(scale.data_ptr()),
        channels,
        frames,
        height,
        width,
        padded_channels,
        padded_height,
        padded_width,
        static_cast<int>(scale.numel()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <typename ScaleT>
torch::Tensor dispatch_dequantize_tin16_to_bcthw(
    torch::Tensor input,
    torch::Tensor scale,
    int real_channels,
    int real_height,
    int real_width) {
    const int frames = static_cast<int>(input.size(0));
    const int padded_channels = static_cast<int>(input.size(1) * input.size(4));
    const int padded_height = static_cast<int>(input.size(2));
    const int padded_width = static_cast<int>(input.size(3));
    auto output = torch::empty(
        {1, real_channels, frames, real_height, real_width},
        input.options().dtype(torch::kFloat16));
    const long long total = output.numel();
    dequantize_tin16_to_bcthw_kernel<ScaleT><<<launch_blocks(total), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.template data_ptr<uint8_t>(),
        half_ptr(output),
        reinterpret_cast<const ScaleT*>(scale.data_ptr()),
        real_channels,
        frames,
        padded_channels,
        padded_height,
        padded_width,
        real_height,
        real_width,
        static_cast<int>(scale.numel()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <int Channels, typename InScaleT, typename OutScaleT>
torch::Tensor dispatch_rmsnorm_tin16(
    torch::Tensor input,
    torch::Tensor input_scale,
    torch::Tensor gamma,
    torch::Tensor output_scale,
    int real_channels,
    bool silu) {
    auto output = torch::empty_like(input);
    constexpr int warps_per_block = (Channels <= 96) ? 4 : 2;
    const int frames = static_cast<int>(input.size(0));
    const int height = static_cast<int>(input.size(2));
    const int width = static_cast<int>(input.size(3));
    const int spatial_count = height * width;
    rmsnorm_tin16_kernel<Channels, InScaleT, OutScaleT><<<
        dim3((spatial_count + warps_per_block - 1) / warps_per_block, frames),
        warps_per_block * 32,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        input.template data_ptr<uint8_t>(),
        output.template data_ptr<uint8_t>(),
        reinterpret_cast<const InScaleT*>(input_scale.data_ptr()),
        half_ptr_const(gamma),
        reinterpret_cast<const OutScaleT*>(output_scale.data_ptr()),
        frames,
        height,
        width,
        real_channels,
        static_cast<int>(input_scale.numel()),
        static_cast<int>(output_scale.numel()),
        silu);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <typename AScaleT, typename BScaleT, typename OutScaleT>
torch::Tensor dispatch_add_tin16(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor output_scale,
    int real_channels,
    bool relu) {
    auto output = torch::empty_like(a);
    const int slices = static_cast<int>(a.size(1));
    const int height = static_cast<int>(a.size(2));
    const int width = static_cast<int>(a.size(3));
    const long long total = a.numel();
    add_tin16_pack_kernel<AScaleT, BScaleT, OutScaleT><<<
        launch_blocks((total + kFp8ChannelsPerSlice - 1) / kFp8ChannelsPerSlice),
        256,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        a.template data_ptr<uint8_t>(),
        b.template data_ptr<uint8_t>(),
        output.template data_ptr<uint8_t>(),
        reinterpret_cast<const AScaleT*>(a_scale.data_ptr()),
        reinterpret_cast<const BScaleT*>(b_scale.data_ptr()),
        reinterpret_cast<const OutScaleT*>(output_scale.data_ptr()),
        total,
        slices,
        height,
        width,
        real_channels,
        static_cast<int>(a_scale.numel()),
        static_cast<int>(b_scale.numel()),
        static_cast<int>(output_scale.numel()),
        relu);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <typename ScaleT>
torch::Tensor dispatch_extract_mu_normalize_tin16(
    torch::Tensor input,
    torch::Tensor input_scale,
    torch::Tensor mean,
    torch::Tensor inv_std,
    int real_height,
    int real_width) {
    const int frames = static_cast<int>(input.size(0));
    const int padded_height = static_cast<int>(input.size(2));
    const int padded_width = static_cast<int>(input.size(3));
    auto output = torch::empty({1, 16, frames, real_height, real_width}, input.options().dtype(torch::kFloat16));
    const long long total = output.numel();
    extract_mu_normalize_kernel<ScaleT><<<launch_blocks(total), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.template data_ptr<uint8_t>(),
        half_ptr(output),
        reinterpret_cast<const ScaleT*>(input_scale.data_ptr()),
        half_ptr_const(mean),
        half_ptr_const(inv_std),
        frames,
        padded_height,
        padded_width,
        real_height,
        real_width,
        static_cast<int>(input_scale.numel()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace

torch::Tensor lightvae_fp8_quantize_bcthw_to_tin16(
    torch::Tensor input,
    torch::Tensor scale,
    int padded_height,
    int padded_width,
    int padded_channels) {
    check_half_cuda_contiguous(input, "input");
    check_scale_cuda_contiguous(scale, "scale");
    TORCH_CHECK(input.dim() == 5 && input.size(0) == 1, "input must be [1,C,T,H,W], got ", input.sizes());
    TORCH_CHECK(padded_channels % kFp8ChannelsPerSlice == 0,
                "padded_channels must be divisible by ", kFp8ChannelsPerSlice);
    TORCH_CHECK(input.size(1) <= padded_channels, "padded_channels is smaller than input channels");
    TORCH_CHECK(padded_height >= input.size(3) && padded_width >= input.size(4),
                "padded spatial shape must cover input shape");
    TORCH_CHECK(scale.numel() == 1 || scale.numel() == input.size(1),
                "scale must be scalar or have one entry per real input channel");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    auto scale_c = scale.to(input.device(), scale.scalar_type(), false, true).contiguous();
    if (scale_c.scalar_type() == at::kHalf) {
        return dispatch_quantize_bcthw_to_tin16<half>(
            input_c, scale_c, padded_height, padded_width, padded_channels);
    }
    return dispatch_quantize_bcthw_to_tin16<float>(
        input_c, scale_c, padded_height, padded_width, padded_channels);
}

torch::Tensor lightvae_fp8_dequantize_tin16_to_bcthw(
    torch::Tensor input,
    torch::Tensor scale,
    int real_channels,
    int real_height,
    int real_width) {
    check_u8_cuda_contiguous(input, "input");
    check_scale_cuda_contiguous(scale, "scale");
    TORCH_CHECK(input.dim() == 5 && input.size(4) == kFp8ChannelsPerSlice,
                "input must be [T,C/16,H,W,16], got ", input.sizes());
    TORCH_CHECK(real_channels > 0 && real_channels <= input.size(1) * input.size(4),
                "real_channels out of range");
    TORCH_CHECK(real_height > 0 && real_height <= input.size(2) &&
                    real_width > 0 && real_width <= input.size(3),
                "real spatial shape must be within the padded tensor");
    TORCH_CHECK(scale.numel() == 1 || scale.numel() == real_channels,
                "scale must be scalar or have one entry per real output channel");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    auto scale_c = scale.to(input.device(), scale.scalar_type(), false, true).contiguous();
    if (scale_c.scalar_type() == at::kHalf) {
        return dispatch_dequantize_tin16_to_bcthw<half>(
            input_c, scale_c, real_channels, real_height, real_width);
    }
    return dispatch_dequantize_tin16_to_bcthw<float>(
        input_c, scale_c, real_channels, real_height, real_width);
}

torch::Tensor lightvae_fp8_rmsnorm_tin16(
    torch::Tensor input,
    torch::Tensor input_scale,
    torch::Tensor gamma,
    torch::Tensor output_scale,
    int real_channels,
    bool silu) {
    check_u8_cuda_contiguous(input, "input");
    check_scale_cuda_contiguous(input_scale, "input_scale");
    check_half_cuda_contiguous(gamma, "gamma");
    check_scale_cuda_contiguous(output_scale, "output_scale");
    TORCH_CHECK(input.dim() == 5 && input.size(4) == kFp8ChannelsPerSlice,
                "input must be [T,C/16,H,W,16], got ", input.sizes());
    const int channels = static_cast<int>(input.size(1) * input.size(4));
    TORCH_CHECK(channels == 32 || channels == 48 || channels == 64 || channels == 96 || channels == 192 ||
                    channels == 288,
                "unsupported LightVAE FP8 RMSNorm padded channel count ", channels);
    TORCH_CHECK(real_channels > 0 && real_channels <= channels, "real_channels out of range");
    TORCH_CHECK(gamma.numel() >= real_channels, "gamma must have at least real_channels entries");
    TORCH_CHECK(input_scale.numel() == 1 || input_scale.numel() == real_channels,
                "input_scale must be scalar or per real channel");
    TORCH_CHECK(output_scale.numel() == 1 || output_scale.numel() == real_channels,
                "output_scale must be scalar or per real channel");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    auto gamma_c = gamma.contiguous();
    auto in_s = input_scale.to(input.device(), input_scale.scalar_type(), false, true).contiguous();
    auto out_s = output_scale.to(input.device(), output_scale.scalar_type(), false, true).contiguous();

#define DISPATCH_RMS(CH, IN_T, OUT_T) \
    dispatch_rmsnorm_tin16<CH, IN_T, OUT_T>(input_c, in_s, gamma_c, out_s, real_channels, silu)
#define DISPATCH_RMS_IN(CH, IN_T) \
    (out_s.scalar_type() == at::kHalf ? DISPATCH_RMS(CH, IN_T, half) : DISPATCH_RMS(CH, IN_T, float))
#define DISPATCH_RMS_CH(CH) \
    (in_s.scalar_type() == at::kHalf ? DISPATCH_RMS_IN(CH, half) : DISPATCH_RMS_IN(CH, float))
    switch (channels) {
        case 32: return DISPATCH_RMS_CH(32);
        case 48: return DISPATCH_RMS_CH(48);
        case 64: return DISPATCH_RMS_CH(64);
        case 96: return DISPATCH_RMS_CH(96);
        case 192: return DISPATCH_RMS_CH(192);
        case 288: return DISPATCH_RMS_CH(288);
        default: TORCH_CHECK(false, "unsupported channel count");
    }
#undef DISPATCH_RMS_CH
#undef DISPATCH_RMS_IN
#undef DISPATCH_RMS
}

torch::Tensor lightvae_fp8_add_tin16(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor output_scale,
    int real_channels) {
    check_u8_cuda_contiguous(a, "a");
    check_u8_cuda_contiguous(b, "b");
    check_scale_cuda_contiguous(a_scale, "a_scale");
    check_scale_cuda_contiguous(b_scale, "b_scale");
    check_scale_cuda_contiguous(output_scale, "output_scale");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b shapes must match");
    TORCH_CHECK(a.dim() == 5 && a.size(4) == kFp8ChannelsPerSlice,
                "inputs must be [T,C/16,H,W,16], got ", a.sizes());
    const int channels = static_cast<int>(a.size(1) * a.size(4));
    if (real_channels < 0) {
        real_channels = channels;
    }
    TORCH_CHECK(real_channels > 0 && real_channels <= channels, "real_channels out of range");
    TORCH_CHECK(a_scale.numel() == 1 || a_scale.numel() == real_channels,
                "a_scale must be scalar or per real channel");
    TORCH_CHECK(b_scale.numel() == 1 || b_scale.numel() == real_channels,
                "b_scale must be scalar or per real channel");
    TORCH_CHECK(output_scale.numel() == 1 || output_scale.numel() == real_channels,
                "output_scale must be scalar or per real channel");
    at::cuda::CUDAGuard guard(a.device());
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto as = a_scale.to(a.device(), a_scale.scalar_type(), false, true).contiguous();
    auto bs = b_scale.to(a.device(), b_scale.scalar_type(), false, true).contiguous();
    auto os = output_scale.to(a.device(), output_scale.scalar_type(), false, true).contiguous();

#define DISPATCH_ADD(A_T, B_T, O_T) \
    dispatch_add_tin16<A_T, B_T, O_T>(a_c, b_c, as, bs, os, real_channels, false)
#define DISPATCH_ADD_O(A_T, B_T) \
    (os.scalar_type() == at::kHalf ? DISPATCH_ADD(A_T, B_T, half) : DISPATCH_ADD(A_T, B_T, float))
#define DISPATCH_ADD_B(A_T) \
    (bs.scalar_type() == at::kHalf ? DISPATCH_ADD_O(A_T, half) : DISPATCH_ADD_O(A_T, float))
    return as.scalar_type() == at::kHalf ? DISPATCH_ADD_B(half) : DISPATCH_ADD_B(float);
#undef DISPATCH_ADD_B
#undef DISPATCH_ADD_O
#undef DISPATCH_ADD
}

torch::Tensor lightvae_fp8_add_relu_tin16(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor output_scale,
    int real_channels) {
    check_u8_cuda_contiguous(a, "a");
    check_u8_cuda_contiguous(b, "b");
    check_scale_cuda_contiguous(a_scale, "a_scale");
    check_scale_cuda_contiguous(b_scale, "b_scale");
    check_scale_cuda_contiguous(output_scale, "output_scale");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b shapes must match");
    TORCH_CHECK(a.dim() == 5 && a.size(4) == kFp8ChannelsPerSlice,
                "inputs must be [T,C/16,H,W,16], got ", a.sizes());
    const int channels = static_cast<int>(a.size(1) * a.size(4));
    if (real_channels < 0) {
        real_channels = channels;
    }
    TORCH_CHECK(real_channels > 0 && real_channels <= channels, "real_channels out of range");
    TORCH_CHECK(a_scale.numel() == 1 || a_scale.numel() == real_channels,
                "a_scale must be scalar or per real channel");
    TORCH_CHECK(b_scale.numel() == 1 || b_scale.numel() == real_channels,
                "b_scale must be scalar or per real channel");
    TORCH_CHECK(output_scale.numel() == 1 || output_scale.numel() == real_channels,
                "output_scale must be scalar or per real channel");
    at::cuda::CUDAGuard guard(a.device());
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto as = a_scale.to(a.device(), a_scale.scalar_type(), false, true).contiguous();
    auto bs = b_scale.to(a.device(), b_scale.scalar_type(), false, true).contiguous();
    auto os = output_scale.to(a.device(), output_scale.scalar_type(), false, true).contiguous();

#define DISPATCH_ADD_RELU(A_T, B_T, O_T) \
    dispatch_add_tin16<A_T, B_T, O_T>(a_c, b_c, as, bs, os, real_channels, true)
#define DISPATCH_ADD_RELU_O(A_T, B_T) \
    (os.scalar_type() == at::kHalf ? DISPATCH_ADD_RELU(A_T, B_T, half) : DISPATCH_ADD_RELU(A_T, B_T, float))
#define DISPATCH_ADD_RELU_B(A_T) \
    (bs.scalar_type() == at::kHalf ? DISPATCH_ADD_RELU_O(A_T, half) : DISPATCH_ADD_RELU_O(A_T, float))
    return as.scalar_type() == at::kHalf ? DISPATCH_ADD_RELU_B(half) : DISPATCH_ADD_RELU_B(float);
#undef DISPATCH_ADD_RELU_B
#undef DISPATCH_ADD_RELU_O
#undef DISPATCH_ADD_RELU
}

torch::Tensor lightvae_fp8_add_relu_nhwc_tin16_to_tin16(
    torch::Tensor a_nhwc,
    torch::Tensor b_tin16,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor output_scale,
    int real_channels) {
    check_u8_cuda_contiguous(a_nhwc, "a_nhwc");
    check_u8_cuda_contiguous(b_tin16, "b_tin16");
    check_scale_cuda_contiguous(a_scale, "a_scale");
    check_scale_cuda_contiguous(b_scale, "b_scale");
    check_scale_cuda_contiguous(output_scale, "output_scale");
    TORCH_CHECK(a_nhwc.dim() == 4, "a_nhwc must be [T,H,W,C], got ", a_nhwc.sizes());
    TORCH_CHECK(b_tin16.dim() == 5 && b_tin16.size(4) == kFp8ChannelsPerSlice,
                "b_tin16 must be [T,C/16,H,W,16], got ", b_tin16.sizes());
    const int channels = static_cast<int>(b_tin16.size(1) * b_tin16.size(4));
    TORCH_CHECK(a_nhwc.size(0) == b_tin16.size(0) && a_nhwc.size(1) == b_tin16.size(2) &&
                    a_nhwc.size(2) == b_tin16.size(3) && a_nhwc.size(3) == channels,
                "a_nhwc shape must match b_tin16 as [T,H,W,C]");
    if (real_channels < 0) {
        real_channels = channels;
    }
    TORCH_CHECK(real_channels > 0 && real_channels <= channels, "real_channels out of range");
    TORCH_CHECK(a_scale.numel() == 1 || a_scale.numel() == real_channels,
                "a_scale must be scalar or per real channel");
    TORCH_CHECK(b_scale.numel() == 1 || b_scale.numel() == real_channels,
                "b_scale must be scalar or per real channel");
    TORCH_CHECK(output_scale.numel() == 1 || output_scale.numel() == real_channels,
                "output_scale must be scalar or per real channel");
    at::cuda::CUDAGuard guard(b_tin16.device());
    auto a_c = a_nhwc.contiguous();
    auto b_c = b_tin16.contiguous();
    auto as = a_scale.to(b_tin16.device(), a_scale.scalar_type(), false, true).contiguous();
    auto bs = b_scale.to(b_tin16.device(), b_scale.scalar_type(), false, true).contiguous();
    auto os = output_scale.to(b_tin16.device(), output_scale.scalar_type(), false, true).contiguous();
    auto output = torch::empty_like(b_c);

#define DISPATCH_ADD_NHWC(A_T, B_T, O_T) \
    add_relu_nhwc_tin16_to_tin16_kernel<A_T, B_T, O_T><<< \
        launch_blocks(output.numel()), 256, 0, at::cuda::getCurrentCUDAStream()>>>( \
        a_c.template data_ptr<uint8_t>(), b_c.template data_ptr<uint8_t>(), output.template data_ptr<uint8_t>(), \
        reinterpret_cast<const A_T*>(as.data_ptr()), reinterpret_cast<const B_T*>(bs.data_ptr()), \
        reinterpret_cast<const O_T*>(os.data_ptr()), output.numel(), static_cast<int>(b_c.size(1)), \
        static_cast<int>(b_c.size(2)), static_cast<int>(b_c.size(3)), real_channels, \
        static_cast<int>(as.numel()), static_cast<int>(bs.numel()), static_cast<int>(os.numel()))
#define DISPATCH_ADD_NHWC_O(A_T, B_T) \
    (os.scalar_type() == at::kHalf ? DISPATCH_ADD_NHWC(A_T, B_T, half) : DISPATCH_ADD_NHWC(A_T, B_T, float))
#define DISPATCH_ADD_NHWC_B(A_T) \
    (bs.scalar_type() == at::kHalf ? DISPATCH_ADD_NHWC_O(A_T, half) : DISPATCH_ADD_NHWC_O(A_T, float))
    as.scalar_type() == at::kHalf ? DISPATCH_ADD_NHWC_B(half) : DISPATCH_ADD_NHWC_B(float);
#undef DISPATCH_ADD_NHWC_B
#undef DISPATCH_ADD_NHWC_O
#undef DISPATCH_ADD_NHWC
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor lightvae_fp8_upsample2x_tin16(torch::Tensor input) {
    check_tin16_cuda_contiguous(input, "input");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    const int frames = static_cast<int>(input_c.size(0));
    const int slices = static_cast<int>(input_c.size(1));
    const int height = static_cast<int>(input_c.size(2));
    const int width = static_cast<int>(input_c.size(3));
    auto output = torch::empty(
        {frames, slices, height * 2, width * 2, kFp8ChannelsPerSlice},
        input_c.options());
    const long long total = output.numel();
    upsample2x_tin16_kernel<<<launch_blocks(total), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        input_c.template data_ptr<uint8_t>(),
        output.template data_ptr<uint8_t>(),
        total,
        frames,
        slices,
        height,
        width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor lightvae_fp8_extract_mu_normalize_tin16(
    torch::Tensor input,
    torch::Tensor input_scale,
    torch::Tensor mean,
    torch::Tensor inv_std,
    int real_height,
    int real_width) {
    check_u8_cuda_contiguous(input, "input");
    check_scale_cuda_contiguous(input_scale, "input_scale");
    check_half_cuda_contiguous(mean, "mean");
    check_half_cuda_contiguous(inv_std, "inv_std");
    TORCH_CHECK(input.dim() == 5 && input.size(1) >= 2 && input.size(4) == kFp8ChannelsPerSlice,
                "input must be [T,C/16,H,W,16] with at least 32 padded channels, got ", input.sizes());
    TORCH_CHECK(mean.numel() >= 16 && inv_std.numel() >= 16, "mean and inv_std must have at least 16 entries");
    TORCH_CHECK(input_scale.numel() == 1 || input_scale.numel() >= 16,
                "input_scale must be scalar or have at least 16 entries");
    TORCH_CHECK(real_height > 0 && real_height <= input.size(2) &&
                    real_width > 0 && real_width <= input.size(3),
                "real spatial shape must be within the padded tensor");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    auto in_s = input_scale.to(input.device(), input_scale.scalar_type(), false, true).contiguous();
    auto mean_c = mean.contiguous();
    auto inv_std_c = inv_std.contiguous();
    if (in_s.scalar_type() == at::kHalf) {
        return dispatch_extract_mu_normalize_tin16<half>(
            input_c, in_s, mean_c, inv_std_c, real_height, real_width);
    }
    return dispatch_extract_mu_normalize_tin16<float>(
        input_c, in_s, mean_c, inv_std_c, real_height, real_width);
}

torch::Tensor cutlass_conv3x3_nhwc_fp8(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int pad,
    bool relu) {
    check_u8_cuda_contiguous(input, "input");
    check_u8_cuda_contiguous(weight, "weight");
    TORCH_CHECK(input.dim() == 4, "input must be NHWC [N,H,W,C], got ", input.sizes());
    TORCH_CHECK(weight.dim() == 4, "weight must be KRSC [K,R,S,C], got ", weight.sizes());
    TORCH_CHECK(weight.size(1) == 3 && weight.size(2) == 3, "weight must be 3x3");
    TORCH_CHECK(input.size(3) == weight.size(3), "weight C must match input C");
    TORCH_CHECK(input.size(3) % 16 == 0 && weight.size(0) % 16 == 0,
                "FP8 Conv2D requires C and K divisible by 16");
    TORCH_CHECK(stride > 0 && pad >= 0, "invalid stride/pad");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    auto weight_c = weight.contiguous();
    const int n = static_cast<int>(input_c.size(0));
    const int h = static_cast<int>(input_c.size(1));
    const int w = static_cast<int>(input_c.size(2));
    const int k = static_cast<int>(weight_c.size(0));
    const int h_out = (h + 2 * pad - 3) / stride + 1;
    const int w_out = (w + 2 * pad - 3) / stride + 1;
    TORCH_CHECK(h_out > 0 && w_out > 0, "invalid output shape");
    auto output = torch::empty({n, h_out, w_out, k}, input_c.options());
    cudaError_t err = relu
        ? cutlass_conv2d_fp8_impl<cutlass::epilogue::thread::ReLu>(
              input_c, weight_c, output, stride, pad, at::cuda::getCurrentCUDAStream())
        : cutlass_conv2d_fp8_impl<cutlass::epilogue::thread::Identity>(
              input_c, weight_c, output, stride, pad, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == cudaSuccess, "CUTLASS FP8 Conv2D failed: ", cudaGetErrorString(err));
    return output;
}

template <typename InScaleT, typename OutScaleT>
torch::Tensor dispatch_prepare_conv2d_weight_krsc(
    torch::Tensor weight_krsc,
    torch::Tensor input_scale,
    torch::Tensor output_scale) {
    auto output = torch::empty(weight_krsc.sizes(), weight_krsc.options().dtype(torch::kUInt8));
    prepare_conv2d_weight_krsc_kernel<InScaleT, OutScaleT><<<
        launch_blocks(output.numel()), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        half_ptr_const(weight_krsc),
        output.template data_ptr<uint8_t>(),
        reinterpret_cast<const InScaleT*>(input_scale.data_ptr()),
        reinterpret_cast<const OutScaleT*>(output_scale.data_ptr()),
        output.numel(),
        static_cast<int>(weight_krsc.size(3)),
        static_cast<int>(weight_krsc.size(0)),
        static_cast<int>(weight_krsc.size(1)),
        static_cast<int>(weight_krsc.size(2)),
        static_cast<int>(input_scale.numel()),
        static_cast<int>(output_scale.numel()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <typename OutScaleT>
void dispatch_add_bias_act_fp8_nhwc(
    torch::Tensor output,
    torch::Tensor bias,
    torch::Tensor output_scale,
    bool relu) {
    add_bias_act_fp8_nhwc_kernel<OutScaleT><<<
        launch_blocks(output.numel()), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        output.template data_ptr<uint8_t>(),
        half_ptr_const(bias),
        reinterpret_cast<const OutScaleT*>(output_scale.data_ptr()),
        output.numel(),
        static_cast<int>(output.size(3)),
        static_cast<int>(output_scale.numel()),
        relu);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SrcScaleT, typename OutScaleT>
torch::Tensor dispatch_nhwc_rescale_to_tin16(
    torch::Tensor input,
    torch::Tensor source_scale,
    torch::Tensor output_scale) {
    const int frames = static_cast<int>(input.size(0));
    const int height = static_cast<int>(input.size(1));
    const int width = static_cast<int>(input.size(2));
    const int slices = static_cast<int>(input.size(3) / kFp8ChannelsPerSlice);
    auto output = torch::empty({frames, slices, height, width, kFp8ChannelsPerSlice}, input.options());
    nhwc_rescale_to_tin16_kernel<SrcScaleT, OutScaleT><<<
        launch_blocks(output.numel()), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.template data_ptr<uint8_t>(),
        output.template data_ptr<uint8_t>(),
        reinterpret_cast<const SrcScaleT*>(source_scale.data_ptr()),
        reinterpret_cast<const OutScaleT*>(output_scale.data_ptr()),
        frames,
        slices,
        height,
        width,
        static_cast<int>(source_scale.numel()),
        static_cast<int>(output_scale.numel()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor lightvae_fp8_prepare_conv2d_weight_krsc(
    torch::Tensor weight_krsc,
    torch::Tensor input_scale,
    torch::Tensor output_scale) {
    check_half_cuda_contiguous(weight_krsc, "weight_krsc");
    check_scale_cuda_contiguous(input_scale, "input_scale");
    check_scale_cuda_contiguous(output_scale, "output_scale");
    TORCH_CHECK(weight_krsc.dim() == 4, "weight_krsc must be [K,R,S,C], got ", weight_krsc.sizes());
    TORCH_CHECK(weight_krsc.size(1) > 0 && weight_krsc.size(2) > 0,
                "weight spatial shape must be non-empty");
    TORCH_CHECK(weight_krsc.size(3) % 16 == 0 && weight_krsc.size(0) % 16 == 0,
                "FP8 Conv2D requires C and K divisible by 16");
    TORCH_CHECK(input_scale.numel() == 1 || input_scale.numel() == weight_krsc.size(3),
                "input_scale must be scalar or have one entry per prepared input channel");
    TORCH_CHECK(output_scale.numel() == 1 || output_scale.numel() == weight_krsc.size(0),
                "output_scale must be scalar or have one entry per prepared output channel");
    at::cuda::CUDAGuard guard(weight_krsc.device());
    auto w = weight_krsc.contiguous();
    auto in_s = input_scale.to(w.device(), input_scale.scalar_type(), false, true).contiguous();
    auto out_s = output_scale.to(w.device(), output_scale.scalar_type(), false, true).contiguous();

#define DISPATCH_PREP(IN_T, OUT_T) dispatch_prepare_conv2d_weight_krsc<IN_T, OUT_T>(w, in_s, out_s)
#define DISPATCH_PREP_OUT(IN_T) \
    (out_s.scalar_type() == at::kHalf ? DISPATCH_PREP(IN_T, half) : DISPATCH_PREP(IN_T, float))
    return in_s.scalar_type() == at::kHalf ? DISPATCH_PREP_OUT(half) : DISPATCH_PREP_OUT(float);
#undef DISPATCH_PREP_OUT
#undef DISPATCH_PREP
}

torch::Tensor cutlass_conv2d_nhwc_fp8_prepared(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    int stride,
    int pad,
    bool relu) {
    check_u8_cuda_contiguous(input, "input");
    check_u8_cuda_contiguous(weight, "weight");
    check_scale_cuda_contiguous(output_scale, "output_scale");
    TORCH_CHECK(input.dim() == 4, "input must be NHWC [N,H,W,C], got ", input.sizes());
    TORCH_CHECK(weight.dim() == 4, "weight must be KRSC [K,R,S,C], got ", weight.sizes());
    TORCH_CHECK(weight.size(1) > 0 && weight.size(2) > 0,
                "weight spatial shape must be non-empty");
    TORCH_CHECK(input.size(3) == weight.size(3), "weight C must match input C");
    TORCH_CHECK(input.size(3) % 16 == 0 && weight.size(0) % 16 == 0,
                "FP8 Conv2D requires C and K divisible by 16");
    TORCH_CHECK(output_scale.numel() == 1 || output_scale.numel() == weight.size(0),
                "output_scale must be scalar or have one entry per output channel");
    TORCH_CHECK(stride > 0 && pad >= 0, "invalid stride/pad");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    auto weight_c = weight.contiguous();
    auto out_s = output_scale.to(input.device(), output_scale.scalar_type(), false, true).contiguous();
    const int n = static_cast<int>(input_c.size(0));
    const int h = static_cast<int>(input_c.size(1));
    const int w = static_cast<int>(input_c.size(2));
    const int k = static_cast<int>(weight_c.size(0));
    const int r = static_cast<int>(weight_c.size(1));
    const int s = static_cast<int>(weight_c.size(2));
    const int h_out = (h + 2 * pad - r) / stride + 1;
    const int w_out = (w + 2 * pad - s) / stride + 1;
    TORCH_CHECK(h_out > 0 && w_out > 0, "invalid output shape");
    auto output = torch::empty({n, h_out, w_out, k}, input_c.options());
    cudaError_t err = cutlass_conv2d_fp8_plain_impl(
        input_c, weight_c, output, stride, pad, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == cudaSuccess, "CUTLASS FP8 Conv2D failed: ", cudaGetErrorString(err));

    if (bias.has_value() && bias.value().defined()) {
        auto b = bias.value();
        check_half_cuda_contiguous(b, "bias");
        TORCH_CHECK(b.numel() == k, "bias must have one entry per output channel");
        auto b_c = b.contiguous();
        if (out_s.scalar_type() == at::kHalf) {
            dispatch_add_bias_act_fp8_nhwc<half>(output, b_c, out_s, relu);
        } else {
            dispatch_add_bias_act_fp8_nhwc<float>(output, b_c, out_s, relu);
        }
    } else if (relu) {
        auto zero_bias = torch::zeros({k}, input_c.options().dtype(torch::kFloat16));
        if (out_s.scalar_type() == at::kHalf) {
            dispatch_add_bias_act_fp8_nhwc<half>(output, zero_bias, out_s, true);
        } else {
            dispatch_add_bias_act_fp8_nhwc<float>(output, zero_bias, out_s, true);
        }
    }
    return output;
}

torch::Tensor lightvae_fp8_tin16_to_nhwc(torch::Tensor input) {
    check_tin16_cuda_contiguous(input, "input");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    const int frames = static_cast<int>(input_c.size(0));
    const int slices = static_cast<int>(input_c.size(1));
    const int height = static_cast<int>(input_c.size(2));
    const int width = static_cast<int>(input_c.size(3));
    auto output = torch::empty(
        {frames, height, width, slices * kFp8ChannelsPerSlice},
        input_c.options());
    tin16_to_nhwc_kernel<<<launch_blocks(output.numel()), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        input_c.template data_ptr<uint8_t>(),
        output.template data_ptr<uint8_t>(),
        frames,
        slices,
        height,
        width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor lightvae_fp8_nhwc_to_tin16(torch::Tensor input) {
    check_u8_cuda_contiguous(input, "input");
    TORCH_CHECK(input.dim() == 4, "input must be NHWC [N,H,W,C], got ", input.sizes());
    TORCH_CHECK(input.size(3) % kFp8ChannelsPerSlice == 0,
                "input channel count must be divisible by ", kFp8ChannelsPerSlice);
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    const int frames = static_cast<int>(input_c.size(0));
    const int height = static_cast<int>(input_c.size(1));
    const int width = static_cast<int>(input_c.size(2));
    const int slices = static_cast<int>(input_c.size(3) / kFp8ChannelsPerSlice);
    auto output = torch::empty(
        {frames, slices, height, width, kFp8ChannelsPerSlice},
        input_c.options());
    nhwc_to_tin16_kernel<<<launch_blocks(output.numel()), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        input_c.template data_ptr<uint8_t>(),
        output.template data_ptr<uint8_t>(),
        frames,
        slices,
        height,
        width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor lightvae_fp8_pack_causal3_tin16(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache) {
    check_tin16_cuda_contiguous(input, "input");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    const int frames = static_cast<int>(input_c.size(0));
    const int slices = static_cast<int>(input_c.size(1));
    const int height = static_cast<int>(input_c.size(2));
    const int width = static_cast<int>(input_c.size(3));
    const uint8_t* cache_ptr = nullptr;
    int cache_frames = 0;
    torch::Tensor cache_c;
    if (cache.has_value() && cache.value().defined()) {
        cache_c = cache.value().contiguous();
        check_tin16_cuda_contiguous(cache_c, "cache");
        TORCH_CHECK(cache_c.device() == input_c.device(), "cache must be on the same device as input");
        TORCH_CHECK(cache_c.size(1) == slices && cache_c.size(2) == height && cache_c.size(3) == width,
                    "cache shape must match input channel/spatial dimensions");
        cache_frames = static_cast<int>(cache_c.size(0));
        TORCH_CHECK(cache_frames >= 0 && cache_frames <= 2, "causal cache must have 0, 1, or 2 frames");
        cache_ptr = cache_c.template data_ptr<uint8_t>();
    }
    auto output = torch::empty(
        {frames, slices * 3, height, width, kFp8ChannelsPerSlice},
        input_c.options());
    pack_causal3_tin16_kernel<<<launch_blocks(output.numel()), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        input_c.template data_ptr<uint8_t>(),
        cache_ptr,
        output.template data_ptr<uint8_t>(),
        frames,
        cache_frames,
        slices,
        height,
        width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor lightvae_fp8_nhwc_rescale_to_tin16(
    torch::Tensor input,
    torch::Tensor source_scale,
    torch::Tensor output_scale) {
    check_u8_cuda_contiguous(input, "input");
    check_scale_cuda_contiguous(source_scale, "source_scale");
    check_scale_cuda_contiguous(output_scale, "output_scale");
    TORCH_CHECK(input.dim() == 4, "input must be NHWC [N,H,W,C], got ", input.sizes());
    TORCH_CHECK(input.size(3) % kFp8ChannelsPerSlice == 0,
                "input channel count must be divisible by ", kFp8ChannelsPerSlice);
    TORCH_CHECK(source_scale.numel() == 1 || source_scale.numel() == input.size(3),
                "source_scale must be scalar or have one entry per channel");
    TORCH_CHECK(output_scale.numel() == 1 || output_scale.numel() == input.size(3),
                "output_scale must be scalar or have one entry per channel");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    auto src_s = source_scale.to(input.device(), source_scale.scalar_type(), false, true).contiguous();
    auto out_s = output_scale.to(input.device(), output_scale.scalar_type(), false, true).contiguous();

#define DISPATCH_RESCALE(SRC_T, OUT_T) dispatch_nhwc_rescale_to_tin16<SRC_T, OUT_T>(input_c, src_s, out_s)
#define DISPATCH_RESCALE_OUT(SRC_T) \
    (out_s.scalar_type() == at::kHalf ? DISPATCH_RESCALE(SRC_T, half) : DISPATCH_RESCALE(SRC_T, float))
    return src_s.scalar_type() == at::kHalf ? DISPATCH_RESCALE_OUT(half) : DISPATCH_RESCALE_OUT(float);
#undef DISPATCH_RESCALE_OUT
#undef DISPATCH_RESCALE
}

torch::Tensor lightvae_fp8_pack_causal3_tin16_to_nhwc(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache) {
    check_tin16_cuda_contiguous(input, "input");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    const int frames = static_cast<int>(input_c.size(0));
    const int slices = static_cast<int>(input_c.size(1));
    const int height = static_cast<int>(input_c.size(2));
    const int width = static_cast<int>(input_c.size(3));
    const uint8_t* cache_ptr = nullptr;
    int cache_frames = 0;
    torch::Tensor cache_c;
    if (cache.has_value() && cache.value().defined()) {
        cache_c = cache.value().contiguous();
        check_tin16_cuda_contiguous(cache_c, "cache");
        TORCH_CHECK(cache_c.device() == input_c.device(), "cache must be on the same device as input");
        TORCH_CHECK(cache_c.size(1) == slices && cache_c.size(2) == height && cache_c.size(3) == width,
                    "cache shape must match input channel/spatial dimensions");
        cache_frames = static_cast<int>(cache_c.size(0));
        TORCH_CHECK(cache_frames >= 0 && cache_frames <= 2, "causal cache must have 0, 1, or 2 frames");
        cache_ptr = cache_c.template data_ptr<uint8_t>();
    }
    auto output = torch::empty(
        {frames, height, width, slices * 3 * kFp8ChannelsPerSlice},
        input_c.options());
    pack_causal3_tin16_to_nhwc_kernel<<<launch_blocks(output.numel()), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        input_c.template data_ptr<uint8_t>(),
        cache_ptr,
        output.template data_ptr<uint8_t>(),
        frames,
        cache_frames,
        slices,
        height,
        width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor lightvae_fp8_update_tail_cache_tin16(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache,
    int cache_frames) {
    check_tin16_cuda_contiguous(input, "input");
    TORCH_CHECK(cache_frames > 0, "cache_frames must be positive");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    const int frames = static_cast<int>(input_c.size(0));
    const int slices = static_cast<int>(input_c.size(1));
    const int height = static_cast<int>(input_c.size(2));
    const int width = static_cast<int>(input_c.size(3));
    const uint8_t* cache_ptr = nullptr;
    int old_cache_frames = 0;
    torch::Tensor cache_c;
    if (cache.has_value() && cache.value().defined()) {
        cache_c = cache.value().contiguous();
        check_tin16_cuda_contiguous(cache_c, "cache");
        TORCH_CHECK(cache_c.device() == input_c.device(), "cache must be on the same device as input");
        TORCH_CHECK(cache_c.size(1) == slices && cache_c.size(2) == height && cache_c.size(3) == width,
                    "cache shape must match input channel/spatial dimensions");
        old_cache_frames = static_cast<int>(cache_c.size(0));
        cache_ptr = cache_c.template data_ptr<uint8_t>();
    }
    const int new_cache_frames = std::min(cache_frames, old_cache_frames + frames);
    auto output = torch::empty(
        {new_cache_frames, slices, height, width, kFp8ChannelsPerSlice},
        input_c.options());
    update_tail_cache_tin16_kernel<<<launch_blocks(output.numel()), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        input_c.template data_ptr<uint8_t>(),
        cache_ptr,
        output.template data_ptr<uint8_t>(),
        frames,
        old_cache_frames,
        new_cache_frames,
        slices,
        height,
        width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor lightvae_fp8_spatial_pad_right_bottom_tin16(torch::Tensor input) {
    check_tin16_cuda_contiguous(input, "input");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    const int frames = static_cast<int>(input_c.size(0));
    const int slices = static_cast<int>(input_c.size(1));
    const int height = static_cast<int>(input_c.size(2));
    const int width = static_cast<int>(input_c.size(3));
    auto output = torch::empty(
        {frames, slices, height + 1, width + 1, kFp8ChannelsPerSlice},
        input_c.options());
    spatial_pad_right_bottom_tin16_kernel<<<launch_blocks(output.numel()), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        input_c.template data_ptr<uint8_t>(),
        output.template data_ptr<uint8_t>(),
        frames,
        slices,
        height,
        width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor lightvae_fp8_pack_temporal3_tin16(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache) {
    check_tin16_cuda_contiguous(input, "input");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    const int frames = static_cast<int>(input_c.size(0));
    const int slices = static_cast<int>(input_c.size(1));
    const int height = static_cast<int>(input_c.size(2));
    const int width = static_cast<int>(input_c.size(3));
    const bool has_cache = cache.has_value() && cache.value().defined();
    const uint8_t* cache_ptr = nullptr;
    torch::Tensor cache_c;
    if (has_cache) {
        cache_c = cache.value().contiguous();
        check_tin16_cuda_contiguous(cache_c, "cache");
        TORCH_CHECK(cache_c.device() == input_c.device(), "cache must be on the same device as input");
        TORCH_CHECK(cache_c.size(0) == 1 && cache_c.size(1) == slices &&
                        cache_c.size(2) == height && cache_c.size(3) == width,
                    "temporal cache must be [1,C/16,H,W,16] matching input");
        cache_ptr = cache_c.template data_ptr<uint8_t>();
    }
    const int t_out = has_cache ? ((frames + 1 - 3) / 2 + 1) : ((frames - 3) / 2 + 1);
    TORCH_CHECK(t_out > 0, "temporal pack produced invalid T_out=", t_out, " from T=", frames);
    auto output = torch::empty(
        {t_out, slices * 3, height, width, kFp8ChannelsPerSlice},
        input_c.options());
    pack_temporal3_tin16_kernel<<<launch_blocks(output.numel()), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        input_c.template data_ptr<uint8_t>(),
        cache_ptr,
        output.template data_ptr<uint8_t>(),
        frames,
        t_out,
        slices,
        height,
        width,
        has_cache);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor lightvae_fp8_pack_temporal3_tin16_to_nhwc(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache) {
    check_tin16_cuda_contiguous(input, "input");
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    const int frames = static_cast<int>(input_c.size(0));
    const int slices = static_cast<int>(input_c.size(1));
    const int height = static_cast<int>(input_c.size(2));
    const int width = static_cast<int>(input_c.size(3));
    const bool has_cache = cache.has_value() && cache.value().defined();
    const uint8_t* cache_ptr = nullptr;
    torch::Tensor cache_c;
    if (has_cache) {
        cache_c = cache.value().contiguous();
        check_tin16_cuda_contiguous(cache_c, "cache");
        TORCH_CHECK(cache_c.device() == input_c.device(), "cache must be on the same device as input");
        TORCH_CHECK(cache_c.size(0) == 1 && cache_c.size(1) == slices &&
                        cache_c.size(2) == height && cache_c.size(3) == width,
                    "temporal cache must be [1,C/16,H,W,16] matching input");
        cache_ptr = cache_c.template data_ptr<uint8_t>();
    }
    const int t_out = has_cache ? ((frames + 1 - 3) / 2 + 1) : ((frames - 3) / 2 + 1);
    TORCH_CHECK(t_out > 0, "temporal pack produced invalid T_out=", t_out, " from T=", frames);
    auto output = torch::empty(
        {t_out, height, width, slices * 3 * kFp8ChannelsPerSlice},
        input_c.options());
    pack_temporal3_tin16_to_nhwc_kernel<<<launch_blocks(output.numel()), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        input_c.template data_ptr<uint8_t>(),
        cache_ptr,
        output.template data_ptr<uint8_t>(),
        frames,
        t_out,
        slices,
        height,
        width,
        has_cache);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor lightvae_fp8_causal_conv3_tin16_prepared(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    bool relu) {
    auto packed_nhwc = lightvae_fp8_pack_causal3_tin16_to_nhwc(input, cache);
    auto y_nhwc = cutlass_conv2d_nhwc_fp8_prepared(
        packed_nhwc,
        weight,
        output_scale,
        bias,
        1,
        1,
        relu);
    return lightvae_fp8_nhwc_to_tin16(y_nhwc);
}

torch::Tensor lightvae_fp8_spatial_conv3_tin16_prepared(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    int stride,
    int pad,
    bool relu,
    bool pad_right_bottom) {
    return lightvae_fp8_spatial_conv3_tin16_warp_mma_prepared(
        input,
        weight,
        output_scale,
        bias,
        stride,
        pad,
        relu,
        pad_right_bottom);
}

torch::Tensor lightvae_fp8_conv1_tin16_prepared(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    bool relu) {
    return lightvae_fp8_conv1_tin16_warp_mma_prepared(input, weight, output_scale, bias, relu);
}

torch::Tensor lightvae_fp8_temporal_conv1_tin16_prepared(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    bool relu) {
    auto input_nhwc = lightvae_fp8_pack_temporal3_tin16_to_nhwc(input, cache);
    auto y_nhwc = cutlass_conv2d_nhwc_fp8_prepared(
        input_nhwc,
        weight,
        output_scale,
        bias,
        1,
        0,
        relu);
    return lightvae_fp8_nhwc_to_tin16(y_nhwc);
}

}  // namespace omnidreams_singleview
