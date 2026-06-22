// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "lightvae_ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>

#include <algorithm>
#include <cutlass/numeric_conversion.h>
#include <type_traits>

namespace omnidreams_singleview {
namespace {

constexpr int kFp8ChannelsPerSlice = 16;

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_u8_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    check_cuda_contiguous(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == at::kByte, name, " must be torch.uint8");
}

void check_half_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    check_cuda_contiguous(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, name, " must be torch.float16");
}

void check_scale_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    check_cuda_contiguous(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kHalf,
                name, " must be torch.float32 or torch.float16");
    TORCH_CHECK(tensor.dim() == 1, name, " must be 1D");
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

__device__ __forceinline__ uint8_t encode_e4m3_half(half value) {
    return encode_e4m3(__half2float(value));
}

constexpr int kWarpTileX = 16;
constexpr int kWarpTileY = 8;
constexpr int kWarpMmaWarps = 8;
constexpr int kWarpMmaThreads = kWarpMmaWarps * 32;
constexpr int kWarpRowsPerWarp = kWarpTileY / kWarpMmaWarps;
constexpr int kWarpLocalPixels = kWarpTileX * kWarpRowsPerWarp;

int default_out_tile_channels(int out_channels) {
    if (out_channels <= 32) {
        return 32;
    }
    if (out_channels <= 64) {
        return 64;
    }
    if (out_channels <= 96) {
        return 96;
    }
    return 16;
}

int default_conv1_out_tile_channels(int out_channels) {
    if (out_channels <= 32) {
        return 32;
    }
    if (out_channels <= 64) {
        return 64;
    }
    return 96;
}

__device__ __forceinline__ int lane_id() {
    int lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

__device__ __forceinline__ uint2 mma_f8e4m3_f16(uint4 a, uint2 b, uint2 c) {
    uint2 d;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
        : "=r"(d.x), "=r"(d.y)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
          "r"(b.x), "r"(b.y),
          "r"(c.x), "r"(c.y));
#else
    d.x = 0;
    d.y = 0;
#endif
    return d;
}

__device__ __forceinline__ int a_fragment_row(int reg_idx, int lane) {
    return (lane / 4) + (reg_idx % 2) * 8;
}

__device__ __forceinline__ int a_fragment_col_base(int reg_idx, int lane) {
    return ((lane % 4) + (reg_idx / 2) * 4) * 4;
}

__device__ __forceinline__ int b_fragment_k_base(int reg_idx, int lane) {
    return ((lane % 4) + (reg_idx % 2) * 4) * 4;
}

__device__ __forceinline__ int b_fragment_col(int reg_idx, int lane) {
    return (lane / 4) + (reg_idx / 2) * 8;
}

__device__ __forceinline__ int c_fragment_row(int reg_idx, int lane) {
    return (lane / 4) + (reg_idx % 2) * 8;
}

__device__ __forceinline__ int c_fragment_col_base(int reg_idx, int lane) {
    return ((lane % 4) + (reg_idx / 2) * 4) * 2;
}

template <bool Causal>
__device__ __forceinline__ uint32_t load_warp_mma_a_u32(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ cache,
    int frame,
    int frames,
    int cache_frames,
    int in_slices,
    int height,
    int width,
    int effective_height,
    int effective_width,
    int tile_y,
    int tile_x,
    int local_pixel,
    int c_base4,
    int r,
    int s,
    int dt,
    int stride,
    int pad) {
    const int local_y = local_pixel / kWarpTileX;
    const int local_x = local_pixel - local_y * kWarpTileX;
    const int oy = tile_y * kWarpTileY + local_y;
    const int ox = tile_x * kWarpTileX + local_x;
    const int iy = oy * stride + r - pad;
    const int ix = ox * stride + s - pad;
    if constexpr (Causal) {
        const int src_ext_t = cache_frames + frame + dt - 2;
        const bool from_cache = src_ext_t >= 0 && src_ext_t < cache_frames && cache != nullptr;
        const int src_t = from_cache ? src_ext_t : (src_ext_t - cache_frames);
        const uint8_t* src = from_cache ? cache : input;
        const int src_frames = from_cache ? cache_frames : frames;
        if (src == nullptr || src_t < 0 || src_t >= src_frames || iy < 0 || iy >= height || ix < 0 || ix >= width) {
            return 0;
        }
        const int slice = c_base4 / kFp8ChannelsPerSlice;
        const int lane = c_base4 - slice * kFp8ChannelsPerSlice;
        const size_t idx =
            (((static_cast<size_t>(src_t) * in_slices + slice) * height + iy) * width + ix) *
                kFp8ChannelsPerSlice + lane;
        return *reinterpret_cast<const uint32_t*>(src + idx);
    } else {
        if (iy < 0 || ix < 0 || iy >= effective_height || ix >= effective_width ||
            iy >= height || ix >= width) {
            return 0;
        }
        const int slice = c_base4 / kFp8ChannelsPerSlice;
        const int lane = c_base4 - slice * kFp8ChannelsPerSlice;
        const size_t idx =
            (((static_cast<size_t>(frame) * in_slices + slice) * height + iy) * width + ix) *
                kFp8ChannelsPerSlice + lane;
        return *reinterpret_cast<const uint32_t*>(input + idx);
    }
}

template <bool Causal>
__device__ __forceinline__ uint32_t load_warp_mma_b_u32(
    const uint8_t* __restrict__ weight,
    int out_channel,
    int c_base4,
    int r,
    int s,
    int dt,
    int in_channels,
    int out_channels) {
    if (out_channel >= out_channels || c_base4 >= in_channels) {
        return 0;
    }
    if constexpr (Causal) {
        const int packed_c = dt * in_channels + c_base4;
        const size_t idx =
            (((static_cast<size_t>(out_channel) * 3 + r) * 3 + s) * (in_channels * 3) + packed_c);
        return *reinterpret_cast<const uint32_t*>(weight + idx);
    } else {
        const size_t idx =
            (((static_cast<size_t>(out_channel) * 3 + r) * 3 + s) * in_channels + c_base4);
        return *reinterpret_cast<const uint32_t*>(weight + idx);
    }
}

__device__ __forceinline__ uint32_t load_conv1_warp_mma_a_u32(
    const uint8_t* __restrict__ input,
    int frame,
    int in_slices,
    int height,
    int width,
    int tile_y,
    int tile_x,
    int local_pixel,
    int c_base4) {
    const int local_y = local_pixel / kWarpTileX;
    const int local_x = local_pixel - local_y * kWarpTileX;
    const int y = tile_y * kWarpTileY + local_y;
    const int x = tile_x * kWarpTileX + local_x;
    if (y >= height || x >= width) {
        return 0;
    }
    const int slice = c_base4 / kFp8ChannelsPerSlice;
    const int lane = c_base4 - slice * kFp8ChannelsPerSlice;
    const size_t idx =
        (((static_cast<size_t>(frame) * in_slices + slice) * height + y) * width + x) *
            kFp8ChannelsPerSlice + lane;
    return *reinterpret_cast<const uint32_t*>(input + idx);
}

__device__ __forceinline__ uint32_t load_conv1_warp_mma_b_u32(
    const uint8_t* __restrict__ weight,
    int out_channel,
    int c_base4,
    int in_channels,
    int out_channels) {
    if (out_channel >= out_channels || c_base4 >= in_channels) {
        return 0;
    }
    const size_t idx = static_cast<size_t>(out_channel) * in_channels + c_base4;
    return *reinterpret_cast<const uint32_t*>(weight + idx);
}

template <bool Causal, int InChannels>
__device__ __forceinline__ uint32_t load_conv3_static_warp_mma_a_u32(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ cache,
    int frame,
    int frames,
    int cache_frames,
    int height,
    int width,
    int effective_height,
    int effective_width,
    int out_height,
    int out_width,
    int tile_y,
    int tile_x,
    int local_pixel,
    int c_base4,
    int r,
    int s,
    int dt) {
    constexpr int InSlices = InChannels / kFp8ChannelsPerSlice;
    const int local_y = local_pixel / kWarpTileX;
    const int local_x = local_pixel - local_y * kWarpTileX;
    const int oy = tile_y * kWarpTileY + local_y;
    const int ox = tile_x * kWarpTileX + local_x;
    if (oy >= out_height || ox >= out_width) {
        return 0;
    }
    int iy;
    int ix;
    if constexpr (Causal) {
        iy = oy + r - 1;
        ix = ox + s - 1;
        const int src_ext_t = cache_frames + frame + dt - 2;
        const bool from_cache = src_ext_t >= 0 && src_ext_t < cache_frames && cache != nullptr;
        const int src_t = from_cache ? src_ext_t : (src_ext_t - cache_frames);
        const uint8_t* src = from_cache ? cache : input;
        const int src_frames = from_cache ? cache_frames : frames;
        if (src == nullptr || src_t < 0 || src_t >= src_frames ||
            iy < 0 || iy >= height || ix < 0 || ix >= width) {
            return 0;
        }
        const int slice = c_base4 / kFp8ChannelsPerSlice;
        const int lane = c_base4 - slice * kFp8ChannelsPerSlice;
        const size_t idx =
            (((static_cast<size_t>(src_t) * InSlices + slice) * height + iy) * width + ix) *
                kFp8ChannelsPerSlice + lane;
        return *reinterpret_cast<const uint32_t*>(src + idx);
    } else {
        iy = oy * 2 + r;
        ix = ox * 2 + s;
        if (iy >= effective_height || ix >= effective_width || iy >= height || ix >= width) {
            return 0;
        }
        const int slice = c_base4 / kFp8ChannelsPerSlice;
        const int lane = c_base4 - slice * kFp8ChannelsPerSlice;
        const size_t idx =
            (((static_cast<size_t>(frame) * InSlices + slice) * height + iy) * width + ix) *
                kFp8ChannelsPerSlice + lane;
        return *reinterpret_cast<const uint32_t*>(input + idx);
    }
}

template <bool Causal, int InChannels, int OutChannels>
__device__ __forceinline__ uint32_t load_conv3_static_warp_mma_b_u32(
    const uint8_t* __restrict__ weight,
    int out_channel,
    int c_base4,
    int r,
    int s,
    int dt) {
    if (out_channel >= OutChannels || c_base4 >= InChannels) {
        return 0;
    }
    if constexpr (Causal) {
        const int packed_c = dt * InChannels + c_base4;
        const size_t idx =
            (((static_cast<size_t>(out_channel) * 3 + r) * 3 + s) * (InChannels * 3) + packed_c);
        return *reinterpret_cast<const uint32_t*>(weight + idx);
    } else {
        const size_t idx =
            (((static_cast<size_t>(out_channel) * 3 + r) * 3 + s) * InChannels + c_base4);
        return *reinterpret_cast<const uint32_t*>(weight + idx);
    }
}

template <typename ScaleT, int OutTileChannels>
__global__ void __launch_bounds__(kWarpMmaThreads, 2) conv1_tin16_warp_mma_kernel(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ weight,
    uint8_t* __restrict__ output,
    const ScaleT* __restrict__ output_scale,
    const half* __restrict__ bias,
    int frames,
    int in_slices,
    int height,
    int width,
    int out_channels,
    int output_scale_count,
    bool relu) {
    const int lane = lane_id();
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int tile_x = static_cast<int>(blockIdx.x);
    const int tile_y = static_cast<int>(blockIdx.y);
    static_assert(OutTileChannels % kFp8ChannelsPerSlice == 0, "output tile must be a multiple of TIN16 channels");
    constexpr int OutGroups = OutTileChannels / kFp8ChannelsPerSlice;
    const int out_blocks = (out_channels + OutTileChannels - 1) / OutTileChannels;
    const int frame = static_cast<int>(blockIdx.z) / out_blocks;
    const int out_block = static_cast<int>(blockIdx.z) - frame * out_blocks;
    const int out_base = out_block * OutTileChannels;
    const int in_channels = in_slices * kFp8ChannelsPerSlice;

    uint32_t acc[OutGroups][4];
    for (int group = 0; group < OutGroups; ++group) {
        for (int reg_idx = 0; reg_idx < 4; ++reg_idx) {
            acc[group][reg_idx] = 0;
        }
    }

    for (int c_base = 0; c_base < in_channels; c_base += 32) {
        uint4 a;
        uint32_t* a_reg = reinterpret_cast<uint32_t*>(&a);
        for (int reg_idx = 0; reg_idx < 4; ++reg_idx) {
            const int local_row = a_fragment_row(reg_idx, lane);
            const int local_pixel = warp * kWarpLocalPixels + local_row;
            const int c0 = c_base + a_fragment_col_base(reg_idx, lane);
            a_reg[reg_idx] = load_conv1_warp_mma_a_u32(
                input, frame, in_slices, height, width, tile_y, tile_x, local_pixel, c0);
        }

        for (int group = 0; group < OutGroups; ++group) {
            uint32_t b_regs[4];
            const int group_out_base = out_base + group * kFp8ChannelsPerSlice;
            for (int reg_idx = 0; reg_idx < 4; ++reg_idx) {
                const int c0 = c_base + b_fragment_k_base(reg_idx, lane);
                const int out_channel = group_out_base + b_fragment_col(reg_idx, lane);
                b_regs[reg_idx] = load_conv1_warp_mma_b_u32(
                    weight, out_channel, c0, in_channels, out_channels);
            }

            uint2 b0{b_regs[0], b_regs[1]};
            uint2 c0{acc[group][0], acc[group][1]};
            uint2 d0 = mma_f8e4m3_f16(a, b0, c0);
            acc[group][0] = d0.x;
            acc[group][1] = d0.y;

            uint2 b1{b_regs[2], b_regs[3]};
            uint2 c1{acc[group][2], acc[group][3]};
            uint2 d1 = mma_f8e4m3_f16(a, b1, c1);
            acc[group][2] = d1.x;
            acc[group][3] = d1.y;
        }
    }

    const int out_slices = out_channels / kFp8ChannelsPerSlice;
    for (int group = 0; group < OutGroups; ++group) {
        const int group_out_base = out_base + group * kFp8ChannelsPerSlice;
        for (int reg_idx = 0; reg_idx < 4; ++reg_idx) {
            const int local_row = c_fragment_row(reg_idx, lane);
            const int local_pixel = warp * kWarpLocalPixels + local_row;
            const int local_y = local_pixel / kWarpTileX;
            const int local_x = local_pixel - local_y * kWarpTileX;
            const int y = tile_y * kWarpTileY + local_y;
            const int x = tile_x * kWarpTileX + local_x;
            if (frame >= frames || y >= height || x >= width) {
                continue;
            }
            half2 hv = *reinterpret_cast<half2*>(&acc[group][reg_idx]);
            const int c_pair = c_fragment_col_base(reg_idx, lane);
            const int out_channel0 = group_out_base + c_pair;
            const int out_channel1 = out_channel0 + 1;
            if (out_channel0 < out_channels) {
                const int out_slice = out_channel0 / kFp8ChannelsPerSlice;
                const int out_lane = out_channel0 - out_slice * kFp8ChannelsPerSlice;
                const size_t out_idx =
                    (((static_cast<size_t>(frame) * out_slices + out_slice) * height + y) * width + x) *
                        kFp8ChannelsPerSlice + out_lane;
                float value = __half2float(hv.x);
                if (bias != nullptr) {
                    const float out_s = fmaxf(
                        scale_to_float(output_scale, output_scale_count == 1 ? 0 : out_channel0),
                        1.0e-12f);
                    value += __half2float(bias[out_channel0]) / out_s;
                }
                if (relu) {
                    value = fmaxf(value, 0.0f);
                }
                output[out_idx] = encode_e4m3(value);
            }
            if (out_channel1 < out_channels) {
                const int out_slice = out_channel1 / kFp8ChannelsPerSlice;
                const int out_lane = out_channel1 - out_slice * kFp8ChannelsPerSlice;
                const size_t out_idx =
                    (((static_cast<size_t>(frame) * out_slices + out_slice) * height + y) * width + x) *
                        kFp8ChannelsPerSlice + out_lane;
                float value = __half2float(hv.y);
                if (bias != nullptr) {
                    const float out_s = fmaxf(
                        scale_to_float(output_scale, output_scale_count == 1 ? 0 : out_channel1),
                        1.0e-12f);
                    value += __half2float(bias[out_channel1]) / out_s;
                }
                if (relu) {
                    value = fmaxf(value, 0.0f);
                }
                output[out_idx] = encode_e4m3(value);
            }
        }
    }
}

template <bool Causal, typename ScaleT, int OutTileChannels, bool ScaledEpilogue>
__global__ void __launch_bounds__(kWarpMmaThreads, 2) conv3_tin16_warp_mma_kernel(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ cache,
    const uint8_t* __restrict__ weight,
    uint8_t* __restrict__ output,
    const ScaleT* __restrict__ output_scale,
    const half* __restrict__ bias,
    int frames,
    int cache_frames,
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
    const int lane = lane_id();
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int tile_x = static_cast<int>(blockIdx.x);
    const int tile_y = static_cast<int>(blockIdx.y);
    static_assert(OutTileChannels % kFp8ChannelsPerSlice == 0, "output tile must be a multiple of TIN16 channels");
    constexpr int OutGroups = OutTileChannels / kFp8ChannelsPerSlice;
    const int out_blocks = (out_channels + OutTileChannels - 1) / OutTileChannels;
    const int frame = static_cast<int>(blockIdx.z) / out_blocks;
    const int out_block = static_cast<int>(blockIdx.z) - frame * out_blocks;
    const int out_base = out_block * OutTileChannels;
    const int in_channels = in_slices * kFp8ChannelsPerSlice;

    uint32_t acc[OutGroups][4];
    for (int group = 0; group < OutGroups; ++group) {
        for (int reg_idx = 0; reg_idx < 4; ++reg_idx) {
            acc[group][reg_idx] = 0;
        }
    }

    const int temporal_taps = Causal ? 3 : 1;
    for (int dt = 0; dt < temporal_taps; ++dt) {
        for (int r = 0; r < 3; ++r) {
            for (int s = 0; s < 3; ++s) {
                for (int c_base = 0; c_base < in_channels; c_base += 32) {
                    uint4 a;
                    uint32_t* a_reg = reinterpret_cast<uint32_t*>(&a);
                    for (int reg_idx = 0; reg_idx < 4; ++reg_idx) {
                        const int local_row = a_fragment_row(reg_idx, lane);
                        const int local_pixel = warp * kWarpLocalPixels + local_row;
                        const int c0 = c_base + a_fragment_col_base(reg_idx, lane);
                        a_reg[reg_idx] = load_warp_mma_a_u32<Causal>(
                            input, cache, frame, frames, cache_frames, in_slices, height, width,
                            effective_height, effective_width, tile_y, tile_x, local_pixel,
                            c0, r, s, dt, stride, pad);
                    }

                    for (int group = 0; group < OutGroups; ++group) {
                        uint32_t b_regs[4];
                        const int group_out_base = out_base + group * kFp8ChannelsPerSlice;
                        for (int reg_idx = 0; reg_idx < 4; ++reg_idx) {
                            const int c0 = c_base + b_fragment_k_base(reg_idx, lane);
                            const int out_channel = group_out_base + b_fragment_col(reg_idx, lane);
                            b_regs[reg_idx] = load_warp_mma_b_u32<Causal>(
                                weight, out_channel, c0, r, s, dt, in_channels, out_channels);
                        }

                        uint2 b0{b_regs[0], b_regs[1]};
                        uint2 c0{acc[group][0], acc[group][1]};
                        uint2 d0 = mma_f8e4m3_f16(a, b0, c0);
                        acc[group][0] = d0.x;
                        acc[group][1] = d0.y;

                        uint2 b1{b_regs[2], b_regs[3]};
                        uint2 c1{acc[group][2], acc[group][3]};
                        uint2 d1 = mma_f8e4m3_f16(a, b1, c1);
                        acc[group][2] = d1.x;
                        acc[group][3] = d1.y;
                    }
                }
            }
        }
    }

    const int out_slices = out_channels / kFp8ChannelsPerSlice;
    for (int group = 0; group < OutGroups; ++group) {
        const int group_out_base = out_base + group * kFp8ChannelsPerSlice;
        for (int reg_idx = 0; reg_idx < 4; ++reg_idx) {
            const int local_row = c_fragment_row(reg_idx, lane);
            const int local_pixel = warp * kWarpLocalPixels + local_row;
            const int local_y = local_pixel / kWarpTileX;
            const int local_x = local_pixel - local_y * kWarpTileX;
            const int oy = tile_y * kWarpTileY + local_y;
            const int ox = tile_x * kWarpTileX + local_x;
            if (frame >= frames || oy >= out_height || ox >= out_width) {
                continue;
            }
            half2 hv = *reinterpret_cast<half2*>(&acc[group][reg_idx]);
            const int c_pair = c_fragment_col_base(reg_idx, lane);
            const int out_channel0 = group_out_base + c_pair;
            const int out_channel1 = out_channel0 + 1;
            if (out_channel0 < out_channels) {
                const int out_slice = out_channel0 / kFp8ChannelsPerSlice;
                const int out_lane = out_channel0 - out_slice * kFp8ChannelsPerSlice;
                const size_t out_idx =
                    (((static_cast<size_t>(frame) * out_slices + out_slice) * out_height + oy) * out_width + ox) *
                        kFp8ChannelsPerSlice + out_lane;
                uint8_t encoded;
                if constexpr (ScaledEpilogue) {
                    float value = __half2float(hv.x) *
                        scale_to_float(output_scale, output_scale_count == 1 ? 0 : out_channel0);
                    if (bias != nullptr) {
                        value += __half2float(bias[out_channel0]);
                    }
                    if (relu) {
                        value = fmaxf(value, 0.0f);
                    }
                    encoded = encode_e4m3(value);
                } else {
                    float value = __half2float(hv.x);
                    if (bias != nullptr) {
                        const float out_s = fmaxf(
                            scale_to_float(output_scale, output_scale_count == 1 ? 0 : out_channel0),
                            1.0e-12f);
                        value += __half2float(bias[out_channel0]) / out_s;
                    }
                    if (relu) {
                        value = fmaxf(value, 0.0f);
                    }
                    encoded = encode_e4m3(value);
                }
                output[out_idx] = encoded;
            }
            if (out_channel1 < out_channels) {
                const int out_slice = out_channel1 / kFp8ChannelsPerSlice;
                const int out_lane = out_channel1 - out_slice * kFp8ChannelsPerSlice;
                const size_t out_idx =
                    (((static_cast<size_t>(frame) * out_slices + out_slice) * out_height + oy) * out_width + ox) *
                        kFp8ChannelsPerSlice + out_lane;
                uint8_t encoded;
                if constexpr (ScaledEpilogue) {
                    float value = __half2float(hv.y) *
                        scale_to_float(output_scale, output_scale_count == 1 ? 0 : out_channel1);
                    if (bias != nullptr) {
                        value += __half2float(bias[out_channel1]);
                    }
                    if (relu) {
                        value = fmaxf(value, 0.0f);
                    }
                    encoded = encode_e4m3(value);
                } else {
                    float value = __half2float(hv.y);
                    if (bias != nullptr) {
                        const float out_s = fmaxf(
                            scale_to_float(output_scale, output_scale_count == 1 ? 0 : out_channel1),
                            1.0e-12f);
                        value += __half2float(bias[out_channel1]) / out_s;
                    }
                    if (relu) {
                        value = fmaxf(value, 0.0f);
                    }
                    encoded = encode_e4m3(value);
                }
                output[out_idx] = encoded;
            }
        }
    }
}

template <bool Causal, typename ScaleT, int InChannels, int OutChannels, bool ScaledEpilogue>
__global__ void __launch_bounds__(kWarpMmaThreads, 2) conv3_tin16_static_warp_mma_kernel(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ cache,
    const uint8_t* __restrict__ weight,
    uint8_t* __restrict__ output,
    const ScaleT* __restrict__ output_scale,
    const half* __restrict__ bias,
    int frames,
    int cache_frames,
    int height,
    int width,
    int effective_height,
    int effective_width,
    int out_height,
    int out_width,
    int output_scale_count,
    bool relu) {
    constexpr int OutGroups = OutChannels / kFp8ChannelsPerSlice;
    constexpr int OutSlices = OutChannels / kFp8ChannelsPerSlice;
    const int lane = lane_id();
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int tile_x = static_cast<int>(blockIdx.x);
    const int tile_y = static_cast<int>(blockIdx.y);
    const int frame = static_cast<int>(blockIdx.z);

    uint32_t acc[OutGroups][4];
    for (int group = 0; group < OutGroups; ++group) {
        for (int reg_idx = 0; reg_idx < 4; ++reg_idx) {
            acc[group][reg_idx] = 0;
        }
    }

    constexpr int TemporalTaps = Causal ? 3 : 1;
    for (int dt = 0; dt < TemporalTaps; ++dt) {
        for (int r = 0; r < 3; ++r) {
            for (int s = 0; s < 3; ++s) {
                for (int c_base = 0; c_base < InChannels; c_base += 32) {
                    uint4 a;
                    uint32_t* a_reg = reinterpret_cast<uint32_t*>(&a);
                    for (int reg_idx = 0; reg_idx < 4; ++reg_idx) {
                        const int local_row = a_fragment_row(reg_idx, lane);
                        const int local_pixel = warp * kWarpLocalPixels + local_row;
                        const int c0 = c_base + a_fragment_col_base(reg_idx, lane);
                        a_reg[reg_idx] = load_conv3_static_warp_mma_a_u32<Causal, InChannels>(
                            input, cache, frame, frames, cache_frames, height, width,
                            effective_height, effective_width, out_height, out_width,
                            tile_y, tile_x, local_pixel, c0, r, s, dt);
                    }

                    for (int group = 0; group < OutGroups; ++group) {
                        uint32_t b_regs[4];
                        const int group_out_base = group * kFp8ChannelsPerSlice;
                        for (int reg_idx = 0; reg_idx < 4; ++reg_idx) {
                            const int c0 = c_base + b_fragment_k_base(reg_idx, lane);
                            const int out_channel = group_out_base + b_fragment_col(reg_idx, lane);
                            b_regs[reg_idx] = load_conv3_static_warp_mma_b_u32<Causal, InChannels, OutChannels>(
                                weight, out_channel, c0, r, s, dt);
                        }

                        uint2 b0{b_regs[0], b_regs[1]};
                        uint2 c0{acc[group][0], acc[group][1]};
                        uint2 d0 = mma_f8e4m3_f16(a, b0, c0);
                        acc[group][0] = d0.x;
                        acc[group][1] = d0.y;

                        uint2 b1{b_regs[2], b_regs[3]};
                        uint2 c1{acc[group][2], acc[group][3]};
                        uint2 d1 = mma_f8e4m3_f16(a, b1, c1);
                        acc[group][2] = d1.x;
                        acc[group][3] = d1.y;
                    }
                }
            }
        }
    }

    for (int group = 0; group < OutGroups; ++group) {
        const int group_out_base = group * kFp8ChannelsPerSlice;
        for (int reg_idx = 0; reg_idx < 4; ++reg_idx) {
            const int local_row = c_fragment_row(reg_idx, lane);
            const int local_pixel = warp * kWarpLocalPixels + local_row;
            const int local_y = local_pixel / kWarpTileX;
            const int local_x = local_pixel - local_y * kWarpTileX;
            const int oy = tile_y * kWarpTileY + local_y;
            const int ox = tile_x * kWarpTileX + local_x;
            if (frame >= frames || oy >= out_height || ox >= out_width) {
                continue;
            }
            half2 hv = *reinterpret_cast<half2*>(&acc[group][reg_idx]);
            const int c_pair = c_fragment_col_base(reg_idx, lane);
            const int out_channel0 = group_out_base + c_pair;
            const int out_channel1 = out_channel0 + 1;
            if constexpr (ScaledEpilogue && std::is_same<ScaleT, half>::value) {
                if (!relu && out_channel1 < OutChannels) {
                    half2 scale2;
                    if (output_scale_count == 1) {
                        scale2 = __halves2half2(output_scale[0], output_scale[0]);
                    } else {
                        scale2 = *reinterpret_cast<const half2*>(output_scale + out_channel0);
                    }
                    half2 value2 = __hmul2(hv, scale2);
                    if (bias != nullptr) {
                        const half2 bias2 = *reinterpret_cast<const half2*>(bias + out_channel0);
                        value2 = __hadd2(value2, bias2);
                    }

                    const int out_slice = out_channel0 / kFp8ChannelsPerSlice;
                    const int out_lane = out_channel0 - out_slice * kFp8ChannelsPerSlice;
                    const size_t out_idx =
                        (((static_cast<size_t>(frame) * OutSlices + out_slice) * out_height + oy) * out_width + ox) *
                            kFp8ChannelsPerSlice + out_lane;
                    output[out_idx] = encode_e4m3_half(value2.x);
                    output[out_idx + 1] = encode_e4m3_half(value2.y);
                    continue;
                }
            }
            if (out_channel0 < OutChannels) {
                const int out_slice = out_channel0 / kFp8ChannelsPerSlice;
                const int out_lane = out_channel0 - out_slice * kFp8ChannelsPerSlice;
                const size_t out_idx =
                    (((static_cast<size_t>(frame) * OutSlices + out_slice) * out_height + oy) * out_width + ox) *
                        kFp8ChannelsPerSlice + out_lane;
                uint8_t encoded;
                if constexpr (ScaledEpilogue) {
                    float value = __half2float(hv.x) *
                        scale_to_float(output_scale, output_scale_count == 1 ? 0 : out_channel0);
                    if (bias != nullptr) {
                        value += __half2float(bias[out_channel0]);
                    }
                    if (relu) {
                        value = fmaxf(value, 0.0f);
                    }
                    encoded = encode_e4m3(value);
                } else {
                    float value = __half2float(hv.x);
                    if (bias != nullptr) {
                        const float out_s = fmaxf(
                            scale_to_float(output_scale, output_scale_count == 1 ? 0 : out_channel0),
                            1.0e-12f);
                        value += __half2float(bias[out_channel0]) / out_s;
                    }
                    if (relu) {
                        value = fmaxf(value, 0.0f);
                    }
                    encoded = encode_e4m3(value);
                }
                output[out_idx] = encoded;
            }
            if (out_channel1 < OutChannels) {
                const int out_slice = out_channel1 / kFp8ChannelsPerSlice;
                const int out_lane = out_channel1 - out_slice * kFp8ChannelsPerSlice;
                const size_t out_idx =
                    (((static_cast<size_t>(frame) * OutSlices + out_slice) * out_height + oy) * out_width + ox) *
                        kFp8ChannelsPerSlice + out_lane;
                uint8_t encoded;
                if constexpr (ScaledEpilogue) {
                    float value = __half2float(hv.y) *
                        scale_to_float(output_scale, output_scale_count == 1 ? 0 : out_channel1);
                    if (bias != nullptr) {
                        value += __half2float(bias[out_channel1]);
                    }
                    if (relu) {
                        value = fmaxf(value, 0.0f);
                    }
                    encoded = encode_e4m3(value);
                } else {
                    float value = __half2float(hv.y);
                    if (bias != nullptr) {
                        const float out_s = fmaxf(
                            scale_to_float(output_scale, output_scale_count == 1 ? 0 : out_channel1),
                            1.0e-12f);
                        value += __half2float(bias[out_channel1]) / out_s;
                    }
                    if (relu) {
                        value = fmaxf(value, 0.0f);
                    }
                    encoded = encode_e4m3(value);
                }
                output[out_idx] = encoded;
            }
        }
    }
}

template <typename ScaleT>
torch::Tensor dispatch_conv1_warp_mma(
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
    const int in_channels = in_slices * kFp8ChannelsPerSlice;
    TORCH_CHECK(in_channels % 32 == 0,
                "warp-MMA/TIN16 FP8 1x1 stage requires padded input channels divisible by 32, got ",
                in_channels);
    const int out_tile_channels = default_conv1_out_tile_channels(out_channels);
    const int out_blocks = (out_channels + out_tile_channels - 1) / out_tile_channels;
    const dim3 grid(
        static_cast<unsigned>((width + kWarpTileX - 1) / kWarpTileX),
        static_cast<unsigned>((height + kWarpTileY - 1) / kWarpTileY),
        static_cast<unsigned>(frames * out_blocks));
#define OMNIDREAMS_VAE_LAUNCH_CONV1_TIN16(OUT_TILE_CHANNELS)                                             \
    conv1_tin16_warp_mma_kernel<ScaleT, OUT_TILE_CHANNELS>                                        \
        <<<grid, kWarpMmaThreads, 0, at::cuda::getCurrentCUDAStream()>>>(                          \
            input.template data_ptr<uint8_t>(),                                                            \
            weight.template data_ptr<uint8_t>(),                                                           \
            output.template data_ptr<uint8_t>(),                                                           \
            reinterpret_cast<const ScaleT*>(output_scale.data_ptr()),                              \
            bias,                                                                                 \
            frames,                                                                               \
            in_slices,                                                                            \
            height,                                                                               \
            width,                                                                                \
            out_channels,                                                                         \
            static_cast<int>(output_scale.numel()),                                               \
            relu)
    switch (out_tile_channels) {
    case 96:
        OMNIDREAMS_VAE_LAUNCH_CONV1_TIN16(96);
        break;
    case 64:
        OMNIDREAMS_VAE_LAUNCH_CONV1_TIN16(64);
        break;
    default:
        OMNIDREAMS_VAE_LAUNCH_CONV1_TIN16(32);
        break;
    }
#undef OMNIDREAMS_VAE_LAUNCH_CONV1_TIN16
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <bool Causal, typename ScaleT, int InChannels, int OutChannels, bool ScaledEpilogue>
torch::Tensor dispatch_conv3_static_exact(
    torch::Tensor input,
    const uint8_t* cache_ptr,
    int cache_frames,
    torch::Tensor weight,
    torch::Tensor output_scale,
    const half* bias,
    int effective_height,
    int effective_width,
    int out_height,
    int out_width,
    bool relu) {
    const int frames = static_cast<int>(input.size(0));
    const int height = static_cast<int>(input.size(2));
    const int width = static_cast<int>(input.size(3));
    auto output = torch::empty(
        {frames, OutChannels / kFp8ChannelsPerSlice, out_height, out_width, kFp8ChannelsPerSlice},
        input.options());
    const dim3 grid(
        static_cast<unsigned>((out_width + kWarpTileX - 1) / kWarpTileX),
        static_cast<unsigned>((out_height + kWarpTileY - 1) / kWarpTileY),
        static_cast<unsigned>(frames));
    conv3_tin16_static_warp_mma_kernel<Causal, ScaleT, InChannels, OutChannels, ScaledEpilogue>
        <<<grid, kWarpMmaThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.template data_ptr<uint8_t>(),
            cache_ptr,
            weight.template data_ptr<uint8_t>(),
            output.template data_ptr<uint8_t>(),
            reinterpret_cast<const ScaleT*>(output_scale.data_ptr()),
            bias,
            frames,
            cache_frames,
            height,
            width,
            effective_height,
            effective_width,
            out_height,
            out_width,
            static_cast<int>(output_scale.numel()),
            relu);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <bool Causal, typename ScaleT, bool ScaledEpilogue>
torch::Tensor try_dispatch_conv3_static_exact(
    torch::Tensor input,
    const uint8_t* cache_ptr,
    int cache_frames,
    torch::Tensor weight,
    torch::Tensor output_scale,
    const half* bias,
    int effective_height,
    int effective_width,
    int out_height,
    int out_width,
    bool relu,
    bool* matched) {
    *matched = true;
    const int in_channels = static_cast<int>(input.size(1)) * kFp8ChannelsPerSlice;
    const int out_channels = static_cast<int>(weight.size(0));
#define OMNIDREAMS_VAE_TRY_CONV3_STATIC(IN_CHANNELS, OUT_CHANNELS)                                      \
    if (in_channels == (IN_CHANNELS) && out_channels == (OUT_CHANNELS)) {                         \
        return dispatch_conv3_static_exact<Causal, ScaleT, IN_CHANNELS, OUT_CHANNELS, ScaledEpilogue>( \
            input, cache_ptr, cache_frames, weight, output_scale, bias, effective_height,          \
            effective_width, out_height, out_width, relu);                                        \
    }
    OMNIDREAMS_VAE_TRY_CONV3_STATIC(32, 32);
    OMNIDREAMS_VAE_TRY_CONV3_STATIC(32, 64);
    OMNIDREAMS_VAE_TRY_CONV3_STATIC(64, 64);
    OMNIDREAMS_VAE_TRY_CONV3_STATIC(64, 96);
    OMNIDREAMS_VAE_TRY_CONV3_STATIC(96, 96);
    OMNIDREAMS_VAE_TRY_CONV3_STATIC(96, 32);
#undef OMNIDREAMS_VAE_TRY_CONV3_STATIC
    *matched = false;
    return torch::Tensor();
}

torch::Tensor checked_conv1_warp_mma(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    bool relu) {
    check_tin16_cuda_contiguous(input, "input");
    check_u8_cuda_contiguous(weight, "weight");
    check_scale_cuda_contiguous(output_scale, "output_scale");
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
    auto weight_c = weight.contiguous();
    auto out_s = output_scale.to(input.device(), output_scale.scalar_type(), false, true).contiguous();
    torch::Tensor bias_c;
    const half* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_c = bias.value().contiguous();
        check_half_cuda_contiguous(bias_c, "bias");
        TORCH_CHECK(bias_c.numel() == weight_c.size(0), "bias must have one entry per output channel");
        bias_ptr = half_ptr_const(bias_c);
    }
    return out_s.scalar_type() == at::kHalf
        ? dispatch_conv1_warp_mma<half>(input_c, weight_c, out_s, bias_ptr, relu)
        : dispatch_conv1_warp_mma<float>(input_c, weight_c, out_s, bias_ptr, relu);
}

template <bool Causal, typename ScaleT, bool ScaledEpilogue>
torch::Tensor dispatch_conv3_warp_mma(
    torch::Tensor input,
    torch::Tensor cache,
    const uint8_t* cache_ptr,
    int cache_frames,
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
    const int effective_height = height + (!Causal && pad_right_bottom ? 1 : 0);
    const int effective_width = width + (!Causal && pad_right_bottom ? 1 : 0);
    const int out_channels = static_cast<int>(weight.size(0));
    const int out_height = Causal ? height : ((effective_height + 2 * pad - 3) / stride + 1);
    const int out_width = Causal ? width : ((effective_width + 2 * pad - 3) / stride + 1);
    TORCH_CHECK(out_height > 0 && out_width > 0, "invalid output shape");

    auto output = torch::empty(
        {frames, out_channels / kFp8ChannelsPerSlice, out_height, out_width, kFp8ChannelsPerSlice},
        input.options());
    const int in_channels = in_slices * kFp8ChannelsPerSlice;
    TORCH_CHECK(in_channels % 32 == 0,
                "warp-MMA/TIN16 FP8 stage requires padded input channels divisible by 32, got ",
                in_channels);
    const int out_tile_channels = default_out_tile_channels(out_channels);
    const int out_blocks = (out_channels + out_tile_channels - 1) / out_tile_channels;
    const dim3 grid(
        static_cast<unsigned>((out_width + kWarpTileX - 1) / kWarpTileX),
        static_cast<unsigned>((out_height + kWarpTileY - 1) / kWarpTileY),
        static_cast<unsigned>(frames * out_blocks));
#define OMNIDREAMS_VAE_LAUNCH_CONV3_TIN16(OUT_TILE_CHANNELS)                                             \
    conv3_tin16_warp_mma_kernel<Causal, ScaleT, OUT_TILE_CHANNELS, ScaledEpilogue>                 \
        <<<grid, kWarpMmaThreads, 0, at::cuda::getCurrentCUDAStream()>>>(                           \
            input.template data_ptr<uint8_t>(),                                                             \
            cache_ptr,                                                                             \
            weight.template data_ptr<uint8_t>(),                                                            \
            output.template data_ptr<uint8_t>(),                                                            \
            reinterpret_cast<const ScaleT*>(output_scale.data_ptr()),                               \
            bias,                                                                                  \
            frames,                                                                                \
            cache_frames,                                                                          \
            in_slices,                                                                             \
            height,                                                                                \
            width,                                                                                 \
            effective_height,                                                                      \
            effective_width,                                                                       \
            out_height,                                                                            \
            out_width,                                                                             \
            out_channels,                                                                          \
            stride,                                                                                \
            pad,                                                                                   \
            static_cast<int>(output_scale.numel()),                                                \
            relu)
    switch (out_tile_channels) {
    case 96:
        OMNIDREAMS_VAE_LAUNCH_CONV3_TIN16(96);
        break;
    case 64:
        OMNIDREAMS_VAE_LAUNCH_CONV3_TIN16(64);
        break;
    case 32:
        OMNIDREAMS_VAE_LAUNCH_CONV3_TIN16(32);
        break;
    default:
        OMNIDREAMS_VAE_LAUNCH_CONV3_TIN16(16);
        break;
    }
#undef OMNIDREAMS_VAE_LAUNCH_CONV3_TIN16
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor normalize_optional_cache(
    c10::optional<torch::Tensor> cache,
    const torch::Tensor& input,
    const char* name,
    const uint8_t** cache_ptr,
    int* cache_frames) {
    *cache_ptr = nullptr;
    *cache_frames = 0;
    if (!(cache.has_value() && cache.value().defined())) {
        return torch::Tensor();
    }
    auto cache_c = cache.value().contiguous();
    check_tin16_cuda_contiguous(cache_c, name);
    TORCH_CHECK(cache_c.device() == input.device(), name, " must be on the same device as input");
    TORCH_CHECK(cache_c.size(1) == input.size(1) && cache_c.size(2) == input.size(2) &&
                    cache_c.size(3) == input.size(3),
                name, " shape must match input channel/spatial dimensions");
    TORCH_CHECK(cache_c.size(0) >= 0 && cache_c.size(0) <= 2, name, " must have 0, 1, or 2 frames");
    *cache_ptr = cache_c.template data_ptr<uint8_t>();
    *cache_frames = static_cast<int>(cache_c.size(0));
    return cache_c;
}

template <bool Causal, bool ScaledEpilogue = false>
torch::Tensor checked_conv3_warp_mma(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    int stride,
    int pad,
    bool relu,
    bool pad_right_bottom) {
    check_tin16_cuda_contiguous(input, "input");
    check_u8_cuda_contiguous(weight, "weight");
    check_scale_cuda_contiguous(output_scale, "output_scale");
    TORCH_CHECK(weight.dim() == 4 && weight.size(1) == 3 && weight.size(2) == 3,
                "weight must be [K,3,3,C], got ", weight.sizes());
    const int packed_factor = Causal ? 3 : 1;
    TORCH_CHECK(weight.size(3) == input.size(1) * kFp8ChannelsPerSlice * packed_factor,
                "weight C must match input channels");
    TORCH_CHECK(weight.size(0) % kFp8ChannelsPerSlice == 0,
                "weight K must be divisible by ", kFp8ChannelsPerSlice);
    TORCH_CHECK(output_scale.numel() == 1 || output_scale.numel() == weight.size(0),
                "output_scale must be scalar or have one entry per output channel");
    TORCH_CHECK(stride > 0 && pad >= 0, "invalid stride/pad");
    if constexpr (Causal) {
        TORCH_CHECK(stride == 1 && pad == 1 && !pad_right_bottom,
                    "causal warp-MMA stage expects stride=1, pad=1, pad_right_bottom=false");
    }
    at::cuda::CUDAGuard guard(input.device());
    auto input_c = input.contiguous();
    const uint8_t* cache_ptr = nullptr;
    int cache_frames = 0;
    auto cache_c = normalize_optional_cache(cache, input_c, "cache", &cache_ptr, &cache_frames);
    auto weight_c = weight.contiguous();
    auto out_s = output_scale.to(input.device(), output_scale.scalar_type(), false, true).contiguous();
    torch::Tensor bias_c;
    const half* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_c = bias.value().contiguous();
        check_half_cuda_contiguous(bias_c, "bias");
        TORCH_CHECK(bias_c.numel() == weight_c.size(0), "bias must have one entry per output channel");
        bias_ptr = half_ptr_const(bias_c);
    }
    const int frames = static_cast<int>(input_c.size(0));
    const int height = static_cast<int>(input_c.size(2));
    const int width = static_cast<int>(input_c.size(3));
    const int effective_height = height + (!Causal && pad_right_bottom ? 1 : 0);
    const int effective_width = width + (!Causal && pad_right_bottom ? 1 : 0);
    const int out_height = Causal ? height : ((effective_height + 2 * pad - 3) / stride + 1);
    const int out_width = Causal ? width : ((effective_width + 2 * pad - 3) / stride + 1);
    TORCH_CHECK(out_height > 0 && out_width > 0, "invalid output shape");
    const bool can_use_static_exact = Causal || (stride == 2 && pad == 0 && pad_right_bottom);
    if (can_use_static_exact && frames > 0) {
        bool matched = false;
        torch::Tensor static_out = out_s.scalar_type() == at::kHalf
            ? try_dispatch_conv3_static_exact<Causal, half, ScaledEpilogue>(
                  input_c, cache_ptr, cache_frames, weight_c, out_s, bias_ptr,
                  effective_height, effective_width, out_height, out_width, relu, &matched)
            : try_dispatch_conv3_static_exact<Causal, float, ScaledEpilogue>(
                  input_c, cache_ptr, cache_frames, weight_c, out_s, bias_ptr,
                  effective_height, effective_width, out_height, out_width, relu, &matched);
        if (matched) {
            return static_out;
        }
    }
    return out_s.scalar_type() == at::kHalf
        ? dispatch_conv3_warp_mma<Causal, half, ScaledEpilogue>(
              input_c, cache_c, cache_ptr, cache_frames, weight_c, out_s, bias_ptr,
              stride, pad, relu, pad_right_bottom)
        : dispatch_conv3_warp_mma<Causal, float, ScaledEpilogue>(
              input_c, cache_c, cache_ptr, cache_frames, weight_c, out_s, bias_ptr,
              stride, pad, relu, pad_right_bottom);
}

}  // namespace

torch::Tensor lightvae_fp8_conv1_tin16_warp_mma_prepared(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    bool relu) {
    return checked_conv1_warp_mma(input, weight, output_scale, bias, relu);
}

torch::Tensor lightvae_fp8_spatial_conv3_tin16_warp_mma_prepared(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    int stride,
    int pad,
    bool relu,
    bool pad_right_bottom) {
    return checked_conv3_warp_mma<false>(
        input, c10::nullopt, weight, output_scale, bias, stride, pad, relu, pad_right_bottom);
}

torch::Tensor lightvae_fp8_causal_conv3_tin16_warp_mma_prepared(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache,
    torch::Tensor weight,
    torch::Tensor output_scale,
    c10::optional<torch::Tensor> bias,
    bool relu) {
    return checked_conv3_warp_mma<true>(
        input, cache, weight, output_scale, bias, 1, 1, relu, false);
}

torch::Tensor lightvae_fp8_spatial_conv3_tin16_warp_mma_scaled_prepared(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor epilogue_scale,
    c10::optional<torch::Tensor> bias_scaled,
    int stride,
    int pad,
    bool relu,
    bool pad_right_bottom) {
    return checked_conv3_warp_mma<false, true>(
        input, c10::nullopt, weight, epilogue_scale, bias_scaled, stride, pad, relu, pad_right_bottom);
}

torch::Tensor lightvae_fp8_causal_conv3_tin16_warp_mma_scaled_prepared(
    torch::Tensor input,
    c10::optional<torch::Tensor> cache,
    torch::Tensor weight,
    torch::Tensor epilogue_scale,
    c10::optional<torch::Tensor> bias_scaled,
    bool relu) {
    return checked_conv3_warp_mma<true, true>(
        input, cache, weight, epilogue_scale, bias_scaled, 1, 1, relu, false);
}

}  // namespace omnidreams_singleview
