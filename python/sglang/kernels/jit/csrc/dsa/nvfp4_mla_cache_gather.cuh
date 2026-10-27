/*
 * Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generation-phase NVFP4 MLA cache gather, adapted from TensorRT-LLM's
 * nvfp4MlaKvCacheGather kernel. The supported SGLang layout is fixed to a
 * 576-element main row with no residual: 288 packed E2M1 bytes and 36 E4M3
 * block-16 scales in separate pools.
 */
#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace sglang {
namespace nvfp4_mla {

constexpr int32_t kWarpSize = 32;
constexpr int32_t kWarpsPerBlock = 8;
constexpr int32_t kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int32_t kBlocksPerSm = 6;
constexpr int32_t kHeadDim = 576;
constexpr int32_t kPackedHeadDim = kHeadDim / 2;
constexpr int32_t kScalesPerToken = kHeadDim / 16;
constexpr int32_t kAsyncCopyBytes = 16;
constexpr int32_t kScaleCopyBytes = 4;
constexpr int32_t kStagingRows = 3;
constexpr int32_t kStagingRowBytes =
    ((kPackedHeadDim + kScalesPerToken + kAsyncCopyBytes - 1) / kAsyncCopyBytes) * kAsyncCopyBytes;

__device__ __forceinline__ void copy_async_16(void* destination, const void* source, uint32_t source_bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  uint32_t shared_address = static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :
               : "r"(shared_address), "l"(source), "r"(source_bytes));
#else
  if (source_bytes == 16) {
    *static_cast<uint4*>(destination) = *static_cast<const uint4*>(source);
  } else {
    *static_cast<uint4*>(destination) = make_uint4(0, 0, 0, 0);
  }
#endif
}

__device__ __forceinline__ void copy_async_4(void* destination, const void* source, uint32_t source_bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  uint32_t shared_address = static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
               :
               : "r"(shared_address), "l"(source), "r"(source_bytes));
#else
  *static_cast<uint32_t*>(destination) = source_bytes == 4 ? *static_cast<const uint32_t*>(source) : 0;
#endif
}

__device__ __forceinline__ void commit_async_copies() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;\n");
#endif
}

__device__ __forceinline__ void wait_async_copies() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group 0;\n");
#endif
}

__device__ __forceinline__ void wait_async_copies_keep_one() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group 1;\n");
#endif
}

__device__ __forceinline__ uint2 e2m1x4_to_fp16x4(uint16_t packed) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t low;
  uint32_t high;
  asm volatile(
      "{\n"
      ".reg .b8 lo_byte, hi_byte;\n"
      "mov.b16 {lo_byte, hi_byte}, %2;\n"
      "cvt.rn.f16x2.e2m1x2 %0, lo_byte;\n"
      "cvt.rn.f16x2.e2m1x2 %1, hi_byte;\n"
      "}\n"
      : "=r"(low), "=r"(high)
      : "h"(packed));
  return make_uint2(low, high);
#else
  return make_uint2(0, 0);
#endif
}

__device__ __forceinline__ float4 e2m1x4_to_float4(uint16_t packed) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint2 fp16 = e2m1x4_to_fp16x4(packed);
  float2 low = __half22float2(reinterpret_cast<const __half2&>(fp16.x));
  float2 high = __half22float2(reinterpret_cast<const __half2&>(fp16.y));
  return make_float4(low.x, low.y, high.x, high.y);
#else
  constexpr float magnitude[8] = {0.F, .5F, 1.F, 1.5F, 2.F, 3.F, 4.F, 6.F};
  float4 output;
  uint8_t code[4] = {
      static_cast<uint8_t>(packed & 0xFU),
      static_cast<uint8_t>((packed >> 4) & 0xFU),
      static_cast<uint8_t>((packed >> 8) & 0xFU),
      static_cast<uint8_t>((packed >> 12) & 0xFU)};
  float* values = &output.x;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float value = magnitude[code[i] & 0x7U];
    values[i] = (code[i] & 0x8U) == 0 ? value : -value;
  }
  return output;
#endif
}

__device__ __forceinline__ uint32_t float4_to_e4m3(float4 values) {
  uint32_t output;
  reinterpret_cast<__nv_fp8x4_e4m3&>(output) = __nv_fp8x4_e4m3(values);
  return output;
}

__device__ __forceinline__ uint32_t scaled_e2m1x4_to_e4m3(uint16_t packed, __nv_fp8_e4m3 scale, float global_scale) {
  // Keep the non-unit global-scale path in FP32 to match the reference
  // conversion's E4M3 rounding boundaries.
  if (global_scale != 1.F) {
    float4 values = e2m1x4_to_float4(packed);
    float combined_scale = static_cast<float>(scale) * global_scale;
    values.x *= combined_scale;
    values.y *= combined_scale;
    values.z *= combined_scale;
    values.w *= combined_scale;
    return float4_to_e4m3(values);
  }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint2 fp16 = e2m1x4_to_fp16x4(packed);
  __half2 scale2 = __half2half2(static_cast<__half>(scale));
  __half2 low = __hmul2(reinterpret_cast<const __half2&>(fp16.x), scale2);
  __half2 high = __hmul2(reinterpret_cast<const __half2&>(fp16.y), scale2);
  return __nv_fp8x4_e4m3(low, high).__x;
#else
  float4 values = e2m1x4_to_float4(packed);
  float scale_float = static_cast<float>(scale);
  values.x *= scale_float;
  values.y *= scale_float;
  values.z *= scale_float;
  values.w *= scale_float;
  return float4_to_e4m3(values);
#endif
}

__device__ __forceinline__ uint4 scaled_e2m1x16_to_e4m3(uint2 packed, __nv_fp8_e4m3 scale, float global_scale) {
  return make_uint4(
      scaled_e2m1x4_to_e4m3(static_cast<uint16_t>(packed.x), scale, global_scale),
      scaled_e2m1x4_to_e4m3(static_cast<uint16_t>(packed.x >> 16), scale, global_scale),
      scaled_e2m1x4_to_e4m3(static_cast<uint16_t>(packed.y), scale, global_scale),
      scaled_e2m1x4_to_e4m3(static_cast<uint16_t>(packed.y >> 16), scale, global_scale));
}

__device__ __forceinline__ void prefetch_row(
    uint8_t* staging, const uint8_t* data_pool, const __nv_fp8_e4m3* scale_pool, int32_t global_idx, int32_t lane) {
  bool valid = global_idx >= 0;
  const uint8_t* data_row = valid ? data_pool + static_cast<int64_t>(global_idx) * kPackedHeadDim : data_pool;
  const __nv_fp8_e4m3* scale_row = valid ? scale_pool + static_cast<int64_t>(global_idx) * kScalesPerToken : scale_pool;
  constexpr int32_t data_lanes = kPackedHeadDim / kAsyncCopyBytes;
  constexpr int32_t scale_lanes = kScalesPerToken / kScaleCopyBytes;
  if (lane < data_lanes) {
    copy_async_16(staging + lane * kAsyncCopyBytes, data_row + lane * kAsyncCopyBytes, valid ? kAsyncCopyBytes : 0U);
  } else if (lane < data_lanes + scale_lanes) {
    int32_t scale_lane = lane - data_lanes;
    copy_async_4(
        staging + kPackedHeadDim + scale_lane * kScaleCopyBytes,
        scale_row + scale_lane * kScaleCopyBytes,
        valid ? kScaleCopyBytes : 0U);
  }
  commit_async_copies();
}

__device__ __forceinline__ int32_t fetch_index(
    const int32_t* global_indices, int32_t* compact_indices, int64_t pair, int64_t num_pool_tokens, int32_t lane) {
  int32_t global_idx = -1;
  if (lane == 0) {
    global_idx = global_indices[pair];
    bool valid = global_idx >= 0 && static_cast<int64_t>(global_idx) < num_pool_tokens;
    compact_indices[pair] = valid ? static_cast<int32_t>(pair) : -1;
    global_idx = valid ? global_idx : -1;
  }
  return __shfl_sync(0xFFFFFFFFU, global_idx, 0);
}

__device__ __forceinline__ void dequantize_row(
    const uint8_t* staged_data,
    const __nv_fp8_e4m3* staged_scales,
    __nv_fp8_e4m3* output,
    int32_t output_idx,
    float global_scale,
    int32_t lane) {
  __nv_fp8_e4m3* output_row = output + static_cast<int64_t>(output_idx) * kHeadDim;
  for (int32_t base = 0; base < kScalesPerToken; base += kWarpSize) {
    int32_t group = base + lane;
    if (group < kScalesPerToken) {
      uint2 packed = *reinterpret_cast<const uint2*>(staged_data + static_cast<int64_t>(group) * 8);
      uint4 values = scaled_e2m1x16_to_e4m3(packed, staged_scales[group], global_scale);
      *reinterpret_cast<uint4*>(output_row + static_cast<int64_t>(group) * 16) = values;
    }
  }
}

__global__ __launch_bounds__(kThreadsPerBlock, kBlocksPerSm) void gather_kernel(
    const uint8_t* __restrict__ data_pool,
    const __nv_fp8_e4m3* __restrict__ scale_pool,
    const int32_t* __restrict__ global_indices,
    __nv_fp8_e4m3* __restrict__ output,
    int32_t* __restrict__ compact_indices,
    const float* __restrict__ global_scale_ptr,
    int64_t num_pairs,
    int64_t num_pool_tokens) {
  int32_t warp = threadIdx.x / kWarpSize;
  int32_t lane = threadIdx.x % kWarpSize;
  float global_scale = global_scale_ptr[0];
  __shared__ __align__(16) uint8_t staging[kStagingRows][kWarpsPerBlock][kStagingRowBytes];

  int64_t pair = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  int64_t pair_stride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;
  if (pair >= num_pairs) return;

  int32_t global_idx = fetch_index(global_indices, compact_indices, pair, num_pool_tokens, lane);
  int32_t stage = 0;
  prefetch_row(staging[stage][warp], data_pool, scale_pool, global_idx, lane);

  int64_t next_pair = pair + pair_stride;
  bool has_next = next_pair < num_pairs;
  int32_t next_global_idx = -1;
  if (has_next) {
    next_global_idx = fetch_index(global_indices, compact_indices, next_pair, num_pool_tokens, lane);
    prefetch_row(staging[1][warp], data_pool, scale_pool, next_global_idx, lane);
  }

  while (true) {
    if (has_next) {
      wait_async_copies_keep_one();
    } else {
      wait_async_copies();
    }
    __syncwarp();

    int64_t following_pair = next_pair + pair_stride;
    bool has_following = has_next && following_pair < num_pairs;
    int32_t following_global_idx = -1;
    if (has_following) {
      following_global_idx = fetch_index(global_indices, compact_indices, following_pair, num_pool_tokens, lane);
      int32_t prefetch_stage = stage == 0 ? 2 : stage - 1;
      prefetch_row(staging[prefetch_stage][warp], data_pool, scale_pool, following_global_idx, lane);
    }

    if (global_idx >= 0) {
      const auto* scales = reinterpret_cast<const __nv_fp8_e4m3*>(staging[stage][warp] + kPackedHeadDim);
      dequantize_row(staging[stage][warp], scales, output, static_cast<int32_t>(pair), global_scale, lane);
    }
    if (!has_next) break;

    pair = next_pair;
    global_idx = next_global_idx;
    next_pair = following_pair;
    next_global_idx = following_global_idx;
    has_next = has_following;
    stage = stage == kStagingRows - 1 ? 0 : stage + 1;
  }
}

struct GatherKernel {
  static void
  run(tvm::ffi::TensorView data_cache,
      tvm::ffi::TensorView scale_cache,
      tvm::ffi::TensorView physical_indices,
      tvm::ffi::TensorView output,
      tvm::ffi::TensorView compact_indices,
      tvm::ffi::TensorView global_scale) {
    using namespace host;
    auto pool_tokens = SymbolicSize{"pool_tokens"};
    auto rows = SymbolicSize{"rows"};
    auto topk = SymbolicSize{"topk"};
    auto output_rows = SymbolicSize{"output_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({pool_tokens, 1, kPackedHeadDim}).with_dtype<uint8_t>().with_device(device).verify(data_cache);
    TensorMatcher({pool_tokens, 1, kScalesPerToken}).with_dtype<uint8_t>().with_device(device).verify(scale_cache);
    TensorMatcher({rows, topk}).with_dtype<int32_t>().with_device(device).verify(physical_indices);
    TensorMatcher({output_rows, 1, kHeadDim}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output);
    TensorMatcher({rows, topk}).with_dtype<int32_t>().with_device(device).verify(compact_indices);
    TensorMatcher({1}).with_dtype<float>().with_device(device).verify(global_scale);

    int64_t num_pairs = rows.unwrap() * topk.unwrap();
    RuntimeCheck(num_pairs > 0, "NVFP4 MLA gather requires at least one index");
    RuntimeCheck(output_rows.unwrap() >= num_pairs, "NVFP4 MLA gather output is too small");
    RuntimeCheck(num_pairs <= INT32_MAX, "NVFP4 MLA gather exceeds int32 compact indexing");

    int32_t work_blocks = static_cast<int32_t>((num_pairs + kWarpsPerBlock - 1) / kWarpsPerBlock);
    static const uint32_t sm_count = host::runtime::get_sm_count(device.unwrap().device_id);
    int32_t blocks = std::max(1, std::min(work_blocks, static_cast<int32_t>(sm_count) * kBlocksPerSm));
    LaunchKernel(blocks, kThreadsPerBlock, device.unwrap())(
        gather_kernel,
        static_cast<const uint8_t*>(data_cache.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(scale_cache.data_ptr()),
        static_cast<const int32_t*>(physical_indices.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()),
        static_cast<int32_t*>(compact_indices.data_ptr()),
        static_cast<const float*>(global_scale.data_ptr()),
        num_pairs,
        pool_tokens.unwrap());
  }
};

}  // namespace nvfp4_mla
}  // namespace sglang
