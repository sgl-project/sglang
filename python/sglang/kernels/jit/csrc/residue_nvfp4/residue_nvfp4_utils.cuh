/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2026 Rong Shuo.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Residue NVFP4 quantization device helpers.
//
// Ported from ResInfer csrc/nvfp4/src/fp4/nvfp4_utils.cuh (itself adapted from
// vLLM/TensorRT-LLM NVFP4 quantization). Changes in this port:
//   - namespace vllm -> sglang::residue_nvfp4
//   - torch type-converter helpers dropped (the JIT wrapper templates on the
//     CUDA element type directly)
//   - launch-bounds macros renamed from VLLM_* to RESIDUE_NVFP4_*
//
// The header carries the device machinery for BOTH residue representations:
//   - mext_r1 (ratio 1.0): cvt_warp_fp16_to_fp4_mext_r1_fast / _fast16 and the
//     compute_mext_r1_{base,residue}_row layout mapping
//   - k_ext (ratios 1/8, 2/8, 4/8): masked quantization + residue channel
//     append (cvt_warp_fp16_to_fp4 and the residue offset helpers)

#pragma once

#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <type_traits>

// ---------------------------------------------------------------------------
// Launch-bounds helpers (adapted from vLLM's launch_bounds_utils.h)
// ---------------------------------------------------------------------------

#ifndef RESIDUE_NVFP4_LAUNCH_BLOCKS_CAP
#define RESIDUE_NVFP4_LAUNCH_BLOCKS_CAP 4
#endif

// Compile-time estimate of max threads per SM for __launch_bounds__.
#ifndef RESIDUE_NVFP4_MAX_THREADS_PER_SM
#ifdef __CUDA_ARCH__
/* 1536 thr/SM: Ampere GA10x (sm_86/87), Ada (sm_89), GB20x consumer
   (sm_120/121), Thor (sm_101/sm_110) */
#if (__CUDA_ARCH__ == 860) || (__CUDA_ARCH__ == 870) || (__CUDA_ARCH__ == 890) || (__CUDA_ARCH__ == 1010) || \
    (__CUDA_ARCH__ == 1100) || (__CUDA_ARCH__ == 1200) || (__CUDA_ARCH__ == 1210)
#define RESIDUE_NVFP4_MAX_THREADS_PER_SM 1536
#elif (__CUDA_ARCH__ == 750)
#define RESIDUE_NVFP4_MAX_THREADS_PER_SM 1024
#else
/* 2048 thr/SM: GA100, Hopper, Blackwell datacenter (sm_100/103), fallback */
#define RESIDUE_NVFP4_MAX_THREADS_PER_SM 2048
#endif
#else
#define RESIDUE_NVFP4_MAX_THREADS_PER_SM 2048
#endif
#endif

#define RESIDUE_NVFP4_BLOCKS_DIV(VAL) (RESIDUE_NVFP4_MAX_THREADS_PER_SM / (VAL))
#define RESIDUE_NVFP4_CLAMP_BLOCKS_PER_SM(VAL) \
  (((VAL) <= 0) ? 1 : (((VAL) < RESIDUE_NVFP4_LAUNCH_BLOCKS_CAP) ? (VAL) : RESIDUE_NVFP4_LAUNCH_BLOCKS_CAP))
#define RESIDUE_NVFP4_BLOCKS_PER_SM(BLOCK_THREADS) \
  RESIDUE_NVFP4_CLAMP_BLOCKS_PER_SM(RESIDUE_NVFP4_BLOCKS_DIV(BLOCK_THREADS))

namespace sglang {
namespace residue_nvfp4 {

// Runtime helper mirroring the compile-time macro above.
static inline int runtime_blocks_per_sm(int block_threads) {
  int device = -1;
  cudaGetDevice(&device);
  int max_threads_per_sm = RESIDUE_NVFP4_MAX_THREADS_PER_SM;
  cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device);
  int blocks = (block_threads > 0) ? (max_threads_per_sm / block_threads) : 1;
  return RESIDUE_NVFP4_CLAMP_BLOCKS_PER_SM(blocks);
}

struct DeviceInfo {
  int multi_processor_count;
  int cc_major;
};

inline DeviceInfo query_device_info(int device_id) {
  // Cached per device: cudaDeviceGetAttribute costs ~1us and this sits on the
  // decode hot path.
  static DeviceInfo cache[64];
  static bool cached[64] = {};
  if (device_id >= 0 && device_id < 64 && cached[device_id]) {
    return cache[device_id];
  }
  DeviceInfo info{};
  cudaDeviceGetAttribute(&info.multi_processor_count, cudaDevAttrMultiProcessorCount, device_id);
  cudaDeviceGetAttribute(&info.cc_major, cudaDevAttrComputeCapabilityMajor, device_id);
  if (device_id >= 0 && device_id < 64) {
    cache[device_id] = info;
    cached[device_id] = true;
  }
  return info;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
constexpr int CVT_FP4_SF_VEC_SIZE = 16;
constexpr int CVT_FP4_ELTS_PER_UINT32 = 8;

__host__ __device__ __forceinline__ int round_up_int(int x, int n) {
  return ((x + n - 1) / n) * n;
}

// Match upstream swizzled-scale quant launch behavior so small-M serving
// iterations still initialize the full tiled SF layout and avoid collapsing to
// a single active CTA.
__host__ __device__ __forceinline__ int computeEffectiveRows(int m) {
  constexpr int ROW_TILE = 128;
  return round_up_int(m, ROW_TILE);
}

// Residue configurations - template-based for multi-ratio support.
// RESIDUE_PER_8_ELTS values: 1 (12.5%), 2 (25%), 4 (50%).
template <int RESIDUE_PER_8_ELTS>
constexpr int get_residue_per_thread() {
  return RESIDUE_PER_8_ELTS * (CVT_FP4_ELTS_PER_THREAD / 8);
}

static_assert(CVT_FP4_ELTS_PER_THREAD == 8, "Current implementation assumes 8 elements per thread");

// Helper function to compute warp mask for a contiguous group of threads.
// THREADS_PER_GROUP must be 1, 2, 4, 8, 16, or 32.
template <int THREADS_PER_GROUP>
constexpr uint32_t compute_group_mask(int group_id) {
  return ((1u << THREADS_PER_GROUP) - 1u) << (group_id * THREADS_PER_GROUP);
}

// Warp group max reduction using shuffle.
template <int GROUP_THREADS>
__device__ __forceinline__ float warp_group_max(float v, int laneId) {
  int groupId = laneId / GROUP_THREADS;
  uint32_t mask = compute_group_mask<GROUP_THREADS>(groupId);
#pragma unroll
  for (int off = 1; off < GROUP_THREADS; off <<= 1) {
    v = fmaxf(v, __shfl_xor_sync(mask, v, off));
  }
  return v;
}

// Extract values from float2 array using sparse bitmask iteration.
template <int MAX_OUTPUT>
__device__ __forceinline__ int
extract_values_by_mask(const float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2], uint8_t mask, float output[MAX_OUTPUT]) {
  int idx = 0;
  uint8_t extractMask = mask;
  while (extractMask) {
    int i = __ffs(extractMask) - 1;
    int pairIdx = i >> 1;
    int offset = i & 1;
    output[idx++] = (offset == 0) ? fp2Vals[pairIdx].x : fp2Vals[pairIdx].y;
    extractMask &= extractMask - 1;
  }
  return idx;
}

// Get type2 from type or vice versa (applied to half and bfloat16).
template <typename T>
struct TypeConverter {
  using Type = half2;
};

template <>
struct TypeConverter<half2> {
  using Type = half;
};

template <>
struct TypeConverter<half> {
  using Type = half2;
};

template <>
struct TypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

// Define a 16 bytes packed data type.
template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[4];
};

template <>
struct PackedVec<__nv_fp8_e4m3> {
  __nv_fp8x2_e4m3 elts[8];
};

// Convert 8 float32 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0]),
        "f"(array[1]),
        "f"(array[2]),
        "f"(array[3]),
        "f"(array[4]),
        "f"(array[5]),
        "f"(array[6]),
        "f"(array[7]));
  return val;
}

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x),
        "f"(array[0].y),
        "f"(array[1].x),
        "f"(array[1].y),
        "f"(array[2].x),
        "f"(array[2].y),
        "f"(array[3].x),
        "f"(array[3].y));
  return val;
}

// Quantize 8 scaled fp32 values to e2m1 and dequantize them back in place:
//   in_out_buffer[i] initially holds scaled input (x, y)
//   after the call it holds the dequantized values
__device__ __forceinline__ uint32_t fp32x8_to_e2m1_with_dequant(float2 (&in_out_buffer)[4]) {
  uint32_t packed_u32;
  uint32_t f16x2_0, f16x2_1, f16x2_2, f16x2_3;

  asm volatile(
      "{\n"
      "  .reg .b8 b0, b1, b2, b3;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b0, %6,  %5;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b1, %8,  %7;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b2, %10, %9;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b3, %12, %11;\n"
      "  cvt.rn.f16x2.e2m1x2           %0, b0;\n"
      "  cvt.rn.f16x2.e2m1x2           %1, b1;\n"
      "  cvt.rn.f16x2.e2m1x2           %2, b2;\n"
      "  cvt.rn.f16x2.e2m1x2           %3, b3;\n"
      "  mov.b32 %4, {b0, b1, b2, b3};\n"
      "}\n"
      : "=r"(f16x2_0), "=r"(f16x2_1), "=r"(f16x2_2), "=r"(f16x2_3), "=r"(packed_u32)
      : "f"(in_out_buffer[0].x),
        "f"(in_out_buffer[0].y),
        "f"(in_out_buffer[1].x),
        "f"(in_out_buffer[1].y),
        "f"(in_out_buffer[2].x),
        "f"(in_out_buffer[2].y),
        "f"(in_out_buffer[3].x),
        "f"(in_out_buffer[3].y));

  union U32Half2 {
    uint32_t u;
    __half2 h2;
  };

  auto residual_inplace = [&](uint32_t f16x2_u32, float2& inout) {
    U32Half2 v;
    v.u = f16x2_u32;
    inout.x = __half2float(__low2half(v.h2));
    inout.y = __half2float(__high2half(v.h2));
  };

  residual_inplace(f16x2_0, in_out_buffer[0]);
  residual_inplace(f16x2_1, in_out_buffer[1]);
  residual_inplace(f16x2_2, in_out_buffer[2]);
  residual_inplace(f16x2_3, in_out_buffer[3]);

  return packed_u32;
}

// Same as above with separate input and output buffers:
//   input_buffer[i] holds the scaled values (preserved)
//   dequant_buffer[i] receives the dequantized values
__device__ __forceinline__ uint32_t
fp32x8_to_e2m1_separate(const float2 (&input_buffer)[4], float2 (&dequant_buffer)[4]) {
  uint32_t packed_u32;
  uint32_t f16x2_0, f16x2_1, f16x2_2, f16x2_3;

  asm volatile(
      "{\n"
      "  .reg .b8 b0, b1, b2, b3;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b0, %6,  %5;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b1, %8,  %7;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b2, %10, %9;\n"
      "  cvt.rn.satfinite.e2m1x2.f32   b3, %12, %11;\n"
      "  cvt.rn.f16x2.e2m1x2           %0, b0;\n"
      "  cvt.rn.f16x2.e2m1x2           %1, b1;\n"
      "  cvt.rn.f16x2.e2m1x2           %2, b2;\n"
      "  cvt.rn.f16x2.e2m1x2           %3, b3;\n"
      "  mov.b32 %4, {b0, b1, b2, b3};\n"
      "}\n"
      : "=r"(f16x2_0), "=r"(f16x2_1), "=r"(f16x2_2), "=r"(f16x2_3), "=r"(packed_u32)
      : "f"(input_buffer[0].x),
        "f"(input_buffer[0].y),
        "f"(input_buffer[1].x),
        "f"(input_buffer[1].y),
        "f"(input_buffer[2].x),
        "f"(input_buffer[2].y),
        "f"(input_buffer[3].x),
        "f"(input_buffer[3].y));

  union U32Half2 {
    uint32_t u;
    __half2 h2;
  };

  auto write_dequant = [&](uint32_t f16x2_u32, float2& out) {
    U32Half2 v;
    v.u = f16x2_u32;
    out.x = __half2float(__low2half(v.h2));
    out.y = __half2float(__high2half(v.h2));
  };

  write_dequant(f16x2_0, dequant_buffer[0]);
  write_dequant(f16x2_1, dequant_buffer[1]);
  write_dequant(f16x2_2, dequant_buffer[2]);
  write_dequant(f16x2_3, dequant_buffer[3]);

  return packed_u32;
}

// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// Compute scale factor value from max absolute value.
// Formula: SF = SFScaleVal * (maxVal / 6.0)
__device__ __forceinline__ float compute_sf_from_max(float maxVal, float SFScaleVal) {
  return SFScaleVal * (maxVal * reciprocal_approximate_ftz(6.0f));
}

// Encode scale factor to FP8 (UE8M0 or E4M3) and return the quantized SF as
// float. Side effect: writes the FP8 byte to *out_fp8 if non-null.
template <bool UE8M0_SF>
__device__ __forceinline__ float encode_sf_to_fp8(float SFValue, uint8_t* out_fp8) {
  uint8_t fp8;
  if constexpr (UE8M0_SF) {
    uint32_t tmp = reinterpret_cast<uint32_t&>(SFValue) >> 23;
    fp8 = tmp & 0xff;
    reinterpret_cast<uint32_t&>(SFValue) = tmp << 23;
  } else {
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    reinterpret_cast<__nv_fp8_e4m3&>(fp8) = tmp;
    SFValue = float(tmp);
  }
  if (out_fp8) *out_fp8 = fp8;
  return SFValue;
}

// Compute output scale with precise numerical ordering for the FP8->FP4
// quantization path. DO NOT rewrite: this ordering is numerically chosen and
// must stay consistent across the main and residue quantization paths.
__device__ __forceinline__ float compute_output_scale_precise(float SFValue, float SFScaleVal) {
  return (SFValue != 0.0f) ? SFScaleVal * reciprocal_approximate_ftz(SFValue) : 0.0f;
}

// Shuffle-based collection of values within a group. Each thread in a group
// collects all values from its group members into outputChunk[4].
template <int RESIDUE_PER_THREAD>
__device__ __forceinline__ void shuffle_collect_group_float2(
    const float2 (&scaledValues)[(RESIDUE_PER_THREAD + 1) / 2], int groupFirstThread, float2 (&outputChunk)[4]) {
  if constexpr (RESIDUE_PER_THREAD == 1) {
    // Special case: 4 groups of 8 threads each.
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int sourceThread0 = groupFirstThread + i * 2;
      int sourceThread1 = groupFirstThread + i * 2 + 1;
      float val0 = __shfl_sync(__activemask(), scaledValues[0].x, sourceThread0);
      float val1 = __shfl_sync(__activemask(), scaledValues[0].x, sourceThread1);
      outputChunk[i] = make_float2(val0, val1);
    }
  } else {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int sourceThread = groupFirstThread + i / (RESIDUE_PER_THREAD / 2);
      int float2Offset = i % (RESIDUE_PER_THREAD / 2);
      unsigned long long tmp = __shfl_sync(
          __activemask(), reinterpret_cast<const unsigned long long&>(scaledValues[float2Offset]), sourceThread);
      outputChunk[i] = reinterpret_cast<float2&>(tmp);
    }
  }
}

// ---------------------------------------------------------------------------
// mext_r1 (ratio 1.0 M-extension) device path
// ---------------------------------------------------------------------------

// Specialized fast path for the M-extension ratio=1.0 layout.
//
// This path intentionally does not reuse the generic residue machinery:
// no sparse mask extraction, no warp-level residue gather, no extended-K
// residue address logic. Instead:
//   1. standard main quantization + QDQ
//   2. residue = original - dequant(main)
//   3. a second simple quantization pass for the residue row
template <class Type, bool UE8M0_SF = false>
__device__ __forceinline__ uint32_t cvt_warp_fp16_to_fp4_mext_r1_fast(
    PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout, uint32_t* residueDataOut, uint8_t* residueSFOut) {
  constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);

  // Main quantization (standard FP4 path).
  auto localMax = __habs2(vec.elts[0]);

#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  if constexpr (CVT_FP4_NUM_THREADS_PER_SF == 2) {
    localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  }
  float vecMax = float(__hmax(localMax.x, localMax.y));

  float SFValue = compute_sf_from_max(vecMax, SFScaleVal);
  SFValue = encode_sf_to_fp8<UE8M0_SF>(SFValue, SFout);
  float outputScale = compute_output_scale_precise(SFValue, SFScaleVal);

  float2 fp2ValsOriginal[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 mainScaled[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2ValsOriginal[i] = __half22float2(vec.elts[i]);
    } else {
      fp2ValsOriginal[i] = __bfloat1622float2(vec.elts[i]);
    }
    mainScaled[i].x = fp2ValsOriginal[i].x * outputScale;
    mainScaled[i].y = fp2ValsOriginal[i].y * outputScale;
  }

  // QDQ the main tensor so we can form the exact runtime residue.
  uint32_t e2m1Vec = fp32x8_to_e2m1_with_dequant(mainScaled);

  // Residue quantization (simple dense ratio=1.0 path).
  float inverseOutputScale = outputScale != 0.0f ? reciprocal_approximate_ftz(outputScale) : 0.0f;

  float2 residueVals[CVT_FP4_ELTS_PER_THREAD / 2];
  float localMaxResidue = 0.0f;

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    float deq_x = mainScaled[i].x * inverseOutputScale;
    float deq_y = mainScaled[i].y * inverseOutputScale;

    residueVals[i].x = fp2ValsOriginal[i].x - deq_x;
    residueVals[i].y = fp2ValsOriginal[i].y - deq_y;

    localMaxResidue = fmaxf(localMaxResidue, fmaxf(fabsf(residueVals[i].x), fabsf(residueVals[i].y)));
  }

  if constexpr (CVT_FP4_NUM_THREADS_PER_SF == 2) {
    localMaxResidue = fmaxf(localMaxResidue, __shfl_xor_sync(uint32_t(-1), localMaxResidue, 1));
  }

  float residueSFValue = compute_sf_from_max(localMaxResidue, SFScaleVal);
  residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
  float residueOutputScale = compute_output_scale_precise(residueSFValue, SFScaleVal);

  float2 residueScaled[CVT_FP4_ELTS_PER_THREAD / 2];
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    residueScaled[i].x = residueVals[i].x * residueOutputScale;
    residueScaled[i].y = residueVals[i].y * residueOutputScale;
  }

  if (residueDataOut) {
    *residueDataOut = fp32_vec_to_e2m1(residueScaled);
  }

  return e2m1Vec;
}

struct u32x2 {
  uint32_t lo, hi;
};

// Load 16 fp16/bf16 elements (one full SF vector, 32 bytes) in a single
// 256-bit global load. Blackwell (sm_100+) exposes ld.global.v8.b32
// (PTX ISA 8.8); older targets fall back to two 128-bit loads. `ptr` must be
// 32-byte aligned -- guaranteed by K % 16 == 0 plus torch's base alignment.
template <class Type>
__device__ __forceinline__ void load_packed_vec16(Type const* ptr, PackedVec<Type>& lo, PackedVec<Type>& hi) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  union {
    uint32_t u[8];
    struct {
      PackedVec<Type> lo, hi;
    } v;
  } tmp;
  asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=r"(tmp.u[0]),
                 "=r"(tmp.u[1]),
                 "=r"(tmp.u[2]),
                 "=r"(tmp.u[3]),
                 "=r"(tmp.u[4]),
                 "=r"(tmp.u[5]),
                 "=r"(tmp.u[6]),
                 "=r"(tmp.u[7])
               : "l"(ptr));
  lo = tmp.v.lo;
  hi = tmp.v.hi;
#else
  lo = reinterpret_cast<PackedVec<Type> const*>(ptr)[0];
  hi = reinterpret_cast<PackedVec<Type> const*>(ptr)[1];
#endif
}

// pack16 path for mext_r1:
//   - one thread owns 16 elements
//   - main/residue SF are each computed once over all 16 elements
//   - quantization reuses the validated 8-element QDQ helpers on the low/high
//     8-element half-chunks
template <class Type, bool UE8M0_SF = false>
__device__ __forceinline__ u32x2 cvt_warp_fp16_to_fp4_mext_r1_fast16(
    PackedVec<Type>& loVec,
    PackedVec<Type>& hiVec,
    float SFScaleVal,
    uint8_t* SFout,
    uint32_t* residueDataOut,
    uint8_t* residueSFOut) {
  static_assert(CVT_FP4_ELTS_PER_THREAD == 8, "pack16 path assumes the 8-element base helper.");

  auto localMax = __habs2(loVec.elts[0]);

#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    localMax = __hmax2(localMax, __habs2(loVec.elts[i]));
  }

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    localMax = __hmax2(localMax, __habs2(hiVec.elts[i]));
  }

  float vecMax = float(__hmax(localMax.x, localMax.y));

  float SFValue = compute_sf_from_max(vecMax, SFScaleVal);
  SFValue = encode_sf_to_fp8<UE8M0_SF>(SFValue, SFout);
  float outputScale = compute_output_scale_precise(SFValue, SFScaleVal);

  float2 loOriginal[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 hiOriginal[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 loScaled[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 hiScaled[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    if constexpr (std::is_same_v<Type, half>) {
      loOriginal[i] = __half22float2(loVec.elts[i]);
      hiOriginal[i] = __half22float2(hiVec.elts[i]);
    } else {
      loOriginal[i] = __bfloat1622float2(loVec.elts[i]);
      hiOriginal[i] = __bfloat1622float2(hiVec.elts[i]);
    }
    loScaled[i].x = loOriginal[i].x * outputScale;
    loScaled[i].y = loOriginal[i].y * outputScale;
    hiScaled[i].x = hiOriginal[i].x * outputScale;
    hiScaled[i].y = hiOriginal[i].y * outputScale;
  }

  u32x2 mainPacked;
  mainPacked.lo = fp32x8_to_e2m1_with_dequant(loScaled);
  mainPacked.hi = fp32x8_to_e2m1_with_dequant(hiScaled);

  float inverseOutputScale = outputScale != 0.0f ? reciprocal_approximate_ftz(outputScale) : 0.0f;

  float2 loResidue[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 hiResidue[CVT_FP4_ELTS_PER_THREAD / 2];
  float localMaxResidue = 0.0f;

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    float loDeqX = loScaled[i].x * inverseOutputScale;
    float loDeqY = loScaled[i].y * inverseOutputScale;
    float hiDeqX = hiScaled[i].x * inverseOutputScale;
    float hiDeqY = hiScaled[i].y * inverseOutputScale;

    loResidue[i].x = loOriginal[i].x - loDeqX;
    loResidue[i].y = loOriginal[i].y - loDeqY;
    hiResidue[i].x = hiOriginal[i].x - hiDeqX;
    hiResidue[i].y = hiOriginal[i].y - hiDeqY;

    localMaxResidue = fmaxf(localMaxResidue, fmaxf(fabsf(loResidue[i].x), fabsf(loResidue[i].y)));
    localMaxResidue = fmaxf(localMaxResidue, fmaxf(fabsf(hiResidue[i].x), fabsf(hiResidue[i].y)));
  }

  float residueSFValue = compute_sf_from_max(localMaxResidue, SFScaleVal);
  residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
  float residueOutputScale = compute_output_scale_precise(residueSFValue, SFScaleVal);

  float2 loResidueScaled[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 hiResidueScaled[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    loResidueScaled[i].x = loResidue[i].x * residueOutputScale;
    loResidueScaled[i].y = loResidue[i].y * residueOutputScale;
    hiResidueScaled[i].x = hiResidue[i].x * residueOutputScale;
    hiResidueScaled[i].y = hiResidue[i].y * residueOutputScale;
  }

  if (residueDataOut) {
    // Single 64-bit store (st.global.v2.b32); the residue offset is always an
    // even uint32 index because numCols % 16 == 0.
    *reinterpret_cast<uint2*>(residueDataOut) =
        make_uint2(fp32_vec_to_e2m1(loResidueScaled), fp32_vec_to_e2m1(hiResidueScaled));
  }

  return mainPacked;
}

// ---------------------------------------------------------------------------
// k_ext (selective residue) device path
// ---------------------------------------------------------------------------

// Swap-residue variant: the salient channels are quantized into the residue
// section and the (dequantization) residues take their place in the main
// section.
template <class Type, bool UE8M0_SF = false, int RESIDUE_PER_8_ELTS = 1>
__device__ uint32_t cvt_warp_fp16_to_fp4_swap(
    PackedVec<Type>& vec,
    float SFScaleVal,
    uint8_t* SFout,
    uint8_t mask,
    uint32_t* residueDataOut,
    uint8_t* salientSFOut) {
  constexpr int RESIDUE_PER_THREAD = get_residue_per_thread<RESIDUE_PER_8_ELTS>();

  // Quantization for salient channels.
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
  }

  float salient[RESIDUE_PER_THREAD > 0 ? RESIDUE_PER_THREAD : 1];
  extract_values_by_mask<RESIDUE_PER_THREAD>(fp2Vals, mask, salient);

  int laneId = threadIdx.x % 32;

  // Phase 1: compute shared SF for every 16 salient values.
  float localMaxSalient = 0.0f;
#pragma unroll
  for (int i = 0; i < RESIDUE_PER_THREAD; i++) {
    localMaxSalient = fmaxf(localMaxSalient, fabsf(salient[i]));
  }

  constexpr int THREADS_PER_SF_GROUP = 16 / RESIDUE_PER_THREAD;

  float sfGroupMaxSalient = warp_group_max<THREADS_PER_SF_GROUP>(localMaxSalient, laneId);
  float salientSFValue = SFScaleVal * (sfGroupMaxSalient * reciprocal_approximate_ftz(6.0f));
  salientSFValue = encode_sf_to_fp8<UE8M0_SF>(salientSFValue, salientSFOut);

  float salientOutputScale = compute_output_scale_precise(salientSFValue, SFScaleVal);

  // Prepare salient values for shuffle (handle odd RESIDUE_PER_THREAD).
  float2 shuffleSalient[(RESIDUE_PER_THREAD + 1) / 2];
#pragma unroll
  for (int i = 0; i < (RESIDUE_PER_THREAD + 1) / 2; i++) {
    float val0 = (i * 2 < RESIDUE_PER_THREAD) ? salient[i * 2] : 0.0f;
    float val1 = (i * 2 + 1 < RESIDUE_PER_THREAD) ? salient[i * 2 + 1] : 0.0f;
    shuffleSalient[i] = make_float2(val0, val1);
  }

  float2 residue[(RESIDUE_PER_THREAD + 1) / 2];
  float2 salientChunk[4];

  constexpr int numThreadsPerGroup = 8 / RESIDUE_PER_THREAD;
  int groupId = laneId / numThreadsPerGroup;
  int threadInGroup = laneId % numThreadsPerGroup;
  int groupFirstThread = groupId * numThreadsPerGroup;

  // Collect UNSCALED salient values via shuffle (residue computation must not
  // pick up scale round-trip error).
  float2 unscaledSalientChunk[4];
  shuffle_collect_group_float2<RESIDUE_PER_THREAD>(shuffleSalient, groupFirstThread, unscaledSalientChunk);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    salientChunk[i] =
        make_float2(unscaledSalientChunk[i].x * salientOutputScale, unscaledSalientChunk[i].y * salientOutputScale);
  }

  // Quantize salient with separate buffers to preserve the originals.
  float2 dequantChunk[4];
  uint32_t quantizedChunk = fp32x8_to_e2m1_separate(salientChunk, dequantChunk);

  if (residueDataOut) {
    *residueDataOut = quantizedChunk;
  }
  // Compute residue: original - dequant.
  float inverseSalientOutputScale = salientOutputScale != 0 ? reciprocal_approximate_ftz(salientOutputScale) : 0.0f;
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    float deq_x = dequantChunk[i].x * inverseSalientOutputScale;
    float deq_y = dequantChunk[i].y * inverseSalientOutputScale;
    float orig_x = unscaledSalientChunk[i].x;
    float orig_y = unscaledSalientChunk[i].y;
    salientChunk[i].x = orig_x - deq_x;
    salientChunk[i].y = orig_y - deq_y;
  }

  // Each thread directly extracts its residue from salientChunk (all threads
  // in the group hold identical chunks after quantization).
  if constexpr (RESIDUE_PER_THREAD == 1) {
    int pairIdx = threadInGroup / 2;
    int elemIdx = threadInGroup % 2;
    float val = (elemIdx == 0) ? salientChunk[pairIdx].x : salientChunk[pairIdx].y;
    residue[0] = make_float2(val, 0.0f);
  } else {
    constexpr int float2PerThread = RESIDUE_PER_THREAD / 2;
#pragma unroll
    for (int i = 0; i < float2PerThread; i++) {
      int idx = threadInGroup * float2PerThread + i;
      residue[i] = salientChunk[idx];
    }
  }

  // Main quantization: place residues back at the salient positions.
  int residueWriteIdx = 0;
  uint8_t tempMask = mask;
  while (tempMask) {
    int i = __ffs(tempMask) - 1;
    int pairIdx = i / 2;
    int offset = i % 2;
    float val;
    if constexpr (RESIDUE_PER_THREAD == 1) {
      val = residue[0].x;
    } else {
      int f2Idx = residueWriteIdx / 2;
      int f2Off = residueWriteIdx % 2;
      val = (f2Off == 0) ? residue[f2Idx].x : residue[f2Idx].y;
    }
    if (offset == 0) {
      fp2Vals[pairIdx].x = val;
    } else {
      fp2Vals[pairIdx].y = val;
    }
    residueWriteIdx++;
    tempMask &= tempMask - 1;
  }

  // Main SF computation in FLOAT precision (max BEFORE converting to
  // half/bfloat16 to avoid precision loss).
  float localMaxFloat = 0.0f;
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMaxFloat = fmaxf(localMaxFloat, fmaxf(fabsf(fp2Vals[i].x), fabsf(fp2Vals[i].y)));
  }
  localMaxFloat = fmaxf(localMaxFloat, __shfl_xor_sync(uint32_t(-1), localMaxFloat, 1));
  float vecMax = localMaxFloat;

  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  SFValue = encode_sf_to_fp8<UE8M0_SF>(SFValue, SFout);

  float outputScale = compute_output_scale_precise(SFValue, SFScaleVal);

  float2 mainVals[CVT_FP4_ELTS_PER_THREAD / 2];
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    mainVals[i].x = fp2Vals[i].x * outputScale;
    mainVals[i].y = fp2Vals[i].y * outputScale;
  }

  uint32_t e2m1Vec = fp32_vec_to_e2m1(mainVals);

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      vec.elts[i] = __float22half2_rn(fp2Vals[i]);
    } else {
      vec.elts[i] = __float22bfloat162_rn(fp2Vals[i]);
    }
  }

  return e2m1Vec;
}

// Standard (non-swap) k_ext path: quantize the full row, then quantize the
// dequantization residues of the salient channels into the extension.
template <class Type, bool UE8M0_SF = false, int RESIDUE_PER_8_ELTS = 1>
__device__ uint32_t cvt_warp_fp16_to_fp4(
    PackedVec<Type>& vec,
    float SFScaleVal,
    uint8_t* SFout,
    uint8_t mask,
    uint32_t* residueDataOut,
    uint8_t* residueSFOut) {
  constexpr int RESIDUE_PER_THREAD = get_residue_per_thread<RESIDUE_PER_8_ELTS>();
  // Main quantization.
  auto localMax = __habs2(vec.elts[0]);

#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  float vecMax = float(__hmax(localMax.x, localMax.y));

  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  SFValue = encode_sf_to_fp8<UE8M0_SF>(SFValue, SFout);

  float outputScale = compute_output_scale_precise(SFValue, SFScaleVal);

  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];
  float2 fp2ValsOriginal[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2ValsOriginal[i] = fp2Vals[i];
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Quantize and dequantize.
  uint32_t e2m1Vec = fp32x8_to_e2m1_with_dequant(fp2Vals);

  // Residue: original - dequantized (in original space).
  float inverseOutputScale = outputScale != 0 ? reciprocal_approximate_ftz(outputScale) : 0.0f;
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    fp2Vals[i].x *= inverseOutputScale;
    fp2Vals[i].y *= inverseOutputScale;
    fp2Vals[i].x = fp2ValsOriginal[i].x - fp2Vals[i].x;
    fp2Vals[i].y = fp2ValsOriginal[i].y - fp2Vals[i].y;
  }

  // Residue quantization.
  float residues[RESIDUE_PER_THREAD > 0 ? RESIDUE_PER_THREAD : 1];
  extract_values_by_mask<RESIDUE_PER_THREAD>(fp2Vals, mask, residues);

  int laneId = threadIdx.x % 32;

  // Phase 1: shared SF for every 16 residues.
  float localMaxResidue = 0.0f;
#pragma unroll
  for (int i = 0; i < RESIDUE_PER_THREAD; i++) {
    localMaxResidue = fmaxf(localMaxResidue, fabsf(residues[i]));
  }

  constexpr int THREADS_PER_SF_GROUP = 16 / RESIDUE_PER_THREAD;

  float sfGroupMaxResidue = warp_group_max<THREADS_PER_SF_GROUP>(localMaxResidue, laneId);
  float residueSFValue = SFScaleVal * (sfGroupMaxResidue * reciprocal_approximate_ftz(6.0f));
  residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);

  float residueOutputScale = compute_output_scale_precise(residueSFValue, SFScaleVal);

  float2 scaledResidues2[(RESIDUE_PER_THREAD + 1) / 2];
#pragma unroll
  for (int i = 0; i < (RESIDUE_PER_THREAD + 1) / 2; i++) {
    float val0 = (i * 2 < RESIDUE_PER_THREAD) ? residues[i * 2] * residueOutputScale : 0.0f;
    float val1 = (i * 2 + 1 < RESIDUE_PER_THREAD) ? residues[i * 2 + 1] * residueOutputScale : 0.0f;
    scaledResidues2[i] = make_float2(val0, val1);
  }

  // Two-stage shuffle for the residue extension:
  //   Stage 1: group-level shuffle - each group collects its residues
  //   Stage 2: inter-group shuffle - gather from group leaders to the writer
  //            threads (0-3)
  constexpr int numThreadsPerGroup = 8 / RESIDUE_PER_THREAD;
  int groupId = laneId / numThreadsPerGroup;
  int groupFirstThread = groupId * numThreadsPerGroup;

  float2 groupResidueChunk[4];
  shuffle_collect_group_float2<RESIDUE_PER_THREAD>(scaledResidues2, groupFirstThread, groupResidueChunk);

  float2 residueChunk2[4];
  int mySourceGroupLeader = laneId * numThreadsPerGroup;

#pragma unroll
  for (int i = 0; i < 4; i++) {
    unsigned long long tmp =
        __shfl_sync(__activemask(), reinterpret_cast<unsigned long long&>(groupResidueChunk[i]), mySourceGroupLeader);
    residueChunk2[i] = reinterpret_cast<float2&>(tmp);
  }

  uint32_t quantizedChunk = fp32_vec_to_e2m1(residueChunk2);

  if (residueDataOut) {
    *residueDataOut = quantizedChunk;
  }

  return e2m1Vec;
}

// ---------------------------------------------------------------------------
// Output offset helpers
// ---------------------------------------------------------------------------

// Residue data write address for the standard k_ext extension.
// Row-interleaved layout: [Row 0: base|residue][Row 1: base|residue]...
template <class ResidueType, int RESIDUE_PER_8_ELTS>
__device__ ResidueType*
cvt_residue_data_get_offset(int rowIdx, int colIdx, int numMainCols, int numTotalCols, ResidueType* outputStart) {
  constexpr int RESIDUE_PER_THREAD = get_residue_per_thread<RESIDUE_PER_8_ELTS>();
  constexpr int RESIDUES_PER_WARP = 32 * RESIDUE_PER_THREAD;
  constexpr int UINT32_PER_WARP = (RESIDUES_PER_WARP + 7) / 8;

  int warpId = colIdx / 32;
  int laneId = colIdx % 32;

  if (laneId < UINT32_PER_WARP) {
    int64_t uint32s_per_row = numTotalCols / CVT_FP4_ELTS_PER_UINT32;
    int64_t base_uint32s_per_row = numMainCols / CVT_FP4_ELTS_PER_UINT32;

    int64_t residueDataIdx = rowIdx * uint32s_per_row + base_uint32s_per_row + warpId * UINT32_PER_WARP + laneId;
    return outputStart + residueDataIdx;
  }
  return nullptr;
}

// Residue data write address for the SWAP extension: each group's first
// thread writes the quantized salient chunk.
template <class ResidueType, int RESIDUE_PER_8_ELTS>
__device__ ResidueType*
cvt_swap_residue_data_get_offset(int rowIdx, int colIdx, int numMainCols, int numTotalCols, ResidueType* outputStart) {
  constexpr int RESIDUE_PER_THREAD = get_residue_per_thread<RESIDUE_PER_8_ELTS>();
  constexpr int RESIDUES_PER_WARP = 32 * RESIDUE_PER_THREAD;
  constexpr int UINT32_PER_WARP = (RESIDUES_PER_WARP + 7) / 8;

  int warpId = colIdx / 32;
  int laneId = colIdx % 32;

  constexpr int numThreadsPerGroup = 8 / RESIDUE_PER_THREAD;
  int groupId = laneId / numThreadsPerGroup;
  int threadInGroup = laneId % numThreadsPerGroup;

  if (threadInGroup == 0 && groupId < UINT32_PER_WARP) {
    int64_t uint32s_per_row = numTotalCols / CVT_FP4_ELTS_PER_UINT32;
    int64_t base_uint32s_per_row = numMainCols / CVT_FP4_ELTS_PER_UINT32;

    int64_t residueDataIdx = rowIdx * uint32s_per_row + base_uint32s_per_row + warpId * UINT32_PER_WARP + groupId;
    return outputStart + residueDataIdx;
  }
  return nullptr;
}

// Tiled (swizzled) scale-factor offset.
// SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4 (kTile)]
template <class SFType>
__device__ inline int64_t compute_tiled_sf_offset(int32_t mIdx, int32_t kIdx, int32_t numTotalCols) {
  int32_t mTileIdx = mIdx / (32 * 4);
  constexpr int factor = CVT_FP4_SF_VEC_SIZE * 4;  // 64
  int32_t numKTiles = (numTotalCols + factor - 1) / factor;
  int64_t mTileStride = numKTiles * 32 * 4 * 4;

  int32_t kTileIdx = (kIdx / 4);
  int64_t kTileStride = 32 * 4 * 4;

  int32_t outerMIdx = (mIdx % 32);
  int64_t outerMStride = 4 * 4;

  int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
  int64_t innerMStride = 4;

  int32_t innerKIdx = (kIdx % 4);
  int64_t innerKStride = 1;

  return mTileIdx * mTileStride + kTileIdx * kTileStride + outerMIdx * outerMStride + innerMIdx * innerMStride +
         innerKIdx * innerKStride;
}

// mext_r1 layout row mapping. layoutMode: 0=concat, 1=row_pair,
// 3=concat_k (concat_k keeps one output row per token; callers handle its
// column placement separately).
__device__ __forceinline__ int32_t compute_mext_r1_base_row(int32_t rowIdx, int32_t numRows, int32_t layoutMode) {
  (void)numRows;
  if (layoutMode == 1) {
    return rowIdx * 2;
  }
  return rowIdx;
}

__device__ __forceinline__ int32_t compute_mext_r1_residue_row(int32_t rowIdx, int32_t numRows, int32_t layoutMode) {
  if (layoutMode == 1) {
    return rowIdx * 2 + 1;
  }
  return rowIdx + numRows;
}

template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(int rowIdx, int colIdx, int numTotalCols, SFType* SFout) {
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);

  // One pair of threads writes one SF to global memory.
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    // SF vector index (16 elements share one SF in the K dimension).
    int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    int64_t SFOffset = compute_tiled_sf_offset<SFType>(mIdx, kIdx, numTotalCols);
    return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
  }
  return nullptr;
}

// Residue scale factor write address for k_ext.
// One RESIDUE scale factor covers 16 residues == 128/residue_per_8 base
// channels; the TP sharding rules depend on that grouping (see
// residue_nvfp4/tp.py).
template <class ResidueSFType, int RESIDUE_PER_8_ELTS>
__device__ ResidueSFType*
cvt_residue_sf_get_offset(int rowIdx, int colIdx, int numTotalCols, int numMainCols, ResidueSFType* residueSFOut) {
  constexpr int RESIDUE_PER_THREAD = get_residue_per_thread<RESIDUE_PER_8_ELTS>();
  // Every 16 residues share 1 SF:
  //   ratio 1:8 -> 16 threads per SF group, 2:8 -> 8, 4:8 -> 4
  constexpr int THREADS_PER_SF_GROUP = 16 / RESIDUE_PER_THREAD;

  int warpId = colIdx / 32;
  int laneId = colIdx % 32;

  int sfGroupId = laneId / THREADS_PER_SF_GROUP;
  int sfGroupThreadId = laneId % THREADS_PER_SF_GROUP;

  // Only the first thread in each SF group writes.
  if (sfGroupThreadId != 0) return nullptr;

  int warpBaseResidueIdx = warpId * 32 * RESIDUE_PER_THREAD;
  int sfGroupResidueIdx = sfGroupId * 16;
  int residueElementIdx = numMainCols + warpBaseResidueIdx + sfGroupResidueIdx;

  int32_t kIdx = residueElementIdx / CVT_FP4_SF_VEC_SIZE;
  int32_t mIdx = rowIdx;
  int64_t SFOffset = compute_tiled_sf_offset<ResidueSFType>(mIdx, kIdx, numTotalCols);

  return residueSFOut + SFOffset;
}

}  // namespace residue_nvfp4
}  // namespace sglang
