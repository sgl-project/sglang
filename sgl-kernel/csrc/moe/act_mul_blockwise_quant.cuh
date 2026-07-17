// Copyright (c) 2025 SGLang Team.
// Fused SiLU+Mul+Blockwise-FP8-Quantization device utilities.
// Extracted & simplified from hpc-ops-main/src/utils/utils.cuh.
// No external dependencies beyond CUDA runtime + fp8 headers.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace act_mul_quant {

// ============================================================
// Constants
// ============================================================
static constexpr float kFP8E4M3Max = 448.0f;
static constexpr float kQuantEps = 1e-8f;
static constexpr int kGroupSize = 128;  // blockwise quant group
static constexpr int kElementsPerThread = 8;  // 8 bf16 = 128 bits

// ============================================================
// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
// With --use_fast_math, __expf compiles to hardware ex2.approx.ftz
// ============================================================
__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + __expf(-x));
}

// ============================================================
// Half-warp (16 lanes) reduce max via butterfly shuffle.
// All 16 lanes get the same result after reduction.
// Used for computing group-wise max over 128 elements
// (16 threads * 8 elements = 128).
// ============================================================
__device__ __forceinline__ float half_warp_reduce_max(float x) {
#pragma unroll
  for (int offset = 8; offset >= 1; offset >>= 1) {
    x = fmaxf(x, __shfl_xor_sync(0xFFFFFFFF, x, offset, 16));
  }
  return x;
}

// ============================================================
// Vectorized bf16 load: load 8 bf16 values (128 bits) and
// convert to 8 floats for computation.
// ============================================================
__device__ __forceinline__ void load_bf16x8_as_float(
    const __nv_bfloat16* ptr, float out[8]) {
  // Load 128 bits = 4 x bfloat162
  const uint4* ptr128 = reinterpret_cast<const uint4*>(ptr);
  uint4 data = *ptr128;
  const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&data);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 f2 = __bfloat1622float2(pairs[i]);
    out[2 * i] = f2.x;
    out[2 * i + 1] = f2.y;
  }
}

// ============================================================
// Vectorized fp8 store: convert 8 floats to 8 fp8_e4m3 values
// and store as 64 bits (2 x uint32).
// ============================================================
__device__ __forceinline__ void store_fp8x8(
    __nv_fp8_e4m3* ptr, const float vals[8]) {
  // Pack 8 fp8 values into 2 x __nv_fp8x4_e4m3 (each 32 bits)
  __nv_fp8x4_e4m3 pack0, pack1;

  // Convert float4 → fp8x4
  float4 f4_0 = make_float4(vals[0], vals[1], vals[2], vals[3]);
  float4 f4_1 = make_float4(vals[4], vals[5], vals[6], vals[7]);
  pack0 = __nv_fp8x4_e4m3(f4_0);
  pack1 = __nv_fp8x4_e4m3(f4_1);

  // Store as 64-bit (8 bytes)
  uint2* out_ptr = reinterpret_cast<uint2*>(ptr);
  uint2 packed;
  packed.x = *reinterpret_cast<uint32_t*>(&pack0);
  packed.y = *reinterpret_cast<uint32_t*>(&pack1);
  *out_ptr = packed;
}

}  // namespace act_mul_quant
