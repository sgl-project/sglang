/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// DeepSeek-V4 indexer Q RoPE + Hadamard + FP8 quant kernel, ported from the JIT kernel in
// python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh for the ROCm build.

#ifndef USE_ROCM
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#else
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cstdint>

#include "utils.h"

// ============================================================================
// Platform-compatible type aliases
// ============================================================================
#ifndef USE_ROCM
using bf16_t = __nv_bfloat16;
using bf16x2_t = __nv_bfloat162;
using fp8x2_e4m3_t = __nv_fp8x2_e4m3;
#else
using bf16_t = __hip_bfloat16;
using bf16x2_t = __hip_bfloat162;
using fp8x2_e4m3_t = uint16_t;
#ifndef __grid_constant__
#define __grid_constant__
#endif
#endif

// ============================================================================
// Utility helpers (inlined, no external header dependency)
// ============================================================================

static constexpr uint32_t kWarpSize = 32;

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (uint32_t mask = kWarpSize / 2; mask > 0; mask >>= 1)
    val = fmaxf(val, SGLANG_SHFL_XOR_SYNC(FULL_MASK, val, mask));
  return val;
}

// Aligned vector for coalesced memory access.
template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedVec {
  T data[N];
  __device__ __forceinline__ T& operator[](int i) {
    return data[i];
  }
  __device__ __forceinline__ T operator[](int i) const {
    return data[i];
  }
  __device__ __forceinline__ void load(const void* ptr, int64_t offset = 0) {
    *this = reinterpret_cast<const AlignedVec*>(ptr)[offset];
  }
  __device__ __forceinline__ void store(void* ptr, int64_t offset = 0) const {
    reinterpret_cast<AlignedVec*>(ptr)[offset] = *this;
  }
};

__device__ __forceinline__ float bf16_to_float(bf16_t v) {
  return __bfloat162float(v);
}

// ============================================================================
// FP8 E4M3 helpers (portable CUDA + HIP)
// ============================================================================

static constexpr float kFP8Max = 448.0f;

#ifndef USE_ROCM
__device__ __forceinline__ fp8x2_e4m3_t pack_fp8(float x, float y) {
  x = fmaxf(fminf(x, kFP8Max), -kFP8Max);
  y = fmaxf(fminf(y, kFP8Max), -kFP8Max);
  return __nv_fp8x2_e4m3(float2{x, y});
}
#elif HIP_FP8_TYPE_OCP && !HIP_FP8_TYPE_FNUZ
// gfx950/gfx12xx write OCP e4m3 natively, so take v_cvt_pk_fp8_f32 -- RNE, both lanes in
// one instruction. Not gfx942: hardware only converts to fnuz there, and this kernel writes
// E4M3FN on every arch (the indexer caller allocates float8_e4m3fn), so gfx942 keeps the
// software cast below. Testing FNUZ too because HIP sets both macros on the host pass and
// on targets outside its list. Clip rather than ask for __HIP_SATFINITE: the x2 fast path
// converts the value it was handed, not the clamped one (ROCm 7.2).
__device__ __forceinline__ fp8x2_e4m3_t pack_fp8(float x, float y) {
  const float2 v{fmaxf(fminf(x, kFP8Max), -kFP8Max), fmaxf(fminf(y, kFP8Max), -kFP8Max)};
  return __hip_cvt_float2_to_fp8x2(v, __HIP_NOSAT, __HIP_E4M3);
}
#else
// Software float -> FP8 E4M3 conversion for ROCm
__device__ __forceinline__ uint8_t cvt_float_to_fp8_e4m3(float val) {
  constexpr float kMax = kFP8Max;
  val = fmaxf(fminf(val, kMax), -kMax);
  if (val == 0.0f) return 0;

  uint32_t f32 = __float_as_uint(val);
  uint8_t sign = static_cast<uint8_t>((f32 >> 24) & 0x80u);
  f32 &= 0x7FFFFFFFu;

  int32_t exp32 = static_cast<int32_t>((f32 >> 23) & 0xFFu);
  uint32_t mant32 = f32 & 0x7FFFFFu;

  // FP8 E4M3 bias=7, FP32 bias=127, offset=120
  int32_t exp8 = exp32 - 120;

  if (exp8 <= 0) {
    mant32 |= 0x800000u;
    int32_t shift = 1 - exp8;
    if (shift > 24) return sign;
    uint32_t shifted = mant32 >> (20 + shift);
    uint32_t rbit = (shift <= 23) ? ((mant32 >> (19 + shift)) & 1u) : 0u;
    uint32_t sbit = (shift <= 23) ? ((mant32 & ((1u << (19 + shift)) - 1u)) != 0) : 0u;
    shifted += (rbit && (sbit || (shifted & 1u)));
    return sign | static_cast<uint8_t>(shifted & 0x7u);
  }
  if (exp8 >= 15) return sign | 0x7Eu;

  uint32_t mant3 = (mant32 >> 20) & 0x7u;
  uint32_t rbit = (mant32 >> 19) & 1u;
  uint32_t sbit = (mant32 & 0x7FFFFu) != 0;
  mant3 += (rbit && (sbit || (mant3 & 1u)));
  if (mant3 > 7) {
    mant3 = 0;
    exp8++;
    if (exp8 >= 15) return sign | 0x7Eu;
  }
  return sign | (static_cast<uint8_t>(exp8) << 3) | static_cast<uint8_t>(mant3);
}

__device__ __forceinline__ fp8x2_e4m3_t pack_fp8(float x, float y) {
  uint8_t x8 = cvt_float_to_fp8_e4m3(x);
  uint8_t y8 = cvt_float_to_fp8_e4m3(y);
  return static_cast<uint16_t>(x8) | (static_cast<uint16_t>(y8) << 8);
}
#endif

namespace {

constexpr uint32_t kFusedQBlockSize = 128;
constexpr uint32_t kFusedQNumWarps = kFusedQBlockSize / kWarpSize;

// ============================================================================
// Fused Q Indexer RoPE + Hadamard + FP8 Quantization
// warp-per-(token, head), no norm, RoPE + Hadamard + fp8 act-quant.
// ============================================================================

struct FusedQIndexerRopeHadamardQuantParams {
  const void* __restrict__ q_input;
  void* __restrict__ q_fp8;
  const void* __restrict__ weight;
  float* __restrict__ weights_out;
  float weight_scale;
  const float* __restrict__ freqs_cis;
  const int32_t* __restrict__ positions;
  uint32_t batch_size;
  uint32_t num_heads;
};

__global__ __launch_bounds__(kFusedQBlockSize, 16) void fused_q_indexer_rope_hadamard_quant_kernel(
    const __grid_constant__ FusedQIndexerRopeHadamardQuantParams params) {
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kRopeDim = 64;
  constexpr int64_t kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  static_assert(kHeadDim == kWarpSize * kVecSize);

  using Storage = AlignedVec<bf16_t, kVecSize>;
  using Float4 = AlignedVec<float, kVecSize>;
  using OutStorage = AlignedVec<fp8x2_e4m3_t, 2>;

  const auto warp_id = threadIdx.x / kWarpSize;
  const auto lane_id = threadIdx.x % kWarpSize;
  const auto work_id = blockIdx.x * kFusedQNumWarps + warp_id;
  const bool is_rope_lane = lane_id >= kWarpSize - kRopeSize;

  const uint32_t total_works = params.batch_size * params.num_heads;
  if (work_id >= total_works) return;

  const uint32_t batch_id = work_id / params.num_heads;
  const auto input_ptr = static_cast<const bf16_t*>(params.q_input) + work_id * kHeadDim;
  const auto position = params.positions[batch_id];
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  Float4 data, freq;
  const float weight_val = bf16_to_float(static_cast<const bf16_t*>(params.weight)[work_id]);

  // Part 1: load (no norm).
  {
    Storage input_vec;
    input_vec.load(input_ptr, lane_id);
    if (is_rope_lane) freq.load(freqs_cis, lane_id - (kWarpSize - kRopeSize));
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] = bf16_to_float(input_vec[i]);
  }

  // Part 2: rope on rope lanes.
  if (is_rope_lane) {
    float x_r = data[0], x_i = data[1], y_r = data[2], y_i = data[3];
    float fxr = freq[0], fxi = freq[1], fyr = freq[2], fyi = freq[3];
    data[0] = x_r * fxr - x_i * fxi;
    data[1] = x_r * fxi + x_i * fxr;
    data[2] = y_r * fyr - y_i * fyi;
    data[3] = y_r * fyi + y_i * fyr;
  }

  // Part 3: 128-point Hadamard (2 local + 5 cross-lane stages).
  {
    {
      float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a1;
      data[1] = a0 - a1;
      data[2] = a2 + a3;
      data[3] = a2 - a3;
    }
    {
      float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
      data[0] = a0 + a2;
      data[1] = a1 + a3;
      data[2] = a0 - a2;
      data[3] = a1 - a3;
    }
#pragma unroll
    for (uint32_t mask = 1; mask < kWarpSize; mask <<= 1) {
#pragma unroll
      for (int i = 0; i < kVecSize; ++i) {
        float other = SGLANG_SHFL_XOR_SYNC_WIDTH(FULL_MASK, data[i], mask, kWarpSize);
        data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
      }
    }
    const float kHadamardScale = rsqrtf(static_cast<float>(kHeadDim));
#pragma unroll
    for (int i = 0; i < kVecSize; ++i)
      data[i] *= kHadamardScale;
  }

  // Part 4: per-warp FP8 quant + store.
  {
    float local_max = fabsf(data[0]);
#pragma unroll
    for (int i = 1; i < kVecSize; ++i)
      local_max = fmaxf(local_max, fabsf(data[i]));
    float abs_max = warp_reduce_max(local_max);
    float scale = fmaxf(1e-4f, abs_max) / kFP8Max;
    float inv_scale = 1.0f / scale;

    OutStorage result;
    result[0] = pack_fp8(data[0] * inv_scale, data[1] * inv_scale);
    result[1] = pack_fp8(data[2] * inv_scale, data[3] * inv_scale);

    auto out_row = static_cast<uint8_t*>(params.q_fp8) + work_id * kHeadDim;
    result.store(out_row, lane_id);
    params.weights_out[work_id] = weight_val * params.weight_scale * scale;
  }
}

}  // anonymous namespace

// ============================================================================
// Host-side launchers (PyTorch C++ extension API)
// ============================================================================

void dsv4_fused_q_indexer_rope_hadamard_quant(
    const at::Tensor& q_input,
    at::Tensor& q_fp8,
    const at::Tensor& weight,
    at::Tensor& weights_out,
    double weight_scale,
    const at::Tensor& freqs_cis,
    const at::Tensor& positions) {
  TORCH_CHECK(q_input.is_cuda(), "q_input must be a CUDA tensor");
  TORCH_CHECK(q_input.scalar_type() == at::ScalarType::BFloat16, "q_input must be bfloat16");
  TORCH_CHECK(q_input.dim() == 3, "q_input must be 3D: (B, H, D)");

  const int64_t B = q_input.size(0);
  const int64_t H = q_input.size(1);
  constexpr int64_t kHeadDim = 128;
  TORCH_CHECK(q_input.size(2) == kHeadDim, "q_input head_dim must be 128 for indexer");
  TORCH_CHECK(
      q_input.stride(2) == 1 && q_input.stride(1) == kHeadDim, "q_input must be contiguous in (head, elem) dims");
  TORCH_CHECK(q_input.stride(0) == H * kHeadDim, "q_input must be contiguous (B, H, D)");
  TORCH_CHECK(q_fp8.stride(0) == H * kHeadDim, "q_fp8 must be contiguous (B, H, D)");
  TORCH_CHECK(positions.scalar_type() == at::ScalarType::Int, "positions must be int32");

  if (B == 0) return;

  const auto stream = at::cuda::getCurrentCUDAStream(q_input.get_device());
  const auto params = FusedQIndexerRopeHadamardQuantParams{
      .q_input = q_input.data_ptr(),
      .q_fp8 = q_fp8.data_ptr(),
      .weight = weight.data_ptr(),
      .weights_out = weights_out.data_ptr<float>(),
      .weight_scale = static_cast<float>(weight_scale),
      .freqs_cis = freqs_cis.data_ptr<float>(),
      .positions = positions.data_ptr<int32_t>(),
      .batch_size = static_cast<uint32_t>(B),
      .num_heads = static_cast<uint32_t>(H),
  };
  const uint32_t total_works = static_cast<uint32_t>(B * H);
  const uint32_t num_blocks = CEILDIV(total_works, kFusedQNumWarps);

  fused_q_indexer_rope_hadamard_quant_kernel<<<num_blocks, kFusedQBlockSize, 0, stream>>>(params);
}
