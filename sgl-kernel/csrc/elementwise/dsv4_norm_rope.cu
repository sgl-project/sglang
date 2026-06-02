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

// DeepSeek-V4 fused norm + RoPE kernels, ported from JIT kernel
// python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh
// to sgl-kernel AOT compilation with CUDA + HIP (ROCm) support.

#ifndef USE_ROCM
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#else
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
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

template <uint32_t kNumThreads = kWarpSize>
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (uint32_t mask = kNumThreads / 2; mask > 0; mask >>= 1)
    val += SGLANG_SHFL_XOR_SYNC(FULL_MASK, val, mask);
  return val;
}

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

__device__ __forceinline__ bf16_t float_to_bf16(float v) {
#ifndef USE_ROCM
  return __float2bfloat16_rn(v);
#else
  return __float2bfloat16(v);
#endif
}

// ============================================================================
// FP8 E4M3 helpers (portable CUDA + HIP)
// ============================================================================

// UE8M0 scale: round a positive float to the nearest power-of-two
// representable in UE8M0 (unsigned 8-bit exponent, no mantissa).
__device__ __forceinline__ int32_t cast_to_ue8m0(float x) {
  uint32_t u = __float_as_uint(x);
  int32_t exp = static_cast<int32_t>((u >> 23) & 0xFFu);
  uint32_t mant = u & 0x7FFFFFu;
  return exp + (mant != 0);
}

__device__ __forceinline__ float inv_scale_ue8m0(int32_t exp) {
  return __uint_as_float(static_cast<uint32_t>((127 + 127 - exp) << 23));
}

static constexpr float kFP8Max = 448.0f;

#ifndef USE_ROCM
__device__ __forceinline__ fp8x2_e4m3_t pack_fp8(float x, float y) {
  x = fmaxf(fminf(x, kFP8Max), -kFP8Max);
  y = fmaxf(fminf(y, kFP8Max), -kFP8Max);
  return __nv_fp8x2_e4m3(float2{x, y});
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

// ============================================================================
// Kernel 1: Fused Q Norm + RoPE
// warp-per-(token, head), rmsnorm-self (no weight) + RoPE + write to q_out.
// ============================================================================

namespace {

constexpr uint32_t kFusedQBlockSize = 128;
constexpr uint32_t kFusedQNumWarps = kFusedQBlockSize / kWarpSize;

constexpr uint32_t kFusedKBlockSize = 256;
constexpr uint32_t kFusedKNumWarps = kFusedKBlockSize / kWarpSize;

struct FusedQNormRopeParams {
  const void* __restrict__ q_input;
  void* __restrict__ q_output;
  const float* __restrict__ freqs_cis;
  const int32_t* __restrict__ positions;
  int64_t q_input_stride_batch;
  int64_t q_output_stride_batch;
  uint32_t batch_size;
  uint32_t num_q_heads;
  float eps;
};

// Compute the largest power-of-2 vec size that divides both kHeadDim and
// fits in 16 bytes, while also dividing kRopeDim.
template <int64_t kHeadDim, int64_t kRopeDim>
struct QKernelTraits {
  static constexpr int64_t kMaxVecSize = 16 / sizeof(bf16_t);  // 8
  // Use kRopeDim/kWarpSize (=2 for kRopeDim=64) as the vec size.
  // This guarantees kRopeDim % kVecSize == 0 and works for all head dims
  // that are multiples of kWarpSize*kVecSize.
  static constexpr int64_t kVecSize = kRopeDim / kWarpSize;  // 2
  static constexpr int64_t kLocalSize = kHeadDim / (kWarpSize * kVecSize);
  static constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  static_assert(kHeadDim % (kWarpSize * kVecSize) == 0);
  static_assert(kRopeDim % kVecSize == 0);
  static_assert(kRopeDim == kWarpSize * 2, "1 (real, imag) pair per lane");
};

template <int64_t kHeadDim, int64_t kRopeDim>
__global__ __launch_bounds__(kFusedQBlockSize, 16) void fused_q_norm_rope_kernel(
    const __grid_constant__ FusedQNormRopeParams params) {
  using Traits = QKernelTraits<kHeadDim, kRopeDim>;
  constexpr int64_t kVecSize = Traits::kVecSize;
  constexpr int64_t kLocalSize = Traits::kLocalSize;
  constexpr uint32_t kRopeSize = Traits::kRopeSize;

  using Storage = AlignedVec<bf16_t, kVecSize>;
  using Float2 = AlignedVec<float, 2>;

  const auto warp_id = threadIdx.x / kWarpSize;
  const auto lane_id = threadIdx.x % kWarpSize;
  const auto work_id = blockIdx.x * kFusedQNumWarps + warp_id;

  const uint32_t total_works = params.batch_size * params.num_q_heads;
  if (work_id >= total_works) return;

  const uint32_t batch_id = work_id / params.num_q_heads;
  const uint32_t head_id = work_id % params.num_q_heads;
  const auto input_ptr =
      static_cast<const bf16_t*>(params.q_input) + batch_id * params.q_input_stride_batch + head_id * kHeadDim;
  const auto output_ptr =
      static_cast<bf16_t*>(params.q_output) + batch_id * params.q_output_stride_batch + head_id * kHeadDim;
  const auto position = params.positions[batch_id];

  __shared__ Storage s_rope[kFusedQNumWarps][kRopeSize];

  // Prefetch freq pair.
  Float2 freq;
  freq.load(params.freqs_cis + position * kRopeDim, lane_id);

  // Part 1: rmsnorm-self (no weight).
  Storage input_vec[kLocalSize];
#pragma unroll
  for (int i = 0; i < kLocalSize; ++i) {
    input_vec[i].load(input_ptr, lane_id + i * kWarpSize);
  }

  float sum_of_squares = 0.0f;
#pragma unroll
  for (int i = 0; i < kLocalSize; ++i) {
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      float x = bf16_to_float(input_vec[i][j]);
      sum_of_squares += x * x;
    }
  }
  sum_of_squares = warp_reduce_sum(sum_of_squares);
  const float norm_factor = rsqrtf(sum_of_squares / static_cast<float>(kHeadDim) + params.eps);

#pragma unroll
  for (int i = 0; i < kLocalSize; ++i) {
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      float x = bf16_to_float(input_vec[i][j]);
      input_vec[i][j] = float_to_bf16(x * norm_factor);
    }
  }

  // Stash rope tail into shared memory; write nope tiles to gmem.
  const bool is_rope_lane = lane_id >= kWarpSize - kRopeSize;
#pragma unroll
  for (int i = 0; i < kLocalSize; ++i) {
    if (i == kLocalSize - 1 && is_rope_lane) {
      const auto rope_id = lane_id - (kWarpSize - kRopeSize);
      s_rope[warp_id][rope_id] = input_vec[i];
    } else {
      input_vec[i].store(output_ptr, lane_id + i * kWarpSize);
    }
  }
  __syncwarp();

  // Part 2: RoPE on all 32 lanes -- one (real, imag) bf16x2 pair per lane.
  auto elem_ptr = reinterpret_cast<bf16x2_t*>(&s_rope[warp_id][0]);
  bf16x2_t elem = elem_ptr[lane_id];
#ifndef USE_ROCM
  float2 elem_f = __bfloat1622float2(elem);
  float x_real = elem_f.x, x_imag = elem_f.y;
#else
  float x_real = __bfloat162float(elem.x), x_imag = __bfloat162float(elem.y);
#endif
  float freq_real = freq[0], freq_imag = freq[1];
  float rot_real = x_real * freq_real - x_imag * freq_imag;
  float rot_imag = x_real * freq_imag + x_imag * freq_real;
  bf16x2_t rotated = __float22bfloat162_rn(make_float2(rot_real, rot_imag));
  auto out_elem = reinterpret_cast<bf16x2_t*>(output_ptr + (kHeadDim - kRopeDim));
  out_elem[lane_id] = rotated;
}

// ============================================================================
// Kernel 2: Fused K Norm + RoPE + FlashMLA Store
// block-per-token, rmsnorm (with kv_weight) + RoPE + FP8 quantized store.
// ============================================================================

struct FusedKNormRopeFlashMLAParams {
  const void* __restrict__ kv;
  const void* __restrict__ kv_weight;
  const float* __restrict__ freqs_cis;
  const int32_t* __restrict__ positions;
  const int32_t* __restrict__ out_loc;
  uint8_t* __restrict__ kvcache;
  int64_t kv_stride_batch;
  uint32_t batch_size;
  float eps;
};

template <int64_t kHeadDim, int64_t kRopeDim, int32_t kPageBits>
__global__ __launch_bounds__(kFusedKBlockSize, 8) void fused_k_norm_rope_flashmla_kernel(
    const __grid_constant__ FusedKNormRopeFlashMLAParams params) {
  constexpr int64_t kVecSize = 2;
  constexpr uint32_t kRopeWarp = kFusedKNumWarps - 1;
  constexpr int64_t kPageBytes = ((584ll << kPageBits) + 575) / 576 * 576;
  static_assert(kHeadDim == kFusedKBlockSize * kVecSize);
  static_assert(kRopeDim == kWarpSize * kVecSize);

  using Storage = AlignedVec<bf16_t, kVecSize>;

  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpSize;
  const auto lane_id = tx % kWarpSize;
  const auto work_id = blockIdx.x;
  if (work_id >= params.batch_size) return;

  const auto input_ptr = static_cast<const bf16_t*>(params.kv) + work_id * params.kv_stride_batch;
  const auto position = params.positions[work_id];
  const auto out_loc = params.out_loc[work_id];
  const auto freqs_cis = params.freqs_cis + position * kRopeDim;

  AlignedVec<float, kVecSize> data, freq;

  // Part 1: norm with block-wide reduction.
  {
    __shared__ float partial_sums[kFusedKNumWarps];

    Storage input_vec, weight_vec;
    input_vec.load(input_ptr, tx);
    weight_vec.load(params.kv_weight, tx);
    if (warp_id == kRopeWarp) freq.load(freqs_cis, lane_id);

    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      float x = bf16_to_float(input_vec[i]);
      sum_of_squares += x * x;
    }
    const float warp_sum = warp_reduce_sum(sum_of_squares);
    if (lane_id == 0) partial_sums[warp_id] = warp_sum;
    __syncthreads();
    sum_of_squares = warp_reduce_sum<kFusedKNumWarps>(partial_sums[lane_id % kFusedKNumWarps]);
    const float norm_factor = rsqrtf(sum_of_squares / static_cast<float>(kHeadDim) + params.eps);

#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      float x = bf16_to_float(input_vec[i]);
      float w = bf16_to_float(weight_vec[i]);
      data[i] = x * norm_factor * w;
    }
  }

  const int32_t page = out_loc >> kPageBits;
  const int32_t offset = out_loc & ((1 << kPageBits) - 1);
  const auto page_ptr = params.kvcache + page * kPageBytes;
  const auto value_ptr = page_ptr + offset * 576;

  // Part 2: rope on last warp (BF16 store), per-warp UE8M0 quant + store on others.
  if (warp_id == kRopeWarp) {
    float x_real = data[0], x_imag = data[1];
    float freq_real = freq[0], freq_imag = freq[1];
    float rot_real = x_real * freq_real - x_imag * freq_imag;
    float rot_imag = x_real * freq_imag + x_imag * freq_real;
    bf16x2_t result = __float22bfloat162_rn(make_float2(rot_real, rot_imag));
    auto rope_ptr = value_ptr + 448;
    reinterpret_cast<bf16x2_t*>(rope_ptr)[lane_id] = result;
  } else {
    float x = data[0], y = data[1];
    float abs_max = warp_reduce_max(fmaxf(fabsf(x), fabsf(y)));
    float scale_raw = fmaxf(1e-4f, abs_max) / kFP8Max;
    int32_t scale_ue8m0 = cast_to_ue8m0(scale_raw);
    float inv_scale = inv_scale_ue8m0(scale_ue8m0);
    fp8x2_e4m3_t result = pack_fp8(x * inv_scale, y * inv_scale);
    auto scale_ptr = page_ptr + (576ll << kPageBits) + offset * 8;
    reinterpret_cast<fp8x2_e4m3_t*>(value_ptr)[tx] = result;
    if (lane_id == 0) static_cast<uint8_t*>(scale_ptr)[warp_id] = static_cast<uint8_t>(scale_ue8m0);
  }
}

// ============================================================================
// Kernel 3: Fused Q Indexer RoPE + Hadamard + FP8 Quantization
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

void dsv4_fused_q_norm_rope(
    const at::Tensor& q_input,
    at::Tensor& q_output,
    const at::Tensor& freqs_cis,
    const at::Tensor& positions,
    double eps) {
  TORCH_CHECK(q_input.is_cuda(), "q_input must be a CUDA tensor");
  TORCH_CHECK(q_output.is_cuda(), "q_output must be a CUDA tensor");
  TORCH_CHECK(q_input.scalar_type() == at::ScalarType::BFloat16, "q_input must be bfloat16");
  TORCH_CHECK(q_output.scalar_type() == at::ScalarType::BFloat16, "q_output must be bfloat16");
  TORCH_CHECK(q_input.dim() == 3, "q_input must be 3D: (B, H, D)");
  TORCH_CHECK(q_output.dim() == 3, "q_output must be 3D: (B, H, D)");
  TORCH_CHECK(positions.scalar_type() == at::ScalarType::Int, "positions must be int32");

  const int64_t B = q_input.size(0);
  const int64_t H = q_input.size(1);
  const int64_t D = q_input.size(2);
  TORCH_CHECK(
      q_output.size(0) == B && q_output.size(1) == H && q_output.size(2) == D, "q_output shape must match q_input");
  TORCH_CHECK(q_input.stride(2) == 1 && q_output.stride(2) == 1, "last dim must be contiguous");
  TORCH_CHECK(q_input.stride(1) == D && q_output.stride(1) == D, "head dim must be contiguous");

  if (B == 0) return;

  const auto stream = at::cuda::getCurrentCUDAStream(q_input.get_device());
  const auto params = FusedQNormRopeParams{
      .q_input = q_input.data_ptr(),
      .q_output = q_output.data_ptr(),
      .freqs_cis = freqs_cis.data_ptr<float>(),
      .positions = positions.data_ptr<int32_t>(),
      .q_input_stride_batch = q_input.stride(0),
      .q_output_stride_batch = q_output.stride(0),
      .batch_size = static_cast<uint32_t>(B),
      .num_q_heads = static_cast<uint32_t>(H),
      .eps = static_cast<float>(eps),
  };
  const uint32_t total_works = static_cast<uint32_t>(B * H);
  const uint32_t num_blocks = CEILDIV(total_works, kFusedQNumWarps);

  // Dispatch on head_dim. DeepSeek V4 uses D=192 with kRopeDim=64.
  constexpr int64_t kRopeDim = 64;
  switch (D) {
    case 128:
      fused_q_norm_rope_kernel<128, kRopeDim><<<num_blocks, kFusedQBlockSize, 0, stream>>>(params);
      break;
    case 192:
      fused_q_norm_rope_kernel<192, kRopeDim><<<num_blocks, kFusedQBlockSize, 0, stream>>>(params);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head_dim for dsv4_fused_q_norm_rope: ", D);
  }
}

void dsv4_fused_k_norm_rope_flashmla(
    const at::Tensor& kv,
    const at::Tensor& kv_weight,
    const at::Tensor& freqs_cis,
    const at::Tensor& positions,
    const at::Tensor& out_loc,
    at::Tensor& kvcache,
    double eps,
    int64_t page_size) {
  TORCH_CHECK(kv.is_cuda(), "kv must be a CUDA tensor");
  TORCH_CHECK(kv.scalar_type() == at::ScalarType::BFloat16, "kv must be bfloat16");
  TORCH_CHECK(kv.dim() == 2, "kv must be 2D: (B, D)");
  TORCH_CHECK(positions.scalar_type() == at::ScalarType::Int, "positions must be int32");
  TORCH_CHECK(out_loc.scalar_type() == at::ScalarType::Int, "out_loc must be int32");

  const int64_t B = kv.size(0);
  const int64_t D = kv.size(1);
  TORCH_CHECK(D == 512, "kv head_dim must be 512 for FlashMLA");
  TORCH_CHECK(kv_weight.size(0) == D, "kv_weight size must match head_dim");

  if (B == 0) return;

  const auto stream = at::cuda::getCurrentCUDAStream(kv.get_device());
  const auto params = FusedKNormRopeFlashMLAParams{
      .kv = kv.data_ptr(),
      .kv_weight = kv_weight.data_ptr(),
      .freqs_cis = freqs_cis.data_ptr<float>(),
      .positions = positions.data_ptr<int32_t>(),
      .out_loc = out_loc.data_ptr<int32_t>(),
      .kvcache = static_cast<uint8_t*>(kvcache.data_ptr()),
      .kv_stride_batch = kv.stride(0),
      .batch_size = static_cast<uint32_t>(B),
      .eps = static_cast<float>(eps),
  };

  constexpr int64_t kHeadDim = 512;
  constexpr int64_t kRopeDim = 64;

  // Dispatch on page_size (must be power of 2).
  TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0, "page_size must be a power of 2");

#define LAUNCH_K_KERNEL(PAGE_BITS)                                 \
  fused_k_norm_rope_flashmla_kernel<kHeadDim, kRopeDim, PAGE_BITS> \
      <<<static_cast<uint32_t>(B), kFusedKBlockSize, 0, stream>>>(params)

  switch (page_size) {
    case 1:
      LAUNCH_K_KERNEL(0);
      break;
    case 2:
      LAUNCH_K_KERNEL(1);
      break;
    case 4:
      LAUNCH_K_KERNEL(2);
      break;
    case 8:
      LAUNCH_K_KERNEL(3);
      break;
    case 16:
      LAUNCH_K_KERNEL(4);
      break;
    case 32:
      LAUNCH_K_KERNEL(5);
      break;
    case 64:
      LAUNCH_K_KERNEL(6);
      break;
    case 128:
      LAUNCH_K_KERNEL(7);
      break;
    case 256:
      LAUNCH_K_KERNEL(8);
      break;
    default:
      TORCH_CHECK(false, "Unsupported page_size: ", page_size);
  }
#undef LAUNCH_K_KERNEL
}

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
