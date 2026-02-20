/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
// Adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

// ---------------------------------------------------------------------------
// Self-contained utilities (no external heavy deps)
// ---------------------------------------------------------------------------

#define SGL_QKNORM_ROPE_FINAL_MASK 0xffffffff

// Packed uint types for vectorized bf16 loads/stores
template <typename T, int num>
struct packed_as_uint;

template <>
struct packed_as_uint<unsigned int, 1> {
  using type = unsigned int;
};

template <>
struct packed_as_uint<unsigned int, 2> {
  using type = uint2;
};

template <>
struct packed_as_uint<unsigned int, 4> {
  using type = uint4;
};

template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(SGL_QKNORM_ROPE_FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
inline __device__ __host__ T div_up(T m, T n) {
  return (m + n - 1) / n;
}

// ---------------------------------------------------------------------------
// YaRN-aware frequency computation
//
// When factor == 1.0, reduces to standard RoPE: base^(-2*half_dim/rotary_dim)
// When factor != 1.0, blends interpolated and extrapolated frequencies.
// ---------------------------------------------------------------------------

__device__ inline float
compute_freq_yarn(float base, int rotary_dim, int half_dim, float factor, float low, float high) {
  float freq = powf(base, -2.0f * half_dim / static_cast<float>(rotary_dim));

  if (factor != 1.0f) {
    float inv_freq_extrapolation = freq;
    float inv_freq_interpolation = freq / factor;

    float high_adj = high;
    if (fabsf(low - high_adj) <= 1e-6f) {
      high_adj += 0.001f;
    }

    float linear_func = (static_cast<float>(half_dim) - low) / (high_adj - low);
    float ramp_func = fminf(fmaxf(linear_func, 0.0f), 1.0f);
    float inv_freq_extrapolation_factor = 1.0f - ramp_func;

    freq = inv_freq_interpolation * (1.0f - inv_freq_extrapolation_factor) +
           inv_freq_extrapolation * inv_freq_extrapolation_factor;
  }

  return freq;
}

// ---------------------------------------------------------------------------
// Fused QK-Norm + RoPE kernel
//
// Each warp processes one (token, head) pair.
//   head_dim:   compile-time head dimension (64, 128, or 256)
//   interleave: true  → interleave / GPT-J style RoPE (!is_neox)
//               false → NeoX style RoPE (is_neox)
// ---------------------------------------------------------------------------

template <int head_dim, bool interleave>
__global__ void fusedQKNormRopeKernel(
    __nv_bfloat16* qkv,  // [num_tokens, (nq+nk+nv)*head_dim], in-place
    int const num_heads_q,
    int const num_heads_k,
    int const num_heads_v,
    float const eps,
    __nv_bfloat16 const* q_weight,  // [head_dim]
    __nv_bfloat16 const* k_weight,  // [head_dim]
    float const base,
    int const* position_ids,  // [num_tokens]
    int const num_tokens,
    float factor,
    float low,
    float high,
    float attention_factor,
    int const rotary_dim) {
  int const warpsPerBlock = blockDim.x / 32;
  int const warpId = threadIdx.x / 32;
  int const laneId = threadIdx.x % 32;

  int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;
  int const total_qk_heads = num_heads_q + num_heads_k;

  int const tokenIdx = globalWarpIdx / total_qk_heads;
  int const localHeadIdx = globalWarpIdx % total_qk_heads;

  if (tokenIdx >= num_tokens) return;

  bool const isQ = localHeadIdx < num_heads_q;
  int const headIdx = isQ ? localHeadIdx : localHeadIdx - num_heads_q;
  int const num_heads = num_heads_q + num_heads_k + num_heads_v;

  static_assert(head_dim % (32 * 2) == 0, "head_dim must be divisible by 64 (each warp handles one head)");
  constexpr int numElemsPerThread = head_dim / 32;
  float elements[numElemsPerThread];
  constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
  static_assert(elemSizeBytes % 4 == 0, "elemSizeBytes must be a multiple of 4");
  constexpr int vecSize = elemSizeBytes / 4;
  using vec_T = typename packed_as_uint<unsigned int, vecSize>::type;

  // Compute flat offset of this warp's head in qkv
  int offsetWarp;
  if (isQ) {
    offsetWarp = tokenIdx * num_heads * head_dim + headIdx * head_dim;
  } else {
    offsetWarp = tokenIdx * num_heads * head_dim + num_heads_q * head_dim + headIdx * head_dim;
  }
  int offsetThread = offsetWarp + laneId * numElemsPerThread;

  // -------------------------------------------------------------------
  // Load and compute sum-of-squares for RMSNorm
  // -------------------------------------------------------------------
  float sumOfSquares = 0.0f;
  {
    vec_T vec = *reinterpret_cast<vec_T const*>(&qkv[offsetThread]);
    for (int i = 0; i < vecSize; i++) {
      float2 vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<unsigned int*>(&vec) + i));
      sumOfSquares += vals.x * vals.x;
      sumOfSquares += vals.y * vals.y;
      elements[2 * i] = vals.x;
      elements[2 * i + 1] = vals.y;
    }
  }

  sumOfSquares = warp_reduce_sum(sumOfSquares);

  // -------------------------------------------------------------------
  // Apply RMSNorm
  // -------------------------------------------------------------------
  float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);
  for (int i = 0; i < numElemsPerThread; i++) {
    int dim = laneId * numElemsPerThread + i;
    float weight = isQ ? __bfloat162float(q_weight[dim]) : __bfloat162float(k_weight[dim]);
    elements[i] *= rms_rcp * weight;
  }

  // -------------------------------------------------------------------
  // Apply RoPE to the first rotary_dim elements
  // -------------------------------------------------------------------
  float elements2[numElemsPerThread];
  float cos_vals[numElemsPerThread];
  float sin_vals[numElemsPerThread];
  float pos_id = static_cast<float>(position_ids[tokenIdx]);
  int const rotary_lanes = rotary_dim / numElemsPerThread;
  bool const applyRotary = (laneId < rotary_lanes);

  if (applyRotary) {
    if constexpr (interleave) {
      // Interleave (GPT-J) style: pairs of consecutive elements share a frequency
      for (int i = 0; i < numElemsPerThread; i++) {
        elements2[i] = (i % 2 == 0) ? -elements[i + 1] : elements[i - 1];

        int dim_idx = laneId * numElemsPerThread + i;
        int half_dim = dim_idx / 2;
        float freq = compute_freq_yarn(base, rotary_dim, half_dim, factor, low, high);
        float theta = pos_id * freq;
        __sincosf(theta, &sin_vals[i], &cos_vals[i]);
      }
    } else {
      // NeoX style: first and second halves of the rotary region are paired
      __syncwarp();
      int const half_rotary_lanes = rotary_lanes / 2;
      // Avoid UB from (1u << 32) when rotary_lanes == 32
      unsigned int active_mask = 0xffffffffu >> (32 - rotary_lanes);
      for (int i = 0; i < numElemsPerThread; i++) {
        elements2[i] = __shfl_xor_sync(active_mask, elements[i], half_rotary_lanes);
        if (laneId < half_rotary_lanes) {
          elements2[i] = -elements2[i];
        }

        int dim_idx = laneId * numElemsPerThread + i;
        // Remap so that both halves use the same set of frequencies
        dim_idx = (dim_idx * 2) % rotary_dim;
        int half_dim = dim_idx / 2;
        float freq = compute_freq_yarn(base, rotary_dim, half_dim, factor, low, high);
        float theta = pos_id * freq;
        __sincosf(theta, &sin_vals[i], &cos_vals[i]);
      }
      __syncwarp();
    }

    for (int i = 0; i < numElemsPerThread; i++) {
      elements[i] = (elements[i] * cos_vals[i] + elements2[i] * sin_vals[i]) * attention_factor;
    }
  }

  // -------------------------------------------------------------------
  // Store (all elements: rotated + pass-through normalized)
  // -------------------------------------------------------------------
  {
    vec_T vec;
    for (int i = 0; i < vecSize; i++) {
      __nv_bfloat162 vals = __float22bfloat162_rn(make_float2(elements[2 * i], elements[2 * i + 1]));
      reinterpret_cast<__nv_bfloat162&>(*(reinterpret_cast<unsigned int*>(&vec) + i)) = vals;
    }
    vec_T* outputPtr = reinterpret_cast<vec_T*>(&qkv[offsetThread]);
    *outputPtr = vec;
  }
}

// ---------------------------------------------------------------------------
// Host-side tvm-ffi entry point
// ---------------------------------------------------------------------------

void fused_qk_norm_rope(
    tvm::ffi::TensorView qkv,           // [num_tokens, (nq+nk+nv)*head_dim] bf16
    tvm::ffi::TensorView q_weight,      // [head_dim] bf16
    tvm::ffi::TensorView k_weight,      // [head_dim] bf16
    tvm::ffi::TensorView position_ids,  // [num_tokens] int32
    int num_heads_q,
    int num_heads_k,
    int num_heads_v,
    int head_dim,
    float eps,
    float base,
    int is_neox,  // 0 = interleave style, 1 = NeoX style
    float factor,
    float low,
    float high,
    float attention_factor,
    int rotary_dim) {
  using namespace host;

  RuntimeCheck(qkv.device().device_type == kDLCUDA, "qkv must be a CUDA tensor");
  RuntimeCheck(qkv.is_contiguous(), "qkv must be contiguous");
  RuntimeCheck(qkv.dtype().code == kDLBfloat && qkv.dtype().bits == 16, "qkv must be bfloat16");
  RuntimeCheck(qkv.ndim() == 2, "qkv must be 2D: [num_tokens, (nq+nk+nv)*head_dim]");

  RuntimeCheck(q_weight.is_contiguous(), "q_weight must be contiguous");
  RuntimeCheck(q_weight.dtype().code == kDLBfloat && q_weight.dtype().bits == 16, "q_weight must be bfloat16");
  RuntimeCheck(
      q_weight.ndim() == 1 && static_cast<int>(q_weight.size(0)) == head_dim, "q_weight must be 1D of size head_dim");

  RuntimeCheck(k_weight.is_contiguous(), "k_weight must be contiguous");
  RuntimeCheck(k_weight.dtype().code == kDLBfloat && k_weight.dtype().bits == 16, "k_weight must be bfloat16");
  RuntimeCheck(
      k_weight.ndim() == 1 && static_cast<int>(k_weight.size(0)) == head_dim, "k_weight must be 1D of size head_dim");

  RuntimeCheck(position_ids.device().device_type == kDLCUDA, "position_ids must be a CUDA tensor");
  RuntimeCheck(position_ids.is_contiguous(), "position_ids must be contiguous");
  RuntimeCheck(position_ids.dtype().code == kDLInt && position_ids.dtype().bits == 32, "position_ids must be int32");
  RuntimeCheck(position_ids.ndim() == 1, "position_ids must be 1D: [num_tokens]");

  int num_tokens = static_cast<int>(qkv.size(0));
  int total_heads = num_heads_q + num_heads_k + num_heads_v;
  RuntimeCheck(
      static_cast<int>(qkv.size(1)) == total_heads * head_dim, "qkv.size(1) must equal (nq + nk + nv) * head_dim");
  RuntimeCheck(static_cast<int>(position_ids.size(0)) == num_tokens, "position_ids must have num_tokens elements");

  int numElemsPerThread = head_dim / 32;
  RuntimeCheck(rotary_dim % numElemsPerThread == 0, "rotary_dim must be divisible by (head_dim / 32)");

  bool neox = static_cast<bool>(is_neox);
  if (neox) {
    // NeoX uses __shfl_xor_sync which requires half_rotary_lanes to be a power of 2
    int rotary_lanes = rotary_dim / numElemsPerThread;
    int half_rotary_lanes = rotary_lanes / 2;
    bool is_pow2 = (half_rotary_lanes >= 1) && ((half_rotary_lanes & (half_rotary_lanes - 1)) == 0);
    RuntimeCheck(is_pow2, "half_rotary_lanes must be a power of 2 for NeoX style RoPE");
  }

  RuntimeCheck(head_dim == 64 || head_dim == 128 || head_dim == 256, "head_dim must be 64, 128, or 256");

  cudaStream_t stream = LaunchKernel::resolve_device(qkv.device());

  constexpr int blockSize = 256;
  int warpsPerBlock = blockSize / 32;
  int totalQKHeads = num_heads_q + num_heads_k;
  int totalWarps = num_tokens * totalQKHeads;
  int gridSize = div_up(totalWarps, warpsPerBlock);

  bool interleave = !neox;

  auto* qkv_ptr = reinterpret_cast<__nv_bfloat16*>(qkv.data_ptr());
  auto const* qw_ptr = reinterpret_cast<__nv_bfloat16 const*>(q_weight.data_ptr());
  auto const* kw_ptr = reinterpret_cast<__nv_bfloat16 const*>(k_weight.data_ptr());
  auto const* pos_ptr = reinterpret_cast<int const*>(position_ids.data_ptr());

#define LAUNCH_KERNEL(HD, IL)                                        \
  fusedQKNormRopeKernel<HD, IL><<<gridSize, blockSize, 0, stream>>>( \
      qkv_ptr,                                                       \
      num_heads_q,                                                   \
      num_heads_k,                                                   \
      num_heads_v,                                                   \
      eps,                                                           \
      qw_ptr,                                                        \
      kw_ptr,                                                        \
      base,                                                          \
      pos_ptr,                                                       \
      num_tokens,                                                    \
      factor,                                                        \
      low,                                                           \
      high,                                                          \
      attention_factor,                                              \
      rotary_dim)

  switch (head_dim) {
    case 64:
      if (interleave)
        LAUNCH_KERNEL(64, true);
      else
        LAUNCH_KERNEL(64, false);
      break;
    case 128:
      if (interleave)
        LAUNCH_KERNEL(128, true);
      else
        LAUNCH_KERNEL(128, false);
      break;
    case 256:
      if (interleave)
        LAUNCH_KERNEL(256, true);
      else
        LAUNCH_KERNEL(256, false);
      break;
    default:
      RuntimeCheck(false, "Unsupported head_dim: must be 64, 128, or 256");
  }

#undef LAUNCH_KERNEL
}

}  // namespace
