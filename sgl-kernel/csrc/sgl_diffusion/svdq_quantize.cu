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

/*
 * SVDQuant W4A4 Activation Quantization Kernel with LoRA Fusion
 * Based on the nunchaku library SVDQuant implementation.
 *
 * This kernel quantizes activations to 4-bit format and optionally computes
 * LoRA down-projection in a fused manner.
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cmath>
#include <optional>

namespace sgl_diffusion {

// Helper to convert to float
template <typename T>
__device__ __forceinline__ float to_float(T val);

template <>
__device__ __forceinline__ float to_float<half>(half val) {
  return __half2float(val);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <>
__device__ __forceinline__ float to_float<float>(float val) {
  return val;
}

// Helper to convert from float
template <typename T>
__device__ __forceinline__ T from_float(float val);

template <>
__device__ __forceinline__ half from_float<half>(float val) {
  return __float2half(val);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) {
  return __float2bfloat16(val);
}

template <>
__device__ __forceinline__ float from_float<float>(float val) {
  return val;
}

// Quantize activations to 4-bit with per-group scales
// INT4 version: group_size = 64, scale dtype = input dtype
template <typename scalar_t>
__global__ void quantize_w4a4_act_kernel_int4(
    const scalar_t* __restrict__ input,   // [M, K]
    uint8_t* __restrict__ output,         // [M, K/2]
    scalar_t* __restrict__ oscales,       // [K/G, M]
    const scalar_t* __restrict__ smooth,  // [K] or nullptr
    int M,
    int K,
    int group_size,
    bool fuse_glu) {
  int row = blockIdx.x;
  int group_idx = blockIdx.y;

  if (row >= M) return;

  int K_effective = fuse_glu ? K / 2 : K;
  int num_groups = K_effective / group_size;

  if (group_idx >= num_groups) return;

  int k_start = group_idx * group_size;
  int k_end = k_start + group_size;

  // Find max absolute value in this group
  float max_val = 0.0f;
  for (int k = k_start + threadIdx.x; k < k_end; k += blockDim.x) {
    float val = to_float(input[row * K + k]);
    if (smooth != nullptr) {
      val *= to_float(smooth[k]);
    }
    max_val = fmaxf(max_val, fabsf(val));
  }

  // Warp reduce to find max
  for (int offset = 16; offset > 0; offset /= 2) {
    max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
  }

  // Compute scale
  float scale = max_val / 7.0f;  // For signed 4-bit [-8, 7], we use [-7, 7] for symmetric
  if (scale < 1e-8f) scale = 1e-8f;

  // Store scale (transposed: [K/G, M])
  if (threadIdx.x == 0) {
    oscales[group_idx * M + row] = from_float<scalar_t>(scale);
  }
  __syncthreads();

  // Quantize and pack
  for (int k = k_start + threadIdx.x * 2; k < k_end; k += blockDim.x * 2) {
    if (k + 1 < k_end) {
      float val0 = to_float(input[row * K + k]);
      float val1 = to_float(input[row * K + k + 1]);

      if (smooth != nullptr) {
        val0 *= to_float(smooth[k]);
        val1 *= to_float(smooth[k + 1]);
      }

      // Quantize to [-8, 7]
      int8_t q0 = static_cast<int8_t>(roundf(val0 / scale));
      int8_t q1 = static_cast<int8_t>(roundf(val1 / scale));
      q0 = max(min(q0, (int8_t)7), (int8_t)-8);
      q1 = max(min(q1, (int8_t)7), (int8_t)-8);

      // Pack two 4-bit values into one byte
      // Low nibble: q0 + 8, High nibble: q1 + 8
      uint8_t packed = ((uint8_t)(q0 + 8) & 0x0F) | (((uint8_t)(q1 + 8) & 0x0F) << 4);
      output[row * (K / 2) + k / 2] = packed;
    }
  }
}

// NVFP4 version: group_size = 16, scale dtype = fp8_e4m3
template <typename scalar_t>
__global__ void quantize_w4a4_act_kernel_fp4(
    const scalar_t* __restrict__ input,   // [M, K]
    uint8_t* __restrict__ output,         // [M, K/2]
    __nv_fp8_e4m3* __restrict__ oscales,  // [K/16, M]
    const scalar_t* __restrict__ smooth,  // [K] or nullptr
    int M,
    int K,
    bool fuse_glu) {
  int row = blockIdx.x;
  int group_idx = blockIdx.y;

  constexpr int group_size = 16;

  if (row >= M) return;

  int K_effective = fuse_glu ? K / 2 : K;
  int num_groups = K_effective / group_size;

  if (group_idx >= num_groups) return;

  int k_start = group_idx * group_size;
  int k_end = k_start + group_size;

  // Find max absolute value in this group
  float max_val = 0.0f;
  for (int k = k_start + threadIdx.x; k < k_end; k += blockDim.x) {
    float val = to_float(input[row * K + k]);
    if (smooth != nullptr) {
      val *= to_float(smooth[k]);
    }
    max_val = fmaxf(max_val, fabsf(val));
  }

  // Warp reduce
  for (int offset = 16; offset > 0; offset /= 2) {
    max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
  }

  float scale = max_val / 7.0f;
  if (scale < 1e-8f) scale = 1e-8f;

  // Store scale as FP8 E4M3 (transposed)
  if (threadIdx.x == 0) {
    oscales[group_idx * M + row] = __nv_fp8_e4m3(scale);
  }
  __syncthreads();

  // Quantize and pack (same as INT4)
  for (int k = k_start + threadIdx.x * 2; k < k_end; k += blockDim.x * 2) {
    if (k + 1 < k_end) {
      float val0 = to_float(input[row * K + k]);
      float val1 = to_float(input[row * K + k + 1]);

      if (smooth != nullptr) {
        val0 *= to_float(smooth[k]);
        val1 *= to_float(smooth[k + 1]);
      }

      int8_t q0 = static_cast<int8_t>(roundf(val0 / scale));
      int8_t q1 = static_cast<int8_t>(roundf(val1 / scale));
      q0 = max(min(q0, (int8_t)7), (int8_t)-8);
      q1 = max(min(q1, (int8_t)7), (int8_t)-8);

      uint8_t packed = ((uint8_t)(q0 + 8) & 0x0F) | (((uint8_t)(q1 + 8) & 0x0F) << 4);
      output[row * (K / 2) + k / 2] = packed;
    }
  }
}

// Compute LoRA down projection: lora_act_out = input @ lora_down
template <typename scalar_t>
__global__ void lora_down_projection_kernel(
    const scalar_t* __restrict__ input,      // [M, K]
    const scalar_t* __restrict__ lora_down,  // [K, R]
    float* __restrict__ lora_act_out,        // [M, R]
    const scalar_t* __restrict__ smooth,     // [K] or nullptr
    int M,
    int K,
    int R,
    bool fuse_glu) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= M || r >= R) return;

  int K_effective = fuse_glu ? K / 2 : K;
  float acc = 0.0f;

  for (int k = 0; k < K_effective; ++k) {
    float val = to_float(input[row * K + k]);
    if (smooth != nullptr) {
      val *= to_float(smooth[k]);
    }
    acc += val * to_float(lora_down[k * R + r]);
  }

  lora_act_out[row * R + r] = acc;
}

}  // namespace sgl_diffusion

// Helper to compute ceil division
inline int64_t ceil_divide(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

std::vector<torch::Tensor> svdq_quantize_w4a4_act_fuse_lora(
    torch::Tensor input,
    std::optional<torch::Tensor> output_opt,
    std::optional<torch::Tensor> oscales_opt,
    std::optional<torch::Tensor> lora_down,
    std::optional<torch::Tensor> lora_act_out_opt,
    std::optional<torch::Tensor> smooth,
    bool fuse_glu,
    bool fp4,
    int64_t pad_size) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be 2D tensor");

  int64_t batch_size = input.size(0);
  int64_t channels = input.size(1);
  int64_t batch_size_pad = ceil_divide(batch_size, pad_size) * pad_size;

  int group_size = fp4 ? 16 : 64;
  TORCH_CHECK(channels % group_size == 0, "channels must be divisible by group_size");

  auto device = input.device();
  auto dtype = input.scalar_type();

  // Allocate output if not provided
  torch::Tensor output;
  if (output_opt.has_value()) {
    output = output_opt.value();
  } else {
    output = torch::empty({batch_size_pad, channels / 2}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
  }

  // Allocate oscales if not provided
  torch::Tensor oscales;
  if (oscales_opt.has_value()) {
    oscales = oscales_opt.value();
  } else {
    if (fp4) {
      oscales = torch::empty(
          {channels / 16, batch_size_pad}, torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device));
    } else {
      oscales = torch::empty({channels / 64, batch_size_pad}, torch::TensorOptions().dtype(dtype).device(device));
    }
  }

  // Allocate lora_act_out if needed
  torch::Tensor lora_act_out;
  int64_t rank = 0;
  if (lora_down.has_value()) {
    rank = lora_down.value().size(1);
    if (lora_act_out_opt.has_value()) {
      lora_act_out = lora_act_out_opt.value();
    } else {
      lora_act_out = torch::empty({batch_size_pad, rank}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }
  } else {
    lora_act_out = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  }

  auto stream = at::cuda::getCurrentCUDAStream();

  int num_groups = channels / group_size;
  dim3 grid(batch_size, num_groups);
  dim3 block(32);

  AT_DISPATCH_SWITCH(
      dtype, "svdq_quantize_w4a4_act_fuse_lora_cuda", AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
        const half* smooth_ptr =
            smooth.has_value() ? reinterpret_cast<const half*>(smooth.value().data_ptr()) : nullptr;

        if (fp4) {
          sgl_diffusion::quantize_w4a4_act_kernel_fp4<half><<<grid, block, 0, stream>>>(
              reinterpret_cast<const half*>(input.data_ptr()),
              reinterpret_cast<uint8_t*>(output.data_ptr()),
              reinterpret_cast<__nv_fp8_e4m3*>(oscales.data_ptr()),
              smooth_ptr,
              static_cast<int>(batch_size),
              static_cast<int>(channels),
              fuse_glu);
        } else {
          sgl_diffusion::quantize_w4a4_act_kernel_int4<half><<<grid, block, 0, stream>>>(
              reinterpret_cast<const half*>(input.data_ptr()),
              reinterpret_cast<uint8_t*>(output.data_ptr()),
              reinterpret_cast<half*>(oscales.data_ptr()),
              smooth_ptr,
              static_cast<int>(batch_size),
              static_cast<int>(channels),
              group_size,
              fuse_glu);
        }

        // Compute LoRA down projection if needed
        if (lora_down.has_value() && rank > 0) {
          dim3 lora_block(16, 16);
          dim3 lora_grid((batch_size + lora_block.x - 1) / lora_block.x, (rank + lora_block.y - 1) / lora_block.y);

          sgl_diffusion::lora_down_projection_kernel<half><<<lora_grid, lora_block, 0, stream>>>(
              reinterpret_cast<const half*>(input.data_ptr()),
              reinterpret_cast<const half*>(lora_down.value().data_ptr()),
              reinterpret_cast<float*>(lora_act_out.data_ptr()),
              smooth_ptr,
              static_cast<int>(batch_size),
              static_cast<int>(channels),
              static_cast<int>(rank),
              fuse_glu);
        }
      }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
        const __nv_bfloat16* smooth_ptr =
            smooth.has_value() ? reinterpret_cast<const __nv_bfloat16*>(smooth.value().data_ptr()) : nullptr;

        if (fp4) {
          sgl_diffusion::quantize_w4a4_act_kernel_fp4<__nv_bfloat16><<<grid, block, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
              reinterpret_cast<uint8_t*>(output.data_ptr()),
              reinterpret_cast<__nv_fp8_e4m3*>(oscales.data_ptr()),
              smooth_ptr,
              static_cast<int>(batch_size),
              static_cast<int>(channels),
              fuse_glu);
        } else {
          sgl_diffusion::quantize_w4a4_act_kernel_int4<__nv_bfloat16><<<grid, block, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
              reinterpret_cast<uint8_t*>(output.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(oscales.data_ptr()),
              smooth_ptr,
              static_cast<int>(batch_size),
              static_cast<int>(channels),
              group_size,
              fuse_glu);
        }

        // Compute LoRA down projection if needed
        if (lora_down.has_value() && rank > 0) {
          dim3 lora_block(16, 16);
          dim3 lora_grid((batch_size + lora_block.x - 1) / lora_block.x, (rank + lora_block.y - 1) / lora_block.y);

          sgl_diffusion::lora_down_projection_kernel<__nv_bfloat16><<<lora_grid, lora_block, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(lora_down.value().data_ptr()),
              reinterpret_cast<float*>(lora_act_out.data_ptr()),
              smooth_ptr,
              static_cast<int>(batch_size),
              static_cast<int>(channels),
              static_cast<int>(rank),
              fuse_glu);
        }
      }));

  return {output, oscales, lora_act_out};
}
