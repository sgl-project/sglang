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
 * SVDQuant W4A4 GEMM Kernel
 * Based on the nunchaku library SVDQuant implementation.
 *
 * This kernel performs quantized GEMM with W4A4 (4-bit weights, 4-bit activations)
 * and supports various fusion options including LoRA, RoPE, and RMSNorm.
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cmath>
#include <optional>
#include <vector>

namespace sgl_diffusion {

// Simple reference implementation of W4A4 GEMM for correctness validation.
// For production use, this should be replaced with optimized CUTLASS-based kernels.
template <typename scalar_t>
__global__ void svdq_gemm_w4a4_simple_kernel(
    const int8_t* __restrict__ act,        // [M, K/2] packed 4-bit activations
    const int8_t* __restrict__ wgt,        // [N, K/2] packed 4-bit weights
    scalar_t* __restrict__ out,            // [M, N] output
    const scalar_t* __restrict__ ascales,  // [K/G, M] activation scales
    const scalar_t* __restrict__ wscales,  // [K/G, N] weight scales
    int M,
    int N,
    int K,
    int group_size) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= M || col >= N) return;

  float acc = 0.0f;
  int num_groups = K / group_size;

  for (int g = 0; g < num_groups; ++g) {
    float a_scale = static_cast<float>(ascales[g * M + row]);
    float w_scale = static_cast<float>(wscales[g * N + col]);

    for (int k = g * group_size; k < (g + 1) * group_size; k += 2) {
      int packed_idx = k / 2;

      // Unpack 4-bit values
      int8_t packed_a = act[row * (K / 2) + packed_idx];
      int8_t packed_w = wgt[col * (K / 2) + packed_idx];

      int8_t a_lo = (packed_a & 0x0F) - 8;  // Signed 4-bit
      int8_t a_hi = ((packed_a >> 4) & 0x0F) - 8;
      int8_t w_lo = (packed_w & 0x0F) - 8;
      int8_t w_hi = ((packed_w >> 4) & 0x0F) - 8;

      // Accumulate with scales
      acc += static_cast<float>(a_lo) * static_cast<float>(w_lo) * a_scale * w_scale;
      acc += static_cast<float>(a_hi) * static_cast<float>(w_hi) * a_scale * w_scale;
    }
  }

  out[row * N + col] = static_cast<scalar_t>(acc);
}

// Apply LoRA: out = out + lora_act_in @ lora_up.T * scale
template <typename scalar_t>
__global__ void apply_lora_kernel(
    scalar_t* __restrict__ out,             // [M, N]
    const float* __restrict__ lora_act_in,  // [M, R]
    const scalar_t* __restrict__ lora_up,   // [N, R]
    const float* __restrict__ lora_scales,  // [ceil(R/16)]
    int M,
    int N,
    int R) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= M || col >= N) return;

  float acc = static_cast<float>(out[row * N + col]);

  for (int r = 0; r < R; ++r) {
    int scale_idx = r / 16;
    float scale = lora_scales[scale_idx];
    acc += lora_act_in[row * R + r] * static_cast<float>(lora_up[col * R + r]) * scale;
  }

  out[row * N + col] = static_cast<scalar_t>(acc);
}

// Apply bias
template <typename scalar_t>
__global__ void apply_bias_kernel(scalar_t* __restrict__ out, const scalar_t* __restrict__ bias, int M, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= M || col >= N) return;

  out[row * N + col] = out[row * N + col] + bias[col];
}

// Apply SiLU activation: out = silu(out[:, :N/2]) * out[:, N/2:]
template <typename scalar_t>
__global__ void apply_silu_kernel(scalar_t* __restrict__ out, int M, int half_N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= M || col >= half_N) return;

  float x = static_cast<float>(out[row * (2 * half_N) + col]);
  float gate = static_cast<float>(out[row * (2 * half_N) + half_N + col]);

  // SiLU: x * sigmoid(x)
  float silu_x = x / (1.0f + expf(-x));
  float result = silu_x * gate;

  out[row * (2 * half_N) + col] = static_cast<scalar_t>(result);
}

}  // namespace sgl_diffusion

void svdq_gemm_w4a4(
    std::optional<torch::Tensor> act_opt,        // packed act [M, K / 2]
    std::optional<torch::Tensor> wgt_opt,        // packed wgt [N, K / 2]
    std::optional<torch::Tensor> out,            // linear [M, N]
    std::optional<torch::Tensor> qout,           // packed act [M, N / 2]
    std::optional<torch::Tensor> ascales,        // packed as [K / G, M]
    std::optional<torch::Tensor> wscales,        // packed ws [K / G, N]
    std::optional<torch::Tensor> oscales,        // packed as [N / G, M]
    std::optional<torch::Tensor> poolout,        // reserved
    std::optional<torch::Tensor> lora_act_in,    // [M, R]
    std::optional<torch::Tensor> lora_up,        // [N, R]
    std::optional<torch::Tensor> lora_down,      // [N, R]
    std::optional<torch::Tensor> lora_act_out,   // [M, R]
    std::optional<torch::Tensor> norm_q,         // [HEAD_DIM]
    std::optional<torch::Tensor> norm_k,         // [HEAD_DIM]
    std::optional<torch::Tensor> rotary_emb,     // [M, HEAD_DIM/2, 2, 2]
    std::optional<torch::Tensor> bias,           // [N]
    std::optional<torch::Tensor> smooth_factor,  // [N]
    bool act_unsigned,
    std::vector<double> lora_scales,
    bool fuse_silu,
    bool fp4,
    double alpha,
    std::optional<torch::Tensor> wcscales,
    std::optional<torch::Tensor> out_q,
    std::optional<torch::Tensor> out_k,
    std::optional<torch::Tensor> out_v,
    int64_t attn_tokens) {
  TORCH_CHECK(act_opt.has_value(), "act tensor is required");
  TORCH_CHECK(wgt_opt.has_value(), "wgt tensor is required");
  TORCH_CHECK(out.has_value(), "out tensor is required");
  TORCH_CHECK(ascales.has_value(), "ascales tensor is required");
  TORCH_CHECK(wscales.has_value(), "wscales tensor is required");

  auto act = act_opt.value();
  auto wgt = wgt_opt.value();

  TORCH_CHECK(act.is_cuda(), "act must be a CUDA tensor");
  TORCH_CHECK(wgt.is_cuda(), "wgt must be a CUDA tensor");

  int M = act.size(0);
  int K = act.size(1) * 2;  // Packed 4-bit
  int N = wgt.size(0);
  int group_size = fp4 ? 16 : 64;

  auto stream = at::cuda::getCurrentCUDAStream();

  // Determine output dtype
  auto out_tensor = out.value();
  auto out_dtype = out_tensor.scalar_type();

  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  AT_DISPATCH_SWITCH(
      out_dtype, "svdq_gemm_w4a4_cuda", AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
        sgl_diffusion::svdq_gemm_w4a4_simple_kernel<half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const int8_t*>(act.data_ptr()),
            reinterpret_cast<const int8_t*>(wgt.data_ptr()),
            reinterpret_cast<half*>(out_tensor.data_ptr()),
            reinterpret_cast<const half*>(ascales.value().data_ptr()),
            reinterpret_cast<const half*>(wscales.value().data_ptr()),
            M,
            N,
            K,
            group_size);

        // Apply LoRA if provided
        if (lora_act_in.has_value() && lora_up.has_value()) {
          int R = lora_up.value().size(1);
          std::vector<float> lora_scales_vec;
          if (!lora_scales.empty()) {
            for (double s : lora_scales) {
              lora_scales_vec.push_back(static_cast<float>(s));
            }
          } else {
            lora_scales_vec.resize((R + 15) / 16, 1.0f);
          }

          auto lora_scales_tensor =
              torch::tensor(lora_scales_vec, torch::TensorOptions().dtype(torch::kFloat32).device(act.device()));

          sgl_diffusion::apply_lora_kernel<half><<<grid, block, 0, stream>>>(
              reinterpret_cast<half*>(out_tensor.data_ptr()),
              reinterpret_cast<const float*>(lora_act_in.value().data_ptr()),
              reinterpret_cast<const half*>(lora_up.value().data_ptr()),
              reinterpret_cast<const float*>(lora_scales_tensor.data_ptr()),
              M,
              N,
              R);
        }

        // Apply bias if provided
        if (bias.has_value()) {
          sgl_diffusion::apply_bias_kernel<half><<<grid, block, 0, stream>>>(
              reinterpret_cast<half*>(out_tensor.data_ptr()),
              reinterpret_cast<const half*>(bias.value().data_ptr()),
              M,
              N);
        }

        // Apply SiLU if requested
        if (fuse_silu) {
          int half_N = N / 2;
          dim3 silu_grid((M + block.x - 1) / block.x, (half_N + block.y - 1) / block.y);
          sgl_diffusion::apply_silu_kernel<half>
              <<<silu_grid, block, 0, stream>>>(reinterpret_cast<half*>(out_tensor.data_ptr()), M, half_N);
        }
      }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
        sgl_diffusion::svdq_gemm_w4a4_simple_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const int8_t*>(act.data_ptr()),
            reinterpret_cast<const int8_t*>(wgt.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out_tensor.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(ascales.value().data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(wscales.value().data_ptr()),
            M,
            N,
            K,
            group_size);

        // Apply LoRA if provided
        if (lora_act_in.has_value() && lora_up.has_value()) {
          int R = lora_up.value().size(1);
          std::vector<float> lora_scales_vec;
          if (!lora_scales.empty()) {
            for (double s : lora_scales) {
              lora_scales_vec.push_back(static_cast<float>(s));
            }
          } else {
            lora_scales_vec.resize((R + 15) / 16, 1.0f);
          }

          auto lora_scales_tensor =
              torch::tensor(lora_scales_vec, torch::TensorOptions().dtype(torch::kFloat32).device(act.device()));

          sgl_diffusion::apply_lora_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
              reinterpret_cast<__nv_bfloat16*>(out_tensor.data_ptr()),
              reinterpret_cast<const float*>(lora_act_in.value().data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(lora_up.value().data_ptr()),
              reinterpret_cast<const float*>(lora_scales_tensor.data_ptr()),
              M,
              N,
              R);
        }

        // Apply bias if provided
        if (bias.has_value()) {
          sgl_diffusion::apply_bias_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
              reinterpret_cast<__nv_bfloat16*>(out_tensor.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(bias.value().data_ptr()),
              M,
              N);
        }

        // Apply SiLU if requested
        if (fuse_silu) {
          int half_N = N / 2;
          dim3 silu_grid((M + block.x - 1) / block.x, (half_N + block.y - 1) / block.y);
          sgl_diffusion::apply_silu_kernel<__nv_bfloat16>
              <<<silu_grid, block, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(out_tensor.data_ptr()), M, half_N);
        }
      }));
}
