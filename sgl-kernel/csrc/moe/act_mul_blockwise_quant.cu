// Copyright (c) 2025 SGLang Team.
// Fused SiLU + Mul + Blockwise FP8 Quantization kernel for MoE.
//
// Combines activation (SiLU(gate) * up) and blockwise FP8 quantization
// into a single kernel, eliminating the intermediate bf16 buffer round-trip.
// Supports expert filtering (skip rows where expert_id == -1) and
// optional swiglu_limit clamping (DeepSeek V4).
//
// Reference: hpc-ops-main/src/activation/activation.cu
//            masked_act_mul_and_blockwise_quant_kernel

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "act_mul_blockwise_quant.cuh"
#include "utils.h"

using namespace act_mul_quant;

// ============================================================
// Kernel implementation
// ============================================================

template <bool kHasSwiGLULimit>
__global__ void sgl_act_mul_blockwise_quant_kernel(
    __nv_fp8_e4m3* __restrict__ output,        // [num_total_tokens, hidden_dim]
    float* __restrict__ output_scale,           // [num_total_tokens, num_groups]
    const __nv_bfloat16* __restrict__ input,    // [num_total_tokens, 2 * hidden_dim]
    const int32_t* __restrict__ expert_ids,     // [num_blocks], -1 = skip
    int expert_step,                            // rows per block (power of 2)
    int num_total_tokens,
    int hidden_dim,                             // C (half of input width)
    int num_groups,                             // hidden_dim / 128
    int num_block_col,                          // grid partitioning
    int num_block_row,                          // grid partitioning
    float swiglu_limit) {
  // 2D grid decomposition: Y = row blocks, X = column blocks
  int block_linear_id = blockIdx.x;
  int iblockx = block_linear_id % num_block_col;
  int iblocky = block_linear_id / num_block_col;

  int lane_id = threadIdx.x % 32;
  int it = threadIdx.x + iblockx * blockDim.x;

  // Each block row processes kRows consecutive rows for better occupancy
  constexpr int kRows = 4;

#pragma unroll 1
  for (int irow0 = iblocky * kRows; irow0 < num_total_tokens;
       irow0 += num_block_row * kRows) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      int irow = irow0 + r;
      if (irow >= num_total_tokens) break;

      // Expert filter: check if this row belongs to a valid expert
      int block_idx = irow / expert_step;
      int expert_id = expert_ids[block_idx];
      if (expert_id == -1) {
        continue;
      }

      // Pointers for this row
      const __nv_bfloat16* gate_row_ptr = input + (int64_t)irow * hidden_dim * 2;
      const __nv_bfloat16* up_row_ptr = gate_row_ptr + hidden_dim;
      __nv_fp8_e4m3* output_row_ptr = output + (int64_t)irow * hidden_dim;
      float* scale_row_ptr = output_scale + (int64_t)irow * num_groups;

      int icol = it * kElementsPerThread;
      if (icol < hidden_dim) {
        // 1. Vectorized load: 8 bf16 gate + 8 bf16 up → 8 float each
        float gate[kElementsPerThread];
        float up[kElementsPerThread];
        load_bf16x8_as_float(gate_row_ptr + icol, gate);
        load_bf16x8_as_float(up_row_ptr + icol, up);

        // 2. Optional swiglu_limit clamp
        float out[kElementsPerThread];
        if constexpr (kHasSwiGLULimit) {
#pragma unroll
          for (int i = 0; i < kElementsPerThread; ++i) {
            gate[i] = fminf(gate[i], swiglu_limit);
            up[i] = fmaxf(fminf(up[i], swiglu_limit), -swiglu_limit);
          }
        }

        // 3. SiLU(gate) * up — use bf16 precision multiply to match act_and_mul_triton
        //    (triton kernel casts silu result back to bf16 before multiplying with up)
        //    Then truncate result to bf16 to match the store+load round-trip in baseline
#pragma unroll
        for (int i = 0; i < kElementsPerThread; ++i) {
          __nv_bfloat16 silu_bf16 = __float2bfloat16_rn(silu(gate[i]));
          __nv_bfloat16 up_bf16 = __float2bfloat16_rn(up[i]);
          __nv_bfloat16 result_bf16 = __hmul(silu_bf16, up_bf16);
          out[i] = __bfloat162float(result_bf16);
        }

        // 4. Half-warp (16 threads) reduce max over 128 elements
        float thread_max = 0.0f;
#pragma unroll
        for (int i = 0; i < kElementsPerThread; ++i) {
          thread_max = fmaxf(thread_max, fabsf(out[i]));
        }
        float group_max = half_warp_reduce_max(thread_max);

        // 5. Compute scale and quantize
        float scale = group_max / kFP8E4M3Max;
        float inv_scale = 1.0f / (scale + kQuantEps);

#pragma unroll
        for (int i = 0; i < kElementsPerThread; ++i) {
          out[i] *= inv_scale;
        }

        // 6. Store fp8 output (vectorized 64-bit store)
        store_fp8x8(output_row_ptr + icol, out);

        // 7. Store scale (one per 128 elements, written by lane 0 and lane 16)
        if (lane_id == 0 || lane_id == 16) {
          int group_idx = icol / kGroupSize;
          scale_row_ptr[group_idx] = scale;
        }
      }
    }  // for r in kRows
  }  // for irow0
}

// ============================================================
// Host launcher
// ============================================================

void sgl_act_mul_blockwise_quant(
    at::Tensor output,        // [total_tokens, hidden_dim], fp8_e4m3fn, pre-allocated
    at::Tensor output_scale,  // [total_tokens, hidden_dim/128], fp32, pre-allocated
    at::Tensor input,         // [total_tokens, 2*hidden_dim], bf16
    at::Tensor expert_ids,    // [num_blocks], int32
    int64_t expert_step,      // BLOCK_SIZE_M (power of 2, e.g. 128)
    double swiglu_limit       // <= 0 means no clamp
) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(input.dtype() == torch::kBFloat16, "input must be bfloat16");
  TORCH_CHECK(output.dtype() == torch::kFloat8_e4m3fn, "output must be fp8_e4m3fn");
  TORCH_CHECK(output_scale.dtype() == torch::kFloat32, "output_scale must be float32");
  TORCH_CHECK(expert_ids.dtype() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(output_scale.is_contiguous(), "output_scale must be contiguous");

  const int num_total_tokens = input.size(0);
  const int hidden_dim = input.size(1) / 2;
  const int num_groups = hidden_dim / kGroupSize;

  TORCH_CHECK(hidden_dim % kGroupSize == 0,
              "hidden_dim must be divisible by group_size (128)");
  TORCH_CHECK(output.size(0) == num_total_tokens && output.size(1) == hidden_dim,
              "output shape mismatch");
  TORCH_CHECK(output_scale.size(0) == num_total_tokens && output_scale.size(1) == num_groups,
              "output_scale shape mismatch");

  if (num_total_tokens == 0) return;

  const at::cuda::OptionalCUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Grid/Block configuration
  constexpr int kBlockSize = 256;
  int num_block_col = (hidden_dim / kElementsPerThread + kBlockSize - 1) / kBlockSize;

  // Get SM count for occupancy-driven grid sizing
  int device_id = input.get_device();
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device_id);
  int num_block_hard = num_sm * 8;  // target ~8 blocks/SM for good occupancy

  int num_block_row = num_block_hard / num_block_col;
  if (num_block_row < 1) num_block_row = 1;
  int num_blocks = num_block_row * num_block_col;

  dim3 grid(num_blocks);
  dim3 block(kBlockSize);

  auto launch = [&](auto kernel) {
    kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_fp8_e4m3*>(output.data_ptr()),
        static_cast<float*>(output_scale.data_ptr()),
        static_cast<const __nv_bfloat16*>(input.data_ptr()),
        static_cast<const int32_t*>(expert_ids.data_ptr()),
        static_cast<int>(expert_step),
        num_total_tokens,
        hidden_dim,
        num_groups,
        num_block_col,
        num_block_row,
        static_cast<float>(swiglu_limit));
  };

  if (swiglu_limit > 0.0) {
    launch(sgl_act_mul_blockwise_quant_kernel<true>);
  } else {
    launch(sgl_act_mul_blockwise_quant_kernel<false>);
  }
}
