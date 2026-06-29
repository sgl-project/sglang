#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>

namespace {

// ---------------------------------------------------------------------------
// Kimi K2 constants (hard-coded for 384 experts)
// ---------------------------------------------------------------------------
static constexpr int K2_WARP_SIZE = 32;
static constexpr int K2_WARPS_PER_CTA = 6;
static constexpr int K2_NUM_EXPERTS = 384;
static constexpr int K2_VPT = K2_NUM_EXPERTS / K2_WARP_SIZE;  // 12

// Small-token kernel constants
static constexpr int K2_SMALL_TOKEN_THRESHOLD = 512;
static constexpr int K2_WARPS_PER_TOKEN_SMALL = 12;
static constexpr int K2_THREADS_PER_BLOCK_SMALL = K2_WARPS_PER_TOKEN_SMALL * K2_WARP_SIZE;  // 384

// Vectorised-load constants (large-token kernel)
static constexpr int K2_VEC_SIZE = 4;                         // float4
static constexpr int K2_VEC_PER_LANE = K2_VPT / K2_VEC_SIZE;  // 3

// ---------------------------------------------------------------------------
// Small-token kernel: one CTA per row, 12 warps collaborate on a single row.
// ---------------------------------------------------------------------------
__global__ void kimi_k2_moe_fused_gate_kernel_small_token(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output_ptr,
    int32_t* __restrict__ indices_ptr,
    int64_t num_rows,
    int64_t topk,
    bool renormalize,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  int64_t row_idx = blockIdx.x;
  if (row_idx >= num_rows) return;

  int tid = threadIdx.x;
  int warp_id = tid / K2_WARP_SIZE;
  int lane_id = tid % K2_WARP_SIZE;

  __shared__ float shared_scores[K2_NUM_EXPERTS];
  __shared__ float shared_original_scores[K2_NUM_EXPERTS];
  __shared__ int selected_experts[8];  // topk <= 6; 8 for alignment
  __shared__ float selected_vals[8];
  __shared__ float warp_maxs[K2_WARPS_PER_TOKEN_SMALL];
  __shared__ int warp_experts[K2_WARPS_PER_TOKEN_SMALL];

  // Load: every thread handles one expert.
  if (tid < K2_NUM_EXPERTS) {
    float inp = input[row_idx * K2_NUM_EXPERTS + tid];
    float b = bias[tid];
    float sigmoid_val = 1.0f / (1.0f + expf(-inp));
    shared_scores[tid] = sigmoid_val + b;
    shared_original_scores[tid] = sigmoid_val;
  }

  __syncthreads();

  // Iterative top-k: each iteration extracts the global maximum.
  for (int k = 0; k < topk; k++) {
    float my_val = (tid < K2_NUM_EXPERTS) ? shared_scores[tid] : -FLT_MAX;
    int my_expert = tid;

    // Warp-level reduction.
    float warp_max_val = my_val;
    int warp_max_expert = my_expert;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xFFFFFFFF, warp_max_val, offset);
      int other_expert = __shfl_down_sync(0xFFFFFFFF, warp_max_expert, offset);
      if (other_val > warp_max_val) {
        warp_max_val = other_val;
        warp_max_expert = other_expert;
      }
    }

    if (lane_id == 0) {
      warp_maxs[warp_id] = warp_max_val;
      warp_experts[warp_id] = warp_max_expert;
    }

    __syncthreads();

    // Reduce across warps (first warp only).
    if (warp_id == 0) {
      float final_max = (lane_id < K2_WARPS_PER_TOKEN_SMALL) ? warp_maxs[lane_id] : -FLT_MAX;
      int final_expert = (lane_id < K2_WARPS_PER_TOKEN_SMALL) ? warp_experts[lane_id] : -1;
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, final_max, offset);
        int other_expert = __shfl_down_sync(0xFFFFFFFF, final_expert, offset);
        if (other_val > final_max) {
          final_max = other_val;
          final_expert = other_expert;
        }
      }
      if (lane_id == 0) {
        selected_experts[k] = final_expert;
        selected_vals[k] = final_max;
      }
    }

    __syncthreads();

    // Mask out the selected expert so the next iteration skips it.
    if (tid == selected_experts[k]) {
      shared_scores[tid] = -FLT_MAX;
    }

    __syncthreads();
  }

  // Write outputs (thread 0 only).
  if (tid == 0) {
    for (int k = 0; k < topk; k++) {
      int expert_id = selected_experts[k];
      if (expert_id >= 0 && expert_id < K2_NUM_EXPERTS) {
        output_ptr[row_idx * topk + k] = shared_original_scores[expert_id];
        indices_ptr[row_idx * topk + k] = expert_id;
      } else {
        output_ptr[row_idx * topk + k] = 0.0f;
        indices_ptr[row_idx * topk + k] = 0;
      }
    }

    if (renormalize) {
      float sum = 0.0f;
      for (int k = 0; k < topk; k++)
        sum += output_ptr[row_idx * topk + k];
      if (sum > 0.0f) {
        for (int k = 0; k < topk; k++) {
          int64_t idx = row_idx * topk + k;
          output_ptr[idx] /= sum;
          if (apply_routed_scaling_factor_on_output) {
            output_ptr[idx] *= static_cast<float>(routed_scaling_factor);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Large-token kernel: WARPS_PER_CTA rows per CTA, vectorised loads.
// ---------------------------------------------------------------------------
__global__ void kimi_k2_moe_fused_gate_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output_ptr,
    int32_t* __restrict__ indices_ptr,
    int64_t num_rows,
    int64_t topk,
    bool renormalize,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  int64_t row_idx = blockIdx.x * K2_WARPS_PER_CTA + threadIdx.y;
  if (row_idx >= num_rows) return;

  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;

  __shared__ float shared_scores[K2_NUM_EXPERTS * K2_WARPS_PER_CTA];
  __shared__ float shared_original_scores[K2_NUM_EXPERTS * K2_WARPS_PER_CTA];

  float* warp_scores = shared_scores + warp_id * K2_NUM_EXPERTS;
  float* warp_original_scores = shared_original_scores + warp_id * K2_NUM_EXPERTS;

  // Vectorised load: each lane loads K2_VEC_PER_LANE float4 chunks.
  const float4* input_vec = reinterpret_cast<const float4*>(input + row_idx * K2_NUM_EXPERTS);
  const float4* bias_vec = reinterpret_cast<const float4*>(bias);

#pragma unroll
  for (int i = 0; i < K2_VEC_PER_LANE; i++) {
    int vec_idx = lane_id * K2_VEC_PER_LANE + i;
    float4 input_val = input_vec[vec_idx];
    float4 bias_val = bias_vec[vec_idx];
#pragma unroll
    for (int j = 0; j < K2_VEC_SIZE; j++) {
      int expert = vec_idx * K2_VEC_SIZE + j;
      float inp = reinterpret_cast<const float*>(&input_val)[j];
      float b = reinterpret_cast<const float*>(&bias_val)[j];
      float sigmoid_val = 1.0f / (1.0f + expf(-inp));
      warp_scores[expert] = sigmoid_val + b;
      warp_original_scores[expert] = sigmoid_val;
    }
  }

  __syncthreads();

  for (int k = 0; k < topk; k++) {
    float max_val = -FLT_MAX;
    int max_expert = -1;

    for (int expert = lane_id; expert < K2_NUM_EXPERTS; expert += K2_WARP_SIZE) {
      if (warp_scores[expert] > max_val) {
        max_val = warp_scores[expert];
        max_expert = expert;
      }
    }

    for (int offset = K2_WARP_SIZE / 2; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
      int other_expert = __shfl_down_sync(0xFFFFFFFF, max_expert, offset);
      if (other_val > max_val || (other_val == max_val && other_expert < max_expert)) {
        max_val = other_val;
        max_expert = other_expert;
      }
    }

    if (lane_id == 0) {
      int64_t output_idx = row_idx * topk + k;
      if (max_expert != -1) {
        output_ptr[output_idx] = warp_original_scores[max_expert];
        indices_ptr[output_idx] = max_expert;
        warp_scores[max_expert] = -FLT_MAX;
      } else {
        output_ptr[output_idx] = 0.0f;
        indices_ptr[output_idx] = 0;
      }
    }

    __syncwarp();
  }

  __syncthreads();

  if (renormalize && lane_id == 0) {
    float sum = 0.0f;
    for (int k = 0; k < topk; k++)
      sum += output_ptr[row_idx * topk + k];
    if (sum > 0.0f) {
      for (int k = 0; k < topk; k++) {
        int64_t idx = row_idx * topk + k;
        output_ptr[idx] /= sum;
        if (apply_routed_scaling_factor_on_output) {
          output_ptr[idx] *= static_cast<float>(routed_scaling_factor);
        }
      }
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
void kimi_k2_moe_fused_gate(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView indices,
    int64_t topk,
    bool renormalize,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  using namespace host;

  // --- Input validation ---
  RuntimeCheck(input.dim() == 2, "input must be 2-D, got dim=", input.dim());
  RuntimeCheck(bias.dim() == 1, "bias must be 1-D, got dim=", bias.dim());

  const int64_t num_rows = input.shape()[0];
  const int64_t num_experts = input.shape()[1];

  RuntimeCheck(
      num_experts == K2_NUM_EXPERTS,
      "kimi_k2_moe_fused_gate only supports ",
      K2_NUM_EXPERTS,
      " experts, got ",
      num_experts);
  RuntimeCheck(bias.shape()[0] == num_experts, "bias size must match num_experts");
  RuntimeCheck(
      output.dim() == 2 && output.shape()[0] == num_rows && output.shape()[1] == topk,
      "output must be [num_rows, topk]");
  RuntimeCheck(
      indices.dim() == 2 && indices.shape()[0] == num_rows && indices.shape()[1] == topk,
      "indices must be [num_rows, topk]");
  RuntimeCheck(topk > 0 && topk <= K2_NUM_EXPERTS, "topk out of range");

  // Dtype checks (float32 only)
  RuntimeCheck(input.dtype().code == DLDataTypeCode::kDLFloat && input.dtype().bits == 32, "input must be float32");
  RuntimeCheck(bias.dtype().code == DLDataTypeCode::kDLFloat && bias.dtype().bits == 32, "bias must be float32");

  const float* inp_ptr = static_cast<const float*>(input.data_ptr());
  const float* bias_ptr = static_cast<const float*>(bias.data_ptr());
  float* out_ptr = static_cast<float*>(output.data_ptr());
  int32_t* idx_ptr = static_cast<int32_t*>(indices.data_ptr());

  cudaStream_t stream = LaunchKernel::resolve_device(input.device());

  if (num_rows <= K2_SMALL_TOKEN_THRESHOLD) {
    // Small-token kernel: one CTA per row, 384 threads per CTA.
    LaunchKernel(
        {static_cast<uint32_t>(num_rows), 1u, 1u}, {static_cast<uint32_t>(K2_THREADS_PER_BLOCK_SMALL), 1u, 1u}, stream)(
        kimi_k2_moe_fused_gate_kernel_small_token,
        inp_ptr,
        bias_ptr,
        out_ptr,
        idx_ptr,
        num_rows,
        topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
  } else {
    // Large-token kernel: K2_WARPS_PER_CTA rows per CTA.
    int64_t num_blocks = (num_rows + K2_WARPS_PER_CTA - 1) / K2_WARPS_PER_CTA;
    LaunchKernel(
        {static_cast<uint32_t>(num_blocks), 1u, 1u},
        {static_cast<uint32_t>(K2_WARP_SIZE), static_cast<uint32_t>(K2_WARPS_PER_CTA), 1u},
        stream)(
        kimi_k2_moe_fused_gate_kernel,
        inp_ptr,
        bias_ptr,
        out_ptr,
        idx_ptr,
        num_rows,
        topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
  }
}
