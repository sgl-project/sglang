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

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

namespace moe {

// Common constants
static constexpr int WARPS_PER_CTA = 6;
static constexpr int SMALL_TOKEN_THRESHOLD = 512;
static constexpr int VEC_SIZE = 4;

// Small token optimized kernel: Each block handles 1 token, NUM_EXPERTS threads collaborate
// to find top-k using iterative warp-level reduction.
// output_stride: row stride in output/indices tensors (>= topk, allows reserved slots for shared experts)
template <int NUM_EXPERTS>
__global__ void moe_fused_gate_ungrouped_kernel_small_token(
    float* input,
    float* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk,
    int64_t output_stride,
    bool renormalize,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  static constexpr int WARPS_PER_TOKEN_SMALL = NUM_EXPERTS / WARP_SIZE;

  int64_t row_idx = blockIdx.x;
  if (row_idx >= num_rows) return;

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  __shared__ float shared_scores[NUM_EXPERTS];
  __shared__ float shared_original_scores[NUM_EXPERTS];
  __shared__ int selected_experts[8];
  __shared__ float selected_vals[8];
  __shared__ float warp_maxs[WARPS_PER_TOKEN_SMALL];
  __shared__ int warp_experts[WARPS_PER_TOKEN_SMALL];

  // Load data: all NUM_EXPERTS threads load one expert each
  if (tid < NUM_EXPERTS) {
    float input_val = input[row_idx * NUM_EXPERTS + tid];
    float bias_val = bias[tid];
    float sigmoid_val = 1.0f / (1.0f + expf(-input_val));
    float biased_val = sigmoid_val + bias_val;
    shared_scores[tid] = biased_val;
    shared_original_scores[tid] = sigmoid_val;
  }

  __syncthreads();

  // Find top-k using iterative selection
  for (int k = 0; k < topk; k++) {
    float my_val = (tid < NUM_EXPERTS) ? shared_scores[tid] : -FLT_MAX;
    int my_expert = tid;

    // Warp-level reduction
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

    // Final reduction among warps (done by first warp)
    if (warp_id == 0) {
      float final_max = (lane_id < WARPS_PER_TOKEN_SMALL) ? warp_maxs[lane_id] : -FLT_MAX;
      int final_expert = (lane_id < WARPS_PER_TOKEN_SMALL) ? warp_experts[lane_id] : -1;

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

    // Mark the selected expert as used
    int selected = selected_experts[k];
    if (tid == selected) {
      shared_scores[tid] = -FLT_MAX;
    }

    __syncthreads();
  }

  // Write output (done by thread 0)
  if (tid == 0) {
    for (int k = 0; k < topk; k++) {
      int expert_id = selected_experts[k];
      if (expert_id >= 0 && expert_id < NUM_EXPERTS) {
        output_ptr[row_idx * output_stride + k] = shared_original_scores[expert_id];
        indices_ptr[row_idx * output_stride + k] = expert_id;
      } else {
        output_ptr[row_idx * output_stride + k] = 0.0f;
        indices_ptr[row_idx * output_stride + k] = 0;
      }
    }

    if (renormalize) {
      float sum = 0.0f;
      for (int k = 0; k < topk; k++) {
        sum += output_ptr[row_idx * output_stride + k];
      }

      if (sum > 0.0f) {
        for (int k = 0; k < topk; k++) {
          int64_t idx = row_idx * output_stride + k;
          output_ptr[idx] /= sum;
          if (apply_routed_scaling_factor_on_output) {
            output_ptr[idx] *= static_cast<float>(routed_scaling_factor);
          }
        }
      }
    }
  }
}

// Large token kernel: Each warp handles one token with vectorized loads
// output_stride: row stride in output/indices tensors (>= topk, allows reserved slots for shared experts)
template <int NUM_EXPERTS>
__global__ void moe_fused_gate_ungrouped_kernel(
    float* input,
    float* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk,
    int64_t output_stride,
    bool renormalize,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  static constexpr int VPT = NUM_EXPERTS / WARP_SIZE;

  int64_t row_idx = blockIdx.x * WARPS_PER_CTA + threadIdx.y;
  if (row_idx >= num_rows) return;

  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;

  __shared__ float shared_scores[NUM_EXPERTS * WARPS_PER_CTA];
  __shared__ float shared_original_scores[NUM_EXPERTS * WARPS_PER_CTA];

  float* warp_scores = shared_scores + warp_id * NUM_EXPERTS;
  float* warp_original_scores = shared_original_scores + warp_id * NUM_EXPERTS;

  // Vectorized loading
  static constexpr int VEC_PER_LANE = VPT / VEC_SIZE;
  float4* input_vec = reinterpret_cast<float4*>(input + row_idx * NUM_EXPERTS);
  float4* bias_vec = reinterpret_cast<float4*>(bias);

#pragma unroll
  for (int i = 0; i < VEC_PER_LANE; i++) {
    int vec_idx = lane_id * VEC_PER_LANE + i;
    float4 input_val = input_vec[vec_idx];
    float4 bias_val = bias_vec[vec_idx];

#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      int expert = vec_idx * VEC_SIZE + j;
      float inp = ((float*)&input_val)[j];
      float b = ((float*)&bias_val)[j];
      float sigmoid_val = 1.0f / (1.0f + expf(-inp));
      float biased_val = sigmoid_val + b;
      warp_scores[expert] = biased_val;
      warp_original_scores[expert] = sigmoid_val;
    }
  }

  __syncthreads();

  for (int k = 0; k < topk; k++) {
    float max_val = -FLT_MAX;
    int max_expert = -1;

    for (int expert = lane_id; expert < NUM_EXPERTS; expert += WARP_SIZE) {
      if (warp_scores[expert] > max_val) {
        max_val = warp_scores[expert];
        max_expert = expert;
      }
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
      int other_expert = __shfl_down_sync(0xFFFFFFFF, max_expert, offset);

      if (other_val > max_val || (other_val == max_val && other_expert < max_expert)) {
        max_val = other_val;
        max_expert = other_expert;
      }
    }

    if (lane_id == 0) {
      int64_t output_idx = row_idx * output_stride + k;
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
    for (int k = 0; k < topk; k++) {
      sum += output_ptr[row_idx * output_stride + k];
    }

    if (sum > 0.0f) {
      for (int k = 0; k < topk; k++) {
        int64_t idx = row_idx * output_stride + k;
        output_ptr[idx] /= sum;
        if (apply_routed_scaling_factor_on_output) {
          output_ptr[idx] *= static_cast<float>(routed_scaling_factor);
        }
      }
    }
  }
}

}  // namespace moe

namespace {

template <int NUM_EXPERTS>
struct MoeFusedGateUngroupedKernel {
  static void
  run(tvm::ffi::TensorView input,
      tvm::ffi::TensorView bias,
      tvm::ffi::TensorView output,
      tvm::ffi::TensorView indices,
      int64_t topk,
      bool renormalize,
      double routed_scaling_factor,
      bool apply_routed_scaling_factor_on_output) {
    using namespace host;

    auto device = input.device();
    const cudaStream_t stream = LaunchKernel::resolve_device(device);

    int64_t num_rows = input.size(0);
    int64_t output_stride = output.size(1);

    float* input_ptr = static_cast<float*>(input.data_ptr());
    float* bias_ptr = static_cast<float*>(bias.data_ptr());
    float* output_ptr = static_cast<float*>(output.data_ptr());
    int32_t* indices_ptr = static_cast<int32_t*>(indices.data_ptr());

    if (num_rows <= moe::SMALL_TOKEN_THRESHOLD) {
      LaunchKernel(dim3(num_rows), dim3(NUM_EXPERTS), stream)(
          moe::moe_fused_gate_ungrouped_kernel_small_token<NUM_EXPERTS>,
          input_ptr,
          bias_ptr,
          output_ptr,
          indices_ptr,
          num_rows,
          topk,
          output_stride,
          renormalize,
          routed_scaling_factor,
          apply_routed_scaling_factor_on_output);
    } else {
      int64_t num_blocks = (num_rows + moe::WARPS_PER_CTA - 1) / moe::WARPS_PER_CTA;
      LaunchKernel(dim3(num_blocks), dim3(WARP_SIZE, moe::WARPS_PER_CTA), stream)(
          moe::moe_fused_gate_ungrouped_kernel<NUM_EXPERTS>,
          input_ptr,
          bias_ptr,
          output_ptr,
          indices_ptr,
          num_rows,
          topk,
          output_stride,
          renormalize,
          routed_scaling_factor,
          apply_routed_scaling_factor_on_output);
    }
  }
};

}  // namespace
