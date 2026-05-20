#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>

namespace {

static constexpr int kNumExperts = 384;
static constexpr int kWarpSize = 32;

// Small token kernel constants
static constexpr int kSmallTokenThreshold = 512;
static constexpr int kWarpsPerToken = 12;  // 384 / 32
static constexpr int kThreadsPerBlockSmall = kWarpsPerToken * kWarpSize;  // 384

// Large token kernel constants
static constexpr int kWarpsPerCTA = 6;
static constexpr int kVPT = 12;      // 384 / 32
static constexpr int kVecSize = 4;   // float4 vectorized loads

// Small token kernel: one block per token, 384 threads (one per expert).
// Sigmoid + bias -> iterative warp-level topk + merge -> renormalize.
__global__ void kimi_k2_moe_fused_gate_kernel_small_token(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int32_t* __restrict__ indices,
    int64_t num_rows,
    int64_t topk,
    bool renormalize,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  const int64_t row_idx = blockIdx.x;
  if (row_idx >= num_rows) return;

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;

  __shared__ float shared_scores[kNumExperts];
  __shared__ float shared_original_scores[kNumExperts];
  __shared__ int selected_experts[8];
  __shared__ float warp_maxs[kWarpsPerToken];
  __shared__ int warp_experts[kWarpsPerToken];

  if (tid < kNumExperts) {
    float input_val = input[row_idx * kNumExperts + tid];
    float bias_val = bias[tid];
    float sigmoid_val = 1.0f / (1.0f + expf(-input_val));
    float biased_val = sigmoid_val + bias_val;
    shared_scores[tid] = biased_val;
    shared_original_scores[tid] = sigmoid_val;
  }
  __syncthreads();

  for (int k = 0; k < topk; k++) {
    float my_val = (tid < kNumExperts) ? shared_scores[tid] : -FLT_MAX;
    int my_expert = tid;

    float warp_max_val = my_val;
    int warp_max_expert = my_expert;

#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
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

    if (warp_id == 0) {
      float final_max = (lane_id < kWarpsPerToken) ? warp_maxs[lane_id] : -FLT_MAX;
      int final_expert = (lane_id < kWarpsPerToken) ? warp_experts[lane_id] : -1;

#pragma unroll
      for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, final_max, offset);
        int other_expert = __shfl_down_sync(0xFFFFFFFF, final_expert, offset);
        if (other_val > final_max) {
          final_max = other_val;
          final_expert = other_expert;
        }
      }

      if (lane_id == 0) {
        selected_experts[k] = final_expert;
      }
    }
    __syncthreads();

    int selected = selected_experts[k];
    if (tid == selected) {
      shared_scores[tid] = -FLT_MAX;
    }
    __syncthreads();
  }

  if (tid == 0) {
    for (int k = 0; k < topk; k++) {
      int expert_id = selected_experts[k];
      if (expert_id >= 0 && expert_id < kNumExperts) {
        output[row_idx * topk + k] = shared_original_scores[expert_id];
        indices[row_idx * topk + k] = expert_id;
      } else {
        output[row_idx * topk + k] = 0.0f;
        indices[row_idx * topk + k] = 0;
      }
    }

    if (renormalize) {
      float sum = 0.0f;
      for (int k = 0; k < topk; k++) {
        sum += output[row_idx * topk + k];
      }
      if (sum > 0.0f) {
        for (int k = 0; k < topk; k++) {
          int64_t idx = row_idx * topk + k;
          output[idx] /= sum;
          if (apply_routed_scaling_factor_on_output) {
            output[idx] *= routed_scaling_factor;
          }
        }
      }
    }
  }
}

// Large token kernel: each block handles kWarpsPerCTA tokens with vectorized loads.
// Each warp independently processes one token.
__global__ void kimi_k2_moe_fused_gate_kernel_large_token(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int32_t* __restrict__ indices,
    int64_t num_rows,
    int64_t topk,
    bool renormalize,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  const int64_t row_idx = blockIdx.x * kWarpsPerCTA + threadIdx.y;
  if (row_idx >= num_rows) return;

  const int lane_id = threadIdx.x;
  const int warp_id = threadIdx.y;

  __shared__ float shared_scores[kNumExperts * kWarpsPerCTA];
  __shared__ float shared_original_scores[kNumExperts * kWarpsPerCTA];

  float* warp_scores = shared_scores + warp_id * kNumExperts;
  float* warp_original_scores = shared_original_scores + warp_id * kNumExperts;

  // Vectorized loading: each lane loads VPT/VEC_SIZE = 3 float4 chunks
  static constexpr int kVecPerLane = kVPT / kVecSize;  // 3
  const float4* input_vec = reinterpret_cast<const float4*>(input + row_idx * kNumExperts);
  const float4* bias_vec = reinterpret_cast<const float4*>(bias);

#pragma unroll
  for (int i = 0; i < kVecPerLane; i++) {
    int vec_idx = lane_id * kVecPerLane + i;
    float4 input_val = input_vec[vec_idx];
    float4 bias_val = bias_vec[vec_idx];

#pragma unroll
    for (int j = 0; j < kVecSize; j++) {
      int expert = vec_idx * kVecSize + j;
      float inp = reinterpret_cast<const float*>(&input_val)[j];
      float b = reinterpret_cast<const float*>(&bias_val)[j];
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

    for (int expert = lane_id; expert < kNumExperts; expert += kWarpSize) {
      if (warp_scores[expert] > max_val) {
        max_val = warp_scores[expert];
        max_expert = expert;
      }
    }

    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
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
        output[output_idx] = warp_original_scores[max_expert];
        indices[output_idx] = max_expert;
        warp_scores[max_expert] = -FLT_MAX;
      } else {
        output[output_idx] = 0.0f;
        indices[output_idx] = 0;
      }
    }

    __syncwarp();
  }

  __syncthreads();

  if (renormalize && lane_id == 0) {
    float sum = 0.0f;
    for (int k = 0; k < topk; k++) {
      sum += output[row_idx * topk + k];
    }

    if (sum > 0.0f) {
      for (int k = 0; k < topk; k++) {
        int64_t idx = row_idx * topk + k;
        output[idx] /= sum;
        if (apply_routed_scaling_factor_on_output) {
          output[idx] *= routed_scaling_factor;
        }
      }
    }
  }
}

void kimi_k2_moe_fused_gate(tvm::ffi::TensorView input,
                            tvm::ffi::TensorView bias,
                            tvm::ffi::TensorView output,
                            tvm::ffi::TensorView indices,
                            int64_t topk,
                            bool renormalize,
                            float routed_scaling_factor,
                            bool apply_routed_scaling_factor_on_output) {
  using namespace host;

  auto M = SymbolicSize{"num_rows"};
  auto E = SymbolicSize{"num_experts"};
  auto K = SymbolicSize{"topk"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({M, E})
      .with_dtype<fp32_t>()
      .with_device(device)
      .verify(input);
  TensorMatcher({E})
      .with_dtype<fp32_t>()
      .with_device(device)
      .verify(bias);
  TensorMatcher({M, K})
      .with_dtype<fp32_t>()
      .with_device(device)
      .verify(output);
  TensorMatcher({M, K})
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(indices);

  const auto num_rows = static_cast<int64_t>(M.unwrap());
  const auto num_experts = static_cast<int64_t>(E.unwrap());

  RuntimeCheck(num_experts == kNumExperts,
               "kimi_k2_moe_fused_gate: only supports 384 experts, got ", num_experts);
  RuntimeCheck(topk <= 8, "kimi_k2_moe_fused_gate: topk must be <= 8, got ", topk);

  auto dev = device.unwrap();

  if (num_rows <= kSmallTokenThreshold) {
    LaunchKernel(dim3(num_rows), dim3(kThreadsPerBlockSmall), dev)(
        kimi_k2_moe_fused_gate_kernel_small_token,
        static_cast<const float*>(input.data_ptr()),
        static_cast<const float*>(bias.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<int32_t*>(indices.data_ptr()),
        num_rows,
        topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
  } else {
    int64_t num_blocks = (num_rows + kWarpsPerCTA - 1) / kWarpsPerCTA;
    LaunchKernel(dim3(num_blocks), dim3(kWarpSize, kWarpsPerCTA), dev)(
        kimi_k2_moe_fused_gate_kernel_large_token,
        static_cast<const float*>(input.data_ptr()),
        static_cast<const float*>(bias.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<int32_t*>(indices.data_ptr()),
        num_rows,
        topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
  }
}

}  // namespace
