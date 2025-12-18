#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cfloat>

// Kimi K2 specific constants
static constexpr int WARP_SIZE = 32;
static constexpr int WARPS_PER_CTA = 6;
static constexpr int NUM_EXPERTS = 384;
static constexpr int VPT = 12;  // 384 / 32 = 12

// Small token optimization constants
static constexpr int SMALL_TOKEN_THRESHOLD = 512;
static constexpr int WARPS_PER_TOKEN_SMALL = 12;  // Use 12 warps per token for small batches
static constexpr int THREADS_PER_BLOCK_SMALL = WARPS_PER_TOKEN_SMALL * WARP_SIZE;  // 384 threads

// Vectorization constants (used by large token kernel)
static constexpr int VEC_SIZE = 4;  // Use float4 for vectorized loads

// Small token optimized kernel: Each warp independently finds top-k, then merge, using warp-level topk
__global__ void kimi_k2_moe_fused_gate_kernel_small_token(
    float* input,
    float* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk,
    bool renormalize,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  int64_t row_idx = blockIdx.x;
  if (row_idx >= num_rows) return;

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  // Shared memory: biased scores and original scores
  __shared__ float shared_scores[NUM_EXPERTS];
  __shared__ float shared_original_scores[NUM_EXPERTS];
  // For storing selected top-k indices and values
  __shared__ int selected_experts[8];  // Up to topk=6, I use 8 for alignment
  __shared__ float selected_vals[8];
  // For warp-level reduction
  __shared__ float warp_maxs[WARPS_PER_TOKEN_SMALL];
  __shared__ int warp_experts[WARPS_PER_TOKEN_SMALL];

  // Load data: all 384 threads load one expert each
  if (tid < NUM_EXPERTS) {
    float input_val = input[row_idx * NUM_EXPERTS + tid];
    float bias_val = bias[tid];
    float sigmoid_val = 1.0f / (1.0f + expf(-input_val));
    float biased_val = sigmoid_val + bias_val;
    shared_scores[tid] = biased_val;
    shared_original_scores[tid] = sigmoid_val;
  }

  __syncthreads();

  // Find top-k using iterative selection, each iteration finds the next maximum
  for (int k = 0; k < topk; k++) {
    // Each thread holds one expert's value
    float my_val = (tid < NUM_EXPERTS) ? shared_scores[tid] : -FLT_MAX;
    int my_expert = tid;

    // Use warp-level reduction first
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

    // Warp leaders write to shared memory
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

    // Mark the selected expert as used for next iteration
    // All threads can read from selected_experts[k]
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
        output_ptr[row_idx * topk + k] = shared_original_scores[expert_id];
        indices_ptr[row_idx * topk + k] = expert_id;
      }
    }

    // Renormalization
    if (renormalize) {
      float sum = 0.0f;
      for (int k = 0; k < topk; k++) {
        sum += output_ptr[row_idx * topk + k];
      }

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

// Large token kernel: Original implementation with vectorized loads
__global__ void kimi_k2_moe_fused_gate_kernel(
    float* input,
    float* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk,
    bool renormalize,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  int64_t row_idx = blockIdx.x * WARPS_PER_CTA + threadIdx.y;
  if (row_idx >= num_rows) return;

  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;

  __shared__ float shared_scores[NUM_EXPERTS * WARPS_PER_CTA];
  __shared__ float shared_original_scores[NUM_EXPERTS * WARPS_PER_CTA];

  float* warp_scores = shared_scores + warp_id * NUM_EXPERTS;
  float* warp_original_scores = shared_original_scores + warp_id * NUM_EXPERTS;

  // Vectorized loading: each lane loads multiple float4 chunks
  // VPT = 12, so we load 12/4 = 3 float4 per lane
  const int VEC_PER_LANE = VPT / VEC_SIZE;  // 3
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

    if (lane_id == 0 && max_expert != -1) {
      int64_t output_idx = row_idx * topk + k;
      output_ptr[output_idx] = warp_original_scores[max_expert];
      indices_ptr[output_idx] = max_expert;
      warp_scores[max_expert] = -FLT_MAX;
    }

    __syncwarp();
  }

  __syncthreads();

  if (renormalize && lane_id == 0) {
    float sum = 0.0f;
    for (int k = 0; k < topk; k++) {
      sum += output_ptr[row_idx * topk + k];
    }

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

std::vector<at::Tensor> kimi_k2_moe_fused_gate(
    at::Tensor& input,
    at::Tensor& bias,
    int64_t topk,
    bool renormalize,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  int64_t num_rows = input.size(0);
  int32_t num_experts = input.size(1);

  // Assert: Only support 384 experts
  TORCH_CHECK(num_experts == 384, "kimi_k2_moe_fused_gate only supports 384 experts, but got ", num_experts);
  TORCH_CHECK(input.dtype() == bias.dtype(), "input and bias should have the same dtype");

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto output = torch::empty({num_rows, topk}, options);
  auto indices = torch::empty({num_rows, topk}, options.dtype(torch::kInt32));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Only support float32
  TORCH_CHECK(input.scalar_type() == at::kFloat, "kimi_k2_moe_fused_gate only supports float32 input");
  TORCH_CHECK(bias.scalar_type() == at::kFloat, "kimi_k2_moe_fused_gate only supports float32 bias");

  bool use_small_token_kernel = num_rows <= SMALL_TOKEN_THRESHOLD;

  if (use_small_token_kernel) {
    // Small token kernel: Each block handles 1 token with multiple warps collaborating
    int64_t num_blocks = num_rows;
    dim3 block_dim(THREADS_PER_BLOCK_SMALL);

    kimi_k2_moe_fused_gate_kernel_small_token<<<num_blocks, block_dim, 0, stream>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        indices.data_ptr<int32_t>(),
        num_rows,
        topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
  } else {
    // Large token kernel: Original implementation
    int64_t num_blocks = (num_rows + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
    dim3 block_dim(WARP_SIZE, WARPS_PER_CTA);

    kimi_k2_moe_fused_gate_kernel<<<num_blocks, block_dim, 0, stream>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        indices.data_ptr<int32_t>(),
        num_rows,
        topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
  }

  return {output, indices};
}
