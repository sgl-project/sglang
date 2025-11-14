#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <torch/all.h>

#include <cfloat>

using bfloat16_t = cutlass::bfloat16_t;
using float16_t = cutlass::half_t;

// Kimi K2 specific constants
static constexpr int WARP_SIZE = 32;
static constexpr int WARPS_PER_CTA = 6;
static constexpr int NUM_EXPERTS = 384;
static constexpr int VPT = 12;  // 384 / 32 = 12

template <typename T>
__device__ inline bool cmp_gt(const T& a, const T& b) {
  return static_cast<float>(a) > static_cast<float>(b);
}

template <typename T>
__device__ inline bool cmp_eq(const T& a, const T& b) {
  return static_cast<float>(a) == static_cast<float>(b);
}

template <typename T>
__global__ void kimi_k2_moe_fused_gate_kernel(
    T* input,
    T* bias,
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

  for (int expert = lane_id; expert < NUM_EXPERTS; expert += WARP_SIZE) {
    T input_val = input[row_idx * NUM_EXPERTS + expert];
    T bias_val = bias[expert];

    float sigmoid_val = 1.0f / (1.0f + expf(-static_cast<float>(input_val)));
    float biased_val = sigmoid_val + static_cast<float>(bias_val);
    float sigmoid_val = 1.0f / (1.0f + expf(-static_cast<float>(input_val)));
    float biased_val = sigmoid_val + static_cast<float>(bias_val);

    warp_scores[expert] = biased_val;
    warp_original_scores[expert] = sigmoid_val;
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

  int64_t num_blocks = (num_rows + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 block_dim(WARP_SIZE, WARPS_PER_CTA);

  if (input.scalar_type() == at::kBFloat16) {
    kimi_k2_moe_fused_gate_kernel<bfloat16_t><<<num_blocks, block_dim, 0, stream>>>(
        reinterpret_cast<bfloat16_t*>(input.data_ptr()),
        reinterpret_cast<bfloat16_t*>(bias.data_ptr()),
        output.data_ptr<float>(),
        indices.data_ptr<int32_t>(),
        num_rows,
        topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
  } else if (input.scalar_type() == at::kHalf) {
    kimi_k2_moe_fused_gate_kernel<float16_t><<<num_blocks, block_dim, 0, stream>>>(
        reinterpret_cast<float16_t*>(input.data_ptr()),
        reinterpret_cast<float16_t*>(bias.data_ptr()),
        output.data_ptr<float>(),
        indices.data_ptr<int32_t>(),
        num_rows,
        topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
  } else if (input.scalar_type() == at::kFloat) {
    kimi_k2_moe_fused_gate_kernel<float><<<num_blocks, block_dim, 0, stream>>>(
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
    TORCH_CHECK(false, "Unsupported data type for kimi_k2_moe_fused_gate");
  }

  return {output, indices};
}
