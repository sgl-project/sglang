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

//------------------------------------------------------------------------------
// Simplified Kernel for Kimi K2 (384 experts, num_expert_group=1)
//------------------------------------------------------------------------------
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
  
  // Each warp processes one row
  int64_t thread_row = blockIdx.x * WARPS_PER_CTA + threadIdx.y;
  if (thread_row >= num_rows) return;

  int tidx = threadIdx.x;
  int first_expert = tidx * VPT;
  
  // Read data for this thread
  T row_chunk[VPT];
  T bias_chunk[VPT];
  
  T* thread_row_ptr = input + thread_row * NUM_EXPERTS + first_expert;
  T* bias_thread_ptr = bias + first_expert;
  
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = thread_row_ptr[ii];
    bias_chunk[ii] = bias_thread_ptr[ii];
  }

  // Sigmoid activation
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = static_cast<T>(1.0f / (1.0f + expf(-float(row_chunk[ii]))));
  }

  // Add bias
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    bias_chunk[ii] = row_chunk[ii] + bias_chunk[ii];
  }

  // TopK selection
  float output_sum = 0.0f;
  
  for (int k_idx = 0; k_idx < topk; ++k_idx) {
    // Find local max in thread's chunk
    T max_val = bias_chunk[0];
    int expert = first_expert;
    
    if (!cmp_eq(max_val, static_cast<T>(-FLT_MAX))) {
#pragma unroll
      for (int ii = 1; ii < VPT; ++ii) {
        T val = bias_chunk[ii];
        if (cmp_gt(val, max_val)) {
          max_val = val;
          expert = first_expert + ii;
        }
      }
    } else {
      max_val = static_cast<T>(-FLT_MAX);
    }

    // Warp reduction to find global max
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
      T other_max = static_cast<T>(__shfl_xor_sync(0xFFFFFFFF, static_cast<float>(max_val), mask, WARP_SIZE));
      int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, WARP_SIZE);

      if (cmp_gt(other_max, max_val) || (cmp_eq(other_max, max_val) && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    // Write result
    int thread_to_clear = expert / VPT;
    int64_t idx = topk * thread_row + k_idx;

    if (tidx == thread_to_clear) {
      int expert_to_clear = expert % VPT;
      bias_chunk[expert_to_clear] = static_cast<T>(-FLT_MAX);
      output_ptr[idx] = static_cast<float>(row_chunk[expert_to_clear]);
      indices_ptr[idx] = static_cast<int32_t>(expert);
    }

    __syncthreads();

    if (tidx == 0) {
      output_sum += output_ptr[idx];
    }
  }

  // Normalize weights
  if (renormalize && tidx == 0) {
#pragma unroll
    for (int ii = 0; ii < topk; ++ii) {
      int64_t idx = topk * thread_row + ii;
      output_ptr[idx] = output_ptr[idx] / output_sum;
      if (apply_routed_scaling_factor_on_output) {
        output_ptr[idx] *= routed_scaling_factor;
      }
    }
  }
}

//------------------------------------------------------------------------------
// Host Launcher
//------------------------------------------------------------------------------
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
  TORCH_CHECK(num_experts == 384, 
      "kimi_k2_moe_fused_gate only supports 384 experts, but got ", num_experts);
  TORCH_CHECK(input.dtype() == bias.dtype(), 
      "input and bias should have the same dtype");
  
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
        num_rows, topk, renormalize, routed_scaling_factor, 
        apply_routed_scaling_factor_on_output);
  } else if (input.scalar_type() == at::kHalf) {
    kimi_k2_moe_fused_gate_kernel<float16_t><<<num_blocks, block_dim, 0, stream>>>(
        reinterpret_cast<float16_t*>(input.data_ptr()),
        reinterpret_cast<float16_t*>(bias.data_ptr()),
        output.data_ptr<float>(),
        indices.data_ptr<int32_t>(),
        num_rows, topk, renormalize, routed_scaling_factor, 
        apply_routed_scaling_factor_on_output);
  } else if (input.scalar_type() == at::kFloat) {
    kimi_k2_moe_fused_gate_kernel<float><<<num_blocks, block_dim, 0, stream>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        indices.data_ptr<int32_t>(),
        num_rows, topk, renormalize, routed_scaling_factor, 
        apply_routed_scaling_factor_on_output);
  } else {
    TORCH_CHECK(false, "Unsupported data type for kimi_k2_moe_fused_gate");
  }

  return {output, indices};
}
