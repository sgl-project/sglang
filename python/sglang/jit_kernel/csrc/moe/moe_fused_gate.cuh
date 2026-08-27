#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>

namespace {

constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kWarpsPerCTA = 6;
constexpr uint32_t kSmallTokenThreshold = 512;
constexpr uint32_t kMaxExperts = 512;
constexpr uint32_t kMaxTopK = 16;

enum class ScoringFunc : uint32_t {
  kSigmoid = 0,
  kSqrtSoftplus = 1,
};

struct MoEFusedGateParams {
  const float* __restrict__ input;
  const float* __restrict__ bias;
  float* __restrict__ output;
  int32_t* __restrict__ indices;
  uint32_t num_rows;
  uint32_t num_experts;
  uint32_t topk;
  uint32_t num_fused_shared_experts;
  bool renormalize;
  float routed_scaling_factor;
  bool apply_routed_scaling_factor_on_output;
};

template <ScoringFunc kScoringFunc>
__device__ __forceinline__ float compute_score(float x) {
  if constexpr (kScoringFunc == ScoringFunc::kSigmoid) {
    // sigmoid(x) = 1 / (1 + exp(-x))
    return 1.0f / (1.0f + expf(-x));
  } else {
    // sqrt(softplus(x)) = sqrt(log(1 + exp(x)))
    float softplus = log1pf(expf(x));
    return sqrtf(softplus);
  }
}

template <uint32_t kWarpsPerToken, ScoringFunc kScoringFunc>
__global__ void moe_fused_gate_kernel_small_token(const MoEFusedGateParams __grid_constant__ params) {
  const auto& [input, bias, output, indices, num_rows, num_experts, topk, num_fused_shared_experts, renormalize, routed_scaling_factor, apply_routed_scaling_factor_on_output] =
      params;

  uint32_t row_idx = blockIdx.x;
  if (row_idx >= num_rows) return;

  // number of routed experts to select (excluding fused shared experts)
  const uint32_t topk_routed = topk - num_fused_shared_experts;

  uint32_t tid = threadIdx.x;
  uint32_t warp_id = tid / kWarpSize;
  uint32_t lane_id = tid % kWarpSize;

  extern __shared__ float shared_mem[];
  float* shared_scores = shared_mem;
  float* shared_original_scores = shared_mem + num_experts;

  // For warp-level reduction
  __shared__ float warp_maxs[kWarpsPerToken];
  __shared__ int warp_experts[kWarpsPerToken];
  __shared__ int selected_experts[kMaxTopK];

  for (uint32_t e = tid; e < num_experts; e += blockDim.x) {
    float input_val = input[row_idx * num_experts + e];
    float bias_val = bias[e];
    float score_val = compute_score<kScoringFunc>(input_val);
    float biased_val = score_val + bias_val;
    shared_scores[e] = biased_val;
    shared_original_scores[e] = score_val;
  }

  __syncthreads();

  // only select topk_routed experts (excluding shared experts)
  for (uint32_t k = 0; k < topk_routed; k++) {
    float my_val = -FLT_MAX;
    int my_expert = -1;
    for (uint32_t e = tid; e < num_experts; e += blockDim.x) {
      if (shared_scores[e] > my_val) {
        my_val = shared_scores[e];
        my_expert = e;
      }
    }

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

    if (lane_id == 0 && warp_id < kWarpsPerToken) {
      warp_maxs[warp_id] = warp_max_val;
      warp_experts[warp_id] = warp_max_expert;
    }

    __syncthreads();

    if (warp_id == 0) {
      float final_max = (lane_id < kWarpsPerToken) ? warp_maxs[lane_id] : -FLT_MAX;
      int final_expert = (lane_id < kWarpsPerToken) ? warp_experts[lane_id] : -1;

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
      }
    }

    __syncthreads();

    int selected = selected_experts[k];
    if (selected >= 0 && tid == 0) {
      shared_scores[selected] = -FLT_MAX;
    }

    __syncthreads();
  }

  static_assert(kMaxTopK <= device::kWarpThreads);
  if (tid >= device::kWarpThreads) return;

  // only use the first warp to perform write to global operation
  float routed_weight = 0.0f;
  int32_t selected_expert = 0;
  if (tid < topk_routed) {
    int expert_id = selected_experts[tid];
    float score = shared_original_scores[expert_id];
    if (expert_id >= 0 && expert_id < static_cast<int>(num_experts)) {
      routed_weight = score;
      selected_expert = expert_id;
    }
  }
  const auto routed_sum = device::warp::reduce_sum<kMaxTopK>(routed_weight);
  if (tid < topk) {
    const bool is_shared = tid >= topk_routed;
    const auto output_offset = row_idx * topk + tid;
    const auto weight = is_shared ? (routed_sum / routed_scaling_factor) : routed_weight;
    const auto expert_id = is_shared ? (num_experts + tid - topk_routed) : selected_expert;
    const auto scale = apply_routed_scaling_factor_on_output ? routed_scaling_factor : 1.0f;
    const auto norm = renormalize && routed_sum > 0.0f ? routed_sum : 1.0f;
    output[output_offset] = weight / norm * scale;
    indices[output_offset] = expert_id;
  }
}

template <ScoringFunc kScoringFunc>
__global__ void moe_fused_gate_kernel(const MoEFusedGateParams __grid_constant__ params) {
  const auto& [input, bias, output, indices, num_rows, num_experts, topk, num_fused_shared_experts, renormalize, routed_scaling_factor, apply_routed_scaling_factor_on_output] =
      params;

  uint32_t row_idx = blockIdx.x * kWarpsPerCTA + threadIdx.y;
  if (row_idx >= num_rows) return;

  // number of routed experts to select (excluding fused shared experts)
  const uint32_t topk_routed = topk - num_fused_shared_experts;

  uint32_t lane_id = threadIdx.x;
  uint32_t warp_id = threadIdx.y;

  extern __shared__ float shared_mem[];
  float* shared_scores = shared_mem + warp_id * num_experts * 2;
  float* shared_original_scores = shared_scores + num_experts;
  __shared__ int selected_experts[kWarpsPerCTA][kMaxTopK];
  int* warp_selected_experts = selected_experts[warp_id];

  for (uint32_t e = lane_id; e < num_experts; e += kWarpSize) {
    float input_val = input[row_idx * num_experts + e];
    float bias_val = bias[e];
    float score_val = compute_score<kScoringFunc>(input_val);
    float biased_val = score_val + bias_val;
    shared_scores[e] = biased_val;
    shared_original_scores[e] = score_val;
  }

  __syncwarp();

  // only select topk_routed experts
  for (uint32_t k = 0; k < topk_routed; k++) {
    float max_val = -FLT_MAX;
    int max_expert = -1;

    for (uint32_t expert = lane_id; expert < num_experts; expert += kWarpSize) {
      if (shared_scores[expert] > max_val) {
        max_val = shared_scores[expert];
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
      warp_selected_experts[k] = max_expert;
      if (max_expert != -1) {
        shared_scores[max_expert] = -FLT_MAX;
      }
    }

    __syncwarp();
  }

  static_assert(kMaxTopK <= device::kWarpThreads);

  float routed_weight = 0.0f;
  int32_t selected_expert = 0;
  if (lane_id < topk_routed) {
    int expert_id = warp_selected_experts[lane_id];
    if (expert_id >= 0 && expert_id < static_cast<int>(num_experts)) {
      routed_weight = shared_original_scores[expert_id];
      selected_expert = expert_id;
    }
  }
  const auto routed_sum = device::warp::reduce_sum<kMaxTopK>(routed_weight);
  if (lane_id < topk) {
    const bool is_shared = lane_id >= topk_routed;
    const auto output_idx = row_idx * topk + lane_id;
    const auto weight = is_shared ? (routed_sum / routed_scaling_factor) : routed_weight;
    const auto expert_id = is_shared ? (num_experts + lane_id - topk_routed) : selected_expert;
    const auto scale = apply_routed_scaling_factor_on_output ? routed_scaling_factor : 1.0f;
    const auto norm = renormalize && routed_sum > 0.0f ? routed_sum : 1.0f;
    output[output_idx] = weight / norm * scale;
    indices[output_idx] = expert_id;
  }
}

template <ScoringFunc kScoringFunc>
void dispatch_small_token_kernel(
    uint32_t num_rows,
    uint32_t threads_per_block,
    uint32_t warps_per_token,
    DLDevice device,
    size_t smem_per_row,
    const MoEFusedGateParams& params) {
  using namespace host;
  if (warps_per_token <= 8) {
    LaunchKernel(num_rows, threads_per_block, device, smem_per_row)(
        moe_fused_gate_kernel_small_token<8, kScoringFunc>, params);
  } else if (warps_per_token <= 12) {
    LaunchKernel(num_rows, threads_per_block, device, smem_per_row)(
        moe_fused_gate_kernel_small_token<12, kScoringFunc>, params);
  } else {
    LaunchKernel(num_rows, threads_per_block, device, smem_per_row)(
        moe_fused_gate_kernel_small_token<16, kScoringFunc>, params);
  }
}

struct MoEFusedGateKernel {
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView indices,
      uint32_t topk,
      uint32_t scoring_func,  // 0 = sigmoid, 1 = sqrtsoftplus
      uint32_t num_fused_shared_experts,
      bool renormalize,
      float routed_scaling_factor,
      bool apply_routed_scaling_factor_on_output) {
    using namespace host;

    auto N = SymbolicSize{"num_rows"};
    auto E = SymbolicSize{"num_experts"};
    auto K = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    K.set_value(topk);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, E}).with_dtype<float>().with_device(device).verify(input);
    TensorMatcher({E}).with_dtype<float>().with_device(device).verify(bias);
    TensorMatcher({N, K}).with_dtype<float>().with_device(device).verify(output);
    TensorMatcher({N, K}).with_dtype<int32_t>().with_device(device).verify(indices);

    const auto num_rows = static_cast<uint32_t>(N.unwrap());
    const auto num_experts = static_cast<uint32_t>(E.unwrap());

    RuntimeCheck(num_experts <= kMaxExperts, "num_experts exceeds maximum supported value");
    RuntimeCheck(scoring_func <= 1, "scoring_func must be 0 (sigmoid) or 1 (sqrtsoftplus)");
    RuntimeCheck(topk > num_fused_shared_experts, "topk must be greater than num_fused_shared_experts");

    const auto params = MoEFusedGateParams{
        .input = static_cast<const float*>(input.data_ptr()),
        .bias = static_cast<const float*>(bias.data_ptr()),
        .output = static_cast<float*>(output.data_ptr()),
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .num_rows = num_rows,
        .num_experts = num_experts,
        .topk = topk,
        .num_fused_shared_experts = num_fused_shared_experts,
        .renormalize = renormalize,
        .routed_scaling_factor = routed_scaling_factor,
        .apply_routed_scaling_factor_on_output = apply_routed_scaling_factor_on_output,
    };

    const size_t smem_per_row = 2 * num_experts * sizeof(float);

    bool use_small_token_kernel = num_rows <= kSmallTokenThreshold;

    if (use_small_token_kernel) {
      // 1 token per block
      uint32_t warps_per_token = div_ceil(num_experts, kWarpSize);
      warps_per_token = std::min(warps_per_token, 16u);
      uint32_t threads_per_block = warps_per_token * kWarpSize;

      if (scoring_func == 0) {
        dispatch_small_token_kernel<ScoringFunc::kSigmoid>(
            num_rows, threads_per_block, warps_per_token, device.unwrap(), smem_per_row, params);
      } else {
        dispatch_small_token_kernel<ScoringFunc::kSqrtSoftplus>(
            num_rows, threads_per_block, warps_per_token, device.unwrap(), smem_per_row, params);
      }
    } else {
      // multiple tokens per block
      uint32_t num_blocks = div_ceil(num_rows, kWarpsPerCTA);
      dim3 block_dim(kWarpSize, kWarpsPerCTA);
      size_t large_smem = smem_per_row * kWarpsPerCTA;

      if (scoring_func == 0) {
        LaunchKernel(num_blocks, block_dim, device.unwrap(), large_smem)(
            moe_fused_gate_kernel<ScoringFunc::kSigmoid>, params);
      } else {
        LaunchKernel(num_blocks, block_dim, device.unwrap(), large_smem)(
            moe_fused_gate_kernel<ScoringFunc::kSqrtSoftplus>, params);
      }
    }
  }
};

}  // namespace
