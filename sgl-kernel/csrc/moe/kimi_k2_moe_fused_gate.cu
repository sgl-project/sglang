#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cfloat>

// Kimi K2 MoE fused gate, supports NUM_EXPERTS in {256 (MiMo V2 Flash), 384 (Kimi K2)}.
// Routing (DeepSeek "noaux_tc" with num_expert_group = 1):
//   1. sigmoid(gate_logit)
//   2. add per-expert correction bias (ranking only)
//   3. pick top-k by biased score
//   4. weights = sigmoid (no bias)
//   5. optional renorm; routed_scaling_factor folded into renorm (no-op when not renormalizing)

__device__ __forceinline__ float sigmoid_accurate(float x) {
  return 1.0f / (1.0f + expf(-x));
}

template <int N>
struct GateConfig {
  static_assert(
      N == 256 || N == 384,
      "kimi_k2_moe_fused_gate currently only supports "
      "NUM_EXPERTS == 256 or 384");
  static constexpr int NUM_EXPERTS = N;
  static constexpr int WARP_SIZE = 32;
  static constexpr int WARPS_PER_CTA = 6;  // only used by the large-token kernel
  static constexpr int VPT = N / 32;       // 8 (256) or 12 (384)
  static constexpr int VEC_SIZE = 4;
  static constexpr int VEC_PER_LANE = VPT / VEC_SIZE;   // 2 or 3
  static constexpr int WARPS_PER_TOKEN_SMALL = N / 32;  // 8 or 12
  static constexpr int THREADS_PER_BLOCK_SMALL = N;     // 256 or 384
  static constexpr int SMALL_TOKEN_THRESHOLD = 512;
  static constexpr int MAX_TOPK = 8;  // must match TORCH_CHECK(topk <= 8) at the host launcher
  static_assert(VPT % VEC_SIZE == 0, "VPT must be a multiple of VEC_SIZE for the float4 vec load");
};

// Small-token kernel: 1 block per token, NUM_EXPERTS threads (1 thread = 1 expert).
template <int N>
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
  using Cfg = GateConfig<N>;
  constexpr int NUM_EXPERTS = Cfg::NUM_EXPERTS;
  constexpr int WARP_SIZE = Cfg::WARP_SIZE;
  constexpr int WARPS_PER_TOKEN_SMALL = Cfg::WARPS_PER_TOKEN_SMALL;
  constexpr int MAX_TOPK = Cfg::MAX_TOPK;

  int64_t row_idx = blockIdx.x;
  if (row_idx >= num_rows) return;

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  // Sigmoid weights (no bias) for final lookup, indexed by expert id.
  __shared__ float shared_original_scores[NUM_EXPERTS];
  __shared__ float warp_maxs[WARPS_PER_TOKEN_SMALL];
  __shared__ int warp_experts[WARPS_PER_TOKEN_SMALL];
  __shared__ int selected_experts[MAX_TOPK];

  // Keep biased_val in register; mask the winner in-place each iteration to
  // avoid round-tripping through shared memory.
  float input_val = input[row_idx * NUM_EXPERTS + tid];
  float bias_val = bias[tid];
  float sigmoid_val = sigmoid_accurate(input_val);
  float biased_val = sigmoid_val + bias_val;
  shared_original_scores[tid] = sigmoid_val;

  __syncthreads();

  // Lane 0 of warp 0 accumulates the renorm sum as it picks each winner,
  // saving a second pass over selected_experts during writeback.
  float sum_for_renorm = 0.0f;

  for (int k = 0; k < topk; k++) {
    // Stage 1: per-warp argmax.
    float warp_max_val = biased_val;
    int warp_max_expert = tid;
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

    // Stage 2: warp 0 merges warp-leaders into a single winner.
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
        if (renormalize && final_expert >= 0 && final_expert < NUM_EXPERTS) {
          sum_for_renorm += shared_original_scores[final_expert];
        }
      }
    }
    __syncthreads();

    int selected = selected_experts[k];
    if (tid == selected) biased_val = -FLT_MAX;
  }

  // Lane 0 of warp 0 writes the output. sum_for_renorm was accumulated
  // during the topk loop, so we just fold it into rcp.
  if (warp_id == 0 && lane_id == 0) {
    float rcp = 1.0f;
    if (renormalize && sum_for_renorm > 0.0f) {
      rcp = 1.0f / sum_for_renorm;
      if (apply_routed_scaling_factor_on_output) {
        rcp *= static_cast<float>(routed_scaling_factor);
      }
    }

    for (int k = 0; k < topk; k++) {
      int expert_id = selected_experts[k];
      bool valid = (expert_id >= 0 && expert_id < NUM_EXPERTS);
      output_ptr[row_idx * topk + k] = valid ? shared_original_scores[expert_id] * rcp : 0.0f;
      indices_ptr[row_idx * topk + k] = valid ? expert_id : 0;
    }
  }
}

// Large-token kernel: 1 warp per token, WARPS_PER_CTA warps per block.
template <int N>
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
  using Cfg = GateConfig<N>;
  constexpr int NUM_EXPERTS = Cfg::NUM_EXPERTS;
  constexpr int WARP_SIZE = Cfg::WARP_SIZE;
  constexpr int WARPS_PER_CTA = Cfg::WARPS_PER_CTA;
  constexpr int VEC_SIZE = Cfg::VEC_SIZE;
  constexpr int VEC_PER_LANE = Cfg::VEC_PER_LANE;
  constexpr int MAX_TOPK = Cfg::MAX_TOPK;

  int64_t row_idx = blockIdx.x * WARPS_PER_CTA + threadIdx.y;
  if (row_idx >= num_rows) return;

  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;

  __shared__ float shared_scores[NUM_EXPERTS * WARPS_PER_CTA];
  __shared__ float shared_original_scores[NUM_EXPERTS * WARPS_PER_CTA];
  float* warp_scores = shared_scores + warp_id * NUM_EXPERTS;
  float* warp_original_scores = shared_original_scores + warp_id * NUM_EXPERTS;
  float4* warp_scores_v4 = reinterpret_cast<float4*>(warp_scores);
  float4* warp_original_scores_v4 = reinterpret_cast<float4*>(warp_original_scores);

  float4* input_vec = reinterpret_cast<float4*>(input + row_idx * NUM_EXPERTS);
  float4* bias_vec = reinterpret_cast<float4*>(bias);

  // Lane-strided vec_idx (each lane k stores at vec_idx k, k+32, k+64, ...) so each
  // iteration's STS.128 is lane-contiguous, avoiding shared-mem bank conflicts.
#pragma unroll
  for (int i = 0; i < VEC_PER_LANE; i++) {
    int vec_idx = lane_id + i * WARP_SIZE;
    float4 input_val = input_vec[vec_idx];
    float4 bias_val = bias_vec[vec_idx];

    float4 sigmoid_v4;
    float4 biased_v4;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float inp = ((float*)&input_val)[j];
      float b = ((float*)&bias_val)[j];
      float sigmoid_val = sigmoid_accurate(inp);
      ((float*)&sigmoid_v4)[j] = sigmoid_val;
      ((float*)&biased_v4)[j] = sigmoid_val + b;
    }
    warp_original_scores_v4[vec_idx] = sigmoid_v4;
    warp_scores_v4[vec_idx] = biased_v4;
  }

  __syncwarp();

  // Lane 0 records the picked expert ids and accumulates the renorm sum as
  // it goes; the global write is a single pass after the loop.
  int top_indices[MAX_TOPK];
  float sum_for_renorm = 0.0f;

  for (int k = 0; k < topk; k++) {
    float max_val = -FLT_MAX;
    int max_expert = -1;

    for (int expert = lane_id; expert < NUM_EXPERTS; expert += WARP_SIZE) {
      if (warp_scores[expert] > max_val) {
        max_val = warp_scores[expert];
        max_expert = expert;
      }
    }

    // warp shfl reduce; tie-break by lower expert id
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
      int other_expert = __shfl_down_sync(0xFFFFFFFF, max_expert, offset);
      if (other_val > max_val || (other_val == max_val && other_expert < max_expert)) {
        max_val = other_val;
        max_expert = other_expert;
      }
    }

    if (lane_id == 0) {
      bool valid = (max_expert >= 0 && max_expert < NUM_EXPERTS);
      top_indices[k] = valid ? max_expert : -1;
      if (renormalize && valid) {
        sum_for_renorm += warp_original_scores[max_expert];
      }
      if (valid) warp_scores[max_expert] = -FLT_MAX;
    }
    __syncwarp();
  }

  if (lane_id == 0) {
    float rcp = 1.0f;
    if (renormalize && sum_for_renorm > 0.0f) {
      rcp = 1.0f / sum_for_renorm;
      if (apply_routed_scaling_factor_on_output) {
        rcp *= static_cast<float>(routed_scaling_factor);
      }
    }

    for (int k = 0; k < topk; k++) {
      int e = top_indices[k];
      bool valid = (e >= 0);
      output_ptr[row_idx * topk + k] = valid ? warp_original_scores[e] * rcp : 0.0f;
      indices_ptr[row_idx * topk + k] = valid ? e : 0;
    }
  }
}

template <int N>
static void launch_for_n(
    at::Tensor& input,
    at::Tensor& bias,
    at::Tensor& output,
    at::Tensor& indices,
    int64_t topk,
    bool renormalize,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output,
    cudaStream_t stream) {
  using Cfg = GateConfig<N>;
  int64_t num_rows = input.size(0);
  bool use_small_token_kernel = num_rows <= Cfg::SMALL_TOKEN_THRESHOLD;

  if (use_small_token_kernel) {
    dim3 grid(num_rows);
    dim3 block(Cfg::THREADS_PER_BLOCK_SMALL);
    kimi_k2_moe_fused_gate_kernel_small_token<N><<<grid, block, 0, stream>>>(
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
    int64_t num_blocks = (num_rows + Cfg::WARPS_PER_CTA - 1) / Cfg::WARPS_PER_CTA;
    dim3 grid(num_blocks);
    dim3 block(Cfg::WARP_SIZE, Cfg::WARPS_PER_CTA);
    kimi_k2_moe_fused_gate_kernel<N><<<grid, block, 0, stream>>>(
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

  TORCH_CHECK(input.dtype() == bias.dtype(), "input and bias should have the same dtype");
  TORCH_CHECK(input.scalar_type() == at::kFloat, "kimi_k2_moe_fused_gate only supports float32 input");
  TORCH_CHECK(bias.scalar_type() == at::kFloat, "kimi_k2_moe_fused_gate only supports float32 bias");
  TORCH_CHECK(topk <= 8, "kimi_k2_moe_fused_gate only supports topk <= 8 (got ", topk, ")");

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto output = torch::empty({num_rows, topk}, options);
  auto indices = torch::empty({num_rows, topk}, options.dtype(torch::kInt32));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (num_experts) {
    case 256:
      launch_for_n<256>(
          input,
          bias,
          output,
          indices,
          topk,
          renormalize,
          routed_scaling_factor,
          apply_routed_scaling_factor_on_output,
          stream);
      break;
    case 384:
      launch_for_n<384>(
          input,
          bias,
          output,
          indices,
          topk,
          renormalize,
          routed_scaling_factor,
          apply_routed_scaling_factor_on_output,
          stream);
      break;
    default:
      TORCH_CHECK(
          false,
          "kimi_k2_moe_fused_gate only supports num_experts in "
          "{256, 384}, got ",
          num_experts);
  }

  return {output, indices};
}
