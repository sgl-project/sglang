#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, Panic, div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>

namespace {

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

// Scalar widening: input/bias may arrive as fp32, bf16, or fp16; the kernel math
// always runs in fp32. Widening bf16/fp16 -> fp32 is exact, so results are
// bitwise identical to upcasting on the host first (the casts we are removing).
__device__ __forceinline__ float to_float(float x) {
  return x;
}
__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
__device__ __forceinline__ float to_float(__half x) {
  return __half2float(x);
}

// Vectorized load of 4 consecutive elements of type T at vector index `vec_idx`,
// widened to a float4. fp32 reads a 16B float4; bf16/fp16 read an 8B float2 and
// expand. Used only by the large-token kernel's lane-strided loads.
template <typename T>
struct VecLoader;

template <>
struct VecLoader<float> {
  __device__ __forceinline__ static float4 load(const float* base, int vec_idx) {
    return reinterpret_cast<const float4*>(base)[vec_idx];
  }
};

template <>
struct VecLoader<__nv_bfloat16> {
  __device__ __forceinline__ static float4 load(const __nv_bfloat16* base, int vec_idx) {
    float2 raw = reinterpret_cast<const float2*>(base)[vec_idx];  // 4 bf16 = 8 bytes
    const __nv_bfloat162* packed = reinterpret_cast<const __nv_bfloat162*>(&raw);
    float2 lo = __bfloat1622float2(packed[0]);
    float2 hi = __bfloat1622float2(packed[1]);
    return make_float4(lo.x, lo.y, hi.x, hi.y);
  }
};

template <>
struct VecLoader<__half> {
  __device__ __forceinline__ static float4 load(const __half* base, int vec_idx) {
    float2 raw = reinterpret_cast<const float2*>(base)[vec_idx];  // 4 fp16 = 8 bytes
    const __half2* packed = reinterpret_cast<const __half2*>(&raw);
    float2 lo = __half22float2(packed[0]);
    float2 hi = __half22float2(packed[1]);
    return make_float4(lo.x, lo.y, hi.x, hi.y);
  }
};

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
  static constexpr int MAX_TOPK = 8;  // must match RuntimeCheck(topk <= 8) at the host launcher
  static_assert(VPT % VEC_SIZE == 0, "VPT must be a multiple of VEC_SIZE for the float4 vec load");
};

// Small-token kernel: 1 block per token, NUM_EXPERTS threads (1 thread = 1 expert).
template <int N, typename InputT, typename BiasT>
__global__ void kimi_k2_moe_fused_gate_kernel_small_token(
    const InputT* input,
    const BiasT* bias,
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
  float input_val = to_float(input[row_idx * NUM_EXPERTS + tid]);
  float bias_val = to_float(bias[tid]);
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
template <int N, typename InputT, typename BiasT>
__global__ void kimi_k2_moe_fused_gate_kernel(
    const InputT* input,
    const BiasT* bias,
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

  const InputT* input_row = input + row_idx * NUM_EXPERTS;

  // Lane-strided vec_idx (each lane k stores at vec_idx k, k+32, k+64, ...) so each
  // iteration's STS.128 is lane-contiguous, avoiding shared-mem bank conflicts.
#pragma unroll
  for (int i = 0; i < VEC_PER_LANE; i++) {
    int vec_idx = lane_id + i * WARP_SIZE;
    float4 input_val = VecLoader<InputT>::load(input_row, vec_idx);
    float4 bias_val = VecLoader<BiasT>::load(bias, vec_idx);

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

// Bundles the dtype-agnostic launch parameters so the templated dispatch below
// only has to thread the typed input/bias pointers.
struct GateLaunchArgs {
  float* output;
  int32_t* indices;
  int64_t num_rows;
  int64_t topk;
  bool renormalize;
  double routed_scaling_factor;
  bool apply_routed_scaling_factor_on_output;
  DLDevice device;
};

template <int N, typename InputT, typename BiasT>
void launch_for_n(const InputT* input, const BiasT* bias, const GateLaunchArgs& args) {
  using namespace host;
  using Cfg = GateConfig<N>;
  bool use_small_token_kernel = args.num_rows <= Cfg::SMALL_TOKEN_THRESHOLD;

  if (use_small_token_kernel) {
    LaunchKernel(
        static_cast<uint32_t>(args.num_rows), static_cast<uint32_t>(Cfg::THREADS_PER_BLOCK_SMALL), args.device)(
        kimi_k2_moe_fused_gate_kernel_small_token<N, InputT, BiasT>,
        input,
        bias,
        args.output,
        args.indices,
        args.num_rows,
        args.topk,
        args.renormalize,
        args.routed_scaling_factor,
        args.apply_routed_scaling_factor_on_output);
  } else {
    uint32_t num_blocks = div_ceil(args.num_rows, static_cast<int64_t>(Cfg::WARPS_PER_CTA));
    dim3 block_dim(Cfg::WARP_SIZE, Cfg::WARPS_PER_CTA);
    LaunchKernel(num_blocks, block_dim, args.device)(
        kimi_k2_moe_fused_gate_kernel<N, InputT, BiasT>,
        input,
        bias,
        args.output,
        args.indices,
        args.num_rows,
        args.topk,
        args.renormalize,
        args.routed_scaling_factor,
        args.apply_routed_scaling_factor_on_output);
  }
}

// input/bias each independently arrive as fp32, bf16, or fp16; widen both to
// fp32 inside the kernel so the host no longer has to upcast. Dispatch is nested:
// num_experts -> input dtype -> bias dtype.
template <int N, typename InputT>
void dispatch_bias(
    const InputT* input, const void* bias, const host::SymbolicDType& bias_dtype, const GateLaunchArgs& args) {
  using namespace host;
  if (bias_dtype.is_type<float>()) {
    launch_for_n<N, InputT, float>(input, static_cast<const float*>(bias), args);
  } else if (bias_dtype.is_type<bf16_t>()) {
    launch_for_n<N, InputT, bf16_t>(input, static_cast<const bf16_t*>(bias), args);
  } else {
    launch_for_n<N, InputT, fp16_t>(input, static_cast<const fp16_t*>(bias), args);
  }
}

template <int N>
void dispatch_input(
    const void* input,
    const host::SymbolicDType& input_dtype,
    const void* bias,
    const host::SymbolicDType& bias_dtype,
    const GateLaunchArgs& args) {
  using namespace host;
  if (input_dtype.is_type<float>()) {
    dispatch_bias<N, float>(static_cast<const float*>(input), bias, bias_dtype, args);
  } else if (input_dtype.is_type<bf16_t>()) {
    dispatch_bias<N, bf16_t>(static_cast<const bf16_t*>(input), bias, bias_dtype, args);
  } else {
    dispatch_bias<N, fp16_t>(static_cast<const fp16_t*>(input), bias, bias_dtype, args);
  }
}

struct KimiK2MoEFusedGateKernel {
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView indices,
      int64_t topk,
      bool renormalize,
      double routed_scaling_factor,
      bool apply_routed_scaling_factor_on_output) {
    using namespace host;

    auto N = SymbolicSize{"num_rows"};
    auto E = SymbolicSize{"num_experts"};
    auto K = SymbolicSize{"topk"};
    auto input_dtype = SymbolicDType{};
    auto bias_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    K.set_value(topk);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, E}).with_dtype<float, bf16_t, fp16_t>(input_dtype).with_device(device).verify(input);
    TensorMatcher({E}).with_dtype<float, bf16_t, fp16_t>(bias_dtype).with_device(device).verify(bias);
    TensorMatcher({N, K}).with_dtype<float>().with_device(device).verify(output);
    TensorMatcher({N, K}).with_dtype<int32_t>().with_device(device).verify(indices);

    const auto num_rows = static_cast<int64_t>(N.unwrap());
    const auto num_experts = static_cast<int64_t>(E.unwrap());

    RuntimeCheck(topk <= 8, "kimi_k2_moe_fused_gate only supports topk <= 8, got ", topk);

    const GateLaunchArgs args{
        .output = static_cast<float*>(output.data_ptr()),
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .num_rows = num_rows,
        .topk = topk,
        .renormalize = renormalize,
        .routed_scaling_factor = routed_scaling_factor,
        .apply_routed_scaling_factor_on_output = apply_routed_scaling_factor_on_output,
        .device = device.unwrap()};

    switch (num_experts) {
      case 256:
        dispatch_input<256>(input.data_ptr(), input_dtype, bias.data_ptr(), bias_dtype, args);
        break;
      case 384:
        dispatch_input<384>(input.data_ptr(), input_dtype, bias.data_ptr(), bias_dtype, args);
        break;
      default:
        Panic("kimi_k2_moe_fused_gate only supports num_experts in {256, 384}, got ", num_experts);
    }
  }
};

}  // namespace
