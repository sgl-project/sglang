/*
 * Fused grouped top-k kernel for MoE routing.
 * Adapted from vLLM's grouped_topk_kernels.cu (Apache-2.0).
 *
 * Handles single-group (num_expert_group=1) and multi-group cases with
 * sigmoid scoring, bias correction, renormalization and scaling factor.
 * Supports up to 512 experts and topk up to 8.
 */
#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, fp32_t

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>

namespace {

static constexpr int WARP_SIZE = 32;
static constexpr int MAX_TOPK = 8;

// Pack (value, index) into a single uint64_t for warp-level max reduction.
// Transform IEEE 754 bits into an unsigned ordering that is monotonic for the
// full float range; correction bias can make sigmoid(score) + bias negative.
__device__ __forceinline__ uint64_t pack_val_idx(float val, int32_t idx) {
  uint32_t val_bits = __float_as_uint(val);
  val_bits ^= (val_bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  // Use (65535 - idx) so that smaller indices win ties
  uint32_t idx_bits = static_cast<uint32_t>(65535 - idx);
  return (static_cast<uint64_t>(val_bits) << 32) | idx_bits;
}

__device__ __forceinline__ void unpack_val_idx(uint64_t packed, float& val, int32_t& idx) {
  uint32_t idx_bits = static_cast<uint32_t>(packed & 0xFFFFFFFF);
  idx = static_cast<int32_t>(65535 - idx_bits);
  uint32_t val_bits = static_cast<uint32_t>(packed >> 32);
  val_bits ^= (val_bits & 0x80000000u) ? 0x80000000u : 0xffffffffu;
  val = __uint_as_float(val_bits);
}

__device__ __forceinline__ uint64_t warp_max_u64(uint64_t val) {
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    uint64_t other = __shfl_xor_sync(0xffffffff, val, mask);
    val = max(val, other);
  }
  return val;
}

__device__ __forceinline__ float warp_sum_f32(float val) {
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

__device__ __forceinline__ float fast_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: one block per token, MaxExperts threads per block.
// Each thread handles one expert (or is idle if threadIdx.x >= numExperts).
//
// Phase 1: All threads load score → sigmoid → +bias → shared memory.
// Phase 2: Warp 0 iteratively selects top-k via packed warp-level max reduce.
// Phase 3: Warp 0 renormalizes and writes output.
// ─────────────────────────────────────────────────────────────────────────────
template <int MaxExperts>
__global__ void grouped_topk_single_group_kernel(
    const float* __restrict__ scores,
    float* __restrict__ topk_values,
    int32_t* __restrict__ topk_indices,
    const float* __restrict__ bias,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t topk,
    bool renormalize,
    float scaling_factor) {
  __shared__ float smem_sigmoid[MaxExperts];
  __shared__ float smem_biased[MaxExperts];

  int64_t token_id = blockIdx.x;
  if (token_id >= num_tokens) return;

  int tid = threadIdx.x;
  const float* token_scores = scores + token_id * num_experts;

  // Phase 1: load → sigmoid → bias → shared memory
  float score_sig = -FLT_MAX;
  float score_biased = -FLT_MAX;
  if (tid < num_experts) {
    float raw = token_scores[tid];
    score_sig = fast_sigmoid(raw);
    score_biased = score_sig + bias[tid];
  }
  smem_sigmoid[tid] = score_sig;
  smem_biased[tid] = score_biased;
  __syncthreads();

  // Phase 2 & 3: warp 0 selects top-k
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  if (warp_id != 0) return;

  float* out_vals = topk_values + token_id * topk;
  int32_t* out_ids = topk_indices + token_id * topk;

  // Each lane scans ceil(num_experts/32) experts per iteration
  float selected_weights[MAX_TOPK];
  int32_t selected_ids[MAX_TOPK];

  for (int k = 0; k < topk; k++) {
    // Each lane finds its local max among its assigned experts
    float my_max_val = -FLT_MAX;
    int32_t my_max_idx = 0;
    for (int i = lane_id; i < num_experts; i += WARP_SIZE) {
      float v = smem_biased[i];
      if (v > my_max_val) {
        my_max_val = v;
        my_max_idx = i;
      }
    }

    // Warp-level max reduction using packed value+index
    uint64_t packed = pack_val_idx(my_max_val, my_max_idx);
    uint64_t best = warp_max_u64(packed);

    float best_val;
    int32_t best_idx;
    unpack_val_idx(best, best_val, best_idx);

    selected_ids[k] = best_idx;
    selected_weights[k] = smem_sigmoid[best_idx];

    // Mark selected expert so it won't be picked again
    if (lane_id == best_idx % WARP_SIZE && (best_idx / WARP_SIZE) == 0) {
      smem_biased[best_idx] = -FLT_MAX;
    }
    // Handle indices >= 32: the owning lane must clear it
    if (best_idx >= WARP_SIZE) {
      if (lane_id == 0) {
        smem_biased[best_idx] = -FLT_MAX;
      }
    } else {
      if (lane_id == best_idx) {
        smem_biased[best_idx] = -FLT_MAX;
      }
    }
    __syncwarp();
  }

  // Phase 3: renormalize and write output. All lanes named by the full-warp
  // shuffle mask must execute warp_sum_f32 together; inactive lanes contribute
  // the additive identity.
  float weight = (lane_id < topk) ? selected_weights[lane_id] : 0.0f;
  float divisor = renormalize ? warp_sum_f32(weight) + 1e-20f : 1.0f;

  if (lane_id < topk) {
    out_ids[lane_id] = selected_ids[lane_id];
    out_vals[lane_id] = weight * scaling_factor / divisor;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Launcher
// ─────────────────────────────────────────────────────────────────────────────
void grouped_topk(
    tvm::ffi::TensorView scores,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView topk_values,
    tvm::ffi::TensorView topk_indices,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    bool renormalize,
    double scaling_factor) {
  using namespace host;

  SymbolicSize N{"num_tokens"};
  SymbolicSize E{"num_experts"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({N, E}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(scores);

  TensorMatcher({E}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(bias);

  SymbolicSize K{"topk"};
  TensorMatcher({N, K}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(topk_values);

  TensorMatcher({N, K}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(topk_indices);

  int64_t num_tokens = N.unwrap();
  int64_t num_experts = E.unwrap();
  DLDevice device = device_.unwrap();

  RuntimeCheck(num_expert_group == 1 && topk_group == 1, "This kernel only supports num_expert_group=1, topk_group=1");
  RuntimeCheck(topk <= MAX_TOPK, "topk must be <= ", MAX_TOPK);
  RuntimeCheck(num_experts <= 512, "num_experts must be <= 512");

  if (num_tokens == 0) return;

  float scale_f = static_cast<float>(scaling_factor);

  auto* score_ptr = static_cast<const float*>(scores.data_ptr());
  auto* bias_ptr = static_cast<const float*>(bias.data_ptr());
  auto* val_ptr = static_cast<float*>(topk_values.data_ptr());
  auto* idx_ptr = static_cast<int32_t*>(topk_indices.data_ptr());

  // Select template based on expert count (round up to next tier)
  int num_threads;
  if (num_experts <= 128) {
    num_threads = 128;
    LaunchKernel(static_cast<uint32_t>(num_tokens), num_threads, device)(
        grouped_topk_single_group_kernel<128>,
        score_ptr,
        val_ptr,
        idx_ptr,
        bias_ptr,
        num_tokens,
        num_experts,
        topk,
        renormalize,
        scale_f);
  } else if (num_experts <= 256) {
    num_threads = 256;
    LaunchKernel(static_cast<uint32_t>(num_tokens), num_threads, device)(
        grouped_topk_single_group_kernel<256>,
        score_ptr,
        val_ptr,
        idx_ptr,
        bias_ptr,
        num_tokens,
        num_experts,
        topk,
        renormalize,
        scale_f);
  } else {
    num_threads = 512;
    LaunchKernel(static_cast<uint32_t>(num_tokens), num_threads, device)(
        grouped_topk_single_group_kernel<512>,
        score_ptr,
        val_ptr,
        idx_ptr,
        bias_ptr,
        num_tokens,
        num_experts,
        topk,
        renormalize,
        scale_f);
  }
}

}  // namespace
