/*
 * Fused biased grouped top-k kernel for Bailing MoE V3 routing.
 *
 * Specialized for 512 routed experts split into 8 groups of 64 experts, with
 * topk_group=4 and routed top-k <= 8.
 */
#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>

namespace {

namespace host = sglang::host;

static constexpr int WARP_SIZE = 32;
static constexpr int NUM_EXPERTS = 512;
static constexpr int NUM_GROUPS = 8;
static constexpr int EXPERTS_PER_GROUP = 64;
static constexpr int TOPK_GROUP = 4;
static constexpr int MAX_ROUTED_TOPK = 8;
static constexpr int MAX_FUSED_SHARED_EXPERTS = 1;
static constexpr int CANDIDATES_PER_GROUP = MAX_ROUTED_TOPK;
static constexpr int NUM_CANDIDATES = NUM_GROUPS * CANDIDATES_PER_GROUP;
static constexpr uint32_t ORDERED_NEG_FLT_MAX = 0x00800000u;

__device__ __forceinline__ float fast_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float2 load_float2(const float* ptr) {
  return *reinterpret_cast<const float2*>(ptr);
}

__device__ __forceinline__ uint32_t float_to_ordered_uint(float val) {
  uint32_t bits = __float_as_uint(val);
  return bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

__device__ __forceinline__ float ordered_uint_to_float(uint32_t ordered) {
  uint32_t bits = ordered ^ ((ordered & 0x80000000u) ? 0x80000000u : 0xffffffffu);
  return __uint_as_float(bits);
}

// Pack (value, index) into uint64_t for max reduction. For equal values, the
// smaller index wins, matching the deterministic tie policy used by the JIT
// single-group top-k kernel.
__device__ __forceinline__ uint64_t pack_val_idx(float val, int32_t idx) {
  uint32_t val_bits = float_to_ordered_uint(val);
  uint32_t idx_bits = static_cast<uint32_t>(65535 - idx);
  return (static_cast<uint64_t>(val_bits) << 32) | idx_bits;
}

__device__ __forceinline__ void unpack_val_idx(uint64_t packed, float& val, int32_t& idx) {
  uint32_t idx_bits = static_cast<uint32_t>(packed & 0xFFFFFFFF);
  idx = static_cast<int32_t>(65535 - idx_bits);
  uint32_t val_bits = static_cast<uint32_t>(packed >> 32);
  val = ordered_uint_to_float(val_bits);
}

__device__ __forceinline__ uint64_t pack_ordered_idx(uint32_t ordered, int32_t idx) {
  uint32_t idx_bits = static_cast<uint32_t>(65535 - idx);
  return (static_cast<uint64_t>(ordered) << 32) | idx_bits;
}

__device__ __forceinline__ int32_t unpack_idx(uint64_t packed) {
  uint32_t idx_bits = static_cast<uint32_t>(packed & 0xFFFFFFFF);
  return static_cast<int32_t>(65535 - idx_bits);
}

__device__ __forceinline__ uint64_t warp_max_u64(uint64_t val) {
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    uint64_t other = __shfl_xor_sync(0xffffffff, val, mask);
    val = max(val, other);
  }
  return val;
}

__device__ __forceinline__ uint32_t warp_max_ordered(uint32_t ordered) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __reduce_max_sync(0xffffffff, ordered);
#else
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    const uint32_t other = __shfl_xor_sync(0xffffffff, ordered, mask);
    ordered = max(ordered, other);
  }
  return ordered;
#endif
}

__device__ __forceinline__ float warp_max_float(float val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const uint32_t best_ordered = warp_max_ordered(float_to_ordered_uint(val));
  return ordered_uint_to_float(best_ordered);
#else
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    const float other = __shfl_xor_sync(0xffffffff, val, mask);
    val = fmaxf(val, other);
  }
  return val;
#endif
}

__device__ __forceinline__ void
warp_max_ordered_idx(uint32_t ordered, int32_t idx, uint32_t& best_ordered, int32_t& best_idx) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  best_ordered = __reduce_max_sync(0xffffffff, ordered);
  const uint32_t idx_for_best = (ordered == best_ordered) ? static_cast<uint32_t>(idx) : 0xffffffffu;
  const uint32_t best_idx_u = __reduce_min_sync(0xffffffff, idx_for_best);
  best_idx = static_cast<int32_t>(best_idx_u);
#else
  const uint64_t packed = pack_ordered_idx(ordered, idx);
  const uint64_t best = warp_max_u64(packed);
  best_ordered = static_cast<uint32_t>(best >> 32);
  best_idx = unpack_idx(best);
#endif
}

__device__ __forceinline__ void warp_max_float_idx(float val, int32_t idx, float& best_val, int32_t& best_idx) {
  uint32_t best_ordered;
  warp_max_ordered_idx(float_to_ordered_uint(val), idx, best_ordered, best_idx);
  best_val = ordered_uint_to_float(best_ordered);
}

__device__ __forceinline__ bool group_worse(float score, int32_t group, float other_score, int32_t other_group) {
  return score < other_score || (score == other_score && group > other_group);
}

__device__ __forceinline__ bool group_better(float score, int32_t group, float other_score, int32_t other_group) {
  return score > other_score || (score == other_score && group < other_group);
}

__device__ __forceinline__ void swap_group(float& score0, int32_t& group0, float& score1, int32_t& group1) {
  const float tmp_score = score0;
  const int32_t tmp_group = group0;
  score0 = score1;
  group0 = group1;
  score1 = tmp_score;
  group1 = tmp_group;
}

__device__ __forceinline__ void sift_down_group_heap4(
    float& score0,
    int32_t& group0,
    float& score1,
    int32_t& group1,
    float& score2,
    int32_t& group2,
    float& score3,
    int32_t& group3) {
  if (group_worse(score2, group2, score1, group1)) {
    if (group_worse(score2, group2, score0, group0)) {
      swap_group(score0, group0, score2, group2);
    }
  } else if (group_worse(score1, group1, score0, group0)) {
    swap_group(score0, group0, score1, group1);
    if (group_worse(score3, group3, score1, group1)) {
      swap_group(score1, group1, score3, group3);
    }
  }
}

__device__ __forceinline__ void push_group_heap4(
    float score,
    int32_t group,
    float& score0,
    int32_t& group0,
    float& score1,
    int32_t& group1,
    float& score2,
    int32_t& group2,
    float& score3,
    int32_t& group3) {
  if (group_better(score, group, score0, group0)) {
    score0 = score;
    group0 = group;
    sift_down_group_heap4(score0, group0, score1, group1, score2, group2, score3, group3);
  }
}

__global__ void bailing_moe_biased_grouped_topk_kernel(
    const float* __restrict__ gating_output,
    const float* __restrict__ correction_bias,
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    int64_t num_tokens,
    int64_t topk,
    bool renormalize,
    int64_t num_fused_shared_experts,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  __shared__ float group_scores[NUM_GROUPS];
  __shared__ int32_t group_selected[NUM_GROUPS];
  __shared__ uint32_t candidate_choice_ordered[NUM_CANDIDATES];
  __shared__ float candidate_score[NUM_CANDIDATES];
  __shared__ int32_t candidate_id[NUM_CANDIDATES];

  const int64_t token_id = blockIdx.x;
  if (token_id >= num_tokens) {
    return;
  }

  const int lane_id = threadIdx.x;
  const int group_id = threadIdx.y;
  const int group_base = group_id * EXPERTS_PER_GROUP;
  const int expert0 = group_base + lane_id * 2;
  const int expert1 = expert0 + 1;
  const int64_t token_offset = token_id * NUM_EXPERTS;

  const float2 gate_pair = load_float2(gating_output + token_offset + expert0);
  const float2 bias_pair = load_float2(correction_bias + expert0);
  const float score0 = fast_sigmoid(gate_pair.x);
  const float score1 = fast_sigmoid(gate_pair.y);
  const uint32_t choice_ordered0 = float_to_ordered_uint(score0 + bias_pair.x);
  const uint32_t choice_ordered1 = float_to_ordered_uint(score1 + bias_pair.y);

  uint32_t group_choice_ordered0 = choice_ordered0;
  uint32_t group_choice_ordered1 = choice_ordered1;

  const bool group_pick1 = group_choice_ordered1 > group_choice_ordered0;
  const uint32_t group_local_top_ordered0 = group_pick1 ? group_choice_ordered1 : group_choice_ordered0;
  const int32_t group_local_idx0 = expert0 + static_cast<int32_t>(group_pick1);

  uint32_t group_best_ordered0;
  int32_t group_best_idx0;
  warp_max_ordered_idx(group_local_top_ordered0, group_local_idx0, group_best_ordered0, group_best_idx0);

  if (group_best_idx0 == expert0) {
    group_choice_ordered0 = ORDERED_NEG_FLT_MAX;
  }
  if (group_best_idx0 == expert1) {
    group_choice_ordered1 = ORDERED_NEG_FLT_MAX;
  }

  const uint32_t group_local_top_ordered1 = max(group_choice_ordered0, group_choice_ordered1);
  const uint32_t group_best_ordered1 = warp_max_ordered(group_local_top_ordered1);
  const float group_best0 = ordered_uint_to_float(group_best_ordered0);
  const float group_best1 = ordered_uint_to_float(group_best_ordered1);
  const float group_score = group_best0 + group_best1;

  if (lane_id == 0) {
    group_scores[group_id] = group_score;
  }
  __syncthreads();

  if (group_id == 0 && lane_id == 0) {
#pragma unroll
    for (int i = 0; i < NUM_GROUPS; ++i) {
      group_selected[i] = 0;
    }

    float heap_score0 = group_scores[0];
    float heap_score1 = group_scores[1];
    float heap_score2 = group_scores[2];
    float heap_score3 = group_scores[3];
    int32_t heap_group0 = 0;
    int32_t heap_group1 = 1;
    int32_t heap_group2 = 2;
    int32_t heap_group3 = 3;

    if (group_worse(heap_score3, heap_group3, heap_score1, heap_group1)) {
      swap_group(heap_score1, heap_group1, heap_score3, heap_group3);
    }
    sift_down_group_heap4(
        heap_score0, heap_group0, heap_score1, heap_group1, heap_score2, heap_group2, heap_score3, heap_group3);

    push_group_heap4(
        group_scores[4],
        4,
        heap_score0,
        heap_group0,
        heap_score1,
        heap_group1,
        heap_score2,
        heap_group2,
        heap_score3,
        heap_group3);
    push_group_heap4(
        group_scores[5],
        5,
        heap_score0,
        heap_group0,
        heap_score1,
        heap_group1,
        heap_score2,
        heap_group2,
        heap_score3,
        heap_group3);
    push_group_heap4(
        group_scores[6],
        6,
        heap_score0,
        heap_group0,
        heap_score1,
        heap_group1,
        heap_score2,
        heap_group2,
        heap_score3,
        heap_group3);
    push_group_heap4(
        group_scores[7],
        7,
        heap_score0,
        heap_group0,
        heap_score1,
        heap_group1,
        heap_score2,
        heap_group2,
        heap_score3,
        heap_group3);

    group_selected[heap_group0] = 1;
    group_selected[heap_group1] = 1;
    group_selected[heap_group2] = 1;
    group_selected[heap_group3] = 1;
  }
  __syncthreads();

  const int candidate_base = group_id * CANDIDATES_PER_GROUP;
  if (group_selected[group_id] != 0) {
    uint32_t local_choice_ordered0 = choice_ordered0;
    uint32_t local_choice_ordered1 = choice_ordered1;

#pragma unroll
    for (int k = 0; k < MAX_ROUTED_TOPK; ++k) {
      const bool pick1 = local_choice_ordered1 > local_choice_ordered0;
      const uint32_t local_choice_ordered = pick1 ? local_choice_ordered1 : local_choice_ordered0;
      const int32_t local_idx = expert0 + static_cast<int32_t>(pick1);

      uint32_t best_choice_ordered;
      int32_t best_idx;
      warp_max_ordered_idx(local_choice_ordered, local_idx, best_choice_ordered, best_idx);

      const int owner_lane = (best_idx - group_base) / 2;
      float owner_score = 0.0f;
      if (best_idx == expert0) {
        owner_score = score0;
      } else if (best_idx == expert1) {
        owner_score = score1;
      }
      const float best_score = __shfl_sync(0xffffffff, owner_score, owner_lane);

      if (lane_id == 0) {
        candidate_choice_ordered[candidate_base + k] = best_choice_ordered;
        candidate_score[candidate_base + k] = best_score;
        candidate_id[candidate_base + k] = best_idx;
      }

      if (best_idx == expert0) {
        local_choice_ordered0 = ORDERED_NEG_FLT_MAX;
      }
      if (best_idx == expert1) {
        local_choice_ordered1 = ORDERED_NEG_FLT_MAX;
      }
    }
  }
  __syncthreads();

  if (group_id != 0) {
    return;
  }

  const int slot0 = lane_id * 2;
  const int slot1 = slot0 + 1;
  const int slot_group = slot0 / CANDIDATES_PER_GROUP;
  uint32_t merge_choice_ordered0 = ORDERED_NEG_FLT_MAX;
  uint32_t merge_choice_ordered1 = ORDERED_NEG_FLT_MAX;
  float merge_score0 = 0.0f;
  float merge_score1 = 0.0f;
  int32_t merge_id0 = 0;
  int32_t merge_id1 = 0;
  if (group_selected[slot_group] != 0) {
    merge_choice_ordered0 = candidate_choice_ordered[slot0];
    merge_choice_ordered1 = candidate_choice_ordered[slot1];
    merge_score0 = candidate_score[slot0];
    merge_score1 = candidate_score[slot1];
    merge_id0 = candidate_id[slot0];
    merge_id1 = candidate_id[slot1];
  }

  const int routed_topk = static_cast<int>(topk - num_fused_shared_experts);
  float selected_scores[MAX_ROUTED_TOPK];
  int32_t selected_ids[MAX_ROUTED_TOPK];

#pragma unroll
  for (int k = 0; k < MAX_ROUTED_TOPK; ++k) {
    if (k < routed_topk) {
      const bool pick1 = merge_choice_ordered1 > merge_choice_ordered0;
      const uint32_t local_choice_ordered = pick1 ? merge_choice_ordered1 : merge_choice_ordered0;
      const int32_t local_slot = slot0 + static_cast<int32_t>(pick1);

      uint32_t best_choice_ordered;
      int32_t best_slot;
      warp_max_ordered_idx(local_choice_ordered, local_slot, best_choice_ordered, best_slot);

      const int owner_lane = best_slot / 2;
      float owner_score = 0.0f;
      int32_t owner_id = 0;
      if (best_slot == slot0) {
        owner_score = merge_score0;
        owner_id = merge_id0;
      } else if (best_slot == slot1) {
        owner_score = merge_score1;
        owner_id = merge_id1;
      }
      selected_scores[k] = __shfl_sync(0xffffffff, owner_score, owner_lane);
      selected_ids[k] = __shfl_sync(0xffffffff, owner_id, owner_lane);

      if (best_slot == slot0) {
        merge_choice_ordered0 = ORDERED_NEG_FLT_MAX;
      }
      if (best_slot == slot1) {
        merge_choice_ordered1 = ORDERED_NEG_FLT_MAX;
      }
    }
  }

  if (lane_id != 0) {
    return;
  }

  float routed_sum = 0.0f;
#pragma unroll
  for (int k = 0; k < MAX_ROUTED_TOPK; ++k) {
    if (k < routed_topk) {
      routed_sum += selected_scores[k];
    }
  }

  const float renorm_denominator = routed_sum > 0.0f ? routed_sum : 1.0f;
  float* out_weights = topk_weights + token_id * topk;
  int32_t* out_ids = topk_ids + token_id * topk;

#pragma unroll
  for (int k = 0; k < MAX_ROUTED_TOPK; ++k) {
    if (k < routed_topk) {
      float weight = selected_scores[k];
      if (renormalize) {
        weight = weight / renorm_denominator;
      }
      if (apply_routed_scaling_factor_on_output) {
        weight *= routed_scaling_factor;
      }
      out_weights[k] = weight;
      out_ids[k] = selected_ids[k];
    }
  }

  if (num_fused_shared_experts > 0) {
    float shared_weight = routed_sum / routed_scaling_factor;
    if (renormalize) {
      shared_weight = shared_weight / renorm_denominator;
    }
    if (apply_routed_scaling_factor_on_output) {
      shared_weight *= routed_scaling_factor;
    }
    out_weights[routed_topk] = shared_weight;
    out_ids[routed_topk] = NUM_EXPERTS;
  }
}

struct BailingMoeBiasedGroupedTopkLaunchArgs {
  float* topk_weights;
  int32_t* topk_ids;
  int64_t num_tokens;
  int64_t topk;
  bool renormalize;
  int64_t num_fused_shared_experts;
  float routed_scaling_factor;
  bool apply_routed_scaling_factor_on_output;
  DLDevice device;
};

void launch_bailing_moe_biased_grouped_topk(
    const float* gating_output, const float* correction_bias, const BailingMoeBiasedGroupedTopkLaunchArgs& args) {
  dim3 block_dim(WARP_SIZE, NUM_GROUPS);
  host::LaunchKernel(static_cast<uint32_t>(args.num_tokens), block_dim, args.device)(
      bailing_moe_biased_grouped_topk_kernel,
      gating_output,
      correction_bias,
      args.topk_weights,
      args.topk_ids,
      args.num_tokens,
      args.topk,
      args.renormalize,
      args.num_fused_shared_experts,
      args.routed_scaling_factor,
      args.apply_routed_scaling_factor_on_output);
}

void bailing_moe_biased_grouped_topk(
    tvm::ffi::TensorView gating_output,
    tvm::ffi::TensorView correction_bias,
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView topk_ids,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    bool renormalize,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  using namespace host;

  SymbolicSize N{"num_tokens"};
  SymbolicSize E{"num_experts"};
  SymbolicSize K{"topk"};
  SymbolicDevice device;
  K.set_value(topk);
  device.set_options<kDLCUDA>();

  TensorMatcher({N, E}).with_dtype<float>().with_device(device).verify(gating_output);
  TensorMatcher({E}).with_dtype<float>().with_device(device).verify(correction_bias);
  TensorMatcher({N, K}).with_dtype<float>().with_device(device).verify(topk_weights);
  TensorMatcher({N, K}).with_dtype<int32_t>().with_device(device).verify(topk_ids);

  const auto num_tokens = N.unwrap();
  const auto num_experts = E.unwrap();
  RuntimeCheck(
      num_experts == NUM_EXPERTS, "bailing_moe_biased_grouped_topk only supports 512 experts, got ", num_experts);
  RuntimeCheck(num_expert_group == NUM_GROUPS, "bailing_moe_biased_grouped_topk only supports num_expert_group=8");
  RuntimeCheck(topk_group == TOPK_GROUP, "bailing_moe_biased_grouped_topk only supports topk_group=4");
  RuntimeCheck(
      num_fused_shared_experts >= 0 && num_fused_shared_experts <= MAX_FUSED_SHARED_EXPERTS,
      "bailing_moe_biased_grouped_topk only supports num_fused_shared_experts in {0, 1}");
  RuntimeCheck(topk - num_fused_shared_experts <= MAX_ROUTED_TOPK, "routed topk must be <= 8");
  RuntimeCheck(topk > num_fused_shared_experts, "topk must be greater than num_fused_shared_experts");

  if (num_tokens == 0) {
    return;
  }

  const BailingMoeBiasedGroupedTopkLaunchArgs args{
      .topk_weights = static_cast<float*>(topk_weights.data_ptr()),
      .topk_ids = static_cast<int32_t*>(topk_ids.data_ptr()),
      .num_tokens = num_tokens,
      .topk = topk,
      .renormalize = renormalize,
      .num_fused_shared_experts = num_fused_shared_experts,
      .routed_scaling_factor = static_cast<float>(routed_scaling_factor),
      .apply_routed_scaling_factor_on_output = apply_routed_scaling_factor_on_output,
      .device = device.unwrap()};

  launch_bailing_moe_biased_grouped_topk(
      static_cast<const float*>(gating_output.data_ptr()), static_cast<const float*>(correction_bias.data_ptr()), args);
}

}  // namespace
