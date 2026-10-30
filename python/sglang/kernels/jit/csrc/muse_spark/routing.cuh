// Fused MuseSpark MoE routing: normalize -> bias -> iterative top-k -> renormalize.
//
// Drop-in replacement for the Triton `fused_muse_spark_routing_kernel_topk8`
// (python/sglang/kernels/ops/moe/muse_spark_routing.py). One warp owns one token
// row; the whole row lives in registers (kExpertsPerLane per lane), each lane
// sorts its slice once, and each of the kTopK rounds pops one head.
//
// Bit-exactness with Triton is a hard requirement (the emitted expert ids must
// match, so every float op must round identically). The float helpers below
// reproduce the exact PTX Triton emits for `tl.sigmoid` / `tl.exp` / `/`.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cstdint>

namespace sglang {

namespace muse_spark {

inline constexpr float kNegInf = -__builtin_huge_valf();
// fp32(log2(e)) == 0x3FB8AA3B, the constant Triton folds into tl.exp.
inline constexpr float kLog2e = 1.4426950408889634f;

SGL_DEVICE float ex2_approx(float x) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// Triton lowers `a / b` on fp32 to div.full.f32, not the div.rn.f32 nvcc emits.
SGL_DEVICE float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// triton/language/standard.py:50 -- `1 / (1 + exp(-x))`, lowered as
// sub.f32 / mul.f32 / ex2.approx.f32 / add.f32 / div.full.f32.
SGL_DEVICE float triton_sigmoid(float x) {
  return div_full(1.0f, ex2_approx((0.0f - x) * kLog2e) + 1.0f);
}

SGL_DEVICE float triton_exp(float x) {
  return ex2_approx(x * kLog2e);
}

// kExpertsPerLane must be a multiple of 4 (128-bit row loads) and covers
// kExpertsPerLane * 32 experts. Lane `l` holds expert v*128 + l*4 + j at
// register slot v*4 + j, monotone in the slot -- both the stable sort and the
// group-ballot tie-break rely on that to reproduce tl.argmax.
template <int kTopK_, int kExpertsPerLane_, int kNormMode_, int kWarpsPerBlock_>
struct RoutingTrait {
  static constexpr int kTopK = kTopK_;
  static constexpr int kExpertsPerLane = kExpertsPerLane_;
  static constexpr int kVecsPerLane = kExpertsPerLane_ / 4;
  static constexpr int kNormMode = kNormMode_;  // 0 = sigmoid, 1 = softmax
  static constexpr int kWarpsPerBlock = kWarpsPerBlock_;
  static constexpr int kMaxExperts = kExpertsPerLane_ * 32;

  static_assert(kExpertsPerLane_ % 4 == 0);
  static_assert(kTopK_ >= 1 && kTopK_ <= 32);
};

struct RoutingParams {
  const float* __restrict__ gating_output;
  const float* __restrict__ bias;
  float* __restrict__ topk_weights;
  int32_t* __restrict__ topk_ids;
  int32_t* __restrict__ packed_topk_ids;             // nullptr => not requested
  const int32_t* __restrict__ local_expert_mapping;  // nullptr => identity
  const int32_t* __restrict__ num_valid_tokens;      // nullptr => all rows real
  int num_tokens;
  int num_experts;
  int64_t stride_gm;
  int64_t stride_wm;
  int64_t stride_im;
  int64_t stride_pm;
  float router_output_multiplier;
  float route_scale;
  float division_eps;
};

/// \brief TRT-LLM packed routing slot: expert id in the high half, the bf16 bits
/// of the weight in the low half. Matches ``_pack_id_weight`` in the Triton op.
SGL_DEVICE int32_t pack_id_weight(int32_t expert_id, float weight) {
  const auto w = DTypeTrait<bf16_t>::from(weight);
  uint16_t bits;
  __builtin_memcpy(&bits, &w, sizeof(bits));
  return (expert_id << 16) | static_cast<int32_t>(bits);
}

template <typename Trait, bool kUsePDL>
__global__ void routing_kernel(const RoutingParams __grid_constant__ params) {
  using namespace device;
  constexpr int kEPL = Trait::kExpertsPerLane;
  constexpr int kTopK = Trait::kTopK;
  constexpr int kGroupExperts = 4 * kWarpThreads;

  const int lane = threadIdx.x % kWarpThreads;
  const int token = blockIdx.x * Trait::kWarpsPerBlock + static_cast<int>(threadIdx.x / kWarpThreads);
  if (token >= params.num_tokens) return;

  // Read before the PDL wait: num_valid_tokens comes from an earlier kernel.
  if (params.num_valid_tokens != nullptr && token >= *params.num_valid_tokens) {
    if (lane < kTopK) {
      params.topk_ids[token * params.stride_im + lane] = -1;
      // The Triton op emits the id sentinel with a zero weight half.
      if (params.packed_topk_ids != nullptr) {
        params.packed_topk_ids[token * params.stride_pm + lane] = -1 << 16;
      }
    }
    return;
  }

  PDLWaitPrimary<kUsePDL>();

  const float* __restrict__ row = params.gating_output + static_cast<int64_t>(token) * params.stride_gm;
  float score[kEPL];
  int expert[kEPL];

#pragma unroll
  for (int v = 0; v < Trait::kVecsPerLane; ++v) {
    const int base = v * kGroupExperts + lane * 4;
    float4 g = make_float4(kNegInf, kNegInf, kNegInf, kNegInf);
    if (base < params.num_experts) g = *reinterpret_cast<const float4*>(row + base);
    score[v * 4 + 0] = g.x;
    score[v * 4 + 1] = g.y;
    score[v * 4 + 2] = g.z;
    score[v * 4 + 3] = g.w;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      expert[v * 4 + j] = base + j;
  }

#pragma unroll
  for (int s = 0; s < kEPL; ++s)
    score[s] *= params.router_output_multiplier;

  if constexpr (Trait::kNormMode == 0) {
#pragma unroll
    for (int s = 0; s < kEPL; ++s)
      score[s] = triton_sigmoid(score[s]);
  } else {
    float lane_max = kNegInf;
#pragma unroll
    for (int s = 0; s < kEPL; ++s)
      if (expert[s] < params.num_experts && score[s] > lane_max) lane_max = score[s];
    const float row_max = warp::reduce_max(lane_max);
    float lane_sum = 0.0f;
#pragma unroll
    for (int s = 0; s < kEPL; ++s) {
      score[s] = (expert[s] < params.num_experts) ? triton_exp(score[s] - row_max) : 0.0f;
      lane_sum += score[s];
    }
    const float row_sum = warp::reduce_sum(lane_sum);
#pragma unroll
    for (int s = 0; s < kEPL; ++s)
      score[s] = div_full(score[s], row_sum);
  }

  float biased[kEPL];
#pragma unroll
  for (int v = 0; v < Trait::kVecsPerLane; ++v) {
    const int base = v * kGroupExperts + lane * 4;
    float4 b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (base < params.num_experts) b = *reinterpret_cast<const float4*>(params.bias + base);
    biased[v * 4 + 0] = b.x;
    biased[v * 4 + 1] = b.y;
    biased[v * 4 + 2] = b.z;
    biased[v * 4 + 3] = b.w;
  }

  PDLTriggerSecondary<kUsePDL>();

  float key[kEPL];
#pragma unroll
  for (int s = 0; s < kEPL; ++s) {
    if (expert[s] < params.num_experts) {
      key[s] = biased[s] + score[s];
    } else {
      score[s] = 0.0f;
      key[s] = kNegInf;
    }
  }

  // Must be a stable sort: equal keys keep their slot order and slots are
  // monotone in expert id, which is tl.argmax's lowest-index-wins tie-break.
#pragma unroll
  for (int r = 0; r < kEPL; ++r) {
#pragma unroll
    for (int i = r % 2; i + 1 < kEPL; i += 2) {
      const bool swap = key[i + 1] > key[i];
      const float k0 = swap ? key[i + 1] : key[i];
      const float k1 = swap ? key[i] : key[i + 1];
      const int e0 = swap ? expert[i + 1] : expert[i];
      const int e1 = swap ? expert[i] : expert[i + 1];
      const float s0 = swap ? score[i + 1] : score[i];
      const float s1 = swap ? score[i] : score[i + 1];
      key[i] = k0;
      key[i + 1] = k1;
      expert[i] = e0;
      expert[i + 1] = e1;
      score[i] = s0;
      score[i + 1] = s1;
    }
  }

  float score_sum = 0.0f;
  int out_id = 0;
  float out_score = 0.0f;

#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    const float round_max = warp::reduce_max(key[0]);
    const bool is_head = key[0] == round_max;

    // Lowest expert id among the tied heads: within one kGroupExperts-wide
    // group the id is monotone in the lane, so the lowest set bit of the lowest
    // non-empty group's ballot owns the round.
    unsigned owners = 0;
#pragma unroll
    for (int g = Trait::kVecsPerLane - 1; g >= 0; --g) {
      const unsigned m = __ballot_sync(warp::kFullMask, is_head && expert[0] / kGroupExperts == g);
      owners = (m != 0u) ? m : owners;
    }
    const int owner = __ffs(owners) - 1;
    const int winner = __shfl_sync(warp::kFullMask, expert[0], owner);
    const float selected = __shfl_sync(warp::kFullMask, score[0], owner);

    // Tapered pop: after round k a lane can be popped at most kTopK-1-k more
    // times, so slots past that can never surface and need not be maintained.
    const bool pop = expert[0] == winner;
#pragma unroll
    for (int s = 0; s + 1 < kEPL; ++s)
      if (s + k <= kTopK - 2) {
        key[s] = pop ? key[s + 1] : key[s];
        expert[s] = pop ? expert[s + 1] : expert[s];
        score[s] = pop ? score[s + 1] : score[s];
      }
    // Only reachable when kTopK > kEPL: a lane popped kEPL times is exhausted
    // and must stop winning rounds.
    if (kEPL - 1 + k <= kTopK - 2) key[kEPL - 1] = pop ? kNegInf : key[kEPL - 1];

    score_sum += selected;
    if (lane == k) {
      out_id = winner;
      out_score = selected;
    }
  }

  score_sum += params.division_eps;
  const float inv_sum_scaled = div_full(params.route_scale, score_sum);
  if (lane < kTopK) {
    const int32_t id =
        (params.local_expert_mapping != nullptr) ? params.local_expert_mapping[out_id] : static_cast<int32_t>(out_id);
    const float weight = out_score * inv_sum_scaled;
    params.topk_ids[token * params.stride_im + lane] = id;
    params.topk_weights[token * params.stride_wm + lane] = weight;
    if (params.packed_topk_ids != nullptr) {
      params.packed_topk_ids[token * params.stride_pm + lane] = pack_id_weight(id, weight);
    }
  }
}

}  // namespace muse_spark

template <int kTopK, int kExpertsPerLane, int kNormMode, int kWarpsPerBlock, bool kUsePDL>
void fused_muse_spark_routing(
    tvm::ffi::TensorView gating_output,                        // [T, E] fp32
    tvm::ffi::TensorView bias,                                 // [E] fp32
    tvm::ffi::TensorView topk_weights,                         // [T, K] fp32 (out)
    tvm::ffi::TensorView topk_ids,                             // [T, K] int32 (out)
    tvm::ffi::Optional<tvm::ffi::TensorView> packed_topk_ids,  // [T, K] int32 (out)
    tvm::ffi::Optional<tvm::ffi::TensorView> local_expert_mapping,
    tvm::ffi::Optional<tvm::ffi::TensorView> num_valid_tokens,
    double router_output_multiplier,
    double route_scale,
    double division_eps) {
  using namespace host;
  using Trait = muse_spark::RoutingTrait<kTopK, kExpertsPerLane, kNormMode, kWarpsPerBlock>;

  SymbolicSize T{"num_tokens"};
  SymbolicSize E{"num_experts"};
  SymbolicSize K{"topk"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({T, E}).with_dtype<fp32_t>().with_device(device_).verify(gating_output);
  TensorMatcher({E}).with_dtype<fp32_t>().with_device(device_).verify(bias);
  TensorMatcher({T, K}).with_dtype<fp32_t>().with_device(device_).verify(topk_weights);
  TensorMatcher({T, K}).with_dtype<int32_t>().with_device(device_).verify(topk_ids);

  const int num_tokens = static_cast<int>(T.unwrap());
  const int num_experts = static_cast<int>(E.unwrap());
  const DLDevice device = device_.unwrap();

  RuntimeCheck(static_cast<int>(K.unwrap()) == kTopK, "topk mismatch: got ", K.unwrap(), ", compiled for ", kTopK);
  RuntimeCheck(num_experts <= Trait::kMaxExperts, "num_experts ", num_experts, " exceeds ", Trait::kMaxExperts);
  // 128-bit row loads: rows must stay 16B aligned and whole vectors in-bounds.
  RuntimeCheck(num_experts % 4 == 0, "num_experts must be a multiple of 4, got ", num_experts);
  RuntimeCheck(gating_output.stride(0) % 4 == 0, "gating_output row stride must be a multiple of 4");
  RuntimeCheck(gating_output.stride(1) == 1 && bias.stride(0) == 1, "gating_output / bias must be row-contiguous");
  RuntimeCheck(topk_weights.stride(1) == 1 && topk_ids.stride(1) == 1, "topk outputs must be row-contiguous");

  muse_spark::RoutingParams params{
      .gating_output = static_cast<const float*>(gating_output.data_ptr()),
      .bias = static_cast<const float*>(bias.data_ptr()),
      .topk_weights = static_cast<float*>(topk_weights.data_ptr()),
      .topk_ids = static_cast<int32_t*>(topk_ids.data_ptr()),
      .packed_topk_ids = nullptr,
      .local_expert_mapping = nullptr,
      .num_valid_tokens = nullptr,
      .num_tokens = num_tokens,
      .num_experts = num_experts,
      .stride_gm = gating_output.stride(0),
      .stride_wm = topk_weights.stride(0),
      .stride_im = topk_ids.stride(0),
      .stride_pm = 0,
      .router_output_multiplier = static_cast<float>(router_output_multiplier),
      .route_scale = static_cast<float>(route_scale),
      .division_eps = static_cast<float>(division_eps),
  };

  if (packed_topk_ids.has_value()) {
    const auto& packed = packed_topk_ids.value();
    TensorMatcher({T, K}).with_dtype<int32_t>().with_device(device_).verify(packed);
    RuntimeCheck(packed.stride(1) == 1, "packed_topk_ids must be row-contiguous");
    params.packed_topk_ids = static_cast<int32_t*>(packed.data_ptr());
    params.stride_pm = packed.stride(0);
  }
  if (local_expert_mapping.has_value()) {
    TensorMatcher({E}).with_dtype<int32_t>().with_device(device_).verify(local_expert_mapping.value());
    params.local_expert_mapping = static_cast<const int32_t*>(local_expert_mapping.value().data_ptr());
  }
  if (num_valid_tokens.has_value()) {
    SymbolicSize One{"num_valid_numel"};
    TensorMatcher({One}).with_dtype<int32_t>().with_device(device_).verify(num_valid_tokens.value());
    RuntimeCheck(One.unwrap() == 1, "num_valid_tokens must be a 1-element tensor");
    params.num_valid_tokens = static_cast<const int32_t*>(num_valid_tokens.value().data_ptr());
  }

  if (num_tokens == 0) return;

  const dim3 grid(static_cast<unsigned>((num_tokens + kWarpsPerBlock - 1) / kWarpsPerBlock));
  LaunchKernel(grid, kWarpsPerBlock * device::kWarpThreads, device, 0)
      .enable_pdl(kUsePDL)(muse_spark::routing_kernel<Trait, kUsePDL>, params);
}

}  // namespace sglang
