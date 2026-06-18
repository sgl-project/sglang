#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <cstdint>
#include <cuda_fp8.h>

namespace {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::pack_fp8;

struct MegaMoEPreDispatchParams {
  const bf16_t* __restrict__ x;            // [num_tokens, hidden]
  const int32_t* __restrict__ topk_idx;    // [num_tokens, top_k]
  const float* __restrict__ topk_weights;  // [num_tokens, top_k]

  fp8_e4m3_t* __restrict__ buf_x;        // [padded_max, hidden]
  int32_t* __restrict__ buf_x_sf;        // contiguous int32 [P, G/4]; see layout comment
  int64_t* __restrict__ buf_topk_idx;    // [padded_max, top_k]
  float* __restrict__ buf_topk_weights;  // [padded_max, top_k]

  uint32_t num_tokens;
  uint32_t padded_max;
  uint32_t hidden;
  uint32_t num_groups;  // hidden / group_size
  uint32_t top_k;
};

struct MegaMoEPreDispatchWaterfillRank2Params {
  const bf16_t* __restrict__ x;            // [num_tokens, hidden]
  const int32_t* __restrict__ topk_idx;    // [num_tokens, routed_top_k]
  const float* __restrict__ topk_weights;  // [num_tokens, routed_top_k]
  const int64_t* __restrict__ rank_load;   // [2]

  fp8_e4m3_t* __restrict__ buf_x;        // [padded_max, hidden]
  int32_t* __restrict__ buf_x_sf;        // contiguous int32 [P, G/4]
  int64_t* __restrict__ buf_topk_idx;    // [padded_max, routed_top_k + 1]
  float* __restrict__ buf_topk_weights;  // [padded_max, routed_top_k + 1]

  uint32_t num_tokens;
  uint32_t padded_max;
  uint32_t hidden;
  uint32_t num_groups;  // hidden / group_size
  uint32_t routed_top_k;
  uint32_t out_top_k;
  uint32_t old_experts_per_rank;
  uint32_t new_experts_per_rank;
  uint32_t shared_replicas_per_rank;
  uint32_t source_rank;
  float shared_weight;
  int32_t local_pref_numer;
  int32_t local_pref_denom;
  int32_t remote_cost_tokens;
  bool allow_all_ranks;
  bool one_way_remote_shared;
};

// kGroupSize must match sglang_per_token_group_quant_fp8_ue8m0(group_size=).
template <uint32_t kGroupSize, bool kUsePDL>
__global__ __launch_bounds__(1024, 2) void  //
    mega_moe_pre_dispatch_kernel(const MegaMoEPreDispatchParams __grid_constant__ params) {
  using namespace device;

  constexpr uint32_t kVecElems = 8;  // 8 bf16 = 16B load per thread
  static_assert(kGroupSize % kVecElems == 0, "group_size must be a multiple of 8");
  constexpr uint32_t kThreadsPerGroup = kGroupSize / kVecElems;
  using InputVec = AlignedVector<bf16x2_t, kVecElems / 2>;
  using OutputVec = AlignedVector<fp8x2_e4m3_t, kVecElems / 2>;

  const uint32_t bid = blockIdx.x;
  const uint32_t tid = threadIdx.x;

  PDLWaitPrimary<kUsePDL>();
  if (bid < params.num_tokens) {
    // ---- Quantize path: one CTA per valid token ----

    const uint32_t token_id = bid;
    const auto token_in = params.x + static_cast<uint64_t>(token_id) * params.hidden;
    const auto token_out = params.buf_x + static_cast<uint64_t>(token_id) * params.hidden;

    InputVec in_vec;
    in_vec.load(token_in, tid);

    float local_max = 0.0f;
    float vals[kVecElems];
#pragma unroll
    for (uint32_t i = 0; i < kVecElems / 2; ++i) {
      const auto [v0, v1] = cast<fp32x2_t>(in_vec[i]);
      vals[2 * i + 0] = v0;
      vals[2 * i + 1] = v1;
      local_max = fmaxf(local_max, fmaxf(fabsf(v0), fabsf(v1)));
    }

    // Absmax across the kThreadsPerGroup threads that cover one group.
    local_max = warp::reduce_max<kThreadsPerGroup>(local_max);

    const float absmax = fmaxf(local_max, 1e-10f);
    const float raw_scale = absmax / math::FP8_E4M3_MAX;
    const uint32_t ue8m0_exp = cast_to_ue8m0(raw_scale);
    // 2^-ue8m0_exp as fp32 (equivalent to 1 / __uint_as_float(ue8m0 << 23)).
    const float inv_scale = __uint_as_float((127u + 127u - ue8m0_exp) << 23);

    OutputVec out_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecElems / 2; ++i) {
      out_vec[i] = pack_fp8(vals[2 * i + 0] * inv_scale, vals[2 * i + 1] * inv_scale);
    }
    out_vec.store(token_out, tid);

    // One thread per group writes its UE8M0 byte into the contiguous
    // row-major int32-packed layout: byte address = t*num_groups + g
    // (see layout comment at the top of the file).
    const uint32_t group_id = tid / kThreadsPerGroup;
    const uint32_t within_group_id = tid % kThreadsPerGroup;
    if (within_group_id == 0 && group_id < params.num_groups) {
      const uint32_t byte_off = token_id * params.num_groups + group_id;
      reinterpret_cast<uint8_t*>(params.buf_x_sf)[byte_off] = static_cast<uint8_t>(ue8m0_exp);
    }

    // Copy this token's topk row (no alignment assumptions; top_k is small).
    if (tid < params.top_k) {
      const uint32_t off = token_id * params.top_k + tid;
      params.buf_topk_idx[off] = params.topk_idx[off];
      params.buf_topk_weights[off] = params.topk_weights[off];
    }
  } else {
    // ---- Pad path: trailing blocks fill [num_tokens, padded_max) with (-1, 0) ----
    const uint32_t copy_bid = bid - params.num_tokens;
    const uint32_t pad_base = params.num_tokens * params.top_k;
    const uint32_t slot = pad_base + copy_bid * blockDim.x + tid;
    const uint32_t total_slots = params.padded_max * params.top_k;

    if (slot < total_slots) {
      params.buf_topk_idx[slot] = -1;
      params.buf_topk_weights[slot] = 0.0f;
    }
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <uint32_t kGroupSize, bool kUsePDL>
__global__ __launch_bounds__(1024, 2) void  //
    mega_moe_pre_dispatch_waterfill_rank2_kernel(
        const MegaMoEPreDispatchWaterfillRank2Params __grid_constant__ params) {
  using namespace device;

  constexpr uint32_t kVecElems = 8;
  static_assert(kGroupSize % kVecElems == 0, "group_size must be a multiple of 8");
  constexpr uint32_t kThreadsPerGroup = kGroupSize / kVecElems;
  using InputVec = AlignedVector<bf16x2_t, kVecElems / 2>;
  using OutputVec = AlignedVector<fp8x2_e4m3_t, kVecElems / 2>;

  const uint32_t bid = blockIdx.x;
  const uint32_t tid = threadIdx.x;

  PDLWaitPrimary<kUsePDL>();
  if (bid < params.num_tokens) {
    const uint32_t token_id = bid;
    const auto token_in = params.x + static_cast<uint64_t>(token_id) * params.hidden;
    const auto token_out = params.buf_x + static_cast<uint64_t>(token_id) * params.hidden;

    InputVec in_vec;
    in_vec.load(token_in, tid);

    float local_max = 0.0f;
    float vals[kVecElems];
#pragma unroll
    for (uint32_t i = 0; i < kVecElems / 2; ++i) {
      const auto [v0, v1] = cast<fp32x2_t>(in_vec[i]);
      vals[2 * i + 0] = v0;
      vals[2 * i + 1] = v1;
      local_max = fmaxf(local_max, fmaxf(fabsf(v0), fabsf(v1)));
    }

    local_max = warp::reduce_max<kThreadsPerGroup>(local_max);

    const float absmax = fmaxf(local_max, 1e-10f);
    const float raw_scale = absmax / math::FP8_E4M3_MAX;
    const uint32_t ue8m0_exp = cast_to_ue8m0(raw_scale);
    const float inv_scale = __uint_as_float((127u + 127u - ue8m0_exp) << 23);

    OutputVec out_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecElems / 2; ++i) {
      out_vec[i] = pack_fp8(vals[2 * i + 0] * inv_scale, vals[2 * i + 1] * inv_scale);
    }
    out_vec.store(token_out, tid);

    const uint32_t group_id = tid / kThreadsPerGroup;
    const uint32_t within_group_id = tid % kThreadsPerGroup;
    if (within_group_id == 0 && group_id < params.num_groups) {
      const uint32_t byte_off = token_id * params.num_groups + group_id;
      reinterpret_cast<uint8_t*>(params.buf_x_sf)[byte_off] = static_cast<uint8_t>(ue8m0_exp);
    }

    if (tid < params.routed_top_k) {
      const uint32_t in_off = token_id * params.routed_top_k + tid;
      const uint32_t out_off = token_id * params.out_top_k + tid;
      const int64_t old_id = static_cast<int64_t>(params.topk_idx[in_off]);
      const bool valid = old_id >= 0;
      const int64_t old_rank = valid ? old_id / static_cast<int64_t>(params.old_experts_per_rank) : 0;
      params.buf_topk_idx[out_off] =
          valid ? old_id + old_rank * static_cast<int64_t>(params.shared_replicas_per_rank) : old_id;
      params.buf_topk_weights[out_off] = valid ? params.topk_weights[in_off] : 0.0f;
    } else if (tid == params.routed_top_k) {
      const int64_t load0 = params.rank_load[0];
      const int64_t load1 = params.rank_load[1];
      const int64_t total_effective = load0 + load1;
      const int64_t total_tokens = total_effective / static_cast<int64_t>(params.routed_top_k);
      const int64_t target_total = (total_effective + total_tokens + 1) / 2;

      const int64_t adjusted_load0 =
          params.source_rank == 0 ? load0 : load0 + static_cast<int64_t>(params.remote_cost_tokens);
      const int64_t adjusted_load1 =
          params.source_rank == 1 ? load1 : load1 + static_cast<int64_t>(params.remote_cost_tokens);
      int32_t deficit0 =
          target_total > adjusted_load0 ? static_cast<int32_t>(target_total - adjusted_load0) : 0;
      int32_t deficit1 =
          target_total > adjusted_load1 ? static_cast<int32_t>(target_total - adjusted_load1) : 0;

      bool has_valid = false;
      bool has_rank0 = params.source_rank == 0;
      bool has_rank1 = params.source_rank == 1;
      for (uint32_t k = 0; k < params.routed_top_k; ++k) {
        const int64_t old_id =
            static_cast<int64_t>(params.topk_idx[token_id * params.routed_top_k + k]);
        if (old_id >= 0) {
          has_valid = true;
          const uint32_t old_rank = static_cast<uint32_t>(old_id) / params.old_experts_per_rank;
          has_rank0 |= old_rank == 0;
          has_rank1 |= old_rank == 1;
        }
      }

      int32_t weight0 = 0;
      int32_t weight1 = 0;
      if (params.allow_all_ranks && params.one_way_remote_shared) {
        const int64_t local_load = params.source_rank == 0 ? load0 : load1;
        const int64_t remote_load = params.source_rank == 0 ? load1 : load0;
        int32_t remote_budget =
            local_load + static_cast<int64_t>(params.num_tokens) > target_total
                ? static_cast<int32_t>(local_load + static_cast<int64_t>(params.num_tokens) - target_total)
                : 0;
        remote_budget = remote_budget > static_cast<int32_t>(params.num_tokens)
                            ? static_cast<int32_t>(params.num_tokens)
                            : remote_budget;
        const bool source_is_heavy = local_load > remote_load;
        const int32_t remote_weight = source_is_heavy ? remote_budget : 0;
        const int32_t local_weight = static_cast<int32_t>(params.num_tokens) - remote_weight;
        if (params.source_rank == 0) {
          weight0 = local_weight;
          weight1 = remote_weight;
        } else {
          weight0 = remote_weight;
          weight1 = local_weight;
        }
      } else {
        if (params.source_rank == 0) {
          weight0 = deficit0;
          weight1 = (deficit1 * params.local_pref_denom) / params.local_pref_numer;
        } else {
          weight0 = (deficit0 * params.local_pref_denom) / params.local_pref_numer;
          weight1 = deficit1;
        }
      }
      if (!params.allow_all_ranks) {
        weight0 = has_rank0 ? weight0 : 0;
        weight1 = has_rank1 ? weight1 : 0;
      }

      const int32_t total_w = weight0 + weight1;
      uint32_t token_seed =
          token_id ^ (params.source_rank * static_cast<uint32_t>(0x9E3779B9u));
      token_seed = token_seed * static_cast<uint32_t>(1664525u) + static_cast<uint32_t>(1013904223u);
      const int32_t u = total_w > 0 ? static_cast<int32_t>(token_seed % static_cast<uint32_t>(total_w)) : 0;
      uint32_t chosen_rank = u < weight0 ? 0u : 1u;

      if (total_w == 0) {
        if (params.allow_all_ranks) {
          chosen_rank = params.source_rank;
        } else {
          bool remote_better = false;
          if (params.source_rank == 0) {
            remote_better =
                (adjusted_load1 * params.local_pref_numer < load0 * params.local_pref_denom) && has_rank1;
          } else {
            remote_better =
                (adjusted_load0 * params.local_pref_numer < load1 * params.local_pref_denom) && has_rank0;
          }
          chosen_rank = remote_better ? (1u - params.source_rank) : params.source_rank;
        }
      }

      uint32_t replica_seed =
          token_id ^ (chosen_rank * static_cast<uint32_t>(0x85EBCA6Bu));
      replica_seed = replica_seed * static_cast<uint32_t>(1103515245u) + static_cast<uint32_t>(12345u);
      const uint32_t shared_replica = replica_seed % params.shared_replicas_per_rank;
      const int64_t shared_id =
          static_cast<int64_t>(chosen_rank) * static_cast<int64_t>(params.new_experts_per_rank) +
          static_cast<int64_t>(params.old_experts_per_rank) + static_cast<int64_t>(shared_replica);
      const uint32_t out_off = token_id * params.out_top_k + params.routed_top_k;
      params.buf_topk_idx[out_off] = has_valid ? shared_id : -1;
      params.buf_topk_weights[out_off] = has_valid ? params.shared_weight : 0.0f;
    }
  } else {
    const uint32_t copy_bid = bid - params.num_tokens;
    const uint32_t pad_base = params.num_tokens * params.out_top_k;
    const uint32_t slot = pad_base + copy_bid * blockDim.x + tid;
    const uint32_t total_slots = params.padded_max * params.out_top_k;

    if (slot < total_slots) {
      params.buf_topk_idx[slot] = -1;
      params.buf_topk_weights[slot] = 0.0f;
    }
  }
  PDLTriggerSecondary<kUsePDL>();
}

// ---- Host wrapper
// ------------------------------------------------------------------------------------------------------------------------

template <int64_t kGroupSize, bool kUsePDL>
struct MegaMoEPreDispatchKernel {
  static_assert(kGroupSize == 32 || kGroupSize == 64 || kGroupSize == 128, "unsupported group_size");
  static constexpr auto kernel = mega_moe_pre_dispatch_kernel<static_cast<uint32_t>(kGroupSize), kUsePDL>;

  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView topk_idx,
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView buf_x,
      const tvm::ffi::TensorView buf_x_sf,
      const tvm::ffi::TensorView buf_topk_idx,
      const tvm::ffi::TensorView buf_topk_weights) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    auto P = SymbolicSize{"padded_max"};
    auto H = SymbolicSize{"hidden"};
    auto K = SymbolicSize{"top_k"};
    auto G4 = SymbolicSize{"num_groups_div_4"};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, H})  // input x
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(x);
    TensorMatcher({M, K})  // topk_idx
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(topk_idx);
    TensorMatcher({M, K})  // topk_weights
        .with_dtype<float>()
        .with_device(device)
        .verify(topk_weights);
    // DeepGEMM versions expose this fp8 dispatch buffer either as raw int8
    // storage or as torch.float8_e4m3fn; the kernel writes fp8 bytes in both.
    TensorMatcher({P, H})  // buf.x
        .with_dtype<int8_t, fp8_e4m3_t>()
        .with_device(device)
        .verify(buf_x);
    // buf.x_sf is the contiguous row-major int32 view from DeepGEMM's mega
    // symm buffer (DeepGEMM/csrc/apis/mega.hpp): shape (P, G/4), strides
    // (G/4, 1). No explicit strides required -> TensorMatcher enforces
    // is_contiguous().
    TensorMatcher({P, G4})  // buf_x_sf
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(buf_x_sf);
    TensorMatcher({P, K})  // buf.topk_idx
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(buf_topk_idx);
    TensorMatcher({P, K})  // buf.topk_weights
        .with_dtype<float>()
        .with_device(device)
        .verify(buf_topk_weights);

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    const auto padded_max = static_cast<uint32_t>(P.unwrap());
    const auto hidden = static_cast<uint32_t>(H.unwrap());
    const auto top_k = static_cast<uint32_t>(K.unwrap());
    const auto num_groups_div_4 = static_cast<uint32_t>(G4.unwrap());

    RuntimeCheck(num_tokens <= padded_max, "num_tokens must not exceed padded_max");
    RuntimeCheck(hidden % kGroupSize == 0, "hidden must be a multiple of group_size");
    const auto num_groups = hidden / static_cast<uint32_t>(kGroupSize);
    RuntimeCheck(num_groups == num_groups_div_4 * 4u, "num_groups must be a multiple of 4");
    RuntimeCheck(hidden % 8u == 0, "hidden must be a multiple of 8 (16B bf16 loads)");
    const auto num_threads = hidden / 8u;
    RuntimeCheck(num_threads <= 1024, "hidden too large for single-block-per-row quant");
    RuntimeCheck(num_threads >= top_k, "top_k must fit into one quant CTA");

    const auto pad_slots = (padded_max - num_tokens) * top_k;
    const uint32_t num_pad_blocks = pad_slots == 0 ? 0u : ((pad_slots + num_threads - 1u) / num_threads);
    const auto num_total_blocks = num_tokens + num_pad_blocks;

    const auto params = MegaMoEPreDispatchParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .topk_idx = static_cast<const int32_t*>(topk_idx.data_ptr()),
        .topk_weights = static_cast<const float*>(topk_weights.data_ptr()),
        .buf_x = static_cast<fp8_e4m3_t*>(buf_x.data_ptr()),
        .buf_x_sf = static_cast<int32_t*>(buf_x_sf.data_ptr()),
        .buf_topk_idx = static_cast<int64_t*>(buf_topk_idx.data_ptr()),
        .buf_topk_weights = static_cast<float*>(buf_topk_weights.data_ptr()),
        .num_tokens = num_tokens,
        .padded_max = padded_max,
        .hidden = hidden,
        .num_groups = num_groups,
        .top_k = top_k,
    };

    if (num_total_blocks == 0) return;
    LaunchKernel(num_total_blocks, num_threads, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

template <int64_t kGroupSize, bool kUsePDL>
struct MegaMoEPreDispatchWaterfillRank2Kernel {
  static_assert(kGroupSize == 32 || kGroupSize == 64 || kGroupSize == 128, "unsupported group_size");
  static constexpr auto kernel =
      mega_moe_pre_dispatch_waterfill_rank2_kernel<static_cast<uint32_t>(kGroupSize), kUsePDL>;

  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView topk_idx,
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView rank_load,
      const tvm::ffi::TensorView buf_x,
      const tvm::ffi::TensorView buf_x_sf,
      const tvm::ffi::TensorView buf_topk_idx,
      const tvm::ffi::TensorView buf_topk_weights,
      int64_t source_rank,
      double shared_weight,
      int64_t local_pref_numer,
      int64_t local_pref_denom,
      int64_t remote_cost_tokens,
      bool allow_all_ranks,
      bool one_way_remote_shared,
      int64_t old_experts_per_rank,
      int64_t new_experts_per_rank,
      int64_t shared_replicas_per_rank) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"num_tokens"};
    auto P = SymbolicSize{"padded_max"};
    auto H = SymbolicSize{"hidden"};
    auto K = SymbolicSize{"routed_top_k"};
    auto KO = SymbolicSize{"out_top_k"};
    auto G4 = SymbolicSize{"num_groups_div_4"};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, H}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({M, K}).with_dtype<int32_t>().with_device(device).verify(topk_idx);
    TensorMatcher({M, K}).with_dtype<float>().with_device(device).verify(topk_weights);
    TensorMatcher({2}).with_dtype<int64_t>().with_device(device).verify(rank_load);
    TensorMatcher({P, H}).with_dtype<int8_t, fp8_e4m3_t>().with_device(device).verify(buf_x);
    TensorMatcher({P, G4}).with_dtype<int32_t>().with_device(device).verify(buf_x_sf);
    TensorMatcher({P, KO}).with_dtype<int64_t>().with_device(device).verify(buf_topk_idx);
    TensorMatcher({P, KO}).with_dtype<float>().with_device(device).verify(buf_topk_weights);

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    const auto padded_max = static_cast<uint32_t>(P.unwrap());
    const auto hidden = static_cast<uint32_t>(H.unwrap());
    const auto routed_top_k = static_cast<uint32_t>(K.unwrap());
    const auto out_top_k = static_cast<uint32_t>(KO.unwrap());
    const auto num_groups_div_4 = static_cast<uint32_t>(G4.unwrap());

    RuntimeCheck(out_top_k == routed_top_k + 1u, "Waterfill output topk must be routed_top_k + 1");
    RuntimeCheck(source_rank == 0 || source_rank == 1, "rank2 Waterfill source_rank must be 0 or 1");
    RuntimeCheck(old_experts_per_rank > 0, "old_experts_per_rank must be positive");
    RuntimeCheck(new_experts_per_rank > old_experts_per_rank, "new_experts_per_rank must include shared slots");
    RuntimeCheck(shared_replicas_per_rank > 0, "shared_replicas_per_rank must be positive");
    RuntimeCheck(
        new_experts_per_rank == old_experts_per_rank + shared_replicas_per_rank,
        "new_experts_per_rank must equal old_experts_per_rank + shared_replicas_per_rank");
    RuntimeCheck(local_pref_numer > 0 && local_pref_denom > 0, "local preference ratio must be positive");
    RuntimeCheck(num_tokens <= padded_max, "num_tokens must not exceed padded_max");
    RuntimeCheck(hidden % kGroupSize == 0, "hidden must be a multiple of group_size");
    const auto num_groups = hidden / static_cast<uint32_t>(kGroupSize);
    RuntimeCheck(num_groups == num_groups_div_4 * 4u, "num_groups must be a multiple of 4");
    RuntimeCheck(hidden % 8u == 0, "hidden must be a multiple of 8 (16B bf16 loads)");
    const auto num_threads = hidden / 8u;
    RuntimeCheck(num_threads <= 1024, "hidden too large for single-block-per-row quant");
    RuntimeCheck(num_threads > routed_top_k, "routed_top_k + shared column must fit into one quant CTA");

    const auto pad_slots = (padded_max - num_tokens) * out_top_k;
    const uint32_t num_pad_blocks = pad_slots == 0 ? 0u : ((pad_slots + num_threads - 1u) / num_threads);
    const auto num_total_blocks = num_tokens + num_pad_blocks;

    const auto params = MegaMoEPreDispatchWaterfillRank2Params{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .topk_idx = static_cast<const int32_t*>(topk_idx.data_ptr()),
        .topk_weights = static_cast<const float*>(topk_weights.data_ptr()),
        .rank_load = static_cast<const int64_t*>(rank_load.data_ptr()),
        .buf_x = static_cast<fp8_e4m3_t*>(buf_x.data_ptr()),
        .buf_x_sf = static_cast<int32_t*>(buf_x_sf.data_ptr()),
        .buf_topk_idx = static_cast<int64_t*>(buf_topk_idx.data_ptr()),
        .buf_topk_weights = static_cast<float*>(buf_topk_weights.data_ptr()),
        .num_tokens = num_tokens,
        .padded_max = padded_max,
        .hidden = hidden,
        .num_groups = num_groups,
        .routed_top_k = routed_top_k,
        .out_top_k = out_top_k,
        .old_experts_per_rank = static_cast<uint32_t>(old_experts_per_rank),
        .new_experts_per_rank = static_cast<uint32_t>(new_experts_per_rank),
        .shared_replicas_per_rank = static_cast<uint32_t>(shared_replicas_per_rank),
        .source_rank = static_cast<uint32_t>(source_rank),
        .shared_weight = static_cast<float>(shared_weight),
        .local_pref_numer = static_cast<int32_t>(local_pref_numer),
        .local_pref_denom = static_cast<int32_t>(local_pref_denom),
        .remote_cost_tokens = static_cast<int32_t>(remote_cost_tokens),
        .allow_all_ranks = allow_all_ranks,
        .one_way_remote_shared = one_way_remote_shared,
    };

    if (num_total_blocks == 0) return;
    LaunchKernel(num_total_blocks, num_threads, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
