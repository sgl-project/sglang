// NOTE(nvpohanh): The metadata must follow trtllm-gen's routingPermutation
// contract (RoutingKernel.cuh): local experts in id order, each padded to whole
// tiles, padding rows of the token map set to -1.

#include "route_quant_fused.cuh"

namespace sglang {

static_assert(
    kNumExperts_ == 896 && kTopK_ == 16 && kQuantHidden_ == 3584,
    "kimi_k3_fused_routing assumes the Kimi-K3 / K3.1 MoE: 896 experts, top-16, hidden 3584");

inline constexpr uint32_t kKimiK3RoutingMaxTokens = 256;
inline constexpr uint32_t kKimiK3RoutingMaxExpanded = kKimiK3RoutingMaxTokens * kTopK_;
inline constexpr uint32_t kKimiK3RoutingExpertsPerThread = kNumExperts_ / LargeRouterRadixTrait::kBlockSize;

struct KimiK3RoutingMetaParams {
  void* __restrict__ expert_weights;
  int32_t* __restrict__ expanded_idx_to_permuted_idx;
  int32_t* __restrict__ permuted_idx_to_token_idx;
  int32_t* __restrict__ cta_idx_xy_to_batch_idx;
  int32_t* __restrict__ cta_idx_xy_to_mn_limit;
  int32_t* __restrict__ num_non_exiting_ctas;
  int32_t* __restrict__ total_num_padded_tokens;
  int32_t* __restrict__ num_tokens_per_expert;
  // NOTE(nvpohanh): The last routing CTA resets the counter to zero, so
  // launches sharing one must not overlap.
  uint32_t* __restrict__ counter;
  const int32_t* __restrict__ num_valid_ptr;
  int32_t num_valid_value;
  int32_t local_expert_offset;
  int32_t num_local_experts;
  int32_t tile_tokens;
  int32_t weights_fp32;
};

struct KimiK3FusedRoutingParams {
  RouteRadixParams route;
  QuantKernelParams quant;
  KimiK3RoutingMetaParams meta;
};

struct KimiK3RoutingSmem {
  uint32_t warp_sum[LargeRouterRadixTrait::kNumWarps];
  int32_t num_ctas;
  int32_t expert_count[kNumExperts_];
  int32_t expert_offset[kNumExperts_];
  int32_t entry[kKimiK3RoutingMaxExpanded];
};

SGL_DEVICE void kimi_k3_build_routing_metadata(
    const RouteRadixParams& route, const KimiK3RoutingMetaParams& meta, KimiK3RoutingSmem& smem) {
  using T = LargeRouterRadixTrait;
  constexpr uint32_t kThreads = T::kBlockSize;
  constexpr uint32_t kEPT = kKimiK3RoutingExpertsPerThread;
  const uint32_t tx = threadIdx.x;
  const uint32_t lane_id = tx % device::kWarpThreads;
  const uint32_t warp_id = tx / device::kWarpThreads;
  const int32_t num_local = meta.num_local_experts;
  const int32_t first = meta.local_expert_offset;
  const int32_t tile = meta.tile_tokens;
  const uint32_t num_expanded = static_cast<uint32_t>(route.M) * T::kTopK;

  for (uint32_t e = tx; e < kNumExperts_; e += kThreads) {
    smem.expert_count[e] = 0;
  }
  __syncthreads();

  constexpr uint32_t kIdsPerThread = (kKimiK3RoutingMaxExpanded + kThreads - 1) / kThreads;
  int32_t ids[kIdsPerThread];
#pragma unroll
  for (uint32_t j = 0; j < kIdsPerThread; ++j) {
    const uint32_t i = tx + j * kThreads;
    ids[j] = i < num_expanded ? __ldcg(route.out_i + (i / T::kTopK) * route.out_i_stride + i % T::kTopK) : -1;
  }
#pragma unroll
  for (uint32_t j = 0; j < kIdsPerThread; ++j) {
    const uint32_t i = tx + j * kThreads;
    if (i < num_expanded) {
      const int32_t local = ids[j] - first;
      int32_t entry = -1;
      if (ids[j] >= 0 && local >= 0 && local < num_local) {
        entry = (local << 16) | atomicAdd(&smem.expert_count[local], 1);
      }
      smem.entry[i] = entry;
    }
  }
  __syncthreads();

  int32_t count[kEPT];
  int32_t num_ctas[kEPT];
  int32_t thread_ctas = 0;
#pragma unroll
  for (uint32_t j = 0; j < kEPT; ++j) {
    const int32_t e = static_cast<int32_t>(tx * kEPT + j);
    count[j] = e < num_local ? smem.expert_count[e] : 0;
    num_ctas[j] = (count[j] + tile - 1) / tile;
    thread_ctas += num_ctas[j];
  }
  int32_t cta_offset = static_cast<int32_t>(
      moe::radix::block_exclusive_sum(static_cast<uint32_t>(thread_ctas), lane_id, warp_id, smem.warp_sum));
  if (tx == kThreads - 1) smem.num_ctas = cta_offset + thread_ctas;

#pragma unroll
  for (uint32_t j = 0; j < kEPT; ++j) {
    const int32_t e = static_cast<int32_t>(tx * kEPT + j);
    if (e < num_local) {
      smem.expert_offset[e] = cta_offset * tile;
      const int32_t limit = cta_offset * tile + count[j];
      for (int32_t c = 0; c < num_ctas[j]; ++c) {
        const int32_t cta = cta_offset + c;
        const int32_t tile_end = (cta + 1) * tile;
        const int32_t mn_limit = min(tile_end, limit);
        meta.cta_idx_xy_to_batch_idx[cta] = e;
        meta.cta_idx_xy_to_mn_limit[cta] = mn_limit;
        for (int32_t p = mn_limit; p < tile_end; ++p) {
          meta.permuted_idx_to_token_idx[p] = -1;
        }
      }
    }
    cta_offset += num_ctas[j];
  }
  for (int32_t e = static_cast<int32_t>(tx); e < static_cast<int32_t>(kNumExperts_); e += kThreads) {
    const int32_t local = e - first;
    meta.num_tokens_per_expert[e] = (local >= 0 && local < num_local) ? smem.expert_count[local] : 0;
  }
  __syncthreads();
  if (tx == 0) {
    meta.num_non_exiting_ctas[0] = smem.num_ctas;
    meta.total_num_padded_tokens[0] = smem.num_ctas * tile;
  }

  for (uint32_t i = tx; i < num_expanded; i += kThreads) {
    const int32_t entry = smem.entry[i];
    int32_t permuted = -1;
    if (entry >= 0) {
      permuted = smem.expert_offset[entry >> 16] + (entry & 0xffff);
      meta.permuted_idx_to_token_idx[permuted] = static_cast<int32_t>(i / T::kTopK);
    }
    meta.expanded_idx_to_permuted_idx[i] = permuted;
  }
}

template <bool kUsePDL, typename TScore, typename TX>
__global__ __launch_bounds__(LargeRouterRadixTrait::kBlockSize)  //
    void kimi_k3_fused_routing_kernel(const __grid_constant__ KimiK3FusedRoutingParams params) {
  using T = LargeRouterRadixTrait;
  const auto M = static_cast<uint32_t>(params.route.M);
  const uint32_t tx = threadIdx.x;
  if (blockIdx.x < M) {
    __shared__ union {
      typename T::Smem route;
      KimiK3RoutingSmem meta;
    } smem;
    __shared__ bool is_last;
    const auto& meta = params.meta;
    device::PDLWaitPrimary<kUsePDL>();
    const int32_t num_valid = meta.num_valid_ptr != nullptr ? meta.num_valid_ptr[0] : meta.num_valid_value;
    route_radix_block</*kUsePDL=*/false, TScore>(params.route, smem.route);
    // NOTE(nvpohanh): Early trigger is safe: the dependent's
    // griddepcontrol.wait still waits for this whole grid, metadata included.
    device::PDLTriggerSecondary<kUsePDL>();

    const uint32_t row = blockIdx.x;
    if (tx < T::kTopK) {
      const float w = params.route.out_w[row * params.route.out_w_stride + tx];
      if (static_cast<int32_t>(row) >= num_valid) {
        params.route.out_i[row * params.route.out_i_stride + tx] = -1;
      }
      if (meta.weights_fp32) {
        static_cast<fp32_t*>(meta.expert_weights)[row * T::kTopK + tx] = w;
      } else {
        static_cast<bf16_t*>(meta.expert_weights)[row * T::kTopK + tx] = __float2bfloat16_rn(w);
      }
    }
    __syncthreads();
    if (tx == 0) {
      __threadfence();
      is_last = atomicAdd(meta.counter, 1u) == M - 1;
      if (is_last) __threadfence();
    }
    __syncthreads();
    if (!is_last) return;
    kimi_k3_build_routing_metadata(params.route, meta, smem.meta);
    if (tx == 0) *meta.counter = 0u;
  } else {
    device::PDLWaitPrimary<kUsePDL>();
    using Q = RouteQuantTraitT<TX>;
    Q::run(params.quant, /*expert_idx=*/0, blockIdx.x - M, threadIdx.x / Q::kNumLanes, threadIdx.x % Q::kNumLanes);
    device::PDLTriggerSecondary<kUsePDL>();
  }
}

template <bool kUsePDL>
struct KimiK3FusedRoutingKernel {
  static void
  run(const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView out_q,
      const tvm::ffi::TensorView out_s,
      const tvm::ffi::TensorView expert_weights,
      const tvm::ffi::TensorView expanded_idx_to_permuted_idx,
      const tvm::ffi::TensorView permuted_idx_to_token_idx,
      const tvm::ffi::TensorView cta_idx_xy_to_batch_idx,
      const tvm::ffi::TensorView cta_idx_xy_to_mn_limit,
      const tvm::ffi::TensorView num_non_exiting_ctas,
      const tvm::ffi::TensorView total_num_padded_tokens,
      const tvm::ffi::TensorView num_tokens_per_expert,
      const tvm::ffi::TensorView counter,
      const tvm::ffi::Optional<tvm::ffi::TensorView> num_valid,
      int64_t topk,
      double routed_scaling_factor,
      bool renormalize,
      bool apply_scale,
      int64_t local_expert_offset,
      int64_t num_local_experts,
      int64_t tile_tokens) {
    using namespace host;

    auto M_ = SymbolicSize{"num_tokens"};
    auto N_ = SymbolicSize{"num_experts"};
    auto K_ = SymbolicSize{"topk"};
    auto P_ = SymbolicSize{"max_padded_rows"};
    auto C_ = SymbolicSize{"max_ctas"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    auto score_dtype = SymbolicDType{};
    TensorMatcher({M_, N_})
        .with_dtype<bf16_t, fp32_t>(score_dtype)
        .with_device(device)
        .with_strides({-1, 1})
        .verify(scores);
    TensorMatcher({N_}).with_dtype<fp32_t>().with_device(device).verify(bias);
    TensorMatcher({M_, K_}).with_dtype<fp32_t>().with_device(device).verify(out_w);
    TensorMatcher({M_, K_}).with_dtype<int32_t>().with_device(device).verify(out_i);
    auto w_dtype = SymbolicDType{};
    TensorMatcher({M_, K_}).with_dtype<bf16_t, fp32_t>(w_dtype).with_strides({K_, 1}).with_device(device).verify(
        expert_weights);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(expanded_idx_to_permuted_idx);
    TensorMatcher({P_}).with_dtype<int32_t>().with_device(device).verify(permuted_idx_to_token_idx);
    TensorMatcher({C_}).with_dtype<int32_t>().with_device(device).verify(cta_idx_xy_to_batch_idx);
    TensorMatcher({C_}).with_dtype<int32_t>().with_device(device).verify(cta_idx_xy_to_mn_limit);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(num_non_exiting_ctas);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(total_num_padded_tokens);
    TensorMatcher({N_}).with_dtype<int32_t>().with_device(device).verify(num_tokens_per_expert);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(counter);

    RuntimeCheck(
        N_.unwrap() == kNumExperts_ && K_.unwrap() == kTopK_ && topk == kTopK_,
        "kimi_k3_fused_routing is specialized for N=896, K=16");
    RuntimeCheck(scores.stride(0) % 4 == 0, "kimi_k3_fused_routing: scores row stride must be a multiple of 4");
    const auto M = static_cast<uint32_t>(M_.unwrap());
    RuntimeCheck(M <= kKimiK3RoutingMaxTokens, "kimi_k3_fused_routing: at most 256 rows");
    RuntimeCheck(
        expanded_idx_to_permuted_idx.size(0) == static_cast<int64_t>(M) * kTopK_,
        "kimi_k3_fused_routing: expanded_idx_to_permuted_idx must hold M * 16 entries");
    RuntimeCheck(
        0 <= local_expert_offset && 0 < num_local_experts &&
            local_expert_offset + num_local_experts <= static_cast<int64_t>(kNumExperts_),
        "kimi_k3_fused_routing: the local expert range must lie within the 896 experts");
    RuntimeCheck(0 < tile_tokens && tile_tokens <= 256, "kimi_k3_fused_routing: tile_tokens must be in [1, 256]");

    const int32_t* num_valid_ptr = nullptr;
    if (num_valid.has_value()) {
      const auto& nv = num_valid.value();
      RuntimeCheck(
          nv.numel() == 1 && nv.device().device_type == kDLCUDA && is_type<int32_t>(nv.dtype()),
          "kimi_k3_fused_routing: num_valid must be a CUDA int32 scalar");
      num_valid_ptr = static_cast<const int32_t*>(nv.data_ptr());
    }

    auto x_dtype = SymbolicDType{};
    TensorMatcher({M_, -1}).with_dtype<bf16_t, fp32_t>(x_dtype).with_device(device).with_strides({-1, 1}).verify(x);
    const auto quant_params =
        x_dtype.is_type<fp32_t>()
            ? build_quant_context<RouteQuantTraitT<fp32_t>, /*kMasked=*/false>(x, out_q, out_s).params
            : build_quant_context<RouteQuantTraitT<bf16_t>, /*kMasked=*/false>(x, out_q, out_s).params;
    RuntimeCheck(quant_params.hidden_size == kQuantHidden_, "kimi_k3_fused_routing: activation row must be 3584 wide");
    if (M == 0) return;

    const auto params = KimiK3FusedRoutingParams{
        .route =
            {scores.data_ptr(),
             static_cast<const fp32_t*>(bias.data_ptr()),
             static_cast<fp32_t*>(out_w.data_ptr()),
             static_cast<int32_t*>(out_i.data_ptr()),
             /*out_packed=*/nullptr,
             static_cast<int>(M),
             static_cast<long long>(scores.stride(0)),
             static_cast<long long>(out_w.stride(0)),
             static_cast<long long>(out_i.stride(0)),
             /*out_packed_stride=*/0,
             static_cast<float>(routed_scaling_factor),
             renormalize ? 1 : 0,
             apply_scale ? 1 : 0,
             /*sorted=*/0},
        .quant = quant_params,
        .meta =
            {expert_weights.data_ptr(),
             static_cast<int32_t*>(expanded_idx_to_permuted_idx.data_ptr()),
             static_cast<int32_t*>(permuted_idx_to_token_idx.data_ptr()),
             static_cast<int32_t*>(cta_idx_xy_to_batch_idx.data_ptr()),
             static_cast<int32_t*>(cta_idx_xy_to_mn_limit.data_ptr()),
             static_cast<int32_t*>(num_non_exiting_ctas.data_ptr()),
             static_cast<int32_t*>(total_num_padded_tokens.data_ptr()),
             static_cast<int32_t*>(num_tokens_per_expert.data_ptr()),
             static_cast<uint32_t*>(counter.data_ptr()),
             num_valid_ptr,
             static_cast<int32_t>(M),
             static_cast<int32_t>(local_expert_offset),
             static_cast<int32_t>(num_local_experts),
             static_cast<int32_t>(tile_tokens),
             w_dtype.is_type<fp32_t>() ? 1 : 0},
    };

#define SGL_KIMI_K3_ROUTING_LAUNCH(TS, TX)                                \
  LaunchKernel(2 * M, LargeRouterRadixTrait::kBlockSize, device.unwrap()) \
      .enable_pdl(kUsePDL)(kimi_k3_fused_routing_kernel<kUsePDL, TS, TX>, params)

    if (score_dtype.is_type<fp32_t>()) {
      if (x_dtype.is_type<fp32_t>()) {
        SGL_KIMI_K3_ROUTING_LAUNCH(fp32_t, fp32_t);
      } else {
        SGL_KIMI_K3_ROUTING_LAUNCH(fp32_t, bf16_t);
      }
    } else {
      if (x_dtype.is_type<fp32_t>()) {
        SGL_KIMI_K3_ROUTING_LAUNCH(bf16_t, fp32_t);
      } else {
        SGL_KIMI_K3_ROUTING_LAUNCH(bf16_t, bf16_t);
      }
    }
#undef SGL_KIMI_K3_ROUTING_LAUNCH
  }
};

}  // namespace sglang
