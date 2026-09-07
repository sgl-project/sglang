// K3 MoE-front prep in one launch: radix routing (+ trtllm packed ids) on the
// first M CTAs, mxfp8 per-token-group quant of the routed activations on the
// next M. At decode batch sizes the unfused chain is three tiny back-to-back
// kernels (route 3.8us + quant 2.6us + pack 1.4us per layer) each leaving the
// SMs idle; fused, the quant CTAs run concurrently with the routing CTA and
// the pack is a 16-store epilogue.
//
// Both halves are the existing kernels verbatim: route_radix_block is the
// standalone route_radix body (same TU, same flags — no fast-math, which the
// routing bit-exactness contract requires and the quant math tolerates: its
// only transcendentals are explicit intrinsics and exact bit manipulation),
// and QuantTrait::run is the per_token_group_quant math. Specialized like
// route_radix itself: 896 experts, top-16, and a 3584-wide bf16 activation row
// (112 ue8m0 groups of 32 = 224 lanes, exactly the routing block width).

#include "../gemm/per_token_group_quant.cuh"
#include "route_radix.cuh"

namespace sglang {

struct RouteQuantFusedParams {
  RouteRadixParams route;
  QuantKernelParams quant;
};

// One quant CTA covers one token row: thread pairs (2g, 2g+1) hold group g
// with lanes (0, 1) — the same subwarp layout the flat quant kernel derives
// from global_tid, so the group reduction and stores are bit-identical.
template <typename TX>
using RouteQuantTraitT = QuantTrait<
    TX,
    fp8_e4m3_t,
    /*kGroupSize=*/32,
    /*kUe8m0=*/true,
    /*kRowMajor=*/true,
    /*kAligned=*/true,
    /*kFuseSiluAndMul=*/false>;

using RouteQuantTrait = RouteQuantTraitT<bf16_t>;

inline constexpr uint32_t kQuantGroupsPerRow_ = LargeRouterRadixTrait::kBlockSize / RouteQuantTrait::kNumLanes;
inline constexpr uint32_t kQuantHidden_ = kQuantGroupsPerRow_ * RouteQuantTrait::kGroupSize;  // 3584

template <bool kUsePDL, typename TScore, typename TX>
__global__ __launch_bounds__(LargeRouterRadixTrait::kBlockSize)  //
    void route_quant_fused_kernel(const __grid_constant__ RouteQuantFusedParams params) {
  const auto M = static_cast<uint32_t>(params.route.M);
  if (blockIdx.x < M) {
    __shared__ typename LargeRouterRadixTrait::Smem smem;
    route_radix_block<kUsePDL, TScore>(params.route, smem);
  } else {
    // Quant CTAs read the same primary-kernel output (the fused-front GEMM)
    // as the routing CTAs, so they carry their own PDL wait/trigger.
    device::PDLWaitPrimary<kUsePDL>();
    const uint32_t token_idx = blockIdx.x - M;
    const uint32_t group_idx = threadIdx.x / RouteQuantTraitT<TX>::kNumLanes;
    const uint32_t lane_id = threadIdx.x % RouteQuantTraitT<TX>::kNumLanes;
    RouteQuantTraitT<TX>::run(params.quant, /*expert_idx=*/0, token_idx, group_idx, lane_id);
    device::PDLTriggerSecondary<kUsePDL>();
  }
}

template <bool kUsePDL>
struct RouteQuantFusedKernel {
  static void
  run(const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      const tvm::ffi::TensorView out_packed,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView out_q,
      const tvm::ffi::TensorView out_s,
      int64_t topk,
      double routed_scaling_factor,
      bool renormalize,
      bool apply_scale) {
    using namespace host;
    using Trait = RouteQuantTrait;

    auto M_ = SymbolicSize{"num_tokens"};
    auto N_ = SymbolicSize{"num_experts"};
    auto K_ = SymbolicSize{"topk"};
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
    TensorMatcher({M_, K_}).with_dtype<int32_t>().with_strides({-1, 1}).with_device(device).verify(out_packed);

    RuntimeCheck(
        N_.unwrap() == kNumExperts_ && K_.unwrap() == kTopK_ && topk == kTopK_,
        "route_quant_fused is specialized for N=896, K=16");
    RuntimeCheck(scores.stride(0) % 4 == 0, "route_quant_fused: scores row stride must be a multiple of 4");

    // Quant half: shape/stride/alignment checks + byte-stride munging shared
    // with the standalone flat kernel.
    auto x_dtype = SymbolicDType{};
    TensorMatcher({M_, -1}).with_dtype<bf16_t, fp32_t>(x_dtype).with_device(device).with_strides({-1, 1}).verify(x);
    const auto quant_params =
        x_dtype.is_type<fp32_t>()
            ? build_quant_context<RouteQuantTraitT<fp32_t>, /*kMasked=*/false>(x, out_q, out_s).params
            : build_quant_context<RouteQuantTraitT<bf16_t>, /*kMasked=*/false>(x, out_q, out_s).params;
    RuntimeCheck(
        quant_params.hidden_size == kQuantHidden_, "route_quant_fused is specialized for a 3584-wide activation row");
    RuntimeCheck(
        quant_params.num_tokens == static_cast<uint32_t>(M_.unwrap()),
        "route_quant_fused: scores and activations must have the same token count");

    const auto M = static_cast<uint32_t>(M_.unwrap());
    if (M == 0) return;

    const auto params = RouteQuantFusedParams{
        .route =
            {scores.data_ptr(),
             static_cast<const fp32_t*>(bias.data_ptr()),
             static_cast<fp32_t*>(out_w.data_ptr()),
             static_cast<int32_t*>(out_i.data_ptr()),
             static_cast<int32_t*>(out_packed.data_ptr()),
             static_cast<int>(M),
             static_cast<long long>(scores.stride(0)),
             static_cast<long long>(out_w.stride(0)),
             static_cast<long long>(out_i.stride(0)),
             static_cast<long long>(out_packed.stride(0)),
             static_cast<float>(routed_scaling_factor),
             renormalize ? 1 : 0,
             apply_scale ? 1 : 0,
             /*sorted=*/0},
        .quant = quant_params,
    };

#define SGL_ROUTE_QUANT_LAUNCH(TS, TX)                                    \
  LaunchKernel(2 * M, LargeRouterRadixTrait::kBlockSize, device.unwrap()) \
      .enable_pdl(kUsePDL)(route_quant_fused_kernel<kUsePDL, TS, TX>, params)

    if (score_dtype.is_type<fp32_t>()) {
      if (x_dtype.is_type<fp32_t>()) {
        SGL_ROUTE_QUANT_LAUNCH(fp32_t, fp32_t);
      } else {
        SGL_ROUTE_QUANT_LAUNCH(fp32_t, bf16_t);
      }
    } else {
      if (x_dtype.is_type<fp32_t>()) {
        SGL_ROUTE_QUANT_LAUNCH(bf16_t, fp32_t);
      } else {
        SGL_ROUTE_QUANT_LAUNCH(bf16_t, bf16_t);
      }
    }
#undef SGL_ROUTE_QUANT_LAUNCH
  }
};

}  // namespace sglang
