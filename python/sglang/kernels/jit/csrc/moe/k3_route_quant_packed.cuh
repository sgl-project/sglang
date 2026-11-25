// K3 routing + MXFP8 activation quant in one launch, for DeepGEMM's MegaMoE.
#include "route_quant_fused.cuh"

namespace sglang {

struct K3RouteQuantPackedParams {
  RouteRadixParams route;
  QuantKernelParams quant;
  const int32_t* __restrict__ num_valid_ptr;
  int32_t num_valid_value;
  // PDL trigger of the next launch: 0 = end of CTA, 1 = after this launch's own wait, 2 = before it
  int32_t early_trigger;
  int32_t flag_w;
  fp32_t* __restrict__ out_wbuf;
};

template <bool kUsePDL, typename TScore, typename TX>
__global__ __launch_bounds__(LargeRouterRadixTrait::kBlockSize)  //
    void k3_route_quant_packed_kernel(const __grid_constant__ K3RouteQuantPackedParams params) {
  using T = LargeRouterRadixTrait;
  const auto M = static_cast<uint32_t>(params.route.M);
  const uint32_t tx = threadIdx.x;
  if (blockIdx.x < M) {
    __shared__ typename T::Smem smem_route;
    if (params.early_trigger == 2) device::PDLTriggerSecondary<kUsePDL>();
    device::PDLWaitPrimary<kUsePDL>();
    if (params.early_trigger == 1) device::PDLTriggerSecondary<kUsePDL>();
    const int32_t num_valid = params.num_valid_ptr != nullptr ? params.num_valid_ptr[0] : params.num_valid_value;
    route_radix_block</*kUsePDL=*/false, TScore>(params.route, smem_route);
    // sorted == 0: thread tx < kTopK wrote slot tx itself, so this store is ordered after it
    const uint32_t row = blockIdx.x;
    if (tx < T::kTopK && static_cast<int32_t>(row) >= num_valid) {
      params.route.out_i[row * params.route.out_i_stride + tx] = -1;
      params.route.out_packed[row * params.route.out_packed_stride + tx] = -1;
    } else if (tx < T::kTopK && params.flag_w) {
      const int32_t id = params.route.out_i[row * params.route.out_i_stride + tx];
      params.out_wbuf[row * T::kTopK + tx] = params.route.out_w[row * params.route.out_w_stride + tx];
      params.route.out_packed[row * params.route.out_packed_stride + tx] = ((id & 0x3fff) << 16) | 0x40000000;
    }
    if (params.early_trigger == 0) device::PDLTriggerSecondary<kUsePDL>();
  } else {
    if (params.early_trigger == 2) device::PDLTriggerSecondary<kUsePDL>();
    device::PDLWaitPrimary<kUsePDL>();
    if (params.early_trigger == 1) device::PDLTriggerSecondary<kUsePDL>();
    using Q = RouteQuantTraitT<TX>;
    Q::run(params.quant, /*expert_idx=*/0, blockIdx.x - M, threadIdx.x / Q::kNumLanes, threadIdx.x % Q::kNumLanes);
    if (params.early_trigger == 0) device::PDLTriggerSecondary<kUsePDL>();
  }
}

template <bool kUsePDL>
struct K3RouteQuantPackedKernel {
  static void
  run(const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      const tvm::ffi::TensorView out_packed,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView out_q,
      const tvm::ffi::TensorView out_s,
      const tvm::ffi::Optional<tvm::ffi::TensorView> num_valid,
      int64_t topk,
      double routed_scaling_factor,
      bool renormalize,
      bool apply_scale,
      int64_t early_trigger,
      const tvm::ffi::Optional<tvm::ffi::TensorView> out_wbuf) {
    using namespace host;

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
        "k3_route_quant_packed is specialized for N=896, K=16");
    RuntimeCheck(scores.stride(0) % 4 == 0, "k3_route_quant_packed: scores row stride must be a multiple of 4");

    const int32_t* num_valid_ptr = nullptr;
    if (num_valid.has_value()) {
      const auto& nv = num_valid.value();
      RuntimeCheck(
          nv.numel() == 1 && nv.device().device_type == kDLCUDA && is_type<int32_t>(nv.dtype()),
          "k3_route_quant_packed: num_valid must be a CUDA int32 scalar");
      num_valid_ptr = static_cast<const int32_t*>(nv.data_ptr());
    }

    auto x_dtype = SymbolicDType{};
    TensorMatcher({M_, -1}).with_dtype<bf16_t, fp32_t>(x_dtype).with_device(device).with_strides({-1, 1}).verify(x);
    const auto quant_params =
        x_dtype.is_type<fp32_t>()
            ? build_quant_context<RouteQuantTraitT<fp32_t>, /*kMasked=*/false>(x, out_q, out_s).params
            : build_quant_context<RouteQuantTraitT<bf16_t>, /*kMasked=*/false>(x, out_q, out_s).params;
    RuntimeCheck(quant_params.hidden_size == kQuantHidden_, "k3_route_quant_packed: activation row must be 3584 wide");
    RuntimeCheck(
        quant_params.num_tokens == static_cast<uint32_t>(M_.unwrap()),
        "k3_route_quant_packed: scores and activations must have the same token count");

    const auto M = static_cast<uint32_t>(M_.unwrap());
    if (M == 0) return;
    fp32_t* out_wbuf_ptr = nullptr;
    if (out_wbuf.has_value()) {
      const auto& wb = out_wbuf.value();
      RuntimeCheck(
          wb.ndim() == 2 && wb.size(0) >= static_cast<int64_t>(M) && wb.size(1) == kTopK_ && wb.stride(1) == 1 &&
              wb.stride(0) == kTopK_ && wb.device().device_type == kDLCUDA && is_type<fp32_t>(wb.dtype()),
          "k3_route_quant_packed: out_wbuf must be a contiguous CUDA fp32 [rows >= M, 16] tensor");
      out_wbuf_ptr = static_cast<fp32_t*>(wb.data_ptr());
    }

    const auto params = K3RouteQuantPackedParams{
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
        .num_valid_ptr = num_valid_ptr,
        .num_valid_value = static_cast<int32_t>(M),
        .early_trigger = static_cast<int32_t>(early_trigger),
        .flag_w = out_wbuf_ptr != nullptr ? 1 : 0,
        .out_wbuf = out_wbuf_ptr,
    };

#define K3_ROUTE_QUANT_PACKED_LAUNCH(TS, TX)                                \
  LaunchKernel(2 * M, LargeRouterRadixTrait::kBlockSize, device.unwrap()) \
      .enable_pdl(kUsePDL)(k3_route_quant_packed_kernel<kUsePDL, TS, TX>, params)

    if (score_dtype.is_type<fp32_t>()) {
      if (x_dtype.is_type<fp32_t>()) {
        K3_ROUTE_QUANT_PACKED_LAUNCH(fp32_t, fp32_t);
      } else {
        K3_ROUTE_QUANT_PACKED_LAUNCH(fp32_t, bf16_t);
      }
    } else {
      if (x_dtype.is_type<fp32_t>()) {
        K3_ROUTE_QUANT_PACKED_LAUNCH(bf16_t, fp32_t);
      } else {
        K3_ROUTE_QUANT_PACKED_LAUNCH(bf16_t, bf16_t);
      }
    }
#undef K3_ROUTE_QUANT_PACKED_LAUNCH
  }
};

}  // namespace sglang
