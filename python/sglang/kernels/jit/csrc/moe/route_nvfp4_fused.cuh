// Opt-in K3 NVFP4 routing + quantization.
// Routing reuses route_radix unchanged; quant uses TRTLLM's NVFP4 conversion.
#include <sgl_kernel/vec.cuh>
#ifndef FLT_MAX
#define FLT_MAX __FLT_MAX__
#endif
#include <tensorrt_llm/kernels/quantization_utils.cuh>
#include "route_radix.cuh"

namespace sglang {

static_assert(LargeRouterRadixTrait::kBlockSize == 224);

struct RouteNvfp4FusedParams {
  RouteRadixParams route;
  const void* x;
  uint64_t* q;
  uint8_t* sf;
  const float* global_scale;
  int64_t x_stride;
};

template <bool kUsePDL, typename TScore, typename TX>
__global__ __launch_bounds__(LargeRouterRadixTrait::kBlockSize)  //
    void route_nvfp4_fused_kernel(const __grid_constant__ RouteNvfp4FusedParams params) {
  const auto M = static_cast<uint32_t>(params.route.M);
  if (blockIdx.x < M) {
    __shared__ typename LargeRouterRadixTrait::Smem smem;
    route_radix_block<kUsePDL, TScore>(params.route, smem);
  } else {
    device::PDLWaitPrimary<kUsePDL>();
    const uint32_t row = blockIdx.x - M;
    const uint32_t group = threadIdx.x;  // 224 groups x 16 = 3584
    tensorrt_llm::kernels::PackedVec<__nv_bfloat16, 16> values;
    auto* dst = reinterpret_cast<__nv_bfloat16*>(&values);
    constexpr int kLoadElems = 32 / sizeof(TX);
#pragma unroll
    for (int base = 0; base < 16; base += kLoadElems) {
      device::AlignedVector<TX, kLoadElems> input;
      input.load(static_cast<const TX*>(params.x) + row * params.x_stride + group * 16 + base);
#pragma unroll
      for (int i = 0; i < kLoadElems; ++i) {
        // Preserve existing FP32->BF16 rounding; each load is at most32 bytes.
        if constexpr (std::is_same_v<TX, fp32_t>) dst[base+i] = __float2bfloat16_rn(input[i]);
        else dst[base+i] = static_cast<__nv_bfloat16>(input[i]);
      }
    }
    const uint32_t offset = row * 224 + group;
    params.q[offset] = tensorrt_llm::kernels::cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, 16, false>(
        values, *params.global_scale, params.sf + offset);
    device::PDLTriggerSecondary<kUsePDL>();
  }
}

template <bool kUsePDL>
struct RouteNvfp4FusedKernel {
  static void
  run(const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      const tvm::ffi::TensorView out_packed,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView out_q,
      const tvm::ffi::TensorView out_s,
      const tvm::ffi::TensorView global_scale,
      int64_t topk,
      double routed_scaling_factor,
      bool renormalize,
      bool apply_scale) {
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
        "route_nvfp4_fused is specialized for N=896, K=16");
    RuntimeCheck(scores.stride(0) % 4 == 0, "route_nvfp4_fused: scores row stride must be a multiple of 4");

    auto x_dtype = SymbolicDType{};
    TensorMatcher({M_, 3584}).with_dtype<bf16_t, fp32_t>(x_dtype).with_device(device).with_strides({-1, 1}).verify(x);
    TensorMatcher({M_, 1792}).with_dtype<uint8_t>().with_device(device).verify(out_q);
    TensorMatcher({M_, 224}).with_dtype<uint8_t>().with_device(device).verify(out_s);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device(device).verify(global_scale);
    RuntimeCheck(x.stride(0) % 16 == 0, "NVFP4 activation stride must be 16-aligned");
    RuntimeCheck(reinterpret_cast<uintptr_t>(x.data_ptr()) % 32 == 0, "NVFP4 activation pointer must be 32B-aligned");
    RuntimeCheck(M_.unwrap() <= 64, "Experimental fused NVFP4 route supports at most64 rows");

    const auto M = static_cast<uint32_t>(M_.unwrap());
    if (M == 0) return;

    const auto params = RouteNvfp4FusedParams{
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
        .x = x.data_ptr(),
        .q = static_cast<uint64_t*>(out_q.data_ptr()),
        .sf = static_cast<uint8_t*>(out_s.data_ptr()),
        .global_scale = static_cast<const float*>(global_scale.data_ptr()),
        .x_stride = x.stride(0),
    };

#define K3_ROUTE_NVFP4_LAUNCH(TS, TX) \
    LaunchKernel(2 * M, LargeRouterRadixTrait::kBlockSize, device.unwrap()) \
        .enable_pdl(kUsePDL)(route_nvfp4_fused_kernel<kUsePDL, TS, TX>, params)
    if (score_dtype.is_type<fp32_t>()) {
      if (x_dtype.is_type<fp32_t>()) { K3_ROUTE_NVFP4_LAUNCH(fp32_t, fp32_t); }
      else { K3_ROUTE_NVFP4_LAUNCH(fp32_t, bf16_t); }
    } else {
      if (x_dtype.is_type<fp32_t>()) { K3_ROUTE_NVFP4_LAUNCH(bf16_t, fp32_t); }
      else { K3_ROUTE_NVFP4_LAUNCH(bf16_t, bf16_t); }
    }
#undef K3_ROUTE_NVFP4_LAUNCH
  }
};

}  // namespace sglang
