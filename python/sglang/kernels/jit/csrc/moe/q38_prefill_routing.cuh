// Qwen3.8 MoE routing at any row count (SGLANG_QWEN_MOE_PREFILL_CUTE_RUBIN): each row's top-10 of 512 BF16 router
// logits and its RenormalizeNaive BF16 weights, written as ids [M, 10] int32 and weights [M, 10] bf16, for a routed
// MoE that takes precomputed routing (the CuTe-DSL prefill path) in place of trtllm-gen's internal routing.
//
// One warp per row runs q38_fused_routing::select_row as is (q38_fused_routing.cuh, H502): the same keys
// (FlashInfer 0.7.2 TopKRedType order: largest first, ties to the smaller expert id, column k holds the k-th
// largest) and the same calcSoftmax expressions over the 10 selected logits. This TU builds with --use_fast_math,
// as FlashInfer's MoE module and the decode routing kernel do, so the softmax lowers identically. Only the per-row
// part is used: no routing metadata, no activation quant, no row cap.

#pragma once

#include "q38_fused_routing.cuh"

namespace sglang {

namespace q38_prefill_routing {

inline constexpr int kWarpsPerCta = 8;
inline constexpr int kThreads = kWarpsPerCta * 32;

}  // namespace q38_prefill_routing

__global__ __launch_bounds__(q38_prefill_routing::kThreads)  //
    void q38_prefill_routing_kernel(const __grid_constant__ q38_fused_routing::Params params) {
  namespace cg = cooperative_groups;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * q38_prefill_routing::kWarpsPerCta + threadIdx.x / 32;
  if (row >= params.num_tokens) return;  // warp-uniform: whole warps leave, so select_row's full-mask syncs hold
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  q38_fused_routing::select_row(params, warp, static_cast<int>(row), static_cast<int>(threadIdx.x % 32));
}

struct Q38PrefillRoutingKernel {
  static void run(const tvm::ffi::TensorView logits, const tvm::ffi::TensorView ids, const tvm::ffi::TensorView weights) {
    using namespace host;
    namespace R = q38_fused_routing;
    namespace P = q38_prefill_routing;

    auto M_ = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M_, R::kNumExperts}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(logits);
    TensorMatcher({M_, R::kTopK}).with_strides({R::kTopK, 1}).with_dtype<int32_t>().with_device(device).verify(ids);
    TensorMatcher({M_, R::kTopK}).with_strides({R::kTopK, 1}).with_dtype<bf16_t>().with_device(device).verify(weights);

    const int64_t M = M_.unwrap();
    RuntimeCheck(M >= 1 && M <= (int64_t{1} << 26), "q38_prefill_routing: 1 to 2^26 rows");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0 && logits.stride(0) % 8 == 0,
        "q38_prefill_routing: logits rows must be 16-byte aligned");

    R::Params params{};
    params.logits = static_cast<const bf16_t*>(logits.data_ptr());
    params.ids = static_cast<int32_t*>(ids.data_ptr());
    params.weights = static_cast<bf16_t*>(weights.data_ptr());
    params.logits_stride = logits.stride(0);
    params.num_tokens = static_cast<int32_t>(M);

    const auto grid = static_cast<uint32_t>(div_ceil(M, int64_t{P::kWarpsPerCta}));
    LaunchKernel(grid, P::kThreads, device.unwrap())(q38_prefill_routing_kernel, params);
  }
};

}  // namespace sglang
