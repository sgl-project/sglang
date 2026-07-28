#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include "cute/tensor.hpp"
#include "es_sm100_mxfp8_blockscaled_moe_group_gemm_functor.cuh"
#include "es_sm100_mxfp8_blockscaled_moe_group_gemm_traits.cuh"

namespace expert_specialization {

using namespace host;

template <typename GemmTraits>
void es_sm100_mxfp8_blockscaled_moe_group_gemm_pre_compute(
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView sfb,
    tvm::ffi::TensorView expert_offsets,
    tvm::ffi::TensorView blockscale_offsets,
    tvm::ffi::TensorView b_ptrs,
    tvm::ffi::TensorView sfb_ptrs,
    tvm::ffi::TensorView d,
    tvm::ffi::TensorView d_ptrs,
    int num_experts,
    int m,
    int k,
    cudaStream_t stream) {
  using OffsetFunctor = Sm100Mxfp8BlockScaledMoeGroupGemmOffsetFunctor<GemmTraits>;
  using ElementB = typename OffsetFunctor::ElementB;
  using ElementSF = typename OffsetFunctor::ElementSF;
  using ElementD = typename OffsetFunctor::ElementD;

  host::RuntimeCheck(num_experts <= 1024, "num_experts more than 1024");
  OffsetFunctor offset_functor(
      reinterpret_cast<int*>(expert_offsets.data_ptr()),
      reinterpret_cast<int*>(blockscale_offsets.data_ptr()),
      reinterpret_cast<ElementB*>(b.data_ptr()),
      reinterpret_cast<ElementSF*>(sfb.data_ptr()),
      reinterpret_cast<ElementD*>(d.data_ptr()),
      reinterpret_cast<ElementB**>(b_ptrs.data_ptr()),
      reinterpret_cast<ElementSF**>(sfb_ptrs.data_ptr()),
      reinterpret_cast<ElementD**>(d_ptrs.data_ptr()));

  sm100_mxfp8_blockscaled_moe_group_gemm_pre_compute_kernel<<<1, num_experts, 0, stream>>>(offset_functor, m, k);
}

template <typename GemmTraits>
void es_sm100_mxfp8_blockscaled_moe_group_gemm(
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView sfa,
    tvm::ffi::TensorView tokens_per_expert,
    tvm::ffi::TensorView b_ptrs,
    tvm::ffi::TensorView sfb_ptrs,
    tvm::ffi::TensorView d_ptrs,
    tvm::ffi::TensorView workspace,
    int num_experts,
    int m,
    int n,
    int k,
    int device_id,
    int sm_count,
    cudaStream_t stream) {
  using Gemm = typename GemmTraits::Gemm;
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementSF = typename GemmTraits::ElementSF;
  using ElementD = typename GemmTraits::ElementD;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count = sm_count;
  hw_info.cluster_shape = GemmTraits::MMAConfig::preferred_cluster;
  hw_info.cluster_shape_fallback = GemmTraits::MMAConfig::fallback_cluster;

  typename Gemm::Arguments arguments = {
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {m, n, k, num_experts, reinterpret_cast<int*>(tokens_per_expert.data_ptr())},
      {reinterpret_cast<const ElementA*>(a.data_ptr()),
       reinterpret_cast<const ElementB**>(b_ptrs.data_ptr()),
       reinterpret_cast<const ElementSF*>(sfa.data_ptr()),
       reinterpret_cast<const ElementSF**>(sfb_ptrs.data_ptr())},
      {{}, nullptr, nullptr, reinterpret_cast<ElementD**>(d_ptrs.data_ptr()), nullptr},
      hw_info,
      {}  // Scheduler
  };

  Gemm gemm;

  auto can_implement_status = gemm.can_implement(arguments);
  host::RuntimeCheck(can_implement_status == cutlass::Status::kSuccess, "Can not implement MoE Group GEMM");

  auto status = gemm.initialize(arguments, reinterpret_cast<uint8_t*>(workspace.data_ptr()), stream);
  host::RuntimeCheck(status == cutlass::Status::kSuccess, "Failed to initialize MoE Group GEMM");

  status = gemm.run(stream, nullptr);
  host::RuntimeCheck(status == cutlass::Status::kSuccess, "Failed to run MoE Group GEMM");
}

template <typename DType>  // CUTLASS dtype
void es_sm100_mxfp8_blockscaled_moe_group_gemm_dispatch_dtype(
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView sfa,
    tvm::ffi::TensorView sfb,
    tvm::ffi::TensorView expert_offsets,
    tvm::ffi::TensorView blockscale_offsets,
    tvm::ffi::TensorView tokens_per_expert,
    tvm::ffi::TensorView b_ptrs,
    tvm::ffi::TensorView sfb_ptrs,
    tvm::ffi::TensorView d,
    tvm::ffi::TensorView d_ptrs,
    tvm::ffi::TensorView workspace,
    int num_experts,
    int m,
    int n,
    int k,
    int device_id,
    int sm_count,
    cudaStream_t stream) {
  using GemmTraits = ExpertSpecializationSm100MXFP8BlockscaledMoeGroupGemmTraits<MMA2SMConfig, DType>;

  es_sm100_mxfp8_blockscaled_moe_group_gemm_pre_compute<GemmTraits>(
      b, sfb, expert_offsets, blockscale_offsets, b_ptrs, sfb_ptrs, d, d_ptrs, num_experts, m, k, stream);
  es_sm100_mxfp8_blockscaled_moe_group_gemm<GemmTraits>(
      a,
      sfa,
      tokens_per_expert,
      b_ptrs,
      sfb_ptrs,
      d_ptrs,
      workspace,
      num_experts,
      m,
      n,
      k,
      device_id,
      sm_count,
      stream);
}

}  // namespace expert_specialization

template <typename DType>
struct EsSm100MXFP8BlockscaledMoeGroupGemm {
  static void
  run(tvm::ffi::TensorView a,
      tvm::ffi::TensorView b,
      tvm::ffi::TensorView sfa,
      tvm::ffi::TensorView sfb,
      tvm::ffi::TensorView expert_offsets,
      tvm::ffi::TensorView blockscale_offsets,
      tvm::ffi::TensorView tokens_per_expert,
      tvm::ffi::TensorView b_ptrs,
      tvm::ffi::TensorView sfb_ptrs,
      tvm::ffi::TensorView d,
      tvm::ffi::TensorView d_ptrs,
      tvm::ffi::TensorView workspace) {
    using namespace host;
    auto num_tokens = SymbolicSize{"num_tokens"};
    auto num_sf_tokens = SymbolicSize{"num_sf_tokens"};
    auto hidden_size = SymbolicSize{"hidden_size"};
    auto num_experts = SymbolicSize{"num_experts"};
    auto M = SymbolicSize{"M"};
    auto K = SymbolicSize{"K"};
    auto M_SF = SymbolicSize{"M_SF"};
    auto K_SF = SymbolicSize{"K_SF"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({num_experts, M, K}).with_dtype<fp8_e4m3_t>().with_device(device).verify(a);
    TensorMatcher({num_tokens, K}).with_dtype<fp8_e4m3_t>().with_device(device).verify(b);
    TensorMatcher({num_experts, M_SF, K_SF}).with_dtype<uint8_t>().with_device(device).verify(sfa);
    TensorMatcher({num_sf_tokens, K_SF}).with_dtype<uint8_t>().with_device(device).verify(sfb);
    RuntimeCheck(K.unwrap() % 128 == 0, "K should align 128");
    RuntimeCheck(K.unwrap() / 32 == K_SF.unwrap(), "K dimension mismatch");

    TensorMatcher({num_experts}).with_dtype<int>().with_device(device).verify(expert_offsets);
    TensorMatcher({num_experts}).with_dtype<int>().with_device(device).verify(blockscale_offsets);
    TensorMatcher({num_experts}).with_dtype<int>().with_device(device).verify(tokens_per_expert);
    TensorMatcher({num_experts}).with_dtype<int64_t>().with_device(device).verify(b_ptrs);
    TensorMatcher({num_experts}).with_dtype<int64_t>().with_device(device).verify(sfb_ptrs);
    TensorMatcher({num_experts}).with_dtype<int64_t>().with_device(device).verify(d_ptrs);
    // Check output
    TensorMatcher({num_tokens, M}).with_strides({M, 1}).with_dtype<DType>().with_device(device).verify(d);

    cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
    int device_id = device.unwrap().device_id;

    if constexpr (std::is_same_v<DType, bf16_t> || std::is_same_v<DType, fp16_t>) {
      using CUTLASS_DTYPE = std::conditional_t<std::is_same_v<DType, bf16_t>, cutlass::bfloat16_t, cutlass::half_t>;
      expert_specialization::es_sm100_mxfp8_blockscaled_moe_group_gemm_dispatch_dtype<CUTLASS_DTYPE>(
          a,
          b,
          sfa,
          sfb,
          expert_offsets,
          blockscale_offsets,
          tokens_per_expert,
          b_ptrs,
          sfb_ptrs,
          d,
          d_ptrs,
          workspace,
          static_cast<int>(num_experts.unwrap()),
          static_cast<int>(M.unwrap()),
          static_cast<int>(num_tokens.unwrap()),
          static_cast<int>(K.unwrap()),
          device_id,
          static_cast<int>(runtime::get_sm_count(device_id)),
          stream);
    } else {
      Panic("Unsupported dtype");
    }
  }
};
