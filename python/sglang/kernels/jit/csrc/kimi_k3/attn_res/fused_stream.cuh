// The attention-residual aggregation of fused_tma.cuh beside the streamed pull
// all-reduce that produces its prefix rows (NVLinkComm::all_reduce_stream_pull,
// launched just before on the same stream).

#pragma once

#include "fused_tma.cuh"

namespace sglang {

// No thread waits for the all-reduce before reading: each prefix row is ordered
// by its stream marker (take_stream_row), so it is aggregated as soon as it is whole.
template <typename Trait, uint32_t kOccupancy>
__global__ void __launch_bounds__(Trait::kNumThreads, kOccupancy)
    attn_res_fused_stream_kernel(const __grid_constant__ AttnResTMAParams params) {
  extern __shared__ char smem_raw[];
  Trait::template forward<true>(params, reinterpret_cast<typename Trait::Smem*>(smem_raw));
}

template <int64_t kDim, uint32_t kMaxBankRows, uint32_t kChunkRows, uint32_t kOccupancy, uint32_t kConsumerRegs>
struct AttnResFusedStreamKernel {
  using Local = AttnResFusedTmaKernel<kDim, kMaxBankRows, kChunkRows, kOccupancy, kConsumerRegs>;
  template <std::size_t... I>
  static constexpr auto make_table(std::index_sequence<I...>) {
    return std::array<typename Local::KernelFn, kMaxBankRows + 1>{
        nullptr, attn_res_fused_stream_kernel<typename Local::template Trait<I + 1>, kOccupancy>...};
  }
  static constexpr auto kTable = make_table(std::make_index_sequence<kMaxBankRows>{});

  /// `stream` is this rank's replica of the all-reduce's output over `world_size`
  /// ranks, which runs on `producer_blocks` blocks. `prefix_out` receives its
  /// rows and may be the all-reduce's input.
  static void
  run(const tvm::ffi::TensorView stream,
      const tvm::ffi::TensorView prefix_out,
      const tvm::ffi::TensorView bank,
      const tvm::ffi::TensorView cw,
      const tvm::ffi::TensorView ow,
      const tvm::ffi::TensorView out,
      int64_t nvb,
      double eps,
      bool write_prefix,
      int64_t world_size,
      int64_t producer_blocks) {
    using namespace host;
    auto local = Local::local_launch(prefix_out, bank, cw, ow, out, nvb, eps, write_prefix);
    TensorMatcher({static_cast<int64_t>(local.params.num_tokens), kDim})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>()
        .verify(stream);
    RuntimeCheck(world_size >= 1 && producer_blocks >= 0, "attn_res_fused_stream: invalid world or producer blocks");
    local.params.prefix_out = static_cast<bf16_t*>(prefix_out.data_ptr());
    local.params.prefix_stream = static_cast<bf16_t*>(stream.data_ptr());
    local.params.world_size = static_cast<uint32_t>(world_size);

    [[maybe_unused]] static const bool attrs_set = [] {
      for (uint32_t i = 1; i <= kMaxBankRows; ++i) {
        RuntimeDeviceCheck(
            cudaFuncSetAttribute(kTable[i], cudaFuncAttributeMaxDynamicSharedMemorySize, Local::kSmemBytes));
      }
      return true;
    }();
    // The all-reduce keeps its SMs: a block that waited for one of them would
    // start when the all-reduce is over.
    const auto free_sm = runtime::get_sm_count(local.device.device_id) - producer_blocks;
    Local::launch(kTable[nvb], std::max<int64_t>(free_sm, 1), local);
  }
};

}  // namespace sglang
