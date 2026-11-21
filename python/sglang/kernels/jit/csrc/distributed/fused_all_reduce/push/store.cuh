// Fused MoE finalize with a gated shared expert + all-reduce (Qwen4 decode):
// `out = allreduce(bf16(bf16(shared_gate * shared_output) + bf16(routed)))`.
#include <sgl_kernel/distributed/push/epilogue/store.cuh>
#include <sgl_kernel/distributed/push/prologue/finalize.cuh>

#include "launch.cuh"
#include <cstdint>

namespace sglang::host::distributed::push {

template <
    uint32_t kWorldSize,
    uint32_t kHiddenSize,
    uint32_t kClusterSize,
    bool kUsePDL,
    uint32_t kTopK,
    typename WeightT>
void moe_finalize_shared_gate_all_reduce(
    CommunicatorRef comm,
    TensorView out,
    TensorView gemm2_out,
    TensorView expanded_idx_to_permuted_idx,
    TensorView expert_weights,
    TensorView shared_output,
    TensorView shared_gate) {
  using Launcher = PushLauncher<kWorldSize, kHiddenSize, kClusterSize, kUsePDL, /*kTwoShot=*/false>;
  using Pro = FinalizeSum<kHiddenSize, kTopK, WeightT, SharedExpertMode::GATED>;
  using Epi = Store<kHiddenSize>;
  CallShape shape;
  const auto epi = Epi::make_params(shape, out);
  const auto pro =
      Pro::make_params(shape, gemm2_out, expanded_idx_to_permuted_idx, expert_weights, shared_output, shared_gate);
  Launcher::template launch<ONE_WAVE, Pro, Epi>(comm, shape, pro, epi);
}

}  // namespace sglang::host::distributed::push
