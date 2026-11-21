// Fused all-reduce + residual add + RMSNorm (MiniMax M3), 1-shot or 2-shot:
// `residual_out = bf16(allreduce(partial) + residual)`, `out = rmsnorm(residual_out)`.
#include <sgl_kernel/distributed/push/epilogue/residual_rms_norm.cuh>
#include <sgl_kernel/distributed/push/prologue/load.cuh>

#include "launch.cuh"
#include <cstdint>

namespace sglang::host::distributed::push {

// The partial is the `partial` tensor.
template <uint32_t kWorldSize, uint32_t kHiddenSize, uint32_t kClusterSize, bool kUsePDL, bool kTwoShot>
void all_reduce_residual_norm(
    CommunicatorRef comm,
    TensorView out,
    TensorView partial,
    TensorView residual,
    TensorView residual_out,
    TensorView norm_weight,
    float eps) {
  using Launcher = PushLauncher<kWorldSize, kHiddenSize, kClusterSize, kUsePDL, kTwoShot>;
  using Pro = Load;
  using Epi = ResidualRMSNorm<kHiddenSize>;
  CallShape shape;
  const auto pro = Pro::make_params(shape, partial, kHiddenSize);
  const auto epi = Epi::make_params(shape, out, residual, residual_out, norm_weight, eps);
  Launcher::template launch<ONE_WAVE, Pro, Epi>(comm, shape, pro, epi);
}

// The partial is the MoE finalize (+ `shared_output` when given).
template <
    uint32_t kWorldSize,
    uint32_t kHiddenSize,
    uint32_t kClusterSize,
    bool kUsePDL,
    bool kTwoShot,
    uint32_t kTopK,
    typename WeightT>
void moe_finalize_all_reduce_residual_norm(
    CommunicatorRef comm,
    TensorView out,
    TensorView gemm2_out,
    TensorView expanded_idx_to_permuted_idx,
    TensorView expert_weights,
    Optional<TensorView> shared_output,
    TensorView residual,
    TensorView residual_out,
    TensorView norm_weight,
    float eps) {
  using Launcher = PushLauncher<kWorldSize, kHiddenSize, kClusterSize, kUsePDL, kTwoShot>;
  using Epi = ResidualRMSNorm<kHiddenSize>;
  CallShape shape;
  const auto epi = Epi::make_params(shape, out, residual, residual_out, norm_weight, eps);
  launch_finalize<Launcher, ONE_WAVE, kHiddenSize, kTopK, WeightT, Epi>(
      comm, shape, gemm2_out, expanded_idx_to_permuted_idx, expert_weights, shared_output, epi);
}

}  // namespace sglang::host::distributed::push
