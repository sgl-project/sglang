// Fused all-reduce + RMSNorm on a row prefix (Kimi K3): `out = rmsnorm(allreduce(partial))` on the first
// `num_norm_rows` rows, the rest reduced only; the dependent grid is released right after the push.
#include <sgl_kernel/distributed/push/epilogue/rms_norm.cuh>
#include <sgl_kernel/distributed/push/prologue/load.cuh>

#include "launch.cuh"
#include <cstdint>

namespace sglang::host::distributed::push {

// In place on `x` (the latent rows, or the flat `[latent, shared]` buffer with the norm on the latent rows).
template <uint32_t kWorldSize, uint32_t kHiddenSize, uint32_t kClusterSize, bool kUsePDL>
void all_reduce_norm_prefix(
    CommunicatorRef comm, TensorView x, TensorView norm_weight, float eps, int64_t num_norm_rows) {
  using Launcher = PushLauncher<kWorldSize, kHiddenSize, kClusterSize, kUsePDL, /*kTwoShot=*/false>;
  using Pro = Load;
  using Epi = RMSNorm<kHiddenSize, /*kNormPrefix=*/true>;
  CallShape shape;
  const auto pro = Pro::make_params(shape, x, kHiddenSize);
  const auto epi = Epi::make_params(shape, x, norm_weight, eps, num_norm_rows);
  Launcher::template launch<ONE_WAVE | TRIGGER_AFTER_PUSH, Pro, Epi>(comm, shape, pro, epi);
}

// The partial is the MoE finalize; the norm on every row.
template <
    uint32_t kWorldSize,
    uint32_t kHiddenSize,
    uint32_t kClusterSize,
    bool kUsePDL,
    uint32_t kTopK,
    typename WeightT>
void moe_finalize_all_reduce_norm(
    CommunicatorRef comm,
    TensorView out,
    TensorView gemm2_out,
    TensorView expanded_idx_to_permuted_idx,
    TensorView expert_weights,
    TensorView norm_weight,
    float eps) {
  using Launcher = PushLauncher<kWorldSize, kHiddenSize, kClusterSize, kUsePDL, /*kTwoShot=*/false>;
  using Pro = FinalizeSum<kHiddenSize, kTopK, WeightT, SharedExpertMode::NONE>;
  using Epi = RMSNorm<kHiddenSize>;
  CallShape shape;
  const auto pro = Pro::make_params(shape, gemm2_out, expanded_idx_to_permuted_idx, expert_weights, {}, {});
  const auto epi = Epi::make_params(shape, out, norm_weight, eps);
  Launcher::template launch<ONE_WAVE | TRIGGER_AFTER_PUSH, Pro, Epi>(comm, shape, pro, epi);
}

}  // namespace sglang::host::distributed::push
