// Fused all-reduce + mHC post (DeepSeek V4, HC = 4): `out = allreduce(partial)`, the post mix into the 4 residual
// streams `mhc_out`; with `kNorm` the pre-collapse + RMSNorm into `normalized`; with `kQuant` also MXFP8.
#include <sgl_kernel/distributed/push/epilogue/mhc_post.cuh>
#include <sgl_kernel/distributed/push/prologue/load.cuh>

#include "launch.cuh"
#include <cstdint>

namespace sglang::host::distributed::push {

// Attention post: the partial is the `partial` tensor, with collapse + RMSNorm.
template <uint32_t kWorldSize, uint32_t kHiddenSize, uint32_t kClusterSize, bool kUsePDL>
void all_reduce_mhc_post_combine_norm(
    CommunicatorRef comm,
    TensorView out,
    TensorView partial,
    TensorView mhc_out,
    TensorView residual,
    TensorView post,
    TensorView comb,
    TensorView pre,
    TensorView norm_weight,
    float eps,
    TensorView normalized) {
  using Launcher = PushLauncher<kWorldSize, kHiddenSize, kClusterSize, kUsePDL, /*kTwoShot=*/false>;
  using Pro = Load;
  using Epi = MHCPost<kHiddenSize, /*kNorm=*/true, /*kQuant=*/false>;
  CallShape shape;
  const auto pro = Pro::make_params(shape, partial, kHiddenSize);
  const auto epi =
      Epi::make_params(shape, out, mhc_out, residual, post, comb, pre, norm_weight, eps, normalized, {}, {});
  Launcher::template launch<EMPTY_FLAG, Pro, Epi>(comm, shape, pro, epi);
}

// MoE post: the partial is the MoE finalize (+ `shared_output` when given).
template <
    uint32_t kWorldSize,
    uint32_t kHiddenSize,
    uint32_t kClusterSize,
    bool kUsePDL,
    uint32_t kTopK,
    typename WeightT,
    bool kNorm,
    bool kQuant>
void moe_finalize_all_reduce_mhc_post(
    CommunicatorRef comm,
    TensorView out,
    TensorView gemm2_out,
    TensorView expanded_idx_to_permuted_idx,
    TensorView expert_weights,
    Optional<TensorView> shared_output,
    TensorView mhc_out,
    TensorView residual,
    TensorView post,
    TensorView comb,
    Optional<TensorView> pre,
    Optional<TensorView> norm_weight,
    float eps,
    Optional<TensorView> normalized,
    Optional<TensorView> quantized,
    Optional<TensorView> scales) {
  using Launcher = PushLauncher<kWorldSize, kHiddenSize, kClusterSize, kUsePDL, /*kTwoShot=*/false>;
  using Epi = MHCPost<kHiddenSize, kNorm, kQuant>;
  CallShape shape;
  const auto epi =
      Epi::make_params(shape, out, mhc_out, residual, post, comb, pre, norm_weight, eps, normalized, quantized, scales);
  launch_finalize<Launcher, EMPTY_FLAG, kHiddenSize, kTopK, WeightT, Epi>(
      comm, shape, gemm2_out, expanded_idx_to_permuted_idx, expert_weights, shared_output, epi);
}

}  // namespace sglang::host::distributed::push
