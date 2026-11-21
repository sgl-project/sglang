// Fused all-reduce + sandwich norm (Beam 1.0): `combined = bf16(bf16(rmsnorm(x) * post_norm_weight * x_scale) +
// bf16(residual * residual_scale))` with `x = allreduce(partial)`; with `next_norm_weight` also
// `normalized = rmsnorm(combined) * next_norm_weight`. The push is multicast when the plane has a multicast VA.
#include <sgl_kernel/distributed/push/epilogue/sandwich_norm.cuh>
#include <sgl_kernel/distributed/push/prologue/load.cuh>
#include <sgl_kernel/distributed/push/prologue/topk_sum.cuh>

#include "launch.cuh"
#include <cstdint>

namespace sglang::host::distributed::push {

// `kNextNorm` from whether the call passes `next_norm_weight`.
template <typename Launcher, uint32_t kHiddenSize, Prologue Pro>
void launch_sandwich_norm(
    CommunicatorRef comm,
    CallShape& shape,
    const typename Pro::Params& pro,
    TensorView residual,
    TensorView post_norm_weight,
    Optional<TensorView> next_norm_weight,
    TensorView combined,
    Optional<TensorView> normalized,
    float eps,
    float x_scale,
    float residual_scale) {
  const auto launch = [&]<typename Epi>(Epi*) {
    const auto epi = Epi::make_params(
        shape, residual, post_norm_weight, next_norm_weight, combined, normalized, eps, x_scale, residual_scale);
    Launcher::template launch_multicast_if_available<ONE_WAVE, Pro, Epi>(comm, shape, pro, epi);
  };

  if (next_norm_weight.has_value()) {
    using Epi = SandwichNorm<kHiddenSize, /*kNextNorm=*/true>;
    launch(static_cast<Epi*>(nullptr));
  } else {
    using Epi = SandwichNorm<kHiddenSize, /*kNextNorm=*/false>;
    launch(static_cast<Epi*>(nullptr));
  }
}

// 2-D branch: `partial [T, H]` is the partial.
template <uint32_t kWorldSize, uint32_t kHiddenSize, uint32_t kClusterSize, bool kUsePDL>
void all_reduce_sandwich_norm(
    CommunicatorRef comm,
    TensorView partial,
    TensorView residual,
    TensorView post_norm_weight,
    Optional<TensorView> next_norm_weight,
    TensorView combined,
    Optional<TensorView> normalized,
    float eps,
    float x_scale,
    float residual_scale) {
  using Launcher = PushLauncher<kWorldSize, kHiddenSize, kClusterSize, kUsePDL, /*kTwoShot=*/false>;
  using Pro = Load;
  CallShape shape;
  const auto pro = Pro::make_params(shape, partial, kHiddenSize);
  launch_sandwich_norm<Launcher, kHiddenSize, Pro>(
      comm,
      shape,
      pro,
      residual,
      post_norm_weight,
      next_norm_weight,
      combined,
      normalized,
      eps,
      x_scale,
      residual_scale);
}

// 3-D branch: the partial is `routed_scale * sum_k partial[:, k]` of `partial [T, kTopK, H]` (expert rows).
template <uint32_t kWorldSize, uint32_t kHiddenSize, uint32_t kClusterSize, bool kUsePDL, uint32_t kTopK>
void topk_sum_all_reduce_sandwich_norm(
    CommunicatorRef comm,
    TensorView partial,
    TensorView residual,
    TensorView post_norm_weight,
    Optional<TensorView> next_norm_weight,
    TensorView combined,
    Optional<TensorView> normalized,
    float eps,
    float x_scale,
    float residual_scale,
    float routed_scale) {
  using Launcher = PushLauncher<kWorldSize, kHiddenSize, kClusterSize, kUsePDL, /*kTwoShot=*/false>;
  using Pro = TopKSum<kHiddenSize, kTopK>;
  CallShape shape;
  const auto pro = Pro::make_params(shape, partial, routed_scale);
  launch_sandwich_norm<Launcher, kHiddenSize, Pro>(
      comm,
      shape,
      pro,
      residual,
      post_norm_weight,
      next_norm_weight,
      combined,
      normalized,
      eps,
      x_scale,
      residual_scale);
}

}  // namespace sglang::host::distributed::push
