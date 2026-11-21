// Fused NVLS pull all-reduce + post-norm + residual add (Gemma 4) on a row-sharded residual stream, rounded as Hugging
// Face's layer: each rank reduces its shard `[row0, row0 + rows)` of the partials, updates its rows of the residual and
// multicasts the result to every rank: the next block's FP8 input (with the next norm) or the bf16 stream (the last
// layer). The layer scalar is optional: `kScalar` from whether the call passes one.
#include <sgl_kernel/distributed/pull/epilogue/rms_norm_residual.cuh>

#include "launch.cuh"
#include <cstdint>

namespace sglang::host::distributed::pull {

// Builds the epilogue's parameters and launches, with `kScalar` from whether `scalar` is given.
template <uint32_t kHiddenSize, bool kUsePDL, bool kNextNorm>
void launch_norm_residual(
    CommunicatorRef comm,
    CallShape& shape,
    TensorView residual,
    int64_t rows,
    TensorView post_norm_weight,
    Optional<TensorView> next_norm_weight,
    Optional<TensorView> scalar,
    float eps,
    TensorView out,
    int64_t out_mc,
    Optional<TensorView> scales,
    int64_t scales_mc,
    int64_t partial_mc,
    int64_t row0,
    int64_t num_blocks,
    int64_t rows_per_group) {
  using Launcher = PullLauncher<kUsePDL, 1, 2>;
  const auto launch = [&]<bool kScalar>() {
    using Epi = RMSNormResidual<kHiddenSize, kNextNorm, kScalar>;
    const auto epi = Epi::make_params(
        shape, residual, rows, post_norm_weight, next_norm_weight, scalar, eps, out, out_mc, scales, scales_mc);
    Launcher::template launch<Epi>(comm, shape, epi, partial_mc, row0, rows, num_blocks, rows_per_group);
  };
  if (scalar.has_value()) {
    launch.template operator()<true>();
  } else {
    launch.template operator()<false>();
  }
}

// `partial [num_tokens, kHiddenSize]` is this rank's view of the partials' buffer, `partial_mc` its multicast VA;
// `out`, `scales` likewise for the FP8 rows and their fp32 scales (`[num_tokens, kHiddenSize / 128]`, any strides).
template <uint32_t kHiddenSize, bool kUsePDL>
void all_reduce_norm_residual_norm_quant(
    CommunicatorRef comm,
    TensorView partial,
    int64_t partial_mc,
    TensorView residual,
    TensorView post_norm_weight,
    TensorView next_norm_weight,
    Optional<TensorView> scalar,
    float eps,
    TensorView out,
    int64_t out_mc,
    TensorView scales,
    int64_t scales_mc,
    int64_t row0,
    int64_t rows,
    int64_t num_blocks,
    int64_t rows_per_group) {
  CallShape shape;
  shape.verify_tensor(partial, kHiddenSize);
  launch_norm_residual<kHiddenSize, kUsePDL, /*kNextNorm=*/true>(
      comm,
      shape,
      residual,
      rows,
      post_norm_weight,
      next_norm_weight,
      scalar,
      eps,
      out,
      out_mc,
      scales,
      scales_mc,
      partial_mc,
      row0,
      num_blocks,
      rows_per_group);
}

// The last layer: the bf16 stream into `out` (may be the partials' buffer: each row is read, then written, by its
// owner only).
template <uint32_t kHiddenSize, bool kUsePDL>
void all_reduce_norm_residual(
    CommunicatorRef comm,
    TensorView partial,
    int64_t partial_mc,
    TensorView residual,
    TensorView post_norm_weight,
    Optional<TensorView> scalar,
    float eps,
    TensorView out,
    int64_t out_mc,
    int64_t row0,
    int64_t rows,
    int64_t num_blocks,
    int64_t rows_per_group) {
  CallShape shape;
  shape.verify_tensor(partial, kHiddenSize);
  launch_norm_residual<kHiddenSize, kUsePDL, /*kNextNorm=*/false>(
      comm,
      shape,
      residual,
      rows,
      post_norm_weight,
      {},
      scalar,
      eps,
      out,
      out_mc,
      {},
      0,
      partial_mc,
      row0,
      num_blocks,
      rows_per_group);
}

}  // namespace sglang::host::distributed::pull
