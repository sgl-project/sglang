#pragma once

#include <sgl_kernel/distributed/push/epilogue/common.cuh>

namespace sglang::device::distributed::push {

using tvm::ffi::TensorView, tvm::ffi::Optional;

/// `SandwichNorm`'s tensors and scales; at namespace level so both `kNextNorm` instantiations share one type.
struct SandwichNormParams {
  bf16_t* combined;                // [num_tokens, hidden_size]
  bf16_t* normalized;              // [num_tokens, hidden_size] (next norm)
  const bf16_t* residual;          // [num_tokens, hidden_size]
  const bf16_t* post_norm_weight;  // [hidden_size]
  const bf16_t* next_norm_weight;  // [hidden_size] (next norm)
  float eps;
  float x_scale;
  float residual_scale;
};

/**
 * \brief Beam 1.0's sandwich norm, every op rounded to bf16 as the unfused path: `y = rmsnorm(x) * post_norm_weight`,
 *        `combined = bf16(bf16(y * x_scale) + bf16(residual * residual_scale))`; with `kNextNorm`, also the next
 *        layer's input `normalized = rmsnorm(combined) * next_norm_weight`. Every multiply and add is a separate `.rn`
 *        op, as in the unfused Triton op (FP fusion disabled), so no FMA merges two roundings: squares summed in one
 *        chain, `rsqrt(sum / hidden + eps)`, then `(x * r) * w`.
 */
template <uint32_t kHiddenSize, bool kNextNorm>
struct SandwichNorm {
 public:
  using Vec = VecStorage<bf16_t>;
  template <uint32_t kClusterSize, uint32_t kNumWarps>
  using Smem = ClusterRowSum<kClusterSize, kNumWarps, kNextNorm ? 2 : 1>;
  using Params = SandwichNormParams;

  SGL_DEVICE SandwichNorm(const Params& p, uint32_t /*row*/, uint32_t col, uint32_t vid) : m_params(p), m_vid(vid) {
    m_residual.load(assume_global(p.residual), m_vid);
    m_post_norm_weight.load(assume_global(p.post_norm_weight), col);
    if constexpr (kNextNorm) m_next_norm_weight.load(assume_global(p.next_norm_weight), col);
  }

  template <typename SmemT>
  SGL_DEVICE void store(const Vec& x, SmemT& smem) const {
    smem.wait_peers();
    const auto y = rms_norm(x, m_post_norm_weight, smem, 0);
    const auto x_scale = fp32x2_t{m_params.x_scale, m_params.x_scale};
    const auto residual_scale = fp32x2_t{m_params.residual_scale, m_params.residual_scale};
    Vec combined;
#pragma unroll
    for (uint32_t j = 0; j < kVecPairs<bf16_t>; ++j) {
      const auto v = cast<bf16x2_t>(math::mul_f32x2(cast<fp32x2_t>(y[j]), x_scale));
      const auto r = cast<bf16x2_t>(math::mul_f32x2(cast<fp32x2_t>(m_residual[j]), residual_scale));
      combined[j] = cast<bf16x2_t>(math::add_f32x2(cast<fp32x2_t>(v), cast<fp32x2_t>(r)));
    }
    combined.store(assume_global(m_params.combined), m_vid);
    if constexpr (kNextNorm) {
      const auto normalized = rms_norm(combined, m_next_norm_weight, smem, 1);
      normalized.store(assume_global(m_params.normalized), m_vid);
    }
  }

  static Params make_params(
      CallShape& shape,  //
      TensorView residual,
      TensorView post_norm_weight,
      Optional<TensorView> next_norm_weight,
      TensorView combined,
      Optional<TensorView> normalized,
      float eps,
      float x_scale,
      float residual_scale) {
    CHECK_HOST(next_norm_weight.has_value() == kNextNorm && normalized.has_value() == kNextNorm);
    shape.verify_tensor(residual, kHiddenSize);
    shape.verify_tensor(combined, kHiddenSize);
    shape.verify_weight(post_norm_weight, kHiddenSize);
    if constexpr (kNextNorm) {
      shape.verify_weight(*next_norm_weight, kHiddenSize);
      shape.verify_tensor(*normalized, kHiddenSize);
    }
    return {
        .combined = get_pointer<bf16_t>(combined),
        .normalized = get_pointer<bf16_t>(normalized),
        .residual = get_pointer<const bf16_t>(residual),
        .post_norm_weight = get_pointer<const bf16_t>(post_norm_weight),
        .next_norm_weight = get_pointer<const bf16_t>(next_norm_weight),
        .eps = eps,
        .x_scale = x_scale,
        .residual_scale = residual_scale,
    };
  }

 private:
  /// RMSNorm with every op rounded on its own, as the unfused Triton op's `inv_rms` and `v * rstd * w`.
  template <typename SmemT>
  SGL_DEVICE Vec rms_norm(const Vec& x, const Vec& weight, SmemT& smem, uint32_t index) const {
    float sum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kVecPairs<bf16_t>; ++j) {
      const auto f = cast<fp32x2_t>(x[j]);
      sum = __fadd_rn(sum, __fmul_rn(f.x, f.x));
      sum = __fadd_rn(sum, __fmul_rn(f.y, f.y));
    }
    const float r = rsqrtf(__fadd_rn(__fdiv_rn(smem.sum(sum, index), float(kHiddenSize)), m_params.eps));
    Vec out;
#pragma unroll
    for (uint32_t j = 0; j < kVecPairs<bf16_t>; ++j) {
      const auto f = cast<fp32x2_t>(x[j]);
      const auto w = cast<fp32x2_t>(weight[j]);
      out[j] = cast<bf16x2_t>(fp32x2_t{__fmul_rn(__fmul_rn(f.x, r), w.x), __fmul_rn(__fmul_rn(f.y, r), w.y)});
    }
    return out;
  }

  const Params& m_params;
  uint32_t m_vid;
  Vec m_residual;
  Vec m_post_norm_weight;
  Vec m_next_norm_weight;  // kNextNorm
};

// Beam 1.0
static_assert(Epilogue<SandwichNorm<5120, false>> && Epilogue<SandwichNorm<5120, true>>);

}  // namespace sglang::device::distributed::push
