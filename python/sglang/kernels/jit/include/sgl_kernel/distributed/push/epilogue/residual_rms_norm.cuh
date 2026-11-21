#pragma once

#include <sgl_kernel/distributed/push/epilogue/common.cuh>

namespace sglang::device::distributed::push {

using tvm::ffi::TensorView;

/// `residual_out = bf16(x + residual)`, `out = rmsnorm(residual_out)`.
template <uint32_t kHiddenSize>
struct ResidualRMSNorm {
 public:
  using Vec = VecStorage<bf16_t>;
  template <uint32_t kClusterSize, uint32_t kNumWarps>
  using Smem = ClusterRowSum<kClusterSize, kNumWarps, 1>;

  struct Params {
    bf16_t* out;                // [num_tokens, kHiddenSize]
    const bf16_t* residual;     // [num_tokens, kHiddenSize]
    bf16_t* residual_out;       // [num_tokens, kHiddenSize], may alias `residual`
    const bf16_t* norm_weight;  // [kHiddenSize]
    float eps;
  };

  SGL_DEVICE ResidualRMSNorm(const Params& p, uint32_t /*row*/, uint32_t col, uint32_t vid) : m_params(p), m_vid(vid) {
    m_residual.load(assume_global(p.residual), m_vid);
    m_norm_weight.load(assume_global(p.norm_weight), col);
  }

  template <typename SmemT>
  SGL_DEVICE void store(const Vec& x, SmemT& smem) const {
    smem.wait_peers();
    Vec sum;
#pragma unroll
    for (uint32_t j = 0; j < kVecPairs<bf16_t>; ++j) {
      sum[j] = __hadd2(x[j], m_residual[j]);
    }
    sum.store(assume_global(m_params.residual_out), m_vid);
    smem.rms_norm(sum, m_norm_weight, m_params.eps, 1.0f / kHiddenSize).store(assume_global(m_params.out), m_vid);
  }

  static Params make_params(
      CallShape& shape,  //
      TensorView out,
      TensorView residual,
      TensorView residual_out,
      TensorView norm_weight,
      float eps) {
    shape.verify_tensor(out, kHiddenSize);
    shape.verify_tensor(residual, kHiddenSize);
    shape.verify_tensor(residual_out, kHiddenSize);
    shape.verify_weight(norm_weight, kHiddenSize);
    return {
        .out = get_pointer<bf16_t>(out),
        .residual = get_pointer<const bf16_t>(residual),
        .residual_out = get_pointer<bf16_t>(residual_out),
        .norm_weight = get_pointer<const bf16_t>(norm_weight),
        .eps = eps,
    };
  }

 private:
  const Params& m_params;
  uint32_t m_vid;
  Vec m_residual;
  Vec m_norm_weight;
};

// MiniMax M3
static_assert(Epilogue<ResidualRMSNorm<6144>>);

}  // namespace sglang::device::distributed::push
