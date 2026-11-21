#pragma once

#include <sgl_kernel/distributed/push/epilogue/common.cuh>

namespace sglang::device::distributed::push {

using tvm::ffi::TensorView;

/// `out = rmsnorm(x)` on every row, or with `kNormPrefix` on the rows below `num_norm_rows` and `out = x` on the rest
/// (Kimi K3's flat `[latent, shared]` buffer); `out` may be the input.
template <uint32_t kHiddenSize, bool kNormPrefix = false>
struct RMSNorm {
 public:
  using Vec = VecStorage<bf16_t>;
  template <uint32_t kClusterSize, uint32_t kNumWarps>
  using Smem = ClusterRowSum<kClusterSize, kNumWarps, 1>;

  struct Params {
    bf16_t* out;                // [num_tokens, kHiddenSize]
    const bf16_t* norm_weight;  // [kHiddenSize]
    float eps;
    uint32_t num_norm_rows;  // (kNormPrefix)
  };

  SGL_DEVICE RMSNorm(const Params& p, uint32_t row, uint32_t col, uint32_t vid) : m_params(p), m_row(row), m_vid(vid) {
    m_norm_weight.load(assume_global(p.norm_weight), col);
  }

  template <typename SmemT>
  SGL_DEVICE void store(const Vec& x, SmemT& smem) const {
    smem.wait_peers();
    const bool norm = !kNormPrefix || m_row < m_params.num_norm_rows;
    const auto out = norm ? smem.rms_norm(x, m_norm_weight, m_params.eps, 1.0f / kHiddenSize) : x;
    out.store(assume_global(m_params.out), m_vid);
  }

  static Params make_params(CallShape& shape, TensorView out, TensorView norm_weight, float eps)
    requires(!kNormPrefix)
  {
    shape.verify_tensor(out, kHiddenSize);
    shape.verify_weight(norm_weight, kHiddenSize);
    return {
        .out = get_pointer<bf16_t>(out),
        .norm_weight = get_pointer<const bf16_t>(norm_weight),
        .eps = eps,
        .num_norm_rows = shape.get_num_tokens(),
    };
  }

  static Params make_params(CallShape& shape, TensorView out, TensorView norm_weight, float eps, int64_t num_norm_rows)
    requires kNormPrefix
  {
    shape.verify_tensor(out, kHiddenSize);
    shape.verify_weight(norm_weight, kHiddenSize);
    const int64_t num_tokens = shape.get_num_tokens();
    CHECK_HOST(0 <= num_norm_rows && num_norm_rows <= num_tokens) << num_norm_rows << ' ' << num_tokens;
    return {
        .out = get_pointer<bf16_t>(out),
        .norm_weight = get_pointer<const bf16_t>(norm_weight),
        .eps = eps,
        .num_norm_rows = static_cast<uint32_t>(num_norm_rows),
    };
  }

 private:
  const Params& m_params;
  uint32_t m_row;
  uint32_t m_vid;
  Vec m_norm_weight;
};

// Kimi K3
static_assert(Epilogue<RMSNorm<3584>> && Epilogue<RMSNorm<3584, true>>);

}  // namespace sglang::device::distributed::push
