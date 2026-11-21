#pragma once

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>

#include <sgl_kernel/distributed/ptx.cuh>
#include <sgl_kernel/distributed/pull/epilogue/common.cuh>

#include <cstdint>

namespace sglang::device::distributed::pull {

using tvm::ffi::TensorView;

/// `y = bf16(x * norm_weight * rsqrt(mean(x^2) + eps))` on every row, or with `kNormPrefix` on the rows below
/// `num_norm_rows` and `y = x` on the rest (Kimi K3's flat `[latent, shared]` buffer); `y` multicast into `out_mc`.
template <uint32_t kHiddenSize, bool kNormPrefix = false>
struct RMSNorm {
 public:
  using Tile = RowTile<kHiddenSize>;
  using Vec = VecStorage<bf16_t>;
  static constexpr uint32_t kNumSums = 1;

  struct Params {
    uint8_t* out_mc;            // [num_tokens, hidden] bf16, multicast VA (the partials' buffer for an in-place call)
    const bf16_t* norm_weight;  // [hidden_size]
    float eps;
    uint32_t num_norm_rows;  // (kNormPrefix)
  };

  template <uint32_t kNumRows>
  struct Inputs {};

  SGL_DEVICE explicit RMSNorm(const Params& p) : m_params(p) {
    m_norm_weight.load(assume_global(p.norm_weight), threadIdx.x);
  }

  template <uint32_t kNumRows>
  SGL_DEVICE Inputs<kNumRows> prefetch(RowGroup) const {
    return {};
  }

  template <uint32_t kNumRows, typename Smem>
  SGL_DEVICE void
  store(RowGroup group, const Vec (&x)[kNumRows], const Inputs<kNumRows>&, Smem& smem, uint32_t phase) const {
    float sum[kNumRows];
#pragma unroll
    for (uint32_t u = 0; u < kNumRows; ++u) {
      float acc0 = 0.0f, acc1 = 0.0f;
      bf16_square_sum(x[u], acc0, acc1);
      sum[u] = acc0 + acc1;
    }
    smem.template publish<kNumRows>(sum, phase);
#pragma unroll
    for (uint32_t u = 0; u < kNumRows; ++u) {
      const uint32_t row = group.row + u;
      const bool norm = !kNormPrefix || row < m_params.num_norm_rows;
      const auto r = rms_factor(smem.total(u, phase), 1.0f / kHiddenSize, m_params.eps);
      const auto y = norm ? bf16_norm_scale(x[u], m_norm_weight, r) : x[u];
      ptx::st_multimem_16B(y, m_params.out_mc, group.template vec<Tile::kRowVecs>(u));
    }
  }

  /// `shape` holds the all-reduced tensor's rows; `out_mc` is its multicast VA.
  static Params
  make_params(CallShape& shape, TensorView norm_weight, float eps, int64_t num_norm_rows, int64_t out_mc) {
    shape.verify_weight(norm_weight, kHiddenSize);
    const int64_t num_tokens = shape.get_num_tokens();
    CHECK_HOST(0 <= num_norm_rows && num_norm_rows <= num_tokens) << num_norm_rows << ' ' << num_tokens;
    CHECK_HOST(out_mc != 0) << "the output needs a multicast address";
    return {
        .out_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(out_mc)),
        .norm_weight = get_pointer<const bf16_t>(norm_weight),
        .eps = eps,
        .num_norm_rows = static_cast<uint32_t>(num_norm_rows),
    };
  }

 private:
  const Params& m_params;
  Vec m_norm_weight;
};

}  // namespace sglang::device::distributed::pull
