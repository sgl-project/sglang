#include "common.h"
#include "vec.h"

namespace {

inline float softplus(float x, double threshold = 20.0) {
  if (x > threshold)
    return x;
  else if (x < -threshold)
    return std::exp(x);
  else
    return std::log1p(std::exp(x));
}

inline at::vec::Vectorized<float> softplus(const at::vec::Vectorized<float>& x, double threshold = 20.0) {
  at::vec::Vectorized<float> mask_hi = x > at::vec::Vectorized<float>(threshold);
  at::vec::Vectorized<float> mask_lo = x < at::vec::Vectorized<float>(-threshold);

  at::vec::Vectorized<float> expx = x.exp();
  at::vec::Vectorized<float> log1pex = (expx + at::vec::Vectorized<float>(1.0f)).log();

  return at::vec::Vectorized<float>::blendv(at::vec::Vectorized<float>::blendv(log1pex, expx, mask_lo), x, mask_hi);
}
template <typename scalar_t>
void fused_gdn_gating_kernel_impl(
    float* __restrict__ A_log,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ dt_bias,
    float* __restrict__ out,
    int64_t batch,
    int64_t num_heads) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int vec_size = bVec::size();
  constexpr int fvec_size = fVec::size();
  fVec neg_one(-1.0f);
  at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      int64_t j = 0;
      for (; j < num_heads - (num_heads % vec_size); j += vec_size) {
        fVec A_log_vec0 = fVec::loadu(A_log + j);
        fVec A_log_vec1 = fVec::loadu(A_log + j + fvec_size);
        bVec dt_bias_vec = bVec::loadu(dt_bias + j);
        bVec a_bvec = bVec::loadu(a + i * num_heads + j);
        fVec a0, a1, dt_bias_vec0, dt_bias_vec1;
        std::tie(a0, a1) = at::vec::convert_to_float(a_bvec);
        std::tie(dt_bias_vec0, dt_bias_vec1) = at::vec::convert_to_float(dt_bias_vec);

        fVec g0 = neg_one * A_log_vec0.exp() * softplus(a0 + dt_bias_vec0);
        fVec g1 = neg_one * A_log_vec1.exp() * softplus(a1 + dt_bias_vec1);

        g0.store(out + i * num_heads + j);
        g1.store(out + i * num_heads + j + fvec_size);
      }
      for (; j < num_heads; ++j) {
        out[i * num_heads + j] = -std::exp(A_log[j]) * softplus(float(a[i * num_heads + j]) + float(dt_bias[j]));
      }
    }
  });
}
}  // anonymous namespace

// A_log: [num_v_heads]
// a: [batch, num_v_heads]
// dt_bias: [num_v_heads]
// -A_log.float().exp() * F.softplus(a.float() + dt_bias)
at::Tensor fused_gdn_gating_cpu(const at::Tensor& A_log, const at::Tensor& a, const at::Tensor& dt_bias) {
  RECORD_FUNCTION("sgl-kernel::fused_gdn_gating_cpu", std::vector<c10::IValue>({A_log, a, dt_bias}));
  CHECK_DIM(1, A_log);
  CHECK_DIM(2, a);
  CHECK_DIM(1, dt_bias);
  CHECK_CONTIGUOUS(a);
  CHECK_EQ(A_log.size(0), a.size(1));
  CHECK_EQ(A_log.size(0), dt_bias.size(0));
  int batch = a.size(0);
  int num_heads = a.size(1);
  at::Tensor out = at::empty_like(a, a.options().dtype(at::kFloat));
  AT_DISPATCH_REDUCED_FLOATING_TYPES(a.scalar_type(), "fused_gdn_gating_kernel", [&] {
    fused_gdn_gating_kernel_impl<scalar_t>(
        A_log.data_ptr<float>(),
        a.data_ptr<scalar_t>(),
        dt_bias.data_ptr<scalar_t>(),
        out.data_ptr<float>(),
        batch,
        num_heads);
  });
  return out;
}
