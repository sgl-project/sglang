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
  using Vec = at::vec::Vectorized<float>;
  Vec mask_hi = x > Vec(threshold);
  Vec mask_lo = x < Vec(-threshold);

  Vec expx = x.exp_u20();
  Vec log1pex = (expx + Vec(1.0f)).log();

  return Vec::blendv(Vec::blendv(log1pex, expx, mask_lo), x, mask_hi);
}

template <typename scalar_t>
void fused_sigmoid_gating_delta_rule_update_kernel_impl(
    scalar_t* __restrict__ q_ptr,
    scalar_t* __restrict__ k_ptr,
    const scalar_t* __restrict__ v_ptr,
    const float* __restrict__ A_log_ptr,
    const scalar_t* __restrict__ a_ptr,
    const scalar_t* __restrict__ dt_bias_ptr,
    const scalar_t* __restrict__ b_ptr,
    const int32_t* __restrict__ indices_ptr,
    float* __restrict__ state_ptr,
    scalar_t* __restrict__ o_ptr,
    float* __restrict__ qk_scale_buf,
    int64_t seq_len,
    int64_t batch_size,
    int64_t num_heads,
    int64_t head_dim,
    int64_t v_num_heads,
    int64_t v_head_dim,
    int64_t q_strideB,
    int64_t q_strideS,
    int64_t q_strideH,
    int64_t k_strideB,
    int64_t k_strideS,
    int64_t k_strideH,
    int64_t v_strideB,
    int64_t v_strideS,
    int64_t v_strideH,
    bool use_qk_l2norm_in_kernel,
    double softplus_threshold) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int64_t VecSize = bVec::size();
  constexpr int64_t fVecSize = fVec::size();
  int64_t group_size = v_num_heads / num_heads;
  double scale = 1 / std::sqrt(head_dim);
  fVec scale_vec = fVec(scale);
  if (use_qk_l2norm_in_kernel) {
    float eps = 1e-5;
    at::parallel_for(0, batch_size * seq_len * num_heads, 0, [&](int64_t begin, int64_t end) {
      int64_t bi{0}, si{0}, ni{0};
      data_index_init(begin, bi, batch_size, si, seq_len, ni, num_heads);
      for (int64_t i = begin; i < end; ++i) {
        float sum_q = float(0);
        float sum_k = float(0);
        fVec sum_q_fvec = fVec(float(0));
        fVec sum_k_fvec = fVec(float(0));
        int64_t q_offset = bi * q_strideB + si * q_strideS + ni * q_strideH;
        int64_t k_offset = bi * k_strideB + si * k_strideS + ni * k_strideH;
        int64_t q_scale_offset = bi * seq_len * num_heads + si * num_heads + ni;
        int64_t k_scale_offset = q_scale_offset + batch_size * seq_len * num_heads;
        int64_t d;
#pragma GCC unroll 4
        for (d = 0; d <= head_dim - VecSize; d += VecSize) {
          bVec q_bvec = bVec::loadu(q_ptr + q_offset + d);
          fVec q_fvec0, q_fvec1;
          std::tie(q_fvec0, q_fvec1) = at::vec::convert_to_float(q_bvec);
          sum_q_fvec += q_fvec0 * q_fvec0;
          sum_q_fvec += q_fvec1 * q_fvec1;
          bVec k_bvec = bVec::loadu(k_ptr + k_offset + d);
          fVec k_fvec0, k_fvec1;
          std::tie(k_fvec0, k_fvec1) = at::vec::convert_to_float(k_bvec);
          sum_k_fvec += k_fvec0 * k_fvec0;
          sum_k_fvec += k_fvec1 * k_fvec1;
        }
#pragma GCC unroll 4
        for (; d < head_dim; ++d) {
          float q_val = static_cast<float>(q_ptr[q_offset + d]);
          sum_q += q_val * q_val;
          float k_val = static_cast<float>(k_ptr[k_offset + d]);
          sum_k += k_val * k_val;
        }

        sum_q += vec_reduce_sum(sum_q_fvec);
        sum_k += vec_reduce_sum(sum_k_fvec);
        qk_scale_buf[q_scale_offset] = float(1) / std::sqrt(sum_q + eps);
        qk_scale_buf[k_scale_offset] = float(1) / std::sqrt(sum_k + eps);

        data_index_step(bi, batch_size, si, seq_len, ni, num_heads);
      }
    });
  }
  at::parallel_for(0, batch_size * seq_len * v_num_heads, 0, [&](int64_t begin, int64_t end) {
    int64_t bi{0}, si{0}, ni{0};
    data_index_init(begin, bi, batch_size, si, seq_len, ni, v_num_heads);
    for (int64_t i = begin; i < end; ++i) {
      int64_t cache_index = indices_ptr[bi];
      int64_t state_offset = (cache_index * v_num_heads + ni) * head_dim * v_head_dim;
      float g_val = -std::exp(A_log_ptr[ni]) *
                    softplus(float(a_ptr[bi * v_num_heads + ni]) + float(dt_bias_ptr[ni]), softplus_threshold);
      float g_val_exp = std::exp(g_val);
      fVec g_val_exp_vec = fVec(g_val_exp);
      int64_t q_offset = si * q_strideS + bi * q_strideB + (ni / group_size) * q_strideH;
      int64_t k_offset = si * k_strideS + bi * k_strideB + (ni / group_size) * k_strideH;
      int64_t q_scale_offset = bi * seq_len * num_heads + si * num_heads + (ni / group_size);
      int64_t k_scale_offset = q_scale_offset + batch_size * seq_len * num_heads;
      float q_scale = use_qk_l2norm_in_kernel ? qk_scale_buf[q_scale_offset] : 1.0f;
      float k_scale = use_qk_l2norm_in_kernel ? qk_scale_buf[k_scale_offset] : 1.0f;
      int64_t v_offset = si * v_strideS + bi * v_strideB + ni * v_strideH;
      int64_t o_offset = ((bi * seq_len + si) * v_num_heads + ni) * v_head_dim;
      float beta_val = 1 / (1 + std::exp(-b_ptr[ni]));
      fVec beta_vec = fVec(beta_val);
      int64_t dvi = 0;
      for (; dvi <= v_head_dim - VecSize; dvi += VecSize) {
        fVec kv_mem_vec0 = fVec(float(0));
        fVec kv_mem_vec1 = fVec(float(0));
        for (int di = 0; di < head_dim; ++di) {
          fVec k_val_vec = fVec(k_ptr[k_offset + di]) * fVec(k_scale);
          fVec state_vec0 = fVec::loadu(state_ptr + state_offset + di * v_head_dim + dvi);
          fVec state_vec1 = fVec::loadu(state_ptr + state_offset + di * v_head_dim + dvi + fVecSize);
          kv_mem_vec0 = kv_mem_vec0 + state_vec0 * g_val_exp_vec * k_val_vec;
          kv_mem_vec1 = kv_mem_vec1 + state_vec1 * g_val_exp_vec * k_val_vec;
        }
        bVec v_bvec = bVec::loadu(v_ptr + v_offset + dvi);
        fVec v_vec0, v_vec1;
        std::tie(v_vec0, v_vec1) = at::vec::convert_to_float(v_bvec);
        fVec dt_vec0 = (v_vec0 - kv_mem_vec0) * beta_vec;
        fVec dt_vec1 = (v_vec1 - kv_mem_vec1) * beta_vec;
        fVec o_vec0 = fVec(float(0));
        fVec o_vec1 = fVec(float(0));
        for (int di = 0; di < head_dim; ++di) {
          fVec q_vec = fVec(q_ptr[q_offset + di]) * fVec(q_scale);
          fVec k_vec = fVec(k_ptr[k_offset + di]) * fVec(k_scale);
          fVec state_vec0 = fVec::loadu(state_ptr + state_offset + di * v_head_dim + dvi);
          fVec state_vec1 = fVec::loadu(state_ptr + state_offset + di * v_head_dim + dvi + fVecSize);
          state_vec0 = state_vec0 * g_val_exp_vec + k_vec * dt_vec0;
          state_vec1 = state_vec1 * g_val_exp_vec + k_vec * dt_vec1;
          o_vec0 = o_vec0 + state_vec0 * q_vec * scale_vec;
          o_vec1 = o_vec1 + state_vec1 * q_vec * scale_vec;
          state_vec0.store(state_ptr + state_offset + di * v_head_dim + dvi);
          state_vec1.store(state_ptr + state_offset + di * v_head_dim + dvi + fVecSize);
        }
        bVec o_vec = at::vec::convert_from_float<scalar_t>(o_vec0, o_vec1);
        o_vec.store(o_ptr + o_offset + dvi);
      }
      for (; dvi < v_head_dim; ++dvi) {
        float kv_mem_val = 0;
        for (int di = 0; di < head_dim; ++di) {
          float k_val = k_ptr[k_offset + di] * k_scale;
          state_ptr[state_offset + di * v_head_dim + dvi] *= g_val_exp;
          kv_mem_val += state_ptr[state_offset + di * v_head_dim + dvi] * k_val;
        }
        float v_val = v_ptr[v_offset + dvi];
        float dt_val = (v_val - kv_mem_val) * beta_val;
        float o_val = 0;
        for (int di = 0; di < head_dim; ++di) {
          float q_val = q_ptr[q_offset + di] * q_scale;
          float k_val = k_ptr[k_offset + di] * k_scale;
          state_ptr[state_offset + di * v_head_dim + dvi] += k_val * dt_val;
          o_val += state_ptr[state_offset + di * v_head_dim + dvi] * q_val * scale;
        }
        o_ptr[o_offset + dvi] = o_val;
      }
      data_index_step(bi, batch_size, si, seq_len, ni, v_num_heads);
    }
  });
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
  const fVec neg_one(-1.0f);
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

        fVec g0 = neg_one * A_log_vec0.exp_u20() * softplus(a0 + dt_bias_vec0);
        fVec g1 = neg_one * A_log_vec1.exp_u20() * softplus(a1 + dt_bias_vec1);

        g0.store(out + i * num_heads + j);
        g1.store(out + i * num_heads + j + fvec_size);
      }
      for (; j < num_heads; ++j) {
        out[i * num_heads + j] = -std::exp(A_log[j]) * softplus(float(a[i * num_heads + j]) + float(dt_bias[j]));
      }
    }
  });
}

}  // namespace

// A_log: [v_num_heads]
// dt_bias: [v_num_heads]
// query: [seq_len, batch_size, num_heads, head_dim]
// key: [seq_len, batch_size, num_heads, head_dim]
// value: [seq_len, batch_size, v_num_heads, v_head_dim]
// a: [batch_size, v_num_heads]
// b: [batch_size, v_num_heads]
// initial_state_source:[num_tokens, v_num_heads, head_dim, v_head_dim]
// initial_state_indices: [batch_size]
// cu_seqlens: [batch_size + 1]
at::Tensor fused_sigmoid_gating_delta_rule_update_cpu(
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    at::Tensor& q,
    at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& a,
    const at::Tensor& b,
    at::Tensor& initial_state_source,
    const at::Tensor& initial_state_indices,
    const at::Tensor& cu_seqlens,
    bool use_qk_l2norm_in_kernel,
    double softplus_beta = 1.0,
    double softplus_threshold = 20.0) {
  RECORD_FUNCTION(
      "sgl-kernel::fused_sigmoid_gating_delta_rule_update_cpu",
      std::vector<c10::IValue>(
          {A_log, dt_bias, q, k, v, a, b, initial_state_source, initial_state_indices, cu_seqlens}));
  CHECK_DIM(4, q);
  CHECK_DIM(4, v);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  int64_t seq_len = q.size(0);
  int64_t batch_size = q.size(1);
  int64_t num_heads = q.size(2);
  int64_t head_dim = q.size(3);
  int64_t v_num_heads = v.size(2);
  int64_t v_head_dim = v.size(3);
  CHECK_INPUT_SHAPE_DTYPE<true, false>(k, {seq_len, batch_size, num_heads, head_dim}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<true, false>(v, {seq_len, batch_size, v_num_heads, v_head_dim}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<false, true>(A_log, {v_num_heads}, at::kFloat);
  CHECK_INPUT_SHAPE_DTYPE<false, true>(a, {batch_size, v_num_heads}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<false, true>(dt_bias, {v_num_heads}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<false, true>(b, {batch_size, v_num_heads}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<false, true>(initial_state_indices, {batch_size}, at::kInt);
  CHECK_INPUT_SHAPE_DTYPE<false, true>(cu_seqlens, {batch_size + 1}, at::kInt);
  CHECK_INPUT_SHAPE_DTYPE<false, true>(
      initial_state_source, {initial_state_source.size(0), v_num_heads, head_dim, v_head_dim}, at::kFloat);
  CHECK(initial_state_source.size(0) >= batch_size);
  CHECK_EQ(v_num_heads % num_heads, 0);

  int64_t q_strideB = q.stride(1);
  int64_t q_strideS = q.stride(0);
  int64_t q_strideH = q.stride(2);
  int64_t k_strideB = k.stride(1);
  int64_t k_strideS = k.stride(0);
  int64_t k_strideH = k.stride(2);
  int64_t v_strideB = v.stride(1);
  int64_t v_strideS = v.stride(0);
  int64_t v_strideH = v.stride(2);
  at::Tensor core_attn_out = at::empty({batch_size, seq_len, v_num_heads, v_head_dim}, q.options());
  at::Tensor qk_scale_buf = at::empty({2 * batch_size, seq_len, num_heads}, at::kFloat);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(), "fused_sigmoid_gating_delta_rule_update_kernel_impl", [&] {
    fused_sigmoid_gating_delta_rule_update_kernel_impl<scalar_t>(
        q.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        A_log.data_ptr<float>(),
        a.data_ptr<scalar_t>(),
        dt_bias.data_ptr<scalar_t>(),
        b.data_ptr<scalar_t>(),
        initial_state_indices.data_ptr<int32_t>(),
        initial_state_source.data_ptr<float>(),
        core_attn_out.data_ptr<scalar_t>(),
        qk_scale_buf.data_ptr<float>(),
        seq_len,
        batch_size,
        num_heads,
        head_dim,
        v_num_heads,
        v_head_dim,
        q_strideB,
        q_strideS,
        q_strideH,
        k_strideB,
        k_strideS,
        k_strideH,
        v_strideB,
        v_strideS,
        v_strideH,
        use_qk_l2norm_in_kernel,
        softplus_threshold);
  });
  return core_attn_out;
}

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
