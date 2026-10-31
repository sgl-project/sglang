#include <torch/all.h>
#include <torch/library.h>

#include "../common.h"
#include "../vec.h"

namespace {

template <typename scalar_t>
void sparse_attention_kernel(
    const at::Tensor& query,
    const at::Tensor& swa,
    const at::Tensor& extra,
    const at::Tensor& swa_slots,
    const at::Tensor& extra_slots,
    const at::Tensor& sink,
    float scale,
    at::Tensor& output) {
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t width = fVec::size();
  const int64_t heads = query.size(1), dim = query.size(2);
  const int64_t window = swa_slots.size(1), selected = extra_slots.size(1), count = window + selected;
  const auto* q = query.const_data_ptr<scalar_t>();
  const auto* sw = swa.const_data_ptr<scalar_t>();
  const auto* ex = extra.const_data_ptr<scalar_t>();
  const auto* sw_loc = swa_slots.const_data_ptr<int64_t>();
  const auto* ex_loc = extra_slots.const_data_ptr<int64_t>();
  const auto* sinks = sink.const_data_ptr<float>();
  auto* out = output.data_ptr<scalar_t>();
  at::parallel_for(0, query.size(0) * heads, 1, [&](int64_t begin, int64_t end) {
    std::vector<const scalar_t*> keys(count);
    std::vector<float> logits(count), values(dim);
    for (int64_t row = begin; row < end; ++row) {
      const int64_t token = row / heads;
      const scalar_t* qr = q + row * dim;
      float maximum = sinks[row % heads];
      for (int64_t i = 0; i < count; ++i) {
        const int64_t slot = i < window ? sw_loc[token * window + i] : ex_loc[token * selected + i - window];
        keys[i] = slot < 0 ? nullptr : (i < window ? sw : ex) + slot * dim;
        if (keys[i] == nullptr) {
          continue;
        }
        fVec sum0(0.f), sum1(0.f);
        int64_t d = 0;
        for (; d + 2 * width <= dim; d += 2 * width) {
          auto [q0, q1] = load_float_vec2(qr + d);
          auto [k0, k1] = load_float_vec2(keys[i] + d);
          sum0 += q0 * k0;
          sum1 += q1 * k1;
        }
        float dot = vec_reduce_sum(sum0 + sum1);
        for (; d < dim; ++d) {
          dot += static_cast<float>(qr[d]) * static_cast<float>(keys[i][d]);
        }
        logits[i] = dot * scale;
        maximum = std::max(maximum, logits[i]);
      }
      float denominator = std::exp(sinks[row % heads] - maximum);
      std::fill(values.begin(), values.end(), 0.f);
      for (int64_t i = 0; i < count; ++i) {
        if (keys[i] == nullptr) {
          continue;
        }
        const float probability = std::exp(logits[i] - maximum);
        denominator += probability;
        const fVec weight(probability);
        int64_t d = 0;
        for (; d + 2 * width <= dim; d += 2 * width) {
          auto [k0, k1] = load_float_vec2(keys[i] + d);
          (fVec::loadu(values.data() + d) + weight * k0).store(values.data() + d);
          (fVec::loadu(values.data() + d + width) + weight * k1).store(values.data() + d + width);
        }
        for (; d < dim; ++d) {
          values[d] += probability * static_cast<float>(keys[i][d]);
        }
      }
      const fVec inverse(1.f / denominator);
      int64_t d = 0;
      for (; d + width <= dim; d += width) {
        store_from_float_ext(out + row * dim + d, fVec::loadu(values.data() + d) * inverse);
      }
      for (; d < dim; ++d) {
        out[row * dim + d] = static_cast<scalar_t>(values[d] / denominator);
      }
    }
  });
}

at::Tensor sparse_attention_cpu(
    const at::Tensor& query,
    const at::Tensor& swa,
    const at::Tensor& extra,
    const at::Tensor& swa_slots,
    const at::Tensor& extra_slots,
    const at::Tensor& sink,
    double scale) {
  CHECK_INPUT(query);
  CHECK_INPUT(swa);
  CHECK_INPUT(extra);
  CHECK_INPUT(swa_slots);
  CHECK_INPUT(extra_slots);
  CHECK_INPUT(sink);
  TORCH_CHECK(query.dim() == 3 && swa.dim() == 2 && extra.dim() == 2, "Invalid sparse attention ranks");
  TORCH_CHECK(
      swa.size(1) == query.size(2) && extra.size(1) == query.size(2) && swa.scalar_type() == query.scalar_type() &&
          extra.scalar_type() == query.scalar_type(),
      "Sparse attention cache shape/dtype mismatch");
  TORCH_CHECK(
      swa_slots.dim() == 2 && extra_slots.dim() == 2 && swa_slots.size(0) == query.size(0) &&
          extra_slots.size(0) == query.size(0) && swa_slots.scalar_type() == at::kLong &&
          extra_slots.scalar_type() == at::kLong && sink.dim() == 1 && sink.size(0) == query.size(1) &&
          sink.scalar_type() == at::kFloat,
      "Invalid sparse attention indices/sinks");
  for (const auto& pair : {std::make_pair(swa_slots, swa.size(0)), std::make_pair(extra_slots, extra.size(0))}) {
    const auto* slots = pair.first.const_data_ptr<int64_t>();
    for (int64_t i = 0; i < pair.first.numel(); ++i) {
      TORCH_CHECK(slots[i] >= -1 && slots[i] < pair.second, "Sparse attention cache slot out of range");
    }
  }
  auto output = at::empty_like(query);
  AT_DISPATCH_REDUCED_FLOATING_TYPES_AND(at::kFloat, query.scalar_type(), "dsv41_sparse_attention_cpu", [&] {
    sparse_attention_kernel<scalar_t>(
        query, swa, extra, swa_slots, extra_slots, sink, static_cast<float>(scale), output);
  });
  return output;
}

std::tuple<at::Tensor, at::Tensor>
sqrtsoftplus_topk_cpu(const at::Tensor& logits, const std::optional<at::Tensor>& bias, int64_t topk, bool renormalize) {
  CHECK_INPUT(logits);
  TORCH_CHECK(logits.dim() == 2 && topk > 0 && topk <= logits.size(1), "Invalid router shape/top-k");
  auto logits_float = logits.to(at::kFloat);
  auto softplus = at::where(logits_float > 20.f, logits_float, at::log1p(at::exp(logits_float)));
  auto scores = softplus.sqrt();
  auto choices = scores;
  if (bias.has_value()) {
    CHECK_INPUT(bias.value());
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == logits.size(1), "Invalid router bias");
    choices = scores + bias->to(at::kFloat);
  }
  auto ids = std::get<1>(choices.topk(topk, -1, true, false));
  auto weights = scores.gather(1, ids);
  if (renormalize) {
    weights = weights / (weights.sum(-1, true) + 1e-20);
  }
  return {weights, ids.to(at::kInt)};
}

void silu_and_mul_clamp_cpu(const at::Tensor& input, at::Tensor output, double limit) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  TORCH_CHECK(
      input.dim() == 2 && output.dim() == 2 && input.size(0) == output.size(0) && input.size(1) == 2 * output.size(1) &&
          input.scalar_type() == output.scalar_type(),
      "Invalid clamped SwiGLU shape/dtype");
  TORCH_CHECK(limit > 0, "SwiGLU limit must be positive");
  const int64_t hidden = output.size(1);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "silu_and_mul_clamp_cpu", [&] {
    const scalar_t* x = input.const_data_ptr<scalar_t>();
    scalar_t* y = output.data_ptr<scalar_t>();
    at::parallel_for(0, output.numel(), GRAIN_SIZE, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        const int64_t offset = i / hidden * 2 * hidden + i % hidden;
        const float gate = std::min(static_cast<float>(x[offset]), static_cast<float>(limit));
        const float up =
            std::clamp(static_cast<float>(x[offset + hidden]), -static_cast<float>(limit), static_cast<float>(limit));
        y[i] = static_cast<scalar_t>((gate / (1.f + std::exp(-gate))) * up);
      }
    });
  });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> hc_split_sinkhorn_cpu(
    const at::Tensor& mixes, const at::Tensor& scale, const at::Tensor& base, int64_t hc, int64_t iters, double eps) {
  CHECK_INPUT(mixes);
  CHECK_INPUT(scale);
  CHECK_INPUT(base);
  TORCH_CHECK(
      mixes.scalar_type() == at::kFloat && scale.scalar_type() == at::kFloat && base.scalar_type() == at::kFloat,
      "mHC coefficients must be float32");
  TORCH_CHECK(hc > 0 && iters > 0 && eps > 0, "Invalid Sinkhorn configuration");
  TORCH_CHECK(mixes.dim() == 3 && mixes.size(2) == (hc + 2) * hc, "Invalid mHC mixes shape");
  TORCH_CHECK(scale.numel() == 3 && base.numel() == (hc + 2) * hc, "Invalid mHC scale/base shape");
  auto pre = at::empty({mixes.size(0), mixes.size(1), hc}, mixes.options());
  auto post = at::empty_like(pre);
  auto comb = at::empty({mixes.size(0), mixes.size(1), hc, hc}, mixes.options());
  const float* s = scale.const_data_ptr<float>();
  const float* b = base.const_data_ptr<float>();
  const float epsilon = static_cast<float>(eps);
  at::parallel_for(0, mixes.size(0) * mixes.size(1), 0, [&](int64_t begin, int64_t end) {
    std::vector<float> sums(hc);
    for (int64_t row = begin; row < end; ++row) {
      const float* x = mixes.const_data_ptr<float>() + row * (hc + 2) * hc;
      float* p = pre.data_ptr<float>() + row * hc;
      float* q = post.data_ptr<float>() + row * hc;
      float* c = comb.data_ptr<float>() + row * hc * hc;
      for (int64_t i = 0; i < hc; ++i) {
        p[i] = 1.f / (1.f + std::exp(-(x[i] * s[0] + b[i]))) + epsilon;
        q[i] = 2.f / (1.f + std::exp(-(x[hc + i] * s[1] + b[hc + i])));
        float max_value = -std::numeric_limits<float>::infinity();
        for (int64_t j = 0; j < hc; ++j) {
          c[i * hc + j] = x[2 * hc + i * hc + j] * s[2] + b[2 * hc + i * hc + j];
          max_value = std::max(max_value, c[i * hc + j]);
        }
        float sum = 0.f;
        for (int64_t j = 0; j < hc; ++j) {
          c[i * hc + j] = std::exp(c[i * hc + j] - max_value);
          sum += c[i * hc + j];
        }
        for (int64_t j = 0; j < hc; ++j) {
          c[i * hc + j] = c[i * hc + j] / sum + epsilon;
        }
      }
      for (int64_t step = 0; step < iters; ++step) {
        if (step > 0) {
          for (int64_t i = 0; i < hc; ++i) {
            float sum = epsilon;
            for (int64_t j = 0; j < hc; ++j) {
              sum += c[i * hc + j];
            }
            for (int64_t j = 0; j < hc; ++j) {
              c[i * hc + j] /= sum;
            }
          }
        }
        std::fill(sums.begin(), sums.end(), epsilon);
        for (int64_t i = 0; i < hc; ++i) {
          for (int64_t j = 0; j < hc; ++j) {
            sums[j] += c[i * hc + j];
          }
        }
        for (int64_t i = 0; i < hc; ++i) {
          for (int64_t j = 0; j < hc; ++j) {
            c[i * hc + j] /= sums[j];
          }
        }
      }
    }
  });
  return {pre, post, comb};
}

at::Tensor hc_combine_cpu(const at::Tensor& input, const at::Tensor& pre, int64_t hc) {
  CHECK_INPUT(input);
  CHECK_INPUT(pre);
  TORCH_CHECK(input.dim() == 2 && hc > 0 && input.size(1) % hc == 0, "Invalid mHC input shape");
  TORCH_CHECK(
      pre.sizes() == at::IntArrayRef({input.size(0), hc}) && pre.scalar_type() == at::kFloat,
      "Invalid mHC pre shape/dtype");
  const int64_t hidden = input.size(1) / hc;
  auto output = at::empty({input.size(0), hidden}, input.options());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, input.scalar_type(), "hc_combine_cpu", [&] {
    const scalar_t* x = input.const_data_ptr<scalar_t>();
    const float* p = pre.const_data_ptr<float>();
    scalar_t* y = output.data_ptr<scalar_t>();
    at::parallel_for(0, input.size(0) * hidden, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        const int64_t row = idx / hidden, col = idx % hidden;
        float sum = 0.f;
        for (int64_t i = 0; i < hc; ++i) {
          sum += static_cast<float>(x[(row * hc + i) * hidden + col]) * p[row * hc + i];
        }
        y[idx] = static_cast<scalar_t>(sum);
      }
    });
  });
  return output;
}

at::Tensor
hc_post_cpu(const at::Tensor& input, const at::Tensor& residual, const at::Tensor& post, const at::Tensor& comb) {
  CHECK_INPUT(input);
  CHECK_INPUT(residual);
  CHECK_INPUT(post);
  CHECK_INPUT(comb);
  TORCH_CHECK(input.dim() == 2 && residual.dim() == 3, "Invalid mHC input/residual rank");
  const int64_t rows = input.size(0), hidden = input.size(1), hc = residual.size(1);
  TORCH_CHECK(
      residual.size(0) == rows && residual.size(2) == hidden && residual.scalar_type() == input.scalar_type(),
      "Invalid mHC residual shape/dtype");
  TORCH_CHECK(
      post.sizes() == at::IntArrayRef({rows, hc}) && comb.sizes() == at::IntArrayRef({rows, hc, hc}) &&
          post.scalar_type() == at::kFloat && comb.scalar_type() == at::kFloat,
      "Invalid mHC coefficient shape/dtype");
  auto output = at::empty_like(residual);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, input.scalar_type(), "hc_post_cpu", [&] {
    const scalar_t* x = input.const_data_ptr<scalar_t>();
    const scalar_t* r = residual.const_data_ptr<scalar_t>();
    const float* p = post.const_data_ptr<float>();
    const float* c = comb.const_data_ptr<float>();
    scalar_t* y = output.data_ptr<scalar_t>();
    at::parallel_for(0, output.numel(), GRAIN_SIZE, [&](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        const int64_t row = idx / (hc * hidden), stream = idx / hidden % hc, col = idx % hidden;
        float sum = 0.f;
        for (int64_t i = 0; i < hc; ++i) {
          sum += c[(row * hc + i) * hc + stream] * static_cast<float>(r[(row * hc + i) * hidden + col]);
        }
        y[idx] = static_cast<scalar_t>(sum + p[row * hc + stream] * static_cast<float>(x[row * hidden + col]));
      }
    });
  });
  return output;
}

void norm_rope_cpu(
    at::Tensor input,
    const std::optional<at::Tensor>& weight,
    const at::Tensor& freqs,
    const at::Tensor& positions,
    const std::optional<double>& eps,
    bool inverse) {
  TORCH_CHECK(
      input.device().is_cpu() && input.dim() >= 2 && input.dim() <= 3 && input.stride(-1) == 1,
      "RoPE input must be a CPU [T, D] or [T, H, D] tensor with contiguous features");
  CHECK_INPUT(freqs);
  CHECK_INPUT(positions);
  TORCH_CHECK(
      freqs.dim() == 2 && freqs.scalar_type() == at::kComplexFloat && input.size(-1) >= 2 * freqs.size(1),
      "Invalid RoPE frequencies");
  TORCH_CHECK(positions.dim() == 1 && positions.size(0) == input.size(0), "Invalid RoPE positions");
  if (weight.has_value()) {
    CHECK_INPUT(weight.value());
    TORCH_CHECK(eps.has_value() && weight->numel() == input.size(-1), "Invalid norm weight");
  }
  auto pos = positions.to(at::kLong);
  auto w = weight.has_value() ? weight->to(at::kFloat) : at::Tensor();
  const int64_t tokens = input.size(0), heads = input.dim() == 3 ? input.size(1) : 1;
  const int64_t dim = input.size(-1), rope_dim = 2 * freqs.size(1);
  const int64_t head_stride = input.dim() == 3 ? input.stride(1) : 0;
  const auto* f = freqs.const_data_ptr<c10::complex<float>>();
  const auto* p = pos.const_data_ptr<int64_t>();
  for (int64_t t = 0; t < tokens; ++t) {
    TORCH_CHECK(p[t] >= 0 && p[t] < freqs.size(0), "RoPE position out of range");
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, input.scalar_type(), "dsv4_norm_rope_cpu", [&] {
    auto* data = input.data_ptr<scalar_t>();
    at::parallel_for(0, tokens * heads, 0, [&](int64_t begin, int64_t end) {
      std::vector<float> values(dim);
      for (int64_t row = begin; row < end; ++row) {
        const int64_t token = row / heads, head = row % heads;
        auto* x = data + token * input.stride(0) + head * head_stride;
        float sum = 0.f;
        for (int64_t d = 0; d < dim; ++d) {
          values[d] = static_cast<float>(x[d]);
          sum += values[d] * values[d];
        }
        if (eps.has_value()) {
          const float rstd = 1.f / std::sqrt(sum / dim + static_cast<float>(*eps));
          for (int64_t d = 0; d < dim; ++d) {
            values[d] *= rstd * (w.defined() ? w.const_data_ptr<float>()[d] : 1.f);
          }
        }
        for (int64_t d = 0; d < rope_dim; d += 2) {
          const auto freq = f[p[token] * freqs.size(1) + d / 2];
          const float cos = freq.real(), sin = inverse ? -freq.imag() : freq.imag();
          const int64_t offset = dim - rope_dim + d;
          const float a = values[offset], b = values[offset + 1];
          values[offset] = a * cos - b * sin;
          values[offset + 1] = a * sin + b * cos;
        }
        for (int64_t d = 0; d < dim; ++d) {
          x[d] = static_cast<scalar_t>(values[d]);
        }
      }
    });
  });
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "dsv41_sparse_attention_cpu(Tensor query, Tensor swa, Tensor extra, Tensor swa_slots, Tensor extra_slots, "
      "Tensor sink, float scale) -> Tensor");
  m.impl("dsv41_sparse_attention_cpu", torch::kCPU, &sparse_attention_cpu);
  m.def("dsv41_sqrtsoftplus_topk_cpu(Tensor logits, Tensor? bias, int topk, bool renormalize) -> (Tensor, Tensor)");
  m.impl("dsv41_sqrtsoftplus_topk_cpu", torch::kCPU, &sqrtsoftplus_topk_cpu);
  m.def("silu_and_mul_clamp_cpu(Tensor input, Tensor(a!) output, float limit) -> ()");
  m.impl("silu_and_mul_clamp_cpu", torch::kCPU, &silu_and_mul_clamp_cpu);
  m.def(
      "dsv4_hc_split_sinkhorn_cpu(Tensor mixes, Tensor scale, Tensor base, int hc, int iters, float eps) -> (Tensor, "
      "Tensor, Tensor)");
  m.impl("dsv4_hc_split_sinkhorn_cpu", torch::kCPU, &hc_split_sinkhorn_cpu);
  m.def("dsv4_hc_combine_cpu(Tensor input, Tensor pre, int hc) -> Tensor");
  m.impl("dsv4_hc_combine_cpu", torch::kCPU, &hc_combine_cpu);
  m.def("dsv4_hc_post_cpu(Tensor input, Tensor residual, Tensor post, Tensor comb) -> Tensor");
  m.impl("dsv4_hc_post_cpu", torch::kCPU, &hc_post_cpu);
  m.def(
      "dsv4_norm_rope_cpu(Tensor(a!) input, Tensor? weight, Tensor freqs, Tensor positions, float? eps, bool inverse) "
      "-> ()");
  m.impl("dsv4_norm_rope_cpu", torch::kCPU, &norm_rope_cpu);
}
