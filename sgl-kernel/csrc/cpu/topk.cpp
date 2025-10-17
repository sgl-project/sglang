#include "common.h"
#include "vec.h"

namespace {

template <typename scalar_t, int SIZE>
inline void softmax(float* __restrict__ out, const scalar_t* __restrict__ input) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int kVecSize = bVec::size();

  // step 1: get max
  fVec max_fvec = fVec(-std::numeric_limits<float>::infinity());
  if constexpr (SIZE < kVecSize) {
    // SIZE = 1, 2, 4, 8, 16; only the top half is used
    bVec x_bvec = bVec::loadu(input, SIZE);
    fVec x_fvec0, x_fvec1;
    std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);
    x_fvec0 = fVec::set(max_fvec, x_fvec0, SIZE);
    max_fvec = at::vec::maximum(max_fvec, x_fvec0);
    x_fvec0.store(out, SIZE);
  } else {
    for (int d = 0; d < SIZE; d += kVecSize) {
      bVec x_bvec = bVec::loadu(input + d);
      fVec x_fvec0, x_fvec1;
      std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

      max_fvec = at::vec::maximum(max_fvec, x_fvec0);
      max_fvec = at::vec::maximum(max_fvec, x_fvec1);
      x_fvec0.store(out + d);
      x_fvec1.store(out + d + fVec::size());
    }
  }
  float max_val = vec_reduce_max(max_fvec);
  max_fvec = fVec(max_val);

  // step 2: sum of (x - max).exp()
  fVec sum_fvec = fVec(float(0));
  if constexpr (SIZE < fVec::size()) {
    // SIZE = 1, 2, 4, 8
    fVec x_fvec = (fVec::loadu(out, SIZE) - max_fvec).exp_u20();
    x_fvec = fVec::set(sum_fvec, x_fvec, SIZE);
    sum_fvec += x_fvec;
    x_fvec.store(out, SIZE);
  } else {
    for (int d = 0; d < SIZE; d += fVec::size()) {
      fVec x_fvec = (fVec::loadu(out + d) - max_fvec).exp_u20();
      sum_fvec += x_fvec;
      x_fvec.store(out + d);
    }
  }
  float sum_val = vec_reduce_sum(sum_fvec);

  // step 3: x * (1 / sum)
  sum_fvec = fVec(1.f / sum_val);
  if constexpr (SIZE < fVec::size()) {
    // SIZE = 1, 2, 4, 8
    fVec out_fvec = fVec::loadu(out, SIZE) * sum_fvec;
    out_fvec.store(out, SIZE);
  } else {
    for (int d = 0; d < SIZE; d += fVec::size()) {
      fVec out_fvec = fVec::loadu(out + d) * sum_fvec;
      out_fvec.store(out + d);
    }
  }
}

template <typename scalar_t, int NUM_EXPERTS>
void grouped_topk_kernel_impl(
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    const scalar_t* __restrict__ gating_output,
    int64_t num_tokens,
    int64_t topk,
    int64_t num_groups,
    int64_t topk_group,
    bool renormalize) {
  const int64_t num_experts_per_group = NUM_EXPERTS / num_groups;
  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    alignas(64) float scores[NUM_EXPERTS];

    using elem_t = std::pair<float, int32_t>;
    std::vector<elem_t> queue(num_groups);
    std::vector<elem_t> queue2(topk_group * num_experts_per_group);

    for (int64_t i = begin; i < end; ++i) {
      // do softmax to get scores
      softmax<scalar_t, NUM_EXPERTS>(scores, gating_output + i * NUM_EXPERTS);

      // find max score per group
      for (int64_t g = 0; g < num_groups; ++g) {
        float gmax = -std::numeric_limits<float>::infinity();
        for (int64_t e = 0; e < num_experts_per_group; ++e) {
          gmax = std::max(gmax, scores[g * num_experts_per_group + e]);
        }
        queue[g] = {gmax, g};
      }

      // find group topk
      std::partial_sort(
          queue.begin(), queue.begin() + topk_group, queue.end(), [](const elem_t& x, const elem_t& y) -> bool {
            return x.first > y.first;
          });

      for (int64_t g = 0; g < topk_group; ++g) {
        int32_t group_idx = queue[g].second;
        for (int64_t e = 0; e < num_experts_per_group; ++e) {
          int32_t expert_idx = group_idx * num_experts_per_group + e;
          queue2[g * num_experts_per_group + e] = {scores[expert_idx], expert_idx};
        }
      }

      // find global topk
      std::partial_sort(
          queue2.begin(), queue2.begin() + topk, queue2.end(), [](const elem_t& x, const elem_t& y) -> bool {
            return x.first > y.first;
          });

      for (int64_t j = 0; j < topk; ++j) {
        topk_weights[i * topk + j] = queue2[j].first;
        topk_ids[i * topk + j] = queue2[j].second;
      }

      if (renormalize) {
        float sum = 0.f;
        for (int64_t j = 0; j < topk; ++j) {
          sum += topk_weights[i * topk + j];
        }
        float scale = 1.f / sum;
        for (int64_t j = 0; j < topk; ++j) {
          topk_weights[i * topk + j] *= scale;
        }
      }
    }
  });
}

template <typename scalar_t, int SIZE>
inline void sigmoid(float* __restrict__ out, const scalar_t* __restrict__ input) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  const fVec one = fVec(1.f);

  constexpr int kVecSize = bVec::size();
  for (int d = 0; d < SIZE; d += kVecSize) {
    bVec x_bvec = bVec::loadu(input + d);
    fVec x_fvec0, x_fvec1;
    std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

    x_fvec0 = one / (one + x_fvec0.neg().exp_u20());
    x_fvec1 = one / (one + x_fvec1.neg().exp_u20());

    x_fvec0.store(out + d);
    x_fvec1.store(out + d + fVec::size());
  }
}

template <typename scalar_t, int NUM_EXPERTS>
void topk_sigmoid_kernel_impl(
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    const scalar_t* __restrict__ gating_output,
    int64_t num_tokens,
    int64_t topk,
    bool renormalize) {
  using Vec = at::vec::Vectorized<float>;
  const int64_t num_experts_per_group = NUM_EXPERTS;
  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    alignas(64) float scores[NUM_EXPERTS];
    using elem_t = std::pair<float, int32_t>;
    std::vector<elem_t> queue(num_experts_per_group);

    for (int64_t i = begin; i < end; ++i) {
      at::vec::convert<scalar_t, float>(gating_output + i * NUM_EXPERTS, scores, NUM_EXPERTS);

      float gmax = at::vec::reduce_all<float>(
          [](Vec& x, Vec& y) { return at::vec::maximum(x, y); }, scores, num_experts_per_group);

      // find position of first max,
      // note that we may have multiple max values.
      int first_max_idx = -1;
      for (int64_t e = 0; e < num_experts_per_group; ++e) {
        if (scores[e] == gmax) {
          first_max_idx = e;
          break;
        }
      }

      // scalar sigmoid
      topk_weights[i] = 1.0 / (1.0 + exp(0.0 - gmax));
      topk_ids[i] = first_max_idx;

      if (renormalize) {
        float sum = 0.f;
        for (int64_t j = 0; j < topk; ++j) {
          sum += topk_weights[i * topk + j];
        }
        float scale = 1.f / sum;
        for (int64_t j = 0; j < topk; ++j) {
          topk_weights[i * topk + j] *= scale;
        }
      }
    }
  });
}

template <typename scalar_t, int NUM_EXPERTS>
void topk_softmax_kernel_impl(
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    const scalar_t* __restrict__ gating_output,
    int64_t num_tokens,
    int64_t topk,
    bool renormalize) {
  const int64_t num_experts_per_group = NUM_EXPERTS;
  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    alignas(64) float scores[NUM_EXPERTS];
    using elem_t = std::pair<float, int32_t>;
    std::vector<elem_t> queue(num_experts_per_group);

    for (int64_t i = begin; i < end; ++i) {
      softmax<scalar_t, NUM_EXPERTS>(scores, gating_output + i * NUM_EXPERTS);

      for (int64_t e = 0; e < num_experts_per_group; ++e) {
        queue[e] = {scores[e], e};
      }

      std::partial_sort(
          queue.begin(),
          queue.begin() + num_experts_per_group,
          queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool { return x.first > y.first; });

      for (int64_t j = 0; j < topk; ++j) {
        topk_weights[i * topk + j] = queue[j].first;
        topk_ids[i * topk + j] = queue[j].second;
      }

      if (renormalize) {
        float sum = 0.f;
        for (int64_t j = 0; j < topk; ++j) {
          sum += topk_weights[i * topk + j];
        }
        float scale = 1.f / sum;
        for (int64_t j = 0; j < topk; ++j) {
          topk_weights[i * topk + j] *= scale;
        }
      }
    }
  });
}

template <typename scalar_t, typename param_t, int SIZE>
inline void
apply_bias(float* __restrict__ scores2, const float* __restrict__ scores, const param_t* __restrict__ bias) {
  using fVec = at::vec::Vectorized<float>;
  using bVec = at::vec::Vectorized<scalar_t>;
  auto vec_size = bVec::size();
  int d = 0;
  for (; d <= SIZE - vec_size; d += vec_size) {
    fVec bias0, bias1, x0, x1;
    std::tie(bias0, bias1) = load_float_vec2(bias + d);
    std::tie(x0, x1) = load_float_vec2(scores + d);
    x0 = x0 + bias0;
    x1 = x1 + bias1;
    x0.store(scores2 + d);
    x1.store(scores2 + d + fVec::size());
  }
  for (; d < SIZE; d++) {
    scores2[d] = scores[d] + (float)bias[d];
  }
}

template <typename scalar_t, typename param_t, int NUM_EXPERTS, int TOPK>
void biased_grouped_topk_kernel_impl(
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    const scalar_t* __restrict__ gating_output,
    const param_t* __restrict__ bias,
    int64_t num_tokens,
    int64_t num_groups,
    int64_t topk_group,
    bool renormalize) {
  using Vec = at::vec::Vectorized<float>;

  const int64_t num_experts_per_group = NUM_EXPERTS / num_groups;
  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    // scores: sigmoid
    alignas(64) float scores[NUM_EXPERTS];
    // scores for choice: sigmoid + bias
    alignas(64) float scores2[NUM_EXPERTS];

    using elem_t = std::pair<float, int32_t>;
    std::vector<elem_t> queue(num_groups);
    std::vector<elem_t> queue2(topk_group * num_experts_per_group);

    for (int64_t i = begin; i < end; ++i) {
      // do sigmoid to get scores
      sigmoid<scalar_t, NUM_EXPERTS>(scores, gating_output + i * NUM_EXPERTS);

      apply_bias<scalar_t, param_t, NUM_EXPERTS>(scores2, scores, bias);

      for (int64_t g = 0; g < num_groups; ++g) {
        // find the max
        float gmax = at::vec::reduce_all<float>(
            [](Vec& x, Vec& y) { return at::vec::maximum(x, y); },
            scores2 + g * num_experts_per_group,
            num_experts_per_group);

        // find position of first max,
        // note that we may have multiple max values.
        int first_max_idx = -1;
        for (int64_t e = 0; e < num_experts_per_group; ++e) {
          if (scores2[g * num_experts_per_group + e] == gmax) {
            first_max_idx = g * num_experts_per_group + e;
            break;
          }
        }

        // find the 2nd max
        scores2[first_max_idx] = -std::numeric_limits<float>::infinity();
        float gmax2 = at::vec::reduce_all<float>(
            [](Vec& x, Vec& y) { return at::vec::maximum(x, y); },
            scores2 + g * num_experts_per_group,
            num_experts_per_group);
        // restore scores for choice
        scores2[first_max_idx] = gmax;

        queue[g] = {gmax + gmax2, g};
      }

      // find group topk
      std::partial_sort(
          queue.begin(), queue.begin() + topk_group, queue.end(), [](const elem_t& x, const elem_t& y) -> bool {
            return x.first > y.first;
          });

      for (int64_t g = 0; g < topk_group; ++g) {
        int32_t group_idx = queue[g].second;
        for (int64_t e = 0; e < num_experts_per_group; ++e) {
          int32_t expert_idx = group_idx * num_experts_per_group + e;
          queue2[g * num_experts_per_group + e] = {scores2[expert_idx], expert_idx};
        }
      }

      // find global topk
      std::partial_sort(
          queue2.begin(), queue2.begin() + TOPK, queue2.end(), [](const elem_t& x, const elem_t& y) -> bool {
            return x.first > y.first;
          });

      for (int j = 0; j < TOPK; ++j) {
        int32_t index = queue2[j].second;
        topk_ids[i * TOPK + j] = index;
        topk_weights[i * TOPK + j] = scores[index];
      }

#if defined(CPU_CAPABILITY_AVX512)
      if (renormalize) {
        __mmask16 mask = (1ULL << TOPK) - 1;
        __m512 x = _mm512_maskz_loadu_ps(mask, topk_weights + i * TOPK);
        float sum = _mm512_reduce_add_ps(x);
        __m512 vscale = _mm512_set1_ps(1.f / sum);
        __m512 y = _mm512_mul_ps(x, vscale);
        _mm512_mask_storeu_ps(topk_weights + i * TOPK, mask, y);
      }
#else
      if (renormalize) {
        float sum = 0.f;
        for (int64_t j = 0; j < TOPK; ++j) {
          sum += topk_weights[i * TOPK + j];
        }
        float scale = 1.f / sum;
        for (int64_t j = 0; j < TOPK; ++j) {
          topk_weights[i * TOPK + j] *= scale;
        }
      }
#endif
    }
  });
}

#define LAUNCH_GROUPED_TOPK_KERNEL(NE)    \
  grouped_topk_kernel_impl<scalar_t, NE>( \
      topk_weights.data_ptr<float>(),     \
      topk_ids.data_ptr<int32_t>(),       \
      gating_output.data_ptr<scalar_t>(), \
      num_tokens,                         \
      topk,                               \
      num_expert_group,                   \
      topk_group,                         \
      renormalize);

#define LAUNCH_TOPK_SIGMOID_KERNEL(NE)    \
  topk_sigmoid_kernel_impl<scalar_t, NE>( \
      topk_weights.data_ptr<float>(),     \
      topk_ids.data_ptr<int32_t>(),       \
      gating_output.data_ptr<scalar_t>(), \
      num_tokens,                         \
      topk,                               \
      renormalize);

#define LAUNCH_TOPK_SOFTMAX_KERNEL(NE)    \
  topk_softmax_kernel_impl<scalar_t, NE>( \
      topk_weights.data_ptr<float>(),     \
      topk_ids.data_ptr<int32_t>(),       \
      gating_output.data_ptr<scalar_t>(), \
      num_tokens,                         \
      topk,                               \
      renormalize);

#define LAUNCH_BIASED_GROUPED_TOPK_KERNEL(NE, NTOPK)             \
  biased_grouped_topk_kernel_impl<scalar_t, param_t, NE, NTOPK>( \
      topk_weights.data_ptr<float>(),                            \
      topk_ids.data_ptr<int32_t>(),                              \
      gating_output.data_ptr<scalar_t>(),                        \
      correction_bias.data_ptr<param_t>(),                       \
      num_tokens,                                                \
      num_expert_group,                                          \
      topk_group,                                                \
      renormalize);

}  // anonymous namespace

std::tuple<at::Tensor, at::Tensor>
topk_sigmoid_cpu(at::Tensor& hidden_states, at::Tensor& gating_output, int64_t topk, bool renormalize) {
  RECORD_FUNCTION("sgl-kernel::topk_sigmoid_cpu", std::vector<c10::IValue>({hidden_states, gating_output}));
  CHECK_INPUT(gating_output);

  const auto st = hidden_states.scalar_type();
  CHECK_EQ(gating_output.scalar_type(), st);

  int64_t num_tokens = hidden_states.size(0);
  int64_t num_experts = gating_output.size(1);
  TORCH_CHECK(gating_output.size(0) == num_tokens, "Number of tokens mismatch");
  TORCH_CHECK(topk == 1, "topk_sigmoid only supports topk=1 case");
  at::Tensor topk_weights = at::empty({num_tokens, topk}, hidden_states.options().dtype(at::kFloat));
  at::Tensor topk_ids = at::empty({num_tokens, topk}, hidden_states.options().dtype(at::kInt));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "topk_sigmoid_kernel", [&] {
    switch (num_experts) {
      case 1:
        LAUNCH_TOPK_SIGMOID_KERNEL(1);
        break;
      case 2:
        LAUNCH_TOPK_SIGMOID_KERNEL(2);
        break;
      case 4:
        LAUNCH_TOPK_SIGMOID_KERNEL(4);
        break;
      case 8:
        LAUNCH_TOPK_SIGMOID_KERNEL(8);
        break;
      case 16:
        LAUNCH_TOPK_SIGMOID_KERNEL(16);
        break;
      case 32:
        LAUNCH_TOPK_SIGMOID_KERNEL(32);
        break;
      case 64:
        LAUNCH_TOPK_SIGMOID_KERNEL(64);
        break;
      case 128:
        LAUNCH_TOPK_SIGMOID_KERNEL(128);
        break;
      case 160:
        LAUNCH_TOPK_SIGMOID_KERNEL(160);
        break;
      case 256:
        LAUNCH_TOPK_SIGMOID_KERNEL(256);
        break;
      default:
        TORCH_CHECK(false, "Unexpected num_experts: ", num_experts);
    }
  });
  return std::make_tuple(topk_weights, topk_ids);
}

std::tuple<at::Tensor, at::Tensor>
topk_softmax_cpu(at::Tensor& hidden_states, at::Tensor& gating_output, int64_t topk, bool renormalize) {
  RECORD_FUNCTION("sgl-kernel::topk_softmax_cpu", std::vector<c10::IValue>({hidden_states, gating_output}));
  CHECK_INPUT(gating_output);

  const auto st = hidden_states.scalar_type();
  CHECK_EQ(gating_output.scalar_type(), st);

  int64_t num_tokens = hidden_states.size(0);
  int64_t num_experts = gating_output.size(1);
  TORCH_CHECK(gating_output.size(0) == num_tokens, "Number of tokens mismatch");

  at::Tensor topk_weights = at::empty({num_tokens, topk}, hidden_states.options().dtype(at::kFloat));
  at::Tensor topk_ids = at::empty({num_tokens, topk}, hidden_states.options().dtype(at::kInt));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "topk_softmax_cpu", [&] {
    switch (num_experts) {
      case 1:
        LAUNCH_TOPK_SOFTMAX_KERNEL(1);
        break;
      case 2:
        LAUNCH_TOPK_SOFTMAX_KERNEL(2);
        break;
      case 4:
        LAUNCH_TOPK_SOFTMAX_KERNEL(4);
        break;
      case 8:
        LAUNCH_TOPK_SOFTMAX_KERNEL(8);
        break;
      case 16:
        LAUNCH_TOPK_SOFTMAX_KERNEL(16);
        break;
      case 32:
        LAUNCH_TOPK_SOFTMAX_KERNEL(32);
        break;
      case 64:
        LAUNCH_TOPK_SOFTMAX_KERNEL(64);
        break;
      case 128:
        LAUNCH_TOPK_SOFTMAX_KERNEL(128);
        break;
      case 160:
        LAUNCH_TOPK_SOFTMAX_KERNEL(160);
        break;
      case 256:
        LAUNCH_TOPK_SOFTMAX_KERNEL(256);
        break;
      default:
        TORCH_CHECK(false, "Unexpected num_experts: ", num_experts);
    }
  });
  return std::make_tuple(topk_weights, topk_ids);
}

// grouped topk for DeepSeek V2
std::tuple<at::Tensor, at::Tensor> grouped_topk_cpu(
    at::Tensor& hidden_states,
    at::Tensor& gating_output,
    int64_t topk,
    bool renormalize,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t num_fused_shared_experts,
    std::optional<double> routed_scaling_factor,
    std::optional<at::Tensor> num_token_non_padded) {
  // TODO: Will support num_fused_shared_experts, routed_scaling_factor and num_token_non_padded.
  // For now, we just check them as default value.
  TORCH_CHECK(
      num_fused_shared_experts == 0,
      "num_fused_shared_experts must be 0 default value, got: ",
      num_fused_shared_experts);
  TORCH_CHECK(
      !routed_scaling_factor.has_value() || routed_scaling_factor.value() == 1.0f,
      "routed_scaling_factor must be None or 1.0f default value, got: ",
      routed_scaling_factor.value());
  TORCH_CHECK(
      !num_token_non_padded.has_value(),
      "num_token_non_padded must be None default value, got: ",
      num_token_non_padded.value());

  RECORD_FUNCTION("sgl-kernel::grouped_topk_cpu", std::vector<c10::IValue>({hidden_states, gating_output}));
  CHECK_INPUT(gating_output);

  const auto st = hidden_states.scalar_type();
  CHECK_EQ(gating_output.scalar_type(), st);

  int64_t num_tokens = hidden_states.size(0);
  int64_t num_experts = gating_output.size(1);
  TORCH_CHECK(gating_output.size(0) == num_tokens, "Number of tokens mismatch");
  at::Tensor topk_weights = at::empty({num_tokens, topk}, hidden_states.options().dtype(at::kFloat));
  at::Tensor topk_ids = at::empty({num_tokens, topk}, hidden_states.options().dtype(at::kInt));

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "grouped_topk_kernel", [&] {
    switch (num_experts) {
      case 1:
        LAUNCH_GROUPED_TOPK_KERNEL(1);
        break;
      case 2:
        LAUNCH_GROUPED_TOPK_KERNEL(2);
        break;
      case 4:
        LAUNCH_GROUPED_TOPK_KERNEL(4);
        break;
      case 8:
        LAUNCH_GROUPED_TOPK_KERNEL(8);
        break;
      case 16:
        LAUNCH_GROUPED_TOPK_KERNEL(16);
        break;
      case 32:
        LAUNCH_GROUPED_TOPK_KERNEL(32);
        break;
      case 64:
        LAUNCH_GROUPED_TOPK_KERNEL(64);
        break;
      case 128:
        LAUNCH_GROUPED_TOPK_KERNEL(128);
        break;
      case 160:
        LAUNCH_GROUPED_TOPK_KERNEL(160);
        break;
      case 256:
        LAUNCH_GROUPED_TOPK_KERNEL(256);
        break;
      default:
        TORCH_CHECK(false, "Unexpected num_experts: ", num_experts);
    }
  });
  return std::make_tuple(topk_weights, topk_ids);
}

// biased grouped topk DeepSeek V3/R1
std::tuple<at::Tensor, at::Tensor> biased_grouped_topk_cpu(
    at::Tensor& hidden_states,
    at::Tensor& gating_output,
    at::Tensor& correction_bias,
    int64_t topk,
    bool renormalize,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t num_fused_shared_experts,
    std::optional<double> routed_scaling_factor,
    std::optional<at::Tensor> num_token_non_padded) {
  // TODO: Will support num_fused_shared_experts, routed_scaling_factor and num_token_non_padded.
  // For now, we just check them as default value.
  TORCH_CHECK(
      num_fused_shared_experts == 0,
      "num_fused_shared_experts must be 0 default value, got: ",
      num_fused_shared_experts);
  TORCH_CHECK(
      !num_token_non_padded.has_value(),
      "num_token_non_padded must be None default value, got: ",
      num_token_non_padded.value());

  RECORD_FUNCTION(
      "sgl-kernel::biased_grouped_topk_cpu", std::vector<c10::IValue>({hidden_states, gating_output, correction_bias}));

  CHECK_INPUT(gating_output);
  CHECK_INPUT(correction_bias);

  const auto st = hidden_states.scalar_type();
  CHECK_EQ(gating_output.scalar_type(), st);

  int64_t num_tokens = hidden_states.size(0);
  int64_t num_experts = gating_output.size(1);
  TORCH_CHECK(gating_output.size(0) == num_tokens, "Number of tokens mismatch");
  TORCH_CHECK(correction_bias.numel() == num_experts, "Bias shape mismatch");
  at::Tensor topk_weights = at::empty({num_tokens, topk}, hidden_states.options().dtype(at::kFloat));
  at::Tensor topk_ids = at::empty({num_tokens, topk}, hidden_states.options().dtype(at::kInt));

  CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT(st, correction_bias.scalar_type(), "biased_grouped_topk_kernel", [&] {
    TORCH_CHECK(topk == 8, "Unexpected topk: ", topk);
    switch (num_experts) {
      case 256:
        LAUNCH_BIASED_GROUPED_TOPK_KERNEL(256, 8);
        break;
      default:
        TORCH_CHECK(false, "Unexpected num_experts: ", num_experts);
    }
  });
  return std::make_tuple(topk_weights, topk_ids);
}
