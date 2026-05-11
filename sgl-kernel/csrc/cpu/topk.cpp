#include "common.h"
#include "vec.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr int C4_TOPK = 512;

template <typename T>
inline int64_t load_index_value(const T* __restrict__ ptr, int64_t idx) {
  return static_cast<int64_t>(ptr[idx]);
}

struct TopKTransformElem {
  float score;
  int32_t index;
};

struct TopKTransformMinHeapCmp {
  bool operator()(const TopKTransformElem& lhs, const TopKTransformElem& rhs) const {
    if (lhs.score == rhs.score) {
      return lhs.index > rhs.index;
    }
    return lhs.score > rhs.score;
  }
};

template <typename seq_t, typename page_t>
void topk_transform_512_cpu_kernel_impl(
    const float* __restrict__ scores,
    const seq_t* __restrict__ seq_lens,
    const page_t* __restrict__ page_tables,
    int32_t* __restrict__ out_page_indices,
    int32_t* __restrict__ out_raw_indices,
    int64_t batch_size,
    int64_t max_seq_len,
    int64_t page_table_stride,
    int64_t out_stride,
    int64_t page_size) {
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK((page_size & (page_size - 1)) == 0, "page_size must be a power of 2");
  const int page_bits = page_size > 1 ? static_cast<int>(std::log2(static_cast<double>(page_size))) : 0;
  const int64_t page_mask = page_size - 1;

  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    std::vector<TopKTransformElem> heap;
    heap.reserve(C4_TOPK);

    for (int64_t b = begin; b < end; ++b) {
      const float* __restrict__ scores_row = scores + b * max_seq_len;
      const page_t* __restrict__ page_table_row = page_tables + b * page_table_stride;
      int32_t* __restrict__ out_page_row = out_page_indices + b * out_stride;
      int32_t* __restrict__ out_raw_row = out_raw_indices == nullptr ? nullptr : out_raw_indices + b * out_stride;

      int64_t seq_len = load_index_value(seq_lens, b);
      seq_len = std::max<int64_t>(0, std::min<int64_t>(seq_len, max_seq_len));
      const int64_t valid_topk = std::min<int64_t>(seq_len, C4_TOPK);

      auto store_slot = [&](int64_t slot, int32_t raw_index) {
        if (raw_index < 0) {
          out_page_row[slot] = -1;
          if (out_raw_row != nullptr) {
            out_raw_row[slot] = -1;
          }
          return;
        }

        const int64_t page_idx = static_cast<int64_t>(raw_index) >> page_bits;
        const int64_t offset_in_page = static_cast<int64_t>(raw_index) & page_mask;
        const int64_t physical_page = load_index_value(page_table_row, page_idx);
        out_page_row[slot] = static_cast<int32_t>((physical_page << page_bits) | offset_in_page);
        if (out_raw_row != nullptr) {
          out_raw_row[slot] = raw_index;
        }
      };

      if (seq_len <= C4_TOPK) {
        for (int64_t i = 0; i < valid_topk; ++i) {
          store_slot(i, static_cast<int32_t>(i));
        }
        for (int64_t i = valid_topk; i < C4_TOPK; ++i) {
          store_slot(i, -1);
        }
        continue;
      }

      heap.clear();
      for (int64_t i = 0; i < C4_TOPK; ++i) {
        heap.push_back({scores_row[i], static_cast<int32_t>(i)});
      }
      std::make_heap(heap.begin(), heap.end(), TopKTransformMinHeapCmp());

      for (int64_t i = C4_TOPK; i < seq_len; ++i) {
        const float score = scores_row[i];
        const TopKTransformElem& current_min = heap.front();
        if (score > current_min.score || (score == current_min.score && static_cast<int32_t>(i) < current_min.index)) {
          std::pop_heap(heap.begin(), heap.end(), TopKTransformMinHeapCmp());
          heap.back() = {score, static_cast<int32_t>(i)};
          std::push_heap(heap.begin(), heap.end(), TopKTransformMinHeapCmp());
        }
      }

      for (int64_t i = 0; i < C4_TOPK; ++i) {
        store_slot(i, heap[i].index);
      }
    }
  });
}

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

template <typename scalar_t, int SIZE, std::enable_if_t<!std::is_same_v<scalar_t, float>, int> = 0>
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

template <typename scalar_t, int SIZE, std::enable_if_t<std::is_same_v<scalar_t, float>, int> = 0>
inline void sigmoid(float* __restrict__ out, const float* __restrict__ input) {
  using fVec = at::vec::Vectorized<float>;
  const fVec one = fVec(1.f);
  constexpr int kVecSize = fVec::size();
  for (int d = 0; d < SIZE; d += kVecSize) {
    fVec in_fvec = fVec::loadu(input + d);
    in_fvec = one / (one + in_fvec.neg().exp_u20());
    in_fvec.store(out + d);
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

      std::partial_sort(queue.begin(), queue.begin() + topk, queue.end(), [](const elem_t& x, const elem_t& y) -> bool {
        return x.first > y.first;
      });

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

template <typename param_t, int SIZE>
inline void
apply_bias(float* __restrict__ scores2, const float* __restrict__ scores, const param_t* __restrict__ bias) {
  using fVec = at::vec::Vectorized<float>;
  auto vec_size = fVec::size() * 2;
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
    scalar_t* __restrict__ gating_output,
    const param_t* __restrict__ bias,
    float scaling_factor_value,
    int64_t num_tokens,
    int64_t num_groups,
    int64_t topk_group,
    bool renormalize) {
  using Vec = at::vec::Vectorized<float>;

  bool apply_scaling_factor = scaling_factor_value != 1.0f;
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
      apply_bias<param_t, NUM_EXPERTS>(scores2, scores, bias);

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
      if (renormalize || apply_scaling_factor) {
        __mmask16 mask = (1ULL << TOPK) - 1;
        __m512 x = _mm512_maskz_loadu_ps(mask, topk_weights + i * TOPK);
        if (renormalize) {
          float sum = _mm512_reduce_add_ps(x);
          __m512 vscale = _mm512_set1_ps(scaling_factor_value / sum);
          __m512 y = _mm512_mul_ps(x, vscale);
          _mm512_mask_storeu_ps(topk_weights + i * TOPK, mask, y);
        } else {
          __m512 vscale = _mm512_set1_ps(scaling_factor_value);
          __m512 y = _mm512_mul_ps(x, vscale);
          _mm512_mask_storeu_ps(topk_weights + i * TOPK, mask, y);
        }
      }
#else
      if (renormalize || apply_scaling_factor){
        if (renormalize) {
          float sum = 0.f;
          for (int64_t j = 0; j < TOPK; ++j) {
            sum += topk_weights[i * TOPK + j];
          }
          float scale = scaling_factor_value / sum;
          for (int64_t j = 0; j < TOPK; ++j) {
            topk_weights[i * TOPK + j] *= scale;
          }
        }else{
          for (int64_t j = 0; j < TOPK; ++j) {
            topk_weights[i * TOPK + j] *= scaling_factor_value;
          }
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
      scaling_factor_value,                                      \
      num_tokens,                                                \
      num_expert_group,                                          \
      topk_group,                                                \
      renormalize);

}  // anonymous namespace

void topk_transform_512_cpu(
    at::Tensor& scores,
    at::Tensor& seq_lens,
    at::Tensor& page_tables,
    at::Tensor& out_page_indices,
    int64_t page_size,
    const std::optional<at::Tensor>& out_raw_indices) {
  CHECK_INPUT(scores);
  CHECK_INPUT(seq_lens);
  CHECK_INPUT(page_tables);
  CHECK_INPUT(out_page_indices);

  TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be float32");
  TORCH_CHECK(out_page_indices.scalar_type() == at::kInt, "out_page_indices must be int32");
  TORCH_CHECK(scores.dim() == 2, "scores must be a 2D tensor");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be a 1D tensor");
  TORCH_CHECK(page_tables.dim() == 2, "page_tables must be a 2D tensor");
  TORCH_CHECK(out_page_indices.dim() == 2, "out_page_indices must be a 2D tensor");

  const int64_t batch_size = scores.size(0);
  const int64_t max_seq_len = scores.size(1);
  TORCH_CHECK(seq_lens.size(0) == batch_size, "seq_lens row count must match scores");
  TORCH_CHECK(page_tables.size(0) == batch_size, "page_tables row count must match scores");
  TORCH_CHECK(out_page_indices.size(0) == batch_size, "out_page_indices row count must match scores");
  TORCH_CHECK(out_page_indices.size(1) >= C4_TOPK, "out_page_indices must have at least 512 columns");

  int32_t* raw_ptr = nullptr;
  if (out_raw_indices.has_value()) {
    at::Tensor raw = out_raw_indices.value();
    CHECK_INPUT(raw);
    TORCH_CHECK(raw.scalar_type() == at::kInt, "out_raw_indices must be int32");
    TORCH_CHECK(raw.dim() == 2, "out_raw_indices must be a 2D tensor");
    TORCH_CHECK(raw.sizes() == out_page_indices.sizes(), "out_raw_indices shape must match out_page_indices");
    raw_ptr = raw.data_ptr<int32_t>();
  }

  auto launch_with_page_type = [&]<typename seq_t>() {
    if (page_tables.scalar_type() == at::kInt) {
      topk_transform_512_cpu_kernel_impl<seq_t, int32_t>(
          scores.data_ptr<float>(),
          seq_lens.data_ptr<seq_t>(),
          page_tables.data_ptr<int32_t>(),
          out_page_indices.data_ptr<int32_t>(),
          raw_ptr,
          batch_size,
          max_seq_len,
          page_tables.stride(0),
          out_page_indices.stride(0),
          page_size);
    } else if (page_tables.scalar_type() == at::kLong) {
      topk_transform_512_cpu_kernel_impl<seq_t, int64_t>(
          scores.data_ptr<float>(),
          seq_lens.data_ptr<seq_t>(),
          page_tables.data_ptr<int64_t>(),
          out_page_indices.data_ptr<int32_t>(),
          raw_ptr,
          batch_size,
          max_seq_len,
          page_tables.stride(0),
          out_page_indices.stride(0),
          page_size);
    } else {
      TORCH_CHECK(false, "page_tables must be int32 or int64");
    }
  };

  if (seq_lens.scalar_type() == at::kInt) {
    launch_with_page_type.template operator()<int32_t>();
  } else if (seq_lens.scalar_type() == at::kLong) {
    launch_with_page_type.template operator()<int64_t>();
  } else {
    TORCH_CHECK(false, "seq_lens must be int32 or int64");
  }
}


std::tuple<at::Tensor, at::Tensor>
topk_sigmoid_cpu(at::Tensor& hidden_states, at::Tensor& gating_output, int64_t topk, bool renormalize) {
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
      case 384:
        LAUNCH_TOPK_SOFTMAX_KERNEL(384);
        break;
      case 512:
        LAUNCH_TOPK_SOFTMAX_KERNEL(512);
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
  // TODO: Will support num_fused_shared_experts and num_token_non_padded.
  // For now, we just check them as default value.
  TORCH_CHECK(
      num_fused_shared_experts == 0,
      "num_fused_shared_experts must be 0 default value, got: ",
      num_fused_shared_experts);
  TORCH_CHECK(
      !num_token_non_padded.has_value(),
      "num_token_non_padded must be None default value, got: ",
      num_token_non_padded.value());

  CHECK_INPUT(gating_output);
  CHECK_INPUT(correction_bias);

  const auto st = gating_output.scalar_type();
  int64_t num_tokens = hidden_states.size(0);
  int64_t num_experts = gating_output.size(1);
  TORCH_CHECK(gating_output.size(0) == num_tokens, "Number of tokens mismatch");
  TORCH_CHECK(correction_bias.numel() == num_experts, "Bias shape mismatch");
  at::Tensor topk_weights = at::empty({num_tokens, topk}, hidden_states.options().dtype(at::kFloat));
  at::Tensor topk_ids = at::empty({num_tokens, topk}, hidden_states.options().dtype(at::kInt));
  float scaling_factor_value = routed_scaling_factor.has_value() ? routed_scaling_factor.value() : 1.0f;

  CPU_DISPATCH_FLOATING_TYPES_EXT(st, correction_bias.scalar_type(), "biased_grouped_topk_kernel", [&] {
    TORCH_CHECK(topk == 8, "Unexpected topk: ", topk);
    switch (num_experts) {
      case 128:
        LAUNCH_BIASED_GROUPED_TOPK_KERNEL(128, 8);
        break;
      case 192:
        LAUNCH_BIASED_GROUPED_TOPK_KERNEL(192, 8);
        break;
      case 256:
        LAUNCH_BIASED_GROUPED_TOPK_KERNEL(256, 8);
        break;
      case 384:
        LAUNCH_BIASED_GROUPED_TOPK_KERNEL(384, 8);
        break;
      default:
        TORCH_CHECK(false, "Unexpected num_experts: ", num_experts);
    }
  });
  return std::make_tuple(topk_weights, topk_ids);
}
