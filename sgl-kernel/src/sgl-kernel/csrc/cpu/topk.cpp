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
      fVec x_fvec= (fVec::loadu(out + d) - max_fvec).exp_u20();
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
    int num_tokens,
    int topk,
    int num_groups,
    int topk_group,
    bool renormalize) {

  const int num_experts_per_group = NUM_EXPERTS / num_groups;
  parallel_for(num_tokens, [&](int begin, int end) {
    static thread_local float scores[NUM_EXPERTS];

    using elem_t = std::pair<float, int32_t>;
    std::vector<elem_t> queue(num_groups);
    std::vector<elem_t> queue2(topk_group * num_experts_per_group);

    for (int i = begin; i < end; ++i) {
      // do softmax to get scores
      softmax<scalar_t, NUM_EXPERTS>(scores, gating_output + i * NUM_EXPERTS);

      // find max score per group
      for (int g = 0; g < num_groups; ++g) {
        float gmax = -std::numeric_limits<float>::infinity();
        for (int e = 0; e < num_experts_per_group; ++e) {
          gmax = std::max(gmax, scores[g * num_experts_per_group + e]);
        }
        queue[g] = {gmax, g};
      }

      // find group topk
      std::partial_sort(queue.begin(), queue.begin() + topk_group, queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            return x.first > y.first;
          });

      for (int g = 0; g < topk_group; ++g) {
        int32_t group_idx = queue[g].second;
        for (int e = 0; e < num_experts_per_group; ++e) {
          int32_t expert_idx = group_idx * num_experts_per_group + e;
          queue2[g * num_experts_per_group + e] = {scores[expert_idx], expert_idx};
        }
      }

      // find global topk
      std::partial_sort(queue2.begin(), queue2.begin() + topk, queue2.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            return x.first > y.first;
          });

      for (int j = 0; j < topk; ++j) {
        topk_weights[i * topk + j] = queue2[j].first;
        topk_ids[i * topk + j] = queue2[j].second;
      }

      if (renormalize) {
        float sum = 0.f;
        for (int j = 0; j < topk; ++j) {
          sum += topk_weights[i * topk + j];
        }
        float scale = 1.f / sum;
        for (int j = 0; j < topk; ++j) {
          topk_weights[i * topk + j] *= scale;
        }
      }
    }
  });
}

#define LAUNCH_GROUPED_TOPK_KERNEL(NE)                      \
    grouped_topk_kernel_impl<scalar_t, NE>(                 \
        topk_weights.data_ptr<float>(),                     \
        topk_ids.data_ptr<int32_t>(),                       \
        gating_output.data_ptr<scalar_t>(),                 \
        num_tokens,                                         \
        topk,                                               \
        num_expert_group,                                   \
        topk_group,                                         \
        renormalize);

} // anonymous namespace

// grouped topk for deepseek
void grouped_topk_cpu(
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    at::Tensor& hidden_states,
    at::Tensor& gating_output,
    int64_t topk,
    bool renormalize,
    int64_t num_expert_group,
    int64_t topk_group) {

  CHECK_EQ(topk_weights.sizes(), topk_ids.sizes());

  const auto st = hidden_states.scalar_type();
  CHECK_EQ(gating_output.scalar_type(), st);
  CHECK_EQ(topk_ids.scalar_type(), at::kInt);
  CHECK_EQ(topk_weights.scalar_type(), at::kFloat);

  int64_t num_tokens = hidden_states.size(0);
  int64_t num_experts = gating_output.size(1);
  TORCH_CHECK(gating_output.size(0) == num_tokens, "Number of tokens mismatch");

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "grouped_topk_kernel", [&] {
    switch(num_experts) {
      case 1:   LAUNCH_GROUPED_TOPK_KERNEL(1);   break;
      case 2:   LAUNCH_GROUPED_TOPK_KERNEL(2);   break;
      case 4:   LAUNCH_GROUPED_TOPK_KERNEL(4);   break;
      case 8:   LAUNCH_GROUPED_TOPK_KERNEL(8);   break;
      case 16:  LAUNCH_GROUPED_TOPK_KERNEL(16);  break;
      case 32:  LAUNCH_GROUPED_TOPK_KERNEL(32);  break;
      case 64:  LAUNCH_GROUPED_TOPK_KERNEL(64);  break;
      case 128: LAUNCH_GROUPED_TOPK_KERNEL(128); break;
      case 256: LAUNCH_GROUPED_TOPK_KERNEL(256); break;
      default: TORCH_CHECK(false, "Unexpected num_experts: ", num_experts);
    }
  });
}
