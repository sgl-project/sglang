#pragma once

#include <ATen/cpu/vec/functional.h>

#include "common.h"

namespace {

template <int BLOCK_M>
int moe_align_block_size(
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ total_cnts,
    int32_t* __restrict__ cumsums,
    int32_t* __restrict__ offsets,
    int num_experts,
    int numel,
    int num_threads) {
#define T_INDEX(tt) total_cnts + (tt) * num_experts

  // accumulate count of expert ids locally
  at::parallel_for(0, numel, 0, [&](int begin, int end) {
    int tid = at::get_thread_num();
    int32_t* __restrict__ local_cnts = T_INDEX(tid + 1);

    for (int i = begin; i < end; ++i) {
      local_cnts[topk_ids[i]]++;
    }
  });

  using iVec = at::vec::Vectorized<int32_t>;
  for (int t = 0; t < num_threads; ++t) {
    at::vec::map2<int32_t>(
        [](iVec x, iVec y) { return x + y; }, T_INDEX(t + 1), T_INDEX(t + 1), T_INDEX(t), num_experts);
  }

  // the last row holds sums of each experts
  int32_t* total_cnts_t_1 = T_INDEX(num_threads);

  cumsums[0] = 0;
  offsets[0] = 0;
  int compact_offset = 0;
  for (int e = 0; e < num_experts; ++e) {
    const int expert_tokens = total_cnts_t_1[e];
    const int expert_start = cumsums[e];
    // accumulate `num_tokens_post_pad`, also as the expert offset
    cumsums[e + 1] = expert_start + div_up(expert_tokens, BLOCK_M) * BLOCK_M;

    for (int k = expert_start; k < cumsums[e + 1]; k += BLOCK_M) {
      expert_ids[k / BLOCK_M] = e;
      offsets[k / BLOCK_M + 1] = compact_offset + std::min(k - expert_start + BLOCK_M, expert_tokens);
    }
    compact_offset += expert_tokens;
  }
  int num_tokens_post_pad = cumsums[num_experts];

  at::parallel_for(0, numel, 0, [&](int begin, int end) {
    int tid = at::get_thread_num();
    // thread tid offsets in `total_cnts`
    int32_t* __restrict__ offsets = T_INDEX(tid);

    for (int i = begin; i < end; ++i) {
      int32_t expert_id = topk_ids[i];
      int32_t b_offset = cumsums[expert_id];
      int32_t t_offset = offsets[expert_id];
      sorted_ids[b_offset + t_offset] = i;
      offsets[expert_id]++;
    }
  });

  // debug: the offset for thread t_1 should be identical to t_2
  int32_t* total_cnts_t_2 = T_INDEX(num_threads - 1);
  for (int e = 0; e < num_experts; ++e) {
    TORCH_CHECK(total_cnts_t_1[e] == total_cnts_t_2[e]);
  }

  // debug: the last value of offsets should be `numel`
  TORCH_CHECK(offsets[num_tokens_post_pad / BLOCK_M] == numel);

  return num_tokens_post_pad;
}

#undef T_INDEX

}  // namespace
