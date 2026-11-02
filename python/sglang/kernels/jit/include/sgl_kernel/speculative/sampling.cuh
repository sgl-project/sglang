/*
 * Copyright (c) 2025 by SGLang team.
 * Copyright (c) 2024-2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#pragma once

#include <flashinfer/sampling.cuh>

namespace sglang {
namespace speculative_sampling {

using namespace flashinfer;
using namespace flashinfer::sampling;
using namespace cub;

template <
    uint32_t BLOCK_THREADS,
    BlockScanAlgorithm SCAN_ALGORITHM,
    BlockReduceAlgorithm REDUCE_ALGORITHM,
    uint32_t VEC_SIZE,
    bool DETERMINISTIC,
    typename DType,
    typename IdType,
    typename IdType2>
__global__ void TreeSpeculativeSamplingTargetOnly(
    IdType* predicts,          // mutable
    IdType* accept_index,      // mutable
    IdType* accept_token_num,  // mutable
    IdType2* candidates,
    IdType2* retrive_index,
    IdType2* retrive_next_token,
    IdType2* retrive_next_sibling,
    DType* uniform_samples,
    DType* uniform_samples_for_final_sampling,
    DType* target_probs,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens,
    uint32_t d,
    DType threshold_single,
    DType threshold_acc) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;

  extern __shared__ __align__(alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(smem_sampling);
  // Tokens rejected at the last accepted node, which the residual excludes. A node has
  // at most num_draft_tokens - 1 children; the host sizes the dynamic smem for
  // num_draft_tokens entries.
  IdType* rejected_tokens = reinterpret_cast<IdType*>(
      smem_sampling + sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>));
  uint32_t num_rejected_tokens = 0;

  DType prob_acc = 0.0;
  uint32_t cur_prob_offset = bx * num_draft_tokens * d;
  DType coin = uniform_samples[bx * num_draft_tokens];
  IdType2 last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];
  accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx;
  uint32_t num_accepted_tokens = 0;
  IdType2 cur_index = 0;

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    cur_index = retrive_next_token[bx * num_draft_tokens + cur_index];
    while (cur_index != -1) {
      IdType2 draft_index = retrive_index[bx * num_draft_tokens + cur_index];
      IdType2 draft_token_id = candidates[bx * num_draft_tokens + cur_index];
      DType target_prob_single = target_probs[cur_prob_offset + draft_token_id];
      prob_acc += target_prob_single;

      const bool has_target_mass = target_prob_single > DType(0);
      if (has_target_mass && (coin < prob_acc / threshold_acc || target_prob_single >= threshold_single)) {
        // accept token
        prob_acc = 0.;
        cur_prob_offset = (bx * num_draft_tokens + cur_index) * d;
        coin = uniform_samples[bx * num_draft_tokens + cur_index];
        predicts[last_accepted_retrive_idx] = draft_token_id;
        ++num_accepted_tokens;
        accept_index[bx * num_speculative_tokens + num_accepted_tokens] = draft_index;
        last_accepted_retrive_idx = draft_index;
        // The residual now comes from this node's row; rejections at its parent do not apply.
        num_rejected_tokens = 0;
        break;
      } else {
        // Every thread walks the same path, so all keep the same count and thread 0 alone
        // stores the id. The bound only matters for a malformed sibling list.
        if (num_rejected_tokens < num_draft_tokens) {
          if (tx == 0) {
            rejected_tokens[num_rejected_tokens] = static_cast<IdType>(draft_token_id);
          }
          ++num_rejected_tokens;
        }
        cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
      }
    }
    if (cur_index == -1) break;
  }
  accept_token_num[bx] = num_accepted_tokens;
  // Publish thread 0's rejected ids to the whole block.
  __syncthreads();

  // we need a different coin for the final sampling
  coin = uniform_samples_for_final_sampling[bx];

  // Sample from relu(q - p), where q is the target row of the last accepted node and p
  // equals q at the tokens rejected there and 0 elsewhere. That is 0 at a rejected token
  // and max(q, 0) everywhere else, so p is never materialized. When every draft level
  // was accepted (the bonus token), nothing is rejected and the residual is relu(q).
  //
  // Iteration i below gives thread tx the tokens [(i * BLOCK_THREADS + tx) * VEC_SIZE,
  // + VEC_SIZE). Only the few threads that own a rejected token read the list again.
  bool owns_rejected_token = false;
  for (uint32_t r = 0; r < num_rejected_tokens; ++r) {
    owns_rejected_token |= (static_cast<uint32_t>(rejected_tokens[r]) / VEC_SIZE) % BLOCK_THREADS == tx;
  }
  // Bit j is set when token (i * BLOCK_THREADS + tx) * VEC_SIZE + j was rejected.
  auto rejected_lanes = [&](uint32_t i) -> uint32_t {
    uint32_t lanes = 0;
    if (owns_rejected_token) {
      const uint32_t first_token = (i * BLOCK_THREADS + tx) * VEC_SIZE;
      for (uint32_t r = 0; r < num_rejected_tokens; ++r) {
        const uint32_t lane = static_cast<uint32_t>(rejected_tokens[r]) - first_token;
        if (lane < VEC_SIZE) {
          lanes |= 1u << lane;
        }
      }
    }
    return lanes;
  };

  DType sum_relu_q_minus_p(0);
  vec_t<DType, VEC_SIZE> q_vec;
  DType relu_q_minus_p[VEC_SIZE];
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(DType(0));
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      q_vec.load(target_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }
    const uint32_t rejected = rejected_lanes(i);
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p[j] = (rejected >> j) & 1u ? DType(0) : max(q_vec[j], DType(0));
    }
    sum_relu_q_minus_p += BlockReduce<DType, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                              .Sum<VEC_SIZE>(relu_q_minus_p);
    __syncthreads();
  }
  if (tx == 0) {
    temp_storage.block_aggregate.value = sum_relu_q_minus_p;
  }

  temp_storage.sampled_id = d;
  temp_storage.last_valid_id = -1;
  __syncthreads();
  sum_relu_q_minus_p = temp_storage.block_aggregate.value;
  DType u = coin * sum_relu_q_minus_p;

  DType aggregate_relu_q_minus_p(0);
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(DType(0));
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      q_vec.load(target_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }

    const uint32_t rejected = rejected_lanes(i);
    vec_t<DType, VEC_SIZE> relu_q_minus_p_vec;
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p_vec[j] = (rejected >> j) & 1u ? DType(0) : max(q_vec[j], DType(0));
    }

    DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, DETERMINISTIC>(
        i, d, [&](DType x) { return x > 0; }, u, relu_q_minus_p_vec, aggregate_relu_q_minus_p, &temp_storage);
    if (aggregate_relu_q_minus_p > u) {
      break;
    }
  }
  __syncthreads();
  // This would happen when u is very close to 1
  // and the sum of probabilities is smaller than u
  // In this case, we use the last valid index as the sampled id
  int sampled_id = temp_storage.sampled_id;
  if (sampled_id == d) {
    if (temp_storage.last_valid_id == -1) {
      sampled_id = d - 1;
    } else {
      sampled_id = temp_storage.last_valid_id;
    }
  }
  // set the first rejected token
  predicts[last_accepted_retrive_idx] = sampled_id;
  // value at not used indices are undefined
}

}  // namespace speculative_sampling
}  // namespace sglang
