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
    typename IdType2,
    // Chains only (one candidate per tree level): draft_probs is not read or
    // written. The walk remembers the one token it rejects, and the final
    // sampling takes p = q there and p = 0 elsewhere, which is exactly what the
    // zero-filled buffer would hold, so the outputs are the same bits.
    bool NO_DRAFT_PROBS = false>
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
    DType* draft_probs,
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

  DType prob_acc = 0.0;
  uint32_t cur_prob_offset = bx * num_draft_tokens * d;
  DType coin = uniform_samples[bx * num_draft_tokens];
  IdType2 last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];
  accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx;
  uint32_t num_accepted_tokens = 0;
  IdType2 cur_index = 0;
  // NO_DRAFT_PROBS: the token rejected in the current row, -1 if none.
  IdType2 rejected_token_id = -1;

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
        rejected_token_id = -1;
        ++num_accepted_tokens;
        accept_index[bx * num_speculative_tokens + num_accepted_tokens] = draft_index;
        last_accepted_retrive_idx = draft_index;
        break;
      } else {
        // FIXME: leverage draft probs
        if constexpr (NO_DRAFT_PROBS) {
          rejected_token_id = draft_token_id;
        } else {
          draft_probs[cur_prob_offset + draft_token_id] = target_probs[cur_prob_offset + draft_token_id];
        }
        cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
      }
    }
    if (cur_index == -1) break;
  }
  accept_token_num[bx] = num_accepted_tokens;

  // we need a different coin for the final sampling
  coin = uniform_samples_for_final_sampling[bx];

  // sample from relu(target_probs - draft_probs)
  DType sum_relu_q_minus_p(0);
  vec_t<DType, VEC_SIZE> q_vec, p_vec;
  DType relu_q_minus_p[VEC_SIZE];
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(DType(0));
    p_vec.fill(DType(0));
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      q_vec.load(target_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      if (num_accepted_tokens != num_speculative_tokens - 1) {
        // there is no draft_probs for the bonus token
        if constexpr (NO_DRAFT_PROBS) {
#pragma unroll
          for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            const IdType2 token = static_cast<IdType2>((i * BLOCK_THREADS + tx) * VEC_SIZE + j);
            if (token == rejected_token_id) p_vec[j] = q_vec[j];
          }
        } else {
          p_vec.load(draft_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
        }
      }
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p[j] = max(q_vec[j] - p_vec[j], DType(0));
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
    p_vec.fill(DType(0));
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      q_vec.load(target_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      if (num_accepted_tokens != num_speculative_tokens - 1) {
        // there is no draft_probs for the bonus token
        if constexpr (NO_DRAFT_PROBS) {
#pragma unroll
          for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            const IdType2 token = static_cast<IdType2>((i * BLOCK_THREADS + tx) * VEC_SIZE + j);
            if (token == rejected_token_id) p_vec[j] = q_vec[j];
          }
        } else {
          p_vec.load(draft_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
        }
      }
    }

    vec_t<DType, VEC_SIZE> relu_q_minus_p_vec;
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p_vec[j] = max(q_vec[j] - p_vec[j], DType(0));
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

// ---------------------------------------------------------------------------
// Chain verify sampling in two launches (NO_DRAFT_PROBS form only).
//
// TreeSpeculativeSamplingTargetOnly<..., NO_DRAFT_PROBS = true> runs one CTA per
// request: a short accept walk, then two sequential passes over the
// vocabulary (the relu(q - p) total, then the inverse-CDF search). The two
// kernels below split the passes over one CTA per vocabulary chunk and give the
// same bits:
//  * ChainVerifyChunkStats: per (request, chunk) the same BlockReduce sums the
//    single-CTA kernel computes for that chunk (the total pass's sum of relu
//    values, and DeviceSamplingFromProb's sum of the values > 0) and the
//    chunk's last valid index (DeviceSamplingFromProb's max-reduce).
//  * ChainVerifyFinish: per request the same walk and outputs; thread 0 adds
//    the chunk sums in chunk order (the single-CTA kernel's thread-0 running
//    sum), finds the chunk where the running aggregate first exceeds u, and the
//    CTA runs DeviceSamplingFromProb on that chunk with the aggregate of the
//    chunks before it. last_valid_id is the last valid index of the chunks the
//    single-CTA loop would have visited.
// ---------------------------------------------------------------------------

template <typename DType, typename IdType2>
struct ChainWalk {
  uint32_t cur_prob_offset;
  uint32_t num_accepted_tokens;
  IdType2 rejected_token_id;
  IdType2 last_accepted_retrive_idx;
};

// The accept walk of TreeSpeculativeSamplingTargetOnly (NO_DRAFT_PROBS), same
// statements; WRITE selects whether predicts / accept_index are written.
template <bool WRITE, typename DType, typename IdType, typename IdType2>
__device__ __forceinline__ ChainWalk<DType, IdType2> chain_verify_walk(
    IdType* predicts,
    IdType* accept_index,
    IdType2* candidates,
    IdType2* retrive_index,
    IdType2* retrive_next_token,
    IdType2* retrive_next_sibling,
    DType* uniform_samples,
    DType* target_probs,
    uint32_t bx,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens,
    uint32_t d,
    DType threshold_single,
    DType threshold_acc) {
  DType prob_acc = 0.0;
  uint32_t cur_prob_offset = bx * num_draft_tokens * d;
  DType coin = uniform_samples[bx * num_draft_tokens];
  IdType2 last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];
  if constexpr (WRITE) accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx;
  uint32_t num_accepted_tokens = 0;
  IdType2 cur_index = 0;
  IdType2 rejected_token_id = -1;

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    cur_index = retrive_next_token[bx * num_draft_tokens + cur_index];
    while (cur_index != -1) {
      IdType2 draft_index = retrive_index[bx * num_draft_tokens + cur_index];
      IdType2 draft_token_id = candidates[bx * num_draft_tokens + cur_index];
      DType target_prob_single = target_probs[cur_prob_offset + draft_token_id];
      prob_acc += target_prob_single;

      const bool has_target_mass = target_prob_single > DType(0);
      if (has_target_mass && (coin < prob_acc / threshold_acc || target_prob_single >= threshold_single)) {
        prob_acc = 0.;
        cur_prob_offset = (bx * num_draft_tokens + cur_index) * d;
        coin = uniform_samples[bx * num_draft_tokens + cur_index];
        if constexpr (WRITE) predicts[last_accepted_retrive_idx] = draft_token_id;
        rejected_token_id = -1;
        ++num_accepted_tokens;
        if constexpr (WRITE) accept_index[bx * num_speculative_tokens + num_accepted_tokens] = draft_index;
        last_accepted_retrive_idx = draft_index;
        break;
      } else {
        rejected_token_id = draft_token_id;
        cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
      }
    }
    if (cur_index == -1) break;
  }
  return {cur_prob_offset, num_accepted_tokens, rejected_token_id, last_accepted_retrive_idx};
}

// relu(q - p) of chunk i for this thread, as the single-CTA kernel loads it.
template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, typename DType, typename IdType2>
__device__ __forceinline__ void chain_verify_relu(
    vec_t<DType, VEC_SIZE>& out,
    DType* target_probs,
    const ChainWalk<DType, IdType2>& w,
    uint32_t i,
    uint32_t d,
    uint32_t num_speculative_tokens) {
  const uint32_t tx = threadIdx.x;
  vec_t<DType, VEC_SIZE> q_vec, p_vec;
  q_vec.fill(DType(0));
  p_vec.fill(DType(0));
  if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
    q_vec.load(target_probs + w.cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    if (w.num_accepted_tokens != num_speculative_tokens - 1) {
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        const IdType2 token = static_cast<IdType2>((i * BLOCK_THREADS + tx) * VEC_SIZE + j);
        if (token == w.rejected_token_id) p_vec[j] = q_vec[j];
      }
    }
  }
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    out[j] = max(q_vec[j] - p_vec[j], DType(0));
  }
}

template <
    uint32_t BLOCK_THREADS,
    BlockScanAlgorithm SCAN_ALGORITHM,
    BlockReduceAlgorithm REDUCE_ALGORITHM,
    uint32_t VEC_SIZE,
    typename DType,
    typename IdType,
    typename IdType2>
__global__ void ChainVerifyChunkStats(
    IdType2* candidates,
    IdType2* retrive_index,
    IdType2* retrive_next_token,
    IdType2* retrive_next_sibling,
    DType* uniform_samples,
    DType* target_probs,
    DType* chunk_relu_sum,  // [batch, num_chunks]
    DType* chunk_agg_sum,   // [batch, num_chunks]
    int* chunk_last_valid,  // [batch, num_chunks]
    uint32_t num_chunks,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens,
    uint32_t d,
    DType threshold_single,
    DType threshold_acc) {
  const uint32_t i = blockIdx.x, bx = blockIdx.y, tx = threadIdx.x;
  extern __shared__ __align__(alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(smem_sampling);

  const auto w = chain_verify_walk<false, DType, IdType, IdType2>(
      nullptr,
      nullptr,
      candidates,
      retrive_index,
      retrive_next_token,
      retrive_next_sibling,
      uniform_samples,
      target_probs,
      bx,
      num_speculative_tokens,
      num_draft_tokens,
      d,
      threshold_single,
      threshold_acc);

  vec_t<DType, VEC_SIZE> relu_vec;
  chain_verify_relu<BLOCK_THREADS, VEC_SIZE, DType, IdType2>(relu_vec, target_probs, w, i, d, num_speculative_tokens);
  DType relu[VEC_SIZE], gt[VEC_SIZE];
  int valid_index[VEC_SIZE];
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    relu[j] = relu_vec[j];
    gt[j] = relu_vec[j] > 0 ? relu_vec[j] : 0;
    const uint32_t idx = (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
    valid_index[j] = (relu_vec[j] > 0 && idx < d) ? static_cast<int>(idx) : -1;
  }
  // The total pass's sum (single-CTA kernel, first loop).
  DType s_relu =
      BlockReduce<DType, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce).Sum<VEC_SIZE>(relu);
  __syncthreads();
  // DeviceSamplingFromProb's chunk aggregate and last valid index.
  DType s_agg = BlockReduce<DType, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce).Sum<VEC_SIZE>(gt);
  __syncthreads();
  int max_valid = BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce_int)
                      .Reduce(valid_index, MaxReduceOp{});
  if (tx == 0) {
    chunk_relu_sum[bx * num_chunks + i] = s_relu;
    chunk_agg_sum[bx * num_chunks + i] = s_agg;
    chunk_last_valid[bx * num_chunks + i] = max_valid;
  }
}

template <
    uint32_t BLOCK_THREADS,
    BlockScanAlgorithm SCAN_ALGORITHM,
    BlockReduceAlgorithm REDUCE_ALGORITHM,
    uint32_t VEC_SIZE,
    bool DETERMINISTIC,
    typename DType,
    typename IdType,
    typename IdType2>
__global__ void ChainVerifyFinish(
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
    const DType* chunk_relu_sum,
    const DType* chunk_agg_sum,
    const int* chunk_last_valid,
    uint32_t num_chunks,
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
  __shared__ uint32_t s_chunk;
  __shared__ DType s_prefix, s_u;
  // The per-chunk values, loaded by all threads at once so that thread 0's
  // ordered accumulation below reads shared memory (num_chunks <= BLOCK_THREADS).
  __shared__ DType s_relu_sum[BLOCK_THREADS], s_agg_sum[BLOCK_THREADS];
  __shared__ int s_last_valid[BLOCK_THREADS];

  const auto w = chain_verify_walk<true, DType, IdType, IdType2>(
      predicts,
      accept_index,
      candidates,
      retrive_index,
      retrive_next_token,
      retrive_next_sibling,
      uniform_samples,
      target_probs,
      bx,
      num_speculative_tokens,
      num_draft_tokens,
      d,
      threshold_single,
      threshold_acc);
  accept_token_num[bx] = w.num_accepted_tokens;

  const DType coin = uniform_samples_for_final_sampling[bx];
  for (uint32_t i = tx; i < num_chunks; i += BLOCK_THREADS) {
    s_relu_sum[i] = chunk_relu_sum[bx * num_chunks + i];
    s_agg_sum[i] = chunk_agg_sum[bx * num_chunks + i];
    s_last_valid[i] = chunk_last_valid[bx * num_chunks + i];
  }
  __syncthreads();
  if (tx == 0) {
    // The single-CTA kernel's thread-0 running sum, chunk by chunk.
    DType sum_relu_q_minus_p(0);
    for (uint32_t i = 0; i < num_chunks; ++i)
      sum_relu_q_minus_p += s_relu_sum[i];
    const DType u = coin * sum_relu_q_minus_p;
    // DeviceSamplingFromProb's aggregate: the first chunk where it exceeds u
    // is where the single-CTA loop scans and stops.
    DType aggregate(0);
    uint32_t hit = num_chunks;
    for (uint32_t i = 0; i < num_chunks; ++i) {
      const DType local = s_agg_sum[i];
      if (aggregate + local > u) {
        hit = i;
        break;
      }
      aggregate += local;
    }
    // The last valid index among the chunks the loop visits before the hit
    // chunk (the hit chunk's own update happens in DeviceSamplingFromProb).
    int last_valid = -1;
    const uint32_t visited = hit < num_chunks ? hit : num_chunks;
    for (uint32_t i = 0; i < visited; ++i) {
      const int v = s_last_valid[i];
      if (v != -1 && v < static_cast<int>(d)) last_valid = v;
    }
    s_u = u;
    temp_storage.sampled_id = d;
    temp_storage.last_valid_id = last_valid;
    s_chunk = hit;
    s_prefix = aggregate;
  }
  __syncthreads();
  const DType u = s_u;
  const uint32_t hit = s_chunk;
  if (hit < num_chunks) {
    DType aggregate_relu_q_minus_p = s_prefix;
    vec_t<DType, VEC_SIZE> relu_vec;
    chain_verify_relu<BLOCK_THREADS, VEC_SIZE, DType, IdType2>(
        relu_vec, target_probs, w, hit, d, num_speculative_tokens);
    DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, DETERMINISTIC>(
        hit, d, [&](DType x) { return x > 0; }, u, relu_vec, aggregate_relu_q_minus_p, &temp_storage);
  }
  __syncthreads();
  int sampled_id = temp_storage.sampled_id;
  if (sampled_id == d) {
    if (temp_storage.last_valid_id == -1) {
      sampled_id = d - 1;
    } else {
      sampled_id = temp_storage.last_valid_id;
    }
  }
  predicts[w.last_accepted_retrive_idx] = sampled_id;
}

}  // namespace speculative_sampling
}  // namespace sglang
