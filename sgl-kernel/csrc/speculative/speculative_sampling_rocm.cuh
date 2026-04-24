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
// ROCm port of speculative_sampling.cuh
// Uses aiter's sampling.cuh (hipCUB-based) instead of FlashInfer's (CUB-based)
#ifndef SPECULATIVE_SAMPLING_ROCM_CUH_
#define SPECULATIVE_SAMPLING_ROCM_CUH_

#include <assert.h>
#include <hip/hip_runtime.h>

// aiter's sampling.cuh provides: SamplingTempStorage, DeviceSamplingFromProb,
// vec_t, ceil_div, DeterministicInclusiveSum — all hipCUB-based equivalents
// of FlashInfer's CUB-based versions.
#include "sampling.cuh"

namespace aiter {

namespace sampling {

// SCAN_ALGO, REDUCE_ALGO, BLOCK_THREADS already defined in aiter::sampling
// via sampling.cuh

template <
    uint32_t BLOCK_THREADS_,
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
    DType* draft_probs,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens,
    uint32_t d,
    DType threshold_single,
    DType threshold_acc) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;

  extern __shared__ __align__(alignof(SamplingTempStorage<BLOCK_THREADS_, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS_, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(smem_sampling);

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

      if (coin <= prob_acc / threshold_acc || target_prob_single >= threshold_single) {
        // accept token
        prob_acc = 0.;
        cur_prob_offset = (bx * num_draft_tokens + cur_index) * d;
        coin = uniform_samples[bx * num_draft_tokens + cur_index];
        predicts[last_accepted_retrive_idx] = draft_token_id;
        ++num_accepted_tokens;
        accept_index[bx * num_speculative_tokens + num_accepted_tokens] = draft_index;
        last_accepted_retrive_idx = draft_index;
        break;
      } else {
        // FIXME: leverage draft probs
        draft_probs[cur_prob_offset + draft_token_id] = target_probs[cur_prob_offset + draft_token_id];
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
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS_ * VEC_SIZE); ++i) {
    q_vec.fill(DType(0));
    p_vec.fill(DType(0));
    if ((i * BLOCK_THREADS_ + tx) * VEC_SIZE < d) {
      q_vec.load(target_probs + cur_prob_offset + i * BLOCK_THREADS_ * VEC_SIZE + tx * VEC_SIZE);
      if (num_accepted_tokens != num_speculative_tokens - 1) {
        // there is no draft_probs for the bonus token
        p_vec.load(draft_probs + cur_prob_offset + i * BLOCK_THREADS_ * VEC_SIZE + tx * VEC_SIZE);
      }
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p[j] = max(q_vec[j] - p_vec[j], DType(0));
    }
    DType local_sum = 0;
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      local_sum += relu_q_minus_p[j];
    }
    sum_relu_q_minus_p += BlockReduce<DType, BLOCK_THREADS_, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                              .Sum(local_sum);
    __syncthreads();
  }
  if (tx == 0) {
    temp_storage.block_aggregate.value = sum_relu_q_minus_p;
  }
  // init the first rejected token to (d - 1)
  temp_storage.sampled_id = d - 1;
  __syncthreads();
  sum_relu_q_minus_p = temp_storage.block_aggregate.value;
  DType u = coin * sum_relu_q_minus_p;

  DType aggregate_relu_q_minus_p(0);
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS_ * VEC_SIZE); ++i) {
    q_vec.fill(DType(0));
    p_vec.fill(DType(0));
    if ((i * BLOCK_THREADS_ + tx) * VEC_SIZE < d) {
      q_vec.load(target_probs + cur_prob_offset + i * BLOCK_THREADS_ * VEC_SIZE + tx * VEC_SIZE);
      if (num_accepted_tokens != num_speculative_tokens - 1) {
        // there is no draft_probs for the bonus token
        p_vec.load(draft_probs + cur_prob_offset + i * BLOCK_THREADS_ * VEC_SIZE + tx * VEC_SIZE);
      }
    }

    vec_t<DType, VEC_SIZE> relu_q_minus_p_vec;
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p_vec[j] = max(q_vec[j] - p_vec[j], DType(0));
    }

    DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS_, SCAN_ALGORITHM, REDUCE_ALGORITHM, DETERMINISTIC>(
        i, d, [&](DType x) { return x > 0; }, u, relu_q_minus_p_vec, aggregate_relu_q_minus_p, &temp_storage);
    if (aggregate_relu_q_minus_p > u) {
      break;
    }
  }
  __syncthreads();
  // set the first rejected token
  predicts[last_accepted_retrive_idx] = temp_storage.sampled_id;
  // value at not used indices are undefined
}

// Dispatch helpers
#define DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, ...)   \
  switch (vec_size) {                                        \
    case 16: { constexpr uint32_t VEC_SIZE = 16; __VA_ARGS__ break; } \
    case 8:  { constexpr uint32_t VEC_SIZE = 8;  __VA_ARGS__ break; } \
    case 4:  { constexpr uint32_t VEC_SIZE = 4;  __VA_ARGS__ break; } \
    case 2:  { constexpr uint32_t VEC_SIZE = 2;  __VA_ARGS__ break; } \
    case 1:  { constexpr uint32_t VEC_SIZE = 1;  __VA_ARGS__ break; } \
    default: break;                                          \
  }

#define DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, ...) \
  if (deterministic) {                                            \
    constexpr bool DETERMINISTIC = true;                          \
    __VA_ARGS__                                                   \
  } else {                                                        \
    constexpr bool DETERMINISTIC = false;                         \
    __VA_ARGS__                                                   \
  }

template <typename DType, typename IdType, typename IdType2>
hipError_t TreeSpeculativeSamplingTargetOnly(
    IdType* predicts,
    IdType* output_token_ids,
    IdType* output_accepted_token_num,
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
    DType threshold_single = 1,
    DType threshold_acc = 1,
    bool deterministic = true,
    hipStream_t stream = 0) {
  constexpr uint32_t BLK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16u / (uint32_t)sizeof(DType), d);

  const uint32_t smem_size = sizeof(SamplingTempStorage<BLK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
  dim3 nblks(batch_size);
  dim3 nthrs(BLK_THREADS);
  float capped_threshold_acc = fmaxf(threshold_acc, 1e-9f);
  void* args[] = {
      &predicts,
      &output_token_ids,
      &output_accepted_token_num,
      &candidates,
      &retrive_index,
      &retrive_next_token,
      &retrive_next_sibling,
      &uniform_samples,
      &uniform_samples_for_final_sampling,
      &target_probs,
      &draft_probs,
      &batch_size,
      &num_speculative_tokens,
      &num_draft_tokens,
      &d,
      &threshold_single,
      &capped_threshold_acc};
  DISPATCH_ALIGNED_VEC_SIZE(
      vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
        auto kernel = TreeSpeculativeSamplingTargetOnly<
            BLK_THREADS,
            SCAN_ALGO,
            REDUCE_ALGO,
            VEC_SIZE,
            DETERMINISTIC,
            DType,
            IdType,
            IdType2>;
        hipFuncSetAttribute((const void*)kernel, hipFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        hipLaunchKernel((const void*)kernel, nblks, nthrs, args, smem_size, stream);
      })});
  return hipSuccess;
}

}  // namespace sampling

}  // namespace aiter

#endif  // SPECULATIVE_SAMPLING_ROCM_CUH_
