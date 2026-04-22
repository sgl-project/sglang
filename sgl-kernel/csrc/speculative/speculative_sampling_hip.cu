/*
 * Copyright (c) 2025 by SGLang team.
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

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

#include <hipcub/hipcub.hpp>

#include "pytorch_extension_utils_rocm.h"

namespace {

constexpr uint32_t BLOCK_THREADS = 1024;

template <uint32_t BLOCK_SIZE>
struct SamplingTempStorage {
  union {
    typename hipcub::BlockReduce<float, BLOCK_SIZE>::TempStorage reduce;
    typename hipcub::BlockScan<float, BLOCK_SIZE, hipcub::BLOCK_SCAN_RAKING>::TempStorage scan;
  } block_prim;
  union {
    float value;
  } block_aggregate;
  int sampled_id;
};

template <uint32_t BLOCK_SIZE>
__global__ void TreeSpecSamplingTargetOnlyKernel(
    int32_t* __restrict__ predicts,
    int32_t* __restrict__ accept_index,
    int32_t* __restrict__ accept_token_num,
    const int64_t* __restrict__ candidates,
    const int64_t* __restrict__ retrive_index,
    const int64_t* __restrict__ retrive_next_token,
    const int64_t* __restrict__ retrive_next_sibling,
    const float* __restrict__ uniform_samples,
    const float* __restrict__ uniform_samples_for_final_sampling,
    float* __restrict__ target_probs,
    float* __restrict__ draft_probs,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens,
    uint32_t d,
    float threshold_single,
    float threshold_acc) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;

  extern __shared__ __align__(alignof(SamplingTempStorage<BLOCK_SIZE>)) uint8_t smem[];
  auto& temp_storage = reinterpret_cast<SamplingTempStorage<BLOCK_SIZE>&>(smem);

  // --- Part 1: Tree traversal with accept/reject ---
  float prob_acc = 0.0f;
  uint32_t cur_prob_offset = bx * num_draft_tokens * d;
  float coin = uniform_samples[bx * num_draft_tokens];
  int64_t last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];
  accept_index[bx * num_speculative_tokens] = static_cast<int32_t>(last_accepted_retrive_idx);
  uint32_t num_accepted_tokens = 0;
  int64_t cur_index = 0;

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    cur_index = retrive_next_token[bx * num_draft_tokens + cur_index];
    while (cur_index != -1) {
      int64_t draft_index = retrive_index[bx * num_draft_tokens + cur_index];
      int64_t draft_token_id = candidates[bx * num_draft_tokens + cur_index];
      float target_prob_single = target_probs[cur_prob_offset + draft_token_id];
      prob_acc += target_prob_single;

      if (coin <= prob_acc / threshold_acc || target_prob_single >= threshold_single) {
        prob_acc = 0.f;
        cur_prob_offset = (bx * num_draft_tokens + static_cast<uint32_t>(cur_index)) * d;
        coin = uniform_samples[bx * num_draft_tokens + cur_index];
        predicts[last_accepted_retrive_idx] = static_cast<int32_t>(draft_token_id);
        ++num_accepted_tokens;
        accept_index[bx * num_speculative_tokens + num_accepted_tokens] = static_cast<int32_t>(draft_index);
        last_accepted_retrive_idx = draft_index;
        break;
      } else {
        draft_probs[cur_prob_offset + draft_token_id] = target_probs[cur_prob_offset + draft_token_id];
        cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
      }
    }
    if (cur_index == -1) break;
  }
  accept_token_num[bx] = static_cast<int32_t>(num_accepted_tokens);

  coin = uniform_samples_for_final_sampling[bx];

  // --- Part 2: Sample from relu(target_probs - draft_probs) ---
  // Each thread handles elements at indices: tx, tx + BLOCK_SIZE, tx + 2*BLOCK_SIZE, ...
  const bool has_draft = (num_accepted_tokens != num_speculative_tokens - 1);
  const uint32_t num_iters = (d + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // First pass: compute total sum of relu(q - p)
  float total_sum = 0.0f;
  for (uint32_t i = 0; i < num_iters; ++i) {
    uint32_t idx = i * BLOCK_SIZE + tx;
    float relu_val = 0.0f;
    if (idx < d) {
      float q = target_probs[cur_prob_offset + idx];
      float p = has_draft ? draft_probs[cur_prob_offset + idx] : 0.0f;
      relu_val = fmaxf(q - p, 0.0f);
    }
    float block_sum = hipcub::BlockReduce<float, BLOCK_SIZE>(temp_storage.block_prim.reduce).Sum(relu_val);
    __syncthreads();
    if (tx == 0) {
      total_sum += block_sum;
    }
  }
  if (tx == 0) {
    temp_storage.block_aggregate.value = total_sum;
  }
  temp_storage.sampled_id = static_cast<int>(d - 1);
  __syncthreads();
  total_sum = temp_storage.block_aggregate.value;
  float u = coin * total_sum;

  // Second pass: inclusive prefix-sum scan to find sampled token (CDF inversion)
  float running_total = 0.0f;
  for (uint32_t i = 0; i < num_iters; ++i) {
    uint32_t idx = i * BLOCK_SIZE + tx;
    float relu_val = 0.0f;
    if (idx < d) {
      float q = target_probs[cur_prob_offset + idx];
      float p = has_draft ? draft_probs[cur_prob_offset + idx] : 0.0f;
      relu_val = fmaxf(q - p, 0.0f);
    }

    float prefix_sum;
    hipcub::BlockScan<float, BLOCK_SIZE, hipcub::BLOCK_SCAN_RAKING>(temp_storage.block_prim.scan)
        .InclusiveSum(relu_val, prefix_sum);
    __syncthreads();

    float cumulative = running_total + prefix_sum;
    if (relu_val > 0.0f && cumulative > u && (cumulative - relu_val) <= u && idx < d) {
      atomicMin(&temp_storage.sampled_id, static_cast<int>(idx));
    }
    __syncthreads();

    if (tx == BLOCK_SIZE - 1) {
      temp_storage.block_aggregate.value = cumulative;
    }
    __syncthreads();
    running_total = temp_storage.block_aggregate.value;

    if (running_total > u) break;
  }
  __syncthreads();
  predicts[last_accepted_retrive_idx] = temp_storage.sampled_id;
}

}  // namespace

void tree_speculative_sampling_target_only(
    at::Tensor predicts,
    at::Tensor accept_index,
    at::Tensor accept_token_num,
    at::Tensor candidates,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    at::Tensor uniform_samples,
    at::Tensor uniform_samples_for_final_sampling,
    at::Tensor target_probs,
    at::Tensor draft_probs,
    double threshold_single,
    double threshold_acc,
    bool deterministic) {
  CHECK_INPUT(candidates);
  CHECK_INPUT(retrive_index);
  CHECK_INPUT(retrive_next_token);
  CHECK_INPUT(retrive_next_sibling);
  CHECK_INPUT(uniform_samples);
  CHECK_INPUT(uniform_samples_for_final_sampling);
  CHECK_INPUT(target_probs);
  auto device = target_probs.device();
  CHECK_EQ(candidates.device(), device);
  CHECK_EQ(retrive_index.device(), device);
  CHECK_EQ(retrive_next_token.device(), device);
  CHECK_EQ(retrive_next_sibling.device(), device);
  CHECK_EQ(uniform_samples.device(), device);
  CHECK_EQ(uniform_samples_for_final_sampling.device(), device);
  CHECK_EQ(target_probs.device(), device);
  CHECK_DIM(1, predicts);
  CHECK_DIM(2, accept_index);
  CHECK_DIM(1, accept_token_num);
  CHECK_DIM(2, candidates);
  CHECK_DIM(2, retrive_index);
  CHECK_DIM(2, retrive_next_token);
  CHECK_DIM(2, retrive_next_sibling);
  CHECK_DIM(2, uniform_samples);
  CHECK_DIM(3, target_probs);
  CHECK_DIM(3, draft_probs);
  unsigned int batch_size = uniform_samples.size(0);
  unsigned int num_spec_step = accept_index.size(1);
  unsigned int num_draft_tokens = candidates.size(1);
  unsigned int vocab_size = target_probs.size(2);
  CHECK_EQ(batch_size, candidates.size(0));
  CHECK_EQ(batch_size, retrive_index.size(0));
  CHECK_EQ(batch_size, retrive_next_token.size(0));
  CHECK_EQ(batch_size, retrive_next_sibling.size(0));
  CHECK_EQ(batch_size, target_probs.size(0));
  CHECK_EQ(num_draft_tokens, retrive_index.size(1));
  CHECK_EQ(num_draft_tokens, retrive_next_token.size(1));
  CHECK_EQ(num_draft_tokens, retrive_next_sibling.size(1));
  CHECK_EQ(num_draft_tokens, uniform_samples.size(1));
  CHECK_EQ(num_draft_tokens, target_probs.size(1));
  CHECK_EQ(vocab_size, target_probs.size(2));
  CHECK_EQ(batch_size, accept_index.size(0));
  CHECK_EQ(batch_size, accept_token_num.size(0));
  if (predicts.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'predicts' to be of type int (torch.int32).");
  }
  if (accept_index.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'accept_index' to be of type int (torch.int32).");
  }
  if (accept_token_num.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'accept_token_num' to be of type int (torch.int32).");
  }
  if (candidates.scalar_type() != at::kLong) {
    throw std::runtime_error("Expected 'candidates' to be of type long (torch.int64).");
  }
  if (retrive_index.scalar_type() != at::kLong) {
    throw std::runtime_error("Expected 'retrive_index' to be of type long (torch.int64).");
  }
  if (retrive_next_token.scalar_type() != at::kLong) {
    throw std::runtime_error("Expected 'retrive_next_token' to be of type long (torch.int64).");
  }
  if (retrive_next_sibling.scalar_type() != at::kLong) {
    throw std::runtime_error("Expected 'retrive_next_sibling' to be of type long (torch.int64).");
  }
  if (uniform_samples.scalar_type() != at::kFloat) {
    throw std::runtime_error("Expected 'uniform_samples' to be of type float (torch.float32).");
  }
  if (uniform_samples_for_final_sampling.scalar_type() != at::kFloat) {
    throw std::runtime_error("Expected 'uniform_samples_for_final_sampling' to be of type float (torch.float32).");
  }
  if (target_probs.scalar_type() != at::kFloat) {
    throw std::runtime_error("Expected 'target_probs' to be of type float (torch.float32).");
  }
  if (draft_probs.scalar_type() != at::kFloat) {
    throw std::runtime_error("Expected 'draft_probs' to be of type float (torch.float32).");
  }
  CHECK_GE(threshold_single, 0);
  CHECK_GE(1, threshold_single);
  CHECK_GE(threshold_acc, 0);
  CHECK_GE(1, threshold_acc);

  float capped_threshold_acc = fmaxf(static_cast<float>(threshold_acc), 1e-9f);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS>);
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);

  hipLaunchKernelGGL(
      (TreeSpecSamplingTargetOnlyKernel<BLOCK_THREADS>),
      nblks,
      nthrs,
      smem_size,
      stream,
      static_cast<int32_t*>(predicts.data_ptr()),
      static_cast<int32_t*>(accept_index.data_ptr()),
      static_cast<int32_t*>(accept_token_num.data_ptr()),
      static_cast<int64_t*>(candidates.data_ptr()),
      static_cast<int64_t*>(retrive_index.data_ptr()),
      static_cast<int64_t*>(retrive_next_token.data_ptr()),
      static_cast<int64_t*>(retrive_next_sibling.data_ptr()),
      static_cast<float*>(uniform_samples.data_ptr()),
      static_cast<float*>(uniform_samples_for_final_sampling.data_ptr()),
      static_cast<float*>(target_probs.data_ptr()),
      static_cast<float*>(draft_probs.data_ptr()),
      batch_size,
      num_spec_step,
      num_draft_tokens,
      vocab_size,
      static_cast<float>(threshold_single),
      capped_threshold_acc);

  auto status = hipGetLastError();
  TORCH_CHECK(status == hipSuccess, "TreeSpecSamplingTargetOnlyKernel failed with error: ", hipGetErrorString(status));
}
