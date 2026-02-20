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
// Adapted from
// https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/speculative/speculative_sampling.cuh

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <flashinfer/sampling.cuh>
#include <tvm/ffi/container/tensor.h>

#include <assert.h>

namespace flashinfer {

namespace sampling {

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
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(DType(0));
    p_vec.fill(DType(0));
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      q_vec.load(target_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      if (num_accepted_tokens != num_speculative_tokens - 1) {
        // there is no draft_probs for the bonus token
        p_vec.load(draft_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
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
  // init the first rejected token to (d - 1)
  temp_storage.sampled_id = d - 1;
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
        p_vec.load(draft_probs + cur_prob_offset + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
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
  // set the first rejected token
  predicts[last_accepted_retrive_idx] = temp_storage.sampled_id;
  // value at not used indices are undefined
}

template <typename DType, typename IdType, typename IdType2>
cudaError_t TreeSpeculativeSamplingTargetOnlyLauncher(
    IdType* predicts,                   // mutable
    IdType* output_token_ids,           // mutable
    IdType* output_accepted_token_num,  // mutable
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
    cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), d);

  const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
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
            BLOCK_THREADS,
            SCAN_ALGO,
            REDUCE_ALGO,
            VEC_SIZE,
            DETERMINISTIC,
            DType,
            IdType,
            IdType2>;
        FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      })});
  return cudaSuccess;
}

}  // namespace sampling

}  // namespace flashinfer

namespace {

// ---------------------------------------------------------------------------
// tvm-ffi entry point
// ---------------------------------------------------------------------------

void tree_speculative_sampling_target_only(
    tvm::ffi::TensorView predicts,                            // [tot_num_draft_tokens] int32, mutable
    tvm::ffi::TensorView accept_index,                        // [bs, num_spec_step] int32, mutable
    tvm::ffi::TensorView accept_token_num,                    // [bs] int32, mutable
    tvm::ffi::TensorView candidates,                          // [bs, num_draft_tokens] int64
    tvm::ffi::TensorView retrive_index,                       // [bs, num_draft_tokens] int64
    tvm::ffi::TensorView retrive_next_token,                  // [bs, num_draft_tokens] int64
    tvm::ffi::TensorView retrive_next_sibling,                // [bs, num_draft_tokens] int64
    tvm::ffi::TensorView uniform_samples,                     // [bs, num_draft_tokens] float32
    tvm::ffi::TensorView uniform_samples_for_final_sampling,  // [bs] float32
    tvm::ffi::TensorView target_probs,                        // [bs, num_draft_tokens, vocab_size] float32
    tvm::ffi::TensorView draft_probs,                         // [bs, num_draft_tokens, vocab_size] float32, mutable
    float threshold_single,
    float threshold_acc,
    int deterministic) {
  using namespace host;

  RuntimeCheck(target_probs.device().device_type == kDLCUDA, "target_probs must be a CUDA tensor");
  RuntimeCheck(target_probs.ndim() == 3, "target_probs must be 3D: [bs, num_draft_tokens, vocab_size]");
  RuntimeCheck(target_probs.is_contiguous(), "target_probs must be contiguous");
  RuntimeCheck(
      target_probs.dtype().code == kDLFloat && target_probs.dtype().bits == 32, "target_probs must be float32");

  uint32_t batch_size = static_cast<uint32_t>(target_probs.size(0));
  uint32_t num_draft_tokens = static_cast<uint32_t>(target_probs.size(1));
  uint32_t vocab_size = static_cast<uint32_t>(target_probs.size(2));

  RuntimeCheck(accept_index.ndim() == 2, "accept_index must be 2D: [bs, num_spec_step]");
  uint32_t num_spec_step = static_cast<uint32_t>(accept_index.size(1));
  RuntimeCheck(static_cast<uint32_t>(accept_index.size(0)) == batch_size, "accept_index batch_size mismatch");

  RuntimeCheck(predicts.ndim() == 1, "predicts must be 1D");
  RuntimeCheck(predicts.dtype().code == kDLInt && predicts.dtype().bits == 32, "predicts must be int32");
  RuntimeCheck(predicts.is_contiguous(), "predicts must be contiguous");

  RuntimeCheck(accept_index.dtype().code == kDLInt && accept_index.dtype().bits == 32, "accept_index must be int32");
  RuntimeCheck(accept_index.is_contiguous(), "accept_index must be contiguous");

  RuntimeCheck(accept_token_num.ndim() == 1, "accept_token_num must be 1D");
  RuntimeCheck(
      accept_token_num.dtype().code == kDLInt && accept_token_num.dtype().bits == 32, "accept_token_num must be int32");
  RuntimeCheck(accept_token_num.is_contiguous(), "accept_token_num must be contiguous");
  RuntimeCheck(static_cast<uint32_t>(accept_token_num.size(0)) == batch_size, "accept_token_num batch_size mismatch");

  RuntimeCheck(candidates.ndim() == 2, "candidates must be 2D: [bs, num_draft_tokens]");
  RuntimeCheck(candidates.dtype().code == kDLInt && candidates.dtype().bits == 64, "candidates must be int64");
  RuntimeCheck(candidates.is_contiguous(), "candidates must be contiguous");
  RuntimeCheck(
      static_cast<uint32_t>(candidates.size(0)) == batch_size &&
          static_cast<uint32_t>(candidates.size(1)) == num_draft_tokens,
      "candidates shape mismatch");

  RuntimeCheck(retrive_index.ndim() == 2, "retrive_index must be 2D: [bs, num_draft_tokens]");
  RuntimeCheck(retrive_index.dtype().code == kDLInt && retrive_index.dtype().bits == 64, "retrive_index must be int64");
  RuntimeCheck(retrive_index.is_contiguous(), "retrive_index must be contiguous");
  RuntimeCheck(
      static_cast<uint32_t>(retrive_index.size(0)) == batch_size &&
          static_cast<uint32_t>(retrive_index.size(1)) == num_draft_tokens,
      "retrive_index shape mismatch");

  RuntimeCheck(retrive_next_token.ndim() == 2, "retrive_next_token must be 2D: [bs, num_draft_tokens]");
  RuntimeCheck(
      retrive_next_token.dtype().code == kDLInt && retrive_next_token.dtype().bits == 64,
      "retrive_next_token must be int64");
  RuntimeCheck(retrive_next_token.is_contiguous(), "retrive_next_token must be contiguous");
  RuntimeCheck(
      static_cast<uint32_t>(retrive_next_token.size(0)) == batch_size &&
          static_cast<uint32_t>(retrive_next_token.size(1)) == num_draft_tokens,
      "retrive_next_token shape mismatch");

  RuntimeCheck(retrive_next_sibling.ndim() == 2, "retrive_next_sibling must be 2D: [bs, num_draft_tokens]");
  RuntimeCheck(
      retrive_next_sibling.dtype().code == kDLInt && retrive_next_sibling.dtype().bits == 64,
      "retrive_next_sibling must be int64");
  RuntimeCheck(retrive_next_sibling.is_contiguous(), "retrive_next_sibling must be contiguous");
  RuntimeCheck(
      static_cast<uint32_t>(retrive_next_sibling.size(0)) == batch_size &&
          static_cast<uint32_t>(retrive_next_sibling.size(1)) == num_draft_tokens,
      "retrive_next_sibling shape mismatch");

  RuntimeCheck(uniform_samples.ndim() == 2, "uniform_samples must be 2D: [bs, num_draft_tokens]");
  RuntimeCheck(
      uniform_samples.dtype().code == kDLFloat && uniform_samples.dtype().bits == 32,
      "uniform_samples must be float32");
  RuntimeCheck(uniform_samples.is_contiguous(), "uniform_samples must be contiguous");
  RuntimeCheck(
      static_cast<uint32_t>(uniform_samples.size(0)) == batch_size &&
          static_cast<uint32_t>(uniform_samples.size(1)) == num_draft_tokens,
      "uniform_samples shape mismatch");

  RuntimeCheck(uniform_samples_for_final_sampling.ndim() == 1, "uniform_samples_for_final_sampling must be 1D: [bs]");
  RuntimeCheck(
      uniform_samples_for_final_sampling.dtype().code == kDLFloat &&
          uniform_samples_for_final_sampling.dtype().bits == 32,
      "uniform_samples_for_final_sampling must be float32");
  RuntimeCheck(
      uniform_samples_for_final_sampling.is_contiguous(), "uniform_samples_for_final_sampling must be contiguous");
  RuntimeCheck(
      static_cast<uint32_t>(uniform_samples_for_final_sampling.size(0)) == batch_size,
      "uniform_samples_for_final_sampling batch_size mismatch");

  RuntimeCheck(draft_probs.ndim() == 3, "draft_probs must be 3D: [bs, num_draft_tokens, vocab_size]");
  RuntimeCheck(draft_probs.dtype().code == kDLFloat && draft_probs.dtype().bits == 32, "draft_probs must be float32");
  RuntimeCheck(draft_probs.is_contiguous(), "draft_probs must be contiguous");
  RuntimeCheck(
      static_cast<uint32_t>(draft_probs.size(0)) == batch_size &&
          static_cast<uint32_t>(draft_probs.size(1)) == num_draft_tokens &&
          static_cast<uint32_t>(draft_probs.size(2)) == vocab_size,
      "draft_probs shape mismatch");

  RuntimeCheck(threshold_single >= 0.0f && threshold_single <= 1.0f, "threshold_single must be in [0, 1]");
  RuntimeCheck(threshold_acc >= 0.0f && threshold_acc <= 1.0f, "threshold_acc must be in [0, 1]");

  cudaStream_t stream = LaunchKernel::resolve_device(target_probs.device());

  cudaError_t status = flashinfer::sampling::TreeSpeculativeSamplingTargetOnlyLauncher<float, int32_t, int64_t>(
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
      threshold_single,
      threshold_acc,
      static_cast<bool>(deterministic),
      stream);

  RuntimeCheck(status == cudaSuccess, "TreeSpeculativeSamplingTargetOnly failed: ", cudaGetErrorString(status));
}

}  // namespace
