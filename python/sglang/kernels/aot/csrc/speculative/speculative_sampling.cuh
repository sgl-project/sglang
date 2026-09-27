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
#ifndef SPECULATIVE_SAMPLING_CUH_
#define SPECULATIVE_SAMPLING_CUH_

#include <assert.h>

#include <sgl_kernel/speculative/sampling.cuh>

namespace flashinfer {

namespace sampling {

using namespace cub;

template <typename DType, typename IdType, typename IdType2>
cudaError_t TreeSpeculativeSamplingTargetOnly(
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
        auto kernel = sglang::speculative_sampling::TreeSpeculativeSamplingTargetOnly<
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

#endif  // SPECULATIVE_SAMPLING_CUH_
