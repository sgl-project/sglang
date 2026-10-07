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

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/speculative/sampling.cuh>

#include <flashinfer/sampling.cuh>

#include <numeric>

namespace sglang {

/// \brief Sample and verify a draft tree using only target probabilities.
template <uint32_t VecSize, bool Deterministic>
void tree_speculative_sampling_target_only(
    tvm::ffi::TensorView predicts,
    tvm::ffi::TensorView accept_index,
    tvm::ffi::TensorView accept_token_num,
    tvm::ffi::TensorView candidates,
    tvm::ffi::TensorView retrive_index,
    tvm::ffi::TensorView retrive_next_token,
    tvm::ffi::TensorView retrive_next_sibling,
    tvm::ffi::TensorView uniform_samples,
    tvm::ffi::TensorView uniform_samples_for_final_sampling,
    tvm::ffi::TensorView target_probs,
    tvm::ffi::TensorView draft_probs,
    double threshold_single,
    double threshold_acc) {
  using namespace host;
  using namespace flashinfer::sampling;
  static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4);
  SymbolicSize batch_size{"batch_size"}, draft_tokens{"draft_tokens"}, spec_tokens{"spec_tokens"},
      vocab_size{"vocab_size"};
  SymbolicDevice device;
  TensorMatcher({batch_size, draft_tokens})
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device)
      .verify(candidates)
      .verify(retrive_index)
      .verify(retrive_next_token)
      .verify(retrive_next_sibling);
  TensorMatcher({batch_size, spec_tokens}).with_dtype<int32_t>().with_device(device).verify(accept_index);
  TensorMatcher({batch_size}).with_dtype<int32_t>().with_device(device).verify(accept_token_num);
  TensorMatcher({batch_size.unwrap() * draft_tokens.unwrap()})
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(predicts);
  TensorMatcher({batch_size, draft_tokens}).with_dtype<float>().with_device(device).verify(uniform_samples);
  TensorMatcher({batch_size}).with_dtype<float>().with_device(device).verify(uniform_samples_for_final_sampling);
  TensorMatcher({batch_size, draft_tokens, vocab_size})
      .with_dtype<float>()
      .with_device(device)
      .verify(target_probs)
      .verify(draft_probs);
  CHECK_HOST(draft_tokens.unwrap() > 0 && spec_tokens.unwrap() > 0 && vocab_size.unwrap() > 0);
  CHECK_HOST(std::gcd(int64_t{4}, vocab_size.unwrap()) == VecSize);
  CHECK_HOST(threshold_single >= 0 && threshold_single <= 1);
  CHECK_HOST(threshold_acc >= 0 && threshold_acc <= 1);
  if (batch_size.unwrap() == 0) return;

  constexpr uint32_t block_threads = 1024;
  constexpr size_t smem_size = sizeof(SamplingTempStorage<block_threads, SCAN_ALGO, REDUCE_ALGO>);
  auto kernel = speculative_sampling::TreeSpeculativeSamplingTargetOnly<
      block_threads,
      SCAN_ALGO,
      REDUCE_ALGO,
      VecSize,
      Deterministic,
      float,
      int32_t,
      int64_t>;
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  LaunchKernel(static_cast<uint32_t>(batch_size.unwrap()), block_threads, device.unwrap(), smem_size)(
      kernel,
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
      static_cast<uint32_t>(batch_size.unwrap()),
      static_cast<uint32_t>(spec_tokens.unwrap()),
      static_cast<uint32_t>(draft_tokens.unwrap()),
      static_cast<uint32_t>(vocab_size.unwrap()),
      static_cast<float>(threshold_single),
      std::max(static_cast<float>(threshold_acc), 1e-9f));
}

}  // namespace sglang
