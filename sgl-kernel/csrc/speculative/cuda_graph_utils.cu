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

#define float16_t at::Half

template <typename input_id_t, typename hidden_state_t>
__global__ void copy_cuda_graph_replay_inputs_kernel(
    int64_t* input_ids_dst,
    int64_t* seq_lens_dst,
    int32_t* extend_seq_lens_dst,
    int64_t* out_cache_loc_dst,
    int64_t* positions_dst,
    int64_t* req_pool_indices_dst,
    int32_t* accept_length_dst,
    hidden_state_t* hidden_states_dst,
    const input_id_t* input_ids_src,
    const int64_t* seq_lens_src,
    const int32_t* extend_seq_lens_src,
    const int64_t* out_cache_loc_src,
    const int64_t* positions_src,
    const int64_t* req_pool_indices_src,
    const int32_t* accept_length_src,
    const hidden_state_t* hidden_states_src,
    int64_t num_tokens,
    int64_t raw_bs,
    int64_t hidden_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

#pragma unroll
  for (int i = tid; i < num_tokens; i += stride) {
    if (input_ids_src != nullptr) {
      input_ids_dst[i] = static_cast<int64_t>(input_ids_src[i]);
    }
    if (out_cache_loc_src != nullptr) {
      out_cache_loc_dst[i] = out_cache_loc_src[i];
    }
    if (positions_src != nullptr) {
      positions_dst[i] = positions_src[i];
    }
  }

#pragma unroll
  for (int i = tid; i < raw_bs; i += stride) {
    if (seq_lens_src != nullptr) {
      seq_lens_dst[i] = seq_lens_src[i];
    }
    if (extend_seq_lens_src != nullptr) {
      extend_seq_lens_dst[i] = extend_seq_lens_src[i];
    }
    if (accept_length_src != nullptr) {
      accept_length_dst[i] = accept_length_src[i];
    }
    if (req_pool_indices_src != nullptr) {
      req_pool_indices_dst[i] = req_pool_indices_src[i];
    }
  }

#pragma unroll
  for (int i = tid; i < num_tokens * hidden_size; i += stride) {
    if (hidden_states_src != nullptr) {
      hidden_states_dst[i] = hidden_states_src[i];
    }
  }
}

void copy_cuda_graph_replay_inputs(
    at::Tensor input_ids_dst,
    at::Tensor seq_lens_dst,
    at::Tensor extend_seq_lens_dst,
    at::Tensor out_cache_loc_dst,
    at::Tensor positions_dst,
    at::Tensor req_pool_indices_dst,
    at::Tensor accept_length_dst,
    at::Tensor hidden_states_dst,
    at::Tensor input_ids_src,
    at::Tensor seq_lens_src,
    at::Tensor extend_seq_lens_src,
    at::Tensor out_cache_loc_src,
    at::Tensor positions_src,
    at::Tensor req_pool_indices_src,
    at::Tensor accept_length_src,
    at::Tensor hidden_states_src,
    int64_t num_tokens,
    int64_t raw_bs,
    int64_t hidden_size) {
  int64_t num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int64_t num_threads = 1024;
  int64_t num_blocks = std::min(num_sm, (num_tokens * hidden_size + num_threads - 1) / num_threads);

  copy_cuda_graph_replay_inputs_kernel<int32_t, float16_t><<<num_blocks, num_threads>>>(
      input_ids_dst.data_ptr<int64_t>(),
      seq_lens_dst.data_ptr<int64_t>(),
      extend_seq_lens_dst.data_ptr<int32_t>(),
      out_cache_loc_dst.data_ptr<int64_t>(),
      positions_dst.data_ptr<int64_t>(),
      req_pool_indices_dst.data_ptr<int64_t>(),
      accept_length_dst.data_ptr<int32_t>(),
      hidden_states_dst.data_ptr<float16_t>(),
      input_ids_src.data_ptr<int32_t>(),
      seq_lens_src.data_ptr<int64_t>(),
      extend_seq_lens_src.data_ptr<int32_t>(),
      out_cache_loc_src.data_ptr<int64_t>(),
      positions_src.data_ptr<int64_t>(),
      req_pool_indices_src.data_ptr<int64_t>(),
      accept_length_src.data_ptr<int32_t>(),
      hidden_states_src.data_ptr<float16_t>(),
      num_tokens,
      raw_bs,
      hidden_size);
}
