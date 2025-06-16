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
#define float32_t float
#define bfloat16_t at::BFloat16

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
    float32_t* topk_p_dst,
    int64_t* topk_index_dst,
    const input_id_t* input_ids_src,
    const int64_t* seq_lens_src,
    const int32_t* extend_seq_lens_src,
    const int64_t* out_cache_loc_src,
    const int64_t* positions_src,
    const int64_t* req_pool_indices_src,
    const int32_t* accept_length_src,
    const hidden_state_t* hidden_states_src,
    const float32_t* topk_p_src,
    const int64_t* topk_index_src,
    int64_t num_tokens,
    int64_t max_num_tokens,
    int64_t raw_bs,
    int64_t max_bs,
    int64_t num_hidden_states,
    int64_t hidden_size,
    int64_t num_speculative_steps,
    int64_t speculative_topk) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // fill seq_lens_dst with 1
  for (int i = tid; i < max_bs; i += stride) {
    seq_lens_dst[i] = 1;
  }

  // fill out_cache_loc_dst with 0
  int max_num_out_cache_loc = num_speculative_steps > 0 ? max_num_tokens * num_speculative_steps : max_num_tokens;
  for (int i = tid; i < max_num_out_cache_loc; i += stride) {
    out_cache_loc_dst[i] = 0;
  }

  // fill positions_dst with 0
  for (int i = tid; i < max_num_tokens; i += stride) {
    positions_dst[i] = 0;
  }

  // fill accept_length_dst with 1
  if (accept_length_dst != nullptr) {
    for (int i = tid; i < max_bs; i += stride) {
      accept_length_dst[i] = 1;
    }
  }

  // copy positions
  for (int i = tid; i < num_tokens; i += stride) {
    positions_dst[i] = positions_src[i];
  }

  // copy input_ids
  if (input_ids_src != nullptr && input_ids_dst != nullptr) {
    for (int i = tid; i < num_tokens; i += stride) {
      input_ids_dst[i] = static_cast<int64_t>(input_ids_src[i]);
    }
  }

  // copy out_cache_loc
  int num_out_cache_loc = num_speculative_steps > 0 ? num_tokens * num_speculative_steps : num_tokens;
  for (int i = tid; i < num_out_cache_loc; i += stride) {
    out_cache_loc_dst[i] = out_cache_loc_src[i];
  }

  // copy seq_lens and req_pool_indices
  for (int i = tid; i < raw_bs; i += stride) {
    seq_lens_dst[i] = seq_lens_src[i];
    req_pool_indices_dst[i] = req_pool_indices_src[i];
  }

  // copy extend_seq_lens and accept_length
  for (int i = tid; i < raw_bs; i += stride) {
    if (extend_seq_lens_src != nullptr && extend_seq_lens_dst != nullptr) {
      extend_seq_lens_dst[i] = extend_seq_lens_src[i];
    }
    if (accept_length_src != nullptr && accept_length_dst != nullptr) {
      accept_length_dst[i] = accept_length_src[i];
    }
  }

  // copy topk_p and topk_index
  for (int i = tid; i < raw_bs * speculative_topk; i += stride) {
    if (topk_p_src != nullptr && topk_p_dst != nullptr) {
      topk_p_dst[i] = topk_p_src[i];
    }
    if (topk_index_src != nullptr && topk_index_dst != nullptr) {
      topk_index_dst[i] = topk_index_src[i];
    }
  }

  // copy hidden_states
  if (hidden_states_src != nullptr && hidden_states_dst != nullptr) {
    for (int i = tid; i < num_hidden_states * hidden_size; i += stride) {
      hidden_states_dst[i] = hidden_states_src[i];
    }
  }
}

#define DISPATCH_INT_TYPES(TYPE, NAME, ...)          \
  [&] {                                              \
    const auto& the_type = TYPE;                     \
    switch (the_type) {                              \
      case at::ScalarType::Int: {                    \
        using scalar_t = int32_t;                    \
        return __VA_ARGS__();                        \
      }                                              \
      case at::ScalarType::Long: {                   \
        using scalar_t = int64_t;                    \
        return __VA_ARGS__();                        \
      }                                              \
      default:                                       \
        AT_ERROR(NAME, " not supported for ", TYPE); \
    }                                                \
  }()

#define DISPATCH_FLOAT_TYPES(TYPE, NAME, ...)        \
  [&] {                                              \
    const auto& the_type = TYPE;                     \
    switch (the_type) {                              \
      case at::ScalarType::Half: {                   \
        using scalar_t = float16_t;                  \
        return __VA_ARGS__();                        \
      }                                              \
      case at::ScalarType::BFloat16: {               \
        using scalar_t = bfloat16_t;                 \
        return __VA_ARGS__();                        \
      }                                              \
      case at::ScalarType::Float: {                  \
        using scalar_t = float;                      \
        return __VA_ARGS__();                        \
      }                                              \
      default:                                       \
        AT_ERROR(NAME, " not supported for ", TYPE); \
    }                                                \
  }()

void copy_cuda_graph_replay_inputs(
    at::Tensor seq_lens_dst,
    at::Tensor seq_lens_src,
    at::Tensor out_cache_loc_dst,
    at::Tensor out_cache_loc_src,
    at::Tensor positions_dst,
    at::Tensor positions_src,
    at::Tensor req_pool_indices_dst,
    at::Tensor req_pool_indices_src,
    c10::optional<at::Tensor> input_ids_dst,
    c10::optional<at::Tensor> input_ids_src,
    c10::optional<at::Tensor> extend_seq_lens_dst,
    c10::optional<at::Tensor> extend_seq_lens_src,
    c10::optional<at::Tensor> accept_length_dst,
    c10::optional<at::Tensor> accept_length_src,
    c10::optional<at::Tensor> hidden_states_dst,
    c10::optional<at::Tensor> hidden_states_src,
    c10::optional<at::Tensor> topk_p_dst,
    c10::optional<at::Tensor> topk_p_src,
    c10::optional<at::Tensor> topk_index_dst,
    c10::optional<at::Tensor> topk_index_src,
    int64_t num_tokens,
    int64_t raw_bs,
    int64_t num_hidden_states,
    int64_t hidden_size,
    int64_t num_speculative_steps,
    int64_t speculative_topk) {
  TORCH_CHECK(seq_lens_dst.dtype() == seq_lens_src.dtype(), "seq_lens_dst and seq_lens_src must have the same dtype");
  TORCH_CHECK(
      out_cache_loc_dst.dtype() == out_cache_loc_src.dtype(),
      "out_cache_loc_dst and out_cache_loc_src must have the same dtype");
  TORCH_CHECK(
      positions_dst.dtype() == positions_src.dtype(), "positions_dst and positions_src must have the same dtype");
  TORCH_CHECK(
      req_pool_indices_dst.dtype() == req_pool_indices_src.dtype(),
      "req_pool_indices_dst and req_pool_indices_src must have the same dtype");

  if (extend_seq_lens_dst) {
    TORCH_CHECK(
        extend_seq_lens_dst->dtype() == extend_seq_lens_src->dtype(),
        "extend_seq_lens_dst and extend_seq_lens_src must have the same dtype");
  }
  if (accept_length_dst) {
    TORCH_CHECK(
        accept_length_dst->dtype() == accept_length_src->dtype(),
        "accept_length_dst and accept_length_src must have the same dtype");
  }
  if (hidden_states_dst) {
    TORCH_CHECK(
        hidden_states_dst->dtype() == hidden_states_src->dtype(),
        "hidden_states_dst and hidden_states_src must have the same dtype");
    TORCH_CHECK(
        hidden_states_dst->dtype() == at::ScalarType::Half || hidden_states_dst->dtype() == at::ScalarType::BFloat16 ||
            hidden_states_dst->dtype() == at::ScalarType::Float,
        "hidden_states_dst must be Float16, BFloat16 or Float");
  }
  if (topk_p_dst) {
    TORCH_CHECK(topk_p_dst->dtype() == topk_p_src->dtype(), "topk_p_dst and topk_p_src must have the same dtype");
    TORCH_CHECK(speculative_topk > 0, "speculative_topk must be greater than 0 if topk_p_dst is not null");
  }
  if (topk_index_dst) {
    TORCH_CHECK(
        topk_index_dst->dtype() == topk_index_src->dtype(),
        "topk_index_dst and topk_index_src must have the same dtype");
    TORCH_CHECK(speculative_topk > 0, "speculative_topk must be greater than 0 if topk_index_dst is not null");
  }
  if (input_ids_dst) {
    TORCH_CHECK(input_ids_dst->dtype() == at::ScalarType::Long, "input_ids_dst must be Long");
  }
  if (input_ids_src) {
    TORCH_CHECK(
        input_ids_src->dtype() == at::ScalarType::Long || input_ids_src->dtype() == at::ScalarType::Int,
        "input_ids_src must be Long or Int");
  }

  int64_t max_bs = seq_lens_dst.numel();
  int64_t max_num_tokens = positions_dst.numel();

  int64_t num_threads = 1024;
  int64_t max_size = std::max(
      std::max(std::max(max_num_tokens, max_bs), num_hidden_states * hidden_size),
      max_num_tokens * num_speculative_steps);
  int64_t num_blocks = (max_size + num_threads - 1) / num_threads;

  auto input_ids_dtype = input_ids_src ? input_ids_src->scalar_type() : at::ScalarType::Int;
  auto hidden_states_dtype = hidden_states_src ? hidden_states_src->scalar_type() : at::ScalarType::Half;

  DISPATCH_INT_TYPES(input_ids_dtype, "dispatch_input_ids", [&]() {
    using input_id_t = scalar_t;
    DISPATCH_FLOAT_TYPES(hidden_states_dtype, "dispatch_hidden_states", [&]() {
      using hidden_state_t = scalar_t;
      copy_cuda_graph_replay_inputs_kernel<input_id_t, hidden_state_t><<<num_blocks, num_threads>>>(
          input_ids_dst ? input_ids_dst->data_ptr<int64_t>() : nullptr,
          seq_lens_dst.data_ptr<int64_t>(),
          extend_seq_lens_dst ? extend_seq_lens_dst->data_ptr<int32_t>() : nullptr,
          out_cache_loc_dst.data_ptr<int64_t>(),
          positions_dst.data_ptr<int64_t>(),
          req_pool_indices_dst.data_ptr<int64_t>(),
          accept_length_dst ? accept_length_dst->data_ptr<int32_t>() : nullptr,
          hidden_states_dst ? hidden_states_dst->data_ptr<hidden_state_t>() : nullptr,
          topk_p_dst ? topk_p_dst->data_ptr<float32_t>() : nullptr,
          topk_index_dst ? topk_index_dst->data_ptr<int64_t>() : nullptr,
          input_ids_src ? input_ids_src->data_ptr<input_id_t>() : nullptr,
          seq_lens_src.data_ptr<int64_t>(),
          extend_seq_lens_src ? extend_seq_lens_src->data_ptr<int32_t>() : nullptr,
          out_cache_loc_src.data_ptr<int64_t>(),
          positions_src.data_ptr<int64_t>(),
          req_pool_indices_src.data_ptr<int64_t>(),
          accept_length_src ? accept_length_src->data_ptr<int32_t>() : nullptr,
          hidden_states_src ? hidden_states_src->data_ptr<hidden_state_t>() : nullptr,
          topk_p_src ? topk_p_src->data_ptr<float32_t>() : nullptr,
          topk_index_src ? topk_index_src->data_ptr<int64_t>() : nullptr,
          num_tokens,
          max_num_tokens,
          raw_bs,
          max_bs,
          num_hidden_states,
          hidden_size,
          num_speculative_steps,
          speculative_topk);
    });
  });
}
