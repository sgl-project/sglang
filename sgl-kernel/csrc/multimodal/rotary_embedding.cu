/*
 * Copyright (c) 2025 by SGLang team.
 * Copyright (c) 2025 by FlashInfer team.
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
#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <flashinfer/vec_dtypes.cuh>

#include "utils.h"
// #include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr,
    int rot_offset,
    int embed_dim) {
  int x_index, y_index;
  scalar_t cos_val_x, sin_val_x;
  if (IS_NEOX) {
    // NEOX-specific case (unchanged)
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos_val_x = SGLANG_LDG(cos_ptr + rot_offset);
    sin_val_x = SGLANG_LDG(sin_ptr + rot_offset);
  } else {
    // GPT-J style - modified to match Python implementation
    x_index = rot_offset;
    y_index = rot_offset + embed_dim;
    cos_val_x = SGLANG_LDG(cos_ptr + rot_offset);
    sin_val_x = SGLANG_LDG(sin_ptr + rot_offset);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];

  // Modified to match Python implementation
  // Python: q_embed = (q * cos) + (rotate_half(q) * sin)
  // Where rotate_half negates the second half
  arr[x_index] = x * cos_val_x - y * sin_val_x;  // First half: q[i]*cos[i] - q[i+half]*sin[i]
  arr[y_index] = y * cos_val_x + x * sin_val_x;  // Second half: q[i+half]*cos[i] + q[i]*sin[i]
}

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
    scalar_t* __restrict__ query,                        // [num_heads, head_size] for current token
    scalar_t* __restrict__ key,                          // nullptr or [num_kv_heads, head_size] for current token
    const scalar_t* __restrict__ current_token_cos_ptr,  // [rot_dim/2] for current token
    const scalar_t* __restrict__ current_token_sin_ptr,  // [rot_dim/2] for current token
    const int head_size,
    const int num_heads,
    const int num_kv_heads,
    const int rot_dim,
    const int64_t head_stride_query,  // Stride to get to next head in query
    const int64_t head_stride_key     // Stride to get to next head in key
) {
  const int embed_dim = rot_dim / 2;  // Number of elements in cos/sin arrays for one token

  // No need to offset current_token_cos_ptr and current_token_sin_ptr further here,
  // they already point to the start of cos/sin values for the current token.

  const int nq_pairs = num_heads * embed_dim;  // Total pairs to rotate for query
  for (int i = threadIdx.x; i < nq_pairs; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int rot_offset = i % embed_dim;  // Offset within the head's part to be rotated

    // query_for_token_head points to the start of the specific head for the current token
    scalar_t* query_for_token_head = query + head_idx * (int)head_stride_query;

    apply_token_rotary_embedding<scalar_t, IS_NEOX>(
        query_for_token_head, current_token_cos_ptr, current_token_sin_ptr, rot_offset, embed_dim);
  }

  if (key != nullptr) {
    const int nk_pairs = num_kv_heads * embed_dim;  // Total pairs to rotate for key
    for (int i = threadIdx.x; i < nk_pairs; i += blockDim.x) {
      const int head_idx = i / embed_dim;
      const int rot_offset = i % embed_dim;

      // key_for_token_head points to the start of the specific head for the current token
      scalar_t* key_for_token_head = key + head_idx * (int)head_stride_key;

      apply_token_rotary_embedding<scalar_t, IS_NEOX>(
          key_for_token_head, current_token_cos_ptr, current_token_sin_ptr, rot_offset, embed_dim);
    }
  }
}

template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
    const scalar_t* __restrict__ cos_data,  // [num_tokens, rot_dim]
    const scalar_t* __restrict__ sin_data,  // [num_tokens, rot_dim]
    scalar_t* __restrict__ query_total,     // [num_tokens, num_heads, head_size] or [num_tokens, num_heads * head_size]
    scalar_t* __restrict__ key_total,       // nullptr or similar shape to query_total
    const int rot_dim,
    const int64_t query_token_stride,  // Elements to skip to get to next token in query_total
    const int64_t key_token_stride,    // Elements to skip to get to next token in key_total
    const int64_t head_stride_query,   // Elements to skip to get to next head within a token's query data
    const int64_t head_stride_key,     // Elements to skip to get to next head within a token's key data
    const int num_heads,
    const int num_kv_heads,
    const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  const int embed_dim = rot_dim / 2;

  const scalar_t* current_token_cos_ptr = cos_data + token_idx * rot_dim;
  const scalar_t* current_token_sin_ptr = sin_data + token_idx * rot_dim;

  scalar_t* query_for_token = query_total + token_idx * (int)query_token_stride;
  scalar_t* key_for_token = (key_total != nullptr) ? (key_total + token_idx * (int)key_token_stride) : nullptr;

  apply_rotary_embedding<scalar_t, IS_NEOX>(
      query_for_token,
      key_for_token,
      current_token_cos_ptr,
      current_token_sin_ptr,
      head_size,
      num_heads,
      num_kv_heads,
      rot_dim,
      head_stride_query,
      head_stride_key);
}

void rotary_embedding(
    at::Tensor& cos_cache,                 // [num_tokens, rot_dim]
    at::Tensor& sin_cache,                 // [num_tokens, rot_dim]
    at::Tensor& query,                     // [num_tokens, num_heads, head_size]
    const std::optional<at::Tensor>& key,  // null or similar to query
    int64_t head_size,
    bool is_neox) {
  TORCH_CHECK(
      query.dim() == 2 || query.dim() == 3,
      "query must have shape [num_tokens, hidden_size] or [num_tokens, num_heads, head_size]");
  if (key.has_value()) {
    TORCH_CHECK(
        key->dim() == 2 || key->dim() == 3,
        "key must have shape [num_tokens, hidden_size] or [num_tokens, num_kv_heads, head_size]");
  }

  int64_t num_tokens = query.size(0);

  TORCH_CHECK(cos_cache.dim() == 2, "cos_cache must have shape [num_tokens, rot_dim/2]");
  TORCH_CHECK(sin_cache.dim() == 2, "sin_cache must have shape [num_tokens, rot_dim/2]");
  TORCH_CHECK(cos_cache.size(0) == num_tokens, "cos_cache num_tokens mismatch with query");
  TORCH_CHECK(sin_cache.size(0) == num_tokens, "sin_cache num_tokens mismatch with query");
  TORCH_CHECK(cos_cache.size(1) == sin_cache.size(1), "cos_cache and sin_cache rot_dim/2 mismatch");

  TORCH_CHECK(cos_cache.scalar_type() == query.scalar_type(), "cos_cache dtype mismatch");
  TORCH_CHECK(sin_cache.scalar_type() == query.scalar_type(), "sin_cache dtype mismatch");
  TORCH_CHECK(cos_cache.is_cuda() && sin_cache.is_cuda() && query.is_cuda(), "All tensors must be on CUDA");
  if (key.has_value()) {
    TORCH_CHECK(key->is_cuda(), "Key tensor must be on CUDA if provided");
    TORCH_CHECK(key->scalar_type() == query.scalar_type(), "Key dtype mismatch");
  }

  // hidden_size = num_heads * head_size
  int query_hidden_size_calculated;
  if (query.dim() == 2) {  // [num_tokens, hidden_size]
    query_hidden_size_calculated = (int)query.size(1);
  } else {  // [num_tokens, num_heads, head_size]
    query_hidden_size_calculated = (int)query.size(1) * (int)query.size(2);
    TORCH_CHECK(query.size(2) == head_size, "Query head_size mismatch in 3D tensor");
  }
  TORCH_CHECK(query_hidden_size_calculated % head_size == 0, "query_hidden_size not divisible by head_size");
  int num_heads = (int)query_hidden_size_calculated / (int)head_size;

  int key_hidden_size_calculated = 0;
  int num_kv_heads = num_heads;  // Default if key is not present or GQA not used
  if (key.has_value()) {
    TORCH_CHECK((int)key->size(0) == num_tokens, "Key num_tokens mismatch");
    if (key->dim() == 2) {  // [num_tokens, kv_hidden_size]
      key_hidden_size_calculated = (int)key->size(1);
    } else {  // [num_tokens, num_kv_heads, head_size]
      key_hidden_size_calculated = (int)key->size(1) * (int)key->size(2);
      TORCH_CHECK((int)key->size(2) == head_size, "Key head_size mismatch in 3D tensor");
    }
    TORCH_CHECK(key_hidden_size_calculated % head_size == 0, "key_hidden_size not divisible by head_size");
    num_kv_heads = key_hidden_size_calculated / (int)head_size;
  }
  TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");

  int rot_dim = (int)cos_cache.size(1);
  //  TORCH_CHECK(rot_dim <= head_size, "rot_dim must be <= head_size");

  // Strides to get to the next token's data
  int64_t query_token_stride = query_hidden_size_calculated;
  int64_t key_token_stride = key.has_value() ? key_hidden_size_calculated : 0;

  // Strides to get to the next head's data *within* a token
  // If query is [num_tokens, num_heads, head_size], stride is query.stride(1)
  // If query is [num_tokens, num_heads * head_size], stride is head_size
  int64_t head_stride_query;
  if (query.dim() == 3 && query.size(1) == num_heads && query.size(2) == head_size) {
    head_stride_query = query.stride(1);
  } else {  // Assumed to be [num_tokens, num_heads * head_size] or will be viewed as such
    head_stride_query = head_size;
  }

  int64_t head_stride_key = head_size;  // Default for key
  if (key.has_value()) {
    if (key->dim() == 3 && key->size(1) == num_kv_heads && key->size(2) == head_size) {
      head_stride_key = key->stride(1);
    } else {
      head_stride_key = head_size;
    }
  }

  dim3 grid((int)num_tokens);
  // Max threads per block is usually 1024.
  // Each thread handles one pair (x,y) to rotate.
  // Total pairs for query for one token: num_heads * (rot_dim / 2)
  // We want enough threads to cover these pairs, up to a limit.
  // The loop inside apply_rotary_embedding handles thread stride.
  int max_pairs_to_rotate_per_token = std::max(num_heads * (rot_dim / 2), num_kv_heads * (rot_dim / 2));
  dim3 block(std::min<int64_t>(max_pairs_to_rotate_per_token, 512L));  // 512L to ensure long comparison
  if (block.x == 0 && num_tokens > 0) block.x = 1;                     // Ensure at least one thread if there's work

  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  SGLANG_DISPATCH_FLOATING_TYPES(query.scalar_type(), "rotary_embedding", [&] {
    if (is_neox) {
      rotary_embedding_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
          cos_cache.data_ptr<scalar_t>(),
          sin_cache.data_ptr<scalar_t>(),
          query.data_ptr<scalar_t>(),
          key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
          rot_dim,
          query_token_stride,
          key_token_stride,
          head_stride_query,
          head_stride_key,
          num_heads,
          num_kv_heads,
          head_size);
    } else {
      rotary_embedding_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
          cos_cache.data_ptr<scalar_t>(),
          sin_cache.data_ptr<scalar_t>(),
          query.data_ptr<scalar_t>(),
          key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
          rot_dim,
          query_token_stride,
          key_token_stride,
          head_stride_query,
          head_stride_key,
          num_heads,
          num_kv_heads,
          head_size);
    }
  });
  // C10_CUDA_KERNEL_LAUNCH_CHECK();
}
