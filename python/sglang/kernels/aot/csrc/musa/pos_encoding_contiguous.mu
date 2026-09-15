/*
 * Copyright (c) 2020-2026, Moore Threads Technology Co., Ltd("Moore Threads").
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>

#include "musa.h"
#include "musa/dispatch_utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding_contiguous(
    scalar_t* __restrict__ arr, const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr, int rot_offset, int embed_dim,
    int64_t head_stride) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = MUSA_LDG(cos_ptr + x_index);
    sin = MUSA_LDG(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = MUSA_LDG(cos_ptr + x_index / 2);
    sin = MUSA_LDG(sin_ptr + x_index / 2);
  }

  scalar_t* x_ptr = arr + x_index * head_stride;
  scalar_t* y_ptr = arr + y_index * head_stride;

  const scalar_t x = *x_ptr;
  const scalar_t y = *y_ptr;
  *x_ptr = x * cos - y * sin;
  *y_ptr = y * cos + x * sin;
}

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding_contiguous(
    scalar_t* __restrict__ query,  // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key,    // [num_tokens, num_kv_heads, head_size]
    const scalar_t* cache_ptr, const int head_size, const int num_heads,
    const int num_kv_heads, const int rot_dim, const int token_idx,
    const int64_t query_token_stride, const int64_t query_head_stride,
    const int64_t query_dim_stride,
    const int64_t key_token_stride, const int64_t key_head_stride,
    const int64_t key_dim_stride) {
  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int rot_offset = i % embed_dim;

    scalar_t* head_query = query +
                           token_idx * query_token_stride +
                           head_idx * query_head_stride;

    apply_token_rotary_embedding_contiguous<scalar_t, IS_NEOX>(
        head_query, cos_ptr, sin_ptr, rot_offset, embed_dim, query_dim_stride);
  }

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int rot_offset = i % embed_dim;

    scalar_t* head_key = key +
                         token_idx * key_token_stride +
                         head_idx * key_head_stride;

    apply_token_rotary_embedding_contiguous<scalar_t, IS_NEOX>(
        head_key, cos_ptr, sin_ptr, rot_offset, embed_dim, key_dim_stride);
  }
}

template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel_contiguous(
    const int64_t* __restrict__ positions,  // [num_tokens]
    scalar_t* __restrict__ query,           // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key,             // [num_tokens, num_kv_heads, head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim // 2]
    const int rot_dim,
    const int64_t query_token_stride, const int64_t query_head_stride,
    const int64_t query_dim_stride,
    const int64_t key_token_stride, const int64_t key_head_stride,
    const int64_t key_dim_stride,
    const int num_heads, const int num_kv_heads, const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding_contiguous<scalar_t, IS_NEOX>(
      query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim,
      token_idx,
      query_token_stride, query_head_stride, query_dim_stride,
      key_token_stride, key_head_stride, key_dim_stride);
}

template <typename scalar_t, bool IS_NEOX>
__global__ void batched_rotary_embedding_kernel_contiguous(
    const int64_t* __restrict__ positions,  // [num_tokens]
    scalar_t* __restrict__ query,           // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key,             // [num_tokens, num_kv_heads, head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim // 2]
    const int64_t* __restrict__ cos_sin_cache_offsets,  // [num_tokens]
    const int rot_dim,
    const int64_t query_token_stride, const int64_t query_head_stride,
    const int64_t query_dim_stride,  // stride for each dimension
    const int64_t key_token_stride, const int64_t key_head_stride,
    const int64_t key_dim_stride,
    const int num_heads, const int num_kv_heads, const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  int64_t cos_sin_cache_offset = cos_sin_cache_offsets[token_idx];
  const scalar_t* cache_ptr =
      cos_sin_cache + (cos_sin_cache_offset + pos) * rot_dim;

  apply_rotary_embedding_contiguous<scalar_t, IS_NEOX>(
      query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim,
      token_idx,
      query_token_stride, query_head_stride, query_dim_stride,
      key_token_stride, key_head_stride, key_dim_stride);
}


void rotary_embedding_contiguous(
    torch::Tensor& positions,  // [num_tokens]
    torch::Tensor& query,      // [num_tokens, num_heads, head_size]
    torch::Tensor& key,        // [num_tokens, num_kv_heads, head_size]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox) {
  int64_t num_tokens = positions.size(0);

  TORCH_CHECK(query.dim() == 3, "query must be 3D [num_tokens, num_heads, head_size]");
  TORCH_CHECK(key.dim() == 3, "key must be 3D [num_tokens, num_kv_heads, head_size]");
  TORCH_CHECK(query.size(0) == num_tokens && key.size(0) == num_tokens,
              "query, key and positions must have the same number of tokens");

  int64_t query_token_stride = query.stride(0);
  int64_t query_head_stride = query.stride(1);
  int64_t query_dim_stride = query.stride(2);

  int64_t key_token_stride = key.stride(0);
  int64_t key_head_stride = key.stride(1);
  int64_t key_dim_stride = key.stride(2);

  int num_heads = query.size(1);
  int num_kv_heads = key.size(1);
  int rot_dim = cos_sin_cache.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));

  const at::musa::OptionalMUSAGuard device_guard(device_of(query));
  const musaStream_t stream = at::musa::getCurrentMUSAStream();

  MUSA_DISPATCH_FLOATING_TYPES(query.scalar_type(), "rotary_embedding_contiguous", [&] {
    if (is_neox) {
      rotary_embedding_kernel_contiguous<scalar_t, true><<<grid, block, 0, stream>>>(
          positions.data_ptr<int64_t>(),
          query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(),
          cos_sin_cache.data_ptr<scalar_t>(),
          rot_dim,
          query_token_stride, query_head_stride, query_dim_stride,
          key_token_stride, key_head_stride, key_dim_stride,
          num_heads, num_kv_heads, head_size);
    } else {
      rotary_embedding_kernel_contiguous<scalar_t, false><<<grid, block, 0, stream>>>(
          positions.data_ptr<int64_t>(),
          query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(),
          cos_sin_cache.data_ptr<scalar_t>(),
          rot_dim,
          query_token_stride, query_head_stride, query_dim_stride,
          key_token_stride, key_head_stride, key_dim_stride,
          num_heads, num_kv_heads, head_size);
    }
  });
}

void batched_rotary_embedding_contiguous(
    torch::Tensor& positions,  // [num_tokens]
    torch::Tensor& query,      // [num_tokens, num_heads, head_size]
    torch::Tensor& key,        // [num_tokens, num_kv_heads, head_size]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox, int64_t rot_dim,
    torch::Tensor& cos_sin_cache_offsets  // [num_tokens]
) {
  int64_t num_tokens = cos_sin_cache_offsets.size(0);


  TORCH_CHECK(positions.size(0) == num_tokens,
              "positions must have the same num_tokens as cos_sin_cache_offsets");
  TORCH_CHECK(query.dim() == 3, "query must be 3D [num_tokens, num_heads, head_size]");
  TORCH_CHECK(key.dim() == 3, "key must be 3D [num_tokens, num_kv_heads, head_size]");

  int64_t query_token_stride = query.stride(0);
  int64_t query_head_stride = query.stride(1);
  int64_t query_dim_stride = query.stride(2);

  int64_t key_token_stride = key.stride(0);
  int64_t key_head_stride = key.stride(1);
  int64_t key_dim_stride = key.stride(2);

  int num_heads = query.size(1);
  int num_kv_heads = key.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));

  const at::musa::OptionalMUSAGuard device_guard(device_of(query));
  const musaStream_t stream = at::musa::getCurrentMUSAStream();

  MUSA_DISPATCH_FLOATING_TYPES(query.scalar_type(), "batched_rotary_embedding_contiguous", [&] {
    if (is_neox) {
      batched_rotary_embedding_kernel_contiguous<scalar_t, true><<<grid, block, 0, stream>>>(
          positions.data_ptr<int64_t>(),
          query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(),
          cos_sin_cache.data_ptr<scalar_t>(),
          cos_sin_cache_offsets.data_ptr<int64_t>(),
          rot_dim,
          query_token_stride, query_head_stride, query_dim_stride,
          key_token_stride, key_head_stride, key_dim_stride,
          num_heads, num_kv_heads, head_size);
    } else {
      batched_rotary_embedding_kernel_contiguous<scalar_t, false><<<grid, block, 0, stream>>>(
          positions.data_ptr<int64_t>(),
          query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(),
          cos_sin_cache.data_ptr<scalar_t>(),
          cos_sin_cache_offsets.data_ptr<int64_t>(),
          rot_dim,
          query_token_stride, query_head_stride, query_dim_stride,
          key_token_stride, key_head_stride, key_dim_stride,
          num_heads, num_kv_heads, head_size);
    }
  });
}
