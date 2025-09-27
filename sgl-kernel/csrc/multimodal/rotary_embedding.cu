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
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include <cmath>
#include <flashinfer/vec_dtypes.cuh>

#include "utils.h"

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr,
    int rot_offset,
    int embed_dim) {
  int x_index, y_index;

  if (IS_NEOX) {
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;

    scalar_t cos_val = SGLANG_LDG(cos_ptr + rot_offset);
    scalar_t sin_val = SGLANG_LDG(sin_ptr + rot_offset);

    const scalar_t x = arr[x_index];
    const scalar_t y = arr[y_index];
    arr[x_index] = x * cos_val - y * sin_val;
    arr[y_index] = y * cos_val + x * sin_val;

  } else {
    // GPT-J style / LLaMA style, matching the Python if cos/sin are [..., head_size]
    x_index = rot_offset;              // first half
    y_index = rot_offset + embed_dim;  // second half

    const scalar_t cos_val_x = SGLANG_LDG(cos_ptr + rot_offset);
    const scalar_t sin_val_x = SGLANG_LDG(sin_ptr + rot_offset);
    const scalar_t cos_val_y = SGLANG_LDG(cos_ptr + rot_offset + embed_dim);
    const scalar_t sin_val_y = SGLANG_LDG(sin_ptr + rot_offset + embed_dim);

    const scalar_t x = arr[x_index];
    const scalar_t y = arr[y_index];
    arr[x_index] = x * cos_val_x - y * sin_val_x;
    arr[y_index] = y * cos_val_y + x * sin_val_y;
  }
}

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
    scalar_t* __restrict__ query,                        // [num_heads, head_size]
    scalar_t* __restrict__ key,                          // [num_kv_heads, head_size]
    const scalar_t* __restrict__ current_token_cos_ptr,  // [rot_dim]
    const scalar_t* __restrict__ current_token_sin_ptr,  // [rot_dim]
    const int head_size,
    const int num_heads,
    const int num_kv_heads,
    const int rot_dim,
    const int64_t head_stride_query,
    const int64_t head_stride_key) {
  const int embed_dim_for_rotation = rot_dim / 2;

  const int nq_pairs = num_heads * embed_dim_for_rotation;
  for (int i = threadIdx.x; i < nq_pairs; i += blockDim.x) {
    const int head_idx = i / embed_dim_for_rotation;
    const int rot_offset = i % embed_dim_for_rotation;

    scalar_t* query_for_token_head = query + head_idx * (int)head_stride_query;

    apply_token_rotary_embedding<scalar_t, IS_NEOX>(
        query_for_token_head, current_token_cos_ptr, current_token_sin_ptr, rot_offset, embed_dim_for_rotation);
  }

  if (key != nullptr) {
    const int nk_pairs = num_kv_heads * embed_dim_for_rotation;
    for (int i = threadIdx.x; i < nk_pairs; i += blockDim.x) {
      const int head_idx = i / embed_dim_for_rotation;
      const int rot_offset = i % embed_dim_for_rotation;

      scalar_t* key_for_token_head = key + head_idx * (int)head_stride_key;

      apply_token_rotary_embedding<scalar_t, IS_NEOX>(
          key_for_token_head, current_token_cos_ptr, current_token_sin_ptr, rot_offset, embed_dim_for_rotation);
    }
  }
}

template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
    const scalar_t* __restrict__ cos_data,  // [num_tokens, rot_dim_arg]
    const scalar_t* __restrict__ sin_data,  // [num_tokens, rot_dim_arg]
    scalar_t* __restrict__ query_total,
    scalar_t* __restrict__ key_total,
    const int rot_dim_arg,
    const int64_t query_token_stride,
    const int64_t key_token_stride,
    const int64_t head_stride_query,
    const int64_t head_stride_key,
    const int num_heads,
    const int num_kv_heads,
    const int head_size) {
  const int token_idx = blockIdx.x;
  const scalar_t* current_token_cos_ptr = cos_data + token_idx * rot_dim_arg;
  const scalar_t* current_token_sin_ptr = sin_data + token_idx * rot_dim_arg;

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
      rot_dim_arg,
      head_stride_query,
      head_stride_key);
}

void rotary_embedding(
    at::Tensor& cos,
    at::Tensor& sin,
    at::Tensor& query,
    const std::optional<at::Tensor>& key,
    int64_t head_size,
    bool is_neox) {
  TORCH_CHECK(
      query.dim() == 2 || query.dim() == 3,
      "query must be in  shape [num_tokens, hidden_size] or [num_tokens, num_heads, head_size]");
  if (key.has_value()) {
    TORCH_CHECK(
        key->dim() == 2 || key->dim() == 3,
        "key must be in  shape [num_tokens, hidden_size] or [num_tokens, num_kv_heads, head_size]");
  }

  int64_t num_tokens = query.size(0);

  TORCH_CHECK(cos.dim() == 2, "cos must be in shape [num_tokens, D_cos]");
  TORCH_CHECK(sin.dim() == 2, "sin must be in  shape [num_tokens, D_sin]");
  TORCH_CHECK(cos.size(0) == num_tokens, "cos num_tokens mismatch with query");
  TORCH_CHECK(sin.size(0) == num_tokens, "sin num_tokens mismatch with query");
  TORCH_CHECK(cos.size(1) == sin.size(1), "cos and sin D_cos/D_sin mismatch");

  TORCH_CHECK(cos.scalar_type() == query.scalar_type(), "cos dtype mismatch");
  TORCH_CHECK(sin.scalar_type() == query.scalar_type(), "sin dtype mismatch");
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda() && query.is_cuda(), "All tensors must be on CUDA");
  if (key.has_value()) {
    TORCH_CHECK(key->is_cuda(), "Key tensor must be on CUDA if provided");
    TORCH_CHECK(key->scalar_type() == query.scalar_type(), "Key dtype mismatch");
  }

  int query_hidden_size_calculated;
  if (query.dim() == 2) {
    query_hidden_size_calculated = (int)query.size(1);
  } else {
    query_hidden_size_calculated = (int)query.size(1) * (int)query.size(2);
    TORCH_CHECK(query.size(2) == head_size, "Query head_size mismatch in 3D tensor");
  }
  TORCH_CHECK(query_hidden_size_calculated % head_size == 0, "query_hidden_size not divisible by head_size");
  int num_heads = (int)query_hidden_size_calculated / (int)head_size;

  int key_hidden_size_calculated = 0;
  int num_kv_heads = num_heads;
  if (key.has_value()) {
    TORCH_CHECK((int)key->size(0) == num_tokens, "Key num_tokens mismatch");
    if (key->dim() == 2) {
      key_hidden_size_calculated = (int)key->size(1);
    } else {
      key_hidden_size_calculated = (int)key->size(1) * (int)key->size(2);
      TORCH_CHECK((int)key->size(2) == head_size, "Key head_size mismatch in 3D tensor");
    }
    TORCH_CHECK(key_hidden_size_calculated % head_size == 0, "key_hidden_size not divisible by head_size");
    num_kv_heads = key_hidden_size_calculated / (int)head_size;
  }
  TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");

  int rot_dim_from_cache = (int)cos.size(1);

  int64_t query_token_stride = query_hidden_size_calculated;
  int64_t key_token_stride = key.has_value() ? key_hidden_size_calculated : 0;

  int64_t head_stride_query;
  if (query.dim() == 3 && query.size(1) == num_heads && query.size(2) == head_size) {
    head_stride_query = query.stride(1);
  } else {
    head_stride_query = head_size;
  }

  int64_t head_stride_key = head_size;
  if (key.has_value()) {
    if (key->dim() == 3 && key->size(1) == num_kv_heads && key->size(2) == head_size) {
      head_stride_key = key->stride(1);
    } else {
      head_stride_key = head_size;
    }
  }

  dim3 grid((int)num_tokens);

  int embed_dim_for_block_calc = rot_dim_from_cache / 2;
  int max_pairs_to_rotate_per_token =
      std::max(num_heads * embed_dim_for_block_calc, num_kv_heads * embed_dim_for_block_calc);
  dim3 block(std::min<int64_t>(max_pairs_to_rotate_per_token, 512L));

  if (block.x == 0 && num_tokens > 0) block.x = 1;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  SGLANG_DISPATCH_FLOATING_TYPES(query.scalar_type(), "rotary_embedding", [&] {
    if (is_neox) {
      rotary_embedding_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
          cos.data_ptr<scalar_t>(),
          sin.data_ptr<scalar_t>(),
          query.data_ptr<scalar_t>(),
          key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
          rot_dim_from_cache,
          query_token_stride,
          key_token_stride,
          head_stride_query,
          head_stride_key,
          num_heads,
          num_kv_heads,
          (int)head_size);
    } else {
      rotary_embedding_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
          cos.data_ptr<scalar_t>(),
          sin.data_ptr<scalar_t>(),
          query.data_ptr<scalar_t>(),
          key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
          rot_dim_from_cache,
          query_token_stride,
          key_token_stride,
          head_stride_query,
          head_stride_key,
          num_heads,
          num_kv_heads,
          (int)head_size);
    }
  });
}
