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
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include <cmath>
#include <vector>

#include "utils.h"

template <typename scalar_t, bool interleaved>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr,            // [head_size]
    const scalar_t* __restrict__ cos_ptr,  // [rot_dim]
    const scalar_t* __restrict__ sin_ptr,  // [rot_dim]
    int rot_offset,
    int embed_dim) {  // for non-interleaved: half dim
  if constexpr (interleaved) {
    // NeoX-style: interleaved layout [x0, y0, x1, y1, ...].
    // cos/sin: [..., rotary_dim/2], one entry per (x, y) pair.
    const int x_index = 2 * rot_offset;
    const int y_index = x_index + 1;

    const float cos_val = static_cast<float>(SGLANG_LDG(cos_ptr + rot_offset));
    const float sin_val = static_cast<float>(SGLANG_LDG(sin_ptr + rot_offset));

    const float x = static_cast<float>(arr[x_index]);
    const float y = static_cast<float>(arr[y_index]);
    arr[x_index] = static_cast<scalar_t>(x * cos_val - y * sin_val);
    arr[y_index] = static_cast<scalar_t>(y * cos_val + x * sin_val);
  } else {
    // GPT-J / LLaMA style: layout [x0, x1, ..., y0, y1, ...]
    // cos/sin: [..., rotary_dim], one entry per (x, y) pair.
    const int x_index = rot_offset;
    const int y_index = rot_offset + embed_dim;

    const float cos_val_x = static_cast<float>(SGLANG_LDG(cos_ptr + rot_offset));
    const float sin_val_x = static_cast<float>(SGLANG_LDG(sin_ptr + rot_offset));
    const float cos_val_y = static_cast<float>(SGLANG_LDG(cos_ptr + rot_offset + embed_dim));
    const float sin_val_y = static_cast<float>(SGLANG_LDG(sin_ptr + rot_offset + embed_dim));

    const float x = static_cast<float>(arr[x_index]);
    const float y = static_cast<float>(arr[y_index]);
    arr[x_index] = static_cast<scalar_t>(x * cos_val_x - y * sin_val_x);
    arr[y_index] = static_cast<scalar_t>(y * cos_val_y + x * sin_val_y);
  }
}

template <>
inline __device__ void apply_token_rotary_embedding<float, true>(
    float* __restrict__ arr,
    const float* __restrict__ cos_ptr,
    const float* __restrict__ sin_ptr,
    int rot_offset,
    int /*embed_dim*/) {
  float2 xy = *reinterpret_cast<const float2*>(arr + rot_offset * 2);
  const float cos_val = static_cast<float>(SGLANG_LDG(cos_ptr + rot_offset));
  const float sin_val = static_cast<float>(SGLANG_LDG(sin_ptr + rot_offset));

  float2 out;
  out.x = xy.x * cos_val - xy.y * sin_val;
  out.y = xy.y * cos_val + xy.x * sin_val;

  *reinterpret_cast<float2*>(arr + rot_offset * 2) = out;
}

// 2D grid kernel: parallel over tokens (grid.x) and pair-tiles (grid.y)
template <typename scalar_t, bool interleaved>
__global__ void rotary_embedding_kernel_2d(
    const scalar_t* __restrict__ cos_data,  // [num_tokens, rot_dim_arg]
    const scalar_t* __restrict__ sin_data,  // [num_tokens, rot_dim_arg]
    scalar_t* __restrict__ query_total,
    scalar_t* __restrict__ key_total,
    const int rot_dim_arg,
    const int embed_dim_for_rotation,
    const int64_t query_token_stride,
    const int64_t key_token_stride,
    const int64_t head_stride_query,
    const int64_t head_stride_key,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int blocks_per_token) {
  const int token_idx = blockIdx.x;
  if (token_idx >= gridDim.x) {
    return;
  }

  const scalar_t* current_token_cos_ptr = cos_data + token_idx * rot_dim_arg;
  const scalar_t* current_token_sin_ptr = sin_data + token_idx * rot_dim_arg;

  scalar_t* query_for_token = query_total + token_idx * (int)query_token_stride;
  scalar_t* key_for_token = (key_total != nullptr) ? (key_total + token_idx * (int)key_token_stride) : nullptr;

  const int local_block_idx = blockIdx.y; 
  const int pair_stride = blockDim.x * blocks_per_token;
  const int thread_pair_offset = local_block_idx * blockDim.x + threadIdx.x;

  const int nq_pairs = num_heads * embed_dim_for_rotation;
  for (int i = thread_pair_offset; i < nq_pairs; i += pair_stride) {
    const int head_idx = i / embed_dim_for_rotation;
    const int rot_offset = i % embed_dim_for_rotation;

    scalar_t* query_for_token_head = query_for_token + head_idx * (int)head_stride_query;
    apply_token_rotary_embedding<scalar_t, interleaved>(
        query_for_token_head, current_token_cos_ptr, current_token_sin_ptr, rot_offset, embed_dim_for_rotation);
  }

  if (key_for_token != nullptr) {
    const int nk_pairs = num_kv_heads * embed_dim_for_rotation;
    for (int i = thread_pair_offset; i < nk_pairs; i += pair_stride) {
      const int head_idx = i / embed_dim_for_rotation;
      const int rot_offset = i % embed_dim_for_rotation;

      scalar_t* key_for_token_head = key_for_token + head_idx * (int)head_stride_key;
      apply_token_rotary_embedding<scalar_t, interleaved>(
          key_for_token_head, current_token_cos_ptr, current_token_sin_ptr, rot_offset, embed_dim_for_rotation);
    }
  }
}

// 1D grid kernel: each block handles one token
template <typename scalar_t, bool interleaved>
__global__ void rotary_embedding_kernel_1d(
    const scalar_t* __restrict__ cos_data,  // [num_tokens, rot_dim_arg]
    const scalar_t* __restrict__ sin_data,  // [num_tokens, rot_dim_arg]
    scalar_t* __restrict__ query_total,
    scalar_t* __restrict__ key_total,
    const int rot_dim_arg,
    const int embed_dim_for_rotation,
    const int64_t query_token_stride,
    const int64_t key_token_stride,
    const int64_t head_stride_query,
    const int64_t head_stride_key,
    const int num_heads,
    const int num_kv_heads,
    const int head_size) {
  const int token_idx = blockIdx.x;
  if (token_idx >= gridDim.x) return;

  const scalar_t* current_token_cos_ptr = cos_data + token_idx * rot_dim_arg;
  const scalar_t* current_token_sin_ptr = sin_data + token_idx * rot_dim_arg;

  scalar_t* query_for_token = query_total + token_idx * (int)query_token_stride;
  scalar_t* key_for_token = (key_total != nullptr) ? (key_total + token_idx * (int)key_token_stride) : nullptr;

  const int nq_pairs = num_heads * embed_dim_for_rotation;
  for (int i = threadIdx.x; i < nq_pairs; i += blockDim.x) {
    const int head_idx = i / embed_dim_for_rotation;
    const int rot_offset = i % embed_dim_for_rotation;
    scalar_t* query_for_token_head = query_for_token + head_idx * (int)head_stride_query;
    apply_token_rotary_embedding<scalar_t, interleaved>(
        query_for_token_head, current_token_cos_ptr, current_token_sin_ptr, rot_offset, embed_dim_for_rotation);
  }

  if (key_for_token != nullptr) {
    const int nk_pairs = num_kv_heads * embed_dim_for_rotation;
    for (int i = threadIdx.x; i < nk_pairs; i += blockDim.x) {
      const int head_idx = i / embed_dim_for_rotation;
      const int rot_offset = i % embed_dim_for_rotation;
      scalar_t* key_for_token_head = key_for_token + head_idx * (int)head_stride_key;
      apply_token_rotary_embedding<scalar_t, interleaved>(
          key_for_token_head, current_token_cos_ptr, current_token_sin_ptr, rot_offset, embed_dim_for_rotation);
    }
  }
}

void rotary_embedding_cos_sin(
    at::Tensor& cos,
    at::Tensor& sin,
    at::Tensor& query,
    const std::optional<at::Tensor>& key,
    int64_t head_size,
    bool interleaved) {
  TORCH_CHECK(
      query.dim() == 2 || query.dim() == 3,
      "query must be in shape [num_tokens, hidden_size] or [num_tokens, num_heads, head_size]");
  if (key.has_value()) {
    TORCH_CHECK(
        key->dim() == 2 || key->dim() == 3,
        "key must be in shape [num_tokens, hidden_size] or [num_tokens, num_kv_heads, head_size]");
  }

  const int64_t num_tokens = query.size(0);

  TORCH_CHECK(cos.dim() == 2, "cos must be in shape [num_tokens, D_cos]");
  TORCH_CHECK(sin.dim() == 2, "sin must be in shape [num_tokens, D_sin]");
  TORCH_CHECK(cos.size(0) == num_tokens, "cos num_tokens mismatch with query");
  TORCH_CHECK(sin.size(0) == num_tokens, "sin num_tokens mismatch with query");
  TORCH_CHECK(cos.size(1) == sin.size(1), "cos and sin D_cos/D_sin mismatch");

  TORCH_CHECK(cos.scalar_type() == query.scalar_type(), "cos dtype mismatch with query");
  TORCH_CHECK(sin.scalar_type() == query.scalar_type(), "sin dtype mismatch with query");
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda() && query.is_cuda(), "cos/sin/query must be CUDA tensors");
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
  if (key.has_value()) {
    TORCH_CHECK(key->is_cuda(), "key must be CUDA tensor if provided");
    TORCH_CHECK(key->scalar_type() == query.scalar_type(), "key dtype mismatch with query");
    TORCH_CHECK(key->is_contiguous(), "key must be contiguous");
  }

  int query_hidden_size_calculated;
  if (query.dim() == 2) {
    query_hidden_size_calculated = (int)query.size(1);
  } else {
    query_hidden_size_calculated = (int)query.size(1) * (int)query.size(2);
    TORCH_CHECK(query.size(2) == head_size, "query head_size mismatch in 3D tensor");
  }
  TORCH_CHECK(query_hidden_size_calculated % head_size == 0, "query_hidden_size not divisible by head_size");
  int num_heads = query_hidden_size_calculated / (int)head_size;

  int key_hidden_size_calculated = 0;
  int num_kv_heads = num_heads;
  if (key.has_value()) {
    TORCH_CHECK((int)key->size(0) == num_tokens, "key num_tokens mismatch");
    if (key->dim() == 2) {
      key_hidden_size_calculated = (int)key->size(1);
    } else {
      key_hidden_size_calculated = (int)key->size(1) * (int)key->size(2);
      TORCH_CHECK((int)key->size(2) == head_size, "key head_size mismatch in 3D tensor");
    }
    TORCH_CHECK(key_hidden_size_calculated % head_size == 0, "key_hidden_size not divisible by head_size");
    num_kv_heads = key_hidden_size_calculated / (int)head_size;
  }
  TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");

  // NeoX interleaved: if cos/sin are full-dim (head_size), downsample once.
  if (interleaved && cos.size(1) == head_size) {
    TORCH_CHECK(head_size % 2 == 0, "interleaved layout requires even head_size");
    const int64_t half = head_size / 2;
    std::vector<int64_t> new_shape = {cos.size(0), half, 2};
    cos = cos.view(new_shape).select(2, 0).contiguous();
    sin = sin.view(new_shape).select(2, 1).contiguous();
  }

  const int rot_dim_from_cache = (int)cos.size(1);

  const int64_t query_token_stride = query_hidden_size_calculated;
  const int64_t key_token_stride = key.has_value() ? key_hidden_size_calculated : 0;

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

  const int embed_dim_for_rotation = interleaved ? rot_dim_from_cache : (rot_dim_from_cache / 2);
  TORCH_CHECK(embed_dim_for_rotation > 0, "embed_dim_for_rotation must be > 0");

  const int max_pairs_to_rotate_per_token =
      std::max(num_heads * embed_dim_for_rotation, num_kv_heads * embed_dim_for_rotation);

  const int threads_per_block = std::min<int>(256, std::max(128, embed_dim_for_rotation));
  const int blocks_per_token = (max_pairs_to_rotate_per_token + threads_per_block - 1) / threads_per_block;

  const bool use_grid_2d = (num_tokens <= 4) && (blocks_per_token > 1);
  dim3 block(threads_per_block);
  dim3 grid_2d((int)num_tokens, std::max(1, blocks_per_token));
  dim3 grid_1d((int)num_tokens);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(), "rotary_embedding_cos_sin", [&] {
        using torch_scalar_t = scalar_t;
        using cuda_scalar_t = typename std::conditional<
            std::is_same<torch_scalar_t, at::Half>::value,
            nv_half,
            typename std::conditional<std::is_same<torch_scalar_t, at::BFloat16>::value, nv_bfloat16, torch_scalar_t>::
                type>::type;

        if (interleaved) {
          if (use_grid_2d) {
            rotary_embedding_kernel_2d<cuda_scalar_t, true><<<grid_2d, block, 0, stream>>>(
                reinterpret_cast<const cuda_scalar_t*>(cos.data_ptr<torch_scalar_t>()),
                reinterpret_cast<const cuda_scalar_t*>(sin.data_ptr<torch_scalar_t>()),
                reinterpret_cast<cuda_scalar_t*>(query.data_ptr<torch_scalar_t>()),
                key.has_value() ? reinterpret_cast<cuda_scalar_t*>(key->data_ptr<torch_scalar_t>()) : nullptr,
                rot_dim_from_cache,
                embed_dim_for_rotation,
                query_token_stride,
                key_token_stride,
                head_stride_query,
                head_stride_key,
                num_heads,
                num_kv_heads,
                (int)head_size,
                blocks_per_token);
          } else {
            rotary_embedding_kernel_1d<cuda_scalar_t, true><<<grid_1d, block, 0, stream>>>(
                reinterpret_cast<const cuda_scalar_t*>(cos.data_ptr<torch_scalar_t>()),
                reinterpret_cast<const cuda_scalar_t*>(sin.data_ptr<torch_scalar_t>()),
                reinterpret_cast<cuda_scalar_t*>(query.data_ptr<torch_scalar_t>()),
                key.has_value() ? reinterpret_cast<cuda_scalar_t*>(key->data_ptr<torch_scalar_t>()) : nullptr,
                rot_dim_from_cache,
                embed_dim_for_rotation,
                query_token_stride,
                key_token_stride,
                head_stride_query,
                head_stride_key,
                num_heads,
                num_kv_heads,
                (int)head_size);
          }
        } else {
          if (use_grid_2d) {
            rotary_embedding_kernel_2d<cuda_scalar_t, false><<<grid_2d, block, 0, stream>>>(
                reinterpret_cast<const cuda_scalar_t*>(cos.data_ptr<torch_scalar_t>()),
                reinterpret_cast<const cuda_scalar_t*>(sin.data_ptr<torch_scalar_t>()),
                reinterpret_cast<cuda_scalar_t*>(query.data_ptr<torch_scalar_t>()),
                key.has_value() ? reinterpret_cast<cuda_scalar_t*>(key->data_ptr<torch_scalar_t>()) : nullptr,
                rot_dim_from_cache,
                embed_dim_for_rotation,
                query_token_stride,
                key_token_stride,
                head_stride_query,
                head_stride_key,
                num_heads,
                num_kv_heads,
                (int)head_size,
                blocks_per_token);
          } else {
            rotary_embedding_kernel_1d<cuda_scalar_t, false><<<grid_1d, block, 0, stream>>>(
                reinterpret_cast<const cuda_scalar_t*>(cos.data_ptr<torch_scalar_t>()),
                reinterpret_cast<const cuda_scalar_t*>(sin.data_ptr<torch_scalar_t>()),
                reinterpret_cast<cuda_scalar_t*>(query.data_ptr<torch_scalar_t>()),
                key.has_value() ? reinterpret_cast<cuda_scalar_t*>(key->data_ptr<torch_scalar_t>()) : nullptr,
                rot_dim_from_cache,
                embed_dim_for_rotation,
                query_token_stride,
                key_token_stride,
                head_stride_query,
                head_stride_key,
                num_heads,
                num_kv_heads,
                (int)head_size);
          }
        }
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
