/*
 * Copyright (c) 2024 by FlashInfer team.
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

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include "pos_enc.cuh"
#include "utils.h"

using namespace flashinfer;

void apply_rope_pos_ids_cos_sin_cache(
    at::Tensor q,
    at::Tensor k,
    at::Tensor q_rope,
    at::Tensor k_rope,
    at::Tensor cos_sin_cache,
    at::Tensor pos_ids,
    bool interleave,
    bool enable_pdl,
    const std::optional<at::Tensor>& v,
    const std::optional<at::Tensor>& k_buffer,
    const std::optional<at::Tensor>& v_buffer,
    const std::optional<at::Tensor>& kv_cache_loc) {
  CHECK_LAST_DIM_CONTIGUOUS(q);
  CHECK_LAST_DIM_CONTIGUOUS(k);

  const bool save_kv_cache = v.has_value();
  if (save_kv_cache) {
    TORCH_CHECK(v.has_value());
    TORCH_CHECK(k_buffer.has_value());
    TORCH_CHECK(v_buffer.has_value());
    TORCH_CHECK(kv_cache_loc.has_value());
    CHECK_LAST_DIM_CONTIGUOUS(v.value());
    CHECK_LAST_DIM_CONTIGUOUS(k_buffer.value());
    CHECK_LAST_DIM_CONTIGUOUS(v_buffer.value());
    CHECK_DIM(3, k_buffer.value());      // k_buffer: (nnz, H_K, D)
    CHECK_DIM(3, v_buffer.value());      // v_buffer: (nnz, H_V, D)
    CHECK_DIM(3, v.value());             // v: (nnz, H_V, D)
    CHECK_DIM(1, kv_cache_loc.value());  // v: (n)
    CHECK_INPUT(kv_cache_loc.value());
  }
  size_t k_buffer_stride_n = save_kv_cache ? k_buffer->stride(0) : 0;
  size_t k_buffer_stride_h = save_kv_cache ? k_buffer->stride(1) : 0;
  size_t v_buffer_stride_n = save_kv_cache ? v_buffer->stride(0) : 0;
  size_t v_buffer_stride_h = save_kv_cache ? v_buffer->stride(1) : 0;
  size_t v_stride_n = save_kv_cache ? v->stride(0) : 0;
  size_t v_stride_h = save_kv_cache ? v->stride(1) : 0;
  auto kv_cache_loc_ptr = save_kv_cache ? static_cast<int64_t*>(kv_cache_loc->data_ptr()) : nullptr;

  CHECK_INPUT(cos_sin_cache);
  CHECK_INPUT(pos_ids);
  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_EQ(cos_sin_cache.device(), device);
  CHECK_EQ(pos_ids.device(), device);
  CHECK_DIM(3, q);  // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);  // k: (nnz, H_K, D)

  // cos_sin_cache: (max_seq_len, R)
  // First half of R is cos, second half is sin
  CHECK_DIM(2, cos_sin_cache);
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  unsigned int rotary_dim = cos_sin_cache.size(1);
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int nnz = q.size(0);
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);

  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(q.scalar_type(), c_type, [&] {
    // TODO temporarily only use `BatchQKApplyRotaryPosIdsCosSinCacheEnhanced` when save_kv_cache
    // to avoid changing original code path; but this branch is feature-complete and should switch to this later
    if (save_kv_cache) {
      cudaError_t status = BatchQKApplyRotaryPosIdsCosSinCacheEnhanced(
          static_cast<c_type*>(q.data_ptr()),
          static_cast<c_type*>(k.data_ptr()),
          save_kv_cache ? static_cast<c_type*>(v->data_ptr()) : nullptr,
          static_cast<c_type*>(q_rope.data_ptr()),
          static_cast<c_type*>(k_rope.data_ptr()),
          save_kv_cache ? static_cast<c_type*>(k_buffer->data_ptr()) : nullptr,
          save_kv_cache ? static_cast<c_type*>(v_buffer->data_ptr()) : nullptr,
          static_cast<float*>(cos_sin_cache.data_ptr()),
          static_cast<int64_t*>(pos_ids.data_ptr()),
          nnz,
          num_qo_heads,
          num_kv_heads,
          rotary_dim,
          head_dim,
          q_stride_n,
          q_stride_h,
          k_stride_n,
          k_stride_h,
          v_stride_n,
          v_stride_h,
          q_rope_stride_n,
          q_rope_stride_h,
          k_rope_stride_n,
          k_rope_stride_h,
          k_buffer_stride_n,
          k_buffer_stride_h,
          v_buffer_stride_n,
          v_buffer_stride_h,
          kv_cache_loc_ptr,
          interleave,
          save_kv_cache,
          enable_pdl,
          stream);
      TORCH_CHECK(
          status == cudaSuccess,
          "BatchQKApplyRotaryPosIdsCosSinCacheEnhanced failed with error code " +
              std::string(cudaGetErrorString(status)));
    } else {
      TORCH_CHECK(!enable_pdl);
      cudaError_t status = BatchQKApplyRotaryPosIdsCosSinCache(
          static_cast<c_type*>(q.data_ptr()),
          static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(q_rope.data_ptr()),
          static_cast<c_type*>(k_rope.data_ptr()),
          static_cast<float*>(cos_sin_cache.data_ptr()),
          static_cast<int64_t*>(pos_ids.data_ptr()),
          nnz,
          num_qo_heads,
          num_kv_heads,
          rotary_dim,
          head_dim,
          q_stride_n,
          q_stride_h,
          k_stride_n,
          k_stride_h,
          q_rope_stride_n,
          q_rope_stride_h,
          k_rope_stride_n,
          k_rope_stride_h,
          interleave,
          stream);
      TORCH_CHECK(
          status == cudaSuccess,
          "BatchQKApplyRotaryPosIdsCosSinCache failed with error code " + std::string(cudaGetErrorString(status)));
    }
    return true;
  });
}

// Adapted from
// https://github.com/vllm-project/vllm/blob/014ece97c7aa49084a1119dca792af081a18dbc1/csrc/pos_encoding_kernels.cu
template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    scalar_t* __restrict__ arr,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr,
    int rot_offset,
    int embed_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = SGLANG_LDG(cos_ptr + x_index);
    sin = SGLANG_LDG(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = SGLANG_LDG(cos_ptr + x_index / 2);
    sin = SGLANG_LDG(sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
    scalar_t* __restrict__ query,  // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,    // nullptr or
                                   // [batch_size, seq_len, num_kv_heads,
                                   // head_size] or [num_tokens, num_kv_heads,
                                   // head_size]
    const scalar_t* cache_ptr,
    const int head_size,
    const int num_heads,
    const int num_kv_heads,
    const int rot_dim,
    const int token_idx,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t head_stride) {
  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * query_stride + head_idx * head_stride;
    const int rot_offset = i % embed_dim;
    apply_token_rotary_embedding<scalar_t, IS_NEOX>(query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }

  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
      const int head_idx = i / embed_dim;
      const int64_t token_head = token_idx * key_stride + head_idx * head_stride;
      const int rot_offset = i % embed_dim;
      apply_token_rotary_embedding<scalar_t, IS_NEOX>(key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
    }
  }
}

template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,       // [batch_size, seq_len] or
                                                 // [num_tokens]
    scalar_t* __restrict__ query,                // [batch_size, seq_len, num_heads,
                                                 // head_size] or [num_tokens, num_heads,
                                                 // head_size]
    scalar_t* __restrict__ key,                  // nullptr or
                                                 // [batch_size, seq_len, num_kv_heads,
                                                 // head_size] or [num_tokens, num_kv_heads,
                                                 // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                 // 2]
    const int rot_dim,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t head_stride,
    const int num_heads,
    const int num_kv_heads,
    const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  apply_rotary_embedding<scalar_t, IS_NEOX>(
      query,
      key,
      cache_ptr,
      head_size,
      num_heads,
      num_kv_heads,
      rot_dim,
      token_idx,
      query_stride,
      key_stride,
      head_stride);
}

void rotary_embedding(
    torch::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
    torch::Tensor& query,      // [batch_size, seq_len, num_heads * head_size] or
                               // [num_tokens, num_heads * head_size] or
                               // [batch_size, seq_len, num_heads, head_size] or
                               // [num_tokens, num_heads, head_size]
    std::optional<torch::Tensor> key,
    // null or
    // [batch_size, seq_len, num_kv_heads * head_size] or
    // [num_tokens, num_kv_heads * head_size] or
    // [batch_size, seq_len, num_heads, head_size] or
    // [num_tokens, num_heads, head_size]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox) {
  // num_tokens = batch_size * seq_len
  int64_t num_tokens = positions.numel();
  int positions_ndim = positions.dim();

  // Make sure num_tokens dim is consistent across positions, query, and key
  TORCH_CHECK(
      positions_ndim == 1 || positions_ndim == 2, "positions must have shape [num_tokens] or [batch_size, seq_len]");
  if (positions_ndim == 1) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) && (!key.has_value() || key->size(0) == positions.size(0)),
        "query, key and positions must have the same number of tokens");
  }
  if (positions_ndim == 2) {
    TORCH_CHECK(
        query.size(0) == positions.size(0) && (!key.has_value() || key->size(0) == positions.size(0)) &&
            query.size(1) == positions.size(1) && (!key.has_value() || key->size(1) == positions.size(1)),
        "query, key and positions must have the same batch_size and seq_len");
  }

  // Make sure head_size is valid for query and key
  // hidden_size = num_heads * head_size
  int query_hidden_size = query.numel() / num_tokens;
  int key_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;
  TORCH_CHECK(query_hidden_size % head_size == 0);
  TORCH_CHECK(key_hidden_size % head_size == 0);

  // Make sure query and key have consistent number of heads
  int num_heads = query_hidden_size / head_size;
  int num_kv_heads = key.has_value() ? key_hidden_size / head_size : num_heads;
  TORCH_CHECK(num_heads % num_kv_heads == 0);

  int rot_dim = cos_sin_cache.size(1);
  int seq_dim_idx = positions_ndim - 1;
  int64_t query_stride = query.stride(seq_dim_idx);
  int64_t key_stride = key.has_value() ? key->stride(seq_dim_idx) : 0;
  // Determine head stride: for [*, heads, head_size] use stride of last dim;
  // for flat [*, heads*head_size], heads blocks are contiguous of size
  // head_size
  int query_ndim = query.dim();
  int64_t head_stride = (query_ndim == positions_ndim + 2) ? query.stride(-2) : head_size;

  dim3 grid(num_tokens);
  dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_FLOAT_TYPES(query.scalar_type(), "rotary_embedding", [&] {
    if (is_neox) {
      rotary_embedding_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
          positions.data_ptr<int64_t>(),
          query.data_ptr<scalar_t>(),
          key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
          cos_sin_cache.data_ptr<scalar_t>(),
          rot_dim,
          query_stride,
          key_stride,
          head_stride,
          num_heads,
          num_kv_heads,
          head_size);
    } else {
      rotary_embedding_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
          positions.data_ptr<int64_t>(),
          query.data_ptr<scalar_t>(),
          key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
          cos_sin_cache.data_ptr<scalar_t>(),
          rot_dim,
          query_stride,
          key_stride,
          head_stride,
          num_heads,
          num_kv_heads,
          head_size);
    }
  });
}
