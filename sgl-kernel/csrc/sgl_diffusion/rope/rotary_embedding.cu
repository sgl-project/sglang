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

    // Directly access SMEM
    const float cos_val = static_cast<float>(cos_ptr[rot_offset]);
    const float sin_val = static_cast<float>(sin_ptr[rot_offset]);

    const float x = static_cast<float>(arr[x_index]);
    const float y = static_cast<float>(arr[y_index]);
    arr[x_index] = static_cast<scalar_t>(x * cos_val - y * sin_val);
    arr[y_index] = static_cast<scalar_t>(y * cos_val + x * sin_val);
  } else {
    // GPT-J / LLaMA style: layout [x0, x1, ..., y0, y1, ...]
    // cos/sin: [..., rotary_dim], one entry per (x, y) pair.
    const int x_index = rot_offset;
    const int y_index = rot_offset + embed_dim;

    // Directly access SMEM
    const float cos_val_x = static_cast<float>(cos_ptr[rot_offset]);
    const float sin_val_x = static_cast<float>(sin_ptr[rot_offset]);
    const float cos_val_y = static_cast<float>(cos_ptr[rot_offset + embed_dim]);
    const float sin_val_y = static_cast<float>(sin_ptr[rot_offset + embed_dim]);

    const float x = static_cast<float>(arr[x_index]);
    const float y = static_cast<float>(arr[y_index]);
    arr[x_index] = static_cast<scalar_t>(x * cos_val_x - y * sin_val_x);
    arr[y_index] = static_cast<scalar_t>(y * cos_val_y + x * sin_val_y);
  }
}

template <typename scalar_t, bool interleaved>
inline __device__ void apply_token_rotary_embedding_vec(
    scalar_t* __restrict__ arr,            // [head_size]
    const scalar_t* __restrict__ cos_ptr,  // [rot_dim]
    const scalar_t* __restrict__ sin_ptr,  // [rot_dim]
    int rot_offset,
    int embed_dim) {
  
  using vec_t = float4;
  constexpr int kVecBytes = sizeof(vec_t);
  constexpr int kScalarBytes = sizeof(scalar_t);
  constexpr int kElePerVec = kVecBytes / kScalarBytes;
  
  // Union for type punning to avoid strict aliasing issues with reinterpret_cast
  union VecUnion {
    vec_t vec;
    scalar_t elems[kElePerVec];
  };

  if constexpr (interleaved) {
    // Interleaved: arr has [x0, y0, x1, y1...]
    // A single vector load contains 'kElePerVec' elements, which is 'kElePerVec / 2' pairs.
    // rot_offset is the index of the PAIR.
    // Address in arr is rot_offset * 2.
    
    VecUnion data;
    data.vec = *reinterpret_cast<const vec_t*>(arr + rot_offset * 2);
    
    #pragma unroll
    for (int i = 0; i < kElePerVec; i += 2) {
      // data.elems[i] is x, data.elems[i+1] is y
      // They correspond to pair index: rot_offset + (i / 2)
      int curr_rot_offset = rot_offset + i / 2;
      
      float cos_val = static_cast<float>(cos_ptr[curr_rot_offset]);
      float sin_val = static_cast<float>(sin_ptr[curr_rot_offset]);
      
      float x = static_cast<float>(data.elems[i]);
      float y = static_cast<float>(data.elems[i+1]);
      
      data.elems[i] = static_cast<scalar_t>(x * cos_val - y * sin_val);
      data.elems[i+1] = static_cast<scalar_t>(y * cos_val + x * sin_val);
    }
    
    *reinterpret_cast<vec_t*>(arr + rot_offset * 2) = data.vec;

  } else {
    // Non-interleaved: X and Y are separated by embed_dim.
    // We process 'kElePerVec' PAIRS at once.
    // Load X vector and Y vector.
    
    VecUnion data_x, data_y;
    data_x.vec = *reinterpret_cast<const vec_t*>(arr + rot_offset);
    data_y.vec = *reinterpret_cast<const vec_t*>(arr + rot_offset + embed_dim);
    
    #pragma unroll
    for (int i = 0; i < kElePerVec; ++i) {
      int curr_rot_offset = rot_offset + i;
      
      // In non-interleaved, we might need different cos/sin for X and Y depending on implementation,
      // but standard RoPE uses the same angle for the pair.
      // Based on original scalar code:
      // cos_val_x = cos_ptr[rot_offset]
      // cos_val_y = cos_ptr[rot_offset + embed_dim]
      
      float cos_val_x = static_cast<float>(cos_ptr[curr_rot_offset]);
      float sin_val_x = static_cast<float>(sin_ptr[curr_rot_offset]);
      float cos_val_y = static_cast<float>(cos_ptr[curr_rot_offset + embed_dim]);
      float sin_val_y = static_cast<float>(sin_ptr[curr_rot_offset + embed_dim]);
      
      float x = static_cast<float>(data_x.elems[i]);
      float y = static_cast<float>(data_y.elems[i]);
      
      data_x.elems[i] = static_cast<scalar_t>(x * cos_val_x - y * sin_val_x);
      data_y.elems[i] = static_cast<scalar_t>(y * cos_val_y + x * sin_val_y);
    }
    
    *reinterpret_cast<vec_t*>(arr + rot_offset) = data_x.vec;
    *reinterpret_cast<vec_t*>(arr + rot_offset + embed_dim) = data_y.vec;
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

  const float cos_val = static_cast<float>(cos_ptr[rot_offset]);
  const float sin_val = static_cast<float>(sin_ptr[rot_offset]);

  float2 out;
  out.x = xy.x * cos_val - xy.y * sin_val;
  out.y = xy.y * cos_val + xy.x * sin_val;

  *reinterpret_cast<float2*>(arr + rot_offset * 2) = out;
}

// 2D grid kernel: parallel over tokens (grid.x) and pair-tiles (grid.y)
template <typename scalar_t, bool interleaved, bool vectorized>
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
  extern __shared__ char smem_[];
  scalar_t* s_cos = reinterpret_cast<scalar_t*>(smem_);
  scalar_t* s_sin = s_cos + rot_dim_arg;

  const int token_idx = blockIdx.x;
  if (token_idx >= gridDim.x) {
    return;
  }

  // Pointers to Global Memory for the current token
  const scalar_t* current_token_cos_ptr = cos_data + token_idx * rot_dim_arg;
  const scalar_t* current_token_sin_ptr = sin_data + token_idx * rot_dim_arg;

  constexpr int kVecBytes = 16;
  const int scalar_size = sizeof(scalar_t);
  const int vec_size = kVecBytes / scalar_size;

  // Load Cos/Sin to SMEM
  // Always use vectorized load if available, but fallback to scalar loop if dim is not multiple
  if constexpr (vectorized) {
    using vec_t = float4;
    const vec_t* cos_vec_ptr = reinterpret_cast<const vec_t*>(current_token_cos_ptr);
    const vec_t* sin_vec_ptr = reinterpret_cast<const vec_t*>(current_token_sin_ptr);
    vec_t* s_cos_vec = reinterpret_cast<vec_t*>(s_cos);
    vec_t* s_sin_vec = reinterpret_cast<vec_t*>(s_sin);

    for (int i = threadIdx.x; i < rot_dim_arg / vec_size; i += blockDim.x) {
      s_cos_vec[i] = SGLANG_LDG(cos_vec_ptr + i);
      s_sin_vec[i] = SGLANG_LDG(sin_vec_ptr + i);
    }
  } else {
    for (int i = threadIdx.x; i < rot_dim_arg; i += blockDim.x) {
      s_cos[i] = SGLANG_LDG(current_token_cos_ptr + i);
      s_sin[i] = SGLANG_LDG(current_token_sin_ptr + i);
    }
  }
  // Essential synchronization
  __syncthreads();

  scalar_t* query_for_token = query_total + token_idx * (int)query_token_stride;
  scalar_t* key_for_token = (key_total != nullptr) ? (key_total + token_idx * (int)key_token_stride) : nullptr;

  const int local_block_idx = blockIdx.y;
  
  if constexpr (vectorized) {
    using vec_t = float4;
    constexpr int kElePerVec = kVecBytes / sizeof(scalar_t);
    constexpr int pairs_per_step = interleaved ? (kElePerVec / 2) : kElePerVec;
    
    const int pair_stride = blockDim.x * blocks_per_token * pairs_per_step;
    const int thread_pair_offset = (local_block_idx * blockDim.x + threadIdx.x) * pairs_per_step;
    const int nq_pairs = num_heads * embed_dim_for_rotation;
    
    for (int i = thread_pair_offset; i < nq_pairs; i += pair_stride) {
      const int head_idx = i / embed_dim_for_rotation;
      const int rot_offset = i % embed_dim_for_rotation;
      
      scalar_t* query_for_token_head = query_for_token + head_idx * (int)head_stride_query;
      apply_token_rotary_embedding_vec<scalar_t, interleaved>(
          query_for_token_head, s_cos, s_sin, rot_offset, embed_dim_for_rotation);
    }
    
    if (key_for_token != nullptr) {
      const int nk_pairs = num_kv_heads * embed_dim_for_rotation;
      for (int i = thread_pair_offset; i < nk_pairs; i += pair_stride) {
        const int head_idx = i / embed_dim_for_rotation;
        const int rot_offset = i % embed_dim_for_rotation;
        
        scalar_t* key_for_token_head = key_for_token + head_idx * (int)head_stride_key;
        apply_token_rotary_embedding_vec<scalar_t, interleaved>(
            key_for_token_head, s_cos, s_sin, rot_offset, embed_dim_for_rotation);
      }
    }
  } else {
    // Fallback to scalar implementation
    const int pair_stride = blockDim.x * blocks_per_token;
    const int thread_pair_offset = local_block_idx * blockDim.x + threadIdx.x;
    
    const int nq_pairs = num_heads * embed_dim_for_rotation;
    for (int i = thread_pair_offset; i < nq_pairs; i += pair_stride) {
      const int head_idx = i / embed_dim_for_rotation;
      const int rot_offset = i % embed_dim_for_rotation;

      scalar_t* query_for_token_head = query_for_token + head_idx * (int)head_stride_query;
      apply_token_rotary_embedding<scalar_t, interleaved>(
          query_for_token_head, s_cos, s_sin, rot_offset, embed_dim_for_rotation);
    }

    if (key_for_token != nullptr) {
      const int nk_pairs = num_kv_heads * embed_dim_for_rotation;
      for (int i = thread_pair_offset; i < nk_pairs; i += pair_stride) {
        const int head_idx = i / embed_dim_for_rotation;
        const int rot_offset = i % embed_dim_for_rotation;

        scalar_t* key_for_token_head = key_for_token + head_idx * (int)head_stride_key;
        apply_token_rotary_embedding<scalar_t, interleaved>(
            key_for_token_head, s_cos, s_sin, rot_offset, embed_dim_for_rotation);
      }
    }
  }
}

// 1D grid kernel: each block handles one token
template <typename scalar_t, bool interleaved, bool vectorized>
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
  extern __shared__ char smem_[];
  scalar_t* s_cos = reinterpret_cast<scalar_t*>(smem_);
  scalar_t* s_sin = s_cos + rot_dim_arg;

  const int token_idx = blockIdx.x;
  if (token_idx >= gridDim.x) return;

  const scalar_t* current_token_cos_ptr = cos_data + token_idx * rot_dim_arg;
  const scalar_t* current_token_sin_ptr = sin_data + token_idx * rot_dim_arg;

  constexpr int kVecBytes = 16;
  const int scalar_size = sizeof(scalar_t);
  const int vec_size = kVecBytes / scalar_size;

  if constexpr (vectorized) {
    using vec_t = float4;
    const vec_t* cos_vec_ptr = reinterpret_cast<const vec_t*>(current_token_cos_ptr);
    const vec_t* sin_vec_ptr = reinterpret_cast<const vec_t*>(current_token_sin_ptr);
    vec_t* s_cos_vec = reinterpret_cast<vec_t*>(s_cos);
    vec_t* s_sin_vec = reinterpret_cast<vec_t*>(s_sin);

    for (int i = threadIdx.x; i < rot_dim_arg / vec_size; i += blockDim.x) {
      s_cos_vec[i] = SGLANG_LDG(cos_vec_ptr + i);
      s_sin_vec[i] = SGLANG_LDG(sin_vec_ptr + i);
    }
  } else {
    for (int i = threadIdx.x; i < rot_dim_arg; i += blockDim.x) {
      s_cos[i] = SGLANG_LDG(current_token_cos_ptr + i);
      s_sin[i] = SGLANG_LDG(current_token_sin_ptr + i);
    }
  }
  __syncthreads();

  scalar_t* query_for_token = query_total + token_idx * (int)query_token_stride;
  scalar_t* key_for_token = (key_total != nullptr) ? (key_total + token_idx * (int)key_token_stride) : nullptr;

  if constexpr (vectorized) {
    using vec_t = float4;
    constexpr int kElePerVec = kVecBytes / sizeof(scalar_t);
    constexpr int pairs_per_step = interleaved ? (kElePerVec / 2) : kElePerVec;
    
    const int nq_pairs = num_heads * embed_dim_for_rotation;
    for (int i = threadIdx.x * pairs_per_step; i < nq_pairs; i += blockDim.x * pairs_per_step) {
      const int head_idx = i / embed_dim_for_rotation;
      const int rot_offset = i % embed_dim_for_rotation;
      scalar_t* query_for_token_head = query_for_token + head_idx * (int)head_stride_query;
      
      apply_token_rotary_embedding_vec<scalar_t, interleaved>(
          query_for_token_head, s_cos, s_sin, rot_offset, embed_dim_for_rotation);
    }
    
    if (key_for_token != nullptr) {
      const int nk_pairs = num_kv_heads * embed_dim_for_rotation;
      for (int i = threadIdx.x * pairs_per_step; i < nk_pairs; i += blockDim.x * pairs_per_step) {
        const int head_idx = i / embed_dim_for_rotation;
        const int rot_offset = i % embed_dim_for_rotation;
        
        scalar_t* key_for_token_head = key_for_token + head_idx * (int)head_stride_key;
        apply_token_rotary_embedding_vec<scalar_t, interleaved>(
            key_for_token_head, s_cos, s_sin, rot_offset, embed_dim_for_rotation);
      }
    }
  } else {
    // Fallback scalar
    const int nq_pairs = num_heads * embed_dim_for_rotation;
    for (int i = threadIdx.x; i < nq_pairs; i += blockDim.x) {
      const int head_idx = i / embed_dim_for_rotation;
      const int rot_offset = i % embed_dim_for_rotation;
      scalar_t* query_for_token_head = query_for_token + head_idx * (int)head_stride_query;
       
      apply_token_rotary_embedding<scalar_t, interleaved>(
          query_for_token_head, s_cos, s_sin, rot_offset, embed_dim_for_rotation);
    }

    if (key_for_token != nullptr) {
      const int nk_pairs = num_kv_heads * embed_dim_for_rotation;
      for (int i = threadIdx.x; i < nk_pairs; i += blockDim.x) {
        const int head_idx = i / embed_dim_for_rotation;
        const int rot_offset = i % embed_dim_for_rotation;
        
        scalar_t* key_for_token_head = key_for_token + head_idx * (int)head_stride_key;
        apply_token_rotary_embedding<scalar_t, interleaved>(
            key_for_token_head, s_cos, s_sin, rot_offset, embed_dim_for_rotation);
      }
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

        // Constants for vectorization
        constexpr int kVecBytes = 16;
        constexpr int kElePerVec = kVecBytes / sizeof(torch_scalar_t);
        
        // Determine how many pairs one thread handles in one vector step
        // Interleaved: 1 vector load (16B) contains 'kElePerVec' elements -> 'kElePerVec / 2' pairs.
        // Non-interleaved: 2 vector loads (32B) contains 'kElePerVec' pairs (X vec + Y vec).
        const int pairs_per_step = interleaved ? (kElePerVec / 2) : kElePerVec;

        // Check if we can guarantee vectorization for ALL tokens
        // We need Base Pointers, Strides, and Dimension to be aligned.
        bool can_vectorize_all = true;
        
        // 1. Check Dimensions
        if (embed_dim_for_rotation % pairs_per_step != 0) can_vectorize_all = false;
        
        // 2. Check Base Pointers
        if (reinterpret_cast<uintptr_t>(query.data_ptr()) % kVecBytes != 0) can_vectorize_all = false;
        if (reinterpret_cast<uintptr_t>(cos.data_ptr()) % kVecBytes != 0) can_vectorize_all = false;
        if (reinterpret_cast<uintptr_t>(sin.data_ptr()) % kVecBytes != 0) can_vectorize_all = false;
        if (key.has_value()) {
          if (reinterpret_cast<uintptr_t>(key->data_ptr()) % kVecBytes != 0) can_vectorize_all = false;
        }

        // 3. Check Strides
        // We need the stride between tokens to be a multiple of vector size 
        // to ensure that if token 0 is aligned, token 1 is also aligned.
        if (query_token_stride * sizeof(torch_scalar_t) % kVecBytes != 0) can_vectorize_all = false;
        if (head_stride_query * sizeof(torch_scalar_t) % kVecBytes != 0) can_vectorize_all = false;
        
        if (key.has_value()) {
           if (key_token_stride * sizeof(torch_scalar_t) % kVecBytes != 0) can_vectorize_all = false;
           if (head_stride_key * sizeof(torch_scalar_t) % kVecBytes != 0) can_vectorize_all = false;
        }

        // Determine launch configuration
        // If we can vectorize all, each thread handles 'pairs_per_step' pairs.
        // Otherwise, fallback to conservative estimate (1 pair per thread) to ensure enough blocks.
        const int launch_pairs_per_thread = can_vectorize_all ? pairs_per_step : 1;

        const int total_threads_needed = (max_pairs_to_rotate_per_token + launch_pairs_per_thread - 1) / launch_pairs_per_thread;
        
        // Case 1: 2D Grid (Split one token across multiple blocks)
        // Keep block size moderate (128-256) aligned with head size
        const int threads_per_block_2d = std::min<int>(256, std::max(128, embed_dim_for_rotation));
        const int blocks_per_token_2d = (total_threads_needed + threads_per_block_2d - 1) / threads_per_block_2d;
        
        // Decide grid strategy
        const bool use_grid_2d = (num_tokens <= 4) && (blocks_per_token_2d > 1);

        // Case 2: 1D Grid (One block per token)
        // Maximize threads per block to cover all heads in one block if possible
        // Cap at 512 threads to balance occupancy and register usage
        const int threads_per_block_1d = std::min<int>(512, std::max(128, (total_threads_needed + 31) / 32 * 32));

        // Final launch config
        const int threads_per_block = use_grid_2d ? threads_per_block_2d : threads_per_block_1d;
        const int blocks_per_token = use_grid_2d ? blocks_per_token_2d : 1;

        dim3 block(threads_per_block);
        dim3 grid_2d((int)num_tokens, std::max(1, blocks_per_token));
        dim3 grid_1d((int)num_tokens);

        // We need 2 arrays (cos, sin) of size 'rot_dim_from_cache', each element is 'sizeof(torch_scalar_t)'
        size_t smem_size = rot_dim_from_cache * sizeof(torch_scalar_t) * 2;

        auto launch_kernel = [&](bool vectorized) {
          if (interleaved) {
            if (use_grid_2d) {
              if (vectorized) {
                rotary_embedding_kernel_2d<cuda_scalar_t, true, true><<<grid_2d, block, smem_size, stream>>>(
                  reinterpret_cast<const cuda_scalar_t*>(cos.data_ptr<torch_scalar_t>()),
                  reinterpret_cast<const cuda_scalar_t*>(sin.data_ptr<torch_scalar_t>()),
                  reinterpret_cast<cuda_scalar_t*>(query.data_ptr<torch_scalar_t>()),
                  key.has_value() ? reinterpret_cast<cuda_scalar_t*>(key->data_ptr<torch_scalar_t>()) : nullptr,
                  rot_dim_from_cache, embed_dim_for_rotation, query_token_stride, key_token_stride,
                  head_stride_query, head_stride_key, num_heads, num_kv_heads, (int)head_size, blocks_per_token);
              } else {
                rotary_embedding_kernel_2d<cuda_scalar_t, true, false><<<grid_2d, block, smem_size, stream>>>(
                  reinterpret_cast<const cuda_scalar_t*>(cos.data_ptr<torch_scalar_t>()),
                  reinterpret_cast<const cuda_scalar_t*>(sin.data_ptr<torch_scalar_t>()),
                  reinterpret_cast<cuda_scalar_t*>(query.data_ptr<torch_scalar_t>()),
                  key.has_value() ? reinterpret_cast<cuda_scalar_t*>(key->data_ptr<torch_scalar_t>()) : nullptr,
                  rot_dim_from_cache, embed_dim_for_rotation, query_token_stride, key_token_stride,
                  head_stride_query, head_stride_key, num_heads, num_kv_heads, (int)head_size, blocks_per_token);
              }
            } else {
              if (vectorized) {
                rotary_embedding_kernel_1d<cuda_scalar_t, true, true><<<grid_1d, block, smem_size, stream>>>(
                  reinterpret_cast<const cuda_scalar_t*>(cos.data_ptr<torch_scalar_t>()),
                  reinterpret_cast<const cuda_scalar_t*>(sin.data_ptr<torch_scalar_t>()),
                  reinterpret_cast<cuda_scalar_t*>(query.data_ptr<torch_scalar_t>()),
                  key.has_value() ? reinterpret_cast<cuda_scalar_t*>(key->data_ptr<torch_scalar_t>()) : nullptr,
                  rot_dim_from_cache, embed_dim_for_rotation, query_token_stride, key_token_stride,
                  head_stride_query, head_stride_key, num_heads, num_kv_heads, (int)head_size);
              } else {
                rotary_embedding_kernel_1d<cuda_scalar_t, true, false><<<grid_1d, block, smem_size, stream>>>(
                  reinterpret_cast<const cuda_scalar_t*>(cos.data_ptr<torch_scalar_t>()),
                  reinterpret_cast<const cuda_scalar_t*>(sin.data_ptr<torch_scalar_t>()),
                  reinterpret_cast<cuda_scalar_t*>(query.data_ptr<torch_scalar_t>()),
                  key.has_value() ? reinterpret_cast<cuda_scalar_t*>(key->data_ptr<torch_scalar_t>()) : nullptr,
                  rot_dim_from_cache, embed_dim_for_rotation, query_token_stride, key_token_stride,
                  head_stride_query, head_stride_key, num_heads, num_kv_heads, (int)head_size);
              }
            }
          } else { // non-interleaved
            if (use_grid_2d) {
              if (vectorized) {
                rotary_embedding_kernel_2d<cuda_scalar_t, false, true><<<grid_2d, block, smem_size, stream>>>(
                  reinterpret_cast<const cuda_scalar_t*>(cos.data_ptr<torch_scalar_t>()),
                  reinterpret_cast<const cuda_scalar_t*>(sin.data_ptr<torch_scalar_t>()),
                  reinterpret_cast<cuda_scalar_t*>(query.data_ptr<torch_scalar_t>()),
                  key.has_value() ? reinterpret_cast<cuda_scalar_t*>(key->data_ptr<torch_scalar_t>()) : nullptr,
                  rot_dim_from_cache, embed_dim_for_rotation, query_token_stride, key_token_stride,
                  head_stride_query, head_stride_key, num_heads, num_kv_heads, (int)head_size, blocks_per_token);
              } else {
                rotary_embedding_kernel_2d<cuda_scalar_t, false, false><<<grid_2d, block, smem_size, stream>>>(
                  reinterpret_cast<const cuda_scalar_t*>(cos.data_ptr<torch_scalar_t>()),
                  reinterpret_cast<const cuda_scalar_t*>(sin.data_ptr<torch_scalar_t>()),
                  reinterpret_cast<cuda_scalar_t*>(query.data_ptr<torch_scalar_t>()),
                  key.has_value() ? reinterpret_cast<cuda_scalar_t*>(key->data_ptr<torch_scalar_t>()) : nullptr,
                  rot_dim_from_cache, embed_dim_for_rotation, query_token_stride, key_token_stride,
                  head_stride_query, head_stride_key, num_heads, num_kv_heads, (int)head_size, blocks_per_token);
              }
            } else {
              if (vectorized) {
                rotary_embedding_kernel_1d<cuda_scalar_t, false, true><<<grid_1d, block, smem_size, stream>>>(
                  reinterpret_cast<const cuda_scalar_t*>(cos.data_ptr<torch_scalar_t>()),
                  reinterpret_cast<const cuda_scalar_t*>(sin.data_ptr<torch_scalar_t>()),
                  reinterpret_cast<cuda_scalar_t*>(query.data_ptr<torch_scalar_t>()),
                  key.has_value() ? reinterpret_cast<cuda_scalar_t*>(key->data_ptr<torch_scalar_t>()) : nullptr,
                  rot_dim_from_cache, embed_dim_for_rotation, query_token_stride, key_token_stride,
                  head_stride_query, head_stride_key, num_heads, num_kv_heads, (int)head_size);
              } else {
                rotary_embedding_kernel_1d<cuda_scalar_t, false, false><<<grid_1d, block, smem_size, stream>>>(
                  reinterpret_cast<const cuda_scalar_t*>(cos.data_ptr<torch_scalar_t>()),
                  reinterpret_cast<const cuda_scalar_t*>(sin.data_ptr<torch_scalar_t>()),
                  reinterpret_cast<cuda_scalar_t*>(query.data_ptr<torch_scalar_t>()),
                  key.has_value() ? reinterpret_cast<cuda_scalar_t*>(key->data_ptr<torch_scalar_t>()) : nullptr,
                  rot_dim_from_cache, embed_dim_for_rotation, query_token_stride, key_token_stride,
                  head_stride_query, head_stride_key, num_heads, num_kv_heads, (int)head_size);
              }
            }
          }
          
        };

        // Close the log file after all launches
        if (can_vectorize_all) {
          launch_kernel(true);
        } else {
          launch_kernel(false);
        }

      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
