/*
 * Copyright (c) 2023 by FlashInfer team.
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
#ifndef FLASHINFER_POS_ENC_CUH_
#define FLASHINFER_POS_ENC_CUH_

#include <cmath>
#include <cstdint>
#include <flashinfer/layout.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/utils.cuh>
#include <flashinfer/vec_dtypes.cuh>
#include <iostream>
#include <string>

namespace flashinfer {

/*!
 * \brief An enumeration class that defines different modes for applying RoPE
 *   (Rotary Positional Embeddings).
 */
enum class PosEncodingMode {
  // No rotary positional embeddings
  kNone = 0U,
  // Apply Llama-style rope.
  kRoPELlama = 1U,
  // Apply ALiBi bias
  kALiBi = 2U
};

/*!
 * \brief Apply RoPE (Rotary Positional Embeddings) to x[0: head_dim],
 *   return thread-local vector
 * \tparam vec_size A template integer indicates the vector size used
 *   in the kernel
 * \tparam bdx A template integer indicates the blockDim.x
 * \tparam T A template type indicates the x data type
 * \param x A pointer to the start of x data
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param offset A integer indicates the offset of the position in RoPE
 */
template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope(
    const T* x, const vec_t<float, vec_size>& freq, int32_t offset, const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    permuted_vec.cast_load(
        x + ((threadIdx.x * vec_size < rotary_dim / 2) ? threadIdx.x * vec_size + rotary_dim / 2
                                                       : threadIdx.x * vec_size - rotary_dim / 2));
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      float embed = float(offset) * freq[i];
      float cos, sin;
      __sincosf(embed, &sin, &cos);
      vec[i] = vec[i] * cos + ((threadIdx.x * vec_size < rotary_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin;
    }
  }
  return vec;
}

template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_cos_sin(
    const T* x,
    const vec_t<float, vec_size>& cos,
    const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    permuted_vec.cast_load(
        x + ((threadIdx.x * vec_size < rotary_dim / 2) ? threadIdx.x * vec_size + rotary_dim / 2
                                                       : threadIdx.x * vec_size - rotary_dim / 2));
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      vec[i] =
          vec[i] * cos[i] + ((threadIdx.x * vec_size < rotary_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin[i];
    }
  }
  return vec;
}

/*!
 * \brief Apply RoPE (Rotary Positional Embeddings) to x[0: head_dim] with interleave,
 *   return thread-local vector.
 * \tparam vec_size A template integer indicates the vector size used
 *   in the kernel
 * \tparam bdx A template integer indicates the blockDim.x
 * \tparam T A template type indicates the x data type
 * \param x A pointer to the start of x data
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param offset A integer indicates the offset of the position in RoPE
 */
template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_interleave(
    const T* x, const vec_t<float, vec_size>& freq, int32_t offset, const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      float embed = float(offset) * freq[i];
      float cos, sin;
      __sincosf(embed, &sin, &cos);
      vec[i] = vec[i] * cos + ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin;
    }
  }
  return vec;
}

template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_cos_sin_interleave(
    const T* x,
    const vec_t<float, vec_size>& cos,
    const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      vec[i] = vec[i] * cos[i] + ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin[i];
    }
  }
  return vec;
}

/*
HACK (ByronHsu): in the interleave mode with cos_sin_cache, we actually only use the first half of
cos and sin

For example,
In the below example, the vec_size is 4
the computation in the kernel is:
    [x1, x2, x3, x4...] * [cos1, cos1, cos2, cos2] + [-x2, x1, -x4, x3...] * [sin1, sin1, sin2,
sin2] the data we loaded are:
    - loaded vec = [x1, x2, x3, x4]
    - loaded cos = [cos1, cos2, cos3, cos4]
    - loaded sin = [sin1, sin2, sin3, sin4]
But only the first half of cos and sin is used in the computation.

However, we argue the additional overhead is acceptable:
    1. loading additional elements of cos and sin is not adding much overhead. The arithmetic
intensity is the same as non-interleave mode. Each elements of cos and sin is load twice
    2. we don't want two code paths of cos and sin vector for interleave and non-interleave mode.
*/
template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_cos_sin_interleave_reuse_half(
    const T* x,
    const vec_t<float, vec_size>& cos,
    const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      // i / 2 is to get the index of the first half of cos and sin
      vec[i] = vec[i] * cos[i / 2] + ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin[i / 2];
    }
  }
  return vec;
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType, typename IdType>
__global__ void BatchQKApplyRotaryPosIdsCosSinCacheWithSetKVBufferHeadParallelismKernel(
    DType* q,
    DType* k,
    DType* v,
    DType* q_rope,
    DType* k_rope,
    DType* k_buffer,
    DType* v_buffer,
    float* __restrict__ cos_sin_cache,
    IdType* __restrict__ pos_ids,
    uint32_t nnz,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t rotary_dim,
    size_t q_stride_n,
    size_t q_stride_h,
    size_t k_stride_n,
    size_t k_stride_h,
    size_t v_stride_n,
    size_t v_stride_h,
    size_t q_rope_stride_n,
    size_t q_rope_stride_h,
    size_t k_rope_stride_n,
    size_t k_rope_stride_h,
    size_t k_buffer_stride_n,
    size_t k_buffer_stride_h,
    size_t v_buffer_stride_n,
    size_t v_buffer_stride_h,
    DType k_scale,
    DType v_scale,
    IdType* __restrict__ cache_loc) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  const uint32_t bdy = blockDim.y;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    const int half_rotary_dim = rotary_dim / 2;

    // 1. if interleave:
    //  - cos = cos_sin_cache[pos_id][tx * vec_size // 2]
    //  - sin = cos_sin_cache[pos_id][(rot_dim // 2) + tx * vec_size // 2]
    // 2. if not interleave
    //  - cos = cos_cache[pos_id][(tx * vec_size) % (rot_dim // 2)]
    //  - sin = sin_cache[pos_id][(rot_dim // 2) + (tx * vec_size) % (rot_dim // 2)]
    if (tx * vec_size < rotary_dim) {
      int sin_offset = rotary_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;  // Force integer division
      } else {
        vec_idx = (tx * vec_size) % half_rotary_dim;  // Use half_rotary_dim
      }
      cos.load(cos_sin_cache + (pos * rotary_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rotary_dim) + (sin_offset + vec_idx));
    }

    if (by < num_qo_heads) {
      uint32_t qo_head_idx = by;
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr = q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    } else {
      uint32_t kv_head_idx = by - num_qo_heads;
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* v_ptr = v + get_elem_offset_impl(idx, kv_head_idx, 0, v_stride_n, v_stride_h);
      DType* k_rope_ptr = k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      const IdType cache_offset = cache_loc[idx];
      DType* k_buffer_ptr =
          k_buffer + get_elem_offset_impl(cache_offset, kv_head_idx, 0, k_buffer_stride_n, k_buffer_stride_h);
      DType* v_buffer_ptr =
          v_buffer + get_elem_offset_impl(cache_offset, kv_head_idx, 0, v_buffer_stride_n, v_buffer_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
      k_vec.cast_store(k_buffer_ptr + tx * vec_size);
      v.cast_store(v_buffer_ptr + tx * vec_size);
    }
  }
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType, typename IdType>
__global__ void BatchQKApplyRotaryPosIdsCosSinCacheWithSetKVBufferKernel(
    DType* q,
    DType* k,
    DType* v,
    DType* q_rope,
    DType* k_rope,
    DType* k_buffer,
    DType* v_buffer,
    float* __restrict__ cos_sin_cache,
    IdType* __restrict__ pos_ids,
    uint32_t nnz,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t rotary_dim,
    size_t q_stride_n,
    size_t q_stride_h,
    size_t k_stride_n,
    size_t k_stride_h,
    size_t v_stride_n,
    size_t v_stride_h,
    size_t q_rope_stride_n,
    size_t q_rope_stride_h,
    size_t k_rope_stride_n,
    size_t k_rope_stride_h,
    size_t k_buffer_stride_n,
    size_t k_buffer_stride_h,
    size_t v_buffer_stride_n,
    size_t v_buffer_stride_h,
    DType k_scale,
    DType v_scale,
    IdType* __restrict__ cache_loc) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];
    const int half_rotary_dim = rotary_dim / 2;

    // 1. if interleave:
    //  - cos = cos_sin_cache[pos_id][tx * vec_size // 2]
    //  - sin = cos_sin_cache[pos_id][(rot_dim // 2) + tx * vec_size // 2]
    // 2. if not interleave
    //  - cos = cos_cache[pos_id][(tx * vec_size) % (rot_dim // 2)]
    //  - sin = sin_cache[pos_id][(rot_dim // 2) + (tx * vec_size) % (rot_dim // 2)]
    if (tx * vec_size < rotary_dim) {
      int sin_offset = rotary_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;  // Force integer division
      } else {
        vec_idx = (tx * vec_size) % half_rotary_dim;  // Use half_rotary_dim
      }
      cos.load(cos_sin_cache + (pos * rotary_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rotary_dim) + (sin_offset + vec_idx));
    }

    // not to unroll the loop, because num head might be large and might lead to worse performance
#pragma unroll 1
    for (uint32_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr = q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    }

#pragma unroll 1
    for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx) {
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* v_ptr = v + get_elem_offset_impl(idx, kv_head_idx, 0, v_stride_n, v_stride_h);
      DType* k_rope_ptr = k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      const IdType cache_offset = cache_loc[idx];
      DType* k_buffer_ptr =
          k_buffer + get_elem_offset_impl(cache_offset, kv_head_idx, 0, k_buffer_stride_n, k_buffer_stride_h);
      DType* v_buffer_ptr =
          v_buffer + get_elem_offset_impl(cache_offset, kv_head_idx, 0, v_buffer_stride_n, v_buffer_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
      k_vec.cast_store(k_buffer_ptr + tx * vec_size);  // store the k_rope to the k_buffer
      v.cast_store(v_buffer_ptr + tx * vec_size);      // store the v to the v_buffer
    }
  }
}

#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                      \
    const bool INTERLEAVE = true;                        \
    __VA_ARGS__                                          \
  } else {                                               \
    const bool INTERLEAVE = false;                       \
    __VA_ARGS__                                          \
  }

template <typename DType, typename IdType>
cudaError_t BatchQKApplyRotaryPosIdsCosSinCacheWithSetKVBuffer(
    DType* q,
    DType* k,
    DType* v,
    DType* q_rope,
    DType* k_rope,
    DType* k_buffer,
    DType* v_buffer,
    float* cos_sin_cache,
    IdType* pos_ids,
    uint32_t nnz,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t rotary_dim,
    uint32_t head_dim,
    size_t q_stride_n,
    size_t q_stride_h,
    size_t k_stride_n,
    size_t k_stride_h,
    size_t v_stride_n,
    size_t v_stride_h,
    size_t q_rope_stride_n,
    size_t q_rope_stride_h,
    size_t k_rope_stride_n,
    size_t k_rope_stride_h,
    size_t k_buffer_stride_n,
    size_t k_buffer_stride_h,
    size_t v_buffer_stride_n,
    size_t v_buffer_stride_h,
    IdType* cache_loc,
    bool interleave,
    cudaStream_t stream = nullptr,
    DType k_scale = 1.0f,
    DType v_scale = 1.0f) {
  int dev_id = 0;
  int num_sms = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      // operate on 16 Bytes at a time
      constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
      // how many threads needed per head_dim
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      // how many threads needed per block
      uint32_t num_threads = std::max(128U, bdx);
      // how many tokens can we process in a block
      uint32_t bdy = num_threads / bdx;
      // how many blocks needed to process all tokens
      uint32_t nblks_x = (nnz + bdy - 1) / bdy;
      void* args[] = {
          (void*)&q,
          (void*)&k,
          (void*)&v,
          (void*)&q_rope,
          (void*)&k_rope,
          (void*)&k_buffer,
          (void*)&v_buffer,
          (void*)&cos_sin_cache,
          (void*)&pos_ids,
          (void*)&nnz,
          (void*)&num_qo_heads,
          (void*)&num_kv_heads,
          (void*)&rotary_dim,
          (void*)&q_stride_n,
          (void*)&q_stride_h,
          (void*)&k_stride_n,
          (void*)&k_stride_h,
          (void*)&v_stride_n,
          (void*)&v_stride_h,
          (void*)&q_rope_stride_n,
          (void*)&q_rope_stride_h,
          (void*)&k_rope_stride_n,
          (void*)&k_rope_stride_h,
          (void*)&k_buffer_stride_n,
          (void*)&k_buffer_stride_h,
          (void*)&v_buffer_stride_n,
          (void*)&v_buffer_stride_h,
          (void*)&k_scale,
          (void*)&v_scale,
          (void*)&cache_loc};
      auto kernel_0 =
          BatchQKApplyRotaryPosIdsCosSinCacheWithSetKVBufferKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;

      int num_blocks_per_sm_0 = 0;
      FLASHINFER_CUDA_CALL(
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm_0, kernel_0, num_threads, /*smem_size=*/0));
      uint32_t num_ctas_0 = num_blocks_per_sm_0 * num_sms;

      if ((nnz + bdy - 1) / bdy >= num_ctas_0) {
        dim3 nblks(nblks_x);
        dim3 nthrs(bdx, bdy);
        FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel_0, nblks, nthrs, args, 0, stream));
      } else {
        dim3 nblks(nblks_x, num_qo_heads + num_kv_heads);
        dim3 nthrs(bdx, bdy);
        auto kernel_1 = BatchQKApplyRotaryPosIdsCosSinCacheWithSetKVBufferHeadParallelismKernel<
            INTERLEAVE,
            HEAD_DIM,
            vec_size,
            bdx,
            DType,
            IdType>;
        FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel_1, nblks, nthrs, args, 0, stream));
      }
    });
  });

  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_POS_ENC_CUH_
