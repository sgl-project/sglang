/*!
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
#ifndef FLASHINFER_CASCADE_CUH_
#define FLASHINFER_CASCADE_CUH_

#include "../cp_async.cuh"
#include "../math.cuh"
#include "../utils.cuh"
#include "state.cuh"

namespace flashinfer {

using cp_async::PrefetchMode;
using cp_async::SharedMemFillMode;

/*!
 * \brief The CUDA kernel that merges the self-attention state of two index sets A and B.
 * \tparam vec_size The vector size used in the kernel.
 * \tparam DTypeIn The data type of v_a and v_b.
 * \tparam DTypeO The data type of v_merged.
 * \param v_a The partial v of index set A. (n, h, d)
 * \param s_a The logsumexp value of index set A. (n, h)
 * \param v_b The partial v of index set B. (n, h, d)
 * \param s_b The logsumexp value of index set B. (n, h)
 * \param v_merged The merged v of index set A union B. (n, h, d)
 * \param s_merged The merged logsumexp value of index set A union B. (n, h)
 * \param num_heads The number of heads of v_a and v_b.
 * \param head_dim The dimension of each head.
 * \note Both s_a and s_b are logsumexp values with base 2.
 */
template <uint32_t vec_size, typename DTypeIn, typename DTypeO>
__global__ void MergeStateKernel(DTypeIn* __restrict__ v_a, float* __restrict__ s_a,
                                 DTypeIn* __restrict__ v_b, float* __restrict__ s_b,
                                 DTypeO* __restrict__ v_merged, float* __restrict__ s_merged,
                                 uint32_t num_heads, uint32_t head_dim) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t pos = blockIdx.x;
  uint32_t head_idx = ty;

  float s_a_val = s_a[pos * num_heads + head_idx];
  float s_b_val = s_b[pos * num_heads + head_idx];
  float s_max = max(s_a_val, s_b_val);
  s_a_val = math::ptx_exp2(s_a_val - s_max);
  s_b_val = math::ptx_exp2(s_b_val - s_max);
  float a_scale = s_a_val / (s_a_val + s_b_val);
  float b_scale = s_b_val / (s_a_val + s_b_val);
  vec_t<float, vec_size> v_a_vec, v_b_vec, v_merged_vec;
  v_a_vec.cast_load(v_a + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  v_b_vec.cast_load(v_b + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    v_merged_vec[i] = a_scale * v_a_vec[i] + b_scale * v_b_vec[i];
  }
  v_merged_vec.cast_store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  if (s_merged != nullptr) {
    s_merged[pos * num_heads + head_idx] = math::ptx_log2(s_a_val + s_b_val) + s_max;
  }
}

/*!
 * \brief The CUDA kernel that merges the self-attention state with another state in-place.
 * \tparam vec_size The vector size used in the kernel.
 * \tparam DType The data type of v and v_other.
 * \param v The partial v to be updated in-place. (n, h, d)
 * \param s The logsumexp value to be updated in-place. (n, h)
 * \param v_other The other v to be merged. (n, h, d)
 * \param s_other The other logsumexp value to be merged. (n, h)
 * \param mask Optional mask of whether to merge given sequences or not. (n)
 * \param num_heads The number of heads of v and v_other.
 * \param head_dim The dimension of each head.
 * \note Both s and s_other are logsumexp values with base 2.
 */
template <uint32_t vec_size, typename DType>
__global__ void MergeStateInPlaceKernel(DType* __restrict__ v, float* __restrict__ s,
                                        DType* __restrict__ v_other, float* __restrict__ s_other,
                                        uint8_t* __restrict__ mask, uint32_t num_heads,
                                        uint32_t head_dim) {
  uint32_t pos = blockIdx.x;

  if (mask != nullptr && mask[pos] == 0) return;

  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t head_idx = ty;

  float s_val = s[pos * num_heads + head_idx];
  float s_other_val = s_other[pos * num_heads + head_idx];
  float s_max = max(s_val, s_other_val);
  s_val = math::ptx_exp2(s_val - s_max);
  s_other_val = math::ptx_exp2(s_other_val - s_max);
  float scale = s_val / (s_val + s_other_val);
  float other_scale = s_other_val / (s_val + s_other_val);
  vec_t<float, vec_size> v_vec, v_other_vec;
  v_vec.cast_load(v + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  v_other_vec.cast_load(v_other + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    v_vec[i] = scale * v_vec[i] + other_scale * v_other_vec[i];
  }
  v_vec.cast_store(v + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  if (s != nullptr) {
    s[pos * num_heads + head_idx] = math::ptx_log2(s_val + s_other_val) + s_max;
  }
}

template <uint32_t bdx, uint32_t bdy, uint32_t vec_size, typename DTypeIn>
__device__ __forceinline__ void threadblock_sync_state(state_t<vec_size>& st, DTypeIn* v_smem,
                                                       float* s_smem,
                                                       const uint32_t tx = threadIdx.x,
                                                       const uint32_t ty = threadIdx.y) {
  constexpr uint32_t head_dim = vec_size * bdx;
  st.o.cast_store(v_smem + ty * head_dim + tx * vec_size);
  s_smem[ty] = st.get_lse();
  st.init();
  __syncthreads();

#pragma unroll
  for (uint32_t iter = 0; iter < bdy; ++iter) {
    float s = s_smem[iter];
    vec_t<float, vec_size> v;
    v.cast_load(v_smem + iter * head_dim + tx * vec_size);
    st.merge(v, s, 1);
  }
}

template <uint32_t bdx, uint32_t bdy, uint32_t vec_size, typename DTypeIn>
__device__ __forceinline__ void warp_sync_state(state_t<vec_size>& st, DTypeIn* v_smem,
                                                float* s_smem, const uint32_t tx = threadIdx.x,
                                                const uint32_t ty = threadIdx.y) {
  constexpr uint32_t head_dim = vec_size * bdx;
  st.o.cast_store(v_smem + ty * head_dim + tx * vec_size);
  s_smem[ty] = st.get_lse();
  st.init();
  __syncwarp();

#pragma unroll
  for (uint32_t iter = 0; iter < bdy; ++iter) {
    float s = s_smem[iter];
    vec_t<float, vec_size> v;
    v.cast_load(v_smem + iter * head_dim + tx * vec_size);
    st.merge(v, s, 1);
  }
}

template <uint32_t bdx, uint32_t bdy, uint32_t vec_size, typename DTypeIn>
__device__ __forceinline__ void threadblock_sum(vec_t<float, vec_size>& v, DTypeIn* v_smem) {
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t head_dim = vec_size * bdx;
  v.cast_store(v_smem + ty * head_dim + tx * vec_size);
  v.fill(DTypeIn(0.f));
  __syncthreads();

#pragma unroll
  for (uint32_t iter = 0; iter < bdy; ++iter) {
    vec_t<float, vec_size> v_iter;
    v_iter.cast_load(v_smem + iter * head_dim + tx * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      v[i] += v_iter[i];
    }
  }
}

template <uint32_t vec_size, typename DTypeIn, typename DTypeO>
__global__ void AttentionSumKernel(DTypeIn* __restrict__ V, DTypeO* __restrict__ v_sum,
                                   uint32_t num_index_sets, uint32_t num_heads, uint32_t head_dim) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t pos = blockIdx.x;
  uint32_t head_idx = ty;

  if (num_index_sets == 0) {
    vec_t<DTypeO, vec_size> v;
    v.fill(DTypeO(0.f));
    v.store(v_sum + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
    return;
  }

  if (num_index_sets == 1) {
    vec_t<DTypeO, vec_size> v;
    v.cast_load(V + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
    v.store(v_sum + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
    return;
  }

  vec_t<float, vec_size> v_sum_vec;
  v_sum_vec.fill(0.f);
#pragma unroll 2
  for (uint32_t iter = 0; iter < num_index_sets; ++iter) {
    vec_t<float, vec_size> v;
    v.cast_load(V + ((pos * num_index_sets + iter) * num_heads + head_idx) * head_dim +
                tx * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      v_sum_vec[i] += v[i];
    }
  }

  v_sum_vec.cast_store(v_sum + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
}

template <uint32_t vec_size, typename DTypeIn, typename DTypeO>
__global__ void MergeStatesKernel(DTypeIn* __restrict__ V, float* __restrict__ S,
                                  DTypeO* __restrict__ v_merged, float* __restrict__ s_merged,
                                  uint32_t num_index_sets, uint32_t num_heads, uint32_t head_dim) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t pos = blockIdx.x;
  uint32_t head_idx = ty;

  if (num_index_sets == 0) {
    vec_t<DTypeO, vec_size> v;
    v.fill(DTypeO(0.f));
    v.store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
    if (s_merged != nullptr) {
      s_merged[pos * num_heads + head_idx] = -math::inf;
    }
    return;
  }

  if (num_index_sets == 1) {
    vec_t<DTypeO, vec_size> v;
    v.cast_load(V + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
    v.store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
    if (s_merged != nullptr) {
      s_merged[pos * num_heads + head_idx] = S[pos * num_heads + head_idx];
    }
    return;
  }

  state_t<vec_size> st;
#pragma unroll 2
  for (uint32_t iter = 0; iter < num_index_sets; ++iter) {
    float s = S[(pos * num_index_sets + iter) * num_heads + head_idx];
    vec_t<float, vec_size> v;
    v.cast_load(V + ((pos * num_index_sets + iter) * num_heads + head_idx) * head_dim +
                tx * vec_size);
    st.merge(v, s, 1);
  }

  st.normalize();
  st.o.cast_store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  if (s_merged != nullptr) {
    s_merged[pos * num_heads + head_idx] = st.get_lse();
  }
}

/*!
 * \brief The CUDA kernel that merges self-attention states of a list of index sets,
 *   accelerated for larger number of index sets.
 * \tparam vec_size The vector size used in the kernel.
 * \tparam bdx The blockDim.x used in the kernel.
 * \tparam bdy The blockDim.y used in the kernel.
 * \tparam num_smem_stages The number of stages of shared memory used in the kernel.
 * \tparam DTypeIn The data type of v.
 * \tparam DTypeO The data type of v_merged.
 * \param V The partial v of index sets. (n, num_index_sets, h, d)
 * \param S The logsumexp value of index sets. (n, num_index_sets, h)
 * \param v_merged The merged v of index sets union. (n, h, d)
 * \param s_merged The merged logsumexp value of index sets union. (n, h)
 * \param num_heads The number of heads of v.
 * \param head_dim The dimension of each head.
 * \note s are logsumexp values with base 2.
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t num_smem_stages, typename DTypeIn,
          typename DTypeO>
__global__ void MergeStatesLargeNumIndexSetsKernel(DTypeIn* __restrict__ V, float* __restrict__ S,
                                                   DTypeO* __restrict__ v_merged,
                                                   float* __restrict__ s_merged,
                                                   uint32_t num_index_sets, uint32_t num_heads) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t pos = blockIdx.x;
  uint32_t head_idx = blockIdx.y;
  state_t<vec_size> st;
  constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;
  constexpr uint32_t head_dim = vec_size * bdx;

  extern __shared__ uint8_t smem[];
  DTypeIn* v_smem = (DTypeIn*)smem;
  float* s_smem = (float*)(smem + num_smem_stages * bdy * head_dim * sizeof(DTypeIn));

#pragma unroll
  for (uint32_t iter = 0; iter < num_smem_stages; ++iter) {
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
        v_smem + (iter * bdy + ty) * head_dim + tx * vec_size,
        V + ((pos * num_index_sets + (iter * bdy + ty)) * num_heads + head_idx) * head_dim +
            tx * vec_size,
        (iter * bdy + ty) < num_index_sets);
    cp_async::commit_group();
  }
#pragma unroll 4
  for (uint32_t iter = 0; iter < ceil_div(num_index_sets, bdy); ++iter) {
    if (iter % bdx == 0) {
      s_smem[ty * bdx + tx] =
          iter * bdy + (ty * bdx + tx) < num_index_sets
              ? S[(pos * num_index_sets + (iter * bdy + ty * bdx + tx)) * num_heads + head_idx]
              : 0.f;
      __syncthreads();
    }
    cp_async::wait_group<num_smem_stages - 1>();
    __syncthreads();
    vec_t<float, vec_size> v;
    v.cast_load(v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size);
    if (iter * bdy + ty < num_index_sets) {
      float s = s_smem[(iter % bdx) * bdy + ty];
      st.merge(v, s, 1);
    }
    __syncthreads();
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
        v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size,
        V +
            ((pos * num_index_sets + ((iter + num_smem_stages) * bdy + ty)) * num_heads +
             head_idx) *
                head_dim +
            tx * vec_size,
        (iter + num_smem_stages) * bdy + ty < num_index_sets);
    cp_async::commit_group();
  }
  cp_async::wait_group<0>();
  __syncthreads();

  st.normalize();
  threadblock_sync_state<bdx, bdy, vec_size>(st, v_smem, s_smem);
  st.normalize();

  st.o.cast_store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  if (s_merged != nullptr) {
    s_merged[pos * num_heads + head_idx] = st.get_lse();
  }
}

/*!
 * \brief The CUDA kernel to merge self-attention states of multiple index sets, the number of
 * index sets at each position might vary.
 *
 * For CUDA graph support, the kernel can be built with a maximum sequence length and executed
 * using a truncated, dynamic sequence length passed through `seq_len_ptr`.
 *
 * \tparam vec_size The vector size used in the kernel.
 * \tparam bdx The blockDim.x used in the kernel.
 * \tparam bdy The blockDim.y used in the kernel.
 * \tparam num_smem_stages The number of stages of shared memory used in the kernel.
 * \tparam DTypeIn The data type of v.
 * \tparam DTypeO The data type of v_merged.
 * \param V The partial v of index sets. (nnz, h, d)
 * \param S The logsumexp value of index sets. (nnz, h)
 * \param indptr The start offsets of each position in the variable length array.
 * \param v_merged The merged v of index sets union. (n, h, d)
 * \param s_merged The merged logsumexp value of index sets union. (n, h)
 * \param max_seq_len The maximum sequence length supported by the kernel.
 * \param seq_len_ptr The current sequence length (number of positions populated in indptr).
 * \param num_heads The number of heads of v.
 * \param head_dim The dimension of each head.
 * \note s are logsumexp values with base 2.
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t num_smem_stages, typename DTypeIn,
          typename DTypeO, typename IdType>
__global__ void PersistentVariableLengthMergeStatesKernel(
    DTypeIn* __restrict__ V, float* __restrict__ S, IdType* indptr, DTypeO* __restrict__ v_merged,
    float* __restrict__ s_merged, uint32_t max_seq_len, uint32_t* __restrict__ seq_len_ptr,
    uint32_t num_heads) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t cta_id = blockIdx.x;
  uint32_t num_ctas = gridDim.x;
  const uint32_t seq_len = seq_len_ptr ? *seq_len_ptr : max_seq_len;
  uint32_t num_iters = ceil_div(seq_len * num_heads, num_ctas);
  constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;
  constexpr uint32_t head_dim = vec_size * bdx;
  extern __shared__ uint8_t smem[];
  DTypeIn* v_smem = (DTypeIn*)smem;
  float* s_smem = (float*)(smem + num_smem_stages * bdy * head_dim * sizeof(DTypeIn));

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll 1
  for (uint32_t i = cta_id; i < seq_len * num_heads; i += num_ctas) {
    uint32_t pos = i / num_heads;
    uint32_t head_idx = i % num_heads;
    state_t<vec_size> st;
    const uint32_t num_index_sets = indptr[pos + 1] - indptr[pos];

    if (num_index_sets == 0) {
      vec_t<DTypeO, vec_size> v;
      v.fill(DTypeO(0.f));
      v.store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
      if (s_merged != nullptr) {
        s_merged[pos * num_heads + head_idx] = -math::inf;
      }
      continue;
    }

    if (num_index_sets == 1) {
      vec_t<DTypeO, vec_size> v;
      v.cast_load(V + (indptr[pos] * num_heads + head_idx) * head_dim + tx * vec_size);
      v.store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
      if (s_merged != nullptr) {
        s_merged[pos * num_heads + head_idx] = S[indptr[pos] * num_heads + head_idx];
      }
      continue;
    }

#pragma unroll
    for (uint32_t iter = 0; iter < num_smem_stages; ++iter) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          v_smem + (iter * bdy + ty) * head_dim + tx * vec_size,
          V + ((indptr[pos] + (iter * bdy + ty)) * num_heads + head_idx) * head_dim + tx * vec_size,
          (iter * bdy + ty) < num_index_sets);
      cp_async::commit_group();
    }
#pragma unroll 4
    for (uint32_t iter = 0; iter < ceil_div(num_index_sets, bdy); ++iter) {
      if (iter % bdx == 0) {
        s_smem[ty * bdx + tx] =
            iter * bdy + (ty * bdx + tx) < num_index_sets
                ? S[(indptr[pos] + (iter * bdy + ty * bdx + tx)) * num_heads + head_idx]
                : 0.f;
        __syncthreads();
      }
      cp_async::wait_group<num_smem_stages - 1>();
      __syncthreads();
      vec_t<float, vec_size> v;
      v.cast_load(v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size);
      if (iter * bdy + ty < num_index_sets) {
        float s = s_smem[(iter % bdx) * bdy + ty];
        st.merge(v, s, 1);
      }
      __syncthreads();
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size,
          V +
              ((indptr[pos] + ((iter + num_smem_stages) * bdy + ty)) * num_heads + head_idx) *
                  head_dim +
              tx * vec_size,
          (iter + num_smem_stages) * bdy + ty < num_index_sets);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    __syncthreads();

    st.normalize();
    threadblock_sync_state<bdx, bdy, vec_size>(st, v_smem, s_smem);
    st.normalize();

    st.o.cast_store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
    if (s_merged != nullptr) {
      s_merged[pos * num_heads + head_idx] = st.get_lse();
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t num_smem_stages, typename DTypeIn,
          typename DTypeO, typename IdType>
__global__ void PersistentVariableLengthAttentionSumKernel(DTypeIn* __restrict__ V, IdType* indptr,
                                                           DTypeO* __restrict__ v_sum,
                                                           uint32_t max_seq_len,
                                                           uint32_t* __restrict__ seq_len_ptr,
                                                           uint32_t num_heads) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t cta_id = blockIdx.x;
  uint32_t num_ctas = gridDim.x;
  const uint32_t seq_len = seq_len_ptr ? *seq_len_ptr : max_seq_len;
  uint32_t num_iters = ceil_div(seq_len * num_heads, num_ctas);
  constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;
  constexpr uint32_t head_dim = vec_size * bdx;
  extern __shared__ uint8_t smem[];
  DTypeIn* v_smem = (DTypeIn*)smem;

  vec_t<float, vec_size> v_sum_vec;
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll 1
  for (uint32_t i = cta_id; i < seq_len * num_heads; i += num_ctas) {
    uint32_t pos = i / num_heads;
    uint32_t head_idx = i % num_heads;
    const uint32_t num_index_sets = indptr[pos + 1] - indptr[pos];

    if (num_index_sets == 0) {
      vec_t<DTypeO, vec_size> v;
      v.fill(DTypeO(0.f));
      v.store(v_sum + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
      continue;
    }

    if (num_index_sets == 1) {
      vec_t<DTypeO, vec_size> v;
      v.cast_load(V + (indptr[pos] * num_heads + head_idx) * head_dim + tx * vec_size);
      v.store(v_sum + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
      continue;
    }

#pragma unroll
    for (uint32_t iter = 0; iter < num_smem_stages; ++iter) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          v_smem + (iter * bdy + ty) * head_dim + tx * vec_size,
          V + ((indptr[pos] + (iter * bdy + ty)) * num_heads + head_idx) * head_dim + tx * vec_size,
          (iter * bdy + ty) < num_index_sets);
      cp_async::commit_group();
    }
#pragma unroll 4
    for (uint32_t iter = 0; iter < ceil_div(num_index_sets, bdy); ++iter) {
      cp_async::wait_group<num_smem_stages - 1>();
      __syncthreads();
      vec_t<float, vec_size> v;
      v.cast_load(v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size);
      if (iter * bdy + ty < num_index_sets) {
#pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
          v_sum_vec[i] += v[i];
        }
      }
      __syncthreads();
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim + tx * vec_size,
          V +
              ((indptr[pos] + ((iter + num_smem_stages) * bdy + ty)) * num_heads + head_idx) *
                  head_dim +
              tx * vec_size,
          (iter + num_smem_stages) * bdy + ty < num_index_sets);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    __syncthreads();

    threadblock_sum<bdx, bdy, vec_size>(v_sum_vec, v_smem);

    v_sum_vec.cast_store(v_sum + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

/*!
 * \brief Merge the self-attention state of two index sets A and B.
 * \tparam DTypeIn The data type of v_a and v_b.
 * \tparam DTypeO The data type of v_merged.
 * \param v_a The partial v of index set A (n, h, d)
 * \param s_a The logsumexp value of index set A. (n, h)
 * \param v_b The partial v of index set B. (n, h, d)
 * \param s_b The logsumexp value of index set B. (n, h)
 * \param v_merged The merged v of index set A union B. (n, h, d)
 * \param s_merged The merged logsumexp value of index set A union B. (n, h)
 * \param seq_len The sequence length.
 * \param num_heads The number of heads of v_a and v_b.
 * \param head_dim The dimension of each head.
 * \param stream The CUDA stream to execute the kernel.
 * \return status Indicates whether CUDA calls are successful
 * \note Both s_a and s_b are logsumexp values with base 2.
 */
template <typename DTypeIn, typename DTypeO>
cudaError_t MergeState(DTypeIn* v_a, float* s_a, DTypeIn* v_b, float* s_b, DTypeO* v_merged,
                       float* s_merged, uint32_t seq_len, uint32_t num_heads, uint32_t head_dim,
                       cudaStream_t stream = nullptr) {
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    dim3 nblks(seq_len);
    dim3 nthrs(bdx, bdy);
    auto kernel = MergeStateKernel<vec_size, DTypeIn, DTypeO>;
    void* args[] = {&v_a, &s_a, &v_b, &s_b, &v_merged, &s_merged, &num_heads, &head_dim};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

/*!
 * \brief Merge the self-attention state with another state in place.
 * \tparam DType The data type of v and v_other.
 * \param v The partial v to be updated in-place. (n, h, d)
 * \param s The logsumexp value to be updated in-place. (n, h)
 * \param v_other The other v to be merged. (n, h, d)
 * \param s_other The other logsumexp value to be merged. (n, h)
 * \param seq_len The sequence length.
 * \param num_heads The number of heads of v and v_other.
 * \param head_dim The dimension of each head.
 * \param mask Optional mask of whether to merge given sequences or not. (n)
 * \param stream The CUDA stream to execute the kernel.
 * \return status Indicates whether CUDA calls are successful
 * \note Both s and s_other are logsumexp values with base 2.
 */
template <typename DType>
cudaError_t MergeStateInPlace(DType* v, float* s, DType* v_other, float* s_other, uint32_t seq_len,
                              uint32_t num_heads, uint32_t head_dim, uint8_t* mask = nullptr,
                              cudaStream_t stream = nullptr) {
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DType), HEAD_DIM / 32U);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    dim3 nblks(seq_len);
    dim3 nthrs(bdx, bdy);
    auto kernel = MergeStateInPlaceKernel<vec_size, DType>;
    void* args[] = {&v, &s, &v_other, &s_other, &mask, &num_heads, &head_dim};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

/*!
 * \brief Merge self-attention states of a list of index sets.
 * \tparam DTypeIn The data type of v.
 * \tparam DTypeO The data type of v_merged.
 * \param v The partial v of index sets. (n, num_index_sets, h, d)
 * \param s The logsumexp value of index sets. (n, num_index_sets, h)
 * \param v_merged The merged v of index sets union. (n, h, d)
 * \param s_merged The merged logsumexp value of index sets union. (n, h)
 * \param num_index_sets The number of index sets.
 * \param seq_len The sequence length.
 * \param num_heads The number of heads of v.
 * \param head_dim The dimension of each head.
 * \param stream The CUDA stream to execute the kernel.
 * \return status Indicates whether CUDA calls are successful
 * \note s are logsumexp values with base 2.
 */
template <typename DTypeIn, typename DTypeO>
cudaError_t MergeStates(DTypeIn* v, float* s, DTypeO* v_merged, float* s_merged,
                        uint32_t num_index_sets, uint32_t seq_len, uint32_t num_heads,
                        uint32_t head_dim, cudaStream_t stream = nullptr) {
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    if (num_index_sets >= seq_len) {
      constexpr uint32_t num_threads = 128;
      constexpr uint32_t bdy = num_threads / bdx;
      dim3 nblks(seq_len, num_heads);
      dim3 nthrs(bdx, bdy);
      constexpr uint32_t num_smem_stages = 4;
      auto kernel =
          MergeStatesLargeNumIndexSetsKernel<vec_size, bdx, bdy, num_smem_stages, DTypeIn, DTypeO>;
      void* args[] = {&v, &s, &v_merged, &s_merged, &num_index_sets, &num_heads};
      uint32_t smem_size =
          num_smem_stages * bdy * head_dim * sizeof(DTypeIn) + num_threads * sizeof(float);
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    } else {
      uint32_t bdy = num_heads;
      dim3 nblks(seq_len);
      dim3 nthrs(bdx, bdy);
      auto kernel = MergeStatesKernel<vec_size, DTypeIn, DTypeO>;
      void* args[] = {&v, &s, &v_merged, &s_merged, &num_index_sets, &num_heads, &head_dim};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
    }
  });
  return cudaSuccess;
}

template <typename DTypeIn, typename DTypeO>
cudaError_t AttentionSum(DTypeIn* v, DTypeO* v_sum, uint32_t num_index_sets, uint32_t seq_len,
                         uint32_t num_heads, uint32_t head_dim, cudaStream_t stream = nullptr) {
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    dim3 nblks(seq_len);
    dim3 nthrs(bdx, bdy);
    auto kernel = AttentionSumKernel<vec_size, DTypeIn, DTypeO>;
    void* args[] = {&v, &v_sum, &num_index_sets, &num_heads, &head_dim};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

template <typename DTypeIn, typename DTypeO, typename IdType>
cudaError_t VariableLengthMergeStates(DTypeIn* v, float* s, IdType* indptr, DTypeO* v_merged,
                                      float* s_merged, uint32_t max_seq_len, uint32_t* seq_len,
                                      uint32_t num_heads, uint32_t head_dim, bool enable_pdl,
                                      cudaStream_t stream = nullptr) {
  int dev_id = 0;
  int num_sms = 0;
  int num_blocks_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    constexpr uint32_t num_threads = 128;
    constexpr uint32_t bdy = num_threads / bdx;
    constexpr uint32_t num_smem_stages = 4;
    uint32_t smem_size =
        num_smem_stages * bdy * head_dim * sizeof(DTypeIn) + num_threads * sizeof(float);
    auto kernel = PersistentVariableLengthMergeStatesKernel<vec_size, bdx, bdy, num_smem_stages,
                                                            DTypeIn, DTypeO, IdType>;
    FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                       num_threads, smem_size));
    num_blocks_per_sm = min(num_blocks_per_sm, ceil_div(max_seq_len * num_heads, num_sms));

    dim3 nblks(num_sms * num_blocks_per_sm);
    dim3 nthrs(bdx, bdy);
    void* args[] = {&v, &s, &indptr, &v_merged, &s_merged, &max_seq_len, &seq_len, &num_heads};
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // PDL launch
    if (enable_pdl) {
      cudaLaunchAttribute attribute[1];
      attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute[0].val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t config;
      config.attrs = attribute;
      config.numAttrs = 1;
      config.gridDim = nblks;
      config.blockDim = nthrs;
      config.dynamicSmemBytes = smem_size;
      config.stream = stream;
      FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, v, s, indptr, v_merged, s_merged,
                                              max_seq_len, seq_len, num_heads));
    } else {
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    }
  });
  return cudaSuccess;
}

template <typename DTypeIn, typename DTypeO, typename IdType>
cudaError_t VariableLengthAttentionSum(DTypeIn* v, IdType* indptr, DTypeO* v_sum,
                                       uint32_t max_seq_len, uint32_t* seq_len, uint32_t num_heads,
                                       uint32_t head_dim, bool enable_pdl,
                                       cudaStream_t stream = nullptr) {
  int dev_id = 0;
  int num_sms = 0;
  int num_blocks_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    constexpr uint32_t num_threads = 128;
    constexpr uint32_t bdy = num_threads / bdx;
    constexpr uint32_t num_smem_stages = 4;
    uint32_t smem_size = num_smem_stages * bdy * head_dim * sizeof(DTypeIn);
    auto kernel = PersistentVariableLengthAttentionSumKernel<vec_size, bdx, bdy, num_smem_stages,
                                                             DTypeIn, DTypeO, IdType>;
    FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                       num_threads, smem_size));
    num_blocks_per_sm = min(num_blocks_per_sm, ceil_div(max_seq_len * num_heads, num_sms));

    dim3 nblks(num_sms * num_blocks_per_sm);
    dim3 nthrs(bdx, bdy);
    void* args[] = {&v, &indptr, &v_sum, &max_seq_len, &seq_len, &num_heads};
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    if (enable_pdl) {
      // PDL launch
      cudaLaunchAttribute attribute[1];
      attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute[0].val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t config;
      config.attrs = attribute;
      config.numAttrs = 1;
      config.gridDim = nblks;
      config.blockDim = nthrs;
      config.dynamicSmemBytes = smem_size;
      config.stream = stream;
      FLASHINFER_CUDA_CALL(
          cudaLaunchKernelEx(&config, kernel, v, indptr, v_sum, max_seq_len, seq_len, num_heads));
    } else {
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    }
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_CASCADE_CUH_
