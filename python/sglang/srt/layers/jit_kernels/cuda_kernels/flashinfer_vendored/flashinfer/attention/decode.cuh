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
#ifndef FLASHINFER_DECODE_CUH_
#define FLASHINFER_DECODE_CUH_
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <iostream>

#include "../cp_async.cuh"
#include "../math.cuh"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "cascade.cuh"
#include "state.cuh"

namespace flashinfer {

DEFINE_HAS_MEMBER(decode_maybe_q_rope_offset)

namespace cg = cooperative_groups;
using cp_async::PrefetchMode;
using cp_async::SharedMemFillMode;

namespace {

/*!
 * \brief Load k tile from smem and compute qk
 * \tparam pos_encoding_mode The positional encoding mode used in the kernel
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param q_vec A vector of float indicates the thread-local query vector
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 *   in shared memory of different pipeline stages
 * \param kv_idx A integer indicates the thread-local kv position in kv-cache
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param s A float indicates the thread-local result of qk
 * \param st The self-attention state to be updated
 */
template <PosEncodingMode pos_encoding_mode, uint32_t vec_size, uint32_t bdx, uint32_t tile_size,
          typename AttentionVariant, typename Params, typename T>
__device__ __forceinline__ void compute_qk(
    const Params& params, AttentionVariant variant, const uint32_t batch_idx, const T* smem,
    const vec_t<float, vec_size>& q_vec, const vec_t<float, vec_size>& freq, uint32_t kv_idx_base,
    uint32_t iter_base, uint32_t iter_bound, uint32_t qo_head_idx, uint32_t kv_head_idx, float* s,
    state_t<vec_size>& st, const uint32_t tx, const uint32_t ty, const uint32_t tz) {
  float m_prev = st.m;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> k_vec;
    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      // apply rotary embedding for all rows in k matrix of kv-cache
      k_vec = vec_apply_llama_rope<vec_size, bdx>(smem + j * bdx * vec_size, freq,
                                                  kv_idx_base + tz * tile_size + j);
    } else {
      // do not apply rotary embedding
      k_vec.cast_load(smem + (j * bdx + tx) * vec_size);
    }
    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      s[j] += q_vec[i] * k_vec[i];
    }
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s[j] += math::shfl_xor_sync(s[j], offset);
    }
    const uint32_t pos = kv_idx_base + tz * tile_size + j;
    s[j] = variant.LogitsTransform(params, s[j], batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos,
                                   qo_head_idx, kv_head_idx);
    if constexpr (variant.use_softmax) {
      s[j] *= variant.sm_scale_log2;
    }

    bool mask = variant.LogitsMask(params, batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos, qo_head_idx,
                                   kv_head_idx);
    s[j] = (iter_base + tz * tile_size + j < iter_bound && mask) ? s[j] : -math::inf;
    st.m = max(st.m, s[j]);
  }

  if constexpr (variant.use_softmax) {
    float o_scale = math::ptx_exp2(m_prev - st.m);
    st.d *= o_scale;
#pragma unroll
    for (uint32_t j = 0; j < tile_size; ++j) {
      s[j] = math::ptx_exp2(s[j] - st.m);
      st.d += s[j];
    }
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[i] = st.o[i] * o_scale;
    }
  }
}

/*!
 * \brief Load v tile from shared memory and update local state
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param s A float indicates the pre-softmax attention score
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 * in shared memory of different pipeline stages
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param st The flashattention state to be updated
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename T>
__device__ __forceinline__ void update_local_state(const T* smem, const float* s,
                                                   uint32_t compute_stage_idx,
                                                   state_t<vec_size>& st, uint32_t tx) {
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> v_vec;
    v_vec.cast_load(smem + (j * bdx + tx) * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[i] = st.o[i] + s[j] * v_vec[i];
    }
  }
}

/*!
 * \brief Synchronize the state of all warps inside a threadblock.
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \param st The warp local state
 * \param smem The pointer to shared memory buffer for o
 * \param smem_md The pointer to shared memory buffer for m/d
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant>
__device__ __forceinline__ void sync_state(AttentionVariant variant, state_t<vec_size>& st,
                                           float* smem, float* smem_md, const uint32_t tx,
                                           const uint32_t ty, const uint32_t tz) {
  if constexpr (bdz > 1) {
    constexpr uint32_t head_dim = bdx * vec_size;
    auto block = cg::this_thread_block();
    st.o.store(smem + (tz * bdy + ty) * head_dim + tx * vec_size);
    if constexpr (variant.use_softmax) {
      smem_md[(tz * bdy + ty) * 2] = st.m;
      smem_md[(tz * bdy + ty) * 2 + 1] = st.d;
      block.sync();
      st.init();
#pragma unroll
      for (uint32_t j = 0; j < bdz; ++j) {
        float mz = smem_md[(j * bdy + ty) * 2], dz = smem_md[(j * bdy + ty) * 2 + 1];
        vec_t<float, vec_size> oz;
        oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
        st.merge(oz, mz, dz);
      }
    } else {
      block.sync();
      st.init();
#pragma unroll
      for (uint32_t j = 0; j < bdz; ++j) {
        vec_t<float, vec_size> oz;
        oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
#pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
          st.o[i] += oz[i];
        }
      }
    }
  }
}

}  // namespace

/*!
 * \brief FlashAttention decoding cuda kernel with kv-cache for a single request
 * \tparam pos_encoding_mode The positional encoding mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \param q [num_qo_heads, head_dim] The query matrix
 * \param k [seq_len, num_kv_heads, head_dim] The key matrix in kv-cache
 * \param v [seq_len, num_kv_heads, head_dim] The value matrix in kv-cache
 * \param o [num_qo_heads, head_dim] The output matrix
 * \param head_dim A integer indicates the head dimension
 * \param rope_rcp_scale A floating number indicate the reciprocal
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_rcp_theta A floating number indicate the reciprocal
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 * \param kv_chunk_size A integer indicates the kv-chunk size
 */
template <PosEncodingMode pos_encoding_mode, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void SingleDecodeWithKVCacheKernel(const __grid_constant__ Params params) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const DTypeQ* q = params.q;
  const DTypeKV* k = params.k;
  const DTypeKV* v = params.v;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  const uint32_t kv_stride_n = params.kv_stride_n;
  const uint32_t kv_stride_h = params.kv_stride_h;
  DTypeO* o = params.o;
  float* lse = params.lse;
  uint32_t kv_chunk_size = params.kv_chunk_size;

  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();

  constexpr uint32_t head_dim = bdx * vec_size;
  uint32_t kv_head_idx = blockIdx.y;
  uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;
  uint32_t kv_chunk_idx = blockIdx.x;
  uint32_t num_qo_heads = params.num_qo_heads;

  extern __shared__ uint8_t smem[];
  AttentionVariant variant(params, /*batch_idx=*/0, smem);
  const uint32_t seq_len = variant.kv_len;
  DTypeKV* k_smem = (DTypeKV*)smem;
  DTypeKV* v_smem = (DTypeKV*)(smem + num_stages_smem * bdy * tile_size_per_bdx * bdz * head_dim *
                                          sizeof(DTypeKV));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * bdy * tile_size_per_bdx * bdz * head_dim *
                                       sizeof(DTypeKV));

  uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_rcp_scale *
                __powf(rope_rcp_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }

    // apply rotary embedding to q matrix
    q_vec = vec_apply_llama_rope<vec_size, bdx>(q + qo_head_idx * q_stride_h, freq, seq_len - 1);
  } else {
    // do not apply rotary embedding to q matrix
    q_vec.cast_load(q + qo_head_idx * q_stride_h + tx * vec_size);
  }
  block.sync();

  uint32_t chunk_start = kv_chunk_idx * kv_chunk_size;
  kv_chunk_size = min(kv_chunk_size, seq_len - chunk_start);
  uint32_t chunk_end = chunk_start + kv_chunk_size;

  // preload k tiles and v tiles
  uint32_t producer_kv_idx_base = chunk_start;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size * 8;
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          v + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();
    producer_kv_idx_base += bdy * bdz * tile_size_per_bdx;
  }

  // pipelining k/v tiles loading and state updating
  uint32_t consumer_kv_idx_base = chunk_start, stage_idx = 0;
  state_t<vec_size> st_local;
  float s[bdy * tile_size_per_bdx];

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(kv_chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) {
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    compute_qk<pos_encoding_mode, vec_size, bdx, bdy * tile_size_per_bdx>(
        params, variant, /*batch_idx=*/0,
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, q_vec, freq,
        consumer_kv_idx_base, iter * bdy * tile_size_per_bdx * bdz, kv_chunk_size, qo_head_idx,
        kv_head_idx, s, st_local, tx, ty, tz);
    block.sync();
    // load k
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();

    // update m/d/o state
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s, stage_idx,
        st_local, tx);
    block.sync();

    // load v
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          v + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();

    stage_idx = (stage_idx + 1) % num_stages_smem;
    producer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
    consumer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
  }
  cp_async::wait_group<0>();
  block.sync();

  // sync local state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy, bdz>(variant, st_local, reinterpret_cast<float*>(smem), smem_md,
                                      tx, ty, tz);
  if constexpr (variant.use_softmax) {
    st_local.normalize();
  }

  st_local.o.cast_store(o + (kv_chunk_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
  if (lse != nullptr) {
    lse[kv_chunk_idx * num_qo_heads + qo_head_idx] = st_local.get_lse();
  }
}

/*!
 * \brief FlashAttention decoding cuda kernel with paged kv-cache for multiple requests
 * \tparam pos_encoding_mode The positional encoding mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam bdz A template integer indicates the block size in z dimension
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param q [batch_size, num_qo_heads, head_dim] The query matrix
 * \param paged_kv The paged kv-cache data structure
 * \param o [num_qo_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer
 * \param lse The logsumexp values
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param rope_rcp_scale A floating number indicate the reciprocal
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_rcp_theta A floating number indicate the reciprocal
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 */
template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__device__ __inline__ void BatchDecodeWithPagedKVCacheDevice(const Params& params, uint8_t smem[],
                                                             const uint32_t bx = blockIdx.x,
                                                             const uint32_t by = blockIdx.y,
                                                             const uint32_t tx = threadIdx.x,
                                                             const uint32_t ty = threadIdx.y,
                                                             const uint32_t tz = threadIdx.z) {
  auto block = cg::this_thread_block();
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const DTypeQ* q = params.q;
  DTypeO* o = params.o;
  float* lse = params.lse;
  const auto paged_kv = params.paged_kv;
  const bool* block_valid_mask = params.block_valid_mask;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const bool partition_kv = params.partition_kv;

  constexpr uint32_t head_dim = bdx * vec_size;
  const uint32_t batch_idx = params.request_indices[bx];
  const uint32_t kv_tile_idx = params.kv_tile_indices[bx];
  const uint32_t kv_head_idx = by;
  const uint32_t qo_head_idx = kv_head_idx * bdy + ty;
  // NOTE(Zihao): when CUDAGraph is enabled, we will launch more blocks than
  // the actual batch size, so we need to check if the current batch is valid
  if (block_valid_mask && !block_valid_mask[bx]) return;
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
  const uint32_t kv_len = paged_kv.get_length(batch_idx);
  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;

  AttentionVariant variant(params, batch_idx, smem);
  DTypeKV* k_smem = (DTypeKV*)smem;
  DTypeKV* v_smem = (DTypeKV*)(smem + num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
                                          sizeof(DTypeKV));
  size_t* kv_offset_smem = (size_t*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz *
                                                head_dim * sizeof(DTypeKV));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
                                       sizeof(DTypeKV));

  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    const IdType* q_rope_offset = nullptr;
    if constexpr (has_decode_maybe_q_rope_offset_v<Params>) {
      q_rope_offset = params.decode_maybe_q_rope_offset;
    }
    int32_t q_rope_offset_val = q_rope_offset == nullptr ? (kv_len - 1) : q_rope_offset[batch_idx];
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_rcp_scale *
                __powf(rope_rcp_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    // apply rotary embedding to q matrix
    q_vec = vec_apply_llama_rope<vec_size, bdx>(
        q + batch_idx * q_stride_n + qo_head_idx * q_stride_h, freq, q_rope_offset_val);
  } else {
// do not apply rotary embedding to q matrix
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    q_vec.cast_load(q + batch_idx * q_stride_n + qo_head_idx * q_stride_h + tx * vec_size);
  }

  // preload k/v tiles
  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size * 8;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  static_assert(num_stages_smem <= bdx);
  uint32_t packed_page_iter_base = paged_kv.indptr[batch_idx] * paged_kv.page_size + chunk_start;
#pragma unroll
  for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    uint32_t q, r;
    paged_kv.page_size.divmod(packed_page_iter_base + ((j * bdz + tz) * bdy + ty) * bdx + tx, q, r);
    kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
        paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
  }
  block.sync();

  size_t kv_offset[tile_size_per_bdx];
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] =
          kv_offset_smem[((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j] + tx * vec_size;
    }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }

  state_t<vec_size> st;
  float s[bdy * tile_size_per_bdx];

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) {
    if ((iter + num_stages_smem) % bdx == 0) {
#pragma unroll
      for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        uint32_t q, r;
        paged_kv.page_size.divmod(
            packed_page_iter_base + ((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
                                     ((j * bdz + tz) * bdy + ty) * bdx + tx),
            q, r);
        kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
            paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
      }
    }
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    compute_qk<POS_ENCODING_MODE, vec_size, bdx, bdy * tile_size_per_bdx>(
        params, variant, batch_idx,
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, q_vec, freq,
        (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[batch_idx]) +
            chunk_start + iter * tile_size_per_bdx * bdy * bdz,
        iter * tile_size_per_bdx * bdy * bdz, chunk_size, qo_head_idx, kv_head_idx, s, st, tx, ty,
        tz);
    block.sync();

#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] = kv_offset_smem[((((iter + num_stages_smem) % bdx) * bdz + tz) * bdy + ty) *
                                        tile_size_per_bdx +
                                    j] +
                     tx * vec_size;
    }

    // load k tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();

    // update m/d/o states
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s, stage_idx, st, tx);
    block.sync();

    // load v tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>();
  block.sync();

  // sync local state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy, bdz>(variant, st, reinterpret_cast<float*>(smem), smem_md, tx, ty,
                                      tz);
  if constexpr (variant.use_softmax) {
    st.normalize();
  }

  if (tz == 0) {
    st.o.cast_store(o + (bx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
    // write lse
    if (lse != nullptr) {
      lse[bx * num_qo_heads + qo_head_idx] = st.get_lse();
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernel(const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  BatchDecodeWithPagedKVCacheDevice<POS_ENCODING_MODE, num_stages_smem, tile_size_per_bdx, vec_size,
                                    bdx, bdy, bdz, AttentionVariant>(params, smem);
}

/*!
 * \brief Get the heuristic number of threads per threadblock
 * \param group_size The number of qo heads that maps to the same kv head in GQA.
 * \param sizeof_dtype The size (in terms of bytes) of the input data type
 */
constexpr uint32_t get_heuristic_num_threads(uint32_t group_size, uint32_t sizeof_dtype) {
  if (group_size == 8U) {
    if (sizeof_dtype == 1U) {
      return 256U;  // not enough registers for 512 threads
    } else {
      return 512U;
    }
  } else {
    return 128U;
  }
}

/*!
 * \brief FlashAttention decoding with kv-cache for a single request
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \param q The query matrix, shape: [num_qo_heads, head_dim]
 * \param k The key matrix in kv-cache, shape: [seq_len, num_kv_heads, head_dim]
 *   for NHD layout, [num_kv_heads, seq_len, head_dim] for HND layout
 * \param v The value matrix in kv-cache, shape: [seq_len, num_kv_heads,
 *   head_dim] for NHD layout, [num_kv_heads, seq_len, head_dim] for HND layout
 * \param o The output matrix, shape: [num_qo_heads, head_dim]
 * \param tmp Used-allocated temporary buffer
 * \param num_qo_heads A integer indicates the number of heads of query and output
 * \param num_kv_heads A integer indicates the number of heads of key and value
 * \param seq_len A integer indicates the sequence length
 * \param head_dim A integer indicates the head dimension
 * \param pos_encoding_mode The positional encoding mode
 * \param rope_scale The scaling factor used in RoPE Interpolation
 * \param rope_theta The theta used in RoPE
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t SingleDecodeWithKVCacheDispatched(Params params, typename Params::DTypeO* tmp,
                                              cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.num_kv_heads;
  const uint32_t seq_len = params.kv_len;

  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  auto compute_capacity = GetCudaComputeCapability();
  static_assert(bdx <= 32U);
  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads =
        std::max(get_heuristic_num_threads(GROUP_SIZE, sizeof(DTypeKV)), bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 8U) : 1U;
    DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
      const uint32_t smem_size =
          2U * NUM_STAGES_SMEM * bdy * tile_size_per_bdx * bdz * HEAD_DIM * sizeof(DTypeKV) +
          2U * bdy * bdz * sizeof(float);
      auto kernel =
          SingleDecodeWithKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
                                        vec_size, bdx, bdy, bdz, AttentionVariant, Params>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      if (seq_len <= 256 || tmp == nullptr) {
        // no need to use partition-kv kernel
        dim3 nblks = dim3(1, num_kv_heads);
        dim3 nthrs = dim3(bdx, bdy, bdz);
        params.kv_chunk_size = seq_len;
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      } else {
        // use partition-kv kernel
        int num_blocks_per_sm = 0;
        int num_sm = 0;
        int dev_id = 0;
        FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
        FLASHINFER_CUDA_CALL(
            cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
        FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, kernel, num_threads, smem_size));
        uint32_t max_grid_size = uint32_t(num_blocks_per_sm) * uint32_t(num_sm);
        uint32_t max_num_kv_chunks = max_grid_size / num_kv_heads;
        uint32_t kv_chunk_size = max(ceil_div(seq_len, max_num_kv_chunks), 256);
        uint32_t num_chunks = ceil_div(seq_len, kv_chunk_size);
        dim3 nblks = dim3(num_chunks, num_kv_heads);
        if (nblks.x == 0 || nblks.y == 0) {
          std::ostringstream err_msg;
          err_msg << "Invalid kernel configuration: nblks=(" << nblks.x << "," << nblks.y << ")";
          FLASHINFER_ERROR(err_msg.str());
        }
        dim3 nthrs = dim3(bdx, bdy, bdz);
        float* tmp_lse = (float*)(tmp + num_chunks * num_qo_heads * HEAD_DIM);
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp;
        params.lse = tmp_lse;
        params.kv_chunk_size = kv_chunk_size;
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(
              MergeStates(tmp, tmp_lse, o, lse, num_chunks, 1, num_qo_heads, HEAD_DIM, stream));
        } else {
          FLASHINFER_CUDA_CALL(AttentionSum(tmp, o, num_chunks, 1, num_qo_heads, HEAD_DIM, stream));
        }
      }
    });
  });
  return cudaSuccess;
}

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                  float* tmp_s, bool enable_pdl,
                                                  cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.paged_kv.num_heads;
  const uint32_t padded_batch_size = params.padded_batch_size;

  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  auto compute_capacity = GetCudaComputeCapability();
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  static_assert(bdx <= 32);
  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;
    DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
      const uint32_t smem_size =
          2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV) +
          std::max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*),
                   2 * bdy * bdz * sizeof(float));
      auto kernel =
          BatchDecodeWithPagedKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
                                            vec_size, bdx, bdy, bdz, AttentionVariant, Params>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      dim3 nblks(padded_batch_size, num_kv_heads);
      dim3 nthrs(bdx, bdy, bdz);

      // PDL launch config
      cudaLaunchAttribute attribute[1];
      cudaLaunchConfig_t config;
      if (enable_pdl) {
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
        config.gridDim = nblks;
        config.blockDim = nthrs;
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
      }
      if (tmp_v == nullptr) {
        // do not use partition-kv kernel
        params.partition_kv = false;

        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          void* args[] = {(void*)&params};
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
      } else {
        // use partition-kv kernel
        params.partition_kv = true;
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp_v;
        params.lse = tmp_s;
        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          void* args[] = {(void*)&params};
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
              tmp_v, tmp_s, params.o_indptr, o, lse, params.paged_kv.batch_size, nullptr,
              num_qo_heads, HEAD_DIM, enable_pdl, stream));
        } else {
          FLASHINFER_CUDA_CALL(
              VariableLengthAttentionSum(tmp_v, params.o_indptr, o, params.paged_kv.batch_size,
                                         nullptr, num_qo_heads, HEAD_DIM, enable_pdl, stream));
        }
      }
    });
  });
  return cudaSuccess;
}

template <uint32_t vec_size_ckv, uint32_t vec_size_kpe, uint32_t bdx, uint32_t tile_size,
          typename AttentionVariant, typename Params, typename T>
__device__ __forceinline__ void compute_qk_and_update_local_stat_mla(
    const Params& params, AttentionVariant variant, const uint32_t batch_idx, const T* ckv_smem,
    const vec_t<float, vec_size_ckv>& q_nope_vec, const T* kpe_smem,
    const vec_t<float, vec_size_kpe>& q_pe_vec, const vec_t<float, vec_size_kpe>& freq,
    uint32_t kv_idx_base, uint32_t iter_base, uint32_t iter_bound, state_t<vec_size_ckv>& st) {
  uint32_t tx = threadIdx.x, tz = threadIdx.z;
  constexpr uint32_t head_dim_ckv = bdx * vec_size_ckv;
  constexpr uint32_t head_dim_kpe = bdx * vec_size_kpe;
  float s[tile_size];
  float m_prev = st.m;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size_ckv> ckv_vec;
    ckv_vec.cast_load(ckv_smem + j * head_dim_ckv + tx * vec_size_ckv);

    vec_t<float, vec_size_kpe> kpe_vec;
    kpe_vec.cast_load(kpe_smem + j * head_dim_kpe + tx * vec_size_kpe);

    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size_ckv; ++i) {
      s[j] += q_nope_vec[i] * ckv_vec[i];
    }
#pragma unroll
    for (uint32_t i = 0; i < vec_size_kpe; ++i) {
      s[j] += q_pe_vec[i] * kpe_vec[i];
    }
    s[j] *= params.sm_scale;
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s[j] += math::shfl_xor_sync(s[j], offset);
    }
    s[j] = (iter_base + tz * tile_size + j < iter_bound) ? s[j] : -math::inf;
    st.m = max(st.m, s[j]);
  }

  float o_scale = math::ptx_exp2(m_prev - st.m);
  st.d *= o_scale;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    s[j] = math::ptx_exp2(s[j] - st.m);
    st.d += s[j];
  }
#pragma unroll
  for (uint32_t i = 0; i < vec_size_ckv; ++i) {
    st.o[i] = st.o[i] * o_scale;
  }

#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size_ckv> v_vec;
    v_vec.cast_load(ckv_smem + j * head_dim_ckv + tx * vec_size_ckv);
#pragma unroll
    for (uint32_t i = 0; i < vec_size_ckv; ++i) {
      st.o[i] = st.o[i] + s[j] * v_vec[i];
    }
  }
}

template <uint32_t num_stages_smem, uint32_t vec_size_ckv, uint32_t vec_size_kpe, uint32_t bdx,
          uint32_t bdy, uint32_t bdz, uint32_t tile_size_qo_heads, typename AttentionVariant,
          typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernelMLA(Params params) {
  auto block = cg::this_thread_block();
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const DTypeQ* q_nope = params.q_nope;
  const DTypeQ* q_pe = params.q_pe;
  DTypeO* o = params.o;
  float* lse = params.lse;
  const auto& paged_kv = params.paged_kv;
  const IdType* q_rope_offset = params.q_rope_offset;
  const bool* block_valid_mask = params.block_valid_mask;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const float rope_rcp_scale = params.rope_rcp_scale;
  const float rope_rcp_theta = params.rope_rcp_theta;
  const bool partition_kv = params.partition_kv;
  params.sm_scale *= math::log2e;

  constexpr uint32_t head_dim_ckv = bdx * vec_size_ckv;
  constexpr uint32_t head_dim_kpe = bdx * vec_size_kpe;
  const uint32_t batch_idx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  const uint32_t t_offset = dim3_offset(bdy, bdx, tz, ty, tx);

  // NOTE(Zihao): when CUDAGraph is enabled, we will launch more blocks than
  // the actual batch size, so we need to check if the current batch is valid
  if (block_valid_mask && !block_valid_mask[batch_idx]) return;
  const uint32_t mapped_batch_idx = params.request_indices[batch_idx];

  const uint32_t orig_seq_len = paged_kv.get_length(mapped_batch_idx);
  int32_t q_rope_offset_val =
      q_rope_offset == nullptr ? (orig_seq_len - 1) : q_rope_offset[mapped_batch_idx];

  const uint32_t kv_chunk_idx_in_orig_mapped_batch = params.kv_tile_indices[batch_idx];
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
  const uint32_t cur_chunk_start =
      partition_kv ? kv_chunk_idx_in_orig_mapped_batch * kv_chunk_size : 0;
  const uint32_t cur_chunk_end =
      partition_kv ? min((kv_chunk_idx_in_orig_mapped_batch + 1) * kv_chunk_size, orig_seq_len)
                   : orig_seq_len;
  const uint32_t cur_chunk_len = cur_chunk_end - cur_chunk_start;

  uint32_t packed_page_iter_base =
      paged_kv.indptr[mapped_batch_idx] * paged_kv.page_size + cur_chunk_start;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  constexpr uint32_t kv_iter_len = bdy * bdz;
  constexpr uint32_t compute_qk_tile = bdy;

  extern __attribute__((shared)) uint8_t smem[];
  DTypeKV* ckv_smem = (DTypeKV*)smem;
  DTypeKV* kpe_smem = (DTypeKV*)((uint8_t*)ckv_smem +
                                 num_stages_smem * kv_iter_len * head_dim_ckv * sizeof(DTypeKV));
  size_t* ckv_offset_smem = (size_t*)((uint8_t*)kpe_smem + num_stages_smem * kv_iter_len *
                                                               head_dim_kpe * sizeof(DTypeKV));
  size_t* kpe_offset_smem = (size_t*)((uint8_t*)ckv_offset_smem + bdx * bdy * bdz * sizeof(size_t));
  float* smem_md = (float*)ckv_offset_smem;

  AttentionVariant variant(params, batch_idx, smem);

  vec_t<float, vec_size_ckv> q_nope_vec[tile_size_qo_heads];
  vec_t<float, vec_size_kpe> q_pe_vec[tile_size_qo_heads];
  state_t<vec_size_ckv> st[tile_size_qo_heads];
  uint32_t qo_head_idx[tile_size_qo_heads];

  vec_t<float, vec_size_kpe> freq;

#pragma unroll
  for (uint32_t i = 0; i < vec_size_kpe; ++i) {
    freq[i] = rope_rcp_scale * __powf(rope_rcp_theta, float(2 * ((tx * vec_size_kpe + i) / 2)) /
                                                          float(head_dim_kpe));
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  // load q_nope and q_pe tile
#pragma unroll
  for (int i = 0; i < tile_size_qo_heads; ++i) {
    qo_head_idx[i] = dim3_offset(bdy, tile_size_qo_heads, blockIdx.y, threadIdx.y, i);
    if (qo_head_idx[i] < num_qo_heads) {
      q_nope_vec[i].cast_load(q_nope +
                              (mapped_batch_idx * num_qo_heads + qo_head_idx[i]) * head_dim_ckv +
                              tx * vec_size_ckv);
      q_pe_vec[i].cast_load(q_pe +
                            (mapped_batch_idx * num_qo_heads + qo_head_idx[i]) * head_dim_kpe +
                            tx * vec_size_kpe);
    }
  }

  // init paged-cache read offset to be used
  uint32_t q, r;
  paged_kv.page_size.divmod(packed_page_iter_base + t_offset, q, r);
  ckv_offset_smem[t_offset] = paged_kv.protective_get_offset_ckv(q, r, /*feat_idx*/ 0, last_indptr);
  kpe_offset_smem[t_offset] = paged_kv.protective_get_offset_kpe(q, r, /*feat_idx*/ 0, last_indptr);
  block.sync();

  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size_ckv * 8;
  constexpr uint32_t tx_fold = vec_size_ckv / vec_size_kpe;
  static_assert(num_stages_smem <= bdx);
  size_t offset_bytes;
  bool is_valid_range;
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    is_valid_range = (iter * kv_iter_len + dim2_offset(bdy, tz, ty)) < cur_chunk_len;

    offset_bytes = ckv_offset_smem[dim3_offset(bdz, bdy, iter, tz, ty)] + tx * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        ckv_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_ckv +
            tx * vec_size_ckv,
        paged_kv.ckv_data + offset_bytes, is_valid_range);

    offset_bytes =
        kpe_offset_smem[dim3_offset(bdz, bdy, iter, tz, ty)] + tx / tx_fold * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        kpe_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_kpe +
            tx / tx_fold * vec_size_ckv,
        paged_kv.kpe_data + offset_bytes, is_valid_range);

    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }

#pragma unroll
  for (uint32_t iter = 0; iter < ceil_div(cur_chunk_len, kv_iter_len); ++iter) {
    cp_async::wait_group<1 * num_stages_smem - 1>();
    block.sync();
    const int32_t kv_idx_base =
        (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[mapped_batch_idx]) +
        cur_chunk_start + iter * kv_iter_len;
#pragma unroll
    for (int i = 0; i < tile_size_qo_heads; ++i) {
      compute_qk_and_update_local_stat_mla<vec_size_ckv, vec_size_kpe, bdx, compute_qk_tile>(
          params, variant, mapped_batch_idx,
          ckv_smem + (stage_idx * kv_iter_len + tz * compute_qk_tile) * head_dim_ckv, q_nope_vec[i],
          kpe_smem + (stage_idx * kv_iter_len + tz * compute_qk_tile) * head_dim_kpe, q_pe_vec[i],
          freq, kv_idx_base,
          /*iter_base*/ iter * kv_iter_len, /*iter_bound*/ cur_chunk_len, st[i]);
    }

    if ((iter + num_stages_smem) % bdx == 0) {
      uint32_t q, r;
      paged_kv.page_size.divmod(
          packed_page_iter_base + (iter + num_stages_smem) * kv_iter_len + t_offset, q, r);
      ckv_offset_smem[t_offset] =
          paged_kv.protective_get_offset_ckv(q, r, /*feat_idx*/ 0, last_indptr);
      kpe_offset_smem[t_offset] =
          paged_kv.protective_get_offset_kpe(q, r, /*feat_idx*/ 0, last_indptr);
    }
    block.sync();

    is_valid_range =
        ((iter + num_stages_smem) * kv_iter_len + dim2_offset(bdy, tz, ty)) < cur_chunk_len;
    offset_bytes = ckv_offset_smem[dim3_offset(bdz, bdy, (iter + num_stages_smem) % bdx, tz, ty)] +
                   tx * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        ckv_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_ckv +
            tx * vec_size_ckv,
        paged_kv.ckv_data + offset_bytes, is_valid_range);

    offset_bytes = kpe_offset_smem[dim3_offset(bdz, bdy, (iter + num_stages_smem) % bdx, tz, ty)] +
                   tx / tx_fold * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        kpe_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_kpe +
            tx / tx_fold * vec_size_ckv,
        paged_kv.kpe_data + offset_bytes, is_valid_range);
    cp_async::commit_group();

    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>();
  block.sync();

  if (bdz != 1) {
#pragma unroll
    for (int i = 0; i < tile_size_qo_heads; ++i) {
      if (qo_head_idx[i] < num_qo_heads)
        sync_state<vec_size_ckv, bdx, bdy, bdz>(variant, st[i], (float*)smem, smem_md, tx, ty, tz);
    }
  }

  if (tz == 0) {
#pragma unroll
    for (int i = 0; i < tile_size_qo_heads; ++i) {
      if (qo_head_idx[i] < num_qo_heads) {
        st[i].normalize();
        st[i].o.cast_store(o + (batch_idx * num_qo_heads + qo_head_idx[i]) * head_dim_ckv +
                           tx * vec_size_ckv);

        if (lse != nullptr) {
          lse[batch_idx * num_qo_heads + qo_head_idx[i]] = st[i].get_lse();
        }
      }
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename AttentionVariant, typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatchedMLA(Params params, typename Params::DTypeO* tmp_v,
                                                     float* tmp_s, bool enable_pdl,
                                                     cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t padded_batch_size = params.padded_batch_size;

  constexpr uint32_t vec_size_ckv = std::max(16UL / sizeof(DTypeKV), HEAD_DIM_CKV / 32UL);
  constexpr uint32_t bdx = HEAD_DIM_CKV / vec_size_ckv;
  constexpr uint32_t vec_size_kpe = HEAD_DIM_KPE / bdx;

  constexpr uint32_t bdy = 8;
  constexpr uint32_t tile_size_qo_heads = 2;
  constexpr uint32_t qo_heads_per_block = bdy * tile_size_qo_heads;
  constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
  constexpr uint32_t bdz = num_threads / (bdx * bdy);
  const uint32_t gdy = ceil_div(num_qo_heads, qo_heads_per_block);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
    const uint32_t smem_size =
        NUM_STAGES_SMEM * bdy * bdz * (HEAD_DIM_CKV + HEAD_DIM_KPE) * sizeof(DTypeKV) +
        std::max(num_threads * sizeof(size_t) * 2, 2 * bdy * bdz * sizeof(float));

    auto kernel =
        BatchDecodeWithPagedKVCacheKernelMLA<NUM_STAGES_SMEM, vec_size_ckv, vec_size_kpe, bdx, bdy,
                                             bdz, tile_size_qo_heads, AttentionVariant, Params>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 nblks(padded_batch_size, gdy);
    dim3 nthrs(bdx, bdy, bdz);

    // PDL launch config
    cudaLaunchAttribute attribute[1];
    cudaLaunchConfig_t config;
    if (enable_pdl) {
      attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute[0].val.programmaticStreamSerializationAllowed = 1;
      config.attrs = attribute;
      config.numAttrs = 1;
      config.gridDim = nblks;
      config.blockDim = nthrs;
      config.dynamicSmemBytes = smem_size;
      config.stream = stream;
    }

    if (tmp_v == nullptr) {
      // do not use partition-kv kernel
      params.partition_kv = false;
      if (enable_pdl) {
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
      } else {
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      }
    } else {
      // use partition-kv kernel
      params.partition_kv = true;
      auto o = params.o;
      auto lse = params.lse;
      params.o = tmp_v;
      params.lse = tmp_s;
      if (enable_pdl) {
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
      } else {
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      }
      FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
          tmp_v, tmp_s, params.o_indptr, o, lse, params.paged_kv.batch_size, nullptr, num_qo_heads,
          HEAD_DIM_CKV, enable_pdl, stream));
    }
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_CUH_
