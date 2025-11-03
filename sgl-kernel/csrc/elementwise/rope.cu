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

#include "pos_enc.cuh"
#include "pytorch_extension_utils.h"

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
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
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
