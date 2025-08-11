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
// #include <flashinfer/pos_enc.cuh>

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
    int64_t cuda_stream,
    const std::optional<at::Tensor>& v,
    const std::optional<at::Tensor>& k_buffer,
    const std::optional<at::Tensor>& v_buffer,
    const std::optional<at::Tensor>& kv_cache_loc) {
  CHECK_LAST_DIM_CONTIGUOUS(q);
  CHECK_LAST_DIM_CONTIGUOUS(k);
  if (save_kv_cache) {
    CHECK_LAST_DIM_CONTIGUOUS(v);
    CHECK_LAST_DIM_CONTIGUOUS(k_buffer);
    CHECK_LAST_DIM_CONTIGUOUS(v_buffer);
    CHECK_DIM(3, k_buffer);   // k_buffer: (nnz, H_K, D)
    CHECK_DIM(3, v_buffer);   // v_buffer: (nnz, H_V, D)
    CHECK_DIM(3, v);          // v: (nnz, H_V, D)
    CHECK_DIM(1, cache_loc);  // v: (n)
    CHECK_INPUT(cache_loc);
    size_t k_buffer_stride_n = k_buffer.stride(0);
    size_t k_buffer_stride_h = k_buffer.stride(1);
    size_t v_buffer_stride_n = v_buffer.stride(0);
    size_t v_buffer_stride_h = v_buffer.stride(1);
    size_t v_stride_n = v.stride(0);
    size_t v_stride_h = v.stride(1);
    auto v_ptr = static_cast<c_type*>(v.data_ptr());
    auto k_buffer_ptr = static_cast<c_type*>(k_buffer.data_ptr());
    auto v_buffer_ptr = static_cast<c_type*>(v_buffer.data_ptr());
    auto cache_loc_ptr = static_cast<int64_t*>(cache_loc.data_ptr());
  } else {
    size_t k_buffer_stride_n = 0;
    size_t k_buffer_stride_h = 0;
    size_t v_buffer_stride_n = 0;
    size_t v_buffer_stride_h = 0;
    size_t v_stride_n = 0;
    size_t v_stride_h = 0;
    auto v_ptr = nullptr;
    auto k_buffer_ptr = nullptr;
    auto v_buffer_ptr = nullptr;
    auto cache_loc_ptr = nullptr;
  }

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

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    cudaError_t status = BatchQKApplyRotaryPosIdsCosSinCacheWithSetKVBuffer(
        static_cast<c_type*>(q.data_ptr()),
        static_cast<c_type*>(k.data_ptr()),
        v_ptr,
        static_cast<c_type*>(q_rope.data_ptr()),
        static_cast<c_type*>(k_rope.data_ptr()),
        k_buffer_ptr,
        v_buffer_ptr,
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
        cache_loc_ptr,
        interleave,
        stream,
        k_scale,
        v_scale,
        save_kv_cache);
    TORCH_CHECK(
        status == cudaSuccess,
        "BatchQKApplyRotaryPosIdsCosSinCacheWithSetKVBuffer failed with error code " +
            std::string(cudaGetErrorString(status)));
    return true;
  });
}
