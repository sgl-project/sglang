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
#include <flashinfer/pos_enc.cuh>

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
) {
  CHECK_LAST_DIM_CONTIGUOUS(q);
  CHECK_LAST_DIM_CONTIGUOUS(k);

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
  cudaStream_t alt_stream = reinterpret_cast<cudaStream_t>(alt_stream_ptr);
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
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
        stream,
        static_cast<c_type*>(k_buffer_ptr.data_ptr()),
        static_cast<c_type*>(v_buffer_ptr.data_ptr()),
        k_scale,
        v_scale,
        static_cast<c_type*>(v.data_ptr()),
        is_capture_mode,
        alt_stream,

    );
    TORCH_CHECK(
        status == cudaSuccess,
        "BatchQKApplyRotaryPosIdsCosSinCache failed with error code " + std::string(cudaGetErrorString(status)));
    return true;
  });
}



void apply_rope_pos_ids_cos_sin_cache_with_set_kv_buffer(
    at::Tensor q,
    at::Tensor k,
    at::Tensor q_rope,
    at::Tensor k_rope,
    at::Tensor cos_sin_cache,
    at::Tensor pos_ids,
    bool interleave,
    int64_t cuda_stream,
    at::Tensor k_buffer_ptr,
    at::Tensor v_buffer_ptr,
    float k_scale,
    float v_scale,
    at::Tensor v,
    bool is_capture_mode,
    int64_t alt_stream_ptr  // Additional stream for overlap
) {
  CHECK_LAST_DIM_CONTIGUOUS(q);
  CHECK_LAST_DIM_CONTIGUOUS(k);
  CHECK_LAST_DIM_CONTIGUOUS(v);
  CHECK_LAST_DIM_CONTIGUOUS(k_buffer_ptr);
  CHECK_LAST_DIM_CONTIGUOUS(v_buffer_ptr);
  CHECK_DIM(3, k_buffer_ptr);  // k_buffer: (nnz, H_K, D)
  CHECK_DIM(3, v_buffer_ptr);  // v_buffer: (nnz, H_V, D)
  CHECK_DIM(3, v);             // v: (nnz, H_V, D)
  

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
  size_t v_stride_n = v.stride(0);
  size_t v_stride_h = v.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);

  size_t k_buffer_stride_n = k_buffer_ptr.stride(0);
  size_t k_buffer_stride_h = k_buffer_ptr.stride(1);
  size_t v_buffer_stride_n = v_buffer_ptr.stride(0);
  size_t v_buffer_stride_h = v_buffer_ptr.stride(1);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaStream_t alt_stream = reinterpret_cast<cudaStream_t>(alt_stream_ptr);
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    cudaError_t status = BatchQKApplyRotaryPosIdsCosSinCacheWithSetKVBuffer(
        static_cast<c_type*>(q.data_ptr()),
        static_cast<c_type*>(k.data_ptr()),
        static_cast<c_type*>(v.data_ptr()),
        static_cast<c_type*>(q_rope.data_ptr()),
        static_cast<c_type*>(k_rope.data_ptr()),
        static_cast<c_type*>(k_buffer_ptr.data_ptr()),
        static_cast<c_type*>(v_buffer_ptr.data_ptr()),
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
        interleave,
        stream,
        k_scale,
        v_scale,
        is_capture_mode,
        alt_stream,
    );
    TORCH_CHECK(
        status == cudaSuccess,
        "BatchQKApplyRotaryPosIdsCosSinCacheWithSetKVBuffer failed with error code " + std::string(cudaGetErrorString(status)));
    return true;
  });


  // // Handle dtype and scaling
  // if (k_rope.scalar_type() != k_buffer_ptr.scalar_type()) {
  //   if (k_scale != 1.0f) {
  //     k_rope.div_(k_scale);
  //   }
  //   if (v_scale != 1.0f) {
  //     v.div_(v_scale);
  //   }
  //   // Convert to buffer dtype
  //   k_rope = k_rope.to(k_buffer_ptr.scalar_type());
  //   v = v.to(v_buffer_ptr.scalar_type());
  // }

  // if (is_capture_mode && alt_stream_ptr != 0) {
  //   cudaStream_t alt_stream = reinterpret_cast<cudaStream_t>(alt_stream_ptr);
  //   cudaStream_t main_stream = stream;

  //   // Wait for main stream to complete RoPE
  //   // Create event for synchronization
  //   cudaEvent_t event;
  //   cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

  //   // Record event on main stream after RoPE completion
  //   cudaEventRecord(event, main_stream);

  //   cudaStreamWaitEvent(alt_stream, event, 0);

  //   // Copy K on main stream
  //   k_buffer_ptr.copy_(k_rope, /*non_blocking=*/true);

  //   // Copy V on alternate stream
  //   at::cuda::CUDAStreamGuard guard(at::cuda::getStreamFromExternal(alt_stream, device.index()));
  //   v_buffer_ptr.copy_(v, /*non_blocking=*/true);

  //   // Record event on alt stream after V copy
  //   cudaEventRecord(event, alt_stream);
  //   // Main stream waits for alt stream
  //   cudaStreamWaitEvent(main_stream, event, 0);
  //   // Clean up
  //   cudaEventDestroy(event);
  // } else {
  //   // Synchronous copy
  //   k_buffer_ptr.copy_(k_rope, /*non_blocking=*/true);
  //   v_buffer_ptr.copy_(v, /*non_blocking=*/true);
  // }
  
}
