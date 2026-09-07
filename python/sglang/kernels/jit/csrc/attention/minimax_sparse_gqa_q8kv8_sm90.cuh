#pragma once

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "fp8_mha_fwd_q8kv8_sm90.cuh"
#include <cstdint>
#include <cuda_runtime.h>

namespace sglang {

void minimax_sparse_gqa_q8kv8_sm90(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k_cache,
    tvm::ffi::TensorView v_cache,
    tvm::ffi::TensorView req_to_token,
    tvm::ffi::TensorView slot_ids,
    tvm::ffi::TensorView topk_idx,
    tvm::ffi::TensorView cu_seqlens,
    tvm::ffi::TensorView seq_lens,
    tvm::ffi::TensorView prefix_lens,
    int64_t total_q,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t max_slots,
    int64_t req_stride,
    int64_t topk,
    int64_t batch_size,
    int64_t q_stride_0,
    int64_t q_stride_1,
    int64_t q_stride_2,
    double sm_scale,
    double q_scale,
    double k_scale,
    double v_scale,
    int64_t cuda_stream) {
  const DLDevice device = q.device();
  cudaSetDevice(device.device_id);
  auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  q8kv8_sm90::launch_fp8_mha_fwd_q8kv8_sm90(
      static_cast<q8kv8_sm90::bf16_t*>(output.data_ptr()),
      static_cast<const q8kv8_sm90::fp8_t*>(q.data_ptr()),
      static_cast<const q8kv8_sm90::fp8_t*>(k_cache.data_ptr()),
      static_cast<const q8kv8_sm90::fp8_t*>(v_cache.data_ptr()),
      static_cast<const int32_t*>(req_to_token.data_ptr()),
      static_cast<const int64_t*>(slot_ids.data_ptr()),
      static_cast<const int32_t*>(topk_idx.data_ptr()),
      static_cast<const int32_t*>(cu_seqlens.data_ptr()),
      static_cast<const int32_t*>(seq_lens.data_ptr()),
      static_cast<const int32_t*>(prefix_lens.data_ptr()),
      static_cast<int>(total_q),
      static_cast<int>(num_q_heads),
      static_cast<int>(num_kv_heads),
      static_cast<int>(max_slots),
      static_cast<int>(req_stride),
      static_cast<int>(topk),
      static_cast<int>(batch_size),
      q_stride_0,
      q_stride_1,
      q_stride_2,
      static_cast<float>(sm_scale * q_scale * k_scale),
      static_cast<float>(v_scale),
      stream);
  host::RuntimeDeviceCheck();
}

}  // namespace sglang
