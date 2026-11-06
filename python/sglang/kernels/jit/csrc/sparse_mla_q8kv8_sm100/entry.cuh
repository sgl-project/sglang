/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// JIT dispatch entry for the SM100 Q8KV8 sparse MLA kernel.
#pragma once

#include <sgl_kernel/runtime.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "kernel.cuh"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace sglang {

/**
 * \brief Run the SM100 Q8KV8 sparse MLA kernel, plus the split merge when `num_splits > 1`.
 *
 * The Python wrapper checks shapes and dtypes and allocates the outputs. Absent optional inputs are
 * zero-size tensors, flagged by `have_attn_sink` and `have_topk_length`.
 *
 * \param q                  [s_q, h_q, 576] e4m3, last dim contiguous
 * \param kv                 [s_kv, 576] e4m3 token rows, last dim contiguous
 * \param indices            [s_q, topk] int32 rows of `kv`; entries < 0 or >= s_kv are masked
 * \param attn_sink          [h_q] float32, or empty
 * \param topk_length        [s_q] int32, or empty
 * \param out                [s_q, h_q, 512] bf16
 * \param max_logits         [s_q, h_q] float32
 * \param lse                [s_q, h_q] float32, natural log without the sink
 * \param partial_out        [num_splits, s_q, h_q, 512] bf16, or empty when num_splits == 1
 * \param partial_max_logits [num_splits, s_q, h_q] float32, or empty when num_splits == 1
 * \param partial_lse        [num_splits, s_q, h_q] float32, or empty when num_splits == 1
 * \param bmm1_scale         q_scale * kv_scale * sm_scale
 * \param bmm2_scale         scale of V
 */
void sparse_mla_q8kv8_sm100_dispatch(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView kv,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView attn_sink,
    tvm::ffi::TensorView topk_length,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView max_logits,
    tvm::ffi::TensorView lse,
    tvm::ffi::TensorView partial_out,
    tvm::ffi::TensorView partial_max_logits,
    tvm::ffi::TensorView partial_lse,
    int64_t s_q,
    int64_t s_kv,
    int64_t h_q,
    int64_t topk,
    int64_t num_splits,
    int64_t stride_q_s_q,
    int64_t stride_q_h_q,
    int64_t stride_kv_s_kv,
    int64_t stride_indices_s_q,
    double bmm1_scale,
    double bmm2_scale,
    int64_t have_attn_sink,
    int64_t have_topk_length) {
  using q8kv8_sm100::run_sparse_mla_q8kv8_sm100;

  q8kv8_sm100::SparseMlaQ8Kv8Sm100Params p;
  p.s_q = static_cast<int32_t>(s_q);
  p.s_kv = static_cast<int32_t>(s_kv);
  p.h_q = static_cast<int32_t>(h_q);
  p.topk = static_cast<int32_t>(topk);
  p.num_splits = static_cast<int32_t>(num_splits);
  p.split_len = static_cast<int32_t>(topk / num_splits);
  p.scale_log2 = static_cast<float>(bmm1_scale * M_LOG2E);
  p.bmm2_scale = static_cast<float>(bmm2_scale);

  p.q = static_cast<const uint8_t*>(q.data_ptr());
  p.kv = static_cast<const uint8_t*>(kv.data_ptr());
  p.indices = static_cast<const int32_t*>(indices.data_ptr());
  p.attn_sink = have_attn_sink ? static_cast<const float*>(attn_sink.data_ptr()) : nullptr;
  p.topk_length = have_topk_length ? static_cast<const int32_t*>(topk_length.data_ptr()) : nullptr;

  p.stride_q_s_q = stride_q_s_q;
  p.stride_q_h_q = stride_q_h_q;
  p.stride_kv_s_kv = stride_kv_s_kv;
  p.stride_indices_s_q = stride_indices_s_q;

  if (num_splits > 1) {
    // The main kernel writes the partials, the merge kernel the final outputs.
    p.out = static_cast<cutlass::bfloat16_t*>(partial_out.data_ptr());
    p.max_logits = static_cast<float*>(partial_max_logits.data_ptr());
    p.lse = static_cast<float*>(partial_lse.data_ptr());
    p.merged_out = static_cast<cutlass::bfloat16_t*>(out.data_ptr());
    p.merged_max_logits = static_cast<float*>(max_logits.data_ptr());
    p.merged_lse = static_cast<float*>(lse.data_ptr());
  } else {
    p.out = static_cast<cutlass::bfloat16_t*>(out.data_ptr());
    p.max_logits = static_cast<float*>(max_logits.data_ptr());
    p.lse = static_cast<float*>(lse.data_ptr());
    p.merged_out = nullptr;
    p.merged_max_logits = nullptr;
    p.merged_lse = nullptr;
  }

  const int32_t device_id = q.device().device_id;
  CHECK_CUDA(cudaSetDevice(device_id));
  p.stream = host::LaunchKernel::resolve_device(q.device());
  const auto num_sms = static_cast<int32_t>(host::runtime::get_sm_count(device_id));

  if (p.topk_length != nullptr) {
    if (p.attn_sink != nullptr) {
      run_sparse_mla_q8kv8_sm100<true, true>(p, num_sms);
    } else {
      run_sparse_mla_q8kv8_sm100<true, false>(p, num_sms);
    }
  } else {
    if (p.attn_sink != nullptr) {
      run_sparse_mla_q8kv8_sm100<false, true>(p, num_sms);
    } else {
      run_sparse_mla_q8kv8_sm100<false, false>(p, num_sms);
    }
  }
}

}  // namespace sglang
