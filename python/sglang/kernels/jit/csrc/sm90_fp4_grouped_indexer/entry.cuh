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

#pragma once

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "kernel.cuh"
#include <cstdint>
#include <cuda_runtime.h>

namespace sglang {

void sm90_fp4_grouped_indexer_dispatch(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView q_scale,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView req_to_token,
    tvm::ffi::TensorView req,
    tvm::ffi::TensorView lens,
    tvm::ffi::TensorView table,
    tvm::ffi::TensorView out,
    int64_t batch_size,
    int64_t width,
    int64_t group_size,
    int64_t page_size,
    int64_t ratio,
    int64_t q_stride_b,
    int64_t q_stride_h,
    int64_t q_scale_stride_b,
    int64_t weight_stride_b,
    int64_t req_stride,
    int64_t table_stride,
    int64_t out_stride,
    int64_t cuda_stream) {
  Sm90Fp4GroupedIndexerParams params;
  params.batch_size = static_cast<int>(batch_size);
  params.width = static_cast<int>(width);
  params.group_size = static_cast<int>(group_size);
  params.page_size = static_cast<int>(page_size);
  params.ratio = static_cast<int>(ratio);
  params.q = q.data_ptr();
  params.q_stride_b = q_stride_b;
  params.q_stride_h = q_stride_h;
  params.q_scale = q_scale.data_ptr();
  params.q_scale_stride_b = q_scale_stride_b;
  params.weights = weights.data_ptr();
  params.weight_stride_b = weight_stride_b;
  params.req_to_token = req_to_token.data_ptr();
  params.req_stride = req_stride;
  params.req = req.data_ptr();
  params.lens = lens.data_ptr();
  params.table = table.data_ptr();
  params.table_stride = table_stride;
  params.out = out.data_ptr();
  params.out_stride = out_stride;
  params.stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  DLDevice dev = q.device();
  cudaSetDevice(dev.device_id);
  fp4_grouped_indexer_sm90::run_sm90_fp4_grouped_indexer(params);
}

}  // namespace sglang
