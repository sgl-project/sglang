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

#include <cstdint>
#include <cuda_runtime.h>

namespace sglang {

struct Sm90Fp4GroupedIndexerParams {
  int batch_size;
  int width;
  int group_size;
  int page_size;
  int ratio;

  const void* q;
  int64_t q_stride_b;
  int64_t q_stride_h;
  const void* q_scale;
  int64_t q_scale_stride_b;

  const void* weights;
  int64_t weight_stride_b;

  const void* req_to_token;
  int64_t req_stride;
  const void* req;
  const void* lens;

  const void* table;
  int64_t table_stride;

  void* out;
  int64_t out_stride;
  cudaStream_t stream;
};

}  // namespace sglang
