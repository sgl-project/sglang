// This is only a plugin used for flashinfer 0.1.6. The new version does not need it.
/*
 * Copyright (c) 2025 by SGLang team.
 * Copyright (c) 2025 by FlashInfer team.
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

#include <flashinfer/quantization.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

// bitorder = "little"
void segment_packbits(
    at::Tensor x, at::Tensor input_indptr, at::Tensor output_indptr, at::Tensor y, int64_t cuda_stream) {
  CHECK_INPUT(x);
  CHECK_INPUT(input_indptr);
  CHECK_INPUT(output_indptr);
  auto device = x.device();
  CHECK_EQ(input_indptr.device(), device);
  CHECK_EQ(output_indptr.device(), device);
  CHECK_EQ(y.device(), device);
  unsigned int batch_size = input_indptr.size(0) - 1;
  CHECK_EQ(output_indptr.size(0), batch_size + 1);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = quantization::SegmentPackBits(
      static_cast<bool*>(x.data_ptr()),
      static_cast<uint8_t*>(y.data_ptr()),
      static_cast<int32_t*>(input_indptr.data_ptr()),
      static_cast<int32_t*>(output_indptr.data_ptr()),
      batch_size,
      quantization::BitOrder::kLittle,
      stream);
}
