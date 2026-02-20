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
// Adapted from
// https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/speculative/packbit.cu

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <flashinfer/quantization.cuh>
#include <tvm/ffi/container/tensor.h>

namespace {

// ---------------------------------------------------------------------------
// tvm-ffi entry point
// ---------------------------------------------------------------------------

// x:             [sum(input_indptr)] bool — input bits (little-endian)
// input_indptr:  [batch_size + 1] int32 — segment start offsets for input
// output_indptr: [batch_size + 1] int32 — segment start offsets for output
// y:             [sum(output_indptr)] uint8 — packed output
// batch_size:    number of segments
void segment_packbits(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView input_indptr,
    tvm::ffi::TensorView output_indptr,
    tvm::ffi::TensorView y,
    int64_t batch_size) {
  using namespace host;

  RuntimeCheck(x.device().device_type == kDLCUDA, "x must be a CUDA tensor");
  RuntimeCheck(x.ndim() == 1, "x must be 1D");
  RuntimeCheck(x.is_contiguous(), "x must be contiguous");
  RuntimeCheck(host::dtype_bytes(x.dtype()) == 1, "x element size must be 1 byte (bool or uint8)");

  RuntimeCheck(input_indptr.ndim() == 1, "input_indptr must be 1D");
  RuntimeCheck(input_indptr.is_contiguous(), "input_indptr must be contiguous");
  RuntimeCheck(
      input_indptr.dtype().code == kDLInt && input_indptr.dtype().bits == 32,
      "input_indptr must be int32");
  RuntimeCheck(input_indptr.size(0) >= batch_size + 1, "input_indptr size must be >= batch_size + 1");

  RuntimeCheck(output_indptr.ndim() == 1, "output_indptr must be 1D");
  RuntimeCheck(output_indptr.is_contiguous(), "output_indptr must be contiguous");
  RuntimeCheck(
      output_indptr.dtype().code == kDLInt && output_indptr.dtype().bits == 32,
      "output_indptr must be int32");
  RuntimeCheck(output_indptr.size(0) >= batch_size + 1, "output_indptr size must be >= batch_size + 1");

  RuntimeCheck(y.ndim() == 1, "y must be 1D");
  RuntimeCheck(y.is_contiguous(), "y must be contiguous");
  RuntimeCheck(
      y.dtype().code == kDLUInt && y.dtype().bits == 8, "y must be uint8");

  cudaStream_t stream = LaunchKernel::resolve_device(x.device());
  cudaError_t status = flashinfer::quantization::SegmentPackBits(
      static_cast<bool*>(x.data_ptr()),
      static_cast<uint8_t*>(y.data_ptr()),
      static_cast<int32_t*>(input_indptr.data_ptr()),
      static_cast<int32_t*>(output_indptr.data_ptr()),
      static_cast<uint32_t>(batch_size),
      flashinfer::quantization::BitOrder::kLittle,
      stream);

  RuntimeCheck(status == cudaSuccess, "segment_packbits failed: ", cudaGetErrorString(status));
}

}  // namespace
