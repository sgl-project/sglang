/*
 * Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (C) 2024-2026, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Adapted from AITER custom_all_reduce.cuh: split-H, native HIP post rounding.
#include "aiter_enum.h"
#include "aiter_stream.h"
#include "rocm_ops.hpp"
#include <custom_all_reduce.cuh>

namespace aiter {
static_assert(8 * 10 <= kMaxBlocks, "mHC launch exceeds AITER signal slots");
__global__ void dsv41_allreduce_mhc_post_kernel(
    RankData* peers,
    RankSignals signals,
    Signal* local_signals,
    int rank,
    opus::bf16_t* output,
    const opus::bf16_t* residual,
    const float* post,
    const float* comb) {
  using P = opus::vector_t<opus::bf16_t, 8>;
  using A = opus::vector_t<opus::fp32_t, 8>;
  const int row = blockIdx.x / 10;
  const int pack = (blockIdx.x % 10) * 64 + threadIdx.x;
  const P* inputs[4];
#pragma unroll
  for (int r = 0; r < 4; ++r)
    inputs[r] = reinterpret_cast<const P*>(peers->ptrs[r]);
  start_sync<4>(signals, local_signals, rank);
  const P reduced = packed_reduce<P, 4, A>(inputs, row * 640 + pack);
#pragma unroll
  for (int out_h = 0; out_h < 4; ++out_h) {
    A mixed;
#pragma unroll
    for (int v = 0; v < 8; ++v)
      mixed[v] = upcast_s(reduced[v]) * post[row * 4 + out_h];
#pragma unroll
    for (int in_h = 0; in_h < 4; ++in_h) {
      const P values = reinterpret_cast<const P*>(residual)[(row * 4 + in_h) * 640 + pack];
      const float coefficient = comb[row * 16 + in_h * 4 + out_h];
#pragma unroll
      for (int v = 0; v < 8; ++v)
        mixed[v] += upcast_s(values[v]) * coefficient;
    }
    reinterpret_cast<P*>(output)[(row * 4 + out_h) * 640 + pack] = downcast<P>(mixed);
  }
  end_sync<4, true>(signals, local_signals, rank);
}

void dsv41_allreduce_mhc_post(
    int64_t ptr,
    aiter_tensor_t& input,
    aiter_tensor_t& output,
    aiter_tensor_t& residual,
    aiter_tensor_t& post,
    aiter_tensor_t& comb,
    int64_t registered_ptr,
    int64_t registered_bytes) {
  HipDeviceGuard guard(input.device_id);
  auto stream = aiter::getCurrentHIPStream();
  auto* comm = reinterpret_cast<CustomAllreduce*>(ptr);
  const int rows = input.size(0);
  if (comm->world_size_ != 4 || rows < 1 || rows > 8 || input.size(-1) != 5120 || input.dtype() != AITER_DTYPE_bf16 ||
      residual.dtype() != AITER_DTYPE_bf16 || output.dtype() != AITER_DTYPE_bf16 || post.dtype() != AITER_DTYPE_fp32 ||
      comb.dtype() != AITER_DTYPE_fp32) {
    throw std::runtime_error("DSV4.1 AR+post requires TP4, 1-8 rows, H5120 and BF16/FP32 operands");
  }
  void* values = input.data_ptr();
  if (registered_ptr) {
    const int64_t bytes = input.numel() * input.element_size();
    if (bytes > registered_bytes) throw std::runtime_error("AR input pool is too small");
    values = reinterpret_cast<void*>(registered_ptr);
    HIP_CALL(hipMemcpyAsync(values, input.data_ptr(), bytes, hipMemcpyDeviceToDevice, stream));
  }
  auto* peers = comm->get_buffer_RD(stream, values);
  dsv41_allreduce_mhc_post_kernel<<<rows * 10, 64, 0, stream>>>(
      peers,
      comm->sg_,
      comm->self_sg_,
      comm->rank_,
      static_cast<opus::bf16_t*>(output.data_ptr()),
      static_cast<opus::bf16_t*>(residual.data_ptr()),
      static_cast<float*>(post.data_ptr()),
      static_cast<float*>(comb.data_ptr()));
}
}  // namespace aiter

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  AITER_SET_STREAM_PYBIND;
  m.def(
      "run",
      &aiter::dsv41_allreduce_mhc_post,
      py::arg("ptr"),
      py::arg("input"),
      py::arg("output"),
      py::arg("residual"),
      py::arg("post"),
      py::arg("comb"),
      py::arg("registered_ptr"),
      py::arg("registered_bytes"));
}
