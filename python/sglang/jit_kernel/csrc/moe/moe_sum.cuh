// Adapt from sgl-kernel/csrc/moe/moe_sum.cu
#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>

using tvm::ffi::TensorView;

namespace {

// ---------------------------------------------------------------------------
// moe_sum_kernel — one CTA per token, sums over topk expert outputs
// ---------------------------------------------------------------------------
template <typename T, int TOPK>
__global__ void moe_sum_kernel(
    T* __restrict__ out,           // [num_tokens, hidden_size]
    const T* __restrict__ input,   // [num_tokens, topk, hidden_size]
    const int hidden_size) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    T x = static_cast<T>(0);
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += __ldg(&input[token_idx * TOPK * hidden_size + k * hidden_size + idx]);
    }
    out[token_idx * hidden_size + idx] = x;
  }
}

// General fallback for topk not covered by static dispatch
template <typename T>
__global__ void moe_sum_kernel_general(
    T* __restrict__ out,
    const T* __restrict__ input,
    const int hidden_size,
    const int topk) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    T x = static_cast<T>(0);
#pragma unroll 1
    for (int k = 0; k < topk; ++k) {
      x += __ldg(&input[token_idx * topk * hidden_size + k * hidden_size + idx]);
    }
    out[token_idx * hidden_size + idx] = x;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host launcher (tvm-ffi interface)
// ---------------------------------------------------------------------------
template <typename T>
void moe_sum(TensorView input, TensorView output) {
  using namespace host;

  // --- Input validation ---
  RuntimeCheck(input.dim() == 3, "input must be 3-D [num_tokens, topk, hidden_size]");
  RuntimeCheck(output.dim() == 2, "output must be 2-D [num_tokens, hidden_size]");
  RuntimeCheck(input.shape()[0] == output.shape()[0], "num_tokens mismatch");
  RuntimeCheck(input.shape()[2] == output.shape()[1], "hidden_size mismatch");

  const int64_t num_tokens = output.shape()[0];
  const int hidden_size = static_cast<int>(input.shape()[2]);
  const int topk = static_cast<int>(input.shape()[1]);

  const T* in_ptr = static_cast<const T*>(input.data_ptr());
  T* out_ptr = static_cast<T*>(output.data_ptr());

  dim3 grid(static_cast<unsigned>(num_tokens));
  dim3 block(static_cast<unsigned>(std::min(hidden_size, 1024)));

  cudaStream_t stream = LaunchKernel::resolve_device(input.device());

  switch (topk) {
    case 1:
      moe_sum_kernel<T, 1><<<grid, block, 0, stream>>>(out_ptr, in_ptr, hidden_size);
      break;
    case 2:
      moe_sum_kernel<T, 2><<<grid, block, 0, stream>>>(out_ptr, in_ptr, hidden_size);
      break;
    case 3:
      moe_sum_kernel<T, 3><<<grid, block, 0, stream>>>(out_ptr, in_ptr, hidden_size);
      break;
    case 4:
      moe_sum_kernel<T, 4><<<grid, block, 0, stream>>>(out_ptr, in_ptr, hidden_size);
      break;
    case 5:
      moe_sum_kernel<T, 5><<<grid, block, 0, stream>>>(out_ptr, in_ptr, hidden_size);
      break;
    case 6:
      moe_sum_kernel<T, 6><<<grid, block, 0, stream>>>(out_ptr, in_ptr, hidden_size);
      break;
    case 7:
      moe_sum_kernel<T, 7><<<grid, block, 0, stream>>>(out_ptr, in_ptr, hidden_size);
      break;
    case 8:
      moe_sum_kernel<T, 8><<<grid, block, 0, stream>>>(out_ptr, in_ptr, hidden_size);
      break;
    case 9:
      moe_sum_kernel<T, 9><<<grid, block, 0, stream>>>(out_ptr, in_ptr, hidden_size);
      break;
    default:
      moe_sum_kernel_general<T><<<grid, block, 0, stream>>>(out_ptr, in_ptr, hidden_size, topk);
  }
}
