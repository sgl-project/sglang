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
// moe_sum_kernel — one CTA per token-slice, sums over topk expert outputs
// ---------------------------------------------------------------------------
// Grid is (num_tokens * hidden_blocks,).  hidden_blocks is chosen adaptively:
//   hidden_blocks = min(ceil(hidden_size/blockDim.x), max(1, TARGET_MAX_BLOCKS/num_tokens))
// This keeps the total grid size near TARGET_MAX_BLOCKS, giving 2-D parallelism
// for small token counts (e.g. tokens=1) while avoiding over-decomposition for
// large counts.  Each thread strides over its slice so no work is dropped.
template <typename T, int TOPK>
__global__ void moe_sum_kernel(
    T* __restrict__ out,          // [num_tokens, hidden_size]
    const T* __restrict__ input,  // [num_tokens, topk, hidden_size]
    const int hidden_size,
    const int hidden_blocks) {
  const int h_block = blockIdx.x % hidden_blocks;
  const int64_t token_idx = blockIdx.x / hidden_blocks;
  const int64_t token_base = token_idx * TOPK * hidden_size;
  const int64_t out_base = token_idx * hidden_size;
  const int stride = hidden_blocks * static_cast<int>(blockDim.x);

  for (int idx = h_block * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
       idx < hidden_size; idx += stride) {
    float x = 0.0f;  // accumulate in float32 to match at::sum_out precision
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += static_cast<float>(__ldg(&input[token_base + k * hidden_size + idx]));
    }
    out[out_base + idx] = static_cast<T>(x);
  }
}

// General fallback for topk not covered by static dispatch
template <typename T>
__global__ void moe_sum_kernel_general(
    T* __restrict__ out,
    const T* __restrict__ input,
    const int hidden_size,
    const int topk,
    const int hidden_blocks) {
  const int h_block = blockIdx.x % hidden_blocks;
  const int64_t token_idx = blockIdx.x / hidden_blocks;
  const int64_t token_base = token_idx * topk * hidden_size;
  const int64_t out_base = token_idx * hidden_size;
  const int stride = hidden_blocks * static_cast<int>(blockDim.x);

  for (int idx = h_block * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
       idx < hidden_size; idx += stride) {
    float x = 0.0f;  // accumulate in float32 to match at::sum_out precision
#pragma unroll 1
    for (int k = 0; k < topk; ++k) {
      x += static_cast<float>(__ldg(&input[token_base + k * hidden_size + idx]));
    }
    out[out_base + idx] = static_cast<T>(x);
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

  constexpr int block_size = 1024;
  const int hidden_blocks_full = (hidden_size + block_size - 1) / block_size;
  // Cap hidden_blocks so total grid stays near TARGET_MAX_BLOCKS.  For large token
  // counts this forces hidden_blocks→1 and each thread strides over the full slice,
  // providing the same occupancy as the original 1-D kernel.  For small token counts
  // (e.g. tokens=1) we keep full decomposition for maximum parallelism.
  constexpr int TARGET_MAX_BLOCKS = 256;
  const int hidden_blocks =
      std::min(hidden_blocks_full, std::max(1, TARGET_MAX_BLOCKS / static_cast<int>(num_tokens)));
  dim3 grid(static_cast<unsigned>(num_tokens) * static_cast<unsigned>(hidden_blocks));
  dim3 block(static_cast<unsigned>(std::min(hidden_size, block_size)));

  LaunchKernel launcher(grid, block, input.device());

#define LAUNCH(TOPK) \
  launcher(moe_sum_kernel<T, TOPK>, out_ptr, in_ptr, hidden_size, hidden_blocks)

  switch (topk) {
    case 1: LAUNCH(1); break;
    case 2: LAUNCH(2); break;
    case 3: LAUNCH(3); break;
    case 4: LAUNCH(4); break;
    case 5: LAUNCH(5); break;
    case 6: LAUNCH(6); break;
    case 7: LAUNCH(7); break;
    case 8: LAUNCH(8); break;
    case 9: LAUNCH(9); break;
    default:
      launcher(moe_sum_kernel_general<T>, out_ptr, in_ptr, hidden_size, topk, hidden_blocks);
  }

#undef LAUNCH
}
