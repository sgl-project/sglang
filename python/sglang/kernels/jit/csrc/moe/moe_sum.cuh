#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <climits>
#include <cstdint>

namespace sglang {

/**
 * \brief Sum the selected expert outputs for each token.
 *
 * \tparam T Element type.
 * \tparam kTopK Number of expert outputs per token.
 * \param out Output buffer with shape [num_tokens, hidden_size].
 * \param input Input buffer with shape [num_tokens, kTopK, hidden_size].
 * \param hidden_size Number of elements in each expert output.
 */
template <typename T, int kTopK>
__global__ void moe_sum_kernel(T* __restrict__ out, const T* __restrict__ input, int32_t hidden_size) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    T x = 0.0;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      x += __ldg(&input[token_idx * kTopK * hidden_size + k * hidden_size + idx]);
    }
    out[token_idx * hidden_size + idx] = x;
  }
}

/**
 * \brief Validate tensors and launch the MoE sum kernel.
 *
 * \tparam T Element type: fp16_t, bf16_t, or fp32_t.
 * \param input Input tensor [num_tokens, topk, hidden_size].
 * \param output Output tensor [num_tokens, hidden_size].
 */
template <typename T>
void moe_sum(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
  using namespace host;
  auto num_tokens = SymbolicSize{"num_tokens"};
  auto topk = SymbolicSize{"topk"};
  auto hidden_size = SymbolicSize{"hidden_size"};
  auto device = SymbolicDevice{};
  TensorMatcher({num_tokens, topk, hidden_size}).with_dtype<T>().template with_device<kDLCUDA>(device).verify(input);
  TensorMatcher({num_tokens, hidden_size}).with_dtype<T>().template with_device<kDLCUDA>(device).verify(output);

  const int64_t num_tokens_value = num_tokens.unwrap();
  const int64_t hidden_size_value = hidden_size.unwrap();
  CHECK_HOST(hidden_size_value <= INT32_MAX) << "hidden_size exceeds int32 range";
  if (num_tokens_value == 0 || hidden_size_value == 0) return;

  const dim3 grid(static_cast<uint32_t>(num_tokens_value));
  const dim3 block(static_cast<uint32_t>(std::min<int64_t>(hidden_size_value, 1024)));
  const auto* input_ptr = static_cast<const T*>(input.data_ptr());
  auto* output_ptr = static_cast<T*>(output.data_ptr());
  const int32_t hidden = static_cast<int32_t>(hidden_size_value);

  switch (topk.unwrap()) {
    case 2:
      LaunchKernel(grid, block, device.unwrap())(moe_sum_kernel<T, 2>, output_ptr, input_ptr, hidden);
      break;
    case 3:
      LaunchKernel(grid, block, device.unwrap())(moe_sum_kernel<T, 3>, output_ptr, input_ptr, hidden);
      break;
    case 4:
      LaunchKernel(grid, block, device.unwrap())(moe_sum_kernel<T, 4>, output_ptr, input_ptr, hidden);
      break;
    default:
      host::Panic("moe_sum JIT kernel only supports topk 2, 3, and 4");
  }
}

}  // namespace sglang
