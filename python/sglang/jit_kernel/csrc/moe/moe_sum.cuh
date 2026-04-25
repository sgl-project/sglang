#include <sgl_kernel/tensor.h>  // TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // RuntimeCheck, div_ceil

#include <sgl_kernel/utils.cuh>  // LaunchKernel, SGLANG_LDG, type aliases

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

template <typename DType, int kTopK>
__global__ void moe_sum_kernel(DType* __restrict__ out, const DType* __restrict__ input, const int hidden_size) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    DType x = static_cast<DType>(0.0f);
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      x += SGLANG_LDG(&input[token_idx * kTopK * hidden_size + k * hidden_size + idx]);
    }
    out[token_idx * hidden_size + idx] = x;
  }
}

template <typename DType, int kTopK>
void moe_sum(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
  using namespace host;

  auto N = SymbolicSize{"num_tokens"};
  auto K = SymbolicSize{"topk"};
  auto D = SymbolicSize{"hidden_size"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({N, K, D})  //
      .with_dtype<DType>()
      .with_device(device)
      .verify(input);
  TensorMatcher({N, D})  //
      .with_dtype<DType>()
      .with_device(device)
      .verify(output);

  const auto num_tokens = static_cast<int>(N.unwrap());
  const auto topk = static_cast<int>(K.unwrap());
  const auto hidden_size = static_cast<int>(D.unwrap());

  RuntimeCheck(topk == kTopK, "moe_sum: expected topk=", kTopK, ", got ", topk);
  RuntimeCheck(num_tokens > 0, "moe_sum: num_tokens must be > 0, got ", num_tokens);
  RuntimeCheck(hidden_size > 0, "moe_sum: hidden_size must be > 0, got ", hidden_size);

  const dim3 grid(num_tokens);
  const dim3 block(std::min(hidden_size, 1024));

  LaunchKernel(grid, block, device.unwrap())(
      moe_sum_kernel<DType, kTopK>,
      static_cast<DType*>(output.data_ptr()),
      static_cast<const DType*>(input.data_ptr()),
      hidden_size);
}

}  // namespace
