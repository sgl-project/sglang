#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

template <typename DType, int kTopK>
__global__ void moe_sum_reduce_kernel(DType* __restrict__ out,
                                      const DType* __restrict__ input,
                                      const int hidden_size,
                                      const float scale) {
  const int64_t token_idx = blockIdx.x;
  const DType* token_input = &input[token_idx * kTopK * hidden_size];
  DType* token_output = &out[token_idx * hidden_size];
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      acc += static_cast<float>(
          SGLANG_LDG(&token_input[k * hidden_size + idx]));
    }
    token_output[idx] = static_cast<DType>(acc * scale);
  }
}

template <typename DType, int kTopK>
void moe_sum_reduce(tvm::ffi::TensorView input,
                    tvm::ffi::TensorView output,
                    float scale) {
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

  RuntimeCheck(topk == kTopK, "moe_sum_reduce: expected topk=", kTopK, ", got ", topk);
  RuntimeCheck(num_tokens > 0, "moe_sum_reduce: num_tokens must be > 0, got ", num_tokens);
  RuntimeCheck(hidden_size > 0, "moe_sum_reduce: hidden_size must be > 0, got ", hidden_size);

  const dim3 grid(num_tokens);
  const dim3 block(std::min(hidden_size, 1024));

  LaunchKernel(grid, block, device.unwrap())(
      moe_sum_reduce_kernel<DType, kTopK>,
      static_cast<DType*>(output.data_ptr()),
      static_cast<const DType*>(input.data_ptr()),
      hidden_size,
      scale);
}

}  // namespace
