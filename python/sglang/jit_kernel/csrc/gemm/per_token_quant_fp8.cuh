#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

template <typename DType, int kVecSize>
__global__ void per_token_quant_fp8_kernel(const DType* __restrict__ input,
                                           fp8_e4m3_t* __restrict__ output_q,
                                           float* __restrict__ output_s,
                                           const int64_t hidden_dim) {
  using namespace device;

  const int64_t token_idx = blockIdx.x;

  const DType* token_input = input + token_idx * hidden_dim;
  fp8_e4m3_t* token_output = output_q + token_idx * hidden_dim;

  const int32_t num_vec_elems = hidden_dim / kVecSize;
  const int tid = threadIdx.x;
  const int block_dim = blockDim.x;

  float max_value = 0.0f;

  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    AlignedVector<DType, kVecSize> vec;
    vec.load(token_input, i);
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      max_value = math::max(max_value, math::abs(static_cast<float>(vec[j])));
    }
  }

  __shared__ float smem[kWarpThreads];
  cta::reduce_max(max_value, smem);
  __syncthreads();

  const float absmax = smem[0];
  const float scale = absmax / math::FP8_E4M3_MAX;
  if (tid == 0) {
    output_s[token_idx] = scale;
  }
  const float scale_inv = (scale == 0.0f) ? 0.0f : 1.0f / scale;

  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    AlignedVector<DType, kVecSize> in_vec;
    in_vec.load(token_input, i);

    AlignedVector<fp8_e4m3_t, kVecSize> out_vec;
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      float val = static_cast<float>(in_vec[j]) * scale_inv;
      val = math::max(math::min(val, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX);
      out_vec[j] = static_cast<fp8_e4m3_t>(val);
    }
    out_vec.store(token_output, i);
  }
}

template <typename DType>
void per_token_quant_fp8(tvm::ffi::TensorView input,
                         tvm::ffi::TensorView output_q,
                         tvm::ffi::TensorView output_s) {
  using namespace host;

  auto M = SymbolicSize{"num_tokens"};
  auto D = SymbolicSize{"hidden_dim"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({M, D})  //
      .with_dtype<DType>()
      .with_device(device)
      .verify(input);
  TensorMatcher({M, D})  //
      .with_dtype<fp8_e4m3_t>()
      .with_device(device)
      .verify(output_q);
  TensorMatcher({M})  //
      .with_dtype<fp32_t>()
      .with_device(device)
      .verify(output_s);

  const auto num_tokens = static_cast<int64_t>(M.unwrap());
  const auto hidden_dim = static_cast<int64_t>(D.unwrap());

  RuntimeCheck(hidden_dim % 4 == 0, "per_token_quant_fp8: hidden_dim must be divisible by 4, got ", hidden_dim);

  constexpr int kBlockSize = 256;
  constexpr int kMaxVecSize = 32 / sizeof(DType);

  const auto* input_ptr = static_cast<const DType*>(input.data_ptr());
  auto* output_q_ptr = static_cast<fp8_e4m3_t*>(output_q.data_ptr());
  auto* output_s_ptr = static_cast<float*>(output_s.data_ptr());

  auto launch = [&](auto kernel) {
    LaunchKernel(num_tokens, kBlockSize, device.unwrap())(
        kernel, input_ptr, output_q_ptr, output_s_ptr, hidden_dim);
  };

  if (hidden_dim % kMaxVecSize == 0) {
    launch(per_token_quant_fp8_kernel<DType, kMaxVecSize>);
  } else if constexpr (kMaxVecSize > 8) {
    if (hidden_dim % 8 == 0) {
      launch(per_token_quant_fp8_kernel<DType, 8>);
    } else {
      launch(per_token_quant_fp8_kernel<DType, 4>);
    }
  } else {
    launch(per_token_quant_fp8_kernel<DType, 4>);
  }
}

}  // namespace
