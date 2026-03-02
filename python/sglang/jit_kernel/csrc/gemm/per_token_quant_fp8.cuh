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
                                           const int64_t hidden_dim,
                                           const int64_t num_tokens) {
  using namespace device;

  const int64_t token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;

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

    fp8_e4m3_t out_arr[kVecSize];
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      float val = static_cast<float>(in_vec[j]) * scale_inv;
      val = math::max(math::min(val, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX);
      out_arr[j] = static_cast<fp8_e4m3_t>(val);
    }

    if constexpr (kVecSize == 16) {
      *reinterpret_cast<uint4*>(token_output + i * kVecSize) = *reinterpret_cast<uint4*>(out_arr);
    } else if constexpr (kVecSize == 8) {
      *reinterpret_cast<uint2*>(token_output + i * kVecSize) = *reinterpret_cast<uint2*>(out_arr);
    } else if constexpr (kVecSize == 4) {
      *reinterpret_cast<unsigned int*>(token_output + i * kVecSize) = *reinterpret_cast<unsigned int*>(out_arr);
    } else {
#pragma unroll
      for (int k = 0; k < kVecSize; ++k) {
        token_output[i * kVecSize + k] = out_arr[k];
      }
    }
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

  if (hidden_dim % kMaxVecSize == 0) {
    constexpr int kVecSize = kMaxVecSize;
    LaunchKernel(num_tokens, kBlockSize, device.unwrap())(
        per_token_quant_fp8_kernel<DType, kVecSize>,
        static_cast<const DType*>(input.data_ptr()),
        static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        hidden_dim,
        num_tokens);
  } else if constexpr (kMaxVecSize > 8) {
    if (hidden_dim % 8 == 0) {
      constexpr int kVecSize = 8;
      LaunchKernel(num_tokens, kBlockSize, device.unwrap())(
          per_token_quant_fp8_kernel<DType, kVecSize>,
          static_cast<const DType*>(input.data_ptr()),
          static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
          static_cast<float*>(output_s.data_ptr()),
          hidden_dim,
          num_tokens);
    } else {
      constexpr int kVecSize = 4;
      LaunchKernel(num_tokens, kBlockSize, device.unwrap())(
          per_token_quant_fp8_kernel<DType, kVecSize>,
          static_cast<const DType*>(input.data_ptr()),
          static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
          static_cast<float*>(output_s.data_ptr()),
          hidden_dim,
          num_tokens);
    }
  } else {
    constexpr int kVecSize = 4;
    LaunchKernel(num_tokens, kBlockSize, device.unwrap())(
        per_token_quant_fp8_kernel<DType, kVecSize>,
        static_cast<const DType*>(input.data_ptr()),
        static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        hidden_dim,
        num_tokens);
  }
}

}  // namespace
