#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <cub/block/block_reduce.cuh>
#include <flashinfer/vec_dtypes.cuh>

#include "utils.h"

template <typename T>
__global__ void per_token_quant_fp8_kernel(
    const T* __restrict__ input,
    FP8_TYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;

  const int tid = threadIdx.x;
  const int block_dim = blockDim.x;

  const T* token_input = input + token_idx * hidden_dim;
  FP8_TYPE* token_output = output_q + token_idx * hidden_dim;

  float max_value = 0.0f;
  constexpr uint32_t vec_size = 16 / sizeof(T);
  const bool use_vector = (hidden_dim % vec_size == 0);

  if (use_vector) {
    using vec_t = flashinfer::vec_t<T, vec_size>;
    const int32_t num_vec_elems = hidden_dim / vec_size;

    for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
      vec_t input_vec;
      input_vec.cast_load(token_input + i * vec_size);

#pragma unroll
      for (uint32_t j = 0; j < vec_size; ++j) {
        max_value = fmaxf(max_value, fabsf(static_cast<float>(input_vec[j])));
      }
    }
  } else {
    for (int32_t idx = tid; idx < hidden_dim; idx += block_dim) {
      max_value = fmaxf(max_value, fabsf(static_cast<float>(token_input[idx])));
    }
  }

  max_value = blockReduceMax(max_value);
  __shared__ float scale_val;
  if (tid == 0) {
    float block_max = max_value / FP8_E4M3_MAX;
    output_s[token_idx] = block_max;
    scale_val = 1.0f / block_max;
  }
  __syncthreads();

  if (use_vector) {
    using vec_t = flashinfer::vec_t<T, vec_size>;
    const int32_t num_vec_elems = hidden_dim / vec_size;

    for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
      vec_t input_vec;
      input_vec.cast_load(token_input + i * vec_size);
      const int32_t base_idx = i * vec_size;

      FP8_TYPE output_arr[vec_size];
#pragma unroll
      for (uint32_t j = 0; j < vec_size; ++j) {
        float val = static_cast<float>(input_vec[j]) * scale_val;
        val = fmaxf(fminf(val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#ifndef USE_ROCM
        output_arr[j] = static_cast<FP8_TYPE>(val);
#else
        output_arr[j] = c10::Float8_e4m3fnuz(
            __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
            c10::Float8_e4m3fnuz::from_bits());
#endif
        token_output[base_idx + j] = output_arr[j];
      }
    }
  } else {
    for (int32_t idx = tid; idx < hidden_dim; idx += block_dim) {
      float val = static_cast<float>(token_input[idx]) * scale_val;
      val = fmaxf(fminf(val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#ifndef USE_ROCM
      token_output[idx] = static_cast<FP8_TYPE>(val);
#else
      token_output[idx] = c10::Float8_e4m3fnuz(
          __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
          c10::Float8_e4m3fnuz::from_bits());
#endif
    }
  }
}

void sgl_per_token_quant_fp8(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);

  const auto input_sizes = input.sizes();
  const int64_t num_tokens = input_sizes[0];
  const int64_t hidden_dim = input_sizes[1];

  const int block_size = 256;
  const int num_blocks = num_tokens;

  dim3 grid(num_blocks);
  dim3 block(block_size);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    per_token_quant_fp8_kernel<scalar_t><<<grid, block, 0, stream>>>(
        static_cast<scalar_t*>(input.data_ptr()),
        static_cast<FP8_TYPE*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        hidden_dim,
        num_tokens);
    return true;
  });
}
