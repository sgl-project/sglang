#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <cub/block/block_reduce.cuh>
#include <flashinfer/vec_dtypes.cuh>

#include "utils.h"

// Function adapted from https://github.com/vllm-project/vllm/blob/53056731fdf8aa8f960cd3473b01a36d5cb0281d/csrc/quantization/compressed_tensors/int8_quant_kernels.cu#L15
static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
  float dst = std::nearbyint(x);

  // saturate
  dst = std::clamp(dst, i8_min, i8_max);
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

template <typename T>
__global__ void per_token_quant_int8_kernel(
    const T* __restrict__ input,
    int8_t* __restrict__ output_q,
    float* __restrict__ output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;

  const int tid = threadIdx.x;
  const int block_dim = blockDim.x;

  const T* token_input = input + token_idx * hidden_dim;
  int8_t* token_output = output_q + token_idx * hidden_dim;

  float max_value = 0.0f;

  constexpr uint32_t vec_size = 16 / sizeof(T);
  using vec_t = flashinfer::vec_t<T, vec_size>;
  const int32_t num_vec_elems = hidden_dim / vec_size;

  // Find max using vectorized loads
  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * vec_size);

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float val = static_cast<float>(input_vec[j]);
      max_value = fmaxf(max_value, fabsf(val));
    }
  }

  max_value = blockReduceMax(max_value);

  __shared__ float block_max;
  if (tid == 0) {
    block_max = max_value / 127.0f;
    output_s[token_idx] = block_max;
  }
  __syncthreads();

  const float scale_val = 1.0f / block_max;

  // Quantize using vectorized loads
  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * vec_size);

    int8_t output_arr[vec_size];
#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float val = fmaxf(fminf(static_cast<float>(input_vec[j]) * scale_val, 127.0f), -128.0f);
      output_arr[j] = float_to_int8_rn(val);
    }

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      token_output[i * vec_size + j] = output_arr[j];
    }
  }
}

void sgl_per_token_quant_int8(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);

  const auto input_sizes = input.sizes();
  const int64_t num_tokens = input_sizes[0];
  const int64_t hidden_dim = input_sizes[1];

  TORCH_CHECK(hidden_dim % 8 == 0, "Hidden dimension must be divisible by 8, but got ", hidden_dim);

  const int block_size = 256;
  const int num_blocks = num_tokens;

  dim3 grid(num_blocks);
  dim3 block(block_size);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    per_token_quant_int8_kernel<scalar_t><<<grid, block, 0, stream>>>(
        static_cast<scalar_t*>(input.data_ptr()),
        static_cast<int8_t*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        hidden_dim,
        num_tokens);
    return true;
  });
}