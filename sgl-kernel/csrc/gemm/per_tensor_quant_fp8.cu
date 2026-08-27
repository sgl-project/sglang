#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>

#include <cmath>
#include <cub/block/block_reduce.cuh>
#include <flashinfer/vec_dtypes.cuh>

#include "utils.h"

template <typename T>
__global__ void
per_tensor_absmax_kernel(const T* __restrict__ input, float* __restrict__ output_s, const int64_t num_elements) {
  float max_value = 0.0f;
  unsigned int tid = threadIdx.x;
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_size = blockDim.x * gridDim.x;

  constexpr uint32_t vec_size = 16 / sizeof(T);
  using vec_t = flashinfer::vec_t<T, vec_size>;

  const int32_t num_vec_elems = num_elements / vec_size;

  for (int32_t i = gid; i < num_vec_elems; i += grid_size) {
    vec_t input_vec;
    input_vec.cast_load(input + i * vec_size);

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float val = static_cast<float>(input_vec[j]);
      max_value = fmaxf(max_value, fabsf(val));
    }
  }

  const int32_t remaining_start = num_vec_elems * vec_size;
  for (int32_t idx = remaining_start + gid; idx < num_elements; idx += grid_size) {
    float val = static_cast<float>(input[idx]);
    max_value = fmaxf(max_value, fabsf(val));
  }

  max_value = blockReduceMax(max_value);

  if (tid == 0) {
    atomicMaxFloat(output_s, max_value / FP8_E4M3_MAX);
  }
}

template <typename T, typename DST_DTYPE>
__global__ void per_tensor_quant_fp8_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output,
    const float* __restrict__ scale,
    const int64_t num_elements) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_size = blockDim.x * gridDim.x;
  const float scale_val = 1.0f / (*scale);

  // We want to store 128 bits of data at a time. 16 = 128 / 8 bits
  // Load is already vectorized, so 16 elements work for T.
  const uint32_t VEC_SIZE = 16;
  using vec_t = flashinfer::vec_t<T, VEC_SIZE>;

  const int32_t num_vec_elems = num_elements / VEC_SIZE;

  for (int32_t i = gid; i < num_vec_elems; i += grid_size) {
    vec_t input_vec;
    input_vec.cast_load(input + i * VEC_SIZE);

    DST_DTYPE output_arr[VEC_SIZE];
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      float val = fmax(fmin(static_cast<float>(input_vec[j]) * scale_val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
      output_arr[j] = static_cast<DST_DTYPE>(val);
#else
      output_arr[j] = c10::Float8_e4m3fnuz(
          __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
          c10::Float8_e4m3fnuz::from_bits());
#endif
    }
    *(uint4*)(output + i * VEC_SIZE) = *(uint4*)output_arr;
  }

  const int32_t remaining_start = num_vec_elems * VEC_SIZE;
  for (int32_t idx = remaining_start + gid; idx < num_elements; idx += grid_size) {
    float val = fmax(-FP8_E4M3_MAX, fmin(static_cast<float>(input[idx]) * scale_val, FP8_E4M3_MAX));
#if !defined(USE_ROCM) || defined(HIP_FP8_TYPE_E4M3)
    output[idx] = static_cast<DST_DTYPE>(val);
#else
    output[idx] = c10::Float8_e4m3fnuz(
        __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
        c10::Float8_e4m3fnuz::from_bits());
#endif
  }
}

void sgl_per_tensor_quant_fp8(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s, bool is_static) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);

  const int block_size = 256;
  const int num_elements = input.numel();
  const int num_blocks = min((num_elements + block_size - 1) / block_size, 1024);

  dim3 grid(num_blocks);
  dim3 block(block_size);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    if (is_static == false) {
      per_tensor_absmax_kernel<scalar_t><<<grid, block, 0, stream>>>(
          static_cast<scalar_t*>(input.data_ptr()), static_cast<float*>(output_s.data_ptr()), num_elements);
    }

    per_tensor_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3><<<grid, block, 0, stream>>>(
        static_cast<scalar_t*>(input.data_ptr()),
        static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        num_elements);
    return true;
  });
}
