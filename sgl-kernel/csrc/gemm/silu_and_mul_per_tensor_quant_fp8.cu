#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>

#include <cmath>
#include <cub/block/block_reduce.cuh>
#include <flashinfer/vec_dtypes.cuh>

#include "utils.h"

__device__ __forceinline__ float silu(const float& val) {
  return val / (1.0f + __expf(-val));
}

template <typename T>
__global__ void silu_and_mul_per_tensor_absmax_kernel(
    T* __restrict__ input,
    const T* __restrict__ input_2,
    float* __restrict__ output_s,
    const int64_t num_elements,
    const int hidden_dim) {
  float max_value = 0.0f;
  unsigned int tid = threadIdx.x;
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_size = blockDim.x * gridDim.x;

  constexpr uint32_t vec_size = 16 / sizeof(T);
  using vec_t = flashinfer::vec_t<T, vec_size>;

  const int32_t num_vec_elems = num_elements / vec_size;

  for (int32_t i = gid; i < num_vec_elems; i += grid_size) {
    vec_t input_vec;
    vec_t input_vec_2;
    input_vec.cast_load(input + i * vec_size);
    input_vec_2.cast_load(input_2 + i * vec_size);

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      input_vec[j] = silu(static_cast<float>(input_vec[j])) * static_cast<float>(input_vec_2[j]);
      float val = static_cast<float>(input_vec[j]);
      max_value = fmaxf(max_value, fabsf(val));
    }
    input_vec.cast_store(input + i * vec_size);
  }

  const int32_t remaining_start = num_vec_elems * vec_size;
  for (int32_t idx = remaining_start + gid; idx < num_elements; idx += grid_size) {
    float val = silu(static_cast<float>(input[idx])) * static_cast<float>(input_2[idx]);
    max_value = fmaxf(max_value, fabsf(static_cast<float>(val)));
    input[idx] = static_cast<T>(val);
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
#ifndef USE_ROCM
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
#ifndef USE_ROCM
    output[idx] = static_cast<DST_DTYPE>(val);
#else
    output[idx] = c10::Float8_e4m3fnuz(
        __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
        c10::Float8_e4m3fnuz::from_bits());
#endif
  }
}

void sgl_silu_and_mul_per_tensor_quant_fp8_chunk(
    torch::Tensor input_gate, torch::Tensor input_up, torch::Tensor output_q, torch::Tensor output_s, bool is_static) {
  CHECK_INPUT(input_gate);
  CHECK_INPUT(input_up);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);
  TORCH_CHECK(is_static == false, "Static mode is not supported for silu_and_mul_per_tensor_quant_fp8");

  const int block_size = 256;
  const int num_elements = input_gate.numel();
  const int num_blocks = min((num_elements + block_size - 1) / block_size, 1024);
  const int hidden_dim = input_gate.size(-1);

  dim3 grid(num_blocks);
  dim3 block(block_size);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input_gate.scalar_type(), scalar_t, [&] {
    silu_and_mul_per_tensor_absmax_kernel<scalar_t><<<grid, block, 0, stream>>>(
        static_cast<scalar_t*>(input_gate.data_ptr()),
        static_cast<scalar_t*>(input_up.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        num_elements,
        hidden_dim);

    per_tensor_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3><<<grid, block, 0, stream>>>(
        static_cast<scalar_t*>(input_gate.data_ptr()),
        static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        num_elements);
    return true;
  });
}

// Fused SiLU and Mul kernel that takes concatenated input [seq_len, hidden_dim*2]
// and produces output [seq_len, hidden_dim] without requiring pre-chunking
template <typename T>
__global__ void fused_silu_mul_per_tensor_absmax_kernel(
    const T* __restrict__ input,   // [seq_len, hidden_dim*2] - concatenated gate and up
    T* __restrict__ output,        // [seq_len, hidden_dim]
    float* __restrict__ output_s,  // scalar for quantization
    const int64_t seq_len,
    const int64_t hidden_dim) {
  float max_value = 0.0f;
  unsigned int tid = threadIdx.x;
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_size = blockDim.x * gridDim.x;

  // Use vectorized operations for better performance
  constexpr uint32_t vec_size = 16 / sizeof(T);
  using vec_t = flashinfer::vec_t<T, vec_size>;

  const int64_t total_output_elements = seq_len * hidden_dim;
  const int64_t num_vec_elems = total_output_elements / vec_size;

  // Vectorized processing
  for (int64_t i = gid; i < num_vec_elems; i += grid_size) {
    // Calculate which row and column range we're processing
    int64_t output_idx = i * vec_size;
    int64_t row = output_idx / hidden_dim;
    int64_t col_start = output_idx % hidden_dim;

    vec_t gate_vec;
    vec_t up_vec;
    vec_t output_vec;

    // Check if we're crossing row boundary
    if (col_start + vec_size <= hidden_dim) {
      // All elements in same row - most common case
      int64_t gate_offset = row * (hidden_dim * 2) + col_start;
      int64_t up_offset = row * (hidden_dim * 2) + hidden_dim + col_start;

      gate_vec.cast_load(input + gate_offset);
      up_vec.cast_load(input + up_offset);

#pragma unroll
      for (uint32_t j = 0; j < vec_size; ++j) {
        float gate_val = static_cast<float>(gate_vec[j]);
        float up_val = static_cast<float>(up_vec[j]);
        float result = silu(gate_val) * up_val;
        output_vec[j] = static_cast<T>(result);
        max_value = fmaxf(max_value, fabsf(result));
      }

      output_vec.cast_store(output + output_idx);
    } else {
      // Handle boundary case - process element by element
      for (uint32_t j = 0; j < vec_size; ++j) {
        int64_t elem_idx = output_idx + j;
        if (elem_idx < total_output_elements) {
          int64_t elem_row = elem_idx / hidden_dim;
          int64_t elem_col = elem_idx % hidden_dim;

          int64_t gate_idx = elem_row * (hidden_dim * 2) + elem_col;
          int64_t up_idx = elem_row * (hidden_dim * 2) + hidden_dim + elem_col;

          float gate_val = static_cast<float>(input[gate_idx]);
          float up_val = static_cast<float>(input[up_idx]);
          float result = silu(gate_val) * up_val;

          output[elem_idx] = static_cast<T>(result);
          max_value = fmaxf(max_value, fabsf(result));
        }
      }
    }
  }

  // Handle remaining elements
  const int64_t remaining_start = num_vec_elems * vec_size;
  for (int64_t idx = remaining_start + gid; idx < total_output_elements; idx += grid_size) {
    int64_t row = idx / hidden_dim;
    int64_t col = idx % hidden_dim;

    int64_t gate_idx = row * (hidden_dim * 2) + col;
    int64_t up_idx = row * (hidden_dim * 2) + hidden_dim + col;

    float gate_val = static_cast<float>(input[gate_idx]);
    float up_val = static_cast<float>(input[up_idx]);
    float result = silu(gate_val) * up_val;

    output[idx] = static_cast<T>(result);
    max_value = fmaxf(max_value, fabsf(result));
  }

  // Reduce max value across block
  max_value = blockReduceMax(max_value);

  if (tid == 0) {
    atomicMaxFloat(output_s, max_value / FP8_E4M3_MAX);
  }
}

// Host function that takes concatenated input without requiring pre-chunking
void sgl_silu_and_mul_per_tensor_quant_fp8(
    torch::Tensor input,     // [seq_len, hidden_dim*2] - concatenated gate and up
    torch::Tensor output,    // [seq_len, hidden_dim] - intermediate output
    torch::Tensor output_q,  // [seq_len, hidden_dim] - quantized output
    torch::Tensor output_s,  // scalar for quantization
    bool is_static) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);

  TORCH_CHECK(is_static == false, "Static mode is not supported for fused_silu_and_mul_per_tensor_quant_fp8");
  TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor");
  TORCH_CHECK(input.size(1) % 2 == 0, "Input hidden dimension must be even (hidden_dim*2)");

  const int seq_len = input.size(0);
  const int hidden_dim = input.size(1) / 2;
  const int num_output_elements = seq_len * hidden_dim;

  const int block_size = 256;
  const int num_blocks = min((num_output_elements + block_size - 1) / block_size, 1024);

  dim3 grid(num_blocks);
  dim3 block(block_size);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    // Fused SiLU and Mul kernel
    fused_silu_mul_per_tensor_absmax_kernel<scalar_t><<<grid, block, 0, stream>>>(
        static_cast<scalar_t*>(input.data_ptr()),
        static_cast<scalar_t*>(output.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        seq_len,
        hidden_dim);

    // Quantization kernel
    per_tensor_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3><<<grid, block, 0, stream>>>(
        static_cast<scalar_t*>(output.data_ptr()),
        static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        num_output_elements);

    return true;
  });
}
