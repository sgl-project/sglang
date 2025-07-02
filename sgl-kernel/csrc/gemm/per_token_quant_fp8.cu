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

  const T* token_input_base = input + token_idx * hidden_dim;
  FP8_TYPE* token_output_base = output_q + token_idx * hidden_dim;

  float max_value = 0.0f;
  const uint32_t VEC_SIZE = 16;
  using vec_t = flashinfer::vec_t<T, VEC_SIZE>;
  const uint32_t FP8_BYTES_PER_VECTOR = VEC_SIZE * sizeof(FP8_TYPE);
  const uint32_t ALIGNMENT_BYTES_OUTPUT = FP8_BYTES_PER_VECTOR;

  // --- Step 1: Find Max Value ---
  size_t token_input_base_addr = reinterpret_cast<size_t>(token_input_base);
  int prologue_elements_count = 0;

  if (token_input_base_addr % ALIGNMENT_BYTES_OUTPUT != 0) {
    prologue_elements_count = (ALIGNMENT_BYTES_OUTPUT - (token_input_base_addr % ALIGNMENT_BYTES_OUTPUT)) / sizeof(T);
  }
  // Ensure we don't go beyond hidden_dim if hidden_dim is very small
  prologue_elements_count = min((int64_t)prologue_elements_count, hidden_dim);

  // Scalar loop for the initial (unaligned) prologue
  for (int32_t i = tid; i < prologue_elements_count; i += block_dim) {
    float val = static_cast<float>(token_input_base[i]);
    max_value = fmaxf(max_value, fabsf(val));
  }

  // Calculate remaining elements after the prologue
  const int64_t remaining_elements = hidden_dim - prologue_elements_count;
  const int32_t num_full_vecs = remaining_elements / VEC_SIZE;
  const int32_t num_tail_elems = remaining_elements % VEC_SIZE;

  const T* token_input_aligned_start = token_input_base + prologue_elements_count;

  // Vectorized loop for the aligned main part
  for (int32_t i = tid; i < num_full_vecs; i += block_dim) {
    vec_t input_vec;
    input_vec.cast_load(token_input_aligned_start + i * VEC_SIZE);

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      float val = static_cast<float>(input_vec[j]);
      max_value = fmaxf(max_value, fabsf(val));
    }
  }

  // Scalar loop for the final (unaligned) tail elements
  if (num_tail_elems > 0) {
    const int32_t tail_start_idx = prologue_elements_count + num_full_vecs * VEC_SIZE;
    for (int32_t i = tail_start_idx + tid; i < hidden_dim; i += block_dim) {
      float val = static_cast<float>(token_input_base[i]);
      max_value = fmaxf(max_value, fabsf(val));
    }
  }

  // Reduce max_value across threads in the block
  max_value = blockReduceMax(max_value);

  __shared__ float scale;
  if (tid == 0) {
    scale = (max_value == 0.0f) ? 1.0f : max_value / FP8_E4M3_MAX;
    output_s[token_idx] = scale;
  }
  __syncthreads();

  const float scale_inv = 1.0f / scale;

  // --- Step 2: Quantize Values ---

  // Calculate the first address in the output row that IS aligned to ALIGNMENT_BYTES_OUTPUT (16 bytes)
  size_t token_output_base_addr = reinterpret_cast<size_t>(token_output_base);
  int prologue_output_elements_count = 0;

  if (token_output_base_addr % ALIGNMENT_BYTES_OUTPUT != 0) {
    prologue_output_elements_count =
        (ALIGNMENT_BYTES_OUTPUT - (token_output_base_addr % ALIGNMENT_BYTES_OUTPUT)) / sizeof(FP8_TYPE);
  }
  prologue_output_elements_count = min((int64_t)prologue_output_elements_count, hidden_dim);

  // Scalar loop for the initial (unaligned) output prologue
  for (int32_t i = tid; i < prologue_output_elements_count; i += block_dim) {
    float val = fmaxf(fminf(static_cast<float>(token_input_base[i]) * scale_inv, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#ifndef USE_ROCM
    token_output_base[i] = static_cast<FP8_TYPE>(val);
#else
    token_output_base[i] = c10::Float8_e4m3fnuz(
        __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
        c10::Float8_e4m3fnuz::from_bits());
#endif
  }

  // Calculate remaining elements for aligned part and tail for output
  const int64_t remaining_output_elements = hidden_dim - prologue_output_elements_count;
  const int32_t num_full_vecs_output = remaining_output_elements / VEC_SIZE;
  const int32_t num_tail_elems_output = remaining_output_elements % VEC_SIZE;

  const T* token_input_for_aligned_output = token_input_base + prologue_output_elements_count;
  FP8_TYPE* token_output_aligned_start = token_output_base + prologue_output_elements_count;

  assert(
      reinterpret_cast<size_t>(token_input_for_aligned_output) % ALIGNMENT_BYTES_OUTPUT == 0 &&
      "Input address must be 16-byte aligned.");

  // Vectorized loop for the aligned main part (output)
  for (int32_t i = tid; i < num_full_vecs_output; i += block_dim) {
    vec_t input_vec;
    input_vec.cast_load(token_input_for_aligned_output + i * VEC_SIZE);

    FP8_TYPE output_arr[VEC_SIZE];
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      float val = fmaxf(fminf(static_cast<float>(input_vec[j]) * scale_inv, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#ifndef USE_ROCM
      output_arr[j] = static_cast<FP8_TYPE>(val);
#else
      output_arr[j] = c10::Float8_e4m3fnuz(
          __hip_cvt_float_to_fp8(val, fp8::fp8_type::__default_saturation, fp8::fp8_type::__default_interpret),
          c10::Float8_e4m3fnuz::from_bits());
#endif
    }
    *(reinterpret_cast<uint4*>(token_output_aligned_start + i * VEC_SIZE)) = *(reinterpret_cast<uint4*>(output_arr));
  }

  // Scalar loop for the final (unaligned) output tail elements
  if (num_tail_elems_output > 0) {
    const int32_t tail_output_start_idx = prologue_output_elements_count + num_full_vecs_output * VEC_SIZE;
    for (int32_t i = tail_output_start_idx + tid; i < hidden_dim; i += block_dim) {
      float val = fmaxf(fminf(static_cast<float>(token_input_base[i]) * scale_inv, FP8_E4M3_MAX), -FP8_E4M3_MAX);
#ifndef USE_ROCM
      token_output_base[i] = static_cast<FP8_TYPE>(val);
#else
      token_output_base[i] = c10::Float8_e4m3fnuz(
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
