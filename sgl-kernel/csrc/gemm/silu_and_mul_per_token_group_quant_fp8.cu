#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>

#include <cmath>
#include <flashinfer/vec_dtypes.cuh>
#include <iostream>

#include "utils.h"

__device__ __forceinline__ float silu(const float& val) {
  return val / (1.0f + __expf(-val));
}

template <typename T, typename DST_DTYPE, bool IS_COLUMN_MAJOR = false>
__global__ void silu_and_mul_per_token_group_quant_8bit_kernel(
    T* __restrict__ input,
    const int d,
    void* __restrict__ output_q,
    float* __restrict__ output_s,
    const int group_size,
    const int num_groups,
    const int groups_per_block,
    const int groups_per_token,
    const float eps,
    const float min_8bit,
    const float max_8bit,
    const int scale_num_rows = 0,
    const int scale_stride = 0) {
  const int threads_per_group = 16;
  const int local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;

  const int block_group_id = blockIdx.x * groups_per_block;
  const int global_group_id = block_group_id + local_group_id;
  const int token_id = global_group_id / groups_per_token;
  const int group_in_token_offset = global_group_id % groups_per_token;
  const int block_group_offset = (token_id * groups_per_token * 2 + group_in_token_offset) * group_size;
  float local_absmax = eps;
  T* group_input = input + block_group_offset;
  DST_DTYPE* group_output = static_cast<DST_DTYPE*>(output_q) + global_group_id * group_size;
  float* scale_output;

  if constexpr (IS_COLUMN_MAJOR) {
    const int row_idx = global_group_id / scale_num_rows;
    const int col_idx = global_group_id % scale_num_rows;
    scale_output = output_s + (col_idx * scale_stride + row_idx);
  } else {
    scale_output = output_s + global_group_id;
  }

  constexpr uint32_t vec_size = 16 / sizeof(T);
  using vec_t = flashinfer::vec_t<T, vec_size>;

  const int32_t num_vec_elems = group_size / vec_size;

  for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
    vec_t x_vec, y_dec;
    x_vec.cast_load(group_input + i * vec_size);
    y_dec.cast_load(group_input + i * vec_size + d);
#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      x_vec[j] = silu(static_cast<float>(x_vec[j])) * static_cast<float>(y_dec[j]);
      float abs_val = fabsf(x_vec[j]);
      local_absmax = fmaxf(local_absmax, abs_val);
    }
    x_vec.cast_store(group_input + i * vec_size);
  }

  local_absmax = GroupReduceMax(local_absmax, lane_id);

  const float y_s = local_absmax / max_8bit;

  if (lane_id == 0) {
    *scale_output = y_s;
  }

  for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
    vec_t input_vec;
    input_vec.cast_load(group_input + i * vec_size);

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float val = static_cast<float>(input_vec[j]);
      float q_val = fminf(fmaxf(val / y_s, min_8bit), max_8bit);
      group_output[i * vec_size + j] = DST_DTYPE(q_val);
    }
  }
}

void sgl_silu_and_mul_per_token_group_quant_8bit(
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_EQ(input.numel() / 2 % group_size, 0);
  CHECK_EQ(output_s.dim(), 2);

  const int d = input.size(-1) / 2;
  const int groups_per_token = d / group_size;
  const int num_groups = input.numel() / group_size / 2;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int THREADS_PER_GROUP = 16;

  int groups_per_block = 1;

  if (num_groups % 16 == 0) {
    groups_per_block = 16;
  } else if (num_groups % 8 == 0) {
    groups_per_block = 8;
  } else if (num_groups % 4 == 0) {
    groups_per_block = 4;
  } else if (num_groups % 2 == 0) {
    groups_per_block = 2;
  }

  auto dst_type = output_q.scalar_type();
  const int num_blocks = num_groups / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;

  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int scale_num_rows = output_s.size(1);
  const int scale_stride = output_s.stride(1);

#define LAUNCH_KERNEL(T, DST_DTYPE)                                                                    \
  do {                                                                                                 \
    dim3 grid(num_blocks);                                                                             \
    dim3 block(num_threads);                                                                           \
    if (is_column_major) {                                                                             \
      silu_and_mul_per_token_group_quant_8bit_kernel<T, DST_DTYPE, true><<<grid, block, 0, stream>>>(  \
          static_cast<T*>(input.data_ptr()),                                                           \
          d,                                                                                           \
          output_q.data_ptr(),                                                                         \
          static_cast<float*>(output_s.data_ptr()),                                                    \
          group_size,                                                                                  \
          num_groups,                                                                                  \
          groups_per_block,                                                                            \
          groups_per_token,                                                                            \
          (float)eps,                                                                                  \
          (float)min_8bit,                                                                             \
          (float)max_8bit,                                                                             \
          scale_num_rows,                                                                              \
          scale_stride);                                                                               \
    } else {                                                                                           \
      silu_and_mul_per_token_group_quant_8bit_kernel<T, DST_DTYPE, false><<<grid, block, 0, stream>>>( \
          static_cast<T*>(input.data_ptr()),                                                           \
          d,                                                                                           \
          output_q.data_ptr(),                                                                         \
          static_cast<float*>(output_s.data_ptr()),                                                    \
          group_size,                                                                                  \
          num_groups,                                                                                  \
          groups_per_block,                                                                            \
          groups_per_token,                                                                            \
          (float)eps,                                                                                  \
          (float)min_8bit,                                                                             \
          (float)max_8bit);                                                                            \
    }                                                                                                  \
  } while (0)

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    if (dst_type == at::ScalarType::Char) {
      LAUNCH_KERNEL(scalar_t, int8_t);
      return true;
    } else if (dst_type == at::ScalarType::Float8_e4m3fn) {
      LAUNCH_KERNEL(scalar_t, c10::Float8_e4m3fn);
      return true;
    }
    return false;
  });

#undef LAUNCH_KERNEL
}

void sgl_silu_and_mul_per_token_group_quant_fp8(
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int64_t group_size,
    double eps,
    double fp8_min,
    double fp8_max) {
  sgl_silu_and_mul_per_token_group_quant_8bit(input, output_q, output_s, group_size, eps, fp8_min, fp8_max);
}
