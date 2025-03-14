#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>

#include <cmath>
#include <flashinfer/vec_dtypes.cuh>

#include "utils.h"

__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
  unsigned mask = 0xffff;

  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  return val;
}

template <typename T, int GROUPS_PER_BLOCK = 16, typename DST_DTYPE>
__global__ void per_token_group_quant_8bit_kernel(
    const T* __restrict__ input,
    void* __restrict__ output_q,
    float* __restrict__ output_s,
    const int group_size,
    const int num_groups,
    const float eps,
    const float min_8bit,
    const float max_8bit) {
  const int threads_per_group = 16;
  const int local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;

  const int block_group_id = blockIdx.x * GROUPS_PER_BLOCK;
  const int block_group_offset = (block_group_id + local_group_id) * group_size;

  float local_absmax = eps;

  const T* group_input = input + block_group_offset;
  DST_DTYPE* group_output = static_cast<DST_DTYPE*>(output_q) + block_group_offset;
  float* scale_output = output_s + (block_group_id + local_group_id);

  constexpr uint32_t vec_size = 16 / sizeof(T);
  using vec_t = flashinfer::vec_t<T, vec_size>;

  const int32_t num_vec_elems = group_size / vec_size;

  for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
    vec_t input_vec;
    input_vec.cast_load(group_input + i * vec_size);

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float val = static_cast<float>(input_vec[j]);
      float abs_val = fabsf(val);
      local_absmax = fmaxf(local_absmax, abs_val);
    }
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

void sgl_per_token_group_quant_8bit(
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);

  const int num_groups = input.numel() / group_size;

  CHECK_EQ(input.numel() % group_size, 0);

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

#define LAUNCH_KERNEL(T, GPB, DST_DTYPE)                                                           \
  do {                                                                                             \
    constexpr int GROUPS_PER_BLOCK = GPB;                                                          \
    dim3 grid((num_groups + GROUPS_PER_BLOCK - 1) / GROUPS_PER_BLOCK);                             \
    dim3 block(GROUPS_PER_BLOCK* THREADS_PER_GROUP);                                               \
    per_token_group_quant_8bit_kernel<T, GROUPS_PER_BLOCK, DST_DTYPE><<<grid, block, 0, stream>>>( \
        static_cast<T*>(input.data_ptr()),                                                         \
        output_q.data_ptr(),                                                                       \
        static_cast<float*>(output_s.data_ptr()),                                                  \
        group_size,                                                                                \
        num_groups,                                                                                \
        (float)eps,                                                                                \
        (float)min_8bit,                                                                           \
        (float)max_8bit);                                                                          \
  } while (0)

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    if (dst_type == at::ScalarType::Char) {
      if (groups_per_block == 16) {
        LAUNCH_KERNEL(scalar_t, 16, int8_t);
      } else if (groups_per_block == 8) {
        LAUNCH_KERNEL(scalar_t, 8, int8_t);
      } else if (groups_per_block == 4) {
        LAUNCH_KERNEL(scalar_t, 4, int8_t);
      } else if (groups_per_block == 2) {
        LAUNCH_KERNEL(scalar_t, 2, int8_t);
      } else {
        LAUNCH_KERNEL(scalar_t, 1, int8_t);
      }
      return true;
    } else if (dst_type == at::ScalarType::Float8_e4m3fn) {
      if (groups_per_block == 16) {
        LAUNCH_KERNEL(scalar_t, 16, c10::Float8_e4m3fn);
      } else if (groups_per_block == 8) {
        LAUNCH_KERNEL(scalar_t, 8, c10::Float8_e4m3fn);
      } else if (groups_per_block == 4) {
        LAUNCH_KERNEL(scalar_t, 4, c10::Float8_e4m3fn);
      } else if (groups_per_block == 2) {
        LAUNCH_KERNEL(scalar_t, 2, c10::Float8_e4m3fn);
      } else {
        LAUNCH_KERNEL(scalar_t, 1, c10::Float8_e4m3fn);
      }
      return true;
    } else {
      std::ostringstream oss;
      oss << "Unsupported output_q scalar type: " << dst_type;
      TORCH_CHECK(false, oss.str());
    }
    return true;
  });

#undef LAUNCH_KERNEL
}
