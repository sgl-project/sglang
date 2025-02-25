#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>

#include <cmath>

#ifndef USE_ROCM
#include <flashinfer/vec_dtypes.cuh>
#else
#include "hip_vec_dtypes.h"
#endif

#include "utils.h"

#ifndef USE_ROCM
#define FULL_MASK 0xffff
#else
#define FULL_MASK 0xffffffff
#endif

#define kGROUPS_PER_BLOCK   16
#define kTHRS_PER_GROUP     16
#define kBitsToLoad         128
#define kBytesToLoad        (kBitsToLoad / 8)

#ifndef USE_ROCM
using FP8_TYPE = c10::Float8_e4m3fn;
#else
using FP8_TYPE = c10::Float8_e4m3fnuz;
#endif

__device__ __forceinline__ float GroupReduce(float val) {
  val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, 8));
  val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, 4));
  val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, 2));
  val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, 1));
  return val;
}

template <typename T>
__global__ void per_token_group_quant_fp8_kernel(const T* __restrict__ input, void* __restrict__ output_q,
                                                 float* __restrict__ output_s, const int group_size,
                                                 const int num_groups, const float eps, const float fp8_min,
                                                 const float fp8_max) {
  const int groups_per_block = kGROUPS_PER_BLOCK;
  const int local_group_id = threadIdx.x / kTHRS_PER_GROUP;
  const int lane_id = threadIdx.x % kTHRS_PER_GROUP;

  const int block_group_id = blockIdx.x * groups_per_block;
  const int block_group_offset = (block_group_id + local_group_id) * group_size;

  __shared__ float s_absmax[kGROUPS_PER_BLOCK];

  float local_absmax = eps;

  const T* group_input = input + block_group_offset;
  FP8_TYPE* group_output = static_cast<FP8_TYPE*>(output_q) + block_group_offset;
  float* scale_output = output_s + block_group_id + local_group_id;

  constexpr uint32_t vec_size = kBytesToLoad / sizeof(T);
  using vec_t = flashinfer::vec_t<T, vec_size>;

  const int32_t num_vec_elems = group_size / vec_size;

  for (int32_t i = lane_id; i < num_vec_elems; i += kTHRS_PER_GROUP) {
    vec_t input_vec;
    input_vec.cast_load(group_input + i * vec_size);

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float val = castToFloat(input_vec[j]);
      float abs_val = fabsf(val);
      local_absmax = fmaxf(local_absmax, abs_val);
    }
  }

  local_absmax = GroupReduce(local_absmax);

  if (lane_id == 0) {
    s_absmax[local_group_id] = local_absmax;
  }
  __syncthreads();

  const float group_absmax = s_absmax[local_group_id];
  const float y_s = group_absmax / fp8_max;

  if (lane_id == 0) {
    *scale_output = y_s;
  }

  for (int32_t i = lane_id; i < num_vec_elems; i += kTHRS_PER_GROUP) {
    vec_t input_vec;
    input_vec.cast_load(group_input + i * vec_size);

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float val = castToFloat(input_vec[j]);
      float q_val = fminf(fmaxf(val / y_s, fp8_min), fp8_max);
      group_output[i * vec_size + j] = FP8_TYPE(q_val);
    }
  }
}

void sgl_per_token_group_quant_fp8(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s,
                                   int64_t group_size, double eps, double fp8_min, double fp8_max) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);

  const int num_groups = input.numel() / group_size;

  CHECK_EQ(input.numel() % group_size, 0);

  dim3 grid(CEILDIV(num_groups, kGROUPS_PER_BLOCK));
  dim3 block(kGROUPS_PER_BLOCK * kTHRS_PER_GROUP);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    per_token_group_quant_fp8_kernel<scalar_t><<<grid, block, 0, stream>>>(
        static_cast<scalar_t*>(input.data_ptr()), output_q.data_ptr(), static_cast<float*>(output_s.data_ptr()),
        group_size, num_groups, (float)eps, (float)fp8_min, (float)fp8_max);
    return true;
  });
}
