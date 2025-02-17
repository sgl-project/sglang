#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>

#include <cmath>

#include "utils.h"

using FP8_TYPE = c10::Float8_e4m3fn;

__device__ __forceinline__ float GroupReduce(float val, const int tid) {
  val = fmaxf(val, __shfl_xor_sync(0xffff, val, 8));
  val = fmaxf(val, __shfl_xor_sync(0xffff, val, 4));
  val = fmaxf(val, __shfl_xor_sync(0xffff, val, 2));
  val = fmaxf(val, __shfl_xor_sync(0xffff, val, 1));
  return val;
}

template <typename T>
__global__ void per_token_group_quant_fp8_kernel(const T* __restrict__ input, void* __restrict__ output_q,
                                                 float* __restrict__ output_s, const int group_size,
                                                 const int num_groups, const float eps, const float fp8_min,
                                                 const float fp8_max) {
  const int groups_per_block = 16;
  const int block_group_id = blockIdx.x * groups_per_block;
  const int tid = threadIdx.x;
  const int local_group_id = tid / 16;  // Each 16 threads handle one group
  const int local_tid = tid % 16;       // Thread ID within the group

  __shared__ float s_absmax[16];  // Only need one value per group to store the final result

  // Local maximum value for each thread
  float local_absmax = eps;

  // Ensure this block doesn't process out-of-bounds groups
  if (block_group_id + local_group_id < num_groups) {
    // Calculate input/output pointers for current group
    const T* group_input = input + (block_group_id + local_group_id) * group_size;
    FP8_TYPE* group_output = static_cast<FP8_TYPE*>(output_q) + (block_group_id + local_group_id) * group_size;
    float* scale_output = output_s + block_group_id + local_group_id;

    // Calculate local maximum absolute value
    for (int i = local_tid; i < group_size; i += 16) {
      float val = static_cast<float>(group_input[i]);
      float abs_val = fabsf(val);
      local_absmax = fmaxf(local_absmax, abs_val);
    }

    // Perform reduction using warp shuffle
    local_absmax = GroupReduce(local_absmax, local_tid);

    // Only the first thread needs to write to shared memory
    if (local_tid == 0) {
      s_absmax[local_group_id] = local_absmax;
    }
    __syncthreads();

    // Get the maximum value for this group
    const float group_absmax = s_absmax[local_group_id];
    const float y_s = group_absmax / fp8_max;

    // Only the first thread in each group writes the scale
    if (local_tid == 0) {
      *scale_output = y_s;
    }

    // Quantize the data
    for (int i = local_tid; i < group_size; i += 16) {
      float val = static_cast<float>(group_input[i]);
      float q_val = fminf(fmaxf(val / y_s, fp8_min), fp8_max);
      group_output[i] = FP8_TYPE(q_val);
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

  // Each block processes 16 groups, adjust grid size accordingly
  dim3 grid((num_groups + 15) / 16);
  dim3 block(256);  // Keep 256 threads, each 16 threads handle one group

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    per_token_group_quant_fp8_kernel<scalar_t><<<grid, block, 0, stream>>>(
        static_cast<scalar_t*>(input.data_ptr()), output_q.data_ptr(), static_cast<float*>(output_s.data_ptr()),
        group_size, num_groups, (float)eps, (float)fp8_min, (float)fp8_max);
    return true;
  });
}
