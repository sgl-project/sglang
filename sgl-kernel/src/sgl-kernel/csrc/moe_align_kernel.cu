// Adapted from https://github.com/vllm-project/vllm/blob/v0.6.5/csrc/moe/moe_align_sum_kernels.cu

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <THC/THCAtomics.cuh>

#include "utils.hpp"

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#endif

#ifndef USE_ROCM
#define WARP_SIZE 32
#else
#define WARP_SIZE warpSize
#endif

#ifndef USE_ROCM
#define DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
  cudaFuncSetAttribute(FUNC, cudaFuncAttributeMaxDynamicSharedMemorySize, VAL)
#else
#define DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
  hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL)
#endif

#define CEILDIV(x, y) (((x) + (y)-1) / (y))

#define DISPATCH_CASE_INTEGRAL_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

__device__ __forceinline__ int32_t index(int32_t total_col, int32_t row, int32_t col) {
  // don't worry about overflow because num_experts is relatively small
  return row * total_col + col;
}
#define OFFSETS_PAD 1
template <typename scalar_t>
__global__ void moe_align_block_size_kernel(scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
                                            int32_t* expert_ids, int32_t* total_tokens_post_pad, int32_t num_experts,
                                            int32_t block_size, size_t numel, int32_t* cumsum) {
  __shared__ int32_t shared_counts[32][8];
  // NOTE (yiakwy) : this assumes num_experts <= 256
  __shared__ int32_t local_offsets[256+OFFSETS_PAD];
  __shared__ int32_t local_offsets_buf[16];

  const int tid = threadIdx.x;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int experts_per_warp = 8;
  const int my_expert_start = warp_id * experts_per_warp;

  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < num_experts) {
      shared_counts[warp_id][i] = 0;
    }
  }

  // NOTE (yiakwy) : this warp of threads may access other warp of threads based on the value of expert id fetched
  __syncthreads();

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int expert_id = topk_ids[i];
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    atomicAdd(&shared_counts[warp_idx][expert_offset], 1);
  }

  __syncthreads();

#define kElementsPerThr    16

  {

  int active_threads = CEILDIV(num_experts, kElementsPerThr);
  if (tid == 0) {
    local_offsets[0] = 0;
  }
  if (tid < active_threads - 1) { // NOTE(yaikwy) : algo here assumes single block execution

    // NOTE (yiakwy) : loop body, a simple reduction prototype, useful for workload with the number of experts upto 256
    // NOTE (yiakwy) : each thread process 16 expert, then only 2 steps needed

    // NOTE (yiakwy) : step 1, loop body
    // [1, 17)
    for (int i=tid*kElementsPerThr+1; i < (tid + 1)*kElementsPerThr+1; ++i) {
      int warp_idx = (i-1) / experts_per_warp;
      int expert_offset = (i-1) % experts_per_warp;

      int expert_count = shared_counts[warp_idx][expert_offset];

      int last_val = (i-1) % kElementsPerThr == 0 ? 0 : local_offsets[i-1];
      local_offsets[i] = last_val + CEILDIV(expert_count, block_size) * block_size;
    }

    local_offsets_buf[tid] = local_offsets[(tid + 1)*kElementsPerThr];

  }

  // NOTE (yiakwy) : step 1, unroll loop tail
  if (tid == active_threads - 1) {
    #pragma unroll
    for (int i=tid * kElementsPerThr+1; i < num_experts+1; ++i) {
      int warp_idx = (i-1) / experts_per_warp;
      int expert_offset = (i-1) % experts_per_warp;

      int expert_count = shared_counts[warp_idx][expert_offset];

      int last_val = (i-1) % kElementsPerThr == 0 ? 0 : local_offsets[i-1];
      local_offsets[i] = last_val + CEILDIV(expert_count, block_size) * block_size;
    }

    local_offsets_buf[tid] = local_offsets[num_experts];

  }

  __syncthreads();

  // NOTE (yiakwy) : step 2, loop body
  if (tid < active_threads - 1 && tid > 0) {
    int offset = 0;
    for (int j=0; j < tid; ++j) {
      offset += local_offsets_buf[j];
    }

    for (int i=tid*kElementsPerThr+1; i < (tid + 1)*kElementsPerThr+1; ++i) {
      local_offsets[i] += offset;
    }
  }

  // NOTE (yiakwy) : step 2, loop tail
  if (tid == active_threads - 1) {
    int offset = 0;
    for (int j=0; j < tid; ++j) {
      offset += local_offsets_buf[j];
    }
    for (int i=tid*kElementsPerThr+1; i < num_experts+1; ++i) {
      local_offsets[i] += offset;
    }
  }

  } // code block of computing cumsum

  __syncthreads();

#define kElementsPerThr    16
#define kElementsPerAccess 4

  {

  int active_threads = CEILDIV(num_experts+1, kElementsPerThr);
  if (tid < active_threads - 1) {

    // NOTE(yiakwy) : loop body useful for workload with the number of experts upto 256
    for (int i=tid * kElementsPerThr ; i < (tid + 1) * kElementsPerThr; i += kElementsPerAccess) {
      *(int4 *)(cumsum + i) = *(int4 *)(local_offsets + i);
    }
  }

  if (tid == active_threads - 1) {
    // NOTE(yiakwy) : unroll loop tail
    #pragma unroll
    for (int i=tid * kElementsPerThr; i < num_experts+1; i++) {
      *(cumsum + i) = *(local_offsets + i);
    }
  }

  if (tid == active_threads) {
    *total_tokens_post_pad = local_offsets[num_experts];
  }

  } // code block of storing to cumsum

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = local_offsets[threadIdx.x]; i < local_offsets[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  __syncthreads();

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad = atomicAdd(&local_offsets[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
                          torch::Tensor sorted_token_ids, torch::Tensor experts_ids, torch::Tensor num_tokens_post_pad,
                          torch::Tensor token_cnts_buffer, torch::Tensor cumsum_buffer) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
    auto kernel = moe_align_block_size_kernel<scalar_t>;
    // NOTE(yiakwy) : this assumes a single block execution, will be slow if too many tokens (>1024) feeded in
    kernel<<<1, 1024, 0, stream>>>(topk_ids.data_ptr<scalar_t>(), sorted_token_ids.data_ptr<int32_t>(),
                                   experts_ids.data_ptr<int32_t>(), num_tokens_post_pad.data_ptr<int32_t>(),
                                   num_experts, block_size, topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>());
  });
}
