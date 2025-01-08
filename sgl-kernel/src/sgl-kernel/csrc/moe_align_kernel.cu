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

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
                                            int32_t* expert_ids, int32_t* total_tokens_post_pad, int32_t num_experts,
                                            int32_t block_size, size_t numel, int32_t* cumsum) {
  __shared__ int32_t shared_counts[32][8];
  __shared__ int32_t local_offsets[256];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int experts_per_warp = 8;
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < num_experts) {
      shared_counts[warp_id][i] = 0;
    }
  }

  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int expert_id = topk_ids[i];
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    atomicAdd(&shared_counts[warp_idx][expert_offset], 1);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      int expert_count = 0;
      int warp_idx = (i - 1) / experts_per_warp;
      int expert_offset = (i - 1) % experts_per_warp;
      expert_count = shared_counts[warp_idx][expert_offset];

      cumsum[i] = cumsum[i - 1] + CEILDIV(expert_count, block_size) * block_size;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
    local_offsets[threadIdx.x] = cumsum[threadIdx.x];
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
    kernel<<<1, 1024, 0, stream>>>(topk_ids.data_ptr<scalar_t>(), sorted_token_ids.data_ptr<int32_t>(),
                                   experts_ids.data_ptr<int32_t>(), num_tokens_post_pad.data_ptr<int32_t>(),
                                   num_experts, block_size, topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>());
  });
}


template <typename scalar_t>
__global__ void moe_align_block_size_stage1_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ tokens_cnts,
    const int32_t num_experts,
    const size_t numel) {
    
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    
    extern __shared__ int32_t shared_counts[];
    
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        shared_counts[i] = 0;
    }
    __syncthreads();
    
    for (size_t i = tid; i < numel; i += stride) {
        int32_t expert_id = topk_ids[i];
        atomicAdd(&shared_counts[expert_id], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        if (shared_counts[i] > 0) {
            atomicAdd(&tokens_cnts[blockIdx.x * num_experts + i], shared_counts[i]);
        }
    }
}

template <typename scalar_t>
__global__ void moe_align_block_size_stage2_kernel(
    int32_t* __restrict__ tokens_cnts,
    const int32_t num_experts) {
    
    const int32_t expert_id = blockIdx.x;
    if (expert_id >= num_experts) return;
    
    if (threadIdx.x == 0) {
        int32_t last_cnt = 0;
        for (int i = 1; i <= num_experts; ++i) {
            int32_t token_cnt = tokens_cnts[i * num_experts + expert_id];
            last_cnt = last_cnt + token_cnt;
            tokens_cnts[i * num_experts + expert_id] = last_cnt;
        }
    }
}

template <typename scalar_t>
__global__ void moe_align_block_size_stage3_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, 
    int32_t* __restrict__ total_tokens_post_pad,
    const int32_t* __restrict__ tokens_cnts,
    int32_t* __restrict__ cumsum,
    const int32_t num_experts,
    const int32_t block_size,
    const size_t numel) {
    
    if(threadIdx.x == 0) {
      int32_t last_cumsum = 0;
      cumsum[0] = 0;
      
      const int32_t off_cnt = num_experts * num_experts;
      for (int i = 0; i < num_experts; ++i) {
          int32_t token_cnt = tokens_cnts[off_cnt + i];
          last_cumsum += (token_cnt + block_size - 1) / block_size * block_size;
          cumsum[i + 1] = last_cumsum;
      }
      
      *total_tokens_post_pad = last_cumsum;
    }

    // __shared__ int32_t local_offsets[256];

    // if (threadIdx.x < num_experts) {
    //   local_offsets[threadIdx.x] = cumsum[threadIdx.x];
    // }
    // __syncthreads();
    // const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
    // const size_t start_idx = threadIdx.x * tokens_per_thread;
    // for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    //   int32_t expert_id = topk_ids[i];
    //   int32_t rank_post_pad = atomicAdd(&local_offsets[expert_id], 1);
    //   sorted_token_ids[rank_post_pad] = i;
    // }
}

template <typename scalar_t>
__global__ void moe_align_block_size_stage4_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, 
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ token_cnts,
    const int32_t* __restrict__ cumsum,
    const int32_t num_experts,
    const int32_t block_size,
    const size_t numel,
    const int32_t total_blocks) {
    
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_blocks) {
        int32_t left = 0, right = num_experts;
        bool found = false;
        int32_t expert_id = -1;
        
        const int32_t block_start = idx * block_size;
        while (left < right) {
            int32_t mid = (left + right) / 2;
            if (cumsum[mid] <= block_start) {
                if (block_start < cumsum[mid + 1]) {
                    expert_id = mid;
                    found = true;
                    break;
                }
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        if (found) {
            expert_ids[idx] = expert_id;
        }
    }
}

void moe_align_block_size_stage1(
    torch::Tensor topk_ids,
    torch::Tensor token_cnts_buffer,
    int64_t num_experts) {
    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const size_t numel = topk_ids.numel();
    
    const int threads_per_block = 256;
    const int num_blocks = min((numel + threads_per_block - 1) / threads_per_block, num_experts + 1);
    const int shared_mem_size = num_experts * sizeof(int32_t);
    
    DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "moe_align_block_size_stage1", [&] {
        auto kernel = moe_align_block_size_stage1_kernel<scalar_t>;
        kernel<<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
            topk_ids.data_ptr<scalar_t>(),
            token_cnts_buffer.data_ptr<int32_t>(),
            num_experts,
            numel);
    });
}

void moe_align_block_size_stage2(
    torch::Tensor token_cnts_buffer,
    int64_t num_experts) {
    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int threads_per_block = 256;
    
    DISPATCH_INTEGRAL_TYPES(token_cnts_buffer.scalar_type(), "moe_align_block_size_stage2", [&] {
        auto kernel = moe_align_block_size_stage2_kernel<scalar_t>;
        kernel<<<num_experts, threads_per_block, 0, stream>>>(
            token_cnts_buffer.data_ptr<int32_t>(),
            num_experts);
    });
}

void moe_align_block_size_stage3(
    torch::Tensor topk_ids,
    torch::Tensor sorted_token_ids,
    torch::Tensor num_tokens_post_pad,
    torch::Tensor token_cnts_buffer,
    torch::Tensor cumsum_buffer,
    int64_t num_experts,
    int64_t block_size) {
    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const size_t numel = topk_ids.numel();
    
    DISPATCH_INTEGRAL_TYPES(token_cnts_buffer.scalar_type(), "moe_align_block_size_stage3", [&] {
        auto kernel = moe_align_block_size_stage3_kernel<scalar_t>;
        kernel<<<1, 1024, 0, stream>>>(
            topk_ids.data_ptr<scalar_t>(),
            sorted_token_ids.data_ptr<int32_t>(),
            num_tokens_post_pad.data_ptr<int32_t>(),
            token_cnts_buffer.data_ptr<int32_t>(),
            cumsum_buffer.data_ptr<int32_t>(),
            num_experts,
            block_size,
            numel);
    });
}

void moe_align_block_size_stage4(
    torch::Tensor topk_ids,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor token_cnts_buffer,
    torch::Tensor cumsum_buffer,
    int64_t num_experts,
    int64_t block_size,
    int64_t total_blocks) {
    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const size_t numel = topk_ids.numel();
    const size_t tokens_per_thread = CEILDIV(numel, num_experts);
    
    const int threads_per_block = 256;
    const int num_blocks = (max(numel, (size_t)total_blocks) + threads_per_block - 1) / threads_per_block;
    
    DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "moe_align_block_size_stage4", [&] {
        auto kernel = moe_align_block_size_stage4_kernel<scalar_t>;
        kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            topk_ids.data_ptr<scalar_t>(),
            sorted_token_ids.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(),
            token_cnts_buffer.data_ptr<int32_t>(),
            cumsum_buffer.data_ptr<int32_t>(),
            num_experts,
            block_size,
            numel,
            total_blocks);
    });
}
